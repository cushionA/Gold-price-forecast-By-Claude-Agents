# Submodel Design Document: real_rate (Attempt 1)

**Feature**: real_rate (DFII10 - 10Y TIPS Yield)
**Attempt**: 1
**Phase**: Smoke Test
**Architect**: Claude Opus 4.6
**Date**: 2026-02-14

---

## 0. Fact-Check Results

### FRED Series IDs
- FRED:DFII10 (10Y TIPS) -- Confirmed. 6,031 obs (2003-01-02 to 2026-02-12), 5,782 non-NaN. NaN values on holidays (e.g., New Year).
- FRED:DFII5 (5Y TIPS) -- Confirmed.
- FRED:DFII7 (7Y TIPS) -- Confirmed.
- FRED:DFII20 (20Y TIPS) -- Confirmed.
- FRED:DFII30 (30Y TIPS) -- Confirmed.
- Multi-country nominal yields (IRLTLT01DEM156N, IRLTLT01JPM156N, IRLTLT01GBM156N, IRLTLT01CAM156N) -- Confirmed but MONTHLY frequency, not daily. Not suitable for daily submodel without interpolation.

### Package Claims
- `hmmlearn` -- Not installed locally. Researcher correctly deprioritized for smoke test.
- `hurst` package -- Not installed. Researcher correctly suggested autocorrelation as simpler alternative.
- `pip install setar` -- DOES NOT EXIST on PyPI. Researcher hallucinated this package. SETAR module also does NOT exist in statsmodels (checked v0.14.6). MarkovRegression in statsmodels is real.
- `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression` -- Confirmed exists.

### Methodology Assessment
- Threshold-based regime detection in PyTorch -- Reasonable for smoke test. Accepted.
- Rolling autocorrelation for persistence -- Reasonable and simple. Accepted.
- Velocity/acceleration features -- Reasonable. 20d and 60d windows are appropriate for real rates. Accepted.
- Hurst exponent -- Researcher correctly flagged unreliability on financial data. Excluded for smoke test.
- Researcher claim about ~6,000 DFII10 observations -- Confirmed (6,031 total, 5,782 non-NaN).
- Researcher claim about multi-country TIPS unavailability on FRED -- Confirmed. Only U.S. TIPS are available as daily series.

### Data Range Alignment
- schema_freeze.json: 2015-01-30 to 2025-02-12, 2,523 rows
- DFII10 within this range: 2,511 non-NaN observations (close match after forward-fill alignment to gold trading days)
- Submodel output MUST align to the same date index as base_features.csv

### Summary
- 5 of 5 FRED series confirmed
- 1 researcher claim incorrect (setar package)
- No impact on design since SETAR was not recommended for smoke test
- All methodology choices for smoke test are sound

---

## 1. Overview

### Purpose
Extract latent dynamics of the US 10Y real interest rate (DFII10) that the baseline XGBoost model cannot capture from the raw level alone. The submodel produces continuous features encoding:
1. **Regime state** -- whether the current real rate environment is low, medium, or high (soft probabilities)
2. **Persistence** -- whether recent rate changes indicate trending or mean-reverting behavior
3. **Velocity** -- normalized rate of change across time horizons
4. **Acceleration** -- second derivative capturing inflection points

### Method and Rationale
**Sliding-window MLP Autoencoder** on hand-crafted input features.

Why this approach:
- **Smoke test priority**: MLP is simpler to implement and debug than GRU/LSTM
- **Feature engineering first**: Pre-compute regime, persistence, velocity, and acceleration features from DFII10 using rolling windows. These are interpretable and verifiable.
- **Autoencoder for compression**: Train an autoencoder to reconstruct the hand-crafted features from a compressed latent space. The latent representation captures the joint dynamics of regime/persistence/velocity in a lower-dimensional, decorrelated form.
- **No external dependencies**: Pure PyTorch + pandas/numpy. No hmmlearn, hurst, or statsmodels required.
- **Fully differentiable**: All operations in the autoencoder are standard PyTorch.

### Expected Effect
The latent features should capture temporal patterns in real rates that are invisible to the point-in-time XGBoost baseline:
- Distinguishing between persistent trends and transitory movements
- Identifying regime transitions (e.g., shift from negative to positive real rates)
- Encoding multi-horizon momentum (20d vs 60d velocity divergence)

---

## 2. Data Specification

### Main Data
| Data | Source | Series ID | Frequency | Range |
|------|--------|-----------|-----------|-------|
| 10Y TIPS Yield | FRED | DFII10 | Daily | 2003-01-02 to present |

### Preprocessing Pipeline

```
1. Fetch DFII10 from FRED (full history from 2003-01-01)
2. Drop NaN (holidays), forward-fill to gold trading days
3. Compute hand-crafted features (see Section 3 input specification)
4. Align to schema_freeze date range: 2015-01-30 to 2025-02-12
   (requires lookback buffer -- fetch from 2014-06-01 to have 120+ trading days before 2015-01-30)
5. Standardize each feature column to zero mean, unit variance
   (fit statistics from train split ONLY, apply to val/test)
6. Create sliding windows of length W for autoencoder input
7. Split: train 70% / val 15% / test 15% (chronological order)
```

### Expected Sample Count
- Total rows after alignment: ~2,523 (matching schema_freeze)
- After windowing (window size W=60): ~2,463 usable samples
- Train: ~1,724 / Val: ~369 / Test: ~370

### No Multi-Country Data
Per fact-check: FRED multi-country nominal yields are monthly only. Synthetic daily real rates would require interpolation and add noise. US DFII10 alone provides sufficient samples for a smoke test MLP.

---

## 3. Model Architecture (PyTorch)

### Input Feature Engineering (pre-model)

From raw DFII10 daily series, compute 8 features at each time step t:

| # | Feature Name | Formula | Interpretation |
|---|-------------|---------|----------------|
| 1 | level | DFII10[t] | Current real rate level |
| 2 | change_1d | DFII10[t] - DFII10[t-1] | Daily change |
| 3 | velocity_20d | (DFII10[t] - DFII10[t-20]) / rolling_std_60d[t] | Normalized 20-day momentum |
| 4 | velocity_60d | (DFII10[t] - DFII10[t-60]) / rolling_std_60d[t] | Normalized 60-day momentum |
| 5 | accel_20d | velocity_20d[t] - velocity_20d[t-20] | Acceleration (change in momentum) |
| 6 | rolling_std_20d | std(DFII10 changes over 20d window) | Short-term volatility |
| 7 | regime_percentile | Rolling 252-day percentile rank of DFII10[t] | Where current level sits in 1-year distribution |
| 8 | autocorr_20d | Rolling 60-day autocorrelation at lag 1 of daily changes | Persistence measure |

These 8 features are computed BEFORE the model. The model sees a sliding window of W consecutive time steps of these 8 features.

### Model: Sliding-Window MLP Autoencoder

```
Input: [batch, W * 8]  (flattened sliding window of W timesteps x 8 features)
                         W is a hyperparameter (default 60)

Encoder:
  Linear(W*8, hidden_dim) -> BatchNorm -> ReLU -> Dropout
  Linear(hidden_dim, hidden_dim//2) -> BatchNorm -> ReLU -> Dropout
  Linear(hidden_dim//2, latent_dim)  -> Tanh

Latent: [batch, latent_dim]    (latent_dim = 4, the submodel output)

Decoder:
  Linear(latent_dim, hidden_dim//2) -> BatchNorm -> ReLU -> Dropout
  Linear(hidden_dim//2, hidden_dim) -> BatchNorm -> ReLU -> Dropout
  Linear(hidden_dim, W*8)

Output (reconstruction): [batch, W*8]
```

### PyTorch Pseudocode

```python
class RealRateAutoencoder(nn.Module):
    def __init__(self, window_size=60, n_features=8, hidden_dim=64,
                 latent_dim=4, dropout=0.2):
        super().__init__()
        input_dim = window_size * n_features

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        self.window_size = window_size
        self.n_features = n_features
        self.latent_dim = latent_dim

    def encode(self, x):
        """x: [batch, window_size * n_features] -> [batch, latent_dim]"""
        return self.encoder(x)

    def decode(self, z):
        """z: [batch, latent_dim] -> [batch, window_size * n_features]"""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass for training"""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
```

### Input Specification
- **Dimensions**: W * 8 (e.g., 60 * 8 = 480 for default window)
- **Meaning**: Each of the 8 engineered features across W consecutive trading days, flattened
- **Normalization**: Z-score standardized (mean=0, std=1) per feature column, statistics from train set only

### Output Specification
- **Dimensions**: 4 continuous values per trading day
- **Range**: [-1, 1] (Tanh activation on latent layer)
- **Column names in output CSV**:
  - `real_rate_latent_0`: Primary latent dimension (expected to correlate with regime state)
  - `real_rate_latent_1`: Secondary dimension (expected to correlate with persistence/trend)
  - `real_rate_latent_2`: Tertiary dimension (expected to correlate with velocity/momentum)
  - `real_rate_latent_3`: Quaternary dimension (expected to correlate with volatility state)
- **Interpretation**: Latent dimensions are learned, not pre-assigned. The evaluator will analyze what each dimension captures via correlation analysis with the input features.

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_features | 8 | Number of hand-crafted input features (fixed by design) |
| latent_dim | 4 | 4 output features for meta-model (within 3-5 range per requirements) |
| batch_size | 64 | Standard for ~1,700 training samples |
| max_epochs | 100 | With early stopping, will terminate earlier |
| early_stop_patience | 10 | Epochs without val improvement before stopping |
| optimizer | Adam | Standard for autoencoders |
| scheduler | ReduceLROnPlateau | factor=0.5, patience=5 |
| regime_percentile_window | 252 | ~1 trading year for percentile rank |
| autocorr_window | 60 | 60 trading days for rolling autocorrelation |
| velocity_norm_window | 60 | 60 trading days for rolling std normalization |

### Optuna Search Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| window_size | {20, 40, 60} | Categorical | Lookback depth: 1-3 months of trading days. Shorter = more samples, less context. |
| hidden_dim | {32, 64, 128} | Categorical | MLP capacity. Small data needs small model. |
| dropout | [0.1, 0.4] | Uniform | Regularization. Higher = more generalization. |
| learning_rate | [1e-4, 1e-2] | Log-uniform | Standard AE learning rate range. |
| weight_decay | [1e-6, 1e-3] | Log-uniform | L2 regularization to prevent overfitting. |

### Search Settings
- n_trials: 5 (smoke test constraint)
- timeout: 600 seconds (10 minutes max)
- pruner: MedianPruner (prune underperforming trials early)
- direction: minimize (val reconstruction loss)

---

## 5. Training Configuration

### Loss Function
**MSE (Mean Squared Error)** on reconstruction:

```python
loss = F.mse_loss(reconstruction, input_window)
```

Rationale: Standard for autoencoders. The goal is to learn a compressed representation that can faithfully reconstruct the input dynamics. No supervised signal needed -- the latent space quality is judged by the meta-model (Gate 2/3), not by reconstruction loss alone.

### Optimizer
- Adam with weight_decay (searched by Optuna)
- Learning rate scheduler: ReduceLROnPlateau(patience=5, factor=0.5)

### Early Stopping
- Monitor: validation MSE loss
- Patience: 10 epochs
- Restore best weights on stop

### Data Loading
- Sliding window: at each time step t (where t >= W), create a sample by taking features[t-W:t] and flattening to (W * 8,)
- No overlap control needed: consecutive windows share W-1 time steps, which is fine for autoencoders (not predicting future)
- Shuffle training batches: YES (windows are independent samples for AE training; time ordering within each window is preserved)

### Validation Strategy
- Time-series split: train ends before val starts, val ends before test starts
- Evaluate reconstruction MSE on validation set at each epoch
- Best model checkpoint saved based on val loss

---

## 6. Kaggle Execution Settings

- **enable_gpu**: false
- **Estimated execution time**: 5-10 minutes
- **Estimated memory usage**: <2 GB
- **Rationale**: Small data (~2,500 rows), MLP architecture (no RNN/Transformer), 5 Optuna trials with max 100 epochs each (early stopping at ~20-30). CPU is sufficient.
- **Required additional pip packages**: [] (none beyond PyTorch, pandas, numpy, optuna, fredapi which are standard)
- **enable_internet**: true (needed for FRED API data fetch)
- **Kaggle Secrets required**: FRED_API_KEY

---

## 7. Implementation Instructions

### For builder_data

#### Data to Fetch
1. FRED:DFII10, from 2014-06-01 to present (buffer for 120-day lookback before schema start date of 2015-01-30)

#### Feature Engineering Pipeline
1. Load DFII10 raw series
2. Drop NaN rows (holidays)
3. Forward-fill to align with gold trading days (use data/raw/gold.csv date index as reference)
4. Compute 8 features:
   - `level`: raw DFII10 value
   - `change_1d`: diff(DFII10, 1)
   - `velocity_20d`: (DFII10[t] - DFII10[t-20]) / rolling_std(change_1d, 60)
   - `velocity_60d`: (DFII10[t] - DFII10[t-60]) / rolling_std(change_1d, 60)
   - `accel_20d`: velocity_20d[t] - velocity_20d[t-20]
   - `rolling_std_20d`: rolling_std(change_1d, 20)
   - `regime_percentile`: rolling percentile rank over 252-day window (pd.Series.rolling(252).apply(lambda x: percentileofscore(x, x.iloc[-1]) / 100))
   - `autocorr_20d`: rolling 60-day autocorrelation of change_1d at lag 1 (pd.Series.rolling(60).apply(lambda x: x.autocorr(lag=1)))
5. Drop rows with NaN from rolling calculations
6. Trim to schema date range: 2015-01-30 to 2025-02-12
7. Save to data/processed/real_rate_features.csv

#### Output Format
CSV with columns: date (index), level, change_1d, velocity_20d, velocity_60d, accel_20d, rolling_std_20d, regime_percentile, autocorr_20d

### For builder_model

#### Self-Contained train.py Structure
The training script must embed ALL data fetching and feature engineering inline (no imports from src/). Structure:

```
1. pip install dependencies (fredapi, optuna if not available)
2. Fetch DFII10 from FRED using Kaggle Secret FRED_API_KEY
3. Compute 8 features (same pipeline as builder_data)
4. Align to gold trading days (fetch GC=F from yfinance for date index)
5. Standardize features (fit on train split)
6. Create sliding windows
7. Train/val/test split (70/15/15 chronological)
8. Define RealRateAutoencoder model class
9. Optuna HPO (5 trials, 600s timeout)
10. Retrain best model on train set
11. Generate latent features for ALL dates using best model
12. Save outputs:
    - submodel_output.csv: date, real_rate_latent_0..3
    - model.pt: model state dict
    - training_result.json: metrics, params, output shape
```

#### PyTorch Class Design
- Class `RealRateAutoencoder` as specified in Section 3
- Inherit from nn.Module (NOT from SubModelBase, since train.py is self-contained)
- Include `transform` method for inference:
  ```python
  def transform(self, data_tensor):
      """Generate latent features from input windows"""
      self.eval()
      with torch.no_grad():
          _, z = self.forward(data_tensor)
      return z
  ```

#### train.py Specific Notes
1. **FRED API key**: Access via `os.environ['FRED_API_KEY']` (Kaggle Secret). Fail immediately with KeyError if missing.
2. **Gold date index**: Fetch GC=F from yfinance to get the trading day calendar. Only need dates, not prices.
3. **Standardization**: Compute mean/std from train split only. Save these statistics in training_result.json for future inference.
4. **NaN handling**: After feature engineering, any remaining NaN rows must be dropped BEFORE windowing. Log the number of dropped rows.
5. **Reconstruction quality check**: After training, compute and log reconstruction MSE on each split. If val MSE > 2x train MSE, flag potential overfitting in the result JSON.
6. **Output alignment**: The submodel_output.csv must have one row per trading day in the schema date range. For the first W-1 days (insufficient lookback), output NaN. The meta-model builder will handle NaN imputation.

---

## 8. Risks and Alternatives

### Risk 1: Autoencoder Learns Trivial Representation
**Description**: The latent space may collapse to a near-constant or capture only the level, making it redundant with base_features.csv.
**Mitigation**: Monitor reconstruction loss per feature. If velocity/acceleration features are poorly reconstructed, the latent space is ignoring dynamics.
**Fallback**: If Gate 2 fails (MI < 5%), attempt 2 should try a supervised approach: train a small MLP to predict next-5-day volatility of DFII10, and use intermediate layer activations as features.

### Risk 2: Overfitting on Small Dataset
**Description**: ~1,700 training samples with 480-dimensional input is a high dimension-to-sample ratio.
**Mitigation**: Aggressive dropout (0.1-0.4), weight decay, early stopping, and small hidden dimensions (32-128). The autoencoder bottleneck (4 dims) itself is strong regularization.
**Fallback**: If overfit ratio > 1.5 at Gate 1, reduce window_size to 20 and hidden_dim to 32.

### Risk 3: Latent Features Have High VIF with Yield Curve
**Description**: DFII10 level is mathematically related to DGS10 (nominal yield). The level feature in the autoencoder input could cause the latent space to correlate with yield_curve submodel outputs.
**Mitigation**: The autoencoder primarily captures dynamics (velocity, acceleration, regime changes) rather than level. The `regime_percentile` feature is rank-based, making it invariant to absolute level. Gate 2 VIF check will catch any issues.
**Fallback**: If VIF > 10, remove `level` from the 8 input features and retrain with 7 features only.

### Risk 4: NaN Propagation in Feature Engineering
**Description**: Rolling window computations (252-day percentile, 60-day autocorrelation) require substantial lookback, producing NaN at the start.
**Mitigation**: Fetch DFII10 from 2014-06-01 (>120 trading days before schema start of 2015-01-30). The 252-day percentile window still needs data from ~2014-01-01. Fetch from 2013-06-01 to be safe.
**Correction**: Revise data fetch start to 2013-06-01 to ensure 252 trading days of history before 2015-01-30.

### Alternative Architectures (for future attempts)
1. **GRU Autoencoder**: If MLP fails to capture temporal ordering within windows, a GRU encoder can model sequential dependencies. Adds ~2x training time.
2. **Variational Autoencoder (VAE)**: Adds KL divergence regularization for smoother latent space. Better for regime detection. Adds ~10% overhead.
3. **Semi-supervised MLP**: Add a small auxiliary loss predicting next-5-day DFII10 volatility. Guides latent space toward dynamics-relevant features. Requires careful leak prevention.

---

## Appendix A: Date Alignment Details

```
Schema freeze date range: 2015-01-30 to 2025-02-12 (2,523 rows)
DFII10 fetch range:       2013-06-01 to 2025-02-14 (buffer for lookback)
Lookback requirements:
  - 252 days for regime_percentile: needs data from ~2014-01
  - 60 days for autocorr_20d / velocity normalization: needs data from ~2014-10
  - 60 days for window_size: needs data from ~2014-10
  - Combined: fetch from 2013-06-01 gives ample buffer
Output rows: 2,523 (matching schema), minus first W-1 = NaN
```

## Appendix B: Output CSV Format

```csv
date,real_rate_latent_0,real_rate_latent_1,real_rate_latent_2,real_rate_latent_3
2015-01-30,NaN,NaN,NaN,NaN
...
2015-04-27,0.342,-0.156,0.789,-0.234
2015-04-28,0.338,-0.161,0.791,-0.229
...
2025-02-12,0.125,0.445,-0.312,0.067
```

First ~59 rows (for window_size=60) will be NaN due to insufficient lookback within the schema range. The meta-model builder must handle these (forward-fill from first non-NaN or drop).
