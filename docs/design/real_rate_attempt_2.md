# Submodel Design Document: real_rate (Attempt 2)

**Feature**: real_rate (DFII10 - 10Y TIPS Yield)
**Attempt**: 2
**Phase**: Smoke Test (continued)
**Architect**: Claude Opus 4.6
**Date**: 2026-02-14

---

## 0. Fact-Check Results

### FRED Series IDs
- FRED:DFII10 (10Y TIPS) -- Confirmed. Data through 2026-02-12. Last value: 1.80%. 278 non-NaN values in 2025 alone.

### PyTorch GRU API
- `torch.nn.GRU` -- Confirmed compatible (PyTorch 2.7.1).
- Output shape: `[batch, seq_len, hidden_size * num_directions]`
- Hidden shape: `[num_layers * num_directions, batch, hidden_size]`
- **CRITICAL**: `dropout` parameter only applies between GRU layers. For `num_layers=1`, it is **ignored** with a warning. Must use `nn.Dropout` separately for output regularization.
- Bidirectional doubles output dimension but not hidden_size.

### Memory Estimate
- Worst-case config (hidden=64, bidirectional, layers=2): 143K params, ~5 MB training memory.
- Well within Kaggle CPU limit (16 GB RAM).

### Data Dimensions
- Schema freeze: 2015-01-30 to 2025-02-12, 2,523 rows
- After windowing (window=60): ~2,463 samples
- Train (70%): ~1,724 / Val (15%): ~369 / Test (15%): ~370
- Concern: 1,724 training samples is small for neural networks. Architecture must be compact.

### Methodology Assessment
- GRU Autoencoder for temporal dynamics -- Appropriate. GRU's hidden state naturally compresses temporal context without flattening. Reduces effective parameter count compared to MLP on flattened windows.
- Contrastive loss for regime separation -- Viable but adds complexity. Deferred: start with MSE + regularization. Add only if overfit ratio is borderline.
- First-difference postprocessing -- Sound. Removes level autocorrelation, focuses on regime transitions. Low-risk addition.
- Bidirectional GRU -- Acceptable for offline analysis (not forecasting). The encoder processes historical windows, so future-within-window context is legitimate.

### Summary
- FRED data: Confirmed accessible
- PyTorch GRU: Confirmed, one critical gotcha documented (single-layer dropout)
- Memory: No GPU needed
- All methodology choices are sound

---

## 1. Overview

### Purpose
Extract latent dynamics of the US 10Y real interest rate (DFII10) that the baseline XGBoost model cannot capture from the raw level alone. Based on Attempt 1 findings, the submodel now produces **2 compressed dimensions** (down from 4) encoding:
1. **Regime-persistence state**: A low-dimensional summary of whether the current real rate environment is trending, mean-reverting, or transitioning
2. **Volatility-momentum state**: Normalized rate-of-change dynamics relative to recent volatility

### Key Changes from Attempt 1

| Aspect | Attempt 1 | Attempt 2 | Rationale |
|--------|-----------|-----------|-----------|
| Architecture | MLP Autoencoder | **GRU Autoencoder** | Temporal dependencies captured natively |
| Latent dim | 4 | **2** | Force tighter compression; Attempt 1 had 1 dead dimension |
| Max hidden dim | 128 | **64** | Reduce capacity for small sample size |
| Dropout | 0.13 (found by Optuna) | **0.3-0.5 (search range)** | Prevent overfitting (overfit ratio was 2.69) |
| Weight decay | 1e-6 (found by Optuna) | **1e-4 to 1e-2 (search range)** | 100-10000x stronger L2 |
| Window size | 20 (found by Optuna) | **40-80 (search range)** | Capture longer regime transitions |
| Optuna trials | 5 | **20** | Better HP exploration |
| Output postprocess | None | **First-difference** | Break autocorrelation > 0.99 |

### Method and Rationale
**GRU Autoencoder** with sequence-to-sequence reconstruction.

Why GRU over MLP:
- **Native temporal modeling**: GRU processes the window sequentially, learning which past states matter via gating mechanism. MLP treats the flattened window as a bag of numbers, losing temporal structure.
- **Parameter efficiency**: A GRU with hidden_dim=32 on 8 features has ~3,840 parameters per layer. An MLP on a flattened 60x8=480 input has 480*32=15,360 parameters in the first layer alone. GRU is 4x more parameter-efficient.
- **Natural resistance to identity mapping**: GRU hidden state is a nonlinear transformation through forget/update gates. It cannot trivially pass input through unchanged, unlike an MLP with sufficient width.
- **Regime detection**: GRU's forget gate naturally learns to "hold" a hidden state during stable regimes and "reset" during transitions -- precisely the dynamics we want to capture.

### Expected Effect
- Overfit ratio: 1.1-1.4 (down from 2.69)
- Autocorrelation of latent outputs: <0.9 (down from >0.995, after first-differencing)
- Gate 2 MI increase: ~15-20% (maintained or improved from 18.5%)
- Gate 3 ablation: DA +0.3-0.5%, Sharpe improvement possible with cleaner signals

---

## 2. Data Specification

### Main Data
| Data | Source | Series ID | Frequency | Range |
|------|--------|-----------|-----------|-------|
| 10Y TIPS Yield | FRED | DFII10 | Daily | 2003-01-02 to present |

### Preprocessing Pipeline

```
1. Fetch DFII10 from FRED (from 2013-06-01 for 252-day lookback buffer)
2. Drop NaN (holidays), forward-fill to gold trading days
3. Compute hand-crafted features (same 8 features as Attempt 1)
4. Align to schema_freeze date range: 2015-01-30 to 2025-02-12
5. Standardize each feature column to zero mean, unit variance
   (fit statistics from train split ONLY, apply to val/test)
6. Create sequential windows of length W for GRU input
   (NOT flattened -- keep [W, 8] shape for GRU processing)
7. Split: train 70% / val 15% / test 15% (chronological order)
```

### Expected Sample Count
- Total rows after alignment: ~2,523 (matching schema_freeze)
- After windowing (window size W=60): ~2,463 usable samples
- Train: ~1,724 / Val: ~369 / Test: ~370

### No Multi-Country Data
Same as Attempt 1: FRED multi-country nominal yields are monthly only. US DFII10 alone provides sufficient samples for this attempt. Multi-country approach reserved for Attempt 3+ if needed.

---

## 3. Model Architecture (PyTorch)

### Input Feature Engineering (pre-model)

Same 8 features as Attempt 1 (verified working, not the source of failure):

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

### Model: GRU Autoencoder (Sequence-to-Sequence)

```
Input: [batch, seq_len, 8]  (sequential window, NOT flattened)

ENCODER:
  GRU(input_size=8, hidden_size=gru_hidden_dim, num_layers=gru_num_layers,
      batch_first=True, dropout=gru_dropout if num_layers>1 else 0,
      bidirectional=bidirectional)
  → Take final hidden state: [num_layers*num_dir, batch, gru_hidden_dim]
  → Reshape last layer: [batch, gru_hidden_dim * num_dir]
  → Dropout(dropout_rate)
  → Linear(gru_hidden_dim * num_dir, latent_dim)
  → Tanh()

LATENT: [batch, latent_dim]  (latent_dim = 2, the submodel output)

DECODER:
  Linear(latent_dim, gru_hidden_dim) → ReLU → Dropout(dropout_rate)
  → Repeat latent across seq_len: [batch, seq_len, gru_hidden_dim]
  → GRU(input_size=gru_hidden_dim, hidden_size=gru_hidden_dim,
        num_layers=1, batch_first=True)
  → Linear(gru_hidden_dim, 8)

Output (reconstruction): [batch, seq_len, 8]
```

### PyTorch Pseudocode

```python
class RealRateGRUAutoencoder(nn.Module):
    """
    GRU-based autoencoder for real rate temporal dynamics.

    Changes from Attempt 1 MLP:
    - Sequential processing (no flattening)
    - Tighter bottleneck (latent_dim=2 vs 4)
    - Separate dropout layers (GRU dropout only between layers)
    - Decoder GRU reconstructs sequence from compressed state
    """
    def __init__(self, n_features=8, gru_hidden_dim=32, gru_num_layers=1,
                 latent_dim=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.n_features = n_features
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # --- Encoder ---
        # GRU dropout only applies between layers; for single layer, set to 0
        gru_dropout = dropout if gru_num_layers > 1 else 0.0
        self.encoder_gru = nn.GRU(
            input_size=n_features,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional
        )
        # Post-GRU dropout (applies to final hidden state)
        self.encoder_dropout = nn.Dropout(dropout)
        # Compress to latent space
        encoder_output_dim = gru_hidden_dim * self.num_directions
        self.encoder_fc = nn.Linear(encoder_output_dim, latent_dim)

        # --- Decoder ---
        self.decoder_fc = nn.Linear(latent_dim, gru_hidden_dim)
        self.decoder_dropout = nn.Dropout(dropout)
        self.decoder_gru = nn.GRU(
            input_size=gru_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder_output = nn.Linear(gru_hidden_dim, n_features)

    def encode(self, x):
        """
        x: [batch, seq_len, n_features] -> z: [batch, latent_dim]
        """
        # GRU encoding
        _, hidden = self.encoder_gru(x)
        # hidden: [num_layers*num_dir, batch, gru_hidden_dim]

        # Take the last layer's hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = hidden[-2]   # [batch, gru_hidden_dim]
            h_backward = hidden[-1]  # [batch, gru_hidden_dim]
            h = torch.cat([h_forward, h_backward], dim=-1)
        else:
            h = hidden[-1]  # [batch, gru_hidden_dim]

        # Dropout + compress to latent
        h = self.encoder_dropout(h)
        z = torch.tanh(self.encoder_fc(h))
        return z

    def decode(self, z, seq_len):
        """
        z: [batch, latent_dim] -> reconstruction: [batch, seq_len, n_features]
        """
        # Expand latent to GRU input dimension
        h = torch.relu(self.decoder_fc(z))
        h = self.decoder_dropout(h)

        # Repeat across time steps
        decoder_input = h.unsqueeze(1).repeat(1, seq_len, 1)

        # Decode sequence
        decoder_output, _ = self.decoder_gru(decoder_input)
        reconstruction = self.decoder_output(decoder_output)
        return reconstruction

    def forward(self, x):
        """
        x: [batch, seq_len, n_features]
        Returns: reconstruction [batch, seq_len, n_features], z [batch, latent_dim]
        """
        seq_len = x.size(1)
        z = self.encode(x)
        reconstruction = self.decode(z, seq_len)
        return reconstruction, z

    def transform(self, x):
        """Generate latent features for inference (no gradient)"""
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
        return z
```

### Why This Architecture Addresses Attempt 1 Failures

1. **Overfit (2.69 -> target <1.5)**: GRU with hidden_dim=32 has ~3.8K params/layer vs MLP's ~15K for first layer. Combined with dropout 0.3-0.5 applied to hidden state and weight_decay 1e-4 to 1e-2, the model has far less capacity to memorize.

2. **Identity mapping (autocorr >0.995)**: GRU hidden state passes through sigmoid/tanh gates, making direct pass-through impossible. Additionally, latent_dim=2 (down from 4) forces the model to discard redundant information. The 8 input features cannot be faithfully compressed into 2 dimensions without lossy abstraction.

3. **Dead dimension (latent_2 MI=0)**: With only 2 latent dimensions, every dimension must carry information. The encoder FC layer maps from gru_hidden_dim (or 2*gru_hidden_dim for bidirectional) to just 2 outputs, creating extreme compression pressure.

4. **Noisy outputs (MAE +0.165 in ablation)**: First-difference postprocessing (Section 3.1) removes the slow-moving level component that dominated Attempt 1 outputs, leaving only the informative transition signals.

### 3.1 Output Postprocessing

After generating raw latent features `z[t]`, apply first-difference:

```python
# Raw latent: z[t] for each trading day
# First-difference: captures regime TRANSITIONS rather than regime LEVELS
z_diff[t] = z[t] - z[t-1]

# Column names in output CSV:
# real_rate_latent_0: first-differenced primary latent (regime transition signal)
# real_rate_latent_1: first-differenced secondary latent (volatility shift signal)
```

Rationale:
- Attempt 1 latent outputs had autocorr > 0.995, meaning they tracked slow-moving levels
- The meta-model already has the raw DFII10 level in base_features
- What the meta-model needs is regime CHANGE information, not regime level
- First-differencing removes the redundant level component and isolates transitions
- This also naturally reduces autocorrelation (difference of a random walk is white noise)

**Fallback**: If first-differencing removes too much signal (Gate 2 MI drops below 5%), fall back to rolling z-score: `(z[t] - mean(z[t-20:t])) / std(z[t-20:t])`

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_features | 8 | Same 8 hand-crafted input features as Attempt 1 (verified working) |
| latent_dim | 2 | Reduced from 4. Forces meaningful compression. Eliminates dead dimensions. |
| max_epochs | 150 | Increased from 100. GRU may converge slower due to sequential processing. |
| early_stop_patience | 15 | Increased from 10. Give GRU more time to find good hidden state dynamics. |
| optimizer | Adam | Standard for RNNs |
| scheduler | ReduceLROnPlateau | factor=0.5, patience=7. More patient than Attempt 1 (was 5). |
| regime_percentile_window | 252 | ~1 trading year (unchanged) |
| autocorr_window | 60 | 60 trading days (unchanged) |
| velocity_norm_window | 60 | 60 trading days (unchanged) |
| postprocessing | first_difference | Break autocorrelation; output regime transitions not levels |

### Optuna Search Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| window_size | {40, 60, 80} | Categorical | Longer than Attempt 1 (was {20,40,60}). 40 = 2 months, 80 = 4 months. Captures regime transition timescales. |
| gru_hidden_dim | {16, 32, 64} | Categorical | Small enough to prevent overfitting. 64 is the max (was 128 in Attempt 1 MLP). |
| gru_num_layers | {1, 2} | Categorical | 1 = simpler, less overfitting risk. 2 = hierarchical temporal features. |
| dropout | [0.3, 0.5] | Uniform | Floor raised from 0.1 to 0.3. Attempt 1 found 0.13 which was insufficient. |
| learning_rate | [1e-5, 1e-3] | Log-uniform | Lower range than Attempt 1 (was 1e-4 to 1e-2). GRU benefits from smaller LR. |
| weight_decay | [1e-4, 1e-2] | Log-uniform | 100-10000x stronger than Attempt 1 (was 1e-6 to 1e-3). Critical for overfitting. |
| bidirectional | {True, False} | Categorical | Bidirectional can capture both forward and backward context within historical window. |
| batch_size | {16, 32} | Categorical | Smaller batches than Attempt 1 (was 64). Better generalization with more gradient updates. |

### Search Settings
- n_trials: 20
- timeout: 1800 seconds (30 minutes)
- pruner: MedianPruner(n_startup_trials=5, n_warmup_steps=15)
- direction: minimize (validation reconstruction MSE)
- sampler: TPESampler(seed=42) for reproducibility

### Search Space Size Analysis
- Total combinations: 3 * 3 * 2 * cont * cont * cont * 2 * 2 = 72 categorical combinations + 3 continuous
- 20 trials covers ~28% of categorical space with TPE-guided continuous optimization
- This is sufficient for the reduced search space (Attempt 1 had 5 trials over a larger space)

---

## 5. Training Configuration

### Loss Function
**MSE (Mean Squared Error)** on sequence reconstruction:

```python
def compute_loss(reconstruction, target):
    """
    reconstruction: [batch, seq_len, n_features]
    target: [batch, seq_len, n_features]
    """
    return F.mse_loss(reconstruction, target)
```

No contrastive loss in this attempt. Rationale:
- Attempt 1 failure was clearly overfitting + identity mapping, not lack of discriminative power
- The evaluator's priority 1 item (improvement_queue.json) specifies regularization as the primary fix
- Contrastive loss adds implementation complexity and an additional hyperparameter (lambda weight)
- If Gate 1 still fails after regularization fixes, contrastive loss can be added in Attempt 3

### Optimizer
- Adam with weight_decay (searched by Optuna, range 1e-4 to 1e-2)
- Learning rate scheduler: ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)

### Gradient Clipping
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Essential for GRU stability. Prevents exploding gradients during long sequence processing.

### Early Stopping
- Monitor: validation MSE loss
- Patience: 15 epochs
- Restore best weights on stop
- Minimum delta: 1e-6 (ignore tiny improvements)

### Data Loading
- Sequential windows: at each time step t (where t >= W), create a sample `features[t-W:t]` as shape `[W, 8]`
- **NOT flattened** (unlike Attempt 1): GRU processes the sequence dimension natively
- Shuffle training batches: YES (windows are independent samples; temporal ordering within each window is preserved)
- Drop last incomplete batch: YES (avoids BatchNorm issues with batch_size=1)

### Validation Strategy
- Time-series split: train ends before val starts, val ends before test starts
- Evaluate reconstruction MSE on validation set at each epoch
- Best model checkpoint saved based on val loss
- Log overfit ratio (val_loss / train_loss) at each epoch for monitoring

---

## 6. Kaggle Execution Settings

- **enable_gpu**: false
- **Estimated execution time**: 15-25 minutes
- **Estimated memory usage**: <1 GB
- **Rationale**: 2,500 rows, GRU architecture, 20 Optuna trials with max 150 epochs each (early stopping expected at ~30-50). CPU is sufficient. GRU on this scale is compute-light.
- **Required additional pip packages**: [] (PyTorch, pandas, numpy, optuna, fredapi, yfinance all available)
- **enable_internet**: true (needed for FRED API and yfinance data fetch)
- **Kaggle Secrets required**: FRED_API_KEY

### Time Budget Breakdown
| Component | Estimated Time |
|-----------|---------------|
| Data fetch (FRED + yfinance) | 1-2 min |
| Feature engineering | <1 min |
| Optuna 20 trials (avg ~30 epochs each, ~2s/epoch) | 15-20 min |
| Final model retraining | 1-2 min |
| Output generation + saving | <1 min |
| **Total** | **18-25 min** |

---

## 7. Implementation Instructions

### For builder_data

#### Data to Fetch
1. FRED:DFII10, from 2013-06-01 to present (buffer for 252-day lookback before schema start date of 2015-01-30)

Data fetching pipeline is IDENTICAL to Attempt 1. The same `data/processed/real_rate_features.csv` can be reused if it exists. No data changes needed.

#### Feature Engineering Pipeline
Same as Attempt 1:
1. Load DFII10 raw series
2. Drop NaN rows (holidays)
3. Forward-fill to align with gold trading days (use data/raw/gold.csv date index)
4. Compute 8 features (level, change_1d, velocity_20d, velocity_60d, accel_20d, rolling_std_20d, regime_percentile, autocorr_20d)
5. Drop rows with NaN from rolling calculations
6. Trim to schema date range: 2015-01-30 to 2025-02-12
7. Save to data/processed/real_rate_features.csv

**Note**: builder_data can skip this step if `data/processed/real_rate_features.csv` already exists from Attempt 1 with correct row count (2,523).

### For builder_model

#### Self-Contained train.py Structure

```
1. pip install dependencies (fredapi, optuna if not available)
2. Fetch DFII10 from FRED using Kaggle Secret FRED_API_KEY
3. Compute 8 features (same pipeline as builder_data)
4. Align to gold trading days (fetch GC=F from yfinance for date index)
5. Standardize features (fit on train split)
6. Create sequential windows [W, 8] (NOT flattened)
7. Train/val/test split (70/15/15 chronological)
8. Define RealRateGRUAutoencoder model class
9. Optuna HPO (20 trials, 1800s timeout)
10. Retrain best model on train set
11. Generate latent features for ALL dates using best model
12. Apply first-difference postprocessing to latent outputs
13. Save outputs:
    - submodel_output.csv: date, real_rate_latent_0, real_rate_latent_1
    - model.pt: model state dict
    - training_result.json: metrics, params, output shape
```

#### PyTorch Class Design
- Class `RealRateGRUAutoencoder` as specified in Section 3
- Inherit from `nn.Module` (self-contained, no external dependencies)
- Include `transform` method for inference
- Include `encode` method that returns latent representation

#### train.py Specific Notes

1. **FRED API key**: Access via `os.environ['FRED_API_KEY']` (Kaggle Secret). Fail immediately with KeyError if missing.

2. **Gold date index**: Fetch GC=F from yfinance to get the trading day calendar. Only need dates, not prices.

3. **Standardization**: Compute mean/std from train split only. Save these statistics in training_result.json for future inference.

4. **Window creation**: Unlike Attempt 1, do NOT flatten windows. Each sample is `[W, 8]` for GRU input.

5. **GRU dropout handling**: When `gru_num_layers == 1`, set GRU's internal dropout to 0.0 (it would be ignored anyway and triggers a warning). Use the separate `nn.Dropout` layers for regularization instead.

6. **Gradient clipping**: Apply `clip_grad_norm_(model.parameters(), max_norm=1.0)` after each backward pass.

7. **Overfit monitoring**: Log `val_loss / train_loss` ratio at each epoch. If this exceeds 2.0 during HPO, Optuna can prune the trial.

8. **First-difference postprocessing**:
   ```python
   # After generating raw latent z for all dates:
   z_diff = z[1:] - z[:-1]  # First difference
   # Prepend NaN for the first date
   z_output = np.vstack([np.full((1, 2), np.nan), z_diff])
   ```

9. **Output alignment**: The submodel_output.csv must have one row per trading day in the schema date range. For the first W days (insufficient lookback + first-difference), output NaN. The meta-model builder will handle NaN.

10. **Reconstruction quality check**: After training, compute and log reconstruction MSE on each split. Flag if val MSE > 1.5x train MSE.

11. **Autocorrelation check**: Compute and log lag-1 autocorrelation of each latent dimension output. This should be <0.95 for raw latent and <0.5 after first-differencing.

#### Optuna Objective Function Design

```python
def objective(trial):
    # Sample hyperparameters
    window_size = trial.suggest_categorical('window_size', [40, 60, 80])
    gru_hidden_dim = trial.suggest_categorical('gru_hidden_dim', [16, 32, 64])
    gru_num_layers = trial.suggest_categorical('gru_num_layers', [1, 2])
    dropout = trial.suggest_float('dropout', 0.3, 0.5)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    batch_size = trial.suggest_categorical('batch_size', [16, 32])

    # Create model
    model = RealRateGRUAutoencoder(
        n_features=8,
        gru_hidden_dim=gru_hidden_dim,
        gru_num_layers=gru_num_layers,
        latent_dim=2,  # FIXED at 2
        dropout=dropout,
        bidirectional=bidirectional
    )

    # Train with early stopping
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(150):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)

        # Optuna pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

    return best_val_loss
```

---

## 8. Risks and Alternatives

### Risk 1: GRU Still Overfits (overfit_ratio > 1.5)
**Probability**: Low (20%). The combination of tighter bottleneck + stronger dropout + weight decay is aggressive.
**Mitigation**: If overfit ratio is 1.5-1.8, still recoverable: reduce gru_hidden_dim to 16, force num_layers=1.
**Fallback**: If overfit ratio > 2.0, abandon neural approach. Attempt 3: classical HMM with 2-3 states.

### Risk 2: First-Differencing Removes Too Much Signal (Gate 2 MI < 5%)
**Probability**: Low-Medium (25%). First-differencing can destroy low-frequency information.
**Mitigation**: The training_result.json will include BOTH raw and first-differenced autocorrelation/MI estimates. If first-differencing hurts MI, switch to rolling z-score (20-day window).
**Fallback**: Output raw latent values if postprocessing consistently degrades signal. The evaluator will re-assess.

### Risk 3: GRU Training Instability (NaN/divergence)
**Probability**: Low (10%). GRU is generally stable, but long sequences with small batch sizes can cause issues.
**Mitigation**: Gradient clipping (max_norm=1.0), learning rate scheduler, and Optuna will prune unstable trials.
**Fallback**: If >50% of trials diverge, reduce max window_size to 40 and increase batch_size to 32.

### Risk 4: Latent Features Correlate with Yield Curve Submodel
**Probability**: Medium (30%). DFII10 and DGS10 are mathematically related.
**Mitigation**: The autoencoder captures dynamics (velocity, regime changes) not levels. First-differencing further removes level correlation. Gate 2 VIF check will catch issues.
**Fallback**: If VIF > 10, remove `level` from the 8 input features (use only 7 change-based features).

### Risk 5: 20 Optuna Trials Insufficient
**Probability**: Low (15%). With 72 categorical combinations, 20 trials covers ~28%. But TPE is adaptive.
**Mitigation**: MedianPruner with n_warmup_steps=15 allows efficient exploration by terminating bad configurations early.
**Fallback**: If best trial is in the top-3 of last 5 trials (suggesting search hasn't converged), note for Attempt 3 to increase to 40 trials.

### Alternative Architectures (for future attempts if this fails)

| Attempt | Architecture | Rationale |
|---------|-------------|-----------|
| 3 | Multi-country GRU or Transformer | More training data from G10 real rates |
| 4 | Classical HMM (2-3 states) | Explicit regime modeling, no overfitting risk |
| 5 | Supervised MLP | Direct regression from rate features to gold returns (abandons unsupervised) |

---

## Appendix A: Attempt 1 vs Attempt 2 Comparison

```
ATTEMPT 1 (FAILED)                    ATTEMPT 2 (PROPOSED)
==================                    ====================
MLP Autoencoder                       GRU Autoencoder
Flattened input [batch, W*8]          Sequential input [batch, W, 8]
latent_dim = 4                        latent_dim = 2
hidden_dim search: {32, 64, 128}      gru_hidden_dim search: {16, 32, 64}
dropout found: 0.13                   dropout search: [0.3, 0.5]
weight_decay found: 1e-6              weight_decay search: [1e-4, 1e-2]
window_size found: 20                 window_size search: {40, 60, 80}
No output postprocessing              First-difference postprocessing
5 Optuna trials (smoke test)          20 Optuna trials
No gradient clipping                  Gradient clipping max_norm=1.0
batch_size = 64 (fixed)               batch_size search: {16, 32}
max_epochs = 100                      max_epochs = 150
early_stop patience = 10              early_stop patience = 15

RESULTS:                              PREDICTIONS:
overfit_ratio = 2.69                  overfit_ratio = 1.1-1.4
autocorr > 0.995                      autocorr < 0.5 (after diff)
MI increase = 18.5%                   MI increase = 15-20%
Gate 3: no improvement                Gate 3: DA +0.3-0.5%
```

## Appendix B: Output CSV Format

```csv
date,real_rate_latent_0,real_rate_latent_1
2015-01-30,NaN,NaN
...
2015-04-28,0.023,-0.015
2015-04-29,0.011,0.032
...
2025-02-12,-0.008,0.019
```

First ~61 rows (window_size + 1 for first-difference) will be NaN. Values are first-differenced latent dimensions (not bounded to [-1,1] like raw Tanh output -- typical range approximately [-0.2, 0.2]).

## Appendix C: Parameter Count Estimates

| Configuration | Encoder Params | Decoder Params | Total |
|---------------|---------------|---------------|-------|
| hidden=16, layers=1, unidir | 1,200 | 1,672 | 2,872 |
| hidden=32, layers=1, unidir | 3,840 | 4,264 | 8,104 |
| hidden=32, layers=2, unidir | 10,176 | 4,264 | 14,440 |
| hidden=64, layers=1, bidir | 14,208 | 12,616 | 26,824 |
| hidden=64, layers=2, bidir | 39,168 | 12,616 | 51,784 |

All configurations are well within CPU training feasibility. Even the largest (52K params) is 200x smaller than a typical image model.
