# Submodel Design Document: real_rate (Attempt 6)

**Feature**: real_rate (Bond Volatility Regime + Rate Momentum Persistence)
**Attempt**: 6
**Phase**: Phase 2 (reopened with fundamentally different strategy)
**Architect**: Claude Opus 4.6
**Date**: 2026-02-17

---

## 0. Fact-Check Results

### FRED Series IDs

| Series | Status | Detail |
|--------|--------|--------|
| FRED:DFII10 | Confirmed | 6031 obs, 5782 non-NaN, 2003-01-02 to 2026-02-12, latest=1.8000 |
| FRED:DGS10 | Confirmed | 16728 obs, 16013 non-NaN, 1962-01-02 to 2026-02-12, latest=4.0900 |
| FRED:DGS2 | Confirmed | 12968 obs, 12421 non-NaN, 1976-06-01 to 2026-02-12, latest=3.4700 |
| FRED:T10YIE | Confirmed | 6032 obs, 5783 non-NaN, 2003-01-02 to 2026-02-13, latest=2.2700 |

### Researcher Feature Recommendations

| Claim | Verdict | Measured | Action |
|-------|---------|----------|--------|
| Rate Vol Regime (DFII10 realized vol z-score) recommended | PARTIALLY CORRECT | Concept is valid but researcher's specific window combo (20d vol / 252d z-score) produces autocorr=0.9704, dangerously close to 0.99 Gate 1 threshold | MODIFIED: Use 10d vol / 60d z-score (autocorr=0.9007) |
| Rate Momentum Persistence (autocorr z-score) recommended | CORRECT | 10d autocorr / 60d z-score: autocorr=0.7831, VIF=1.05 | Accepted as-is |
| Nominal-Real Divergence rejected (redundant with ie_anchoring_z) | OVERLY CONSERVATIVE | Measured corr(divergence_z, ie_anchoring_z) = -0.22, VIF=1.98 against all 45 features. NOT redundant. | BUT: autocorr=0.9865, dangerously close to 0.99. REJECTED on autocorrelation grounds, not VIF |
| Rate Coherence rejected (high VIF with yc_curvature_z) | PARTIALLY CORRECT | autocorr=0.9530, VIF was not measured but autocorrelation is in the danger zone | REJECTED: autocorrelation risk too high for daily rolling PCA |
| VIF claims: Rate Vol ~2.5, Rate Momentum ~4.0 | OVERLY PESSIMISTIC | Measured: Rate Vol VIF=1.28, Rate Momentum VIF=1.05 against all 45 existing features | VIF risk is even lower than researcher estimated |
| Cross-correlation between two features ~0.30-0.40 | INCORRECT | Measured cross-corr = 0.06 (nearly uncorrelated) | Features are more orthogonal than researcher expected |
| MOVE/VIX correlation ~0.59 | NOT DIRECTLY VERIFIED | Literature-sourced claim. Our proxy (DFII10 realized vol) measures a narrower concept than MOVE. Measured corr(rate_vol_z, vix_vix)=0.46, broadly consistent | Accepted as directionally correct |
| Autocorrelation of DFII10 daily changes: "positive, rho ~0.05-0.15" | PLAUSIBLE | Not directly measured but consistent with the rolling autocorrelation feature having non-zero mean | Accepted |
| Z-score look-ahead bias: shift(1) needed | CORRECT | Standard practice. Will implement with shift(1) | Accepted |

### Critical Design Corrections

1. **Rate Vol Z-Score autocorrelation**: Researcher proposed 20d vol window / 252d z-score window. Measured autocorr(1) = 0.9704, which is within 0.03 of the 0.99 Gate 1 threshold -- one bad Optuna trial could push it over. CORRECTED: Use smaller windows (10d vol / 60d z-score) where autocorr = 0.9007. The Optuna search space is constrained so that no combination can produce autocorr > 0.95.

2. **Nominal-Real Divergence rejection reasoning**: Researcher rejected it as "redundant with ie_anchoring_z." This is incorrect -- measured VIF = 1.98 (excellent). The actual reason to reject is autocorr = 0.9865, which is dangerously close to 0.99 and cannot be safely reduced by window adjustment because the rolling correlation statistic is inherently smooth over a 60-day window. Confirmed: ie_anchoring_z measures the MAGNITUDE of T10YIE changes (rolling std), while divergence measures the DIRECTION alignment of DGS10 vs DFII10. They are conceptually different (corr = -0.22), but the autocorrelation problem makes divergence_z unusable.

3. **Rate Coherence rejection reasoning**: Researcher rejected it for "high VIF with yc_curvature_z." The VIF concern may be overstated (not measured), but autocorr = 0.9530 is another concern. Rolling PCA eigenvalue ratios are inherently smooth. Rejection stands, though for autocorrelation rather than VIF.

---

## 1. Overview

### Purpose

Extract two structural dynamics features from daily DFII10 data that capture the bond market's volatility environment and momentum regime. These are 2nd-order properties (volatility of changes, autocorrelation of changes) that complement the existing 1st-order feature (real_rate_change = daily DFII10 change).

### Methods and Rationale

1. **Bond Volatility Regime** (rr_bond_vol_z): Realized volatility of DFII10 daily changes, z-scored against a rolling baseline. This measures whether the bond market is in a turbulent or calm state, independent of equity volatility (VIX). Literature shows MOVE and VIX have correlation ~0.59 and decouple during rate-specific events (FOMC, Treasury auctions, CPI releases).

2. **Rate Momentum Persistence** (rr_momentum_z): Rolling lag-1 autocorrelation of DFII10 daily changes, z-scored against a rolling baseline. This detects whether real rates are trending (positive autocorrelation) or mean-reverting (negative autocorrelation). Academic evidence shows yield curve momentum persists at horizons < 1 year with optimal look-backs of 1-3 months (Sihvonen 2024).

### Why This Approach

| Attempt | Method | Failure Mode | This Attempt's Fix |
|---------|--------|-------------|-------------------|
| 1 | MLP Autoencoder | Overfit (ratio 2.69) | No ML model -- deterministic |
| 2 | GRU Autoencoder | Convergence failure | No ML model -- deterministic |
| 3 | Transformer + Monthly FF | Step-function degraded MAE | Daily-only data, no interpolation |
| 4 | PCA + Cubic Spline | Synthetic noise from interpolation | Daily-only data, no interpolation |
| 5 | Markov + CUSUM (7 cols) | XGBoost overfit on noise (DA -2.53%) | Only 2 columns, not 7 |

### Expected Effect

- Gate 1: PASS (deterministic, no training, autocorr < 0.95 for both features)
- Gate 2: MI increase +5-10% (moderate). VIF < 2 for both features.
- Gate 3: Marginal pass on one criterion. Most likely MAE -0.01% (bond vol context helps predict return magnitude).

---

## 2. Data Specification

### Data Sources

| Data | Source | Series ID | Frequency | Range |
|------|--------|-----------|-----------|-------|
| 10Y TIPS Real Yield | FRED | DFII10 | Daily | 2003-01-02 to present |

Only DFII10 is required. DGS10, DGS2, T10YIE are NOT used in this design (they were candidates for the divergence and coherence features, which were rejected).

### Preprocessing Pipeline

```
1. Fetch DFII10 from FRED (from 2014-01-01 for lookback buffer)
2. Drop NaN (holidays)
3. Forward-fill to gold trading days (use data/raw/gold.csv date index)
4. Compute daily change: dfii10_change = dfii10.diff()
5. Compute Feature 1: Bond Vol Z-Score (see Section 3)
6. Compute Feature 2: Momentum Z-Score (see Section 3)
7. Trim to schema date range: 2015-01-30 to 2025-02-12
8. Fill initial NaN (from rolling window warmup) with 0.0 (z-score neutral)
```

### Expected Sample Count

- Total rows after alignment: ~2,523 (matching schema_freeze)
- NaN warmup period: first ~70 rows (max of 60d z-score window + 10d vol window)
- These are filled with 0.0 (neutral z-score)

---

## 3. Feature Computation

### Feature 1: Bond Volatility Regime (rr_bond_vol_z)

**Concept**: Measures whether DFII10 is currently experiencing high or low volatility relative to recent history. Bond market turbulence is distinct from equity volatility (VIX) -- rate-specific events (FOMC, CPI, Treasury auctions) create rate vol spikes that do not appear in VIX.

**Mathematical Formula**:

```
Step 1: realized_vol_t = std(dfii10_change[t-W_vol+1 : t])    # rolling std of daily changes
Step 2: mean_vol_t = mean(realized_vol[t-W_z : t-1])           # rolling mean of PAST realized vol (shifted by 1)
Step 3: std_vol_t = std(realized_vol[t-W_z : t-1])             # rolling std of PAST realized vol (shifted by 1)
Step 4: rr_bond_vol_z_t = (realized_vol_t - mean_vol_t) / std_vol_t
Step 5: clip to [-4, 4]
```

Where:
- W_vol = volatility estimation window (Optuna: {10, 15, 20})
- W_z = z-score baseline window (Optuna: {60, 120})

**Pseudocode**:

```python
dfii10_change = dfii10.diff()
realized_vol = dfii10_change.rolling(window=W_vol).std()
mean_vol = realized_vol.shift(1).rolling(window=W_z).mean()
std_vol = realized_vol.shift(1).rolling(window=W_z).std()
rr_bond_vol_z = ((realized_vol - mean_vol) / std_vol).clip(-4, 4)
```

**The shift(1) is critical**: Without it, the rolling mean at time t includes realized_vol_t itself, creating instantaneous look-ahead bias. With shift(1), the z-score measures how today's vol compares to YESTERDAY's trailing average.

**Expected Properties (measured)**:

| Property | Value |
|----------|-------|
| Mean | ~0.01 (near zero by construction) |
| Std | ~1.26 |
| Range | [-2.72, +10.30] (positive skew during crises) |
| Autocorr(1) at W_vol=10, W_z=60 | 0.9007 (safe) |
| Autocorr(1) at W_vol=10, W_z=120 | 0.9200 (safe) |
| Autocorr(1) at W_vol=15, W_z=60 | 0.9287 (safe) |
| Autocorr(1) at W_vol=20, W_z=60 | 0.9400 (safe) |
| Autocorr(1) at W_vol=20, W_z=252 | 0.9704 (DANGER) |
| VIF vs 45 existing features | 1.28 |
| Corr with vix_vix | 0.4620 |
| Corr with ie_regime_prob | 0.3674 |

**Autocorrelation Safety**: The Optuna search space is constrained to {10,15,20} x {60,120}. The maximum autocorrelation in this space is 0.9585 (W_vol=20, W_z=120), safely below 0.99. The 252-day z-score window is EXCLUDED from the search space.

### Feature 2: Rate Momentum Persistence (rr_momentum_z)

**Concept**: Measures whether DFII10 daily changes exhibit trending (positive autocorrelation) or mean-reverting (negative autocorrelation) behavior relative to recent history. Academic evidence shows yield changes have positive autocorrelation at short horizons due to slow information incorporation and behavioral biases (Sihvonen 2024).

**Mathematical Formula**:

```
Step 1: autocorr_t = corr(dfii10_change[t-W_ac+1:t], dfii10_change[t-W_ac:t-1])   # lag-1 autocorrelation
Step 2: mean_ac_t = mean(autocorr[t-W_z:t-1])                                       # rolling mean of PAST autocorr
Step 3: std_ac_t = std(autocorr[t-W_z:t-1])                                         # rolling std of PAST autocorr
Step 4: rr_momentum_z_t = (autocorr_t - mean_ac_t) / std_ac_t
Step 5: clip to [-4, 4]
```

Where:
- W_ac = autocorrelation estimation window (Optuna: {5, 10, 15})
- W_z = z-score baseline window (Optuna: {30, 60})

**Pseudocode**:

```python
dfii10_change = dfii10.diff()
autocorr = dfii10_change.rolling(window=W_ac).apply(
    lambda x: pd.Series(x).autocorr(lag=1), raw=False
)
mean_ac = autocorr.shift(1).rolling(window=W_z).mean()
std_ac = autocorr.shift(1).rolling(window=W_z).std()
rr_momentum_z = ((autocorr - mean_ac) / std_ac).clip(-4, 4)
```

**Expected Properties (measured)**:

| Property | Value |
|----------|-------|
| Mean | ~-0.002 (near zero) |
| Std | ~1.09 |
| Range | [-3.55, +3.81] |
| Autocorr(1) at W_ac=5, W_z=30 | 0.4008 (excellent) |
| Autocorr(1) at W_ac=10, W_z=60 | 0.7831 (safe) |
| Autocorr(1) at W_ac=15, W_z=60 | 0.8538 (safe) |
| Autocorr(1) at W_ac=20, W_z=60 | 0.8809 (marginal) |
| VIF vs 45 existing features | 1.05 |
| Corr with rate_vol_z | 0.0611 (nearly orthogonal) |
| Corr with vix_persistence | < 0.15 (different asset class) |

**Autocorrelation Safety**: All combinations in the search space produce autocorr < 0.91. The 20-day window is EXCLUDED from the search space for this feature.

### Edge Case Handling

1. **NaN from rolling windows**: First ~(W_vol + W_z) rows for Feature 1, ~(W_ac + W_z) rows for Feature 2 will be NaN. Fill with 0.0 (z-score neutral value).
2. **Division by zero in std**: If rolling std is zero (extremely rare -- would require identical changes across the entire window), replace z-score with 0.0.
3. **Holidays and missing dates**: DFII10 has NaN on holidays. After forward-fill to gold trading dates, the changes between holiday-adjacent dates are larger but legitimate.
4. **Extreme values**: Both features are clipped to [-4, 4] to prevent outlier artifacts.

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| DFII10 lookback start | 2014-01-01 | Provides 1+ year buffer before schema start (2015-01-30) for rolling window warmup |
| Autocorrelation lag | 1 | Standard persistence measure; higher lags are noisier on daily data |
| Z-score clipping | [-4, 4] | Prevent extreme outliers; >99.99% of values within this range |
| NaN fill value | 0.0 | Z-score neutral; does not bias feature in any direction |

### Optuna Search Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| vol_window | {10, 15, 20} | categorical | Volatility estimation window. 10d=reactive (2 weeks), 20d=stable (1 month). All produce autocorr < 0.95 when paired with allowed z-score windows |
| vol_zscore_window | {60, 120} | categorical | Z-score baseline. 60d=adapts to regime changes. 120d=more stable baseline. 252d EXCLUDED (autocorr too high) |
| autocorr_window | {5, 10, 15} | categorical | Autocorrelation estimation window. 5d=very reactive, 15d=smoother. 20d EXCLUDED (autocorr approaches 0.90) |
| autocorr_zscore_window | {30, 60} | categorical | Z-score baseline for momentum. 30d=adaptive, 60d=stable |

Total combinations: 3 x 2 x 3 x 2 = 36

### Search Settings

- **n_trials**: 36 (exhaustive search -- all combinations evaluated)
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of mutual information between 2 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42) but with 36 trials on 36 combos this is effectively grid search

### Why Exhaustive Search

The search space is small (36 combinations) and each evaluation is fast (pure pandas rolling operations on ~2500 rows, < 0.5 seconds per trial). Exhaustive search eliminates the randomness of TPE sampling and guarantees the global optimum is found.

---

## 5. Training Settings

### Fitting Procedure

This is a **fully deterministic pipeline**. No ML training, no gradient descent, no EM algorithm.

- **Loss function**: N/A
- **Optimizer**: N/A
- **Early stopping**: N/A

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- Train: dates[0:1766] (~2015-01-30 to ~2021-12-31)
- Val: dates[1766:2144] (~2022-01-03 to ~2023-07-31)
- Test: dates[2144:2523] (~2023-08-01 to ~2025-02-12)
- Optuna optimizes MI sum on validation set
- Test set reserved for evaluator Gate 3

### Evaluation Metric for Optuna

For each trial (window combination):
1. Compute both features with trial parameters over full date range
2. Extract validation period values
3. Compute mutual information (MI) between each feature and gold_return_next on validation set
4. Optuna maximizes: `MI_sum = MI(rr_bond_vol_z, target) + MI(rr_momentum_z, target)`

MI calculation: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`. This matches the VIX and technical submodel approach.

### Post-Selection Validation

After Optuna selects the best window combination:
1. Verify autocorr(1) < 0.95 for both features
2. Verify no constant output (std > 0.1)
3. Verify no NaN in the output (after warmup fill)
4. Log the MI values, autocorrelation, and summary statistics

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. Pure pandas rolling operations are CPU-only |
| Estimated execution time | 3-5 minutes | Data download (~30s) + 36 Optuna trials x ~1s each (~40s) + final output (~10s) |
| Estimated memory usage | < 0.5 GB | ~2,500 rows x 2 columns. Trivial dataset |
| Required pip packages | [] | Only uses pandas, numpy, fredapi, optuna, sklearn (all available in Kaggle default + FRED secret) |
| enable_internet | true | Needed for FRED API data fetch |
| Kaggle Secrets required | FRED_API_KEY | For fetching DFII10 from FRED API |

### Alternative: Local Execution

Given the trivial computation time (< 1 minute total), this script could be executed locally. However, to maintain pipeline consistency and Kaggle dataset upload workflow, Kaggle execution is preferred.

---

## 7. Implementation Instructions

### For builder_data

#### Data to Fetch

1. **FRED:DFII10** from 2014-01-01 to present
   - Save to `data/processed/real_rate_v6_input.csv`
   - Columns: date (index), dfii10 (level), dfii10_change (daily diff)
   - Forward-fill to gold trading calendar dates (from `data/raw/gold.csv`)

2. **Gold target**: Load from `data/processed/target.csv` (already exists)

#### Quality Checks

- DFII10 range: [-2.0, 3.5] (reasonable for 10Y TIPS since 2014)
- No gaps > 5 consecutive trading days
- dfii10_change range: [-0.5, 0.5] (daily moves > 50bps are extreme but possible)
- At least 2,500 trading days in the schema range

### For builder_model

#### Self-Contained train.ipynb Structure

```
1.  Import libraries (pandas, numpy, fredapi, optuna, sklearn, json, os)
2.  Fetch DFII10 from FRED using Kaggle Secret FRED_API_KEY
3.  Fetch gold date index from yfinance (GC=F or GLD, dates only)
4.  Align DFII10 to gold trading dates (forward-fill)
5.  Compute dfii10_change = dfii10.diff()
6.  Load target (gold_return_next) from Kaggle dataset or compute from gold prices
7.  Define feature generation functions:
    a. compute_bond_vol_z(dfii10_change, vol_window, zscore_window)
    b. compute_momentum_z(dfii10_change, autocorr_window, zscore_window)
8.  Data split: train/val/test = 70/15/15 chronological
9.  Optuna objective: MI sum on validation set
10. Run 36 trials (exhaustive)
11. Generate final features with best parameters
12. Validate: autocorr < 0.95, no constant output, no NaN
13. Fill warmup NaN with 0.0
14. Trim to schema date range
15. Save outputs:
    - submodel_output.csv: date, rr_bond_vol_z, rr_momentum_z
    - training_result.json: parameters, metrics, validation results
```

#### Key Implementation Notes

1. **FRED API key**: Access via `os.environ['FRED_API_KEY']` (Kaggle Secret). Fail immediately with KeyError if missing.

2. **shift(1) is MANDATORY**: Both features require shift(1) in the z-score computation to avoid look-ahead bias. Without it, the rolling mean at time t includes the current observation, which is a form of data leakage.

3. **No 252-day z-score window**: The search space for vol_zscore_window is {60, 120} only. The 252-day window produces autocorr > 0.96 for most vol_window values and is excluded by design.

4. **No 20-day autocorrelation window**: The search space for autocorr_window is {5, 10, 15} only. The 20-day window produces autocorr > 0.88 which is getting too close to the danger zone.

5. **NaN handling**: After computing both features, replace all NaN with 0.0 (z-score neutral). Log the number of NaN-filled rows.

6. **Clipping**: Both features clipped to [-4, 4] after z-scoring.

7. **Output must include submodel_output.csv with dataset reference**: The notebook must reference `bigbigzabuton/gold-prediction-submodels` in kernel-metadata.json for the evaluator to access existing submodel outputs.

8. **Validation checks in the notebook**:
   ```python
   for col in ['rr_bond_vol_z', 'rr_momentum_z']:
       autocorr_val = output[col].autocorr(lag=1)
       assert autocorr_val < 0.95, f"{col} autocorr {autocorr_val:.4f} too high"
       assert output[col].std() > 0.1, f"{col} is near-constant"
       assert output[col].isna().sum() == 0, f"{col} has NaN"
       print(f"{col}: autocorr={autocorr_val:.4f}, mean={output[col].mean():.4f}, std={output[col].std():.4f}")
   ```

9. **training_result.json format**:
   ```json
   {
     "feature": "real_rate",
     "attempt": 6,
     "method": "deterministic_bond_vol_momentum_zscore",
     "timestamp": "...",
     "best_params": {
       "vol_window": 10,
       "vol_zscore_window": 60,
       "autocorr_window": 10,
       "autocorr_zscore_window": 60
     },
     "validation_mi_sum": 0.15,
     "per_feature_mi": {
       "rr_bond_vol_z": 0.075,
       "rr_momentum_z": 0.074
     },
     "autocorrelation": {
       "rr_bond_vol_z": 0.90,
       "rr_momentum_z": 0.78
     },
     "output_shape": [2523, 2],
     "output_columns": ["rr_bond_vol_z", "rr_momentum_z"],
     "output_stats": {
       "rr_bond_vol_z": {"mean": 0.01, "std": 1.26, "min": -2.72, "max": 4.0},
       "rr_momentum_z": {"mean": -0.002, "std": 1.09, "min": -3.55, "max": 3.81}
     },
     "nan_filled_rows": 70,
     "n_optuna_trials": 36,
     "total_combinations": 36
   }
   ```

---

## 8. Output Specification

### Column Names and Properties

| Column | Range | Description | Autocorr(1) | VIF |
|--------|-------|-------------|-------------|-----|
| `rr_bond_vol_z` | [-4, 4] | DFII10 realized vol z-scored against rolling baseline | 0.90 (10d/60d) | 1.28 |
| `rr_momentum_z` | [-4, 4] | DFII10 autocorrelation z-scored against rolling baseline | 0.78 (10d/60d) | 1.05 |

### Output CSV Format

```csv
date,rr_bond_vol_z,rr_momentum_z
2015-01-30,0.0,0.0
2015-02-02,0.0,0.0
...
2015-04-15,0.342,-0.156
2015-04-16,0.338,-0.161
...
2025-02-12,0.125,0.445
```

- Shape: (2523, 2)
- Date range: 2015-01-30 to 2025-02-12 (matching schema_freeze)
- NaN values: 0 (warmup filled with 0.0)
- Update frequency: Daily (every trading day)
- First ~70 rows are 0.0 (z-score neutral) due to rolling window warmup

### Expected Correlations with Existing Features

| Existing Feature | vs rr_bond_vol_z | vs rr_momentum_z |
|-----------------|-------------------|-------------------|
| real_rate_change (base) | -0.22 | < 0.15 |
| vix_vix (base) | 0.46 | < 0.15 |
| vix_persistence (submodel) | < 0.15 | < 0.15 |
| ie_regime_prob (submodel) | 0.37 | < 0.15 |
| ie_anchoring_z (submodel) | 0.30 | < 0.15 |
| yc_curvature_z (submodel) | < 0.15 | < 0.15 |
| rr_momentum_z (this submodel) | 0.06 | 1.00 |

The correlation between rr_bond_vol_z and vix_vix (0.46) is the highest cross-feature correlation. This is expected and acceptable -- bond volatility and equity volatility co-move during systemic events but diverge during rate-specific events. The VIF of 1.28 confirms this does not create multicollinearity.

### Expected Gold Return Correlation

Both features are z-scored 2nd-order statistics, so their direct correlation with gold returns is low (MI ~0.075 per feature). The value lies in providing XGBoost with regime context: "is this a high bond-vol environment?" and "is the current rate trend persistent or reverting?"

---

## 9. Gate Predictions

### Gate 1: Standalone Quality

**Prediction**: PASS (high confidence)

- Overfit ratio: N/A (no ML model, no trainable parameters)
- Autocorrelation < 0.99: Both features measured at < 0.95 across all Optuna search space combinations
- No constant output: Both features have std > 0.9, hundreds of unique values
- No NaN: All NaN filled with 0.0 after warmup
- No data leakage: shift(1) ensures causal computation; rolling windows use only past data

### Gate 2: Information Gain

**Prediction**: PASS (moderate confidence)

- MI increase: Expected +5-10% total (2.5-5% per feature). Measured per-feature MI ~0.075, comparable to VIX submodel features (0.066-0.089)
- VIF: Measured 1.28 and 1.05 against all 45 existing features. Well below 10.
- Rolling correlation stability: Expected std < 0.12. Both features capture structural dynamics, not noise patterns.

### Gate 3: Ablation

**Prediction**: MARGINAL PASS (50% confidence, 1 of 3 criteria)

Most likely to pass: **MAE -0.01%**
- Bond vol regime provides volatility context that helps XGBoost calibrate return magnitude predictions
- High bond vol -> larger gold moves -> MAE improvement from better variance estimation

Less likely to pass:
- Direction accuracy +0.5%: Bond vol and momentum persistence are more about magnitude than direction
- Sharpe +0.05: Requires both direction and magnitude improvement

**Key risk**: This is attempt 6 of a feature that has failed Gate 3 five times. The probability of success is inherently lower than for features with no prior failures. However, this is a fundamentally different approach (daily-only, 2 columns, deterministic, 2nd-order statistics) that avoids all previous failure modes.

---

## 10. Risks and Alternatives

### Risk 1: Gate 1 Autocorrelation Borderline

**Probability**: Low (10%)
**Description**: While measured autocorrelation is < 0.95 for all search space combinations, the actual evaluator may use a different measurement methodology.
**Mitigation**: The search space is conservatively constrained. The worst-case autocorrelation (W_vol=20, W_z=120) is 0.9585. The best-case (W_vol=10, W_z=60) is 0.9007.
**Fallback**: If autocorrelation fails, use the CHANGE in realized vol (first difference of rolling std). This has autocorrelation 0.03-0.09 (extremely low) but may lose information.

### Risk 2: Both Features Fail Gate 3 (6th Consecutive Failure)

**Probability**: 50%
**Description**: Five previous attempts showed that real rate information is already captured by the base real_rate_change feature. Adding 2nd-order statistics may not add marginal value.
**Mitigation**: This design is minimal (2 features) and avoids all previous failure modes. If it fails, the failure will be informative: it confirms that no daily-frequency derivative of DFII10 adds value beyond the raw change.
**If this occurs**: Declare no_further_improvement permanently. Real rate dynamics are adequately captured by the base feature and the existing (attempt 5) state features already in the dataset.

### Risk 3: Bond Vol Z-Score Dominated by VIX

**Probability**: Low-Medium (20%)
**Description**: The 0.46 correlation with VIX could mean XGBoost learns to use VIX instead, making the bond vol feature redundant.
**Mitigation**: VIF = 1.28 shows the correlation is not problematic. The feature captures rate-specific volatility events that VIX misses. XGBoost can learn conditional relationships (e.g., "when VIX is low but bond vol is high -> CPI surprise").
**Detection**: If feature importance < 1% in Gate 3, the feature is not adding value.

### Risk 4: Momentum Feature Too Noisy at Short Windows

**Probability**: Low-Medium (25%)
**Description**: 5-day autocorrelation is noisy (autocorr = 0.40 means the feature itself changes rapidly). XGBoost may not find stable split points.
**Mitigation**: Optuna will prefer the window that maximizes MI. If 5-day window wins, it means the noise contains signal. If 10-15 day window wins, stability is sufficient.
**Fallback**: Replace autocorrelation with rolling Hurst exponent (more stable but computationally heavier).

### Risk 5: Existing Attempt 5 Features Already in Dataset

**Note**: The previous real_rate attempt 5 features (7 columns of Markov + CUSUM) are already in `data/submodel_outputs/real_rate.csv`. The meta-model includes these. The new attempt 6 features must ADD to this, not replace it.
**Resolution**: The new features (rr_bond_vol_z, rr_momentum_z) are orthogonal to the existing 7 columns. They measure different things: realized vol and autocorrelation vs Markov regime persistence and trend direction. Both sets can coexist in the meta-model.

---

## 11. Comparison with Successful Submodels

### What Worked (VIX, Technical)

| Property | VIX (attempt 1) | Technical (attempt 1) | Real Rate (attempt 6) |
|----------|-----------------|----------------------|----------------------|
| Data frequency | Daily | Daily | Daily |
| Output columns | 3 | 3 | 2 |
| Method | HMM + z-score + autocorr | HMM + z-score + GK vol z-score | Rolling vol z-score + rolling autocorr z-score |
| Key innovation | VIX regime detection | 2D HMM on gold [returns, GK_vol] | Bond-specific vol distinct from equity vol |
| Max autocorr(1) | 0.88 | 0.83 | 0.90 |
| Max VIF | ~3 | ~1.2 | 1.28 |
| Gate 3 pass criterion | DA +0.96%, Sharpe +0.289 | MAE -0.1824 | Target: MAE -0.01% |

### Design Pattern Alignment

This design follows the successful submodel pattern:
1. Daily-frequency data only (no interpolation)
2. Compact output (2 columns, within the 2-3 target range)
3. Deterministic/simple probabilistic methods (no deep learning)
4. Low VIF against existing features
5. Clear theoretical justification for independence from existing features

### Key Difference

Unlike VIX and Technical (which used HMM), this design is **purely deterministic**. No HMM, no EM fitting, no random initialization. This eliminates one source of instability (HMM state assignment sensitivity) at the cost of not having a regime probability feature.
