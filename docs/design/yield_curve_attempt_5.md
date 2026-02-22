# Submodel Design Document: Yield Curve (Attempt 5)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| FRED: DGS10 daily | Confirmed | 11 obs in Jan 2025, latest 4.660 |
| FRED: DGS2 daily | Confirmed | Same coverage |
| FRED: DGS5 daily | Confirmed | Same coverage |
| FRED: DGS3MO daily | Confirmed | Same coverage |
| FRED: DGS1 daily | Confirmed | 11 obs in Jan 2025, latest 4.190 |
| Option B (Forward Rates) internal VIF | REJECTED | fwd_2y3y_z vs fwd_5y5y_z corr=0.804. fwd_5y5y_z vs att2 yc_10y3m_velocity_z corr=0.843. Combined VIF reaches 12.5 with fwd_term_spread. Too much redundancy |
| Option C (Cross-Tenor Corr) autocorr | EXCELLENT | All cross-tenor correlation change z-scores have autocorr 0.004-0.053. Near white noise |
| Option C orthogonality with att2 | EXCELLENT | Max correlation with any att2 feature = 0.073. Nearly perfectly orthogonal |
| Option C internal VIF | EXCELLENT | Max internal VIF = 1.52. Max combined VIF (att2 + proposed) = 1.55 |
| Option C MI with gold_return_next | CONFIRMED | 0.063-0.072 range (20-bin). Comparable to att2 feature MIs (0.067-0.077) |
| Option C stability | EXCELLENT | Rolling corr std max 0.072 (threshold < 0.15). Well below limit |
| Option D (Simple Combination) | NOT SELECTED | Att4 features failed Gate 3. Adding them to att2 unlikely to help. Also att4 yc_mom_divergence_z had 0.62 corr with att2 yc_10y3m_velocity_z |
| Option A (Multi-Country) | NOT SELECTED | Non-US FRED yield series availability uncertain. International yield proxies via yfinance (^TNX etc) are US-only equity tickers, not foreign yields |
| corr_10y3m vs corr_front (3M-2Y) overlap | REJECTED | Internal corr = 0.713. Replaced corr_front with corr_1y10y (corr = 0.231) |

### Critical Design Decisions from Fact-Check

1. **Option B rejected**: Forward rate change z-scores are heavily correlated with attempt 2's velocity features (fwd_5y5y_z vs yc_10y3m_velocity_z = 0.843). This is because forward rates are linear combinations of spot yields, so their changes are linear combinations of spot yield changes -- the same first-derivative information attempt 2 already captures. Internal VIF reaches 12.5 when combined with term spread.

2. **Option C selected**: Cross-tenor correlation dynamics are a fundamentally different information type -- they capture the STRUCTURE of co-movement between yield tenors, not the direction or speed of movement. Maximum correlation with any attempt 2 feature is 0.073, confirming true orthogonality.

3. **3-feature set optimized**: Initial 3-feature set had corr_10y3m and corr_3mo2y with 0.713 internal correlation (shared DGS3MO component). Replaced corr_3mo2y with corr_1y10y, reducing max internal correlation to 0.566 and max VIF to 1.52.

4. **No ML model needed**: Like attempt 2, this is pure deterministic feature engineering. Cross-tenor correlation changes are computed from rolling windows and z-scored. No neural network or HMM.

---

## 1. Overview

- **Purpose**: Capture yield curve co-movement structure dynamics -- specifically, how the correlation between different tenor yield changes evolves over time. When cross-tenor correlations break down (e.g., long-end and short-end yields start moving independently), this signals a regime shift in rate markets that historically precedes gold price movements.

- **Methods and rationale**:
  1. **Rolling cross-tenor correlation**: Compute 60-day rolling correlation between daily yield changes at different tenors (10Y vs 3M, 10Y vs 2Y, 1Y vs 10Y).
  2. **Daily change in correlation**: Take the first difference of rolling correlation to avoid level autocorrelation (level autocorr = 0.98, change autocorr = 0.05).
  3. **Z-score normalization**: Z-score the correlation changes over a rolling window to make them stationary and comparable.

- **Economic intuition**:
  - **High cross-tenor correlation** = parallel curve shift (single-factor dominance, typically Fed or macro shock). Gold responds predictably.
  - **Low cross-tenor correlation** = yield curve twist or butterfly (multiple factors, term premium repricing, supply/demand imbalance). Gold response is regime-dependent.
  - **Change in correlation** = transition between these states. A sudden decorrelation signals regime uncertainty -- historically associated with safe-haven gold demand.

- **Expected effect**: These features provide the meta-model with information about WHETHER the yield curve is moving as a unit or fragmenting. Attempt 2 tells the meta-model HOW FAST each part moves; attempt 5 tells it WHETHER the parts are moving together.

### Key Differences from Previous Attempts

| Aspect | Attempt 1 | Attempt 2 (best) | Attempt 3 | Attempt 4 | Attempt 5 |
|--------|-----------|-------------------|-----------|-----------|-----------|
| Core method | HMM 2-state | Velocity z-scores | Longer-window z-scores | 2nd-order dynamics | Cross-tenor correlation dynamics |
| Information type | Regime detection | Movement speed | Movement speed (slower) | Acceleration/structure | Co-movement structure |
| Max corr with att2 | N/A | 1.0 | ~0.6 | 0.62 | 0.073 |
| Max autocorr | 1.0 (FAIL) | 0.756 | 0.994 (FAIL) | 0.773 | 0.053 |
| Internal VIF | N/A | 1.74 | N/A | 1.09 | 1.52 |
| Combined VIF (w/att2) | N/A | N/A | N/A | N/A | 1.55 |
| Gate 1 risk | HIGH | LOW | HIGH | LOW | VERY LOW |
| Novelty | Regime state | First derivative | None | Second derivative | Cross-sectional correlation |

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Already Available |
|------|--------|--------|-----------|-------------------|
| 10Y Treasury yield | FRED | DGS10 | Daily | Yes: in base_features |
| 2Y Treasury yield | FRED | DGS2 | Daily | Yes: in base_features |
| 5Y Treasury yield | FRED | DGS5 | Daily | Yes: fetched in attempt 1 |
| 3M Treasury yield | FRED | DGS3MO | Daily | Yes: fetched in attempt 2 |
| 1Y Treasury yield | FRED | DGS1 | Daily | Yes: confirmed available |

### Preprocessing Steps

1. Fetch DGS10, DGS2, DGS5, DGS3MO, DGS1 from FRED, start=2014-06-01 (buffer for warmup: need ~120+ trading days for corr_window + zscore_window)
2. Inner-join on dates to ensure alignment (drop dates where any series has NaN)
3. Compute daily yield changes: `dgs10_chg = DGS10.diff()`, etc.
4. Compute rolling cross-tenor correlations:
   - `corr_10y_3m = dgs10_chg.rolling(corr_window).corr(dgs3mo_chg)`
   - `corr_10y_2y = dgs10_chg.rolling(corr_window).corr(dgs2_chg)`
   - `corr_1y_10y = dgs1_chg.rolling(corr_window).corr(dgs10_chg)`
5. Compute correlation CHANGES: `corr_10y_3m_chg = corr_10y_3m.diff()`
6. Z-score the changes over rolling window
7. Clip to [-4, 4]
8. Forward-fill gaps up to 3 days, then drop remaining NaN
9. Trim output to base_features date range

### Expected Sample Count

- ~2,660 daily observations (matching base_features after warmup)
- Warmup period: corr_window + zscore_window + buffer (~120-180 days)
- All features available for full base_features date range after warmup

---

## 3. Model Architecture

This is a **pure deterministic feature engineering** approach. No ML model.

### Feature 1: Long-End vs Short-End Correlation Change Z-Score (`yc_corr_long_short_z`)

Rolling correlation between DGS10 and DGS3MO daily changes, differenced and z-scored.

```
Input: DGS10, DGS3MO [T x 2]
       |
   dgs10_chg = DGS10.diff()
   dgs3mo_chg = DGS3MO.diff()
       |
   corr = dgs10_chg.rolling(corr_window).corr(dgs3mo_chg)
       |
   corr_chg = corr.diff()       # daily change (avoids level autocorr 0.98)
       |
   z = (corr_chg - rolling_mean(zscore_window)) / rolling_std(zscore_window)
       |
   clip(-4, 4)
       |
Output: yc_corr_long_short_z (typically -3 to +3)
```

- **Interpretation**: Positive = correlation is increasing (curve synchronizing, parallel shift developing). Negative = correlation is decreasing (curve fragmenting, twist developing -- regime uncertainty).
- **Measured autocorrelation**: 0.053 (near white noise)
- **Measured correlation with attempt 2**: max 0.029 (vs yc_spread_velocity_z). Effectively orthogonal.
- **MI with gold_return_next**: 0.068

### Feature 2: Long-End vs Mid-Range Correlation Change Z-Score (`yc_corr_long_mid_z`)

Rolling correlation between DGS10 and DGS2 daily changes, differenced and z-scored.

```
Input: DGS10, DGS2 [T x 2]
       |
   dgs10_chg = DGS10.diff()
   dgs2_chg = DGS2.diff()
       |
   corr = dgs10_chg.rolling(corr_window).corr(dgs2_chg)
       |
   corr_chg = corr.diff()
       |
   z = (corr_chg - rolling_mean(zscore_window)) / rolling_std(zscore_window)
       |
   clip(-4, 4)
       |
Output: yc_corr_long_mid_z (typically -3 to +3)
```

- **Interpretation**: Positive = 10Y and 2Y moving more in sync. Negative = 10Y and 2Y decorrelating (potential term premium shift -- relevant for gold as a term premium asset).
- **Measured autocorrelation**: 0.004 (effectively white noise)
- **Measured correlation with attempt 2**: max -0.054 (vs yc_10y3m_velocity_z). Effectively orthogonal.
- **MI with gold_return_next**: 0.063

### Feature 3: Policy-End vs Long-End Correlation Change Z-Score (`yc_corr_1y10y_z`)

Rolling correlation between DGS1 and DGS10 daily changes, differenced and z-scored.

```
Input: DGS1, DGS10 [T x 2]
       |
   dgs1_chg = DGS1.diff()
   dgs10_chg = DGS10.diff()
       |
   corr = dgs1_chg.rolling(corr_window).corr(dgs10_chg)
       |
   corr_chg = corr.diff()
       |
   z = (corr_chg - rolling_mean(zscore_window)) / rolling_std(zscore_window)
       |
   clip(-4, 4)
       |
Output: yc_corr_1y10y_z (typically -3 to +3)
```

- **Interpretation**: 1Y yield is most directly influenced by Fed policy expectations. 10Y by term premium and long-term inflation expectations. When their correlation drops = disconnect between policy expectations and long-term rates (e.g., market pricing rate cuts while long-end rises on supply concerns). Gold benefits from this confusion.
- **Measured autocorrelation**: 0.034 (near white noise)
- **Measured correlation with attempt 2**: max 0.073 (vs yc_dgs3mo_velocity_z). Effectively orthogonal.
- **MI with gold_return_next**: 0.072

### Combined Output

| Column | Range | Autocorr(1) | VIF (internal) | VIF (w/att2) | Max corr w/att2 |
|--------|-------|-------------|----------------|-------------|-----------------|
| `yc_corr_long_short_z` | [-4, +4] | 0.053 | 1.057 | 1.064 | 0.029 |
| `yc_corr_long_mid_z` | [-4, +4] | 0.004 | 1.472 | 1.475 | 0.054 |
| `yc_corr_1y10y_z` | [-4, +4] | 0.034 | 1.518 | 1.520 | 0.073 |

Total: **3 columns**. All autocorrelations far below 0.95. All VIFs far below 10. All effectively orthogonal to attempt 2 production features.

### Internal Cross-Correlation Matrix

|                       | long_short_z | long_mid_z | 1y10y_z |
|-----------------------|-------------|-----------|---------|
| long_short_z          | 1.000       | 0.153     | 0.231   |
| long_mid_z            | 0.153       | 1.000     | 0.566   |
| 1y10y_z               | 0.231       | 0.566     | 1.000   |

The 0.566 correlation between long_mid_z (10Y-2Y) and 1y10y_z (1Y-10Y) is moderate but acceptable -- both involve DGS10 but capture different maturity pair dynamics.

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Correlation method | Pearson | Standard for yield change co-movement analysis |
| Z-score clipping | [-4, 4] | Prevents extreme outliers consistent with all previous attempts |
| Data start buffer | 2014-06-01 | 180+ trading days before base_features start for all rolling window warmup |
| Yield series | DGS10, DGS2, DGS3MO, DGS1 | 4 tenors covering full curve (3M, 1Y, 2Y, 10Y) |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| corr_window | {30, 45, 60, 90} | categorical | Rolling correlation estimation window. Shorter = more reactive, noisier. 60d default = ~3 trading months |
| zscore_window | {30, 45, 60, 90} | categorical | Z-score normalization window. Independent of corr_window |

Note: All 3 features share the same corr_window and zscore_window. This is intentional -- the correlation dynamics across tenors should use a consistent estimation horizon for interpretability. Total search space: 4 x 4 = 16 combinations.

### Exploration Settings

- **n_trials**: 30 (16 unique combinations + some repetition for robustness check)
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of mutual information between 3 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)
- **Rationale**: Only 16 unique parameter combinations. 30 trials provides complete coverage. Each trial is extremely fast (~0.3 seconds, pure pandas computation).

---

## 5. Training Settings

### Fitting Procedure

This is entirely deterministic feature engineering. No ML model is trained.

1. Fetch DGS10, DGS2, DGS3MO, DGS1 from FRED
2. Compute daily yield changes for all 4 tenors
3. Compute 3 rolling cross-tenor correlations using trial's corr_window
4. Take first difference of each correlation
5. Z-score each correlation change using trial's zscore_window
6. Evaluate MI against gold_return_next on validation set
7. Optuna selects parameters that maximize MI sum

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- All features use backward-looking rolling windows (inherently no lookahead)
- Optuna optimizes MI sum on validation set
- Test set reserved for evaluator Gate 3

### Evaluation Metric for Optuna

For each trial (corr_window, zscore_window combination):
1. Compute all 3 features for full dataset using trial parameters
2. Compute MI between each feature and gold_return_next on validation set
3. Optuna maximizes: `MI_sum = MI(corr_long_short_z, target) + MI(corr_long_mid_z, target) + MI(corr_1y10y_z, target)`

MI calculation: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`.

### Loss Function

N/A -- no gradient-based training. Pure deterministic feature engineering.

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- no iterative training.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. Pure pandas rolling statistics |
| Estimated execution time | 3-5 minutes | Data download (~30s) + 30 Optuna trials x ~0.3s each (~10s) + final output (~30s). Very fast |
| Estimated memory usage | < 0.5 GB | ~2,660 rows x 10 columns. Tiny dataset |
| Required pip packages | None | All computation uses pandas, numpy, sklearn, fredapi (pre-installed on Kaggle) |

---

## 7. Implementation Instructions

### builder_data Instructions

1. All required data (DGS10, DGS2, DGS5, DGS3MO) was already fetched in previous attempts
2. **NEW**: DGS1 (1-Year Treasury) needs to be fetched from FRED
3. Verify DGS1 is available from 2014-06-01 onward
4. If re-fetching needed: start from 2014-06-01 for warmup buffer (need corr_window + zscore_window ~120+ days before base_features start)
5. **Quality checks**:
   - No gaps > 3 consecutive trading days in any series
   - Missing data < 5% for each series
   - All yields in reasonable range (0-8% for recent data)
   - DGS1 available from 1962 onward (no coverage concern)

### builder_model Instructions

#### train.ipynb Structure

```python
"""
Gold Prediction SubModel Training - Yield Curve Attempt 5
Self-contained: Data fetch -> Feature Engineering -> Optuna HPO -> Save results
Approach: Cross-Tenor Correlation Dynamics
"""

# === 1. Libraries ===
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import optuna
import json
import os
from datetime import datetime

# === 2. Constants ===
FEATURE_NAME = "yield_curve"
ATTEMPT = 5
OUTPUT_COLUMNS = ['yc_corr_long_short_z', 'yc_corr_long_mid_z', 'yc_corr_1y10y_z']
CLIP_RANGE = (-4, 4)

# === 3. Dataset Path Resolution ===
# Standard block from MEMORY.md
import glob as _glob
PROBE_FILES = ['base_features.csv', 'base_features_raw.csv', 'vix.csv']
candidates = [
    '/kaggle/input/gold-prediction-submodels',
    '/kaggle/input/datasets/bigbigzabuton/gold-prediction-submodels'
]
DATASET_PATH = None
for c in candidates:
    if os.path.isdir(c) and any(f in os.listdir(c) for f in PROBE_FILES):
        DATASET_PATH = c
        break
    elif os.path.isdir(c):
        print(f'Dir exists but probe files missing: {c} -> {os.listdir(c)[:5]}')
if DATASET_PATH is None:
    raise RuntimeError(
        f'Dataset not found. Tried: {candidates}. '
        f'/kaggle/input/: {os.listdir("/kaggle/input")}'
    )
print(f"DATASET_PATH = {DATASET_PATH}")

# === 4. Data Fetching ===
# 4a. FRED API key from Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    FRED_API_KEY = UserSecretsClient().get_secret("FRED_API_KEY")
except Exception:
    FRED_API_KEY = os.environ.get('FRED_API_KEY')
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY not found in Kaggle Secrets or environment")

from fredapi import Fred
fred = Fred(api_key=FRED_API_KEY)

# 4b. Fetch yield data
print("Fetching FRED yield data...")
dgs10 = fred.get_series('DGS10', observation_start='2014-06-01')
dgs2 = fred.get_series('DGS2', observation_start='2014-06-01')
dgs3mo = fred.get_series('DGS3MO', observation_start='2014-06-01')
dgs1 = fred.get_series('DGS1', observation_start='2014-06-01')

yields_df = pd.DataFrame({
    'dgs10': dgs10, 'dgs2': dgs2, 'dgs3mo': dgs3mo, 'dgs1': dgs1
}).dropna()
print(f"Yield data: {len(yields_df)} rows, {yields_df.index[0]} to {yields_df.index[-1]}")

# 4c. Fetch gold price for target (next-day return)
import yfinance as yf
gold = yf.download('GC=F', start='2014-06-01', progress=False)
gold_close = gold['Close'].squeeze()
gold_ret_next = gold_close.pct_change().shift(-1) * 100
gold_ret_next.index = gold_ret_next.index.tz_localize(None)
gold_ret_next.name = 'gold_return_next'

# 4d. Load base_features for date alignment
bf_path = os.path.join(DATASET_PATH, 'base_features.csv')
if not os.path.exists(bf_path):
    bf_path = os.path.join(DATASET_PATH, 'base_features_raw.csv')
base_features = pd.read_csv(bf_path, parse_dates=['Date'], index_col='Date')
print(f"Base features: {len(base_features)} rows, {base_features.index[0]} to {base_features.index[-1]}")

# === 5. Feature Generation Functions ===

def rolling_zscore(x, window):
    """Rolling z-score with NaN handling."""
    min_per = max(window // 2, 10)
    m = x.rolling(window, min_periods=min_per).mean()
    s = x.rolling(window, min_periods=min_per).std()
    z = (x - m) / s
    z = z.replace([np.inf, -np.inf], np.nan)
    return z


def compute_corr_change_z(series_a_chg, series_b_chg, corr_window, zscore_window):
    """
    Compute the z-scored daily change in rolling correlation
    between two yield change series.

    Steps:
    1. Rolling correlation between daily yield changes
    2. First difference of correlation (to avoid level autocorrelation)
    3. Z-score over rolling window
    4. Clip to [-4, 4]
    """
    corr = series_a_chg.rolling(corr_window, min_periods=max(corr_window // 2, 15)).corr(series_b_chg)
    corr_chg = corr.diff()
    z = rolling_zscore(corr_chg, zscore_window)
    return z.clip(*CLIP_RANGE)


def generate_all_features(yields_df, corr_window, zscore_window):
    """Generate all 3 cross-tenor correlation change z-scores."""
    dgs10_chg = yields_df['dgs10'].diff()
    dgs2_chg = yields_df['dgs2'].diff()
    dgs3mo_chg = yields_df['dgs3mo'].diff()
    dgs1_chg = yields_df['dgs1'].diff()

    features = pd.DataFrame(index=yields_df.index)

    # Feature 1: 10Y vs 3M correlation change
    features['yc_corr_long_short_z'] = compute_corr_change_z(
        dgs10_chg, dgs3mo_chg, corr_window, zscore_window
    )

    # Feature 2: 10Y vs 2Y correlation change
    features['yc_corr_long_mid_z'] = compute_corr_change_z(
        dgs10_chg, dgs2_chg, corr_window, zscore_window
    )

    # Feature 3: 1Y vs 10Y correlation change
    features['yc_corr_1y10y_z'] = compute_corr_change_z(
        dgs1_chg, dgs10_chg, corr_window, zscore_window
    )

    return features

# === 6. MI Computation ===

def compute_mi(feature, target, n_bins=20):
    """MI between feature and target using quantile binning."""
    valid = feature.dropna().index.intersection(target.dropna().index)
    f = feature[valid]
    t = target[valid]
    if len(f) < 50:
        return 0.0
    try:
        f_binned = pd.qcut(f, q=n_bins, labels=False, duplicates='drop')
        t_binned = pd.qcut(t, q=n_bins, labels=False, duplicates='drop')
        return mutual_info_score(f_binned, t_binned)
    except Exception:
        return 0.0

# === 7. Data Split ===
# Align to base_features dates
common_dates = yields_df.index.intersection(base_features.index)
common_dates = common_dates.intersection(gold_ret_next.dropna().index)
common_dates = common_dates.sort_values()

n = len(common_dates)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_dates = common_dates[:train_end]
val_dates = common_dates[train_end:val_end]
test_dates = common_dates[val_end:]

print(f"Split: train={len(train_dates)}, val={len(val_dates)}, test={len(test_dates)}")
print(f"Train: {train_dates[0]} to {train_dates[-1]}")
print(f"Val: {val_dates[0]} to {val_dates[-1]}")
print(f"Test: {test_dates[0]} to {test_dates[-1]}")

# Create val_mask for Optuna
val_mask = yields_df.index.isin(val_dates)

# === 8. Optuna Objective ===

def objective(trial):
    corr_window = trial.suggest_categorical('corr_window', [30, 45, 60, 90])
    zscore_window = trial.suggest_categorical('zscore_window', [30, 45, 60, 90])

    features = generate_all_features(yields_df, corr_window, zscore_window)

    # Compute MI on validation set
    target_val = gold_ret_next.reindex(yields_df.index)[val_mask]
    mi_sum = 0.0
    for col in OUTPUT_COLUMNS:
        feat_val = features[col][val_mask]
        mi_sum += compute_mi(feat_val, target_val)

    return mi_sum

# === 9. Run Optuna ===
print("\nRunning Optuna HPO...")
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=30, timeout=300, show_progress_bar=True)

best_params = study.best_params
best_value = study.best_value
print(f"\nBest params: {best_params}")
print(f"Best MI sum: {best_value:.4f}")

# === 10. Generate Final Features ===
print("\nGenerating final features with best params...")
final_features = generate_all_features(
    yields_df,
    corr_window=best_params['corr_window'],
    zscore_window=best_params['zscore_window']
)

# === 11. Quality Checks ===
print("\n=== Quality Checks ===")

# Autocorrelation check
autocorr_results = {}
for col in OUTPUT_COLUMNS:
    ac = final_features[col].dropna().autocorr(lag=1)
    autocorr_results[col] = ac
    status = "PASS" if abs(ac) < 0.95 else "FAIL"
    print(f"Autocorr {col}: {ac:.4f} [{status}]")

# NaN check
for col in OUTPUT_COLUMNS:
    nan_count = final_features[col].isna().sum()
    nan_pct = nan_count / len(final_features) * 100
    print(f"NaN {col}: {nan_count} ({nan_pct:.1f}%)")

# Internal correlation check
corr_matrix = final_features[OUTPUT_COLUMNS].dropna().corr()
print(f"\nInternal correlation matrix:")
print(corr_matrix.round(3).to_string())

# VIF check
from numpy.linalg import inv as np_inv
X = final_features[OUTPUT_COLUMNS].dropna().values
cm = np.corrcoef(X.T)
try:
    inv_cm = np_inv(cm)
    vif_values = np.diag(inv_cm)
    for col, v in zip(OUTPUT_COLUMNS, vif_values):
        print(f"VIF {col}: {v:.3f}")
except Exception as e:
    print(f"VIF calculation failed: {e}")

# Descriptive statistics
print(f"\nDescriptive statistics:")
print(final_features[OUTPUT_COLUMNS].describe().round(4).to_string())

# === 12. Align to base_features dates and save output ===
output = final_features[OUTPUT_COLUMNS].reindex(base_features.index)
output.index.name = 'Date'

# Forward-fill up to 3 days for minor gaps
output = output.ffill(limit=3)

# Drop rows that are entirely NaN
output = output.dropna(how='all')

print(f"\nOutput shape: {output.shape}")
print(f"Date range: {output.index[0]} to {output.index[-1]}")
print(f"NaN per column after alignment:")
for col in OUTPUT_COLUMNS:
    print(f"  {col}: {output[col].isna().sum()}")

# === 13. Save Results ===
output.to_csv("submodel_output.csv")
print(f"Saved submodel_output.csv ({len(output)} rows)")

# Save training result JSON
metrics = {
    "autocorrelations": autocorr_results,
    "mi_sum_val": best_value,
    "internal_corr_max": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
    "output_rows": len(output),
    "output_nan_counts": {col: int(output[col].isna().sum()) for col in OUTPUT_COLUMNS},
}

result = {
    "feature": FEATURE_NAME,
    "attempt": ATTEMPT,
    "timestamp": datetime.now().isoformat(),
    "best_params": best_params,
    "n_trials": len(study.trials),
    "best_value": best_value,
    "metrics": metrics,
    "output_shape": list(output.shape),
    "output_columns": OUTPUT_COLUMNS,
    "approach": "Cross-Tenor Correlation Dynamics",
    "description": "Z-scored daily changes in rolling cross-tenor yield correlations. 3 features: 10Y-3M corr change, 10Y-2Y corr change, 1Y-10Y corr change."
}

with open("training_result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

print("\nTraining complete!")
print(f"Output columns: {OUTPUT_COLUMNS}")
print(f"Best params: corr_window={best_params['corr_window']}, zscore_window={best_params['zscore_window']}")
```

#### Key Implementation Notes

1. **No pip installs needed**: All computation uses pandas, numpy, sklearn, fredapi (pre-installed on Kaggle).

2. **FRED API key**: Use `from kaggle_secrets import UserSecretsClient; FRED_API_KEY = UserSecretsClient().get_secret("FRED_API_KEY")` with fallback to `os.environ.get('FRED_API_KEY')`.

3. **CHANGE not level for correlations**: Rolling correlation LEVELS have autocorr ~0.98. Taking `.diff()` reduces to ~0.05. This is the critical implementation detail. NEVER output the rolling correlation level directly.

4. **Shared corr_window and zscore_window**: All 3 features use the same windows. This is deliberate -- comparing correlations at different horizons across tenors would mix temporal scales and make interpretation harder.

5. **DGS1 is new data**: Previous attempts used DGS10, DGS2, DGS5, DGS3MO. This attempt adds DGS1 (1-Year Treasury), which needs to be fetched from FRED.

6. **Rolling z-score min_periods**: Set `min_periods=max(window//2, 10)` to avoid NaN explosion at the start.

7. **No lookahead bias**: All features use backward-looking rolling windows. No model fitting occurs. No label information is used in feature computation.

8. **Output format**: CSV with columns [Date, yc_corr_long_short_z, yc_corr_long_mid_z, yc_corr_1y10y_z]. Aligned to trading dates matching base_features.

9. **Autocorrelation sanity check**: After generating final features, compute and print autocorrelation(lag=1) for all 3 features. If any > 0.95, log a warning.

10. **Gold target data**: Fetch GC=F from yfinance, compute next-day return. Used ONLY for MI evaluation in Optuna, NOT for feature computation.

11. **Dataset reference**: kernel-metadata.json MUST include `"bigbigzabuton/gold-prediction-submodels"` in dataset_sources. Load base_features from this dataset for date alignment.

12. **Dynamic dataset mount path**: Use the standard path resolution block from MEMORY.md (try both candidate paths with probe file verification).

13. **Correlation min_periods**: Use `min_periods=max(corr_window//2, 15)` for rolling correlation to avoid computing correlation from too few samples.

---

## 8. Risks and Alternatives

### Risk 1: Cross-Tenor Correlations Are Too Noisy

- **Description**: With autocorrelations 0.004-0.053, all three features are effectively white noise with respect to their own history. XGBoost may not find stable split points.
- **Likelihood**: Moderate (30%)
- **Mitigation**: The MI values (0.063-0.072) are comparable to attempt 2's features (0.067-0.077), confirming non-trivial information content. White noise autocorrelation does NOT mean white noise with respect to the target -- it means the signal is unpredictable from its own past, which is actually desirable (no redundancy with lagged values).
- **Detection**: Feature importance < 0.5% in Gate 3 ablation test.
- **Fallback**: Smooth the correlation change before z-scoring: use `.diff(3)` or `.diff(5)` instead of `.diff(1)` for the correlation. This would increase autocorrelation slightly (to ~0.1-0.2) while potentially improving signal-to-noise.

### Risk 2: Gate 3 Failure Due to Same Pattern as Attempts 3-4

- **Description**: Attempts 3 and 4 both added features with decent MI that failed Gate 3. The same could happen here.
- **Likelihood**: Moderate (40%)
- **Mitigation**: Attempts 3-4 failed for specific identifiable reasons: attempt 3 had a level variable (autocorr 0.994), attempt 4 had 0.62 correlation with existing features (redundancy). Attempt 5 has neither problem: autocorr max 0.053, correlation with att2 max 0.073. The failure mode is fundamentally different. However, adding ANY features to an already-optimized meta-model can degrade performance through noise (the "kitchen sink" problem).
- **Detection**: Gate 3 ablation results.
- **Fallback**: If all 3 features fail as a REPLACEMENT for att2, test as ADDITIVE features alongside att2 (7 total yield curve features). The near-zero correlation with att2 means they should not conflict.

### Risk 3: corr_long_mid_z and corr_1y10y_z Partial Redundancy

- **Description**: Internal correlation between these two features is 0.566 (both involve DGS10). This is the highest internal correlation.
- **Likelihood**: Low (15%)
- **Mitigation**: VIF is only 1.52, well below 10. The 0.566 correlation means ~68% of variance is unique to each. DGS10 is the anchor, but one feature captures the 10Y-2Y relationship (term premium dynamics) while the other captures 1Y-10Y (policy-to-long rate transmission). These are economically distinct channels.
- **Detection**: Gate 2 VIF check. If combined VIF > 5 with other submodel features, consider dropping the lower-MI feature.
- **Fallback**: Drop corr_long_mid_z (lowest MI at 0.063), keep 2 features.

### Risk 4: Optuna Window Selection Has Minimal Impact

- **Description**: With only 16 unique parameter combinations and near-white-noise features, the difference between the best and worst parameter combinations may be negligible.
- **Likelihood**: Moderate (40%)
- **Mitigation**: Even if Optuna impact is small, this is a feature of the approach (robustness to parameter choices), not a bug. The default corr_window=60 should work well.
- **Detection**: Compare best and worst trial MI sums. If < 5% difference, parameters are effectively arbitrary.
- **Fallback**: Use fixed corr_window=60, zscore_window=60 (reasonable defaults based on ~3 months of trading data).

---

## 9. Expected Performance Against Gates

### Gate 1: Standalone Quality

- **Overfit ratio**: N/A (deterministic, no neural network)
- **No constant output**: Confirmed empirically -- all features vary with yield dynamics
- **Autocorrelation < 0.95**: max 0.053 (far below threshold)
- **No NaN values**: After warmup with forward-fill

**Expected Result**: PASS (very high confidence, 98%)

### Gate 2: Information Gain

- **MI increase > 5%**: Individual MI values 0.063-0.072 (measured). Sum MI ~0.20 should provide >5% increase over base. Near-zero correlation with existing features maximizes unique MI contribution.
- **VIF < 10**: Internal VIF max 1.52. Combined VIF with att2 max 1.55. Far below threshold.
- **Rolling correlation std < 0.15**: All features max 0.072. Far below threshold.

**Expected Result**: PASS (high confidence, 85%)

### Gate 3: Ablation Test

- **Direction accuracy +0.5%**: Uncertain. Yield curve features have historically struggled with DA.
- **OR Sharpe +0.05**: Possible. New orthogonal information could improve risk-adjusted returns.
- **OR MAE -0.01%**: Most likely path. Near-zero correlation with existing features means the new information should be additive rather than conflicting.

**Expected Result**: 50-55% confidence of PASS. The orthogonality is very strong but the overall MI per feature (~0.07) is modest.

**Overall Confidence**: 55% (moderate). The strong orthogonality and excellent Gate 1/2 properties give confidence in feature quality. The uncertainty is entirely in Gate 3, where even high-quality features may not translate to meta-model improvement.
