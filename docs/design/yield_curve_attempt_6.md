# Submodel Design Document: Yield Curve (Attempt 6)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| FRED: DGS10 daily | Confirmed | 297 obs in 2025, latest 4.08 (2026-02-19) |
| FRED: DFII10 daily | Confirmed | 297 obs in 2025, latest 1.79 (2026-02-19). 10Y TIPS real yield |
| FRED: T10YIE daily | Confirmed | 298 obs in 2025, latest 2.28 (2026-02-20). 10Y Breakeven Inflation Rate |
| FRED: DGS2 daily | Confirmed | Available from 2014-06-01 |
| FRED: DGS3MO daily | Confirmed | Available from 2014-06-01 |
| Identity: DGS10 = DFII10 + T10YIE (approx) | Confirmed | Nominal 10Y = Real 10Y + Breakeven Inflation |
| Proposed be_vel_z autocorr | Confirmed: 0.046 | Near white noise. No Gate 1 risk |
| Proposed tips_vel_z autocorr | Confirmed: 0.023 | Near white noise. No Gate 1 risk |
| Proposed be_vel_z corr with att2 features | Confirmed: max 0.157 | Low. 97% unique variance vs existing features |
| Proposed tips_vel_z corr with att2 features | Confirmed: max 0.240 | Low. 94% unique variance vs existing features |
| Internal corr(be_vel_z, tips_vel_z) | Confirmed: -0.040 | Near-zero. Effectively independent |
| Internal VIF | Confirmed: max 1.04 | Excellent |
| Combined VIF (with att2 features) | Confirmed: max 1.36 | Excellent |
| MI(be_vel_z, gold_return_next) | Confirmed: 0.063 | Comparable to att2 feature MIs (0.067-0.077) |
| MI(tips_vel_z, gold_return_next) | Confirmed: 0.066 | Comparable to att2 feature MIs (0.067-0.077) |
| R2(be_vel_z ~ att2 features) | Confirmed: 0.027 | Only 2.7% explained by existing -- 97.3% unique |
| R2(tips_vel_z ~ att2 features) | Confirmed: 0.059 | Only 5.9% explained by existing -- 94.1% unique |
| Spread level z (long window) autocorr | REJECTED | Even window=504 gives autocorr 0.9945. Level variables always fail Gate 1 |
| 3M velocity z as 3rd feature | CONSIDERED BUT EXCLUDED | Corr with existing yc_spread_velocity_z = -0.21 (highest overlap). Adds complexity for marginal MI. Simpler 2-feature approach preferred given attempts 3-5 all failed by adding too many features |
| nom_real_div_z redundancy | CONFIRMED REDUNDANT | corr(be_vel_z, nom_real_div_z) = 1.000. Identical by construction (DGS10.diff() - DFII10.diff() = breakeven.diff()). Use be_vel_z only |

### Critical Design Decisions from Fact-Check

1. **Yield decomposition is genuinely novel**: The existing yc_spread_velocity_z captures the velocity of (DGS10 - DGS3MO). But DGS10 itself equals DFII10 + T10YIE (real rate + breakeven inflation). The existing feature mixes real rate dynamics and inflation premium dynamics into a single number. Decomposing yields into their real and inflation components provides information the meta-model currently cannot access.

2. **Only 2 features**: Attempts 3 (4 features), 4 (4 features), and 5 (3 features) all failed Gate 3. The pattern is clear: adding more yield curve features introduces noise that hurts the meta-model. Attempt 2 succeeded with 4 features but was the first attempt. Attempt 6 uses 2 features to minimize noise risk.

3. **Velocity approach maintained**: Attempt 2 succeeded with velocity z-scores. Attempts 3-5 tried different information types (level, acceleration, cross-tenor correlation) and all failed. Attempt 6 returns to velocity z-scores but applied to different underlying series (real rate and breakeven, not nominal spread).

4. **No level variables**: Every level variable tested across 5 attempts had autocorrelation above 0.95. Even with a 504-day z-score window, spread levels have autocorr 0.9945. Level variables are permanently excluded.

5. **Economic orthogonality**: be_vel_z captures how fast the market's inflation premium is changing. tips_vel_z captures how fast real rates are changing. yc_spread_velocity_z captures nominal spread velocity. yc_curvature_z captures butterfly dynamics. These are four economically distinct channels.

---

## 1. Overview

- **Purpose**: Decompose nominal yield dynamics into their real rate and inflation premium components, then capture the velocity of each component independently. This provides the meta-model with information about WHETHER yield moves are driven by real rate changes (monetary policy tightening/easing) or inflation expectation changes (inflation fears/comfort) -- a distinction that has different implications for gold prices.

- **Methods and rationale**:
  1. **Breakeven velocity z-score** (`yc_be_vel_z`): Daily change in the 10Y breakeven inflation rate (DGS10 - DFII10), z-scored over a rolling window. Captures the speed of change in the market's inflation premium. Positive = rising inflation expectations (gold-positive). Negative = falling inflation expectations (gold-negative).
  2. **TIPS velocity z-score** (`yc_tips_vel_z`): Daily change in the 10Y TIPS real yield (DFII10), z-scored over a rolling window. Captures the speed of real rate changes. Positive = rising real rates (gold-negative, higher opportunity cost). Negative = falling real rates (gold-positive, lower opportunity cost).

- **Economic intuition**:
  - Gold is priced in nominal terms but responds to REAL rates (opportunity cost) and INFLATION expectations (real asset demand).
  - When yields rise, the gold market cares whether it is because real rates are rising (negative for gold) or because inflation expectations are rising (positive for gold).
  - The existing yc_spread_velocity_z cannot distinguish these two channels because it uses nominal yields only.
  - Example: In Q4 2022, 10Y nominal yields were stable (~3.8-4.0%), but DFII10 rose 50bp while T10YIE fell 50bp. This was a massive real rate tightening masked by stable nominal yields. Gold fell 8% despite "stable" yields. The proposed features would capture this.

- **Expected effect**: Provide the meta-model with the ability to distinguish real-rate-driven yield moves (gold-negative) from inflation-expectation-driven yield moves (gold-positive or neutral). This decomposition is economically fundamental and not captured by any existing submodel feature.

### Key Differences from Previous Attempts

| Aspect | Attempt 1 | Attempt 2 (best) | Attempt 3 | Attempt 4 | Attempt 5 | Attempt 6 |
|--------|-----------|-------------------|-----------|-----------|-----------|-----------|
| Core method | HMM 2-state | Velocity z-scores | Longer-window z | 2nd-order dynamics | Cross-tenor corr | Yield decomposition velocity |
| Information type | Regime detection | Nominal spread speed | Same (slower) | Acceleration/structure | Co-movement structure | Real vs inflation speed |
| # features | 2 | 4 | 4 | 4 | 3 | 2 |
| Max corr w/att2 | N/A | 1.0 | ~0.6 | 0.62 | 0.073 | 0.240 |
| Max autocorr | 1.0 (FAIL) | 0.760 | 0.994 (FAIL) | 0.773 | 0.074 | 0.046 |
| Gate 1 risk | HIGH | LOW | HIGH | LOW | VERY LOW | VERY LOW |
| Gate 3 result | PASS (MAE) | PASS (all) | FAIL | FAIL | FAIL | TBD |
| Novel information | Regime state | Nominal 1st deriv | None | Nominal 2nd deriv | Cross-sectional corr | Real/inflation decomposition |
| Uses DFII10/T10YIE? | No | No | No | No | No | Yes (new data source) |

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Already Available |
|------|--------|--------|-----------|-------------------|
| 10Y Treasury yield | FRED | DGS10 | Daily | Yes: in base_features |
| 10Y TIPS yield | FRED | DFII10 | Daily | Yes: real_rate submodel uses it |
| 2Y Treasury yield | FRED | DGS2 | Daily | Yes: in base_features |
| 3M Treasury yield | FRED | DGS3MO | Daily | Yes: fetched in attempt 2 |

Note: T10YIE (10Y Breakeven) is NOT fetched directly. It is computed as DGS10 - DFII10, which equals T10YIE by definition. This avoids an extra API call and ensures exact alignment.

### Preprocessing Steps

1. Fetch DGS10, DFII10, DGS2, DGS3MO from FRED, start=2014-06-01 (buffer for warmup)
2. Inner-join on dates to ensure alignment (drop dates where any series has NaN)
3. Compute:
   - `breakeven = DGS10 - DFII10` (10Y breakeven inflation rate)
   - `be_vel = breakeven.diff()` (daily change in breakeven)
   - `tips_vel = DFII10.diff()` (daily change in real rate)
4. Z-score each velocity over rolling window
5. Clip to [-4, 4]
6. Forward-fill gaps up to 3 days, then drop remaining NaN
7. Trim output to base_features date range

### Expected Sample Count

- ~2,660 daily observations (matching base_features after warmup)
- Warmup period: zscore_window + 1 day (for diff) + buffer (~90-120 days)
- DFII10 available from 2003 onward (no coverage concern for 2014-06-01 start)

---

## 3. Model Architecture

This is a **pure deterministic feature engineering** approach. No ML model is trained.

### Feature 1: Breakeven Velocity Z-Score (`yc_be_vel_z`)

Rolling z-score of the daily change in the 10Y breakeven inflation rate.

```
Input: DGS10, DFII10 [T x 2]
       |
   breakeven = DGS10 - DFII10
       |
   be_vel = breakeven.diff()       # daily change in breakeven
       |
   z = (be_vel - rolling_mean(zscore_window)) / rolling_std(zscore_window)
       |
   clip(-4, 4)
       |
Output: yc_be_vel_z (typically -3 to +3)
```

- **Interpretation**: Positive = inflation expectations rising (breakeven widening). Gold-positive because it signals inflation fear / real asset demand. Negative = inflation expectations falling. Gold-negative.
- **Measured autocorrelation**: 0.046 (near white noise)
- **Measured correlation with attempt 2 features**: max 0.157 (vs yc_curvature_z). Low.
- **R-squared vs existing features**: 0.027 (97.3% unique variance)
- **MI with gold_return_next**: 0.063

### Feature 2: TIPS Velocity Z-Score (`yc_tips_vel_z`)

Rolling z-score of the daily change in the 10Y TIPS real yield.

```
Input: DFII10 [T x 1]
       |
   tips_vel = DFII10.diff()         # daily change in real yield
       |
   z = (tips_vel - rolling_mean(zscore_window)) / rolling_std(zscore_window)
       |
   clip(-4, 4)
       |
Output: yc_tips_vel_z (typically -3 to +3)
```

- **Interpretation**: Positive = real rates rising (TIPS yields up). Gold-negative because higher real rates increase the opportunity cost of holding gold. Negative = real rates falling. Gold-positive.
- **Measured autocorrelation**: 0.023 (near white noise)
- **Measured correlation with attempt 2 features**: max 0.240 (vs yc_curvature_z). Low.
- **R-squared vs existing features**: 0.059 (94.1% unique variance)
- **MI with gold_return_next**: 0.066

### Combined Output

| Column | Range | Autocorr(1) | Internal VIF | VIF (w/att2) | Max corr w/att2 | Unique var vs att2 |
|--------|-------|-------------|-------------|-------------|-----------------|-------------------|
| `yc_be_vel_z` | [-4, +4] | 0.046 | 1.002 | 1.066 | 0.157 | 97.3% |
| `yc_tips_vel_z` | [-4, +4] | 0.023 | 1.002 | 1.173 | 0.240 | 94.1% |

Total: **2 columns**. All autocorrelations far below 0.95. Internal correlation -0.040 (nearly independent). Combined VIF max 1.36 (far below 10).

### Why These Two Features Are Nearly Independent

By the yield decomposition identity:

```
DGS10 = DFII10 + T10YIE
DGS10.diff() = DFII10.diff() + breakeven.diff()
```

So nominal yield change = real rate change + breakeven change. These two components are mechanically constrained to sum to the nominal change, but they can vary independently:

- When the Fed raises rates but inflation expectations are anchored, DFII10 rises and breakeven stays flat (tips_vel_z positive, be_vel_z near zero)
- When commodity prices spike, breakeven rises but the Fed hasn't acted yet (be_vel_z positive, tips_vel_z near zero)
- When risk-off flight occurs, both real rates and breakeven can move (both features move)

The measured correlation of -0.040 confirms empirical near-independence.

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Z-score clipping | [-4, 4] | Prevents extreme outliers. Consistent with all previous attempts |
| Data start buffer | 2014-06-01 | 120+ trading days before base_features start for warmup |
| Yield series | DGS10, DFII10 | Core decomposition pair. DGS2, DGS3MO used only for existing att2 features |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| zscore_window | {30, 45, 60, 90, 120} | categorical | Normalization window. Both features share the same window for consistency. Shorter = more responsive to recent regime. Longer = smoother, more stable |

Note: Only 1 hyperparameter with 5 values. This is intentional -- the simpler the search space, the less risk of overfitting the HPO to validation noise. Attempt 5 had 16 combinations; this has 5.

### Exploration Settings

- **n_trials**: 20 (5 unique values + repetition for robustness)
- **timeout**: 180 seconds (3 minutes)
- **objective**: Maximize sum of mutual information between 2 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)
- **Rationale**: Only 5 unique parameter values. 20 trials provides complete coverage with margin. Each trial is extremely fast (~0.2 seconds).

---

## 5. Training Settings

### Fitting Procedure

Entirely deterministic feature engineering. No ML model is trained.

1. Fetch DGS10, DFII10 from FRED
2. Compute breakeven = DGS10 - DFII10
3. Compute daily changes: be_vel = breakeven.diff(), tips_vel = DFII10.diff()
4. Z-score each using trial's zscore_window
5. Evaluate MI against gold_return_next on validation set
6. Optuna selects the window that maximizes MI sum

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- All features use backward-looking rolling windows (inherently no lookahead)
- Optuna optimizes MI sum on validation set only
- Test set reserved for evaluator Gate 3

### Evaluation Metric for Optuna

For each trial (zscore_window value):
1. Compute both features for full dataset using trial parameter
2. Compute MI between each feature and gold_return_next on validation set
3. Optuna maximizes: `MI_sum = MI(yc_be_vel_z, target) + MI(yc_tips_vel_z, target)`

MI calculation: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`.

### Loss Function

N/A -- no gradient-based training.

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- no iterative training.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. Pure pandas rolling statistics |
| Estimated execution time | 2-4 minutes | Data download (~30s) + 20 Optuna trials x ~0.2s each (~4s) + final output (~30s). Very fast |
| Estimated memory usage | < 0.5 GB | ~2,800 rows x 6 columns. Tiny dataset |
| Required pip packages | None | All computation uses pandas, numpy, sklearn, fredapi (pre-installed on Kaggle) |

---

## 7. Implementation Instructions

### builder_data Instructions

1. All required data (DGS10, DGS2, DGS3MO) was already fetched in previous attempts
2. DFII10 is already used by the real_rate submodel and is available in FRED
3. No new data fetch needed -- train.ipynb will fetch directly from FRED inside Kaggle
4. **Quality checks** (performed inside train.ipynb):
   - No gaps > 3 consecutive trading days in DGS10 or DFII10
   - Missing data < 5% for each series
   - All yields in reasonable range (DGS10: 0-8%, DFII10: -2% to 4%)
   - Breakeven (DGS10 - DFII10) should be positive and 0.5-3.5% range for most of the sample

### builder_model Instructions

#### train.ipynb Structure

```python
"""
Gold Prediction SubModel Training - Yield Curve Attempt 6
Self-contained: Data fetch -> Feature Engineering -> Optuna HPO -> Save results
Approach: Yield Decomposition Velocity (Breakeven + TIPS velocity z-scores)
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
ATTEMPT = 6
OUTPUT_COLUMNS = ['yc_be_vel_z', 'yc_tips_vel_z']
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
dfii10 = fred.get_series('DFII10', observation_start='2014-06-01')

yields_df = pd.DataFrame({
    'dgs10': dgs10, 'dfii10': dfii10
}).dropna()
print(f"Yield data: {len(yields_df)} rows, {yields_df.index[0]} to {yields_df.index[-1]}")

# Sanity check: breakeven should be positive
breakeven = yields_df['dgs10'] - yields_df['dfii10']
print(f"Breakeven range: {breakeven.min():.2f}% to {breakeven.max():.2f}%")
print(f"Breakeven mean: {breakeven.mean():.2f}%")

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


def generate_all_features(yields_df, zscore_window):
    """Generate both yield decomposition velocity z-scores."""
    # Breakeven inflation rate = DGS10 - DFII10
    breakeven = yields_df['dgs10'] - yields_df['dfii10']
    be_vel = breakeven.diff()  # daily change in breakeven

    # TIPS real yield change
    tips_vel = yields_df['dfii10'].diff()  # daily change in real rate

    features = pd.DataFrame(index=yields_df.index)

    # Feature 1: Breakeven velocity z-score
    features['yc_be_vel_z'] = rolling_zscore(be_vel, zscore_window).clip(*CLIP_RANGE)

    # Feature 2: TIPS velocity z-score
    features['yc_tips_vel_z'] = rolling_zscore(tips_vel, zscore_window).clip(*CLIP_RANGE)

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

val_mask = yields_df.index.isin(val_dates)

# === 8. Optuna Objective ===

def objective(trial):
    zscore_window = trial.suggest_categorical('zscore_window', [30, 45, 60, 90, 120])

    features = generate_all_features(yields_df, zscore_window)

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
study.optimize(objective, n_trials=20, timeout=180, show_progress_bar=True)

best_params = study.best_params
best_value = study.best_value
print(f"\nBest params: {best_params}")
print(f"Best MI sum: {best_value:.4f}")

# Print all trial results for transparency
print("\n=== All Optuna Trials ===")
for trial in sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True):
    print(f"  zscore_window={trial.params['zscore_window']:>4d}, MI_sum={trial.value:.4f}")

# === 10. Generate Final Features ===
print("\nGenerating final features with best params...")
final_features = generate_all_features(
    yields_df,
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
print(corr_matrix.round(4).to_string())

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

# MI per feature on validation
print(f"\nMI per feature (validation set):")
target_val = gold_ret_next.reindex(yields_df.index)[val_mask]
individual_mi = {}
for col in OUTPUT_COLUMNS:
    mi = compute_mi(final_features[col][val_mask], target_val)
    individual_mi[col] = mi
    print(f"  MI({col}): {mi:.4f}")

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
    "individual_mi": individual_mi,
    "internal_corr": float(corr_matrix.values[0, 1]),
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
    "approach": "Yield Decomposition Velocity",
    "description": "Z-scored daily changes in breakeven inflation rate and TIPS real yield. 2 features: yc_be_vel_z (breakeven velocity), yc_tips_vel_z (real rate velocity). Decomposes nominal yield dynamics into inflation premium and real rate components."
}

with open("training_result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

print("\nTraining complete!")
print(f"Output columns: {OUTPUT_COLUMNS}")
print(f"Best params: zscore_window={best_params['zscore_window']}")
```

#### Key Implementation Notes

1. **No pip installs needed**: All computation uses pandas, numpy, sklearn, fredapi (pre-installed on Kaggle).

2. **FRED API key**: Use `from kaggle_secrets import UserSecretsClient; FRED_API_KEY = UserSecretsClient().get_secret("FRED_API_KEY")` with fallback to `os.environ.get('FRED_API_KEY')`.

3. **Breakeven computed, not fetched**: Compute as `DGS10 - DFII10` rather than fetching T10YIE directly. This ensures exact date alignment and avoids an extra API call.

4. **Only 2 FRED series needed**: DGS10 and DFII10. Much simpler than attempt 5's 4 series.

5. **Shared zscore_window**: Both features use the same normalization window. This is deliberate -- comparing velocities at different time scales would mix temporal semantics.

6. **Velocity, not level**: Always output the z-scored CHANGE (.diff()), never the level. Level variables have autocorr > 0.95 and fail Gate 1.

7. **No lookahead bias**: All features use backward-looking rolling windows. No model fitting occurs. No label information is used in feature computation.

8. **Output format**: CSV with columns [Date, yc_be_vel_z, yc_tips_vel_z]. Aligned to trading dates matching base_features.

9. **Autocorrelation sanity check**: After generating final features, compute and print autocorrelation(lag=1) for both features. If any > 0.95, log a warning.

10. **Gold target data**: Fetch GC=F from yfinance, compute next-day return. Used ONLY for MI evaluation in Optuna, NOT for feature computation.

11. **Dataset reference**: kernel-metadata.json MUST include `"bigbigzabuton/gold-prediction-submodels"` in dataset_sources. Load base_features from this dataset for date alignment.

12. **Dynamic dataset mount path**: Use the standard path resolution block from MEMORY.md (try both candidate paths with probe file verification).

---

## 8. Risks and Alternatives

### Risk 1: Gate 3 Failure (Pattern of Attempts 3-5)

- **Description**: Three consecutive post-attempt-2 attempts all passed Gate 1/2 but failed Gate 3. This could happen again regardless of feature quality.
- **Likelihood**: Moderate (45%)
- **Mitigation**:
  - Attempts 3-5 had specific failure causes: level variable (att3), redundancy with att2 (att4, 0.62 corr), near-zero MI for 2/3 features (att5).
  - Attempt 6 avoids all three: velocity-only (no levels), low correlation with att2 (max 0.24), and both features have verified MI > 0.06.
  - Reduced from 3-4 features to 2 features to minimize noise injection.
  - Uses genuinely new data (DFII10) that no previous yield_curve attempt has used.
- **Detection**: Gate 3 ablation results.
- **Fallback**: If replacement fails, test as additive features alongside att2. The near-zero mutual correlation (<0.24) means the 2 proposed features should not conflict with att2.

### Risk 2: Low Absolute MI

- **Description**: Individual MI values (0.063, 0.066) are at the lower end of att2's range (0.067-0.077). Two features may not provide enough MI sum to meaningfully improve Gate 2.
- **Likelihood**: Low-Moderate (25%)
- **Mitigation**: Total MI sum of ~0.13 from 2 features with >94% unique variance should provide >5% MI increase. Att5 barely passed Gate 2 with MI sum that was dominated by one feature. Att6 has balanced MI across both features.
- **Detection**: Gate 2 MI increase check.
- **Fallback**: Accept if Gate 2 marginally fails but Gate 3 passes (precedent established by vix, technical, cross_asset, etf_flow, cny_demand, regime_classification).

### Risk 3: Overlap with Inflation Expectation Submodel

- **Description**: The inflation_expectation submodel already uses T10YIE (breakeven inflation). The be_vel_z feature is closely related.
- **Likelihood**: Moderate (35%)
- **Mitigation**: The inflation_expectation submodel uses HMM regime detection on T10YIE, producing regime_prob and transition features. be_vel_z is a simple velocity z-score. The information type is completely different (regime state vs. change speed). However, there may be some VIF increase.
- **Detection**: Gate 2 VIF check when combined with inflation_expectation features.
- **Fallback**: If VIF > 5 with inflation_expectation features, consider dropping be_vel_z and keeping only tips_vel_z (which uses DFII10, not T10YIE).

---

## 9. Expected Performance Against Gates

### Gate 1: Standalone Quality

- **Overfit ratio**: N/A (deterministic, no neural network). Will report 1.0.
- **No constant output**: Confirmed empirically -- both features vary daily with yield dynamics.
- **Autocorrelation < 0.95**: max 0.046 (far below threshold).
- **No NaN values**: After warmup with forward-fill.

**Expected Result**: PASS (very high confidence, 99%)

### Gate 2: Information Gain

- **MI increase > 5%**: Both features have individual MI 0.063-0.066 with >94% unique variance vs existing features. Total unique MI contribution should exceed 5% of base sum.
- **VIF < 10**: Combined VIF max 1.36 with att2 features. Should remain low even with other submodel features.
- **Rolling correlation std < 0.15**: Near-white-noise autocorrelation suggests very stable correlation patterns.

**Expected Result**: PASS (high confidence, 80%)

### Gate 3: Ablation Test

- **Direction accuracy +0.5%**: Possible. The real-vs-inflation decomposition provides a genuinely new channel that existing features cannot access. However, historical yield_curve attempts have struggled with DA.
- **OR Sharpe +0.05**: Possible. Real rate velocity is directly linked to gold's opportunity cost, which should help risk-adjusted returns.
- **OR MAE -0.01%**: Most likely path, consistent with att1 (-0.069) and att2 (-0.013) successes.

**Expected Result**: 55% confidence of PASS. The yield decomposition is a fundamentally new information type (not just a different transformation of the same nominal yields). The 2-feature simplicity reduces noise risk. But the track record of 3 consecutive Gate 3 failures warrants caution.

**Overall Confidence**: 55% (moderate). Higher than att5's 50-55% because the approach uses genuinely new data (DFII10) rather than different transformations of the same 4 nominal yield series.
