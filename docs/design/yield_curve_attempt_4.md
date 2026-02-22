# Submodel Design Document: Yield Curve (Attempt 4)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| FRED: DGS10 daily | Confirmed | 7 obs in Jan 2025, latest 2025-01-10 |
| FRED: DGS2 daily | Confirmed | Same coverage |
| FRED: DGS3MO daily | Confirmed | Same coverage |
| FRED: DGS5 daily | Confirmed | Same coverage |
| FRED: DGS1MO daily | Confirmed | Available if needed |
| FRED: DGS6MO daily | Confirmed | Available if needed |
| Option A multi-horizon VIF | CONFIRMED HIGH | corr(5d, 10d) = 0.685, corr(10d, 15d) = 0.790. High cross-correlation makes Option A risky for VIF |
| Option B regime proportions autocorr | TOO HIGH | bear_steep_z autocorr=0.948, bear_flat_z=0.953, bull_flat_z=0.951. All dangerously close to Gate 1 threshold (0.99). Optuna window selection could push these over |
| Option C acceleration autocorr | EXCELLENT | accel_10y3m_z autocorr=-0.496, curv_change_z autocorr=-0.149. Both mean-reverting, no Gate 1 risk |
| Option D momentum divergence autocorr | GOOD | mom_divergence autocorr=0.733 at window=5. Rises to 0.921 at window=20, so Optuna must cap momentum window at 10 |
| Vol ratio change z autocorr | EXCELLENT | Range 0.066-0.137 for all vol windows tested. No Gate 1 risk |
| Cross-feature VIF | EXCELLENT | All VIF 1.01-1.09. Near-perfectly orthogonal feature set |
| MI with gold_ret_next | CONFIRMED | All features MI 0.15-0.19 (20-bin). Higher than attempt 2 raw MI values |
| Correlation with attempt 2 velocity_z | MIXED | accel_z: 0.004 (orthogonal), curv_change_z: 0.233, mom_divergence_z: 0.620 (moderate overlap), vol_ratio_chg_z: -0.078 (orthogonal) |

### Critical Design Decisions from Fact-Check

1. **Option A rejected**: Multi-horizon velocity features have correlations 0.61-0.79, creating VIF risk and redundancy. They also overlap heavily with attempt 2's best feature (yc_10y3m_velocity_z).

2. **Option B rejected**: Rolling regime proportions have autocorrelations 0.92-0.95, dangerously close to Gate 1 failure. A slightly longer Optuna-selected proportion window would push autocorr above 0.99. The autocorrelation comes from the rolling mean of binary indicators -- fundamentally level-like behavior.

3. **Options C + D hybrid selected**: Combining acceleration (2nd derivative, autocorr -0.50) with momentum divergence (autocorr 0.73) and vol ratio change (autocorr 0.12) gives a 4-feature set that is (a) all well below autocorr 0.95, (b) mutually orthogonal (max inter-feature corr 0.25), and (c) fundamentally different from attempt 2's velocity features.

4. **Momentum window capped at 10**: Momentum divergence autocorr rises from 0.60 (window=3) to 0.92 (window=20). Capping at 10 (autocorr 0.84) ensures Gate 1 safety.

5. **Mom divergence has 0.62 correlation with attempt 2 velocity_z**: This is moderate overlap. However, divergence captures WHICH END of the curve is moving (structural information), while velocity captures HOW FAST the spread changes (speed information). The meta-model can distinguish these via feature importance.

---

## 1. Overview

- **Purpose**: Extract second-order dynamics and structural decomposition of yield curve movements that complement attempt 2's first-order velocity features. Attempt 2 captured "how fast" the curve moves; attempt 4 captures "how the movement is accelerating", "which end is driving the move", and "how the volatility structure is shifting".

- **Methods and rationale**:
  1. **Spread acceleration z-score**: Second derivative of the 10Y-3M spread. Captures whether curve flattening/steepening is accelerating or decelerating. Mean-reverting (autocorr -0.50), maximally orthogonal to velocity.
  2. **Curvature change z-score**: Daily change in the 2Y-5Y-10Y butterfly, z-scored. Captures belly dynamics independent of overall slope direction. Nearly white-noise (autocorr -0.15).
  3. **Momentum divergence z-score**: Difference between long-end (DGS10) and short-end (DGS3MO) momentum z-scores. Positive = long-end rising faster = bear steepening (term premium). Negative = short-end rising faster = policy-driven flattening. Decomposes curve moves into their structural drivers.
  4. **Volatility ratio change z-score**: Z-score of the daily change in (short-end vol / long-end vol). Captures regime shifts between policy-dominated (high short-end vol) and term-premium-dominated (high long-end vol) environments. Nearly white-noise (autocorr 0.12).

- **Expected effect**: Attempt 2's velocity features tell the meta-model "the curve is flattening rapidly". Attempt 4 adds: "the flattening is decelerating" (acceleration), "it's the short-end driving the flattening" (divergence), "the belly is bowing out" (curvature change), and "short-end volatility is declining relative to long-end" (vol ratio). This richer context should help the meta-model distinguish transient from persistent curve moves.

### Key Differences from Previous Attempts

| Aspect | Attempt 1 | Attempt 2 (best) | Attempt 3 | Attempt 4 |
|--------|-----------|-------------------|-----------|-----------|
| Core method | 2D HMM + z-scores | Velocity z-scores | Longer-window z-scores | Acceleration + structural decomposition |
| Feature order | Mixed (HMM is 0th order) | 1st derivative | 1st derivative + level | 2nd derivative + structural |
| Gate 1 risk | HIGH (HMM constant) | LOW | HIGH (level autocorr) | LOW (all autocorr < 0.85) |
| Max autocorr | 1.0 (constant) | 0.756 | 0.994 (FAIL) | 0.73 (estimated) |
| Correlation with att.2 | N/A | 1.0 | 0.62 overlap | 0.62 max (divergence) |
| Novel information | Regime detection | Slope velocity | None (worse version of att.2) | Acceleration, structure, vol regime |

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Already Available |
|------|--------|--------|-----------|-------------------|
| 10Y Treasury yield | FRED | DGS10 | Daily | Yes: in base_features |
| 2Y Treasury yield | FRED | DGS2 | Daily | Yes: in base_features |
| 5Y Treasury yield | FRED | DGS5 | Daily | Yes: fetched in attempt 1 |
| 3M Treasury yield | FRED | DGS3MO | Daily | Yes: fetched in attempt 2 |

### Preprocessing Steps

1. Fetch DGS10, DGS2, DGS5, DGS3MO from FRED, start=2014-10-01 (buffer for warmup)
2. Inner-join on dates to ensure alignment (drop dates where any series has NaN)
3. Compute derived quantities:
   - `spread_10y3m = DGS10 - DGS3MO`
   - `vel_10y3m = spread_10y3m.diff()` (1st derivative)
   - `accel_10y3m = vel_10y3m.diff()` (2nd derivative)
   - `curvature = 2 * DGS5 - DGS10 - DGS2` (butterfly)
   - `curv_change = curvature.diff()` (daily change)
   - `dgs10_mom_Nd = DGS10.diff().rolling(N).sum()` (N-day momentum)
   - `dgs3mo_mom_Nd = DGS3MO.diff().rolling(N).sum()`
   - `dgs10_vol_Nd = abs(DGS10.diff()).rolling(N).mean()`
   - `dgs3mo_vol_Nd = abs(DGS3MO.diff()).rolling(N).mean()`
   - `vol_ratio = dgs3mo_vol / dgs10_vol`
   - `vol_ratio_change = vol_ratio.diff()`
4. Forward-fill gaps up to 3 days, then drop remaining NaN
5. Trim output to base_features date range

### Expected Sample Count

- ~2,500 daily observations (matching base_features)
- Warmup period: max(zscore_window + momentum_window + vol_window, ~80-100 days)
- All features available for full base_features date range after warmup

---

## 3. Model Architecture

This is a **pure deterministic feature engineering** approach. No ML model, no HMM. All four features are computed via rolling window statistics on FRED yield data.

### Component 1: Spread Acceleration Z-Score

Second derivative of the 10Y-3M spread, z-scored over a rolling window.

```
Input: DGS10, DGS3MO [T x 2]
       |
   spread_10y3m = DGS10 - DGS3MO
       |
   vel = spread_10y3m.diff()          # 1st derivative (daily)
       |
   accel = vel.diff()                  # 2nd derivative (daily)
       |
   z = (accel - rolling_mean(accel, zscore_window)) / rolling_std(accel, zscore_window)
       |
   clip(-4, 4)
       |
Output: yc_spread_accel_z (typically -3 to +3)
```

- **Interpretation**: Positive = curve steepening is accelerating (or flattening is decelerating). Negative = flattening is accelerating. Captures inflection points in curve dynamics.
- **Measured autocorrelation**: -0.496 (strongly mean-reverting, no Gate 1 risk)
- **Measured correlation with attempt 2 velocity**: 0.004 (orthogonal)

### Component 2: Curvature Change Z-Score

Daily change in the 2Y-5Y-10Y butterfly, z-scored.

```
Input: DGS5, DGS2, DGS10 [T x 3]
       |
   curvature = 2 * DGS5 - DGS10 - DGS2     # butterfly
       |
   curv_change = curvature.diff()             # daily change
       |
   z = (curv_change - rolling_mean(curv_change, zscore_window)) / rolling_std(curv_change, zscore_window)
       |
   clip(-4, 4)
       |
Output: yc_curv_change_z (typically -3 to +3)
```

- **Interpretation**: Positive = belly is bowing outward (5Y yield rising relative to 2Y/10Y interpolation). Negative = belly is flattening. Captures shape dynamics independent of slope.
- **Measured autocorrelation**: -0.149 (near white noise, no Gate 1 risk)
- **Note**: This feature appeared in attempts 1-3 (as yc_curvature_z) and consistently passes all gates. Retained because it is the most orthogonal yield curve feature.

### Component 3: Momentum Divergence Z-Score

Difference between long-end and short-end momentum z-scores. Decomposes curve moves into their structural drivers.

```
Input: DGS10, DGS3MO [T x 2]
       |
   dgs10_mom = DGS10.diff().rolling(momentum_window).sum()    # long-end N-day momentum
   dgs3mo_mom = DGS3MO.diff().rolling(momentum_window).sum()  # short-end N-day momentum
       |
   dgs10_mom_z = (dgs10_mom - rolling_mean(zscore_window)) / rolling_std(zscore_window)
   dgs3mo_mom_z = (dgs3mo_mom - rolling_mean(zscore_window)) / rolling_std(zscore_window)
       |
   divergence = dgs10_mom_z - dgs3mo_mom_z
       |
   clip(-6, 6)    # wider clip since it's a difference of two z-scores
       |
Output: yc_mom_divergence_z (typically -4 to +4)
```

- **Interpretation**: Positive = long-end rising faster than short-end = bear steepening (term premium increasing, typically bullish for gold as risk premium rises). Negative = short-end rising faster = policy-driven flattening (tightening expectations, bearish for gold).
- **Measured autocorrelation**: 0.733 at momentum_window=5 (safe, but rises with window; capped at 10)
- **Measured correlation with attempt 2 velocity**: 0.620 (moderate overlap, but captures different information: WHICH end moves vs HOW FAST the spread changes)
- **VIF with other features**: 1.02 (excellent orthogonality)

### Component 4: Volatility Ratio Change Z-Score

Z-score of the daily change in (short-end volatility / long-end volatility ratio). Captures regime shifts between policy-dominated and term-premium-dominated environments.

```
Input: DGS10, DGS3MO [T x 2]
       |
   dgs10_vol = abs(DGS10.diff()).rolling(vol_window).mean()   # long-end realized vol
   dgs3mo_vol = abs(DGS3MO.diff()).rolling(vol_window).mean() # short-end realized vol
       |
   vol_ratio = dgs3mo_vol / dgs10_vol
       |
   vol_ratio_change = vol_ratio.diff()    # CHANGE to avoid level autocorr
       |
   z = (vol_ratio_change - rolling_mean(zscore_window)) / rolling_std(zscore_window)
       |
   clip(-4, 4)
       |
Output: yc_vol_ratio_chg_z (typically -3 to +3)
```

- **Interpretation**: Positive = short-end vol increasing relative to long-end = shift toward policy uncertainty (Fed action imminent). Negative = long-end vol increasing relative = shift toward term premium uncertainty (supply/inflation concerns). This is a meta-signal about what is DRIVING yield volatility.
- **Measured autocorrelation**: 0.121 at vol_window=20 (near white noise, no Gate 1 risk)
- **Measured correlation with attempt 2 velocity**: -0.078 (orthogonal)
- **Note**: Using CHANGE in vol_ratio (not level) is critical. The level has autocorr ~0.95+ and would risk Gate 1 failure.

### Combined Output

| Column | Range | Autocorr(1) | VIF (internal) | Corr with att.2 velocity |
|--------|-------|-------------|----------------|--------------------------|
| `yc_spread_accel_z` | [-4, +4] | -0.496 | 1.07 | 0.004 |
| `yc_curv_change_z` | [-4, +4] | -0.149 | 1.09 | 0.233 |
| `yc_mom_divergence_z` | [-6, +6] | 0.733 | 1.02 | 0.620 |
| `yc_vol_ratio_chg_z` | [-4, +4] | 0.121 | 1.01 | -0.078 |

Total: **4 columns**. Maximum inter-feature correlation: 0.248 (accel vs curvature change). All autocorrelations well below 0.95.

### Cross-Correlation Matrix (Measured)

|                      | accel_z | curv_change_z | mom_div_z | vol_ratio_chg_z |
|----------------------|---------|---------------|-----------|-----------------|
| accel_z              | 1.000   | 0.248         | -0.003    | -0.040          |
| curv_change_z        | 0.248   | 1.000         | 0.003     | -0.038          |
| mom_divergence_z     | -0.003  | 0.003         | 1.000     | -0.058          |
| vol_ratio_chg_z      | -0.040  | -0.038        | -0.058    | 1.000           |

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Acceleration input | DGS10 - DGS3MO | 10Y-3M spread captures full curve, including inversion dynamics. Matches attempt 2's best feature |
| Curvature formula | 2*DGS5 - DGS10 - DGS2 | Standard belly butterfly. Proven in attempts 1-3 |
| Z-score clipping | [-4, 4] for single z-scores, [-6, 6] for divergence | Prevent extreme outliers. Divergence gets wider range since it's a difference of two z-scores |
| Data start buffer | 2014-10-01 | 100+ trading days before 2015-01-30 for all rolling window warmup |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| accel_zscore_window | {30, 45, 60, 90} | categorical | Shorter = more reactive acceleration signal. 60d default follows proven pattern |
| curv_zscore_window | {30, 45, 60, 90} | categorical | Same rationale. Curvature change is near-white-noise so window matters less |
| momentum_window | {3, 5, 7, 10} | categorical | N-day cumulative yield change. Capped at 10 to keep autocorr < 0.85. 5d = 1 trading week |
| mom_zscore_window | {30, 45, 60, 90} | categorical | Z-score normalization window for each end's momentum |
| vol_window | {10, 15, 20, 30} | categorical | Realized vol estimation window. 20d = 1 trading month. Shorter = noisier vol estimate |
| vol_zscore_window | {30, 45, 60, 90} | categorical | Z-score normalization for vol ratio change |

### Exploration Settings

- **n_trials**: 50
- **timeout**: 600 seconds (10 minutes)
- **objective**: Maximize sum of mutual information between 4 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)
- **Rationale**: Total combinations = 4 * 4 * 4 * 4 * 4 * 4 = 4,096. 50 trials with TPE sampler provides reasonable coverage of the space. Each trial is fast (~0.5 seconds, pure pandas computation).

---

## 5. Training Settings

### Fitting Procedure

This is entirely deterministic feature engineering. No ML model is trained.

1. Fetch DGS10, DGS2, DGS5, DGS3MO from FRED
2. Compute all 4 features using the given window parameters
3. Evaluate MI against gold_return_next on validation set
4. Optuna selects window parameters that maximize MI sum

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- All features use backward-looking rolling windows (inherently no lookahead)
- Optuna optimizes MI sum on validation set
- Test set reserved for evaluator Gate 3

### Evaluation Metric for Optuna

For each trial (window parameter combination):
1. Compute all 4 features for full dataset using trial parameters
2. Compute MI between each feature and gold_return_next on validation set
3. Optuna maximizes: `MI_sum = MI(accel_z, target) + MI(curv_change_z, target) + MI(mom_div_z, target) + MI(vol_ratio_chg_z, target)`

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
| enable_gpu | false | No neural network or HMM. Pure pandas rolling statistics |
| Estimated execution time | 3-5 minutes | Data download (~30s) + 50 Optuna trials x ~0.5s each (~25s) + final output (~30s). Very fast |
| Estimated memory usage | < 0.5 GB | ~2,500 rows x 10 columns. Tiny dataset |
| Required pip packages | None | All computation uses pandas, numpy, sklearn (pre-installed on Kaggle) |

---

## 7. Implementation Instructions

### builder_data Instructions

1. All required data (DGS10, DGS2, DGS5, DGS3MO) was already fetched in attempts 1-2
2. Verify data files exist in `data/processed/` or `data/raw/`
3. If re-fetching needed: start from 2014-10-01 for warmup buffer
4. **Quality checks**:
   - No gaps > 3 consecutive trading days in any series
   - Missing data < 5% for each series
   - All yields in reasonable range (0-8% for recent data)
   - DGS3MO available from 1982 onward (no coverage concern)

### builder_model Instructions

#### train.ipynb Structure

```python
"""
Gold Prediction SubModel Training - Yield Curve Attempt 4
Self-contained: Data fetch -> Feature Engineering -> Optuna HPO -> Save results
Approach: Acceleration + Structural Decomposition (2nd-order dynamics)
"""

# === 1. Libraries ===
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import optuna
import json
import os
from datetime import datetime

# === 2. Data Fetching ===
# Fetch DGS10, DGS2, DGS5, DGS3MO from FRED using Kaggle Secret FRED_API_KEY
# Use: from kaggle_secrets import UserSecretsClient; FRED_API_KEY = UserSecretsClient().get_secret("FRED_API_KEY")
# With fallback: except Exception: FRED_API_KEY = os.environ.get('FRED_API_KEY')
# Fetch gold price from Yahoo Finance (GC=F) for next-day return target
# Align all series on common dates

# === 3. Feature Generation Functions ===

def rolling_zscore(x, window):
    """Rolling z-score with NaN handling."""
    m = x.rolling(window, min_periods=max(window//2, 10)).mean()
    s = x.rolling(window, min_periods=max(window//2, 10)).std()
    z = (x - m) / s
    z = z.replace([np.inf, -np.inf], np.nan)
    return z

def generate_spread_accel(dgs10, dgs3mo, zscore_window):
    """Second derivative of 10Y-3M spread, z-scored."""
    spread = dgs10 - dgs3mo
    velocity = spread.diff()
    accel = velocity.diff()
    z = rolling_zscore(accel, zscore_window)
    return z.clip(-4, 4)

def generate_curv_change(dgs5, dgs2, dgs10, zscore_window):
    """Z-score of daily curvature change (butterfly)."""
    curvature = 2 * dgs5 - dgs10 - dgs2
    curv_change = curvature.diff()
    z = rolling_zscore(curv_change, zscore_window)
    return z.clip(-4, 4)

def generate_mom_divergence(dgs10, dgs3mo, momentum_window, zscore_window):
    """
    Difference between long-end and short-end momentum z-scores.
    Positive = long-end rising faster (bear steepening / term premium).
    Negative = short-end rising faster (policy tightening).
    """
    dgs10_mom = dgs10.diff().rolling(momentum_window).sum()
    dgs3mo_mom = dgs3mo.diff().rolling(momentum_window).sum()
    dgs10_mom_z = rolling_zscore(dgs10_mom, zscore_window)
    dgs3mo_mom_z = rolling_zscore(dgs3mo_mom, zscore_window)
    divergence = dgs10_mom_z - dgs3mo_mom_z
    return divergence.clip(-6, 6)

def generate_vol_ratio_chg(dgs10, dgs3mo, vol_window, zscore_window):
    """
    Z-score of daily CHANGE in (short-end vol / long-end vol ratio).
    Using CHANGE (not level) to avoid autocorrelation.
    Positive = shift toward policy uncertainty (short-end vol rising).
    Negative = shift toward term premium uncertainty (long-end vol rising).
    """
    dgs10_vol = dgs10.diff().abs().rolling(vol_window).mean()
    dgs3mo_vol = dgs3mo.diff().abs().rolling(vol_window).mean()
    vol_ratio = dgs3mo_vol / dgs10_vol
    vol_ratio_change = vol_ratio.diff()
    z = rolling_zscore(vol_ratio_change, zscore_window)
    return z.clip(-4, 4)

# === 4. MI Computation ===

def compute_mi(feature, target, n_bins=20):
    """MI between feature and target using quantile binning."""
    valid = feature.dropna().index.intersection(target.dropna().index)
    f = feature[valid]
    t = target[valid]
    if len(f) < 50:
        return 0.0
    f_binned = pd.qcut(f, q=n_bins, labels=False, duplicates='drop')
    t_binned = pd.qcut(t, q=n_bins, labels=False, duplicates='drop')
    return mutual_info_score(f_binned, t_binned)

# === 5. Optuna Objective ===

def objective(trial, dgs10, dgs2, dgs5, dgs3mo, gold_ret_next, val_mask):
    accel_zscore_window = trial.suggest_categorical('accel_zscore_window', [30, 45, 60, 90])
    curv_zscore_window = trial.suggest_categorical('curv_zscore_window', [30, 45, 60, 90])
    momentum_window = trial.suggest_categorical('momentum_window', [3, 5, 7, 10])
    mom_zscore_window = trial.suggest_categorical('mom_zscore_window', [30, 45, 60, 90])
    vol_window = trial.suggest_categorical('vol_window', [10, 15, 20, 30])
    vol_zscore_window = trial.suggest_categorical('vol_zscore_window', [30, 45, 60, 90])

    f1 = generate_spread_accel(dgs10, dgs3mo, accel_zscore_window)
    f2 = generate_curv_change(dgs5, dgs2, dgs10, curv_zscore_window)
    f3 = generate_mom_divergence(dgs10, dgs3mo, momentum_window, mom_zscore_window)
    f4 = generate_vol_ratio_chg(dgs10, dgs3mo, vol_window, vol_zscore_window)

    # Compute MI on validation set only
    target_val = gold_ret_next[val_mask]
    mi_sum = 0.0
    for feat in [f1, f2, f3, f4]:
        feat_val = feat[val_mask]
        mi_sum += compute_mi(feat_val, target_val)

    return mi_sum

# === 6. Main Execution ===
# 1. Fetch data
# 2. Compute gold_return_next = gold_close.pct_change().shift(-1)
# 3. Split train/val/test 70/15/15
# 4. Run Optuna 50 trials, 600s timeout
# 5. Generate final features with best params
# 6. Check autocorrelation of all features (warn if > 0.95)
# 7. Save to submodel_output.csv and training_result.json
```

#### Key Implementation Notes

1. **No pip installs needed**: All computation uses pandas, numpy, sklearn, fredapi (pre-installed on Kaggle). No hmmlearn.

2. **FRED API key**: Use `from kaggle_secrets import UserSecretsClient; FRED_API_KEY = UserSecretsClient().get_secret("FRED_API_KEY")` with fallback to `os.environ.get('FRED_API_KEY')`.

3. **CHANGE not level for vol_ratio**: The vol_ratio level has autocorr ~0.95+. Taking `.diff()` reduces it to ~0.12. This is the single most important implementation detail to avoid Gate 1 failure (the same mistake that killed attempt 3 with spread_level_z).

4. **Momentum window cap at 10**: Do NOT allow momentum_window > 10 in the Optuna space. At window=15, divergence autocorr=0.88; at window=20, autocorr=0.92. These are safe but uncomfortably close if combined with large zscore windows.

5. **Divergence clipping at [-6, 6]**: Since mom_divergence is a DIFFERENCE of two z-scores, its range is wider than single z-scores. Clip at +/-6 instead of +/-4.

6. **Rolling z-score min_periods**: Set `min_periods=max(window//2, 10)` to avoid NaN explosion at the start. This is safe because the warmup period is excluded from evaluation anyway.

7. **No lookahead bias**: All features use backward-looking rolling windows. No model fitting occurs. No label information is used in feature computation.

8. **Output format**: CSV with columns [Date, yc_spread_accel_z, yc_curv_change_z, yc_mom_divergence_z, yc_vol_ratio_chg_z]. Aligned to trading dates matching base_features.

9. **Autocorrelation sanity check**: After generating final features, compute and print autocorrelation(lag=1) for all 4 features. If any > 0.95, log a warning. This catches unexpected parameter combinations.

10. **Gold target data**: Fetch GC=F from yfinance, compute next-day return. Used only for MI evaluation in Optuna, NOT for feature computation.

11. **Dataset reference**: kernel-metadata.json MUST include `"bigbigzabuton/gold-prediction-submodels"` in dataset_sources. Load base_features from this dataset for date alignment.

12. **Dynamic dataset mount path**: Try both `/kaggle/input/gold-prediction-submodels/` and `/kaggle/input/datasets/bigbigzabuton/gold-prediction-submodels/`. Raise with debug info if neither found.

---

## 8. Risks and Alternatives

### Risk 1: Momentum Divergence Overlaps with Attempt 2 Velocity

- **Description**: yc_mom_divergence_z has 0.62 correlation with attempt 2's yc_10y3m_velocity_z. The evaluator's replacement test may show that divergence adds little beyond what velocity already provides.
- **Likelihood**: Moderate (35%)
- **Mitigation**: The 0.62 correlation means ~38% of variance is unique to divergence. This unique component captures WHICH END drives the move (structural), while velocity captures HOW FAST the spread changes (speed). The meta-model (XGBoost) can exploit this distinction.
- **Detection**: If Gate 3 ablation shows no improvement, check if removing mom_divergence_z and keeping only the other 3 features improves results.
- **Fallback**: Drop mom_divergence_z, keep 3 features (accel, curvature change, vol ratio change). These 3 all have < 0.25 correlation with attempt 2 features.

### Risk 2: Acceleration Feature Too Noisy (Mean-Reverting)

- **Description**: With autocorrelation -0.50, the acceleration z-score oscillates rapidly. Each day's value is largely independent of the previous day. This could be too noisy for the daily prediction meta-model.
- **Likelihood**: Low-Moderate (25%)
- **Mitigation**: The meta-model (XGBoost) handles noisy features well via tree-based feature selection. If the feature is pure noise, it will receive low feature importance and not harm predictions. The MI measurement (0.1534) confirms non-trivial information content.
- **Detection**: Feature importance < 0.5% in ablation test.
- **Fallback**: Replace with a 3-day or 5-day smoothed acceleration: `accel_smooth = vel.diff(3)` instead of `vel.diff(1)`. This would have autocorr closer to 0 (instead of -0.50) and be less noisy.

### Risk 3: Vol Ratio Change Captures Same Info as VIX Submodel

- **Description**: Yield volatility regime shifts may correlate with VIX dynamics, since both capture market uncertainty. The vol_ratio_chg_z could be redundant with vix_persistence or vix_regime_probability.
- **Likelihood**: Low (15%)
- **Mitigation**: VIX captures equity option-implied vol, while vol_ratio captures fixed income realized vol decomposition. These are different asset classes with different drivers. The vol ratio specifically captures the SHORT-END vs LONG-END decomposition, which VIX cannot provide.
- **Detection**: Gate 2 VIF check. If VIF > 5 against VIX features, the feature is partially redundant.
- **Fallback**: Drop vol_ratio_chg_z, keep 3 features.

### Risk 4: Gate 3 Failure Due to Marginal Improvement Over Attempt 2

- **Description**: Attempt 2 already passes Gate 3 via MAE (-0.0127). Attempt 4 may not provide sufficient ADDITIONAL improvement to justify replacement. The evaluator tests replacement, not addition.
- **Likelihood**: Moderate (40%)
- **Mitigation**: Attempt 4's features are largely orthogonal to attempt 2 (max corr 0.62). If both feature sets are available to the meta-model, the combined information should exceed either alone. The evaluator could test 4+4=8 yield curve features instead of replacement.
- **Detection**: If Gate 3 fails as replacement but MI increase (Gate 2) is strong, recommend testing as ADDITIVE features alongside attempt 2 in the meta-model.
- **Fallback**: Accept attempt 2 as final. The yield curve submodel already passes all gates. Attempt 4 is an improvement attempt, not a necessity.

### Risk 5: Optuna Selects Parameters That Push Autocorrelation Near Threshold

- **Description**: Certain combinations (momentum_window=10 + mom_zscore_window=30) could produce autocorrelation > 0.90 for the divergence feature.
- **Likelihood**: Low (10%) -- measured max autocorr at momentum_window=10 is 0.84
- **Mitigation**: Post-Optuna autocorrelation check. If any feature > 0.95, reduce the offending window parameter by one step and regenerate.
- **Detection**: Explicit autocorrelation logging in the notebook after final feature generation.
- **Fallback**: Fix momentum_window=5 (autocorr=0.73) and remove it from Optuna space.

---

## 9. Expected Performance Against Gates

### Gate 1: Standalone Quality

- **Overfit ratio**: N/A (deterministic, no neural network)
- **No constant output**: Confirmed empirically -- all features vary with yield dynamics
- **Autocorrelation < 0.99**: accel=-0.50, curv_change=-0.15, mom_div=0.73, vol_ratio=0.12. All far below threshold.
- **No NaN values**: After warmup with forward-fill

**Expected Result**: PASS (high confidence, 95%)

### Gate 2: Information Gain

- **MI increase > 5%**: Individual feature MI values 0.15-0.19 (measured, 20-bin). Sum MI significantly exceeds 5% threshold.
- **VIF < 10**: Internal VIF all 1.01-1.09 (measured). VIF against base features expected similar to attempts 1-3 (~1.01-1.74).
- **Rolling correlation std < 0.15**: Likely passes for all features given low autocorrelation. Attempts 1-3 had max stability 0.149.

**Expected Result**: PASS (moderate-high confidence, 80%)

### Gate 3: Ablation Test

- **Direction accuracy +0.5%**: Uncertain. Attempt 2 showed -1.08% DA delta (FAIL).
- **OR Sharpe +0.05**: Uncertain. Attempt 2 showed -0.015 Sharpe delta (FAIL).
- **OR MAE -0.01%**: Most likely path. Attempt 2 achieved -0.0127 (1.27x threshold). New orthogonal features should provide at least marginal MAE improvement.

**Expected Result**: PASS via MAE (60% confidence). The main uncertainty is whether replacement of attempt 2 or addition alongside it is the right framing.

**Overall Confidence**: 65% (moderate)
