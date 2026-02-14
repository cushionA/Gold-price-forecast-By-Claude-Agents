# Submodel Design Document: Yield Curve (Attempt 1)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| FRED: DGS5 daily, continuous 2015-2025 | Confirmed | 2532 non-NaN obs, no gaps > 5 days. Range: 2015-01-02 to 2025-02-14 |
| FRED: DGS10 daily | Confirmed | 2532 non-NaN obs, same coverage as DGS5 |
| FRED: DGS2 daily | Confirmed | 2532 non-NaN obs, same coverage as DGS5 |
| DGS30 "2002-2006 gap" | INCORRECT | FRED DGS30 has continuous data throughout 2002-2006 (~250 obs/year). It is a constant-maturity ESTIMATED series, not tied to issuance. The 30-year bond was not auctioned 2001-2006, but the FRED series continued. Irrelevant to design since we use DGS5 |
| Multi-country yields monthly only | Confirmed | IRLTLT01DEM156N (Germany 10Y): 6 obs in Jan-Jun 2020 = monthly. Fatal for daily pipeline |
| Curvature formula: DGS5 - 0.5*(DGS2+DGS10) | Confirmed valid | corr(curvature_raw, spread) = 0.704 at LEVEL. But corr(curvature_z, spread_level) = -0.002. Z-score of changes is orthogonal |
| corr(DGS10_change, DGS2_change) | Measured: 0.761 | Moderate-high. 2D HMM inputs are correlated but capture twist dynamics (parallel vs steepening). Not redundant |
| Expected autocorrelation 0.75-0.95 for regime_prob | Plausible | Actual VIX regime_prob autocorr = 0.14, cross_asset = 0.85, technical = 0.97. Wide range depending on implementation |
| Expected VIF < 5 for all features | Confirmed EXCELLENT | Measured VIF: curvature_z = 1.01, spread_velocity_z = 1.01 against base features including DGS10, DGS2, yield_spread, real_rate |
| Spread velocity z-score autocorr | Measured: 0.749 (5d), 0.615 (3d), 0.855 (10d), 0.914 (20d) | Researcher claimed 0.80-0.90. 5d is actually 0.749, slightly below claim. All below 0.99 |
| Curvature z-score autocorr | Measured: -0.153 | Researcher claimed 0.75-0.85. MUCH LOWER than expected. Daily curvature change z-score is essentially white noise after normalization |
| Quarterly persistence 0.974 for spread | Not directly verifiable | Plausible for levels. Irrelevant since we use changes |
| hmmlearn n_init parameter | DOES NOT EXIST | hmmlearn 0.3.3 GaussianHMM has no n_init. Researcher's HP space includes it. Must use manual multi-restart loop over random_state values |
| MI(curvature_z, gold_return) | Measured: 0.076 | Comparable to VIX regime MI = 0.079 |
| MI(spread_velocity_z, gold_return) | Measured: 0.074 | Comparable to VIX regime MI = 0.079 |
| corr(dgs10_chg, gold_return_next) | Measured: -0.071 | Negative as expected (rising yields bearish for gold) |

### Critical Design Corrections from Fact-Check

1. **hmmlearn has no `n_init` parameter**: The researcher listed `hmm_n_init: [3, 5, 10]` in the Optuna space. This does not exist in hmmlearn 0.3.3. Must implement manual multi-restart via loop over `random_state` values, selecting the model with highest log-likelihood. This matches the pattern used in successful technical submodel.

2. **DGS30 gap claim is incorrect**: The researcher stated DGS30 has a "2002-2006 gap" but FRED data shows continuous coverage. The 30-year bond was not auctioned 2001-2006, but the FRED constant-maturity series (estimated from the yield curve) continued. This does not affect our design since we use DGS5.

3. **Curvature z-score autocorrelation is MUCH lower than claimed**: Researcher predicted 0.75-0.85 but measured -0.153. This is because the daily curvature change is essentially a second derivative of yield levels, which removes most persistence. This is excellent for Gate 1.

4. **2D HMM inputs are moderately correlated (0.761)**: DGS10 and DGS2 changes have 0.761 correlation. The 2D HMM can still distinguish parallel shifts (both positive) from twists (one positive, one negative), but the regime detection may be noisier than with more orthogonal inputs. Design includes fallback to 1D HMM on spread changes if 2D produces poor regime separation.

---

## 1. Overview

- **Purpose**: Extract three dimensions of yield curve dynamics -- curve regime (steepening/neutral/flattening), slope velocity (how fast the curve is moving), and shape dynamics (convexity changes) -- that provide context beyond the raw DGS10, DGS2, and yield_spread levels already in base features.

- **Methods and rationale**:
  1. **2D HMM on [DGS10_change, DGS2_change]**: Detects latent yield curve regimes from daily yield changes. Three states capture (a) parallel shift, (b) steepening (DGS10 rising faster than DGS2), (c) flattening/inversion (DGS2 rising faster). Follows the proven VIX/technical/cross_asset HMM pattern.
  2. **Z-score of 5-day spread change (60-day window)**: Captures the VELOCITY of curve slope movement, normalized by recent history. Tells the meta-model how fast the curve is steepening or flattening relative to its recent behavior.
  3. **Z-score of daily curvature change (60-day window)**: Captures belly convexity dynamics using the DGS5 midpoint. This is orthogonal to the slope (which measures tilt) and captures whether the curve is bowing or straightening.

- **Expected effect**: Identical spread levels have very different gold implications depending on dynamics. A spread of +50bp rapidly narrowing (flattening = tightening expectations = bearish gold) differs from +50bp widening (steepening = easing expectations = bullish gold). The submodel provides this dynamic context that raw levels cannot.

### Key Advantages

1. **Daily native frequency**: All data from FRED at daily frequency. No interpolation (real_rate failure root cause eliminated)
2. **Exactly 3 features**: Compact, follows VIX/technical/cross_asset success pattern
3. **Empirically measured VIF = 1.01**: Lowest VIF of any proposed submodel against base features
4. **All autocorrelations well below 0.99**: curvature_z = -0.15, spread_velocity_z = 0.75
5. **MI comparable to VIX**: curvature_z MI = 0.076, spread_velocity_z MI = 0.074 (VIX regime MI = 0.079)
6. **No multi-country data needed**: US-only design avoids frequency mismatch issues

### Why Not Multi-Country?

FRED only provides monthly yields for Germany, UK, Japan (confirmed: 6 observations in 6 months for IRLTLT01DEM156N). Monthly-to-daily interpolation created step functions that killed all 5 real_rate attempts. Alternative sources require new API keys or paid services (violates constraints). US-only design is sufficient -- prior submodels (VIX, technical, cross_asset, DXY) all succeeded with single-market data.

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Already Available |
|------|--------|--------|-----------|-------------------|
| 10Y Treasury yield | FRED | DGS10 | Daily | Yes: in base_features |
| 2Y Treasury yield | FRED | DGS2 | Daily | Yes: in base_features |
| 5Y Treasury yield | FRED | DGS5 | Daily | Must fetch |

### Preprocessing Steps

1. Fetch DGS10, DGS2, DGS5 from FRED, start=2014-10-01 (buffer for 60-day warmup before 2015-01-30)
2. Inner-join on dates to ensure alignment (drop dates where any series has NaN)
3. Compute derived quantities:
   - `dgs10_change = DGS10.diff()` (daily first difference)
   - `dgs2_change = DGS2.diff()` (daily first difference)
   - `spread = DGS10 - DGS2`
   - `spread_change_Nd = spread.diff(N)` where N = change_window (Optuna parameter)
   - `curvature_raw = DGS5 - 0.5 * (DGS2 + DGS10)`
   - `curvature_change = curvature_raw.diff()`
4. Forward-fill gaps up to 3 days, then drop remaining NaN
5. Trim output to base_features date range: 2015-01-30 to 2025-02-12

### Expected Sample Count

- ~2,523 daily observations (matching base_features row_count)
- Joint non-NaN observations for DGS10, DGS2, DGS5: 2,511 (measured)
- Warmup period: max(60, change_window) days for z-score rolling windows
- HMM trains on full training set, generates probabilities for full dataset

---

## 3. Model Architecture

This is a **hybrid deterministic-probabilistic** approach, matching the pattern used by all 4 successful submodels (VIX, DXY, technical, cross_asset). No PyTorch is required. The pipeline consists of three independent components.

### Component 1: 2D HMM Regime Detection

- **Model**: `hmmlearn.hmm.GaussianHMM`
- **Input**: 2D array of [DGS10_change, DGS2_change] (daily first differences)
  - 2D rationale: Captures both parallel shifts (both change together) and twist/rotation (one changes more than the other). Single DGS10-DGS2 correlation = 0.761, meaning ~42% of variance is in the twist dimension.
- **States**: 2, 3, or 4 (Optuna selects)
  - 3 states (expected optimal): steepening regime, neutral regime, flattening/inversion regime
- **Covariance type**: "full" or "diag" (Optuna selects)
  - "full" captures cross-covariance between DGS10 and DGS2 changes per state
  - "diag" treats them independently per state (simpler, less overfit risk)
- **Training**: Fit on training set data only (70% split). Multi-restart: loop over 5 random_state values [0, 42, 123, 456, 789], select model with highest log-likelihood.
- **Output**: Probability of the state most negatively correlated with next-day gold returns on the training set. This identifies the "most informative for gold" regime.
- **State labeling**: After fitting, compute mean gold_return_next for each state on training data. The state with the most negative mean gold return (tightening/bearish) is the output probability target.

```
Input: [DGS10_change, DGS2_change] [T x 2]
       |
   GaussianHMM.fit(train_data[:, :2]) -> learn 2-4 state model
   (multi-restart: 5 seeds, select best log-likelihood)
       |
   GaussianHMM.predict_proba(full_data) -> [T x n_states]
       |
   Select P(state with most negative gold return correlation) -> [T x 1]
       |
Output: yc_regime_prob (0-1)
```

### Component 2: Spread Velocity Z-Score

- **Model**: Pure pandas computation (no ML)
- **Input**: Yield spread = DGS10 - DGS2
- **Change window**: 3, 5, or 10 days (Optuna selects)
  - Measured autocorrelations: 3d=0.615, 5d=0.749, 10d=0.855 (all safe)
- **Z-score window**: 30, 60, 90, or 120 days (Optuna selects)
  - 60-day default follows VIX success pattern
- **Computation**:
  ```
  spread = DGS10 - DGS2
  spread_change = spread[t] - spread[t - change_window]
  velocity_z = (spread_change - rolling_mean(change_window)) / rolling_std(change_window)
  velocity_z = clip(velocity_z, -4, 4)
  ```
- **Output**: Z-score value (clipped to [-4, +4])
  - Positive: Curve is steepening faster than recently (bullish gold context)
  - Negative: Curve is flattening faster than recently (bearish gold context)

```
Input: DGS10 - DGS2 spread [T x 1]
       |
   diff(change_window) -> spread change velocity
       |
   rolling(zscore_window).mean/std -> normalize
       |
   clip(-4, 4)
       |
Output: yc_spread_velocity_z (typically -3 to +3)
```

### Component 3: Curvature Z-Score

- **Model**: Pure pandas computation (no ML)
- **Input**: DGS5, DGS2, DGS10 levels
- **Curvature formula**: `curvature_raw = DGS5 - 0.5 * (DGS2 + DGS10)`
  - This is the "belly" curvature: how much DGS5 deviates from the linear interpolation between DGS2 and DGS10
  - Positive = curve is bowed/humped at 5Y (5Y yield above linear interpolation)
  - Negative = curve is concave at 5Y (5Y yield below linear interpolation)
- **Z-score window**: 30, 60, 90, or 120 days (Optuna selects)
- **Computation**:
  ```
  curvature_raw = DGS5 - 0.5 * (DGS2 + DGS10)
  curvature_change = diff(curvature_raw)  # daily first difference
  curvature_z = (curvature_change - rolling_mean) / rolling_std
  curvature_z = clip(curvature_z, -4, 4)
  ```
- **Output**: Z-score of daily curvature change (clipped to [-4, +4])
  - Positive: Curve belly is bowing outward faster than recently
  - Negative: Curve belly is flattening/concaving faster than recently
  - Measured autocorrelation: -0.153 (excellent -- nearly white noise)

```
Input: DGS5, DGS2, DGS10 [T x 3]
       |
   curvature_raw = DGS5 - 0.5*(DGS2 + DGS10)
       |
   diff() -> curvature_change (daily first difference)
       |
   rolling(zscore_window).mean/std -> normalize
       |
   clip(-4, 4)
       |
Output: yc_curvature_z (typically -3 to +3)
```

### Combined Output

| Column | Range | Description | VIF vs Base Features | Autocorr(1) |
|--------|-------|-------------|---------------------|-------------|
| `yc_regime_prob` | [0, 1] | P(bearish-for-gold yield curve regime) from 2D HMM | 1.01 (est.) | 0.7-0.95 (est.) |
| `yc_spread_velocity_z` | [-4, +4] | Z-score of N-day spread change | 1.01 (measured) | 0.62-0.85 (measured by window) |
| `yc_curvature_z` | [-4, +4] | Z-score of daily curvature change | 1.01 (measured) | -0.15 (measured) |

Total: **3 columns** (within the proven 2-4 range).

### Orthogonality Analysis (Measured)

| Feature Pair | Correlation | Assessment |
|-------------|-------------|------------|
| curvature_z vs yield_curve_dgs10 (base) | -0.002 | Excellent (orthogonal) |
| curvature_z vs yield_curve_dgs2 (base) | -0.000 | Excellent (orthogonal) |
| curvature_z vs yield_curve_yield_spread (base) | -0.003 | Excellent (orthogonal) |
| curvature_z vs real_rate_real_rate (base) | -0.003 | Excellent (orthogonal) |
| spread_velocity_z vs yield_curve_dgs10 (base) | 0.017 | Excellent (orthogonal) |
| spread_velocity_z vs yield_curve_yield_spread (base) | -0.006 | Excellent (orthogonal) |
| spread_velocity_z vs real_rate_real_rate (base) | 0.023 | Excellent (orthogonal) |
| curvature_z vs spread_velocity_z (between outputs) | 0.071 | Excellent (near-orthogonal) |

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM n_iter | 100 | Standard convergence limit; EM typically converges in 20-50 iterations |
| HMM tol | 1e-4 | Standard convergence tolerance |
| HMM multi-restart seeds | [0, 42, 123, 456, 789] | 5 restarts to avoid local optima. Select best log-likelihood. No n_init in hmmlearn 0.3.3 |
| Log-transform for HMM input | No | Yield changes are approximately normal (symmetric), unlike VIX (right-skewed). No log needed |
| Z-score clipping | [-4, 4] | Prevent extreme outliers from dominating |
| Curvature formula | DGS5 - 0.5*(DGS2 + DGS10) | Standard belly butterfly. Uses DGS5 instead of DGS30 to avoid any data issues |
| Data start buffer | 2014-10-01 | 60+ trading days before 2015-01-30 for z-score warmup |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3, 4} | categorical | 2=bull/bear, 3=steepen/neutral/flatten, 4=adds crisis. Gold HMMs typically work best with 3 |
| hmm_covariance_type | {"full", "diag"} | categorical | full captures DGS10-DGS2 cross-covariance; diag is simpler |
| velocity_change_window | {3, 5, 10} | categorical | 3d=reactive (autocorr 0.62), 5d=balanced (0.75), 10d=smooth (0.85) |
| velocity_zscore_window | {30, 60, 90, 120} | categorical | Shorter=reactive, longer=stable. 60d follows VIX success |
| curvature_zscore_window | {30, 60, 90, 120} | categorical | Same rationale as velocity z-score window |

### Exploration Settings

- **n_trials**: 30
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of mutual information between 3 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)
- **Rationale**: Total combinations = 3 * 2 * 3 * 4 * 4 = 288. 30 trials with TPE sampler provides good exploration. Each trial is fast (HMM fit < 5 seconds, deterministic features < 1 second).

---

## 5. Training Settings

### Fitting Procedure

This is not a gradient-based training loop. The procedure is:

1. **HMM**: `GaussianHMM.fit(yield_changes_train)` -- EM algorithm, converges in seconds. Multi-restart via loop over 5 seeds.
2. **Spread Velocity Z-Score**: Rolling window statistics on spread changes -- deterministic, no fitting.
3. **Curvature Z-Score**: Rolling window statistics on curvature changes -- deterministic, no fitting.

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- HMM fits on train set only
- HMM generates probabilities for full dataset using predict_proba (no lookahead: HMM posterior at time t depends only on observations up to t given the fitted model)
- Z-scores use backward-looking rolling windows (inherently no lookahead)
- Optuna optimizes MI sum on validation set
- Test set reserved for evaluator Gate 3

### Evaluation Metric for Optuna

For each trial (hyperparameter combination):
1. Fit 2D HMM on train set [DGS10_change, DGS2_change]
2. Generate all 3 features for full dataset using fitted HMM and trial window parameters
3. Compute mutual information (MI) between each of the 3 features and `gold_return_next` on validation set
4. Optuna maximizes: `MI_sum = MI(regime_prob, target) + MI(velocity_z, target) + MI(curvature_z, target)`

MI calculation method: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`. This matches the VIX/DXY/technical/cross_asset approach.

### Loss Function

N/A -- no gradient-based training. HMM uses EM (maximum likelihood). Z-score features are deterministic.

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- HMM EM converges via `n_iter` and `tol`. No early stopping needed.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. HMM and rolling statistics are CPU-only |
| Estimated execution time | 3-5 minutes | Data download (~30s) + 30 Optuna trials x ~5s each (~2.5min) + final output (~30s) |
| Estimated memory usage | < 1 GB | ~2,500 rows x 5 columns. Tiny dataset |
| Required pip packages | `hmmlearn` | Must `pip install hmmlearn` at start of train.py. sklearn, pandas, numpy, fredapi pre-installed on Kaggle |

---

## 7. Implementation Instructions

### builder_data Instructions

1. **Fetch DGS5 from FRED**: This is the only NEW data needed (DGS10 and DGS2 already in base_features)
   - Series: DGS5
   - Start: 2014-10-01 (warmup buffer)
   - End: 2025-02-12
2. **Also re-fetch DGS10 and DGS2** for the warmup period (base_features starts at 2015-01-30, we need from 2014-10-01)
3. **Save preprocessed data**: `data/processed/yield_curve_features_input.csv`
   - Columns: Date, DGS10, DGS2, DGS5, dgs10_change, dgs2_change, spread, curvature_raw, curvature_change
   - Start from 2014-10-01
4. **Quality checks**:
   - No gaps > 3 consecutive trading days in any series
   - Missing data < 5% for each series
   - All yields in reasonable range (0-8% for recent data)
   - DGS10 > DGS2 for most observations (not always -- inversions exist)
   - DGS5 between DGS2 and DGS10 for most observations (curvature_raw typically small magnitude)

### builder_model Instructions

#### train.py Structure

```python
"""
Gold Prediction SubModel Training - Yield Curve Attempt 1
Self-contained: Data fetch -> Preprocessing -> HMM + Velocity Z-Score + Curvature Z-Score -> Optuna HPO -> Save results
"""

# === 1. Libraries ===
import subprocess
subprocess.check_call(['pip', 'install', 'hmmlearn'])

import numpy as np
import pandas as pd
from fredapi import Fred
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mutual_info_score
import optuna
import json
import os
from datetime import datetime

# === 2. Data Fetching ===
# Fetch DGS10, DGS2, DGS5 from FRED using Kaggle Secret FRED_API_KEY
# Compute: dgs10_change, dgs2_change, spread, curvature_raw, curvature_change
# Align to base_features date range with warmup buffer

# === 3. Feature Generation Functions ===

def generate_regime_feature(dgs10_changes, dgs2_changes, n_components, covariance_type,
                            train_size, gold_returns_train):
    """
    Fit 2D HMM on [DGS10_change, DGS2_change] and return P(bearish-for-gold state) for full data.
    Multi-restart: 5 seeds, select best log-likelihood.
    """
    X_train = np.column_stack([dgs10_changes[:train_size], dgs2_changes[:train_size]])
    X_full = np.column_stack([dgs10_changes, dgs2_changes])

    # Remove NaN rows from training data
    valid_train = ~np.any(np.isnan(X_train), axis=1)
    X_train_clean = X_train[valid_train]

    best_model = None
    best_score = -np.inf
    seeds = [0, 42, 123, 456, 789]

    for seed in seeds:
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=100,
                tol=1e-4,
                random_state=seed
            )
            model.fit(X_train_clean)
            score = model.score(X_train_clean)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        return np.full(len(dgs10_changes), np.nan)

    # Handle NaN in full data for predict_proba
    valid_full = ~np.any(np.isnan(X_full), axis=1)
    probs = np.full((len(dgs10_changes), n_components), np.nan)
    if valid_full.sum() > 0:
        probs[valid_full] = best_model.predict_proba(X_full[valid_full])

    # Identify state most negatively correlated with gold returns
    # Compute mean gold return per state on training set
    states_train = best_model.predict(X_train_clean)
    gold_train_clean = gold_returns_train[valid_train[:len(gold_returns_train)]]

    state_gold_means = []
    for s in range(n_components):
        mask = states_train == s
        if mask.sum() > 0:
            state_gold_means.append(np.nanmean(gold_train_clean[mask[:len(gold_train_clean)]]))
        else:
            state_gold_means.append(0.0)

    target_state = np.argmin(state_gold_means)  # most bearish for gold
    return probs[:, target_state]

def generate_velocity_feature(spread, change_window, zscore_window):
    """
    Z-score of N-day spread change.
    """
    spread_change = spread.diff(change_window)
    rolling_mean = spread_change.rolling(zscore_window).mean()
    rolling_std = spread_change.rolling(zscore_window).std()
    z = (spread_change - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z

def generate_curvature_feature(curvature_raw, zscore_window):
    """
    Z-score of daily curvature change.
    curvature_raw = DGS5 - 0.5*(DGS2 + DGS10)
    """
    curvature_change = curvature_raw.diff()
    rolling_mean = curvature_change.rolling(zscore_window).mean()
    rolling_std = curvature_change.rolling(zscore_window).std()
    z = (curvature_change - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z

# === 4. Optuna Objective ===
# Maximize MI sum on validation set
# Parameters: hmm_n_components, hmm_covariance_type,
#             velocity_change_window, velocity_zscore_window, curvature_zscore_window

# === 5. Main ===
# Data split: train/val/test = 70/15/15 (time-series order)
# Run Optuna with 30 trials, 300s timeout
# Generate final output with best params
# Save submodel_output.csv, training_result.json
```

#### Key Implementation Notes

1. **hmmlearn installation**: Must `pip install hmmlearn` at the top of train.py. It is not pre-installed on Kaggle.

2. **FRED API key**: Use `os.environ['FRED_API_KEY']` from Kaggle Secrets. Fail with KeyError if not set (no fallback).

3. **No log-transform for HMM input**: Unlike VIX (right-skewed), yield changes are approximately symmetric. Raw daily changes are appropriate HMM input.

4. **Multi-restart HMM**: hmmlearn 0.3.3 does NOT have `n_init`. Implement manually: loop over seeds [0, 42, 123, 456, 789], fit each, select model with highest `model.score()` (log-likelihood).

5. **State labeling for regime_prob**: After HMM fitting, compute mean gold_return_next per state on training data. Output P(state with most negative mean gold return). This identifies the "bearish yield curve regime" -- the one most informative for gold prediction.

6. **No lookahead bias**:
   - HMM: Fit on training data only, generate probabilities for full dataset
   - Z-scores: Backward-looking rolling windows only
   - State labeling: Uses only training set gold returns

7. **NaN handling**: First ~60 rows (max of warmup periods) will have NaN. Forward-fill output after generation. The evaluator will align dates with base_features.

8. **Reproducibility**: Fix random_state seeds for HMM, seed=42 for Optuna. Z-score features are deterministic.

9. **Output format**: CSV with columns [Date, yc_regime_prob, yc_spread_velocity_z, yc_curvature_z]. Aligned to trading dates matching base_features.

10. **Gold target data**: Fetch gold price from Yahoo Finance (GC=F) and compute next-day return, or load from Kaggle dataset if available. Needed for HMM state labeling and Optuna MI evaluation.

11. **Fallback if 2D HMM fails**: If all HMM trials produce constant probabilities (std < 0.05), fall back to 1D HMM on spread_change = diff(DGS10 - DGS2). This is simpler and may produce better regime separation for the yield curve.

---

## 8. Risks and Alternatives

### Risk 1: 2D HMM Produces Poor Regime Separation

- **Description**: With DGS10 and DGS2 changes having 0.761 correlation, the 2D input may not offer enough dimensionality for clean regime detection. One state may dominate (>90% occupancy), similar to gold returns in the technical submodel.
- **Likelihood**: Moderate (40%)
- **Mitigation**: Optuna explores 2, 3, and 4 states. Even with imbalanced states, the minority state probabilities capture tail events. The technical submodel succeeded with 93%/7%/0.4% occupancy.
- **Detection**: If regime_prob std < 0.05 on validation set, the feature is uninformative.
- **Fallback**: Switch to 1D HMM on spread_change (DGS10-DGS2 daily diff). This directly models slope dynamics and has lower input dimensionality, possibly cleaner regime separation.

### Risk 2: Spread Velocity Z-Score High Autocorrelation

- **Description**: With 10-day change window, autocorrelation reaches 0.855. If Optuna selects 10d change + 120d z-score window, autocorrelation could approach 0.90+.
- **Likelihood**: Low (15%) -- all measured values are well below 0.99
- **Mitigation**: Even 0.914 (20-day, which is NOT in the search space) is below 0.99. The search space is limited to 3/5/10 day change windows, all below 0.86.
- **Detection**: Compute autocorrelation of best trial output. If > 0.95, reduce change window.
- **Fallback**: Use the first difference of the z-score (like cross_asset's recession_signal), reducing autocorrelation to near zero.

### Risk 3: VIF with Existing Submodel Outputs

- **Description**: The yield curve regime may correlate with VIX regime (both capture macro regimes) or cross_asset recession signal (yield curve inversion = recession signal).
- **Likelihood**: Low-Moderate (25%) -- yield curve captures rate dynamics, VIX captures equity vol, cross_asset captures commodity dynamics
- **Mitigation**: Empirically measured: all proposed features have near-zero correlation with base features. Cross-submodel correlation needs evaluator measurement but is expected low based on different input data domains.
- **Detection**: Gate 2 VIF check
- **Fallback**: Drop the most correlated feature and keep 2 columns

### Risk 4: Curvature Feature Adds Noise Without Signal

- **Description**: With autocorrelation of -0.153 and MI of 0.076, the curvature z-score is volatile and may add noise to the meta-model rather than useful signal.
- **Likelihood**: Low (20%) -- MI = 0.076 is comparable to VIX regime (0.079). The very low autocorrelation means each day's value is relatively independent, which is actually good for daily prediction.
- **Mitigation**: The meta-model (XGBoost) can assign low importance to noisy features via its feature selection mechanism.
- **Detection**: Feature importance < 1% in ablation test
- **Fallback**: Replace curvature with spread volatility: `yc_spread_vol_z = zscore(rolling_std(spread_change, 20), window=60)`. This captures turbulence instead of convexity.

### Risk 5: Gate 3 Failure (Information Exists But Doesn't Help Prediction)

- **Description**: Features pass Gate 2 (MI increase) but fail Gate 3 (ablation), like real_rate attempts 3-5.
- **Why yield_curve is different from real_rate**:
  - Daily frequency (no interpolation artifacts)
  - All features on changes/derivatives (not levels that create step functions)
  - 3 compact features (not 7 like real_rate attempt 5)
  - Direct economic link: yield curve changes directly affect gold opportunity cost
  - Measured VIF of 1.01 (real_rate had VIF 1.33-3.08, still passed, but noise was from interpolation not VIF)
- **Fallback for Attempt 2**: Fully deterministic approach (spread velocity + acceleration + volatility). No HMM. Reduces overfitting risk.

---

## 9. Empirical Validation Summary

All measurements performed on actual data (2015-01-30 to 2025-02-12):

### VIF Against Base Features (Measured)

| Feature | vs yield_curve_dgs10 | vs yield_curve_dgs2 | vs yield_curve_yield_spread | vs real_rate_real_rate | VIF |
|---------|---------------------|--------------------|-----------------------------|----------------------|-----|
| curvature_z | -0.002 | -0.000 | -0.003 | -0.003 | 1.01 |
| spread_velocity_z | 0.017 | 0.014 | -0.006 | 0.023 | 1.01 |

### Autocorrelation (Measured)

| Feature | Autocorr(lag=1) | Gate 1 Status |
|---------|----------------|---------------|
| curvature_z (60d window) | -0.153 | SAFE |
| spread_velocity_z (5d change, 60d window) | 0.749 | SAFE |
| spread_velocity_z (3d change) | 0.615 | SAFE |
| spread_velocity_z (10d change) | 0.855 | SAFE |
| spread level (reference) | 0.998 | WOULD FAIL |

### Mutual Information with gold_return_next (Measured)

| Feature | MI Score | Reference |
|---------|----------|-----------|
| curvature_z | 0.076 | VIX regime_prob MI = 0.079 |
| spread_velocity_z | 0.074 | Technical MI range = 0.079-0.089 |

### Curvature-Spread Correlation (Measured)

| Level | Correlation | Implication |
|-------|-------------|-------------|
| curvature_raw vs spread (levels) | 0.704 | High at level -- would cause VIF |
| curvature_change vs spread_change | 0.233 | Moderate at change level |
| curvature_z vs spread_level | -0.002 | Near zero -- z-score of change eliminates level correlation |
| curvature_z vs spread_velocity_z | 0.071 | Near zero -- the two output features are orthogonal |

---

## 10. Expected Performance Against Gates

### Gate 1: Standalone Quality

- **Overfit ratio**: N/A (deterministic HMM, no neural network)
- **No constant output**: Confirmed -- regime probability varies 0-1, z-scores vary with yield dynamics
- **Autocorrelation < 0.99**: curvature_z = -0.15, spread_velocity_z = 0.62-0.86. All SAFE.
- **No NaN values**: Confirmed after warmup with forward-fill

**Expected Result**: PASS

### Gate 2: Information Gain

- **MI increase > 5%**: MI = 0.076 + 0.074 = 0.150 from two deterministic features alone, plus HMM regime MI. Expected sum MI increase > 5%.
- **VIF < 10**: Measured VIF = 1.01 for both features. EXCELLENT.
- **Rolling correlation std < 0.15**: Likely marginal for regime_prob (VIX/technical/cross_asset all had stability ~0.15-0.21 for regime features, accepted as precedent).

**Expected Result**: PASS (or marginal on stability for regime_prob, which is accepted precedent)

### Gate 3: Ablation Test

- **Direction accuracy +0.5%**: Moderate probability -- yield curve dynamics provide context for interpreting spread levels already in base features
- **OR Sharpe +0.05**: Moderate probability
- **OR MAE -0.01%**: Higher probability -- technical achieved MAE -0.18 (18x threshold), cross_asset achieved MAE -0.087 (8.7x)

**Expected Result**: PASS (most likely via MAE, following technical/cross_asset pattern)

**Overall Confidence**: 70% (moderate-high)

---

## 11. Design Rationale

### Why 2D HMM Over 1D?

The 1D HMM on spread changes captures only slope velocity (steepening vs flattening). The 2D HMM on [DGS10_change, DGS2_change] additionally captures the MECHANISM: parallel shift (both yields move together) vs twist (one moves more than the other). A 50bp flattening from DGS10 falling -50bp is fundamentally different from DGS2 rising +50bp, even though the spread change is identical. The 2D HMM can distinguish these.

### Why DGS5 Curvature Over DGS30?

Both are available daily from FRED. DGS5 belly curvature is preferred because:
1. DGS5 is the most-traded maturity point, with highest liquidity
2. The 2Y-5Y-10Y butterfly captures the "belly" of the curve, which is most sensitive to Fed policy expectations
3. DGS30 adds a longer maturity dimension that may introduce duration risk correlation with DGS10 (less orthogonal)
4. Measured MI(curvature_z, gold_return) = 0.076, confirming information content

### Why Not Mean-Reversion Feature?

Unlike VIX (which has a stable long-term mean of ~18-20), the yield spread does NOT have a stable mean. Pre-GFC average was ~150-200bp, post-GFC ~100-150bp, 2022-2024 was negative. A z-score of the spread LEVEL would (a) correlate with the spread base feature (VIF issue) and (b) assume a fixed mean that does not exist. The velocity and curvature features capture DYNAMICS without assuming a fixed reversion target.

### Why Z-Score of CHANGES, Not Z-Score of Levels?

The spread level has autocorrelation 0.998 (near unit root). A z-score of the level would inherit this persistence and fail Gate 1. Z-score of CHANGES (first differences) removes the unit root, producing autocorrelation of 0.62-0.86 (depending on change window). This is the core lesson from real_rate failures: never use level-based features for persistent time series.
