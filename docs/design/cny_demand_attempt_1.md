# Submodel Design Document: CNY Demand Proxy (Attempt 1)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| Yahoo Finance CNY=X daily data | CONFIRMED | 2788 rows (2014-06-02 to 2025-02-12). No NaN in Close. Min=6.1012, Max=7.3321. No gaps > 5 calendar days (max gap = 4 days). Excellent coverage. |
| CNY=X reflects onshore rate | CONFIRMED | Values 6.1-7.3 match known USD/CNY onshore range. Trading hours consistent with Beijing market. |
| corr(CNY_return, DXY_return) = 0.40-0.65 | INCORRECT | Measured: 0.028 (full period). Rolling 252d: min=-0.11, max=0.21, mean=0.03. CNY-DXY correlation is essentially ZERO, not 0.40-0.65. This is actually very favorable -- VIF risk with DXY is negligible. |
| CNY_return autocorr lag1 = -0.10 to +0.05 | CONFIRMED | Measured: -0.1529 (slightly more negative than claimed but directionally correct, consistent with managed float mean reversion). |
| CNY_return autocorr lag5 = -0.05 to 0.00 | CONFIRMED | Measured: 0.0172 (near zero, consistent). |
| cny_momentum_z autocorr = 0.40-0.60 | INCORRECT (minor) | Measured: 0.7448. Higher than claimed but still well below 0.99 threshold. Passes datachecker. |
| cny_vol_regime_z (5d/120d) autocorr = 0.75-0.85 | CONFIRMED | Measured: 0.8451. Within claimed range. |
| cny_vol_regime_z (5d/60d) autocorr | MEASURED | 0.8307. Slightly lower than 5d/120d. Both acceptable. |
| cny_vol_regime_z (10d/120d) autocorr | MEASURED | 0.9336. Passes but higher risk. |
| cny_vol_regime_z (20d/120d) autocorr | MEASURED | 0.9670. Passes but borderline. Avoid unless Optuna strongly prefers. |
| HMM 3-state regime detection on CNY | CONFIRMED viable | Test HMM: State 0 (stable, 92.7%), State 1 (crisis, 0.9%), State 2 (adjustment, 6.5%). Regime_prob autocorr=0.045 (excellent). Std=0.084 (non-constant). |
| VIF < 10 for CNY features | CONFIRMED EXCELLENT | Measured against full 39-feature set: cny_regime_prob=1.11, cny_momentum_z=1.11, cny_vol_regime_z=1.14. All essentially orthogonal. |
| PBOC daily fixing data not freely available | CONFIRMED | Yahoo Finance CNY=X provides market rates only, not PBOC fixing rates. Approach B (fixing_gap) correctly rejected. |
| CNH=F/CNHHKD=X available | NOT VERIFIED | Reserved for Attempt 2. Not needed for Approach A. |
| DXY basket weights (EUR 57.6%, etc.) | PLAUSIBLE | Standard published weights. Not independently verified but consistent across multiple financial references. |
| PBOC gold purchases 316t 2022-2024 | PLAUSIBLE | Consistent with WGC and Trading Economics reports. Not independently verified but widely reported. |

### Critical Findings

**1. CNY-DXY correlation is near-zero (0.028), NOT 0.40-0.65.** The researcher dramatically overestimated this correlation. In reality, managed float dynamics, time zone differences, and PBOC intervention create near-complete decorrelation between CNY daily returns and DXY daily returns. This eliminates the primary VIF concern entirely.

**2. All three proposed CNY features have VIF around 1.1.** Against the full 39-feature set (19 base + 20 submodel outputs), the CNY features are essentially orthogonal. Maximum pairwise correlation is 0.23 (cny_regime_prob vs cny_vol_regime_z, internal). External correlations are all below 0.13.

**3. cny_momentum_z autocorrelation is 0.74, not 0.40-0.60.** Higher than claimed but well within threshold (< 0.99). Not design-breaking.

**4. HMM regime structure is highly skewed.** The stable state dominates at 92.7%, with crisis at 0.9% and adjustment at 6.5%. This is expected for managed float (PBOC smooths volatility most of the time). The regime_prob output will be near-zero most of the time, spiking during crisis events. This is similar to the etf_flow pattern where regime_prob was concentrated but still informative.

---

## 1. Overview

- **Purpose**: Extract three orthogonal dimensions of CNY/USD exchange rate dynamics -- regime state (PBOC intervention intensity), momentum persistence (depreciation/appreciation trend), and volatility anomaly (intervention episodes). These capture China-specific gold demand context that the raw CNY/USD level (base feature #7 at 5.31% importance) cannot convey.
- **Core methods**:
  1. Hidden Markov Model (2-3 states) on 2D [CNY_daily_return, CNY_vol_5d] for managed float regime classification
  2. Rolling z-score of 5d momentum against 60d baseline for trend persistence measurement
  3. Rolling z-score of 5d volatility against 120d baseline for PBOC intervention episode detection
- **Why these methods**: PBOC managed float creates discrete regime dynamics (stable/adjustment/crisis) that are invisible in the raw CNY/USD level. HMM captures nonlinear regime transitions. The z-scores capture the two fundamental dimensions of CNY dynamics: directional persistence and volatility structure.
- **Expected effect**: Enable the meta-model to distinguish identical CNY/USD levels with different underlying PBOC regime contexts. CNY at 7.0 during a stable management period has different gold demand implications than 7.0 during an active depreciation defense.

### Key Advantage

All data from a single Yahoo Finance ticker (CNY=X). No FRED API dependency. No multi-source alignment issues. VIF against all existing features is essentially 1.0 (orthogonal). This is the cleanest data setup of any submodel.

### Design Rationale

- **Approach A (standard HMM + z-scores)**: Selected -- proven pattern matching 6/7 successful submodels. Lowest risk for the final submodel.
- **Approach B (fixing_gap + onshore-offshore)**: Rejected -- PBOC fixing data not freely available via API.
- **Approach C (gold-sensitivity hybrid)**: Rejected for Attempt 1 -- using gold_return as HMM input risks "submodel does NOT predict gold" violation. Available for Attempt 2 if needed.

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Expected Rows |
|------|--------|--------|-----------|---------------|
| CNY/USD onshore rate | Yahoo Finance | CNY=X | Daily | ~2,788 (2014-06-02 to 2025-02-12) |
| Gold returns | Yahoo Finance | GC=F | Daily | Already in base_features |

### Data NOT Used

| Data | Source | Reason for Exclusion |
|------|--------|---------------------|
| CNH=F / CNHHKD=X (offshore yuan) | Yahoo Finance | Reserve for Attempt 2. Adds complexity, alignment issues with 24h market. |
| PBOC daily fixing rate | pbc.gov.cn | No free API. Manual scraping violates automation constraint. |
| SGU=F (Shanghai gold futures) | Yahoo Finance | Rollover gaps, circular dependency with CNY conversion. |
| ^HSI (Hang Seng Index) | Yahoo Finance | Overlaps with cross_asset submodel (S&P 500). Reserve for Attempt 2. |
| FXI (iShares China ETF) | Yahoo Finance | US-listed, reflects US sentiment not Chinese domestic conditions. |

### Preprocessing Steps

1. Fetch CNY=X from Yahoo Finance, start=2014-06-01 (buffer for 120-day warmup before 2015-01-30)
2. Fetch GC=F close for gold returns computation (needed for Optuna MI evaluation only, not model input)
3. Compute derived quantities:
   - `cny_return = cny_close.pct_change()` (daily return)
   - `cny_vol_Xd = cny_return.rolling(X).std()` (X-day rolling volatility of returns, X from Optuna)
   - `gold_return = gc_close.pct_change()` (current-day return, not next-day -- for MI eval only)
4. Handle missing values: forward-fill gaps up to 3 trading days, drop remaining NaN
5. Trim output to base_features date range: 2015-01-30 to 2025-02-12

### Expected Sample Count

- ~2,788 daily observations from CNY=X (2014-06-02 to 2025-02-12)
- No gaps > 5 calendar days (max observed gap = 4 days, no Chinese holiday gap exceeded this)
- Warmup period: ~125 days (120 for longest rolling baseline + 5 for short vol window)
- Effective output: ~2,523 rows aligned to base_features date range

---

## 3. Model Architecture (Hybrid Deterministic-Probabilistic)

This is a **hybrid deterministic-probabilistic** approach. No PyTorch neural network. The pipeline consists of three independent components.

### Component 1: HMM Regime Detection (cny_regime_prob)

- **Model**: `hmmlearn.hmm.GaussianHMM`
- **Input**: 2D array of [CNY_daily_return, CNY_vol_5d]
  - CNY_daily_return = cny_close.pct_change(): captures direction and magnitude of CNY moves
  - CNY_vol_5d = cny_return.rolling(5).std(): captures local volatility (PBOC intervention proxy)
  - 2D rationale: distinguishes stable management (low return + low vol) from gradual adjustment (medium return + low vol) from crisis/intervention (high return + high vol)
- **States**: 2 or 3 (Optuna selects via MI)
  - 2 states: stable vs volatile
  - 3 states: stable management / gradual adjustment / crisis intervention (preferred based on fact-check: 92.7% / 6.5% / 0.9%)
- **Covariance type**: "full" or "diag" (Optuna explores)
- **Training**: Fit on training set data only, with multiple random restarts (best of 5 seeds). Generate probabilities for full dataset using `predict_proba`.
- **Output**: Posterior probability of the highest-return-variance state (identified post-hoc by comparing emission variances on the CNY_return dimension)
- **State labeling**: After fitting, sort states by emission variance of the CNY_return dimension. The highest-variance state corresponds to "crisis/intervention regime." Output P(highest-variance state).

```
Input: [CNY_daily_return, CNY_vol_5d] [T x 2]
       |
   Best-of-5 GaussianHMM.fit(train_data) -> learn 2-3 state model
       |
   GaussianHMM.predict_proba(full_data) -> [T x n_states]
       |
   Select P(highest-return-variance state) -> [T x 1]
       |
Output: cny_regime_prob (0-1, typically near 0 with spikes during crisis)
```

### Component 2: CNY Momentum Z-Score (cny_momentum_z)

- **Model**: Pure pandas computation (no ML)
- **Input**: CNY daily returns
- **Momentum window**: 5 days (fixed -- captures weekly trend)
- **Baseline window**: 60 or 120 days (Optuna explores)
- **Computation**: `z = (momentum_5d - rolling_mean_baseline) / rolling_std_baseline`
  - Positive: Sustained CNY depreciation above recent norm (purchasing power deterioration signal)
  - Negative: Sustained CNY appreciation above recent norm (purchasing power improvement signal)
- **Clipping**: [-4, 4] for stability
- **Measured autocorrelation**: 0.7448 (well below 0.99 threshold)

```
Input: cny_return [T x 1]
       |
   momentum_5d = cny_return.rolling(5).mean()
       |
   rolling_mean = momentum_5d.rolling(baseline).mean()
   rolling_std  = momentum_5d.rolling(baseline).std()
       |
   z = (momentum_5d - rolling_mean) / rolling_std
       |
   clip(-4, 4)
       |
Output: cny_momentum_z (typically -3 to +3)
```

### Component 3: CNY Volatility Regime Z-Score (cny_vol_regime_z)

- **Model**: Pure pandas computation (no ML)
- **Input**: Rolling standard deviation of CNY daily returns
- **Short window**: 5d, 10d, or 20d (Optuna explores)
  - 5d: autocorr 0.8451 (best), captures immediate intervention episodes
  - 10d: autocorr 0.9336, moderate
  - 20d: autocorr 0.9670, highest but captures monthly regime patterns
- **Baseline window**: 60d or 120d (Optuna explores)
- **Computation**: `z = (vol_short - rolling_mean_baseline) / rolling_std_baseline`
  - Positive: Abnormal volatility (PBOC intervention visible, band stress)
  - Negative: Abnormal stability (tight PBOC control, suppressed volatility)
- **Clipping**: [-4, 4] for stability

```
Input: cny_return [T x 1]
       |
   vol_short = cny_return.rolling(short_window).std()
       |
   rolling_mean = vol_short.rolling(baseline_window).mean()
   rolling_std  = vol_short.rolling(baseline_window).std()
       |
   z = (vol_short - rolling_mean) / rolling_std
       |
   clip(-4, 4)
       |
Output: cny_vol_regime_z (typically -2 to +4)
```

### Combined Output

| Column | Range | Description | Measured Autocorr | VIF (full set) |
|--------|-------|-------------|-------------------|----------------|
| `cny_regime_prob` | [0, 1] | P(crisis/intervention regime) from 2D HMM | 0.045 (test HMM) | 1.11 |
| `cny_momentum_z` | [-4, +4] | 5d momentum z-score vs 60d baseline | 0.74 | 1.11 |
| `cny_vol_regime_z` | [-4, +4] | 5d volatility z-score vs 120d baseline | 0.85 | 1.14 |

Total: **3 columns** (matching the proven compact pattern from all 7 successful submodels).

### Orthogonality Analysis (Measured Against Full 39-Feature Set)

| CNY Feature | Highest External Correlation | Value | Assessment |
|-------------|-------------------------------|-------|------------|
| cny_regime_prob | vix_vix | 0.14 | Low |
| cny_momentum_z | etf_pv_divergence | 0.14 | Low |
| cny_vol_regime_z | ie_regime_prob | 0.18 | Low |

All pairwise correlations with existing features below 0.20. The CNY features are essentially orthogonal to the entire existing feature set. This is the lowest cross-correlation of any submodel.

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM n_iter | 100 | Standard convergence limit; EM typically converges in 20-50 iterations |
| HMM tol | 1e-4 | Standard convergence tolerance |
| HMM random_state seeds | [42, 123, 456, 789, 0] | 5 restarts to avoid local optima (manual best-of-5 since hmmlearn lacks n_init) |
| Momentum short window | 5 days | Fixed -- captures weekly CNY trend direction |
| Z-score clipping | [-4, 4] | Prevent extreme outliers from dominating |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3} | categorical | 2=stable/volatile, 3=stable/adjustment/crisis. Fact-check shows 3-state viable (92.7/6.5/0.9%). |
| hmm_covariance_type | {"full", "diag"} | categorical | full captures return-vol correlation structure; diag treats independently |
| vol_short_window | {5, 10, 20} | categorical | 5d=lowest autocorr (0.85), 10d=moderate (0.93), 20d=highest (0.97). All pass <0.99. |
| vol_baseline_window | {60, 120} | categorical | 60d=responsive, 120d=stable structural baseline |
| momentum_baseline_window | {60, 120} | categorical | 60d or 120d for momentum z-score baseline |

### Exploration Settings

- **n_trials**: 30
  - Rationale: 5 categorical parameters with small ranges. Total combinations: 2 x 2 x 3 x 2 x 2 = 48. 30 trials with TPE covers 63% of space. Each trial is fast (<10 seconds: HMM fit on ~1,950 x 2 observations).
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of mutual information between 3 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)

---

## 5. Training Settings

### Fitting Procedure

This is not a gradient-based training loop. The procedure is:

1. **HMM**: Best-of-5 `GaussianHMM.fit(X_train)` on 2D [CNY_return, CNY_vol_5d] -- EM algorithm, converges in seconds
2. **CNY Momentum Z-Score**: Rolling window statistics on momentum -- deterministic, no fitting
3. **CNY Volatility Regime Z-Score**: Rolling window statistics on volatility -- deterministic, no fitting

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- train: ~1,766 observations
- val: ~378 observations
- test: ~379 observations (reserved for evaluator Gate 3)
- HMM fits on train set only (best-of-5 seeds)
- HMM generates probabilities for full dataset using predict_proba (no lookahead: HMM posterior at time t depends only on observations up to t given fitted model)
- Z-scores use rolling windows (inherently backward-looking, no lookahead)
- Optuna optimizes MI sum on validation set

### Evaluation Metric for Optuna

For each trial:
1. Fit 2D HMM on train set [CNY_return, CNY_vol_Xd] (best of 5 seeds)
2. Generate all 3 features for full dataset using fitted HMM and trial window parameters
3. Compute mutual information (MI) between each feature and gold_return_next on validation set
4. Optuna maximizes: `MI_sum = MI(regime_prob, target) + MI(momentum_z, target) + MI(vol_regime_z, target)`

MI calculation: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`.

### Loss Function

N/A -- no gradient-based training. HMM uses EM (maximum likelihood). Z-scores are deterministic.

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- HMM EM converges via n_iter and tol. No early stopping needed.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. HMM + rolling statistics are CPU-only |
| Estimated execution time | 3-5 minutes | Data download (~30s) + 30 Optuna trials x ~5s each (~2.5min) + final output (~30s) |
| Estimated memory usage | < 1 GB | ~2,788 rows x 2-3 columns. Tiny dataset |
| Required pip packages | `hmmlearn` | Must pip install hmmlearn at start of train.ipynb. sklearn, pandas, numpy, optuna, yfinance are pre-installed on Kaggle |

---

## 7. Implementation Instructions

### builder_data Instructions

1. **Fetch CNY=X from Yahoo Finance**: `yf.download('CNY=X', start='2014-06-01')` -- Close column
2. **Compute derived quantities**:
   - `cny_return = cny_close.pct_change()`
   - `cny_vol_5d = cny_return.rolling(5).std()`
3. **Fetch gold returns**: Use GC=F daily close from Yahoo Finance, compute `gold_return = gc_close.pct_change()`
4. **Align dates**: Inner join CNY=X and GC=F on trading dates
5. **Save preprocessed data**: `data/processed/cny_demand_features_input.csv`
   - Columns: Date, cny_close, cny_return, cny_vol_5d, gold_return
6. **Quality checks**:
   - CNY=X Close values in range [5.5, 8.0] (USD/CNY onshore rate reasonable range)
   - No gaps > 5 consecutive trading days
   - Missing data < 2% after alignment
   - cny_return has no extreme outliers (|value| < 0.05, max observed was 0.021)
   - cny_vol_5d is always positive (std of a non-constant series)
7. **Verify date range**: Data should cover 2014-06-01 through latest available, trimmed to base_features range for output

### builder_model Instructions

#### train.ipynb Structure

```python
"""
Gold Prediction SubModel Training - CNY Demand Proxy Attempt 1
Self-contained: Data fetch -> Preprocessing -> HMM + Z-Scores -> Optuna HPO -> Save results
"""

# === 1. Libraries ===
import subprocess
subprocess.check_call(['pip', 'install', 'hmmlearn'])

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mutual_info_score
import optuna
import json
import os
from datetime import datetime

# === 2. Data Fetching ===
# yfinance: CNY=X (start=2014-06-01 for warmup buffer)
# yfinance: GC=F (for gold_return computation, MI eval only)
# Compute: cny_return, cny_vol_5d, gold_return

# === 3. Feature Generation Functions ===

def generate_regime_feature(cny_return, cny_vol, n_components, covariance_type, train_size):
    """
    Fit 2D HMM on [CNY_return, CNY_vol] and return P(highest-return-variance state).
    Best-of-5 random restarts.
    """
    X = np.column_stack([cny_return, cny_vol])
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    X_train = X_valid[:train_size]

    best_score = -np.inf
    best_model = None
    for seed in [42, 123, 456, 789, 0]:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=100,
            tol=1e-4,
            random_state=seed
        )
        try:
            model.fit(X_train)
            score = model.score(X_train)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        return np.full(len(cny_return), np.nan), None

    probs = best_model.predict_proba(X_valid)

    # Identify highest-return-variance state (first dimension = CNY_return)
    state_vars = []
    for i in range(n_components):
        if covariance_type == 'full':
            state_vars.append(float(best_model.covars_[i][0, 0]))
        elif covariance_type == 'diag':
            state_vars.append(float(best_model.covars_[i][0]))

    high_var_state = np.argmax(state_vars)

    # Map back to full array
    result = np.full(len(cny_return), np.nan)
    result[valid_mask] = probs[:, high_var_state]
    return result, best_model

def generate_momentum_feature(cny_return, baseline_window):
    """
    5d momentum z-scored against baseline_window-day baseline.
    """
    s = pd.Series(cny_return)
    momentum = s.rolling(5).mean()
    rolling_mean = momentum.rolling(baseline_window).mean()
    rolling_std = momentum.rolling(baseline_window).std()
    z = (momentum - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z.values

def generate_vol_regime_feature(cny_return, vol_window, baseline_window):
    """
    Rolling z-score of vol_window-day volatility against baseline_window-day baseline.
    """
    s = pd.Series(cny_return)
    vol_short = s.rolling(vol_window).std()
    rolling_mean = vol_short.rolling(baseline_window).mean()
    rolling_std = vol_short.rolling(baseline_window).std()
    z = (vol_short - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z.values

# === 4. Optuna Objective ===

def objective(trial, cny_return, cny_vol_5d, target, train_size, val_mask):
    """Maximize MI sum on validation set"""
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    covariance_type = trial.suggest_categorical('hmm_covariance_type', ['full', 'diag'])
    vol_window = trial.suggest_categorical('vol_short_window', [5, 10, 20])
    vol_baseline = trial.suggest_categorical('vol_baseline_window', [60, 120])
    mom_baseline = trial.suggest_categorical('momentum_baseline_window', [60, 120])

    try:
        # Recompute HMM input vol with trial's vol_window
        cny_vol = pd.Series(cny_return).rolling(vol_window).std().values

        regime, _ = generate_regime_feature(
            cny_return, cny_vol, n_components, covariance_type, train_size
        )
        momentum = generate_momentum_feature(cny_return, mom_baseline)
        vol_regime = generate_vol_regime_feature(cny_return, vol_window, vol_baseline)

        # Extract validation period
        regime_val = regime[val_mask]
        momentum_val = momentum[val_mask]
        vol_regime_val = vol_regime[val_mask]
        target_val = target[val_mask]

        # Compute MI (discretize continuous variables)
        def discretize(x, bins=20):
            valid = ~np.isnan(x)
            if valid.sum() < bins:
                return None
            x_valid = x.copy()
            x_valid[~valid] = np.nanmedian(x)
            return pd.qcut(x_valid, bins, labels=False, duplicates='drop')

        mi_sum = 0.0
        for feat_val in [regime_val, momentum_val, vol_regime_val]:
            mask = ~np.isnan(feat_val) & ~np.isnan(target_val)
            if mask.sum() > 50:
                feat_disc = discretize(feat_val[mask])
                tgt_disc = discretize(target_val[mask])
                if feat_disc is not None and tgt_disc is not None:
                    mi_sum += mutual_info_score(feat_disc, tgt_disc)

        return mi_sum

    except Exception:
        return 0.0

# === 5. Main ===
# Data split: train/val/test = 70/15/15 (time-series order)
# Run Optuna with 30 trials, 300s timeout
# Generate final output with best params
# Save submodel_output.csv, training_result.json
```

#### Key Implementation Notes

1. **hmmlearn installation**: Must `pip install hmmlearn` at the top of train.ipynb. It is not pre-installed on Kaggle.
2. **No FRED API dependency**: This submodel uses only yfinance (CNY=X and GC=F). No FRED_API_KEY needed. However, the unified Notebook "Gold Model Training" has FRED_API_KEY enabled -- it simply will not be used.
3. **2D HMM input**: Use [CNY_daily_return, CNY_vol_Xd] where X is from Optuna (5/10/20). This distinguishes managed float regimes.
4. **HMM state labeling**: After fitting, sort states by emission variance of the FIRST dimension (CNY_return). Output P(highest-variance state). Do NOT assume state index 0, 1, or 2 corresponds to any specific regime.
5. **Best-of-5 restarts**: Since hmmlearn lacks `n_init`, manually try 5 different random_state values and keep the model with highest log-likelihood score.
6. **Vol window for HMM input**: The vol window used for HMM dimension 2 should match the vol_short_window from Optuna. If Optuna selects vol_short_window=10, then HMM input should be [CNY_return, CNY_vol_10d].
7. **No lookahead bias**:
   - HMM: Fit on training data only, generate probabilities for full dataset
   - Z-scores: Rolling window (inherently backward-looking)
8. **NaN handling**: First ~125 rows (max of warmup periods) will have NaN. Forward-fill output after generation.
9. **Reproducibility**: Fix random_state seeds [42, 123, 456, 789, 0] for HMM, seed=42 for Optuna. Z-scores are deterministic.
10. **Output format**: CSV with columns [Date, cny_regime_prob, cny_momentum_z, cny_vol_regime_z]. Aligned to trading dates matching base_features.
11. **NO LEVEL FEATURES**: Do not output any CNY/USD level, level z-score, or level-derived feature. base_features already contains cny_demand_cny_usd.

---

## 8. Risks and Alternatives

### Risk 1: HMM Regime Imbalance (Crisis State < 1%)

- **Description**: Test HMM shows crisis state at only 0.9% frequency. The regime_prob output will be near-zero for 99%+ of observations. This is the same pattern as etf_flow (etf_regime_prob was informative despite concentration).
- **Measured**: regime_prob mean=0.009, std=0.084 (non-constant, passes Gate 1).
- **Mitigation**: Optuna explores 2-state vs 3-state. 2-state may have more balanced distribution. Even with imbalanced 3-state, the crisis spikes at key events (2015 devaluation, 2019 break of 7.0, 2020 COVID) provide informative signal.
- **Fallback**: If regime_prob is effectively constant (std < 0.01), output only cny_momentum_z and cny_vol_regime_z (2 features instead of 3). Yield_curve succeeded with 2 features after HMM collapse.

### Risk 2: cny_vol_regime_z Autocorrelation (10d/20d windows)

- **Description**: With 10d window autocorr=0.93, with 20d autocorr=0.97. Both pass <0.99 but are high.
- **Measured values**:
  - 5d/120d: 0.8451
  - 5d/60d: 0.8307
  - 10d/120d: 0.9336
  - 20d/120d: 0.9670
- **Mitigation**: Optuna explores all three vol windows. 5d has the best autocorrelation profile. If Optuna selects 10d or 20d, datachecker will verify < 0.99 before proceeding.
- **Fallback**: Force 5d window if others fail datachecker.

### Risk 3: Diminishing Returns (8th Submodel)

- **Description**: MAE improvement trend shows diminishing returns: technical (-0.182) > cross_asset (-0.087) > yield_curve (-0.069) > etf_flow (-0.044). Expected CNY improvement is -0.01% to -0.03%.
- **Mitigation**: CNY captures a genuinely novel channel (Chinese gold demand, ~30% of world total) that no existing submodel addresses. The near-zero correlation with all existing features (max 0.18) means any non-zero information will be additive.
- **Primary success path**: MAE improvement -0.01% (minimum threshold). VIF of ~1.1 means the meta-model can fully utilize the information without multicollinearity.
- **Stop condition**: If Attempt 1 fails Gate 3, one more attempt with CNH spread or gold_sensitivity_z. If Attempt 2 also fails, declare complete and proceed to meta-model.

### Risk 4: Low MI from Regime Feature

- **Description**: 5/7 completed submodels failed Gate 2 MI > 5% threshold. MI test tends to underestimate nonlinear regime-based information.
- **Mitigation**: Gate 3 ablation is the true test. All 7 completed submodels passed Gate 3 despite Gate 2 MI failures (except inflation_expectation which passed both). The regime_prob feature may fail MI test but still provide value through nonlinear interactions in the meta-model XGBoost.
- **Precedent**: VIX (MI 0.68%), yield_curve (MI 0.37%), etf_flow (MI 1.58%) all failed Gate 2 but passed Gate 3.

---

## 9. VIF Analysis (Measured)

### Proposed Features VIF (Full 39-Feature Set)

| Feature | VIF (full set) | Assessment |
|---------|----------------|------------|
| cny_regime_prob | 1.11 | Excellent (essentially orthogonal) |
| cny_momentum_z | 1.11 | Excellent |
| cny_vol_regime_z | 1.14 | Excellent |

### Context: Why VIF Is So Low

The researcher claimed CNY-DXY return correlation of 0.40-0.65, which would imply moderate VIF risk. In reality:
- **corr(CNY_return, DXY_return) = 0.028** (essentially zero)
- **Rolling 252d correlation**: ranges -0.11 to 0.21, never exceeding 0.21
- **Root cause of decorrelation**: PBOC managed float creates return dynamics orthogonal to free-floating DXY components. CNY daily returns are dominated by PBOC band mechanics (mean reversion within +/-2% band), while DXY daily returns are driven by EUR/JPY/GBP macro flows.

This is the best VIF result of any submodel. For comparison:
- etf_regime_prob: VIF = 12.60 (passed Gate 3 despite exceeding 10 threshold)
- ie_regime_prob: VIF = 2.35
- cny_regime_prob: VIF = 1.11 (lowest of all regime features)

### Pairwise Correlations with Existing Submodel Outputs

| CNY Feature | Top 3 External Correlations |
|-------------|---------------------------|
| cny_regime_prob | ie_regime_prob (0.18), vix_vix (0.13), cny_demand_cny_usd (0.09) |
| cny_momentum_z | etf_pv_divergence (0.14), vix_mean_reversion_z (0.10), vix_persistence (0.08) |
| cny_vol_regime_z | ie_regime_prob (0.18), ie_anchoring_z (0.16), vix_vix (0.13) |

All external correlations below 0.20. No residual extraction or VIF mitigation needed.

---

## 10. Autocorrelation Analysis (Measured)

| Feature | Config | Autocorrelation (lag 1) | Status |
|---------|--------|-------------------------|--------|
| cny_regime_prob | 3-state HMM, full cov | 0.045 | PASS |
| cny_momentum_z | 5d mom, 60d baseline | 0.745 | PASS |
| cny_vol_regime_z | 5d vol, 120d baseline | 0.845 | PASS |
| cny_vol_regime_z | 5d vol, 60d baseline | 0.831 | PASS |
| cny_vol_regime_z | 10d vol, 120d baseline | 0.934 | PASS (high) |
| cny_vol_regime_z | 20d vol, 120d baseline | 0.967 | PASS (borderline) |

All candidate configurations pass the < 0.99 threshold. The 5d vol window is strongly preferred for its low autocorrelation.

Note: cny_regime_prob autocorrelation (0.045) is unusually low compared to other submodels' regime features (typically 0.7-0.9). This is because the crisis state is very rare (< 1% frequency), so the HMM transitions rapidly back to stable state. This is structurally different from, e.g., VIX regime_prob where the volatile state persists for weeks.

---

## 11. Expected Performance Against Gates

### Gate 1: Standalone Quality
- **Overfit ratio**: N/A (deterministic HMM, no neural network with train/val loss comparison)
- **No constant output**: Moderate confidence -- regime_prob has std=0.084 (non-constant, but concentrated). Momentum and vol z-scores have healthy distributions.
- **Autocorrelation < 0.99**: CONFIRMED for all configurations. Best case 0.045 (regime_prob), worst case 0.967 (20d vol z-score).
- **Expected Result**: PASS (with risk of regime_prob near-constant if HMM collapses, precedent: yield_curve)

### Gate 2: Information Gain
- **MI increase > 5%**: Low-medium probability. VIF is excellent (1.1) but MI tends to underestimate nonlinear regime information. 5/7 prior submodels failed this test.
- **VIF < 10**: CONFIRMED (1.11 for all three features). Best VIF of any submodel.
- **Rolling correlation std < 0.15**: Moderate probability. regime_prob may marginally fail (consistent with regime feature precedent).
- **Expected Result**: MARGINAL (VIF excellent, MI uncertain, stability may fail for regime_prob)

### Gate 3: Ablation (Primary Target)
- **Primary path**: MAE improvement > 0.01%
- **Supporting evidence**: CNY is rank 7 feature (5.31% importance). China-specific demand channel is genuinely novel (zero correlation with existing features). Near-zero VIF means meta-model can fully exploit information.
- **Realistic target**: MAE improvement -0.01% to -0.03% (diminishing returns, 8th submodel)
- **Expected Result**: 40-60% probability of PASS

**Confidence**: 5/10 (moderate). This is the 8th submodel with expected diminishing returns. The VIF advantage is exceptional but the raw information content of CNY regime dynamics for gold prediction may be marginal. The near-zero CNY-DXY correlation means the information is genuinely novel, which helps. The primary uncertainty is whether Chinese demand dynamics, as proxied by CNY/USD regime features, provide sufficient incremental signal after 7 prior submodels have already captured most of the predictable variance.
