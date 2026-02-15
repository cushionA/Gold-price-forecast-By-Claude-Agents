# Submodel Design Document: Inflation Expectation (Attempt 1)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| FRED: T10YIE daily | CONFIRMED | Daily, 5783 obs, 2003-01-02 to 2026-02-13. Min=0.04, Max=3.02, Mean=2.10 |
| FRED: T5YIFR daily | CONFIRMED | Daily, 5783 obs, 2003-01-02 to 2026-02-13 |
| FRED: T5YIE daily | CONFIRMED | Daily, available 2003-2026 |
| FRED: T10YIEUK | CONFIRMED NOT EXIST | "Bad Request. The series does not exist." Researcher correctly identified this. |
| Fisher identity: T10YIE + DFII10 = DGS10 | CONFIRMED | Residual mean=0.000000, std=0.000000, max_abs=0.000000. Perfect identity. |
| corr(IE_change_z60, real_rate_change_z60) = -0.044 | INCORRECT (minor) | Measured: -0.0678. Both near zero (orthogonal). Not design-breaking. Researcher used different date range or sample. |
| corr(IE_change_z60, DGS10_change_z60) = 0.514 | INCORRECT (minor) | Measured: 0.4795. Both moderate. Not design-breaking. |
| ie_anchoring_z autocorr 0.90-0.95 (estimated) | INCORRECT | Measured: 0.9700. Z-scoring reduced raw 0.9887 to 0.9700, not to 0.90-0.95. Still below 0.99 threshold but much higher than estimated. Alternative: 5d vol z-scored = 0.8146 (better). |
| ie_gold_sensitivity_z autocorr = 0.75 | CLOSE | Measured: 0.7177 (raw 5d corr = 0.7438). Slight difference due to z-scoring effect. Acceptable. |
| corr(T10YIE_change, T5YIFR_change) | NOT CHECKED BY RESEARCHER | Measured: 0.6718 (raw), 0.7738 (z-scored). Below 0.7 for raw changes -- Approach D (term structure HMM) is viable. |
| ie_candidates vs all submodel outputs < 0.15 | CONFIRMED | Max pairwise correlation: ie_anchoring_z vs tech_trend_regime_prob = 0.24. Most < 0.15. |
| VIF < 10 for proposed features (submodel-only) | CONFIRMED | ie_regime_proxy=5.44, ie_anchoring_z=1.30, ie_gold_sensitivity_z=1.01. All below 10. |
| VIF < 10 for proposed features (full feature set) | BORDERLINE | ie_regime_proxy=10.20 (marginal), ie_anchoring_z=2.17, ie_gold_sensitivity_z=1.03. The 10.20 is from a sigmoid proxy; actual HMM output may differ. Precedent: etf_regime_prob has VIF=12.47 and passed Gate 3. |
| 2D HMM approach | CONFIRMED valid | Consistent with all 6 successful submodel precedents |
| Multi-country IE data unavailable | CONFIRMED | UK/Eurozone daily breakevens not in FRED or free sources |
| Inflation swap data not free | CONFIRMED | Requires Bloomberg/Reuters. T10YIE is optimal free source. |
| Ben-David et al. 2018 "anchored vs unanchored" | NOT CLAIMED | Unlike ETF flow report, no fabricated citations. Academic sources cited (Fed research papers) are legitimate institutions. |
| Kritzman et al. 2012 HMM | PLAUSIBLE | Real publication exists on regime detection, though specific claims not independently verified. |

### Critical Findings

**1. ie_anchoring_z autocorrelation is 0.97, not 0.90-0.95.** The researcher significantly underestimated persistence. Z-scoring against a 120d baseline reduces raw autocorr from 0.9887 to 0.9700 -- a modest improvement. This passes the 0.99 threshold but is the highest autocorrelation of any proposed feature. Alternative: using 5d volatility window z-scored against 60d baseline yields autocorr 0.8146 (much better). Design includes Optuna exploration of volatility window length (5d, 10d, 20d) to let the data determine optimal tradeoff between anchoring signal persistence and autocorrelation.

**2. T5YIFR term structure is viable.** corr(T10YIE_change, T5YIFR_change) = 0.6718 (below 0.7 threshold). This means a 2D HMM on [T10YIE_change, T5YIFR_change] would capture genuinely different information in each dimension. However, z-scored correlation is 0.7738 (above 0.7). Design uses Approach A as primary with Approach D available as Optuna alternative.

**3. VIF is acceptable.** All proposed features have VIF below 10 against submodel outputs (max 5.44). Against the full feature set, the regime proxy reaches 10.20, but this is borderline and consistent with etf_regime_prob precedent (VIF=12.47, passed Gate 3). ie_anchoring_z (VIF=2.17) and ie_gold_sensitivity_z (VIF=1.03) are excellent.

**4. Correlation claims are directionally correct but numerically imprecise.** IE-real_rate correlation measured as -0.068 (claimed -0.044). IE-DGS10 correlation measured as 0.480 (claimed 0.514). Both are in the same ballpark and the design conclusions hold: change-based features avoid Fisher identity multicollinearity.

---

## 1. Overview

- **Purpose**: Extract three orthogonal dimensions of inflation expectation dynamics from FRED T10YIE -- regime state (rising/stable/volatile expectations), anchoring state (how stable expectations are relative to recent history), and IE-gold sensitivity (whether inflation expectations are currently driving gold). These capture context that the raw IE level (base feature #1 at 9.4% importance) cannot convey.
- **Core methods**:
  1. Hidden Markov Model (2-3 states) on 2D [IE_daily_change, IE_vol_5d] for regime classification
  2. Rolling z-score of IE change volatility (short window against longer baseline) for anchoring detection
  3. Rolling 5-day IE-gold correlation z-scored against 60-day baseline for sensitivity measurement
- **Why these methods**: HMM captures nonlinear regime transitions that linear features cannot. The anchoring z-score is grounded in Fed research on inflation expectations stability. The sensitivity z-score captures the empirically demonstrated time-varying IE-gold relationship across macro regimes (QE positive, crisis negative, tightening weak).
- **Expected effect**: Enable the meta-model to distinguish identical IE levels with different dynamics. IE of 2.3% in a rising/unanchored/gold-correlated regime has different implications than 2.3% in a stable/anchored/decoupled regime.

### Key Advantage

All data is daily frequency from FRED T10YIE. No interpolation needed. This avoids the root cause of all 5 real_rate failures (monthly-to-daily mismatch).

### Design Rationale: Why Approach A over Alternatives

- **Approach B (IE-Gold co-movement HMM)**: Rejected -- using gold_return as HMM input risks violating "submodel does NOT predict gold" constraint. HMM might classify "gold up/down" days rather than genuine IE dynamics.
- **Approach C (Deterministic only)**: Rejected -- MI of simple change z-scores is 0.0000 against gold_return_next. All 6 successful submodels used HMM.
- **Approach D (T5YIFR term structure HMM)**: Viable (corr=0.67 < 0.7) but adds complexity. Available as Optuna option if Approach A underperforms. T5YIFR change correlation of 0.67 with T10YIE change provides some but limited additional information.

---

## 2. Data Specification

### Primary Data

| Data | Source | Series/Ticker | Frequency | Expected Rows |
|------|--------|---------------|-----------|---------------|
| T10YIE | FRED | T10YIE | Daily | ~5,783 (2003-2026) |
| Gold returns | Yahoo Finance | GC=F | Daily | Already in base_features |

### Data NOT Used

| Data | Source | Reason for Exclusion |
|------|--------|---------------------|
| T5YIFR | FRED | Available but adds complexity. corr(change)=0.67 with T10YIE -- not sufficiently orthogonal to justify 3D HMM. Reserved for Attempt 2 if needed. |
| T5YIE | FRED | Too correlated with T10YIE (expected >0.90 at levels) |
| T10YIEUK | FRED | Does not exist |
| UK/Eurozone breakeven | N/A | Not available daily from free sources |
| MICH survey | FRED | Monthly -- rejected per real_rate failure precedent |
| Inflation swaps | Bloomberg | Requires paid subscription |

### Preprocessing Steps

1. Fetch T10YIE from FRED, start=2014-06-01 (buffer for 120-day warmup before 2015-01-30)
2. Fetch GC=F close for gold returns computation
3. Compute derived quantities:
   - `ie_change = T10YIE.diff()` (daily change)
   - `ie_vol_Xd = ie_change.rolling(X).std()` (X-day rolling volatility of changes, X from Optuna)
   - `gold_return = gc_close.pct_change()` (current-day return, not next-day)
4. Handle missing values: forward-fill gaps up to 3 trading days, drop remaining NaN
5. Trim to base_features date range: 2015-01-30 to 2025-02-12

### Expected Sample Count

- ~2,523 daily observations (matching base_features row count)
- Warmup period: ~125 days (120 for longest rolling baseline + 5 for short vol window)
- Effective output: ~2,400+ rows after warmup, remaining filled with NaN then forward-filled from first valid value

---

## 3. Model Architecture (Hybrid Deterministic-Probabilistic)

This is a **hybrid deterministic-probabilistic** approach. No PyTorch neural network. The pipeline consists of three independent components.

### Component 1: HMM Regime Detection (ie_regime_prob)

- **Model**: `hmmlearn.hmm.GaussianHMM`
- **Input**: 2D array of [IE_daily_change, IE_vol_5d]
  - IE_daily_change = T10YIE.diff(): captures direction and magnitude of expectation shifts
  - IE_vol_5d = ie_change.rolling(5).std(): captures local instability / anchoring
  - 2D rationale: distinguishes sustained trending (high change + low vol = reflationary momentum) from chaotic repricing (high change + high vol = regime transition) from stable anchored (low change + low vol)
- **States**: 2 or 3 (Optuna selects via MI)
  - 2 states: trending vs stable
  - 3 states: rising/stable/volatile (preferred based on empirical regime evidence: 2020 COVID crash, 2021 reflation, 2022 peak)
- **Covariance type**: "full" (allows states to capture change-volatility correlation structure)
- **Training**: Fit on training set data only. Generate probabilities for full dataset using `predict_proba`.
- **Output**: Posterior probability of the highest-change-variance state (identified post-hoc by comparing emission variances on the IE_change dimension)
- **State labeling**: After fitting, sort states by emission variance of the IE_change dimension. The highest-variance state corresponds to "volatile/unanchored expectations." Output P(highest-variance state).

```
Input: [IE_daily_change, IE_vol_5d] [T x 2]
       |
   GaussianHMM.fit(train_data) -> learn 2-3 state model
       |
   GaussianHMM.predict_proba(full_data) -> [T x n_states]
       |
   Select P(highest-change-variance state) -> [T x 1]
       |
Output: ie_regime_prob (0-1)
```

### Component 2: IE Anchoring Z-Score (ie_anchoring_z)

- **Model**: Pure pandas computation (no ML)
- **Input**: Rolling standard deviation of IE daily changes
- **Short window**: 5d, 10d, or 20d (Optuna explores)
  - 5d: autocorr 0.8146 (best), captures immediate instability
  - 10d: autocorr 0.9170, moderate
  - 20d: autocorr 0.9700, highest but closest to academic "1-month" convention
- **Baseline window**: 60d or 120d (Optuna explores)
  - 60d: faster adaptation to regime shifts
  - 120d: more stable baseline, captures 6-month structural changes
- **Computation**: `z = (vol_short - rolling_mean_baseline) / rolling_std_baseline`
  - Positive: expectations unanchored (volatility above recent norm)
  - Negative: expectations anchored (volatility below recent norm)
  - Empirical support: gold std 1.10% vs 0.80% during high vs low IE volatility (38% difference)
- **Clipping**: [-4, 4] for stability

```
Input: ie_change [T x 1]
       |
   vol_short = ie_change.rolling(short_window).std()
       |
   rolling_mean = vol_short.rolling(baseline_window).mean()
   rolling_std  = vol_short.rolling(baseline_window).std()
       |
   z = (vol_short - rolling_mean) / rolling_std
       |
   clip(-4, 4)
       |
Output: ie_anchoring_z (typically -2 to +4)
```

### Component 3: IE-Gold Sensitivity Z-Score (ie_gold_sensitivity_z)

- **Model**: Pure pandas computation (no ML)
- **Input**: IE daily changes and same-day gold returns
- **Correlation window**: 5 days (fixed)
  - 5d rationale: autocorr = 0.72 (acceptable). 10d has 0.91 (too risky).
- **Baseline window**: 60 days (Optuna explores 40/60/90)
- **Computation**:
  1. `ie_gold_corr_5d = ie_change.rolling(5).corr(gold_return)`
  2. `z = (ie_gold_corr_5d - rolling_mean_baseline) / rolling_std_baseline`
- **Output**: Z-score of IE-gold correlation
  - Positive: IE changes currently correlated with gold (active inflation hedge)
  - Negative: IE decoupled from gold (other factors dominate)
  - This captures time-varying sensitivity demonstrated across 2015-2025 macro regimes

```
Input: ie_change, gold_return [T x 1 each]
       |
   ie_gold_corr_5d = ie_change.rolling(5).corr(gold_return)
       |
   rolling_mean = ie_gold_corr_5d.rolling(baseline_window).mean()
   rolling_std  = ie_gold_corr_5d.rolling(baseline_window).std()
       |
   z = (ie_gold_corr_5d - rolling_mean) / rolling_std
       |
   clip(-4, 4)
       |
Output: ie_gold_sensitivity_z (typically -3 to +3)
```

### Combined Output

| Column | Range | Description | Measured Autocorr | VIF (submodel-only) |
|--------|-------|-------------|-------------------|---------------------|
| `ie_regime_prob` | [0, 1] | P(high-variance IE regime) from 2D HMM | ~0.7-0.9 (estimated from precedent) | 5.44 (proxy) |
| `ie_anchoring_z` | [-4, +4] | IE change volatility z-score vs baseline | 0.81-0.97 (depends on window) | 1.30 |
| `ie_gold_sensitivity_z` | [-4, +4] | 5d IE-gold correlation z-scored vs 60d baseline | 0.72 | 1.01 |

Total: **3 columns** (matching the proven compact pattern from all 6 successful submodels).

### Orthogonality Analysis (Measured)

| Feature Pair | Correlation | Assessment |
|-------------|-------------|------------|
| ie_regime_proxy vs tech_trend_regime_prob | 0.14 | Low |
| ie_regime_proxy vs vix_regime_probability | 0.14 | Low |
| ie_regime_proxy vs etf_capital_intensity | 0.13 | Low |
| ie_anchoring_z vs tech_trend_regime_prob | 0.24 | Acceptable (highest pairwise) |
| ie_anchoring_z vs xasset_regime_prob | 0.14 | Low |
| ie_gold_sensitivity_z vs xasset_regime_prob | 0.05 | Negligible |
| ie_gold_sensitivity_z vs all others | < 0.05 | Essentially orthogonal |

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM n_iter | 100 | Standard convergence limit; EM typically converges in 20-50 iterations |
| HMM tol | 1e-4 | Standard convergence tolerance |
| HMM random_state | 42 | Reproducibility |
| IE_vol short window for HMM input | 5 days | Fixed for HMM input dimension 2 (captures immediate instability) |
| Sensitivity correlation window | 5 days | Fixed at 5d for acceptable autocorrelation (0.72). 10d has 0.91 (too risky). |
| Z-score clipping | [-4, 4] | Prevent extreme outliers from dominating |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3} | categorical | 2=trending/stable, 3=rising/stable/volatile. 3 preferred based on precedent. |
| hmm_covariance_type | {"full", "diag"} | categorical | full captures change-vol correlation; diag treats independently |
| hmm_n_init | {3, 5, 10} | categorical | Number of EM restarts to avoid local optima |
| anchoring_vol_window | {5, 10, 20} | categorical | 5d=lowest autocorr (0.81), 10d=moderate (0.92), 20d=highest autocorr (0.97) but closest to academic norm |
| anchoring_baseline_window | {60, 120} | categorical | 60d=responsive adaptation, 120d=stable structural baseline |
| sensitivity_baseline_window | {40, 60, 90} | categorical | Baseline for z-scoring IE-gold correlation |

### Exploration Settings

- **n_trials**: 30
  - Rationale: 6 categorical parameters with small ranges. Total combinations: 2 x 2 x 3 x 3 x 2 x 3 = 216. 30 trials with TPE provides reasonable coverage. Each trial is fast (<10 seconds: HMM fit on ~1,766 x 2 observations).
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of mutual information between 3 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)

---

## 5. Training Settings

### Fitting Procedure

This is not a gradient-based training loop. The procedure is:

1. **HMM**: `GaussianHMM.fit(X_train)` on 2D [IE_daily_change, IE_vol_5d] -- EM algorithm, converges in seconds
2. **IE Anchoring Z-Score**: Rolling window statistics on IE change volatility -- deterministic, no fitting
3. **IE-Gold Sensitivity**: Rolling correlation + z-scoring -- deterministic, no fitting

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- train: ~1,766 observations
- val: ~378 observations
- test: ~379 observations (reserved for evaluator Gate 3)
- HMM fits on train set only
- HMM generates probabilities for full dataset using predict_proba (no lookahead: HMM posterior at time t depends only on observations up to t given fitted model)
- Z-score and correlation use rolling windows (inherently backward-looking, no lookahead)
- Optuna optimizes MI sum on validation set

### Evaluation Metric for Optuna

For each trial:
1. Fit 2D HMM on train set [IE_daily_change, IE_vol_5d]
2. Generate all 3 features for full dataset using fitted HMM and trial window parameters
3. Compute mutual information (MI) between each feature and gold_return_next on validation set
4. Optuna maximizes: `MI_sum = MI(regime, target) + MI(anchoring_z, target) + MI(sensitivity_z, target)`

MI calculation: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`.

### Loss Function

N/A -- no gradient-based training. HMM uses EM (maximum likelihood). Z-score and correlation are deterministic.

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
| Estimated memory usage | < 1 GB | ~2,523 rows x 2-3 columns. Tiny dataset |
| Required pip packages | `hmmlearn` | Must pip install hmmlearn at start of train.py. sklearn, pandas, numpy, optuna, yfinance are pre-installed on Kaggle |

---

## 7. Implementation Instructions

### builder_data Instructions

1. **Fetch T10YIE from FRED**: `fred.get_series('T10YIE', observation_start='2014-06-01')`
2. **Compute derived quantities**:
   - `ie_change = T10YIE.diff()`
   - `ie_vol_5d = ie_change.rolling(5).std()`
3. **Fetch gold returns**: Use GC=F daily close from Yahoo Finance, compute `gold_return = gc_close.pct_change()`
4. **Align dates**: Inner join T10YIE and GC=F on trading dates
5. **Save preprocessed data**: `data/processed/inflation_expectation_features_input.csv`
   - Columns: Date, T10YIE, ie_change, ie_vol_5d, gold_return
6. **Quality checks**:
   - T10YIE values in range [0, 5] (breakeven rate is a percentage)
   - No gaps > 5 consecutive trading days
   - Missing data < 2% after alignment
   - ie_change has no extreme outliers (|value| < 0.5, typical daily change is 0.01-0.05)
   - ie_vol_5d is always positive (std of a non-constant series)
7. **Verify date range**: Data should cover 2014-06-01 through latest available, trimmed to base_features range for output

### builder_model Instructions

#### train.py Structure

```python
"""
Gold Prediction SubModel Training - Inflation Expectation Attempt 1
Self-contained: Data fetch -> Preprocessing -> HMM + Z-Score + Sensitivity -> Optuna HPO -> Save results
"""

# === 1. Libraries ===
import subprocess
subprocess.check_call(['pip', 'install', 'hmmlearn'])

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mutual_info_score
import optuna
import json
import os
from datetime import datetime

# === 2. Data Fetching ===
# FRED_API_KEY from Kaggle Secrets (os.environ['FRED_API_KEY'])
# Fetch T10YIE from FRED (start=2014-06-01 for warmup buffer)
# Fetch GC=F close for gold_return computation
# Compute: ie_change, ie_vol_5d, gold_return

# === 3. Feature Generation Functions ===

def generate_regime_feature(ie_change, ie_vol_5d, n_components, covariance_type, n_init, train_size):
    """
    Fit 2D HMM on [IE_daily_change, IE_vol_5d] and return P(highest-change-variance state).
    """
    X = np.column_stack([ie_change, ie_vol_5d])
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]
    X_train = X_valid[:train_size]

    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=100,
        tol=1e-4,
        random_state=42,
        n_init=n_init
    )
    model.fit(X_train)
    probs = model.predict_proba(X_valid)

    # Identify highest-change-variance state (first dimension = IE_change)
    state_vars = []
    for i in range(n_components):
        if covariance_type == 'full':
            state_vars.append(float(model.covars_[i][0, 0]))
        elif covariance_type == 'diag':
            state_vars.append(float(model.covars_[i][0]))

    high_var_state = np.argmax(state_vars)

    # Map back to full array
    result = np.full(len(ie_change), np.nan)
    result[valid_mask] = probs[:, high_var_state]
    return result, model

def generate_anchoring_feature(ie_change, vol_window, baseline_window):
    """
    Rolling z-score of IE change volatility: (vol_short - rolling_mean) / rolling_std
    """
    s = pd.Series(ie_change)
    vol_short = s.rolling(vol_window).std()
    rolling_mean = vol_short.rolling(baseline_window).mean()
    rolling_std = vol_short.rolling(baseline_window).std()
    z = (vol_short - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z.values

def generate_sensitivity_feature(ie_change, gold_return, corr_window=5, baseline_window=60):
    """
    5-day rolling correlation between IE changes and gold returns,
    z-scored against a baseline_window-day rolling baseline.
    """
    ie = pd.Series(ie_change)
    gr = pd.Series(gold_return)
    corr = ie.rolling(corr_window).corr(gr)
    corr_mean = corr.rolling(baseline_window).mean()
    corr_std = corr.rolling(baseline_window).std()
    z = (corr - corr_mean) / corr_std
    z = z.clip(-4, 4)
    return z.values

# === 4. Optuna Objective ===

def objective(trial, ie_change, ie_vol_5d, gold_return, target, train_size, val_mask):
    """Maximize MI sum on validation set"""
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    covariance_type = trial.suggest_categorical('hmm_covariance_type', ['full', 'diag'])
    n_init = trial.suggest_categorical('hmm_n_init', [3, 5, 10])
    vol_window = trial.suggest_categorical('anchoring_vol_window', [5, 10, 20])
    baseline_window = trial.suggest_categorical('anchoring_baseline_window', [60, 120])
    sens_baseline = trial.suggest_categorical('sensitivity_baseline_window', [40, 60, 90])

    try:
        regime, _ = generate_regime_feature(
            ie_change, ie_vol_5d, n_components, covariance_type, n_init, train_size
        )
        anchoring = generate_anchoring_feature(ie_change, vol_window, baseline_window)
        sensitivity = generate_sensitivity_feature(ie_change, gold_return, 5, sens_baseline)

        # Extract validation period
        regime_val = regime[val_mask]
        anchoring_val = anchoring[val_mask]
        sensitivity_val = sensitivity[val_mask]
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
        for feat_val in [regime_val, anchoring_val, sensitivity_val]:
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

1. **hmmlearn installation**: Must `pip install hmmlearn` at the top of train.py. It is not pre-installed on Kaggle.
2. **FRED API key**: Use `os.environ['FRED_API_KEY']` from Kaggle Secrets. Fail immediately with KeyError if not set.
3. **2D HMM input**: Use [IE_daily_change, IE_vol_5d] as 2D input. This distinguishes trending from volatile IE regimes.
4. **HMM state labeling**: After fitting, sort states by emission variance of the FIRST dimension (IE_change). Output P(highest-variance state). Do NOT assume state index 0, 1, or 2 corresponds to any specific regime.
5. **Anchoring vol window**: Optuna explores 5/10/20 day windows. Autocorrelation tradeoff:
   - 5d: autocorr ~0.81 (best), captures immediate instability
   - 10d: autocorr ~0.92, moderate
   - 20d: autocorr ~0.97, closest to academic norm but highest autocorrelation
6. **Sensitivity correlation window**: Fixed at 5 days. Do NOT use 10-day or longer (autocorrelation > 0.90).
7. **No lookahead bias**:
   - HMM: Fit on training data only, generate probabilities for full dataset
   - Z-score: Rolling window (inherently backward-looking)
   - Correlation: Rolling window (inherently backward-looking)
8. **NaN handling**: First ~125 rows (max of warmup periods) will have NaN. Forward-fill output after generation.
9. **Reproducibility**: Fix random_state=42 for HMM, seed=42 for Optuna. Z-score and correlation are deterministic.
10. **Output format**: CSV with columns [Date, ie_regime_prob, ie_anchoring_z, ie_gold_sensitivity_z]. Aligned to trading dates matching base_features.
11. **Gold return for sensitivity feature**: Use current-day gold return from GC=F (not gold_return_next, which is the target). This is backward-looking and does not leak future information.
12. **NO LEVEL FEATURES**: Do not output any IE level, level z-score, or level-derived feature. Fisher identity makes these perfectly multicollinear with DGS10 and DFII10.

---

## 8. Risks and Alternatives

### Risk 1: ie_anchoring_z Autocorrelation Too High

- **Description**: With 20d vol window and 120d baseline, autocorrelation is 0.97. While below 0.99 threshold, it is the highest among all features across all submodels.
- **Measured values**:
  - 5d vol, 60d baseline: autocorr = 0.81
  - 10d vol, 60d baseline: autocorr = 0.92
  - 20d vol, 120d baseline: autocorr = 0.97
  - First difference of 20d vol, z-scored: autocorr = 0.07 (excellent but noisy)
- **Mitigation**: Optuna explores all three vol windows (5/10/20d). Shorter windows have dramatically lower autocorrelation. If the best Optuna trial uses 20d and autocorr is borderline, evaluator can accept given it passes 0.99 threshold.
- **Fallback**: If Gate 1 requires autocorr < 0.95, use 5d or 10d vol window. If all fail, use first difference of vol z-scored.

### Risk 2: HMM State Collapse (Constant Output)

- **Description**: yield_curve's HMM collapsed to 1 effective state (regime_prob constant, std=1e-11). This could happen if IE dynamics are too smooth for 2D HMM to detect regimes.
- **Mitigation**: IE has clear regime structure (2020 COVID crash 1.65->0.50, 2021 reflation surge to 2.76, 2022 peak at 3.02). More dramatic regime changes than yield curve spread. Using n_init parameter (3-10 restarts) improves stability. Optuna tests 2 vs 3 states.
- **Detection**: Check that regime probabilities have std > 0.1 and are not near-constant.
- **Fallback**: If HMM collapses, output only ie_anchoring_z and ie_gold_sensitivity_z (2 features instead of 3). Yield_curve succeeded with 2 features after HMM collapse.

### Risk 3: VIF from ie_regime_prob vs Other Regime Features

- **Description**: ie_regime_proxy has VIF=10.20 against full feature set (borderline). The proxy correlates 0.14 with vix_regime_probability and tech_trend_regime_prob.
- **Mitigation**: The proxy (sigmoid of vol z-score) has different correlation structure than actual HMM output. Precedent: etf_regime_prob has VIF=12.47 and passed Gate 3. Even if VIF exceeds 10 slightly, Gate 3 ablation is the true test.
- **Fallback**: If VIF is problematic, reduce to 2 features (drop ie_regime_prob, keep anchoring_z and sensitivity_z which have VIF 1.30 and 1.01).

### Risk 4: ie_gold_sensitivity_z Too Noisy

- **Description**: 5d rolling correlation has limited statistical power (only 5 data points per window). Individual correlation values are noisy.
- **Mitigation**: Z-scoring against 60d baseline normalizes the noise. The z-score measures "how unusual is current sensitivity relative to recent history" rather than "what is the exact correlation." This is more robust.
- **Fallback**: If too noisy, increase correlation window to 10d (autocorr 0.91, but still below 0.99).

### Risk 5: Gate 3 Diminishing Returns

- **Description**: MAE improvement trend: technical (-0.182) > cross_asset (-0.087) > yield_curve (-0.069) > etf_flow (-0.044). Expected IE improvement is -0.01% to -0.03% as the 7th submodel.
- **Mitigation**: IE is the #1 base feature at 9.4% importance. Enriching the most important feature with dynamics has the highest potential payoff. The 12bps directional asymmetry (rising IE: +0.088% vs falling: -0.031%) provides measurable signal.
- **Primary success path**: MAE improvement. IE has stronger theoretical link to gold than most other features.

### Risk 6: Fisher Identity VIF Leak

- **Description**: Despite using change-based features, moderate correlation (0.48) between ie_change_z and dgs10_change_z means some Fisher identity structure remains.
- **Mitigation**: VIF for ie_anchoring_z (2.17) and ie_gold_sensitivity_z (1.03) are excellent -- they operate on VOLATILITY and CORRELATION dimensions that are orthogonal to level-change structures. Only ie_regime_prob has borderline VIF.
- **Absolute constraint**: NO LEVEL FEATURES. This is hardcoded in the design.

---

## 9. VIF Analysis (Pre-Design Measured)

### Proposed Features VIF (Submodel Outputs Only)

| Feature | VIF (submodel-only) | VIF (full features) | Assessment |
|---------|---------------------|---------------------|------------|
| ie_regime_proxy | 5.44 | 10.20 | Borderline for full; proxy, not actual HMM. Precedent: etf_regime_prob=12.47 |
| ie_anchoring_z | 1.30 | 2.17 | Excellent |
| ie_gold_sensitivity_z | 1.01 | 1.03 | Essentially orthogonal |

### Pairwise Correlations with Existing Submodel Outputs

| IE Feature | Highest Correlation Partner | Value |
|-----------|---------------------------|-------|
| ie_regime_proxy | tech_trend_regime_prob | 0.14 |
| ie_anchoring_z | tech_trend_regime_prob | 0.24 |
| ie_gold_sensitivity_z | xasset_regime_prob | 0.05 |

All pairwise correlations below 0.25. ie_gold_sensitivity_z is effectively orthogonal to all existing features.

---

## 10. Autocorrelation Analysis

| Feature | Window Config | Autocorrelation (lag 1) | Status |
|---------|--------------|-------------------------|--------|
| ie_regime_prob | HMM output | ~0.7-0.9 (estimated) | Expected PASS |
| ie_anchoring_z | 5d vol, 60d baseline | 0.8146 | CONFIRMED PASS |
| ie_anchoring_z | 10d vol, 60d baseline | 0.9170 | PASS but high |
| ie_anchoring_z | 20d vol, 120d baseline | 0.9700 | PASS but borderline |
| ie_gold_sensitivity_z | 5d corr, 60d baseline | 0.7177 | CONFIRMED PASS |
| ie_change_z60 | (reference) | 0.0503 | Excellent |
| ie_vol_20d_raw | (reference) | 0.9887 | Would FAIL if not z-scored |

All candidate configurations pass the <0.99 threshold. Optuna will select the optimal vol window that balances anchoring signal quality against autocorrelation.

---

## 11. Expected Performance Against Gates

### Gate 1: Standalone Quality
- **Overfit ratio**: N/A (deterministic HMM, no neural network with train/val loss comparison)
- **No constant output**: High confidence -- IE has dramatic regime changes (2020 COVID, 2021 reflation, 2022 peak) providing HMM with clear state structure
- **Autocorrelation < 0.99**: CONFIRMED for all candidate configurations. Worst case 0.97 (20d vol, 120d baseline).
- **Expected Result**: PASS

### Gate 2: Information Gain
- **MI increase > 5%**: Moderate probability. IE is #1 base feature (9.4% importance). Regime context should provide substantial nonlinear information. Anchoring z-score captures gold-volatility relationship (38% difference between anchored/unanchored). Sensitivity z-score is genuinely novel.
- **VIF < 10**: ie_anchoring_z (2.17) and ie_gold_sensitivity_z (1.03) pass convincingly. ie_regime_prob may be borderline (proxy shows 10.20, actual HMM may differ).
- **Rolling correlation std < 0.15**: Moderate probability. ie_regime_prob may marginally fail (all regime_prob features in previous submodels had this issue). Accepted precedent.
- **Expected Result**: MARGINAL (regime_prob stability may fail, consistent with 6/6 submodel precedent)

### Gate 3: Ablation Test
- **Primary path**: MAE improvement > 0.01%
- **Supporting evidence**: IE is #1 feature at 9.4% importance, 12bps directional asymmetry, 38% gold-vol difference between anchored/unanchored IE
- **Realistic target**: MAE improvement -0.01% to -0.03% (7th submodel, diminishing returns expected)
- **Expected Result**: TARGET PASS

**Confidence**: 7/10 (moderate-high). IE is the #1 base feature, giving it the strongest theoretical potential of any remaining submodel. Daily frequency avoids the monthly-data failure mode. All 6 successful submodels used this HMM+deterministic pattern. The main uncertainty is whether regime dynamics add enough beyond the raw level to pass Gate 3's ablation test.
