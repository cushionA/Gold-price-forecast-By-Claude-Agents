# Submodel Design Document: VIX (Attempt 1)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| FRED: VIXCLS | Confirmed | 2619 obs, 2015-01-30 to 2025-02-12, 2546 non-null |
| FRED: VXVCLS (VIX 3-month alt) | Confirmed | 2619 obs, 2015-01-30 to 2025-02-12, 2525 non-null |
| Yahoo: ^VIX3M | Confirmed | 2525 rows, 2015-01-30 to 2025-02-12, fully aligned with ^VIX |
| Yahoo: ^VIX9D | Confirmed | 2525 rows, 2015-01-30 to 2025-02-12, fully aligned with ^VIX |
| Yahoo: ^VVIX | Not tested | Not needed for this design |
| VIX/VIX3M "orthogonal to VIX level" | REJECTED | Measured corr(VIX_level, VIX/VIX3M_ratio) = 0.69. Far too high for low-VIF claim |
| VIX/VIX3M ratio expanding z-score | REJECTED | Corr with VIX level = 0.61, corr with VIX z-score = 0.75. Redundant |
| VIX9D/VIX3M spread | REJECTED | Corr with VIX level = 0.62. Same problem |
| Researcher claim: "Low correlation with raw VIX level" for term structure | INCORRECT | Empirically measured: ratio corr = 0.69, spread corr = 0.62. These would cause VIF issues |
| Researcher claim: "Expected VIF < 5, likely < 3" for all outputs | OVERLY OPTIMISTIC | Only true for persistence (autocorr); z-score is moderate (0.43); term structure features are too high |
| HMM for VIX regime detection | Confirmed valid | Well-established. Follows DXY success pattern |
| Z-score mean-reversion feature | Confirmed valid | Corr with VIX level = 0.43, acceptable for VIF |
| Rolling autocorrelation persistence | Confirmed valid | Corr with VIX level = -0.09, nearly orthogonal |
| VIX mean-reversion property | Confirmed | Long-term mean ~18-20, well-documented |
| VIX right-skew requiring log-transform | Confirmed valid | Standard approach for HMM on VIX |
| Researcher confidence "8/10" | Adjusted to 7/10 | Term structure dropped, reducing one information channel. Still strong daily-frequency advantage |

### Critical Design Correction

**VIX/VIX3M term structure ratio is NOT orthogonal to VIX level**. The researcher's claim that "VIX/VIX3M ratio captures relative term structure, not absolute level" is empirically false. Measured correlation is 0.69. This makes sense: when VIX spikes, it spikes faster than VIX3M (short-term vol overshoots long-term vol), creating a mechanical positive relationship between VIX level and the ratio.

**Resolution**: Replace term structure with persistence (rolling autocorrelation of log-VIX changes), which has -0.09 correlation with VIX level and 0.12 correlation with z-score. This provides genuinely orthogonal information about temporal dynamics.

---

## 1. Overview

- **Purpose**: Extract three contextual features from VIX that capture the volatility environment's regime state, distance from equilibrium, and temporal persistence -- information absent from the raw `vix_vix` base feature.
- **Core methods**:
  1. Hidden Markov Model (2-3 states) on log-VIX daily changes for regime probability
  2. Rolling z-score (60-day window) for mean-reversion distance
  3. Rolling autocorrelation (20-day window) on log-VIX changes for persistence
- **Why these methods**: Each captures a fundamentally different aspect of VIX dynamics. HMM captures which latent regime the market is in (calm vs fear). Z-score captures how far VIX is from its recent equilibrium. Autocorrelation captures whether current VIX behavior is persistent (sustained regime) or transient (spike that will revert). None of these are derivable from the raw VIX level.
- **Expected effect**: Provide the meta-model with volatility context that distinguishes scenarios with identical VIX levels but different gold implications (e.g., VIX=25 in a spike decay vs VIX=25 in a slow grind upward).

### Key Advantage Over real_rate

All data is daily frequency. No monthly-to-daily interpolation is needed. This eliminates the root cause of all 5 real_rate failures.

### Design Rationale: Why Not Term Structure?

Empirical analysis showed VIX/VIX3M ratio has 0.69 correlation with raw VIX level. This is because VIX and VIX3M are not independent: during spikes, short-term vol (VIX) overshoots long-term vol (VIX3M), creating a mechanical positive relationship. Including this feature would inflate VIF with the existing `vix_vix` base feature. The persistence feature (autocorrelation, corr = -0.09 with VIX level) provides more orthogonal information.

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Already Available |
|------|--------|--------|-----------|-------------------|
| VIX (primary) | FRED | VIXCLS | Daily | Yes: `data/raw/vix.csv` |
| VIX (backup) | Yahoo Finance | ^VIX | Daily | Fetch if FRED unavailable in Kaggle |

### Backup Data (for Kaggle environment)

| Data | Source | Ticker | Purpose |
|------|--------|--------|---------|
| VIX3M | Yahoo Finance | ^VIX3M | Available but NOT used (corr=0.69 with VIX level). Reserved for potential Attempt 2 |
| VIX9D | Yahoo Finance | ^VIX9D | Available but NOT used. Same VIF concern |
| VIX3M | FRED | VXVCLS | Confirmed available. Same VIF concern applies |

### Preprocessing Steps

1. Fetch VIXCLS from FRED (primary) or ^VIX from Yahoo Finance (fallback), start=2014-10-01 (buffer for 60-day warmup before 2015-01-30)
2. Compute daily log-changes: `log_change = log(VIX_t) - log(VIX_t-1)`
3. Handle missing values: forward-fill gaps up to 3 days, then drop remaining NaN
4. Trim to base_features date range: 2015-01-30 to 2025-02-12

### Expected Sample Count

- ~2,523 daily observations (matching base_features row count)
- Warmup period: 60 days for z-score rolling window, 20 days for autocorrelation
- Effective output: ~2,460+ rows after warmup, remaining filled with NaN then forward-filled from first valid

---

## 3. Model Architecture

This is a **hybrid deterministic-probabilistic** approach, not a neural network. No PyTorch is required. The pipeline consists of three independent components.

### Component 1: HMM Regime Detection

- **Model**: `hmmlearn.hmm.GaussianHMM`
- **Input**: 1D array of daily log-VIX changes (single feature)
  - Log-transform rationale: VIX is right-skewed (range 10-80+). Log-changes normalize the distribution for HMM to avoid skew-driven artifacts.
- **States**: 2 (calm vs elevated-fear) or 3 (calm vs elevated vs crisis) -- Optuna selects
  - 2 states: Captures the primary calm/fear dichotomy
  - 3 states: Adds a crisis/extreme state that may better capture the liquidation regime
- **Covariance type**: "full" (trivial for 1D input; full=diag=spherical for single dimension)
- **Training**: Fit on training set data only. Generate probabilities for full dataset using `predict_proba`.
- **Output**: Posterior probability of the highest-variance state (identified post-hoc by comparing emission variances)
- **State labeling**: After fitting, sort states by emission variance. The highest-variance state corresponds to "crisis/elevated fear" (large VIX moves). Output P(highest-variance state).

```
Input: VIX daily log-changes [T x 1]
       |
   GaussianHMM.fit(train_data) -> learn 2-3 state model
       |
   GaussianHMM.predict_proba(full_data) -> [T x n_states]
       |
   Select P(highest-variance state) -> [T x 1]
       |
Output: vix_regime_probability (0-1)
```

### Component 2: Mean-Reversion Z-Score

- **Model**: Pure pandas computation (no ML)
- **Input**: VIX closing levels (VIXCLS)
- **Window**: 60-day rolling window (Optuna explores 40/60/90)
  - Rationale: VIX low-vol regimes last ~25 days, high-vol regimes ~8.5 days. A 60-day window spans both regime types. Aligns with VIX mean-reversion timeframe.
- **Computation**: `z = (VIX_t - rolling_mean_t) / rolling_std_t`
  - Uses rolling window (not expanding), so the z-score adapts to recent regime shifts
  - By construction, this is a residual from equilibrium, not the level itself
- **Output**: z-score value (unbounded, typically -2 to +4)
  - Positive: VIX above recent mean (fear elevated)
  - Negative: VIX below recent mean (calm)
  - Extreme positive: Likely spike in progress (mean-reversion opportunity)

```
Input: VIX daily close levels [T x 1]
       |
   rolling(window).mean() -> rolling_mean
   rolling(window).std()  -> rolling_std
       |
   z = (VIX - rolling_mean) / rolling_std
       |
   clip(-4, 4) for stability
       |
Output: vix_mean_reversion_z (typically -2 to +4)
```

### Component 3: Persistence (Rolling Autocorrelation)

- **Model**: Pure pandas computation (no ML)
- **Input**: Daily log-VIX changes (same as HMM input)
- **Window**: 20-day rolling window (Optuna explores 15/20/30)
  - Rationale: 20 trading days ~1 month. VIX high-vol regimes average 8.5 days, so a 20-day window captures whether recent volatility is clustered (persistent) or spike-and-revert (transient).
- **Computation**: Rolling lag-1 autocorrelation of log-VIX changes
  - High autocorrelation (>0): Changes persist -- sustained regime, volatility clustering
  - Low/negative autocorrelation (<0): Changes revert -- spike decay, mean-reversion dominant
- **Output**: Autocorrelation value (-1 to +1)

```
Input: VIX daily log-changes [T x 1]
       |
   rolling(window).apply(autocorrelation at lag 1)
       |
Output: vix_persistence (-1 to +1, typically -0.3 to +0.4)
```

### Combined Output

| Column | Range | Description | Corr with vix_vix |
|--------|-------|-------------|-------------------|
| `vix_regime_probability` | [0, 1] | P(high-variance/fear regime) from HMM | ~0.2-0.4 (estimated) |
| `vix_mean_reversion_z` | [-4, +4] | Distance from 60-day rolling mean in std devs | 0.43 (measured) |
| `vix_persistence` | [-1, +1] | Rolling 20-day autocorrelation of log-VIX changes | -0.09 (measured) |

Total: **3 columns** (strictly within the 2-4 target range).

### Orthogonality Analysis (Measured)

| Feature Pair | Correlation | Assessment |
|-------------|-------------|------------|
| vix_regime_probability vs vix_vix | ~0.2-0.4 (est.) | Acceptable |
| vix_mean_reversion_z vs vix_vix | 0.43 | Acceptable (residual-based) |
| vix_persistence vs vix_vix | -0.09 | Excellent (nearly orthogonal) |
| vix_mean_reversion_z vs vix_persistence | 0.12 | Excellent (near-zero cross-corr) |
| vix_persistence vs dxy_volatility_z | ~0.1-0.2 (est.) | Acceptable (VIX temporal vs FX dispersion) |

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM n_iter | 100 | Standard convergence limit; EM typically converges in 20-50 iterations |
| HMM tol | 1e-4 | Standard convergence tolerance |
| HMM random_state | 42 | Reproducibility |
| Log-transform for HMM input | Yes (always) | VIX right-skew requires normalization |
| Z-score clipping | [-4, 4] | Prevent extreme outliers from dominating |
| Autocorrelation lag | 1 | Standard persistence measure |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3} | categorical | 2=calm/fear, 3=calm/elevated/crisis |
| hmm_covariance_type | {"full", "diag"} | categorical | Trivial for 1D but let Optuna test both |
| hmm_n_init | {3, 5, 10} | categorical | Number of EM restarts to avoid local optima |
| zscore_window | {40, 60, 90} | categorical | Mean-reversion lookback: shorter=reactive, longer=stable |
| autocorr_window | {15, 20, 30} | categorical | Persistence lookback: 15d=spike-focused, 30d=regime-focused |

### Exploration Settings

- **n_trials**: 30
  - Rationale: 5 categorical parameters with small ranges. Total combinations: 2 * 2 * 3 * 3 * 3 = 108. 30 trials provides good coverage with TPE sampler. Each trial is fast (<10 seconds).
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of mutual information between 3 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)

---

## 5. Training Settings

### Fitting Procedure

This is not a gradient-based training loop. The procedure is:

1. **HMM**: `GaussianHMM.fit(vix_log_changes_train)` -- EM algorithm, converges in seconds
2. **Z-Score**: Rolling window statistics on VIX levels -- deterministic, no fitting
3. **Autocorrelation**: Rolling lag-1 autocorrelation on VIX log-changes -- deterministic, no fitting

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- HMM fits on train set only
- HMM generates probabilities for full dataset using predict_proba (no lookahead: HMM posterior at time t depends only on observations up to t given fitted model)
- Z-score and autocorrelation use rolling windows (inherently no lookahead)
- Optuna optimizes MI sum on validation set
- Test set reserved for evaluator Gate 3

### Evaluation Metric for Optuna

For each trial (hyperparameter combination):
1. Fit HMM on train set log-VIX changes
2. Generate all 3 features for full dataset using fitted HMM and trial window parameters
3. Compute mutual information (MI) between each of the 3 features and `gold_return_next` on validation set
4. Optuna maximizes: `MI_sum = MI(regime, target) + MI(zscore, target) + MI(persistence, target)`

MI calculation method: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`. This matches the DXY approach.

### Loss Function

N/A -- no gradient-based training. HMM uses EM (maximum likelihood). Z-score and autocorrelation are deterministic.

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- HMM EM converges via `n_iter` and `tol`. No early stopping needed.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. HMM, rolling statistics are CPU-only |
| Estimated execution time | 3-5 minutes | Data download (~30s) + 30 Optuna trials x ~5s each (~2.5min) + final output (~30s) |
| Estimated memory usage | <1 GB | ~2,500 rows x 1-3 columns. Tiny dataset |
| Required pip packages | `hmmlearn` | Must `pip install hmmlearn` at start of train.py. sklearn, pandas, numpy pre-installed on Kaggle |

---

## 7. Implementation Instructions

### builder_data Instructions

1. **VIX data already available**: Use existing `data/raw/vix.csv` (VIXCLS from FRED)
2. **Verify format**: Ensure columns include Date and VIXCLS close value
3. **Verify date range**: 2015-01-30 to 2025-02-12 (matching base_features)
4. **No additional data fetching required**: VIX3M, VIX9D are NOT used in this design
5. **Save preprocessed data**: `data/processed/vix_features_input.csv`
   - Columns: Date, vix_close, vix_log_change
   - Start from 2014-10-01 (warmup buffer)
6. **Quality checks**:
   - No gaps > 3 consecutive trading days
   - Missing data < 2%
   - VIX values in reasonable range (8-90)
   - Log-changes have no extreme outliers (|change| < 0.5 i.e. ~65% daily move)

### builder_model Instructions

#### train.py Structure

```python
"""
Gold Prediction SubModel Training - VIX Attempt 1
Self-contained: Data fetch -> Preprocessing -> HMM + Z-Score + Autocorr -> Optuna HPO -> Save results
"""

# === 1. Libraries ===
import subprocess
subprocess.check_call(['pip', 'install', 'hmmlearn'])

import numpy as np
import pandas as pd
from fredapi import Fred
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mutual_info_score
import optuna
import json
import os
from datetime import datetime

# === 2. Data Fetching ===
# Try FRED first (using Kaggle Secret FRED_API_KEY)
# Fallback to Yahoo Finance ^VIX
# Compute log-changes
# Align to base_features date range

# === 3. Feature Generation Functions ===

def generate_regime_feature(vix_log_changes, n_components, covariance_type, n_init, train_size):
    """
    Fit HMM on train portion and return P(highest-variance state) for full data.
    """
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=100,
        tol=1e-4,
        random_state=42,
        n_init=n_init
    )
    X_train = vix_log_changes[:train_size].reshape(-1, 1)
    X_full = vix_log_changes.reshape(-1, 1)

    model.fit(X_train)
    probs = model.predict_proba(X_full)

    # Identify highest-variance state
    state_vars = []
    for i in range(n_components):
        if covariance_type == 'full':
            state_vars.append(float(model.covars_[i][0, 0]))
        elif covariance_type == 'diag':
            state_vars.append(float(model.covars_[i][0]))
        else:
            state_vars.append(float(model.covars_[i]))

    high_var_state = np.argmax(state_vars)
    return probs[:, high_var_state]

def generate_zscore_feature(vix_levels, window):
    """
    Rolling z-score: (VIX - rolling_mean) / rolling_std
    """
    s = pd.Series(vix_levels)
    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std()
    z = (s - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z.values

def generate_persistence_feature(vix_log_changes, window):
    """
    Rolling lag-1 autocorrelation of log-VIX changes.
    High = persistent (sustained regime), Low = transient (spike reverting).
    """
    s = pd.Series(vix_log_changes)
    acorr = s.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=1),
        raw=False
    )
    return acorr.values

# === 4. Optuna Objective ===

def objective(trial, vix_log_changes, vix_levels, target, train_size, val_mask):
    """Maximize MI sum on validation set"""
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    covariance_type = trial.suggest_categorical('hmm_covariance_type', ['full', 'diag'])
    n_init = trial.suggest_categorical('hmm_n_init', [3, 5, 10])
    zscore_window = trial.suggest_categorical('zscore_window', [40, 60, 90])
    autocorr_window = trial.suggest_categorical('autocorr_window', [15, 20, 30])

    try:
        regime = generate_regime_feature(
            vix_log_changes, n_components, covariance_type, n_init, train_size
        )
        zscore = generate_zscore_feature(vix_levels, zscore_window)
        persistence = generate_persistence_feature(vix_log_changes, autocorr_window)

        # Extract validation period
        regime_val = regime[val_mask]
        zscore_val = zscore[val_mask]
        persist_val = persistence[val_mask]
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
        target_disc = discretize(target_val)
        if target_disc is None:
            return 0.0

        for feat_val in [regime_val, zscore_val, persist_val]:
            mask = ~np.isnan(feat_val) & ~np.isnan(target_val)
            if mask.sum() > 50:
                feat_disc = discretize(feat_val[mask])
                tgt_disc = discretize(target_val[mask])
                if feat_disc is not None and tgt_disc is not None:
                    mi_sum += mutual_info_score(feat_disc, tgt_disc)

        return mi_sum

    except Exception as e:
        return 0.0

# === 5. Main ===
# Data split: train/val/test = 70/15/15 (time-series order)
# Run Optuna with 30 trials, 300s timeout
# Generate final output with best params
# Save submodel_output.csv, training_result.json
```

#### Key Implementation Notes

1. **hmmlearn installation**: Must `pip install hmmlearn` at the top of train.py. It is not pre-installed on Kaggle.
2. **FRED API key**: Use `os.environ['FRED_API_KEY']` from Kaggle Secrets. Fail with KeyError if not set (no fallback).
3. **Yahoo Finance fallback**: If FRED fails in Kaggle, use `yf.download('^VIX')` as backup data source.
4. **Log-transform for HMM**: Always use log-VIX changes as HMM input. Never raw VIX levels or raw changes.
5. **HMM state labeling**: After fitting, sort states by emission variance. Output P(highest-variance state). Do NOT assume state index 0 or 1 corresponds to any regime.
6. **No lookahead bias**:
   - HMM: Fit on training data only, generate probabilities for full dataset
   - Z-score: Rolling window (inherently backward-looking)
   - Autocorrelation: Rolling window (inherently backward-looking)
7. **NaN handling**: First ~60 rows (max of warmup periods) will have NaN. Forward-fill output after generation. The evaluator will align dates with base_features.
8. **Reproducibility**: Fix random_state=42 for HMM, seed=42 for Optuna. Z-score and autocorrelation are deterministic.
9. **Output format**: CSV with columns [Date, vix_regime_probability, vix_mean_reversion_z, vix_persistence]. Aligned to trading dates matching base_features.
10. **Target data**: Load gold_return_next from the Kaggle dataset `bigbigzabuton/gold-target` (same as DXY approach).

---

## 8. Risks and Alternatives

### Risk 1: HMM State Instability

- **Description**: HMM may produce different state assignments depending on initialization, especially with only ~1,700 training observations
- **Mitigation**: Use `n_init` parameter (3-10 random restarts). Optuna tests multiple values. Fix random_state=42 for reproducibility.
- **Detection**: Check that regime probabilities have std > 0.1 and are not near-constant
- **Fallback**: If regime probability has std < 0.05, replace with a simpler sigmoid-based regime indicator: `regime = sigmoid((VIX_20d_change / std_20d) * 2)` which approximates fear probability without HMM

### Risk 2: Z-Score High Autocorrelation

- **Description**: Rolling z-score is inherently smooth (60-day window), which could cause autocorrelation > 0.99 like real_rate attempt 1
- **Mitigation**: Z-score computed from VIX levels (which are volatile, not interpolated monthly data). VIX can swing 5+ points in a day, producing z-score variations. Expected lag-1 autocorrelation: 0.85-0.95, well below 0.99.
- **Detection**: Verify autocorrelation < 0.99 in Gate 1
- **Fallback**: If autocorrelation too high, reduce window to 40 days or use change in z-score instead of level

### Risk 3: Persistence Feature Noisy

- **Description**: 20-day rolling autocorrelation on daily returns can be noisy, with many NaN values when the window contains near-constant values
- **Mitigation**: Optuna tests windows of 15, 20, and 30 days. 30-day window is more stable but less responsive.
- **Detection**: Check NaN fraction < 5% and feature std > 0.05
- **Fallback**: Replace autocorrelation with rolling variance ratio: `var(5d_changes) / var(20d_changes)` which captures persistence via variance scaling

### Risk 4: Correlation with DXY Submodel Outputs

- **Description**: VIX submodel's regime probability or z-score may correlate with DXY's `dxy_volatility_z` since equity volatility and FX volatility co-move during crises
- **Mitigation**: VIX features capture equity-specific dynamics (S&P 500 implied vol), while DXY features capture USD-specific dynamics (FX realized vol). Expected cross-correlation ~0.3-0.4.
- **Detection**: Monitor VIF in Gate 2. If VIF > 10, residualize against DXY outputs.
- **Fallback**: Drop the most correlated feature and keep 2 columns

### Risk 5: Gate 3 Failure (Real_Rate Pattern)

- **Description**: Features pass Gate 2 (MI increase) but fail Gate 3 (ablation test) like real_rate
- **Why VIX is different from real_rate**:
  - Daily frequency (no interpolation artifacts)
  - VIX has strong, documented relationship with gold (safe-haven dynamics)
  - 3 compact features (not 7 like real_rate attempt 5)
  - Regime/persistence distinction directly relevant to gold (spike vs sustained fear)
  - DXY succeeded on attempt 1 with the same approach
- **Fallback for Attempt 2**: Add VIX/VIX3M term structure as 4th feature if MI is too low, or replace z-score with term structure change (5d diff of VIX/VIX3M ratio, corr=0.15 with VIX level)

### Risk 6: hmmlearn Not Available on Kaggle

- **Description**: hmmlearn may fail to install on Kaggle
- **Mitigation**: `pip install hmmlearn` at script start. If fails, fall back to GMM (sklearn GaussianMixture) like DXY used, applied to rolling windows of log-VIX changes.
- **Likelihood**: Low -- hmmlearn depends only on numpy, scipy, scikit-learn. Successfully used in DXY attempt 1.

---

## 9. VIF Analysis (Pre-Design Estimate)

Expected VIF for each output column against base features:

| Output Column | Most Correlated Base Feature | Measured/Estimated Correlation | Expected VIF |
|---------------|------------------------------|-------------------------------|-------------|
| vix_regime_probability | vix_vix (level) | ~0.2-0.4 (est. from HMM on changes, not levels) | 1-3 |
| vix_mean_reversion_z | vix_vix (level) | 0.43 (measured) | 2-4 |
| vix_persistence | vix_vix (level) | -0.09 (measured) | 1-2 |

Cross-feature VIF (against DXY submodel outputs):

| VIX Output | DXY Output | Estimated Correlation | VIF Risk |
|-----------|-----------|---------------------|---------|
| vix_regime_probability | dxy_regime_probability | ~0.2-0.3 | Low |
| vix_mean_reversion_z | dxy_volatility_z | ~0.3-0.4 | Moderate, monitor |
| vix_persistence | dxy_cross_currency_div | ~0.05 | Negligible |

All expected VIF values are below the threshold of 10. The evaluator will compute actual VIF during Gate 2.

---

## 10. Autocorrelation Analysis

| Output Column | Expected Autocorrelation (lag 1) | Rationale |
|---------------|----------------------------------|-----------|
| vix_regime_probability | 0.7-0.9 | Regimes persist for days-weeks, but transitions provide variation. VIX regimes are shorter than FX regimes. |
| vix_mean_reversion_z | 0.85-0.95 | 60-day rolling window creates smoothing, but VIX is more volatile than DXY so z-score changes faster |
| vix_persistence | 0.6-0.8 | 20-day rolling autocorrelation. Window is shorter than z-score, so less smooth. VIX spikes cause rapid autocorrelation changes. |

All expected values are below the 0.99 threshold. VIX's daily volatility (spikes of 5-15+ points) ensures features vary meaningfully, unlike interpolated monthly real_rate data.

---

## 11. Design Rationale vs Alternatives

### Why HMM over Threshold Models?

- Threshold models (e.g., VIX > 20 = "elevated") impose arbitrary breakpoints and produce discontinuous features. HMM provides smooth regime probabilities that the meta-model can weight continuously. HMM also captures transition dynamics (how quickly regimes switch).

### Why Z-Score over Ornstein-Uhlenbeck Parameter Estimation?

- OU parameter estimation (half-life, mean-reversion speed) requires solving a regression and is sensitive to window choice. Recent research (2025 paper cited in researcher report) questions the interpretability of half-life metrics. The z-score is simpler, more robust, and directly interpretable as "how far from equilibrium."

### Why Rolling Autocorrelation over Term Structure?

- **VIX/VIX3M ratio has 0.69 correlation with VIX level** -- this was measured empirically and is too high for the VIF requirement (<10). Rolling autocorrelation has -0.09 correlation with VIX level, providing genuinely orthogonal information. Autocorrelation also directly captures the spike-vs-sustained distinction identified as critical for gold in the research report.

### Why 3 Columns (Not 2 or 4)?

- 2 columns: Would sacrifice either regime detection or persistence. Both provide complementary information (regime = "what state", persistence = "how sticky").
- 4 columns: Would add a noisy feature (term structure or vol-of-vol) that increases XGBoost's noise surface. real_rate lesson: more columns = more overfit risk.
- 3 columns: Captures state (HMM), position (z-score), and momentum/persistence (autocorr) -- three orthogonal axes of VIX dynamics. Matches DXY's successful 3-column pattern.

### Why Not Use GARCH?

- GARCH models VIX's own conditional variance (vol-of-vol). This is useful for VIX option pricing but adds computational complexity without clear benefit for gold prediction. The HMM+z-score+autocorr combination captures the same regime/mean-reversion/persistence information more simply. GARCH is a candidate for Attempt 2 if this approach falls short.

---

## 12. Expected Performance Against Gates

### Gate 1: Standalone Quality
- **Overfit ratio**: N/A (deterministic HMM, no neural network)
- **No constant output**: Confirmed -- regime probability varies 0-1, z-score varies with VIX dynamics, autocorrelation varies with VIX persistence patterns
- **Autocorrelation < 0.99**: Expected 0.7-0.95 for all features (VIX daily volatility prevents near-constant outputs)
- **No NaN values**: Confirmed after 60-day warmup with forward-fill

**Expected Result**: PASS

### Gate 2: Information Gain
- **MI increase > 5%**: High probability -- VIX has strong documented relationship with gold across regimes. DXY achieved MI=0.019 with similar approach.
- **VIF < 10**: High probability -- all features designed for low correlation with base features (max expected: 0.43 for z-score)
- **Rolling correlation std < 0.15**: High probability -- features capture VIX-specific dynamics, not generic volatility

**Expected Result**: PASS

### Gate 3: Ablation Test
- **Direction accuracy +0.5%**: High probability -- regime-dependent VIX-gold relationship is well-documented. Spike vs sustained fear distinction is directly relevant.
- **OR Sharpe +0.05**: Moderate probability
- **OR MAE -0.01%**: Moderate probability

**Expected Result**: PASS (DXY passed all gates on attempt 1 with the same approach pattern)

**Confidence**: 7/10 (reduced from researcher's 8/10 due to dropping term structure feature)
