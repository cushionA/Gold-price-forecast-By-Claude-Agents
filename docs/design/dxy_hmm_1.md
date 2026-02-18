# Submodel Design Document: DXY (Attempt 1 - Revised)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| Yahoo: DX-Y.NYB | Confirmed | Daily data available, 2545 rows (2015-01-02 to 2025-02-13), latest=107.31 |
| Yahoo: EURUSD=X | Confirmed | Daily data available, latest=1.0397 |
| Yahoo: JPY=X | Confirmed | USD/JPY format (~154), daily data available |
| Yahoo: GBPUSD=X | Confirmed | Daily data available, latest=1.2424 |
| Yahoo: CAD=X | Confirmed | USD/CAD format (~1.45), daily data available |
| Yahoo: SEK=X | Confirmed | USD/SEK format (~11.0), daily data available |
| Yahoo: CHF=X | Confirmed | USD/CHF format (~0.91), daily data available |
| DXY weights sum to 100% | Confirmed | EUR 57.6% + JPY 13.6% + GBP 11.9% + CAD 9.1% + SEK 4.2% + CHF 3.6% = 100.0% |
| Currency direction normalization | CRITICAL | EURUSD=X and GBPUSD=X returns must be negated for USD-strength convention |
| HMM for FX regime detection | Confirmed valid | Well-established in FX literature; proven for daily frequency |
| 20-day realized volatility | Confirmed valid | Industry standard for FX volatility measurement |
| DXY attempt 1 (GMM) output exists | Confirmed | `data/submodel_outputs/dxy/submodel_output.csv` exists but NEVER used in meta-model |
| DXY attempt 1 MI=0.019 | Confirmed | Auto-evaluated, never tested in Gate 2/3 |
| Previous design used GMM not HMM | Issue identified | GMM does not capture temporal transitions between regimes |

### Critical Design Corrections from Previous Attempt

**1. GMM → HMM**: The previous attempt used GaussianMixture (GMM) instead of HMM. GMM treats each observation independently and does not model regime transitions over time. HMM's Markov transition matrix captures regime persistence, which is critical for FX dynamics where regimes last days to weeks.

**2. 3-state HMM instead of 2-state**: Following the proven pattern from VIX (2-3 states via Optuna), cross_asset (3-state MI=0.14), and inflation_expectation (2-3 states via Optuna), we use 3-state HMM as the primary configuration with 2-state as Optuna alternative.

**3. Momentum z-score instead of PCA divergence**: The previous attempt's PCA divergence had many 0.0 values in early periods due to expanding MinMax scaling issues. The simpler and more robust approach is DXY momentum z-score (20-day return, expanding z-score), which follows the proven pattern from vix_mean_reversion_z and tech_mean_reversion_z.

**4. Proper Gate 2/3 testing**: The previous attempt was auto-evaluated with MI=0.019 and marked "completed" without ablation testing. This attempt will undergo full Gate 1/2/3 evaluation.

---

## 1. Overview

- **Purpose**: Extract USD regime dynamics, momentum persistence, and volatility state from DXY to provide the meta-model with dollar structural context that distinguishes trending vs consolidating vs crisis regimes. The raw `dxy_change` base feature (rank #17, 3.50% importance) captures only 1-day directional move with no regime or persistence information.

- **Core methods**:
  1. Hidden Markov Model (3 states) on DXY daily log-returns for regime probability
  2. Z-score of DXY 20-day momentum (expanding window normalization)
  3. Z-score of DXY 20-day realized volatility (expanding window normalization)

- **Why these methods**: Each captures a fundamentally different aspect of USD dynamics. HMM captures which latent regime the dollar is in (trending strength/weakness, consolidation, or crisis). Momentum z-score captures whether USD is in a persistent directional move relative to history. Volatility z-score captures FX turbulence state. None of these are derivable from the 1-day raw change.

- **Expected effect**: Enable the meta-model to distinguish scenarios where the same 1-day DXY change has different gold implications. Example: DXY -0.3% in a trending-weakness regime (strong gold signal) vs DXY -0.3% in a consolidation regime (weak gold signal).

### Key Advantage Over real_rate

All data is daily frequency from Yahoo Finance DX-Y.NYB. No monthly-to-daily interpolation is needed. This eliminates the root cause of all 5 real_rate failures.

### Design Rationale: Following Proven Success Pattern

**Architecture precedent**: This design follows the exact pattern of the 6 most successful submodels:
- vix: 3-feature HMM + z-score (Gate 3 pass: DA +0.96%, Sharpe +0.289)
- cross_asset: 3-feature HMM + z-score (Gate 3 pass: DA +0.76%, MAE -0.0866)
- technical: 3-feature HMM + z-score (Gate 3 pass: MAE -0.1824, 18x threshold)
- etf_flow: 3-feature HMM + z-score (Gate 3 pass: Sharpe +0.377, MAE -0.0436)
- inflation_expectation: 3-feature HMM + z-score (all 3 Gates pass)
- yield_curve: 2-feature deterministic after HMM collapse (Gate 3 pass: MAE -0.069)

**Why NOT use PCA divergence**: The previous DXY attempt's PCA cross-currency divergence had many 0.0 values in the early period, indicating expanding MinMax scaling issues. The simpler momentum z-score is proven to work (vix_mean_reversion_z, tech_mean_reversion_z both succeeded).

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Already Available |
|------|--------|--------|-----------|-------------------|
| DXY Index | Yahoo Finance | DX-Y.NYB | Daily | Yes: `data/raw/dxy.csv` (2545 rows, 2015-01-02 to 2025-02-13) |

### Expansion Data (Optional for Attempt 2)

| Currency Pair | Ticker | DXY Weight | Purpose |
|---------------|--------|------------|---------|
| EUR/USD | EURUSD=X | 57.6% | Reserved for cross-currency divergence in Attempt 2 if needed |
| USD/JPY | JPY=X | 13.6% | Reserved for cross-currency divergence in Attempt 2 if needed |
| GBP/USD | GBPUSD=X | 11.9% | Reserved for cross-currency divergence in Attempt 2 if needed |
| USD/CAD | CAD=X | 9.1% | Reserved for cross-currency divergence in Attempt 2 if needed |
| USD/SEK | SEK=X | 4.2% | Reserved for cross-currency divergence in Attempt 2 if needed |
| USD/CHF | CHF=X | 3.6% | Reserved for cross-currency divergence in Attempt 2 if needed |

**Note**: Attempt 1 uses only DXY itself (single ticker) to minimize complexity and follow the proven 3-feature pattern. Multi-currency data is reserved for Attempt 2 if Attempt 1 fails Gate 3.

### Preprocessing Steps

1. Fetch DX-Y.NYB from Yahoo Finance, start=2014-10-01 (buffer for 120-day warmup before 2015-01-30)
2. Compute daily log-returns: `dxy_log_ret = log(DXY_t) - log(DXY_t-1)`
3. Handle missing values: forward-fill gaps up to 3 days, then drop remaining NaN
4. Trim to base_features date range: 2015-01-30 to 2025-02-12

### Expected Sample Count

- ~2,523 daily observations (matching base_features row count)
- Warmup period: ~120 days for expanding z-score baseline
- Effective output: ~2,400+ rows after warmup, remaining filled with NaN then forward-filled from first valid value

---

## 3. Model Architecture (Hybrid Deterministic-Probabilistic)

This is a **hybrid deterministic-probabilistic** approach. No PyTorch neural network. The pipeline consists of three independent components.

### Component 1: HMM Regime Detection

- **Model**: `hmmlearn.hmm.GaussianHMM`
- **Input**: 1D array of daily DXY log-returns (single feature)
  - Log-transform rationale: Standard practice for FX returns; handles percentage changes correctly
- **States**: 3 (trending-strength / consolidation / trending-weakness) -- Optuna explores 2 vs 3
  - 3 states: Captures the primary FX regime structure (trending up, sideways, trending down)
  - 2 states: Simpler alternative (trending vs consolidation)
- **Covariance type**: "full" (trivial for 1D input; full=diag=spherical for single dimension)
- **Training**: Fit on training set data only. Generate probabilities for full dataset using `predict_proba`.
- **Output**: Posterior probability of the highest-variance state (identified post-hoc by comparing emission variances)
- **State labeling**: After fitting, sort states by emission variance. The highest-variance state corresponds to "trending regime" (large DXY moves). Output P(highest-variance state).

```
Input: DXY daily log-returns [T x 1]
       |
   GaussianHMM.fit(train_data) -> learn 2-3 state model
       |
   GaussianHMM.predict_proba(full_data) -> [T x n_states]
       |
   Select P(highest-variance state) -> [T x 1]
       |
Output: dxy_regime_prob (0-1)
```

### Component 2: Momentum Z-Score

- **Model**: Pure pandas computation (no ML)
- **Input**: DXY closing levels (DX-Y.NYB)
- **Window**: 20-day momentum (Optuna explores 10/15/20/30)
  - 20d rationale: FX momentum typically manifests over weeks, not days. Aligns with industry standard realized vol window.
- **Computation**:
  1. `momentum_20d = (DXY_t / DXY_t-20) - 1` (20-day return)
  2. `z = (momentum_20d - expanding_mean) / expanding_std` (expanding window normalization)
- **Output**: z-score value (unbounded, typically -2 to +3)
  - Positive: DXY above recent momentum trend (strengthening accelerating)
  - Negative: DXY below recent momentum trend (weakening or consolidating)
  - This is a residual from equilibrium momentum, not the level itself

```
Input: DXY daily close levels [T x 1]
       |
   momentum = (DXY_t / DXY_t-window) - 1
       |
   expanding_mean = momentum.expanding().mean()
   expanding_std  = momentum.expanding().std()
       |
   z = (momentum - expanding_mean) / expanding_std
       |
   clip(-4, 4) for stability
       |
Output: dxy_momentum_z (typically -2 to +3)
```

### Component 3: Volatility Z-Score

- **Model**: Pure pandas computation (no ML)
- **Input**: Daily DXY log-returns
- **Window**: 20-day rolling window (Optuna explores 10/15/20/30)
  - 20d rationale: Industry standard for realized FX volatility
- **Computation**:
  1. `vol_20d = dxy_log_ret.rolling(20).std()`
  2. `z = (vol_20d - expanding_mean) / expanding_std` (expanding window normalization)
- **Output**: z-score value (unbounded, typically -2 to +3)
  - Positive: FX volatility elevated (crisis/turbulence regime)
  - Negative: FX volatility subdued (calm regime)

```
Input: DXY daily log-returns [T x 1]
       |
   vol = dxy_log_ret.rolling(window).std()
       |
   expanding_mean = vol.expanding().mean()
   expanding_std  = vol.expanding().std()
       |
   z = (vol - expanding_mean) / expanding_std
       |
   clip(-4, 4) for stability
       |
Output: dxy_vol_z (typically -2 to +3)
```

### Combined Output

| Column | Range | Description | Expected Autocorr(1) | Expected VIF |
|--------|-------|-------------|---------------------|-------------|
| `dxy_regime_prob` | [0, 1] | P(high-variance/trending regime) from 3-state HMM | 0.70-0.85 | < 3 |
| `dxy_momentum_z` | [-4, +4] | 20-day momentum z-score (expanding normalization) | 0.80-0.90 | < 3 |
| `dxy_vol_z` | [-4, +4] | 20-day realized vol z-score (expanding normalization) | 0.80-0.90 | < 4 |

Total: **3 columns** (matching the proven compact pattern from all successful submodels).

### Three Orthogonal Dimensions

1. **Regime** (dxy_regime_prob): WHICH DXY state are we in? (Trending strength / consolidation / trending weakness)
2. **Momentum** (dxy_momentum_z): Is USD in a persistent directional move relative to historical norms?
3. **Volatility** (dxy_vol_z): Is FX volatility elevated or subdued relative to historical norms?

These capture fundamentally different information:
- Regime is a state variable (persistent, autocorr 0.7-0.85)
- Momentum is a position variable (moderately persistent, autocorr 0.8-0.9)
- Volatility is a dispersion variable (moderately persistent, autocorr 0.8-0.9)

Expected inter-feature correlation: < 0.35 (different dimensions of USD dynamics).

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM n_iter | 200 | Sufficient for EM convergence on FX returns. Increased from previous 100 to ensure convergence. |
| HMM tol | 1e-4 | Standard convergence tolerance |
| HMM random_state | 42 | Reproducibility |
| Z-score clipping | [-4, 4] | Prevent extreme outliers from dominating |
| Z-score normalization | expanding window | Prevents lookahead bias; uses only past data |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3} | categorical | 2=trending/consolidation, 3=trending-strength/consolidation/trending-weakness. Empirical: VIX/cross_asset succeeded with 3 states. |
| hmm_covariance_type | {"full", "diag"} | categorical | Trivial for 1D but let Optuna test both |
| hmm_n_init | {5, 10, 15} | categorical | Number of EM restarts to avoid local optima. Increased from previous {3, 5, 10} for better stability. |
| momentum_window | {10, 15, 20, 30} | categorical | 10d=reactive, 20d=standard, 30d=smooth. Corresponds to 2-week, 1-month, 1.5-month FX momentum. |
| vol_window | {10, 15, 20, 30} | categorical | Same range as momentum for consistency |
| expanding_warmup | {60, 120, 252} | categorical | Expanding z-score warmup period: 60d=3mo, 120d=6mo, 252d=1yr. Determines when z-score normalization becomes stable. |

### Exploration Settings

- **n_trials**: 30
  - Rationale: 6 categorical parameters. Total combinations: 2 * 2 * 3 * 4 * 4 * 3 = 576. 30 trials with TPE sampler provides good coverage. Each trial is fast (<10 seconds: HMM fit on ~1,766 x 1 observations).
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of mutual information between 3 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)

---

## 5. Training Settings

### Fitting Procedure

This is not a gradient-based training loop. The procedure is:

1. **HMM**: `GaussianHMM.fit(dxy_log_returns_train)` -- EM algorithm, converges in seconds
2. **Momentum Z-Score**: Expanding window statistics on momentum -- deterministic, no fitting
3. **Volatility Z-Score**: Rolling window std + expanding z-score -- deterministic, no fitting

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- train: ~1,766 observations
- val: ~378 observations
- test: ~379 observations (reserved for evaluator Gate 3)
- HMM fits on training set only
- HMM generates probabilities for full dataset using predict_proba (no lookahead: HMM posterior at time t depends only on observations up to t given fitted model)
- Z-scores use expanding windows (inherently backward-looking, no lookahead)
- Optuna optimizes MI sum on validation set

### Evaluation Metric for Optuna

For each trial (hyperparameter combination):
1. Fit HMM on training set DXY log-returns
2. Generate all 3 features for full dataset using fitted HMM and trial window parameters
3. Compute mutual information (MI) between each of the 3 features and `gold_return_next` on validation set
4. Optuna maximizes: `MI_sum = MI(regime, target) + MI(momentum_z, target) + MI(vol_z, target)`

MI calculation method: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`. This matches the proven approach from VIX, cross_asset, inflation_expectation.

### Loss Function

N/A -- no gradient-based training. HMM uses EM (maximum likelihood). Z-scores are deterministic.

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- HMM EM converges via `n_iter` and `tol`. No early stopping needed.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. HMM + rolling statistics are CPU-only |
| Estimated execution time | 3-5 minutes | Data download (~30s) + 30 Optuna trials x ~5s each (~2.5min) + final output (~30s) |
| Estimated memory usage | < 1 GB | ~2,545 rows x 1 column of returns + rolling stats. Tiny dataset |
| Required pip packages | `hmmlearn` | Must `pip install hmmlearn` at start of train.py. sklearn, pandas, numpy, yfinance pre-installed on Kaggle |

---

## 7. Implementation Instructions

### builder_data Instructions

1. **DXY data already available**: Use existing `data/raw/dxy.csv` (DX-Y.NYB from Yahoo Finance)
2. **Verify format**: Ensure columns include Date and Close value
3. **Verify date range**: 2015-01-02 to 2025-02-13 (2545 rows), covers base_features range 2015-01-30 to 2025-02-12
4. **No additional data fetching required**: 6 constituent currencies are NOT used in Attempt 1
5. **Save preprocessed data**: `data/processed/dxy_features_input.csv`
   - Columns: Date, dxy_close, dxy_log_ret
   - Start from 2014-10-01 (warmup buffer)
6. **Quality checks**:
   - No gaps > 3 consecutive trading days
   - Missing data < 1%
   - DXY values in reasonable range (70-130)
   - Log-returns have no extreme outliers (|log_ret| < 0.05 i.e. ~5% daily move)

### builder_model Instructions

#### train.py Structure

```python
"""
Gold Prediction SubModel Training - DXY Attempt 1 (Revised)
Self-contained: Data fetch -> Preprocessing -> HMM + Momentum Z + Vol Z -> Optuna HPO -> Save results
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
# Fetch DX-Y.NYB close prices from Yahoo Finance
# Compute log-returns
# Align to base_features date range

# === 3. Feature Generation Functions ===

def generate_regime_feature(dxy_log_returns, n_components, covariance_type, n_init, train_size):
    """
    Fit HMM on train portion and return P(highest-variance state) for full data.

    CRITICAL: Use proper HMM (not GMM) to capture temporal regime transitions.
    """
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=200,
        tol=1e-4,
        random_state=42,
        n_init=n_init
    )
    X_train = dxy_log_returns[:train_size].values.reshape(-1, 1)
    X_full = dxy_log_returns.values.reshape(-1, 1)

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

def generate_momentum_feature(dxy_close, momentum_window, expanding_warmup):
    """
    Expanding z-score of N-day momentum (N-day return).

    Follows proven pattern from vix_mean_reversion_z, tech_mean_reversion_z.
    """
    s = pd.Series(dxy_close)
    momentum = s.pct_change(momentum_window)  # N-day return

    # Expanding z-score (no lookahead)
    result = np.full(len(s), np.nan)
    for i in range(expanding_warmup, len(s)):
        past_vals = momentum.iloc[:i+1]
        past_valid = past_vals.dropna()
        if len(past_valid) >= 30:  # Minimum 30 observations for stable mean/std
            mean = past_valid.mean()
            std = past_valid.std()
            if std > 1e-10:
                result[i] = (momentum.iloc[i] - mean) / std
            else:
                result[i] = 0.0
        else:
            result[i] = 0.0

    # Clip extreme outliers
    result = np.clip(result, -4, 4)
    return result

def generate_volatility_feature(dxy_log_returns, vol_window, expanding_warmup):
    """
    Expanding z-score of N-day realized volatility.

    Industry standard: 20-day rolling std for FX volatility.
    """
    vol = dxy_log_returns.rolling(vol_window).std()

    # Expanding z-score (no lookahead)
    result = np.full(len(vol), np.nan)
    for i in range(expanding_warmup, len(vol)):
        past_vals = vol.iloc[:i+1]
        past_valid = past_vals.dropna()
        if len(past_valid) >= 30:
            mean = past_valid.mean()
            std = past_valid.std()
            if std > 1e-10:
                result[i] = (vol.iloc[i] - mean) / std
            else:
                result[i] = 0.0
        else:
            result[i] = 0.0

    # Clip extreme outliers
    result = np.clip(result, -4, 4)
    return result

# === 4. Optuna Objective ===

def objective(trial, dxy_log_returns, dxy_close, target, train_size, val_mask):
    """Maximize MI sum on validation set"""
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    covariance_type = trial.suggest_categorical('hmm_covariance_type', ['full', 'diag'])
    n_init = trial.suggest_categorical('hmm_n_init', [5, 10, 15])
    momentum_window = trial.suggest_categorical('momentum_window', [10, 15, 20, 30])
    vol_window = trial.suggest_categorical('vol_window', [10, 15, 20, 30])
    expanding_warmup = trial.suggest_categorical('expanding_warmup', [60, 120, 252])

    try:
        regime = generate_regime_feature(
            dxy_log_returns, n_components, covariance_type, n_init, train_size
        )
        momentum_z = generate_momentum_feature(dxy_close, momentum_window, expanding_warmup)
        vol_z = generate_volatility_feature(dxy_log_returns, vol_window, expanding_warmup)

        # Extract validation period
        regime_val = regime[val_mask]
        momentum_val = momentum_z[val_mask]
        vol_val = vol_z[val_mask]
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

        for feat_val in [regime_val, momentum_val, vol_val]:
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
2. **HMM not GMM**: Use `GaussianHMM` (captures temporal transitions), NOT `GaussianMixture` (treats observations independently). This is the critical correction from previous attempt.
3. **HMM state labeling**: After fitting, sort states by emission variance. Output P(highest-variance state). Do NOT assume state index 0 or 1 corresponds to any regime.
4. **Expanding z-score warmup**: The `expanding_warmup` parameter determines when z-score normalization begins. Before warmup, output NaN (to be forward-filled). Optuna tests 60d (3mo), 120d (6mo), 252d (1yr).
5. **No lookahead bias**:
   - HMM: Fit on training data only, generate probabilities for full dataset
   - Momentum z-score: Expanding window (uses only past data at each time t)
   - Volatility z-score: Expanding window (uses only past data at each time t)
6. **NaN handling**: First ~252 rows (max of expanding_warmup values) may have NaN. Forward-fill output after generation. The evaluator will align dates with base_features.
7. **Reproducibility**: Fix random_state=42 for HMM, seed=42 for Optuna. Z-scores are deterministic given window parameters.
8. **Output format**: CSV with columns [Date, dxy_regime_prob, dxy_momentum_z, dxy_vol_z]. Aligned to trading dates matching base_features.
9. **Target data**: Load gold_return_next from the Kaggle dataset `bigbigzabuton/gold-prediction-submodels` (same as all other submodels).
10. **Single ticker simplicity**: Attempt 1 uses only DX-Y.NYB. No multi-currency complexity. This follows the principle of simplest-first, matching VIX (single ticker) and technical (single GC=F ticker).

---

## 8. Risks and Alternatives

### Risk 1: HMM State Instability

- **Description**: HMM may produce different state assignments depending on initialization, especially with only ~1,766 training observations
- **Mitigation**: Use `n_init` parameter (5-15 random restarts, increased from previous 3-10). Optuna tests multiple values. Fix random_state=42 for reproducibility.
- **Detection**: Check that regime probabilities have std > 0.1 and are not near-constant
- **Fallback**: If regime probability has std < 0.05, replace with a simpler sigmoid-based regime indicator: `regime = sigmoid((DXY_20d_momentum / std_20d) * 2)` which approximates trend probability without HMM

### Risk 2: Momentum Z-Score High Autocorrelation

- **Description**: 20-day momentum inherently has overlap (19/20 days shared between consecutive observations), which could cause autocorrelation > 0.99
- **Mitigation**: Expanding z-score normalization reduces autocorrelation by removing long-term drift. Expected lag-1 autocorrelation: 0.80-0.90, well below 0.99. Optuna tests shorter windows (10d, 15d) if needed.
- **Detection**: Verify autocorrelation < 0.99 in Gate 1
- **Fallback**: If autocorrelation too high, use daily return z-score instead of momentum (autocorr ~0.0)

### Risk 3: Volatility Z-Score Correlated with VIX

- **Description**: `dxy_vol_z` might correlate with `vix_mean_reversion_z` or `vix_persistence`, causing VIF issues
- **Mitigation**: DXY volatility captures FX turbulence (USD-specific), while VIX captures equity volatility (S&P 500 implied vol). Different instruments, different dynamics. Expected cross-correlation ~0.3-0.4.
- **Detection**: Monitor VIF in Gate 2. If VIF > 10, residualize against VIX outputs.
- **Fallback**: Drop dxy_vol_z and keep 2 columns (regime + momentum)

### Risk 4: Gate 3 Failure (Real_Rate Pattern)

- **Description**: Features pass Gate 2 (MI increase) but fail Gate 3 (ablation test) like real_rate
- **Why DXY is different from real_rate**:
  - Daily frequency (no interpolation artifacts)
  - DXY has strong, documented inverse relationship with gold
  - 3 compact features (not 7 like real_rate attempt 5)
  - Regime/momentum/volatility distinction directly relevant to gold (trending USD vs consolidation)
  - VIX and cross_asset succeeded on attempt 1 with the same HMM + z-score approach
- **Fallback for Attempt 2**: Add 6-currency PCA divergence as 4th feature, or replace momentum_z with PCA divergence (3 columns total)

### Risk 5: Previous Attempt's Auto-Evaluation Bias

- **Description**: DXY attempt 1 was auto-evaluated with MI=0.019 and marked "completed" without Gate 2/3 testing. This may have hidden real issues.
- **Mitigation**: This attempt undergoes full Gate 1/2/3 evaluation with proper ablation testing. No shortcuts.
- **Expectation**: MI may be higher than 0.019 due to HMM (vs GMM) capturing temporal structure.

---

## 9. VIF Analysis (Pre-Design Estimate)

Expected VIF for each output column against base features:

| Output Column | Most Correlated Base Feature | Expected Correlation | Expected VIF |
|---------------|------------------------------|---------------------|-------------|
| dxy_regime_prob | dxy_change (1-day) | ~0.15 (regime vs 1-day change) | 1-2 |
| dxy_momentum_z | dxy_change (1-day) | ~0.30 (20-day vs 1-day) | 2-3 |
| dxy_vol_z | vix_mean_reversion_z | ~0.30 (FX vol vs equity vol) | 2-4 |

All expected VIF values are well below the threshold of 10. The evaluator will compute actual VIF during Gate 2.

Cross-feature VIF (against existing submodel outputs):

| DXY Output | Submodel Output | Estimated Correlation | VIF Risk |
|-----------|-----------------|---------------------|---------|
| dxy_regime_prob | vix_regime_probability | ~0.20 | Low |
| dxy_momentum_z | tech_mean_reversion_z | ~0.25 | Low |
| dxy_vol_z | vix_persistence | ~0.30 | Moderate, monitor |

All expected VIF values are below 10. Maximum expected: dxy_vol_z at VIF ~4 due to cross-correlation with VIX-related features.

---

## 10. Autocorrelation Analysis

| Output Column | Expected Autocorrelation (lag 1) | Rationale |
|---------------|----------------------------------|-----------|
| dxy_regime_prob | 0.70-0.85 | Regimes persist for days-weeks, but transitions provide variation. HMM with 3 states has more frequent transitions than 2 states. |
| dxy_momentum_z | 0.80-0.90 | 20-day momentum has inherent overlap, but expanding z-score normalization adds variation. Shorter than VIX's 60-day z-score window. |
| dxy_vol_z | 0.80-0.90 | 20-day rolling vol is smooth but z-score normalization adds variation. DXY volatility changes faster than interpolated monthly data. |

All expected values are below the 0.99 threshold. DXY's daily volatility (typical daily move 0.2-0.5%) ensures features vary meaningfully, unlike interpolated monthly real_rate data.

---

## 11. Design Rationale vs Alternatives

### Why HMM over GMM (Previous Attempt's Mistake)?

- **GMM treats each observation independently**: GaussianMixture assigns each day to a cluster based solely on that day's return, ignoring temporal structure.
- **HMM captures regime persistence**: GaussianHMM uses a Markov transition matrix to model how regimes persist over time. This is critical for FX dynamics where trending regimes last days-weeks.
- **Empirical evidence**: Cross_asset's 3D HMM achieved MI=0.14 (higher than any VIX/technical feature individually). VIX's HMM succeeded. GMM would not capture this temporal structure.

### Why Momentum Z-Score over PCA Divergence?

- **Simplicity**: Momentum z-score requires only DXY (1 ticker). PCA divergence requires 6 currency pairs (7 tickers total).
- **Proven pattern**: vix_mean_reversion_z and tech_mean_reversion_z both succeeded with z-score approach. This is the #1 most successful feature type.
- **Previous attempt's issue**: PCA divergence had many 0.0 values in early period (expanding MinMax scaling issue). Momentum z-score avoids this.
- **Reserve for Attempt 2**: If momentum z-score proves insufficient, Attempt 2 can add PCA divergence as 4th feature or replace momentum_z.

### Why 3 States (Not 2)?

- **VIX precedent**: VIX tested 2 vs 3 states via Optuna; final choice depended on MI.
- **Cross_asset precedent**: 3-state HMM achieved MI=0.14; 2-state achieved MI=0.11. Both acceptable but 3 states superior.
- **FX regime structure**: USD exhibits 3 clear regimes: trending strength (2014-2015, 2022), consolidation (2015-2020), trending weakness (2020-2021, 2024-2025).
- **Optuna decides**: Let Optuna test both 2 and 3 states. Starting hypothesis is 3 states based on precedent.

### Why 3 Columns (Not 2 or 4)?

- 2 columns: Would sacrifice either momentum or volatility. Both provide complementary information (position vs dispersion).
- 4 columns: Would require adding PCA divergence. Real_rate lesson: more columns = more overfit risk.
- 3 columns: Captures regime (state), momentum (position), volatility (dispersion) -- three orthogonal axes of USD dynamics. Matches VIX/cross_asset/inflation_expectation successful pattern.

### Why NOT Multi-Currency in Attempt 1?

- **Simplicity first**: VIX succeeded with single ticker (^VIX). Technical succeeded with single ticker (GC=F). Follow proven pattern.
- **Complexity deferred**: 6-currency PCA adds data fetching complexity (7 tickers total), direction normalization complexity (EURUSD/GBPUSD must be negated), and feature engineering complexity (rolling PCA).
- **Incremental improvement**: If Attempt 1 passes Gate 3, no need for complexity. If Attempt 1 fails, Attempt 2 adds multi-currency.

---

## 12. Expected Performance Against Gates

### Gate 1: Standalone Quality
- **Overfit ratio**: N/A (deterministic HMM, no neural network with train/val loss comparison)
- **No constant output**: High confidence -- HMM has clear regime structure (2014-2015 USD strength, 2020-2021 USD weakness), momentum varies with directional moves, volatility spikes during crises
- **Autocorrelation < 0.99**: Expected 0.70-0.90 for all features (daily DXY volatility prevents near-constant outputs)
- **No NaN values**: Confirmed after expanding_warmup period with forward-fill

**Expected Result**: PASS

### Gate 2: Information Gain
- **MI increase > 5%**: High probability -- DXY has strong documented inverse relationship with gold. Regime context should provide substantial nonlinear information. Previous attempt achieved MI=0.019 with GMM; HMM should exceed this.
- **VIF < 10**: High probability -- all features designed for low correlation with base features (max expected: 0.30 for momentum_z vs dxy_change)
- **Rolling correlation std < 0.15**: Moderate probability -- regime features naturally have time-varying correlation (consistent with all 6 successful submodels having marginal rolling correlation std)

**Expected Result**: PASS (VIF and MI highly likely; rolling correlation std may be marginal but precedent accepts this)

### Gate 3: Ablation Test
- **Direction accuracy +0.5%**: High probability -- regime-dependent DXY-gold relationship is well-documented. Trending vs consolidation distinction is directly relevant.
- **OR Sharpe +0.05**: Moderate probability
- **OR MAE -0.01%**: Moderate probability

**Expected Result**: Cautiously optimistic (7/10 confidence). DXY is gold's strongest inverse correlator (base feature rank #17 understates importance). Daily frequency avoids real_rate's failure mode. HMM + z-score pattern has 100% Gate 3 pass rate across 6 submodels (VIX, cross_asset, technical, etf_flow, inflation_expectation, yield_curve).

**Confidence**: 7/10 (moderate-high). The main uncertainty is whether the single-ticker simplicity provides enough signal vs multi-currency complexity, but the precedent (VIX single ticker succeeded) supports this approach.
