# Submodel Design Document: Options Market (Attempt 2)

## 0. Fact-Check Results

| Claim / Source | Result | Detail |
|----------------|--------|--------|
| Yahoo: ^SKEW (CBOE SKEW Index) | CONFIRMED (re-verified) | 4 rows for 2026-02-10 to 2026-02-13. Latest: 139.35 (2026-02-13). |
| FRED: GVZCLS (Gold Vol Index) | CONFIRMED (re-verified) | 3 rows for 2026-02-10 to 2026-02-12. Latest: 30.23 (2026-02-12). |
| Yahoo: ^GVZ (Gold Vol backup) | CONFIRMED (re-verified) | 4 rows for 2026-02-10 to 2026-02-13. Matches FRED GVZCLS. |
| Attempt 1 HMM architecture | CONFIRMED working | 30 Optuna trials completed, best MI sum = 1.728. HMM converged. |
| Attempt 1 regime_prob value | CONFIRMED high quality | Rank #2/22, importance 5.70%, MI 0.031, autocorr 0.873 |
| Attempt 1 tail_risk_z value | CONFIRMED noise | MI 0.002 (near zero), rank #13/22 |
| Attempt 1 skew_momentum_z value | CONFIRMED marginal | MI 0.017, rank #15/22, importance 4.18% |
| Method: Single-output HMM | VALID | Consistent with successful submodels (vix, technical, yield_curve all have regime_prob as strongest output) |
| Data source: No FRED dependency needed | CONFIRMED | GVZ available from Yahoo Finance ^GVZ as backup. SKEW from Yahoo Finance ^SKEW. FRED GVZCLS is optional enhancement. |

### Design Corrections from Attempt 1

**1. Output dimensionality is the root cause of Gate 3 failure.** Attempt 1 produced 3 columns. DA degraded in ALL 5 folds (-1.05% average), MAE increased (+0.018), Sharpe dropped (-0.234). This matches the pattern seen in real_rate attempts 4-5 where more output columns gave XGBoost more noise to overfit on. Gate 2 PASS (MI +17.12%) confirms the information exists; it is the delivery format (3 noisy columns) that causes harm.

**2. Reduce to single column output.** `options_risk_regime_prob` ranked #2/22 in feature importance (5.70%) and has the highest individual MI (0.031). The other two columns (`options_tail_risk_z` MI=0.002, `options_skew_momentum_z` MI=0.017) collectively add noise. Dropping them removes 2 dimensions that XGBoost can overfit on while preserving the strongest signal.

**3. Optimize HMM specifically for regime_prob MI.** Attempt 1's Optuna objective maximized sum of MI across 3 columns. This meant the HMM was partly optimized for features we now discard. Attempt 2 optimizes solely for regime_prob MI, potentially yielding a better HMM configuration.

---

## 1. Overview

- **Purpose**: Extract a single contextual feature -- the probability of being in a high-volatility options risk regime -- from CBOE SKEW Index and Gold Volatility Index (GVZ) data. This captures the joint state of equity tail risk perception and gold-specific implied volatility, providing a compact nonlinear regime indicator for the meta-model.
- **Core method**: Hidden Markov Model (2-3 states) on 2D [SKEW daily changes, GVZ daily changes], outputting P(highest-variance state).
- **Why this method**: HMM captures joint regime patterns between equity tail risk (SKEW) and gold-specific uncertainty (GVZ) that are invisible in raw levels or linear statistics. This exact pattern (2D HMM regime detection) has succeeded in 5/8 completed submodels (VIX, Technical, Cross-Asset, ETF-Flow, Inflation Expectation).
- **Key change from Attempt 1**: Output reduced from 3 columns to 1. Only `options_risk_regime_prob` is retained (rank #2/22, MI 0.031). `options_tail_risk_z` (MI 0.002, noise) and `options_skew_momentum_z` (MI 0.017, marginal) are dropped entirely.
- **Expected effect**: By delivering only the highest-quality signal in a single column, the meta-model gains the options-derived risk context without the noise penalty that caused Gate 3 failure in Attempt 1.

### Why Single Column?

Evidence from this project strongly supports minimal output dimensionality:

| Submodel | Output Columns | Gate 3 Result | Notes |
|----------|---------------|---------------|-------|
| real_rate attempt 5 | 7 columns | FAIL | Highest MI (+39.3%) but worst Gate 3 degradation |
| real_rate attempt 4 | 2 columns | FAIL | MI +10.29%, still degraded all metrics |
| options_market attempt 1 | 3 columns | FAIL | MI +17.12%, DA degraded ALL 5 folds |
| vix | 3 columns | PASS | But regime_prob alone was 14.7% importance |
| technical | 3 columns | PASS | regime_prob alone was 7.96% importance |
| inflation_expectation | 3 columns | PASS | All 3 gates passed |

The successful 3-column submodels had ALL 3 columns contributing meaningfully. In options_market attempt 1, only 1 of 3 columns had strong MI (0.031 vs 0.002 and 0.017). The weak columns dragged performance down.

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Delay | Status |
|------|--------|--------|-----------|-------|--------|
| SKEW Index | Yahoo Finance | ^SKEW | Daily | 0 days | CONFIRMED |
| GVZ (Gold Vol) | FRED (primary) | GVZCLS | Daily | 0-1 days | CONFIRMED |
| GVZ (backup) | Yahoo Finance | ^GVZ | Daily | 0 days | CONFIRMED |

**No FRED dependency**: GVZ is available from Yahoo Finance (^GVZ) as a reliable fallback. The notebook should try FRED first (via Kaggle Secrets FRED_API_KEY) and fall back to Yahoo Finance if FRED fails, ensuring robustness.

### Preprocessing Steps

1. Fetch SKEW from Yahoo Finance (^SKEW), start=2014-10-01 (buffer for warmup), end=2025-02-15
2. Fetch GVZCLS from FRED (primary) or ^GVZ from Yahoo Finance (fallback), same date range
3. Compute daily changes: `skew_change = SKEW_t - SKEW_{t-1}`, `gvz_change = GVZ_t - GVZ_{t-1}`
4. Handle missing values: forward-fill gaps up to 3 days, then drop remaining NaN
5. Align SKEW and GVZ on common trading dates (inner join on date index)
6. Trim to base_features date range: 2015-01-30 to 2025-02-12 (plus warmup buffer)

### Expected Sample Count

- ~2,523 daily observations (matching base_features row count after alignment)
- Warmup: First row lost to diff operation
- Effective HMM input: ~2,730+ rows (2014-10-01 to 2025-02-12)
- Effective output: ~2,523 rows aligned to base_features

### Data Quality Expectations

| Metric | SKEW (^SKEW) | GVZ (GVZCLS/^GVZ) |
|--------|-------------|---------------------|
| Mean | ~135 | ~20-25 |
| Std | ~12 | ~8-10 |
| Min | ~110 | ~10 |
| Max | ~183 | ~60 |
| Autocorr(1) level | ~0.95 | ~0.95 |
| Autocorr(1) change | ~0.15 | ~0.15 |

---

## 3. Model Architecture

This is a **single-component HMM** approach. No neural network. No PyTorch required.

### Component: 2D HMM Regime Detection

- **Model**: `hmmlearn.hmm.GaussianHMM`
- **Input**: 2D array of [SKEW daily changes, GVZ daily changes]
  - SKEW changes: Captures shifts in equity tail risk perception
  - GVZ changes: Captures shifts in gold-specific implied volatility
  - Cross-correlation of these changes: r = -0.03 (essentially independent axes, confirmed in attempt 1)
- **States**: 2 or 3 (Optuna selects)
  - 2 states: Normal vs elevated-risk
  - 3 states: Calm vs moderate-risk vs extreme-risk
- **Covariance type**: "full" (captures any correlation structure between SKEW and GVZ changes)
- **Training**: Fit on training set data only. Generate probabilities for full dataset using `predict_proba`.
- **State labeling**: After fitting, compute trace of covariance matrix for each state. Highest-trace state = "extreme risk regime" (large moves in both SKEW and GVZ). Output P(highest-trace state).
- **Output**: `options_risk_regime_prob` in [0, 1]

```
Input: [SKEW daily changes, GVZ daily changes] [T x 2]
       |
   GaussianHMM.fit(train_data) -> learn 2-3 state model
       |
   GaussianHMM.predict_proba(full_data) -> [T x n_states]
       |
   Compute trace of covars for each state
   Select P(highest-trace state) -> [T x 1]
       |
Output: options_risk_regime_prob (single column, range 0-1)
```

### Combined Output

| Column | Range | Description |
|--------|-------|-------------|
| `options_risk_regime_prob` | [0, 1] | P(extreme risk regime) from 2D HMM on [SKEW changes, GVZ changes] |

**Total: 1 column** (reduced from 3 in attempt 1).

### Removed Components (with rationale)

| Component | Attempt 1 Output | MI | Importance | Decision |
|-----------|------------------|----|------------|----------|
| SKEW tail risk z-score | `options_tail_risk_z` | 0.002 | 4.24% (#13) | **DROPPED** -- MI near zero, likely pure noise |
| SKEW momentum z-score | `options_skew_momentum_z` | 0.017 | 4.18% (#15) | **DROPPED** -- Moderate MI but low importance; adds an overfitting dimension |

### Orthogonality Analysis

With only 1 output column, VIF risk is minimal:

| Feature Pair | Expected Correlation | Assessment |
|-------------|---------------------|------------|
| options_risk_regime_prob vs vix_regime_probability | est. 0.1-0.3 | Acceptable (different HMM inputs: 1D log-VIX vs 2D SKEW+GVZ) |
| options_risk_regime_prob vs tech_trend_regime_prob | est. 0.1-0.2 | Acceptable (completely different data) |
| options_risk_regime_prob vs any base feature | est. <0.3 | Acceptable (regime probability is nonlinear transformation) |

Attempt 1 confirmed VIF max 2.12 for this column specifically (VIF 1.87). With fewer total columns, VIF can only improve.

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM n_iter | 100 | Standard convergence limit; EM typically converges in 20-50 iterations |
| HMM tol | 1e-4 | Standard convergence tolerance |
| HMM random_state | 42 | Reproducibility |
| HMM covariance_type | "full" | Required for 2D input to capture cross-correlation structure |
| Z-score clipping | N/A | No z-score features in attempt 2 |
| Output columns | 1 | Single regime_prob column |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3} | categorical | 2=normal/extreme, 3=calm/moderate/extreme. Attempt 1 best: 3. |
| hmm_n_init | {3, 5, 10} | categorical | EM restarts to avoid local optima. More restarts = more stable fit. |
| input_scaling | {true, false} | categorical | Whether to standardize [skew_change, gvz_change] before HMM fit. NEW: may improve regime separation when change scales differ. |

**Changes from Attempt 1:**
- **Removed** `skew_zscore_window` and `skew_momentum_window` -- these parameters controlled the dropped features.
- **Added** `input_scaling` -- SKEW changes (std ~5-8) and GVZ changes (std ~2-4) have different scales. Standardizing before HMM may improve regime separation.

### Exploration Settings

- **n_trials**: 30
  - Rationale: 2 * 3 * 2 = 12 total combinations. 30 trials with TPE provides full coverage with repeated evaluations of promising configs (important since HMM has stochastic initialization even with fixed random_state).
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize MI between `options_risk_regime_prob` and `gold_return_next` on validation set
  - **Key change from Attempt 1**: Objective is now MI of single column, not sum of 3. This focuses optimization on the regime probability quality.
- **direction**: maximize
- **sampler**: TPESampler(seed=42)
- **pruner**: None (trials too fast to benefit from pruning)

---

## 5. Training Settings

### Fitting Procedure

This is NOT a gradient-based training loop. The procedure is:

1. **HMM**: `GaussianHMM.fit(hmm_input_train)` -- EM algorithm on 2D [SKEW changes, GVZ changes], converges in seconds
2. Optionally standardize input if `input_scaling=True` (fit StandardScaler on train, transform full data)

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- HMM fits on training set data only
- HMM generates probabilities for full dataset using `predict_proba` (no lookahead: HMM posterior at time t depends only on observations up to t given fitted model)
- Optuna optimizes MI on validation set
- Test set reserved for evaluator Gate 3

### Evaluation Metric for Optuna

For each trial (hyperparameter combination):
1. Optionally standardize input (if `input_scaling=True`): fit StandardScaler on train portion, transform full data
2. Fit HMM on training set [SKEW changes, GVZ changes] (or standardized equivalents)
3. Generate `options_risk_regime_prob` for full dataset
4. Compute mutual information between `options_risk_regime_prob` and `gold_return_next` on validation set
5. Optuna maximizes: `MI(regime_prob, target)` (single column, not sum)

MI calculation method: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`. This matches all previous submodels.

### Loss Function

N/A -- no gradient-based training. HMM uses EM (maximum likelihood).

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- HMM EM converges via `n_iter` and `tol`.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. HMM + statistics are CPU-only. |
| Estimated execution time | 3-5 minutes | Data download (~30s) + 30 Optuna trials x ~3s each (~1.5min) + final output (~30s). Faster than attempt 1 (no z-score/momentum computation). |
| Estimated memory usage | <1 GB | ~2,700 rows x 2-3 columns. Tiny dataset. |
| Required pip packages | hmmlearn | Must `pip install hmmlearn` at start of notebook. sklearn, pandas, numpy pre-installed on Kaggle. |

---

## 7. Implementation Instructions

### builder_data Instructions

**Same as attempt 1** -- the input data has not changed. If `data/processed/options_market_features_input.csv` already exists and passes datachecker, reuse it.

1. **Fetch SKEW data**: Download ^SKEW from Yahoo Finance, start=2014-10-01 (buffer), end=2025-02-15
2. **Fetch GVZ data**: Download GVZCLS from FRED (primary). If FRED fails, use ^GVZ from Yahoo Finance.
3. **Align dates**: Inner join SKEW and GVZ on date index. Forward-fill gaps up to 3 days.
4. **Compute derived columns**:
   - `skew_change = SKEW_t - SKEW_{t-1}`
   - `gvz_change = GVZ_t - GVZ_{t-1}`
5. **Save preprocessed data**: `data/processed/options_market_features_input.csv`
   - Columns: Date (index), skew_close, gvz_close, skew_change, gvz_change
   - Date range: 2014-10-01 to 2025-02-12 (includes warmup buffer)
6. **Quality checks**:
   - No gaps > 3 consecutive trading days
   - Missing data < 2% after forward-fill
   - SKEW values in range [100, 200]
   - GVZ values in range [5, 80]
   - Changes (daily diffs) have no extreme outliers (|SKEW change| < 30, |GVZ change| < 20)

### builder_model Instructions

#### Notebook Structure

```python
"""
Gold Prediction SubModel Training - Options Market Attempt 2
Self-contained: Data fetch -> Preprocessing -> HMM Regime Detection -> Optuna HPO -> Save results

KEY CHANGE FROM ATTEMPT 1:
- Output reduced from 3 columns to 1 (regime_prob only)
- Dropped options_tail_risk_z (MI=0.002, noise) and options_skew_momentum_z (MI=0.017, marginal)
- Optuna objective: single-column MI (not sum of 3)
- Added input_scaling as Optuna parameter
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
from sklearn.preprocessing import StandardScaler
import optuna
import json
import os
from datetime import datetime

# === 2. Data Fetching ===
# SKEW from Yahoo Finance (^SKEW)
# GVZ from FRED (GVZCLS), fallback to Yahoo (^GVZ)
# Gold target from yfinance GC=F
# Align on common dates
# Compute changes: skew_change, gvz_change

# === 3. Feature Generation Function ===

def generate_regime_feature(skew_changes, gvz_changes, n_components, n_init,
                            train_size, input_scaling=False):
    """
    2D HMM on [SKEW changes, GVZ changes].
    Returns P(highest-trace-covariance state) for full data.

    Args:
        skew_changes: array of SKEW daily changes
        gvz_changes: array of GVZ daily changes
        n_components: number of HMM states (2 or 3)
        n_init: number of EM restarts
        train_size: index for train/val split
        input_scaling: whether to standardize inputs before HMM fit

    Returns:
        regime_prob: array of P(highest-variance state) for full dataset
    """
    X = np.column_stack([skew_changes, gvz_changes])

    if input_scaling:
        scaler = StandardScaler()
        X_train_raw = X[:train_size]
        scaler.fit(X_train_raw)
        X = scaler.transform(X)

    X_train = X[:train_size]

    model = GaussianHMM(
        n_components=n_components,
        covariance_type='full',
        n_iter=100,
        tol=1e-4,
        random_state=42
    )
    model.fit(X_train)
    probs = model.predict_proba(X)

    # Identify highest-trace (most volatile) state
    traces = [np.trace(model.covars_[i]) for i in range(n_components)]
    high_var_state = np.argmax(traces)
    return probs[:, high_var_state]

# === 4. Optuna Objective ===

def objective(trial, skew_changes, gvz_changes, target, train_size, val_mask):
    """
    Single-column MI objective: maximize MI(regime_prob, gold_return_next)
    on validation set.
    """
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    n_init = trial.suggest_categorical('hmm_n_init', [3, 5, 10])
    input_scaling = trial.suggest_categorical('input_scaling', [True, False])

    try:
        regime = generate_regime_feature(
            skew_changes, gvz_changes,
            n_components, n_init, train_size, input_scaling
        )

        # Extract validation period
        regime_val = regime[val_mask]
        target_val = target[val_mask]

        # Compute MI (single column)
        mask = ~np.isnan(regime_val) & ~np.isnan(target_val)
        if mask.sum() < 50:
            return 0.0

        def discretize(x, bins=20):
            valid = ~np.isnan(x)
            if valid.sum() < bins:
                return None
            x_c = x.copy()
            x_c[~valid] = np.nanmedian(x)
            return pd.qcut(x_c, bins, labels=False, duplicates='drop')

        feat_disc = discretize(regime_val[mask])
        tgt_disc = discretize(target_val[mask])
        if feat_disc is not None and tgt_disc is not None:
            return mutual_info_score(feat_disc, tgt_disc)

        return 0.0
    except Exception:
        return 0.0

# === 5. Main ===
# Data split: train/val/test = 70/15/15 (time-series order)
# Run Optuna with 30 trials, 300s timeout
# Generate final output with best params
# Save:
#   - submodel_output.csv (single column: options_risk_regime_prob)
#   - training_result.json (params, metrics, output_shape)
```

#### Key Implementation Notes

1. **hmmlearn installation**: Must `pip install hmmlearn` at the start. Not pre-installed on Kaggle.
2. **FRED API key**: Use `os.environ['FRED_API_KEY']` from Kaggle Secrets for GVZCLS. If FRED fails, fall back to `yf.download('^GVZ')`.
3. **Single column output**: Output CSV must have columns `[Date, options_risk_regime_prob]`. Only 1 feature column.
4. **No n_init in GaussianHMM constructor**: Pass `n_init` to control number of EM restarts. Note: hmmlearn GaussianHMM accepts `n_init` parameter.
   - **CRITICAL FIX**: Check hmmlearn version. In some versions, `n_init` is not a GaussianHMM parameter. If so, implement manual n_init by fitting the model multiple times with different random seeds and selecting the fit with highest log-likelihood.
5. **Input scaling**: When `input_scaling=True`, fit StandardScaler on training data only, then transform full data. This prevents lookahead bias.
6. **2D HMM input**: Stack [skew_changes, gvz_changes] into Nx2 array. Both should be daily differences (not levels).
7. **HMM state labeling**: After fitting, compute `np.trace(model.covars_[i])` for each state. Output P(highest-trace state). This identifies the "extreme/volatile" regime.
8. **No lookahead bias**:
   - HMM: Fit on training data only, predict_proba on full dataset
   - StandardScaler: Fit on training data only, transform full dataset
9. **NaN handling**: First row will have NaN from diff operation. Forward-fill from first valid value.
10. **Reproducibility**: Fix random_state=42 for HMM, seed=42 for Optuna.
11. **Output column name**: Must be exactly `options_risk_regime_prob` (same as attempt 1, for consistency with meta-model feature registry).
12. **training_result.json must include**:
    - `output_columns: ["options_risk_regime_prob"]` (list with single element)
    - `output_shape: [N, 1]` (reflecting single column)
    - `best_params` including `input_scaling` boolean
    - `optuna_best_value` (MI of single column, not sum)

---

## 8. Risks and Alternatives

### Risk 1: Single Column Insufficient for Gate 3 (MODERATE RISK)

- **Description**: `options_risk_regime_prob` alone may not provide enough marginal predictive power to pass Gate 3 thresholds (DA +0.5%, Sharpe +0.05, or MAE -0.01%).
- **Probability**: 35%
- **Why this risk is acceptable**: Attempt 1 showed this feature ranks #2/22 in importance (5.70%). The problem was not insufficient signal but excessive noise from companion features. With noise removed, the signal should shine through.
- **Detection**: Gate 3 evaluation
- **Mitigation for Attempt 3**: If Gate 3 fails with single column, add back `options_skew_momentum_z` (MI 0.017, the better of the two dropped features) as a second column. This follows the improvement queue's priority 2 plan.

### Risk 2: HMM State Collapse (LOW RISK)

- **Description**: HMM may collapse to effectively 1 state, producing near-constant regime probability.
- **Probability**: 10% (low because attempt 1 HMM worked well with autocorr 0.873)
- **Detection**: regime probability std < 0.05 or autocorrelation > 0.99
- **Mitigation**: n_init parameter ensures multiple EM restarts. `input_scaling` option may help separate states when raw scales differ.

### Risk 3: Input Scaling Changes Regime Interpretation

- **Description**: Standardizing inputs before HMM may change which state has highest trace, potentially worsening regime quality.
- **Probability**: 15%
- **Mitigation**: This is why `input_scaling` is an Optuna parameter, not a fixed choice. If scaling hurts MI, Optuna will select `input_scaling=False`.

### Risk 4: Optuna Objective Change Finds Worse HMM

- **Description**: Switching from sum-of-3-MI to single-column-MI changes the optimization landscape. The best HMM for regime_prob alone may differ from the best for all 3.
- **Probability**: 10%
- **Mitigation**: The search space is small (12 total combinations) and 30 trials provide exhaustive coverage. Attempt 1's best config (n_components=3) will be tried regardless.

### Alternative Design for Attempt 3 (If Attempt 2 Fails)

If this design fails at Gate 3:

1. **Add back momentum**: Output 2 columns: `options_risk_regime_prob` + `options_skew_momentum_z`. This follows the improvement queue's priority 2 plan.
2. **Conditional features**: Output regime_prob only when GVZ is above median (risk-on periods), set to 0.5 otherwise. This creates a conditional feature that activates only when options data is most informative.
3. **Abandon options_market**: If attempt 3 also fails, declare `no_further_improvement` and proceed to Phase 3 meta-model. The meta-model already has 7 completed submodels. Gate 2 confirms information exists but may not be extractable in a way that helps XGBoost generalize.

---

## 9. Expected Performance Against Gates

### Gate 1: Standalone Quality

- **Overfit ratio**: N/A (no neural network)
- **Constant output check**: Will pass -- attempt 1 showed regime_prob std is meaningful, autocorr 0.873 (well below 0.99)
- **Autocorrelation < 0.99**: Expected 0.85-0.90 (same as attempt 1: 0.873)
- **No NaN values**: Will pass after forward-filling warmup

**Expected Result**: PASS (confidence 9/10)

### Gate 2: Information Gain

- **MI increase > 5%**: Uncertain. Single column MI was 0.031 in attempt 1. Total MI increase will be lower than attempt 1's +17.12% (which was sum of 3 columns). However, the threshold is based on sum-based MI relative to baseline, and a single clean column may still clear 5%.
  - Calculation: base MI sum was 0.310. A single column adding 0.031 = 10% increase. Should pass.
  - **CAVEAT**: MI may differ in attempt 2 due to different Optuna objective. Could be higher or lower.
- **VIF < 10**: Very high probability. Attempt 1 had VIF 1.87 for this column. Fewer total columns means even lower VIF.
- **Rolling correlation std < 0.15**: High probability. Attempt 1 had stability 0.146 for this column. Borderline but passed.

**Expected Result**: LIKELY PASS (confidence 7/10)

### Gate 3: Ablation Test

- **Key question**: Can a single regime_prob column improve XGBoost generalization?
- **Evidence for**: Feature ranks #2/22 (5.70% importance), confirmed by XGBoost in attempt 1.
- **Evidence against**: Attempt 1 with 3 columns degraded DA in ALL 5 folds. Even though regime_prob was the best feature, the combined 3-column addition was net negative.
- **Hypothesis**: Removing the 2 noisy columns eliminates the source of degradation. The clean regime_prob signal can then contribute positively.
- **Direction accuracy +0.5%**: Possible but uncertain (3/10)
- **Sharpe +0.05**: Possible (4/10) -- removing noise columns may stabilize returns
- **MAE -0.01%**: Most likely path to pass (5/10) -- consistent with Technical, Cross-Asset, Yield Curve, ETF-Flow, CNY submodel patterns where MAE was the passing criterion

**Expected Result**: MODERATE (confidence 5/10)

### Overall Gate 3 Pass Probability: 5/10

Same confidence as attempt 1 but for different reasons:
- Attempt 1: Uncertainty about whether information existed (it did)
- Attempt 2: Confidence that information exists; uncertainty about whether single-column delivery is sufficient for Gate 3 threshold

---

## 10. Summary of Changes from Attempt 1

| Aspect | Attempt 1 | Attempt 2 | Rationale |
|--------|-----------|-----------|-----------|
| Output columns | 3 | 1 | Reduce noise, keep strongest signal |
| Output features | regime_prob + tail_risk_z + momentum_z | regime_prob only | tail_risk_z MI=0.002 (noise), momentum_z MI=0.017 (marginal) |
| Optuna objective | MI sum of 3 columns | MI of single column | Optimize for retained feature |
| Optuna params | 4 (n_components, n_init, zscore_window, momentum_window) | 3 (n_components, n_init, input_scaling) | Removed params for dropped features, added input scaling |
| HMM architecture | 2D [SKEW, GVZ] | 2D [SKEW, GVZ] (unchanged) | HMM itself worked well |
| Data sources | ^SKEW + GVZCLS/^GVZ | ^SKEW + GVZCLS/^GVZ (unchanged) | Data confirmed working |
| n_trials | 30 | 30 (unchanged) | Sufficient for 12 combinations |
| Kaggle GPU | false | false (unchanged) | CPU-only workload |
