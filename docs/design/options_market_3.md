# Submodel Design Document: Options Market (Attempt 3)

## 0. Fact-Check Results

| Claim / Source | Result | Detail |
|----------------|--------|--------|
| Yahoo: ^GVZ (Gold Volatility Index) | CONFIRMED | 4,058 rows, 2010-01-04 to 2026-02-20. Range [8.88, 48.98], mean 17.43. Full coverage 2010-2026. |
| Yahoo: ^SKEW (CBOE SKEW Index) | CONFIRMED | 3,996 rows, 2010-01-04 to 2026-02-20. Range [110.34, 183.12], mean 131.03. Full coverage 2010-2026. |
| FRED: VIXCLS (VIX) | CONFIRMED | Latest 2026-02-19: 20.23. Available if needed as fallback. |
| Data alignment GVZ+SKEW | CONFIRMED | Inner join yields ~2,807 aligned rows (2014-10-01 to present). |
| EMA smoothing fixes stability | CONFIRMED empirically | Raw regime_prob stability=0.1509 (FAIL). EMA(5)=0.1167, EMA(7)=0.1091, EMA(10)=0.1014 (all PASS <0.15). |
| EMA(10) autocorrelation risk | CONFIRMED | EMA(10) autocorr=0.9927, too close to 0.99 threshold. EMA(5)=0.9855 is safe. |
| GVZ vol trend raw z-score | REJECTED | Stability 0.25-0.26, far above 0.15 threshold. Cannot be used as output. |
| GVZ EMA momentum z-score | CONFIRMED viable | Stability 0.09-0.11 (PASS), autocorr 0.96-0.98 (PASS), MI 0.53-0.57 (strong), VIF with regime=1.06 (excellent). |
| VIF between outputs | CONFIRMED <10 | All tested combinations VIF < 1.10. Outputs are nearly orthogonal. |
| Method: EMA-smoothed HMM + momentum signal | VALID | Addresses both Gate 2 failures (stability and MI) with minimal risk to MAE. |

### Design Decisions from Fact-Checking

1. **Rejected GVZ vol trend z-score (raw daily change z-score)**: Stability 0.25+ makes it unusable. The evaluator's original suggestion of "rolling z-score of daily GVZ changes" would fail Gate 2 stability.

2. **Selected GVZ EMA momentum z-score instead**: This signal computes the difference between short-term and long-term EMA of GVZ level, then normalizes as a z-score. It captures the same directional trend information but with inherently smoother dynamics that pass stability (<0.12).

3. **EMA span range narrowed to 3-8**: EMA(10) causes autocorrelation 0.9927 (too close to 0.99). EMA(5) at 0.9855 is safe. Range [3, 8] ensures all candidates pass autocorrelation while providing meaningful smoothing.

4. **Alternative fallback**: If GVZ EMA momentum fails to improve MI above 5%, SKEW change z-score (stability 0.087, near-zero VIF) is the secondary candidate. It has lower MI but excellent stability and orthogonality.

---

## 1. Overview

- **Purpose**: Extract two complementary contextual features from options market data: (1) a smoothed regime state probability capturing joint SKEW/GVZ volatility regimes, and (2) a gold-volatility momentum signal capturing directional trends in GVZ.
- **Core method**: 2D Hidden Markov Model on [SKEW changes, GVZ changes] with EMA post-smoothing for regime probability, plus a deterministic GVZ EMA momentum z-score for the second column.
- **Why this approach**: Attempt 2 marginally missed Gate 2 on both MI (4.96% vs 5%) and stability (0.1555 vs 0.15). EMA smoothing empirically reduces stability from 0.151 to 0.12 (fixing stability). A second orthogonal column (VIF ~1.06) provides additional MI to push above 5%. The GVZ EMA momentum captures directional information absent from regime probability, potentially improving DA and Sharpe.
- **Key changes from Attempt 2**:
  1. EMA post-smoothing on regime probability (Optuna-tuned span 3-8)
  2. Add second output column: GVZ EMA momentum z-score (deterministic, Optuna-tuned windows)
  3. Total output: 2 columns (up from 1 in attempt 2, down from 3 in attempt 1)
- **Expected effect**: Gate 2 compliance (MI >5%, stability <0.15) while preserving the strong MAE improvement (-0.1562) and targeting positive DA/Sharpe deltas from the directional momentum signal.

### Why 2 Columns (Not 1 or 3)?

| Attempts | Columns | Gate 2 MI | Gate 3 | Lesson |
|----------|---------|-----------|--------|--------|
| Attempt 1 | 3 | 17.12% PASS | FAIL (DA -1.05% all 5 folds) | Too many columns = noise for XGBoost |
| Attempt 2 | 1 | 4.96% FAIL | PASS (MAE -0.156) | Too few columns = insufficient MI |
| Attempt 3 | 2 | Target >5% | Target: maintain MAE, improve DA/Sharpe | Sweet spot between information and noise |

The 2-column design targets the midpoint. Attempt 1 proved 3 columns is too many. Attempt 2 proved 1 column is marginally insufficient for MI. 2 carefully chosen orthogonal columns should provide enough MI while avoiding the noise penalty.

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Coverage | Status |
|------|--------|--------|-----------|----------|--------|
| SKEW Index | Yahoo Finance | ^SKEW | Daily | 2010-01-04 to present | CONFIRMED |
| GVZ (Gold Vol) | Yahoo Finance | ^GVZ | Daily | 2010-01-04 to present | CONFIRMED |
| Gold Price | Yahoo Finance | GC=F | Daily | Full history | CONFIRMED |

**No FRED dependency for primary data**: Both SKEW and GVZ are available from Yahoo Finance. This eliminates the FRED API key dependency inside the notebook.

### Preprocessing Steps

1. Fetch SKEW from Yahoo Finance (^SKEW), start=2014-10-01, end=dynamic (today)
2. Fetch GVZ from Yahoo Finance (^GVZ), same date range
3. Inner join SKEW and GVZ on date index to get aligned daily data
4. Forward-fill gaps up to 3 days
5. Compute daily changes: `skew_change = SKEW_t - SKEW_{t-1}`, `gvz_change = GVZ_t - GVZ_{t-1}`
6. Fetch Gold price (GC=F) for target variable
7. Compute next-day gold return: `gold_return_next = gold_return.shift(-1)`
8. Drop rows with NaN from diff operations

### Expected Sample Count

- ~2,800 rows from 2014-10-01 to present (aligned SKEW+GVZ)
- After preprocessing and alignment with base_features: ~2,500 rows
- Sufficient for HMM fitting (train ~1,750 rows) and Optuna optimization

### Data Quality Expectations

| Metric | SKEW (^SKEW) | GVZ (^GVZ) |
|--------|-------------|------------|
| Level range | [110, 183] | [8.9, 49.0] |
| Level mean | ~131 | ~17.4 |
| Change std | ~5-8 | ~1.5-3 |
| Change autocorr(1) | ~0.15 | ~0.15 |
| Coverage | ~250 rows/year | ~252 rows/year |

---

## 3. Model Architecture

This design has two components: an HMM-based regime detector (with EMA smoothing) and a deterministic momentum signal. No neural network. No PyTorch required.

### Component 1: EMA-Smoothed 2D HMM Regime Probability

Same HMM as attempt 2, with EMA post-smoothing applied to the raw regime probability output.

```
Input: [SKEW daily changes, GVZ daily changes] [T x 2]
       |
   (Optional) StandardScaler on inputs (fit on train only)
       |
   GaussianHMM.fit(train_data) -> learn 2-3 state model
       |
   GaussianHMM.predict_proba(full_data) -> [T x n_states]
       |
   Compute trace of covars for each state
   Select P(highest-trace state) -> [T x 1]
       |
   EMA smoothing (span = Optuna param, range 3-8)
       |
Output: options_regime_smooth (single column, range ~0-1, smoothed)
```

**Why EMA smoothing**: Raw HMM `predict_proba` produces sharp transitions between states (e.g., 0.01 -> 0.95 in one day). These transitions cause:
- High rolling correlation standard deviation (0.1555, above 0.15 threshold)
- Frequent position switches in the meta-model (hurting Sharpe via transaction costs)

EMA dampens these transitions. Empirically confirmed:
- EMA(5): stability 0.1167 (from 0.1509), autocorr 0.9855
- EMA(7): stability 0.1091, autocorr 0.9895
- No risk of constant output: the underlying regime signal has std=0.31

### Component 2: GVZ EMA Momentum Z-Score

A deterministic signal capturing the directional trend in gold-specific implied volatility.

```
Input: GVZ level series
       |
   Short EMA(span_short) = fast-moving average
   Long EMA(span_long) = slow-moving average
       |
   momentum = EMA_short - EMA_long
       |
   Normalize: z = (momentum - rolling_mean(norm_window)) / rolling_std(norm_window)
       |
   Clip to [-3, 3]
       |
Output: options_gvz_momentum_z (single column, range [-3, 3])
```

**Why this signal (not raw GVZ change z-score)**:
- Raw GVZ change z-score has stability 0.25+ (FAILS Gate 2)
- GVZ EMA momentum has stability 0.09-0.11 (PASSES Gate 2)
- EMA momentum captures the same directional trend but with inherently smoother dynamics
- MI is comparable or better (0.53-0.57 vs 0.47-0.50 for raw z-score)

**Rationale for this signal type**:
- Regime probability captures "what state are we in" (level/magnitude)
- GVZ momentum captures "which direction is volatility trending" (direction/rate of change)
- These are complementary information axes (confirmed by near-zero correlation, VIF ~1.06)
- Directional information is what was missing from attempt 2 (DA was negative)

### Combined Output

| Column | Range | Description | Expected Autocorr | Expected Stability |
|--------|-------|-------------|--------------------|--------------------|
| `options_regime_smooth` | [0, 1] | EMA-smoothed P(high-variance regime) from 2D HMM | 0.985-0.990 | 0.10-0.12 |
| `options_gvz_momentum_z` | [-3, 3] | GVZ EMA momentum z-score (trend direction) | 0.95-0.98 | 0.09-0.11 |

**Total: 2 columns**.

### Orthogonality Analysis (Empirically Verified)

| Pair | Measured Correlation | VIF | Assessment |
|------|---------------------|-----|------------|
| options_regime_smooth vs options_gvz_momentum_z | 0.21-0.24 | 1.05-1.06 | Excellent orthogonality |
| Either output vs existing VIX submodel | est. 0.1-0.3 | <1.10 | Acceptable (different inputs) |
| Either output vs any base feature | est. <0.3 | <1.10 | Acceptable (nonlinear transforms) |

The two outputs are nearly orthogonal because:
- Regime probability is a nonlinear state classification (binary-like: "are we in a volatile regime?")
- Momentum z-score is a directional rate-of-change signal (continuous: "is volatility rising or falling?")
- These capture fundamentally different aspects of the same underlying data

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM covariance_type | "full" | Required for 2D input to capture cross-correlation |
| HMM n_iter | 100 | Standard EM convergence limit |
| HMM tol | 1e-4 | Standard convergence tolerance |
| HMM random_state | 42 | Reproducibility |
| Z-score clip range | [-3, 3] | Standard outlier control |
| Output columns | 2 | Design constraint |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3} | categorical | 2=normal/extreme, 3=calm/moderate/extreme |
| input_scaling | {True, False} | categorical | Whether to standardize HMM inputs |
| ema_span | [3, 8] | int | Smoothing span for regime prob. Lower bound 3 = minimal smoothing. Upper bound 8 = avoids autocorr >0.99. |
| gvz_ema_short | [3, 10] | int | Short EMA window for GVZ momentum |
| gvz_ema_long | [15, 50] | int | Long EMA window for GVZ momentum. Must be > gvz_ema_short. |
| gvz_norm_window | [30, 90] | int | Rolling window for z-score normalization of momentum |

**Constraint**: `gvz_ema_long > gvz_ema_short + 5` (enforce sufficient separation between short and long EMA).

### Exploration Settings

- **n_trials**: 50
  - Rationale: 6 parameters with mixed types. 50 trials with TPE provides sufficient exploration. Each trial runs in <5 seconds (HMM fit + deterministic computation).
- **timeout**: 600 seconds (10 minutes)
- **objective**: Maximize sum of MI for both output columns on validation set: `MI(regime_smooth, target) + MI(gvz_momentum_z, target)`
  - This is the same sum-based objective as attempt 1, but with only 2 columns (not 3). Sum-based MI matches the Gate 2 evaluation metric.
  - **Penalty**: If either column has autocorrelation > 0.98, return 0.0 (discard trial). This prevents Optuna from selecting configurations that would fail Gate 1.
- **direction**: maximize
- **sampler**: TPESampler(seed=42)
- **pruner**: None (trials are fast)

---

## 5. Training Settings

### Fitting Procedure

1. **HMM**: `GaussianHMM.fit(hmm_input_train)` -- EM algorithm on 2D [SKEW changes, GVZ changes]. Fit on training data only.
2. **HMM predict**: `GaussianHMM.predict_proba(full_data)` -- generate state probabilities for all data.
3. **EMA smoothing**: `pd.Series.ewm(span=ema_span).mean()` applied to raw regime probability. No lookahead (EMA is causal by construction).
4. **GVZ momentum**: Compute deterministically from GVZ level series. EMA and rolling statistics use the full data but are causal (each value depends only on current and past observations).

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- HMM fits on training set data only
- StandardScaler (if input_scaling=True) fits on training data only, transforms full dataset
- Optuna optimizes MI on validation set
- Test set reserved for evaluator Gate 3

### Loss Function

N/A -- no gradient-based training. HMM uses EM (maximum likelihood). GVZ momentum is deterministic.

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- HMM EM converges via `n_iter` and `tol`.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. HMM + statistics are CPU-only. |
| Estimated execution time | 5-8 minutes | Data download (~30s) + 50 Optuna trials x ~5s each (~4min) + final output (~30s). |
| Estimated memory usage | <1 GB | ~2,800 rows x 6-8 columns. Tiny dataset. |
| Required pip packages | hmmlearn | Must `pip install hmmlearn` at start. sklearn, pandas, numpy, yfinance pre-installed on Kaggle. |

---

## 7. Implementation Instructions

### builder_data Instructions

**Data is the same as attempts 1-2.** The data fetching code in the notebook is self-contained (yfinance calls for ^SKEW, ^GVZ, GC=F). No separate builder_data step is needed since the notebook fetches data directly.

If builder_data does create a separate file:

1. Fetch ^SKEW from Yahoo Finance, start=2014-10-01, end=dynamic
2. Fetch ^GVZ from Yahoo Finance, same date range
3. Inner join on common dates, forward-fill up to 3 days
4. Compute: `skew_change`, `gvz_change`
5. Save: `data/processed/options_market_features_input.csv`
   - Columns: Date (index), skew_close, gvz_close, skew_change, gvz_change
6. Quality checks:
   - No gaps > 3 consecutive trading days
   - Missing data < 2%
   - SKEW in [100, 200], GVZ in [5, 80]
   - |SKEW change| < 30, |GVZ change| < 20

### builder_model Instructions

#### Notebook Structure

```python
"""
Gold Prediction SubModel Training - Options Market Attempt 3
Self-contained: Data fetch -> Preprocessing -> HMM + Momentum -> Optuna HPO -> Save results

KEY CHANGES FROM ATTEMPT 2:
1. EMA post-smoothing on regime probability (Optuna-tuned span 3-8)
2. Added second column: GVZ EMA momentum z-score (directional trend signal)
3. Total output: 2 columns (options_regime_smooth + options_gvz_momentum_z)
4. Optuna objective: sum of MI for both columns
5. Autocorrelation guard: reject trials with autocorr > 0.98
"""

# === 1. Libraries ===
import subprocess
subprocess.check_call(['pip', 'install', 'hmmlearn', '--quiet'])

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
import optuna
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# === 2. Data Fetching ===
# Fetch ^SKEW, ^GVZ, GC=F from Yahoo Finance
# Align on common dates
# Compute: skew_change, gvz_change, gold_return, gold_return_next

# === 3. Feature Generation Functions ===

def generate_regime_feature(data, n_components, train_size, input_scaling, ema_span):
    """
    2D HMM on [SKEW changes, GVZ changes], with EMA post-smoothing.
    Returns EMA-smoothed P(highest-trace-covariance state).
    """
    X = data[['skew_change', 'gvz_change']].values

    if input_scaling:
        scaler = StandardScaler()
        scaler.fit(X[:train_size])
        X = scaler.transform(X)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type='full',
        n_iter=100,
        tol=1e-4,
        random_state=42
    )
    model.fit(X[:train_size])
    probs = model.predict_proba(X)

    traces = [np.trace(model.covars_[i]) for i in range(n_components)]
    high_var_state = np.argmax(traces)
    raw_regime = probs[:, high_var_state]

    # EMA post-smoothing (causal, no lookahead)
    smoothed = pd.Series(raw_regime).ewm(span=ema_span).mean().values
    return smoothed


def generate_gvz_momentum_z(gvz_series, ema_short, ema_long, norm_window):
    """
    GVZ EMA momentum z-score.
    Captures directional trend in gold-specific implied volatility.
    """
    ema_s = gvz_series.ewm(span=ema_short).mean()
    ema_l = gvz_series.ewm(span=ema_long).mean()
    momentum = ema_s - ema_l

    # Normalize as z-score
    mom_mean = momentum.rolling(norm_window).mean()
    mom_std = momentum.rolling(norm_window).std()
    z = ((momentum - mom_mean) / mom_std).clip(-3, 3)
    return z.values

# === 4. Optuna Objective ===

def objective(trial, data, train_size, val_mask, target_val):
    """
    Maximize sum of MI for both output columns on validation set.
    Reject trials with autocorrelation > 0.98 for either column.
    """
    # HMM parameters
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    input_scaling = trial.suggest_categorical('input_scaling', [True, False])
    ema_span = trial.suggest_int('ema_span', 3, 8)

    # GVZ momentum parameters
    gvz_ema_short = trial.suggest_int('gvz_ema_short', 3, 10)
    gvz_ema_long = trial.suggest_int('gvz_ema_long', 15, 50)
    gvz_norm_window = trial.suggest_int('gvz_norm_window', 30, 90)

    # Constraint: long must be sufficiently larger than short
    if gvz_ema_long <= gvz_ema_short + 5:
        return 0.0

    try:
        # Generate both features
        regime = generate_regime_feature(
            data, n_components, train_size, input_scaling, ema_span
        )
        momentum_z = generate_gvz_momentum_z(
            data['gvz_close'], gvz_ema_short, gvz_ema_long, gvz_norm_window
        )

        # Autocorrelation guard
        regime_series = pd.Series(regime)
        mom_series = pd.Series(momentum_z)
        if regime_series.dropna().autocorr(lag=1) > 0.98:
            return 0.0
        if mom_series.dropna().autocorr(lag=1) > 0.98:
            return 0.0

        # Extract validation portion
        regime_val = regime[val_mask]
        mom_val = momentum_z[val_mask]

        # Compute MI for each column
        mi_regime = calc_mi(regime_val, target_val)
        mi_mom = calc_mi(mom_val, target_val)

        return mi_regime + mi_mom

    except Exception:
        return 0.0

# === 5. Main Execution ===
# Data split: train/val/test = 70/15/15
# Run Optuna: 50 trials, 600s timeout
# Generate final output with best params
# Save:
#   - submodel_output.csv (2 columns: options_regime_smooth, options_gvz_momentum_z)
#   - training_result.json
```

#### Key Implementation Notes

1. **hmmlearn installation**: Must `pip install hmmlearn --quiet` at start. Not pre-installed on Kaggle.
2. **No FRED dependency**: All data from Yahoo Finance (^SKEW, ^GVZ, GC=F). Eliminates need for FRED API key.
3. **Two column output**: Output CSV must have columns `[Date, options_regime_smooth, options_gvz_momentum_z]`.
4. **Column naming**: `options_regime_smooth` (not `options_risk_regime_prob` from attempt 2). This distinguishes the smoothed version from the raw version.
5. **EMA smoothing**: Use `pd.Series.ewm(span=ema_span).mean()`. This is causal (no lookahead). Apply AFTER HMM predict_proba.
6. **GVZ momentum**: Use `gvz.ewm(span=short).mean() - gvz.ewm(span=long).mean()`. Then normalize with rolling z-score (rolling mean/std over `norm_window`). Clip to [-3, 3].
7. **Autocorrelation guard**: In the Optuna objective, check that both outputs have autocorr(1) < 0.98. Return 0.0 for trials that violate this. This prevents selecting configurations that would fail Gate 1.
8. **Input scaling**: When `input_scaling=True`, fit StandardScaler on train portion only.
9. **NaN handling**: Forward-fill warmup NaN from diff/rolling operations. Do NOT forward-fill final rows. Output should have NaN only in the initial warmup period.
10. **Dataset reference**: kernel-metadata.json must include `"dataset_sources": ["bigbigzabuton/gold-prediction-submodels"]`.
11. **Standard path resolution**: Use the standard dataset path resolution block from MEMORY.md.
12. **XGBoost compatibility**: If XGBoost is used in evaluation, `early_stopping_rounds` must be in the constructor (v3.x).
13. **training_result.json must include**:
    - `output_columns: ["options_regime_smooth", "options_gvz_momentum_z"]`
    - `output_shape: [N, 2]`
    - `best_params` including all 6 Optuna parameters
    - Per-column statistics: mean, std, min, max, autocorr, NaN ratio

---

## 8. Risks and Alternatives

### Risk 1: EMA Smoothing Degrades MAE (LOW)

- **Description**: Smoothing may delay regime transitions, losing some of the timing precision that drives MAE improvement.
- **Probability**: 15%
- **Evidence against**: MAE improvement in attempt 2 came from the level of regime probability (high-risk periods have higher abs returns), not from transition timing. Smoothing preserves level information.
- **Mitigation**: EMA span is Optuna-tuned (3-8). Span=3 provides minimal smoothing if more responsive output is better.

### Risk 2: Second Column Adds Noise (MODERATE)

- **Description**: Adding options_gvz_momentum_z could add noise similar to attempt 1's 3-column problem.
- **Probability**: 25%
- **Evidence against**:
  - Attempt 1 added 2 weak columns (MI=0.002 and MI=0.017). This column has MI 0.53-0.57 (strong).
  - VIF is 1.06 (nearly orthogonal), so it provides genuinely new information.
  - Momentum captures directional information absent from regime probability.
- **Mitigation**: If Gate 3 fails, attempt 4 can revert to 1 column (EMA-smoothed regime only).

### Risk 3: Gate 2 MI Still Below 5% (LOW)

- **Description**: Even with 2 columns, total MI increase may not reach 5%.
- **Probability**: 10%
- **Evidence against**: Attempt 2 achieved 4.96% MI with 1 column. Adding a second column with MI ~0.53 should push total above 5%.
- **Mitigation**: SKEW change z-score (stability 0.087, MI 0.47-0.52) is a ready fallback if GVZ momentum underperforms.

### Risk 4: Autocorrelation Exceeds 0.99 (LOW)

- **Description**: EMA smoothing increases autocorrelation. If ema_span is too large, output could exceed 0.99.
- **Probability**: 5%
- **Evidence against**: EMA span is capped at 8 (autocorr ~0.99 at span=10). The Optuna objective includes an autocorrelation guard (reject trials with autocorr > 0.98).
- **Mitigation**: Hard cap at ema_span=8 in search space. Autocorrelation guard in objective.

### Alternative Design for Attempt 4 (If Attempt 3 Fails)

If Gate 2 fails:
- Try 3-state HMM with SKEW as additional input, outputting 2 of 3 state probabilities

If Gate 3 fails:
- Revert to single EMA-smoothed regime column (fixes stability) and accept Gate 2 MI miss
- Since attempt 2 already passed Gate 3 via MAE, the submodel is already completed

---

## 9. Expected Performance Against Gates

### Gate 1: Standalone Quality

| Check | Expected Value | Threshold | Confidence |
|-------|---------------|-----------|------------|
| Overfit ratio | N/A (unsupervised HMM + deterministic) | <1.5 | 10/10 |
| All-NaN columns | 0 | 0 | 10/10 |
| Constant columns | 0 | 0 | 10/10 |
| options_regime_smooth autocorr | 0.985-0.990 | <0.99 | 8/10 |
| options_gvz_momentum_z autocorr | 0.95-0.98 | <0.99 | 9/10 |
| NaN ratio | <5% (warmup only) | <10% | 10/10 |

**Expected Result**: PASS (confidence 8/10)

### Gate 2: Information Gain

| Check | Expected Value | Threshold | Confidence |
|-------|---------------|-----------|------------|
| MI increase (sum-based) | 6-10% | >5% | 8/10 |
| VIF (between outputs) | ~1.06 | <10 | 10/10 |
| VIF (with existing features) | <3.0 | <10 | 9/10 |
| Stability (regime_smooth) | 0.10-0.12 | <0.15 | 9/10 |
| Stability (gvz_momentum_z) | 0.09-0.11 | <0.15 | 9/10 |

**Expected Result**: PASS (confidence 8/10 -- primary improvement over attempt 2)

### Gate 3: Ablation Test

| Metric | Attempt 2 Delta | Expected Delta | Threshold | Confidence |
|--------|----------------|----------------|-----------|------------|
| Direction Accuracy | -0.24% | -0.1% to +0.5% | >+0.5% | 3/10 |
| Sharpe | -0.141 | -0.05 to +0.1 | >+0.05 | 4/10 |
| MAE | -0.1562 | -0.10 to -0.16 | <-0.01 | 8/10 |

**Expected Result**: PASS via MAE (confidence 7/10). Positive DA/Sharpe delta is the stretch goal.

---

## 10. Summary of Changes from Attempt 2

| Aspect | Attempt 2 | Attempt 3 | Rationale |
|--------|-----------|-----------|-----------|
| Output columns | 1 | 2 | Fix MI (4.96% -> >5%) |
| Column 1 | options_risk_regime_prob (raw) | options_regime_smooth (EMA) | Fix stability (0.1555 -> <0.12) |
| Column 2 | N/A | options_gvz_momentum_z | Complementary directional signal |
| EMA smoothing | None | span 3-8 (Optuna) | Dampen regime transitions |
| HMM architecture | 2D [SKEW, GVZ] | 2D [SKEW, GVZ] (unchanged) | Proven to work |
| Optuna params | 3 (n_components, n_init, scaling) | 6 (+ema_span, gvz_ema_short, gvz_ema_long, gvz_norm_window; -n_init) | More parameters for 2 outputs |
| Optuna objective | Single-column MI | Sum of 2-column MI | Match Gate 2 evaluation |
| Optuna trials | 30 | 50 | Larger search space |
| Autocorr guard | None | Reject if >0.98 | Prevent Gate 1 failure |
| FRED dependency | Optional (GVZ fallback) | None (all Yahoo Finance) | Simplify notebook |
| GPU | false | false | CPU-only workload |
