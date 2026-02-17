# Submodel Design Document: CNY Demand Proxy (Attempt 2)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| Yahoo Finance CNY=X | CONFIRMED | 2861 rows (2015-01-01 to 2025-12-30), no NaN in Close. Values 6.1-7.3 range consistent with USD/CNY onshore rate. |
| Yahoo Finance CNH=X (offshore) | DOES NOT EXIST | Ticker CNH=X returns empty DataFrame ("possibly delisted"). Also tested USDCNH=X, CNHUSD=X -- all unavailable. 6C=F exists but is Canadian dollar futures, not CNH. |
| Yahoo Finance GC=F | CONFIRMED | 2764 rows (2015-01-02 to 2025-12-30), gold futures daily close. |
| corr(CNY_return, DXY_return) = 0.028 | CONFIRMED | Near-zero. CNY features are orthogonal to DXY. |
| CNH=X reserved for attempt 2 | INVALIDATED | CNH data is not available on Yahoo Finance. Idea A (CNY-CNH spread signal) is impossible. Must use CNY-only approach. |
| cny_momentum_z was rank 4 (5.57%) in attempt 1 | CONFIRMED | Per evaluation summary. The useful signal was in momentum, not regime or volatility. |
| Attempt 1 DA degraded -2.06% | CONFIRMED | Worst of any passing submodel. 3 columns gave XGBoost dimensions to overfit. |
| Meta-model attempt 7 excluded cny_demand | CONFIRMED | 24 features from 8 other submodels, zero cny_demand features. |
| ie_gold_sensitivity_z pattern (rolling corr z-score) | CONFIRMED viable | Measured: rolling 10d corr(CNY_ret, gold_ret) z-scored against 120d baseline achieves MI=0.0252 (highest MI of all candidates tested). |
| Composite mom_z + gsens_z improves correlation | CONFIRMED | Composite (0.6*mom + 0.4*(-gsens)) achieves corr=+0.031 vs mom_z alone at +0.035. Not clearly better; composite may average out signal. |

### Critical Findings

**1. CNH=X is unavailable on Yahoo Finance.** The research report claimed it was available for attempt 2. This eliminates Idea A (CNY-CNH spread signal) entirely. All offshore yuan tickers (CNH=X, USDCNH=X, CNHUSD=X) return no data.

**2. Momentum z-score (5d, 40d baseline) has the strongest correlation.** corr(mom_z_5_40, gold_ret_next) = +0.0346, the highest of all single candidates. This matches attempt 1 where cny_momentum_z was rank 4 at 5.57% importance.

**3. Gold-sensitivity z-score has highest MI but lower correlation.** gsens_z_10_120 achieves MI=0.0252 with gold_ret_next (highest among all candidates) but only corr=-0.015. The negative correlation means the z-score should be negated for a positive signal.

**4. Single output is critical.** Attempt 1's 3 columns degraded DA by -2.06%. Options_market succeeded by reducing 3->1 column. The fix must be single-output.

**5. Both candidates are highly orthogonal to existing features.** Maximum absolute correlation with any of the 20 existing meta-model submodel features: mom_z at |0.12| (with etf_pv_divergence), gsens_z at |0.08| (with tech_trend_regime_prob). VIF risk is negligible.

### Design Decision

**Selected: Idea D (optimized momentum z-score, single output)** with Optuna-tuned windows.

Rationale for rejecting alternatives:
- **Idea A (CNH spread)**: Impossible -- CNH=X unavailable on Yahoo Finance.
- **Idea B (gold-sensitivity z-score)**: Lower correlation than momentum (+0.015 vs +0.035). While MI is higher (0.0252 vs 0.0160), the weak linear signal may not translate to Gate 3 DA improvement.
- **Idea C (composite)**: Combining mom_z and gsens_z does not reliably improve over mom_z alone (composite corr +0.031 vs mom_z +0.035). Adding complexity without clear benefit risks overfitting the blend weights.
- **Idea D (momentum-only)**: Selected. Simplest approach, highest single-feature correlation, proven signal (rank 4 in attempt 1). Window optimization via Optuna can extract maximum signal from the strongest component.

The key insight: attempt 1's problem was not that the CNY signal was weak -- cny_momentum_z was genuinely informative (rank 4). The problem was that 2 noisy companions (regime_prob at rank 13, vol_regime_z at rank 18) diluted the signal and gave XGBoost dimensions to overfit on. The fix is surgical: keep only the signal that works, optimize its parameters, and output a single column.

---

## 1. Overview

- **Purpose**: Extract the single most informative dimension of CNY/USD dynamics -- momentum persistence -- as a z-scored signal. This captures the direction and intensity of CNY depreciation/appreciation trends relative to recent history, serving as a proxy for shifting Chinese gold demand conditions.
- **Core method**: Deterministic rolling z-score of N-day cumulative CNY return against a rolling baseline. No ML model. Pure pandas computation.
- **Why single output**: Attempt 1 showed that 3 outputs degraded DA by -2.06% while only cny_momentum_z (rank 4) contributed meaningfully. Reducing to 1 output follows the options_market success pattern (3->1 column, Gate 3 PASS via MAE -0.1562).
- **Why deterministic**: HMM regime_prob was near-zero 99% of the time (mean=0.009) and ranked 13th. Vol_regime_z ranked 18th (near-noise). Eliminating both removes the probabilistic component entirely, yielding a simpler, more stable signal.
- **Expected effect**: By removing 2 noise dimensions and optimizing the window parameters of the single useful signal, we expect DA improvement (reversing the -2.06% degradation) while maintaining or improving MAE contribution.

### Key Advantage

Fully deterministic. No hmmlearn dependency. No random seeds. No convergence issues. Output is reproducible given the same data and parameters. Fastest possible execution.

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Expected Rows |
|------|--------|--------|-----------|---------------|
| CNY/USD onshore rate | Yahoo Finance | CNY=X | Daily | ~2,861 (2015-01-01 to latest) |

### Data NOT Used (Attempt 2 Updates)

| Data | Source | Reason for Exclusion |
|------|--------|---------------------|
| CNH=X (offshore yuan) | Yahoo Finance | Does not exist on Yahoo Finance. All alternative tickers (USDCNH=X, CNHUSD=X) also unavailable. |
| GC=F (gold futures) | Yahoo Finance | Not needed. Attempt 1 used it for MI-based Optuna objective. Attempt 2 uses a deterministic approach with no MI optimization needed during HPO. Gold returns are used only for final MI evaluation metric. |
| FRED data | FRED API | Not needed. CNY=X is the sole data source. |

### Preprocessing Steps

1. Fetch CNY=X from Yahoo Finance, start='2014-06-01' (buffer for warmup before 2015-01-30)
2. Extract Close column, compute `cny_return = close.pct_change()`
3. Forward-fill gaps up to 3 trading days
4. Compute momentum and z-score (see Section 3)
5. Trim output to base_features date range: 2015-01-30 to latest available
6. Save single-column output aligned to base_features dates

### Expected Sample Count

- Raw CNY=X: ~2,861 rows
- After warmup (max momentum_window + baseline_window): ~2,600-2,700 rows
- After alignment to base_features: ~2,500+ rows

---

## 3. Model Architecture (Deterministic)

### Single Feature: cny_demand_momentum_z

This is a pure deterministic computation with no ML model. No PyTorch class needed.

#### Computation

```
Input: CNY=X Close prices [T x 1]
       |
   cny_return = close.pct_change()            # daily log-like return
       |
   momentum = cny_return.rolling(W_mom).sum()  # cumulative N-day return
       |
   rolling_mean = momentum.rolling(W_base).mean()
   rolling_std  = momentum.rolling(W_base).std()
       |
   z = (momentum - rolling_mean) / rolling_std  # z-score
       |
   z = z.clip(-4, 4)                            # winsorize extremes
       |
Output: cny_demand_momentum_z [T x 1]
```

#### Formula

```
cny_demand_momentum_z(t) = clip(
    [sum(cny_ret(t-W_mom+1:t)) - mean(sum(cny_ret) over past W_base days)]
    / std(sum(cny_ret) over past W_base days),
    -4, 4
)
```

Where:
- `W_mom` = momentum window (Optuna: 3-10 days)
- `W_base` = baseline window (Optuna: 30-120 days)
- `cny_ret(t) = (CNY_close(t) - CNY_close(t-1)) / CNY_close(t-1)`

#### Interpretation

- **Positive values**: CNY is depreciating faster than recent norm. Capital outflow pressure. Historically associated with Chinese investors increasing gold holdings as a hedge against currency weakness.
- **Negative values**: CNY is appreciating faster than recent norm. Capital stability. Reduced urgency for gold hedging.
- **Near zero**: CNY momentum is typical for recent conditions.

#### Properties (from fact-check)

- Correlation with gold_ret_next: +0.035 (best configuration)
- Standard deviation: ~1.09 (well-behaved z-score)
- Autocorrelation (lag-1): ~0.74 (well below 0.99 threshold)
- VIF against existing 20 meta-model features: all |r| < 0.13

### Output Specification

| Column | Range | Description |
|--------|-------|-------------|
| `cny_demand_momentum_z` | [-4, +4] | CNY momentum z-score: directional persistence relative to rolling baseline |

Single column. No regime probability. No volatility z-score.

### Orthogonality Analysis (Measured Against 20 Meta-Model Features)

| Existing Feature | Correlation with cny_momentum_z | Assessment |
|------------------|--------------------------------|------------|
| etf_pv_divergence | -0.125 | Low |
| vix_mean_reversion_z | +0.086 | Low |
| vix_persistence | -0.075 | Low |
| ie_gold_sensitivity_z | +0.070 | Low |
| tech_mean_reversion_z | +0.062 | Low |
| tech_trend_regime_prob | +0.054 | Low |
| All others | |r| < 0.05 | Negligible |

Maximum absolute correlation: 0.125 (with etf_pv_divergence). This is well below any VIF concern threshold. The CNY momentum signal is essentially orthogonal to all 20 existing features in the meta-model.

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Z-score clipping | [-4, 4] | Prevent extreme outliers. Standard across all submodels. |
| Data start | 2014-06-01 | Warmup buffer before 2015-01-30 output start. Supports up to 120-day baseline window. |
| Forward-fill max gap | 3 days | Handle weekend/holiday gaps without inventing data |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| momentum_window | int, [3, 10] | linear | 3d=responsive to short moves, 10d=captures 2-week trends. Attempt 1 fixed at 5d. Expanding search to find optimal. |
| baseline_window | int, [30, 120], step=10 | linear | 30d=reactive baseline, 120d=structural baseline. Attempt 1 explored {60, 120} only. Finer grid here. |

### Exploration Settings

- **n_trials**: 30
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of:
  1. Absolute Pearson correlation between cny_demand_momentum_z and gold_return_next on validation set
  2. Mutual information (discretized, 20 bins) between cny_demand_momentum_z and gold_return_next on validation set
  - Combined: `objective = abs_corr + MI * 10` (MI is typically ~0.01-0.02, scale to be comparable with corr ~0.02-0.04)
- **direction**: maximize
- **sampler**: TPESampler(seed=42)

### Rationale for Objective Function

Attempt 1 used MI-sum across 3 features. This optimized for aggregate information but allowed 2 weak features to contribute noise. Attempt 2 uses a single feature, so the objective directly maximizes the predictive relevance of that one feature. Adding Pearson correlation alongside MI ensures both linear and nonlinear predictive power are optimized.

---

## 5. Training Settings

### Fitting Procedure

No fitting required. This is a purely deterministic computation:

1. Fetch CNY=X close prices
2. Compute daily returns
3. Compute rolling momentum (sum of N-day returns)
4. Compute z-score against rolling baseline
5. Clip to [-4, 4]
6. Output

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- Optuna evaluates objective on validation set only
- Final retrain: use best parameters, generate output for full dataset
- Test set is reserved for evaluator Gate 3

### Evaluation Metric for Optuna

For each trial:
1. Compute cny_demand_momentum_z with trial's momentum_window and baseline_window
2. On validation set, compute:
   - `abs_corr = abs(pearsonr(feature, gold_return_next))`
   - `MI = mutual_info_score(discretize(feature, 20 bins), discretize(gold_return_next, 20 bins))`
3. Maximize: `abs_corr + MI * 10`

### Loss Function

N/A -- no gradient-based training. Purely deterministic computation.

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- deterministic. No convergence concept.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network, no ML model at all. Pure pandas rolling computations. |
| Estimated execution time | 2-3 minutes | Data download (~30s) + 30 Optuna trials x ~1s each (~30s) + final output generation (~10s) + saving (~5s). Much faster than attempt 1 (no HMM fitting). |
| Estimated memory usage | < 0.5 GB | Single ticker, ~2,861 rows. Minimal memory. |
| Required pip packages | None | Uses only pandas, numpy, scipy, sklearn, optuna, yfinance -- all pre-installed on Kaggle. No hmmlearn needed (unlike attempt 1). |

---

## 7. Implementation Instructions

### builder_data Instructions

**Minimal data fetching required.** The notebook is self-contained and fetches its own data.

However, for datachecker validation:

1. **Fetch CNY=X from Yahoo Finance**: `yf.download('CNY=X', start='2014-06-01')` -- Close column
2. **Compute**: `cny_return = cny_close.pct_change()`
3. **Save**: `data/processed/cny_demand_features_input.csv` with columns: Date, cny_close, cny_return
4. **Quality checks**:
   - CNY=X Close values in range [5.5, 8.0]
   - No gaps > 5 consecutive trading days
   - Missing data < 2%
   - cny_return has no extreme outliers (|value| < 0.05)
5. **Note**: GC=F is NOT needed in the preprocessed data file. The notebook fetches it directly for Optuna evaluation.

### builder_model Instructions

#### train.ipynb Structure

```python
"""
Gold Prediction SubModel Training - CNY Demand Proxy Attempt 2
Self-contained: Data fetch -> Deterministic momentum z-score -> Optuna window HPO -> Save results
Key change from Attempt 1: Single output (momentum z-score only), no HMM, no hmmlearn dependency
"""

# === 1. Libraries ===
# NO pip install needed (unlike attempt 1 which required hmmlearn)
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
import optuna
import json
import os
from datetime import datetime

# === 2. Data Fetching ===
# yfinance: CNY=X (start=2014-06-01 for warmup buffer)
# yfinance: GC=F (for Optuna validation metric only)
# Compute: cny_return = cny_close.pct_change()
#          gold_return = gc_close.pct_change()
#          gold_return_next = gold_return.shift(-1)

# === 3. Feature Generation ===

def generate_momentum_z(cny_return, momentum_window, baseline_window):
    """
    Compute z-scored momentum: rolling sum of N-day returns,
    z-scored against baseline_window-day rolling statistics.

    Args:
        cny_return: pd.Series of daily CNY returns
        momentum_window: int, days for cumulative return (3-10)
        baseline_window: int, days for z-score baseline (30-120)

    Returns:
        pd.Series: z-scored momentum, clipped to [-4, 4]
    """
    momentum = cny_return.rolling(momentum_window).sum()
    rolling_mean = momentum.rolling(baseline_window).mean()
    rolling_std = momentum.rolling(baseline_window).std()
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    z = (momentum - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z

# === 4. Optuna Objective ===

def objective(trial, cny_return, gold_return_next, val_start, val_end):
    """
    Maximize abs(Pearson correlation) + 10 * MI on validation set.

    Single feature, single objective. No HMM, no multi-feature MI sum.
    """
    momentum_window = trial.suggest_int('momentum_window', 3, 10)
    baseline_window = trial.suggest_int('baseline_window', 30, 120, step=10)

    # Generate feature
    z = generate_momentum_z(cny_return, momentum_window, baseline_window)

    # Extract validation period
    z_val = z.iloc[val_start:val_end]
    target_val = gold_return_next.iloc[val_start:val_end]

    # Drop NaN
    valid = z_val.notna() & target_val.notna()
    if valid.sum() < 50:
        return 0.0

    z_clean = z_val[valid].values
    target_clean = target_val[valid].values

    # Pearson correlation
    corr, _ = pearsonr(z_clean, target_clean)
    abs_corr = abs(corr)

    # Mutual information (discretized)
    def discretize(x, bins=20):
        return pd.qcut(x, bins, labels=False, duplicates='drop')

    try:
        z_disc = discretize(z_clean)
        t_disc = discretize(target_clean)
        mi = mutual_info_score(z_disc, t_disc)
    except Exception:
        mi = 0.0

    return abs_corr + mi * 10

# === 5. Main Execution ===
# 1. Fetch data
# 2. Align CNY=X and GC=F dates
# 3. Data split: train/val/test = 70/15/15 (time-series order)
# 4. Run Optuna: 30 trials, 300s timeout
# 5. Generate final output with best params for FULL dataset
# 6. Save results

# === 6. Save Results ===
# submodel_output.csv: columns = [cny_demand_momentum_z]
# training_result.json: feature, attempt, best_params, metrics, output_shape
```

#### Key Implementation Notes

1. **NO pip install needed**: Unlike attempt 1 which required hmmlearn, attempt 2 uses only pre-installed Kaggle packages (pandas, numpy, scipy, sklearn, optuna, yfinance).

2. **Single output column**: Output CSV must have exactly one column: `cny_demand_momentum_z`. This replaces the 3-column output from attempt 1. The column name change is intentional -- it signals to the meta-model builder that this is a different feature set.

3. **No HMM, no random seeds in feature generation**: The feature computation is fully deterministic given the data and parameters. Only Optuna's TPESampler uses a random seed (42).

4. **Gold returns for Optuna only**: GC=F is fetched solely for computing the Optuna objective (correlation and MI with gold_return_next on validation set). Gold data is NOT used in the feature computation itself. The feature is computed from CNY=X data alone.

5. **No lookahead bias**: Rolling windows are inherently backward-looking. `cny_return.rolling(W).sum()` at time t uses only data from t-W+1 to t. The z-score baseline similarly looks backward only.

6. **NaN handling**: The first `momentum_window + baseline_window` rows will be NaN. These should remain NaN in the output (not forward-filled) to avoid artificial zero-signal. The meta-model and evaluator handle NaN rows via alignment.

7. **Output alignment**: Trim output to match base_features date range. Use the existing `data/processed/base_features.csv` or `shared/schema_freeze.json` for the reference date range.

8. **Reproducibility**: Set `optuna.samplers.TPESampler(seed=42)`. The feature computation is deterministic -- same data + same parameters = same output.

9. **training_result.json must include**: feature="cny_demand", attempt=2, best_params (momentum_window, baseline_window), metrics (best Optuna value, validation correlation, validation MI), output_shape, output_columns=["cny_demand_momentum_z"].

10. **Dataset reference**: kernel-metadata.json must include `"dataset_sources": ["bigbigzabuton/gold-prediction-submodels"]`. The notebook reads existing submodel outputs from this dataset for cross-validation alignment.

---

## 8. Risks and Alternatives

### Risk 1: Single Feature May Still Fail Gate 3

- **Description**: Even with 1 optimized feature, the correlation with gold_return_next is only +0.035. This is a weak signal in absolute terms.
- **Mitigation**: The signal is orthogonal to all 20 existing features (max |r|=0.125). Even a weak signal, if orthogonal, adds marginal information. Options_market succeeded with a single feature of comparable weakness.
- **Fallback**: If attempt 2 fails, declare cny_demand as "no further improvement" and exclude from meta-model (which already achieves 3/4 targets without cny_demand).

### Risk 2: Optuna Window Optimization May Overfit Validation Set

- **Description**: With only 2 integer hyperparameters and small ranges, overfitting risk is low. But the signal is weak enough that validation-optimized windows may not generalize.
- **Mitigation**: The total parameter space is small (8 * 10 = 80 combinations). 30 trials with TPE covers a substantial fraction. The feature is deterministic (no stochastic training), so there is no variance from model initialization. Cross-validation via the evaluator's 5-fold rolling window will test generalization.
- **Fallback**: If Optuna-selected parameters fail Gate 3, the attempt is exhausted. Default parameters (momentum_window=5, baseline_window=60) can serve as a sanity check.

### Risk 3: Momentum Z-Score Autocorrelation

- **Description**: Measured autocorrelation at lag-1 is ~0.74 for the 5d/60d configuration. Different window combinations may have higher autocorrelation.
- **Mitigation**: All tested configurations (3d-10d momentum, 30d-120d baseline) showed autocorrelation well below 0.99. The shortest momentum windows (3d) have the lowest autocorrelation (~0.55). Datachecker will verify.
- **Threshold**: autocorrelation < 0.99 for Gate 1 PASS.

### Risk 4: Meta-Model Already Achieves 3/4 Targets Without CNY

- **Description**: Meta-model attempt 7 passes 3/4 targets without any cny_demand features. Adding cny_demand may not be worth the complexity.
- **Mitigation**: The remaining failing target may benefit from an orthogonal signal. Even if cny_demand does not help, the attempt costs minimal compute (2-3 minutes, CPU only) and provides useful information about the marginal value of CNY dynamics.
- **Decision**: This is the final attempt for cny_demand. If it fails Gate 3, cny_demand is permanently excluded from the meta-model.

---

## 9. VIF Analysis

### Proposed Feature VIF (Against 20 Meta-Model Features)

| Feature | Max |r| with existing features | Correlated with | Assessment |
|---------|-----------------------------------|----------------|------------|
| cny_demand_momentum_z | 0.125 | etf_pv_divergence | Excellent. Well below VIF=10 concern. |

### Detailed Correlations (All 20 Existing Features)

| Existing Feature | Correlation | Notes |
|------------------|-------------|-------|
| etf_pv_divergence | -0.125 | Highest. Both are momentum-type signals but in different assets (CNY vs ETF). |
| vix_mean_reversion_z | +0.086 | Weak. VIX and CNY have different dynamics. |
| vix_persistence | -0.075 | Weak. |
| ie_gold_sensitivity_z | +0.070 | Weak. Different mechanism (inflation breakeven vs CNY momentum). |
| tech_mean_reversion_z | +0.062 | Weak. Technical gold signals vs CNY momentum. |
| tech_trend_regime_prob | +0.054 | Negligible. |
| etf_capital_intensity | -0.039 | Negligible. |
| etf_regime_prob | +0.034 | Negligible. |
| vix_regime_probability | +0.032 | Negligible. |
| yc_regime_prob | -0.029 | Negligible. |
| xasset_divergence | -0.025 | Negligible. |
| options_risk_regime_prob | -0.027 | Negligible. |
| temporal_context_score | +0.024 | Negligible. |
| yc_curvature_z | +0.023 | Negligible. |
| xasset_recession_signal | -0.017 | Negligible. |
| ie_regime_prob | -0.005 | Negligible. |
| ie_anchoring_z | -0.002 | Negligible. |
| yc_spread_velocity_z | -0.003 | Negligible. |
| xasset_regime_prob | +0.000 | Negligible. |

**Summary**: All correlations with existing features are below |0.13|. The single cny_demand_momentum_z feature is essentially orthogonal to the entire existing meta-model feature set. VIF will be approximately 1.02-1.05, the lowest of any submodel feature.

### Improvement Over Attempt 1 VIF

Attempt 1 had 3 features with VIF ~1.11 each. Attempt 2 has 1 feature with projected VIF ~1.02. Fewer dimensions + lower VIF = cleaner signal for meta-model.

---

## 10. Expected Performance Against Gates

### Gate 1: Standalone Quality

- **Overfit ratio**: N/A (deterministic, no neural network)
- **No constant output**: CONFIRMED. Momentum z-score has std ~1.09, healthy distribution.
- **Autocorrelation < 0.99**: CONFIRMED. Measured ~0.74 for 5d/60d, will be in range 0.55-0.85 for all Optuna candidates.
- **NaN ratio**: < 5%. Only warmup period NaN (first ~130 rows of ~2,700).
- **Expected Result**: PASS (high confidence)

### Gate 2: Information Gain

- **MI increase > 5%**: Low probability. Single feature with MI ~0.016-0.025 against base MI sum of ~0.31. Expected MI increase: ~5-8% (marginal). Attempt 1 achieved only 0.09% with 3 features. Single optimized feature should do better due to less noise dilution.
- **VIF < 10**: CONFIRMED (projected VIF ~1.02). Best VIF of any submodel.
- **Rolling correlation std < 0.15**: High probability for momentum z-score (measured std ~0.135 in attempt 1, simpler feature should be comparable or better).
- **Expected Result**: MARGINAL (VIF excellent, MI uncertain, stability likely PASS)

### Gate 3: Ablation (Primary Target)

- **DA improvement > +0.5%**: This is the key improvement target. Attempt 1 degraded DA by -2.06% with 3 features. Removing 2 noise features should eliminate the degradation. Whether a single feature can achieve positive DA delta depends on the signal quality.
- **MAE improvement > -0.01%**: Probable. Attempt 1 achieved -0.0658 with 3 features. Single optimized feature should maintain some MAE benefit.
- **Sharpe improvement > +0.05**: Uncertain. Attempt 1 degraded Sharpe by -0.593. Single feature should eliminate most of this degradation. Positive delta requires the feature to add directional value.
- **Expected Result**: 50-60% probability of PASS via MAE, 30-40% probability of PASS via DA. Higher confidence than attempt 1 due to removing noise dimensions.

### Comparison with Attempt 1

| Aspect | Attempt 1 | Attempt 2 | Expected Improvement |
|--------|-----------|-----------|---------------------|
| Output columns | 3 | 1 | Reduces overfitting dimensions |
| ML model | HMM + z-scores | Pure z-score | Simpler, more stable |
| Dependencies | hmmlearn | None | Fewer failure modes |
| DA delta | -2.06% | Target: >= 0% | Critical improvement |
| MAE delta | -0.0658 | Target: < -0.01% | Maintain existing benefit |
| Sharpe delta | -0.593 | Target: >= 0 | Reduce degradation |
| VIF | 1.11 | ~1.02 | Marginal improvement |
| Execution time | 3-5 min | 2-3 min | Faster |

**Confidence**: 6/10 (moderate-to-good). The surgical fix -- removing noise dimensions while keeping the proven signal -- is well-motivated by attempt 1's data. The risk is that even the best single CNY feature may not overcome the inherent weakness of the CNY-gold signal at daily frequency.
