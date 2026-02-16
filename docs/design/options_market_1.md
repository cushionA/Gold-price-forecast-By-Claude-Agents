# Submodel Design Document: Options Market (Attempt 1)

## 0. Fact-Check Results

| Claim / Source | Result | Detail |
|----------------|--------|--------|
| Yahoo: ^CPCE (Equity P/C Ratio) | UNAVAILABLE | HTTP 404 "Quote not found for symbol: ^CPCE". Ticker does not exist on Yahoo Finance. |
| Yahoo: ^CPC (Total P/C Ratio) | UNAVAILABLE | HTTP 404 "Quote not found for symbol: ^CPC". Ticker does not exist on Yahoo Finance. |
| FRED: any P/C ratio series | NOT FOUND | FRED search returns no put/call ratio series. Not available on FRED. |
| Yahoo: ^SKEW (CBOE SKEW Index) | CONFIRMED | 280 rows since 2025-01-01, data from 2015+ available. Latest: 139.35 (2026-02-13). |
| FRED: GVZCLS (Gold Vol Index) | CONFIRMED | 279 non-null values since 2025-01-01. Latest: 30.23 (2026-02-12). Daily frequency. |
| Yahoo: ^GVZ (Gold Vol backup) | CONFIRMED | 280 rows since 2025-01-01. Matches FRED GVZCLS. Usable as fallback. |
| FRED: VIXCLS (for GVZ context) | CONFIRMED | Available, latest: 20.82 (2026-02-12). |
| Yahoo: ^VIX3M | CONFIRMED | 280 rows, available for VIX term structure if needed. |
| Researcher: "SKEW-VIX low correlation" | CONFIRMED | Measured: r = -0.2051 (levels). Orthogonal as claimed. |
| Researcher: "SKEW captures distinct risk dimension from VIX" | CONFIRMED | Empirically confirmed: SKEW z-score has <0.27 correlation with all VIX submodel outputs. |
| Researcher: "P/C ratio captures positioning distinct from VIX" | CANNOT VERIFY | P/C ratio data unavailable from any accessible source. |
| Researcher: "GVZ/VIX ratio removes common VIX component" | REJECTED | Measured: GVZ/VIX ratio z-score has -0.67 correlation with vix_mean_reversion_z. VIF will be extremely high. |
| Researcher: "Expected VIF <5 for GVZ/VIX ratio" | INCORRECT | Measured correlation -0.67 with VIX submodel implies VIF >> 5. GVZ-VIX spread even worse (-0.72). |
| Researcher: "2D HMM on [P/C changes, SKEW changes]" | INFEASIBLE | P/C ratio unavailable. Must redesign HMM inputs. |
| Researcher: "SKEW is very noisy" (citing academic lit) | CONFIRMED | SKEW change has r=-0.006 with next-day gold return. Near-zero predictive power in raw form. |
| Researcher: "No academic evidence of P/C or SKEW predicting gold" | CONFIRMED | All measured correlations with next-day gold return are <0.06. |
| Method: HMM for regime detection | VALID | Proven in 7/8 completed submodels. Appropriate for this data. |
| Method: 2D HMM (proven pattern) | VALID | Technical, Cross-Asset, ETF-Flow all succeeded with 2D HMM. |
| SKEW stationarity | CONFIRMED | ADF test: p=0.003. Stationary. |
| GVZ/VIX ratio stationarity | CONFIRMED | ADF test: p=0.003. Stationary. |
| SKEW change vs GVZ change correlation | CONFIRMED LOW | r = -0.03. Essentially uncorrelated. Good for 2D HMM. |

### Critical Design Corrections

**1. P/C Ratio Completely Unavailable.** The researcher flagged this as a risk ("^CPCE availability NOT CONFIRMED") and it materialized. Neither ^CPCE, ^CPC, nor any FRED series provides put/call ratio data. The entire P/C ratio component must be replaced.

**Resolution**: Replace P/C ratio with GVZ (Gold Volatility Index) changes as the second HMM dimension. GVZ and SKEW have -0.03 correlation in changes, confirming they capture distinct information. GVZ captures gold-specific implied volatility dynamics, while SKEW captures equity tail risk perception. Together they form a 2D representation of options-derived risk state.

**2. GVZ/VIX Ratio Has Extreme VIF with VIX Submodel.** The GVZ/VIX ratio z-score has -0.67 correlation with vix_mean_reversion_z. The GVZ-VIX spread z-score is even worse at -0.72. This makes any GVZ/VIX ratio-based standalone feature unacceptable.

**Resolution**: GVZ is used ONLY within the 2D HMM (as daily changes, not levels or ratios). No GVZ-based standalone z-score feature is included. The GVZ z-score (without VIX division) has 0.449 correlation with vix_mean_reversion_z -- still borderline. Safer to confine GVZ to the HMM where it contributes to regime detection alongside SKEW, rather than expose it as a raw feature.

**3. Researcher Confidence Adjusted.** Researcher stated 6/10 for Gate 3 pass and 4/10 for P/C-gold predictive power. Given that:
- P/C ratio is unavailable (core hypothesis weakened)
- All raw correlations with gold returns are <0.06
- This is entirely hypothesis-driven (no academic validation)

Adjusted confidence: 5/10 for Gate 3 pass. The submodel is worth attempting because HMM regime detection may capture nonlinear patterns invisible to raw correlations (this is exactly what succeeded in VIX, Technical, and ETF-Flow submodels).

---

## 1. Overview

- **Purpose**: Extract 3 contextual features from CBOE options market indicators (SKEW Index and Gold Volatility Index) that capture latent risk perception patterns -- specifically tail risk regimes, deviation from historical norms, and momentum of tail risk perception. These features complement the existing VIX submodel by capturing distribution SHAPE (skewness/tail risk) rather than distribution WIDTH (volatility level).
- **Core methods**:
  1. Hidden Markov Model (2-3 states) on 2D [SKEW daily changes, GVZ daily changes] for regime probability
  2. Rolling z-score of SKEW level for tail risk deviation
  3. Rolling momentum (5-10 day change) of SKEW, z-scored, for hedging acceleration
- **Why these methods**: HMM captures joint regime states between equity tail risk (SKEW) and gold-specific uncertainty (GVZ) -- states invisible in raw levels. Z-score captures how unusual current tail risk perception is relative to recent history. Momentum captures acceleration/deceleration of tail risk pricing, providing a leading indicator of regime transitions.
- **Expected effect**: Provide the meta-model with options-derived risk context that distinguishes scenarios with identical VIX levels but different tail risk environments (e.g., VIX=20 with SKEW=120 vs VIX=20 with SKEW=145 have very different implied return distributions).

### Key Difference from VIX Submodel

| Dimension | VIX Submodel | Options Market Submodel |
|-----------|-------------|------------------------|
| What it captures | Volatility LEVEL regime (how much uncertainty) | Tail risk SHAPE regime (how asymmetric the risk) |
| HMM input | 1D log-VIX changes | 2D [SKEW changes, GVZ changes] |
| Deterministic features | VIX z-score, VIX persistence | SKEW z-score, SKEW momentum |
| Information type | "How scared is the market?" | "How much tail risk is priced in, and is it gold-specific?" |
| Measured cross-correlation | N/A | SKEW z-score vs vix_regime_prob: r=-0.04 |

### Why Not P/C Ratio?

The CBOE Put/Call Ratio (^CPCE, ^CPC) is NOT available on Yahoo Finance or FRED. No free, programmatic, Kaggle-compatible data source exists. CBOE publishes daily P/C data on their website, but web scraping is unreliable in Kaggle notebooks and may violate terms of service. The P/C ratio must be excluded entirely.

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Delay | Status |
|------|--------|--------|-----------|-------|--------|
| SKEW Index | Yahoo Finance | ^SKEW | Daily | 0 days | CONFIRMED (2015-01-01 to present) |
| GVZ (Gold Vol) | FRED | GVZCLS | Daily | 0-1 days | CONFIRMED (2008-06-03 to present) |
| GVZ (backup) | Yahoo Finance | ^GVZ | Daily | 0 days | CONFIRMED (fallback if FRED fails) |
| VIX | FRED | VIXCLS | Daily | 0-1 days | CONFIRMED (needed for context, NOT as output) |

### Data NOT Used (with rationale)

| Data | Ticker | Reason for Exclusion |
|------|--------|---------------------|
| Equity P/C Ratio | ^CPCE | HTTP 404 on Yahoo Finance. Not on FRED. Unavailable. |
| Total P/C Ratio | ^CPC | HTTP 404 on Yahoo Finance. Not on FRED. Unavailable. |
| VIX/VIX3M Term Structure | ^VIX3M | Correlation 0.71 with VIX level. VIF issue with VIX submodel. |
| GVZ/VIX Ratio z-score | Derived | Correlation -0.67 with vix_mean_reversion_z. Extreme VIF risk. |
| GVZ-VIX Spread z-score | Derived | Correlation -0.72 with vix_mean_reversion_z. Even worse. |
| GVZ z-score (standalone) | Derived | Correlation 0.449 with vix_mean_reversion_z. Borderline VIF. |

### Preprocessing Steps

1. Fetch SKEW from Yahoo Finance (^SKEW), start=2014-10-01 (buffer for 90-day warmup before 2015-01-30)
2. Fetch GVZCLS from FRED (primary) or ^GVZ from Yahoo Finance (fallback), same date range
3. Compute daily changes: `skew_change = SKEW_t - SKEW_{t-1}`, `gvz_change = GVZ_t - GVZ_{t-1}`
4. Handle missing values: forward-fill gaps up to 3 days, then drop remaining NaN
5. Align SKEW and GVZ on common trading dates (inner join on date index)
6. Trim to base_features date range: 2015-01-30 to 2025-02-12 (plus warmup buffer)

### Expected Sample Count

- ~2,523 daily observations (matching base_features row count after alignment)
- Warmup period: up to 90 days for z-score rolling window
- Effective HMM input: ~2,730+ rows (2014-10-01 to 2025-02-12)
- Effective output: ~2,430+ rows after warmup, remaining filled with NaN then forward-filled

### Data Quality Expectations

| Metric | SKEW (^SKEW) | GVZ (GVZCLS) |
|--------|-------------|-------------|
| Mean | 134.56 | ~20-25 |
| Std | 12.21 | ~8-10 |
| Min | 110.34 | ~10 |
| Max | 183.12 | ~60 |
| Autocorr(1) level | 0.9507 | ~0.95 |
| Autocorr(1) change | ~0.15 | ~0.15 |

---

## 3. Model Architecture

This is a **hybrid deterministic-probabilistic** approach, not a neural network. No PyTorch is required. The pipeline consists of three independent components.

### Component 1: 2D HMM Regime Detection

- **Model**: `hmmlearn.hmm.GaussianHMM`
- **Input**: 2D array of [SKEW daily changes, GVZ daily changes]
  - SKEW changes: Captures shifts in equity tail risk perception
  - GVZ changes: Captures shifts in gold-specific implied volatility
  - Cross-correlation of these changes: r = -0.03 (essentially independent axes)
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
Output: options_risk_regime_prob (0-1)
```

**Expected Regime Interpretation (3-state case)**:
1. **Normal**: Small SKEW changes, small GVZ changes (routine market)
2. **Moderate risk**: SKEW elevated, GVZ moderate (equity tail risk without gold-specific stress)
3. **Extreme risk**: Large SKEW changes AND large GVZ changes (systemic stress affecting both equity options and gold options)

### Component 2: SKEW Tail Risk Z-Score

- **Model**: Pure pandas computation (no ML)
- **Input**: SKEW closing levels (^SKEW)
- **Window**: 60-day rolling window (Optuna explores {40, 60, 90})
- **Computation**: `z = (SKEW_t - rolling_mean_t) / rolling_std_t`
- **Output**: `options_tail_risk_z` in [-4, +4] (clipped)
  - Positive: SKEW above recent mean (elevated tail risk perception)
  - Negative: SKEW below recent mean (low tail risk, possible complacency)
  - Extreme positive (>2): Unusual tail risk pricing, potential safe-haven trigger

```
Input: SKEW daily close levels [T x 1]
       |
   rolling(window).mean() -> rolling_mean
   rolling(window).std()  -> rolling_std
       |
   z = (SKEW - rolling_mean) / rolling_std
       |
   clip(-4, 4) for stability
       |
Output: options_tail_risk_z (typically -2 to +3)
```

**VIF Analysis**: SKEW z-score vs VIX submodel outputs:
- vs vix_regime_probability: r = -0.04 (excellent)
- vs vix_mean_reversion_z: r = -0.27 (acceptable)
- vs vix_persistence: r = 0.06 (excellent)
- vs vix_vix (base feature): r = -0.18 (excellent)

### Component 3: SKEW Momentum (Rate of Change)

- **Model**: Pure pandas computation (no ML)
- **Input**: SKEW closing levels (^SKEW)
- **Window**: 5-day or 10-day change window (Optuna explores {5, 10, 15})
- **Computation**: `momentum = SKEW_t - SKEW_{t-window}`, then z-scored with 60-day rolling window
- **Output**: `options_skew_momentum_z` in [-4, +4] (clipped)
  - Positive: SKEW rising (tail risk perception increasing)
  - Negative: SKEW falling (tail risk perception decreasing)
  - Extreme positive: Rapid tail risk escalation (fear acceleration)

```
Input: SKEW daily close levels [T x 1]
       |
   momentum_raw = SKEW_t - SKEW_{t-window}
       |
   z = (momentum_raw - rolling(60).mean()) / rolling(60).std()
       |
   clip(-4, 4) for stability
       |
Output: options_skew_momentum_z (typically -3 to +3)
```

**VIF Analysis**: SKEW momentum vs VIX submodel outputs:
- vs vix_regime_probability: r = -0.02 (excellent)
- vs vix_mean_reversion_z: r = -0.10 (excellent)
- vs vix_persistence: r = 0.04 (excellent)
- vs vix_vix (base feature): r = 0.005 (nearly zero)

### Combined Output

| Column | Range | Description | Max Corr with VIX Submodel |
|--------|-------|-------------|---------------------------|
| `options_risk_regime_prob` | [0, 1] | P(extreme risk regime) from 2D HMM on [SKEW changes, GVZ changes] | Estimated <0.3 (regime on different dimensions) |
| `options_tail_risk_z` | [-4, +4] | SKEW deviation from 60-day rolling mean (tail risk deviation) | 0.27 (with vix_mean_reversion_z) |
| `options_skew_momentum_z` | [-4, +4] | SKEW rate-of-change z-score (tail risk acceleration) | 0.10 (with vix_mean_reversion_z) |

**Total: 3 columns** (within the recommended 2-4 range).

### Orthogonality Analysis (Measured and Estimated)

| Feature Pair | Correlation | Assessment |
|-------------|-------------|------------|
| options_risk_regime_prob vs vix_regime_probability | est. 0.1-0.3 | Acceptable (different HMM inputs) |
| options_tail_risk_z vs vix_mean_reversion_z | -0.27 (measured) | Acceptable |
| options_skew_momentum_z vs any VIX output | max 0.10 (measured) | Excellent |
| options_tail_risk_z vs options_skew_momentum_z | est. 0.3-0.5 | Moderate (both SKEW-derived, but level vs change) |
| options_risk_regime_prob vs options_tail_risk_z | est. 0.2-0.4 | Acceptable (HMM nonlinear vs linear z-score) |

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM n_iter | 100 | Standard convergence limit; EM typically converges in 20-50 iterations |
| HMM tol | 1e-4 | Standard convergence tolerance |
| HMM random_state | 42 | Reproducibility |
| HMM covariance_type | "full" | Required for 2D input to capture cross-correlation structure |
| Z-score clipping | [-4, 4] | Prevent extreme outliers from dominating meta-model |
| Momentum z-score window | 60 days | Rolling window for z-scoring the raw momentum |
| Autocorrelation lag (not used) | N/A | Persistence feature not included (would overlap with VIX submodel) |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3} | categorical | 2=normal/extreme, 3=calm/moderate/extreme |
| hmm_n_init | {3, 5, 10} | categorical | EM restarts to avoid local optima |
| skew_zscore_window | {40, 60, 90} | categorical | Tail risk lookback: shorter=reactive, longer=stable |
| skew_momentum_window | {5, 10, 15} | categorical | Momentum period: 5d=short-term, 15d=medium-term |

### Exploration Settings

- **n_trials**: 30
  - Rationale: 4 categorical parameters. Total combinations: 2 * 3 * 3 * 3 = 54. 30 trials with TPE provides good coverage. Each trial is fast (<5 seconds, no neural network).
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of mutual information between 3 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)
- **pruner**: None (trials too fast to benefit from pruning)

---

## 5. Training Settings

### Fitting Procedure

This is NOT a gradient-based training loop. The procedure is:

1. **HMM**: `GaussianHMM.fit(hmm_input_train)` -- EM algorithm on 2D [SKEW changes, GVZ changes], converges in seconds
2. **SKEW Z-Score**: Rolling window statistics on SKEW levels -- deterministic, no fitting
3. **SKEW Momentum**: Rolling window change + z-score on SKEW levels -- deterministic, no fitting

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- HMM fits on training set data only
- HMM generates probabilities for full dataset using `predict_proba` (no lookahead: HMM posterior at time t depends only on observations up to t given fitted model)
- Z-score and momentum use rolling windows (inherently backward-looking)
- Optuna optimizes MI sum on validation set
- Test set reserved for evaluator Gate 3

### Evaluation Metric for Optuna

For each trial (hyperparameter combination):
1. Fit HMM on training set [SKEW changes, GVZ changes]
2. Generate all 3 features for full dataset using fitted HMM and trial window parameters
3. Compute mutual information (MI) between each feature and `gold_return_next` on validation set
4. Optuna maximizes: `MI_sum = MI(regime_prob, target) + MI(tail_risk_z, target) + MI(momentum_z, target)`

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
| enable_gpu | false | No neural network. HMM + rolling statistics are CPU-only. |
| Estimated execution time | 3-5 minutes | Data download (~30s) + 30 Optuna trials x ~5s each (~2.5min) + final output (~30s) |
| Estimated memory usage | <1 GB | ~2,700 rows x 2-3 columns. Tiny dataset. |
| Required pip packages | hmmlearn | Must `pip install hmmlearn` at start of notebook. sklearn, pandas, numpy pre-installed on Kaggle. |

---

## 7. Implementation Instructions

### builder_data Instructions

1. **Fetch SKEW data**: Download ^SKEW from Yahoo Finance, start=2014-10-01 (buffer), end=2025-02-15
2. **Fetch GVZ data**: Download GVZCLS from FRED (primary). If FRED fails, use ^GVZ from Yahoo Finance.
3. **Fetch VIX data**: Download VIXCLS from FRED (needed only for reference/analysis, NOT as model input)
4. **Align dates**: Inner join SKEW and GVZ on date index. Forward-fill gaps up to 3 days.
5. **Compute derived columns**:
   - `skew_change = SKEW_t - SKEW_{t-1}`
   - `gvz_change = GVZ_t - GVZ_{t-1}`
6. **Save preprocessed data**: `data/processed/options_market_features_input.csv`
   - Columns: Date (index), skew_close, gvz_close, skew_change, gvz_change
   - Date range: 2014-10-01 to 2025-02-12 (includes warmup buffer)
7. **Quality checks**:
   - No gaps > 3 consecutive trading days
   - Missing data < 2% after forward-fill
   - SKEW values in range [100, 200]
   - GVZ values in range [5, 80]
   - Changes (daily diffs) have no extreme outliers (|SKEW change| < 30, |GVZ change| < 20)

### builder_model Instructions

#### Notebook Structure

```python
"""
Gold Prediction SubModel Training - Options Market Attempt 1
Self-contained: Data fetch -> Preprocessing -> HMM + Z-Score + Momentum -> Optuna HPO -> Save results
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
# SKEW from Yahoo Finance
# GVZ from FRED (GVZCLS), fallback to Yahoo (^GVZ)
# Gold target from base gold data
# Align on common dates
# Compute changes

# === 3. Feature Generation Functions ===

def generate_regime_feature(skew_changes, gvz_changes, n_components, n_init, train_size):
    """
    2D HMM on [SKEW changes, GVZ changes].
    Returns P(highest-trace-covariance state) for full data.
    """
    X = np.column_stack([skew_changes, gvz_changes])
    X_train = X[:train_size]

    model = GaussianHMM(
        n_components=n_components,
        covariance_type='full',
        n_iter=100,
        tol=1e-4,
        random_state=42,
        n_init=n_init
    )
    model.fit(X_train)
    probs = model.predict_proba(X)

    # Identify highest-trace (most volatile) state
    traces = [np.trace(model.covars_[i]) for i in range(n_components)]
    high_var_state = np.argmax(traces)
    return probs[:, high_var_state]

def generate_tail_risk_z(skew_levels, window):
    """
    Rolling z-score of SKEW level.
    High z = elevated tail risk perception relative to recent history.
    """
    s = pd.Series(skew_levels)
    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std()
    z = (s - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z.values

def generate_skew_momentum_z(skew_levels, momentum_window, zscore_window=60):
    """
    SKEW momentum (rate of change) z-scored.
    Captures acceleration/deceleration of tail risk perception.
    """
    s = pd.Series(skew_levels)
    momentum = s.diff(momentum_window)
    # Z-score the raw momentum
    rolling_mean = momentum.rolling(zscore_window).mean()
    rolling_std = momentum.rolling(zscore_window).std()
    z = (momentum - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z.values

# === 4. Optuna Objective ===

def objective(trial, skew_changes, gvz_changes, skew_levels, target, train_size, val_mask):
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    n_init = trial.suggest_categorical('hmm_n_init', [3, 5, 10])
    skew_zscore_window = trial.suggest_categorical('skew_zscore_window', [40, 60, 90])
    skew_momentum_window = trial.suggest_categorical('skew_momentum_window', [5, 10, 15])

    try:
        regime = generate_regime_feature(skew_changes, gvz_changes, n_components, n_init, train_size)
        tail_risk_z = generate_tail_risk_z(skew_levels, skew_zscore_window)
        momentum_z = generate_skew_momentum_z(skew_levels, skew_momentum_window)

        # Extract validation period
        regime_val = regime[val_mask]
        tail_risk_val = tail_risk_z[val_mask]
        momentum_val = momentum_z[val_mask]
        target_val = target[val_mask]

        # Compute MI sum
        def discretize(x, bins=20):
            valid = ~np.isnan(x)
            if valid.sum() < bins:
                return None
            x_c = x.copy()
            x_c[~valid] = np.nanmedian(x)
            return pd.qcut(x_c, bins, labels=False, duplicates='drop')

        mi_sum = 0.0
        for feat_val in [regime_val, tail_risk_val, momentum_val]:
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

1. **hmmlearn installation**: Must `pip install hmmlearn` at the start. Not pre-installed on Kaggle.
2. **FRED API key**: Use `os.environ['FRED_API_KEY']` from Kaggle Secrets. Fail with KeyError if not set.
3. **Yahoo Finance fallback for GVZ**: If FRED fails, use `yf.download('^GVZ')['Close']`.
4. **2D HMM input**: Stack [skew_changes, gvz_changes] into Nx2 array. Both should be daily differences (not levels).
5. **HMM state labeling**: After fitting, compute `np.trace(model.covars_[i])` for each state. Output P(highest-trace state). This identifies the "extreme/volatile" regime.
6. **No lookahead bias**:
   - HMM: Fit on training data only, predict_proba on full dataset
   - Z-score: Rolling window (inherently backward-looking)
   - Momentum: Rolling window change (inherently backward-looking)
7. **NaN handling**: First ~90 rows will have NaN (max of warmup periods). Forward-fill from first valid value. Evaluator aligns dates with base_features.
8. **Reproducibility**: Fix random_state=42 for HMM, seed=42 for Optuna. Z-score and momentum are deterministic.
9. **Output format**: CSV with columns [Date, options_risk_regime_prob, options_tail_risk_z, options_skew_momentum_z]. Aligned to trading dates.
10. **Output column names matter**: Use exact names above for consistency with meta-model feature registry.

---

## 8. Risks and Alternatives

### Risk 1: No Predictive Signal (HIGHEST RISK)

- **Description**: All measured raw correlations between SKEW/GVZ features and next-day gold return are <0.06. This submodel may provide zero information gain.
- **Probability**: 40% (highest risk factor)
- **Why proceed anyway**: HMM captures nonlinear regime patterns invisible to linear correlation. The VIX submodel had MI increase of only 0.68% (below Gate 2 threshold) but passed Gate 3 with DA +0.96%. Linear correlation is not the right test for regime-based information. Several successful submodels (VIX, ETF-Flow, Yield Curve) had weak linear correlations but strong nonlinear contributions through HMM regimes.
- **Detection**: Gate 2 MI increase < 5%
- **Mitigation for Attempt 2**: If Gate 2 fails, focus solely on GVZ-derived features (gold-specific, more direct connection to gold than SKEW).

### Risk 2: HMM State Collapse

- **Description**: With only ~1,700 training observations of 2D data, HMM may fail to find distinct states, collapsing to effectively 1 state (as happened with yield_curve HMM).
- **Probability**: 20%
- **Detection**: Check regime probability std < 0.05 or near-constant values
- **Mitigation**: Use n_init = {3, 5, 10} for multiple EM restarts. If collapse occurs, fall back to 1D HMM on SKEW changes only (follows VIX/Inflation pattern).

### Risk 3: SKEW Z-Score High Autocorrelation

- **Description**: SKEW level has autocorrelation of 0.95. Rolling z-score may inherit this, approaching the 0.99 Gate 1 threshold.
- **Probability**: 15%
- **Mitigation**: Z-score is computed from SKEW levels, not smoothed data. Daily SKEW variations (std=12.21, range 110-183) ensure z-score changes meaningfully. Expected autocorrelation: 0.85-0.95, well below 0.99.
- **Fallback**: If autocorrelation > 0.99, reduce window to 40 days or use change-in-z-score.

### Risk 4: VIF with VIX Submodel

- **Description**: Despite low measured correlations, the HMM regime probability from [SKEW, GVZ] may correlate with vix_regime_probability because GVZ and VIX co-move (measured r=0.65 at level).
- **Probability**: 15%
- **Mitigation**: Using GVZ CHANGES (not levels) in the HMM reduces level-based correlation. The HMM captures joint regime patterns, not individual indicator levels.
- **Detection**: Gate 2 VIF check
- **Fallback**: If VIF > 10, drop regime_prob and keep only SKEW z-score and momentum (which have VIF-safe measured correlations).

### Risk 5: SKEW Noisiness

- **Description**: Academic literature describes SKEW as "very noisy" with weak information content for option prices. This may translate to noisy features.
- **Probability**: 25%
- **Mitigation**: Z-scoring with 60-day window smooths out daily noise. Momentum uses 5-10 day windows, further filtering noise. HMM extracts systematic regime patterns from noise.
- **Fallback for Attempt 2**: Replace SKEW z-score with GVZ z-score (gold-specific, potentially stronger signal despite VIF concerns -- GVZ z-score vs vix_mean_reversion_z is 0.449, borderline but possibly acceptable if MI is strong enough).

### Alternative Design for Attempt 2 (If Attempt 1 Fails)

If this design fails at Gate 2 or Gate 3:
1. **Focus on GVZ-only**: Use 1D HMM on GVZ changes, GVZ z-score, GVZ momentum. Accept higher VIF with VIX submodel (0.449 correlation) if Gate 3 shows value.
2. **Add VIX term structure**: Use VIX/VIX3M ratio change (not level) as HMM input alongside GVZ changes. Despite 0.71 level correlation with VIX, the CHANGE in term structure ratio has lower correlation.
3. **Simpler approach**: Drop HMM entirely. Use only SKEW z-score + SKEW momentum + GVZ z-score as 3 deterministic features. Eliminates HMM collapse risk.

---

## 9. Expected Performance Against Gates

### Gate 1: Standalone Quality

- **Overfit ratio**: N/A (no neural network)
- **Constant output check**: Will pass -- SKEW varies meaningfully (std=12.21), GVZ varies meaningfully. Z-scores and momentum are non-constant by construction.
- **Autocorrelation < 0.99**: Expected 0.7-0.95 for all features (daily SKEW/GVZ volatility prevents near-constant outputs)
- **No NaN values**: Will pass after forward-filling warmup period

**Expected Result**: PASS (confidence 9/10)

### Gate 2: Information Gain

- **MI increase > 5%**: Uncertain. Raw correlations are weak (<0.06). HMM may capture nonlinear patterns. VIX submodel had MI=0.68% but still passed Gate 3.
- **VIF < 10**: High probability. All measured correlations with VIX submodel are <0.27 for standalone features. Regime prob TBD but expected <0.3.
- **Rolling correlation std < 0.15**: High probability. SKEW-derived features are distinct from VIX-derived features.

**Expected Result**: UNCERTAIN (confidence 5/10). This is the highest-risk gate for this submodel.

### Gate 3: Ablation Test

- **Direction accuracy +0.5%**: Possible if HMM regime captures tail risk regimes that help distinguish gold up/down days.
- **OR Sharpe +0.05**: Possible if tail risk features help avoid false signals during regime transitions.
- **OR MAE -0.01%**: Possible if SKEW/GVZ regime context improves return magnitude estimation.

**Expected Result**: MODERATE (confidence 5/10). This is a hypothesis-driven experiment with no empirical precedent.

### Overall Gate 3 Pass Probability: 5/10

This is lower than most completed submodels (VIX 7/10, Technical 7/10, ETF-Flow 8/10) because:
1. No P/C ratio (core hypothesis weakened)
2. No academic evidence linking SKEW/GVZ to gold returns
3. All raw correlations are <0.06
4. SKEW described as "very noisy" in academic literature

However, it is worth attempting because:
1. SKEW and GVZ are genuinely new information sources not captured by any existing feature
2. HMM may extract nonlinear patterns invisible to raw correlation
3. Multiple submodels (VIX, Yield Curve, CNY) passed Gate 3 despite failing Gate 2 MI tests
4. The 2D HMM + z-score + momentum pattern has a strong track record (5 successful submodels)

---

## 10. Autocorrelation Analysis

| Output Column | Expected Autocorrelation (lag 1) | Rationale |
|---------------|----------------------------------|-----------|
| options_risk_regime_prob | 0.7-0.9 | Regimes persist for days-weeks, but transitions provide variation. SKEW and GVZ are volatile enough to drive regime shifts. |
| options_tail_risk_z | 0.85-0.95 | 60-day rolling window creates smoothing, but SKEW has meaningful daily variation (std=12.21). |
| options_skew_momentum_z | 0.6-0.85 | Momentum is inherently more responsive than level z-score. 5-10 day window creates moderate smoothing. |

All expected values are well below the 0.99 threshold.
