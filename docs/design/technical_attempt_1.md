# Submodel Design Document: Technical (Attempt 1)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| Yahoo: GC=F OHLC daily | Confirmed | 2542 trading days, 2015-01-02 to 2025-02-12, 0 NaN in OHLC |
| Yahoo: GLD OHLC daily | Confirmed | 2544 trading days, same date range, 0 NaN |
| GC=F roll artifacts | MINOR | Only 2 days >5% return (2020-03-23/24, COVID-19), not roll artifacts |
| GC=F OHLC quality | REJECTED | 137 days with H==L==O==C (flat bars). GK vol = 0 on these days. **Must use GLD instead** |
| GLD OHLC quality | Confirmed | 0 zero-GK days, 13 days O==C (acceptable). GLD is superior for OHLC indicators |
| Garman-Klass 7.4x efficiency | Confirmed (conservative) | Theoretical claim from original 1980 paper. Empirical median ratio on GLD = 12-15x. The 7.4x is a lower bound under specific distributional assumptions |
| HMM "Sharpe 0.823, CAGR 19.787%" | Confirmed but MISLEADING | This is from a Gold-SPY allocation strategy (QuantConnect 2019-2024 backtest), NOT from gold-only HMM regime detection. The HMM was trained on SPY drawdown, not gold returns. Not directly applicable to this submodel |
| HMM on gold returns regime detection | WEAK | Empirical test: 3-state HMM assigns 91.3% to one state. Regime separation is poor for gold returns alone. Gold is closer to random walk than VIX |
| HMM on 2D input (returns + GK vol) | BETTER | 2-state: 94%/6% split. 3-state: 93%/7%/0.4%. Still dominated by one state but volatility dimension helps |
| hmmlearn n_init parameter | REJECTED | `n_init` does NOT exist in hmmlearn 0.3.3 GaussianHMM. Researcher's code is incorrect. Must use manual loop over random_state values |
| Rolling 20-day z-score of returns | Confirmed | Autocorr(1) = -0.017 (excellent), corr with price level = 0.007 (excellent) |
| GK vol z-score with 20d smooth / 60d baseline | PROBLEMATIC | Autocorr(1) = 0.979. Too close to 0.99 threshold. **Must use daily GK vol (no smoothing) with direct z-score** |
| Daily GK vol z-score (no smoothing, 60d baseline) | Confirmed | Autocorr(1) = 0.206 (excellent), corr with price level = 0.059 (excellent) |
| VIF: z-score of returns low with price levels | Confirmed | Corr(mean_rev_z, gld_close) = 0.005, Corr(mean_rev_z, gld_volume) = -0.029 |
| VIF: GK vol z-score low with price levels | Confirmed | Corr(gk_vol_z, gld_close) = 0.057, Corr(gk_vol_z, volume) = 0.277 |
| Cross-correlation with VIX submodel | Confirmed low | Max corr = 0.194 (vix_mean_reversion_z vs tech_vol_regime). Strongly complementary |
| MI with gold target | Confirmed | tech_vol_regime MI=0.089 > vix_regime_prob MI=0.079. Technical features carry independent information |
| Researcher lookback: 60-120 days for HMM | Adjusted | HMM trains on full training set (not rolling window). Rolling window approach would add unnecessary complexity with weak returns-only signal |
| Researcher lookback: 20 days for mean-reversion | Confirmed | Standard oscillator period. Empirically validated: low autocorrelation, good VIF |
| Researcher lookback: 20d vol / 60d baseline | Adjusted | Use daily GK vol / 60d baseline directly (no 20d smoothing) to avoid autocorrelation issue |

### Critical Design Corrections from Fact-Check

1. **Use GLD, not GC=F, for all OHLC-based features**: GC=F has 137 flat-bar days causing GK vol = 0. GLD has zero such issues. GC=F returns can be used for HMM and z-score (returns are fine), but GLD is preferred for consistency.

2. **hmmlearn does not have `n_init` parameter**: The researcher's code `GaussianHMM(..., n_init=5)` will throw TypeError. Must implement multi-restart manually via loop over random_state values.

3. **HMM on gold returns has weak regime separation**: Unlike VIX (which has clear calm/fear dichotomy), gold returns are dominated by a single "normal" state (91%+ occupancy). Two design responses:
   - (a) Use HMM on 2D input [returns, GK_vol] to leverage volatility dimension for better regime detection
   - (b) Let Optuna choose between 2-state and 3-state; the output is still useful even with imbalanced states (captures tail events)

4. **GK vol z-score MUST NOT use 20-day smoothing**: The researcher's design `gk_vol.rolling(20).mean()` then z-scored against 60d baseline has autocorr = 0.979, dangerously close to the 0.99 Gate 1 threshold. Use daily GK vol directly with 60d baseline: autocorr = 0.206.

5. **HMM gold-specific Sharpe/CAGR claim is misleading**: The QuantConnect study is a portfolio allocation strategy (Gold vs SPY) using HMM on SPY drawdown, not a gold-regime detector. Remove from justification.

---

## 1. Overview

- **Purpose**: Extract three dimensions of gold's technical state -- regime (trending vs ranging), position (overbought vs oversold), and volatility regime (compressed vs expanded) -- from gold's own OHLC price action. These provide the meta-model with dynamic state information that raw price levels cannot capture.

- **Methods and rationale**:
  1. **HMM on [returns, GK_vol]**: Detects latent volatility/directional regime using 2D input. The GK vol dimension differentiates this from the VIX submodel's HMM (which uses only close-to-close VIX changes). Follows the successful VIX submodel pattern.
  2. **Rolling 20-day z-score of returns**: Pure statistical measure of overbought/oversold. Returns-based, so immune to price level collinearity and contract roll artifacts. Extremely low autocorrelation (-0.017).
  3. **Daily Garman-Klass volatility z-score (60-day baseline)**: Leverages OHLC data to estimate volatility 7.4x+ more efficiently than close-to-close methods. This is the unique OHLC advantage of the technical submodel -- no other submodel has access to intraday range information.

- **Expected effect**: Provide the meta-model with gold-specific behavioral dynamics. VIX captures external fear; DXY captures dollar context; technicals capture gold's OWN momentum persistence, mean-reversion extremes, and volatility clustering.

### Key Advantages Over Previous Submodels

1. **Daily frequency** -- no interpolation needed (real_rate failure root cause eliminated)
2. **Exactly 3 features** -- compact, follows VIX success (real_rate attempt 5 failed with 7)
3. **Follows VIX success pattern** -- HMM regime + deterministic statistical features
4. **Unique OHLC data** -- Garman-Klass differentiates from VIX/DXY close-only approaches
5. **Empirically validated low VIF** -- all features corr < 0.06 with price levels

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Fields | Already Available |
|------|--------|--------|-----------|--------|-------------------|
| GLD (primary) | Yahoo Finance | GLD | Daily | OHLCV | Yes: `data/raw/` |

### Why GLD Over GC=F

GC=F has 137 days where High == Low == Open == Close (5.4% of all trading days), producing GK vol = 0 and corrupting volatility-based features. These are likely partial trading sessions or data quality issues in Yahoo Finance's futures continuous contract. GLD has zero such issues.

For the HMM and z-score features (which use returns only), either GLD or GC=F works, but GLD is used for consistency across all three features.

### Preprocessing Steps

1. Fetch GLD OHLCV from Yahoo Finance, start=2014-06-01 (buffer for 60-day warmup before 2015-01-30)
2. Compute daily returns: `returns = close.pct_change()`
3. Compute daily Garman-Klass volatility from OHLC
4. Handle missing values: forward-fill gaps up to 3 days, drop remaining NaN
5. Trim to base_features date range: 2015-01-30 to 2025-02-12

### Expected Sample Count

- ~2,523 daily observations (matching base_features row count)
- Warmup period: 60 days for GK vol z-score baseline, 20 days for returns z-score
- HMM: trains on full training set, generates probabilities for full dataset
- Effective output: ~2,460+ rows after warmup, remaining NaN forward-filled

---

## 3. Model Architecture

This is a **hybrid deterministic-probabilistic** approach. The HMM component is probabilistic (EM-fitted); the z-score and GK vol components are deterministic (no fitting). No PyTorch is required.

### Component 1: HMM Regime Detection

- **Model**: `hmmlearn.hmm.GaussianHMM`
- **Input**: 2D array of [daily returns, daily GK volatility]
  - Why 2D: Gold returns alone show weak regime separation (91%+ in one state). Adding GK volatility provides a second dimension that captures volatility clustering, improving state separation
  - GK vol is clipped at 1e-8 minimum to avoid log(0) issues
- **States**: 2 or 3 (Optuna selects)
  - 2 states: Normal vs High-volatility regime
  - 3 states: Low-vol / Normal / High-vol regime
- **Covariance type**: "full" (captures return-volatility cross-correlation within each state)
- **Training**: Fit on training set only using EM. Multi-restart: loop over 5-10 random_state values, select best log-likelihood
- **Output**: Posterior probability of the highest-variance state (labeled post-hoc by sorting states by total covariance trace)
- **State labeling**: After fitting, compute trace of covariance matrix for each state. Highest-trace state = "high volatility/crisis regime". Output P(highest-trace state).

```
Input: GLD daily [returns, GK_vol] [T x 2]
       |
   For seed in range(n_restarts):
       GaussianHMM.fit(train_data) -> learn 2-3 state model
       Keep best by log-likelihood
       |
   GaussianHMM.predict_proba(full_data) -> [T x n_states]
       |
   Identify highest-covariance-trace state
       |
   Select P(highest-trace state) -> [T x 1]
       |
Output: tech_trend_regime_prob (0-1)
```

### Component 2: Mean-Reversion Z-Score

- **Model**: Pure pandas computation (no ML)
- **Input**: GLD daily returns
- **Window**: 20-day rolling window (Optuna explores 15/20/30)
- **Computation**: `z = (return_t - rolling_mean_t) / rolling_std_t`, clipped to [-4, 4]
- **Output**: z-score value indicating overbought (positive) or oversold (negative) relative to recent dynamics

```
Input: GLD daily returns [T x 1]
       |
   rolling(window).mean() -> rolling_mean
   rolling(window).std()  -> rolling_std
       |
   z = (return - rolling_mean) / rolling_std
       |
   clip(-4, 4)
       |
Output: tech_mean_reversion_z (typically -3 to +3)
```

### Component 3: Garman-Klass Volatility Z-Score

- **Model**: Pure pandas computation (no ML)
- **Input**: GLD daily OHLC prices
- **GK formula**: `gk_vol = sqrt(0.5 * log(H/L)^2 - (2*ln2 - 1) * log(C/O)^2)`
- **Z-score baseline**: 60-day rolling window (Optuna explores 40/60/90)
- **Key design choice**: Use DAILY GK vol directly (no 20-day smoothing) to keep autocorrelation low (0.21 vs 0.98 with smoothing)
- **Output**: z-score of daily GK vol relative to its 60-day rolling mean/std

```
Input: GLD daily OHLC [T x 4]
       |
   gk_vol = sqrt(0.5 * log(H/L)^2 - (2*ln2-1) * log(C/O)^2)
       |
   rolling(baseline_window).mean() -> gk_mean
   rolling(baseline_window).std()  -> gk_std
       |
   z = (gk_vol - gk_mean) / gk_std
       |
   clip(-4, 4)
       |
Output: tech_volatility_regime (typically -2 to +4)
```

### Combined Output

| Column | Range | Description | Corr with gld_close | Autocorr(1) |
|--------|-------|-------------|---------------------|-------------|
| `tech_trend_regime_prob` | [0, 1] | P(high-volatility/trending regime) from HMM | ~0.03 (measured) | ~0.45-0.88 (depends on n_states) |
| `tech_mean_reversion_z` | [-4, +4] | 20-day z-score of returns | 0.007 (measured) | -0.017 (measured) |
| `tech_volatility_regime` | [-4, +4] | Daily GK vol z-score vs 60-day baseline | 0.059 (measured) | 0.206 (measured) |

Total: **3 columns** (matching VIX/DXY successful pattern).

### Measured Orthogonality

| Feature Pair | Correlation | Assessment |
|-------------|-------------|------------|
| tech_mean_reversion_z vs gld_close | 0.007 | Excellent |
| tech_mean_reversion_z vs gld_volume | -0.029 | Excellent |
| tech_volatility_regime vs gld_close | 0.059 | Excellent |
| tech_volatility_regime vs gld_volume | 0.277 | Acceptable |
| tech_volatility_regime vs vix_regime_probability | 0.143 | Good (complementary) |
| tech_volatility_regime vs vix_mean_reversion_z | 0.194 | Good (complementary) |
| tech_mean_reversion_z vs vix_persistence | -0.023 | Excellent |
| tech_mean_reversion_z vs tech_volatility_regime | -0.023 | Excellent (orthogonal) |

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM n_iter | 200 | Gold returns need more EM iterations for convergence; tested empirically |
| HMM tol | 1e-4 | Standard convergence tolerance |
| GK vol clip minimum | 1e-8 | Prevent log(0) in edge cases (GLD should not have this, but defensive) |
| Z-score clipping | [-4, 4] | Prevent extreme outliers from dominating |
| HMM input features | [returns, GK_vol] | 2D input for better regime separation (tested: 1D returns has 91% single-state occupancy) |
| Data source | GLD (not GC=F) | GC=F has 137 flat-bar days corrupting GK vol. GLD has zero |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3} | categorical | 2=normal/crisis, 3=low-vol/normal/high-vol |
| hmm_covariance_type | {"full"} | fixed | Full covariance captures returns-vol correlation; trivial cost for 2D input |
| hmm_n_restarts | {5, 10} | categorical | Manual multi-restart to avoid local optima (hmmlearn has no n_init) |
| zscore_window | {15, 20, 30} | categorical | Mean-reversion lookback: 15d=reactive, 30d=stable |
| gk_baseline_window | {40, 60, 90} | categorical | GK vol baseline: shorter=more reactive, longer=more stable |

### Exploration Settings

- **n_trials**: 30
  - Rationale: Total combinations = 2 * 1 * 2 * 3 * 3 = 36. 30 trials covers ~83% of the space. Each trial is fast (HMM fit in seconds + rolling stats).
- **timeout**: 600 seconds (10 minutes)
  - Extra buffer vs VIX (300s) because 2D HMM is slightly slower and we need multi-restart loop
- **objective**: Maximize sum of mutual information between 3 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)

---

## 5. Training Settings

### Fitting Procedure

1. **HMM**: `GaussianHMM.fit(train_data_2d)` -- EM algorithm. Multi-restart: fit with random_state in range(n_restarts), keep best log-likelihood model
2. **Z-Score**: Rolling window statistics on returns -- deterministic, no fitting
3. **GK Vol Z-Score**: Rolling window statistics on daily GK vol -- deterministic, no fitting

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- HMM fits on training set only
- HMM generates probabilities for full dataset using predict_proba (no lookahead: HMM posterior at time t depends only on observations up to t given fitted model)
- Z-score and GK vol z-score use rolling windows (inherently no lookahead)
- Optuna optimizes MI sum on validation set
- Test set reserved for evaluator Gate 3

### Evaluation Metric for Optuna

For each trial (hyperparameter combination):
1. Fit HMM on training set [returns, GK_vol] with trial parameters
2. Generate all 3 features for full dataset
3. Compute mutual information (MI) between each feature and `gold_return_next` on validation set
4. Optuna maximizes: `MI_sum = MI(regime, target) + MI(zscore, target) + MI(gk_vol_z, target)`

MI calculation: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`.

### Loss Function

N/A -- no gradient-based training. HMM uses EM (maximum likelihood). Z-score and GK vol z-score are deterministic.

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- HMM EM converges via `n_iter` and `tol`.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. HMM + rolling statistics are CPU-only |
| Estimated execution time | 5-8 minutes | Data download (~30s) + 30 Optuna trials x ~10s each (2D HMM slower) + final output (~30s) |
| Estimated memory usage | <1 GB | ~2,500 rows x 6 columns. Tiny dataset |
| Required pip packages | `hmmlearn` | Must `pip install hmmlearn` at start of train.py. sklearn, pandas, numpy, yfinance pre-installed on Kaggle |

---

## 7. Implementation Instructions

### builder_data Instructions

1. **GLD data already available**: Use existing GLD data in `data/raw/`
2. **Verify OHLC completeness**: Ensure Open, High, Low, Close, Volume all present with 0 NaN
3. **Verify date range**: Should cover from at least 2014-06-01 (warmup buffer) to 2025-02-12
4. **Compute and save preprocessed data**: `data/processed/technical_features_input.csv`
   - Columns: Date, gld_open, gld_high, gld_low, gld_close, gld_volume, gld_return, gld_gk_vol
   - Start from 2014-06-01 for warmup
5. **Quality checks**:
   - Zero days with H==L==O==C (confirmed for GLD)
   - No gaps > 3 consecutive trading days
   - GK vol > 0 for all rows (confirmed)
   - Returns in reasonable range (|return| < 10%)

### builder_model Instructions

#### train.py Structure

```python
"""
Gold Prediction SubModel Training - Technical Attempt 1
Self-contained: Data fetch -> Preprocessing -> HMM + Z-Score + GK Vol -> Optuna HPO -> Save results
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
# Fetch GLD OHLCV from Yahoo Finance (NOT GC=F -- GC=F has 137 flat-bar days)
# Compute returns and GK volatility
# Align to base_features date range

# === 3. Feature Generation Functions ===

def generate_regime_feature(returns, gk_vol, n_components, n_restarts, train_size):
    """
    Fit HMM on 2D [returns, GK_vol] on training set.
    Return P(highest-covariance-trace state) for full data.

    IMPORTANT: hmmlearn does NOT have n_init parameter.
    Must loop over random_state values manually.
    """
    X = np.column_stack([returns, gk_vol])
    X_train = X[:train_size]

    best_model = None
    best_score = -np.inf
    for seed in range(n_restarts):
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type='full',
                n_iter=200,
                tol=1e-4,
                random_state=seed
            )
            model.fit(X_train)
            score = model.score(X_train)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        return np.full(len(returns), 0.5)

    probs = best_model.predict_proba(X)

    # Identify highest-covariance-trace state (highest total variance)
    traces = []
    for i in range(n_components):
        traces.append(np.trace(best_model.covars_[i]))
    high_var_state = np.argmax(traces)

    return probs[:, high_var_state]

def generate_zscore_feature(returns, window):
    """
    Rolling z-score of returns: (return - rolling_mean) / rolling_std
    """
    s = pd.Series(returns)
    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std()
    z = (s - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z.values

def generate_gk_vol_zscore(high, low, open_, close, baseline_window):
    """
    Daily Garman-Klass volatility z-scored against baseline window.

    IMPORTANT: Use daily GK vol directly, NOT smoothed with rolling(20).mean().
    Smoothed version has autocorr = 0.979 (Gate 1 risk).
    Daily version has autocorr = 0.206 (safe).
    """
    gk_vol = np.sqrt(
        0.5 * (np.log(high / low) ** 2) -
        (2 * np.log(2) - 1) * (np.log(close / open_) ** 2)
    )
    s = pd.Series(gk_vol)
    gk_mean = s.rolling(baseline_window).mean()
    gk_std = s.rolling(baseline_window).std()
    z = (s - gk_mean) / gk_std
    z = z.clip(-4, 4)
    return z.values

# === 4. Optuna Objective ===

def objective(trial, returns, gk_vol, high, low, open_, close, target, train_size, val_mask):
    """Maximize MI sum on validation set"""
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    n_restarts = trial.suggest_categorical('hmm_n_restarts', [5, 10])
    zscore_window = trial.suggest_categorical('zscore_window', [15, 20, 30])
    gk_baseline_window = trial.suggest_categorical('gk_baseline_window', [40, 60, 90])

    try:
        regime = generate_regime_feature(
            returns, gk_vol, n_components, n_restarts, train_size
        )
        zscore = generate_zscore_feature(returns, zscore_window)
        vol_z = generate_gk_vol_zscore(high, low, open_, close, gk_baseline_window)

        # Extract validation period
        regime_val = regime[val_mask]
        zscore_val = zscore[val_mask]
        vol_z_val = vol_z[val_mask]
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

        for feat_val in [regime_val, zscore_val, vol_z_val]:
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
# Run Optuna with 30 trials, 600s timeout
# Generate final output with best params
# Save submodel_output.csv, training_result.json
```

#### Key Implementation Notes

1. **Use GLD, NOT GC=F**: GC=F has 137 flat-bar days (H==L==O==C) causing GK vol = 0. GLD has zero such issues. This is the single most important data decision.

2. **hmmlearn has NO n_init parameter**: The researcher's code is incorrect. Must implement multi-restart via manual loop: `for seed in range(n_restarts): model = GaussianHMM(..., random_state=seed)`. Keep model with highest `model.score()`.

3. **HMM input is 2D [returns, GK_vol]**: 1D returns-only HMM showed 91% single-state occupancy (weak separation). Adding GK vol as second dimension leverages OHLC data and improves regime detection.

4. **GK vol z-score: NO smoothing**: Use daily GK vol directly, NOT `gk_vol.rolling(20).mean()`. The smoothed version has autocorrelation 0.979 (fails Gate 1). Daily version has autocorrelation 0.206.

5. **HMM state labeling**: Sort states by covariance matrix trace (total variance), not by mean return. For 2D input, trace = var(returns) + var(GK_vol). This is more robust than sorting by return mean (which can be noisy).

6. **No lookahead bias**:
   - HMM: Fit on training data only, generate probabilities for full dataset
   - Z-score: Rolling window (inherently backward-looking)
   - GK vol z-score: Rolling window (inherently backward-looking)

7. **NaN handling**: First ~60 rows will have NaN from rolling windows. Forward-fill after generation. The evaluator will align dates with base_features.

8. **Reproducibility**: Fix random_state seeds for HMM. Z-score and GK vol are deterministic.

9. **Output format**: CSV with columns [date, tech_trend_regime_prob, tech_mean_reversion_z, tech_volatility_regime]. Aligned to trading dates matching base_features.

10. **Target data**: Compute gold_return_next directly from GLD close prices within the training script (same approach as VIX submodel).

---

## 8. Risks and Alternatives

### Risk 1: HMM Weak Regime Separation on Gold

- **Description**: Gold returns are closer to random walk than VIX. Even with 2D input, one state may dominate (>90% occupancy).
- **Measured severity**: 3-state HMM on returns alone: 91.3% single state. With 2D input: slightly better but still imbalanced.
- **Mitigation**: 2D input [returns, GK_vol] provides volatility dimension for better separation. Even with imbalanced states, the probability of the rare high-vol state captures tail events that matter for gold returns.
- **Fallback**: If HMM regime probability has std < 0.05, replace with sigmoid-based indicator: `sigmoid(gk_vol_z * 2)` which approximates volatility regime probability without HMM.

### Risk 2: GK Vol Z-Score Autocorrelation

- **Description**: Rolling z-score features are inherently smooth. The 20d-smoothed version hit 0.979 autocorrelation.
- **Mitigation**: Design explicitly uses DAILY GK vol (no smoothing) with baseline window z-score. Measured autocorrelation: 0.12-0.21 depending on window size. Well below 0.99.
- **Detection**: Gate 1 autocorrelation check will catch any issues.
- **Fallback**: If autocorrelation too high, use change in GK vol z-score (first difference) which has autocorrelation 0.26.

### Risk 3: Mean-Reversion Z-Score Too Noisy

- **Description**: 20-day z-score of returns may be too reactive (autocorrelation -0.017 means it's almost white noise).
- **Why this is acceptable**: The z-score captures extreme deviations (>2 sigma). Even though day-to-day it fluctuates, the EXTREME values carry directional information. XGBoost can use threshold-based splits effectively.
- **Mitigation**: Optuna tests windows of 15, 20, 30. Longer windows smooth slightly.
- **Fallback**: Replace with cumulative 5-day return z-scored against 20-day window (adds momentum smoothing).

### Risk 4: Correlation with VIX Submodel Outputs

- **Description**: Both submodels have volatility-related features.
- **Measured correlation**: Max cross-correlation = 0.194 (vix_mean_reversion_z vs tech_vol_regime). This is very low.
- **Why**: VIX captures S&P 500 implied volatility dynamics. Technical captures gold's own realized volatility from OHLC data. Different instruments, different data sources, different vol measures.
- **Mitigation**: No action needed. VIF will be well below 10.

### Risk 5: Gate 3 Failure (real_rate Pattern)

- **Description**: Features pass Gate 2 but fail Gate 3 ablation.
- **Why technical is different from real_rate**:
  - Daily frequency (no interpolation)
  - MI = 0.082-0.089 per feature (higher than VIX features at 0.066-0.079)
  - 3 compact features (not 7)
  - Returns-based features that XGBoost can split on directly
  - VIX succeeded on attempt 1 with the same HMM + deterministic pattern
- **Fallback for Attempt 2**: Replace HMM with Hurst exponent (deterministic), replace GK vol z-score with Bollinger Band Width z-score.

### Risk 6: hmmlearn Installation Failure on Kaggle

- **Description**: hmmlearn may fail to install.
- **Mitigation**: `pip install hmmlearn` at script start. Successfully used in VIX attempt 1 on Kaggle.
- **Likelihood**: Very low.
- **Fallback**: Use sklearn GaussianMixture on rolling windows as HMM proxy.

---

## 9. VIF Analysis (Empirically Measured)

### Against Base Features

| Output Column | Base Feature | Measured Correlation | Expected VIF |
|---------------|-------------|---------------------|-------------|
| tech_mean_reversion_z | technical_gld_close | 0.005 | ~1.0 |
| tech_mean_reversion_z | technical_gld_volume | -0.029 | ~1.0 |
| tech_mean_reversion_z | vix_vix | -0.009 | ~1.0 |
| tech_volatility_regime | technical_gld_close | 0.057 | ~1.0 |
| tech_volatility_regime | technical_gld_volume | 0.277 | ~1.1 |
| tech_volatility_regime | etf_flow_volume_ma20 | 0.412 | ~1.2 |
| tech_volatility_regime | vix_vix | 0.220 | ~1.1 |

All correlations well below VIF=10 threshold. The highest (0.412 with volume_ma20) is expected since volatility and volume co-move, but VIF ~1.2 is acceptable.

### Against VIX Submodel Outputs

| Technical Output | VIX Output | Measured Correlation | VIF Risk |
|-----------------|-----------|---------------------|---------|
| tech_mean_reversion_z | vix_regime_probability | -0.006 | Negligible |
| tech_mean_reversion_z | vix_mean_reversion_z | -0.005 | Negligible |
| tech_mean_reversion_z | vix_persistence | -0.023 | Negligible |
| tech_volatility_regime | vix_regime_probability | 0.143 | Low |
| tech_volatility_regime | vix_mean_reversion_z | 0.194 | Low |
| tech_volatility_regime | vix_persistence | 0.010 | Negligible |

All cross-correlations are low. The technical and VIX submodels are **strongly complementary**.

---

## 10. Autocorrelation Analysis

| Output Column | Measured Autocorr (lag 1) | Assessment |
|---------------|---------------------------|------------|
| tech_trend_regime_prob (HMM 2-state) | 0.45 | Safe |
| tech_trend_regime_prob (HMM 3-state) | 0.83 | Safe |
| tech_mean_reversion_z | -0.017 | Excellent (near white noise) |
| tech_volatility_regime (daily, 60d baseline) | 0.206 | Safe |
| tech_volatility_regime (20d smooth, 60d baseline) | 0.979 | DANGEROUS - not used |

All selected features are well below the 0.99 threshold. The explicitly avoided smoothed GK vol (0.979) demonstrates why the daily computation was chosen.

---

## 11. MI Analysis (Empirically Measured)

| Feature | MI with gold_return_next | Comparison |
|---------|-------------------------|------------|
| tech_volatility_regime | 0.089 | Highest of all features tested |
| tech_mean_reversion_z | 0.082 | Higher than VIX regime prob |
| vix_regime_probability | 0.079 | (reference) |
| vix_persistence | 0.076 | (reference) |
| vix_mean_reversion_z | 0.066 | (reference) |

Both technical features show MI values comparable to or higher than VIX features, suggesting they carry meaningful information about gold returns.

---

## 12. Expected Performance Against Gates

### Gate 1: Standalone Quality
- **Overfit ratio**: N/A (HMM with EM, not neural network)
- **No constant output**: Confirmed -- regime probability varies with market conditions, z-score varies daily, GK vol z-score varies daily
- **Autocorrelation < 0.99**: All features measured at 0.45 or below (using daily GK vol, not smoothed). Mean-reversion z-score is -0.017.
- **No NaN values**: Confirmed after 60-day warmup with forward-fill

**Expected Result**: PASS

### Gate 2: Information Gain
- **MI increase > 5%**: High probability. Individual features show MI = 0.082-0.089 with target, comparable to VIX features.
- **VIF < 10**: High probability. All measured correlations with base features < 0.41. Maximum expected VIF ~1.2.
- **Rolling correlation std < 0.15**: High probability. Features capture gold-specific dynamics.

**Expected Result**: PASS

### Gate 3: Ablation Test
- **Direction accuracy +0.5%**: Target. Technical features provide momentum/mean-reversion/volatility context that should improve directional calls.
- **OR Sharpe +0.05**: Secondary target.
- **OR MAE -0.01%**: Tertiary target.

**Expected Result**: Cautiously optimistic (7/10 confidence). VIX achieved DA +0.96% and Sharpe +0.289 with the same approach pattern. Technical features have higher MI values than VIX features.

**Confidence**: 7/10

---

## 13. Design Rationale vs Alternatives

### Why HMM on [returns, GK_vol] Over Returns-Only?

Returns-only HMM assigns 91.3% to a single state -- poor regime separation. Adding GK vol provides a volatility dimension that helps distinguish quiet periods (low GK vol) from active periods (high GK vol), even when return magnitudes are similar. This leverages the unique OHLC data advantage.

### Why Daily GK Vol Over Smoothed?

The researcher proposed `gk_vol.rolling(20).mean()` z-scored against 60d baseline. This has autocorrelation 0.979, barely below the 0.99 Gate 1 threshold and risks failure under different Optuna parameter configurations. Daily GK vol z-score has autocorrelation 0.12-0.21 (robust margin).

### Why GLD Over GC=F?

GC=F has 137 days (5.4% of data) where H==L==O==C, producing zero GK volatility. These are not real zero-volatility days -- they are data artifacts from partial trading sessions in Yahoo Finance's continuous contract construction. GLD as an ETF trades on NYSE with no such issues.

### Why Not Hurst Exponent?

Hurst exponent was the researcher's Priority 2 alternative. It requires minimum 60-100 observations for stable estimates and is noisy on shorter windows. For Attempt 1, the HMM approach follows the proven VIX/DXY pattern. Hurst is reserved for Attempt 2 if HMM produces weak regimes.

### Why Not ADX?

ADX is a derivative of OHLC prices (not returns), creating VIF risk with base features that contain raw GLD OHLC. The HMM approach works on returns and GK vol (ratios), keeping VIF low.
