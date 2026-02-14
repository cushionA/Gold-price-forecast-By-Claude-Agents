# Submodel Design Document: DXY (Attempt 1)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| Yahoo: DX-Y.NYB | Confirmed | 7 rows fetched for 2025-02-01..02-12, latest=107.31 |
| Yahoo: EURUSD=X | Confirmed | Daily data available, latest=1.0397 |
| Yahoo: JPY=X | Confirmed | USD/JPY format (~154), daily data available |
| Yahoo: GBPUSD=X | Confirmed | Daily data available, latest=1.2424 |
| Yahoo: CAD=X | Confirmed | USD/CAD format (~1.45), daily data available |
| Yahoo: SEK=X | Confirmed | USD/SEK format (~11.0), daily data available |
| Yahoo: CHF=X | Confirmed | USD/CHF format (~0.91), daily data available |
| DXY weights sum to 100% | Confirmed | EUR 57.6% + JPY 13.6% + GBP 11.9% + CAD 9.1% + SEK 4.2% + CHF 3.6% = 100.0% |
| hmmlearn library | Available on PyPI (v0.3.3) | Not installed locally but available for Kaggle via `pip install hmmlearn`. Must add to pip install in train.py |
| sklearn PCA | Confirmed | sklearn.decomposition.PCA works correctly |
| Currency pair directions | Verified | EURUSD=X and GBPUSD=X are inverse to DXY (Foreign/USD); JPY=X, CAD=X, SEK=X, CHF=X are same direction (USD/Foreign). PCA input must normalize all to USD-strength direction |
| HMM for FX regime detection | Confirmed valid | Well-established in FX literature; GaussianHMM with 2-3 states is standard |
| PCA for cross-currency divergence | Confirmed valid | PC1 of currency basket returns captures common USD factor; residual variance = divergence |
| 20-day realized volatility | Confirmed valid | Industry standard for FX volatility measurement |
| Researcher claim: "MI increase 10-15%" | Skeptical | Real_rate showed MI 10-39% across 5 attempts but all failed Gate 3. MI alone is not predictive of Gate 3 success |
| Researcher claim: "Gate 3 uncertain but promising" | Appropriate caution | Honest assessment. DXY has daily frequency advantage over real_rate |

### Critical Design Correction

**Currency direction normalization**: The researcher's code example fetches raw currency pair prices but does not address the direction convention. For PCA to work correctly, all 6 currency returns must be expressed in the same direction (USD strength = positive return). EURUSD=X and GBPUSD=X returns must be negated before PCA.

---

## 1. Overview

- **Purpose**: Extract three contextual features from DXY and its 6 constituent currencies that capture dollar regime dynamics, cross-currency agreement/divergence, and FX volatility state -- information absent from the raw `dxy_dxy` base feature.
- **Core methods**:
  1. Hidden Markov Model (2 states) on DXY returns for regime probability
  2. Rolling PCA on 6 constituent currency returns for cross-currency divergence
  3. Realized volatility z-score for volatility state
- **Why these methods**: All three capture fundamentally different aspects of USD dynamics. HMM captures temporal regime structure. PCA captures cross-sectional structure. Volatility z-score captures dispersion. None of these are available from the raw DXY level.
- **Expected effect**: Provide the meta-model with USD context that distinguishes between scenarios where the same DXY level has different gold implications (e.g., broad-based USD strength vs EUR-driven DXY move).

### Key Advantage Over real_rate

All data is daily frequency. No monthly-to-daily interpolation is needed. This eliminates the root cause of all 5 real_rate failures.

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Already Available |
|------|--------|--------|-----------|-------------------|
| DXY Index | Yahoo Finance | DX-Y.NYB | Daily | Yes: `data/raw/dxy.csv` (2545 rows, 2015-01-02 to 2025-02-13) |

### Expansion Data (6 constituent currencies)

| Currency Pair | Ticker | DXY Weight | Direction | Return Sign for PCA |
|---------------|--------|------------|-----------|---------------------|
| EUR/USD | EURUSD=X | 57.6% | Inverse (higher = weaker USD) | Negate |
| USD/JPY | JPY=X | 13.6% | Same (higher = stronger USD) | Keep |
| GBP/USD | GBPUSD=X | 11.9% | Inverse (higher = weaker USD) | Negate |
| USD/CAD | CAD=X | 9.1% | Same (higher = stronger USD) | Keep |
| USD/SEK | SEK=X | 4.2% | Same (higher = stronger USD) | Keep |
| USD/CHF | CHF=X | 3.6% | Same (higher = stronger USD) | Keep |

### Preprocessing Steps

1. Fetch all 7 tickers (DXY + 6 currencies) from Yahoo Finance, start=2014-12-01 (buffer for warmup)
2. Compute daily log-returns for all 7 series
3. For EURUSD=X and GBPUSD=X: negate returns (so positive = USD strength for all)
4. Align dates: inner join on trading dates, forward-fill gaps up to 3 days
5. Drop dates with any remaining NaN
6. Trim to base_features date range: 2015-01-30 to 2025-02-12

### Expected Sample Count

- ~2,500 daily observations (matching base_features: 2523 rows)
- Warmup period: 60 days for rolling PCA, 20 days for volatility
- Effective output: ~2,450+ rows after warmup

---

## 3. Model Architecture

This is a **hybrid deterministic-probabilistic** approach, not a single PyTorch neural network. No PyTorch is required. The pipeline consists of three independent components.

### Component 1: HMM Regime Detection

- **Model**: `hmmlearn.hmm.GaussianHMM`
- **Input**: 1D array of daily DXY log-returns (single feature)
- **States**: 2 (trending vs mean-reverting)
  - Rationale: 2 states avoids overfitting with ~2,500 observations. The researcher suggested 2-3; we choose 2 for simplicity and let Optuna test 3 as an option.
- **Covariance type**: "full" (single dimension, so full = diag = spherical for 1D)
- **Training**: Fit on all available data (no train/val split needed -- HMM is generative, not predictive)
- **Output**: Posterior probability of the higher-variance state (identified post-hoc)
- **State labeling**: After fitting, label states by their emission variance. The state with higher variance is "trending" (larger moves), the state with lower variance is "mean-reverting" (consolidation).

```
Input: DXY daily log-returns [T x 1]
       |
   GaussianHMM.fit() -> learn 2-state model
       |
   GaussianHMM.predict_proba() -> [T x 2] posterior probabilities
       |
   Select P(high-variance state) -> [T x 1]
       |
Output: dxy_regime_probability (0-1)
```

### Component 2: Rolling PCA Cross-Currency Divergence

- **Model**: `sklearn.decomposition.PCA`
- **Input**: 6-column matrix of daily currency returns (direction-normalized)
- **Window**: 60-day rolling window
  - Rationale: FX regimes typically last weeks to months. 60 days captures sufficient structure while allowing time variation. Shorter windows (20-30 days) would be too noisy.
- **Components**: Fit PCA with n_components=6 (full), extract explained_variance_ratio_[0] (PC1 share)
- **Output**: `divergence = 1 - explained_variance_ratio_[0]`
  - When PC1 explains 90%+ variance -> divergence near 0.1 (all currencies moving together)
  - When PC1 explains 50% variance -> divergence near 0.5 (currencies diverging)

```
Input: 6 currency daily returns (direction-normalized) [T x 6]
       |
   For each date t:
     PCA.fit(returns[t-59:t+1]) -> explained_variance_ratio_
     divergence[t] = 1 - explained_variance_ratio_[0]
       |
Output: dxy_cross_currency_div (typically 0.1 to 0.6)
       |
   MinMax scale to [0, 1] using expanding window min/max (no lookahead)
```

### Component 3: Realized Volatility Z-Score

- **Model**: Pure pandas computation (no ML)
- **Input**: DXY daily log-returns
- **Window**: 20-day rolling standard deviation (industry standard)
- **Normalization**: Expanding-window z-score (no lookahead)
  - `z = (vol_20d - expanding_mean(vol_20d)) / expanding_std(vol_20d)`

```
Input: DXY daily log-returns [T x 1]
       |
   rolling(20).std() -> realized_vol [T x 1]
       |
   expanding z-score (mean and std computed using only past data)
       |
Output: dxy_volatility_z (unbounded, typically -2 to +3)
```

### Combined Output

| Column | Range | Description |
|--------|-------|-------------|
| `dxy_regime_probability` | [0, 1] | P(high-variance/trending regime) from HMM |
| `dxy_cross_currency_div` | [0, 1] | Cross-currency divergence from rolling PCA |
| `dxy_volatility_z` | unbounded | Realized vol z-score (20-day window, expanding normalization) |

Total: **3 columns** (strictly within the 2-4 target range).

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM n_iter | 100 | Standard convergence limit; EM typically converges in 20-50 iterations |
| HMM tol | 1e-4 | Standard convergence tolerance |
| HMM random_state | 42 | Reproducibility |
| PCA window | 60 days | FX regime duration; balances signal freshness vs noise |
| Volatility window | 20 days | Industry standard for realized FX volatility |
| Volatility z-score | expanding window | Prevents lookahead bias |
| PCA divergence scaling | expanding MinMax | Prevents lookahead bias |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3} | categorical | 2 states = simple regime; 3 states adds neutral/transition regime |
| hmm_covariance_type | {"full", "diag"} | categorical | Full captures full covariance (trivial for 1D input); diag is simpler |
| hmm_n_init | {3, 5, 10} | categorical | Number of EM initializations to avoid local optima |
| pca_window | {40, 60, 90} | categorical | Shorter=more responsive, longer=more stable |
| vol_window | {10, 20, 30} | categorical | Standard range for realized volatility lookback |

### Exploration Settings

- **n_trials**: 30
  - Rationale: 5 parameters, all categorical with small ranges. 30 trials covers most of the 2 * 2 * 3 * 3 * 3 = 108 possible combinations well. Each trial is fast (<10 seconds).
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of mutual information between 3 output columns and gold_return_next on validation set
- **direction**: maximize

---

## 5. Training Settings

### Fitting Procedure

This is not a gradient-based training loop. The procedure is:

1. **HMM**: `GaussianHMM.fit(dxy_returns_train)` -- EM algorithm, converges in seconds
2. **PCA**: Rolling window PCA on currency returns -- deterministic, no fitting
3. **Volatility**: Rolling std + expanding z-score -- deterministic, no fitting

### Evaluation Metric for Optuna

For each trial (hyperparameter combination):
1. Fit HMM on train set returns
2. Generate all 3 features for train+val+test using the fitted HMM and the trial's window parameters
3. Compute mutual information (MI) between the 3 features and `gold_return_next` on the validation set
4. Optuna maximizes: `MI_sum = MI(regime, target) + MI(divergence, target) + MI(vol_z, target)`

### Loss Function

N/A -- no gradient-based training. HMM uses EM (maximum likelihood). PCA and volatility are deterministic.

### Optimizer

N/A -- no gradient-based optimization.

### Early Stopping

N/A -- HMM EM converges via `n_iter` and `tol`. No early stopping needed.

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No neural network. HMM, PCA, and rolling statistics are CPU-only. |
| Estimated execution time | 3-5 minutes | Data download (~30s) + 30 Optuna trials * ~5s each (~2.5min) + final output generation (~30s) |
| Estimated memory usage | <1 GB | ~2,500 rows * 7 columns. Tiny dataset. |
| Required pip packages | `hmmlearn` | Must `pip install hmmlearn` at the start of train.py. sklearn, pandas, numpy are pre-installed on Kaggle. |

---

## 7. Implementation Instructions

### builder_data Instructions

1. **Fetch DXY data**: Use existing `data/raw/dxy.csv` (already available)
2. **Fetch 6 currency pairs**: Download from Yahoo Finance using tickers listed in Section 2
   - Start date: 2014-12-01 (60-day buffer before base_features start 2015-01-30)
   - End date: 2025-02-13
3. **Direction normalization**: Negate returns for EURUSD=X and GBPUSD=X
4. **Save as**: `data/multi_country/dxy_currencies.csv`
   - Columns: Date, EURUSD_ret, USDJPY_ret, GBPUSD_ret, USDCAD_ret, USDSEK_ret, USDCHF_ret
   - All returns expressed as USD-strength direction (positive = USD strengthening)
5. **Quality checks**:
   - No gaps > 3 consecutive trading days for any pair
   - Missing data < 1% for each pair
   - Date range covers 2015-01-30 to 2025-02-12 (matching base_features)
6. **Also save raw currency levels**: `data/raw/dxy_currencies.csv` for datachecker verification
   - Columns: Date, EURUSD, USDJPY, GBPUSD, USDCAD, USDSEK, USDCHF (raw Close prices)

### builder_model Instructions

#### train.py Structure

```python
"""
Gold Prediction SubModel Training - DXY Attempt 1
Self-contained: Data fetch -> Preprocessing -> HMM + PCA + Vol -> Optuna HPO -> Save results
"""

# === 1. Libraries ===
import subprocess
subprocess.check_call(['pip', 'install', 'hmmlearn'])

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import optuna
import json
import os
from datetime import datetime

# === 2. Data Fetching ===
# Fetch DXY and 6 currency pairs directly from Yahoo Finance
# Apply direction normalization (negate EURUSD, GBPUSD returns)
# Compute log-returns
# Align dates via inner join

# === 3. Feature Generation Functions ===

def generate_regime_feature(dxy_returns, n_components, covariance_type, n_init):
    """Fit HMM and return P(high-variance state)"""
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=100,
        tol=1e-4,
        random_state=42,
        init_params='stmc',  # initialize all params
        n_init=n_init
    )
    X = dxy_returns.values.reshape(-1, 1)
    model.fit(X)
    probs = model.predict_proba(X)

    # Identify which state has higher variance
    variances = [model.covars_[i] for i in range(n_components)]
    # For 1D: covars_ shape depends on covariance_type
    # Extract scalar variance for each state
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

def generate_divergence_feature(currency_returns_df, pca_window):
    """Rolling PCA divergence: 1 - PC1_explained_variance_ratio"""
    n = len(currency_returns_df)
    divergence = np.full(n, np.nan)

    for i in range(pca_window - 1, n):
        window_data = currency_returns_df.iloc[i - pca_window + 1: i + 1].values
        if np.any(np.isnan(window_data)):
            continue
        pca = PCA(n_components=min(6, pca_window))
        pca.fit(window_data)
        divergence[i] = 1.0 - pca.explained_variance_ratio_[0]

    # Expanding MinMax normalization (no lookahead)
    result = np.full(n, np.nan)
    for i in range(pca_window - 1, n):
        past_vals = divergence[pca_window - 1: i + 1]
        past_valid = past_vals[~np.isnan(past_vals)]
        if len(past_valid) < 2:
            result[i] = 0.5  # default for insufficient history
        else:
            vmin, vmax = past_valid.min(), past_valid.max()
            if vmax - vmin < 1e-10:
                result[i] = 0.5
            else:
                result[i] = (divergence[i] - vmin) / (vmax - vmin)
    return result

def generate_volatility_feature(dxy_returns, vol_window):
    """Realized volatility z-score with expanding normalization"""
    vol = dxy_returns.rolling(vol_window).std()

    # Expanding z-score (no lookahead)
    expanding_mean = vol.expanding().mean()
    expanding_std = vol.expanding().std()
    z_score = (vol - expanding_mean) / expanding_std
    z_score = z_score.clip(-4, 4)  # clip extreme outliers
    return z_score.values

# === 4. Optuna Objective ===

def objective(trial, dxy_returns_train, dxy_returns_full,
              currency_returns_train, currency_returns_full,
              target_val, val_mask):
    """Maximize MI sum on validation set"""
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    covariance_type = trial.suggest_categorical('hmm_covariance_type', ['full', 'diag'])
    n_init = trial.suggest_categorical('hmm_n_init', [3, 5, 10])
    pca_window = trial.suggest_categorical('pca_window', [40, 60, 90])
    vol_window = trial.suggest_categorical('vol_window', [10, 20, 30])

    try:
        regime = generate_regime_feature(dxy_returns_full, n_components, covariance_type, n_init)
        divergence = generate_divergence_feature(currency_returns_full, pca_window)
        vol_z = generate_volatility_feature(dxy_returns_full, vol_window)

        # Extract validation period
        regime_val = regime[val_mask]
        div_val = divergence[val_mask]
        vol_val = vol_z[val_mask]

        # Compute MI (discretize continuous variables for MI calculation)
        def discretize(x, bins=20):
            valid = x[~np.isnan(x)]
            if len(valid) < bins:
                return None
            return pd.qcut(x, bins, labels=False, duplicates='drop')

        mi_sum = 0.0
        for feat in [regime_val, div_val, vol_val]:
            feat_disc = discretize(feat)
            target_disc = discretize(target_val)
            if feat_disc is not None and target_disc is not None:
                # Align non-NaN
                mask = ~(np.isnan(feat) | np.isnan(target_val))
                if mask.sum() > 50:
                    mi_sum += mutual_info_score(
                        discretize(feat[mask]),
                        discretize(target_val[mask])
                    )

        return mi_sum

    except Exception as e:
        return 0.0  # failed trial

# === 5. Main ===
# Data split: train/val/test = 70/15/15 (time-series order)
# Run Optuna
# Generate final output with best params
# Save results
```

#### Key Implementation Notes

1. **hmmlearn installation**: Must `pip install hmmlearn` at the top of train.py since it is not guaranteed to be pre-installed on all Kaggle environments
2. **Direction normalization**: EURUSD and GBPUSD returns MUST be negated. This is the most critical preprocessing step.
3. **No lookahead bias**: All normalizations (MinMax for divergence, expanding z-score for volatility) use only past data via expanding windows.
4. **HMM state labeling**: After fitting, sort states by variance. Do NOT assume state 0 = trending. Always identify the high-variance state empirically.
5. **NaN handling**: The first `max(pca_window, vol_window)` rows will have NaN. Forward-fill the output after generation to cover these warmup rows. The evaluator will align dates with base_features.
6. **Reproducibility**: Fix `random_state=42` for HMM. PCA is deterministic. Optuna uses `seed=42`.
7. **Output format**: CSV with columns [Date, dxy_regime_probability, dxy_cross_currency_div, dxy_volatility_z]. Aligned to trading dates matching base_features.

---

## 8. Risks and Alternatives

### Risk 1: HMM State Instability

- **Description**: HMM may produce different state assignments depending on initialization
- **Mitigation**: Use `n_init` parameter (3-10 random restarts). Optuna will find the best setting.
- **Detection**: Check that regime probabilities are not near-constant (std > 0.1)
- **Fallback**: If regime probability has std < 0.05, replace with a simpler momentum indicator (e.g., sign of 20-day DXY return)

### Risk 2: PCA Divergence Too Stable

- **Description**: If EUR dominates DXY (~57.6% weight), PC1 may always explain 85%+ variance, making divergence near-constant
- **Mitigation**: Rolling 60-day window allows time variation. MinMax normalization amplifies the available range.
- **Detection**: Check std of divergence feature > 0.05 after MinMax scaling
- **Fallback**: Replace PCA divergence with dispersion index: `std(6 currency returns)` normalized

### Risk 3: Volatility Z-Score Correlated with VIX

- **Description**: `dxy_volatility_z` might correlate with `vix_vix`, causing VIF issues
- **Mitigation**: FX volatility and equity volatility (VIX) have different dynamics. DXY vol captures USD-specific turbulence.
- **Expected VIF**: 3-5 (based on typical FX-equity vol correlation of 0.3-0.5)
- **Fallback**: If VIF > 10, residualize against VIX: output residuals from regressing dxy_vol_z on vix_vix

### Risk 4: Gate 3 Failure (Real_Rate Pattern)

- **Description**: Features pass Gate 2 (MI increase) but fail Gate 3 (ablation test)
- **Why DXY is different from real_rate**:
  - Daily frequency (no interpolation artifacts)
  - Structural features (regime, divergence) vs temporal features (smoothed series)
  - 3 columns (vs real_rate attempt 5: 7 columns)
- **Fallback for Attempt 2**: Drop volatility column (most likely to correlate with existing features), keep only regime + divergence (2 columns)

### Risk 5: hmmlearn Not Available on Kaggle

- **Description**: hmmlearn may fail to install on Kaggle
- **Mitigation**: `pip install hmmlearn` at script start. If fails, fall back to a simple threshold-based regime detection: `regime = sigmoid(dxy_return_20d / std_20d)` which approximates trend probability without HMM.
- **Likelihood**: Low -- hmmlearn has no exotic dependencies (just numpy, scipy, scikit-learn)

---

## 9. VIF Analysis (Pre-Design Estimate)

Expected VIF for each output column against base features:

| Output Column | Most Correlated Base Feature | Expected Correlation | Expected VIF |
|---------------|------------------------------|---------------------|-------------|
| dxy_regime_probability | dxy_dxy (level) | ~0.1 (regime vs level) | 1-2 |
| dxy_cross_currency_div | dxy_dxy (level) | ~0.05 (structure vs level) | 1-2 |
| dxy_volatility_z | vix_vix | ~0.3-0.4 (FX vol vs equity vol) | 2-4 |

All expected VIF values are well below the threshold of 10. The evaluator will compute actual VIF during Gate 2.

---

## 10. Autocorrelation Analysis

| Output Column | Expected Autocorrelation (lag 1) | Rationale |
|---------------|----------------------------------|-----------|
| dxy_regime_probability | 0.7-0.9 | Regimes persist for days-weeks, but HMM transitions provide variation |
| dxy_cross_currency_div | 0.8-0.95 | Rolling 60-day window creates smoothing, but not near-constant |
| dxy_volatility_z | 0.85-0.95 | Rolling 20-day vol is smooth but z-score normalization adds variation |

All expected values are below the 0.99 threshold that caused issues with real_rate attempt 1. The key difference: these are structurally varying features (regime switches, cross-sectional decomposition changes), not interpolated monthly data.

---

## 11. Design Rationale vs Alternatives

### Why HMM over Markov-Switching Autoregressive?

- Markov-Switching AR explicitly models momentum vs mean-reversion in the autoregressive coefficients, which is theoretically superior. However, implementation via `statsmodels.tsa.regime_switching.MarkovAutoregression` is more complex and slower. HMM via hmmlearn is simpler, faster, and sufficient for Attempt 1. If HMM regime feature proves useful, we can upgrade to Markov-Switching in Attempt 2.

### Why 2 States (Not 3)?

- With ~2,500 observations, 3-state HMM has 3 means + 3 variances + 6 transition probabilities = 12 parameters to estimate. This is feasible but adds complexity. We let Optuna test both 2 and 3 states. Starting assumption is 2 states because the core distinction (trending vs mean-reverting) maps naturally to 2 regimes.

### Why Rolling PCA (Not Static)?

- Static PCA on the full sample assumes the factor structure is constant. In reality, the EUR-dominance of DXY varies over time (e.g., EUR lost weight during the Greek debt crisis). Rolling 60-day PCA captures this time variation while being computationally cheap.

### Why 3 Columns (Not 2 or 4)?

- 2 columns: Would sacrifice either volatility or divergence. Both provide complementary information.
- 4 columns: Would require splitting one feature (e.g., separate PC1 loading and divergence). This adds noise dimensions for XGBoost without clear benefit. Real_rate lesson: more columns = more noise surface.
- 3 columns: Captures regime (temporal), divergence (cross-sectional), volatility (dispersion) -- three orthogonal aspects of USD dynamics.
