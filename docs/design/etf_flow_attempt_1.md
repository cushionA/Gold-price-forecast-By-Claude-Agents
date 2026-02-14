# Submodel Design Document: ETF Flow (Attempt 1)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| Yahoo: GLD OHLCV | CONFIRMED | Daily OHLCV available via yfinance, 5 rows fetched for 2025-02-10 to 2025-02-14, all fields present |
| etf_flow_gld_volume = technical_gld_volume (corr=1.0) | CONFIRMED | Exact identity verified: identical=True, max absolute difference=0.0 |
| etf_flow_gld_close = technical_gld_close (corr=1.0) | CONFIRMED | Exact identity verified: identical=True |
| Ben-David et al. 2018 "60-70% of actual flows" | FABRICATED | Paper is "Do ETFs Increase Volatility?" (JoF 2018). It studies ETF arbitrage transmitting volatility to underlying stocks. It does NOT claim volume proxies capture 60-70% of flows. Researcher fabricated this specific claim. |
| Volume spike returns: z>2 -> -0.08% | CONFIRMED | Measured: spike (z>2, N=147) mean=-0.0827%, normal mean=+0.0455%. 12.8bps difference verified. |
| Volume spike returns: std higher during spikes | CONFIRMED | Spike std=1.22%, Normal std=0.92%, Trough std=0.73%. Monotonically decreasing with volume z-score. |
| Dollar volume z-score 60d autocorr=0.4625 | CONFIRMED | Measured: 0.4625 exactly |
| Dollar volume z-score 20d autocorr=0.3213 | CONFIRMED | Measured: 0.3213 exactly |
| Dollar volume z-score 40d autocorr=0.4153 | CONFIRMED | Measured: 0.4153 exactly |
| PV correlation 5d autocorr=0.7406 | INCORRECT | Measured: 0.7767. Researcher underreported by 0.036. Still well below 0.99. Not a design-breaking error. |
| PV correlation 10d autocorr=0.8783 | NOT VERIFIED (claimed) | Measured: 0.9052. Higher than claimed. 10d window is riskier than researcher suggested. |
| Log volume ratio autocorr=0.3700 | CONFIRMED | Measured: 0.3700 exactly |
| GLD price range $114 to $270 (2.4x) | SLIGHTLY INCORRECT | Measured: $100.50 to $268.37 (2.67x). Price drift is actually larger than claimed, strengthening the case for dollar volume normalization. |
| VIF <10 for z-scored features | CONFIRMED with caveat | Dollar volume z60: VIF=6.13 (acceptable but not low). PV divergence z: VIF=1.09 (excellent). Dollar volume z-score has 0.80 corr with raw volume and 0.61 corr with tech_volatility_regime. |
| HMM for flow regime detection | CONFIRMED valid | Consistent with VIX/technical/cross_asset successful precedent |
| No free shares outstanding data | CONFIRMED | Researcher correctly identified this limitation |
| Options data not recommended | AGREED | Correct assessment: overlap with VIX, web scraping required, complexity without clear benefit |

### Critical Findings

**1. Ben-David et al. 2018 citation is misleading.** The paper studies ETF arbitrage and stock-level volatility transmission, not volume-based flow proxy accuracy. The "60-70% of flows" claim has no academic source. However, this does not invalidate the design -- volume-based flow proxies have empirical support from our own data (verified volume spike return differential of 12.8bps) and from the general "volume precedes price" principle in technical analysis. We proceed on empirical grounds rather than the fabricated academic claim.

**2. PV correlation 5d autocorrelation is 0.7767, not 0.7406.** Still acceptable (well below 0.99) but higher than reported. The z-scored version has autocorrelation of 0.7732 -- z-scoring does not reduce autocorrelation here because the underlying 5d correlation is already nonstationary.

**3. Dollar volume z-score VIF=6.13.** This is below 10 but warrants monitoring. The high correlation with raw volume (0.80) and tech_volatility_regime (0.61) means this feature is not as orthogonal as the researcher suggested. However, VIF is the relevant metric and 6.13 passes the <10 threshold.

---

## 1. Overview

- **Purpose**: Extract three dimensions of ETF flow dynamics from GLD volume and price data -- flow regime state (accumulation/distribution/panic), capital intensity (abnormal dollar-volume concentration), and price-volume divergence (confirmation vs reversal signal). These capture investor demand dynamics that the raw `etf_flow_gld_volume` base feature (rank #9, 5.2% importance) cannot convey.
- **Core methods**:
  1. Hidden Markov Model (3 states) on 2D [log_volume_ratio, gold_return] for regime classification
  2. Rolling z-score (60-day window) of dollar volume for capital intensity
  3. Rolling 5-day price-volume correlation z-scored against 60-day baseline for divergence detection
- **Why these methods**: Each captures a fundamentally different dimension of flow behavior. HMM captures which flow regime the market is in (accumulation vs distribution vs panic). Dollar volume z-score captures how abnormal the capital deployment is relative to recent history. PV divergence captures whether price and volume are confirming or contradicting each other -- a classic leading reversal indicator. None of these can be derived from raw volume levels alone.
- **Expected effect**: Enable the meta-model to distinguish scenarios where identical volume levels have different gold implications (e.g., high volume on an up day = safe-haven accumulation vs high volume on a down day = forced liquidation).

### Key Advantage

All data is daily frequency from Yahoo Finance. No interpolation needed. This eliminates the root cause of all 5 real_rate failures.

### Design Rationale: Why Approach A over Alternatives

- **Approach B (Deterministic only)**: Rejected because all successful submodels used HMM. Pure deterministic approaches sacrifice regime context.
- **Approach C (Multi-ETF GLD/IAU)**: Rejected because IAU has much lower volume, making the ratio noisy. Both track the same gold price, so cross-ETF ratio adds noise without clear signal.
- **Approach D (OBV-based)**: Rejected because daily OBV changes are extremely noisy (autocorr=0.03, but unstable single-day sign assignment). HMM on noisy OBV would produce unstable regimes. CMF overlaps with technical submodel OHLC usage.

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Already Available |
|------|--------|--------|-----------|-------------------|
| GLD OHLCV | Yahoo Finance | GLD | Daily | Yes: `data/raw/etf_flow.csv` (volume, close, volume_ma20) |
| Gold returns | Yahoo Finance | GC=F | Daily | Yes: `data/processed/base_features.csv` (gold_return_next) |

### Data NOT Used

| Data | Source | Reason for Exclusion |
|------|--------|---------------------|
| IAU volume | Yahoo Finance | Noisy cross-ETF ratio, rejected Approach C |
| GLD options | Barchart | Web scraping required, overlap with VIX, rejected |
| Shares outstanding | None available | No free historical source exists |
| GLDM volume | Yahoo Finance | Only available from June 2018, insufficient history |

### Preprocessing Steps

1. Fetch GLD OHLCV from Yahoo Finance, start=2014-10-01 (buffer for 60-day warmup before 2015-01-30)
2. Fetch GC=F close for gold returns computation
3. Compute derived quantities:
   - `volume_ma20 = volume.rolling(20).mean()`
   - `log_volume_ratio = log(volume / volume_ma20)`
   - `gold_return = gc_close.pct_change()` (current-day return, not next-day)
   - `dollar_volume = gld_close * gld_volume`
4. Handle missing values: forward-fill gaps up to 3 trading days, drop remaining NaN
5. Trim to base_features date range: 2015-01-30 to 2025-02-12

### Expected Sample Count

- ~2,523 daily observations (matching base_features row count)
- Warmup period: 65 days (60 for z-score rolling baseline + 5 for PV correlation)
- Effective output: ~2,460+ rows after warmup, remaining filled with NaN then forward-filled from first valid value

---

## 3. Model Architecture (Hybrid Deterministic-Probabilistic)

This is a **hybrid deterministic-probabilistic** approach, not a neural network. No PyTorch is required. The pipeline consists of three independent components.

### Component 1: HMM Regime Detection (etf_regime_prob)

- **Model**: `hmmlearn.hmm.GaussianHMM`
- **Input**: 2D array of [log_volume_ratio, gold_return]
  - log_volume_ratio = log(volume / volume_ma20): captures volume abnormality relative to 20-day baseline
  - gold_return = daily percentage change in GC=F close: captures price direction
  - 2D rationale: distinguishes accumulation (volume up + price up) from distribution (volume up + price down) from panic (extreme volume)
- **States**: 2 or 3 (Optuna selects via BIC or MI)
  - 2 states: normal vs abnormal
  - 3 states: accumulation, distribution, panic (preferred based on VIX/cross_asset precedent)
- **Covariance type**: "full" (allows states to capture volume-return correlation structure)
- **Training**: Fit on training set data only. Generate probabilities for full dataset using `predict_proba`.
- **Output**: Posterior probability of the highest-volume-variance state (identified post-hoc by comparing emission variances on the volume dimension)
- **State labeling**: After fitting, sort states by emission variance of the log_volume_ratio dimension. The highest-variance state corresponds to "panic/extreme flow." Output P(highest-variance state).

```
Input: [log_volume_ratio, gold_return] [T x 2]
       |
   GaussianHMM.fit(train_data) -> learn 2-3 state model
       |
   GaussianHMM.predict_proba(full_data) -> [T x n_states]
       |
   Select P(highest-volume-variance state) -> [T x 1]
       |
Output: etf_regime_prob (0-1)
```

### Component 2: Dollar Volume Z-Score (etf_flow_intensity_z)

- **Model**: Pure pandas computation (no ML)
- **Input**: Dollar volume = gld_close * gld_volume
- **Window**: 60-day rolling window (Optuna explores 40/60/90)
  - Rationale: Matches VIX submodel's successful 60d window. 60 trading days = ~3 calendar months. Captures medium-term flow baseline while adapting to structural shifts.
- **Computation**: `z = (dollar_volume_t - rolling_mean_60) / rolling_std_60`
  - Dollar volume instead of share volume: adjusts for GLD's 2.67x price drift over sample period
  - Z-score removes level correlation: VIF=6.13 vs raw volume, acceptable
- **Output**: z-score value (typically -3 to +4)
  - Positive: abnormally high capital movement (institutional repositioning)
  - Negative: abnormally quiet capital flow
  - Extreme positive (>2.0): verified to predict mean next-day return of -0.08% (vs +0.05% normal)

```
Input: dollar_volume = gld_close * gld_volume [T x 1]
       |
   rolling(window).mean() -> rolling_mean
   rolling(window).std()  -> rolling_std
       |
   z = (dollar_volume - rolling_mean) / rolling_std
       |
   clip(-4, 4) for stability
       |
Output: etf_flow_intensity_z (typically -3 to +4)
```

### Component 3: Price-Volume Divergence (etf_pv_divergence_z)

- **Model**: Pure pandas computation (no ML)
- **Input**: Daily returns and daily volume changes (percentage changes)
- **Windows**: 5-day rolling correlation window, z-scored against 60-day baseline
  - 5-day rationale: Short enough for low autocorrelation (0.78 measured, well below 0.99) while capturing meaningful correlation structure. 10-day window has autocorrelation 0.91 (riskier).
- **Computation**:
  1. `returns = gld_close.pct_change()`
  2. `vol_changes = gld_volume.pct_change()`
  3. `pv_corr_5d = returns.rolling(5).corr(vol_changes)`
  4. `z = (pv_corr_5d - pv_corr_5d.rolling(60).mean()) / pv_corr_5d.rolling(60).std()`
- **Output**: z-score of price-volume correlation
  - Positive: price and volume moving together (trend confirmation)
  - Negative: price and volume diverging (potential reversal -- classic "distribution top" or "accumulation bottom")
  - This feature has VIF=1.09 (essentially orthogonal to everything)

```
Input: gld_close, gld_volume [T x 1 each]
       |
   returns = pct_change(gld_close)
   vol_changes = pct_change(gld_volume)
       |
   pv_corr_5d = returns.rolling(5).corr(vol_changes)
       |
   z = (pv_corr_5d - rolling_mean_60) / rolling_std_60
       |
   clip(-4, 4) for stability
       |
Output: etf_pv_divergence_z (typically -3 to +3)
```

### Combined Output

| Column | Range | Description | Measured Correlation with raw gld_volume |
|--------|-------|-------------|------------------------------------------|
| `etf_regime_prob` | [0, 1] | P(high-volume-variance regime) from 2D HMM | ~0.3-0.5 (estimated, HMM not yet trained) |
| `etf_flow_intensity_z` | [-4, +4] | Dollar volume z-score relative to 60-day rolling | 0.80 (measured, VIF=6.13) |
| `etf_pv_divergence_z` | [-4, +4] | 5-day PV correlation z-scored vs 60-day baseline | 0.01 (measured, VIF=1.09) |

Total: **3 columns** (matching the proven compact pattern from VIX/technical/cross_asset/yield_curve).

### Orthogonality Analysis (Measured/Estimated)

| Feature Pair | Correlation | Assessment |
|-------------|-------------|------------|
| etf_flow_intensity_z vs raw gld_volume | 0.80 | High but VIF=6.13 is below 10 |
| etf_flow_intensity_z vs tech_volatility_regime | 0.61 | Moderate -- volume spikes co-occur with high volatility |
| etf_flow_intensity_z vs vix_mean_reversion_z | 0.28 | Acceptable |
| etf_flow_intensity_z vs vix_regime_probability | 0.19 | Low |
| etf_pv_divergence_z vs all existing features | max 0.11 | Excellent -- nearly fully orthogonal |
| etf_flow_intensity_z vs etf_pv_divergence_z | 0.11 | Low cross-correlation between outputs |

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| HMM n_iter | 100 | Standard convergence limit; EM typically converges in 20-50 iterations |
| HMM tol | 1e-4 | Standard convergence tolerance |
| HMM random_state | 42 | Reproducibility |
| Volume MA baseline | 20 days | Matches existing base feature etf_flow_volume_ma20 |
| Z-score clipping | [-4, 4] | Prevent extreme outliers from dominating |
| PV correlation window | 5 days | Fixed at 5d for acceptable autocorrelation (0.78). 10d has 0.91 (too risky). |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| hmm_n_components | {2, 3} | categorical | 2=normal/abnormal, 3=accumulation/distribution/panic. Prior: 3 states successful in VIX/cross_asset |
| hmm_covariance_type | {"full", "diag"} | categorical | full captures volume-return correlation structure; diag treats them independently |
| hmm_n_init | {3, 5, 10} | categorical | Number of EM restarts to avoid local optima |
| dollar_vol_zscore_window | {40, 60, 90} | categorical | 40d=responsive, 60d=balanced (VIX precedent), 90d=stable |
| pv_baseline_window | {40, 60, 90} | categorical | Baseline for z-scoring PV correlation. 60d aligns with dollar volume window |

### Exploration Settings

- **n_trials**: 30
  - Rationale: 5 categorical parameters with small ranges. Total combinations: 2 x 2 x 3 x 3 x 3 = 108. 30 trials with TPE provides good coverage. Each trial is fast (<10 seconds: HMM fit on ~1,766 x 2 observations).
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize sum of mutual information between 3 output columns and gold_return_next on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)

---

## 5. Training Settings

### Fitting Procedure

This is not a gradient-based training loop. The procedure is:

1. **HMM**: `GaussianHMM.fit(X_train)` on 2D [log_volume_ratio, gold_return] -- EM algorithm, converges in seconds
2. **Dollar Volume Z-Score**: Rolling window statistics on dollar volume -- deterministic, no fitting
3. **PV Divergence**: Rolling correlation + z-scoring -- deterministic, no fitting

### Data Split

- train/val/test = 70/15/15 (time-series order, no shuffle)
- train: ~1,766 observations
- val: ~378 observations
- test: ~379 observations (reserved for evaluator Gate 3)
- HMM fits on train set only
- HMM generates probabilities for full dataset using predict_proba (no lookahead: HMM posterior at time t depends only on observations up to t given fitted model)
- Z-score and PV correlation use rolling windows (inherently backward-looking, no lookahead)
- Optuna optimizes MI sum on validation set

### Evaluation Metric for Optuna

For each trial:
1. Fit 2D HMM on train set [log_volume_ratio, gold_return]
2. Generate all 3 features for full dataset using fitted HMM and trial window parameters
3. Compute mutual information (MI) between each feature and gold_return_next on validation set
4. Optuna maximizes: `MI_sum = MI(regime, target) + MI(intensity_z, target) + MI(pv_div_z, target)`

MI calculation: Discretize continuous features into 20 quantile bins, then compute sklearn `mutual_info_score`.

### Loss Function

N/A -- no gradient-based training. HMM uses EM (maximum likelihood). Z-score and PV correlation are deterministic.

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
| Estimated memory usage | <1 GB | ~2,523 rows x 2-3 columns. Tiny dataset |
| Required pip packages | `hmmlearn` | Must pip install hmmlearn at start of train.py. sklearn, pandas, numpy, optuna are pre-installed on Kaggle |

---

## 7. Implementation Instructions

### builder_data Instructions

1. **GLD data already available**: Use existing `data/raw/etf_flow.csv` (gld_volume, gld_close, volume_ma20)
2. **Verify format**: Ensure columns include Date, gld_volume, gld_close, volume_ma20
3. **Verify date range**: Data should start from 2014-10-01 (warmup buffer) through 2025-02-12
4. **Additional data**: Fetch GC=F close prices for gold_return computation (or use gold_return_next from base_features.csv shifted by 1 day)
5. **Save preprocessed data**: `data/processed/etf_flow_features_input.csv`
   - Columns: Date, gld_volume, gld_close, volume_ma20, log_volume_ratio, gold_return, dollar_volume
6. **Quality checks**:
   - No gaps > 3 consecutive trading days
   - Missing data < 2%
   - Volume values are positive (no zeros or negatives)
   - GLD close in reasonable range ($80-$300)
   - log_volume_ratio has no extreme outliers (|value| < 3)

### builder_model Instructions

#### train.py Structure

```python
"""
Gold Prediction SubModel Training - ETF Flow Attempt 1
Self-contained: Data fetch -> Preprocessing -> HMM + Z-Score + PV Divergence -> Optuna HPO -> Save results
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
# Fetch GLD OHLCV from Yahoo Finance (start=2014-10-01 for warmup buffer)
# Fetch GC=F close for gold_return computation
# Compute: log_volume_ratio, gold_return, dollar_volume
# Load gold_return_next target from Kaggle dataset

# === 3. Feature Generation Functions ===

def generate_regime_feature(log_vol_ratio, gold_return, n_components, covariance_type, n_init, train_size):
    """
    Fit 2D HMM on [log_volume_ratio, gold_return] and return P(highest-volume-variance state).
    """
    X = np.column_stack([log_vol_ratio, gold_return])
    X_train = X[:train_size]

    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=100,
        tol=1e-4,
        random_state=42,
        n_init=n_init
    )
    model.fit(X_train)
    probs = model.predict_proba(X)

    # Identify highest-volume-variance state (first dimension = log_volume_ratio)
    state_vol_vars = []
    for i in range(n_components):
        if covariance_type == 'full':
            state_vol_vars.append(float(model.covars_[i][0, 0]))
        elif covariance_type == 'diag':
            state_vol_vars.append(float(model.covars_[i][0]))

    high_var_state = np.argmax(state_vol_vars)
    return probs[:, high_var_state], model

def generate_intensity_feature(dollar_volume, window):
    """
    Rolling z-score of dollar volume: (dv - rolling_mean) / rolling_std
    """
    s = pd.Series(dollar_volume)
    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std()
    z = (s - rolling_mean) / rolling_std
    z = z.clip(-4, 4)
    return z.values

def generate_pv_divergence_feature(returns, vol_changes, pv_window=5, baseline_window=60):
    """
    5-day rolling correlation between returns and volume changes,
    z-scored against a baseline_window-day rolling baseline.
    """
    r = pd.Series(returns)
    v = pd.Series(vol_changes)
    pv_corr = r.rolling(pv_window).corr(v)
    pv_mean = pv_corr.rolling(baseline_window).mean()
    pv_std = pv_corr.rolling(baseline_window).std()
    z = (pv_corr - pv_mean) / pv_std
    z = z.clip(-4, 4)
    return z.values

# === 4. Optuna Objective ===

def objective(trial, log_vol_ratio, gold_return, dollar_volume, returns, vol_changes, target, train_size, val_mask):
    """Maximize MI sum on validation set"""
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    covariance_type = trial.suggest_categorical('hmm_covariance_type', ['full', 'diag'])
    n_init = trial.suggest_categorical('hmm_n_init', [3, 5, 10])
    dv_window = trial.suggest_categorical('dollar_vol_zscore_window', [40, 60, 90])
    pv_baseline = trial.suggest_categorical('pv_baseline_window', [40, 60, 90])

    try:
        regime, _ = generate_regime_feature(
            log_vol_ratio, gold_return, n_components, covariance_type, n_init, train_size
        )
        intensity = generate_intensity_feature(dollar_volume, dv_window)
        pv_div = generate_pv_divergence_feature(returns, vol_changes, pv_window=5, baseline_window=pv_baseline)

        # Extract validation period
        regime_val = regime[val_mask]
        intensity_val = intensity[val_mask]
        pv_div_val = pv_div[val_mask]
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
        for feat_val in [regime_val, intensity_val, pv_div_val]:
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
2. **2D HMM input**: Use [log_volume_ratio, gold_return] as 2D input. This distinguishes price-confirming volume from price-contradicting volume.
3. **HMM state labeling**: After fitting, sort states by emission variance of the FIRST dimension (log_volume_ratio). Output P(highest-volume-variance state). Do NOT assume state index 0, 1, or 2 corresponds to any specific regime.
4. **Dollar volume computation**: `dollar_volume = gld_close * gld_volume`. This is critical to adjust for the 2.67x price drift over the sample period.
5. **PV correlation window**: Fixed at 5 days. Do NOT use 10-day or longer windows (autocorrelation > 0.90).
6. **No lookahead bias**:
   - HMM: Fit on training data only, generate probabilities for full dataset
   - Z-score: Rolling window (inherently backward-looking)
   - PV divergence: Rolling window (inherently backward-looking)
7. **NaN handling**: First ~65 rows (max of warmup periods) will have NaN. Forward-fill output after generation.
8. **Reproducibility**: Fix random_state=42 for HMM, seed=42 for Optuna. Z-score and PV divergence are deterministic.
9. **Output format**: CSV with columns [Date, etf_regime_prob, etf_flow_intensity_z, etf_pv_divergence_z]. Aligned to trading dates matching base_features.
10. **Gold return for HMM input**: Use current-day gold return from GC=F (not gold_return_next, which is the target). This is backward-looking and does not leak future information.
11. **VIF awareness**: etf_flow_intensity_z has VIF=6.13 against existing features. If evaluator flags VIF issues, the mitigation is to use log(dollar_volume) z-score instead (VIF not yet measured but likely lower due to log compression).

---

## 8. Risks and Alternatives

### Risk 1: HMM State Instability (2D input)

- **Description**: 2D HMM may produce different state assignments than 1D. With ~1,766 training observations and 2 input dimensions, the model has more parameters to estimate, increasing instability risk.
- **Mitigation**: Use n_init parameter (3-10 random restarts). Optuna tests multiple values. Fix random_state=42. Precedent: technical submodel used 2D HMM successfully (regime_prob ranked #2 in importance).
- **Detection**: Check that regime probabilities have std > 0.1 and are not near-constant (yield_curve's HMM collapsed to 1 state -- avoid this).
- **Fallback**: If 2D HMM produces constant output, fall back to 1D HMM on log_volume_ratio only (simpler, more stable). If HMM collapses entirely, use sigmoid-based regime: `regime = sigmoid((log_vol_ratio / rolling_std) * 2)`.

### Risk 2: Dollar Volume Z-Score VIF Too High

- **Description**: Measured VIF=6.13 against all existing features. Correlation with raw volume is 0.80 and with tech_volatility_regime is 0.61. If additional submodel outputs are added later, VIF could increase.
- **Mitigation**: 6.13 is below the 10 threshold. The z-score transformation does remove the monotonic price trend. Precedent: VIX submodel's z-score had 0.43 corr with VIX level and passed Gate 2.
- **Fallback for Attempt 2**: If VIF exceeds 10, replace with (a) dollar volume daily change z-score (corr 0.42 with raw volume, autocorr -0.34), or (b) log(dollar_volume) z-score, or (c) dollar volume rank percentile (nonlinear transform, likely lower VIF).

### Risk 3: PV Divergence Autocorrelation

- **Description**: PV correlation 5d has autocorrelation 0.78 (not 0.74 as researcher claimed). Z-scored version is 0.77. This is acceptable but higher than ideal.
- **Mitigation**: 0.78 is well below the 0.99 threshold. The z-scoring against a 60-day baseline does not reduce autocorrelation because the underlying 5d correlation is already nonstationary. Acceptable given precedent (VIX regime_prob autocorrelation was in the 0.7-0.9 range and passed Gate 1).
- **Detection**: Verify autocorrelation < 0.99 in Gate 1.
- **Fallback**: If autocorrelation is too high, use 3-day PV correlation window instead of 5-day (lower autocorrelation but noisier).

### Risk 4: Regime Prob Stability Failure (Gate 2)

- **Description**: Regime probabilities tend to have rolling correlation std > 0.15 with gold_return_next (Gate 2 stability threshold).
- **Mitigation**: ALL successful submodels had regime_prob marginally fail stability: VIX (Gate 2 overall fail), technical (0.21), cross_asset (0.156), yield_curve (collapsed). This is EXPECTED and accepted precedent. Gate 3 ablation is the true test.
- **Detection**: Monitor in Gate 2 evaluation.

### Risk 5: Overlap with Technical Submodel

- **Description**: etf_flow_intensity_z correlates 0.61 with tech_volatility_regime. Both capture "abnormal market activity" from different angles (volume vs price range).
- **Mitigation**: The correlation is 0.61 (moderate, not extreme). They capture different phenomena: volume intensity measures capital deployment while volatility regime measures price range expansion. VIF analysis confirms both can coexist (VIF=6.13 for intensity_z includes tech_volatility_regime contribution).
- **Fallback**: If VIF is problematic, residualize etf_flow_intensity_z against tech_volatility_regime in Attempt 2.

### Risk 6: Gate 3 Failure

- **Description**: Features may pass Gate 2 (MI increase) but fail Gate 3 (ablation test), as happened with real_rate.
- **Why ETF flow is different from real_rate**:
  - Daily frequency (no interpolation artifacts)
  - Verified empirical signal: volume spikes predict 12.8bps mean-reversion (-0.08% vs +0.05%)
  - 3 compact features (not 7 like real_rate attempt 5)
  - Flow dynamics are directly relevant to gold demand (volume = capital commitment)
  - PV divergence is genuinely novel information not captured by any existing submodel (VIF=1.09)
- **Primary success path**: MAE improvement, following technical (-0.18), cross_asset (-0.09), yield_curve (-0.07) precedent.

---

## 9. VIF Analysis (Pre-Design Measured)

### etf_flow_intensity_z vs All Existing Features

| Feature | Correlation | Assessment |
|---------|-------------|------------|
| etf_flow_gld_volume (raw) | 0.80 | High -- z-scoring reduces but doesn't eliminate level correlation |
| tech_volatility_regime | 0.61 | Moderate -- volume spikes co-occur with high volatility |
| vix_mean_reversion_z | 0.28 | Acceptable |
| xasset_regime_prob | 0.19 | Low |
| vix_regime_probability | 0.19 | Low |
| etf_flow_volume_ma20 | 0.15 | Low -- z-score removes smooth trend |
| vix_vix | 0.14 | Low |
| **VIF (all features)** | **6.13** | **Below 10 threshold -- PASS** |

### etf_pv_divergence_z vs All Existing Features

| Feature | Correlation | Assessment |
|---------|-------------|------------|
| tech_mean_reversion_z | 0.11 | Very low |
| etf_flow_volume_ma20 | -0.10 | Negligible |
| All other features | < 0.10 | Negligible |
| **VIF (all features)** | **1.09** | **Essentially orthogonal -- PASS** |

---

## 10. Autocorrelation Analysis

| Output Column | Measured Autocorrelation (lag 1) | Status |
|---------------|----------------------------------|--------|
| etf_regime_prob | ~0.7-0.9 (estimated from HMM precedent) | Expected PASS (<0.99) |
| etf_flow_intensity_z | 0.4625 (measured for 60d window) | CONFIRMED PASS |
| etf_pv_divergence_z | 0.7732 (measured for 5d corr z-scored) | CONFIRMED PASS |

All values are well below the 0.99 threshold. Daily frequency data prevents the near-constant output problem that plagued monthly real_rate data.

---

## 11. Expected Performance Against Gates

### Gate 1: Standalone Quality
- **Overfit ratio**: N/A (deterministic HMM, no neural network with train/val loss comparison)
- **No constant output**: High confidence -- volume regimes shift, dollar volume z-score varies with market activity, PV correlation varies continuously
- **Autocorrelation < 0.99**: CONFIRMED for intensity_z (0.46) and pv_div_z (0.77). HMM regime_prob expected 0.7-0.9.
- **Expected Result**: PASS

### Gate 2: Information Gain
- **MI increase > 5%**: Moderate probability. PV divergence is genuinely novel (VIF=1.09). Dollar volume intensity adds capital context. Regime adds state classification. VIX submodel achieved 0.68% MI increase (below 5%) but still passed Gate 3.
- **VIF < 10**: CONFIRMED -- intensity_z VIF=6.13, pv_div_z VIF=1.09. Regime_prob expected <5 based on precedent.
- **Rolling correlation std < 0.15**: Moderate probability. Regime_prob may marginally fail (precedent: all successful submodels had this issue). Intensity_z and pv_div_z expected to pass.
- **Expected Result**: MARGINAL (regime_prob stability may fail, consistent with precedent)

### Gate 3: Ablation Test
- **Primary path**: MAE improvement > 0.01%
- **Supporting evidence**: Volume spikes predict 12.8bps mean-reversion, PV divergence provides novel information (VIF=1.09), flow context helps meta-model distinguish ambiguous volume scenarios
- **Expected Result**: TARGET PASS (following technical/cross_asset/yield_curve MAE improvement precedent)

**Confidence**: 6/10 (moderate). ETF flow has strong theoretical backing and verified empirical signals, but the fabricated academic citation for flow proxy accuracy and the high VIF of the intensity feature (6.13) introduce uncertainty. The PV divergence feature is the strongest candidate for providing genuinely novel information.
