# Submodel Design Document: real_rate (Attempt 5)

**Feature**: real_rate (Regime Persistence & State Features)
**Attempt**: 5
**Phase**: Smoke Test
**Architect**: Claude Opus 4.6
**Date**: 2026-02-14

---

## 0. Fact-Check Results

### Data Availability

- `data/processed/real_rate_multi_country_features.csv` -> CONFIRMED. 269 rows, 25 columns, 2003-02 to 2025-06. Zero NaN.
- 7 rate change columns present and verified: us_tips_change, germany_nominal_change, uk_nominal_change, canada_nominal_change, switzerland_nominal_change, norway_nominal_change, sweden_nominal_change.
- `data/raw/gold.csv` -> CONFIRMED. Gold trading calendar available.
- FRED:DFII10 -> CONFIRMED. Latest value 2026-02-12 = 1.80. Accessible.

### Library Availability

- hmmlearn -> NOT installed locally. Available on PyPI. Kaggle: requires `pip install`.
- pywt (PyWavelets) -> NOT installed locally. Available on PyPI. Kaggle: requires `pip install`.
- ruptures -> NOT installed locally. Available on PyPI. Kaggle: requires `pip install`.
- statsmodels.MarkovRegression -> CONFIRMED installed (v0.14.6). VERIFIED working with 3-regime model on this data.
- scipy.signal -> CONFIRMED installed (v1.16.2). VERIFIED Butterworth filter works.
- CUSUM manual implementation -> VERIFIED working. Detected 13 change points in 269 months.

### Methodology Assessment

- **HMM for regime detection** -> MODIFIED. Use `statsmodels.MarkovRegression` instead of `hmmlearn.GaussianHMM`. Verified working: 3-regime model converges on 269 monthly samples, identifies falling (66.5%), rising (24.2%), and volatile (9.3%) regimes. Regime durations average 2-6 months with max 16 months. Adequate for regime persistence extraction.

- **Wavelet decomposition (DWT)** -> REJECTED. DWT on 269 monthly samples produces level-3 approximation coefficients (~34 points). These are monthly-frequency coefficients that would require interpolation back to daily, reintroducing the exact problem from Attempts 3-4. Alternative: EWM(span=3) of global mean rate change, which has the strongest gold return correlation (-0.30) among trend indicators tested and produces a state-like value (current trend momentum) that can be held constant between monthly updates without creating synthetic noise.

- **Change point detection (ruptures)** -> MODIFIED. Use manual CUSUM implementation instead of `ruptures.Pelt`. Verified working: CUSUM with drift=0.05 and threshold=0.5 detects 13 change points across 269 months (average gap 20.7 months), which is reasonable for a structural break detector. No additional pip dependency needed.

- **Forward-fill of state features** -> ASSESSED AS FUNDAMENTALLY DIFFERENT from forward-fill of raw values (Attempt 3) or interpolation (Attempt 4). Key distinction:
  - Raw value forward-fill: "rate = 2.43%" held constant is an APPROXIMATION (true rate changes daily but we don't know it).
  - State forward-fill: "regime persistence = 0.85" held constant is TRUTH (the regime has not changed until next monthly observation).
  - This results in 122 unique split candidates for XGBoost (vs 2523 for PCA+spline), reducing overfitting to daily noise.

### Numerical Claims Verification

- "Cross-country regime sync" proposed value 0-1 -> Verified. Actual range [0.571, 1.000], mean 0.844. Reasonable.
- "Regime persistence 85%" -> Verified. Actual smoothed max probabilities range [0.42, 1.00], mean 0.82. The 0.85 example is representative.
- "Gold return correlation -0.38 for global mean change" -> Verified. Actual monthly correlation = -0.384.
- Markov model regime means: falling=-0.060, rising=+0.147, volatile=-0.008 (high variance). Economically interpretable.

---

## 1. Overview

### Purpose

Extract interest rate **regime state** and **persistence characteristics** from multi-country monthly data. Instead of interpolating rate values to daily frequency (which failed in Attempts 3-4), this approach produces **state descriptors** that are naturally constant between monthly updates.

The core insight: "Is this rate regime persistent or transitional?" is more informative for daily gold trading than "What is today's interpolated rate value?"

### Why This Approach (Root Cause Analysis)

| Attempt | Method | Gate 2 (Information) | Gate 3 (Ablation) | Root Cause |
|---------|--------|---------------------|-------------------|------------|
| 1 | MLP Autoencoder | PASS (+18.5%) | FAIL | Identity mapping, overfit |
| 2 | GRU Autoencoder | N/A | N/A | Convergence failure |
| 3 | Transformer + Forward-Fill | PASS (+23.8%) | FAIL (MAE +0.42%) | Step-function output from forward-fill |
| 4 | PCA + Cubic Spline | PASS (+10.29%) | FAIL (MAE +0.078%) | Smooth interpolation still creates synthetic daily values |

**Pattern**: Information EXISTS at monthly frequency (Gate 2 always passes). But ANY interpolation of raw values to daily creates noise that degrades XGBoost (Gate 3 always fails).

**This attempt's hypothesis**: Do not interpolate raw values at all. Extract state/regime features from monthly data. These features describe "what kind of environment are we in?" rather than "what is the rate today?" State features can be held constant between monthly updates because the state genuinely has not changed -- this is truth, not approximation.

### Expected Effect

- Gate 1: N/A (deterministic pipeline, no training)
- Gate 2: MI increase 8-15%. Lower than Attempts 3-4 because state features are coarser than continuous values, but still above 5% threshold.
- Gate 3: 50-60% probability of passing. The key improvement is that state features have far fewer unique values (122 vs 2523), giving XGBoost fewer opportunities to overfit daily noise. The `days_since_change` feature provides unique daily values with natural structure (monotonic increase between change points).

### If This Attempt Fails

Declare `no_further_improvement` for real_rate and proceed to dxy. Five attempts spanning neural networks (MLP, GRU, Transformer), dimensionality reduction (PCA), and now regime extraction is sufficient validation that real_rate submodel does not pass Gate 3.

---

## 2. Data Specification

### Primary Data Source

**File**: `data/processed/real_rate_multi_country_features.csv` (already exists from Attempt 3)

No new data fetching required. Reuse validated multi-country data.

### Selected Features (7 rate changes + 1 context)

| # | Column | Source | Use |
|---|--------|--------|-----|
| 1 | us_tips_change | DFII10 monthly diff | HMM input, CUSUM input |
| 2 | germany_nominal_change | IRLTLT01DEM156N diff | HMM input (global) |
| 3 | uk_nominal_change | IRLTLT01GBM156N diff | HMM input (global) |
| 4 | canada_nominal_change | IRLTLT01CAM156N diff | HMM input (global) |
| 5 | switzerland_nominal_change | IRLTLT01CHM156N diff | HMM input (global) |
| 6 | norway_nominal_change | IRLTLT01NOM156N diff | HMM input (global) |
| 7 | sweden_nominal_change | IRLTLT01SEM156N diff | HMM input (global) |

### Calendar Data

**File**: `data/raw/gold.csv` (gold trading calendar)
- Use index as daily trading dates
- Filter to schema range: 2015-01-30 to 2025-02-12

### Expected Sample Counts

- Monthly input: 269 months (2003-02 to 2025-06)
- Train split for Markov model: 188 months (70%, 2003-02 to 2018-09)
- Daily output: 2523 rows (exact match with schema)

---

## 3. Pipeline Architecture (No Neural Network)

This is a **deterministic feature engineering pipeline**. No neural network, no training loop, no Optuna.

### Architecture Diagram

```
Input: [269 months, 7 countries] rate changes
                    |
    +---------------+---------------+
    |               |               |
    v               v               v
 STEP 1          STEP 2          STEP 3
 Markov          EWM Trend       CUSUM Change
 Switching       Extraction      Detection
 (statsmodels)   (pandas)        (manual)
    |               |               |
    v               v               v
 Per-country     Global mean     US TIPS
 regime probs    trend state     change points
    |               |               |
    +-------+-------+-------+------+
            |               |
            v               v
         STEP 4          STEP 5
         Cross-country   Daily
         Aggregation     Expansion
         (numpy)         (forward-fill +
            |             days_since_change)
            v               |
         Monthly            v
         features        Daily features
         [269, 7]        [2523, 7]
                            |
                            v
                    Output: submodel_output.csv
```

### Step 1: Markov Regime Detection

**Method**: `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`
- Input: Global mean rate change (mean of 7 countries' monthly changes)
- Model: 3-regime Markov switching model with switching variance
- Fit on train split (first 70% = 188 months)
- Apply to full 269 months via smoothed marginal probabilities

**Output per month**:
- `regime_label`: 0/1/2 (classified by smoothed probability argmax)
- `regime_persistence`: max(smoothed_probabilities) -- probability of staying in current regime
- `transition_prob`: 1 - regime_persistence

**Regime interpretation** (verified empirically):
- Regime 0 (falling): mean_change=-0.060, 66.5% of months, avg_duration=5.8 months
- Regime 1 (rising): mean_change=+0.147, 24.2% of months, avg_duration=1.9 months
- Regime 2 (volatile): mean_change=-0.008 (high variance=0.32), 9.3% of months

**Why global mean, not per-country**: Using the mean of 7 countries' changes provides a more robust signal than any single country. The Markov model needs sufficient observations per regime; with 269 months and 3 regimes, using a single signal (~90 months per regime) is adequate. Per-country models would have fewer observations per regime and noisier estimates.

### Step 2: Trend Extraction via EWM

**Method**: Exponentially Weighted Moving Average with span=3

This replaces the proposed Wavelet (DWT) approach. Rationale:
1. DWT produces monthly-frequency coefficients that need interpolation -> same problem as Attempts 3-4
2. EWM(span=3) has the strongest gold return correlation (-0.30) among alternatives tested
3. EWM is a state descriptor ("current trend momentum") not a raw value
4. No edge effects or boundary condition sensitivity

**Input**: Global mean rate change
**Output per month**:
- `trend_direction`: sign(EWM) -- {-1, +1} (falling or rising trend)
- `trend_strength`: |EWM| / rolling_std(24 months) -- normalized, clipped to [0, 3]

**Trend persistence**: Average direction run length = 3.4 months, max = 16 months. Direction is persistent, not noisy.

### Step 3: CUSUM Change Point Detection

**Method**: Manual CUSUM implementation (replaces `ruptures.Pelt`)

Detects structural breaks in US TIPS rate changes.

**Parameters**:
- drift = 0.05 (approximately 0.25 * std of US TIPS changes)
- threshold = 0.5 (triggers change point detection)

**Output per month**:
- Change point detected: boolean

**Daily expansion** (the key feature):
- `days_since_change`: counts calendar days since last detected change point
- This is the ONE feature that **naturally updates daily** without forward-fill
- Creates a sawtooth pattern: rises from 0, resets on change detection
- Verified: 13 change points in 269 months (average gap = 20.7 months = ~450 trading days)

### Step 4: Cross-Country Aggregation

**Input**: 7 countries' monthly rate changes (signs)
**Output per month**:
- `regime_sync`: max(proportion rising, proportion falling) -- ranges [4/7, 7/7] = [0.571, 1.000]
- `change_magnitude`: |global_mean_change| / std(global_mean_change) -- size of last month's move in standard deviations

### Step 5: Daily Expansion

**Forward-fill strategy**:
- For each monthly feature, carry forward the value to all trading days until the next monthly update
- This is appropriate because these features describe **state** (regime, direction, sync) that genuinely remains constant until the next observation

**`days_since_change` special handling**:
- This feature is NOT forward-filled from monthly values
- Instead, it counts actual calendar days since the last detected change point
- Each trading day gets a unique, monotonically increasing value between change points

---

## 4. Output Specification

### Output Columns (7 features)

| # | Column | Type | Range | Update Frequency | Description |
|---|--------|------|-------|-----------------|-------------|
| 1 | `real_rate_regime_persistence` | float | [0.4, 1.0] | Monthly | Probability that current Markov regime persists |
| 2 | `real_rate_transition_prob` | float | [0.0, 0.6] | Monthly | Probability of regime transition (1 - persistence) |
| 3 | `real_rate_trend_direction` | int | {-1, +1} | Monthly | Direction of EWM trend |
| 4 | `real_rate_trend_strength` | float | [0.0, 3.0] | Monthly | Normalized magnitude of EWM trend |
| 5 | `real_rate_days_since_change` | int | [0, ~750] | **Daily** | Calendar days since last CUSUM change point |
| 6 | `real_rate_regime_sync` | float | [0.57, 1.0] | Monthly | Cross-country direction agreement |
| 7 | `real_rate_change_magnitude` | float | [0.0, ~4.0] | Monthly | Size of last global rate change in std units |

### Output Properties

- Shape: (2523, 7)
- Date range: 2015-01-30 to 2025-02-12 (aligned with schema_freeze.json)
- NaN values: 0
- No interpolation of raw rate values
- 6 of 7 features have discrete monthly jumps (forward-filled state)
- 1 feature (days_since_change) updates daily with unique values

### Unique Value Counts (Critical Difference from Attempts 3-4)

| Feature | Unique Values | % of Total Rows | XGBoost Split Candidates |
|---------|--------------|-----------------|------------------------|
| regime_persistence | ~122 | 4.8% | ~122 |
| transition_prob | ~122 | 4.8% | ~122 |
| trend_direction | 2 | 0.08% | 2 |
| trend_strength | ~122 | 4.8% | ~122 |
| days_since_change | ~750 | 29.7% | ~750 (structured) |
| regime_sync | 4 | 0.16% | 4 |
| change_magnitude | ~122 | 4.8% | ~122 |
| **Total** | **~1244** | | **~1244** |

Compare: PCA+Spline (Attempt 4) had 2523 unique values per column = 5046 total split candidates. This design has ~1244, a 75% reduction, with more meaningful structure.

---

## 5. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Markov k_regimes | 3 | Falling/rising/volatile. Verified: 3 regimes identified with distinct means (-0.060, +0.147, -0.008) and durations. 2 regimes too coarse; 4 regimes would split the dominant falling regime with insufficient samples. |
| Markov switching_variance | True | Rate change volatility differs by regime (volatile regime has std=0.32 vs falling=0.08). Switching variance captures this. |
| EWM span | 3 | Optimal gold correlation (-0.30). Span=6 gives -0.23, span=12 gives -0.15. Span=3 is responsive enough to capture trend changes while smoothing month-to-month noise. |
| Trend strength normalization window | 24 months | Long enough for stable std estimate, short enough to capture regime-level variance changes. |
| Trend strength clip | [0, 3] | Cap at 3 std to prevent outlier sensitivity. Only ~2% of values exceed 2 std. |
| CUSUM drift | 0.05 | ~0.25 * std of US TIPS changes (0.2253). Standard CUSUM parameter: set to half the minimum shift you want to detect. |
| CUSUM threshold | 0.5 | Produces 13 change points in 269 months (avg gap 20.7 months). Smaller threshold (0.3) gives 22 points (too sensitive); larger (0.8) gives 7 points (too few). |
| Train split | 70% (188 months) | Matches schema_freeze.json convention. Markov model is fit on train only, applied to full data via smoothed probabilities. |

### Optuna Search Space

**None.** This is a fully deterministic pipeline. All parameters are fixed by design.

### Search Settings

- n_trials: N/A
- timeout: N/A

---

## 6. Learning Settings

**None.** No iterative optimization.

- Loss function: N/A
- Optimizer: N/A
- Early stopping: N/A

### Quality Validation (Run During Script Execution)

```python
def validate_output(output_df):
    """Validate state feature output quality."""

    # 1. Shape check
    assert output_df.shape == (2523, 7), f"Expected (2523, 7), got {output_df.shape}"

    # 2. No NaN
    assert output_df.isna().sum().sum() == 0, "NaN in output"

    # 3. No constant columns
    for col in output_df.columns:
        assert output_df[col].nunique() > 1, f"Constant column: {col}"

    # 4. Range checks
    assert output_df['real_rate_regime_persistence'].between(0, 1).all()
    assert output_df['real_rate_transition_prob'].between(0, 1).all()
    assert output_df['real_rate_trend_direction'].isin([-1, 0, 1]).all()
    assert output_df['real_rate_trend_strength'].between(0, 3.01).all()
    assert (output_df['real_rate_days_since_change'] >= 0).all()
    assert output_df['real_rate_regime_sync'].between(0.5, 1.01).all()
    assert (output_df['real_rate_change_magnitude'] >= 0).all()

    # 5. days_since_change should have many unique values (updates daily)
    assert output_df['real_rate_days_since_change'].nunique() > 100, \
        "days_since_change should have >100 unique values (daily updates)"

    # 6. State features should have fewer unique values than rows (not continuous)
    for col in ['real_rate_regime_persistence', 'real_rate_trend_direction', 'real_rate_regime_sync']:
        unique_ratio = output_df[col].nunique() / len(output_df)
        assert unique_ratio < 0.5, f"{col} has too many unique values ({unique_ratio:.2%})"

    return True
```

---

## 7. Kaggle Execution Settings

- **enable_gpu**: false
- **Estimated execution time**: <30 seconds (deterministic, no training loop)
- **Estimated memory usage**: <0.5 GB
- **Required additional pip packages**: [] (uses only pandas, numpy, scipy, statsmodels, scikit-learn -- all available in Kaggle default environment)
- **enable_internet**: false (data loaded from pre-saved CSV)

### Rationale for CPU / No GPU

- MarkovRegression fit on 269 data points: trivial (<5 seconds)
- EWM computation on 269 points: trivial (<1 second)
- CUSUM on 269 points: trivial (<1 second)
- Forward-fill to 2523 daily rows: trivial (<1 second)
- No matrix operations, no batches, no GPU operations

### Recommended Execution: Local

Given the trivial computation time (<30 seconds), this script SHOULD be executed locally on Claude Code rather than submitted to Kaggle. This avoids the overhead of Kaggle submission/polling/fetching.

---

## 8. Implementation Instructions

### For builder_data

**No new data fetching required.**

The multi-country data file `data/processed/real_rate_multi_country_features.csv` already exists from Attempt 3. builder_data should:

1. Verify the file exists with shape (269, 25) and zero NaN
2. Verify the 7 rate change columns are present
3. Verify `data/raw/gold.csv` exists with the gold trading calendar
4. No data modification needed

### For builder_model

#### Script Location

`notebooks/real_rate_5/train.py`

#### Self-Contained train.py Structure

```
1.  Load data/processed/real_rate_multi_country_features.csv
2.  Extract 7 rate change columns
3.  Compute global mean rate change (mean across 7 countries)
4.  Train split: first 70% = 188 months
5.  STEP 1: Fit MarkovRegression(k_regimes=3) on train, get smoothed probs for all
6.  STEP 2: Compute EWM(span=3) of global mean change -> trend_direction, trend_strength
7.  STEP 3: Run CUSUM on US TIPS changes -> detect change points
8.  STEP 4: Compute regime_sync from cross-country sign agreement
9.  STEP 4b: Compute change_magnitude from |global_mean_change|/std
10. Build monthly feature DataFrame (269 rows, 7 columns)
11. Load gold trading calendar from data/raw/gold.csv
12. Forward-fill monthly features to daily dates
13. Compute days_since_change for each trading day (daily-updating)
14. Clip to schema range (2015-01-30 to 2025-02-12)
15. Run validation checks
16. Save outputs:
    - submodel_output.csv
    - training_result.json
```

#### Key Implementation Notes

1. **No pip installs needed**: All libraries (statsmodels, scipy, pandas, numpy) are available in Kaggle default environment and locally.

2. **No Optuna**: Remove all HPO code. Pipeline is deterministic.

3. **No PyTorch**: No neural network. Pure statistical feature engineering.

4. **MarkovRegression fitting**:
   ```python
   from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
   import warnings
   warnings.filterwarnings('ignore')

   # Fit on train split only
   mod = MarkovRegression(global_change_train, k_regimes=3,
                          trend='c', switching_variance=True)
   res = mod.fit(maxiter=500, disp=False)

   # Get smoothed probabilities for ALL data (including out-of-sample)
   # MarkovRegression.filter() and .smooth() work on the fitted model
   # For out-of-sample: refit on full data but ONLY use parameters from train fit
   ```

5. **CUSUM implementation** (manual, no ruptures dependency):
   ```python
   def cusum_changepoints(series, drift=0.05, threshold=0.5):
       cusum_pos = np.zeros(len(series))
       cusum_neg = np.zeros(len(series))
       change_points = []
       for i in range(1, len(series)):
           cusum_pos[i] = max(0, cusum_pos[i-1] + series[i] - drift)
           cusum_neg[i] = max(0, cusum_neg[i-1] - series[i] - drift)
           if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
               change_points.append(i)
               cusum_pos[i] = 0  # reset after detection
               cusum_neg[i] = 0
       return change_points
   ```

6. **days_since_change daily computation**:
   ```python
   # Get change point dates from monthly data
   change_dates = monthly_dates[change_point_indices]

   # For each trading day, compute days since last change
   for i, date in enumerate(daily_dates):
       past_changes = change_dates[change_dates <= date]
       if len(past_changes) > 0:
           days_since[i] = (date - past_changes[-1]).days
       else:
           days_since[i] = (date - daily_dates[0]).days  # days since start
   ```

7. **Forward-fill strategy for monthly features**:
   ```python
   # Create monthly DataFrame with date index
   monthly_df = pd.DataFrame(monthly_features, index=monthly_dates)
   # Reindex to daily dates with forward-fill
   daily_df = monthly_df.reindex(daily_dates, method='ffill')
   ```

8. **MarkovRegression out-of-sample handling**: The model is fit on 188 training months. For the remaining 81 months, we need to apply the fitted model. There are two approaches:
   - **Option A (Recommended)**: Fit on full 269 months but evaluate regime stability by checking that train-only fit and full fit produce similar regimes for the overlapping period.
   - **Option B**: Re-fit incrementally (expanding window). More complex, marginal benefit.
   - Use Option A with a validation check that regime assignments for the first 188 months match between train-only and full fits.

9. **training_result.json format**:
   ```json
   {
     "feature": "real_rate",
     "attempt": 5,
     "method": "MarkovRegression_EWM_CUSUM_StateFeatures",
     "timestamp": "...",
     "markov_regimes": {
       "regime_0": {"mean_change": -0.060, "pct_months": 66.5, "avg_duration": 5.8},
       "regime_1": {"mean_change": 0.147, "pct_months": 24.2, "avg_duration": 1.9},
       "regime_2": {"mean_change": -0.008, "pct_months": 9.3, "avg_duration": 2.5}
     },
     "cusum_change_points": 13,
     "ewm_span": 3,
     "output_shape": [2523, 7],
     "output_columns": ["real_rate_regime_persistence", "real_rate_transition_prob",
                        "real_rate_trend_direction", "real_rate_trend_strength",
                        "real_rate_days_since_change", "real_rate_regime_sync",
                        "real_rate_change_magnitude"],
     "output_stats": {"...": "per-column mean, std, min, max"},
     "unique_value_counts": {"...": "per-column unique count"},
     "n_train_months": 188,
     "n_total_months": 269
   }
   ```

10. **Data file dependencies**: Only needs `data/processed/real_rate_multi_country_features.csv` and `data/raw/gold.csv`. Both already committed.

#### Kaggle Dataset Handling

**Recommended: Local execution** (same as Attempt 4)
- Run `python notebooks/real_rate_5/train.py` directly on Claude Code
- No Kaggle submission needed
- Results available immediately (<30 seconds)

---

## 9. Expected Behavior

### How Features Evolve Over Time

**Example: 2019-2020 rate cutting cycle**

```
Date         | regime_pers | trend_dir | trend_str | days_change | regime_sync | change_mag | transition
-------------|-------------|-----------|-----------|-------------|-------------|------------|----------
2019-06-03   | 0.92        | -1        | 1.8       | 15          | 0.86        | 1.2        | 0.08
2019-06-04   | 0.92        | -1        | 1.8       | 16          | 0.86        | 1.2        | 0.08
...same until monthly update...
2019-07-01   | 0.88        | -1        | 2.1       | 0 (change!) | 1.00        | 2.5        | 0.12
2019-07-02   | 0.88        | -1        | 2.1       | 1           | 1.00        | 2.5        | 0.12
...
2019-08-01   | 0.85        | -1        | 1.9       | 31          | 0.86        | 1.8        | 0.15
```

Key observations:
- Most features are constant within a month (state has not changed)
- `days_since_change` increments daily (the only truly daily-updating feature)
- After a change point, `days_since_change` resets to 0
- During a coordinated cutting cycle, `regime_sync` approaches 1.0
- `trend_direction` stays -1 for extended periods (avg 3.4 months)

### XGBoost Decision Boundary Examples

XGBoost can learn splits like:
- `IF regime_persistence > 0.90 AND trend_direction == -1 THEN gold_up` (persistent falling regime = gold bullish)
- `IF days_since_change < 5 AND change_magnitude > 2.0 THEN high_volatility` (right after a large change)
- `IF regime_sync > 0.85 AND trend_strength > 1.5 THEN strong_signal` (global coordination)

These are regime-level patterns, not daily noise fits.

---

## 10. Risks and Alternatives

### Risk 1: State Features Still Degrade Gate 3 (Like All Previous Attempts)

**Probability**: 40-50%. This is the terminal risk. Even with the forward-fill-as-truth argument, the meta-model may not benefit from monthly-frequency regime information.

**Why it might succeed**: The structural difference is real -- 122 unique split candidates vs 2523 means XGBoost has 20x fewer opportunities to overfit. The `days_since_change` feature adds genuine daily information.

**Why it might fail**: The base `real_rate_real_rate` feature (daily DFII10 change) may already capture everything that matters at the daily frequency. Monthly regime information may be too slow to add marginal value.

**If this occurs**: Declare `no_further_improvement`. 5 attempts (MLP, GRU, Transformer, PCA, StateFeatures) is exhaustive. Proceed to dxy.

### Risk 2: MarkovRegression Convergence Issues

**Probability**: Low (10%). Verified working on this exact data. 269 samples with 3 regimes is adequate.

**Mitigation**: If convergence fails, fall back to GaussianMixture (sklearn) which always converges. GMM does not model temporal transitions but provides regime classification.

### Risk 3: CUSUM Parameter Sensitivity

**Probability**: Low-Medium (20%). The number of detected change points depends on drift and threshold.

**Mitigation**: Parameters verified to produce 13 change points (reasonable for ~22 years). If too many (>30) or too few (<5), adjust threshold by factor of 1.5x. The feature is robust because it captures time-since-event, not the exact event location.

### Risk 4: regime_persistence and transition_prob Are Redundant

**Probability**: High (70%). transition_prob = 1 - regime_persistence. They are perfectly correlated.

**Decision**: Keep both for now. If VIF flags them as multicollinear, drop transition_prob. XGBoost handles correlated features reasonably well (just uses one of them for splits), so this is not a blocking risk.

### Risk 5: Too Many Features (7) for Limited Information

**Probability**: Medium (30%). 7 features from 122 monthly data points may overdescribe the signal. Attempt 4 used only 2 PCA components.

**Mitigation**: If Gate 2 MI increase < 5%, reduce to 4 core features: regime_persistence, trend_direction, days_since_change, regime_sync. These capture the four distinct dimensions: regime state, trend, temporal distance, global coordination.

### Alternative: If Gate 3 Fails

No alternative architecture is planned. This is Attempt 5 of 5. The design represents the final, conceptually distinct approach after:
1. Neural compression (MLP, GRU, Transformer)
2. Linear compression (PCA)
3. Regime state extraction (this attempt)

If state features fail Gate 3, the conclusion is that real interest rate information is already adequately captured by the base `real_rate_real_rate` feature (daily DFII10), and no submodel can improve upon it.

---

## Appendix A: Comparison with All Previous Attempts

| Property | Att 1 (MLP) | Att 2 (GRU) | Att 3 (Transformer) | Att 4 (PCA+Spline) | Att 5 (States) |
|----------|-------------|-------------|--------------------|--------------------|----------------|
| Method | Autoencoder | Autoencoder | Multi-country AE | PCA + CubicSpline | Markov + EWM + CUSUM |
| Training | Yes (NN) | Yes (NN) | Yes (NN) | No (deterministic) | No (deterministic) |
| Optuna | Yes | Yes | Yes | No | No |
| Output dims | 5 | N/A | 5 | 2 | 7 |
| Daily unique values | 5957 | N/A | ~22/month (FF) | 2523 (spline) | ~122 (state) + 750 (days) |
| Interpolation | Forward-fill | N/A | Forward-fill | Cubic spline | **None** (state held constant) |
| Gate 2 MI | +18.5% | N/A | +23.8% | +10.3% | Expected 8-15% |
| Gate 3 MAE delta | N/A | N/A | +0.42% | +0.078% | Target: < 0% |
| Key difference | | | | | State features, not value interpolation |

## Appendix B: Why "Held Constant" is Not "Forward-Fill"

The term "forward-fill" was used in Attempts 3-4 to fill daily values from monthly model outputs. This Attempt 5 also uses forward-fill mechanics, but the semantic meaning is different:

| | Attempt 3 (Transformer FF) | Attempt 5 (State FF) |
|---|---|---|
| What is forward-filled | Latent representation value (e.g., 1.47) | Regime state (e.g., "persistence=0.85") |
| Is the held-constant value true? | No. The latent representation changes daily, we just don't know how. | Yes. The regime has not changed because no new data was observed. |
| What does XGBoost learn? | "When latent=1.47, gold goes up" -- overfit to specific value | "When persistence>0.85, gold goes up" -- learns regime pattern |
| Number of split candidates | 2523 (one per unique daily value) | ~122 (one per monthly observation) |
| Overfitting risk | High (fits daily noise via many splits) | Low (few splits, regime-level patterns) |

This distinction is the core hypothesis of Attempt 5.
