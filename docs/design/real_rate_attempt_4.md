# Submodel Design Document: real_rate (Attempt 4)

**Feature**: real_rate (Multi-Country Interest Rate Dynamics -- PCA)
**Attempt**: 4
**Phase**: Smoke Test
**Architect**: Claude Opus 4.6
**Date**: 2026-02-14

---

## 0. Fact-Check Results

### Data Availability
- `data/processed/real_rate_multi_country_features.csv` -> Confirmed. 269 rows, 25 columns, 2003-02 to 2025-06. Zero NaN.
- Rate change columns (7): `us_tips_change`, `germany_nominal_change`, `uk_nominal_change`, `canada_nominal_change`, `switzerland_nominal_change`, `norway_nominal_change`, `sweden_nominal_change` -> All present and verified.
- Gold calendar: `data/raw/gold.csv` -> Confirmed. 2523 rows in schema range (2015-01-30 to 2025-02-12).

### PCA Feasibility (Empirically Verified)
- PCA on 7 rate change features, fit on train (70%, 188 months):
  - PC0 explains 70.5% of variance (global rate trend)
  - PC1 explains 12.3% of variance (US TIPS vs Europe divergence)
  - PC2 explains 7.4% of variance
  - Cumulative: 2 PCs = 82.8%, 3 PCs = 90.2%
- PC loadings confirmed interpretable:
  - PC0: All positive (0.25-0.42), dominated by DE/UK/SE/CA -- global rate co-movement
  - PC1: US TIPS dominates (0.83), negative on DE/NO/SE -- US exceptionalism
- Gold return correlations (monthly):
  - PC0 vs gold return: -0.357 (global rates up -> gold down)
  - PC1 vs gold return: -0.319 (US rates up -> gold down)
  - PC2 vs gold return: +0.168 (moderate)

### Cubic Spline Interpolation (Empirically Verified)
- `scipy.interpolate.CubicSpline` -> Confirmed available.
- Spline produces 2523 daily values aligned exactly with gold trading calendar. Zero missing dates.
- No significant overshoot: daily range stays within or slightly narrower than monthly range.
  - PC0: monthly [-7.49, 9.22], daily [-6.98, 9.23] -- overshoot <0.1%
  - PC1: monthly [-2.74, 4.83], daily [-2.26, 3.49] -- no overshoot (narrower)
- Daily autocorrelation: 0.997 (PC0), 0.994 (PC1). Expected for smooth interpolation between monthly knots. This is fundamentally different from forward-fill autocorrelation (step function) because the spline values change continuously every day.
- Boundary conditions: 'not-a-knot', 'natural', 'clamped' all produce identical results for interior points. No sensitivity.

### Methodology Assessment
- PCA on standardized rate changes: APPROPRIATE. Linear dimensionality reduction is robust with 269 months (far more samples than dimensions).
- Cubic spline interpolation: APPROPRIATE. Produces C2-continuous output, addressing the step-function problem from Attempt 3.
- 10-feature option (adding aggregates): REJECTED. Adding yield_change_dispersion, mean_cpi_change, us_vs_global_spread reduces PC0 explained variance from 70.5% to 49.6% and destroys PC1 gold correlation (-0.32 -> -0.02). The aggregate features have different scales and variance structures that dilute the core rate-change signal.

### Feature Selection Comparison (Empirically Verified)

| Feature Set | PC0 Var | PC1 Var | PC0 Gold Corr | PC1 Gold Corr |
|-------------|---------|---------|---------------|---------------|
| 7 rate changes | 70.5% | 12.3% | -0.357 | -0.319 |
| 10 changes + aggregates | 49.6% | 11.9% | -0.362 | -0.017 |

**Decision**: Use 7 rate changes only. Cleaner signal, higher explained variance, both PCs informative.

---

## 1. Overview

### Purpose

Extract the dominant patterns of global interest rate co-movement from multi-country rate changes via PCA, then interpolate the monthly principal component scores to smooth daily values using cubic splines. This provides the meta-model with two continuous signals:
- **PC0**: Global rate trend (coordinated rate moves across 7 countries)
- **PC1**: US rate divergence from European/Scandinavian rates (US exceptionalism)

### Why PCA (Root Cause Analysis of Attempts 1-3)

| Attempt | Architecture | Root Cause of Failure |
|---------|-------------|----------------------|
| 1 | MLP Autoencoder (US-only) | Identity mapping. Overfit ratio 2.69. |
| 2 | GRU Autoencoder (US-only) | All trials pruned. GRU could not converge. |
| 3 | Transformer (multi-country) | Gate 1/2 PASS. Gate 3 FAIL: forward-fill step-function output degraded MAE in all 5 CV folds (+0.42%). |

**Key insight from Attempt 3**: The information exists (MI increase +23.8%), but the step-function output format prevents XGBoost from exploiting it. Three of five latent dimensions had zero MI -- only 2 carried signal.

**Why PCA solves this**:
1. **No overfitting risk**: PCA is deterministic. No training loop, no epochs, no gradient descent. 269 monthly samples are more than sufficient for 7-feature PCA.
2. **Cubic spline eliminates step functions**: Output changes smoothly every day, not abruptly at month boundaries. XGBoost can learn smooth decision boundaries.
3. **Only 2 output dimensions**: No noise from uninformative dimensions (Attempt 3 had 3 of 5 at zero MI).
4. **Interpretable**: PC0 = global rate trend, PC1 = US divergence. Clear economic meaning.

### Expected Effect

- Gate 1: N/A (deterministic, no training, no overfit ratio)
- Gate 2: MI increase 15-25%. Both PCs have strong gold return correlation (-0.36, -0.32).
- Gate 3: 65-70% probability of passing. Cubic spline smoothness should prevent the MAE degradation that killed Attempt 3. The same underlying information (multi-country rate patterns) is being delivered in a format that XGBoost can exploit.

---

## 2. Data Specification

### Primary Data Source

**File**: `data/processed/real_rate_multi_country_features.csv` (already exists from Attempt 3)

**No new data fetching required.** Reuse the multi-country data that was validated and committed during Attempt 3.

### Selected Features (7 rate changes)

| # | Column | Source | Interpretation |
|---|--------|--------|----------------|
| 1 | us_tips_change | DFII10 monthly diff | US real rate momentum |
| 2 | germany_nominal_change | IRLTLT01DEM156N diff | Germany yield momentum |
| 3 | uk_nominal_change | IRLTLT01GBM156N diff | UK yield momentum |
| 4 | canada_nominal_change | IRLTLT01CAM156N diff | Canada yield momentum |
| 5 | switzerland_nominal_change | IRLTLT01CHM156N diff | Switzerland yield momentum |
| 6 | norway_nominal_change | IRLTLT01NOM156N diff | Norway yield momentum |
| 7 | sweden_nominal_change | IRLTLT01SEM156N diff | Sweden yield momentum |

### Why Rate Changes Only (Not Levels or CPI)

1. **Rate changes are stationary**: Levels are non-stationary (trending), which violates PCA assumptions and inflates PC0 with trend information that has no predictive value for next-day returns.
2. **CPI features are lagged and slow-moving**: Month-to-month CPI changes have very low variance and would be dominated by nominal yield changes in PCA.
3. **Aggregate features (yield_dispersion, etc.)**: Empirically verified to reduce PC0 explained variance from 70.5% to 49.6% and destroy PC1 gold correlation. Excluded.
4. **Nominal vs TIPS**: Using us_tips_change alongside nominal changes provides the US real-rate signal that has the strongest gold correlation (-0.37 monthly), while nominal changes capture the cross-country dimension.

### Preprocessing Pipeline

```
1. Load data/processed/real_rate_multi_country_features.csv
2. Select 7 rate change columns
3. Verify no NaN (confirmed: zero NaN in source data)
4. Time-series split: train = first 70% (months 0-187, 188 months)
5. StandardScaler: fit on train, transform all 269 months
6. PCA(n_components=2): fit on train, transform all 269 months
7. Cubic spline: interpolate monthly PC scores to gold trading calendar
8. Clip to schema date range: 2015-01-30 to 2025-02-12
9. Save output
```

### Expected Sample Counts

- Monthly input: 269 months (2003-02 to 2025-06)
- Train split (70%): 188 months (2003-02 to 2018-09)
- Daily output: 2523 rows (exact match with schema)

### Calendar Data

**File**: `data/raw/gold.csv` (gold trading calendar)
- Use the index of this file as the daily trading dates
- Filter to schema range: 2015-01-30 to 2025-02-12

---

## 3. Model Architecture

### Architecture: PCA + Cubic Spline (No Neural Network)

This is NOT a neural network model. It is a deterministic statistical pipeline:

```
Input: [269 months, 7 features]  (monthly rate changes)
                  |
      StandardScaler (fit on train 70%)
                  |
      PCA(n_components=2) (fit on train 70%)
                  |
      PC Scores: [269 months, 2 PCs]
                  |
      CubicSpline (per PC, months -> days)
                  |
      Daily Scores: [2523 trading days, 2 PCs]
                  |
Output: real_rate_pc_0, real_rate_pc_1
```

### Pseudocode

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline

def generate_pca_submodel():
    # 1. Load existing multi-country data
    df = pd.read_csv('data/processed/real_rate_multi_country_features.csv',
                     index_col=0, parse_dates=True)

    # 2. Select 7 rate change features
    feature_cols = [
        'us_tips_change',
        'germany_nominal_change', 'uk_nominal_change',
        'canada_nominal_change', 'switzerland_nominal_change',
        'norway_nominal_change', 'sweden_nominal_change'
    ]
    X = df[feature_cols]
    assert X.isna().sum().sum() == 0, "Unexpected NaN in rate changes"

    # 3. Train/full split (70% for fitting)
    n_train = int(0.7 * len(X))
    X_train = X.iloc[:n_train]

    # 4. Standardize (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X)

    # 5. PCA (fit on train only)
    pca = PCA(n_components=2)
    pca.fit(X_train_scaled)
    pc_scores = pca.transform(X_full_scaled)  # [269, 2]

    # 6. Cubic spline interpolation to daily
    gold = pd.read_csv('data/raw/gold.csv', index_col=0, parse_dates=True)
    gold_dates = gold.index

    monthly_ts = X.index.astype(np.int64) // 10**9
    gold_in_range = gold_dates[
        (gold_dates >= X.index[0]) & (gold_dates <= X.index[-1])
    ]
    gold_ts = gold_in_range.astype(np.int64) // 10**9

    pc_daily = []
    for i in range(2):
        cs = CubicSpline(monthly_ts, pc_scores[:, i])
        pc_daily.append(cs(gold_ts))

    output_df = pd.DataFrame(
        np.column_stack(pc_daily),
        index=gold_in_range,
        columns=['real_rate_pc_0', 'real_rate_pc_1']
    )

    # 7. Clip to schema range
    output_df = output_df.loc['2015-01-30':'2025-02-12']

    # 8. Verify alignment
    assert len(output_df) == 2523, f"Expected 2523 rows, got {len(output_df)}"
    assert output_df.isna().sum().sum() == 0, "NaN in output"

    return output_df, {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'components': pca.components_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'feature_cols': feature_cols,
        'n_train_months': n_train,
    }
```

### Why This Architecture Addresses Previous Failures

1. **Attempt 1 (MLP, overfit 2.69)**: PCA cannot overfit -- it is a closed-form eigendecomposition. No training loop, no gradient descent.

2. **Attempt 2 (GRU, all trials pruned)**: PCA always produces a result. No convergence issues, no hyperparameter sensitivity.

3. **Attempt 3 (Transformer, MAE +0.42%)**: The step-function output from forward-fill was the root cause of MAE degradation. Cubic spline produces C2-continuous (twice continuously differentiable) daily output. Each day has a unique value, not a repeated monthly value. XGBoost can learn smooth gradients from this.

### Spline vs Forward-Fill: The Critical Difference

| Property | Forward-Fill (Attempt 3) | Cubic Spline (This) |
|----------|------------------------|---------------------|
| Daily values | Same value repeated ~22 days | Unique value each day |
| Continuity | Discontinuous at month boundaries | C2-continuous everywhere |
| Daily autocorrelation | 1.000 within months | 0.997 (smooth transition) |
| Change signal | Step function (jumps at month-start) | Gradual daily change |
| XGBoost compatibility | Poor (creates threshold artifacts) | Good (smooth feature) |

---

## 4. Hyperparameters

### Fixed Values (Design Decisions)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_components | 2 | PC0 explains 70.5%, PC1 explains 12.3%. Together 82.8%. PC2 adds only 7.4% and has weaker gold correlation (+0.17 vs -0.36, -0.32). Fewer dimensions = less noise. |
| Feature count | 7 | Rate changes only. Empirically verified to produce stronger PCs than 10-feature or 21-feature alternatives. |
| Train split | 70% (188 months) | Matches schema_freeze.json convention. PCA is robust with 188 samples for 7 features (27:1 ratio). |
| Interpolation | CubicSpline (not-a-knot) | C2-continuous. No overshoot verified. Boundary condition has no effect on interior points. |
| Standardization | StandardScaler | Required for PCA -- features have different scales (TIPS change in pp vs nominal changes in pp, but different magnitudes). |

### Optuna Search Space

**None.** PCA is deterministic. The only design choices are n_components and feature selection, both of which were determined by empirical analysis during design.

### Search Settings

- n_trials: N/A
- timeout: N/A

---

## 5. Learning Settings

**None.** PCA is a closed-form solution (eigendecomposition of the covariance matrix). No iterative optimization.

- Loss function: N/A
- Optimizer: N/A
- Early stopping: N/A

### Quality Validation (Run During Script Execution)

```python
def validate_output(output_df, pca, scaler, X):
    """Validate PCA submodel output quality."""

    # 1. Explained variance check
    assert pca.explained_variance_ratio_[0] > 0.5, \
        "PC0 should explain majority of variance"

    # 2. Output shape check
    assert output_df.shape == (2523, 2), \
        f"Expected (2523, 2), got {output_df.shape}"

    # 3. No NaN check
    assert output_df.isna().sum().sum() == 0, "NaN in output"

    # 4. No constant columns
    for col in output_df.columns:
        assert output_df[col].std() > 0.01, f"Near-constant: {col}"

    # 5. Spline overshoot check (daily range within 20% of monthly range)
    n_train = int(0.7 * len(X))
    X_scaled = scaler.transform(X)
    pc_monthly = pca.transform(X_scaled)
    for i in range(2):
        monthly_range = pc_monthly[:, i].max() - pc_monthly[:, i].min()
        daily_min = output_df.iloc[:, i].min()
        daily_max = output_df.iloc[:, i].max()
        monthly_min = pc_monthly[:, i].min()
        monthly_max = pc_monthly[:, i].max()
        overshoot = max(daily_max - monthly_max, monthly_min - daily_min)
        assert overshoot < 0.2 * monthly_range, \
            f"PC{i} spline overshoot too large: {overshoot:.4f}"

    return True
```

---

## 6. Kaggle Execution Settings

- **enable_gpu**: false
- **Estimated execution time**: 1-2 minutes
- **Estimated memory usage**: <0.5 GB
- **Required additional pip packages**: [] (pandas, numpy, scikit-learn, scipy all available in Kaggle)
- **enable_internet**: false (data is loaded from pre-saved CSV, no API calls needed)

### Time Budget Breakdown

| Component | Estimated Time |
|-----------|---------------|
| Load CSV data | <5 seconds |
| StandardScaler fit/transform | <1 second |
| PCA fit/transform | <1 second |
| CubicSpline interpolation (2 PCs) | <1 second |
| Validation checks | <1 second |
| Save output | <1 second |
| **Total** | **<10 seconds** |

### Rationale for CPU / No GPU

- PCA is a matrix eigendecomposition on a [188, 7] matrix -- trivial.
- CubicSpline interpolation of 2 curves with 269 knots to 2523 points -- trivial.
- No training loop, no batches, no GPU operations.

### Alternative: Local Execution

Given the trivial computation (~10 seconds), this script could be executed locally on Claude Code instead of submitted to Kaggle. However, for pipeline consistency with other submodels, it should still follow the Kaggle notebook structure. The orchestrator may choose to run this locally.

---

## 7. Implementation Instructions

### For builder_data

**No new data fetching required.**

The multi-country data file `data/processed/real_rate_multi_country_features.csv` already exists from Attempt 3 and contains all required features. builder_data should:

1. Verify the file exists and has the expected shape (269 rows, 25 columns)
2. Verify the 7 rate change columns are present and have zero NaN
3. Verify the gold trading calendar file `data/raw/gold.csv` exists
4. No data modification needed

### For builder_model

#### Script Location

`notebooks/real_rate_4/train.py` (following naming convention)

However, since this is a deterministic pipeline (no training), the script is much simpler than previous attempts.

#### Self-Contained train.py Structure

```
1. Load data/processed/real_rate_multi_country_features.csv
2. Select 7 rate change columns
3. Compute train split (70% = first 188 months)
4. StandardScaler: fit on train, transform all
5. PCA(n_components=2): fit on train, transform all
6. Load gold trading calendar from data/raw/gold.csv
7. CubicSpline interpolation: monthly PC scores -> daily
8. Clip to schema range (2015-01-30 to 2025-02-12)
9. Run validation checks
10. Save outputs:
    - submodel_output.csv: date, real_rate_pc_0, real_rate_pc_1
    - training_result.json: explained variance, loadings, metadata
```

#### Key Implementation Notes

1. **No FRED API key needed**: All data is already saved in the CSV. No internet access required.

2. **No Optuna**: Remove all HPO code. The script is deterministic.

3. **No PyTorch**: This script uses only scikit-learn and scipy. No neural network.

4. **Timestamp conversion for CubicSpline**: Convert DatetimeIndex to Unix timestamps (int64 // 10^9) for numerical interpolation. CubicSpline requires numerical x-values.

5. **Gold calendar alignment**: Use the index of `data/raw/gold.csv` as the target daily dates. Filter to dates within the PCA monthly date range, then clip to schema range.

6. **Output column naming**: `real_rate_pc_0`, `real_rate_pc_1` (not `real_rate_sem_N` from Attempt 3, to distinguish PCA from Transformer output).

7. **training_result.json format**:
```json
{
    "feature": "real_rate",
    "attempt": 4,
    "method": "PCA_CubicSpline",
    "timestamp": "...",
    "n_components": 2,
    "explained_variance_ratio": [0.7049, 0.1231],
    "cumulative_variance": 0.828,
    "feature_cols": ["us_tips_change", "germany_nominal_change", ...],
    "pc0_loadings": {"us_tips_change": 0.248, ...},
    "pc1_loadings": {"us_tips_change": 0.825, ...},
    "pc0_interpretation": "Global rate co-movement",
    "pc1_interpretation": "US rate divergence from Europe",
    "interpolation_method": "CubicSpline",
    "output_shape": [2523, 2],
    "output_columns": ["real_rate_pc_0", "real_rate_pc_1"],
    "output_stats": {
        "real_rate_pc_0": {"mean": ..., "std": ..., "min": ..., "max": ...},
        "real_rate_pc_1": {"mean": ..., "std": ..., "min": ..., "max": ...}
    },
    "daily_autocorrelation": {"real_rate_pc_0": ..., "real_rate_pc_1": ...},
    "gold_correlation_monthly": {"PC0": -0.357, "PC1": -0.319},
    "train_period": "2003-02 to 2018-09",
    "full_period": "2003-02 to 2025-06",
    "output_period": "2015-01-30 to 2025-02-12"
}
```

8. **Data file dependencies**: The script must include `data/processed/real_rate_multi_country_features.csv` and `data/raw/gold.csv` as Kaggle dataset sources, OR embed the data directly. Since enable_internet is false, the data must be available locally.

9. **Alternative: Run locally**: Given the trivial computation time (<10 seconds), the orchestrator should consider running this script locally instead of submitting to Kaggle. This avoids the overhead of Kaggle submission/polling/fetching and saves 5-10 minutes of wall-clock time.

#### Kaggle Dataset Handling

Since this script needs CSV files (not API calls), there are two options:

**Option A (Recommended): Local execution**
- Run `python notebooks/real_rate_4/train.py` directly on Claude Code
- No Kaggle submission needed
- Results available immediately

**Option B: Kaggle execution**
- Upload `data/processed/real_rate_multi_country_features.csv` and `data/raw/gold.csv` as a Kaggle dataset
- Reference in kernel-metadata.json as dataset_sources
- More complex setup for a trivial computation

---

## 8. Risks and Alternatives

### Risk 1: PCA Captures Only Linear Patterns

**Probability**: Low-Medium (25%). PCA is a linear method and cannot capture nonlinear regime changes.

**Assessment**: The Attempt 3 Transformer (nonlinear) achieved MI increase of +23.8%. If PCA achieves MI increase of +15-20%, the linear approximation is sufficient. The cubic spline smoothness is more important for Gate 3 than capturing nonlinear patterns.

**Mitigation**: Not needed. If MI is >5%, Gate 2 passes. The key improvement target is Gate 3 (MAE).

**Fallback**: If Gate 2 MI <5%, the linear signal is too weak. Declare no_further_improvement.

### Risk 2: Cubic Spline Introduces Artifacts

**Probability**: Very Low (5%). Empirically verified: no significant overshoot, daily range stays within monthly range. C2-continuity is well-behaved.

**Mitigation**: Validation check ensures daily range is within 20% of monthly range. If violated, fall back to linear interpolation.

**Fallback**: Replace CubicSpline with `np.interp` (piecewise linear). Less smooth but guaranteed monotone between knots.

### Risk 3: High Daily Autocorrelation Hurts XGBoost

**Probability**: Medium (30%). Daily autocorrelation of 0.997 is high, though it represents smooth variation rather than step-function repetition.

**Assessment**: This is fundamentally different from forward-fill autocorrelation (1.000 within months). Spline-interpolated values change every day, providing gradient information to XGBoost. The autocorrelation reflects that global rate regimes are persistent (which is the real-world truth), not an artifact of output construction.

**Mitigation**: XGBoost uses tree-based splits and is agnostic to autocorrelation in features. The concern is whether the day-to-day variation is informative or just noise. Since the variation follows the cubic spline shape (which is determined by monthly PC scores), it should be informative.

**Fallback**: If Gate 3 still fails due to autocorrelation, compute rolling change (5-day or 20-day) of the PC scores as additional features.

### Risk 4: Gate 3 Fails Again Despite Smooth Output

**Probability**: 30-35%. Even with smooth output, the real_rate submodel may not provide sufficient marginal improvement over the raw DFII10 feature.

**Assessment**: This is the terminal risk. If PCA + cubic spline cannot pass Gate 3, it means the multi-country rate change signal, while informative at the monthly level (MI >15%), does not translate to daily gold return prediction improvement. The base real_rate feature (DFII10 daily) already captures the strongest signal.

**Action if this occurs**: Declare `no_further_improvement` for real_rate. Proceed to dxy submodel. The base real_rate feature is sufficient.

### Risk 5: 2 PCs Insufficient Information

**Probability**: Low (15%). 2 PCs capture 82.8% of variance and both have strong gold correlations.

**Fallback**: If Gate 2 MI <5% with 2 PCs, retry with 3 PCs (90.2% variance). PC2 has weaker but nonzero gold correlation (+0.17).

### Alternative: No Alternative Architecture

This is Attempt 4 of 5. The evaluator's current_task.json explicitly states: "If PCA with interpolation cannot pass Gate 3, the marginal value of real_rate submodel is insufficient to justify further attempts." No alternative architecture is planned for Attempt 5 unless a specific, data-driven hypothesis emerges from the Gate 3 failure analysis.

---

## Appendix A: PCA Component Interpretation

### PC0: Global Rate Co-Movement (70.5% variance)

| Feature | Loading | Interpretation |
|---------|---------|----------------|
| germany_nominal_change | 0.425 | Largest contributor |
| sweden_nominal_change | 0.414 | |
| uk_nominal_change | 0.408 | |
| canada_nominal_change | 0.401 | |
| norway_nominal_change | 0.397 | |
| switzerland_nominal_change | 0.319 | Smaller (safe-haven dynamics) |
| us_tips_change | 0.248 | Smallest (TIPS != nominal) |

All loadings are positive: PC0 measures coordinated global rate movements. When PC0 is high, all countries' rates moved up together. This captures the "global monetary tightening/easing" signal.

Gold correlation: -0.357 (global rate tightening -> gold down). Consistent with economic theory.

### PC1: US Rate Divergence (12.3% variance)

| Feature | Loading | Interpretation |
|---------|---------|----------------|
| us_tips_change | 0.825 | Dominant contributor |
| switzerland_nominal_change | 0.374 | Same direction as US (safe haven pair) |
| germany_nominal_change | -0.161 | Opposite to US |
| norway_nominal_change | -0.280 | Opposite to US |
| sweden_nominal_change | -0.267 | Opposite to US |
| canada_nominal_change | -0.054 | Near zero (follows US closely) |
| uk_nominal_change | -0.031 | Near zero |

PC1 captures US rate moves that diverge from European/Scandinavian rates. High PC1 = US rates rising while European rates fall or lag. This measures "US monetary policy divergence."

Gold correlation: -0.319 (US divergent tightening -> gold down). Consistent: US-specific tightening strengthens USD and reduces gold demand.

## Appendix B: Schema Alignment Verification

| Property | Schema (schema_freeze.json) | This Design |
|----------|----------------------------|-------------|
| Date range start | 2015-01-30 | 2015-01-30 |
| Date range end | 2025-02-12 | 2025-02-12 |
| Row count | 2523 | 2523 (verified) |
| Missing dates | 0 | 0 (verified) |
| NaN values | N/A | 0 (verified) |
| Column dtypes | float64 | float64 |

## Appendix C: Comparison with Attempt 3 Output

| Property | Attempt 3 (Transformer + FF) | Attempt 4 (PCA + Spline) |
|----------|------------------------------|--------------------------|
| Output dimensions | 5 (3 uninformative) | 2 (both informative) |
| Daily values | ~22 identical per month | Unique each day |
| Continuity | Discontinuous (steps) | C2-continuous |
| Monthly autocorrelation | 0.937 | N/A (output is daily) |
| Daily autocorrelation | 0.997 (step function) | 0.997 (smooth curve) |
| Method | Neural (98K params, 188 samples) | Deterministic (no params) |
| Overfitting risk | Medium | Zero |
| MI with gold return | +23.8% | Expected 15-25% |
| Expected Gate 3 MAE | +0.42% (FAIL) | <= 0% (target) |
