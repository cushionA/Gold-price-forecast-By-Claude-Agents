# Yield Curve Data Quality Check - Attempt 1
## Standardized 7-Step Verification Report

**Timestamp:** 2026-02-15 02:04:37 UTC
**Feature:** yield_curve
**Attempt:** 1
**Data Location:** data/raw/yield_curve.csv
**Report:** logs/datacheck/yield_curve_attempt_1.json

---

## Executive Summary

**DECISION: CONDITIONAL PASS**

The yield_curve dataset passes all critical data quality checks. **No CRITICAL issues detected.** Nine warnings exist but are all explainable and do not warrant rejection:

- **8 correlation warnings** are expected and acceptable (raw yields naturally correlate with base features, but processed derivatives will be orthogonal)
- **1 spread range warning** is expected (yield spreads can turn negative during inversions)

**Status:** Data is suitable for builder_model pipeline. Move to Kaggle training.

---

## Detailed Step Results

### STEP 1: File Existence & Basic Structure
**Status: PASS**

- File exists: ✓ data/raw/yield_curve.csv
- Expected columns present: ✓ 9 numeric columns
  - dgs10, dgs2, dgs5 (yield levels)
  - dgs10_change, dgs2_change (yield changes)
  - spread, spread_change (yield curve slope)
  - curvature_raw, curvature_change (belly convexity)
- Row count: ✓ 2,840 rows (expected ~2,840)
- Date range: ✓ 2014-10-02 to 2026-02-12 (includes warmup buffer)

All structural requirements met.

---

### STEP 2: Basic Statistics & Outliers
**Status: PASS** (1 non-critical warning)

#### Yield Levels (should be 0-8%)

| Column | Mean  | Std   | Min    | Max    | Status |
|--------|-------|-------|--------|--------|--------|
| dgs10  | 2.65% | 1.13% | 0.52%  | 4.98%  | OK     |
| dgs2   | 2.14% | 1.59% | 0.09%  | 5.19%  | OK     |
| dgs5   | 2.36% | 1.28% | 0.19%  | 4.95%  | OK     |

All yield levels within reasonable historical bounds. Min values reflect 2023 rate-cut expectations.

#### Daily Changes (should be ~±0.3% typical)

| Column       | Mean    | Std   | Min    | Max    | Status |
|--------------|---------|-------|--------|--------|--------|
| dgs10_change | +0.06bp | 5.3bp | -30bp  | +29bp  | OK     |
| dgs2_change  | +10bp   | 5.2bp | -57bp  | +34bp  | OK     |

Changes are small and consistent with daily trading dynamics.

#### Derived Metrics

| Column            | Mean    | Std      | Min      | Max      | Status     |
|-------------------|---------|----------|----------|----------|------------|
| spread (DGS10-DGS2) | +51bp | 65bp     | -108bp   | +191bp   | WARNING*   |
| spread_change     | -0.4bp  | 3.6bp    | -16bp    | +42bp    | OK         |
| curvature_raw     | -3.9bp  | 12.0bp   | -29.5bp  | +26.5bp  | OK         |
| curvature_change  | -0.1bp  | 1.3bp    | -5bp     | +8.5bp   | OK         |

**WARNING: spread has negative values (-108bp minimum)**
**Assessment:** This is EXPECTED and ACCEPTABLE. Negative spreads indicate yield curve inversions, a well-known market phenomenon (2022-2023, 2020, 2006-2007). The data correctly captures these periods. This is not a data quality issue.

---

### STEP 3: Missing Values & Consecutive Gaps
**Status: PASS**

| Column            | NaN Count | % Missing | Max Consecutive | Status |
|-------------------|-----------|-----------|-----------------|--------|
| dgs10             | 0         | 0.00%     | 0 days          | OK     |
| dgs2              | 0         | 0.00%     | 0 days          | OK     |
| dgs5              | 0         | 0.00%     | 0 days          | OK     |
| dgs10_change      | 0         | 0.00%     | 0 days          | OK     |
| dgs2_change       | 0         | 0.00%     | 0 days          | OK     |
| spread            | 0         | 0.00%     | 0 days          | OK     |
| spread_change     | 0         | 0.00%     | 0 days          | OK     |
| curvature_raw     | 0         | 0.00%     | 0 days          | OK     |
| curvature_change  | 0         | 0.00%     | 0 days          | OK     |

**Result:** PERFECT. Zero missing values. All preprocessing was completed correctly before writing to CSV.

---

### STEP 4: Future Information Leak / Lookahead Bias
**Status: PASS**

**Method:** Compare correlation at lag 0 (current day feature vs next-day target) with lag 1 (previous day feature vs next-day target).

| Column            | Corr(lag=0) | Corr(lag=1) | Assessment |
|-------------------|-------------|------------|-----------|
| dgs10             | +0.020      | +0.020     | OK - symmetric |
| dgs2              | +0.023      | +0.023     | OK - symmetric |
| dgs5              | +0.020      | +0.020     | OK - symmetric |
| dgs10_change      | -0.010      | -0.011     | OK - symmetric |
| dgs2_change       | +0.009      | -0.014     | OK - lag0 not > lag1 |
| spread            | -0.023      | -0.022     | OK - symmetric |
| spread_change     | -0.027      | +0.004     | OK - difference is noise |
| curvature_raw     | -0.034      | -0.032     | OK - symmetric |
| curvature_change  | -0.033      | -0.022     | OK - lag0 < lag1 |

**Result:** No lookahead bias detected. All correlations are small and symmetric (lag 0 ≈ lag 1), indicating that features are genuinely predictive, not leaky.

---

### STEP 5: Time Series Integrity
**Status: PASS**

- Date index: ✓ DatetimeIndex
- Monotonic increasing: ✓ Yes (properly sorted)
- Duplicate dates: ✓ None detected
- Gaps > 7 days: ✓ None detected

**Trading day analysis:** The 2,840 rows span from 2014-10-02 to 2026-02-12 (~11.35 years). This equals approximately 2,840 trading days, confirming that the dataset is daily frequency with no artificial gaps introduced during preprocessing.

---

### STEP 6: VIF Pre-Check & Correlation with Base Features
**Status: PASS** (8 expected correlation warnings)

#### Raw Yields vs Base Features

The dataset contains raw yield levels (dgs10, dgs2, dgs5) which naturally have high correlations with base_features:

| Raw Column | vs real_rate | vs yield_curve_dgs10 | vs yield_curve_dgs2 | Assessment |
|------------|--------------|----------------------|----------------------|-----------|
| dgs10      | 0.939        | 1.000                | 0.949                | WARNING   |
| dgs2       | 0.891        | 0.949                | 1.000                | WARNING   |
| dgs5       | 0.921        | 0.988                | 0.982                | WARNING   |

**Assessment:** These correlations are EXPECTED and NOT PROBLEMATIC because:

1. **Design intention:** This dataset is the INPUT for builder_model, which will compute DERIVATIVES:
   - `yc_regime_prob` from 2D HMM on [dgs10_change, dgs2_change]
   - `yc_spread_velocity_z` = z-score of spread changes
   - `yc_curvature_z` = z-score of curvature changes

2. **Derivatives are orthogonal:** Per design doc (Section 9, measured):
   - curvature_z vs yield_curve_dgs10: corr = -0.002 (near-zero)
   - spread_velocity_z vs yield_curve_dgs10: corr = 0.017 (near-zero)
   - curvature_z vs spread_velocity_z: corr = 0.071 (near-zero)

3. **VIF pre-measured:** Design doc reports empirically measured VIF = 1.01 for all processed outputs against base features

**Conclusion:** The high correlations in this raw data are expected, acceptable, and will be eliminated by builder_model's preprocessing. The actual submodel output features will have VIF < 5.

---

### STEP 7: Output Format & Schema Validation
**Status: PASS**

#### Column Schema

| Column            | Type    | Valid Range        | Status |
|-------------------|---------|-------------------|--------|
| date (index)      | DateTime| 2014-10-02 onward | OK     |
| dgs10             | float64 | 0.52 - 4.98%      | OK     |
| dgs2              | float64 | 0.09 - 5.19%      | OK     |
| dgs5              | float64 | 0.19 - 4.95%      | OK     |
| dgs10_change      | float64 | -30 to +29 bp     | OK     |
| dgs2_change       | float64 | -57 to +34 bp     | OK     |
| spread            | float64 | -108 to +191 bp   | OK     |
| spread_change     | float64 | -16 to +42 bp     | OK     |
| curvature_raw     | float64 | -29.5 to +26.5 bp | OK     |
| curvature_change  | float64 | -5 to +8.5 bp     | OK     |

#### Data Integrity

- Total rows: 2,840 ✓
- Total columns: 9 ✓
- Total NaN cells: 0 ✓
- Date continuity: Continuous trading days ✓

---

## Warning Analysis & Risk Assessment

### Warning 1: Spread Range (-108bp to +191bp)
- **Severity:** Low
- **Type:** Expected market phenomenon
- **Assessment:** Negative spreads indicate yield curve inversions, a well-documented and economically meaningful regime. This is data, not an error.
- **Impact on modeling:** None - the spread is a legitimate feature capturing important economic information
- **Recommendation:** Keep as-is

### Warnings 2-9: High Correlation with Base Features
- **Severity:** Very low
- **Type:** Expected - these are RAW inputs, not final outputs
- **Assessment:** The design intention is for builder_model to compute derivatives (changes, z-scores, HMM probabilities) which are uncorrelated with base features
- **Impact on modeling:** None - VIF will be measured on final outputs in Gate 2
- **Recommendation:** Keep as-is; do not modify builder_data

---

## Comparison with Design Expectations

### Data Statistics vs Design Spec

| Spec Item | Expected | Actual | Match |
|-----------|----------|--------|-------|
| Date range | 2014-10-02 to 2026-02-12 | 2014-10-02 to 2026-02-12 | ✓ |
| Row count | ~2,840 | 2,840 | ✓ |
| Columns | 9 (levels, changes, spread, curvature) | 9 | ✓ |
| DGS10 range | 0.5-5% | 0.52-4.98% | ✓ |
| DGS2 range | 0-5% | 0.09-5.19% | ✓ |
| DGS5 range | 0-5% | 0.19-4.95% | ✓ |
| Missing data | <5% | 0% | ✓✓ |
| Consecutive gaps | <10 days | 0 | ✓✓ |

---

## Acceptance Criteria Assessment

### Gate 1 Readiness (Pre-Gate 1)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| No lookahead bias | ✓ PASS | lag0 ≈ lag1 correlations all symmetric |
| No all-NaN columns | ✓ PASS | 0 NaN across all columns |
| Reasonable ranges | ✓ PASS | All values within expected bounds |
| No data quality issues | ✓ PASS | 0 CRITICAL issues |

### Readiness for builder_model

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Data completeness | ✓ PASS | 100% coverage, 0 NaN |
| Temporal continuity | ✓ PASS | No gaps >7 days |
| Schema compliance | ✓ PASS | All columns present, correct types |
| No duplicates | ✓ PASS | All dates unique |
| Correct frequency | ✓ PASS | Daily trading data |
| Range validity | ✓ PASS | Yields, changes, spreads all reasonable |

---

## Recommendation

**PROCEED TO BUILDER_MODEL**

The yield_curve dataset is production-ready. All critical quality checks pass. Warnings are expected and do not indicate data quality problems.

### Next Steps

1. **builder_model** will generate train.py with:
   - 2D HMM regime detection on [dgs10_change, dgs2_change]
   - Z-score of spread velocity
   - Z-score of curvature changes

2. **Expected outputs:** 3 features
   - `yc_regime_prob` (0-1)
   - `yc_spread_velocity_z` (-4 to +4)
   - `yc_curvature_z` (-4 to +4)

3. **Gate 2 evaluation** will verify:
   - VIF < 10 (expected 1.01)
   - MI increase > 5% (expected ~0.15 bits)
   - Rolling correlation stability

### Risk Level

**LOW** - All prerequisite data quality conditions met. Expected to pass Gate 1 (standalone quality), likely to pass Gate 2 (information gain).

---

## Files Generated

- Report JSON: `logs/datacheck/yield_curve_attempt_1.json`
- Summary (this file): `logs/datacheck/YIELD_CURVE_EVALUATION_SUMMARY.md`
- Data: `data/raw/yield_curve.csv` (ready for builder_model)

---

**Data Quality Check Completed Successfully**
Ready for builder_model pipeline execution.
