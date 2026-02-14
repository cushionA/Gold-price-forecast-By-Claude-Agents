# Data Quality Check Report: Cross-Asset (Attempt 1)

**Status:** REJECT
**Date:** 2026-02-15
**Feature:** cross_asset
**Attempt:** 1

---

## Executive Summary

The cross_asset.csv data file **FAILED** the standardized 7-step data quality check at **STEP 3: Autocorrelation Check**. Five critical issues were identified where price level columns contain autocorrelation exceeding the Gate 1 threshold of 0.99.

**Root Cause:** The data file contains raw price levels (gold_close, copper_close, silver_close) and derived level-based ratios (gsr, gcr), which inherently have high autocorrelation due to price persistence. These columns are not appropriate inputs for the meta-model.

---

## Detailed Results

### Step 1: Missing Values Check - PASS
- All 8 columns have 0% missing values
- No consecutive NaN sequences
- **Status:** PASS

### Step 2: Basic Statistics & Outliers - PASS
- Row count: 2,860 (acceptable)
- All numeric columns have reasonable statistics:
  - gold_close: min=1050.8, max=5318.4, mean=1790.55, std=729.66
  - copper_close: min=1.94, max=6.18, mean=3.39, std=0.91
  - silver_close: min=11.74, max=115.08, mean=22.48, std=10.34
  - Returns: mean≈0, std≈1-2% (normal for daily returns)
- **Status:** PASS

### Step 3: Autocorrelation Check - FAIL ✗ CRITICAL
This is a **CRITICAL FAILURE**. Five columns exceed the Gate 1 threshold of 0.99:

| Column | Autocorr (lag-1) | Status | Issue |
|--------|------------------|--------|-------|
| gold_close | 0.9994 | CRITICAL | Exceeds 0.99 |
| copper_close | 0.9978 | CRITICAL | Exceeds 0.99 |
| silver_close | 0.9952 | CRITICAL | Exceeds 0.99 |
| gsr (gold/silver ratio) | 0.9920 | CRITICAL | Exceeds 0.99 |
| gcr (gold/copper ratio) | 0.9962 | CRITICAL | Exceeds 0.99 |

All return columns pass (near white noise):
- gold_return: -0.0292 ✓
- silver_return: -0.0608 ✓
- copper_return: -0.0490 ✓

**Status:** REJECT

### Step 4: Future Leak Check - PASS
- No critical leaks detected
- silver_return has 0.7727 correlation with gold_return (expected for precious metals, not a leak)
- All other correlations reasonable
- **Status:** PASS

### Step 5: Time Series Alignment - PASS
- Dates properly sorted
- No duplicate dates
- Date range: 2014-10-01 to 2026-02-13 (matches expected range)
- No gaps > 7 days
- **Status:** PASS

### Step 6: VIF Pre-Check - PASS (with warnings)
- Price levels show perfect correlation with raw base features (expected: gold_close matches technical_gld_close, etc.)
- These correlations are not true multicollinearity issues - they reflect the same data
- However, they highlight that this file contains redundant information
- **Status:** PASS (acceptable for pre-check)

### Step 7: Schema Validation - PASS
- All expected columns present
- All columns are float64 type
- **Status:** PASS

---

## Critical Issues

**5 CRITICAL FAILURES in Step 3 - Autocorrelation:**

The fundamental problem is that the data file contains **price levels** and **level-based ratios**, which are inherently non-stationary time series with high autocorrelation. This violates Gate 1 requirements.

### Why This Is Critical

According to the design doc (Section 3, Architecture), the cross_asset submodel should output:
1. **xasset_regime_prob** (from 3D HMM) - autocorr expected 0.60-0.83 ✓
2. **xasset_recession_signal** (first difference of z-score) - autocorr expected ~-0.04 ✓
3. **xasset_divergence** (daily return difference z-score) - autocorr expected ~0.03 ✓

Instead, the builder_data provided:
- price_close columns (autocorr 0.99+) ✗
- ratio levels: gsr, gcr (autocorr 0.99+) ✗
- returns (autocorr near 0) ✓

**The ratio levels (gsr, gcr) were supposed to be converted to first-differences per the design doc:**
- "Use the FIRST DIFFERENCE (daily change) of the z-score instead, which has autocorr = -0.039"
- "Use daily gold-silver return difference z-scored against rolling std, which will have much lower autocorrelation"

But builder_data delivered raw ratio levels instead of their first differences or z-score changes.

---

## Impact on Meta-Model

Feeding columns with autocorr > 0.99 to the meta-model violates Gate 1 assumptions:
- High autocorrelation means adjacent observations are nearly identical
- XGBoost trees would learn trivial patterns (today's price ≈ yesterday's price)
- This provides **zero information** about next-day returns
- Wastes meta-model capacity

---

## Required Actions

### Return to builder_data (Attempt 2)

The builder_data agent must:

1. **DO NOT** include price level columns (gold_close, copper_close, silver_close, gsr, gcr raw levels)
2. **ONLY** provide the three submodel outputs designed:
   - xasset_regime_prob (3D HMM posterior probability)
   - xasset_recession_signal (first difference of gold/copper ratio z-score)
   - xasset_divergence (daily gold-silver return difference z-score)
3. Verify autocorrelation of all output columns:
   - All must be < 0.99 (preferably < 0.85)
   - Returns-based features should be near white noise (autocorr < 0.1)
4. Ensure output format:
   - CSV with columns: [date, xasset_regime_prob, xasset_recession_signal, xasset_divergence]
   - Aligned to trading dates (same date index as base_features.csv)
   - ~2,523 rows after alignment

### Specific Design Doc Corrections to Implement

From design doc Section 3, Component 2 (Gold/Copper Ratio):
```
z_change = z.diff()   # FIRST DIFFERENCE
clip(-4, 4)
Output: xasset_recession_signal (typically -1 to +1)
```

The z-score level has autocorr 0.9587 - this is explicitly stated as a danger.
**First difference must be used.**

From design doc Section 3, Component 3 (Gold-Silver Divergence):
```
Input: gold_ret - silver_ret = gs_ret_diff
Z-score the daily difference against 20d rolling std
Output: xasset_divergence (typically -3 to +3)
```

Daily return difference is near-white-noise. This is what should be in the file.

---

## Differences Report

| Issue | Expected (Design Doc) | Provided (cross_asset.csv) | Gap |
|-------|----------------------|--------------------------|-----|
| Regime probability | HMM posterior, autocorr ~0.83 | Not provided | Missing |
| Recession signal | First diff of z-score, autocorr ~-0.04 | gsr (ratio level), autocorr 0.992 | Wrong form |
| Divergence | Daily ret diff z-score, autocorr ~0.03 | Not provided | Missing |
| Price levels | Should not be included | Included all closes | Unnecessary |

---

## Attempt Count

- **Current Attempt:** 1 of 5
- **Remaining Attempts:** 4
- **Consecutive Rejects:** 1 (not yet at 3-reject limit)

---

## Next Steps

1. builder_data: Fix data generation per design doc (Attempt 2)
2. Verify autocorrelation of all outputs
3. Resubmit data/raw/cross_asset.csv
4. Run datacheck again (this script)

---

## Files

- **Data file:** `/absolute/path/C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents\data\raw\cross_asset.csv`
- **Design doc:** `/absolute/path/C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents\docs\design\cross_asset_attempt_1.md`
- **Datacheck report:** `/absolute/path/C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents\logs\datacheck\cross_asset_attempt_1.json`
- **This evaluation:** `/absolute/path/C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents\logs\evaluation\cross_asset_datacheck_attempt_1.md`

---

## Recommendation

**REJECT and return to builder_data for Attempt 2.**

The data file fundamentally mismatches the design specification. However, the issue is clear and straightforward to fix:
- Generate the three submodel outputs as designed
- Ensure all outputs are computed from returns or first-differences
- Verify autocorrelation < 0.99 before submission

This is a correctable data generation error, not a design flaw.
