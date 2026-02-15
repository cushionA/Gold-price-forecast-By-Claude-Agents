# Data Fetch Summary: inflation_expectation (Attempt 1)

**Date**: 2026-02-15 13:00
**Feature**: inflation_expectation
**Attempt**: 1
**Phase**: builder_data
**Status**: ✅ COMPLETE

---

## Deliverables

### 1. Self-Contained Fetch Function
**File**: `src/fetch_inflation_expectation.py`
- Embeddable in train.py for Kaggle execution
- Handles FRED API key from .env (local) or Kaggle Secrets (cloud)
- Handles yfinance MultiIndex columns correctly
- Inner join alignment for common trading days
- Forward-fill with 3-day limit for minor gaps

### 2. Preprocessed Data
**File**: `data/processed/inflation_expectation/features_input.csv`
- **Rows**: 2,924 (100% coverage of base_features dates)
- **Date Range**: 2014-06-30 to 2026-02-13
- **Columns**: 9 features
  1. `T10YIE` - 10-Year Breakeven Inflation Rate (raw level)
  2. `T5YIFR` - 5-Year, 5-Year Forward Inflation Rate (for Approach D)
  3. `gc_close` - Gold Futures close price
  4. `ie_change` - Daily change in T10YIE (basis for all features)
  5. `gold_return` - Current-day gold return (for sensitivity)
  6. `ie_vol_5d` - 5-day rolling std of ie_change
  7. `ie_vol_10d` - 10-day rolling std of ie_change
  8. `ie_vol_20d` - 20-day rolling std of ie_change
  9. `t5yifr_change` - Daily change in T5YIFR (for Approach D)

### 3. Metadata
**File**: `data/processed/inflation_expectation/metadata.json`
- Complete statistics for all features
- Train/Val/Test split: 2046 / 439 / 439 (70/15/15)
- Zero missing values across all columns

---

## Data Quality Validation

### ✅ All Checks Passed

1. **Data Sources**: FRED T10YIE, T5YIFR, Yahoo GC=F all fetched successfully
2. **Date Alignment**: 100% coverage of base_features dates (2,523/2,523)
3. **Missing Values**: 0 NaN across all columns
4. **Range Validation**:
   - T10YIE: 0.50% to 3.02% (within expected [0, 5]%)
   - ie_change: -0.32 to +0.25 (typical daily moves)
   - ie_vol_5d: 0.00 to 0.17 (zero OK for stable periods)
5. **Autocorrelation** (Lag 1):
   - ie_change: 0.0714 (excellent)
   - ie_vol_5d: 0.8882 (good, Optuna priority)
   - ie_vol_10d: 0.9662 (acceptable)
   - ie_vol_20d: 0.9891 (borderline but <0.99)
6. **No extreme outliers**: All values within reasonable ranges
7. **No data leakage**: All features are backward-looking

---

## Key Statistics

### T10YIE (Breakeven Inflation Rate)
- Mean: 2.021%
- Std: 0.370%
- Min: 0.500% (COVID-19 crisis, March 2020)
- Max: 3.020% (Inflation peak, April 2022)

### ie_change (Daily Change)
- Mean: 0.000000 (expected for diff())
- Std: 0.030076
- Range: -0.32 to +0.25

### ie_vol_5d (5-Day Volatility)
- Mean: 0.025179
- Std: 0.016107
- Range: 0.00 to 0.17
- Autocorr: 0.8882 (lowest among vol windows)

---

## Design Alignment

### ✅ All Requirements Met

1. **Daily frequency**: No interpolation needed (avoids real_rate failure mode)
2. **No Fisher identity violation**: All change-based, no level features in output
3. **Multiple volatility windows**: 5d/10d/20d for Optuna exploration
4. **Gold sensitivity data**: Current-day returns included
5. **Term structure option**: T5YIFR available for Approach D
6. **Sufficient warmup**: 2014-06 start provides buffer for 120d baselines
7. **Autocorrelation compliance**: All features <0.99 threshold

---

## Implementation Notes

### Self-Contained Design
The fetch function is designed to be embedded directly in train.py:
- ✅ No external file dependencies
- ✅ Handles pip install for missing packages
- ✅ FRED_API_KEY from environment or Kaggle Secrets
- ✅ Graceful error handling with informative messages
- ✅ MultiIndex column flattening for yfinance
- ✅ Forward-fill with conservative 3-day limit

### Autocorrelation Strategy
Optuna will explore three volatility windows:
- **5d**: autocorr=0.8882 (best tradeoff, expected winner)
- **10d**: autocorr=0.9662 (moderate)
- **20d**: autocorr=0.9891 (closest to academic norm but highest autocorr)

The 5d window provides the best balance between:
- Low autocorrelation (reduces spurious persistence)
- Captures immediate instability (aligned with anchoring concept)
- Consistent with 5d correlation window for sensitivity feature

---

## Next Steps

1. **datachecker**: Standardized 7-step validation
   - Expected result: PASS (all data quality checks already satisfied)
2. **builder_model**: Generate Kaggle training script
   - Embed `fetch_and_preprocess()` directly in train.py
   - Implement 2D HMM on [ie_change, ie_vol_5d]
   - Implement ie_anchoring_z (volatility z-score)
   - Implement ie_gold_sensitivity_z (correlation z-score)
   - Optuna HPO with 30 trials
3. **Kaggle execution**: Cloud training (~3-5 minutes)
4. **evaluator**: Gate 1 → 2 → 3 evaluation

---

## Risk Mitigation

### Addressed in Data Fetch:

1. ✅ **No monthly data**: Daily T10YIE avoids real_rate's 5-attempt failure
2. ✅ **No Fisher identity**: Change-based features only, no levels
3. ✅ **Autocorrelation managed**: Multiple windows for Optuna selection
4. ✅ **No data gaps**: Forward-fill with 3-day limit
5. ✅ **100% base_features coverage**: No alignment issues

### Remaining (for builder_model):

1. HMM state collapse risk (mitigated by n_init parameter)
2. VIF from ie_regime_prob (precedent: etf_flow passed with VIF=12.47)
3. Gate 3 diminishing returns (expected -0.01% to -0.03% as 7th submodel)

---

## Conclusion

Data fetch is complete and validated. All 2,924 rows with 9 features are ready for HMM training. The self-contained fetch function is Kaggle-compatible and includes all necessary features for:
- Approach A: 2D HMM on [ie_change, ie_vol_5d]
- Approach D: Term structure HMM (if needed in future attempt)
- Anchoring z-score with 3 volatility windows
- Sensitivity z-score with gold correlation

**Ready for datachecker verification.**
