# Data Fetching Summary: regime_classification (Attempt 1)

**Date**: 2026-02-18
**Agent**: builder_data

## Overview

Fetched and preprocessed data for the regime_classification GMM submodel according to design document `docs/design/regime_classification_gmm_1.md`.

## Data Sources

| Source | ID/Ticker | Purpose | Records | Date Range |
|--------|-----------|---------|---------|------------|
| FRED | VIXCLS | VIX level | 3,164 | 2014-01-02 to present |
| FRED | DGS10 | 10Y Treasury Yield | 3,163 | 2014-01-02 to present |
| FRED | DGS2 | 2Y Treasury Yield | 3,163 | 2014-01-02 to present |
| Yahoo | ^GSPC | S&P 500 | 3,049 | 2014-01-02 to present |
| Yahoo | GC=F | Gold Futures | 3,049 | 2014-01-02 to present |

## Feature Engineering

Created 4 z-scored input features:

1. **vix_z**: z-score of VIX level (20d rolling window)
2. **yield_spread_z**: z-score of (DGS10 - DGS2) (60d rolling window)
3. **equity_return_z**: z-score of S&P 500 5d return (60d rolling window)
4. **gold_rvol_z**: z-score of Gold 10d realized volatility (60d rolling window)

All z-scores are clipped to [-4, 4] to handle extreme outliers.

## Processing Steps

1. Fetched raw data from FRED and Yahoo Finance
2. Aligned all series on common trading dates (inner join)
3. Forward-filled missing values (5 days for FRED, 3 days for Yahoo)
4. Computed intermediate features:
   - yield_spread = DGS10 - DGS2
   - spx_5d_ret = S&P 500 5-day return
   - gold_rvol_10d = Gold 10-day realized volatility (annualized)
5. Computed rolling z-scores for each feature
6. Clipped z-scores to [-4, 4]
7. Dropped rows with NaN (from rolling window warmup)

## Output

**Processed features**: `data/processed/regime_classification_features.csv`
- Rows: 2,976
- Date range: 2014-04-11 to 2026-02-13
- Columns: vix_z, yield_spread_z, equity_return_z, gold_rvol_z
- NaN count: 0

**Raw data saved to**: `data/raw/`
- vix.csv
- dgs10.csv
- dgs2.csv
- spx.csv
- gold_futures.csv

**Reusable fetching code**: `src/fetch_regime_classification.py`
- Self-contained function for embedding in Kaggle notebook
- Handles both local (.env) and Kaggle Secrets authentication

## Data Split (70/15/15)

- **Train**: 2,083 rows (2014-04-11 to 2022-07-25)
- **Val**: 446 rows (2022-07-26 to 2024-05-02)
- **Test**: 447 rows (2024-05-03 to 2026-02-13)

## Data Quality Validation

### Missing Values
- vix_z: 0 (0.00%)
- yield_spread_z: 0 (0.00%)
- equity_return_z: 0 (0.00%)
- gold_rvol_z: 0 (0.00%)

### Value Ranges (expected: approximately [-4, 4])
- vix_z: [-2.879, 4.000]
- yield_spread_z: [-3.609, 4.000]
- equity_return_z: [-4.000, 3.280]
- gold_rvol_z: [-3.520, 4.000]

### Z-Score Statistics (expected: mean≈0, std≈1)
- vix_z: mean=-0.038, std=1.208
- yield_spread_z: mean=-0.197, std=1.388
- equity_return_z: mean=-0.050, std=1.060
- gold_rvol_z: mean=0.015, std=1.208

Note: Means and stds deviate slightly from 0 and 1 due to rolling window edge effects and clipping, but are within acceptable ranges.

### Pairwise Correlations
```
                 vix_z  yield_spread_z  equity_return_z  gold_rvol_z
vix_z            1.000           0.032           -0.664        0.085
yield_spread_z   0.032           1.000           -0.026        0.032
equity_return_z -0.664          -0.026            1.000       -0.036
gold_rvol_z      0.085           0.032           -0.036        1.000
```

**Key observations**:
- Strong negative correlation between vix_z and equity_return_z (-0.664): Expected risk-off pattern
- Low correlations for yield_spread_z: Provides independent information
- Low correlation for gold_rvol_z: Provides independent information

## Validation Checks

- [PASS] No NaN values
- [PASS] All values in [-4, 4]
- [PASS] Z-scores approximately normalized
- [PASS] Adequate sample size (>2500)

**Overall**: ALL CHECKS PASSED

## Notes for builder_model

1. **Self-contained code**: The function in `src/fetch_regime_classification.py` can be directly embedded in the Kaggle notebook
2. **API authentication**: Code handles both local (.env) and Kaggle Secrets (FRED_API_KEY)
3. **yfinance compatibility**: Handles both old and new yfinance versions (with/without MultiIndex columns)
4. **No external dependencies**: Function only requires fredapi, yfinance, pandas, numpy (all available in Kaggle)
5. **Time-series split**: The 70/15/15 split maintains temporal order (no shuffling)

## Issues Encountered and Resolved

1. **yfinance MultiIndex columns**: New yfinance versions return MultiIndex columns. Fixed by checking column structure and extracting appropriately.
2. **Unicode encoding in Windows console**: cp932 encoding issue with checkmark symbols. Not a data issue, just logging.

## Ready for datachecker

The data is ready for datachecker validation. All standardized checks should pass.
