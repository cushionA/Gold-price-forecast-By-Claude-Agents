# Gold DOWN Classifier - Data Validation Summary

**Date**: 2026-02-18
**Status**: READY FOR builder_model

---

## Data Sources Validation

### yfinance Tickers (all accessible)
| Ticker | Fields | Date Range | Rows | Status |
|--------|--------|------------|------|--------|
| GC=F | OHLCV | 2014-01-02 to 2026-02-17 | 3,048 | [OK] |
| GLD | Volume | 2014-01-02 to 2026-02-17 | 3,049 | [OK] |
| SI=F | Close | 2014-01-02 to 2026-02-17 | 3,048 | [OK] |
| HG=F | Close | 2014-01-02 to 2026-02-17 | 3,049 | [OK] |
| DX-Y.NYB | Close | 2014-01-02 to 2026-02-17 | 3,050 | [OK] |
| ^GSPC | Close | 2014-01-02 to 2026-02-17 | 3,049 | [OK] |

### FRED Series (all accessible)
| Series | Description | Date Range | Rows | Status |
|--------|-------------|------------|------|--------|
| GVZCLS | Gold VIX | 2014-01-01 to 2026-02-13 | 3,163 | [OK] |
| VIXCLS | VIX | 2014-01-01 to 2026-02-16 | 3,164 | [OK] |
| DFII10 | 10Y TIPS Yield | 2014-01-01 to 2026-02-12 | 3,162 | [OK] |
| DGS10 | 10Y Nominal Yield | 2014-01-01 to 2026-02-12 | 3,162 | [OK] |
| DGS2 | 2Y Nominal Yield | 2014-01-01 to 2026-02-12 | 3,162 | [OK] |

**Note**: GVZCLS (Gold VIX) starts 2014-01-02, well before the 2015-01-01 requirement. No data availability issues.

---

## Data Quality

### Missing Data Handling
- **Before forward-fill**: <1% NaN across all series (mostly weekends/holidays)
- **After forward-fill**: 0 NaN (100% clean)
- **Forward-fill limits**:
  - FRED series: max 5 days
  - yfinance series: max 3 days

### Final Dataset Dimensions
- **Total rows after merge**: 3,048 trading days
- **Effective rows after warmup**: 2,959 rows (89 days dropped for 60-day rolling windows)
- **Features**: 18
- **Date range (effective)**: 2014-05-12 to 2026-02-17

### Target Balance
- **UP days** (return > 0%): 52.72%
- **DOWN days** (return <= 0%): 47.28%
- **Class imbalance ratio**: 1.12:1 (mild, acceptable for XGBoost)

---

## Data Splits (time-series order, no shuffle)

| Split | Rows | Percentage | Date Range |
|-------|------|------------|------------|
| Train | 2,071 | 70% | 2014-05-12 to 2022-08-04 |
| Validation | 444 | 15% | 2022-08-05 to 2024-05-10 |
| Test | 444 | 15% | 2024-05-13 to 2026-02-17 |

**Samples-per-feature ratio**: 2,071 / 18 = 115:1 (excellent, well above 10:1 minimum)

---

## Feature Engineering Verification

All 18 features successfully computed:

### Category A: Volatility Regime (5 features)
1. `rv_ratio_10_30` - Realized volatility ratio
2. `rv_ratio_10_30_z` - Z-scored vol ratio
3. `gvz_level_z` - Gold VIX z-score
4. `gvz_vix_ratio` - Gold vol / equity vol
5. `intraday_range_ratio` - Daily range vs average

### Category B: Cross-Asset Stress (4 features)
6. `risk_off_score` - Multi-asset stress composite
7. `gold_silver_ratio_change` - Gold/silver divergence
8. `equity_gold_beta_20d` - Rolling beta to S&P 500
9. `gold_copper_ratio_change` - Gold/copper divergence

### Category C: Rate and Currency Shock (3 features)
10. `rate_surprise` - Real rate shock magnitude
11. `rate_surprise_signed` - Directional rate surprise
12. `dxy_acceleration` - DXY second derivative

### Category D: Volume and Flow (2 features)
13. `gld_volume_z` - GLD volume z-score
14. `volume_return_sign` - Volume-price agreement

### Category E: Momentum Context (2 features)
15. `momentum_divergence` - Short vs medium term
16. `distance_from_20d_high` - Position in 20-day range

### Category F: Calendar (2 features)
17. `day_of_week` - 0=Monday, 4=Friday
18. `month_of_year` - 1-12

---

## Sample Statistics

### Gold Return Distribution
- **Mean**: 0.0504%
- **Std**: 0.9981%
- **Min**: -11.37%
- **Max**: 6.08%

### GVZ/VIX Ratio
- **Mean**: 0.97
- **Std**: 0.25
- **Range**: [0.38, 2.73]

---

## Issues & Warnings

**No critical issues detected.**

Minor notes:
- 89 rows (3%) dropped due to 60-day rolling window warmup (expected)
- All data sources cover the required 2015-01-01 start date with margin
- No feature correlations >0.85 (checked in validation script)

---

## Files Generated

1. **Raw data**: `data/raw/classifier_raw.csv` (3,048 rows × 19 columns)
2. **Validation log**: `logs/datacheck/classifier_data_validation.log`
3. **Data fetching function**: `src/fetch_classifier.py` (self-contained, ready for Kaggle)

---

## Ready for builder_model

The data fetching function in `src/fetch_classifier.py` is:
- **Self-contained**: No external file dependencies
- **Kaggle-compatible**: Handles FRED_API_KEY from Kaggle Secrets
- **Deterministic**: All features computed using vectorized pandas operations
- **Tested**: Runs successfully and produces expected output

**Next step**: builder_model can embed this function directly into `notebooks/classifier_1/train.ipynb`.

---

## Recommendations for builder_model

1. **Use the exact `fetch_and_preprocess()` function** from `src/fetch_classifier.py`
2. **Flatten multi-level column names** from yfinance using:
   ```python
   if isinstance(df.columns, pd.MultiIndex):
       df.columns = [col[0] for col in df.columns]
   ```
3. **Expected runtime on Kaggle**: ~5 minutes for data fetching + feature engineering
4. **Memory usage**: <500 MB (tiny dataset)
5. **No GPU needed**: XGBoost with tree_method=hist on <3K rows is fast on CPU

---

**Validation Status**: PASS
**Ready for training**: YES
**Estimated Kaggle execution time**: 20-25 minutes (including 100 Optuna trials)
