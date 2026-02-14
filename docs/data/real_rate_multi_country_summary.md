# Multi-Country Real Rate Features - Data Summary

**Feature**: real_rate (Attempt 3)
**Created**: 2026-02-14
**Status**: ✅ READY FOR TRAINING

---

## Overview

Multi-country interest rate and inflation features for Transformer-based submodel. Uses 6 developed countries (Germany, UK, Canada, Switzerland, Norway, Sweden) plus US TIPS to capture global monetary policy regime patterns.

### Key Design Decision

**Problem**: Synthetic real rates (Nominal - CPI_YoY) have only 0.49 correlation with true TIPS rates.

**Solution**: Provide nominal yields and CPI as **separate inputs** to the Transformer, allowing it to learn the complex, non-linear relationship between backward-looking inflation and forward-looking rate dynamics.

---

## Dataset Specification

### Dimensions
- **Rows**: 269 months (2003-02 to 2025-06)
- **Columns**: 25 features
- **Frequency**: Monthly (month-start)
- **File size**: 95.6 KB
- **Missing values**: 0

### Feature Groups (25 total)

| Group | Count | Features |
|-------|-------|----------|
| US TIPS | 2 | level, change |
| Country Nominal Yields | 6 | DE, UK, CA, CH, NO, SE levels |
| Country Nominal Changes | 6 | DE, UK, CA, CH, NO, SE monthly changes |
| Country CPI (lagged) | 6 | DE, UK, CA, CH, NO, SE year-over-year % |
| Cross-Country Aggregates | 4 | dispersion, change_dispersion, mean_cpi, spread |
| Market Context | 1 | VIX monthly average |

---

## Data Sources

### Primary Series (FRED)

| Country | Nominal 10Y Yield | CPI YoY |
|---------|------------------|---------|
| **US** | DFII10 (daily TIPS, resampled) | - |
| Germany | IRLTLT01DEM156N | CPALTT01DEM659N |
| UK | IRLTLT01GBM156N | CPALTT01GBM659N |
| Canada | IRLTLT01CAM156N | CPALTT01CAM659N |
| Switzerland | IRLTLT01CHM156N | CPALTT01CHM659N |
| Norway | IRLTLT01NOM156N | CPALTT01NOM659N |
| Sweden | IRLTLT01SEM156N | CPALTT01SEM659N |

**VIX**: FRED:VIXCLS (daily, resampled to monthly average)

### Excluded

**Japan**: CPI series (CPALTT01JPM659N) ends 2021-06, too early for current analysis.

---

## Preprocessing Pipeline

1. **Fetch**: US TIPS (daily), VIX (daily), 6 countries × 2 series (monthly)
2. **Resample**: US TIPS and VIX to month-start (MS frequency)
3. **Lag**: CPI shifted 1 month to avoid publication lag
4. **Alignment**: All series aligned to common month-start index
5. **Fill**: Forward-fill missing values (max 3-month limit)
6. **Clean**: Drop rows with remaining NaN values
7. **Output**: 269 months × 25 features, no missing values

---

## Quality Metrics

### Feature Statistics

| Feature Group | Mean | Std | Range |
|--------------|------|-----|-------|
| US TIPS level | 0.93% | 1.01% | -1.16% to 3.14% |
| Country nominal yields | 1.08% - 2.94% | 1.17 - 1.59% | Reasonable |
| Country CPI YoY | 0.57% - 2.63% | 1.07 - 2.49% | Reasonable |
| VIX monthly avg | 19.12 | 8.06 | 10.1 to 62.7 |

### Cross-Country Correlations

**US TIPS vs Country Nominal Yields**: 0.73 - 0.89 (high, as expected)

**Cross-Country Nominal Yields**: Mean correlation 0.95 (very high)
- This confirms yields are dominated by a global factor
- Transformer's task: extract residual country-specific variation and divergence patterns

### Temporal Properties

- **Change columns**: Low autocorrelation (good for stationarity)
- **CPI aggregate**: High autocorrelation 0.98 (expected for inflation)
- **Outliers**: 5 total (0.07% of data), all within 5-6 std (acceptable)

---

## Design Implications

### Why This Matters for the Transformer

1. **Rich Input**: 25 features × 12-24 month windows = 300-600 input dimensions per sample
2. **Compression**: Transformer must compress to 4-6 latent dimensions → ~50-100x compression
3. **Non-trivial task**: High cross-country correlation means genuine information is in the residuals and timing of divergences
4. **Prevents identity mapping**: Narrow bottleneck + diverse inputs makes memorization impossible

### Sample Size Considerations

- **269 monthly windows**: Small for a Transformer
- **Architecture constraints**: d_model=24-48, 2-3 layers, dropout 0.3-0.5
- **Expected overfit ratio**: 1.1-1.3 (with proper regularization)

---

## Files

- **Data**: `data/processed/real_rate_multi_country_features.csv`
- **Metadata**: `data/processed/real_rate_multi_country_metadata.json`
- **Fetch script**: `scripts/fetch_multi_country_features.py`
- **Verification**: `scripts/verify_multi_country_features.py`

---

## Next Steps

1. **datachecker**: Validate data quality (7-step standardized check)
2. **builder_model**: Generate self-contained train.py for Kaggle
3. **Training**: Optuna HPO (30 trials, Transformer Autoencoder)
4. **Evaluation**: Gate 1/2/3 checks

---

## Notes

- Monthly resolution is a deliberate design choice (reduces autocorrelation, increases sample diversity)
- Output will be forward-filled to daily for meta-model alignment
- First-difference postprocessing will be applied to latent outputs to break level autocorrelation
- CPI lag of 1 month prevents look-ahead bias from publication delays

**Status**: Data fetching complete. Ready for datachecker.
