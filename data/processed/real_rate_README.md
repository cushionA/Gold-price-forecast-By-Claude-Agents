# Real Rate Feature Data - Attempt 2

## Overview

**Feature**: Real Interest Rate (10-Year TIPS)
**Attempt**: 2
**Generated**: 2026-02-14 15:15:21
**Status**: Ready for training

## Files

- `real_rate_features.csv` - Feature data (2523 rows × 8 features)
- `real_rate_metadata.json` - Metadata and statistics
- `../scripts/fetch_real_rate_features.py` - Reproducible fetching script

## Data Source

- **API**: FRED (Federal Reserve Economic Data)
- **Series**: DFII10 (10-Year Treasury Inflation-Indexed Security, Constant Maturity)
- **Fetch Range**: 2013-06-01 to latest (buffer for rolling windows)
- **Output Range**: 2015-01-30 to 2025-02-12 (schema-aligned)

## Features (8 columns)

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| level | float64 | Raw 10Y TIPS yield (%) | -1.19 to 2.52 |
| change_1d | float64 | Daily change | -0.45 to 0.39 |
| velocity_20d | float64 | 20-day normalized change | normalized |
| velocity_60d | float64 | 60-day normalized change | normalized |
| accel_20d | float64 | Acceleration (velocity change) | normalized |
| rolling_std_20d | float64 | 20-day volatility | >0 |
| regime_percentile | float64 | 252-day percentile rank | 0.0 to 1.0 |
| autocorr_20d | float64 | 60-day lag-1 autocorrelation | -0.36 to 0.48 |

## Data Quality

- **NaN Count**: 0 (all features complete)
- **Alignment**: Gold trading calendar (GC=F from yfinance)
- **Forward Fill**: Max 5 days for FRED holidays
- **Buffer Strategy**: Fetch from 2013-06-01 to allow 252-day rolling windows before schema start
- **Schema Compliance**: YES (matches schema_freeze.json)

## Statistics

### Level (Raw TIPS Yield)
- Mean: 0.5147%
- Std: 0.9157%
- Min: -1.19% (negative real rates during QE)
- Max: 2.52% (recent tightening cycle)

### Change (Daily)
- Mean: 0.0008%
- Std: 0.0491%
- Shows mean-reverting behavior with occasional large moves

## Usage for builder_model

The `fetch_real_rate_features()` function in `scripts/fetch_real_rate_features.py` is **self-contained** and can be embedded directly into Kaggle training scripts.

### Integration Steps

1. Copy the `fetch_real_rate_features()` function from lines 14-108
2. Paste into train.py before the model definition
3. Call it to get features: `df = fetch_real_rate_features()`
4. Features are ready for standardization and model input

### Key Requirements

- FRED_API_KEY must be set in Kaggle Secrets
- Libraries: fredapi, yfinance, pandas, numpy, scipy
- No external file dependencies
- Returns DataFrame with DatetimeIndex and 8 float64 columns

## Reuse from Attempt 1

This data uses the **exact same logic** validated in Attempt 1:
- Same 8 feature definitions
- Same rolling window sizes (20d, 60d, 252d)
- Same alignment to gold trading calendar
- Same NaN handling (no NaN in schema range)

**Difference from Attempt 1**: Only the model architecture changes (Attempt 2 uses GRU-Autoencoder instead of MLP-Autoencoder). The feature engineering is identical.

## Validation Results

```
[OK] Row count: 2523 (matches schema_freeze.json)
[OK] Date range: 2015-01-30 to 2025-02-12 (covers schema)
[OK] NaN count: 0 (all features complete)
[OK] Feature count: 8 (as designed)
[OK] Data types: All float64 (numeric)
[OK] TIPS yield range: Reasonable (0.5% mean, 0.9% std)
```

## Next Steps

builder_model will:
1. Embed `fetch_real_rate_features()` into train.py
2. Implement GRU-Autoencoder (as per architect design)
3. Add Optuna HPO (search space from architect)
4. Generate self-contained Kaggle notebook
5. Submit for training via Kaggle API
