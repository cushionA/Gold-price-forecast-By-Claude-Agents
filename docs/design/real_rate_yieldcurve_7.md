# Real Rate Submodel - Attempt 7 Design Document
# Real Yield Curve Shape Features

## Architect Fact-Check

Research report verified:
- DFII5, DFII7, DFII10, DFII20, DFII30: all confirmed daily series on FRED
- All start 2015-01-02, all end 2025-02-12, all have 2530 observations
- No missing values in study period
- No forward-fill needed (all daily business day frequency)
- Root cause of attempts 3-5 failure confirmed: monthly-to-daily forward-fill step function

Research report is factually accurate. Proceeding to design.

---

## Design Overview

| Attribute | Value |
|-----------|-------|
| Model Type | Deterministic transformation (no neural network) |
| Data Sources | FRED DFII5, DFII7, DFII10, DFII20, DFII30 (all daily) |
| Output Columns | 4 features |
| Training Required | No (pure computation) |
| Kaggle GPU | Not needed (CPU sufficient) |
| Estimated Runtime | < 2 minutes |

---

## Feature Specification

### Feature 1: rr_level_change_z
- **Formula**: rolling_zscore(DFII10.diff(), window=30)
- **Interpretation**: Standardized daily change in 10Y real yield (velocity of level)
- **Gold correlation**: -0.1097 (stronger negative: rising rates hurt gold)
- **Autocorrelation**: 0.0233 (no persistence - changes are nearly iid)
- **Rationale**: DFII10 daily change is the cleanest signal. Z-scoring with 30-day window normalizes across rate regimes (ZIRP vs normal rates).

### Feature 2: rr_slope_chg_z
- **Formula**: rolling_zscore((DFII30 - DFII5).diff(), window=60)
- **Interpretation**: Standardized daily change in slope (steepening/flattening velocity)
- **Gold correlation**: +0.0643 (steepening historically gold-positive in risk-off)
- **Autocorrelation**: -0.0405 (no persistence)
- **Rationale**: Cross-sectional shape change is orthogonal to level change. 60-day window for slope change captures medium-term steepening/flattening regimes.

### Feature 3: rr_curvature_chg_z
- **Formula**: rolling_zscore((2*DFII10 - DFII5 - DFII30).diff(), window=60)
- **Interpretation**: Standardized daily change in curvature (belly distortion velocity)
- **Gold correlation**: +0.0121
- **Autocorrelation**: -0.1851 (slight mean-reversion - expected for second differences)
- **Rationale**: Curvature captures the third Nelson-Siegel factor. Independent information from both level and slope changes (correlations: 0.32 with level chg, 0.21 with slope chg).

### Feature 4: rr_slope_level_z
- **Formula**: rolling_zscore(DFII30 - DFII5, window=60)
- **Interpretation**: Regime indicator - how steep/inverted is the real yield curve currently
- **Gold correlation**: +0.0502
- **Autocorrelation**: 0.9370 (high persistence - expected for regime indicator)
- **Rationale**: Absolute slope level indicates current real rate regime (inverted = recession risk = gold positive). Z-score normalizes across economic cycles. High autocorrelation is acceptable for a regime indicator.

---

## Data Pipeline

```python
# Step 1: Fetch all 5 DFII series from FRED
series = {
    'DFII5': fred.get_series('DFII5', start='2015-01-01', end='2025-02-12'),
    'DFII7': fred.get_series('DFII7', start='2015-01-01', end='2025-02-12'),
    'DFII10': fred.get_series('DFII10', start='2015-01-01', end='2025-02-12'),
    'DFII20': fred.get_series('DFII20', start='2015-01-01', end='2025-02-12'),
    'DFII30': fred.get_series('DFII30', start='2015-01-01', end='2025-02-12'),
}
df = pd.DataFrame(series).dropna(how='all')

# Step 2: Compute intermediate series
df['slope'] = df['DFII30'] - df['DFII5']          # long-short spread
df['curvature'] = 2*df['DFII10'] - df['DFII5'] - df['DFII30']  # belly distortion

# Step 3: Compute features
def rolling_zscore(series, window):
    mu = series.rolling(window, min_periods=window//2).mean()
    sigma = series.rolling(window, min_periods=window//2).std()
    sigma = sigma.replace(0, np.nan)
    return (series - mu) / sigma

features = pd.DataFrame(index=df.index)
features['rr_level_change_z'] = rolling_zscore(df['DFII10'].diff(), 30)
features['rr_slope_chg_z'] = rolling_zscore(df['slope'].diff(), 60)
features['rr_curvature_chg_z'] = rolling_zscore(df['curvature'].diff(), 60)
features['rr_slope_level_z'] = rolling_zscore(df['slope'], 60)

# Step 4: Align to target date range (2015-01-02 to 2025-02-12)
target = pd.read_csv('data/processed/target.csv', index_col=0, parse_dates=True)
features = features.reindex(target.index)  # business days only

# Step 5: Save
features.to_csv('data/processed/real_rate/data.csv')
```

---

## Expected Output

| Column | Min | Max | Autocorr | Std |
|--------|-----|-----|----------|-----|
| rr_level_change_z | ~-4 | ~4 | ~0.02 | ~1.0 |
| rr_slope_chg_z | ~-4 | ~4 | ~-0.04 | ~1.0 |
| rr_curvature_chg_z | ~-4 | ~4 | ~-0.19 | ~1.0 |
| rr_slope_level_z | ~-3 | ~3 | ~0.94 | ~1.0 |

NaN rows: approximately 60 at start (z-score warmup). Total valid rows: ~2470 of 2530.

---

## VIF Analysis

Cross-feature correlations (all < 0.35, VIF expected < 2):
- rr_level_change_z vs rr_slope_chg_z: -0.13
- rr_level_change_z vs rr_curvature_chg_z: +0.32
- rr_level_change_z vs rr_slope_level_z: -0.04
- rr_slope_chg_z vs rr_curvature_chg_z: +0.21
- rr_slope_chg_z vs rr_slope_level_z: +0.20
- rr_curvature_chg_z vs rr_slope_level_z: +0.04

All correlations are below 0.35. VIF for all features should be < 2 (well below the 10 threshold).

---

## Hyperparameter Search Space

No model training required. All features are deterministic. No Optuna needed.

The only "parameters" are the rolling windows:
- level_window: 30 days (fixed - standard for monthly regime)
- slope_window: 60 days (fixed - standard for medium-term trend)
- curvature_window: 60 days (fixed - same as slope for consistency)

These are fixed by design, not optimized.

---

## Kaggle Notebook Design

Since no training is required, the Kaggle notebook simply:
1. Fetches all 5 DFII series using Kaggle Secrets FRED_API_KEY
2. Computes the 4 features deterministically
3. Aligns to the target date range
4. Saves submodel_output.csv to Kaggle output
5. Reports statistics in training_result.json

The notebook also loads the existing submodel dataset to generate a comparison with the meta-model input structure.

---

## Gate Expectations

### Gate 1: Standalone Quality
- Overfit ratio: N/A (no model = no overfitting). Gate 1 pass by design.
- All-NaN columns: None expected (all series have full coverage)
- Constant columns: None expected (z-scores have variance ~1.0)
- Status: PASS expected

### Gate 2: Information Gain
- MI increase target: > 5%
- Attempt 3 achieved MI +23.8% with real rate data (even with forward-fill noise)
- Daily multi-tenor without forward-fill should achieve comparable or better MI
- VIF: All features have correlations < 0.35, expected VIF < 2
- Stability (rolling corr std): Change-based features should have low stability (std < 0.15)
- rr_slope_level_z has high autocorrelation but is a regime feature - expected to have stable relationship
- Status: PASS expected

### Gate 3: Ablation
- DA delta target: +0.5%
- Sharpe delta target: +0.05
- MAE delta target: -0.01%
- Risk: rr_slope_level_z has high autocorrelation (0.937) - may add noise. If Gate 3 fails, dropping this feature is the first improvement action.
- Status: UNCERTAIN (Gate 3 has failed 6 times previously)

---

## Risk Mitigation

1. If Gate 3 fails on attempt 7 - the improvement plan for attempt 8 is: drop rr_slope_level_z (3 features only)
2. If rr_curvature_chg_z fails datachecker due to correlation > 0.8 with rr_level_change_z: check actual correlation (expected 0.32, well below 0.8)
3. Slope and curvature windows can be varied (45/30/90 days) in future attempts

---

## File Output Specification

```
data/processed/real_rate/
├── data.csv        <- 4 feature columns, date index (required by datachecker)
└── metadata.json   <- Required by datachecker STEP 1
```

metadata.json:
```json
{
  "feature": "real_rate",
  "attempt": 7,
  "description": "Real yield curve shape features from DFII5/DFII7/DFII10/DFII20/DFII30",
  "columns": ["rr_level_change_z", "rr_slope_chg_z", "rr_curvature_chg_z", "rr_slope_level_z"],
  "date_range_start": "2015-01-02",
  "date_range_end": "2025-02-12",
  "frequency": "daily",
  "sources": ["FRED:DFII5", "FRED:DFII7", "FRED:DFII10", "FRED:DFII20", "FRED:DFII30"],
  "created_at": "2026-02-21"
}
```

---

## Notebook Kernel Configuration

```json
{
  "id": "bigbigzabuton/gold-real-rate-7",
  "title": "Gold Real Rate Model - Attempt 7",
  "code_file": "train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["bigbigzabuton/gold-prediction-submodels"],
  "competition_sources": [],
  "kernel_sources": []
}
```

GPU disabled (not needed for deterministic computation). Internet enabled for FRED API access.
