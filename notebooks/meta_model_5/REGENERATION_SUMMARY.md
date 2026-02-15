# Meta-Model Attempt 5 - Regeneration Summary

**Date:** 2026-02-16

**Task:** Regenerate meta-model attempt 5 training notebook with API-based data fetching and hardcoded FRED API key.

---

## Changes Made

### 1. Data Fetching Strategy

**Previous (CSV-only):**
- Loaded pre-processed CSVs from Kaggle dataset
- Required dataset: `bigbigzabuton/gold-prediction-complete`
- Enable internet: `false`

**Current (API-based):**
- Fetches data directly from APIs (yfinance, FRED)
- Hardcoded FRED API key: `3ffb68facdf6321e180e380c00e909c8`
- Loads submodel outputs from local filesystem
- Enable internet: `true`

### 2. API Data Sources

**yfinance (no authentication):**
- Gold (GC=F) - target variable
- DXY (DX-Y.NYB) - dollar index

**FRED API (hardcoded key):**
- DFII10 - real interest rate
- VIXCLS - VIX
- DGS10, DGS2 - yield curve
- T10YIE - inflation expectation

**Local CSVs (submodel outputs):**
- data/submodel_outputs/vix.csv
- data/submodel_outputs/technical.csv
- data/submodel_outputs/cross_asset.csv
- data/submodel_outputs/yield_curve.csv
- data/submodel_outputs/etf_flow.csv
- data/submodel_outputs/inflation_expectation.csv
- data/submodel_outputs/options_market.csv

### 3. kernel-metadata.json

**Changed settings:**
```json
{
  "id": "bigbigzabuton/gold-model-training",
  "enable_internet": true,
  "dataset_sources": []
}
```

**Removed:**
- Dataset dependency: `bigbigzabuton/gold-prediction-complete`

---

## Implementation Details

### Data Fetching Cell

```python
# === FRED API key (hardcoded) ===
FRED_API_KEY = "3ffb68facdf6321e180e380c00e909c8"
fred = Fred(api_key=FRED_API_KEY)

# === Fetch from APIs ===
gold = yf.download('GC=F', start='2014-01-01', end='2026-02-20', progress=False)
real_rate = fred.get_series('DFII10', observation_start='2014-01-01')
dxy = yf.download('DX-Y.NYB', start='2014-01-01', end='2026-02-20', progress=False)
vix = fred.get_series('VIXCLS', observation_start='2014-01-01')
dgs10 = fred.get_series('DGS10', observation_start='2014-01-01')
dgs2 = fred.get_series('DGS2', observation_start='2014-01-01')
infl_exp = fred.get_series('T10YIE', observation_start='2014-01-01')

# === Load submodel CSVs from local files ===
for feature in ['vix', 'technical', 'cross_asset', 'yield_curve', 'etf_flow', 'inflation_expectation', 'options_market']:
    df = pd.read_csv(f'data/submodel_outputs/{feature}.csv')
    # ... (date normalization)
```

### Critical Date Normalization

**Timezone-aware CSVs (require `utc=True`):**
- technical.csv
- options_market.csv

```python
df['Date'] = pd.to_datetime(df['date'], utc=True).dt.strftime('%Y-%m-%d')
```

---

## Model Specifications (Unchanged)

- **Architecture:** Single XGBoost with reg:squarederror
- **Features:** 23 (5 base + 18 submodel outputs)
- **Optuna trials:** 100
- **Optuna weights:** HCDA 20%, DA 30%, MAE 10%, Sharpe 40%
- **Confidence threshold:** Fixed at 80th percentile
- **Fallback:** Attempt 2 best params on 23 features

---

## Files Updated

1. **notebooks/meta_model_5/train.ipynb** - Complete regeneration with API-based fetching
2. **notebooks/meta_model_5/kernel-metadata.json** - Updated to enable internet, remove dataset dependency

---

## Verification Checklist

- [x] FRED API key hardcoded (no Kaggle Secrets)
- [x] yfinance API calls included
- [x] Submodel CSVs loaded from local filesystem
- [x] Timezone-aware date normalization (technical.csv, options_market.csv)
- [x] enable_internet: true
- [x] dataset_sources: [] (empty)
- [x] All 23 features defined
- [x] Optuna weights: 40/30/10/20
- [x] Fixed confidence threshold (80th percentile)

---

## Next Steps

1. Verify local submodel CSVs exist at specified paths
2. Submit notebook to Kaggle via `kaggle kernels push -p notebooks/meta_model_5/`
3. Monitor training execution on Kaggle
4. Fetch results when complete

---

**Status:** Ready for Kaggle submission
