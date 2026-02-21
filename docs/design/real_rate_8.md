# Real Rate Submodel - Attempt 8 Design Document
# Real Yield Curve Change Features (3-feature clean design)

## 0. Architect Fact-Check

Based on attempt 7 verified data sources - no new FRED series required.

- FRED:DFII5, DFII7, DFII10, DFII20, DFII30: all confirmed daily series, verified in attempt 7
- All start 2015-01-02, all have full daily coverage through 2025-02-12
- No additional series needed for attempt 8 (deterministic transformation only)

**Root Cause Analysis from Attempt 7:**
- Gate 3 PASS via Sharpe (+0.329, 6.6x threshold, 3/4 folds)
- Gate 3 FAIL on DA (-0.00149pp) and MAE (-0.00221%)
- Primary culprit: `rr_slope_level_z` has autocorr=0.937 (near-unit-root)
  - Level features introduce slow-moving regime signals that XGBoost can overfit in some folds
  - Fold 3 showed DA degradation of -2.84pp (worst fold), consistent with level-feature regime mismatch
- Resolution: Remove `rr_slope_level_z`, use 3 pure change features

---

## Design Overview

| Attribute | Value |
|-----------|-------|
| Model Type | Deterministic transformation (no neural network) |
| Data Sources | FRED DFII5, DFII10, DFII30 (daily) |
| Output Columns | 3 features |
| Training Required | No (pure computation) |
| Kaggle GPU | false |
| Estimated Runtime | < 2 minutes |
| Change from Attempt 7 | Remove rr_slope_level_z (autocorr=0.937). 4 -> 3 features. |

---

## Feature Specification

### Feature 1: rr_level_change_z (retained from attempt 7)
- **Formula**: `rolling_zscore(DFII10.diff(), window=30)`
- **Interpretation**: Standardized daily change in 10Y real yield
- **Gold correlation**: -0.1097 (stronger negative: rising rates hurt gold)
- **Autocorrelation**: 0.032 (near-iid, minimal persistence)
- **Rationale**: Cleanest signal. No modification.

### Feature 2: rr_slope_chg_z (retained from attempt 7)
- **Formula**: `rolling_zscore((DFII30 - DFII5).diff(), window=60)`
- **Interpretation**: Standardized daily change in slope (steepening/flattening velocity)
- **Gold correlation**: +0.0643
- **Autocorrelation**: -0.042 (near-iid)
- **Rationale**: Orthogonal to level change. No modification.

### Feature 3: rr_curvature_chg_z (retained from attempt 7)
- **Formula**: `rolling_zscore((2*DFII10 - DFII5 - DFII30).diff(), window=60)`
- **Interpretation**: Standardized daily change in curvature (belly distortion)
- **Gold correlation**: +0.0121
- **Autocorrelation**: -0.198 (slight mean-reversion, as expected for 2nd differences)
- **Rationale**: Independent information from level and slope (corr < 0.35 with both). No modification.

### REMOVED: rr_slope_level_z
- **Reason**: autocorr=0.937 introduces a near-unit-root regime signal
- **Effect**: In 3/4 folds Sharpe improved (+0.329 avg), but DA was hurt (-0.00149pp avg)
  because high-persistence level features encourage XGBoost to learn spurious regime effects
- **Evidence**: Fold 3 DA delta = -2.84pp (worst), consistent with level-feature overfitting in a specific sub-period
- **Decision**: Remove permanently. Pure change features are cleaner for short-horizon forecasting.

---

## Feature Properties (expected)

| Column | Autocorr | VIF | Gold Corr |
|--------|----------|-----|-----------|
| rr_level_change_z | 0.032 | 1.174 | -0.110 |
| rr_slope_chg_z | -0.042 | 1.100 | +0.064 |
| rr_curvature_chg_z | -0.198 | 1.209 | +0.012 |

All autocorrelations < 0.25 (well below 0.95 threshold)
Max VIF = 1.21 (well below 10 threshold)
Cross-correlations: max 0.326 (level vs curvature) - all below 0.35

---

## Data Pipeline

```python
# Step 1: Fetch DFII series from FRED (Kaggle Secrets)
from kaggle_secrets import UserSecretsClient
try:
    FRED_API_KEY = UserSecretsClient().get_secret("FRED_API_KEY")
except Exception:
    FRED_API_KEY = os.environ.get('FRED_API_KEY')

from fredapi import Fred
fred = Fred(api_key=FRED_API_KEY)

series = {
    'DFII5':  fred.get_series('DFII5',  observation_start='2015-01-01', observation_end='2025-02-12'),
    'DFII10': fred.get_series('DFII10', observation_start='2015-01-01', observation_end='2025-02-12'),
    'DFII30': fred.get_series('DFII30', observation_start='2015-01-01', observation_end='2025-02-12'),
}
df = pd.DataFrame(series).dropna(how='all')

# Step 2: Compute intermediate series
df['slope']     = df['DFII30'] - df['DFII5']
df['curvature'] = 2*df['DFII10'] - df['DFII5'] - df['DFII30']

# Step 3: Rolling z-score function
def rolling_zscore(series, window):
    mu    = series.rolling(window, min_periods=window//2).mean()
    sigma = series.rolling(window, min_periods=window//2).std()
    sigma = sigma.replace(0, np.nan)
    return ((series - mu) / sigma).clip(-4, 4)

# Step 4: Compute 3 features (no rr_slope_level_z)
features = pd.DataFrame(index=df.index)
features['rr_level_change_z']    = rolling_zscore(df['DFII10'].diff(), 30)
features['rr_slope_chg_z']       = rolling_zscore(df['slope'].diff(), 60)
features['rr_curvature_chg_z']   = rolling_zscore(df['curvature'].diff(), 60)

# Step 5: Load target.csv to get exact business day index
target = pd.read_csv('/kaggle/input/gold-prediction-submodels/target.csv', index_col=0)
target.index = pd.to_datetime(target.index)
features = features.reindex(target.index)

# Step 6: Save
features.to_csv('/kaggle/working/submodel_output.csv')
```

---

## Hyperparameter Search Space

No model training. All features are deterministic. Windows are fixed:
- `level_window = 30` (same as attempt 7)
- `slope_window = 60` (same as attempt 7)
- `curvature_window = 60` (same as attempt 7)

No Optuna required.

---

## Kaggle Notebook Design

The notebook:
1. Fetches DFII5, DFII10, DFII30 using Kaggle Secrets FRED_API_KEY
2. Computes 3 features deterministically (same windows as attempt 7, but removes rr_slope_level_z)
3. Aligns to target.csv date index
4. Saves submodel_output.csv (3 columns)
5. Runs Gate 1/2/3 validation in-notebook
6. Saves training_result.json with all gate results

---

## Gate Expectations

### Gate 1
- Overfit ratio: 1.0 (no model, no overfitting)
- All columns non-NaN, non-constant: YES (same 3 features as attempt 7, all passed)
- Expected: PASS

### Gate 2
- MI increase: Removing rr_slope_level_z may reduce MI slightly (it had MI=0.526)
  - Attempt 7 proper MI: +10.92% with 4 features
  - 3-feature estimate: approximately +8-9% (still above 5% threshold)
- VIF: max 1.21 (unchanged, all below 10)
- Stability: max std 0.087 (unchanged, all below 0.15)
- Expected: PASS

### Gate 3
- DA delta: removing high-autocorr level feature should reduce noise in DA signal
  - Attempt 7 DA delta was -0.00149 (nearly flat). Removing level noise should push this positive.
  - Target: > 0.005 (+0.5pp)
- Sharpe delta: attempt 7 achieved +0.329 with 4 features; 3 features may reduce Sharpe contribution slightly
  - Minimum acceptable: maintain > 0.05 (6.6x improvement room)
- MAE delta: attempt 7 was -0.00221 (not enough). May slightly improve with cleaner features.
- Expected: PASS on Sharpe (robust), possibly PASS on DA

---

## Kernel Configuration

```json
{
  "id": "bigbigzabuton/gold-real-rate-8",
  "title": "Gold Real Rate Model - Attempt 8",
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

---

## Risk and Alternatives

1. If Gate 3 DA still fails with 3 features: the change features are genuinely not directionally informative; accept attempt 7 as the real_rate submodel output.
2. If Gate 2 MI drops below 5%: the 4th feature (rr_slope_level_z) was carrying too much MI. Not a design problem - the 3 change features still provide genuine information.
3. If Sharpe degrades significantly: re-add rr_slope_level_z and accept the DA compromise.

**Note**: This is attempt 8 per user request. Attempt 9 will auto-run via auto_resume. The evaluation history confirms attempt 7 already passes Gate 3, so attempt 8 is an improvement attempt.
