# Real Rate Submodel - Attempt 9 Design Document
# Real Yield Curve Shape + Interaction Feature (5-feature design)

## 0. Architect Fact-Check

Based on attempt 7/8 verified data sources - no new FRED series required.

- FRED:DFII5, DFII10, DFII30: confirmed daily series, available 2015-01-02 through present
- All series have full daily business day coverage
- No additional series, no new API keys needed
- Attempt 9 is the final attempt for real_rate (attempt count 9/5+ user-authorized extension)

**Root Cause Analysis from Attempts 7 and 8:**

| Metric | Attempt 7 (4 feat, incl rr_slope_level_z) | Attempt 8 (3 feat, excl rr_slope_level_z) |
|--------|---------------------------------------------|---------------------------------------------|
| DA delta | -0.00149pp (FAIL, near zero) | -1.471pp (worse) |
| Sharpe delta | +0.329 (PASS, 6.6x) | -0.260 (reversed) |
| MAE delta | -0.0022% (FAIL, small) | -0.0203% (PASS, 2.03x) |

Key finding: rr_slope_level_z (autocorr=0.937) is the source of Sharpe gain in attempt 7.
Its removal in attempt 8 traded Sharpe for MAE improvement.

**Attempt 9 Goal**: Retain all 4 attempt-7 features AND add a 5th feature that is orthogonal
to all existing features, low-autocorr, and provides directional information to push DA positive.

**Approach Selected**: Slope-curvature interaction (product of daily changes) as 5th feature.

- interaction = rolling_zscore(slope.diff() * curvature.diff(), window=60)
- Autocorr: near 0 (product of two near-iid series)
- Captures: simultaneous slope AND curvature stress events (butterfly distortions)
- Does NOT add another high-autocorr feature (avoids attempt 8's MAE degradation)

---

## Design Overview

| Attribute | Value |
|-----------|-------|
| Model Type | Deterministic transformation (no neural network, no Optuna) |
| Data Sources | FRED DFII5, DFII10, DFII30 (daily) |
| Output Columns | 5 features |
| Training Required | No (pure computation) |
| Kaggle GPU | false |
| Enable Internet | true (for FRED API) |
| Estimated Runtime | < 2 minutes |
| Change from Attempt 7 | Retain all 4 features + add rr_slope_curvature_interaction_z as 5th |
| Change from Attempt 8 | Re-add rr_slope_level_z (for Sharpe) + add interaction feature |

---

## Feature Specification

### Feature 1: rr_level_change_z (retained from attempts 7 and 8)
- **Formula**: `rolling_zscore(DFII10.diff(), window=30)`
- **Interpretation**: Standardized daily change in 10Y real yield
- **Gold correlation**: -0.1097
- **Autocorrelation**: 0.032 (near-iid)
- **Rationale**: Cleanest signal. No modification.

### Feature 2: rr_slope_chg_z (retained from attempts 7 and 8)
- **Formula**: `rolling_zscore((DFII30 - DFII5).diff(), window=60)`
- **Interpretation**: Standardized daily change in slope (steepening/flattening velocity)
- **Gold correlation**: +0.0643
- **Autocorrelation**: -0.042 (near-iid)
- **Rationale**: Orthogonal to level change. No modification.

### Feature 3: rr_curvature_chg_z (retained from attempts 7 and 8)
- **Formula**: `rolling_zscore((2*DFII10 - DFII5 - DFII30).diff(), window=60)`
- **Interpretation**: Standardized daily change in curvature
- **Gold correlation**: +0.0121
- **Autocorrelation**: -0.198 (slight mean-reversion)
- **Rationale**: Independent information from level and slope. No modification.

### Feature 4: rr_slope_level_z (retained from attempt 7, removed in attempt 8)
- **Formula**: `rolling_zscore((DFII30 - DFII5), window=60)`
- **Interpretation**: Regime indicator - current real yield curve steepness level
- **Gold correlation**: +0.0502
- **Autocorrelation**: 0.937 (high persistence - regime indicator)
- **Rationale**: Empirically demonstrated to improve Sharpe by +0.329 in attempt 7.
  Removal caused Sharpe to degrade to -0.260 in attempt 8. This feature must be retained.

### Feature 5: rr_slope_curvature_interaction_z (NEW for attempt 9)
- **Formula**: `rolling_zscore(slope.diff() * curvature.diff(), window=60)`
  - where `slope = DFII30 - DFII5` and `curvature = 2*DFII10 - DFII5 - DFII30`
- **Interpretation**: Captures days when both the slope AND curvature are simultaneously
  moving in unusual magnitudes. This represents butterfly distortion events in the TIPS
  market - when neither pure flattening nor pure steepening describes the rate movement.
- **Expected autocorrelation**: near 0 (product of two near-iid series: -0.042 * -0.198)
- **Expected VIF**: < 1.5 (orthogonal to all 4 existing features by construction)
- **Gold correlation**: uncertain, but novel - not captured by any individual feature
- **Rationale**:
  1. Low autocorr: will not degrade MAE like rr_slope_level_z does
  2. Nonlinear: captures joint stress that individual features cannot represent
  3. Orthogonal: product of orthogonal series produces an orthogonal series
  4. Directional: butterfly distortions have asymmetric gold implications depending on
     whether they represent curve compression (flattening + belly-out) vs curve convexity
     (steepening + belly-in)

---

## Feature Properties (expected)

| Column | Autocorr | VIF (expected) | Gold Corr (expected) |
|--------|----------|-----------------|----------------------|
| rr_level_change_z | 0.032 | ~1.18 | -0.110 |
| rr_slope_chg_z | -0.042 | ~1.14 | +0.064 |
| rr_curvature_chg_z | -0.198 | ~1.21 | +0.012 |
| rr_slope_level_z | 0.937 | ~1.04 | +0.050 |
| rr_slope_curvature_interaction_z | ~0.00 | ~1.10 | ~0.00-0.05 |

Max VIF expected: < 1.5 (well below 10 threshold)
All autocorrelations within design tolerance.

---

## Data Pipeline

```python
# Step 1: FRED API authentication (Kaggle Secrets)
from kaggle_secrets import UserSecretsClient
try:
    FRED_API_KEY = UserSecretsClient().get_secret("FRED_API_KEY")
except Exception:
    FRED_API_KEY = os.environ.get('FRED_API_KEY')

from fredapi import Fred
fred = Fred(api_key=FRED_API_KEY)

# Step 2: Fetch DFII series
series = {
    'DFII5':  fred.get_series('DFII5',  observation_start='2015-01-01'),
    'DFII10': fred.get_series('DFII10', observation_start='2015-01-01'),
    'DFII30': fred.get_series('DFII30', observation_start='2015-01-01'),
}
df = pd.DataFrame(series).dropna(how='all')

# Step 3: Compute intermediate series
df['slope']     = df['DFII30'] - df['DFII5']
df['curvature'] = 2*df['DFII10'] - df['DFII5'] - df['DFII30']

# Step 4: Rolling z-score helper
def rolling_zscore(series, window):
    mu    = series.rolling(window, min_periods=window//2).mean()
    sigma = series.rolling(window, min_periods=window//2).std()
    sigma = sigma.replace(0, np.nan)
    return ((series - mu) / sigma).clip(-4, 4)

# Step 5: Compute 5 features
features = pd.DataFrame(index=df.index)
features['rr_level_change_z']                = rolling_zscore(df['DFII10'].diff(), 30)
features['rr_slope_chg_z']                   = rolling_zscore(df['slope'].diff(), 60)
features['rr_curvature_chg_z']               = rolling_zscore(df['curvature'].diff(), 60)
features['rr_slope_level_z']                 = rolling_zscore(df['slope'], 60)
features['rr_slope_curvature_interaction_z'] = rolling_zscore(
    df['slope'].diff() * df['curvature'].diff(), 60
)

# Step 6: Align to target.csv date index
target = pd.read_csv('/kaggle/input/gold-prediction-submodels/target.csv', index_col=0)
target.index = pd.to_datetime(target.index)
features = features.reindex(target.index)

# Step 7: Save output
features.to_csv('/kaggle/working/submodel_output.csv')
```

---

## Hyperparameter Search Space

No model training. All features are deterministic. Windows are fixed:
- `level_window = 30` (same as attempts 7 and 8)
- `slope_window = 60` (same as attempts 7 and 8)
- `curvature_window = 60` (same as attempts 7 and 8)
- `slope_level_window = 60` (same as attempt 7)
- `interaction_window = 60` (same as slope and curvature windows for consistency)

No Optuna required.

---

## Gate Expectations

### Gate 1
- Overfit ratio: 1.0 (deterministic, no model)
- All 5 columns non-NaN: YES (same DFII series as attempts 7-8, all have full coverage)
- All 5 columns non-constant: YES (z-scored values have std ~1.0)
- Expected: PASS

### Gate 2
- MI increase: 5 features vs attempt 7's 4 features. All 4 attempt-7 features have MI ~0.51-0.56.
  The 5th feature (interaction) may contribute additional MI ~0.3-0.5.
  Estimate: +10-13% MI increase above base_features baseline.
- VIF: All 5 features have low cross-correlation. Expected max VIF < 1.5.
- Stability: 3 near-iid features have stability ~0.087 (attempt 7 local calc). rr_slope_level_z had 0.053 in attempt 7. Interaction feature expected similar. Overall max < 0.15.
- Expected: PASS

### Gate 3 Target
- Sharpe: Attempt 7's 4 features gave +0.329. Retaining those 4 + adding low-autocorr interaction should maintain Sharpe at least at attempt 7 levels. Target: > +0.05 (attempt 7 level preferred).
- DA: Attempt 7 was -0.00149pp (nearly zero). Adding a directional interaction signal should push this into positive territory. Target: > +0.005 (+0.5pp).
- MAE: Not the primary focus, but interaction feature (low autocorr) should not degrade MAE. Target: any improvement or neutral vs baseline.
- Expected: PASS on Sharpe (strong), uncertain on DA and MAE.

---

## Kernel Configuration

```json
{
  "id": "bigbigzabuton/gold-real-rate-9",
  "title": "Gold Real Rate Model - Attempt 9",
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

## Risk Assessment

1. If interaction feature has VIF > 5 with rr_slope_chg_z or rr_curvature_chg_z:
   - Clip to [-4, 4] should prevent extreme values; expected VIF < 1.5 by orthogonality argument
   - Gate 2 VIF threshold is 10 - very generous margin
2. If Sharpe degrades below attempt 7:
   - Indicates interaction feature is adding noise in the Sharpe-critical folds
   - Accept attempt 7 as best real_rate submodel if attempt 9 fails Gate 3
3. If DA still fails:
   - Both attempts 7 and 9 have the same 4 base features; the interaction may not add enough directional signal
   - But Sharpe pass is sufficient for Gate 3 - DA failure alone is not a dealbreaker

**This is attempt 9, the final authorized attempt for real_rate.**
If Gate 3 fails, accept attempt 7 (Sharpe +0.329) as the real_rate submodel.
