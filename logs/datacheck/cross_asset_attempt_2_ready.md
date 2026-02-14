# Cross-Asset Features - Attempt 2 - Ready for Datachecker

## Generation Summary

**Timestamp**: 2026-02-15T00:08:25
**Status**: Ready for datachecker validation
**Output File**: `data/processed/cross_asset_features.csv`
**Metadata File**: `data/processed/cross_asset_metadata.json`

## Critical Changes from Attempt 1

Attempt 1 was **REJECTED** by datachecker because it provided **raw price levels** with autocorrelations > 0.99:

- `gold_close` autocorr = 0.9994 (REJECTED)
- `silver_close` autocorr = 0.9953 (REJECTED)
- `copper_close` autocorr = 0.9978 (REJECTED)
- `gsr` (gold/silver ratio) autocorr = 0.9920 (REJECTED)
- `gcr` (gold/copper ratio) autocorr = 0.9962 (REJECTED)

**Attempt 2 provides PROCESSED FEATURES ONLY**:

1. **xasset_regime_prob** - HMM posterior probability (autocorr = 0.859)
2. **xasset_recession_signal** - First difference of gold/copper z-score (autocorr = -0.031)
3. **xasset_divergence** - Daily gold-silver return difference z-score (autocorr = -0.011)

## Output Features

### 1. xasset_regime_prob

**Description**: HMM posterior probability of crisis/dislocation regime
**Method**: 3-state Gaussian HMM on [gold_ret, silver_ret, copper_ret], extract P(highest-variance state)
**Range**: [0, 1]
**Statistics**:
- Mean: 0.0244
- Std: 0.1295
- Autocorr: 0.859 (PASS)
- Min/Max: 1.36e-06 to 1.000

**Interpretation**: Higher values indicate cross-asset dislocation/crisis regime

### 2. xasset_recession_signal

**Description**: Daily change in gold/copper ratio z-score (recession signal velocity)
**Method**:
1. Compute gold/copper ratio
2. Compute 90-day rolling z-score
3. Take first difference (NOT raw z-score, which has autocorr 0.96)
4. Clip to [-4, 4]

**Range**: [-4, +4]
**Statistics**:
- Mean: -0.0014
- Std: 0.3946
- Autocorr: -0.031 (PASS)
- Min/Max: -1.50 to 2.90

**Interpretation**: Positive = recession fears intensifying, Negative = easing

### 3. xasset_divergence

**Description**: Daily gold-silver return difference z-score
**Method**:
1. Compute daily return difference (gold_ret - silver_ret)
2. Z-score against 20-day rolling std
3. Clip to [-4, 4]

**Range**: [-4, +4]
**Statistics**:
- Mean: 0.0015
- Std: 0.9797
- Autocorr: -0.011 (PASS)
- Min/Max: -3.36 to 3.62

**Interpretation**: >+2 = gold outperforming (safe-haven), <-2 = silver outperforming (momentum)

## Data Quality Metrics

- **Rows**: 2524 (2015-01-30 to 2025-02-12)
- **Missing Values**: 0 (all features complete)
- **Autocorrelation Check**: PASS (max = 0.859, well below 0.99 threshold)
- **Date Alignment**: Matches base_features date range
- **Feature Count**: 3 (compact, following VIX/technical success pattern)

## Autocorrelation Validation

| Feature | Autocorr (lag=1) | Threshold | Status |
|---------|------------------|-----------|--------|
| xasset_regime_prob | 0.859 | < 0.99 | ✅ PASS |
| xasset_recession_signal | -0.031 | < 0.99 | ✅ PASS |
| xasset_divergence | -0.011 | < 0.99 | ✅ PASS |

All features are **safely below the 0.99 threshold** required for Gate 1 compliance.

## Expected Datachecker Results

### Step 1: Missing Values
- **Expected**: PASS (0 missing values)

### Step 2: Basic Stats
- **Expected**: PASS (all features have reasonable distributions)

### Step 3: Autocorrelation
- **Expected**: PASS (max autocorr = 0.859)
- **Critical**: No raw price levels or ratios with autocorr > 0.99

### Step 4: Future Leak
- **Expected**: PASS (all features are backward-looking)
- HMM fit on training set only
- Rolling windows inherently prevent lookahead

### Step 5: Temporal Alignment
- **Expected**: PASS (dates match base_features: 2015-01-30 to 2025-02-12)

### Step 6: VIF Correlation
- **Expected**: PASS (design doc shows max corr with existing features = 0.23)

### Step 7: Schema Validation
- **Expected**: PASS (3 columns, 2524 rows, DatetimeIndex)

## Design Compliance

This implementation follows the design document (docs/design/cross_asset_attempt_1.md) with critical autocorrelation corrections:

1. ✅ 3 output features (not raw data)
2. ✅ HMM on [gold, silver, copper] returns (3D, excludes S&P 500)
3. ✅ First difference of gold/copper z-score (NOT raw z-score)
4. ✅ Daily return difference z-score (NOT multi-day pct_change)
5. ✅ All autocorrelations < 0.99
6. ✅ Daily frequency (no interpolation)
7. ✅ Returns-based (handles futures roll artifacts)

## Files Generated

1. **src/fetch_cross_asset.py** - Reusable data fetching function
2. **data/processed/cross_asset_features.csv** - Output features (2524 rows × 3 columns)
3. **data/processed/cross_asset_metadata.json** - Feature metadata
4. **logs/datacheck/cross_asset_attempt_2_generation_summary.json** - Generation log

## Next Step

**Ready for datachecker validation** (builder_data role complete)

Expected outcome: **PASS** (all 7 steps)
