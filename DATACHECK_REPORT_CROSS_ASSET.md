# Data Quality Check Report
## Cross-Asset Feature - Attempt 1

**Status:** REJECT
**Timestamp:** 2026-02-15 00:03:22
**Feature:** cross_asset
**Attempt:** 1 of 5

---

## Test Summary

| Step | Name | Result | Issues | Notes |
|------|------|--------|--------|-------|
| 1 | Missing Values | PASS | 0 | Zero NaN in all columns |
| 2 | Basic Statistics | PASS | 0 | Row count 2,860 acceptable |
| 3 | Autocorrelation | **REJECT** | **5 CRITICAL** | Price levels exceed 0.99 threshold |
| 4 | Future Leak | PASS | 1 warning | No critical leaks |
| 5 | Temporal Alignment | PASS | 0 | Dates properly sorted, no gaps |
| 6 | VIF Pre-Check | PASS | 10 warnings | Expected correlations with base features |
| 7 | Schema Validation | PASS | 0 | All expected columns present |

**Result: 6/7 steps passed | 5 CRITICAL failures in Step 3**

---

## Critical Issues

### Step 3: Autocorrelation Check - GATE 1 FAILURE

Five columns have lag-1 autocorrelation exceeding the 0.99 threshold:

```
gold_close:   0.9994  CRITICAL (price level)
copper_close: 0.9978  CRITICAL (price level)
silver_close: 0.9952  CRITICAL (price level)
gsr:          0.9920  CRITICAL (ratio level)
gcr:          0.9962  CRITICAL (ratio level)
```

**Expected (per design spec):**
- xasset_regime_prob: autocorr 0.60-0.83 ✓
- xasset_recession_signal: autocorr ~-0.04 ✓
- xasset_divergence: autocorr ~0.03 ✓

**Received:**
- Price levels (not submodel outputs)
- Ratio levels (not first-differences/z-scores)
- Returns (correct form, but insufficient)

---

## Root Cause Analysis

The builder_data agent provided **raw data** instead of **processed submodel features**.

According to the design document (cross_asset_attempt_1.md):

### What Should Have Been Provided

#### Feature 1: Cross-Asset HMM Regime Probability
- **Source:** 3D HMM on [gold_return, silver_return, copper_return]
- **Output:** Posterior probability of high-variance (crisis) state
- **Autocorr:** Expected 0.83 (state persistence)
- **Status:** NOT PROVIDED ✗

#### Feature 2: Gold/Copper Recession Signal (Velocity)
- **Formula:** First difference of gold/copper z-score (90d rolling window)
- **Code:** `z_change = z.diff()` (not z level!)
- **Autocorr:** Expected -0.039 (near white noise)
- **Design note:** "Raw z-score has autocorr 0.9587. Use FIRST DIFFERENCE."
- **Status:** Provided as `gsr` (raw ratio) with autocorr 0.9920 ✗

#### Feature 3: Daily Gold-Silver Divergence Z-Score
- **Formula:** Daily return difference z-scored against 20d rolling std
- **Code:** Daily returns (not multi-day), then z-score
- **Autocorr:** Expected ~0.03 (near white noise)
- **Design note:** "NOT multi-day pct_change(20) which has autocorr 0.91"
- **Status:** NOT PROVIDED ✗

### What Was Provided

Instead, builder_data delivered 8 columns:

| Column | Type | Autocorr | Issue |
|--------|------|----------|-------|
| gold_close | Price level | 0.9994 | Should not be included |
| copper_close | Price level | 0.9978 | Should not be included |
| silver_close | Price level | 0.9952 | Should not be included |
| gold_return | Daily return | -0.0292 | Correct form but unused |
| silver_return | Daily return | -0.0608 | Correct form but unused |
| copper_return | Daily return | -0.0490 | Correct form but unused |
| gsr | Ratio level | 0.9920 | Should be first-diff |
| gcr | Ratio level | 0.9962 | Should be first-diff |

---

## Impact Assessment

### Why This Fails Gate 1

**Gate 1 Requirement:** Standalone quality check for model outputs
- Autocorrelation < 0.99 (to avoid trivial patterns)
- No constant output
- No NaN/all-NaN issues

**Why High Autocorrelation Is Fatal:**
1. Price levels inherently persist (today ≈ yesterday)
2. XGBoost learns: "gold_close today is highly predictive of gold_close tomorrow"
3. This **is trivial** and adds **zero information** about returns
4. Violates the fundamental assumption that features must have signal

### Implications for Meta-Model

If this data were passed to builder_model:
- The meta-model would include price levels
- These would dominate feature importance (trivial predictions)
- XGBoost could not learn actual return prediction logic
- Gate 3 would fail (worse than baseline)

---

## What Needs to Happen

### Builder_Data Requirements for Attempt 2

The builder_data agent must generate exactly 3 columns:

```python
output = pd.DataFrame({
    'date': trading_dates,
    'xasset_regime_prob': regime_prob,          # HMM posterior (0-1)
    'xasset_recession_signal': recession_signal, # z-score first-diff (-4 to +4)
    'xasset_divergence': divergence_zscore       # return diff z-score (-4 to +4)
})
```

**Verification Checklist:**
1. [ ] xasset_regime_prob autocorr < 0.85 (expected 0.60-0.83)
2. [ ] xasset_recession_signal autocorr < 0.5 (expected ~-0.04)
3. [ ] xasset_divergence autocorr < 0.5 (expected ~0.03)
4. [ ] No NaN values (or only first ~90 rows)
5. [ ] Date alignment with base_features.csv
6. [ ] All values within expected ranges (probabilities 0-1, z-scores -4 to +4)
7. [ ] Correlation with base features < 0.7 (VIF check)

---

## Historical Context

### Similar Issue: real_rate Feature

The real_rate submodel also struggled with Gate 1. The key difference:
- **real_rate:** Monthly data, tried interpolation → autocorr problems
- **cross_asset:** Daily data, simpler HMM + deterministic features
- **Solution:** Return to designed submodel outputs (HMM + first-differences)

### Successful Precedents

Two submodels already passed:
1. **VIX (Attempt 1):** HMM-based, 3 features, passed Gate 1 and Gate 3
2. **Technical (Attempt 1):** HMM-based, 3 features, passed Gate 1 and Gate 3

Both followed the same pattern:
- HMM on returns (autocorr 0.6-0.83) ✓
- Deterministic statistics (z-scores, differences) ✓
- Daily frequency (no interpolation) ✓

**Cross-asset should follow this proven pattern.**

---

## Data Specification (Design Reference)

From **docs/design/cross_asset_attempt_1.md**, Section 3:

### Component 1: HMM Regime Detection
```
Input: 3D [gold_daily_return, silver_daily_return, copper_daily_return]
Model: GaussianHMM with 2-3 states
Output: Posterior probability of highest-variance state
Autocorr expected: 0.60-0.83
```

### Component 2: Recession Signal (CORRECTED per fact-check)
```
Input: gold/copper ratio = GC=F_close / HG=F_close
Step 1: z-score with 90d rolling window
Step 2: TAKE FIRST DIFFERENCE (daily change in z-score)
Clip to [-4, 4]
Autocorr expected: -0.039 (near white noise, safe)

Design note (Section 1, fact-check):
"Raw z-score (90d) has autocorr 0.9587 - DANGEROUS.
Use FIRST DIFFERENCE instead, which has autocorr -0.039."
```

### Component 3: Daily Divergence Z-Score (CORRECTED per fact-check)
```
Input: gold_daily_return - silver_daily_return
Step 1: Compute rolling mean (20d window)
Step 2: Compute rolling std (20d window)
Step 3: z = (daily_diff - rolling_mean) / rolling_std
Clip to [-4, 4]
Autocorr expected: ~0.03 (near white noise)

Design note (Section 1, fact-check):
"Multi-day pct_change(20) has autocorr 0.91 - DANGEROUS.
Use DAILY return difference instead, autocorr ~0.03."
```

---

## Files

**Datacheck Report:**
`C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents\logs\datacheck\cross_asset_attempt_1.json`

**Evaluation Summary:**
`C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents\logs\evaluation\cross_asset_datacheck_attempt_1.md`

**Design Document:**
`C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents\docs\design\cross_asset_attempt_1.md`

**Data File (Rejected):**
`C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents\data\raw\cross_asset.csv`

---

## Next Action

**REJECT - Return to builder_data for Attempt 2**

Current attempt count: 1/5
Consecutive rejections: 1 (not at 3-reject limit)
Resume from: builder_data

The issue is clear and actionable. Builder_data should focus on implementing the three submodel outputs as specified in the design doc.

---

## Confirmation Checklist for Re-Submission

Before resubmitting, verify:

- [ ] HMM fitted on training set only
- [ ] HMM.predict_proba() generates full dataset probabilities
- [ ] Recession signal is computed as FIRST DIFFERENCE of z-score (not level)
- [ ] Divergence uses DAILY return difference (not multi-day pct_change)
- [ ] All three features autocorr < 0.99 (prefer < 0.85)
- [ ] Output CSV has exactly 3 feature columns + date
- [ ] Dates match base_features.csv trading dates
- [ ] No forward-looking bias (HMM fit on train only)
- [ ] VIF pre-check passed (corr with existing features < 0.7)

Once all checks pass, run:
```bash
python scripts/datacheck_cross_asset.py
```

Expected output:
```
✓ OVERALL RESULT: PASS
Proceed to builder_model
```

---

**Report Generated:** 2026-02-15 00:03:22
**By:** datachecker (Haiku 4.5)
**Decision:** REJECT Attempt 1 → Proceed to Builder_Data Attempt 2
