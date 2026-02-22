# Evaluation Summary: yield_curve attempt 2

## Gate 1: Standalone Quality -- PASS

- Overfit ratio: N/A (deterministic feature engineering, no train/val loss) -- skipped
- All-NaN columns: 0 -- PASS
- Constant output columns: 0 -- PASS
- Column statistics (all z-scores, properly normalized):
  - yc_curvature_z: mean=0.003, std=0.986, range=[-3.28, 4.00], count=2739 (97.9%)
  - yc_spread_velocity_z: mean=0.016, std=1.067, range=[-3.84, 4.00], count=2765 (98.8%)
  - yc_10y3m_velocity_z: mean=-0.009, std=1.071, range=[-3.29, 3.65], count=2765 (98.8%)
  - yc_dgs3mo_velocity_z: mean=-0.006, std=1.100, range=[-4.00, 4.00], count=2765 (98.8%)
- Autocorrelation (lag=1):
  - yc_curvature_z: -0.165 -- PASS
  - yc_spread_velocity_z: 0.738 -- PASS
  - yc_10y3m_velocity_z: 0.756 -- PASS
  - yc_dgs3mo_velocity_z: 0.701 -- PASS
- Optuna trials: 50 -- PASS

**All standalone quality checks pass.** No constant columns (unlike attempt 1 where yc_regime_prob was constant). All 4 columns are properly normalized z-scores with reasonable autocorrelation.

## Gate 2: Information Gain -- PASS (all 3 criteria)

- MI increase: +15.17% (threshold > 5%) -- PASS
  - MI base sum: 0.3008
  - MI extended sum: 0.3465
  - MI individual: yc_curvature_z=0.004, yc_spread_velocity_z=0.000, yc_10y3m_velocity_z=0.036, yc_dgs3mo_velocity_z=0.000
- Max VIF (submodel): 1.74 (threshold < 10) -- PASS
  - yc_curvature_z: VIF=1.06
  - yc_spread_velocity_z: VIF=1.64
  - yc_10y3m_velocity_z: VIF=1.74
  - yc_dgs3mo_velocity_z: VIF=1.07
- Correlation stability:
  - yc_curvature_z: std=0.149 -- PASS
  - yc_spread_velocity_z: std=0.140 -- PASS
  - yc_10y3m_velocity_z: std=0.125 -- PASS
  - yc_dgs3mo_velocity_z: std=0.147 -- PASS
  - Max stability std: 0.149 (threshold < 0.15) -- PASS

**Major improvement over attempt 1**: MI increase +15.17% vs +0.37%. Attempt 1 failed Gate 2 entirely; attempt 2 passes all 3 Gate 2 criteria. VIF values are excellent (max 1.74), confirming near-orthogonality to existing features.

## Gate 3: Ablation -- PASS (via MAE, marginal)

Note: This is a REPLACEMENT test. Baseline includes yield_curve v1 (3 cols: yc_regime_prob, yc_spread_velocity_z, yc_curvature_z). Extended replaces v1 with v2 (4 cols: yc_curvature_z, yc_spread_velocity_z, yc_10y3m_velocity_z, yc_dgs3mo_velocity_z).

Samples: 2,357 (common index after dropna). Features: baseline=43, extended=44.

| Metric | Baseline (with v1) | Extended (with v2) | Delta | Threshold | Result |
|--------|-------------------|--------------------|-------|-----------|--------|
| Direction Accuracy | 52.65% | 51.56% | -1.08% | > +0.50% | FAIL |
| Sharpe | 0.6455 | 0.6304 | -0.015 | > +0.05 | FAIL |
| MAE | 0.9333 | 0.9206 | -0.0127 | < -0.01 | PASS (1.27x) |

### Fold Consistency

| Fold | Train | Test | DA delta | MAE delta | Sharpe delta |
|------|-------|------|----------|-----------|--------------|
| 1 | 397 | 392 | -1.28% | +0.1602 | -1.134 |
| 2 | 789 | 392 | -3.62% | +0.0375 | -0.171 |
| 3 | 1181 | 392 | -2.05% | -0.1419 | -0.118 |
| 4 | 1573 | 392 | -3.32% | +0.1108 | -0.809 |
| 5 | 1965 | 392 | +4.85% | -0.2304 | +2.156 |

- DA improved: 1/5 folds
- MAE improved: 2/5 folds
- Sharpe improved: 1/5 folds

### Feature Importance (last fold)

| Feature | Importance | Rank |
|---------|-----------|------|
| yc_10y3m_velocity_z (NEW) | 0.0361 | 6/44 |
| yc_spread_velocity_z | 0.0289 | 9/44 |
| yc_curvature_z | 0.0220 | 20/44 |
| yc_dgs3mo_velocity_z (NEW) | 0.0159 | 38/44 |

Comparison with v1 baseline feature importance (last fold):
- yc_spread_velocity_z: rank 4/43 (baseline) vs rank 9/44 (extended)
- yc_curvature_z: rank 37/43 (baseline) vs rank 20/44 (extended)
- yc_regime_prob: rank 33/43 (baseline, effectively useless constant)

## Decision: completed (Gate 3 PASS, but attempt 1 remains stronger on MAE)

### Gate 3 Pass Rationale
MAE delta of -0.0127 exceeds the -0.01 threshold (1.27x). While this is a marginal pass, it meets the formal criterion.

### Comparison with Attempt 1

| Metric | Attempt 1 | Attempt 2 | Better |
|--------|-----------|-----------|--------|
| Gate 1 | FAIL (constant col) | PASS | Attempt 2 |
| Gate 2 MI | +0.37% (FAIL) | +15.17% (PASS) | Attempt 2 |
| Gate 2 VIF | 1.17 | 1.74 | Attempt 1 (both PASS) |
| Gate 2 Stability | 0.132 | 0.149 | Attempt 1 (both PASS) |
| Gate 3 DA delta | +0.20% | -1.08% | Attempt 1 |
| Gate 3 Sharpe delta | -0.089 | -0.015 | Attempt 2 |
| Gate 3 MAE delta | -0.0693 (6.9x) | -0.0127 (1.27x) | Attempt 1 |
| Gate 3 MAE consistency | 4/5 folds | 2/5 folds | Attempt 1 |

### Assessment

Attempt 2 is a clear improvement in information quality (Gate 1 all PASS, Gate 2 all PASS vs both FAIL in attempt 1). The MI increase of +15.17% confirms the z-score features contain genuine information about gold returns.

However, for Gate 3 ablation, attempt 2 is weaker than attempt 1 in the key MAE metric (1.27x vs 6.9x threshold, 2/5 vs 4/5 folds). The yc_10y3m_velocity_z feature is valuable (rank 6/44), but the overall replacement does not yield a stronger Gate 3 result than keeping the v1 output.

**Key insight**: Fold 5 (most recent data, 2023-2025) shows dramatic improvement with v2 (DA +4.85%, MAE -0.23, Sharpe +2.16), while folds 1-4 all show DA degradation. This suggests v2's extended yield curve features (10Y-3M and DGS3MO velocity) capture structural changes in recent interest rate regimes that v1 misses. This recent-period advantage may be valuable for forward-looking predictions.

### Recommendation

Gate 3 PASS is formal. Both attempt 1 and attempt 2 pass Gate 3. Given this is a re-improvement run:

1. **For the automation pipeline**: Mark as completed (Gate 3 PASS).
2. **For production dataset update**: The decision of whether to replace v1 with v2 in the Kaggle dataset should consider:
   - v2 has more robust information content (Gate 2: +15.17% MI)
   - v2 has no collapsed columns (v1's regime_prob was useless)
   - v2's yc_10y3m_velocity_z is highly ranked (6th) -- a genuinely new signal
   - v2 performs better on the most recent fold
   - However, v1 has stronger aggregate MAE improvement and better fold consistency
3. **Note**: The meta-model (attempt 7) was trained with v1. Changing to v2 would require meta-model retraining.
