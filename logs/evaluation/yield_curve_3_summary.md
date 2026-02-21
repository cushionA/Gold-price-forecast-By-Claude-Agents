# Evaluation Summary: yield_curve attempt 3

## Gate 1: Standalone Quality -- FAIL
- All-NaN columns: 0 -- PASS
- Constant output columns: 0 -- PASS
- Autocorrelation (threshold < 0.95):
  - yc_curvature_z: -0.1750 -- PASS
  - yc_10y3m_velocity_z: 0.6190 -- PASS
  - yc_spread_level_z: 0.9937 -- FAIL (pure level variable, not a signal)
  - yc_vol_regime_z: 0.8615 -- PASS
- NaN counts: yc_curvature_z 10 (0.4%), yc_10y3m_velocity_z 12 (0.4%), yc_spread_level_z 177 (6.3%), yc_vol_regime_z 15 (0.5%)

## Gate 2: Information Gain -- PASS
- MI increase: +6.49% (threshold > 5%) -- PASS
  - MI base: 0.3569, MI extended: 0.3800
  - Note: Nearly all MI from yc_spread_level_z (0.0177), other 3 cols contribute ~0
- Max VIF: 1.24 (threshold < 10) -- PASS (excellent orthogonality)
- Correlation stability (max std): 0.1477 (threshold < 0.15) -- PASS

## Gate 3: Ablation (v2 replaced by v3) -- FAIL

| Metric | Baseline (v2) | Extended (v3) | Delta | Threshold | Result |
|--------|---------------|---------------|-------|-----------|--------|
| Direction Accuracy | 49.34% | 47.77% | -1.57pp | >+0.5pp | FAIL |
| Sharpe | -0.157 | -0.390 | -0.233 | >+0.05 | FAIL |
| MAE | 0.766% | 0.771% | +0.005 | <-0.01 | FAIL |

### Per-fold breakdown

| Fold | DA delta (pp) | Sharpe delta | MAE delta | v3 better? |
|------|---------------|--------------|-----------|------------|
| 1 | -1.90 | -0.673 | -0.003 | MAE only |
| 2 | -2.86 | -0.608 | +0.006 | None |
| 3 | +0.24 | +0.752 | +0.045 | DA+Sharpe only |
| 4 | -2.61 | -0.635 | -0.018 | MAE only |
| 5 | -0.71 | -0.002 | -0.006 | MAE only |

Folds where v3 improves: DA 1/5, Sharpe 1/5, MAE 3/5

### v3 feature importance (last fold)
- yc_vol_regime_z: rank 16/43, 2.40%
- yc_10y3m_velocity_z: rank 18/43, 2.35%
- yc_spread_level_z: rank 21/43, 2.27%
- yc_curvature_z: rank 26/43, 2.17%

## Decision: no_further_improvement

### Rationale

Attempt 3 FAILS all three gates:
1. **Gate 1 FAIL**: yc_spread_level_z has autocorrelation 0.9937, indicating it is a near-constant level variable (z-score of the yield spread level with a 356-day window). This is not a tradable signal but a slowly-moving state descriptor.
2. **Gate 2 PASS**: MI increase of 6.49%, VIF excellent (1.24), stability passes -- but the MI is almost entirely driven by yc_spread_level_z (the failing Gate 1 column).
3. **Gate 3 FAIL**: Replacing v2 with v3 degrades all three metrics. DA -1.57pp, Sharpe -0.233, MAE +0.005. v3 improves DA in only 1/5 folds and Sharpe in only 1/5 folds.

### Comparison with attempt 2 (current production)

| Metric | Attempt 2 Gate 3 delta | Attempt 3 Gate 3 delta | Better? |
|--------|----------------------|----------------------|---------|
| DA | -1.08pp (FAIL) | -1.57pp (FAIL) | Attempt 2 |
| Sharpe | -0.015 (FAIL) | -0.233 (FAIL) | Attempt 2 |
| MAE | -0.013 (PASS, 1.27x) | +0.005 (FAIL) | Attempt 2 |

Attempt 3 is strictly worse than attempt 2 on ALL Gate 3 metrics. Attempt 2 passed Gate 3 via MAE (-0.013); attempt 3 fails MAE entirely.

### Pattern analysis across 3 attempts

| Attempt | Gate 1 | Gate 2 | Gate 3 | Gate 3 best metric |
|---------|--------|--------|--------|-------------------|
| 1 | FAIL | FAIL | PASS (MAE -0.069) | MAE 6.9x threshold |
| 2 | PASS | PASS | PASS (MAE -0.013) | MAE 1.27x threshold |
| 3 | FAIL | PASS | FAIL | None |

The MAE improvement has degraded across attempts: -0.069 -> -0.013 -> +0.005 (no improvement). This is a clear downward trend. Additionally, attempt 3 introduced a high-autocorrelation level variable (yc_spread_level_z at 0.9937) that did not help and may have hurt performance.

### Recommendation

**Do NOT continue to attempt 4.** The yield curve submodel has been fully explored across 3 attempts with diminishing returns. Attempt 2 remains the best production output with all 3 gates passing. The max_attempt is 4, but further iteration is not justified because:

1. Attempt 3 is strictly worse than attempt 2 on every metric
2. The MAE improvement trend is clearly deteriorating (6.9x -> 1.27x -> negative)
3. The new features (yc_spread_level_z, yc_vol_regime_z) added no value
4. yc_curvature_z was already in v2, so v3 only changed 3 of 4 columns and all changes were detrimental

Yield curve attempt 2 output should remain as the production submodel.
