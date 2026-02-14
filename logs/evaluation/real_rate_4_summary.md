# Evaluation Summary: real_rate Attempt 4

**Method**: PCA + Cubic Spline (deterministic, <10s execution)
**Date**: 2026-02-14

---

## Gate 1: Standalone Quality -- N/A (Deterministic PCA)

PCA is deterministic with no train/val loss. No overfit ratio to compute.

- All-NaN columns: 0 [PASS]
- Zero-variance columns: 0 [PASS]
- Autocorrelation pc_0: 0.9967 [FLAGGED - expected for cubic spline on monthly data, not a leak]
- Autocorrelation pc_1: 0.9942 [FLAGGED - expected for cubic spline on monthly data, not a leak]

---

## Gate 2: Information Gain -- PASS

- MI increase: +10.29% (threshold > 5%) [PASS]
  - MI(base only) = 0.3095
  - MI(base + PCA) = 0.3414
  - MI(real_rate_pc_0) = 0.0313
  - MI(real_rate_pc_1) = 0.0026
- Max VIF: inf (real_rate_real_rate, pre-existing base feature issue) [PASS with note]
  - VIF(real_rate_pc_0) = 1.33 -- well below threshold
  - VIF(real_rate_pc_1) = 1.17 -- well below threshold
  - Note: The inf VIF is in the base features (real_rate vs yield_curve collinearity), NOT caused by the PCA submodel output. The submodel adds no multicollinearity.
- Rolling correlation stability (std): 0.1347 (threshold < 0.15) [PASS]

---

## Gate 3: Ablation -- FAIL

| Metric | Baseline | Extended | Delta | Threshold | Result |
|--------|----------|----------|-------|-----------|--------|
| Direction Accuracy | 49.60% | 47.64% | -1.96% | > +0.5% | FAIL |
| Sharpe Ratio | 0.274 | 0.055 | -0.219 | > +0.05 | FAIL |
| MAE | 0.881% | 0.959% | +0.078% | < -0.01% | FAIL |

**All 3 criteria failed. All 5 folds showed degradation.**

### Per-Fold Breakdown

| Fold | Train | Test | DA Delta | MAE Delta | Sharpe Delta |
|------|-------|------|----------|-----------|--------------|
| 1 | 423 | 420 | -2.39% | +0.184 | -0.673 |
| 2 | 843 | 420 | -1.92% | +0.018 | +0.074 |
| 3 | 1263 | 420 | -0.96% | +0.046 | +0.115 |
| 4 | 1683 | 420 | -2.15% | +0.050 | -0.266 |
| 5 | 2103 | 420 | -2.38% | +0.093 | -0.345 |

Key observations:
- Direction accuracy degraded in all 5 folds (average -1.96%)
- MAE degraded in all 5 folds (average +0.078)
- Sharpe showed mixed results but average strongly negative (-0.219)

---

## Comparison: Attempt 3 (Forward-Fill) vs Attempt 4 (Cubic Spline)

| Metric | Attempt 3 | Attempt 4 | Improvement? |
|--------|-----------|-----------|-------------|
| Interpolation | Forward-fill (step functions) | Cubic spline (smooth) | -- |
| Gate 2 MI increase | +23.8% | +10.29% | Worse (fewer dimensions) |
| Gate 3 MAE delta | +0.42% | +0.078% | Better but still failed |
| Gate 3 DA delta | -0.48% (marginal) | -1.96% | Worse |
| Parameters | 98K (Transformer) | 0 (PCA) | Simpler |

Cubic spline reduced the MAE degradation magnitude (from +0.42% to +0.078%), but:
1. MAE still worsened in all 5 folds
2. Direction accuracy degraded more severely (-1.96% vs -0.48%)
3. The interpolation method is not the root cause

---

## Root Cause Analysis

The fundamental issue is the **monthly-to-daily frequency mismatch**:

1. Multi-country real rate data is released monthly
2. Even with cubic spline interpolation, the daily values between monthly releases are synthetic
3. These synthetic daily values add noise rather than signal to daily gold return prediction
4. The base `real_rate_real_rate` feature (daily US TIPS change) already captures the direct rate information available at daily frequency
5. PCA on monthly multi-country data captures cross-country co-movement patterns, but these patterns evolve too slowly to inform daily returns

This is confirmed by the consistent pattern across all 4 attempts:
- Gate 2 always passes (information exists at monthly frequency)
- Gate 3 always fails (information cannot be usefully interpolated to daily frequency)

---

## Decision: no_further_improvement

**Rationale**: After 4 attempts spanning MLP, GRU, Transformer, and PCA approaches, the real_rate submodel consistently passes Gate 2 but fails Gate 3. The root cause is a fundamental frequency mismatch that cannot be resolved by changing the model architecture or interpolation method.

**Action**: The base `real_rate_real_rate` feature remains in the feature set. No submodel output will be added for real_rate. Proceed to dxy submodel.

---

## Attempt History

| Attempt | Method | Gate 1 | Gate 2 | Gate 3 | Decision |
|---------|--------|--------|--------|--------|----------|
| 1 | MLP (US-only) | FAIL (overfit=2.69) | PASS (+18.5%) | FAIL | attempt+1 |
| 2 | GRU (US-only) | N/A (no convergence) | N/A | N/A | attempt+1 |
| 3 | Transformer (multi-country) | PASS (1.28) | PASS (+23.8%) | FAIL (MAE +0.42%) | attempt+1 |
| 4 | PCA + Cubic Spline | N/A (deterministic) | PASS (+10.29%) | FAIL (all metrics) | no_further_improvement |

---

## Lessons for Future Features

1. **Monthly data cannot supplement daily predictions via submodels** -- the frequency gap is too large for interpolation to bridge meaningfully.
2. **Gate 2 passing does not guarantee Gate 3 will pass** -- MI measures information content, but ablation tests whether the model can use that information without overfitting to noise.
3. **Simpler is not always better** -- PCA was the simplest approach but performed worse than the Transformer on direction accuracy, suggesting the Transformer was at least partially learning useful patterns.
4. **For future multi-country features (yield_curve, inflation_expectation)**: consider using only daily-frequency data, or designing features that explicitly model the monthly-to-daily relationship rather than interpolating.
