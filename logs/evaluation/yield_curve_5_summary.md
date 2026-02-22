# Evaluation Summary: yield_curve attempt 5

**Method**: Cross-Tenor Correlation Dynamics
**Date**: 2026-02-22 09:19
**Samples (Gate 3)**: 2357 (aligned, no NaN)
**Strategy**: REPLACEMENT (attempt 5 replaces attempt 2)

## Gate 1: Standalone Quality -- PASS

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| All-NaN columns | 0 | 0 | PASS |
| Constant columns | 0 | 0 | PASS |
| Autocorr yc_corr_long_short_z | 0.073753 | < 0.99 | PASS |
| Autocorr yc_corr_long_mid_z | -0.025359 | < 0.99 | PASS |
| Autocorr yc_corr_1y10y_z | 0.038322 | < 0.99 | PASS |
| Optuna trials | 30 | >= 10 | PASS |
| Overfit ratio | 1.0000 | < 1.5 | PASS |

## Gate 2: Information Gain -- PASS

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| MI increase | 5.33% | > 5% | PASS |
| VIF yc_corr_long_short_z | 1.0733 | < 10 | PASS |
| VIF yc_corr_long_mid_z | 1.4166 | < 10 | PASS |
| VIF yc_corr_1y10y_z | 1.4696 | < 10 | PASS |
| Max rolling corr std | 0.145780 | < 0.15 | PASS |

Individual MI contributions (new features):
- yc_corr_long_short_z: 0.022637
- yc_corr_long_mid_z: 0.000000
- yc_corr_1y10y_z: 0.000000

## Gate 3: Ablation (Replacement) -- FAIL

| Metric | Baseline (att2) | Extended (att5) | Delta | Threshold | Folds Improved | Result |
|--------|-----------------|-----------------|-------|-----------|---------------|--------|
| Direction Accuracy | 53.05% | 52.44% | -0.61pp | +0.5pp | 2/5 | FAIL |
| Sharpe | 1.0576 | 0.9532 | -0.1044 | +0.05 | 2/5 | FAIL |
| MAE | 0.718694 | 0.731363 | +0.012669 | -0.01 | 2/5 | FAIL |

### Per-Fold Results

| Fold | DA Delta | Sharpe Delta | MAE Delta |
|------|----------|-------------|-----------|
| 1 | +1.28pp | +0.8688 | -0.005467 |
| 2 | -0.26pp | +0.3730 | -0.012487 |
| 3 | -2.56pp | -0.9512 | +0.015055 |
| 4 | +0.26pp | -0.5343 | +0.019037 |
| 5 | -1.79pp | -0.2783 | +0.047205 |

## Decision: ATTEMPT+1

yield_curve attempt 5 **fails Gate 3**. No criterion met the threshold.

### Context from Prior Attempts
- Attempt 2 (production): Gate 3 PASS via MAE (-0.0127)
- Attempt 3: Gate 3 FAIL (DA -1.57pp, MAE +0.005)
- Attempt 4: Gate 3 FAIL (DA avg -1.47pp across all folds)
- Attempt 5 (this): Gate 3 FAIL