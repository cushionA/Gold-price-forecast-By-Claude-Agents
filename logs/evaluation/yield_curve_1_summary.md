# Evaluation Summary: yield_curve attempt 1

## Gate 1: Standalone Quality -- FAIL

- Overfit ratio: N/A (HMM unsupervised, no train/val loss) -- skipped
- All-NaN columns: 0 -- PASS
- Constant output columns: 1 (yc_regime_prob, std=1.07e-11) -- FAIL
- Autocorrelation (lag=1):
  - yc_regime_prob: 0.138 -- PASS
  - yc_spread_velocity_z: 0.760 -- PASS
  - yc_curvature_z: 0.027 -- PASS
- Optuna trials: 30 -- PASS

**Failure detail:** yc_regime_prob is effectively constant across all 2,794 samples. All values are in the range 1e-15 to 3.3e-10, meaning the HMM 2-component model collapsed to a single state. The regime detection completely failed for this feature. The remaining 2 columns (spread_velocity_z and curvature_z) pass all standalone checks.

## Gate 2: Information Gain -- FAIL

- MI increase: 0.37% (threshold > 5%) -- FAIL
- Max VIF (all): inf (base feature multicollinearity, not caused by submodel) -- FAIL
- Max VIF (submodel only): 1.17 -- PASS
- Correlation stability (std): 0.1315 (threshold < 0.15) -- PASS

**Failure detail:** All 3 submodel columns have MI = 0.000 against gold returns individually. The MI test indicates the linear information content is negligible. However, previous successful submodels (VIX, Technical, Cross-Asset) also failed Gate 2 MI and still passed Gate 3, suggesting MI underestimates nonlinear contributions.

## Gate 3: Ablation -- PASS (via MAE)

Note: Evaluated with 2 columns only (regime_prob excluded as constant).

| Metric | Baseline | Extended | Delta | Threshold | Result |
|--------|----------|----------|-------|-----------|--------|
| Direction Accuracy | 48.16% | 48.36% | +0.20% | > +0.50% | FAIL |
| Sharpe | -0.222 | -0.311 | -0.089 | > +0.05 | FAIL |
| MAE | 1.3438 | 1.2745 | -0.0693 | < -0.01 | PASS (6.9x) |

### Fold Consistency

| Fold | DA delta | MAE delta | Sharpe delta |
|------|----------|-----------|--------------|
| 1 | -0.25% | -0.2005 | -0.219 |
| 2 | -0.50% | -0.0516 | +0.243 |
| 3 | +3.48% | -0.1697 | +0.098 |
| 4 | -0.25% | -0.3069 | -0.096 |
| 5 | -1.49% | +0.3819 | -0.471 |

- DA improved: 1/5 folds
- MAE improved: 4/5 folds
- Sharpe improved: 2/5 folds

### Feature Importance (last fold)

- yc_curvature_z: 0.0426 (rank 15/21)
- yc_spread_velocity_z: 0.0383 (rank 17/21)

## Decision: completed

**Rationale:** Gate 3 passes via MAE criterion (-0.0693, 6.9x threshold, 4/5 folds consistent). This follows the same pattern as Technical (-0.1824, 18x) and Cross-Asset (-0.0866, 8.7x) submodels which also passed primarily via MAE. The yield curve's spread velocity and curvature z-scores provide meaningful error reduction through nonlinear interactions in XGBoost.

**Important caveats:**
1. Only 2 of 3 output columns are usable. yc_regime_prob MUST be excluded from the meta-model (constant column).
2. Sharpe degraded by -0.089 on average, and fold 5 shows significant MAE degradation (+0.38).
3. Feature importance ranks are modest (15th and 17th out of 21), suggesting marginal but real contribution.

**Comparison with prior submodels:**

| Submodel | MAE delta | DA delta | Sharpe delta | Status |
|----------|-----------|----------|--------------|--------|
| VIX | +0.016 | +0.96% | +0.289 | completed (DA/Sharpe) |
| Technical | -0.1824 | +0.05% | -0.092 | completed (MAE 18x) |
| Cross-Asset | -0.0866 | +0.76% | +0.040 | completed (DA + MAE 8.7x) |
| **Yield Curve** | **-0.0693** | **+0.20%** | **-0.089** | **completed (MAE 6.9x)** |

The yield_curve MAE improvement (6.9x threshold) is the weakest among MAE-passing submodels but still comfortably exceeds the threshold and is consistent across 4/5 folds.
