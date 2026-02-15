# Evaluation Summary: cny_demand attempt 1

## Gate 1: Standalone Quality -- PASS
- Overfit ratio: N/A (HMM unsupervised, MI-based HPO) -- PASS
- All-NaN columns: 0 -- PASS
- Constant output columns: 0 -- PASS
- Autocorrelation (cny_regime_prob): 0.639 (threshold < 0.99) -- PASS
- Autocorrelation (cny_momentum_z): 0.737 (threshold < 0.99) -- PASS
- Autocorrelation (cny_vol_regime_z): 0.921 (threshold < 0.99) -- PASS
- Optuna trials completed: 30 (>= 10) -- PASS
- NaN %: regime_prob 0.32%, momentum_z 4.44%, vol_regime_z 2.45% -- PASS

## Gate 2: Information Gain -- FAIL
- MI increase: 0.09% (threshold > 5%) -- FAIL
  - Base MI sum: 0.3142
  - Extended MI sum: 0.3145
  - Submodel MI nearly zero for momentum_z and vol_regime_z
- Max submodel VIF: 1.26 (threshold < 10) -- PASS
  - regime_prob: 1.26, momentum_z: 1.07, vol_regime_z: 1.11
  - Excellent orthogonality to base features
- Correlation stability (max std): 0.143 (threshold < 0.15) -- PASS
  - regime_prob: 0.143, momentum_z: 0.135, vol_regime_z: 0.117

## Gate 3: Ablation -- PASS (via MAE)
| Metric | Baseline | With Submodel | Delta | Threshold | Verdict |
|--------|----------|---------------|-------|-----------|---------|
| Direction Accuracy | 48.85% | 46.79% | -2.06% | > +0.5% | FAIL |
| Sharpe | 0.161 | -0.433 | -0.593 | > +0.05 | FAIL |
| MAE | 1.2417 | 1.1759 | -0.0658 | < -0.01 | PASS |

### Fold-by-fold results:
| Fold | DA delta | MAE delta | Sharpe delta |
|------|----------|-----------|--------------|
| 1 | +0.76% | -0.3103 | -0.152 |
| 2 | -0.76% | +0.0799 | -1.348 |
| 3 | -6.28% | +0.1715 | -0.984 |
| 4 | -2.76% | -0.0488 | -0.774 |
| 5 | -1.25% | -0.2215 | +0.290 |

MAE improved in 3/5 folds (folds 1, 4, 5).
DA improved in 1/5 folds. Sharpe improved in 1/5 folds.

### Feature Importance (last fold):
- cny_momentum_z: rank 4/22, importance 5.57%
- cny_regime_prob: rank 13/22, importance 4.81%
- cny_vol_regime_z: rank 18/22, importance 4.05%

## Verdict: COMPLETED (Gate 3 PASS via MAE)

Gate 3 passes on MAE criterion (-0.0658, 6.6x threshold). This is the weakest Gate 3 pass among all submodels in terms of side effects: DA degrades -2.06% and Sharpe degrades -0.593, both the worst of any passing submodel. The pattern suggests CNY demand features attenuate prediction magnitudes (improving MAE) while adding noise to directional signals.

### Warnings for Meta-Model Integration:
1. DA degradation (-2.06%) is the worst of any MAE-passing submodel
2. Sharpe degradation (-0.593) is the worst of any MAE-passing submodel
3. CNY features may attenuate prediction magnitudes while adding directional noise
4. Meta-model should be tested with and without cny_demand features (ablation)
5. Consider using only cny_momentum_z (rank 4, highest individual importance) and dropping regime_prob and vol_regime_z if meta-model DA degrades

### Phase 2 Status: COMPLETE (9/9 features processed)
This is the final submodel. All 9 features have been evaluated:
- 7 completed (vix, technical, cross_asset, yield_curve, etf_flow, inflation_expectation, cny_demand)
- 1 completed with no submodel output (dxy - auto-evaluated)
- 1 no further improvement (real_rate - 5 attempts exhausted)

Ready for Phase 3: Meta-Model Construction.
