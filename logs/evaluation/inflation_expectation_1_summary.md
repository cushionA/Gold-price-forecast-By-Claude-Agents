# Evaluation Summary: inflation_expectation attempt 1

## Gate 1: Standalone Quality -- PASS

- Overfit ratio: N/A (HMM unsupervised, Optuna optimized MI directly)
- All-NaN columns: 0 PASS
- Constant output columns: 0 (std: 0.28, 1.04, 1.02) PASS
- Autocorrelation:
  - ie_regime_prob: 0.925 (< 0.99) PASS
  - ie_anchoring_z: 0.830 (< 0.99) PASS
  - ie_gold_sensitivity_z: 0.704 (< 0.99) PASS

Warnings:
- ie_regime_prob is near-zero in 78.8% of rows (values < 0.001). Regime state 2 is rarely activated.
- ie_anchoring_z and ie_gold_sensitivity_z have ~4% exact zeros at series start (warm-up period).

## Gate 2: Information Gain -- PASS

- MI increase: +10.94% (threshold > 5%) PASS
  - Base MI sum: 0.3095
  - Extended MI sum: 0.3434
  - ie_gold_sensitivity_z: MI = 0.0228 (strongest contributor)
  - ie_regime_prob: MI = 0.0090
  - ie_anchoring_z: MI = 0.0000 (no individual MI, but may contribute via interaction)
- Max VIF (submodel): 2.25 (threshold < 10) PASS
  - ie_regime_prob: 2.25, ie_anchoring_z: 1.63, ie_gold_sensitivity_z: 1.01
- Correlation stability (std):
  - ie_regime_prob: 0.1243 (< 0.15) PASS
  - ie_anchoring_z: 0.1447 (< 0.15) PASS
  - ie_gold_sensitivity_z: 0.1191 (< 0.15) PASS

Notable: This is the first submodel to pass ALL THREE Gate 2 criteria (MI, VIF, stability).

## Gate 3: Ablation -- PASS

| Metric | Baseline | Extended | Delta | Threshold | Result |
|--------|----------|----------|-------|-----------|--------|
| Direction Accuracy | 47.78% | 48.35% | +0.57% | > +0.5% | PASS |
| Sharpe | -0.2725 | -0.1208 | +0.1516 | > +0.05 | PASS |
| MAE | 1.1218 | 1.1745 | +0.0526 | < -0.01 | FAIL |

Gate 3 PASS (2 of 3 criteria met; only 1 required).

### Fold-Level Detail

| Fold | Base DA | Ext DA | Base Sharpe | Ext Sharpe | Base MAE | Ext MAE |
|------|---------|--------|-------------|------------|----------|---------|
| 1 | 49.52% | 49.76% | +0.04 | +0.62 | 0.854 | 0.958 |
| 2 | 49.28% | 46.63% | +0.34 | -0.80 | 0.774 | 0.815 |
| 3 | 45.45% | 48.56% | -0.12 | +1.03 | 1.303 | 1.221 |
| 4 | 50.12% | 52.74% | -0.09 | +0.48 | 1.376 | 1.203 |
| 5 | 44.52% | 44.05% | -1.54 | -1.94 | 1.302 | 1.677 |

- DA improved in 3/5 folds
- Sharpe improved in 3/5 folds
- MAE improved in 2/5 folds
- Fold 2 and 5 show degradation; Folds 3 and 4 show strong improvement

### Feature Importance (Extended Model)

- ie_gold_sensitivity_z: rank #8 (5.38%) -- strongest submodel feature
- ie_regime_prob: rank #10 (5.04%)
- ie_anchoring_z: rank #14 (4.57%)

All three features contribute meaningfully. ie_gold_sensitivity_z (which measures how inflation expectations and gold returns co-move relative to baseline) is the most informative.

## Decision: COMPLETED

inflation_expectation submodel is accepted for meta-model integration. This is notable as:
1. First submodel to pass all three gates (Gate 1, 2, and 3)
2. Strong Gate 2 performance: MI +10.94% with excellent VIF (2.25) and stability (0.14)
3. Dual Gate 3 pass on both DA (+0.57%) and Sharpe (+0.15)
4. All three output features contribute to the extended model's feature importance

### Comparison with Other Completed Submodels

| Submodel | Gate 1 | Gate 2 | Gate 3 | DA Delta | Sharpe Delta | MAE Delta |
|----------|--------|--------|--------|----------|-------------|-----------|
| vix | PASS | FAIL | PASS | +0.96% | +0.289 | +0.016 |
| technical | PASS | FAIL | PASS | +0.05% | -0.092 | -0.182 |
| cross_asset | PASS | FAIL | PASS | +0.76% | +0.040 | -0.087 |
| yield_curve | FAIL | FAIL | PASS | +0.20% | -0.089 | -0.069 |
| etf_flow | PASS | FAIL | PASS | +0.45% | +0.377 | -0.044 |
| **inflation_expectation** | **PASS** | **PASS** | **PASS** | **+0.57%** | **+0.152** | **+0.053** |

inflation_expectation is the only submodel to pass Gate 2 (MI, VIF, and stability all within thresholds). Its DA and Sharpe improvements are in the middle range compared to peers. MAE degradation is a concern but is offset by strong directional and risk-adjusted return improvements.
