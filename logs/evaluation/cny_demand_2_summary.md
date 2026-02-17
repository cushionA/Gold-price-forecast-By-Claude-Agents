# Evaluation Summary: cny_demand attempt 2

## Architecture
Deterministic CNY-CNH spread change z-score (single output column: cny_demand_spread_z)

## Gate 1: Standalone Quality -- PASS
- Overfit ratio: N/A (deterministic model, no training)
- All-NaN columns: 0
- Constant output columns: 0 (std = 1.018)
- Autocorrelation lag-1: 0.394 (threshold < 0.99)
- Optuna trials: 30 (window parameter optimization)
- NaN percentage: 0.0%

## Gate 2: Information Gain -- FAIL
- MI increase: -0.13% (threshold > 5%) -- FAIL
  - Base MI (sum, 24 features): 0.2621
  - Extended MI (sum, 25 features): 0.2617
  - cny_demand_spread_z individual MI: 0.0055
- Max VIF: 1.14 (threshold < 10) -- PASS
- Rolling correlation stability (std): 0.134 (threshold < 0.15) -- PASS

## Gate 3: Ablation -- PASS

Comparison: 24 features (current meta-model 7) vs 25 features (+cny_demand_spread_z)
Using meta-model attempt 7 best XGBoost hyperparameters. 5-fold time-series CV.

| Metric | Baseline (24) | Extended (25) | Delta | Threshold | Result |
|--------|--------------|---------------|-------|-----------|--------|
| Direction Accuracy | 55.04% | 56.57% | +1.53pp | > +0.5% | PASS (5/5 folds) |
| Sharpe (after 5bps) | 1.4426 | 1.6591 | +0.2166 | > +0.05 | PASS (5/5 folds) |
| MAE | 0.6623 | 0.6616 | -0.0008 | < -0.01 | FAIL (3/5 folds) |

### Fold-by-Fold Details

| Fold | DA Base | DA Ext | DA Delta | Sharpe Base | Sharpe Ext | Sharpe Delta | MAE Delta |
|------|---------|--------|----------|-------------|------------|-------------|-----------|
| 1 | 51.20% | 53.11% | +1.91pp | 0.228 | 0.373 | +0.146 | -0.003 |
| 2 | 52.88% | 54.57% | +1.68pp | 0.716 | 0.764 | +0.048 | -0.001 |
| 3 | 58.37% | 59.09% | +0.72pp | 2.787 | 2.980 | +0.193 | +0.002 |
| 4 | 55.85% | 58.00% | +2.15pp | 1.784 | 2.351 | +0.567 | +0.001 |
| 5 | 56.90% | 58.10% | +1.19pp | 1.698 | 1.827 | +0.129 | -0.003 |

### Feature Importance
- cny_demand_spread_z: avg importance 0.031 (avg rank 22.4/25)
- Low importance but consistent positive contribution to DA and Sharpe across all 5 folds

## Decision: completed

Gate 3 PASS via DA (+1.53pp, 3.1x threshold, 5/5 folds) and Sharpe (+0.217, 4.3x threshold, 5/5 folds).

## Comparison with Attempt 1

| Metric | Attempt 1 (HMM 3-output) | Attempt 2 (Spread z-score) | Improvement |
|--------|-------------------------|---------------------------|-------------|
| Gate 1 | PASS | PASS | Same |
| Gate 2 | FAIL (MI +0.09%) | FAIL (MI -0.13%) | Similar |
| Gate 3 DA delta | -2.06% (4/5 degraded) | +1.53% (5/5 improved) | +3.59pp better |
| Gate 3 Sharpe delta | -0.593 (4/5 degraded) | +0.217 (5/5 improved) | +0.810 better |
| Gate 3 MAE delta | -0.066 (3/5 improved) | -0.001 (3/5 improved) | MAE weaker |
| Overall Gate 3 | PASS (MAE only) | PASS (DA + Sharpe) | Much stronger |
| Meta-model 7 usage | 0/24 features | To be tested | -- |

Attempt 2 is dramatically superior to attempt 1:
- Attempt 1 passed Gate 3 via MAE only, but degraded DA and Sharpe (worst of any passing submodel)
- Attempt 2 passes Gate 3 via DA and Sharpe with improvement in ALL 5 folds
- Attempt 2's DA improvement (+1.53pp) reverses attempt 1's DA degradation (-2.06pp)
- Attempt 2's Sharpe improvement (+0.217) reverses attempt 1's Sharpe degradation (-0.593)

## Recommendation
This submodel output (cny_demand_spread_z) should replace the attempt 1 outputs (cny_regime_prob, cny_momentum_z, cny_vol_regime_z) in the Kaggle dataset and be included as a 25th feature in future meta-model iterations. The consistent DA and Sharpe improvements across all 5 folds suggest genuine informational value from the CNY-CNH spread dynamics.
