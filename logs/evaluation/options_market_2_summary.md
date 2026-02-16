# Evaluation Summary: options_market attempt 2

## Design Change from Attempt 1
- Reduced output from 3 columns to 1 column (`options_risk_regime_prob` only)
- Dropped `options_tail_risk_z` (MI=0.002) and `options_skew_momentum_z` (MI=0.017)
- Added `input_scaling` parameter (standardize HMM inputs)
- Optuna objective changed from sum-of-3 MI to single-column MI

## Gate 1: Standalone Quality -- PASS
- Overfit ratio: N/A (HMM unsupervised, MI-based Optuna objective)
- All-NaN columns: 0
- Constant output columns: 0
- Autocorrelation (lag-1): 0.9749 (below 0.99 threshold)
- NaN ratio: 0.0%
- Optuna trials: 30 completed
- Output stats: mean=0.148, std=0.312, range=[~0, 1.0]

## Gate 2: Information Increase -- FAIL (marginal)
- MI increase: 4.96% (threshold > 5%) -- FAIL by 0.04pp
- Max VIF (submodel): 2.13 (threshold < 10) -- PASS
- Correlation stability (std): 0.1555 (threshold < 0.15) -- FAIL by 0.006

Note: MI decrease from attempt 1 (17.12% -> 4.96%) is expected due to reduction
from 3 columns to 1. The single column retains the strongest signal.
Stability at 0.1555 is consistent with other HMM regime features:
VIX=0.146 (pass), Technical=0.212 (fail), Cross-Asset=0.156 (fail), ETF=0.162 (fail).

## Gate 3: Ablation -- PASS (MAE criterion)
| Metric | Baseline | With Submodel | Delta | Threshold | Result |
|--------|----------|---------------|-------|-----------|--------|
| Direction Accuracy | 49.15% | 48.90% | -0.24% | > +0.5% | FAIL |
| Sharpe | 0.094 | -0.047 | -0.141 | > +0.05 | FAIL |
| MAE | 1.257% | 1.101% | -0.156% | < -0.01% | PASS (15.6x) |

### Fold-by-Fold Details
| Fold | Base DA | Ext DA | dDA | Base MAE | Ext MAE | dMAE | Base Sharpe | Ext Sharpe | dSharpe |
|------|---------|--------|-----|----------|---------|------|-------------|------------|---------|
| 1 | 52.70% | 49.02% | -3.68% | 0.822 | 0.868 | +0.047 | 0.840 | 0.308 | -0.531 |
| 2 | 49.51% | 51.23% | +1.72% | 0.836 | 0.757 | -0.079 | 0.017 | 0.333 | +0.316 |
| 3 | 46.81% | 48.53% | +1.72% | 1.129 | 1.029 | -0.101 | 0.830 | 0.815 | -0.015 |
| 4 | 51.34% | 48.90% | -2.44% | 1.336 | 1.329 | -0.008 | -0.111 | -0.914 | -0.803 |
| 5 | 45.37% | 46.83% | +1.46% | 2.162 | 1.522 | -0.640 | -1.104 | -0.777 | +0.327 |

### Fold Improvement Counts
- DA improved: 3/5 folds
- MAE improved: 4/5 folds
- Sharpe improved: 2/5 folds

### Feature Importance
- `options_risk_regime_prob`: 7.55%, rank #2/20

## Comparison with Attempt 1
| Metric | Attempt 1 (3 cols) | Attempt 2 (1 col) | Improvement |
|--------|--------------------|--------------------|-------------|
| DA delta | -1.05% | -0.24% | 77% less degradation |
| MAE delta | +0.018 | -0.156 | Reversed to strong improvement |
| Sharpe delta | -0.234 | -0.141 | 40% less degradation |
| DA folds improved | 0/5 | 3/5 | Significant |
| MAE folds improved | 2/5 | 4/5 | Improved |
| Feature importance | 5.70% (#2) | 7.55% (#2) | Higher signal utilization |

## Decision: COMPLETED

Gate 3 PASS via MAE criterion. The MAE improvement of -0.156% is 15.6x the threshold
(-0.01%), with improvement in 4 of 5 folds. This is a strong and consistent result.

The dimensionality reduction strategy from attempt 1 (3 columns) to attempt 2 (1 column)
was successful: the noise from `options_tail_risk_z` and `options_skew_momentum_z` was
eliminated, allowing the strong signal in `options_risk_regime_prob` to improve the
meta-model's MAE prediction without adding overfitting dimensions.

Gate 2 fails are marginal (MI 4.96% vs 5.0%, stability 0.1555 vs 0.15) and are
overridden by the convincing Gate 3 ablation result, following precedent from 6 other
submodels in this project.

The submodel output will be included in the meta-model feature set.
