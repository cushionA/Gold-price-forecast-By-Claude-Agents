# Evaluation Summary: regime_classification attempt 1

## Gate 1: Standalone Quality -- PASS
- Overfit ratio: 0.986 (threshold < 1.5) PASS
- All-NaN columns: 0 PASS
- Constant output columns: None PASS
- Regime balance: R0=30.1%, R1=69.9% (threshold: no regime > 80%) PASS
- Avg regime duration: 10.0 days (threshold: 5-30) PASS
- Probability sum validation: max deviation=0.000000 (threshold < 0.01) PASS
- Autocorrelation checks:
  - regime_prob_0: 0.9195 (threshold < 0.99) PASS
  - regime_prob_1: 0.9195 (threshold < 0.99) PASS
  - regime_transition_velocity: 0.4680 (threshold < 0.99) PASS

## Gate 2: Information Gain -- FAIL
- MI increase: 0.77% (threshold > 5%) FAIL
  - Base MI sum: 0.169791
  - Extended MI sum: 0.171103
- VIF(regime_prob_0): 3.8991 (threshold < 10) PASS
- VIF(regime_prob_1): 3.5896 (threshold < 10) PASS
- VIF(regime_transition_velocity): 1.1865 (threshold < 10) PASS
- Max submodel VIF: 3.8991 PASS
- Rolling corr stability(regime_prob_0): std=0.046110 (threshold < 0.15) PASS
- Rolling corr stability(regime_prob_1): std=0.046110 (threshold < 0.15) PASS
- Rolling corr stability(regime_transition_velocity): std=0.057921 (threshold < 0.15) PASS

## Gate 3: Ablation -- PASS

| Metric | Baseline | With Submodel | Delta | Threshold | Verdict |
|--------|----------|---------------|-------|-----------|---------|
| Direction Accuracy | 50.88% | 52.22% | +1.34% | +0.50% | PASS |
| Sharpe | 0.4384 | 0.8158 | +0.3774 | +0.05 | PASS |
| MAE | 0.6717 | 0.6704 | -0.0012 | -0.01 | FAIL |

### Fold Details

| Fold | Train | Test | DA Delta | Sharpe Delta | MAE Delta |
|------|-------|------|----------|--------------|-----------|
| 1 | 423 | 420 | +2.39% | +0.7889 | +0.0052 |
| 2 | 843 | 420 | +0.48% | +0.1486 | -0.0004 |
| 3 | 1263 | 420 | +0.48% | -0.3827 | -0.0042 |
| 4 | 1683 | 420 | +3.34% | +1.1368 | -0.0061 |
| 5 | 2103 | 420 | +0.00% | +0.1953 | -0.0007 |

Improved folds: DA 4/5, Sharpe 4/5, MAE 4/5

### Feature Importance (new features, last fold)
- regime_prob_0: rank 18/28, importance 3.57%
- regime_prob_1: rank 20/28, importance 3.44%
- regime_transition_velocity: rank 25/28, importance 2.34%

## Verdict: PASS
## Decision: completed

Gate 1 PASS, Gate 2 FAIL, Gate 3 PASS via DA (+1.34%, 4/5 folds), Sharpe (+0.3774, 4/5 folds). Gate 2 mi failed but Gate 3 passes override (precedent: vix, etf_flow, cross_asset, etc.).
