# Evaluation Summary: real_rate attempt 6

## Method
Deterministic bond volatility z-score + rate momentum z-score (US-only daily DFII10)
Best params: vol_window=10, vol_zscore_window=60, autocorr_window=10, autocorr_zscore_window=30

## Gate 1: Standalone Quality -- PASS
- Overfit ratio: N/A (deterministic model) -- PASS
- All-NaN columns: 0 -- PASS
- Constant output columns: 0 -- PASS
- Std(rr_bond_vol_z): 1.1673 (threshold > 0.1) -- PASS
- Std(rr_momentum_z): 1.2112 (threshold > 0.1) -- PASS
- Autocorr(rr_bond_vol_z): 0.8989 (threshold < 0.99) -- PASS
- Autocorr(rr_momentum_z): 0.7695 (threshold < 0.99) -- PASS
- NaN values: 0 -- PASS
- Row count: 2523/2523 -- PASS
- HPO: 36/36 exhaustive search -- PASS

## Gate 2: Information Gain -- FAIL
- MI increase: 1.95% (threshold > 5%) -- FAIL
  - Base MI sum: 0.3095
  - Extended MI sum: 0.3156
- VIF (rr_bond_vol_z): 1.1762785114854466
- VIF (rr_momentum_z): 1.0275324277229965
- Max VIF (new features): 1.1762785114854466 (threshold < 10) -- PASS
- Rolling corr stability (max std): 0.1393 (threshold < 0.15) -- PASS

## Gate 3: Ablation -- FAIL

| Metric | Baseline (24-feat) | Extended (26-feat) | Delta | Threshold | Result |
|--------|--------------------|--------------------|-------|-----------|--------|
| Direction Accuracy | 54.61% | 54.03% | -0.574pp | +0.5pp | FAIL |
| Sharpe | 1.243 | 0.994 | -0.2488 | +0.05 | FAIL |
| MAE | 0.6490 | 0.6482 | -0.00074 | -0.01 | FAIL |

Per-fold consistency:
- DA improved in 2/5 folds
- Sharpe improved in 3/5 folds
- MAE improved in 4/5 folds

## Decision: NO_FURTHER_IMPROVEMENT

Gate 3 FAIL on attempt 6. After 5 previous failures (all Gate 3), plus this 6th attempt with a fundamentally different approach, declare no_further_improvement permanently. Real rate dynamics are adequately captured by base features.

## Historical Context (Attempts 1-6)

| Attempt | Method | Gate 1 | Gate 2 | Gate 3 | Failure Mode |
|---------|--------|--------|--------|--------|--------------|
| 1 | MLP Autoencoder | FAIL | PASS | FAIL | Overfit ratio 2.69, autocorr > 0.99 |
| 2 | GRU Autoencoder | N/A | N/A | N/A | All Optuna trials pruned, GRU convergence failure |
| 3 | Transformer + Monthly FF | PASS | PASS | FAIL | Step-function degraded MAE (+0.48pp DA miss) |
| 4 | PCA + Cubic Spline | PASS | PASS | FAIL | All metrics degraded (DA -1.96%, MAE +0.078) |
| 5 | Markov + CUSUM (7 cols) | PASS | PASS | FAIL | Worst degradation (DA -2.53%, MAE +0.160) |
| 6 | Deterministic Vol+Momentum | PASS | FAIL | FAIL | See above |
