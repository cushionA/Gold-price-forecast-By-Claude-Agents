# Evaluation Summary: temporal_context attempt 1

## Gate 1: Standalone Quality -- PASS
- Overfit ratio: 1.2155 (threshold < 1.5) PASS
- All-NaN columns: 0 PASS
- Constant output columns: 0 PASS
- Inf values: 0 PASS
- Autocorrelation(temporal_context_score, lag=1): 0.970 (threshold < 0.99) PASS
- Optuna trials: 30 completed PASS

## Gate 2: Information Gain -- PASS
- MI increase: +8.05% (threshold > 5%) PASS
- Submodel VIF: 2.98 (threshold < 10) PASS
- Rolling correlation stability (std): 0.129 (threshold < 0.15) PASS

## Gate 3: Ablation -- PASS (all 3 criteria met)

| Metric | Baseline | Extended | Delta | Threshold | Result |
|--------|----------|----------|-------|-----------|--------|
| Direction Accuracy | 50.47% | 51.06% | +0.58% | > +0.5% | PASS |
| Sharpe | 0.2985 | 0.4118 | +0.1132 | > +0.05 | PASS |
| MAE | 1.3282 | 1.1702 | -0.1580 | < -0.01 | PASS |

### Fold-by-Fold Consistency

| Fold | DA Delta | Sharpe Delta | MAE Delta |
|------|----------|-------------|-----------|
| 1 | +0.25% | -0.2234 | -0.0533 (improved) |
| 2 | -1.23% | -0.2116 | -0.0286 (improved) |
| 3 | -0.98% | -0.4155 | -0.0163 (improved) |
| 4 | +3.67% | +1.1172 | -0.4812 (improved) |
| 5 | +1.22% | +0.2996 | -0.2109 (improved) |
| Improved | 3/5 | 2/5 | **5/5** |

### Feature Importance
- temporal_context_score: Rank #10/20, Importance 5.17%

## Decision: COMPLETED

### Key Findings:

1. **All 3 gates PASS** -- Only the second submodel (after inflation_expectation) to achieve this.

2. **MAE is the strongest signal**: -0.158, which is 15.8x the threshold, and improved in ALL 5/5 folds. This indicates the temporal context Transformer captures genuine information about prediction magnitude.

3. **DA marginally passes** (+0.58% vs +0.5% threshold) with 3/5 folds improved. The improvement is concentrated in Folds 4-5 (later time periods with higher volatility).

4. **Sharpe passes** (+0.1132 vs +0.05 threshold) but only 2/5 folds improved. Fold 4 contributes disproportionately (+1.1172). This suggests the temporal context signal is especially valuable during volatile regimes.

5. **Model characteristics**: Masked Temporal Context Transformer with d_model=24, n_heads=2, n_layers=2, window_size=5. Single-column output (temporal_context_score) in [0,1] range. 11,007 parameters. Well-regularized (overfit ratio 1.22).

6. **Information quality**: MI increase of 8.05% with low VIF (2.98) and stable rolling correlations (std=0.129). The feature adds genuine orthogonal information.

### Comparison with Other Submodels (Gate 3 MAE criterion):
- temporal_context: -0.158 (15.8x threshold) -- **3rd best**
- technical: -0.182 (18.2x threshold) -- 1st
- options_market: -0.156 (15.6x threshold) -- 4th
- cross_asset: -0.087 (8.7x threshold) -- 5th
- yield_curve: -0.069 (6.9x threshold) -- 6th
- cny_demand: -0.066 (6.6x threshold) -- 7th
- etf_flow: -0.044 (4.4x threshold) -- 8th

### Next Action:
Move submodel output to `data/submodel_outputs/temporal_context.csv` and update `shared/completed.json`. This submodel is ready for inclusion in the meta-model feature set.
