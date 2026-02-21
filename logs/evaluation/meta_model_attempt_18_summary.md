# Evaluation Summary: meta_model attempt 18

## Gate 1: Standalone Quality -- PASS
- Overfitting gap: 5.96pp (threshold < 10pp) PASS
- All-NaN columns: 0 PASS
- Constant output columns: 0 (std=0.2132, positive_pct=53.9%) PASS
- Autocorrelation: No values > 0.99 PASS
- HPO: N/A (used attempt 7 exact params) PASS
- Bootstrap ensemble: 12 models, 80% data subsampling PASS

## Gate 2 Substitute: Feature Importance Distribution -- PASS
- Top feature: real_rate_change at 5.28% (no single feature dominates) PASS
- Real_rate submodel total: 14.60% (4 features, ranked 2/12/15/25) PASS
- Feature importance spread: healthy, no concentration PASS

## Gate 3: Final Targets -- 3/4 PASS (nominal), FAIL vs attempt 7

| Metric | Target | Attempt 18 | Attempt 7 (Best) | Delta vs 7 | Target | vs Best |
|--------|--------|------------|-------------------|------------|--------|---------|
| Direction Accuracy | > 56% | 58.30% | 60.04% | -1.74pp | PASS | REGRESSION |
| High-Confidence DA | > 60% | 63.04% | 64.13% | -1.09pp | PASS | REGRESSION |
| MAE | < 0.75% | 0.9527% | 0.9429% | +0.0098% | FAIL | REGRESSION |
| Sharpe | > 0.80 | 1.86 | 2.46 | -0.60 | PASS | REGRESSION |

Attempt 18 REGRESSES on ALL 4 metrics vs attempt 7.

## vs Attempt 17 (prior attempt, 24 features)

| Metric | Attempt 17 | Attempt 18 | Delta |
|--------|------------|------------|-------|
| DA | 58.73% | 58.30% | -0.43pp |
| HCDA | 59.78% | 63.04% | +3.26pp |
| MAE | 0.9558% | 0.9527% | -0.0031% |
| Sharpe | 1.96 | 1.86 | -0.10 |

Adding 4 real_rate features improved HCDA but regressed DA and Sharpe.

## vs Attempt 16 (best HCDA, LightGBM + bootstrap)

| Metric | Attempt 16 | Attempt 18 | Delta |
|--------|------------|------------|-------|
| DA | 58.52% | 58.30% | -0.22pp |
| HCDA | 68.48% | 63.04% | -5.44pp |
| MAE | 0.9534% | 0.9527% | -0.0007% |
| Sharpe | 1.76 | 1.86 | +0.10 |

XGBoost + data subsampling does NOT replicate LightGBM's bootstrap HCDA technique.

## Decision: no_further_improvement -- Attempt 7 declared FINAL meta-model

This is the FINAL evaluation. After 11 consecutive improvement attempts (8-18), none have beaten attempt 7 overall. Attempt 7 is the permanent final meta-model.

### Final Model: Attempt 7
- Architecture: XGBoost reg:squarederror + Bootstrap confidence + OLS scaling
- Features: 24
- DA: 60.04% (target > 56%) -- PASS
- HCDA: 64.13% (target > 60%) -- PASS
- MAE: 0.9429% (target < 0.75%) -- FAIL (structurally infeasible)
- Sharpe: 2.4636 (target > 0.80) -- PASS
- Targets: 3/4
