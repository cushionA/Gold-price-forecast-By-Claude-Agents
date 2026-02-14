# Evaluation Summary: real_rate Attempt 5

**Method:** Markov Regime + CUSUM Change Points + State Features (no interpolation)
**Hypothesis:** Forward-filling state descriptors (regime persistence, transition probability) avoids interpolation noise because held-constant states are truth, not approximations.

## Gate 1: Standalone Quality -- PASS
- Overfit ratio: N/A (deterministic approach) -- PASS
- All-NaN columns: 0 -- PASS
- Zero-variance columns: 0 -- PASS
- Autocorrelation: days_since_change = 0.998 (structural, overridden) -- PASS (override)
- All other columns: autocorr 0.956-0.975 (below 0.99 threshold) -- PASS

## Gate 2: Information Gain -- PASS
- MI increase: +39.3% (threshold > 5%) -- PASS
  - Best MI contributors: regime_persistence (0.033), transition_prob (0.033), days_since_change (0.029)
  - regime_sync: 0.000 MI (no information)
- VIF: max raw = 37.36 (FAIL), max true = 3.08 (PASS, override applied)
  - Override reason: Base features have VIF=inf due to duplicate columns (gld_close = etf_flow_gld_close). Raw statsmodels VIF is a numerical artifact of rank-deficient base matrix. True VIF via manual R^2 regression = 3.08, well below threshold. Same override as Attempt 4.
- Stability (rolling corr std): 0.132 (threshold < 0.15) -- PASS

## Gate 3: Ablation (XGBoost 5-fold TSCV) -- FAIL

| Metric | Baseline | Extended | Delta | Threshold | Result |
|--------|----------|----------|-------|-----------|--------|
| Direction Accuracy | 49.60% | 47.06% | -2.53% | > +0.5% | FAIL |
| Sharpe | 0.274 | -0.435 | -0.709 | > +0.05 | FAIL |
| MAE | 0.881% | 1.041% | +0.160% | < -0.01% | FAIL |

### Per-Fold Results

| Fold | Train | Test | DA Delta | MAE Delta | Sharpe Delta |
|------|-------|------|----------|-----------|--------------|
| 1 | 423 | 420 | -2.87% | +0.531 | -0.861 |
| 2 | 843 | 420 | -1.20% | +0.067 | -0.207 |
| 3 | 1263 | 420 | -2.15% | +0.192 | -0.553 |
| 4 | 1683 | 420 | -5.25% | +0.055 | -2.113 |
| 5 | 2103 | 420 | -1.19% | -0.042 | +0.187 |

**Folds with improvement:** DA=0/5, MAE=1/5, Sharpe=1/5

### Feature Importance (extended model, last fold)
All 7 submodel columns ranked 12th-26th out of 26 features (lowest tier). XGBoost assigns them minimal importance, confirming they add noise rather than signal.

## Decision: no_further_improvement

This is the **FINAL attempt** (5 of 5). Base real_rate feature retained. Proceeding to dxy.

## Cross-Attempt Comparison (5 Attempts)

| Attempt | Method | Gate 1 | Gate 2 MI | Gate 3 DA Delta |
|---------|--------|--------|-----------|-----------------|
| 1 | MLP (US-only) | FAIL (overfit=2.69) | +18.5% | N/A |
| 2 | GRU (US-only) | N/A (no convergence) | N/A | N/A |
| 3 | Transformer (multi-country) | PASS (1.28) | +23.8% | -0.48% |
| 4 | PCA + Cubic Spline | N/A (deterministic) | +10.29% | -1.96% |
| 5 | Markov Regime + CUSUM | PASS | +39.3% | -2.53% |

## Conclusion

All 5 approaches that reached Gate 2 showed significant mutual information increase (10-39%), confirming that real rate dynamics contain information relevant to gold returns. However, none improved XGBoost ablation metrics. The fundamental issue: **monthly macro regime information does not help predict daily gold returns when base features already contain the direct real rate level**. The information is real but operates at a frequency incompatible with daily prediction granularity.

Notably, Attempt 5 achieved the highest MI increase (+39.3%) but the worst Gate 3 degradation (DA -2.53%), suggesting the 7 state feature columns (vs 2 PCA columns in Attempt 4) gave XGBoost more opportunities to overfit on noise patterns.
