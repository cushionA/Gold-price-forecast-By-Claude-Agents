# Evaluation Summary: options_market attempt 1

## Gate 1: Standalone Quality -- PASS
- Overfit ratio: N/A (HMM unsupervised, Optuna MI-based objective) -- N/A
- All-NaN columns: 0 -- PASS
- Constant output columns: 0 -- PASS
- Autocorrelation (lag 1):
  - options_risk_regime_prob: 0.895 (<0.99) -- PASS
  - options_tail_risk_z: 0.824 (<0.99) -- PASS
  - options_skew_momentum_z: 0.765 (<0.99) -- PASS
- Notes: 39 NaN rows in tail_risk_z, 69 in skew_momentum_z (window warmup, forward-filled). 30 Optuna trials completed, best MI sum = 1.728.

## Gate 2: Information Gain -- PASS
- MI increase: +17.12% (threshold >5%) -- PASS
  - MI base sum: 0.310
  - MI extended sum: 0.363
  - options_risk_regime_prob MI: 0.031
  - options_tail_risk_z MI: 0.002
  - options_skew_momentum_z MI: 0.017
- Max VIF (submodel): 2.12 (threshold <10) -- PASS
  - options_risk_regime_prob VIF: 1.87
  - options_tail_risk_z VIF: 2.12
  - options_skew_momentum_z VIF: 1.94
- Correlation stability (std): 0.146 (threshold <0.15) -- PASS
  - options_risk_regime_prob: 0.146
  - options_tail_risk_z: 0.126
  - options_skew_momentum_z: 0.126

## Gate 3: Ablation -- FAIL

| Metric | Baseline | Extended | Delta | Threshold | Result |
|--------|----------|----------|-------|-----------|--------|
| Direction Accuracy | 49.31% | 48.26% | -1.05% | >+0.5% | FAIL |
| Sharpe | 0.262 | 0.028 | -0.234 | >+0.05 | FAIL |
| MAE | 0.8886 | 0.9065 | +0.018 | <-0.01 | FAIL |

### Fold-Level Results

| Fold | Base DA | Ext DA | Base MAE | Ext MAE | Base Sharpe | Ext Sharpe |
|------|---------|--------|----------|---------|-------------|------------|
| 1 | 50.72% | 49.04% | 0.7657 | 0.8463 | 0.761 | -0.249 |
| 2 | 50.24% | 49.28% | 0.5860 | 0.5994 | 0.254 | 0.157 |
| 3 | 47.37% | 46.17% | 1.0326 | 1.0651 | 0.718 | 0.712 |
| 4 | 52.27% | 51.55% | 0.9141 | 0.8959 | 0.273 | 0.374 |
| 5 | 45.95% | 45.24% | 1.1443 | 1.1259 | -0.698 | -0.855 |

- DA improved: 0/5 folds
- MAE improved: 2/5 folds (folds 4, 5)
- Sharpe improved: 1/5 folds (fold 4)

### Feature Importance (Last Fold)
- options_risk_regime_prob: 5.70% (rank #2/22)
- options_tail_risk_z: 4.24% (rank #13/22)
- options_skew_momentum_z: 4.18% (rank #15/22)

## Decision: attempt+1

### Analysis

Gate 1 and Gate 2 pass convincingly. The MI increase of +17.12% confirms that options market data (SKEW index + GVZ) contains meaningful information about gold returns. VIF values are excellent (max 2.12), indicating low collinearity with existing base features. Stability is borderline but passes (0.146 vs 0.15 threshold).

However, Gate 3 fails on all three criteria. DA degrades in all 5 folds (-1.05% average), MAE increases (+0.018), and Sharpe drops sharply (-0.234). The pattern is clear: adding 3 extra features gives XGBoost more dimensions to overfit on without improving generalization.

Notably, options_risk_regime_prob ranks #2 in feature importance (5.70%), which is very high. This confirms the model actively uses this feature. The issue is not that the information is useless, but that the 3-column output format introduces too much noise relative to signal.

### Improvement Direction for Attempt 2

**Primary strategy: Output dimensionality reduction**
- Reduce from 3 output columns to 1 (regime_prob only, or PCA single component)
- options_risk_regime_prob is clearly the most valuable feature (rank #2, MI 0.031)
- options_tail_risk_z and options_skew_momentum_z add moderate importance but may contribute more noise than signal
- Alternative: Apply PCA to compress 3 columns into 1 principal component

**Resume from: architect** (redesign output format)
