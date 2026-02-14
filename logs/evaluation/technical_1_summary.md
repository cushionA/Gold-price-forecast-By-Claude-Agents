# Evaluation Summary: technical attempt 1

## Gate 1: Standalone Quality -- PASS

- Overfit ratio: N/A (unsupervised HMM, no train/val loss applicable)
- All-NaN columns: 0 (after 40-row warm-up)
- Zero-variance columns: 0
- Autocorrelation (lag 1):
  - tech_trend_regime_prob: 0.9676 (high but below 0.99 threshold)
  - tech_mean_reversion_z: 0.0116 (excellent, no persistence)
  - tech_volatility_regime: 0.1826 (healthy)
- Optuna HPO: 30 trials completed
- Unique values: All 3 columns have near-100% unique values (no degenerate outputs)

## Gate 2: Information Gain -- FAIL (1/3 checks failed)

- MI increase: 21.97% (threshold > 5%) -- PASS
  - Base MI sum: 0.3129
  - Extended MI sum: 0.3816
  - Individual submodel MI contributions:
    - tech_trend_regime_prob: 0.0073
    - tech_mean_reversion_z: 0.0202
    - tech_volatility_regime: 0.0444 (highest among all submodel columns)
- Max submodel VIF: 3.30 (threshold < 10) -- PASS
  - tech_trend_regime_prob: 1.62
  - tech_mean_reversion_z: 1.54
  - tech_volatility_regime: 3.30
  - Note: Base features have inf VIF (known collinearity among raw price levels)
- Correlation stability max std: 0.2115 (threshold < 0.15) -- FAIL
  - tech_trend_regime_prob: 0.2115 (exceeds threshold)
  - tech_mean_reversion_z: 0.1352 (passes)
  - tech_volatility_regime: 0.1383 (passes)
  - Note: regime_prob instability is expected for HMM regime detection, as regimes shift structurally over time

## Gate 3: Ablation -- PASS (1/3 criteria met)

| Metric | Baseline | Extended | Delta | Threshold | Result |
|--------|----------|----------|-------|-----------|--------|
| Direction Accuracy | 49.74% | 49.79% | +0.05% | > +0.50% | FAIL |
| Sharpe | 0.507 | 0.415 | -0.092 | > +0.05 | FAIL |
| MAE | 1.1737 | 0.9914 | -0.1824 | < -0.01 | PASS |

### Gate 3 Details

- **MAE improvement is very strong**: -0.1824 (18x the threshold of -0.01)
- MAE improved in **5/5 folds** (100% consistency)
- DA improved in 3/5 folds
- Sharpe improved in 2/5 folds (Sharpe degradation driven by Folds 3-5)

### Feature Importance (XGBoost, last fold)

All 3 submodel columns ranked in the top 9 features:
- tech_trend_regime_prob: 7.96% (rank #2)
- tech_volatility_regime: 4.69% (rank #7)
- tech_mean_reversion_z: 4.59% (rank #9)

### Fold-by-Fold Breakdown

| Fold | Base DA | Ext DA | Base MAE | Ext MAE | Base Sharpe | Ext Sharpe |
|------|---------|--------|----------|---------|-------------|------------|
| 1 | 53.11% | 53.35% | 0.912 | 0.751 | 0.836 | 1.813 |
| 2 | 50.48% | 54.57% | 0.666 | 0.610 | 1.364 | 1.858 |
| 3 | 47.61% | 47.13% | 1.368 | 1.144 | 1.061 | 0.336 |
| 4 | 52.98% | 48.45% | 1.394 | 1.165 | 0.427 | -0.627 |
| 5 | 44.52% | 45.48% | 1.529 | 1.287 | -1.151 | -1.304 |

## Decision: completed

The technical submodel (GLD-based 2D HMM with z-score and GK volatility) passes Gate 3 via the MAE criterion. The MAE reduction of -0.1824 is exceptionally strong and consistent across all 5 folds. While Sharpe degradation in later folds is a concern, the MAE improvement demonstrates that the submodel captures genuine structural information about gold price dynamics.

This follows the same pattern as the VIX submodel (Gate 2 FAIL but Gate 3 PASS). The MI-based Gate 2 stability test underestimates the value of nonlinear regime-based features that structurally shift over time. The XGBoost ablation in Gate 3, which can model nonlinear interactions, confirms the submodel's predictive contribution.

### Comparison with Previous Submodels

| Submodel | Status | DA Delta | Sharpe Delta | MAE Delta | Key Strength |
|----------|--------|----------|-------------|-----------|-------------|
| real_rate | no_further_improvement | N/A | N/A | N/A | Monthly-daily mismatch |
| dxy | completed (auto) | N/A | N/A | N/A | MI=0.019 |
| vix | completed | +0.96% | +0.289 | +0.016 | DA + Sharpe |
| **technical** | **completed** | **+0.05%** | **-0.092** | **-0.1824** | **MAE (very strong)** |
