# Evaluation Summary: dxy attempt 1

**Method**: GaussianHMM + Momentum Z-Score + Volatility Z-Score
**Date**: 2026-02-18 00:09
**Samples**: 2357 (aligned, no NaN)

## Gate 1: Standalone Quality -- PASS

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| All-NaN columns | 0 | 0 | PASS |
| Constant columns | 0 | 0 | PASS |
| Autocorr dxy_regime_prob | 0.344728 | < 0.99 | PASS |
| Autocorr dxy_momentum_z | 0.922912 | < 0.99 | PASS |
| Autocorr dxy_vol_z | 0.970634 | < 0.99 | PASS |
| Optuna trials | 30 | >= 10 | PASS |
| Overfit ratio | N/A (unsupervised) | < 1.5 | PASS (N/A) |

**Warning**: dxy_regime_prob is highly skewed (mean=0.000582, max=0.1842). Only 0.9% of days above 0.01. May contribute limited information.

## Gate 2: Information Gain -- PASS

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| MI increase | 6.24% | > 5% | PASS |
| VIF dxy_regime_prob | 1.0997 | < 10 | PASS |
| VIF dxy_momentum_z | 1.4680 | < 10 | PASS |
| VIF dxy_vol_z | 2.9231 | < 10 | PASS |
| Max rolling corr std | 0.149159 | < 0.15 | PASS |

Individual MI contributions:
- dxy_regime_prob: 0.006568
- dxy_momentum_z: 0.000000
- dxy_vol_z: 0.028811

## Gate 3: Ablation -- PASS

| Metric | Baseline | Extended | Delta | Threshold | Folds Improved | Result |
|--------|----------|----------|-------|-----------|---------------|--------|
| Direction Accuracy | 51.77% | 52.49% | +0.73pp | +0.5pp | 4/5 | PASS |
| Sharpe | 0.8647 | 1.1193 | +0.2546 | +0.05 | 4/5 | PASS |
| MAE | 0.729305 | 0.737725 | +0.008419 | -0.01 | 1/5 | FAIL |

### Per-Fold Results

| Fold | DA Delta | Sharpe Delta | MAE Delta |
|------|----------|-------------|-----------|
| 1 | +0.77pp | +0.3366 | +0.011594 |
| 2 | +3.88pp | +0.8049 | -0.021954 |
| 3 | +0.77pp | +0.3410 | +0.005057 |
| 4 | +1.02pp | +0.6075 | +0.012398 |
| 5 | -2.81pp | -0.8172 | +0.035000 |

## Decision: COMPLETED

DXY submodel attempt 1 **passes Gate 3** via: DA (+0.73pp), Sharpe (+0.2546).
The HMM-based DXY features provide measurable improvement to the meta-model.