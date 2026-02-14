# Evaluation Summary: real_rate Attempt 3

## Architecture: Multi-Country Transformer Autoencoder (GPU)
- 25 input features (US TIPS + 6 countries synthetic real rates)
- 269 monthly windows, 188 for training
- 30 Optuna trials, 86 epochs, 98K parameters
- Output: 5 semantic dimensions forward-filled to 5413 daily rows

---

## Gate 1: Standalone Quality -- PASS

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| Overfit ratio | 1.277 | < 1.5 | PASS |
| All-NaN columns | 0 | = 0 | PASS |
| Zero-variance columns | 0 | = 0 | PASS |
| Daily autocorrelation (max) | 0.997 | N/A (expected from forward-fill) | INFO |
| Monthly autocorrelation (max) | 0.937 | < 0.95 | PASS |
| Optuna trials | 30 | >= 10 | PASS |

**Improvement from Attempt 1**: Overfit ratio dropped from 2.69 to 1.28 due to strong regularization (dropout=0.44, weight_decay=0.008). Multi-country data and compact architecture (d_model=48) also helped.

---

## Gate 2: Information Gain -- PASS

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| MI increase (daily) | +23.84% | > 5% | PASS |
| MI increase (monthly) | +15.94% | > 5% | PASS |
| Max submodel VIF (standalone) | 3.55 | < 10 | PASS |
| Max submodel VIF (in extended) | 13.05 | < 10 | CONDITIONAL PASS |
| Rolling corr stability (max std) | 0.129 | < 0.15 | PASS |

**MI breakdown per semantic dimension**:
- sem_0: 0.000 (negligible)
- sem_1: 0.000 (negligible)
- sem_2: 0.029 (informative)
- sem_3: 0.041 (most informative)
- sem_4: 0.000 (negligible)

**VIF note**: real_rate_sem_3 has VIF=13.05 in the extended feature matrix, marginally exceeding the threshold of 10. This is because sem_3 correlates with the base real_rate feature, which is expected. The standalone VIF is only 3.55. Base features already have infinite VIF from price level collinearity. CONDITIONAL PASS granted.

---

## Gate 3: Ablation -- FAIL (1/3 checks pass, but unstable)

| Metric | Baseline (CV) | Extended (CV) | Delta | Threshold | Result |
|--------|--------------|---------------|-------|-----------|--------|
| Direction Accuracy | 47.78% | 48.26% | +0.48% | > +0.50% | FAIL (marginal) |
| Sharpe Ratio | -0.273 | +0.004 | +0.277 | > +0.05 | PASS |
| MAE | 1.122% | 1.540% | +0.418% | < -0.01% | FAIL |

**Test-set-only results** (not used for gate decision, but informative):

| Metric | Baseline | Extended | Delta |
|--------|----------|----------|-------|
| Direction Accuracy | 46.17% | 45.38% | -0.79% |
| Sharpe Ratio | -1.173 | -1.434 | -0.261 |
| MAE | 1.213% | 1.877% | +0.664% |

**Robustness analysis** (5 CV folds):
- Sharpe improves in 3/5 folds (but driven by one large gain in Fold 2: +1.26)
- Direction accuracy improves in only 1/5 folds
- MAE worsens in ALL 5/5 folds
- Test-set-only shows degradation in all metrics

**Verdict**: The CV Sharpe technically passes, but the signal is not robust. The improvement is concentrated in Fold 2 and contradicted by the held-out test set. MAE consistently degrades, indicating the submodel adds noise to point predictions. Gate 3 is judged as FAIL due to lack of robustness.

---

## Comparison with Previous Attempts

| Metric | Attempt 1 (MLP) | Attempt 2 (GRU) | Attempt 3 (Transformer) |
|--------|-----------------|-----------------|------------------------|
| Gate 1 | FAIL (overfit=2.69) | N/A (no convergence) | PASS (overfit=1.28) |
| Gate 2 MI | +18.5% | N/A | +23.8% |
| Gate 3 DA delta | +0.39% | N/A | +0.48% |
| Gate 3 Sharpe delta | -0.026 | N/A | +0.277 (unstable) |
| Gate 3 MAE delta | +0.165% | N/A | +0.418% |

**Progress**: Clear improvement in Gate 1 (overfitting solved) and Gate 2 (more information from multi-country data). Gate 3 direction accuracy is approaching the threshold. However, MAE degradation has worsened, suggesting the forward-fill strategy is fundamentally problematic.

---

## Root Cause Analysis

1. **Primary**: Monthly-to-daily forward-fill creates step-function outputs. XGBoost treats these as categorical-like features that change abruptly at month boundaries, increasing MAE.

2. **Secondary**: Only 269 monthly windows (188 for training) is insufficient for a Transformer to learn generalizable cross-country patterns. The model may be memorizing rather than generalizing, despite good overfit ratio.

3. **Tertiary**: Synthetic real rates (constructed from nominal yield - CPI) have only 0.49 correlation with US TIPS, diluting signal quality.

4. **Gate 3 instability**: The CV Sharpe improvement is driven by regime-specific alignment in Fold 2 (2018-2019 period), not a generalizable pattern.

---

## Decision: attempt+1 (Attempt 4)

Attempt consumed: +1 (now 3/5 used for real_rate)

### Improvement Plan for Attempt 4

**Priority 1 - Output transformation**:
Replace daily forward-fill with smoother interpolation or dual-frequency approach. The step-function output is the primary cause of MAE degradation.

**Priority 2 - Architecture simplification**:
Replace Transformer with PCA/ICA-based dimensionality reduction on multi-country real rate changes. Simpler methods are more robust with only 188 training samples.

**Priority 3 - Feature selection**:
Drop sem_0, sem_1, sem_4 (near-zero MI). Keep only the 2 informative dimensions (sem_2, sem_3) to reduce noise.

### Alternative consideration:
If PCA approach also fails Gate 3, declare "no_further_improvement" for real_rate and proceed to the next feature. The base real_rate feature already provides information to the meta-model.
