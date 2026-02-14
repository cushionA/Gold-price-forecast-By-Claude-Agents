# Evaluation Summary: real_rate attempt 1

Phase: 1.5_smoke_test | Date: 2026-02-14

## Gate 1: Standalone Quality -- FAIL

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| Overfit ratio | 2.6919 | < 1.5 | FAIL |
| All-NaN columns | 0 | 0 | PASS |
| Zero-variance columns | 0 | 0 | PASS |
| Extra NaN rows | 0 | 0 | PASS |
| Autocorr (latent_0) | 0.9956 | < 0.99 | FAIL |
| Autocorr (latent_1) | 0.9970 | < 0.99 | FAIL |
| Autocorr (latent_2) | 0.9962 | < 0.99 | FAIL |
| Autocorr (latent_3) | 0.9958 | < 0.99 | FAIL |
| Optuna trials | 5 | >= 10 | PASS (warning) |

**Gate 1 Failures:**
- **Overfit ratio 2.69**: val_loss (1.10) is 2.69x train_loss (0.41). The autoencoder memorizes training patterns without generalizing.
- **Autocorrelation > 0.99 on all latent dims**: The latent representations change minimally between consecutive timesteps, suggesting the model is performing near-identity mapping (copying smoothed input) rather than extracting meaningful regime/state features.

## Gate 2: Information Gain -- PASS

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| MI increase | 18.49% | > 5% | PASS |
| Max submodel VIF | 2.55 | < 10 | PASS |
| Max rolling corr std | 0.124 | < 0.15 | PASS |

**Gate 2 Details:**
- MI base sum: 0.3042, extended sum: 0.3605 (+18.5%)
- Per-feature MI: latent_0=0.017, latent_1=0.018, latent_2=0.000, latent_3=0.018
- latent_2 contributes zero MI -- potentially redundant dimension
- VIF values are excellent (2.01-2.55), no multicollinearity concern
- Rolling correlation stability is within bounds (std 0.112-0.124)

**Note:** Despite passing Gate 2, the MI contribution per feature is small (0.017-0.018), and one dimension (latent_2) contributes nothing. Reducing latent_dim from 4 to 3 or 2 may be warranted.

## Gate 3: Ablation -- FAIL

| Metric | Baseline | + Submodel | Delta | Threshold | Result |
|--------|----------|------------|-------|-----------|--------|
| Direction Accuracy | 47.79% | 48.17% | +0.39% | > +0.50% | FAIL |
| Sharpe Ratio | -0.2793 | -0.3050 | -0.0257 | > +0.05 | FAIL |
| MAE | 1.1359% | 1.3010% | +0.1651% | < -0.01% | FAIL |

**Gate 3 Details (5-fold TimeSeriesSplit):**

| Fold | Base DA | Ext DA | Base Sharpe | Ext Sharpe | Base MAE | Ext MAE |
|------|---------|--------|-------------|------------|----------|---------|
| 1 | 51.08% | 49.40% | 0.895 | 0.392 | 0.710 | 0.800 |
| 2 | 45.76% | 49.39% | -1.020 | -0.153 | 0.875 | 0.975 |
| 3 | 49.64% | 50.12% | 0.213 | 0.583 | 1.005 | 1.170 |
| 4 | 48.08% | 48.08% | -0.708 | -1.264 | 1.282 | 1.662 |
| 5 | 44.36% | 43.88% | -0.776 | -1.083 | 1.807 | 1.897 |

**Key observations:**
- Direction accuracy is marginally improved (+0.39%) but below the +0.50% threshold
- MAE consistently worsens across all folds -- the noisy latent representations add prediction error
- Sharpe ratio degrades, suggesting the submodel outputs introduce harmful noise into trading signals
- Folds 2 and 3 show improvement, but folds 4 and 5 show degradation -- the submodel is not robust across time periods

## Overall Decision: FAIL -- Proceed to attempt 2

**Root Cause Analysis:**

1. **Severe overfitting (primary issue)**: The AE-MLP architecture with window_size=20, hidden_dim=64 is too expressive for the training data (1,766 samples). It memorizes training patterns and produces smoothed near-identity outputs rather than extracting latent dynamics.

2. **Near-identity mapping**: Autocorrelation > 0.995 on all latents indicates the model's bottleneck is not tight enough to force meaningful compression. The latent space preserves temporal ordering rather than capturing regime shifts or state transitions.

3. **Latent dimension 2 is dead**: real_rate_latent_2 contributes zero MI with gold returns. This dimension may have collapsed during training.

4. **Noisy output hurts downstream**: While Gate 2 shows some MI increase, the noise level is high enough that XGBoost cannot leverage the information -- MAE gets worse, not better.

## Improvement Recommendations for Attempt 2

**Priority 1 -- Reduce overfitting (Gate 1 fix):**
- Increase dropout from 0.13 to 0.3-0.5
- Reduce hidden_dim from 64 to 32
- Increase weight_decay by 2-3 orders of magnitude (1e-6 -> 1e-3 to 1e-4)
- Consider early stopping with patience based on val_loss (not just fixed epochs)

**Priority 2 -- Break identity mapping (Gate 1 autocorrelation fix):**
- Add temporal contrastive loss (push apart latent representations of different regimes)
- Use a GRU/LSTM encoder instead of MLP to capture temporal dynamics
- Increase window_size from 20 to 40-60 to capture longer regime transitions
- Reduce latent_dim from 4 to 2 (tighter bottleneck forces meaningful compression)

**Priority 3 -- Improve downstream utility (Gate 3 fix):**
- Apply output postprocessing: first-difference the latent representations (removes autocorrelation)
- Or apply rolling z-score normalization to outputs
- Focus output on regime classification rather than continuous latent values

**Approach for attempt 2:** Focus on model change -- GRU encoder with contrastive loss, tighter bottleneck (latent_dim=2), stronger regularization (dropout=0.3, weight_decay=1e-3). This addresses all three failure modes simultaneously.
