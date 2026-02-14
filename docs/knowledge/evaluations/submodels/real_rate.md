# Knowledge Base: real_rate Submodel Evaluation

## Attempt 1 (2026-02-14) -- Phase 1.5 Smoke Test

### Architecture
- Type: Autoencoder MLP (sliding window)
- Window size: 20
- Hidden dim: 64
- Latent dim: 4
- Dropout: 0.13
- Weight decay: 1e-6
- Epochs trained: 21
- Optuna trials: 5

### Results
- Gate 1: **FAIL** (overfit_ratio=2.69, autocorr>0.99)
- Gate 2: **PASS** (MI +18.5%, VIF 2.55, stability std 0.124)
- Gate 3: **FAIL** (DA +0.39%, Sharpe -0.026, MAE +0.165)

### Key Findings

1. **Information exists but extraction is noisy**: Gate 2 passing with 18.5% MI increase confirms that real rate dynamics contain useful information for gold prediction. The challenge is extracting it cleanly.

2. **AE-MLP overfits severely with 1766 training samples**: Overfit ratio 2.69 indicates the architecture is too expressive. The model memorizes training distribution rather than learning generalizable patterns.

3. **Near-identity mapping problem**: All 4 latent dimensions have autocorrelation > 0.995. The bottleneck (dim 4) is wide enough that the model learns to pass smoothed input through without meaningful compression. This is the primary failure mode.

4. **Dead dimension**: latent_2 contributes zero MI with gold returns. This dimension collapsed during training, likely caught in a local minimum.

5. **Noisy outputs hurt downstream**: Despite MI increase, the XGBoost ablation shows consistent MAE degradation. The submodel outputs add more noise than signal to downstream predictions.

### Hypotheses for Attempt 2

- **H1**: GRU encoder with tighter bottleneck (latent_dim=2) will force meaningful temporal compression, breaking the identity mapping.
- **H2**: Stronger regularization (dropout 0.3+, weight_decay 1e-3) will reduce overfit ratio below 1.5.
- **H3**: Temporal contrastive loss will push apart latent representations of different regimes, creating more discriminative features.
- **H4**: First-differencing the outputs may reduce autocorrelation and make the features more useful downstream.
- **H5**: If neural approach continues to struggle, classical HMM/Markov-switching may be more appropriate for sample size.

### Lessons Learned

- For small sample sizes (~1800), prefer smaller architectures with strong regularization
- Latent dim 4 is too large for extracting 2-3 meaningful dynamics from a single economic indicator
- Autocorrelation > 0.99 is a reliable indicator of identity mapping / insufficient compression
- MI increase does not guarantee downstream utility -- output quality matters as much as information content
- Smoke test with 5 Optuna trials is sufficient for pipeline verification but not for HPO convergence

## Attempt 2 (2026-02-14) -- Phase 1.5 Smoke Test

### Architecture
- Type: GRU Autoencoder (US-only)
- Optuna trials: 20 (all pruned)

### Results
- Gate 1: **N/A** (no completed trials)
- Gate 2: **N/A**
- Gate 3: **N/A**

### Key Findings
- All 20 Optuna trials pruned. GRU failed to converge on US-only data.
- Single-country approach with ~1800 samples is insufficient for recurrent architectures.
- Decision: Switch to multi-country data strategy.

## Attempt 3 (2026-02-14) -- Phase 1.5 Smoke Test

### Architecture
- Type: Multi-Country Transformer Autoencoder (GPU)
- Countries: 7 (US, Germany, UK, Canada, Switzerland, Norway, Sweden)
- Parameters: ~98K
- Optuna trials: 30
- Training: Kaggle GPU

### Results
- Gate 1: **PASS** (overfit_ratio=1.28)
- Gate 2: **PASS** (MI +23.8%, VIF <10, stability std <0.15)
- Gate 3: **FAIL** (DA -0.48%, MAE +0.42%, Sharpe unstable)

### Key Findings

1. **Multi-country data solved overfitting**: Overfit ratio dropped from 2.69 (Attempt 1) to 1.28. Additional training samples from 6 other countries provided sufficient regularization.

2. **Information content is high**: MI increase of 23.8% is the best across all attempts. The Transformer successfully captures cross-country co-movement patterns.

3. **Forward-fill step functions degrade MAE**: Monthly data forward-filled to daily frequency creates step-function outputs. XGBoost picks up on the step boundaries as artificial features, degrading MAE in all 5 CV folds.

4. **Only 2 of 5 latent dimensions carry information**: sem_2 and sem_3 have meaningful MI with gold returns; the other 3 are noise.

5. **98K parameters for 188 monthly samples**: Despite regularization, the model is overparameterized for the actual unique information content.

### Root Cause Analysis
The fundamental bottleneck is **monthly-to-daily frequency mismatch**. Multi-country real rate data updates monthly. Any interpolation to daily frequency (forward-fill, linear, etc.) creates synthetic values that add noise to daily gold return prediction.

## Attempt 4 (2026-02-14) -- Phase 1.5 Smoke Test [FINAL]

### Architecture
- Type: PCA (deterministic, no training)
- Components: 2 (explaining 82.8% of variance)
- Features: 7 multi-country rate changes
- Interpolation: Cubic spline (smooth daily values)
- Execution time: <10 seconds
- Parameters: 0

### Results
- Gate 1: **N/A** (deterministic PCA, no overfit ratio)
- Gate 2: **PASS** (MI +10.29%, VIF 1.33, stability std 0.135)
- Gate 3: **FAIL** (DA -1.96%, MAE +0.078, Sharpe -0.219, all 5 folds degraded on all metrics)

### Key Findings

1. **Cubic spline reduced but did not eliminate MAE degradation**: MAE delta improved from +0.42% (Attempt 3, forward-fill) to +0.078% (Attempt 4, cubic spline). But it still worsened in all 5 folds.

2. **Direction accuracy degraded more than Attempt 3**: DA dropped -1.96% vs -0.48% in Attempt 3. The simpler PCA approach captures less nuance than the Transformer.

3. **Low VIF confirms PCA adds orthogonal information**: PCA columns have VIF of 1.33 and 1.17, meaning they are nearly uncorrelated with existing base features. The problem is not redundancy but noise.

4. **MI decrease from Attempt 3**: MI increase dropped from +23.8% (Transformer, 5 dims) to +10.29% (PCA, 2 dims). PCA captures less information, but the core issue remains frequency mismatch.

### Decision: no_further_improvement

After 4 attempts (MLP, GRU, Transformer, PCA), the pattern is clear:
- Gate 2 always passes: multi-country real rate data contains information about gold returns
- Gate 3 always fails: this information cannot be extracted at daily frequency without introducing noise

The base `real_rate_real_rate` feature (daily US TIPS yield change) already captures the direct real rate signal available at daily frequency. No submodel output will be used.

---

## Cross-Attempt Summary

| Attempt | Method | Gate 1 | Gate 2 MI | Gate 3 DA | Gate 3 MAE | Decision |
|---------|--------|--------|-----------|-----------|------------|----------|
| 1 | MLP (US) | FAIL (2.69) | +18.5% | +0.39% | +0.165 | attempt+1 |
| 2 | GRU (US) | N/A | N/A | N/A | N/A | attempt+1 |
| 3 | Transformer (multi) | PASS (1.28) | +23.8% | -0.48% | +0.42% | attempt+1 |
| 4 | PCA + Spline | N/A | +10.29% | -1.96% | +0.078 | no_further_improvement |

## Key Lessons for Future Features

1. **Monthly-to-daily frequency gap**: Any feature with monthly update frequency will face this same challenge. For yield_curve, inflation_expectation, and other monthly FRED series, consider:
   - Using only the raw daily-frequency base feature (as we ended up doing for real_rate)
   - Designing features that model the expected path between monthly releases, rather than interpolating levels
   - Using change-point detection or regime indicators rather than continuous outputs

2. **Gate 2 is necessary but not sufficient**: Information content (MI) does not guarantee downstream utility. The format, frequency, and noise characteristics of the output matter as much as the information content.

3. **Simplicity baseline**: PCA (0 parameters, <10s) captured 43% of the MI that a 98K-parameter Transformer learned. For small monthly samples, classical methods are competitive.

4. **The autocorrelation dilemma**: Smooth interpolated outputs have high autocorrelation (0.99+), which is expected for slow-moving macro variables. But this means adjacent daily values are nearly identical, providing minimal new information to the daily meta-model.

5. **Multi-country data helps with overfitting but not frequency mismatch**: Going from 1 country to 7 reduced overfit ratio from 2.69 to 1.28, but did not solve the Gate 3 problem because the fundamental issue is temporal resolution, not sample size.
