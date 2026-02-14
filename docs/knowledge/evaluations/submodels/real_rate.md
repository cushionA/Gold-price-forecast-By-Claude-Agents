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
