# Real Rate Attempt 3: Multi-Country Transformer (GPU-Enabled)

**Generated**: 2026-02-14
**Agent**: builder_model
**Status**: Ready for Kaggle submission

## Overview

This attempt uses a **Transformer-based autoencoder** to extract global monetary policy regime context from multi-country interest rate data. GPU-enabled for faster training.

## Key Design Changes from Attempt 2

| Aspect | Attempt 2 (GRU) | Attempt 3 (Transformer) |
|--------|-----------------|------------------------|
| Architecture | GRU Autoencoder | Transformer Autoencoder |
| Data source | US-only | 7 countries (multi-country) |
| Sample size | ~2,463 daily | ~253 monthly windows |
| Result | All trials pruned (non-convergence) | Expected to converge with GPU |
| GPU | Disabled | **Enabled** (user request) |

## Architecture Specifications

- **Input**: [batch, seq_len=12-24, n_features=25]
- **Model**: Compact Transformer with pre-norm layers
- **Latent**: [batch, latent_dim=4-6]
- **Output**: Daily semantic features (forward-filled from monthly)

### Model Size Range
- Minimum: ~8,300 params (d_model=24, 2 layers)
- Maximum: ~44,500 params (d_model=48, 3 layers)

## Features (25 total)

### US Features (2)
- `us_tips_level`: DFII10 real interest rate
- `us_tips_change`: Month-to-month change

### Multi-Country Features (6 countries × 3 = 18)
Countries: Germany, UK, Canada, Switzerland, Norway, Sweden

For each country:
- `{country}_nominal_level`: 10Y nominal yield
- `{country}_nominal_change`: Month-to-month change
- `{country}_cpi_lagged`: CPI YoY (1-month lag)

### Cross-Country Aggregates (4)
- `yield_dispersion`: Std of nominal yields across countries
- `yield_change_dispersion`: Std of yield changes
- `mean_cpi_change`: Mean CPI change
- `us_vs_global_spread`: US vs others spread

### Context (1)
- `vix_monthly`: Monthly VIX average

## Hyperparameter Search Space

```python
window_size: {12, 18, 24} months
d_model: {24, 32, 48}
nhead: {2, 4}
num_encoder_layers: {2, 3}
latent_dim: {4, 5, 6}
dropout: [0.3, 0.5]
learning_rate: [5e-5, 5e-4] (log-uniform)
weight_decay: [1e-3, 5e-2] (log-uniform)
batch_size: {16, 32}
```

**Optuna Settings**:
- 30 trials
- 3600s (1 hour) timeout
- MedianPruner(n_startup_trials=7, n_warmup_steps=30)
- TPESampler(seed=42)

## GPU Configuration

**CRITICAL**: This kernel **requires GPU**.

```json
{
  "enable_gpu": true
}
```

The code automatically detects CUDA and uses GPU if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Training Configuration

- **Optimizer**: AdamW (better for Transformers than Adam)
- **Scheduler**: CosineAnnealingWarmRestarts(T_0=50, T_mult=2)
- **Gradient clipping**: 1.0 (essential for Transformers)
- **Early stopping**: 20 epochs patience
- **Max epochs**: 200

## Output Files

After Kaggle execution, the following files will be in the output directory:

### 1. submodel_output.csv
Daily latent semantic features (forward-filled from monthly)

Format:
```csv
date,real_rate_sem_0,real_rate_sem_1,...,real_rate_sem_N
2015-03-03,0.012,-0.023,0.005,0.018
2015-03-04,0.012,-0.023,0.005,0.018
...
```

- First ~13-25 rows will be NaN (window size + 1 for first-difference)
- Values change only at month boundaries (step function)
- N = latent_dim (4-6)

### 2. model.pt
Trained model weights + config

Contains:
- `model_state`: PyTorch state dict
- `config`: Best hyperparameters from Optuna
- `train_mean`: Feature means (for inference)
- `train_std`: Feature stds (for inference)

### 3. training_result.json
Training metadata + metrics

Contains:
- Best hyperparameters
- Train/val loss, overfit ratio
- Optuna statistics
- Output shape and columns
- Data split info

## Expected Performance

Based on architect design:

- **Overfit ratio**: 1.1-1.3 (target: <1.5)
- **Autocorrelation**: <0.85 (monthly resolution helps)
- **Gate 2 MI increase**: >10% (cross-country signal)
- **Gate 3**: DA +0.5%, Sharpe +0.05

## Kaggle Submission

```bash
# From project root
kaggle kernels push -p notebooks/real_rate_3/
```

Check status:
```bash
kaggle kernels status bigbigzabuton/gold-real-rate-3
```

Fetch results (after completion):
```bash
kaggle kernels output bigbigzabuton/gold-real-rate-3 \
  -p data/submodel_outputs/real_rate/
```

## Key Implementation Details

### 1. Monthly to Daily Expansion
```
Monthly latents → First-difference → Forward-fill to daily
```

This creates step-function features that update once per month, suitable as regime indicators.

### 2. No Synthetic Real Rates
Unlike initial researcher proposal, we do NOT pre-compute synthetic real rates (Nominal - CPI). Instead, we provide nominal and CPI as separate inputs, letting the Transformer learn the relationship.

Reason: Synthetic rates have only 0.49 correlation with true TIPS rates (fact-checked).

### 3. Pre-Norm Transformer
Uses `norm_first=True` in TransformerEncoderLayer for more stable training on small datasets.

### 4. Gradient Clipping
Essential for Transformer stability:
```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5. CUDA Optimization
Model and data are moved to GPU:
```python
model.to(device)
batch = batch.to(device)
```

## Risk Mitigation

### Small Sample Size (~253 windows)
- Compact model: 8-45K params
- Aggressive regularization: dropout 0.3-0.5, weight_decay 1e-3 to 5e-2
- Pre-norm architecture
- Optuna finds regularization-performance tradeoff

### High Cross-Country Correlation
- Include dispersion and spread features
- Transformer attention learns to focus on divergence periods
- Reconstruction loss on 28 features forces country-specific learning

### Monthly Resolution
- Forward-fill to daily creates regime indicators
- Meta-model can use alongside daily base features
- If insufficient, Attempt 4 can use hybrid (daily US + monthly multi-country)

## Notes

- **GPU required**: User explicitly requested GPU support
- **Self-contained**: All data fetching in notebook (no external files)
- **Kaggle Secrets**: FRED_API_KEY must be configured in Kaggle settings
- **Estimated runtime**: 30-60 minutes with GPU (potentially 2-3 hours on CPU)
