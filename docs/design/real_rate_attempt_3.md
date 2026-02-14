# Submodel Design Document: real_rate (Attempt 3)

**Feature**: real_rate (Multi-Country Interest Rate Dynamics)
**Attempt**: 3
**Phase**: Smoke Test (fundamental architecture change)
**Architect**: Claude Opus 4.6
**Date**: 2026-02-14

---

## 0. Fact-Check Results

### FRED Series IDs

#### Nominal 10Y Yields (OECD via FRED) -- ALL CONFIRMED
- IRLTLT01USM156N (US) -> Confirmed. 1953-04 to 2025-12. Last: 4.14%
- IRLTLT01DEM156N (DE) -> Confirmed. 1956-05 to 2025-12. Last: 2.81%
- IRLTLT01GBM156N (UK) -> Confirmed. 1960-01 to 2025-12. Last: 4.48%
- IRLTLT01CAM156N (CA) -> Confirmed. 1955-01 to 2025-12. Last: 3.39%
- IRLTLT01CHM156N (CH) -> Confirmed. 1955-01 to 2025-11. Last: 0.19%
- IRLTLT01NOM156N (NO) -> Confirmed. 1985-01 to 2025-12. Last: 4.09%
- IRLTLT01SEM156N (SE) -> Confirmed. 1986-12 to 2025-12. Last: 2.82%

#### CPI YoY (OECD via FRED) -- ALL CONFIRMED, END DATES NOTED
- CPALTT01USM659N (US) -> Confirmed. Ends 2025-04. Last: 2.31%
- CPALTT01DEM659N (DE) -> Confirmed. Ends 2025-03. Last: 2.19%
- CPALTT01GBM659N (UK) -> Confirmed. Ends 2025-03. Last: 3.40%
- CPALTT01CAM659N (CA) -> Confirmed. Ends 2025-03. Last: 2.32%
- CPALTT01CHM659N (CH) -> Confirmed. Ends 2025-04. Last: 0.03%
- CPALTT01NOM659N (NO) -> Confirmed. Ends 2025-04. Last: 2.47%
- CPALTT01SEM659N (SE) -> Confirmed. Ends 2025-03. Last: 0.47%
- CPALTT01JPM659N (JP) -> OUTDATED. Ends 2021-06 (researcher said 2022-04 -- incorrect). EXCLUDED.

#### Reference Series -- CONFIRMED
- DFII10 (US 10Y TIPS) -> Confirmed. 2003-01-02 to 2026-02-12. Last: 1.80%
- VIXCLS -> Confirmed. 1990-01-02 to 2026-02-12. Last: 20.82

### CRITICAL: Synthetic Real Rate Validity -- FAILED

**Researcher claimed**: "Correlation >0.90-0.95, RMSE ~0.5-1.0%"

**Actual measured values**:
- Pearson correlation (US synthetic vs DFII10): **0.49** (FAIL, expected >0.85)
- RMSE: **1.73%** (FAIL, expected <1.0%)
- Change correlation (month-to-month): **0.25** (near useless)
- Synthetic change captures only **1%** of TIPS-gold signal

**Root cause**: DFII10 = DGS10 - T10YIE (breakeven inflation, forward-looking). Synthetic = Nominal - CPI_YoY (backward-looking). CPI_YoY vs T10YIE correlation is only 0.60. The inflation expectation gap makes synthetic real rates a poor proxy for TIPS-equivalent real rates.

**Impact on design**: Cannot use pre-computed synthetic real rates as primary features. Must use nominal yields + CPI as SEPARATE inputs and let the Transformer learn the implicit relationships.

### Cross-Country Dynamics -- PARTIALLY CONFIRMED

- Cross-country nominal yield change correlations: 0.74-0.88 (high, US-dominated)
- Gold return vs cross-country yield level dispersion change: -0.15 (marginal but present)
- Gold return vs TIPS change: -0.37 (strong negative)
- Gold return vs nominal yield change: -0.24 (moderate negative)
- Cross-country patterns exist but are dominated by common global factor

### Sample Size
- Common period: 2003-01 to 2025-02 (265 months, constrained by CPI end dates)
- Stacked: 7 countries x 265 = 1,855 monthly samples (US uses DFII10 directly)
- With US daily + monthly context: 5,782 samples (hybrid approach)
- Researcher claimed 2,192 -- slightly high (actual is 2,120 for 8 countries, or 1,855 for 7 synthetic + separate US)

### Memory Feasibility -- CONFIRMED
- 4-layer Transformer, d_model=64: ~403K params, ~5 MB training memory
- 4-layer Transformer, d_model=128: ~1.6M params, ~18 MB training memory
- Both well within Kaggle CPU 16GB limit. No GPU needed.

### Methodology Assessment
- Transformer for cross-country patterns: APPROPRIATE, but needs careful design given high inter-country correlation
- Synthetic real rates: INAPPROPRIATE as pre-computed feature (correlation too low)
- Separate nominal + CPI inputs: APPROPRIATE (let model learn relationship)
- Autoencoder (unsupervised): APPROPRIATE given previous attempts
- Researcher's 85% feasibility claim: DOWNGRADED to 65% due to synthetic rate failure

---

## 1. Overview

### Purpose

Extract **global monetary policy regime context** from multi-country interest rate dynamics. The core hypothesis is that gold reacts to the *pattern* of global rate movements -- not just the US rate level, which the baseline already has.

### Why Multi-Country (Root Cause of Attempt 1/2 Failures)

| Attempt | Architecture | Root Cause of Failure |
|---------|-------------|----------------------|
| 1 | MLP Autoencoder (US-only) | Identity mapping. 8 US features compressed to 4 latent dims was too easy. Overfit ratio 2.69, autocorr >0.995 |
| 2 | GRU Autoencoder (US-only) | All 20 Optuna trials pruned. GRU could not find a stable latent representation from US-only data. Insufficient information content. |
| 3 (this) | Transformer (multi-country) | Address root cause: add cross-country information that provides genuine new signal |

The fundamental problem with US-only approaches: the baseline XGBoost already sees the raw DFII10 level. Any US-only submodel must extract *additional* information beyond the level, but there is limited additional structure in a single time series (Attempt 1 learned identity, Attempt 2 could not converge).

Multi-country data provides genuinely new information: whether US rate movements are part of a global pattern (coordinated) or US-specific (divergent). This distinction is invisible to single-country models.

### Method and Rationale

**Transformer Autoencoder** on multi-country interest rate data (nominal yields + CPI as separate inputs).

Why Transformer over GRU for this task:
1. **Attention mechanism** explicitly models cross-timestep relationships, ideal for detecting when countries move together vs apart
2. **No recurrence** -- avoids the convergence issues that killed Attempt 2's GRU
3. **Parallel processing** -- faster training on monthly data (short sequences)
4. **Interpretable** -- attention weights reveal which countries/timepoints matter

Why separate nominal + CPI instead of pre-computed synthetic rates:
- Fact-check showed synthetic real rate (Nominal - CPI_YoY) has only 0.49 correlation with true TIPS rate
- Letting the Transformer see both nominal and CPI separately preserves information
- The model can learn the non-trivial relationship between backward-looking CPI and forward-looking rate dynamics

### Expected Effect

- Overfit ratio: 1.1-1.3 (2,120 stacked samples, Transformer regularization)
- Autocorrelation of outputs: <0.85 (monthly resolution naturally reduces autocorr)
- Gate 2 MI increase: >10% (cross-country patterns provide new information)
- Gate 3: DA +0.5%, Sharpe +0.05 (regime context helps gold prediction)

---

## 2. Data Specification

### Primary Data Sources

| Data | Source | Series ID | Frequency | Period |
|------|--------|-----------|-----------|--------|
| US Real Rate | FRED | DFII10 | Daily (resample to monthly) | 2003-01 to 2026-02 |
| US Nominal 10Y | FRED/OECD | IRLTLT01USM156N | Monthly | 2003-01 to 2025-12 |
| DE Nominal 10Y | FRED/OECD | IRLTLT01DEM156N | Monthly | 2003-01 to 2025-12 |
| UK Nominal 10Y | FRED/OECD | IRLTLT01GBM156N | Monthly | 2003-01 to 2025-12 |
| CA Nominal 10Y | FRED/OECD | IRLTLT01CAM156N | Monthly | 2003-01 to 2025-12 |
| CH Nominal 10Y | FRED/OECD | IRLTLT01CHM156N | Monthly | 2003-01 to 2025-11 |
| NO Nominal 10Y | FRED/OECD | IRLTLT01NOM156N | Monthly | 2003-01 to 2025-12 |
| SE Nominal 10Y | FRED/OECD | IRLTLT01SEM156N | Monthly | 2003-01 to 2025-12 |
| US CPI YoY | FRED/OECD | CPALTT01USM659N | Monthly | 2003-01 to 2025-04 |
| DE CPI YoY | FRED/OECD | CPALTT01DEM659N | Monthly | 2003-01 to 2025-03 |
| UK CPI YoY | FRED/OECD | CPALTT01GBM659N | Monthly | 2003-01 to 2025-03 |
| CA CPI YoY | FRED/OECD | CPALTT01CAM659N | Monthly | 2003-01 to 2025-03 |
| CH CPI YoY | FRED/OECD | CPALTT01CHM659N | Monthly | 2003-01 to 2025-04 |
| NO CPI YoY | FRED/OECD | CPALTT01NOM659N | Monthly | 2003-01 to 2025-04 |
| SE CPI YoY | FRED/OECD | CPALTT01SEM659N | Monthly | 2003-01 to 2025-03 |
| VIX | FRED | VIXCLS | Daily (resample to monthly) | 2003-01 to 2026-02 |
| Gold Price | Yahoo | GC=F | Daily (resample to monthly) | 2003-01 to 2026-02 |

### Context Data (for alignment and calendar)

| Data | Source | Purpose |
|------|--------|---------|
| Gold (GC=F) | Yahoo Finance | Trading calendar + target alignment |

### Preprocessing Pipeline

```
1. Fetch all FRED series
2. Resample daily series (DFII10, VIXCLS) to month-start frequency (mean)
3. Apply 1-month CPI lag: cpi_lagged[t] = cpi[t-1] for all countries
4. Compute per-country features:
   a. nominal_level: Raw nominal 10Y yield
   b. nominal_change: Month-to-month change in nominal yield
   c. cpi_yoy_lagged: 1-month lagged CPI YoY
   d. cpi_change: Change in CPI YoY (lagged)
   e. implied_real_approx: nominal_level - cpi_yoy_lagged (synthetic, for reference only)
5. For US, also include:
   a. tips_level: DFII10 monthly average
   b. tips_change: Month-to-month change in DFII10
6. Compute cross-country aggregate features:
   a. yield_dispersion: std of nominal yields across 7 countries
   b. yield_change_dispersion: std of nominal yield changes across 7 countries
   c. mean_cpi_change: mean of CPI YoY changes across 7 countries
   d. us_vs_global_spread: US nominal - mean(other 6 nominal)
7. Include VIX monthly average as context feature
8. Align all to common date range: 2003-02 to 2025-02 (after CPI lag)
9. Forward-fill missing values with max 2-month limit
10. Drop any remaining NaN rows
11. Standardize features (fit on train split only, apply to all)
12. Create sliding windows of length W months
13. Time-series split: train 70% / val 15% / test 15%
```

### Expected Sample Count

- Common period after CPI lag: 2003-02 to 2025-02 = 265 months
- After windowing (W=12): 253 monthly windows
- After windowing (W=24): 241 monthly windows
- Train (70%): ~177 / Val (15%): ~38 / Test (15%): ~38 (for W=12)

**CRITICAL NOTE ON SAMPLE SIZE**: 253 samples is very small for a Transformer. This is addressed in Section 8 (Risk Mitigation) with specific architecture constraints.

### Input Feature Matrix (per timestep)

| # | Feature | Source | Interpretation |
|---|---------|--------|----------------|
| 1 | us_tips_level | DFII10 monthly avg | US true real rate level |
| 2 | us_tips_change | DFII10 monthly diff | US real rate momentum |
| 3 | us_nominal | IRLTLT01USM156N | US nominal yield |
| 4 | us_nominal_change | IRLTLT01USM156N diff | US nominal yield momentum |
| 5 | us_cpi_lagged | CPALTT01USM659N shifted | US backward inflation |
| 6 | de_nominal | IRLTLT01DEM156N | Germany nominal yield |
| 7 | de_nominal_change | IRLTLT01DEM156N diff | Germany yield momentum |
| 8 | de_cpi_lagged | CPALTT01DEM659N shifted | Germany inflation |
| 9 | uk_nominal | IRLTLT01GBM156N | UK nominal yield |
| 10 | uk_nominal_change | IRLTLT01GBM156N diff | UK yield momentum |
| 11 | uk_cpi_lagged | CPALTT01GBM659N shifted | UK inflation |
| 12 | ca_nominal | IRLTLT01CAM156N | Canada nominal yield |
| 13 | ca_nominal_change | IRLTLT01CAM156N diff | Canada yield momentum |
| 14 | ca_cpi_lagged | CPALTT01CAM659N shifted | Canada inflation |
| 15 | ch_nominal | IRLTLT01CHM156N | Switzerland nominal yield |
| 16 | ch_nominal_change | IRLTLT01CHM156N diff | Switzerland yield momentum |
| 17 | ch_cpi_lagged | CPALTT01CHM659N shifted | Switzerland inflation |
| 18 | no_nominal | IRLTLT01NOM156N | Norway nominal yield |
| 19 | no_nominal_change | IRLTLT01NOM156N diff | Norway yield momentum |
| 20 | no_cpi_lagged | CPALTT01NOM659N shifted | Norway inflation |
| 21 | se_nominal | IRLTLT01SEM156N | Sweden nominal yield |
| 22 | se_nominal_change | IRLTLT01SEM156N diff | Sweden yield momentum |
| 23 | se_cpi_lagged | CPALTT01SEM659N shifted | Sweden inflation |
| 24 | yield_dispersion | cross-country std | Policy coordination measure |
| 25 | yield_change_disp | cross-country change std | Change coordination |
| 26 | mean_cpi_change | cross-country CPI change mean | Global inflation trend |
| 27 | us_vs_global_spread | US nominal - mean(others) | US exceptionalism |
| 28 | vix_monthly | VIXCLS monthly avg | Risk regime |

**Total input features per timestep: 28**

---

## 3. Model Architecture (PyTorch)

### Architecture: Compact Transformer Autoencoder

Given the small sample size (~253 windows), the architecture must be deliberately constrained. A standard 4-6 layer Transformer would overfit on this data. The design prioritizes regularization and simplicity.

```
Input: [batch, seq_len, 28]  (monthly windows, 28 features per month)

INPUT PROJECTION:
  Linear(28, d_model) + LayerNorm + Dropout

POSITIONAL ENCODING:
  Sinusoidal positional encoding (months are evenly spaced)

TRANSFORMER ENCODER:
  num_layers=2-4 TransformerEncoderLayer
  Each layer:
    MultiHeadAttention(d_model, nhead) + Dropout + LayerNorm (pre-norm)
    FeedForward(d_model, d_ff=2*d_model) + Dropout + LayerNorm (pre-norm)

TEMPORAL AGGREGATION:
  Mean pooling across time dimension: [batch, seq_len, d_model] -> [batch, d_model]

LATENT PROJECTION:
  Linear(d_model, latent_dim) + Tanh

LATENT: [batch, latent_dim]  (4-6 dimensions, the submodel output)

DECODER:
  Linear(latent_dim, d_model) + ReLU + Dropout
  Repeat across seq_len: [batch, seq_len, d_model]
  Add sinusoidal positional encoding
  num_layers=2 TransformerDecoderLayer (simpler than encoder)
  Linear(d_model, 28)

Output (reconstruction): [batch, seq_len, 28]
```

### PyTorch Pseudocode

```python
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for monthly time series."""
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiCountryRateTransformer(nn.Module):
    """
    Compact Transformer Autoencoder for multi-country interest rate dynamics.

    Design principles:
    - Small model to prevent overfitting on ~253 samples
    - Pre-norm Transformer (more stable training)
    - Mean pooling (not CLS token) for temporal aggregation
    - Tanh activation on latent to bound output range
    - Separate nominal + CPI inputs (NOT pre-computed synthetic rates)

    Input: [batch, seq_len, 28] (monthly windows of multi-country features)
    Output latent: [batch, latent_dim] (4-6 semantic features)
    Output reconstruction: [batch, seq_len, 28]
    """
    def __init__(self, n_features=28, d_model=32, nhead=2, num_encoder_layers=2,
                 dim_feedforward=64, latent_dim=4, dropout=0.3, max_seq_len=48):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.latent_dim = latent_dim

        # --- Input Projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        # --- Latent Projection ---
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.Tanh()
        )

        # --- Decoder ---
        self.latent_expand = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pos_decoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=max(1, num_encoder_layers - 1),  # Decoder is simpler
            norm=nn.LayerNorm(d_model)
        )

        self.output_proj = nn.Linear(d_model, n_features)

    def encode(self, x):
        """
        x: [batch, seq_len, n_features] -> z: [batch, latent_dim]
        """
        # Project input to model dimension
        h = self.input_proj(x)

        # Add positional encoding
        h = self.pos_encoder(h)

        # Transformer encoding
        h = self.transformer_encoder(h)  # [batch, seq_len, d_model]

        # Temporal aggregation: mean pooling
        h = h.mean(dim=1)  # [batch, d_model]

        # Project to latent space
        z = self.latent_proj(h)  # [batch, latent_dim]

        return z

    def decode(self, z, seq_len):
        """
        z: [batch, latent_dim] -> reconstruction: [batch, seq_len, n_features]
        """
        # Expand latent to model dimension
        h = self.latent_expand(z)  # [batch, d_model]

        # Repeat across time steps
        h = h.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, d_model]

        # Add positional encoding
        h = self.pos_decoder(h)

        # Transformer decoding
        h = self.transformer_decoder(h)  # [batch, seq_len, d_model]

        # Project to output features
        reconstruction = self.output_proj(h)  # [batch, seq_len, n_features]

        return reconstruction

    def forward(self, x):
        """
        x: [batch, seq_len, n_features]
        Returns: reconstruction [batch, seq_len, n_features], z [batch, latent_dim]
        """
        seq_len = x.size(1)
        z = self.encode(x)
        reconstruction = self.decode(z, seq_len)
        return reconstruction, z

    def transform(self, x):
        """Generate latent features for inference (no gradient)."""
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
        return z
```

### Why This Architecture Addresses Previous Failures

1. **Attempt 1 (MLP, overfit ratio 2.69, autocorr >0.99)**:
   - Multi-country data provides ~253 monthly windows vs ~2,463 daily windows
   - But each window contains 28 features from 7 countries, far richer information
   - Monthly resolution naturally reduces autocorrelation (month-to-month changes are larger)
   - Compact model (d_model=32, 2 layers) has ~10-20K params, preventing memorization

2. **Attempt 2 (GRU, all trials pruned)**:
   - Transformer has no recurrence, avoiding gradient issues that caused GRU non-convergence
   - Pre-norm architecture is more stable than post-norm
   - Mean pooling for aggregation is simpler and more stable than using GRU final hidden state
   - Multi-country features provide richer reconstruction target, giving the loss function more to work with

3. **Both attempts (identity mapping)**:
   - 28 input features compressed to 4-6 latent dims is a ~5-7x compression ratio
   - The diversity of features (7 countries x nominal + CPI + changes + cross-country) makes identity mapping impossible through a narrow bottleneck
   - Tanh activation bounds latent output to [-1, 1]

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_features | 28 | 7 countries x 3 features + US TIPS (2) + cross-country aggregates (4) + VIX (1) |
| max_epochs | 300 | Transformers on small data may need more epochs. Early stopping prevents overtraining. |
| early_stop_patience | 25 | Generous patience for small batches on small data. |
| optimizer | AdamW | Better weight decay implementation than Adam for Transformers. |
| scheduler | CosineAnnealingWarmRestarts | T_0=50, T_mult=2. Cyclic learning allows escape from local minima. |
| gradient_clip_norm | 1.0 | Standard for Transformers. |
| norm_first | True | Pre-norm Transformer (more stable training). |
| temporal_aggregation | mean_pooling | Simpler and more robust than CLS token or last-position. |
| postprocessing | first_difference | Break autocorrelation in latent outputs (same strategy as Attempt 2). |

### Optuna Search Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| window_size | {12, 18, 24} | Categorical | 12-24 months of history. Longer windows increase sequence length but reduce sample count. 36 excluded (too few samples). |
| d_model | {24, 32, 48} | Categorical | Small embedding dimensions to prevent overfitting. Must be divisible by nhead. |
| nhead | {2, 4} | Categorical | 2-4 heads. More heads with d_model=24-48 makes head_dim too small. |
| num_encoder_layers | {2, 3} | Categorical | 2-3 encoder layers. 4+ would overfit on ~250 samples. |
| latent_dim | {4, 5, 6} | Categorical | 4-6 output dimensions. Enough for semantic features, narrow enough to prevent identity mapping. |
| dropout | [0.3, 0.5] | Uniform | Aggressive dropout for small sample size. Floor at 0.3 based on Attempt 1 lesson (0.13 was too low). |
| learning_rate | [5e-5, 5e-4] | Log-uniform | Conservative range for Transformer. Too high causes instability; too low prevents convergence. |
| weight_decay | [1e-3, 5e-2] | Log-uniform | Strong regularization. Attempt 1 had 1e-6 which was far too weak. |
| batch_size | {16, 32} | Categorical | Small batches for small dataset. 32 may be too large relative to ~177 training samples. |

**Constraint**: d_model must be divisible by nhead. The search space is designed so all combinations are valid: 24/2=12, 24/4=6, 32/2=16, 32/4=8, 48/2=24, 48/4=12. All head dimensions >= 6.

### Search Settings

- n_trials: 30
- timeout: 3600 seconds (1 hour)
- pruner: MedianPruner(n_startup_trials=7, n_warmup_steps=30)
  - n_warmup_steps=30: Higher than Attempt 2's 15, since Transformers converge slower on small data and we must avoid the Attempt 2 failure of pruning all trials
- direction: minimize (validation reconstruction MSE)
- sampler: TPESampler(seed=42)
- Pruning override: At least 3 trials must complete without pruning (set n_startup_trials=7)

### Search Space Size Analysis

- Categorical combinations: 3 (window) x 3 (d_model) x 2 (nhead) x 2 (layers) x 3 (latent) x 2 (batch) = 216
- With 3 continuous parameters, effective space is large
- 30 trials with TPE is ~14% of categorical space -- acceptable for guided search
- n_startup_trials=7 ensures at least 7 random trials for baseline, preventing premature pruning

---

## 5. Training Configuration

### Loss Function

**Primary: MSE (Mean Squared Error)** on multi-country feature reconstruction.

```python
def compute_loss(reconstruction, target):
    """
    reconstruction: [batch, seq_len, 28]
    target: [batch, seq_len, 28]
    """
    return F.mse_loss(reconstruction, target)
```

Rationale for MSE-only:
- Attempt 1/2 failed on basic model quality (overfitting, non-convergence), not on loss design
- Adding contrastive or supervised components increases complexity without addressing the root cause
- MSE on 28-dimensional multi-country data provides a rich training signal
- If Gate 2 MI is insufficient with MSE alone, contrastive loss can be added in Attempt 4

### Optimizer

- AdamW (decoupled weight decay, better for Transformers than Adam)
- Learning rate and weight_decay searched by Optuna
- Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2)
  - Cycle 1: epochs 0-50
  - Cycle 2: epochs 50-150
  - Cycle 3: epochs 150-450 (beyond max_epochs, so effectively 150-300)
  - Warm restarts help escape local minima on small datasets

### Gradient Clipping

- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Essential for Transformer stability, especially with small batches

### Early Stopping

- Monitor: validation MSE loss
- Patience: 25 epochs
- Restore best weights on stop
- Minimum delta: 1e-6

### Data Loading

- Monthly windows: at each month t (where t >= W), create sample `features[t-W:t]` as shape `[W, 28]`
- Shuffle training windows: YES (windows may overlap but shuffling still helps)
- Drop last incomplete batch: YES
- No data augmentation (financial data should not be augmented)

### Validation Strategy

- Time-series split: train (first 70%) / val (next 15%) / test (last 15%)
- Evaluate reconstruction MSE on validation set at each epoch
- Log overfit ratio (val_loss / train_loss) at each epoch
- Best model checkpoint saved based on val loss

---

## 6. Kaggle Execution Settings

- **enable_gpu**: false
- **Estimated execution time**: 30-60 minutes
- **Estimated memory usage**: <2 GB
- **Required additional pip packages**: [] (PyTorch, pandas, numpy, optuna, fredapi, yfinance all available in Kaggle)
- **enable_internet**: true (needed for FRED API and yfinance)
- **Kaggle Secrets required**: FRED_API_KEY

### Time Budget Breakdown

| Component | Estimated Time |
|-----------|---------------|
| Data fetch (15 FRED series + yfinance) | 3-5 min |
| Feature engineering | <1 min |
| Optuna 30 trials (avg ~100 epochs each, ~0.5s/epoch on CPU) | 20-40 min |
| Final model retraining | 2-3 min |
| Output generation + saving | <1 min |
| US synthetic validation check | <1 min |
| **Total** | **30-50 min** |

### Rationale for CPU

- Data: ~265 months x 28 features = tiny (7,420 values per sample)
- Model: 10-50K parameters (extremely small)
- Batch processing: 16-32 samples per batch
- Transformer self-attention on seq_len=12-24 is trivial (no quadratic blowup)
- GPU overhead (data transfer) would likely *slow down* training at this scale

---

## 7. Implementation Instructions

### For builder_data

#### Data to Fetch

All data is fetched directly in the self-contained train.py. builder_data does NOT need to pre-fetch data for this attempt.

However, builder_data should verify that the gold trading calendar (data/raw/gold.csv) exists for output alignment.

#### Data Validation (in train.py)

Before training, the script must run:

```python
def validate_data(df_monthly):
    """Validate multi-country data quality."""
    # 1. US synthetic vs DFII10 correlation check
    # Compute US synthetic = US_nominal - US_CPI_lagged
    us_synthetic = df_monthly['us_nominal'] - df_monthly['us_cpi_lagged']
    corr = us_synthetic.corr(df_monthly['us_tips_level'])
    print(f"US synthetic vs TIPS correlation: {corr:.3f}")
    # NOTE: Expected ~0.49 (low). This is EXPECTED. We use separate inputs.

    # 2. Check no NaN in training data
    assert df_monthly.isna().sum().sum() == 0, "NaN values in training data"

    # 3. Check feature ranges (after standardization)
    for col in df_monthly.columns:
        assert df_monthly[col].std() > 0.01, f"Near-zero variance: {col}"

    # 4. Check cross-country yield dispersion is non-trivial
    assert df_monthly['yield_dispersion'].std() > 0, "Zero yield dispersion"

    return True
```

### For builder_model

#### Self-Contained train.py Structure

```
1. pip install dependencies (fredapi, optuna)
2. Fetch DFII10 from FRED (daily, for monthly resampling)
3. Fetch 7 country nominal yields from FRED (monthly)
4. Fetch 7 country CPI YoY from FRED (monthly, apply 1-month lag)
5. Fetch VIXCLS from FRED (daily, resample to monthly)
6. Fetch gold GC=F from yfinance (for calendar alignment)
7. Compute 28 input features (see Section 2)
8. Run US synthetic validation check (log correlation, expected ~0.49)
9. Align to common date range, drop NaN, standardize (train-only fit)
10. Create monthly windows [W, 28]
11. Train/val/test split (70/15/15 chronological)
12. Define MultiCountryRateTransformer model class
13. Optuna HPO (30 trials, 3600s timeout)
14. Retrain best model on train set
15. Generate latent features for ALL months using best model
16. Apply first-difference postprocessing to latent outputs
17. Expand monthly outputs to daily using forward-fill (align to gold calendar)
18. Save outputs:
    - submodel_output.csv: date, real_rate_sem_0, ..., real_rate_sem_N
    - model.pt: model state dict
    - training_result.json: metrics, params, output shape, validation check
```

#### PyTorch Class Design

- Class `MultiCountryRateTransformer` as specified in Section 3
- Class `PositionalEncoding` as specified in Section 3
- Both classes are self-contained in train.py (no external imports beyond PyTorch)

#### train.py Specific Notes

1. **FRED API key**: Access via `os.environ['FRED_API_KEY']`. Fail immediately with KeyError if missing.

2. **CPI lag implementation**: `cpi_lagged = cpi_series.shift(1)`. This shifts by 1 month. The first observation becomes NaN and is dropped during common-period alignment.

3. **Standardization**: Compute mean/std from train split ONLY. Save statistics in training_result.json. Apply same transform to val/test/full data.

4. **Window creation**: At each month t >= W, create `features[t-W:t]` as shape `[W, 28]`. Assign the window to date `features.index[t-1]` (last month in window).

5. **Monthly-to-daily expansion**: After generating monthly latent outputs, expand to daily by forward-filling. Each trading day gets the latent value from the most recent completed month. This ensures no look-ahead bias (monthly data for month M is assigned to trading days starting the next month).

6. **Gradient clipping**: `clip_grad_norm_(model.parameters(), max_norm=1.0)` after each backward pass.

7. **Overfit monitoring**: Log `val_loss / train_loss` ratio. Optuna trials with overfit ratio >2.0 at epoch 50+ can be reported for potential pruning.

8. **First-difference postprocessing**:
   ```python
   # After generating raw latent z for all months:
   z_diff = z[1:] - z[:-1]  # Monthly first difference
   # Prepend NaN for first month
   z_output = np.vstack([np.full((1, latent_dim), np.nan), z_diff])
   ```

9. **Output alignment**: The submodel_output.csv must have one row per trading day in the schema date range. Monthly latent values are forward-filled to daily. The first W+1 months (window size + 1 for first-difference) will have NaN.

10. **Reconstruction quality check**: After training, log reconstruction MSE per feature group (US TIPS, country nominals, CPIs, aggregates) to identify which features the model struggles to reconstruct.

11. **Attention weight logging** (optional but useful for interpretability): After training, save average attention weights for a few representative samples. This can reveal which countries/timepoints the model focuses on.

#### Optuna Objective Function Design

```python
def objective(trial):
    # Sample hyperparameters
    window_size = trial.suggest_categorical('window_size', [12, 18, 24])
    d_model = trial.suggest_categorical('d_model', [24, 32, 48])
    nhead = trial.suggest_categorical('nhead', [2, 4])
    num_encoder_layers = trial.suggest_categorical('num_encoder_layers', [2, 3])
    latent_dim = trial.suggest_categorical('latent_dim', [4, 5, 6])
    dropout = trial.suggest_float('dropout', 0.3, 0.5)
    lr = trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 5e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])

    # Recompute windows for this window_size
    train_loader, val_loader = create_dataloaders(
        features_standardized, window_size, batch_size, train_end_idx, val_end_idx
    )

    # Create model
    model = MultiCountryRateTransformer(
        n_features=28,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=2 * d_model,  # Fixed at 2x d_model
        latent_dim=latent_dim,
        dropout=dropout,
        max_seq_len=window_size
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(300):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss = evaluate(model, val_loader)

        # Optuna pruning (after warmup)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                break

    return best_val_loss
```

---

## 8. Risks and Alternatives

### Risk 1: Small Sample Size (253 windows) Causes Overfitting

**Probability**: Medium (35%). This is the primary risk. 253 monthly windows is small even for a compact Transformer.

**Mitigation**:
- Deliberately small model: d_model=24-48, 2-3 layers (~10-50K params)
- Aggressive regularization: dropout 0.3-0.5, weight_decay 1e-3 to 5e-2
- Pre-norm architecture (more stable training)
- Optuna will find the regularization-performance tradeoff

**Fallback (if overfit_ratio >1.5)**:
- Option A: Force d_model=24, num_layers=2 (minimum configuration, ~8K params)
- Option B: Remove half the countries (US + DE + UK + CA only = 4 countries, 16 features)
- Option C: Use hand-crafted cross-country features directly (no Transformer), pass to MLP

### Risk 2: Monthly Resolution Loses Intra-Month Information

**Probability**: Low (15%). The gold prediction target is daily, but cross-country context changes monthly.

**Mitigation**: The monthly latent outputs are forward-filled to daily. This creates step-function features that change once per month. The meta-model can still use these as regime indicators alongside daily base features.

**Fallback**: If Gate 3 shows no improvement from monthly features, switch to hybrid approach (Attempt 4): daily US data processed by GRU + monthly multi-country context processed by Transformer, with cross-attention fusion.

### Risk 3: High Cross-Country Correlation Limits Information Gain

**Probability**: Medium (30%). Nominal yields are 74-88% correlated across countries. The Transformer may struggle to extract signals beyond the dominant global factor.

**Mitigation**:
- Include cross-country aggregate features (dispersion, US-vs-global spread) that explicitly capture divergence
- The Transformer's attention mechanism should learn to focus on periods of divergence
- Reconstruction loss on 28 features forces the model to capture country-specific variation

**Fallback (if Gate 2 MI increase <5%)**:
- Replace country-level features with explicit cross-country statistics only (dispersion, PCA, correlation)
- These can be computed without a Transformer (simpler model, fewer params)

### Risk 4: Transformer Training Instability

**Probability**: Low (15%). Unlike Attempt 2's GRU, Transformers on short sequences are generally stable.

**Mitigation**:
- Pre-norm architecture (more stable than post-norm)
- Gradient clipping at 1.0
- CosineAnnealingWarmRestarts scheduler
- n_warmup_steps=30 for pruner (prevents premature pruning)
- n_startup_trials=7 ensures diversity in initial trials

**Fallback**: If >50% of trials are pruned, increase n_warmup_steps to 50 and reduce max_epochs to 200.

### Risk 5: Monthly-to-Daily Forward-Fill Creates Autocorrelation

**Probability**: Medium-High (40%). Forward-filling monthly values to daily will create within-month autocorrelation of 1.0 (same value repeated).

**Mitigation**:
- First-difference postprocessing at monthly level BEFORE forward-fill removes the slowly-changing level component
- After forward-fill, daily values change only at month boundaries (step function)
- The evaluator should compute autocorrelation on the original monthly values, not the forward-filled daily values
- The meta-model treats these as "regime indicators" that update monthly, not daily signals

**Fallback**: If autocorrelation is still too high after first-differencing:
- Compute rolling z-score instead: (z[t] - mean(z[t-6:t])) / std(z[t-6:t])
- Or output monthly values only, and have the meta-model handle the frequency mismatch

### Risk 6: Synthetic Real Rate Approximation Error

**Probability**: N/A -- already confirmed by fact-check. Handled by design.

**Handling**: This design does NOT use pre-computed synthetic real rates. Nominal yields and CPI are provided as separate inputs. The Transformer can learn whatever relationship exists between them. The low correlation (0.49) between synthetic and true real rates motivated this design decision.

### Alternative Architectures (for future attempts)

| Attempt | Architecture | When to Use |
|---------|-------------|-------------|
| 4 | Hybrid (daily US GRU + monthly multi-country Transformer) | If monthly resolution is insufficient |
| 4 alt | Hand-crafted cross-country features + MLP | If Transformer overfits on 253 samples |
| 5 | HMM / Markov Switching | If all neural approaches fail (classical regime detection) |

---

## Appendix A: Fact-Check Detail -- Synthetic Real Rate Analysis

### Why Synthetic Real Rate (Nominal - CPI_YoY) Fails as TIPS Proxy

**Identity**:
- DFII10 = DGS10 - T10YIE (verified: correlation = 1.0000, RMSE = 0.0000%)
- Where T10YIE = 10-Year Breakeven Inflation Rate (market-implied, forward-looking)

**Synthetic formula**:
- Synthetic_Real = Nominal_10Y - CPI_YoY (backward-looking realized inflation)

**The gap**: T10YIE vs CPI_YoY
- Correlation: 0.60 (only moderate)
- T10YIE reflects *expectations* of future inflation
- CPI_YoY reflects *actual* past 12-month inflation
- During periods of changing inflation expectations (2008, 2021-2022), the gap is very large

**Measured performance (US test case)**:
- Synthetic vs DFII10 level correlation: 0.49
- Synthetic vs DFII10 change correlation: 0.25
- Synthetic change captures 1% of TIPS-gold signal (effectively zero)

**Conclusion**: Pre-computing synthetic real rates destroys information. The Transformer should see nominal and CPI separately.

### What the Researcher Got Wrong

1. **Claimed "correlation >0.90-0.95"** -- Actual: 0.49. Off by a factor of 2.
2. **Claimed "RMSE ~0.5-1.0%"** -- Actual: 1.73%. Nearly 2x the upper bound.
3. **Claimed Japan CPI "ends 2022-04"** -- Actual: ends 2021-06 (6 months earlier).
4. **Claimed 2,192 samples** -- Actual: ~2,120 (265 months x 8 countries), or 1,855 for 7 synthetic + separate US.
5. **Overall feasibility "85%"** -- Downgraded to 65% due to synthetic rate failure.

The researcher's methodology was sound in identifying available data sources, but the quantitative claims about synthetic rate quality were significantly overstated. This is consistent with Haiku-level fact-checking limitations noted in the agent architecture.

## Appendix B: Cross-Country Signal Analysis

### Gold Return Correlations (Monthly)

| Signal | Correlation with Gold Return | Significance |
|--------|------------------------------|-------------|
| TIPS change (DFII10) | -0.37 | Strong -- primary signal |
| Nominal yield change (US) | -0.24 | Moderate |
| Cross-country yield dispersion change | -0.15 | Weak but present |
| Cross-country yield change dispersion | +0.01 | Negligible |
| Synthetic real rate change | -0.002 | Zero (useless) |

### Cross-Country Nominal Yield Correlations

|    | US    | DE    | UK    | CA    |
|----|-------|-------|-------|-------|
| US | 1.000 | 0.792 | 0.783 | 0.882 |
| DE | 0.792 | 1.000 | 0.832 | 0.767 |
| UK | 0.783 | 0.832 | 1.000 | 0.742 |
| CA | 0.882 | 0.767 | 0.742 | 1.000 |

High correlations (0.74-0.88) confirm that cross-country yields are dominated by a global factor. The Transformer must extract the residual country-specific variation to add value.

## Appendix C: Output CSV Format

```csv
date,real_rate_sem_0,real_rate_sem_1,real_rate_sem_2,real_rate_sem_3
2015-01-30,NaN,NaN,NaN,NaN
...
2015-03-02,NaN,NaN,NaN,NaN
2015-03-03,0.012,-0.023,0.005,0.018
2015-03-04,0.012,-0.023,0.005,0.018
...
2015-04-01,0.008,0.015,-0.012,0.003
2015-04-02,0.008,0.015,-0.012,0.003
...
```

Monthly latent values (first-differenced) are forward-filled to daily trading days. Values change only at month boundaries. First ~13-25 months (window size + 1) are NaN.

Column naming: `real_rate_sem_N` (semantic feature N), not `real_rate_latent_N` (to distinguish from Attempt 1/2 outputs).

## Appendix D: Parameter Count Estimates

| Configuration | Total Params | Training Memory |
|---------------|-------------|-----------------|
| d_model=24, layers=2, heads=2 | ~8,300 | ~0.1 MB |
| d_model=32, layers=2, heads=2 | ~14,500 | ~0.2 MB |
| d_model=32, layers=3, heads=4 | ~21,000 | ~0.3 MB |
| d_model=48, layers=3, heads=4 | ~44,500 | ~0.5 MB |

All configurations are extremely small. Overfitting risk comes from sample count (253), not model capacity.
