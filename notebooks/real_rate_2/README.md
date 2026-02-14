# Real Rate Attempt 2 - GRU Autoencoder Training Notebook

**Generated**: 2026-02-14
**Agent**: builder_model (Sonnet)
**Feature**: real_rate (DFII10 - 10Y TIPS Yield)
**Attempt**: 2
**Architecture**: GRU Autoencoder (Sequence-to-Sequence)

## Files Generated

1. **kernel-metadata.json** (332 bytes)
   - Kaggle API configuration
   - Notebook type: Jupyter Notebook (train.ipynb)
   - GPU: Disabled (CPU sufficient for ~2,500 samples)
   - Internet: Enabled (for FRED API and yfinance)

2. **train.ipynb** (35 KB)
   - Self-contained Jupyter Notebook
   - 15 cells (8 markdown, 7 code)
   - All data fetching embedded (no external file dependency)
   - Complete training pipeline from data fetch to result export

## Notebook Structure

### Cell 1: Imports and Device Configuration
- PyTorch, pandas, numpy, optuna
- Random seed initialization (42)
- Device setup (CPU)

### Cell 2: Data Fetching (Self-Contained)
- `fetch_real_rate_features()` function embedded from `scripts/fetch_real_rate_features.py`
- FRED API key loaded from Kaggle Secrets
- Auto-installs fredapi and yfinance if needed
- Fetches DFII10, aligns to gold trading days
- Engineers 8 hand-crafted features
- Returns ~2,523 rows (2015-01-30 to 2025-02-12)

### Cell 3: Dataset Class
- `SlidingWindowDataset` for GRU input
- Returns windows with shape [seq_len, n_features] (NOT flattened)
- Different from Attempt 1 MLP which flattened input

### Cell 4: GRU Autoencoder Model
- `RealRateGRUAutoencoder` class
- Encoder: GRU → hidden state → Dropout → FC → Tanh (latent_dim=2)
- Decoder: FC → Dropout → Repeat → GRU → FC (reconstructs sequence)
- Handles bidirectional GRU correctly
- PyTorch gotcha addressed: single-layer GRU dropout = 0

### Cell 5: Training Function
- `train_model()` with early stopping
- Adam optimizer + ReduceLROnPlateau scheduler
- Gradient clipping (max_norm=1.0) for GRU stability
- Tracks train/val loss and overfit ratio
- Restores best checkpoint

### Cell 6: Optuna HPO Function
- `run_hpo()` with 20 trials, 1800s timeout
- Search space:
  - window_size: {40, 60, 80}
  - gru_hidden_dim: {16, 32, 64}
  - gru_num_layers: {1, 2}
  - dropout: [0.3, 0.5]
  - learning_rate: [1e-5, 1e-3] (log-uniform)
  - weight_decay: [1e-4, 1e-2] (log-uniform)
  - bidirectional: {True, False}
  - batch_size: {16, 32}
- TPE sampler with seed=42
- MedianPruner for early trial termination
- Prunes trials with overfit_ratio > 2.0

### Cell 7: Main Execution
1. Split data (70/15/15 chronological)
2. Standardize features (fit on train only)
3. Run Optuna HPO (20 trials)
4. Train final model with best hyperparameters
5. Generate latent features for all dates
6. Apply first-difference postprocessing
7. Save results:
   - `submodel_output.csv`: First-differenced latent features (2 columns)
   - `model.pt`: Model weights + config + standardization stats
   - `training_result.json`: Metrics + hyperparameters + autocorrelation stats

## Key Implementation Details

### First-Difference Postprocessing
```python
# After generating raw latent features z[t]:
output_df_clean = output_df.dropna()  # Remove rows with insufficient lookback
raw_autocorr = [output_df_clean[col].autocorr(lag=1) for col in output_df_clean.columns]

output_diff = output_df_clean.diff().dropna()  # First-difference
diff_autocorr = [output_diff[col].autocorr(lag=1) for col in output_diff.columns]

# Save differenced version (NOT raw latent)
output_diff.to_csv("submodel_output.csv")
```

Rationale: Raw latent had autocorr >0.995 in Attempt 1. First-differencing isolates regime transitions (not levels) and breaks autocorrelation.

### Autocorrelation Logging
Both raw and differenced autocorrelation are logged in `training_result.json` for evaluator analysis:

```json
"autocorrelation": {
  "raw_latent": [0.XXX, 0.XXX],
  "differenced_latent": [0.XXX, 0.XXX]
}
```

### Gradient Clipping
Essential for GRU stability with small batch sizes:

```python
# After loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### GRU Dropout Handling
PyTorch gotcha: GRU's `dropout` parameter only applies between layers. For single-layer GRU, it's ignored.

```python
gru_dropout = dropout if gru_num_layers > 1 else 0.0
self.encoder_gru = nn.GRU(..., dropout=gru_dropout)

# Use separate Dropout layer for output regularization
self.encoder_dropout = nn.Dropout(dropout)
```

## Expected Outputs

### submodel_output.csv
```csv
Date,real_rate_latent_0,real_rate_latent_1
2015-01-30,NaN,NaN
...
2015-05-01,0.023,-0.015
2015-05-02,0.011,0.032
...
2025-02-12,-0.008,0.019
```

- First ~window_size+1 rows are NaN (insufficient lookback + first-difference)
- Values are first-differenced latent dimensions (typical range: [-0.2, 0.2])
- NOT raw Tanh output (which would be bounded to [-1, 1])

### model.pt
```python
{
  'model_state': OrderedDict(...),  # Model weights
  'config': {...},                   # Best hyperparameters
  'standardization': {
    'mean': {...},                   # Train mean (8 features)
    'std': {...}                     # Train std (8 features)
  }
}
```

### training_result.json
```json
{
  "feature": "real_rate",
  "attempt": 2,
  "architecture": "GRU_Autoencoder",
  "timestamp": "2026-02-14T...",
  "best_params": {...},
  "metrics": {
    "train_loss": 0.XXX,
    "val_loss": 0.XXX,
    "overfit_ratio": 1.XXX,
    "epochs_trained": XXX
  },
  "optuna_trials_completed": 20,
  "optuna_best_value": 0.XXX,
  "output_shape": [~2462, 2],
  "output_columns": ["real_rate_latent_0", "real_rate_latent_1"],
  "autocorrelation": {
    "raw_latent": [0.XXX, 0.XXX],
    "differenced_latent": [0.XXX, 0.XXX]
  },
  "data_info": {...},
  "model_params": XXXXX
}
```

## Expected Performance Improvements

| Metric | Attempt 1 | Attempt 2 Target |
|--------|-----------|------------------|
| Overfit ratio | 2.69 | **1.1-1.4** |
| Autocorrelation (lag 1) | >0.995 | **<0.5** (after diff) |
| MI increase (Gate 2) | 18.5% | **15-20%** |
| Gate 3 DA improvement | No improvement | **+0.3-0.5%** |

## Estimated Execution Time

- Data fetch: 1-2 min
- Feature engineering: <1 min
- Optuna HPO (20 trials): 15-20 min
- Final model training: 1-2 min
- Output generation: <1 min
- **Total: 18-25 min**

## Quality Checklist

- ✅ Self-contained (no external file dependencies)
- ✅ GRU architecture matches architect design
- ✅ 20 Optuna trials (not 5 like Attempt 1)
- ✅ Gradient clipping applied (max_norm=1.0)
- ✅ First-difference postprocessing implemented
- ✅ Autocorrelation logged (both raw and differenced)
- ✅ Stronger regularization (dropout 0.3-0.5, weight_decay 1e-4 to 1e-2)
- ✅ Longer window sizes (40-80 vs 20 in Attempt 1)
- ✅ Tighter bottleneck (latent_dim=2 vs 4)
- ✅ Schema validation before saving
- ✅ No `verbose` parameter in ReduceLROnPlateau

## Next Steps for Orchestrator

1. Submit Kaggle Notebook:
   ```bash
   kaggle kernels push -p notebooks/real_rate_2/
   ```

2. Update state.json:
   ```json
   {
     "status": "waiting_training",
     "resume_from": "evaluator",
     "kaggle_kernel": "bigbigzabuton/gold-real-rate-2",
     "submitted_at": "2026-02-14T...",
     "current_feature": "real_rate",
     "current_attempt": 2
   }
   ```

3. Git commit:
   ```bash
   git add notebooks/real_rate_2/
   git commit -m "model: real_rate attempt 2 - GRU notebook generated"
   git push
   ```

4. Monitor execution:
   ```bash
   kaggle kernels status bigbigzabuton/gold-real-rate-2
   ```

5. On completion, fetch results:
   ```bash
   kaggle kernels output bigbigzabuton/gold-real-rate-2 \
     -p data/submodel_outputs/real_rate/
   ```

## Notes

- **No GPU required**: Architect confirmed CPU is sufficient (~5MB memory, 15-25 min runtime)
- **Kaggle Secrets**: User must add `FRED_API_KEY` to Kaggle Secrets before execution
- **Same data as Attempt 1**: The notebook embeds the exact same data fetching logic from `scripts/fetch_real_rate_features.py`
- **Main differences**: GRU architecture, stronger regularization, first-difference postprocessing
