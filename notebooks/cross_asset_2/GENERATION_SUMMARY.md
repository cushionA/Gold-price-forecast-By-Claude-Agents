# Cross-Asset Submodel Training Script Generation Summary

**Generated:** 2026-02-15
**Feature:** cross_asset
**Attempt:** 2
**Agent:** builder_model

## Overview

Generated Kaggle training notebook for cross_asset submodel (Attempt 2) that uses **pre-computed deterministic features** from builder_data instead of running HP optimization.

## Key Design Differences from Typical Submodels

### 1. Pre-Computed Features (No HP Optimization)

Unlike VIX/Technical which generate features during training with Optuna:
- Features are **deterministic** (fixed HMM with n_components=3, n_restarts=10)
- No hyperparameter search needed
- Script focuses on: Load → Compute MI → Save

### 2. Why No HP Optimization?

From design doc:
- HMM regime detection uses fixed parameters (3 components, 10 restarts)
- Z-score windows are fixed (90d for ratio, 20d for divergence)
- Features are deterministic transformations (first difference, z-score)

### 3. MI as Quality Metric

Since there's no loss function to optimize:
- Compute MI between each feature and gold_return_next
- Report MI sum for comparison with design targets
- Compute autocorrelation for Gate 1 validation

## Output Files

```
notebooks/cross_asset_2/
├── kernel-metadata.json   ✅ Kaggle API config
├── train.py               ✅ Self-contained training script
└── GENERATION_SUMMARY.md  ✅ This file
```

## train.py Structure

```python
# 1. IMPORTS
# - pandas, numpy, yfinance, sklearn, hmmlearn

# 2. PRE-COMPUTED FEATURES LOADING
# - Fetch GC=F, SI=F, HG=F from Yahoo Finance
# - Compute daily returns
# - Fit 3D HMM (multi-restart, best log-likelihood)
# - Generate xasset_regime_prob (crisis state probability)
# - Compute gold/copper ratio z-score first difference
# - Compute gold-silver return divergence z-score

# 3. GOLD TARGET FETCHING
# - Download GLD close prices
# - Compute next-day return (%)

# 4. MUTUAL INFORMATION COMPUTATION
# - Split train/val/test (70/15/15)
# - Discretize features into 20 quantile bins
# - Compute MI for each feature vs target on validation set
# - Compute autocorrelation for Gate 1 check

# 5. MAIN EXECUTION
# - Install hmmlearn (not pre-installed on Kaggle)
# - Load features
# - Fetch target
# - Compute metrics
# - Save outputs:
#   - submodel_output.csv (3 columns: regime, recession, divergence)
#   - model_metadata.json (fixed parameters)
#   - training_result.json (MI metrics, autocorr, data info)
```

## Expected Outputs (Kaggle)

After `kaggle kernels push`:

```
submodel_output.csv:
  - Date index (2015-01-30 to 2025-02-12)
  - xasset_regime_prob: [0, 1]
  - xasset_recession_signal: [-4, +4]
  - xasset_divergence: [-4, +4]

training_result.json:
  {
    "feature": "cross_asset",
    "attempt": 2,
    "metrics": {
      "mi_individual": {
        "xasset_regime_prob": <float>,
        "xasset_recession_signal": <float>,
        "xasset_divergence": <float>
      },
      "mi_sum": <float>,
      "autocorr": {
        "xasset_regime_prob": <float>,
        "xasset_recession_signal": <float>,
        "xasset_divergence": <float>
      }
    },
    "optuna_trials_completed": 0,
    "optuna_best_value": <mi_sum>,
    "output_shape": [~2523, 3],
    "output_columns": ["xasset_regime_prob", "xasset_recession_signal", "xasset_divergence"]
  }

model_metadata.json:
  {
    "method": "deterministic_hmm",
    "hmm_n_components": 3,
    "hmm_n_restarts": 10,
    "zscore_window": 90,
    "div_window": 20
  }
```

## Gate Expectations

### Gate 1: Standalone Quality
- **Overfit ratio**: N/A (no neural network)
- **Autocorrelation**: All features should be < 0.99
  - xasset_regime_prob: ~0.83 (design expectation)
  - xasset_recession_signal: ~-0.04 (first diff of z-score)
  - xasset_divergence: ~0.03 (daily return diff z-score)
- **No NaN**: Forward-fill after warmup
- **Expected**: PASS

### Gate 2: Information Gain
- **MI increase > 5%**: High probability
  - Design MI = 0.14 for HMM (higher than VIX 0.079)
- **VIF < 10**: High probability (all measured corr < 0.23)
- **Expected**: PASS

### Gate 3: Ablation Test
- **DA +0.5% OR Sharpe +0.05 OR MAE -0.01%**
- **Confidence**: 7/10 (same HMM pattern as VIX/Technical)

## Kaggle Execution

```bash
# Submit to Kaggle
kaggle kernels push -p notebooks/cross_asset_2/

# Check status
kaggle kernels status tatukado/gold-cross-asset-2

# Fetch results (after completion)
kaggle kernels output tatukado/gold-cross-asset-2 \
  -p data/submodel_outputs/cross_asset/
```

## Design Principles Followed

1. ✅ **Self-contained**: No external file dependencies
2. ✅ **No HP optimization**: Deterministic features only
3. ✅ **Reproducibility**: Fixed random seeds for HMM
4. ✅ **No lookahead**: HMM fit on training data, rolling windows
5. ✅ **API key safety**: hmmlearn installed via pip, no secrets needed
6. ✅ **Returns-based**: Handles futures roll artifacts automatically
7. ✅ **Python syntax**: Verified with ast.parse()

## Implementation Notes

### Why Regenerate Features Inside train.py?

The user asked to "load pre-computed features" but Kaggle notebooks can't easily access local CSV files without uploading them as datasets. Instead:

1. **Replicate builder_data logic** inside train.py
2. Use exact same computation (HMM with fixed params, z-scores)
3. This ensures reproducibility on Kaggle cloud
4. Output matches builder_data's local CSV

### HMM Multi-Restart

hmmlearn doesn't have `n_init` parameter. Must loop manually:

```python
for seed in range(n_restarts):
    model = GaussianHMM(..., random_state=seed)
    model.fit(X_train)
    if model.score(X_train) > best_score:
        best_model = model
```

### First Difference (Critical)

Design correction from Attempt 1:
- Raw gold/copper ratio z-score has autocorr 0.9587 (Gate 1 risk)
- First difference has autocorr -0.039 (safe)

```python
gc_z = (gc_ratio - rolling_mean) / rolling_std
gc_z_diff = gc_z.diff()  # KEY: First difference
```

### Daily Divergence (Not Multi-Day)

Design correction:
- Researcher proposed pct_change(20) → autocorr 0.91 (overlapping windows)
- Daily return diff z-scored → autocorr ~0.03 (safe)

```python
gs_diff = gold_ret - silver_ret  # DAILY returns
gs_z = (gs_diff - rolling_mean) / rolling_std
```

## Estimated Execution Time

- Data download: ~30s
- HMM fitting (3D, 10 restarts): ~2-3 min
- Feature generation: ~10s
- MI computation: ~5s
- **Total**: ~3-4 minutes

## Resource Usage

- **CPU**: Yes (HMM EM algorithm)
- **GPU**: No (enable_gpu: false)
- **Memory**: < 1 GB (~2,500 rows × 6 columns)
- **Internet**: Yes (yfinance downloads)

## Next Steps (After Kaggle Execution)

1. Orchestrator submits via `kaggle kernels push`
2. Wait for "complete" status
3. Fetch results via `kaggle kernels output`
4. Pass to evaluator for Gate 1/2/3
5. If Gate 3 fails: Improvement plan for Attempt 3

---

**Status**: Ready for Kaggle submission
**Syntax Check**: ✅ PASSED
