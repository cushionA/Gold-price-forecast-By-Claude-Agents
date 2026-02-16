# Meta-Model Attempt 2 - Kaggle Notebook Summary

**Generated**: 2025-02-15
**Feature**: meta_model
**Attempt**: 2
**Status**: Ready for Kaggle submission

## Files Generated

```
notebooks/meta_model_2/
├── kernel-metadata.json  (293 bytes)
└── train.ipynb           (20 KB, 10 cells)
```

## Notebook Structure

### Cell 1: Introduction
- Markdown cell documenting key improvements from Attempt 1

### Cell 2: Imports
- XGBoost, Optuna, pandas, numpy, sklearn
- Reproducibility: np.random.seed(42)

### Cell 3: Data Loading
- Load 3 pre-split CSVs from Kaggle dataset input
- Verify 22 features exactly
- Train: 1765 samples, Val: 378 samples, Test: 379 samples

### Cell 4: Evaluation Metrics
- `direction_accuracy()` - excludes zeros
- `high_confidence_direction_accuracy()` - top 25% predictions
- `sharpe_ratio()` - position-change cost only (5bps per trade)
- `compute_all_metrics()` - aggregate function

### Cell 5: Optuna Objective
- **Weighted composite**: 0.50*Sharpe + 0.30*DA + 0.10*(1-MAE) + 0.10*HCDA
- **NEW**: Direct overfitting penalty for train-val DA gap > 10pp
- HP search space (aggressive regularization):
  - max_depth: [2, 4]
  - min_child_weight: [10, 30]
  - reg_lambda: [3.0, 20.0]
  - reg_alpha: [1.0, 10.0]
  - subsample: [0.4, 0.7]
  - colsample_bytree: [0.3, 0.6]
  - learning_rate: [0.001, 0.05]
  - n_estimators: [100, 500]

### Cell 6: Hyperparameter Search
- 80 Optuna trials (increased from 50 in Attempt 1)
- TPESampler with seed=42
- MedianPruner with n_warmup_steps=5

### Cell 7: Final Model Training
- Train XGBoost with best hyperparameters
- Objective: reg:squarederror (NO custom directional loss)

### Cell 8: Evaluation on All Splits
- Train, val, test metrics
- Overfitting analysis (train-val gaps)

### Cell 9: Feature Importance
- Top 15 features by XGBoost importance

### Cell 10: Save Results
- model.json (XGBoost model)
- predictions.csv (test set predictions)
- feature_importance.csv
- training_result.json (comprehensive results)

## Key Improvements from Attempt 1

### ✓ Architecture Fixes
1. **NO directional-weighted MAE** - using reg:squarederror
2. **NO price-level features** - excluded by builder_data
3. **NO CNY features** - excluded by builder_data (Gate 2 VIF issues)

### ✓ Regularization Enhancements
1. **Aggressive HP ranges** - max_depth 2-4 (was 3-6), high L1/L2, low subsample
2. **Direct overfitting penalty** - penalize train-val DA gap > 10pp in Optuna objective

### ✓ Sharpe Formula Fix
1. **Position-change cost only** - trades * 0.0005 (was positions * actuals * 0.0005)
2. Matches CLAUDE.md specification exactly

### ✓ Data Quality
1. **1765 train samples** (not 964) - correct data from builder_data
2. **22 features** verified - no leakage or VIF issues

### ✓ HPO Improvements
1. **80 trials** (was 50) - more exploration
2. **Composite weights** - emphasize Sharpe (50%) and DA (30%)

## Expected Kaggle Dataset Path

The notebook expects data files at:
```
../input/gold-model-training/meta_model_attempt_2_train.csv
../input/gold-model-training/meta_model_attempt_2_val.csv
../input/gold-model-training/meta_model_attempt_2_test.csv
```

## Submission Instructions

1. Ensure data files are uploaded to Kaggle dataset "gold-model-training"
2. Run from project root:
   ```bash
   kaggle kernels push -p notebooks/meta_model_2/
   ```
3. Monitor status:
   ```bash
   kaggle kernels status bigbigzabuton/gold-model-training
   ```
4. Fetch results when complete:
   ```bash
   kaggle kernels output bigbigzabuton/gold-model-training -p data/meta_model_outputs/
   ```

## Expected Training Time

- ~15-25 minutes (80 trials, CPU-only, 1765 train samples)
- GPU not required (enable_gpu: false)

## Output Files

After successful training, Kaggle will produce:
- `model.json` - trained XGBoost model
- `predictions.csv` - test set predictions
- `feature_importance.csv` - feature importance scores
- `training_result.json` - comprehensive metrics and metadata
