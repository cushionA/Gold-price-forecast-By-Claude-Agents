# Meta-Model Attempt 3 - Training Notebook Summary

**Generated**: 2025-02-15
**Agent**: builder_model
**Feature**: meta_model
**Attempt**: 3

## Notebook Structure

**Location**: `notebooks/meta_model_3/train.ipynb`
**Type**: Jupyter Notebook (self-contained)
**Total cells**: 11 (1 markdown + 10 code)

## Architecture Implementation

### 5-Model Ensemble
- **Seeds**: [42, 137, 256, 389, 512]
- **Base model**: XGBoost Regressor
- **Prediction aggregation**: Mean of 5 models

### Confidence-Based Re-Ranking for HCDA
```python
confidence_score = alpha * magnitude_rank + (1 - alpha) * agreement_fraction
# Top 20% by confidence_score → HCDA calculation
```

### Optuna Objective (Reweighted)
```
Objective = 0.35*Sharpe + 0.25*DA + 0.15*(1-MAE/2) + 0.25*HCDA_reranked - overfitting_penalty
```

**Overfitting penalties**:
- DA gap >0.10 → -0.1
- Sharpe gap >0.3 → -0.1
- MAE ratio >1.3 → -0.05

## Hyperparameter Search Space

### Relaxed Regularization (vs Attempt 2)
| Parameter | Attempt 2 | Attempt 3 |
|-----------|-----------|-----------|
| max_depth | [2, 4] | [2, 5] |
| reg_lambda | [5.0, 50.0] | [2.0, 20.0] |
| reg_alpha | [1.0, 20.0] | [0.5, 10.0] |

### Full Search Space
- n_estimators: [50, 300] step 50
- max_depth: [2, 5]
- learning_rate: [0.01, 0.15] log
- min_child_weight: [3, 15]
- subsample: [0.6, 0.95]
- colsample_bytree: [0.6, 0.95]
- reg_lambda: [2.0, 20.0] log
- reg_alpha: [0.5, 10.0] log
- gamma: [0.0, 0.5]
- **alpha_confidence**: [0.2, 0.8] (NEW: confidence score weight)

## Data Source

**Kaggle Dataset**: `bigbigzabuton/gold-prediction-complete`

**Files**:
- `meta_model_attempt_2_train.csv` (reused from Attempt 2)
- `meta_model_attempt_2_val.csv`
- `meta_model_attempt_2_test.csv`

**Dimensions**:
- Features: 22
- Target: `gold_return_next`
- Split: 70/15/15 (train/val/test, time-series order)

## Optimization Configuration

- **n_trials**: 80
- **timeout**: 3600 seconds (1 hour)
- **pruner**: MedianPruner with 5 warmup steps

## Evaluation Metrics

The notebook reports both **standard HCDA** (backward compatibility) and **reranked HCDA** (target metric):

### Standard Metrics
- MAE (Mean Absolute Error)
- DA (Direction Accuracy, excluding zeros)
- Sharpe Ratio (with 5bps transaction costs)
- HCDA (standard): Top 20% by magnitude

### New Metric
- **HCDA (reranked)**: Top 20% by confidence score
  - Confidence = alpha * magnitude_rank + (1-alpha) * agreement_fraction
  - alpha optimized by Optuna

## Output Files

The notebook generates:

1. **training_result.json** - Complete metrics and metadata
2. **model_seed_42.json** - Model with seed 42
3. **model_seed_137.json** - Model with seed 137
4. **model_seed_256.json** - Model with seed 256
5. **model_seed_389.json** - Model with seed 389
6. **model_seed_512.json** - Model with seed 512
7. **train_predictions.csv** - Train set predictions
8. **val_predictions.csv** - Validation set predictions
9. **test_predictions.csv** - Test set predictions
10. **feature_importance.csv** - Feature importance from first model

## Target Metrics (Test Set)

| Metric | Target | Comparison |
|--------|--------|------------|
| Direction Accuracy | >56% | ✓/✗ |
| HCDA (reranked) | >60% | ✓/✗ |
| MAE | <0.75% | ✓/✗ |
| Sharpe Ratio | >0.80 | ✓/✗ |

## Key Improvements from Attempt 2

1. **Ensemble diversity**: 5 models with different seeds (vs 1 model)
2. **Smarter high-confidence selection**: Confidence-based re-ranking using both magnitude and model agreement
3. **Better optimization target**: HCDA reranked gets 25% weight (vs 20% for standard HCDA)
4. **Relaxed regularization**: Allow models to capture more signal without excessive constraints
5. **Tunable confidence alpha**: Optuna optimizes the balance between magnitude and agreement

## Execution Flow

```
1. Load data from Kaggle dataset
2. Define evaluation metrics (standard + reranked HCDA)
3. Define Optuna objective (5-model ensemble, new weights)
4. Run HP optimization (80 trials, 1 hour timeout)
5. Train final 5 models with best HP + different seeds
6. Generate ensemble predictions with confidence scores
7. Evaluate on train/val/test (both standard and reranked HCDA)
8. Compute feature importance (from first model)
9. Save all results to Kaggle output directory
```

## Next Steps (Orchestrator)

1. Submit to Kaggle: `kaggle kernels push -p notebooks/meta_model_3/`
2. Monitor status: `kaggle kernels status bigbigzabuton/gold-meta-model-attempt-3`
3. Fetch results when complete: `kaggle kernels output bigbigzabuton/gold-meta-model-attempt-3`
4. Pass to evaluator for Gate evaluation

## Validation Status

✅ Notebook JSON structure validated
✅ 11 cells (1 markdown + 10 code)
✅ kernel-metadata.json validated
✅ Dataset source configured correctly
✅ All metrics implemented (standard + reranked HCDA)
✅ 5-model ensemble logic implemented
✅ Confidence-based re-ranking implemented
✅ Optuna objective reweighted correctly
✅ Relaxed regularization ranges applied

**Ready for Kaggle submission.**
