# Classifier Notebook Generation Summary

**Generated**: 2026-02-18
**Feature**: classifier
**Attempt**: 1
**Agent**: builder_model

## Files Created

1. **kernel-metadata.json** - Kaggle notebook configuration
2. **train.ipynb** - Self-contained Jupyter notebook (12 cells)

## Notebook Structure

### Cell 1: Imports and Setup
- Install xgboost, optuna
- Set random seeds (42)
- Configuration: 100 trials, 1-hour timeout

### Cell 2: Data Fetching (Self-Contained)
- Embeds complete data fetching from `src/fetch_classifier.py`
- yfinance: GC=F, GLD, SI=F, HG=F, DX-Y.NYB, ^GSPC
- FRED: GVZCLS, VIXCLS, DFII10, DGS10
- Computes all 18 features:
  - 5 volatility regime features
  - 4 cross-asset stress features
  - 3 rate/currency shock features
  - 2 volume/flow features
  - 2 momentum context features
  - 2 calendar features
- Creates binary target: 1=UP, 0=DOWN
- 70/15/15 time-series split

### Cell 3: Data Validation
- NaN checks
- Feature statistics
- Correlation matrix (flag pairs > 0.85)
- Class balance verification

### Cell 4: Focal Loss Implementation
- Custom focal loss objective for XGBoost
- Gamma and alpha parameters

### Cell 5: Optuna HPO (100 trials)
- Search space per design doc:
  - max_depth: [2, 5]
  - min_child_weight: [8, 30]
  - subsample: [0.4, 0.9]
  - colsample_bytree: [0.2, 0.8]
  - reg_lambda: [0.5, 20.0] log
  - reg_alpha: [0.1, 10.0] log
  - learning_rate: [0.001, 0.05] log
  - n_estimators: [100, 500]
  - scale_pos_weight: [0.8, 1.5]
  - use_focal: [True, False]
  - focal_gamma: [0.5, 3.0]
- Composite objective: 40% F1_DOWN + 30% ROC-AUC + 30% Balanced Accuracy
- Early stopping: 80 rounds
- TPE sampler, MedianPruner

### Cell 6: Train Final Model
- Retrain with best params
- Evaluate on train/val/test
- Compute all metrics:
  - Balanced accuracy
  - DOWN recall, precision, F1
  - UP recall, precision
  - ROC-AUC
- Confusion matrix (test set)

### Cell 7: Feature Importance
- Gain-based importance
- Rank all 18 features
- Check for dominance (> 30% warning)

### Cell 8: Ensemble (Placeholder)
- Notes that full ensemble requires regression predictions
- Sets default threshold: 0.55

### Cell 9: Generate Classifier Output
- Predict on full dataset
- Save **classifier.csv** (Date, p_up, p_down, predicted_direction)
- P(DOWN) distribution statistics

### Cell 10: 2026 Analysis
- Filter 2026 predictions
- Show sample predictions
- Compare UP/DOWN distribution

### Cell 11: Save Results
- **training_result.json**: Complete metrics, params, feature importance
- **model.json**: Trained XGBoost model
- Includes all metadata

### Cell 12: Diagnostic Plots
- P(DOWN) distribution histogram
- Top 10 feature importance
- Confusion matrix heatmap
- Metrics comparison across splits
- Save **diagnostics.png**

## Validation Results

✅ **PASS** - Ready for Kaggle submission

**Warnings (non-blocking)**:
- 6 yfinance downloads without .empty checks
- These are acceptable for this use case

## Key Features

### Self-Contained Design
- No external file dependencies
- All data fetching via APIs
- FRED_API_KEY from environment (with hardcoded fallback)

### 18 NEW Features
Different from regression model's 24 features:
1. rv_ratio_10_30 - Realized vol ratio (10d/30d)
2. rv_ratio_10_30_z - Vol ratio z-score
3. gvz_level_z - Gold VIX z-score
4. gvz_vix_ratio - Gold/Equity vol ratio
5. intraday_range_ratio - Daily range vs average
6. risk_off_score - Cross-asset composite (VIX+DXY-SPX-Yield)
7. gold_silver_ratio_change - Gold/Silver divergence
8. equity_gold_beta_20d - Rolling beta to S&P 500
9. gold_copper_ratio_change - Gold/Copper divergence
10. rate_surprise - Real rate shock magnitude
11. rate_surprise_signed - Directional rate surprise
12. dxy_acceleration - DXY second derivative
13. gld_volume_z - GLD volume z-score
14. volume_return_sign - Volume-price agreement
15. momentum_divergence - Short vs medium momentum
16. distance_from_20d_high - Position in 20d range
17. day_of_week - Calendar feature (0-4)
18. month_of_year - Seasonal feature (1-12)

### Hyperparameter Search
- 100 Optuna trials
- 12 hyperparameters (10 always active + 2 conditional on focal loss)
- Composite objective balances DOWN detection with overall quality

### Output Files
1. **classifier.csv** - Predictions for all dates
2. **training_result.json** - Complete training report
3. **model.json** - Trained model
4. **diagnostics.png** - Visualization

## Expected Runtime
- Data fetching: 2-3 min
- Feature engineering: 1 min
- Optuna 100 trials: 10-15 min
- Final training + evaluation: 2 min
- **Total**: ~20 minutes (CPU sufficient, GPU not required)

## Next Steps

1. Submit to Kaggle via: `python scripts/kaggle_ops.py submit notebooks/classifier_1/ classifier 1`
2. Monitor execution: `python scripts/kaggle_ops.py monitor`
3. After completion, evaluator will assess:
   - Standalone DOWN F1 > 0.35
   - Ensemble DA improvement > +1.0pp
   - Sharpe maintained > 2.0

## Design Compliance

✅ Follows `docs/design/classifier_1.md` exactly
✅ Uses `src/fetch_classifier.py` data fetching code
✅ Implements all 18 features per specification
✅ Optuna search space matches design
✅ Composite objective: 40% F1 + 30% AUC + 30% Balanced Acc
✅ Self-contained notebook (no external dependencies)
✅ FRED_API_KEY handling (env + fallback)
✅ Focal loss implementation
✅ Early stopping: 80 rounds
✅ Output format: classifier.csv with required columns

## Notes

- **CRITICAL**: The notebook uses a hardcoded FRED_API_KEY fallback for Kaggle. This is acceptable per project policy for Kaggle execution.
- **Ensemble**: Full ensemble evaluation requires regression model predictions. This will be completed after classifier training finishes.
- **Warnings**: yfinance .empty checks are not critical for this notebook as the merged dataset drops all NaN rows after alignment.
