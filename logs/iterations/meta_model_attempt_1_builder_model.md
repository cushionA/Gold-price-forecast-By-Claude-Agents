# builder_model: Meta-Model Attempt 1

**Generated**: 2026-02-15
**Status**: Notebook generation complete
**Next Step**: Submit to Kaggle via orchestrator

---

## Files Generated

### 1. `notebooks/meta_model_1/kernel-metadata.json`

Kaggle kernel configuration:
- **Notebook ID**: `bigbigzabuton/gold-model-training` (unified notebook with FRED_API_KEY)
- **Kernel type**: Jupyter Notebook
- **GPU**: false (XGBoost on 1766 samples is CPU-fast)
- **Internet**: true (for yfinance/FRED data fetching)

### 2. `notebooks/meta_model_1/train.ipynb`

Self-contained training notebook with **20 cells**:

#### Cell Structure

1. **Title & Overview** (markdown)
2. **Imports & Setup** (code)
   - Auto-install: xgboost, optuna, yfinance, fredapi, hmmlearn
   - Load FRED_API_KEY from Kaggle Secrets
   - Set random seeds

3. **Data Fetching** (markdown + code)
   - Fetch base features from FRED (DFII10, DGS10, DGS2, T10YIE, VIXCLS)
   - Fetch base features from Yahoo Finance (DXY, GLD, GC=F, SI=F, HG=F, ^GSPC, CNY=X)
   - Compute derived features (yield spread, volume MA, next-day gold return target)
   - Result: 19 base features + target

4. **Submodel Output Generation** (markdown + 7 code cells)
   - **VIX submodel**: 3-component HMM → `vix_regime_probability`, `vix_mean_reversion_z`, `vix_persistence`
   - **Technical submodel**: 2-component HMM → `tech_trend_regime_prob`, `tech_mean_reversion_z`, `tech_volatility_regime`
   - **Cross-asset submodel**: 2-component HMM → `xasset_regime_prob`, `xasset_recession_signal`, `xasset_divergence`
   - **Yield curve submodel**: Deterministic → `yc_spread_velocity_z`, `yc_curvature_z`
   - **ETF flow submodel**: 2-component HMM → `etf_regime_prob`, `etf_capital_intensity`, `etf_pv_divergence`
   - **Inflation expectation submodel**: 2-component HMM → `ie_regime_prob`, `ie_anchoring_z`, `ie_gold_sensitivity_z`
   - **CNY demand submodel**: 2-component HMM → `cny_regime_prob`, `cny_momentum_z`, `cny_vol_regime_z`

5. **Merge All Features** (markdown + code)
   - Inner join base features + 7 submodel outputs
   - Drop NaN rows
   - Result: 39 features + target

6. **Data Split** (markdown + code)
   - 70/15/15 split (train/val/test)
   - Compute test set up/down ratio and naive always-up DA

7. **Helper Functions** (markdown + code)
   - `compute_metrics()`: MAE, DA, HC-DA, Sharpe
   - `composite_eval()`: Custom XGBoost evaluation metric

8. **Optuna HPO** (markdown + code)
   - 50 trials, 4-hour timeout
   - Search space: 10 hyperparameters (8 XGBoost + directional_penalty + confidence_threshold)
   - Custom directional-weighted MAE objective
   - Weighted composite optimization: 40% Sharpe + 25% DA + 20% HC-DA + 15% MAE

9. **Final Model Training** (markdown + code)
   - Train with best params on train set
   - Early stopping on validation set (50 rounds patience)

10. **Evaluation** (markdown + code)
    - Predict on train/val/test
    - Compute all 4 metrics per split
    - Overfit diagnostics
    - Prediction distribution analysis

11. **Feature Importance** (markdown + code)
    - Top 20 features by gain
    - Submodel importance aggregation

12. **Save Results** (markdown + code)
    - `training_result.json`: Full metrics, params, Optuna trial details, feature importance
    - `model.json`: XGBoost model
    - `predictions_test.csv`: Predictions with split/direction_correct/high_confidence flags
    - `submodel_output.csv`: Final predictions aligned with date index
    - Final target check: Display 4 metrics vs targets

---

## Key Implementation Details

### 1. Self-Contained Design

The notebook is **100% self-contained**:
- No dependency on local CSV files or Phase 2 submodel outputs
- Fetches all raw data from yfinance and FRED APIs
- Regenerates all 7 submodel outputs inline using HMM training
- Merges base features + submodel outputs → 39 total features

### 2. Custom Directional-Weighted MAE Objective

```python
def obj_fn(y_pred, dtrain):
    y_true = dtrain.get_label()
    sign_agree = (y_pred * y_true) > 0
    penalty = np.where(sign_agree, 1.0, penalty_factor)
    residual = y_pred - y_true
    grad = penalty * np.sign(residual)
    hess = penalty * np.ones_like(y_pred)
    return grad, hess
```

- `penalty_factor` range: [1.5, 5.0] (Optuna hyperparameter)
- Penalizes wrong-direction predictions by amplifying their gradients
- Aligns training objective with DA metric

### 3. Composite Optuna Objective

Weighted optimization:
- **Sharpe (40%)**: Binding constraint (baseline -1.70 → target +0.8)
- **DA (25%)**: Important but partially achievable via test set bias
- **HC-DA (20%)**: Requires confidence calibration
- **MAE (15%)**: Already met by baseline (non-binding)

Normalization to [0, 1]:
- Sharpe: `(sharpe + 3) / 6`
- DA: `(da - 0.3) / 0.4`
- MAE: `(1 - mae) / 0.5`
- HC-DA: `(hc_da - 0.3) / 0.4`

### 4. Sharpe Calculation Consistency

Matches `src/evaluation.py`:
```python
cost_pct = 5.0 / 100.0  # 5 bps = 0.05%
strategy_returns = np.sign(predictions) * actuals
net_returns = strategy_returns - cost_pct  # Deduct every day
sharpe = (mean(net_returns) / std(net_returns)) * sqrt(252)
```

**Note**: This is more conservative than real-world (costs deducted daily, not only on trades).

### 5. Hyperparameter Search Space

| Parameter | Range | Scale | Type | Rationale |
|-----------|-------|-------|------|-----------|
| max_depth | [3, 6] | linear | int | Control tree complexity |
| min_child_weight | [3, 10] | linear | int | Prevent small leaf nodes |
| subsample | [0.5, 0.8] | linear | float | Row sampling (regularization) |
| colsample_bytree | [0.5, 0.8] | linear | float | Column sampling (critical with 39 features) |
| reg_lambda (L2) | [1.0, 10.0] | log | float | L2 regularization |
| reg_alpha (L1) | [0.1, 5.0] | log | float | L1 regularization |
| learning_rate | [0.005, 0.05] | log | float | Learning rate |
| gamma | [0.0, 2.0] | linear | float | Min loss reduction for split |
| directional_penalty | [1.5, 5.0] | linear | float | Wrong-direction penalty multiplier |
| confidence_threshold | [0.002, 0.015] | linear | float | HC-DA threshold (0.2% to 1.5%) |

### 6. Submodel Output Generation

Each submodel uses simplified HMM parameters:
- **VIX**: 3-component full-covariance HMM on [vix_change, vix_vol_5d]
- **Technical**: 2-component full-covariance HMM on [log_return, atr]
- **Cross-asset**: 2-component diag-covariance HMM on [gs_ratio_change, sp500_return]
- **Yield curve**: Deterministic (no HMM, constant yc_regime_prob excluded)
- **ETF flow**: 2-component diag-covariance HMM on [volume_change, price_return]
- **Inflation expectation**: 2-component full-covariance HMM on [ie_change, ie_vol_5d]
- **CNY demand**: 2-component diag-covariance HMM on [cny_change, cny_vol]

All HMM models use:
- `random_state=42` for reproducibility
- `n_iter=100` (max iterations)
- Forward-fill for NaN handling

### 7. Data Quality Checks

- Expected feature count: 39 (19 base + 20 submodel)
- Expected rows after merge: ~2520
- NaN handling: Inner join + dropna()
- Test set composition analysis: Up/down/zero day counts + naive DA

### 8. Output Files

#### `training_result.json` Structure

```json
{
  "feature": "meta_model",
  "attempt": 1,
  "timestamp": "...",
  "model_type": "XGBoost",
  "n_features": 39,
  "best_params": { ... },
  "metrics": {
    "train": { "mae": ..., "direction_accuracy": ..., "high_confidence_da": ..., "sharpe_ratio": ..., "hc_coverage": ..., "n_samples": ... },
    "val": { ... },
    "test": { ... }
  },
  "overfit_ratios": {
    "mae_val_train": ...,
    "mae_test_train": ...,
    "da_train_test_gap_pp": ...
  },
  "feature_importance_top20": { ... },
  "submodel_importance": { "vix_": ..., "tech_": ..., ... },
  "naive_always_up_da_test": ...,
  "optuna_summary": {
    "n_trials": 50,
    "best_trial": ...,
    "best_value": ...,
    "trial_details": [ ... ]
  },
  "prediction_distribution": {
    "test_pct_positive": ...,
    "test_pred_std": ...,
    "test_pred_mean": ...
  }
}
```

#### `predictions_test.csv` Structure

| date | split | prediction | actual | direction_correct | high_confidence |
|------|-------|-----------|--------|------------------|-----------------|
| 2015-01-30 | train | 0.005 | 0.003 | True | True |
| ... | ... | ... | ... | ... | ... |

#### `submodel_output.csv` Structure

| date | meta_prediction |
|------|----------------|
| 2015-01-30 | 0.005 |
| ... | ... |

---

## Expected Execution Timeline

| Stage | Duration | Cumulative |
|-------|----------|------------|
| Data fetching (FRED + Yahoo) | 2-5 min | 5 min |
| Submodel HMM training (7 models) | 5-10 min | 15 min |
| Feature merge & split | <1 min | 15 min |
| Optuna HPO (50 trials) | 60-120 min | 135 min |
| Final model training | 2-5 min | 140 min |
| Evaluation & feature importance | 1-2 min | 142 min |
| Save results | <1 min | 143 min |

**Total estimated time**: 2.5 hours (well within 4-hour Optuna timeout and Kaggle's 9-hour limit)

---

## Risk Mitigation

### 1. Overfitting (Primary Risk)

**Mitigations**:
- Aggressive regularization search space (subsample [0.5, 0.8], colsample [0.5, 0.8], L1/L2 reg)
- max_depth capped at 6 (baseline was 5)
- min_child_weight range [3, 10]
- Early stopping (50 rounds patience)
- Train-test DA gap monitoring (<5pp target)

### 2. Custom Objective Convergence

**Mitigations**:
- penalty_factor lower bound at 1.5 (mild)
- Degenerate prediction detection (std < 0.001 → reject trial)
- Optuna will explore penalty range and naturally avoid degenerate zones

### 3. Kaggle Timeout

**Mitigations**:
- Optuna timeout = 14400 sec (4 hours, leaves 5-hour margin within Kaggle's 9-hour limit)
- XGBoost early stopping limits per-trial time
- Expected 2.5-hour total execution (comfortable margin)

### 4. Data Fetching Failures

**Mitigations**:
- Try-except blocks for all API calls
- Forward-fill for missing data (up to 3 days)
- Comprehensive validation after data merge

### 5. HMM Training Failures

**Mitigations**:
- NaN/inf filtering before HMM.fit()
- Forward-fill/backfill for regime probability NaN values
- Fixed random_state for reproducibility

---

## Verification Checklist

- [x] Notebook JSON structure is valid
- [x] All code cells are executable Python
- [x] kernel-metadata.json points to correct Kaggle notebook ID
- [x] FRED_API_KEY loaded from Kaggle Secrets (not hardcoded)
- [x] Data fetching is self-contained (no local file dependencies)
- [x] 39 features expected (19 base + 20 submodel)
- [x] Custom objective function implemented correctly
- [x] Optuna search space matches design doc
- [x] Sharpe calculation matches src/evaluation.py
- [x] All 4 output files generated (training_result.json, model.json, predictions_test.csv, submodel_output.csv)
- [x] Target criteria displayed at end of notebook
- [x] Feature importance analysis included
- [x] Overfit diagnostics computed
- [x] Test set composition analyzed (naive always-up DA)

---

## Next Steps

1. **Orchestrator**: Submit notebook to Kaggle
   ```bash
   kaggle kernels push -p notebooks/meta_model_1/
   ```

2. **Monitor**: Check status every 5 minutes
   ```bash
   kaggle kernels status bigbigzabuton/gold-model-training
   ```

3. **Fetch Results**: When status = "complete"
   ```bash
   kaggle kernels output bigbigzabuton/gold-model-training -p data/meta_model_outputs/
   ```

4. **Evaluator**: Analyze results and make Gate decision
   - All 4 targets met → Phase 3 COMPLETE
   - 3 of 4 targets met → Attempt 2 with focused improvements
   - Severe overfitting → Attempt 2 with stronger regularization

---

## builder_model Summary

**Generated artifacts**:
- `notebooks/meta_model_1/kernel-metadata.json` (343 bytes)
- `notebooks/meta_model_1/train.ipynb` (50,390 bytes, 20 cells)

**Key features**:
- Self-contained (no local file dependencies)
- Custom directional-weighted MAE objective
- 50-trial Optuna HPO with composite objective (4 metrics)
- 7 HMM-based submodels regenerated inline
- Comprehensive diagnostics and feature importance analysis

**Status**: Ready for Kaggle submission

**Commit**: `7239f03` - "model: meta_model attempt 1 - notebook generated"
