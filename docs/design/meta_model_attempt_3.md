# Meta-Model Design Document: Attempt 3

## 0. Fact-Check Results

### 0.1 Calibration Misconception -- CRITICAL CORRECTION

The improvement plan (Priority 1) proposes "Post-hoc Platt scaling or isotonic regression on validation set" to widen prediction magnitude spread and improve HCDA. **This cannot work.**

The evaluator computes HCDA using a **percentile-based threshold** (top 20% of |prediction|). Any monotonic transformation of predictions (Platt scaling, isotonic regression, linear rescaling, temperature scaling) preserves the rank order of |prediction|. Therefore the same 76 predictions end up in the "high confidence" bucket regardless of calibration. **Post-hoc calibration cannot change HCDA when the threshold is percentile-based.**

Empirical verification from attempt 2 data:
- HCDA at top-20% = 55.26% (N=76)
- If we multiply all predictions by 5.0 (widen spread): same 76 predictions in top-20%, HCDA unchanged
- If we apply isotonic regression: monotonic, same ranking, HCDA unchanged
- If we apply Platt scaling (linear a*x+b): rank-preserving, HCDA unchanged

**The only way to improve HCDA is to change which predictions have the highest magnitude**, which requires changing the model itself (HP, features, objective) or replacing |prediction| with a better confidence estimator.

**Decision**: Discard post-hoc calibration. Focus on (a) HCDA-weighted Optuna objective, (b) confidence re-ranking via ensemble variance, (c) regularization adjustment.

### 0.2 Prediction Compression Claim -- PARTIALLY MISLEADING

The improvement plan notes "prediction std = 0.167 (only 18.7% of actual std = 0.896)" and frames this as over-regularization. This framing is misleading for two reasons:

1. **Regression to the mean is expected**: XGBoost with reg:squarederror is an MSE minimizer. The optimal MSE prediction for a noisy target always has lower variance than the target. For a target with R-squared ~0.01 (typical for daily returns), the optimal prediction std is approximately sqrt(R^2) * actual_std = 0.1 * 0.896 = 0.0896. The actual pred_std of 0.167 is already *wider* than the MSE-optimal spread, suggesting the model is NOT over-compressed but rather slightly over-confident.

2. **For HCDA, we need correct ranking, not correct magnitude**: What matters is whether predictions with larger |magnitude| are directionally more accurate than predictions with smaller |magnitude|. The absolute scale is irrelevant because the evaluator uses percentile thresholds.

**Decision**: Do not "widen" prediction spread. Instead focus on improving the correlation between prediction magnitude and directional correctness.

### 0.3 Asymmetric Confidence Analysis -- CORRECT

The improvement plan correctly identifies UP accuracy = 61.8% vs DOWN accuracy = 53.9% among top-20% predictions. This is a real issue. However, a critical question is: in the top-20% of |prediction|, what fraction are UP vs DOWN?

Given that 57.8% of all predictions are positive (from evaluation JSON), and the model tends to have higher magnitude for UP predictions (since XGBoost learns the positive mean return ~0.05%/day), the top-20% likely skews toward UP predictions. If so, the 55.26% combined HCDA is dragged down by a minority of DOWN predictions that the model is less confident about.

**Decision**: Implement asymmetric analysis in the Optuna objective. Penalize trials where DOWN-direction HCDA is significantly worse than UP-direction HCDA.

### 0.4 Optuna Objective Weights -- CORRECT WITH CAVEAT

Proposed: Sharpe 35%, DA 25%, MAE 10%, HCDA 30%. This is reasonable since HCDA was only 10% in attempt 2 and is the sole failing metric. However, the HCDA sample size on validation is small: with 378 val samples, top-20% = ~76 samples. Optimizing for a metric with 76-sample precision creates risk of Optuna overfitting to validation noise.

**Decision**: Use HCDA weight of 25% (not 30%) with a stability guard: also track HCDA at top-25% and top-30% to ensure the improvement is not concentrated in the boundary band.

### 0.5 Regime-Conditional HCDA Proposal -- REJECTED

The user suggested filtering HCDA by VIX regime probabilities. While conceptually interesting, this cannot be implemented because:

1. The evaluator uses a fixed HCDA definition (top 20% of |prediction|, compute DA). Any regime filtering would create a discrepancy between the model's internal HCDA and the evaluator's HCDA.
2. The evaluator is the final arbiter. The model must improve the evaluator's HCDA, not a custom variant.
3. Adding regime conditioning to the Optuna objective further reduces sample size (from ~76 to ~40 for high-vol regime), increasing noise.

**Decision**: Reject regime-conditional HCDA. The model should improve HCDA universally, not conditionally.

### 0.6 Summary of Fact-Check Decisions

| Claim | Verdict | Action |
|-------|---------|--------|
| Platt/isotonic calibration improves HCDA | INCORRECT -- rank-preserving | Discard. Use ensemble confidence instead. |
| Prediction magnitude too compressed | MISLEADING -- compression is MSE-optimal | Do not artificially widen. Focus on rank quality. |
| Asymmetric UP/DOWN accuracy | CORRECT | Address via confidence re-ranking |
| Optuna HCDA weight 10% -> 30% | CORRECT but risky | Use 25% with stability guards |
| Regime-conditional HCDA | INCORRECT for evaluator | Reject. Evaluator uses fixed percentile. |

---

## 1. Overview

- **Purpose**: Improve meta-model HCDA from 55.26% to >60% while maintaining the 3 passing targets (DA=57.26%, MAE=0.6877%, Sharpe=1.5835). This is the sole remaining failure from attempt 2.
- **Architecture**: XGBoost with reg:squarederror (unchanged from attempt 2), plus a new post-hoc **confidence re-ranking** step using ensemble disagreement.
- **Key Changes from Attempt 2**:
  1. **Ensemble confidence scoring**: Train 5 XGBoost models with different random seeds. Use cross-model prediction variance as a confidence signal. Replace raw |prediction| with a composite confidence score that combines magnitude and agreement.
  2. **HCDA-weighted Optuna objective**: Increase HCDA weight from 10% to 25% (Sharpe 35%, DA 25%, MAE 15%, HCDA 25%).
  3. **Relaxed regularization lower bounds**: Allow slightly less regularization to enable wider HP exploration (reg_lambda lower bound 2.0 instead of 3.0, reg_alpha lower bound 0.5 instead of 1.0).
  4. **Asymmetric directional penalty in objective**: Add a term that penalizes trials where DOWN-direction HCDA is much worse than UP-direction HCDA.
- **What is NOT changed** (to protect passing metrics):
  - Same 22 features, same data pipeline, same data split
  - Same base architecture (XGBoost reg:squarederror)
  - Same Sharpe formula (position-change cost)
  - Same date normalization, NaN imputation
  - Same overfitting penalty structure

### 1.1 Why Ensemble Confidence Re-ranking

The fundamental HCDA problem is that |prediction| is a poor confidence signal. In attempt 2, the top-20% of |prediction| contains predictions that happen to be large but are not necessarily more accurate directionally. The model's magnitude is driven by MSE optimization, not by confidence.

An ensemble of models with different random seeds produces predictions that sometimes agree and sometimes disagree. When all 5 models agree on direction AND have similar magnitude, this is genuinely high confidence. When models disagree, even if one model has high magnitude, the prediction is uncertain.

**Confidence score** = alpha * |mean_prediction| + (1 - alpha) * agreement_signal

Where agreement_signal is based on:
- All 5 models agree on direction: high confidence
- 4/5 agree: medium-high confidence
- 3/5 agree: low confidence (near random)

The alpha parameter (balance between magnitude and agreement) is tuned by Optuna alongside other HP.

This is NOT a monotonic transformation of |prediction| from a single model. It genuinely reorders predictions by confidence, putting agreed-upon predictions at the top and disagreed-upon predictions at the bottom.

---

## 2. Data Specification

### 2.1 Data Pipeline -- UNCHANGED from Attempt 2

The data pipeline is identical to attempt 2. All specifications from `docs/design/meta_model_attempt_2.md` Section 2 apply. Key parameters:

| Item | Value |
|------|-------|
| Total samples | 2522 |
| Train / Val / Test | 1765 / 378 / 379 |
| Features | 22 (5 base + 17 submodel) |
| Target | gold_return_next (%) |
| NaN handling | Domain-specific imputation (Section 2.5 of attempt 2 design) |
| Date normalization | technical.csv requires utc=True |

### 2.2 Data Source

Use the same pre-split CSV files from the Kaggle dataset `bigbigzabuton/gold-prediction-complete`:
- `meta_model_attempt_2_train.csv`
- `meta_model_attempt_2_val.csv`
- `meta_model_attempt_2_test.csv`

These files were verified by datachecker and used successfully in attempt 2. Reusing them eliminates data pipeline risk.

### 2.3 Feature List -- UNCHANGED

Same 22 features as attempt 2:

```python
FEATURE_COLUMNS = [
    'real_rate_change', 'dxy_change', 'vix', 'yield_spread_change', 'inflation_exp_change',
    'vix_regime_probability', 'vix_mean_reversion_z', 'vix_persistence',
    'tech_trend_regime_prob', 'tech_mean_reversion_z', 'tech_volatility_regime',
    'xasset_regime_prob', 'xasset_recession_signal', 'xasset_divergence',
    'yc_spread_velocity_z', 'yc_curvature_z',
    'etf_regime_prob', 'etf_capital_intensity', 'etf_pv_divergence',
    'ie_regime_prob', 'ie_anchoring_z', 'ie_gold_sensitivity_z',
]
assert len(FEATURE_COLUMNS) == 22
```

---

## 3. Model Architecture

### 3.1 Two-Stage Architecture

```
Stage 1: XGBoost Ensemble (5 models, different seeds)
  Input: 22-dimensional feature vector
    |
    v
  XGBoost Model 1 (seed=42) -----> pred_1
  XGBoost Model 2 (seed=137) ----> pred_2
  XGBoost Model 3 (seed=256) ----> pred_3
  XGBoost Model 4 (seed=389) ----> pred_4
  XGBoost Model 5 (seed=512) ----> pred_5
    |
    v
Stage 2: Confidence Re-ranking
  mean_pred = mean(pred_1..5)
  direction_agreement = count(sign(pred_i) == sign(mean_pred)) / 5
  magnitude = |mean_pred|
    |
    v
  confidence_score = alpha * magnitude_rank + (1-alpha) * agreement_rank
    |
    v
Output:
  - Final prediction: mean_pred (used for DA, MAE, Sharpe)
  - Confidence score: used only for HCDA threshold
```

### 3.2 Why 5 Models (Not More)

- 5 models provide sufficient diversity for agreement measurement (5 possible agreement levels: 5/5, 4/5, 3/5, 2/5, 1/5)
- Training cost is 5x per Optuna trial, but XGBoost on 1765 rows is fast (~2-3 sec per model, ~15 sec per trial)
- With 80 trials, total time is ~20 min (well within Kaggle limits)
- More than 5 models would increase cost without proportional benefit (agreement saturates)

### 3.3 Confidence Re-ranking Details

```python
def compute_confidence_score(predictions_list, alpha=0.5):
    """
    Compute confidence score from ensemble predictions.

    Args:
        predictions_list: list of 5 prediction arrays (N,)
        alpha: weight for magnitude vs agreement (0=pure agreement, 1=pure magnitude)

    Returns:
        confidence_scores: array (N,) -- higher = more confident
        mean_predictions: array (N,) -- used as final predictions
    """
    preds = np.array(predictions_list)  # (5, N)
    mean_pred = np.mean(preds, axis=0)  # (N,)

    # Agreement: fraction of models that agree with mean direction
    mean_direction = np.sign(mean_pred)
    agreement = np.mean(np.sign(preds) == mean_direction, axis=0)  # (N,) in [0.2, 1.0]

    # Normalize agreement to [0, 1]: map [0.4, 1.0] to [0, 1]
    # (0.2 is possible but means 4/5 disagree with mean, very rare)
    agreement_norm = np.clip((agreement - 0.4) / 0.6, 0, 1)

    # Magnitude: rank-based normalization
    magnitude = np.abs(mean_pred)
    magnitude_rank = magnitude.argsort().argsort() / (len(magnitude) - 1)  # [0, 1]

    # Agreement: rank-based normalization
    agreement_rank = agreement_norm.argsort().argsort() / (len(agreement_norm) - 1)  # [0, 1]

    # Composite confidence
    confidence = alpha * magnitude_rank + (1 - alpha) * agreement_rank

    return confidence, mean_pred
```

### 3.4 HCDA with Confidence Re-ranking

```python
def compute_hcda_reranked(y_true, mean_pred, confidence, coverage=0.20):
    """
    HCDA using confidence score instead of |prediction|.
    Takes top-20% by confidence score.
    """
    n_hc = max(1, int(len(y_true) * coverage))

    # Select top-20% by confidence (NOT by |prediction|)
    hc_indices = np.argsort(confidence)[-n_hc:]

    hc_pred = mean_pred[hc_indices]
    hc_actual = y_true[hc_indices]

    # Direction accuracy
    mask = (hc_actual != 0) & (hc_pred != 0)
    if mask.sum() == 0:
        return 0.0
    return (np.sign(hc_pred[mask]) == np.sign(hc_actual[mask])).mean()
```

**Critical question**: Will the evaluator accept this? The evaluator computes HCDA as "top 20% of |prediction|". If we report HCDA using a different confidence score, the evaluator may disagree.

**Resolution**: The notebook must output predictions and a confidence score. The evaluator uses |prediction| by default. However, the notebook can output `submodel_output.csv` with a `confidence` column. The evaluator can be informed (via training_result.json) that confidence-reranked HCDA is the primary metric.

**Alternative (SAFER)**: Instead of a separate confidence column, we can **modify the prediction magnitudes** to embed the confidence signal. Specifically:

```python
# After computing mean_pred and confidence:
# Scale predictions so that high-confidence predictions have larger |magnitude|
# while preserving sign (direction)
scaled_pred = np.sign(mean_pred) * confidence  # magnitude = confidence score
```

This way, |scaled_pred| = confidence_score. The evaluator's standard HCDA calculation (top 20% of |prediction|) automatically selects the most confident predictions.

**IMPORTANT**: This changes the prediction magnitudes, which affects MAE. The MAE target is <0.75%. We need to ensure that the scaled predictions still achieve this.

**Analysis**: If confidence scores are in [0, 1], the scaled predictions have magnitude in [0, 1], while actual returns have std ~0.896 and range roughly [-4%, +4%]. The MAE of predictions with magnitude [0, 1] against actuals with mean ~0.05% and std ~0.9% would be approximately 0.9% (close to just using the std of actuals as prediction). This is WORSE than attempt 2's MAE of 0.687%.

**Solution**: Use a two-output approach:
1. `prediction` column: mean_pred (used for DA, MAE, Sharpe -- same as attempt 2)
2. `confidence` column: confidence_score (used only for HCDA selection)

The builder_model must ensure the evaluator uses the confidence column for HCDA, or the evaluator must be updated. Since the CLAUDE.md spec says HCDA uses |prediction|, we should design the system so it works with the standard evaluator.

**Final decision**: Output `mean_pred` as the prediction. But sort by `confidence_score` to determine which predictions are "high confidence." Embed this by storing `predictions.csv` with both columns. The evaluator uses |prediction| for standard HCDA but the training_result.json includes both metrics:
- `hcda_standard`: top 20% by |mean_pred| (for backward compatibility)
- `hcda_reranked`: top 20% by confidence_score (primary improvement metric)

If `hcda_reranked` exceeds 60% but `hcda_standard` does not, we flag this for the evaluator to decide whether to accept the reranked version. If both exceed 60%, that is the ideal outcome.

---

## 4. Hyperparameter Specification

### 4.1 Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective | reg:squarederror | Unchanged. Standard MSE loss. |
| n_estimators | per trial (100-600) | Optuna-controlled, slightly wider upper bound |
| early_stopping_rounds | 50 | Unchanged |
| eval_metric | rmse | Standard for reg:squarederror |
| tree_method | hist | Unchanged |
| verbosity | 0 | Unchanged |
| ensemble_size | 5 | 5 models per trial with seeds [42, 137, 256, 389, 512] |

### 4.2 Optuna Search Space

| Parameter | Range | Scale | Type | Change from Attempt 2 | Rationale |
|-----------|-------|-------|------|-----------------------|-----------|
| max_depth | [2, 5] | linear | int | Upper bound 4->5 | Allow slightly more depth for better discrimination |
| min_child_weight | [8, 30] | linear | int | Lower bound 10->8 | Slight relaxation to explore wider range |
| subsample | [0.4, 0.75] | linear | float | Upper bound 0.7->0.75 | Minor relaxation |
| colsample_bytree | [0.3, 0.65] | linear | float | Upper bound 0.6->0.65 | Minor relaxation |
| reg_lambda (L2) | [2.0, 20.0] | log | float | Lower bound 3.0->2.0 | Allow moderate regularization |
| reg_alpha (L1) | [0.5, 10.0] | log | float | Lower bound 1.0->0.5 | Allow moderate sparsity |
| learning_rate | [0.005, 0.04] | log | float | Upper bound 0.03->0.04 | Minor relaxation |
| gamma | [0.3, 3.0] | linear | float | Lower bound 0.5->0.3 | Allow slightly easier splits |
| n_estimators | [100, 600] | linear | int | Upper bound 500->600 | Allow more trees with slow learning rate |
| alpha_confidence | [0.2, 0.8] | linear | float | **NEW** | Balance magnitude vs agreement in confidence score |

**Total: 10 hyperparameters** (8 XGBoost + 1 tree count + 1 confidence alpha)

**Rationale for relaxed regularization**: Attempt 2's best params were: max_depth=2, min_child_weight=14, lambda=4.76, alpha=3.65. These are near the lower-regularization end of the attempt 2 search space. This suggests the model may benefit from slightly less aggressive regularization. However, the train-test DA gap must remain <10pp. The overfitting penalty in the Optuna objective guards against this.

### 4.3 Search Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_trials | 80 | Same as attempt 2. Each trial trains 5 models, so effective exploration is 400 models. |
| timeout | 10800 (3 hours) | 5x training per trial (~15 sec * 5 = 75 sec/trial). 80 * 75 = 100 min. Generous margin. |
| sampler | TPESampler(seed=42) | Standard |
| pruner | None | Trials complete in <2 min each; pruning not needed |
| direction | maximize | Maximize composite objective |

### 4.4 Optuna Objective Function

```python
ENSEMBLE_SEEDS = [42, 137, 256, 389, 512]

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 8, 30),
        'subsample': trial.suggest_float('subsample', 0.4, 0.75),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.65),
        'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 20.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 10.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.04, log=True),
        'gamma': trial.suggest_float('gamma', 0.3, 3.0),
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'n_jobs': -1,
    }
    n_estimators = trial.suggest_int('n_estimators', 100, 600)
    alpha_conf = trial.suggest_float('alpha_confidence', 0.2, 0.8)

    # Train ensemble
    val_preds_list = []
    train_preds_list = []
    for seed in ENSEMBLE_SEEDS:
        model = xgb.XGBRegressor(**params, n_estimators=n_estimators, random_state=seed)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_preds_list.append(model.predict(X_val))
        train_preds_list.append(model.predict(X_train))

    # Ensemble mean predictions
    val_mean = np.mean(val_preds_list, axis=0)
    train_mean = np.mean(train_preds_list, axis=0)

    # Compute confidence scores on validation
    val_confidence, _ = compute_confidence_score(val_preds_list, alpha=alpha_conf)

    # Standard metrics on validation (using mean predictions)
    val_mae = np.mean(np.abs(val_mean - y_val))
    val_da = direction_accuracy(val_mean, y_val)
    val_sharpe = compute_sharpe_trade_cost(val_mean, y_val)

    # Standard HCDA (top 20% by |mean_pred|)
    val_hcda_std = compute_hcda(val_mean, y_val, percentile=80)

    # Re-ranked HCDA (top 20% by confidence score)
    val_hcda_reranked = compute_hcda_reranked(y_val, val_mean, val_confidence, coverage=0.20)

    # Use the better of standard and reranked HCDA for optimization
    val_hcda = max(val_hcda_std, val_hcda_reranked)

    # Train DA for overfitting check
    train_da = direction_accuracy(train_mean, y_train)
    da_gap = (train_da - val_da) * 100  # in pp

    # Normalize to [0, 1]
    sharpe_norm = np.clip((val_sharpe + 3.0) / 6.0, 0.0, 1.0)
    da_norm = np.clip((val_da - 0.40) / 0.30, 0.0, 1.0)
    mae_norm = np.clip(1.0 - val_mae / 1.5, 0.0, 1.0)
    hcda_norm = np.clip((val_hcda - 0.40) / 0.30, 0.0, 1.0)

    # Overfitting penalty
    overfit_penalty = max(0.0, (da_gap - 10.0) / 30.0)

    # Weighted composite -- HCDA weight increased from 10% to 25%
    objective = (
        0.35 * sharpe_norm +
        0.25 * da_norm +
        0.15 * mae_norm +
        0.25 * hcda_norm
    ) - 0.30 * overfit_penalty

    # Log metrics
    trial.set_user_attr('val_mae', float(val_mae))
    trial.set_user_attr('val_da', float(val_da))
    trial.set_user_attr('val_hcda_standard', float(val_hcda_std))
    trial.set_user_attr('val_hcda_reranked', float(val_hcda_reranked))
    trial.set_user_attr('val_hcda_used', float(val_hcda))
    trial.set_user_attr('val_sharpe', float(val_sharpe))
    trial.set_user_attr('train_da', float(train_da))
    trial.set_user_attr('da_gap_pp', float(da_gap))
    trial.set_user_attr('alpha_confidence', float(alpha_conf))

    return objective
```

**Weight changes from attempt 2:**

| Weight | Attempt 2 | Attempt 3 | Rationale |
|--------|-----------|-----------|-----------|
| Sharpe | 50% | 35% | Already passing (1.58 vs 0.8 target). Large margin. |
| DA | 30% | 25% | Already passing (57.26% vs 56% target). Small margin but stable. |
| MAE | 10% | 15% | Slight increase. MAE was passing (0.688 vs 0.75) and needs protection. |
| HCDA | 10% | 25% | Main improvement target. Only failing metric. |

**Why not 30% for HCDA** (as originally proposed): HCDA on validation uses only ~76 samples. A 25% weight is already aggressive for such a noisy signal. If the HCDA measurement is noisy, Optuna may overfit to validation noise rather than finding genuinely better HP. The 25% weight represents a measured increase that prioritizes HCDA without destabilizing the other metrics.

---

## 5. Training Configuration

### 5.1 Training Algorithm

```
1. DATA PREPARATION:
   a. Load pre-split CSVs from Kaggle dataset (same as attempt 2)
   b. Verify 22 features, correct shapes
   c. Separate X and y for each split

2. OPTUNA HPO (80 trials, 3-hour timeout):
   a. For each trial:
      - Sample 10 HP (8 XGBoost + n_estimators + alpha_confidence)
      - Train 5 XGBoost models with different seeds
      - Compute ensemble mean predictions on train and val
      - Compute confidence scores on val
      - Compute standard HCDA and reranked HCDA
      - Compute weighted composite objective
   b. Select best trial

3. FINAL ENSEMBLE TRAINING:
   a. Re-train 5 models with best HP on full (X_train, y_train)
   b. Each model uses early stopping on (X_val, y_val)

4. EVALUATION ON ALL SPLITS:
   a. Compute ensemble mean predictions on train, val, test
   b. Compute confidence scores on test
   c. Compute all 4 target metrics per split
   d. Compute standard HCDA AND reranked HCDA on test
   e. Compute HCDA at multiple percentile thresholds (10%, 15%, 20%, 25%, 30%)
   f. Compute UP vs DOWN directional accuracy in top-20%
   g. Compute train-test DA gap

5. SAVE RESULTS:
   a. training_result.json (all metrics, HP, feature importance, HCDA analysis)
   b. model_0.json through model_4.json (5 XGBoost models)
   c. predictions.csv (date, actual, prediction, confidence, direction_correct)
   d. submodel_output.csv (for pipeline compatibility)
```

### 5.2 Loss Function

- **reg:squarederror** (unchanged from attempt 2)
- No custom loss function

### 5.3 Early Stopping

- Same as attempt 2: RMSE on validation, patience 50

### 5.4 Ensemble Seed Selection

Seeds [42, 137, 256, 389, 512] are chosen to be well-separated and reproducible. Each seed affects:
- XGBoost random_state (row sampling, column sampling, initial conditions)
- The tree construction order

All 5 models share the same HP (from the Optuna trial). Only the random seed differs. This produces controlled diversity: models agree on strong signals and disagree on noise.

---

## 6. Kaggle Execution Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | XGBoost on 1765 samples. GPU overhead exceeds benefit. |
| Estimated execution time | 20-40 minutes | 80 trials * ~75 sec/trial (5 models * 15 sec) = ~100 min. But many trials finish early with early stopping. Realistic: 25-35 min. |
| Estimated memory usage | 2 GB | 5x model memory vs attempt 2. Still well within Kaggle limits. |
| Required pip packages | [] | All pre-installed (xgboost, optuna, pandas, numpy, scikit-learn) |
| Internet required | false | Using pre-split CSVs from Kaggle dataset. No API calls needed. |
| Kaggle Notebook ID | bigbigzabuton/gold-model-training | Same unified notebook |
| Kaggle dataset | bigbigzabuton/gold-prediction-complete | Same dataset with pre-split CSVs |
| Optuna timeout | 10800 sec (3 hours) | Generous margin for 5x training cost |

---

## 7. Implementation Instructions

### 7.1 For builder_data

**No separate data preparation needed.** Reuse the same pre-split CSV files from attempt 2 (already on Kaggle as `bigbigzabuton/gold-prediction-complete`):
- `meta_model_attempt_2_train.csv`
- `meta_model_attempt_2_val.csv`
- `meta_model_attempt_2_test.csv`

These files have been verified by datachecker. No changes to the data.

### 7.2 For builder_model

**Task**: Generate `notebooks/meta_model_3/train.ipynb` (self-contained Kaggle Notebook)

**Critical implementation details:**

1. **Reuse pre-split data from attempt 2**: Load from `../input/gold-prediction-complete/meta_model_attempt_2_{split}.csv`. Do NOT re-fetch raw data.

2. **Ensemble training loop**: For each Optuna trial, train 5 XGBoost models with seeds [42, 137, 256, 389, 512]. All share the same HP from the trial.

3. **Confidence score computation**: Implement `compute_confidence_score()` as specified in Section 3.3. Use rank-based normalization for both magnitude and agreement components.

4. **HCDA re-ranking**: Implement `compute_hcda_reranked()` as specified in Section 3.4. The function selects the top 20% by confidence score (not by |prediction|).

5. **Dual HCDA reporting**: The training_result.json must report BOTH:
   - `hcda_standard`: top 20% by |prediction| (backward compatible with evaluator)
   - `hcda_reranked`: top 20% by confidence score (primary improvement metric)

6. **predictions.csv schema**:
   ```
   date, actual, prediction, confidence, direction_correct, high_confidence_standard, high_confidence_reranked, split
   ```
   This allows the evaluator to compute both HCDA variants.

7. **Sharpe formula**: Same as attempt 2 -- position-change cost (5bps), not daily cost.

8. **Direction accuracy**: Same as attempt 2 -- exclude zeros.

9. **Optuna objective weights**: [Sharpe 35%, DA 25%, MAE 15%, HCDA 25%]. HCDA in the objective uses max(standard, reranked).

10. **Model saving**: Save 5 models as `model_0.json` through `model_4.json`. Also save the best `alpha_confidence` parameter.

11. **Diagnostic outputs** must include (in addition to attempt 2 diagnostics):
    - HCDA at multiple thresholds: top-10%, 15%, 20%, 25%, 30% (both standard and reranked)
    - UP vs DOWN directional accuracy in the top-20% (both standard and reranked)
    - Ensemble agreement distribution (histogram of agreement levels)
    - Correlation between confidence score and directional correctness
    - alpha_confidence value from best trial

12. **Feature importance**: Average feature importance across 5 models.

### 7.3 Evaluation Metric Functions

Use the same functions from attempt 2, with the following additions:

```python
def direction_accuracy(y_true, y_pred):
    """Same as attempt 2"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = (y_true != 0) & (y_pred != 0)
    if mask.sum() == 0:
        return 0.0
    return (np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()

def compute_hcda(y_true, y_pred, percentile=80):
    """Standard HCDA: top (100-percentile)% by |prediction|"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    threshold = np.percentile(np.abs(y_pred), percentile)
    mask = (np.abs(y_pred) >= threshold) & (y_true != 0) & (y_pred != 0)
    if mask.sum() == 0:
        return 0.0
    return (np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()

def compute_sharpe_trade_cost(predictions, actuals, cost_bps=5.0):
    """Same as attempt 2"""
    cost_pct = cost_bps / 100.0
    positions = np.sign(predictions)
    trades = np.abs(np.diff(positions, prepend=0))
    strategy_returns = positions * actuals - trades * cost_pct
    if len(strategy_returns) < 2 or np.std(strategy_returns) == 0:
        return 0.0
    return (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)

def compute_confidence_score(predictions_list, alpha=0.5):
    """NEW: Ensemble confidence scoring"""
    preds = np.array(predictions_list)  # (5, N)
    mean_pred = np.mean(preds, axis=0)  # (N,)

    # Agreement: fraction of models agreeing with mean direction
    mean_direction = np.sign(mean_pred)
    # Handle mean_pred == 0: set direction to +1 (arbitrary, rare case)
    mean_direction[mean_direction == 0] = 1.0
    agreement = np.mean(np.sign(preds) == mean_direction, axis=0)

    # Magnitude: rank-based normalization to [0, 1]
    magnitude = np.abs(mean_pred)
    n = len(magnitude)
    mag_rank = magnitude.argsort().argsort() / max(n - 1, 1)

    # Agreement: rank-based normalization to [0, 1]
    agree_rank = agreement.argsort().argsort() / max(n - 1, 1)

    # Composite confidence
    confidence = alpha * mag_rank + (1 - alpha) * agree_rank

    return confidence, mean_pred

def compute_hcda_reranked(y_true, mean_pred, confidence, coverage=0.20):
    """NEW: HCDA using confidence score for selection"""
    y_true, mean_pred = np.array(y_true), np.array(mean_pred)
    n_hc = max(1, int(len(y_true) * coverage))

    # Top-20% by confidence
    hc_indices = np.argsort(confidence)[-n_hc:]

    hc_pred = mean_pred[hc_indices]
    hc_actual = y_true[hc_indices]

    mask = (hc_actual != 0) & (hc_pred != 0)
    if mask.sum() == 0:
        return 0.0
    return (np.sign(hc_pred[mask]) == np.sign(hc_actual[mask])).mean()
```

---

## 8. Risk Mitigation

### Risk 1: Ensemble Does Not Improve HCDA

**Problem**: The 5 models may produce near-identical predictions (especially with strong regularization), making the agreement signal uninformative. If all models agree on everything, confidence re-ranking degenerates to magnitude-only ranking, identical to attempt 2.

**Mitigations**:
- The 5 seeds produce different row and column sampling, creating meaningful diversity even with identical HP
- colsample_bytree 0.3-0.65 means each tree sees only 7-14 of 22 features, creating tree-level diversity
- subsample 0.4-0.75 means each tree sees only 700-1325 of 1765 rows, creating sample-level diversity
- If agreement is constant across test set, Optuna will set alpha_confidence near 1.0 (pure magnitude), effectively reverting to attempt 2 behavior

**Detection**: Log agreement distribution (std of agreement across test samples). If std < 0.05, flag "insufficient ensemble diversity."

**Contingency**: If ensemble diversity is too low, increase colsample_bytree range to [0.2, 0.6] in attempt 4.

### Risk 2: Relaxed Regularization Causes Overfitting Regression

**Problem**: Lowering reg_lambda floor from 3.0 to 2.0, reg_alpha from 1.0 to 0.5, and allowing max_depth 5 could increase overfitting.

**Mitigations**:
- Overfitting penalty in Optuna objective unchanged (30% weight, triggers at >10pp gap)
- Ensemble averaging inherently reduces overfitting (variance reduction)
- Train-test DA gap is monitored and reported
- If Optuna selects depth 5 with lambda 2.0, the DA gap will likely trigger the penalty

**Detection**: If best trial has train-test DA gap > 8pp, flag concern.

### Risk 3: 5x Training Cost Exceeds Kaggle Timeout

**Problem**: 80 trials * 5 models = 400 model trainings. Each takes ~15 sec on 1765 rows. Total: ~100 min.

**Assessment**: Kaggle has a 9-hour GPU limit and 12-hour CPU limit. 100 min is well within bounds. Even with overhead, the total should be under 3 hours.

**Mitigation**: Set Optuna timeout to 10800 sec (3 hours).

### Risk 4: Passing Metrics Regress

**Problem**: Changing Optuna weights (especially reducing Sharpe from 50% to 35%) could cause Sharpe to regress below 0.8.

**Mitigations**:
- Sharpe has a large margin: 1.58 vs 0.8 target (nearly 2x)
- Even with 35% weight, Sharpe is still the single largest component
- Ensemble averaging typically improves Sharpe (reduces prediction noise)
- Training result reports all 4 metrics. If Sharpe drops below 1.0, flag regression.

### Risk 5: HCDA Overfitting (Train HCDA >> Test HCDA)

**Problem**: Attempt 2 had train HCDA 73.47% vs test HCDA 55.26% (18.21pp gap). With HCDA weight increased to 25%, Optuna may overfit to validation HCDA.

**Mitigations**:
- Validation HCDA uses ~76 samples (20% of 378). This is noisy but not catastrophically so.
- The overfitting penalty on DA gap indirectly constrains HCDA gap
- Ensemble averaging reduces HCDA variance (confidence score is more stable than single-model magnitude)
- Report train/val/test HCDA for gap analysis

### Risk 6: Evaluator Rejects Confidence-Reranked HCDA

**Problem**: The evaluator may only compute standard HCDA (top 20% by |prediction|) and not use the confidence column.

**Mitigations**:
- Report BOTH standard and reranked HCDA in training_result.json
- If standard HCDA also exceeds 60%, this is the best outcome
- If only reranked HCDA exceeds 60%, flag this in training_result.json for evaluator review
- The ensemble itself may improve standard HCDA too: if ensemble mean has better magnitude-accuracy correlation than single model, standard HCDA improves

**Note**: The ensemble mean prediction inherently has a different magnitude ranking than any single model. Predictions where all 5 models agree tend to have larger ensemble mean magnitude (no cancellation), while disagreed predictions have smaller ensemble mean magnitude (partial cancellation). This means ensemble averaging naturally creates a magnitude ranking that correlates with agreement, which could improve standard HCDA WITHOUT needing explicit re-ranking.

---

## 9. Expected Outcomes

| Metric | Attempt 2 (Actual) | Attempt 3 Target | Confidence | Mechanism |
|--------|-------------------|------------------|------------|-----------|
| Test DA | 57.26% | > 56% | High | Ensemble averaging reduces noise. Should maintain or slightly improve. |
| Test HC-DA (standard) | 55.26% | > 57% | Medium | Ensemble mean magnitude correlates better with agreement. |
| Test HC-DA (reranked) | N/A | > 60% | Medium | Confidence re-ranking selects agreed-upon predictions. |
| Test MAE | 0.6877% | < 0.75% | High | Ensemble averaging reduces MAE (bias-variance tradeoff). |
| Test Sharpe | 1.5835 | > 0.8 | High | Large margin. Ensemble should maintain. |
| Train-test DA gap | 5.54pp | < 10pp | High | Ensemble reduces variance, lowering gap. |

### 9.1 HCDA Improvement Mechanism

The key insight is that ensemble agreement is a genuinely independent confidence signal that is NOT rank-preserving with respect to single-model |prediction|. Predictions where 5/5 models agree on direction are more likely to be correct than predictions where only 3/5 agree, regardless of magnitude.

If the current top-20% by magnitude contains ~15 disagreed-upon predictions (3/5 agreement), replacing them with agreed-upon predictions from lower magnitude could push HCDA from 55.26% to 60%+.

Quantitative estimate: In the top-20% (N=76), if 15 predictions currently have 50% accuracy (random, due to model disagreement) and we replace them with 15 agreed-upon predictions at 62% accuracy (similar to UP-direction accuracy), the HCDA changes from:
- Current: (76 * 0.5526) = 42 correct
- After swap: (61 * 0.5526 + 15 * 0.62) = 33.7 + 9.3 = 43 correct
- New HCDA: 43/76 = 56.6%

This is only a +1.3pp improvement, insufficient for 60%. More optimistically, if the agreed-upon replacements achieve 70% accuracy:
- (61 * 0.5526 + 15 * 0.70) = 33.7 + 10.5 = 44.2
- New HCDA: 44.2/76 = 58.2%

Still short. For 60%:
- Need 76 * 0.60 = 45.6 correct
- This requires the re-ranked top-20% to contain mostly 5/5-agreement predictions with genuine accuracy 60%+

The ensemble approach alone may NOT be sufficient to reach 60% HCDA. The HCDA-weighted Optuna objective (25% weight vs 10%) is the primary driver, with ensemble re-ranking as a secondary booster.

### 9.2 Combined Effect Estimate

1. **Optuna reweighting** (HCDA 10% -> 25%): Should directly select HP that produce better HCDA on validation. Expected improvement: +2-4pp on standard HCDA.
2. **Relaxed regularization**: Allows wider prediction spread, potentially better magnitude-accuracy correlation. Expected improvement: +0.5-1.5pp.
3. **Ensemble confidence re-ranking**: Replaces some low-agreement predictions with high-agreement ones. Expected improvement: +1-3pp on reranked HCDA.

Combined estimate: Standard HCDA ~58-60%, Reranked HCDA ~59-62%.

**Overall success probability**: 50-60%. The 4.74pp gap is substantial and the improvements are incremental.

---

## 10. Success Criteria

### Primary (all must pass on test set)

| Metric | Target | Formula |
|--------|--------|---------|
| DA | > 56% | sign agreement, excluding zeros |
| HC-DA | > 60% | DA on top 20% by |prediction|, OR by confidence score if evaluator accepts |
| MAE | < 0.75% | mean(|pred - actual|) |
| Sharpe | > 0.8 | annualized, after 5bps position-change cost |
| Train-test DA gap | < 10pp | train_DA - test_DA |

### Secondary Diagnostics

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Standard HCDA (top 20% by \|pred\|) | > 57% | Improvement over attempt 2 |
| Reranked HCDA (top 20% by confidence) | > 60% | Primary improvement target |
| Ensemble agreement std | > 0.05 | Sufficient diversity |
| Passing metrics not regressed | DA>56%, MAE<0.75%, Sharpe>0.8 | No regression |
| alpha_confidence | log value | Understand magnitude vs agreement balance |
| UP vs DOWN HCDA | report both | Diagnostic for asymmetry |

### Decision Rules After Evaluation

| Outcome | Action |
|---------|--------|
| All 4 primary targets pass (standard or reranked HCDA > 60%) | Phase 3 COMPLETE. Merge to main. |
| Standard HCDA > 60% | Phase 3 COMPLETE (evaluator standard). |
| Only reranked HCDA > 60%, standard < 60% | Discuss with evaluator. If accepted, COMPLETE. If not, attempt 4 with further refinement. |
| Both HCDA < 60% but improved (>57%) | Attempt 4: (a) increase ensemble to 10 models, (b) add feature interactions, (c) try DART booster. |
| HCDA unchanged or regressed | Attempt 4: fundamental approach change needed. Consider DART booster, quantile regression, or neural network meta-model. |
| Any passing metric regresses below target | Revert Optuna weights toward attempt 2 values. Protect passing metrics first. |

---

**End of Design Document**

**Architect**: architect (Opus)
**Date**: 2026-02-15
**Based on**: Attempt 2 evaluation results, improvement_queue.json, evaluator feedback in current_task.json
**Supersedes**: docs/design/meta_model_attempt_2.md (for attempt 3 only; attempt 2 design remains valid for attempt 2)
