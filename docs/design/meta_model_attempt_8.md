# Meta-Model Design Document: Attempt 8

## 0. Fact-Check Results

### 0.1 LightGBM / CatBoost on Kaggle

| Check | Result | Detail |
|-------|--------|--------|
| LightGBM pre-installed | PASS | Kaggle notebooks include LightGBM by default. No pip install required. |
| CatBoost pre-installed | PASS | Kaggle notebooks include CatBoost by default. No pip install required. |
| Local installation | FAIL (irrelevant) | Neither is installed locally, but Kaggle execution does not depend on local env. |

### 0.2 Researcher Claim: "12-15% accuracy improvement" from stacking

| Claim | Verdict | Correction |
|-------|---------|------------|
| "12-15% accuracy gains" from stacking | INFLATED | Cited sources (Medium, Johal blog) discuss classification AUC on non-financial datasets. For daily return regression on ~1500 train samples, realistic improvement is 0.5-2.0pp DA, 0.1-0.3 Sharpe. |
| "MLPerf 2024: 12% better AUC-ROC" | MISLEADING | MLPerf benchmarks are classification/ranking tasks with large datasets, not financial regression with ~1500 samples. Not transferable. |
| "MDPI 2025 stock prediction" | PARTIALLY VALID | That paper uses classification (up/down), not regression. Stacking does improve classification ensembles, but the magnitude is overstated for our regression task. |

**Correction applied**: Expected improvement from stacking: DA +0.3-1.0pp, HCDA +0.5-1.5pp, Sharpe +0.1-0.3. These are realistic for a 24-feature GBDT regression task with ~1500 training samples.

### 0.3 Researcher Claim: "15-20% forecast improvement" from regime features

| Claim | Verdict | Correction |
|-------|---------|------------|
| "15-20% forecast accuracy improvement" | INFLATED | RegimeFolio paper (arXiv 2510.14986) measures portfolio optimization metrics (cumulative return, Sharpe ratio), not point forecast accuracy. The improvement refers to strategy-level metrics, not DA. |
| VIX-based regime segmentation | VALID | Regime-conditional features are sound. However, XGBoost already captures nonlinear interactions between VIX features and other variables through tree splits. Explicit regime features provide marginal value. |
| Regime features improve DA | UNCERTAIN | Regime interaction features have diminishing returns when added on top of tree-based models that already handle interactions natively. Expected: DA +0.0-0.5pp. |

**Correction applied**: Expected improvement from regime features alone: DA +0.0-0.5pp, Sharpe +0.0-0.2. Low marginal value because XGBoost natively captures feature interactions via tree structure.

### 0.4 Out-of-Fold Stacking with Time-Series

| Check | Result | Detail |
|-------|--------|--------|
| Researcher: "5-fold CV on train set" | PROBLEMATIC | Standard k-fold CV is invalid for time-series data. It would cause future leakage. |
| Correction needed | YES | Must use **expanding window time-series split** instead of random 5-fold. Each fold uses all data before the fold boundary as training, and the fold itself as OOF prediction target. |
| Impact on stacking | MODERATE | With expanding window, earlier folds have less training data, creating heterogeneous base learner quality. This reduces stacking effectiveness compared to standard CV. |

**Design decision**: Use simple holdout-based stacking instead of OOF stacking. Train base learners on train set, generate stacking features on validation set, and train meta-learner on validation set. This avoids time-series CV complexity while maintaining temporal integrity.

**Alternative considered and rejected**: Time-series-aware OOF with expanding window. Rejected because: (1) train set is only ~1500 samples, splitting further reduces effective training data; (2) the time-series expanding window creates unbalanced folds; (3) the added complexity is not justified for the marginal gain from OOF vs holdout stacking.

### 0.5 Multi-Objective Optuna

| Check | Result | Detail |
|-------|--------|--------|
| `create_study(directions=['maximize','maximize'])` | CONFIRMED | Tested locally with Optuna 4.7.0. NSGA-II sampler works. |
| Pareto front selection | REQUIRES DESIGN | `study.best_trials` returns all Pareto-optimal trials. A selection strategy is needed: e.g., "pick trial with DA > 58% and max Sharpe" or "pick trial closest to DA=0.60, Sharpe=2.5 target". |
| "One-line API change" (researcher) | MISLEADING | The `create_study` call is one line, but the objective function return signature changes (return DA, Sharpe instead of composite), and the post-optimization selection logic is new code. |
| n_trials for multi-objective | NEEDS MORE | Multi-objective with NSGA-II typically needs 1.5-2x the trials of single-objective to adequately populate the Pareto front. |

**Design decision**: Keep single-objective Optuna with an improved composite formula instead of multi-objective. Rationale: (1) Attempt 7 already achieved DA 60.04% and Sharpe 2.46, so the DA/Sharpe tradeoff is not the bottleneck; (2) multi-objective adds selection complexity without clear benefit; (3) the trial budget is better spent on exploring stacking configurations.

### 0.6 Computation Time

| Component | Estimate | Detail |
|-----------|----------|--------|
| XGBoost HPO (100 trials) | ~20-30 min | Same as attempt 7 |
| LightGBM HPO (80 trials) | ~15-25 min | LightGBM is typically 2-3x faster than XGBoost |
| CatBoost HPO (80 trials) | ~20-35 min | CatBoost is slower due to ordered boosting |
| Stacking meta-learner | ~5 min | Simple Lasso/Ridge, minimal computation |
| Bootstrap ensemble (5 models x 3 learners) | ~10 min | Quick prediction generation |
| Regime feature engineering | ~1 min | Simple column operations |
| Total | ~70-110 min | Within Kaggle 12-hour limit. No GPU needed. |

**Researcher claim "2-3 hours"**: Reasonable upper bound. Actual execution is likely 70-110 minutes.

### 0.7 Attempt 7 Data Dimensions -- VERIFIED

| Split | Samples | Proportion |
|-------|---------|------------|
| Train | ~1492 | 70% |
| Val | ~320 | 15% |
| Test | ~458 | 15% |
| Total | ~2270 | 100% |

Features: 24 (attempt 7). Samples-per-feature: 62.2:1 (train).

### 0.8 Regime Feature Impact on Samples-per-Feature

| Scenario | Features | Samples-per-Feature | Assessment |
|----------|----------|-------------------|------------|
| Attempt 7 (current) | 24 | 62.2:1 | Adequate |
| +8 regime features | 32 | 46.6:1 | Marginal but acceptable for GBDT |
| +12 regime features | 36 | 41.4:1 | At the edge. Requires strong regularization |

**Design decision**: Add 6 regime features (conservative), not 8-12. This keeps ratio at ~53:1, a safer margin. XGBoost's colsample_bytree (0.2-0.7) provides additional feature-level regularization.

### 0.9 Summary

| Researcher Claim | Architect Verdict | Action |
|------------------|-------------------|--------|
| Stacking: 12-15% accuracy gain | INFLATED (realistic: 0.5-2pp DA) | Adopt stacking but with realistic expectations |
| Regime features: 15-20% improvement | INFLATED (realistic: 0-0.5pp DA) | Add 6 features (conservative set) |
| Multi-objective Optuna: "one-line change" | MISLEADING | Keep single-objective with improved weights |
| 5-fold CV for OOF predictions | INVALID for time-series | Use holdout stacking instead |
| Total compute: 2-3 hours | REASONABLE upper bound | Actual: 70-110 min |

---

## 1. Overview

- **Purpose**: Improve DA, HCDA, and Sharpe beyond attempt 7 (DA 60.04%, HCDA 64.13%, Sharpe 2.46) through GBDT ensemble stacking and regime-conditional features.
- **Architecture**: 3-model GBDT stacking (XGBoost + LightGBM + CatBoost) with Ridge meta-learner, replacing single XGBoost. Bootstrap confidence and OLS scaling preserved.
- **Key Changes from Attempt 7**:
  1. **Stacking ensemble**: 3 independently tuned GBDT base learners (XGBoost, LightGBM, CatBoost), combined via Ridge regression meta-learner.
  2. **+6 regime-conditional interaction features** (30 total features).
  3. **Improved Optuna objective weights**: Prioritize DA more heavily (currently DA and HCDA are the metrics with most headroom above target).
- **What is NOT changed**: OLS output scaling, bootstrap confidence scoring (5 models per base learner), metric functions, data pipeline, time-series split, fallback mechanism.
- **Expected Effect**: DA +0.3-1.0pp (60.04% -> 60.3-61.0%), HCDA +0.5-1.5pp (64.13% -> 64.6-65.6%), Sharpe +0.0-0.3 (2.46 -> 2.5-2.8). Incremental improvement, not transformative.

**Rationale for stacking over single XGBoost**: Different GBDT algorithms (leaf-wise vs level-wise, ordered boosting) make systematically different errors. When combined, these errors partially cancel, reducing variance and improving generalization. This is particularly valuable in our setting where the XGBoost prediction std (0.023) is extremely small relative to actuals -- even minor improvements in prediction quality translate to meaningful DA gains at the margin.

---

## 2. Data Specification

### 2.1 Input Data

Identical to attempt 7 (24 base + submodel features), plus 6 new regime-conditional features computed during preprocessing.

### 2.2 Original Feature Set (24 features, from attempt 7)

```python
BASE_FEATURE_COLUMNS = [
    # Base features (5)
    'real_rate_change', 'dxy_change', 'vix',
    'yield_spread_change', 'inflation_exp_change',
    # VIX submodel (3)
    'vix_regime_probability', 'vix_mean_reversion_z', 'vix_persistence',
    # Technical submodel (3)
    'tech_trend_regime_prob', 'tech_mean_reversion_z', 'tech_volatility_regime',
    # Cross-asset submodel (3)
    'xasset_regime_prob', 'xasset_recession_signal', 'xasset_divergence',
    # Yield curve submodel (2)
    'yc_spread_velocity_z', 'yc_curvature_z',
    # ETF flow submodel (3)
    'etf_regime_prob', 'etf_capital_intensity', 'etf_pv_divergence',
    # Inflation expectation submodel (3)
    'ie_regime_prob', 'ie_anchoring_z', 'ie_gold_sensitivity_z',
    # Options market submodel (1)
    'options_risk_regime_prob',
    # Temporal context submodel (1)
    'temporal_context_score',
]
assert len(BASE_FEATURE_COLUMNS) == 24
```

### 2.3 Regime-Conditional Features (6 new features)

These features capture nonlinear interactions that are difficult for single trees to learn in one split. Each is a product of a continuous feature and a binary regime indicator derived from existing submodel outputs.

```python
REGIME_FEATURE_COLUMNS = [
    # High-volatility regime interactions (VIX persistence > 0.7)
    'real_rate_x_high_vol',        # real_rate_change * (vix_persistence > 0.7)
    'dxy_x_high_vol',              # dxy_change * (vix_persistence > 0.7)

    # Risk-off regime interactions (xasset_recession_signal > 0.5)
    'etf_flow_x_risk_off',        # etf_capital_intensity * (xasset_recession_signal > 0.5)
    'yc_curvature_x_risk_off',    # yc_curvature_z * (xasset_recession_signal > 0.5)

    # Trend-regime interactions (tech_trend_regime_prob > 0.7)
    'inflation_x_trend',           # inflation_exp_change * (tech_trend_regime_prob > 0.7)
    'temporal_x_trend',            # temporal_context_score * (tech_trend_regime_prob > 0.7)
]
assert len(REGIME_FEATURE_COLUMNS) == 6
```

**Feature generation logic** (pseudo-code):

```python
def generate_regime_features(df):
    # High-vol regime (vix_persistence > 0.7)
    high_vol = (df['vix_persistence'] > 0.7).astype(float)
    df['real_rate_x_high_vol'] = df['real_rate_change'] * high_vol
    df['dxy_x_high_vol'] = df['dxy_change'] * high_vol

    # Risk-off regime (recession signal > 0.5)
    risk_off = (df['xasset_recession_signal'] > 0.5).astype(float)
    df['etf_flow_x_risk_off'] = df['etf_capital_intensity'] * risk_off
    df['yc_curvature_x_risk_off'] = df['yc_curvature_z'] * risk_off

    # Trend regime (trend prob > 0.7)
    trend_on = (df['tech_trend_regime_prob'] > 0.7).astype(float)
    df['inflation_x_trend'] = df['inflation_exp_change'] * trend_on
    df['temporal_x_trend'] = df['temporal_context_score'] * trend_on

    return df
```

**Design rationale for the 6 chosen interactions**:

1. `real_rate_x_high_vol`: Real rate changes matter more when volatility is persistent (flight-to-quality amplification).
2. `dxy_x_high_vol`: Dollar movements have outsized gold impact during high-vol episodes.
3. `etf_flow_x_risk_off`: ETF capital flows are most informative during risk-off regimes (demand-driven).
4. `yc_curvature_x_risk_off`: Yield curve shape signals are amplified during recession fears.
5. `inflation_x_trend`: Inflation expectation changes are more relevant when gold is in a technical uptrend.
6. `temporal_x_trend`: Temporal context score (pattern-based) is most useful during trend regimes.

**Why 6 and not 8-12**: At 30 features with ~1492 train samples, the samples-per-feature ratio is 49.7:1. This is adequate for GBDT but leaves less margin than 24 features. Adding more would risk diluting signal.

### 2.4 Combined Feature Set (30 features)

```python
ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS
assert len(ALL_FEATURE_COLUMNS) == 30
```

### 2.5 NaN Handling for Regime Features

All regime features are products of existing features. If either factor is NaN, the regime feature is NaN. After base imputation (regime_probs -> 0.5, z-scores -> 0.0), all regime features will be computable. No additional NaN handling needed.

### 2.6 Data Split

Unchanged: 70/15/15 time-series split, no shuffle. Same split boundaries as attempt 7.

---

## 3. Model Architecture

### 3.1 Architecture Diagram

```
Input: 30-dimensional feature vector (24 base + 6 regime)
  |
  +--> [XGBoost] --train on train set--> XGB predictions (train, val, test)
  |
  +--> [LightGBM] --train on train set--> LGBM predictions (train, val, test)
  |
  +--> [CatBoost] --train on train set--> CB predictions (train, val, test)
  |
  v
Stacking Input: 3 predictions per sample (XGB_pred, LGBM_pred, CB_pred)
  |
  v
[Ridge Meta-Learner]
  - Trained on VALIDATION SET (val_XGB_pred, val_LGBM_pred, val_CB_pred -> val_y)
  - Regularization prevents over-reliance on any one model
  |
  v
Raw ensemble prediction (single scalar per sample)
  |
  v
POST-TRAINING STEP 1: OLS Output Scaling (from attempt 6)
  - alpha_ols from validation set, capped [0.5, 10.0]
  |
  v
POST-TRAINING STEP 2: Bootstrap Confidence (from attempt 6)
  - 5 XGBoost models with seeds [42,43,44,45,46] (same as attempt 7)
  - Confidence = 1 / (1 + std_across_models)
  - Also compute |prediction| confidence (primary HCDA method in att 7)
  |
  v
Output: prediction, scaled_prediction, bootstrap_std, confidence
  |
  v
Metrics: DA, HCDA (both methods), MAE (raw + scaled), Sharpe
```

### 3.2 Base Learner Details

**XGBoost** (primary learner, proven in attempt 7):
- Objective: reg:squarederror
- HP tuning: Optuna 100 trials
- Early stopping: patience=100 on val RMSE

**LightGBM** (complementary learner):
- Objective: regression (MSE)
- Leaf-wise growth (vs XGBoost's level-wise) captures different patterns
- HP tuning: Optuna 80 trials
- Early stopping: patience=100 on val RMSE

**CatBoost** (complementary learner):
- Objective: RMSE
- Ordered boosting (reduces prediction shift)
- HP tuning: Optuna 80 trials
- Early stopping: patience=100 on val RMSE

### 3.3 Stacking Strategy: Holdout-Based

**Why holdout instead of OOF**:
- Time-series data prohibits random k-fold CV (future leakage)
- Time-series expanding window CV on ~1492 samples creates unbalanced folds with too-small early folds
- Holdout stacking is simpler, leak-free, and sufficient for 3 base learners

**Stacking procedure**:
1. Train each base learner on train set (with early stopping on val set)
2. Generate predictions from each base learner on val set and test set
3. Train Ridge meta-learner on val-set predictions (X = [xgb_val_pred, lgbm_val_pred, cb_val_pred], y = val_actual)
4. Apply Ridge meta-learner to test-set predictions for final ensemble prediction

**Ridge regularization**: Alpha searched via Optuna (log-uniform [0.01, 100.0]) during the meta-learner tuning phase. This prevents the meta-learner from overfitting to the ~320 validation samples.

### 3.4 Fallback Mechanism

If stacking degrades validation metrics vs single XGBoost, revert to single XGBoost (attempt 7 configuration). The fallback is evaluated automatically:

```python
if stacking_val_objective < single_xgb_val_objective:
    use_single_xgb()  # Revert to attempt 7 behavior
else:
    use_stacking()
```

This ensures attempt 8 is never worse than attempt 7 on validation.

### 3.5 All Metric Functions

Identical to attempt 7. No changes to:
- `compute_direction_accuracy()`
- `compute_mae()`
- `compute_sharpe_trade_cost()`
- `compute_hcda()` (|prediction| method)
- `compute_hcda_bootstrap()` (bootstrap variance method)

---

## 4. Hyperparameter Specification

### 4.1 XGBoost HP Search Space (100 trials)

Same as attempt 7 with one change: max_depth range widened to [2, 5] to allow slightly deeper trees when interacting with regime features.

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| max_depth | [2, 5] | int | Wider: regime features need deeper splits |
| n_estimators | [100, 800] | int | Unchanged |
| learning_rate | [0.001, 0.05] | log | Unchanged |
| colsample_bytree | [0.2, 0.7] | linear | Unchanged |
| subsample | [0.4, 0.85] | linear | Unchanged |
| min_child_weight | [12, 25] | int | Unchanged |
| reg_lambda (L2) | [1.0, 15.0] | log | Unchanged |
| reg_alpha (L1) | [0.5, 10.0] | log | Unchanged |

### 4.2 LightGBM HP Search Space (80 trials)

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| num_leaves | [8, 64] | int | Leaf-wise growth: controls complexity |
| max_depth | [-1, 6] | int | -1 = unlimited (controlled by num_leaves) |
| n_estimators | [100, 800] | int | Same as XGBoost |
| learning_rate | [0.001, 0.05] | log | Same as XGBoost |
| feature_fraction | [0.2, 0.7] | linear | Equivalent to colsample_bytree |
| bagging_fraction | [0.4, 0.85] | linear | Equivalent to subsample |
| bagging_freq | [1, 7] | int | Bagging frequency |
| min_child_samples | [15, 30] | int | Similar to min_child_weight |
| reg_lambda (L2) | [1.0, 15.0] | log | Same as XGBoost |
| reg_alpha (L1) | [0.5, 10.0] | log | Same as XGBoost |

### 4.3 CatBoost HP Search Space (80 trials)

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| depth | [2, 6] | int | CatBoost's equivalent of max_depth |
| iterations | [100, 800] | int | Same as n_estimators |
| learning_rate | [0.001, 0.05] | log | Same as XGBoost |
| l2_leaf_reg | [1.0, 15.0] | log | L2 regularization |
| random_strength | [0.5, 5.0] | linear | Randomization for regularization |
| bagging_temperature | [0.0, 2.0] | linear | Bayesian bootstrap temperature |
| rsm | [0.2, 0.7] | linear | Random subspace method (colsample) |
| min_data_in_leaf | [15, 30] | int | Minimum samples per leaf |

### 4.4 Ridge Meta-Learner HP

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| alpha | [0.01, 100.0] | log | L2 penalty for meta-learner |
| fit_intercept | True (fixed) | -- | Allow bias correction |

The Ridge alpha is tuned as part of the stacking objective, evaluated on validation DA/Sharpe.

### 4.5 Search Configuration

| Setting | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| n_trials | 100 | 80 | 80 |
| timeout | 3600 sec | 2400 sec | 3000 sec |
| sampler | TPESampler(seed=42) | TPESampler(seed=43) | TPESampler(seed=44) |
| direction | maximize | maximize | maximize |

**Total trials**: 260 (100 + 80 + 80). Each trial trains one model and evaluates on validation. Total HPO time: ~60-90 minutes.

### 4.6 Optuna Objective Function (All 3 Base Learners)

Same composite formula as attempt 7 with one weight adjustment:

```python
# Attempt 7 weights: 40/30/10/20 (Sharpe/DA/MAE/HCDA)
# Attempt 8 weights: 35/35/10/20 (Sharpe/DA/MAE/HCDA)
objective = (
    0.35 * sharpe_norm +
    0.35 * da_norm +      # INCREASED from 0.30 to 0.35
    0.10 * mae_norm +
    0.20 * hc_da_norm
) - 0.30 * overfit_penalty
```

**Rationale for weight change**: Attempt 7 Sharpe is 3.1x the target (2.46 vs 0.80), so Sharpe is over-optimized relative to DA. Shifting 5% weight from Sharpe to DA encourages the optimizer to explore configurations that prioritize direction accuracy. DA has more practical value for trading (correct direction = profitable trade).

---

## 5. Training Configuration

### 5.1 Training Algorithm (Detailed Pseudocode)

```
1. DATA PREPARATION:
   (identical to attempt 7, PLUS regime feature generation)
   a. Fetch raw data using yfinance and fredapi
   b. Construct base features (5)
   c. Load 8 submodel output CSVs (same as attempt 7)
   d. Merge base + submodel + target on Date
   e. Apply NaN imputation (same as attempt 7)
>> f. Generate 6 regime-conditional features (NEW)
   g. Verify: 30 features, 0 remaining NaN
   h. Split: train (70%), val (15%), test (15%)

2. XGBOOST HPO (100 trials, 1-hour timeout):
   - Same objective function as attempt 7 (with updated weights)
   - 30-feature input
   - Early stopping: 100 rounds
   - Select best trial params

3. LIGHTGBM HPO (80 trials, 40-min timeout):
   - Same objective function (adapted for LightGBM API)
   - 30-feature input
   - Early stopping: 100 rounds
   - Select best trial params

4. CATBOOST HPO (80 trials, 50-min timeout):
   - Same objective function (adapted for CatBoost API)
   - 30-feature input
   - Early stopping: 100 rounds
   - Select best trial params

5. TRAIN FINAL BASE MODELS:
   - XGBoost with best params on train, early stop on val
   - LightGBM with best params on train, early stop on val
   - CatBoost with best params on train, early stop on val

6. GENERATE STACKING FEATURES:
   - For each base model: predict on val set and test set
   - Stacking val features: [xgb_val_pred, lgbm_val_pred, cb_val_pred]
   - Stacking test features: [xgb_test_pred, lgbm_test_pred, cb_test_pred]

7. TRAIN RIDGE META-LEARNER:
   - X = stacking val features (Nx3)
   - y = val actual returns
   - Tune Ridge alpha via Optuna (20 trials, leave-one-out CV on val)
   - Train final Ridge with best alpha

8. GENERATE ENSEMBLE PREDICTIONS:
   - val_ensemble_pred = ridge.predict(stacking_val_features)
   - test_ensemble_pred = ridge.predict(stacking_test_features)

9. FALLBACK COMPARISON:
   - Compare stacking val objective vs single XGBoost val objective
   - If stacking is worse: revert to single XGBoost predictions
   - If stacking is better: use ensemble predictions
   - Also compare with attempt 7 fallback params (attempt 2 best)

10. POST-TRAINING STEP 1: OLS OUTPUT SCALING:
    (identical to attempt 7)

11. POST-TRAINING STEP 2: BOOTSTRAP ENSEMBLE CONFIDENCE:
    (identical to attempt 7 -- 5 XGBoost models, seeds [42-46])
    Note: Bootstrap confidence uses XGBoost only (not the stacking ensemble),
    because bootstrap is for HCDA filtering, not prediction averaging.

12. EVALUATION ON ALL SPLITS:
    (identical to attempt 7 -- DA, HCDA both methods, MAE both, Sharpe,
     feature importance, quarterly breakdown, decile analysis)
>>  a. Report stacking vs single XGBoost comparison
>>  b. Report regime feature importance rankings
>>  c. Report Ridge meta-learner coefficients (base model weights)

13. SAVE RESULTS:
    (same output files as attempt 7)
```

### 5.2 Loss Function

All base learners: MSE / reg:squarederror / RMSE (equivalent objectives).

### 5.3 Early Stopping

- Metric: RMSE on validation set
- Patience: 100 rounds
- Maximum rounds: Optuna-controlled (100-800 for all 3 learners)

### 5.4 Fallback Configuration

Two-level fallback:
1. **Stacking fallback**: If stacking composite < single XGBoost composite, use single XGBoost
2. **Params fallback**: If Optuna XGBoost composite < attempt 2 fallback composite, use attempt 2 params

```python
FALLBACK_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 2,
    'min_child_weight': 14,
    'reg_lambda': 4.76,
    'reg_alpha': 3.65,
    'subsample': 0.478,
    'colsample_bytree': 0.371,
    'learning_rate': 0.025,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'verbosity': 0,
    'seed': 42,
}
FALLBACK_N_ESTIMATORS = 300
```

---

## 6. Kaggle Execution Configuration

| Setting | Value | Change from Att 7 | Rationale |
|---------|-------|-------------------|-----------|
| enable_gpu | false | CHANGED from true | GBDT models are CPU-efficient. GPU overhead for small dataset (~1500 train rows) does not help. CPU gives longer time quota. |
| Estimated execution time | 70-110 minutes | INCREASED from 25-45 | 3 HPO searches instead of 1 |
| Estimated memory usage | 3.0 GB | INCREASED from 1.5 | 3 models + stacking arrays |
| Required pip packages | [] | unchanged | LightGBM and CatBoost are pre-installed on Kaggle |
| Internet required | true | unchanged | For data fetching |
| Kaggle Notebook ID | bigbigzabuton/gold-model-training-meta-model | unchanged | Same notebook, new version |
| dataset_sources | bigbigzabuton/gold-prediction-submodels | unchanged | All submodel CSVs |
| Optuna total timeout | 9000 sec (2.5 hrs) | INCREASED | 3 HPO searches + overhead |

**GPU decision rationale**: The dataset has ~1500 training samples and 30 features. This is too small for GPU acceleration to provide speedup over CPU for tree-based models. XGBoost, LightGBM, and CatBoost all run faster on CPU for this data size. Disabling GPU also gives a longer time quota on Kaggle (12 hours CPU vs 9 hours GPU).

### 6.1 kernel-metadata.json

```json
{
  "id": "bigbigzabuton/gold-model-training-meta-model",
  "title": "Gold Meta-Model Training - Attempt 8",
  "code_file": "train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["bigbigzabuton/gold-prediction-submodels"],
  "competition_sources": [],
  "kernel_sources": []
}
```

---

## 7. Implementation Instructions

### 7.1 For builder_data

No separate data preparation needed. The meta-model notebook is self-contained.

### 7.2 For builder_model

**Task**: Generate `notebooks/meta_model_8/train.ipynb` (self-contained Kaggle Notebook)

**Base**: Attempt 7 notebook with significant modifications.

#### 7.2.1 New Imports (Cell 1)

```python
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import Ridge
import optuna
from optuna.samplers import TPESampler
```

#### 7.2.2 Feature Definitions (Cell 3)

Update FEATURE_COLUMNS to include 24 base + 6 regime features (30 total). Assert 30.

#### 7.2.3 Regime Feature Generation (New Cell After Cell 7)

Add the `generate_regime_features()` function and call it on `final_df` after NaN imputation.

**CRITICAL**: Regime features must be generated AFTER NaN imputation, because they depend on imputed values for regime probabilities.

#### 7.2.4 LightGBM Optuna Objective (New Cell)

```python
def lgbm_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        'max_depth': trial.suggest_int('max_depth', -1, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.7),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.85),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 30),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 10.0, log=True),
        'seed': 43 + trial.number,
    }
    n_estimators = trial.suggest_int('n_estimators', 100, 800)

    model = lgb.LGBMRegressor(**params, n_estimators=n_estimators)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)

    # Same composite objective as XGBoost
    val_da = compute_direction_accuracy(y_val, val_pred)
    val_sharpe = compute_sharpe_trade_cost(y_val, val_pred)
    val_mae = compute_mae(y_val, val_pred)
    val_hc_da, _ = compute_hcda(y_val, val_pred)
    train_da = compute_direction_accuracy(y_train, train_pred)
    da_gap = (train_da - val_da) * 100
    overfit_penalty = max(0.0, (da_gap - 10.0) / 30.0)

    sharpe_norm = np.clip((val_sharpe + 3.0) / 6.0, 0.0, 1.0)
    da_norm = np.clip((val_da * 100 - 40.0) / 30.0, 0.0, 1.0)
    mae_norm = np.clip((1.0 - val_mae) / 0.5, 0.0, 1.0)
    hc_da_norm = np.clip((val_hc_da * 100 - 40.0) / 30.0, 0.0, 1.0)

    objective = (
        0.35 * sharpe_norm + 0.35 * da_norm + 0.10 * mae_norm + 0.20 * hc_da_norm
    ) - 0.30 * overfit_penalty

    trial.set_user_attr('val_da', float(val_da))
    trial.set_user_attr('val_sharpe', float(val_sharpe))
    trial.set_user_attr('val_hc_da', float(val_hc_da))
    trial.set_user_attr('val_mae', float(val_mae))
    trial.set_user_attr('train_da', float(train_da))

    return objective
```

#### 7.2.5 CatBoost Optuna Objective (New Cell)

```python
def catboost_objective(trial):
    params = {
        'loss_function': 'RMSE',
        'depth': trial.suggest_int('depth', 2, 6),
        'iterations': trial.suggest_int('iterations', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.5, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
        'rsm': trial.suggest_float('rsm', 0.2, 0.7),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 15, 30),
        'random_seed': 44 + trial.number,
        'verbose': 0,
    }

    model = cb.CatBoostRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val),
              early_stopping_rounds=100,
              verbose=0)

    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)

    # Same composite objective as XGBoost
    # ... (same computation as lgbm_objective)

    return objective
```

#### 7.2.6 Stacking Meta-Learner (New Cell)

```python
# Generate stacking features
xgb_val_pred = xgb_final.predict(X_val)
lgbm_val_pred = lgbm_final.predict(X_val)
cb_val_pred = cb_final.predict(X_val)

xgb_test_pred = xgb_final.predict(X_test)
lgbm_test_pred = lgbm_final.predict(X_test)
cb_test_pred = cb_final.predict(X_test)

stack_val = np.column_stack([xgb_val_pred, lgbm_val_pred, cb_val_pred])
stack_test = np.column_stack([xgb_test_pred, lgbm_test_pred, cb_test_pred])

# Tune Ridge alpha
def ridge_objective(trial):
    alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(stack_val, y_val)
    val_pred = ridge.predict(stack_val)

    da = compute_direction_accuracy(y_val, val_pred)
    sharpe = compute_sharpe_trade_cost(y_val, val_pred)
    # Use DA + Sharpe composite for meta-learner selection
    return 0.5 * da + 0.5 * np.clip((sharpe + 3.0) / 6.0, 0.0, 1.0)

ridge_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=45))
ridge_study.optimize(ridge_objective, n_trials=20, timeout=120)

# Train final Ridge
best_alpha = ridge_study.best_params['alpha']
ridge_meta = Ridge(alpha=best_alpha, fit_intercept=True)
ridge_meta.fit(stack_val, y_val)

# Ensemble predictions
ensemble_val_pred = ridge_meta.predict(stack_val)
ensemble_test_pred = ridge_meta.predict(stack_test)

# Report Ridge coefficients
print(f"Ridge coefficients: XGB={ridge_meta.coef_[0]:.4f}, "
      f"LGBM={ridge_meta.coef_[1]:.4f}, CB={ridge_meta.coef_[2]:.4f}")
print(f"Ridge intercept: {ridge_meta.intercept_:.4f}")
```

#### 7.2.7 Stacking vs Single XGBoost Comparison (New Cell)

```python
# Evaluate stacking vs single XGBoost on validation
stack_da = compute_direction_accuracy(y_val, ensemble_val_pred)
stack_sharpe = compute_sharpe_trade_cost(y_val, ensemble_val_pred)
stack_hcda, _ = compute_hcda(y_val, ensemble_val_pred)

single_da = compute_direction_accuracy(y_val, xgb_val_pred)
single_sharpe = compute_sharpe_trade_cost(y_val, xgb_val_pred)
single_hcda, _ = compute_hcda(y_val, xgb_val_pred)

# Composite comparison
# ... compute both composites ...

if stacking_composite > single_xgb_composite:
    use_stacking = True
    pred_test = ensemble_test_pred
    pred_val = ensemble_val_pred
else:
    use_stacking = False
    pred_test = xgb_test_pred
    pred_val = xgb_val_pred
```

#### 7.2.8 Bootstrap Confidence (Unchanged)

Use XGBoost-only bootstrap (5 models, seeds [42-46]) exactly as in attempt 7. The bootstrap is for HCDA confidence scoring, independent of the stacking ensemble.

#### 7.2.9 OLS Scaling (Unchanged)

Identical to attempt 7.

#### 7.2.10 Result Saving Updates

```python
training_result['attempt'] = 8
training_result['architecture'] = 'GBDT Stacking (XGB+LGBM+CB) + Ridge meta-learner + Bootstrap confidence + OLS scaling'
training_result['model_config']['n_features'] = 30
training_result['stacking'] = {
    'base_learners': ['XGBoost', 'LightGBM', 'CatBoost'],
    'meta_learner': 'Ridge',
    'ridge_alpha': float(best_alpha),
    'ridge_coefficients': {
        'xgb': float(ridge_meta.coef_[0]),
        'lgbm': float(ridge_meta.coef_[1]),
        'cb': float(ridge_meta.coef_[2]),
    },
    'ridge_intercept': float(ridge_meta.intercept_),
    'stacking_used': use_stacking,
    'stacking_vs_single': {
        'stacking_val_da': float(stack_da),
        'single_xgb_val_da': float(single_da),
        'delta_da_pp': float((stack_da - single_da) * 100),
    },
}
training_result['regime_features'] = {
    'n_regime_features': 6,
    'features': REGIME_FEATURE_COLUMNS,
}
```

### 7.3 Implementation Checklist for builder_model

1. Copy attempt 7 notebook structure (`notebooks/meta_model_7/train.ipynb`)
2. Update markdown header to "Attempt 8" (Cell 0)
3. Add `import lightgbm as lgb, catboost as cb` and `from sklearn.linear_model import Ridge` (Cell 1)
4. Update FEATURE_COLUMNS to 30 features (24 base + 6 regime) (Cell 3)
5. Add `generate_regime_features()` function AFTER NaN imputation (new cell after Cell 7)
6. Change assertion from 24 to 30 features in all relevant places
7. Update Optuna objective weights from 40/30/10/20 to 35/35/10/20 (Cell 13)
8. Widen XGBoost max_depth range to [2, 5] (Cell 13)
9. Add LightGBM Optuna HPO (new cell, 80 trials, 40-min timeout)
10. Add CatBoost Optuna HPO (new cell, 80 trials, 50-min timeout)
11. Add stacking meta-learner code (Ridge + Optuna alpha tuning, new cell)
12. Add stacking vs single XGBoost comparison logic
13. Update bootstrap confidence to use final pred_test (stacking or single XGBoost)
14. Update all "24 features" text to "30 features"
15. Update `training_result['attempt']` to 8
16. Add stacking and regime feature metadata to training_result
17. Update kernel-metadata.json: title to "Attempt 8", enable_gpu to false
18. Run `scripts/validate_notebook.py` to verify notebook

**CRITICAL implementation notes**:
- LightGBM early stopping API: use `callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]` (NOT `early_stopping_rounds` parameter which is deprecated in newer versions)
- CatBoost verbose: set `verbose=0` in both constructor and `fit()` to suppress output
- Ridge meta-learner: train on VALIDATION set, not train set. This is the stacking paradigm.
- Regime features: generate AFTER NaN imputation to avoid NaN propagation
- Feature importance: report XGBoost feature importance (30 features) -- LightGBM and CatBoost importances can be reported separately as diagnostics

---

## 8. Risk Mitigation

### Risk 1: Stacking Overfits on Validation Set (MODERATE)

**Scenario**: Ridge meta-learner trained on ~320 validation samples may memorize val-specific patterns. This leads to optimistic val metrics and poor test performance.

**Probability**: 25-35%.

**Mitigation**:
1. Ridge L2 regularization (alpha tuned via Optuna) prevents overfitting
2. Only 3 input features to Ridge (XGB/LGBM/CB predictions), so degrees of freedom are minimal
3. Automatic fallback to single XGBoost if stacking degrades val metrics
4. Report train/val/test metrics for all configurations (single vs stacking) for comparison

### Risk 2: Regime Features Add Noise (LOW-MODERATE)

**Scenario**: The 6 regime features are sparse (many zeros when regime is off). XGBoost may assign them spurious importance, especially with max_depth=5.

**Probability**: 20-30%.

**Mitigation**:
1. XGBoost L1/L2 regularization (reg_alpha, reg_lambda in [0.5, 15.0]) naturally zeros out noisy features
2. colsample_bytree in [0.2, 0.7] means many trees do not see regime features
3. If regime features rank in bottom 5 of importance, they are effectively ignored (no harm)
4. max_depth [2, 5] range still includes shallow trees that skip interaction features

### Risk 3: LightGBM/CatBoost Perform Worse Than XGBoost (MODERATE)

**Scenario**: On this specific dataset, LightGBM and CatBoost do not add diversity. All 3 learners make similar predictions, and stacking degrades to averaging (no improvement).

**Probability**: 30-40%.

**Evidence for concern**: The dataset is small (~1500 train) and low-signal (prediction std 0.023 vs actual std 1.4%). All GBDT variants may converge to similar patterns.

**Mitigation**:
1. Automatic fallback to single XGBoost if stacking is worse
2. Different random seeds and growth strategies (leaf-wise vs level-wise vs ordered) maximize diversity
3. Even marginal diversity (10-20% different predictions) can improve DA at the margin

### Risk 4: Bootstrap HCDA Degrades (LOW)

**Scenario**: If stacking predictions have different variance characteristics than single XGBoost, the bootstrap confidence scores may be miscalibrated.

**Probability**: 10-15%.

**Mitigation**: Bootstrap still uses XGBoost-only ensemble (5 models), independent of stacking. The |prediction| HCDA method (which was the primary method in attempt 7 at 64.13%) does not depend on bootstrap at all.

### Risk 5: MAE Target Remains Infeasible (CERTAIN)

**Probability**: >95%.

**Mitigation**: Accept. MAE is waived. Focus on DA, HCDA, Sharpe.

### Risk 6: Kaggle Execution Timeout (LOW)

**Scenario**: 260 total Optuna trials take longer than expected.

**Probability**: 10-15%.

**Mitigation**:
1. CPU mode gives 12-hour quota (vs 9 hours GPU)
2. Individual timeouts (1hr + 40min + 50min + overhead = ~3 hours worst case)
3. LightGBM is 2-3x faster than XGBoost per trial, so 80 LightGBM trials ~ 30-40 XGBoost trials

---

## 9. Expected Outcomes

| Metric | Attempt 7 (actual) | Attempt 8 (expected) | Delta | Confidence |
|--------|-------------------|---------------------|-------|------------|
| DA | 60.04% | 60.3-61.0% | +0.3-1.0pp | Medium |
| HCDA | 64.13% | 64.5-65.5% | +0.4-1.4pp | Medium |
| MAE | 0.943% | 0.92-0.95% | -0.02-+0.01 | Low |
| Sharpe | 2.46 | 2.4-2.8 | -0.06-+0.34 | Medium-High |
| Train-test DA gap | -5.28pp | -6 to 0pp | -- | High |
| Targets passed | 3/4 | 3/4 | 0 | High |

**Probability of outcomes**:

| Outcome | Probability |
|---------|------------|
| Improvement on >= 2 metrics (DA, HCDA, Sharpe) | 45-55% |
| No change (fallback to XGBoost single model) | 25-35% |
| Regression (worse than attempt 7) | 10-15% |
| All 4 targets met | <3% |

**Important note**: Attempt 7 already performs very well. The expected improvements are incremental. A "no change" result (where the fallback mechanism reverts to single XGBoost) is an acceptable outcome -- it confirms that the attempt 7 architecture is near-optimal for this data.

---

## 10. Success Criteria

### Primary Targets (on test set)

| Metric | Target | Attempt 7 Actual | Note |
|--------|--------|------------------|------|
| DA | > 56% | 60.04% (PASS) | Maintain or improve |
| HCDA | > 60% | 64.13% (PASS) | Maintain or improve |
| MAE | < 0.75% | 0.943% (FAIL, waived) | Not targeted |
| Sharpe | > 0.80 | 2.46 (PASS) | Maintain or improve |

### Attempt 8 Specific Success Criteria

| Criterion | Threshold | Purpose |
|-----------|-----------|---------|
| No regression vs attempt 7 | DA >= 59.5%, Sharpe >= 2.3 | Guard against degradation |
| Stacking adds value | stacking_val_composite > single_xgb_val_composite | Validate the architectural change |
| Regime features contribute | >= 1 regime feature in top 15 importance | Validate feature engineering |
| Ridge coefficients reasonable | All 3 base learner weights > 0 | Confirm ensemble diversity |

### Decision Rules After Evaluation

| Outcome | Action |
|---------|--------|
| DA > 60.0% AND HCDA > 64.1% AND Sharpe > 2.4 | Improvement confirmed. Accept as new best. |
| DA >= 59.5% AND metrics stable | Marginal. Compare in detail with attempt 7. Accept if any improvement. |
| DA < 59.5% OR Sharpe < 2.3 | Regression. Revert to attempt 7 as final model. |
| Stacking not used (fallback triggered) | No improvement from stacking. Report attempt 7 equivalence. |

---

## 11. Comparison with Previous Attempts

| Aspect | Att 2 | Att 5 | Att 7 (current best) | Att 8 (this) |
|--------|-------|-------|---------------------|--------------|
| Architecture | Single XGBoost | Single XGBoost | Single XGBoost | **3-GBDT Stacking** |
| Features | 22 | 23 | 24 | **30** (24+6 regime) |
| HP Search | 80 trials | 100 trials | 100 trials | **260 trials** (100+80+80) |
| Meta-learner | None | None | None | **Ridge** |
| Obj weights | 50/30/10/10 | 40/30/10/20 | 40/30/10/20 | **35/35/10/20** |
| max_depth | [2,4] | [2,5] | [2,4] | **[2,5]** |
| enable_gpu | true | true | true | **false** |
| HCDA method | |pred| | |pred| | Bootstrap+|pred| | Bootstrap+|pred| |
| OLS scaling | No | No | Yes | Yes |
| Fallback | No | No | Att 2 params | **Att 2 params + single XGBoost** |

---

**End of Design Document**

**Architect**: architect (Opus 4.6)
**Date**: 2026-02-17
**Based on**: Attempt 7 evaluation + researcher meta_model_attempt_8 research report (fact-checked)
**Supersedes**: docs/design/meta_model_attempt_7.md
