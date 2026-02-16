# Meta-Model Design Document: Weekly (5-Day Return) Attempt 2

## 0. Fact-Check Results

### 0.1 Attempt 1 Failure Analysis -- VERIFIED from logs/evaluation/meta_model_weekly_attempt_1.json

| Evidence | Value | Source |
|----------|-------|--------|
| Test DA vs naive | 0.00pp (66.08% = naive) | evaluation JSON gate1.model_vs_naive |
| Unique predictions | 10 out of 457 test samples | evaluation JSON gate1.constant_output |
| All predictions positive | 100% | evaluation JSON gate1.constant_output |
| Prediction std (raw) | 0.0055 | training_result.json prediction_characteristics |
| Trades in test | 0 | evaluation JSON gate1.leak_check |
| Best params: max_depth | 2 | training_result.json model_config.best_params |
| Best params: min_child_weight | 21 | training_result.json model_config.best_params |
| OLS alpha | 2.76 | training_result.json ols_scaling |
| Top 5 Optuna val_da | All 53.07% (= naive) | training_result.json optuna_search.top_5_trials |
| Train samples | 2129 | training_result.json model_config |
| Val samples | 456 | training_result.json model_config |
| Test samples | 457 | training_result.json model_config |

### 0.2 Root Cause Chain -- VERIFIED

```
Overlapping 5-day targets (4/5 day overlap between adjacent rows)
  -> Massive target autocorrelation in training set
  -> XGBoost with strong regularization (max_depth=2, mcw=21) learns only the unconditional mean
  -> Unconditional mean is positive (~0.3-0.5% weekly)
  -> All predictions positive, near-constant
  -> DA = naive always-up fraction
  -> 0 trades, Sharpe = buy-and-hold
```

### 0.3 Non-Overlapping Sample Counts -- COMPUTED

| Split | Overlapping | Non-overlapping (every 5th) | Ratio |
|-------|-------------|---------------------------|-------|
| Train | 2129 | 425 | 5.01x reduction |
| Val | 456 | 91 | 5.01x reduction |
| Test | 457 | 91 | 5.02x reduction |
| Samples per feature (train) | 88.7:1 | 17.7:1 | Borderline but viable |

### 0.4 XGBoost with 425 Samples -- FEASIBILITY CHECK

- 17.7 samples per feature: above the minimum ~10:1 rule of thumb for tree models
- With max_depth=6 and min_child_weight=5: max ~64 leaves, typical ~20-30 leaves
- 425 / 25 leaves = 17 samples per leaf average: sufficient for stable splits
- XGBoost handles small datasets better than neural networks due to sequential tree construction
- Risk: overfitting with too many trees. Mitigation: early stopping + moderate n_estimators cap

**Verdict**: 425 independent samples with 24 features is viable for XGBoost but requires careful regularization. The improved signal-to-noise ratio from independent targets should more than compensate for the sample reduction.

### 0.5 Evaluator Improvement Plan -- CROSS-CHECKED

| Priority | Evaluator Recommendation | Architect Decision |
|----------|------------------------|--------------------|
| 1 | Non-overlapping training (every 5th row) | ADOPT -- eliminates root cause |
| 2 | Naive-aware DA in objective | ADOPT -- prevents trivial solutions |
| 3 | Relaxed regularization | ADOPT with bounds -- prevent overfitting |
| 4 | Centered targets (subtract train mean) | ADOPT -- removes positive bias |
| 5 | Approach B Sharpe in objective | PARTIAL -- use trade-activity gate instead |

### 0.6 Summary

| Check | Verdict |
|-------|---------|
| Attempt 1 failure diagnosis | CONFIRMED -- trivial always-positive predictor |
| Non-overlapping feasibility | PASS -- 425 samples, 17.7:1 ratio |
| Evaluator plan items | 4/5 adopted directly, 1 modified |
| Dataset reference | PASS -- same dataset, no update needed |

---

## 1. Overview

- **Purpose**: Fix the collapsed-to-constant prediction problem of Attempt 1. The model must demonstrate directional skill above the naive always-up strategy by training on independent (non-overlapping) 5-day returns and using an objective that explicitly rewards skill over naive baselines.
- **Architecture**: Single XGBoost model with reg:squarederror + Bootstrap confidence (5 models) + OLS output scaling. Same architecture as Attempt 1 but with 5 fundamental changes to training.
- **Key Changes from Attempt 1**:
  1. **Non-overlapping training**: Every 5th row only (425 train, 91 val samples)
  2. **Centered targets**: Subtract training mean to remove positive bias
  3. **Naive-aware objective**: DA and Sharpe components measure skill above naive, not raw values
  4. **Relaxed regularization**: Wider HP ranges to allow conditional variation
  5. **Trade-activity gate**: Zero Sharpe reward if model makes fewer than 5 position changes
- **What is NOT changed**: Feature set (24), data pipeline, NaN imputation, submodel loading, bootstrap ensemble (5 models), OLS scaling mechanics, Kaggle dataset.
- **Expected Effect**: Model produces meaningful directional variation (both positive and negative predictions), achieving DA genuinely above naive. Trade-off: higher variance in metrics due to fewer samples.

---

## 2. Data Specification

### 2.1 Target Variable

Identical computation to Attempt 1:

```python
gold_df['gold_return_5d'] = (gold_df['gold_price'].shift(-5) / gold_df['gold_price'] - 1) * 100
```

### 2.2 Non-Overlapping Sampling

**Critical change**: Train and validate on every 5th row only.

```python
# After standard 70/15/15 split on ALL rows (same as Attempt 1)
train_df_full = final_df.iloc[:n_train]  # ~2129 rows
val_df_full = final_df.iloc[n_train:n_train+n_val]  # ~456 rows
test_df_full = final_df.iloc[n_train+n_val:]  # ~457 rows

# Non-overlapping subsampling for training and validation
train_df = train_df_full.iloc[::5].copy()  # Every 5th row -> ~425
val_df = val_df_full.iloc[::5].copy()      # Every 5th row -> ~91

# IMPORTANT: Keep test_df as full overlapping for evaluation continuity
# Non-overlapping test metrics are computed as secondary diagnostics
test_df = test_df_full.copy()  # ~457 rows (all)
```

**Design rationale**:
- Training on non-overlapping rows eliminates 4/5-day target autocorrelation
- This forces XGBoost to learn from genuinely independent weekly outcomes
- Test set remains overlapping for comparability with Attempt 1 metrics
- Non-overlapping test metrics reported as secondary diagnostics

### 2.3 Target Centering

```python
# Compute train mean on non-overlapping training targets
train_mean_5d = y_train.mean()
print(f"Training mean weekly return: {train_mean_5d:.4f}%")

# Center all targets
y_train_centered = y_train - train_mean_5d
y_val_centered = y_val - train_mean_5d
# Test targets NOT centered (centering only affects training)
# But for evaluation, predictions are un-centered: pred_final = pred_centered + train_mean_5d
```

**Design rationale**:
- Attempt 1 train mean was ~0.3-0.5% (positive), creating a basin of attraction for always-positive predictions
- After centering, the mean target is 0.0. The model must learn to predict both positive and negative deviations
- At prediction time, add back `train_mean_5d` to restore the original scale
- Direction accuracy is computed on un-centered predictions vs un-centered actuals

**Critical implementation note**:
- XGBoost trains on `y_train_centered` and `y_val_centered`
- All raw predictions are centered (mean near 0)
- Before computing DA, Sharpe, MAE: `pred = pred_centered + train_mean_5d`
- OLS scaling is applied AFTER un-centering

### 2.4 Input Data

Identical to Attempt 1. All 24 features, same 8 submodel CSVs, same date handling.

### 2.5 Feature Set (24 features -- unchanged)

```python
FEATURE_COLUMNS = [
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

TARGET = 'gold_return_5d'
assert len(FEATURE_COLUMNS) == 24
```

### 2.6 Data Split

```python
# Step 1: Standard 70/15/15 on full dataset (same date boundaries as Attempt 1)
n_total = len(final_df)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)

train_df_full = final_df.iloc[:n_train]
val_df_full = final_df.iloc[n_train:n_train+n_val]
test_df_full = final_df.iloc[n_train+n_val:]

# Step 2: Non-overlapping subsampling for train/val
train_df = train_df_full.iloc[::5].copy()
val_df = val_df_full.iloc[::5].copy()
test_df = test_df_full.copy()  # Full test for evaluation

# Step 3: Center targets
train_mean_5d = train_df[TARGET].mean()
```

Expected sizes:
- Train (non-overlapping): ~425 rows
- Val (non-overlapping): ~91 rows
- Test (full overlapping): ~457 rows

### 2.7 NaN Imputation

Identical to Attempt 1:
- Regime probability columns (7 cols) -> 0.5
- Z-score columns (8 cols) -> 0.0
- Divergence/signal columns (2 cols) -> 0.0
- Continuous state columns (2 cols) -> median

---

## 3. Model Architecture

### 3.1 Architecture: Single XGBoost + Bootstrap Confidence + OLS Scaling

Same pipeline structure as Attempt 1. Changes are in training data, targets, objective, and HP ranges.

```
Input: 24-dimensional feature vector (same as Attempt 1)
  |
  v
XGBoost Ensemble (gradient boosted trees)
  - Objective: reg:squarederror
  - CHANGE: Trained on NON-OVERLAPPING samples (~425 train)
  - CHANGE: Trained on CENTERED targets (mean-subtracted)
  - CHANGE: Relaxed regularization (max_depth [2,6], min_child_weight [5,25])
  |
  v
Raw Output: Single scalar (centered prediction)
  |
  v
UN-CENTER: Add train_mean_5d back to predictions
  |
  v
POST-TRAINING STEP 1: OLS Output Scaling
  - Computed on non-overlapping validation set (~91 samples)
  - alpha_ols capped [0.5, 10.0]
  |
  v
POST-TRAINING STEP 2: Bootstrap Ensemble Confidence
  - 5 models with seeds [42, 43, 44, 45, 46]
  - Confidence = 1 / (1 + std_across_models)
  |
  v
Output Metrics: DA, HCDA, MAE, Sharpe -- ALL computed on un-centered predictions
```

### 3.2 Metric Functions

All metric functions from Attempt 1 are reused unchanged. The centering is handled transparently: all metrics receive un-centered predictions and original targets.

#### 3.2.1 Direction Accuracy (unchanged)
#### 3.2.2 MAE (unchanged)
#### 3.2.3 Sharpe -- compute_sharpe_weekly_simple (unchanged)
#### 3.2.4 Sharpe -- compute_sharpe_weekly_rebalance / Approach A (unchanged)
#### 3.2.5 HCDA functions (unchanged)

### 3.3 Naive Baseline Computation

**New requirement**: Compute naive baselines on the same data splits for explicit comparison.

```python
def compute_naive_da(y_true):
    """DA of naive always-up strategy (predict positive for all samples)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return (y_true[mask] > 0).mean()

def compute_naive_sharpe(y_true, cost_bps=5.0):
    """Sharpe of buy-and-hold (always long) strategy."""
    positions = np.ones(len(y_true))
    strategy_returns = positions * y_true / 100.0
    # Single entry trade cost
    position_changes = np.zeros(len(positions))
    position_changes[0] = 1.0  # Enter at start
    trade_costs = position_changes * (cost_bps / 10000.0)
    net_returns = strategy_returns - trade_costs
    if len(net_returns) < 2 or net_returns.std() == 0:
        return 0.0
    return (net_returns.mean() / net_returns.std()) * np.sqrt(52)
```

---

## 4. Hyperparameter Specification

### 4.1 Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective | reg:squarederror | Standard MSE. Unchanged. |
| early_stopping_rounds | 50 | REDUCED from 100. With 425 training samples, models converge faster. 50 rounds is sufficient patience. |
| eval_metric | rmse | Standard |
| tree_method | hist | Fast |
| verbosity | 0 | Suppress |
| seed | 42 + trial.number | Reproducible |

### 4.2 Optuna Search Space -- RELAXED

| Parameter | Attempt 1 Range | Attempt 2 Range | Scale | Rationale |
|-----------|----------------|-----------------|-------|-----------|
| max_depth | [2, 4] | **[2, 6]** | int | Allow deeper trees to capture conditional patterns. With 425 samples and early stopping, overfitting risk is controlled. |
| n_estimators | [100, 800] | **[50, 500]** | int | REDUCED upper bound. Fewer samples need fewer trees. Lower bound reduced to allow light models. |
| learning_rate | [0.001, 0.05] | **[0.005, 0.1]** | log | WIDENED. Higher LR helps with fewer samples (converge before overfitting). |
| colsample_bytree | [0.2, 0.7] | **[0.3, 0.8]** | linear | Slightly wider. With fewer samples, using more features per tree helps. |
| subsample | [0.4, 0.85] | **[0.5, 1.0]** | linear | WIDENED upward. With only 425 samples, 40% subsample = 170 samples is too few for stable splits. |
| min_child_weight | [12, 25] | **[3, 20]** | int | SIGNIFICANTLY RELAXED. Attempt 1's mcw=21 meant each leaf needed 21 samples -- too coarse for 425-sample dataset. mcw=3 allows fine-grained splits. |
| reg_lambda (L2) | [1.0, 15.0] | **[0.1, 10.0]** | log | RELAXED lower bound. Allow less regularization to permit conditional variation. |
| reg_alpha (L1) | [0.5, 10.0] | **[0.01, 5.0]** | log | RELAXED. Same reasoning. L1 sparsity less needed with 24 meaningful features. |

**Total: 8 hyperparameters** (unchanged count).

**Key changes summary**:
1. min_child_weight floor dropped from 12 to 3 (most impactful -- enables fine-grained conditional splits)
2. max_depth ceiling raised from 4 to 6 (allows richer interaction modeling)
3. Both regularization terms (lambda, alpha) have lower floors (permit more expressiveness)
4. n_estimators ceiling reduced from 800 to 500 (fewer samples need fewer trees)
5. subsample floor raised from 0.4 to 0.5 (protect against too-small bootstrap samples)

### 4.3 Search Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_trials | 100 | Same as Attempt 1. With faster training (425 samples), each trial is ~3x faster. |
| timeout | 3600 sec | REDUCED from 7200. Faster individual trials + same n_trials. 1 hour is ample. |
| sampler | TPESampler(seed=42) | Reproducible |
| pruner | None | Not needed for XGBoost |
| direction | maximize | Higher composite is better |

### 4.4 Optuna Objective Function -- REDESIGNED

The objective function is the most critical change. Every component is redesigned to prevent trivial solutions.

```python
def optuna_objective(trial):
    """
    Naive-aware objective function for weekly prediction.

    Key differences from Attempt 1:
    1. Train on centered targets (y_train_centered, y_val_centered)
    2. Un-center predictions before computing metrics
    3. DA component measures SKILL above naive, not raw DA
    4. Sharpe component has trade-activity gate
    5. Constant-output penalty prevents near-constant predictions
    """

    # === Sample hyperparameters (RELAXED RANGES) ===
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.8),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'seed': 42 + trial.number,
    }

    n_estimators = trial.suggest_int('n_estimators', 50, 500)

    # === Train model on CENTERED targets ===
    model = xgb.XGBRegressor(
        **params, n_estimators=n_estimators, early_stopping_rounds=50
    )
    model.fit(
        X_train, y_train_centered,
        eval_set=[(X_val, y_val_centered)],
        verbose=False
    )

    # === Predictions (un-center for metric computation) ===
    train_pred_centered = model.predict(X_train)
    val_pred_centered = model.predict(X_val)

    train_pred = train_pred_centered + train_mean_5d
    val_pred = val_pred_centered + train_mean_5d

    # === Compute naive baselines ===
    naive_da_val = compute_naive_da(y_val)

    # === Compute model metrics ===
    train_da = compute_direction_accuracy(y_train, train_pred)
    val_da = compute_direction_accuracy(y_val, val_pred)
    val_mae = compute_mae(y_val, val_pred)
    val_sharpe = compute_sharpe_weekly_simple(y_val, val_pred)
    val_hc_da, val_hc_coverage = compute_hcda(y_val, val_pred, threshold_percentile=80)

    # === COMPONENT 1: DA SKILL (35%) ===
    # Measures directional accuracy ABOVE naive always-up
    da_skill_pp = (val_da - naive_da_val) * 100  # In percentage points
    da_skill_norm = np.clip(da_skill_pp / 10.0, -0.5, 1.0)
    # Maps: -5pp -> -0.5, 0pp -> 0, +10pp -> 1.0
    # Negative skill is penalized, not just ignored

    # === COMPONENT 2: SHARPE with trade-activity gate (30%) ===
    # Require minimum position variation to earn Sharpe reward
    positions = np.sign(val_pred)
    n_position_changes = np.sum(np.abs(np.diff(positions)) > 0)

    if n_position_changes < 3:
        # Fewer than 3 position changes = effectively constant direction
        sharpe_norm = 0.0
    else:
        sharpe_norm = np.clip((val_sharpe + 2.0) / 4.0, 0.0, 1.0)
        # Maps [-2, +2] to [0, 1]. Tighter range than Attempt 1's [-3, +3]/6

    # === COMPONENT 3: MAE (15%) ===
    mae_norm = np.clip((2.5 - val_mae) / 1.5, 0.0, 1.0)
    # Same as Attempt 1: maps [1.0%, 2.5%] to [1, 0]

    # === COMPONENT 4: HCDA SKILL (20%) ===
    naive_hcda_val = compute_naive_da(y_val)  # Naive DA on full val
    hcda_skill_pp = (val_hc_da - naive_hcda_val) * 100
    hcda_skill_norm = np.clip(hcda_skill_pp / 10.0, -0.5, 1.0)

    # === OVERFITTING PENALTY ===
    da_gap = (train_da - val_da) * 100
    overfit_penalty = max(0.0, (da_gap - 8.0) / 20.0)
    # Stricter: penalty starts at 8pp gap (was 10pp), ramps faster

    # === CONSTANT-OUTPUT PENALTY ===
    pred_std = np.std(val_pred_centered)  # Variation in centered predictions
    if pred_std < 0.01:
        constant_penalty = 1.0  # Nuclear: effectively zero objective
    elif pred_std < 0.1:
        constant_penalty = (0.1 - pred_std) / 0.09 * 0.5
    else:
        constant_penalty = 0.0

    # === COMPOSITE OBJECTIVE ===
    objective = (
        0.35 * da_skill_norm +
        0.30 * sharpe_norm +
        0.15 * mae_norm +
        0.20 * hcda_skill_norm
    ) - 0.30 * overfit_penalty - constant_penalty

    # === Log trial details ===
    trial.set_user_attr('val_da', float(val_da))
    trial.set_user_attr('val_mae', float(val_mae))
    trial.set_user_attr('val_sharpe', float(val_sharpe))
    trial.set_user_attr('val_hc_da', float(val_hc_da))
    trial.set_user_attr('val_hc_coverage', float(val_hc_coverage))
    trial.set_user_attr('train_da', float(train_da))
    trial.set_user_attr('da_gap_pp', float(da_gap))
    trial.set_user_attr('da_skill_pp', float(da_skill_pp))
    trial.set_user_attr('naive_da_val', float(naive_da_val))
    trial.set_user_attr('n_position_changes', int(n_position_changes))
    trial.set_user_attr('pred_std_centered', float(pred_std))
    trial.set_user_attr('constant_penalty', float(constant_penalty))
    trial.set_user_attr('n_estimators_used',
                         int(model.best_iteration + 1) if hasattr(model, 'best_iteration')
                         and model.best_iteration is not None else n_estimators)

    return objective
```

### 4.5 Objective Component Analysis

| Component | Weight | Attempt 1 Behavior | Attempt 2 Design | Always-Up Score |
|-----------|--------|--------------------|--------------------|-----------------|
| DA | 30% -> 35% | Raw DA = naive DA = 0.43 norm | DA skill above naive | 0.0 (0pp skill) |
| Sharpe | 40% -> 30% | Always-long got 0.68 norm | Trade-gate: 0 if <3 changes | 0.0 (gated) |
| MAE | 10% -> 15% | Near-zero prediction, low MAE norm | Same formula, higher weight | Moderate |
| HCDA | 20% -> 20% | HCDA = naive DA on subset | HCDA skill above naive | 0.0 (0pp skill) |
| Overfit | -30% | Not triggered (-12pp gap) | Stricter: starts at 8pp | 0.0 |
| Constant | 0% -> penalty | Not present | -1.0 if std < 0.01 | -1.0 penalty |

**Always-up model under new objective**: ~0.15 * mae_norm - 1.0 constant_penalty = strongly negative. Optuna will avoid this region.

---

## 5. Training Configuration

### 5.1 Training Algorithm

```
1. DATA PREPARATION:
   a. Identical to Attempt 1: API fetch, base features, submodel merge, NaN imputation
   b. Compute gold_return_5d and gold_return_daily (same as Attempt 1)
   c. Standard 70/15/15 split on full dataset
>> d. NON-OVERLAPPING SUBSAMPLING: train_df = train_df_full.iloc[::5]
>> e. NON-OVERLAPPING SUBSAMPLING: val_df = val_df_full.iloc[::5]
   f. test_df = test_df_full (keep overlapping for evaluation)
>> g. CENTERING: train_mean_5d = y_train.mean()
>> h. y_train_centered = y_train - train_mean_5d
>> i. y_val_centered = y_val - train_mean_5d
   j. Compute naive_da_val = (y_val > 0).sum() / len(y_val)
   k. Verify: 24 features, 0 remaining NaN

2. OPTUNA HPO (100 trials, 1-hour timeout):
>> a. Train on y_train_centered, evaluate on y_val_centered
>> b. Un-center predictions before metric computation
>> c. Use naive-aware DA/HCDA components
>> d. Trade-activity gate on Sharpe component
>> e. Constant-output penalty

3. FALLBACK EVALUATION:
   a. Use Attempt 1 best params as fallback (max_depth=2, mcw=21)
   b. ALSO try a "medium" configuration: max_depth=4, mcw=8, moderate regularization
   c. Select best of {Optuna best, Attempt 1 fallback, medium config}

4. FINAL MODEL TRAINING:
>> a. Train on y_train_centered with selected params
>> b. Predict: pred_centered + train_mean_5d = pred_final

5. POST-TRAINING STEP 1: OLS OUTPUT SCALING:
>> a. Computed on non-overlapping val set (91 samples)
>> b. Applied to un-centered predictions
   c. alpha_ols capped [0.5, 10.0]

6. POST-TRAINING STEP 2: BOOTSTRAP ENSEMBLE CONFIDENCE:
>> a. All 5 models trained on y_train_centered
>> b. Predictions un-centered before computing std
   c. Seeds [42, 43, 44, 45, 46]

7. EVALUATION ON ALL SPLITS:
   a. All metrics computed on un-centered predictions vs original targets
   b. Overlapping metrics on full test set (457 samples)
   c. Non-overlapping metrics on every-5th test set (91 samples)
   d. Approach A Sharpe (daily returns, weekly rebalance)
>> e. MANDATORY: Compare DA vs naive_always_up on each split
>> f. MANDATORY: Report n_unique_predictions, n_position_changes, pred_std
>> g. MANDATORY: Report Optuna top-5 trial DA_skill values
   h. Feature importance, quarterly breakdown, decile analysis

8. SAVE RESULTS:
   a. Same files as Attempt 1 plus new diagnostic fields
   b. Add: naive comparison, centering parameters, non-overlapping training info
```

### 5.2 Fallback Configuration -- UPDATED

Three fallback configurations tested:

```python
# Fallback A: Attempt 1 best params (conservative)
FALLBACK_A_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 2, 'min_child_weight': 21,
    'reg_lambda': 5.19, 'reg_alpha': 2.04,
    'subsample': 0.459, 'colsample_bytree': 0.375,
    'learning_rate': 0.017,
    'tree_method': 'hist', 'eval_metric': 'rmse', 'verbosity': 0, 'seed': 42,
}
FALLBACK_A_N_EST = 175

# Fallback B: Medium expressiveness (new)
FALLBACK_B_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 4, 'min_child_weight': 8,
    'reg_lambda': 2.0, 'reg_alpha': 0.5,
    'subsample': 0.7, 'colsample_bytree': 0.6,
    'learning_rate': 0.03,
    'tree_method': 'hist', 'eval_metric': 'rmse', 'verbosity': 0, 'seed': 42,
}
FALLBACK_B_N_EST = 200
```

**Selection rule**: Evaluate both fallbacks with the new naive-aware objective. Select whichever produces the highest composite score. Compare against Optuna best. Use the overall best.

### 5.3 Loss Function

- reg:squarederror on centered targets (unchanged)

### 5.4 Early Stopping

- Metric: RMSE on centered validation set
- Patience: 50 rounds (REDUCED from 100 -- faster convergence with fewer samples)
- Maximum rounds: Optuna-controlled (50-500)

---

## 6. Evaluation Framework

### 6.1 Primary Targets (on test set, overlapping evaluation)

| Metric | Target | Method | Change from Attempt 1 |
|--------|--------|--------|----------------------|
| DA | > 56% | sign agreement, excluding zeros | Same target |
| HCDA | > 60% | top 20% by BEST of (bootstrap, \|pred\|) | Same target |
| MAE | < 1.70% | BEST of (raw, OLS-scaled) | Same target |
| Sharpe | > 0.8 | Approach A: daily returns, weekly rebalance | Same target |

### 6.2 Substantive Skill Tests (NEW -- MANDATORY)

These are not formal targets but are required diagnostics to assess whether nominal targets represent genuine model skill:

| Test | Requirement | What It Checks |
|------|-------------|----------------|
| DA vs naive (test) | DA > naive_always_up_da + 0.5pp | Model has directional skill |
| DA vs naive (val) | DA > naive_always_up_da + 0.5pp | Not just test-period artifact |
| Prediction diversity | n_unique_predictions > 50 | Model produces varied output |
| Trade activity | n_position_changes > 10 in test | Model generates trading signals |
| Prediction balance | positive_pct in [30%, 90%] | Not always-one-direction |
| Pred std | pred_std > 0.1 (un-centered) | Meaningful magnitude variation |

### 6.3 Secondary Diagnostics

Same as Attempt 1:
- Non-overlapping DA, MAE, Sharpe (Approach B)
- Train-test DA gap < 10pp
- OLS alpha reasonableness
- Feature importance ranking
- Quarterly breakdown
- Comparison with daily attempt 7

### 6.4 Decision Rules After Evaluation

| Outcome | Action |
|---------|--------|
| 3/4 targets + all substantive tests pass | ACCEPT as weekly model |
| 3/4 targets + DA skill > 1pp above naive | ACCEPT with note |
| 3/4 targets but DA = naive | FAIL (same as Attempt 1 problem) |
| DA > naive by 2pp+ but targets < 3/4 | Genuine skill, attempt+1 with tuning |
| Model still collapsed to constant | Fundamental approach change needed |
| Severe overfitting (gap > 15pp) | Increase regularization in attempt 3 |

---

## 7. Kaggle Execution Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | true | Consistent with previous attempts |
| Estimated execution time | 15-25 minutes | REDUCED from 25-45. Fewer training samples (425 vs 2129) means each Optuna trial is ~5x faster. |
| Estimated memory usage | 1.0 GB | Less data in memory |
| Required pip packages | [] | No additional packages |
| Internet required | true | For data fetching |
| Kaggle Notebook ID | bigbigzabuton/gold-model-training | Same kernel (new version) |
| dataset_sources | bigbigzabuton/gold-prediction-submodels | Same dataset |
| Optuna timeout | 3600 sec | REDUCED from 7200. Faster trials. |

---

## 8. Implementation Instructions

### 8.1 For builder_data

No separate data preparation step needed. The notebook is self-contained.

### 8.2 For builder_model

**Task**: Generate `notebooks/meta_model_weekly_2/train.ipynb` (self-contained Kaggle Notebook)

**Base**: Copy Attempt 1 notebook (`notebooks/meta_model_weekly_1/train.ipynb`) with the following modifications.

#### 8.2.1 Markdown Header (Cell 0)

```markdown
# Gold Meta-Model Training - Weekly (5-Day Return) Attempt 2

**Architecture:** Single XGBoost with reg:squarederror (weekly target)

**Key Changes from Attempt 1 (FAILED - trivial always-positive predictor):**
1. **Non-overlapping training**: Every 5th row only (~425 train, ~91 val)
2. **Centered targets**: Subtract training mean to remove positive bias
3. **Naive-aware objective**: DA/HCDA measure skill above naive, not raw values
4. **Trade-activity gate**: Sharpe component = 0 if < 3 position changes
5. **Relaxed regularization**: max_depth [2,6], min_child_weight [3,20]
6. **Constant-output penalty**: -1.0 if prediction std < 0.01

**Unchanged from Attempt 1:**
- Same 24 features (5 base + 19 submodel outputs)
- Bootstrap variance-based confidence (5 models for HCDA)
- OLS output scaling
- Same metric functions and evaluation targets

**Design:** `docs/design/meta_model_weekly_attempt_2.md`
```

#### 8.2.2 Data Split (Cell 9) -- ADD NON-OVERLAPPING SUBSAMPLING

After the standard 70/15/15 split, add:

```python
# === Non-overlapping subsampling ===
print("\n--- NON-OVERLAPPING SUBSAMPLING ---")
print(f"Full splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Keep full DataFrames for evaluation
train_df_full = train_df.copy()
val_df_full = val_df.copy()
test_df_full = test_df.copy()

# Subsample train and val (every 5th row for independent 5-day windows)
train_df = train_df_full.iloc[::5].copy()
val_df = val_df_full.iloc[::5].copy()
# Test remains full for evaluation comparability

print(f"Non-overlapping: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} (full)")
print(f"Samples per feature (non-overlapping train): {len(train_df) / len(FEATURE_COLUMNS):.1f}:1")
```

#### 8.2.3 Target Centering (Cell 9) -- ADD AFTER SPLIT

```python
# === Target centering ===
X_train = train_df[FEATURE_COLUMNS].values
y_train = train_df[TARGET].values
X_val = val_df[FEATURE_COLUMNS].values
y_val = val_df[TARGET].values
X_test = test_df[FEATURE_COLUMNS].values
y_test = test_df[TARGET].values

# Compute and apply centering
train_mean_5d = y_train.mean()
y_train_centered = y_train - train_mean_5d
y_val_centered = y_val - train_mean_5d

print(f"\nTarget centering:")
print(f"  Train mean (5d return): {train_mean_5d:.4f}%")
print(f"  Centered train mean: {y_train_centered.mean():.6f}% (should be ~0)")
print(f"  Train positive fraction: {(y_train > 0).sum() / len(y_train)*100:.1f}%")
print(f"  Centered train positive fraction: {(y_train_centered > 0).sum() / len(y_train_centered)*100:.1f}%")

# Compute naive baselines
naive_da_train = compute_naive_da(y_train)
naive_da_val = compute_naive_da(y_val)
naive_da_test = compute_naive_da(y_test)
print(f"\nNaive always-up DA:")
print(f"  Train: {naive_da_train*100:.2f}%")
print(f"  Val:   {naive_da_val*100:.2f}%")
print(f"  Test:  {naive_da_test*100:.2f}%")
```

#### 8.2.4 Naive Baseline Functions (Cell 11) -- ADD

```python
def compute_naive_da(y_true):
    """DA of naive always-up strategy."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return (y_true[mask] > 0).mean()
```

#### 8.2.5 Optuna Objective (Cell 13) -- REPLACE ENTIRELY

Replace the entire objective function with the one from Section 4.4 above.

**Critical changes to ensure**:
1. `model.fit(X_train, y_train_centered, eval_set=[(X_val, y_val_centered)], ...)`
2. `train_pred = train_pred_centered + train_mean_5d` before computing DA
3. `val_pred = val_pred_centered + train_mean_5d` before computing all metrics
4. DA skill = val_da - naive_da_val
5. Trade-activity gate on Sharpe component
6. Constant-output penalty
7. New HP ranges: max_depth [2,6], mcw [3,20], etc.
8. `early_stopping_rounds=50` (reduced from 100)

#### 8.2.6 Optuna Run (Cell 14) -- UPDATE TIMEOUT

```python
study.optimize(
    optuna_objective,
    n_trials=100,
    timeout=3600,  # REDUCED from 7200
    show_progress_bar=True
)

# After optimization, print skill analysis
print(f"\nBest trial naive analysis:")
bt = study.best_trial
print(f"  Val DA:       {bt.user_attrs['val_da']*100:.2f}%")
print(f"  Naive DA val: {bt.user_attrs['naive_da_val']*100:.2f}%")
print(f"  DA skill:     {bt.user_attrs['da_skill_pp']:+.2f}pp")
print(f"  Position changes: {bt.user_attrs['n_position_changes']}")
print(f"  Pred std (centered): {bt.user_attrs['pred_std_centered']:.4f}")
```

#### 8.2.7 Fallback (Cell 16) -- ADD SECOND FALLBACK

Add Fallback B (medium expressiveness) alongside the existing Fallback A. Compare three configs.

```python
# Fallback B: Medium expressiveness
FALLBACK_B_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 4, 'min_child_weight': 8,
    'reg_lambda': 2.0, 'reg_alpha': 0.5,
    'subsample': 0.7, 'colsample_bytree': 0.6,
    'learning_rate': 0.03,
    'tree_method': 'hist', 'eval_metric': 'rmse', 'verbosity': 0, 'seed': 42,
}
FALLBACK_B_N_EST = 200

# Train both fallbacks on centered targets
# ... evaluate with naive-aware objective
# Select best of {Optuna, Fallback A, Fallback B}
```

#### 8.2.8 Final Model Training (Cell 18) -- TRAIN ON CENTERED TARGETS

```python
final_model.fit(X_train, y_train_centered, eval_set=[(X_val, y_val_centered)], verbose=False)

# Generate predictions (UN-CENTER)
pred_train_centered = final_model.predict(X_train)
pred_val_centered = final_model.predict(X_val)
pred_test_centered = final_model.predict(X_test)

pred_train = pred_train_centered + train_mean_5d
pred_val = pred_val_centered + train_mean_5d
pred_test = pred_test_centered + train_mean_5d
```

#### 8.2.9 OLS Scaling (Cell 20) -- APPLY ON UN-CENTERED PREDICTIONS

```python
# OLS scaling on un-centered predictions
numerator = np.sum(pred_val * y_val)
denominator = np.sum(pred_val ** 2)
alpha_ols = numerator / denominator if denominator != 0 else 1.0
alpha_ols = np.clip(alpha_ols, 0.5, 10.0)
```

#### 8.2.10 Bootstrap (Cell 22) -- TRAIN ON CENTERED TARGETS

All 5 bootstrap models must train on `y_train_centered` and un-center predictions.

#### 8.2.11 Evaluation (Cell 24) -- ADD SUBSTANTIVE SKILL TESTS

Add after target evaluation:

```python
# === SUBSTANTIVE SKILL TESTS ===
print("\n" + "="*60)
print("SUBSTANTIVE SKILL TESTS")
print("="*60)

naive_da_test_check = compute_naive_da(y_test)
da_vs_naive = test_m['direction_accuracy'] - naive_da_test_check
n_unique = len(np.unique(np.round(pred_test, 6)))
n_pos_changes = np.sum(np.abs(np.diff(np.sign(pred_test))) > 0)
positive_pct = (pred_test > 0).sum() / len(pred_test) * 100
pred_std_test = np.std(pred_test)

skill_tests = {
    'da_above_naive': da_vs_naive > 0.005,         # > 0.5pp
    'prediction_diversity': n_unique > 50,
    'trade_activity': n_pos_changes > 10,
    'prediction_balance': 30 < positive_pct < 90,
    'prediction_variation': pred_std_test > 0.1,
}

for name, passed in skill_tests.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status}")

print(f"\n  DA vs naive: {da_vs_naive*100:+.2f}pp")
print(f"  Unique predictions: {n_unique}")
print(f"  Position changes in test: {n_pos_changes}")
print(f"  Positive prediction %: {positive_pct:.1f}%")
print(f"  Prediction std: {pred_std_test:.4f}")
print(f"\n  Substantive tests passed: {sum(skill_tests.values())}/5")
```

#### 8.2.12 training_result.json (Cell 28) -- ADD NEW FIELDS

```python
training_result['feature'] = 'meta_model_weekly'
training_result['attempt'] = 2
training_result['design_changes'] = {
    'non_overlapping_training': True,
    'target_centering': True,
    'train_mean_5d': float(train_mean_5d),
    'naive_aware_objective': True,
    'trade_activity_gate': True,
    'constant_output_penalty': True,
    'relaxed_regularization': True,
    'non_overlapping_train_samples': len(X_train),
    'non_overlapping_val_samples': len(X_val),
}
training_result['substantive_skill_tests'] = skill_tests
training_result['naive_comparison'] = {
    'naive_da_train': float(naive_da_train),
    'naive_da_val': float(naive_da_val),
    'naive_da_test': float(naive_da_test_check),
    'model_da_test': float(test_m['direction_accuracy']),
    'da_skill_pp': float(da_vs_naive * 100),
}
```

#### 8.2.13 kernel-metadata.json

```json
{
  "id": "bigbigzabuton/gold-meta-weekly-2",
  "title": "Gold Meta-Model Training - Weekly Attempt 2",
  "code_file": "train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["bigbigzabuton/gold-prediction-submodels"],
  "competition_sources": [],
  "kernel_sources": []
}
```

### 8.3 Complete Modification Checklist

Changes from Attempt 1 notebook:

1. Update markdown header (Cell 0) -- describe all 6 changes
2. Add `compute_naive_da()` function (Cell 11)
3. Non-overlapping subsampling after split (Cell 9): `train_df = train_df_full.iloc[::5]`
4. Target centering computation and application (Cell 9)
5. Print naive baseline DA for each split (Cell 9)
6. REPLACE entire Optuna objective (Cell 13) with naive-aware version
7. Update HP ranges in objective: max_depth [2,6], mcw [3,20], etc.
8. Reduce `early_stopping_rounds` to 50 in objective
9. Reduce Optuna timeout to 3600 (Cell 14)
10. Add DA skill analysis print after Optuna (Cell 14)
11. Add Fallback B configuration (Cell 16)
12. Train all models on `y_train_centered` (Cells 16, 18, 22)
13. Un-center predictions: `pred = pred_centered + train_mean_5d` (Cells 16, 18, 22)
14. OLS scaling on un-centered predictions (Cell 20)
15. Add substantive skill tests (Cell 24)
16. Update training_result.json fields (Cell 28)
17. Create kernel-metadata.json with attempt 2

**Items NOT changed** (verify these remain intact):
- Feature columns (24, same list)
- Submodel loading (all 8 CSVs, same paths, same date handling)
- NaN imputation (same rules)
- Metric functions (DA, MAE, Sharpe, HCDA -- same formulas)
- OLS scaling mechanics (same formula, same cap)
- Bootstrap ensemble (5 models, seeds [42-46])
- Target computation: `shift(-5)` price ratio
- Approach A Sharpe computation for final evaluation
- Feature importance reporting
- Quarterly breakdown
- Comparison with daily attempt 7

---

## 9. Risk Mitigation

### Risk 1: Overfitting with 425 Samples and Relaxed Regularization (HIGH)

**Scenario**: With only 425 training samples and max_depth up to 6, the model may overfit to training patterns that don't generalize.

**Probability**: 40-50%.

**Mitigation**:
1. Early stopping (patience=50) is the primary overfitting control
2. Optuna objective includes overfitting penalty (starts at 8pp DA gap)
3. n_estimators capped at 500 (reduced from 800)
4. subsample floor at 0.5 ensures bootstrap diversity
5. min_child_weight floor at 3 still requires 3+ samples per leaf
6. If train-test DA gap > 12pp, the evaluator should recommend tighter regularization for Attempt 3

### Risk 2: 91 Validation Samples Too Noisy for HPO (HIGH)

**Scenario**: With only 91 non-overlapping validation samples, the composite objective has high variance. Optuna may select parameters that perform well on this specific val set but not on test.

**Probability**: 50-60%.

**Mitigation**:
1. Multiple fallback configurations provide alternatives if Optuna best is an outlier
2. Composite objective has 4 components which reduces variance vs single-metric optimization
3. Constant-output penalty and trade-gate prevent degenerate solutions regardless of noise
4. The evaluator checks test performance -- if val and test disagree strongly, this indicates val noise
5. Consider: if this risk materializes, Attempt 3 could use overlapping val set for HPO but non-overlapping for final evaluation

### Risk 3: Centered Target Makes XGBoost Harder to Train (LOW-MODERATE)

**Scenario**: XGBoost initializes at base_score (default 0.5). With centered targets near 0, the initial residuals are already small. Trees may not grow enough.

**Probability**: 15-25%.

**Mitigation**:
1. XGBoost's default base_score=0.5 should be overridden or left to auto-detect (XGBoost 2.0+ auto-detects)
2. Actually: XGBoost `reg:squarederror` with centered targets works fine -- initial prediction = mean of centered targets = ~0, and boosting proceeds normally. This is NOT a real risk.
3. Verify: print base_score after training.

### Risk 4: Model Skill Exists But Below Target Thresholds (MODERATE)

**Scenario**: The model learns genuine conditional patterns (DA > naive by 2-3pp) but cannot reach 56% DA target because the test set's naive DA is 66% and DA below 56% would require many correct negative predictions.

**Probability**: 20-30%.

**Wait -- important clarification**: DA > 56% is easy if the model simply predicts positive for most samples (test is 66% positive). The REAL test is DA vs naive. If the model predicts negative when gold actually goes down, DA could exceed 66%. The 56% target is a floor, not a challenge given the test period.

**Revised assessment**: DA > 56% should be straightforward. The challenge is producing DA significantly ABOVE naive (66%). Any genuine directional skill on negative-return days would push DA well above 66%.

**Mitigation**: If DA is 56-66% (below naive), the model is actively wrong on the margin. If DA is 66-70%, the model has some skill. Target is achievable.

### Risk 5: Non-Overlapping Subsampling Misses Important Patterns (LOW)

**Scenario**: The every-5th-row strategy may systematically miss certain day-of-week effects or temporal patterns.

**Probability**: 10-15%.

**Mitigation**:
1. The starting offset is row 0, so sampling is deterministic
2. Features already capture temporal information (temporal_context_score)
3. Alternative: could use random offset (e.g., offset = trial.number % 5) in Optuna for implicit augmentation, but this adds complexity for marginal benefit
4. Not worth addressing in Attempt 2; consider for Attempt 3 if needed

---

## 10. Expected Outcomes

| Metric | Attempt 1 (Nominal) | Attempt 1 (Substantive) | Attempt 2 Expected | Confidence |
|--------|---------------------|------------------------|--------------------|------------|
| DA (test) | 66.08% | = naive | 58-68% | Medium |
| DA vs naive | +0.00pp | 0 skill | +1-5pp | Medium |
| HCDA | 68.48% | = naive subset | 55-68% | Low |
| MAE | 2.07% | Near zero-pred | 1.5-2.2% | Medium |
| Sharpe (App A) | 2.03 | = buy-hold | 0.5-2.5 | Medium |
| Pred std | 0.0055 | Near constant | > 0.5 | High |
| Unique preds | 10 | Collapsed | > 100 | High |
| Position changes | 0 | No trades | > 20 | High |
| Train-test DA gap | -12.3pp | Misleading | 0-15pp | Medium |

**Probability of outcomes**:

| Outcome | Probability |
|---------|------------|
| Model produces genuine variation (std > 0.1, both signs) | 85-90% |
| DA skill > 0.5pp above naive | 50-60% |
| 3/4 formal targets met | 30-40% |
| 3/4 formal + substantive skill | 25-35% |
| 4/4 targets met | 5-10% |
| Model still collapsed (same as Attempt 1) | 5-10% |
| Severe overfitting (gap > 15pp) | 20-30% |

---

## 11. Success Criteria

### Primary Targets (same as Attempt 1)

| Metric | Target | Method |
|--------|--------|--------|
| DA | > 56% | sign agreement, excluding zeros (overlapping test) |
| HCDA | > 60% | top 20% by BEST of (bootstrap confidence, \|prediction\|) |
| MAE | < 1.70% | BEST of (raw, OLS-scaled) predictions |
| Sharpe | > 0.80 | Approach A: daily returns, weekly rebalance, sqrt(252) |

### Substantive Skill Criteria (NEW -- MANDATORY)

| Test | Requirement |
|------|-------------|
| DA above naive (test) | > +0.5pp |
| Prediction diversity | > 50 unique values |
| Trade activity | > 10 position changes in test |
| Prediction balance | 30-90% positive |
| Prediction variation | std > 0.1 |

### Combined Decision Rules

| Formal Targets | Substantive Skill | Decision |
|----------------|-------------------|----------|
| 3/4+ | All 5 pass | ACCEPT |
| 3/4+ | DA skill > 1pp, rest pass | ACCEPT |
| 3/4+ | DA skill < 0.5pp | FAIL (same as Attempt 1) |
| 2/4 | All 5 pass | Genuine skill, attempt+1 |
| Any | Prediction collapsed | FAIL, fundamental redesign |

---

## 12. Comparison of Attempt 1 vs Attempt 2 Design

| Aspect | Attempt 1 | Attempt 2 |
|--------|-----------|-----------|
| Training samples | 2129 (overlapping) | 425 (non-overlapping) |
| Val samples | 456 (overlapping) | 91 (non-overlapping) |
| Target centering | None (raw returns) | Subtract train mean |
| DA in objective | Raw DA (naive = 53%) | DA skill above naive (naive = 0) |
| Sharpe in objective | Raw Sharpe (always-long rewarded) | Gated: 0 if < 3 position changes |
| Constant-output penalty | None | -1.0 if std < 0.01 |
| max_depth | [2, 4] | [2, 6] |
| min_child_weight | [12, 25] | [3, 20] |
| reg_lambda | [1.0, 15.0] | [0.1, 10.0] |
| reg_alpha | [0.5, 10.0] | [0.01, 5.0] |
| n_estimators | [100, 800] | [50, 500] |
| learning_rate | [0.001, 0.05] | [0.005, 0.1] |
| early_stopping | 100 | 50 |
| Optuna timeout | 7200s | 3600s |
| Objective weights | 40/30/10/20 (Sharpe/DA/MAE/HCDA) | 30/35/15/20 (Sharpe/DA/MAE/HCDA) |
| Overfit penalty start | 10pp | 8pp |
| Fallback configs | 1 (Attempt 2 params) | 2 (Attempt 1 params + medium) |
| Skill tests | None | 5 mandatory substantive tests |

---

**End of Design Document**

**Architect**: architect (Opus 4.6)
**Date**: 2026-02-17
**Based on**: Weekly Attempt 1 FAIL (trivial always-positive predictor, 0/4 substantive)
**Purpose**: Fix collapsed model via non-overlapping training + naive-aware objective + relaxed regularization
