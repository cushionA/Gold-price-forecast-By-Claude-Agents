# Meta-Model Design Document: Attempt 1

## 0. Fact-Check Results

### 0.1 Technical Claims

| Claim | Verdict | Detail |
|-------|---------|--------|
| XGBoost supports custom objective (grad/hess) | PASS | Verified with XGBoost 3.1.2. `xgb.train(obj=custom_obj)` works. |
| XGBoost custom eval metric + early stopping | PASS | `early_stopping_rounds` param works with `custom_metric`. Note: `EarlyStopping` callback uses `maximize` not `minimize`. |
| MAE gradient = sign(pred - true) | PASS | Mathematically correct. Hessian = 0 (non-differentiable), using 1.0 as constant approximation is standard practice. |
| Directional penalty with hard np.where | PASS | Functional for tree models. Gradient discontinuity is not problematic for tree splits (splits are discrete thresholds). |
| L1 (alpha) regularization zeros out features | CAUTION | L1 in XGBoost regularizes leaf weights, not feature selection directly. It encourages smaller leaf weights, which indirectly reduces overfitting. Feature importance reduction is a side effect, not guaranteed feature elimination. |
| Tree models are scale-invariant | PASS | Correct. XGBoost splits based on thresholds; feature scale does not affect split quality. |
| XGBoost handles multicollinearity robustly | PASS | Trees randomly select among correlated features across ensemble. VIF=12.47 on etf_regime_prob is not a training issue for XGBoost. |

### 0.2 Benchmark Claims

| Claim | Verdict | Detail |
|-------|---------|--------|
| AdjMSE2 "doubled Sharpe ratios" (OAAIML paper) | CAUTION | Paper exists but results are on specific equity datasets. Transferability to gold returns is not guaranteed. Directional penalty concept is sound but magnitude of improvement is dataset-dependent. |
| LSTM + HMM regime features: 15-20% MAE improvement | CAUTION | ScienceDirect paper is on credit spreads, not commodities. Improvement magnitude is specific to that domain. |
| Gold LSTM DA = 50.67% (arxiv 2512.22606) | PASS | Paper exists. DA benchmark is reasonable reference. |
| 56% DA is "at the upper end of realistic" | CORRECTED | See critical finding below. |

### 0.3 Critical Finding: Test Set Class Imbalance

**The researcher missed a critical fact about the test set.**

| Split | Samples | Up % | Down % | Naive Always-Up DA |
|-------|---------|------|--------|-------------------|
| Train | 1766 | 51.8% | 47.7% | 51.8% |
| Val | 378 | 51.3% | 48.4% | 51.3% |
| Test | 379 | 56.7% | 43.3% | 56.7% |

The test period (Aug 2023 - Feb 2025) coincides with a major gold rally. A naive "always predict positive" strategy achieves 56.7% DA, which already exceeds the 56% target.

**Implications:**

1. The baseline's 43.5% test DA is WORSE than naive always-up (56.7%), meaning the baseline actively learned wrong patterns from training data that do not generalize.
2. The 56% DA target is not inherently difficult given the test set composition. The real challenge is Sharpe > 0.8, which requires correct magnitude AND direction.
3. The model's prediction bias is important. If the model systematically predicts slightly positive (aligning with the upward trend in test), it will naturally achieve high DA on this specific test set.
4. **Risk**: A model that achieves 56% DA by exploiting this bias may not generalize to future periods with different up/down ratios.

**Design Response:**

- The directional loss function must reward correct DIRECTION regardless of bias, not just magnitude accuracy.
- The Optuna objective should weight Sharpe more heavily than DA, since Sharpe is the binding constraint.
- High-confidence DA (HC-DA > 60%) remains meaningful because it requires the model to distinguish confident from uncertain predictions.

### 0.4 Sharpe Calculation Discrepancy

The existing `src/evaluation.py` deducts cost on EVERY day (`net_returns = returns - cost_pct`), not only on position changes. This is more conservative than real-world trading (where cost is only incurred on trades, not on holding). The meta-model training notebook must use the SAME formula for consistency.

---

## 1. Overview

- **Purpose**: Integrate 19 base features and 20 submodel outputs (39 total) from 7 HMM-based submodels to predict next-day gold return (%). This is the Phase 3 meta-model that combines all Phase 2 submodel work into a final predictor.
- **Architecture**: XGBoost with custom directional-weighted objective function.
- **Rationale**: XGBoost is the baseline model, proven to handle heterogeneous features and multicollinearity. The baseline's failure (DA 43.5%, Sharpe -1.70) is due to overfitting (11pp train-test gap) and direction-agnostic loss, not architecture limitations. Fixing regularization and loss function addresses root causes without introducing architecture risk.
- **Expected Effect**: Close the train-test gap via aggressive regularization (target < 5pp), improve directional accuracy via directional penalty in loss function, and improve Sharpe via properly calibrated predictions.

---

## 2. Data Specification

### 2.1 Input Data

| Source | Path | Rows | Columns |
|--------|------|------|---------|
| Base features | data/processed/base_features.csv | 2523 | 19 (+ target) |
| VIX submodel | data/submodel_outputs/vix.csv | 2857 | 3 |
| Technical submodel | data/submodel_outputs/technical.csv | 2860 | 3 |
| Cross-asset submodel | data/submodel_outputs/cross_asset.csv | 2522 | 3 |
| Yield curve submodel | data/submodel_outputs/yield_curve.csv | 2794 | 3 (use 2, exclude yc_regime_prob) |
| ETF flow submodel | data/submodel_outputs/etf_flow.csv | 2838 | 3 |
| Inflation expectation | data/submodel_outputs/inflation_expectation.csv | 2924 | 3 |
| CNY demand submodel | data/submodel_outputs/cny_demand.csv | 2771 | 3 |
| Target | data/processed/target.csv | 2542 | 1 (gold_return_next) |

### 2.2 Data Merge Strategy

1. Load base_features.csv (2523 rows) as the primary dataframe (index = date).
2. Separate target column (gold_return_next) from features.
3. For each submodel CSV: inner join on date index with base features.
4. Exclude yc_regime_prob column (constant, std=1.07e-11).
5. Drop rows with any NaN values in feature columns.
6. Expected result: approximately 2520 rows, 39 feature columns + 1 target column.

### 2.3 Data Split (frozen from Phase 1)

| Split | Method | Expected Rows |
|-------|--------|---------------|
| Train | First 70% | ~1766 |
| Val | Next 15% | ~378 |
| Test | Last 15% | ~379 |

Time-series order preserved. No shuffling.

### 2.4 Feature List (39 features)

**Base Features (19):**

| # | Column | Type |
|---|--------|------|
| 1 | real_rate_real_rate | Macro (interest rate) |
| 2 | dxy_dxy | FX (dollar index) |
| 3 | vix_vix | Volatility index |
| 4 | technical_gld_open | Price level |
| 5 | technical_gld_high | Price level |
| 6 | technical_gld_low | Price level |
| 7 | technical_gld_close | Price level |
| 8 | technical_gld_volume | Volume |
| 9 | cross_asset_silver_close | Price level |
| 10 | cross_asset_copper_close | Price level |
| 11 | cross_asset_sp500_close | Price level |
| 12 | yield_curve_dgs10 | Interest rate |
| 13 | yield_curve_dgs2 | Interest rate |
| 14 | yield_curve_yield_spread | Spread |
| 15 | etf_flow_gld_volume | Volume |
| 16 | etf_flow_gld_close | Price level |
| 17 | etf_flow_volume_ma20 | Volume MA |
| 18 | inflation_expectation_inflation_expectation | Macro |
| 19 | cny_demand_cny_usd | FX rate |

**Submodel Features (20):**

| # | Column | Source | Type |
|---|--------|--------|------|
| 20 | vix_regime_probability | vix | HMM regime prob [0,1] |
| 21 | vix_mean_reversion_z | vix | z-score |
| 22 | vix_persistence | vix | Continuous state |
| 23 | tech_trend_regime_prob | technical | HMM regime prob [0,1] |
| 24 | tech_mean_reversion_z | technical | z-score |
| 25 | tech_volatility_regime | technical | Regime state |
| 26 | xasset_regime_prob | cross_asset | HMM regime prob [0,1] |
| 27 | xasset_recession_signal | cross_asset | Binary {0,1} |
| 28 | xasset_divergence | cross_asset | Continuous |
| 29 | yc_spread_velocity_z | yield_curve | z-score |
| 30 | yc_curvature_z | yield_curve | z-score |
| 31 | etf_regime_prob | etf_flow | HMM regime prob [0,1] (VIF=12.47) |
| 32 | etf_capital_intensity | etf_flow | z-score |
| 33 | etf_pv_divergence | etf_flow | z-score |
| 34 | ie_regime_prob | inflation_expectation | HMM regime prob [0,1] |
| 35 | ie_anchoring_z | inflation_expectation | z-score |
| 36 | ie_gold_sensitivity_z | inflation_expectation | z-score |
| 37 | cny_regime_prob | cny_demand | HMM regime prob [0,1] |
| 38 | cny_momentum_z | cny_demand | z-score |
| 39 | cny_vol_regime_z | cny_demand | z-score |

---

## 3. Model Architecture

### 3.1 Model Selection: XGBoost (xgboost.train API)

**Why xgb.train (not XGBRegressor)**:
- Custom objective function requires the low-level `xgb.train` API for proper gradient/hessian control.
- XGBRegressor.fit does not support `obj` parameter with early stopping as cleanly.
- Direct DMatrix creation allows full control over data handling.

### 3.2 Architecture Specification

```
Input: 39-dimensional feature vector (mixed types, no preprocessing needed for XGBoost)
  |
  v
XGBoost Ensemble (gradient boosted trees)
  - Custom directional-weighted MAE objective
  - Up to 1000 boosting rounds with early stopping (patience=50)
  - Regularization: max_depth, min_child_weight, subsample, colsample, L1, L2
  |
  v
Output: Single scalar (predicted next-day gold return %)
  |
  v
Post-processing:
  - High-confidence threshold: |prediction| > confidence_threshold
  - Trade signal: sign(prediction) for direction, |prediction| for magnitude
```

### 3.3 Custom Objective Function

**Mathematical Specification:**

The standard MAE loss is:

```
L(y_pred, y_true) = |y_pred - y_true|
```

The directional-weighted MAE adds a penalty when the predicted direction disagrees with the actual direction:

```
penalty(y_pred, y_true) = 1.0               if sign(y_pred) == sign(y_true)
                        = penalty_factor     if sign(y_pred) != sign(y_true)

L_dir(y_pred, y_true) = penalty(y_pred, y_true) * |y_pred - y_true|
```

Where `penalty_factor` is an Optuna hyperparameter in range [1.5, 5.0].

**Gradient and Hessian:**

```
gradient = penalty * sign(y_pred - y_true)
hessian  = penalty * 1.0    (constant approximation; MAE hessian is technically 0)
```

**Implementation:**

```python
def directional_mae_obj(y_pred, dtrain):
    y_true = dtrain.get_label()
    sign_agree = (y_pred * y_true) > 0
    penalty = np.where(sign_agree, 1.0, PENALTY_FACTOR)

    residual = y_pred - y_true
    grad = penalty * np.sign(residual)
    hess = penalty * np.ones_like(y_pred)
    return grad, hess
```

**Why not smooth penalty**: Tree models use discrete split thresholds, not continuous gradient descent. The hard `np.where` discontinuity at the sign boundary does not cause oscillation in XGBoost. A smooth sigmoid penalty would add complexity without benefit.

**Fallback**: If the custom objective causes convergence issues (all predictions collapse to 0), use standard `reg:squarederror` and optimize DA via the Optuna objective function instead.

### 3.4 Custom Evaluation Metric

For early stopping, use a composite metric that balances DA and MAE:

```python
def composite_eval(y_pred, dtrain):
    y_true = dtrain.get_label()

    # MAE component
    mae = np.mean(np.abs(y_pred - y_true))

    # DA component (exclude zeros)
    nonzero = (y_true != 0) & (y_pred != 0)
    if nonzero.sum() > 0:
        da = np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero]))
    else:
        da = 0.5

    # Composite: lower is better
    # Normalize MAE to similar scale as DA (MAE ~ 0.7, DA ~ 0.5)
    # We want to MINIMIZE this score
    score = mae - 0.5 * da  # Lower score = better (lower MAE, higher DA)

    return 'composite', score
```

Early stopping monitors this composite metric on the validation set, stopping when it stops improving for 50 rounds.

### 3.5 Sharpe Calculation (for Optuna Objective)

Must match `src/evaluation.py` exactly:

```python
def compute_sharpe(predictions, actuals, cost_bps=5.0):
    """Compute Sharpe ratio with transaction costs.

    IMPORTANT: Matches src/evaluation.py which deducts cost every day,
    not only on position changes.
    """
    cost_pct = cost_bps / 100.0  # 5bps = 0.05%

    # Strategy: go long when pred > 0, short when pred < 0
    strategy_returns = np.sign(predictions) * actuals

    # Deduct cost every day (matches existing evaluation.py)
    net_returns = strategy_returns - cost_pct

    if len(net_returns) < 2 or np.std(net_returns) == 0:
        return 0.0

    sharpe = (np.mean(net_returns) / np.std(net_returns)) * np.sqrt(252)
    return sharpe
```

---

## 4. Hyperparameter Specification

### 4.1 Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 1000 | Upper bound; early stopping prevents overtraining |
| early_stopping_rounds | 50 | Patience for convergence detection |
| tree_method | hist | Fast histogram-based algorithm |
| disable_default_eval_metric | 1 | Use custom composite eval metric |
| verbosity | 0 | Suppress output for Optuna |
| seed | 42 | Reproducibility (base seed; Optuna trial number added) |

### 4.2 Optuna Search Space

| Parameter | Range | Scale | Type | Rationale |
|-----------|-------|-------|------|-----------|
| max_depth | [3, 6] | linear | int | Controls tree complexity. Baseline used 5 and overfit severely. Range includes conservative (3) and slightly aggressive (6). |
| min_child_weight | [3, 10] | linear | int | Prevents small leaf nodes. With 1766 samples, leaves need >= 3 samples minimum for robustness. |
| subsample | [0.5, 0.8] | linear | float | Row sampling per tree. Lower = more regularization. |
| colsample_bytree | [0.5, 0.8] | linear | float | Column sampling per tree. Critical with 39 features to prevent memorization. |
| reg_lambda (L2) | [1.0, 10.0] | log | float | L2 regularization on leaf weights. Stronger than baseline to combat overfitting. |
| reg_alpha (L1) | [0.1, 5.0] | log | float | L1 regularization. Encourages sparser leaf weights. |
| learning_rate | [0.005, 0.05] | log | float | Slower learning = more trees but less overfitting. Baseline used 0.05 (upper end). |
| gamma | [0.0, 2.0] | linear | float | Minimum loss reduction for tree split. Higher = more conservative. |
| directional_penalty | [1.5, 5.0] | linear | float | Penalty factor for wrong-direction predictions in custom objective. |
| confidence_threshold | [0.002, 0.015] | linear | float | Threshold for high-confidence predictions (0.2% to 1.5% predicted return). |

**Total: 10 hyperparameters** (8 XGBoost + 1 loss function + 1 post-processing)

### 4.3 Search Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_trials | 50 | Minimum per CLAUDE.md. 10 parameters => ~5 trials per parameter. |
| timeout | 14400 (4 hours) | Conservative. Expected: 50 trials * 2-3 min/trial = 100-150 min. Timeout prevents runaway trials. |
| sampler | TPESampler(seed=42) | Tree-structured Parzen Estimator. Standard for mixed continuous/integer spaces. |
| pruner | MedianPruner(n_startup_trials=10) | Prune poor trials early after 10 baseline trials. Not used for XGBoost (fast enough). |
| direction | maximize | Maximize Optuna objective (see below). |

### 4.4 Optuna Objective Function

The Optuna objective must optimize all four targets simultaneously. Since Sharpe is the binding constraint (baseline -1.70 vs target +0.80), it receives the highest weight.

```python
def optuna_objective(trial, X_train, y_train, X_val, y_val):
    # Sample hyperparameters
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),
        'tree_method': 'hist',
        'disable_default_eval_metric': 1,
        'verbosity': 0,
        'seed': 42 + trial.number,
    }

    penalty_factor = trial.suggest_float('directional_penalty', 1.5, 5.0)
    conf_threshold = trial.suggest_float('confidence_threshold', 0.002, 0.015)

    # Custom objective with this trial's penalty factor
    def obj_fn(y_pred, dtrain):
        y_true = dtrain.get_label()
        sign_agree = (y_pred * y_true) > 0
        penalty = np.where(sign_agree, 1.0, penalty_factor)
        grad = penalty * np.sign(y_pred - y_true)
        hess = penalty * np.ones_like(y_pred)
        return grad, hess

    # Train
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    bst = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        obj=obj_fn,
        evals=[(dval, 'val')],
        custom_metric=composite_eval,
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Evaluate on validation set
    val_pred = bst.predict(dval)

    # Compute metrics
    val_mae = np.mean(np.abs(val_pred - y_val))

    nonzero = (y_val != 0) & (val_pred != 0)
    val_da = np.mean(np.sign(val_pred[nonzero]) == np.sign(y_val[nonzero]))

    # High-confidence DA
    hc_mask = np.abs(val_pred) > conf_threshold
    if hc_mask.sum() >= 0.2 * len(val_pred):  # At least 20% of samples
        hc_da = np.mean(np.sign(val_pred[hc_mask & nonzero]) == np.sign(y_val[hc_mask & nonzero]))
        hc_coverage = hc_mask.sum() / len(val_pred)
    else:
        hc_da = 0.0
        hc_coverage = 0.0

    # Sharpe
    strategy_returns = np.sign(val_pred) * y_val
    cost_pct = 5.0 / 100.0
    net_returns = strategy_returns - cost_pct
    if np.std(net_returns) > 0:
        val_sharpe = (np.mean(net_returns) / np.std(net_returns)) * np.sqrt(252)
    else:
        val_sharpe = 0.0

    # Composite objective: weight Sharpe most heavily (binding constraint)
    # Sharpe range: [-3, +3] typical -> normalize to [0, 1] via (sharpe + 3) / 6
    # DA range: [0.3, 0.7] typical -> normalize to [0, 1] via (da - 0.3) / 0.4
    # MAE range: [0.5, 1.0] typical -> normalize to [0, 1] via (1.0 - mae) / 0.5

    sharpe_norm = np.clip((val_sharpe + 3.0) / 6.0, 0.0, 1.0)
    da_norm = np.clip((val_da - 0.3) / 0.4, 0.0, 1.0)
    mae_norm = np.clip((1.0 - val_mae) / 0.5, 0.0, 1.0)
    hc_da_norm = np.clip((hc_da - 0.3) / 0.4, 0.0, 1.0) if hc_da > 0 else 0.0

    # Weighted composite: Sharpe 40%, DA 25%, HC-DA 20%, MAE 15%
    objective_value = (
        0.40 * sharpe_norm +
        0.25 * da_norm +
        0.20 * hc_da_norm +
        0.15 * mae_norm
    )

    # Log metrics for analysis
    trial.set_user_attr('val_mae', float(val_mae))
    trial.set_user_attr('val_da', float(val_da))
    trial.set_user_attr('val_hc_da', float(hc_da))
    trial.set_user_attr('val_sharpe', float(val_sharpe))
    trial.set_user_attr('val_hc_coverage', float(hc_coverage))
    trial.set_user_attr('n_estimators', int(bst.best_iteration + 1))

    return objective_value
```

**Weight Rationale:**
- Sharpe (40%): Binding constraint. Baseline -1.70, target +0.80. Largest gap.
- DA (25%): Important but partially achievable via test set bias (56.7% up days).
- HC-DA (20%): Requires confidence calibration beyond DA.
- MAE (15%): Already met by baseline (0.714 < 0.75). Non-binding.

---

## 5. Training Configuration

### 5.1 Training Algorithm (Pseudocode)

```
1. DATA PREPARATION:
   a. Load base_features.csv, all submodel CSVs, target.csv
   b. Merge on date index (inner join)
   c. Exclude yc_regime_prob column
   d. Drop NaN rows
   e. Split: train (70%), val (15%), test (15%) by time order
   f. Separate features (X) from target (y = gold_return_next)

2. OPTUNA HPO (50 trials):
   a. For each trial:
      - Sample 10 hyperparameters
      - Create custom objective with sampled directional_penalty
      - Train XGBoost on (X_train, y_train) with early stopping on (X_val, y_val)
      - Compute val metrics: MAE, DA, HC-DA, Sharpe
      - Return weighted composite objective
   b. Select best trial (highest composite objective)

3. FINAL MODEL TRAINING:
   a. Re-train with best hyperparameters on (X_train, y_train)
   b. Early stop on (X_val, y_val)
   c. Record best_iteration

4. EVALUATION ON ALL SPLITS:
   a. Predict on train, val, test
   b. Compute all 4 metrics per split
   c. Compute overfit ratios (train/val, train/test)
   d. Compute feature importance
   e. Generate diagnostic plots

5. SAVE RESULTS:
   a. training_result.json: metrics, params, feature importance
   b. model.json: XGBoost model file
   c. predictions.csv: predictions for all splits
   d. submodel_output.csv: final predictions aligned with dates
```

### 5.2 Loss Function

- **Primary**: Custom directional-weighted MAE (described in Section 3.3)
- **Fallback**: If custom objective produces NaN/constant predictions in early Optuna trials, switch to `reg:squarederror` for remaining trials

### 5.3 Optimizer

XGBoost's internal gradient boosting optimizer. No external optimizer needed.

### 5.4 Early Stopping

- **Metric**: Custom composite (MAE - 0.5 * DA) on validation set. Lower is better.
- **Patience**: 50 rounds
- **Maximum rounds**: 1000

### 5.5 Evaluation Metrics

Computed on each split (train, val, test):

| Metric | Formula | Target |
|--------|---------|--------|
| DA | mean(sign(pred) == sign(actual)) excluding pred=0 or actual=0 | > 56% |
| HC-DA | DA on subset where abs(pred) > confidence_threshold, min 20% coverage | > 60% |
| MAE | mean(abs(pred - actual)) | < 0.75% |
| Sharpe | (mean(strategy_returns - 0.05%) / std(strategy_returns - 0.05%)) * sqrt(252) | > 0.8 |
| Overfit Ratio | val_MAE / train_MAE | < 1.5 |

---

## 6. Kaggle Execution Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | XGBoost on 1766 samples with max 1000 rounds is CPU-fast. GPU adds overhead for small data. |
| Estimated execution time | 30-90 minutes | 50 trials * 30-90 sec/trial + data loading + final evaluation |
| Estimated memory usage | 2 GB | 2523 rows * 39 features * 8 bytes = ~0.8 MB data. XGBoost model < 100 MB. |
| Required pip packages | [] | All pre-installed on Kaggle (xgboost, optuna, pandas, numpy, scikit-learn) |
| Internet required | true | For data fetching (yfinance, fredapi) |
| Kaggle Notebook ID | bigbigzabuton/gold-model-training | Unified notebook with FRED_API_KEY secret |
| Timeout guard | 14400 sec (4 hours) for Optuna | Leaves 5-hour margin within Kaggle's 9-hour limit |

---

## 7. Implementation Instructions

### 7.1 For builder_data

**Task**: Create `data/processed/meta_model_input.csv`

1. Load `data/processed/base_features.csv` (index_col=0, parse_dates=True)
2. For each submodel in [vix, technical, cross_asset, yield_curve, etf_flow, inflation_expectation, cny_demand]:
   - Load `data/submodel_outputs/{submodel}.csv` (index_col=0, parse_dates=True)
   - Handle timezone-aware vs naive datetime indices (technical.csv and etf_flow.csv may have timezone info)
3. Merge all dataframes on date index using inner join
4. Exclude column `yc_regime_prob` from yield_curve
5. Verify 39 feature columns + 1 target column (gold_return_next)
6. Drop rows with any NaN
7. Save to `data/processed/meta_model_input.csv`
8. Log: row count, column count, date range, NaN counts before/after cleaning

### 7.2 For builder_model

**Task**: Generate `notebooks/meta_model_1/train.ipynb` (self-contained Kaggle Notebook)

The notebook must be fully self-contained. It fetches all data, trains the model, and saves results.

**Critical Implementation Details:**

1. **Data Fetching Inside Notebook**: The notebook must replicate the data fetching and submodel output generation from scratch. It cannot depend on local CSV files. This means:
   - Fetch base features using yfinance and fredapi
   - Run each HMM submodel to generate outputs
   - OR: Embed the submodel outputs directly as constants/dicts in the notebook
   - **Recommended approach**: Since submodel outputs are deterministic (HMM parameters are fixed), embed the submodel generation code in the notebook. The builder_model should include the HMM fitting and transformation code for each submodel.

2. **Alternative approach (simpler)**: Upload `meta_model_input.csv` as a Kaggle dataset and load it in the notebook. This avoids duplicating all submodel code but requires an extra upload step.

3. **XGBoost Train API**: Use `xgb.train()` (not XGBRegressor) for custom objective support.

4. **Seed Management**: Use `42 + trial.number` as seed for each Optuna trial to ensure reproducibility while maintaining diversity.

5. **Custom Objective**: Define `directional_mae_obj` as a closure that captures `penalty_factor` from the Optuna trial. Must handle edge cases:
   - When `y_pred` is all zeros (early in training)
   - When `y_true` contains zeros (exclude from sign comparison)

6. **Output Files** (saved to Kaggle output):
   - `training_result.json`: All metrics, best params, feature importance, per-split results
   - `model.json`: XGBoost model (use `bst.save_model('model.json')`)
   - `predictions.csv`: Columns = [date, split, prediction, actual, direction_correct, high_confidence]
   - `submodel_output.csv`: Final predictions aligned with date index (for compatibility with existing pipeline)

7. **Diagnostic Information** to include in training_result.json:
   - Train/val/test metrics (all 4 + overfit ratio)
   - Best Optuna trial number and parameters
   - All Optuna trial summaries (trial number, params, objective value, individual metrics)
   - Feature importance (top 20)
   - Number of boosting rounds used (best_iteration)
   - HC-DA coverage (% of test predictions above confidence threshold)
   - Naive always-up DA on test set (for comparison)
   - Total training time

### 7.3 Handling the Data Pipeline in Kaggle

**Recommended Strategy**: Inline data fetching + submodel code

Since the notebook must be self-contained:

1. **Fetch raw data** using yfinance and fredapi (same as Phase 1/2 notebooks)
2. **Construct base features** inline (replicate Phase 1 logic)
3. **Run each submodel's HMM** inline using the saved model parameters from Phase 2
4. **Merge** base features + submodel outputs + target

The builder_model agent should refer to the existing submodel training notebooks in `notebooks/` to extract the HMM fitting and transformation code for each submodel.

**If data fetching is too complex for a single notebook**: Use the alternative approach of embedding pre-computed data. The builder_model can include a function that loads data from a CSV uploaded as a Kaggle dataset.

---

## 8. Risk Mitigation

### Risk 1: Overfitting (PRIMARY RISK)

**Problem**: Baseline showed 11pp train-test DA gap with only 19 features. 39 features amplifies this risk.

**Mitigations**:
- max_depth range [3, 6] (baseline was 5)
- min_child_weight range [3, 10] (baseline had no constraint)
- subsample [0.5, 0.8] (baseline was 0.8)
- colsample_bytree [0.5, 0.8] (baseline was 0.8)
- L2 regularization [1.0, 10.0] (baseline had default ~1.0)
- L1 regularization [0.1, 5.0] (baseline had none)
- gamma [0.0, 2.0] (minimum loss reduction for split)
- learning_rate [0.005, 0.05] (baseline was 0.05, upper end)
- Early stopping patience = 50

**Monitoring**: Report train/val/test metrics. Flag if train-test DA gap > 5pp.

### Risk 2: Custom Objective Convergence

**Problem**: The directional MAE objective has non-smooth gradients. If penalty_factor is too high (e.g., 5.0), all predictions may collapse toward zero (the model learns that being neutral avoids penalty).

**Mitigations**:
- Lower bound penalty_factor at 1.5 (mild penalty)
- Upper bound at 5.0 (strong but not extreme)
- Monitor prediction variance: if std(predictions) < 0.001 on val set, log warning
- Fallback: If > 20% of Optuna trials produce degenerate predictions (std < 0.001), add `reg:squarederror` trials to the search

### Risk 3: CNY Demand Degradation

**Problem**: CNY features individually degraded DA -2.06% and Sharpe -0.593. In a multi-feature model, these may still be harmful.

**Mitigations**:
- L1 regularization should down-weight noisy features
- Log feature importance for all CNY features after training
- If CNY features rank below 30/39 in importance, consider excluding them in Attempt 2
- Optuna's feature selection toggle is deferred to Attempt 2 (keep scope simple in Attempt 1)

### Risk 4: Test Set Class Imbalance

**Problem**: Test set has 56.7% up days. A biased model could achieve 56% DA trivially without genuine predictive skill.

**Mitigations**:
- Report naive always-up DA alongside model DA
- Check if model prediction distribution is biased (>60% positive predictions)
- Sharpe > 0.8 target ensures the model has real magnitude accuracy, not just directional bias
- HC-DA > 60% requires above-random accuracy on confident subset

### Risk 5: Kaggle Timeout

**Problem**: 50 Optuna trials * variable training time could exceed 9-hour Kaggle limit.

**Mitigations**:
- Optuna timeout = 14400 seconds (4 hours)
- Per-trial early stopping limits maximum boosting rounds
- XGBoost on 1766 rows is inherently fast (1-3 min/trial typical)
- Expected total: 30-90 minutes. Very conservative margin.

### Risk 6: Evaluation Metric Consistency

**Problem**: Training uses custom directional MAE; evaluation uses standard metrics (DA, MAE, Sharpe). If the custom loss optimizes for something different from the evaluation metrics, results may be suboptimal.

**Mitigations**:
- Optuna objective directly optimizes all 4 evaluation metrics (weighted composite)
- The custom objective aligns training with DA (via directional penalty) and MAE (via absolute error)
- Sharpe is optimized via Optuna selection, not training loss

---

## 9. Expected Output Format

### training_result.json

```json
{
    "feature": "meta_model",
    "attempt": 1,
    "timestamp": "2026-02-15T...",
    "model_type": "XGBoost",
    "n_features": 39,
    "best_params": {
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.65,
        "colsample_bytree": 0.7,
        "reg_lambda": 3.5,
        "reg_alpha": 1.2,
        "learning_rate": 0.02,
        "gamma": 0.5,
        "directional_penalty": 2.5,
        "confidence_threshold": 0.005,
        "n_estimators_used": 200
    },
    "metrics": {
        "train": {
            "mae": 0.65,
            "direction_accuracy": 0.55,
            "high_confidence_da": 0.58,
            "sharpe_ratio": 1.5,
            "n_samples": 1766
        },
        "val": {
            "mae": 0.70,
            "direction_accuracy": 0.54,
            "high_confidence_da": 0.57,
            "sharpe_ratio": 0.9,
            "n_samples": 378,
            "hc_coverage": 0.35
        },
        "test": {
            "mae": 0.72,
            "direction_accuracy": 0.56,
            "high_confidence_da": 0.61,
            "sharpe_ratio": 0.85,
            "n_samples": 379,
            "hc_coverage": 0.30
        }
    },
    "overfit_ratios": {
        "mae_val_train": 1.08,
        "mae_test_train": 1.11,
        "da_train_test_gap_pp": 1.0
    },
    "feature_importance_top20": { ... },
    "naive_always_up_da_test": 0.567,
    "optuna_summary": {
        "n_trials": 50,
        "best_trial": 35,
        "best_value": 0.72,
        "trial_details": [ ... ]
    },
    "prediction_distribution": {
        "test_pct_positive": 0.55,
        "test_pred_std": 0.008,
        "test_pred_mean": 0.001
    },
    "training_time_seconds": 3600
}
```

### predictions.csv

| date | split | prediction | actual | direction_correct | high_confidence |
|------|-------|-----------|--------|------------------|-----------------|
| 2015-01-30 | train | 0.005 | 0.003 | True | True |
| ... | ... | ... | ... | ... | ... |
| 2025-02-12 | test | -0.002 | -0.005 | True | False |

### submodel_output.csv

For compatibility with the meta-model evaluation pipeline:

| date | meta_prediction |
|------|----------------|
| 2015-01-30 | 0.005 |
| ... | ... |
| 2025-02-12 | -0.002 |

---

## 10. Success Criteria (Attempt 1)

### Primary (all must pass on test set)

| Metric | Target | How Measured |
|--------|--------|-------------|
| DA | > 56% | sign agreement, excluding zeros |
| HC-DA | > 60% | DA on abs(pred) > confidence_threshold, min 20% coverage |
| MAE | < 0.75% | mean absolute error |
| Sharpe | > 0.8 | annualized, after 5bps daily cost |

### Secondary Diagnostics

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Train-test DA gap | < 5pp | Overfitting control |
| Overfit ratio (MAE) | < 1.5 | Overfitting control |
| Prediction std | > 0.001 | Not degenerate |
| HC coverage | >= 20% | Meaningful confidence threshold |
| Submodel feature importance | > 0 for majority | Submodels contribute |
| Naive DA comparison | Model DA > naive DA | Real predictive skill |

### Decision Rules After Evaluation

| Outcome | Action |
|---------|--------|
| All 4 targets met | Phase 3 COMPLETE. Merge to main. |
| 3 of 4 targets met | Attempt 2: tune failing metric. |
| Severe overfitting (gap > 8pp) | Attempt 2: stronger regularization or drop features. |
| DA < 50% | Attempt 2: switch to `reg:squarederror` + Optuna DA weight. |
| Sharpe < 0 | Attempt 2: increase directional_penalty range or add feature selection flags. |
| Custom objective degeneracy | Attempt 2: switch to standard MSE + post-hoc directional optimization. |

---

**End of Design Document**
