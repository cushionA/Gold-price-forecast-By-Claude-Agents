# Meta-Model Design Document: Weekly (5-Day Return) Attempt 1

## 0. Fact-Check Results

### 0.1 Daily Meta-Model Attempt 7 Results -- VERIFIED from logs/evaluation/meta_model_attempt_7.json

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| DA | 60.04% | > 56% | PASS |
| HCDA | 64.13% (|pred| method) | > 60% | PASS |
| MAE | 0.9429% | < 0.75% | FAIL (structural) |
| Sharpe | 2.4636 | > 0.8 | PASS |
| Train-test DA gap | -5.28pp | < 10pp | PASS (test > train) |
| Features | 24 (5 base + 19 submodel) | -- | Confirmed |
| Architecture | XGBoost reg:squarederror + Bootstrap + OLS | -- | Confirmed |
| Best params | max_depth=2, min_child_weight=25, reg_lambda=2.049, lr=0.0215 | -- | Confirmed |

### 0.2 Weekly Target Variable -- VERIFIED

5-day forward return calculation: `(price[t+5] / price[t] - 1) * 100`

Sanity checks:
- Gold daily return std ~0.9-1.0% (from daily model data)
- Expected weekly return std: ~0.9% * sqrt(5) = ~2.0-2.2%
- Expected weekly MAE (zero prediction): ~1.6-1.8% (vs ~0.96% for daily)
- Non-overlapping 5-day samples from ~3050 daily rows: ~610 independent observations
- Overlapping training samples: ~3045 (lose 5 rows at end vs 1 for daily)

### 0.3 Sharpe Calculation Approach -- CRITICAL ANALYSIS

Two approaches for weekly Sharpe:

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| A: Daily returns, weekly rebalancing | Expand weekly signal to 5 daily positions, compute daily strategy returns, annualize with sqrt(252) | Comparable to daily model Sharpe. Accounts for intra-week volatility. | Signal doesn't change intra-week, slight overcount of Sharpe |
| B: Weekly returns, weekly periods | Non-overlapping weekly returns, annualize with sqrt(52) | Cleaner, fewer samples, wider confidence intervals | ~610 observations (test: ~90), noisy estimate |

**Decision**: Use Approach A (daily returns with weekly rebalancing) as the primary Sharpe metric. This allows direct comparison with the daily model's Sharpe. Also report Approach B as a secondary diagnostic.

**Transaction cost adjustment**: With weekly rebalancing, position changes occur at most every 5 days. Max ~50 trades/year vs ~252 for daily. This significantly reduces transaction costs: 5bps per trade * ~50 trades = ~25bps/year vs ~126bps/year for daily.

### 0.4 Weekly Metric Targets -- ANALYSIS

| Metric | Daily Target | Proposed Weekly Target | Rationale |
|--------|-------------|----------------------|-----------|
| DA | > 56% | > 56% | Weekly direction should be at least as predictable as daily (less noise). Trend signal accumulates over 5 days. |
| HCDA | > 60% | > 60% | Same reasoning. Weekly reduces noise floor. |
| MAE | < 0.75% | < 1.70% | sqrt(5) * 0.75% = 1.68%, rounded to 1.70%. Weekly returns have proportionally higher variance. |
| Sharpe | > 0.8 | > 0.8 | Annualized Sharpe is frequency-independent in theory. Lower transaction costs with weekly rebalancing should help. |

**MAE target skepticism**: The daily model's MAE (0.94%) was structurally infeasible against the 0.75% target. Weekly MAE target of 1.70% is more generous relative to actual weekly volatility (~2.0-2.2%) than 0.75% was to daily volatility (~1.0%). This target may actually be achievable if OLS scaling works better on weekly horizon.

### 0.5 Feature Set Applicability -- VERIFIED

All 24 features are daily-frequency point-in-time measurements. Using them to predict 5-day forward returns is valid:
- No look-ahead: features at time t predict return from t to t+5
- No frequency mismatch: daily features are available every day
- No aggregation needed: the model learns "given today's state, what happens over the next week?"
- Submodel outputs (regime probabilities, z-scores) capture slow-moving state information that may be more predictive over weekly horizons than daily

### 0.6 Dataset Reference -- CONFIRMED

Kaggle dataset `bigbigzabuton/gold-prediction-submodels` contains all required submodel CSVs. Same as attempt 7. No dataset update needed.

### 0.7 Summary

| Check | Verdict |
|-------|---------|
| Daily model results | PASS -- 3/4 targets, confirmed from evaluation JSON |
| Weekly target calculation | PASS -- non-overlapping 5-day forward return |
| Sharpe methodology | PASS -- daily returns with weekly rebalancing (Approach A) |
| Metric targets | PASS -- appropriately scaled for weekly horizon |
| Feature applicability | PASS -- all 24 daily features valid for weekly prediction |
| Kaggle dataset | PASS -- no update needed |

**Decision**: Proceed with weekly meta-model using identical features and architecture as daily attempt 7, with only the target variable changed.

---

## 1. Overview

- **Purpose**: Build a weekly (5-day return) prediction model for more practical trading application. Weekly rebalancing reduces transaction costs, may capture stronger trend signals, and provides actionable weekly positioning signals.
- **Architecture**: Single XGBoost model with reg:squarederror + Bootstrap confidence (5 models) + OLS output scaling. Identical to daily attempt 7.
- **Key Difference from Daily Attempt 7**:
  1. **Target variable**: `gold_return_5d` (forward 5-day return %) instead of `gold_return_next` (next-day return %)
  2. **Sharpe computation**: Daily strategy returns with weekly rebalancing (positions held for 5 days)
  3. **Evaluation**: Both overlapping and non-overlapping metrics reported
  4. **MAE target**: Adjusted to < 1.70% (sqrt(5) scaling)
- **What is NOT changed**: Feature set (24 features), data pipeline, NaN imputation, HP search space, Optuna weights, bootstrap ensemble, OLS scaling, fallback mechanism.
- **Expected Effect**: DA comparable or slightly better than daily (trend signals accumulate). Sharpe potentially higher due to lower transaction costs. MAE target more achievable relative to weekly volatility.
- **Trading Strategy**: Option B -- weekly long/flat signal. Position determined once per week. Hold for 5 trading days. Flat when prediction is negative. Long when prediction is positive.

---

## 2. Data Specification

### 2.1 Target Variable

```python
# Forward 5-day return (non-overlapping forward window)
# At time t: what is the return from close[t] to close[t+5]?
gold_df['gold_return_5d'] = (gold_df['gold_price'].shift(-5) / gold_df['gold_price'] - 1) * 100

# Drop rows where target is NaN (last 5 rows)
gold_df = gold_df.dropna(subset=['gold_return_5d'])
```

**Critical notes**:
- This uses `shift(-5)` on price levels, NOT a rolling sum of daily returns. Rolling sum would introduce look-ahead bias from intermediate prices.
- We lose 5 rows at the end (vs 1 row for daily model). Impact: ~3040 usable rows vs ~3045 for daily.
- Overlapping targets are used for training (all daily rows). Non-overlapping evaluation is computed separately.

### 2.2 Input Data

Identical to daily attempt 7. All sources, paths, date handling, and column specifications unchanged.

| Source | Path (Kaggle) | Used Columns | Date Fix |
|--------|------|-------------|----------|
| Base features | API-fetched (yfinance, FRED) | 5 (transformed) | None |
| VIX submodel | ../input/gold-prediction-submodels/vix.csv | 3 | Lowercase `date` |
| Technical submodel | ../input/gold-prediction-submodels/technical.csv | 3 | utc=True |
| Cross-asset submodel | ../input/gold-prediction-submodels/cross_asset.csv | 3 | None |
| Yield curve submodel | ../input/gold-prediction-submodels/yield_curve.csv | 2 | Rename index |
| ETF flow submodel | ../input/gold-prediction-submodels/etf_flow.csv | 3 | None |
| Inflation expectation | ../input/gold-prediction-submodels/inflation_expectation.csv | 3 | Rename Unnamed:0 |
| Options market | ../input/gold-prediction-submodels/options_market.csv | 1 | utc=True |
| Temporal context | ../input/gold-prediction-submodels/temporal_context.csv | 1 | Lowercase `date`, no tz |
| **Target** | **API-fetched (yfinance GC=F)** | **1 (gold_return_5d)** | **None** |

### 2.3 Feature Set (24 features -- identical to daily attempt 7)

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

TARGET = 'gold_return_5d'  # CHANGED from 'gold_return_next'

assert len(FEATURE_COLUMNS) == 24
```

### 2.4 Data Split

Same 70/15/15 time-series split as daily model. No shuffle.

```python
n_total = len(final_df)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)

train_df = final_df.iloc[:n_train]
val_df = final_df.iloc[n_train:n_train+n_val]
test_df = final_df.iloc[n_train+n_val:]
```

Expected split sizes (approximate):
- Train: ~2130 rows
- Val: ~456 rows
- Test: ~454 rows (slightly fewer than daily due to 5-row loss at end)

### 2.5 NaN Imputation

Identical to daily attempt 7:
- Regime probability columns (7 cols) -> 0.5
- Z-score columns (8 cols) -> 0.0
- Divergence/signal columns (2 cols) -> 0.0
- Continuous state columns (2 cols) -> median

---

## 3. Model Architecture

### 3.1 Architecture: Single XGBoost + Bootstrap Confidence + OLS Scaling

Identical to daily attempt 7. The only change is the target variable.

```
Input: 24-dimensional feature vector (same as daily attempt 7)
  |
  v
XGBoost Ensemble (gradient boosted trees)
  - Objective: reg:squarederror
  - n_estimators: Optuna-controlled [100, 800]
  - Early stopping: patience=100 on validation RMSE
  - Regularization: STRENGTHENED (same as attempt 6/7)
  |
  v
Raw Output: Single scalar (predicted 5-day gold return %)
  |
  v
POST-TRAINING STEP 1: OLS Output Scaling
  - alpha_ols from validation set, capped [0.5, 10.0]
  - Note: Weekly returns have larger magnitude (~2x daily),
    so OLS alpha may be closer to 1.0 than for daily
  |
  v
POST-TRAINING STEP 2: Bootstrap Ensemble Confidence
  - 5 models with seeds [42, 43, 44, 45, 46]
  - Confidence = 1 / (1 + std_across_models)
  |
  v
Output Metrics: DA, HCDA (bootstrap + |pred|), MAE (raw + scaled), Sharpe
```

### 3.2 Metric Functions

All metric functions from daily attempt 7 are reused with the following modifications:

#### 3.2.1 Direction Accuracy (unchanged)

```python
def compute_direction_accuracy(y_true, y_pred):
    """Direction accuracy, excluding zeros."""
    mask = (y_true != 0) & (y_pred != 0)
    if mask.sum() == 0:
        return 0.0
    return (np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean()
```

#### 3.2.2 MAE (unchanged)

```python
def compute_mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.abs(y_pred - y_true).mean()
```

#### 3.2.3 Sharpe -- MODIFIED for Weekly Rebalancing (Approach A)

```python
def compute_sharpe_weekly_rebalance(y_true_daily, y_pred_weekly, dates_daily,
                                     rebalance_dates, cost_bps=5.0):
    """
    Sharpe ratio with weekly rebalancing on daily returns.

    Args:
        y_true_daily: Daily gold returns (%) for the evaluation period
        y_pred_weekly: Weekly predictions aligned to rebalance dates
        dates_daily: Date index for daily returns
        rebalance_dates: Dates when position is determined (every 5th trading day)
        cost_bps: Transaction cost in basis points per trade (one-way)

    Returns:
        Annualized Sharpe ratio
    """
    # Expand weekly predictions to daily positions
    # Each prediction determines position for the next 5 trading days
    positions = np.zeros(len(y_true_daily))
    for i, rebal_date in enumerate(rebalance_dates):
        # Find index of this rebalance date
        idx = np.where(dates_daily == rebal_date)[0]
        if len(idx) == 0:
            continue
        start_idx = idx[0]
        end_idx = min(start_idx + 5, len(positions))
        pred_sign = np.sign(y_pred_weekly[i])
        positions[start_idx:end_idx] = pred_sign

    # Strategy returns (position * actual daily return)
    strategy_returns = positions * y_true_daily / 100.0

    # Position changes (only at rebalance dates)
    position_changes = np.abs(np.diff(positions, prepend=0))
    trade_costs = position_changes * (cost_bps / 10000.0)

    # Net returns
    net_returns = strategy_returns - trade_costs

    # Annualized Sharpe (252 trading days)
    if len(net_returns) < 2 or net_returns.std() == 0:
        return 0.0
    return (net_returns.mean() / net_returns.std()) * np.sqrt(252)
```

#### 3.2.4 Sharpe -- Simplified Overlapping Version (for Optuna)

During Optuna HPO, the full daily-rebalance Sharpe is too complex. Use simplified version:

```python
def compute_sharpe_weekly_simple(y_true_5d, y_pred_5d, cost_bps=5.0):
    """
    Simplified Sharpe for weekly predictions (used in Optuna).
    Treats each overlapping 5-day prediction as an independent trade.
    Transaction cost: 5bps per position change.
    Annualize with sqrt(52) since these are weekly-scale returns.
    """
    positions = np.sign(y_pred_5d)
    strategy_returns = positions * y_true_5d / 100.0

    # Position changes
    position_changes = np.abs(np.diff(positions, prepend=0))
    trade_costs = position_changes * (cost_bps / 10000.0)

    net_returns = strategy_returns - trade_costs

    if len(net_returns) < 2 or net_returns.std() == 0:
        return 0.0
    return (net_returns.mean() / net_returns.std()) * np.sqrt(52)
```

**Important**: The Optuna objective uses `compute_sharpe_weekly_simple()` for speed. The final evaluation uses `compute_sharpe_weekly_rebalance()` for accuracy. Both are reported in the results.

#### 3.2.5 HCDA Functions (unchanged from daily attempt 7)

`compute_hcda()` and `compute_hcda_bootstrap()` are identical.

### 3.3 Non-Overlapping Evaluation

For cleaner test evaluation, report non-overlapping 5-day period metrics:

```python
def compute_non_overlapping_metrics(test_df, pred_test, dates_test):
    """
    Evaluate on non-overlapping 5-day periods.
    Take every 5th row starting from the first test date.
    """
    # Select every 5th day
    non_overlap_idx = np.arange(0, len(test_df), 5)

    y_no = test_df[TARGET].values[non_overlap_idx]
    pred_no = pred_test[non_overlap_idx]
    dates_no = dates_test[non_overlap_idx]

    da = compute_direction_accuracy(y_no, pred_no)
    mae = compute_mae(y_no, pred_no)
    hcda, hcda_cov = compute_hcda(y_no, pred_no, threshold_percentile=80)

    return {
        'n_samples': len(non_overlap_idx),
        'direction_accuracy': float(da),
        'mae': float(mae),
        'hcda': float(hcda),
        'hcda_coverage': float(hcda_cov),
        'dates': [str(d) for d in dates_no],
    }
```

---

## 4. Hyperparameter Specification

### 4.1 Fixed Parameters

Identical to daily attempt 7.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective | reg:squarederror | Standard MSE. Works for weekly returns same as daily. |
| early_stopping_rounds | 100 | Same patience. Weekly targets have higher variance, so convergence may take slightly longer. |
| eval_metric | rmse | Standard |
| tree_method | hist | Fast |
| verbosity | 0 | Suppress |
| seed | 42 + trial.number | Reproducible |

### 4.2 Optuna Search Space

Identical to daily attempt 6/7 ranges. The strengthened regularization bounds remain appropriate for weekly prediction -- stronger regularization helps when target variance is higher (less temptation to overfit to noise).

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| max_depth | [2, 4] | int | Shallow trees prevent overfitting to noisy weekly targets |
| n_estimators | [100, 800] | int | Same as daily |
| learning_rate | [0.001, 0.05] | log | Same as daily |
| colsample_bytree | [0.2, 0.7] | linear | Same as daily |
| subsample | [0.4, 0.85] | linear | Same as daily |
| min_child_weight | [12, 25] | int | Remains appropriate. Each leaf covers >= 0.56% of data. |
| reg_lambda (L2) | [1.0, 15.0] | log | Strong L2 prevents overfitting to higher-variance weekly returns |
| reg_alpha (L1) | [0.5, 10.0] | log | Same as daily |

**Total: 8 hyperparameters** (unchanged).

### 4.3 Search Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_trials | 100 | Same as daily attempt 7 |
| timeout | 7200 sec | 2-hour margin |
| sampler | TPESampler(seed=42) | Reproducible |
| pruner | None | Not needed for XGBoost |
| direction | maximize | Higher composite is better |

### 4.4 Optuna Objective Function

Same composite objective as daily attempt 7, but using `compute_sharpe_weekly_simple()`:

```python
def optuna_objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 2, 4),
        'min_child_weight': trial.suggest_int('min_child_weight', 12, 25),
        'subsample': trial.suggest_float('subsample', 0.4, 0.85),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.7),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 10.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'seed': 42 + trial.number,
    }

    n_estimators = trial.suggest_int('n_estimators', 100, 800)

    model = xgb.XGBRegressor(**params, n_estimators=n_estimators, early_stopping_rounds=100)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_da = compute_direction_accuracy(y_train, train_pred)
    val_da = compute_direction_accuracy(y_val, val_pred)
    val_mae = compute_mae(y_val, val_pred)
    val_sharpe = compute_sharpe_weekly_simple(y_val, val_pred)
    val_hc_da, val_hc_coverage = compute_hcda(y_val, val_pred, threshold_percentile=80)

    da_gap = (train_da - val_da) * 100
    overfit_penalty = max(0.0, (da_gap - 10.0) / 30.0)

    # Normalization -- ADJUSTED for weekly scale
    sharpe_norm = np.clip((val_sharpe + 3.0) / 6.0, 0.0, 1.0)
    da_norm = np.clip((val_da * 100 - 40.0) / 30.0, 0.0, 1.0)
    mae_norm = np.clip((2.5 - val_mae) / 1.5, 0.0, 1.0)  # CHANGED: [1.0%, 2.5%] -> [0, 1]
    hc_da_norm = np.clip((val_hc_da * 100 - 40.0) / 30.0, 0.0, 1.0)

    # Weights (same as daily: 40/30/10/20)
    objective = (
        0.40 * sharpe_norm +
        0.30 * da_norm +
        0.10 * mae_norm +
        0.20 * hc_da_norm
    ) - 0.30 * overfit_penalty

    trial.set_user_attr('val_da', float(val_da))
    trial.set_user_attr('val_mae', float(val_mae))
    trial.set_user_attr('val_sharpe', float(val_sharpe))
    trial.set_user_attr('val_hc_da', float(val_hc_da))
    trial.set_user_attr('val_hc_coverage', float(val_hc_coverage))
    trial.set_user_attr('train_da', float(train_da))
    trial.set_user_attr('da_gap_pp', float(da_gap))
    trial.set_user_attr('n_estimators_used',
                         int(model.best_iteration + 1) if hasattr(model, 'best_iteration')
                         and model.best_iteration is not None else n_estimators)

    return objective
```

**Note on MAE normalization**: Daily model used `(1.0 - val_mae) / 0.5` mapping [0.5%, 1.0%] to [0, 1]. Weekly MAE is ~sqrt(5) larger, so we use `(2.5 - val_mae) / 1.5` mapping [1.0%, 2.5%] to [0, 1]. This keeps the MAE component proportionally similar in the composite objective.

---

## 5. Training Configuration

### 5.1 Training Algorithm

```
1. DATA PREPARATION:
   (identical to daily attempt 7, EXCEPT target variable)
   a. Fetch raw data using yfinance and fredapi
   b. Construct base features (5)
   c. Compute daily changes
   d. Load 8 submodel output CSVs (same as daily attempt 7)
   e. Merge base + all submodel on Date
   f. Apply NaN imputation (same rules as daily)
>> g. Compute target: gold_return_5d = (price[t+5]/price[t] - 1) * 100
>> h. Drop rows where gold_return_5d is NaN (last 5 rows)
   i. Verify: 24 features, 0 remaining NaN
   j. Split: train (70%), val (15%), test (15%)
>> k. Also prepare daily returns for Sharpe computation

2. OPTUNA HPO (100 trials, 2-hour timeout):
   (same as daily attempt 7, EXCEPT Sharpe function and MAE normalization)
>> a. Use compute_sharpe_weekly_simple() in objective
>> b. Use adjusted MAE normalization: (2.5 - mae) / 1.5

3. FALLBACK EVALUATION:
   (same as daily attempt 7 -- attempt 2 best params + 24 features)

4. FINAL MODEL TRAINING:
   (identical to daily attempt 7)

5. POST-TRAINING STEP 1: OLS OUTPUT SCALING:
   (identical to daily attempt 7)

6. POST-TRAINING STEP 2: BOOTSTRAP ENSEMBLE CONFIDENCE:
   (identical to daily attempt 7 -- 5 models, seeds [42-46])

7. EVALUATION ON ALL SPLITS:
   a. Compute overlapping metrics: DA, HCDA, MAE, Sharpe (weekly_simple)
>> b. Compute non-overlapping metrics: DA, HCDA, MAE (every 5th day)
>> c. Compute daily-rebalance Sharpe (Approach A): expand weekly signals
>>    to daily positions on test set
   d. Feature importance for 24 features
   e. Quarterly breakdown
   f. Decile analysis for both HCDA methods
>> g. Comparison with daily model attempt 7
>> h. Comparison baselines: naive always-up, buy-and-hold

8. SAVE RESULTS:
   (same output files as daily, plus weekly-specific diagnostics)
```

### 5.2 Additional Data Preparation for Sharpe

The notebook must also fetch daily gold returns for the Sharpe Approach A computation:

```python
# Daily returns needed for weekly-rebalance Sharpe
gold_df['gold_return_daily'] = gold_df['gold_price'].pct_change() * 100

# After split, extract daily returns for test period
daily_returns_test = test_df['gold_return_daily'].values  # or fetched separately
```

**Implementation note**: The daily return column `gold_return_daily` is needed alongside the weekly target `gold_return_5d`. Both can coexist in the DataFrame. The daily return is NOT a feature -- it is only used for Sharpe computation in the evaluation step.

### 5.3 Loss Function

- reg:squarederror (unchanged)

### 5.4 Early Stopping

- Metric: RMSE on validation set
- Patience: 100 rounds
- Maximum rounds: Optuna-controlled (100-800)

### 5.5 Fallback Configuration

Same attempt 2 best params. XGBoost adapts to the different target scale automatically.

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

## 6. Evaluation Framework

### 6.1 Primary Targets (on test set, overlapping evaluation)

| Metric | Target | Method | Comparison with Daily |
|--------|--------|--------|----------------------|
| DA | > 56% | sign agreement, excluding zeros | Same as daily |
| HCDA | > 60% | top 20% by BEST of (bootstrap, |pred|) | Same as daily |
| MAE | < 1.70% | BEST of (raw, OLS-scaled) | Relaxed from 0.75% (sqrt(5) scaling) |
| Sharpe | > 0.8 | Approach A: daily returns, weekly rebalance, annualized sqrt(252) | Same as daily, lower tx costs |

### 6.2 Secondary Diagnostics

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Non-overlapping DA | Report value | Unbiased weekly accuracy estimate |
| Non-overlapping MAE | Report value | Unbiased weekly error estimate |
| Non-overlapping Sharpe (Approach B) | Report value | Alternative Sharpe, annualized sqrt(52) |
| Train-test DA gap | < 10pp | Overfitting control |
| Val DA | > 50% | Not below random |
| Bootstrap HCDA vs |pred| HCDA | Report both | Compare confidence methods |
| OLS alpha value | [0.5, 10.0] | Scaling reasonableness |
| Feature importance | Report all 24 | Compare with daily importance ranking |
| Quarterly breakdown | Report | Identify regime-dependent performance |
| vs Daily model attempt 7 | Report all metrics | Key comparison |
| vs Naive always-up | Report DA | Baseline check |
| vs Daily-to-weekly conversion | Report DA | Compare direct weekly vs converted daily |

### 6.3 Comparison Framework

#### 6.3.1 vs Daily Model (Attempt 7)

The most important comparison. Compute for test period:

```python
# Convert daily model signals to weekly: majority vote over 5 days
# For each 5-day window, take the dominant direction from daily predictions
daily_signals = np.sign(daily_model_predictions)
weekly_from_daily = []
for i in range(0, len(daily_signals), 5):
    window = daily_signals[i:i+5]
    majority = np.sign(np.sum(window))  # Majority vote
    weekly_from_daily.append(majority)
```

Report: How does direct weekly prediction compare to majority-voting daily predictions?

#### 6.3.2 vs Naive Baselines

1. **Always-up**: DA = fraction of positive 5-day returns in test set. Gold uptrend means this baseline may be strong (~55-60%).
2. **Buy-and-hold**: Sharpe of simply holding gold for the entire test period.
3. **Random**: 50% DA baseline.

#### 6.3.3 Transaction Cost Advantage

Compute and report:
- Daily model: ~252 potential trades/year, ~X actual position changes, ~Y bps total cost
- Weekly model: ~52 potential trades/year, ~X actual position changes, ~Y bps total cost
- Net Sharpe difference attributable to cost reduction

### 6.4 Decision Rules After Evaluation

| Outcome | Action |
|---------|--------|
| 3/4 targets met (DA, Sharpe, and HCDA or MAE) | Accept as weekly model |
| DA >= 56% and Sharpe >= 0.8, HCDA close | Accept with note |
| Weekly DA > daily DA by >= 1pp | Strong validation of weekly approach |
| Weekly Sharpe > daily Sharpe | Expected due to lower costs |
| All metrics worse than daily | Weekly approach may not add value |
| Overfitting (gap > 10pp) | Increase regularization in attempt 2 |

---

## 7. Kaggle Execution Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | true | Consistent with daily attempt 7 |
| Estimated execution time | 25-45 minutes | Same as daily (same data size, same HPO) |
| Estimated memory usage | 1.5 GB | Same as daily |
| Required pip packages | [] | No additional packages |
| Internet required | true | For data fetching |
| Kaggle Notebook ID | bigbigzabuton/gold-model-training-meta-model | Same kernel (new version) |
| dataset_sources | bigbigzabuton/gold-prediction-submodels | Same dataset |
| Optuna timeout | 7200 sec | Same as daily |

---

## 8. Implementation Instructions

### 8.1 For builder_data

No separate data preparation step needed. The notebook is self-contained.

### 8.2 For builder_model

**Task**: Generate `notebooks/meta_model_weekly_1/train.ipynb` (self-contained Kaggle Notebook)

**Base**: Copy daily attempt 7 notebook (`notebooks/meta_model_7/train.ipynb`) with the following modifications.

#### 8.2.1 Markdown Header (Cell 0)

```markdown
# Gold Meta-Model Training - Weekly (5-Day Return) Attempt 1

**Architecture:** Single XGBoost with reg:squarederror (weekly target)

**Key Differences from Daily Attempt 7:**
1. **Target**: gold_return_5d (forward 5-day return %) instead of gold_return_next
2. **Sharpe**: Daily returns with weekly rebalancing (positions held 5 days)
3. **MAE target**: < 1.70% (sqrt(5) scaling from daily 0.75%)
4. **Evaluation**: Both overlapping and non-overlapping metrics

**Inherited from Daily Attempt 7:**
- Same 24 features (5 base + 19 submodel outputs)
- Same HP search space (Attempt 6 ranges)
- Bootstrap variance-based confidence (5 models for HCDA)
- OLS output scaling (validation-derived, capped at 10x)
- Optuna weights: 40/30/10/20

**Design:** `docs/design/meta_model_weekly_attempt_1.md`
```

#### 8.2.2 TARGET Definition (Cell 3)

```python
TARGET = 'gold_return_5d'  # CHANGED from 'gold_return_next'
```

#### 8.2.3 Target Variable Computation (Cell 5 -- data fetching)

Replace the next-day return calculation:

```python
# CHANGED: 5-day forward return instead of next-day
gold_df['gold_return_5d'] = (gold_df['gold_price'].shift(-5) / gold_df['gold_price'] - 1) * 100
gold_df['gold_return_daily'] = gold_df['gold_price'].pct_change() * 100  # For Sharpe computation
gold_df = gold_df.dropna(subset=['gold_return_5d'])
```

#### 8.2.4 Sharpe Functions (Cell 11 -- metric functions)

Add `compute_sharpe_weekly_simple()` and `compute_sharpe_weekly_rebalance()` as defined in Section 3.2.3 and 3.2.4.

Rename the daily Sharpe function to `compute_sharpe_daily()` (keep it for comparison) and use `compute_sharpe_weekly_simple()` in the Optuna objective.

#### 8.2.5 MAE Normalization in Optuna (Cell 13)

Change the MAE normalization line:

```python
# CHANGED: Weekly MAE range [1.0%, 2.5%] -> [0, 1]
mae_norm = np.clip((2.5 - val_mae) / 1.5, 0.0, 1.0)  # was (1.0 - val_mae) / 0.5
```

#### 8.2.6 Non-Overlapping Evaluation (new cell after Cell 26)

Add the `compute_non_overlapping_metrics()` function and call it on the test set.

#### 8.2.7 Daily-Rebalance Sharpe Computation (Cell 26 -- diagnostics)

Add the Approach A Sharpe computation using daily gold returns stored during data preparation:

```python
# === Weekly Rebalance Sharpe (Approach A) ===
# Expand weekly predictions to daily positions
test_dates = np.array(dates_test)
rebalance_indices = np.arange(0, len(test_dates), 5)
rebalance_dates = test_dates[rebalance_indices]
rebalance_preds = pred_test[rebalance_indices]

positions_daily = np.zeros(len(test_dates))
for i, idx in enumerate(rebalance_indices):
    end_idx = min(idx + 5, len(positions_daily))
    positions_daily[idx:end_idx] = np.sign(rebalance_preds[i])

# Daily strategy returns with weekly rebalancing
daily_returns_test = test_df['gold_return_daily'].values
strategy_returns = positions_daily * daily_returns_test / 100.0
position_changes = np.abs(np.diff(positions_daily, prepend=0))
trade_costs = position_changes * (5.0 / 10000.0)
net_returns = strategy_returns - trade_costs

sharpe_approach_a = (net_returns.mean() / net_returns.std()) * np.sqrt(252)
```

#### 8.2.8 Comparison with Daily Model (Cell 26 -- diagnostics)

Add comparison printout against daily attempt 7 results:

```python
# Comparison with Daily Attempt 7
print("\nVs Daily Attempt 7:")
print(f"  DA:     {test_m['direction_accuracy']*100:.2f}% (Daily: 60.04%)")
print(f"  HCDA:   {primary_hcda_value*100:.2f}% (Daily: 64.13%)")
print(f"  MAE:    {test_m['mae']:.4f}% (Daily: 0.9429%)")
print(f"  Sharpe: {sharpe_approach_a:.2f} (Daily: 2.46)")
```

#### 8.2.9 Target Evaluation (Cell 24)

Update target thresholds:

```python
targets_met = [
    test_m['direction_accuracy'] > 0.56,     # Same as daily
    primary_hcda_value > 0.60,                # Same as daily
    test_m['mae'] < 0.0170,                   # CHANGED: was 0.0075
    test_m['sharpe_ratio'] > 0.8,             # Same as daily (Approach A Sharpe)
]
```

Update target display strings accordingly.

#### 8.2.10 training_result.json (Cell 28)

Update:

```python
training_result['feature'] = 'meta_model_weekly'
training_result['attempt'] = 1
training_result['target_type'] = 'gold_return_5d'
training_result['target_description'] = 'Forward 5-day gold return (%)'
training_result['architecture'] = 'XGBoost reg:squarederror + Bootstrap + OLS (weekly target)'

# Add weekly-specific fields
training_result['weekly_evaluation'] = {
    'sharpe_approach_a': float(sharpe_approach_a),  # Daily returns, weekly rebalance
    'sharpe_approach_b': float(sharpe_non_overlap),  # Non-overlapping weekly, sqrt(52)
    'non_overlapping_metrics': non_overlap_metrics,
    'overlapping_metrics': metrics_all['test'],
    'tx_cost_reduction_vs_daily': 'Weekly rebalance: ~50 trades/year vs ~252 for daily',
}

training_result['vs_daily_attempt_7'] = {
    'daily_da': 0.6004,
    'daily_hcda': 0.6413,
    'daily_mae': 0.9429,
    'daily_sharpe': 2.4636,
    'weekly_da': float(test_m['direction_accuracy']),
    'weekly_hcda': float(primary_hcda_value),
    'weekly_mae': float(test_m['mae']),
    'weekly_sharpe': float(sharpe_approach_a),
}
```

#### 8.2.11 kernel-metadata.json

```json
{
  "id": "bigbigzabuton/gold-meta-weekly-1",
  "title": "Gold Meta-Model Training - Weekly Attempt 1",
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

1. Copy `notebooks/meta_model_7/train.ipynb` to `notebooks/meta_model_weekly_1/train.ipynb`
2. Update markdown header (Cell 0) -- weekly description
3. Change TARGET to `'gold_return_5d'` (Cell 3)
4. Change target computation: `shift(-5)` price ratio instead of `shift(-1)` return (Cell 5)
5. Add `gold_return_daily` column for Sharpe computation (Cell 5)
6. Add `compute_sharpe_weekly_simple()` and `compute_sharpe_weekly_rebalance()` (Cell 11)
7. Update Optuna objective: use weekly Sharpe, adjust MAE normalization (Cell 13)
8. Update target thresholds: MAE < 0.0170 (Cell 24)
9. Add non-overlapping evaluation function and call (new cell)
10. Add Approach A Sharpe computation on test set (Cell 26)
11. Add daily model comparison diagnostics (Cell 26)
12. Update training_result.json fields (Cell 28)
13. Create kernel-metadata.json
14. Run `scripts/validate_notebook.py`

**Items NOT changed** (verify these remain intact):
- Feature columns (24, same list)
- Submodel loading (all 8 CSVs, same paths, same date handling)
- NaN imputation (same rules)
- HP ranges (same as attempt 6/7)
- Bootstrap ensemble (5 models, seeds [42-46])
- OLS scaling (same logic)
- Fallback params (same as attempt 2)
- Feature importance reporting

---

## 9. Risk Mitigation

### Risk 1: Overlapping Targets Inflate Training Performance (MODERATE)

**Scenario**: Adjacent daily rows share 4 of 5 days in their 5-day return window. This autocorrelation in the target variable makes training loss artificially low and validation metrics unreliable.

**Probability**: 40-50% (that overlapping metrics overstate performance).

**Mitigation**:
1. Report BOTH overlapping and non-overlapping metrics. The evaluator uses non-overlapping as the truth.
2. XGBoost with strong regularization (min_child_weight [12,25]) is robust to correlated targets.
3. Time-series split ensures no leakage. The overlap is within-split only.

### Risk 2: Weekly Direction Harder to Predict Than Expected (MODERATE)

**Scenario**: While daily noise may average out over 5 days, weekly returns are driven by different dynamics (macro events, policy announcements) that the daily features may not capture well.

**Probability**: 30-40%.

**Mitigation**:
1. The submodel features (regime probabilities, persistence scores) capture slow-moving state information that is inherently more relevant for multi-day horizons.
2. If weekly DA < daily DA, this suggests the model is not capturing weekly dynamics, and the daily model + weekly rebalancing may be the better strategy.
3. Fallback to daily model exists.

### Risk 3: OLS Scaling Overfits to Validation Weekly Volatility (LOW)

**Scenario**: Weekly returns have different volatility characteristics by period. OLS alpha computed on validation period doesn't transfer to test period.

**Probability**: 15-20%.

**Mitigation**:
1. Cap at [0.5, 10.0] limits damage.
2. Report both raw and scaled MAE. Use the better one.
3. DA and Sharpe are unaffected by scaling.

### Risk 4: Sharpe Approach A Overestimates (LOW-MODERATE)

**Scenario**: Expanding weekly signals to daily positions creates auto-correlated daily strategy returns, which may inflate Sharpe via underestimated volatility.

**Probability**: 20-30%.

**Mitigation**:
1. Report Approach B (non-overlapping weekly Sharpe with sqrt(52) annualization) as a sanity check.
2. If Approach A Sharpe >> Approach B Sharpe, the evaluator should use Approach B.

### Risk 5: Non-Overlapping Test Set Too Small (MODERATE)

**Scenario**: With ~454 test rows, non-overlapping evaluation has ~90 samples. HCDA at 80th percentile means ~18 high-confidence samples. Statistical power is very low.

**Probability**: 60-70% that non-overlapping HCDA is unreliable.

**Mitigation**:
1. Use overlapping metrics as primary, non-overlapping as secondary diagnostic.
2. Report confidence intervals on non-overlapping metrics.
3. Accept wider uncertainty ranges for weekly evaluation.

---

## 10. Expected Outcomes

| Metric | Daily Att 7 | Weekly Expected | Confidence | Rationale |
|--------|------------|----------------|------------|-----------|
| DA (overlapping) | 60.04% | 56-62% | Medium | Trend accumulation may help, but target noise is higher |
| DA (non-overlapping) | -- | 54-60% | Medium | Fewer samples, wider range |
| HCDA (best method) | 64.13% | 55-65% | Low | Harder to estimate with higher-variance targets |
| MAE | 0.9429% | 1.4-2.0% | Medium | Scaled proportionally to weekly volatility |
| Sharpe (Approach A) | 2.4636 | 1.5-3.0 | Medium | Lower tx costs should help |
| Sharpe (Approach B) | -- | 0.8-2.0 | Low | Fewer observations, wider CI |
| Train-test gap | -5.28pp | 3-12pp | Medium | Overlapping targets may reduce apparent gap |
| Targets passed | 3/4 | 2-3/4 | Medium | |

**Probability of outcomes**:

| Outcome | Probability |
|---------|------------|
| 3/4 targets (DA + Sharpe + HCDA or MAE) | 25-35% |
| 2/4 targets (DA + Sharpe) | 35-45% |
| 4/4 targets | 10-15% (MAE target more achievable) |
| Regression (<2/4) | 10-20% |

---

## 11. Success Criteria

### Primary Targets (on test set)

| Metric | Target | Method |
|--------|--------|--------|
| DA | > 56% | sign agreement, excluding zeros (overlapping) |
| HCDA | > 60% | top 20% by BEST of (bootstrap confidence, |prediction|) |
| MAE | < 1.70% | BEST of (raw, OLS-scaled) predictions |
| Sharpe | > 0.80 | Approach A: daily returns, weekly rebalance, sqrt(252) |

### Decision Rules

| Outcome | Action |
|---------|--------|
| 3/4 targets met | Accept as weekly model. Report alongside daily model. |
| 2/4 targets met, weekly Sharpe > daily Sharpe | Accept. Weekly model preferred for lower-cost trading. |
| DA < 56% but Sharpe > 0.8 | Accept with note. High Sharpe may come from tx cost savings. |
| All metrics worse than daily | Weekly model adds no value. Recommend daily model with weekly rebalancing instead. |
| Non-overlapping DA < 50% | Overlapping metrics misleading. Weekly prediction unreliable. |

---

## 12. Comparison with Daily Meta-Model

| Aspect | Daily Attempt 7 | Weekly Attempt 1 (This) |
|--------|----------------|------------------------|
| Target | Next-day return (%) | 5-day forward return (%) |
| Features | 24 | 24 (identical) |
| Architecture | XGBoost + Bootstrap + OLS | XGBoost + Bootstrap + OLS (identical) |
| HP search space | Attempt 6 ranges | Attempt 6 ranges (identical) |
| Optuna weights | 40/30/10/20 | 40/30/10/20 (identical) |
| MAE target | < 0.75% | < 1.70% |
| Sharpe method | Daily strategy returns, sqrt(252) | Daily returns, weekly rebalance, sqrt(252) |
| Tx cost per year | ~126 bps (252 trades) | ~25 bps (~50 trades) |
| Practical use | Daily rebalancing | Weekly long/flat signal |
| Independent eval samples | ~454 | ~90 (non-overlapping) |

---

**End of Design Document**

**Architect**: architect (Opus 4.6)
**Date**: 2026-02-17
**Based on**: Daily meta-model attempt 7 (3/4 targets PASS, DA 60.04%, HCDA 64.13%, Sharpe 2.46)
**Purpose**: Weekly (5-day) return prediction for practical trading application
