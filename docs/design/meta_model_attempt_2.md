# Meta-Model Design Document: Attempt 2

## 0. Fact-Check Results

### 0.1 Researcher Stationarity Claims (Q3) -- CRITICAL CORRECTIONS

The researcher claimed real_rate, yield_spread, and inflation_expectation are "stationary in levels." ADF tests on the actual data/processed/base_features.csv refute this:

| Feature | Researcher Claim | ADF p-value | Verdict | Correction |
|---------|-----------------|-------------|---------|------------|
| real_rate (DFII10) | "YES (bounded)" | 0.844 | **FAIL** -- non-stationary | Use real_rate_change (ADF p<0.001) |
| dxy (DX-Y.NYB) | "NO (convert to changes)" | 0.473 | Correct | Use dxy_change (ADF p<0.001) |
| vix (VIXCLS) | "YES (bounded)" | 0.000007 | **Correct** -- stationary | Use vix level |
| yield_spread (DGS10-DGS2) | "YES (bounded)" | 0.521 | **FAIL** -- non-stationary | Use yield_spread_change (ADF p<0.001) |
| inflation_expectation (T10YIE) | "YES (bounded)" | 0.279 | **FAIL** -- non-stationary | Use inflation_exp_change (ADF p<0.001) |

**Distribution shift between train (70%) and test (last 15%) confirms the danger:**

| Feature | Train Mean | Test Mean | Shift (std units) |
|---------|-----------|-----------|-------------------|
| real_rate | 0.114% | 1.997% | 2.95 std |
| dxy | 95.6 | 104.6 | 3.05 std |
| vix | 17.9 | 15.6 | 0.28 std |
| yield_spread | 0.80% | -0.20% | 2.20 std |
| inflation_exp | 1.84% | 2.29% | 1.34 std |

Only VIX has negligible shift (0.28 std). All others have severe distribution drift that would cause XGBoost tree splits learned in training to be invalid in test -- the exact same problem that caused Attempt 1 failure with price-level features.

**Design decision**: Use daily changes for real_rate, dxy, yield_spread, and inflation_expectation. Use level for VIX only. This aligns with the correlation evidence: changes have 2-10x stronger correlation with gold_return_next than levels (real_rate_change: -0.138 vs level: 0.029; inflation_exp_change: +0.092 vs level: -0.015).

### 0.2 Feature Count Error (Q3)

The researcher claims "19 submodel features" and "24 total features" (5 base + 19 submodel). However, their own itemized list sums to 17 submodel features:

- vix: 3
- technical: 3
- cross_asset: 3
- yield_curve: 2 (yc_regime_prob excluded per evaluator, constant std=1.07e-11)
- etf_flow: 3
- inflation_expectation: 3
- **Total: 17** (not 19)

current_task.json also lists exactly 17 submodel columns. The correct total is **22 features** (5 base + 17 submodel), not 24.

### 0.3 Data Loss Claims (Q5) -- SIGNIFICANTLY OVERSTATED

The researcher claims "45.4% data loss" and proposes imputation to "recover ~1000 rows." Empirical testing reveals the actual situation is far less severe:

**Actual data pipeline test** (with proper date normalization, including utc=True for technical.csv):

| Merge Step | Matched Rows | NaN Rows | Cause |
|-----------|-------------|----------|-------|
| base + target | 2523 | 0 | -- |
| + vix | 2523 | 0 | VIX warm-up period (39 rows) predates base_features start |
| + technical | 2522 | 1 | 1 unmatched date at idx 2499 |
| + cross_asset | 2522 | 1 | 1 unmatched date at idx 2499 |
| + yield_curve | 2507 | 16 | 16 early-period NaN in spread_velocity_z |
| + etf_flow | 2522 | 1 | 1 unmatched date at idx 2499 |
| + inflation_exp | 2523 | 0 | Predates base_features start |

**Total rows with any NaN: 102** (4.0% of 2523), not 1145 (45.4%).

The 45.4% data loss in Attempt 1 was caused by:
1. **technical.csv timezone mismatch** -- dates stored as "2014-10-01 00:00:00-04:00" while base_features uses "2015-01-30". Inner join on raw strings matched 0 rows.
2. This single issue cascaded: with 0 technical matches, inner join + dropna eliminated all rows.

**Root cause was a DATE PARSING BUG, not warm-up NaN.** With proper date normalization (pd.to_datetime with utc=True for timezone-aware dates, then strftime '%Y-%m-%d'), the maximum NaN count is 102 rows (yc_curvature_z), concentrated in the first ~15 dates.

**Imputation is still recommended** (recover 102 rows, keep all 2523 for training), but the researcher's framing of "recovering 1000 rows" and "45% data loss" was based on the Attempt 1 bug, not an inherent data problem.

### 0.4 VIX NaN Claim (Q5) -- INCORRECT

Researcher claims "VIX: 39 rows NaN in mean_reversion_z (40-day rolling window), 29 rows in persistence." This is true for the raw vix.csv file. However, after merging with base_features (which starts 2015-01-30 while VIX data starts 2014-10-02), all VIX warm-up NaN are already resolved. **VIX has 0 NaN after merge.** The researcher's claim is misleading because it reports raw file NaN counts, not post-merge counts.

### 0.5 Regularization Recommendations (Q1) -- ADEQUATE

The researcher's recommended ranges are appropriate given the Attempt 1 catastrophe:

| Parameter | Attempt 1 Best | Recommended Range | Assessment |
|-----------|---------------|-------------------|------------|
| max_depth | 3 | [2, 4] | Adequate. Depth 3 was already used but overfitting came from the loss function, not depth. |
| min_child_weight | 5 | [10, 30] | Good. Forces larger leaves. With 1766 train samples, min=30 means leaves cover >1.7% of data. |
| subsample | 0.695 | [0.4, 0.7] | Good. Lower bound more aggressive. |
| colsample_bytree | 0.535 | [0.3, 0.6] | Good. With 22 features, 0.3 samples ~7 features per tree. |
| reg_lambda | 1.56 | [3.0, 20.0] | Good. 2x-13x stronger than Attempt 1. |
| reg_alpha | 0.249 | [1.0, 10.0] | Good. 4x-40x stronger than Attempt 1. |
| learning_rate | 0.0108 | [0.005, 0.03] | Good. Similar range with lower upper bound. |
| gamma | 0.360 | [0.5, 3.0] | Good. Higher minimum prevents trivial splits. |

**However**: The PRIMARY cause of Attempt 1's 47pp gap was the directional-weighted MAE (penalty=4.52), not regularization weakness. With reg:squarederror and 22 features (not 39), the regularization needs are less extreme. The baseline with 19 features and max_depth=5 only had 11pp gap. With fewer features and stronger regularization, the gap should reduce to <5pp.

### 0.6 Sharpe Formula (Q4) -- CORRECT

The researcher correctly identifies the discrepancy:
- src/evaluation.py line 215: `net_returns = returns - cost_pct` (daily cost, every day)
- CLAUDE.md spec: `ret = positions * actual - trades * cost` (cost on position changes only)

Empirical verification on test set:
- Always-long Sharpe with daily cost: 1.183
- Always-long Sharpe with trade cost: 2.065

The CLAUDE.md formula is the authoritative spec. The training notebook and evaluator must both use trade-cost formula. The researcher's recommendation to align to CLAUDE.md is correct.

### 0.7 Optuna Objective Structure (Q2) -- ADEQUATE WITH ADJUSTMENT

The composite objective weights (50% Sharpe, 30% DA, 10% MAE, 10% HC-DA) are reasonable. The normalization ranges need minor correction:

- Researcher proposes DA normalization range [40%, 70%]. Actual baseline DA is 43.5% and target is 56%, so [40%, 70%] is reasonable.
- Sharpe normalization [-3, +3] maps to [0, 1]. Always-long Sharpe with trade cost is 2.065. A model should score between 0 and 2, so the range is adequate.
- The claim that "DA does NOT emerge naturally from reg:squarederror" is correct for small-magnitude predictions but overstated for well-calibrated models. Still, explicit DA measurement in Optuna is the safe approach.

### 0.8 Summary

| Check | Verdict |
|-------|---------|
| Stationarity of real_rate, yield_spread, inflation_exp | FAIL -- all non-stationary, must use changes |
| Feature count (24) | FAIL -- correct count is 22 |
| Data loss magnitude (45.4%) | FAIL -- actual NaN rate is 4.0%, not 45.4% |
| VIX NaN count (39 rows) | FAIL -- 0 NaN after merge with base_features |
| DXY stationarity (NO, convert) | PASS |
| VIX stationarity (YES, level) | PASS |
| Regularization ranges | PASS |
| Sharpe formula alignment | PASS |
| Optuna objective structure | PASS |
| reg:squarederror recommendation | PASS |
| yc_regime_prob exclusion | PASS (verified: std=1.07e-11, effectively constant) |

**Decision**: Corrections are incorporated into the design below. No need to return to researcher -- the errors are factual data issues that the architect can correct directly from empirical evidence.

---

## 1. Overview

- **Purpose**: Integrate 5 stationary/change-transformed base features and 17 submodel outputs (22 total) from 6 submodels to predict next-day gold return (%). This is the Phase 3 meta-model attempt 2, correcting Attempt 1's catastrophic overfitting (train DA 94.3%, test DA 54.1%, 40pp gap).
- **Architecture**: XGBoost with standard reg:squarederror objective. No custom loss function.
- **Key Changes from Attempt 1**:
  1. Drop directional-weighted MAE (root cause of 47pp DA gap) -- use reg:squarederror
  2. Drop 11 non-stationary price-level features + 4 CNY features (39 -> 22 features)
  3. Convert 4 base features to daily changes (only VIX kept as level)
  4. Fix date parsing bug (technical.csv timezone normalization)
  5. Apply domain-specific NaN imputation (recover 102 rows, keep all 2523)
  6. Align Sharpe formula to CLAUDE.md (cost on position changes only)
  7. Strengthen regularization (lambda up to 20, alpha up to 10, min_child_weight up to 30)
- **Expected Effect**: Train-test DA gap < 10pp. Test DA > 56% (comparable to naive always-up 56.9%). Test Sharpe > 0.8 with trade-cost formula. MAE < 0.75%.

---

## 2. Data Specification

### 2.1 Input Data

| Source | Path | Rows | Used Columns | Date Column | Timezone Fix |
|--------|------|------|-------------|-------------|-------------|
| Base features | data/processed/base_features.csv | 2523 | 5 (see below) | Date | None |
| VIX submodel | data/submodel_outputs/vix.csv | 2857 | 3 | date | None (lowercase) |
| Technical submodel | data/submodel_outputs/technical.csv | 2860 | 3 | date | **utc=True required** |
| Cross-asset submodel | data/submodel_outputs/cross_asset.csv | 2522 | 3 | Date | None |
| Yield curve submodel | data/submodel_outputs/yield_curve.csv | 2794 | 2 | index | None |
| ETF flow submodel | data/submodel_outputs/etf_flow.csv | 2838 | 3 | Date | None |
| Inflation expectation | data/submodel_outputs/inflation_expectation.csv | 2924 | 3 | Unnamed: 0 | None |
| Target | data/processed/target.csv | 2542 | 1 (gold_return_next) | Date | None |

### 2.2 Base Feature Transformation

All base features except VIX must be transformed to daily changes to ensure stationarity:

| # | Raw Column | Transformation | Output Name | Justification |
|---|-----------|---------------|-------------|---------------|
| 1 | real_rate_real_rate | .diff() | real_rate_change | ADF p=0.84 (non-stationary), 2.95 std shift train-to-test |
| 2 | dxy_dxy | .diff() | dxy_change | ADF p=0.47 (non-stationary), 3.05 std shift |
| 3 | vix_vix | None (level) | vix | ADF p=0.000007 (stationary), 0.28 std shift |
| 4 | yield_curve_yield_spread | .diff() | yield_spread_change | ADF p=0.52 (non-stationary), 2.20 std shift |
| 5 | inflation_expectation_inflation_expectation | .diff() | inflation_exp_change | ADF p=0.28 (non-stationary), 1.34 std shift |

**Note**: The first row of each differenced feature will be NaN. This affects only 1 row (the earliest date). Use dropna or impute with 0.0 for this single row.

**Correlation with target (gold_return_next)**:
- real_rate_change: -0.138 (strongest of all base features)
- inflation_exp_change: +0.092
- dxy_change: -0.045
- vix_change: -0.054 (but VIX level is used; level corr is 0.004)
- yield_spread_change: -0.016

### 2.3 Submodel Features (17 columns)

| # | Column | Source | Type | NaN After Merge |
|---|--------|--------|------|----------------|
| 6 | vix_regime_probability | vix | HMM prob [0,1] | 0 |
| 7 | vix_mean_reversion_z | vix | z-score | 0 |
| 8 | vix_persistence | vix | continuous state | 0 |
| 9 | tech_trend_regime_prob | technical | HMM prob [0,1] | 1 |
| 10 | tech_mean_reversion_z | technical | z-score | 1 |
| 11 | tech_volatility_regime | technical | regime state | 1 |
| 12 | xasset_regime_prob | cross_asset | HMM prob [0,1] | 1 |
| 13 | xasset_recession_signal | cross_asset | binary {0,1} | 1 |
| 14 | xasset_divergence | cross_asset | continuous | 1 |
| 15 | yc_spread_velocity_z | yield_curve | z-score | 16 |
| 16 | yc_curvature_z | yield_curve | z-score | 102 |
| 17 | etf_regime_prob | etf_flow | HMM prob [0,1] | 1 |
| 18 | etf_capital_intensity | etf_flow | z-score-like | 1 |
| 19 | etf_pv_divergence | etf_flow | z-score-like | 1 |
| 20 | ie_regime_prob | inflation_expectation | HMM prob [0,1] | 0 |
| 21 | ie_anchoring_z | inflation_expectation | z-score | 0 |
| 22 | ie_gold_sensitivity_z | inflation_expectation | z-score | 0 |

**Excluded submodel columns**:
- yc_regime_prob: constant (std=1.07e-11, HMM collapsed to 1 state)
- All real_rate submodel outputs (7 columns): no_further_improvement after 5 attempts
- All dxy submodel outputs: auto-evaluated, not used
- All cny_demand outputs (3 columns): DA -2.06%, Sharpe -0.593 in Phase 2 ablation

### 2.4 Data Merge Pipeline

```
1. Load base_features.csv (Date column, 2523 rows)
2. Select 5 base columns + Date
3. Compute daily changes for 4 non-stationary features (real_rate, dxy, yield_spread, inflation_exp)
4. Drop first row (NaN from diff) -> 2522 rows
5. Merge with target.csv on Date (inner join) -> 2522 rows
6. For each submodel CSV:
   a. Read CSV
   b. Normalize date column:
      - technical.csv: pd.to_datetime(date, utc=True).dt.strftime('%Y-%m-%d')
      - yield_curve.csv: rename 'index' to 'Date'
      - inflation_expectation.csv: rename 'Unnamed: 0' to 'Date'
      - Others: pd.to_datetime(date_col).dt.strftime('%Y-%m-%d')
   c. Select required columns only (exclude yc_regime_prob)
   d. Left join on Date with main dataframe
7. Apply NaN imputation (see Section 2.5)
8. Verify: 2522 rows, 22 features + 1 target
9. Split: train (first 70%) = 1765, val (next 15%) = 378, test (last 15%) = 379
```

### 2.5 NaN Imputation Strategy

Total NaN rows: ~102 (4.0% of 2522), concentrated in early dates (yield_curve warm-up) and 1 unmatched date at index ~2499.

**Strategy**: Domain-specific imputation based on feature semantics:

| Feature Type | Imputation Value | Rationale |
|-------------|-----------------|-----------|
| regime_prob columns | 0.5 | Maximum uncertainty = no regime information |
| z-score columns | 0.0 | At mean = no signal |
| divergence/signal columns | 0.0 | No information = no signal |
| tech_volatility_regime | median of non-NaN values | Continuous state, median is robust central tendency |
| Base feature changes (row 0 NaN from diff) | 0.0 | No previous day = no change signal |

**Implementation**:
```python
# Regime probabilities -> 0.5
regime_cols = [c for c in feature_cols if 'regime' in c.lower() and ('prob' in c.lower() or 'probability' in c.lower())]
df[regime_cols] = df[regime_cols].fillna(0.5)

# Z-scores -> 0.0
z_cols = [c for c in feature_cols if '_z' in c]
df[z_cols] = df[z_cols].fillna(0.0)

# Divergence/signal -> 0.0
signal_cols = ['xasset_divergence', 'xasset_recession_signal', 'etf_capital_intensity', 'etf_pv_divergence']
df[signal_cols] = df[signal_cols].fillna(0.0)

# Continuous state -> median
df['tech_volatility_regime'] = df['tech_volatility_regime'].fillna(df['tech_volatility_regime'].median())
```

### 2.6 Data Split (frozen from Phase 1)

| Split | Method | Expected Rows | Note |
|-------|--------|---------------|------|
| Train | First 70% | 1765 | |
| Val | Next 15% | 378 | Used for early stopping and Optuna |
| Test | Last 15% | 379 | Final evaluation only |

**Samples per feature ratio**: 1765 / 22 = 80.2:1 (vs Attempt 1: 964/39 = 24.7:1, a 3.2x improvement)

**Test set characteristics** (from Phase 1 baseline_score.json):
- Up days: ~56.9% (215 of 378 non-zero days)
- Naive always-up DA: 56.9%
- Always-long Sharpe (trade cost): 2.065

---

## 3. Model Architecture

### 3.1 Model Selection: XGBoost with reg:squarederror

**Why reg:squarederror (not custom directional MAE)**:

Attempt 1 used directional-weighted MAE with penalty=4.52. This caused:
1. Train DA 94.3% / Val DA 46.9% (47pp gap) -- memorization
2. Prediction std 0.062% vs actual std 1.307% (4.7% ratio) -- magnitude collapse
3. Val 77.3% positive predictions vs 50.2% actual up days -- bias propagation

The custom loss created an adversarial training signal. The model learned to make tiny predictions that minimize penalty rather than predict actual returns. With reg:squarederror:
- Loss is symmetric and well-behaved
- Magnitude accuracy is directly optimized
- Direction accuracy is measured post-hoc via Optuna objective (not in the loss)
- The baseline with reg:squarederror and 19 features had only 11pp gap

### 3.2 Architecture Specification

```
Input: 22-dimensional feature vector
  - 5 base features (1 level + 4 daily changes)
  - 17 submodel features (regime probs, z-scores, signals)
  |
  v
XGBoost Ensemble (gradient boosted trees)
  - Objective: reg:squarederror
  - Up to 1000 boosting rounds with early stopping (patience=50)
  - Aggressive regularization (see Section 4)
  |
  v
Output: Single scalar (predicted next-day gold return %)
  |
  v
Post-processing:
  - Direction: sign(prediction)
  - High-confidence: |prediction| > confidence_threshold
  - Trade signal: sign(prediction) for position, |prediction| for magnitude
```

### 3.3 Sharpe Calculation (CLAUDE.md spec)

```python
def compute_sharpe_trade_cost(predictions, actuals, cost_bps=5.0):
    """
    Compute Sharpe with cost on position changes only.
    Matches CLAUDE.md evaluator specification.
    """
    cost_pct = cost_bps / 100.0  # 5bps = 0.05%

    # Positions: +1 (long) if pred > 0, -1 (short) if pred < 0
    positions = np.sign(predictions)

    # Trades: absolute change in position (0, 1, or 2)
    trades = np.abs(np.diff(positions, prepend=0))

    # Strategy returns
    strategy_returns = positions * actuals

    # Net returns after trade costs
    net_returns = strategy_returns - trades * cost_pct

    if len(net_returns) < 2 or np.std(net_returns) == 0:
        return 0.0

    sharpe = (np.mean(net_returns) / np.std(net_returns)) * np.sqrt(252)
    return sharpe
```

### 3.4 Direction Accuracy Calculation

```python
def compute_direction_accuracy(predictions, actuals):
    """
    DA excluding zeros (CLAUDE.md spec: np.sign(0) = 0 problem).
    """
    nonzero = (actuals != 0) & (predictions != 0)
    if nonzero.sum() == 0:
        return 0.0
    return (np.sign(predictions[nonzero]) == np.sign(actuals[nonzero])).mean() * 100
```

### 3.5 High-Confidence DA Calculation

```python
def compute_hc_da(predictions, actuals, threshold):
    """
    DA on subset where |prediction| > threshold.
    Requires minimum 20% coverage.
    """
    hc_mask = np.abs(predictions) > threshold
    coverage = hc_mask.sum() / len(predictions)
    if coverage < 0.20:
        return 0.0, coverage

    hc_pred = predictions[hc_mask]
    hc_actual = actuals[hc_mask]
    nonzero = (hc_actual != 0) & (hc_pred != 0)
    if nonzero.sum() == 0:
        return 0.0, coverage

    da = (np.sign(hc_pred[nonzero]) == np.sign(hc_actual[nonzero])).mean() * 100
    return da, coverage
```

---

## 4. Hyperparameter Specification

### 4.1 Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective | reg:squarederror | Standard MSE loss. No custom objective. Eliminates Attempt 1 root cause. |
| n_estimators | 1000 | Upper bound; early stopping at patience 50 prevents overtraining |
| early_stopping_rounds | 50 | Patience for validation loss convergence |
| eval_metric | rmse | Standard metric for reg:squarederror early stopping |
| tree_method | hist | Fast histogram-based algorithm |
| verbosity | 0 | Suppress output for Optuna |
| seed | 42 | Base seed; trial-specific: 42 + trial.number |

### 4.2 Optuna Search Space

| Parameter | Range | Scale | Type | Rationale |
|-----------|-------|-------|------|-----------|
| max_depth | [2, 4] | linear | int | Attempt 1 used depth 3 but overfit due to loss fn. With corrected loss and fewer features, depth 2-4 prevents leaf memorization. |
| min_child_weight | [10, 30] | linear | int | Forces leaves to cover >= 0.6-1.7% of training data. Prevents pattern memorization on small subsets. |
| subsample | [0.4, 0.7] | linear | float | Row sampling. More aggressive than Attempt 1 (0.695). |
| colsample_bytree | [0.3, 0.6] | linear | float | With 22 features, samples 7-13 features per tree. Prevents feature-specific memorization. |
| reg_lambda (L2) | [3.0, 20.0] | log | float | 2x-13x stronger than Attempt 1 (1.56). Strong weight shrinkage. |
| reg_alpha (L1) | [1.0, 10.0] | log | float | 4x-40x stronger than Attempt 1 (0.249). Encourages sparser leaf weights. |
| learning_rate | [0.005, 0.03] | log | float | Slower learning. Lower upper bound than Attempt 1 (0.05). |
| gamma | [0.5, 3.0] | linear | float | Minimum loss reduction per split. Higher minimum (0.5 vs 0.0) prevents trivial splits. |
| n_estimators | [100, 800] | linear | int | Allow Optuna to control tree count directly alongside early stopping. |
| confidence_threshold | [0.01, 0.10] | log | float | For HC-DA. Attempt 1 used 0.0116 (too low, almost all predictions were "high confidence"). Range 1-10% of predicted return. |

**Total: 10 hyperparameters** (8 XGBoost regularization + 1 tree count + 1 post-processing threshold)

### 4.3 Search Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_trials | 80 | More trials than Attempt 1 (50) to better explore aggressive regularization space |
| timeout | 7200 (2 hours) | XGBoost on 1765 rows is fast (~1 min/trial). 80 min expected + 40 min margin. |
| sampler | TPESampler(seed=42) | Standard for mixed continuous/integer spaces |
| pruner | None | XGBoost trials are fast enough; no pruning needed |
| direction | maximize | Maximize composite objective |

### 4.4 Optuna Objective Function

```python
def optuna_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 2, 4),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 30),
        'subsample': trial.suggest_float('subsample', 0.4, 0.7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.6),
        'reg_lambda': trial.suggest_float('reg_lambda', 3.0, 20.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 10.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        'gamma': trial.suggest_float('gamma', 0.5, 3.0),
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'seed': 42 + trial.number,
    }

    n_estimators = trial.suggest_int('n_estimators', 100, 800)
    conf_threshold = trial.suggest_float('confidence_threshold', 0.01, 0.10, log=True)

    model = xgb.XGBRegressor(**params, n_estimators=n_estimators, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)

    # Compute all 4 metrics on validation set
    val_mae = np.mean(np.abs(val_pred - y_val))
    val_da = compute_direction_accuracy(val_pred, y_val)
    val_sharpe = compute_sharpe_trade_cost(val_pred, y_val)
    val_hc_da, val_hc_coverage = compute_hc_da(val_pred, y_val, conf_threshold)

    # Also compute train DA for overfitting check
    train_pred = model.predict(X_train)
    train_da = compute_direction_accuracy(train_pred, y_train)
    da_gap = train_da - val_da

    # Normalize to [0, 1]
    sharpe_norm = np.clip((val_sharpe + 3.0) / 6.0, 0.0, 1.0)   # Sharpe in [-3, +3]
    da_norm = np.clip((val_da - 40.0) / 30.0, 0.0, 1.0)          # DA in [40%, 70%]
    mae_norm = np.clip((1.0 - val_mae) / 0.5, 0.0, 1.0)          # MAE in [0.5%, 1.0%]
    hc_da_norm = np.clip((val_hc_da - 40.0) / 30.0, 0.0, 1.0)    # HC-DA in [40%, 70%]

    # Overfitting penalty: reduce objective if train-val DA gap > 10pp
    overfit_penalty = max(0.0, (da_gap - 10.0) / 30.0)  # 0 if gap<=10, up to 1.0 if gap=40

    # Weighted composite (higher is better)
    objective = (
        0.50 * sharpe_norm +     # Sharpe is binding (Attempt 1: 0.428 vs target 0.8)
        0.30 * da_norm +         # DA is critical (Attempt 1: 54.1% vs target 56%)
        0.10 * mae_norm +        # MAE is non-binding (baseline 0.714 < 0.75 target)
        0.10 * hc_da_norm        # HC-DA requires confidence calibration
    ) - 0.30 * overfit_penalty   # Penalize overfitting directly

    # Log for analysis
    trial.set_user_attr('val_mae', float(val_mae))
    trial.set_user_attr('val_da', float(val_da))
    trial.set_user_attr('val_hc_da', float(val_hc_da))
    trial.set_user_attr('val_hc_coverage', float(val_hc_coverage))
    trial.set_user_attr('val_sharpe', float(val_sharpe))
    trial.set_user_attr('train_da', float(train_da))
    trial.set_user_attr('da_gap_pp', float(da_gap))
    trial.set_user_attr('n_estimators_used', int(model.best_iteration + 1) if hasattr(model, 'best_iteration') else n_estimators)

    return objective
```

**Weight rationale**:
- Sharpe 50%: Binding constraint. Attempt 1 achieved 0.428 (daily cost) vs target 0.8. Largest gap from target.
- DA 30%: Critical. Attempt 1: 54.1% vs target 56%. Small gap but high leverage.
- MAE 10%: Non-binding. Baseline MAE 0.714 already < 0.75 target.
- HC-DA 10%: Requires confidence calibration. Secondary priority.
- Overfitting penalty 30%: Directly penalizes train-val DA gap > 10pp. This is new vs Attempt 1 and explicitly targets the primary failure mode.

---

## 5. Training Configuration

### 5.1 Training Algorithm

```
1. DATA PREPARATION:
   a. Fetch raw data using yfinance and fredapi
   b. Construct base features (replicate Phase 1 logic)
   c. Compute daily changes for real_rate, dxy, yield_spread, inflation_exp
   d. Load submodel output CSVs with proper date normalization
   e. Merge base features + submodel outputs + target on Date (left join from base)
   f. Apply NaN imputation (Section 2.5)
   g. Drop rows with any remaining NaN (should be 0 after imputation)
   h. Split: train (70%), val (15%), test (15%) by time order
   i. Separate features (X) from target (y = gold_return_next)

2. OPTUNA HPO (80 trials, 2-hour timeout):
   a. For each trial:
      - Sample 10 hyperparameters
      - Train XGBoost with reg:squarederror and early stopping
      - Compute val metrics: DA, Sharpe (trade cost), MAE, HC-DA
      - Compute train-val DA gap and overfitting penalty
      - Return weighted composite objective
   b. Select best trial (highest composite objective)

3. FINAL MODEL TRAINING:
   a. Re-train with best hyperparameters on (X_train, y_train)
   b. Early stop on (X_val, y_val) using RMSE
   c. Record best_iteration

4. EVALUATION ON ALL SPLITS:
   a. Predict on train, val, test
   b. Compute all 4 target metrics per split
   c. Compute train-test DA gap
   d. Compute overfit ratios
   e. Compute feature importance
   f. Compute naive always-up DA and always-long Sharpe for comparison

5. SAVE RESULTS:
   a. training_result.json
   b. model.json (XGBoost model)
   c. predictions.csv
   d. submodel_output.csv (for pipeline compatibility)
```

### 5.2 Loss Function

- **Primary and only**: reg:squarederror (standard MSE)
- **No fallback**: The custom objective is eliminated entirely. If reg:squarederror underperforms, the response is to adjust regularization or features, not to reintroduce custom losses.

### 5.3 Early Stopping

- **Metric**: RMSE on validation set (native to reg:squarederror)
- **Patience**: 50 rounds
- **Maximum rounds**: Optuna-controlled (100-800), not fixed at 1000

### 5.4 Evaluation Metrics

| Metric | Formula | Target | Attempt 1 Result |
|--------|---------|--------|-----------------|
| DA | sign agreement, excluding pred=0 or actual=0 | > 56% | 54.1% |
| HC-DA | DA on |pred| > threshold, min 20% coverage | > 60% | 54.3% |
| MAE | mean(|pred - actual|) | < 0.75% | 0.978% |
| Sharpe | (mean(net) / std(net)) * sqrt(252), trade-cost | > 0.8 | 1.027 (trade cost) |
| Train-test DA gap | train_DA - test_DA | < 10pp | 40.15pp |
| Overfit ratio (MAE) | test_MAE / train_MAE | < 1.5 | 1.586 |

---

## 6. Kaggle Execution Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | XGBoost on 1765 samples is CPU-fast. GPU overhead exceeds benefit for small data. |
| Estimated execution time | 15-30 minutes | 80 trials * 10-20 sec/trial + data loading + final evaluation |
| Estimated memory usage | 1.5 GB | 2522 rows * 22 features * 8 bytes = ~0.4 MB data. XGBoost model < 50 MB. Optuna overhead ~100 MB. |
| Required pip packages | [] | All pre-installed on Kaggle (xgboost, optuna, pandas, numpy, scikit-learn, yfinance, fredapi) |
| Internet required | true | For data fetching (yfinance, fredapi) |
| Kaggle Notebook ID | bigbigzabuton/gold-model-training | Unified notebook with FRED_API_KEY secret |
| Optuna timeout | 7200 sec (2 hours) | 80 trials * ~15 sec = ~20 min expected. Generous margin. |

---

## 7. Implementation Instructions

### 7.1 For builder_data

**No separate data preparation step needed.** All data loading and merging is handled inside the self-contained Kaggle notebook. The notebook:
1. Fetches raw data via yfinance/fredapi
2. Constructs base features
3. Loads pre-generated submodel CSV files (these are already committed to the repo and available on Kaggle via data fetching or embedding)

However, if builder_data is invoked, it should verify:
- data/submodel_outputs/vix.csv exists with columns [date, vix_regime_probability, vix_mean_reversion_z, vix_persistence]
- data/submodel_outputs/technical.csv exists with columns [date, tech_trend_regime_prob, tech_mean_reversion_z, tech_volatility_regime]
- data/submodel_outputs/cross_asset.csv exists with columns [Date, xasset_regime_prob, xasset_recession_signal, xasset_divergence]
- data/submodel_outputs/yield_curve.csv exists with columns [index, yc_spread_velocity_z, yc_curvature_z] (yc_regime_prob present but unused)
- data/submodel_outputs/etf_flow.csv exists with columns [Date, etf_regime_prob, etf_capital_intensity, etf_pv_divergence]
- data/submodel_outputs/inflation_expectation.csv exists with columns [Unnamed: 0, ie_regime_prob, ie_anchoring_z, ie_gold_sensitivity_z]

### 7.2 For builder_model

**Task**: Generate `notebooks/meta_model_2/train.ipynb` (self-contained Kaggle Notebook)

**Critical implementation details**:

1. **Date normalization is mandatory**. The Attempt 1 bug that lost 45% of data was caused by technical.csv having timezone-aware dates. The notebook MUST:
   ```python
   # For technical.csv ONLY:
   tech = pd.read_csv('data/submodel_outputs/technical.csv')
   tech['Date'] = pd.to_datetime(tech['date'], utc=True).dt.strftime('%Y-%m-%d')

   # For yield_curve.csv:
   yc = pd.read_csv('data/submodel_outputs/yield_curve.csv')
   yc = yc.rename(columns={'index': 'Date'})

   # For inflation_expectation.csv:
   ie = pd.read_csv('data/submodel_outputs/inflation_expectation.csv')
   ie = ie.rename(columns={'Unnamed: 0': 'Date'})

   # For all others:
   df['Date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
   ```

2. **Base feature transformation**:
   ```python
   # Differencing for non-stationary features
   base['real_rate_change'] = base['real_rate_real_rate'].diff()
   base['dxy_change'] = base['dxy_dxy'].diff()
   base['yield_spread_change'] = base['yield_curve_yield_spread'].diff()
   base['inflation_exp_change'] = base['inflation_expectation_inflation_expectation'].diff()
   # VIX stays as level
   base['vix'] = base['vix_vix']
   # Drop first row (NaN from diff)
   base = base.iloc[1:].reset_index(drop=True)
   ```

3. **Use XGBRegressor API** (not xgb.train). Since we are using standard reg:squarederror with no custom objective, XGBRegressor.fit() with eval_set and early_stopping_rounds is the clean approach.

4. **Sharpe formula must use trade-cost (not daily cost)**. Copy the compute_sharpe_trade_cost function from Section 3.3 exactly.

5. **Include overfitting penalty in Optuna objective**. This is a new addition vs Attempt 1.

6. **Diagnostic outputs** must include:
   - Naive always-up DA on test set
   - Always-long Sharpe (trade cost) on test set
   - Train/val/test DA, Sharpe, MAE, HC-DA
   - Train-test DA gap in percentage points
   - Feature importance (all 22 features)
   - Prediction distribution stats (mean, std, % positive)
   - All Optuna trial summaries
   - Overfit ratio (MAE test/train)

7. **Output files** (saved to Kaggle output):
   - `training_result.json`: All metrics, params, feature importance, per-split results
   - `model.json`: XGBoost model (model.save_model('model.json'))
   - `predictions.csv`: [date, split, prediction, actual, direction_correct, high_confidence]
   - `submodel_output.csv`: Final predictions aligned with dates

8. **Data fetching strategy**: Since the notebook must be self-contained, embed the submodel data fetching code from previous notebooks. The builder_model should:
   - Include yfinance/fredapi calls for base features
   - Include HMM fitting code for each submodel (copy from existing notebooks)
   - OR: Use a hybrid approach where submodel outputs are embedded as compressed data

### 7.3 Feature List for builder_model Reference

The exact 22 feature column names that the XGBoost model receives:

```python
FEATURE_COLUMNS = [
    # Base features (5)
    'real_rate_change',
    'dxy_change',
    'vix',
    'yield_spread_change',
    'inflation_exp_change',
    # VIX submodel (3)
    'vix_regime_probability',
    'vix_mean_reversion_z',
    'vix_persistence',
    # Technical submodel (3)
    'tech_trend_regime_prob',
    'tech_mean_reversion_z',
    'tech_volatility_regime',
    # Cross-asset submodel (3)
    'xasset_regime_prob',
    'xasset_recession_signal',
    'xasset_divergence',
    # Yield curve submodel (2)
    'yc_spread_velocity_z',
    'yc_curvature_z',
    # ETF flow submodel (3)
    'etf_regime_prob',
    'etf_capital_intensity',
    'etf_pv_divergence',
    # Inflation expectation submodel (3)
    'ie_regime_prob',
    'ie_anchoring_z',
    'ie_gold_sensitivity_z',
]
assert len(FEATURE_COLUMNS) == 22
```

---

## 8. Risk Mitigation

### Risk 1: Overfitting (PRIMARY RISK -- downgraded from Attempt 1)

**Problem**: Attempt 1 had 40pp train-test DA gap with 39 features and custom loss.

**Why this risk is reduced**:
- Root cause (directional MAE penalty=4.52) is eliminated
- Feature count reduced from 39 to 22 (43% reduction)
- Non-stationary features eliminated (no more price levels causing regime-dependent splits)
- Samples-per-feature ratio improved from 24.7:1 to 80.2:1
- Overfitting penalty explicitly included in Optuna objective
- Regularization ranges are 2-13x stronger

**Remaining risk**: XGBoost may still overfit if regularization is insufficient. The baseline with 19 features had 11pp gap; with 22 features and stronger regularization, target is <5pp.

**Monitoring**: Report train/val/test DA. Flag if gap > 10pp.

### Risk 2: Underfitting from Excessive Regularization

**Problem**: Aggressive regularization (max_depth=2, min_child_weight=30, reg_lambda=20) may prevent the model from learning any useful patterns, resulting in predictions near zero.

**Mitigations**:
- Optuna explores the full range [2,4] for depth, [10,30] for min_child_weight
- If Optuna converges to the least-regularized corner consistently (depth=4, min_child_weight=10, lambda=3), this indicates regularization is too strong
- Monitor prediction std: if < 0.05% on val set (comparable to Attempt 1's collapsed predictions), flag

**Contingency**: If Attempt 2 shows train DA < 55% AND test DA < 53% (both low, gap < 5pp but underfitting), relax regularization in Attempt 3.

### Risk 3: Differencing Removes Useful Level Information

**Problem**: By converting real_rate, yield_spread, inflation_exp to daily changes, we lose the ability to make regime-conditional predictions (e.g., "when real rates are high, gold tends to fall"). This information exists in levels but causes distribution shift.

**Mitigations**:
- Submodel features already capture regime states (regime_prob columns encode HMM-detected regimes)
- VIX is kept as a level (only stationary base feature)
- The correlation data shows changes have 2-10x stronger linear correlation with gold returns than levels
- XGBoost can learn nonlinear interactions between change features and regime probabilities

**Contingency**: If Attempt 2 underperforms baseline, consider adding z-scored versions of levels (subtract rolling mean / divide by rolling std) instead of raw levels. This preserves regime information while reducing distribution shift.

### Risk 4: Test Set Bias (Always-Up DA = 56.9%)

**Problem**: The test period (Aug 2023 - Feb 2025) had a gold rally. Naive always-up achieves 56.9% DA, already exceeding the 56% target.

**Mitigations**:
- Model DA must exceed naive always-up DA to demonstrate real predictive skill
- Sharpe > 0.8 requires correct magnitude, not just directional bias
- HC-DA > 60% requires above-random accuracy on confident predictions
- Report model DA alongside naive DA in results

### Risk 5: Sharpe Formula Change Inflates Metrics

**Problem**: Switching from daily-cost to trade-cost Sharpe will mechanically increase all Sharpe values. The 0.8 target may become trivially achievable.

**Assessment**: Always-long Sharpe increases from 1.183 (daily cost) to 2.065 (trade cost). If the model produces a reasonable long-biased strategy (given the bull market in test), Sharpe > 0.8 may be relatively easy. The binding constraints become DA and HC-DA.

**Mitigations**: Report both daily-cost and trade-cost Sharpe for comparison with Attempt 1. The evaluator should check that the model outperforms always-long strategy, not just the 0.8 threshold.

### Risk 6: Kaggle Timeout

**Problem**: Increased trial count (80 vs 50) may extend runtime.

**Assessment**: With reg:squarederror (no custom objective overhead), XGBoost on 1765 rows trains in <20 sec per trial. 80 trials = ~25 min. Well within Kaggle's 9-hour limit. Optuna timeout set to 2 hours as safety margin.

---

## 9. Expected Outcomes

| Metric | Attempt 1 (Actual) | Attempt 2 Target | Confidence |
|--------|-------------------|------------------|------------|
| Train DA | 94.3% | 55-65% | High |
| Test DA | 54.1% | > 56% | Medium-High |
| Train-test DA gap | 40.15pp | < 10pp (ideally < 5pp) | High |
| Test HC-DA | 54.3% | > 60% | Medium |
| Test MAE | 0.978% | < 0.75% | Medium |
| Test Sharpe (trade cost) | 1.027 | > 0.8 | Medium-High |
| Features | 39 | 22 | Fixed |
| Training samples | 964 | ~1765 | Fixed |
| Samples/feature | 24.7:1 | 80.2:1 | Fixed |

**Overall success probability**: 60-70%.

The structural fixes (remove custom loss, remove non-stationary features, fix data pipeline, correct Sharpe formula) address all 5 root causes from Attempt 1. The primary uncertainty is whether the 22 stationary features + submodel outputs carry enough signal for DA > 56% and HC-DA > 60%, given that the baseline with 19 features (mostly non-stationary) achieved only 43.5% test DA. However, Phase 2 Gate 3 results demonstrate that submodel features provide genuine information gain (VIX: DA +0.96%, cross_asset: DA +0.76%, inflation_expectation: DA +0.57%).

---

## 10. Success Criteria

### Primary (all must pass on test set)

| Metric | Target | Formula |
|--------|--------|---------|
| DA | > 56% | sign agreement, excluding zeros |
| HC-DA | > 60% | DA on |pred| > threshold, min 20% coverage |
| MAE | < 0.75% | mean(|pred - actual|) |
| Sharpe | > 0.8 | annualized, after 5bps trade cost (CLAUDE.md spec) |
| Train-test DA gap | < 10pp | train_DA - test_DA |

### Secondary Diagnostics

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Overfit ratio (MAE test/train) | < 1.5 | Overfitting control |
| Prediction std | > 0.05% | Not degenerate (Attempt 1: 0.062%) |
| HC coverage | >= 20% | Meaningful confidence threshold |
| Model DA > Naive always-up DA | True | Real predictive skill |
| Feature importance | Submodel features in top 10 | Submodels contribute |

### Decision Rules After Evaluation

| Outcome | Action |
|---------|--------|
| All 5 primary criteria met | Phase 3 COMPLETE. Merge to main. |
| Sharpe and DA met, HC-DA/MAE missed | Attempt 3: adjust confidence threshold, add feature engineering |
| DA gap > 10pp despite corrections | Attempt 3: further regularization, reduce features, add bagging |
| DA < 50% (underfitting) | Attempt 3: relax regularization, add z-scored levels |
| All targets missed but gap < 10pp | Attempt 3: the model is well-regularized but lacks signal. Add feature interactions or time-windowed features. |

---

**End of Design Document**

**Architect**: architect (Opus)
**Date**: 2026-02-15
**Based on**: docs/research/meta_model_attempt_2.md (with corrections)
**Supersedes**: docs/design/meta_model_attempt_1.md
