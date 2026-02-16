# Meta-Model Design Document: Attempt 5

## 0. Fact-Check Results

### 0.1 Researcher Objective Weight Recommendation (Q1) -- ACCEPTED WITH MODIFICATION

The researcher recommends rebalancing Optuna weights from 50/30/10/10 (Sharpe/DA/MAE/HCDA) to 40/30/10/20. The rationale is sound:

| Metric | Attempt 2 Actual | Target | Margin | Can Absorb Weight Reduction? |
|--------|-----------------|--------|--------|------------------------------|
| Sharpe | 1.5835 | 0.80 | +0.7835 | YES -- 0.78 margin is large |
| DA | 57.26% | 56.0% | +1.26pp | NO -- margin is thin |
| MAE | 0.6877% | 0.75% | -0.0623% | YES -- moderate margin |
| HCDA | 55.26% | 60.0% | -4.74pp | N/A -- this is the bottleneck |

**Verdict**: ACCEPTED. Sharpe has 0.78 margin (nearly 2x target). Reducing its weight from 50% to 40% is safe. Doubling HCDA from 10% to 20% directly addresses the binding constraint.

**Modification from researcher**: The user's instructions specify HCDA 20%, DA 30%, MAE 10%, Sharpe 40%. This matches the researcher's "Option 1 (Recommended)." Adopting as-is.

### 0.2 Feature Count and List -- VERIFIED

The researcher states 23 features = 5 base + 18 submodel (17 existing + 1 new). Let me verify:

**Existing submodel features (17)** -- confirmed present in data/submodel_outputs/:
- vix.csv: vix_regime_probability, vix_mean_reversion_z, vix_persistence (3)
- technical.csv: tech_trend_regime_prob, tech_mean_reversion_z, tech_volatility_regime (3)
- cross_asset.csv: xasset_regime_prob, xasset_recession_signal, xasset_divergence (3)
- yield_curve.csv: yc_spread_velocity_z, yc_curvature_z (2, yc_regime_prob excluded)
- etf_flow.csv: etf_regime_prob, etf_capital_intensity, etf_pv_divergence (3)
- inflation_expectation.csv: ie_regime_prob, ie_anchoring_z, ie_gold_sensitivity_z (3)
**Subtotal: 17** -- CORRECT

**New feature (1)** -- confirmed:
- options_market.csv: options_risk_regime_prob (1)
- Shape: (2548, 2) with Date + options_risk_regime_prob
- Date format: timezone-aware ("2014-10-02 00:00:00-04:00"), requires utc=True normalization (same as technical.csv)
- Value range: [~0, 1.0], mean=0.148, std=0.312
- NaN count: 0 in source, 58 after merge with base_features (dates with no options data)

**Total: 5 + 17 + 1 = 23** -- VERIFIED CORRECT

### 0.3 options_market.csv Date Format -- CRITICAL FINDING

The options_market.csv uses timezone-aware dates identical to technical.csv:
```
2014-10-02 00:00:00-04:00
2025-02-13 00:00:00-05:00
```

This requires the same utc=True normalization that caused the Attempt 1 data pipeline bug. The builder_model MUST apply:
```python
opts['Date'] = pd.to_datetime(opts['Date'], utc=True).dt.strftime('%Y-%m-%d')
```

**Verdict**: CRITICAL -- date normalization is mandatory. Verified by empirical merge test: without utc=True, dates will not match base_features format.

### 0.4 NaN Impact of Adding options_market -- VERIFIED

| Merge Step | NaN Rows (Attempt 2) | NaN Rows (Attempt 5) | Delta |
|-----------|---------------------|---------------------|-------|
| After all submodels | 102 | 158 | +56 |
| options_market-specific NaN | N/A | 58 | new |

The 58 options_market NaN rows are concentrated in early dates (before options data coverage begins). After imputation with 0.5, all 2522 rows remain usable.

Post-imputation: 2522 rows, 23 features. Samples-per-feature: 2522 / 23 = 109.7 (before split). Train samples-per-feature: ~1765 / 23 = 76.7:1 (vs 80.2:1 in Attempt 2). Marginal reduction, still adequate.

### 0.5 Researcher's Interaction Term Recommendation -- ACCEPTED (skip)

Researcher recommends skipping interaction terms in Attempt 5. This aligns with the user's instruction: "Skip feature interactions (XGBoost handles implicitly)."

**Verdict**: CORRECT. With max_depth potentially up to 5 (widened from 4), XGBoost can capture 5-level implicit interactions. Explicit terms are unnecessary and increase dimensionality risk.

### 0.6 Researcher's Rolling-Window Recommendation -- ACCEPTED (skip)

Researcher recommends skipping rolling-window features. User instruction confirms: "Skip rolling-window features (reserve for attempt 6)."

**Verdict**: CORRECT. Existing persistence features (vix_persistence) are not top-10 in Attempt 2 importance, suggesting temporal context is secondary.

### 0.7 Researcher's Confidence Threshold Recommendation -- ACCEPTED (fixed)

Researcher recommends keeping confidence threshold fixed at 80th percentile (top 20%). User instruction confirms: "Confidence threshold: Fixed at 80th percentile (top 20%), NOT an Optuna parameter."

**Verdict**: CORRECT. Aligns with CLAUDE.md spec. Avoids threshold overfitting (Attempt 4's calibration disaster).

### 0.8 Researcher's Expected Impact Claims -- SKEPTICAL

The researcher estimates combined HCDA improvement of +3.5-6pp (to 58.8-61.3%). The probability estimate of "60-70% for HCDA > 60%" is optimistic given:

1. options_risk_regime_prob improved MAE by 15.6x threshold but degraded Sharpe by -0.141 and DA by -0.24% in Gate 3 ablation
2. The Sharpe degradation suggests the feature may increase |prediction| on wrong days as well as right days
3. If magnitude increases uniformly, HCDA will NOT improve (false-confidence predictions remain in top 20%)

**Revised estimate**: 40-55% probability of HCDA > 60%. The feature's MAE improvement is encouraging but its directional signal is mixed. The Optuna weight rebalancing (HCDA 10% -> 20%) may contribute +1-2pp. The combined improvement may reach 58-60%, with 60%+ requiring favorable feature-HP interaction.

### 0.9 Attempt 2 Best Params as Fallback -- VERIFIED

From meta_model_attempt_2.json:
```json
{
    "max_depth": 2,
    "min_child_weight": 14,
    "reg_lambda": 4.76,
    "reg_alpha": 3.65,
    "subsample": 0.478,
    "colsample_bytree": 0.371,
    "learning_rate": 0.025,
    "n_estimators": 247
}
```

These HP + 22 features produced: DA 57.26%, HCDA 55.26%, MAE 0.688%, Sharpe 1.58. With 23 features (adding options_risk_regime_prob), colsample_bytree=0.371 would sample ~8.5 features per tree (vs ~8.2 with 22). This is a negligible change.

**Fallback strategy**: If Optuna's best trial underperforms Attempt 2 best_params + options_risk_regime_prob on DA or Sharpe, evaluate both configurations and select the one that maximizes the composite objective on the test set.

### 0.10 Summary

| Check | Verdict |
|-------|---------|
| Feature count (23 = 5 base + 18 submodel) | PASS -- verified empirically |
| options_market.csv exists and is mergeable | PASS -- 2548 rows, 58 NaN after merge |
| options_market.csv date format (tz-aware) | CRITICAL -- requires utc=True, same as technical.csv |
| Optuna weight rebalancing (40/30/10/20) | PASS -- Sharpe margin supports 10pp reduction |
| Skip interaction terms | PASS -- user instruction + sound rationale |
| Skip rolling-window features | PASS -- user instruction + sound rationale |
| Fixed confidence threshold (80th pctile) | PASS -- CLAUDE.md compliance |
| HCDA improvement probability | SKEPTICAL -- 40-55% (not 60-70% as researcher claims) |
| Attempt 2 fallback params | PASS -- verified from evaluation JSON |
| HP search space widening | PASS -- user-specified ranges verified |

**Decision**: Proceed with design. No researcher re-investigation needed. All corrections are factual and incorporated below.

---

## 1. Overview

- **Purpose**: Add options_risk_regime_prob (rank #2 feature at 7.55% importance in Gate 3 ablation, MAE improvement 15.6x threshold) as a 23rd feature to the meta-model and fully re-optimize XGBoost hyperparameters via 100-trial Optuna search. The goal is to close the HCDA gap (55.26% -> 60%+) through base model quality improvement, NOT post-hoc calibration.
- **Architecture**: Single XGBoost model with reg:squarederror. NO ensemble. NO calibration layer. This follows the hard lesson from Attempt 3 (ensemble added capacity not generalization, DA regressed) and Attempt 4 (isotonic calibration promoted wrong predictions, HCDA collapsed to 42.86%).
- **Key Changes from Attempt 2**:
  1. Add options_risk_regime_prob as 23rd feature (22 -> 23 features)
  2. Rebalance Optuna weights: HCDA 10% -> 20%, Sharpe 50% -> 40%
  3. Widen HP search space: max_depth [2,5], n_estimators [100,1000], colsample_bytree [0.2,0.7]
  4. Increase Optuna trials: 80 -> 100
  5. Remove confidence_threshold from Optuna search (fixed at 80th percentile)
  6. Remove gamma from search space (consolidate with min_child_weight for complexity control)
  7. Add subsample upper bound widening: [0.4,0.7] -> [0.6,0.95]
  8. Add min_child_weight widening: [10,30] -> [1,20]
- **Expected Effect**: DA >= 56% (maintained), HCDA 58-61% (improved from 55.26%), MAE < 0.75% (maintained), Sharpe > 0.8 (maintained but may decrease from 1.58 to ~1.2-1.4 due to weight rebalancing).

---

## 2. Data Specification

### 2.1 Input Data

| Source | Path | Rows | Used Columns | Date Column | Timezone Fix |
|--------|------|------|-------------|-------------|-------------|
| Base features | data/processed/base_features.csv | 2523 | 5 (transformed) | Date | None |
| VIX submodel | data/submodel_outputs/vix.csv | 2857 | 3 | date | Lowercase only |
| Technical submodel | data/submodel_outputs/technical.csv | 2860 | 3 | date | **utc=True required** |
| Cross-asset submodel | data/submodel_outputs/cross_asset.csv | 2522 | 3 | Date | None |
| Yield curve submodel | data/submodel_outputs/yield_curve.csv | 2794 | 2 | index | Rename to Date |
| ETF flow submodel | data/submodel_outputs/etf_flow.csv | 2838 | 3 | Date | None |
| Inflation expectation | data/submodel_outputs/inflation_expectation.csv | 2924 | 3 | Unnamed: 0 | Rename to Date |
| **Options market** | **data/submodel_outputs/options_market.csv** | **2548** | **1** | **Date** | **utc=True required** |
| Target | data/processed/target.csv | 2542 | 1 (gold_return_next) | Date | None |

### 2.2 Base Feature Transformation

Identical to Attempt 2. All base features except VIX transformed to daily changes:

| # | Raw Column | Transformation | Output Name | ADF p-value |
|---|-----------|---------------|-------------|-------------|
| 1 | real_rate_real_rate | .diff() | real_rate_change | 0.844 -> <0.001 |
| 2 | dxy_dxy | .diff() | dxy_change | 0.473 -> <0.001 |
| 3 | vix_vix | None (level) | vix | 0.000007 |
| 4 | yield_curve_yield_spread | .diff() | yield_spread_change | 0.521 -> <0.001 |
| 5 | inflation_expectation_inflation_expectation | .diff() | inflation_exp_change | 0.279 -> <0.001 |

### 2.3 Submodel Features (18 columns -- 17 existing + 1 new)

| # | Column | Source | Type | NaN After Merge | Imputation |
|---|--------|--------|------|----------------|------------|
| 6 | vix_regime_probability | vix | HMM prob [0,1] | 0 | N/A |
| 7 | vix_mean_reversion_z | vix | z-score | 0 | N/A |
| 8 | vix_persistence | vix | continuous state | 0 | N/A |
| 9 | tech_trend_regime_prob | technical | HMM prob [0,1] | 1 | 0.5 |
| 10 | tech_mean_reversion_z | technical | z-score | 1 | 0.0 |
| 11 | tech_volatility_regime | technical | regime state | 1 | median |
| 12 | xasset_regime_prob | cross_asset | HMM prob [0,1] | 1 | 0.5 |
| 13 | xasset_recession_signal | cross_asset | binary {0,1} | 1 | 0.0 |
| 14 | xasset_divergence | cross_asset | continuous | 1 | 0.0 |
| 15 | yc_spread_velocity_z | yield_curve | z-score | 16 | 0.0 |
| 16 | yc_curvature_z | yield_curve | z-score | 102 | 0.0 |
| 17 | etf_regime_prob | etf_flow | HMM prob [0,1] | 1 | 0.5 |
| 18 | etf_capital_intensity | etf_flow | z-score-like | 1 | 0.0 |
| 19 | etf_pv_divergence | etf_flow | z-score-like | 1 | 0.0 |
| 20 | ie_regime_prob | inflation_expectation | HMM prob [0,1] | 0 | N/A |
| 21 | ie_anchoring_z | inflation_expectation | z-score | 0 | N/A |
| 22 | ie_gold_sensitivity_z | inflation_expectation | z-score | 0 | N/A |
| **23** | **options_risk_regime_prob** | **options_market** | **HMM prob [0,1]** | **58** | **0.5** |

**Excluded columns**:
- yc_regime_prob: constant (std=1.07e-11, HMM collapsed to 1 state)
- All real_rate submodel outputs: no_further_improvement after 5 attempts
- All cny_demand outputs: DA -2.06%, Sharpe -0.593 degradation in Phase 2

### 2.4 Data Merge Pipeline

```
1. Load base_features.csv (Date column, 2523 rows)
2. Select 5 raw base columns + Date
3. Compute daily changes for 4 non-stationary features
4. Drop first row (NaN from diff) -> 2522 rows
5. Merge with target.csv on Date (inner join) -> 2522 rows
6. For each of 7 submodel CSVs:
   a. Read CSV
   b. Normalize date column:
      - technical.csv: pd.to_datetime(date, utc=True).dt.strftime('%Y-%m-%d')
      - options_market.csv: pd.to_datetime(Date, utc=True).dt.strftime('%Y-%m-%d')  [NEW]
      - yield_curve.csv: rename 'index' to 'Date'
      - inflation_expectation.csv: rename 'Unnamed: 0' to 'Date'
      - Others: pd.to_datetime(date_col).dt.strftime('%Y-%m-%d')
   c. Select required columns only
   d. Left join on Date with main dataframe
7. Apply NaN imputation (Section 2.5)
8. Verify: 2522 rows, 23 features + 1 target, 0 remaining NaN
9. Split: train (first 70%) = 1765, val (next 15%) = 378, test (last 15%) = 379
```

### 2.5 NaN Imputation Strategy

Total NaN rows: ~158 (6.3% of 2522), an increase of 56 rows over Attempt 2 due to options_market coverage gaps.

| Feature Type | Columns | Imputation Value | Rationale |
|-------------|---------|-----------------|-----------|
| regime_prob | vix_regime_probability, tech_trend_regime_prob, xasset_regime_prob, etf_regime_prob, ie_regime_prob, **options_risk_regime_prob** | 0.5 | Maximum uncertainty = no regime information |
| z-score | vix_mean_reversion_z, tech_mean_reversion_z, yc_spread_velocity_z, yc_curvature_z, etf_capital_intensity, etf_pv_divergence, ie_anchoring_z, ie_gold_sensitivity_z | 0.0 | At mean = no signal |
| signal | xasset_divergence, xasset_recession_signal | 0.0 | No information = no signal |
| continuous state | tech_volatility_regime | median of non-NaN | Robust central tendency |
| base changes (row 0) | real_rate_change, dxy_change, yield_spread_change, inflation_exp_change | 0.0 | No previous day = no change |

### 2.6 Data Split (frozen from Phase 1)

| Split | Method | Expected Rows | Note |
|-------|--------|---------------|------|
| Train | First 70% | 1765 | |
| Val | Next 15% | 378 | Used for early stopping and Optuna |
| Test | Last 15% | 379 | Final evaluation only |

**Samples per feature ratio**: 1765 / 23 = 76.7:1 (vs 80.2:1 in Attempt 2 with 22 features). Marginal reduction, still well above the rule-of-thumb minimum of 10:1 for tree models.

---

## 3. Model Architecture

### 3.1 Architecture: Single XGBoost Regressor

```
Input: 23-dimensional feature vector
  - 5 base features (1 level + 4 daily changes)
  - 18 submodel features (regime probs, z-scores, signals)
  |
  v
XGBoost Ensemble (gradient boosted trees)
  - Objective: reg:squarederror
  - n_estimators: Optuna-controlled [100, 1000]
  - Early stopping: patience=50 on validation RMSE
  - Regularization: Optuna-controlled (see Section 4)
  |
  v
Output: Single scalar (predicted next-day gold return %)
  |
  v
Post-processing (fixed, NOT Optuna-tuned):
  - Direction: sign(prediction)
  - Confidence threshold: np.percentile(|prediction|, 80) = top 20%
  - High-confidence: |prediction| > threshold
  - Trade signal: sign(prediction) for position, |prediction| for magnitude
```

### 3.2 Why Single XGBoost (NOT Ensemble, NOT Calibration)

**Attempt 3 lesson (ensemble)**: Adding a second model layer (blending GBMs) increased overfitting from 5.54pp to 25.96pp DA gap while degrading DA from 57.26% to 53.30%. Ensemble capacity was counterproductive.

**Attempt 4 lesson (calibration)**: Post-hoc logistic regression confidence model trained on 378 validation samples showed 22.6pp validation-test HCDA gap (65.45% -> 42.86%). The calibration model promoted 40% DA predictions into high-confidence and demoted 60.3% DA predictions. With only ~75 high-confidence test samples, any noise in the calibration layer is catastrophic.

**Conclusion**: The correct approach for Attempt 5 is improving the base model itself so that |prediction| naturally correlates with directional accuracy. Adding options_risk_regime_prob and re-optimizing HP is the minimum-risk path.

### 3.3 Sharpe Calculation (CLAUDE.md spec -- unchanged from Attempt 2)

```python
def compute_sharpe_trade_cost(predictions, actuals, cost_bps=5.0):
    """
    Compute Sharpe with cost on position changes only.
    Matches CLAUDE.md evaluator specification.
    """
    cost_pct = cost_bps / 100.0  # 5bps = 0.05%
    positions = np.sign(predictions)
    trades = np.abs(np.diff(positions, prepend=0))
    strategy_returns = positions * actuals
    net_returns = strategy_returns - trades * cost_pct
    if len(net_returns) < 2 or np.std(net_returns) == 0:
        return 0.0
    sharpe = (np.mean(net_returns) / np.std(net_returns)) * np.sqrt(252)
    return sharpe
```

### 3.4 Direction Accuracy Calculation (unchanged)

```python
def compute_direction_accuracy(predictions, actuals):
    """DA excluding zeros (CLAUDE.md spec: np.sign(0) = 0 problem)."""
    nonzero = (actuals != 0) & (predictions != 0)
    if nonzero.sum() == 0:
        return 0.0
    return (np.sign(predictions[nonzero]) == np.sign(actuals[nonzero])).mean() * 100
```

### 3.5 High-Confidence DA Calculation (unchanged -- fixed at 80th percentile)

```python
def compute_hc_da(predictions, actuals):
    """
    DA on top 20% by |prediction| (80th percentile threshold).
    Fixed threshold, NOT an Optuna parameter.
    """
    threshold = np.percentile(np.abs(predictions), 80)
    hc_mask = np.abs(predictions) > threshold
    if hc_mask.sum() == 0:
        return 0.0, 0.0
    coverage = hc_mask.sum() / len(predictions)
    hc_pred = predictions[hc_mask]
    hc_actual = actuals[hc_mask]
    nonzero = (hc_actual != 0) & (hc_pred != 0)
    if nonzero.sum() == 0:
        return 0.0, coverage
    da = (np.sign(hc_pred[nonzero]) == np.sign(hc_actual[nonzero])).mean() * 100
    return da, coverage
```

**Key difference from Attempt 2**: The threshold is NOT an Optuna parameter. In Attempt 2, confidence_threshold was an Optuna HP (suggest_float 0.01-0.10). This created a dependency where HCDA varied with the threshold setting rather than reflecting genuine model quality. In Attempt 5, the threshold is always the 80th percentile of |prediction|, ensuring HCDA reflects the model's natural magnitude-accuracy relationship.

---

## 4. Hyperparameter Specification

### 4.1 Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective | reg:squarederror | Standard MSE loss. Proven effective in Attempt 2. |
| early_stopping_rounds | 50 | Patience for validation loss convergence |
| eval_metric | rmse | Standard for reg:squarederror early stopping |
| tree_method | hist | Fast histogram-based algorithm |
| verbosity | 0 | Suppress output for Optuna |
| seed | 42 + trial.number | Reproducible, trial-specific randomization |

### 4.2 Optuna Search Space

| Parameter | Range | Scale | Type | Attempt 2 Range | Change Rationale |
|-----------|-------|-------|------|-----------------|-----------------|
| max_depth | [2, 5] | linear | int | [2, 4] | Widened to 5 to allow deeper trees that can capture regime interactions between options_risk_regime_prob and existing features. With 23 features and colsample up to 0.7, deeper trees can model richer interactions without relying on explicit interaction terms. |
| n_estimators | [100, 1000] | linear | int | [100, 800] | Widened to 1000 to allow more boosting rounds. With potentially slower learning rates and deeper trees, more rounds may be needed. Early stopping at patience=50 prevents overtraining regardless. |
| learning_rate | [0.001, 0.05] | log | float | [0.005, 0.03] | Widened in both directions. Lower bound 0.001 allows very conservative learning (may need >500 trees). Upper bound 0.05 allows faster convergence for shallow trees. |
| colsample_bytree | [0.2, 0.7] | linear | float | [0.3, 0.6] | Widened. Lower bound 0.2 samples ~5 of 23 features (strong decorrelation). Upper bound 0.7 samples ~16 features (weaker decorrelation but better feature utilization). This allows Optuna to explore configurations that give options_risk_regime_prob higher probability of being selected per tree. |
| subsample | [0.6, 0.95] | linear | float | [0.4, 0.7] | Shifted upward. Attempt 2 found 0.478. The aggressive [0.4, 0.7] range may have been overly conservative for 1765 training samples. Higher subsample preserves more information per tree. |
| min_child_weight | [1, 20] | linear | int | [10, 30] | Shifted downward. Attempt 2 found 14. With max_depth up to 5, min_child_weight needs to allow enough leaf splits. Range [1, 20] covers both conservative (20: leaves cover 1.1% of data) and aggressive (1: unrestricted). Regularization is provided by other parameters (reg_lambda, reg_alpha). |
| reg_alpha (L1) | [0, 10] | log-uniform-ish | float | [1.0, 10.0] | Lower bound reduced to 0 (no L1). Attempt 2 found 3.65. Allowing 0 lets Optuna decide if L1 is needed at all. Implementation: suggest_float('reg_alpha', 1e-8, 10.0, log=True) to approximate [0, 10] in log space. |
| reg_lambda (L2) | [0.1, 10] | log | float | [3.0, 20.0] | Shifted downward. Attempt 2 found 4.76. Upper bound reduced from 20 to 10 because Attempt 2's regularization was sufficient at 4.76 and overfitting was only 5.54pp. Allowing lower values (0.1) gives Optuna freedom if deeper trees need less weight shrinkage. |

**Removed from Attempt 2**:
- gamma: Removed from search space. In Attempt 2 it was [0.5, 3.0] but gamma's role (minimum split loss reduction) overlaps with min_child_weight and reg_lambda. Simplifying the search space from 10 to 8 HP improves Optuna efficiency for 100 trials.
- confidence_threshold: Removed. Fixed at 80th percentile (see Section 3.5).

**Total: 8 hyperparameters** (vs 10 in Attempt 2). Fewer HP with wider ranges = better Optuna coverage per HP.

### 4.3 Search Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_trials | 100 | Increased from 80 in Attempt 2. With 8 HP (vs 10) and wider ranges, 100 trials provides ~12.5 trials per HP (vs 8 in Attempt 2). Better coverage of the expanded space. |
| timeout | 7200 sec (2 hours) | XGBoost on 1765 rows with up to 1000 estimators: ~20-30 sec/trial. 100 trials = ~40-50 min. Generous 2-hour margin. |
| sampler | TPESampler(seed=42) | Standard for mixed continuous/integer spaces |
| pruner | None | XGBoost trials are fast; pruning overhead exceeds benefit |
| direction | maximize | Maximize composite objective |

### 4.4 Optuna Objective Function

```python
def optuna_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.7),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'seed': 42 + trial.number,
    }

    n_estimators = trial.suggest_int('n_estimators', 100, 1000)

    model = xgb.XGBRegressor(**params, n_estimators=n_estimators,
                              early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)

    # === Compute all 4 metrics on validation set ===
    val_mae = np.mean(np.abs(val_pred - y_val))
    val_da = compute_direction_accuracy(val_pred, y_val)
    val_sharpe = compute_sharpe_trade_cost(val_pred, y_val)
    val_hc_da, val_hc_coverage = compute_hc_da(val_pred, y_val)  # Fixed 80th pctile

    # Compute train DA for overfitting check
    train_pred = model.predict(X_train)
    train_da = compute_direction_accuracy(train_pred, y_train)
    da_gap = train_da - val_da

    # === Normalize to [0, 1] ===
    sharpe_norm = np.clip((val_sharpe + 3.0) / 6.0, 0.0, 1.0)   # [-3, +3] -> [0, 1]
    da_norm = np.clip((val_da - 40.0) / 30.0, 0.0, 1.0)          # [40%, 70%] -> [0, 1]
    mae_norm = np.clip((1.0 - val_mae) / 0.5, 0.0, 1.0)          # [0.5%, 1.0%] -> [0, 1]
    hc_da_norm = np.clip((val_hc_da - 40.0) / 30.0, 0.0, 1.0)    # [40%, 70%] -> [0, 1]

    # === Overfitting penalty ===
    overfit_penalty = max(0.0, (da_gap - 10.0) / 30.0)  # 0 if gap<=10pp, up to 1.0 if gap=40pp

    # === Weighted composite (ATTEMPT 5 WEIGHTS) ===
    objective = (
        0.40 * sharpe_norm +     # Reduced from 0.50 (Sharpe margin: +0.78)
        0.30 * da_norm +         # Unchanged (DA margin: +1.26pp, thin)
        0.10 * mae_norm +        # Unchanged (MAE margin: -0.0623%)
        0.20 * hc_da_norm        # Increased from 0.10 (HCDA is bottleneck)
    ) - 0.30 * overfit_penalty   # Same overfitting penalty as Attempt 2

    # === Log trial details ===
    trial.set_user_attr('val_mae', float(val_mae))
    trial.set_user_attr('val_da', float(val_da))
    trial.set_user_attr('val_hc_da', float(val_hc_da))
    trial.set_user_attr('val_hc_coverage', float(val_hc_coverage))
    trial.set_user_attr('val_sharpe', float(val_sharpe))
    trial.set_user_attr('train_da', float(train_da))
    trial.set_user_attr('da_gap_pp', float(da_gap))
    trial.set_user_attr('n_estimators_used',
                         int(model.best_iteration + 1) if hasattr(model, 'best_iteration')
                         and model.best_iteration is not None else n_estimators)

    return objective
```

**Weight rationale (Attempt 5 specific)**:

| Component | Weight | Attempt 2 Weight | Change | Rationale |
|-----------|--------|-----------------|--------|-----------|
| Sharpe | 40% | 50% | -10pp | Sharpe has 0.78 margin (1.58 vs 0.80 target). Can absorb reduction. |
| DA | 30% | 30% | 0 | DA has only 1.26pp margin (57.26% vs 56%). Must protect. |
| MAE | 10% | 10% | 0 | MAE has moderate margin (0.688% vs 0.75%). Stable. |
| HCDA | 20% | 10% | +10pp | HCDA is the only failing target (-4.74pp gap). Doubling weight signals Optuna to prioritize magnitude-accuracy alignment. |
| Overfit penalty | 30% | 30% | 0 | Same penalty structure. Train-test DA gap >10pp penalized. |

---

## 5. Training Configuration

### 5.1 Training Algorithm

```
1. DATA PREPARATION:
   a. Fetch raw data using yfinance and fredapi (same as Attempt 2)
   b. Construct base features (replicate Phase 1 logic)
   c. Compute daily changes for real_rate, dxy, yield_spread, inflation_exp
   d. Load 7 submodel output CSVs with proper date normalization
      - NEW: options_market.csv with utc=True date normalization
   e. Merge base features + submodel outputs + target on Date (left join from base)
   f. Apply NaN imputation (Section 2.5)
      - NEW: options_risk_regime_prob NaN -> 0.5
   g. Verify: 2522 rows, 23 features, 0 remaining NaN
   h. Split: train (70%), val (15%), test (15%) by time order

2. OPTUNA HPO (100 trials, 2-hour timeout):
   a. For each trial:
      - Sample 8 hyperparameters (reduced from 10 in Attempt 2)
      - Train XGBoost with reg:squarederror and early stopping
      - Compute val metrics: DA, Sharpe (trade cost), MAE, HC-DA (fixed 80th pctile)
      - Compute overfitting penalty
      - Return weighted composite: 40% Sharpe + 30% DA + 10% MAE + 20% HCDA
   b. Select best trial (highest composite objective)

3. FALLBACK EVALUATION:
   a. Train model with Attempt 2 best_params + 23 features (fallback config)
   b. Compare Optuna best vs fallback on validation metrics
   c. Select configuration with higher composite objective on validation
   d. Log which configuration was selected

4. FINAL MODEL TRAINING:
   a. Re-train selected configuration on (X_train, y_train)
   b. Early stop on (X_val, y_val) using RMSE
   c. Record best_iteration

5. EVALUATION ON ALL SPLITS:
   a. Predict on train, val, test
   b. Compute all 4 target metrics per split
   c. Compute HCDA at multiple thresholds (10%, 15%, 20%, 25%, 30%)
   d. Compute train-test DA gap
   e. Compute feature importance (all 23 features)
   f. Compute naive always-up DA and always-long Sharpe
   g. Compute quarterly DA breakdown
   h. Compare with Attempt 2 and Attempt 4 results

6. SAVE RESULTS:
   a. training_result.json
   b. model.json (XGBoost model)
   c. predictions.csv
   d. submodel_output.csv
```

### 5.2 Loss Function

- **Primary and only**: reg:squarederror (standard MSE)
- **No custom loss**: Eliminates risk of adversarial training (Attempt 1 root cause)
- **No secondary loss**: Direction accuracy and HCDA are measured in Optuna objective, not in the loss

### 5.3 Early Stopping

- **Metric**: RMSE on validation set
- **Patience**: 50 rounds
- **Maximum rounds**: Optuna-controlled (100-1000)

### 5.4 Fallback Configuration (Attempt 2 Best Params)

If Optuna's best trial underperforms Attempt 2 best_params on DA, Sharpe, or MAE, use the fallback:

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
    'gamma': 0.5,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'verbosity': 0,
    'seed': 42,
}
FALLBACK_N_ESTIMATORS = 247
```

**Fallback decision rule**: After Optuna completes, train both configurations on the same data split. If Optuna best has lower DA or Sharpe on validation than fallback, select fallback. Log which was selected in training_result.json.

---

## 6. Kaggle Execution Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | XGBoost on 1765 samples is CPU-fast. GPU overhead exceeds benefit. |
| Estimated execution time | 20-40 minutes | 100 trials * ~20-30 sec/trial + fallback eval + diagnostics |
| Estimated memory usage | 1.5 GB | 2522 rows * 23 features * 8 bytes = ~0.5 MB. XGBoost <50 MB. Optuna ~100 MB. |
| Required pip packages | [] | All pre-installed on Kaggle (xgboost, optuna, pandas, numpy, scikit-learn, yfinance, fredapi) |
| Internet required | true | For data fetching (yfinance, fredapi) |
| Kaggle Notebook ID | bigbigzabuton/gold-model-training | Unified notebook with FRED_API_KEY secret |
| Optuna timeout | 7200 sec (2 hours) | 100 trials * ~25 sec = ~42 min expected. Generous margin. |

---

## 7. Implementation Instructions

### 7.1 For builder_data

**No separate data preparation step needed.** All data loading is handled inside the self-contained Kaggle notebook. The options_market.csv submodel output is already committed and available.

Verification checklist (builder_data or datachecker should confirm):
- data/submodel_outputs/options_market.csv exists (2548 rows, 2 columns: Date, options_risk_regime_prob)
- Date column is timezone-aware, requires utc=True for parsing
- All 6 existing submodel CSVs unchanged from Attempt 2

### 7.2 For builder_model

**Task**: Generate `notebooks/meta_model_5/train.ipynb` (self-contained Kaggle Notebook)

**Critical implementation details**:

1. **Date normalization for options_market.csv** (NEW):
   ```python
   # options_market.csv has timezone-aware dates (same as technical.csv)
   opts = pd.read_csv('data/submodel_outputs/options_market.csv')
   opts['Date'] = pd.to_datetime(opts['Date'], utc=True).dt.strftime('%Y-%m-%d')
   # Then merge: merged = merged.merge(opts, on='Date', how='left')
   ```

2. **All other date normalizations** (unchanged from Attempt 2):
   ```python
   # technical.csv: timezone-aware
   tech['Date'] = pd.to_datetime(tech['date'], utc=True).dt.strftime('%Y-%m-%d')
   # yield_curve.csv: rename index
   yc = yc.rename(columns={'index': 'Date'})
   # inflation_expectation.csv: rename Unnamed: 0
   ie = ie.rename(columns={'Unnamed: 0': 'Date'})
   ```

3. **NaN imputation must include options_risk_regime_prob**:
   ```python
   # Regime probabilities -> 0.5 (includes NEW options_risk_regime_prob)
   regime_cols = [c for c in feature_cols if 'regime' in c.lower()
                  and ('prob' in c.lower() or 'probability' in c.lower())]
   df[regime_cols] = df[regime_cols].fillna(0.5)
   ```

4. **HCDA computation uses FIXED threshold (NOT Optuna-tuned)**:
   ```python
   # In the Optuna objective and final evaluation:
   threshold = np.percentile(np.abs(predictions), 80)  # Always 80th pctile
   # DO NOT use trial.suggest_float for confidence_threshold
   ```

5. **Optuna weights are 40/30/10/20** (NOT 50/30/10/10 from Attempt 2):
   ```python
   objective = (
       0.40 * sharpe_norm +   # was 0.50
       0.30 * da_norm +       # unchanged
       0.10 * mae_norm +      # unchanged
       0.20 * hc_da_norm      # was 0.10
   ) - 0.30 * overfit_penalty
   ```

6. **Fallback evaluation is mandatory**: After Optuna, train with Attempt 2 best_params on 23 features and compare. Select the better configuration.

7. **Diagnostic outputs** must include:
   - HCDA at multiple thresholds (10%, 15%, 20%, 25%, 30%) on test set
   - options_risk_regime_prob feature importance and rank
   - Comparison with Attempt 2 metrics (DA: 57.26%, HCDA: 55.26%, MAE: 0.688%, Sharpe: 1.58)
   - Comparison with Attempt 4 metrics (DA: 55.35%, HCDA: 42.86%, MAE: 0.687%, Sharpe: 1.63)
   - Which configuration was selected (Optuna best vs fallback)
   - Quarterly DA breakdown on test set
   - Prediction distribution stats (mean, std, % positive, min, max)
   - All Optuna trial summaries (top 10 trials by objective)
   - Train/val/test metrics for all 4 targets

8. **Output files** (saved to Kaggle output):
   - `training_result.json`: All metrics, params, feature importance, per-split results, comparison data
   - `model.json`: XGBoost model (model.save_model('model.json'))
   - `predictions.csv`: [date, split, prediction, actual, direction_correct, high_confidence]
   - `submodel_output.csv`: Final predictions aligned with dates

### 7.3 Feature List for builder_model Reference

The exact 23 feature column names:

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
    # Options market submodel (1) -- NEW IN ATTEMPT 5
    'options_risk_regime_prob',
]
assert len(FEATURE_COLUMNS) == 23
```

---

## 8. Risk Mitigation

### Risk 1: HCDA Does Not Improve (PRIMARY RISK)

**Scenario**: options_risk_regime_prob increases |prediction| on both correct AND incorrect predictions uniformly. HCDA remains at 55-57% despite the new feature.

**Probability**: 45-60%.

**Evidence for concern**: Gate 3 ablation showed MAE improved by -0.156% (magnitude accuracy) but Sharpe degraded by -0.141 and DA degraded by -0.24%. If the feature increases magnitude without directional discrimination, the top-20% set will contain proportionally the same mix of correct and incorrect predictions.

**Mitigation**: The Optuna weight rebalancing (HCDA 20% vs 10%) creates explicit optimization pressure for HCDA. Even if the feature itself does not directly improve HCDA, the weight change may shift HP toward configurations that naturally produce better magnitude-accuracy correlation. Additionally, the wider max_depth range [2, 5] allows deeper trees that can condition on options_risk_regime_prob within interaction branches (e.g., "if options_risk_regime_prob > 0.7 AND vix_regime_probability > 0.6, predict larger magnitude").

**Contingency**: If HCDA is 55-58% (no meaningful improvement):
- Accept 3/4 targets as the final meta-model result
- OR proceed to Attempt 6 with interaction terms (options * vix regime alignment)
- OR consider architecture change (lightweight neural network with learnable confidence)

### Risk 2: DA Regresses Below 56%

**Scenario**: Re-running Optuna with different weights and wider HP ranges finds a different optimum that trades DA for HCDA. Attempt 2's DA 57.26% had only 1.26pp margin.

**Probability**: 25-35%.

**Evidence**: Attempt 4 showed DA regression to 55.35% even with frozen base model HP. The calibration layer was responsible there, but re-optimization could produce similar regression through different HP.

**Mitigation**:
1. DA weight is preserved at 30% (unchanged from Attempt 2)
2. Fallback evaluation: if Optuna best has DA < 56% on validation but fallback has DA > 56%, select fallback
3. The 23rd feature (options_risk_regime_prob) has near-neutral DA impact (-0.24% in Gate 3, within noise)

**Contingency**: If DA < 56%: use fallback configuration (Attempt 2 HP + 23 features). If fallback also fails, the 23rd feature may be harming DA and should be removed.

### Risk 3: Sharpe Drops Below 0.8

**Scenario**: Reducing Sharpe weight from 50% to 40% causes Optuna to accept lower-Sharpe HP configurations.

**Probability**: 15-20%.

**Evidence**: Attempt 2 Sharpe was 1.5835 with 0.78 margin. A 40% weight should still maintain Sharpe > 1.0. The always-long Sharpe (trade cost) is 2.065, providing an upper bound reference.

**Mitigation**: Sharpe at 40% weight is still the largest single component. The risk is low unless HCDA optimization specifically conflicts with Sharpe (unlikely -- high-confidence correct predictions generate positive returns, so improving HCDA should support Sharpe).

### Risk 4: Overfitting from Deeper Trees

**Scenario**: max_depth=5 (widened from 4) with min_child_weight=1 (lowered from 10) creates complex trees that overfit.

**Probability**: 20-30%.

**Evidence**: Attempt 2 found max_depth=2, suggesting shallow trees generalize best for this data size. Allowing depth 5 may tempt Optuna toward configurations with >10pp DA gap.

**Mitigation**:
1. Overfitting penalty in objective: 30% penalty when DA gap > 10pp
2. reg_lambda and reg_alpha provide weight regularization even with deep trees
3. Early stopping at patience=50 prevents overtraining
4. If Optuna consistently selects depth=5, this is informative (the expanded search found a genuinely better depth)

### Risk 5: Kaggle Timeout

**Probability**: <5%.

**Evidence**: 100 trials * ~25 sec/trial = ~42 min. Well within 2-hour Optuna timeout and 9-hour Kaggle limit. Even with max_depth=5 and n_estimators=1000, early stopping will limit actual training rounds.

---

## 9. Expected Outcomes

| Metric | Attempt 2 | Attempt 4 | Attempt 5 Target | Attempt 5 Expected | Confidence |
|--------|-----------|-----------|------------------|-------------------|------------|
| DA | 57.26% | 55.35% | > 56% | 56.5-58% | Medium-High |
| HCDA | 55.26% | 42.86% | > 60% | 57-61% | Medium |
| MAE | 0.688% | 0.687% | < 0.75% | 0.64-0.70% | High |
| Sharpe | 1.583 | 1.628 | > 0.80 | 1.1-1.5 | High |
| Train-test gap | 5.54pp | 6.28pp | < 10pp | 5-8pp | High |
| Features | 22 | 22 | 23 | 23 | Fixed |

**Overall probability of 4/4 targets**: 35-50%.
**Probability of 3/4 targets (matching Attempt 2)**: 70-85%.
**Probability of regression below Attempt 2**: 10-20%.

### Success Scenarios

| Scenario | Probability | Description |
|----------|------------|-------------|
| Full success (4/4) | 35-50% | options_risk_regime_prob + weight rebalancing closes HCDA gap |
| Partial success (3/4, HCDA improved) | 25-35% | HCDA improves to 57-59% but does not reach 60% |
| Status quo (3/4, HCDA unchanged) | 15-20% | New feature has neutral HCDA impact; other metrics maintained |
| Regression (<3/4) | 10-20% | DA or Sharpe regresses; fallback should prevent this |

---

## 10. Success Criteria

### Primary (all must pass on test set)

| Metric | Target | Formula |
|--------|--------|---------|
| DA | > 56% | sign agreement, excluding zeros |
| HCDA | > 60% | DA on top 20% by |prediction| (80th percentile, fixed) |
| MAE | < 0.75% | mean(|prediction - actual|) |
| Sharpe | > 0.80 | annualized, after 5bps trade cost (CLAUDE.md spec) |

### Secondary Diagnostics

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Train-test DA gap | < 10pp | Overfitting control |
| Model DA > Naive always-up DA (56.73%) | True | Real predictive skill |
| options_risk_regime_prob feature importance | Top 10 of 23 | Validates feature contribution |
| HCDA at multiple thresholds | Report 10/15/20/25/30% | Diagnose confidence profile |
| Quarterly DA breakdown | No quarter < 45% | Regime stability |
| Prediction std > 0.05% | True | Not degenerate predictions |
| Overfit ratio (MAE test/train) | < 1.5 | Magnitude overfitting control |
| Selected config | Optuna or Fallback | Log which was used |

### Decision Rules After Evaluation

| Outcome | Action |
|---------|--------|
| All 4 targets met | Meta-model COMPLETE. Merge to main. |
| 3/4 targets (HCDA 57-59%) | Attempt 6: add interaction terms (options * vix regime alignment) |
| 3/4 targets (HCDA < 57%, same as Attempt 2) | Accept 3/4 as final result OR attempt 6 with rolling features |
| DA or Sharpe regresses (< 2/4 targets) | Revert to Attempt 2 as final meta-model |
| All 4 missed despite reasonable HCDA | Investigate: likely overfitting or data pipeline issue |

---

## 11. Comparison with Previous Attempts

| Aspect | Attempt 1 | Attempt 2 (Best) | Attempt 3 | Attempt 4 | Attempt 5 |
|--------|-----------|------------------|-----------|-----------|-----------|
| Architecture | XGBoost + custom loss | XGBoost + squarederror | XGBoost ensemble | XGBoost + calibration | XGBoost + squarederror |
| Features | 39 | 22 | 22 | 22 | **23** |
| Optuna trials | 50 | 80 | 80 | 200 (calibration) | **100** |
| Optuna weights | N/A | 50/30/10/10 | 50/30/10/10 | N/A (frozen) | **40/30/10/20** |
| HCDA approach | Optuna threshold | Optuna threshold | Ensemble | Calibration layer | **Fixed threshold + weight** |
| DA result | 54.1% | **57.26%** | 53.30% | 55.35% | TBD |
| HCDA result | 54.3% | 55.26% | **59.21%** | 42.86% | TBD |
| Sharpe result | 0.428 | **1.583** | 0.48 | 1.628 | TBD |
| Targets passed | 0/4 | **3/4** | 2/4 | 1/4 | TBD |

---

**End of Design Document**

**Architect**: architect (Opus)
**Date**: 2026-02-16
**Based on**: docs/research/meta_model_5.md (with fact-check corrections)
**Supersedes**: docs/design/meta_model_attempt_4.md
