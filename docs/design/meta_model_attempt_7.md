# Meta-Model Design Document: Attempt 7

## 0. Fact-Check Results

### 0.1 temporal_context.csv -- VERIFIED

| Check | Result | Detail |
|-------|--------|--------|
| File exists | PASS | `data/submodel_outputs/temporal_context.csv` (2461 rows, 2 columns) |
| Columns | PASS | `date`, `temporal_context_score` |
| Date format | PASS | `YYYY-MM-DD` (no timezone, same as vix.csv). No `utc=True` needed |
| Date column name | PASS | Lowercase `date` (same convention as vix.csv) |
| Value range | PASS | [0.0066, 0.9999] -- continuous 0-1 range as expected |
| NaN count | PASS | 0 NaN in source file |
| Date range | PASS | 2015-02-05 to 2025-02-12 (2461 rows) |
| Value distribution | NOTE | mean=0.287, median=0.076, std=0.364. Right-skewed. Most values near 0, some near 1 |

**Date handling**: Same pattern as vix.csv -- lowercase `date` column, no timezone awareness required. Simple `pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')` suffices.

### 0.2 Gate 3 Performance -- VERIFIED from completed.json

| Metric | Value | Threshold | Assessment |
|--------|-------|-----------|------------|
| DA delta | +0.584% | +0.5% | PASS (3/5 folds) |
| Sharpe delta | +0.1132 | +0.05 | PASS (2/5 folds) |
| MAE delta | -0.158 | -0.01 | PASS (15.8x threshold, 5/5 folds) |
| MI increase | +8.05% | +5% | PASS |
| VIF | 2.98 | <10 | PASS |
| Stability | 0.129 | <0.15 | PASS |
| Feature importance rank | #10/20 (5.17%) | -- | Moderate |

temporal_context is the second submodel (after inflation_expectation) to pass all 3 gates. All metrics are credible and consistent.

### 0.3 Attempt 6 Design Inheritance -- VERIFIED

Attempt 6 design document reviewed. The following components are inherited unchanged:
- XGBoost architecture with reg:squarederror
- Bootstrap variance-based confidence (5-model ensemble for HCDA)
- OLS output scaling (validation-derived, capped at [0.5, 10.0])
- Strengthened regularization bounds
- Optuna weights: 40/30/10/20 (Sharpe/DA/MAE/HCDA)
- Fallback to attempt 2 best params
- All metric functions

### 0.4 Feature Count Verification

| Category | Attempt 6 | Attempt 7 | Delta |
|----------|-----------|-----------|-------|
| Base features | 5 | 5 | 0 |
| VIX submodel | 3 | 3 | 0 |
| Technical submodel | 3 | 3 | 0 |
| Cross-asset submodel | 3 | 3 | 0 |
| Yield curve submodel | 2 | 2 | 0 |
| ETF flow submodel | 3 | 3 | 0 |
| Inflation expectation submodel | 3 | 3 | 0 |
| Options market submodel | 1 | 1 | 0 |
| **Temporal context submodel** | **0** | **1** | **+1** |
| **Total** | **23** | **24** | **+1** |

Samples-per-feature ratio (train): ~2131 * 0.70 / 24 = 62.2:1 (vs 64.9:1 in attempt 6). Marginal reduction, still adequate.

### 0.5 NaN Impact of Adding temporal_context -- ESTIMATED

temporal_context.csv starts 2015-02-05 while base_features starts approximately 2015-01-30. The 4-day gap (plus window_size=5 offset) means ~10-15 additional NaN rows at the start. After imputation with 0.5 (maximum uncertainty, consistent with regime_prob convention since temporal_context_score is a 0-1 continuous score), all rows remain usable.

### 0.6 Kaggle Dataset -- CONFIRMED UPDATED

Per user instruction, the Kaggle dataset (`bigbigzabuton/gold-prediction-complete`) has been updated with temporal_context.csv. The file will be available at `../input/gold-prediction-complete/temporal_context.csv`.

### 0.7 Summary

| Check | Verdict |
|-------|---------|
| temporal_context.csv file | PASS -- 2461 rows, clean dates, 0 NaN |
| Gate 3 results | PASS -- all 3 criteria met |
| Feature count (24) | PASS -- adequate samples-per-feature |
| Attempt 6 design inheritance | PASS -- all components reusable |
| Architecture change | NONE -- same XGBoost + bootstrap + OLS |
| Kaggle dataset updated | PASS -- temporal_context.csv added |

**Decision**: Proceed with minimal change -- add temporal_context_score as the 24th feature. No other modifications.

---

## 1. Overview

- **Purpose**: Incorporate the newly completed temporal_context submodel output into the meta-model. This is a pure feature expansion from 23 to 24 features, inheriting all other design choices from attempt 6.
- **Architecture**: Single XGBoost model with reg:squarederror + Bootstrap confidence (5 models) + OLS output scaling. Identical to attempt 6.
- **Key Change from Attempt 6**:
  1. **+1 feature**: `temporal_context_score` added to the feature set (from 23 to 24 features)
  2. **Data pipeline update**: Load temporal_context.csv with appropriate date normalization and NaN imputation
- **What is NOT changed**: XGBoost architecture, HP search space, Optuna weights (40/30/10/20), bootstrap ensemble approach, OLS scaling, metric functions, fallback mechanism, early stopping (100 rounds), all other submodel features.
- **Expected Effect**: DA +0.3-0.6pp (from temporal_context Gate 3 DA delta +0.584%), MAE improvement (Gate 3 MAE delta -0.158), Sharpe maintained or slightly improved. temporal_context passed all 3 gates, making it the strongest new feature addition available.

---

## 2. Data Specification

### 2.1 Input Data

All sources from attempt 6, plus temporal_context.csv.

| Source | Path (Kaggle) | Used Columns | Date Fix |
|--------|------|-------------|----------|
| Base features | API-fetched (yfinance, FRED) | 5 (transformed) | None |
| VIX submodel | ../input/gold-prediction-complete/vix.csv | 3 | Lowercase `date` |
| Technical submodel | ../input/gold-prediction-complete/technical.csv | 3 | utc=True |
| Cross-asset submodel | ../input/gold-prediction-complete/cross_asset.csv | 3 | None |
| Yield curve submodel | ../input/gold-prediction-complete/yield_curve.csv | 2 | Rename index |
| ETF flow submodel | ../input/gold-prediction-complete/etf_flow.csv | 3 | None |
| Inflation expectation | ../input/gold-prediction-complete/inflation_expectation.csv | 3 | Rename Unnamed:0 |
| Options market | ../input/gold-prediction-complete/options_market.csv | 1 | utc=True |
| **Temporal context** | **../input/gold-prediction-complete/temporal_context.csv** | **1** | **Lowercase `date`, no tz** |
| Target | API-fetched (yfinance GC=F) | 1 | None |

### 2.2 Feature Set (24 features)

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
    # Temporal context submodel (1) -- NEW in Attempt 7
    'temporal_context_score',
]
assert len(FEATURE_COLUMNS) == 24
```

### 2.3 Data Split

Unchanged from attempt 6: 70/15/15 time-series split, no shuffle.

### 2.4 temporal_context_score NaN Imputation

- **Imputation value**: 0.5 (maximum uncertainty)
- **Rationale**: temporal_context_score is a 0-1 continuous score output by a sigmoid activation. The value 0.5 represents the midpoint / maximum uncertainty, consistent with the regime_prob imputation convention used for all other 0-1 range submodel outputs (vix_regime_probability, tech_trend_regime_prob, xasset_regime_prob, etf_regime_prob, ie_regime_prob, options_risk_regime_prob).
- **Expected NaN count**: ~10-15 rows at the start of the dataset (temporal_context starts 2015-02-05 vs base_features ~2015-01-30).

---

## 3. Model Architecture

### 3.1 Architecture: Single XGBoost + Bootstrap Confidence + OLS Scaling

Identical to attempt 6. The only change is input dimensionality (23 -> 24).

```
Input: 24-dimensional feature vector (+1 from attempt 6)
  |
  v
XGBoost Ensemble (gradient boosted trees)
  - Objective: reg:squarederror
  - n_estimators: Optuna-controlled [100, 800]
  - Early stopping: patience=100 on validation RMSE
  - Regularization: STRENGTHENED (same as attempt 6)
  |
  v
Raw Output: Single scalar (predicted next-day gold return %)
  |
  v
POST-TRAINING STEP 1: OLS Output Scaling (same as attempt 6)
  - alpha_ols from validation set, capped [0.5, 10.0]
  |
  v
POST-TRAINING STEP 2: Bootstrap Ensemble Confidence (same as attempt 6)
  - 5 models with seeds [42, 43, 44, 45, 46]
  - Confidence = 1 / (1 + std_across_models)
  |
  v
Output Metrics: DA, HCDA (bootstrap + |pred|), MAE (raw + scaled), Sharpe
```

### 3.2 All Metric Functions

Identical to attempt 6. No changes to:
- `compute_direction_accuracy()`
- `compute_mae()`
- `compute_sharpe_trade_cost()`
- `compute_hcda()` (|prediction| method)
- `compute_hcda_bootstrap()` (bootstrap variance method)

---

## 4. Hyperparameter Specification

### 4.1 Fixed Parameters

Identical to attempt 6. No changes.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective | reg:squarederror | Standard MSE |
| early_stopping_rounds | 100 | Stronger early stopping |
| eval_metric | rmse | Standard |
| tree_method | hist | Fast |
| verbosity | 0 | Suppress |
| seed | 42 + trial.number | Reproducible |

### 4.2 Optuna Search Space

Identical to attempt 6. No changes.

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| max_depth | [2, 4] | int | Prevent outlier-memorization trees |
| n_estimators | [100, 800] | int | Reduced upper bound |
| learning_rate | [0.001, 0.05] | log | Unchanged |
| colsample_bytree | [0.2, 0.7] | linear | Unchanged |
| subsample | [0.4, 0.85] | linear | Lowered and widened |
| min_child_weight | [12, 25] | int | Raised lower bound |
| reg_lambda (L2) | [1.0, 15.0] | log | Raised floor 10x |
| reg_alpha (L1) | [0.5, 10.0] | log | Raised floor |

**Total: 8 hyperparameters** (unchanged from attempt 6).

### 4.3 Search Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_trials | 100 | Same as attempt 6 |
| timeout | 7200 sec | 2-hour margin |
| sampler | TPESampler(seed=42) | Reproducible |
| pruner | None | Not needed |
| direction | maximize | Higher composite is better |

### 4.4 Optuna Objective Function

Identical to attempt 6. Same HP ranges, same weights (40/30/10/20), same overfitting penalty. The only implicit change is that XGBoost now receives 24-dimensional input instead of 23.

---

## 5. Training Configuration

### 5.1 Training Algorithm

```
1. DATA PREPARATION:
   (identical to attempt 6, EXCEPT temporal_context addition)
   a. Fetch raw data using yfinance and fredapi
   b. Construct base features (5)
   c. Compute daily changes
   d. Load 7 submodel output CSVs (same as attempt 6)
>> e. Load temporal_context.csv (NEW)
      - date_col: 'date' (lowercase, no timezone)
      - columns: ['temporal_context_score']
      - No tz_aware handling needed
   f. Merge base + all submodel + target on Date
   g. Apply NaN imputation (temporal_context_score -> 0.5)
   h. Verify: 24 features, 0 remaining NaN
   i. Split: train (70%), val (15%), test (15%)

2. OPTUNA HPO (100 trials, 2-hour timeout):
   (identical to attempt 6 -- same HP ranges, same weights, same objective)

3. FALLBACK EVALUATION:
   (identical to attempt 6 -- attempt 2 best params + 24 features)

4. FINAL MODEL TRAINING:
   (identical to attempt 6)

5. POST-TRAINING STEP 1: OLS OUTPUT SCALING:
   (identical to attempt 6)

6. POST-TRAINING STEP 2: BOOTSTRAP ENSEMBLE CONFIDENCE:
   (identical to attempt 6 -- 5 models, seeds [42-46])

7. EVALUATION ON ALL SPLITS:
   (identical to attempt 6 -- DA, HCDA both methods, MAE both, Sharpe,
    feature importance for 24 features, quarterly breakdown, decile analysis)
>> a. Report temporal_context_score rank and importance
>> b. Compare with attempt 6 results

8. SAVE RESULTS:
   (same output files as attempt 6)
```

### 5.2 Loss Function

- reg:squarederror (unchanged)

### 5.3 Early Stopping

- Metric: RMSE on validation set
- Patience: 100 rounds
- Maximum rounds: Optuna-controlled (100-800)

### 5.4 Fallback Configuration

Same as attempt 6 -- attempt 2 best params with 24 features (was 23 in attempt 6). The model automatically adapts to the wider input.

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

| Setting | Value | Change from Att 6 | Rationale |
|---------|-------|-------------------|-----------|
| enable_gpu | true | unchanged | Consistent with attempt 6 |
| Estimated execution time | 25-45 minutes | unchanged | +1 feature has negligible impact |
| Estimated memory usage | 1.5 GB | unchanged | +1 feature adds ~2KB |
| Required pip packages | [] | -shap (removed) | SHAP not used in attempt 6/7 (replaced by bootstrap) |
| Internet required | true | unchanged | For data fetching |
| Kaggle Notebook ID | bigbigzabuton/gold-model-training-meta-model | unchanged | |
| dataset_sources | bigbigzabuton/gold-prediction-complete | unchanged | temporal_context.csv already added |
| Optuna timeout | 7200 sec | unchanged | |

---

## 7. Implementation Instructions

### 7.1 For builder_data

No separate data preparation step needed. The meta-model notebook is self-contained and fetches all data directly. temporal_context.csv is loaded from the Kaggle dataset.

### 7.2 For builder_model

**Task**: Generate `notebooks/meta_model_7/train.ipynb` (self-contained Kaggle Notebook)

**Base**: Copy attempt 6 notebook structure with the following modifications.

#### 7.2.1 FEATURE_COLUMNS Update (Cell 3)

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
    # Options market submodel (1)
    'options_risk_regime_prob',
    # Temporal context submodel (1) -- NEW in Attempt 7
    'temporal_context_score',
]

TARGET = 'gold_return_next'

assert len(FEATURE_COLUMNS) == 24, f"Expected 24 features, got {len(FEATURE_COLUMNS)}"
```

#### 7.2.2 Submodel Loading Update (Cell 5)

Add temporal_context to the `submodel_files` dictionary:

```python
submodel_files = {
    # ... (all existing 7 submodels unchanged) ...
    'options_market': {
        'path': '../input/gold-prediction-complete/options_market.csv',
        'columns': ['options_risk_regime_prob'],
        'date_col': 'Date',
        'tz_aware': True,
    },
    # NEW in Attempt 7
    'temporal_context': {
        'path': '../input/gold-prediction-complete/temporal_context.csv',
        'columns': ['temporal_context_score'],
        'date_col': 'date',       # lowercase 'date'
        'tz_aware': False,         # no timezone in dates
    },
}
```

#### 7.2.3 NaN Imputation Update (Cell 7)

Add temporal_context_score to the regime probability imputation group:

```python
# Regime probability / 0-1 score columns -> 0.5 (maximum uncertainty)
regime_cols = ['vix_regime_probability', 'tech_trend_regime_prob',
               'xasset_regime_prob', 'etf_regime_prob', 'ie_regime_prob',
               'options_risk_regime_prob',
               'temporal_context_score',  # NEW in Attempt 7
               ]
```

#### 7.2.4 Feature Count Assert Updates

All `assert len(...) == 23` must become `assert len(...) == 24`:
- Cell 3: Feature definition assertion
- Any verification prints: "23 features" -> "24 features"

#### 7.2.5 Diagnostic Output Updates

Update comparison baselines to include attempt 6 results:

```python
# Add attempt 6 comparison (values to be filled after attempt 6 evaluation)
# If attempt 6 results are not yet available, use attempt 5 as latest comparison
print("\nVs Attempt 5:")
print(f"  DA:     ... (Attempt 5: 56.77%)")
print(f"  HCDA:   ... (Attempt 5: 57.61%)")
print(f"  MAE:    ... (Attempt 5: 0.9520%)")
print(f"  Sharpe: ... (Attempt 5: 1.83)")
```

#### 7.2.6 training_result.json Updates

```python
training_result['feature'] = 'meta_model'
training_result['attempt'] = 7  # was 6
training_result['architecture'] = 'XGBoost reg:squarederror + Bootstrap confidence + OLS scaling'
training_result['model_config']['n_features'] = 24  # was 23
```

Report temporal_context_score rank and importance:

```python
# Find temporal_context_score rank
tc_rank = (feature_ranking.reset_index(drop=True).reset_index()
           .loc[feature_ranking['feature'] == 'temporal_context_score', 'index'].values[0] + 1)
tc_importance = feature_ranking.loc[feature_ranking['feature'] == 'temporal_context_score', 'importance'].values[0]

training_result['feature_importance']['temporal_context_score_rank'] = int(tc_rank)
training_result['feature_importance']['temporal_context_score_importance'] = float(tc_importance)
```

#### 7.2.7 Markdown Header Update (Cell 0)

```markdown
# Gold Meta-Model Training - Attempt 7

**Architecture:** Single XGBoost with reg:squarederror

**Key Changes from Attempt 6:**
1. **+1 feature**: temporal_context_score added (24 total features, was 23)
   - Temporal Context Transformer output (0-1 score)
   - Gate 3 PASS: DA +0.584%, Sharpe +0.113, MAE -0.158

**Inherited from Attempt 6:**
- Bootstrap variance-based confidence (5 models for HCDA)
- OLS output scaling (validation-derived, capped at 10x)
- Strengthened regularization (same HP ranges)
- Optuna weights: 40/30/10/20

**Design:** `docs/design/meta_model_attempt_7.md`
```

#### 7.2.8 kernel-metadata.json

```json
{
  "id": "bigbigzabuton/gold-model-training-meta-model",
  "title": "Gold Meta-Model Training - Attempt 7",
  "code_file": "train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["bigbigzabuton/gold-prediction-complete"],
  "competition_sources": [],
  "kernel_sources": []
}
```

### 7.3 Complete Feature List (24 features)

| # | Feature | Source | Type | Imputation |
|---|---------|--------|------|------------|
| 1 | real_rate_change | FRED DFII10 (diff) | Base | dropna |
| 2 | dxy_change | Yahoo DX-Y.NYB (diff) | Base | dropna |
| 3 | vix | FRED VIXCLS (level) | Base | dropna |
| 4 | yield_spread_change | FRED DGS10-DGS2 (diff) | Base | dropna |
| 5 | inflation_exp_change | FRED T10YIE (diff) | Base | dropna |
| 6 | vix_regime_probability | vix.csv | Submodel | 0.5 |
| 7 | vix_mean_reversion_z | vix.csv | Submodel | 0.0 |
| 8 | vix_persistence | vix.csv | Submodel | median |
| 9 | tech_trend_regime_prob | technical.csv | Submodel | 0.5 |
| 10 | tech_mean_reversion_z | technical.csv | Submodel | 0.0 |
| 11 | tech_volatility_regime | technical.csv | Submodel | median |
| 12 | xasset_regime_prob | cross_asset.csv | Submodel | 0.5 |
| 13 | xasset_recession_signal | cross_asset.csv | Submodel | 0.0 |
| 14 | xasset_divergence | cross_asset.csv | Submodel | 0.0 |
| 15 | yc_spread_velocity_z | yield_curve.csv | Submodel | 0.0 |
| 16 | yc_curvature_z | yield_curve.csv | Submodel | 0.0 |
| 17 | etf_regime_prob | etf_flow.csv | Submodel | 0.5 |
| 18 | etf_capital_intensity | etf_flow.csv | Submodel | 0.0 |
| 19 | etf_pv_divergence | etf_flow.csv | Submodel | 0.0 |
| 20 | ie_regime_prob | inflation_expectation.csv | Submodel | 0.5 |
| 21 | ie_anchoring_z | inflation_expectation.csv | Submodel | 0.0 |
| 22 | ie_gold_sensitivity_z | inflation_expectation.csv | Submodel | 0.0 |
| 23 | options_risk_regime_prob | options_market.csv | Submodel | 0.5 |
| 24 | **temporal_context_score** | **temporal_context.csv** | **Submodel** | **0.5** |

---

## 8. Risk Mitigation

### Risk 1: temporal_context Adds Noise Rather Than Signal (MODERATE)

**Scenario**: Despite passing Gate 3 as a standalone submodel addition, temporal_context_score becomes redundant with existing features when combined with all 10 submodel outputs. XGBoost assigns it low importance and overall metrics do not improve.

**Probability**: 25-35%.

**Evidence for concern**: temporal_context was evaluated in Gate 3 against a 22-feature XGBoost baseline (before options_market). In the 24-feature context (with options_market), the marginal value may differ. Additionally, feature importance rank was #10/20 (5.17%), which is moderate but not top-tier.

**Mitigation**:
1. XGBoost naturally handles redundant features via feature selection (colsample_bytree in [0.2, 0.7])
2. Fallback to attempt 2 best params provides safety net
3. If temporal_context_score ranks last in importance, it effectively becomes a no-op (minimal harm)

### Risk 2: Overfitting from Additional Feature (LOW)

**Scenario**: Adding the 24th feature degrades generalization. Samples-per-feature ratio drops from 64.9:1 to 62.2:1.

**Probability**: 10-15%.

**Evidence against**: The reduction is marginal (4%). Strong regularization (min_child_weight [12,25], reg_lambda [1.0,15.0]) prevents overfitting on 24 features. Attempt 5 worked with 23 features without issues.

**Mitigation**: Same strengthened regularization as attempt 6.

### Risk 3: Bootstrap HCDA Method Still Does Not Reach 60% (HIGH)

**Scenario**: Neither bootstrap variance nor |prediction| HCDA reaches 60% target, regardless of temporal_context addition.

**Probability**: 50-60%.

**Evidence**: Attempt 5 best HCDA was 57.61%. Attempt 6 results are pending. The 60% HCDA target has never been achieved across 5 previous attempts.

**Mitigation**: This is a structural challenge, not specific to attempt 7. If attempt 7 achieves 3/4 targets (DA, Sharpe, MAE or HCDA), it should be accepted as a strong final result.

### Risk 4: MAE Target Remains Infeasible (HIGH)

**Probability**: 85-90%.

**Evidence**: Attempt 2 achieved MAE 0.688% (only attempt to meet the 0.75% target) but with only 22 features and a test set that did not include the extreme 2025-2026 gold volatility. The expanded test set makes <0.75% structurally difficult.

**Mitigation**: Accept that MAE target is infeasible with current test set. Focus on DA, Sharpe, and HCDA.

---

## 9. Expected Outcomes

| Metric | Attempt 2 | Attempt 5 | Attempt 7 Expected | Confidence |
|--------|-----------|-----------|-------------------|------------|
| DA | 57.26% | 56.77% | 56.5-58.0% | Medium-High |
| HCDA (best method) | 55.26% | 57.61% | 57-61% | Medium |
| MAE (best of raw/scaled) | 0.688% | 0.952% | 0.90-0.95% | Low |
| Sharpe | 1.583 | 1.834 | 1.2-1.9 | High |
| Val DA | 53.85% | 49.23% | 50-55% | Medium |
| Train-test gap | 5.54pp | 7.35pp | 4-8pp | High |
| Targets passed | 3/4 | 2/4 | 2-3/4 | Medium |

**Probability of outcomes**:

| Outcome | Probability |
|---------|------------|
| 3/4 targets (DA + Sharpe + HCDA or MAE) | 25-35% |
| 2/4 targets (DA + Sharpe) | 45-55% |
| 4/4 targets | <5% |
| Regression (<2/4) | 10-15% |

---

## 10. Success Criteria

### Primary Targets (on test set)

| Metric | Target | Method |
|--------|--------|--------|
| DA | > 56% | sign agreement, excluding zeros |
| HCDA | > 60% | top 20% by BEST of (bootstrap confidence, \|prediction\|) |
| MAE | < 0.75% (stretch) or < 0.85% (acceptable) | BEST of (raw, OLS-scaled) predictions |
| Sharpe | > 0.80 | annualized, 5bps trade cost |

### Secondary Diagnostics

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Train-test DA gap | < 10pp | Overfitting control |
| Val DA | > 50% | Not below random |
| Bootstrap HCDA vs \|pred\| HCDA | Report both | Compare confidence methods |
| temporal_context_score importance rank | Report value | Validate feature contribution |
| temporal_context_score importance vs Gate 3 | Compare | Cross-validate with standalone evaluation |
| OLS alpha value | [0.5, 10.0] | Scaling reasonableness |
| Decile analysis (both methods) | Report all 10 | Verify ordering not inverted |

### Decision Rules After Evaluation

| Outcome | Action |
|---------|--------|
| 3/4 targets met (DA, Sharpe, HCDA or MAE) | Accept as final meta-model |
| DA >= 56% and Sharpe >= 0.80 and HCDA >= 59% | Near-success. Consider final. |
| Regression on DA or Sharpe vs attempt 5 | Revert to attempt 5 or attempt 2 |
| temporal_context_score ranked last/near-last | Feature adds no value. Revert to attempt 6 features |

---

## 11. Comparison with Previous Attempts

| Aspect | Att 2 (Best Overall) | Att 5 (Latest Eval) | Att 6 (Pending) | Att 7 (This) |
|--------|---------------------|---------------------|-----------------|--------------|
| Architecture | XGBoost | XGBoost | XGBoost | XGBoost |
| Features | 22 | 23 | 23 | **24** |
| New feature | -- | options_market | -- | **temporal_context** |
| HCDA method | \|pred\| | \|pred\| | Bootstrap + \|pred\| | Bootstrap + \|pred\| |
| MAE method | raw | raw | raw + OLS | raw + OLS |
| Optuna trials | 80 | 100 | 100 | 100 |
| Weights | 50/30/10/10 | 40/30/10/20 | 40/30/10/20 | 40/30/10/20 |
| max_depth | [2,4] | [2,5] | [2,4] | [2,4] |
| min_child_weight | [10,30] | [1,20] | [12,25] | [12,25] |
| reg_lambda | [3,20] | [0.1,10] | [1,15] | [1,15] |
| early_stopping | 50 | 50 | 100 | 100 |

---

## 12. Implementation Checklist for builder_model

1. Copy attempt 6 notebook structure (`notebooks/meta_model_6/train.ipynb`)
2. Update markdown header to "Attempt 7" (Cell 0)
3. Add `'temporal_context_score'` to FEATURE_COLUMNS (Cell 3)
4. Change assertion from 23 to 24 features (Cell 3)
5. Add temporal_context entry to `submodel_files` dict (Cell 5): date_col='date', tz_aware=False
6. Add `'temporal_context_score'` to regime_cols imputation list (Cell 7)
7. Update all "23 features" text to "24 features"
8. Update `training_result['attempt']` to 7 (Cell 28)
9. Add temporal_context_score rank/importance to training_result (Cell 28)
10. Update kernel-metadata.json title to "Attempt 7"
11. Run `scripts/validate_notebook.py` to verify notebook

**Important**: No changes needed to:
- HP ranges (same as attempt 6)
- Metric functions (same as attempt 6)
- Optuna objective (same as attempt 6)
- Bootstrap ensemble logic (same as attempt 6)
- OLS scaling logic (same as attempt 6)
- Fallback params (same as attempt 6)
- Decile analysis (same as attempt 6, 24 features handled automatically)

---

**End of Design Document**

**Architect**: architect (Opus)
**Date**: 2026-02-16
**Based on**: Attempt 6 design + temporal_context submodel completion (Gate 3 PASS all criteria)
**Supersedes**: docs/design/meta_model_attempt_6.md
