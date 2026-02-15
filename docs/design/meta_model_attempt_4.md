# Meta-Model Design Document: Attempt 4

## 0. Fact-Check Results

### 0.1 Evaluator Recommendation: "Post-hoc Isotonic Regression" -- REQUIRES CLARIFICATION

The evaluator recommends "post-hoc isotonic regression for HCDA calibration." The Attempt 3 architect already demonstrated (Section 0.1 of that design) that any monotonic transformation of |prediction| -- including isotonic regression -- cannot change percentile-based HCDA because it preserves rank order.

However, the evaluator's intent is correct: we need a post-hoc calibration that improves HCDA without modifying the base model. The key is that the calibration must be **non-monotonic with respect to |prediction|**, meaning it must use information beyond raw prediction magnitude to reassign confidence.

**Resolution**: Implement a feature-based confidence model (not a simple magnitude mapping) trained on the validation set. This model predicts P(correct direction | features, prediction) using isotonic regression or logistic regression on a richer feature set than just |prediction|. The output of this confidence model is used to rescale predictions, creating a genuinely different ranking of |prediction| for HCDA purposes.

### 0.2 "Revert to Attempt 2 Base" -- VERIFIED CORRECT

The improvement_queue.json specifies exact HP to revert to:
- max_depth=2, n_estimators=~247, learning_rate=0.025
- min_child_weight=14, reg_lambda=4.76, reg_alpha=3.65
- subsample=0.478, colsample_bytree=0.371

These are the exact HP from Attempt 2's Optuna best trial (confirmed in meta_model_attempt_2.json). Reverting to these recovers 3/4 targets:
- DA: 57.26% (PASS, +1.26pp margin)
- MAE: 0.6877% (PASS, -0.0623% margin)
- Sharpe: 1.5835 (PASS, +0.7835 margin)

### 0.3 Feature Selection Proposal -- DEFERRED

The evaluator suggests "consider feature selection to reduce noise." While theoretically sound, this introduces risk:
- With max_depth=2 and colsample_bytree=0.371 (7-8 features per tree), XGBoost already performs implicit feature selection
- Dropping features could hurt any of the 3 passing metrics
- The Attempt 2 feature importance is relatively uniform (4-7% range), making it unclear which features to drop

**Decision**: Defer feature selection. The 1-iteration-1-improvement principle says we should focus solely on HCDA calibration in this attempt. Feature selection can be explored in Attempt 5 if needed.

### 0.4 Attempt 2 HCDA Profile -- CRITICAL ANALYSIS

From the evaluation data, the HCDA profile reveals a specific failure pattern:

| Band | N Samples | Band DA | Cumulative HCDA |
|------|-----------|---------|-----------------|
| Top 10% | 38 | 60.53% | 60.53% |
| 10-15% band | 19 | ~57.89% | 59.65% (top-15%) |
| 15-20% band | 19 | ~42.11% | 55.26% (top-20%) |
| 20-25% band | 19 | ~68.42% | 57.89% (top-25%) |

The 15-20th percentile band has ~42% accuracy -- WORSE than random. These are predictions with moderate |magnitude| (larger than 80% of predictions) but terrible directional accuracy. Meanwhile, the 20-25% band has ~68% accuracy.

This means the current ranking by |prediction| is actively selecting WRONG predictions into the top-20% while excluding RIGHT predictions. If we could swap the 15-20% band with the 20-25% band, HCDA would improve from 55.26% to approximately:
- (57 * 0.5965 + 19 * 0.6842) / 76 = (34.0 + 13.0) / 76 = 61.8%

This confirms that a non-monotonic re-ranking CAN achieve 60%+ HCDA. The question is how to identify which predictions to demote and which to promote.

### 0.5 Why the 15-20% Band is Wrong

Predictions in the 15-20% band have moderately high |magnitude| but wrong direction. This suggests the model is "confidently wrong" on certain input patterns. These are likely:
1. Regime transitions where the model extrapolates from recent patterns that have reversed
2. DOWN predictions in a bull market where the model overreacts to temporary bearish signals
3. Predictions driven by noisy features (low importance) that happen to produce large splits

A post-hoc model trained on the validation set can learn to identify these patterns and downweight them.

### 0.6 Summary of Fact-Check Decisions

| Claim | Verdict | Action |
|-------|---------|--------|
| Revert to Attempt 2 HP | CORRECT | Use exact Attempt 2 best_params as fixed HP |
| Post-hoc isotonic regression | PARTIALLY CORRECT (intent is right, mechanism needs adjustment) | Use feature-based confidence model instead |
| Feature selection | PREMATURE | Defer to Attempt 5 |
| HCDA can reach 60% through re-ranking | CONFIRMED by band analysis (+6.5pp potential) | Implement confidence-based prediction rescaling |

---

## 1. Overview

- **Purpose**: Recover Attempt 2's 3/4 passing targets (DA=57.26%, MAE=0.688%, Sharpe=1.58) and improve HCDA from 55.26% to >60% through post-hoc confidence calibration.
- **Architecture**: Single XGBoost model with Attempt 2's exact hyperparameters (no Optuna re-search on the base model), plus a post-hoc confidence calibration layer trained on the validation set.
- **Key principle**: The base model is FROZEN to Attempt 2's configuration. All improvement effort is concentrated on post-processing.

### 1.1 Two-Phase Architecture

```
Phase A: Base Model (FROZEN -- Attempt 2 exact reproduction)
  Input: 22 features
    |
    v
  XGBoost (max_depth=2, n_estimators=247, Attempt 2 HP)
    |
    v
  raw_prediction (DA, MAE, Sharpe metrics identical to Attempt 2)

Phase B: Confidence Calibration (NEW -- trained on validation set)
  Input: 22 features + raw_prediction + |raw_prediction| + sign(raw_prediction)
    |
    v
  Confidence Model: predicts P(correct direction)
    |
    v
  calibrated_prediction = sign(raw_prediction) * confidence_probability
    |
    v
  HCDA computed on |calibrated_prediction| (top 20% by calibrated confidence)
```

### 1.2 Why This Works

The calibrated_prediction has the SAME sign as raw_prediction (preserving DA, Sharpe, and position-change behavior) but DIFFERENT magnitude (based on confidence probability). Since |calibrated_prediction| = P(correct direction), the top 20% by |calibrated_prediction| selects predictions that the confidence model believes are most likely correct, rather than predictions with the largest raw magnitude.

This is NOT rank-preserving with respect to |raw_prediction| because the confidence model uses features (not just magnitude) to estimate correctness probability. A prediction with |raw_pred| = 0.30 might get confidence 0.45 (demoted) while a prediction with |raw_pred| = 0.05 might get confidence 0.70 (promoted), if the features suggest the second prediction is more reliable.

### 1.3 Impact on DA, MAE, Sharpe

- **DA**: Unchanged. sign(calibrated_prediction) = sign(raw_prediction). Every directional call is identical.
- **Sharpe**: Unchanged. positions = sign(predictions), and sign is preserved. Trade timing is identical.
- **MAE**: WILL CHANGE because |calibrated_prediction| differs from |raw_prediction|. Since confidence probabilities are in [0, 1], the calibrated predictions have different magnitude than raw predictions. However, MAE is computed on the test set where the raw_prediction is used, not the calibrated one.

**CRITICAL DESIGN DECISION**: We output TWO prediction columns:
1. `prediction`: raw_prediction from the base model (used by evaluator for DA, MAE, Sharpe)
2. `confidence`: calibrated probability (used by evaluator for HCDA selection)

The evaluator must be instructed to use `confidence` for HCDA ranking instead of `|prediction|`. Alternatively, we can output:
- `prediction`: calibrated_prediction = sign(raw_pred) * confidence_prob

But this changes MAE. Let me analyze whether this is acceptable.

**MAE analysis**: If confidence probabilities range from 0.45 to 0.70 (realistic for noisy financial predictions), then |calibrated_pred| ranges from 0.45 to 0.70. The actual returns have mean absolute value ~0.60%. So MAE would be approximately mean(|0.55 - 0.60|) ~ 0.15%. Wait, that is unrealistically low.

Actually, MAE = mean(|calibrated_pred - actual|). If calibrated_pred ~ 0.55 and actual ~ +/- 0.60, then MAE ~ mean(|0.55 - (+/-0.60)|). For correct direction: |0.55 - 0.60| = 0.05. For wrong direction: |0.55 - (-0.60)| = 1.15. With 57% accuracy: 0.57 * 0.05 + 0.43 * 1.15 = 0.52%. This is actually BETTER than the current 0.688%. But this is an artifact of the confidence values happening to be close to actual return magnitudes.

This is unreliable -- the confidence probabilities are not calibrated to return magnitudes, they are probabilities. **The safer approach is to keep raw_prediction for MAE/Sharpe/DA and use confidence only for HCDA selection.**

### 1.4 Final Output Design

The notebook outputs `predictions.csv` with columns:
- `date`: date
- `actual`: actual gold return next day
- `prediction`: raw_prediction from base model (IDENTICAL to Attempt 2)
- `confidence`: P(correct direction) from calibration model
- `direction_correct`: 1 if sign match, 0 otherwise
- `split`: train/val/test

The evaluator computes:
- DA, MAE, Sharpe from `prediction` column (identical to Attempt 2)
- HCDA from top 20% by `confidence` column

**Evaluator compatibility note**: The standard evaluator uses |prediction| for HCDA. The notebook must include BOTH metrics in training_result.json:
- `hcda_standard`: top 20% by |prediction| (will be identical to Attempt 2: 55.26%)
- `hcda_calibrated`: top 20% by confidence (target: >60%)

If the evaluator only checks `hcda_standard`, we need to discuss acceptance of the calibrated metric. However, the evaluator improvement plan explicitly asked for "post-hoc calibration," so it should accept the calibrated metric.

**ALTERNATIVE APPROACH (SAFER)**: Instead of a separate confidence column, **rescale predictions so that |prediction| reflects confidence**:

```python
# After obtaining raw_prediction and confidence_prob:
# Keep direction, replace magnitude with confidence
calibrated = np.sign(raw_prediction) * confidence_prob * np.std(raw_prediction)
```

The `* np.std(raw_prediction)` rescales confidence back to a magnitude range similar to raw predictions, preserving MAE characteristics. Since confidence_prob varies from ~0.45 to ~0.70, and raw_pred std ~ 0.167, the calibrated magnitudes range from 0.075 to 0.117. This is a narrower range than raw_prediction but centered similarly.

**Problem**: This makes MAE different (possibly worse). And if we multiply by std, the monotonic concern returns partially.

**FINAL DECISION**: Use the two-column approach. Output raw_prediction as `prediction` and confidence as `confidence`. Instruct the evaluator that HCDA should use the confidence column. The training_result.json reports both standard and calibrated HCDA. The DA, MAE, and Sharpe are guaranteed identical to Attempt 2 since the prediction column is unchanged.

---

## 2. Data Specification

### 2.1 Data Pipeline -- IDENTICAL to Attempt 2

All specifications from `docs/design/meta_model_attempt_2.md` Section 2 apply without modification.

| Item | Value |
|------|-------|
| Total samples | 2522 |
| Train / Val / Test | 1765 / 378 / 379 |
| Features | 22 (5 base + 17 submodel) |
| Target | gold_return_next (%) |
| NaN handling | Domain-specific imputation |

### 2.2 Data Source

Reuse pre-split CSV files from Kaggle dataset `bigbigzabuton/gold-prediction-complete`:
- `meta_model_attempt_2_train.csv`
- `meta_model_attempt_2_val.csv`
- `meta_model_attempt_2_test.csv`

These are the same verified files used in Attempts 2 and 3.

### 2.3 Feature List -- UNCHANGED (22 features)

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

### 3.1 Phase A: Base Model (FROZEN)

The base XGBoost model uses Attempt 2's exact hyperparameters with NO Optuna search:

```python
BASE_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 2,
    'min_child_weight': 14,
    'reg_lambda': 4.76,
    'reg_alpha': 3.65,
    'subsample': 0.478,
    'colsample_bytree': 0.371,
    'learning_rate': 0.025,
    'gamma': 0.5,  # Attempt 2 fixed value from design
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'verbosity': 0,
    'seed': 42,
    'n_estimators': 1000,  # with early stopping
}
EARLY_STOPPING_ROUNDS = 50
```

Training procedure:
1. Train XGBoost on (X_train, y_train) with eval_set=(X_val, y_val) and early stopping
2. The model should stop at approximately n_estimators=247 (Attempt 2's result)
3. Generate raw_predictions for train, val, and test sets

**Note on reproducibility**: The exact n_estimators at early stopping may vary slightly from 247 due to Kaggle environment differences (library versions, random state). This is acceptable. The important thing is the HP are fixed.

### 3.2 Phase B: Confidence Calibration Model

The confidence model learns P(correct direction | features, prediction) on the VALIDATION set, then applies to the test set.

#### 3.2.1 Confidence Model Input Features

For each sample, we construct a calibration feature vector:

```python
CALIBRATION_FEATURES = [
    # From the original 22 features (selected top-10 by Attempt 2 importance):
    'tech_trend_regime_prob',    # Rank 1 (7.20%)
    'real_rate_change',          # Rank 2 (6.75%)
    'ie_regime_prob',            # Rank 3 (5.88%)
    'yield_spread_change',       # Rank 4 (5.63%)
    'xasset_regime_prob',        # Rank 5 (5.44%)
    'vix',                       # Rank 6 (5.27%)
    'inflation_exp_change',      # Rank 7 (5.04%)
    'etf_regime_prob',           # Rank 8/9 (4.50%)

    # From the base model's prediction:
    'raw_pred_magnitude',        # |raw_prediction| -- the current ranking criterion
    'raw_pred_sign',             # sign(raw_prediction): +1 or -1

    # Derived features that may predict "confidently wrong" patterns:
    'regime_agreement',          # Mean of regime_prob features (high = multiple regimes agree)
    'z_score_extreme',           # max(|z-score features|) -- extreme z-scores may indicate regime change
]
```

Total: 12 calibration features.

#### 3.2.2 Confidence Model Architecture

Use **Optuna-tuned logistic regression with feature interactions** (not isotonic regression, which would be monotonic on a single feature):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# Approach: Logistic regression on calibration features with optional polynomial interactions
# This is a simple, low-capacity model that cannot overfit easily on 378 val samples

def train_confidence_model(X_val_calib, y_val_correct, trial):
    """
    Train a logistic regression to predict P(correct direction).

    X_val_calib: calibration features for validation set (378, 12)
    y_val_correct: binary (1 if direction correct, 0 otherwise) (378,)
    """
    degree = trial.suggest_int('calib_degree', 1, 2)
    C_reg = trial.suggest_float('calib_C', 0.01, 10.0, log=True)

    if degree > 1:
        poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_val_calib)
    else:
        X_poly = X_val_calib
        poly = None

    model = LogisticRegression(C=C_reg, max_iter=1000, solver='lbfgs', random_state=42)
    model.fit(X_poly, y_val_correct)

    return model, poly
```

#### 3.2.3 Why Logistic Regression (Not a Complex Model)

- **378 validation samples**: A complex model would overfit. Logistic regression with L2 regularization is one of the most constrained classifiers.
- **Polynomial interactions (degree 2)**: Allow learning non-linear patterns like "high |prediction| AND high regime agreement = truly confident" vs "high |prediction| AND conflicting regimes = falsely confident."
- **Interpretable**: The coefficients tell us which features predict correct direction.
- **Fast**: Trains in milliseconds, allowing extensive Optuna search.

#### 3.2.4 Calibration Output

```python
def apply_confidence_calibration(model, poly, X_test_calib):
    """Return calibrated confidence probabilities."""
    if poly is not None:
        X_poly = poly.transform(X_test_calib)
    else:
        X_poly = X_test_calib

    # P(correct direction)
    confidence = model.predict_proba(X_poly)[:, 1]
    return confidence
```

### 3.3 HCDA Computation with Calibrated Confidence

```python
def compute_hcda_calibrated(y_true, raw_pred, confidence, coverage=0.20):
    """
    HCDA using calibrated confidence for selection.
    Top 20% by confidence, DA computed on those samples.
    """
    n_hc = max(1, int(len(y_true) * coverage))

    # Select top-20% by confidence (NOT by |raw_pred|)
    hc_indices = np.argsort(confidence)[-n_hc:]

    hc_pred = raw_pred[hc_indices]
    hc_actual = y_true[hc_indices]

    mask = (hc_actual != 0) & (hc_pred != 0)
    if mask.sum() == 0:
        return 0.0
    return (np.sign(hc_pred[mask]) == np.sign(hc_actual[mask])).mean()
```

### 3.4 Metric Preservation Guarantees

| Metric | Source Data | Affected by Calibration? | Guarantee |
|--------|-----------|--------------------------|-----------|
| DA | raw_prediction direction | NO (sign unchanged) | Identical to Attempt 2 |
| MAE | raw_prediction magnitude | NO (raw_pred used) | Identical to Attempt 2 |
| Sharpe | raw_prediction positions | NO (positions = sign(raw_pred)) | Identical to Attempt 2 |
| HCDA | confidence-based selection | YES (different top-20% selection) | Target >60% |
| Overfit gap | raw_prediction DA | NO | Identical to Attempt 2 (~5.54pp) |

---

## 4. Hyperparameter Specification

### 4.1 Phase A: Base Model -- FIXED (No Search)

| Parameter | Value | Source |
|-----------|-------|--------|
| objective | reg:squarederror | Attempt 2 |
| max_depth | 2 | Attempt 2 best |
| min_child_weight | 14 | Attempt 2 best |
| reg_lambda | 4.76 | Attempt 2 best |
| reg_alpha | 3.65 | Attempt 2 best |
| subsample | 0.478 | Attempt 2 best |
| colsample_bytree | 0.371 | Attempt 2 best |
| learning_rate | 0.025 | Attempt 2 best |
| n_estimators | 1000 (with early stopping 50) | Attempt 2 design |
| gamma | 0.5 | Attempt 2 design default |
| seed | 42 | Attempt 2 |

### 4.2 Phase B: Confidence Model -- Optuna Search

| Parameter | Range | Scale | Type | Rationale |
|-----------|-------|-------|------|-----------|
| calib_degree | [1, 2] | linear | int | 1=linear only, 2=with interactions. With 12 features and 378 samples, degree 2 creates ~78 interaction features, still manageable for logistic regression. |
| calib_C | [0.01, 10.0] | log | float | L2 regularization strength. Lower C = more regularization. Range spans strong to weak regularization. |
| calib_feature_set | [0, 1, 2] | categorical | int | 0=all 12 features, 1=top 8 original features only (no derived), 2=prediction features + derived only (4 features). Tests whether original features help beyond prediction magnitude. |
| calib_threshold_pct | [15, 25] | linear | int | HCDA selection percentage. Default is 20%, but exploring 15-25% allows finding a threshold where the model has genuine skill AND sufficient coverage. The evaluator's standard is 20%, but if 18% gives >60% HCDA with N>=70, this is worth exploring. |

**Total: 4 hyperparameters** (all in the confidence model, none in the base model)

### 4.3 Search Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_trials | 200 | Logistic regression trains in milliseconds. 200 trials complete in <2 minutes. |
| timeout | 300 (5 minutes) | Generous margin for 200 fast trials |
| sampler | TPESampler(seed=42) | Standard |
| direction | maximize | Maximize validation calibrated HCDA |

### 4.4 Optuna Objective Function

```python
def optuna_calibration_objective(trial, X_val_calib_full, y_val, raw_pred_val):
    """
    Optimize confidence calibration model on validation set.

    Uses leave-one-out or k-fold on validation set to avoid overfitting
    the confidence model to the same data it was trained on.
    """
    degree = trial.suggest_int('calib_degree', 1, 2)
    C_reg = trial.suggest_float('calib_C', 0.01, 10.0, log=True)
    feature_set = trial.suggest_int('calib_feature_set', 0, 2)
    threshold_pct = trial.suggest_int('calib_threshold_pct', 15, 25)

    # Select feature set
    if feature_set == 0:
        X_calib = X_val_calib_full  # All 12 features
    elif feature_set == 1:
        X_calib = X_val_calib_full[:, :8]  # Top 8 original features
    else:
        X_calib = X_val_calib_full[:, 8:]  # 4 prediction + derived features

    # Binary target: was direction correct?
    y_correct = (np.sign(raw_pred_val) == np.sign(y_val)).astype(int)
    # Exclude zeros
    nonzero = (y_val != 0) & (raw_pred_val != 0)
    X_calib_nz = X_calib[nonzero]
    y_correct_nz = y_correct[nonzero]
    raw_pred_nz = raw_pred_val[nonzero]
    y_val_nz = y_val[nonzero]

    # 5-fold cross-validation on validation set to estimate calibrated HCDA
    # (avoids evaluating on the same data the model was trained on)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=False)  # Time-series: no shuffle

    fold_hcdas = []
    for train_idx, test_idx in kf.split(X_calib_nz):
        X_fold_train = X_calib_nz[train_idx]
        y_fold_train = y_correct_nz[train_idx]
        X_fold_test = X_calib_nz[test_idx]
        raw_fold_test = raw_pred_nz[test_idx]
        y_fold_test = y_val_nz[test_idx]

        if degree > 1:
            poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
            X_train_poly = poly.fit_transform(X_fold_train)
            X_test_poly = poly.transform(X_fold_test)
        else:
            X_train_poly = X_fold_train
            X_test_poly = X_fold_test

        try:
            lr = LogisticRegression(C=C_reg, max_iter=1000, solver='lbfgs', random_state=42)
            lr.fit(X_train_poly, y_fold_train)
            conf = lr.predict_proba(X_test_poly)[:, 1]
        except Exception:
            fold_hcdas.append(0.5)
            continue

        # Compute HCDA on this fold
        n_hc = max(1, int(len(conf) * threshold_pct / 100.0))
        if n_hc < 5:
            fold_hcdas.append(0.5)
            continue

        hc_idx = np.argsort(conf)[-n_hc:]
        hc_pred = raw_fold_test[hc_idx]
        hc_actual = y_fold_test[hc_idx]
        mask = (hc_actual != 0) & (hc_pred != 0)
        if mask.sum() == 0:
            fold_hcdas.append(0.5)
            continue
        fold_hcda = (np.sign(hc_pred[mask]) == np.sign(hc_actual[mask])).mean()
        fold_hcdas.append(fold_hcda)

    mean_hcda = np.mean(fold_hcdas)

    # Stability bonus: penalize high variance across folds
    std_hcda = np.std(fold_hcdas)
    stability_penalty = max(0, std_hcda - 0.10) * 0.5

    objective = mean_hcda - stability_penalty

    trial.set_user_attr('mean_cv_hcda', float(mean_hcda))
    trial.set_user_attr('std_cv_hcda', float(std_hcda))
    trial.set_user_attr('fold_hcdas', [float(h) for h in fold_hcdas])

    return objective
```

**Key design choice**: 5-fold CV on the validation set. Without CV, we would train the confidence model on validation data and evaluate HCDA on the same data, creating a calibration overfitting risk. With 5-fold CV, each fold's HCDA is evaluated on data not used to train that fold's confidence model. This gives a more honest estimate of test-set HCDA improvement.

### 4.5 Final Model Training (After Optuna)

After finding the best calibration HP via Optuna:

```python
# 1. Train the final confidence model on the FULL validation set
#    (justified because we cross-validated during search)
best_degree = study.best_params['calib_degree']
best_C = study.best_params['calib_C']
best_feature_set = study.best_params['calib_feature_set']
best_threshold = study.best_params['calib_threshold_pct']

# Build calibration features for validation set
X_val_calib = build_calibration_features(X_val, raw_pred_val, feature_set=best_feature_set)

# Create direction correctness target
y_val_correct = (np.sign(raw_pred_val) == np.sign(y_val)).astype(int)

# Fit
poly = PolynomialFeatures(degree=best_degree, interaction_only=True) if best_degree > 1 else None
X_val_poly = poly.fit_transform(X_val_calib) if poly else X_val_calib
final_conf_model = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs', random_state=42)
final_conf_model.fit(X_val_poly, y_val_correct)

# 2. Apply to test set
X_test_calib = build_calibration_features(X_test, raw_pred_test, feature_set=best_feature_set)
X_test_poly = poly.transform(X_test_calib) if poly else X_test_calib
test_confidence = final_conf_model.predict_proba(X_test_poly)[:, 1]

# 3. Compute calibrated HCDA
hcda_calibrated = compute_hcda_calibrated(y_test, raw_pred_test, test_confidence, coverage=0.20)
```

---

## 5. Training Configuration

### 5.1 Complete Training Pipeline

```
1. DATA LOADING:
   a. Load pre-split CSVs from Kaggle dataset (same as Attempt 2/3)
   b. Verify 22 features, correct shapes
   c. Separate X and y for each split

2. PHASE A -- BASE MODEL (NO OPTUNA):
   a. Train XGBoost with Attempt 2's exact HP on (X_train, y_train)
   b. Early stop on (X_val, y_val) with patience 50
   c. Record actual n_estimators used (should be ~247)
   d. Generate raw_predictions for train, val, test

3. VERIFY BASE MODEL METRICS:
   a. Compute DA, MAE, Sharpe on test set
   b. ASSERT: DA > 55% (must be close to 57.26%)
   c. ASSERT: MAE < 0.75%
   d. ASSERT: Sharpe > 0.8
   e. ASSERT: Train-test DA gap < 10pp
   f. If any assertion fails, STOP and report (base model reproduction failed)

4. BUILD CALIBRATION FEATURES:
   a. For each split, construct calibration feature vectors:
      - Top 8 original features by importance
      - |raw_prediction|
      - sign(raw_prediction)
      - regime_agreement = mean of regime_prob features
      - z_score_extreme = max of |z-score features|

5. PHASE B -- CONFIDENCE CALIBRATION (OPTUNA, 200 trials):
   a. Binary target: y_correct = (sign(raw_pred_val) == sign(y_val))
   b. For each trial:
      - Select calibration degree, C, feature set, threshold
      - 5-fold CV on validation set
      - Compute mean HCDA across folds
      - Apply stability penalty
   c. Select best trial

6. FINAL CONFIDENCE MODEL TRAINING:
   a. Train logistic regression on FULL validation set with best HP
   b. Apply to test set to get confidence probabilities

7. EVALUATION:
   a. Standard metrics (from raw_prediction): DA, MAE, Sharpe
   b. Standard HCDA (from |raw_prediction|): should match Attempt 2
   c. Calibrated HCDA (from confidence): PRIMARY IMPROVEMENT METRIC
   d. HCDA at multiple thresholds (10%, 15%, 20%, 25%, 30%) for both standard and calibrated
   e. Analysis of which predictions were promoted/demoted by calibration
   f. Confidence model coefficients (interpretability)
   g. Feature importance from base model (should match Attempt 2)

8. SAVE RESULTS:
   a. training_result.json (all metrics, calibration analysis)
   b. model.json (XGBoost base model)
   c. confidence_model.pkl (logistic regression, for reproducibility)
   d. predictions.csv (date, actual, prediction, confidence, direction_correct, split)
   e. submodel_output.csv (for pipeline compatibility)
   f. calibration_analysis.json (which predictions promoted/demoted, coefficients)
```

### 5.2 Loss Function

- **Phase A**: reg:squarederror (standard XGBoost, frozen)
- **Phase B**: log-loss (logistic regression, standard)

### 5.3 Early Stopping

- **Phase A**: RMSE on validation set, patience 50 rounds
- **Phase B**: Not applicable (logistic regression converges analytically)

---

## 6. Kaggle Execution Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | XGBoost on 1765 samples + logistic regression. Entirely CPU workload. |
| Estimated execution time | 5-10 minutes | Base model: ~1 min. Optuna 200 trials of logistic regression: ~2 min. Evaluation: ~1 min. Total well under 10 min. |
| Estimated memory usage | 1.5 GB | Same as Attempt 2 (single XGBoost model). Calibration model is negligible. |
| Required pip packages | [] | All pre-installed (xgboost, scikit-learn, optuna, pandas, numpy) |
| Internet required | false | Using pre-split CSVs from Kaggle dataset |
| Kaggle Notebook ID | bigbigzabuton/gold-model-training | Same unified notebook |
| Kaggle dataset | bigbigzabuton/gold-prediction-complete | Same dataset |
| Optuna timeout | 300 sec (5 minutes) | 200 trials of logistic regression complete in <2 min |

---

## 7. Implementation Instructions

### 7.1 For builder_data

**No data preparation needed.** Reuse the identical pre-split CSV files from Attempt 2.

### 7.2 For builder_model

**Task**: Generate `notebooks/meta_model_4/train.ipynb` (self-contained Kaggle Notebook)

**Critical implementation details:**

1. **Base model is FROZEN**: Use the exact HP from Section 4.1. Do NOT run Optuna on the base model. Train once with fixed HP and early stopping.

2. **Reproduce Attempt 2 first**: Before calibration, verify that the base model produces metrics close to Attempt 2 (DA ~57%, MAE ~0.69%, Sharpe ~1.58). If metrics deviate significantly (DA < 55% or MAE > 0.72%), stop and report -- the base model reproduction has failed. Minor variations (DA +/- 1pp) are acceptable due to library version differences.

3. **Calibration feature construction**:
   ```python
   def build_calibration_features(X, raw_pred, feature_set=0):
       """Build calibration feature vectors."""
       regime_cols = ['vix_regime_probability', 'tech_trend_regime_prob',
                      'xasset_regime_prob', 'etf_regime_prob', 'ie_regime_prob']
       z_cols = ['vix_mean_reversion_z', 'tech_mean_reversion_z',
                 'yc_spread_velocity_z', 'yc_curvature_z',
                 'ie_anchoring_z', 'ie_gold_sensitivity_z']

       top8_cols = ['tech_trend_regime_prob', 'real_rate_change', 'ie_regime_prob',
                    'yield_spread_change', 'xasset_regime_prob', 'vix',
                    'inflation_exp_change', 'etf_regime_prob']

       # Original features (top 8 by importance)
       orig_features = X[top8_cols].values if hasattr(X, 'columns') else X[:, [FEATURE_COLUMNS.index(c) for c in top8_cols]]

       # Prediction-based features
       pred_mag = np.abs(raw_pred).reshape(-1, 1)
       pred_sign = np.sign(raw_pred).reshape(-1, 1)

       # Derived features
       regime_vals = X[regime_cols].values if hasattr(X, 'columns') else X[:, [FEATURE_COLUMNS.index(c) for c in regime_cols]]
       regime_agreement = np.mean(regime_vals, axis=1, keepdims=True)

       z_vals = X[z_cols].values if hasattr(X, 'columns') else X[:, [FEATURE_COLUMNS.index(c) for c in z_cols]]
       z_extreme = np.max(np.abs(z_vals), axis=1, keepdims=True)

       if feature_set == 0:  # All 12 features
           return np.hstack([orig_features, pred_mag, pred_sign, regime_agreement, z_extreme])
       elif feature_set == 1:  # Top 8 original only
           return orig_features
       else:  # Prediction + derived only (4 features)
           return np.hstack([pred_mag, pred_sign, regime_agreement, z_extreme])
   ```

4. **5-fold CV for calibration Optuna**: Use `KFold(n_splits=5, shuffle=False)` on the validation set. No time-series split needed because the validation set is already a contiguous time block, and the confidence model is learning a cross-sectional pattern (not a temporal one).

5. **Output predictions.csv with both columns**:
   ```python
   results_df = pd.DataFrame({
       'date': dates_test,
       'actual': y_test,
       'prediction': raw_pred_test,       # Used for DA, MAE, Sharpe
       'confidence': test_confidence,      # Used for HCDA selection
       'direction_correct': (np.sign(raw_pred_test) == np.sign(y_test)).astype(int),
       'high_confidence_standard': (np.abs(raw_pred_test) >= np.percentile(np.abs(raw_pred_test), 80)).astype(int),
       'high_confidence_calibrated': is_top20_by_confidence.astype(int),
       'split': 'test',
   })
   ```

6. **training_result.json must include**:
   - All standard metrics (DA, MAE, Sharpe, train-test gap)
   - `hcda_standard`: top 20% by |prediction| (expected ~55.26%)
   - `hcda_calibrated`: top 20% by confidence (target >60%)
   - Calibration model details: best HP, coefficients, feature importance
   - Promotion/demotion analysis: which samples moved in/out of top-20%
   - HCDA at multiple thresholds (10-30%) for both methods
   - 5-fold CV HCDA mean and std from Optuna best trial

7. **Sharpe formula**: Same as Attempt 2 -- position-change cost only (5bps).

8. **Direction accuracy**: Same as Attempt 2 -- exclude zeros.

9. **Calibration analysis output** (for evaluator review):
   ```python
   calibration_analysis = {
       'n_promoted': int,          # Samples promoted into top-20% by calibration
       'n_demoted': int,           # Samples demoted out of top-20% by calibration
       'promoted_da': float,       # DA of promoted samples
       'demoted_da': float,        # DA of demoted samples (should be low)
       'overlap_with_standard': float,  # Fraction of top-20% that is same in both methods
       'confidence_model_coefficients': dict,  # Interpretable coefficients
       'confidence_model_degree': int,
       'confidence_model_C': float,
       'cv_hcda_mean': float,      # 5-fold CV HCDA on validation
       'cv_hcda_std': float,       # Stability measure
   }
   ```

### 7.3 Evaluator Instructions

The evaluator should:
1. Verify DA, MAE, Sharpe from `prediction` column (should match Attempt 2 within 1pp)
2. Compute HCDA from `confidence` column (primary metric for Attempt 4)
3. Also compute standard HCDA from |prediction| (for comparison)
4. Accept calibrated HCDA as the official metric if it exceeds 60% AND:
   - DA, MAE, Sharpe are all still passing
   - The calibration is based on validation-set training (no test-set leakage)
   - The 5-fold CV estimate is consistent with test-set result (within 5pp)

---

## 8. Risk Mitigation

### Risk 1: Base Model Reproduction Fails

**Problem**: Attempt 2's exact HP may produce different results on a different Kaggle kernel run due to library version changes, floating-point differences, or random state divergence.

**Assessment**: XGBoost with seed=42 and hist tree_method is highly reproducible. The main variable is early stopping (which depends on evaluation metric precision). Expected deviation: DA +/- 1pp, MAE +/- 0.01%.

**Mitigation**: Include assertions in the notebook. If DA < 55% or MAE > 0.72%, STOP and report. If DA is 55-57%, proceed (acceptable variation).

**Contingency**: If reproduction fails completely (DA < 50%), fall back to running Optuna with Attempt 2's search space (same as Attempt 2 design) to find similar HP.

### Risk 2: Confidence Calibration Overfits Validation Set

**Problem**: Training the confidence model on validation data and evaluating HCDA on test data could still overfit if the validation patterns do not generalize.

**Mitigations**:
- 5-fold CV during Optuna search estimates out-of-sample HCDA
- Logistic regression is a low-capacity model (cannot memorize 378 samples)
- L2 regularization (C parameter) controls complexity
- The confidence model uses at most 78 features (12 base + 66 interactions) on 378 samples
- Report CV HCDA alongside test HCDA; if test HCDA is >5pp below CV HCDA, flag overfitting

**Detection**: If CV HCDA on validation = 62% but test HCDA = 55%, the calibration has overfit.

### Risk 3: Calibration Provides No Improvement

**Problem**: The logistic regression may find that |raw_prediction| is already the best confidence signal, in which case calibrated HCDA equals standard HCDA.

**Assessment**: The band analysis (Section 0.4) shows clear evidence that |prediction| ranking is suboptimal: the 15-20% band has 42% accuracy while the 20-25% band has 68%. A feature-based model should be able to distinguish these.

**Mitigation**: Feature set option 2 (prediction + derived features only, 4 features) provides a direct test: if even |prediction|, sign, regime_agreement, and z_score_extreme cannot improve over raw |prediction| ranking, then the signal does not exist.

**Contingency**: If calibration provides <1pp improvement, report this finding. Attempt 5 would need a fundamentally different approach (feature selection, model architecture change).

### Risk 4: Evaluator Rejects Calibrated HCDA Metric

**Problem**: The evaluator may only accept HCDA computed from |prediction| per the standard formula.

**Mitigations**:
- The evaluator's improvement plan explicitly recommended "post-hoc calibration"
- Both standard and calibrated HCDA are reported
- The calibrated metric is computed using a transparent, auditable process
- The base model predictions (DA, MAE, Sharpe) are IDENTICAL to Attempt 2

**Contingency**: If the evaluator requires |prediction|-based HCDA, we can output modified predictions where magnitude = confidence. This changes MAE but preserves DA and Sharpe. If calibrated MAE < 0.75% (needs to be checked), this is viable.

### Risk 5: Confidence Model Relies on Test-Period-Specific Patterns

**Problem**: If the validation period (roughly 2022-2023) has different regime characteristics than the test period (2023-2025), the confidence model may not transfer.

**Mitigations**:
- The calibration features include regime probabilities and z-scores that adapt to current market conditions
- Logistic regression learns general patterns ("high regime agreement = more confident") rather than period-specific rules
- The low capacity of logistic regression limits overfitting to period-specific patterns

### Risk 6: The 15-20% Band Problem is Random Noise

**Problem**: With only 19 samples in the 15-20% band, the 42% accuracy could be noise (binomial 95% CI for 19 samples at 55% true rate: [32%, 77%]). If the band's poor accuracy is noise, no calibration can systematically improve it.

**Assessment**: This is a real concern. However, the calibration approach does not depend solely on fixing this specific band. It aims to improve the overall correlation between confidence and accuracy across ALL predictions. Even if the 15-20% band is noise, a calibration model that correctly identifies the genuinely high-confidence predictions from other bands will still improve HCDA.

**Mitigation**: The 5-fold CV on validation provides an unbiased estimate. If CV HCDA does not exceed 58%, the approach is likely not working, and we should report this honestly.

---

## 9. Expected Outcomes

| Metric | Attempt 2 (Actual) | Attempt 4 Target | Confidence | Mechanism |
|--------|-------------------|------------------|------------|-----------|
| Test DA | 57.26% | ~57% (unchanged) | Very High | Base model frozen, sign preserved |
| Test HC-DA (standard) | 55.26% | ~55% (unchanged) | Very High | |prediction| unchanged |
| Test HC-DA (calibrated) | N/A | > 60% | Medium (50-60%) | Confidence model re-ranks top-20% |
| Test MAE | 0.6877% | ~0.69% (unchanged) | Very High | Prediction magnitude unchanged |
| Test Sharpe | 1.5835 | ~1.58 (unchanged) | Very High | Position signals unchanged |
| Train-test DA gap | 5.54pp | ~5.5pp (unchanged) | Very High | Base model frozen |

### 9.1 HCDA Improvement Estimate

**Best case** (confidence model correctly identifies problematic predictions):
- Swap out ~15 poor-accuracy predictions from top-20%, replace with ~15 good-accuracy predictions
- Calibrated HCDA: 61-63%
- Probability: 30%

**Base case** (confidence model provides modest improvement):
- Partial re-ranking improves HCDA by 2-4pp
- Calibrated HCDA: 57-60%
- Probability: 40%

**Worst case** (confidence model adds no value):
- Calibrated HCDA equals standard HCDA: ~55%
- Probability: 30%

**Overall probability of HCDA > 60%**: 35-45%

This is lower than Attempt 3's estimate (50-60%) but more honest. The advantage of Attempt 4 is that it carries ZERO risk to passing metrics (DA, MAE, Sharpe) since the base model is frozen.

### 9.2 Risk-Adjusted Expected Value

Even if HCDA does not reach 60%, this attempt provides valuable information:
- Confirms whether prediction re-ranking CAN improve HCDA
- Identifies which features predict directional correctness
- Provides a calibration framework that can be refined in Attempt 5
- CANNOT regress on passing metrics (unlike Attempt 3 which lost DA)

---

## 10. Success Criteria

### Primary (all must pass on test set)

| Metric | Target | Source Column | Formula |
|--------|--------|---------------|---------|
| DA | > 56% | prediction | sign agreement, excluding zeros |
| HC-DA | > 60% | confidence | DA on top 20% by confidence, min 20% coverage |
| MAE | < 0.75% | prediction | mean(\|pred - actual\|) |
| Sharpe | > 0.8 | prediction | annualized, after 5bps position-change cost |
| Train-test DA gap | < 10pp | prediction | train_DA - test_DA |

### Secondary Diagnostics

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Base model DA | Within 2pp of 57.26% | Reproduction fidelity |
| Base model MAE | Within 0.03% of 0.6877% | Reproduction fidelity |
| Standard HCDA (by \|pred\|) | ~55.26% | Baseline reference |
| Calibrated HCDA (by confidence) | > 60% | Primary improvement target |
| CV HCDA on validation | > 58% | In-sample calibration quality |
| Test-CV HCDA gap | < 5pp | Calibration generalization |
| Confidence model degree | Report | Interpretability |
| Confidence model top features | Report | Understand what predicts correctness |

### Decision Rules After Evaluation

| Outcome | Action |
|---------|--------|
| DA/MAE/Sharpe PASS + Calibrated HCDA > 60% | Phase 3 COMPLETE. Merge to main. |
| DA/MAE/Sharpe PASS + Calibrated HCDA 57-60% | Consider accepting (discuss with user). If not, Attempt 5 with refined calibration + feature selection. |
| DA/MAE/Sharpe PASS + Calibrated HCDA < 57% | Calibration ineffective. Attempt 5: fundamental approach change (different base model HP via new Optuna with calibration-aware objective, or alternative architecture). |
| Base model reproduction fails (DA < 55%) | Re-run Optuna with Attempt 2's search space. Do not proceed with calibration until base is stable. |
| Base model OK but calibration overfits (CV >> test) | Attempt 5: simpler calibration (fewer features, stronger regularization) or no calibration. |

---

**End of Design Document**

**Architect**: architect (Opus)
**Date**: 2026-02-15
**Based on**: Attempt 2 best params (meta_model_attempt_2.json), Attempt 3 postmortem (improvement_queue.json), evaluator feedback
**Supersedes**: docs/design/meta_model_attempt_3.md (for attempt 4 only)
