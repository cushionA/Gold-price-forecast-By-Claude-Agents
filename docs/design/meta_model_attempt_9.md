# Meta-Model Design Document: Attempt 9

## 0. Fact-Check Results

### 0.1 Evaluator Improvement Plan -- FACT-CHECKED

| Recommendation | Verdict | Detail |
|----------------|---------|--------|
| Revert to single XGBoost (attempt 7 arch) | CORRECT | Attempt 8 stacking collapsed. Single XGBoost is proven. |
| Asymmetric/directional loss function | PARTIALLY VALID | Custom loss works in XGBoost 3.1.2+ (verified locally). API signature is `(y_true, y_pred)` not `(y_pred, DMatrix)`. Kaggle XGBoost >= 2.0 also uses this signature. However, directional loss gradient is discontinuous at y_pred=0, creating training instability risk. |
| Continuous interaction features (not binary) | VALID | Binary regime features failed because thresholds created 0% activation (vix_persistence > 0.7 was never true). Continuous products are always nonzero. |
| Multi-objective Optuna | NOT RECOMMENDED | Attempt 7 already used single-objective with composite weights. Multi-objective adds selection complexity without clear benefit. Adjusting weights within single-objective is simpler and safer. |

### 0.2 Directional Loss Function -- RISK ASSESSMENT

| Aspect | Assessment |
|--------|------------|
| XGBoost custom objective API (v2.0+) | `def custom_obj(y_true, y_pred)` -- verified locally on v3.1.2 |
| XGBoost custom objective API (v1.x) | `def custom_obj(y_pred, dtrain)` -- incompatible, must handle at runtime |
| Gradient discontinuity at y_pred=0 | MODERATE RISK: tree-based models are less sensitive than NNs, but leaf weight computation uses gradients |
| Softened directional loss (sigmoid-based) | VALID alternative that eliminates discontinuity |
| Training stability with wrong-direction 2x penalty | UNCERTAIN: may cause model to overfit to avoiding negative predictions entirely, collapsing to always-positive (same failure mode as attempt 8) |
| Requirement for eval_metric | CRITICAL: custom objective requires explicit `eval_metric='rmse'` since the custom loss cannot be used as eval metric |

**Decision**: Do NOT use a custom loss function as the primary objective. Instead:
1. Use standard `reg:squarederror` (proven stable, attempt 7 architecture)
2. Train a SECOND model with directional loss as an Optuna trial variant
3. Compare both on validation and select the better one

This "dual-track" approach eliminates the risk of directional loss collapse while allowing it to contribute if beneficial.

### 0.3 Continuous Interaction Features -- VERIFIED

| Feature | Components | Correlation with inputs | Economic rationale |
|---------|------------|----------------------|-------------------|
| `yc_curvature_x_recession` | yc_curvature_z * xasset_recession_signal | Moderate (both are submodel outputs) | Yield curve distortions amplified during recession risk |
| `real_rate_x_temporal` | real_rate_change * temporal_context_score | Low (different sources) | Rate changes matter more when temporal patterns are strong |
| `dxy_x_vix_persistence` | dxy_change * vix_persistence | Low (different dynamics) | Dollar moves amplified during sustained high volatility |

VIF risk: Interaction features typically have moderate VIF (3-7) with their constituent features. With only 3 interactions, total features = 27, samples-per-feature = 55:1 (adequate for max_depth [2,4]).

### 0.4 Attempt 7 Prediction Decomposition -- CRITICAL ANALYSIS

Estimated confusion matrix from attempt 7 (DA=60.04%, 87.3% positive predictions):

```
                 Predicted +    Predicted -    Total
Actual +              243           26          269 (58.7%)
Actual -              157           32          189 (41.3%)
Total                 400           58          458
```

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Positive prediction accuracy | 60.8% (243/400) | Above random |
| Negative prediction accuracy | 55.2% (32/58) | Marginally above random |
| Net gain from negative predictions | +6 (32 correct - 26 wrong) | Only 1.31pp above naive |
| False positive rate | 83.1% (157/189) | Model predicts positive for 83% of actual negative days |

**Key insight**: Attempt 7's advantage over naive comes entirely from 58 negative predictions that are 55.2% accurate. To improve DA, the model needs either:
- (A) More negative predictions while maintaining >50% accuracy on them
- (B) Better accuracy on existing negative predictions (55.2% -> 60%+)
- (C) Both

The directional loss targets this directly by penalizing wrong-direction more heavily.

### 0.5 XGBoost Version Compatibility

| Environment | XGBoost Version | Custom Objective Signature |
|-------------|-----------------|---------------------------|
| Local | 3.1.2 | `(y_true, y_pred)` |
| Kaggle (2025-2026) | 2.0+ | `(y_true, y_pred)` |

Safe approach: check version at runtime and use correct signature. Both v2.0+ and v3.x use `(y_true, y_pred)`.

### 0.6 Summary

| Check | Verdict |
|-------|---------|
| Revert to single XGBoost | APPROVED -- attempt 7 architecture |
| Custom directional loss | APPROVED as secondary track only (with MSE as primary) |
| Continuous interaction features (3) | APPROVED -- low risk, 27 total features |
| Optuna weight adjustment | APPROVED -- shift weight toward DA |
| Stacking / ensemble meta-learner | REJECTED -- proven failure in attempt 8 |
| Binary regime features | REJECTED -- proven failure in attempt 8 |
| Prediction calibration | REJECTED -- proven failure in attempts 3-4 |
| Multi-objective Optuna | REJECTED -- unnecessary complexity |

---

## 1. Overview

- **Purpose**: Improve DA beyond attempt 7 (60.04%) while maintaining HCDA (64.13%) and Sharpe (2.46). Conservative approach that avoids the stacking/regime feature mistakes of attempt 8.
- **Architecture**: Single XGBoost (identical to attempt 7) + 3 continuous interaction features + dual-track loss function comparison (MSE vs directional) + increased DA weight in Optuna.
- **Key Changes from Attempt 7** (3 targeted modifications):
  1. **+3 continuous interaction features** (24 -> 27 features): Products of top-importance features, always nonzero
  2. **Dual-track loss comparison**: Train with both MSE and directional loss, pick better on validation
  3. **Optuna weight shift**: DA weight 30% -> 40%, Sharpe weight 40% -> 30% (Sharpe is 3.1x target, over-optimized)
- **What is NOT changed**: Bootstrap confidence (5 models), OLS output scaling, data pipeline, time-series split, fallback mechanism (attempt 2 params), early stopping (100 rounds), all metric functions.
- **Strategic Approach**: Option (C) -- Conservative. Attempt 7 already passes 3/4 targets. We must not regress. All 3 changes are low-risk and additive. The fallback mechanism guarantees no worse than attempt 7's HP range.

### 1.1 Why NOT Focus on MAE

The evaluator from attempt 7 declared MAE (0.9429% vs 0.75% target) structurally infeasible:
- Zero-prediction MAE would be ~0.96% on this test set
- Model prediction std (0.023) is 60x smaller than actual return std (~1.4%)
- Amplifying predictions to reduce MAE would destroy Sharpe (2.46)
- No attempt with the expanded 2025-2026 test set has achieved MAE < 0.75%

Attempting to fix MAE carries >60% probability of regressing DA/HCDA/Sharpe (evaluator's own estimate). With only 2 attempts remaining, this is not a rational allocation of budget.

### 1.2 Why NOT Focus on Improving Already-Passing Metrics

Attempt 7 already exceeds targets on DA (+4.04pp), HCDA (+4.13pp), and Sharpe (3.1x). Further improvements are subject to diminishing returns. The priority is to demonstrate that attempt 7's performance is robust and potentially slightly improvable, not to chase marginal gains that risk regression.

---

## 2. Data Specification

### 2.1 Input Data

All sources identical to attempt 7 (24 base + submodel features), plus 3 new continuous interaction features computed during preprocessing.

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

### 2.3 Continuous Interaction Features (3 new features)

```python
INTERACTION_COLUMNS = [
    'yc_curvature_x_recession',    # yc_curvature_z * xasset_recession_signal
    'real_rate_x_temporal',         # real_rate_change * temporal_context_score
    'dxy_x_vix_persistence',       # dxy_change * vix_persistence
]
assert len(INTERACTION_COLUMNS) == 3
```

**Feature generation logic**:

```python
def generate_interaction_features(df):
    """Generate continuous interaction features. All products, no binary thresholds."""
    # Top 1 x Top 2: yield curve distortion during recession risk
    df['yc_curvature_x_recession'] = df['yc_curvature_z'] * df['xasset_recession_signal']

    # Top 4 x Top 3: rate changes amplified by temporal pattern strength
    df['real_rate_x_temporal'] = df['real_rate_change'] * df['temporal_context_score']

    # DXY x VIX: dollar moves amplified by sustained high volatility
    df['dxy_x_vix_persistence'] = df['dxy_change'] * df['vix_persistence']

    return df
```

**Why these 3 interactions** (and not 6 or 8):

| Feature | Top-K x Top-K | Economic Story | Non-zero % |
|---------|---------------|---------------|------------|
| yc_curvature_x_recession | #1 x #2 (8.68% x 7.80%) | Yield curve shape signals amplified during recession fears | ~100% (both continuous) |
| real_rate_x_temporal | #4 x #3 (5.54% x 5.78%) | Rate changes more predictive when temporal patterns are active | ~100% (both continuous) |
| dxy_x_vix_persistence | #9 x #7 (4.42% x 4.82%) | Dollar-gold inverse relation amplified in volatile regimes | ~100% (both continuous) |

**Why only 3**: At 27 features with ~1492 train samples, samples-per-feature = 55.3:1 (vs 62.2:1 in attempt 7). This is adequate but conservative. Adding more interactions risks diluting signal. The 3 chosen are the highest-importance pairings with clear economic stories.

**Critical difference from attempt 8**: All 3 are CONTINUOUS products (always nonzero) vs attempt 8's BINARY products (0% to 6.7% activation). This ensures every training sample contributes to learning the interaction.

### 2.4 Combined Feature Set (27 features)

```python
ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + INTERACTION_COLUMNS
assert len(ALL_FEATURE_COLUMNS) == 27
```

### 2.5 NaN Handling

Interaction features are products of existing features. After base imputation (regime_probs -> 0.5, z-scores -> 0.0, etc.), all interaction features are computable. No additional NaN handling needed.

### 2.6 Data Split

Unchanged: 70/15/15 time-series split, no shuffle. Same split boundaries as attempt 7.

---

## 3. Model Architecture

### 3.1 Architecture: Dual-Track Single XGBoost + Bootstrap + OLS

```
Input: 27-dimensional feature vector (24 base + 3 interactions)
  |
  +---> [TRACK A: XGBoost with reg:squarederror] --> MSE predictions
  |         (same as attempt 7, proven stable)
  |
  +---> [TRACK B: XGBoost with directional_mse] --> Directional predictions
  |         (custom loss: 2x penalty for wrong-direction errors)
  |
  v
[VALIDATION COMPARISON]
  - Compute composite objective for both tracks
  - Select track with higher validation composite
  - If Track B composite < Track A composite: use Track A (safe fallback)
  |
  v
Selected raw prediction (single scalar per sample)
  |
  v
POST-TRAINING STEP 1: OLS Output Scaling (from attempt 7)
  - alpha_ols from validation set, capped [0.5, 10.0]
  |
  v
POST-TRAINING STEP 2: Bootstrap Ensemble Confidence (from attempt 7)
  - 5 models with seeds [42, 43, 44, 45, 46]
  - Confidence = 1 / (1 + std_across_models)
  - Also compute |prediction| confidence (primary HCDA method in att 7)
  |
  v
Output: prediction, scaled_prediction, bootstrap_std, confidence
  |
  v
Metrics: DA, HCDA (both methods), MAE (raw + scaled), Sharpe
```

### 3.2 Track A: Standard MSE (Identical to Attempt 7)

- Objective: `reg:squarederror`
- HP tuning: Optuna 60 trials (reduced from 100 to share budget with Track B)
- Early stopping: patience=100 on val RMSE

### 3.3 Track B: Directional MSE (NEW)

Custom objective that penalizes wrong-direction predictions more heavily:

```python
def directional_mse(y_true, y_pred):
    """MSE with 2x penalty for wrong-direction predictions.

    Gradient and Hessian:
      If sign(y_pred) == sign(y_true): standard MSE gradient
      If sign(y_pred) != sign(y_true): 2x MSE gradient

    Uses smooth sigmoid transition near y_pred=0 to avoid discontinuity.
    """
    error = y_pred - y_true

    # Smooth directional agreement: sigmoid(y_pred * y_true * k)
    # k=20 gives sharp transition, k=5 gives soft transition
    # Use k=10 as compromise
    agreement = 1.0 / (1.0 + np.exp(-y_pred * y_true * 10.0))

    # Weight: 1.0 when directions match (agreement -> 1), 2.0 when mismatch (agreement -> 0)
    weights = 2.0 - agreement  # [1.0, 2.0]

    grad = weights * 2.0 * error
    hess = weights * 2.0

    return grad, hess
```

- HP tuning: Optuna 40 trials (smaller budget, exploratory)
- Early stopping: patience=100 on val RMSE (eval_metric='rmse' set explicitly)
- Same HP search space as Track A

**Why sigmoid smoothing**: Hard `np.sign()` creates a gradient discontinuity at y_pred=0. The sigmoid with k=10 creates a smooth transition over the range [-0.3, +0.3]%, which is well within the model's typical prediction range (std=0.023%). This prevents training instability while preserving the directional penalty.

### 3.4 Track Selection Logic

```python
# After training both tracks
if track_b_composite > track_a_composite:
    selected_track = 'directional'
    pred_test = track_b_pred_test
    pred_val = track_b_pred_val
    final_model = track_b_model
else:
    selected_track = 'mse'
    pred_test = track_a_pred_test
    pred_val = track_a_pred_val
    final_model = track_a_model
```

This ensures attempt 9 is never worse than a pure MSE approach.

### 3.5 Fallback Mechanism (3-Level)

1. **Track fallback**: If Track B < Track A, use Track A (MSE)
2. **Optuna fallback**: If both tracks < attempt 2 fallback params, use attempt 2 params
3. **Architecture fallback**: All of above run on 27 features. If interaction features hurt, XGBoost's colsample_bytree will naturally exclude them

### 3.6 All Metric Functions

Identical to attempt 7. No changes to:
- `compute_direction_accuracy()`
- `compute_mae()`
- `compute_sharpe_trade_cost()`
- `compute_hcda()` (|prediction| method)
- `compute_hcda_bootstrap()` (bootstrap variance method)

---

## 4. Hyperparameter Specification

### 4.1 Fixed Parameters

Identical to attempt 7.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective (Track A) | reg:squarederror | Proven stable |
| objective (Track B) | directional_mse (custom) | Directional penalty |
| early_stopping_rounds | 100 | Unchanged |
| eval_metric | rmse | Required for both tracks |
| tree_method | hist | Fast |
| verbosity | 0 | Suppress |
| seed | 42 + trial.number | Reproducible |

### 4.2 Optuna Search Space (SAME for Both Tracks)

Identical to attempt 7 HP ranges. No changes.

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| max_depth | [2, 4] | int | Prevent outlier-memorization |
| n_estimators | [100, 800] | int | Same as attempt 7 |
| learning_rate | [0.001, 0.05] | log | Same as attempt 7 |
| colsample_bytree | [0.2, 0.7] | linear | Same as attempt 7 |
| subsample | [0.4, 0.85] | linear | Same as attempt 7 |
| min_child_weight | [12, 25] | int | Same as attempt 7 |
| reg_lambda (L2) | [1.0, 15.0] | log | Same as attempt 7 |
| reg_alpha (L1) | [0.5, 10.0] | log | Same as attempt 7 |

**Total: 8 hyperparameters** (unchanged from attempt 7).

### 4.3 Search Configuration

| Setting | Track A (MSE) | Track B (Directional) |
|---------|---------------|----------------------|
| n_trials | 60 | 40 |
| timeout | 3600 sec | 2400 sec |
| sampler | TPESampler(seed=42) | TPESampler(seed=43) |
| direction | maximize | maximize |

**Total trials**: 100 (60 + 40). Same total budget as attempt 7 (100 trials), split between two tracks. This ensures total execution time is comparable.

### 4.4 Optuna Objective Function (UPDATED WEIGHTS)

```python
# Attempt 7 weights: 40/30/10/20 (Sharpe/DA/MAE/HCDA)
# Attempt 9 weights: 30/40/10/20 (Sharpe/DA/MAE/HCDA)
#
# Rationale: Sharpe is 3.1x target (2.46 vs 0.80) -- massively over-optimized.
# DA has more marginal value: 60.04% vs 56% target = 4.04pp headroom.
# Shifting 10% weight from Sharpe to DA encourages configurations
# that prioritize directional accuracy over risk-adjusted magnitude.

objective = (
    0.30 * sharpe_norm +     # DECREASED from 0.40
    0.40 * da_norm +         # INCREASED from 0.30
    0.10 * mae_norm +
    0.20 * hc_da_norm
) - 0.30 * overfit_penalty
```

### 4.5 Normalization Functions

Identical to attempt 7:

```python
sharpe_norm = np.clip((val_sharpe + 3.0) / 6.0, 0.0, 1.0)    # [-3, +3] -> [0, 1]
da_norm = np.clip((val_da * 100 - 40.0) / 30.0, 0.0, 1.0)    # [40%, 70%] -> [0, 1]
mae_norm = np.clip((1.0 - val_mae) / 0.5, 0.0, 1.0)          # [0.5%, 1.0%] -> [0, 1]
hc_da_norm = np.clip((val_hc_da * 100 - 40.0) / 30.0, 0.0, 1.0)  # [40%, 70%] -> [0, 1]
```

---

## 5. Training Configuration

### 5.1 Training Algorithm

```
1. DATA PREPARATION:
   (identical to attempt 7, PLUS interaction feature generation)
   a. Fetch raw data using yfinance and fredapi
   b. Construct base features (5)
   c. Load 8 submodel output CSVs (same as attempt 7)
   d. Merge base + submodel + target on Date
   e. Apply NaN imputation (same as attempt 7)
>> f. Generate 3 continuous interaction features (NEW)
   g. Verify: 27 features, 0 remaining NaN
   h. Split: train (70%), val (15%), test (15%)

2. TRACK A: MSE XGBOOST HPO (60 trials, 1-hour timeout):
   - Objective: reg:squarederror
   - Optuna weights: 30/40/10/20 (Sharpe/DA/MAE/HCDA)
   - 27-feature input
   - Early stopping: 100 rounds
   - Select best trial params -> best_a_params

3. TRACK B: DIRECTIONAL XGBOOST HPO (40 trials, 40-min timeout):
   - Objective: directional_mse (custom)
   - Same Optuna weights: 30/40/10/20
   - eval_metric: 'rmse' (explicit, required for custom objective)
   - 27-feature input
   - Early stopping: 100 rounds
   - Select best trial params -> best_b_params

4. TRACK COMPARISON:
   - Compute validation composite for both tracks
   - Select track with higher composite
   - Report both tracks' metrics for transparency

5. FALLBACK EVALUATION:
   (same as attempt 7 -- attempt 2 best params + 27 features)
   - If fallback composite > both tracks: use fallback
   - If either track > fallback: use winning track

6. FINAL MODEL TRAINING:
   - Train with selected params (winning track or fallback)
   - 27-feature input, same train/val split

7. POST-TRAINING STEP 1: OLS OUTPUT SCALING:
   (identical to attempt 7)

8. POST-TRAINING STEP 2: BOOTSTRAP ENSEMBLE CONFIDENCE:
   (identical to attempt 7 -- 5 models, seeds [42-46])
   NOTE: Bootstrap models always use reg:squarederror regardless of
   which track won, because bootstrap is for confidence estimation
   and we want stable, comparable confidence scores.

9. EVALUATION ON ALL SPLITS:
   (identical to attempt 7 -- DA, HCDA both methods, MAE both, Sharpe,
    feature importance for 27 features, quarterly breakdown, decile analysis)
>> a. Report Track A vs Track B comparison
>> b. Report interaction feature importance rankings
>> c. Report confusion matrix (TP/FP/TN/FN) for test set
>> d. Compare negative prediction accuracy with attempt 7

10. SAVE RESULTS:
    (same output files as attempt 7)
```

### 5.2 Loss Functions

- Track A: reg:squarederror (standard MSE)
- Track B: directional_mse (custom, 2x penalty for wrong direction, sigmoid-smoothed)

### 5.3 Early Stopping

- Metric: RMSE on validation set (for both tracks)
- Patience: 100 rounds
- Maximum rounds: Optuna-controlled (100-800)

### 5.4 Fallback Configuration

Same as attempt 7:

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
| enable_gpu | false | CHANGED from true | Tree-based models with <2000 train samples are faster on CPU. CPU gives 12-hour quota vs 9 hours GPU. |
| Estimated execution time | 30-55 minutes | Similar to att 7 | 100 total trials (60+40) same as att 7's 100 |
| Estimated memory usage | 1.5 GB | Unchanged | +3 features adds ~3KB |
| Required pip packages | [] | Unchanged | No new dependencies |
| Internet required | true | Unchanged | For data fetching |
| Kaggle Notebook ID | bigbigzabuton/gold-model-training-meta-model | Unchanged | Same notebook, new version |
| dataset_sources | bigbigzabuton/gold-prediction-submodels | Unchanged | All submodel CSVs |
| Optuna total timeout | 6000 sec (1.67 hrs) | REDUCED | Only 2 tracks, not 3 learners |

### 6.1 GPU Decision

Changed from `true` (attempt 7) to `false` (attempt 9). Rationale:
- Dataset has ~1492 training samples and 27 features
- XGBoost `hist` tree method on CPU is faster than GPU for <10,000 samples
- CPU quota is 12 hours vs GPU quota of 9 hours
- Attempt 8 ran successfully on CPU in ~90 minutes

### 6.2 kernel-metadata.json

```json
{
  "id": "bigbigzabuton/gold-model-training-meta-model",
  "title": "Gold Meta-Model Training - Attempt 9",
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

**Task**: Generate `notebooks/meta_model_9/train.ipynb` (self-contained Kaggle Notebook)

**Base**: Copy attempt 7 notebook (`notebooks/meta_model_7/train.ipynb`) with the following modifications.

#### 7.2.1 Feature Definitions Update (Cell 3)

```python
BASE_FEATURE_COLUMNS = [
    # ... (same 24 as attempt 7) ...
]

INTERACTION_COLUMNS = [
    'yc_curvature_x_recession',
    'real_rate_x_temporal',
    'dxy_x_vix_persistence',
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + INTERACTION_COLUMNS
TARGET = 'gold_return_next'

assert len(FEATURE_COLUMNS) == 27, f"Expected 27 features, got {len(FEATURE_COLUMNS)}"
```

#### 7.2.2 Interaction Feature Generation (New Cell After Cell 7)

Add after NaN imputation, before data split:

```python
print("\nGenerating interaction features...")

def generate_interaction_features(df):
    """Generate continuous interaction features. All products, no binary thresholds."""
    df['yc_curvature_x_recession'] = df['yc_curvature_z'] * df['xasset_recession_signal']
    df['real_rate_x_temporal'] = df['real_rate_change'] * df['temporal_context_score']
    df['dxy_x_vix_persistence'] = df['dxy_change'] * df['vix_persistence']
    return df

final_df = generate_interaction_features(final_df)

# Verify interaction features
for col in INTERACTION_COLUMNS:
    assert col in final_df.columns, f"Missing interaction feature: {col}"
    nan_count = final_df[col].isna().sum()
    print(f"  {col}: range=[{final_df[col].min():.4f}, {final_df[col].max():.4f}], NaN={nan_count}")

print(f"\nAll {len(FEATURE_COLUMNS)} features present (24 base + 3 interaction)")
```

**CRITICAL**: Interaction features must be generated AFTER NaN imputation because they depend on imputed values.

#### 7.2.3 Directional Loss Function (New Cell Before Optuna)

```python
import xgboost as xgb

def directional_mse(y_true, y_pred):
    """MSE with 2x penalty for wrong-direction predictions.
    Sigmoid-smoothed to avoid gradient discontinuity at y_pred=0.

    Args:
        y_true: actual values (numpy array)
        y_pred: predicted values (numpy array)
    Returns:
        grad: gradient (numpy array)
        hess: hessian (numpy array)
    """
    error = y_pred - y_true

    # Smooth directional agreement via sigmoid
    # agreement -> 1 when signs match, -> 0 when signs differ
    # k=10 gives sharp but smooth transition
    agreement = 1.0 / (1.0 + np.exp(-np.clip(y_pred * y_true * 10.0, -50, 50)))

    # Weight: 1.0 when directions match, 2.0 when mismatch
    weights = 2.0 - agreement

    grad = weights * 2.0 * error
    hess = weights * 2.0

    return grad, hess

print("Directional MSE loss function defined")
```

#### 7.2.4 Optuna Objective Update (Cell 13)

Update the weights:

```python
# ATTEMPT 9 WEIGHTS: 30/40/10/20 (Sharpe/DA/MAE/HCDA)
# Changed from Attempt 7: 40/30/10/20
objective = (
    0.30 * sharpe_norm +     # DECREASED from 0.40
    0.40 * da_norm +         # INCREASED from 0.30
    0.10 * mae_norm +
    0.20 * hc_da_norm
) - 0.30 * overfit_penalty
```

#### 7.2.5 Dual-Track HPO (Replace Single Optuna Block)

```python
# ==== TRACK A: Standard MSE (60 trials) ====
print("="*60)
print("TRACK A: Standard MSE XGBoost (60 trials)")
print("="*60)

study_a = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_a.optimize(
    lambda trial: optuna_objective(trial, objective_fn='reg:squarederror'),
    n_trials=60,
    timeout=3600,
)

# ==== TRACK B: Directional MSE (40 trials) ====
print("="*60)
print("TRACK B: Directional MSE XGBoost (40 trials)")
print("="*60)

study_b = optuna.create_study(direction='maximize', sampler=TPESampler(seed=43))
study_b.optimize(
    lambda trial: optuna_objective(trial, objective_fn=directional_mse),
    n_trials=40,
    timeout=2400,
)

# ==== TRACK COMPARISON ====
print("="*60)
print("TRACK COMPARISON")
print("="*60)
print(f"  Track A (MSE):         composite={study_a.best_value:.4f}")
print(f"  Track B (Directional): composite={study_b.best_value:.4f}")

if study_b.best_value > study_a.best_value:
    print("  -> Track B (Directional) selected")
    selected_study = study_b
    selected_track = 'directional'
else:
    print("  -> Track A (MSE) selected")
    selected_study = study_a
    selected_track = 'mse'
```

**IMPORTANT**: The `optuna_objective` function must accept an `objective_fn` parameter:

```python
def optuna_objective(trial, objective_fn='reg:squarederror'):
    params = {
        'objective': objective_fn,
        'max_depth': trial.suggest_int('max_depth', 2, 4),
        # ... (all same HP ranges as attempt 7) ...
        'eval_metric': 'rmse',  # CRITICAL: required for custom objective
        'verbosity': 0,
        'seed': 42 + trial.number,
    }
    n_estimators = trial.suggest_int('n_estimators', 100, 800)

    model = xgb.XGBRegressor(**params, n_estimators=n_estimators, early_stopping_rounds=100)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # ... (same metric computation and composite as attempt 7, with UPDATED WEIGHTS) ...
```

#### 7.2.6 Bootstrap Confidence (Modified)

Bootstrap models ALWAYS use `reg:squarederror` regardless of which track won:

```python
# Bootstrap for confidence estimation - always use MSE for stability
for i, seed in enumerate(bootstrap_seeds):
    model_boot = xgb.XGBRegressor(
        objective='reg:squarederror',  # ALWAYS MSE for bootstrap
        # ... (same params as selected_params, but with MSE objective) ...
    )
```

This ensures bootstrap confidence scores are comparable across tracks.

#### 7.2.7 Confusion Matrix Diagnostic (New Cell in Diagnostics)

```python
# Confusion matrix for test set
print("\nCONFUSION MATRIX (test set):")
pred_positive = pred_test > 0
actual_positive = y_test > 0

TP = (pred_positive & actual_positive).sum()
FP = (pred_positive & ~actual_positive).sum()
TN = (~pred_positive & ~actual_positive).sum()
FN = (~pred_positive & actual_positive).sum()

print(f"  True Positives:  {TP}")
print(f"  False Positives: {FP}")
print(f"  True Negatives:  {TN}")
print(f"  False Negatives: {FN}")
print(f"  Positive pred accuracy: {TP/(TP+FP)*100:.1f}%")
print(f"  Negative pred accuracy: {TN/(TN+FN)*100:.1f}%" if (TN+FN) > 0 else "  No negative predictions")
print(f"  Positive prediction %: {(pred_test > 0).sum()/len(pred_test)*100:.1f}%")
```

#### 7.2.8 Result Saving Updates

```python
training_result['attempt'] = 9
training_result['architecture'] = 'XGBoost (dual-track MSE/directional) + 3 interaction features + Bootstrap confidence + OLS scaling'
training_result['model_config']['n_features'] = 27
training_result['dual_track'] = {
    'track_a_mse': {
        'trials': len(study_a.trials),
        'best_value': float(study_a.best_value),
        'best_da': float(study_a.best_trial.user_attrs['val_da']),
    },
    'track_b_directional': {
        'trials': len(study_b.trials),
        'best_value': float(study_b.best_value),
        'best_da': float(study_b.best_trial.user_attrs['val_da']),
    },
    'selected_track': selected_track,
}
training_result['interaction_features'] = {
    'n_interaction_features': 3,
    'features': INTERACTION_COLUMNS,
}
training_result['optuna_weights'] = '30/40/10/20 (Sharpe/DA/MAE/HCDA)'
```

#### 7.2.9 kernel-metadata.json

```json
{
  "id": "bigbigzabuton/gold-model-training-meta-model",
  "title": "Gold Meta-Model Training - Attempt 9",
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

### 7.3 Implementation Checklist for builder_model

1. Copy attempt 7 notebook structure (`notebooks/meta_model_7/train.ipynb`)
2. Update markdown header to "Attempt 9" (Cell 0)
3. Add `INTERACTION_COLUMNS` and update FEATURE_COLUMNS to 27 (Cell 3)
4. Add `generate_interaction_features()` function AFTER NaN imputation (new cell)
5. Change feature count assertion from 24 to 27 in all relevant places
6. Add `directional_mse()` custom loss function (new cell)
7. Update Optuna objective weights from 40/30/10/20 to 30/40/10/20
8. Modify `optuna_objective()` to accept `objective_fn` parameter
9. Replace single Optuna block with dual-track (Track A: 60 trials MSE, Track B: 40 trials directional)
10. Add track comparison logic
11. Update fallback comparison to compare against BOTH tracks
12. Ensure bootstrap models always use `reg:squarederror` (not the custom loss)
13. Add confusion matrix diagnostic
14. Update all "24 features" text to "27 features"
15. Update `training_result['attempt']` to 9
16. Add dual_track and interaction_features metadata to training_result
17. Update kernel-metadata.json: title to "Attempt 9", enable_gpu to false
18. Run `scripts/validate_notebook.py` to verify notebook

**CRITICAL implementation notes**:
- XGBoost custom objective API (v2.0+): `def fn(y_true, y_pred)` -- NOT `(y_pred, DMatrix)`
- Custom objective REQUIRES explicit `eval_metric='rmse'` in params
- np.clip the sigmoid argument to [-50, 50] to prevent overflow
- Interaction features must be generated AFTER NaN imputation
- Bootstrap confidence uses MSE only, never the custom loss
- `enable_gpu` is `false` (CPU mode for 12-hour quota)

---

## 8. Risk Mitigation

### Risk 1: Interaction Features Add Noise (LOW)

**Scenario**: 3 interaction features dilute signal. XGBoost assigns them low importance.

**Probability**: 20-25%.

**Mitigation**:
1. Only 3 features (not 6 or 8 like attempt 8). Minimal dimensionality increase.
2. XGBoost colsample_bytree [0.2, 0.7] naturally excludes noisy features.
3. All 3 are continuous (always nonzero), unlike attempt 8's sparse binary features.
4. Worst case: they rank last in importance and are effectively ignored (no harm).

### Risk 2: Directional Loss Causes Training Instability (MODERATE)

**Scenario**: The custom loss function causes XGBoost to diverge or produce degenerate predictions (all zeros, all positive, etc.).

**Probability**: 15-25%.

**Mitigation**:
1. Track B is EXPLORATORY -- Track A (MSE) is always available as safe fallback.
2. Sigmoid smoothing prevents gradient discontinuity at y_pred=0.
3. eval_metric='rmse' provides a stable early stopping criterion independent of the custom loss.
4. If all Track B trials produce poor composites, the system automatically falls back to Track A.

### Risk 3: Optuna Weight Shift Hurts Sharpe (LOW-MODERATE)

**Scenario**: Shifting 10% weight from Sharpe to DA causes Optuna to find HP configurations with DA 61% but Sharpe 1.5 (still passing, but lower).

**Probability**: 25-35%.

**Mitigation**:
1. Sharpe still has 30% weight (not zero). Configurations with Sharpe < 0.8 would still be penalized.
2. The normalization range [-3, +3] -> [0, 1] means Sharpe 2.46 maps to 0.91 (nearly saturated). Reducing Sharpe from 2.46 to 2.0 only reduces sharpe_norm from 0.91 to 0.83 (-0.08). The DA gain from 60% to 61% increases da_norm from 0.67 to 0.70 (+0.03). Net effect on composite: approximately neutral if DA gains 1pp and Sharpe drops 0.46.
3. Attempt 7's best trial had val_da=52.53% and val_sharpe unknown but presumably moderate. The weight shift may help discover configurations that were previously unexplored.

### Risk 4: Regression from Attempt 7 (LOW)

**Scenario**: Despite all safeguards, the combination of changes produces worse test metrics than attempt 7.

**Probability**: 10-15%.

**Mitigation**:
1. Three-level fallback: Track B -> Track A -> attempt 2 params
2. Same HP ranges as attempt 7 (Track A with MSE is essentially attempt 7 + 3 features)
3. Track A alone (60 trials MSE on 27 features) is a strict superset of attempt 7's approach. The only difference is 3 additional features and 40 fewer trials.
4. If attempt 9 regresses, attempt 7 remains the final model.

### Risk 5: MAE Target Remains Infeasible (CERTAIN)

**Probability**: >95%.

**Mitigation**: Accept. MAE is waived. Focus on DA, HCDA, Sharpe.

---

## 9. Expected Outcomes

| Metric | Attempt 7 (actual) | Attempt 9 (expected) | Delta | Confidence |
|--------|-------------------|---------------------|-------|------------|
| DA | 60.04% | 60.0-61.5% | 0-+1.5pp | Medium |
| HCDA | 64.13% | 63.0-65.0% | -1.1-+0.9pp | Medium |
| MAE | 0.943% | 0.93-0.95% | -0.01-+0.01 | Low |
| Sharpe | 2.46 | 2.0-2.5 | -0.46-+0.04 | Medium |
| Train-test DA gap | -5.28pp | -6 to 0pp | -- | High |
| Targets passed | 3/4 | 3/4 | 0 | High |

**Probability of outcomes**:

| Outcome | Probability |
|---------|------------|
| Improvement on DA (>60.04%) with 3/4 targets maintained | 35-45% |
| No significant change from attempt 7 (within noise) | 35-40% |
| Regression (DA < 59.5% or Sharpe < 2.0) | 10-15% |
| All 4 targets met | <3% |

**Important note**: The primary goal is to maintain attempt 7's 3/4 target pass while exploring whether the DA/Sharpe weight shift and directional loss can push DA slightly higher. A "no change" result is acceptable.

---

## 10. Success Criteria

### Primary Targets (on test set)

| Metric | Target | Attempt 7 Actual | Note |
|--------|--------|------------------|------|
| DA | > 56% | 60.04% (PASS) | Maintain or improve |
| HCDA | > 60% | 64.13% (PASS) | Maintain or improve |
| MAE | < 0.75% | 0.943% (FAIL, waived) | Not targeted |
| Sharpe | > 0.80 | 2.46 (PASS) | Maintain (accept slight decrease) |

### Attempt 9 Specific Success Criteria

| Criterion | Threshold | Purpose |
|-----------|-----------|---------|
| No regression vs attempt 7 | DA >= 59.5%, Sharpe >= 1.8 | Guard against degradation |
| Interaction features contribute | >= 1 interaction feature in top 15 importance | Validate feature engineering |
| Track comparison informative | Report both tracks' metrics | Understand MSE vs directional tradeoff |
| Negative prediction accuracy | Report % and compare with attempt 7 (55.2%) | Measure directional loss impact |
| Confusion matrix balanced | Fewer false negatives or more true negatives than att 7 | Validate improvement mechanism |

### Decision Rules After Evaluation

| Outcome | Action |
|---------|--------|
| DA > 60.0% AND Sharpe > 2.0 AND HCDA > 62% | Accept as new best if any metric improves over att 7 |
| DA >= 59.5% AND all 3 passing targets maintained | Marginal. Compare with attempt 7 in detail. |
| DA < 59.5% OR Sharpe < 1.8 | Regression. Revert to attempt 7 as final model. |
| Track B (directional) selected and DA improved | Directional loss validated. Use for attempt 10 if needed. |
| Track A (MSE) selected | Directional loss adds no value. Report for posterity. |

---

## 11. Comparison with Previous Attempts

| Aspect | Att 2 | Att 5 | Att 7 (current best) | Att 8 (FAIL) | Att 9 (this) |
|--------|-------|-------|---------------------|--------------|--------------|
| Architecture | Single XGBoost | Single XGBoost | Single XGBoost | 3-GBDT Stacking | **Single XGBoost** |
| Features | 22 | 23 | 24 | 30 (24+6 binary) | **27** (24+3 continuous) |
| HP Search | 80 trials | 100 trials | 100 trials | 260 trials | **100 trials** (60+40) |
| Obj weights | 50/30/10/10 | 40/30/10/20 | 40/30/10/20 | 35/35/10/20 | **30/40/10/20** |
| max_depth | [2,4] | [2,5] | [2,4] | [2,5] | **[2,4]** |
| Loss function | MSE | MSE | MSE | MSE | **MSE + directional** (dual-track) |
| Interaction features | None | None | None | 6 binary (FAIL) | **3 continuous** |
| Meta-learner | None | None | None | Ridge (FAIL) | **None** |
| enable_gpu | true | true | true | false | **false** |
| HCDA method | |pred| | |pred| | Bootstrap+|pred| | Bootstrap+|pred| | Bootstrap+|pred| |
| OLS scaling | No | No | Yes | Yes | Yes |
| Fallback | No | No | Att 2 params | Att 2 + single XGB | **Att 2 params + Track A** |

---

## 12. Philosophical Note on Attempt Budget

With 2 attempts remaining (9 and 10), the optimal strategy is:
- **Attempt 9** (this): Conservative. Attempt 7 base + targeted improvements. Low risk.
- **Attempt 10** (if needed): Can be more aggressive, knowing attempt 7 is the fallback.

This attempt does NOT need to be transformative. It needs to explore whether small, low-risk modifications can push DA higher while maintaining the other passing targets. Even a "no change" result is informative: it confirms that attempt 7 is near the practical performance ceiling for this data and architecture.

---

**End of Design Document**

**Architect**: architect (Opus 4.6)
**Date**: 2026-02-17
**Based on**: Attempt 7 architecture (proven) + attempt 8 failure analysis + evaluator improvement plan (fact-checked)
**Supersedes**: docs/design/meta_model_attempt_8.md
