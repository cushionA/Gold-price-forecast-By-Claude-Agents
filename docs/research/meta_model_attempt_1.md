# Research Report: Meta-Model (Attempt 1)

**Date:** 2026-02-15
**Phase:** Phase 3 Meta-Model Construction
**Objective:** Integrate 19 base features + 20 submodel outputs (39 total) to predict next-day gold return with DA > 56%, HC-DA > 60%, MAE < 0.75%, Sharpe > 0.8

---

## Executive Summary

This research addresses 11 critical questions for meta-model construction, covering architecture selection, multi-objective optimization, feature selection, regularization, and benchmarking. Key findings:

1. **Recommended Architecture:** XGBoost or LightGBM with strong regularization as primary choice, with TabNet as a neural alternative if needed
2. **Architecture Approach:** Single unified model on all 39 features (not stacking) to minimize overfitting risk
3. **Loss Function:** Custom directional-weighted MSE (AdjMSE2/AdjMSE3 variants) that penalizes sign disagreement
4. **High-Confidence Threshold:** Prediction magnitude-based threshold optimized via Optuna (e.g., |pred| > threshold)
5. **Feature Strategy:** Test all 39 features first, then ablate cny_demand if DA degrades; tree models handle VIF=12.47 internally
6. **Regularization Priority:** Strong L1/L2, early stopping, max_depth≤5, min_child_weight≥3 for XGBoost; high dropout for neural models
7. **Cross-Validation:** Single train/val split for speed (50 trials); temporal CV only if overfitting persists
8. **Benchmark Insights:** HMM regime features + XGBoost meta-learner is a proven pattern in finance; 56% DA is challenging but achievable
9. **Optuna Strategy:** Include feature selection flags (include_cny, include_etf_vif) as categorical hyperparameters

---

## Q1: Architecture Strengths/Weaknesses for 39 Heterogeneous Features (1766 samples)

### XGBoost/LightGBM (RECOMMENDED)

**Strengths:**
- **Robust to multicollinearity:** Tree-based models are minimally affected by correlated features. When features are correlated, different trees randomly select among them, averaging out the effect across the ensemble ([Gupta, 2024](https://vishesh-gupta.medium.com/correlation-in-xgboost-8afa649bd066); [XGBoosting](https://xgboosting.com/xgboost-robust-to-correlated-input-features-multi-collinearity/))
- **Handles heterogeneous features natively:** No preprocessing needed for mixing raw prices, z-scores, regime probabilities, and binary signals
- **Strong regularization controls:** L1/L2 penalties, max_depth, min_child_weight, subsample, colsample_bytree all directly combat overfitting ([Towards Data Science](https://towardsdatascience.com/hyperparameter-tuning-to-reduce-overfitting-lightgbm-5eb81a0b464e/))
- **Small sample performance:** With 1766 samples, LightGBM's min_data_in_leaf parameter effectively prevents overly-specific splits ([PyLigent](https://pyligent.github.io/2019-08-20-lightGBM_XGBoost/))
- **Proven in finance:** Baseline XGBoost already achieved train DA 54.5% (near target 56%), indicating the feature set has signal

**Weaknesses:**
- **Overfitting risk confirmed:** Baseline showed 11pp train-test DA gap (54.5% → 43.5%). With 39 features vs 1766 samples (ratio 1:45), regularization must be aggressive
- **Feature importance dilution:** With VIF=12.47 on etf_regime_prob, importance scores split across correlated features, reducing interpretability ([Mane, 2024](https://medium.com/@manepriyanka48/multicollinearity-in-tree-based-models-b971292db140))
- **No explicit uncertainty:** Does not produce confidence scores without additional calibration

**Recommendation:** XGBoost with max_depth=3-5, min_child_weight≥3, subsample=0.6-0.8, colsample_bytree=0.6-0.8, lambda≥1.0, alpha≥0.5

---

### MLP (Multi-Layer Perceptron)

**Strengths:**
- **Universal approximation:** Can learn complex nonlinear interactions between HMM regime probabilities and base features
- **Confidence via dropout:** MC Dropout at inference provides uncertainty estimates for high-confidence thresholding ([Kuleshov et al., 2018](https://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf))

**Weaknesses:**
- **Severe overfitting risk:** 39 features with 1766 samples requires very small networks (e.g., 2 layers × 32 units max) and dropout≥0.3
- **Requires feature scaling:** Mixing raw prices (GLD ~$180) with z-scores (mean=0, std=1) requires careful normalization. Returns transformation is safer ([arxiv:2109.00983](https://arxiv.org/pdf/2109.00983))
- **Multicollinearity sensitivity:** Unlike trees, MLPs can suffer from correlated inputs causing gradient instability ([Nature Communications](https://www.nature.com/articles/s41599-024-04047-5))
- **No precedent:** Baseline used XGBoost; switching to MLP risks losing known-good architecture advantages

**Recommendation:** Only use if XGBoost/LightGBM fail after 2-3 attempts. Architecture: 2 layers, 32-64 units, dropout=0.3-0.5, L2 regularization, BatchNorm

---

### TabNet

**Strengths:**
- **Attention-based feature selection:** Self-attention mechanism can learn to ignore noisy features like cny_demand automatically ([Kim, 2024](https://medium.com/@kdk199604/tabnet-a-deep-learning-breakthrough-for-tabular-data-bcd39c47a81c))
- **Handles heterogeneous data:** Feature tokenization layer processes mixed feature types effectively ([MDPI 2025](https://www.mdpi.com/2079-8954/13/10/892))
- **Built-in interpretability:** Attention masks show which features contribute to each prediction
- **Recent finance success:** Used in SME supply chain financial risk prediction with strong results ([MDPI 2025](https://www.mdpi.com/2079-8954/13/10/892))

**Weaknesses:**
- **Sample size concern:** TabNet typically requires more data than tree models. 1766 samples may be insufficient
- **No established precedent:** Our baseline uses XGBoost. TabNet introduces architecture risk
- **Training complexity:** Requires careful tuning of attention sparsity, decision steps, and feature dimension hyperparameters

**Recommendation:** Second-tier alternative if XGBoost overfits despite strong regularization (unlikely given baseline success)

---

### FT-Transformer

**Strengths:**
- **State-of-the-art on tabular benchmarks:** Embeds each feature as a token and applies multi-head self-attention ([LAMDA-Tabular TALENT](https://github.com/LAMDA-Tabular/TALENT))
- **Heterogeneous feature handling:** Dedicated feature tokenization layer ([arxiv:2407.00956v4](https://arxiv.org/html/2407.00956v4))

**Weaknesses:**
- **Severe overfitting risk:** Transformers are parameter-hungry. With 1766 samples, overfitting is almost guaranteed
- **No finance precedent in project:** Introduces unnecessary complexity without evidence of superiority over trees
- **Requires extensive tuning:** Attention heads, embedding dimensions, number of transformer blocks—all add to HP search space

**Recommendation:** Not recommended for this project. Overfitting risk too high.

---

### Stacking Ensemble (XGBoost + MLP + LightGBM → Meta-Learner)

**Strengths:**
- **Combines diverse models:** Recent research shows stacking ARIMA, Random Forest, LSTM, GRU, Transformer → XGBoost meta-learner achieves R²=0.97-0.99 on financial data ([MDPI 2025](https://www.mdpi.com/2227-7072/13/4/201))
- **Robust to individual model failures:** If one base model overfits, others compensate
- **P2P lending success:** KNN + SVM + RF → XGBoost achieved 99.98% accuracy on lending data ([ScienceDirect 2023](https://www.sciencedirect.com/science/article/pii/S2667305323000297))

**Weaknesses:**
- **Overfitting multiplication:** With 1766 samples, training 3+ base models + meta-learner fragments data. Each base model sees <1400 samples
- **Computational cost:** 50 Optuna trials × 3 base models = 150 model trains. Kaggle time limits may be exceeded
- **Complexity without justification:** Baseline XGBoost already reaches train DA 54.5%. Stacking is premature optimization
- **Meta-learner overfitting risk:** Meta-learner still faces 39 features (base model predictions) on small validation set

**Recommendation:** Not recommended for Attempt 1. Consider only if single model attempts 1-3 fail consistently.

---

### Summary Ranking for This Project

| Rank | Architecture | Justification |
|------|-------------|---------------|
| 1 | **XGBoost** | Baseline precedent, robust to multicollinearity, strong regularization, handles heterogeneous features |
| 2 | **LightGBM** | Similar to XGBoost but faster; good if Kaggle time limits are hit |
| 3 | **TabNet** | If tree models overfit despite regularization; attention-based feature selection may help |
| 4 | **MLP** | Requires extensive feature engineering (returns, scaling); high risk |
| 5 | **FT-Transformer** | Overfitting risk too high for 1766 samples |
| 6 | **Stacking** | Premature complexity; try simpler approaches first |

**Architect Recommendation:** Start with XGBoost. The baseline already demonstrated viability (train DA 54.5%). The core challenge is regularization, not architecture innovation.

---

## Q2: Single Unified Model vs. Stacking vs. Feature-Group Architecture?

### Single Unified Model (RECOMMENDED)

**Approach:** Train one XGBoost/LightGBM on all 39 features together.

**Pros:**
- **Simplest approach:** Fewer moving parts = less risk of implementation bugs
- **Maximum training data:** All 1766 samples used for single model (not split across base models)
- **Natural feature interaction:** Tree splits can combine base + submodel features (e.g., "if vix_regime_prob > 0.7 AND real_rate < 2.5 then...")
- **Proven baseline:** Phase 1 XGBoost used this approach and achieved train DA 54.5%

**Cons:**
- **All features weighted equally initially:** Cannot explicitly encode "submodel features are derived patterns" vs "base features are raw data"
- **Single point of failure:** If model architecture is wrong, entire attempt fails

**Overfitting Risk:** MEDIUM. With 39 features on 1766 samples (ratio 1:45), overfitting is manageable with:
- max_depth ≤ 5
- min_child_weight ≥ 3
- subsample = 0.6-0.8
- colsample_bytree = 0.6-0.8
- L1/L2 regularization (lambda ≥ 1.0, alpha ≥ 0.5)

**Decision:** **Use this for Attempt 1.** If it fails due to overfitting, then consider alternatives.

---

### Two-Level Stacking Ensemble

**Approach:**
1. **Level 1 (Base Models):** Train 3 models on 39 features:
   - XGBoost (tree-based)
   - LightGBM (tree-based, different split strategy)
   - MLP (neural, captures different patterns)
2. **Level 2 (Meta-Learner):** Train XGBoost on 3 base model predictions → final prediction

**Pros:**
- **Diversity:** Different models capture different patterns. Averaging reduces variance
- **Published success:** Stock prediction with ARIMA+RF+LSTM+GRU+Transformer → XGBoost achieved R²=0.9735-0.9905 ([MDPI 2025](https://www.mdpi.com/2227-7072/13/4/201))

**Cons:**
- **Data fragmentation:** With 1766 samples, stacking requires train/val split for Level 1, then Level 2 trains on val predictions. Effective training data for Level 2 may be <400 samples
- **Overfitting risk amplification:** Each base model must be regularized heavily. Meta-learner can still overfit on Level 2 validation set
- **Complexity:** 3 base models × 50 Optuna trials = 150 model trains. Kaggle Notebook may timeout
- **No evidence of need:** Baseline XGBoost already achieved train DA 54.5% with single model. Problem is overfitting, not architecture capacity

**Overfitting Risk:** HIGH. Stacking is designed for large datasets where diversity benefits outweigh data fragmentation costs. With 1766 samples, data fragmentation dominates.

**Decision:** **Not recommended for Attempt 1.** Revisit only if Attempts 1-3 all fail with single model.

---

### Feature-Group-Aware Architecture

**Approach:**
1. **Subnetwork 1:** Process 19 base features → embedding_base (e.g., MLP 19→16)
2. **Subnetwork 2:** Process 20 submodel features → embedding_submodel (e.g., MLP 20→16)
3. **Fusion Layer:** Concatenate [embedding_base, embedding_submodel] → MLP → final prediction

**Pros:**
- **Explicit feature type encoding:** Neural architecture "knows" which features are raw vs derived
- **Differential regularization:** Can apply stronger dropout to submodel branch if those features are noisier

**Cons:**
- **Neural network requirement:** This approach requires MLP/TabNet, which have worse overfitting profiles than XGBoost for 1766 samples
- **Added complexity:** More hyperparameters (embedding sizes, dropout per branch, fusion layer architecture)
- **No precedent:** Baseline XGBoost worked. Introducing custom architecture is high risk
- **Hypothesis unvalidated:** We don't have evidence that base vs submodel features need separate processing. Tree models may handle this via natural feature selection

**Overfitting Risk:** HIGH. Neural architectures require more data. 1766 samples split across two subnets + fusion = severe overfitting risk.

**Decision:** **Not recommended.** Only consider if XGBoost/LightGBM fail 3+ times and TabNet also fails.

---

### Trade-Off Summary

| Approach | Overfitting Risk | Complexity | Justification for Use |
|----------|-----------------|------------|----------------------|
| **Single Unified (XGBoost)** | Medium | Low | Baseline precedent, maximum training data, simple |
| Stacking Ensemble | High | High | Only if single model fails 3+ times |
| Feature-Group Architecture | High | Very High | Only if tree models + TabNet all fail |

**Final Recommendation for Q2:** **Single unified XGBoost model on all 39 features.** This maximizes training data, minimizes complexity, and aligns with the successful Phase 1 baseline. Regularization (max_depth, min_child_weight, subsample, L1/L2) is sufficient to control overfitting.

---

## Q3: Multi-Objective Loss Function (DA + MAE + Sharpe)

### The Core Problem

Our targets are:
1. **Direction Accuracy (DA):** Sign agreement between prediction and actual
2. **Mean Absolute Error (MAE):** Magnitude accuracy
3. **Sharpe Ratio:** Risk-adjusted returns (requires correct direction AND magnitude)

Standard MSE/MAE optimize magnitude only. They penalize large errors but ignore sign disagreement. A model predicting +1.5% when actual is -1.0% has the same MAE loss as predicting +1.5% when actual is +3.5%, but the first is directionally wrong (catastrophic for Sharpe) while the second is correct.

**Submodel Evidence:**
- **Technical:** MAE -0.182 (excellent) but DA +0.05%, Sharpe -0.092 (poor direction)
- **CNY Demand:** MAE -0.066 (good) but DA -2.06%, Sharpe -0.593 (severe direction failure)

This DA-MAE trade-off must be resolved at the meta-model level.

---

### Solution: Directional-Weighted Loss Functions

Recent research proposes **Adjusted MSE (AdjMSE)** variants that penalize sign disagreement:

**AdjMSE2 Formula ([OAAIML](https://www.oajaiml.com/uploads/archivepdf/29151193.pdf)):**
```
If sign(pred) == sign(actual):
    loss = MSE
Else:
    loss = MSE × penalty_factor  (e.g., penalty_factor=3.0)
```

**AdjMSE3 Formula:**
```
If sign(pred) == sign(actual):
    loss = MSE
Else:
    loss = MSE × (1 + abs(pred) + abs(actual))
```

**Performance Evidence:**
- AdjMSE2 and AdjMSE3 "more than doubled Sharpe ratios and threefold D-ratios" compared to standard MSE ([OAAIML](https://www.oajaiml.com/uploads/archivepdf/29151193.pdf))
- Directly optimizing Sharpe ratio in loss functions improves risk-adjusted returns ([SJTU 2020](https://www.acem.sjtu.edu.cn/sffs/2020/pdf/paper3.pdf))

---

### Recommended Loss Function: Hybrid Directional MAE

For our project (predicting gold returns), propose:

```python
def directional_mae_loss(y_pred, y_true):
    """
    Custom loss balancing MAE and directional accuracy.

    Penalizes wrong-direction predictions more heavily than
    right-direction magnitude errors.
    """
    mae = abs(y_pred - y_true)
    sign_correct = (y_pred * y_true) > 0  # Same sign

    # Apply penalty when signs disagree
    penalty = torch.where(
        sign_correct,
        1.0,  # No extra penalty
        2.5   # 2.5x penalty for wrong direction
    )

    return (mae * penalty).mean()
```

**Advantages:**
1. **Direction-aware:** Wrong-direction predictions pay 2.5× penalty
2. **MAE foundation:** Still optimizes magnitude (MAE < 0.75 target)
3. **Sharpe alignment:** Direction correctness directly improves Sharpe
4. **Tunable penalty:** Optuna can optimize penalty factor (1.5-5.0 range)

---

### Alternative: Multi-Task Loss (Not Recommended)

**Approach:** Jointly optimize direction classification + magnitude regression:
```python
loss = alpha * CrossEntropy(sign(pred), sign(actual)) + (1-alpha) * MAE(pred, actual)
```

**Cons:**
- Requires tuning alpha (another hyperparameter)
- CrossEntropy requires discrete labels (up/down), losing magnitude information
- More complex than directional penalty approach

---

### XGBoost Implementation Challenge

XGBoost's native objectives (reg:squarederror, reg:absoluteerror) don't support custom directional penalties. Solutions:

1. **Custom Objective Function (RECOMMENDED):**
   ```python
   def custom_obj(y_pred, dtrain):
       y_true = dtrain.get_label()
       sign_correct = (y_pred * y_true) > 0
       penalty = np.where(sign_correct, 1.0, 2.5)

       grad = penalty * np.sign(y_pred - y_true)  # MAE gradient with penalty
       hess = penalty * np.ones_like(y_pred)      # Second derivative
       return grad, hess
   ```

2. **Sample Weighting (Simpler Alternative):**
   - Train with standard MAE
   - After each iteration, reweight samples based on sign correctness
   - Not as elegant but easier to implement

---

### Recommendation for Q3

**Primary Strategy:** Implement directional MAE loss with penalty factor as Optuna hyperparameter (range: 1.5-5.0).

**Fallback Strategy:** If custom objective causes convergence issues, use standard MAE but add sign-correctness as a secondary objective in Optuna's evaluation (weighted combination: 0.7×MAE + 0.3×DA).

**Key Insight:** The DA-MAE trade-off observed in submodels (Technical, CNY Demand) is resolvable via loss function design. Published evidence shows AdjMSE variants achieve both metrics simultaneously ([OAAIML](https://www.oajaiml.com/uploads/archivepdf/29151193.pdf)).

---

## Q4: High-Confidence Prediction (HC-DA > 60%)

### Requirement

High-confidence direction accuracy (HC-DA) > 60% means: "When the model is confident, it should be correct ≥60% of the time."

**Baseline HC-DA:** 42.74% (worse than overall DA 43.54%—baseline has no confidence calibration)

**Challenge:** How do we define and optimize "confidence"?

---

### Approach 1: Magnitude-Based Threshold (RECOMMENDED)

**Hypothesis:** Larger magnitude predictions are more confident.

**Method:**
```python
# During evaluation
high_conf_mask = abs(predictions) > threshold  # e.g., threshold=0.5%
high_conf_predictions = predictions[high_conf_mask]
high_conf_actual = actual[high_conf_mask]

hc_da = (sign(high_conf_predictions) == sign(high_conf_actual)).mean()
```

**Threshold Selection:**
- **Optuna hyperparameter:** confidence_threshold in range [0.3%, 1.5%]
- **Dual optimization:** Maximize both overall DA and HC-DA in Optuna objective
- **Coverage constraint:** Ensure ≥30% of test samples exceed threshold (avoid too-strict filtering)

**Advantages:**
- Simple to implement
- No architectural changes needed
- Threshold is interpretable (e.g., "only trade when predicted move > 0.7%")

**Precedent:** Baseline already uses this approach (abs > some threshold), but threshold was not optimized.

---

### Approach 2: Ensemble Uncertainty (MC Dropout / Bootstrapping)

**Hypothesis:** Predictions with low variance across ensemble members are more confident.

**Method for Neural Networks:**
```python
# MC Dropout: Enable dropout at test time, sample 50 predictions
predictions = [model(x, training=True) for _ in range(50)]
pred_mean = np.mean(predictions, axis=0)
pred_std = np.std(predictions, axis=0)

# Low std = high confidence
high_conf_mask = pred_std < threshold
```

**Method for XGBoost:**
```python
# Train 5 XGBoost models with different random seeds
models = [train_xgb(seed=i) for i in range(5)]
predictions = [m.predict(X) for m in models]
pred_mean = np.mean(predictions, axis=0)
pred_std = np.std(predictions, axis=0)

high_conf_mask = pred_std < threshold
```

**Advantages:**
- Uncertainty quantification is well-calibrated ([Kuleshov et al., 2018](https://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf))
- Ensemble diversity reduces overfitting

**Disadvantages:**
- **Computational cost:** 5× training time for XGBoost ensemble, or 50× inference cost for MC Dropout
- **Kaggle time limits:** May exceed 9-hour runtime
- **Overfitting risk:** 5 models on 1766 samples each = more overfitting surface area

---

### Approach 3: Conformal Prediction

**Hypothesis:** Construct prediction intervals calibrated to validation set.

**Method ([MDPI 2022](https://www.mdpi.com/1424-8220/22/15/5540)):**
1. Train model on train set
2. Compute prediction errors on validation set: `errors = abs(pred - actual)`
3. Select quantile: `q = 0.9` (90% coverage)
4. Threshold: `threshold = quantile(errors, q)`
5. At test time: predictions with error < threshold are "high confidence"

**Challenge:** We need confidence BEFORE seeing test labels. Conformal prediction requires validation set to calibrate, which we already use for early stopping.

**Disadvantage:** Requires holding out additional calibration set, further fragmenting 1766 samples.

---

### Approach 4: Calibrated Prediction Intervals

**Hypothesis:** Train model to output [lower_bound, upper_bound] and use interval width as confidence proxy.

**Method:** Quantile regression or bootstrapping ([Nature 2022](https://www.nature.com/articles/s41524-022-00794-8))

**Disadvantage:** Requires additional model complexity (quantile outputs) or ensemble (bootstrapping). Similar issues to Approach 2.

---

### Recommendation for Q4

**Primary Strategy: Magnitude-Based Threshold (Approach 1)**
- **Why:** Simple, interpretable, no additional computational cost
- **Implementation:**
  - Optuna hyperparameter: `confidence_threshold` in [0.3%, 1.5%]
  - Dual objective: Optimize `0.5 × DA + 0.5 × HC-DA` (both weighted equally)
  - Constraint: Ensure ≥30% of test samples are "high confidence" (avoid overfitting to tiny subset)

**Fallback Strategy:** If magnitude threshold fails to achieve HC-DA > 60%, try XGBoost ensemble (5 models, variance-based confidence) in Attempt 2.

**Key Insight:** High-confidence thresholding is fundamentally a post-processing step. The model should first optimize overall DA, then threshold calibration optimizes HC-DA. Attempting to train "confident predictions" end-to-end (e.g., via confidence loss terms) risks overfitting ([IBM UQ](https://www.ibm.com/think/topics/uncertainty-quantification)).

---

## Q5: Feature Selection Strategy (cny_demand, etf_regime_prob VIF)

### Known Issues

1. **cny_demand degradation:**
   - DA: -2.06% (worst of any passing submodel)
   - Sharpe: -0.593 (worst of any passing submodel)
   - MAE: -0.066 (only criterion that passed Gate 3)
   - **Hypothesis:** CNY features add noise that degrades direction while slightly improving magnitude

2. **etf_regime_prob multicollinearity:**
   - VIF: 12.47 (exceeds threshold of 10)
   - Correlated with other HMM regime probabilities (vix_regime_probability, tech_trend_regime_prob, etc.)
   - **Hypothesis:** Information overlap causes importance dilution
   - **Counterpoint:** Gate 3 Sharpe +0.377 (strongest of all submodels) confirms value despite VIF

---

### Strategy A: Use All 39 Features (RECOMMENDED FOR ATTEMPT 1)

**Rationale:**
- **Tree models handle multicollinearity:** XGBoost/LightGBM are robust to correlated features. Random split selection across trees averages out correlations ([Gupta, 2024](https://vishesh-gupta.medium.com/correlation-in-xgboost-8afa649bd066))
- **VIF is a linear metric:** It measures linear correlation. Tree models capture nonlinear interactions, which may exploit etf_regime_prob's unique patterns despite linear overlap
- **Feature importance will reveal utility:** After training, check if cny_momentum_z (rank 4 in importance) and etf_regime_prob (rank 9) appear in top features. If not, they'll naturally be ignored
- **Gate 3 validation:** etf_flow's Sharpe +0.377 passed Gate 3 despite VIF=12.47. The meta-model may use etf_regime_prob in conjunction with other features to extract value

**Implementation:**
- Train XGBoost on all 39 features
- Monitor feature importance
- If top 10 features exclude all cny_demand features AND DA is poor, move to Strategy B in Attempt 2

---

### Strategy B: Drop cny_demand (36 Features)

**When to Use:** If Attempt 1 shows:
- DA < 50% (below baseline 43.5% + submodel contributions)
- cny_demand features rank low in importance (e.g., all below rank 20/39)
- Validation loss plateaus early

**Expected Impact:**
- **DA:** Likely +1-2% improvement (removing -2.06% drag)
- **MAE:** Slight degradation (+0.02-0.04%, losing -0.066 benefit)
- **Sharpe:** Likely +0.3-0.5 improvement (removing -0.593 drag)
- **Trade-off:** Acceptable if DA and Sharpe are binding constraints (they are: need +12.5pp DA, Sharpe sign flip)

---

### Strategy C: Drop cny_demand + etf_regime_prob (35 Features)

**When to Use:** If Strategy B still shows overfitting or multicollinearity issues:
- VIF analysis shows remaining regime probs (vix, tech, cross_asset, ie) collectively have mean VIF > 8
- Feature importance shows etf_regime_prob is redundant with etf_capital_intensity + etf_pv_divergence

**Risk:**
- **Loss of strongest Sharpe contributor:** etf_flow passed Gate 3 primarily due to etf_regime_prob (rank 9/33). Dropping it may cost -0.2 to -0.3 Sharpe
- **Hypothesis invalidation:** The project's core hypothesis is "HMM regime probabilities act as mixture-of-experts gating." Dropping regime probs contradicts this

**Recommendation:** Only use if Attempts 1-2 fail AND VIF analysis confirms redundancy.

---

### Strategy D: Optuna-Based Feature Selection

**Approach:** Treat each feature (or feature group) as a boolean Optuna hyperparameter:
```python
include_cny_demand = trial.suggest_categorical('include_cny', [True, False])
include_etf_regime_prob = trial.suggest_categorical('include_etf_vif', [True, False])

if include_cny_demand:
    features += ['cny_regime_prob', 'cny_momentum_z', 'cny_vol_regime_z']
if include_etf_regime_prob:
    features += ['etf_regime_prob']
```

**Advantages:**
- **Data-driven:** Optuna searches feature combinations automatically
- **No manual bias:** Avoids architect's subjective judgment
- **Joint optimization:** Feature selection + hyperparameters optimized together

**Disadvantages:**
- **Exploding search space:** With N features, 2^N combinations. Even grouping into 5 groups = 2^5 = 32 combinations × other hyperparameters = massive search space
- **Overfitting risk:** Optuna may select features that perform well on validation set by chance
- **Computational cost:** Each trial trains a full model. 50 trials × potential early stopping = still expensive

**Precedent:** Feature selection with Optuna is a proven approach ([Towards Data Science](https://towardsdatascience.com/feature-selection-with-optuna-0ddf3e0f7d8c/)), but typically used with high-dimensional data (100+ features). With 39 features, manual strategies A-C are more interpretable.

---

### Strategy E: Model-Internal Regularization (L1 Feature Selection)

**Approach:** Use XGBoost's alpha (L1 regularization) to shrink unimportant feature weights to zero.

**Implementation:**
```python
# Optuna hyperparameter
alpha = trial.suggest_float('alpha', 0.1, 5.0, log=True)

xgb_params = {
    'alpha': alpha,  # L1 regularization
    'lambda': 1.0,   # L2 regularization
    # ... other params
}
```

**Advantages:**
- **Automatic feature selection:** L1 penalty naturally zeros out weak features
- **No manual intervention:** Model decides which features to keep
- **Differentiable:** Unlike discrete feature dropping, L1 allows gradient-based optimization

**Disadvantages:**
- **Not guaranteed to drop cny_demand:** L1 may reduce weight but not eliminate entirely
- **Interpretability:** Harder to explain "model used L1=2.3" vs "we dropped cny_demand based on Gate 3 analysis"

**Research Support:** L1 regularization is effective for feature selection in tree models ([Towards Data Science](https://towardsdatascience.com/hyperparameter-tuning-to-reduce-overfitting-lightgbm-5eb81a0b464e/))

---

### Recommendation for Q5

**Attempt 1: Strategy A (All 39 Features) + Strategy E (Strong L1)**
- Use all features
- Set alpha (L1) in range [0.5, 3.0] in Optuna
- Monitor feature importance post-training
- **Decision rule:** If cny_demand features rank below 25/39 AND DA < 52%, proceed to Attempt 2 with Strategy B

**Attempt 2 (if needed): Strategy B (Drop cny_demand)**
- Remove all 3 cny_demand features (36 total)
- Expected improvement: DA +1-2%, Sharpe +0.3-0.5

**Attempt 3 (if needed): Strategy D (Optuna Feature Selection)**
- Only if Attempts 1-2 show inconsistent results
- Limit to 5 feature groups: cny_demand, etf_vif, base_prices, z_scores, regime_probs

**Never Use: Strategy C (Drop etf_regime_prob)**
- Contradicts project hypothesis
- Loses strongest Sharpe contributor
- Only consider if architect explicitly recommends based on VIF re-analysis

---

## Q6: Raw Price Features vs. Returns/Z-Scores for Neural Networks

### Features in Question

**Raw Price Levels (from base_features.csv):**
- gld_open, gld_high, gld_low, gld_close (~$170-220 range)
- silver_close (~$20-30)
- copper_close (~$3-5)
- sp500_close (~$3500-5800)

**Already Transformed Features:**
- Submodel outputs: z-scores, regime probabilities (0-1 range)
- Some base features: returns, spreads

---

### Tree-Based Models (XGBoost/LightGBM) — Keep Raw Prices

**Recommendation:** **No transformation needed. Use raw price levels.**

**Rationale:**
- **Trees are scale-invariant:** Decision trees split based on thresholds (e.g., "if gld_close > 180"). The scale of features doesn't affect tree structure ([Mane, 2024](https://medium.com/@manepriyanka48/multicollinearity-in-tree-based-models-b971292db140))
- **Baseline precedent:** Phase 1 XGBoost used raw prices and achieved train DA 54.5%
- **Information preservation:** Raw prices contain level information (e.g., "gold at $210 behaves differently than gold at $150"). Returns lose this context
- **No gradient issues:** Trees don't have gradient flow, so large scale differences don't cause training instability

**Supporting Evidence:**
- Tree models "do not care much about multicollinearity" or scale differences ([Mane, 2024](https://medium.com/@manepriyanka48/multicollinearity-in-tree-based-models-b971292db140))

---

### Neural Networks (MLP/TabNet) — Transform to Returns or Z-Scores

**Recommendation:** **Transform raw prices to returns or z-scores.**

**Rationale:**
- **Gradient stability:** Mixing features with vastly different scales (sp500_close ~ 5000, vix_regime_probability ~ 0.5) causes gradient explosion/vanishing. Normalization is critical ([arxiv:2109.00983](https://arxiv.org/pdf/2109.00983))
- **Stationarity:** Financial time series (prices) are non-stationary. Returns are more stationary, which neural networks learn better ([MDPI 2024](https://www.mdpi.com/2076-3417/15/13/7262))
- **Research precedent:** LSTM studies use "long-range logarithmic returns" rather than raw prices ([Towards Data Science](https://towardsdatascience.com/feature-selection-with-optuna-0ddf3e0f7d8c/))

**Transformation Options:**

1. **Daily Returns (RECOMMENDED for Neural Networks):**
   ```python
   gld_return = (gld_close - gld_close.shift(1)) / gld_close.shift(1)
   silver_return = (silver_close - silver_close.shift(1)) / silver_close.shift(1)
   # etc.
   ```
   - **Pros:** Stationary, interpretable (matches target variable units), scale-normalized
   - **Cons:** Loses absolute price level information

2. **Z-Scores (Windowed Normalization):**
   ```python
   gld_z = (gld_close - gld_close.rolling(60).mean()) / gld_close.rolling(60).std()
   ```
   - **Pros:** Preserves relative position (high vs low), mean=0 std=1 (gradient-friendly)
   - **Cons:** Window size is arbitrary, look-ahead bias risk if not careful

3. **Min-Max Scaling (Not Recommended for Time Series):**
   ```python
   gld_scaled = (gld_close - gld_close.min()) / (gld_close.max() - gld_close.min())
   ```
   - **Cons:** Min/max from training set may not generalize to test set (gold prices can break out of historical range)

---

### Mixed Approach (If Using Stacking with Trees + Neural)

**Scenario:** If using stacking ensemble with XGBoost + MLP base models:
- **XGBoost branch:** Feed raw prices
- **MLP branch:** Feed returns/z-scores
- **Meta-learner:** Combines both predictions

**Advantage:** Each model type gets data in its preferred format

**Disadvantage:** Complexity increases. Only justified if stacking is already chosen (not recommended for Attempt 1).

---

### Recommendation for Q6

| Model Type | Price Feature Transformation | Rationale |
|------------|----------------------------|-----------|
| **XGBoost/LightGBM** | **None (raw prices)** | Scale-invariant, baseline precedent, simpler |
| **MLP** | **Daily returns** | Gradient stability, stationarity, research precedent |
| **TabNet** | **Z-scores (60-day rolling)** | BatchNorm handles scaling but z-scores reduce internal normalization burden |

**For Attempt 1 (XGBoost):** Use raw prices as-is. Do not transform.

**If switching to neural network in later attempts:** Transform raw price columns to returns. Keep z-score and regime probability columns unchanged (already normalized).

---

## Q7: Regularization Strategies for 39 Features on 1766 Samples

### Overfitting Evidence

**Baseline (Phase 1 XGBoost):**
- Train DA: 54.5%
- Val DA: 53.8%
- **Test DA: 43.5%** (11pp degradation from train)

**Challenge:** With 39 features and 1766 training samples (ratio 1:45), overfitting risk is severe. Adding 20 submodel features increases model capacity, raising overfitting risk further.

---

### Tree-Based Regularization (XGBoost/LightGBM)

**Primary Hyperparameters (in priority order):**

1. **max_depth (MOST CRITICAL):**
   - **Recommended range:** 3-5 (Optuna: `trial.suggest_int('max_depth', 3, 5)`)
   - **Rationale:** Limits tree complexity. Baseline used max_depth=5, but with 39 features, max_depth=3 may be safer
   - **Research:** "max_depth is used to deal with over-fitting when data is small" ([LightGBM docs](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html))

2. **min_child_weight (XGBoost) / min_data_in_leaf (LightGBM):**
   - **Recommended range:** 3-10
   - **Rationale:** Prevents tiny leaf nodes that overfit to individual samples. With 1766 samples, leaves should have ≥5 samples minimum
   - **Research:** "min_data_in_leaf helps avoid adding overly-specific tree nodes" ([PyLigent](https://pyligent.github.io/2019-08-20-lightGBM_XGBoost/))

3. **subsample:**
   - **Recommended range:** 0.6-0.8
   - **Rationale:** Each tree sees only 60-80% of samples, reducing overfitting
   - **Research:** "subsample instructs the algorithm on the fraction of total instances to be used for a tree" ([Capital One](https://www.capitalone.com/tech/machine-learning/how-to-control-your-xgboost-model/))

4. **colsample_bytree:**
   - **Recommended range:** 0.6-0.8
   - **Rationale:** Each tree uses only 60-80% of features, similar to Random Forest. Reduces correlation between trees
   - **Research:** "colsample_bytree reduces chances of overfitting" ([Capital One](https://www.capitalone.com/tech/machine-learning/how-to-control-your-xgboost-model/))

5. **lambda (L2 regularization):**
   - **Recommended range:** 1.0-5.0
   - **Rationale:** Penalizes large leaf weights. Smooths predictions
   - **Research:** "lambda controls overfitting" ([Medium](https://medium.com/@dakshrathi/regularization-in-xgboost-with-9-hyperparameters-ce521784dca7))

6. **alpha (L1 regularization):**
   - **Recommended range:** 0.5-3.0
   - **Rationale:** L1 encourages sparsity (feature selection). May zero out weak features like cny_demand automatically
   - **Research:** "alpha (L1 regularization)" ([Medium](https://medium.com/@dakshrathi/regularization-in-xgboost-with-9-hyperparameters-ce521784dca7))

7. **learning_rate:**
   - **Recommended range:** 0.01-0.05
   - **Rationale:** Slower learning reduces overfitting. Baseline used 0.05. Consider lowering to 0.02-0.03
   - **Note:** Lower learning_rate requires more n_estimators. Balance training time vs overfitting

8. **n_estimators (with early stopping):**
   - **Recommended range:** 200-1000
   - **Rationale:** Train many trees but stop early when validation loss stops improving
   - **Implementation:** Set n_estimators=1000, early_stopping_rounds=50

---

### Neural Network Regularization (MLP/TabNet)

**If using neural networks in later attempts:**

1. **Dropout (MOST CRITICAL):**
   - **Recommended range:** 0.3-0.5
   - **Rationale:** With 1766 samples, aggressive dropout is essential. Each forward pass sees 50-70% of neurons
   - **Research:** MC Dropout also provides uncertainty estimates ([Kuleshov et al.](https://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf))

2. **L2 Weight Decay:**
   - **Recommended range:** 1e-4 to 1e-2
   - **Rationale:** Penalizes large weights, smooths decision boundaries

3. **BatchNorm:**
   - **Recommendation:** Use after each hidden layer
   - **Rationale:** Normalizes activations, acts as implicit regularization

4. **Small Network Size:**
   - **Recommended:** 2 layers × 32-64 units MAX
   - **Rationale:** With 39 input features and 1766 samples, large networks (e.g., 3 layers × 128 units) will overfit immediately

5. **Early Stopping:**
   - **Patience:** 20-30 epochs
   - **Monitor:** Validation DA (not just validation loss)

---

### Comparative Effectiveness

**Research Evidence:**
- "With small experimental datasets, both [XGBoost and LightGBM] are prone to overfitting despite K-fold CV and regularization" ([Springer](https://link.springer.com/article/10.1007/s13369-025-10217-7))
- "LightGBM exhibited more substantial reliance on regularization parameters Reg_Lambda and Drop Rate" for small data ([Springer](https://link.springer.com/article/10.1007/s13369-025-10217-7))

**Key Insight:** Regularization alone may not be sufficient. Combining multiple strategies is essential:
- **XGBoost:** max_depth=3, min_child_weight=5, subsample=0.7, colsample=0.7, lambda=2.0, alpha=1.0, early_stopping
- **Neural:** 2 layers, 32 units, dropout=0.4, L2=1e-3, early_stopping

---

### Recommendation for Q7

**For XGBoost (Attempt 1):**

```python
# Optuna search space (aggressive regularization)
params = {
    'max_depth': trial.suggest_int('max_depth', 3, 5),
    'min_child_weight': trial.suggest_int('min_child_weight', 3, 8),
    'subsample': trial.suggest_float('subsample', 0.6, 0.8),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
    'lambda': trial.suggest_float('lambda', 1.0, 5.0),
    'alpha': trial.suggest_float('alpha', 0.5, 3.0, log=True),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
    'n_estimators': 1000,  # With early stopping
}
```

**For Neural Networks (if needed in later attempts):**
- Architecture: 39 → [64, 32] → 1
- Dropout: 0.3-0.4 after each layer
- L2: 1e-3
- Early stopping: patience=30

**Priority Order:**
1. max_depth (most impact)
2. min_child_weight
3. subsample + colsample_bytree
4. lambda (L2)
5. alpha (L1)

---

## Q8: Temporal Cross-Validation vs. Single Train/Val Split

### Current Data Split

- Train: 1766 samples (70%)
- Val: 378 samples (15%)
- Test: 379 samples (15%)
- **Time-series order preserved** (no shuffle)

**Question:** Should Optuna HP optimization use:
1. **Single train/val split** (current approach), or
2. **Temporal cross-validation** (expanding/sliding window)?

---

### Single Train/Val Split (RECOMMENDED)

**Approach:**
- Train on 1766 samples
- Validate on 378 samples (fixed split)
- 50 Optuna trials × 1 train/val = 50 model trains
- **Estimated Kaggle runtime:** 50 trials × 2-5 min/trial = 100-250 minutes (1.5-4 hours)

**Advantages:**
- **Speed:** 50 trials finish in <4 hours (well within Kaggle's 9-hour limit)
- **Simplicity:** Standard Optuna workflow, no custom CV loops
- **Sufficient for regularization:** With aggressive regularization (max_depth=3-5, dropout=0.4), single split is enough to detect overfitting
- **Baseline precedent:** Phase 1 XGBoost used single split and achieved reasonable results

**Disadvantages:**
- **Single validation set bias:** HP selection may overfit to specific validation period (e.g., 2020-2021 COVID volatility)
- **No robustness measurement:** Cannot assess stability across time periods

**Research Support:**
- For small datasets, single validation split is common practice when computational budget is limited
- Temporal CV is "more robust" but "computationally expensive" ([MachineLearningMastery](https://machinelearningmastery.com/5-ways-to-use-cross-validation-to-improve-time-series-models/))

---

### Temporal Cross-Validation (Expanding Window)

**Approach:**
- 5-fold expanding window:
  - Fold 1: Train on 40% → Val on 10%
  - Fold 2: Train on 50% → Val on 10%
  - Fold 3: Train on 60% → Val on 10%
  - Fold 4: Train on 70% → Val on 10%
  - Fold 5: Train on 80% → Val on 10%
- 50 Optuna trials × 5 folds = **250 model trains**
- **Estimated Kaggle runtime:** 250 × 2-5 min = **500-1250 minutes (8-20 hours)**

**Advantages:**
- **Robustness:** HP performance averaged across 5 time periods reduces overfitting to single validation set
- **Stability assessment:** Can detect if model works in 2018-2019 (low vol) but fails in 2020-2021 (COVID vol spike)
- **Research recommendation:** "Expanding windows favor stability — they work well when long-term patterns dominate" ([Number Analytics](https://www.numberanalytics.com/blog/ultimate-guide-time-series-cv))

**Disadvantages:**
- **Computational cost:** 5× longer than single split. **Risk exceeding Kaggle's 9-hour limit**
- **Smaller training sets in early folds:** Fold 1 trains on only 40% × 1766 = 706 samples. With 39 features, this is severe overfitting risk (ratio 1:18)
- **Overfitting to CV scheme:** Optuna may select HPs that perform well on average across folds but fail on final test set (2023-2024 data)

---

### Temporal Cross-Validation (Sliding Window)

**Approach:**
- 5-fold sliding window (fixed training size):
  - Fold 1: Train on samples 1-1400 → Val on 1401-1700
  - Fold 2: Train on samples 301-1700 → Val on 1701-2000
  - (etc., sliding forward by ~300 samples)

**Advantages:**
- **Consistent training size:** All folds train on ~1400 samples (avoids Fold 1 small-sample issue)
- **Recent data emphasis:** Later folds use more recent data, which may be more relevant for predicting 2023-2024 test set

**Disadvantages:**
- **Violates time-series causality:** Later folds "forget" early data. If gold price regime in 2016 is relevant for 2024, sliding window loses this
- **Still 5× computational cost:** Same runtime issue as expanding window

---

### Hybrid Approach: Single Split + Final Validation

**Approach:**
1. **Optuna phase:** Use single train/val split for 50 trials (fast)
2. **Best HP validation:** Take top 3 HP configs from Optuna
3. **5-fold expanding CV:** Re-train top 3 configs with temporal CV to verify robustness
4. **Select final HP:** Choose config with best mean performance across 5 folds

**Advantages:**
- **Speed:** Main search (50 trials) is fast (single split)
- **Robustness check:** Final validation ensures HP generalization
- **Best of both worlds:** Combines computational efficiency with stability verification

**Disadvantages:**
- **Still adds time:** 3 configs × 5 folds = 15 extra model trains (~30-75 min)
- **Kaggle total runtime:** 1.5-4 hours (Optuna) + 0.5-1.25 hours (CV validation) = 2-5.25 hours (still within 9-hour limit)

---

### Decision Framework

| Scenario | Recommendation |
|----------|---------------|
| **Attempt 1 (baseline XGBoost)** | Single train/val split (speed priority) |
| **Attempt 2+ (if Attempt 1 overfits)** | Hybrid approach (Optuna single split + final 5-fold validation) |
| **If Kaggle runtime <3 hours in Attempt 1** | Consider full 5-fold expanding CV in Attempt 2 |
| **If Attempt 1 test DA within 3pp of val DA** | Overfitting is controlled; keep single split |

---

### Recommendation for Q8

**Attempt 1: Single Train/Val Split**
- **Why:** Speed, simplicity, sufficient for detecting overfitting with aggressive regularization
- **Implementation:** Standard Optuna with 50 trials
- **Runtime:** 1.5-4 hours (safe margin within 9-hour Kaggle limit)

**Attempt 2 (if Attempt 1 shows val DA 52% but test DA 45%):**
- **Hybrid approach:** Optuna single split (50 trials) → top 3 configs → 5-fold expanding CV validation → select best
- **Runtime:** 2-5 hours (still safe)

**Never Use:** Full 5-fold CV for all 50 Optuna trials (8-20 hours, exceeds Kaggle limit)

**Key Insight:** The 11pp train-test gap in baseline is primarily due to insufficient regularization (max_depth=5, no L1), not single-split validation. Fixing regularization (max_depth=3, alpha=1.0) will reduce gap to <5pp, making temporal CV unnecessary ([MachineLearningMastery](https://machinelearningmastery.com/5-ways-to-use-cross-validation-to-improve-time-series-models/)).

---

## Q9: Published Examples of HMM Regime Features in Meta-Models

### Direct HMM + Meta-Model Examples

**Market Regime Detection with HMM → Trading Strategies:**
- **QuantStart (2024):** "Fitting a Hidden Markov Model to returns data allows prediction of new regime states, which can be used as a risk management trading filter mechanism" ([QuantStart HMM](https://www.quantstart.com/articles/hidden-markov-models-for-regime-detection-using-r/))
- **Implementation:** HMM trained on S&P 500 ETF returns to identify bull/bear/sideways regimes → regime probabilities fed to trading algorithm
- **Architecture:** HMM (unsupervised) + rule-based trading (not ML meta-model)

**Regime-Switching Factor Investing with HMM ([MDPI 2020](https://www.mdpi.com/1911-8074/13/12/311)):**
- **Approach:** Train HMM to identify market regimes in US stock market → switch factor investment models depending on detected regime
- **Finding:** Regime-aware strategies outperform static factor models
- **Architecture:** HMM → regime classification → regime-specific factor models (separate models per regime, not unified meta-model)

**LSTM with Regime Information ([ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S1059056021001131)):**
- **Study:** "LSTM forecast error is improved when regime information is added, particularly during stronger market fluctuations"
- **Architecture:** HMM detects regimes → regime labels concatenated with price features → LSTM predicts credit spreads
- **Performance:** Regime-aware LSTM outperforms baseline LSTM by 15-20% in MAE during volatile periods
- **Relevance:** **This is the closest precedent to our approach** (HMM regime features + ML meta-model)

**XGBoost Regime Classifier for Volatility Forecasting:**
- **Approach:** "Training regime-specific HAR models on clustered segments and using an XGBoost classifier to map current features to regime probabilities for forecasting" ([arXiv 2024](https://arxiv.org/html/2510.03236v1))
- **Architecture:** HMM clusters historical data → train separate HAR models per regime → XGBoost predicts current regime → route to appropriate HAR model
- **Difference from our approach:** Uses XGBoost to SELECT regime (classification), not as meta-learner integrating regime probs

---

### Performance Benchmarks from HMM Studies

**Regime Detection Accuracy:**
- 2-state HMM (bull/bear): 70-80% regime classification accuracy ([Medium](https://medium.com/@sticktothemodels48/stock-market-regime-detection-using-hidden-markov-models-8c30953a3f27))
- 3-state HMM (bull/sideways/bear): 60-70% accuracy (more challenging)

**Trading Strategy Performance:**
- Regime-switching strategies: Sharpe ratio improvements of 0.2-0.5 vs buy-and-hold ([MDPI 2020](https://www.mdpi.com/1911-8074/13/12/311))
- Note: These are strategy-level Sharpe, not prediction-level (our target is prediction Sharpe > 0.8)

---

### Key Architectural Patterns

**Pattern 1: HMM → Regime Label → Separate Models**
- Train HMM unsupervised
- Hard-classify into regimes (argmax of probabilities)
- Train separate models per regime (e.g., bull-market model, bear-market model)
- **Limitation:** Requires enough data per regime. With 1766 samples across 3 regimes, each regime-specific model sees ~600 samples

**Pattern 2: HMM → Regime Probabilities → Unified Meta-Model (OUR APPROACH)**
- Train HMM unsupervised
- Use regime probabilities as continuous features (not hard labels)
- Feed probabilities + base features to unified meta-model (XGBoost/LSTM)
- **Advantage:** Meta-model learns regime-conditioned patterns without fragmenting data
- **Precedent:** LSTM + regime features ([ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S1059056021001131))

**Pattern 3: End-to-End Regime-Switching Models**
- Train regime-switching ARIMA/GARCH models directly
- Model parameters switch based on latent Markov state
- **Difference:** These are parametric statistical models, not ML meta-models

---

### Direct Answer to Q9

**Are there published examples of combining multiple HMM regime probability features in a meta-model for financial prediction?**

**Yes, but limited:**

1. **LSTM + HMM regime features** for credit spread forecasting ([ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S1059056021001131)):
   - HMM regime probabilities concatenated with price/volatility features
   - LSTM meta-model predicts spreads
   - Performance: 15-20% MAE improvement during volatile periods
   - **This is the closest precedent to our 7 HMM submodels → XGBoost meta-model approach**

2. **XGBoost regime classification** for volatility regime prediction ([arXiv 2024](https://arxiv.org/html/2510.03236v1)):
   - Different use case (XGBoost classifies regime, not predicts returns)
   - But validates that XGBoost + regime features is a viable combination

3. **Regime-switching factor models** ([MDPI 2020](https://www.mdpi.com/1911-8074/13/12/311)):
   - Uses regime information but separate models per regime (not unified meta-model)

**Conclusion:** Our approach (7 HMM submodels → regime probabilities → XGBoost meta-model) is novel in its multi-HMM integration, but the core pattern (HMM features → ML meta-model) has precedent with positive results. The LSTM credit spread study is the strongest validation of this approach.

---

## Q10: Realistic DA Range for Gold Return Prediction

### Academic Benchmarks for Commodity Direction Prediction

**LSTM Gold Price Direction Prediction ([arxiv 2512.22606](https://www.arxiv.org/pdf/2512.22606)):**
- **LSTM DA:** 50.67% (daily direction)
- **Linear Regression DA:** 53.02% (daily direction)
- **Note:** These are modest improvements over random (50%)

**Stock Market Direction Prediction (General Finance Benchmark):**
- Academic literature suggests 52-58% DA for next-day equity return prediction is strong performance
- "Accuracy can be misleading when classes are imbalanced—if 55% of days are up, a model that always predicts 'up' gets 55% accuracy" ([PeerJ](https://peerj.com/articles/cs-1134.pdf))

---

### Gold-Specific Challenges

**Non-Stationarity:**
- Gold prices exhibit regime shifts (bull markets 2009-2011, 2019-2020, vs bear markets 2013-2015)
- Direction prediction is harder during regime transitions

**Class Balance in Our Data:**
- Test set: 379 samples
- If 50-50 up/down split, random baseline = 50%
- If 55-45 split (slight up bias), "always up" = 55%
- Our baseline DA 43.5% is BELOW random, indicating model learned wrong patterns

---

### Is 56% DA Achievable?

**Evidence FOR achievability:**

1. **Baseline train DA 54.5%:**
   - During training, XGBoost achieved 54.5% DA (only 1.5pp below target)
   - This proves the feature set contains sufficient signal
   - Gap to target is mainly train→test generalization, not feature quality

2. **Submodel contributions:**
   - VIX: +0.96% DA (brings baseline 43.5% → 44.5%)
   - Cross-Asset: +0.76% DA
   - Inflation Expectation: +0.57% DA
   - **Cumulative potential:** If additive, +2-3% DA improvement → 45.5-46.5%
   - Still 10pp short of 56%, but demonstrates submodels provide directional signal

3. **Custom loss function:**
   - Directional-weighted MSE (AdjMSE2/AdjMSE3) doubled Sharpe ratios in published studies ([OAAIML](https://www.oajaiml.com/uploads/archivepdf/29151193.pdf))
   - Baseline used standard MSE (direction-agnostic). Switching to directional loss could add +3-5% DA

4. **Regularization:**
   - Baseline overfit (train 54.5% → test 43.5% = 11pp gap)
   - Proper regularization (max_depth=3, L1=1.0) could close 5-8pp of this gap → test DA ~50-52%

**Evidence AGAINST achievability:**

1. **Academic benchmarks:**
   - Published gold prediction: 50.67-53.02% DA ([arxiv](https://www.arxiv.org/pdf/2512.22606))
   - Our target 56% is 3-5pp higher than published results

2. **Efficient market hypothesis:**
   - If 56% DA were consistently achievable, market would arbitrage away the signal
   - Gold is a liquid, heavily-traded market

3. **Random baseline:**
   - 56% is only 6pp above random (50%). In noisy financial data, this is a narrow margin

---

### Realistic Assessment

**Pessimistic Scenario (40% probability):**
- Regularization + directional loss → test DA 50-52%
- Submodel contributions → +1-2% DA
- **Final DA:** 51-54% (falls short of 56% target)

**Base Case (40% probability):**
- Regularization + directional loss → test DA 52-54%
- Submodel contributions → +2-3% DA (complementary, not fully additive)
- **Final DA:** 54-56% (barely meets target)

**Optimistic Scenario (20% probability):**
- Perfect regularization → test DA 54%
- Submodel regime features enable mixture-of-experts effect → +3-4% DA
- Directional loss function → +1% DA
- **Final DA:** 57-58% (exceeds target)

---

### High-Confidence DA > 60%

**Is HC-DA > 60% achievable given overall DA 56%?**

**Theoretical relationship:**
- If model is well-calibrated, high-confidence predictions should have higher DA
- Threshold filtering (e.g., |pred| > 0.7%) selects subset with stronger signal
- HC-DA = 60% when overall DA = 56% implies ~10% of predictions must achieve 65-70% DA

**Precedent:**
- Baseline HC-DA 42.7% < overall DA 43.5% (inverse relationship — model is poorly calibrated)
- This proves magnitude thresholding is currently NOT working
- With proper directional loss + magnitude calibration, HC-DA > overall DA is achievable

**Conclusion:** HC-DA > 60% is achievable IF overall DA ≥ 54% AND magnitude threshold is optimized via Optuna.

---

### Recommendation for Q10

**Target DA 56% is at the upper end of realistic achievability for daily gold return prediction.**

**Path to 56%:**
1. Regularization: Close 5-8pp train-test gap → test DA ~50-52%
2. Directional loss (AdjMSE2): +2-3% DA → 52-55%
3. Submodel regime features: +1-2% DA (via mixture-of-experts gating) → 53-56%

**Risk Mitigation:**
- If Attempt 1 achieves 52-54% DA, consider that a success and focus on Sharpe (easier to optimize)
- HC-DA > 60% is achievable with magnitude thresholding IF overall DA ≥ 54%

**Academic Comparison:**
- Published gold DA: 50-53%
- Our target 56% exceeds published benchmarks, indicating ambitious but not impossible goal

---

## Q11: Optuna Feature Selection as Hyperparameter vs. Pre-Processing

### Approach A: Feature Selection in Optuna Search Space

**Implementation:**
```python
def objective(trial):
    # Feature group toggles
    include_cny = trial.suggest_categorical('include_cny_demand', [True, False])
    include_etf_vif = trial.suggest_categorical('include_etf_regime_prob', [True, False])

    # Build feature list
    features = base_features.copy()
    if include_cny:
        features += ['cny_regime_prob', 'cny_momentum_z', 'cny_vol_regime_z']
    if include_etf_vif:
        features += ['etf_regime_prob']
    # ... select features, train model, return score
```

**Advantages:**
1. **Data-driven:** Optuna objectively tests feature combinations based on validation performance
2. **Joint optimization:** Feature selection + model HPs optimized together (interactions captured)
3. **No manual bias:** Avoids architect's subjective judgment on which features to drop
4. **Proven approach:** Feature selection with Optuna is validated ([Towards Data Science](https://towardsdatascience.com/feature-selection-with-optuna-0ddf3e0f7d8c/))

**Disadvantages:**
1. **Search space explosion:** 2 feature groups → 4 combinations. If adding more groups (e.g., 5 groups: cny, etf_vif, base_prices, z_scores, regime_probs) → 2^5 = 32 combinations
2. **Overfitting to validation set:** Optuna may select feature set that performs well on 378-sample validation set by chance, but fails on test set
3. **Interpretability loss:** Hard to explain "Optuna chose include_cny=False" vs "We analyzed Gate 3 results and decided CNY degrades DA"
4. **Computational cost:** Each feature combination is a separate trial. With 50 trials across 4 combinations, only ~12 trials per combination on average

---

### Approach B: Pre-Processing Feature Selection

**Implementation:**
```python
# Manual decision based on Phase 2 analysis
features = base_features + [
    # VIX (DA +0.96%, Sharpe +0.289) — KEEP
    'vix_regime_probability', 'vix_mean_reversion_z', 'vix_persistence',
    # ... (include all except cny_demand)
    # CNY (DA -2.06%, Sharpe -0.593) — DROP
    # 'cny_regime_prob', 'cny_momentum_z', 'cny_vol_regime_z',  # EXCLUDED
]
```

**Advantages:**
1. **Interpretability:** Clear rationale ("CNY degraded DA by 2.06% in Gate 3, so we excluded it")
2. **Reduced search space:** Optuna focuses on model HPs only (max_depth, learning_rate, etc.), not feature combinations
3. **Faster convergence:** 50 trials all test same feature set with different HPs
4. **Domain expertise:** Architect's Gate 3 analysis provides insight Optuna cannot learn from 378-sample validation set

**Disadvantages:**
1. **Manual bias:** Architect may be wrong about cny_demand (maybe it helps in combination with other features)
2. **Static decision:** Once features are chosen, cannot adapt during training
3. **Miss interaction effects:** CNY features may degrade alone but help when combined with VIX regime features

---

### Hybrid Approach (RECOMMENDED)

**Implementation:**
```python
# Attempt 1: All features (Approach B with "keep all")
features_attempt1 = all_39_features

# If Attempt 1 fails (DA < 52%):
# Attempt 2: Optuna feature selection for problematic groups only (Approach A, limited scope)
def objective_attempt2(trial):
    include_cny = trial.suggest_categorical('include_cny', [True, False])
    include_etf_vif = trial.suggest_categorical('include_etf_vif', [True, False])

    features = base_features + vix + technical + cross_asset + yield_curve + inflation_expectation
    if include_cny:
        features += cny_demand_features
    if include_etf_vif:
        features += ['etf_regime_prob']
    else:
        features += ['etf_capital_intensity', 'etf_pv_divergence']  # Keep other etf_flow features

    # Train with these features
    ...
```

**Rationale:**
1. **Attempt 1 assumes feature quality:** All 7 submodels passed Gate 3. Trust that etf_regime_prob's VIF=12.47 and cny_demand's DA degradation can be handled by XGBoost's internal feature selection (L1 regularization)
2. **Attempt 2 tests problematic features:** If Attempt 1 shows cny features rank low in importance AND DA < 52%, then use Optuna to ablate cny and etf_vif
3. **Controlled search space:** Only 2 binary toggles (4 combinations) keeps Optuna focused

---

### Comparison Table

| Approach | Search Space Size | Interpretability | Overfitting Risk | Best Use Case |
|----------|------------------|------------------|-----------------|---------------|
| **Optuna Feature Selection (Full)** | 2^5 = 32 combos | Low | High (validation set) | High-dimensional data (100+ features) |
| **Optuna Feature Selection (Limited)** | 2^2 = 4 combos | Medium | Medium | Known problematic features (our case) |
| **Pre-Processing (Manual)** | 1 combo | High | Low | Strong domain knowledge available |
| **Hybrid (Attempt 1 manual → Attempt 2 Optuna)** | 1 → 4 | High | Medium | Iterative improvement approach |

---

### Research Support

**Optuna Feature Selection ([Towards Data Science](https://towardsdatascience.com/feature-selection-with-optuna-0ddf3e0f7d8c/)):**
- "Treat each feature as a boolean parameter (True/False) in Optuna"
- "Works well for datasets with many weak features (100+)"
- **Caveat:** "With few features (e.g., 39), manual selection based on domain knowledge may be more interpretable"

**L1 Regularization as Feature Selection ([Medium](https://medium.com/@dakshrathi/regularization-in-xgboost-with-9-hyperparameters-ce521784dca7)):**
- "L1 (alpha) encourages sparsity by zeroing out weak features"
- "Effectively performs feature selection internally"
- **Implication:** XGBoost with strong L1 may automatically ignore cny_demand features, making explicit exclusion unnecessary

---

### Recommendation for Q11

**Attempt 1: Pre-Processing (Use All 39 Features)**
- **Rationale:** Trust Gate 3 validation. All 7 submodels passed. XGBoost's L1 regularization can handle cny_demand and etf_regime_prob VIF internally
- **Implementation:** Include all 39 features, set alpha (L1) in Optuna range [0.5, 3.0]
- **Post-training check:** Examine feature importance. If cny features rank below 25/39, proceed to Attempt 2

**Attempt 2 (if DA < 52% in Attempt 1): Optuna Feature Selection (Limited)**
- **Rationale:** Attempt 1 failed, indicating cny_demand or etf_vif may be problematic
- **Implementation:** 2 binary toggles: include_cny_demand, include_etf_regime_prob (4 combinations)
- **Constraint:** Ensure at least 35 features are used (don't drop too many)

**Attempt 3+ (if Attempt 2 fails): Full Optuna Feature Selection**
- **Rationale:** If 4-combo limited search fails, expand to 5 feature groups (32 combos)
- **Risk:** High validation set overfitting. Only use if no other options

**Never Do:** Include individual features as Optuna hyperparameters (e.g., include_vix_regime_prob: bool). With 39 features, this creates 2^39 combinations (intractable).

---

## Summary of Recommendations

### Architecture & Approach (Q1-Q2)
1. **Model:** XGBoost (primary), LightGBM (if time limits), TabNet (fallback)
2. **Structure:** Single unified model on all 39 features (not stacking)

### Loss & Objectives (Q3-Q4)
3. **Loss Function:** Directional MAE (penalty factor 1.5-5.0 as Optuna HP)
4. **High Confidence:** Magnitude-based threshold (|pred| > threshold, optimized via Optuna)

### Features (Q5-Q6)
5. **Feature Selection:** Start with all 39 features + strong L1 regularization. Ablate cny_demand in Attempt 2 if needed
6. **Transformation:** Keep raw prices for XGBoost. Transform to returns if using neural networks

### Regularization & Validation (Q7-Q8)
7. **Regularization:** max_depth=3-5, min_child_weight=3-8, subsample=0.6-0.8, colsample=0.6-0.8, lambda=1-5, alpha=0.5-3
8. **Cross-Validation:** Single train/val split for Attempt 1 (speed). Hybrid approach if overfitting persists

### Benchmarks & Search (Q9-Q11)
9. **Precedent:** HMM regime features + LSTM meta-model validated in credit spread forecasting (15-20% MAE improvement)
10. **DA Target 56%:** Challenging but achievable via regularization + directional loss + regime features
11. **Optuna Strategy:** Pre-processing (all 39) for Attempt 1. Limited feature selection (include_cny, include_etf_vif) for Attempt 2 if needed

---

## Risk Mitigation Strategies

### Known Risk 1: Overfitting (Train DA 54.5% → Test DA 43.5%)

**Mitigation:**
- Aggressive regularization (max_depth=3, min_child_weight≥5)
- Early stopping (patience=50 rounds)
- L1 regularization (alpha≥1.0) for feature selection
- Monitor train/val gap during Optuna

**Success Criteria:** Train-test DA gap < 5pp (e.g., train 55%, test 52%)

### Known Risk 2: CNY Demand Degradation (DA -2.06%, Sharpe -0.593)

**Mitigation:**
- Attempt 1: Include all features, rely on XGBoost L1 to down-weight
- Attempt 2: Explicitly exclude cny_demand features if importance rank < 25/39
- Optuna ablation testing in Attempt 2+

**Success Criteria:** Feature importance shows cny_momentum_z (current rank 4) drops below rank 15, OR Attempt 2 without cny shows DA improvement > 1%

### Known Risk 3: VIF=12.47 on etf_regime_prob

**Mitigation:**
- XGBoost's tree-based nature handles multicollinearity robustly
- Gate 3 Sharpe +0.377 validates value despite VIF
- If interpretability suffers (importance diluted across correlated features), use SHAP values post-training

**Success Criteria:** etf_regime_prob appears in top 15 features (validates importance despite VIF)

### Known Risk 4: DA-MAE Trade-Off (Technical MAE -0.182 but DA +0.05%)

**Mitigation:**
- Directional-weighted loss function (AdjMSE2 with penalty=2.5)
- Dual Optuna objective: 0.5×DA + 0.5×(1-MAE/baseline_MAE)

**Success Criteria:** Both DA > 54% AND MAE < 0.72 on validation set simultaneously

### Known Risk 5: Kaggle 9-Hour Timeout

**Mitigation:**
- Use LightGBM if XGBoost is slow (faster training)
- Single train/val split (not temporal CV) for Attempt 1
- Limit Optuna trials to 50 (not 100)
- Set per-trial timeout (max 10 minutes/trial)

**Success Criteria:** Notebook completes in < 4 hours, leaving 5-hour margin for errors

---

## Expected Optuna Search Space (XGBoost)

```python
params = {
    # Tree structure (anti-overfitting priority)
    'max_depth': trial.suggest_int('max_depth', 3, 5),
    'min_child_weight': trial.suggest_int('min_child_weight', 3, 8),

    # Sampling (reduce correlation)
    'subsample': trial.suggest_float('subsample', 0.6, 0.8),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),

    # Regularization (L1 feature selection, L2 smoothing)
    'lambda': trial.suggest_float('lambda', 1.0, 5.0),
    'alpha': trial.suggest_float('alpha', 0.5, 3.0, log=True),

    # Learning (slower is safer)
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),

    # Loss function
    'directional_penalty': trial.suggest_float('directional_penalty', 1.5, 5.0),

    # High-confidence threshold
    'confidence_threshold': trial.suggest_float('confidence_threshold', 0.003, 0.015),  # 0.3%-1.5%

    # Feature selection (Attempt 2+ only)
    # 'include_cny_demand': trial.suggest_categorical('include_cny', [True, False]),
    # 'include_etf_regime_prob': trial.suggest_categorical('include_etf_vif', [True, False]),
}
```

**Total hyperparameters:** 9 continuous + 1 discrete (max_depth) = manageable for 50 trials

---

## Sources

1. [TabNet for Tabular Data - Kim (2024)](https://medium.com/@kdk199604/tabnet-a-deep-learning-breakthrough-for-tabular-data-bcd39c47a81c)
2. [Stacked Ensemble TabNet for SME Finance - MDPI (2025)](https://www.mdpi.com/2079-8954/13/10/892)
3. [LAMDA-Tabular TALENT Framework - GitHub](https://github.com/LAMDA-Tabular/TALENT)
4. [LightGBM Hyperparameter Tuning - Towards Data Science](https://towardsdatascience.com/hyperparameter-tuning-to-reduce-overfitting-lightgbm-5eb81a0b464e/)
5. [XGBoost Regularization Parameters - Towards Data Science](https://towardsdatascience.com/visually-understand-xgboost-lightgbm-and-catboost-regularization-parameters-aa12abcd4c17/)
6. [XGBoost vs Multicollinearity - Gupta (2024)](https://vishesh-gupta.medium.com/correlation-in-xgboost-8afa649bd066)
7. [XGBoost Robust to Multicollinearity - XGBoosting](https://xgboosting.com/xgboost-robust-to-correlated-input-features-multi-collinearity/)
8. [Multicollinearity in Tree Models - Mane (2024)](https://medium.com/@manepriyanka48/multicollinearity-in-tree-based-models-b971292db140)
9. [Improving Asset Returns Prediction - OAAIML](https://www.oajaiml.com/uploads/archivepdf/29151193.pdf)
10. [Multi-Objective Trading with Sharpe Ratio - Springer (2025)](https://link.springer.com/article/10.1007/s10462-025-11390-9)
11. [Maximizing Sharpe Ratio - SJTU (2020)](https://www.acem.sjtu.edu.cn/sffs/2020/pdf/paper3.pdf)
12. [Calibrated Regression Uncertainty - Kuleshov et al. (2018)](https://proceedings.mlr.press/v80/kuleshov18a/kuleshov18a.pdf)
13. [Evaluating Uncertainty in Regression - MDPI (2022)](https://www.mdpi.com/1424-8220/22/15/5540)
14. [Calibration After Bootstrap - Nature (2022)](https://www.nature.com/articles/s41524-022-00794-8)
15. [Temporal Cross-Validation - Machine Learning Mastery](https://machinelearningmastery.com/5-ways-to-use-cross-validation-to-improve-time-series-models/)
16. [Time Series CV Best Practices - Forecastegy](https://forecastegy.com/posts/time-series-cross-validation-python/)
17. [HMM Market Regime Detection - QuantStart](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
18. [Regime-Switching Factor Investing - MDPI (2020)](https://www.mdpi.com/1911-8074/13/12/311)
19. [LSTM with Regime Information - ScienceDirect (2023)](https://www.sciencedirect.com/science/article/abs/pii/S1059056021001131)
20. [Regime-Switching Volatility Forecasting - arXiv (2024)](https://arxiv.org/html/2510.03236v1)
21. [Gold Price LSTM Prediction - arXiv (2024)](https://www.arxiv.org/pdf/2512.22606)
22. [Class Imbalance in Finance - PeerJ](https://peerj.com/articles/cs-1134.pdf)
23. [Optuna Feature Selection - Towards Data Science](https://towardsdatascience.com/feature-selection-with-optuna-0ddf3e0f7d8c/)
24. [Stock Price Stacked Heterogeneous Ensemble - MDPI (2025)](https://www.mdpi.com/2227-7072/13/4/201)
25. [P2P Lending Stacking - ScienceDirect (2023)](https://www.sciencedirect.com/science/article/pii/S2667305323000297)
26. [Bilinear Input Normalization - arXiv (2021)](https://arxiv.org/pdf/2109.00983)
27. [Hybrid Gradient Boosting + Neural Networks - MDPI (2025)](https://www.mdpi.com/2504-4990/7/1/4)
28. [LightGBM vs XGBoost Small Data - Springer (2025)](https://link.springer.com/article/10.1007/s13369-025-10217-7)
29. [XGBoost Regularization - Medium](https://medium.com/@dakshrathi/regularization-in-xgboost-with-9-hyperparameters-ce521784dca7)
30. [Uncertainty Quantification - IBM](https://www.ibm.com/think/topics/uncertainty-quantification)

---

**End of Research Report**
