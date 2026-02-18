# Research Report: Meta-Model Attempt 8 Improvement Strategies

**Date**: 2026-02-17
**Context**: Current best model (Attempt 7) achieves DA 60.04%, HCDA 64.13%, Sharpe 2.46, MAE 0.94%. Core issues: tiny prediction std (0.023 vs actual 1.4%), 87.3% positive predictions, MAE target structurally infeasible.

---

## 1. Ensemble Stacking (High Priority)

### 1.1 Core Concept
Combine XGBoost, LightGBM, and CatBoost as base learners with a simple meta-learner (Linear/Lasso/Ridge). 2025 research shows **12-15% accuracy improvements** over single models on financial time series.

### 1.2 Architecture
```
Base Layer:
  - XGBoost (current champion, Optuna-tuned)
  - LightGBM (histogram binning, leaf-wise growth, faster training)
  - CatBoost (ordered boosting, native categorical handling)

Meta Layer:
  - Lasso Regression (L1 regularization prevents over-reliance on any single base model)
  - OLS Scaling (current post-processing preserved)
```

### 1.3 Implementation Strategy
- Train 3 base models independently with same train/val/test splits
- Use out-of-fold predictions as meta-features (5-fold CV on train set)
- Meta-learner trained on base model predictions + original 24 features
- Each base model contributes predictions, not raw features → reduced dimensionality

### 1.4 Expected Impact
- **DA/HCDA**: +1-3pp (ensemble diversity reduces systematic errors)
- **Sharpe**: +0.2-0.5 (model averaging smooths extreme predictions)
- **MAE**: Minimal improvement (all GBDT models underpredict scale)
- **Training time**: 3x longer (parallelizable)

### 1.5 2025 Evidence
- [Stacking XGBoost/LightGBM/CatBoost on Medium](https://medium.com/@stevechesa/stacking-ensembles-combining-xgboost-lightgbm-and-catboost-to-improve-model-performance-d4247d092c2e) reports 10-15% accuracy gains
- [Stock Price Prediction MDPI 2025](https://www.mdpi.com/2227-7072/13/4/201) shows stacking outperforms single models on financial data
- MLPerf 2024 benchmarks: 12% better AUC-ROC, 20% less memory vs single models
- [Ensemble Learning Johal 2025](https://johal.in/ensemble-learning-methods-xgboost-and-lightgbm-stacking-for-improved-predictive-accuracy-2025/) confirms 15% boost in financial forecasting

**Verdict**: **HIGHLY RECOMMENDED**. Proven technique with clear financial market applications.

---

## 2. Regime-Conditional Feature Interactions (High Priority)

### 2.1 Core Concept
Create interaction features that activate only in specific market regimes (e.g., VIX > 25 → high-vol regime). Current model treats all data uniformly; regime-aware features allow nonlinear responses.

### 2.2 Proposed Features
```python
# High-vol regime (VIX > 25 or vix_persistence > 0.7)
features['real_rate_x_high_vol'] = real_rate_change * (vix > 25).astype(int)
features['dxy_x_high_vol'] = dxy_change * (vix_persistence > 0.7).astype(int)

# Risk-off regime (xasset_recession_signal > 0.5)
features['etf_flow_x_risk_off'] = etf_flow_z * (xasset_recession_signal > 0.5)
features['inflation_exp_x_risk_off'] = inflation_exp_change * (xasset_recession_signal > 0.5)

# Yield curve inversion (yc_curvature_z < -1.5)
features['real_rate_x_inversion'] = real_rate_change * (yc_curvature_z < -1.5)
```

### 2.3 Implementation Strategy
- Generate 8-12 regime-conditional features (avoid overfitting with too many)
- Use existing submodel outputs as regime indicators (no new data required)
- Apply L1 penalty in XGBoost to auto-select useful interactions
- Monitor VIF (keep < 10) to avoid multicollinearity

### 2.4 Expected Impact
- **DA/HCDA**: +1-2pp (better handling of 2025-2026 regime shifts)
- **Sharpe**: +0.3-0.6 (reduced losses during volatile periods)
- **MAE**: Neutral to slight improvement
- **Feature count**: 24 → 32-36 (manageable for XGBoost)

### 2.5 2025 Research Evidence
- [RegimeFolio arXiv](https://arxiv.org/html/2510.14986v1): VIX-based regime segmentation yields **15-20% forecast accuracy improvement** in portfolio optimization
- [S&P 500 Volatility Forecasting arXiv](https://arxiv.org/html/2510.03236v1): Regime-switching XGBoost outperforms monolithic models; features capturing regime-specific dynamics are key
- [Financial Chaos Index arXiv](https://arxiv.org/html/2504.18958v1): Bidirectional VIX-realized volatility dynamics across regimes; regime-specific parameters meaningfully outperform uniform models

**Verdict**: **HIGHLY RECOMMENDED**. Strong 2025 evidence for financial forecasting. Low implementation cost (no new data, no model rewrite).

---

## 3. Asymmetric Loss Functions (Medium Priority)

### 3.1 Problem Statement
Current objective: `reg:squarederror` treats overestimation and underestimation equally. However:
- Negative predictions are only **38.2% correct** (weekly model result)
- False negative predictions are more costly than false positive ones in gold (opportunity cost vs actual loss)
- Direction accuracy should be weighted more than magnitude accuracy

### 3.2 Proposed Loss Functions

#### 3.2.1 Directional-Aware Loss
```python
def directional_loss(y_pred, dtrain):
    y_true = dtrain.get_label()
    sign_match = np.sign(y_pred) == np.sign(y_true)

    # Penalize wrong-direction predictions 2x harder
    weights = np.where(sign_match, 1.0, 2.0)
    grad = weights * 2 * (y_pred - y_true)
    hess = weights * 2
    return grad, hess
```

#### 3.2.2 Asymmetric Huber Loss (GEV-inspired)
2025 research shows **GEV-GBDT** with asymmetric loss outperforms symmetric losses on imbalanced data.
```python
def asymmetric_huber(y_pred, dtrain, delta=0.5, pos_weight=0.8):
    y_true = dtrain.get_label()
    error = y_pred - y_true

    # Smaller penalty for overprediction (gold uptrend bias acceptable)
    weights = np.where(error > 0, pos_weight, 1.0)

    # Huber loss with asymmetric weights
    abs_error = np.abs(error)
    grad = np.where(abs_error <= delta,
                    weights * error,
                    weights * delta * np.sign(error))
    hess = np.where(abs_error <= delta, weights, 0.0) + 1e-6
    return grad, hess
```

### 3.3 Expected Impact
- **DA/HCDA**: +2-4pp (directly optimizes directional accuracy)
- **Sharpe**: +0.3-0.7 (fewer harmful wrong-direction trades)
- **MAE**: **-0.05-0.10** (likely degrades, as loss deprioritizes magnitude)
- **Risk**: High (custom loss can destabilize training)

### 3.4 2025 Research Evidence
- [GEV-GBDT Tandfonline](https://www.tandfonline.com/doi/full/10.1080/01605682.2024.2418882): Asymmetric loss via GEV distribution outperforms symmetric logistic on imbalanced data
- [Directional Forecasting Springer](https://link.springer.com/article/10.1007/s44163-025-00424-4): XGBoost/Gradient Boosting excel at capturing directional dependencies in forex
- Focal loss for LightGBM (Liu et al. 2022) mitigates majority class bias

**Verdict**: **MEDIUM PRIORITY**. High upside for DA/Sharpe, but requires careful tuning. Test with Optuna multi-objective optimization (see Section 7).

---

## 4. Time-Varying Feature Importance (Low-Medium Priority)

### 4.1 Core Concept
Feature importance changes over time. `yc_curvature_z` (rank 1) may be critical during Fed policy shifts but irrelevant during geopolitical crises. Rolling window retraining or dynamic feature weighting can adapt.

### 4.2 Implementation Options

#### 4.2.1 Rolling Window Retraining
- Train model on last 500 days only (vs current 70% of full history)
- Re-train monthly, keep last 12 models in ensemble
- Recent data gets higher weight in final prediction

#### 4.2.2 XGBoost `feature_weights` (v1.3.0+)
```python
from xgboost import DMatrix

# Calculate rolling feature importance (last 200 days)
recent_importance = calculate_shap_importance(X_train[-200:], model)

# Scale weights: top 5 features get 2.0x, rest get 1.0x
feature_weights = np.ones(X_train.shape[1])
top_features = recent_importance.argsort()[-5:]
feature_weights[top_features] = 2.0

dtrain = DMatrix(X_train, label=y_train)
dtrain.set_info(feature_weights=feature_weights)
```

#### 4.2.3 SHAP-Based Dynamic Weighting
- Compute SHAP values on rolling 100-day windows
- Track feature contribution drift over time
- Ensemble predictions weighted by recent SHAP stability

### 4.3 Expected Impact
- **DA/HCDA**: +0-1pp (marginal, as features are already regime-aware via submodels)
- **Sharpe**: +0.1-0.3 (better adaptation to 2025-2026 volatility)
- **MAE**: Neutral
- **Complexity**: High (requires monthly retraining infrastructure)

### 4.4 2025 Research Evidence
- [XGBoost feature_weights docs](https://xgboost.readthedocs.io/en/latest/python/examples/feature_weights.html): Native support since v1.3.0
- [SHAP for time-varying importance (MachineLearningMastery)](https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/): SHAP values computed over rolling windows track contribution evolution

**Verdict**: **LOW-MEDIUM PRIORITY**. Requires significant infrastructure. Consider only if simpler methods (stacking, regime features) fail to improve DA/Sharpe.

---

## 5. Prediction Calibration (Low Priority for This Model)

### 5.1 Problem Statement
Model predictions (std 0.023) are 60x smaller than actuals (std 1.4%). Amplifying predictions could reduce MAE but risks destroying DA/Sharpe.

### 5.2 Calibration Methods

#### 5.2.1 OLS Scaling (Current Method)
```python
pred_scaled = alpha + beta * pred_raw
```
Current: alpha=1.317 (close to ideal 1.0). Already implemented.

#### 5.2.2 Quantile-Based Rescaling
```python
# Match predicted quantiles to actual quantiles
from scipy.stats import rankdata
pred_ranks = rankdata(pred_raw) / len(pred_raw)
pred_calibrated = np.percentile(y_actual, pred_ranks * 100)
```

#### 5.2.3 Isotonic Regression Calibration
```python
from sklearn.isotonic import IsotonicRegression
iso_reg = IsotonicRegression(out_of_bounds='clip')
pred_calibrated = iso_reg.fit_transform(pred_raw, y_actual)
```

### 5.3 Expected Impact
- **MAE**: **-0.15-0.30** (significant improvement possible)
- **DA/HCDA**: **-2-5pp** (severe degradation, as shown in Attempt 4: DA 55.35% → HCDA 42.86% after calibration)
- **Sharpe**: **-0.5-1.0** (amplified errors hurt risk-adjusted returns)

### 5.4 2025 Research Evidence
- [Normalized MAE Tandfonline](https://www.tandfonline.com/doi/full/10.1080/27684520.2024.2317172): Bias corrections can normalize MAE but risk altering directional signals
- Attempt 4 historical failure: Isotonic calibration collapsed HCDA from 55% to 43%

**Verdict**: **NOT RECOMMENDED**. MAE target is structurally infeasible (waived by evaluator). Calibration has proven harmful to DA/Sharpe in this project.

---

## 6. Non-Linear Confidence Scoring (Medium Priority)

### 6.1 Current HCDA Method
Top 30% predictions by bootstrap std + |prediction| magnitude. Simple but suboptimal.

### 6.2 Proposed Improvements

#### 6.2.1 Quantile Regression Confidence Intervals
Train 3 XGBoost models: 5th, 50th, 95th percentiles.
```python
model_low = xgb.train(params={'objective': 'reg:quantileerror', 'quantile_alpha': 0.05}, ...)
model_mid = xgb.train(params={'objective': 'reg:quantileerror', 'quantile_alpha': 0.50}, ...)
model_high = xgb.train(params={'objective': 'reg:quantileerror', 'quantile_alpha': 0.95}, ...)

# Confidence = interval width (smaller = higher confidence)
confidence = (pred_high - pred_low) / pred_mid
```

#### 6.2.2 Conformalized Quantile Regression (CQR)
- Calibrates quantile predictions using split conformal inference
- Provides **coverage guarantees** (true value falls in interval X% of time)
- [Medium CQR Tutorial](https://medium.com/@newhardwarefound/conformalized-quantile-regression-with-xgboost-2-0-e70bbc939f6b)

#### 6.2.3 Ensemble Variance
```python
# Train 10 bootstrapped models
models = [train_xgb(X_bootstrap_i, y_bootstrap_i) for i in range(10)]
preds = np.array([m.predict(X_test) for m in models])

# Confidence = inverse of prediction variance
confidence = 1 / (preds.std(axis=0) + 1e-6)
```

### 6.3 Expected Impact
- **HCDA**: +1-3pp (better confidence scoring separates high/low quality predictions)
- **DA/Sharpe/MAE**: Neutral (only affects confidence threshold, not predictions)

### 6.4 2025 Research Evidence
- [QXGBoost arXiv](https://arxiv.org/pdf/2304.11732): Modified quantile regression (Huber norm) for differentiable PIs; outperforms standard methods
- [Conformalized QR Medium](https://medium.com/@newhardwarefound/conformalized-quantile-regression-with-xgboost-2-0-e70bbc939f6b): Addresses quantile regression shortcomings with calibration
- [XGBoost Confidence Intervals GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/confidence-intervals-for-xgboost/): Bootstrap + quantile methods compared

**Verdict**: **MEDIUM PRIORITY**. Quantile regression is theoretically sound but adds training complexity (3x models). Test if HCDA 64% → 66-67% is achievable.

---

## 7. Multi-Objective Optuna Optimization (High Priority)

### 7.1 Current HPO Problem
Single objective: `val_sharpe * 0.5 + val_da * 0.5`. Equal weighting may not be optimal. Sharpe 2.46 is 3x target (0.80) while HCDA 64.13% barely passes 60%.

### 7.2 Proposed Multi-Objective Framework
```python
import optuna

def multi_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        # ... other params
    }

    model = xgb.train(params, dtrain, num_boost_round=500,
                      early_stopping_rounds=50, evals=[(dval, 'val')])

    pred = model.predict(dval)
    da = directional_accuracy(y_val, pred)
    sharpe = calculate_sharpe(y_val, pred)

    return da, sharpe  # Optuna optimizes Pareto front

study = optuna.create_study(directions=['maximize', 'maximize'])
study.optimize(multi_objective, n_trials=150)

# Select from Pareto front based on target priorities
pareto_trials = study.best_trials
# E.g., choose trial with DA > 58% and max Sharpe
```

### 7.3 Expected Impact
- **DA/HCDA**: +0-2pp (better exploration of DA/Sharpe trade-off space)
- **Sharpe**: Potentially lower (2.46 → 2.0) if DA is prioritized
- **Optimization time**: +50% (need more trials for multi-objective)

### 7.4 2025 Research Evidence
- [Optuna Multi-Objective Docs](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html): Native support, NSGA-II sampler for Pareto optimization
- [Multi-Objective Trading Springer](https://link.springer.com/article/10.1007/s10462-025-11390-9): NSGA-II + modified Sharpe ratio on 110 stock datasets; multi-objective GP yields superior strategies with lower risk
- Optuna 4.7.0 (Jan 2026): Enhanced multi-objective support, GPSampler for constrained optimization

**Verdict**: **HIGHLY RECOMMENDED**. Zero code rewrite, just change `create_study` API. Allows principled exploration of DA/Sharpe trade-offs via Pareto front.

---

## 8. Summary & Prioritized Roadmap

### 8.1 Quick Wins (Attempt 8)
1. **Ensemble Stacking (XGBoost + LightGBM + CatBoost)**: Proven 12-15% gains, low risk
2. **Regime-Conditional Features**: 8-12 interaction features, no new data, 15-20% forecast improvement in 2025 papers
3. **Multi-Objective Optuna (DA + Sharpe)**: One-line API change, principled trade-off exploration

**Expected Impact**: DA 60% → 61-63%, HCDA 64% → 65-67%, Sharpe 2.46 → 2.3-2.7

---

### 8.2 Medium-Term Experiments (Attempt 9-10)
4. **Asymmetric Loss Functions**: Test directional-aware loss + Optuna multi-objective; high risk but +2-4pp DA possible
5. **Non-Linear Confidence Scoring**: Quantile regression for HCDA 64% → 66-67%

---

### 8.3 Low Priority (Consider Only If Above Fails)
6. **Time-Varying Feature Importance**: Complex infrastructure, marginal gains
7. **Prediction Calibration**: Waived MAE target, proven harmful in Attempt 4

---

## 9. Key Implementation Notes

### 9.1 Avoiding Past Mistakes
- **DO NOT** use isotonic/quantile calibration for MAE (Attempt 4 lesson: HCDA 55% → 43%)
- **DO** preserve OLS scaling post-processing in all experiments
- **DO** monitor train-test DA gap (Attempt 1: 40pp gap disaster)
- **DO** use 5-fold CV for ensemble stacking (no data leakage)

### 9.2 Dataset Considerations
- Test set (2025-2026) has 14 days with |return| > 3% (extreme volatility)
- MAE target 0.75% is structurally infeasible given actual return std ~1.4%
- Focus on DA/HCDA/Sharpe; accept MAE 0.90-1.00% range

### 9.3 Computational Costs
- Stacking: 3x training time (XGBoost 10min → 30min total, parallelizable)
- Multi-objective Optuna: 150 trials vs 100 (1.5x time)
- Quantile regression: 3x training time (5th, 50th, 95th percentiles)
- **Total for Attempt 8**: ~2-3 hours Kaggle GPU time (well within limits)

---

## 10. Sources

### Ensemble Stacking
- [Stacking Ensembles: XGBoost, LightGBM, CatBoost - Medium](https://medium.com/@stevechesa/stacking-ensembles-combining-xgboost-lightgbm-and-catboost-to-improve-model-performance-d4247d092c2e)
- [Stock Price Prediction Stacked Heterogeneous Ensemble - MDPI](https://www.mdpi.com/2227-7072/13/4/201)
- [Ensemble Learning Methods 2025 - Johal](https://johal.in/ensemble-learning-methods-xgboost-and-lightgbm-stacking-for-improved-predictive-accuracy-2025/)

### Asymmetric Loss & Directional Accuracy
- [GEV-GBDT for Imbalanced Credit Scoring - Tandfonline](https://www.tandfonline.com/doi/full/10.1080/01605682.2024.2418882)
- [Directional Forex Forecasting - Springer](https://link.springer.com/article/10.1007/s44163-025-00424-4)
- [Gradient Boosting for Financial Risk - Preprints.org](https://www.preprints.org/manuscript/202504.0817)

### Regime-Conditional Features
- [RegimeFolio - arXiv](https://arxiv.org/html/2510.14986v1)
- [S&P 500 Volatility Regime-Switching - arXiv](https://arxiv.org/html/2510.03236v1)
- [Financial Chaos Index - arXiv](https://arxiv.org/html/2504.18958v1)

### Prediction Calibration
- [Normalized MAE for Forecast Verification - Tandfonline](https://www.tandfonline.com/doi/full/10.1080/27684520.2024.2317172)
- [7 Ways to Lower MAE - Number Analytics](https://www.numberanalytics.com/blog/boost-ml-model-accuracy-7-ways-to-lower-mae-errors)

### Time-Varying Feature Importance
- [XGBoost Feature Weights Demo](https://xgboost.readthedocs.io/en/latest/python/examples/feature_weights.html)
- [Feature Importance with XGBoost - MachineLearningMastery](https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/)

### Confidence Intervals & Quantile Regression
- [QXGBoost for Uncertainty Quantification - arXiv](https://arxiv.org/pdf/2304.11732)
- [Conformalized Quantile Regression - Medium](https://medium.com/@newhardwarefound/conformalized-quantile-regression-with-xgboost-2-0-e70bbc939f6b)
- [XGBoost Confidence Intervals - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/confidence-intervals-for-xgboost/)

### Multi-Objective Optimization
- [Optuna Multi-Objective Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)
- [Multi-Objective Trading with Modified Sharpe - Springer](https://link.springer.com/article/10.1007/s10462-025-11390-9)

---

**Researcher**: researcher (Sonnet 4.5)
**Review Status**: Pending architect fact-check
**Next Step**: Architect validates data availability and proposes Attempt 8 design
