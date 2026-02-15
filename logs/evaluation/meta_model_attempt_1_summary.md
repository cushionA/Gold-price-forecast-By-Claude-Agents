# Evaluation Summary: meta_model Attempt 1

## Overall Verdict: FAIL -- All 4 targets missed

| Metric | Target | Actual | Gap | Status |
|--------|--------|--------|-----|--------|
| Direction Accuracy | > 56.0% | 54.1% | -1.9pp | FAIL |
| High-Confidence DA | > 60.0% | 54.3% | -5.7pp | FAIL |
| MAE | < 0.75% | 0.978% | +0.228% | FAIL |
| Sharpe Ratio | > 0.80 | 0.428 | -0.372 | FAIL |

Naive always-up DA on test set: 59.9% (model is 5.8pp WORSE than naive)

---

## 1. Root Cause Analysis

### 1.1 Severe Overfitting (PRIMARY FAILURE)

| Metric | Train | Val | Test | Train-Test Gap |
|--------|-------|-----|------|----------------|
| DA | 94.3% | 46.9% | 54.1% | 40.2pp |
| MAE | 0.617% | 0.627% | 0.978% | +0.361% |
| Sharpe | 11.66 | -0.31 | 0.43 | 11.23 |
| Pred-Actual Corr | 0.590 | -0.090 | 0.047 | -- |

The model achieved 94.3% train DA but only 46.9% val DA (below random 50%). This is catastrophic overfitting. The correlation between predictions and actuals is 0.59 on training data but collapses to -0.09 on validation and 0.05 on test. The model memorized training patterns rather than learning generalizable signals.

The design document specified a target of < 5pp train-test DA gap. The actual gap is 40.2pp -- 8x the threshold.

### 1.2 Data Split Size Discrepancy

A significant issue: the notebook used only 1,378 rows (964 train / 207 val / 207 test) instead of the expected 2,523 rows (1,766 / 378 / 379). This means 45.4% of the data was lost during the inner join and NaN removal step in the Kaggle notebook.

With only 964 training samples and 39 features, the samples-per-feature ratio is 24.7:1. This is dangerously low for tree-based models, especially with max_depth=3 and 98 boosting rounds (~784 effective parameters).

### 1.3 Why Directional-Weighted MAE Failed

The best trial (trial 38) used directional_penalty=4.52, meaning wrong-direction errors were penalized 4.52x more than correct-direction errors. This caused three problems:

1. **Overfit incentive**: The high penalty made the model aggressively fit training directions, achieving 94.3% train DA through memorization rather than pattern recognition.

2. **Magnitude suppression**: To avoid the heavy penalty for wrong-direction predictions, the model learned to make very small predictions. Test prediction std = 0.062% versus actual return std = 1.307%. Predictions are only 4.7% of actual magnitude.

3. **Positive bias propagation**: The model learned a positive prediction bias from training (52.1% positive actuals), producing 77.3% positive predictions on validation (where only 50.2% of actuals are positive). This gave val DA below 50%.

### 1.4 Validation Set Unreliability

| Split | Up% | Mean Return | Return Std |
|-------|-----|-------------|------------|
| Train | 52.1% | +0.030% | 0.954% |
| Val | 50.2% | +0.069% | 0.813% |
| Test | 59.9% | +0.222% | 1.307% |

The test period (Mar 2024 - Dec 2025) saw a major gold rally with higher volatility and stronger positive skew. The validation period is relatively balanced. The model's test DA of 54.1% is partly an artifact of slight positive prediction bias coincidentally aligning with the bullish test period, not genuine predictive skill.

### 1.5 Cost Formula Impact

The training notebook and src/evaluation.py deduct 5bps cost EVERY trading day, not only on position changes. This creates a 12.6% annual drag. With this formula, achieving Sharpe > 0.8 requires approximately DA >= 56% (50% probability at that DA level). The model's 54.1% DA is insufficient.

Note: The CLAUDE.md evaluator specification (calc_metrics) uses cost only on position changes. If that formula were applied, the model's Sharpe would be 1.027 (passing). However, consistency with the training notebook and src/evaluation.py requires using the daily-cost formula, yielding Sharpe = 0.428.

---

## 2. Feature Analysis

### 2.1 Feature Importance (Top 10)

| Rank | Feature | Importance | Type |
|------|---------|-----------|------|
| 1 | etf_flow_gld_close | 18.66 | BASE (price level) |
| 2 | tech_trend_regime_prob | 15.28 | SUBMODEL |
| 3 | cross_asset_silver_close | 14.46 | BASE (price level) |
| 4 | technical_gld_open | 14.06 | BASE (price level) |
| 5 | etf_pv_divergence | 13.79 | SUBMODEL |
| 6 | ie_gold_sensitivity_z | 13.23 | SUBMODEL |
| 7 | xasset_regime_prob | 13.04 | SUBMODEL |
| 8 | etf_regime_prob | 12.94 | SUBMODEL |
| 9 | cny_vol_regime_z | 12.93 | SUBMODEL |
| 10 | technical_gld_volume | 12.78 | BASE |

### 2.2 Price Level Features Are Problematic

The top-ranked features include raw price levels (gld_close: 18.66, silver_close: 14.46, gld_open: 14.06, gld_high: 11.67, gld_low: 11.61). These non-stationary features follow long-term trends that change regime across train/val/test periods. Gold prices were $1,200-$2,000 in training but $2,000-$2,700 in the test period. The model learned price-level-specific split thresholds that are meaningless out-of-sample.

This is a classic leakage-adjacent pattern: the model uses price levels as a proxy for "time period" rather than learning return-predictive patterns.

### 2.3 CNY Features

CNY features rank #9 (cny_vol_regime_z: 12.93) and #15 (cny_demand_cny_usd: 12.04) in importance. Combined CNY submodel importance is 46.17 (second highest after ETF at 76.55). Given that cny_demand individually degraded DA by -2.06% and Sharpe by -0.593 in Phase 2, these features are likely adding noise that the meta-model overfits to.

### 2.4 Submodel Group Importance

| Group | Total Importance | Per-Feature |
|-------|-----------------|-------------|
| etf_ | 76.55 | 25.52 |
| cny_ | 46.17 | 15.39 |
| vix_ | 45.86 | 15.29 |
| tech_ | 37.61 | 12.54 |
| ie_ | 34.44 | 11.48 |
| yc_ | 23.07 | 11.53 |
| xasset_ | 24.72 | 8.24 |

Submodel features are being used by the model. The issue is not that submodels are ignored but that the model overfits to their interaction with price-level features.

---

## 3. Optuna Trial Analysis

### 3.1 Validation DA Distribution

| Statistic | Value |
|-----------|-------|
| Val DA range | 45.9% - 52.7% |
| Val DA mean | 49.1% |
| Trials with val DA > 50% | 9 / 50 (18%) |
| Best val DA (trial 38) | 52.7% |

Only 18% of trials achieved val DA above random (50%). The best trial (38) had val DA of 52.7% and val Sharpe of 0.41. No trial achieved val DA > 53% or val Sharpe > 0.75 simultaneously.

### 3.2 Sharpe Distribution

| Statistic | Value |
|-----------|-------|
| Val Sharpe range | -2.10 to +0.75 |
| Trials with val Sharpe > 0 | 34 / 50 (68%) |
| Trials with val Sharpe > 0.5 | 4 / 50 (8%) |
| Best val Sharpe (trial 47) | 0.748 |

Trial 47 achieved the highest val Sharpe (0.748) but with val DA of only 49.3%. The composite objective (40% Sharpe, 25% DA, 20% HC-DA, 15% MAE) selected trial 38, which balanced DA and Sharpe rather than maximizing either one.

### 3.3 Best Trial Parameters

| Parameter | Value | Comment |
|-----------|-------|---------|
| max_depth | 3 | Minimum of range (good for regularization) |
| min_child_weight | 5 | Mid-range |
| subsample | 0.695 | Mid-range |
| colsample_bytree | 0.535 | Lower end (more regularized) |
| reg_lambda (L2) | 1.56 | Lower end of [1.0, 10.0] range |
| reg_alpha (L1) | 0.25 | Lower end of [0.1, 5.0] range |
| learning_rate | 0.0108 | Mid-range |
| gamma | 0.36 | Lower end |
| directional_penalty | 4.52 | Near maximum (problematic) |
| confidence_threshold | 0.0116 | Mid-range |
| n_estimators_used | 98 | Early stopped at 98/1000 |

Key observations:
- Optuna selected max_depth=3 (most conservative), but regularization parameters (lambda=1.56, alpha=0.25) are at the weak end of their ranges.
- directional_penalty=4.52 is near the maximum (5.0), confirming the loss function is aggressively overfitting to training directions.
- Early stopping at 98 rounds suggests the model diverged quickly from optimal.

---

## 4. Decision: FAIL -- Proceed to Attempt 2

Improvement is clearly possible. The failures are attributable to specific, addressable issues:

1. Price-level features causing non-stationary overfitting
2. Directional-weighted MAE loss with high penalty causing memorization
3. Reduced dataset size (964 vs expected 1766 training samples)

### 4.1 Recommended Primary Change: Option A + D Combined

**Stronger Regularization AND Feature Engineering** (combined approach):

#### Feature Changes (Option D, modified):
- **Drop all raw price-level base features**: gld_open, gld_high, gld_low, gld_close, silver_close, copper_close, sp500_close, gld_volume (technical), gld_close and gld_volume (etf_flow), volume_ma20 (etf_flow). These 11 non-stationary features dominate importance but cause regime-dependent overfitting.
- **Drop CNY features**: cny_regime_prob, cny_momentum_z, cny_vol_regime_z, cny_demand_cny_usd (4 features). Phase 2 showed DA -2.06% and Sharpe -0.593 degradation.
- **Keep**: Stationary/transformed features only -- rates, spreads, z-scores, regime probabilities from submodels.
- **Result**: ~24-25 features (from 39), all stationary or bounded.

#### Regularization Changes (Option A):
- **Drop directional-weighted MAE entirely**. Use standard `reg:squarederror` objective.
- **Increase reg_lambda range**: [3.0, 20.0] (was [1.0, 10.0])
- **Increase reg_alpha range**: [1.0, 10.0] (was [0.1, 5.0])
- **Reduce max_depth range**: [2, 4] (was [3, 6])
- **Increase min_child_weight range**: [10, 30] (was [3, 10])
- **Reduce subsample range**: [0.4, 0.7] (was [0.5, 0.8])
- **Reduce colsample_bytree range**: [0.3, 0.6] (was [0.5, 0.8])

#### Optuna Objective Change:
- Remove directional_penalty from search space
- Change composite weights: Sharpe 50%, DA 30%, MAE 10%, HC-DA 10% (from 40/25/15/20)
- DA is the binding constraint more than Sharpe (DA achievability drives Sharpe)

#### Data Fix:
- Investigate and fix the data loss from 2523 to 1378 rows. Ensure the notebook properly handles timezone mismatches and NaN imputation for early-period submodel outputs.
- Target: >= 2000 training samples (from current 964)

### 4.2 Rationale

The directional-weighted MAE was the primary design hypothesis for Attempt 1. It failed definitively: 94.3% train DA with 46.9% val DA proves the custom loss causes memorization, not learning. Removing it is the single highest-impact change.

Price-level features are the second root cause. They dominate the top-5 importance but are inherently non-stationary -- gold at $1,300 (2018) vs $2,500 (2024) means the model learns price-range-specific patterns that do not generalize.

The data loss issue (45% of rows dropped) is an implementation bug, not a design issue. Fixing it provides 80% more training data at zero cost.

---

## 5. Improvement Plan Summary

| Priority | Change | Expected Impact | Risk |
|----------|--------|-----------------|------|
| 1 | Fix data loss (964 -> ~1766 train) | More training data, less overfit | Low |
| 2 | Drop price-level features (39 -> ~25) | Remove non-stationary overfit source | Low |
| 3 | Switch to reg:squarederror loss | Prevent directional memorization | Low |
| 4 | Drop CNY features (4 fewer) | Remove known DA/Sharpe degraders | Low |
| 5 | Stronger regularization ranges | Reduce train-test gap | Low |
| 6 | Reweight Optuna objective | Better DA/Sharpe optimization | Low |

All 6 changes are low-risk because they simplify the model (fewer features, standard loss, stronger regularization). Simpler models are less likely to overfit on 964-1766 training samples.

**Target for Attempt 2**: Train-test DA gap < 10pp, Test DA > 56%, Test Sharpe > 0.5 (intermediate target)

---

## Appendix: Sharpe Formula Discrepancy

The CLAUDE.md evaluator specification (Section "Meta-Model Final Targets" and calc_metrics function) uses cost on position changes only:

```
trades = np.abs(np.diff(positions, prepend=0))
ret = positions * actual - trades * cost
```

But the training notebook and src/evaluation.py use cost deducted every day:

```
net_returns = strategy_returns - cost_pct
```

With the CLAUDE.md formula, the model's test Sharpe would be 1.027 (passing). With the daily-cost formula, it is 0.428 (failing). This discrepancy should be resolved in Attempt 2 by aligning the training notebook with the CLAUDE.md evaluator specification.

**Recommendation**: Use the CLAUDE.md formula (cost on position changes only) for Attempt 2. This is economically more realistic and matches the evaluator specification.
