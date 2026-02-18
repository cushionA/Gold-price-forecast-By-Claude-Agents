# Evaluation Summary: classifier attempt 1

## CRITICAL FINDING: TRIVIAL ALL-DOWN PREDICTOR

The model predicts ALL 2,959 samples as DOWN (100%). P(DOWN) distribution: mean=0.538, std=0.007, range=[0.517, 0.559]. This is a degenerate trivial predictor with ZERO classification skill. Every metric that appears to "pass" is an artifact of the trivial prediction.

---

## Gate 1: Standalone Quality -- FAIL

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| Balanced Accuracy | 50.00% | > 52% | FAIL |
| DOWN Recall | 100% | > 30% | PASS (trivial -- all pred DOWN) |
| DOWN Precision | 41.0-48.6% | > 45% | FAIL (= class base rate) |
| DOWN F1 | 0.58-0.65 | > 0.35 | PASS (trivial -- inflated by recall=1) |
| ROC-AUC (val) | 0.503 | > 0.52 | FAIL (random) |
| ROC-AUC (test) | 0.550 | > 0.52 | MARGINAL PASS |
| P(DOWN) std | 0.007 | > 0.05 | FAIL (near-constant output) |
| Prediction balance | 100% DOWN | 20%-60% | FAIL (completely degenerate) |
| Train-Val AUC gap | 0.297 | < 0.08 | FAIL (severe overfitting) |

**Substantive passes: 0/9. Nominal passes (2) are all artifacts of trivial all-DOWN prediction.**

## Gate 2: Ensemble -- SKIPPED

Cannot evaluate ensemble when classifier outputs constant DOWN for all samples. Any threshold-based override would either (a) override nothing (threshold > 0.559) or (b) override everything to DOWN (threshold < 0.517). Neither produces a useful ensemble.

---

## Root Cause Analysis

### Primary: Optuna Objective Function Exploit (CRITICAL)

The composite objective `0.40 * F1_DOWN + 0.30 * AUC + 0.30 * balanced_acc` is EXPLOITABLE by trivial all-DOWN prediction:

- F1_DOWN for all-DOWN with 48.6% DOWN base rate = 0.654
- balanced_acc for all-DOWN = 0.500
- Trivial floor: 0.40*0.654 + 0.30*0.50 + 0.30*0.50 = **0.562**
- Optuna best value: **0.567** (barely above trivial floor)

Optuna correctly found that predicting all-DOWN achieves near-optimal composite score. There is no incentive for the optimizer to find genuine discrimination.

### Secondary: Over-Regularization

Best params include reg_lambda=6.28, reg_alpha=1.38, learning_rate=0.003, n_estimators=100. Effective learning: 100 * 0.003 = 0.3 gradient steps. The model cannot escape the prior distribution with this level of regularization.

### Tertiary: scale_pos_weight Direction Error

scale_pos_weight=0.81 DOWN-weights the UP class, making DOWN relatively more important. Combined with the objective exploit, this AMPLIFIES the all-DOWN incentive.

### Tertiary: Zero Validation Signal

Val ROC-AUC = 0.503 (random). The 18 features show no discriminative power on the validation period (2022-08 to 2024-05). However, test ROC-AUC = 0.55 suggests weak but non-zero signal in 2024-2026. The val period (aggressive Fed rate hikes) may be a particularly difficult regime.

---

## Historical Context: This Is a Recurring Failure Mode

| Instance | Pathology | Similarity |
|----------|-----------|------------|
| Meta-model attempt 8 | 100% positive predictions | Same: trivial constant predictor |
| Meta-model attempt 9 | 100% positive predictions | Same: regularization + weak signal |
| Weekly attempt 1 | 10 unique predictions / 457 | Near-constant output |
| Backtest classification (24 feat) | P(UP) 0.46-0.51 | Same: zero discrimination |
| **Classifier attempt 1** | **100% DOWN predictions** | **Same: trivial predictor** |

The common thread: when signal-to-noise ratio is low, optimization finds that trivial prediction minimizes loss. The fix must include EXPLICIT rejection of trivial predictors.

---

## Improvement Plan for Attempt 2

### MUST FIX (Priority 1-4)

1. **Replace Optuna Objective with MCC**
   - Matthews Correlation Coefficient = 0 for ANY trivial predictor
   - MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
   - Range [-1, +1], 0 = no skill, immune to class imbalance
   - Alternative: MCC + 0.2*AUC if pure MCC prunes too aggressively

2. **Add Trivial Prediction Guard**
   - In Optuna objective: if minority_pct < 15% or prediction_std < 0.03, return -1.0 (force prune)
   - This is a hard safety net even if MCC is the primary metric

3. **Relax Regularization Bounds**
   - learning_rate: [0.01, 0.15] (was [0.005, 0.1])
   - n_estimators: [200, 800] (was [100, 500])
   - reg_lambda: [0.1, 3.0] (was [0.5, 10.0])
   - reg_alpha: [0.0, 1.0] (was [0.0, 5.0])
   - min_child_weight: [3, 15] (was [5, 20])

4. **Fix scale_pos_weight**
   - Constrain to [0.9, 2.0] or fix to natural class ratio
   - Remove perverse incentive to down-weight UP class

5. **Change Early Stopping Metric**
   - Use eval_metric='auc' (not logloss)
   - Logloss can improve as model converges to prior (constant output)
   - AUC directly measures discrimination

### SHOULD FIX (Priority 5-6)

5. Drop rate_surprise (keep only rate_surprise_signed -- correlated pair)
6. Consider adding LightGBM as alternative within Optuna search

### KEEP UNCHANGED

- 18 features (core set, with minor adjustments above)
- Data splits (70/15/15 time-series)
- 100 Optuna trials
- Ensemble threshold optimization approach

---

## Decision: attempt+1

**Rationale**: Clear FAIL, but the failure is caused by a fixable objective function bug and excessive regularization, NOT by fundamentally inadequate features. Train AUC=0.80 proves features contain in-sample information. The MCC objective + trivial-prediction guard directly address the root cause. High confidence (70%+) that attempt 2 will produce non-trivial predictions.

**Resume from**: architect (to update design doc with fixed objective and HP bounds)

**This is NOT a "no_further_improvement" situation**: We have not yet tested whether the 18 features can discriminate UP/DOWN because the model was never given the opportunity to learn. Declaring the features inadequate without a fair test would be premature.
