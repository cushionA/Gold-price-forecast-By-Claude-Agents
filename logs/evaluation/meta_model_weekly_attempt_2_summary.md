# Evaluation Summary: meta_model_weekly Attempt 2

## Overall Verdict: FAIL (0/4 substantive, 2/4 nominal)

---

## Gate 1: Standalone Quality -- CONDITIONAL PASS

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| Overfit ratio (train-test DA gap) | -6.67pp | < 15pp | PASS |
| Constant output (pred std) | 0.104 | > 0.01 | PASS |
| Unique predictions | 18 | > 1 | PASS (but still coarse) |
| NaN in output | 0 | 0 | PASS |
| Position changes | 66 | > 0 | PASS |
| DA vs naive (test) | -2.84pp | > 0pp | **FAIL** |
| DA vs naive (val) | -3.26pp | > 0pp | **FAIL** |
| Optuna convergence | top-5 varied | not all identical | PASS |
| OLS alpha | 0.85 | 0.5-10 range | PASS |

Gate 1 passes on structural checks (collapse is fixed) but fails the model-vs-naive check. The model produces genuine variation but that variation is net-harmful.

---

## Gate 2: Information Gain -- SKIPPED (meta-model)

---

## Gate 3: Meta-Model Targets -- FAIL (2/4 nominal, 0/4 substantive)

| Metric | Target | Actual | Gap | Nominal | Substantive | Note |
|--------|--------|--------|-----|---------|-------------|------|
| DA | > 56.0% | 63.24% | +7.24pp | PASS | **FAIL** | -2.84pp below naive (66.08%) |
| HCDA | > 60.0% | 79.35% | +19.35pp | PASS | **FAIL** | = naive on HC subset (79.35%) |
| MAE | < 1.70% | 2.125% | -0.425% | **FAIL** | **FAIL** | 1.3% better than zero-prediction |
| Sharpe A | > 0.80 | 0.71 | -0.09 | **FAIL** | **FAIL** | Dropped from 2.03 (trades destroy value) |

---

## Substantive Skill Tests (3/5 pass)

| Test | Requirement | Actual | Result |
|------|-------------|--------|--------|
| DA above naive (test) | > +0.5pp | -2.84pp | **FAIL** |
| Prediction diversity | > 50 unique | 18 | **FAIL** |
| Trade activity | > 10 changes | 66 | PASS |
| Prediction balance | 30-90% positive | 88.0% | PASS (borderline) |
| Prediction variation | std > 0.1 | 0.104 | PASS (barely) |

---

## Attempt 1 vs Attempt 2 Comparison

| Metric | Attempt 1 | Attempt 2 | Change | Assessment |
|--------|-----------|-----------|--------|------------|
| Prediction std | 0.005 | 0.104 | 19x better | Collapse FIXED |
| Unique predictions | 10 | 18 | +80% | Improved |
| Position changes | 0 | 66 | +66 new | Collapse FIXED |
| Positive % | 100.0% | 88.0% | -12pp | Improved |
| OLS alpha | 2.76 | 0.85 | Reasonable | Improved |
| DA | 66.08% | 63.24% | -2.84pp | **Regressed** |
| DA vs naive | 0.00pp | -2.84pp | -2.84pp | **Regressed** |
| HCDA (bootstrap) | 68.48% | 79.35% | +10.87pp | Nominal only (both = naive on subset) |
| MAE | 2.070% | 2.125% | +0.055% | Slightly worse |
| Sharpe A | 2.03 | 0.71 | -1.32 | **Major regression** |
| Nominal targets | 3/4 | 2/4 | -1 | **Regressed** |
| Substantive targets | 0/4 | 0/4 | same | No improvement |

**Net assessment**: Design changes successfully fixed the constant-output collapse. The model now produces genuine conditional variation. However, that variation is net-harmful: the model's negative predictions are wrong 62% of the time, destroying DA and Sharpe. Moved from "no skill, no harm" to "some variation, net harmful" -- a lateral move.

---

## Root Cause Analysis

### Why Negative Predictions Fail

The model predicts negative 55 times (12% of test). Of those 55:
- 21 are correct (actual < 0) -- 38.2%
- 34 are wrong (actual > 0) -- 61.8%

In a test set where 66% of days have positive weekly returns, the model needs >50% accuracy on negative calls to beat naive. At 38.2%, every negative prediction set costs the model on average.

**Net contribution of negative predictions**: -13 (21 correct - 34 false negatives)

### Why Val-to-Test Generalization Fails

| Split | DA | Naive DA | Skill |
|-------|-----|---------|-------|
| Train (425) | 56.57% | 53.29% | +3.29pp |
| Val (92) | 48.91% | 52.17% | -3.26pp |
| Test (457) | 63.24% | 66.08% | -2.84pp |
| Optuna best trial (val) | 58.70% | 52.17% | +6.52pp |

Optuna found +6.52pp skill on 91 val samples, but the final model shows -3.26pp skill on the same val set. The standard error with 91 samples is ~5pp, so +6.52pp is within 1.3 SE -- not statistically significant. The 91-sample non-overlapping val set is too small for reliable hyperparameter selection.

### Quarterly DA Skill (inconsistent)

| Quarter | DA | Naive | Skill | Neg Preds |
|---------|-----|------|-------|-----------|
| 2024Q2 | 54.9% | 52.9% | +2.0pp | 5 |
| 2024Q3 | 68.8% | 73.4% | -4.7pp | 11 |
| 2024Q4 | 56.2% | 54.7% | +1.6pp | 7 |
| 2025Q1 | 77.0% | 82.0% | -4.9pp | 3 |
| 2025Q2 | 51.6% | 59.7% | -8.1pp | 17 |
| 2025Q3 | 72.3% | 73.8% | -1.5pp | 65 |
| 2025Q4 | 60.9% | 59.4% | +1.6pp | 5 |
| 2026Q1 | 61.5% | 76.9% | -15.4pp | 4 |

Positive skill in only 3 of 8 quarters. No temporal consistency.

---

## MAE Target Feasibility Assessment

| Metric | Value |
|--------|-------|
| Weekly MAE target | 1.70% |
| Current MAE | 2.125% |
| Zero-prediction MAE | 2.153% |
| Median |weekly return| | 1.774% |
| Weekly return std | 2.678% |
| Daily MAE target | 0.75% |
| Daily return std | ~1.00% |
| Vol ratio (weekly/daily) | 2.68x |
| MAE target ratio (weekly/daily) | 2.27x |

**Assessment: LIKELY INFEASIBLE**. The MAE target of 1.70% is below the median absolute weekly return (1.77%). Achieving this would require near-perfect magnitude prediction. Even the best daily model achieves only 12% improvement over zero-prediction MAE. Weekly target requires 21% improvement -- likely structurally impossible. Recommended revised target: 2.0% or use relative metric.

---

## Decision: ATTEMPT+1

### Improvement Plan for Attempt 3

**Priority 1: Fix HPO validation strategy**
- The 91 non-overlapping val samples are too noisy for Optuna (SE ~5pp)
- Option A: Use overlapping targets for training/HPO, non-overlapping for final evaluation
- Option B: Time-series cross-validation with multiple non-overlapping folds

**Priority 2: Consider classification objective**
- Current regression (reg:squarederror) optimizes magnitude, evaluated on direction
- Direct binary classification (logistic loss) would optimize for DA
- Or: use asymmetric loss that penalizes wrong-sign predictions more than magnitude errors

**Priority 3: Calibrate negative prediction threshold**
- Model mean prediction = 0.098 (positive)
- Using sign(pred) classifies 88% as positive, 12% as negative
- A threshold at ~0.05 would rebalance to ~70/30, better matching the actual 66/34 split
- Or: use model confidence to only trade when deviation from mean is significant

**Priority 4: Revise MAE target**
- Current 1.70% is likely structurally infeasible
- Recommend 2.0% or relative metric (MAE / zero-prediction MAE < 0.85)

---

## Comparison with Daily Meta-Model (Attempt 7)

| Metric | Daily (Att 7) | Weekly (Att 2) | Assessment |
|--------|---------------|----------------|------------|
| DA | 60.04% | 63.24% | Weekly nominally higher (but =naive) |
| DA vs naive | +2.3pp | -2.84pp | Daily has genuine skill, weekly does not |
| HCDA | 64.13% | 79.35% | Weekly nominally higher (but =naive subset) |
| MAE | 0.94% | 2.13% | Not comparable (different scales) |
| Sharpe | 2.46 | 0.71 | Daily vastly superior |
| Substantive skill | YES | NO | Daily model remains the benchmark |

The daily model (Attempt 7) remains strictly superior. It is the only model that demonstrates genuine predictive skill above naive baselines.

---

**Evaluator**: evaluator (Opus 4.6)
**Date**: 2026-02-17
**Attempt consumed**: Yes (attempt 2 of weekly meta-model)
**Next**: Architect redesign for Attempt 3 focusing on HPO validation strategy and classification objective
