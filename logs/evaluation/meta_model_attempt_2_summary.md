# Evaluation Summary: meta_model Attempt 2

## Overall Verdict: PARTIAL SUCCESS -- 3/4 targets met

| Metric | Target | Actual | Gap | Status |
|--------|--------|--------|-----|--------|
| Direction Accuracy | > 56.0% | 57.26% | +1.26pp | PASS |
| High-Confidence DA | > 60.0% | 55.26% | -4.74pp | FAIL |
| MAE | < 0.75% | 0.6877% | -0.0623% | PASS |
| Sharpe Ratio | > 0.80 | 1.5835 | +0.7835 | PASS |

Naive always-up DA on test set: 56.73% (model is +0.53pp better than naive)

---

## 1. Massive Improvement Over Attempt 1

| Metric | Attempt 1 | Attempt 2 | Delta | Improved? |
|--------|-----------|-----------|-------|-----------|
| Direction Accuracy | 54.1% | 57.26% | +3.16pp | YES |
| High-Confidence DA | 54.3% | 55.26% | +0.96pp | YES |
| MAE | 0.978% | 0.6877% | -0.2903% | YES |
| Sharpe Ratio | 0.428 | 1.5835 | +1.1555 | YES |
| Train DA | 94.3% | 62.79% | -31.51pp | YES (less overfit) |
| Train-Test DA Gap | 40.15pp | 5.54pp | -34.61pp | YES |

All metrics improved substantially. The improvement plan from Attempt 1 (drop price-level features, switch to reg:squarederror, stronger regularization, fix data pipeline, drop CNY features) was highly effective.

---

## 2. Overfitting Analysis -- RESOLVED

| Metric | Train | Val | Test | Train-Test Gap |
|--------|-------|-----|------|----------------|
| DA | 62.79% | 53.85% | 57.26% | 5.54pp |
| HCDA | 73.47% | 59.57% | 55.26% | 18.21pp |
| MAE | 0.607% | 0.709% | 0.688% | +0.081% |
| Sharpe | 5.131 | 2.208 | 1.584 | 3.547 |

Train-Test DA gap: 5.54pp (target <10pp) -- PASS

The catastrophic overfitting from Attempt 1 (40.15pp gap) has been resolved. The 5.54pp gap is well within the 10pp target. Key factors:
- Switching from directional-weighted MAE to reg:squarederror eliminated memorization
- Dropping 15 non-stationary features (price levels + CNY) removed regime-dependent splits
- Stronger regularization (max_depth=2, min_child_weight=14, lambda=4.76, alpha=3.65) constrained complexity
- Data pipeline fix (1765 vs 964 train samples) provided 83% more training data

Note: Test DA (57.26%) exceeds val DA (53.85%), likely because the test period (Aug 2023 - Feb 2025) contains a sustained gold rally with more predictable upward trends. This is not overfitting but regime-dependent performance.

---

## 3. High-Confidence DA Failure Analysis

The sole remaining failure is HCDA at 55.26% vs the 60% target. Deeper analysis:

| Percentile Threshold | N Samples | HCDA |
|---------------------|-----------|------|
| Top 30% | 114 | 57.89% |
| Top 25% | 95 | 57.89% |
| Top 20% (standard) | 76 | 55.26% |
| Top 15% | 57 | 59.65% |
| Top 10% | 38 | 60.53% |

The model does exhibit improved accuracy at higher confidence levels: the top 10% of predictions achieves 60.53% (would pass the target). The issue is that the 15th-20th percentile band includes predictions with sufficient magnitude to cross the threshold but insufficient signal quality.

Directional analysis:
- Top 20% UP predictions: 61.8% of actual were up (good)
- Top 20% DOWN predictions: 53.9% of actual were down (weaker)

The model is better at identifying bullish than bearish signals. Down-direction high-confidence predictions underperform, pulling the combined HCDA below 60%.

Prediction magnitude suppression remains: prediction std is 0.167 vs actual std 0.896 (18.7% ratio). The model is conservative in its magnitude, making the absolute confidence threshold less discriminating.

---

## 4. Quarterly Stability

| Quarter | DA | MAE | Sharpe | N |
|---------|-----|-----|--------|---|
| 2023 Q3 | 54.3% | 0.477 | 1.54 | 35 |
| 2023 Q4 | 57.1% | 0.645 | 1.13 | 63 |
| 2024 Q1 | 63.9% | 0.552 | 3.07 | 61 |
| 2024 Q2 | 52.4% | 0.864 | 1.26 | 63 |
| 2024 Q3 | 51.6% | 0.713 | 1.50 | 64 |
| 2024 Q4 | 65.6% | 0.784 | 2.48 | 64 |
| 2025 Q1 | 51.7% | 0.669 | -0.99 | 29 |

Performance is variable across quarters: DA ranges from 51.6% to 65.6%. Sharpe is positive in 6/7 quarters. 2025 Q1 shows negative Sharpe (-0.99) with only 29 samples (incomplete quarter). The overall Sharpe of 1.58 is driven by strong 2024 Q1 and Q4 performance.

---

## 5. Model Architecture Summary

- Algorithm: XGBoost reg:squarederror
- Features: 22 (down from 39 in Attempt 1)
  - 5 base features (rates, spreads, VIX) -- all stationary
  - 17 submodel features (regime probs, z-scores, signals) -- all bounded
- Regularization: max_depth=2, min_child_weight=14, lambda=4.76, alpha=3.65, subsample=0.48, colsample=0.37
- Optuna: 80 trials, composite objective (50% Sharpe, 30% DA, 10% MAE, 10% HCDA)
- Data: 1765 train / 378 val / 379 test (fixed from 964 in Attempt 1)

---

## 6. Decision: PARTIAL SUCCESS -- Proceed to Attempt 3

3/4 targets met. The only failing metric is High-Confidence DA (55.26% vs 60.0% target). Since HCDA at top-10% reaches 60.53%, the signal exists but needs better separation at the top-20% level.

### Recommended Improvements for Attempt 3

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| 1 | Calibrate prediction magnitude to improve HCDA separation | The model's prediction std (0.167) is only 18.7% of actual std (0.896). Wider magnitude spread would create better separation between high and low confidence predictions. Consider post-hoc calibration or adjusting the Optuna objective to include a magnitude penalty. |
| 2 | Add asymmetric confidence: separate UP vs DOWN thresholds | DOWN predictions underperform (53.9% vs 61.8% for UP). A confidence-aware threshold or regime-dependent confidence gating could improve HCDA. |
| 3 | Feature interaction terms for regime-dependent confidence | Combine regime probabilities with VIX level to create interaction features that better capture when the model should have high confidence. |

### Risk Assessment

This is a low-risk improvement cycle. 3/4 targets are met and the HCDA gap (-4.74pp) is addressable through prediction calibration rather than fundamental model changes. The core model architecture and feature set are sound.

---

## Appendix: Improvement Plan History

| Change from Attempt 1 | Status | Effect |
|----------------------|--------|--------|
| Fix data pipeline (964 -> 1765 train) | Applied | +83% more training data |
| Drop price-level features (39 -> 22) | Applied | Eliminated non-stationary overfitting |
| Switch to reg:squarederror | Applied | Eliminated directional memorization |
| Drop CNY features | Applied | Removed known DA/Sharpe degraders |
| Stronger regularization | Applied | Train-Test gap: 40.15pp -> 5.54pp |
| Reweight Optuna objective | Applied | Better Sharpe/DA optimization |
| Align Sharpe formula to CLAUDE.md | Applied | Position-change cost only |
