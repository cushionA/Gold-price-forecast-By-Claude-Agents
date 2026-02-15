# Evaluation Summary: meta_model Attempt 3

## Overall Verdict: REGRESSION -- 2/4 targets met (down from 3/4 in Attempt 2)

| Metric | Target | Attempt 2 | Attempt 3 | Delta | Status |
|--------|--------|-----------|-----------|-------|--------|
| Direction Accuracy | > 56.0% | 57.26% | 53.30% | -3.96pp | FAIL (was PASS) |
| High-Confidence DA | > 60.0% | 55.26% | 59.21% | +3.95pp | FAIL (improved) |
| MAE | < 0.75% | 0.6877% | 0.7173% | +0.030% | PASS (weakened) |
| Sharpe Ratio | > 0.80 | 1.5835 | 1.2201 | -0.363 | PASS (weakened) |

Naive always-up DA on test set: 56.73% (model is -3.43pp WORSE than naive)

---

## 1. Data Integrity Issue

**CRITICAL**: The file `predictions.csv` contained stale Attempt 2 predictions (identical metrics). The actual Attempt 3 ensemble output was in `test_predictions.csv` (capital `Date` column, different prediction values, wider prediction range). This evaluation uses `test_predictions.csv` as the authoritative source.

Evidence:
- `predictions.csv` prediction std = 0.168 (identical to Attempt 2)
- `test_predictions.csv` prediction std = 0.367 (different, consistent with ensemble)
- `predictions.csv` metrics match Attempt 2 exactly (DA=57.26%, HCDA=55.26%, MAE=0.6877%, Sharpe=1.5835)

---

## 2. Overfitting Analysis -- SEVERE REGRESSION

| Metric | Train | Val | Test | Train-Test Gap |
|--------|-------|-----|------|----------------|
| DA | 79.26% | 58.09% | 53.30% | 25.96pp |
| HCDA | 96.88% | 53.33% | 59.21% | 37.67pp |
| MAE | 0.427% | 0.746% | 0.717% | +0.290% |
| Sharpe | 11.04 | 3.22 | 1.22 | 9.82 |
| Correlation | 0.847 | 0.194 | 0.067 | 0.780 |

| Overfitting Metric | Attempt 2 | Attempt 3 | Change |
|--------------------|-----------|-----------|--------|
| Train DA | 62.79% | 79.26% | +16.47pp |
| Train-Test DA gap | 5.54pp | 25.96pp | +20.42pp (4.7x worse) |
| Train HCDA | 73.47% | 96.88% | +23.41pp |
| Train correlation | ~0.35 | 0.847 | massive increase |

The 5-model ensemble with max_depth=3 and n_estimators=300 memorized training patterns far more than Attempt 2's single model with max_depth=2. The train correlation of 0.847 vs test correlation of 0.067 is a textbook overfitting signature.

---

## 3. What Happened: Ensemble Tradeoff

The ensemble succeeded at its intended goal (improving HCDA) but failed overall:

**HCDA improvement mechanism**: Averaging 5 models amplified predictions where all models agreed (genuine high-confidence signals) and attenuated predictions where models disagreed (noisy signals). This improved top-20% HCDA from 55.26% to 59.21% (+3.95pp).

**DA degradation mechanism**: The same averaging process expanded prediction magnitudes (std 0.37 vs 0.17) but did so asymmetrically. For medium-confidence predictions, the ensemble occasionally flipped directional signs or assigned wrong confidence levels. The net effect: 53.30% DA, which is 3.43pp below the naive always-up strategy.

**Re-ranking was a no-op**: Despite alpha_confidence=0.7975 being optimized, the re-ranked HCDA was identical to standard HCDA (59.21% = 59.21%). This means the confidence re-ranking formula added no information beyond what the raw ensemble magnitude already provided.

---

## 4. Quarterly Stability Comparison

| Quarter | Att 2 DA | Att 3 DA | Att 2 Sharpe | Att 3 Sharpe | N |
|---------|----------|----------|--------------|--------------|---|
| 2023 Q3 | 54.3% | 65.7% | 1.54 | 1.37 | 35 |
| 2023 Q4 | 57.1% | 52.4% | 1.13 | 0.02 | 63 |
| 2024 Q1 | 63.9% | 44.3% | 3.07 | -0.08 | 61 |
| 2024 Q2 | 52.4% | 52.4% | 1.26 | 2.45 | 63 |
| 2024 Q3 | 51.6% | 53.1% | 1.50 | 2.84 | 64 |
| 2024 Q4 | 65.6% | 56.2% | 2.48 | 1.08 | 64 |
| 2025 Q1 | 51.7% | 55.2% | -0.99 | -0.39 | 29 |

Attempt 3 shows catastrophic DA failure in 2024 Q1 (44.3%) which was Attempt 2's best quarter (63.9%). The ensemble likely overfit to training-period regime patterns that reversed in 2024 Q1's gold rally. However, Attempt 3 improved in 2023 Q3, 2024 Q2, 2024 Q3, and 2025 Q1.

---

## 5. HCDA Threshold Analysis

| Threshold | N | Att 2 HCDA | Att 3 HCDA | Delta |
|-----------|---|------------|------------|-------|
| Top 30% | 114 | 57.89% | 59.65% | +1.76pp |
| Top 25% | 95 | 57.89% | 58.95% | +1.06pp |
| Top 20% | 76 | 55.26% | 59.21% | +3.95pp |
| Top 15% | 57 | 59.65% | 59.65% | +0.00pp |
| Top 10% | 38 | 60.53% | 60.53% | +0.00pp |
| Top 5% | 19 | 47.37% | 57.89% | +10.52pp |

The HCDA improvement is concentrated in the top-20% and top-5% bands. Interestingly, top-10% remains 60.53% for both attempts. The ensemble specifically improved the 15-20% band where Attempt 2 was weakest (that band previously dragged HCDA from 59.65% down to 55.26%).

---

## 6. Feature Importance (Attempt 3)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | tech_trend_regime_prob | 7.20% |
| 2 | real_rate_change | 6.75% |
| 3 | ie_regime_prob | 5.88% |
| 4 | yield_spread_change | 5.63% |
| 5 | xasset_regime_prob | 5.44% |
| 6 | vix | 5.27% |
| 7 | inflation_exp_change | 5.04% |
| 8 | tech_mean_reversion_z | 4.78% |
| 9 | etf_regime_prob | 4.50% |
| 10 | yc_spread_velocity_z | 4.39% |

Feature importance is relatively uniform (4-7% range), suggesting no single feature dominates. This is expected with strong regularization but also indicates the ensemble was using many weak signals rather than a few strong ones.

---

## 7. Decision: REGRESSION -- Revert to Attempt 2 Architecture for Attempt 4

Attempt 3 is a clear regression from Attempt 2:
- Lost 1 passing target (DA: PASS -> FAIL)
- Gained 0 passing targets (HCDA improved but still FAIL)
- Overfitting dramatically worsened (5.54pp -> 25.96pp gap)
- Test DA below naive always-up baseline

### Root Cause

The 5-model ensemble with relaxed regularization (max_depth 2->3, n_estimators 247->300) was the wrong approach. The added model capacity was absorbed by training-set memorization rather than improved generalization. The HCDA objective reweighting (10% -> 25%) partially worked for HCDA itself but distorted the overall optimization.

### Improvement Plan for Attempt 4

| Priority | Change | Rationale |
|----------|--------|-----------|
| 1 | Revert to single-model architecture (Attempt 2's max_depth=2, n_estimators=~250) | Attempt 2's regularization was correct. Ensemble added complexity without generalization benefit. |
| 2 | Post-hoc HCDA calibration instead of training-time optimization | Instead of distorting the training objective, train the best possible DA/Sharpe model, then apply a confidence calibration layer on top. |
| 3 | Isotonic regression for confidence calibration | Map prediction magnitude to empirical confidence using isotonic regression on validation set. This can improve HCDA without affecting DA. |
| 4 | Consider reducing features | With 22 features and high regularization, some features may be pure noise. Try forward selection keeping only top-10 features by importance. |

### Risk Assessment

Attempt 4 should recover Attempt 2's 3/4 passing metrics as a floor, then focus narrowly on HCDA improvement through post-hoc calibration. This is a conservative, lower-risk approach compared to Attempt 3's architectural changes.

---

## Appendix: Attempt History

| Metric | Att 1 | Att 2 | Att 3 | Target |
|--------|-------|-------|-------|--------|
| DA | 54.1% | 57.26% | 53.30% | > 56% |
| HCDA | 54.3% | 55.26% | 59.21% | > 60% |
| MAE | 0.978% | 0.688% | 0.717% | < 0.75% |
| Sharpe | 0.43 | 1.58 | 1.22 | > 0.80 |
| Targets | 0/4 | 3/4 | 2/4 | 4/4 |
| Train-Test DA gap | 40.15pp | 5.54pp | 25.96pp | < 10pp |
