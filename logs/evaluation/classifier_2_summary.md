# Evaluation Summary: Classifier Attempt 2

## Decision: FAIL (attempt+1 recommended)

The classifier produces non-trivial predictions (fixing attempt 1's all-DOWN problem), but the signal is too weak and noisy to add value when ensembled with the regression model. Every ensemble threshold tested degrades DA and Sharpe compared to regression-only. 2026 YTD analysis shows the classifier's P(DOWN) peaks on UP days, causing harmful overrides.

---

## Standalone Classifier Assessment (Test Set)

| Metric | Value | Target | Result |
|--------|-------|--------|--------|
| Balanced Accuracy | 51.69% | > 52% | FAIL (marginal miss) |
| DOWN Recall | 42.31% | > 30% | PASS |
| DOWN Precision | 43.02% | > 45% | FAIL (marginal miss) |
| F1(DOWN) | 0.427 | > 0.35 | PASS |
| MCC | 0.034 | > 0 | PASS (barely) |
| P(DOWN) std | 0.188 | > 0.03 | PASS (non-trivial) |

Standalone criteria met: **3/5**

### Overfitting Assessment: SEVERE

| Split | MCC | AUC |
|-------|-----|-----|
| Train | 0.796 | 0.976 |
| Val | -0.002 | 0.508 |
| Test | 0.034 | 0.542 |

- Overfit ratio (train/test MCC): **23.5x**
- The model memorizes training data completely but generalizes almost at random
- Val MCC is negative, meaning the model is worse than random on validation
- Test MCC barely positive (0.034), suggesting near-zero genuine signal

### Improvement vs Attempt 1

- Attempt 1: Trivial all-DOWN predictor (P(DOWN) std=0.007, 100% DOWN predictions)
- Attempt 2: Non-trivial predictions (P(DOWN) std=0.188, 39.5% DOWN predictions)
- MCC objective + trivial guards successfully prevented degenerate solution
- However, the classifier's discriminative power remains near-random on out-of-sample data

---

## Ensemble Evaluation (Test Set)

### Best Threshold: P(DOWN) > 0.68

Optimized on validation set to maximize Sharpe + 0.5*(DA - 0.5).

| Metric | Regression-Only | Ensemble | Delta |
|--------|----------------|----------|-------|
| DA | 59.17% | 56.11% | -3.06pp |
| Sharpe | 2.05 | 0.83 | -1.22 |
| DOWN capture | 0.0% | 8.6% | +8.6pp |
| UP capture | 100.0% | 88.9% | -11.1pp |
| #DOWN predicted | 0 | 46 | +46 |

### Ensemble Criteria Assessment

| Criterion | Value | Target | Result |
|-----------|-------|--------|--------|
| DA maintained (>= regression) | 56.11% | >= 59.17% | FAIL |
| DA improvement > +1.0pp | -3.06pp | > +1.0pp | FAIL |
| Sharpe > 2.0 | 0.83 | > 2.0 | FAIL |
| DOWN capture > 25% | 8.6% | > 25% | FAIL |
| UP accuracy > 85% | 88.9% | > 85% | PASS |

Ensemble criteria met: **1/5**

### Threshold Scan Analysis (Test Set)

| Threshold | DA | Sharpe | DOWN cap | UP cap |
|-----------|-----|--------|----------|--------|
| 0.45 | 53.7% | -0.10 | 57.2% | 51.3% |
| 0.50 | 53.5% | 0.14 | 42.2% | 61.3% |
| 0.55 | 57.4% | 1.03 | 33.2% | 74.2% |
| 0.60 | 57.4% | 0.76 | 24.1% | 80.4% |
| 0.65 | 55.5% | 0.61 | 10.2% | 86.7% |
| 0.70 | 57.4% | 1.30 | 7.5% | 91.9% |
| Regression-only | 59.17% | 2.05 | 0.0% | 100.0% |

**Key finding**: NO threshold improves upon regression-only. Even the best test threshold (0.55 or 0.60) achieves only 57.4% DA and Sharpe ~1.0, both worse than regression-only (59.17% DA, Sharpe 2.05). The classifier's DOWN overrides lose more UP days than DOWN days they catch correctly.

---

## 2026 YTD Analysis

| Metric | Regression | Ensemble | Delta |
|--------|-----------|----------|-------|
| DA | 68.8% | 56.2% | -12.5pp |
| DOWN captured | 0/10 | 0/10 | +0 |
| UP captured | 22/22 | 18/22 | -4 |
| Strategy return | +15.94% | +1.72% | -14.22pp |

### Critical Problem: P(DOWN) Peaks on UP Days

The classifier's 4 overrides in 2026 (marked *** in output) were ALL wrong:
1. **2026-01-07**: P(DOWN)=0.722, overrode to DN -- Actual: UP (+0.009%)
2. **2026-01-16**: P(DOWN)=0.697, overrode to DN -- Actual: UP (+3.731%)
3. **2026-01-22**: P(DOWN)=0.756, overrode to DN -- Actual: UP (+1.373%)
4. **2026-02-12**: P(DOWN)=0.786, overrode to DN -- Actual: UP (+1.996%)

The classifier had its HIGHEST P(DOWN) readings on days that were strongly UP. This is worse than random -- it is anti-predictive in the most recent data. Meanwhile, on actual large DOWN days:
- 2026-01-29 (-11.37%): P(DOWN) = 0.568 (below threshold)
- 2026-02-11 (-2.92%): P(DOWN) = 0.649 (below threshold)
- 2026-02-13 (-2.77%): P(DOWN) = 0.568 (below threshold)

The classifier fails to identify the largest DOWN days and instead fires on UP days.

---

## Root Cause Analysis

### 1. Severe Overfitting Eliminates Signal
Train MCC 0.796 vs Test MCC 0.034 (23.5x ratio). The 17 features are fit perfectly to training patterns that do not generalize. This is an inherent limitation of trying to classify next-day direction with 17 noisy features on ~2000 training samples.

### 2. Weak Out-of-Sample Signal
Test AUC 0.542 is barely above 0.5 (random). MCC 0.034 means the classifier's predictions are 3.4% better than random coin flip. This tiny edge is entirely consumed by the ensemble's transaction costs and UP accuracy loss.

### 3. Feature-Target Relationship Too Weak
The 17 DOWN-specific features (momentum exhaustion, volatility ratios, cross-asset stress, etc.) have very limited predictive power for next-day direction. Feature importance is uniformly distributed (all features at 5.0-6.6% gain), suggesting no single feature or group provides strong signal.

### 4. Regression Model's Bullish Bias is Actually Informative
The regression model predicts 100% positive (always UP) because in the current gold bull market (2024-2026), gold returns ARE predominantly positive (~59% UP in test set, ~69% UP in 2026). The regression model's "bias" reflects genuine market conditions, not a defect. Overriding it with a near-random classifier destroys value.

---

## Recommendation for Attempt 3

### Decision: attempt+1 (with significant design changes)

The fundamental problem is that **next-day gold direction is extremely hard to classify** with daily features. Two approaches:

### Option A: Higher threshold, smaller scope
- Only override when classifier has extreme confidence (P(DOWN) > 0.85)
- Target: catch only the most obvious DOWN days (e.g., -5% or larger crashes)
- Expected impact: 2-3 additional DOWN catches per year, minimal UP loss
- Risk: still may not add value if extreme P(DOWN) events are anti-correlated with actual downs

### Option B: Different classifier architecture
- Move from XGBoost binary classifier to a **calibrated probability model**
- Use isotonic regression or Platt scaling on validation set to fix probability calibration
- The raw model probabilities are anti-correlated with actual outcomes (highest P(DOWN) on UP days) -- calibration might fix this
- Alternatively: try LightGBM with dart boosting for better regularization

### Option C: Abandon binary classification, try anomaly detection
- Instead of predicting DOWN, detect "abnormal conditions" using isolation forest or autoencoder
- Flag days where feature patterns deviate significantly from recent history
- Override to DOWN only on anomalous days
- This sidesteps the overfitting problem since anomaly detection is unsupervised

### Anti-patterns (do NOT repeat)
- Do NOT use XGBoost binary:logistic with same 17 features (2 attempts, both failed)
- Do NOT expect supervised classification to work with this signal-to-noise ratio
- Do NOT scan P(DOWN) thresholds on validation -- the validation signal is too noisy to select a good threshold

### If attempt 3 also fails: declare "no_further_improvement"
This would be the 3rd consecutive classifier failure. The hypothesis that DOWN-specific features can improve ensemble performance may be fundamentally wrong for daily gold prediction.
