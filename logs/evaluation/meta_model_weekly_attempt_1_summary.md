# Evaluation Summary: meta_model_weekly attempt 1

## VERDICT: FAIL -- Model is a trivial always-positive predictor with zero learned skill

---

## Gate 1: Standalone Quality -- FAIL

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| Train-test DA gap | -12.32pp (test > train) | < 10pp | PASS (nominal) |
| Prediction unique values | 10 out of 457 samples | >> 10 expected | FAIL (CRITICAL) |
| All predictions positive | 100% | Should have both signs | FAIL (CRITICAL) |
| Prediction std | 0.0055 (range 0.095-0.140) | << actual std 2.68 | FAIL (CRITICAL) |
| Model vs naive DA (test) | 66.08% vs 66.08% = 0.00pp | > 0 | FAIL (CRITICAL) |
| Model vs naive DA (val) | 53.07% vs 53.07% = 0.00pp | > 0 | FAIL (CRITICAL) |
| NaN in output | 0 | 0 | PASS |
| Optuna top-5 val DA | All 53.07% (identical) | Should vary | FAIL |
| OLS alpha | 2.76 (predictions 3x too small) | ~1.0 | FAIL |
| Trades in test | 0 | > 0 expected | FAIL |

**Critical Finding**: The model outputs only 10 distinct prediction values. 67.4% of test predictions are the single value 0.11269. All predictions are positive. The model is functionally equivalent to predicting "gold goes up" every day.

---

## Gate 2: Information Gain -- SKIPPED

Skipped for meta-model evaluation (no submodel ablation applicable).

---

## Gate 3: Meta-Model Target Evaluation -- FAIL (0/4 substantive)

### Formal Target Assessment

| Metric | Target | Actual | Formal | Substantive | Problem |
|--------|--------|--------|--------|-------------|---------|
| DA | > 56% | 66.08% | PASS | FAIL | = naive always-up (66.08%). Zero model skill. |
| HCDA | > 60% | 68.48% (bootstrap) | PASS | FAIL | HC subset naive = 68.37%. Zero model skill. |
| MAE | < 1.70% | 2.07% | FAIL | FAIL | Only 4% better than predicting zero (2.15%). |
| Sharpe | > 0.80 | 2.03 (Approach A) | PASS | FAIL | 0 trades. Sharpe = gold buy-and-hold Sharpe. |

### Why Formal Passes Are Meaningless

**Direction Accuracy (66.08%)**: Gold returned positive over 5 days in 66.08% of test period samples (Apr 2024 - Feb 2026 was a strong gold uptrend). A strategy that always predicts "up" achieves exactly 66.08% DA. The model's DA is 0.00pp above this baseline in both test and validation sets.

**HCDA (68.48%)**: Bootstrap confidence selects samples where the 5 bootstrap models agree most. Since all 5 models produce nearly identical output (mean bootstrap std = 0.014), "high confidence" simply selects periods where one model variant deviates least -- which correlates with low-volatility trending periods where gold is most likely positive. The naive always-up DA on the bootstrap-selected subset is 68.37%, matching the model.

**Sharpe (2.03)**: The model makes 0 position changes during the test period. It holds a permanent long position. The Sharpe ratio is therefore the Sharpe ratio of gold itself during the test period, minus negligible entry cost. This is buy-and-hold, not model-driven trading.

---

## Comparison with Daily Attempt 7

| Metric | Daily Att 7 | Weekly Att 1 | Delta | Notes |
|--------|------------|-------------|-------|-------|
| DA | 60.04% | 66.08% | +6.0pp | Meaningless -- weekly naive is 66%, daily naive was 58% |
| DA vs Naive | +2.3pp | +0.0pp | -2.3pp | Daily model had genuine skill; weekly has none |
| HCDA | 64.13% | 68.48% | +4.4pp | Same problem -- naive bias in HC subset |
| MAE | 0.94% | 2.07% | +1.13pp | Weekly target larger, but model barely beats zero-pred |
| Sharpe | 2.46 | 2.03 | -0.43 | Daily had real trades; weekly has 0 trades |
| Unique predictions | >100 | 10 | -- | Daily model learned meaningful variation |
| Positive pred % | ~80% | 100% | -- | Daily model predicted negative sometimes |

**Key insight**: Daily attempt 7 was the first model to beat the naive strategy. Weekly attempt 1 has regressed to a model indistinguishable from naive.

---

## Root Cause Analysis

### Why the model collapsed to constant output:

1. **Overlapping weekly targets**: Each row's 5-day return shares 4/5 days with adjacent rows. This creates massive target autocorrelation. XGBoost with strong regularization (max_depth=2, min_child_weight=21) learns the moving average rather than conditional differences.

2. **Higher noise floor**: Weekly return std (~2.7%) is ~2.7x daily (~1.0%). The conditional signal (what features tell us beyond the unconditional mean) is a much smaller fraction of total variance.

3. **Objective function bias**: The composite objective rewards positive predictions because:
   - Train set is 54% positive => always-up gives ~54% DA
   - Sharpe component rewards long-only during historical positive drift
   - MAE component is insensitive to direction when predictions are near zero

4. **Regularization too strong for weak signal**: The HP ranges (max_depth [2,4], min_child_weight [12,25], reg_lambda [1,15]) were tuned for daily returns. For noisier weekly returns, the model needs to express more conditional variation, but regularization prevents it.

5. **OLS scaling cannot fix directionality**: OLS scaling multiplies all predictions by a constant. Since all predictions are positive, scaling preserves the always-positive pattern.

---

## Improvement Plan for Attempt 2

### Priority 1: Non-overlapping training targets (CRITICAL)

Train on non-overlapping 5-day periods only (every 5th row). This:
- Eliminates target autocorrelation (root cause #1)
- Reduces training samples from ~2130 to ~426, but with independent observations
- Forces model to learn from actual 5-day regime differences, not smoothed trends

### Priority 2: Directional prediction incentive (HIGH)

Modify Optuna objective to explicitly penalize naive-matching:
- Compute naive always-up DA on validation set
- DA component should measure DA **minus** naive DA, not raw DA
- Add penalty term: `naive_penalty = max(0, naive_da - val_da) * 2.0`
- This prevents Optuna from converging on always-positive solutions

### Priority 3: Relaxed regularization for weekly scale (MEDIUM)

Widen HP search space:
- max_depth: [2, 6] (was [2, 4])
- min_child_weight: [5, 25] (was [12, 25])
- reg_lambda: [0.1, 15.0] (was [1.0, 15.0])
- reg_alpha: [0.01, 10.0] (was [0.5, 10.0])

This allows the model to capture more granular conditional patterns needed for weekly prediction.

### Priority 4: Symmetrized target (MEDIUM)

Subtract unconditional mean from weekly returns before training:
```python
train_mean = y_train.mean()
y_train_centered = y_train - train_mean
y_val_centered = y_val - train_mean
```
Add back during prediction. This removes the positive bias that lets the model "cheat" by predicting positive.

### Priority 5: Alternative Sharpe calculation (LOW)

Use non-overlapping weekly Sharpe (Approach B, sqrt(52)) as primary metric in the objective, not Approach A. Approach A inflates Sharpe for always-long strategies due to autocorrelated daily returns within each 5-day holding period.

---

## Decision

| Field | Value |
|-------|-------|
| Overall Passed | false |
| Nominal Targets | 3/4 (DA, HCDA, Sharpe) |
| Substantive Targets | 0/4 |
| Decision | attempt+1 |
| Attempt consumed | Yes (attempt 1 -> next is attempt 2) |
| Resume from | architect (fundamental design changes needed) |
| Priority improvement | Non-overlapping training + directional incentive |
