# Evaluation Summary: meta_model attempt 8

## Architecture
GBDT Stacking (XGBoost + LightGBM + CatBoost) + Ridge Meta-Learner + 6 Regime-Conditional Features + Bootstrap Confidence + OLS Scaling

## Gate 1: Standalone Quality -- FAIL

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| Overfit ratio (train-test DA gap) | -7.08pp | < 10pp | PASS |
| All-NaN columns | 0 | 0 | PASS |
| Prediction collapse (near-constant) | std=0.000107 | std > 0.001 | FAIL |
| Regime feature quality | 5/6 zero importance | Useful features | FAIL |
| Base model DA | 50.8%-52.5% | > coin-flip skill | FAIL |

**Root cause**: Ridge meta-learner with alpha=90.37 collapsed all stacking predictions to near-constant output (mean=0.058, std=0.0001). Ridge coefficients near-zero (XGB=0.0004, LGBM=0.006, CB=0.005); intercept (0.058) dominates entirely.

## Final Targets: Nominal 3/4 PASS, Substantive 0/4 PASS

| Metric | Target | Actual | Nominal | Substantive | Note |
|--------|--------|--------|---------|-------------|------|
| Direction Accuracy | > 56% | 58.73% | PASS | FAIL | Equals naive always-up (58.73% = 58.73%), delta = 0.0pp |
| High-Confidence DA | > 60% | 61.96% | PASS | FAIL | HC threshold from range of 0.0012; unreliable |
| MAE | < 0.75% | 0.9424% | FAIL | FAIL | Structurally infeasible |
| Sharpe Ratio | > 0.80 | 2.06 | PASS | FAIL | Always-long Sharpe in bull market; zero model skill |

## Comparison vs Attempt 7 (Previous Best)

| Metric | Attempt 7 | Attempt 8 | Delta | Verdict |
|--------|-----------|-----------|-------|---------|
| DA | 60.04% | 58.73% | -1.31pp | Regressed |
| HCDA | 64.13% | 61.96% | -2.17pp | Regressed |
| MAE | 0.9429% | 0.9424% | -0.0005 | Unchanged |
| Sharpe | 2.46 | 2.06 | -0.40 | Regressed |
| vs Naive DA | +1.31pp | +0.00pp | -1.31pp | Lost skill |
| Pred std | 0.0226 | 0.000107 | -211x | Collapsed |
| Positive pct | 87.3% | 100% | +12.7pp | Trivial |

## Failure Diagnosis

### Primary: Ridge Meta-Learner Prediction Collapse
The Ridge regression with alpha=90.37 shrank base model coefficients to near-zero. The intercept (0.058) became the entire prediction. This happened because:

1. **Base models had near-coin-flip DA** (50.8-52.5%): Ridge correctly identified predictions as low-quality
2. **High base model correlation** (r=0.57-0.78): No ensemble diversity for Ridge to exploit
3. **Ridge Optuna found extreme alpha as "optimal"**: Minimized validation loss by ignoring noisy predictions, but produced trivially constant output

### Secondary: Sparse Regime Features
5 of 6 regime-conditional features had zero importance. The high-vol regime was active in 0.0% of samples (VIX threshold too aggressive). These features added noise to an already noisy feature space (30 features from 24 base).

## Substantive Skill Tests: 0/4 PASS

| Test | Result | Detail |
|------|--------|--------|
| DA vs Naive | 0.0pp delta | Zero directional skill |
| Prediction diversity | std=0.0001, 100% positive | Trivial constant predictor |
| Trade activity | 0 position changes | Never changes position |
| vs Attempt 7 | All metrics regressed | Strictly worse |

## Decision: FAIL -- attempt+1

This is attempt 1 of 3 (attempts 8-10). Two more attempts remain.

## Improvement Plan for Attempt 9

### Priority 1: Revert to Single XGBoost Architecture
Abandon stacking. Attempt 7's single XGBoost was proven superior. Preserve its architecture as the foundation.

### Priority 2: Replace Binary Regime Features with Continuous Interactions
Instead of binary `real_rate_change * (vix > 25)` (0% activation), use continuous `real_rate_change * vix_level`. This preserves signal across all samples.

### Priority 3: Asymmetric/Directional Loss Function
Core challenge is beating naive always-up. Use a custom loss that penalizes wrong-direction predictions more heavily, forcing the model to learn when NOT to predict positive.

### Anti-patterns to Avoid
- Do NOT use stacking/ensemble meta-learners (Ridge collapse proven)
- Do NOT use binary regime indicators with sparse activation
- Do NOT try prediction calibration (proven failure in attempts 3-4)
- Do NOT change the core feature set dramatically
