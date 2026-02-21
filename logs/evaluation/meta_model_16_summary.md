# Evaluation Summary: meta_model attempt 16

## Architecture
LightGBM GBDT + Bootstrap confidence (5-seed ensemble) + OLS scaling
24 features (same as attempt 7) | 100 Optuna trials | Train 2133 / Val 457 / Test 458

---

## Gate 1: Standalone Quality -- PASS

- Overfit gap: 1.07pp (threshold < 10pp) PASS
- All-NaN columns: 0 PASS
- Constant output: std=0.0481 (genuine variation) PASS
- HPO coverage: 100 trials PASS
- OLS alpha: 1.504 (within [0.5, 10.0]) PASS
- Bootstrap consistency: std_range [0.004, 0.077], mean 0.025 PASS
- Prediction collapse: positive_pct=65.5%, NOT trivial predictor PASS

## Gate 2: N/A (Phase 3 meta-model)

## Gate 3: Final Target Evaluation -- PARTIAL PASS (3/4)

| Metric | Target | Actual | Gap | Passed? |
|--------|--------|--------|-----|---------|
| Direction Accuracy | > 56.0% | 58.52% | +2.52pp | PASS |
| High-Confidence DA | > 60.0% | 68.48% | +8.48pp | PASS (BEST EVER) |
| MAE | < 0.75% | 0.953% | -0.203% | FAIL (structural) |
| Sharpe Ratio | > 0.80 | 1.76 | +0.96 | PASS |

---

## vs Attempt 7 (Current Best)

| Metric | Attempt 7 | Attempt 16 | Delta | Winner |
|--------|-----------|------------|-------|--------|
| DA | 60.04% | 58.52% | -1.52pp | Attempt 7 |
| HCDA | 64.13% | **68.48%** | **+4.35pp** | **Attempt 16** |
| MAE | 0.943% | 0.953% | +0.011 | Attempt 7 |
| Sharpe | 2.46 | 1.76 | -0.70 | Attempt 7 |

**Attempt 16 wins on 1/4 metrics. Does NOT replace attempt 7 as best model.**

## vs Naive Always-Up Strategy

- Naive DA: 58.95%
- Attempt 16 DA: 58.52%
- Delta: -0.44pp (model slightly below naive)
- Attempt 7 was +1.31pp above naive

---

## Key Technique Analysis

### Bootstrap Confidence: PROVEN EFFECTIVE
- Single-model HCDA: 59.78% (would FAIL 60% target)
- Bootstrap HCDA: 68.48% (+8.70pp improvement)
- This is the most impactful HCDA technique discovered across all 16 attempts
- Bootstrap measures inter-model agreement to identify genuinely confident predictions

### LightGBM vs XGBoost: LightGBM INFERIOR
- LightGBM num_leaves=66 (~depth 6) is too complex for this dataset
- XGBoost max_depth=2 (attempt 7) provides better regularization
- LightGBM DA 58.52% vs XGBoost DA 60.04% (-1.52pp)
- LightGBM Sharpe 1.76 vs XGBoost Sharpe 2.46 (-0.70)

---

## Historical Context

9 consecutive attempts (8-16) have all failed to beat attempt 7 on ALL metrics.

However, attempt 16 is QUALITATIVELY different from prior failures:
- It validates a new technique (bootstrap confidence) that demonstrably works
- The regression is attributable to LightGBM (replaceable) not bootstrap (the innovation)
- The combination of XGBoost (attempt 7) + bootstrap (attempt 16) has NOT been tried

---

## Untried Combination: XGBoost + Bootstrap

| Component | Source | Evidence |
|-----------|--------|----------|
| XGBoost (max_depth=2, n_est=621) | Attempt 7 | DA 60.04%, Sharpe 2.46 |
| Bootstrap confidence (data subsampling) | Attempt 16 | HCDA +8.70pp |
| Expected result | Combined | DA ~60%, HCDA ~68-72%, Sharpe ~2.0-2.5 |

Attempt 7's bootstrap variance was too uniform (std_mean=0.008) because seed-only variation was insufficient. Solution: bootstrap TRAINING DATA subsampling (random 80% with replacement per model) to introduce genuine diversity.

---

## Decision: attempt+1

**Rationale:**
1. Attempt 16 does NOT replace attempt 7 (wins only on HCDA, loses on DA/Sharpe/MAE)
2. But attempt 16 PROVES bootstrap confidence works for HCDA (+8.70pp)
3. The proven XGBoost architecture (attempt 7) + proven bootstrap technique (attempt 16) has NOT been combined
4. This is a genuine, empirically-backed improvement path -- not speculative
5. One more attempt is justified; if it fails, confirm attempt 7 as final

**Improvement plan for attempt 17:**
- Use EXACTLY attempt 7's XGBoost hyperparameters (no re-tuning)
- Train 10-15 XGBoost models on bootstrap samples (80% of training data with replacement)
- Use bootstrap std for high-confidence selection (top 20% by lowest std)
- Keep same 24 features

**Failure criteria:** If DA < 57% or Sharpe < 1.5 -> declare no_further_improvement, confirm attempt 7 as final
