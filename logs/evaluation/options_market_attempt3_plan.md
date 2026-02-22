# Evaluation Summary: options_market attempt 2 + Improvement Plan for Attempt 3

## Attempt 2 Results Review

### Gate 1: Standalone Quality -- PASS
- Overfit ratio: N/A (HMM unsupervised) -- PASS
- All-NaN columns: 0 -- PASS
- Constant columns: 0 -- PASS
- Autocorrelation: 0.9749 (threshold < 0.99) -- PASS
- Optuna HPO: 30 trials completed -- PASS
- Output: 1 column `options_risk_regime_prob`, mean=0.148, std=0.312

### Gate 2: Information Gain -- FAIL (marginal)
- MI increase: 4.96% (threshold > 5%) -- FAIL by 0.04pp
- Max VIF: 2.13 (threshold < 10) -- PASS
- Stability (rolling corr std): 0.1555 (threshold < 0.15) -- FAIL by 0.006

Both failures are marginal. MI is 0.04 percentage points below threshold, stability is 0.0055 above threshold.

### Gate 3: Ablation -- PASS (MAE criterion only)

| Metric | Baseline | Extended | Delta | Threshold | Judgment |
|--------|----------|----------|-------|-----------|----------|
| Direction Accuracy | 49.15% | 48.90% | -0.24% | > +0.5% | FAIL |
| Sharpe | 0.0942 | -0.0471 | -0.1413 | > +0.05 | FAIL |
| MAE | 1.2570 | 1.1008 | -0.1562 | < -0.01 | PASS (15.6x) |

Fold-level details:

| Fold | DA Delta | MAE Delta | Sharpe Delta |
|------|----------|-----------|--------------|
| 1 | -3.68% | +0.047 | -0.531 |
| 2 | +1.72% | -0.079 | +0.316 |
| 3 | +1.72% | -0.101 | -0.015 |
| 4 | -2.44% | -0.008 | -0.803 |
| 5 | +1.46% | -0.640 | +0.327 |

Feature importance: options_risk_regime_prob = 7.55%, rank #2/20.

### Attempt 2 Decision: PASS (Gate 3 via MAE)
Attempt 2 correctly passed Gate 3 via MAE criterion (-0.1562, 15.6x threshold). However, DA and Sharpe remain negative. Since this is automation_test mode, we continue improving.

---

## Diagnosis: What Needs Fixing

### Priority 1: Gate 2 Compliance (close misses)

1. **MI increase 4.96% vs 5% threshold**: The reduction from 3 to 1 output column (attempt 1 had 17.12% MI) lost MI. Adding a second complementary column should recover MI above 5%.

2. **Stability 0.1555 vs 0.15 threshold**: Raw HMM regime probabilities exhibit sharp transitions (state switching). Smoothing with EMA (span 5-10 days) dampens these transitions and reduces rolling correlation standard deviation.

### Priority 2: DA and Sharpe Improvement

1. **DA delta = -0.24%**: The regime probability alone captures risk level but not direction. A complementary signal capturing directional information (e.g., options volume trend, SKEW change direction) could provide directional edge.

2. **Sharpe delta = -0.141**: Sharpe is penalized by both DA degradation and position change costs. Smoother regime probabilities would reduce position switching frequency, lowering transaction costs in the Sharpe calculation.

### Fold-Level Pattern Analysis

- **Folds 2, 3, 5**: DA improved (+1.46 to +1.72%). These are later periods (2018+) where options market data may be more informative.
- **Folds 1, 4**: DA degraded (-2.44 to -3.68%). Possible that HMM regime assignments are less stable in these periods.
- **Fold 5**: MAE improvement is extreme (-0.640) -- this is the 2023-2025 high-volatility period where options regime detection is most valuable.

---

## Improvement Plan for Attempt 3

### Recommended Approach: Option A -- Smoothed Regime Prob + Complementary Signal

This is the highest-probability approach because it addresses both Gate 2 failures with minimal risk to the existing strong MAE performance.

#### Component 1: EMA-Smoothed Regime Probability

```
Input:  raw options_risk_regime_prob (autocorr=0.9749, stability=0.1555)
Output: smoothed_regime_prob = EMA(raw_regime_prob, span=5-10 days)
```

Expected effects:
- Stability reduction: 0.1555 -> ~0.13-0.14 (EMA dampens sharp transitions)
- Autocorrelation: may increase slightly but still well below 0.99
- MAE: minimal impact (smoothing preserves level information)
- Sharpe: potential improvement (fewer position switches = lower transaction costs)

#### Component 2: Options Volatility Trend Signal

A second output column capturing directional momentum in options-implied volatility:

```
Candidate signals (architect to choose):
1. VIX rate-of-change z-score (5-20 day window, normalized)
   - Pro: readily available, captures fear acceleration
   - Con: VIX submodel already exists; VIF must be checked

2. SKEW change z-score (rate of change in CBOE SKEW index)
   - Pro: unique signal not captured by VIX submodel
   - Con: SKEW data availability on Kaggle may be limited

3. Put/Call ratio trend (if CBOE data available via FRED)
   - Pro: direct measure of options positioning
   - Con: availability uncertain

4. GVZ (Gold VIX) rate-of-change z-score
   - Pro: gold-specific volatility momentum
   - Con: GVZ series shorter than other options data
```

Expected effects on Gate 2:
- MI increase: +1-3% from complementary directional signal -> total MI >5%
- VIF: Must be < 10 with smoothed_regime_prob (likely fine given VIF=2.13 for single column)

Expected effects on Gate 3:
- DA: Directional signal may help push DA delta into positive territory
- Sharpe: Combined smoothing + directional signal could improve Sharpe
- MAE: Preserve the existing -0.1562 improvement

#### Output Specification

```
Columns: 2 maximum
  1. options_smoothed_regime_prob  (EMA-smoothed HMM regime probability)
  2. options_vol_trend_z           (volatility momentum z-score)

VIF constraint: < 10 between the two columns
Date range: matching target.csv
NaN handling: warmup period NaN, no forward-fill
```

### Alternative Approaches (for architect consideration)

**Option B: 3-State HMM**
- Expand from 2 to 3 states (low/medium/high risk)
- Output: 2 most discriminative state probabilities
- Risk: More states increase model complexity and may reduce stability further
- Assessment: Medium probability of success

**Option C: Different Input Features**
- Use Put/Call ratio + VIX term structure as HMM inputs instead of SKEW/GVZ
- Keep 2-state HMM
- Risk: Data availability on Kaggle is uncertain
- Assessment: Lower probability of success due to data constraints

### Recommendation to Architect

Option A is recommended as the primary approach. It is the most conservative change that directly addresses both Gate 2 failures while preserving the strong MAE performance. The smoothing component is nearly certain to fix stability (0.1555 -> <0.15). The second column has good probability of pushing MI above 5%.

The architect should:
1. Verify data availability for the second signal
2. Choose between VIX-based or SKEW-based volatility trend
3. Check VIF between the two proposed outputs
4. Design the Optuna search space for EMA span and z-score window parameters

---

## Attempt History Summary

| Attempt | Gate 1 | Gate 2 | Gate 3 | DA Delta | Sharpe Delta | MAE Delta | Decision |
|---------|--------|--------|--------|----------|--------------|-----------|----------|
| 1 | PASS | PASS (MI 17.12%) | FAIL | -1.05% (0/5 folds) | N/A | +0.018 | attempt+1 |
| 2 | PASS | FAIL (MI 4.96%, stab 0.1555) | PASS (MAE) | -0.24% (3/5 folds) | -0.141 (2/5 folds) | -0.156 (4/5 folds) | completed (automation_test: continue) |
| 3 | TBD | Target: MI >5%, stab <0.15 | Target: DA >0 or Sharpe >0 | - | - | - | Planned |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Smoothing degrades MAE | Low | Medium | EMA preserves level; test with validation |
| Second column VIF > 10 | Low | High | Pre-check VIF; drop if problematic |
| Second column adds noise (repeat attempt 1) | Medium | High | Limit to 2 columns max; require MI >0.01 per column |
| Data unavailability for chosen signal | Medium | Medium | Have fallback signals ready |

---

## State Transition

```
options_market attempt 2: PASS (Gate 3 MAE) -- automation_test continues
  -> attempt 3: resume_from = architect
  -> Improvement focus: Gate 2 compliance + DA/Sharpe improvement
  -> Max attempts remaining: 3 (attempts 3, 4, 5)
```
