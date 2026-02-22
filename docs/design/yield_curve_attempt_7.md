# Submodel Design Document: Yield Curve (Attempt 7)

## 0. Context: Attempt 6 Evaluation

| Gate | Result | Detail |
|------|--------|--------|
| Gate 1 | PASS | All autocorr < 0.95; VIF max 1.18; cross-corr 0.065; stability 0.139 |
| Gate 2 | FAIL | MI increase only +1.95% (below 5% threshold) |
| Gate 3 | FAIL | DA −0.574pp, Sharpe −0.249, MAE −0.0007 |

**Normal decision**: no_further_improvement
**Automation test override**: continue to attempt 7 (final attempt, max_attempt=7)

### Track Record Summary

| Att | Approach | Gate 1 | Gate 2 | Gate 3 |
|-----|----------|--------|--------|--------|
| 1 | HMM 2-state | FAIL | FAIL | PASS (MAE) |
| 2 | Velocity z-scores: 10y-3m spread + curvature | PASS | PASS | PASS (BEST) |
| 3 | Optimized z-score windows (level variables) | FAIL | PASS | FAIL |
| 4 | 2nd-order acceleration + structural decomp | PASS | PASS | FAIL |
| 5 | Cross-tenor correlation dynamics | PASS | PASS | FAIL |
| 6 | Bond vol regime + DFII10 momentum | PASS | FAIL | FAIL |

---

## 1. Overview

- **Approach**: 2Y Policy Velocity + 2Y-10Y Slope Velocity
- **Core hypothesis**: The 2-year Treasury yield is the most policy-sensitive rate for gold. Unlike the 3-month (current overnight liquidity; used in att2) or the 10-year (long-run growth + inflation), the 2Y captures the Fed's expected policy path over the next 2 years — the most actionable window for gold positioning. The 10Y-2Y spread is the canonical recession forecasting indicator.

**Why different from previous attempts**:
- att2 used `(DGS10 − DGS3MO)` velocity — overnight-rate-anchored
- att7 uses `DGS2` velocity and `(DGS10 − DGS2)` velocity — policy-expectation-anchored
- The 2Y yield is driven by Fed meeting decisions and forward guidance, not current liquidity
- att6 used DFII10 (real rate) — att7 uses DGS2 (nominal policy expectation), which is different

**Economic intuition**:
- Rising DGS2 = market expects more Fed hikes → higher opportunity cost for gold → gold-negative
- Falling DGS2 = market expects Fed cuts → lower opportunity cost → gold-positive
- Flattening 10Y-2Y (spread falling) = recession concerns / Fed overtightening → gold-positive (safe haven)
- Steepening 10Y-2Y (spread rising) = growth recovery / fiscal concerns → complex gold signal

---

## 2. Features

### Feature 1: `yc_2y_vel_z`
Z-scored daily change in the 2Y Treasury yield (DGS2).

```
Input: DGS2 [T x 1]
  |
dgs2_vel = DGS2.diff()   # daily change in 2Y yield
  |
z = rolling_zscore(dgs2_vel, window)
  |
clip(-4, 4)
  |
Output: yc_2y_vel_z
```

- Positive = 2Y yields rising (Fed expected to hike) → gold-negative
- Negative = 2Y yields falling (Fed expected to cut) → gold-positive
- Expected autocorr: < 0.10 (daily changes in yields are roughly IID)

### Feature 2: `yc_2y10y_vel_z`
Z-scored daily change in the (DGS10 − DGS2) spread.

```
Input: DGS10, DGS2 [T x 2]
  |
spread_2y10y = DGS10 - DGS2
spread_vel = spread_2y10y.diff()
  |
z = rolling_zscore(spread_vel, window)
  |
clip(-4, 4)
  |
Output: yc_2y10y_vel_z
```

- Positive (steepening) = long-term yields rising faster than 2Y → growth optimism or inflation fears
- Negative (flattening) = 2Y rising faster than 10Y → recession concerns, Fed overtightening → gold-positive

**Note on independence**: yc_2y_vel_z = Δ(DGS2); yc_2y10y_vel_z = Δ(DGS10 - DGS2) = Δ(DGS10) - Δ(DGS2). These are not mechanically identical. The DGS10 component adds orthogonal information. Expected internal correlation: moderate (the DGS2 change appears in both but with opposite signs).

---

## 3. Hyperparameters

### Optuna Search Space

| Parameter | Range | Rationale |
|-----------|-------|-----------|
| zscore_window | {20, 30, 45, 60, 90, 120} | 6 values covering short to long normalization windows |

- **n_trials**: 25 (good coverage of 6 unique values)
- **timeout**: 300 seconds (5 minutes)
- **objective**: Maximize MI sum on validation set
- **sampler**: TPESampler(seed=42)

---

## 4. Expected Gate Performance

### Gate 1
- autocorr: PASS expected (daily yield changes are roughly IID; autocorr expected < 0.15)
- VIF: PASS expected (not perfectly correlated despite shared DGS2 component)
- No constant output: PASS (yields change every trading day)

### Gate 2
- MI: Uncertain. The 2Y policy rate is a key gold driver. However, if att2's 3M-10Y velocity already captures the same information through the short end, MI gain may be < 5%.
- VIF vs existing: Low expected (2Y dynamics different from 3M; the 10Y-2Y spread different from 10Y-3M spread)
- Stability: PASS expected (velocity features are stable)

### Gate 3
- Uncertain. The 2Y policy channel is economically meaningful for gold but the track record shows consistent Gate 3 failures post-att2.
- Most likely path if PASS: via Sharpe or DA (policy rate signals have more directional content than pure noise)

**Overall confidence**: 40-50% PASS. The 2Y policy channel is genuinely different from previous approaches. If anything can improve on att2, it's the more actionable 2Y tenor. But the pattern of consistent Gate 3 failures warrants caution.

---

## 5. Data Requirements

| Series | Source | Ticker | Notes |
|--------|--------|--------|-------|
| 10Y Treasury | FRED | DGS10 | Available in base_features |
| 2Y Treasury | FRED | DGS2 | Fetched in att2/att5 |
| 3M Treasury | FRED | DGS3MO | Available (fetched for reference) |

All three already fetched in previous attempts. No new data sources.
