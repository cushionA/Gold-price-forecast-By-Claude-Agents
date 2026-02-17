# Submodel Design Document: CNY Demand Proxy (Attempt 2)

## 0. Fact-Check Results

| Claim | Result | Detail |
|-------|--------|--------|
| Yahoo Finance CNY=X | CONFIRMED | 3156 rows (2014-01-01 to 2026-02-17). Onshore USD/CNY rate. Values [6.10, 7.35]. |
| Yahoo Finance CNH=X | DOES NOT EXIST | Ticker returns no data. All variants (USDCNH=X, CNHUSD=X) also unavailable. |
| Yahoo Finance CNH=F | **CONFIRMED** | 3035 rows (2014-01-02 to 2026-02-17). Offshore CNH futures. Values [6.03, 7.41]. Max gap 5 days. Only 1 move >2% (2015-08-11 devaluation: +3.32%). |
| CNY-CNH spread exists | CONFIRMED | 3028 overlapping rows. Mean spread -0.011, std 0.030. Range [-0.224, +0.124]. |
| Spread change z-score (5d/120d) corr | CONFIRMED | corr(spread_change_z, gold_ret_next) = +0.0424, p=0.022. **Statistically significant.** |
| Spread level z-score (60d) corr | CONFIRMED | corr(spread_level_z, gold_ret_next) = +0.0383, p=0.037. Also significant. |
| Momentum z-score (5d/40d) corr | CONFIRMED | corr(momentum_z, gold_ret_next) = +0.0204, p=0.264. **NOT significant.** |
| Spread z-score autocorrelation | CONFIRMED | Spread change z (5d/120d): 0.319 (excellent). Spread level z (60d): 0.442. Both well below 0.99. |
| VIF with existing features | CONFIRMED | All correlations with existing 20 meta-model features below |0.13|. Essentially orthogonal. |

### Critical Findings

**1. CNH=F (offshore CNH futures) is available with full 2014-2026 daily data.** The architect's initial fact-check tested CNH=X (which doesn't exist) but missed CNH=F. This enables the onshore-offshore spread approach (Idea A).

**2. Spread change z-score dominates momentum z-score.** corr +0.0424 (p=0.022) vs +0.0204 (p=0.264). The spread signal is 2x stronger and statistically significant; momentum is not.

**3. Spread change z-score has the lowest autocorrelation.** 0.319 vs momentum's 0.74. This means the spread signal is more informative per observation (less redundancy between consecutive days).

**4. CNH=F data quality is excellent.** Max gap 5 days, only 1 large move >2% (2015 devaluation). No systematic rollover gaps.

### Design Decision

**Selected: Spread Change Z-Score (single output)** — onshore-offshore CNY spread dynamics.

Rationale:
- 2x higher correlation with gold_ret_next than momentum z-score
- Statistically significant (p=0.022)
- Lowest autocorrelation (0.319) among all candidates
- Captures capital control tension, a unique China-specific signal orthogonal to all existing features
- Single output follows the options_market success pattern (3→1 column)

---

## 1. Overview

- **Purpose**: Extract the onshore-offshore CNY spread momentum as a z-scored signal. The CNY-CNH spread reflects capital control tension, PBOC intervention intensity, and cross-border capital flow pressure — a unique China-specific gold demand proxy.
- **Core method**: Compute daily CNY-CNH spread change, accumulate over N days, z-score against a rolling baseline. Pure deterministic computation.
- **Why spread > momentum**: The spread captures information from TWO markets (onshore CNY managed by PBOC, offshore CNH free-floating), encoding policy tension that raw CNY momentum alone cannot capture. Empirically: corr +0.042 (significant) vs +0.020 (not significant).
- **Why single output**: Attempt 1's 3 columns degraded DA by -2.06%. Single output follows options_market's success pattern.

### Key Advantage

- Statistically significant gold predictive signal (p=0.022)
- Lowest autocorrelation of any candidate (0.319)
- Fully deterministic, no ML model, no pip installs
- Captures a genuinely unique signal: capital control tension between onshore and offshore yuan markets

---

## 2. Data Specification

### Primary Data

| Data | Source | Ticker | Frequency | Expected Rows |
|------|--------|--------|-----------|---------------|
| CNY/USD onshore rate | Yahoo Finance | CNY=X | Daily | ~3,156 |
| CNH/USD offshore futures | Yahoo Finance | CNH=F | Daily | ~3,035 |
| Gold futures (for Optuna only) | Yahoo Finance | GC=F | Daily | ~3,000 |

### Data NOT Used

| Data | Source | Reason |
|------|--------|--------|
| FRED DEXCHUS | FRED | Same as CNY=X (onshore rate). Redundant. |
| CNH=X | Yahoo Finance | Does not exist. |

### Preprocessing Steps

1. Fetch CNY=X and CNH=F from Yahoo Finance, start='2014-01-01'
2. Fetch GC=F for gold returns (Optuna objective only)
3. Align all three on common trading dates (inner join)
4. Compute: `spread = CNY_close - CNH_close`
5. Compute: `spread_change = spread.diff()` (daily change in spread)
6. Compute spread change momentum and z-score (see Section 3)
7. Trim output to base_features date range: 2015-01-30 to latest

### Expected Sample Count

- Aligned overlap: ~3,028 rows (2014-01-02 to 2026-02-17)
- After warmup (momentum + baseline windows): ~2,850-2,900 rows
- After trimming to base_features range: ~2,700+ rows

---

## 3. Model Architecture (Deterministic)

### Single Feature: cny_demand_spread_z

Pure deterministic computation. No ML model.

#### Computation

```
Input: CNY=X Close [T], CNH=F Close [T]
       |
   spread = CNY_close - CNH_close        # onshore-offshore spread
       |
   spread_change = spread.diff()          # daily spread change
       |
   spread_mom = spread_change.rolling(W_mom).sum()  # N-day cumulative spread change
       |
   rolling_mean = spread_mom.rolling(W_base).mean()
   rolling_std  = spread_mom.rolling(W_base).std()
       |
   z = (spread_mom - rolling_mean) / rolling_std
       |
   z = z.clip(-4, 4)
       |
Output: cny_demand_spread_z [T x 1]
```

#### Interpretation

- **Positive values**: Spread is widening faster than recent norm. Capital outflow pressure increasing. PBOC tightening controls. Risk-off signal for Chinese gold demand (historically positive for gold).
- **Negative values**: Spread is narrowing faster than recent norm. Capital controls relaxing. Reduced urgency for gold hedging.
- **Near zero**: Spread dynamics are typical for recent conditions.

#### Properties (Measured)

- Correlation with gold_ret_next: +0.0424 (p=0.022, significant)
- Autocorrelation (lag-1): ~0.319 (excellent, well below 0.99)
- Standard deviation: ~1.04 (healthy z-score distribution)
- VIF against existing 20 meta-model features: all |r| < 0.13

### Output Specification

| Column | Range | Description |
|--------|-------|-------------|
| `cny_demand_spread_z` | [-4, +4] | CNY-CNH spread change momentum z-score |

Single column. Captures onshore-offshore capital control tension.

---

## 4. Hyperparameters

### Fixed Values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Z-score clipping | [-4, 4] | Standard across all submodels |
| Forward-fill max gap | 3 days | Handle holiday gaps |

### Optuna Exploration Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| momentum_window | int, [3, 10] | linear | Days of spread change to accumulate. Best measured: 5d. |
| baseline_window | int, [30, 120], step=10 | linear | Z-score baseline. Best measured: 120d. |

### Exploration Settings

- **n_trials**: 30
- **timeout**: 300 seconds
- **objective**: `abs(pearson_corr) + MI * 10` on validation set
- **direction**: maximize
- **sampler**: TPESampler(seed=42)

---

## 5. Training Settings

No fitting. Purely deterministic computation.

### Data Split

- train/val/test = 70/15/15 (time-series order)
- Optuna evaluates on validation set
- Test set reserved for evaluator Gate 3

---

## 6. Kaggle Execution Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | No ML model. Pure pandas. |
| Estimated execution time | 2-3 minutes | Data download + 30 Optuna trials |
| Required pip packages | None | All pre-installed on Kaggle |

---

## 7. Implementation Instructions

### builder_model Instructions

#### train.ipynb Structure

1. Libraries (no pip install needed)
2. Data Fetching: CNY=X, CNH=F, GC=F from yfinance
3. Spread computation: `spread = CNY - CNH`, `spread_change = spread.diff()`
4. Feature generation function: `generate_spread_z(spread_change, mom_window, base_window)`
5. Optuna HPO: 30 trials optimizing window parameters
6. Final output generation with best params
7. Save submodel_output.csv and training_result.json

#### Key Implementation Notes

1. **CNH=F is a futures contract**: Values are in the same units as CNY=X (USD/CNH). The spread CNY-CNH is typically small (-0.22 to +0.12).
2. **Single output column**: `cny_demand_spread_z`. This replaces the 3-column output from attempt 1.
3. **No HMM, no random seeds**: Fully deterministic given data and parameters.
4. **Gold returns for Optuna only**: GC=F used solely for validation objective. NOT used in feature computation.
5. **No lookahead**: Rolling windows are backward-looking. Spread at time t uses only CNY(t) and CNH(t).
6. **NaN handling**: First `momentum_window + baseline_window` rows will be NaN. Leave as NaN.
7. **kernel-metadata.json**: id=`bigbigzabuton/gold-model-training`, enable_gpu=false, dataset_sources includes `bigbigzabuton/gold-prediction-submodels`.

---

## 8. Risks and Alternatives

### Risk 1: CNH=F Rollover Gaps

- **Description**: Futures contracts have rollover dates. CNH=F showed only 1 move >2% (2015 devaluation).
- **Mitigation**: Yahoo Finance provides continuous contract data. Max gap is 5 days (same as CNY=X).
- **Fallback**: If rollover artifacts detected, apply 3-day median filter.

### Risk 2: Single Feature May Still Fail Gate 3

- **Description**: Even with significant correlation (+0.042, p=0.022), the signal is weak in absolute terms.
- **Mitigation**: The signal is orthogonal to all 20 existing features. Even weak orthogonal signals add marginal value.
- **Fallback**: If attempt 2 fails Gate 3, declare cny_demand as "no further improvement".

### Risk 3: Spread Dynamics Changed Over Time

- **Description**: CNH market matured since 2014. Early spread dynamics may differ from recent.
- **Mitigation**: Z-scoring against a rolling baseline adapts to changing spread dynamics. Optuna selects the baseline window that generalizes best.

---

## 9. VIF Analysis

| Feature | Max |r| with existing features | Assessment |
|---------|-----------------------------------|------------|
| cny_demand_spread_z | < 0.13 | Excellent. Essentially orthogonal. |

The onshore-offshore spread captures a unique signal (capital control tension) that no existing submodel addresses.

---

## 10. Expected Performance Against Gates

### Gate 1: PASS (high confidence)
- Deterministic, no overfit. Autocorrelation 0.32 (well below 0.99).

### Gate 2: MARGINAL
- VIF excellent (~1.02). MI uncertain. Stability likely PASS.

### Gate 3: 55-65% probability of PASS
- Significant correlation (p=0.022) suggests real signal.
- MAE improvement likely (attempt 1 achieved -0.0658 with weaker signal).
- DA improvement possible (spread signal is stronger and less noisy than 3-column attempt 1).

### Improvement Over Attempt 1

| Aspect | Attempt 1 | Attempt 2 |
|--------|-----------|-----------|
| Output columns | 3 | **1** |
| Architecture | HMM + z-scores | **Deterministic spread z-score** |
| Correlation with gold_ret_next | ~0.02 (not significant) | **+0.042 (p=0.022)** |
| Autocorrelation | 0.64-0.92 | **0.32** |
| Dependencies | hmmlearn | **None** |
| DA delta | -2.06% | Target: >= 0% |
| Unique signal | CNY dynamics only | **Onshore-offshore tension** |

**Confidence**: 7/10 — Statistically significant signal with excellent orthogonality. Main risk is signal weakness in absolute terms.
