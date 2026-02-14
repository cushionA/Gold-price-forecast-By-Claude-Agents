# Window Lengths for Financial Feature Engineering

## Q1: What window lengths should I use for momentum/velocity features?

**Answer**: Standard practice uses 20-day and 60-day windows for daily data.

**Common Window Lengths**:

| Window | Trading Days | Calendar Days | Financial Interpretation |
|--------|--------------|---------------|--------------------------|
| **20 days** | 1 month | ~28 days | Short-term trend |
| **60 days** | 3 months | ~84 days | Medium-term trend |
| 120 days | 6 months | ~168 days | Long-term trend |
| 252 days | 1 year | ~365 days | Annual cycle |

**Recommendation for Velocity**:
- Short-term: 20-day momentum (captures monthly trends)
- Medium-term: 60-day momentum (captures quarterly trends)
- Both normalized by rolling 60-day standard deviation

**Code Example**:
```python
# Velocity = change over window / rolling volatility
df['velocity_20d'] = (
    df['value'].diff(20) / df['value'].rolling(60).std()
)

df['velocity_60d'] = (
    df['value'].diff(60) / df['value'].rolling(60).std()
)
```

**Evidence**:
- Technical analysis standard: 20/50/200-day moving averages
- Monetary policy transmission lag: 6-12 months (supports 60-120 day windows)

**Last Verified**: 2026-02-14 (real_rate research)

---

## Q2: What window length for acceleration features?

**Answer**: 20-day change in velocity (second derivative)

**Formula**:
```
Acceleration_t = Velocity_t - Velocity_{t-20}
```

**Rationale**:
- Captures changes in momentum (trend acceleration/deceleration)
- 20 days = 1 month sensitivity
- Aligns with short-term velocity window

**Implementation**:
```python
# First compute velocity
df['velocity_20d'] = df['value'].diff(20) / df['value'].rolling(60).std()

# Then compute acceleration as change in velocity
df['accel_20d'] = df['velocity_20d'].diff(20)
```

**Last Verified**: 2026-02-14 (real_rate design)

---

## Q3: What window for rolling standard deviation (volatility)?

**Answer**: 20-day rolling standard deviation is standard.

**Common Volatility Windows**:
- **20 days** (most common): Daily volatility measure
- 60 days: Smoother, less reactive
- 252 days: Annual volatility (for risk metrics)

**Implementation**:
```python
# 20-day rolling standard deviation
df['rolling_std_20d'] = df['value'].rolling(20).std()
```

**Use Cases**:
- Normalizing momentum features (divide by volatility)
- Regime detection (high vol vs low vol)
- Risk metrics (annualized: std_20d * sqrt(252))

**Last Verified**: 2026-02-14

---

## Q4: What window for regime percentile calculation?

**Answer**: 252 days (1 trading year) is standard for regime detection.

**Rationale**:
- 252 trading days ≈ 1 calendar year
- Captures full annual cycle (seasonality)
- Long enough to identify long-term regimes
- Short enough to adapt to structural breaks

**Implementation**:
```python
# 252-day rolling percentile rank
df['regime_percentile'] = (
    df['value'].rolling(252).rank() / 252
).fillna(0.5)
```

**Interpretation**:
- 0.0-0.33: Low regime (bottom third of past year)
- 0.33-0.67: Mid regime (middle third)
- 0.67-1.0: High regime (top third)

**Last Verified**: 2026-02-14 (real_rate design)

---

## Q5: What window for autocorrelation features?

**Answer**: 60-day rolling window, lags 1/5/10

**Configuration**:
- **Rolling window**: 60 days (3 months of data for AC calculation)
- **Lags tested**: 1, 5, 10 days
- **Output**: Mean of AC(1), AC(5), AC(10) as "persistence score"

**Rationale**:
- 60 days provides enough data for stable AC estimation
- Lags 1/5/10 capture short/medium-term memory
- Averaging reduces noise

**Implementation**:
```python
# Rolling 60-day autocorrelation at lag 1
df['autocorr_20d'] = df['change_1d'].rolling(60).apply(
    lambda x: x.autocorr(lag=1) if len(x) >= 2 else 0
)
```

**Last Verified**: 2026-02-14 (real_rate design)

---

## Q6: Should window lengths differ for different features (real_rate vs VIX vs DXY)?

**Answer**: Use consistent windows across features for comparability, but adjust if data characteristics differ significantly.

**General Rule**:
- **Keep windows consistent** (20/60/252 days) across features
- Enables cross-feature comparison
- Simplifies architecture

**When to Adjust**:
- High-frequency data (e.g., VIX changes faster) → Consider shorter windows (10/30 days)
- Low-frequency data (e.g., economic indicators) → Consider longer windows (60/120 days)
- Highly volatile features → May need longer windows for stability

**Recommendation for This Project**:
- Start with 20/60/252 for all features (consistency)
- Evaluate in Attempt 2+ if feature-specific tuning is needed

**Last Verified**: 2026-02-14

---

## Q7: What if I don't have enough historical data for long windows?

**Answer**: Use the longest window you can afford, minimum 2x the window length.

**Minimum Data Requirements**:

| Window Length | Minimum Historical Data | Reason |
|---------------|-------------------------|--------|
| 20 days | 40-60 days | Need 2-3x for stable statistics |
| 60 days | 120-180 days | Rolling window needs warmup |
| 252 days | 504 days (2 years) | Full cycle + warmup |

**For This Project**:
- Schema starts: 2015-01-30
- Need 252-day windows → Fetch data from 2013-06-01 (gives ~1.5 year buffer)
- After dropping initial NaNs: 2,523 clean observations

**Fallback**:
- If insufficient data: Reduce window (e.g., 120 days instead of 252)
- Or use forward-fill with caution (max 5 days per CLAUDE.md)

**Last Verified**: 2026-02-14 (builder_data implementation)
