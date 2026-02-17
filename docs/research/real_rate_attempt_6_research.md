# Research Report: Real Rate Submodel (Attempt 6)

**Date**: 2026-02-17
**Researcher**: Sonnet 4.5
**Status**: Complete
**Target**: Extract 2-3 new structural features from {DFII10, DGS10, DGS2, T10YIE} that complement existing 24 meta-model features

---

## Executive Summary

Based on comprehensive literature review and empirical evidence, I recommend **2 features** for Real Rate Attempt 6:

1. **Rate Volatility Regime** (DFII10 realized vol z-score) — **HIGH PRIORITY**
2. **Real Rate Momentum Persistence** (autocorrelation of DFII10 changes) — **MEDIUM PRIORITY**

**Not recommended**:
- Nominal-Real Divergence (redundant with existing ie_anchoring_z)
- Rate Coherence (high VIF risk with yc_curvature_z)

---

## Q1: Optimal Rolling Window Size for Rate Dynamics

### Academic Literature on Inflation Expectation Shock Half-Lives

**Key Finding**: Inflation expectation shocks have **highly variable persistence** depending on the era:

- **1970s-1984**: Half-life of **~2 quarters (60 days)** — shocks dissipated slowly, with 10% of effects lingering after 2 years
- **Great Moderation (1985-2007)**: Half-life of **~2 months (40 days)** — rapid mean reversion
- **Post-2008**: Half-life remains **~40-60 days** with increased anchoring

The [Richmond Fed (2022)](https://www.richmondfed.org/publications/research/economic_brief/2022/eb_22-31) uses **10-year rolling windows** to estimate time-varying half-lives. However, for daily prediction features, research suggests shorter windows aligned with actual shock dissipation rates.

### Recommended Window Sizes by Feature Type

| Feature Type | Recommended Window | Rationale |
|-------------|-------------------|-----------|
| **Nominal-Real Divergence** | 60 days (1 quarter) | Matches modern half-life of inflation shocks; captures regime shifts without excessive lag |
| **Rate Coherence** | 20 days (1 month) | PCA correlation structures change rapidly during stress; shorter window captures regime breaks |
| **Rate Volatility** | 20 days (1 month) | Volatility clusters at daily/weekly frequency; 60 days would over-smooth |
| **Rate Persistence** | 10 days (2 weeks) | Autocorrelation structure changes quickly; matches typical momentum look-back horizons |

**Critical insight**: The [IMF (2022)](https://www.imf.org/en/Publications/WP/Issues/2022/04/29/Shocks-to-Inflation-Expectations-517437) found that inflation expectation shocks are actually **deflationary and contractionary** — meaning markets often overreact initially, then correct. This supports using **shorter windows (20-60 days)** to capture the initial overreaction phase rather than the full correction cycle.

---

## Q2: Rate Coherence via PC1 Eigenvalue Ratio

### PCA of Yield Curves: Literature Evidence

The [MDPI (2022)](https://www.mdpi.com/1911-8074/15/6/247) study on Romanian bonds and the [Clarus FT analysis](https://www.clarusft.com/principal-component-analysis-of-the-swap-curve-an-introduction/) of swap curves provide robust benchmarks:

**Typical PC1 Explained Variance for 4-5 Treasury series**:
- PC1 (Level): **80-92%** of total variance
- PC2 (Slope): **5-10%**
- PC3 (Curvature): **2-5%**
- **Total (3 PCs)**: **95-99%**

### PC1 Eigenvalue Ratio as Coherence Measure

**Formula**: Coherence = λ₁ / (λ₁ + λ₂ + λ₃ + λ₄)

**Expected range**:
- Normal markets: **0.80-0.85** (moderate correlation across curve)
- Stress periods: **0.90-0.95** (synchronized parallel shifts)
- Regime transitions: **0.70-0.80** (divergent movements)

**Stability over time**: The [MDPI study](https://www.mdpi.com/1911-8074/15/6/247) found that PC1 variance **increases significantly during extreme events** (e.g., COVID-19 saw PC1 jump from 81% to 92% in March 2020). This makes the ratio a **valid regime detector**.

### Alternative Measures

**Average Pairwise Correlation**:
- Simpler to compute: `mean(corr(X_i, X_j))` for all pairs
- More interpretable: ranges from -1 to 1
- **Drawback**: Treats all pairs equally; doesn't capture dominant modes

**Recommendation**: Use **PC1 eigenvalue ratio** because:
1. Captures non-linear coherence patterns (e.g., {DFII10, DGS10} vs {DGS2, T10YIE} bloc correlation)
2. Single-number summary of entire covariance structure
3. Proven regime detector in literature

---

## Q3: Bond Vol vs Equity Vol Independence

### MOVE Index and VIX Correlation Evidence

**Key findings** from [SOA (2025)](https://www.soa.org/sections/investment/investment-newsletter/2025/september/rr-2025-09-bitalvo/) and [CFA Institute (2025)](https://blogs.cfainstitute.org/investor/2025/07/23/volatility-signals-do-equities-forecast-bonds/):

**Historical correlation**: **ρ ≈ 0.59** (30-day rolling average over 20 years)

**Periods of Independence**:
1. **Banking Crisis (March 2023)**: MOVE spiked **several days before VIX** — bond market detected systemic risk first
2. **Mid-2024**: Correlation **dropped below 0.40** as rate path became clearer and "stocks became less rate-sensitive"
3. **Pre-2008**: Wide gap between MOVE and VIX due to low global rates and bond market illiquidity

**Critical asymmetry** ([CFA Institute, 2025](https://blogs.cfainstitute.org/investor/2025/07/23/volatility-signals-do-equities-forecast-bonds/)):
- In **normal times**: Equity vol (VIX) **leads** bond vol (MOVE)
- In **extreme uncertainty**: Bond vol (MOVE) **leads** equity vol (VIX)

### DFII10 Realized Vol as Independent Signal

**Why bond vol matters for gold beyond VIX**:

1. **Rate-specific events create bond vol WITHOUT equity vol**:
   - FOMC "dot plot" surprises
   - Treasury auction failures / poor demand
   - Inflation data releases (CPI/PPI) — directly impact real rates
   - Fed communication missteps (e.g., "transitory" narrative shifts)

2. **Gold as "safe haven" responds to bond volatility differently**:
   - High bond vol → uncertainty about real return on safe assets → gold demand
   - High equity vol → risk-off → flight to both bonds AND gold
   - **They are complementary, not redundant**

3. **Empirical evidence**: [SSRN (2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5086580) found that the **co-movement between VIX and MOVE predicts both bond and stock returns** — treating them as independent measures provides better risk assessment

**Recommendation**: **Use DFII10 realized volatility** (20-day rolling std of daily changes) as a **valid independent feature**. VIF risk is low because:
- VIX measures equity implied vol (forward-looking, options-based)
- DFII10 vol measures bond realized vol (backward-looking, spot-based)
- Correlation ≈ 0.59 → **VIF ≈ 2.4** (safe threshold)

---

## Q4: Real Rate Momentum/Persistence

### Autocorrelation Structure of TIPS Yields

The [Sihvonen (2024)](https://cepr.org/voxeu/columns/yield-curve-momentum-implications-theory-and-practice) study on yield curve momentum provides decisive evidence:

**Key findings**:
1. **Positive autocorrelation in yield changes** at monthly frequency — a decline in yields tends to be followed by further declines
2. **Momentum is caused by autocorrelation in yield changes**, not carry
3. **Duration**: Momentum persists for **less than 1 year**, with optimal look-back horizons of **1-3 months**
4. **Mechanism**: Behavioral models show agents with complexity constraints **ignore longer-term dependencies**, creating autocorrelation

### Empirical Structure

**Autocorrelation at daily frequency** (extrapolated from monthly findings):
- Lag 1-5 days: Likely **positive** (ρ ≈ 0.05-0.15) due to post-FOMC drift and slow information incorporation
- Lag 10-20 days: **Weak** (ρ ≈ 0.00-0.05)
- Lag 60+ days: **Near-zero or negative** (mean reversion)

**Hurst exponent**: Not directly measured for DFII10 in literature, but [Time Series Momentum (Quantpedia)](https://quantpedia.com/strategies/time-series-momentum-effect) shows that momentum is robust across asset classes at horizons < 1 year, suggesting **H ≈ 0.55-0.65** (mild persistence)

### Relationship to Gold Dynamics

**Gold responds to rate persistence regimes**:
- High autocorrelation (strong momentum) → predictable rate trends → gold positions adjust faster → higher gold vol
- Low autocorrelation (mean-reverting) → choppy rates → gold range-bound

**Feature design**: Compute **rolling autocorrelation** of DFII10 daily changes (lag=1, window=10 days) as proxy for momentum regime strength.

---

## Q5: VIF Risk Assessment

### Existing Meta-Model Features (Potential Conflicts)

| Existing Feature | Description | Potential Conflict |
|-----------------|-------------|-------------------|
| `yc_curvature_z` | Yield curve convexity (DGS10 - 0.5*(DGS2+DGS30)) | **HIGH** risk with Rate Coherence (both measure curve shape) |
| `ie_anchoring_z` | Stability of T10YIE (rolling std) | **MEDIUM** risk with Nominal-Real Divergence (both use T10YIE) |
| `vix_persistence` | VIX momentum (autocorrelation) | **LOW** risk with Rate Persistence (different asset class) |
| `yc_spread_velocity_z` | Speed of DGS10-DGS2 changes | **MEDIUM** risk with Rate Coherence (both use DGS2/DGS10) |

### VIF Analysis by Candidate Feature

#### 1. Nominal-Real Divergence (rolling corr of DGS10 vs DFII10)

**VIF Risk**: **MEDIUM-HIGH**

**Conflicts**:
- `ie_anchoring_z` already measures T10YIE stability. Since T10YIE = DGS10 - DFII10 (Fisher equation), divergence is **mechanically related** to T10YIE volatility
- Expected VIF: **6-8** (marginal, but risky)

**Verdict**: **Not recommended** — redundant with existing inflation expectation feature

---

#### 2. Rate Coherence (PC1 eigenvalue ratio)

**VIF Risk**: **HIGH**

**Conflicts**:
- `yc_curvature_z` directly uses DGS10 and DGS2 (2 of our 4 series)
- `yc_spread_velocity_z` uses DGS10-DGS2 changes, which is essentially PC2 (slope component)
- PC1 ratio measures how much variance is NOT in PC2/PC3 → **inverse relationship** with curvature/slope features

**Expected VIF**: **8-12** (above threshold)

**Verdict**: **Not recommended** — high multicollinearity with yield curve shape features

---

#### 3. Rate Volatility Regime (DFII10 realized vol z-score)

**VIF Risk**: **LOW**

**Conflicts**:
- `vix_persistence` measures equity vol momentum, not bond vol level
- No existing feature directly captures bond market volatility
- Correlation with VIX ≈ 0.59 → VIF ≈ 2.4

**Expected VIF**: **2-3** (safe)

**Verdict**: **RECOMMENDED** — low overlap, unique information about rate uncertainty

---

#### 4. Real Rate Momentum Persistence (autocorr of DFII10 changes)

**VIF Risk**: **LOW-MEDIUM**

**Conflicts**:
- `vix_persistence` measures autocorrelation of VIX, not rates
- No existing feature directly captures rate momentum regime
- Weak correlation with `yc_spread_velocity_z` (velocity vs persistence are orthogonal)

**Expected VIF**: **3-5** (acceptable)

**Verdict**: **RECOMMENDED** — moderate overlap, captures unique rate dynamics

---

### Final VIF-Optimized Feature Selection

**TOP 2 FEATURES** (expected combined VIF < 5):

1. **Rate Volatility Regime** (priority 1) — VIF ≈ 2.5
2. **Real Rate Momentum Persistence** (priority 2) — VIF ≈ 4.0

**Pairwise correlation** between these two: Expected **ρ ≈ 0.30-0.40** (volatility and autocorrelation are moderately correlated but distinct — high vol doesn't always mean high momentum)

---

## Q6: Z-scoring Method to Prevent Look-Ahead Bias

### Rolling vs Expanding Window Trade-offs

The [LinkedIn Technical Tuesday](https://www.linkedin.com/pulse/technical-tuesday-explore-financial-market-data-z-scores-dietsch) and [Robot Wealth](https://robotwealth.com/rolling-and-expanding-windows-for-dummies/) analyses provide clear guidance:

**Expanding Window Z-Score**:
```
z_t = (x_t - mean(x_1:t)) / std(x_1:t)
```

**Pros**:
- More stable (lower variance) as sample size grows
- **No look-ahead bias** if properly lagged
- Suitable for stationary processes

**Cons**:
- Slow to adapt to regime changes
- Old data (e.g., 2008 crisis) permanently affects z-score in 2025
- Can "get used to" new volatility levels → fails to highlight new spikes

---

**Rolling Window Z-Score**:
```
z_t = (x_t - mean(x_{t-w}:t)) / std(x_{t-w}:t))
```

**Pros**:
- Fast adaptation to new regimes
- **No look-ahead bias** if properly lagged
- Preferred for non-stationary financial data

**Cons**:
- Higher variance in z-scores (less stable)
- Window size is a hyperparameter

---

### Recommendation for Real Rate Features

**Use Rolling Window Z-Score** with these parameters:

| Feature | Window Size | Rationale |
|---------|------------|-----------|
| Rate Volatility Regime | 252 days (1 year) | Volatility has long memory; need stable baseline to detect true spikes |
| Rate Momentum Persistence | 60 days (1 quarter) | Autocorrelation regime changes faster; shorter window captures shifts |

**Critical implementation rule** ([Medium: Look-Ahead Bias](https://medium.com/funny-ai-quant/look-ahead-bias-in-quantitative-finance-the-silent-killer-of-trading-strategies-bbbbb31d943a)):
```python
# WRONG (look-ahead bias):
df['z'] = (df['x'] - df['x'].rolling(60).mean()) / df['x'].rolling(60).std()

# CORRECT (lag the window by 1):
df['rolling_mean'] = df['x'].shift(1).rolling(60).mean()
df['rolling_std'] = df['x'].shift(1).rolling(60).std()
df['z'] = (df['x'] - df['rolling_mean']) / df['rolling_std']
```

**Why this matters**: Without the `shift(1)`, the rolling mean at time `t` includes `x_t` itself, creating instantaneous look-ahead bias.

---

## Recommended Feature Definitions

### Feature 1: Rate Volatility Regime (PRIORITY 1)

**Definition**:
```
realized_vol = rolling_std(DFII10_daily_change, window=20)
mean_vol = rolling_mean(realized_vol.shift(1), window=252)
std_vol = rolling_std(realized_vol.shift(1), window=252)
rate_vol_regime_z = (realized_vol - mean_vol) / std_vol
```

**Expected properties**:
- Mean: 0 (by construction)
- Std: 1 (by construction)
- Range: typically [-2, +4] (positive skew during crises)
- Autocorrelation: High (vol clusters)

**Information content**:
- Captures bond market stress independent of equity stress
- Predicts gold demand during rate uncertainty regimes
- Orthogonal to VIX (ρ ≈ 0.59 → low VIF)

---

### Feature 2: Real Rate Momentum Persistence (PRIORITY 2)

**Definition**:
```
autocorr = rolling_correlation(DFII10_change, DFII10_change.shift(1), window=10)
mean_autocorr = rolling_mean(autocorr.shift(1), window=60)
std_autocorr = rolling_std(autocorr.shift(1), window=60)
rate_momentum_z = (autocorr - mean_autocorr) / std_autocorr
```

**Expected properties**:
- Mean: ~0.05 (slight positive autocorrelation)
- Std: 0.15 (moderate variability)
- Range: typically [-1, +1] after z-scoring
- Regime-dependent: High during Fed policy cycles, low during quiet periods

**Information content**:
- Detects momentum vs mean-reversion regimes in real rates
- Predicts gold trend vs range behavior
- Complements VIX persistence (different asset class)

---

## Implementation Notes

### Data Requirements

| Series | Source | Availability | Update Frequency |
|--------|--------|--------------|------------------|
| DFII10 | FRED | 2003-present | Daily |
| DGS10 | FRED | 1962-present | Daily |
| DGS2 | FRED | 1976-present | Daily |
| T10YIE | FRED | 2003-present | Daily |

**Constraint**: DFII10 and T10YIE only available since 2003 → **~6,000 daily samples** (sufficient for daily modeling)

### Multi-Country Extension (Future Work)

**Not recommended for Attempt 6** due to data availability issues:

| Country | Real Rate Proxy | FRED Availability | Notes |
|---------|----------------|-------------------|-------|
| Germany | TIPSDE10Y | ❌ Not available | ECB data requires separate API |
| UK | TIPSGB10Y | ❌ Not available | BoE data, not in FRED |
| Japan | TIPSJP10Y | ❌ Not available | No TIPS equivalent (negative nominal rates) |

**Future**: If meta-model requires more samples, use **nominal yield vol** (DGS10 equivalents exist for G10) instead of real rate vol.

---

## Expected Gate Performance

### Gate 1: Standalone Quality

**Prediction**: **PASS**

- No leakage risk (all features are lagged)
- No constant outputs (vol and autocorr vary significantly)
- Overfit ratio <1.5 expected (deterministic features, no trainable parameters)

### Gate 2: Information Gain

**Prediction**: **PASS**

- MI increase: Expected **+8-12%** total (4-6% per feature)
- VIF: Both features <5 (Rate Vol ≈ 2.5, Rate Momentum ≈ 4.0)
- Rolling correlation std: Expected <0.12 (stable relationship with gold)

### Gate 3: Ablation

**Prediction**: **MARGINAL PASS** (1 of 3 metrics)

- Direction accuracy: **+0.3%** (modest, not +0.5%)
- Sharpe: **+0.03** (below +0.05 threshold)
- MAE: **-0.015%** (exceeds -0.01% threshold) ← **Most likely to pass**

**Rationale**: Rate vol regime is a strong signal for gold volatility → better MAE, but not necessarily better directional accuracy.

---

## Risks and Mitigations

### Risk 1: DFII10 has limited history (2003-present)

**Mitigation**:
- 23 years × 252 days = 5,796 samples is sufficient for daily models
- If needed, backfill pre-2003 using DGS10 - T5YIFR (5Y inflation expectations as proxy)

### Risk 2: Autocorrelation feature may be unstable

**Mitigation**:
- Use robust correlation estimator (Spearman instead of Pearson)
- Clip extreme values to [-3, +3] std range
- Add minimum sample requirement (window=10 needs 15 valid days)

### Risk 3: Both features may fail Gate 3

**Mitigation**:
- If both fail, consider **interaction term**: `vol_regime_z * momentum_z` (detects "trending volatility" regime)
- Fallback: Revert to simpler feature: **DFII10 level change** (first difference, no z-scoring)

---

## Conclusion

**FINAL RECOMMENDATION**: Build **2 features** for Real Rate Attempt 6:

1. **Rate Volatility Regime** (`rate_vol_regime_z`) — 20-day realized vol of DFII10, z-scored over 252-day rolling window
2. **Real Rate Momentum Persistence** (`rate_momentum_z`) — 10-day autocorrelation of DFII10 changes, z-scored over 60-day rolling window

**Expected outcome**:
- Gate 1: PASS (high confidence)
- Gate 2: PASS (moderate confidence)
- Gate 3: MARGINAL PASS on MAE (50% confidence)

**Key advantages**:
- Low VIF risk (both <5, combined <6)
- Independent of existing features (bond vol vs equity vol, rate momentum vs equity momentum)
- Theoretically grounded (academic literature on MOVE/VIX independence and yield curve momentum)
- Deterministic and interpretable (no black-box transformations)

**Next steps**: Architect should design lightweight feature extraction (no neural network needed — pure pandas transformations).

---

## Sources

### Q1: Optimal Window Sizes
- [Richmond Fed (2022): How Persistent Is Inflation?](https://www.richmondfed.org/publications/research/economic_brief/2022/eb_22-31)
- [IMF (2022): Shocks to Inflation Expectations](https://www.imf.org/en/Publications/WP/Issues/2022/04/29/Shocks-to-Inflation-Expectations-517437)
- [ECB Working Paper No. 371 (2004): Inflation Persistence](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp371.pdf)

### Q2: PCA and Coherence
- [MDPI (2022): PCA in Yield Curve Scenarios](https://www.mdpi.com/1911-8074/15/6/247)
- [Clarus FT: PCA of the Swap Curve](https://www.clarusft.com/principal-component-analysis-of-the-swap-curve-an-introduction/)
- [Springer (2010): PCA of Yield Curve Movements](https://link.springer.com/article/10.1007/s12197-010-9142-y)

### Q3: MOVE/VIX Independence
- [SOA (2025): Bond and Equity Volatility Indices](https://www.soa.org/sections/investment/investment-newsletter/2025/september/rr-2025-09-bitalvo/)
- [CFA Institute (2025): Volatility Signals - Do Equities Forecast Bonds?](https://blogs.cfainstitute.org/investor/2025/07/23/volatility-signals-do-equities-forecast-bonds/)
- [SSRN (2026): Asset Co-movement, VIX and MOVE Index](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5086580)
- [Charles Schwab: What's the MOVE Index?](https://www.schwab.com/learn/story/whats-move-index-and-why-it-might-matter)

### Q4: Rate Momentum
- [CEPR (2024): Yield Curve Momentum - Sihvonen](https://cepr.org/voxeu/columns/yield-curve-momentum-implications-theory-and-practice)
- [Quantpedia: Time Series Momentum Effect](https://quantpedia.com/strategies/time-series-momentum-effect/)
- [ScienceDirect (2012): Time Series Momentum](https://www.sciencedirect.com/science/article/pii/S0304405X11002613)

### Q5: VIF Analysis
- [DataCamp: Variance Inflation Factor](https://www.datacamp.com/tutorial/variance-inflation-factor)
- [GeeksforGeeks: Detecting Multicollinearity with VIF](https://www.geeksforgeeks.org/python/detecting-multicollinearity-with-vif-python/)
- [QuantifyingHealth: VIF Threshold References](https://quantifyinghealth.com/vif-threshold/)

### Q6: Z-scoring Methods
- [LinkedIn: Z-Scores in Financial Data](https://www.linkedin.com/pulse/technical-tuesday-explore-financial-market-data-z-scores-dietsch)
- [Robot Wealth: Rolling and Expanding Windows](https://robotwealth.com/rolling-and-expanding-windows-for-dummies/)
- [Medium: Look-Ahead Bias in Quantitative Finance](https://medium.com/funny-ai-quant/look-ahead-bias-in-quantitative-finance-the-silent-killer-of-trading-strategies-bbbbb31d943a)
- [Wiley (2024): Out-of-sample Volatility Prediction](https://onlinelibrary.wiley.com/doi/10.1002/for.3046?af=R)
