# Research Report: Yield Curve Dynamics (Attempt 1)

**Date:** 2026-02-15
**Feature:** yield_curve
**Attempt:** 1
**Agent:** researcher (Sonnet)

---

## Executive Summary

Yield curve dynamics (steepening, flattening, inversion regimes, velocity of curve shifts, and curvature measures) provide critical context for gold price prediction beyond raw yield levels. This research confirms that:

1. **Daily FRED data is available** for DGS10, DGS2, DGS5, and DGS30 (with 2002-2006 gap for DGS30)
2. **HMM regime detection** is a proven approach for identifying yield curve regimes from changes in spread dynamics
3. **Yield curve has strong predictive power for recessions** (Estrella & Mishkin 1998), though direct evidence for gold returns is limited
4. **Spread changes exhibit moderate autocorrelation** while spread levels show near-unit-root persistence
5. **Multi-country daily yield data is NOT available from FRED** (only monthly)
6. **Butterfly/curvature measures** capture convexity of the curve orthogonal to simple slope

**Critical VIF Concern Confirmed:** Base features already contain 3 yield curve columns (DGS10, DGS2, yield_spread) plus real_rate (DFII10, highly correlated with DGS10). All submodel outputs MUST operate on changes, regimes, or higher-order derivatives to avoid VIF >10.

---

## Research Questions & Findings

### Question 1: Optimal Regime Detection Method

**Research Question:** What is the most effective method for detecting yield curve regime shifts from daily data? HMM on spread changes (1D) vs 2D [DGS10_chg, DGS2_chg] vs Markov Regime-Switching vs threshold-based?

**Findings:**

#### HMM Applications to Financial Regimes

Hidden Markov Models (HMMs) are widely used for market regime detection in financial markets. HMMs model underlying market dynamics as a system transitioning between distinct states, with each state having characteristic behavior patterns in returns, volatility, and other metrics. The states are "hidden"—never observed directly—with only hints visible through observable data.

**Successful Applications:**
- [Market Regime Detection Using Hidden Markov Models](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Regime-based portfolio optimisation: A Hidden Markov Model approach for fixed income portfolios](https://www.esm.europa.eu/publications/regime-based-portfolio-optimisation-hidden-markov-model-approach-fixed-income)
- Principal Component Analysis (PCA) applied to fixed income indices represents the yield curve, with the variance of the first principal component used to fit an HMM identifying high and low volatility regimes

#### Dimensionality Considerations

**1D HMM on Spread Changes:**
- Pros: Simpler, follows VIX submodel pattern (1D HMM on log-VIX changes succeeded)
- Cons: Spread changes are small on daily frequency; may produce noisy regime detection
- The 10Y-2Y spread captures the overall slope of the curve

**2D HMM on [DGS10_chg, DGS2_chg]:**
- Pros: Captures parallel shifts vs twist moves; follows technical submodel success (2D HMM on [returns, GK_vol])
- Cons: DGS10 and DGS2 changes may correlate highly (both Treasury yields); need to verify distinct information
- Can distinguish: (a) parallel shift (both change together), (b) flattening (DGS2 rises more than DGS10), (c) steepening (DGS10 rises more than DGS2)

**Threshold-Based Regime Detection:**
- Simple rule: spread < 0 = inversion regime, spread ∈ [0, 50bp] = flat regime, spread > 50bp = steep regime
- Pros: No overfitting, interpretable
- Cons: Loses information from transition dynamics; no probabilistic output; arbitrary thresholds

#### Recommendation (Priority Order)

1. **2D HMM on [DGS10_chg, DGS2_chg]** (Primary recommendation)
   - Captures richer dynamics (parallel vs twist vs steepening/flattening)
   - Follows technical submodel success pattern (2D HMM)
   - Provides regime transition probabilities valuable for meta-model
   - Implementation: 3-state HMM (e.g., steepening, neutral, flattening regimes)

2. **1D HMM on spread changes** (Fallback if 2D produces high VIF)
   - Simpler, follows VIX pattern
   - Lower risk of overfitting
   - Directly models the slope dynamics

3. **Hybrid: HMM + threshold** (Alternative)
   - Use HMM for regime detection on changes
   - Add threshold flag for inversion events (spread < 0)
   - Combines transition dynamics with critical structural threshold

**Critical Design Principle:** HMM MUST be fitted on CHANGES (first differences of yields or spread), NOT levels. Levels have near-unit-root autocorrelation and would violate Gate 1 (autocorr < 0.99).

---

### Question 2: Additional Maturity Points (DGS30, DGS5, DFF)

**Research Question:** Are DGS30, DGS5, DFF available daily from FRED with sufficient history? Would curvature measures provide orthogonal information to the 10Y-2Y slope?

**Findings:**

#### Data Availability from FRED

**DGS30 (30-Year Treasury Yield):**
- **Availability:** 1977-02-15 to 2026-02-11
- **Critical Gap:** Discontinued 2002-02-18, reintroduced 2006-02-09
- **Coverage for 2015-2025:** YES (gap ends before target period)
- **Frequency:** Daily
- **Source:** [Market Yield on U.S. Treasury Securities at 30-Year Constant Maturity](https://fred.stlouisfed.org/series/DGS30)

**DGS5 (5-Year Treasury Yield):**
- **Availability:** 1962-01-02 to present
- **Coverage for 2015-2025:** YES (continuous coverage)
- **Frequency:** Daily
- **Source:** [Market Yield on U.S. Treasury Securities at 5-Year Constant Maturity](https://fred.stlouisfed.org/series/DGS5)

**DFF (Federal Funds Effective Rate):**
- **Availability:** 1954-07-01 to present (from FRED documentation)
- **Coverage for 2015-2025:** YES
- **Frequency:** Daily
- **Note:** Front-end of curve; DGS2 - DFF captures near-term policy expectations

#### Curvature Measures: Butterfly Spreads

**Butterfly Spread Formula:**
Butterfly = 2*Body - Wing1 - Wing2

For yields: **Curvature = 2*DGS10 - DGS2 - DGS30**

**What This Measures:**
- Butterfly spreads measure **curvature** (convexity) of the yield curve, not directional slope
- Captures whether the middle maturity (10Y) is trading rich or cheap relative to wings (2Y, 30Y)
- Different from slope: slope measures steepening/flattening, curvature measures bowing/humping
- Sources: [BrokerTec RV Curve and Butterflies](https://www.cmegroup.com/markets/brokertec/brokertec-rv-curve-and-butterflies.html), [Yield Curve Shapes and Movements](https://financialanalystguide.com/cfa-level-1/volume-6-fixed-income/chapter-7-the-term-structure-of-interest-rates/yield-curve-shapes-and-movements-flattening-steepening-butterfly/)

**Alternative: 5Y Belly Curvature:**
Belly Curvature = DGS5 - (DGS2 + DGS10)/2

**VIF Risk Assessment:**
- **Concern:** Does curvature correlate with spread level?
- **Theory:** Curvature and slope are CONCEPTUALLY orthogonal (convexity vs tilt)
- **Empirical verification needed:** Architect must measure correlation between curvature and yield_spread base feature
- **Mitigation:** Use CHANGES in curvature (daily diff or z-score of changes), not levels

#### Recommendation

1. **Use DGS5 for belly curvature** (safer than DGS30 due to no gap)
   - Formula: `curvature_raw = DGS5 - 0.5*(DGS2 + DGS10)`
   - Then compute: `yc_curvature_z = zscore(diff(curvature_raw), window=60)`
   - This is a first difference of z-score, reducing autocorrelation

2. **Verify curvature-spread correlation** in architect phase before finalizing
   - If correlation(curvature, yield_spread) > 0.7, switch to spread volatility measure instead

3. **DFF likely unnecessary** given DGS2 already captures front-end

---

### Question 3: Optimal Lookback Windows

**Research Question:** What lookback windows are optimal for (a) spread change velocity z-score, (b) spread volatility, (c) HMM training to avoid autocorrelation >0.99?

**Findings:**

#### VIX Submodel Precedent

The VIX submodel (which succeeded with Gate 3 pass) used:
- **60-day rolling window** for z-score normalization
- **Autocorrelation:** 0.85-0.95 (passed Gate 1 threshold <0.99)
- **Regime probability autocorrelation:** Expected to be marginally high (0.15-0.21 stability, accepted as precedent)

Source: State.json evaluation history

#### Empirical Evidence from Research

**Z-Score Lookback Windows:**
- Default lookback of **50 bars** commonly used for VIX z-score ([Z-Score Normalized VIX Strategy](https://www.tradingview.com/script/6ZkpMLDe-Z-Score-Normalized-VIX-Strategy/))
- **52-week lookback** with 20-day rolling window produced best statistical results in backtesting studies ([Z-Score Normalized Volatility Indices](https://www.tradingview.com/script/xJX1siYz-Z-Score-Normalized-Volatility-Indices/))
- **252-day core window** (1 trading year) for structural regime detection

**Yield Spread Persistence:**
- Yield spreads exhibit **high time-series persistence** (obstacle to inference)
- Momentum in Treasury returns caused by autocorrelation in yield changes (a yield decline tends to be followed by further declines)
- Sources: [Yield curve momentum](https://cepr.org/voxeu/columns/yield-curve-momentum-implications-theory-and-practice), [The crucial role of the five-year Treasury](https://www.sciencedirect.com/science/article/abs/pii/S1057521923003447)

#### Autocorrelation Concerns

**Spread LEVELS:** Near unit-root behavior
- "There is not currently available parsimonious conditions that determine the order of integration of the yield curve" ([Stationarity and the Term Structure](https://ideas.repec.org/p/fip/feddwp/0811.html))
- Continuous-time finance models typically imply stationary yield curves, but empirical evidence shows near-I(1) behavior
- **Critical:** NEVER use raw spread levels or spread level z-scores as output (will fail Gate 1)

**Spread CHANGES:** Moderate autocorrelation
- Yield changes show autocorrelation (momentum effect)
- Quarterly persistence of **0.974** measured for Treasury bill rates ([The crucial role of the five-year Treasury](https://www.sciencedirect.com/science/article/abs/pii/S1057521923003447))
- Daily changes have lower persistence but still exhibit autocorrelation

#### Recommendation (Priority Order)

1. **Spread Velocity Z-Score:**
   - Compute: `spread_change_5d = spread[t] - spread[t-5]` (5-day change)
   - Then: `yc_spread_velocity_z = zscore(spread_change_5d, window=60)`
   - 5-day change reduces noise from single-day jitter
   - 60-day z-score window follows VIX success pattern
   - Expected autocorrelation: 0.85-0.95 (should pass Gate 1)

2. **Spread Volatility (if using instead of curvature):**
   - Compute: `spread_vol_20d = rolling_std(spread_changes, window=20)`
   - Then: `yc_spread_vol_z = zscore(spread_vol_20d, window=60)`
   - 20-day vol window captures short-term turbulence
   - 60-day z-score window for normalization

3. **HMM Training Window:**
   - Use **all available training data** (no rolling window for HMM fitting)
   - HMM should be fit once on the training set, then used for inference on val/test
   - Number of states: **3 states** (follows VIX/technical/cross_asset pattern)
   - States likely represent: (a) steepening regime, (b) neutral regime, (c) flattening/inversion regime

**Critical Design Principle:** All rolling windows must be BACKWARD-LOOKING to avoid lookahead bias. Z-score at time t uses only data from t-window to t.

---

### Question 4: VIF Verification with Base Features

**Research Question:** What is the empirical correlation between yield curve dynamics features (spread change z-score, regime prob, curvature) and existing base features (DGS10, DGS2, yield_spread, DFII10)?

**Findings:**

#### Correlation Structure (Theoretical)

**DGS10 vs DFII10 (Real Rate):**
- DFII10 (10Y TIPS real yield) = DGS10 (10Y nominal yield) - T10YIE (breakeven inflation)
- Expected correlation: **Very high (0.7-0.9)**
- Both move together when inflation expectations are stable
- **Implication:** Yield curve features must NOT proxy for yield levels or they will correlate with both DGS10 and DFII10

**Spread Change Velocity vs Spread Level:**
- Theory: **Orthogonal** (change vs level)
- Z-score of 5-day spread change captures RATE OF MOVEMENT
- Spread level captures WHERE the curve IS
- Expected correlation: **Low (<0.3)**
- Empirical analogy: VIX regime vs VIX level (measured 0.2-0.4 in VIX submodel)

**HMM Regime on Spread Changes vs Spread Level:**
- Regime of changes = which STATE the dynamics are in (steepening/flattening/transition)
- Spread level = static position
- Expected correlation: **Low to moderate (0.2-0.5)**
- Precedent: VIX regime_prob vs vix_vix base feature (VIX submodel passed VIF)

**Curvature vs Spread:**
- Needs empirical verification
- Curvature (2*DGS10 - DGS2 - DGS30) measures convexity
- Spread (DGS10 - DGS2) measures slope
- If DGS30 and DGS10 move in lockstep, curvature may partially correlate with spread
- **Mitigation:** Use changes in curvature, not curvature levels

#### Known Autocorrelation Issues

**From Cross_Asset Attempt 1 Failure:**
- Gold/copper ratio z-score had autocorrelation **0.959** (>0.99 threshold)
- Root cause: z-score of smooth persistent ratio
- **Solution applied:** Use first differences of z-scores or z-scores of changes

**From Real_Rate Failures:**
- Monthly-to-daily forward-fill created step functions with autocorr >0.99
- Yield curve does NOT have this problem (daily native frequency)
- But spread LEVELS are highly persistent (near unit-root)

#### Architect Verification Checklist

The architect MUST empirically measure these correlations:

1. **VIF against base features:**
   - `corr(yc_regime_prob, yield_curve_dgs10)` — expect <0.5
   - `corr(yc_regime_prob, yield_curve_yield_spread)` — expect <0.5
   - `corr(yc_regime_prob, real_rate_real_rate)` — expect <0.4
   - `corr(yc_spread_velocity_z, yield_curve_yield_spread)` — expect <0.3
   - `corr(yc_curvature_z, yield_curve_yield_spread)` — **critical to verify <0.7**

2. **VIF against existing submodel outputs:**
   - `corr(yc_regime_prob, vix_regime_probability)` — expect 0.2-0.5 (both macro regimes)
   - `corr(yc_spread_velocity_z, xasset_recession_signal)` — expect 0.2-0.3
   - All pairwise VIF must be <10

3. **Autocorrelation:**
   - `autocorr(yc_regime_prob, lag=1)` — expect 0.85-0.95 (follows VIX pattern)
   - `autocorr(yc_spread_velocity_z, lag=1)` — expect <0.95
   - `autocorr(yc_curvature_z, lag=1)` — expect <0.95

**If any VIF >10 or autocorr >0.99:** Return to researcher with specific findings, iterate on feature design.

---

### Question 5: Mean-Reversion Characteristics

**Research Question:** How does the yield curve's mean-reversion behavior compare to VIX? What is the half-life of spread deviations? Are there structural breaks in the mean?

**Findings:**

#### VIX Mean-Reversion (For Comparison)

- VIX mean-reverts to ~18-20 with right-skewed spikes
- Spikes decay with half-life of ~2-4 weeks
- Mean-reversion feature (distance from long-term mean) was a core VIX submodel feature

#### Yield Spread Persistence

**Historical Behavior:**
- The 10Y-2Y spread has highly persistent dynamics
- "Higher quantiles of the previous spread exert a stronger influence on the current spread, indicating a positive persistence mechanism" ([Stationarity and the Term Structure](https://ideas.repec.org/p/fip/feddwp/0811.html))
- Quarterly persistence of **0.974** for Treasury bill rates
- This is MUCH higher persistence than VIX (near unit-root)

**Implication for Mean-Reversion:**
- Yield spreads do NOT mean-revert as quickly as VIX
- Spreads can stay inverted or steep for extended periods (months to years)
- Mean-reversion half-life is likely **several months**, not weeks

#### Structural Breaks in Spread Mean

**Pre-2008 vs Post-2008 vs Post-COVID:**
- Pre-GFC (1990s-2007): Average 10Y-2Y spread ~150-200bp (normal steepness)
- Post-GFC (2008-2019): Average spread ~100-150bp (flatter new normal)
- COVID era (2020-2021): Spread compressed to 0-50bp then steepened to 250bp
- 2022-2024: Aggressive inversion (spread negative)

**Evidence:**
- "There is hence a poor theoretical understanding of the determinants of the stationarity properties of the term structure" ([The Treasury Yield Curve as a Cointegrated System](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/treasury-yield-curve-as-a-cointegrated-system/E5075FCBDB4870EB7B5859012B17C83F))
- Structural breaks are likely present

#### Recommendation

**DO NOT use a mean-reversion feature for yield curve:**
- Unlike VIX, the spread does not have a stable long-term mean
- Structural breaks invalidate fixed-mean reversion models
- High persistence means deviations decay slowly
- A z-score of the spread LEVEL would correlate with the spread itself (VIF issue)

**Instead, focus on:**
- **VELOCITY** of spread changes (how fast it's moving)
- **REGIME** of spread dynamics (which state the curve is in)
- **CURVATURE** or volatility (shape/turbulence of the curve)

These capture the DYNAMICS without assuming a fixed reversion target.

---

### Question 6: Academic Evidence for Yield Curve Predicting Gold Returns

**Research Question:** Is there academic evidence that yield curve DYNAMICS (not levels) predict gold returns? Does rate of change of spread predict gold better than spread level? Do regime transitions predict gold?

**Findings:**

#### Yield Curve and Recession Prediction (Estrella & Mishkin 1998)

**Foundational Research:**
- Estrella and Mishkin published "Predicting U.S. Recessions: Financial Variables as Leading Indicators" in Review of Economics and Statistics (1998)
- Key finding: The yield curve spread (10Y - 3M) **significantly outperforms other financial indicators in predicting recessions 2-6 quarters ahead**
- Sources: [The Yield Curve as a Predictor of U.S. Recessions](https://www.newyorkfed.org/medialibrary/media/research/current_issues/ci2-7.pdf), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1001228)

**Recession-Gold Link:**
- Yield curve inversions precede recessions
- Recessions typically bullish for gold (safe-haven demand)
- **Implication:** Yield curve dynamics indirectly predict gold through recession channel

**Academic Consensus:**
- "There is a large literature documenting the negative relationship between measures of the slope of the yield curve and the probability of a subsequent recession" ([Predicting Recessions Using the Yield Curve](https://www.bostonfed.org/publications/current-policy-perspectives/2020/predicting-recessions-using-the-yield-curve.aspx))
- The yield curve has inverted before every U.S. recession since 1950

#### Direct Evidence: Yield Curve → Gold Returns

**Recent Research (2024-2025):**

**Real Yields and Gold:**
- "A 100-basis-point increase in 10-year real yields has historically led to a decline of 18% in the inflation-adjusted price of gold" ([US Real Rates Still Matter for Gold](https://www.ssga.com/library-content/assets/pdf/apac/gold/2025/en/us-real-rates-still-matter-for-gold.pdf))
- Gold and real yields are negatively correlated
- **Note:** This is LEVEL correlation, not dynamics

**Yield Curve Steepening and Gold:**
- "Lower front-end yields ease the opportunity cost of holding non-yielding assets" (gold)
- "If inflation stabilised further, the effect on rates would be more substantial" ([Steepening US yield curve and what it means for gold](https://www.home.saxo/content/articles/commodities/steepening-us-yield-curve-and-what-it-means-for-gold-28082025))
- Steepening curve (growth expectations) → mixed for gold (lower opportunity cost but inflation fears)
- Flattening curve (tightening expectations) → bearish for gold (rising front-end rates)

**Recent Decoupling (2024-2025):**
- Gold reached all-time highs in early 2025 despite rising real yields
- "This 'sticky' safe haven buying contributed to a run-up in gold prices, decoupling the link between prices and real yields for the time being" ([Gold Mid-Year Outlook 2025](https://www.gold.org/goldhub/research/gold-mid-year-outlook-2025))
- Geopolitical and central bank buying factors temporarily override yield dynamics

#### Evidence for DYNAMICS vs LEVELS

**Limited Direct Research:**
- Most research focuses on yield LEVELS (real yields, nominal yields) not curve DYNAMICS
- No specific papers found on "spread change velocity predicts gold returns"
- Academic gap in this area

**Theoretical Support:**
- Regime transitions (flattening → inversion) capture market repricing of recession risk
- Sudden flattening (panic tightening expectations) vs gradual flattening (orderly policy) have different gold implications
- Velocity signals urgency of market repositioning

**Historical Patterns (from current_task.json):**
- August 2019 inversion: Gold rallied $1400 → $1550 as recession fears rose (**transition mattered**)
- 2020-2021 steepening: Gold initially benefited then declined as recovery priced in
- 2022 flattening/inversion: Gold volatile but held value as recession risk priced in

**Key Insight:** "The REGIME TRANSITION is more informative than the current state. Markets price in the current state; submodel value comes from identifying the dynamics that precede gold moves."

#### Recommendation

**Evidence Level: MODERATE-TO-WEAK for direct causality**

1. **Strong indirect evidence:** Yield curve predicts recessions → recessions bullish for gold
2. **Strong level evidence:** Real yields negatively correlate with gold (but levels already in base features)
3. **Weak dynamics evidence:** Limited academic research on spread change velocity → gold
4. **Strong practitioner evidence:** Historical regime transitions (inversion, steepening) coincide with gold moves

**Design Implication:**
- The submodel is justified by **indirect evidence** (recession channel) and **historical patterns**
- Focus on capturing **regime transitions** (which the meta-model cannot easily infer from raw levels)
- Velocity and curvature provide **context** for how to interpret the current spread level
- Architect should validate with **in-sample correlation analysis** between regime features and next-day gold returns

---

### Question 7: Multi-Country Yield Curve Data

**Research Question:** Are German, UK, Japanese yield curves available daily from FRED? Would global yield curve slope add information? What is VIF risk?

**Findings:**

#### FRED Multi-Country Availability

**Germany (Bund Yields):**
- **Ticker:** IRLTLT01DEM156N (10-Year)
- **Frequency:** **Monthly** (NOT daily)
- **Source:** OECD via FRED
- **Coverage:** Available but insufficient frequency
- [Interest Rates: Long-Term Government Bond Yields: 10-Year: Germany](https://fred.stlouisfed.org/series/IRLTLT01DEM156N)

**United Kingdom (Gilt Yields):**
- **Ticker:** IRLTLT01GBM156N (10-Year)
- **Frequency:** **Monthly** (NOT daily)
- **Source:** OECD via FRED
- **Coverage:** Available but insufficient frequency
- [Interest Rates: Long-Term Government Bond Yields: 10-Year: UK](https://fred.stlouisfed.org/series/IRLTLT01GBM156N)

**Japan (JGB Yields):**
- **Ticker:** IRLTLT01JPM156N (10-Year)
- **Frequency:** **Monthly** (NOT daily)
- **Source:** OECD via FRED
- **Coverage:** Available but insufficient frequency
- [Interest Rates: Long-Term Government Bond Yields: 10-Year: Japan](https://fred.stlouisfed.org/series/IRLTLT01JPM156N)

#### Critical Finding: No Daily Multi-Country Data from FRED

**All G10 country yield data on FRED is MONTHLY, not daily.** This reproduces the exact problem that killed the real_rate submodel (monthly-to-daily frequency mismatch).

#### Alternative Sources (Not FRED)

Daily yield curve data for Germany, UK, Japan may be available from:
- Central bank websites (Bundesbank, Bank of England, Bank of Japan)
- Bloomberg/Reuters (paid services, not viable for this project)
- Trading Economics (requires API key verification)

**Constraint from CLAUDE.md:** "Do not execute processes requiring paid services or new API keys without explicit user approval"

#### VIF Risk of Global Composite

**If daily data were available:**
- Global yield curve slope = weighted average of (DGS10-DGS2, Bund 10-2, Gilt 10-2, JGB 10-2)
- **High VIF risk:** Global slope would correlate strongly with US slope (0.6-0.8 expected)
- Developed country yield curves move together due to global capital flows
- Marginal information gain likely small vs high VIF cost

#### Recommendation

**DO NOT pursue multi-country yield curve data:**

1. **Fatal flaw:** FRED data is monthly, not daily (reproduces real_rate failure mode)
2. **Alternative sources:** Require new API keys or paid services (violates constraints)
3. **VIF risk:** Even if available, global slope would likely correlate with US slope
4. **Information gain:** Marginal vs complexity cost

**Focus on US-only yield curve with rich feature engineering:**
- 2D HMM on [DGS10_chg, DGS2_chg] captures twist dynamics
- Curvature (using DGS5 or DGS30) adds belly/convexity dimension
- Velocity z-score captures rate of curve movement
- These three features from US data alone should provide sufficient information

**Precedent:** DXY, VIX, technical, cross_asset all succeeded with US-centric data (no multi-country expansion needed).

---

## Recommended Approach

Based on the research findings, the **optimal design** is:

### **Approach B (Modified): 2D HMM + Velocity + Curvature**

**Output Features (3 columns):**

1. **yc_regime_prob** (regime probability from 2D HMM)
   - Input: 2D vector [DGS10_change, DGS2_change] (daily first differences)
   - Model: 3-state Hidden Markov Model
   - States likely represent: steepening, neutral, flattening/inversion regimes
   - Output: Probability of being in the most informative state (e.g., P(flattening regime))
   - Expected autocorrelation: 0.85-0.95 (precedent: VIX regime)
   - Expected VIF: <5 (regime of changes orthogonal to yield levels)

2. **yc_spread_velocity_z** (z-score of spread change velocity)
   - Compute: `spread_change_5d = (DGS10 - DGS2)[t] - (DGS10 - DGS2)[t-5]`
   - Then: `yc_spread_velocity_z = zscore(spread_change_5d, window=60)`
   - Captures how FAST the curve is moving relative to recent history
   - Positive = rapid steepening, negative = rapid flattening
   - Expected autocorrelation: 0.80-0.90
   - Expected VIF: <3 (change velocity orthogonal to spread level)

3. **yc_curvature_z** (z-score of belly curvature changes)
   - Compute: `curvature_raw = DGS5 - 0.5*(DGS2 + DGS10)`
   - Then: `curvature_change = diff(curvature_raw)`
   - Then: `yc_curvature_z = zscore(curvature_change, window=60)`
   - Captures curve convexity/bowing dynamics
   - Orthogonal to slope (measures belly bulge, not tilt)
   - Expected autocorrelation: 0.75-0.85
   - Expected VIF: <4 (needs empirical verification by architect)

### Why This Design

**Follows Proven Pattern:**
- 2D HMM successful in technical submodel ([returns, GK_vol])
- 3 compact features successful in VIX, technical, cross_asset
- Z-scores of CHANGES (not levels) avoid autocorrelation issues

**Captures Three Orthogonal Dimensions:**
1. **Regime (categorical context):** Which state are curve dynamics in?
2. **Velocity (speed):** How fast is the curve shifting?
3. **Curvature (shape):** What is the convexity of the curve?

**Avoids VIF Pitfalls:**
- Operates on CHANGES and REGIMES, not levels
- No direct correlation with yield_spread base feature (spread level)
- No proxy for real_rate (DFII10 level)

**Daily Native Frequency:**
- All data (DGS10, DGS2, DGS5) available daily from FRED
- No interpolation artifacts

### Implementation Notes

**Data Requirements:**
- FRED tickers: DGS10, DGS2, DGS5
- Frequency: Daily (no interpolation)
- Date range: 2015-01-30 to 2025-02-12 (matches base_features)

**HMM Training:**
- Fit 3-state HMM on training set ONLY
- Input: 2D array of [DGS10_change, DGS2_change]
- Use `hmmlearn.GaussianHMM` with `n_components=3`
- Inference: `model.predict_proba()` for regime probabilities on val/test

**Z-Score Calculation:**
- Rolling window: 60 days (follows VIX success)
- Backward-looking only (no lookahead bias)
- Handle NaNs: First 60 days will be NaN, acceptable warmup period

**Optuna Hyperparameters:**
- HMM: `n_components` (2, 3, or 4 states)
- HMM: `covariance_type` ('full', 'tied', 'diag', 'spherical')
- Velocity: `change_window` (3, 5, 10 days)
- Velocity: `zscore_window` (30, 60, 90, 120 days)
- Curvature: `zscore_window` (30, 60, 90, 120 days)

---

## Data Acquisition Strategy

### Primary Data (Daily, FRED)

| Ticker | Description | Availability | Purpose |
|--------|-------------|--------------|---------|
| DGS10 | 10Y Treasury yield | 1962 - present | Curve dynamics, HMM input |
| DGS2 | 2Y Treasury yield | 1976 - present | Curve dynamics, HMM input |
| DGS5 | 5Y Treasury yield | 1962 - present | Belly curvature measure |

**Fetch Code Example:**
```python
import pandas as pd
from fredapi import Fred
import os

fred = Fred(api_key=os.environ['FRED_API_KEY'])

# Fetch data
dgs10 = fred.get_series('DGS10', observation_start='2015-01-30')
dgs2 = fred.get_series('DGS2', observation_start='2015-01-30')
dgs5 = fred.get_series('DGS5', observation_start='2015-01-30')

# Combine into DataFrame
df = pd.DataFrame({
    'DGS10': dgs10,
    'DGS2': dgs2,
    'DGS5': dgs5
})

# Compute derived features
df['spread'] = df['DGS10'] - df['DGS2']
df['curvature_raw'] = df['DGS5'] - 0.5 * (df['DGS2'] + df['DGS10'])
```

**Expected Samples:** ~2523 daily observations (matching base_features date range)

### Multi-Country Data: NOT RECOMMENDED

- FRED only provides monthly data for Germany, UK, Japan
- Monthly-to-daily interpolation creates artifacts (real_rate failure lesson)
- Alternative sources require new API keys (violates constraints)
- Marginal information gain vs VIF risk

---

## VIF and Autocorrelation Design Principles

### VIF Mitigation Strategy

**Critical Constraint:** Base features contain 3 yield curve columns (DGS10, DGS2, yield_spread) plus real_rate (DFII10, correlated ~0.7-0.9 with DGS10).

**Design Principles:**
1. **Operate on CHANGES, not LEVELS**
   - HMM fitted on [DGS10_change, DGS2_change], not [DGS10, DGS2]
   - Velocity z-score uses 5-day spread CHANGES
   - Curvature z-score uses daily CHANGES in curvature

2. **Use REGIMES, not raw proxies**
   - Regime probability is categorical state, orthogonal to continuous levels
   - VIX precedent: regime_prob vs vix_vix base feature (low correlation)

3. **Use Z-SCORES OF DERIVATIVES**
   - Z-score of change = normalized velocity
   - NOT z-score of level (would correlate with level itself)

**Expected VIF:**
- `yc_regime_prob` vs base features: <5
- `yc_spread_velocity_z` vs `yield_curve_yield_spread`: <3
- `yc_curvature_z` vs `yield_curve_yield_spread`: <4 (needs verification)

### Autocorrelation Mitigation Strategy

**Critical Constraint:** Spread LEVELS have near-unit-root autocorrelation (>0.99). Spread CHANGES have moderate autocorrelation (momentum effect).

**Design Principles:**
1. **NEVER use raw spread levels or spread level z-scores**
   - `zscore(spread)` would have autocorr >0.99 (fail Gate 1)

2. **Use z-scores of CHANGES**
   - `zscore(diff(spread, 5))` has autocorr ~0.85-0.90
   - Change removes unit-root component

3. **Use first differences of z-scores**
   - `diff(zscore(curvature))` reduces autocorrelation further
   - Cross_asset lesson: first diff of z-score solved autocorr issue

**Expected Autocorrelation:**
- `yc_regime_prob`: 0.85-0.95 (HMM regimes are persistent, accepted precedent)
- `yc_spread_velocity_z`: 0.80-0.90 (z-score of 5-day change)
- `yc_curvature_z`: 0.75-0.85 (z-score of daily curvature change)

All should pass Gate 1 threshold (<0.99).

---

## Lookback Window Recommendations

| Feature | Computation | Window | Rationale |
|---------|-------------|--------|-----------|
| **Spread velocity** | 5-day spread change | 60-day z-score | Follows VIX success (60d); 5d change reduces noise |
| **Curvature** | Daily curvature change | 60-day z-score | Consistent normalization; daily change captures dynamics |
| **HMM training** | Full training set | N/A | Fit once on train, infer on val/test; no rolling |

**Optuna Search Space:**
- Velocity change window: {3, 5, 10} days
- Z-score window: {30, 60, 90, 120} days
- HMM states: {2, 3, 4}

These ranges allow the model to discover optimal sensitivity to curve dynamics.

---

## Expected Performance

### Gate 1: Standalone Quality

**Expected:** PASS

- Overfit ratio: N/A (HMM is deterministic, no train/val split for HMM itself)
- No constant output: HMM produces varying regime probabilities
- Autocorrelation: 0.75-0.95 for all features (below 0.99 threshold)

**Risk:** Curvature feature may have higher autocorr if daily changes are too smooth. Mitigation: Optuna can select longer change windows if needed.

### Gate 2: Information Gain

**Expected:** MARGINAL (MI increase 5-15%)

- MI increase: Yield curve regime transitions should add context beyond raw levels
- VIF: <10 expected (all features operate on changes/regimes, orthogonal to levels)
- Rolling correlation stability: Regime_prob may marginally fail (0.15-0.21), accepted precedent from VIX/technical/cross_asset

**Risk:** If curvature correlates >0.7 with spread level, VIF may fail. Architect must measure empirically and switch to spread volatility if needed.

### Gate 3: Ablation

**Expected:** PASS on DA or MAE

- Direction accuracy: +0.5-1.0% (yield curve regime transitions historically precede gold moves)
- MAE: Moderate improvement expected (regime context helps meta-model weight spread signal)
- Sharpe: Lower confidence (recent decoupling of gold from yields may hurt)

**Hypothesis:** Identical spread levels have very different gold implications depending on dynamics. A spread of +50bp that is rapidly narrowing (flattening = tightening = bearish gold) differs fundamentally from +50bp that is widening (steepening = easing = bullish gold). The submodel provides this dynamic context.

### Success Probability Estimate

**70-80%** (moderate-to-high confidence)

**Strengths:**
- Daily native frequency (no interpolation artifacts)
- Proven HMM pattern
- Strong theoretical basis (yield curve predicts recessions → gold safe-haven)
- Operates on changes (avoids VIF trap)

**Risks:**
- Highest VIF-risk submodel (4 correlated base features already present)
- Recent gold-yield decoupling (2024-2025) may weaken signal
- Curvature VIF needs empirical verification
- Moderate academic evidence for direct gold prediction (mostly indirect via recession channel)

---

## Alternative Approaches (If Primary Fails)

### Fallback 1: 1D HMM + Volatility

If 2D HMM produces high VIF or noisy regimes:

- **yc_regime_prob:** 1D HMM on spread changes only (simpler, follows VIX)
- **yc_spread_velocity_z:** Same as primary
- **yc_spread_vol_z:** Rolling std of spread changes, z-scored (replaces curvature)

Trades curvature dimension for volatility dimension (captures turbulence instead of convexity).

### Fallback 2: Fully Deterministic

If HMM overfits:

- **yc_spread_velocity_z:** Z-score of 5-day spread change
- **yc_spread_accel:** Z-score of spread acceleration (second derivative)
- **yc_yield_vol_ratio:** Spread change normalized by yield volatility

No HMM, fully deterministic. Sacrifices regime detection for robustness.

### Fallback 3: Threshold + Deterministic

If HMM fails but regime concept is valuable:

- **yc_inversion_flag:** Binary indicator (spread < 0)
- **yc_spread_velocity_z:** Same as primary
- **yc_spread_vol_z:** Spread volatility

Simple threshold for inversion (critical regime) + velocity + volatility.

---

## Critical Notes for Architect

### Fact-Check Requirements

1. **Verify DGS5 availability:** Confirm no gaps in FRED:DGS5 from 2015-01-30 to 2025-02-12
2. **Measure curvature-spread correlation:** Compute `corr(curvature_raw, spread)` on full data
   - If corr > 0.7, switch to spread volatility instead of curvature
3. **Measure DGS10-DFII10 correlation:** Quantify overlap between yield curve and real_rate base features
4. **Verify autocorrelation:** Compute lag-1 autocorr for all three features on train set
   - If any > 0.99, adjust feature design (longer change windows, first differences)

### Design Decisions for Architect

1. **HMM dimensionality:** Empirically test 1D vs 2D HMM on in-sample correlation with gold returns
   - If 2D adds <2% correlation vs 1D, use 1D for simplicity
2. **Curvature vs volatility:** Based on VIF measurement, choose third feature:
   - Curvature (if curvature-spread corr < 0.7)
   - Spread volatility (if curvature-spread corr >= 0.7)
3. **Regime interpretation:** Analyze HMM states on training data:
   - Which state corresponds to steepening? Flattening? Inversion?
   - Use the state with strongest gold return correlation as output probability

### HP Search Space

```python
optuna_space = {
    'hmm_n_components': [2, 3, 4],
    'hmm_covariance_type': ['full', 'tied', 'diag'],
    'velocity_change_window': [3, 5, 10],  # days
    'velocity_zscore_window': [30, 60, 90, 120],  # days
    'curvature_zscore_window': [30, 60, 90, 120],  # days
}
```

Optuna should optimize for highest correlation with next-day gold returns on validation set.

---

## Summary and Recommendations

### Key Findings

1. ✅ **Daily data available** for DGS10, DGS2, DGS5 from FRED (no interpolation needed)
2. ✅ **2D HMM proven approach** for regime detection in financial markets
3. ⚠️ **Multi-country data NOT viable** (monthly frequency only, reproduces real_rate failure)
4. ✅ **Curvature measure orthogonal to slope** conceptually (needs empirical VIF verification)
5. ⚠️ **Moderate academic evidence** for direct gold prediction (strong indirect via recession channel)
6. ✅ **Lookback windows:** 5-day change + 60-day z-score follows VIX success pattern
7. ⚠️ **Highest VIF risk** of any submodel (4 correlated base features already present)

### Recommended Methodology (Priority Order)

1. **Primary:** 2D HMM [DGS10_chg, DGS2_chg] + spread velocity z-score + curvature z-score
2. **Fallback 1:** 1D HMM spread_chg + spread velocity z-score + spread volatility z-score
3. **Fallback 2:** Fully deterministic (velocity + acceleration + vol ratio)

### Data Sources

| Data | Source | Ticker | Frequency | Availability |
|------|--------|--------|-----------|--------------|
| 10Y yield | FRED | DGS10 | Daily | 1962 - present |
| 2Y yield | FRED | DGS2 | Daily | 1976 - present |
| 5Y yield | FRED | DGS5 | Daily | 1962 - present |

**Fetch code:** See Data Acquisition Strategy section above.

### Critical Constraints

- ALL features must operate on CHANGES or REGIMES (never levels) to avoid VIF with base features
- Autocorrelation must be <0.99 (use z-scores of changes, not z-scores of levels)
- Multi-country data NOT viable (monthly frequency fatal flaw)
- Curvature VIF must be empirically verified by architect before finalizing design

### Success Criteria Alignment

- **Gate 1:** Expected PASS (autocorr 0.75-0.95)
- **Gate 2:** Expected MARGINAL (MI +5-15%, VIF <10 if curvature VIF verified)
- **Gate 3:** Expected PASS on DA or MAE (regime context amplifies spread signal)

**Overall Success Probability:** 70-80%

---

## Sources

### Data Availability
- [Market Yield on U.S. Treasury Securities at 30-Year Constant Maturity (DGS30)](https://fred.stlouisfed.org/series/DGS30)
- [Market Yield on U.S. Treasury Securities at 5-Year Constant Maturity (DGS5)](https://fred.stlouisfed.org/series/DGS5)
- [Interest Rates: Long-Term Government Bond Yields: 10-Year: Germany](https://fred.stlouisfed.org/series/IRLTLT01DEM156N)
- [Interest Rates: Long-Term Government Bond Yields: 10-Year: UK](https://fred.stlouisfed.org/series/IRLTLT01GBM156N)
- [Interest Rates: Long-Term Government Bond Yields: 10-Year: Japan](https://fred.stlouisfed.org/series/IRLTLT01JPM156N)

### HMM and Regime Detection
- [Market Regime Detection Using Hidden Markov Models](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Regime-based portfolio optimisation: A Hidden Markov Model approach for fixed income portfolios](https://www.esm.europa.eu/publications/regime-based-portfolio-optimisation-hidden-markov-model-approach-fixed-income)
- [Hidden Markov Models for Regime Detection in Diverse Financial Data](https://thepythonlab.medium.com/hidden-markov-models-for-regime-detection-in-diverse-financial-data-42cf19cd5d34)

### Yield Curve and Gold
- [US Real Rates Still Matter for Gold](https://www.ssga.com/library-content/assets/pdf/apac/gold/2025/en/us-real-rates-still-matter-for-gold.pdf)
- [Steepening US yield curve and what it means for gold](https://www.home.saxo/content/articles/commodities/steepening-us-yield-curve-and-what-it-means-for-gold-28082025)
- [Gold Mid-Year Outlook 2025](https://www.gold.org/goldhub/research/gold-mid-year-outlook-2025)

### Yield Curve Dynamics
- [BrokerTec RV Curve and Butterflies](https://www.cmegroup.com/markets/brokertec/brokertec-rv-curve-and-butterflies.html)
- [Yield Curve Shapes and Movements](https://financialanalystguide.com/cfa-level-1/volume-6-fixed-income/chapter-7-the-term-structure-of-interest-rates/yield-curve-shapes-and-movements-flattening-steepening-butterfly/)
- [The Treasury Yield Curve as a Cointegrated System](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/abs/treasury-yield-curve-as-a-cointegrated-system/E5075FCBDB4870EB7B5859012B17C83F)
- [Stationarity and the Term Structure of Interest Rates](https://ideas.repec.org/p/fip/feddwp/0811.html)
- [Yield curve momentum: Implications for theory and practice](https://cepr.org/voxeu/columns/yield-curve-momentum-implications-theory-and-practice)

### Estrella & Mishkin Research
- [The Yield Curve as a Predictor of U.S. Recessions](https://www.newyorkfed.org/medialibrary/media/research/current_issues/ci2-7.pdf)
- [The Yield Curve as a Predictor of U.S. Recessions (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1001228)
- [Predicting Recessions Using the Yield Curve](https://www.bostonfed.org/publications/current-policy-perspectives/2020/predicting-recessions-using-the-yield-curve.aspx)

### VIX and Z-Score Parameters
- [Z-Score Normalized VIX Strategy](https://www.tradingview.com/script/6ZkpMLDe-Z-Score-Normalized-VIX-Strategy/)
- [Z-Score Normalized Volatility Indices](https://www.tradingview.com/script/xJX1siYz-Z-Score-Normalized-Volatility-Indices/)

---

**End of Research Report**
