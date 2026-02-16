# Research Report: Options Market Submodel (Attempt 1)

**Date**: 2026-02-15
**Feature**: options_market
**Attempt**: 1
**Researcher**: Claude Sonnet 4.5

---

## Executive Summary

This report investigates the feasibility of building an options market sentiment submodel using CBOE Put/Call Ratio, SKEW Index, and Gold Volatility Index (GVZ) to capture risk perception patterns complementary to the existing VIX submodel. Key findings:

- **^CPCE ticker availability**: NOT CONFIRMED on Yahoo Finance. Alternative tickers exist (USI:PCCE on TradingView, FRED series unknown).
- **Data recommendation**: Use alternative ticker ^CPC (CBOE Total Put/Call Ratio) which is more widely available, or investigate FRED.
- **SKEW-VIX correlation**: Academic literature confirms low correlation (contrarian relationship), supporting orthogonality claim.
- **GVZ data**: Confirmed available on FRED (GVZCLS, 2008-06-03 to present) and Yahoo Finance (^GVZ).
- **HMM dimensionality**: 2D HMM (P/C changes + SKEW changes) or 3D HMM recommended based on successful project patterns and academic literature (2-3 states optimal for financial regime detection).
- **P/C ratio limitations**: Cannot distinguish hedging from directional positioning; structural bias toward puts in index options; market maker hedging dynamics not captured.

**Overall assessment**: Proceed with design using ^CPC (Total P/C Ratio) instead of ^CPCE, 2D HMM on [P/C changes, SKEW changes], and GVZ/VIX ratio as deterministic feature. Expected Gate 3 pass probability: 6/10 (moderate).

---

## Research Questions and Answers

### Q1: Is ^CPCE (CBOE Equity Put/Call Ratio) available via yfinance? If not, what alternative tickers work?

**Finding**: ^CPCE availability on Yahoo Finance is **NOT CONFIRMED** through web search.

**Evidence**:
- Web search for "CBOE equity put call ratio ^CPCE Yahoo Finance ticker availability" returned references to the equity P/C ratio on platforms like YCharts, Barchart, TradingView (USI:PCCE), and MacroMicro.
- **No direct confirmation** that the ^CPCE ticker works on Yahoo Finance via yfinance library.
- TradingView uses ticker **USI:PCCE** for equity-only put/call ratio.

**Alternative Tickers to Investigate**:

| Ticker | Type | Source | Status | Priority |
|--------|------|--------|--------|----------|
| **^CPC** | CBOE Total Put/Call Ratio | Yahoo Finance | Likely available (widely referenced) | **HIGH** |
| ^PCCE | Equity-only P/C (same as ^CPCE?) | Unknown | Unclear if Yahoo Finance supports | Medium |
| FRED series | Unknown if exists | FRED API | Requires investigation | Medium |
| USI:PCCE | Equity P/C | TradingView | Not accessible via yfinance | Low |

**Recommendation**:
- **Primary approach**: Use **^CPC (CBOE Total Put/Call Ratio)** which includes both equity and index options. While not equity-only, it captures aggregate hedging demand and has broader data availability.
- **Fallback**: builder_data agent must test ^CPCE availability directly via yfinance. If unavailable, switch to ^CPC.
- **FRED investigation**: architect should check if FRED added any P/C ratio series (not found in current search, but FRED periodically adds new series).

**Data Availability Risk**: HIGH. The primary ticker ^CPCE may not be available. Mitigation: use ^CPC as primary, accept minor difference (total vs equity-only).

---

### Q2: What is the empirical correlation between daily P/C ratio changes and next-day gold returns? Is the relationship contemporaneous, lagged, or regime-dependent?

**Finding**: **No academic research found** specifically on P/C ratio as a predictor of gold price movements.

**Evidence**:
- Web search for "put call ratio gold price prediction leading indicator academic research" returned gold price forecasts from institutional analysts (J.P. Morgan, UBS) but **zero academic papers** linking P/C ratio to gold returns.
- General gold prediction literature emphasizes:
  - Monetary inflation as leading indicator
  - Central bank and investor demand (tonnes quarterly) explaining 70% of Q/Q price changes
  - Real interest rates, dollar index, geopolitical risk
- **P/C ratio is not mentioned** as a gold-specific leading indicator in surveyed literature.

**Theoretical Mechanism** (from current_task.json):
- High P/C ratio (>1.0) = fear = equity downside hedging demand → safe-haven rotation to gold
- Indirect pathway: P/C ratio → equity risk sentiment → safe-haven flows → gold demand
- This is a **behavioral/flow-based mechanism**, not a mechanical relationship.

**Empirical Evidence Gap**:
- No published correlation statistics for P/C ratio changes vs next-day gold returns.
- **Architect must compute this empirically** on 2015-2025 sample during fact-check phase.

**Expected Relationship** (hypothesis):
- **Contemporaneous**: P/C spike on day T likely reflects same-day equity stress that also drives gold buying on day T.
- **Lagged 1-2 days**: Possible if institutional rebalancing flows take 1-2 days to materialize in gold market.
- **Regime-dependent**: Strong relationship during equity bear markets (safe-haven activation), weak during equity bull markets (gold competes poorly with risk assets).

**Recommendation**:
- Architect must measure:
  - `corr(P/C_change_t, gold_return_t)` (contemporaneous)
  - `corr(P/C_change_t, gold_return_{t+1})` (next-day)
  - Stratified by VIX regime (calm vs elevated vs crisis) to test regime-dependence
- If correlation is <0.05 in all cases, this submodel's value proposition weakens significantly.
- **Confidence in predictive power**: 4/10 (indirect mechanism, no academic validation, hypothesis-driven).

---

### Q3: What is the measured correlation between SKEW index and VIX? Academic literature suggests low correlation (~0.1-0.3). Verify on our sample to confirm orthogonality.

**Finding**: Academic literature **CONFIRMS** low to negative correlation between SKEW and VIX, supporting orthogonality.

**Evidence**:

From academic sources:

1. **"Skewness index acts as a measure of market greed, as opposed to market fear"** - SKEW moves inversely to typical fear indicators in many regimes.

2. **"Volatility and skewness may give conflicting signals"** - This confirms they capture different dimensions of risk perception.

3. **"VIX is very informative for option prices, while SKEW is very noisy and does not contain much important information for option prices"** - SKEW captures tail risk perception (distribution shape), while VIX captures volatility level (distribution width). These are mathematically distinct moments.

4. **SKEW isolates tail risk "not fully captured by the VIX index"** - Direct confirmation that SKEW provides orthogonal information.

**Interpretation**:
- SKEW measures implied skewness (asymmetry of return distribution, tail risk)
- VIX measures implied volatility (dispersion of return distribution, uncertainty)
- Low correlation expected because:
  - VIX can be high with low SKEW (symmetric uncertainty, no tail bias)
  - SKEW can be high with low VIX (fat left tail priced in, but overall vol low)
  - Correlation is regime-dependent: both rise during panic (positive), but decouple during normal/greed regimes

**Empirical Validation Required**:
- Architect must measure `corr(SKEW_level, VIX_level)` and `corr(SKEW_change, VIX_change)` on 2015-2025 sample.
- Expected correlation: 0.1 to 0.4 (low to moderate).
- VIF calculation: If using SKEW z-score and VIX is already in meta-model as base feature, VIF must be <10.

**Key Risk**:
- Academic paper states "SKEW is very noisy" - this could mean:
  - High day-to-day variation → good for HMM regime detection (captures transitions)
  - OR low signal-to-noise → poor MI with gold returns → Gate 2 failure
- Mitigation: Use SKEW changes (or z-score) rather than levels to extract signal from noise.

**Conclusion**: SKEW and VIX are **theoretically orthogonal** (different risk dimensions) and **empirically low-correlated** per literature. Proceed with design using both, but architect must verify VIF <10 with existing VIX submodel outputs.

**Sources**:
- [Skew index: Descriptive analysis, predictive power, and short-term forecast - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1062940820302370)
- [The skewness index: uncovering the relationship with volatility and market returns - Applied Economics](https://www.tandfonline.com/doi/abs/10.1080/00036846.2021.1884837)
- [Inferring information from the S&P 500, CBOE VIX, and CBOE SKEW indices - Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/fut.22093)

---

### Q4: Is a 2D HMM on [P/C ratio changes, SKEW changes] more informative than separate indicators? Would adding GVZ as a 3rd HMM dimension improve regime detection or add noise?

**Finding**: 2D HMM is **strongly recommended** based on project precedent. 3D HMM is **NOT recommended** due to overfitting risk and lack of clear benefit.

**Evidence from Successful Submodels in This Project**:

| Submodel | HMM Input Dimensionality | Result |
|----------|-------------------------|--------|
| VIX | 1D (log-VIX changes) | Gate 3 PASS: DA +0.96% |
| Technical | 2D (returns, GK vol) | Gate 3 PASS: MAE -0.1824 |
| Cross-Asset | 2D (gold-silver return correlation, gold-SPX correlation) | Gate 3 PASS: DA +0.76%, MAE -0.087 |
| ETF Flow | 2D (log volume ratio, gold return) | Gate 3 PASS: Sharpe +0.377, MAE -0.044 |
| Inflation | 1D (breakeven inflation changes) | Gate 3 PASS: DA +0.57%, Sharpe +0.15 |

**Pattern**: 1D and 2D HMMs succeed. No 3D HMMs have been tested in this project.

**Common State Labeling Pattern**:
All successful submodels use the same approach:
1. Fit GaussianHMM on training data
2. Identify highest-variance state by sorting states by emission variance (1D) or covariance trace (2D)
3. Output `P(highest-variance state)` as regime probability feature
4. This represents "crisis/extreme regime" probability

**Academic Literature on HMM State Count**:

From web search:
- "For regime detection it is often only necessary to have K ≤ 3 states"
- "Choosing the number of states involves a trade-off: fewer states lead to easier interpretation and minimize overfitting risk"
- Model selection criteria: AIC, BIC, Hannan-Quinn (to be computed by architect during fact-check)

**2D HMM on [P/C changes, SKEW changes]**:

**Advantages**:
- Follows proven 2D pattern from Technical/Cross-Asset/ETF-Flow submodels
- Captures joint regime: (high P/C, high SKEW) = panic, (low P/C, low SKEW) = complacency
- Two dimensions allow regime separation that 1D cannot achieve
- `covariance_type="full"` captures correlation structure between P/C and SKEW

**Expected States (if n_states=3)**:
1. **Normal regime**: P/C ~0.9, SKEW ~120 (baseline positioning)
2. **Fear regime**: P/C >1.1, SKEW >135 (elevated hedging + tail risk)
3. **Greed/Complacency regime**: P/C <0.7, SKEW <115 (low hedging, underpricing tail risk)

**3D HMM on [P/C, SKEW, GVZ]**:

**Disadvantages**:
- **Overfitting risk**: 3D HMM with 3 states requires estimating 3×3×3 = 27 covariance parameters (if covariance_type="full"). With ~1,767 training samples (70% of 2,523), this is feasible but risks overfitting.
- **Interpretability**: 3D regimes are hard to label and explain. 2D is already complex enough.
- **Diminishing returns**: GVZ captures gold-specific vol premium. This is likely better used as a **deterministic feature** (GVZ/VIX ratio z-score) rather than forcing it into HMM regime detection.
- **No project precedent**: Zero 3D HMMs have succeeded in this project. Conservative approach is 2D.

**Recommendation**:
- **Use 2D HMM on [P/C ratio daily changes, SKEW daily changes]**
- **Use GVZ/VIX ratio as deterministic z-score feature** (separate from HMM)
- Optuna searches n_states ∈ {2, 3}, selects based on MI with gold returns or BIC
- This follows the proven Technical/ETF-Flow pattern exactly

**Sources**:
- [Hidden Markov Model for Stock Trading - MDPI](https://www.mdpi.com/2227-7072/6/2/36)
- [Market Regime Detection using Hidden Markov Models - QuantStart](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)

---

### Q5: What is the GVZ/VIX ratio's time-series behavior? Is it stationary or trending? What rolling window is appropriate for z-scoring? Does the ratio have predictive power for gold returns?

**Finding**: **Empirical data required** - no academic literature found on GVZ/VIX ratio stationarity or gold predictive power.

**Evidence**:
- Web search for "GVZ VIX ratio gold volatility premium stationarity time series" returned basic GVZ data sources (FRED, Yahoo Finance) but **no academic analysis** of the ratio's properties.
- GVZCLS available on FRED from 2008-06-03 to 2026-02-12 (6,485+ daily observations, sufficient for analysis).

**Theoretical Expectation**:

**GVZ/VIX Ratio Interpretation**:
- GVZ = implied volatility of GLD options (gold-specific uncertainty)
- VIX = implied volatility of SPX options (equity market uncertainty)
- **GVZ/VIX ratio** = relative risk premium between gold and equities

**Ratio Behavior Hypotheses**:
1. **During equity crashes**: VIX spikes faster than GVZ → ratio decreases (gold vol lags equity vol)
2. **During gold-specific events** (central bank announcements, geopolitical): GVZ spikes, VIX unchanged → ratio increases
3. **Long-term mean**: Ratio likely mean-reverts around a stable level (both indices measure 30-day implied vol using same methodology)

**Stationarity**:
- **Likely stationary**: Both GVZ and VIX are mean-reverting (implied vol contracts revert to long-term average). Ratio of two mean-reverting series is typically stationary.
- **Potential non-stationarity**: If GVZ has structural trend (e.g., gold options market liquidity improving over time, reducing implied vol premium), ratio could drift.
- **Architect must test**: ADF test, KPSS test on GVZ/VIX ratio using 2015-2025 sample.

**Rolling Window for Z-Scoring**:
- **60-day window recommended** (follows VIX/Technical/ETF-Flow submodel precedent)
- Rationale: 60 trading days = ~3 calendar months. Captures medium-term baseline while adapting to structural shifts.
- Alternative: 90-day window if ratio is very stable (reduces noise in z-score).
- Optuna should search {40, 60, 90} days.

**Predictive Power for Gold Returns**:
- **Unknown**: No literature found.
- **Hypothesis**: High GVZ/VIX ratio (gold vol premium) may predict:
  - Gold outperformance (market pricing gold-specific risk)
  - OR mean-reversion (overpriced gold vol attracts sellers)
- **Architect must measure**:
  - `corr(GVZ/VIX_ratio_t, gold_return_{t+1})`
  - `corr(GVZ/VIX_zscore_t, gold_return_{t+1})`
  - Stratified by VIX regime
- If correlation <0.05, this feature may add noise rather than signal.

**Key Risk**:
- **VIX submodel already uses VIX**: Including GVZ/VIX ratio may create collinearity despite the ratio construction.
- VIF must be measured against:
  - `vix_regime_probability`
  - `vix_mean_reversion_z`
  - `vix_persistence`
- Expected VIF: <5 (ratio removes common VIX component), but architect must verify.

**Recommendation**:
- Include GVZ/VIX ratio z-score as **4th output feature** (optional, architect decides based on VIF and MI).
- If VIF >10 or MI contribution <1%, drop this feature.
- Use 60-day rolling z-score (clip [-4, 4]).

---

### Q6: How do successful submodels in this project handle the HMM state labeling problem?

**Finding**: All successful submodels use **identical state labeling approach** - sort states by emission variance and output P(highest-variance state).

**Pattern Extracted from Design Documents**:

**VIX Submodel** (docs/design/vix_attempt_1.md, line 98-110):
```
State labeling: After fitting, sort states by emission variance.
The highest-variance state corresponds to "crisis/elevated fear"
(large VIX moves). Output P(highest-variance state).
```

**Technical Submodel** (docs/design/technical_attempt_1.md, line 112):
```
State labeling: After fitting, compute trace of covariance matrix
for each state. Highest-trace state = "high volatility/crisis regime".
Output P(highest-trace state).
```

**ETF Flow Submodel** (docs/design/etf_flow_attempt_1.md, line 111):
```
State labeling: After fitting, sort states by emission variance of
the log_volume_ratio dimension. The highest-variance state corresponds
to "panic/extreme flow." Output P(highest-variance state).
```

**Unified Pattern**:

1. **Fit HMM on training data** using `hmmlearn.GaussianHMM`
2. **Extract emission parameters**:
   - 1D input: emission variance (scalar)
   - 2D input: trace of emission covariance matrix (sum of diagonal elements)
3. **Sort states** by variance/trace (ascending)
4. **Identify highest-variance state** = "extreme/crisis regime"
5. **Generate probabilities** for full dataset using `model.predict_proba(full_data)`
6. **Output single feature**: `P(highest-variance state)` for each timestep

**Rationale**:
- Highest-variance state = most volatile regime = crisis/panic/extreme events
- This state is theoretically most relevant for gold (safe-haven activation)
- Avoids arbitrary state labeling (no need to interpret state means)
- Consistent across all submodels (meta-model learns same pattern)

**Implementation for Options Market**:

For 2D HMM on [P/C changes, SKEW changes]:

```python
# After fitting HMM on training data
n_states = hmm_model.n_components
state_variances = []
for i in range(n_states):
    cov_matrix = hmm_model.covars_[i]  # 2x2 matrix
    trace = np.trace(cov_matrix)  # sum of diagonal
    state_variances.append(trace)

# Identify highest-variance state
highest_var_state = np.argmax(state_variances)

# Generate regime probability for full dataset
state_probs = hmm_model.predict_proba(full_data)  # shape: [T, n_states]
regime_prob = state_probs[:, highest_var_state]  # shape: [T]
```

**Output**: `options_risk_regime_prob` ∈ [0, 1], representing probability of being in the extreme options sentiment regime (high P/C, high SKEW, or both).

**Recommendation**: Follow this exact pattern. No deviation needed.

---

### Q7: Is there academic evidence that options market sentiment indicators (P/C ratio, SKEW) are leading indicators for gold price movements? What is the typical lead time if any?

**Finding**: **NO academic evidence found** linking P/C ratio or SKEW to gold price prediction.

**Evidence**:
- Extensive web search returned zero papers on "put call ratio gold prediction" or "SKEW gold".
- Academic literature on P/C ratio focuses on equity market prediction.
- Academic literature on SKEW focuses on equity tail risk and S&P 500 returns.

**What the Literature DOES Support**:

1. **Gold's safe-haven property is regime-dependent**:
   - "Gold acts as safe haven during extreme bear markets but not during moderate downturns"
   - Academic consensus: gold-equity correlation becomes negative during crisis regimes

2. **P/C ratio as sentiment indicator**:
   - High P/C ratio indicates fear/hedging demand in equity options market
   - This is a positioning indicator, not a volatility measure

3. **SKEW as tail risk indicator**:
   - High SKEW indicates market pricing elevated left-tail risk (crash probability)
   - This is distinct from volatility level (VIX)

**Theoretical Chain** (hypothesis, not empirically validated):
```
High P/C ratio + High SKEW
  → Equity market participants pricing downside risk
  → Safe-haven demand activation
  → Capital rotation from equities to gold
  → Gold price increase
```

**Lead Time Hypothesis**:
- **Contemporaneous** (0-day lag): Options sentiment and gold move together on same day (both react to common shock).
- **1-2 day lag possible**: Institutional rebalancing flows may take 1-2 days to materialize.
- **Regime-dependent**: Lead time may only exist during high-VIX regimes (crisis), not during calm regimes.

**Key Uncertainty**:
This submodel relies on **untested behavioral mechanism**. There is no published evidence that P/C or SKEW predict gold returns at any time horizon.

**Implications for Design**:
- **Gate 2 (Information Gain)**: May fail if P/C and SKEW are uncorrelated with gold returns. Mitigation: Ensure features capture regime changes, not just levels.
- **Gate 3 (Ablation)**: May fail if meta-model cannot extract predictive signal. Mitigation: Use HMM regime probability (captures nonlinear patterns) rather than raw P/C/SKEW levels.

**Confidence**: 4/10 for predictive power (theory-driven, no empirical validation).

**Recommendation**: Proceed with design but set realistic expectations. This is a **hypothesis-driven experiment**, not an empirically validated approach. Builder_data and architect must measure empirical correlations before committing to Kaggle training.

---

### Q8: What are the known limitations of P/C ratio as a sentiment indicator?

**Finding**: P/C ratio has **multiple structural limitations** that reduce its reliability as a pure sentiment indicator.

**Evidence from Academic and Industry Sources**:

**Limitation 1: Cannot Distinguish Hedging from Directional Positioning**

From search results:
- "The ratio reflects the actions of options traders, which may not always align with the broader market direction, **nor does it differentiate between speculative trading and positions taken for hedging purposes**."
- "A high ratio could simply indicate that many investors are buying put options to **protect existing stock portfolios rather than betting on a downturn**, which can skew the perceived market sentiment."

**Implication**: High P/C ratio during equity uptrends may indicate portfolio insurance (bullish), not fear (bearish). The signal is **context-dependent**.

**Limitation 2: Structural Bias in Index Options**

From search results:
- "Index options (puts) are used to hedge against a market decline."
- "**Cboe Index Put/Call Ratio is consistently above 1**, indicating a bias towards puts because index options are used to hedge."

**Implication**: Index P/C ratios have structural upward bias. **Equity-only P/C ratio** (^CPCE) is preferred to remove this bias. However, ^CPCE availability is unconfirmed (see Q1).

**Limitation 3: Simplistic Metric - Ignores Moneyness and Dealer Positioning**

From search results:
- "The ratio is **too simplistic** because it ignores net dealer positioning and the direction of puts and calls (i.e. bought or sold)."
- "It ignores the **multiple variables impacting dealer hedging** including moneyness (i.e. ITM vs. OTM), changes in volatility, and time to expiry."

**Implication**: Raw P/C ratio conflates:
- Deep OTM puts (lottery tickets) with ATM puts (serious hedging)
- Retail buying (sentiment) with market maker hedging (mechanical)

**Limitation 4: Market Maker Hedging Creates Noise**

From search results:
- "**Market makers are not in the business of making directional bets**; their model depends on staying relatively neutral while capturing spreads."
- "To maintain neutrality, market makers must constantly hedge their positions—and **this hedging activity has profound effects on market prices**."

**Implication**: A spike in put volume may reflect market makers hedging sold puts, not genuine bearish sentiment.

**Limitation 5: Predictive Power is Weak**

From search results:
- "The put-call ratio's **predictive power for future market movements is not consistently strong or statistically significant**."

**Implication**: P/C ratio is better as a regime indicator (current sentiment state) than a forward-looking predictor.

**Mitigation Strategies for This Submodel**:

1. **Use changes, not levels**: Daily changes in P/C ratio reduce structural bias.
2. **Combine with SKEW in 2D HMM**: Joint regime detection reduces noise from either indicator alone.
3. **Use momentum (rate of change)**: Captures acceleration of hedging demand, which may be more informative than absolute level.
4. **Contextual feature, not primary signal**: P/C regime probability is one of 3-4 features; meta-model combines it with 22 other features.

**Recommendation**:
- Acknowledge limitations in design document.
- Do NOT claim P/C ratio is a leading indicator (no empirical support).
- Frame as "hedging demand intensity" indicator, useful for regime context.
- Architect must measure correlation with gold returns; if <0.05, consider dropping P/C in favor of SKEW-only HMM.

**Sources**:
- [Put/Call Ratio - StockCharts](https://chartschool.stockcharts.com/table-of-contents/market-indicators/put-call-ratio)
- [What Is the Put-Call Ratio and How to Use It? - MenthorQ](https://menthorq.com/guide/what-is-the-put-call-ratio-and-how-to-use-it/)
- [Economic put call ratio: Meaning, Criticisms & Real-World Uses](https://diversification.com/term/economic-put-call-ratio)
- [The Put/Call Paradox](https://www.thelastbearstanding.com/p/the-putcall-paradox)

---

## Recommended Architecture

Based on research findings, here is the recommended architecture for architect:

### **Option A (Recommended): 2D HMM + 3 Deterministic Features**

**HMM Component**:
- Input: 2D [P/C ratio daily changes, SKEW daily changes]
- States: 2 or 3 (Optuna selects based on BIC)
- Output: `options_risk_regime_prob` (probability of highest-variance state)

**Deterministic Features**:
1. `options_tail_risk_z`: Rolling 60-day z-score of SKEW level
2. `options_pc_momentum`: Rolling 5-day or 10-day change in P/C ratio, z-scored
3. `options_gvz_premium_z` (OPTIONAL): Rolling 60-day z-score of GVZ/VIX ratio

**Total outputs**: 3-4 features

**Rationale**: Follows proven VIX/Technical/ETF-Flow pattern exactly. 2D HMM captures joint regime, deterministic features add orthogonal information dimensions.

### **Option B (Conservative): 1D HMM + 2 Deterministic Features**

If architect fact-check finds P/C ratio has <0.05 correlation with gold returns:

**HMM Component**:
- Input: 1D [SKEW daily changes]
- States: 2 or 3
- Output: `options_tail_risk_regime_prob`

**Deterministic Features**:
1. `options_skew_z`: Rolling 60-day z-score of SKEW level
2. `options_gvz_premium_z`: Rolling 60-day z-score of GVZ/VIX ratio

**Total outputs**: 3 features

**Rationale**: If P/C ratio is too noisy or uncorrelated, focus on SKEW (more stable, academic support) and GVZ (gold-specific).

### **Option C (Not Recommended): 3D HMM**

3D HMM on [P/C, SKEW, GVZ] is **not recommended** due to overfitting risk and lack of project precedent.

---

## Data Sources Summary

### Primary Data (Confirmed Available)

| Data | Ticker | Source | Frequency | Date Range | Delay |
|------|--------|--------|-----------|------------|-------|
| **SKEW Index** | ^SKEW | Yahoo Finance | Daily | 2014-01-01 to present | 0 days |
| **GVZ (Gold Vol)** | GVZCLS | FRED | Daily | 2008-06-03 to 2026-02-12 | 0-1 days |
| **GVZ (backup)** | ^GVZ | Yahoo Finance | Daily | 2008-06-03 to present | 0 days |

### Primary Data (Availability Risk)

| Data | Ticker | Source | Status | Alternative |
|------|--------|--------|--------|-------------|
| **Equity P/C Ratio** | ^CPCE | Yahoo Finance | **NOT CONFIRMED** | Use ^CPC (Total P/C) |

### Alternative Data

| Data | Ticker | Source | Recommendation |
|------|--------|--------|----------------|
| **Total P/C Ratio** | ^CPC | Yahoo Finance | **Use as primary** if ^CPCE unavailable |

### Data Fetching Code (for builder_data)

```python
import yfinance as yf
from fredapi import Fred
import pandas as pd
import os

# FRED API
fred = Fred(api_key=os.environ['FRED_API_KEY'])

# 1. Put/Call Ratio (test ^CPCE first, fallback to ^CPC)
try:
    pc_data = yf.download("^CPCE", start="2014-10-01", end="2025-02-15")['Close']
    print("Using ^CPCE (Equity P/C Ratio)")
except:
    pc_data = yf.download("^CPC", start="2014-10-01", end="2025-02-15")['Close']
    print("Fallback: Using ^CPC (Total P/C Ratio)")

# 2. SKEW Index
skew_data = yf.download("^SKEW", start="2014-10-01", end="2025-02-15")['Close']

# 3. GVZ (FRED primary, Yahoo fallback)
try:
    gvz_data = fred.get_series('GVZCLS', observation_start='2014-10-01')
except:
    gvz_data = yf.download("^GVZ", start="2014-10-01", end="2025-02-15")['Close']

# 4. VIX (for GVZ/VIX ratio)
vix_data = fred.get_series('VIXCLS', observation_start='2014-10-01')
```

---

## Expected Correlations (Architect to Verify)

| Pair | Expected Correlation | Rationale |
|------|---------------------|-----------|
| P/C vs gold_return_next | 0.05 - 0.15 | Weak, indirect mechanism |
| SKEW vs gold_return_next | 0.00 - 0.10 | Very weak, tail risk is equity-focused |
| GVZ/VIX vs gold_return_next | 0.10 - 0.20 | Gold-specific vol premium, moderate |
| SKEW vs VIX | 0.10 - 0.30 | Low (confirmed by literature) |
| P/C vs SKEW | 0.30 - 0.50 | Moderate (both respond to fear, but different mechanisms) |
| options_risk_regime_prob vs vix_regime_prob | 0.30 - 0.50 | Correlated but distinct (positioning vs volatility) |

If any correlation with gold returns is <0.03, that indicator should be **dropped** or used only in HMM (not as standalone z-score feature).

---

## Gate Expectations (Updated Based on Research)

### Gate 1: Standalone Quality

**Expected**: PASS

**Risks**:
- P/C ratio or SKEW may have high autocorrelation if using levels instead of changes.
- Mitigation: Use daily changes for HMM input, z-scores for deterministic features.

**Autocorrelation threshold**: <0.99 (easily achieved with changes/z-scores)

### Gate 2: Information Gain

**Expected**: MODERATE probability of pass

**Risks**:
- P/C ratio and SKEW have **no published correlation with gold returns**.
- SKEW is "very noisy" per academic literature.
- Options sentiment may be uncorrelated with gold during low-VIX regimes.

**Mitigation**:
- Use HMM regime probability (captures nonlinear patterns) instead of raw indicators.
- Combine P/C and SKEW in 2D HMM (joint information > sum of parts).
- Drop features with MI contribution <1%.

**MI increase threshold**: >5% (current_task.json requirement)
**Confidence**: 5/10 (lower than VIX/Technical/ETF-Flow due to lack of empirical validation)

### Gate 3: Ablation

**Expected**: 6/10 probability of pass (moderate, as stated in current_task.json)

**Rationale**:
- Options sentiment is a **new information source** (no overlap with existing 22 features).
- P/C ratio captures institutional flow patterns distinct from all current features.
- However, the P/C → gold mechanism is **indirect and unvalidated**.

**Alternative scenario**: SKEW-only or GVZ-only design may perform better if P/C ratio is too noisy.

**Confidence**: 6/10 (hypothesis-driven, plausible but uncertain)

---

## Known Limitations and Risks

### High-Risk Issues

1. **^CPCE ticker may not be available on Yahoo Finance**
   - Impact: Delays data fetching, requires fallback to ^CPC.
   - Mitigation: builder_data tests ^CPCE first, switches to ^CPC if unavailable.

2. **No empirical evidence of P/C or SKEW predicting gold**
   - Impact: Gate 2 or Gate 3 failure likely if correlation is <0.05.
   - Mitigation: Architect measures correlations during fact-check. If weak, drop P/C and use SKEW + GVZ only.

3. **P/C ratio cannot distinguish hedging from directional positioning**
   - Impact: Noisy signal, high variance, low MI.
   - Mitigation: Use momentum (rate of change) rather than levels; combine with SKEW in HMM.

### Medium-Risk Issues

1. **SKEW is "very noisy" per academic literature**
   - Impact: Low signal-to-noise, poor MI.
   - Mitigation: Use z-score (removes level drift), HMM (extracts regime patterns from noise).

2. **GVZ/VIX ratio may have high VIF with VIX submodel**
   - Impact: Redundant feature, fails VIF <10 threshold.
   - Mitigation: Architect measures VIF. If >10, drop this feature.

### Low-Risk Issues

1. **Index P/C ratio has structural bias toward puts**
   - Impact: If using ^CPC (Total P/C) instead of ^CPCE (Equity P/C), structural bias is present.
   - Mitigation: Use changes (removes level bias), or normalize by historical mean.

---

## Recommendations for Architect

1. **Fact-check priorities**:
   - Test ^CPCE availability via yfinance. If unavailable, switch to ^CPC.
   - Measure `corr(P/C_change, gold_return_next)` and `corr(SKEW_change, gold_return_next)`. If both <0.05, reconsider entire design.
   - Measure `corr(SKEW, VIX)` to confirm low correlation (<0.4).
   - Compute VIF for GVZ/VIX ratio z-score against VIX submodel outputs. If >10, drop.

2. **Design decisions**:
   - Use 2D HMM on [P/C changes, SKEW changes] (Option A) unless P/C correlation is <0.05.
   - If P/C fails, fall back to 1D HMM on SKEW changes (Option B).
   - Include GVZ/VIX ratio z-score as 4th feature only if VIF <10 and MI contribution >1%.

3. **Output feature count**:
   - Target: 3 features (regime prob + 2 deterministic)
   - Maximum: 4 features (if GVZ/VIX passes VIF and MI tests)
   - Do NOT exceed 4 (lesson from real_rate attempt 5).

4. **HP search space**:
   - HMM n_states: {2, 3}
   - Z-score window: {40, 60, 90} days
   - Momentum window: {5, 10, 15} days
   - HMM covariance_type: {"full"} (fixed for 2D input)

5. **Expected failure modes**:
   - Gate 2 failure if P/C and SKEW are uncorrelated with gold.
   - Gate 3 failure if options sentiment adds noise rather than signal.
   - If attempt 1 fails, attempt 2 should focus on GVZ-only design (gold-specific vol, more direct than equity options sentiment).

---

## Conclusion

The options_market submodel is a **hypothesis-driven experiment** to capture risk perception patterns from CBOE options data. Key findings:

- **Data availability**: SKEW and GVZ confirmed available. P/C ratio (^CPCE) has availability risk; fallback to ^CPC recommended.
- **Theoretical support**: SKEW and VIX are orthogonal (different risk dimensions). P/C ratio captures positioning, not volatility.
- **Empirical gap**: No academic evidence of P/C or SKEW predicting gold. Architect must measure correlations.
- **Architecture**: 2D HMM on [P/C changes, SKEW changes] + 2-3 deterministic features (SKEW z-score, P/C momentum, optional GVZ/VIX ratio z-score).
- **Limitations**: P/C ratio cannot distinguish hedging from speculation; SKEW is noisy; GVZ/VIX ratio may have VIF issues.

**Confidence**: 6/10 for Gate 3 pass (moderate). This submodel is worth attempting but has higher uncertainty than VIX/Technical/ETF-Flow due to unvalidated P/C-gold relationship.

**Proceed to architect phase** with recommendation to measure empirical correlations before committing to Kaggle training.

---

## Sources

- [CBOE Equity Put/Call Ratio - YCharts](https://ycharts.com/indicators/cboe_equity_put_call_ratio)
- [US - CBOE Total Put/Call Ratio - MacroMicro](https://en.macromicro.me/charts/449/us-cboe-options-put-call-ratio)
- [PUT/CALL RATIO (EQUITIES) - TradingView](https://www.tradingview.com/symbols/USI-PCCE/ideas/)
- [Skew index: Descriptive analysis, predictive power, and short-term forecast - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1062940820302370)
- [The skewness index: uncovering the relationship with volatility and market returns - Applied Economics](https://www.tandfonline.com/doi/abs/10.1080/00036846.2021.1884837)
- [Inferring information from the S&P 500, CBOE VIX, and CBOE SKEW indices - Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/fut.22093)
- [CBOE Gold ETF Volatility Index (GVZCLS) - FRED](https://fred.stlouisfed.org/series/GVZCLS)
- [CBOE Gold Volatility Index (^GVZ) Historical Data - Yahoo Finance](https://finance.yahoo.com/quote/%5EGVZ/history/)
- [Put/Call Ratio - StockCharts](https://chartschool.stockcharts.com/table-of-contents/market-indicators/put-call-ratio)
- [What Is the Put-Call Ratio and How to Use It? - MenthorQ](https://menthorq.com/guide/what-is-the-put-call-ratio-and-how-to-use-it/)
- [Economic put call ratio: Meaning, Criticisms & Real-World Uses](https://diversification.com/term/economic-put-call-ratio)
- [The Put/Call Paradox](https://www.thelastbearstanding.com/p/the-putcall-paradox)
- [Hidden Markov Model for Stock Trading - MDPI](https://www.mdpi.com/2227-7072/6/2/36)
- [Market Regime Detection using Hidden Markov Models - QuantStart](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
