# Research Report: Real Interest Rate Submodel (Attempt 1)

**Feature**: real_rate (DFII10 - 10Y TIPS)
**Attempt**: 1
**Phase**: Smoke Test
**Date**: 2026-02-14
**Researcher**: Claude Sonnet 4.5

---

## Research Questions

1. What are the most effective methods for detecting regime changes in real interest rates?
2. How should persistence of real rate changes be measured?
3. Can multi-country real rate data (G7/G10) be used to increase training samples?
4. What window lengths are appropriate for rate-of-change features (velocity, acceleration)?
5. What is the typical lag between real rate regime shifts and gold price reactions?

---

## 1. Regime Change Detection Methods

### Overview

Three primary approaches exist for regime detection in financial time series:

1. **Hidden Markov Models (HMM)**
2. **Markov-Switching Regression Models**
3. **Self-Exciting Threshold Autoregression (SETAR)**

### Hidden Markov Models (HMM)

**Concept**: HMM treats market regimes as hidden states that generate observable returns. The model captures regime transitions probabilistically.

**Strengths**:
- Well-established in finance for detecting "Bull/Bear" or "Volatile/Calm" regimes
- Captures volatility persistence, time-varying correlations, skewness, and kurtosis
- Probabilistic output suitable for downstream models

**Implementation**:
- Python library: `hmmlearn` (GaussianHMM with Gaussian emissions)
- Not native PyTorch but can be integrated
- Typical setup: 2-3 hidden states for financial regimes

**Limitations**:
- Standard implementations use scikit-learn style APIs, not PyTorch
- Requires careful selection of number of states (2-3 recommended for simplicity)

### Markov-Switching Regression

**Concept**: Extension of HMM where regime-dependent parameters affect the model directly.

**Implementation**:
- Available in `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`
- Set `switching_variance=True` for regime detection based on variance changes

**Suitability**: Better for econometric analysis than feature extraction for neural networks.

### Self-Exciting Threshold Autoregression (SETAR)

**Concept**: SETAR(k, p) models use threshold values to separate k+1 regimes based on past observations. The delay parameter d determines which lag is used for threshold comparison.

**Strengths**:
- Non-parametric regime identification
- No distributional assumptions required
- Captures asymmetric dynamics

**Implementation**:
- `statsmodels` has SETAR module
- Standalone package: `pip install setar`
- Example: `setar.SETAR(x, p=2, d=0, q=0, k=1)` for 2 regimes

**Challenges**:
- Requires grid search over delay (d) and threshold (c) parameters
- Objective function has many local optima
- Computationally expensive for large searches

### Recommendation for Smoke Test

**Priority 1: Simple Threshold-Based Regime Detection (Manual Implementation in PyTorch)**

For the smoke test, implement a simple threshold-based approach directly in PyTorch:

- Define regimes based on percentiles (e.g., low: <33rd, mid: 33-67th, high: >67th)
- Use rolling windows to compute regime probabilities
- Output: Soft regime probabilities via sigmoid/softmax

**Rationale**:
- Fastest to implement
- Full PyTorch compatibility
- Transparent and debuggable
- Sufficient for smoke test verification

**Priority 2: HMM with hmmlearn (if time permits)**

Use GaussianHMM from hmmlearn to fit regimes, then convert outputs to PyTorch tensors for the submodel.

**Avoid for Smoke Test**: Markov-Switching Regression and full SETAR (too complex for initial verification).

---

## 2. Persistence Measurement Methods

### Hurst Exponent

**Definition**: Measures long-term memory in time series. Values range from 0 to 1.

**Interpretation**:
- H = 0.5: Random walk (Brownian motion)
- 0.5 < H < 1.0: Persistent (trending) behavior
- 0 < H < 0.5: Anti-persistent (mean-reverting) behavior

**Calculation Methods**:

1. **R/S (Rescaled Range) Method**:
   - Most common implementation
   - Available in `hurst` library: `pip install hurst`
   - Computes variance of lagged differences, then linear fit on log-log plot

2. **Simple Implementation**:
   ```python
   # Pseudo-code
   lags = range(2, max_lag)
   tau = [np.std(np.diff(series, lag)) for lag in lags]
   poly = np.polyfit(np.log(lags), np.log(tau), 1)
   hurst_exponent = poly[0] * 2.0
   ```

**Challenges**:
- Results sensitive to max_lag parameter choice
- Different implementations yield inconsistent results on real financial data
- Works well on synthetic data but diverges on actual time series

**Recommendation**: Use rolling Hurst exponent with fixed max_lag (20-60 days for daily data).

### Variance Ratio Test

**Concept**: Compares variance of k-period returns to k times the variance of 1-period returns. Under random walk, ratio should equal 1.

**Interpretation**:
- VR > 1: Positive autocorrelation (persistence)
- VR < 1: Negative autocorrelation (mean reversion)

**Advantage**: More robust than Hurst in some applications, well-grounded in financial econometrics.

**Implementation**: Available in various Python packages, can be implemented directly.

### Autocorrelation Analysis

**Simple Alternative**: Rolling autocorrelation at lag 1, 5, 20.

**Advantage**: Simple, interpretable, fast to compute.

**Output**: Positive ACF indicates persistence, negative indicates mean reversion.

### Recommendation for Smoke Test

**Priority 1: Rolling Autocorrelation (AC)**

- Compute AC at lags 1, 5, 10 on rolling 60-day windows
- Fast, interpretable, PyTorch-friendly
- Output single "persistence_score" as mean of AC(1), AC(5), AC(10)

**Priority 2: Hurst Exponent**

- Use rolling 60-day Hurst via `hurst` library
- Integrate into data preprocessing before PyTorch model

**Avoid for Smoke Test**: Variance Ratio tests (adds complexity without clear benefit for initial test).

---

## 3. Multi-Country Real Rate Data Availability (FRED)

### FRED TIPS Data Scope

**U.S. Coverage**:
- FRED maintains **177 economic data series** for Treasury Inflation-Indexed Securities
- **173 series** tagged with TIPS
- **93 series** specifically for 10-year TIPS
- Primary series: DFII10 (10-Year TIPS, daily, 2003-present)
- Alternative maturities: 5Y, 7Y, 20Y, 30Y TIPS available

**Date Range Example**:
- DFII10: 2003-01-02 to present (daily)
- DFII7: 2003-01-03 to 2026-02-06

### G7/G10 Availability

**Critical Finding**: FRED's inflation-indexed securities data is **overwhelmingly U.S.-focused**.

- FRED has **64 series** tagged with both "G7" and "CPI" (Consumer Price Index)
- **No direct TIPS-equivalent series found for other G7/G10 countries** in search results

**Implication**: Multi-country TIPS data via FRED is **NOT readily available**.

### Alternative Approaches for Multi-Country Data

1. **Construct Synthetic Real Rates**:
   - Use nominal government bond yields (available for G10 on FRED)
   - Subtract inflation expectations or realized inflation
   - Formula: Real Rate ≈ Nominal Yield - Inflation Expectation
   - FRED series examples:
     - Germany: IRLTLT01DEM156N (10Y nominal)
     - Japan: IRLTLT01JPM156N
     - UK: IRLTLT01GBM156N
     - Canada: IRLTLT01CAM156N

2. **Limit Scope to U.S. Data**:
   - DFII10 provides ~6,000 daily observations (2003-present)
   - Split: 4,200 train / 900 val / 900 test
   - Sufficient for smoke test

### Recommendation

**For Smoke Test**: Use **U.S. data only** (DFII10).

**For Future Attempts**: Explore synthetic real rate construction for G7/G10 using nominal yields minus inflation expectations, but verify data quality and alignment carefully. This requires architect approval to ensure VIF concerns are addressed.

**Note for Architect**: Multi-country expansion requires verification of:
1. Actual FRED series IDs for G10 nominal yields and inflation
2. Data quality and completeness (avoid series with gaps)
3. VIF impact when combining with DGS10 (yield curve feature)

---

## 4. Appropriate Window Lengths for Velocity/Acceleration Features

### Standard Technical Analysis Windows

**Common Moving Average Windows**:
- **Short-term (10-20 days)**: Captures immediate price action, high sensitivity, more false signals
- **Medium-term (50 days)**: Gauges intermediate trends, balanced responsiveness
- **Long-term (100-200 days)**: Reflects broader trend, slower to react, fewer false signals

**Popular Combinations**:
- 20/50/200-day triple moving average system
- 50/200-day crossover for trend reversals

### Trade-offs

**Shorter Windows (10-20 days)**:
- React faster to changes
- Generate more signals (including false positives)
- Suitable for tactical signals

**Longer Windows (50-200 days)**:
- Smoother, more stable
- Delayed reaction to regime changes
- Better for strategic trends

### Application to Real Interest Rates

Real interest rates change slowly due to monetary policy inertia. Federal Reserve policy rate changes take **6-12 months to affect economic variables meaningfully**.

**Velocity (First Derivative)**:
- Short window: 20-day change (captures recent momentum)
- Medium window: 60-day change (aligns with quarterly monetary policy cycles)
- Long window: 120-day change (semi-annual trend)

**Acceleration (Second Derivative)**:
- 20-day change in 20-day velocity (captures inflection points)
- 60-day change in 60-day velocity (medium-term turning points)

### Recommendation for Smoke Test

**Velocity Features**:
- `velocity_20d`: (DFII10[t] - DFII10[t-20]) / rolling_std_60d
- `velocity_60d`: (DFII10[t] - DFII10[t-60]) / rolling_std_60d

**Acceleration Features**:
- `accel_20d`: velocity_20d[t] - velocity_20d[t-20]

**Normalization**: Divide by rolling 60-day standard deviation to make comparable across regimes.

**Rationale**:
- 20-day captures tactical shifts (roughly 1 month)
- 60-day captures quarterly patterns
- Normalization prevents scale issues during regime changes

---

## 5. Lag Between Real Rate Regime Shifts and Gold Price Reactions

### Empirical Evidence

**Strong Inverse Relationship**:
- The negative correlation between real interest rates and gold prices is well-established in academic literature
- Gold is a long-duration durable asset; its price has a strong inverse relationship with long-term real interest rates
- Theoretical basis: Gold pays no yield, so its opportunity cost rises with real rates

### Time-Period Variation

**Critical Finding**: The relationship strength varies significantly across time periods.

**Before 2001**:
- Predicted negative co-movement did NOT show up strongly

**2001-2012**:
- Long-term real rates fell ~400 basis points
- Real gold price rose over 5x
- Strong negative correlation observed

**Post-2012**:
- Negative relationship confirmed in annual levels, quarterly innovations, and **daily differences**
- Relationship strongest when real rates are low

**Post-Pandemic (2022+)**:
- Real rates appear to be **losing prominence** as primary drivers
- Physical demand from emerging markets (e.g., China, India) becoming more decisive

### Lag Structure

**Daily Differences**: Negative effect confirmed in daily data, suggesting **near-immediate** or **same-day** reaction in modern markets (post-2001).

**Policy Transmission Lag**: Federal Reserve policy changes take **6-12 months** to affect real economic variables, but **asset prices react faster** due to forward-looking expectations.

**Implication for Gold**:
- When real rates change due to expectations (TIPS yields reflect market expectations), gold reacts quickly (0-5 days)
- When real rates change due to policy, gold may lead or lag by days to weeks depending on anticipation

### Regime-Dependent Dynamics

**Low Real Rate Regimes** (e.g., negative real rates):
- Stronger negative correlation
- Gold more sensitive to changes

**High Real Rate Regimes**:
- Weaker correlation
- Other factors (USD strength, risk appetite) dominate

### Recommendation for Smoke Test

**Lag Features**:
- Use contemporaneous (t) and short lags (t-1, t-5) for real rate features
- Do NOT use lags longer than 5 days to avoid staleness

**Regime Interaction**:
- Submodel should output regime state so meta-model can learn regime-dependent lag dynamics
- Meta-model receives both real_rate[t] and submodel regime features simultaneously

**Avoid**: Long lags (20+ days) are unlikely to improve signal and risk information leakage if not handled carefully.

---

## Recommended Methodology (Smoke Test)

### Priority Stack for Architect

Based on smoke test constraints (simple, fast, verifiable), recommend the following design:

1. **Regime Detection**: Percentile-based thresholds (33rd/67th percentile) → soft probabilities via rolling window
2. **Persistence**: Rolling 60-day autocorrelation at lags 1, 5, 10 → mean as persistence_score
3. **Velocity**: 20-day and 60-day changes, normalized by rolling 60-day std
4. **Acceleration**: 20-day change in 20-day velocity
5. **Data**: U.S. DFII10 only (multi-country expansion in future attempts)

### Submodel Architecture Suggestion

**Input**: Raw DFII10 daily series
**Output**: 4-5 continuous features

| Feature | Description | Calculation |
|---------|-------------|-------------|
| regime_prob_low | Probability of low real rate regime | Softmax over rolling percentile bins |
| regime_prob_high | Probability of high real rate regime | Softmax over rolling percentile bins |
| persistence_score | Trend persistence measure | Mean of AC(1,5,10) on 60-day window |
| velocity_norm | Normalized rate of change | (DFII10[t] - DFII10[t-20]) / std_60d |
| accel_norm | Normalized acceleration | Change in velocity_norm over 20 days |

**Model Type**: Simple feedforward MLP or GRU with lookback window (e.g., 60 days) → linear projection to 5 outputs.

**Why This Works**:
- All features computable from DFII10 alone (no external dependencies)
- Outputs are continuous, suitable for meta-model regression input
- Low VIF risk (features measure different aspects: regime, persistence, momentum)
- Fast to implement and train

---

## Data Sources Summary

| Data Type | Source | ID/Ticker | Period | Delay |
|-----------|--------|-----------|--------|-------|
| 10Y TIPS Yield | FRED | DFII10 | 2003-present | T+1 |
| 5Y TIPS Yield | FRED | DFII5 | 2003-present | T+1 |
| 7Y TIPS Yield | FRED | DFII7 | 2003-present | T+1 |
| 20Y TIPS Yield | FRED | DFII20 | 2004-present | T+1 |
| 30Y TIPS Yield | FRED | DFII30 | 2010-present | T+1 |

**Retrieval Code Example**:
```python
from fredapi import Fred
import os

fred = Fred(api_key=os.environ['FRED_API_KEY'])
dfii10 = fred.get_series('DFII10')  # Returns pandas Series
```

**Multi-Country (Nominal Yields, for future reference)**:
- Germany 10Y: IRLTLT01DEM156N
- Japan 10Y: IRLTLT01JPM156N
- UK 10Y: IRLTLT01GBM156N
- Canada 10Y: IRLTLT01CAM156N

**Caution**: Multi-country TIPS equivalents are NOT available on FRED. Synthetic real rates require nominal yield - inflation expectation, which adds complexity and potential VIF issues.

---

## Attention Points for Architect

### VIF Concern

The real_rate feature (DFII10) is mathematically related to:
- Nominal yield (DGS10) via Fisher equation: DFII10 ≈ DGS10 - Inflation Expectation
- Yield curve feature will also use DGS10

**Mitigation Strategy**:
- Submodel should output **dynamics** (regime, persistence, velocity, acceleration), NOT the raw level
- Raw DFII10 level goes to meta-model directly via base_features.csv
- Submodel features capture temporal patterns invisible to tree-based models
- Architect must verify VIF < 10 after combining submodel outputs with base features

### Implementation Complexity

**HMM via hmmlearn**:
- Requires fitting separate HMM model
- Convert outputs to PyTorch tensors
- Adds dependency outside PyTorch ecosystem

**PyTorch-Native Approach**:
- Implement threshold-based regime detection directly in PyTorch
- Use rolling window operations (e.g., unfold) for persistence and velocity
- Fully differentiable, end-to-end trainable

**Recommendation**: For smoke test, prefer PyTorch-native approach. HMM can be explored in attempt 2+ if simple methods fail Gate 3.

### Hyperparameter Search Space

Suggest for Optuna:

| Hyperparameter | Range | Note |
|----------------|-------|------|
| lookback_window | [30, 60, 120] | Days of history for regime/persistence |
| hidden_dim | [32, 64, 128] | If using GRU/LSTM |
| num_layers | [1, 2] | Depth |
| dropout | [0.1, 0.3, 0.5] | Regularization |
| regime_percentiles | Fixed: [33, 67] | Or search [25,75], [20,80] |
| persistence_lags | Fixed: [1,5,10] | Or search combinations |
| velocity_window | [10, 20, 30, 60] | Days for velocity calculation |
| learning_rate | [1e-4, 1e-3, 1e-2] | Log scale |

**Smoke Test Simplification**: Fix some parameters (e.g., regime percentiles, persistence lags) to reduce search space for 5-trial Optuna run.

---

## Limitations and Uncertainties

1. **Multi-Country Data**: FRED does not provide TIPS-equivalent data for G7/G10. Synthetic real rate construction is possible but requires careful validation. **[Requires confirmation by architect]**

2. **HMM Implementation**: Standard HMM libraries (hmmlearn) are not PyTorch-native. Integration is possible but adds complexity. **[Architect to decide: use hmmlearn or implement threshold-based approach]**

3. **Lag Structure**: While daily differences show negative correlation, the exact lag (0, 1, or 5 days) for maximum predictive power is not specified in literature. **[Requires empirical testing]**

4. **Regime Stability Post-2022**: Recent evidence suggests real rates losing prominence as gold drivers. The submodel may perform differently in recent data vs historical 2003-2021 period. **[Monitor in evaluation]**

5. **Hurst Exponent Reliability**: Literature notes that Hurst exponent yields inconsistent results across implementations on real financial data. If used, results should be validated carefully. **[Consider using simpler autocorrelation instead]**

---

## Sources

### Regime Detection
- [Market Regime Detection using Hidden Markov Models in QSTrader | QuantStart](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Stock Market Regime Detection using Hidden Markov Models | Medium](https://medium.com/@sticktothemodels48/stock-market-regime-detection-using-hidden-markov-models-8c30953a3f27)
- [Regime Shift Models - A Fascinating Use Case of Time Series Modeling](https://www.analyticsvidhya.com/blog/2019/10/regime-shift-models-time-series-modeling-financial-markets/)
- [SETAR Model Functionality | Chad Fulton](http://www.chadfulton.com/topics/setar_model_functionality.html)
- [SETAR Model | Wikipedia](https://en.wikipedia.org/wiki/SETAR_(model))

### Persistence Measurement
- [Unit root test and Hurst exponent — Time series analysis with Python](https://filippomb.github.io/python-time-series-handbook/notebooks/06/unit-root-hurst.html)
- [Introduction to the Hurst exponent - with code in Python | Towards Data Science](https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e/)
- [GitHub - Mottl/hurst: Hurst exponent evaluation and R/S-analysis in Python](https://github.com/Mottl/hurst)
- [Detecting trends and mean reversion with the Hurst exponent | Macrosynergy](https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/)

### FRED Data Availability
- [Treasury Inflation-Indexed Securities | FRED | St. Louis Fed](https://fred.stlouisfed.org/categories/82)
- [10-Year, TIPS - Economic Data Series | FRED | St. Louis Fed](https://fred.stlouisfed.org/tags/series?t=10-year%3Btips)
- [TIPS - Economic Data Series | FRED | St. Louis Fed](https://fred.stlouisfed.org/tags/series?t=tips)

### Window Lengths
- [Simple Moving Average (SMA) — Indicators and Strategies — TradingView](https://www.tradingview.com/scripts/simplemovingaverage/)
- [What Are Moving Averages? Gauging Market Trends | Britannica Money](https://www.britannica.com/money/what-are-moving-averages)
- [Velocity of money | FRED Blog](https://fredblog.stlouisfed.org/2015/01/the-velocity-of-money/)

### Real Rate-Gold Relationship
- [NBER WORKING PAPER SERIES GOLD'S VALUE AS AN INVESTMENT](https://www.nber.org/system/files/working_papers/w31386/w31386.pdf)
- [Real Interest Rates and Gold - Explained](https://www.goldpriceforecast.com/explanations/gold-real-interest-rates/)
- [What drives gold prices? - Federal Reserve Bank of Chicago](https://www.chicagofed.org/publications/chicago-fed-letter/2021/464)
- [Do gold prices respond to real interest rates? Evidence from the Bayesian Markov Switching VECM model - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1042443118300167)

---

## Conclusion

For the smoke test, prioritize simplicity and PyTorch compatibility:

1. **Regime detection**: Percentile-based thresholds with rolling windows
2. **Persistence**: Rolling autocorrelation (simpler and more reliable than Hurst)
3. **Velocity/Acceleration**: 20-day and 60-day windows, normalized by volatility
4. **Data scope**: U.S. DFII10 only (sufficient samples for smoke test)
5. **Architecture**: Small GRU or MLP with 60-day lookback → 5 continuous outputs

This approach balances research best practices with smoke test constraints, ensuring the pipeline can be verified quickly while maintaining scientific rigor for future iterations.

---

**Word Count**: ~2,950 (within target range, detailed enough for architect to design)

**Next Step**: Pass to architect for fact-checking and design document creation.
