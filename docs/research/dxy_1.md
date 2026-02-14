# Research Report: DXY Submodel (Attempt 1)

**Date**: 2026-02-14
**Feature**: Dollar Index (DXY)
**Attempt**: 1
**Researcher**: Claude Sonnet 4.5

---

## Executive Summary

This research investigates optimal methods for extracting regime, divergence, and volatility features from DXY and its 6 constituent currency pairs. **Key advantage over real_rate**: All data is daily frequency with no interpolation required.

**Primary Recommendation**: **Hybrid Approach C** - Combine Hidden Markov Model regime detection on DXY with PCA-based cross-currency divergence analysis. Output 3 features: regime probability, cross-currency divergence index, and FX volatility state.

**Critical Success Factors**:
1. Daily frequency eliminates real_rate's fatal interpolation flaw
2. Keep output to 3 columns maximum (lesson from real_rate's 7-column failure)
3. Use deterministic PCA for divergence to ensure Gate 1 pass
4. HMM on DXY provides regime context without overfitting risk

**Expected Performance**: Gate 2 pass likely (MI increase >5%), Gate 3 uncertain but promising given clean daily data and structural feature design.

---

## Research Questions & Findings

### Question 1: FX Regime Detection Methods

**Research Question**: What methods are most effective for detecting momentum vs mean-reversion regimes in FX markets? Candidates: Hidden Markov Model, threshold autoregressive model (TAR), Markov-switching model. Which has the best track record for daily FX data?

**Findings**:

#### Hidden Markov Models (HMM)
- **Best suited for daily frequency**: HMM works best with medium- to long-term data such as daily or weekly prices, less effective for high-frequency intraday data due to noise
- **Strong forex performance**: HMM demonstrated superior performance compared to LSTM, ARIMA, and RNN for forex applications
- **Python implementation**: Well-established via `hmmlearn` library with `GaussianHMM` class; modular and beginner-friendly
- **Regime interpretation**: For each market state (bullish, bearish, neutral), assumes price changes follow normal distribution
- **Enhanced detection**: HMM applied to directional change indicators detects regime shifts better than when applied to price return volatility alone

#### Threshold Autoregressive (TAR) / SETAR Models
- **Regime switching mechanism**: Self-Exciting Threshold AutoRegressive (SETAR) models allow higher flexibility through regime-dependent AR parameters
- **Threshold challenge**: Major difficulty is estimating thresholds; requires optimization over finite intervals
- **Testing complexity**: Likelihood ratio tests needed to compare TAR vs linear AR models
- **Limited forex evidence**: Search results focused on methodology rather than demonstrated forex performance

#### Markov-Switching Models
- **Regime persistence**: Semi-Markov models where state termination probability increases with age can induce short-term momentum and subsequent reversal
- **Multivariate capability**: Joint multivariate HMM can detect regime-dependent correlation matrices across currency pairs
- **Well-established in FX**: Extensive literature documents momentum and mean-reversion patterns detected via Markov-switching frameworks

**Recommendation for Attempt 1**: **Hidden Markov Model (HMM)** with 2-3 states
- Proven daily-frequency performance in forex
- Straightforward Python implementation
- Outputs regime probabilities (not just discrete states) providing soft transitions
- Lower implementation risk than TAR (threshold estimation complexity)
- Can be trained deterministically with fixed random seed

---

### Question 2: Cross-Currency Divergence Quantification

**Research Question**: How to quantify cross-currency divergence within a currency index? Candidates: PCA on daily currency returns (PC1 = common factor, residuals = divergence), pairwise correlation rolling window, dispersion index (std of individual currency returns). Which provides the most informative signal?

**Findings**:

#### Principal Component Analysis (PCA)
- **Common factor extraction**: First principal component (PC1) captures common USD movement across currency basket
- **Divergence measurement**: Residual variance after PC1 represents cross-currency divergence
- **Proven forex application**: Analysis of 15 currencies showed 3 PCs explained ~65% of co-variation (commodity prices, central bank activity, Asia-specific driver)
- **Forecasting utility**: PC loadings from bilateral exchange rate growth used successfully for exchange rate forecasting
- **Implementation**: Forms linear combinations of observed variables into independent components summarizing cross-variation
- **Interpretation**: Sign and size of component loadings reveal what each component represents

#### Correlation-Based Approaches
- **Rolling correlation**: Pairwise correlation windows track changing relationships but produce large output matrices (15 pairs for 6 currencies)
- **Practical limitation**: High dimensionality makes it unsuitable for 3-column output constraint

#### Dispersion Index
- **Direct measurement**: Standard deviation of individual currency returns provides simple divergence metric
- **Less informative**: Does not distinguish between common movement and idiosyncratic shocks
- **Potential complementary role**: Could serve as volatility proxy but not as informative as PCA residuals

**Recommendation for Attempt 1**: **PCA on daily returns of 6 constituent currencies**
- PC1 loading strength = common USD factor
- Explained variance ratio by PC1 = inverse of divergence (high PC1 variance → low divergence)
- Residual variance or (1 - PC1_explained_variance) = divergence index
- **Critical advantage**: Deterministic, no overfitting risk, guaranteed Gate 1 pass
- Output single divergence scalar (contribution to 3-column limit: 1 column)

---

### Question 3: DXY-Gold Correlation Regime Shifts

**Research Question**: What is the empirical evidence on DXY-gold correlation regime shifts? When does the standard negative correlation break down (e.g., 2008 crisis, 2020 COVID)? How frequent are these regime shifts?

**Findings**:

#### Historical Correlation Pattern
- **Baseline relationship**: Gold and US dollar negatively correlated (-0.26 on average)
- **Inverse movement**: Assets typically move in opposite directions, preventing simultaneous safe-haven function
- **2026 context**: Correlation has become less tight in recent years, relationship evolving significantly

#### Crisis Period Behavior (2008 GFC)
- **Weak safe-haven co-movement**: Intermediate evidence for USD as safe haven, Gold showed varying effectiveness
- **Correlation maintained**: Negative correlation generally held but with reduced magnitude
- **Both assets rose**: Some periods showed simultaneous USD and gold strength during peak crisis

#### Crisis Period Behavior (2020 COVID)
- **Divergent performance**: Gold's safe-haven role diminished; became "very risky in some settings"
- **Conflicting evidence**: Some studies found gold maintained safe-haven status, others found it failed to protect wealth
- **USD weakness early 2020**: Greenback weakness coincided with gold rally, then reversed
- **Swiss franc advantage**: CHF served as better safe haven than USD during COVID vs both being safe havens in 2008

#### 2025-2026 Regime Shift
- **Policy-driven volatility**: "Warsh Shock" announcement triggered violent rotation; gold dropped from $5,600 to $4,400
- **Non-traditional drivers**: Central bank diversification, geopolitical uncertainty, institutional flows increasingly influential beyond dollar correlation
- **Record central bank buying**: Structural shift supporting gold regardless of dollar movements

#### Regime Shift Frequency
- **Multiple regimes**: Standard inverse correlation, safe-haven co-movement, policy-driven divergence
- **Shift triggers**: Risk events, policy pivots, structural market changes
- **Increasing complexity**: 2020s show more frequent correlation regime shifts than 2000s-2010s

**Recommendation for Attempt 1**:
- **Regime detection critical**: DXY-gold relationship is non-stationary; regime features will help meta-model adapt
- **Context matters**: Same DXY level in different regimes has different gold implications
- **Expected benefit**: Regime probability feature should provide valuable context for meta-model to distinguish when standard inverse correlation holds vs breaks down

---

### Question 4: Lookback Windows for FX Momentum/Volatility

**Research Question**: What lookback windows are appropriate for FX momentum and volatility features? Literature suggests 5/10/20/60 day windows. Is there consensus on optimal lookback for gold-relevant FX signals?

**Findings**:

#### General Momentum Literature
- **Established range**: 1 month through 12 months for momentum to manifest
- **Regime-dependent performance**: 12-month lookback best 1988-2008; 3-month lookback best 2008-2019
- **Common practitioner windows**: 3-6 months for momentum strategies
- **GTAA standard**: 10-month lookback; Dual Momentum uses 60-day and 100-day combination

#### Short-Term Windows (5-20 days)
- **Limited specific evidence**: Search results didn't provide 2026 research comparing these exact timeframes for FX
- **Mean-reversion bias**: Shorter lookbacks favor mean-reversion rather than momentum detection
- **Higher noise**: Very short windows (<10 days) may capture noise rather than regime structure

#### Volatility-Adjusted Approaches
- **Adaptive lookback**: Volatility-Adjusted Time Series Momentum (VATSM) dynamically adjusts lookback
- **High volatility → shorter lookback**: Avoids whipsaws during turbulent periods
- **Low volatility → longer lookback**: Captures sustained trends in calm markets
- **Superior to fixed windows**: Adapts to market conditions

#### Recommendations for Daily FX Data
- **Momentum detection**: Longer windows (20-60 days) more appropriate than 5-10 days
- **Regime persistence**: FX regimes typically last weeks to months, not days
- **Volatility measurement**: 20-day rolling standard deviation is industry standard

**Recommendation for Attempt 1**:
- **HMM training**: Use all available daily data; model learns regime durations endogenously
- **Volatility window**: 20-day rolling realized volatility (industry standard)
- **Avoid overfitting**: Do not optimize lookback windows in Attempt 1; use well-established defaults
- **PCA window**: Daily returns (no lookback smoothing) to preserve signal freshness

---

### Question 5: DXY Seasonality Effects

**Research Question**: Are there known seasonality or day-of-week effects in DXY that could create spurious patterns? Should we control for these?

**Findings**:

#### Seasonal Patterns
- **January-February strength**: Analysis shows buy date January 11, sell date February 25 produced 16.11% total return over 10 years (60% win rate)
- **Limited intra-month evidence**: Specific day-of-week effects for DXY not found in search results
- **Broader FX seasonality**: Month-end flows, quarter-end rebalancing can create temporary patterns

#### 2026 Market Outlook
- **Policy-driven**: Seasonality less relevant than regime shifts driven by Fed policy, fiscal dynamics
- **Downward bias**: 2026 consensus expects two-way volatility with dollar weakness bias
- **Event-driven**: Election year, trade tensions override typical seasonal patterns

#### Spurious Pattern Risk
- **Low concern for daily data**: Seasonality more problematic for monthly data with small sample sizes
- **DXY sample size**: ~2,500 daily observations reduces spurious pattern risk
- **Real concern**: Real_rate's monthly interpolation created artificial patterns; DXY daily data avoids this

**Recommendation for Attempt 1**:
- **Do NOT add seasonal controls**: With 2,500+ daily observations, seasonality is minor factor
- **Focus on regime dynamics**: Policy and risk sentiment shifts dominate seasonal effects
- **Revisit if needed**: If evaluator identifies suspicious seasonal artifacts in output, address in Attempt 2
- **Rationale**: Avoid premature complexity; let HMM capture regime patterns organically

---

### Question 6: FX Volatility-Gold Relationship

**Research Question**: What is the relationship between FX implied volatility and gold returns? Can realized volatility of currency pairs serve as a useful proxy?

**Findings**:

#### FX Volatility Impact on Gold
- **Significant relationship**: Gold posted 67% annual return with variation across currencies due to FX volatility
- **Currency-specific effects**: FX fluctuations are significant factor in gold returns across different markets

#### Commodity-FX Linkages
- **AUD/JPY as risk gauge**: Accurately gauges global risk sentiment; AUD influenced by commodity prices including gold
- **ZAR-gold connection**: South Africa as major gold producer means gold price changes impact USD/ZAR directly

#### Recent Volatility Context (2026)
- **Elevated levels**: CBOE Gold ETF Volatility at 36.93 in February 2026
- **Volatility drivers**: Currency devaluation, protests, geopolitical tension drive gold price swings
- **Market turbulence**: Economic events (interest rate changes, commodity price drops) generate FX volatility affecting gold

#### Realized vs Implied Volatility
- **Implied volatility**: Ideal but not readily available for all currency pairs via free APIs
- **Realized volatility proxy**: Historical volatility calculable from daily data serves as practical alternative
- **Implementation**: Rolling standard deviation of returns captures volatility regime

**Recommendation for Attempt 1**:
- **Use realized volatility**: 20-day rolling standard deviation of DXY returns
- **Z-score normalization**: Convert to z-score relative to long-term average to create volatility state feature
- **Avoid individual pair volatility**: Computing volatility for all 6 currencies would exceed column limit; DXY aggregate volatility sufficient
- **Expected signal**: Elevated FX volatility → increased gold price uncertainty → useful meta-model context

---

### Question 7: VIF and Multicollinearity Management

**Research Question**: How should cross-currency features be normalized to avoid VIF issues with the raw dxy_dxy base feature? The submodel output must have VIF < 10 against existing base features.

**Findings**:

#### VIF Interpretation Standards
- **VIF 1-5**: Moderate multicollinearity, generally acceptable
- **VIF > 5**: High multicollinearity, standard error inflation begins
- **VIF > 10**: Serious multicollinearity, unstable coefficient estimates
- **Project threshold**: VIF < 10 required for Gate 2 pass

#### Normalization Strategies
- **Z-score standardization**: Brings all variables to common scale, reduces numerical instability
- **Mean-centering**: Reduces correlation with intercept but doesn't eliminate variable collinearity
- **Scaling**: Does not change correlation structure or VIF values

#### Feature Engineering Approaches
- **Avoid redundancy**: Do not output features that are linear transformations of base features
- **Orthogonalization**: PCA creates orthogonal components by design
- **Residual features**: Use residuals from regression of feature on base_dxy to remove linear dependence
- **Regularization**: Ridge/Lasso can handle multicollinearity but doesn't solve VIF for feature engineering

#### Specific VIF Concerns for DXY
- **Base feature**: `dxy_dxy` is raw DXY close price (single daily value)
- **Risk scenarios**:
  - Raw DXY level → high VIF with dxy_dxy
  - DXY change/return → moderate correlation, likely acceptable
  - Regime probability → low correlation (captures state, not level)
  - Divergence index → very low correlation (captures cross-sectional structure)
  - Volatility state → low correlation (captures dispersion, not level)

**Recommendation for Attempt 1**:
- **Regime probability**: Output as-is; represents state, not price level (VIF safe)
- **Cross-currency divergence**: PCA-based divergence orthogonal to common DXY movement (VIF safe)
- **Volatility state**: Z-score of realized volatility, not level-dependent (VIF safe)
- **Verification step**: Architect must compute VIF during design; if VIF > 10, use residualization (regress submodel feature on dxy_dxy, output residuals)
- **Safety margin**: Target VIF < 7 to ensure robust pass even with numerical variations

---

## Recommended Approach for Attempt 1

### Architecture: Hybrid Deterministic-Probabilistic

**Component 1: HMM Regime Detection on DXY**
- Model: Gaussian HMM with 2 states (trending vs mean-reverting)
- Input: Daily DXY returns
- Training: Full historical data (2015-2025)
- Output: `dxy_regime_trend_prob` (probability of being in trending regime, 0-1)

**Component 2: PCA Cross-Currency Divergence**
- Model: PCA on 6 constituent currency daily returns
- Input: Daily returns of EURUSD, USDJPY, GBPUSD, USDCAD, USDSEK, USDCHF
- Method: Rolling 60-day window PCA to allow time-varying factor structure
- Output: `dxy_cross_currency_divergence` = 1 - PC1_explained_variance_ratio

**Component 3: FX Volatility State**
- Model: Realized volatility z-score
- Input: 20-day rolling standard deviation of DXY returns
- Normalization: Z-score relative to full-sample mean/std
- Output: `dxy_volatility_zscore`

**Total Output**: 3 columns, all daily frequency, no interpolation

---

## Data Sources & Acquisition

### Primary DXY Data
| Data | Source | Ticker | Frequency | Available From | Notes |
|------|--------|--------|-----------|----------------|-------|
| DXY Index | Yahoo Finance | DX-Y.NYB | Daily | 2015-01-02 | Already in data/raw/dxy.csv |

### Constituent Currency Pairs
| Currency | Source | Ticker | Weight in DXY | Notes |
|----------|--------|--------|---------------|-------|
| EUR/USD | Yahoo Finance | EURUSD=X | 57.6% | Largest component |
| USD/JPY | Yahoo Finance | JPY=X | 13.6% | Safe-haven pair |
| GBP/USD | Yahoo Finance | GBPUSD=X | 11.9% | Cable |
| USD/CAD | Yahoo Finance | CAD=X | 9.1% | Commodity currency |
| USD/SEK | Yahoo Finance | SEK=X | 4.2% | Nordic currency |
| USD/CHF | Yahoo Finance | CHF=X | 3.6% | Safe-haven currency |

### Data Fetching Code Example

```python
import yfinance as yf
import pandas as pd

# Fetch DXY
dxy = yf.download('DX-Y.NYB', start='2015-01-01', end='2025-02-12')['Close']

# Fetch constituent currencies
currencies = {
    'EURUSD': 'EURUSD=X',
    'USDJPY': 'JPY=X',
    'GBPUSD': 'GBPUSD=X',
    'USDCAD': 'CAD=X',
    'USDSEK': 'SEK=X',
    'USDCHF': 'CHF=X'
}

currency_data = {}
for name, ticker in currencies.items():
    currency_data[name] = yf.download(ticker, start='2015-01-01', end='2025-02-12')['Close']

currency_df = pd.DataFrame(currency_data)
```

**Note**: Yahoo Finance provides same-day or T+1 data; well within 5-day delay limit.

---

## Implementation Complexity Assessment

### Deterministic Components (PCA, Volatility)
- **Complexity**: Low
- **Training time**: <10 seconds
- **Overfitting risk**: None (deterministic)
- **Gate 1**: Automatic pass

### HMM Component
- **Complexity**: Medium
- **Training time**: <1 minute (hmmlearn is efficient)
- **Overfitting risk**: Low (2 states, simple Gaussian emissions)
- **Gate 1**: Should pass easily with train/val overfit ratio <1.5

### Overall Assessment
- **Total training time**: <2 minutes (well within 5-minute deterministic limit for HMM training phase)
- **Kaggle execution time**: <5 minutes including data download and result saving
- **Implementation risk**: Low; all components have mature Python libraries

---

## Expected Gate Performance

### Gate 1: Standalone Quality
- **Overfit ratio**: N/A for PCA/volatility (deterministic); <1.3 for HMM (simple model)
- **No constant output**: Guaranteed; features vary by construction
- **Autocorrelation**: Expected <0.8 for regime_prob and divergence; <0.9 for volatility_zscore
- **Verdict**: **PASS LIKELY**

### Gate 2: Information Gain
- **MI increase**: Expected 10-15% (regime and divergence capture structure absent from raw level)
- **VIF**: Expected 2-5 for all features (orthogonal by design to raw level)
- **Rolling correlation std**: Expected 0.10-0.20 (regime features naturally have time-varying correlation)
- **Verdict**: **PASS LIKELY**

### Gate 3: Ablation Test
- **Direction accuracy**: +0.3% to +0.8% (provides USD context beyond raw level)
- **Sharpe improvement**: +0.03 to +0.10 (depends on meta-model's ability to exploit regime information)
- **MAE improvement**: Uncertain; regime features help direction more than magnitude
- **Verdict**: **UNCERTAIN but PROMISING** - Daily frequency and structural features favor success vs real_rate's interpolated features

---

## Risks & Mitigation Strategies

### Risk 1: HMM State Interpretation Instability
- **Description**: HMM states may not align with intuitive "trending" vs "mean-reverting" labels
- **Mitigation**: Label states by empirical properties (autocorrelation, volatility) rather than assuming semantic meaning
- **Fallback**: If states are non-interpretable, output both state probabilities as separate features

### Risk 2: PCA Divergence Too Stable
- **Description**: PC1 might explain 95%+ of variance consistently, making divergence index near-constant
- **Mitigation**: Use rolling 60-day window PCA to allow time-varying factor structure
- **Monitoring**: Architect must verify divergence index has meaningful variation (std > 0.05)
- **Fallback**: Use alternative divergence metric (std of currency returns) if PCA variance ratio is too stable

### Risk 3: VIF Violation Despite Design
- **Description**: Volatility state might correlate with existing VIX feature
- **Mitigation**: Architect computes VIF matrix including all base features + VIX
- **Corrective action**: If VIF > 10, regress feature on violating base feature, output residuals

### Risk 4: Gate 3 Failure Despite Gate 2 Pass
- **Description**: Real_rate pattern - MI increases but XGBoost can't exploit it
- **Root cause**: Output dimensionality amplified noise for XGBoost
- **Mitigation**: 3-column limit strictly enforced (vs real_rate's 7 columns)
- **Attempt 2 path**: If Gate 3 fails, reduce to 2 columns (drop volatility, keep regime + divergence)

### Risk 5: Currency Pair Data Quality Issues
- **Description**: Yahoo Finance occasional missing data for minor currencies (SEK, CHF)
- **Mitigation**: builder_data must forward-fill gaps <3 days; reject if gaps >3 days
- **Datachecker enforcement**: Explicit check for missing data percentage <1%

---

## Lessons Applied from real_rate Failures

### Lesson 1: Frequency Mismatch (CRITICAL)
- **Real_rate error**: Monthly-to-daily interpolation created artificial smoothness
- **DXY advantage**: All data daily; NO interpolation
- **Impact**: Eliminates root cause of all 5 real_rate failures

### Lesson 2: Output Dimensionality
- **Real_rate error**: Attempt 5 (7 columns) worse than Attempt 4 (2 columns)
- **DXY design**: 3 columns maximum, strictly enforced
- **Impact**: Reduces XGBoost noise overfitting surface

### Lesson 3: Gate 2 ≠ Gate 3
- **Real_rate pattern**: All attempts passed Gate 2, all failed Gate 3
- **DXY design**: Structural features (regime, divergence) easier for tree-based models to exploit than interpolated time-series features
- **Impact**: Higher Gate 3 success probability

### Lesson 4: VIF Computed Correctly
- **Real_rate issue**: Base features had existing multicollinearity
- **DXY enforcement**: Architect must compute VIF using true R² regression, include all base features
- **Impact**: Prevents false passes

### Lesson 5: Deterministic Preferred
- **Real_rate lesson**: Deterministic approaches can pass Gate 1 trivially
- **DXY design**: PCA and volatility are deterministic; only HMM has training variance
- **Impact**: Reduces overfitting risk

---

## Alternative Approaches (If Attempt 1 Fails)

### Alternative A: Pure Deterministic (Simpler)
- Remove HMM component
- Output: PCA divergence + DXY momentum (20-day return) + volatility state
- Pro: No overfitting possible
- Con: Less sophisticated regime detection

### Alternative B: Markov-Switching Model (More Complex)
- Replace HMM with Markov-Switching Autoregressive model
- Explicitly models momentum vs mean-reversion regimes
- Pro: Theoretically superior regime characterization
- Con: More complex estimation, longer training time

### Alternative C: TAR Model
- Use threshold autoregressive model with optimized threshold
- Pro: Parsimonious, interpretable threshold
- Con: Threshold estimation challenging, may be unstable

### Alternative D: Increase Multi-Country Coverage
- Expand beyond DXY to G10 currencies
- Build global FX regime detector
- Pro: Larger sample size (~50,000 observations)
- Con: Complexity increases; defer to Attempt 3+ if needed

---

## Success Hypothesis

**Hypothesis**: DXY regime probability (trending vs mean-reverting) and cross-currency divergence index, computed at daily frequency with no interpolation, will provide the meta-model with USD structural context that distinguishes between:

1. **Scenario A**: DXY rising with all currencies weakening uniformly + trending regime → Strong negative gold signal
2. **Scenario B**: DXY rising but driven only by EUR weakness while JPY/CHF strengthen + mean-reverting regime → Weak or neutral gold signal
3. **Scenario C**: DXY falling with high cross-currency divergence + high volatility → Uncertain gold signal, meta-model needs other features

This structural context, absent from the raw DXY level, will improve direction accuracy by +0.5% to +1.0%, sufficient for Gate 3 pass.

---

## Conclusion & Architect Handoff

### Primary Recommendation
**Implement Approach C (Hybrid)** with 3-column output:
1. `dxy_regime_trend_prob`: HMM probability of trending regime
2. `dxy_cross_currency_divergence`: 1 - PC1_explained_variance (rolling 60-day)
3. `dxy_volatility_zscore`: Z-score of 20-day realized volatility

### Data Acquisition Ready
- All tickers identified with Yahoo Finance API access
- No new API keys required
- Expected ~2,500 daily observations matching base_features date range

### Implementation Risk: LOW
- Mature libraries: `hmmlearn`, `sklearn.decomposition.PCA`, `pandas`
- Total training time <5 minutes
- No paid services or exotic data sources

### Critical Architect Tasks
1. Verify VIF < 10 for all 3 output columns against base features
2. Confirm HMM achieves overfit ratio <1.5 (train vs val)
3. Validate divergence index has meaningful variation (std >0.05)
4. Design Optuna search space for HMM hyperparameters (n_components=2 or 3, covariance_type)

### Expected Outcome
- **Gate 1**: Pass (high confidence)
- **Gate 2**: Pass (high confidence)
- **Gate 3**: Pass (moderate confidence - better than real_rate due to daily frequency)

### Fallback Plan
If Gate 3 fails: Attempt 2 reduces to 2 columns (regime + divergence only), removes volatility to minimize XGBoost noise surface.

---

## Sources

### Question 1: FX Regime Detection
- [Step-by-Step Python Guide for Regime-Specific Trading Using HMM and Random Forest](https://blog.quantinsti.com/regime-adaptive-trading-python/)
- [Hidden Markov Models for Forex Trends Prediction | IEEE Xplore](https://ieeexplore.ieee.org/document/6847408/)
- [Forex Market Regime Estimation via Hidden Markov Models](https://dspace.cuni.cz/bitstream/handle/20.500.11956/200417/120507159.pdf?sequence=1&isAllowed=y)
- [Introduction to Hidden Markov Models (HMM) for Traders: Python Tutorial](https://www.marketcalls.in/python/introduction-to-hidden-markov-models-hmm-for-traders-python-tutorial.html)
- [Market Regime Detection using Hidden Markov Models in QSTrader | QuantStart](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [SETAR (model) - Wikipedia](https://en.wikipedia.org/wiki/SETAR_(model))
- [Threshold Autoregressive Models — beyond ARIMA + R Code | Medium](https://medium.com/data-science/threshold-autoregressive-models-beyond-arima-r-code-6af3331e2755)

### Question 2: Cross-Currency Divergence
- [Using a principal component analysis for multi-currencies-trading in the foreign exchange market](https://www.researchgate.net/publication/281424639_Using_a_principal_component_analysis_for_multi-currencies-trading_in_the_foreign_exchange_market)
- [How Principal Component Analysis Can Improve a Long/Short Currency Strategy](https://www.linkedin.com/pulse/how-principal-component-analysis-can-improve-longshort-charles-ellis)
- [Currency Hedging and Principal Component Analysis | Dean Markwick](https://dm13450.github.io/2024/04/25/Currency-Hedging-and-Principal-Component-Analysis.html)
- [Forecasting exchange rates using principal components - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1042443118304517)

### Question 3: DXY-Gold Correlation Regime Shifts
- [2026 Market Outlook: DXY Weakness, Gold's New Floor, and Bitcoin Consolidation | FXEmpire](https://www.fxempire.com/forecasts/article/2026-market-outlook-dxy-weakness-golds-new-floor-and-bitcoin-consolidation-1579296)
- [Gold 2026 Outlook: Can the structural bull cycle continue to $5,000?](https://www.ssga.com/us/en/intermediary/insights/gold-2026-outlook-can-the-structural-bull-cycle-continue-to-5000)
- [The 2008 global financial crisis and COVID-19 pandemic: How safe are the safe haven assets? - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9329144/)
- [Revisiting the safe haven role of Gold across time and frequencies during the COVID-19 pandemic - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8940724/)
- [Are safe haven assets really safe during the 2008 global financial crisis and COVID-19 pandemic? - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8575456/)

### Question 4: Lookback Windows
- [Optimal Lookback Period For Momentum Strategies | Seeking Alpha](https://seekingalpha.com/article/4240540-optimal-lookback-period-for-momentum-strategies)
- [Volatility-Adjusted Time Series Momentum — A Smarter Way to Trade | Medium](https://pyquantlab.medium.com/volatility-adjusted-time-series-momentum-a-smarter-way-to-trade-bcc1c63a06bf)
- [The Evolution of Optimal Lookback Horizon - ReSolve Asset Management](https://investresolve.com/half-life-of-optimal-lookback-horizon/)
- [Time Series Momentum Effect - Quantpedia](https://quantpedia.com/strategies/time-series-momentum-effect/)

### Question 5: Seasonality
- [US Dollar Index :: SeasonalCharts.de](https://www.seasonalcharts.com/classic_usdindex.html)
- [US Dollar Index Futures Seasonal Chart | Equity Clock](https://charts.equityclock.com/us-dollar-index-futures-seasonal-chart)
- [US Dollar Forecast 2026 | DXY Outlook, Key Levels & Rate Risks](https://cambridgecurrencies.com/usd-forecast-2026/)

### Question 6: FX Volatility-Gold Relationship
- [Gold Market Commentary: Precious Metal Thunder | World Gold Council](https://www.gold.org/goldhub/research/gold-market-commentary-december-2025)
- [Gold Price Volatility | History Chart | World Gold Council](https://www.gold.org/goldhub/data/gold-price-volatility)
- [United States - CBOE Gold ETF Volatility](https://tradingeconomics.com/united-states/cboe-gold-etf-volatility-index-fed-data.html)

### Question 7: VIF and Multicollinearity
- [5 Expert VIF Strategies: Reducing Multicollinearity in Regression Models](https://www.numberanalytics.com/blog/5-expert-vif-strategies-reducing-multicollinearity-regression-models)
- [Variance Inflation Factor: How to Detect Multicollinearity | DataCamp](https://www.datacamp.com/tutorial/variance-inflation-factor)
- [Multicollinearity in Regression Analysis: Problems, Detection, and Solutions - Statistics By Jim](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/)
- [Understanding Multicollinearity: Detection and Solutions Using Variance Inflation Factor (VIF) | Medium](https://medium.com/@prathik.codes/understanding-multicollinearity-detection-and-solutions-using-variance-inflation-factor-vif-2673b8bba8a3)
