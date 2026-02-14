# Research Report: Technical (Attempt 1)

## Research Questions

This report addresses the following research questions from current_task.json:

1. **Momentum regime detection**: HMM vs Hurst exponent vs ADX
2. **Mean-reversion indicators**: RSI vs z-score vs Stochastic (VIF with base features)
3. **Volatility compression**: Bollinger Band width vs ATR vs Parkinson volatility
4. **Optimal lookback windows** for gold (literature/empirical)
5. **Volume integration**: VIF risk with existing volume features
6. **Interaction with VIX submodel**: complementary or redundant
7. **GC=F futures contract roll handling**

---

## Executive Summary

Gold technical analysis benefits from a multi-method approach combining probabilistic regime detection with deterministic statistical features. The research strongly supports:

1. **HMM for regime detection** - proven success in VIX/DXY submodels, clean separation of trending/ranging states
2. **Rolling z-score over raw RSI** - lower VIF with price levels, captures overbought/oversold dynamics
3. **Garman-Klass volatility estimator** - 7.4x more efficient than close-to-close, leverages unique OHLC advantage
4. **6-12 month lookback for momentum**, 14-20 days for mean-reversion, 20-60 days for volatility
5. **Avoid volume-based features** - high VIF risk with existing etf_flow_gld_volume base features
6. **Technical and VIX are complementary** - VIX captures external fear, technicals capture gold's internal dynamics
7. **Use GC=F with ratio-adjusted continuous contract** or returns-based features to avoid roll artifacts

---

## Question 1: Momentum Regime Detection - HMM vs Hurst Exponent vs ADX

### 1.1 Hidden Markov Model (HMM)

**Strengths:**
- Proven success in VIX submodel (DA +0.96%, passed Gate 3 on attempt 1)
- Estimates probability of regime continuation vs regime shift
- Smooth probabilistic output [0,1] ideal for XGBoost
- Captures latent states in noisy time series
- Recent 2025 research shows strong performance with gold specifically: "optimizing a Gold-SPY portfolio using HMM achieved Sharpe Ratio of 0.823 with 19.787% CAGR"

**Application to Gold:**
HMM can identify 2-3 states in gold daily returns:
- **State 1**: Trending up (positive mean, moderate variance)
- **State 2**: Range-bound (near-zero mean, low variance)
- **State 3**: Trending down (negative mean, moderate variance)

**Implementation:**
```python
from hmmlearn.hmm import GaussianHMM
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(gold_returns.reshape(-1, 1))
regime_probs = model.predict_proba(gold_returns.reshape(-1, 1))
# Output: tech_trend_regime_prob = prob(State 1 or State 3) = 1 - prob(State 2)
```

**Limitation:**
Gold returns are closer to random walk than VIX (which has clearer regime structure). HMM may produce less distinct regimes than VIX case, but empirical evidence from gold-SPY portfolio suggests effectiveness.

### 1.2 Hurst Exponent

**Strengths:**
- **Direct measure of trending vs mean-reversion**: H > 0.5 = trending (persistence), H < 0.5 = mean-reverting (anti-persistence), H ≈ 0.5 = random walk
- Specific gold research available: "A Comparison of Fractal Dimension Algorithms by Hurst Exponent using Gold Price Time Series" demonstrates empirical application
- Sliding window method allows regime tracking over time
- Deterministic (no model fitting required)

**Calculation Methods:**
1. **R/S Analysis** (Rescaled Range): Classic method, robust
2. **DFA** (Detrended Fluctuation Analysis): Better for non-stationary series

**Application to Gold:**
```python
# Rolling 60-day Hurst exponent
def hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]

tech_fractal_dimension = gold_close.rolling(60).apply(hurst_exponent)
```

**Limitation:**
- Requires minimum 60-100 observations for stable estimates
- Noisy on short windows (< 100 days)
- Single scalar value may be less informative than HMM's probabilistic regime

### 1.3 ADX (Average Directional Index)

**Strengths:**
- Industry-standard trend strength measure
- Well-understood by market participants (self-fulfilling effect)
- ADX > 25 = trending, ADX < 20 = ranging
- Default 14-period setting widely used

**Application to Gold:**
```python
# Standard ADX calculation
adx_14 = talib.ADX(high, low, close, timeperiod=14)
# Normalize to [0,1]
tech_trend_strength = adx_14 / 100
```

**Limitations:**
- **Derivative of price levels** - potential VIF with raw GLD close/high/low
- Lagging indicator (14-period smoothing)
- Less novel than HMM (baseline XGBoost may already capture similar patterns from OHLC)

### 1.4 Recommendation: HMM (Priority 1) + Hurst Exponent (Priority 2)

**Recommended Approach:**
1. **Primary: HMM on gold returns** - follows VIX/DXY success pattern, probabilistic regime output, low VIF
2. **Alternative: Hurst exponent** - if HMM produces weak regimes, Hurst provides direct trend/mean-reversion measure
3. **Avoid: ADX** - too correlated with price levels, less novel information

**Rationale:**
- HMM proved successful in VIX submodel with similar characteristics (regime detection on returns)
- Gold-specific HMM research shows strong empirical performance
- Hurst exponent offers pure statistical measure without ML overfitting risk
- ADX is derivative of OHLC already in base features (redundancy risk)

---

## Question 2: Mean-Reversion Indicators - RSI vs Z-Score vs Stochastic

### 2.1 RSI (Relative Strength Index)

**Strengths:**
- Industry standard (14-period default)
- Clear overbought (>70) / oversold (<30) thresholds
- Gold-specific insight: "RSI values frequently remain above 70 for weeks during strong gold rallies, signaling powerful uptrends rather than exhaustion"

**VIF Risk:**
- Current_task.json notes: "measured correlation between GLD close and RSI is ~0.025"
- **Low correlation with price level** (good for VIF)
- However, RSI is derivative of returns, which may overlap with momentum regime features

**Application:**
```python
rsi_14 = talib.RSI(close, timeperiod=14)
# Normalize to [-3, +3] z-score
tech_rsi_z = (rsi_14 - 50) / 15  # approximate normalization
```

**Gold-Specific Consideration:**
Gold exhibits "strength zone" RSI 70-80 during major rallies. Standard RSI >70 = overbought may be misleading for gold. Better to use z-score of RSI deviation from 50.

### 2.2 Rolling Z-Score of Returns

**Strengths:**
- **Direct measure of overbought/oversold relative to recent dynamics**
- Successfully used in VIX submodel (vix_mean_reversion_z)
- Lower VIF than price-level-dependent indicators
- Captures distance from dynamic equilibrium

**Application:**
```python
# 20-day rolling z-score of returns
returns = close.pct_change()
mean = returns.rolling(20).mean()
std = returns.rolling(20).std()
tech_mean_reversion_z = (returns - mean) / std
```

**Advantages:**
- Returns-based, not level-based (low VIF with GLD close)
- Adaptive to changing volatility regimes
- Consistent with VIX submodel's successful approach

### 2.3 Stochastic Oscillator

**Strengths:**
- Measures closing price relative to high-low range
- %K > 80 = overbought, %K < 20 = oversold
- Gold-specific: "Stochastic readings consistently above 80 during strong uptrends more reliably indicate potential short-term exhaustion"

**VIF Risk:**
- **High correlation with price levels** - Stochastic uses high/low/close directly
- Likely moderate-to-high VIF with technical_gld_high, technical_gld_low, technical_gld_close in base features

**Application:**
```python
slowk, slowd = talib.STOCH(high, low, close,
                            fastk_period=14,
                            slowk_period=3,
                            slowd_period=3)
tech_stochastic_z = (slowk - 50) / 25
```

### 2.4 Recommendation: Rolling Z-Score (Priority 1) + RSI Z-Score (Priority 2)

**Recommended Approach:**
1. **Primary: Rolling z-score of returns (20-day window)** - follows VIX success, lowest VIF, adaptive
2. **Alternative: RSI z-score** - if rolling z-score is too volatile, RSI provides smoothed alternative
3. **Avoid: Stochastic** - high VIF risk with OHLC base features

**Implementation:**
```python
# 20-day rolling z-score (mean-reversion position)
returns = gold_close.pct_change()
tech_mean_reversion_z = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
# Clip extreme values to [-4, +4]
tech_mean_reversion_z = tech_mean_reversion_z.clip(-4, 4)
```

**Rationale:**
- VIX submodel used rolling z-score successfully (low VIF, high information gain)
- Returns-based features avoid price level collinearity
- Adaptive to gold's changing volatility (gold volatility clusters strongly)

---

## Question 3: Volatility Compression - Bollinger Band Width vs ATR vs Parkinson Volatility

### 3.1 Bollinger Band Width (BBW)

**Strengths:**
- **Direct measure of volatility compression**: BBW = (upper_band - lower_band) / middle_band
- "Squeeze" patterns precede breakouts: "unusually low BBW values suggest consolidation often followed by a breakout"
- Widely used by gold traders (self-fulfilling effect)

**Application:**
```python
upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
bbw = (upper - lower) / middle
# Z-score to capture compression relative to history
tech_volatility_regime = (bbw - bbw.rolling(60).mean()) / bbw.rolling(60).std()
```

**Evidence:**
"Looking for a combination of narrow Bollinger Bands and low ATR is most likely to produce sustained moves when the market breaks the range"

### 3.2 Average True Range (ATR)

**Strengths:**
- Measures absolute price volatility (accounts for gaps)
- Commonly used for stop-loss and position sizing
- "Traders can use Bollinger Bands to spot potential breakouts and ATR to confirm the strength of the move"

**Application:**
```python
atr_14 = talib.ATR(high, low, close, timeperiod=14)
# Ratio of short-term to long-term ATR
tech_atr_ratio = atr_14 / atr_14.rolling(60).mean()
```

**Limitation:**
- Absolute measure (not normalized by price level)
- Less direct measure of "compression" than BBW

### 3.3 Parkinson Volatility Estimator

**Strengths:**
- **Leverages OHLC data** (unique advantage of technical submodel)
- **5x more efficient than close-to-close volatility** (uses intraday range)
- "Garman-Klass estimator based on OHLC is 7.4 times more efficient than close-to-close estimator"
- Provides **orthogonal signal to VIX/DXY** which only use close prices

**Calculation:**
```python
# Parkinson volatility estimator
parkinson_vol = np.sqrt((1/(4*np.log(2))) * np.log(high/low)**2)

# Garman-Klass (even better - uses all OHLC)
gk_vol = np.sqrt(0.5 * np.log(high/low)**2 -
                 (2*np.log(2)-1) * np.log(close/open)**2)

# Z-score for regime detection
tech_volatility_regime = (gk_vol - gk_vol.rolling(60).mean()) / gk_vol.rolling(60).std()
```

**Advantages:**
- **Differentiates from VIX/DXY submodels** - they use close-only data
- More efficient estimation (less noise)
- Captures intraday dynamics (overnight gaps, intraday reversals)

**Note:**
"Parkinson volatility estimator tends to underestimate volatility since it does not take into account opening jumps" - Garman-Klass addresses this by including open/close.

### 3.4 Recommendation: Garman-Klass Volatility Z-Score (Priority 1)

**Recommended Approach:**
1. **Primary: Garman-Klass volatility z-score** - most efficient, leverages unique OHLC data, orthogonal to VIX/DXY
2. **Alternative: Bollinger Band Width z-score** - if Garman-Klass is too noisy, BBW provides smoother measure
3. **Secondary: ATR ratio** - complementary confirmation signal (not needed if Garman-Klass works)

**Implementation:**
```python
# Garman-Klass volatility estimator (daily)
high = data['technical_gld_high']
low = data['technical_gld_low']
open_ = data['technical_gld_open']
close = data['technical_gld_close']

gk_vol = np.sqrt(
    0.5 * np.log(high/low)**2 -
    (2*np.log(2)-1) * np.log(close/open_)**2
)

# 60-day z-score for regime detection
tech_volatility_regime = (gk_vol - gk_vol.rolling(60).mean()) / gk_vol.rolling(60).std()
```

**Rationale:**
- Garman-Klass is **7.4x more efficient** than methods available to VIX/DXY submodels
- Captures compression/expansion directly from intraday range
- Low VIF with VIX/DXY volatility measures (different data source)
- Proven in academic literature for volatility forecasting

---

## Question 4: Optimal Lookback Windows for Gold

### 4.1 Empirical Evidence from Literature

**Momentum Strategies:**
- **Classic 12-month lookback**: Historically strong (1988-2008), but "performed very poorly in recent years"
- **Modern 3-6 month lookback**: Better performance (2008-2019)
- **Medium-term sweet spot**: "6-12 month trend horizon achieves low turnover and reasonably fast response to market regime shifts"

**Mean-Reversion Strategies:**
- **Shorter windows**: "< 1-3 months often relate to mean-reversion effects"
- **Empirical gold standard**: 14-20 days for oscillators (RSI-14, Stochastic-14)

**Volatility Measures:**
- **Short-term**: 10-20 days for recent volatility
- **Long-term reference**: 60 days for historical comparison
- **Ratio approach**: short-term / long-term captures regime shifts

### 4.2 Gold-Specific Parameters from Search Results

**RSI Configuration:**
- "14-period RSI with 30 oversold / 70 overbought" - industry standard
- Gold rallies: "RSI frequently remains above 70 for weeks"

**Moving Averages:**
- "10/50-day MA crossover for short-term signals"
- "50/200-day MA crossover (golden cross) for major trend"
- "21/50/200-day MA system for trend turning points"

**ADX:**
- "14-period default setting works well for most traders"

### 4.3 Recommended Lookback Windows for Technical Submodel

| Feature | Window | Rationale |
|---------|--------|-----------|
| **HMM Regime Detection** | 60-120 days | Medium-term regimes, stable estimation |
| **Hurst Exponent** | 60 days | Minimum for stable fractal dimension |
| **Mean-Reversion Z-Score** | 20 days | Standard oscillator period, captures short-term extremes |
| **Volatility Regime (GK)** | 20-day vol, 60-day reference | Short-term volatility vs long-term baseline |

**Justification:**
- **60-120 days for HMM**: Captures medium-term regime persistence (gold's momentum half-life)
- **20 days for mean-reversion**: Aligns with gold's short-term overbought/oversold cycles
- **20/60 days for volatility**: Short-term compression relative to intermediate-term baseline

### 4.4 Gold Momentum Persistence

**Empirical Insight:**
"Gold exhibits stronger momentum than most assets. Trend-following strategies have historically worked well on gold due to macroeconomic regime persistence (e.g., QE eras, inflation fears). Momentum is typically measured over 10-60 day windows."

**Implication:**
- Use 60-day minimum for regime detection (captures multi-week trends)
- Avoid very long lookbacks (>120 days) - gold's regimes shift with macro catalysts
- 20-day mean-reversion captures counter-trend exhaustion within larger trends

---

## Question 5: Volume Integration - VIF Risk

### 5.1 Existing Volume Features in Base Features

**From base_features.csv:**
- `technical_gld_volume` - raw daily volume (feature importance: 5.2%, rank 10th)
- `etf_flow_gld_volume` - same data, different category (feature importance: 5.2%, rank 9th)
- `etf_flow_volume_ma20` - 20-day moving average of volume

**Concern:**
These are **duplicate/highly correlated features** in baseline. Adding more volume-based features will exacerbate VIF issues.

### 5.2 Volume-Based Technical Indicators

**Candidates:**
1. **On-Balance Volume (OBV)**: Cumulative volume with direction
2. **Accumulation/Distribution Line**: Volume-weighted price accumulation
3. **Volume-Relative-to-MA**: Volume / MA(volume)

**VIF Risk Analysis:**
- OBV and A/D Line are **cumulative** - will have near-perfect correlation with price levels (VIF >> 10)
- Volume-relative-to-MA is **highly correlated** with raw volume (VIF >> 5)
- "Half of all retail investors trading technical indicators have poor overall results due to using multicollinear analysis"

### 5.3 Recommendation: Exclude Volume-Based Features

**Rationale:**
1. **High VIF risk**: Volume features already present in base_features
2. **Redundancy**: etf_flow_gld_volume and technical_gld_volume are duplicates
3. **Low marginal information**: Volume confirmation is useful in discretionary trading but adds noise in ML context
4. **3-feature constraint**: Budget limited to 3 outputs, volume features are lower priority than regime/position/volatility

**Alternative:**
If volume integration is required, use **implicit volume weighting** in existing features:
- Garman-Klass volatility already incorporates intraday range (proxy for volume)
- HMM on returns already captures volume-driven regime shifts

---

## Question 6: Interaction with VIX Submodel - Complementary or Redundant?

### 6.1 VIX Submodel Outputs (Already in Pipeline)

From successful VIX attempt 1:
- `vix_regime_probability` - HMM regime on VIX levels
- `vix_mean_reversion_z` - Z-score of VIX deviation from mean
- `vix_persistence` - Autocorrelation of VIX changes

### 6.2 Correlation Analysis

**Expected Correlations:**

| VIX Feature | Technical Feature | Expected Correlation | Risk |
|-------------|-------------------|---------------------|------|
| vix_regime_probability | tech_trend_regime_prob | 0.1-0.3 | **Low** - VIX regime = external fear, Gold regime = internal momentum |
| vix_mean_reversion_z | tech_mean_reversion_z | 0.1-0.2 | **Low** - Different assets, different dynamics |
| vix_persistence | tech_volatility_regime | 0.2-0.4 | **Moderate** - Both volatility measures, but VIX = external, Gold = internal |

**Empirical Evidence:**
"When VIX spikes, the market is likely in a risk-off regime, and that's when capital seeks refuge and gold demand rises. However, when the VIX rises, gold tends to benefit."

**Key Insight:**
- **VIX and gold can move together**, but their **technical regimes are independent**
- VIX = external fear environment (S&P volatility)
- Gold technicals = internal supply/demand dynamics reflected in price action
- **Complementary, not redundant**

### 6.3 Gold-VIX Interaction Evidence

**Regime-Dependent Correlation:**
"Does gold momentum persistence increase during high-VIX periods? Does the mean-reversion speed of gold change across volatility regimes?"

**Research Finding:**
"Regime-dependency and time-varying correlations between oil, stock, gold, VIX, and exchange rates should be kept in consideration"

**Implication:**
- Gold's technical regime may **interact** with VIX regime (e.g., stronger gold momentum during VIX spikes)
- This is **beneficial** - creates nonlinear interaction terms for XGBoost to exploit
- Not redundant - provides **orthogonal information**

### 6.4 Differentiation Strategy

**Technical Submodel Unique Advantages:**
1. **OHLC data** - Garman-Klass volatility uses high/low/open/close (VIX only uses close)
2. **Self-referential** - Gold's own regime, not external context
3. **Trader behavior** - "Technical patterns in gold are self-fulfilling because many traders use them"

**VIX Submodel Covers:**
1. External fear/greed environment
2. Equity market volatility spillover
3. Risk-off/risk-on regime

**Conclusion:**
Technical and VIX submodels are **highly complementary**. Minimal redundancy risk.

---

## Question 7: GC=F Futures Contract Roll Handling

### 7.1 Contract Roll Challenge

**Issue:**
GC=F is a continuous contract formed by stitching individual monthly contracts. Roll dates create artificial price jumps.

**From CME:**
- Gold futures contract: 100 troy ounces
- Active months: Feb, Apr, Jun, Aug, Oct, Dec (6 contracts/year)
- Roll typically occurs 1-2 weeks before expiration

### 7.2 Roll Adjustment Methods

**From research:**
"There are multiple approaches to creating continuous contracts:"

1. **Difference (Back-Adjusted)**: Adjust historical prices by the difference at roll
   - **Pro**: Maintains returns continuity
   - **Con**: Historical prices become "artificial" (negative prices possible)

2. **Ratio (Proportional)**: Adjust by the ratio at roll
   - **Pro**: Maintains percentage returns
   - **Con**: Compounds adjustment errors over time

3. **No Adjustment**: Use raw continuous contract
   - **Pro**: Real prices
   - **Con**: Artificial jumps at rolls

**Critical Warning:**
"If you analyze an underlying asset and use a continuous contract adjusted by difference, you shouldn't enter conditions based on percentage calculations, because these percentages, going back in time, would be calculated on a price that had never been fixed"

### 7.3 Impact on Technical Indicators

**Indicators Affected by Rolls:**

| Indicator | Impact | Mitigation |
|-----------|--------|------------|
| HMM on returns | **Minimal** - returns-based | Use returns, not levels |
| Rolling z-score | **Minimal** - returns-based | Use returns, not levels |
| Garman-Klass volatility | **Moderate** - uses OHLC levels | Use ratio-adjusted contract OR compute on returns |
| Price-level features | **High** - levels distorted | Avoid entirely |

### 7.4 Recommended Solutions

**Option A: Use GLD Instead of GC=F**
- **Advantage**: No roll artifacts, continuous price series
- **Disadvantage**: ETF premium/discount to NAV (~0.40% expense ratio), less liquid OHLC data

**Option B: Returns-Based Features Only**
- **Advantage**: Roll-invariant
- **Disadvantage**: Cannot use price levels (not an issue for this submodel)

**Option C: Ratio-Adjusted GC=F + Returns**
- **Advantage**: Best of both - clean continuous series, returns-based features
- **Implementation**: Yahoo Finance GC=F is typically ratio-adjusted

### 7.5 Recommended Approach: GC=F with Returns-Based Features

**Implementation:**
1. Use Yahoo Finance GC=F (ratio-adjusted continuous contract)
2. **Compute all features on returns, not levels:**
   - HMM on daily returns ✓
   - Rolling z-score of returns ✓
   - Garman-Klass volatility from log(high/low) and log(close/open) ✓ (ratio-invariant)

**Validation:**
```python
# Check for roll artifacts
gc_returns = gc_close.pct_change()
# Detect outliers (>5 sigma) - likely roll jumps
outliers = gc_returns[np.abs((gc_returns - gc_returns.mean()) / gc_returns.std()) > 5]
print(f"Potential roll artifacts: {len(outliers)}")
# If >10, switch to GLD
```

**Fallback:**
If roll artifacts are severe (>10 outliers in 2015-2025), switch to GLD as primary source and use GC=F only for validation.

---

## Recommended Methodology (Priority Ranking)

### Approach 1: HMM + Z-Score + Garman-Klass (RECOMMENDED)

**Features:**
1. `tech_trend_regime_prob` - HMM on gold returns (60-day training window, 3 states)
2. `tech_mean_reversion_z` - Rolling 20-day z-score of returns
3. `tech_volatility_regime` - Garman-Klass volatility z-score (20-day vol, 60-day baseline)

**Strengths:**
- Follows VIX/DXY success pattern (HMM + statistical features)
- All returns-based (roll-invariant, low VIF)
- Leverages unique OHLC data advantage (Garman-Klass)
- 3 orthogonal dimensions: regime, position, volatility

**Expected Performance:**
- Gate 2 MI increase: >10% (VIX achieved +14.7%)
- Gate 3 DA improvement: >+0.5% (VIX achieved +0.96%)

**Implementation Difficulty:** Moderate
**Data Requirements:** GC=F OHLC daily (2015-2025)

---

### Approach 2: Hurst + RSI Z-Score + BBW (ALTERNATIVE)

**Features:**
1. `tech_hurst_exponent` - Rolling 60-day Hurst exponent (trending vs mean-reversion)
2. `tech_rsi_z` - RSI-14 normalized to z-score
3. `tech_bbw_regime` - Bollinger Band Width z-score (volatility compression)

**Strengths:**
- All deterministic (no ML fitting, zero overfitting risk)
- Hurst directly measures trend/mean-reversion dichotomy
- Industry-standard indicators (RSI, BBW) - self-fulfilling effect

**Weaknesses:**
- Hurst noisy on short windows
- Less novel than HMM (baseline may capture similar patterns)

**Use Case:** Fallback if HMM produces weak regimes

---

### Approach 3: HMM + Hurst + Parkinson (HYBRID)

**Features:**
1. `tech_regime_probability` - HMM on returns (trending probability)
2. `tech_fractal_dimension` - Hurst exponent (fractal structure)
3. `tech_volatility_z` - Parkinson volatility z-score

**Strengths:**
- Best of both worlds: probabilistic + deterministic
- Hurst complements HMM (provides fractal dimension context)
- Parkinson simpler than Garman-Klass

**Weaknesses:**
- No explicit mean-reversion position feature
- Hurst and HMM may be partially redundant

**Use Case:** If z-score mean-reversion is too volatile

---

## Data Sources and Availability

### Primary Data: GC=F (Gold Futures)

**Source:** Yahoo Finance
**Ticker:** `GC=F`
**Frequency:** Daily
**Fields:** Open, High, Low, Close, Volume
**Date Range:** 2015-01-30 to 2025-02-12 (2523 days expected)

**Acquisition Code:**
```python
import yfinance as yf
gc = yf.download('GC=F', start='2015-01-30', end='2025-02-13')
# Verify no excessive gaps
print(f"Expected: 2523 days, Actual: {len(gc)}")
# Check for roll artifacts
returns = gc['Close'].pct_change()
outliers = returns[np.abs(returns) > 0.05]  # >5% daily moves
print(f"Potential roll artifacts: {len(outliers)}")
```

**Roll Artifact Handling:**
- If <10 outliers: Use GC=F with returns-based features
- If ≥10 outliers: Switch to GLD as primary source

### Backup Data: GLD (SPDR Gold ETF)

**Source:** Yahoo Finance
**Ticker:** `GLD`
**Advantage:** No roll artifacts, already in base_features
**Disadvantage:** ETF premium/discount, expense ratio drag

**When to Use:**
- GC=F has excessive roll artifacts
- OHLC data from GC=F is incomplete
- Validation/cross-check of GC=F features

### Data Quality Checks

1. **Date alignment** with base_features.csv (2015-01-30 to 2025-02-12)
2. **No missing OHLC** (gold trades daily, minimal holidays)
3. **Roll artifact detection** (>5% single-day moves)
4. **VIF calculation** against all base features before proceeding

---

## Expected Challenges and Mitigations

### Challenge 1: HMM Regime Separation on Gold Returns

**Issue:** Gold returns are closer to random walk than VIX (VIX has clearer volatility regimes)

**Mitigation:**
- Use 3 states instead of 2 (up-trend, range, down-trend)
- Incorporate volume as secondary feature (if VIF permits)
- Validate regime stability: compare posterior probabilities across train/val/test

**Fallback:** Switch to Hurst exponent if HMM regimes are unstable

### Challenge 2: VIF with OHLC Base Features

**Issue:** technical_gld_open/high/low/close already in base features

**Mitigation:**
- Use **returns-based** features exclusively (not price levels)
- Garman-Klass uses **ratios** (log(high/low)), not levels
- Validate VIF < 10 against all base features before Gate 2

**Acceptance Criteria:**
- tech_trend_regime_prob VIF < 5 (regime is orthogonal to level)
- tech_mean_reversion_z VIF < 3 (z-score is orthogonal to level)
- tech_volatility_regime VIF < 5 (volatility is orthogonal to level)

### Challenge 3: Contract Roll Artifacts in GC=F

**Issue:** Monthly futures rolls create price discontinuities

**Mitigation:**
- Verify Yahoo Finance GC=F uses ratio-adjustment (standard)
- Use returns-based indicators (roll-invariant)
- Garman-Klass uses log ratios (roll-invariant)
- Validate <10 outliers (>5% daily moves) in 2015-2025 data

**Fallback:** Use GLD if GC=F has >10 roll artifacts

### Challenge 4: Overlapping Information with VIX/DXY Submodels

**Issue:** All three submodels have volatility-related features

**Mitigation:**
- **VIX**: External volatility (S&P fear)
- **DXY**: Dollar volatility (currency risk)
- **Technical**: Gold-specific volatility (internal dynamics)
- Use Garman-Klass (OHLC-based) to differentiate from close-only VIX/DXY
- Validate correlation matrix: tech_volatility_regime vs vix_persistence <0.4

---

## Implementation Roadmap

### Step 1: Data Acquisition
- Fetch GC=F OHLC daily 2015-01-30 to 2025-02-13
- Validate date alignment with base_features.csv
- Check for roll artifacts (outlier detection)
- Fallback to GLD if needed

### Step 2: Feature Engineering
- HMM on returns (60-day training window, 3 states) → tech_trend_regime_prob
- Rolling 20-day z-score of returns → tech_mean_reversion_z
- Garman-Klass volatility (20-day vol, 60-day baseline) → tech_volatility_regime

### Step 3: VIF Validation
- Compute VIF against all base features
- Ensure VIF < 10 for all three outputs
- If VIF > 10, adjust feature construction (e.g., longer rolling window, different normalization)

### Step 4: Model Architecture (Architect Phase)
- HMM: 3-state Gaussian HMM with full covariance
- Deterministic: NumPy/Pandas operations (no training)
- Output format: (2523, 3) DataFrame with Date index

### Step 5: Kaggle Training Script
- Self-contained data fetching (yfinance)
- Feature engineering in single script
- Save outputs: submodel_output.csv (2523, 3)
- Training metrics: N/A (deterministic submodel)

### Step 6: Evaluation (Gate 1/2/3)
- Gate 1: No NaN (after 120-day warmup), no constant output, autocorr <0.99
- Gate 2: MI increase >5%, VIF <10, rolling corr std <0.15
- Gate 3: DA +0.5% OR Sharpe +0.05 OR MAE -0.01%

---

## Theoretical Foundation

### Why Technical Patterns Work for Gold

1. **Self-Fulfilling Prophecy**: "Technical patterns in gold are self-fulfilling because many traders use them. This is a stronger theoretical basis than for random assets -- gold's technical trading community creates real market dynamics."

2. **Behavioral Foundation**: "Gold is one of the most technically traded commodities. Institutional and retail traders heavily use momentum, support/resistance, and mean-reversion signals."

3. **Regime Persistence**: "Gold exhibits stronger momentum than most assets. Trend-following strategies have historically worked well on gold due to macroeconomic regime persistence (e.g., QE eras, inflation fears)."

4. **Volatility Clustering**: "Gold volatility clusters strongly (GARCH effects). Low-volatility periods precede breakouts. Bollinger Band squeeze is a widely used breakout predictor in gold trading."

### What Base Features Miss

Base features provide:
- `technical_gld_open/high/low/close/volume` - **current price levels**

Base features do NOT provide:
- **Momentum regime** - is gold trending or ranging?
- **Mean-reversion position** - is gold overbought or oversold relative to its recent dynamics?
- **Volatility regime** - is gold's volatility expanding or compressing?

**Submodel Value Add:**
The technical submodel extracts **dynamic state information** that XGBoost cannot easily derive from raw levels alone. While XGBoost can compute first-order derivatives (returns), it struggles with:
- Regime detection (requires probabilistic modeling - HMM)
- Fractal dimension (requires specialized algorithms - Hurst)
- Efficient volatility estimation (requires OHLC range - Garman-Klass)

---

## Success Criteria Mapping

### Gate 1: Standalone Quality
- **Overfit ratio <1.5**: N/A (deterministic submodel, no training)
- **No constant output**: Validated by rolling window variance >0
- **Autocorrelation <0.99**: Ensured by using returns-based features (not cumulative)

### Gate 2: Information Gain
- **MI increase >5%**: Expected 10-15% based on VIX precedent
- **VIF <10**: Enforced by returns-based construction
- **Rolling corr std <0.15**: Validated in datachecker phase

### Gate 3: Ablation (Any One)
- **DA +0.5%**: Primary target (VIX achieved +0.96%)
- **Sharpe +0.05**: Secondary (VIX achieved +0.289)
- **MAE -0.01%**: Tertiary

**Hypothesis:**
Gold's technical regime provides **directional information** (momentum continuation vs reversal) that improves classification accuracy. Expected DA improvement: +0.5% to +1.0%.

---

## References and Sources

### Academic and Quantitative Research
- [Intraday Application of Hidden Markov Models - QuantConnect](https://www.quantconnect.com/research/17900/intraday-application-of-hidden-markov-models/)
- [Market Regime Detection using Hidden Markov Models in QSTrader](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Optimizing a Gold-SPY Portfolio Using Hidden Markov Models - QuantConnect](https://www.quantconnect.com/research/18811/optimizing-a-gold-spy-portfolio-using-hidden-markov-models-for-market-downtime/)
- [Detecting trends and mean reversion with the Hurst exponent - Macrosynergy](https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/)
- [A Comparison of Fractal Dimension Algorithms by Hurst Exponent using Gold Price Time Series - ResearchGate](https://www.researchgate.net/publication/329845330_A_Comparison_of_Fractal_Dimension_Algorithms_by_Hurst_Exponent_using_Gold_Price_Time_Series)

### Range-Based Volatility Estimators
- [Range-Based Volatility Estimators: Overview and Examples - Portfolio Optimizer](https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/)
- [Mark B. Garman and Michael J. Klass - Estimation of Security Price Volatilities from Historical Data - CME Group](https://www.cmegroup.com/trading/fx/files/a_estimation_of_security_price.pdf)
- [Properties of range-based volatility estimators - Peter Molnár](http://mmquant.net/wp-content/uploads/2016/09/range_based_estimators.pdf)

### Gold Technical Analysis
- [Technical Analysis of Gold Prices: Key Indicators & Patterns - Discovery Alert](https://discoveryalert.com.au/technical-analysis-gold-prices-2025-indicators-strategies/)
- [How to Build a Profitable Gold Strategy in H2 2025 Using VIX, US Yields, and the Dollar - ACY](https://acy.com/en/market-news/education/gold-strategy-using-vix-yields-dxy-2025-l-s-162409/)
- [Gold Technical Analysis - Barchart](https://www.barchart.com/futures/quotes/GC*0/technical-analysis)

### Mean Reversion and Momentum
- [Mean Reversion Explained - Alchemy Markets](https://alchemymarkets.com/education/strategies/mean-reversion/)
- [Mean Reversion Trading: Understanding Strategies & Indicators - Forex Tester](https://forextester.com/blog/mean-reversion-trading/)
- [Time Series Momentum Effect - Quantpedia](https://quantpedia.com/strategies/time-series-momentum-effect)
- [The Evolution of Optimal Lookback Horizon - ReSolve Asset Management](https://investresolve.com/half-life-of-optimal-lookback-horizon/)

### Volatility Analysis
- [Bollinger Bands Strategy: Squeeze then Surge - LuxAlgo](https://www.luxalgo.com/blog/bollinger-bands-strategy-squeeze-then-surge/)
- [Volatility Analysis: Combining Bollinger Bands and Average True Ranges - Traders Log](https://www.traderslog.com/volatility-analysis)
- [Band Indicators: Volatility in Trading Explained - LuxAlgo](https://www.luxalgo.com/blog/band-indicators-volatility-in-trading-explained/)

### Volume Analysis and Multicollinearity
- [Why Using Different Types of Indicators Is Important To Successful Trading - UseThinkScript](https://usethinkscript.com/threads/why-using-different-types-of-indicators-is-important-to-successful-trading-in-thinkorswim.6114/)
- [Gold price analysis with the help of indicators for volume analysis - ATAS](https://atas.net/volume-analysis/basics-of-volume-analysis/gold-price-analysis/)

### Futures Contract Roll Handling
- [Gold Futures Contract Specs - CME Group](https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html)
- [Continuous Futures Contract Charts - Sierra Chart](https://www.sierrachart.com/index.php?page=doc/ContinuousFuturesContractCharts.html)
- [Futures Trading: Everything You Need To Know about Continuous Contracts - Unger Academy](https://ungeracademy.com/blog/futures-trading-everything-you-need-to-know-about-continuous-contracts)

### Gold-VIX Interaction
- [Do volatility indices diminish gold's appeal as a safe haven - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8463038/)
- [Nonlinear Contagion and Causality Nexus between Oil, Gold, VIX - MDPI](https://www.mdpi.com/2227-7390/10/21/4035)

---

## Appendix: Feature Construction Pseudocode

### Feature 1: tech_trend_regime_prob (HMM)

```python
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# 1. Compute daily returns
returns = gold_close.pct_change().dropna()

# 2. Fit HMM (3 states: down-trend, range, up-trend)
model = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=1000,
    random_state=42
)
model.fit(returns.values.reshape(-1, 1))

# 3. Predict regime probabilities
regime_probs = model.predict_proba(returns.values.reshape(-1, 1))

# 4. Identify states by mean
state_means = model.means_.flatten()
state_order = np.argsort(state_means)  # [down, range, up]

# 5. Compute trending probability (up-trend + down-trend)
prob_down = regime_probs[:, state_order[0]]
prob_range = regime_probs[:, state_order[1]]
prob_up = regime_probs[:, state_order[2]]

tech_trend_regime_prob = prob_up + prob_down  # Probability NOT ranging
```

### Feature 2: tech_mean_reversion_z (Rolling Z-Score)

```python
# 1. Compute daily returns
returns = gold_close.pct_change()

# 2. Rolling 20-day mean and std
mean_20 = returns.rolling(20).mean()
std_20 = returns.rolling(20).std()

# 3. Z-score
tech_mean_reversion_z = (returns - mean_20) / std_20

# 4. Clip extreme values
tech_mean_reversion_z = tech_mean_reversion_z.clip(-4, 4)

# 5. Fill warmup NaNs with 0
tech_mean_reversion_z = tech_mean_reversion_z.fillna(0)
```

### Feature 3: tech_volatility_regime (Garman-Klass)

```python
# 1. Garman-Klass volatility estimator
high = gold_high
low = gold_low
open_ = gold_open
close = gold_close

gk_vol = np.sqrt(
    0.5 * (np.log(high / low) ** 2) -
    (2 * np.log(2) - 1) * (np.log(close / open_) ** 2)
)

# 2. Rolling 20-day Garman-Klass volatility
gk_vol_20 = gk_vol.rolling(20).mean()

# 3. Rolling 60-day baseline
gk_mean_60 = gk_vol_20.rolling(60).mean()
gk_std_60 = gk_vol_20.rolling(60).std()

# 4. Z-score (volatility regime)
tech_volatility_regime = (gk_vol_20 - gk_mean_60) / gk_std_60

# 5. Fill warmup NaNs with 0
tech_volatility_regime = tech_volatility_regime.fillna(0)
```

---

## Final Recommendations Summary

### Recommended Methodology: Approach 1 (HMM + Z-Score + Garman-Klass)

| Feature | Method | Lookback | Output Range | Priority |
|---------|--------|----------|--------------|----------|
| **tech_trend_regime_prob** | HMM on returns (3 states) | 60-120 days | [0, 1] | **P1** |
| **tech_mean_reversion_z** | Rolling z-score of returns | 20 days | [-4, +4] | **P1** |
| **tech_volatility_regime** | Garman-Klass z-score | 20d vol / 60d baseline | [-3, +3] | **P1** |

### Data Source: GC=F (Primary) + GLD (Backup)

- **Primary**: Yahoo Finance GC=F (gold futures continuous contract)
- **Backup**: GLD if GC=F has >10 roll artifacts
- **Frequency**: Daily, 2015-01-30 to 2025-02-13

### Expected Performance

- **Gate 1**: Pass (deterministic, no overfitting)
- **Gate 2**: MI increase 10-15% (based on VIX precedent)
- **Gate 3**: DA improvement +0.5% to +1.0%

### Key Success Factors

1. **Returns-based construction** - low VIF, roll-invariant
2. **Follows VIX/DXY success pattern** - HMM + statistical features
3. **Leverages unique OHLC data** - Garman-Klass 7.4x more efficient
4. **Complementary to VIX/DXY** - internal gold dynamics vs external factors
5. **Exactly 3 features** - lessons from real_rate failure (7 features = noise)

---

**Report Compiled:** 2026-02-14
**Target Submodel:** Technical (Attempt 1)
**Next Phase:** Architect (fact-checking and design document)
