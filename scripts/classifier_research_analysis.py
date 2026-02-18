"""
Comprehensive data analysis for classifier DOWN-specific features research
Research questions 1-5 and 9 with actual data
"""

import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict

# Load environment
load_dotenv('C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents/.env')
fred = Fred(api_key=os.environ['FRED_API_KEY'])

# Fetch data (2015-2025, aligned with task requirements)
start_date = '2015-01-01'
end_date = '2025-12-31'

print("=" * 80)
print("CLASSIFIER RESEARCH: DOWN-SPECIFIC FEATURES")
print("=" * 80)
print(f"\nFetching data from {start_date} to {end_date}...\n")

# 1. Gold data
gold = yf.download('GC=F', start=start_date, end=end_date, progress=False)
# Flatten MultiIndex columns if present
if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.get_level_values(0)
gold['return'] = gold['Close'].pct_change() * 100  # percentage
gold['down_day'] = (gold['return'] < 0).astype(int)

# 2. GLD ETF
gld = yf.download('GLD', start=start_date, end=end_date, progress=False)
if isinstance(gld.columns, pd.MultiIndex):
    gld.columns = gld.columns.get_level_values(0)
gld['return'] = gld['Close'].pct_change() * 100

# 3. Silver
silver = yf.download('SI=F', start=start_date, end=end_date, progress=False)
if isinstance(silver.columns, pd.MultiIndex):
    silver.columns = silver.columns.get_level_values(0)
silver['return'] = silver['Close'].pct_change() * 100

# 4. VIX
try:
    vix_fred = fred.get_series('VIXCLS', start_date, end_date)
    vix = pd.DataFrame({'VIX': vix_fred})
except:
    vix_yahoo = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = pd.DataFrame({'VIX': vix_yahoo['Close']})

# 5. GVZ
try:
    gvz_fred = fred.get_series('GVZCLS', start_date, end_date)
    gvz = pd.DataFrame({'GVZ': gvz_fred})
except:
    gvz_yahoo = yf.download('^GVZ', start=start_date, end=end_date, progress=False)
    gvz = pd.DataFrame({'GVZ': gvz_yahoo['Close']})

# 6. S&P 500
sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)
sp500['return'] = sp500['Close'].pct_change() * 100

print(f"Data fetched. Gold data shape: {gold.shape}\n")

# ============================================================================
# QUESTION 1: DOWN day magnitude distribution
# ============================================================================
print("=" * 80)
print("Q1: DOWN DAY MAGNITUDE DISTRIBUTION")
print("=" * 80)

down_returns = gold[gold['return'] < 0]['return'].dropna()
total_down_days = len(down_returns)

# Magnitude bins
bins = {
    'small_down (-0.1% to 0%)': (down_returns >= -0.1).sum(),
    'moderate_down (-0.5% to -0.1%)': ((down_returns < -0.1) & (down_returns >= -0.5)).sum(),
    'significant_down (-1.0% to -0.5%)': ((down_returns < -0.5) & (down_returns >= -1.0)).sum(),
    'large_down (< -1.0%)': (down_returns < -1.0).sum(),
}

print(f"\nTotal DOWN days: {total_down_days}")
print(f"Total trading days: {len(gold.dropna())}")
print(f"DOWN day frequency: {total_down_days / len(gold.dropna()) * 100:.2f}%\n")

print("DOWN day magnitude distribution:")
for label, count in bins.items():
    pct = count / total_down_days * 100
    print(f"  {label}: {count:4d} days ({pct:5.2f}%)")

# Significant DOWN (> 0.3% decline)
significant_down = (down_returns < -0.3).sum()
print(f"\nSignificant DOWN (< -0.3%): {significant_down} days ({significant_down/total_down_days*100:.2f}%)")
print(f"Very significant DOWN (< -0.5%): {(down_returns < -0.5).sum()} days ({(down_returns < -0.5).sum()/total_down_days*100:.2f}%)")

# Statistics
print(f"\nDOWN day statistics:")
print(f"  Mean: {down_returns.mean():.3f}%")
print(f"  Median: {down_returns.median():.3f}%")
print(f"  Std: {down_returns.std():.3f}%")
print(f"  Min: {down_returns.min():.3f}%")
print(f"  25th percentile: {down_returns.quantile(0.25):.3f}%")
print(f"  75th percentile: {down_returns.quantile(0.75):.3f}%")

# ============================================================================
# QUESTION 2: Momentum exhaustion indicators
# ============================================================================
print("\n" + "=" * 80)
print("Q2: MOMENTUM EXHAUSTION INDICATORS")
print("=" * 80)

# 2a. RSI
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

gold['rsi_14'] = compute_rsi(gold['Close'], 14)

# RSI > 70 reversal analysis
rsi_overbought = gold[gold['rsi_14'] > 70].copy()
rsi_overbought['next_return'] = gold['return'].shift(-1)
rsi_overbought_next = rsi_overbought['next_return'].dropna()

print(f"\n2a. RSI > 70 Analysis:")
print(f"  Days with RSI > 70: {len(rsi_overbought)}")
print(f"  Next-day DOWN probability: {(rsi_overbought_next < 0).sum() / len(rsi_overbought_next) * 100:.2f}%")
print(f"  Next-day average return: {rsi_overbought_next.mean():.3f}%")
print(f"  Baseline DOWN probability (all days): {gold['down_day'].mean() * 100:.2f}%")

# 2b. Consecutive streak
def compute_streak(returns):
    """Compute consecutive streak (positive = up streak, negative = down streak)"""
    streak = []
    current_streak = 0
    for r in returns:
        if pd.isna(r):
            streak.append(np.nan)
        elif r > 0:
            current_streak = current_streak + 1 if current_streak > 0 else 1
            streak.append(current_streak)
        elif r < 0:
            current_streak = current_streak - 1 if current_streak < 0 else -1
            streak.append(current_streak)
        else:  # exactly 0
            streak.append(0)
    return pd.Series(streak, index=returns.index)

gold['streak'] = compute_streak(gold['return'])
gold['next_return'] = gold['return'].shift(-1)

# Extended winning streak (5+ days) reversal
long_streaks = gold[gold['streak'] >= 5].copy()
long_streaks_next = long_streaks['next_return'].dropna()

print(f"\n2b. Consecutive Up Streak >= 5 days:")
print(f"  Occurrences: {len(long_streaks)}")
print(f"  Next-day DOWN probability: {(long_streaks_next < 0).sum() / len(long_streaks_next) * 100:.2f}%")
print(f"  Next-day average return: {long_streaks_next.mean():.3f}%")

# 2c. Distance from 20-day high
gold['high_20d'] = gold['Close'].rolling(20).max()
gold['low_20d'] = gold['Close'].rolling(20).min()
gold['range_20d'] = gold['high_20d'] - gold['low_20d']
gold['dist_from_high_20d'] = (gold['Close'] - gold['high_20d']) / gold['range_20d']

# At 20-day high (within 5% of range from high)
at_high = gold[gold['dist_from_high_20d'] > -0.05].copy()
at_high_next = at_high['next_return'].dropna()

print(f"\n2c. At 20-day High (within 5% of range):")
print(f"  Occurrences: {len(at_high)}")
print(f"  Next-day DOWN probability: {(at_high_next < 0).sum() / len(at_high_next) * 100:.2f}%")
print(f"  Next-day average return: {at_high_next.mean():.3f}%")

# 2d. Bollinger Band width
gold['sma_20'] = gold['Close'].rolling(20).mean()
gold['std_20'] = gold['Close'].rolling(20).std()
gold['bb_width'] = (gold['std_20'] * 2) / gold['sma_20']
gold['bb_width_pct'] = gold['bb_width'].rank(pct=True)

# High BB width (>80th percentile)
high_bb = gold[gold['bb_width_pct'] > 0.8].copy()
high_bb_next = high_bb['next_return'].dropna()

print(f"\n2d. High Bollinger Band Width (>80th percentile):")
print(f"  Occurrences: {len(high_bb)}")
print(f"  Next-day DOWN probability: {(high_bb_next < 0).sum() / len(high_bb_next) * 100:.2f}%")
print(f"  Next-day average return: {high_bb_next.mean():.3f}%")

# ============================================================================
# QUESTION 3: GVZ/VIX ratio
# ============================================================================
print("\n" + "=" * 80)
print("Q3: GVZ/VIX RATIO PREDICTIVE POWER")
print("=" * 80)

# Merge VIX and GVZ
vol_data = pd.concat([vix['VIX'], gvz['GVZ']], axis=1).dropna()
vol_data['gvz_vix_ratio'] = vol_data['GVZ'] / vol_data['VIX']
vol_data['gvz_vix_ratio_change'] = vol_data['gvz_vix_ratio'].pct_change()

# Align with gold returns
aligned = gold[['return', 'next_return']].join(vol_data, how='inner')

# Compute 3-day forward returns BEFORE filtering
aligned['return_3d'] = aligned['return'].shift(-1).rolling(3).sum()

# GVZ rising faster than VIX (ratio change > 5%)
gvz_rising = aligned[aligned['gvz_vix_ratio_change'] > 0.05].copy()
gvz_rising_1d = gvz_rising['next_return'].dropna()
gvz_rising_3d = gvz_rising['return_3d'].dropna()

print(f"\nGVZ/VIX ratio change > 5% (GVZ rising faster):")
print(f"  Occurrences: {len(gvz_rising)}")
print(f"  Next 1-day DOWN probability: {(gvz_rising_1d < 0).sum() / len(gvz_rising_1d) * 100:.2f}%")
print(f"  Next 1-day average return: {gvz_rising_1d.mean():.3f}%")
if len(gvz_rising_3d) > 0:
    print(f"  Next 3-day average return: {gvz_rising_3d.mean():.3f}%")
    print(f"  Next 3-day DOWN probability: {(gvz_rising_3d < 0).sum() / len(gvz_rising_3d) * 100:.2f}%")

# Correlation
print(f"\nGVZ/VIX ratio change vs next-day return correlation: {aligned['gvz_vix_ratio_change'].corr(aligned['next_return']):.4f}")

# ============================================================================
# QUESTION 4: Realized vol ratio optimal lookback
# ============================================================================
print("\n" + "=" * 80)
print("Q4: REALIZED VOL RATIO OPTIMAL LOOKBACK")
print("=" * 80)

def realized_vol(returns, window):
    """Compute realized volatility (annualized std)"""
    return returns.rolling(window).std() * np.sqrt(252)

gold['rv_5d'] = realized_vol(gold['return'], 5)
gold['rv_10d'] = realized_vol(gold['return'], 10)
gold['rv_20d'] = realized_vol(gold['return'], 20)
gold['rv_30d'] = realized_vol(gold['return'], 30)
gold['rv_60d'] = realized_vol(gold['return'], 60)

# Ratios
gold['rv_ratio_5_20'] = gold['rv_5d'] / gold['rv_20d']
gold['rv_ratio_5_60'] = gold['rv_5d'] / gold['rv_60d']
gold['rv_ratio_10_30'] = gold['rv_10d'] / gold['rv_30d']

# Information gain: mutual information with next-day direction
from sklearn.feature_selection import mutual_info_classif

features = ['rv_ratio_5_20', 'rv_ratio_5_60', 'rv_ratio_10_30']
X = gold[features].dropna()
y_aligned = gold.loc[X.index, 'next_return'].shift(-1).dropna()
X_aligned = X.loc[y_aligned.index]
y_direction = (y_aligned > 0).astype(int)

mi_scores = mutual_info_classif(X_aligned, y_direction, random_state=42)

print(f"\nMutual Information with next-day direction:")
for feat, mi in zip(features, mi_scores):
    print(f"  {feat}: {mi:.6f}")

# High vol ratio (>1.2) reversal analysis
for feat in features:
    high_vol_ratio = gold[gold[feat] > 1.2].copy()
    high_vol_next = high_vol_ratio['next_return'].dropna()
    if len(high_vol_next) > 0:
        print(f"\n{feat} > 1.2:")
        print(f"  Occurrences: {len(high_vol_next)}")
        print(f"  Next-day DOWN probability: {(high_vol_next < 0).sum() / len(high_vol_next) * 100:.2f}%")

# ============================================================================
# QUESTION 5: Day-of-week effect
# ============================================================================
print("\n" + "=" * 80)
print("Q5: DAY-OF-WEEK EFFECT")
print("=" * 80)

gold['dayofweek'] = gold.index.dayofweek  # 0=Mon, 4=Fri
gold_clean = gold[['return', 'down_day', 'dayofweek']].dropna()

dow_stats = gold_clean.groupby('dayofweek').agg({
    'return': ['mean', 'std', 'count'],
    'down_day': 'mean'
}).round(4)

dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
dow_stats.index = [dow_names[i] for i in dow_stats.index]

print(f"\nDay-of-week statistics (2015-2025):")
print(dow_stats)

# Statistical significance (t-test vs overall mean)
overall_mean = gold_clean['return'].mean()
print(f"\nOverall mean return: {overall_mean:.4f}%")
print(f"\nStatistical significance (t-test vs overall mean):")
for i, day in enumerate(dow_names):
    day_returns = gold_clean[gold_clean['dayofweek'] == i]['return']
    if len(day_returns) > 30:
        t_stat, p_val = stats.ttest_1samp(day_returns, overall_mean)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        print(f"  {day:10s}: t={t_stat:6.3f}, p={p_val:.4f} {sig}")

# ANOVA across all days
print(f"\nANOVA test (are any days different?):")
day_groups = [gold_clean[gold_clean['dayofweek'] == i]['return'].dropna() for i in range(5)]
f_stat, p_val = stats.f_oneway(*day_groups)
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {p_val:.4f}")
if p_val < 0.05:
    print(f"  Result: Significant day-of-week effect detected (p < 0.05)")
else:
    print(f"  Result: No significant day-of-week effect (p >= 0.05)")

# ============================================================================
# QUESTION 9: VIX regime interaction with DOWN probability
# ============================================================================
print("\n" + "=" * 80)
print("Q9: VIX REGIME INTERACTION WITH DOWN PROBABILITY")
print("=" * 80)

# Align VIX with gold
gold_vix = gold[['return', 'next_return', 'down_day']].join(vix, how='inner')

# Define VIX regimes (quantile-based)
gold_vix['vix_regime'] = pd.qcut(gold_vix['VIX'], q=3, labels=['low', 'medium', 'high'])

# DOWN probability by VIX regime
regime_stats = gold_vix.groupby('vix_regime').agg({
    'down_day': ['mean', 'count'],
    'return': 'mean'
}).round(4)

print(f"\nGold behavior by VIX regime:")
print(regime_stats)

# Next-day DOWN probability
gold_vix['next_down'] = gold_vix['down_day'].shift(-1)
next_down_by_regime = gold_vix.groupby('vix_regime')['next_down'].mean()

print(f"\nNext-day DOWN probability by current VIX regime:")
for regime in ['low', 'medium', 'high']:
    prob = next_down_by_regime[regime]
    count = gold_vix[gold_vix['vix_regime'] == regime].shape[0]
    print(f"  VIX {regime:6s}: {prob*100:.2f}% (n={count})")

# Chi-square test for independence
from scipy.stats import chi2_contingency
contingency = pd.crosstab(gold_vix['vix_regime'], gold_vix['next_down'])
chi2, p_val, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test (VIX regime vs next-day DOWN):")
print(f"  Chi2: {chi2:.4f}, p-value: {p_val:.4f}")
if p_val < 0.05:
    print(f"  Result: VIX regime and DOWN probability are significantly related (p < 0.05)")
else:
    print(f"  Result: No significant relationship (p >= 0.05)")

# Gold-VIX correlation by regime
print(f"\nGold return vs VIX change correlation by regime:")
gold_vix['vix_change'] = gold_vix['VIX'].pct_change()
for regime in ['low', 'medium', 'high']:
    subset = gold_vix[gold_vix['vix_regime'] == regime]
    corr = subset['return'].corr(subset['vix_change'])
    print(f"  VIX {regime:6s}: {corr:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
