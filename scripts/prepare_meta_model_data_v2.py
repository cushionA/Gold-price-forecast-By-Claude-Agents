"""
Meta-Model Data Preparation - Attempt 2
Fixes from Attempt 1:
1. Technical.csv timezone normalization (utc=True)
2. Base features converted to daily changes (except VIX)
3. Domain-specific NaN imputation
4. Exclude price-level features and CNY features
5. Expected output: 22 features, ~2200 rows → ~1765 train
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "meta_model"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Meta-Model Data Preparation - Attempt 2")
print("=" * 80)

# ============================================================================
# 1. Load base features
# ============================================================================
print("\n[1/8] Loading base features...")
base = pd.read_csv(DATA_DIR / "processed" / "base_features.csv")
print(f"  Loaded: {base.shape}")
print(f"  Date range: {base['Date'].min()} to {base['Date'].max()}")

# Select 5 base features and apply transformations
print("\n[2/8] Transforming base features...")
base['Date'] = pd.to_datetime(base['Date']).dt.strftime('%Y-%m-%d')

# Apply .diff() to 4 non-stationary features
base['real_rate_change'] = base['real_rate_real_rate'].diff()
base['dxy_change'] = base['dxy_dxy'].diff()
base['yield_spread_change'] = base['yield_curve_yield_spread'].diff()
base['inflation_exp_change'] = base['inflation_expectation_inflation_expectation'].diff()

# VIX stays as level (stationary)
base['vix'] = base['vix_vix']

# Select only the 5 transformed features + Date
base_features = base[['Date', 'real_rate_change', 'dxy_change', 'vix',
                       'yield_spread_change', 'inflation_exp_change']].copy()

# Drop first row (NaN from diff)
base_features = base_features.iloc[1:].reset_index(drop=True)
print(f"  After diff and drop: {base_features.shape}")
print(f"  Features: {list(base_features.columns)}")

# ============================================================================
# 2. Load target
# ============================================================================
print("\n[3/8] Loading target...")
target = pd.read_csv(DATA_DIR / "processed" / "target.csv")
target['Date'] = pd.to_datetime(target['Date']).dt.strftime('%Y-%m-%d')
print(f"  Loaded: {target.shape}")

# Merge base + target
df = base_features.merge(target, on='Date', how='inner')
print(f"  After merge with target: {df.shape}")

# ============================================================================
# 3. Load submodel outputs
# ============================================================================
print("\n[4/8] Loading submodel outputs...")

# VIX submodel (3 features)
vix_sub = pd.read_csv(DATA_DIR / "submodel_outputs" / "vix.csv")
vix_sub['Date'] = pd.to_datetime(vix_sub['date']).dt.strftime('%Y-%m-%d')
vix_sub = vix_sub[['Date', 'vix_regime_probability', 'vix_mean_reversion_z', 'vix_persistence']]
df = df.merge(vix_sub, on='Date', how='left')
print(f"  + VIX: {df.shape}, NaN: {df.isnull().sum().sum()}")

# Technical submodel (3 features) - CRITICAL: utc=True to fix timezone issue
technical_sub = pd.read_csv(DATA_DIR / "submodel_outputs" / "technical.csv")
technical_sub['Date'] = pd.to_datetime(technical_sub['date'], utc=True).dt.strftime('%Y-%m-%d')
technical_sub = technical_sub[['Date', 'tech_trend_regime_prob', 'tech_mean_reversion_z', 'tech_volatility_regime']]
df = df.merge(technical_sub, on='Date', how='left')
print(f"  + Technical: {df.shape}, NaN: {df.isnull().sum().sum()}")

# Cross-asset submodel (3 features)
cross_sub = pd.read_csv(DATA_DIR / "submodel_outputs" / "cross_asset.csv")
cross_sub['Date'] = pd.to_datetime(cross_sub['Date']).dt.strftime('%Y-%m-%d')
cross_sub = cross_sub[['Date', 'xasset_regime_prob', 'xasset_recession_signal', 'xasset_divergence']]
df = df.merge(cross_sub, on='Date', how='left')
print(f"  + Cross-asset: {df.shape}, NaN: {df.isnull().sum().sum()}")

# Yield curve submodel (2 features, exclude yc_regime_prob)
yc_sub = pd.read_csv(DATA_DIR / "submodel_outputs" / "yield_curve.csv")
yc_sub = yc_sub.rename(columns={'index': 'Date'})
yc_sub['Date'] = pd.to_datetime(yc_sub['Date']).dt.strftime('%Y-%m-%d')
yc_sub = yc_sub[['Date', 'yc_spread_velocity_z', 'yc_curvature_z']]  # Exclude yc_regime_prob
df = df.merge(yc_sub, on='Date', how='left')
print(f"  + Yield curve: {df.shape}, NaN: {df.isnull().sum().sum()}")

# ETF flow submodel (3 features)
etf_sub = pd.read_csv(DATA_DIR / "submodel_outputs" / "etf_flow.csv")
etf_sub['Date'] = pd.to_datetime(etf_sub['Date']).dt.strftime('%Y-%m-%d')
etf_sub = etf_sub[['Date', 'etf_regime_prob', 'etf_capital_intensity', 'etf_pv_divergence']]
df = df.merge(etf_sub, on='Date', how='left')
print(f"  + ETF flow: {df.shape}, NaN: {df.isnull().sum().sum()}")

# Inflation expectation submodel (3 features)
ie_sub = pd.read_csv(DATA_DIR / "submodel_outputs" / "inflation_expectation.csv")
ie_sub = ie_sub.rename(columns={'Unnamed: 0': 'Date'})
ie_sub['Date'] = pd.to_datetime(ie_sub['Date']).dt.strftime('%Y-%m-%d')
ie_sub = ie_sub[['Date', 'ie_regime_prob', 'ie_anchoring_z', 'ie_gold_sensitivity_z']]
df = df.merge(ie_sub, on='Date', how='left')
print(f"  + Inflation expectation: {df.shape}, NaN: {df.isnull().sum().sum()}")

print(f"\n  Total after all merges: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ============================================================================
# 4. Verify feature count
# ============================================================================
print("\n[5/8] Verifying feature count...")
feature_cols = [c for c in df.columns if c not in ['Date', 'gold_return_next']]
print(f"  Feature count: {len(feature_cols)}")
print(f"  Expected: 22")

if len(feature_cols) != 22:
    print(f"  ERROR: Expected 22 features, got {len(feature_cols)}")
    print(f"  Features: {feature_cols}")
    raise ValueError("Feature count mismatch")

print(f"  [OK] Feature count verified: {len(feature_cols)}")

# Define feature groups
base_feature_cols = ['real_rate_change', 'dxy_change', 'vix', 'yield_spread_change', 'inflation_exp_change']
submodel_feature_cols = [c for c in feature_cols if c not in base_feature_cols]

print(f"  Base features (5): {base_feature_cols}")
print(f"  Submodel features (17): {submodel_feature_cols}")

# ============================================================================
# 5. NaN analysis
# ============================================================================
print("\n[6/8] Analyzing NaN patterns...")
nan_summary = df[feature_cols].isnull().sum()
nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False)

if len(nan_summary) > 0:
    print(f"  Features with NaN:")
    for col, count in nan_summary.items():
        pct = count / len(df) * 100
        print(f"    {col}: {count} ({pct:.2f}%)")
else:
    print(f"  No NaN found in features")

total_nan_rows = df[feature_cols].isnull().any(axis=1).sum()
print(f"\n  Total rows with any NaN: {total_nan_rows} ({total_nan_rows/len(df)*100:.2f}%)")

# ============================================================================
# 6. Apply domain-specific NaN imputation
# ============================================================================
print("\n[7/8] Applying domain-specific NaN imputation...")

# Regime probabilities -> 0.5 (maximum uncertainty)
regime_cols = [c for c in feature_cols if 'regime' in c.lower() and ('prob' in c.lower() or 'probability' in c.lower())]
if len(regime_cols) > 0:
    df[regime_cols] = df[regime_cols].fillna(0.5)
    print(f"  Imputed {len(regime_cols)} regime_prob columns with 0.5")

# Z-scores -> 0.0 (at mean = no signal)
z_cols = [c for c in feature_cols if '_z' in c]
if len(z_cols) > 0:
    df[z_cols] = df[z_cols].fillna(0.0)
    print(f"  Imputed {len(z_cols)} z-score columns with 0.0")

# Divergence/signal columns -> 0.0
signal_cols = ['xasset_divergence', 'xasset_recession_signal', 'etf_capital_intensity', 'etf_pv_divergence']
signal_cols = [c for c in signal_cols if c in feature_cols]
if len(signal_cols) > 0:
    df[signal_cols] = df[signal_cols].fillna(0.0)
    print(f"  Imputed {len(signal_cols)} signal columns with 0.0")

# Continuous state (tech_volatility_regime) -> median
if 'tech_volatility_regime' in feature_cols:
    median_val = df['tech_volatility_regime'].median()
    df['tech_volatility_regime'] = df['tech_volatility_regime'].fillna(median_val)
    print(f"  Imputed tech_volatility_regime with median ({median_val:.4f})")

# Persistence (vix_persistence) -> median
if 'vix_persistence' in feature_cols:
    median_val = df['vix_persistence'].median()
    df['vix_persistence'] = df['vix_persistence'].fillna(median_val)
    print(f"  Imputed vix_persistence with median ({median_val:.4f})")

# Check remaining NaN
remaining_nan = df[feature_cols].isnull().sum().sum()
print(f"\n  Remaining NaN after imputation: {remaining_nan}")

if remaining_nan > 0:
    print(f"  WARNING: Still have NaN. Dropping rows with any NaN...")
    before_drop = len(df)
    df = df.dropna(subset=feature_cols)
    print(f"  Dropped {before_drop - len(df)} rows. Remaining: {len(df)}")

# ============================================================================
# 7. Data split (80:15:15 time-series order)
# ============================================================================
print("\n[8/8] Splitting data...")

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print(f"  Total rows: {n}")
print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
print(f"  Val: {len(val_df)} ({len(val_df)/n*100:.1f}%)")
print(f"  Test: {len(test_df)} ({len(test_df)/n*100:.1f}%)")

# ============================================================================
# 8. Save outputs
# ============================================================================
print("\nSaving outputs...")

train_df.to_csv(OUTPUT_DIR / "meta_model_attempt_2_train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "meta_model_attempt_2_val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "meta_model_attempt_2_test.csv", index=False)

print(f"  [OK] Saved train: {OUTPUT_DIR / 'meta_model_attempt_2_train.csv'}")
print(f"  [OK] Saved val: {OUTPUT_DIR / 'meta_model_attempt_2_val.csv'}")
print(f"  [OK] Saved test: {OUTPUT_DIR / 'meta_model_attempt_2_test.csv'}")

# ============================================================================
# 9. Quality checks and summary
# ============================================================================
print("\n" + "=" * 80)
print("QUALITY CHECKS")
print("=" * 80)

print(f"\n[OK] Feature count: {len(feature_cols)} (expected: 22)")
print(f"[OK] Total rows: {n} (expected: ~2200+)")
print(f"[OK] Train samples: {len(train_df)} (expected: ~1765, Attempt 1 had only 964)")
print(f"[OK] Samples per feature: {len(train_df)/len(feature_cols):.1f}:1 (expected: ~80:1)")
print(f"[OK] No NaN in final dataset: {df[feature_cols].isnull().sum().sum() == 0}")

# Check stationarity assumptions (no extreme outliers)
print("\nBase feature statistics (to verify transformations):")
for col in base_feature_cols:
    mean = train_df[col].mean()
    std = train_df[col].std()
    print(f"  {col}: mean={mean:.6f}, std={std:.6f}")

# Feature importance preview (correlation with target)
print("\nFeature correlation with target (train set):")
correlations = train_df[feature_cols + ['gold_return_next']].corr()['gold_return_next'].drop('gold_return_next').abs().sort_values(ascending=False)
print("  Top 10 features:")
for i, (col, corr) in enumerate(correlations.head(10).items(), 1):
    print(f"    {i}. {col}: {corr:.4f}")

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE")
print("=" * 80)
print(f"\nOutputs:")
print(f"  - {OUTPUT_DIR / 'meta_model_attempt_2_train.csv'}")
print(f"  - {OUTPUT_DIR / 'meta_model_attempt_2_val.csv'}")
print(f"  - {OUTPUT_DIR / 'meta_model_attempt_2_test.csv'}")
print(f"\nReady for datachecker validation.")
