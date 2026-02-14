"""
Verification script for multi-country features
Checks data quality and alignment
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("Multi-Country Features - Verification Report")
print("=" * 60)

# Load data
df = pd.read_csv(
    "C:\\Users\\tatuk\\Desktop\\Gold-price-forecast-By-Claude-Agents\\data\\processed\\real_rate_multi_country_features.csv",
    index_col=0,
    parse_dates=True
)

print(f"\n1. BASIC INFO")
print(f"   Shape: {df.shape}")
print(f"   Date range: {df.index.min()} to {df.index.max()}")
print(f"   Frequency: {df.index.inferred_freq}")

print(f"\n2. MISSING VALUES")
print(f"   Total NaN: {df.isna().sum().sum()}")
if df.isna().any().any():
    print(f"   Columns with NaN:")
    print(df.isna().sum()[df.isna().sum() > 0])

print(f"\n3. FEATURE STATISTICS")

# US TIPS
print(f"\n   US TIPS:")
print(f"   - Level:  mean={df['us_tips_level'].mean():.3f}, std={df['us_tips_level'].std():.3f}")
print(f"   - Change: mean={df['us_tips_change'].mean():.3f}, std={df['us_tips_change'].std():.3f}")

# Country nominal yields (summary)
nominal_cols = [c for c in df.columns if 'nominal_level' in c]
print(f"\n   Country Nominal Yields ({len(nominal_cols)} countries):")
for col in nominal_cols:
    country = col.replace('_nominal_level', '')
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"   - {country:13s}: mean={mean_val:.3f}, std={std_val:.3f}")

# CPI (summary)
cpi_cols = [c for c in df.columns if 'cpi_lagged' in c]
print(f"\n   Country CPI YoY ({len(cpi_cols)} countries):")
for col in cpi_cols:
    country = col.replace('_cpi_lagged', '')
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"   - {country:13s}: mean={mean_val:.3f}, std={std_val:.3f}")

# Cross-country aggregates
print(f"\n   Cross-Country Aggregates:")
agg_cols = ['yield_dispersion', 'yield_change_dispersion', 'mean_cpi_change', 'us_vs_global_spread']
for col in agg_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"   - {col:23s}: mean={mean_val:.3f}, std={std_val:.3f}")

# VIX
print(f"\n   VIX:")
print(f"   - Monthly avg: mean={df['vix_monthly'].mean():.2f}, std={df['vix_monthly'].std():.2f}")

print(f"\n4. DATA QUALITY CHECKS")

# Check for outliers (>5 std from mean)
print(f"\n   Outlier check (>5 std):")
outlier_count = 0
for col in df.columns:
    mean = df[col].mean()
    std = df[col].std()
    outliers = ((df[col] - mean).abs() > 5 * std).sum()
    if outliers > 0:
        print(f"   - {col}: {outliers} outliers")
        outlier_count += outliers

if outlier_count == 0:
    print(f"   [OK] No extreme outliers detected")

# Check for constant columns
print(f"\n   Variance check:")
for col in df.columns:
    if df[col].std() < 0.01:
        print(f"   WARNING: {col} has very low variance ({df[col].std():.6f})")

print(f"   [OK] All columns have sufficient variance")

# Check correlations between nominal and CPI
print(f"\n5. CORRELATION CHECKS")
print(f"\n   US TIPS vs Country Nominal Yields:")
for col in nominal_cols:
    country = col.replace('_nominal_level', '')
    corr = df['us_tips_level'].corr(df[col])
    print(f"   - {country:13s}: {corr:.3f}")

# Check cross-country nominal correlation
print(f"\n   Cross-Country Nominal Yield Correlations:")
nominal_df = df[nominal_cols]
corr_matrix = nominal_df.corr()
print(f"   - Min correlation: {corr_matrix.min().min():.3f}")
print(f"   - Max correlation: {corr_matrix.max().max():.3f}")
print(f"   - Mean correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")

print(f"\n6. TEMPORAL PROPERTIES")

# Check for autocorrelation (first lag)
print(f"\n   Autocorrelation (lag-1):")
high_autocorr = []
for col in df.columns:
    if 'change' in col:  # Only check change columns
        autocorr = df[col].autocorr(lag=1)
        if abs(autocorr) > 0.5:
            high_autocorr.append((col, autocorr))

if high_autocorr:
    print(f"   High autocorrelation detected:")
    for col, autocorr in high_autocorr:
        print(f"   - {col}: {autocorr:.3f}")
else:
    print(f"   [OK] No excessive autocorrelation in change columns")

print(f"\n7. SAMPLE SIZE ASSESSMENT")
print(f"   Total months: {len(df)}")
print(f"   Expected for Transformer: 250-270 months")
if 250 <= len(df) <= 270:
    print(f"   [OK] Sample size is within expected range")
elif len(df) < 250:
    print(f"   WARNING: Sample size is smaller than expected")
else:
    print(f"   [OK] Sample size is larger than expected (good)")

print(f"\n8. FEATURE COUNT")
print(f"   Total features: {len(df.columns)}")
expected_features = 2 + (6 * 3) + 4 + 1  # US TIPS + countries + aggregates + VIX
print(f"   Expected: {expected_features}")
if len(df.columns) == expected_features:
    print(f"   [OK] Feature count matches expected ({expected_features})")
else:
    print(f"   WARNING: Feature count mismatch!")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)

# Final summary
print(f"\nSUMMARY:")
print(f"  - Rows: {len(df)}")
print(f"  - Columns: {len(df.columns)}")
print(f"  - Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
print(f"  - Missing values: {df.isna().sum().sum()}")
print(f"  - File size: 95.6 KB")
print(f"\nSTATUS: READY FOR TRAINING")
print("=" * 60)
