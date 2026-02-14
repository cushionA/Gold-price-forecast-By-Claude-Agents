import pandas as pd
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime, timedelta

# Load data
dxy_data = pd.read_csv('data/multi_country/dxy_constituents.csv', parse_dates=['Date'], index_col='Date')
with open('data/multi_country/dxy_metadata.json') as f:
    metadata = json.load(f)

base_features = pd.read_csv('data/processed/base_features.csv', parse_dates=['Date'], index_col='Date')

print("=" * 80)
print("DATACHECKER REPORT: DXY (Attempt 1)")
print("=" * 80)
print()

# STEP 1: FILE EXISTENCE & BASIC STRUCTURE
print("STEP 1: File Existence & Basic Structure")
print("-" * 80)
try:
    assert dxy_data is not None
    assert len(dxy_data) > 0
    print(f"[PASS] dxy_constituents.csv exists: {len(dxy_data)} rows")
    print(f"[PASS] dxy_metadata.json exists")
    print(f"  Columns: {list(dxy_data.columns)}")
    print(f"  Date range: {dxy_data.index.min().date()} to {dxy_data.index.max().date()}")
except Exception as e:
    print(f"[FAIL] File structure error: {e}")

print()

# STEP 2: MISSING VALUES CHECK
print("STEP 2: Missing Values Analysis")
print("-" * 80)
missing_issues = []
for col in dxy_data.columns:
    missing_pct = dxy_data[col].isnull().mean() * 100
    missing_count = dxy_data[col].isnull().sum()
    print(f"{col:15} | Missing: {missing_count:4} ({missing_pct:5.2f}%)", end="")

    if missing_pct > 20:
        print(" [CRITICAL]")
        missing_issues.append(f"CRITICAL: {col} has {missing_pct:.1f}% missing values")
    elif missing_pct > 5:
        print(" [WARNING]")
        missing_issues.append(f"WARNING: {col} has {missing_pct:.1f}% missing values")
    else:
        print(" [OK]")

print()

# STEP 3: OUTLIERS & EXTREME VALUES (Z-score)
print("STEP 3: Outliers & Extreme Values Analysis")
print("-" * 80)
outlier_issues = []
for col in dxy_data.select_dtypes(include=[np.number]).columns:
    valid_data = dxy_data[col].dropna()
    if len(valid_data) > 1:
        z_scores = np.abs(stats.zscore(valid_data))
        extreme_count = (z_scores > 4).sum()

        print(f"{col:15} | Z>4: {extreme_count:3} rows", end="")
        if extreme_count > 0:
            print(f" | Min/Max: {valid_data.min():.4f} / {valid_data.max():.4f} [WARNING]")
            outlier_issues.append(f"WARNING: {col} has {extreme_count} extreme values (Z>4)")
        else:
            print(f" | Min/Max: {valid_data.min():.4f} / {valid_data.max():.4f} [OK]")

print()

# STEP 4: VARIANCE & CONSTANT COLUMNS
print("STEP 4: Variance Check")
print("-" * 80)
variance_issues = []
for col in dxy_data.select_dtypes(include=[np.number]).columns:
    std_val = dxy_data[col].std()
    print(f"{col:15} | Std: {std_val:10.6f}", end="")
    if std_val == 0 or np.isnan(std_val):
        print(" [CRITICAL - zero variance]")
        variance_issues.append(f"CRITICAL: {col} has zero variance")
    elif std_val < 1e-8:
        print(" [WARNING - near-zero variance]")
        variance_issues.append(f"WARNING: {col} has near-zero variance")
    else:
        print(" [OK]")

print()

# STEP 5: TEMPORAL CONSISTENCY
print("STEP 5: Temporal Consistency Check")
print("-" * 80)
temporal_issues = []

# Check monotonic increasing
is_sorted = dxy_data.index.is_monotonic_increasing
print(f"Date order monotonic increasing: {is_sorted}", end="")
if not is_sorted:
    print(" [CRITICAL]")
    temporal_issues.append("CRITICAL: Dates are not monotonically increasing")
else:
    print(" [OK]")

# Check for duplicates
dupes = dxy_data.index.duplicated().sum()
print(f"Duplicate dates: {dupes}", end="")
if dupes > 0:
    print(" [CRITICAL]")
    temporal_issues.append(f"CRITICAL: {dupes} duplicate dates found")
else:
    print(" [OK]")

# Check for gaps > 5 days
diffs = dxy_data.index.to_series().diff()
large_gaps = (diffs > timedelta(days=5)).sum()
print(f"Gaps > 5 days: {large_gaps}", end="")
if large_gaps > 0:
    gap_dates = dxy_data.index[diffs > timedelta(days=5)]
    print(" [WARNING]")
    for gd in gap_dates[:3]:
        temporal_issues.append(f"WARNING: {diffs[gd].days}-day gap before {gd.date()}")
else:
    print(" [OK]")

print()

# STEP 6: SCHEMA & DATE ALIGNMENT
print("STEP 6: Schema & Date Alignment")
print("-" * 80)
schema_issues = []

expected_date_range = (base_features.index.min(), base_features.index.max())
print(f"Base features date range: {expected_date_range[0].date()} to {expected_date_range[1].date()}")
print(f"DXY data date range:      {dxy_data.index.min().date()} to {dxy_data.index.max().date()}")

# Check overlap
min_overlap = max(dxy_data.index.min(), base_features.index.min())
max_overlap = min(dxy_data.index.max(), base_features.index.max())
overlap_rows = len(dxy_data.loc[min_overlap:max_overlap])
print(f"Overlap rows: {overlap_rows} out of {len(base_features)} base_features rows", end="")

if overlap_rows >= len(base_features) * 0.95:
    print(" [OK]")
else:
    print(" [WARNING]")
    schema_issues.append(f"WARNING: Only {overlap_rows}/{len(base_features)} overlap ({100*overlap_rows/len(base_features):.1f}%)")

# Expected columns per design doc
expected_columns = ['dxy_close', 'eur_usd', 'jpy', 'gbp_usd', 'usd_cad', 'usd_sek', 'usd_chf']
actual_columns = list(dxy_data.columns)
print(f"Expected columns: {expected_columns}")
print(f"Actual columns:   {actual_columns}")
if set(expected_columns) == set(actual_columns):
    print("[PASS] Column match [OK]")
else:
    print("[FAIL] Column mismatch [CRITICAL]")
    schema_issues.append("CRITICAL: Column mismatch with design doc")

print()

# STEP 7: CORRELATION WITH BASE FEATURE (dxy_dxy)
print("STEP 7: Correlation Sanity Check (dxy_close vs base dxy_dxy)")
print("-" * 80)
correlation_issues = []

# Align dates
aligned = pd.DataFrame({
    'dxy_close': dxy_data.loc[min_overlap:max_overlap, 'dxy_close'],
    'dxy_base': base_features.loc[min_overlap:max_overlap, 'dxy_dxy']
}).dropna()

if len(aligned) > 100:
    corr = aligned['dxy_close'].corr(aligned['dxy_base'])
    print(f"Correlation (dxy_close vs base dxy_dxy): {corr:.4f}", end="")

    if corr >= 0.95:
        print(" [OK]")
    elif corr >= 0.85:
        print(" [WARNING - acceptable but check for issues]")
        correlation_issues.append(f"WARNING: Correlation {corr:.4f} slightly below 0.95")
    else:
        print(" [CRITICAL - potential data mismatch]")
        correlation_issues.append(f"CRITICAL: Correlation {corr:.4f} below 0.85 (possible data source mismatch)")
else:
    print(f"Insufficient overlapping data ({len(aligned)} rows) [WARNING]")
    correlation_issues.append(f"WARNING: Only {len(aligned)} overlapping rows for correlation check")

print()

# SUMMARY
print("=" * 80)
print("SUMMARY")
print("=" * 80)

all_issues = missing_issues + outlier_issues + variance_issues + temporal_issues + schema_issues + correlation_issues

critical_count = len([i for i in all_issues if 'CRITICAL' in i])
warning_count = len([i for i in all_issues if 'WARNING' in i])

print(f"Critical issues: {critical_count}")
print(f"Warnings: {warning_count}")
print()

if all_issues:
    print("Issues found:")
    for issue in all_issues:
        print(f"  - {issue}")
else:
    print("[PASS] All checks passed!")

print()

# DETERMINE PASS/FAIL
if critical_count > 0:
    result = "REJECT"
elif warning_count > 5:
    result = "CONDITIONAL_PASS"
else:
    result = "PASS"

print(f"FINAL RESULT: {result}")
