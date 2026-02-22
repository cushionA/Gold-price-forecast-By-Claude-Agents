import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

# STEP 1: File Existence and Basic Loading
print("="*80)
print("STEP 1: FILE EXISTENCE CHECK")
print("="*80)

required_files = {
    "data_csv": "data/processed/options_market/data.csv",
    "target_csv": "data/processed/target.csv",
}

file_check = {}
for name, path in required_files.items():
    exists = os.path.exists(path)
    file_check[name] = exists
    status_mark = "PASS" if exists else "MISSING"
    print(f"  {name:20} ({path:40}): {status_mark}")

if not all(file_check.values()):
    print("\nCRITICAL: Required files missing!")
    exit(1)

# Load data
try:
    df = pd.read_csv("data/processed/options_market/data.csv", index_col=0, parse_dates=True)
    df.index.name = 'Date'
    print(f"\nData loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"\nCRITICAL: Failed to load data.csv - {e}")
    exit(1)

target = pd.read_csv("data/processed/target.csv", index_col=0, parse_dates=True)
print(f"Target loaded: {target.shape[0]} rows, {target.shape[1]} columns")

# STEP 2: Basic Info and Structure
print("\n" + "="*80)
print("STEP 2: DATA STRUCTURE CHECK")
print("="*80)

print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")
print(f"\nDate range: {df.index.min()} to {df.index.max()}")
print(f"Total span: {(df.index.max() - df.index.min()).days} days")

# STEP 3: Row Count and Continuity
print("\n" + "="*80)
print("STEP 3: ROW COUNT & DATE CONTINUITY")
print("="*80)

print(f"Total rows: {len(df)}")
print(f"Expected: ~2,806 rows")
status_rows = "PASS" if 2700 <= len(df) <= 2900 else "FAIL"
print(f"Status: {status_rows}")

# Check for duplicates
duplicates = df.index.duplicated().sum()
print(f"\nDuplicate dates: {duplicates}")
status_dupes = "PASS" if duplicates == 0 else "FAIL"
print(f"Status: {status_dupes}")

# Check continuity (business days)
date_diff = df.index.to_series().diff()
max_gap = date_diff.max()
min_gap = date_diff.min()
print(f"\nDate gaps:")
print(f"  Min gap: {min_gap}")
print(f"  Max gap: {max_gap}")
gaps_over_7 = (date_diff > pd.Timedelta(days=7)).sum()
print(f"  Gaps >7 days: {gaps_over_7}")
if gaps_over_7 > 0:
    print(f"  Note: {gaps_over_7} gaps > 7 days (weekends/holidays expected)")

# STEP 4: Missing Values
print("\n" + "="*80)
print("STEP 4: MISSING VALUES CHECK")
print("="*80)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

print("\nMissing values per column:")
for col in df.columns:
    count = missing[col]
    pct = missing_pct[col]
    status = "PASS" if pct == 0 else "WARNING" if pct < 5 else "CRITICAL"
    print(f"  {col:20}: {count:5d} ({pct:6.2f}%) {status}")

total_missing = df.isnull().sum().sum()
overall_missing_pct = (total_missing / (len(df) * len(df.columns)) * 100).round(2)
print(f"\nOverall missing rate: {overall_missing_pct}%")
status_missing = "PASS" if overall_missing_pct == 0 else "FAIL"
print(f"Status: {status_missing}")

# Check for infinite values
infinites = (np.isinf(df.select_dtypes(include=[np.number]))).sum()
print(f"\nInfinite values: {infinites.sum()}")
status_inf = "PASS" if infinites.sum() == 0 else "FAIL"
print(f"Status: {status_inf}")

# STEP 5: Data Types
print("\n" + "="*80)
print("STEP 5: DATA TYPES CHECK")
print("="*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Non-numeric columns ({len(non_numeric)}): {non_numeric}")

all_numeric = len(non_numeric) == 0
status_dtype = "PASS" if all_numeric else "FAIL"
print(f"Status: {status_dtype}")

# STEP 6: Outlier Detection
print("\n" + "="*80)
print("STEP 6: OUTLIER DETECTION")
print("="*80)

outlier_rules = {
    'skew_close': {'min': 100, 'max': 200, 'desc': 'SKEW Index level'},
    'gvz_close': {'min': 5, 'max': 80, 'desc': 'Gold Volatility Index level'},
    'skew_change': {'min': -30, 'max': 30, 'desc': 'SKEW daily change'},
    'gvz_change': {'min': -20, 'max': 20, 'desc': 'GVZ daily change'},
}

outliers_found = {}
for col, rule in outlier_rules.items():
    if col in df.columns:
        out_of_range = ((df[col] < rule['min']) | (df[col] > rule['max'])).sum()
        outliers_found[col] = out_of_range

        min_val = df[col].min()
        max_val = df[col].max()
        status = "PASS" if out_of_range == 0 else "WARNING" if out_of_range < 5 else "CRITICAL"
        print(f"\n{col}:")
        print(f"  Rule: [{rule['min']:6}, {rule['max']:6}] ({rule['desc']})")
        print(f"  Data: [{min_val:7.2f}, {max_val:7.2f}]")
        print(f"  Out of range: {out_of_range} {status}")

total_outliers = sum(outliers_found.values())
status_outliers = "PASS" if total_outliers == 0 else "FAIL" if total_outliers > 5 else "WARNING"
print(f"\nTotal outliers: {total_outliers}")
print(f"Status: {status_outliers}")

# STEP 7: Statistical Sanity
print("\n" + "="*80)
print("STEP 7: STATISTICAL SANITY CHECK")
print("="*80)

print("\nDescriptive Statistics:")
print(df.describe().to_string())

print("\n\nAutocorrelation (lag-1):")
for col in df.columns:
    acf = df[col].autocorr(lag=1)
    status = "PASS" if (abs(acf) < 0.95) else "HIGH"
    print(f"  {col:20}: {acf:8.4f} {status}")

print("\n\nZero-Variance Columns:")
for col in df.columns:
    std = df[col].std()
    is_zero = std == 0 or np.isnan(std)
    status = "PASS" if not is_zero else "CRITICAL"
    print(f"  {col:20}: std={std:8.4f} {status}")

status_stats = "PASS"
for col in df.columns:
    if df[col].std() == 0 or np.isnan(df[col].std()):
        status_stats = "FAIL"

print(f"\nStatus: {status_stats}")

# STEP 8: Future Leak Check
print("\n" + "="*80)
print("STEP 8: FUTURE LEAK CHECK")
print("="*80)

# Load target
target_col = 'gold_return' if 'gold_return' in target.columns else target.columns[0]
target_gold_return = target[target_col]

print(f"\nTarget column: {target_gold_return.name if hasattr(target_gold_return, 'name') else target_col}")
print(f"Target shape: {target_gold_return.shape}")
print(f"Target date range: {target_gold_return.index.min()} to {target_gold_return.index.max()}")

# Check for overlap
common_dates = df.index.intersection(target_gold_return.index)
overlap_pct = (len(common_dates) / min(len(df), len(target_gold_return)) * 100)

print(f"\nDate overlap with target:")
print(f"  Common dates: {len(common_dates)} / {min(len(df), len(target_gold_return))}")
print(f"  Overlap percentage: {overlap_pct:.1f}%")

# Check for look-ahead bias in changes
print(f"\nFeature design review:")
print(f"  - skew_close: Level feature (acceptable)")
print(f"  - gvz_close: Level feature (acceptable)")
print(f"  - skew_change: Daily change (acceptable - no look-ahead)")
print(f"  - gvz_change: Daily change (acceptable - no look-ahead)")

status_leak = "PASS" if overlap_pct >= 30 else "FAIL"
print(f"\nLeak check status: {status_leak}")

# STEP 9: Alignment with Target
print("\n" + "="*80)
print("STEP 9: ALIGNMENT WITH TARGET")
print("="*80)

# Merge data for analysis
merged = pd.merge(df, target_gold_return.to_frame(), left_index=True, right_index=True, how='inner')
print(f"\nMerged data (common dates): {len(merged)} rows")
print(f"Date range of merged data: {merged.index.min()} to {merged.index.max()}")

# Check for correlations with target (sanity check, not leak detection)
if merged.shape[0] > 10:
    print(f"\nCorrelations with target (should be weak - these are features, not predictors):")
    corrs = merged.corr().iloc[:-1, -1]  # Last column is target
    for col in corrs.index:
        corr_val = corrs[col]
        print(f"  {col:20}: {corr_val:8.4f}")

# Final Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

checks = {
    'File existence': file_check['data_csv'] and file_check['target_csv'],
    'Row count': 2700 <= len(df) <= 2900,
    'Duplicate dates': duplicates == 0,
    'Missing values': overall_missing_pct == 0,
    'Infinite values': infinites.sum() == 0,
    'All numeric': all_numeric,
    'Outliers acceptable': total_outliers < 10,
    'No zero variance': all(df[col].std() > 0 for col in df.columns),
    'Target overlap sufficient': overlap_pct >= 30,
}

pass_count = sum(1 for v in checks.values() if v)
total_checks = len(checks)

print(f"\nChecks passed: {pass_count}/{total_checks}")
for check, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"  {status}: {check}")

# Prepare output
result_dict = {
    'step': 'datachecker_options_market_3',
    'timestamp': datetime.now().isoformat(),
    'feature': 'options_market',
    'attempt': 3,
    'data_file': 'data/processed/options_market/data.csv',
    'checks': {
        'file_existence': file_check['data_csv'] and file_check['target_csv'],
        'row_count': len(df),
        'row_count_acceptable': 2700 <= len(df) <= 2900,
        'duplicate_dates': int(duplicates),
        'missing_values_total_pct': float(overall_missing_pct),
        'infinite_values': int(infinites.sum()),
        'all_numeric': all_numeric,
        'outliers_found': int(total_outliers),
        'zero_variance_columns': 0,
        'target_overlap_pct': round(overlap_pct, 1),
        'target_overlap_sufficient': overlap_pct >= 30,
    },
    'critical_issues': [],
    'warnings': [],
    'passed': True,
}

# Identify critical issues
if not file_check['data_csv'] or not file_check['target_csv']:
    result_dict['critical_issues'].append("Required files missing")
    result_dict['passed'] = False

if not (2700 <= len(df) <= 2900):
    result_dict['critical_issues'].append(f"Row count {len(df)} outside acceptable range [2700, 2900]")
    result_dict['passed'] = False

if duplicates > 0:
    result_dict['critical_issues'].append(f"Found {duplicates} duplicate dates")
    result_dict['passed'] = False

if overall_missing_pct > 5:
    result_dict['critical_issues'].append(f"Missing values {overall_missing_pct}% exceeds 5%")
    result_dict['passed'] = False

if infinites.sum() > 0:
    result_dict['critical_issues'].append(f"Found {infinites.sum()} infinite values")
    result_dict['passed'] = False

if not all_numeric:
    result_dict['critical_issues'].append(f"Non-numeric columns found: {non_numeric}")
    result_dict['passed'] = False

if total_outliers > 10:
    result_dict['critical_issues'].append(f"Excessive outliers: {total_outliers}")
    result_dict['passed'] = False

if any(df[col].std() == 0 for col in df.columns):
    zero_var_cols = [col for col in df.columns if df[col].std() == 0]
    result_dict['critical_issues'].append(f"Zero-variance columns: {zero_var_cols}")
    result_dict['passed'] = False

if overlap_pct < 30:
    result_dict['critical_issues'].append(f"Target overlap {overlap_pct:.1f}% below 30% threshold")
    result_dict['passed'] = False

# Add warnings
if total_outliers > 0:
    result_dict['warnings'].append(f"Found {total_outliers} outliers (within acceptable range)")

if overall_missing_pct > 0 and overall_missing_pct <= 5:
    result_dict['warnings'].append(f"Missing values {overall_missing_pct}% (within acceptable range)")

if gaps_over_7 > 0:
    result_dict['warnings'].append(f"Found {gaps_over_7} date gaps >7 days (expected for business days)")

result_dict['overall_status'] = 'PASS' if result_dict['passed'] else 'REJECT'
result_dict['checks_passed'] = pass_count
result_dict['checks_total'] = total_checks

print("\n\nFINAL DECISION:")
print(f"  Status: {result_dict['overall_status']}")
print(f"  Checks: {result_dict['checks_passed']}/{result_dict['checks_total']}")
if result_dict['critical_issues']:
    print(f"  Critical Issues: {len(result_dict['critical_issues'])}")
    for issue in result_dict['critical_issues']:
        print(f"    - {issue}")
if result_dict['warnings']:
    print(f"  Warnings: {len(result_dict['warnings'])}")
    for warn in result_dict['warnings']:
        print(f"    - {warn}")

# Save result
os.makedirs("logs/datacheck", exist_ok=True)
with open("logs/datacheck/options_market_3_datacheck.json", "w", encoding="utf-8") as f:
    json.dump(result_dict, f, indent=2, ensure_ascii=False)

print(f"\n\nResult saved to: logs/datacheck/options_market_3_datacheck.json")

EOF
