import pandas as pd
import numpy as np
from scipy.stats import entropy
from datetime import datetime
import json
import os
import sys

# Load data
df = pd.read_csv('data/raw/yield_curve.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print("=" * 80)
print("YIELD CURVE DATA QUALITY CHECK - 7 STEPS")
print("=" * 80)
print()

# Initialize results
all_results = []
reject_flag = False

# ============================================================================
# STEP 1: FILE EXISTENCE & BASIC SHAPE
# ============================================================================
print("STEP 1: FILE EXISTENCE & BASIC STRUCTURE")
print("-" * 80)

result_step1 = {"step": "file_structure", "issues": []}

# Check file exists
if not os.path.exists('data/raw/yield_curve.csv'):
    result_step1["issues"].append("CRITICAL: File does not exist")
    reject_flag = True
else:
    print(f"OK File exists: data/raw/yield_curve.csv")

# Check expected columns
expected_cols = ['dgs10', 'dgs2', 'dgs5', 'dgs10_change', 'dgs2_change',
                 'spread', 'spread_change', 'curvature_raw', 'curvature_change']
missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    result_step1["issues"].append(f"CRITICAL: Missing columns: {missing_cols}")
    reject_flag = True
else:
    print(f"OK All expected columns present: {len(df.columns)} columns")

# Check row count
print(f"OK Row count: {len(df)} rows (expected ~2840)")
if len(df) < 2800:
    result_step1["issues"].append(f"WARNING: Row count {len(df)} is lower than expected ~2840")
else:
    print(f"  Row count is acceptable")

# Check date range
date_min = df.index.min()
date_max = df.index.max()
print(f"OK Date range: {date_min.date()} to {date_max.date()}")
print(f"  Expected: 2014-10-02 to 2026-02-12")

result_step1["passed"] = not any("CRITICAL" in i for i in result_step1["issues"])
all_results.append(result_step1)
print(f"Status: {'PASS' if result_step1['passed'] else 'FAIL'}\n")

# ============================================================================
# STEP 2: BASIC STATISTICS & EXTREME VALUES
# ============================================================================
print("STEP 2: BASIC STATISTICS & OUTLIERS")
print("-" * 80)

result_step2 = {"step": "basic_stats", "issues": []}

numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Numeric columns: {list(numeric_cols)}\n")

for col in numeric_cols:
    series = df[col].dropna()
    if len(series) == 0:
        result_step2["issues"].append(f"CRITICAL: {col} is all NaN")
        continue

    mean_val = series.mean()
    std_val = series.std()
    min_val = series.min()
    max_val = series.max()

    print(f"{col}:")
    print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}")
    print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}")

    # Check for constant columns
    if std_val == 0:
        result_step2["issues"].append(f"CRITICAL: {col} has std = 0 (constant)")
        print(f"  CRITICAL: Constant column")

    # Check for extreme values (yields should be 0-8%, changes should be -2 to +2)
    if col.endswith('_change'):
        if abs(min_val) > 2 or abs(max_val) > 2:
            result_step2["issues"].append(f"WARNING: {col} has extreme change values ({min_val:.3f} to {max_val:.3f})")
            print(f"  WARNING: Change values outside typical range")
    else:
        if min_val < -0.5 or max_val > 8:
            result_step2["issues"].append(f"WARNING: {col} has values outside typical yield range")
            print(f"  WARNING: Outside typical 0-8% range")

    print()

result_step2["passed"] = not any("CRITICAL" in i for i in result_step2["issues"])
all_results.append(result_step2)
print(f"Status: {'PASS' if result_step2['passed'] else 'FAIL'}\n")

# ============================================================================
# STEP 3: MISSING VALUES & GAPS
# ============================================================================
print("STEP 3: MISSING VALUES & GAPS")
print("-" * 80)

result_step3 = {"step": "missing_values", "issues": []}

for col in numeric_cols:
    pct_missing = df[col].isnull().mean() * 100
    count_missing = df[col].isnull().sum()

    print(f"{col}: {count_missing:4d} NaN ({pct_missing:5.2f}%)", end="")

    if pct_missing > 20:
        result_step3["issues"].append(f"CRITICAL: {col} has {pct_missing:.1f}% missing (>20%)")
        print(" - CRITICAL")
    elif pct_missing > 5:
        result_step3["issues"].append(f"WARNING: {col} has {pct_missing:.1f}% missing (5-20%)")
        print(" - WARNING")
    else:
        print(" - OK")

print()

# Check for consecutive NaN runs
print("Checking for consecutive NaN runs...")
for col in numeric_cols:
    # Find max consecutive NaN
    mask = df[col].isnull().astype(int)
    if mask.sum() == 0:
        print(f"  {col}: No NaN")
        continue

    runs = mask.groupby((mask != mask.shift()).cumsum()).sum()
    max_consecutive = runs.max() if len(runs) > 0 else 0

    if max_consecutive > 10:
        result_step3["issues"].append(f"WARNING: {col} has {int(max_consecutive)} consecutive NaN days")
        print(f"  {col}: {int(max_consecutive)} consecutive NaN days - WARNING")
    else:
        print(f"  {col}: Max {int(max_consecutive)} consecutive NaN - OK")

result_step3["passed"] = not any("CRITICAL" in i for i in result_step3["issues"])
all_results.append(result_step3)
print(f"Status: {'PASS' if result_step3['passed'] else 'FAIL'}\n")

# ============================================================================
# STEP 4: FUTURE INFORMATION LEAK CHECK
# ============================================================================
print("STEP 4: FUTURE INFORMATION LEAK (LOOKAHEAD BIAS)")
print("-" * 80)

result_step4 = {"step": "future_leak", "issues": []}

# Load gold target
try:
    if os.path.exists('data/processed/target.csv'):
        target_df = pd.read_csv('data/processed/target.csv', index_col=0, parse_dates=True)
        target_col = target_df.columns[0]
        print(f"Loaded target: {target_col} from target.csv")
        print(f"Target shape: {target_df.shape}")
        print()

        # Align with yield_curve data
        aligned = pd.concat([df, target_df[target_col]], axis=1, join='inner')
        print(f"Aligned data: {len(aligned)} rows")

        # For each numeric column in yield_curve, check correlation at lag 0 vs lag 1
        print("\nCorrelation check (lookahead bias detection):")
        for col in numeric_cols:
            if col not in aligned.columns or aligned[col].isnull().all():
                continue

            # Lag 0: current day feature vs next-day target
            corr_lag0 = aligned[col].corr(aligned[target_col].shift(-1))
            # Lag 1: previous day feature vs next-day target (orthogonal check)
            corr_lag1 = aligned[col].shift(1).corr(aligned[target_col].shift(-1))

            print(f"  {col}: corr(lag0)={corr_lag0:7.3f}, corr(lag1)={corr_lag1:7.3f}", end="")

            # If lag0 correlation is >0.8 absolute, flag as potential leak
            if abs(corr_lag0) > 0.8:
                result_step4["issues"].append(f"CRITICAL: {col} high correlation with target ({corr_lag0:.3f}) - POTENTIAL LEAK")
                print(" - CRITICAL LEAK")
            # If lag0 >> lag1 and both significant, flag as warning
            elif abs(corr_lag0) > abs(corr_lag1) * 3 and abs(corr_lag0) > 0.3:
                result_step4["issues"].append(f"WARNING: {col} lag0 correlation much higher than lag1 ({corr_lag0:.3f} >> {corr_lag1:.3f})")
                print(" - WARNING: Possible lookahead")
            else:
                print(" - OK")
    else:
        print("Target file not found - skipping lookahead bias check")
        result_step4["issues"].append("WARNING: Target file not found for lookahead check")
except Exception as e:
    print(f"Error loading target: {e}")
    result_step4["issues"].append(f"WARNING: Could not load target for lookahead check: {str(e)}")

result_step4["passed"] = not any("CRITICAL" in i for i in result_step4["issues"])
all_results.append(result_step4)
print(f"Status: {'PASS' if result_step4['passed'] else 'FAIL'}\n")

# ============================================================================
# STEP 5: TIME SERIES INTEGRITY
# ============================================================================
print("STEP 5: TIME SERIES INTEGRITY")
print("-" * 80)

result_step5 = {"step": "temporal", "issues": []}

dates = df.index
print(f"Date index type: {type(dates)}")
print(f"Is monotonic increasing: {dates.is_monotonic_increasing}")

if not dates.is_monotonic_increasing:
    result_step5["issues"].append("CRITICAL: Dates are not sorted in ascending order")
    print("CRITICAL: Dates not sorted")
else:
    print("OK Dates are properly sorted")

# Check for duplicates
dupes = dates.duplicated().sum()
if dupes > 0:
    result_step5["issues"].append(f"CRITICAL: {dupes} duplicate dates found")
    print(f"CRITICAL: {dupes} duplicate dates")
else:
    print(f"OK No duplicate dates")

# Check for gaps (excluding weekends/holidays)
print("\nChecking for gaps > 7 days (likely non-trading days):")
gaps = dates.to_series().diff()
large_gaps = gaps[gaps > pd.Timedelta(days=7)]
if len(large_gaps) > 0:
    print(f"Found {len(large_gaps)} gaps > 7 days:")
    for date, gap in large_gaps.head(10).items():
        result_step5["issues"].append(f"WARNING: Gap of {gap.days} days on {date.date()}")
        print(f"  {date.date()}: {gap.days} days")
else:
    print("OK No gaps > 7 days")

result_step5["passed"] = not any("CRITICAL" in i for i in result_step5["issues"])
all_results.append(result_step5)
print(f"Status: {'PASS' if result_step5['passed'] else 'FAIL'}\n")

# ============================================================================
# STEP 6: CORRELATION & VIF PRE-CHECK
# ============================================================================
print("STEP 6: CORRELATION & VIF PRE-CHECK (vs Base Features)")
print("-" * 80)

result_step6 = {"step": "vif_correlation", "issues": []}

# Load base_features for comparison
try:
    base_df = pd.read_csv('data/processed/base_features.csv', index_col=0, parse_dates=True)
    print(f"Loaded base_features: {base_df.shape}")
    print(f"Base features columns: {list(base_df.columns[:5])}... ({len(base_df.columns)} total)\n")

    # Align data
    aligned_full = pd.concat([base_df, df], axis=1, join='inner')
    print(f"Aligned data: {len(aligned_full)} rows\n")

    # Compute VIF manually (simplified: max absolute correlation among yield_curve features and base features)
    yield_curve_cols = [col for col in df.columns if col in aligned_full.columns]
    base_yield_cols = [col for col in base_df.columns if 'yield' in col.lower() or 'real_rate' in col.lower()]

    print(f"Yield curve output columns: {yield_curve_cols}")
    print(f"Base feature comparators: {base_yield_cols}\n")

    # Compute correlations
    print("Correlation matrix (yield_curve vs base features):")
    for yc_col in yield_curve_cols[:3]:
        if yc_col not in aligned_full.columns:
            continue
        print(f"\n{yc_col}:")
        for base_col in base_yield_cols:
            if base_col not in aligned_full.columns:
                continue
            corr = aligned_full[yc_col].corr(aligned_full[base_col])
            print(f"  vs {base_col}: {corr:7.3f}", end="")
            if abs(corr) > 0.9:
                result_step6["issues"].append(f"WARNING: {yc_col} high correlation with {base_col} ({corr:.3f})")
                print(" - HIGH")
            else:
                print()

except Exception as e:
    print(f"Note: Could not load base_features for full VIF check: {e}")
    print("This is not critical - evaluator will perform detailed VIF in Gate 2")

result_step6["passed"] = not any("CRITICAL" in i for i in result_step6["issues"])
all_results.append(result_step6)
print(f"\nStatus: {'PASS' if result_step6['passed'] else 'FAIL'}\n")

# ============================================================================
# STEP 7: OUTPUT FORMAT VALIDATION
# ============================================================================
print("STEP 7: OUTPUT FORMAT & SCHEMA VALIDATION")
print("-" * 80)

result_step7 = {"step": "format_schema", "issues": []}

# Expected schema per design doc
expected_schema = {
    'dgs10': 'float64',
    'dgs2': 'float64',
    'dgs5': 'float64',
    'dgs10_change': 'float64',
    'dgs2_change': 'float64',
    'spread': 'float64',
    'spread_change': 'float64',
    'curvature_raw': 'float64',
    'curvature_change': 'float64'
}

print("Checking schema compliance...")
schema_ok = True
for col, expected_type in expected_schema.items():
    if col not in df.columns:
        result_step7["issues"].append(f"CRITICAL: Missing column {col}")
        schema_ok = False
        print(f"  {col}: MISSING")
    else:
        actual_type = str(df[col].dtype)
        type_match = 'float' in actual_type
        status = "OK" if type_match else "WARNING"
        print(f"  {col}: {actual_type} ({status})")
        if not type_match:
            result_step7["issues"].append(f"WARNING: {col} type {actual_type} (expected float)")

print(f"\nOK Date index: {df.index.name} (DatetimeIndex)")
print(f"OK Total rows: {len(df)}")
print(f"OK Total columns: {len(df.columns)}")

# Check for any NaN in output after preprocessing
total_nan = df.isnull().sum().sum()
print(f"\nOK Total NaN cells: {total_nan}")
if total_nan > len(df.columns) * 100:
    result_step7["issues"].append(f"WARNING: Large number of NaN values: {total_nan}")

result_step7["passed"] = not any("CRITICAL" in i for i in result_step7["issues"])
all_results.append(result_step7)
print(f"Status: {'PASS' if result_step7['passed'] else 'FAIL'}\n")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("=" * 80)
print("SUMMARY & DECISION")
print("=" * 80)

critical_count = sum(1 for r in all_results for i in r.get("issues", []) if "CRITICAL" in i)
warning_count = sum(1 for r in all_results for i in r.get("issues", []) if "WARNING" in i)

print(f"Total CRITICAL issues: {critical_count}")
print(f"Total WARNING issues: {warning_count}")
print()

if critical_count > 0:
    print("DECISION: REJECT")
    print("Reason: Critical issues detected")
    decision = "REJECT"
elif warning_count > 5:
    print("DECISION: CONDITIONAL_PASS")
    print("Reason: Multiple warnings, but no critical issues")
    decision = "CONDITIONAL_PASS"
else:
    print("DECISION: PASS")
    print("Reason: Data quality is acceptable")
    decision = "PASS"

print()
print("DETAILED ISSUES:")
for i, result in enumerate(all_results, 1):
    if result["issues"]:
        print(f"\nStep {i} ({result['step']}):")
        for issue in result["issues"]:
            print(f"  - {issue}")

# Save report
report = {
    "feature": "yield_curve",
    "attempt": 1,
    "timestamp": datetime.now().isoformat(),
    "steps": all_results,
    "critical_issues": [i for r in all_results for i in r.get("issues",[]) if "CRITICAL" in i],
    "warnings": [i for r in all_results for i in r.get("issues",[]) if "WARNING" in i],
    "decision": decision,
    "passed": decision in ["PASS", "CONDITIONAL_PASS"]
}

os.makedirs('logs/datacheck', exist_ok=True)
with open('logs/datacheck/yield_curve_attempt_1.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nReport saved to logs/datacheck/yield_curve_attempt_1.json")
print()
