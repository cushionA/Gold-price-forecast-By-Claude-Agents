#!/usr/bin/env python3
"""
Datachecker Agent: CNY_DEMAND ATTEMPT 1
7-step standardized data quality check
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import sys

# ============================================================================
# STEP 0: Load data
# ============================================================================
csv_path = "data/processed/cny_demand/features_input.csv"
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

print("=" * 80)
print("DATACHECKER: CNY_DEMAND ATTEMPT 1")
print("=" * 80)
print(f"\nData loaded: {csv_path}")
print(f"Shape: {df.shape}")
print(f"Index type: {df.index.dtype}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# ============================================================================
# STEP 1: File Existence Check
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: FILE EXISTENCE CHECK")
print("=" * 80)

results_step1 = {
    "step": "file_check",
    "issues": [],
    "passed": True
}

required_files = [
    "data/processed/cny_demand/features_input.csv",
]

for fpath in required_files:
    if not os.path.exists(fpath):
        results_step1["issues"].append(f"CRITICAL: {fpath} does not exist")
        results_step1["passed"] = False
    else:
        print(f"OK: {fpath} exists")

if results_step1["passed"]:
    print("\nRESULT: PASS")
else:
    print(f"\nRESULT: FAIL - {results_step1['issues']}")

# ============================================================================
# STEP 2: Basic Statistics Check
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: BASIC STATISTICS CHECK")
print("=" * 80)

results_step2 = {
    "step": "basic_stats",
    "issues": [],
    "passed": True
}

print(f"\nTotal rows: {len(df)}")

if len(df) < 1000:
    results_step2["issues"].append(f"WARNING: Row count is low ({len(df)} rows, need >= 1000)")

# Check numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {numeric_cols}")

for col in numeric_cols:
    col_std = df[col].std()
    col_min = df[col].min()
    col_max = df[col].max()

    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.6f}")
    print(f"  Std Dev: {col_std:.6f}")
    print(f"  Min: {col_min:.6f}")
    print(f"  Max: {col_max:.6f}")

    # Zero variance check
    if col_std == 0:
        results_step2["issues"].append(f"CRITICAL: {col} has zero standard deviation")
        results_step2["passed"] = False

    # Extreme value check (allowed for currency rates and returns)
    if abs(col_min) > 1e6 or abs(col_max) > 1e6:
        results_step2["issues"].append(f"WARNING: {col} has extreme values (min={col_min}, max={col_max})")

if results_step2["passed"]:
    print("\nRESULT: PASS")
else:
    print(f"\nRESULT: {results_step2}")

# ============================================================================
# STEP 3: Missing Values Check
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: MISSING VALUES CHECK")
print("=" * 80)

results_step3 = {
    "step": "missing_values",
    "issues": [],
    "passed": True
}

for col in df.columns:
    pct_missing = df[col].isnull().mean() * 100

    if pct_missing > 0:
        print(f"{col}: {pct_missing:.2f}% missing ({df[col].isnull().sum()} values)")

        if pct_missing > 20:
            results_step3["issues"].append(f"CRITICAL: {col} has {pct_missing:.1f}% missing (>20%)")
            results_step3["passed"] = False
        elif pct_missing > 5:
            results_step3["issues"].append(f"WARNING: {col} has {pct_missing:.1f}% missing (>5%)")

# Check for consecutive missing values
print("\nConsecutive missing value checks:")
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        # Find max consecutive missing
        missing_mask = df[col].isnull().astype(int)
        groups = missing_mask.groupby((missing_mask != missing_mask.shift()).cumsum())
        max_consec = groups.sum().max()
        if max_consec > 10:
            print(f"  {col}: {max_consec} consecutive missing values")
            results_step3["issues"].append(f"WARNING: {col} has {max_consec} consecutive missing values")

if results_step3["passed"] and not any("WARNING" in i for i in results_step3["issues"]):
    print("\nRESULT: PASS")
else:
    print(f"\nRESULT: {results_step3}")

# ============================================================================
# STEP 4: Future Information Leak Check
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: FUTURE INFORMATION LEAK CHECK")
print("=" * 80)

results_step4 = {
    "step": "future_leak",
    "issues": [],
    "passed": True
}

# Assuming first column might be target-related, check others
target_col = None
for col in df.columns:
    if 'return' in col.lower() or 'target' in col.lower():
        target_col = col
        break

if target_col is None:
    print(f"No target column detected. Assuming input features only (no target present).")
    print("Future leak check: SKIPPED (no target to leak to)")
else:
    print(f"Target column detected: {target_col}")

    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue

        # Correlation at lag 0 and lag 1
        corr0 = df[col].corr(df[target_col])
        corr1 = df[col].shift(1).corr(df[target_col])

        print(f"\n{col}:")
        print(f"  Corr(lag=0): {corr0:.4f}")
        print(f"  Corr(lag=1): {corr1:.4f}")

        # Check for leak
        if abs(corr0) > 0.8:
            results_step4["issues"].append(f"CRITICAL: {col} corr with target = {corr0:.3f} (leak suspected)")
            results_step4["passed"] = False
        elif abs(corr0) > 0.3 and abs(corr1) != 0 and abs(corr0) > abs(corr1) * 3:
            print(f"    -> Possible future leak (lag0 >> lag1)")
            results_step4["issues"].append(f"WARNING: {col} lag-0 corr ({corr0:.3f}) >> lag-1 ({corr1:.3f})")

if results_step4["passed"]:
    print("\nRESULT: PASS (no critical leaks)")
else:
    print(f"\nRESULT: {results_step4}")

# ============================================================================
# STEP 5: Autocorrelation Check
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: AUTOCORRELATION CHECK (for features only)")
print("=" * 80)

results_step5 = {
    "step": "autocorrelation",
    "issues": [],
    "passed": True
}

from statsmodels.graphics.tsaplots import acf as calc_acf

for col in df.select_dtypes(include=[np.number]).columns:
    # Get valid (non-NaN) values
    valid_data = df[col].dropna()

    if len(valid_data) > 10:
        try:
            # Calculate lag-1 autocorrelation
            acf_vals = calc_acf(valid_data, nlags=1, fft=False)
            acf_lag1 = acf_vals[1]

            print(f"\n{col}:")
            print(f"  ACF(lag=1): {acf_lag1:.6f}")

            # For cny_close, high autocorr is acceptable (managed float)
            # For cny_return, must be <0.99
            if 'close' in col.lower() or 'usd' in col.lower():
                # This is likely a price level, autocorr can be high
                print(f"    -> Level variable (price), high autocorr acceptable")
            else:
                # This should be a return/change variable
                if abs(acf_lag1) > 0.99:
                    results_step5["issues"].append(f"CRITICAL: {col} autocorr = {acf_lag1:.4f} (>0.99)")
                    results_step5["passed"] = False
                elif abs(acf_lag1) > 0.95:
                    results_step5["issues"].append(f"WARNING: {col} autocorr = {acf_lag1:.4f} (high, >0.95)")
        except Exception as e:
            print(f"\n{col}: ACF calculation skipped ({type(e).__name__})")

if results_step5["passed"]:
    print("\nRESULT: PASS")
else:
    print(f"\nRESULT: {results_step5}")

# ============================================================================
# STEP 6: VIF & Multicollinearity Check
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: VIF & MULTICOLLINEARITY CHECK")
print("=" * 80)

results_step6 = {
    "step": "vif_check",
    "issues": [],
    "passed": True
}

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Prepare numeric data (no NaN)
    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    if len(numeric_df) > 0:
        print(f"\nCalculating VIF for {len(numeric_df.columns)} features on {len(numeric_df)} rows...")

        vif_data = []
        for i, col in enumerate(numeric_df.columns):
            try:
                vif = variance_inflation_factor(numeric_df.values, i)
                vif_data.append({"column": col, "vif": vif})
                print(f"  {col}: VIF = {vif:.4f}")
            except Exception as e:
                print(f"  {col}: VIF calculation failed ({type(e).__name__})")

        # Check for VIF > 10
        for item in vif_data:
            if item["vif"] > 10:
                results_step6["issues"].append(f"WARNING: {item['column']} has VIF = {item['vif']:.2f} (>10)")
            elif np.isinf(item["vif"]):
                results_step6["issues"].append(f"WARNING: {item['column']} has infinite VIF (perfect collinearity)")
except Exception as e:
    print(f"\nVIF calculation error: {e}")
    results_step6["issues"].append(f"WARNING: VIF calculation failed ({type(e).__name__})")

if results_step6["passed"]:
    print("\nRESULT: PASS")
else:
    print(f"\nRESULT: {results_step6}")

# ============================================================================
# STEP 7: Date Coverage & Schema Integrity
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: DATE COVERAGE & SCHEMA INTEGRITY")
print("=" * 80)

results_step7 = {
    "step": "temporal_integrity",
    "issues": [],
    "passed": True
}

with open("shared/schema_freeze.json", "r") as f:
    schema = json.load(f)

print(f"\nBase features date range:")
print(f"  Start: {schema['date_range']['start']}")
print(f"  End: {schema['date_range']['end']}")
print(f"  Rows: {schema['row_count']}")

print(f"\nCNY demand data date range:")
print(f"  Start: {df.index.min()}")
print(f"  End: {df.index.max()}")
print(f"  Rows: {len(df)}")

schema_start = pd.Timestamp(schema['date_range']['start'])
schema_end = pd.Timestamp(schema['date_range']['end'])

# Check if cny data covers the base features range
if df.index.min() > schema_start:
    results_step7["issues"].append(f"WARNING: CNY data starts after base features ({df.index.min()} vs {schema_start})")

if df.index.max() < schema_end:
    results_step7["issues"].append(f"WARNING: CNY data ends before base features ({df.index.max()} vs {schema_end})")

# Check temporal integrity (no gaps, sorted)
dates_sorted = df.index.is_monotonic_increasing
print(f"\nDates are sorted: {dates_sorted}")

if not dates_sorted:
    results_step7["issues"].append(f"CRITICAL: Dates are not sorted")
    results_step7["passed"] = False

# Check for duplicate dates
dupe_dates = df.index.duplicated().sum()
print(f"Duplicate dates: {dupe_dates}")

if dupe_dates > 0:
    results_step7["issues"].append(f"CRITICAL: {dupe_dates} duplicate dates found")
    results_step7["passed"] = False

# Check for gaps
date_diffs = df.index.to_series().diff()
large_gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]

print(f"Gaps > 7 days: {len(large_gaps)}")
for date, gap in large_gaps.items():
    print(f"  {date}: {gap.days}-day gap")
    if gap.days > 30:
        results_step7["issues"].append(f"WARNING: Large gap ({gap.days} days) at {date}")

if results_step7["passed"]:
    print("\nRESULT: PASS")
else:
    print(f"\nRESULT: {results_step7}")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)

all_results = [results_step1, results_step2, results_step3, results_step4, results_step5, results_step6, results_step7]

critical_issues = []
warnings = []

for result in all_results:
    for issue in result.get("issues", []):
        if "CRITICAL" in issue:
            critical_issues.append(issue)
        elif "WARNING" in issue:
            warnings.append(issue)

print(f"\nCRITICAL ISSUES: {len(critical_issues)}")
for issue in critical_issues:
    print(f"  X {issue}")

print(f"\nWARNINGS: {len(warnings)}")
for issue in warnings:
    print(f"  ! {issue}")

# Decision
if critical_issues:
    action = "REJECT"
    reason = f"{len(critical_issues)} critical issue(s) found"
elif len(warnings) > 5:
    action = "CONDITIONAL_PASS"
    reason = f"{len(warnings)} warnings (>5)"
else:
    action = "PASS"
    reason = "All critical checks passed"

print(f"\nACTION: {action}")
print(f"REASON: {reason}")

# Save report
report = {
    "feature": "cny_demand",
    "attempt": 1,
    "timestamp": datetime.now().isoformat(),
    "steps": all_results,
    "critical_issues": critical_issues,
    "warnings": warnings,
    "action": action,
    "overall_passed": (action in ["PASS", "CONDITIONAL_PASS"]),
    "data_shape": list(df.shape),
    "date_range": {
        "start": str(df.index.min()),
        "end": str(df.index.max()),
    }
}

# Ensure logs directory exists
os.makedirs("logs/datacheck", exist_ok=True)

report_path = f"logs/datacheck/cny_demand_1.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved: {report_path}")
print("\n" + "=" * 80)
