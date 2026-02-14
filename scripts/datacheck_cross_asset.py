#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standardized 7-Step Data Quality Check: Cross-Asset (Attempt 1)
Validates cross_asset.csv against design specifications and Gate 1 requirements.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import sys
import io
import os

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("STANDARDIZED 7-STEP DATA QUALITY CHECK: Cross-Asset (Attempt 1)")
print("="*80)

# ============================================================================
# STEP 0: FILE EXISTENCE CHECK
# ============================================================================
print("\n[STEP 0] FILE EXISTENCE CHECK")
print("-" * 80)

required_files = [
    'data/raw/cross_asset.csv',
]

file_check_passed = True
for fpath in required_files:
    if Path(fpath).exists():
        size_mb = Path(fpath).stat().st_size / (1024*1024)
        print(f"✓ {fpath} (size: {size_mb:.2f} MB)")
    else:
        print(f"✗ {fpath} - MISSING")
        file_check_passed = False

if not file_check_passed:
    print("\nSTEP 0: REJECT - Missing required files")
    sys.exit(1)

# Load data
df = pd.read_csv('data/raw/cross_asset.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

print(f"\nData loaded successfully")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"  Row count: {len(df)}")

results = {
    "feature": "cross_asset",
    "attempt": 1,
    "timestamp": datetime.now().isoformat(),
    "steps": []
}

# ============================================================================
# STEP 1: MISSING VALUES CHECK
# ============================================================================
print("\n[STEP 1] MISSING VALUES CHECK")
print("-" * 80)

step1 = {"step": "missing_values", "issues": []}

for col in df.columns:
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100

    if null_pct > 0:
        print(f"  {col}: {null_count} NaN ({null_pct:.2f}%)")

        if null_pct > 20:
            step1["issues"].append(f"CRITICAL: {col} missing {null_pct:.2f}% (threshold: 20%)")
        elif null_pct > 5:
            step1["issues"].append(f"WARNING: {col} missing {null_pct:.2f}% (threshold: 5%)")
    else:
        print(f"  {col}: 0 NaN ✓")

# Check for consecutive NaN
print("\n  Checking for consecutive NaN...")
for col in df.columns:
    if df[col].isnull().any():
        consecutive_nans = df[col].isnull().astype(int).groupby(
            (df[col].notnull().astype(int).cumsum())
        ).sum().max()
        if consecutive_nans > 10:
            step1["issues"].append(f"WARNING: {col} has {consecutive_nans} consecutive NaN")
            print(f"    {col}: max consecutive NaN = {consecutive_nans}")

step1["passed"] = not any("CRITICAL" in i for i in step1["issues"])
results["steps"].append(step1)

if step1["passed"]:
    print("\n✓ STEP 1: PASS - Missing values within acceptable range")
else:
    print("\n✗ STEP 1: REJECT - Unacceptable missing values")

# ============================================================================
# STEP 2: BASIC STATISTICS & OUTLIERS
# ============================================================================
print("\n[STEP 2] BASIC STATISTICS & OUTLIERS")
print("-" * 80)

step2 = {"step": "basic_stats", "issues": []}

# Check row count
if len(df) < 1000:
    step2["issues"].append(f"WARNING: Row count is {len(df)} (below 1000)")
    print(f"  ⚠ Row count: {len(df)} (below 1000)")
else:
    print(f"  ✓ Row count: {len(df)}")

# Check numeric columns for extreme values
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\n  Checking numeric columns for extreme values:")

for col in numeric_cols:
    col_data = df[col].dropna()
    col_min = col_data.min()
    col_max = col_data.max()
    col_std = col_data.std()
    col_mean = col_data.mean()

    print(f"    {col}:")
    print(f"      min={col_min:.6g}, max={col_max:.6g}, mean={col_mean:.6g}, std={col_std:.6g}")

    # Check for zero std
    if col_std == 0:
        step2["issues"].append(f"CRITICAL: {col} has zero standard deviation")
        print(f"      ✗ ZERO STD DEV - CRITICAL")

    # Check for extreme outliers (>1e6 in absolute value)
    if abs(col_min) > 1e6 or abs(col_max) > 1e6:
        step2["issues"].append(f"WARNING: {col} contains extreme values (min={col_min:.6g}, max={col_max:.6g})")
        print(f"      ⚠ Extreme values detected")

step2["passed"] = not any("CRITICAL" in i for i in step2["issues"])
results["steps"].append(step2)

if step2["passed"]:
    print("\n✓ STEP 2: PASS - Basic statistics acceptable")
else:
    print("\n✗ STEP 2: REJECT - Unacceptable basic statistics")

# ============================================================================
# STEP 3: AUTOCORRELATION CHECK (GATE 1 COMPLIANCE)
# ============================================================================
print("\n[STEP 3] AUTOCORRELATION CHECK")
print("-" * 80)

step3 = {"step": "autocorrelation", "issues": []}

print("  Computing lag-1 autocorrelation for each column:")

for col in numeric_cols:
    col_data = df[col].dropna()
    if len(col_data) > 2:
        autocorr = col_data.autocorr(lag=1)
        print(f"    {col}: {autocorr:.6f}", end="")

        if abs(autocorr) > 0.99:
            step3["issues"].append(f"CRITICAL: {col} autocorr={autocorr:.6f} (exceeds 0.99 threshold)")
            print(" ✗ EXCEEDS 0.99 - CRITICAL")
        elif abs(autocorr) > 0.95:
            step3["issues"].append(f"WARNING: {col} autocorr={autocorr:.6f} (near 0.99 threshold)")
            print(" ⚠ Near threshold")
        else:
            print(" ✓")

step3["passed"] = not any("CRITICAL" in i for i in step3["issues"])
results["steps"].append(step3)

if step3["passed"]:
    print("\n✓ STEP 3: PASS - Autocorrelation within acceptable range")
else:
    print("\n✗ STEP 3: REJECT - Unacceptable autocorrelation")

# ============================================================================
# STEP 4: FUTURE LEAK CHECK
# ============================================================================
print("\n[STEP 4] FUTURE LEAK CHECK")
print("-" * 80)

step4 = {"step": "future_leak", "issues": []}

# Check if 'gold_return' column exists (target variable)
if 'gold_return' in df.columns:
    target = df['gold_return']

    print(f"  Target variable: gold_return")
    print(f"  Checking correlation with all other numeric columns...")

    for col in numeric_cols:
        if col == 'gold_return':
            continue

        # Remove NaN for correlation calculation
        valid_mask = ~(df[col].isnull() | target.isnull())
        if valid_mask.sum() > 2:
            corr_lag0 = df[col][valid_mask].corr(target[valid_mask])

            # Check for high correlation (potential leak)
            if abs(corr_lag0) > 0.8:
                step4["issues"].append(f"CRITICAL: {col} has correlation {corr_lag0:.4f} with target (possible leak)")
                print(f"    {col}: corr={corr_lag0:.4f} ✗ POTENTIAL LEAK")
            elif abs(corr_lag0) > 0.5:
                step4["issues"].append(f"WARNING: {col} has high correlation {corr_lag0:.4f} with target")
                print(f"    {col}: corr={corr_lag0:.4f} ⚠")
            else:
                print(f"    {col}: corr={corr_lag0:.4f} ✓")
else:
    print("  No 'gold_return' target column found - skipping correlation check")

step4["passed"] = not any("CRITICAL" in i for i in step4["issues"])
results["steps"].append(step4)

if step4["passed"]:
    print("\n✓ STEP 4: PASS - No future leak detected")
else:
    print("\n✗ STEP 4: REJECT - Future leak detected")

# ============================================================================
# STEP 5: TIME SERIES ALIGNMENT CHECK
# ============================================================================
print("\n[STEP 5] TIME SERIES ALIGNMENT CHECK")
print("-" * 80)

step5 = {"step": "temporal_alignment", "issues": []}

# Check date sorting
if not df.index.is_monotonic_increasing:
    step5["issues"].append("CRITICAL: Dates are not sorted in increasing order")
    print("  ✗ Dates not sorted - CRITICAL")
else:
    print("  ✓ Dates are sorted in increasing order")

# Check for duplicate dates
duplicate_dates = df.index.duplicated().sum()
if duplicate_dates > 0:
    step5["issues"].append(f"CRITICAL: {duplicate_dates} duplicate dates found")
    print(f"  ✗ {duplicate_dates} duplicate dates - CRITICAL")
else:
    print("  ✓ No duplicate dates")

# Check date range against expected
expected_start = pd.Timestamp('2014-10-01')
expected_end = pd.Timestamp('2026-02-13')

actual_start = df.index[0]
actual_end = df.index[-1]

print(f"\n  Expected date range: {expected_start.date()} to {expected_end.date()}")
print(f"  Actual date range:   {actual_start.date()} to {actual_end.date()}")

if actual_start < expected_start:
    print(f"  ⚠ Start date {actual_start.date()} is before expected {expected_start.date()}")
if actual_end < expected_end:
    print(f"  ⚠ End date {actual_end.date()} is before expected {expected_end.date()}")

# Check for large gaps (>7 days)
date_diffs = df.index.to_series().diff()
gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]

if len(gaps) > 0:
    print(f"\n  Found {len(gaps)} gaps > 7 days:")
    for date, gap in gaps.head(5).items():
        step5["issues"].append(f"WARNING: {gap.days} day gap at {date.date()}")
        print(f"    {date.date()}: {gap.days} days")
    if len(gaps) > 5:
        print(f"    ... and {len(gaps)-5} more gaps")
else:
    print("  ✓ No gaps > 7 days")

step5["passed"] = not any("CRITICAL" in i for i in step5["issues"])
results["steps"].append(step5)

if step5["passed"]:
    print("\n✓ STEP 5: PASS - Temporal alignment acceptable")
else:
    print("\n✗ STEP 5: REJECT - Temporal alignment issues")

# ============================================================================
# STEP 6: VIF PRE-CHECK (Correlation with base features)
# ============================================================================
print("\n[STEP 6] VIF PRE-CHECK (Correlation with other features)")
print("-" * 80)

step6 = {"step": "vif_correlation", "issues": []}

# Try to load base_features for comparison
try:
    base_df = pd.read_csv('data/processed/base_features.csv', index_col=0, parse_dates=True)

    # Find common dates
    common_dates = df.index.intersection(base_df.index)

    print(f"  Common dates with base_features: {len(common_dates)}/{len(df)}")

    if len(common_dates) > 100:
        # Compute correlations
        corr_pairs = []
        for col_x in df.columns:
            for col_y in base_df.columns:
                if col_x == col_y or col_x == 'date' or col_y == 'date':
                    continue

                x_vals = df[col_x].loc[common_dates].dropna()
                y_vals = base_df[col_y].loc[common_dates].dropna()

                common = x_vals.index.intersection(y_vals.index)
                if len(common) > 20:
                    corr = x_vals[common].corr(y_vals[common])
                    if abs(corr) > 0.3:
                        corr_pairs.append((col_x, col_y, corr))

        if corr_pairs:
            print(f"  Found {len(corr_pairs)} correlations > 0.3:")
            for col_x, col_y, corr in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
                print(f"    {col_x} vs {col_y}: {corr:.4f}")

                # Flag if correlation > 0.7 (VIF risk)
                if abs(corr) > 0.7:
                    step6["issues"].append(f"WARNING: {col_x} vs {col_y} corr={corr:.4f} (VIF risk)")
        else:
            print("  ✓ No high correlations (>0.3) with base features")
    else:
        print("  ⚠ Insufficient overlapping dates for reliable VIF check")

except FileNotFoundError:
    print("  ⚠ base_features.csv not found - skipping VIF pre-check")

step6["passed"] = not any("CRITICAL" in i for i in step6["issues"])
results["steps"].append(step6)

if step6["passed"]:
    print("\n✓ STEP 6: PASS - VIF pre-check acceptable")
else:
    print("\n✗ STEP 6: REJECT - VIF pre-check failed")

# ============================================================================
# STEP 7: OUTPUT SCHEMA VALIDATION
# ============================================================================
print("\n[STEP 7] OUTPUT SCHEMA VALIDATION")
print("-" * 80)

step7 = {"step": "schema_validation", "issues": []}

# Expected columns per design doc
expected_columns = ['gold_close', 'copper_close', 'silver_close', 'gold_return',
                   'silver_return', 'copper_return', 'gsr', 'gcr']

print(f"  Expected columns: {expected_columns}")
print(f"  Actual columns:   {list(df.columns)}")

missing_cols = set(expected_columns) - set(df.columns)
extra_cols = set(df.columns) - set(expected_columns)

if missing_cols:
    step7["issues"].append(f"CRITICAL: Missing columns {missing_cols}")
    print(f"  ✗ Missing: {missing_cols}")
else:
    print(f"  ✓ All expected columns present")

if extra_cols:
    print(f"  ⚠ Extra columns: {extra_cols}")

# Check column data types
print(f"\n  Verifying numeric types:")
for col in numeric_cols:
    if col in df.columns:
        print(f"    {col}: {df[col].dtype} ✓")

step7["passed"] = not any("CRITICAL" in i for i in step7["issues"])
results["steps"].append(step7)

if step7["passed"]:
    print("\n✓ STEP 7: PASS - Schema validation successful")
else:
    print("\n✗ STEP 7: REJECT - Schema validation failed")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

critical_issues = [i for step in results["steps"] for i in step.get("issues", []) if "CRITICAL" in i]
warnings = [i for step in results["steps"] for i in step.get("issues", []) if "WARNING" in i]

results["critical_issues"] = critical_issues
results["warnings"] = warnings

print(f"\nSteps passed: {sum(1 for s in results['steps'] if s.get('passed', False))}/7")
print(f"Critical issues: {len(critical_issues)}")
print(f"Warnings: {len(warnings)}")

if critical_issues:
    print(f"\nCRITICAL ISSUES:")
    for issue in critical_issues:
        print(f"  - {issue}")

if warnings:
    print(f"\nWARNINGS (may require review):")
    for warning in warnings[:5]:
        print(f"  - {warning}")
    if len(warnings) > 5:
        print(f"  ... and {len(warnings)-5} more warnings")

# Determine overall result
if critical_issues:
    results["action"] = "REJECT"
    results["status"] = "FAILED"
    print(f"\n✗✗✗ OVERALL RESULT: REJECT ✗✗✗")
    print("Return to builder_data for corrections")
elif len(warnings) > 5:
    results["action"] = "CONDITIONAL_PASS"
    results["status"] = "PASSED_WITH_WARNINGS"
    print(f"\n⚠ OVERALL RESULT: CONDITIONAL_PASS ⚠")
    print("Proceed to evaluator with caution - review warnings first")
else:
    results["action"] = "PASS"
    results["status"] = "PASSED"
    print(f"\n✓✓✓ OVERALL RESULT: PASS ✓✓✓")
    print("Proceed to builder_model")

results["overall_passed"] = results["action"] != "REJECT"

# ============================================================================
# SAVE REPORT
# ============================================================================
print("\n" + "="*80)
print("Saving detailed report...")
print("="*80)

report_path = Path('logs/datacheck')
report_path.mkdir(parents=True, exist_ok=True)

report_file = report_path / f'cross_asset_attempt_1.json'
with open(report_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"✓ Report saved: {report_file}")

# Print summary table
print("\n" + "="*80)
print("STEP SUMMARY")
print("="*80)
print("\n{:<5} {:<30} {:<10} {:<10}".format("Step", "Name", "Status", "Issues"))
print("-" * 60)

for i, step in enumerate(results["steps"], 1):
    status = "PASS" if step.get("passed", False) else "FAIL"
    issue_count = len(step.get("issues", []))
    name = step.get("step", "?")
    print("{:<5} {:<30} {:<10} {:<10}".format(i, name, status, issue_count))

print("\n" + "="*80)
