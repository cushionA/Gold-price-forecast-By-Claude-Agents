"""
VIX Data Quality Check - 7-Step Standardized Process
Datachecker Agent for VIX (attempt 1)
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os

# Configuration
DATA_FILE = "data/processed/vix_processed.csv"
BASE_FEATURES_FILE = "data/processed/base_features.csv"
TARGET_FILE = "data/processed/target.csv"
OUTPUT_DIR = "logs/datacheck"
FEATURE_NAME = "vix"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== STEP 1: FILE EXISTENCE CHECK =====
def step1_file_check():
    """Check if required files exist"""
    results = {"step": 1, "name": "file_check", "issues": []}

    if not os.path.exists(DATA_FILE):
        results["issues"].append(f"CRITICAL: {DATA_FILE} does not exist")
    else:
        results["issues"].append(f"PASS: {DATA_FILE} exists")

    results["passed"] = not any("CRITICAL" in i for i in results["issues"])
    return results

# ===== STEP 2: BASIC STATISTICS CHECK =====
def step2_basic_stats(df):
    """Check basic statistics for anomalies"""
    results = {"step": 2, "name": "basic_stats", "issues": []}

    # Row count check
    if len(df) < 1000:
        results["issues"].append(f"WARNING: Row count low ({len(df)} rows, expected ~2700)")
    else:
        results["issues"].append(f"PASS: Row count adequate ({len(df)} rows)")

    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()

        # Standard deviation check
        if col_data.std() == 0:
            results["issues"].append(f"CRITICAL: {col} has zero standard deviation (constant)")
            continue

        # Extreme values check
        if abs(col_data.min()) > 1e6 or abs(col_data.max()) > 1e6:
            results["issues"].append(f"WARNING: {col} has extreme values (min={col_data.min():.2f}, max={col_data.max():.2f})")

        # VIX-specific checks
        if col == "vix":
            if col_data.min() < 5:
                results["issues"].append(f"WARNING: {col} minimum below typical range ({col_data.min():.2f})")
            if col_data.max() > 100:
                results["issues"].append(f"WARNING: {col} maximum above typical range ({col_data.max():.2f})")
            else:
                results["issues"].append(f"PASS: {col} range reasonable ({col_data.min():.2f}-{col_data.max():.2f})")

        # Log-change bounds check
        if col == "vix_log_change":
            if col_data.std() < 0.001:
                results["issues"].append(f"WARNING: {col} very low volatility (std={col_data.std():.6f})")
            elif col_data.abs().max() > 2.0:
                results["issues"].append(f"WARNING: {col} large moves detected (max abs={col_data.abs().max():.4f})")
            else:
                results["issues"].append(f"PASS: {col} log-change range normal (std={col_data.std():.4f})")

    results["passed"] = not any("CRITICAL" in i for i in results["issues"])
    return results

# ===== STEP 3: MISSING VALUES CHECK =====
def step3_missing_values(df):
    """Check for missing values and patterns"""
    results = {"step": 3, "name": "missing_values", "issues": []}

    missing_counts = {}
    for col in df.columns:
        pct = df[col].isnull().mean() * 100
        missing_counts[col] = pct

        if pct > 20:
            results["issues"].append(f"CRITICAL: {col} has {pct:.1f}% missing values (threshold: 20%)")
        elif pct > 5:
            results["issues"].append(f"WARNING: {col} has {pct:.1f}% missing values (threshold: 5%)")
        elif pct == 0:
            results["issues"].append(f"PASS: {col} is complete (0% missing)")
        else:
            results["issues"].append(f"WARNING: {col} has {pct:.2f}% missing values")

    # Check for consecutive missing values in numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            # Find max consecutive nulls
            mask = df[col].isnull().astype(int)
            groups = (mask.diff() != 0).cumsum()
            max_consec = mask.groupby(groups).sum().max()

            if max_consec > 10:
                results["issues"].append(f"WARNING: {col} has {int(max_consec)} consecutive missing values")

    results["missing_summary"] = missing_counts
    results["passed"] = not any("CRITICAL" in i for i in results["issues"])
    return results

# ===== STEP 4: FUTURE LEAKAGE CHECK =====
def step4_future_leak(df, target_df, target_col='gold_return_next'):
    """Check for future information leakage"""
    results = {"step": 4, "name": "future_leak", "issues": []}

    # Merge VIX data with target
    df_merged = pd.merge(df, target_df[['Date', target_col]],
                         left_on='date', right_on='Date', how='inner')

    if len(df_merged) == 0:
        results["issues"].append("WARNING: No date overlap between VIX and target data")
        results["passed"] = False
        return results

    results["issues"].append(f"INFO: Merged {len(df_merged)} rows for leakage analysis")

    # Analyze correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    leakage_detected = False

    for col in numeric_cols:
        if col == target_col:
            continue

        # Current day vs target correlation
        corr0 = df_merged[col].corr(df_merged[target_col])

        # Previous day correlation (lag)
        df_merged_shifted = df_merged.copy()
        df_merged_shifted[col + '_lag1'] = df_merged_shifted[col].shift(1)
        corr1 = df_merged_shifted[col + '_lag1'].corr(df_merged_shifted[target_col])

        # Check for lookahead bias
        if abs(corr0) > 0.8:
            results["issues"].append(f"CRITICAL: {col} has very high correlation with target ({corr0:.3f}) - possible leakage")
            leakage_detected = True
        elif abs(corr0) > abs(corr1) * 3 and abs(corr0) > 0.3:
            results["issues"].append(f"WARNING: {col} lag-0 correlation ({corr0:.3f}) >> lag-1 ({corr1:.3f}) - possible leakage")
            leakage_detected = True
        else:
            results["issues"].append(f"PASS: {col} correlation normal (lag0={corr0:.3f}, lag1={corr1:.3f})")

    results["passed"] = not leakage_detected
    return results

# ===== STEP 5: TEMPORAL INTEGRITY CHECK =====
def step5_temporal(df):
    """Check time-series integrity"""
    results = {"step": 5, "name": "temporal", "issues": []}

    # Convert to datetime
    dates = pd.to_datetime(df['date'])

    # Check monotonicity
    if not dates.is_monotonic_increasing:
        results["issues"].append("CRITICAL: Dates are not sorted in ascending order")
        results["passed"] = False
        return results
    else:
        results["issues"].append("PASS: Dates are monotonically increasing")

    # Check for duplicates
    dupes = dates.duplicated().sum()
    if dupes > 0:
        results["issues"].append(f"CRITICAL: Found {dupes} duplicate dates")
        results["passed"] = False
        return results
    else:
        results["issues"].append(f"PASS: No duplicate dates")

    # Check for large gaps (> 7 days = weekend + holiday)
    dates_series = pd.Series(dates.values)
    gaps = dates_series.diff()
    large_gaps = gaps[gaps > pd.Timedelta(days=7)]

    if len(large_gaps) > 0:
        results["issues"].append(f"WARNING: Found {len(large_gaps)} gaps > 7 days")
        for date, gap in large_gaps.items():
            results["issues"].append(f"  Gap on {date.strftime('%Y-%m-%d')}: {gap.days} days")
    else:
        results["issues"].append("PASS: No unusual gaps (> 7 days)")

    # Check frequency (should be trading days)
    min_gap = gaps.min()
    max_gap = gaps.max()
    mode_gap = gaps.mode()[0] if len(gaps.mode()) > 0 else None

    results["issues"].append(f"INFO: Date gaps - min={min_gap}, mode={mode_gap}, max={max_gap}")

    results["passed"] = not any("CRITICAL" in i for i in results["issues"])
    return results

# ===== STEP 6: SCHEMA COMPLIANCE CHECK =====
def step6_schema_compliance(df, base_features_df):
    """Check alignment with base_features schema"""
    results = {"step": 6, "name": "schema_compliance", "issues": []}

    # Expected columns
    if 'date' not in df.columns:
        results["issues"].append("CRITICAL: Missing 'date' column")
        results["passed"] = False
        return results

    if 'vix' not in df.columns:
        results["issues"].append("CRITICAL: Missing 'vix' column")
        results["passed"] = False
        return results

    results["issues"].append("PASS: Required columns present")

    # Check date range alignment
    df_dates = pd.to_datetime(df['date'])
    bf_dates = pd.to_datetime(base_features_df['Date'])

    # Find overlap
    df_min, df_max = df_dates.min(), df_dates.max()
    bf_min, bf_max = bf_dates.min(), bf_dates.max()

    overlap_min = max(df_min, bf_min)
    overlap_max = min(df_max, bf_max)
    overlap_days = (overlap_max - overlap_min).days

    if overlap_days < 500:
        results["issues"].append(f"WARNING: Limited date range overlap ({overlap_days} days)")
    else:
        results["issues"].append(f"PASS: Good date range overlap ({overlap_days} days)")

    # Check VIX value range
    vix_vals = df['vix'].dropna()
    if len(vix_vals) > 0:
        results["issues"].append(f"INFO: VIX range in data: {vix_vals.min():.2f} - {vix_vals.max():.2f}")

    results["passed"] = not any("CRITICAL" in i for i in results["issues"])
    return results

# ===== STEP 7: COMPREHENSIVE SUMMARY & DECISION =====
def step7_final_decision(all_results):
    """Generate final report and PASS/REJECT decision"""
    report = {
        "feature": FEATURE_NAME,
        "attempt": 1,
        "timestamp": datetime.now().isoformat(),
        "steps": all_results,
        "critical_issues": [],
        "warnings": [],
    }

    # Aggregate findings
    for result in all_results:
        for issue in result.get("issues", []):
            if "CRITICAL" in issue:
                report["critical_issues"].append(issue)
            elif "WARNING" in issue:
                report["warnings"].append(issue)

    # Make decision
    if report["critical_issues"]:
        report["action"] = "REJECT"
        report["reason"] = f"{len(report['critical_issues'])} CRITICAL issues found"
    elif len(report["warnings"]) > 5:
        report["action"] = "CONDITIONAL_PASS"
        report["reason"] = f"{len(report['warnings'])} warnings (manual review needed)"
    else:
        report["action"] = "PASS"
        report["reason"] = "All checks passed or resolved"

    report["overall_passed"] = report["action"] != "REJECT"
    report["summary_stats"] = {
        "total_steps": len(all_results),
        "critical_count": len(report["critical_issues"]),
        "warning_count": len(report["warnings"]),
    }

    return report

# ===== MAIN EXECUTION =====
def main():
    print("=" * 80)
    print(f"VIX DATA QUALITY CHECK - 7-STEP STANDARDIZED PROCESS")
    print(f"Feature: {FEATURE_NAME} | Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # Check if file exists first
    step1_result = step1_file_check()
    print(f"\nSTEP 1: FILE CHECK")
    for issue in step1_result["issues"]:
        print(f"  {issue}")

    if not step1_result["passed"]:
        print("\nCRITICAL: Cannot proceed without data file")
        return

    # Load data
    try:
        df = pd.read_csv(DATA_FILE)
        base_features_df = pd.read_csv(BASE_FEATURES_FILE)
        target_df = pd.read_csv(TARGET_FILE)
        print(f"\nLoaded data: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"CRITICAL: Failed to load data - {str(e)}")
        return

    all_results = [step1_result]

    # STEP 2: Basic Statistics
    print(f"\nSTEP 2: BASIC STATISTICS CHECK")
    step2_result = step2_basic_stats(df)
    for issue in step2_result["issues"]:
        print(f"  {issue}")
    all_results.append(step2_result)

    # STEP 3: Missing Values
    print(f"\nSTEP 3: MISSING VALUES CHECK")
    step3_result = step3_missing_values(df)
    for issue in step3_result["issues"]:
        print(f"  {issue}")
    all_results.append(step3_result)

    # STEP 4: Future Leakage
    print(f"\nSTEP 4: FUTURE LEAKAGE CHECK")
    step4_result = step4_future_leak(df, target_df)
    for issue in step4_result["issues"]:
        print(f"  {issue}")
    all_results.append(step4_result)

    # STEP 5: Temporal Integrity
    print(f"\nSTEP 5: TEMPORAL INTEGRITY CHECK")
    step5_result = step5_temporal(df)
    for issue in step5_result["issues"]:
        print(f"  {issue}")
    all_results.append(step5_result)

    # STEP 6: Schema Compliance
    print(f"\nSTEP 6: SCHEMA COMPLIANCE CHECK")
    step6_result = step6_schema_compliance(df, base_features_df)
    for issue in step6_result["issues"]:
        print(f"  {issue}")
    all_results.append(step6_result)

    # STEP 7: Final Decision
    print(f"\nSTEP 7: FINAL DECISION & REPORT GENERATION")
    report = step7_final_decision(all_results)

    print(f"\n" + "=" * 80)
    print(f"FINAL DECISION: {report['action']}")
    print(f"Reason: {report['reason']}")
    print(f"=" * 80)
    print(f"\nSUMMARY:")
    print(f"  Total Steps Completed: {report['summary_stats']['total_steps']}")
    print(f"  Critical Issues: {report['summary_stats']['critical_count']}")
    print(f"  Warnings: {report['summary_stats']['warning_count']}")

    if report["critical_issues"]:
        print(f"\nCRITICAL ISSUES:")
        for issue in report["critical_issues"]:
            print(f"  - {issue}")

    if report["warnings"]:
        print(f"\nWARNINGS:")
        for warning in report["warnings"][:10]:  # Show first 10
            print(f"  - {warning}")
        if len(report["warnings"]) > 10:
            print(f"  ... and {len(report['warnings']) - 10} more warnings")

    # Save report
    report_file = os.path.join(OUTPUT_DIR, f"{FEATURE_NAME}_attempt_1.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_file}")

    return report

if __name__ == "__main__":
    report = main()
    exit(0 if report["overall_passed"] else 1)
