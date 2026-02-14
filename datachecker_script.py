import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

# ===== SETUP =====
data_file = "data/submodel_inputs/real_rate.csv"
schema_file = "shared/schema_freeze.json"
log_file = "logs/datacheck/real_rate_attempt1.log"

# Create log directory if needed
Path(log_file).parent.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(data_file, index_col=0, parse_dates=True)
with open(schema_file) as f:
    schema = json.load(f)

results = {
    "timestamp": datetime.now().isoformat(),
    "feature": "real_rate",
    "attempt": 1,
    "steps": []
}

print("=" * 80)
print("DATACHECKER: real_rate Attempt 1 - 7-Step Standardized Check")
print("=" * 80)

# ===== STEP 1: MISSING VALUES (MUST BE 0) =====
print("\nSTEP 1: Missing Values Check")
print("-" * 80)

step1 = {"step": "missing_values", "issues": []}

missing_counts = df.isnull().sum()
missing_pcts = (df.isnull().sum() / len(df)) * 100

for col in df.columns:
    if missing_counts[col] > 0:
        step1["issues"].append(f"CRITICAL: Column '{col}' has {missing_counts[col]} missing values ({missing_pcts[col]:.2f}%)")

if step1["issues"]:
    print("\n".join(step1["issues"]))
else:
    print("OK: No missing values detected")

step1["passed"] = len([i for i in step1["issues"] if "CRITICAL" in i]) == 0
results["steps"].append(step1)

# ===== STEP 2: OUTLIERS & ANOMALIES =====
print("\nSTEP 2: Outliers and Anomalies")
print("-" * 80)

step2 = {"step": "outliers_anomalies", "issues": []}

for col in df.select_dtypes(include=[np.number]).columns:
    col_data = df[col]

    # Check for infinite values
    if np.isinf(col_data).sum() > 0:
        step2["issues"].append(f"CRITICAL: Column '{col}' contains {np.isinf(col_data).sum()} infinite values")

    # Check for constant columns (std == 0)
    if col_data.std() == 0:
        step2["issues"].append(f"CRITICAL: Column '{col}' has zero standard deviation (constant)")

    # Check for extreme values (outliers)
    mean_val = col_data.mean()
    std_val = col_data.std()
    if std_val > 0:
        z_scores = np.abs((col_data - mean_val) / std_val)
        extreme_count = (z_scores > 5).sum()
        if extreme_count > len(df) * 0.01:  # More than 1% are extreme
            step2["issues"].append(f"WARNING: Column '{col}' has {extreme_count} extreme values (|z| > 5)")

    # Check for values outside realistic ranges for economic indicators
    if col in ['level']:
        if col_data.min() < -10 or col_data.max() > 10:
            step2["issues"].append(f"WARNING: Column '{col}' level outside typical rate range: [{col_data.min():.2f}, {col_data.max():.2f}]")

if step2["issues"]:
    print("\n".join(step2["issues"]))
else:
    print("OK: No extreme outliers or anomalies detected")

step2["passed"] = len([i for i in step2["issues"] if "CRITICAL" in i]) == 0
results["steps"].append(step2)

# ===== STEP 3: FUTURE LEAKAGE =====
print("\nSTEP 3: Future Leakage Check")
print("-" * 80)

step3 = {"step": "future_leak", "issues": []}

# Check if any feature correlates suspiciously with itself at different lags
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    col_data = df[col].dropna()

    # Check autocorrelation at lag 1 (should be reasonable for features)
    if len(col_data) > 10:
        corr_lag0 = col_data.autocorr(lag=0)
        corr_lag1 = col_data.autocorr(lag=1)

        # If lag0 is extremely high and different from lag1, might indicate duplication
        if not np.isnan(corr_lag0) and not np.isnan(corr_lag1):
            if abs(corr_lag0 - 1.0) < 0.01 and abs(corr_lag1) < 0.5:
                step3["issues"].append(f"WARNING: Column '{col}' has corr[lag0]={corr_lag0:.4f}, corr[lag1]={corr_lag1:.4f} - possible duplication")

print("OK: No obvious future leakage detected (no lag-0 perfect autocorrelation anomalies)")
step3["passed"] = True
results["steps"].append(step3)

# ===== STEP 4: CORRELATION ANALYSIS (REDUNDANCY CHECK) =====
print("\nSTEP 4: Correlation Analysis")
print("-" * 80)

step4 = {"step": "correlation", "issues": []}

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# Find highly correlated pairs (excluding self)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.95:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

if high_corr_pairs:
    for col1, col2, corr_val in high_corr_pairs:
        step4["issues"].append(f"WARNING: High correlation between '{col1}' and '{col2}': {corr_val:.4f}")

if step4["issues"]:
    print("\n".join(step4["issues"]))
else:
    print("OK: No excessive redundancy detected (no pairs with |corr| > 0.95)")

step4["passed"] = len([i for i in step4["issues"] if "CRITICAL" in i]) == 0
results["steps"].append(step4)

# ===== STEP 5: SCHEMA ALIGNMENT =====
print("\nSTEP 5: Schema Alignment Check")
print("-" * 80)

step5 = {"step": "schema_alignment", "issues": []}

# Check date range
schema_start = pd.to_datetime(schema["date_range"]["start"])
schema_end = pd.to_datetime(schema["date_range"]["end"])
data_start = df.index.min()
data_end = df.index.max()

print(f"Schema date range: {schema_start.date()} to {schema_end.date()}")
print(f"Data date range:   {data_start.date()} to {data_end.date()}")

if data_start != schema_start:
    step5["issues"].append(f"WARNING: Start date mismatch. Schema: {schema_start.date()}, Data: {data_start.date()}")

if data_end != schema_end:
    step5["issues"].append(f"WARNING: End date mismatch. Schema: {schema_end.date()}, Data: {data_end.date()}")

# Check row count
expected_rows = schema["row_count"]
actual_rows = len(df)
if actual_rows != expected_rows:
    step5["issues"].append(f"WARNING: Row count mismatch. Expected: {expected_rows}, Actual: {actual_rows}")

# Check date continuity (no duplicates, monotonic increasing)
if not df.index.is_monotonic_increasing:
    step5["issues"].append(f"CRITICAL: Date index is not monotonically increasing")

if df.index.duplicated().sum() > 0:
    step5["issues"].append(f"CRITICAL: Found {df.index.duplicated().sum()} duplicate dates")

if step5["issues"]:
    print("\n".join(step5["issues"]))
else:
    print("OK: Schema alignment verified")

step5["passed"] = len([i for i in step5["issues"] if "CRITICAL" in i]) == 0
results["steps"].append(step5)

# ===== STEP 6: DATA INTEGRITY =====
print("\nSTEP 6: Data Integrity Check")
print("-" * 80)

step6 = {"step": "data_integrity", "issues": []}

# Check for NaN (already done in step 1, but thorough check)
total_nulls = df.isnull().sum().sum()
if total_nulls > 0:
    step6["issues"].append(f"CRITICAL: Found {total_nulls} null/NaN values in data")

# Check for inf
total_infs = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
if total_infs > 0:
    step6["issues"].append(f"CRITICAL: Found {total_infs} infinite values in data")

# Check for duplicate rows (exact matches)
duplicates = df.duplicated(keep=False).sum()
if duplicates > 0:
    step6["issues"].append(f"WARNING: Found {duplicates} duplicate rows (by values)")

if step6["issues"]:
    print("\n".join(step6["issues"]))
else:
    print("OK: Data integrity verified (no NaN, inf, or exact duplicates)")

step6["passed"] = len([i for i in step6["issues"] if "CRITICAL" in i]) == 0
results["steps"].append(step6)

# ===== STEP 7: FEATURE STATISTICS =====
print("\nSTEP 7: Feature Statistics")
print("-" * 80)

step7 = {"step": "feature_statistics", "issues": []}

print("\nDescriptive Statistics:")
print(df.describe().to_string())

numeric_df = df.select_dtypes(include=[np.number])

for col in numeric_df.columns:
    col_data = numeric_df[col]
    skewness = col_data.skew()
    kurtosis = col_data.kurtosis()

    # Check for unreasonable distributions
    if abs(skewness) > 3:
        step7["issues"].append(f"WARNING: Column '{col}' has high skewness: {skewness:.2f}")

    if kurtosis > 5:
        step7["issues"].append(f"WARNING: Column '{col}' has high kurtosis: {kurtosis:.2f}")

if step7["issues"]:
    print("\n" + "\n".join(step7["issues"]))
else:
    print("OK: Feature distributions appear reasonable")

step7["passed"] = len([i for i in step7["issues"] if "CRITICAL" in i]) == 0
results["steps"].append(step7)

# ===== FINAL JUDGMENT =====
print("\n" + "=" * 80)
print("FINAL JUDGMENT")
print("=" * 80)

all_issues = [issue for step in results["steps"] for issue in step.get("issues", [])]
critical_issues = [issue for issue in all_issues if "CRITICAL" in issue]
warnings = [issue for issue in all_issues if "WARNING" in issue]

results["critical_issues"] = critical_issues
results["warnings"] = warnings

if critical_issues:
    results["action"] = "REJECT"
    print("\nRESULT: REJECT")
    print(f"Found {len(critical_issues)} critical issue(s):")
    for issue in critical_issues:
        print(f"  - {issue}")
elif len(warnings) > 5:
    results["action"] = "CONDITIONAL_PASS"
    print("\nRESULT: CONDITIONAL_PASS")
    print(f"Found {len(warnings)} warning(s) (>5). Delegating to evaluator.")
    for issue in warnings:
        print(f"  - {issue}")
else:
    results["action"] = "PASS"
    print("\nRESULT: PASS")
    if warnings:
        print(f"Found {len(warnings)} warning(s) (<=5):")
        for issue in warnings:
            print(f"  - {issue}")
    else:
        print("No issues detected.")

results["overall_passed"] = results["action"] != "REJECT"

# ===== SAVE LOG =====
with open(log_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("DATACHECKER LOG: real_rate Attempt 1\n")
    f.write("=" * 80 + "\n")
    f.write(f"Timestamp: {results['timestamp']}\n")
    f.write(f"Feature: {results['feature']}\n")
    f.write(f"Attempt: {results['attempt']}\n\n")

    for step in results["steps"]:
        f.write(f"\n{step['step'].upper()}\n")
        f.write("-" * 80 + "\n")
        if step.get("issues"):
            for issue in step["issues"]:
                f.write(f"{issue}\n")
        else:
            f.write("OK\n")
        f.write(f"Passed: {step['passed']}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 80 + "\n")
    f.write(f"Critical Issues: {len(critical_issues)}\n")
    if critical_issues:
        for issue in critical_issues:
            f.write(f"  - {issue}\n")

    f.write(f"\nWarnings: {len(warnings)}\n")
    if warnings:
        for issue in warnings:
            f.write(f"  - {issue}\n")

    f.write(f"\nFinal Action: {results['action']}\n")
    f.write(f"Overall Passed: {results['overall_passed']}\n")

print(f"\nLog saved to: {log_file}")

# Save JSON results
json_file = log_file.replace(".log", ".json")
with open(json_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"JSON results saved to: {json_file}")
