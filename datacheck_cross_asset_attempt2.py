import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os

# ===== LOAD DATA =====
df = pd.read_csv('data/processed/cross_asset_features.csv', parse_dates=['Date'], index_col='Date')

print("="*80)
print("DATACHECK: CROSS_ASSET_FEATURES (Attempt 2)")
print("="*80)
print(f"\nData shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Columns: {list(df.columns)}\n")

all_results = []

# ===== STEP 1: FILE STRUCTURE & BASIC INFO =====
print("\n" + "="*80)
print("STEP 1: FILE STRUCTURE & BASIC INFO")
print("="*80)

step1_issues = []

# Check expected columns
expected_cols = ['xasset_regime_prob', 'xasset_recession_signal', 'xasset_divergence']
for col in expected_cols:
    if col not in df.columns:
        step1_issues.append(f"CRITICAL: Missing expected column '{col}'")

if len(df) < 2000:
    step1_issues.append(f"WARNING: Row count ({len(df)}) is below 2000")

if df.index.has_duplicates:
    step1_issues.append(f"CRITICAL: {df.index.duplicated().sum()} duplicate dates found")

step1_result = {
    "step": "file_structure",
    "issues": step1_issues,
    "row_count": len(df),
    "column_count": len(df.columns),
    "date_range": f"{df.index.min()} to {df.index.max()}",
    "passed": not any("CRITICAL" in i for i in step1_issues)
}

print(f"Row count: {len(df)}")
print(f"Column count: {len(df.columns)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Duplicates: {df.index.duplicated().sum()}")
print(f"Status: {'PASS' if step1_result['passed'] else 'FAIL'}")
for issue in step1_issues:
    print(f"  - {issue}")

all_results.append(step1_result)

# ===== STEP 2: MISSING VALUES & NaN =====
print("\n" + "="*80)
print("STEP 2: MISSING VALUES & NaN")
print("="*80)

step2_issues = []

for col in df.columns:
    missing_count = df[col].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    if missing_count > 0:
        print(f"  {col}: {missing_count} missing ({missing_pct:.2f}%)")
        if missing_pct > 20:
            step2_issues.append(f"CRITICAL: {col} has {missing_pct:.1f}% missing values")
        elif missing_pct > 5:
            step2_issues.append(f"WARNING: {col} has {missing_pct:.1f}% missing values")
    else:
        print(f"  {col}: 0 missing")

# Check for all-NaN rows
all_nan_rows = df.isna().all(axis=1).sum()
if all_nan_rows > 0:
    step2_issues.append(f"CRITICAL: {all_nan_rows} all-NaN rows found")

step2_result = {
    "step": "missing_values",
    "issues": step2_issues,
    "passed": not any("CRITICAL" in i for i in step2_issues)
}

print(f"Status: {'PASS' if step2_result['passed'] else 'FAIL'}")
for issue in step2_issues:
    print(f"  - {issue}")

all_results.append(step2_result)

# ===== STEP 3: OUTLIERS & DISTRIBUTION =====
print("\n" + "="*80)
print("STEP 3: OUTLIERS & DISTRIBUTION")
print("="*80)

step3_issues = []

for col in df.columns:
    data = df[col].dropna()
    mean = data.mean()
    std = data.std()

    if std == 0:
        step3_issues.append(f"CRITICAL: {col} has std=0 (constant)")
    else:
        # Check for extreme outliers (>5 std deviations)
        extreme_count = ((np.abs(data - mean) > 5 * std).sum())
        if extreme_count > len(data) * 0.05:  # >5% extreme outliers
            step3_issues.append(f"WARNING: {col} has {extreme_count} extreme outliers (>5σ)")

        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()

        print(f"  {col}:")
        print(f"    Mean: {mean:.6f}, Std: {std:.6f}")
        print(f"    Min: {data.min():.6f}, Max: {data.max():.6f}")
        print(f"    IQR outliers: {outlier_count} ({outlier_count/len(data)*100:.2f}%)")

step3_result = {
    "step": "outliers",
    "issues": step3_issues,
    "passed": not any("CRITICAL" in i for i in step3_issues)
}

print(f"Status: {'PASS' if step3_result['passed'] else 'FAIL'}")
for issue in step3_issues:
    print(f"  - {issue}")

all_results.append(step3_result)

# ===== STEP 4: AUTOCORRELATION (CRITICAL FOR ATTEMPT 2) =====
print("\n" + "="*80)
print("STEP 4: AUTOCORRELATION (Critical - Attempt 2 Focus)")
print("="*80)

step4_issues = []

for col in df.columns:
    data = df[col].dropna()
    # Lag-1 autocorrelation
    lag1_corr = data.autocorr(lag=1)

    print(f"  {col}: Lag-1 ACF = {lag1_corr:.6f}")

    if np.isnan(lag1_corr):
        step4_issues.append(f"WARNING: {col} has NaN autocorrelation")
    elif abs(lag1_corr) > 0.99:
        step4_issues.append(f"CRITICAL: {col} autocorrelation {lag1_corr:.6f} > 0.99 (too persistent)")
    elif abs(lag1_corr) > 0.95:
        step4_issues.append(f"WARNING: {col} autocorrelation {lag1_corr:.6f} > 0.95 (high persistence)")

step4_result = {
    "step": "autocorrelation",
    "issues": step4_issues,
    "passed": not any("CRITICAL" in i for i in step4_issues)
}

print(f"Status: {'PASS' if step4_result['passed'] else 'FAIL'}")
for issue in step4_issues:
    print(f"  - {issue}")

all_results.append(step4_result)

# ===== STEP 5: FUTURE LEAK CHECK =====
print("\n" + "="*80)
print("STEP 5: FUTURE LEAK CHECK")
print("="*80)

step5_issues = []

# Check if target variable exists
if os.path.exists('data/processed/target.csv'):
    target = pd.read_csv('data/processed/target.csv', parse_dates=['Date'], index_col='Date')

    # Merge on dates
    merged = pd.merge(df, target, left_index=True, right_index=True, how='inner')
    target_col = merged.columns[-1]  # Assume last col is target

    print(f"Target column found: {target_col}")
    print(f"Merged data shape: {merged.shape}")

    for col in df.columns:
        if col in merged.columns:
            # Lag-0 correlation
            corr0 = merged[col].corr(merged[target_col])
            # Lag-1 correlation (feature lagged)
            corr1 = merged[col].shift(1).corr(merged[target_col])

            print(f"  {col}: corr(lag0)={corr0:.4f}, corr(lag1)={corr1:.4f}")

            if abs(corr0) > 0.8:
                step5_issues.append(f"CRITICAL: {col} has high target correlation {corr0:.4f} (leak suspected)")
            elif abs(corr0) > 0.5 and abs(corr1) < 0.2:
                step5_issues.append(f"WARNING: {col} high lag0 ({corr0:.4f}) vs low lag1 ({corr1:.4f}) - leak indicator")
else:
    print("  Target file not found - skipping leak check")
    step5_issues.append("INFO: Target file not available for leak check")

step5_result = {
    "step": "future_leak",
    "issues": step5_issues,
    "passed": not any("CRITICAL" in i for i in step5_issues)
}

print(f"Status: {'PASS' if step5_result['passed'] else 'FAIL'}")
for issue in step5_issues:
    print(f"  - {issue}")

all_results.append(step5_result)

# ===== STEP 6: TEMPORAL ALIGNMENT =====
print("\n" + "="*80)
print("STEP 6: TEMPORAL ALIGNMENT")
print("="*80)

step6_issues = []

dates = df.index
is_sorted = dates.is_monotonic_increasing
print(f"  Monotonic increasing: {is_sorted}")

if not is_sorted:
    step6_issues.append("CRITICAL: Dates are not sorted in ascending order")

# Check for gaps > 7 days
date_diffs = dates.to_series().diff().dt.days
gaps = date_diffs[date_diffs > 7]
if len(gaps) > 0:
    print(f"  Gaps > 7 days: {len(gaps)} instances")
    for gap_date, gap_days in gaps.items():
        if gap_days > 30:
            step6_issues.append(f"WARNING: {gap_days}-day gap at {gap_date}")

# Market closed (3-day gaps) are normal, check for extreme gaps
extreme_gaps = date_diffs[date_diffs > 60]
if len(extreme_gaps) > 0:
    step6_issues.append(f"WARNING: {len(extreme_gaps)} gaps > 60 days found")

step6_result = {
    "step": "temporal_alignment",
    "issues": step6_issues,
    "passed": not any("CRITICAL" in i for i in step6_issues)
}

print(f"Status: {'PASS' if step6_result['passed'] else 'FAIL'}")
for issue in step6_issues:
    print(f"  - {issue}")

all_results.append(step6_result)

# ===== STEP 7: OUTPUT FORMAT VALIDATION =====
print("\n" + "="*80)
print("STEP 7: OUTPUT FORMAT VALIDATION")
print("="*80)

step7_issues = []

# Check dtypes
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        step7_issues.append(f"CRITICAL: {col} is not numeric type (type={df[col].dtype})")
    else:
        print(f"  {col}: dtype={df[col].dtype}")

# Check index
if not isinstance(df.index, pd.DatetimeIndex):
    step7_issues.append("CRITICAL: Index is not DatetimeIndex")
else:
    print(f"  Index: DatetimeIndex (OK)")

# Check for NaN in numeric columns
for col in df.columns:
    if df[col].isna().any():
        step7_issues.append(f"WARNING: {col} contains NaN values")

step7_result = {
    "step": "output_format",
    "issues": step7_issues,
    "passed": not any("CRITICAL" in i for i in step7_issues)
}

print(f"Status: {'PASS' if step7_result['passed'] else 'FAIL'}")
for issue in step7_issues:
    print(f"  - {issue}")

all_results.append(step7_result)

# ===== FINAL REPORT =====
print("\n" + "="*80)
print("FINAL REPORT")
print("="*80)

critical_issues = [i for r in all_results for i in r.get("issues",[]) if "CRITICAL" in i]
warnings = [i for r in all_results for i in r.get("issues",[]) if "WARNING" in i]
info_msgs = [i for r in all_results for i in r.get("issues",[]) if "INFO" in i]

print(f"\nCritical Issues: {len(critical_issues)}")
for issue in critical_issues:
    print(f"  ✗ {issue}")

print(f"\nWarnings: {len(warnings)}")
for issue in warnings[:5]:  # Show first 5
    print(f"  ⚠ {issue}")
if len(warnings) > 5:
    print(f"  ... and {len(warnings)-5} more")

print(f"\nInfo: {len(info_msgs)}")
for msg in info_msgs:
    print(f"  ℹ {msg}")

# Determine pass/fail
if critical_issues:
    action = "REJECT"
elif len(warnings) > 5:
    action = "CONDITIONAL_PASS"
else:
    action = "PASS"

print(f"\n{'='*80}")
print(f"DECISION: {action}")
print(f"{'='*80}\n")

# Save report
report = {
    "feature": "cross_asset",
    "attempt": 2,
    "timestamp": datetime.now().isoformat(),
    "steps": all_results,
    "critical_issues": critical_issues,
    "warnings": warnings,
    "action": action,
    "summary": {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "date_range": f"{df.index.min()} to {df.index.max()}",
        "critical_count": len(critical_issues),
        "warning_count": len(warnings)
    }
}

# Ensure directory exists
os.makedirs('logs/datacheck', exist_ok=True)

# Save report
with open('logs/datacheck/cross_asset_attempt_2.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"Report saved to: logs/datacheck/cross_asset_attempt_2.json")
