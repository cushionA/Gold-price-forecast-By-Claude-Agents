import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

# ===== LOAD DATA =====
df_ie = pd.read_csv('data/processed/inflation_expectation/features_input.csv', index_col=0)
df_ie.index = pd.to_datetime(df_ie.index)
df_ie = df_ie.sort_index()

df_base = pd.read_csv('data/processed/base_features.csv')
df_base['Date'] = pd.to_datetime(df_base['Date'])
df_base = df_base.sort_values('Date').set_index('Date')

# ===== INITIALIZE REPORT =====
report = {
    "feature": "inflation_expectation",
    "attempt": 1,
    "timestamp": datetime.now().isoformat(),
    "data_file": "data/processed/inflation_expectation/features_input.csv",
    "steps": [],
    "critical_issues": [],
    "warnings": [],
    "action": "PENDING",
    "overall_passed": False
}

# ===== STEP 1: FILE CHECK =====
step1 = {"step": "file_check", "issues": []}
required_files = ["data/processed/inflation_expectation/features_input.csv"]
for f in required_files:
    if os.path.exists(f):
        step1["issues"].append(f"OK: {f} exists")
    else:
        step1["issues"].append(f"CRITICAL: {f} missing")
report["steps"].append(step1)

# ===== STEP 2: BASIC STATISTICS =====
step2 = {"step": "basic_stats", "issues": []}
step2["issues"].append(f"OK: Row count = {len(df_ie)} rows")

# Check for low row count
if len(df_ie) < 1000:
    step2["issues"].append(f"WARNING: Row count {len(df_ie)} is less than 1000")

# Check numeric columns for extreme values and zero variance
for col in df_ie.select_dtypes(include=[np.number]).columns:
    std_val = df_ie[col].std()
    min_val = df_ie[col].min()
    max_val = df_ie[col].max()

    if std_val == 0:
        step2["issues"].append(f"CRITICAL: {col} has zero standard deviation")
    elif abs(min_val) > 1e6 or abs(max_val) > 1e6:
        step2["issues"].append(f"WARNING: {col} has extreme values (min={min_val:.2e}, max={max_val:.2e})")
    else:
        step2["issues"].append(f"OK: {col} valid (std={std_val:.4f})")

report["steps"].append(step2)

# ===== STEP 3: MISSING VALUES =====
step3 = {"step": "missing_values", "issues": []}
for col in df_ie.columns:
    pct = df_ie[col].isnull().sum() / len(df_ie) * 100
    if pct > 20:
        step3["issues"].append(f"CRITICAL: {col} missing {pct:.1f}%")
    elif pct > 5:
        step3["issues"].append(f"WARNING: {col} missing {pct:.1f}%")
    elif pct > 0:
        step3["issues"].append(f"OK: {col} missing {pct:.1f}%")
    else:
        step3["issues"].append(f"OK: {col} is complete")

# Check for consecutive NaN sequences
for col in df_ie.select_dtypes(include=[np.number]).columns:
    # Only check if there are any NaN
    if df_ie[col].isnull().sum() > 0:
        max_consec = df_ie[col].isnull().astype(int).groupby(
            df_ie[col].notnull().astype(int).cumsum()
        ).sum().max()
        if max_consec > 10:
            step3["issues"].append(f"WARNING: {col} has {max_consec} consecutive NaN values")

report["steps"].append(step3)

# ===== STEP 4: FUTURE LEAK CHECK =====
step4 = {"step": "future_leak", "issues": []}
if 'gold_return' in df_ie.columns:
    target = 'gold_return'
    for col in df_ie.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue
        corr0 = df_ie[col].corr(df_ie[target])
        corr1 = df_ie[col].shift(1).corr(df_ie[target])

        if abs(corr0) > 0.8:
            step4["issues"].append(f"CRITICAL: {col} vs target corr={corr0:.3f} (leak suspected)")
        elif abs(corr0) > abs(corr1) * 3 and abs(corr0) > 0.3:
            step4["issues"].append(f"WARNING: {col} lag0 corr({corr0:.3f}) >> lag1 corr({corr1:.3f})")
        else:
            step4["issues"].append(f"OK: {col} vs target corr={corr0:.3f}")
else:
    step4["issues"].append("OK: No target column (gold_return) to check for leaks")

report["steps"].append(step4)

# ===== STEP 5: AUTOCORRELATION CHECK =====
step5 = {"step": "autocorrelation", "issues": []}
for col in df_ie.select_dtypes(include=[np.number]).columns:
    autocorr = df_ie[col].autocorr(lag=1)
    if pd.isna(autocorr):
        step5["issues"].append(f"WARNING: {col} autocorr is NaN")
    elif autocorr >= 0.99:
        step5["issues"].append(f"CRITICAL: {col} autocorr={autocorr:.6f} (>= 0.99)")
    elif autocorr >= 0.95:
        step5["issues"].append(f"WARNING: {col} autocorr={autocorr:.6f} (high, near 0.95)")
    else:
        step5["issues"].append(f"OK: {col} autocorr={autocorr:.6f}")

report["steps"].append(step5)

# ===== STEP 6: TEMPORAL INTEGRITY CHECK =====
step6 = {"step": "temporal_integrity", "issues": []}
dates = df_ie.index.to_series()
is_sorted = dates.is_monotonic_increasing
duplicates = dates.duplicated().sum()
gaps = dates.diff()
large_gaps = gaps[gaps > pd.Timedelta(days=7)]

if not is_sorted:
    step6["issues"].append("CRITICAL: Dates are not monotonically increasing")
else:
    step6["issues"].append("OK: Dates are monotonically increasing")

if duplicates > 0:
    step6["issues"].append(f"CRITICAL: {duplicates} duplicate dates found")
else:
    step6["issues"].append("OK: No duplicate dates")

if len(large_gaps) > 0:
    for date, gap in large_gaps.items():
        step6["issues"].append(f"WARNING: Gap of {gap.days} days on {date}")
else:
    step6["issues"].append("OK: No gaps > 7 days")

# Check date range
expected_start = pd.to_datetime('2015-01-30')
expected_end = pd.to_datetime('2025-02-12')
if df_ie.index.min() >= expected_start and df_ie.index.max() <= expected_end:
    step6["issues"].append(f"OK: Date range matches design")
else:
    step6["issues"].append(f"WARNING: Date range {df_ie.index.min()} to {df_ie.index.max()} differs from design")

report["steps"].append(step6)

# ===== STEP 7: DATA INTEGRITY vs BASE_FEATURES =====
step7 = {"step": "data_integrity", "issues": []}

# Check date alignment
overlap = df_ie.index.intersection(df_base.index)
if len(overlap) > 0:
    step7["issues"].append(f"OK: {len(overlap)} rows overlap with base_features")
else:
    step7["issues"].append("CRITICAL: No date overlap with base_features")

# Check column presence
if 'T10YIE' in df_ie.columns and 'ie_change' in df_ie.columns:
    step7["issues"].append("OK: T10YIE and ie_change columns present")
else:
    step7["issues"].append("WARNING: Expected columns missing")

# Check for NaN in critical columns
critical_cols = ['ie_vol_5d', 'ie_vol_10d', 'ie_vol_20d'] if 'ie_vol_5d' in df_ie.columns else []
for col in critical_cols:
    nan_count = df_ie[col].isnull().sum()
    if nan_count == len(df_ie):
        step7["issues"].append(f"CRITICAL: {col} is entirely NaN")
    elif nan_count > 0:
        step7["issues"].append(f"OK: {col} has {nan_count} NaN values (expected warmup period)")
    else:
        step7["issues"].append(f"OK: {col} is complete")

report["steps"].append(step7)

# ===== DETERMINE ACTION =====
all_issues = []
for step in report["steps"]:
    all_issues.extend(step.get("issues", []))

critical_count = sum(1 for issue in all_issues if "CRITICAL" in issue)
warning_count = sum(1 for issue in all_issues if "WARNING" in issue)

report["critical_issues"] = [issue for issue in all_issues if "CRITICAL" in issue]
report["warnings"] = [issue for issue in all_issues if "WARNING" in issue]

if critical_count > 0:
    report["action"] = "REJECT"
    report["reason"] = f"CRITICAL issues detected ({critical_count}). Return to builder_data."
    report["overall_passed"] = False
elif warning_count > 5:
    report["action"] = "CONDITIONAL_PASS"
    report["reason"] = f"No critical issues but {warning_count} warnings. Evaluator decides."
    report["overall_passed"] = True
else:
    report["action"] = "PASS"
    report["reason"] = "All checks passed."
    report["overall_passed"] = True

# ===== SAVE REPORT =====
os.makedirs("logs/datacheck", exist_ok=True)
with open("logs/datacheck/inflation_expectation_attempt_1.json", "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print("\n" + "="*80)
print(f"FINAL DECISION: {report['action']}")
print("="*80)
