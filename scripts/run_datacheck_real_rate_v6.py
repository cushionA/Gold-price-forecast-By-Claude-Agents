"""
7-step data check for real_rate attempt 6
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

BASE = "C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents"

df = pd.read_csv(f"{BASE}/data/processed/real_rate_v6_input.csv", parse_dates=["date"])
df = df.set_index("date")

target_df = pd.read_csv(f"{BASE}/data/processed/target.csv", parse_dates=["Date"])
target_df = target_df.set_index("Date")

all_results = []

# ================================================================
# STEP 1: Schema Compliance
# ================================================================
step1 = {"step": "schema_compliance", "issues": []}
expected_cols = {"date", "dfii10", "dfii10_change"}
actual_cols = set(df.reset_index().columns)
missing_cols = expected_cols - actual_cols
extra_cols = actual_cols - expected_cols
if missing_cols:
    step1["issues"].append(f"CRITICAL: Missing columns: {list(missing_cols)}")
if extra_cols:
    step1["issues"].append(f"WARNING: Extra columns: {list(extra_cols)}")

min_date = df.index.min()
max_date = df.index.max()
if min_date > pd.Timestamp("2015-01-30"):
    step1["issues"].append(f"CRITICAL: Start date {min_date.date()} is after expected 2015-01-30")
if max_date < pd.Timestamp("2025-02-12"):
    step1["issues"].append(f"CRITICAL: End date {max_date.date()} is before expected 2025-02-12")

schema_rows = df[(df.index >= "2015-01-30") & (df.index <= "2025-02-12")].shape[0]
if schema_rows < 2500:
    step1["issues"].append(f"CRITICAL: Only {schema_rows} rows in schema range (need >= 2500)")

step1["stats"] = {
    "total_rows": int(df.shape[0]),
    "schema_range_rows": int(schema_rows),
    "start_date": str(min_date.date()),
    "end_date": str(max_date.date()),
    "columns": list(df.reset_index().columns)
}
step1["passed"] = not any("CRITICAL" in i for i in step1["issues"])
all_results.append(step1)

# ================================================================
# STEP 2: Missing Values
# ================================================================
step2 = {"step": "missing_values", "issues": []}
missing_stats = {}
for col in df.columns:
    n_null = int(df[col].isnull().sum())
    pct = float(n_null / len(df) * 100)
    missing_stats[col] = {"count": n_null, "pct": round(pct, 2)}
    if pct > 20:
        step2["issues"].append(f"CRITICAL: {col} missing rate {pct:.1f}%")
    elif pct > 5:
        step2["issues"].append(f"WARNING: {col} missing rate {pct:.1f}%")

consec_stats = {}
for col in df.select_dtypes(include=[np.number]).columns:
    null_vals = df[col].isnull().astype(int).values
    max_consec = curr = 0
    for v in null_vals:
        if v == 1:
            curr += 1
            max_consec = max(max_consec, curr)
        else:
            curr = 0
    consec_stats[col] = int(max_consec)
    if max_consec > 10:
        step2["issues"].append(f"WARNING: {col} has {max_consec} consecutive NaN")

step2["stats"] = {"missing_by_column": missing_stats, "max_consecutive_nan": consec_stats}
step2["passed"] = not any("CRITICAL" in i for i in step2["issues"])
all_results.append(step2)

# ================================================================
# STEP 3: Outliers
# ================================================================
step3 = {"step": "outliers", "issues": []}
dfii10_min = float(df["dfii10"].dropna().min())
dfii10_max = float(df["dfii10"].dropna().max())
chg_min = float(df["dfii10_change"].dropna().min())
chg_max = float(df["dfii10_change"].dropna().max())

n_outliers_level = int(((df["dfii10"] < -2.0) | (df["dfii10"] > 3.5)).sum())
n_outliers_change = int(((df["dfii10_change"] < -0.5) | (df["dfii10_change"] > 0.5)).sum())

if n_outliers_level > 0:
    step3["issues"].append(f"CRITICAL: {n_outliers_level} dfii10 values outside [-2.0, 3.5]")
if n_outliers_change > 0:
    step3["issues"].append(f"WARNING: {n_outliers_change} dfii10_change values outside [-0.5, 0.5]")

step3["stats"] = {
    "dfii10_actual_range": [dfii10_min, dfii10_max],
    "dfii10_expected_range": [-2.0, 3.5],
    "dfii10_outliers": n_outliers_level,
    "dfii10_change_actual_range": [round(chg_min, 4), round(chg_max, 4)],
    "dfii10_change_expected_range": [-0.5, 0.5],
    "dfii10_change_outliers": n_outliers_change
}
step3["passed"] = not any("CRITICAL" in i for i in step3["issues"])
all_results.append(step3)

# ================================================================
# STEP 4: Future Leak
# ================================================================
step4 = {"step": "future_leak", "issues": []}
n_future = int((df.index > pd.Timestamp("2026-02-17")).sum())
if n_future > 0:
    step4["issues"].append(f"CRITICAL: {n_future} rows have future dates beyond today")

merged = df[["dfii10_change"]].join(target_df[["gold_return_next"]], how="inner").dropna()
corr_lag0 = float(merged["dfii10_change"].corr(merged["gold_return_next"]))
corr_lag1 = float(merged["dfii10_change"].shift(1).corr(merged["gold_return_next"]))

if abs(corr_lag0) > 0.8:
    step4["issues"].append(f"CRITICAL: dfii10_change vs target corr {corr_lag0:.4f} > 0.8 (LEAK)")
elif abs(corr_lag0) > abs(corr_lag1) * 3 and abs(corr_lag0) > 0.3:
    step4["issues"].append(f"WARNING: dfii10_change lag0 corr ({corr_lag0:.4f}) >> lag1 ({corr_lag1:.4f})")

step4["stats"] = {
    "future_dates_count": n_future,
    "dfii10_change_corr_lag0": round(corr_lag0, 4),
    "dfii10_change_corr_lag1": round(corr_lag1, 4),
    "merged_rows": int(len(merged))
}
step4["passed"] = not any("CRITICAL" in i for i in step4["issues"])
all_results.append(step4)

# ================================================================
# STEP 5: Autocorrelation
# ================================================================
step5 = {"step": "autocorrelation", "issues": []}
ac_dfii10 = float(df["dfii10"].dropna().autocorr(lag=1))
ac_change = float(df["dfii10_change"].dropna().autocorr(lag=1))

if ac_dfii10 < 0.99:
    step5["issues"].append(f"WARNING: dfii10 lag-1 autocorr {ac_dfii10:.4f} below 0.99 threshold")
if abs(ac_change) > 0.15:
    step5["issues"].append(f"WARNING: dfii10_change lag-1 autocorr {ac_change:.4f} above 0.15 threshold")

step5["stats"] = {
    "dfii10_lag1_autocorr": round(ac_dfii10, 4),
    "dfii10_change_lag1_autocorr": round(ac_change, 4)
}
step5["passed"] = not any("CRITICAL" in i for i in step5["issues"])
all_results.append(step5)

# ================================================================
# STEP 6: Correlation with Target
# ================================================================
step6 = {"step": "target_correlation", "issues": []}
if corr_lag0 > 0:
    step6["issues"].append(
        f"WARNING: dfii10_change vs gold_return_next is positive ({corr_lag0:.4f}), expected negative"
    )

step6["stats"] = {
    "dfii10_change_vs_gold_return_corr": round(corr_lag0, 4),
    "direction": "negative (expected)" if corr_lag0 < 0 else "positive (unexpected)"
}
step6["passed"] = not any("CRITICAL" in i for i in step6["issues"])
all_results.append(step6)

# ================================================================
# STEP 7: Integrity
# ================================================================
step7 = {"step": "integrity", "issues": []}
is_mono = bool(df.index.is_monotonic_increasing)
n_dupes = int(df.index.duplicated().sum())

if not is_mono:
    step7["issues"].append("CRITICAL: Date index not monotonically increasing")
if n_dupes > 0:
    step7["issues"].append(f"CRITICAL: {n_dupes} duplicate dates found")

expected_change = df["dfii10"].diff()
diff_check = (df["dfii10_change"].dropna() - expected_change.dropna()).abs()
max_diff = float(diff_check.max())
n_mismatch = int((diff_check > 1e-9).sum())
if n_mismatch > 0:
    step7["issues"].append(
        f"WARNING: {n_mismatch} rows where dfii10_change != dfii10.diff() (max deviation: {max_diff:.2e})"
    )

dates = df.index.to_series()
gaps = dates.diff()
large_gaps = gaps[gaps > pd.Timedelta(days=7)]
for date, gap in large_gaps.items():
    step7["issues"].append(f"WARNING: {date.date()} has {gap.days}-day gap")

step7["stats"] = {
    "monotonically_increasing": is_mono,
    "duplicate_dates": n_dupes,
    "dfii10_change_math_consistency_max_deviation": max_diff,
    "dfii10_change_mismatches": n_mismatch,
    "gaps_over_7_days": int(len(large_gaps))
}
step7["passed"] = not any("CRITICAL" in i for i in step7["issues"])
all_results.append(step7)

# ================================================================
# Generate Report
# ================================================================
critical_issues = [i for r in all_results for i in r.get("issues", []) if "CRITICAL" in i]
warnings = [i for r in all_results for i in r.get("issues", []) if "WARNING" in i]

if critical_issues:
    action = "REJECT"
elif len(warnings) > 5:
    action = "CONDITIONAL_PASS"
else:
    action = "PASS"

report = {
    "feature": "real_rate",
    "attempt": 6,
    "timestamp": datetime.now().isoformat(),
    "steps": all_results,
    "critical_issues": critical_issues,
    "warnings": warnings,
    "action": action,
    "overall_passed": action != "REJECT"
}

os.makedirs(f"{BASE}/logs/datacheck", exist_ok=True)
report_path = f"{BASE}/logs/datacheck/real_rate_v6_check.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"Report saved to: {report_path}")
print()
print("=== SUMMARY ===")
print(f"Action: {action}")
print(f"Critical issues: {len(critical_issues)}")
print(f"Warnings: {len(warnings)}")
for s in all_results:
    status = "PASS" if s["passed"] else "FAIL"
    print(f"  {s['step']}: {status}")
if warnings:
    print("Warnings:")
    for w in warnings:
        print(f"  {w}")
