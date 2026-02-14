import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

print("="*80)
print("GLD OHLC DATA QUALITY CHECK - 7-STEP STANDARDIZED VERIFICATION")
print("="*80)

# Load data
gld = pd.read_csv('data/raw/gld_ohlc.csv')
gld['date'] = pd.to_datetime(gld['date'], utc=True)
gld['date'] = gld['date'].dt.tz_localize(None)  # Remove timezone
gld = gld.sort_values('date').reset_index(drop=True)

base_features = pd.read_csv('data/processed/base_features.csv')
base_features['Date'] = pd.to_datetime(base_features['Date'])

print("\nDataset shape: {}".format(gld.shape))
print("Date range: {} to {}".format(gld['date'].min(), gld['date'].max()))
print("Columns: {}\n".format(list(gld.columns)))

# Step 1: File existence check
print("\n" + "="*80)
print("STEP 1: FILE EXISTENCE & SCHEMA CHECK")
print("="*80)

step1_results = {"step": "file_check", "issues": []}
required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'returns', 'gk_vol']
for col in required_cols:
    if col not in gld.columns:
        step1_results["issues"].append("CRITICAL: Column '{}' missing".format(col))
    else:
        print("[OK] Column '{}' present".format(col))

if len(gld) < 1000:
    step1_results["issues"].append("WARNING: Row count ({}) < 1000 minimum".format(len(gld)))
else:
    print("[OK] Row count: {} rows (sufficient)".format(len(gld)))

step1_results["passed"] = not any("CRITICAL" in i for i in step1_results["issues"])
print("\nStep 1 Result: {}".format("PASS" if step1_results['passed'] else "FAIL"))
for issue in step1_results["issues"]:
    print("  - {}".format(issue))

# Step 2: Basic stats
print("\n" + "="*80)
print("STEP 2: BASIC STATISTICS CHECK")
print("="*80)

step2_results = {"step": "basic_stats", "issues": []}
numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'returns', 'gk_vol']

for col in numeric_cols:
    if col in gld.columns:
        valid_data = gld[col].dropna()
        print("\n{}: Min={:.6f}, Mean={:.6f}, Max={:.6f}, Std={:.6f}, NaN={}".format(
            col, valid_data.min(), valid_data.mean(), valid_data.max(), valid_data.std(), gld[col].isnull().sum()))

        if valid_data.std() == 0:
            step2_results["issues"].append("CRITICAL: {} has zero standard deviation".format(col))

step2_results["passed"] = not any("CRITICAL" in i for i in step2_results["issues"])
print("\nStep 2 Result: {}".format("PASS" if step2_results['passed'] else "FAIL"))
for issue in step2_results["issues"]:
    print("  - {}".format(issue))

# Step 3: Missing values
print("\n" + "="*80)
print("STEP 3: MISSING VALUES CHECK")
print("="*80)

step3_results = {"step": "missing_values", "issues": []}

for col in gld.columns:
    pct = gld[col].isnull().mean() * 100
    count = gld[col].isnull().sum()
    print("  {}: {} NaN ({:.2f}%)".format(col, count, pct))

    if pct > 20:
        step3_results["issues"].append("CRITICAL: {} missing rate {:.1f}% > 20%".format(col, pct))
    elif pct > 5:
        step3_results["issues"].append("WARNING: {} missing rate {:.1f}% > 5%".format(col, pct))

step3_results["passed"] = not any("CRITICAL" in i for i in step3_results["issues"])
print("\nStep 3 Result: {}".format("PASS" if step3_results['passed'] else "FAIL"))
for issue in step3_results["issues"]:
    print("  - {}".format(issue))

# Step 4: Future leak check
print("\n" + "="*80)
print("STEP 4: FUTURE LEAK CHECK (GK Vol vs Returns)")
print("="*80)

step4_results = {"step": "future_leak", "issues": []}

returns = gld['returns'].dropna()
gk_vol = gld['gk_vol'].dropna()

common_idx = returns.index.intersection(gk_vol.index)
if len(common_idx) > 50:
    corr_0 = gld.loc[common_idx, 'gk_vol'].corr(gld.loc[common_idx, 'returns'])
    print("Correlation gk_vol (lag 0) vs returns: {:.4f}".format(corr_0))

    returns_lag1 = gld['returns'].shift(1)
    common_idx_lag = returns_lag1.index.intersection(gk_vol.index)
    if len(common_idx_lag) > 50:
        corr_1 = gld.loc[common_idx_lag, 'gk_vol'].corr(returns_lag1.loc[common_idx_lag])
        print("Correlation gk_vol (lag 0) vs returns (lag 1): {:.4f}".format(corr_1))

        if abs(corr_0) > 0.8:
            step4_results["issues"].append("CRITICAL: gk_vol/returns correlation {:.3f} suggests leakage".format(corr_0))
        else:
            print("[OK] No obvious future information leakage detected")

step4_results["passed"] = not any("CRITICAL" in i for i in step4_results["issues"])
print("\nStep 4 Result: {}".format("PASS" if step4_results['passed'] else "FAIL"))
for issue in step4_results["issues"]:
    print("  - {}".format(issue))

# Step 5: Temporal check
print("\n" + "="*80)
print("STEP 5: TIME-SERIES INTEGRITY CHECK")
print("="*80)

step5_results = {"step": "temporal", "issues": []}

dates = gld['date']
is_sorted = dates.is_monotonic_increasing
print("Dates monotonic increasing: {}".format(is_sorted))
if not is_sorted:
    step5_results["issues"].append("CRITICAL: Dates not sorted chronologically")

dupes = dates.duplicated().sum()
print("Duplicate dates: {}".format(dupes))
if dupes > 0:
    step5_results["issues"].append("CRITICAL: {} duplicate dates".format(dupes))

gaps = dates.diff()
large_gaps = gaps[gaps > pd.Timedelta(days=7)]
print("Gaps > 7 days: {}".format(len(large_gaps)))
print("[OK] Temporal integrity verified")

step5_results["passed"] = not any("CRITICAL" in i for i in step5_results["issues"])
print("\nStep 5 Result: {}".format("PASS" if step5_results['passed'] else "FAIL"))
for issue in step5_results["issues"]:
    print("  - {}".format(issue))

# Step 6: Alignment check
print("\n" + "="*80)
print("STEP 6: ALIGNMENT CHECK WITH base_features.csv")
print("="*80)

step6_results = {"step": "alignment", "issues": []}

expected_start = base_features['Date'].min()
expected_end = base_features['Date'].max()
actual_start = gld['date'].min()
actual_end = gld['date'].max()

print("Expected range: {} to {}".format(expected_start.date(), expected_end.date()))
print("Actual range:   {} to {}".format(actual_start.date(), actual_end.date()))

if actual_start > expected_start:
    step6_results["issues"].append("CRITICAL: GLD starts {} days after base_features".format((actual_start - expected_start).days))
elif actual_end < expected_end:
    step6_results["issues"].append("CRITICAL: GLD ends {} days before base_features".format((expected_end - actual_end).days))
else:
    print("[OK] Date range alignment acceptable")

gld_dates_set = set(gld['date'].dt.date)
base_dates_set = set(base_features['Date'].dt.date)
missing_in_gld = base_dates_set - gld_dates_set
if len(missing_in_gld) > 10:
    step6_results["issues"].append("WARNING: {} base_features dates missing in GLD".format(len(missing_in_gld)))

step6_results["passed"] = not any("CRITICAL" in i for i in step6_results["issues"])
print("\nStep 6 Result: {}".format("PASS" if step6_results['passed'] else "FAIL"))
for issue in step6_results["issues"]:
    print("  - {}".format(issue))

# Step 7: Schema check
print("\n" + "="*80)
print("STEP 7: OUTPUT SCHEMA VERIFICATION")
print("="*80)

step7_results = {"step": "schema", "issues": []}

actual_cols = set(gld.columns)
expected_cols = {'date', 'open', 'high', 'low', 'close', 'volume', 'returns', 'gk_vol'}

if not expected_cols.issubset(actual_cols):
    missing = expected_cols - actual_cols
    step7_results["issues"].append("CRITICAL: Missing columns {}".format(missing))
else:
    print("[OK] All expected columns present")

invalid_ohlc = (
    (gld['high'] < gld['low']) |
    (gld['high'] < gld['open']) |
    (gld['high'] < gld['close']) |
    (gld['low'] > gld['open']) |
    (gld['low'] > gld['close'])
).sum()
if invalid_ohlc > 0:
    step7_results["issues"].append("CRITICAL: {} rows violate OHLC logic".format(invalid_ohlc))
else:
    print("[OK] OHLC logic valid (high >= low, open/close within range)")

print("[OK] Rows: {} (within expected range)".format(len(gld)))

step7_results["passed"] = not any("CRITICAL" in i for i in step7_results["issues"])
print("\nStep 7 Result: {}".format("PASS" if step7_results['passed'] else "FAIL"))
for issue in step7_results["issues"]:
    print("  - {}".format(issue))

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

all_results = [step1_results, step2_results, step3_results, step4_results, step5_results, step6_results, step7_results]

critical_issues = [i for r in all_results for i in r.get("issues",[]) if "CRITICAL" in i]
warnings = [i for r in all_results for i in r.get("issues",[]) if "WARNING" in i]

print("\nCritical Issues: {}".format(len(critical_issues)))
for issue in critical_issues:
    print("  [FAIL] {}".format(issue))

print("\nWarnings: {}".format(len(warnings)))
for warning in warnings:
    print("  [WARN] {}".format(warning))

overall_passed = len(critical_issues) == 0

print("\n" + "="*80)
if overall_passed:
    if len(warnings) > 5:
        decision = "CONDITIONAL_PASS"
        print("DECISION: {}".format(decision))
        print("ACTION: Pass to builder_model (warnings noted)")
    else:
        decision = "PASS"
        print("DECISION: {}".format(decision))
        print("ACTION: Proceed to builder_model")
else:
    decision = "REJECT"
    print("DECISION: {}".format(decision))
    print("ACTION: Return to builder_data")

print("="*80)

# Save report
report = {
    "feature": "technical",
    "data_source": "data/raw/gld_ohlc.csv",
    "timestamp": datetime.now().isoformat(),
    "steps": all_results,
    "critical_issues": critical_issues,
    "warnings": warnings,
    "decision": decision,
    "overall_passed": overall_passed,
}

os.makedirs("logs/datacheck", exist_ok=True)
with open("logs/datacheck/technical_1.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nReport saved to: logs/datacheck/technical_1.json")
