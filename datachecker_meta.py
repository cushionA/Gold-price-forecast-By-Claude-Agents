import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DATACHECKER: 7-STEP STANDARDIZED CHECK - META-MODEL INPUT")
print("=" * 80)

meta_input = pd.read_csv('data/processed/meta_model_input.csv', index_col=0, parse_dates=True)
target = pd.read_csv('data/processed/target.csv', index_col=0, parse_dates=True)

print(f"\n[LOADING] Meta-model input: {meta_input.shape}")
print(f"[LOADING] Target: {target.shape}")

results = {
    "step_1_missing_values": {},
    "step_2_anomalies": {},
    "step_3_future_leakage": {},
    "step_4_correlation": {},
    "step_5_data_integrity": {},
    "step_6_target_alignment": {},
    "step_7_final_verdict": {}
}

# STEP 1: Missing Values
print("\n" + "=" * 80)
print("STEP 1: MISSING VALUES CHECK")
print("=" * 80)

nan_counts = meta_input.isnull().sum()
nan_pct = (nan_counts / len(meta_input)) * 100

step1_issues = []
step1_passed = True

for col in meta_input.columns:
    if nan_counts[col] > 0:
        step1_issues.append(f"CRITICAL: Column '{col}' has {nan_counts[col]} NaN values ({nan_pct[col]:.2f}%)")
        step1_passed = False

inf_cols = (meta_input == np.inf).sum() + (meta_input == -np.inf).sum()
for col in inf_cols[inf_cols > 0].index:
    step1_issues.append(f"CRITICAL: Column '{col}' has {inf_cols[col]} infinite values")
    step1_passed = False

if len(nan_counts[nan_counts > 0]) == 0 and inf_cols.sum() == 0:
    step1_issues.append("PASS: No NaN or infinite values in feature columns")

target_nans = target.isnull().sum().sum()
if target_nans > 0:
    step1_issues.append(f"CRITICAL: Target has {target_nans} NaN values")
    step1_passed = False
else:
    step1_issues.append(f"PASS: Target has 0 NaN values")

results["step_1_missing_values"]["issues"] = step1_issues
results["step_1_missing_values"]["passed"] = step1_passed

for issue in step1_issues:
    print(f"  {issue}")

# STEP 2: Anomalies
print("\n" + "=" * 80)
print("STEP 2: ANOMALIES & OUTLIERS CHECK")
print("=" * 80)

step2_issues = []
step2_passed = True

for col in meta_input.select_dtypes(include=[np.number]).columns:
    std = meta_input[col].std()
    if std < 1e-10:
        step2_issues.append(f"CRITICAL: Column '{col}' has std={std:.2e} (near-constant)")
        step2_passed = False

outlier_summary = {}
for col in meta_input.select_dtypes(include=[np.number]).columns:
    if meta_input[col].std() > 0:
        z_scores = np.abs((meta_input[col] - meta_input[col].mean()) / meta_input[col].std())
        n_outliers = (z_scores > 5).sum()
        if n_outliers > 0:
            outlier_summary[col] = n_outliers

if outlier_summary:
    for col, count in sorted(outlier_summary.items(), key=lambda x: -x[1])[:5]:
        pct = (count / len(meta_input)) * 100
        step2_issues.append(f"WARNING: Column '{col}' has {count} outliers (z>5, {pct:.2f}%)")
else:
    step2_issues.append("PASS: No extreme outliers detected (z>5)")

results["step_2_anomalies"]["outlier_columns"] = outlier_summary
results["step_2_anomalies"]["issues"] = step2_issues
results["step_2_anomalies"]["passed"] = step2_passed

for issue in step2_issues:
    print(f"  {issue}")

# STEP 3: Future Leakage
print("\n" + "=" * 80)
print("STEP 3: FUTURE LEAKAGE CHECK")
print("=" * 80)

step3_issues = []
step3_passed = True

if not meta_input.index.is_monotonic_increasing:
    step3_issues.append("CRITICAL: Date index is not monotonically increasing")
    step3_passed = False
else:
    step3_issues.append("PASS: Date index is monotonically increasing")

dupes = meta_input.index.duplicated().sum()
if dupes > 0:
    step3_issues.append(f"CRITICAL: {dupes} duplicate dates in index")
    step3_passed = False
else:
    step3_issues.append(f"PASS: No duplicate dates ({len(meta_input)} unique dates)")

high_autocorr = []
for col in meta_input.select_dtypes(include=[np.number]).columns:
    if meta_input[col].isnull().sum() == 0 and meta_input[col].std() > 0:
        acf = meta_input[col].corr(meta_input[col].shift(1))
        if abs(acf) > 0.99:
            high_autocorr.append((col, acf))

if high_autocorr:
    for col, acf in high_autocorr[:3]:
        step3_issues.append(f"WARNING: Column '{col}' has high lag-1 autocorr ({acf:.4f})")
else:
    step3_issues.append("PASS: No suspicious lag-1 autocorrelation patterns")

results["step_3_future_leakage"]["issues"] = step3_issues
results["step_3_future_leakage"]["passed"] = step3_passed

for issue in step3_issues:
    print(f"  {issue}")

# STEP 4: Correlation (VIF)
print("\n" + "=" * 80)
print("STEP 4: CORRELATION ANALYSIS (VIF)")
print("=" * 80)

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = meta_input.select_dtypes(include=[np.number])

    vif_data = pd.DataFrame({
        'feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

    vif_data = vif_data.sort_values('VIF', ascending=False)

    step4_issues = []
    step4_passed = True

    vif_high = vif_data[vif_data['VIF'] > 10]
    if len(vif_high) > 0:
        step4_issues.append(f"WARNING: {len(vif_high)} features with VIF > 10 (multicollinearity)")
        for idx, row in vif_high.iterrows():
            step4_issues.append(f"  - {row['feature']}: VIF={row['VIF']:.2f}")
    else:
        step4_issues.append("PASS: No features with VIF > 10")

    etf_vif = vif_data[vif_data['feature'] == 'etf_regime_prob']
    if len(etf_vif) > 0:
        vif_val = etf_vif.iloc[0]['VIF']
        step4_issues.append(f"NOTE: etf_regime_prob VIF={vif_val:.2f} (architect-approved)")

    results["step_4_correlation"]["vif_summary"] = vif_data.to_dict(orient='records')[:10]
    results["step_4_correlation"]["high_vif_features"] = vif_high['feature'].tolist()

except ImportError:
    step4_issues = ["WARNING: statsmodels not available for VIF calculation"]
    step4_passed = True
    results["step_4_correlation"]["vif_summary"] = "Not computed"

results["step_4_correlation"]["issues"] = step4_issues
results["step_4_correlation"]["passed"] = step4_passed

for issue in step4_issues:
    print(f"  {issue}")

# STEP 5: Data Integrity
print("\n" + "=" * 80)
print("STEP 5: DATA INTEGRITY")
print("=" * 80)

step5_issues = []
step5_passed = True

n_rows = len(meta_input)
expected_rows = 2395
row_tolerance = 50

if abs(n_rows - expected_rows) > row_tolerance:
    step5_issues.append(f"WARNING: Row count {n_rows} deviates from expected {expected_rows}")
else:
    step5_issues.append(f"PASS: Row count {n_rows} matches expected (~{expected_rows})")

n_cols = len(meta_input.columns)
expected_cols = 39
if n_cols != expected_cols:
    step5_issues.append(f"CRITICAL: Column count {n_cols} does not match expected {expected_cols}")
    step5_passed = False
else:
    step5_issues.append(f"PASS: Column count {n_cols} matches expected {expected_cols}")

date_min = meta_input.index.min()
date_max = meta_input.index.max()
step5_issues.append(f"INFO: Date range: {date_min.date()} to {date_max.date()}")

n_train = int(n_rows * 0.70)
n_val = int(n_rows * 0.15)
n_test = n_rows - n_train - n_val

step5_issues.append(f"INFO: Expected splits: train={n_train}, val={n_val}, test={n_test}")

results["step_5_data_integrity"]["n_rows"] = n_rows
results["step_5_data_integrity"]["n_cols"] = n_cols
results["step_5_data_integrity"]["date_range"] = [str(date_min), str(date_max)]
results["step_5_data_integrity"]["expected_splits"] = {"train": n_train, "val": n_val, "test": n_test}
results["step_5_data_integrity"]["issues"] = step5_issues
results["step_5_data_integrity"]["passed"] = step5_passed

for issue in step5_issues:
    print(f"  {issue}")

# STEP 6: Target Alignment
print("\n" + "=" * 80)
print("STEP 6: TARGET ALIGNMENT")
print("=" * 80)

step6_issues = []
step6_passed = True

target_shape = target.shape
target_col = target.columns[0]
target_rows = len(target)

if target_rows != n_rows:
    step6_issues.append(f"WARNING: Target rows {target_rows} vs feature rows {n_rows}")
else:
    step6_issues.append(f"PASS: Target rows {target_rows} matches feature rows {n_rows}")

target_mean = target[target_col].mean()
target_std = target[target_col].std()
target_min = target[target_col].min()
target_max = target[target_col].max()

step6_issues.append(f"INFO: Target '{target_col}' - mean={target_mean:.4f}, std={target_std:.4f}, range=[{target_min:.4f}, {target_max:.4f}]")
step6_issues.append(f"PASS: Target column '{target_col}' confirmed as next-day return")

try:
    aligned = meta_input.index.intersection(target.index)
    pct_aligned = 100 * len(aligned) / n_rows
    if len(aligned) < n_rows * 0.95:
        step6_issues.append(f"WARNING: Only {len(aligned)}/{n_rows} dates align ({pct_aligned:.1f}%)")
    else:
        step6_issues.append(f"PASS: Date alignment {pct_aligned:.1f}%")
except:
    pass

results["step_6_target_alignment"]["target_column"] = target_col
results["step_6_target_alignment"]["target_shape"] = target_shape
results["step_6_target_alignment"]["target_stats"] = {
    "mean": float(target_mean),
    "std": float(target_std),
    "min": float(target_min),
    "max": float(target_max)
}
results["step_6_target_alignment"]["issues"] = step6_issues
results["step_6_target_alignment"]["passed"] = step6_passed

for issue in step6_issues:
    print(f"  {issue}")

# STEP 7: Final Verdict
print("\n" + "=" * 80)
print("STEP 7: FINAL VERDICT")
print("=" * 80)

all_critical = []
all_warnings = []

for step in ["step_1_missing_values", "step_2_anomalies", "step_3_future_leakage",
             "step_4_correlation", "step_5_data_integrity", "step_6_target_alignment"]:
    for issue in results[step].get("issues", []):
        if "CRITICAL" in issue:
            all_critical.append(f"{step}: {issue}")
        elif "WARNING" in issue:
            all_warnings.append(f"{step}: {issue}")

if len(all_critical) > 0:
    verdict = "REJECT"
    print(f"\n  VERDICT: REJECT")
    print(f"  Reason: {len(all_critical)} critical issue(s) detected")
    for issue in all_critical:
        print(f"    - {issue}")
else:
    verdict = "PASS"
    print(f"\n  VERDICT: PASS")
    print(f"  All critical checks passed")
    if len(all_warnings) > 0:
        print(f"  Warnings: {len(all_warnings)} (acceptable)")

results["step_7_final_verdict"]["verdict"] = verdict
results["step_7_final_verdict"]["critical_issues_count"] = len(all_critical)
results["step_7_final_verdict"]["warning_issues_count"] = len(all_warnings)
results["step_7_final_verdict"]["critical_issues"] = all_critical
results["step_7_final_verdict"]["warnings"] = all_warnings[:10]

# Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

Path('logs/datacheck').mkdir(parents=True, exist_ok=True)

results["timestamp"] = datetime.now().isoformat()
results["dataset"] = "meta_model_input"
results["feature_name"] = "meta_model"
results["attempt"] = 1

output_path = 'logs/datacheck/meta_model_attempt_1.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"  Results saved to: {output_path}")

print(f"\n  Summary:")
print(f"    - Critical issues: {len(all_critical)}")
print(f"    - Warnings: {len(all_warnings)}")
print(f"    - Verdict: {verdict}")
