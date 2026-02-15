#!/usr/bin/env python
"""
DataChecker - Standardized 7-Step Quality Check
Meta-Model Attempt 2
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 0: Load Data
# ============================================
print("=" * 80)
print("DATACHECKER - STEP 0: Load Data")
print("=" * 80)

train = pd.read_csv('data/meta_model/meta_model_attempt_2_train.csv', index_col=0)
val = pd.read_csv('data/meta_model/meta_model_attempt_2_val.csv', index_col=0)
test = pd.read_csv('data/meta_model/meta_model_attempt_2_test.csv', index_col=0)

print(f"[OK] Train shape: {train.shape}")
print(f"[OK] Val shape: {val.shape}")
print(f"[OK] Test shape: {test.shape}")
print(f"[OK] Train columns: {train.columns.tolist()}")

# ============================================
# STEP 1: Missing Values Check
# ============================================
print("\n" + "=" * 80)
print("STEP 1: Missing Values & Infinite Values Check")
print("=" * 80)

step1_issues = []
step1_passed = True

for split_name, df in [('train', train), ('val', val), ('test', test)]:
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        step1_issues.append(f"CRITICAL: {split_name} has {nan_count} NaN values")
        step1_passed = False

    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        step1_issues.append(f"CRITICAL: {split_name} has {inf_count} infinite values")
        step1_passed = False

if step1_passed:
    print("[OK] No NaN values in any split")
    print("[OK] No infinite values in any split")
else:
    for issue in step1_issues:
        print(f"[FAIL] {issue}")

step1_result = {
    "step": "missing_values",
    "issues": step1_issues,
    "passed": step1_passed
}

# ============================================
# STEP 2: Anomalies Check
# ============================================
print("\n" + "=" * 80)
print("STEP 2: Anomalies (Outliers & Constant Columns)")
print("=" * 80)

step2_issues = []
step2_passed = True

for split_name, df in [('train', train), ('val', val), ('test', test)]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Check for constant columns
    for col in numeric_cols:
        std = df[col].std()
        if std < 1e-6:
            step2_issues.append(f"CRITICAL: {split_name}.{col} is constant (std={std:.2e})")
            step2_passed = False

    # Check for extreme outliers (>5 std)
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            z_scores = np.abs((df[col] - mean) / std)
            outliers = (z_scores > 5).sum()
            if outliers > 0:
                step2_issues.append(f"WARNING: {split_name}.{col} has {outliers} outliers >5std")

if step2_passed and not step2_issues:
    print("[OK] No constant columns (all std > 1e-6)")
    print("[OK] Outliers check passed")
else:
    for issue in step2_issues:
        print(f"{'[FAIL]' if 'CRITICAL' in issue else '[WARN]'} {issue}")

step2_result = {
    "step": "anomalies",
    "issues": step2_issues,
    "passed": step2_passed
}

# ============================================
# STEP 3: Future Leakage Check
# ============================================
print("\n" + "=" * 80)
print("STEP 3: Future Leakage Check")
print("=" * 80)

step3_issues = []
step3_passed = True

# Separate target from features
target_col = 'gold_return'
if target_col not in train.columns:
    print("[WARN] Target column 'gold_return' not found in data")
    step3_issues.append(f"WARNING: Target column '{target_col}' not found")
else:
    numeric_features = train.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns

    # Check for high correlation with target
    for col in numeric_features:
        corr_current = train[col].corr(train[target_col])
        if abs(corr_current) > 0.8:
            step3_issues.append(f"CRITICAL: {col} has correlation {corr_current:.3f} with target (leakage suspected)")
            step3_passed = False

    # Check for autocorrelation > 0.99 (exact duplicate pattern)
    for col in numeric_features:
        if col != target_col:
            autocorr = train[col].autocorr(lag=1)
            if abs(autocorr) > 0.99:
                step3_issues.append(f"CRITICAL: {col} has autocorr={autocorr:.4f} (potential duplicates)")
                step3_passed = False

    # Check for exact duplicates (entire rows)
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            step3_issues.append(f"CRITICAL: {split_name} has {dup_count} duplicate rows")
            step3_passed = False

if step3_passed and not step3_issues:
    print("[OK] No correlation > 0.8 between features and target")
    print("[OK] No autocorrelation > 0.99")
    print("[OK] No exact duplicate rows")
else:
    for issue in step3_issues:
        print(f"[FAIL] {issue}")

step3_result = {
    "step": "future_leakage",
    "issues": step3_issues,
    "passed": step3_passed
}

# ============================================
# STEP 4: Data Alignment Check
# ============================================
print("\n" + "=" * 80)
print("STEP 4: Data Alignment (Dates & Gaps)")
print("=" * 80)

step4_issues = []
step4_passed = True

# Convert index to dates
train_dates = pd.to_datetime(train.index)
val_dates = pd.to_datetime(val.index)
test_dates = pd.to_datetime(test.index)

# Check if sorted
if not train_dates.is_monotonic_increasing:
    step4_issues.append("CRITICAL: Train dates not sorted")
    step4_passed = False
if not val_dates.is_monotonic_increasing:
    step4_issues.append("CRITICAL: Val dates not sorted")
    step4_passed = False
if not test_dates.is_monotonic_increasing:
    step4_issues.append("CRITICAL: Test dates not sorted")
    step4_passed = False

# Check for overlaps
if train_dates[-1] >= val_dates[0]:
    step4_issues.append(f"CRITICAL: Train-Val overlap (train ends {train_dates[-1]}, val starts {val_dates[0]})")
    step4_passed = False
if val_dates[-1] >= test_dates[0]:
    step4_issues.append(f"CRITICAL: Val-Test overlap (val ends {val_dates[-1]}, test starts {test_dates[0]})")
    step4_passed = False

# Check for gaps
gap_val = val_dates[0] - train_dates[-1]
gap_test = test_dates[0] - val_dates[-1]

if gap_val > pd.Timedelta(days=7):
    step4_issues.append(f"WARNING: Train-Val gap of {gap_val.days} days")
if gap_test > pd.Timedelta(days=7):
    step4_issues.append(f"WARNING: Val-Test gap of {gap_test.days} days")

if step4_passed and not step4_issues:
    print("[OK] All splits sorted chronologically")
    print("[OK] No overlap between train/val/test")
    print(f"[OK] Train: {train_dates[0].date()} to {train_dates[-1].date()}")
    print(f"[OK] Val: {val_dates[0].date()} to {val_dates[-1].date()}")
    print(f"[OK] Test: {test_dates[0].date()} to {test_dates[-1].date()}")
else:
    for issue in step4_issues:
        print(f"{'[FAIL]' if 'CRITICAL' in issue else '[WARN]'} {issue}")

step4_result = {
    "step": "data_alignment",
    "issues": step4_issues,
    "passed": step4_passed
}

# ============================================
# STEP 5: Feature Count Check
# ============================================
print("\n" + "=" * 80)
print("STEP 5: Feature Count & Schema Consistency")
print("=" * 80)

step5_issues = []
step5_passed = True

expected_features = 22  # 5 base + 17 submodel
actual_features = len(train.columns)

if actual_features != expected_features:
    step5_issues.append(f"CRITICAL: Expected {expected_features} features, got {actual_features}")
    step5_passed = False

# Check schema consistency across splits
train_cols = set(train.columns)
val_cols = set(val.columns)
test_cols = set(test.columns)

if train_cols != val_cols:
    diff = train_cols.symmetric_difference(val_cols)
    step5_issues.append(f"CRITICAL: Train-Val schema mismatch: {diff}")
    step5_passed = False

if train_cols != test_cols:
    diff = train_cols.symmetric_difference(test_cols)
    step5_issues.append(f"CRITICAL: Train-Test schema mismatch: {diff}")
    step5_passed = False

# Check for forbidden columns
forbidden_columns = [
    'gld_close', 'silver_close', 'copper_close', 'sp500_close',  # Price levels
    'gld_open', 'gld_high', 'gld_low', 'gld_volume',  # OHLCV
    'dgs10', 'dgs2',  # Yield levels
    'cny_regime_prob', 'cny_momentum_z', 'cny_volatility_regime_z',  # CNY
    'yc_regime_prob',  # Constant column
    'real_rate_regime_prob', 'real_rate_momentum_z'  # Excluded submodel
]

for col in forbidden_columns:
    if col in train.columns:
        step5_issues.append(f"CRITICAL: Forbidden column '{col}' present in data")
        step5_passed = False

if step5_passed and not step5_issues:
    print(f"[OK] Feature count: {actual_features} (expected: {expected_features})")
    print("[OK] Train/Val/Test have identical schema")
    print("[OK] No forbidden price-level or CNY columns")
else:
    for issue in step5_issues:
        print(f"[FAIL] {issue}")

step5_result = {
    "step": "feature_count",
    "issues": step5_issues,
    "passed": step5_passed
}

# ============================================
# STEP 6: Sample Count Check
# ============================================
print("\n" + "=" * 80)
print("STEP 6: Sample Count Requirements")
print("=" * 80)

step6_issues = []
step6_passed = True

min_train = 1700
min_val = 350
min_test = 350

train_count = len(train)
val_count = len(val)
test_count = len(test)

if train_count < min_train:
    step6_issues.append(f"CRITICAL: Train has {train_count} rows, need >= {min_train}")
    step6_passed = False

if val_count < min_val:
    step6_issues.append(f"CRITICAL: Val has {val_count} rows, need >= {min_val}")
    step6_passed = False

if test_count < min_test:
    step6_issues.append(f"CRITICAL: Test has {test_count} rows, need >= {min_test}")
    step6_passed = False

# Calculate ratio
ratio = train_count / actual_features if actual_features > 0 else 0

if step6_passed and not step6_issues:
    print(f"[OK] Train: {train_count} rows (requirement: >= {min_train})")
    print(f"[OK] Val: {val_count} rows (requirement: >= {min_val})")
    print(f"[OK] Test: {test_count} rows (requirement: >= {min_test})")
    print(f"[OK] Samples/Feature ratio: {ratio:.1f}:1 (healthy)")
else:
    for issue in step6_issues:
        print(f"[FAIL] {issue}")

step6_result = {
    "step": "sample_count",
    "issues": step6_issues,
    "passed": step6_passed
}

# ============================================
# STEP 7: Correlation Check
# ============================================
print("\n" + "=" * 80)
print("STEP 7: Feature Correlation with Target")
print("=" * 80)

step7_issues = []
step7_passed = True

if target_col in train.columns:
    numeric_features = train.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns
    correlations = []

    for col in numeric_features:
        corr = train[col].corr(train[target_col])
        correlations.append((col, abs(corr), corr))

    correlations.sort(key=lambda x: x[1], reverse=True)

    # Check if at least 5 features have |corr| > 0.05
    strong_features = [c for c in correlations if c[1] > 0.05]

    if len(strong_features) < 5:
        step7_issues.append(f"WARNING: Only {len(strong_features)} features with |corr| > 0.05 (expected >= 5)")

    if step7_passed and not step7_issues:
        print(f"[OK] {len(strong_features)} features with |corr| > 0.05")
        print(f"\nTop 10 Correlations:")
        for i, (col, abs_corr, corr) in enumerate(correlations[:10], 1):
            print(f"  {i:2d}. {col:35s} | {corr:7.4f} (abs: {abs_corr:.4f})")
    else:
        print("\nAll Correlations (sorted by magnitude):")
        for col, abs_corr, corr in correlations:
            print(f"  {col:35s} | {corr:7.4f} (abs: {abs_corr:.4f})")
        for issue in step7_issues:
            print(f"[WARN] {issue}")
else:
    print("[FAIL] Target column not found, skipping correlation check")
    step7_issues.append("WARNING: Target column 'gold_return' not found")

step7_result = {
    "step": "correlations",
    "issues": step7_issues,
    "passed": step7_passed
}

# ============================================
# Summary & Decision
# ============================================
print("\n" + "=" * 80)
print("DATACHECKER SUMMARY")
print("=" * 80)

all_results = [step1_result, step2_result, step3_result, step4_result,
               step5_result, step6_result, step7_result]

critical_issues = [i for r in all_results for i in r.get("issues", []) if "CRITICAL" in i]
warnings_list = [i for r in all_results for i in r.get("issues", []) if "WARNING" in i]

print(f"\nCritical Issues: {len(critical_issues)}")
if critical_issues:
    for issue in critical_issues:
        print(f"  [FAIL] {issue}")

print(f"\nWarnings: {len(warnings_list)}")
if warnings_list:
    for issue in warnings_list:
        print(f"  [WARN] {issue}")

# Decision logic
if critical_issues:
    decision = "REJECT"
elif len(warnings_list) > 5:
    decision = "CONDITIONAL_PASS"
else:
    decision = "PASS"

print(f"\n{'=' * 80}")
print(f"DECISION: {decision}")
print(f"{'=' * 80}")

# ============================================
# Save Report
# ============================================
os.makedirs('logs/datacheck', exist_ok=True)

report = {
    "feature": "meta_model",
    "attempt": 2,
    "timestamp": datetime.now().isoformat(),
    "steps": all_results,
    "critical_issues": critical_issues,
    "warnings": warnings_list,
    "decision": decision,
    "overall_passed": decision != "REJECT",
    "samples": {
        "train": int(train_count),
        "val": int(val_count),
        "test": int(test_count),
        "total": int(train_count + val_count + test_count)
    },
    "features": int(actual_features),
    "feature_list": sorted(train.columns.tolist())
}

with open('logs/datacheck/meta_model_attempt_2.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n[OK] Report saved to logs/datacheck/meta_model_attempt_2.json")
