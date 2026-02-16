#!/usr/bin/env python
"""
DataChecker - Standardized 7-Step Quality Check
Meta-Model Attempt 2 (Corrected)
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

print("[OK] Train shape: {}".format(train.shape))
print("[OK] Val shape: {}".format(val.shape))
print("[OK] Test shape: {}".format(test.shape))

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
        step1_issues.append("CRITICAL: {} has {} NaN values".format(split_name, nan_count))
        step1_passed = False

    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        step1_issues.append("CRITICAL: {} has {} infinite values".format(split_name, inf_count))
        step1_passed = False

if step1_passed:
    print("[OK] No NaN values in any split")
    print("[OK] No infinite values in any split")
else:
    for issue in step1_issues:
        print("[FAIL] {}".format(issue))

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

    # Check for constant columns (std < 1e-6)
    for col in numeric_cols:
        std = df[col].std()
        if std < 1e-6:
            step2_issues.append("CRITICAL: {}.{} is constant (std={:.2e})".format(split_name, col, std))
            step2_passed = False

if step2_passed:
    print("[OK] No constant columns (all std > 1e-6)")
    print("[OK] Outliers expected in financial data (not flagged as critical)")
else:
    critical = [i for i in step2_issues if 'CRITICAL' in i]
    for issue in critical:
        print("[FAIL] {}".format(issue))

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

# Correct target column name
target_col = 'gold_return_next'
if target_col not in train.columns:
    print("[FAIL] Target column '{}' not found in data".format(target_col))
    step3_issues.append("CRITICAL: Target column '{}' not found".format(target_col))
    step3_passed = False
else:
    numeric_features = train.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns

    # Check for high correlation with target (>0.8 = leakage)
    high_corr = []
    for col in numeric_features:
        corr_current = train[col].corr(train[target_col])
        if abs(corr_current) > 0.8:
            step3_issues.append("CRITICAL: {} has correlation {:.3f} with target (leakage suspected)".format(col, corr_current))
            step3_passed = False

    # Check for autocorrelation > 0.99
    for col in numeric_features:
        if col != target_col:
            autocorr = train[col].autocorr(lag=1)
            if abs(autocorr) > 0.99:
                step3_issues.append("CRITICAL: {} has autocorr={:.4f} (potential duplicates)".format(col, autocorr))
                step3_passed = False

    # Check for exact duplicates
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            step3_issues.append("CRITICAL: {} has {} duplicate rows".format(split_name, dup_count))
            step3_passed = False

    if step3_passed:
        print("[OK] No correlation > 0.8 between features and target")
        print("[OK] No autocorrelation > 0.99")
        print("[OK] No exact duplicate rows")

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
    step4_issues.append("CRITICAL: Train-Val overlap (train ends {}, val starts {})".format(
        train_dates[-1], val_dates[0]))
    step4_passed = False
if val_dates[-1] >= test_dates[0]:
    step4_issues.append("CRITICAL: Val-Test overlap (val ends {}, test starts {})".format(
        val_dates[-1], test_dates[0]))
    step4_passed = False

# Check for gaps
gap_val = val_dates[0] - train_dates[-1]
gap_test = test_dates[0] - val_dates[-1]

if gap_val > pd.Timedelta(days=7):
    step4_issues.append("WARNING: Train-Val gap of {} days".format(gap_val.days))
if gap_test > pd.Timedelta(days=7):
    step4_issues.append("WARNING: Val-Test gap of {} days".format(gap_test.days))

if step4_passed:
    print("[OK] All splits sorted chronologically")
    print("[OK] No overlap between train/val/test")
    print("[OK] Train: {} to {}".format(train_dates[0].date(), train_dates[-1].date()))
    print("[OK] Val: {} to {}".format(val_dates[0].date(), val_dates[-1].date()))
    print("[OK] Test: {} to {}".format(test_dates[0].date(), test_dates[-1].date()))
else:
    for issue in step4_issues:
        print("[FAIL] {}".format(issue))

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

# 23 columns total: 22 features + 1 target
expected_total = 23
expected_features = 22
actual_cols = len(train.columns)

if actual_cols != expected_total:
    step5_issues.append("CRITICAL: Expected {} total columns, got {}".format(expected_total, actual_cols))
    step5_passed = False

# Check schema consistency
train_cols = set(train.columns)
val_cols = set(val.columns)
test_cols = set(test.columns)

if train_cols != val_cols:
    diff = train_cols.symmetric_difference(val_cols)
    step5_issues.append("CRITICAL: Train-Val schema mismatch: {}".format(diff))
    step5_passed = False

if train_cols != test_cols:
    diff = train_cols.symmetric_difference(test_cols)
    step5_issues.append("CRITICAL: Train-Test schema mismatch: {}".format(diff))
    step5_passed = False

# Check for forbidden columns
forbidden_columns = [
    'gld_close', 'silver_close', 'copper_close', 'sp500_close',
    'gld_open', 'gld_high', 'gld_low', 'gld_volume',
    'dgs10', 'dgs2',
    'cny_regime_prob', 'cny_momentum_z', 'cny_volatility_regime_z',
    'yc_regime_prob',
    'real_rate_regime_prob', 'real_rate_momentum_z'
]

for col in forbidden_columns:
    if col in train.columns:
        step5_issues.append("CRITICAL: Forbidden column '{}' present in data".format(col))
        step5_passed = False

if step5_passed:
    features_count = actual_cols - 1  # Exclude target
    print("[OK] Total columns: {} ({} features + 1 target)".format(actual_cols, features_count))
    print("[OK] Train/Val/Test have identical schema")
    print("[OK] No forbidden price-level or CNY columns")
else:
    for issue in step5_issues:
        print("[FAIL] {}".format(issue))

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
    step6_issues.append("CRITICAL: Train has {} rows, need >= {}".format(train_count, min_train))
    step6_passed = False

if val_count < min_val:
    step6_issues.append("CRITICAL: Val has {} rows, need >= {}".format(val_count, min_val))
    step6_passed = False

if test_count < min_test:
    step6_issues.append("CRITICAL: Test has {} rows, need >= {}".format(test_count, min_test))
    step6_passed = False

# Calculate ratio
ratio = train_count / expected_features if expected_features > 0 else 0

if step6_passed:
    print("[OK] Train: {} rows (requirement: >= {})".format(train_count, min_train))
    print("[OK] Val: {} rows (requirement: >= {})".format(val_count, min_val))
    print("[OK] Test: {} rows (requirement: >= {})".format(test_count, min_test))
    print("[OK] Samples/Feature ratio: {:.1f}:1 (healthy)".format(ratio))
else:
    for issue in step6_issues:
        print("[FAIL] {}".format(issue))

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
        step7_issues.append("WARNING: Only {} features with |corr| > 0.05 (expected >= 5)".format(len(strong_features)))

    if len(strong_features) >= 5:
        print("[OK] {} features with |corr| > 0.05 with target".format(len(strong_features)))
        print("\nTop 10 Correlations:")
        for i, (col, abs_corr, corr) in enumerate(correlations[:10], 1):
            print("  {:2d}. {:35s} | {:7.4f}".format(i, col, corr))
    else:
        print("[WARN] Weak correlation profile detected")
        print("All Correlations (sorted by magnitude):")
        for col, abs_corr, corr in correlations:
            print("  {:35s} | {:7.4f}".format(col, corr))
else:
    print("[FAIL] Target column not found, skipping correlation check")
    step7_issues.append("CRITICAL: Target column '{}' not found".format(target_col))
    step7_passed = False

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

print("\nCritical Issues: {}".format(len(critical_issues)))
if critical_issues:
    for issue in critical_issues:
        print("  [FAIL] {}".format(issue))

print("\nWarnings: {}".format(len(warnings_list)))
if warnings_list and len(warnings_list) <= 10:
    for issue in warnings_list:
        print("  [WARN] {}".format(issue))
elif warnings_list:
    for issue in warnings_list[:5]:
        print("  [WARN] {}".format(issue))
    print("  ... and {} more warnings".format(len(warnings_list) - 5))

# Decision logic
if critical_issues:
    decision = "REJECT"
elif len(warnings_list) > 5:
    decision = "CONDITIONAL_PASS"
else:
    decision = "PASS"

print("\n" + "=" * 80)
print("DECISION: {}".format(decision))
print("=" * 80)

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
    "features": int(expected_features),
    "feature_list": sorted([c for c in train.columns if c != target_col])
}

with open('logs/datacheck/meta_model_attempt_2.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n[OK] Report saved to logs/datacheck/meta_model_attempt_2.json")
