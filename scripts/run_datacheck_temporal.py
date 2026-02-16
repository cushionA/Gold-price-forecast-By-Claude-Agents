"""
Datachecker: temporal_context_transformer attempt 1
7-step standardized check
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime

base = 'C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents'

# Load files
raw   = pd.read_csv(f'{base}/data/processed/temporal_context_raw.csv',   index_col=0, parse_dates=True)
train = pd.read_csv(f'{base}/data/processed/temporal_context_train.csv', index_col=0, parse_dates=True)
val   = pd.read_csv(f'{base}/data/processed/temporal_context_val.csv',   index_col=0, parse_dates=True)
test  = pd.read_csv(f'{base}/data/processed/temporal_context_test.csv',  index_col=0, parse_dates=True)

critical_issues = []
warnings = []

# ---- STEP 1: NaN check ----
for name, df in [('raw', raw), ('train', train), ('val', val), ('test', test)]:
    nan_total = df.isnull().sum().sum()
    if nan_total > 0:
        critical_issues.append(f'STEP1: {name} に {nan_total} 個のNaN')

step1_result = {
    'step': 'nan_check',
    'passed': not any('STEP1' in i for i in critical_issues),
    'total_nan_raw': int(raw.isnull().sum().sum()),
    'issues': [i for i in critical_issues if 'STEP1' in i],
}

# ---- STEP 2: Inf / Outlier check ----
for name, df in [('raw', raw), ('train', train), ('val', val), ('test', test)]:
    num_df = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(num_df).sum().sum()
    if inf_count > 0:
        critical_issues.append(f'STEP2: {name} に {inf_count} 個のInf')

outlier_info = {}
for col in raw.columns:
    col_data = raw[col].dropna()
    if col_data.std() > 0:
        z = (col_data - col_data.mean()) / col_data.std()
        n5 = int((z.abs() > 5).sum())
        max_z = float(z.abs().max())
        if n5 > 0:
            outlier_info[col] = {'count_z5': n5, 'max_z': round(max_z, 4)}
            if max_z > 20:
                critical_issues.append(f'STEP2: {col} に極端な外れ値 (max z={max_z:.2f})')
            else:
                warnings.append(f'STEP2: {col} に z>5 外れ値 {n5}件 (max z={max_z:.2f})')

step2_result = {
    'step': 'outlier_check',
    'passed': not any('STEP2' in i for i in critical_issues),
    'inf_count_raw': int(np.isinf(raw.select_dtypes(include=[np.number])).sum().sum()),
    'outlier_cols': outlier_info,
    'issues': [i for i in critical_issues if 'STEP2' in i],
}

# ---- STEP 3: Temporal ordering / future leak ----
for name, df in [('raw', raw), ('train', train), ('val', val), ('test', test)]:
    if not df.index.is_monotonic_increasing:
        critical_issues.append(f'STEP3: {name} の日付がソートされていない')
    dupes = int(df.index.duplicated().sum())
    if dupes > 0:
        critical_issues.append(f'STEP3: {name} に {dupes}件の重複日付')

if train.index.max() >= val.index.min():
    critical_issues.append('STEP3: train/val 期間が重複')
if val.index.max() >= test.index.min():
    critical_issues.append('STEP3: val/test 期間が重複')

step3_result = {
    'step': 'temporal_check',
    'passed': not any('STEP3' in i for i in critical_issues),
    'raw_sorted': bool(raw.index.is_monotonic_increasing),
    'raw_duplicates': int(raw.index.duplicated().sum()),
    'train_range': [str(train.index.min().date()), str(train.index.max().date())],
    'val_range':   [str(val.index.min().date()),   str(val.index.max().date())],
    'test_range':  [str(test.index.min().date()),  str(test.index.max().date())],
    'no_overlap': (train.index.max() < val.index.min()) and (val.index.max() < test.index.min()),
    'issues': [i for i in critical_issues if 'STEP3' in i],
}

# ---- STEP 4: Correlation / VIF ----
vif_results = {}
X = raw.select_dtypes(include=[np.number]).dropna()
cols = list(X.columns)
for col in cols:
    y = X[col].values
    other_cols = [c for c in cols if c != col]
    X_other = X[other_cols].values
    X_int = np.column_stack([np.ones(len(X_other)), X_other])
    try:
        coeffs = np.linalg.lstsq(X_int, y, rcond=None)[0]
        y_pred = X_int @ coeffs
        ss_res = float(np.sum((y - y_pred)**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
        vif_results[col] = round(vif, 4)
        if vif > 10:
            critical_issues.append(f'STEP4: {col} のVIF={vif:.2f} (>10)')
        elif vif > 5:
            warnings.append(f'STEP4: {col} のVIF={vif:.2f} (>5)')
    except Exception as e:
        warnings.append(f'STEP4: {col} VIF計算エラー: {e}')

high_corr_pairs = []
corr_matrix = raw.corr().abs()
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        c = float(corr_matrix.iloc[i, j])
        if c > 0.9:
            critical_issues.append(
                f'STEP4: {corr_matrix.columns[i]} vs {corr_matrix.columns[j]} corr={c:.3f} (>0.9)')
            high_corr_pairs.append({'col1': corr_matrix.columns[i], 'col2': corr_matrix.columns[j], 'corr': round(c, 4)})
        elif c > 0.7:
            warnings.append(
                f'STEP4: {corr_matrix.columns[i]} vs {corr_matrix.columns[j]} corr={c:.3f} (>0.7)')
            high_corr_pairs.append({'col1': corr_matrix.columns[i], 'col2': corr_matrix.columns[j], 'corr': round(c, 4)})

step4_result = {
    'step': 'correlation_vif_check',
    'passed': not any('STEP4' in i for i in critical_issues),
    'vif_results': vif_results,
    'max_vif': max(vif_results.values()) if vif_results else None,
    'high_corr_pairs': high_corr_pairs,
    'issues': [i for i in critical_issues if 'STEP4' in i],
}

# ---- STEP 5: Split consistency ----
total = len(raw)
n_train = len(train)
n_val   = len(val)
n_test  = len(test)
ratio_train = n_train / total * 100
ratio_val   = n_val   / total * 100
ratio_test  = n_test  / total * 100

for name, ratio, target in [('train', ratio_train, 70), ('val', ratio_val, 15), ('test', ratio_test, 15)]:
    if abs(ratio - target) > 2:
        critical_issues.append(
            f'STEP5: {name} split比率 {ratio:.1f}% (target={target}%, diff={abs(ratio-target):.1f}% > 2%)')

if len(raw.columns) != 14:
    critical_issues.append(f'STEP5: 特徴量数 {len(raw.columns)} (expected 14)')

cols_consistent = (set(raw.columns) == set(train.columns) == set(val.columns) == set(test.columns))
if not cols_consistent:
    critical_issues.append('STEP5: ファイル間でカラムが一致しない')

step5_result = {
    'step': 'consistency_check',
    'passed': not any('STEP5' in i for i in critical_issues),
    'n_features': len(raw.columns),
    'total_rows': total,
    'train_rows': n_train, 'train_pct': round(ratio_train, 2),
    'val_rows':   n_val,   'val_pct':   round(ratio_val, 2),
    'test_rows':  n_test,  'test_pct':  round(ratio_test, 2),
    'columns_consistent': cols_consistent,
    'issues': [i for i in critical_issues if 'STEP5' in i],
}

# ---- STEP 6: Statistical summary / standardization ----
non_std = []
stats_summary = {}
for col in raw.columns:
    m = float(raw[col].mean())
    s = float(raw[col].std())
    mn = float(raw[col].min())
    mx = float(raw[col].max())
    stats_summary[col] = {
        'mean': round(m, 6),
        'std':  round(s, 6),
        'min':  round(mn, 6),
        'max':  round(mx, 6),
    }
    if abs(m) > 1.0 or not (0.3 < s < 3.0):
        non_std.append(col)
        warnings.append(f'STEP6: {col} 標準化に問題 (mean={m:.4f}, std={s:.4f})')

step6_result = {
    'step': 'statistics_check',
    'passed': len(non_std) == 0,
    'all_standardized': len(non_std) == 0,
    'non_standardized_cols': non_std,
    'stats_summary': stats_summary,
    'issues': [],
}

# ---- STEP 7: Final judgment ----
if critical_issues:
    action = 'REJECT'
elif len(warnings) > 5:
    action = 'CONDITIONAL_PASS'
else:
    action = 'PASS'

report = {
    'feature': 'temporal_context_transformer',
    'attempt': 1,
    'timestamp': datetime.now().isoformat(),
    'steps': {
        'step1_nan':           step1_result,
        'step2_outliers':      step2_result,
        'step3_temporal':      step3_result,
        'step4_correlation':   step4_result,
        'step5_consistency':   step5_result,
        'step6_statistics':    step6_result,
    },
    'critical_issues': critical_issues,
    'warnings': warnings,
    'action': action,
    'overall_passed': action != 'REJECT',
    'summary': {
        'n_critical': len(critical_issues),
        'n_warnings': len(warnings),
        'files_checked': [
            'data/processed/temporal_context_raw.csv',
            'data/processed/temporal_context_train.csv',
            'data/processed/temporal_context_val.csv',
            'data/processed/temporal_context_test.csv',
        ],
        'total_rows': total,
        'n_features': len(raw.columns),
        'date_range': [str(raw.index.min().date()), str(raw.index.max().date())],
    }
}

# Print summary
print('=== STEP 7: 総合判定 ===')
print(f'CRITICAL issues: {len(critical_issues)}')
for i in critical_issues:
    print(f'  [CRITICAL] {i}')
print(f'WARNINGS: {len(warnings)}')
for w in warnings:
    print(f'  [WARNING] {w}')
print()
print(f'ACTION: {action}')
print(f'Overall passed: {report["overall_passed"]}')

# Save report
report_path = f'{base}/logs/datacheck/temporal_context_transformer_1.json'
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f'Report saved to: {report_path}')
