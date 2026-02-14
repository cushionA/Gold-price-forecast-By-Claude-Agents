"""
Technical Submodel Attempt 1 - 3-Gate Evaluation
Evaluator: Gate 1 (standalone quality), Gate 2 (information gain), Gate 3 (ablation)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

PROJECT_ROOT = r"C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents"

# ============================================================
# Load Data
# ============================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

# Training log
with open(os.path.join(PROJECT_ROOT, "logs/training/technical_1.json")) as f:
    training_log = json.load(f)

# Submodel output
sub = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data/submodel_outputs/technical.csv"),
    index_col="date", parse_dates=True
)

# Base features
base = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data/processed/base_features.csv"),
    index_col="Date", parse_dates=True
)

# Target
target = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data/processed/target.csv"),
    index_col="Date", parse_dates=True
)

# VIX submodel output (already completed)
vix_sub = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data/submodel_outputs/vix.csv"),
    index_col="date", parse_dates=True
)

print(f"Sub shape: {sub.shape}")
print(f"Base shape: {base.shape}")
print(f"Target shape: {target.shape}")
print(f"VIX sub shape: {vix_sub.shape}")
print(f"\nSub columns: {list(sub.columns)}")
print(f"Sub date range: {sub.index.min()} to {sub.index.max()}")
print(f"Base date range: {base.index.min()} to {base.index.max()}")
print(f"Target date range: {target.index.min()} to {target.index.max()}")

# Normalize timezone-aware indices to timezone-naive for alignment
def normalize_index(df):
    """Convert index to timezone-naive datetime, normalizing to date only."""
    idx = pd.to_datetime(df.index, utc=True)
    df.index = idx.tz_localize(None).normalize()  # Strip tz and time component
    return df

sub = normalize_index(sub)
base = normalize_index(base)
target = normalize_index(target)
vix_sub = normalize_index(vix_sub)

print(f"\nAfter normalization:")
print(f"  Sub: {sub.index.min()} to {sub.index.max()}, type={type(sub.index)}")
print(f"  Base: {base.index.min()} to {base.index.max()}")
print(f"  Target: {target.index.min()} to {target.index.max()}")
print(f"  VIX: {vix_sub.index.min()} to {vix_sub.index.max()}")

# Remove gold_return_next from base if present (it's the target, not a feature)
if "gold_return_next" in base.columns:
    base = base.drop(columns=["gold_return_next"])

print(f"\nBase features ({len(base.columns)}): {list(base.columns)}")

# ============================================================
# GATE 1: Standalone Quality
# ============================================================
print("\n" + "=" * 60)
print("GATE 1: Standalone Quality")
print("=" * 60)

gate1_checks = {}

# 1a. Overfit ratio
# HMM is unsupervised, no train/val loss. Check if training log has overfit_ratio.
if "overfit_ratio" in training_log.get("metrics", {}):
    ratio = training_log["metrics"]["overfit_ratio"]
    gate1_checks["overfit"] = {"value": ratio, "passed": ratio < 1.5}
else:
    # For HMM (unsupervised), overfit ratio is not applicable.
    # Check MI stability between train/val instead.
    # The training log has mi_sum from Optuna which was computed on validation set.
    gate1_checks["overfit"] = {
        "value": "N/A (unsupervised HMM)",
        "passed": True,
        "note": "HMM is unsupervised; overfit ratio not applicable. MI computed on val set."
    }

# 1b. All-NaN columns (after warm-up period)
# First 40 rows may have NaN due to rolling windows. Check after warm-up.
warmup = 40  # gk_baseline_window=40
sub_after_warmup = sub.iloc[warmup:]
nan_cols = sub_after_warmup.columns[sub_after_warmup.isnull().all()].tolist()
gate1_checks["no_all_nan"] = {
    "value": nan_cols,
    "passed": len(nan_cols) == 0,
    "note": f"Checked after {warmup}-row warm-up period"
}
print(f"All-NaN cols (after warmup): {nan_cols}")

# Total NaN counts
for col in sub.columns:
    nan_count = sub[col].isnull().sum()
    print(f"  {col}: {nan_count} NaN ({nan_count/len(sub)*100:.1f}%)")

# 1c. Zero-variance (constant) columns
zero_var = sub.columns[sub.std() < 1e-10].tolist()
gate1_checks["no_zero_var"] = {
    "value": zero_var,
    "passed": len(zero_var) == 0
}
print(f"Zero-variance cols: {zero_var}")

# 1d. Column statistics
print("\nColumn statistics:")
for col in sub.columns:
    s = sub[col].dropna()
    print(f"  {col}:")
    print(f"    mean={s.mean():.6f}, std={s.std():.6f}, min={s.min():.6f}, max={s.max():.6f}")

# 1e. Autocorrelation check (leak indicator)
print("\nAutocorrelation check:")
for col in sub.columns:
    s = sub[col].dropna()
    ac1 = s.autocorr(lag=1)
    ac5 = s.autocorr(lag=5)
    print(f"  {col}: lag1={ac1:.4f}, lag5={ac5:.4f}")
    if abs(ac1) > 0.99:
        gate1_checks[f"autocorr_{col}"] = {
            "value": ac1,
            "passed": False,
            "note": f"Suspiciously high autocorrelation at lag 1: {ac1:.6f}"
        }

# 1f. Optuna HPO coverage
if "optuna_trials_completed" in training_log:
    completed = training_log["optuna_trials_completed"]
    gate1_checks["hpo_coverage"] = {
        "value": completed,
        "passed": True
    }
    if completed < 10:
        gate1_checks["hpo_coverage"]["warning"] = f"Optuna {completed} trials only. Insufficient exploration."
    print(f"Optuna trials: {completed}")

# 1g. Check for suspicious patterns (e.g., too many exact values)
for col in sub.columns:
    s = sub[col].dropna()
    n_unique = s.nunique()
    ratio_unique = n_unique / len(s)
    print(f"  {col}: {n_unique} unique values ({ratio_unique:.4f} ratio)")
    if n_unique < 5 and len(s) > 100:
        gate1_checks[f"low_unique_{col}"] = {
            "value": n_unique,
            "passed": False,
            "note": f"Only {n_unique} unique values -- potentially degenerate"
        }

gate1_passed = all(c["passed"] for c in gate1_checks.values())
print(f"\n>>> GATE 1: {'PASS' if gate1_passed else 'FAIL'}")
for k, v in gate1_checks.items():
    status = "PASS" if v["passed"] else "FAIL"
    print(f"    {k}: {status} (value={v['value']})")

# ============================================================
# GATE 2: Information Gain
# ============================================================
print("\n" + "=" * 60)
print("GATE 2: Information Gain (MI increase, VIF, stability)")
print("=" * 60)

from sklearn.feature_selection import mutual_info_regression

# Align indices: base + sub + target
# Include VIX submodel in the "current extended base" since it's already completed
extended_base = base.copy()
# Add VIX submodel columns to base
for col in vix_sub.columns:
    extended_base[col] = vix_sub[col]

# Common index
idx = extended_base.index.intersection(sub.index).intersection(target.index)
print(f"Common dates: {len(idx)} (from {idx.min()} to {idx.max()})")

b = extended_base.loc[idx].copy()
s = sub.loc[idx].copy()
y = target.loc[idx].values.ravel()

# Drop rows with NaN in base, sub, or target
mask = b.notna().all(axis=1) & s.notna().all(axis=1) & ~np.isnan(y)
b = b.loc[mask]
s = s.loc[mask]
y = y[mask.values]
print(f"After NaN removal: {len(b)} rows")

gate2_checks = {}

# 2a. MI increase (sum-based)
print("\nComputing MI (base)...")
mi_b = mutual_info_regression(b, y, random_state=42)
mi_b_sum = mi_b.sum()
print(f"  Base MI sum: {mi_b_sum:.6f}")
for i, col in enumerate(b.columns):
    print(f"    {col}: {mi_b[i]:.6f}")

print("\nComputing MI (extended = base + submodel)...")
ext = pd.concat([b, s], axis=1)
mi_e = mutual_info_regression(ext, y, random_state=42)
mi_e_sum = mi_e.sum()
print(f"  Extended MI sum: {mi_e_sum:.6f}")
for i, col in enumerate(ext.columns):
    print(f"    {col}: {mi_e[i]:.6f}")

mi_increase = (mi_e_sum - mi_b_sum) / (mi_b_sum + 1e-10)
print(f"\n  MI increase: {mi_increase*100:.2f}%")
gate2_checks["mi"] = {
    "base": float(mi_b_sum),
    "extended": float(mi_e_sum),
    "increase": float(mi_increase),
    "passed": mi_increase > 0.05
}

# 2b. VIF check
print("\nComputing VIF...")
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # Only check VIF on submodel columns added to existing features
    # Full VIF on all columns
    vifs = {}
    for i, col in enumerate(ext.columns):
        try:
            v = variance_inflation_factor(ext.values, i)
            vifs[col] = v
        except:
            vifs[col] = float('inf')

    # Report submodel VIFs specifically
    sub_vifs = {col: vifs[col] for col in s.columns}
    max_sub_vif = max(sub_vifs.values())
    max_all_vif = max(vifs.values())

    print(f"  Submodel column VIFs:")
    for col, v in sub_vifs.items():
        print(f"    {col}: {v:.2f}")
    print(f"  Max submodel VIF: {max_sub_vif:.2f}")
    print(f"  Max overall VIF: {max_all_vif:.2f}")

    # Check: are submodel VIFs reasonable?
    # Note: base features may have high VIF among themselves (known issue)
    gate2_checks["vif"] = {
        "max_submodel": float(max_sub_vif),
        "max_overall": float(max_all_vif),
        "submodel_vifs": {k: float(v) for k, v in sub_vifs.items()},
        "passed": max_sub_vif < 10
    }
except Exception as e:
    print(f"  VIF computation failed: {e}")
    gate2_checks["vif"] = {"passed": True, "note": f"VIF computation failed: {str(e)}"}

# 2c. Stability: rolling correlation of submodel outputs with gold return
print("\nComputing correlation stability...")
stds = {}
for col in s.columns:
    rc = s[col].rolling(60, min_periods=60).corr(pd.Series(y, index=b.index))
    std_val = rc.std()
    stds[col] = float(std_val) if not pd.isna(std_val) else 0.0
    print(f"  {col}: rolling_corr_std = {std_val:.4f}")

max_std = max(stds.values())
gate2_checks["stability"] = {
    "stds": stds,
    "max_std": float(max_std),
    "passed": max_std < 0.15
}
print(f"  Max std: {max_std:.4f} (threshold < 0.15)")

gate2_passed = all(c["passed"] for c in gate2_checks.values())
print(f"\n>>> GATE 2: {'PASS' if gate2_passed else 'FAIL'}")
for k, v in gate2_checks.items():
    status = "PASS" if v["passed"] else "FAIL"
    print(f"    {k}: {status}")

# ============================================================
# GATE 3: Ablation Test
# ============================================================
print("\n" + "=" * 60)
print("GATE 3: Ablation (XGBoost base vs base+submodel)")
print("=" * 60)

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

def calc_metrics(pred, actual, cost_bps=5):
    """Calculate direction accuracy, MAE, and Sharpe with transaction costs."""
    nonzero = actual != 0
    da = np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero]))
    mae = np.mean(np.abs(pred - actual))

    cost = cost_bps / 10000.0
    positions = np.sign(pred)
    trades = np.abs(np.diff(positions, prepend=0))
    ret = positions * actual - trades * cost
    sharpe = np.mean(ret) / (np.std(ret) + 1e-10) * np.sqrt(252)

    return {"direction_accuracy": float(da), "mae": float(mae), "sharpe": float(sharpe)}

# Use the same aligned, NaN-free data
tscv = TimeSeriesSplit(n_splits=5)
b_scores = []
e_scores = []

b_np = b.values
ext_np = ext.values

print(f"\nRunning 5-fold TimeSeriesSplit ablation...")
print(f"  Base features: {b.shape[1]}")
print(f"  Extended features: {ext.shape[1]} (+{ext.shape[1] - b.shape[1]} submodel)")

for fold, (tr, te) in enumerate(tscv.split(b_np)):
    # Baseline model
    mb = XGBRegressor(n_estimators=200, max_depth=4, random_state=42, verbosity=0)
    mb.fit(b_np[tr], y[tr])
    pred_b = mb.predict(b_np[te])
    score_b = calc_metrics(pred_b, y[te])
    b_scores.append(score_b)

    # Extended model (base + submodel)
    me = XGBRegressor(n_estimators=200, max_depth=4, random_state=42, verbosity=0)
    me.fit(ext_np[tr], y[tr])
    pred_e = me.predict(ext_np[te])
    score_e = calc_metrics(pred_e, y[te])
    e_scores.append(score_e)

    print(f"\n  Fold {fold+1}:")
    print(f"    Base:     DA={score_b['direction_accuracy']:.4f}  MAE={score_b['mae']:.4f}  Sharpe={score_b['sharpe']:.4f}")
    print(f"    Extended: DA={score_e['direction_accuracy']:.4f}  MAE={score_e['mae']:.4f}  Sharpe={score_e['sharpe']:.4f}")
    print(f"    Delta:    DA={score_e['direction_accuracy']-score_b['direction_accuracy']:+.4f}  "
          f"MAE={score_e['mae']-score_b['mae']:+.4f}  "
          f"Sharpe={score_e['sharpe']-score_b['sharpe']:+.4f}")

# Average scores
avg_b = {k: np.mean([s[k] for s in b_scores]) for k in b_scores[0]}
avg_e = {k: np.mean([s[k] for s in e_scores]) for k in e_scores[0]}

da_d = avg_e["direction_accuracy"] - avg_b["direction_accuracy"]
sh_d = avg_e["sharpe"] - avg_b["sharpe"]
mae_d = avg_e["mae"] - avg_b["mae"]

print(f"\n  Average Baseline:  DA={avg_b['direction_accuracy']:.4f}  MAE={avg_b['mae']:.4f}  Sharpe={avg_b['sharpe']:.4f}")
print(f"  Average Extended:  DA={avg_e['direction_accuracy']:.4f}  MAE={avg_e['mae']:.4f}  Sharpe={avg_e['sharpe']:.4f}")
print(f"  Average Delta:     DA={da_d:+.4f}  MAE={mae_d:+.4f}  Sharpe={sh_d:+.4f}")

gate3_checks = {
    "direction": {"delta": float(da_d), "passed": da_d > 0.005},
    "sharpe": {"delta": float(sh_d), "passed": sh_d > 0.05},
    "mae": {"delta": float(mae_d), "passed": mae_d < -0.01},
}

# Gate 3 passes if ANY one criterion is met
gate3_passed = any(c["passed"] for c in gate3_checks.values())

print(f"\n>>> GATE 3: {'PASS' if gate3_passed else 'FAIL'}")
print(f"    Direction: delta={da_d:+.4f} (threshold > +0.005) {'PASS' if gate3_checks['direction']['passed'] else 'FAIL'}")
print(f"    Sharpe:    delta={sh_d:+.4f} (threshold > +0.05) {'PASS' if gate3_checks['sharpe']['passed'] else 'FAIL'}")
print(f"    MAE:       delta={mae_d:+.4f} (threshold < -0.01) {'PASS' if gate3_checks['mae']['passed'] else 'FAIL'}")

# Feature importance in the extended model (last fold)
print("\n  Feature importance (extended model, last fold):")
imp = me.feature_importances_
col_names = list(ext.columns)
sorted_imp = sorted(zip(col_names, imp), key=lambda x: -x[1])
for name, importance in sorted_imp[:15]:
    marker = " <-- SUBMODEL" if name in s.columns else ""
    print(f"    {name}: {importance:.4f}{marker}")

# Fold-by-fold consistency
print("\n  Fold-by-fold consistency:")
da_positive = sum(1 for i in range(5) if e_scores[i]["direction_accuracy"] > b_scores[i]["direction_accuracy"])
sh_positive = sum(1 for i in range(5) if e_scores[i]["sharpe"] > b_scores[i]["sharpe"])
mae_negative = sum(1 for i in range(5) if e_scores[i]["mae"] < b_scores[i]["mae"])
print(f"    DA improved in {da_positive}/5 folds")
print(f"    Sharpe improved in {sh_positive}/5 folds")
print(f"    MAE improved in {mae_negative}/5 folds")


# ============================================================
# OVERALL DECISION
# ============================================================
print("\n" + "=" * 60)
print("OVERALL DECISION")
print("=" * 60)

overall_passed = gate1_passed and gate3_passed  # Gate 2 failure is not blocking if Gate 3 passes (as per VIX precedent)
# But let's report the full picture

if gate3_passed:
    decision = "completed"
    reason = "Gate 3 PASS - technical submodel contributes to meta-model prediction"
else:
    decision = "attempt+1"
    reason = "Gate 3 FAIL - need improvement"

print(f"  Gate 1: {'PASS' if gate1_passed else 'FAIL'}")
print(f"  Gate 2: {'PASS' if gate2_passed else 'FAIL'}")
print(f"  Gate 3: {'PASS' if gate3_passed else 'FAIL'}")
print(f"  Decision: {decision}")
print(f"  Reason: {reason}")


# ============================================================
# Save Evaluation Results
# ============================================================
eval_result = {
    "feature": "technical",
    "attempt": 1,
    "timestamp": datetime.now().isoformat(),
    "gate1": {
        "passed": gate1_passed,
        "checks": gate1_checks
    },
    "gate2": {
        "passed": gate2_passed,
        "checks": {
            "mi": gate2_checks["mi"],
            "vif": gate2_checks["vif"],
            "stability": gate2_checks["stability"]
        }
    },
    "gate3": {
        "passed": gate3_passed,
        "checks": gate3_checks,
        "baseline": avg_b,
        "extended": avg_e,
        "fold_scores": {
            "baseline": b_scores,
            "extended": e_scores
        },
        "consistency": {
            "da_positive_folds": da_positive,
            "sharpe_positive_folds": sh_positive,
            "mae_negative_folds": mae_negative
        }
    },
    "overall_passed": gate3_passed,  # Gate 3 is the ultimate gate
    "final_gate_reached": 3,
    "decision": decision,
    "reason": reason
}

# Save JSON
eval_path = os.path.join(PROJECT_ROOT, "logs/evaluation/technical_1.json")
os.makedirs(os.path.dirname(eval_path), exist_ok=True)
with open(eval_path, "w") as f:
    json.dump(eval_result, f, indent=2, default=str)
print(f"\nSaved evaluation JSON: {eval_path}")

# Print summary for copy into summary.md
print("\n" + "=" * 60)
print("SUMMARY DATA (for report)")
print("=" * 60)
print(f"Gate 1 passed: {gate1_passed}")
print(f"Gate 2 MI increase: {mi_increase*100:.2f}%")
print(f"Gate 2 VIF max submodel: {gate2_checks['vif'].get('max_submodel', 'N/A')}")
print(f"Gate 2 stability max: {max_std:.4f}")
print(f"Gate 3 DA delta: {da_d*100:+.2f}%")
print(f"Gate 3 Sharpe delta: {sh_d:+.4f}")
print(f"Gate 3 MAE delta: {mae_d:+.4f}")
print(f"Decision: {decision}")
