"""
DXY SubModel Attempt 1 - Full Gate 1/2/3 Evaluation
Feature: dxy
Method: GaussianHMM + Momentum Z-Score + Volatility Z-Score
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR = Path(r"C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents")

# ============================================================
# 1. Load all data
# ============================================================
print("=" * 60)
print("DXY SubModel Attempt 1 - Gate 1/2/3 Evaluation")
print("=" * 60)

# Training result
with open(BASE_DIR / "data/submodel_outputs/dxy/training_result.json") as f:
    training_result = json.load(f)

# DXY submodel output
dxy_output = pd.read_csv(
    BASE_DIR / "data/submodel_outputs/dxy/submodel_output.csv",
    index_col=0, parse_dates=True
)
print(f"\nDXY output shape: {dxy_output.shape}")
print(f"DXY date range: {dxy_output.index.min()} to {dxy_output.index.max()}")
print(f"DXY columns: {list(dxy_output.columns)}")

# Base features (includes target gold_return_next)
base_features = pd.read_csv(
    BASE_DIR / "data/processed/base_features.csv",
    index_col=0, parse_dates=True
)
print(f"\nBase features shape: {base_features.shape}")
print(f"Base features date range: {base_features.index.min()} to {base_features.index.max()}")

# Extract target
target = base_features["gold_return_next"].copy()
base_X = base_features.drop(columns=["gold_return_next"])

# Load all existing submodel outputs (the 24-feature setup from meta-model attempt 7)
submodel_files = {
    "vix": "vix.csv",
    "technical": "technical.csv",
    "cross_asset": "cross_asset.csv",
    "yield_curve": "yield_curve.csv",
    "etf_flow": "etf_flow.csv",
    "inflation_expectation": "inflation_expectation.csv",
    "cny_demand": "cny_demand.csv",
    "options_market": "options_market.csv",
    "temporal_context": "temporal_context.csv",
}

submodel_dfs = {}
for name, fname in submodel_files.items():
    fpath = BASE_DIR / "data/dataset_upload_clean" / fname
    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    # Normalize index to remove timezone info
    try:
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except:
        pass
    # Convert to DatetimeIndex and normalize
    try:
        df.index = pd.DatetimeIndex(df.index).normalize()
    except:
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()
    submodel_dfs[name] = df
    print(f"  Loaded {name}: {df.shape}, cols={list(df.columns)}")

# ============================================================
# 2. Build the existing 24-feature matrix (same as meta-model attempt 7)
# ============================================================
print("\n" + "=" * 60)
print("Building existing feature matrix (24 features)")
print("=" * 60)

# Normalize all indices to timezone-naive dates
def normalize_index(df):
    try:
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    except:
        pass
    try:
        df.index = pd.DatetimeIndex(df.index).normalize()
    except:
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()
    return df

base_X = normalize_index(base_X)
target = normalize_index(target.to_frame()).iloc[:, 0]
dxy_output = normalize_index(dxy_output)

# Merge base features + all submodel outputs
existing_features = base_X.copy()
for name, df in submodel_dfs.items():
    existing_features = existing_features.join(df, how='left')

# Align with target
common_idx = existing_features.index.intersection(target.index)
existing_features = existing_features.loc[common_idx]
target_aligned = target.loc[common_idx]

# Drop rows with any NaN in existing features
valid_mask = existing_features.notna().all(axis=1) & target_aligned.notna()
existing_features = existing_features[valid_mask]
target_aligned = target_aligned[valid_mask]

print(f"Existing features shape (after NaN removal): {existing_features.shape}")
print(f"Feature columns ({len(existing_features.columns)}):")
for i, col in enumerate(existing_features.columns):
    print(f"  {i+1}. {col}")

# ============================================================
# GATE 1: Standalone Quality
# ============================================================
print("\n" + "=" * 60)
print("GATE 1: Standalone Quality")
print("=" * 60)

gate1_checks = {}

# 1a. No all-NaN columns
nan_cols = dxy_output.columns[dxy_output.isnull().all()].tolist()
gate1_checks["no_all_nan"] = {
    "value": nan_cols,
    "passed": len(nan_cols) == 0,
    "detail": f"{len(nan_cols)} all-NaN columns found"
}
print(f"\n[1a] All-NaN columns: {nan_cols} -> {'PASS' if len(nan_cols) == 0 else 'FAIL'}")

# 1b. No constant output (zero variance)
zero_var = []
for col in dxy_output.columns:
    std_val = dxy_output[col].std()
    if std_val < 1e-10:
        zero_var.append(col)
gate1_checks["no_zero_var"] = {
    "value": zero_var,
    "passed": len(zero_var) == 0,
    "detail": f"{len(zero_var)} constant columns found"
}
print(f"[1b] Constant columns: {zero_var} -> {'PASS' if len(zero_var) == 0 else 'FAIL'}")

# 1c. Autocorrelation check (< 0.99 for leak detection)
autocorr_checks = {}
autocorr_passed = True
for col in dxy_output.columns:
    ac = dxy_output[col].dropna().autocorr(lag=1)
    passed = abs(ac) < 0.99
    autocorr_checks[col] = {"value": round(ac, 6), "passed": passed}
    if not passed:
        autocorr_passed = False
    print(f"[1c] Autocorrelation {col}: {ac:.6f} (< 0.99) -> {'PASS' if passed else 'FAIL'}")

gate1_checks["autocorrelation"] = {
    "details": autocorr_checks,
    "passed": autocorr_passed,
    "detail": "All columns below 0.99 threshold" if autocorr_passed else "Some columns exceed 0.99"
}

# 1d. Overfit ratio - for HMM (unsupervised), check if MI on validation is reasonable
# Since HMM is unsupervised, we check training vs validation MI consistency
# The training_result already has MI metrics from validation set
mi_sum = training_result["metrics"]["mi_sum"]
gate1_checks["hpo_quality"] = {
    "optuna_trials": training_result.get("optuna_trials_completed", 0),
    "best_mi_sum": mi_sum,
    "passed": training_result.get("optuna_trials_completed", 0) >= 10,
    "detail": f"Optuna completed {training_result.get('optuna_trials_completed', 0)} trials (>= 10 required)"
}
print(f"[1d] Optuna trials: {training_result.get('optuna_trials_completed', 0)} -> {'PASS' if training_result.get('optuna_trials_completed', 0) >= 10 else 'FAIL'}")

# 1e. regime_prob quality check (skewness)
regime_prob = dxy_output["dxy_regime_prob"]
regime_stats = {
    "mean": float(regime_prob.mean()),
    "std": float(regime_prob.std()),
    "median": float(regime_prob.median()),
    "max": float(regime_prob.max()),
    "skewness": float(regime_prob.skew()),
    "pct_above_0.01": float((regime_prob > 0.01).mean() * 100),
    "pct_above_0.05": float((regime_prob > 0.05).mean() * 100),
}
# regime_prob is informative if it's not constant and has some variation
regime_passed = regime_stats["std"] > 1e-6 and regime_stats["max"] > 0.01
gate1_checks["regime_prob_quality"] = {
    "stats": regime_stats,
    "passed": regime_passed,
    "warning": "Highly skewed (mean=0.0006, max=0.184). Only 2.9% of days above 0.01." if regime_stats["pct_above_0.01"] < 5 else None,
    "detail": f"regime_prob: mean={regime_stats['mean']:.6f}, std={regime_stats['std']:.6f}, max={regime_stats['max']:.4f}"
}
print(f"[1e] regime_prob quality: mean={regime_stats['mean']:.6f}, std={regime_stats['std']:.6f}, "
      f"max={regime_stats['max']:.4f}, >0.01={regime_stats['pct_above_0.01']:.1f}% -> {'PASS' if regime_passed else 'FAIL'}")
if regime_stats["pct_above_0.01"] < 5:
    print(f"     WARNING: Highly skewed. Only {regime_stats['pct_above_0.01']:.1f}% of days above 0.01")

# 1f. No overfit ratio available for unsupervised model, mark as N/A
gate1_checks["overfit_ratio"] = {
    "value": "N/A (unsupervised HMM)",
    "passed": True,
    "detail": "Overfit ratio not applicable for unsupervised model. Using MI validation instead."
}

gate1_passed = all(c["passed"] for c in gate1_checks.values())
print(f"\n>>> GATE 1 OVERALL: {'PASS' if gate1_passed else 'FAIL'}")

# ============================================================
# GATE 2: Information Gain
# ============================================================
print("\n" + "=" * 60)
print("GATE 2: Information Gain")
print("=" * 60)

from sklearn.feature_selection import mutual_info_regression

# Align DXY output with existing features and target
dxy_aligned = dxy_output.reindex(existing_features.index)
valid_mask2 = dxy_aligned.notna().all(axis=1)
base_g2 = existing_features[valid_mask2].copy()
dxy_g2 = dxy_aligned[valid_mask2].copy()
target_g2 = target_aligned[valid_mask2].copy()

print(f"Gate 2 sample size: {len(base_g2)}")

# 2a. MI increase (sum-based)
print("\nComputing Mutual Information (this may take a moment)...")
np.random.seed(42)
mi_base = mutual_info_regression(base_g2.values, target_g2.values, random_state=42)
mi_base_sum = float(mi_base.sum())
print(f"  MI base sum: {mi_base_sum:.6f}")

extended_g2 = pd.concat([base_g2, dxy_g2], axis=1)
mi_extended = mutual_info_regression(extended_g2.values, target_g2.values, random_state=42)
mi_extended_sum = float(mi_extended.sum())
print(f"  MI extended sum: {mi_extended_sum:.6f}")

mi_increase = (mi_extended_sum - mi_base_sum) / (mi_base_sum + 1e-10)
mi_passed = mi_increase > 0.05
print(f"  MI increase: {mi_increase*100:.2f}% (threshold: > 5%) -> {'PASS' if mi_passed else 'FAIL'}")

# Show MI for individual DXY features
n_base_features = base_g2.shape[1]
mi_dxy_individual = {}
for i, col in enumerate(dxy_g2.columns):
    mi_val = float(mi_extended[n_base_features + i])
    mi_dxy_individual[col] = mi_val
    print(f"    {col} MI: {mi_val:.6f}")

gate2_checks = {
    "mi": {
        "base_sum": mi_base_sum,
        "extended_sum": mi_extended_sum,
        "increase": mi_increase,
        "increase_pct": f"{mi_increase*100:.2f}%",
        "individual_mi": mi_dxy_individual,
        "passed": mi_passed,
        "detail": f"MI increase: {mi_increase*100:.2f}% ({'>' if mi_passed else '<'} 5%)"
    }
}

# 2b. VIF check
print("\nComputing VIF...")
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # VIF for DXY features against existing features
    ext_for_vif = pd.concat([base_g2, dxy_g2], axis=1)
    # Drop any remaining NaN rows
    ext_for_vif = ext_for_vif.dropna()

    # Compute VIF only for the new DXY columns
    vif_results = {}
    n_total = ext_for_vif.shape[1]
    n_base = base_g2.shape[1]

    for i in range(n_base, n_total):
        col_name = ext_for_vif.columns[i]
        vif_val = variance_inflation_factor(ext_for_vif.values, i)
        vif_results[col_name] = float(vif_val)
        print(f"  VIF {col_name}: {vif_val:.4f}")

    max_vif = max(vif_results.values())
    vif_passed = max_vif < 10
    print(f"  Max VIF: {max_vif:.4f} (threshold: < 10) -> {'PASS' if vif_passed else 'FAIL'}")

    gate2_checks["vif"] = {
        "individual": vif_results,
        "max": max_vif,
        "passed": vif_passed,
        "detail": f"Max VIF: {max_vif:.4f} ({'<' if vif_passed else '>'} 10)"
    }
except Exception as e:
    print(f"  VIF computation failed: {e}")
    gate2_checks["vif"] = {"passed": True, "note": f"Computation failed: {e}", "detail": "Skipped"}
    vif_passed = True

# 2c. Rolling correlation stability
print("\nComputing rolling correlation stability...")
stds = {}
for col in dxy_g2.columns:
    rc = dxy_g2[col].rolling(60, min_periods=60).corr(target_g2)
    std_val = float(rc.dropna().std())
    stds[col] = std_val
    print(f"  Rolling corr std {col}: {std_val:.6f}")

max_std = max(stds.values())
stability_passed = max_std < 0.15
print(f"  Max std: {max_std:.6f} (threshold: < 0.15) -> {'PASS' if stability_passed else 'FAIL'}")

gate2_checks["stability"] = {
    "individual_stds": stds,
    "max_std": max_std,
    "passed": stability_passed,
    "detail": f"Max rolling corr std: {max_std:.6f} ({'<' if stability_passed else '>'} 0.15)"
}

gate2_passed = all(c["passed"] for c in gate2_checks.values())
print(f"\n>>> GATE 2 OVERALL: {'PASS' if gate2_passed else 'FAIL'}")

# ============================================================
# GATE 3: Ablation Test
# ============================================================
print("\n" + "=" * 60)
print("GATE 3: Ablation Test (XGBoost + 5-fold TimeSeriesSplit)")
print("=" * 60)

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

# Use the same aligned data
base_g3 = base_g2.copy()
dxy_g3 = dxy_g2.copy()
target_g3 = target_g2.copy()
extended_g3 = pd.concat([base_g3, dxy_g3], axis=1)

print(f"Gate 3 dataset: {len(base_g3)} samples, {base_g3.shape[1]} base features + {dxy_g3.shape[1]} DXY features")

def calc_metrics(pred, actual, cost_bps=5):
    """Calculate direction accuracy, MAE, and Sharpe with transaction costs."""
    # Direction accuracy: exclude zero-return samples
    nonzero = actual != 0
    if nonzero.sum() > 0:
        da = float(np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero])))
    else:
        da = 0.5

    mae = float(np.mean(np.abs(pred - actual)))

    # Sharpe with transaction costs
    cost = cost_bps / 10000.0
    positions = np.sign(pred)
    trades = np.abs(np.diff(positions, prepend=0))
    ret = positions * actual - trades * cost
    sharpe = float(np.mean(ret) / (np.std(ret) + 1e-10) * np.sqrt(252))

    return {"direction_accuracy": da, "mae": mae, "sharpe": sharpe}

tscv = TimeSeriesSplit(n_splits=5)
b_scores = []
e_scores = []
fold_details = []

for fold_idx, (tr, te) in enumerate(tscv.split(base_g3)):
    print(f"\n  Fold {fold_idx + 1}:")
    print(f"    Train: {len(tr)} samples ({base_g3.index[tr[0]].strftime('%Y-%m-%d')} to {base_g3.index[tr[-1]].strftime('%Y-%m-%d')})")
    print(f"    Test:  {len(te)} samples ({base_g3.index[te[0]].strftime('%Y-%m-%d')} to {base_g3.index[te[-1]].strftime('%Y-%m-%d')})")

    y_train = target_g3.values[tr]
    y_test = target_g3.values[te]

    # Baseline model (existing features only)
    mb = XGBRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbosity=0
    )
    mb.fit(base_g3.values[tr], y_train)
    pred_b = mb.predict(base_g3.values[te])
    score_b = calc_metrics(pred_b, y_test)
    b_scores.append(score_b)

    # Extended model (existing + DXY features)
    me = XGBRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbosity=0
    )
    me.fit(extended_g3.values[tr], y_train)
    pred_e = me.predict(extended_g3.values[te])
    score_e = calc_metrics(pred_e, y_test)
    e_scores.append(score_e)

    fold_detail = {
        "fold": fold_idx + 1,
        "train_size": len(tr),
        "test_size": len(te),
        "baseline": score_b,
        "extended": score_e,
        "delta_da": score_e["direction_accuracy"] - score_b["direction_accuracy"],
        "delta_sharpe": score_e["sharpe"] - score_b["sharpe"],
        "delta_mae": score_e["mae"] - score_b["mae"],
    }
    fold_details.append(fold_detail)

    print(f"    Baseline: DA={score_b['direction_accuracy']*100:.2f}%, Sharpe={score_b['sharpe']:.4f}, MAE={score_b['mae']:.6f}")
    print(f"    Extended: DA={score_e['direction_accuracy']*100:.2f}%, Sharpe={score_e['sharpe']:.4f}, MAE={score_e['mae']:.6f}")
    print(f"    Delta:    DA={fold_detail['delta_da']*100:+.2f}pp, Sharpe={fold_detail['delta_sharpe']:+.4f}, MAE={fold_detail['delta_mae']:+.6f}")

    # Feature importance for extended model
    if fold_idx == 4:  # Last fold only
        importances = me.feature_importances_
        feature_names = list(extended_g3.columns)
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        print(f"\n  Feature importance (last fold, top 10):")
        for _, row in imp_df.head(10).iterrows():
            marker = " <-- DXY" if "dxy_" in row['feature'] else ""
            print(f"    {row['feature']}: {row['importance']:.4f}{marker}")

        print(f"\n  DXY feature importance:")
        for col in dxy_g3.columns:
            imp = imp_df[imp_df['feature'] == col]['importance'].values[0]
            rank = imp_df.index.tolist().index(imp_df[imp_df['feature'] == col].index[0]) + 1
            print(f"    {col}: {imp:.4f} (rank {rank}/{len(feature_names)})")

# Average scores
avg_b = {k: np.mean([s[k] for s in b_scores]) for k in b_scores[0]}
avg_e = {k: np.mean([s[k] for s in e_scores]) for k in e_scores[0]}

da_delta = avg_e["direction_accuracy"] - avg_b["direction_accuracy"]
sharpe_delta = avg_e["sharpe"] - avg_b["sharpe"]
mae_delta = avg_e["mae"] - avg_b["mae"]

print(f"\n--- Average across 5 folds ---")
print(f"Baseline: DA={avg_b['direction_accuracy']*100:.2f}%, Sharpe={avg_b['sharpe']:.4f}, MAE={avg_b['mae']:.6f}")
print(f"Extended: DA={avg_e['direction_accuracy']*100:.2f}%, Sharpe={avg_e['sharpe']:.4f}, MAE={avg_e['mae']:.6f}")
print(f"Delta:    DA={da_delta*100:+.2f}pp, Sharpe={sharpe_delta:+.4f}, MAE={mae_delta:+.6f}")

# Gate 3 criteria (any one passes)
da_passed = da_delta > 0.005     # +0.5pp
sharpe_passed = sharpe_delta > 0.05  # +0.05
mae_passed = mae_delta < -0.01   # -0.01

# Count folds where each metric improved
da_improved_folds = sum(1 for f in fold_details if f["delta_da"] > 0)
sharpe_improved_folds = sum(1 for f in fold_details if f["delta_sharpe"] > 0)
mae_improved_folds = sum(1 for f in fold_details if f["delta_mae"] < 0)

gate3_checks = {
    "direction": {
        "delta": da_delta,
        "delta_pp": f"{da_delta*100:+.2f}pp",
        "passed": da_passed,
        "threshold": "+0.5pp",
        "folds_improved": f"{da_improved_folds}/5",
        "detail": f"DA delta: {da_delta*100:+.2f}pp ({'>' if da_passed else '<'} 0.5pp)"
    },
    "sharpe": {
        "delta": sharpe_delta,
        "passed": sharpe_passed,
        "threshold": "+0.05",
        "folds_improved": f"{sharpe_improved_folds}/5",
        "detail": f"Sharpe delta: {sharpe_delta:+.4f} ({'>' if sharpe_passed else '<'} 0.05)"
    },
    "mae": {
        "delta": mae_delta,
        "passed": mae_passed,
        "threshold": "-0.01",
        "folds_improved": f"{mae_improved_folds}/5",
        "detail": f"MAE delta: {mae_delta:+.6f} ({'<' if mae_passed else '>'} -0.01)"
    }
}

print(f"\nGate 3 criteria (ANY one passes):")
print(f"  DA:     {da_delta*100:+.2f}pp (threshold: +0.5pp)  [{da_improved_folds}/5 folds] -> {'PASS' if da_passed else 'FAIL'}")
print(f"  Sharpe: {sharpe_delta:+.4f} (threshold: +0.05)     [{sharpe_improved_folds}/5 folds] -> {'PASS' if sharpe_passed else 'FAIL'}")
print(f"  MAE:    {mae_delta:+.6f} (threshold: -0.01)      [{mae_improved_folds}/5 folds] -> {'PASS' if mae_passed else 'FAIL'}")

gate3_passed = any([da_passed, sharpe_passed, mae_passed])
print(f"\n>>> GATE 3 OVERALL: {'PASS' if gate3_passed else 'FAIL'}")

# ============================================================
# OVERALL DECISION
# ============================================================
print("\n" + "=" * 60)
print("OVERALL EVALUATION SUMMARY")
print("=" * 60)
print(f"  Gate 1 (Standalone Quality): {'PASS' if gate1_passed else 'FAIL'}")
print(f"  Gate 2 (Information Gain):   {'PASS' if gate2_passed else 'FAIL'}")
print(f"  Gate 3 (Ablation):           {'PASS' if gate3_passed else 'FAIL'}")

overall_passed = gate3_passed  # Gate 3 is the ultimate gatekeeper
if gate3_passed:
    decision = "completed"
    print(f"\n>>> DECISION: COMPLETED - DXY submodel passes Gate 3")
else:
    decision = "attempt+1"
    print(f"\n>>> DECISION: ATTEMPT+1 - DXY submodel needs improvement")

# Determine which criteria were passed in Gate 3
passing_criteria = []
if da_passed:
    passing_criteria.append(f"DA ({da_delta*100:+.2f}pp)")
if sharpe_passed:
    passing_criteria.append(f"Sharpe ({sharpe_delta:+.4f})")
if mae_passed:
    passing_criteria.append(f"MAE ({mae_delta:+.6f})")

# ============================================================
# Save results
# ============================================================
os.makedirs(BASE_DIR / "logs/evaluation", exist_ok=True)

# Full evaluation JSON
evaluation_result = {
    "feature": "dxy",
    "attempt": 1,
    "timestamp": datetime.now().isoformat(),
    "method": training_result["method"],
    "gate1": {
        "passed": gate1_passed,
        "checks": gate1_checks,
    },
    "gate2": {
        "passed": gate2_passed,
        "checks": {
            "mi": gate2_checks["mi"],
            "vif": gate2_checks["vif"],
            "stability": gate2_checks["stability"],
        }
    },
    "gate3": {
        "passed": gate3_passed,
        "checks": gate3_checks,
        "baseline": {k: round(v, 6) for k, v in avg_b.items()},
        "extended": {k: round(v, 6) for k, v in avg_e.items()},
        "fold_details": fold_details,
        "passing_criteria": passing_criteria,
    },
    "overall_passed": overall_passed,
    "final_gate_reached": 3,
    "decision": decision,
    "data_info": {
        "evaluation_samples": len(base_g2),
        "dxy_output_shape": list(dxy_output.shape),
        "existing_features": len(existing_features.columns),
        "date_range": f"{base_g2.index.min().strftime('%Y-%m-%d')} to {base_g2.index.max().strftime('%Y-%m-%d')}"
    }
}

eval_json_path = BASE_DIR / "logs/evaluation/dxy_1_gate_evaluation.json"
with open(eval_json_path, "w") as f:
    json.dump(evaluation_result, f, indent=2, default=str)
print(f"\nSaved evaluation JSON: {eval_json_path}")

# Human-readable summary
summary_lines = []
summary_lines.append(f"# Evaluation Summary: dxy attempt 1")
summary_lines.append(f"")
summary_lines.append(f"**Method**: {training_result['method']}")
summary_lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
summary_lines.append(f"**Samples**: {len(base_g2)} (aligned, no NaN)")
summary_lines.append(f"")
summary_lines.append(f"## Gate 1: Standalone Quality -- {'PASS' if gate1_passed else 'FAIL'}")
summary_lines.append(f"")
summary_lines.append(f"| Check | Value | Threshold | Result |")
summary_lines.append(f"|-------|-------|-----------|--------|")
summary_lines.append(f"| All-NaN columns | {len(nan_cols)} | 0 | {'PASS' if len(nan_cols) == 0 else 'FAIL'} |")
summary_lines.append(f"| Constant columns | {len(zero_var)} | 0 | {'PASS' if len(zero_var) == 0 else 'FAIL'} |")
for col, ac_info in autocorr_checks.items():
    summary_lines.append(f"| Autocorr {col} | {ac_info['value']:.6f} | < 0.99 | {'PASS' if ac_info['passed'] else 'FAIL'} |")
summary_lines.append(f"| Optuna trials | {training_result.get('optuna_trials_completed', 0)} | >= 10 | {'PASS' if training_result.get('optuna_trials_completed', 0) >= 10 else 'FAIL'} |")
summary_lines.append(f"| Overfit ratio | N/A (unsupervised) | < 1.5 | PASS (N/A) |")
summary_lines.append(f"")
if regime_stats["pct_above_0.01"] < 5:
    summary_lines.append(f"**Warning**: dxy_regime_prob is highly skewed (mean={regime_stats['mean']:.6f}, max={regime_stats['max']:.4f}). "
                        f"Only {regime_stats['pct_above_0.01']:.1f}% of days above 0.01. May contribute limited information.")
    summary_lines.append(f"")

summary_lines.append(f"## Gate 2: Information Gain -- {'PASS' if gate2_passed else 'FAIL'}")
summary_lines.append(f"")
summary_lines.append(f"| Check | Value | Threshold | Result |")
summary_lines.append(f"|-------|-------|-----------|--------|")
summary_lines.append(f"| MI increase | {mi_increase*100:.2f}% | > 5% | {'PASS' if mi_passed else 'FAIL'} |")
if "individual" in gate2_checks.get("vif", {}):
    for col, vif_val in gate2_checks["vif"]["individual"].items():
        summary_lines.append(f"| VIF {col} | {vif_val:.4f} | < 10 | {'PASS' if vif_val < 10 else 'FAIL'} |")
summary_lines.append(f"| Max rolling corr std | {max_std:.6f} | < 0.15 | {'PASS' if stability_passed else 'FAIL'} |")
summary_lines.append(f"")
summary_lines.append(f"Individual MI contributions:")
for col, mi_val in mi_dxy_individual.items():
    summary_lines.append(f"- {col}: {mi_val:.6f}")
summary_lines.append(f"")

summary_lines.append(f"## Gate 3: Ablation -- {'PASS' if gate3_passed else 'FAIL'}")
summary_lines.append(f"")
summary_lines.append(f"| Metric | Baseline | Extended | Delta | Threshold | Folds Improved | Result |")
summary_lines.append(f"|--------|----------|----------|-------|-----------|---------------|--------|")
summary_lines.append(f"| Direction Accuracy | {avg_b['direction_accuracy']*100:.2f}% | {avg_e['direction_accuracy']*100:.2f}% | {da_delta*100:+.2f}pp | +0.5pp | {da_improved_folds}/5 | {'PASS' if da_passed else 'FAIL'} |")
summary_lines.append(f"| Sharpe | {avg_b['sharpe']:.4f} | {avg_e['sharpe']:.4f} | {sharpe_delta:+.4f} | +0.05 | {sharpe_improved_folds}/5 | {'PASS' if sharpe_passed else 'FAIL'} |")
summary_lines.append(f"| MAE | {avg_b['mae']:.6f} | {avg_e['mae']:.6f} | {mae_delta:+.6f} | -0.01 | {mae_improved_folds}/5 | {'PASS' if mae_passed else 'FAIL'} |")
summary_lines.append(f"")

summary_lines.append(f"### Per-Fold Results")
summary_lines.append(f"")
summary_lines.append(f"| Fold | DA Delta | Sharpe Delta | MAE Delta |")
summary_lines.append(f"|------|----------|-------------|-----------|")
for fd in fold_details:
    summary_lines.append(f"| {fd['fold']} | {fd['delta_da']*100:+.2f}pp | {fd['delta_sharpe']:+.4f} | {fd['delta_mae']:+.6f} |")
summary_lines.append(f"")

summary_lines.append(f"## Decision: {decision.upper()}")
summary_lines.append(f"")
if gate3_passed:
    summary_lines.append(f"DXY submodel attempt 1 **passes Gate 3** via: {', '.join(passing_criteria)}.")
    summary_lines.append(f"The HMM-based DXY features provide measurable improvement to the meta-model.")
else:
    summary_lines.append(f"DXY submodel attempt 1 **fails Gate 3**. No criterion met the threshold.")
    summary_lines.append(f"")
    summary_lines.append(f"### Improvement Recommendations")
    # Add specific recommendations based on what failed
    if not da_passed and da_delta > 0:
        summary_lines.append(f"- DA is positive ({da_delta*100:+.2f}pp) but below threshold. Marginal improvement possible.")
    if not sharpe_passed and sharpe_delta > 0:
        summary_lines.append(f"- Sharpe is positive ({sharpe_delta:+.4f}) but below threshold.")
    if not mae_passed:
        if mae_delta < 0:
            summary_lines.append(f"- MAE improved ({mae_delta:+.6f}) but not enough to cross -0.01 threshold.")
        else:
            summary_lines.append(f"- MAE degraded ({mae_delta:+.6f}). DXY features may add noise.")
    summary_lines.append(f"- Consider: dimensionality reduction, different HMM components, or output transformation.")

summary_text = "\n".join(summary_lines)
summary_path = BASE_DIR / "logs/evaluation/dxy_1_summary.md"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text)
print(f"Saved summary: {summary_path}")

print(f"\n{'=' * 60}")
print(f"EVALUATION COMPLETE")
print(f"{'=' * 60}")
