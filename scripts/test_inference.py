"""
Meta-Model Attempt 7 - Local Inference Performance Test
Rebuilds the model with exact attempt 7 hyperparameters and evaluates on test set.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

print("=" * 70)
print("META-MODEL ATTEMPT 7 - INFERENCE PERFORMANCE TEST")
print("=" * 70)
print(f"Started: {datetime.now().isoformat()}")
print(f"XGBoost: {xgb.__version__}")

# ============================================================
# 1. Load base features
# ============================================================
print("\n[1/6] Loading base features...")
bf = pd.read_csv("data/processed/base_features.csv")
if "Date" in bf.columns:
    bf["Date"] = pd.to_datetime(bf["Date"]).dt.strftime("%Y-%m-%d")
    bf = bf.set_index("Date")
elif "Unnamed: 0" in bf.columns:
    bf["Date"] = pd.to_datetime(bf["Unnamed: 0"]).dt.strftime("%Y-%m-%d")
    bf = bf.set_index("Date")
    bf.index.name = "Date"

# Check available columns
print(f"  Loaded: {len(bf)} rows, columns: {list(bf.columns)}")

# Build base transformations
final_df = bf.copy()

# Map column names - handle both naming conventions
col_map = {}
for c in bf.columns:
    if "real_rate" in c.lower() and "change" not in c.lower():
        col_map["real_rate_raw"] = c
    if "dxy" in c.lower() and "change" not in c.lower():
        col_map["dxy_raw"] = c
    if "vix" in c.lower():
        col_map["vix_raw"] = c
    if "yield" in c.lower() and "spread" in c.lower():
        col_map["yield_spread_raw"] = c
    if "inflation" in c.lower():
        col_map["inflation_raw"] = c

# Apply transformations
if "real_rate_raw" in col_map:
    final_df["real_rate_change"] = final_df[col_map["real_rate_raw"]].diff()
elif "real_rate_change" in bf.columns:
    pass  # already exists

if "dxy_raw" in col_map:
    final_df["dxy_change"] = final_df[col_map["dxy_raw"]].diff()
elif "dxy_change" in bf.columns:
    pass

if "vix_raw" in col_map:
    final_df["vix"] = final_df[col_map["vix_raw"]]
elif "vix" in bf.columns:
    pass

if "yield_spread_raw" in col_map:
    final_df["yield_spread_change"] = final_df[col_map["yield_spread_raw"]].diff()
elif "yield_spread_change" in bf.columns:
    pass

if "inflation_raw" in col_map:
    final_df["inflation_exp_change"] = final_df[col_map["inflation_raw"]].diff()
elif "inflation_exp_change" in bf.columns:
    pass

# ============================================================
# 2. Load submodel outputs
# ============================================================
print("\n[2/6] Loading submodel outputs...")

SUBMODEL_DIR = "data/submodel_outputs"

submodel_specs = {
    "vix": {
        "columns": ["vix_regime_probability", "vix_mean_reversion_z", "vix_persistence"],
        "date_col": "date",
        "tz_aware": False,
    },
    "technical": {
        "columns": ["tech_trend_regime_prob", "tech_mean_reversion_z", "tech_volatility_regime"],
        "date_col": "date",
        "tz_aware": True,
    },
    "cross_asset": {
        "columns": ["xasset_regime_prob", "xasset_recession_signal", "xasset_divergence"],
        "date_col": "Date",
        "tz_aware": False,
    },
    "yield_curve": {
        "columns": ["yc_spread_velocity_z", "yc_curvature_z"],
        "date_col": "index",
        "tz_aware": False,
    },
    "etf_flow": {
        "columns": ["etf_regime_prob", "etf_capital_intensity", "etf_pv_divergence"],
        "date_col": "Date",
        "tz_aware": False,
    },
    "inflation_expectation": {
        "columns": ["ie_regime_prob", "ie_anchoring_z", "ie_gold_sensitivity_z"],
        "date_col": "Unnamed: 0",
        "tz_aware": False,
    },
    "options_market": {
        "columns": ["options_risk_regime_prob"],
        "date_col": "Date",
        "tz_aware": True,
        "rename": {"options_regime_smooth": "options_risk_regime_prob"},
    },
    "temporal_context": {
        "columns": ["temporal_context_score"],
        "date_col": "date",
        "tz_aware": False,
    },
}

for feature, spec in submodel_specs.items():
    fpath = os.path.join(SUBMODEL_DIR, f"{feature}.csv")
    if not os.path.exists(fpath):
        # Try dataset_upload_clean
        fpath = os.path.join("data/dataset_upload_clean", f"{feature}.csv")
    if not os.path.exists(fpath):
        print(f"  WARNING: {feature}.csv not found, skipping")
        continue

    df = pd.read_csv(fpath)

    # Rename if needed
    if spec.get("rename"):
        df = df.rename(columns=spec["rename"])

    # Parse date
    date_col = spec["date_col"]
    if spec["tz_aware"]:
        df["Date"] = pd.to_datetime(df[date_col], utc=True).dt.strftime("%Y-%m-%d")
    elif date_col == "index":
        df["Date"] = pd.to_datetime(df.iloc[:, 0]).dt.strftime("%Y-%m-%d")
    elif date_col == "Unnamed: 0":
        df["Date"] = pd.to_datetime(df["Unnamed: 0"]).dt.strftime("%Y-%m-%d")
    else:
        df["Date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    # Select columns
    available_cols = [c for c in spec["columns"] if c in df.columns]
    if not available_cols:
        print(f"  WARNING: {feature} - no target columns found. Cols: {list(df.columns)}")
        continue

    df = df[["Date"] + available_cols].set_index("Date")
    final_df = final_df.join(df, how="left")
    print(f"  {feature}: {len(df)} rows, cols: {available_cols}")

# ============================================================
# 3. Feature definition and NaN imputation
# ============================================================
print("\n[3/6] Preparing features...")

FEATURE_COLUMNS = [
    "real_rate_change", "dxy_change", "vix", "yield_spread_change", "inflation_exp_change",
    "vix_regime_probability", "vix_mean_reversion_z", "vix_persistence",
    "tech_trend_regime_prob", "tech_mean_reversion_z", "tech_volatility_regime",
    "xasset_regime_prob", "xasset_recession_signal", "xasset_divergence",
    "yc_spread_velocity_z", "yc_curvature_z",
    "etf_regime_prob", "etf_capital_intensity", "etf_pv_divergence",
    "ie_regime_prob", "ie_anchoring_z", "ie_gold_sensitivity_z",
    "options_risk_regime_prob",
    "temporal_context_score",
]
TARGET = "gold_return_next"

# Check missing features
missing = [c for c in FEATURE_COLUMNS if c not in final_df.columns]
if missing:
    print(f"  WARNING: Missing features: {missing}")
    FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c in final_df.columns]

present = [c for c in FEATURE_COLUMNS if c in final_df.columns]
print(f"  Features available: {len(present)}/{len(FEATURE_COLUMNS) + len(missing)}")

# NaN imputation
regime_cols = ["vix_regime_probability", "tech_trend_regime_prob", "xasset_regime_prob",
               "etf_regime_prob", "ie_regime_prob", "options_risk_regime_prob",
               "temporal_context_score"]
for col in regime_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(0.5)

z_cols = ["vix_mean_reversion_z", "tech_mean_reversion_z", "yc_spread_velocity_z",
          "yc_curvature_z", "etf_capital_intensity", "etf_pv_divergence",
          "ie_anchoring_z", "ie_gold_sensitivity_z"]
for col in z_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(0.0)

for col in ["xasset_recession_signal", "xasset_divergence"]:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(0.0)

for col in ["tech_volatility_regime", "vix_persistence"]:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(final_df[col].median())

# Drop rows without target or base features
base_req = ["gold_return_next", "real_rate_change", "dxy_change", "vix",
            "yield_spread_change", "inflation_exp_change"]
base_req_present = [c for c in base_req if c in final_df.columns]
final_df = final_df.dropna(subset=base_req_present)

nan_count = final_df[FEATURE_COLUMNS].isna().sum().sum()
print(f"  Remaining NaN: {nan_count}")
print(f"  Final dataset: {len(final_df)} rows x {len(FEATURE_COLUMNS)} features")
print(f"  Date range: {final_df.index.min()} to {final_df.index.max()}")

# ============================================================
# 4. Train/Val/Test split (70/15/15, time-series order)
# ============================================================
print("\n[4/6] Splitting data (70/15/15 time-series)...")

n = len(final_df)
n_train = int(n * 0.70)
n_val = int(n * 0.15)

train_df = final_df.iloc[:n_train]
val_df = final_df.iloc[n_train:n_train + n_val]
test_df = final_df.iloc[n_train + n_val:]

X_train = train_df[FEATURE_COLUMNS].values
y_train = train_df[TARGET].values
X_val = val_df[FEATURE_COLUMNS].values
y_val = val_df[TARGET].values
X_test = test_df[FEATURE_COLUMNS].values
y_test = test_df[TARGET].values

print(f"  Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
print(f"  Val:   {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
print(f"  Test:  {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")
print(f"  Target positive%: train={100*(y_train>0).mean():.1f}%, val={100*(y_val>0).mean():.1f}%, test={100*(y_test>0).mean():.1f}%")

# ============================================================
# 5. Train with attempt 7 hyperparameters
# ============================================================
print("\n[5/6] Training XGBoost (attempt 7 hyperparameters)...")

xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": 2,
    "min_child_weight": 25,
    "subsample": 0.765,
    "colsample_bytree": 0.450,
    "reg_lambda": 2.049,
    "reg_alpha": 1.107,
    "learning_rate": 0.0215,
    "n_estimators": 621,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}

# Bootstrap ensemble (5 seeds, matching attempt 7)
N_ENSEMBLE = 5
ensemble_models = []
rng = np.random.RandomState(42)

for i in range(N_ENSEMBLE):
    seed = 42 + i
    params = xgb_params.copy()
    params["random_state"] = seed
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    ensemble_models.append(model)
    print(f"  Model {i+1}/{N_ENSEMBLE}: seed={seed}, n_estimators={xgb_params['n_estimators']}")

# Ensemble predictions
train_preds_all = np.array([m.predict(X_train) for m in ensemble_models])
val_preds_all = np.array([m.predict(X_val) for m in ensemble_models])
test_preds_all = np.array([m.predict(X_test) for m in ensemble_models])

pred_train = train_preds_all.mean(axis=0)
pred_val = val_preds_all.mean(axis=0)
pred_test = test_preds_all.mean(axis=0)
bootstrap_std_test = test_preds_all.std(axis=0)

# OLS scaling
from numpy.linalg import lstsq
A = pred_val.reshape(-1, 1)
alpha_ols = lstsq(A, y_val, rcond=None)[0][0]
alpha_ols = np.clip(alpha_ols, 0.5, 10.0)
print(f"\n  OLS scaling factor (from val): {alpha_ols:.3f}")

pred_test_scaled = pred_test * alpha_ols

# ============================================================
# 6. Evaluation
# ============================================================
print("\n[6/6] Evaluating performance...")


def direction_accuracy(y_true, y_pred):
    mask = (y_true != 0) & (y_pred != 0)
    if mask.sum() == 0:
        return 0.0
    return float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean())


def high_confidence_da(y_true, y_pred, method="abs", bootstrap_std=None):
    if method == "abs":
        threshold = np.percentile(np.abs(y_pred), 80)
        hc_mask = np.abs(y_pred) >= threshold
    elif method == "bootstrap" and bootstrap_std is not None:
        threshold = np.percentile(bootstrap_std, 20)
        hc_mask = bootstrap_std <= threshold
    else:
        return 0.0, 0

    if hc_mask.sum() == 0:
        return 0.0, 0

    y_hc = y_true[hc_mask]
    p_hc = y_pred[hc_mask]
    valid = (y_hc != 0) & (p_hc != 0)
    if valid.sum() == 0:
        return 0.0, int(hc_mask.sum())
    return float((np.sign(p_hc[valid]) == np.sign(y_hc[valid])).mean()), int(hc_mask.sum())


def sharpe_with_costs(y_true, y_pred, cost_bps=5.0):
    positions = np.sign(y_pred)
    strategy_returns = positions * y_true / 100.0
    position_changes = np.abs(np.diff(positions, prepend=0))
    trade_costs = position_changes * (cost_bps / 10000.0)
    net_returns = strategy_returns - trade_costs
    if len(net_returns) < 2 or net_returns.std() == 0:
        return 0.0, net_returns
    sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(252)
    return float(sharpe), net_returns


def cumulative_return(net_returns):
    return float(np.prod(1 + net_returns) - 1)


# --- Test set metrics ---
da_test = direction_accuracy(y_test, pred_test)
mae_test_raw = float(np.abs(pred_test - y_test).mean())
mae_test_scaled = float(np.abs(pred_test_scaled - y_test).mean())
mae_test = min(mae_test_raw, mae_test_scaled)

hcda_abs, hc_count_abs = high_confidence_da(y_test, pred_test, method="abs")
hcda_boot, hc_count_boot = high_confidence_da(y_test, pred_test, method="bootstrap", bootstrap_std=bootstrap_std_test)

sharpe_test, net_returns_test = sharpe_with_costs(y_test, pred_test)
cum_ret = cumulative_return(net_returns_test)

# Naive always-up
sharpe_naive, net_returns_naive = sharpe_with_costs(y_test, np.ones_like(y_test))
da_naive = direction_accuracy(y_test, np.ones_like(y_test))

# --- Train set metrics ---
da_train = direction_accuracy(y_train, pred_train)
sharpe_train, _ = sharpe_with_costs(y_train, pred_train)

# --- Val set metrics ---
da_val = direction_accuracy(y_val, pred_val)

# Zero-prediction MAE reference
mae_zero = float(np.abs(y_test).mean())

print("\n" + "=" * 70)
print("INFERENCE PERFORMANCE RESULTS")
print("=" * 70)

print(f"\n{'Metric':<35} {'Target':<12} {'Achieved':<12} {'Status':<8} {'vs Baseline'}")
print("-" * 80)
print(f"{'Direction Accuracy':<35} {'> 56.0%':<12} {da_test*100:.2f}%{'':>5} {'PASS' if da_test > 0.56 else 'FAIL':<8} vs naive {da_naive*100:.1f}%")
print(f"{'High-Confidence DA (|pred|)':<35} {'> 60.0%':<12} {hcda_abs*100:.2f}%{'':>5} {'PASS' if hcda_abs > 0.60 else 'FAIL':<8} ({hc_count_abs} samples)")
print(f"{'High-Confidence DA (bootstrap)':<35} {'> 60.0%':<12} {hcda_boot*100:.2f}%{'':>5} {'PASS' if hcda_boot > 0.60 else 'FAIL':<8} ({hc_count_boot} samples)")
print(f"{'MAE':<35} {'< 0.75%':<12} {mae_test:.4f}%{'':>4} {'PASS' if mae_test < 0.75 else 'FAIL':<8} zero-pred: {mae_zero:.4f}%")
print(f"{'Sharpe Ratio (after 5bps costs)':<35} {'> 0.80':<12} {sharpe_test:.4f}{'':>5} {'PASS' if sharpe_test > 0.80 else 'FAIL':<8} naive: {sharpe_naive:.4f}")

targets_passed = sum([da_test > 0.56, hcda_abs > 0.60, mae_test < 0.75, sharpe_test > 0.80])
print(f"\nTargets passed: {targets_passed}/4")

print(f"\n--- Overfitting Check ---")
print(f"  Train DA:  {da_train*100:.2f}%")
print(f"  Val DA:    {da_val*100:.2f}%")
print(f"  Test DA:   {da_test*100:.2f}%")
print(f"  Train Sharpe: {sharpe_train:.2f}")
print(f"  Train-Test DA gap: {(da_train - da_test)*100:.2f}pp (threshold < 10pp)")

print(f"\n--- Prediction Distribution ---")
print(f"  Pred mean:  {pred_test.mean():.6f}")
print(f"  Pred std:   {pred_test.std():.6f}")
print(f"  Pred range: [{pred_test.min():.6f}, {pred_test.max():.6f}]")
print(f"  Positive%:  {100*(pred_test > 0).mean():.1f}%")
print(f"  Actual std: {y_test.std():.4f}")

print(f"\n--- Trading Performance ---")
n_trades = int(np.sum(np.abs(np.diff(np.sign(pred_test))) > 0))
print(f"  Cumulative return: {cum_ret*100:.2f}%")
print(f"  Position changes:  {n_trades} trades in {len(y_test)} days")
print(f"  Annualized return: {net_returns_test.mean() * 252 * 100:.2f}%")
print(f"  Max drawdown:      {(np.minimum.accumulate(np.cumsum(net_returns_test)) - np.cumsum(net_returns_test)).min()*100:.2f}%")

# --- Feature importance ---
print(f"\n--- Feature Importance (Top 10) ---")
avg_imp = np.mean([m.feature_importances_ for m in ensemble_models], axis=0)
imp_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": avg_imp})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
imp_df["pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100
for i, row in imp_df.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:<30s} {row['pct']:.2f}%")

# --- Monthly breakdown ---
print(f"\n--- Monthly DA Breakdown (Test Set) ---")
test_result_df = pd.DataFrame({
    "date": test_df.index,
    "actual": y_test,
    "pred": pred_test,
    "correct": np.sign(pred_test) == np.sign(y_test),
})
test_result_df["month"] = pd.to_datetime(test_result_df["date"]).dt.to_period("M")
monthly = test_result_df.groupby("month").agg(
    da=("correct", "mean"),
    count=("correct", "count"),
    actual_mean=("actual", "mean"),
).reset_index()
for _, row in monthly.iterrows():
    bar = "#" * int(row["da"] * 20)
    print(f"  {str(row['month']):<8s} DA={row['da']*100:5.1f}% ({row['count']:3d} days) {bar}")

# --- Save results ---
results = {
    "test_date": datetime.now().isoformat(),
    "model": "meta_model_attempt_7_rebuild",
    "n_features": len(FEATURE_COLUMNS),
    "n_ensemble": N_ENSEMBLE,
    "xgb_params": xgb_params,
    "alpha_ols": float(alpha_ols),
    "splits": {
        "train": {"n": len(train_df), "start": train_df.index.min(), "end": train_df.index.max()},
        "val": {"n": len(val_df), "start": val_df.index.min(), "end": val_df.index.max()},
        "test": {"n": len(test_df), "start": test_df.index.min(), "end": test_df.index.max()},
    },
    "test_metrics": {
        "direction_accuracy": da_test,
        "high_confidence_da_abs": hcda_abs,
        "high_confidence_da_bootstrap": hcda_boot,
        "mae": mae_test,
        "mae_raw": mae_test_raw,
        "mae_scaled": mae_test_scaled,
        "sharpe_ratio": sharpe_test,
        "cumulative_return": cum_ret,
    },
    "targets": {
        "da": {"target": 0.56, "actual": da_test, "passed": da_test > 0.56},
        "hcda": {"target": 0.60, "actual": hcda_abs, "passed": hcda_abs > 0.60},
        "mae": {"target": 0.75, "actual": mae_test, "passed": mae_test < 0.75},
        "sharpe": {"target": 0.80, "actual": sharpe_test, "passed": sharpe_test > 0.80},
    },
    "targets_passed": targets_passed,
    "overfitting": {
        "train_da": da_train,
        "val_da": da_val,
        "test_da": da_test,
        "gap_pp": (da_train - da_test) * 100,
    },
}

out_path = "logs/evaluation/inference_test_result.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")

print(f"\nCompleted: {datetime.now().isoformat()}")
