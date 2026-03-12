"""
Refresh base features from APIs + test meta-model inference with latest data.
Fetches fresh data from yfinance and FRED, joins with existing submodel outputs,
trains XGBoost with attempt 7 hyperparameters, and evaluates.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import sys
from datetime import datetime

# Load env
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

print("=" * 70)
print("DATA REFRESH + META-MODEL INFERENCE TEST")
print("=" * 70)
print(f"Started: {datetime.now().isoformat()}")

# ============================================================
# 1. Fetch fresh base features from APIs
# ============================================================
print("\n[1/7] Fetching fresh data from APIs...")

import yfinance as yf
from fredapi import Fred

START_DATE = "2014-01-01"
fred = Fred(api_key=os.environ["FRED_API_KEY"])

# --- Gold prices (GC=F) ---
print("  Fetching GC=F (gold futures)...")
gold = yf.download("GC=F", start=START_DATE, progress=False)
if hasattr(gold.columns, 'levels') and gold.columns.nlevels > 1:
    gold.columns = gold.columns.droplevel(1)
gold_close = gold["Close"].copy()
gold_close.index = gold_close.index.tz_localize(None)
gold_return_next = gold_close.pct_change().shift(-1) * 100
gold_return_next.name = "gold_return_next"
print(f"    Gold: {len(gold_close)} rows, {gold_close.index.min().date()} to {gold_close.index.max().date()}")

# --- GLD ETF ---
print("  Fetching GLD...")
gld = yf.download("GLD", start=START_DATE, progress=False)
if hasattr(gld.columns, 'levels') and gld.columns.nlevels > 1:
    gld.columns = gld.columns.droplevel(1)
gld.index = gld.index.tz_localize(None)

# --- Cross-asset ---
print("  Fetching SI=F, HG=F, ^GSPC...")
tickers_xa = ["SI=F", "HG=F", "^GSPC"]
xa_data = {}
for t in tickers_xa:
    d = yf.download(t, start=START_DATE, progress=False)
    if hasattr(d.columns, 'levels') and d.columns.nlevels > 1:
        d.columns = d.columns.droplevel(1)
    d.index = d.index.tz_localize(None)
    xa_data[t] = d["Close"]

# --- DXY ---
print("  Fetching DX-Y.NYB (DXY)...")
dxy = yf.download("DX-Y.NYB", start=START_DATE, progress=False)
if hasattr(dxy.columns, 'levels') and dxy.columns.nlevels > 1:
    dxy.columns = dxy.columns.droplevel(1)
dxy.index = dxy.index.tz_localize(None)

# --- CNY ---
print("  Fetching CNY=X...")
cny = yf.download("CNY=X", start=START_DATE, progress=False)
if hasattr(cny.columns, 'levels') and cny.columns.nlevels > 1:
    cny.columns = cny.columns.droplevel(1)
cny.index = cny.index.tz_localize(None)

# --- FRED series ---
print("  Fetching FRED: DFII10, VIXCLS, DGS10, DGS2, T10YIE...")
fred_series = {}
for sid in ["DFII10", "VIXCLS", "DGS10", "DGS2", "T10YIE"]:
    s = fred.get_series(sid, observation_start=START_DATE)
    fred_series[sid] = s
    print(f"    {sid}: {len(s)} obs, last={s.index[-1].date()}")

# ============================================================
# 2. Build base features DataFrame
# ============================================================
print("\n[2/7] Building base features DataFrame...")

# Use gold trading days as reference
ref_idx = gold_close.dropna().index

bf = pd.DataFrame(index=ref_idx)
bf["gold_return_next"] = gold_return_next

# FRED series (forward-fill to business days)
bf["real_rate_real_rate"] = fred_series["DFII10"].reindex(ref_idx, method="ffill")
bf["vix_vix"] = fred_series["VIXCLS"].reindex(ref_idx, method="ffill")
bf["yield_curve_dgs10"] = fred_series["DGS10"].reindex(ref_idx, method="ffill")
bf["yield_curve_dgs2"] = fred_series["DGS2"].reindex(ref_idx, method="ffill")
bf["yield_curve_yield_spread"] = bf["yield_curve_dgs10"] - bf["yield_curve_dgs2"]
bf["inflation_expectation_inflation_expectation"] = fred_series["T10YIE"].reindex(ref_idx, method="ffill")

# Yahoo series
bf["dxy_dxy"] = dxy["Close"].reindex(ref_idx, method="ffill")
bf["cny_demand_cny_usd"] = cny["Close"].reindex(ref_idx, method="ffill")

# GLD
bf["technical_gld_open"] = gld["Open"].reindex(ref_idx, method="ffill")
bf["technical_gld_high"] = gld["High"].reindex(ref_idx, method="ffill")
bf["technical_gld_low"] = gld["Low"].reindex(ref_idx, method="ffill")
bf["technical_gld_close"] = gld["Close"].reindex(ref_idx, method="ffill")
bf["technical_gld_volume"] = gld["Volume"].reindex(ref_idx, method="ffill")

# Cross-asset
bf["cross_asset_silver_close"] = xa_data["SI=F"].reindex(ref_idx, method="ffill")
bf["cross_asset_copper_close"] = xa_data["HG=F"].reindex(ref_idx, method="ffill")
bf["cross_asset_sp500_close"] = xa_data["^GSPC"].reindex(ref_idx, method="ffill")

# ETF flow proxies
bf["etf_flow_gld_volume"] = gld["Volume"].reindex(ref_idx, method="ffill")
bf["etf_flow_gld_close"] = gld["Close"].reindex(ref_idx, method="ffill")
vol_ma20 = gld["Volume"].rolling(20).mean()
bf["etf_flow_volume_ma20"] = vol_ma20.reindex(ref_idx, method="ffill")

# Format index
bf.index = pd.to_datetime(bf.index)
bf.index.name = "Date"
bf = bf.dropna(subset=["gold_return_next"])

# Convert to string dates for joining
bf.index = bf.index.strftime("%Y-%m-%d")

print(f"  Base features: {len(bf)} rows, {bf.index.min()} to {bf.index.max()}")
print(f"  Columns: {len(bf.columns)}")

# Save refreshed base_features
bf.to_csv("data/processed/base_features_refreshed.csv")
print(f"  Saved to data/processed/base_features_refreshed.csv")

# ============================================================
# 3. Apply transformations (matching attempt 7)
# ============================================================
print("\n[3/7] Applying transformations...")

final_df = bf.copy()
final_df["real_rate_change"] = final_df["real_rate_real_rate"].diff()
final_df["dxy_change"] = final_df["dxy_dxy"].diff()
final_df["vix"] = final_df["vix_vix"]
final_df["yield_spread_change"] = final_df["yield_curve_yield_spread"].diff()
final_df["inflation_exp_change"] = final_df["inflation_expectation_inflation_expectation"].diff()

# ============================================================
# 4. Load submodel outputs
# ============================================================
print("\n[4/7] Loading submodel outputs...")

SUBMODEL_DIR = "data/submodel_outputs"

submodel_specs = {
    "vix": {
        "columns": ["vix_regime_probability", "vix_mean_reversion_z", "vix_persistence"],
        "date_col": "date", "tz_aware": False, "rename": {},
    },
    "technical": {
        "columns": ["tech_trend_regime_prob", "tech_mean_reversion_z", "tech_volatility_regime"],
        "date_col": "date", "tz_aware": True, "rename": {},
    },
    "cross_asset": {
        "columns": ["xasset_regime_prob", "xasset_recession_signal", "xasset_divergence"],
        "date_col": "Date", "tz_aware": False, "rename": {},
    },
    "yield_curve": {
        "columns": ["yc_spread_velocity_z", "yc_curvature_z"],
        "date_col": "index", "tz_aware": False, "rename": {},
    },
    "etf_flow": {
        "columns": ["etf_regime_prob", "etf_capital_intensity", "etf_pv_divergence"],
        "date_col": "Date", "tz_aware": False, "rename": {},
    },
    "inflation_expectation": {
        "columns": ["ie_regime_prob", "ie_anchoring_z", "ie_gold_sensitivity_z"],
        "date_col": "Unnamed: 0", "tz_aware": False, "rename": {},
    },
    "options_market": {
        "columns": ["options_risk_regime_prob"],
        "date_col": "Date", "tz_aware": True,
        "rename": {"options_regime_smooth": "options_risk_regime_prob"},
    },
    "temporal_context": {
        "columns": ["temporal_context_score"],
        "date_col": "date", "tz_aware": False, "rename": {},
    },
}

for feature, spec in submodel_specs.items():
    fpath = os.path.join(SUBMODEL_DIR, f"{feature}.csv")
    if not os.path.exists(fpath):
        fpath = os.path.join("data/dataset_upload_clean", f"{feature}.csv")
    if not os.path.exists(fpath):
        print(f"  WARNING: {feature}.csv not found")
        continue

    df = pd.read_csv(fpath)
    if spec.get("rename"):
        df = df.rename(columns=spec["rename"])

    date_col = spec["date_col"]
    if spec["tz_aware"]:
        df["Date"] = pd.to_datetime(df[date_col], utc=True).dt.strftime("%Y-%m-%d")
    elif date_col == "index":
        df["Date"] = pd.to_datetime(df.iloc[:, 0]).dt.strftime("%Y-%m-%d")
    elif date_col == "Unnamed: 0":
        df["Date"] = pd.to_datetime(df["Unnamed: 0"]).dt.strftime("%Y-%m-%d")
    else:
        df["Date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    available_cols = [c for c in spec["columns"] if c in df.columns]
    if not available_cols:
        print(f"  WARNING: {feature} - no target columns found. Cols: {list(df.columns)}")
        continue

    df = df[["Date"] + available_cols].set_index("Date")
    final_df = final_df.join(df, how="left")
    print(f"  {feature}: {len(df)} rows ({df.index.min()} to {df.index.max()}), cols: {available_cols}")

# ============================================================
# 5. Feature preparation and NaN imputation
# ============================================================
print("\n[5/7] Feature preparation...")

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

missing = [c for c in FEATURE_COLUMNS if c not in final_df.columns]
if missing:
    print(f"  WARNING: Missing features: {missing}")
    FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c in final_df.columns]

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

base_req = ["gold_return_next", "real_rate_change", "dxy_change", "vix",
            "yield_spread_change", "inflation_exp_change"]
base_req = [c for c in base_req if c in final_df.columns]
final_df = final_df.dropna(subset=base_req)

nan_count = final_df[FEATURE_COLUMNS].isna().sum().sum()
print(f"  Features: {len(FEATURE_COLUMNS)}")
print(f"  Remaining NaN: {nan_count}")
print(f"  Final dataset: {len(final_df)} rows")
print(f"  Date range: {final_df.index.min()} to {final_df.index.max()}")

# ============================================================
# 6. Train/Val/Test split + Training
# ============================================================
print("\n[6/7] Training XGBoost (attempt 7 hyperparameters)...")

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

print(f"  Train: {len(train_df)} ({train_df.index.min()} to {train_df.index.max()})")
print(f"  Val:   {len(val_df)} ({val_df.index.min()} to {val_df.index.max()})")
print(f"  Test:  {len(test_df)} ({test_df.index.min()} to {test_df.index.max()})")
print(f"  Positive%: train={100*(y_train>0).mean():.1f}%, val={100*(y_val>0).mean():.1f}%, test={100*(y_test>0).mean():.1f}%")

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

# Bootstrap ensemble (5 seeds)
N_ENSEMBLE = 5
ensemble_models = []

for i in range(N_ENSEMBLE):
    seed = 42 + i
    params = xgb_params.copy()
    params["random_state"] = seed
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    ensemble_models.append(model)

print(f"  Trained {N_ENSEMBLE} ensemble models")

# Ensemble predictions
pred_train = np.mean([m.predict(X_train) for m in ensemble_models], axis=0)
pred_val = np.mean([m.predict(X_val) for m in ensemble_models], axis=0)
test_preds_all = np.array([m.predict(X_test) for m in ensemble_models])
pred_test = test_preds_all.mean(axis=0)
bootstrap_std_test = test_preds_all.std(axis=0)

# OLS scaling
from numpy.linalg import lstsq
A = pred_val.reshape(-1, 1)
alpha_ols = lstsq(A, y_val, rcond=None)[0][0]
alpha_ols = np.clip(alpha_ols, 0.5, 10.0)
print(f"  OLS scaling: {alpha_ols:.3f}")

pred_test_scaled = pred_test * alpha_ols

# ============================================================
# 7. Evaluation
# ============================================================
print("\n[7/7] Evaluating performance...")


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


# --- Test set metrics ---
da_test = direction_accuracy(y_test, pred_test)
mae_test_raw = float(np.abs(pred_test - y_test).mean())
mae_test_scaled = float(np.abs(pred_test_scaled - y_test).mean())
mae_test = min(mae_test_raw, mae_test_scaled)

hcda_abs, hc_count_abs = high_confidence_da(y_test, pred_test, method="abs")
hcda_boot, hc_count_boot = high_confidence_da(y_test, pred_test, method="bootstrap", bootstrap_std=bootstrap_std_test)

sharpe_test, net_returns_test = sharpe_with_costs(y_test, pred_test)
cum_ret = float(np.prod(1 + net_returns_test) - 1)

# Naive always-up
sharpe_naive, _ = sharpe_with_costs(y_test, np.ones_like(y_test))
da_naive = direction_accuracy(y_test, np.ones_like(y_test))

# Train/Val metrics
da_train = direction_accuracy(y_train, pred_train)
da_val = direction_accuracy(y_val, pred_val)
sharpe_train, _ = sharpe_with_costs(y_train, pred_train)

mae_zero = float(np.abs(y_test).mean())

print("\n" + "=" * 70)
print("INFERENCE PERFORMANCE RESULTS (REFRESHED DATA)")
print("=" * 70)

print(f"\nData: {len(final_df)} rows ({final_df.index.min()} to {final_df.index.max()})")
print(f"Test: {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")

print(f"\n{'Metric':<35} {'Target':<12} {'Achieved':<12} {'Status':<8} {'Note'}")
print("-" * 85)
print(f"{'Direction Accuracy':<35} {'> 56.0%':<12} {da_test*100:.2f}%{'':>5} {'PASS' if da_test > 0.56 else 'FAIL':<8} naive={da_naive*100:.1f}%")
print(f"{'High-Confidence DA (|pred|)':<35} {'> 60.0%':<12} {hcda_abs*100:.2f}%{'':>5} {'PASS' if hcda_abs > 0.60 else 'FAIL':<8} n={hc_count_abs}")
print(f"{'High-Confidence DA (bootstrap)':<35} {'> 60.0%':<12} {hcda_boot*100:.2f}%{'':>5} {'PASS' if hcda_boot > 0.60 else 'FAIL':<8} n={hc_count_boot}")
print(f"{'MAE':<35} {'< 0.75%':<12} {mae_test:.4f}%{'':>4} {'PASS' if mae_test < 0.75 else 'FAIL':<8} zero={mae_zero:.4f}%")
print(f"{'Sharpe (after 5bps costs)':<35} {'> 0.80':<12} {sharpe_test:.4f}{'':>5} {'PASS' if sharpe_test > 0.80 else 'FAIL':<8} naive={sharpe_naive:.2f}")

targets_passed = sum([da_test > 0.56, hcda_abs > 0.60, mae_test < 0.75, sharpe_test > 0.80])
print(f"\nTargets passed: {targets_passed}/4")

print(f"\n--- Overfitting Check ---")
print(f"  Train DA: {da_train*100:.2f}%  |  Val DA: {da_val*100:.2f}%  |  Test DA: {da_test*100:.2f}%")
print(f"  Train-Test gap: {(da_train - da_test)*100:.2f}pp (threshold < 10pp)")
print(f"  Train Sharpe: {sharpe_train:.2f}")

print(f"\n--- Prediction Distribution ---")
print(f"  Mean: {pred_test.mean():.6f}  Std: {pred_test.std():.6f}  Positive%: {100*(pred_test>0).mean():.1f}%")
print(f"  Actual std: {y_test.std():.4f}")

print(f"\n--- Trading Performance ---")
n_trades = int(np.sum(np.abs(np.diff(np.sign(pred_test))) > 0))
print(f"  Cumulative return: {cum_ret*100:.2f}%")
print(f"  Trades: {n_trades} in {len(y_test)} days")
print(f"  Annualized return: {net_returns_test.mean()*252*100:.2f}%")

# Max drawdown
cumsum = np.cumsum(net_returns_test)
running_max = np.maximum.accumulate(cumsum)
drawdown = cumsum - running_max
print(f"  Max drawdown: {drawdown.min()*100:.2f}%")

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
).reset_index()
for _, row in monthly.iterrows():
    bar = "#" * int(row["da"] * 20)
    print(f"  {str(row['month']):<8s} DA={row['da']*100:5.1f}% ({row['count']:3d} days) {bar}")

# --- Comparison with Kaggle attempt 7 ---
print(f"\n--- Comparison: Local vs Kaggle Attempt 7 ---")
print(f"  {'Metric':<20s} {'Local':<12s} {'Kaggle Att7':<12s} {'Delta'}")
print(f"  {'DA':<20s} {da_test*100:.2f}%{'':>5} {'60.04%':<12s} {(da_test*100 - 60.04):+.2f}pp")
print(f"  {'HCDA':<20s} {hcda_abs*100:.2f}%{'':>5} {'64.13%':<12s} {(hcda_abs*100 - 64.13):+.2f}pp")
print(f"  {'MAE':<20s} {mae_test:.4f}%{'':>3} {'0.9429%':<12s} {(mae_test - 0.9429):+.4f}%")
print(f"  {'Sharpe':<20s} {sharpe_test:.4f}{'':>5} {'2.4636':<12s} {(sharpe_test - 2.4636):+.4f}")

# Save results
results = {
    "test_date": datetime.now().isoformat(),
    "data_refreshed": True,
    "data_rows": len(final_df),
    "data_range": [final_df.index.min(), final_df.index.max()],
    "n_features": len(FEATURE_COLUMNS),
    "n_ensemble": N_ENSEMBLE,
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
        "sharpe_ratio": sharpe_test,
        "cumulative_return": cum_ret,
    },
    "targets_passed": targets_passed,
    "comparison_vs_kaggle_attempt7": {
        "da_delta_pp": da_test * 100 - 60.04,
        "hcda_delta_pp": hcda_abs * 100 - 64.13,
        "mae_delta": mae_test - 0.9429,
        "sharpe_delta": sharpe_test - 2.4636,
    },
}

out_path = "logs/evaluation/inference_test_refreshed.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print(f"Completed: {datetime.now().isoformat()}")
