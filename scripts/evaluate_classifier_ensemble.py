"""
Classifier Attempt 2 - Ensemble Evaluation Script
==================================================
1. Re-run regression meta-model (attempt 7 architecture) with Optuna HPO
2. Load external classifier P(DOWN) from classifier_2/classifier.csv
3. Scan ensemble thresholds on validation set
4. Evaluate best threshold on test set and 2026 YTD
5. Compare regression-only vs ensemble
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

# Fix Windows cp932 encoding
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMODEL_DIR = os.path.join(PROJECT_ROOT, "data", "submodel_outputs")
CLASSIFIER_CSV = os.path.join(PROJECT_ROOT, "notebooks", "classifier_2", "classifier.csv")

# Same 24 features as attempt 7
FEATURE_COLUMNS = [
    'real_rate_change', 'dxy_change', 'vix', 'yield_spread_change', 'inflation_exp_change',
    'vix_regime_probability', 'vix_mean_reversion_z', 'vix_persistence',
    'tech_trend_regime_prob', 'tech_mean_reversion_z', 'tech_volatility_regime',
    'xasset_regime_prob', 'xasset_recession_signal', 'xasset_divergence',
    'yc_spread_velocity_z', 'yc_curvature_z',
    'etf_regime_prob', 'etf_capital_intensity', 'etf_pv_divergence',
    'ie_regime_prob', 'ie_anchoring_z', 'ie_gold_sensitivity_z',
    'options_risk_regime_prob',
    'temporal_context_score',
]
TARGET = 'gold_return_next'


# ============================================================
# 1. DATA FETCHING (same as backtest_meta7.py)
# ============================================================
def fetch_data():
    import yfinance as yf
    from fredapi import Fred
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    FRED_API_KEY = os.environ.get('FRED_API_KEY')
    if not FRED_API_KEY:
        print("[ERROR] FRED_API_KEY not found in .env")
        sys.exit(1)

    fred = Fred(api_key=FRED_API_KEY)
    print("=== FETCHING DATA ===")

    print("  Gold (GC=F)...", end=" ")
    gold = yf.download('GC=F', start='2014-01-01', end='2026-02-20', progress=False)
    gold_df = gold[['Close']].copy()
    gold_df.columns = ['gold_price']
    gold_df['gold_return'] = gold_df['gold_price'].pct_change() * 100
    gold_df['gold_return_next'] = gold_df['gold_return'].shift(-1)
    gold_df.index = pd.to_datetime(gold_df.index).strftime('%Y-%m-%d')
    print(f"{len(gold_df)} rows, last={gold_df.index[-1]}")

    print("  Real Rate (DFII10)...", end=" ")
    rr = fred.get_series('DFII10', observation_start='2014-01-01')
    rr_df = rr.to_frame('real_rate_real_rate')
    rr_df.index = pd.to_datetime(rr_df.index).strftime('%Y-%m-%d')
    print(f"{len(rr_df)} rows")

    print("  DXY (DX-Y.NYB)...", end=" ")
    dxy = yf.download('DX-Y.NYB', start='2014-01-01', end='2026-02-20', progress=False)
    dxy_df = dxy[['Close']].copy()
    dxy_df.columns = ['dxy_dxy']
    dxy_df.index = pd.to_datetime(dxy_df.index).strftime('%Y-%m-%d')
    print(f"{len(dxy_df)} rows")

    print("  VIX (VIXCLS)...", end=" ")
    vix = fred.get_series('VIXCLS', observation_start='2014-01-01')
    vix_df = vix.to_frame('vix_vix')
    vix_df.index = pd.to_datetime(vix_df.index).strftime('%Y-%m-%d')
    print(f"{len(vix_df)} rows")

    print("  Yield Curve...", end=" ")
    dgs10 = fred.get_series('DGS10', observation_start='2014-01-01')
    dgs2 = fred.get_series('DGS2', observation_start='2014-01-01')
    yc_df = pd.DataFrame({'DGS10': dgs10, 'DGS2': dgs2})
    yc_df['yield_curve_yield_spread'] = yc_df['DGS10'] - yc_df['DGS2']
    yc_df = yc_df[['yield_curve_yield_spread']]
    yc_df.index = pd.to_datetime(yc_df.index).strftime('%Y-%m-%d')
    print(f"{len(yc_df)} rows")

    print("  Inflation Expectation (T10YIE)...", end=" ")
    ie = fred.get_series('T10YIE', observation_start='2014-01-01')
    ie_df = ie.to_frame('inflation_expectation_inflation_expectation')
    ie_df.index = pd.to_datetime(ie_df.index).strftime('%Y-%m-%d')
    print(f"{len(ie_df)} rows")

    # Merge base features
    base = gold_df[['gold_return_next', 'gold_price', 'gold_return']].copy()
    for df in [rr_df, dxy_df, vix_df, yc_df, ie_df]:
        base = base.join(df, how='left')
    base = base.ffill()

    # Submodel features
    print("\n  Loading submodel outputs...")
    submodel_specs = {
        'vix': {'cols': ['vix_regime_probability', 'vix_mean_reversion_z', 'vix_persistence'], 'date_col': 'date', 'tz': False},
        'technical': {'cols': ['tech_trend_regime_prob', 'tech_mean_reversion_z', 'tech_volatility_regime'], 'date_col': 'date', 'tz': True},
        'cross_asset': {'cols': ['xasset_regime_prob', 'xasset_recession_signal', 'xasset_divergence'], 'date_col': 'Date', 'tz': False},
        'yield_curve': {'cols': ['yc_spread_velocity_z', 'yc_curvature_z'], 'date_col': 'index', 'tz': False},
        'etf_flow': {'cols': ['etf_regime_prob', 'etf_capital_intensity', 'etf_pv_divergence'], 'date_col': 'Date', 'tz': False},
        'inflation_expectation': {'cols': ['ie_regime_prob', 'ie_anchoring_z', 'ie_gold_sensitivity_z'], 'date_col': 'Unnamed: 0', 'tz': False},
        'options_market': {'cols': ['options_risk_regime_prob'], 'date_col': 'Date', 'tz': True},
        'temporal_context': {'cols': ['temporal_context_score'], 'date_col': 'date', 'tz': False},
    }

    for name, spec in submodel_specs.items():
        path = os.path.join(SUBMODEL_DIR, f"{name}.csv")
        df = pd.read_csv(path)
        dc = spec['date_col']
        if spec['tz']:
            df['Date'] = pd.to_datetime(df[dc], utc=True).dt.strftime('%Y-%m-%d')
        elif dc == 'index':
            df['Date'] = pd.to_datetime(df.iloc[:, 0]).dt.strftime('%Y-%m-%d')
        elif dc == 'Unnamed: 0':
            df['Date'] = pd.to_datetime(df['Unnamed: 0']).dt.strftime('%Y-%m-%d')
        else:
            df['Date'] = pd.to_datetime(df[dc]).dt.strftime('%Y-%m-%d')
        df = df[['Date'] + spec['cols']].set_index('Date')
        base = base.join(df, how='left')
        print(f"    {name}: {len(df)} rows, last={df.index[-1]}")

    # Transformations
    base['real_rate_change'] = base['real_rate_real_rate'].diff()
    base['dxy_change'] = base['dxy_dxy'].diff()
    base['vix'] = base['vix_vix']
    base['yield_spread_change'] = base['yield_curve_yield_spread'].diff()
    base['inflation_exp_change'] = base['inflation_expectation_inflation_expectation'].diff()
    base.drop(columns=['real_rate_real_rate', 'dxy_dxy', 'vix_vix',
                        'yield_curve_yield_spread', 'inflation_expectation_inflation_expectation'], inplace=True)

    # NaN imputation
    regime_cols = ['vix_regime_probability', 'tech_trend_regime_prob', 'xasset_regime_prob',
                   'etf_regime_prob', 'ie_regime_prob', 'options_risk_regime_prob', 'temporal_context_score']
    for c in regime_cols:
        if c in base.columns:
            base[c] = base[c].fillna(0.5)

    z_cols = ['vix_mean_reversion_z', 'tech_mean_reversion_z', 'yc_spread_velocity_z', 'yc_curvature_z',
              'etf_capital_intensity', 'etf_pv_divergence', 'ie_anchoring_z', 'ie_gold_sensitivity_z']
    for c in z_cols:
        if c in base.columns:
            base[c] = base[c].fillna(0.0)

    div_cols = ['xasset_recession_signal', 'xasset_divergence']
    for c in div_cols:
        if c in base.columns:
            base[c] = base[c].fillna(0.0)

    cont_cols = ['tech_volatility_regime', 'vix_persistence']
    for c in cont_cols:
        if c in base.columns:
            base[c] = base[c].fillna(base[c].median())

    base_req = ['real_rate_change', 'dxy_change', 'vix', 'yield_spread_change', 'inflation_exp_change']
    base = base.dropna(subset=base_req)

    return base


# ============================================================
# 2. METRIC FUNCTIONS
# ============================================================
def direction_accuracy(y_true, y_pred):
    mask = (y_true != 0) & (y_pred != 0)
    if mask.sum() == 0: return 0.0
    return (np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean()


def compute_mae(y_true, y_pred):
    return np.abs(y_pred - y_true).mean()


def compute_sharpe(y_true, y_pred, cost_bps=5.0):
    positions = np.sign(y_pred)
    strategy_returns = positions * y_true / 100.0
    changes = np.abs(np.diff(positions, prepend=0))
    costs = changes * (cost_bps / 10000.0)
    net = strategy_returns - costs
    if len(net) < 2 or net.std() == 0: return 0.0
    return (net.mean() / net.std()) * np.sqrt(252)


def compute_sharpe_from_signs(y_true, signs, cost_bps=5.0):
    """Compute Sharpe from directional signs (+1/-1) and actual returns."""
    strategy_returns = signs * y_true / 100.0
    changes = np.abs(np.diff(signs, prepend=0))
    costs = changes * (cost_bps / 10000.0)
    net = strategy_returns - costs
    if len(net) < 2 or net.std() == 0: return 0.0
    return (net.mean() / net.std()) * np.sqrt(252)


def compute_hcda(y_true, y_pred, pct=80):
    thr = np.percentile(np.abs(y_pred), pct)
    mask = np.abs(y_pred) > thr
    if mask.sum() == 0: return 0.0
    hc_p, hc_a = y_pred[mask], y_true[mask]
    m2 = (hc_a != 0) & (hc_p != 0)
    if m2.sum() == 0: return 0.0
    return (np.sign(hc_p[m2]) == np.sign(hc_a[m2])).mean()


# ============================================================
# 3. OPTUNA HPO (same as attempt 7)
# ============================================================
def run_optuna(X_train, y_train, X_val, y_val, n_trials=100):
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'min_child_weight': trial.suggest_int('min_child_weight', 12, 25),
            'subsample': trial.suggest_float('subsample', 0.4, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.7),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 10.0, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
            'tree_method': 'hist',
            'eval_metric': 'rmse',
            'verbosity': 0,
            'seed': 42 + trial.number,
        }
        n_est = trial.suggest_int('n_estimators', 100, 800)

        model = xgb.XGBRegressor(**params, n_estimators=n_est, early_stopping_rounds=100)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        tp = model.predict(X_train)
        vp = model.predict(X_val)

        t_da = direction_accuracy(y_train, tp)
        v_da = direction_accuracy(y_val, vp)
        v_mae = compute_mae(y_val, vp)
        v_sharpe = compute_sharpe(y_val, vp)
        v_hcda = compute_hcda(y_val, vp, 80)

        da_gap = (t_da - v_da) * 100
        op = max(0.0, (da_gap - 10.0) / 30.0)

        sn = np.clip((v_sharpe + 3) / 6, 0, 1)
        dn = np.clip((v_da * 100 - 40) / 30, 0, 1)
        mn = np.clip((1.0 - v_mae) / 0.5, 0, 1)
        hn = np.clip((v_hcda * 100 - 40) / 30, 0, 1)

        obj = 0.40 * sn + 0.30 * dn + 0.10 * mn + 0.20 * hn - 0.30 * op

        trial.set_user_attr('val_da', float(v_da))
        trial.set_user_attr('val_sharpe', float(v_sharpe))
        trial.set_user_attr('val_hcda', float(v_hcda))
        trial.set_user_attr('val_mae', float(v_mae))
        trial.set_user_attr('da_gap', float(da_gap))

        return obj

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=3600)
    return study


# ============================================================
# 4. ENSEMBLE FUNCTIONS
# ============================================================
def ensemble_predict_pdown(p_down, reg_pred, threshold):
    """
    If p_down > threshold -> override to DOWN (-1)
    Else -> follow regression sign
    """
    signs = np.where(p_down > threshold, -1, np.sign(reg_pred)).astype(float)
    signs[signs == 0] = 1  # tie -> UP
    return signs


def evaluate_ensemble(y_true, reg_pred, p_down, threshold):
    """Evaluate ensemble at a given P(DOWN) threshold."""
    ens_signs = ensemble_predict_pdown(p_down, reg_pred, threshold)
    reg_signs = np.sign(reg_pred).astype(float)
    reg_signs[reg_signs == 0] = 1

    # DA
    mask = y_true != 0
    ens_da = (ens_signs[mask] == np.sign(y_true[mask])).mean() if mask.sum() > 0 else 0
    reg_da = (reg_signs[mask] == np.sign(y_true[mask])).mean() if mask.sum() > 0 else 0

    # Sharpe
    ens_sharpe = compute_sharpe_from_signs(y_true, ens_signs)
    reg_sharpe = compute_sharpe_from_signs(y_true, reg_signs)

    # DOWN capture
    down_mask = y_true < 0
    up_mask = y_true > 0
    ens_down_cap = (ens_signs[down_mask] == -1).mean() if down_mask.sum() > 0 else 0
    reg_down_cap = (reg_signs[down_mask] == -1).mean() if down_mask.sum() > 0 else 0
    ens_up_cap = (ens_signs[up_mask] == 1).mean() if up_mask.sum() > 0 else 0
    reg_up_cap = (reg_signs[up_mask] == 1).mean() if up_mask.sum() > 0 else 0

    n_down_pred = int((ens_signs == -1).sum())

    return {
        'ens_da': ens_da, 'reg_da': reg_da,
        'ens_sharpe': ens_sharpe, 'reg_sharpe': reg_sharpe,
        'ens_down_cap': ens_down_cap, 'reg_down_cap': reg_down_cap,
        'ens_up_cap': ens_up_cap, 'reg_up_cap': reg_up_cap,
        'n_down_pred': n_down_pred,
    }


# ============================================================
# 5. MAIN
# ============================================================
def main():
    print("=" * 70)
    print("CLASSIFIER ATTEMPT 2 - ENSEMBLE EVALUATION")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # --- Load classifier predictions ---
    print("\n=== LOADING CLASSIFIER ===")
    cls_df = pd.read_csv(CLASSIFIER_CSV)
    cls_df['Date'] = pd.to_datetime(cls_df['Date']).dt.strftime('%Y-%m-%d')
    cls_df = cls_df.set_index('Date')
    print(f"  Classifier rows: {len(cls_df)}")
    print(f"  Date range: {cls_df.index[0]} to {cls_df.index[-1]}")
    print(f"  P(DOWN) mean={cls_df['p_down'].mean():.3f}, std={cls_df['p_down'].std():.3f}")
    print(f"  P(DOWN) range: [{cls_df['p_down'].min():.3f}, {cls_df['p_down'].max():.3f}]")

    # --- Fetch regression data ---
    data = fetch_data()

    # Separate tomorrow row
    tomorrow_row = None
    if pd.isna(data[TARGET].iloc[-1]):
        tomorrow_row = data.iloc[[-1]].copy()
        data_with_target = data.dropna(subset=[TARGET])
    else:
        data_with_target = data.dropna(subset=[TARGET])

    print(f"\n=== DATASET ===")
    print(f"  Total rows (with target): {len(data_with_target)}")
    print(f"  Date range: {data_with_target.index[0]} to {data_with_target.index[-1]}")

    # Verify features
    for c in FEATURE_COLUMNS:
        assert c in data_with_target.columns, f"Missing feature: {c}"

    # Split 70/15/15
    n = len(data_with_target)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = data_with_target.iloc[:n_train]
    val = data_with_target.iloc[n_train:n_train + n_val]
    test = data_with_target.iloc[n_train + n_val:]

    X_train = train[FEATURE_COLUMNS].values
    y_train = train[TARGET].values
    X_val = val[FEATURE_COLUMNS].values
    y_val = val[TARGET].values
    X_test = test[FEATURE_COLUMNS].values
    y_test = test[TARGET].values

    print(f"\n=== SPLIT ===")
    print(f"  Train: {len(train)} ({train.index[0]} to {train.index[-1]})")
    print(f"  Val:   {len(val)} ({val.index[0]} to {val.index[-1]})")
    print(f"  Test:  {len(test)} ({test.index[0]} to {test.index[-1]})")

    # --- Run Optuna for regression model ---
    print(f"\n=== OPTUNA HPO (100 trials) ===")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=100)
    best = study.best_trial
    print(f"  Best trial #{best.number}: obj={best.value:.4f}")
    print(f"    val_da={best.user_attrs['val_da'] * 100:.1f}%, val_sharpe={best.user_attrs['val_sharpe']:.2f}")
    print(f"    val_hcda={best.user_attrs['val_hcda'] * 100:.1f}%, val_mae={best.user_attrs['val_mae']:.4f}")

    bp = study.best_params
    print(f"\n  Best params:")
    for k, v in bp.items():
        print(f"    {k}: {v}")

    # Train final regression model
    print(f"\n=== TRAINING FINAL REGRESSION MODEL ===")
    final = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=bp['max_depth'], min_child_weight=bp['min_child_weight'],
        subsample=bp['subsample'], colsample_bytree=bp['colsample_bytree'],
        reg_lambda=bp['reg_lambda'], reg_alpha=bp['reg_alpha'],
        learning_rate=bp['learning_rate'],
        tree_method='hist', eval_metric='rmse', verbosity=0, seed=42,
        n_estimators=bp['n_estimators'], early_stopping_rounds=100,
    )
    final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred_train = final.predict(X_train)
    pred_val = final.predict(X_val)
    pred_test = final.predict(X_test)

    # Regression-only metrics
    reg_test_da = direction_accuracy(y_test, pred_test)
    reg_test_hcda = compute_hcda(y_test, pred_test, 80)
    reg_test_sharpe = compute_sharpe(y_test, pred_test)
    reg_test_mae = compute_mae(y_test, pred_test)
    reg_train_da = direction_accuracy(y_train, pred_train)

    print(f"\n=== REGRESSION-ONLY METRICS ===")
    print(f"  Test DA:     {reg_test_da * 100:.2f}%")
    print(f"  Test HCDA:   {reg_test_hcda * 100:.2f}%")
    print(f"  Test Sharpe: {reg_test_sharpe:.2f}")
    print(f"  Test MAE:    {reg_test_mae:.4f}%")
    print(f"  Train DA:    {reg_train_da * 100:.2f}%")
    print(f"  Positive prediction %: {(pred_test > 0).sum() / len(pred_test) * 100:.1f}%")

    # --- Align classifier with regression data ---
    print(f"\n=== ALIGNING CLASSIFIER WITH REGRESSION DATA ===")

    # Get P(DOWN) for val and test dates
    val_dates = list(val.index)
    test_dates = list(test.index)
    train_dates = list(train.index)

    # Check overlap
    cls_dates = set(cls_df.index)
    val_overlap = [d for d in val_dates if d in cls_dates]
    test_overlap = [d for d in test_dates if d in cls_dates]
    train_overlap = [d for d in train_dates if d in cls_dates]
    print(f"  Train dates in classifier: {len(train_overlap)}/{len(train_dates)}")
    print(f"  Val dates in classifier:   {len(val_overlap)}/{len(val_dates)}")
    print(f"  Test dates in classifier:  {len(test_overlap)}/{len(test_dates)}")

    # Get aligned p_down
    p_down_val = np.array([cls_df.loc[d, 'p_down'] if d in cls_dates else 0.5 for d in val_dates])
    p_down_test = np.array([cls_df.loc[d, 'p_down'] if d in cls_dates else 0.5 for d in test_dates])
    p_down_train = np.array([cls_df.loc[d, 'p_down'] if d in cls_dates else 0.5 for d in train_dates])

    missing_val = sum(1 for d in val_dates if d not in cls_dates)
    missing_test = sum(1 for d in test_dates if d not in cls_dates)
    print(f"  Missing val dates (default 0.5): {missing_val}")
    print(f"  Missing test dates (default 0.5): {missing_test}")

    # P(DOWN) distribution in each split
    print(f"\n  P(DOWN) distribution:")
    print(f"    Val:  mean={p_down_val.mean():.3f}, std={p_down_val.std():.3f}")
    print(f"    Test: mean={p_down_test.mean():.3f}, std={p_down_test.std():.3f}")

    # ============================================================
    # ENSEMBLE THRESHOLD SCANNING ON VALIDATION SET
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"ENSEMBLE THRESHOLD SCANNING (Validation Set)")
    print(f"{'=' * 70}")
    print(f"\n  Logic: if P(DOWN) > threshold -> override to DOWN, else follow regression")
    print(f"  Note: classifier outputs P(DOWN), so HIGHER threshold = MORE conservative (fewer overrides)")
    print(f"\n  {'thr':>6} {'DA':>7} {'Sharpe':>8} {'DOWN%':>7} {'UP%':>7} {'#DOWN':>6} {'obj':>8}")
    print(f"  {'-' * 55}")

    best_thr = 0.50
    best_obj = -999
    threshold_results = []

    for thr in np.arange(0.40, 0.71, 0.01):
        res = evaluate_ensemble(y_val, pred_val, p_down_val, thr)
        # Objective: maximize Sharpe with reasonable DA
        obj = res['ens_sharpe'] + 0.5 * (res['ens_da'] - 0.5)

        threshold_results.append({
            'threshold': round(thr, 2),
            'val_da': res['ens_da'],
            'val_sharpe': res['ens_sharpe'],
            'val_down_cap': res['ens_down_cap'],
            'val_up_cap': res['ens_up_cap'],
            'n_down_pred': res['n_down_pred'],
            'objective': obj,
        })

        if abs(thr * 100 % 5) < 0.5 or abs(thr - 0.50) < 0.005 or abs(thr - 0.55) < 0.005:
            print(f"  {thr:>6.2f} {res['ens_da'] * 100:>6.1f}% {res['ens_sharpe']:>7.2f} "
                  f"{res['ens_down_cap'] * 100:>6.1f}% {res['ens_up_cap'] * 100:>6.1f}% "
                  f"{res['n_down_pred']:>5} {obj:>7.3f}")

        if obj > best_obj:
            best_obj = obj
            best_thr = round(thr, 2)

    print(f"\n  Regression-only on val: DA={direction_accuracy(y_val, pred_val) * 100:.1f}%, "
          f"Sharpe={compute_sharpe(y_val, pred_val):.2f}")
    print(f"\n  Best threshold: P(DOWN) > {best_thr:.2f} -> override to DOWN")

    # ============================================================
    # TEST SET EVALUATION
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"TEST SET EVALUATION (threshold={best_thr:.2f})")
    print(f"{'=' * 70}")

    test_res = evaluate_ensemble(y_test, pred_test, p_down_test, best_thr)

    print(f"\n  {'Metric':<25} {'Regression':>12} {'Ensemble':>12} {'Delta':>10}")
    print(f"  {'-' * 65}")
    print(f"  {'DA':<25} {test_res['reg_da'] * 100:>11.2f}% {test_res['ens_da'] * 100:>11.2f}% "
          f"{(test_res['ens_da'] - test_res['reg_da']) * 100:>+9.2f}pp")
    print(f"  {'Sharpe':<25} {test_res['reg_sharpe']:>11.2f} {test_res['ens_sharpe']:>11.2f} "
          f"{test_res['ens_sharpe'] - test_res['reg_sharpe']:>+9.2f}")
    print(f"  {'DOWN capture':<25} {test_res['reg_down_cap'] * 100:>11.1f}% {test_res['ens_down_cap'] * 100:>11.1f}% "
          f"{(test_res['ens_down_cap'] - test_res['reg_down_cap']) * 100:>+9.1f}pp")
    print(f"  {'UP capture':<25} {test_res['reg_up_cap'] * 100:>11.1f}% {test_res['ens_up_cap'] * 100:>11.1f}% "
          f"{(test_res['ens_up_cap'] - test_res['reg_up_cap']) * 100:>+9.1f}pp")
    print(f"  {'#DOWN predicted':<25} {int((np.sign(pred_test) == -1).sum()):>11} {test_res['n_down_pred']:>11}")

    # Also evaluate at a few alternative thresholds on test
    print(f"\n  --- Alternative thresholds on TEST ---")
    print(f"  {'thr':>6} {'DA':>7} {'Sharpe':>8} {'DOWN%':>7} {'UP%':>7} {'#DOWN':>6}")
    print(f"  {'-' * 50}")
    for alt_thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        alt_res = evaluate_ensemble(y_test, pred_test, p_down_test, alt_thr)
        marker = " <-- best_val" if abs(alt_thr - best_thr) < 0.005 else ""
        print(f"  {alt_thr:>6.2f} {alt_res['ens_da'] * 100:>6.1f}% {alt_res['ens_sharpe']:>7.2f} "
              f"{alt_res['ens_down_cap'] * 100:>6.1f}% {alt_res['ens_up_cap'] * 100:>6.1f}% "
              f"{alt_res['n_down_pred']:>5}{marker}")

    # Regression-only baseline on test (using sign-based Sharpe for fair comparison)
    reg_signs_test = np.sign(pred_test).astype(float)
    reg_signs_test[reg_signs_test == 0] = 1
    reg_sharpe_sign = compute_sharpe_from_signs(y_test, reg_signs_test)
    print(f"\n  Regression-only (no override): DA={test_res['reg_da'] * 100:.2f}%, Sharpe={reg_sharpe_sign:.2f}")

    # ============================================================
    # 2026 YTD ANALYSIS
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"2026 YTD DAY-BY-DAY ANALYSIS")
    print(f"{'=' * 70}")

    # Combine all predictions
    pred_all = np.concatenate([pred_train, pred_val, pred_test])
    p_down_all = np.concatenate([p_down_train, p_down_val, p_down_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    all_dates_list = list(data_with_target.index)

    dates_2026 = [d for d in all_dates_list if d.startswith('2026-')]

    if len(dates_2026) == 0:
        print("  No 2026 data in dataset!")
    else:
        print(f"\n  {'Date':<12} {'Actual%':>8} {'RegPred':>8} {'P(DOWN)':>8} {'Reg':>5} {'Ens':>5} {'Actual':>7} {'Gold$':>10}")
        print(f"  {'-' * 80}")

        correct_reg_2026 = 0
        correct_ens_2026 = 0
        total_2026 = 0
        reg_down_correct_2026 = 0
        ens_down_correct_2026 = 0
        actual_down_2026 = 0
        reg_up_correct_2026 = 0
        ens_up_correct_2026 = 0
        actual_up_2026 = 0

        strat_reg_2026 = 0.0
        strat_ens_2026 = 0.0

        for d in dates_2026:
            idx = all_dates_list.index(d)
            actual = y_all[idx]
            pred_r = pred_all[idx]
            p_d = p_down_all[idx]

            # Ensemble sign
            ens_sign = -1 if p_d > best_thr else (1 if pred_r > 0 else -1)
            if pred_r == 0 and p_d <= best_thr:
                ens_sign = 1

            reg_sign = 1 if pred_r > 0 else (-1 if pred_r < 0 else 1)
            actual_sign = 1 if actual > 0 else (-1 if actual < 0 else 0)

            gold_price = data_with_target.loc[d, 'gold_price'] if 'gold_price' in data_with_target.columns else np.nan

            reg_dir = "UP" if reg_sign > 0 else "DN"
            ens_dir = "UP" if ens_sign > 0 else "DN"
            actual_dir = "UP" if actual > 0 else ("DN" if actual < 0 else "--")

            if actual != 0:
                total_2026 += 1
                reg_ok = (reg_sign == actual_sign)
                ens_ok = (ens_sign == actual_sign)
                if reg_ok: correct_reg_2026 += 1
                if ens_ok: correct_ens_2026 += 1

                if actual < 0:
                    actual_down_2026 += 1
                    if reg_sign == -1: reg_down_correct_2026 += 1
                    if ens_sign == -1: ens_down_correct_2026 += 1
                elif actual > 0:
                    actual_up_2026 += 1
                    if reg_sign == 1: reg_up_correct_2026 += 1
                    if ens_sign == 1: ens_up_correct_2026 += 1
            else:
                reg_ok = None
                ens_ok = None

            mr = "O" if reg_ok else ("X" if reg_ok is not None else "-")
            me = "O" if ens_ok else ("X" if ens_ok is not None else "-")

            strat_reg_2026 += reg_sign * actual / 100
            strat_ens_2026 += ens_sign * actual / 100

            changed = " ***" if reg_dir != ens_dir else ""
            print(f"  {d}  {actual:>+7.3f}% {pred_r:>+7.4f}% {p_d:>7.3f} "
                  f"{reg_dir}({mr}) {ens_dir}({me}) {actual_dir:>5}  ${gold_price:>8.2f}{changed}")

        print(f"  {'-' * 80}")
        da_reg_2026 = correct_reg_2026 / total_2026 if total_2026 > 0 else 0
        da_ens_2026 = correct_ens_2026 / total_2026 if total_2026 > 0 else 0

        print(f"\n  2026 YTD Summary:")
        print(f"  {'Metric':<30} {'Regression':>12} {'Ensemble':>12} {'Delta':>10}")
        print(f"  {'-' * 70}")
        print(f"  {'DA':<30} {da_reg_2026 * 100:>11.1f}% {da_ens_2026 * 100:>11.1f}% "
              f"{(da_ens_2026 - da_reg_2026) * 100:>+9.1f}pp")
        print(f"  {'DOWN captured':<30} {reg_down_correct_2026:>11}/{actual_down_2026} "
              f"{ens_down_correct_2026:>11}/{actual_down_2026} "
              f"{ens_down_correct_2026 - reg_down_correct_2026:>+9}")
        print(f"  {'UP captured':<30} {reg_up_correct_2026:>11}/{actual_up_2026} "
              f"{ens_up_correct_2026:>11}/{actual_up_2026} "
              f"{ens_up_correct_2026 - reg_up_correct_2026:>+9}")
        print(f"  {'Strategy return':<30} {strat_reg_2026 * 100:>+11.3f}% {strat_ens_2026 * 100:>+11.3f}%")

        if actual_down_2026 > 0:
            reg_down_rate = reg_down_correct_2026 / actual_down_2026
            ens_down_rate = ens_down_correct_2026 / actual_down_2026
            print(f"  {'DOWN capture rate':<30} {reg_down_rate * 100:>11.1f}% {ens_down_rate * 100:>11.1f}%")
        if actual_up_2026 > 0:
            reg_up_rate = reg_up_correct_2026 / actual_up_2026
            ens_up_rate = ens_up_correct_2026 / actual_up_2026
            print(f"  {'UP capture rate':<30} {reg_up_rate * 100:>11.1f}% {ens_up_rate * 100:>11.1f}%")

    # ============================================================
    # EVALUATION CRITERIA ASSESSMENT
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"EVALUATION CRITERIA ASSESSMENT")
    print(f"{'=' * 70}")

    # Load standalone metrics from training_result.json
    with open(os.path.join(PROJECT_ROOT, "notebooks", "classifier_2", "training_result.json")) as f:
        cls_result = json.load(f)

    standalone = cls_result['standalone_metrics']['test']

    print(f"\n  --- Standalone Classifier (Test Set) ---")
    checks_standalone = {
        'balanced_accuracy > 52%': (standalone['balanced_acc'] * 100, standalone['balanced_acc'] > 0.52),
        'down_recall > 30%': (standalone['down_recall'] * 100, standalone['down_recall'] > 0.30),
        'down_precision > 45%': (standalone['down_precision'] * 100, standalone['down_precision'] > 0.45),
        'f1_down > 0.35': (standalone['f1_down'], standalone['f1_down'] > 0.35),
        'MCC > 0': (standalone['mcc'], standalone['mcc'] > 0),
    }

    for name, (val, passed) in checks_standalone.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {name:<30} {val:>8.3f}  [{status}]")

    standalone_pass = sum(1 for _, (_, p) in checks_standalone.items() if p)
    standalone_total = len(checks_standalone)
    print(f"\n    Standalone: {standalone_pass}/{standalone_total} criteria met")

    # Overfit assessment
    print(f"\n  --- Overfitting Assessment ---")
    print(f"    Train MCC:  {cls_result['standalone_metrics']['train']['mcc']:.3f}")
    print(f"    Val MCC:    {cls_result['standalone_metrics']['val']['mcc']:.3f}")
    print(f"    Test MCC:   {cls_result['standalone_metrics']['test']['mcc']:.3f}")
    print(f"    Train AUC:  {cls_result['standalone_metrics']['train']['roc_auc']:.3f}")
    print(f"    Val AUC:    {cls_result['standalone_metrics']['val']['roc_auc']:.3f}")
    print(f"    Test AUC:   {cls_result['standalone_metrics']['test']['roc_auc']:.3f}")
    overfit_ratio_mcc = abs(cls_result['standalone_metrics']['train']['mcc']) / max(abs(cls_result['standalone_metrics']['test']['mcc']), 0.001)
    print(f"    Overfit ratio (train/test MCC): {overfit_ratio_mcc:.1f}x  [SEVERE]")

    print(f"\n  --- Ensemble (Test Set, threshold={best_thr:.2f}) ---")
    checks_ensemble = {
        'DA maintained (>= regression)': (test_res['ens_da'] * 100, test_res['ens_da'] >= test_res['reg_da'] - 0.005),
        'DA improvement > +1.0pp': ((test_res['ens_da'] - test_res['reg_da']) * 100, (test_res['ens_da'] - test_res['reg_da']) > 0.01),
        'Sharpe > 2.0': (test_res['ens_sharpe'], test_res['ens_sharpe'] > 2.0),
        'DOWN capture > 25%': (test_res['ens_down_cap'] * 100, test_res['ens_down_cap'] > 0.25),
        'UP accuracy > 85%': (test_res['ens_up_cap'] * 100, test_res['ens_up_cap'] > 0.85),
    }

    for name, (val, passed) in checks_ensemble.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {name:<35} {val:>8.2f}  [{status}]")

    ensemble_pass = sum(1 for _, (_, p) in checks_ensemble.items() if p)
    ensemble_total = len(checks_ensemble)
    print(f"\n    Ensemble: {ensemble_pass}/{ensemble_total} criteria met")

    # 2026 YTD criteria
    if len(dates_2026) > 0:
        print(f"\n  --- 2026 YTD ---")
        ytd_down_target = 3
        ytd_check = ens_down_correct_2026 >= ytd_down_target
        print(f"    DOWN captured >= {ytd_down_target}: {ens_down_correct_2026}/{actual_down_2026}  "
              f"[{'PASS' if ytd_check else 'FAIL'}]")

    # ============================================================
    # FINAL DECISION
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"FINAL DECISION")
    print(f"{'=' * 70}")

    # Decision logic
    da_delta = test_res['ens_da'] - test_res['reg_da']
    sharpe_ok = test_res['ens_sharpe'] > 2.0
    da_improved = da_delta > 0.01
    da_maintained = da_delta >= -0.005
    down_improved = test_res['ens_down_cap'] > 0.25

    if da_improved and sharpe_ok and down_improved:
        decision = "PASS"
        reason = "Ensemble improves DA by >1pp AND Sharpe >2.0 AND DOWN capture >25%"
    elif da_maintained and down_improved and sharpe_ok:
        decision = "MARGINAL_PASS"
        reason = "DA maintained, DOWN capture improved significantly, Sharpe >2.0"
    elif da_delta < -0.005 or test_res['ens_sharpe'] < 2.0:
        decision = "FAIL"
        reason = f"Ensemble hurts: DA delta={da_delta * 100:+.2f}pp, Sharpe={test_res['ens_sharpe']:.2f}"
    else:
        decision = "FAIL"
        reason = "Insufficient improvement across criteria"

    print(f"\n  Decision: {decision}")
    print(f"  Reason: {reason}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    eval_dir = os.path.join(PROJECT_ROOT, "logs", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    evaluation_result = {
        "feature": "classifier",
        "attempt": 2,
        "timestamp": datetime.now().isoformat(),
        "decision": decision,
        "reason": reason,

        "standalone_assessment": {
            "balanced_accuracy": standalone['balanced_acc'],
            "balanced_accuracy_passed": standalone['balanced_acc'] > 0.52,
            "down_recall": standalone['down_recall'],
            "down_recall_passed": standalone['down_recall'] > 0.30,
            "down_precision": standalone['down_precision'],
            "down_precision_passed": standalone['down_precision'] > 0.45,
            "f1_down": standalone['f1_down'],
            "f1_down_passed": standalone['f1_down'] > 0.35,
            "mcc": standalone['mcc'],
            "mcc_passed": standalone['mcc'] > 0,
            "p_down_std": cls_result['p_down_distribution']['std'],
            "non_trivial": cls_result['p_down_distribution']['std'] > 0.03,
            "overfitting": {
                "train_mcc": cls_result['standalone_metrics']['train']['mcc'],
                "val_mcc": cls_result['standalone_metrics']['val']['mcc'],
                "test_mcc": cls_result['standalone_metrics']['test']['mcc'],
                "severity": "SEVERE",
                "overfit_ratio": overfit_ratio_mcc,
            },
            "standalone_criteria_met": f"{standalone_pass}/{standalone_total}",
        },

        "regression_model": {
            "test_da": reg_test_da,
            "test_hcda": reg_test_hcda,
            "test_sharpe": reg_test_sharpe,
            "test_mae": reg_test_mae,
            "positive_pct": float((pred_test > 0).sum() / len(pred_test)),
            "optuna_best_params": bp,
        },

        "ensemble_evaluation": {
            "best_threshold": best_thr,
            "threshold_logic": f"if P(DOWN) > {best_thr} then override to DOWN, else follow regression sign",
            "test_set": {
                "regression_da": test_res['reg_da'],
                "ensemble_da": test_res['ens_da'],
                "da_delta_pp": (test_res['ens_da'] - test_res['reg_da']) * 100,
                "regression_sharpe": test_res['reg_sharpe'],
                "ensemble_sharpe": test_res['ens_sharpe'],
                "sharpe_delta": test_res['ens_sharpe'] - test_res['reg_sharpe'],
                "regression_down_capture": test_res['reg_down_cap'],
                "ensemble_down_capture": test_res['ens_down_cap'],
                "regression_up_capture": test_res['reg_up_cap'],
                "ensemble_up_capture": test_res['ens_up_cap'],
                "n_down_predicted_ensemble": test_res['n_down_pred'],
                "n_down_predicted_regression": int((np.sign(pred_test) == -1).sum()),
            },
            "criteria_met": f"{ensemble_pass}/{ensemble_total}",
        },

        "threshold_scan_results": threshold_results,
    }

    # Add 2026 YTD if available
    if len(dates_2026) > 0:
        evaluation_result["ytd_2026"] = {
            "n_days": total_2026,
            "regression_da": da_reg_2026,
            "ensemble_da": da_ens_2026,
            "da_delta_pp": (da_ens_2026 - da_reg_2026) * 100,
            "actual_down_days": actual_down_2026,
            "reg_down_captured": reg_down_correct_2026,
            "ens_down_captured": ens_down_correct_2026,
            "actual_up_days": actual_up_2026,
            "reg_up_captured": reg_up_correct_2026,
            "ens_up_captured": ens_up_correct_2026,
            "reg_strategy_return": strat_reg_2026,
            "ens_strategy_return": strat_ens_2026,
        }

    eval_json_path = os.path.join(eval_dir, "classifier_2_evaluation.json")
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, indent=2, default=str)
    print(f"\n  Saved: {eval_json_path}")

    print(f"\n{'=' * 70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'=' * 70}")

    return evaluation_result


if __name__ == '__main__':
    result = main()
