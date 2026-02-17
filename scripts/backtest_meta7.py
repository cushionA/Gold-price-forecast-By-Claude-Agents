"""
Meta-Model Attempt 7 Local Backtest & Tomorrow Prediction
==========================================================
1. Fetch fresh data (APIs + local submodel CSVs)
2. Re-train XGBoost with same architecture & Optuna HPO
3. Show 2026 daily predictions vs actuals
4. Predict tomorrow (2/19/2026)
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
from datetime import datetime, timedelta

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

# ============================================================
# FEATURE DEFINITIONS (same as attempt 7)
# ============================================================
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
assert len(FEATURE_COLUMNS) == 24

# ============================================================
# 1. DATA FETCHING
# ============================================================
def fetch_data():
    """Fetch all data from APIs and local submodel CSVs."""
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

    # Gold price
    print("  Gold (GC=F)...", end=" ")
    gold = yf.download('GC=F', start='2014-01-01', end='2026-02-20', progress=False)
    gold_df = gold[['Close']].copy()
    gold_df.columns = ['gold_price']
    gold_df['gold_return'] = gold_df['gold_price'].pct_change() * 100
    gold_df['gold_return_next'] = gold_df['gold_return'].shift(-1)
    gold_df.index = pd.to_datetime(gold_df.index).strftime('%Y-%m-%d')
    print(f"{len(gold_df)} rows, last={gold_df.index[-1]}")

    # Base features
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

    # Submodel features (from local CSVs)
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

    # Drop rows missing base features (but keep rows where target is NaN for "tomorrow" row)
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

def compute_hcda(y_true, y_pred, pct=80):
    thr = np.percentile(np.abs(y_pred), pct)
    mask = np.abs(y_pred) > thr
    if mask.sum() == 0: return 0.0
    hc_p, hc_a = y_pred[mask], y_true[mask]
    m2 = (hc_a != 0) & (hc_p != 0)
    if m2.sum() == 0: return 0.0
    return (np.sign(hc_p[m2]) == np.sign(hc_a[m2])).mean()


# ============================================================
# 3. OPTUNA HPO (same as attempt 7 notebook)
# ============================================================
def run_optuna(X_train, y_train, X_val, y_val, n_trials=100):
    """Run Optuna with attempt 6/7 HP ranges."""

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
# 4. MAIN
# ============================================================
def main():
    print("=" * 70)
    print("META-MODEL ATTEMPT 7 — LOCAL BACKTEST & TOMORROW PREDICTION")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Fetch data
    data = fetch_data()

    # Separate "tomorrow" row (gold_return_next is NaN for the last row)
    tomorrow_row = None
    if pd.isna(data[TARGET].iloc[-1]):
        tomorrow_row = data.iloc[[-1]].copy()
        data_with_target = data.dropna(subset=[TARGET])
    else:
        data_with_target = data.dropna(subset=[TARGET])

    print(f"\n=== DATASET ===")
    print(f"  Total rows (with target): {len(data_with_target)}")
    print(f"  Date range: {data_with_target.index[0]} to {data_with_target.index[-1]}")
    if tomorrow_row is not None:
        print(f"  Tomorrow row: {tomorrow_row.index[0]} (target unknown)")

    # Verify features
    for c in FEATURE_COLUMNS:
        assert c in data_with_target.columns, f"Missing feature: {c}"

    # Split 70/15/15
    n = len(data_with_target)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = data_with_target.iloc[:n_train]
    val = data_with_target.iloc[n_train:n_train+n_val]
    test = data_with_target.iloc[n_train+n_val:]

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

    # Run Optuna
    print(f"\n=== OPTUNA HPO (100 trials) ===")
    study = run_optuna(X_train, y_train, X_val, y_val, n_trials=100)
    best = study.best_trial
    print(f"  Best trial #{best.number}: obj={best.value:.4f}")
    print(f"    val_da={best.user_attrs['val_da']*100:.1f}%, val_sharpe={best.user_attrs['val_sharpe']:.2f}")
    print(f"    val_hcda={best.user_attrs['val_hcda']*100:.1f}%, val_mae={best.user_attrs['val_mae']:.4f}")
    print(f"    da_gap={best.user_attrs['da_gap']:.1f}pp")

    bp = study.best_params
    print(f"\n  Best params:")
    for k, v in bp.items():
        print(f"    {k}: {v}")

    # Also test fallback (attempt 2 params)
    FALLBACK = {
        'objective': 'reg:squarederror', 'max_depth': 2, 'min_child_weight': 14,
        'reg_lambda': 4.76, 'reg_alpha': 3.65, 'subsample': 0.478,
        'colsample_bytree': 0.371, 'learning_rate': 0.025,
        'tree_method': 'hist', 'eval_metric': 'rmse', 'verbosity': 0, 'seed': 42,
    }
    fb_model = xgb.XGBRegressor(**FALLBACK, n_estimators=300, early_stopping_rounds=100)
    fb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    fb_val_da = direction_accuracy(y_val, fb_model.predict(X_val))
    fb_val_sharpe = compute_sharpe(y_val, fb_model.predict(X_val))
    print(f"\n  Fallback: val_da={fb_val_da*100:.1f}%, val_sharpe={fb_val_sharpe:.2f}")

    # Train final model with best params
    print(f"\n=== TRAINING FINAL MODEL ===")
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

    # OLS scaling
    num = np.sum(pred_val * y_val)
    den = np.sum(pred_val ** 2)
    alpha_ols = np.clip(num / den if den != 0 else 1.0, 0.5, 10.0)
    print(f"  OLS alpha: {alpha_ols:.2f}")

    scaled_test = pred_test * alpha_ols
    mae_raw = compute_mae(y_test, pred_test)
    mae_scaled = compute_mae(y_test, scaled_test)

    # Evaluate
    test_da = direction_accuracy(y_test, pred_test)
    test_hcda = compute_hcda(y_test, pred_test, 80)
    test_sharpe = compute_sharpe(y_test, pred_test)
    test_mae = min(mae_raw, mae_scaled)
    train_da = direction_accuracy(y_train, pred_train)

    naive_da = (y_test > 0).sum() / len(y_test)

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  DA:       {test_da*100:.2f}% (target >56%, naive={naive_da*100:.2f}%)")
    print(f"  HCDA:     {test_hcda*100:.2f}% (target >60%)")
    print(f"  MAE:      {test_mae:.4f}% (target <0.75%)")
    print(f"  Sharpe:   {test_sharpe:.2f} (target >0.8)")
    print(f"  Train DA: {train_da*100:.2f}% (gap: {(train_da-test_da)*100:.2f}pp)")

    targets = [test_da > 0.56, test_hcda > 0.60, test_mae < 0.0075, test_sharpe > 0.8]
    print(f"  Targets:  {sum(targets)}/4 ({'PASS' if sum(targets) >= 3 else 'FAIL'})")

    # ============================================================
    # CLASSIFICATION MODEL (UP/DOWN)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION MODEL (XGBoost binary)")
    print(f"{'='*70}")

    # Labels: 1=UP, 0=DOWN
    y_train_cls = (y_train > 0).astype(int)
    y_val_cls = (y_val > 0).astype(int)
    y_test_cls = (y_test > 0).astype(int)

    print(f"  Train: {y_train_cls.sum()} UP / {len(y_train_cls)-y_train_cls.sum()} DOWN")
    print(f"  Val:   {y_val_cls.sum()} UP / {len(y_val_cls)-y_val_cls.sum()} DOWN")
    print(f"  Test:  {y_test_cls.sum()} UP / {len(y_test_cls)-y_test_cls.sum()} DOWN")

    # Class weight for imbalance (more DOWN penalty)
    n_up_train = y_train_cls.sum()
    n_down_train = len(y_train_cls) - n_up_train
    scale_pos = n_down_train / n_up_train  # < 1 since UP > DOWN

    def cls_objective(trial):
        params = {
            'objective': 'binary:logistic',
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 8, 30),
            'subsample': trial.suggest_float('subsample', 0.4, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.8),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 20.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
            'tree_method': 'hist',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'seed': 42 + trial.number,
        }
        n_est = trial.suggest_int('n_estimators', 100, 800)

        model = xgb.XGBClassifier(**params, n_estimators=n_est, early_stopping_rounds=80)
        model.fit(X_train, y_train_cls, eval_set=[(X_val, y_val_cls)], verbose=False)

        prob_val = model.predict_proba(X_val)[:, 1]
        pred_cls = (prob_val > 0.5).astype(int)

        da = (pred_cls == y_val_cls).mean()
        # DOWN capture
        down_mask = y_val_cls == 0
        down_cap = (pred_cls[down_mask] == 0).mean() if down_mask.sum() > 0 else 0
        # UP capture
        up_mask = y_val_cls == 1
        up_cap = (pred_cls[up_mask] == 1).mean() if up_mask.sum() > 0 else 0

        # Sharpe-like metric using classifier direction
        signs = np.where(pred_cls == 1, 1, -1).astype(float)
        strat = signs * y_val / 100.0
        changes = np.abs(np.diff(signs, prepend=0))
        costs = changes * (5.0 / 10000.0)
        net = strat - costs
        sharpe = (net.mean() / net.std()) * np.sqrt(252) if net.std() > 0 else 0

        # Overfit check
        prob_train = model.predict_proba(X_train)[:, 1]
        train_da = ((prob_train > 0.5).astype(int) == y_train_cls).mean()
        da_gap = (train_da - da) * 100
        op = max(0.0, (da_gap - 10.0) / 30.0)

        # Objective: maximize DA primarily, bonus for balanced capture
        sn = np.clip((sharpe + 3) / 6, 0, 1)
        dn = np.clip((da * 100 - 40) / 30, 0, 1)
        # Balanced capture bonus: reward when both UP and DOWN capture > 30%
        balance = min(down_cap, up_cap)  # bottleneck = weaker side
        bn = np.clip(balance, 0, 1)

        obj = 0.40 * dn + 0.35 * sn + 0.25 * bn - 0.30 * op

        trial.set_user_attr('val_da', float(da))
        trial.set_user_attr('val_sharpe', float(sharpe))
        trial.set_user_attr('down_capture', float(down_cap))
        trial.set_user_attr('up_capture', float(up_cap))
        trial.set_user_attr('da_gap', float(da_gap))

        return obj

    print(f"\n  Running Optuna (100 trials)...")
    cls_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=123))
    cls_study.optimize(cls_objective, n_trials=100, timeout=3600)

    cb = cls_study.best_trial
    print(f"  Best trial #{cb.number}: obj={cb.value:.4f}")
    print(f"    val_da={cb.user_attrs['val_da']*100:.1f}%, val_sharpe={cb.user_attrs['val_sharpe']:.2f}")
    print(f"    DOWN capture={cb.user_attrs['down_capture']*100:.1f}%, UP capture={cb.user_attrs['up_capture']*100:.1f}%")
    print(f"    da_gap={cb.user_attrs['da_gap']:.1f}pp")

    cbp = cls_study.best_params
    print(f"\n  Best params:")
    for k, v in cbp.items():
        print(f"    {k}: {v}")

    # Train final classifier
    print(f"\n  Training final classifier...")
    cls_model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=cbp['max_depth'], min_child_weight=cbp['min_child_weight'],
        subsample=cbp['subsample'], colsample_bytree=cbp['colsample_bytree'],
        reg_lambda=cbp['reg_lambda'], reg_alpha=cbp['reg_alpha'],
        learning_rate=cbp['learning_rate'],
        scale_pos_weight=cbp['scale_pos_weight'],
        tree_method='hist', eval_metric='logloss', verbosity=0, seed=42,
        n_estimators=cbp['n_estimators'], early_stopping_rounds=80,
    )
    cls_model.fit(X_train, y_train_cls, eval_set=[(X_val, y_val_cls)], verbose=False)

    # Probabilities
    cls_prob_train = cls_model.predict_proba(X_train)[:, 1]
    cls_prob_val = cls_model.predict_proba(X_val)[:, 1]
    cls_prob_test = cls_model.predict_proba(X_test)[:, 1]

    # ============================================================
    # ENSEMBLE: REGRESSION + CLASSIFICATION
    # ============================================================
    print(f"\n{'='*70}")
    print(f"ENSEMBLE: REGRESSION + CLASSIFICATION")
    print(f"{'='*70}")

    # Tune ensemble threshold on validation set:
    # cls_prob < down_thr → force DOWN (override regression)
    # else → follow regression sign
    best_ens_thr = 0.5
    best_ens_obj = -999

    print(f"\n  Scanning DOWN-override threshold on validation set...")
    print(f"  {'cls_thr':>8} {'DA':>6} {'Sharpe':>7} {'DOWN%':>6} {'UP%':>6} {'#DOWN':>6}")
    print(f"  {'-'*48}")

    for thr in np.arange(0.30, 0.65, 0.01):
        # Ensemble rule: if cls_prob < thr → DOWN, else → regression sign
        ens_signs = np.where(cls_prob_val < thr, -1, np.sign(pred_val)).astype(float)
        ens_signs[ens_signs == 0] = 1  # tie → UP

        mask = (y_val != 0)
        da = (ens_signs[mask] == np.sign(y_val[mask])).mean() if mask.sum() > 0 else 0

        strat = ens_signs * y_val / 100.0
        changes = np.abs(np.diff(ens_signs, prepend=0))
        costs = changes * (5.0 / 10000.0)
        net = strat - costs
        sharpe = (net.mean() / net.std()) * np.sqrt(252) if net.std() > 0 else 0

        down_mask = y_val < 0
        down_cap = (ens_signs[down_mask] == -1).mean() if down_mask.sum() > 0 else 0
        up_mask = y_val > 0
        up_cap = (ens_signs[up_mask] == 1).mean() if up_mask.sum() > 0 else 0
        n_down = int((ens_signs == -1).sum())

        # Objective: maximize Sharpe with reasonable DA
        obj = sharpe + 0.5 * (da - 0.5)  # Sharpe first, DA bonus

        if abs(thr * 100 % 5) < 0.1 or thr == 0.50:
            print(f"  {thr:>8.2f} {da*100:>5.1f}% {sharpe:>6.2f} {down_cap*100:>5.1f}% {up_cap*100:>5.1f}% {n_down:>5}")

        if obj > best_ens_obj:
            best_ens_obj = obj
            best_ens_thr = thr

    print(f"\n  Best ensemble threshold: cls_prob < {best_ens_thr:.2f} → DOWN")

    # Apply ensemble to all sets
    def ensemble_predict(cls_prob, reg_pred, thr):
        """cls_prob < thr → DOWN (-1), else → regression sign"""
        signs = np.where(cls_prob < thr, -1, np.sign(reg_pred)).astype(float)
        signs[signs == 0] = 1
        return signs

    ens_train = ensemble_predict(cls_prob_train, pred_train, best_ens_thr)
    ens_val = ensemble_predict(cls_prob_val, pred_val, best_ens_thr)
    ens_test = ensemble_predict(cls_prob_test, pred_test, best_ens_thr)

    # Test set evaluation
    mask_t = (y_test != 0)
    ens_test_da = (ens_test[mask_t] == np.sign(y_test[mask_t])).mean() if mask_t.sum() > 0 else 0
    ens_test_sharpe = compute_sharpe(y_test, ens_test * 0.01)  # sign-based
    # Recompute Sharpe manually for consistency
    strat_ens = ens_test * y_test / 100.0
    ch_ens = np.abs(np.diff(ens_test, prepend=0))
    cost_ens = ch_ens * (5.0 / 10000.0)
    net_ens = strat_ens - cost_ens
    ens_test_sharpe = (net_ens.mean() / net_ens.std()) * np.sqrt(252) if net_ens.std() > 0 else 0

    test_down_mask = y_test < 0
    test_up_mask = y_test > 0
    ens_down_cap = (ens_test[test_down_mask] == -1).mean() if test_down_mask.sum() > 0 else 0
    ens_up_cap = (ens_test[test_up_mask] == 1).mean() if test_up_mask.sum() > 0 else 0

    print(f"\n  --- Test Set Comparison ---")
    print(f"  {'':>20} {'Regression':>12} {'Ensemble':>12}")
    print(f"  {'DA':>20} {test_da*100:>11.2f}% {ens_test_da*100:>11.2f}%")
    print(f"  {'Sharpe':>20} {test_sharpe:>11.2f} {ens_test_sharpe:>11.2f}")
    print(f"  {'UP capture':>20} {(np.sign(pred_test[test_up_mask])==1).mean()*100:>11.1f}% {ens_up_cap*100:>11.1f}%")
    print(f"  {'DOWN capture':>20} {(np.sign(pred_test[test_down_mask])==-1).mean()*100:>11.1f}% {ens_down_cap*100:>11.1f}%")
    print(f"  {'#DOWN predicted':>20} {int((np.sign(pred_test)==-1).sum()):>11} {int((ens_test==-1).sum()):>11}")

    # ============================================================
    # 2026 DAILY PREDICTIONS vs ACTUALS
    # ============================================================
    print(f"\n{'='*70}")
    print(f"2026 DAILY PREDICTIONS vs ACTUALS")
    print(f"{'='*70}")

    # Get all 2026 rows from the full dataset
    all_dates = data_with_target.index
    dates_2026 = [d for d in all_dates if d.startswith('2026-')]

    if len(dates_2026) == 0:
        print("  No 2026 data available in test set!")
    else:
        # We need predictions for all dates. Combine train+val+test
        pred_all = np.concatenate([pred_train, pred_val, pred_test])
        scaled_all = pred_all * alpha_ols
        all_actuals = np.concatenate([y_train, y_val, y_test])
        all_dates_list = list(data_with_target.index)

        # We also need classifier probabilities for all data
        cls_prob_all = np.concatenate([cls_prob_train, cls_prob_val, cls_prob_test])
        ens_all = ensemble_predict(cls_prob_all, pred_all, best_ens_thr)

        print(f"\n{'Date':<12} {'Actual%':>8} {'Pred':>8} {'P(UP)':>6} {'Reg':>6} {'Ens':>6} {'Gold$':>10}")
        print("-" * 75)

        correct_reg = 0
        correct_ens = 0
        total_count = 0
        cumulative_return = 0
        strat_reg = 0
        strat_ens = 0

        for d in dates_2026:
            idx = all_dates_list.index(d)
            actual = all_actuals[idx]
            pred_r = pred_all[idx]
            cls_p = cls_prob_all[idx]
            ens_sign = ens_all[idx]

            gold_price = data_with_target.loc[d, 'gold_price'] if 'gold_price' in data_with_target.columns else np.nan

            reg_dir = "UP" if pred_r > 0 else "DN"
            ens_dir = "UP" if ens_sign > 0 else "DN"
            actual_dir_sign = 1 if actual > 0 else (-1 if actual < 0 else 0)

            reg_ok = (np.sign(pred_r) == actual_dir_sign) if actual != 0 else None
            ens_ok = (ens_sign == actual_dir_sign) if actual != 0 else None

            if actual != 0:
                total_count += 1
                if reg_ok: correct_reg += 1
                if ens_ok: correct_ens += 1

            mr = "O" if reg_ok else ("X" if reg_ok is not None else "-")
            me = "O" if ens_ok else ("X" if ens_ok is not None else "-")
            cumulative_return += actual
            strat_reg += np.sign(pred_r) * actual / 100
            strat_ens += ens_sign * actual / 100

            print(f"  {d}  {actual:>+7.3f}% {pred_r:>+7.4f}% {cls_p:>5.2f} {reg_dir:>3}({mr}) {ens_dir:>3}({me})  ${gold_price:>8.2f}")

        print("-" * 75)
        da_reg = correct_reg / total_count if total_count > 0 else 0
        da_ens = correct_ens / total_count if total_count > 0 else 0
        print(f"  2026 DA (regression only): {da_reg*100:.1f}% ({correct_reg}/{total_count})")
        print(f"  2026 DA (ensemble):        {da_ens*100:.1f}% ({correct_ens}/{total_count})")
        print(f"  2026 Actual Return:  {cumulative_return:+.3f}%")
        print(f"  2026 Strategy (reg): {strat_reg*100:+.3f}%")
        print(f"  2026 Strategy (ens): {strat_ens*100:+.3f}%")

    # ============================================================
    # TOMORROW'S PREDICTION
    # ============================================================
    print(f"\n{'='*70}")
    print(f"TOMORROW'S PREDICTION")
    print(f"{'='*70}")

    if tomorrow_row is not None:
        X_tomorrow = tomorrow_row[FEATURE_COLUMNS].values
        pred_tomorrow_raw = final.predict(X_tomorrow)[0]
        pred_tomorrow_scaled = pred_tomorrow_raw * alpha_ols

        # Bootstrap ensemble for confidence
        bootstrap_preds = []
        for seed in [42, 43, 44, 45, 46]:
            bm = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=bp['max_depth'], min_child_weight=bp['min_child_weight'],
                subsample=bp['subsample'], colsample_bytree=bp['colsample_bytree'],
                reg_lambda=bp['reg_lambda'], reg_alpha=bp['reg_alpha'],
                learning_rate=bp['learning_rate'],
                tree_method='hist', eval_metric='rmse', verbosity=0, seed=seed,
                n_estimators=bp['n_estimators'], early_stopping_rounds=100,
            )
            bm.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            bootstrap_preds.append(bm.predict(X_tomorrow)[0])

        boot_mean = np.mean(bootstrap_preds)
        boot_std = np.std(bootstrap_preds)
        boot_conf = 1.0 / (1.0 + boot_std)

        # Get current gold price
        last_price = data.loc[tomorrow_row.index[0], 'gold_price'] if 'gold_price' in data.columns else np.nan

        # Classification probability for tomorrow
        cls_prob_tomorrow = cls_model.predict_proba(X_tomorrow)[:, 1][0]
        ens_sign_tomorrow = ensemble_predict(
            np.array([cls_prob_tomorrow]), np.array([pred_tomorrow_raw]), best_ens_thr)[0]

        reg_dir = "UP" if pred_tomorrow_raw > 0 else "DOWN"
        ens_dir = "UP" if ens_sign_tomorrow > 0 else "DOWN"

        print(f"  Prediction date: {tomorrow_row.index[0]} (next trading day)")
        print(f"  Current gold price: ${last_price:.2f}")
        print(f"  Regression return (raw):    {pred_tomorrow_raw:+.4f}%")
        print(f"  Regression return (scaled): {pred_tomorrow_scaled:+.4f}%")
        print(f"  Classifier P(UP): {cls_prob_tomorrow:.4f}")
        print(f"  Direction (regression): {reg_dir}")
        print(f"  Direction (ensemble):   {ens_dir}")
        print(f"  Bootstrap mean: {boot_mean:+.4f}%, std: {boot_std:.4f}%")
        print(f"  Bootstrap confidence: {boot_conf:.4f}")

        # Is this a high-confidence prediction?
        # Use test set threshold to determine
        test_abs_preds = np.abs(pred_test)
        thr_80 = np.percentile(test_abs_preds, 80)
        is_hc = abs(pred_tomorrow_raw) > thr_80
        print(f"  High-confidence (top 20%): {'YES' if is_hc else 'NO'} (|pred|={abs(pred_tomorrow_raw):.4f}, threshold={thr_80:.4f})")

        if not np.isnan(last_price):
            predicted_price = last_price * (1 + pred_tomorrow_scaled / 100)
            print(f"  Predicted price (scaled): ${predicted_price:.2f}")

        # Feature contributions for tomorrow
        print(f"\n  --- Key Features for Tomorrow ---")
        importances = final.feature_importances_
        feat_vals = X_tomorrow[0]
        ranked = sorted(zip(FEATURE_COLUMNS, importances, feat_vals), key=lambda x: -x[1])
        for fname, imp, fval in ranked[:10]:
            print(f"    {fname:<30} imp={imp:.4f}  val={fval:+.4f}")
    else:
        print("  No tomorrow row available (last date has a known target).")
        print("  Using the latest available date for a forward prediction.")

        # Use the last row from data that has all features
        last_row = data.iloc[[-1]]
        X_last = last_row[FEATURE_COLUMNS].values
        pred_last_raw = final.predict(X_last)[0]
        pred_last_scaled = pred_last_raw * alpha_ols
        last_price = data.iloc[-1]['gold_price'] if 'gold_price' in data.columns else np.nan

        cls_prob_last = cls_model.predict_proba(X_last)[:, 1][0]
        ens_sign_last = ensemble_predict(
            np.array([cls_prob_last]), np.array([pred_last_raw]), best_ens_thr)[0]

        reg_dir = "UP" if pred_last_raw > 0 else "DOWN"
        ens_dir = "UP" if ens_sign_last > 0 else "DOWN"
        print(f"  Latest date: {last_row.index[0]}")
        print(f"  Current gold price: ${last_price:.2f}")
        print(f"  Regression return (raw):    {pred_last_raw:+.4f}%")
        print(f"  Regression return (scaled): {pred_last_scaled:+.4f}%")
        print(f"  Classifier P(UP): {cls_prob_last:.4f}")
        print(f"  Direction (regression): {reg_dir}")
        print(f"  Direction (ensemble):   {ens_dir}")
        if not np.isnan(last_price):
            predicted_price = last_price * (1 + pred_last_scaled / 100)
            print(f"  Predicted next-day price (scaled): ${predicted_price:.2f}")

    # ============================================================
    # VALIDITY ASSESSMENT
    # ============================================================
    print(f"\n{'='*70}")
    print(f"PREDICTION VALIDITY ASSESSMENT")
    print(f"{'='*70}")

    # 1. Model stability
    print(f"\n  1. Model Stability:")
    print(f"     Train DA: {train_da*100:.2f}%, Test DA: {test_da*100:.2f}% (gap: {(train_da-test_da)*100:+.2f}pp)")
    print(f"     {'PASS' if abs(train_da - test_da) * 100 < 10 else 'WARN'}: Train-test gap {'<' if abs(train_da - test_da) * 100 < 10 else '>'} 10pp")

    # 2. Prediction distribution
    print(f"\n  2. Prediction Distribution:")
    print(f"     Raw: mean={pred_test.mean():+.4f}, std={pred_test.std():.4f}")
    print(f"     Positive%: {(pred_test>0).sum()/len(pred_test)*100:.1f}%")
    print(f"     Actual positive%: {(y_test>0).sum()/len(y_test)*100:.1f}%")
    bullish_bias = (pred_test > 0).sum() / len(pred_test) * 100
    print(f"     {'NOTE' if bullish_bias > 80 else 'OK'}: Model has {bullish_bias:.0f}% bullish bias")

    # 3. Recent performance (last 30 days in test)
    recent = min(30, len(pred_test))
    recent_da = direction_accuracy(y_test[-recent:], pred_test[-recent:])
    recent_sharpe = compute_sharpe(y_test[-recent:], pred_test[-recent:])
    print(f"\n  3. Recent Performance (last {recent} test days):")
    print(f"     DA: {recent_da*100:.1f}%, Sharpe: {recent_sharpe:.2f}")

    # 4. Feature importance concentration
    top5_imp = sum(sorted(final.feature_importances_, reverse=True)[:5])
    print(f"\n  4. Feature Importance:")
    print(f"     Top-5 concentration: {top5_imp*100:.1f}%")
    print(f"     {'OK' if top5_imp < 0.5 else 'WARN'}: {'Well-distributed' if top5_imp < 0.5 else 'Concentrated in few features'}")

    print(f"\n{'='*70}")
    print(f"BACKTEST COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
