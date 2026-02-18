"""
Multi-date test for Gold Price Prediction Notebook.
Tests data availability, feature completeness, and prediction quality across dates.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
np.random.seed(42)

from fredapi import Fred
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
import yfinance as yf
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not found in .env")
    sys.exit(1)

fred = Fred(api_key=FRED_API_KEY)
fetch_start = '2014-01-01'
fetch_end = '2026-02-25'

# ============ Helper functions ============
def fit_hmm_best(X, n_components, covariance_type='full', n_restarts=10, n_iter=200, seeds=None):
    if seeds is None:
        seeds = list(range(n_restarts))
    best_model = None
    best_score = -np.inf
    for seed in seeds:
        try:
            model = GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                                n_iter=n_iter, random_state=seed)
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue
    return best_model

def hmm_highest_var_state(model):
    traces = []
    for i in range(model.n_components):
        if model.covariance_type == 'full':
            traces.append(np.trace(model.covars_[i]))
        else:
            traces.append(np.sum(model.covars_[i]))
    return np.argmax(traces)

# ============ Fetch all data ============
print("=" * 60)
print("FETCHING DATA...")
print("=" * 60)

gold_raw = yf.download('GC=F', start=fetch_start, end=fetch_end, progress=False)
gold_close = gold_raw['Close'].squeeze()
gold_close.index = pd.to_datetime(gold_close.index).tz_localize(None)
gold_return = gold_close.pct_change() * 100
gold_return_next = gold_return.shift(-1)
print(f"Gold: {len(gold_close)} rows, {gold_close.index.min()} to {gold_close.index.max()}")

dxy_raw = yf.download('DX-Y.NYB', start=fetch_start, end=fetch_end, progress=False)
dxy_close = dxy_raw['Close'].squeeze()
dxy_close.index = pd.to_datetime(dxy_close.index).tz_localize(None)
print(f"DXY: {len(dxy_close)} rows")

gld_raw = yf.Ticker('GLD').history(start=fetch_start, end=fetch_end, auto_adjust=True)
gld_raw.index = gld_raw.index.tz_localize(None)
print(f"GLD: {len(gld_raw)} rows")

silver_raw = yf.download('SI=F', start=fetch_start, end=fetch_end, progress=False)
copper_raw = yf.download('HG=F', start=fetch_start, end=fetch_end, progress=False)
silver_close = silver_raw['Close'].squeeze()
copper_close = copper_raw['Close'].squeeze()
silver_close.index = pd.to_datetime(silver_close.index).tz_localize(None)
copper_close.index = pd.to_datetime(copper_close.index).tz_localize(None)
print(f"Silver: {len(silver_close)}, Copper: {len(copper_close)}")

real_rate_raw = fred.get_series('DFII10', observation_start=fetch_start)
real_rate_raw.index = pd.to_datetime(real_rate_raw.index)
print(f"DFII10: {len(real_rate_raw)} rows")

vix_raw = fred.get_series('VIXCLS', observation_start=fetch_start)
vix_raw.index = pd.to_datetime(vix_raw.index)
print(f"VIX: {len(vix_raw)} rows")

dgs10_raw = fred.get_series('DGS10', observation_start=fetch_start)
dgs2_raw = fred.get_series('DGS2', observation_start=fetch_start)
dgs10_raw.index = pd.to_datetime(dgs10_raw.index)
dgs2_raw.index = pd.to_datetime(dgs2_raw.index)
yield_spread = dgs10_raw - dgs2_raw
print(f"DGS10: {len(dgs10_raw)}, DGS2: {len(dgs2_raw)}")

ie_raw = fred.get_series('T10YIE', observation_start=fetch_start)
ie_raw.index = pd.to_datetime(ie_raw.index)
print(f"T10YIE: {len(ie_raw)} rows")

skew_data = yf.Ticker('^SKEW').history(start=fetch_start, end=fetch_end, auto_adjust=True)
gvz_data = yf.Ticker('^GVZ').history(start=fetch_start, end=fetch_end, auto_adjust=True)
skew_data.index = skew_data.index.tz_localize(None)
gvz_data.index = gvz_data.index.tz_localize(None)
print(f"SKEW: {len(skew_data)}, GVZ: {len(gvz_data)}")

tnx = yf.Ticker('^TNX').history(start=fetch_start, end=fetch_end)
irx = yf.Ticker('^IRX').history(start=fetch_start, end=fetch_end)
fvx = yf.Ticker('^FVX').history(start=fetch_start, end=fetch_end)
tnx.index = tnx.index.tz_localize(None)
irx.index = irx.index.tz_localize(None)
fvx.index = fvx.index.tz_localize(None)
print(f"TNX: {len(tnx)}, IRX: {len(irx)}, FVX: {len(fvx)}")

# ============ Compute submodels ============
print("\n" + "=" * 60)
print("COMPUTING SUBMODELS...")
print("=" * 60)

# VIX
vix_series = vix_raw.dropna()
vix_log_change = np.log(vix_series).diff().dropna()
gmm_vix = GaussianMixture(n_components=2, covariance_type='diag', n_init=3, max_iter=100, random_state=42)
gmm_vix.fit(vix_log_change.values.reshape(-1, 1))
high_var_idx = np.argmax(gmm_vix.covariances_.flatten())
probs = gmm_vix.predict_proba(vix_log_change.values.reshape(-1, 1))
vix_regime_prob = pd.Series(probs[:, high_var_idx], index=vix_log_change.index)
vix_z = ((vix_series - vix_series.rolling(40).mean()) / vix_series.rolling(40).std()).clip(-4, 4)
vix_persistence = vix_log_change.rolling(30).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
vix_submodel = pd.DataFrame({'vix_regime_probability': vix_regime_prob, 'vix_mean_reversion_z': vix_z, 'vix_persistence': vix_persistence})
print(f"VIX submodel: {len(vix_submodel)} rows")

# Technical
gld_close_s = gld_raw['Close']
gld_returns = gld_close_s.pct_change()
gk_vol = np.sqrt(0.5 * np.log(gld_raw['High'] / gld_raw['Low'])**2 - (2*np.log(2)-1) * np.log(gld_raw['Close'] / gld_raw['Open'])**2).clip(lower=1e-8)
tech_input = pd.DataFrame({'returns': gld_returns, 'gk_vol': gk_vol}).dropna()
hmm_tech = fit_hmm_best(tech_input.values, n_components=2, covariance_type='full', n_restarts=10)
high_var_state = hmm_highest_var_state(hmm_tech)
tech_probs = hmm_tech.predict_proba(tech_input.values)
tech_regime_prob = pd.Series(tech_probs[:, high_var_state], index=tech_input.index)
tech_z = ((gld_returns - gld_returns.rolling(15).mean()) / gld_returns.rolling(15).std()).clip(-4, 4)
gk_vol_z = ((gk_vol - gk_vol.rolling(60).mean()) / gk_vol.rolling(60).std()).clip(-4, 4)
tech_submodel = pd.DataFrame({'tech_trend_regime_prob': tech_regime_prob, 'tech_mean_reversion_z': tech_z, 'tech_volatility_regime': gk_vol_z})
print(f"Technical submodel: {len(tech_submodel)} rows")

# Cross-Asset
gold_ret = gold_close.pct_change()
silver_ret = silver_close.pct_change()
copper_ret = copper_close.pct_change()
xasset_df = pd.DataFrame({'gold_ret': gold_ret, 'silver_ret': silver_ret, 'copper_ret': copper_ret, 'gold_close': gold_close, 'copper_close': copper_close}).dropna()
X_xasset = xasset_df[['gold_ret', 'silver_ret', 'copper_ret']].values
hmm_xasset = fit_hmm_best(X_xasset, n_components=3, covariance_type='full', n_restarts=10)
high_var_xa = hmm_highest_var_state(hmm_xasset)
xa_probs = hmm_xasset.predict_proba(X_xasset)
xasset_regime_prob = pd.Series(xa_probs[:, high_var_xa], index=xasset_df.index)
gc_ratio = xasset_df['gold_close'] / xasset_df['copper_close']
gc_z = (gc_ratio - gc_ratio.rolling(90).mean()) / gc_ratio.rolling(90).std()
xasset_recession = gc_z.diff().clip(-4, 4)
gs_diff = xasset_df['gold_ret'] - silver_ret.reindex(xasset_df.index)
gs_z = ((gs_diff - gs_diff.rolling(20).mean()) / gs_diff.rolling(20).std()).clip(-4, 4)
xasset_submodel = pd.DataFrame({'xasset_regime_prob': xasset_regime_prob, 'xasset_recession_signal': xasset_recession, 'xasset_divergence': gs_z})
print(f"Cross-Asset submodel: {len(xasset_submodel)} rows")

# Yield Curve
dgs10_yf = tnx['Close'] / 100.0
dgs2_yf = irx['Close'] / 100.0 * (365.0 / 91.0)
dgs5_yf = fvx['Close'] / 100.0
yc_df = pd.DataFrame({'dgs10': dgs10_yf, 'dgs2': dgs2_yf, 'dgs5': dgs5_yf}).dropna()
yc_spread = yc_df['dgs10'] - yc_df['dgs2']
yc_curvature = yc_df['dgs5'] - 0.5 * (yc_df['dgs2'] + yc_df['dgs10'])
dgs10_change = yc_df['dgs10'].diff()
dgs2_change = yc_df['dgs2'].diff()
yc_hmm_input = pd.DataFrame({'dgs10_chg': dgs10_change, 'dgs2_chg': dgs2_change}).dropna()
X_yc = yc_hmm_input.values
n_train_yc = int(len(X_yc) * 0.70)
hmm_yc = fit_hmm_best(X_yc[:n_train_yc], n_components=2, covariance_type='diag', n_restarts=5, seeds=[0, 42, 123, 456, 789])
high_var_yc = hmm_highest_var_state(hmm_yc)
yc_probs = hmm_yc.predict_proba(X_yc)
spread_change_5 = yc_spread.diff(5)
yc_spread_vel_z = ((spread_change_5 - spread_change_5.rolling(30).mean()) / spread_change_5.rolling(30).std()).clip(-4, 4)
curvature_change = yc_curvature.diff()
yc_curv_z = ((curvature_change - curvature_change.rolling(120).mean()) / curvature_change.rolling(120).std()).clip(-4, 4)
yc_submodel = pd.DataFrame({'yc_spread_velocity_z': yc_spread_vel_z, 'yc_curvature_z': yc_curv_z})
print(f"Yield Curve submodel: {len(yc_submodel)} rows")

# ETF Flow
gld_vol = gld_raw['Volume']
gld_cl = gld_raw['Close']
dollar_volume = gld_cl * gld_vol
volume_ma20 = gld_vol.rolling(20).mean()
log_volume_ratio = np.log(gld_vol / volume_ma20).replace([np.inf, -np.inf], 0)
gld_ret = gld_cl.pct_change()
etf_input = pd.DataFrame({'lvr': log_volume_ratio, 'gold_ret': gold_ret.reindex(gld_raw.index)}).dropna()
X_etf = etf_input.values
n_train_etf = int(len(X_etf) * 0.70)
hmm_etf = fit_hmm_best(X_etf[:n_train_etf], n_components=4, covariance_type='full', n_restarts=11, seeds=list(range(42, 53)))
train_states = hmm_etf.predict(X_etf[:n_train_etf])
train_gold_ret_by_state = pd.Series(X_etf[:n_train_etf, 1]).groupby(train_states).mean()
accum_state = train_gold_ret_by_state.idxmax()
etf_probs = hmm_etf.predict_proba(X_etf)
etf_regime_prob = pd.Series(etf_probs[:, accum_state], index=etf_input.index)
cap_z = ((dollar_volume - dollar_volume.rolling(60).mean()) / dollar_volume.rolling(60).std()).replace([np.inf, -np.inf], 0)
vol_changes = gld_vol.pct_change()
rolling_corr = gld_ret.rolling(10).corr(vol_changes)
pv_div = ((rolling_corr - rolling_corr.rolling(40).mean()) / rolling_corr.rolling(40).std()).replace([np.inf, -np.inf], 0)
etf_submodel = pd.DataFrame({'etf_regime_prob': etf_regime_prob, 'etf_capital_intensity': cap_z, 'etf_pv_divergence': pv_div})
print(f"ETF Flow submodel: {len(etf_submodel)} rows")

# Inflation Expectation
ie_change = ie_raw.diff().dropna()
ie_vol_5d = ie_change.rolling(5).std()
ie_hmm_input = pd.DataFrame({'ie_chg': ie_change, 'ie_vol': ie_vol_5d}).dropna()
X_ie = ie_hmm_input.values
n_train_ie = int(len(X_ie) * 0.70)
hmm_ie = fit_hmm_best(X_ie[:n_train_ie], n_components=3, covariance_type='full', n_restarts=3, seeds=[42, 43, 44])
high_var_ie = hmm_highest_var_state(hmm_ie)
ie_probs = hmm_ie.predict_proba(X_ie)
ie_regime_prob = pd.Series(ie_probs[:, high_var_ie], index=ie_hmm_input.index)
ie_anchor_z = ((ie_vol_5d - ie_vol_5d.rolling(120).mean()) / ie_vol_5d.rolling(120).std()).clip(-4, 4).replace([np.inf, -np.inf], 0)
gold_ret_for_ie = gold_ret.reindex(ie_change.index)
ie_gold_corr = ie_change.rolling(5).corr(gold_ret_for_ie)
ie_sens_z = ((ie_gold_corr - ie_gold_corr.rolling(40).mean()) / ie_gold_corr.rolling(40).std()).clip(-4, 4).replace([np.inf, -np.inf], 0)
ie_submodel = pd.DataFrame({'ie_regime_prob': ie_regime_prob, 'ie_anchoring_z': ie_anchor_z, 'ie_gold_sensitivity_z': ie_sens_z})
print(f"Inflation Exp submodel: {len(ie_submodel)} rows")

# Options Market
skew_change = skew_data['Close'].diff()
gvz_change = gvz_data['Close'].diff()
opt_input = pd.DataFrame({'skew_chg': skew_change, 'gvz_chg': gvz_change}).dropna()
X_opt = opt_input.values
n_train_opt = int(len(X_opt) * 0.70)
hmm_opt = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42)
hmm_opt.fit(X_opt[:n_train_opt])
high_var_opt = hmm_highest_var_state(hmm_opt)
opt_probs = hmm_opt.predict_proba(X_opt)
opt_submodel = pd.DataFrame({'options_risk_regime_prob': pd.Series(opt_probs[:, high_var_opt], index=opt_input.index)})
print(f"Options submodel: {len(opt_submodel)} rows")

# Temporal Context (default 0.5)
tc_path = '../data/dataset_upload_clean/temporal_context.csv'
if os.path.exists(tc_path):
    tc_df = pd.read_csv(tc_path)
    tc_df['date'] = pd.to_datetime(tc_df['date'])
    tc_df = tc_df.set_index('date')
    print(f"Temporal Context: loaded CSV ({len(tc_df)} rows)")
else:
    tc_df = pd.DataFrame(index=gold_close.index)
    tc_df['temporal_context_score'] = 0.5
    print(f"Temporal Context: using default 0.5")
tc_submodel = tc_df[['temporal_context_score']]

# ============ Assemble features ============
print("\n" + "=" * 60)
print("ASSEMBLING FEATURES...")
print("=" * 60)

base_df = pd.DataFrame(index=gold_close.index)
base_df['gold_return_next'] = gold_return_next
base_df['_real_rate'] = real_rate_raw.reindex(base_df.index)
base_df['_dxy'] = dxy_close.reindex(base_df.index)
base_df['_vix'] = vix_raw.reindex(base_df.index)
base_df['_yield_spread'] = yield_spread.reindex(base_df.index)
base_df['_ie'] = ie_raw.reindex(base_df.index)
base_df = base_df.ffill()
base_df['real_rate_change'] = base_df['_real_rate'].diff()
base_df['dxy_change'] = base_df['_dxy'].diff()
base_df['vix'] = base_df['_vix']
base_df['yield_spread_change'] = base_df['_yield_spread'].diff()
base_df['inflation_exp_change'] = base_df['_ie'].diff()
base_df = base_df.drop(columns=['_real_rate', '_dxy', '_vix', '_yield_spread', '_ie'])

for name, sub_df in [('vix', vix_submodel), ('technical', tech_submodel), ('cross_asset', xasset_submodel),
                      ('yield_curve', yc_submodel), ('etf_flow', etf_submodel), ('inflation_exp', ie_submodel),
                      ('options', opt_submodel), ('temporal', tc_submodel)]:
    base_df = base_df.join(sub_df, how='left')

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

# NaN imputation
regime_cols = ['vix_regime_probability', 'tech_trend_regime_prob', 'xasset_regime_prob',
               'etf_regime_prob', 'ie_regime_prob', 'options_risk_regime_prob', 'temporal_context_score']
for col in regime_cols:
    if col in base_df.columns:
        base_df[col] = base_df[col].fillna(0.5)

z_cols = ['vix_mean_reversion_z', 'tech_mean_reversion_z', 'yc_spread_velocity_z', 'yc_curvature_z',
          'etf_capital_intensity', 'etf_pv_divergence', 'ie_anchoring_z', 'ie_gold_sensitivity_z']
for col in z_cols:
    if col in base_df.columns:
        base_df[col] = base_df[col].fillna(0.0)

for col in ['xasset_recession_signal', 'xasset_divergence']:
    if col in base_df.columns:
        base_df[col] = base_df[col].fillna(0.0)

for col in ['tech_volatility_regime', 'vix_persistence']:
    if col in base_df.columns:
        base_df[col] = base_df[col].fillna(base_df[col].median())

base_df = base_df.dropna(subset=['real_rate_change', 'dxy_change', 'vix', 'yield_spread_change', 'inflation_exp_change'])

missing = [c for c in FEATURE_COLUMNS if c not in base_df.columns]
if missing:
    print(f"ERROR: Missing features: {missing}")
    sys.exit(1)

print(f"Final dataset: {len(base_df)} rows, {len(FEATURE_COLUMNS)} features")
print(f"Date range: {base_df.index.min()} to {base_df.index.max()}")
nan_counts = base_df[FEATURE_COLUMNS].isna().sum()
nan_total = nan_counts.sum()
print(f"NaN in features: {nan_total}")
if nan_total > 0:
    for col, cnt in nan_counts[nan_counts > 0].items():
        print(f"  {col}: {cnt} NaN")

# ============ Train model ============
print("\n" + "=" * 60)
print("TRAINING MODEL...")
print("=" * 60)

trainable = base_df.dropna(subset=[TARGET]).copy()
n_total = len(trainable)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)
train_data = trainable.iloc[:n_train]
val_data = trainable.iloc[n_train:n_train+n_val]
test_data = trainable.iloc[n_train+n_val:]

X_train = train_data[FEATURE_COLUMNS].values
y_train = train_data[TARGET].values
X_val = val_data[FEATURE_COLUMNS].values
y_val = val_data[TARGET].values
X_test = test_data[FEATURE_COLUMNS].values
y_test = test_data[TARGET].values

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

MODEL_PARAMS = {
    'objective': 'reg:squarederror', 'max_depth': 2, 'min_child_weight': 14,
    'reg_lambda': 4.76, 'reg_alpha': 3.65, 'subsample': 0.478,
    'colsample_bytree': 0.371, 'learning_rate': 0.025,
    'tree_method': 'hist', 'eval_metric': 'rmse', 'verbosity': 0, 'seed': 42,
}

model = xgb.XGBRegressor(**MODEL_PARAMS, n_estimators=300, early_stopping_rounds=100)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

bootstrap_models = []
for seed in [42, 43, 44, 45, 46]:
    p = MODEL_PARAMS.copy()
    p['seed'] = seed
    m = xgb.XGBRegressor(**p, n_estimators=300, early_stopping_rounds=100)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    bootstrap_models.append(m)

pred_val = model.predict(X_val)
alpha_ols = np.clip(np.sum(pred_val * y_val) / (np.sum(pred_val**2) + 1e-10), 0.5, 10.0)
print(f"OLS alpha: {alpha_ols:.3f}")

pred_test = model.predict(X_test)
mask = (y_test != 0) & (pred_test != 0)
test_da = (np.sign(pred_test[mask]) == np.sign(y_test[mask])).mean()
print(f"Test DA: {test_da*100:.2f}%")

# ============ Test multiple dates ============
print("\n" + "=" * 60)
print("TESTING PREDICTIONS FOR MULTIPLE DATES")
print("=" * 60)

test_dates = ['2026-02-18', '2026-02-14', '2026-01-15', '2025-12-01', '2025-06-15', '2025-01-02']
results = []

for date_str in test_dates:
    target = pd.Timestamp(date_str)
    available = base_df.index[base_df.index <= target]

    if len(available) == 0:
        print(f"\n{date_str}: NO DATA AVAILABLE")
        results.append({'date': date_str, 'status': 'NO_DATA'})
        continue

    actual_date = available[-1]
    used_fallback = actual_date != target

    # Check features
    row = base_df.loc[actual_date, FEATURE_COLUMNS]
    nan_features = row[row.isna()].index.tolist()

    # Predict
    X_pred = base_df.loc[[actual_date], FEATURE_COLUMNS].values
    pred_raw = model.predict(X_pred)[0]
    pred_scaled = pred_raw * alpha_ols

    boot_preds = np.array([m.predict(X_pred)[0] for m in bootstrap_models])
    boot_std = boot_preds.std()

    direction = 'UP' if pred_raw > 0 else 'DOWN'
    gold_price = gold_close.loc[:actual_date].iloc[-1]

    # Check if actual next-day return is available
    actual_next = None
    if actual_date in base_df.index and not pd.isna(base_df.loc[actual_date, 'gold_return_next']):
        actual_next = base_df.loc[actual_date, 'gold_return_next']

    result = {
        'date': date_str,
        'actual_date': actual_date.strftime('%Y-%m-%d'),
        'fallback': used_fallback,
        'nan_features': nan_features,
        'pred_raw': pred_raw,
        'pred_scaled': pred_scaled,
        'direction': direction,
        'gold_price': gold_price,
        'boot_std': boot_std,
        'actual_next': actual_next,
        'status': 'OK' if len(nan_features) == 0 else 'HAS_NAN',
    }
    results.append(result)

    status = "OK" if len(nan_features) == 0 else f"NaN in: {nan_features}"
    fallback_note = f" (used {actual_date.strftime('%Y-%m-%d')})" if used_fallback else ""
    correct_note = ""
    if actual_next is not None:
        correct = (pred_raw > 0 and actual_next > 0) or (pred_raw < 0 and actual_next < 0)
        correct_note = f" {'CORRECT' if correct else 'WRONG'}"

    print(f"\n{date_str}{fallback_note}:")
    print(f"  Status:    {status}")
    print(f"  Gold:      ${gold_price:.2f}")
    print(f"  Direction: {direction}")
    print(f"  Raw:       {pred_raw:+.4f}%")
    print(f"  Scaled:    {pred_scaled:+.4f}%")
    print(f"  Boot Std:  {boot_std:.4f}")
    if actual_next is not None:
        print(f"  Actual:    {actual_next:+.4f}%{correct_note}")

# ============ Summary ============
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
ok_count = sum(1 for r in results if r.get('status') == 'OK')
nan_count = sum(1 for r in results if r.get('status') == 'HAS_NAN')
no_data_count = sum(1 for r in results if r.get('status') == 'NO_DATA')
print(f"  Total tests: {len(results)}")
print(f"  OK: {ok_count}, Has NaN: {nan_count}, No data: {no_data_count}")

# Check overall feature coverage
print(f"\nFeature coverage across full dataset:")
for col in FEATURE_COLUMNS:
    non_null = base_df[col].notna().sum()
    total = len(base_df)
    pct = non_null / total * 100
    if pct < 100:
        print(f"  {col}: {non_null}/{total} ({pct:.1f}%)")
    else:
        print(f"  {col}: 100%")

# Direction accuracy for past dates
past_results = [r for r in results if r.get('actual_next') is not None]
if past_results:
    correct_count = sum(1 for r in past_results
                       if (r['pred_raw'] > 0 and r['actual_next'] > 0) or
                          (r['pred_raw'] < 0 and r['actual_next'] < 0))
    print(f"\nDirection accuracy on tested past dates: {correct_count}/{len(past_results)} ({correct_count/len(past_results)*100:.0f}%)")

print("\nDone.")
