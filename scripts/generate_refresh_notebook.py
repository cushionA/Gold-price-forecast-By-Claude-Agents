"""Generate the refresh_all_submodels notebook."""
import json

cells = []

def _to_lines(source):
    lines = source.split("\n")
    # Each line except the last needs a trailing newline
    return [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _to_lines(source)})

def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": _to_lines(source), "outputs": [], "execution_count": None})

# ============================================================
# Cell 0: Title
# ============================================================
md("""# Gold - Refresh All Submodels
Regenerates all 8 submodel outputs + base_features_raw.csv with latest API data.
Uses fixed best hyperparameters from original successful runs (no Optuna).""")

# ============================================================
# Cell 1: Setup
# ============================================================
code("""import subprocess, sys
print("Installing hmmlearn...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'hmmlearn', '-q'])

import numpy as np
import pandas as pd
import yfinance as yf
import json, os, glob, warnings
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings('ignore')

np.random.seed(42)
print(f"Started: {datetime.now().isoformat()}")

# FRED API
try:
    from kaggle_secrets import UserSecretsClient
    FRED_API_KEY = UserSecretsClient().get_secret("FRED_API_KEY")
except Exception:
    import os
    FRED_API_KEY = os.environ.get('FRED_API_KEY')
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY not available")

from fredapi import Fred
fred = Fred(api_key=FRED_API_KEY)
print("FRED API ready")

START_DATE = "2014-01-01"

def yf_close(ticker, start=START_DATE):
    d = yf.download(ticker, start=start, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)
    d.index = d.index.tz_localize(None)
    return d

def fred_series(sid, start=START_DATE):
    return fred.get_series(sid, observation_start=start)

print("Utility functions ready")""")

# ============================================================
# Cell 2: Fetch all raw data
# ============================================================
code("""print("="*60)
print("FETCHING ALL RAW DATA")
print("="*60)

# Gold
gold_raw = yf_close("GC=F")
gold_close = gold_raw["Close"].copy()
gold_return = gold_close.pct_change() * 100
gold_return_next = gold_return.shift(-1)
print(f"Gold: {len(gold_close)} rows, last={gold_close.index[-1].date()}")

# GLD ETF
gld_raw = yf_close("GLD")
print(f"GLD: {len(gld_raw)} rows")

# Silver, Copper, SP500
silver_raw = yf_close("SI=F")
copper_raw = yf_close("HG=F")
sp500_raw = yf_close("^GSPC")

# DXY, CNY
dxy_raw = yf_close("DX-Y.NYB")
cny_raw = yf_close("CNY=X")

# SKEW, GVZ (options)
skew_raw = yf_close("^SKEW", start="2014-10-01")
gvz_raw = yf_close("^GVZ", start="2014-10-01")

# Yield curve proxies
tnx_raw = yf_close("^TNX", start="2014-10-01")  # 10Y
irx_raw = yf_close("^IRX", start="2014-10-01")  # 13-week
fvx_raw = yf_close("^FVX", start="2014-10-01")  # 5Y

# FRED series
real_rate = fred_series("DFII10")
vix_fred = fred_series("VIXCLS")
dgs10 = fred_series("DGS10")
dgs2 = fred_series("DGS2")
t10yie = fred_series("T10YIE")

print(f"\\nAll data fetched. FRED last dates:")
print(f"  DFII10: {real_rate.index[-1].date()}")
print(f"  VIXCLS: {vix_fred.index[-1].date()}")
print(f"  T10YIE: {t10yie.index[-1].date()}")""")

# ============================================================
# Cell 3: VIX submodel
# ============================================================
code("""print("="*60)
print("SUBMODEL 1: VIX")
print("="*60)

# GMM on VIX log-changes
vix_df = pd.DataFrame({'date': vix_fred.index, 'vix': vix_fred.values})
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_df = vix_df.dropna(subset=['vix'])
vix_df = vix_df.set_index('date').asfreq('D').ffill(limit=3).reset_index()
vix_df = vix_df.dropna(subset=['vix'])
vix_df['vix_log_change'] = np.log(vix_df['vix']).diff()
vix_df = vix_df.dropna(subset=['vix_log_change']).reset_index(drop=True)

n_train = int(len(vix_df) * 0.70)
X_train = vix_df['vix_log_change'].values[:n_train].reshape(-1, 1)
X_full = vix_df['vix_log_change'].values.reshape(-1, 1)

gmm = GaussianMixture(n_components=2, covariance_type='full', max_iter=100,
                       random_state=42, n_init=5)
gmm.fit(X_train)
probs = gmm.predict_proba(X_full)
state_vars = [float(cov[0, 0]) for cov in gmm.covariances_]
regime = probs[:, np.argmax(state_vars)]

s = pd.Series(vix_df['vix'].values)
zscore = ((s - s.rolling(60, min_periods=60).mean()) /
           s.rolling(60, min_periods=60).std()).clip(-4, 4).values

def autocorr_lag1(x):
    return pd.Series(x).autocorr(lag=1) if len(x) >= 2 else np.nan
persistence = (pd.Series(vix_df['vix_log_change'].values)
               .rolling(20, min_periods=20)
               .apply(autocorr_lag1, raw=True).values)

vix_out = pd.DataFrame({
    'date': vix_df['date'],
    'vix_regime_probability': regime,
    'vix_mean_reversion_z': zscore,
    'vix_persistence': persistence,
})
vix_out = vix_out.ffill()
print(f"VIX output: {len(vix_out)} rows, {vix_out['date'].min().date()} to {vix_out['date'].max().date()}")""")

# ============================================================
# Cell 4: Technical submodel
# ============================================================
code("""print("="*60)
print("SUBMODEL 2: TECHNICAL")
print("="*60)

tech_df = gld_raw[['Open','High','Low','Close','Volume']].copy()
tech_df.columns = ['open','high','low','close','volume']
tech_df = tech_df.reset_index()
tech_df.columns = ['date'] + list(tech_df.columns[1:])
tech_df['date'] = pd.to_datetime(tech_df['date'])

tech_df['returns'] = tech_df['close'].pct_change()
h, l = tech_df['high'].clip(1e-8), tech_df['low'].clip(1e-8)
c, o = tech_df['close'].clip(1e-8), tech_df['open'].clip(1e-8)
tech_df['gk_vol'] = np.sqrt(0.5*(np.log(h/l)**2) - (2*np.log(2)-1)*(np.log(c/o)**2)).clip(1e-8)
tech_df = tech_df.dropna(subset=['returns','gk_vol']).reset_index(drop=True)

n_train = int(len(tech_df) * 0.70)
X = np.column_stack([tech_df['returns'].values, tech_df['gk_vol'].values])
X_train = X[:n_train]
mask = ~np.isnan(X_train).any(axis=1)
X_train_clean = X_train[mask]

best_model, best_score = None, -np.inf
for seed in range(5):
    try:
        m = GaussianHMM(n_components=2, covariance_type='full', n_iter=200, tol=1e-4, random_state=seed)
        m.fit(X_train_clean)
        s = m.score(X_train_clean)
        if s > best_score: best_score, best_model = s, m
    except: continue

X_full = X.copy()
for i in range(X_full.shape[1]):
    med = np.nanmedian(X_full[:, i])
    X_full[np.isnan(X_full[:, i]), i] = med
probs = best_model.predict_proba(X_full)
high_var = np.argmax([np.trace(best_model.covars_[i]) for i in range(2)])
regime = probs[:, high_var]

s_ret = pd.Series(tech_df['returns'].values)
zscore = ((s_ret - s_ret.rolling(20, min_periods=20).mean()) /
           s_ret.rolling(20, min_periods=20).std()).clip(-4, 4).values

gk_daily = np.sqrt(0.5*(np.log(h.values/l.values)**2) - (2*np.log(2)-1)*(np.log(c.values/o.values)**2)).clip(1e-8)
sg = pd.Series(gk_daily[tech_df.index])
vol_z = ((sg - sg.rolling(60, min_periods=60).mean()) /
          sg.rolling(60, min_periods=60).std()).clip(-4, 4).values

tech_out = pd.DataFrame({
    'date': tech_df['date'],
    'tech_trend_regime_prob': regime,
    'tech_mean_reversion_z': zscore,
    'tech_volatility_regime': vol_z,
}).ffill()
print(f"Technical output: {len(tech_out)} rows")""")

# ============================================================
# Cell 5: Cross-asset submodel
# ============================================================
code("""print("="*60)
print("SUBMODEL 3: CROSS-ASSET")
print("="*60)

xa = (pd.DataFrame({'gold_close': gold_close})
      .join(pd.DataFrame({'silver_close': silver_raw['Close']}), how='inner')
      .join(pd.DataFrame({'copper_close': copper_raw['Close']}), how='inner'))
xa = xa.ffill(limit=3).dropna()
xa['gold_ret'] = xa['gold_close'].pct_change()
xa['silver_ret'] = xa['silver_close'].pct_change()
xa['copper_ret'] = xa['copper_close'].pct_change()

hmm_data = xa[['gold_ret','silver_ret','copper_ret']].dropna()
X_xa = hmm_data.values
best_model, best_score = None, -np.inf
for seed in range(10):
    try:
        m = GaussianHMM(n_components=3, covariance_type='full', n_iter=200, tol=1e-4, random_state=seed)
        m.fit(X_xa)
        s = m.score(X_xa)
        if s > best_score: best_score, best_model = s, m
    except: continue

probs = best_model.predict_proba(X_xa)
traces = [np.trace(best_model.covars_[i]) for i in range(3)]
xa.loc[hmm_data.index, 'xasset_regime_prob'] = probs[:, np.argmax(traces)]

gc_ratio = xa['gold_close'] / xa['copper_close']
gc_z = (gc_ratio - gc_ratio.rolling(90).mean()) / gc_ratio.rolling(90).std()
xa['xasset_recession_signal'] = gc_z.diff().clip(-4, 4)

gs_diff = xa['gold_ret'] - xa['silver_ret']
xa['xasset_divergence'] = ((gs_diff - gs_diff.rolling(20).mean()) /
                            gs_diff.rolling(20).std()).clip(-4, 4)

xa_out = xa[['xasset_regime_prob','xasset_recession_signal','xasset_divergence']].copy()
xa_out = xa_out.ffill()
xa_out = xa_out[xa_out.index >= '2015-01-30']
xa_out.index.name = 'Date'
print(f"Cross-asset output: {len(xa_out)} rows")""")

# ============================================================
# Cell 6: Yield curve submodel
# ============================================================
code("""print("="*60)
print("SUBMODEL 4: YIELD CURVE")
print("="*60)

yc = pd.DataFrame({
    'dgs10': tnx_raw['Close'].values / 100,
    'dgs2': irx_raw['Close'].values / 100 * (365/91),
    'dgs5': fvx_raw['Close'].values / 100,
}, index=tnx_raw.index)
yc = yc.dropna()
yc['dgs10_change'] = yc['dgs10'].diff()
yc['dgs2_change'] = yc['dgs2'].diff()
yc['spread'] = yc['dgs10'] - yc['dgs2']
yc['curvature_raw'] = yc['dgs5'] - 0.5*(yc['dgs2'] + yc['dgs10'])
yc = yc.dropna(subset=['dgs10_change','dgs2_change'])

# Gold target for state selection
gold_ret_yc = gold_close.pct_change().shift(-1) * 100
common = yc.index.intersection(gold_ret_yc.dropna().index)
yc = yc.loc[common]
gold_ret_vals = gold_ret_yc.loc[common].values

n_train = int(len(yc) * 0.70)
X_yc = np.column_stack([yc['dgs10_change'].values, yc['dgs2_change'].values])
X_train = X_yc[:n_train]
valid = ~np.isnan(X_train).any(axis=1)

best_model, best_score = None, -np.inf
for seed in [0, 42, 123, 456, 789]:
    try:
        m = GaussianHMM(n_components=3, covariance_type='full', n_iter=100, tol=1e-4, random_state=seed)
        m.fit(X_train[valid])
        s = m.score(X_train[valid])
        if s > best_score: best_score, best_model = s, m
    except: continue

valid_full = ~np.isnan(X_yc).any(axis=1)
probs = np.full((len(yc), 3), np.nan)
probs[valid_full] = best_model.predict_proba(X_yc[valid_full])
states_train = best_model.predict(X_train[valid])
gold_train = gold_ret_vals[:n_train][valid]
state_means = [np.nanmean(gold_train[states_train==s]) if (states_train==s).sum()>0 else 0.0
               for s in range(3)]
target_state = np.argmin(state_means)
regime = pd.Series(probs[:, target_state]).ffill().values

spread_change = yc['spread'].diff(5)
vel_z = ((spread_change - spread_change.rolling(60).mean()) /
          spread_change.rolling(60).std()).clip(-4, 4).ffill()

curv_change = yc['curvature_raw'].diff()
curv_z = ((curv_change - curv_change.rolling(60).mean()) /
           curv_change.rolling(60).std()).clip(-4, 4).ffill()

yc_out = pd.DataFrame({
    'yc_spread_velocity_z': vel_z.values,
    'yc_curvature_z': curv_z.values,
}, index=yc.index)
yc_out = yc_out.ffill()
yc_out.index.name = 'index'
print(f"Yield curve output: {len(yc_out)} rows")""")

# ============================================================
# Cell 7: ETF flow submodel
# ============================================================
code("""print("="*60)
print("SUBMODEL 5: ETF FLOW")
print("="*60)

etf = pd.DataFrame({
    'gld_close': gld_raw['Close'],
    'gld_volume': gld_raw['Volume'],
}, index=gld_raw.index).dropna()
etf['volume_ma20'] = etf['gld_volume'].rolling(20).mean()
etf['dollar_volume'] = etf['gld_close'] * etf['gld_volume']
etf['log_vol_ratio'] = np.log(etf['gld_volume'] / etf['volume_ma20'].clip(1))
etf['gld_ret'] = etf['gld_close'].pct_change()
etf['gold_return'] = gold_close.reindex(etf.index).pct_change()
etf = etf.dropna()

n_train = int(len(etf) * 0.70)
X_etf = etf[['log_vol_ratio','gold_return']].values
X_train = X_etf[:n_train]
mask = ~np.isnan(X_train).any(axis=1)

best_model, best_score = None, -np.inf
for seed in range(10):
    try:
        m = GaussianHMM(n_components=3, covariance_type='full', n_iter=200, tol=1e-4, random_state=seed)
        m.fit(X_train[mask])
        s = m.score(X_train[mask])
        if s > best_score: best_score, best_model = s, m
    except: continue

X_full = X_etf.copy()
for i in range(2):
    med = np.nanmedian(X_full[:, i])
    X_full[np.isnan(X_full[:, i]), i] = med
probs = best_model.predict_proba(X_full)
states_train = best_model.predict(X_train[mask])
gold_train = etf['gold_return'].values[:n_train][mask]
state_means = [np.nanmean(gold_train[states_train==s]) if (states_train==s).sum()>0 else 0.0
               for s in range(3)]
accum_state = np.argmax(state_means)
regime = probs[:, accum_state]

dv = etf['dollar_volume']
cap_mean = dv.rolling(60).mean()
cap_std = dv.rolling(60).std()
capital_z = ((dv - cap_mean) / cap_std.clip(1e-8)).replace([np.inf, -np.inf], 0).fillna(0)

vol_changes = etf['gld_volume'].pct_change()
rolling_corr = etf['gld_ret'].rolling(5).corr(vol_changes)
corr_mean = rolling_corr.rolling(60).mean()
corr_std = rolling_corr.rolling(60).std()
pv_div = ((rolling_corr - corr_mean) / corr_std.clip(1e-8)).replace([np.inf,-np.inf], 0).fillna(0)

etf_out = pd.DataFrame({
    'Date': etf.index,
    'etf_regime_prob': regime,
    'etf_capital_intensity': capital_z.values,
    'etf_pv_divergence': pv_div.values,
}).ffill()
print(f"ETF flow output: {len(etf_out)} rows")""")

# ============================================================
# Cell 8: Inflation expectation submodel
# ============================================================
code("""print("="*60)
print("SUBMODEL 6: INFLATION EXPECTATION")
print("="*60)

ie_series = t10yie.dropna()
ie_df = pd.DataFrame({'ie': ie_series.values}, index=ie_series.index)
ie_df = ie_df.asfreq('D').ffill(limit=3).dropna()
ie_df['ie_change'] = ie_df['ie'].diff()
ie_df['ie_vol_5d'] = ie_df['ie_change'].rolling(5).std()
ie_df = ie_df.dropna()

gold_ret_ie = gold_close.pct_change() * 100
ie_df['gold_return'] = gold_ret_ie.reindex(ie_df.index).ffill()
ie_df = ie_df.dropna()

n_train = int(len(ie_df) * 0.70)
X_ie = ie_df[['ie_change','ie_vol_5d']].values
X_train = X_ie[:n_train]
mask = ~np.isnan(X_train).any(axis=1)

best_model, best_score = None, -np.inf
for seed in [42, 0, 123, 456, 789]:
    try:
        m = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, tol=1e-4, random_state=seed)
        m.fit(X_train[mask])
        s = m.score(X_train[mask])
        if s > best_score: best_score, best_model = s, m
    except: continue

X_full = X_ie.copy()
for i in range(2):
    med = np.nanmedian(X_full[:, i])
    X_full[np.isnan(X_full[:, i]), i] = med
probs = best_model.predict_proba(X_full)
high_var = np.argmax([best_model.covars_[i][0,0] for i in range(2)])
regime = probs[:, high_var]

vol_short = ie_df['ie_change'].rolling(10).std()
vol_mean = vol_short.rolling(120).mean()
vol_std = vol_short.rolling(120).std()
anchoring_z = ((vol_short - vol_mean) / vol_std.clip(1e-8)).clip(-4, 4).fillna(0)

rolling_corr = ie_df['ie_change'].rolling(5).corr(ie_df['gold_return'])
corr_mean = rolling_corr.rolling(60).mean()
corr_std = rolling_corr.rolling(60).std()
sensitivity_z = ((rolling_corr - corr_mean) / corr_std.clip(1e-8)).clip(-4, 4).fillna(0)

ie_out = pd.DataFrame({
    'ie_regime_prob': regime,
    'ie_anchoring_z': anchoring_z.values,
    'ie_gold_sensitivity_z': sensitivity_z.values,
}, index=ie_df.index)
ie_out = ie_out.ffill()
ie_out.index.name = 'Unnamed: 0'
print(f"Inflation expectation output: {len(ie_out)} rows")""")

# ============================================================
# Cell 9: Options market submodel
# ============================================================
code("""print("="*60)
print("SUBMODEL 7: OPTIONS MARKET")
print("="*60)

skew_close = skew_raw['Close']
gvz_close = gvz_raw['Close']
gold_opt = gold_close.reindex(skew_close.index)

opt_df = pd.DataFrame({
    'skew': skew_close,
    'gvz': gvz_close,
    'gold': gold_opt,
}).ffill(limit=3).dropna()
opt_df['skew_change'] = opt_df['skew'].diff()
opt_df['gvz_change'] = opt_df['gvz'].diff()
opt_df = opt_df.dropna()

n_train = int(len(opt_df) * 0.70)
X_opt = opt_df[['skew_change','gvz_change']].values

scaler = StandardScaler()
scaler.fit(X_opt[:n_train])
X_scaled = scaler.transform(X_opt)

hmm = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42)
hmm.fit(X_scaled[:n_train])
probs = hmm.predict_proba(X_scaled)
traces = [np.trace(hmm.covars_[i]) for i in range(2)]
raw_regime = probs[:, np.argmax(traces)]
smoothed = pd.Series(raw_regime).ewm(span=5, adjust=False).mean().values

ema_s = opt_df['gvz'].ewm(span=5, adjust=False).mean()
ema_l = opt_df['gvz'].ewm(span=30, adjust=False).mean()
momentum = ema_s - ema_l
mom_mean = momentum.rolling(60).mean()
mom_std = momentum.rolling(60).std()
gvz_z = ((momentum - mom_mean) / (mom_std + 1e-8)).clip(-3, 3)

opt_out = pd.DataFrame({
    'Date': opt_df.index,
    'options_regime_smooth': smoothed,
    'options_gvz_momentum_z': gvz_z.values,
}).ffill()
print(f"Options market output: {len(opt_out)} rows")""")

# ============================================================
# Cell 10: Temporal context Transformer
# ============================================================
code("""print("="*60)
print("SUBMODEL 8: TEMPORAL CONTEXT (Transformer)")
print("="*60)

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Build input features (14 total: 5 base + 9 submodel)
ref_idx = gold_close.dropna().index
base = pd.DataFrame(index=ref_idx)
base['real_rate_change'] = real_rate.reindex(ref_idx, method='ffill').diff()
base['dxy_change'] = dxy_raw['Close'].reindex(ref_idx, method='ffill').diff()
base['vix'] = vix_fred.reindex(ref_idx, method='ffill')
base['yield_spread_change'] = (dgs10.reindex(ref_idx, method='ffill') - dgs2.reindex(ref_idx, method='ffill')).diff()
base['inflation_exp_change'] = t10yie.reindex(ref_idx, method='ffill').diff()

# Merge submodel outputs
def to_daily_index(df, date_col=None):
    if date_col and date_col in df.columns:
        df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df

vix_sub = to_daily_index(vix_out, 'date')[['vix_regime_probability','vix_mean_reversion_z']]
tech_sub = to_daily_index(tech_out, 'date')[['tech_trend_regime_prob','tech_mean_reversion_z','tech_volatility_regime']]
xa_sub = xa_out[['xasset_regime_prob','xasset_divergence']]
etf_sub = to_daily_index(etf_out, 'Date')[['etf_regime_prob']]
opt_sub = to_daily_index(opt_out, 'Date')[['options_regime_smooth']]
opt_sub = opt_sub.rename(columns={'options_regime_smooth': 'options_risk_regime_prob'})

for sub in [vix_sub, tech_sub, xa_sub, etf_sub, opt_sub]:
    base = base.join(sub, how='left')

base = base.dropna(subset=['real_rate_change','dxy_change','vix'])
base = base.ffill().fillna(0)

TC_FEATURES = [
    'real_rate_change','dxy_change','vix','yield_spread_change','inflation_exp_change',
    'vix_regime_probability','vix_mean_reversion_z',
    'tech_trend_regime_prob','tech_mean_reversion_z','tech_volatility_regime',
    'xasset_regime_prob','xasset_divergence',
    'etf_regime_prob','options_risk_regime_prob',
]
tc_data = base[TC_FEATURES].values.astype(np.float32)
n_total = len(tc_data)
n_train_tc = int(n_total * 0.70)
n_val_tc = int(n_total * 0.15)

scaler_tc = StandardScaler()
scaler_tc.fit(tc_data[:n_train_tc])
tc_scaled = scaler_tc.transform(tc_data)
tc_scaled = np.nan_to_num(tc_scaled, 0)

print(f"TC input: {tc_scaled.shape}, features: {len(TC_FEATURES)}")

# Transformer architecture
WINDOW = 10
D_MODEL = 24
N_HEADS = 2
N_LAYERS = 1
FFN_RATIO = 2
DROPOUT = 0.2
MASK_RATIO = 0.2
LR = 5e-4
WD = 0.03
PATIENCE = 10
MAX_EPOCHS = 200

class TemporalContextTransformer(nn.Module):
    def __init__(self, input_dim=14, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 ffn_ratio=FFN_RATIO, dropout=DROPOUT, max_seq_len=WINDOW):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*ffn_ratio,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.bottleneck = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.decoder_expand = nn.Linear(1, d_model)
        self.decoder_output = nn.Linear(d_model, input_dim)
        self.input_dropout = nn.Dropout(dropout)

    def encode(self, x):
        h = self.input_proj(x)
        pos = torch.arange(x.size(1), device=x.device)
        h = h + self.pos_encoding(pos)
        h = self.input_dropout(h)
        encoded = self.encoder(h)
        pooled = encoded.mean(dim=1)
        return self.bottleneck(pooled)

    def forward(self, x, mask=None):
        if mask is not None:
            x_masked = x.clone()
            x_masked[mask] = 0.0
        else:
            x_masked = x
        score = self.encode(x_masked)
        expanded = self.decoder_expand(score).unsqueeze(1).expand(-1, x.size(1), -1)
        recon = self.decoder_output(expanded)
        return recon, score

n_params = sum(p.numel() for p in TemporalContextTransformer().parameters())
print(f"Model params: {n_params}")

# Create windows
def make_windows(data, window_size):
    windows = []
    for i in range(window_size, len(data)+1):
        windows.append(data[i-window_size:i])
    return np.array(windows)

all_windows = make_windows(tc_scaled, WINDOW)
train_windows = all_windows[:n_train_tc - WINDOW + 1]
val_start = n_train_tc - WINDOW + 1
val_end = val_start + n_val_tc
val_windows = all_windows[val_start:val_end]

train_tensor = torch.tensor(train_windows, dtype=torch.float32)
val_tensor = torch.tensor(val_windows, dtype=torch.float32)

# Training
model_tc = TemporalContextTransformer(input_dim=len(TC_FEATURES)).to(device)
optimizer = torch.optim.AdamW(model_tc.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

best_val_loss = float('inf')
patience_counter = 0
batch_size = 256

for epoch in range(MAX_EPOCHS):
    model_tc.train()
    perm = torch.randperm(len(train_tensor))
    total_loss = 0
    n_batches = 0
    for i in range(0, len(train_tensor), batch_size):
        batch = train_tensor[perm[i:i+batch_size]].to(device)
        mask = torch.rand(batch.size(0), batch.size(1), device=device) < MASK_RATIO
        mask[:, 0] = False
        mask[:, -1] = True
        recon, _ = model_tc(batch, mask)
        loss = ((recon[mask] - batch[mask])**2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_tc.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    scheduler.step()

    model_tc.eval()
    with torch.no_grad():
        val_batch = val_tensor.to(device)
        mask_v = torch.rand(val_batch.size(0), val_batch.size(1), device=device) < MASK_RATIO
        mask_v[:, 0] = False; mask_v[:, -1] = True
        recon_v, _ = model_tc(val_batch, mask_v)
        val_loss = ((recon_v[mask_v] - val_batch[mask_v])**2).mean().item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model_tc.state_dict().items()}
    else:
        patience_counter += 1

    if epoch % 20 == 0:
        print(f"  Epoch {epoch}: train={total_loss/max(n_batches,1):.6f}, val={val_loss:.6f}, patience={patience_counter}")

    if patience_counter >= PATIENCE:
        print(f"  Early stopping at epoch {epoch}")
        break

model_tc.load_state_dict(best_state)
model_tc.to(device)
model_tc.eval()

# Generate scores for all windows
all_tensor = torch.tensor(all_windows, dtype=torch.float32)
scores = []
with torch.no_grad():
    for i in range(0, len(all_tensor), batch_size):
        batch = all_tensor[i:i+batch_size].to(device)
        s = model_tc.encode(batch)
        scores.append(s.cpu().numpy())
scores = np.concatenate(scores).flatten()

tc_dates = base.index[WINDOW-1:]
tc_out = pd.DataFrame({
    'date': tc_dates[:len(scores)],
    'temporal_context_score': scores,
})
print(f"Temporal context output: {len(tc_out)} rows, score range: [{scores.min():.4f}, {scores.max():.4f}]")""")

# ============================================================
# Cell 11: Generate base_features_raw.csv
# ============================================================
code("""print("="*60)
print("GENERATING base_features_raw.csv")
print("="*60)

ref = gold_close.dropna().index
bf = pd.DataFrame(index=ref)
bf['gold_return_next'] = gold_return_next
bf['real_rate_real_rate'] = real_rate.reindex(ref, method='ffill')
bf['dxy_dxy'] = dxy_raw['Close'].reindex(ref, method='ffill')
bf['vix_vix'] = vix_fred.reindex(ref, method='ffill')
bf['yield_curve_yield_spread'] = (dgs10.reindex(ref, method='ffill') - dgs2.reindex(ref, method='ffill'))
bf['inflation_expectation_inflation_expectation'] = t10yie.reindex(ref, method='ffill')
bf = bf.dropna(subset=['gold_return_next'])
bf.index.name = 'Date'

print(f"base_features_raw: {len(bf)} rows, {bf.index.min().date()} to {bf.index.max().date()}")
print(f"Columns: {list(bf.columns)}")""")

# ============================================================
# Cell 12: Save all outputs
# ============================================================
code("""print("="*60)
print("SAVING ALL OUTPUTS")
print("="*60)

# VIX
vix_save = vix_out.copy()
vix_save['date'] = pd.to_datetime(vix_save['date']).dt.strftime('%Y-%m-%d')
vix_save.to_csv('vix.csv', index=False)
print(f"Saved vix.csv: {len(vix_save)} rows")

# Technical
tech_save = tech_out.copy()
tech_save['date'] = pd.to_datetime(tech_save['date']).dt.strftime('%Y-%m-%d')
tech_save.to_csv('technical.csv', index=False)
print(f"Saved technical.csv: {len(tech_save)} rows")

# Cross-asset
xa_save = xa_out.reset_index()
xa_save.columns = ['Date'] + list(xa_save.columns[1:])
xa_save['Date'] = pd.to_datetime(xa_save['Date']).dt.strftime('%Y-%m-%d')
xa_save.to_csv('cross_asset.csv', index=False)
print(f"Saved cross_asset.csv: {len(xa_save)} rows")

# Yield curve
yc_save = yc_out.reset_index()
yc_save.columns = ['index'] + list(yc_save.columns[1:])
yc_save['index'] = pd.to_datetime(yc_save['index']).dt.strftime('%Y-%m-%d')
yc_save.to_csv('yield_curve.csv', index=False)
print(f"Saved yield_curve.csv: {len(yc_save)} rows")

# ETF flow
etf_save = etf_out.copy()
etf_save['Date'] = pd.to_datetime(etf_save['Date']).dt.strftime('%Y-%m-%d')
etf_save.to_csv('etf_flow.csv', index=False)
print(f"Saved etf_flow.csv: {len(etf_save)} rows")

# Inflation expectation
ie_save = ie_out.reset_index()
ie_save.columns = ['Unnamed: 0'] + list(ie_save.columns[1:])
ie_save['Unnamed: 0'] = pd.to_datetime(ie_save['Unnamed: 0']).dt.strftime('%Y-%m-%d')
ie_save.to_csv('inflation_expectation.csv', index=False)
print(f"Saved inflation_expectation.csv: {len(ie_save)} rows")

# Options market
opt_save = opt_out.copy()
opt_save['Date'] = pd.to_datetime(opt_save['Date']).dt.strftime('%Y-%m-%d')
opt_save.to_csv('options_market.csv', index=False)
print(f"Saved options_market.csv: {len(opt_save)} rows")

# Temporal context
tc_save = tc_out.copy()
tc_save['date'] = pd.to_datetime(tc_save['date']).dt.strftime('%Y-%m-%d')
tc_save.to_csv('temporal_context.csv', index=False)
print(f"Saved temporal_context.csv: {len(tc_save)} rows")

# Base features raw
bf_save = bf.reset_index()
bf_save['Date'] = pd.to_datetime(bf_save['Date']).dt.strftime('%Y-%m-%d')
bf_save.to_csv('base_features_raw.csv', index=False)
print(f"Saved base_features_raw.csv: {len(bf_save)} rows")

print(f"\\n{'='*60}")
print("ALL OUTPUTS SAVED SUCCESSFULLY")
print(f"{'='*60}")
print(f"Finished: {datetime.now().isoformat()}")
print(f"\\nFiles: vix.csv, technical.csv, cross_asset.csv, yield_curve.csv,")
print(f"  etf_flow.csv, inflation_expectation.csv, options_market.csv,")
print(f"  temporal_context.csv, base_features_raw.csv")""")

# ============================================================
# Build notebook
# ============================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

import os
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "notebooks", "refresh_all_submodels", "train.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Generated: {out_path}")
print(f"Total cells: {len(cells)}")
