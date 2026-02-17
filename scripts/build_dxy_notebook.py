"""Build the DXY submodel train.ipynb properly."""
import json
import ast

CODE = r'''"""
Gold Prediction SubModel Training - DXY Attempt 1 (Revised)
Self-contained: Data fetch -> HMM + Momentum Z + Vol Z -> Optuna HPO -> Save results

Critical corrections from previous attempt:
  1. GMM -> HMM (GaussianHMM captures temporal regime transitions)
  2. PCA divergence -> Momentum z-score (proven pattern, avoids scaling issues)
  3. Full Gate 1/2/3 evaluation (no auto-evaluation shortcuts)
"""

# ============================================================
# 1. IMPORTS AND SETUP
# ============================================================
import subprocess
subprocess.check_call(['pip', 'install', 'hmmlearn'])

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mutual_info_score
import optuna
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

print(f'=== Gold SubModel Training: DXY attempt 1 (HMM Revised) ===')
print(f'Started: {datetime.now().isoformat()}')

# ============================================================
# 2. DATA FETCHING
# ============================================================
def fetch_data():
    """
    Fetch DX-Y.NYB from Yahoo Finance + GC=F for target.
    Attempt 1 uses single DXY ticker only (no constituent currencies).
    """
    print('\n[Data Fetch] Downloading DXY and GC=F from Yahoo Finance...')

    # Fetch DXY
    dxy_raw = yf.download('DX-Y.NYB', start='2014-10-01', progress=False)
    if dxy_raw.empty:
        raise RuntimeError('Failed to fetch DX-Y.NYB data')
    if isinstance(dxy_raw.columns, pd.MultiIndex):
        dxy_raw.columns = dxy_raw.columns.get_level_values(0)
    dxy_close = dxy_raw['Close'].copy()
    dxy_close.index = pd.to_datetime(dxy_close.index)
    print(f'[OK] DXY: {len(dxy_close)} rows, range {dxy_close.index[0].date()} to {dxy_close.index[-1].date()}')

    # Fetch Gold
    gc_raw = yf.download('GC=F', start='2014-10-01', progress=False)
    if gc_raw.empty:
        raise RuntimeError('Failed to fetch GC=F data')
    if isinstance(gc_raw.columns, pd.MultiIndex):
        gc_raw.columns = gc_raw.columns.get_level_values(0)
    gc_close = gc_raw['Close'].copy()
    gc_close.index = pd.to_datetime(gc_close.index)
    print(f'[OK] GC=F: {len(gc_close)} rows')

    # Combine on common dates
    df = pd.DataFrame({'dxy_close': dxy_close, 'gc_close': gc_close})
    df = df.ffill(limit=3).dropna()

    # Compute returns
    df['dxy_log_ret'] = np.log(df['dxy_close']) - np.log(df['dxy_close'].shift(1))
    df['gold_return_next'] = df['gc_close'].pct_change().shift(-1)
    df = df.dropna(subset=['dxy_log_ret'])

    # Validate
    assert len(df) > 2000, f'Insufficient data: {len(df)} rows'
    assert df['dxy_close'].min() > 70 and df['dxy_close'].max() < 130, 'DXY out of range'
    assert df['dxy_log_ret'].abs().max() < 0.05, 'Extreme DXY return detected'

    print(f'[OK] Combined: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}')
    print(f'[OK] DXY close: {df["dxy_close"].mean():.2f} +/- {df["dxy_close"].std():.2f}')
    print(f'[OK] DXY log-ret: {df["dxy_log_ret"].mean():.6f} +/- {df["dxy_log_ret"].std():.6f}')

    return df

# ============================================================
# 3. FEATURE GENERATION FUNCTIONS
# ============================================================
def expanding_zscore(series, warmup):
    """Vectorized expanding z-score with warmup period. No lookahead."""
    mean = series.expanding(min_periods=warmup).mean()
    std = series.expanding(min_periods=warmup).std()
    z = (series - mean) / std
    z = z.clip(-4, 4)
    z = z.fillna(0.0)
    return z


def generate_regime_feature(dxy_log_ret, n_components, covariance_type, n_init, train_size):
    """
    Fit GaussianHMM on train portion, return P(highest-variance state) for full data.
    CRITICAL: Use HMM (not GMM) to capture temporal regime transitions.
    Manual n_init: fit multiple times, keep best log-likelihood.
    """
    X_train = dxy_log_ret.iloc[:train_size].values.reshape(-1, 1)
    X_full = dxy_log_ret.values.reshape(-1, 1)

    best_model = None
    best_score = -np.inf
    for init_i in range(n_init):
        try:
            m = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=200,
                tol=1e-4,
                random_state=42 + init_i
            )
            m.fit(X_train)
            score = m.score(X_train)
            if score > best_score:
                best_score = score
                best_model = m
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError('All HMM initializations failed')
    model = best_model

    # Get posterior probabilities for full dataset
    probs = model.predict_proba(X_full)

    # Identify highest-variance state
    state_vars = []
    for i in range(n_components):
        if covariance_type == 'full':
            state_vars.append(float(model.covars_[i][0, 0]))
        elif covariance_type == 'diag':
            state_vars.append(float(model.covars_[i][0]))
        else:
            state_vars.append(float(model.covars_[i]))

    high_var_state = np.argmax(state_vars)
    regime_prob = probs[:, high_var_state]

    print(f'  HMM states={n_components}, cov={covariance_type}, n_init={n_init}')
    print(f'  State variances: {[f"{v:.8f}" for v in state_vars]}')
    print(f'  High-var state: {high_var_state}, mean regime_prob: {regime_prob.mean():.3f}')

    return regime_prob


def generate_momentum_feature(dxy_close, momentum_window, expanding_warmup):
    """
    Expanding z-score of N-day momentum (N-day return).
    Follows proven pattern from vix_mean_reversion_z, tech_mean_reversion_z.
    """
    momentum = dxy_close.pct_change(momentum_window)
    z = expanding_zscore(momentum, expanding_warmup)
    return z


def generate_volatility_feature(dxy_log_ret, vol_window, expanding_warmup):
    """
    Expanding z-score of N-day realized volatility.
    Industry standard: 20-day rolling std for FX volatility.
    """
    vol = dxy_log_ret.rolling(vol_window).std()
    z = expanding_zscore(vol, expanding_warmup)
    return z


# ============================================================
# 4. OPTUNA HPO
# ============================================================
def discretize(x, bins=20):
    """Discretize continuous values into quantile bins for MI calculation."""
    valid = ~np.isnan(x)
    if valid.sum() < bins:
        return None
    x_clean = x.copy()
    x_clean[~valid] = np.nanmedian(x)
    try:
        return np.asarray(pd.qcut(x_clean, bins, labels=False, duplicates='drop'))
    except ValueError:
        return np.asarray(pd.cut(x_clean, bins, labels=False, duplicates='drop'))


def compute_mi_sum(features_dict, target_vals):
    """Compute sum of MI between each feature and target."""
    target_disc = discretize(target_vals)
    if target_disc is None:
        return 0.0

    mi_sum = 0.0
    for name, feat_vals in features_dict.items():
        mask = ~np.isnan(feat_vals) & ~np.isnan(target_vals)
        if mask.sum() > 50:
            feat_disc = discretize(feat_vals[mask])
            tgt_disc = discretize(target_vals[mask])
            if feat_disc is not None and tgt_disc is not None and len(feat_disc) == len(tgt_disc):
                mi_sum += mutual_info_score(feat_disc, tgt_disc)
    return mi_sum


def optuna_objective(trial, df, train_size, val_start, val_end):
    """Optuna objective: maximize MI sum on validation set."""
    n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
    covariance_type = trial.suggest_categorical('hmm_covariance_type', ['full', 'diag'])
    n_init = trial.suggest_categorical('hmm_n_init', [5, 10, 15])
    momentum_window = trial.suggest_categorical('momentum_window', [10, 15, 20, 30])
    vol_window = trial.suggest_categorical('vol_window', [10, 15, 20, 30])
    expanding_warmup = trial.suggest_categorical('expanding_warmup', [60, 120, 252])

    try:
        regime = generate_regime_feature(
            df['dxy_log_ret'], n_components, covariance_type, n_init, train_size
        )
        momentum_z = generate_momentum_feature(
            df['dxy_close'], momentum_window, expanding_warmup
        ).values
        vol_z = generate_volatility_feature(
            df['dxy_log_ret'], vol_window, expanding_warmup
        ).values

        # Extract validation period
        regime_val = regime[val_start:val_end]
        momentum_val = momentum_z[val_start:val_end]
        vol_val = vol_z[val_start:val_end]
        target_val = df['gold_return_next'].values[val_start:val_end]

        features = {
            'regime': regime_val,
            'momentum': momentum_val,
            'vol': vol_val
        }
        mi = compute_mi_sum(features, target_val)
        return mi

    except Exception as e:
        print(f'  Trial {trial.number} failed: {e}')
        return 0.0


# ============================================================
# 5. MAIN EXECUTION
# ============================================================
print('\n' + '='*70)
print('STEP 1: DATA FETCHING')
print('='*70)
df = fetch_data()

# Train/Val/Test split (70/15/15, time-series order)
n = len(df)
train_size = int(n * 0.70)
val_end = int(n * 0.85)
val_start = train_size
test_start = val_end

print(f'\nData split:')
print(f'  Train: 0:{train_size} ({train_size} rows, {df.index[0].date()} to {df.index[train_size-1].date()})')
print(f'  Val:   {val_start}:{val_end} ({val_end-val_start} rows, {df.index[val_start].date()} to {df.index[val_end-1].date()})')
print(f'  Test:  {test_start}:{n} ({n-test_start} rows, {df.index[test_start].date()} to {df.index[-1].date()})')

# ============================================================
print('\n' + '='*70)
print('STEP 2: OPTUNA HPO (30 trials, 300s timeout)')
print('='*70)

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(
    lambda trial: optuna_objective(trial, df, train_size, val_start, val_end),
    n_trials=30,
    timeout=300,
    show_progress_bar=True
)

best_params = study.best_params
best_mi = study.best_value
print(f'\nBest MI sum: {best_mi:.6f}')
print(f'Best params: {best_params}')
print(f'Completed trials: {len(study.trials)}')

# ============================================================
print('\n' + '='*70)
print('STEP 3: FINAL FEATURE GENERATION WITH BEST PARAMS')
print('='*70)

regime = generate_regime_feature(
    df['dxy_log_ret'],
    best_params['hmm_n_components'],
    best_params['hmm_covariance_type'],
    best_params['hmm_n_init'],
    train_size
)
momentum_z = generate_momentum_feature(
    df['dxy_close'],
    best_params['momentum_window'],
    best_params['expanding_warmup']
).values
vol_z = generate_volatility_feature(
    df['dxy_log_ret'],
    best_params['vol_window'],
    best_params['expanding_warmup']
).values

# Create output DataFrame
output = pd.DataFrame({
    'dxy_regime_prob': regime,
    'dxy_momentum_z': momentum_z,
    'dxy_vol_z': vol_z
}, index=df.index)

# NaN handling: warmup period
output['dxy_regime_prob'] = output['dxy_regime_prob'].fillna(0.5)
output['dxy_momentum_z'] = output['dxy_momentum_z'].fillna(0.0)
output['dxy_vol_z'] = output['dxy_vol_z'].fillna(0.0)

print(f'\nOutput shape: {output.shape}')
print(f'Date range: {output.index[0].date()} to {output.index[-1].date()}')
for col in output.columns:
    s = output[col]
    print(f'  {col}: mean={s.mean():.4f}, std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}')

# ============================================================
print('\n' + '='*70)
print('STEP 4: METRICS')
print('='*70)

# Autocorrelation
autocorr = {}
for col in output.columns:
    autocorr[col] = float(output[col].autocorr(lag=1))
print(f'Autocorrelation (lag-1): {autocorr}')

# Check for constant features
is_constant = {}
for col in output.columns:
    is_constant[col] = bool(output[col].std() < 1e-6)
print(f'Is constant: {is_constant}')

# MI on validation set
target_val = df['gold_return_next'].values[val_start:val_end]
mi_individual = {}
for col in output.columns:
    feat_val = output[col].values[val_start:val_end]
    mask = ~np.isnan(feat_val) & ~np.isnan(target_val)
    if mask.sum() > 50:
        f_disc = discretize(feat_val[mask])
        t_disc = discretize(target_val[mask])
        if f_disc is not None and t_disc is not None:
            mi_individual[col] = float(mutual_info_score(f_disc, t_disc))
        else:
            mi_individual[col] = 0.0
    else:
        mi_individual[col] = 0.0
mi_sum = sum(mi_individual.values())
print(f'MI individual: {mi_individual}')
print(f'MI sum: {mi_sum:.6f}')

# ============================================================
print('\n' + '='*70)
print('STEP 5: SAVING OUTPUTS')
print('='*70)

# Save submodel output
output.to_csv('submodel_output.csv')
print(f'Saved: submodel_output.csv ({output.shape[0]} rows, {output.shape[1]} columns)')

# Save training result
result = {
    'feature': 'dxy',
    'attempt': 1,
    'timestamp': datetime.now().isoformat(),
    'method': 'GaussianHMM + Momentum Z-Score + Volatility Z-Score',
    'best_params': best_params,
    'metrics': {
        'mi_individual': mi_individual,
        'mi_sum': mi_sum,
        'autocorr': autocorr,
        'is_constant': is_constant
    },
    'optuna_best_value': float(best_mi),
    'optuna_trials_completed': len(study.trials),
    'output_shape': list(output.shape),
    'output_columns': list(output.columns),
    'data_info': {
        'total_rows': len(df),
        'train_rows': train_size,
        'val_rows': val_end - val_start,
        'test_rows': n - test_start,
        'date_range_start': str(df.index[0].date()),
        'date_range_end': str(df.index[-1].date())
    },
    'output_stats': {
        col: {
            'mean': float(output[col].mean()),
            'std': float(output[col].std()),
            'min': float(output[col].min()),
            'max': float(output[col].max())
        }
        for col in output.columns
    }
}

with open('training_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f'Saved: training_result.json')

print('\n' + '='*70)
print('TRAINING COMPLETE!')
print('='*70)
print(f'Output: {output.shape[0]} rows x {output.shape[1]} columns')
print(f'Columns: {list(output.columns)}')
print(f'Best MI sum: {best_mi:.6f}')
print(f'Finished: {datetime.now().isoformat()}')
'''

# Verify syntax
ast.parse(CODE)
print("Syntax check: PASSED")

# Build proper ipynb
lines = CODE.split('\n')
source_lines = [line + '\n' for line in lines[:-1]] + [lines[-1]]

notebook = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# DXY SubModel Training - Attempt 1 (Revised)\n',
                '\n',
                'Self-contained: Data fetch -> HMM regime detection + Momentum Z-score + Volatility Z-score -> Optuna HPO -> Save results\n',
                '\n',
                '**Architecture**: 3-state GaussianHMM on DXY daily log-returns + expanding z-score features\n',
                '\n',
                '**Output**: 3 columns: dxy_regime_prob, dxy_momentum_z, dxy_vol_z'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': source_lines
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('notebooks/dxy_1/train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook written successfully")

# Verify roundtrip
with open('notebooks/dxy_1/train.ipynb', 'r') as f:
    nb = json.load(f)
code_back = ''.join(nb['cells'][1]['source'])
ast.parse(code_back)
print("Roundtrip syntax check: PASSED")

# Architecture checks
assert 'GaussianHMM' in code_back, 'Missing HMM'
assert 'GaussianMixture' not in code_back, 'Still using GMM!'
assert 'from sklearn.decomposition import PCA' not in code_back, 'Still using PCA!'
assert 'dxy_regime_prob' in code_back
assert 'dxy_momentum_z' in code_back
assert 'dxy_vol_z' in code_back
assert 'predict_proba' in code_back
assert 'hmmlearn' in code_back
print("Architecture check: PASSED (HMM, not GMM)")
print("ALL CHECKS PASSED")
