import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)

print(f"XGBoost version: {xgb.__version__}")
print(f"Optuna version: {optuna.__version__}")
print(f"Started: {datetime.now().isoformat()}")
FEATURE_COLUMNS = [
    # Base features (5)
    'real_rate_change',
    'dxy_change',
    'vix',
    'yield_spread_change',
    'inflation_exp_change',
    # VIX submodel (3)
    'vix_regime_probability',
    'vix_mean_reversion_z',
    'vix_persistence',
    # Technical submodel (3)
    'tech_trend_regime_prob',
    'tech_mean_reversion_z',
    'tech_volatility_regime',
    # Cross-asset submodel (3)
    'xasset_regime_prob',
    'xasset_recession_signal',
    'xasset_divergence',
    # Yield curve submodel (2)
    'yc_spread_velocity_z',
    'yc_curvature_z',
    # ETF flow submodel (3)
    'etf_regime_prob',
    'etf_capital_intensity',
    'etf_pv_divergence',
    # Inflation expectation submodel (3)
    'ie_regime_prob',
    'ie_anchoring_z',
    'ie_gold_sensitivity_z',
    # Options market submodel (1) -- NEW IN ATTEMPT 5
    'options_risk_regime_prob',
]

TARGET = 'gold_return_next'

assert len(FEATURE_COLUMNS) == 23, f"Expected 23 features, got {len(FEATURE_COLUMNS)}"
print(f"Features defined: {len(FEATURE_COLUMNS)} features")
# ============================================================
# API-BASED DATA FETCHING
# ============================================================
print("="*60)
print("FETCHING DATA FROM APIs")
print("="*60)

# === Import libraries ===
import yfinance as yf

# FRED API (install if needed)
try:
    from fredapi import Fred
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "fredapi"], check=True)
    from fredapi import Fred

# === FRED API key (hardcoded) ===
FRED_API_KEY = "3ffb68facdf6321e180e380c00e909c8"
fred = Fred(api_key=FRED_API_KEY)
print("✓ FRED API initialized")

# === 1. Fetch Gold Price (target) ===
print("\nFetching gold price (GC=F)...")
gold = yf.download('GC=F', start='2014-01-01', end='2026-02-20', progress=False)
gold_df = gold[['Close']].copy()
gold_df.columns = ['gold_price']
gold_df['gold_return'] = gold_df['gold_price'].pct_change() * 100
gold_df['gold_return_next'] = gold_df['gold_return'].shift(-1)  # Next-day return
gold_df = gold_df.dropna(subset=['gold_return_next'])
gold_df.index = pd.to_datetime(gold_df.index).strftime('%Y-%m-%d')
print(f"  Gold: {len(gold_df)} rows")

# === 2. Fetch Base Features ===
print("\nFetching base features...")

# Real Rate (DFII10)
print("  Fetching real rate (DFII10)...")
real_rate = fred.get_series('DFII10', observation_start='2014-01-01')
real_rate_df = real_rate.to_frame('real_rate_real_rate')
real_rate_df.index = pd.to_datetime(real_rate_df.index).strftime('%Y-%m-%d')

# DXY (DX-Y.NYB)
print("  Fetching DXY (DX-Y.NYB)...")
dxy = yf.download('DX-Y.NYB', start='2014-01-01', end='2026-02-20', progress=False)
dxy_df = dxy[['Close']].copy()
dxy_df.columns = ['dxy_dxy']
dxy_df.index = pd.to_datetime(dxy_df.index).strftime('%Y-%m-%d')

# VIX (VIXCLS)
print("  Fetching VIX (VIXCLS)...")
vix = fred.get_series('VIXCLS', observation_start='2014-01-01')
vix_df = vix.to_frame('vix_vix')
vix_df.index = pd.to_datetime(vix_df.index).strftime('%Y-%m-%d')

# Yield Curve (DGS10 - DGS2)
print("  Fetching yield curve (DGS10, DGS2)...")
dgs10 = fred.get_series('DGS10', observation_start='2014-01-01')
dgs2 = fred.get_series('DGS2', observation_start='2014-01-01')
yc_df = pd.DataFrame({'DGS10': dgs10, 'DGS2': dgs2})
yc_df['yield_curve_yield_spread'] = yc_df['DGS10'] - yc_df['DGS2']
yc_df = yc_df[['yield_curve_yield_spread']]
yc_df.index = pd.to_datetime(yc_df.index).strftime('%Y-%m-%d')

# Inflation Expectation (T10YIE)
print("  Fetching inflation expectation (T10YIE)...")
infl_exp = fred.get_series('T10YIE', observation_start='2014-01-01')
infl_exp_df = infl_exp.to_frame('inflation_expectation_inflation_expectation')
infl_exp_df.index = pd.to_datetime(infl_exp_df.index).strftime('%Y-%m-%d')

# Merge base features
base_features = gold_df[['gold_return_next']].copy()
for df in [real_rate_df, dxy_df, vix_df, yc_df, infl_exp_df]:
    base_features = base_features.join(df, how='left')

# Forward-fill missing values (weekends, holidays)
base_features = base_features.ffill()
print(f"  Base features: {len(base_features)} rows, {len(base_features.columns)} columns")

# === 3. Load Submodel Outputs (from local files) ===
print("\nLoading submodel outputs from local files...")

submodel_files = {
    'vix': {
        'path': 'data/submodel_outputs/vix.csv',
        'columns': ['vix_regime_probability', 'vix_mean_reversion_z', 'vix_persistence'],
        'date_col': 'date',
        'tz_aware': False,
    },
    'technical': {
        'path': 'data/submodel_outputs/technical.csv',
        'columns': ['tech_trend_regime_prob', 'tech_mean_reversion_z', 'tech_volatility_regime'],
        'date_col': 'date',
        'tz_aware': True,  # CRITICAL: timezone-aware dates
    },
    'cross_asset': {
        'path': 'data/submodel_outputs/cross_asset.csv',
        'columns': ['xasset_regime_prob', 'xasset_recession_signal', 'xasset_divergence'],
        'date_col': 'Date',
        'tz_aware': False,
    },
    'yield_curve': {
        'path': 'data/submodel_outputs/yield_curve.csv',
        'columns': ['yc_spread_velocity_z', 'yc_curvature_z'],
        'date_col': 'index',  # Special: rename from 'index'
        'tz_aware': False,
    },
    'etf_flow': {
        'path': 'data/submodel_outputs/etf_flow.csv',
        'columns': ['etf_regime_prob', 'etf_capital_intensity', 'etf_pv_divergence'],
        'date_col': 'Date',
        'tz_aware': False,
    },
    'inflation_expectation': {
        'path': 'data/submodel_outputs/inflation_expectation.csv',
        'columns': ['ie_regime_prob', 'ie_anchoring_z', 'ie_gold_sensitivity_z'],
        'date_col': 'Unnamed: 0',  # Special: rename from 'Unnamed: 0'
        'tz_aware': False,
    },
    'options_market': {  # NEW IN ATTEMPT 5
        'path': 'data/submodel_outputs/options_market.csv',
        'columns': ['options_risk_regime_prob'],
        'date_col': 'Date',
        'tz_aware': True,  # CRITICAL: timezone-aware dates (same as technical.csv)
    },
}

submodel_dfs = {}
for feature, spec in submodel_files.items():
    # Load CSV
    df = pd.read_csv(spec['path'])

    # Normalize date column
    date_col = spec['date_col']
    if spec['tz_aware']:
        # CRITICAL: timezone-aware dates require utc=True
        df['Date'] = pd.to_datetime(df[date_col], utc=True).dt.strftime('%Y-%m-%d')
    else:
        if date_col == 'index':
            df['Date'] = pd.to_datetime(df.index).dt.strftime('%Y-%m-%d')
        elif date_col == 'Unnamed: 0':
            df['Date'] = pd.to_datetime(df['Unnamed: 0']).dt.strftime('%Y-%m-%d')
        else:
            df['Date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')

    df = df[['Date'] + spec['columns']]
    df = df.set_index('Date')
    submodel_dfs[feature] = df
    print(f"  {feature}: {len(df)} rows")

print(f"\n✓ Data fetching complete")
# === Apply transformations (stationary conversion) ===
print("\nApplying transformations...")

# Create final feature DataFrame
final_df = base_features.copy()

# Base features (4 diff, 1 level)
final_df['real_rate_change'] = final_df['real_rate_real_rate'].diff()
final_df['dxy_change'] = final_df['dxy_dxy'].diff()
final_df['vix'] = final_df['vix_vix']  # Level (stationary)
final_df['yield_spread_change'] = final_df['yield_curve_yield_spread'].diff()
final_df['inflation_exp_change'] = final_df['inflation_expectation_inflation_expectation'].diff()

# Drop original raw columns
final_df = final_df.drop(columns=['real_rate_real_rate', 'dxy_dxy', 'vix_vix',
                                    'yield_curve_yield_spread', 'inflation_expectation_inflation_expectation'])

# === Merge submodel features ===
print("\nMerging submodel outputs...")
for feature, df in submodel_dfs.items():
    final_df = final_df.join(df, how='left')  # Left join preserves all base dates

print(f"  Features after merge: {final_df.shape[1]} columns, {len(final_df)} rows")

# === NaN Imputation (domain-specific) ===
print("\nApplying NaN imputation...")

nan_before = final_df.isna().sum().sum()
print(f"  NaN before imputation: {nan_before}")

# Regime probability columns → 0.5 (maximum uncertainty)
regime_cols = ['vix_regime_probability', 'tech_trend_regime_prob', 
               'xasset_regime_prob', 'etf_regime_prob', 'ie_regime_prob',
               'options_risk_regime_prob']  # NEW: include options_market
for col in regime_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(0.5)

# Z-score columns → 0.0 (at mean)
z_cols = ['vix_mean_reversion_z', 'tech_mean_reversion_z', 
          'yc_spread_velocity_z', 'yc_curvature_z',
          'etf_capital_intensity', 'etf_pv_divergence',
          'ie_anchoring_z', 'ie_gold_sensitivity_z']
for col in z_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(0.0)

# Divergence/signal columns → 0.0 (neutral)
div_cols = ['xasset_recession_signal', 'xasset_divergence']
for col in div_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(0.0)

# Continuous state columns → median
cont_cols = ['tech_volatility_regime', 'vix_persistence']
for col in cont_cols:
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna(final_df[col].median())

# Drop rows with NaN in target or base features (critical rows)
final_df = final_df.dropna(subset=['gold_return_next', 'real_rate_change', 'dxy_change', 
                                     'vix', 'yield_spread_change', 'inflation_exp_change'])

nan_after = final_df.isna().sum().sum()
print(f"  NaN after imputation: {nan_after}")
print(f"  Final dataset: {len(final_df)} rows")

# === Verify feature set ===
assert all(col in final_df.columns for col in FEATURE_COLUMNS), "Missing features after merge!"
assert TARGET in final_df.columns, "Target not found!"
print(f"\n✓ All {len(FEATURE_COLUMNS)} features present")
print(f"✓ Dataset shape: {final_df.shape}")
print(f"✓ Date range: {final_df.index.min()} to {final_df.index.max()}")
# === Train/Val/Test Split (70/15/15, time-series order) ===
n_total = len(final_df)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.15)

train_df = final_df.iloc[:n_train].copy()
val_df = final_df.iloc[n_train:n_train+n_val].copy()
test_df = final_df.iloc[n_train+n_val:].copy()

print(f"\n✓ Data split complete:")
print(f"  Train: {len(train_df)} rows ({len(train_df)/n_total*100:.1f}%) - {train_df.index.min()} to {train_df.index.max()}")
print(f"  Val:   {len(val_df)} rows ({len(val_df)/n_total*100:.1f}%) - {val_df.index.min()} to {val_df.index.max()}")
print(f"  Test:  {len(test_df)} rows ({len(test_df)/n_total*100:.1f}%) - {test_df.index.min()} to {test_df.index.max()}")
print(f"  Total: {n_total} rows")
print(f"  Samples per feature: {n_train / len(FEATURE_COLUMNS):.1f}:1 (train)")

# Verify no data leakage
assert train_df.index.max() < val_df.index.min(), "Train-val overlap detected!"
assert val_df.index.max() < test_df.index.min(), "Val-test overlap detected!"
print(f"\n✓ No time-series leakage detected")
print("="*60)

# Prepare arrays for training
X_train = train_df[FEATURE_COLUMNS].values
y_train = train_df[TARGET].values

X_val = val_df[FEATURE_COLUMNS].values
y_val = val_df[TARGET].values

X_test = test_df[FEATURE_COLUMNS].values
y_test = test_df[TARGET].values

# Store dates for output
dates_train = train_df.index
dates_val = val_df.index
dates_test = test_df.index

print(f"\nArray shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
def compute_direction_accuracy(y_true, y_pred):
    """Direction accuracy, excluding zeros."""
    mask = (y_true != 0) & (y_pred != 0)
    if mask.sum() == 0:
        return 0.0
    return (np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean()

def compute_mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.abs(y_pred - y_true).mean()

def compute_sharpe_trade_cost(y_true, y_pred, cost_bps=5.0):
    """Sharpe ratio with position-change cost (5bps per change)."""
    positions = np.sign(y_pred)
    
    # Strategy returns (position * actual return)
    strategy_returns = positions * y_true / 100.0  # Convert % to decimal
    
    # Position changes
    position_changes = np.abs(np.diff(positions, prepend=0))
    trade_costs = position_changes * (cost_bps / 10000.0)  # 5bps = 0.0005
    
    # Net returns
    net_returns = strategy_returns - trade_costs
    
    # Annualized Sharpe (252 trading days)
    if len(net_returns) < 2 or net_returns.std() == 0:
        return 0.0
    return (net_returns.mean() / net_returns.std()) * np.sqrt(252)

def compute_hcda(y_true, y_pred, threshold_percentile=80):
    """High-confidence direction accuracy (top 20% by |prediction|)."""
    threshold = np.percentile(np.abs(y_pred), threshold_percentile)
    hc_mask = np.abs(y_pred) > threshold
    
    if hc_mask.sum() == 0:
        return 0.0, 0.0
    
    coverage = hc_mask.sum() / len(y_pred)
    hc_pred = y_pred[hc_mask]
    hc_actual = y_true[hc_mask]
    
    mask = (hc_actual != 0) & (hc_pred != 0)
    if mask.sum() == 0:
        return 0.0, coverage
    
    da = (np.sign(hc_pred[mask]) == np.sign(hc_actual[mask])).mean()
    return da, coverage

print("Metric functions defined")
def optuna_objective(trial):
    """Optuna objective function with Attempt 5 specifications."""
    
    # === Sample hyperparameters (8 parameters) ===
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.7),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'seed': 42 + trial.number,
    }
    
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    
    # === Train model ===
    model = xgb.XGBRegressor(**params, n_estimators=n_estimators, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # === Predictions ===
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # === Compute metrics ===
    train_da = compute_direction_accuracy(y_train, train_pred)
    val_da = compute_direction_accuracy(y_val, val_pred)
    val_mae = compute_mae(y_val, val_pred)
    val_sharpe = compute_sharpe_trade_cost(y_val, val_pred)
    val_hc_da, val_hc_coverage = compute_hcda(y_val, val_pred, threshold_percentile=80)
    
    # === Overfitting penalty ===
    da_gap = (train_da - val_da) * 100  # In percentage points
    overfit_penalty = max(0.0, (da_gap - 10.0) / 30.0)  # 0 if gap<=10pp, up to 1.0 if gap=40pp
    
    # === Normalize metrics to [0, 1] ===
    sharpe_norm = np.clip((val_sharpe + 3.0) / 6.0, 0.0, 1.0)   # [-3, +3] -> [0, 1]
    da_norm = np.clip((val_da * 100 - 40.0) / 30.0, 0.0, 1.0)   # [40%, 70%] -> [0, 1]
    mae_norm = np.clip((1.0 - val_mae) / 0.5, 0.0, 1.0)         # [0.5%, 1.0%] -> [0, 1]
    hc_da_norm = np.clip((val_hc_da * 100 - 40.0) / 30.0, 0.0, 1.0)  # [40%, 70%] -> [0, 1]
    
    # === Weighted composite (ATTEMPT 5 WEIGHTS) ===
    objective = (
        0.40 * sharpe_norm +     # Reduced from 0.50 (Attempt 2)
        0.30 * da_norm +         # Unchanged
        0.10 * mae_norm +        # Unchanged
        0.20 * hc_da_norm        # Increased from 0.10 (Attempt 2)
    ) - 0.30 * overfit_penalty   # Same penalty as Attempt 2
    
    # === Log trial details ===
    trial.set_user_attr('val_da', float(val_da))
    trial.set_user_attr('val_mae', float(val_mae))
    trial.set_user_attr('val_sharpe', float(val_sharpe))
    trial.set_user_attr('val_hc_da', float(val_hc_da))
    trial.set_user_attr('val_hc_coverage', float(val_hc_coverage))
    trial.set_user_attr('train_da', float(train_da))
    trial.set_user_attr('da_gap_pp', float(da_gap))
    trial.set_user_attr('n_estimators_used',
                         int(model.best_iteration + 1) if hasattr(model, 'best_iteration')
                         and model.best_iteration is not None else n_estimators)
    
    return objective

print("Optuna objective function defined")
print("\n" + "="*60)
print("RUNNING OPTUNA HPO (100 trials, 2-hour timeout)")
print("="*60)

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
)

study.optimize(
    optuna_objective,
    n_trials=100,
    timeout=7200,  # 2 hours
    show_progress_bar=True
)

print(f"\nOptuna optimization complete")
print(f"  Trials completed: {len(study.trials)}")
print(f"  Best value: {study.best_value:.4f}")
print(f"\nBest hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

best_trial = study.best_trial
print(f"\nBest trial validation metrics:")
print(f"  DA:     {best_trial.user_attrs['val_da']*100:.2f}%")
print(f"  HCDA:   {best_trial.user_attrs['val_hc_da']*100:.2f}%")
print(f"  MAE:    {best_trial.user_attrs['val_mae']:.4f}%")
print(f"  Sharpe: {best_trial.user_attrs['val_sharpe']:.2f}")
print(f"  DA gap: {best_trial.user_attrs['da_gap_pp']:.2f}pp")
print("\n" + "="*60)
print("FALLBACK: Testing Attempt 2 Best Params on 23 Features")
print("="*60)

# Attempt 2 best params (from evaluation JSON)
FALLBACK_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 2,
    'min_child_weight': 14,
    'reg_lambda': 4.76,
    'reg_alpha': 3.65,
    'subsample': 0.478,
    'colsample_bytree': 0.371,
    'learning_rate': 0.025,
    'tree_method': 'hist',
    'eval_metric': 'rmse',
    'verbosity': 0,
    'seed': 42,
}
FALLBACK_N_ESTIMATORS = 247

print("Training fallback model...")
fallback_model = xgb.XGBRegressor(**FALLBACK_PARAMS, n_estimators=FALLBACK_N_ESTIMATORS, early_stopping_rounds=50)
fallback_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Predictions
fallback_train_pred = fallback_model.predict(X_train)
fallback_val_pred = fallback_model.predict(X_val)

# Metrics
fallback_train_da = compute_direction_accuracy(y_train, fallback_train_pred)
fallback_val_da = compute_direction_accuracy(y_val, fallback_val_pred)
fallback_val_mae = compute_mae(y_val, fallback_val_pred)
fallback_val_sharpe = compute_sharpe_trade_cost(y_val, fallback_val_pred)
fallback_val_hc_da, _ = compute_hcda(y_val, fallback_val_pred, threshold_percentile=80)
fallback_da_gap = (fallback_train_da - fallback_val_da) * 100

# Composite objective (same formula as Optuna)
sharpe_norm_fb = np.clip((fallback_val_sharpe + 3.0) / 6.0, 0.0, 1.0)
da_norm_fb = np.clip((fallback_val_da * 100 - 40.0) / 30.0, 0.0, 1.0)
mae_norm_fb = np.clip((1.0 - fallback_val_mae) / 0.5, 0.0, 1.0)
hc_da_norm_fb = np.clip((fallback_val_hc_da * 100 - 40.0) / 30.0, 0.0, 1.0)
overfit_penalty_fb = max(0.0, (fallback_da_gap - 10.0) / 30.0)
fallback_objective = (
    0.40 * sharpe_norm_fb + 0.30 * da_norm_fb + 0.10 * mae_norm_fb + 0.20 * hc_da_norm_fb
) - 0.30 * overfit_penalty_fb

print(f"\nFallback validation metrics:")
print(f"  DA:     {fallback_val_da*100:.2f}%")
print(f"  HCDA:   {fallback_val_hc_da*100:.2f}%")
print(f"  MAE:    {fallback_val_mae:.4f}%")
print(f"  Sharpe: {fallback_val_sharpe:.2f}")
print(f"  DA gap: {fallback_da_gap:.2f}pp")
print(f"  Composite objective: {fallback_objective:.4f}")

print(f"\nOptuna best vs Fallback:")
print(f"  Optuna objective: {study.best_value:.4f}")
print(f"  Fallback objective: {fallback_objective:.4f}")

# Select configuration
if study.best_value >= fallback_objective:
    print("\n✓ Using Optuna best configuration")
    selected_config = 'optuna'
    selected_params = study.best_params
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=selected_params['max_depth'],
        min_child_weight=selected_params['min_child_weight'],
        subsample=selected_params['subsample'],
        colsample_bytree=selected_params['colsample_bytree'],
        reg_lambda=selected_params['reg_lambda'],
        reg_alpha=selected_params['reg_alpha'],
        learning_rate=selected_params['learning_rate'],
        tree_method='hist',
        eval_metric='rmse',
        verbosity=0,
        seed=42,
        n_estimators=selected_params['n_estimators'],
        early_stopping_rounds=50
    )
else:
    print("\n✓ Using Attempt 2 fallback configuration")
    selected_config = 'fallback'
    selected_params = FALLBACK_PARAMS.copy()
    selected_params['n_estimators'] = FALLBACK_N_ESTIMATORS
    final_model = xgb.XGBRegressor(**FALLBACK_PARAMS, n_estimators=FALLBACK_N_ESTIMATORS, early_stopping_rounds=50)
print("\n" + "="*60)
print(f"TRAINING FINAL MODEL ({selected_config.upper()} CONFIG)")
print("="*60)

final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Generate predictions
pred_train = final_model.predict(X_train)
pred_val = final_model.predict(X_val)
pred_test = final_model.predict(X_test)

# Combine all predictions for full dataset
pred_full = np.concatenate([pred_train, pred_val, pred_test])
dates_full = pd.Index(list(dates_train) + list(dates_val) + list(dates_test))
y_full = np.concatenate([y_train, y_val, y_test])

print("\nPredictions generated:")
print(f"  Train: mean={pred_train.mean():.4f}, std={pred_train.std():.4f}")
print(f"  Val:   mean={pred_val.mean():.4f}, std={pred_val.std():.4f}")
print(f"  Test:  mean={pred_test.mean():.4f}, std={pred_test.std():.4f}")
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

# Compute metrics for all splits
metrics_all = {}
for split_name, y_true, y_pred in [
    ('train', y_train, pred_train),
    ('val', y_val, pred_val),
    ('test', y_test, pred_test),
]:
    da = compute_direction_accuracy(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    sharpe = compute_sharpe_trade_cost(y_true, y_pred)
    hc_da, hc_coverage = compute_hcda(y_true, y_pred, threshold_percentile=80)
    
    metrics_all[split_name] = {
        'direction_accuracy': float(da),
        'high_confidence_da': float(hc_da),
        'high_confidence_coverage': float(hc_coverage),
        'mae': float(mae),
        'sharpe_ratio': float(sharpe),
    }

# Print metrics
for split_name in ['train', 'val', 'test']:
    m = metrics_all[split_name]
    print(f"\n{split_name.upper()}:")
    print(f"  DA:     {m['direction_accuracy']*100:.2f}%")
    print(f"  HCDA:   {m['high_confidence_da']*100:.2f}% (coverage: {m['high_confidence_coverage']*100:.1f}%)")
    print(f"  MAE:    {m['mae']:.4f}%")
    print(f"  Sharpe: {m['sharpe_ratio']:.2f}")

# Overfitting analysis
train_test_da_gap = (metrics_all['train']['direction_accuracy'] - metrics_all['test']['direction_accuracy']) * 100
print(f"\nOVERFITTING:")
print(f"  Train-Test DA gap: {train_test_da_gap:.2f}pp (target: <10pp)")

# Target evaluation
test_m = metrics_all['test']
targets_met = [
    test_m['direction_accuracy'] > 0.56,
    test_m['high_confidence_da'] > 0.60,
    test_m['mae'] < 0.0075,
    test_m['sharpe_ratio'] > 0.8,
]

print(f"\nTARGET STATUS:")
print(f"  DA > 56%:     {'✓' if targets_met[0] else '✗'} ({test_m['direction_accuracy']*100:.2f}%)")
print(f"  HCDA > 60%:   {'✓' if targets_met[1] else '✗'} ({test_m['high_confidence_da']*100:.2f}%)")
print(f"  MAE < 0.75%:  {'✓' if targets_met[2] else '✗'} ({test_m['mae']:.4f}%)")
print(f"  Sharpe > 0.8: {'✓' if targets_met[3] else '✗'} ({test_m['sharpe_ratio']:.2f})")
print(f"\nTargets passed: {sum(targets_met)}/4")
print("\n" + "="*60)
print("DIAGNOSTIC ANALYSIS")
print("="*60)

# 1. HCDA at multiple thresholds
print("\nHCDA at different confidence thresholds (test set):")
for pct in [70, 75, 80, 85, 90]:
    hc_da, hc_cov = compute_hcda(y_test, pred_test, threshold_percentile=pct)
    n_samples = int(len(y_test) * hc_cov)
    print(f"  Top {100-pct}% (N={n_samples}): {hc_da*100:.2f}%")

# 2. Feature importance
feature_importance = final_model.feature_importances_
feature_ranking = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 10 Feature Importance:")
for i, row in feature_ranking.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Find options_risk_regime_prob rank
options_rank = (feature_ranking.reset_index(drop=True).reset_index()
                .loc[feature_ranking['feature'] == 'options_risk_regime_prob', 'index'].values[0] + 1)
options_importance = feature_ranking.loc[feature_ranking['feature'] == 'options_risk_regime_prob', 'importance'].values[0]
print(f"\noptions_risk_regime_prob: Rank {options_rank}/23, Importance {options_importance:.4f}")

# 3. Prediction distribution
print("\nPrediction distribution (test set):")
print(f"  Mean:     {pred_test.mean():.4f}%")
print(f"  Std:      {pred_test.std():.4f}%")
print(f"  Min:      {pred_test.min():.4f}%")
print(f"  Max:      {pred_test.max():.4f}%")
print(f"  Positive: {(pred_test > 0).sum() / len(pred_test) * 100:.1f}%")

# 4. Comparison with baselines
naive_always_up_da = (y_test > 0).sum() / len(y_test)
print(f"\nNaive Baseline:")
print(f"  Always-up DA: {naive_always_up_da*100:.2f}%")
print(f"  Model vs naive: {(test_m['direction_accuracy'] - naive_always_up_da)*100:+.2f}pp")

# 5. Attempt 2 comparison
print("\nVs Attempt 2:")
print(f"  DA:     {test_m['direction_accuracy']*100:.2f}% (Attempt 2: 57.26%, delta: {(test_m['direction_accuracy']-0.5726)*100:+.2f}pp)")
print(f"  HCDA:   {test_m['high_confidence_da']*100:.2f}% (Attempt 2: 55.26%, delta: {(test_m['high_confidence_da']-0.5526)*100:+.2f}pp)")
print(f"  MAE:    {test_m['mae']:.4f}% (Attempt 2: 0.6877%, delta: {(test_m['mae']-0.6877)*100:+.2f}pp)")
print(f"  Sharpe: {test_m['sharpe_ratio']:.2f} (Attempt 2: 1.58, delta: {test_m['sharpe_ratio']-1.5835:+.2f})")

# 6. Quarterly breakdown
test_df_with_pred = test_df.copy()
test_df_with_pred['prediction'] = pred_test
test_df_with_pred['quarter'] = pd.to_datetime(test_df_with_pred.index).to_period('Q')

print("\nQuarterly Performance (test set):")
for quarter in test_df_with_pred['quarter'].unique():
    qtr_data = test_df_with_pred[test_df_with_pred['quarter'] == quarter]
    qtr_da = compute_direction_accuracy(qtr_data[TARGET].values, qtr_data['prediction'].values)
    qtr_mae = compute_mae(qtr_data[TARGET].values, qtr_data['prediction'].values)
    qtr_sharpe = compute_sharpe_trade_cost(qtr_data[TARGET].values, qtr_data['prediction'].values)
    print(f"  {quarter}: DA={qtr_da*100:.1f}%, MAE={qtr_mae:.3f}%, Sharpe={qtr_sharpe:.2f}, N={len(qtr_data)}")
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# 1. predictions.csv (full dataset)
split_labels = ['train'] * len(dates_train) + ['val'] * len(dates_val) + ['test'] * len(dates_test)
predictions_df = pd.DataFrame({
    'date': dates_full,
    'split': split_labels,
    'actual': y_full,
    'prediction': pred_full,
    'direction_correct': (np.sign(pred_full) == np.sign(y_full)).astype(int),
    'abs_prediction': np.abs(pred_full),
})

# Add high_confidence flag (80th percentile)
threshold_80 = np.percentile(np.abs(pred_full), 80)
predictions_df['high_confidence'] = (predictions_df['abs_prediction'] > threshold_80).astype(int)

predictions_df.to_csv('predictions.csv', index=False)
print("✓ Saved predictions.csv")

# 2. test_predictions.csv (test set only)
test_predictions_df = predictions_df[predictions_df['split'] == 'test'].copy()
test_predictions_df.to_csv('test_predictions.csv', index=False)
print("✓ Saved test_predictions.csv")

# 3. submodel_output.csv (for pipeline compatibility)
predictions_df.to_csv('submodel_output.csv', index=False)
print("✓ Saved submodel_output.csv")

# 4. model.json (XGBoost model)
final_model.save_model('model.json')
print("✓ Saved model.json")

# 5. training_result.json
training_result = {
    'feature': 'meta_model',
    'attempt': 5,
    'timestamp': datetime.now().isoformat(),
    'architecture': 'XGBoost reg:squarederror',
    'phase': '3_meta_model',
    
    'model_config': {
        'n_features': 23,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'samples_per_feature_ratio': len(X_train) / 23,
        'selected_configuration': selected_config,
        'optuna_trials_completed': len(study.trials),
        'best_params': selected_params,
    },
    
    'optuna_search': {
        'n_trials': len(study.trials),
        'best_value': float(study.best_value),
        'best_trial_number': study.best_trial.number,
        'top_5_trials': [
            {
                'number': t.number,
                'value': float(t.value),
                'params': t.params,
                'val_da': float(t.user_attrs['val_da']),
                'val_hc_da': float(t.user_attrs['val_hc_da']),
            }
            for t in sorted(study.trials, key=lambda x: x.value, reverse=True)[:5]
        ],
    },
    
    'fallback_comparison': {
        'fallback_objective': float(fallback_objective),
        'optuna_objective': float(study.best_value),
        'selected': selected_config,
        'fallback_metrics': {
            'da': float(fallback_val_da),
            'hcda': float(fallback_val_hc_da),
            'mae': float(fallback_val_mae),
            'sharpe': float(fallback_val_sharpe),
        },
    },
    
    'metrics': metrics_all,
    
    'target_evaluation': {
        'direction_accuracy': {
            'target': '> 56.0%',
            'actual': f"{test_m['direction_accuracy']*100:.2f}%",
            'gap': f"{(test_m['direction_accuracy'] - 0.56)*100:+.2f}pp",
            'passed': bool(targets_met[0]),
        },
        'high_confidence_da': {
            'target': '> 60.0%',
            'actual': f"{test_m['high_confidence_da']*100:.2f}%",
            'gap': f"{(test_m['high_confidence_da'] - 0.60)*100:+.2f}pp",
            'passed': bool(targets_met[1]),
        },
        'mae': {
            'target': '< 0.75%',
            'actual': f"{test_m['mae']:.4f}%",
            'gap': f"{(0.0075 - test_m['mae']):.4f}%",
            'passed': bool(targets_met[2]),
        },
        'sharpe_ratio': {
            'target': '> 0.80',
            'actual': f"{test_m['sharpe_ratio']:.2f}",
            'gap': f"{(test_m['sharpe_ratio'] - 0.8):+.2f}",
            'passed': bool(targets_met[3]),
        },
    },
    
    'targets_passed': sum(targets_met),
    'targets_total': 4,
    'overall_passed': all(targets_met),
    
    'overfitting_analysis': {
        'train_test_da_gap_pp': float(train_test_da_gap),
        'target_gap_pp': 10.0,
        'overfitting_check': 'PASS' if train_test_da_gap < 10 else 'FAIL',
    },
    
    'feature_importance': {
        'top_10': feature_ranking.head(10).to_dict('records'),
        'options_risk_regime_prob_rank': int(options_rank),
        'options_risk_regime_prob_importance': float(options_importance),
    },
    
    'vs_attempt_2': {
        'da_delta_pp': float((test_m['direction_accuracy'] - 0.5726) * 100),
        'hcda_delta_pp': float((test_m['high_confidence_da'] - 0.5526) * 100),
        'mae_delta': float(test_m['mae'] - 0.6877),
        'sharpe_delta': float(test_m['sharpe_ratio'] - 1.5835),
    },
    
    'vs_naive': {
        'naive_always_up_da': f"{naive_always_up_da*100:.2f}%",
        'model_vs_naive_pp': float((test_m['direction_accuracy'] - naive_always_up_da) * 100),
    },
    
    'prediction_characteristics': {
        'mean': float(pred_test.mean()),
        'std': float(pred_test.std()),
        'min': float(pred_test.min()),
        'max': float(pred_test.max()),
        'positive_pct': float((pred_test > 0).sum() / len(pred_test) * 100),
    },
}

with open('training_result.json', 'w') as f:
    json.dump(training_result, f, indent=2, default=str)
print("✓ Saved training_result.json")

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Finished: {datetime.now().isoformat()}")
print(f"\nFinal Status:")
print(f"  Configuration: {selected_config.upper()}")
print(f"  Targets passed: {sum(targets_met)}/4")
if all(targets_met):
    print(f"  ✓✓✓ ALL TARGETS MET ✓✓✓")
else:
    failed = [t for t, m in zip(['DA', 'HCDA', 'MAE', 'Sharpe'], targets_met) if not m]
    print(f"  Improvements needed on: {failed}")
