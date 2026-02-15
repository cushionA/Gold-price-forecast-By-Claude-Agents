"""
Gold Prediction SubModel Training - ETF Flow Attempt 1
Self-contained: Data fetch -> Preprocessing -> Training -> Evaluation -> Save results

Architecture: 2D HMM (3 states) on [log_volume_ratio, gold_return]
              + Dollar volume z-score (60d window)
              + Price-volume divergence z-score (5d corr vs 60d baseline)

Output: 3 features
  - etf_regime_prob: Probability of accumulation state (HMM Component 1)
  - etf_capital_intensity: Dollar volume z-score (Component 2)
  - etf_pv_divergence: Price-volume correlation divergence z-score (Component 3)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# === 1. Data Fetching ===
def fetch_and_preprocess():
    """Self-contained. Fetches GLD and GC=F data, computes derived features.
    Returns: (train_df, val_df, test_df, full_df)
    """
    # --- Yahoo Finance ---
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "yfinance"], check=True)
        import yfinance as yf

    # Fetch GLD OHLCV data (start early for warmup buffer)
    gld = yf.download('GLD', start='2014-10-01', progress=False)

    if gld.empty:
        raise RuntimeError("Failed to fetch GLD data from Yahoo Finance")

    # Flatten MultiIndex columns if present
    if isinstance(gld.columns, pd.MultiIndex):
        gld.columns = gld.columns.get_level_values(0)

    # Extract needed columns
    df = pd.DataFrame({
        'gld_close': gld['Close'],
        'gld_volume': gld['Volume']
    })
    df.index = pd.to_datetime(df.index)

    # Fetch GC=F (Gold Futures) for gold returns
    gc = yf.download('GC=F', start='2014-10-01', progress=False)

    if gc.empty:
        raise RuntimeError("Failed to fetch GC=F data from Yahoo Finance")

    # Flatten MultiIndex columns if present
    if isinstance(gc.columns, pd.MultiIndex):
        gc.columns = gc.columns.get_level_values(0)

    gc_close = gc['Close']
    gc_close.index = pd.to_datetime(gc_close.index)

    # Align dates (join on common trading days)
    df = df.join(gc_close.rename('gc_close'), how='inner')

    # Forward-fill gaps up to 3 trading days
    df = df.ffill(limit=3)

    # Drop any remaining NaN
    df = df.dropna()

    # Compute derived features
    # 1. Returns (for GLD)
    df['gld_returns'] = df['gld_close'].pct_change()

    # 2. Gold returns (from GC=F)
    df['gold_return'] = df['gc_close'].pct_change()

    # 3. Dollar volume
    df['dollar_volume'] = df['gld_close'] * df['gld_volume']

    # 4. Volume MA20 (for log volume ratio)
    df['volume_ma20'] = df['gld_volume'].rolling(20).mean()

    # 5. Log volume ratio
    df['log_volume_ratio'] = np.log(df['gld_volume'] / df['volume_ma20'])

    # 6. Volume changes (percentage)
    df['vol_changes'] = df['gld_volume'].pct_change()

    # Drop rows with NaN (first ~20 rows from rolling operations)
    df = df.dropna()

    # Basic validation
    if len(df) < 2000:
        raise RuntimeError(f"Insufficient data: only {len(df)} rows after preprocessing")

    if df['gld_volume'].min() <= 0:
        raise RuntimeError("Invalid data: volume contains non-positive values")

    if not (80 <= df['gld_close'].min() <= df['gld_close'].max() <= 600):
        raise RuntimeError(f"GLD close price out of expected range: {df['gld_close'].min():.2f} to {df['gld_close'].max():.2f}")

    # Check for extreme outliers in log_volume_ratio
    extreme_log_vol = df['log_volume_ratio'].abs() > 3
    if extreme_log_vol.sum() > len(df) * 0.05:  # More than 5% extreme values
        print(f"Warning: {extreme_log_vol.sum()} extreme log_volume_ratio values detected")

    # Split into train/val/test (70/15/15, time-series order)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Data fetched successfully:")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    print(f"  GLD close range: ${df['gld_close'].min():.2f} to ${df['gld_close'].max():.2f}")
    print(f"  Average daily volume: {df['gld_volume'].mean():.0f}")

    return train_df, val_df, test_df, df


# === 2. HMM Component ===
def train_hmm_regime(train_df, n_components=3, n_restarts=10):
    """Train HMM on [log_volume_ratio, gold_return] to detect flow regimes.

    Returns: fitted HMM model
    """
    try:
        from hmmlearn import hmm as hmmlearn_hmm
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "hmmlearn"], check=True)
        from hmmlearn import hmm as hmmlearn_hmm

    # Prepare 2D input
    X_train = train_df[['log_volume_ratio', 'gold_return']].values

    # Remove any remaining NaN or inf
    valid_mask = np.isfinite(X_train).all(axis=1)
    X_train_clean = X_train[valid_mask]

    if len(X_train_clean) < 100:
        raise RuntimeError(f"Insufficient valid data for HMM training: {len(X_train_clean)} rows")

    # Train GaussianHMM with multiple restarts
    best_model = None
    best_score = -np.inf

    for restart in range(n_restarts):
        model = hmmlearn_hmm.GaussianHMM(
            n_components=n_components,
            covariance_type='full',
            n_iter=200,
            random_state=42 + restart,
            init_params='stmc'
        )

        try:
            model.fit(X_train_clean)
            score = model.score(X_train_clean)

            if score > best_score:
                best_score = score
                best_model = model

        except Exception as e:
            print(f"HMM restart {restart} failed: {e}")
            continue

    if best_model is None:
        raise RuntimeError("All HMM training attempts failed")

    print(f"HMM training complete: {n_components} components, best log-likelihood={best_score:.2f}")

    return best_model


def compute_regime_probabilities(hmm_model, df):
    """Compute regime probabilities for all data.

    Returns: DataFrame with regime probabilities
    """
    X = df[['log_volume_ratio', 'gold_return']].values

    # Handle NaN/inf
    valid_mask = np.isfinite(X).all(axis=1)

    # Initialize output
    probs = np.full((len(df), hmm_model.n_components), np.nan)

    if valid_mask.sum() > 0:
        probs[valid_mask] = hmm_model.predict_proba(X[valid_mask])

    # Forward-fill NaN values
    probs_df = pd.DataFrame(probs, index=df.index)
    probs_df = probs_df.fillna(method='ffill').fillna(method='bfill')

    return probs_df.values


def identify_accumulation_state(hmm_model, train_df):
    """Identify which HMM state corresponds to accumulation (positive gold returns).

    Returns: state index (0, 1, or 2)
    """
    # Get state sequence for training data
    X_train = train_df[['log_volume_ratio', 'gold_return']].values
    valid_mask = np.isfinite(X_train).all(axis=1)
    X_train_clean = X_train[valid_mask]

    states = hmm_model.predict(X_train_clean)
    gold_returns_clean = train_df['gold_return'].values[valid_mask]

    # Compute mean gold return for each state
    state_returns = {}
    for state_id in range(hmm_model.n_components):
        state_mask = (states == state_id)
        if state_mask.sum() > 0:
            state_returns[state_id] = gold_returns_clean[state_mask].mean()
        else:
            state_returns[state_id] = -np.inf

    # Accumulation state = state with highest average gold return
    accumulation_state = max(state_returns, key=state_returns.get)

    print(f"State returns: {state_returns}")
    print(f"Accumulation state identified: {accumulation_state} (mean return: {state_returns[accumulation_state]:.4f})")

    return accumulation_state


# === 3. Deterministic Components ===
def compute_capital_intensity(df, window=60):
    """Compute dollar volume z-score (rolling 60-day window).

    Returns: Series of z-scores
    """
    dollar_vol = df['dollar_volume']

    # Rolling mean and std
    rolling_mean = dollar_vol.rolling(window).mean()
    rolling_std = dollar_vol.rolling(window).std()

    # Z-score
    z_score = (dollar_vol - rolling_mean) / rolling_std

    # Replace inf/nan with 0
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)

    return z_score


def compute_pv_divergence(df, corr_window=5, baseline_window=60):
    """Compute price-volume divergence: 5-day PV correlation z-scored vs 60-day baseline.

    Returns: Series of z-scores
    """
    # Rolling 5-day correlation between returns and volume changes
    rolling_corr = df['gld_returns'].rolling(corr_window).corr(df['vol_changes'])

    # Rolling 60-day mean and std of the 5-day correlation
    corr_mean = rolling_corr.rolling(baseline_window).mean()
    corr_std = rolling_corr.rolling(baseline_window).std()

    # Z-score
    z_score = (rolling_corr - corr_mean) / corr_std

    # Replace inf/nan with 0
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)

    return z_score


# === 4. Hyperparameter Optimization ===
def run_hpo(train_df, val_df, n_trials=20):
    """Run Optuna HPO for HMM hyperparameters.

    Returns: best_params dict
    """
    try:
        import optuna
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "optuna"], check=True)
        import optuna

    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "scikit-learn"], check=True)
        from sklearn.feature_selection import mutual_info_regression

    # Prepare validation target (next-day gold return)
    val_target = val_df['gold_return'].shift(-1).dropna()

    def objective(trial):
        # Sample hyperparameters
        n_components = trial.suggest_int('n_components', 2, 4)
        n_restarts = trial.suggest_int('n_restarts', 5, 15)
        capital_window = trial.suggest_int('capital_window', 40, 90, step=10)
        pv_corr_window = trial.suggest_int('pv_corr_window', 3, 10)
        pv_baseline_window = trial.suggest_int('pv_baseline_window', 40, 90, step=10)

        try:
            # Train HMM
            hmm_model = train_hmm_regime(train_df, n_components, n_restarts)
            accumulation_state = identify_accumulation_state(hmm_model, train_df)

            # Generate features for validation set
            regime_probs = compute_regime_probabilities(hmm_model, val_df)
            regime_prob = regime_probs[:, accumulation_state]

            capital_intensity = compute_capital_intensity(val_df, capital_window).values
            pv_divergence = compute_pv_divergence(val_df, pv_corr_window, pv_baseline_window).values

            # Stack features
            X_val = np.column_stack([regime_prob, capital_intensity, pv_divergence])

            # Align with target (drop last row since target is shifted)
            X_val_aligned = X_val[:-1]

            if len(X_val_aligned) != len(val_target):
                return -np.inf

            # Compute mutual information
            mi_scores = mutual_info_regression(X_val_aligned, val_target, random_state=42)
            mi_total = mi_scores.sum()

            # Check for NaN/constant features
            if np.isnan(mi_total) or np.any(np.isnan(X_val_aligned)):
                return -np.inf

            return mi_total

        except Exception as e:
            print(f"Trial failed: {e}")
            return -np.inf

    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptuna optimization complete:")
    print(f"  Best MI: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params, study.best_value


# === 5. Feature Generation ===
def generate_features(train_df, val_df, test_df, full_df, params):
    """Generate final 3-feature output using best hyperparameters.

    Returns: DataFrame with 3 columns
    """
    # Train HMM on full training data
    hmm_model = train_hmm_regime(train_df, params['n_components'], params['n_restarts'])
    accumulation_state = identify_accumulation_state(hmm_model, train_df)

    # Generate regime probabilities for all data
    regime_probs = compute_regime_probabilities(hmm_model, full_df)
    regime_prob = regime_probs[:, accumulation_state]

    # Generate deterministic features
    capital_intensity = compute_capital_intensity(full_df, params['capital_window']).values
    pv_divergence = compute_pv_divergence(full_df, params['pv_corr_window'], params['pv_baseline_window']).values

    # Create output DataFrame
    output = pd.DataFrame({
        'etf_regime_prob': regime_prob,
        'etf_capital_intensity': capital_intensity,
        'etf_pv_divergence': pv_divergence
    }, index=full_df.index)

    print(f"\nFeature generation complete:")
    print(f"  Output shape: {output.shape}")
    print(f"  Columns: {list(output.columns)}")
    print(f"  Date range: {output.index.min().date()} to {output.index.max().date()}")

    return output


# === 6. Evaluation Metrics ===
def compute_metrics(train_df, val_df, test_df, full_df, output, params):
    """Compute evaluation metrics for Gate 1 & 2.

    Returns: metrics dict
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "scikit-learn"], check=True)
        from sklearn.feature_selection import mutual_info_regression

    # Prepare target (next-day gold return)
    target = full_df['gold_return'].shift(-1).dropna()
    X_aligned = output.iloc[:-1].values  # Drop last row to align with shifted target

    # 1. Mutual Information (individual and sum)
    mi_scores = mutual_info_regression(X_aligned, target, random_state=42)
    mi_individual = dict(zip(output.columns, mi_scores))
    mi_sum = mi_scores.sum()

    # 2. Autocorrelation (for Gate 1)
    autocorr = {}
    for col in output.columns:
        series = output[col].dropna()
        if len(series) > 1:
            autocorr[col] = series.autocorr(lag=1)
        else:
            autocorr[col] = np.nan

    # 3. Check for constant/NaN features
    has_nan = output.isnull().any().to_dict()
    is_constant = (output.std() < 1e-6).to_dict()

    metrics = {
        'mi_individual': mi_individual,
        'mi_sum': mi_sum,
        'autocorr': autocorr,
        'has_nan': has_nan,
        'is_constant': is_constant,
        'output_stats': {
            'mean': output.mean().to_dict(),
            'std': output.std().to_dict(),
            'min': output.min().to_dict(),
            'max': output.max().to_dict()
        }
    }

    print(f"\nMetrics computed:")
    print(f"  MI individual: {mi_individual}")
    print(f"  MI sum: {mi_sum:.6f}")
    print(f"  Autocorrelation: {autocorr}")

    return metrics


# === 7. Main Execution ===
if __name__ == "__main__":
    print("=" * 80)
    print("ETF Flow SubModel Training - Attempt 1")
    print("=" * 80)

    # Step 1: Fetch data
    print("\n[1/5] Fetching and preprocessing data...")
    train_df, val_df, test_df, full_df = fetch_and_preprocess()

    # Step 2: Hyperparameter optimization
    print("\n[2/5] Running Optuna hyperparameter optimization...")
    best_params, best_mi = run_hpo(train_df, val_df, n_trials=20)

    # Step 3: Generate features with best params
    print("\n[3/5] Generating features with best hyperparameters...")
    output = generate_features(train_df, val_df, test_df, full_df, best_params)

    # Step 4: Compute metrics
    print("\n[4/5] Computing evaluation metrics...")
    metrics = compute_metrics(train_df, val_df, test_df, full_df, output, best_params)

    # Step 5: Save results
    print("\n[5/5] Saving results...")

    # Save output CSV
    output.to_csv('submodel_output.csv')
    print(f"  Saved: submodel_output.csv ({output.shape[0]} rows, {output.shape[1]} columns)")

    # Save training result JSON
    result = {
        'feature': 'etf_flow',
        'attempt': 1,
        'timestamp': datetime.now().isoformat(),
        'best_params': best_params,
        'metrics': {
            'mi_individual': metrics['mi_individual'],
            'mi_sum': metrics['mi_sum'],
            'autocorr': metrics['autocorr'],
            'has_nan': metrics['has_nan'],
            'is_constant': metrics['is_constant']
        },
        'optuna_best_value': best_mi,
        'output_shape': list(output.shape),
        'output_columns': list(output.columns),
        'data_info': {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df)
        }
    }

    with open('training_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: training_result.json")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nFinal MI: {metrics['mi_sum']:.6f}")
    print(f"Output shape: {output.shape}")
    print(f"Columns: {list(output.columns)}")
