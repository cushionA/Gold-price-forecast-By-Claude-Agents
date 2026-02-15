"""
Gold Prediction SubModel Training - Inflation Expectation Attempt 1
Self-contained: Data fetch -> Preprocessing -> Training -> Evaluation -> Save results

Architecture: 2D HMM on [ie_change, ie_vol_5d]
              + IE anchoring z-score (vol z-scored)
              + IE-gold sensitivity z-score (5d corr z-scored)

Output: 3 features
  - ie_regime_prob: Probability of high-variance IE regime (HMM)
  - ie_anchoring_z: IE change volatility z-score vs baseline
  - ie_gold_sensitivity_z: 5d IE-gold correlation z-scored vs baseline
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# === 1. Data Fetching ===
def fetch_and_preprocess():
    """Self-contained. Fetches T10YIE from FRED and GC=F from Yahoo Finance.
    Returns: (train_df, val_df, test_df, full_df)
    """
    # --- FRED API ---
    try:
        from fredapi import Fred
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "fredapi"], check=True)
        from fredapi import Fred

    # --- Yahoo Finance ---
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "yfinance"], check=True)
        import yfinance as yf

    # FRED API key (embedded for Kaggle execution)
    api_key = "3ffb68facdf6321e180e380c00e909c8"

    fred = Fred(api_key=api_key)

    # Fetch T10YIE (10-Year Breakeven Inflation Rate)
    # Start 2014-06-01 for warmup buffer (120d baseline + additional margin)
    print("Fetching T10YIE from FRED...")
    t10yie = fred.get_series('T10YIE', observation_start='2014-06-01')

    if len(t10yie) < 1000:
        raise RuntimeError(f"Insufficient T10YIE data: only {len(t10yie)} observations")

    # Convert to DataFrame
    df = pd.DataFrame({'T10YIE': t10yie})
    df.index = pd.to_datetime(df.index)

    # Fetch GC=F (Gold Futures) for gold returns
    print("Fetching GC=F from Yahoo Finance...")
    gc = yf.download('GC=F', start='2014-06-01', progress=False)

    if gc.empty:
        raise RuntimeError("Failed to fetch GC=F data from Yahoo Finance")

    # Flatten MultiIndex columns if present
    if isinstance(gc.columns, pd.MultiIndex):
        gc.columns = gc.columns.get_level_values(0)

    gc_close = gc['Close']
    gc_close.index = pd.to_datetime(gc_close.index)

    # Align dates (inner join on common trading days)
    # FRED data includes weekends/holidays with same values, Yahoo only has trading days
    df = df.join(gc_close.rename('gc_close'), how='inner')

    # Forward-fill gaps up to 3 trading days
    df = df.ffill(limit=3)

    # Drop any remaining NaN
    df = df.dropna()

    # === Compute Derived Features ===

    # 1. IE daily change (basis for all features)
    df['ie_change'] = df['T10YIE'].diff()

    # 2. Gold returns (current-day, not next-day - for sensitivity feature)
    df['gold_return'] = df['gc_close'].pct_change()

    # 3. IE volatility windows (for HMM input and anchoring feature)
    df['ie_vol_5d'] = df['ie_change'].rolling(5).std()
    df['ie_vol_10d'] = df['ie_change'].rolling(10).std()
    df['ie_vol_20d'] = df['ie_change'].rolling(20).std()

    # Drop rows with NaN from rolling operations
    df = df.dropna()

    # === Basic Validation ===

    # Check row count
    if len(df) < 2000:
        raise RuntimeError(f"Insufficient data: only {len(df)} rows after preprocessing")

    # Check T10YIE range (breakeven rates are percentages, typically 0-5%)
    if not (0 <= df['T10YIE'].min() <= df['T10YIE'].max() <= 5):
        raise RuntimeError(
            f"T10YIE out of expected range [0, 5]: "
            f"{df['T10YIE'].min():.3f} to {df['T10YIE'].max():.3f}"
        )

    # Check for extreme outliers in ie_change (typical daily change is 0.01-0.05)
    extreme_changes = df['ie_change'].abs() > 0.5
    if extreme_changes.any():
        print(f"Warning: {extreme_changes.sum()} extreme ie_change values (|value| > 0.5)")

    # Check ie_vol_5d for excessive zeros (some zeros are OK if IE is stable)
    zero_vol_pct = (df['ie_vol_5d'] == 0).sum() / len(df)
    if zero_vol_pct > 0.10:  # More than 10% zero volatility is suspicious
        print(f"Warning: {zero_vol_pct*100:.1f}% of ie_vol_5d values are zero")

    # Check for negative volatility (should never happen)
    if (df['ie_vol_5d'] < 0).any():
        raise RuntimeError("Invalid data: ie_vol_5d contains negative values")

    # Check for excessive gaps (no more than 5% of data should be forward-filled)
    date_diffs = df.index.to_series().diff().dt.days
    large_gaps = (date_diffs > 5).sum()
    if large_gaps > len(df) * 0.05:
        print(f"Warning: {large_gaps} date gaps > 5 days detected")

    # === Split into train/val/test (70/15/15, time-series order) ===
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Print summary statistics
    print(f"\nData fetched successfully:")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    print(f"\nT10YIE statistics:")
    print(f"  Range: {df['T10YIE'].min():.3f}% to {df['T10YIE'].max():.3f}%")
    print(f"  Mean: {df['T10YIE'].mean():.3f}%")
    print(f"  Std: {df['T10YIE'].std():.3f}%")
    print(f"\nie_change statistics:")
    print(f"  Mean: {df['ie_change'].mean():.6f}")
    print(f"  Std: {df['ie_change'].std():.6f}")
    print(f"  Range: {df['ie_change'].min():.6f} to {df['ie_change'].max():.6f}")
    print(f"\nie_vol_5d statistics:")
    print(f"  Mean: {df['ie_vol_5d'].mean():.6f}")
    print(f"  Std: {df['ie_vol_5d'].std():.6f}")
    print(f"  Range: {df['ie_vol_5d'].min():.6f} to {df['ie_vol_5d'].max():.6f}")

    return train_df, val_df, test_df, df


# === 2. HMM Component ===
def train_hmm_regime(train_df, n_components, covariance_type, n_init):
    """Train 2D HMM on [ie_change, ie_vol_5d] to detect IE regimes.

    Returns: fitted HMM model
    """
    try:
        from hmmlearn import hmm as hmmlearn_hmm
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "hmmlearn"], check=True)
        from hmmlearn import hmm as hmmlearn_hmm

    # Prepare 2D input [ie_change, ie_vol_5d]
    X_train = train_df[['ie_change', 'ie_vol_5d']].values

    # Remove any remaining NaN or inf
    valid_mask = np.isfinite(X_train).all(axis=1)
    X_train_clean = X_train[valid_mask]

    if len(X_train_clean) < 100:
        raise RuntimeError(f"Insufficient valid data for HMM training: {len(X_train_clean)} rows")

    # Train GaussianHMM with multiple restarts
    best_model = None
    best_score = -np.inf

    for restart in range(n_init):
        model = hmmlearn_hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=100,
            tol=1e-4,
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

    print(f"HMM training complete: {n_components} components, covariance={covariance_type}, best log-likelihood={best_score:.2f}")

    return best_model


def compute_regime_probabilities(hmm_model, df):
    """Compute regime probabilities for all data.

    Returns: ndarray with regime probabilities
    """
    X = df[['ie_change', 'ie_vol_5d']].values

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


def identify_high_variance_state(hmm_model):
    """Identify which HMM state has highest variance in IE_change dimension.

    Returns: state index
    """
    state_vars = []

    for i in range(hmm_model.n_components):
        if hmm_model.covariance_type == 'full':
            # Extract variance of first dimension (ie_change) from full covariance matrix
            state_vars.append(float(hmm_model.covars_[i][0, 0]))
        elif hmm_model.covariance_type == 'diag':
            # Extract first diagonal element
            state_vars.append(float(hmm_model.covars_[i][0]))

    # High-variance state = state with highest variance in IE_change dimension
    high_var_state = np.argmax(state_vars)

    print(f"State variances (ie_change dimension): {state_vars}")
    print(f"High-variance state identified: {high_var_state} (variance: {state_vars[high_var_state]:.6f})")

    return high_var_state


# === 3. Deterministic Components ===
def compute_anchoring_z(df, vol_window, baseline_window):
    """Compute IE anchoring z-score: (vol_short - rolling_mean) / rolling_std

    Args:
        df: DataFrame with ie_change column
        vol_window: Window for short-term volatility (5, 10, or 20 days)
        baseline_window: Window for baseline statistics (60 or 120 days)

    Returns: Series of z-scores
    """
    # Select appropriate vol column based on window
    vol_col = f'ie_vol_{vol_window}d'

    if vol_col not in df.columns:
        # Compute on the fly if not pre-computed
        vol_short = df['ie_change'].rolling(vol_window).std()
    else:
        vol_short = df[vol_col]

    # Rolling mean and std of volatility
    rolling_mean = vol_short.rolling(baseline_window).mean()
    rolling_std = vol_short.rolling(baseline_window).std()

    # Z-score
    z_score = (vol_short - rolling_mean) / rolling_std

    # Clip to [-4, 4] for stability
    z_score = z_score.clip(-4, 4)

    # Replace inf/nan with 0
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)

    return z_score


def compute_sensitivity_z(df, corr_window=5, baseline_window=60):
    """Compute IE-gold sensitivity z-score: 5d rolling correlation z-scored.

    Args:
        df: DataFrame with ie_change and gold_return columns
        corr_window: Window for rolling correlation (fixed at 5)
        baseline_window: Window for baseline statistics (40, 60, or 90 days)

    Returns: Series of z-scores
    """
    # Rolling 5-day correlation between IE changes and gold returns
    rolling_corr = df['ie_change'].rolling(corr_window).corr(df['gold_return'])

    # Rolling baseline mean and std
    corr_mean = rolling_corr.rolling(baseline_window).mean()
    corr_std = rolling_corr.rolling(baseline_window).std()

    # Z-score
    z_score = (rolling_corr - corr_mean) / corr_std

    # Clip to [-4, 4] for stability
    z_score = z_score.clip(-4, 4)

    # Replace inf/nan with 0
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)

    return z_score


# === 4. Hyperparameter Optimization ===
def run_hpo(train_df, val_df, n_trials=30, timeout=300):
    """Run Optuna HPO for all hyperparameters.

    Returns: best_params dict, best_value
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
        n_components = trial.suggest_categorical('hmm_n_components', [2, 3])
        covariance_type = trial.suggest_categorical('hmm_covariance_type', ['full', 'diag'])
        n_init = trial.suggest_categorical('hmm_n_init', [3, 5, 10])
        anchoring_vol_window = trial.suggest_categorical('anchoring_vol_window', [5, 10, 20])
        anchoring_baseline_window = trial.suggest_categorical('anchoring_baseline_window', [60, 120])
        sensitivity_baseline_window = trial.suggest_categorical('sensitivity_baseline_window', [40, 60, 90])

        try:
            # Train HMM
            hmm_model = train_hmm_regime(train_df, n_components, covariance_type, n_init)
            high_var_state = identify_high_variance_state(hmm_model)

            # Generate features for validation set
            regime_probs = compute_regime_probabilities(hmm_model, val_df)
            regime_prob = regime_probs[:, high_var_state]

            anchoring_z = compute_anchoring_z(val_df, anchoring_vol_window, anchoring_baseline_window).values
            sensitivity_z = compute_sensitivity_z(val_df, 5, sensitivity_baseline_window).values

            # Stack features
            X_val = np.column_stack([regime_prob, anchoring_z, sensitivity_z])

            # Align with target (drop last row since target is shifted)
            X_val_aligned = X_val[:-1]

            if len(X_val_aligned) != len(val_target):
                return -np.inf

            # Check for NaN/constant features
            if np.any(np.isnan(X_val_aligned)) or np.any(np.std(X_val_aligned, axis=0) < 1e-6):
                return -np.inf

            # Compute mutual information for each feature
            mi_scores = mutual_info_regression(X_val_aligned, val_target, random_state=42)
            mi_total = mi_scores.sum()

            if np.isnan(mi_total):
                return -np.inf

            return mi_total

        except Exception as e:
            print(f"Trial failed: {e}")
            return -np.inf

    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    print(f"\nOptuna optimization complete:")
    print(f"  Best MI sum: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")
    print(f"  Trials completed: {len(study.trials)}")

    return study.best_params, study.best_value


# === 5. Feature Generation ===
def generate_features(train_df, val_df, test_df, full_df, params):
    """Generate final 3-feature output using best hyperparameters.

    Returns: DataFrame with 3 columns
    """
    # Train HMM on full training data
    hmm_model = train_hmm_regime(
        train_df,
        params['hmm_n_components'],
        params['hmm_covariance_type'],
        params['hmm_n_init']
    )
    high_var_state = identify_high_variance_state(hmm_model)

    # Generate regime probabilities for all data
    regime_probs = compute_regime_probabilities(hmm_model, full_df)
    regime_prob = regime_probs[:, high_var_state]

    # Generate deterministic features
    anchoring_z = compute_anchoring_z(
        full_df,
        params['anchoring_vol_window'],
        params['anchoring_baseline_window']
    ).values

    sensitivity_z = compute_sensitivity_z(
        full_df,
        5,  # corr_window is fixed at 5
        params['sensitivity_baseline_window']
    ).values

    # Create output DataFrame
    output = pd.DataFrame({
        'ie_regime_prob': regime_prob,
        'ie_anchoring_z': anchoring_z,
        'ie_gold_sensitivity_z': sensitivity_z
    }, index=full_df.index)

    print(f"\nFeature generation complete:")
    print(f"  Output shape: {output.shape}")
    print(f"  Columns: {list(output.columns)}")
    print(f"  Date range: {output.index.min().date()} to {output.index.max().date()}")
    print(f"\nFeature statistics:")
    for col in output.columns:
        print(f"  {col}:")
        print(f"    Mean: {output[col].mean():.4f}, Std: {output[col].std():.4f}")
        print(f"    Range: [{output[col].min():.4f}, {output[col].max():.4f}]")

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
    print("Inflation Expectation SubModel Training - Attempt 1")
    print("=" * 80)

    # Step 1: Fetch data
    print("\n[1/5] Fetching and preprocessing data...")
    train_df, val_df, test_df, full_df = fetch_and_preprocess()

    # Step 2: Hyperparameter optimization
    print("\n[2/5] Running Optuna hyperparameter optimization...")
    best_params, best_mi = run_hpo(train_df, val_df, n_trials=30, timeout=300)

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
        'feature': 'inflation_expectation',
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
            'test_samples': len(test_df),
            'full_samples': len(full_df)
        }
    }

    with open('training_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: training_result.json")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nFinal MI sum: {metrics['mi_sum']:.6f}")
    print(f"Output shape: {output.shape}")
    print(f"Columns: {list(output.columns)}")
