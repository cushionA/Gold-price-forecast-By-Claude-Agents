"""
Data Fetching: Cross-Asset
Generates 3 PROCESSED features (not raw prices) from gold/silver/copper futures.

CRITICAL: Outputs processed features only:
  1. xasset_regime_prob: HMM posterior probability of crisis regime
  2. xasset_recession_signal: Daily change in gold/copper ratio z-score
  3. xasset_divergence: Daily gold-silver return difference z-score
"""
import pandas as pd
import numpy as np
import subprocess
import sys


def fetch_and_preprocess():
    """
    Fetch gold/silver/copper futures data and generate processed features.
    Self-contained function for embedding in train.py.

    Returns:
        tuple: (train_df, val_df, test_df, full_df)
    """

    # === 1. Install hmmlearn if needed ===
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("Installing hmmlearn...")
        subprocess.run([sys.executable, "-m", "pip", "install", "hmmlearn"], check=True)
        from hmmlearn.hmm import GaussianHMM

    # === 2. Install yfinance if needed ===
    try:
        import yfinance as yf
    except ImportError:
        print("Installing yfinance...")
        subprocess.run([sys.executable, "-m", "pip", "install", "yfinance"], check=True)
        import yfinance as yf

    # === 3. Fetch data ===
    print("Fetching futures data from Yahoo Finance...")

    tickers = ['GC=F', 'SI=F', 'HG=F']
    start_date = '2014-06-01'  # Buffer for 90-day warmup

    raw_data = yf.download(tickers, start=start_date, end='2025-02-15', progress=False)

    # Extract Close prices
    if 'Close' in raw_data.columns:
        close_data = raw_data['Close']
    else:
        close_data = raw_data

    # Rename columns
    close_data = close_data.rename(columns={
        'GC=F': 'gold_close',
        'SI=F': 'silver_close',
        'HG=F': 'copper_close'
    })

    df = close_data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Forward-fill missing values (max 3 days)
    df = df.ffill(limit=3).dropna()

    print(f"Fetched {len(df)} rows from {df.index.min()} to {df.index.max()}")

    # === 4. Compute daily returns ===
    df['gold_ret'] = df['gold_close'].pct_change()
    df['silver_ret'] = df['silver_close'].pct_change()
    df['copper_ret'] = df['copper_close'].pct_change()

    # === 5. Feature 1: HMM Regime Probability ===
    print("Computing HMM regime probability...")

    # Prepare 3D input for HMM
    X = df[['gold_ret', 'silver_ret', 'copper_ret']].dropna().values

    # Use 3-state HMM with multi-restart
    n_components = 3
    n_restarts = 10

    best_model = None
    best_score = -np.inf

    for seed in range(n_restarts):
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type='full',
                n_iter=200,
                tol=1e-4,
                random_state=seed
            )
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as e:
            print(f"HMM restart {seed} failed: {e}")
            continue

    if best_model is None:
        raise RuntimeError("All HMM restarts failed")

    # Generate probabilities
    probs = best_model.predict_proba(X)

    # Identify highest-variance state (crisis regime)
    traces = [np.trace(best_model.covars_[i]) for i in range(n_components)]
    high_var_state = np.argmax(traces)

    # Extract regime probability
    regime_prob = probs[:, high_var_state]

    # Align back to original dataframe
    df_clean = df.dropna(subset=['gold_ret', 'silver_ret', 'copper_ret'])
    df_clean['xasset_regime_prob'] = regime_prob

    # Merge back to full dataframe
    df = df.merge(df_clean[['xasset_regime_prob']], left_index=True, right_index=True, how='left')

    # === 6. Feature 2: Recession Signal (Gold/Copper Ratio Z-Score Change) ===
    print("Computing recession signal...")

    # Compute gold/copper ratio
    df['gc_ratio'] = df['gold_close'] / df['copper_close']

    # Compute z-score with 90-day window
    zscore_window = 90
    rolling_mean = df['gc_ratio'].rolling(zscore_window).mean()
    rolling_std = df['gc_ratio'].rolling(zscore_window).std()
    df['gc_ratio_z'] = (df['gc_ratio'] - rolling_mean) / rolling_std

    # Take FIRST DIFFERENCE (critical for autocorrelation)
    df['xasset_recession_signal'] = df['gc_ratio_z'].diff()

    # Clip to [-4, 4]
    df['xasset_recession_signal'] = df['xasset_recession_signal'].clip(-4, 4)

    # === 7. Feature 3: Divergence (Gold-Silver Return Difference Z-Score) ===
    print("Computing divergence...")

    # Compute daily return difference
    df['gs_ret_diff'] = df['gold_ret'] - df['silver_ret']

    # Z-score against 20-day rolling window
    div_window = 20
    rolling_mean = df['gs_ret_diff'].rolling(div_window).mean()
    rolling_std = df['gs_ret_diff'].rolling(div_window).std()
    df['xasset_divergence'] = (df['gs_ret_diff'] - rolling_mean) / rolling_std

    # Clip to [-4, 4]
    df['xasset_divergence'] = df['xasset_divergence'].clip(-4, 4)

    # === 8. Extract ONLY processed features ===
    output_df = df[['xasset_regime_prob', 'xasset_recession_signal', 'xasset_divergence']].copy()

    # Forward-fill remaining NaN (from warmup period)
    output_df = output_df.ffill(limit=5)

    # Trim to target date range (matching base_features)
    output_df = output_df.loc['2015-01-30':'2025-02-12']

    print(f"Output shape: {output_df.shape}")
    print(f"Date range: {output_df.index.min()} to {output_df.index.max()}")

    # Check autocorrelations
    print("\nAutocorrelation check:")
    for col in output_df.columns:
        autocorr = output_df[col].autocorr(lag=1)
        print(f"  {col}: {autocorr:.6f}")
        if autocorr > 0.99:
            raise RuntimeError(f"{col} autocorrelation {autocorr:.6f} exceeds 0.99 threshold")

    # === 9. Split into train/val/test ===
    n = len(output_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = output_df.iloc[:train_end]
    val_df = output_df.iloc[train_end:val_end]
    test_df = output_df.iloc[val_end:]

    print(f"\nData split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    return train_df, val_df, test_df, output_df


if __name__ == "__main__":
    train_df, val_df, test_df, full_df = fetch_and_preprocess()

    print("\n=== Summary Statistics ===")
    print(full_df.describe())

    print("\n=== Missing Values ===")
    print(full_df.isnull().sum())

    print("\n=== Sample Output (first 5 rows) ===")
    print(full_df.head())

    print("\n=== Sample Output (last 5 rows) ===")
    print(full_df.tail())
