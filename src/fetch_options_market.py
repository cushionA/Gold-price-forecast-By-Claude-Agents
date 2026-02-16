"""
Data fetching: options_market
builder_model will embed this code in train.ipynb for Kaggle execution
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def fetch_and_preprocess():
    """Self-contained data fetcher for options_market submodel.
    Fetches SKEW from Yahoo Finance, GVZ from FRED (fallback to Yahoo).
    Returns: (train_df, val_df, test_df, full_df)
    """
    # --- yfinance ---
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "yfinance"], check=True)
        import yfinance as yf

    # --- fredapi ---
    try:
        from fredapi import Fred
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "fredapi"], check=True)
        from fredapi import Fred

    # --- Load environment ---
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Kaggle environment doesn't need dotenv

    # Get FRED API key
    api_key = os.environ.get('FRED_API_KEY')
    if api_key is None:
        try:
            from kaggle_secrets import UserSecretsClient
            api_key = UserSecretsClient().get_secret("FRED_API_KEY")
        except Exception:
            raise RuntimeError(
                "FRED_API_KEY not found. "
                "Local: set in .env / Kaggle: register in Secrets"
            )

    # --- Fetch SKEW from Yahoo Finance ---
    print("Fetching SKEW from Yahoo Finance...")
    skew_ticker = yf.Ticker("^SKEW")
    # Start from 2014-10-01 for warmup buffer (90 days before 2015-01-30)
    skew_data = skew_ticker.history(start='2014-10-01', end='2026-02-15')

    if len(skew_data) == 0:
        raise RuntimeError("SKEW data fetch failed: no data returned from Yahoo Finance")

    skew_df = pd.DataFrame({
        'skew_close': skew_data['Close']
    })
    skew_df.index = pd.to_datetime(skew_df.index).tz_localize(None)
    skew_df = skew_df.sort_index()

    print(f"SKEW: {len(skew_df)} rows from {skew_df.index.min()} to {skew_df.index.max()}")

    # --- Fetch GVZ from FRED (primary) ---
    print("Fetching GVZ from FRED...")
    fred = Fred(api_key=api_key)

    try:
        gvz_series = fred.get_series('GVZCLS', observation_start='2014-10-01')
        gvz_df = pd.DataFrame({'gvz_close': gvz_series})
        gvz_df.index = pd.to_datetime(gvz_df.index)
        print(f"GVZ (FRED): {len(gvz_df)} rows from {gvz_df.index.min()} to {gvz_df.index.max()}")
    except Exception as e:
        print(f"FRED GVZ fetch failed: {e}. Falling back to Yahoo Finance...")
        # Fallback to Yahoo Finance ^GVZ
        gvz_ticker = yf.Ticker("^GVZ")
        gvz_data = gvz_ticker.history(start='2014-10-01', end='2026-02-15')

        if len(gvz_data) == 0:
            raise RuntimeError("GVZ data fetch failed from both FRED and Yahoo Finance")

        gvz_df = pd.DataFrame({'gvz_close': gvz_data['Close']})
        gvz_df.index = pd.to_datetime(gvz_df.index).tz_localize(None)
        print(f"GVZ (Yahoo): {len(gvz_df)} rows from {gvz_df.index.min()} to {gvz_df.index.max()}")

    gvz_df = gvz_df.sort_index()

    # --- Align SKEW and GVZ on common dates ---
    print("Aligning SKEW and GVZ on common dates...")
    df = pd.merge(skew_df, gvz_df, left_index=True, right_index=True, how='inner')

    print(f"After alignment: {len(df)} rows from {df.index.min()} to {df.index.max()}")

    # --- Handle missing values (forward-fill max 3 days) ---
    df = df.ffill(limit=3)

    # Drop any remaining NaN rows
    initial_rows = len(df)
    df = df.dropna()
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows with NaN after forward-fill")

    # --- Compute daily changes ---
    df['skew_change'] = df['skew_close'].diff()
    df['gvz_change'] = df['gvz_close'].diff()

    # --- Calculate SKEW z-scores (multiple windows for exploration) ---
    # These will be used by builder_model during Optuna HPO
    for window in [40, 60, 90]:
        rolling_mean = df['skew_close'].rolling(window).mean()
        rolling_std = df['skew_close'].rolling(window).std()
        z = (df['skew_close'] - rolling_mean) / rolling_std
        df[f'skew_z_{window}d'] = z.clip(-4, 4)

    # --- Calculate SKEW momentum z-scores (multiple windows) ---
    for momentum_window in [5, 10, 15]:
        momentum_raw = df['skew_close'].diff(momentum_window)
        rolling_mean = momentum_raw.rolling(60).mean()
        rolling_std = momentum_raw.rolling(60).std()
        z = (momentum_raw - rolling_mean) / rolling_std
        df[f'skew_momentum_z_{momentum_window}d'] = z.clip(-4, 4)

    # --- Basic statistics ---
    print("\n=== SKEW Statistics ===")
    print(f"Mean: {df['skew_close'].mean():.2f}")
    print(f"Std: {df['skew_close'].std():.2f}")
    print(f"Min: {df['skew_close'].min():.2f}")
    print(f"Max: {df['skew_close'].max():.2f}")
    print(f"Autocorr(1): {df['skew_close'].autocorr(lag=1):.4f}")
    print(f"Change Autocorr(1): {df['skew_change'].autocorr(lag=1):.4f}")

    print("\n=== GVZ Statistics ===")
    print(f"Mean: {df['gvz_close'].mean():.2f}")
    print(f"Std: {df['gvz_close'].std():.2f}")
    print(f"Min: {df['gvz_close'].min():.2f}")
    print(f"Max: {df['gvz_close'].max():.2f}")
    print(f"Autocorr(1): {df['gvz_close'].autocorr(lag=1):.4f}")
    print(f"Change Autocorr(1): {df['gvz_change'].autocorr(lag=1):.4f}")

    print(f"\n=== Change Correlation ===")
    print(f"SKEW change vs GVZ change: {df[['skew_change', 'gvz_change']].corr().iloc[0, 1]:.4f}")

    # --- Data quality checks ---
    print("\n=== Data Quality Checks ===")
    # Check SKEW range
    if df['skew_close'].min() < 100 or df['skew_close'].max() > 200:
        print(f"WARNING: SKEW values outside expected range [100, 200]")
    else:
        print(f"OK: SKEW values in expected range [100, 200]")

    # Check GVZ range
    if df['gvz_close'].min() < 5 or df['gvz_close'].max() > 80:
        print(f"WARNING: GVZ values outside expected range [5, 80]")
    else:
        print(f"OK: GVZ values in expected range [5, 80]")

    # Check extreme outliers in changes
    extreme_skew = df['skew_change'].abs() > 30
    extreme_gvz = df['gvz_change'].abs() > 20

    if extreme_skew.sum() > 0:
        print(f"WARNING: {extreme_skew.sum()} extreme SKEW changes (|change| > 30)")
    else:
        print(f"OK: No extreme SKEW changes")

    if extreme_gvz.sum() > 0:
        print(f"WARNING: {extreme_gvz.sum()} extreme GVZ changes (|change| > 20)")
    else:
        print(f"OK: No extreme GVZ changes")

    # Check missing data percentage
    missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
    print(f"Missing data: {missing_pct:.2f}%")

    # --- Train/val/test split (70/15/15, time-series order) ---
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\n=== Data Split ===")
    print(f"Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Val: {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    print(f"Test: {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")
    print(f"Total: {len(df)} rows")

    return train_df, val_df, test_df, df


if __name__ == "__main__":
    # Test locally
    train_df, val_df, test_df, full_df = fetch_and_preprocess()
    print("\n=== Fetch Complete ===")
    print(f"Full dataset shape: {full_df.shape}")
    print(f"\nColumns: {list(full_df.columns)}")
    print(f"\nFirst 5 rows:\n{full_df.head()}")
    print(f"\nLast 5 rows:\n{full_df.tail()}")
