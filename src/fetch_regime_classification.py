"""
Data fetching for regime_classification submodel
Self-contained script for embedding in Kaggle notebook
Fetches FRED and Yahoo data, computes 4 z-scored input features
"""
import pandas as pd
import numpy as np
import os


def fetch_and_preprocess():
    """
    Self-contained data fetching for regime_classification submodel.
    Returns: (train_df, val_df, test_df, full_df)

    Features:
    - vix_z: z-score of VIX (20d rolling window)
    - yield_spread_z: z-score of (DGS10 - DGS2) (60d rolling window)
    - equity_return_z: z-score of S&P 500 5d return (60d rolling window)
    - gold_rvol_z: z-score of Gold 10d realized volatility (60d rolling window)
    """

    # === FRED API ===
    try:
        from fredapi import Fred
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "fredapi"], check=True)
        from fredapi import Fred

    # Get FRED API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Kaggle environment doesn't need dotenv

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

    fred = Fred(api_key=api_key)

    # === Yahoo Finance ===
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "yfinance"], check=True)
        import yfinance as yf

    # === 1. Fetch raw data ===
    print("Fetching FRED data...")
    vix = fred.get_series('VIXCLS', observation_start='2014-01-01')
    dgs10 = fred.get_series('DGS10', observation_start='2014-01-01')
    dgs2 = fred.get_series('DGS2', observation_start='2014-01-01')

    print("Fetching Yahoo data...")
    spx_data = yf.download('^GSPC', start='2014-01-01', progress=False)
    gold_data = yf.download('GC=F', start='2014-01-01', progress=False)

    # Extract Close prices (yfinance returns MultiIndex columns)
    if isinstance(spx_data.columns, pd.MultiIndex):
        spx_close = spx_data['Close'].iloc[:, 0]  # First ticker
    else:
        spx_close = spx_data['Close']

    if isinstance(gold_data.columns, pd.MultiIndex):
        gold_close = gold_data['Close'].iloc[:, 0]  # First ticker
    else:
        gold_close = gold_data['Close']

    # === 2. Convert to DataFrames and align indices ===
    df_vix = pd.DataFrame({'vix': vix})
    df_dgs10 = pd.DataFrame({'dgs10': dgs10})
    df_dgs2 = pd.DataFrame({'dgs2': dgs2})
    df_spx = pd.DataFrame({'spx': spx_close})
    df_gold = pd.DataFrame({'gold': gold_close})

    # Ensure datetime index
    for df in [df_vix, df_dgs10, df_dgs2, df_spx, df_gold]:
        df.index = pd.to_datetime(df.index)

    # Forward fill missing values (holidays, weekends)
    df_vix = df_vix.ffill(limit=5)
    df_dgs10 = df_dgs10.ffill(limit=5)
    df_dgs2 = df_dgs2.ffill(limit=5)
    df_spx = df_spx.ffill(limit=3)
    df_gold = df_gold.ffill(limit=3)

    # Inner join on date (only keep dates with all data available)
    df = df_vix.join(df_dgs10, how='inner')
    df = df.join(df_dgs2, how='inner')
    df = df.join(df_spx, how='inner')
    df = df.join(df_gold, how='inner')

    print(f"Raw data aligned: {len(df)} rows from {df.index.min()} to {df.index.max()}")

    # === 3. Feature engineering ===

    # vix_z: z-score of VIX (20d rolling window)
    vix_mean = df['vix'].rolling(20).mean()
    vix_std = df['vix'].rolling(20).std()
    df['vix_z'] = (df['vix'] - vix_mean) / vix_std
    df['vix_z'] = df['vix_z'].replace([np.inf, -np.inf], np.nan)

    # yield_spread_z: z-score of (DGS10 - DGS2) (60d rolling window)
    df['yield_spread'] = df['dgs10'] - df['dgs2']
    spread_mean = df['yield_spread'].rolling(60).mean()
    spread_std = df['yield_spread'].rolling(60).std()
    df['yield_spread_z'] = (df['yield_spread'] - spread_mean) / spread_std
    df['yield_spread_z'] = df['yield_spread_z'].replace([np.inf, -np.inf], np.nan)

    # equity_return_z: z-score of S&P 500 5d return (60d rolling window)
    df['spx_5d_ret'] = df['spx'].pct_change(5)
    ret_mean = df['spx_5d_ret'].rolling(60).mean()
    ret_std = df['spx_5d_ret'].rolling(60).std()
    df['equity_return_z'] = (df['spx_5d_ret'] - ret_mean) / ret_std
    df['equity_return_z'] = df['equity_return_z'].replace([np.inf, -np.inf], np.nan)

    # gold_rvol_z: z-score of Gold 10d realized volatility (60d rolling window)
    df['gold_log_ret'] = np.log(df['gold'] / df['gold'].shift(1))
    df['gold_rvol_10d'] = df['gold_log_ret'].rolling(10).std() * np.sqrt(252)  # annualized
    rvol_mean = df['gold_rvol_10d'].rolling(60).mean()
    rvol_std = df['gold_rvol_10d'].rolling(60).std()
    df['gold_rvol_z'] = (df['gold_rvol_10d'] - rvol_mean) / rvol_std
    df['gold_rvol_z'] = df['gold_rvol_z'].replace([np.inf, -np.inf], np.nan)

    # Clip z-scores to [-4, 4] to handle extreme outliers
    for col in ['vix_z', 'yield_spread_z', 'equity_return_z', 'gold_rvol_z']:
        df[col] = df[col].clip(-4, 4)

    # === 4. Drop rows with NaN (from rolling window warmup) ===
    features = ['vix_z', 'yield_spread_z', 'equity_return_z', 'gold_rvol_z']
    df_clean = df[features].dropna()

    print(f"After dropping NaN: {len(df_clean)} rows from {df_clean.index.min()} to {df_clean.index.max()}")
    print(f"NaN counts per feature:")
    for col in features:
        nan_count = df[col].isna().sum()
        print(f"  {col}: {nan_count} ({nan_count / len(df) * 100:.1f}%)")

    # === 5. Train/Val/Test split (70/15/15, time-series order) ===
    n = len(df_clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df_clean.iloc[:train_end]
    val_df = df_clean.iloc[train_end:val_end]
    test_df = df_clean.iloc[val_end:]

    print(f"\nData split:")
    print(f"  Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    print(f"  Test:  {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")

    return train_df, val_df, test_df, df_clean


if __name__ == "__main__":
    # Test the function locally
    train_df, val_df, test_df, full_df = fetch_and_preprocess()

    print("\n=== Data Quality Checks ===")
    print(f"\nFull dataset shape: {full_df.shape}")
    print(f"Date range: {full_df.index.min()} to {full_df.index.max()}")

    print("\nNaN counts:")
    print(full_df.isna().sum())

    print("\nBasic statistics:")
    print(full_df.describe())

    print("\nZ-score validation (should be approximately mean=0, std=1):")
    print(f"  vix_z:           mean={full_df['vix_z'].mean():.4f}, std={full_df['vix_z'].std():.4f}")
    print(f"  yield_spread_z:  mean={full_df['yield_spread_z'].mean():.4f}, std={full_df['yield_spread_z'].std():.4f}")
    print(f"  equity_return_z: mean={full_df['equity_return_z'].mean():.4f}, std={full_df['equity_return_z'].std():.4f}")
    print(f"  gold_rvol_z:     mean={full_df['gold_rvol_z'].mean():.4f}, std={full_df['gold_rvol_z'].std():.4f}")

    print("\nClipping validation (all values should be in [-4, 4]):")
    for col in full_df.columns:
        min_val = full_df[col].min()
        max_val = full_df[col].max()
        print(f"  {col}: min={min_val:.4f}, max={max_val:.4f}")

    print("\nPairwise correlations:")
    print(full_df.corr())
