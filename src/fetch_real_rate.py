"""
Data fetching: real_rate
builder_model will embed this code in train.py
"""
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import os


def fetch_and_preprocess():
    """Self-contained. No external file dependencies.
    Returns: DataFrame with engineered features
    """
    # === 1. Libraries ===
    try:
        from fredapi import Fred
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "fredapi"], check=True)
        from fredapi import Fred

    # Load API key (local: .env, Kaggle: Secrets)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Kaggle environment

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

    # === 2. Data Fetching ===
    # Fetch from 2013-06-01 to allow 252-day rolling windows before schema start
    print("Fetching DFII10 from FRED...")
    series = fred.get_series('DFII10', observation_start='2013-06-01')

    df = pd.DataFrame({'level': series})
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'

    # Drop NaN (holidays)
    df = df.dropna()

    # Forward fill to align with gold trading days (max 5 days)
    # Fetch gold trading days from yfinance for reference
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "yfinance"], check=True)
        import yfinance as yf

    print("Fetching gold trading days from yfinance...")
    gold = yf.download('GC=F', start='2013-06-01', progress=False)
    gold_dates = pd.DatetimeIndex(gold.index)

    # Reindex to gold trading days
    df = df.reindex(gold_dates)
    df['level'] = df['level'].ffill(limit=5)
    df = df.dropna()

    print(f"Raw data shape after alignment: {df.shape}")

    # === 3. Feature Engineering ===
    print("Computing hand-crafted features...")

    # Feature 2: Daily change
    df['change_1d'] = df['level'].diff()

    # Rolling std for normalization (60-day window)
    rolling_std_60d = df['change_1d'].rolling(60).std()

    # Feature 3: Velocity 20-day (normalized)
    df['velocity_20d'] = (df['level'] - df['level'].shift(20)) / rolling_std_60d

    # Feature 4: Velocity 60-day (normalized)
    df['velocity_60d'] = (df['level'] - df['level'].shift(60)) / rolling_std_60d

    # Feature 5: Acceleration (change in 20-day velocity)
    df['accel_20d'] = df['velocity_20d'] - df['velocity_20d'].shift(20)

    # Feature 6: Rolling std 20-day
    df['rolling_std_20d'] = df['change_1d'].rolling(20).std()

    # Feature 7: Regime percentile (252-day rolling percentile rank)
    def percentile_rank(x):
        if len(x) < 2:
            return np.nan
        return percentileofscore(x, x.iloc[-1]) / 100.0

    df['regime_percentile'] = df['level'].rolling(252).apply(percentile_rank, raw=False)

    # Feature 8: Autocorrelation of daily changes (60-day window, lag 1)
    def rolling_autocorr(x):
        if len(x) < 2:
            return np.nan
        try:
            return pd.Series(x).autocorr(lag=1)
        except:
            return np.nan

    df['autocorr_20d'] = df['change_1d'].rolling(60).apply(rolling_autocorr, raw=False)

    # Drop rows with NaN from rolling calculations
    initial_rows = len(df)
    df = df.dropna()
    print(f"Dropped {initial_rows - len(df)} rows due to rolling window NaN")

    # === 4. Align to schema date range ===
    schema_start = '2015-01-30'
    schema_end = '2025-02-12'

    df = df.loc[schema_start:schema_end]
    print(f"Final shape after schema alignment: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


if __name__ == "__main__":
    # Test execution
    df = fetch_and_preprocess()
    print("\n=== Feature Summary ===")
    print(df.describe())
    print("\n=== First 5 rows ===")
    print(df.head())
    print("\n=== Last 5 rows ===")
    print(df.tail())
