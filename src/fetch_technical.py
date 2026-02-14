"""
Data fetching and preprocessing: Technical Features (GLD OHLC)
Builder_model will embed this code in train.py for Kaggle execution.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


def fetch_and_preprocess():
    """
    Fetch GLD OHLC data and compute returns + Garman-Klass volatility.

    Self-contained function for Kaggle integration.

    Returns:
        tuple: (train_df, val_df, test_df, full_df)
    """
    print("Fetching GLD OHLC data from Yahoo Finance...")

    # Fetch GLD data with buffer for warmup period
    # Need at least 60 trading days before 2015-01-30 for GK vol z-score baseline
    # Starting from 2014-10-01 provides ~90 trading days buffer
    # Filter to historical data only (up to today)
    today = datetime.now().strftime("%Y-%m-%d")
    ticker = yf.Ticker("GLD")
    df = ticker.history(start="2014-10-01", end=today, auto_adjust=False)

    if df.empty:
        raise ValueError("Failed to fetch GLD data from Yahoo Finance")

    print(f"Fetched {len(df)} rows from Yahoo Finance")

    # Rename columns to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Keep only required columns
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Reset index to make date a column
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date'}, inplace=True)

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Filter to historical data only (exclude future dates)
    today = pd.Timestamp.now(tz='UTC').normalize()
    df = df[df['date'] <= today].copy()

    # Sort by date
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Check for flat bars (H==L==O==C) - should be 0 for GLD
    flat_bars = ((df['high'] == df['low']) &
                 (df['low'] == df['open']) &
                 (df['open'] == df['close']))
    n_flat = flat_bars.sum()
    if n_flat > 0:
        print(f"WARNING: Found {n_flat} flat bars (H==L==O==C). This should not occur with GLD.")

    # Compute returns
    df['returns'] = df['close'].pct_change()

    # Compute Garman-Klass volatility
    # GK formula: sqrt(0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2)
    # Clip minimum values to avoid log(0) issues
    high_safe = df['high'].clip(lower=1e-8)
    low_safe = df['low'].clip(lower=1e-8)
    close_safe = df['close'].clip(lower=1e-8)
    open_safe = df['open'].clip(lower=1e-8)

    log_hl = np.log(high_safe / low_safe)
    log_co = np.log(close_safe / open_safe)

    df['gk_vol'] = np.sqrt(
        0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    )

    # Check for zero GK volatility days
    zero_gk = (df['gk_vol'] == 0).sum()
    if zero_gk > 0:
        print(f"WARNING: Found {zero_gk} days with GK vol = 0")

    # Handle missing values
    # Forward-fill gaps up to 3 days (using ffill() instead of deprecated fillna method)
    df['returns'] = df['returns'].ffill(limit=3)
    df['gk_vol'] = df['gk_vol'].ffill(limit=3)

    # Check for remaining NaN
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print(f"NaN counts after forward-fill:\n{nan_counts[nan_counts > 0]}")

    # Basic statistics
    print("\n=== Data Statistics ===")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nReturns statistics:")
    print(df['returns'].describe())
    print(f"\nGK volatility statistics:")
    print(df['gk_vol'].describe())

    # Check for extreme returns (>10% single day)
    extreme_returns = df[df['returns'].abs() > 0.10]
    if len(extreme_returns) > 0:
        print(f"\nWARNING: Found {len(extreme_returns)} days with |return| > 10%:")
        print(extreme_returns[['date', 'returns']])

    # Split data into train/val/test (70/15/15, time-series order)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\n=== Data Split ===")
    print(f"Train: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Val:   {len(val_df)} rows ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

    return train_df, val_df, test_df, df


if __name__ == "__main__":
    # Test the function locally
    train_df, val_df, test_df, full_df = fetch_and_preprocess()

    print("\n=== Fetch complete ===")
    print(f"Full dataset shape: {full_df.shape}")
    print(f"Columns: {list(full_df.columns)}")
