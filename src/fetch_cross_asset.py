"""
Data Fetching: Cross-Asset
Builder_model will embed this code in train.py for Kaggle execution
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


def fetch_and_preprocess():
    """
    Fetch and preprocess cross-asset data (gold, silver, copper futures).
    Self-contained function for embedding in train.py.

    Returns:
        tuple: (train_df, val_df, test_df, full_df)
    """
    # Fetch data from Yahoo Finance
    # Start from 2014-06-01 for 90-day warmup buffer
    start_date = '2014-06-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    tickers = ['GC=F', 'SI=F', 'HG=F']

    # Download all tickers at once to align dates
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        # Handle single vs multi-ticker download format
        if len(tickers) == 1:
            close_data = raw_data['Close'].to_frame()
            close_data.columns = ['gold_close']
        else:
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

        if close_data.empty:
            raise ValueError("No data downloaded")

    except Exception as e:
        raise RuntimeError(f"Failed to download data: {e}")

    # Create DataFrame
    df = close_data.copy()

    # Ensure DatetimeIndex
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Handle missing values (forward-fill up to 3 days)
    df = df.ffill(limit=3)

    # Drop rows with remaining NaN values
    df = df.dropna()

    # Compute daily returns
    df['gold_return'] = df['gold_close'].pct_change()
    df['silver_return'] = df['silver_close'].pct_change()
    df['copper_return'] = df['copper_close'].pct_change()

    # Compute gold/silver ratio
    df['gsr'] = df['gold_close'] / df['silver_close']

    # Compute gold/copper ratio
    df['gcr'] = df['gold_close'] / df['copper_close']

    # Reset index to make date a column
    df = df.reset_index()
    df.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
    if 'date' not in df.columns and df.index.name == 'Date':
        df = df.reset_index()
        df.rename(columns={'Date': 'date'}, inplace=True)

    # Ensure date column exists
    if 'date' not in df.columns:
        df['date'] = df.index

    # Filter to target date range (2014-10-01 onwards to match design requirement)
    df = df[df['date'] >= '2014-10-01'].copy()

    # Handle NaN from pct_change (first row)
    df = df.dropna()

    # Split into train/val/test (70/15/15, time-series order)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df, df


if __name__ == "__main__":
    # Local execution test
    train_df, val_df, test_df, full_df = fetch_and_preprocess()

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Full shape: {full_df.shape}")
    print(f"\nDate range: {full_df['date'].min()} to {full_df['date'].max()}")
    print(f"\nColumns: {list(full_df.columns)}")
    print(f"\nFirst 5 rows:\n{full_df.head()}")
    print(f"\nLast 5 rows:\n{full_df.tail()}")
    print(f"\nBasic statistics:\n{full_df.describe()}")
