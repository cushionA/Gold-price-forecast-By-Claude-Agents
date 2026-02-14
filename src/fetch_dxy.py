"""
Data Fetching: DXY and Constituent Currencies
Builder_model will embed this code in train.py for Kaggle execution
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def fetch_and_preprocess():
    """
    Fetch DXY index and 6 constituent currency pairs.
    Returns: (raw_prices_df, dxy_currencies_df)
    - raw_prices_df: raw Close prices for all 7 tickers (for verification)
    - dxy_currencies_df: processed daily data with DXY close and 6 currency pair closes
    """

    # === Configuration ===
    tickers = {
        'DXY': 'DX-Y.NYB',
        'EURUSD': 'EURUSD=X',
        'USDJPY': 'JPY=X',
        'GBPUSD': 'GBPUSD=X',
        'USDCAD': 'CAD=X',
        'USDSEK': 'SEK=X',
        'USDCHF': 'CHF=X'
    }

    # Start date with 60-day buffer for rolling windows
    start_date = '2014-12-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    # === Fetch data from Yahoo Finance ===
    print(f"Fetching data from {start_date} to {end_date}...")

    data_frames = {}
    for name, ticker in tickers.items():
        try:
            print(f"Downloading {name} ({ticker})...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            # Extract Close column - handle both single and multi-ticker downloads
            if isinstance(df.columns, pd.MultiIndex):
                close_series = df['Close'].iloc[:, 0]
            else:
                close_series = df['Close']
            close_series.name = name
            data_frames[name] = close_series
            print(f"  -> {len(df)} rows fetched, latest: {close_series.iloc[-1]:.4f}")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch {name} ({ticker}): {e}")

    # === Combine all series ===
    raw_prices = pd.concat(data_frames.values(), axis=1)
    raw_prices.index.name = 'Date'

    # Forward-fill missing values (max 3 days)
    raw_prices = raw_prices.ffill(limit=3)

    # Drop rows with any remaining NaN
    initial_rows = len(raw_prices)
    raw_prices = raw_prices.dropna()
    dropped_rows = initial_rows - len(raw_prices)
    print(f"\nDropped {dropped_rows} rows with missing data after 3-day forward-fill")
    print(f"Final dataset: {len(raw_prices)} rows from {raw_prices.index[0]} to {raw_prices.index[-1]}")

    # === Create processed dataframe ===
    # Column naming: dxy_close, eur_usd, jpy, gbp_usd, usd_cad, usd_sek, usd_chf
    dxy_currencies = pd.DataFrame({
        'dxy_close': raw_prices['DXY'],
        'eur_usd': raw_prices['EURUSD'],
        'jpy': raw_prices['USDJPY'],
        'gbp_usd': raw_prices['GBPUSD'],
        'usd_cad': raw_prices['USDCAD'],
        'usd_sek': raw_prices['USDSEK'],
        'usd_chf': raw_prices['USDCHF']
    })

    dxy_currencies.index.name = 'Date'

    # === Quality checks ===
    print("\n=== Quality Checks ===")
    for col in dxy_currencies.columns:
        missing_pct = dxy_currencies[col].isna().sum() / len(dxy_currencies) * 100
        print(f"{col}: {missing_pct:.2f}% missing")
        if missing_pct > 1.0:
            raise ValueError(f"{col} has {missing_pct:.2f}% missing data (>1% threshold)")

    # Check for gaps > 3 consecutive days
    date_diff = dxy_currencies.index.to_series().diff()
    max_gap = date_diff.max().days
    print(f"\nMax gap between trading days: {max_gap} days")
    if max_gap > 5:  # Allow for weekends + 1 holiday
        print(f"WARNING: Large gap detected ({max_gap} days)")

    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(dxy_currencies.describe())

    return raw_prices, dxy_currencies


if __name__ == "__main__":
    # For local testing
    raw, processed = fetch_and_preprocess()
    print(f"\nRaw prices shape: {raw.shape}")
    print(f"Processed data shape: {processed.shape}")
    print("\nFirst 5 rows of processed data:")
    print(processed.head())
    print("\nLast 5 rows of processed data:")
    print(processed.tail())
