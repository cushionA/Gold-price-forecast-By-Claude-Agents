"""
Data Fetcher - Common data fetching utilities
Used by builder_data and embedded in train.py for self-contained execution
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
from typing import Optional, List, Tuple


def get_fred_api() -> Fred:
    """Get FRED API instance with error handling"""
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        raise KeyError("FRED_API_KEY not found in environment variables")
    return Fred(api_key=api_key)


def fetch_yahoo_data(ticker: str, start_date: str, end_date: Optional[str] = None,
                    interval: str = '1d') -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance

    Args:
        ticker: Yahoo ticker symbol (e.g., 'GC=F', 'DX-Y.NYB')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date (default: today)
        interval: Data interval (default: '1d')

    Returns:
        DataFrame with OHLCV data, indexed by date
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

    if data.empty:
        raise ValueError(f"No data fetched for {ticker}")

    return data


def fetch_fred_series(series_id: str, start_date: str,
                     end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch data from FRED

    Args:
        series_id: FRED series ID (e.g., 'DFII10', 'VIXCLS')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date (default: today)

    Returns:
        DataFrame with series data, indexed by date
    """
    fred = get_fred_api()

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    series = fred.get_series(series_id,
                            observation_start=start_date,
                            observation_end=end_date)

    df = pd.DataFrame({'value': series})
    df.index.name = 'date'

    return df


def fetch_multi_country_data(base_series: str, country_codes: List[str],
                            start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch similar data across multiple countries

    Args:
        base_series: Base series pattern (e.g., 'IR{}10Y' where {} is country code)
        country_codes: List of country codes (e.g., ['US', 'GB', 'DE'])
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with columns for each country
    """
    fred = get_fred_api()

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    data = {}
    for code in country_codes:
        series_id = base_series.format(code)
        try:
            series = fred.get_series(series_id,
                                   observation_start=start_date,
                                   observation_end=end_date)
            data[code] = series
        except Exception as e:
            print(f"Warning: Failed to fetch {series_id}: {e}")
            continue

    if not data:
        raise ValueError(f"No data fetched for any country")

    df = pd.DataFrame(data)
    df.index.name = 'date'

    return df


def align_to_business_days(df: pd.DataFrame, reference_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align DataFrame to reference business day index (forward fill for missing days)

    Args:
        df: DataFrame with date index
        reference_index: Target date index (typically gold trading days)

    Returns:
        Reindexed DataFrame
    """
    df = df.reindex(reference_index, method='ffill')
    return df


def compute_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Compute percentage returns

    Args:
        prices: Price series
        periods: Number of periods for return calculation

    Returns:
        Return series (%)
    """
    returns = prices.pct_change(periods) * 100
    return returns


def compute_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Compute log returns

    Args:
        prices: Price series
        periods: Number of periods

    Returns:
        Log return series (%)
    """
    log_returns = np.log(prices / prices.shift(periods)) * 100
    return log_returns


def split_train_val_test(df: pd.DataFrame,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data into train/val/test (chronological order, no shuffle)

    Args:
        df: Input DataFrame
        train_ratio: Training set ratio
        val_ratio: Validation set ratio

    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df
