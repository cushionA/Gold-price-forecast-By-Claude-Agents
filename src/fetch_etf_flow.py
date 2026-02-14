"""
Data fetching: etf_flow
builder_model will embed this code in train.py
"""
import pandas as pd
import numpy as np

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
