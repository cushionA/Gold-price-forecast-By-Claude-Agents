"""
Data Fetching: CNY Demand Proxy
builder_model will embed this code in train.ipynb
Self-contained function for Kaggle compatibility
"""
import pandas as pd
import numpy as np


def fetch_and_preprocess():
    """
    Fetch and preprocess CNY demand proxy data.
    Self-contained: No external file dependencies.

    Returns:
        tuple: (train_df, val_df, test_df, full_df)
               Each df contains: cny_return, cny_vol_5d, gold_return
    """
    # === 1. Library imports with fallback installation ===
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "yfinance"], check=True)
        import yfinance as yf

    # === 2. Fetch CNY=X from Yahoo Finance ===
    # Start date: 2014-06-01 (buffer for 120-day warmup before 2015-01-30)
    print("Fetching CNY=X (CNY/USD exchange rate)...")
    cny_data = yf.download('CNY=X', start='2014-06-01', progress=False)

    if cny_data.empty:
        raise RuntimeError("Failed to fetch CNY=X data from Yahoo Finance")

    # Extract Close price
    if isinstance(cny_data.columns, pd.MultiIndex):
        cny_close = cny_data['Close']['CNY=X'].copy()
    else:
        cny_close = cny_data['Close'].copy()

    cny_close = cny_close.dropna()
    print(f"CNY=X data fetched: {len(cny_close)} rows from {cny_close.index[0]} to {cny_close.index[-1]}")

    # Validate CNY/USD range (reasonable onshore rate)
    if cny_close.min() < 5.5 or cny_close.max() > 8.0:
        print(f"WARNING: CNY/USD range [{cny_close.min():.2f}, {cny_close.max():.2f}] outside expected [5.5, 8.0]")

    # === 3. Fetch GC=F for gold returns ===
    print("Fetching GC=F (Gold Futures) for return computation...")
    gc_data = yf.download('GC=F', start='2014-06-01', progress=False)

    if gc_data.empty:
        raise RuntimeError("Failed to fetch GC=F data from Yahoo Finance")

    # Extract Close price
    if isinstance(gc_data.columns, pd.MultiIndex):
        gc_close = gc_data['Close']['GC=F'].copy()
    else:
        gc_close = gc_data['Close'].copy()

    gc_close = gc_close.dropna()
    print(f"GC=F data fetched: {len(gc_close)} rows from {gc_close.index[0]} to {gc_close.index[-1]}")

    # === 4. Compute derived quantities ===
    # CNY daily return
    cny_return = cny_close.pct_change()

    # CNY 5-day rolling volatility (for initial HMM input)
    cny_vol_5d = cny_return.rolling(5).std()

    # Gold current-day return (for MI evaluation, not model input)
    gold_return = gc_close.pct_change()

    # === 5. Align dates (inner join on trading dates) ===
    df = pd.DataFrame({
        'cny_close': cny_close,
        'cny_return': cny_return,
        'cny_vol_5d': cny_vol_5d,
        'gold_return': gold_return
    })

    # Drop NaN rows from returns/volatility computation
    initial_rows = len(df)
    df = df.dropna()
    print(f"After alignment and NaN removal: {len(df)} rows (dropped {initial_rows - len(df)} rows)")

    # === 6. Validate data quality ===
    # Check for extreme CNY returns (managed float should not have |return| > 0.05)
    extreme_returns = (df['cny_return'].abs() > 0.05).sum()
    if extreme_returns > 0:
        print(f"WARNING: {extreme_returns} CNY returns exceed 5% (max: {df['cny_return'].abs().max():.4f})")

    # Check for constant volatility (should be varying)
    if df['cny_vol_5d'].std() < 1e-6:
        raise RuntimeError("CNY volatility is constant (std < 1e-6)")

    # Check for zero/negative volatility (can occur in managed float periods)
    zero_vol_count = (df['cny_vol_5d'] == 0).sum()
    if zero_vol_count > 0:
        print(f"INFO: {zero_vol_count} periods with zero volatility (PBOC fixed rate)")

    if (df['cny_vol_5d'] < 0).any():
        raise RuntimeError("CNY volatility contains negative values (implementation error)")

    # === 7. Trim to base_features date range ===
    # Expected: 2015-01-30 to latest
    df.index = pd.to_datetime(df.index)
    base_start = pd.Timestamp('2015-01-30')

    if df.index[0] > base_start:
        print(f"WARNING: CNY data starts at {df.index[0]}, later than expected {base_start}")

    df = df[df.index >= base_start]
    print(f"After trimming to base_features range (>= {base_start}): {len(df)} rows")

    # === 8. Data split (70/15/15, time-series order) ===
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nData split:")
    print(f"  Train: {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  Val:   {len(val_df)} rows ({val_df.index[0]} to {val_df.index[-1]})")
    print(f"  Test:  {len(test_df)} rows ({test_df.index[0]} to {test_df.index[-1]})")

    # === 9. Summary statistics ===
    print(f"\nSummary statistics (full dataset):")
    print(f"  CNY/USD range: [{df['cny_close'].min():.4f}, {df['cny_close'].max():.4f}]")
    print(f"  CNY return mean: {df['cny_return'].mean():.6f}, std: {df['cny_return'].std():.6f}")
    print(f"  CNY return range: [{df['cny_return'].min():.6f}, {df['cny_return'].max():.6f}]")
    print(f"  CNY vol_5d mean: {df['cny_vol_5d'].mean():.6f}, std: {df['cny_vol_5d'].std():.6f}")
    print(f"  Gold return mean: {df['gold_return'].mean():.6f}, std: {df['gold_return'].std():.6f}")

    # Autocorrelation check (lag 1)
    cny_return_autocorr = df['cny_return'].autocorr(lag=1)
    print(f"  CNY return autocorr(lag=1): {cny_return_autocorr:.4f}")

    return train_df, val_df, test_df, df


if __name__ == "__main__":
    # Test execution
    train_df, val_df, test_df, full_df = fetch_and_preprocess()
    print("\nData fetching complete!")
    print(f"Total samples: {len(full_df)}")
    print(f"Columns: {list(full_df.columns)}")
