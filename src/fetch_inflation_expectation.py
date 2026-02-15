"""
Data fetching: inflation_expectation
builder_model will embed this code in train.py
"""
import pandas as pd
import numpy as np

def fetch_and_preprocess():
    """Self-contained. Fetches T10YIE, T5YIFR from FRED and GC=F from Yahoo Finance.
    Computes derived features for inflation expectation submodel.
    Returns: (train_df, val_df, test_df, full_df)
    """
    # --- FRED API ---
    try:
        from fredapi import Fred
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "fredapi"], check=True)
        from fredapi import Fred

    # --- Yahoo Finance ---
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "yfinance"], check=True)
        import yfinance as yf

    # Get FRED API key from environment (local) or Kaggle Secrets
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Kaggle environment doesn't need python-dotenv

    api_key = os.environ.get('FRED_API_KEY')
    if api_key is None:
        try:
            from kaggle_secrets import UserSecretsClient
            api_key = UserSecretsClient().get_secret("FRED_API_KEY")
        except Exception:
            raise RuntimeError(
                "FRED_API_KEY not found. "
                "Local: set in .env file / Kaggle: register in Secrets"
            )

    fred = Fred(api_key=api_key)

    # Fetch T10YIE (10-Year Breakeven Inflation Rate)
    # Start 2014-06-01 for warmup buffer (120d baseline + additional margin)
    print("Fetching T10YIE from FRED...")
    t10yie = fred.get_series('T10YIE', observation_start='2014-06-01')

    if len(t10yie) < 1000:
        raise RuntimeError(f"Insufficient T10YIE data: only {len(t10yie)} observations")

    # Convert to DataFrame
    df = pd.DataFrame({'T10YIE': t10yie})
    df.index = pd.to_datetime(df.index)

    # Fetch T5YIFR (5-Year, 5-Year Forward Inflation Expectation Rate)
    # This is available for Approach D (term structure HMM) if needed
    print("Fetching T5YIFR from FRED...")
    t5yifr = fred.get_series('T5YIFR', observation_start='2014-06-01')

    if len(t5yifr) < 1000:
        raise RuntimeError(f"Insufficient T5YIFR data: only {len(t5yifr)} observations")

    # Add T5YIFR to dataframe
    df['T5YIFR'] = t5yifr

    # Fetch GC=F (Gold Futures) for gold returns
    print("Fetching GC=F from Yahoo Finance...")
    gc = yf.download('GC=F', start='2014-06-01', progress=False)

    if gc.empty:
        raise RuntimeError("Failed to fetch GC=F data from Yahoo Finance")

    # Flatten MultiIndex columns if present
    if isinstance(gc.columns, pd.MultiIndex):
        gc.columns = gc.columns.get_level_values(0)

    gc_close = gc['Close']
    gc_close.index = pd.to_datetime(gc_close.index)

    # Align dates (inner join on common trading days)
    # FRED data includes weekends/holidays with same values, Yahoo only has trading days
    df = df.join(gc_close.rename('gc_close'), how='inner')

    # Forward-fill gaps up to 3 trading days
    df = df.ffill(limit=3)

    # Drop any remaining NaN
    df = df.dropna()

    # === Compute Derived Features ===

    # 1. IE daily change (basis for all features)
    df['ie_change'] = df['T10YIE'].diff()

    # 2. Gold returns (current-day, not next-day - for sensitivity feature)
    df['gold_return'] = df['gc_close'].pct_change()

    # 3. IE volatility windows (for HMM input and anchoring feature)
    df['ie_vol_5d'] = df['ie_change'].rolling(5).std()
    df['ie_vol_10d'] = df['ie_change'].rolling(10).std()
    df['ie_vol_20d'] = df['ie_change'].rolling(20).std()

    # 4. T5YIFR change (for Approach D if needed)
    df['t5yifr_change'] = df['T5YIFR'].diff()

    # Drop rows with NaN from rolling operations
    df = df.dropna()

    # === Basic Validation ===

    # Check row count
    if len(df) < 2000:
        raise RuntimeError(f"Insufficient data: only {len(df)} rows after preprocessing")

    # Check T10YIE range (breakeven rates are percentages, typically 0-5%)
    if not (0 <= df['T10YIE'].min() <= df['T10YIE'].max() <= 5):
        raise RuntimeError(
            f"T10YIE out of expected range [0, 5]: "
            f"{df['T10YIE'].min():.3f} to {df['T10YIE'].max():.3f}"
        )

    # Check for extreme outliers in ie_change (typical daily change is 0.01-0.05)
    extreme_changes = df['ie_change'].abs() > 0.5
    if extreme_changes.any():
        print(f"Warning: {extreme_changes.sum()} extreme ie_change values (|value| > 0.5)")

    # Check ie_vol_5d for excessive zeros (some zeros are OK if IE is stable)
    zero_vol_pct = (df['ie_vol_5d'] == 0).sum() / len(df)
    if zero_vol_pct > 0.10:  # More than 10% zero volatility is suspicious
        print(f"Warning: {zero_vol_pct*100:.1f}% of ie_vol_5d values are zero")

    # Check for negative volatility (should never happen)
    if (df['ie_vol_5d'] < 0).any():
        raise RuntimeError("Invalid data: ie_vol_5d contains negative values")

    # Check for excessive gaps (no more than 5% of data should be forward-filled)
    # This is a proxy check - we already dropped NaN, but check for suspicious patterns
    date_diffs = df.index.to_series().diff().dt.days
    large_gaps = (date_diffs > 5).sum()
    if large_gaps > len(df) * 0.05:
        print(f"Warning: {large_gaps} date gaps > 5 days detected")

    # === Split into train/val/test (70/15/15, time-series order) ===
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Print summary statistics
    print(f"\nData fetched successfully:")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    print(f"\nT10YIE statistics:")
    print(f"  Range: {df['T10YIE'].min():.3f}% to {df['T10YIE'].max():.3f}%")
    print(f"  Mean: {df['T10YIE'].mean():.3f}%")
    print(f"  Std: {df['T10YIE'].std():.3f}%")
    print(f"\nie_change statistics:")
    print(f"  Mean: {df['ie_change'].mean():.6f}")
    print(f"  Std: {df['ie_change'].std():.6f}")
    print(f"  Range: {df['ie_change'].min():.6f} to {df['ie_change'].max():.6f}")
    print(f"\nie_vol_5d statistics:")
    print(f"  Mean: {df['ie_vol_5d'].mean():.6f}")
    print(f"  Std: {df['ie_vol_5d'].std():.6f}")
    print(f"  Range: {df['ie_vol_5d'].min():.6f} to {df['ie_vol_5d'].max():.6f}")

    return train_df, val_df, test_df, df
