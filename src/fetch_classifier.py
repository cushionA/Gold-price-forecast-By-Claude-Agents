"""
Data Fetching Function for Gold DOWN Classifier
Self-contained, ready to be embedded in Kaggle notebook by builder_model

This function fetches all raw data and computes all 18 features.
NO external file dependencies - can run standalone on Kaggle.
"""

import pandas as pd
import numpy as np
import os


def fetch_and_preprocess():
    """
    Fetch all data sources and compute 18 classifier features.

    Returns:
        train_df, val_df, test_df, full_df (each with 18 features + target)
    """

    # ========================================
    # STEP 1: Import libraries
    # ========================================
    try:
        import yfinance as yf
        from fredapi import Fred
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "yfinance", "fredapi"], check=True)
        import yfinance as yf
        from fredapi import Fred

    # ========================================
    # STEP 2: Get FRED API key
    # ========================================
    # Kaggle: try Kaggle Secrets first
    # Local: try .env via python-dotenv
    api_key = os.environ.get('FRED_API_KEY')
    if api_key is None:
        try:
            from kaggle_secrets import UserSecretsClient
            api_key = UserSecretsClient().get_secret("FRED_API_KEY")
        except Exception:
            # Local environment - try python-dotenv
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get('FRED_API_KEY')
            except ImportError:
                pass

    if api_key is None:
        raise RuntimeError(
            "FRED_API_KEY not found. "
            "Kaggle: add to Secrets. Local: add to .env file"
        )

    fred = Fred(api_key=api_key)

    # ========================================
    # STEP 3: Fetch raw data
    # ========================================
    START_DATE = "2014-01-01"  # 1 year extra for warmup

    print("Fetching yfinance data...")

    # GC=F: Gold futures (OHLCV)
    gc = yf.download('GC=F', start=START_DATE, progress=False, auto_adjust=True)
    if isinstance(gc.columns, pd.MultiIndex):
        gc.columns = [col[0] for col in gc.columns]
    gc = gc[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    gc.columns = [f'GC_{col}' for col in gc.columns]

    # GLD: Gold ETF (for volume features)
    gld = yf.download('GLD', start=START_DATE, progress=False, auto_adjust=True)
    if isinstance(gld.columns, pd.MultiIndex):
        gld.columns = [col[0] for col in gld.columns]
    gld = gld[['Volume']].copy()
    gld.columns = ['GLD_Volume']

    # SI=F: Silver futures
    si = yf.download('SI=F', start=START_DATE, progress=False, auto_adjust=True)
    if isinstance(si.columns, pd.MultiIndex):
        si.columns = [col[0] for col in si.columns]
    si = si[['Close']].copy()
    si.columns = ['SI_Close']

    # HG=F: Copper futures
    hg = yf.download('HG=F', start=START_DATE, progress=False, auto_adjust=True)
    if isinstance(hg.columns, pd.MultiIndex):
        hg.columns = [col[0] for col in hg.columns]
    hg = hg[['Close']].copy()
    hg.columns = ['HG_Close']

    # DX-Y.NYB: Dollar Index
    dxy = yf.download('DX-Y.NYB', start=START_DATE, progress=False, auto_adjust=True)
    if isinstance(dxy.columns, pd.MultiIndex):
        dxy.columns = [col[0] for col in dxy.columns]
    dxy = dxy[['Close']].copy()
    dxy.columns = ['DXY_Close']

    # ^GSPC: S&P 500
    spx = yf.download('^GSPC', start=START_DATE, progress=False, auto_adjust=True)
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = [col[0] for col in spx.columns]
    spx = spx[['Close']].copy()
    spx.columns = ['SPX_Close']

    print("Fetching FRED data...")

    # FRED series
    gvz = pd.DataFrame({'GVZ': fred.get_series('GVZCLS', observation_start=START_DATE)})
    vix = pd.DataFrame({'VIX': fred.get_series('VIXCLS', observation_start=START_DATE)})
    dfii10 = pd.DataFrame({'DFII10': fred.get_series('DFII10', observation_start=START_DATE)})
    dgs10 = pd.DataFrame({'DGS10': fred.get_series('DGS10', observation_start=START_DATE)})
    dgs2 = pd.DataFrame({'DGS2': fred.get_series('DGS2', observation_start=START_DATE)})

    # Convert index to datetime
    for df in [gvz, vix, dfii10, dgs10, dgs2]:
        df.index = pd.to_datetime(df.index)

    # ========================================
    # STEP 4: Merge all data
    # ========================================
    print("Merging data sources...")

    # Start with GC=F
    df = gc.copy()

    # Join all data sources
    for data in [gld, si, hg, dxy, spx, gvz, vix, dfii10, dgs10, dgs2]:
        df = df.join(data, how='left')

    # Forward-fill missing values
    # FRED: max 5 days
    fred_cols = ['GVZ', 'VIX', 'DFII10', 'DGS10', 'DGS2']
    for col in fred_cols:
        df[col] = df[col].ffill(limit=5)

    # yfinance: max 3 days
    yf_cols = [col for col in df.columns if col not in fred_cols]
    for col in yf_cols:
        df[col] = df[col].ffill(limit=3)

    # Drop any remaining NaN rows
    df = df.dropna()

    print(f"Merged data: {len(df)} rows, {len(df.columns)} columns")

    # ========================================
    # STEP 5: Compute features
    # ========================================
    print("Computing features...")

    # Helper function: rolling z-score
    def rolling_zscore(series, window=60):
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std.clip(lower=1e-8)

    # Helper function: rolling beta
    def rolling_beta(y, x, window=20):
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var()
        return cov / var.clip(lower=1e-8)

    # Gold return (for target and some features)
    df['gold_return'] = df['GC_Close'].pct_change() * 100

    # --- Category A: Volatility Regime Features (5) ---

    # A1: rv_ratio_10_30
    rv_10 = df['gold_return'].rolling(10).std()
    rv_30 = df['gold_return'].rolling(30).std()
    df['rv_ratio_10_30'] = rv_10 / rv_30.clip(lower=1e-8)

    # A2: rv_ratio_10_30_z
    df['rv_ratio_10_30_z'] = rolling_zscore(df['rv_ratio_10_30'], 60)

    # A3: gvz_level_z
    df['gvz_level_z'] = rolling_zscore(df['GVZ'], 60)

    # A4: gvz_vix_ratio
    df['gvz_vix_ratio'] = df['GVZ'] / df['VIX'].clip(lower=1e-8)

    # A5: intraday_range_ratio
    daily_range = (df['GC_High'] - df['GC_Low']) / df['GC_Close'].clip(lower=1e-8)
    avg_range = daily_range.rolling(20).mean()
    df['intraday_range_ratio'] = daily_range / avg_range.clip(lower=1e-8)

    # --- Category B: Cross-Asset Stress Features (4) ---

    # B1: risk_off_score (composite)
    vix_change = df['VIX'].pct_change() * 100
    dxy_change = df['DXY_Close'].pct_change() * 100
    spx_return = df['SPX_Close'].pct_change() * 100
    yield_change = df['DGS10'].diff()

    vix_z = rolling_zscore(vix_change, 20)
    dxy_z = rolling_zscore(dxy_change, 20)
    spx_z = rolling_zscore(spx_return, 20)
    yield_z = rolling_zscore(yield_change, 20)

    df['risk_off_score'] = vix_z + dxy_z - spx_z - yield_z

    # B2: gold_silver_ratio_change
    gold_5d_ret = df['GC_Close'].pct_change(5) * 100
    silver_5d_ret = df['SI_Close'].pct_change(5) * 100
    divergence = gold_5d_ret - silver_5d_ret
    df['gold_silver_ratio_change'] = rolling_zscore(divergence, 60)

    # B3: equity_gold_beta_20d
    df['equity_gold_beta_20d'] = rolling_beta(df['gold_return'], spx_return, 20)

    # B4: gold_copper_ratio_change
    copper_5d_ret = df['HG_Close'].pct_change(5) * 100
    divergence_copper = gold_5d_ret - copper_5d_ret
    df['gold_copper_ratio_change'] = rolling_zscore(divergence_copper, 60)

    # --- Category C: Rate and Currency Shock Features (3) ---

    # C1: rate_surprise (unsigned)
    rate_change = df['DFII10'].diff()
    rate_std_20 = rate_change.rolling(20).std()
    df['rate_surprise'] = np.abs(rate_change) / rate_std_20.clip(lower=1e-8)

    # C2: rate_surprise_signed
    df['rate_surprise_signed'] = np.sign(rate_change) * df['rate_surprise']

    # C3: dxy_acceleration
    dxy_accel = dxy_change - dxy_change.shift(1)
    df['dxy_acceleration'] = rolling_zscore(dxy_accel, 20)

    # --- Category D: Volume and Flow Features (2) ---

    # D1: gld_volume_z
    df['gld_volume_z'] = rolling_zscore(df['GLD_Volume'], 20)

    # D2: volume_return_sign
    df['volume_return_sign'] = np.sign(df['gold_return']) * df['gld_volume_z']

    # --- Category E: Momentum Context Features (2) ---

    # E1: momentum_divergence
    ret_5d = df['GC_Close'].pct_change(5) * 100
    ret_20d = df['GC_Close'].pct_change(20) * 100
    mom_div = ret_5d - ret_20d
    df['momentum_divergence'] = rolling_zscore(mom_div, 60)

    # E2: distance_from_20d_high
    high_20d = df['GC_Close'].rolling(20).max()
    low_20d = df['GC_Close'].rolling(20).min()
    range_20d = (high_20d - low_20d).clip(lower=1e-8)
    df['distance_from_20d_high'] = (df['GC_Close'] - high_20d) / range_20d

    # --- Category F: Calendar and Auxiliary (2) ---

    # F1: day_of_week (0=Monday, 4=Friday)
    df['day_of_week'] = df.index.dayofweek

    # F2: month_of_year (1-12)
    df['month_of_year'] = df.index.month

    # ========================================
    # STEP 6: Create target variable
    # ========================================
    # Target: 1=UP (next-day return > 0), 0=DOWN (next-day return <= 0)
    df['target'] = (df['gold_return'].shift(-1) > 0).astype(int)

    # ========================================
    # STEP 7: Select features and drop warmup rows
    # ========================================
    feature_cols = [
        'rv_ratio_10_30', 'rv_ratio_10_30_z', 'gvz_level_z', 'gvz_vix_ratio',
        'intraday_range_ratio', 'risk_off_score', 'gold_silver_ratio_change',
        'equity_gold_beta_20d', 'gold_copper_ratio_change', 'rate_surprise',
        'rate_surprise_signed', 'dxy_acceleration', 'gld_volume_z',
        'volume_return_sign', 'momentum_divergence', 'distance_from_20d_high',
        'day_of_week', 'month_of_year'
    ]

    # Keep only features + target
    df_final = df[feature_cols + ['target']].copy()

    # Drop rows with NaN (warmup period + last row where target is NaN)
    df_final = df_final.dropna()

    print(f"Final dataset: {len(df_final)} rows, {len(feature_cols)} features")

    # Verify target balance
    up_pct = 100 * (df_final['target'] == 1).sum() / len(df_final)
    down_pct = 100 * (df_final['target'] == 0).sum() / len(df_final)
    print(f"Target balance: UP={up_pct:.2f}%, DOWN={down_pct:.2f}%")

    # ========================================
    # STEP 8: Train/val/test split (70/15/15)
    # ========================================
    n = len(df_final)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df_final.iloc[:train_end].copy()
    val_df = df_final.iloc[train_end:val_end].copy()
    test_df = df_final.iloc[val_end:].copy()

    print(f"Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Val:   {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    print(f"Test:  {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")

    return train_df, val_df, test_df, df_final


# ========================================
# Test execution (for validation)
# ========================================
if __name__ == "__main__":
    print("Testing data fetching function...")
    train, val, test, full = fetch_and_preprocess()

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Total rows: {len(full)}")
    print(f"Features: {len(full.columns) - 1}")
    print(f"Train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print("\nFeature list:")
    for i, col in enumerate(full.columns[:-1], 1):
        print(f"  {i:2d}. {col}")
    print("\nSample data (first 3 rows):")
    print(full.head(3))
    print("\nData fetching function ready for builder_model!")
