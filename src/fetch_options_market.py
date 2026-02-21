"""
Data fetch: options_market (attempt 3)
Sources: Yahoo Finance ^SKEW (CBOE SKEW Index), ^GVZ (Gold Volatility Index)
builder_model embeds this code in train.ipynb for Kaggle execution.

Design: docs/design/options_market_3.md
No FRED API dependency. All data from Yahoo Finance.

Output columns:
  - skew_close: SKEW Index daily level
  - gvz_close: GVZ (Gold Vol Index) daily level
  - skew_change: SKEW daily first difference
  - gvz_change: GVZ daily first difference
  (HMM + EMA smoothing applied inside train.ipynb, not here)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def fetch_and_preprocess(start_date='2014-10-01', end_date=None):
    """
    Fetch SKEW and GVZ from Yahoo Finance, compute daily changes.

    Self-contained. No external file dependencies. No FRED API key needed.
    Compatible with both local and Kaggle environments.

    Parameters
    ----------
    start_date : str
        Start date for data fetch. Design spec: 2014-10-01.
    end_date : str or None
        End date. Defaults to today (dynamic).

    Returns
    -------
    train_df, val_df, test_df, full_df : pd.DataFrame
        Time-series split 70/15/15 (no shuffle).
        Columns: skew_close, gvz_close, skew_change, gvz_change
    """
    try:
        import yfinance as yf
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "yfinance"], check=True)
        import yfinance as yf

    from datetime import datetime

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"[fetch_options_market] Fetching data: {start_date} to {end_date}")

    # --- 1. Fetch SKEW Index ---
    print("  Fetching ^SKEW (CBOE SKEW Index)...")
    skew_raw = yf.download('^SKEW', start=start_date, end=end_date, progress=False)
    if skew_raw.empty:
        raise RuntimeError("^SKEW download returned empty DataFrame")
    if skew_raw.columns.nlevels > 1:
        skew_raw.columns = skew_raw.columns.droplevel(1)
    skew_close = skew_raw['Close'].rename('skew_close')
    skew_close.index = pd.to_datetime(skew_close.index).tz_localize(None)
    print(f"  ^SKEW: {len(skew_close)} rows, "
          f"{skew_close.index.min().date()} to {skew_close.index.max().date()}, "
          f"range [{skew_close.min():.2f}, {skew_close.max():.2f}]")

    # --- 2. Fetch GVZ (Gold Volatility Index) ---
    print("  Fetching ^GVZ (Gold Volatility Index)...")
    gvz_raw = yf.download('^GVZ', start=start_date, end=end_date, progress=False)
    if gvz_raw.empty:
        raise RuntimeError("^GVZ download returned empty DataFrame")
    if gvz_raw.columns.nlevels > 1:
        gvz_raw.columns = gvz_raw.columns.droplevel(1)
    gvz_close = gvz_raw['Close'].rename('gvz_close')
    gvz_close.index = pd.to_datetime(gvz_close.index).tz_localize(None)
    print(f"  ^GVZ: {len(gvz_close)} rows, "
          f"{gvz_close.index.min().date()} to {gvz_close.index.max().date()}, "
          f"range [{gvz_close.min():.2f}, {gvz_close.max():.2f}]")

    # --- 3. Inner join on common trading dates ---
    print("  Aligning on common dates (inner join)...")
    df = skew_close.to_frame().join(gvz_close.to_frame(), how='inner')
    print(f"  After inner join: {len(df)} rows, "
          f"{df.index.min().date()} to {df.index.max().date()}")

    # --- 4. Forward-fill gaps up to 3 days ---
    df = df.ffill(limit=3)

    # --- 5. Compute daily changes (first difference) ---
    df['skew_change'] = df['skew_close'].diff()
    df['gvz_change'] = df['gvz_close'].diff()

    # --- 6. Drop NaN from diff (first row) ---
    df = df.dropna(subset=['skew_change', 'gvz_change'])

    # --- 7. Data quality assertions ---
    assert len(df) > 2000, f"Expected >2000 rows, got {len(df)}"
    assert df['skew_close'].between(100, 200).all(), "SKEW out of expected [100,200] range"
    assert df['gvz_close'].between(5, 80).all(), "GVZ out of expected [5,80] range"
    assert (df['skew_change'].abs() < 30).all(), "SKEW change outlier >= 30"
    assert (df['gvz_change'].abs() < 20).all(), "GVZ change outlier >= 20"
    missing_pct = df[['skew_close', 'gvz_close']].isna().mean().max()
    assert missing_pct < 0.02, f"Missing data {missing_pct:.2%} exceeds 2%"

    print(f"\n  Final dataset: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Statistics
    print("\n=== SKEW Statistics ===")
    print(f"  Mean: {df['skew_close'].mean():.2f}")
    print(f"  Std:  {df['skew_close'].std():.2f}")
    print(f"  Range: [{df['skew_close'].min():.2f}, {df['skew_close'].max():.2f}]")
    print(f"  Autocorr(1): {df['skew_close'].autocorr(lag=1):.4f}")
    print(f"  Change std: {df['skew_change'].std():.3f}")

    print("\n=== GVZ Statistics ===")
    print(f"  Mean: {df['gvz_close'].mean():.2f}")
    print(f"  Std:  {df['gvz_close'].std():.2f}")
    print(f"  Range: [{df['gvz_close'].min():.2f}, {df['gvz_close'].max():.2f}]")
    print(f"  Autocorr(1): {df['gvz_close'].autocorr(lag=1):.4f}")
    print(f"  Change std: {df['gvz_change'].std():.3f}")

    corr = df['skew_change'].corr(df['gvz_change'])
    vif = 1.0 / (1 - corr ** 2)
    print(f"\n=== Orthogonality check ===")
    print(f"  SKEW-GVZ change correlation: {corr:.4f}  (design: <0.8)")
    print(f"  VIF (SKEW-GVZ changes): {vif:.4f}  (design: <10)")

    print("\n=== Data Quality Checks ===")
    print(f"  SKEW range [100,200]: {'OK' if df['skew_close'].between(100,200).all() else 'WARN'}")
    print(f"  GVZ range [5,80]: {'OK' if df['gvz_close'].between(5,80).all() else 'WARN'}")
    print(f"  No SKEW outliers (|chg|<30): {'OK' if (df['skew_change'].abs()<30).all() else 'WARN'}")
    print(f"  No GVZ outliers (|chg|<20): {'OK' if (df['gvz_change'].abs()<20).all() else 'WARN'}")
    print(f"  Missing data: {df[['skew_close','gvz_close']].isna().mean().max()*100:.2f}%  (design: <2%)")

    # --- 8. Train/val/test split (70/15/15, time-series order, no shuffle) ---
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\n=== Data Split ===")
    print(f"  Train: {len(train_df)} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"  Val:   {len(val_df)} rows ({val_df.index.min().date()} to {val_df.index.max().date()})")
    print(f"  Test:  {len(test_df)} rows ({test_df.index.min().date()} to {test_df.index.max().date()})")

    return train_df, val_df, test_df, df


if __name__ == "__main__":
    train_df, val_df, test_df, full_df = fetch_and_preprocess()

    print("\n=== Fetch Complete ===")
    print(f"Full dataset shape: {full_df.shape}")
    print(f"\nFirst 5 rows:\n{full_df.head()}")
    print(f"\nLast 5 rows:\n{full_df.tail()}")

    # Save to data/processed/options_market/
    from pathlib import Path
    out_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'options_market'
    out_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_dir / 'options_market_features_input.csv')
    full_df.to_csv(out_dir / 'data.csv')
    print(f"\nSaved to {out_dir}/")
