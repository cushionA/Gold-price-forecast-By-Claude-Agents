"""
Data Fetching: yield_curve
builder_model will embed this code into train.py for Kaggle execution
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

def fetch_and_preprocess():
    """
    Fetch yield curve data from FRED (DGS10, DGS2, DGS5).
    Self-contained. No external file dependencies.

    Returns:
        tuple: (train_df, val_df, test_df, full_df)
    """
    # --- FRED API Setup ---
    try:
        from fredapi import Fred
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "fredapi"], check=True)
        from fredapi import Fred

    # Credential handling: .env locally, Kaggle Secrets on Kaggle
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Kaggle doesn't need dotenv

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

    # --- Data Fetching ---
    print("Fetching DGS10, DGS2, DGS5 from FRED...")
    # Start from 2014-10-01 for warmup buffer (60+ trading days before 2015-01-30)
    start_date = '2014-10-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    dgs10 = fred.get_series('DGS10', observation_start=start_date, observation_end=end_date)
    dgs2 = fred.get_series('DGS2', observation_start=start_date, observation_end=end_date)
    dgs5 = fred.get_series('DGS5', observation_start=start_date, observation_end=end_date)

    # --- Combine into DataFrame ---
    df = pd.DataFrame({
        'dgs10': dgs10,
        'dgs2': dgs2,
        'dgs5': dgs5
    })
    df.index.name = 'date'
    df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'])

    print(f"Raw data fetched: {len(df)} rows from {df['date'].min()} to {df['date'].max()}")

    # --- Handle Missing Values ---
    # Drop rows where any series is NaN (inner join approach)
    initial_rows = len(df)
    df = df.dropna(subset=['dgs10', 'dgs2', 'dgs5'])
    print(f"After dropping NaN: {len(df)} rows (dropped {initial_rows - len(df)} rows)")

    # --- Compute Derived Features ---
    # 1. Daily changes
    df['dgs10_change'] = df['dgs10'].diff()
    df['dgs2_change'] = df['dgs2'].diff()

    # 2. Spread: DGS10 - DGS2
    df['spread'] = df['dgs10'] - df['dgs2']
    df['spread_change'] = df['spread'].diff()

    # 3. Curvature: DGS5 - 0.5*(DGS2 + DGS10)
    df['curvature_raw'] = df['dgs5'] - 0.5 * (df['dgs2'] + df['dgs10'])
    df['curvature_change'] = df['curvature_raw'].diff()

    # --- Forward-fill small gaps (max 3 days) ---
    # Only for change columns that now have NaN
    change_cols = ['dgs10_change', 'dgs2_change', 'spread_change', 'curvature_change']
    for col in change_cols:
        df[col] = df[col].ffill(limit=3)

    # Drop remaining NaN rows (first row after diff)
    df = df.dropna(subset=change_cols)

    print(f"After preprocessing: {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nBasic statistics:")
    print(df[['dgs10', 'dgs2', 'dgs5', 'spread', 'curvature_raw']].describe())

    # --- Data Validation ---
    # Check yields are in reasonable range
    for col in ['dgs10', 'dgs2', 'dgs5']:
        if (df[col] < 0).any() or (df[col] > 20).any():
            print(f"WARNING: {col} has values outside [0, 20]% range")

    # Check spread is mostly positive (some inversions are OK)
    inversion_pct = (df['spread'] < 0).sum() / len(df) * 100
    print(f"Yield curve inversion: {inversion_pct:.1f}% of days")

    # --- Train/Val/Test Split (70/15/15, time-series order) ---
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nData split:")
    print(f"  Train: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"  Test:  {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

    return train_df, val_df, test_df, df


if __name__ == "__main__":
    # Test locally
    train_df, val_df, test_df, full_df = fetch_and_preprocess()

    # Save to data/raw/ for datachecker
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'yield_curve.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Columns: {list(full_df.columns)}")
