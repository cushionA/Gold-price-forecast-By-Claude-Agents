"""
Data Fetching: VIX
Builder_model will embed this code in train.py for Kaggle execution.
"""
import pandas as pd
import numpy as np
import os

def fetch_and_preprocess():
    """
    Self-contained VIX data fetching and preprocessing.
    Returns: (train_df, val_df, test_df, full_df)

    Data sources:
    - FRED: VIXCLS (primary)
    - Yahoo Finance: ^VIX (fallback)

    Preprocessing:
    1. Fetch VIX from FRED (or Yahoo as backup)
    2. Compute log-returns
    3. Handle missing values (forward-fill up to 3 days)
    4. Align to date range 2014-10-01 to 2025-02-12 (buffer for warmup)
    """

    # --- FRED API Setup ---
    try:
        from fredapi import Fred
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "fredapi"], check=True)
        from fredapi import Fred

    # Get API key (local .env or Kaggle Secrets)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Kaggle environment doesn't need dotenv

    api_key = os.environ.get('FRED_API_KEY')
    if api_key is None:
        try:
            from kaggle_secrets import UserSecretsClient
            api_key = UserSecretsClient().get_secret("FRED_API_KEY")
        except Exception:
            raise RuntimeError(
                "FRED_API_KEY not found. "
                "Local: Set in .env / Kaggle: Register in Secrets"
            )

    # --- Fetch VIX Data ---
    fred = Fred(api_key=api_key)

    try:
        # Primary: FRED VIXCLS
        vix_series = fred.get_series('VIXCLS', observation_start='2014-10-01')
        df = pd.DataFrame({'vix': vix_series})
        df.index.name = 'date'
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['date'])
        source = 'FRED:VIXCLS'

    except Exception as e:
        print(f"FRED fetch failed: {e}")
        print("Falling back to Yahoo Finance ^VIX")

        # Fallback: Yahoo Finance
        try:
            import yfinance as yf
        except ImportError:
            import subprocess
            subprocess.run(["pip", "install", "yfinance"], check=True)
            import yfinance as yf

        vix_data = yf.download('^VIX', start='2014-10-01', progress=False)
        df = pd.DataFrame({
            'date': vix_data.index,
            'vix': vix_data['Close'].values
        })
        df['date'] = pd.to_datetime(df['date'])
        source = 'Yahoo:^VIX'

    # --- Preprocessing ---

    # 1. Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # 2. Handle missing values (forward-fill up to 3 days)
    df['vix'] = df['vix'].ffill(limit=3)

    # 3. Drop remaining NaN values
    missing_count = df['vix'].isna().sum()
    if missing_count > 0:
        print(f"Warning: Dropping {missing_count} rows with missing VIX values after forward-fill")
        df = df.dropna(subset=['vix'])

    # 4. Compute log-changes
    df['vix_log_change'] = np.log(df['vix']) - np.log(df['vix'].shift(1))

    # 5. Verify date range
    print(f"VIX data fetched from {source}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total rows: {len(df)}")
    print(f"VIX range: {df['vix'].min():.2f} to {df['vix'].max():.2f}")

    # 6. Check for extreme outliers in log-changes
    extreme_mask = df['vix_log_change'].abs() > 0.5  # >65% daily move
    if extreme_mask.sum() > 0:
        print(f"Warning: {extreme_mask.sum()} days with extreme VIX log-changes (>0.5)")
        print(df[extreme_mask][['date', 'vix', 'vix_log_change']])

    # 7. Basic statistics
    print("\nVIX Statistics:")
    print(df['vix'].describe())
    print("\nVIX Log-Change Statistics:")
    print(df['vix_log_change'].describe())

    # --- Train/Val/Test Split (70/15/15, time-series order) ---
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nTrain: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Val:   {len(val_df)} rows ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

    return train_df, val_df, test_df, df

if __name__ == "__main__":
    # Test the function
    train_df, val_df, test_df, full_df = fetch_and_preprocess()
    print("\n✓ VIX data fetching complete")
    print(f"Full dataset shape: {full_df.shape}")
    print(f"Columns: {list(full_df.columns)}")
