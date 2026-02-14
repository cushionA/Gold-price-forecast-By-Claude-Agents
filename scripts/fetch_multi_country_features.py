"""
Fetch multi-country real rate features for Attempt 3
25 features: US TIPS + 6 countries (nominal + CPI) + aggregates + VIX

Data sources:
- US TIPS: FRED DFII10 (daily, resample to month-start)
- 6 countries nominal yields: FRED OECD series (monthly)
  Germany, UK, Canada, Switzerland, Norway, Sweden
  (Japan excluded: CPI data ends 2021-06, too early)
- 6 countries CPI YoY: FRED OECD series (monthly)
- VIX: FRED VIXCLS (daily, resample to month-start)

Output: data/processed/real_rate_multi_country_features.csv
Expected shape: (~265 months, 25 features)
"""

import pandas as pd
import numpy as np
from fredapi import Fred
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_multi_country_features():
    """
    Fetch and process multi-country interest rate features.

    Returns:
        pd.DataFrame: Monthly features with DatetimeIndex
    """

    # Initialize FRED API
    api_key = os.environ.get('FRED_API_KEY')
    if api_key is None:
        raise RuntimeError("FRED_API_KEY not found in environment variables")

    fred = Fred(api_key=api_key)

    print("Fetching data from FRED...")
    print("=" * 60)

    # === 1. Fetch US TIPS (daily) ===
    print("1/9: Fetching US TIPS (DFII10)...")
    us_tips = fred.get_series('DFII10', observation_start='2003-01-01')
    print(f"  -> US TIPS: {len(us_tips)} daily observations")

    # === 2. Fetch 6 countries nominal yields + CPI ===
    # Note: Japan excluded due to CPI data ending in 2021-06
    countries = {
        'germany': ('IRLTLT01DEM156N', 'CPALTT01DEM659N'),
        'uk': ('IRLTLT01GBM156N', 'CPALTT01GBM659N'),
        'canada': ('IRLTLT01CAM156N', 'CPALTT01CAM659N'),
        'switzerland': ('IRLTLT01CHM156N', 'CPALTT01CHM659N'),
        'norway': ('IRLTLT01NOM156N', 'CPALTT01NOM659N'),
        'sweden': ('IRLTLT01SEM156N', 'CPALTT01SEM659N')
    }

    country_data = {}
    for i, (country, (nominal_id, cpi_id)) in enumerate(countries.items(), start=2):
        print(f"{i}/9: Fetching {country.upper()} nominal yield + CPI...")

        try:
            nominal = fred.get_series(nominal_id, observation_start='2003-01-01')
            cpi = fred.get_series(cpi_id, observation_start='2003-01-01')

            country_data[country] = {
                'nominal': nominal,
                'cpi': cpi
            }

            print(f"  -> {country}: {len(nominal)} nominal, {len(cpi)} CPI observations")
        except Exception as e:
            print(f"  WARNING: Failed to fetch {country} data: {e}")
            # Skip this country if data unavailable
            continue

    # === 3. Fetch VIX (daily) ===
    print("9/9: Fetching VIX (VIXCLS)...")
    vix = fred.get_series('VIXCLS', observation_start='2003-01-01')
    print(f"  -> VIX: {len(vix)} daily observations")

    print("\n" + "=" * 60)
    print("Data fetching complete. Processing features...")
    print("=" * 60 + "\n")

    # === 4. Resample to monthly (month-START for alignment) ===
    print("Step 1: Resampling to month-start...")

    # US TIPS: use last value of each month, but align to month-start
    us_tips_monthly = us_tips.resample('MS').last()
    print(f"  -> US TIPS monthly: {len(us_tips_monthly)} months")

    # VIX: use mean of each month, align to month-start
    vix_monthly = vix.resample('MS').mean()
    print(f"  -> VIX monthly: {len(vix_monthly)} months")

    # Country data is already monthly at month-start (no resampling needed)

    # === 5. Create feature DataFrame ===
    print("\nStep 2: Computing features...")

    # Create common monthly index (month-start)
    # Use the intersection of all available dates
    all_dates = us_tips_monthly.index

    # Start with US TIPS
    df = pd.DataFrame(index=all_dates)
    df['us_tips_level'] = us_tips_monthly
    df['us_tips_change'] = us_tips_monthly.diff()

    print(f"  [OK] US TIPS features (2)")

    # Add country features
    country_count = 0
    for country, data in country_data.items():
        nominal = data['nominal']
        cpi = data['cpi']

        # Align to monthly index
        df[f'{country}_nominal_level'] = nominal
        df[f'{country}_nominal_change'] = nominal.diff()

        # CPI YoY with 1-month lag
        df[f'{country}_cpi_lagged'] = cpi.shift(1)

        country_count += 1

    print(f"  [OK] Country features ({country_count} countries x 3 = {country_count * 3})")

    # === 6. Compute cross-country aggregates ===
    print("\nStep 3: Computing cross-country aggregates...")

    # Get all country nominal levels and changes
    country_list = list(country_data.keys())
    level_cols = [f'{c}_nominal_level' for c in country_list]
    change_cols = [f'{c}_nominal_change' for c in country_list]
    cpi_cols = [f'{c}_cpi_lagged' for c in country_list]

    # Yield dispersion (std across countries)
    df['yield_dispersion'] = df[level_cols].std(axis=1)

    # Yield change dispersion
    df['yield_change_dispersion'] = df[change_cols].std(axis=1)

    # Mean CPI change
    df['mean_cpi_change'] = df[cpi_cols].mean(axis=1)

    # US vs global spread
    df['us_vs_global_spread'] = df['us_tips_level'] - df[level_cols].mean(axis=1)

    print(f"  [OK] Aggregate features (4)")

    # === 7. Add VIX ===
    df['vix_monthly'] = vix_monthly
    print(f"  [OK] VIX feature (1)")

    # === 8. Handle missing data ===
    print("\nStep 4: Handling missing data...")

    print(f"  Before cleaning: {df.shape[0]} rows, {df.isna().sum().sum()} NaN values")

    # Forward-fill with max 3-month limit
    df = df.ffill(limit=3)

    print(f"  After forward-fill: {df.isna().sum().sum()} NaN values")

    # Drop any remaining NaN rows
    df = df.dropna()

    print(f"  After dropping NaN: {df.shape[0]} rows, {df.shape[1]} columns")

    # === 9. Final validation ===
    print("\nStep 5: Final validation...")

    # Check column count
    # 2 (US TIPS) + (6 countries x 3) + 4 (aggregates) + 1 (VIX) = 25 features
    expected_cols = 2 + (len(country_data) * 3) + 4 + 1
    actual_cols = len(df.columns)

    print(f"  Expected columns: {expected_cols}")
    print(f"  Actual columns: {actual_cols}")

    if actual_cols != expected_cols:
        print(f"  WARNING: Column count mismatch!")

    # Check for any remaining NaN
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values remain!")
    else:
        print(f"  [OK] No NaN values")

    # Check date range
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Total months: {len(df)}")

    return df


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Multi-Country Real Rate Features - Data Fetching")
    print("=" * 60 + "\n")

    try:
        # Fetch and process data
        df = fetch_multi_country_features()

        print("\n" + "=" * 60)
        print("Data Processing Complete")
        print("=" * 60 + "\n")

        # === Quality Checks ===
        print("QUALITY CHECKS")
        print("=" * 60)

        print(f"\n1. Shape: {df.shape}")
        print(f"   Expected: (~265 rows, 25 columns)")

        if df.shape[0] < 250:
            print(f"   WARNING: Row count is lower than expected")
        elif df.shape[0] > 275:
            print(f"   WARNING: Row count is higher than expected")
        else:
            print(f"   [OK] Row count is within expected range")

        if df.shape[1] != 25:
            print(f"   WARNING: Column count is {df.shape[1]}, expected 25")
        else:
            print(f"   [OK] Column count is correct (25)")

        print(f"\n2. Date Range:")
        print(f"   First: {df.index.min()}")
        print(f"   Last: {df.index.max()}")
        print(f"   Total: {len(df)} months")

        print(f"\n3. Missing Values:")
        nan_count = df.isna().sum().sum()
        print(f"   Total NaN: {nan_count}")
        if nan_count > 0:
            print(f"   WARNING: NaN values found!")
        else:
            print(f"   [OK] No NaN values")

        print(f"\n4. First 3 Rows:")
        print(df.head(3))

        print(f"\n5. Last 3 Rows:")
        print(df.tail(3))

        print(f"\n6. Descriptive Statistics:")
        print(df.describe())

        # Sanity checks
        print(f"\n7. Sanity Checks:")

        us_tips_mean = df['us_tips_level'].mean()
        print(f"   US TIPS mean: {us_tips_mean:.3f}%")
        if 0.0 < us_tips_mean < 3.0:
            print(f"   [OK] US TIPS mean is reasonable (0-3%)")
        else:
            print(f"   WARNING: US TIPS mean is outside expected range")

        # Check nominal yields
        nominal_cols = [c for c in df.columns if '_nominal_level' in c]
        for col in nominal_cols:
            mean_val = df[col].mean()
            if 0.0 < mean_val < 6.0:
                status = "[OK]"
            else:
                status = "[!!]"
            print(f"   {status} {col}: mean = {mean_val:.3f}%")

        vix_mean = df['vix_monthly'].mean()
        print(f"   VIX mean: {vix_mean:.2f}")
        if 10 < vix_mean < 30:
            print(f"   [OK] VIX mean is reasonable (10-30)")
        else:
            print(f"   WARNING: VIX mean is outside expected range")

        # === Save Output ===
        print("\n" + "=" * 60)
        print("Saving Output")
        print("=" * 60 + "\n")

        output_path = "C:\\Users\\tatuk\\Desktop\\Gold-price-forecast-By-Claude-Agents\\data\\processed\\real_rate_multi_country_features.csv"
        df.to_csv(output_path)

        print(f"[OK] Saved: {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")

        print("\n" + "=" * 60)
        print("SUCCESS: Multi-country features fetched and saved")
        print("=" * 60 + "\n")

    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Data fetching failed")
        print("=" * 60)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
