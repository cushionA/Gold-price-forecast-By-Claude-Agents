"""Debug script to check data alignment issues"""

import pandas as pd
from fredapi import Fred
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get('FRED_API_KEY')
fred = Fred(api_key=api_key)

# Fetch US TIPS and Germany data
print("Fetching US TIPS...")
us_tips = fred.get_series('DFII10', observation_start='2003-01-01')
us_tips_monthly = us_tips.resample('ME').last()

print(f"US TIPS monthly: {len(us_tips_monthly)} rows")
print(f"Date range: {us_tips_monthly.index.min()} to {us_tips_monthly.index.max()}")
print(f"First 5 dates:")
print(us_tips_monthly.head())

print("\nFetching Germany nominal...")
de_nominal = fred.get_series('IRLTLT01DEM156N', observation_start='2003-01-01')
print(f"Germany nominal: {len(de_nominal)} rows")
print(f"Date range: {de_nominal.index.min()} to {de_nominal.index.max()}")
print(f"First 5 dates:")
print(de_nominal.head())

print("\nFetching Germany CPI...")
de_cpi = fred.get_series('CPALTT01DEM659N', observation_start='2003-01-01')
print(f"Germany CPI: {len(de_cpi)} rows")
print(f"Date range: {de_cpi.index.min()} to {de_cpi.index.max()}")
print(f"First 5 dates:")
print(de_cpi.head())

# Try to combine
print("\n\nCombining data...")
df = pd.DataFrame(index=us_tips_monthly.index)
df['us_tips'] = us_tips_monthly
df['de_nominal'] = de_nominal
df['de_cpi'] = de_cpi

print(f"Combined shape: {df.shape}")
print(f"NaN counts: {df.isna().sum()}")
print(f"\nFirst 10 rows:")
print(df.head(10))
