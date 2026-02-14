"""
Validate specific FRED series for multi-country analysis
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fredapi import Fred
import pandas as pd

# Direct API key loading
FRED_API_KEY = os.environ.get('FRED_API_KEY')
if not FRED_API_KEY:
    with open('.env', 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('FRED_API_KEY='):
                FRED_API_KEY = line.strip().split('=')[1]
                break

fred = Fred(api_key=FRED_API_KEY)

# Test series
test_series = {
    'US_real': 'DFII10',
    'US_nominal': 'IRLTLT01USM156N',
    'EU_nominal': 'IRLTLT01EZM156N',
    'DE_nominal': 'IRLTLT01DEM156N',
    'JP_nominal': 'IRLTLT01JPM156N',
    'UK_nominal': 'IRLTLT01GBM156N',
    'CA_nominal': 'IRLTLT01CAM156N',
    'CH_nominal': 'IRLTLT01CHM156N',
    'AU_nominal': 'IRLTLT01AUM156N',
    'NZ_nominal': 'IRLTLT01NZM156N',
    'NO_nominal': 'IRLTLT01NOM156N',
    'SE_nominal': 'IRLTLT01SEM156N',
}

print("=" * 80)
print("MULTI-COUNTRY DATA VALIDATION")
print("=" * 80)

results = []
all_data = {}

for name, series_id in test_series.items():
    try:
        data = fred.get_series(series_id)
        if data is not None and len(data) > 0:
            all_data[name] = data
            results.append({
                'name': name,
                'series_id': series_id,
                'start': str(data.index.min().date()),
                'end': str(data.index.max().date()),
                'count': len(data),
                'missing_pct': round(data.isna().sum() / len(data) * 100, 2),
                'latest_value': round(data.dropna().iloc[-1], 3) if len(data.dropna()) > 0 else None,
                'status': 'OK'
            })
            print(f"OK {name:15s}: {data.index.min().date()} to {data.index.max().date()} ({len(data)} obs, {round(data.isna().sum() / len(data) * 100, 1)}% missing)")
        else:
            results.append({'name': name, 'series_id': series_id, 'status': 'EMPTY'})
            print(f"X {name:15s}: No data")
    except Exception as e:
        results.append({'name': name, 'series_id': series_id, 'status': f'ERROR: {str(e)}'})
        print(f"X {name:15s}: {str(e)}")

# Save
df_validation = pd.DataFrame(results)
df_validation.to_csv('validation_multi_country.csv', index=False)
print(f"\nOK Saved to validation_multi_country.csv")

# Common date range
if all_data:
    print("\n" + "=" * 80)
    print("COMMON DATE RANGE ANALYSIS")
    print("=" * 80)

    start_dates = [data.index.min() for data in all_data.values()]
    end_dates = [data.index.max() for data in all_data.values()]
    common_start = max(start_dates)
    common_end = min(end_dates)

    print(f"Common range: {common_start.date()} to {common_end.date()}")
    total_months = (common_end.year - common_start.year) * 12 + (common_end.month - common_start.month)
    print(f"Total months: {total_months}")

    print(f"\nObservations in common range:")
    total_samples = 0
    for name, data in all_data.items():
        common_data = data[(data.index >= common_start) & (data.index <= common_end)]
        print(f"  {name:15s}: {len(common_data):4d}")
        total_samples += len(common_data)

    print(f"\n** Total samples (all countries): {total_samples:,}")
    print(f"** Countries available: {len(all_data)}")
