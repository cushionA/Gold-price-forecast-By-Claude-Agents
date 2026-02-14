"""
Research Script: Multi-Country Real Interest Rate Data Availability
Explores FRED and other sources for G10 real interest rates
"""

import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd
import json
from datetime import datetime

# Load environment
load_dotenv()
fred = Fred(api_key=os.environ['FRED_API_KEY'])

# Target countries
COUNTRIES = {
    'US': ['United States', 'USA', 'U.S.'],
    'EU': ['Euro', 'Germany', 'European'],
    'JP': ['Japan', 'Japanese'],
    'UK': ['United Kingdom', 'Britain', 'UK'],
    'CA': ['Canada', 'Canadian'],
    'CH': ['Switzerland', 'Swiss'],
    'AU': ['Australia', 'Australian'],
    'NZ': ['New Zealand'],
    'NO': ['Norway', 'Norwegian'],
    'SE': ['Sweden', 'Swedish']
}

# Search keywords
KEYWORDS = [
    'real yield',
    'real interest rate',
    'inflation indexed',
    'TIPS',
    'inflation linked',
    'linker',
    'index linked'
]

def search_fred_series(country_code, country_names, keywords):
    """Search FRED for real rate series for a country"""
    results = []

    for keyword in keywords:
        for country_name in country_names:
            search_term = f"{country_name} {keyword}"
            try:
                search_results = fred.search(search_term)
                if not search_results.empty:
                    # Filter for relevant series
                    for idx, row in search_results.iterrows():
                        title = row.get('title', '').lower()
                        notes = str(row.get('notes', '')).lower()

                        # Check if it's about real rates/yields
                        if any(kw in title or kw in notes for kw in ['real', 'inflation', 'tips', 'index']):
                            results.append({
                                'country': country_code,
                                'series_id': idx,
                                'title': row.get('title', ''),
                                'frequency': row.get('frequency', ''),
                                'units': row.get('units', ''),
                                'seasonal_adjustment': row.get('seasonal_adjustment', ''),
                                'last_updated': row.get('last_updated', ''),
                                'notes': str(row.get('notes', ''))[:200]
                            })
            except Exception as e:
                print(f"Error searching {search_term}: {e}")

    return results

def search_nominal_yields():
    """Search for 10Y nominal government bond yields"""
    nominal_keywords = [
        '10 year government bond',
        '10-year treasury',
        'long term interest rate'
    ]

    results = []
    for country_code, country_names in COUNTRIES.items():
        for keyword in nominal_keywords:
            for country_name in country_names:
                search_term = f"{country_name} {keyword}"
                try:
                    search_results = fred.search(search_term)
                    if not search_results.empty:
                        for idx, row in search_results.iterrows():
                            title = row.get('title', '').lower()
                            if '10' in title and ('year' in title or 'yr' in title):
                                results.append({
                                    'country': country_code,
                                    'series_id': idx,
                                    'title': row.get('title', ''),
                                    'type': 'nominal_yield',
                                    'frequency': row.get('frequency', ''),
                                    'last_updated': row.get('last_updated', '')
                                })
                except Exception as e:
                    print(f"Error searching {search_term}: {e}")

    return results

def search_inflation_rates():
    """Search for CPI/inflation data"""
    inflation_keywords = [
        'consumer price index',
        'CPI',
        'inflation rate'
    ]

    results = []
    for country_code, country_names in COUNTRIES.items():
        for keyword in inflation_keywords:
            for country_name in country_names:
                search_term = f"{country_name} {keyword}"
                try:
                    search_results = fred.search(search_term)
                    if not search_results.empty:
                        for idx, row in search_results.iterrows():
                            title = row.get('title', '').lower()
                            if 'cpi' in title or 'inflation' in title:
                                results.append({
                                    'country': country_code,
                                    'series_id': idx,
                                    'title': row.get('title', ''),
                                    'type': 'inflation',
                                    'frequency': row.get('frequency', ''),
                                    'last_updated': row.get('last_updated', '')
                                })
                except Exception as e:
                    print(f"Error searching {search_term}: {e}")

    return results

def validate_series(series_id):
    """Fetch sample data to validate availability"""
    try:
        data = fred.get_series(series_id)
        if data is not None and len(data) > 0:
            return {
                'available': True,
                'start_date': str(data.index.min()),
                'end_date': str(data.index.max()),
                'count': len(data),
                'missing_pct': (data.isna().sum() / len(data) * 100),
                'recent_value': float(data.iloc[-1]) if not pd.isna(data.iloc[-1]) else None
            }
    except Exception as e:
        return {'available': False, 'error': str(e)}

    return {'available': False}

if __name__ == '__main__':
    print("=" * 80)
    print("MULTI-COUNTRY REAL INTEREST RATE DATA RESEARCH")
    print("=" * 80)

    # 1. Search for direct real rates
    print("\n[1] Searching for DIRECT real interest rates...")
    all_real_rates = []
    for country_code, country_names in COUNTRIES.items():
        print(f"\n  Searching {country_code}...")
        results = search_fred_series(country_code, country_names, KEYWORDS)
        all_real_rates.extend(results)
        print(f"    Found {len(results)} potential series")

    # Save results
    df_real = pd.DataFrame(all_real_rates)
    if not df_real.empty:
        df_real.to_csv('research_real_rates.csv', index=False)
        print(f"\n  Total direct real rate series found: {len(df_real)}")
        print(f"  Saved to: research_real_rates.csv")

    # 2. Search for nominal yields
    print("\n[2] Searching for NOMINAL 10Y government bond yields...")
    nominal_results = search_nominal_yields()
    df_nominal = pd.DataFrame(nominal_results)
    if not df_nominal.empty:
        df_nominal.to_csv('research_nominal_yields.csv', index=False)
        print(f"  Total nominal yield series found: {len(df_nominal)}")
        print(f"  Saved to: research_nominal_yields.csv")

    # 3. Search for inflation rates
    print("\n[3] Searching for INFLATION/CPI data...")
    inflation_results = search_inflation_rates()
    df_inflation = pd.DataFrame(inflation_results)
    if not df_inflation.empty:
        df_inflation.to_csv('research_inflation_rates.csv', index=False)
        print(f"  Total inflation series found: {len(df_inflation)}")
        print(f"  Saved to: research_inflation_rates.csv")

    # 4. Validate top candidates
    print("\n[4] Validating TOP candidate series...")

    # Known good series
    validation_targets = {
        'US_real': 'DFII10',  # US 10Y TIPS
        'US_nominal': 'DGS10',  # US 10Y Treasury
        'DE_nominal': 'IRLTLT01DEM156N',  # Germany 10Y (OECD)
        'JP_nominal': 'IRLTLT01JPM156N',  # Japan 10Y (OECD)
        'UK_nominal': 'IRLTLT01GBM156N',  # UK 10Y (OECD)
        'CA_nominal': 'IRLTLT01CAM156N',  # Canada 10Y (OECD)
    }

    validation_results = {}
    for name, series_id in validation_targets.items():
        print(f"\n  Validating {name} ({series_id})...")
        result = validate_series(series_id)
        validation_results[name] = result
        if result.get('available'):
            print(f"    ✓ Available: {result['start_date']} to {result['end_date']}")
            print(f"      Count: {result['count']}, Missing: {result['missing_pct']:.1f}%")
        else:
            print(f"    ✗ Not available: {result.get('error', 'Unknown error')}")

    # Save validation results
    with open('research_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("Research complete. Check generated CSV files for details.")
    print("=" * 80)
