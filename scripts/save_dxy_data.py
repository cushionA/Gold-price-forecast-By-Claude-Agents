"""
Save DXY and constituent currency data to appropriate directories
"""
import pandas as pd
import json
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fetch_dxy import fetch_and_preprocess


def main():
    # Fetch data
    print("Fetching DXY and currency data...")
    raw_prices, dxy_currencies = fetch_and_preprocess()

    # Load base_features to get date range
    base_features = pd.read_csv('data/processed/base_features.csv', parse_dates=['Date'], index_col='Date')
    print(f"\nBase features date range: {base_features.index[0]} to {base_features.index[-1]}")
    print(f"Base features rows: {len(base_features)}")

    # Align with base_features date range
    start_date = base_features.index[0]
    end_date = base_features.index[-1]

    # Filter to base_features date range
    dxy_aligned = dxy_currencies[(dxy_currencies.index >= start_date) & (dxy_currencies.index <= end_date)]
    raw_aligned = raw_prices[(raw_prices.index >= start_date) & (raw_prices.index <= end_date)]

    print(f"\nAligned DXY data: {len(dxy_aligned)} rows from {dxy_aligned.index[0]} to {dxy_aligned.index[-1]}")

    # === Save raw prices ===
    raw_output_path = 'data/raw/dxy_currencies.csv'
    raw_aligned.to_csv(raw_output_path)
    print(f"\nSaved raw currency prices to: {raw_output_path}")
    print(f"  Rows: {len(raw_aligned)}")
    print(f"  Columns: {list(raw_aligned.columns)}")

    # === Save processed data ===
    processed_output_path = 'data/multi_country/dxy_constituents.csv'
    dxy_aligned.to_csv(processed_output_path)
    print(f"\nSaved processed currency data to: {processed_output_path}")
    print(f"  Rows: {len(dxy_aligned)}")
    print(f"  Columns: {list(dxy_aligned.columns)}")

    # === Create metadata ===
    metadata = {
        "feature": "dxy",
        "attempt": 1,
        "created_at": datetime.now().isoformat(),
        "sources": [
            "Yahoo:DX-Y.NYB (DXY Index)",
            "Yahoo:EURUSD=X",
            "Yahoo:JPY=X",
            "Yahoo:GBPUSD=X",
            "Yahoo:CAD=X",
            "Yahoo:SEK=X",
            "Yahoo:CHF=X"
        ],
        "date_range": [
            dxy_aligned.index[0].strftime('%Y-%m-%d'),
            dxy_aligned.index[-1].strftime('%Y-%m-%d')
        ],
        "rows": len(dxy_aligned),
        "columns": list(dxy_aligned.columns),
        "missing_values": {col: int(dxy_aligned[col].isna().sum()) for col in dxy_aligned.columns},
        "fetch_script": "src/fetch_dxy.py",
        "notes": [
            "Raw Close prices for DXY and 6 constituent currencies",
            "EURUSD and GBPUSD are inverse to USD strength (higher = weaker USD)",
            "JPY, CAD, SEK, CHF are same direction as USD strength (higher = stronger USD)",
            "Return negation will be applied during feature engineering in train.py",
            "Forward-filled missing values up to 3 days",
            "Aligned to base_features date range"
        ],
        "statistics": {
            col: {
                "mean": float(dxy_aligned[col].mean()),
                "std": float(dxy_aligned[col].std()),
                "min": float(dxy_aligned[col].min()),
                "max": float(dxy_aligned[col].max())
            } for col in dxy_aligned.columns
        }
    }

    metadata_path = 'data/multi_country/dxy_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to: {metadata_path}")

    # === Generate summary report ===
    print("\n" + "="*60)
    print("DXY DATA SUMMARY")
    print("="*60)
    print(f"\nDate Range: {metadata['date_range'][0]} to {metadata['date_range'][1]}")
    print(f"Total Rows: {metadata['rows']}")
    print(f"Columns: {len(metadata['columns'])}")
    print("\nColumn Details:")
    for col in metadata['columns']:
        stats = metadata['statistics'][col]
        print(f"  {col}:")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # Check alignment with base_features
    print("\n" + "="*60)
    print("ALIGNMENT CHECK")
    print("="*60)

    # Merge on dates to check alignment
    merged = base_features.merge(dxy_aligned, left_index=True, right_index=True, how='inner')
    print(f"Base features rows: {len(base_features)}")
    print(f"DXY data rows: {len(dxy_aligned)}")
    print(f"Matched rows: {len(merged)}")
    print(f"Match rate: {len(merged) / len(base_features) * 100:.2f}%")

    if len(merged) < len(base_features):
        print("\nWARNING: Some base_features dates are missing in DXY data")
        missing_dates = base_features.index.difference(dxy_aligned.index)
        print(f"Missing dates: {len(missing_dates)}")
        print(f"First 5 missing: {missing_dates[:5].tolist()}")

    print("\n" + "="*60)
    print("DATA FETCHING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
