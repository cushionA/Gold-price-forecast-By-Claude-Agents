"""
Prepare baseline features for Phase 1
Integrates 9 key features into a single dataset for XGBoost baseline
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Paths
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
SHARED = Path("shared")

def load_and_process_feature(feature_name):
    """Load raw feature data and extract relevant columns"""
    df = pd.read_csv(DATA_RAW / f"{feature_name}.csv")

    # Handle both 'Date' and 'date' column names
    date_col = 'Date' if 'Date' in df.columns else 'date'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Rename columns with feature prefix to avoid conflicts
    df.columns = [f"{feature_name}_{col}" for col in df.columns]

    return df

def main():
    print("Loading target variable...")
    target = pd.read_csv(DATA_PROCESSED / "target.csv")
    date_col = 'Date' if 'Date' in target.columns else 'date'
    target[date_col] = pd.to_datetime(target[date_col])
    target = target.set_index(date_col).sort_index()

    print("Loading 9 key features...")
    features = [
        'real_rate',        # 1. Real Interest Rate
        'dxy',              # 2. Dollar Index
        'vix',              # 3. VIX
        'technical',        # 4. Gold Technicals
        'cross_asset',      # 5. Cross-Asset
        'yield_curve',      # 6. Yield Curve
        'etf_flow',         # 7. ETF Flows
        'inflation_expectation',  # 8. Inflation Expectation
        'cny_demand'        # 9. CNY Demand Proxy
    ]

    feature_dfs = []
    for feat in features:
        print(f"  Loading {feat}...")
        df = load_and_process_feature(feat)
        feature_dfs.append(df)

    print("\nMerging all features...")
    # Start with target
    combined = target.copy()

    # Merge each feature
    for feat_df in feature_dfs:
        combined = combined.join(feat_df, how='left')

    # Forward fill for missing values (max 5 days as per CLAUDE.md)
    combined = combined.ffill(limit=5)

    # Drop rows with any remaining NaN
    rows_before = len(combined)
    combined = combined.dropna()
    rows_after = len(combined)
    print(f"\nDropped {rows_before - rows_after} rows with NaN values")
    print(f"Final dataset shape: {combined.shape}")

    # Save base features
    output_path = DATA_PROCESSED / "base_features.csv"
    combined.to_csv(output_path)
    print(f"\nSaved base features to {output_path}")

    # Create schema freeze for Gate 2/3 validation
    schema = {
        "columns": list(combined.columns),
        "dtypes": {col: str(dtype) for col, dtype in combined.dtypes.items()},
        "date_range": {
            "start": combined.index.min().isoformat(),
            "end": combined.index.max().isoformat()
        },
        "row_count": len(combined),
        "feature_count": len(combined.columns) - 1,  # Exclude target
        "created_at": datetime.now().isoformat()
    }

    schema_path = SHARED / "schema_freeze.json"
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"Saved schema freeze to {schema_path}")

    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Date range: {schema['date_range']['start']} to {schema['date_range']['end']}")
    print(f"Total rows: {schema['row_count']}")
    print(f"Total features: {schema['feature_count']}")
    print(f"Target variable: {combined.columns[0]}")
    print("\nFeature groups:")
    for feat in features:
        feat_cols = [col for col in combined.columns if col.startswith(feat)]
        print(f"  {feat}: {len(feat_cols)} columns")

    print("\nBase features preparation complete!")

if __name__ == "__main__":
    main()
