"""
VIX Data Processing Script
Fetches VIX data and saves to processed directory for datachecker validation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fetch_vix import fetch_and_preprocess
import pandas as pd
import json
from datetime import datetime

def main():
    print("Fetching and preprocessing VIX data...")

    # Fetch data using the reusable function
    train_df, val_df, test_df, full_df = fetch_and_preprocess()

    # Align with base_features date range (2015-01-30 to 2025-02-12)
    schema_freeze_path = "C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents/shared/schema_freeze.json"
    with open(schema_freeze_path, 'r') as f:
        schema = json.load(f)

    target_start = pd.to_datetime(schema['date_range']['start'])
    target_end = pd.to_datetime(schema['date_range']['end'])

    print(f"\nAligning to base_features date range: {target_start} to {target_end}")

    # Filter to target date range
    full_df_aligned = full_df[
        (full_df['date'] >= target_start) &
        (full_df['date'] <= target_end)
    ].copy()

    print(f"Aligned rows: {len(full_df_aligned)}")
    print(f"Aligned date range: {full_df_aligned['date'].min()} to {full_df_aligned['date'].max()}")

    # Save raw data (already exists, but update if needed)
    raw_output_path = "C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents/data/raw/vix.csv"
    full_df[['date', 'vix']].to_csv(raw_output_path, index=False)
    print(f"\nSaved raw data: {raw_output_path}")

    # Save processed data for datachecker
    processed_dir = "C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents/data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    processed_output_path = os.path.join(processed_dir, "vix_processed.csv")
    full_df_aligned.to_csv(processed_output_path, index=False)
    print(f"Saved processed data: {processed_output_path}")

    # Create metadata
    metadata = {
        "feature": "vix",
        "created_at": datetime.now().isoformat(),
        "sources": ["FRED:VIXCLS"],
        "date_range": [
            full_df_aligned['date'].min().strftime('%Y-%m-%d'),
            full_df_aligned['date'].max().strftime('%Y-%m-%d')
        ],
        "rows": len(full_df_aligned),
        "columns": list(full_df_aligned.columns),
        "missing_values": {
            "vix": int(full_df_aligned['vix'].isna().sum()),
            "vix_log_change": int(full_df_aligned['vix_log_change'].isna().sum())
        },
        "fetch_script": "src/fetch_vix.py",
        "statistics": {
            "vix_mean": float(full_df_aligned['vix'].mean()),
            "vix_std": float(full_df_aligned['vix'].std()),
            "vix_min": float(full_df_aligned['vix'].min()),
            "vix_max": float(full_df_aligned['vix'].max()),
            "log_change_mean": float(full_df_aligned['vix_log_change'].mean()),
            "log_change_std": float(full_df_aligned['vix_log_change'].std())
        }
    }

    metadata_path = os.path.join(processed_dir, "vix_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    # Display summary
    print("\n" + "="*60)
    print("VIX DATA PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total rows: {len(full_df_aligned)}")
    print(f"Date range: {full_df_aligned['date'].min()} to {full_df_aligned['date'].max()}")
    print(f"Expected base_features rows: {schema['row_count']}")
    print(f"Match: {'YES' if len(full_df_aligned) == schema['row_count'] else 'NO'}")
    print(f"\nMissing values:")
    print(f"  vix: {metadata['missing_values']['vix']}")
    print(f"  vix_log_change: {metadata['missing_values']['vix_log_change']}")
    print(f"\nVIX Statistics:")
    print(f"  Mean: {metadata['statistics']['vix_mean']:.2f}")
    print(f"  Std:  {metadata['statistics']['vix_std']:.2f}")
    print(f"  Min:  {metadata['statistics']['vix_min']:.2f}")
    print(f"  Max:  {metadata['statistics']['vix_max']:.2f}")
    print(f"\nVIX Log-Change Statistics:")
    print(f"  Mean: {metadata['statistics']['log_change_mean']:.6f}")
    print(f"  Std:  {metadata['statistics']['log_change_std']:.6f}")
    print("="*60)

    return full_df_aligned

if __name__ == "__main__":
    df = main()
