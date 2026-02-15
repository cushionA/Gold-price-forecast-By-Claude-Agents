"""
Local script to merge base features and submodel outputs for meta-model.
Creates data/processed/meta_model_input.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


def main():
    project_root = Path(__file__).parent.parent
    print("=== Meta-Model Data Preparation ===\n")

    # === 1. Load Base Features ===
    base_path = project_root / "data" / "processed" / "base_features.csv"
    print(f"Loading base features from: {base_path}")
    df_base = pd.read_csv(base_path, index_col=0, parse_dates=True)

    # Make index timezone-naive if needed
    if hasattr(df_base.index, 'tz') and df_base.index.tz is not None:
        df_base.index = df_base.index.tz_localize(None)

    # Separate target
    target = df_base['gold_return_next'].copy()
    df_base = df_base.drop(columns=['gold_return_next'])

    print(f"  Base features: {df_base.shape[0]} rows, {df_base.shape[1]} columns")
    print(f"  Date range: {df_base.index.min()} to {df_base.index.max()}")

    # === 2. Load Submodel Outputs ===
    print(f"\nLoading submodel outputs:")
    submodel_dir = project_root / "data" / "submodel_outputs"

    submodel_files = {
        'vix': 'vix.csv',
        'technical': 'technical.csv',
        'cross_asset': 'cross_asset.csv',
        'yield_curve': 'yield_curve.csv',
        'etf_flow': 'etf_flow.csv',
        'inflation_expectation': 'inflation_expectation.csv',
        'cny_demand': 'cny_demand.csv'
    }

    submodel_dfs = {}
    submodel_info = {}

    for name, filename in submodel_files.items():
        path = submodel_dir / filename
        df = pd.read_csv(path, index_col=0)

        # Convert index to datetime, handling timezone-aware dates
        # Extract date only (not datetime with time) to match base_features
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True).date)

        submodel_dfs[name] = df
        submodel_info[name] = {
            'rows': df.shape[0],
            'columns': list(df.columns),
            'date_range': [str(df.index.min()), str(df.index.max())]
        }

        print(f"  {name:25s}: {df.shape[0]:4d} rows, {df.shape[1]} columns - {list(df.columns)}")

    # === 3. Exclude yc_regime_prob ===
    print(f"\nExcluding constant column:")
    if 'yc_regime_prob' in submodel_dfs['yield_curve'].columns:
        submodel_dfs['yield_curve'] = submodel_dfs['yield_curve'].drop(columns=['yc_regime_prob'])
        print(f"  Excluded: yc_regime_prob from yield_curve")
        print(f"  Remaining yield_curve columns: {list(submodel_dfs['yield_curve'].columns)}")

    # === 4. Merge All Data ===
    print(f"\nMerging all data:")
    df = df_base.copy()
    print(f"  Starting with base features: {df.shape}")

    for name, submodel_df in submodel_dfs.items():
        before_rows = df.shape[0]
        df = df.join(submodel_df, how='inner')
        after_rows = df.shape[0]
        print(f"  + {name:25s}: {df.shape} (lost {before_rows - after_rows} rows)")

    # === 5. Data Quality Checks ===
    print(f"\nData quality checks:")
    print(f"  Total features: {df.shape[1]}")
    print(f"  Expected: 39 (19 base + 20 submodel)")

    if df.shape[1] != 39:
        print(f"  WARNING: Expected 39 columns, got {df.shape[1]}")
        print(f"  Columns: {list(df.columns)}")

    # Check for NaN values before alignment
    nan_before = df.isna().sum().sum()
    print(f"  NaN values in features (before alignment): {nan_before}")

    # Align target with merged features
    target = target.reindex(df.index)
    target_nan = target.isna().sum()
    print(f"  NaN values in target: {target_nan}")

    # Check for inf values
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if inf_counts.sum() > 0:
        print(f"  WARNING: Inf values found in {(inf_counts > 0).sum()} columns:")
        print(inf_counts[inf_counts > 0])

    # Drop rows with NaN or inf in features or target
    valid_mask = ~(df.isna().any(axis=1) | target.isna() | np.isinf(df.select_dtypes(include=[np.number])).any(axis=1))
    rows_before = len(df)
    df = df[valid_mask]
    target = target[valid_mask]
    rows_dropped = rows_before - len(df)

    print(f"  Rows dropped (NaN/inf): {rows_dropped}")
    print(f"  Final row count: {df.shape[0]}")
    print(f"  Final date range: {df.index.min()} to {df.index.max()}")

    # === 6. Verify Column Count ===
    print(f"\nColumn verification:")
    print(f"  Total columns: {df.shape[1]}")

    # Count base vs submodel columns
    base_cols = [col for col in df.columns if not any(
        col.startswith(prefix) for prefix in ['vix_', 'tech_', 'xasset_', 'yc_', 'etf_', 'ie_', 'cny_']
    )]
    submodel_cols = [col for col in df.columns if col not in base_cols]

    print(f"  Base features: {len(base_cols)}")
    print(f"  Submodel features: {len(submodel_cols)}")
    print(f"  Submodel breakdown:")

    for name in submodel_files.keys():
        prefix_map = {
            'vix': 'vix_',
            'technical': 'tech_',
            'cross_asset': 'xasset_',
            'yield_curve': 'yc_',
            'etf_flow': 'etf_',
            'inflation_expectation': 'ie_',
            'cny_demand': 'cny_'
        }
        prefix = prefix_map[name]
        cols = [col for col in df.columns if col.startswith(prefix)]
        print(f"    {name:25s}: {len(cols)} columns - {cols}")

    # === 7. Save Merged Dataset ===
    output_path = project_root / "data" / "processed" / "meta_model_input.csv"
    print(f"\nSaving merged dataset to: {output_path}")

    # Save without target column (target stays in target.csv)
    df.to_csv(output_path)
    print(f"  Saved: {df.shape[0]} rows, {df.shape[1]} columns")

    # === 8. Generate Summary Statistics ===
    summary_path = project_root / "logs" / "datacheck" / "meta_model_data_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        f.write("=== Meta-Model Input Data Summary ===\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("1. DATASET OVERVIEW\n")
        f.write(f"   Total rows: {df.shape[0]}\n")
        f.write(f"   Total columns: {df.shape[1]}\n")
        f.write(f"   Date range: {df.index.min()} to {df.index.max()}\n")
        f.write(f"   Expected rows: 2523 (base features)\n")
        if rows_before > 0:
            f.write(f"   Rows dropped: {rows_dropped} ({rows_dropped/rows_before*100:.2f}%)\n\n")
        else:
            f.write(f"   Rows dropped: {rows_dropped} (N/A - no rows before)\n\n")

        f.write("2. FEATURE BREAKDOWN\n")
        f.write(f"   Base features: {len(base_cols)}\n")
        f.write(f"   Submodel features: {len(submodel_cols)}\n")
        f.write(f"   Total: {len(base_cols) + len(submodel_cols)}\n\n")

        f.write("3. SUBMODEL CONTRIBUTIONS\n")
        for name in submodel_files.keys():
            prefix_map = {
                'vix': 'vix_',
                'technical': 'tech_',
                'cross_asset': 'xasset_',
                'yield_curve': 'yc_',
                'etf_flow': 'etf_',
                'inflation_expectation': 'ie_',
                'cny_demand': 'cny_'
            }
            prefix = prefix_map[name]
            cols = [col for col in df.columns if col.startswith(prefix)]
            f.write(f"   {name:25s}: {len(cols)} features - {cols}\n")

        f.write("\n4. MISSING VALUE SUMMARY\n")
        nan_summary = df.isna().sum()
        if nan_summary.sum() == 0:
            f.write("   No missing values\n")
        else:
            f.write("   Missing values by column:\n")
            for col, count in nan_summary[nan_summary > 0].items():
                f.write(f"     {col}: {count} ({count/len(df)*100:.2f}%)\n")

        f.write("\n5. BASIC STATISTICS (selected features)\n")
        selected_cols = [
            'real_rate_real_rate', 'dxy_dxy', 'vix_vix',
            'vix_regime_probability', 'tech_trend_regime_prob',
            'etf_regime_prob', 'ie_regime_prob', 'cny_regime_prob'
        ]
        selected_cols = [col for col in selected_cols if col in df.columns]
        stats = df[selected_cols].describe()
        f.write(stats.to_string())
        f.write("\n\n")

        f.write("6. DATA SPLITS (70/15/15)\n")
        n = len(df)
        if n > 0:
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            f.write(f"   Train: {train_end} rows ({train_end/n*100:.1f}%)\n")
            f.write(f"   Val:   {val_end - train_end} rows ({(val_end-train_end)/n*100:.1f}%)\n")
            f.write(f"   Test:  {n - val_end} rows ({(n-val_end)/n*100:.1f}%)\n\n")
        else:
            f.write(f"   ERROR: No data available for splits\n\n")

        f.write("7. TARGET ALIGNMENT\n")
        f.write(f"   Target variable: gold_return_next\n")
        f.write(f"   Target rows: {len(target)}\n")
        f.write(f"   Target NaN: {target.isna().sum()}\n")
        f.write(f"   Target mean: {target.mean():.6f}\n")
        f.write(f"   Target std:  {target.std():.6f}\n")
        f.write(f"   Target range: [{target.min():.6f}, {target.max():.6f}]\n\n")

        f.write("8. COLUMN NAMES\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"   {i:2d}. {col}\n")

    print(f"\nSummary saved to: {summary_path}")

    print(f"\n=== Data preparation complete ===")
    print(f"Next step: Run datachecker to validate merged dataset")


if __name__ == "__main__":
    main()
