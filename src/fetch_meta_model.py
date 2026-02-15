"""
Data fetching and preprocessing: meta_model
builder_model will embed this code in train.ipynb for Kaggle execution
"""
import pandas as pd
import numpy as np
from pathlib import Path


def fetch_and_preprocess():
    """
    Self-contained meta-model data preparation.
    Merges base features + submodel outputs.

    Returns: (train_df, val_df, test_df, full_df)
    """

    # === 1. Load Base Features ===
    # This function assumes base_features and submodel outputs are already available
    # In the actual Kaggle notebook, these will be generated from raw data

    # For local execution
    try:
        base_path = Path(__file__).parent.parent
    except NameError:
        # When running in Kaggle/Jupyter
        base_path = Path(".")

    # Load base features
    base_features_path = base_path / "data" / "processed" / "base_features.csv"
    df_base = pd.read_csv(base_features_path, index_col=0, parse_dates=True)

    # Separate target from base features
    target = df_base['gold_return_next'].copy()
    df_base = df_base.drop(columns=['gold_return_next'])

    print(f"Base features loaded: {df_base.shape[0]} rows, {df_base.shape[1]} columns")

    # === 2. Load Submodel Outputs ===
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
    for name, filename in submodel_files.items():
        path = base_path / "data" / "submodel_outputs" / filename
        df = pd.read_csv(path, index_col=0)

        # Convert index to datetime, handling timezone-aware dates
        # Extract date only (not datetime with time) to match base_features
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True).date)

        submodel_dfs[name] = df
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")

    # === 3. Exclude yc_regime_prob (constant column) ===
    if 'yc_regime_prob' in submodel_dfs['yield_curve'].columns:
        submodel_dfs['yield_curve'] = submodel_dfs['yield_curve'].drop(columns=['yc_regime_prob'])
        print(f"  Excluded yc_regime_prob from yield_curve")

    # === 4. Merge All Data ===
    # Start with base features
    df = df_base.copy()

    # Merge each submodel output (inner join to ensure alignment)
    for name, submodel_df in submodel_dfs.items():
        df = df.join(submodel_df, how='inner')
        print(f"  After merging {name}: {df.shape[0]} rows, {df.shape[1]} columns")

    # Align target with merged features
    target = target.reindex(df.index)

    # === 5. Data Quality Checks ===
    print(f"\nData quality checks:")
    print(f"  Total features: {df.shape[1]}")
    print(f"  Expected: 39 (19 base + 20 submodel)")

    # Check for NaN values
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"  NaN values found in {(nan_counts > 0).sum()} columns:")
        print(nan_counts[nan_counts > 0])
    else:
        print(f"  No NaN values in feature columns")

    target_nan = target.isna().sum()
    if target_nan > 0:
        print(f"  WARNING: {target_nan} NaN values in target")

    # Drop rows with NaN in either features or target
    valid_mask = ~(df.isna().any(axis=1) | target.isna())
    df = df[valid_mask]
    target = target[valid_mask]

    print(f"  After dropping NaN rows: {df.shape[0]} rows")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # === 6. Split Data (70/15/15, time-series order) ===
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_features = df.iloc[:train_end]
    val_features = df.iloc[train_end:val_end]
    test_features = df.iloc[val_end:]

    train_target = target.iloc[:train_end]
    val_target = target.iloc[train_end:val_end]
    test_target = target.iloc[val_end:]

    # Combine features and target for each split
    train_df = train_features.copy()
    train_df['gold_return_next'] = train_target

    val_df = val_features.copy()
    val_df['gold_return_next'] = val_target

    test_df = test_features.copy()
    test_df['gold_return_next'] = test_target

    full_df = df.copy()
    full_df['gold_return_next'] = target

    print(f"\nData splits:")
    print(f"  Train: {len(train_df)} rows ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} rows ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} rows ({len(test_df)/n*100:.1f}%)")

    # === 7. Basic Statistics ===
    print(f"\nBasic statistics (first 5 features):")
    print(df.iloc[:, :5].describe())

    return train_df, val_df, test_df, full_df


if __name__ == "__main__":
    # Local testing
    train_df, val_df, test_df, full_df = fetch_and_preprocess()

    print(f"\n=== Data preparation complete ===")
    print(f"Feature columns: {list(full_df.columns)}")
