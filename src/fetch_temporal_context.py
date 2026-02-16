"""
Data preparation: temporal_context_transformer
Combines base features and submodel outputs for Temporal Context Transformer
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def fetch_and_preprocess():
    """
    Self-contained data preparation.
    Returns: (train_df, val_df, test_df, full_df, scaler)
    """

    print("=" * 80)
    print("Temporal Context Transformer - Data Preparation")
    print("=" * 80)

    # ===== 1. Load base features =====
    print("\n[1/6] Loading base features...")
    base_path = "C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents/data/processed/base_features.csv"
    base_df = pd.read_csv(base_path)
    base_df['Date'] = pd.to_datetime(base_df['Date'])
    base_df = base_df.set_index('Date').sort_index()

    print(f"   Base features loaded: {base_df.shape[0]} rows, {base_df.shape[1]} columns")
    print(f"   Date range: {base_df.index.min()} to {base_df.index.max()}")

    # Extract and transform 5 base features
    # According to design doc Section 2, we need:
    # 1. real_rate_change (from real_rate_real_rate)
    # 2. dxy_change (from dxy_dxy)
    # 3. vix (from vix_vix - no change, already stationary)
    # 4. yield_spread_change (from yield_curve_yield_spread)
    # 5. inflation_exp_change (from inflation_expectation_inflation_expectation)

    base_features = pd.DataFrame(index=base_df.index)
    base_features['real_rate_change'] = base_df['real_rate_real_rate'].diff()
    base_features['dxy_change'] = base_df['dxy_dxy'].diff()
    base_features['vix'] = base_df['vix_vix']  # No transformation
    base_features['yield_spread_change'] = base_df['yield_curve_yield_spread'].diff()
    base_features['inflation_exp_change'] = base_df['inflation_expectation_inflation_expectation'].diff()

    print(f"   Base features extracted: {list(base_features.columns)}")

    # ===== 2. Load submodel outputs =====
    print("\n[2/6] Loading submodel outputs...")
    submodel_path = "C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents/data/submodel_outputs/"

    # VIX submodel (2 columns)
    vix_df = pd.read_csv(submodel_path + "vix.csv")
    vix_df['date'] = pd.to_datetime(vix_df['date'])
    vix_df = vix_df.set_index('date').sort_index()
    vix_df.index.name = 'Date'  # Normalize index name
    vix_features = vix_df[['vix_regime_probability', 'vix_mean_reversion_z']].copy()
    print(f"   VIX: {vix_features.shape[0]} rows, columns: {list(vix_features.columns)}")

    # Technical submodel (3 columns)
    tech_df = pd.read_csv(submodel_path + "technical.csv")
    # Handle timezone-aware dates - extract just the date part as string, then parse
    tech_df['date'] = tech_df['date'].str[:10]  # Extract YYYY-MM-DD
    tech_df['date'] = pd.to_datetime(tech_df['date'])
    tech_df = tech_df.set_index('date').sort_index()
    tech_df.index.name = 'Date'  # Normalize index name
    tech_features = tech_df[['tech_trend_regime_prob', 'tech_mean_reversion_z', 'tech_volatility_regime']].copy()
    print(f"   Technical: {tech_features.shape[0]} rows, columns: {list(tech_features.columns)}")

    # Cross-asset submodel (2 columns: regime_prob and divergence)
    xasset_df = pd.read_csv(submodel_path + "cross_asset.csv")
    xasset_df['Date'] = pd.to_datetime(xasset_df['Date'])
    xasset_df = xasset_df.set_index('Date').sort_index()
    xasset_features = xasset_df[['xasset_regime_prob', 'xasset_divergence']].copy()
    print(f"   Cross-asset: {xasset_features.shape[0]} rows, columns: {list(xasset_features.columns)}")

    # ETF flow submodel (1 column: regime_prob only)
    etf_df = pd.read_csv(submodel_path + "etf_flow.csv")
    etf_df['Date'] = pd.to_datetime(etf_df['Date'])
    etf_df = etf_df.set_index('Date').sort_index()
    etf_features = etf_df[['etf_regime_prob']].copy()
    print(f"   ETF flow: {etf_features.shape[0]} rows, columns: {list(etf_features.columns)}")

    # Options market submodel (1 column)
    options_df = pd.read_csv(submodel_path + "options_market.csv")
    # Handle timezone-aware dates - extract just the date part as string, then parse
    options_df['Date'] = options_df['Date'].str[:10]  # Extract YYYY-MM-DD
    options_df['Date'] = pd.to_datetime(options_df['Date'])
    options_df = options_df.set_index('Date').sort_index()
    options_features = options_df[['options_risk_regime_prob']].copy()
    print(f"   Options: {options_features.shape[0]} rows, columns: {list(options_features.columns)}")

    # ===== 3. Merge all data =====
    print("\n[3/6] Merging all features...")

    # Start with base features
    merged_df = base_features.copy()
    print(f"   Starting with base: {merged_df.shape[0]} rows, index name: {merged_df.index.name}")

    # Join submodel outputs (inner join to ensure alignment)
    merged_df = merged_df.join(vix_features, how='inner')
    print(f"   After VIX join: {merged_df.shape[0]} rows")

    merged_df = merged_df.join(tech_features, how='inner')
    print(f"   After Technical join: {merged_df.shape[0]} rows")

    merged_df = merged_df.join(xasset_features, how='inner')
    print(f"   After Cross-asset join: {merged_df.shape[0]} rows")

    merged_df = merged_df.join(etf_features, how='inner')
    print(f"   After ETF join: {merged_df.shape[0]} rows")

    merged_df = merged_df.join(options_features, how='inner')
    print(f"   After Options join: {merged_df.shape[0]} rows")

    print(f"   Final: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    print(f"   Date range: {merged_df.index.min()} to {merged_df.index.max()}")

    # Verify we have exactly 14 columns
    expected_cols = [
        'real_rate_change', 'dxy_change', 'vix', 'yield_spread_change', 'inflation_exp_change',
        'vix_regime_probability', 'vix_mean_reversion_z',
        'tech_trend_regime_prob', 'tech_mean_reversion_z', 'tech_volatility_regime',
        'xasset_regime_prob', 'xasset_divergence',
        'etf_regime_prob',
        'options_risk_regime_prob'
    ]

    assert merged_df.shape[1] == 14, f"Expected 14 columns, got {merged_df.shape[1]}"
    assert all(col in merged_df.columns for col in expected_cols), "Missing expected columns"

    print(f"   [OK] All 14 expected columns present")

    # ===== 4. Handle missing values =====
    print("\n[4/6] Handling missing values...")

    # Check initial missing values
    missing_before = merged_df.isnull().sum()
    print(f"   Missing values before processing:")
    for col in missing_before[missing_before > 0].index:
        print(f"      {col}: {missing_before[col]}")

    # Forward fill (max 5 days)
    merged_df = merged_df.fillna(method='ffill', limit=5)

    # Backward fill for remaining
    merged_df = merged_df.fillna(method='bfill')

    # Drop any remaining rows with NaN
    rows_before = len(merged_df)
    merged_df = merged_df.dropna()
    rows_after = len(merged_df)

    if rows_before > rows_after:
        print(f"   Dropped {rows_before - rows_after} rows with remaining NaN")

    # Check for inf values
    inf_mask = np.isinf(merged_df.values).any(axis=1)
    if inf_mask.any():
        print(f"   Dropping {inf_mask.sum()} rows with infinite values")
        merged_df = merged_df[~inf_mask]

    print(f"   Final dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    print(f"   [OK] No missing values: {merged_df.isnull().sum().sum() == 0}")

    # ===== 5. Time-series split (70/15/15) =====
    print("\n[5/6] Splitting data (70/15/15)...")

    n = len(merged_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = merged_df.iloc[:train_end].copy()
    val_df = merged_df.iloc[train_end:val_end].copy()
    test_df = merged_df.iloc[val_end:].copy()

    print(f"   Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"   Val:   {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    print(f"   Test:  {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")

    # ===== 6. Standardization =====
    print("\n[6/6] Standardizing features...")

    # Fit scaler on train only
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    full_scaled = scaler.transform(merged_df)

    # Convert back to DataFrame
    train_df_scaled = pd.DataFrame(train_scaled, index=train_df.index, columns=train_df.columns)
    val_df_scaled = pd.DataFrame(val_scaled, index=val_df.index, columns=val_df.columns)
    test_df_scaled = pd.DataFrame(test_scaled, index=test_df.index, columns=test_df.columns)
    full_df_scaled = pd.DataFrame(full_scaled, index=merged_df.index, columns=merged_df.columns)

    print(f"   [OK] Features standardized using train set statistics")
    print(f"   Train mean range: [{train_scaled.mean(axis=0).min():.4f}, {train_scaled.mean(axis=0).max():.4f}]")
    print(f"   Train std range: [{train_scaled.std(axis=0).min():.4f}, {train_scaled.std(axis=0).max():.4f}]")

    # ===== Summary =====
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(merged_df)}")
    print(f"Train samples: {len(train_df_scaled)} ({len(train_df_scaled)/len(merged_df)*100:.1f}%)")
    print(f"Val samples: {len(val_df_scaled)} ({len(val_df_scaled)/len(merged_df)*100:.1f}%)")
    print(f"Test samples: {len(test_df_scaled)} ({len(test_df_scaled)/len(merged_df)*100:.1f}%)")
    print(f"Features: {merged_df.shape[1]}")
    print(f"Date range: {merged_df.index.min().strftime('%Y-%m-%d')} to {merged_df.index.max().strftime('%Y-%m-%d')}")
    print("=" * 80)

    return train_df_scaled, val_df_scaled, test_df_scaled, full_df_scaled, scaler


if __name__ == "__main__":
    # Run data preparation
    train_df, val_df, test_df, full_df, scaler = fetch_and_preprocess()

    # Save processed data
    import os
    output_dir = "C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents/data/processed/"
    os.makedirs(output_dir, exist_ok=True)

    # Save as CSV (with date index)
    full_df.reset_index().to_csv(output_dir + "temporal_context_raw.csv", index=False)
    train_df.reset_index().to_csv(output_dir + "temporal_context_train.csv", index=False)
    val_df.reset_index().to_csv(output_dir + "temporal_context_val.csv", index=False)
    test_df.reset_index().to_csv(output_dir + "temporal_context_test.csv", index=False)

    print(f"\n[OK] Saved to {output_dir}")
    print(f"  - temporal_context_raw.csv ({len(full_df)} rows)")
    print(f"  - temporal_context_train.csv ({len(train_df)} rows)")
    print(f"  - temporal_context_val.csv ({len(val_df)} rows)")
    print(f"  - temporal_context_test.csv ({len(test_df)} rows)")
