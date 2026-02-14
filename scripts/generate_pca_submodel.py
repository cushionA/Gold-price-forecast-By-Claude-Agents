"""
Generate PCA-based submodel for real_rate Attempt 4
Deterministic approach: No training, no overfitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
import json
from datetime import datetime

def generate_pca_submodel():
    print("=" * 60)
    print("real_rate Attempt 4: PCA + Cubic Spline")
    print("=" * 60)

    # 1. Load multi-country data
    print("\n1. Loading multi-country data...")
    df = pd.read_csv('data/processed/real_rate_multi_country_features.csv',
                     index_col=0, parse_dates=True)
    print(f"Loaded: {df.shape}")

    # 2. Select features (7 rate changes)
    feature_cols = [
        'germany_nominal_change', 'uk_nominal_change', 'canada_nominal_change',
        'switzerland_nominal_change', 'norway_nominal_change', 'sweden_nominal_change',
        'us_tips_change'
    ]
    X = df[feature_cols].dropna()
    print(f"Features: {len(feature_cols)} rate changes")
    print(f"Samples: {len(X)} months")

    # 3. Train split (70% for fitting StandardScaler + PCA)
    n_train = int(0.7 * len(X))
    X_train = X.iloc[:n_train]
    print(f"\n2. Fitting on train period: {n_train} months")

    # 4. Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X)

    # 5. PCA (fit on train, transform full)
    pca = PCA(n_components=2)
    pca.fit(X_train_scaled)
    pc_scores_monthly = pca.transform(X_full_scaled)

    print(f"\n3. PCA Results:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative variance: {pca.explained_variance_ratio_.sum():.3f}")

    # 6. Cubic spline interpolation to daily
    print(f"\n4. Interpolating monthly→daily with cubic spline...")
    monthly_dates = X.index

    # Create daily date range
    gold_dates = pd.date_range(monthly_dates[0], monthly_dates[-1], freq='D')

    # Convert dates to numeric (seconds since epoch) for spline
    monthly_numeric = (monthly_dates - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    daily_numeric = (gold_dates - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

    pc_daily = []
    for i in range(2):
        cs = CubicSpline(monthly_numeric, pc_scores_monthly[:, i])
        pc_daily_i = cs(daily_numeric)
        pc_daily.append(pc_daily_i)

    pc_daily = np.column_stack(pc_daily)
    print(f"Interpolated to {len(gold_dates)} daily values")

    # 7. Create output DataFrame
    output_df = pd.DataFrame(
        pc_daily,
        index=gold_dates,
        columns=['real_rate_pc_0', 'real_rate_pc_1']
    )

    # 8. Align to schema range (2015-01-30 to 2025-02-12)
    schema_start = '2015-01-30'
    schema_end = '2025-02-12'
    output_df = output_df.loc[schema_start:schema_end]

    print(f"\n5. Schema alignment:")
    print(f"Date range: {output_df.index[0]} to {output_df.index[-1]}")
    print(f"Rows: {len(output_df)}")
    print(f"Columns: {list(output_df.columns)}")

    # 9. Quality checks
    print(f"\n6. Quality checks:")
    print(f"NaN count: {output_df.isna().sum().sum()}")
    print(f"PC0 range: [{output_df['real_rate_pc_0'].min():.3f}, {output_df['real_rate_pc_0'].max():.3f}]")
    print(f"PC1 range: [{output_df['real_rate_pc_1'].min():.3f}, {output_df['real_rate_pc_1'].max():.3f}]")

    # 10. Save output
    output_path = 'data/submodel_outputs/real_rate/real_rate.csv'
    output_df.to_csv(output_path)
    print(f"\n7. Saved: {output_path}")

    # 11. Save metadata
    result = {
        'feature': 'real_rate',
        'attempt': 4,
        'method': 'PCA',
        'timestamp': datetime.now().isoformat(),
        'n_components': 2,
        'features_used': feature_cols,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': float(pca.explained_variance_ratio_.sum()),
        'interpolation': 'cubic_spline',
        'output_shape': list(output_df.shape),
        'output_columns': list(output_df.columns),
        'date_range': {
            'start': str(output_df.index[0]),
            'end': str(output_df.index[-1])
        },
        'execution_time_seconds': '<10'
    }

    result_path = 'logs/training/real_rate_4.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved metadata: {result_path}")

    print("\n" + "=" * 60)
    print("PCA submodel generation complete!")
    print("=" * 60)

    return output_df, result

if __name__ == "__main__":
    output_df, result = generate_pca_submodel()
