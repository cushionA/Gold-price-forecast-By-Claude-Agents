"""
Investigate VIF failure for real_rate attempt 5.
Question: Is high VIF caused by multicollinearity WITH base features,
or BETWEEN submodel columns themselves?
"""
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = r"C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents"
sub = pd.read_csv(f"{BASE_DIR}/data/submodel_outputs/real_rate.csv", index_col=0, parse_dates=True)
base = pd.read_csv(f"{BASE_DIR}/data/processed/base_features.csv", index_col=0, parse_dates=True)
if "gold_return_next" in base.columns:
    base = base.drop(columns=["gold_return_next"])

common_idx = base.index.intersection(sub.index)
base = base.loc[common_idx].dropna()
sub = sub.loc[common_idx]
common_idx = base.index.intersection(sub.dropna().index)
base = base.loc[common_idx]
sub = sub.loc[common_idx]

print("=== Investigation 1: Correlation between submodel columns ===")
corr = sub.corr()
print(corr.to_string())

print("\n=== Investigation 2: regime_persistence + transition_prob ===")
print(f"  Sum: {(sub['real_rate_regime_persistence'] + sub['real_rate_transition_prob']).describe()}")
# They sum to ~1.0, so they are perfectly linearly dependent

print("\n=== Investigation 3: VIF of each submodel column against BASE ONLY ===")
print("(Each column tested independently against base features)")
for col in sub.columns:
    test_df = pd.concat([base, sub[[col]]], axis=1)
    vif = variance_inflation_factor(test_df.values, test_df.shape[1] - 1)
    print(f"  {col}: VIF = {vif:.4f}")

print("\n=== Investigation 4: VIF of submodel columns AMONG THEMSELVES ===")
for i, col in enumerate(sub.columns):
    vif = variance_inflation_factor(sub.values, i)
    print(f"  {col}: VIF (among submodel cols) = {vif:.4f}")

print("\n=== Investigation 5: VIF with reduced submodel (drop redundant columns) ===")
# Drop transition_prob (complement of persistence) and regime_sync (if redundant)
reduced_cols = ['real_rate_regime_persistence', 'real_rate_trend_direction',
                'real_rate_trend_strength', 'real_rate_change_magnitude',
                'real_rate_days_since_change']
sub_reduced = sub[reduced_cols]
print(f"Reduced columns: {reduced_cols}")
for col in reduced_cols:
    test_df = pd.concat([base, sub_reduced[[col]]], axis=1)
    vif = variance_inflation_factor(test_df.values, test_df.shape[1] - 1)
    print(f"  {col}: VIF vs base = {vif:.4f}")

# Full reduced set VIF
ext_reduced = pd.concat([base, sub_reduced], axis=1)
print("\nFull reduced set VIF (all submodel cols together against base):")
for i in range(base.shape[1], ext_reduced.shape[1]):
    col = ext_reduced.columns[i]
    vif = variance_inflation_factor(ext_reduced.values, i)
    print(f"  {col}: VIF = {vif:.4f}")
