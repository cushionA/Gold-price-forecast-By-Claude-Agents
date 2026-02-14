"""
Debug: Why does statsmodels VIF give 37.36 when R^2 is only 0.12?
"""
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

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

# The statsmodels VIF adds a constant internally.
# Let's replicate exactly what it does.
test_df = pd.concat([base, sub[['real_rate_regime_persistence']]], axis=1)
print(f"test_df shape: {test_df.shape}")
print(f"test_df columns: {list(test_df.columns)}")

# VIF from statsmodels
idx = test_df.shape[1] - 1  # last column
vif_sm = variance_inflation_factor(test_df.values, idx)
print(f"\nstatsmodels VIF for regime_persistence: {vif_sm:.4f}")

# Manual VIF computation
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

y_col = test_df.values[:, idx]
x_cols = np.delete(test_df.values, idx, axis=1)

# statsmodels adds constant by default in VIF
x_with_const = add_constant(x_cols)
model = OLS(y_col, x_with_const).fit()
r2_manual = model.rsquared
vif_manual = 1.0 / (1.0 - r2_manual)
print(f"Manual OLS R^2: {r2_manual:.6f}")
print(f"Manual VIF (1/(1-R^2)): {vif_manual:.4f}")

# Check: are there inf/nan columns causing numerical issues?
print(f"\nBase feature scales:")
print(base.describe().loc[['mean', 'std', 'min', 'max']].to_string())

# Check if base features have high multicollinearity themselves
print(f"\nBase-only VIF (first 5 columns):")
base_values = base.values
for i in range(min(5, base.shape[1])):
    try:
        v = variance_inflation_factor(base_values, i)
        print(f"  {base.columns[i]}: {v:.4f}")
    except:
        print(f"  {base.columns[i]}: ERROR")

# The issue might be that the BASE FEATURES themselves have very high VIF,
# and adding any column to an already multicollinear set inflates VIF
print(f"\nAll base VIFs:")
for i in range(base.shape[1]):
    try:
        v = variance_inflation_factor(base_values, i)
        print(f"  {base.columns[i]}: {v:.2f}")
    except Exception as e:
        print(f"  {base.columns[i]}: ERROR ({e})")
