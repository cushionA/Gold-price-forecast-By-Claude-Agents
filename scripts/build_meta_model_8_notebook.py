"""
Build meta_model attempt 8 notebook from attempt 7.
This script applies all required modifications specified in the design document.
"""

import json
import copy

# Path to attempt 7 notebook
NB7_PATH = 'notebooks/meta_model_7/train.ipynb'
NB8_PATH = 'notebooks/meta_model_8/train.ipynb'

def load_notebook(path):
    """Load Jupyter notebook."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb, path):
    """Save Jupyter notebook."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def make_code_cell(source_lines):
    """Create a code cell."""
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines
    }

def make_markdown_cell(source_lines):
    """Create a markdown cell."""
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': source_lines
    }

# Load attempt 7
nb7 = load_notebook(NB7_PATH)
print(f"Loaded attempt 7 notebook with {len(nb7['cells'])} cells")

# Create attempt 8
nb8 = copy.deepcopy(nb7)
new_cells = []

# ==================================================================
# MODIFICATIONS PER DESIGN DOC SECTION 7.3
# ==================================================================

# 1. Update markdown header (Cell 0)
new_cells.append(make_markdown_cell([
    '# Gold Meta-Model Training - Attempt 8\n',
    '\n',
    '**Architecture:** GBDT Stacking (XGBoost + LightGBM + CatBoost) with Ridge meta-learner\n',
    '\n',
    '**Key Changes from Attempt 7:**\n',
    '1. **+6 regime-conditional features** (30 total features, was 24)\n',
    '2. **GBDT Stacking Ensemble** (XGBoost + LightGBM + CatBoost + Ridge)\n',
    '3. **Improved Optuna weights**: 35/35/10/20 (was 40/30/10/20)\n',
    '4. **Wider XGBoost max_depth**: [2, 5] (was [2, 4])\n',
    '5. **CPU mode**: enable_gpu=false\n',
    '\n',
    '**Inherited:** Bootstrap confidence, OLS scaling, Attempt 2 fallback\n',
    '\n',
    '**Design:** `docs/design/meta_model_attempt_8.md`'
]))

# 2. Imports - Add lightgbm, catboost, Ridge (Cell 1)
cell1 = copy.deepcopy(nb7['cells'][1])
imports = cell1['source']
# Insert after xgboost line
new_imports = []
for line in imports:
    new_imports.append(line)
    if 'import xgboost as xgb' in line:
        new_imports.extend([
            'import lightgbm as lgb\n',
            'import catboost as cb\n',
            'from sklearn.linear_model import Ridge\n'
        ])
    elif 'print(f"XGBoost version' in line:
        new_imports.extend([
            'print(f"LightGBM version: {lgb.__version__}")\n',
            'print(f"CatBoost version: {cb.__version__}")\n'
        ])

cell1['source'] = new_imports
new_cells.append(cell1)

# 3. Feature Definitions markdown (Cell 2)
new_cells.append(nb7['cells'][2])

# 4. Feature columns - UPDATE to 30 (Cell 3)
new_cells.append(make_code_cell([
    '# Base features (24)\n',
    'BASE_FEATURE_COLUMNS = [\n',
    '    # Base features (5)\n',
    "    'real_rate_change', 'dxy_change', 'vix', 'yield_spread_change', 'inflation_exp_change',\n",
    '    # VIX submodel (3)\n',
    "    'vix_regime_probability', 'vix_mean_reversion_z', 'vix_persistence',\n",
    '    # Technical submodel (3)\n',
    "    'tech_trend_regime_prob', 'tech_mean_reversion_z', 'tech_volatility_regime',\n",
    '    # Cross-asset submodel (3)\n',
    "    'xasset_regime_prob', 'xasset_recession_signal', 'xasset_divergence',\n",
    '    # Yield curve submodel (2)\n',
    "    'yc_spread_velocity_z', 'yc_curvature_z',\n",
    '    # ETF flow submodel (3)\n',
    "    'etf_regime_prob', 'etf_capital_intensity', 'etf_pv_divergence',\n',
    '    # Inflation expectation submodel (3)\n',
    "    'ie_regime_prob', 'ie_anchoring_z', 'ie_gold_sensitivity_z',\n",
    '    # Options market submodel (1)\n',
    "    'options_risk_regime_prob',\n",
    '    # Temporal context submodel (1)\n',
    "    'temporal_context_score',\n",
    ']\n',
    '\n',
    '# Regime-conditional features (6) - NEW in Attempt 8\n',
    'REGIME_FEATURE_COLUMNS = [\n',
    "    'real_rate_x_high_vol', 'dxy_x_high_vol',\n",
    "    'etf_flow_x_risk_off', 'yc_curvature_x_risk_off',\n",
    "    'inflation_x_trend', 'temporal_x_trend',\n",
    ']\n',
    '\n',
    'FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS\n',
    "TARGET = 'gold_return_next'\n",
    '\n',
    'assert len(BASE_FEATURE_COLUMNS) == 24\n',
    'assert len(REGIME_FEATURE_COLUMNS) == 6\n',
    'assert len(FEATURE_COLUMNS) == 30\n',
    'print(f"Features: {len(BASE_FEATURE_COLUMNS)} base + {len(REGIME_FEATURE_COLUMNS)} regime = {len(FEATURE_COLUMNS)} total")'
]))

# 5-7. Data fetching, transformation, NaN imputation (Cells 4-7)
# Keep as-is but update assertions
for i in range(4, 8):
    cell = copy.deepcopy(nb7['cells'][i])
    # In cell 7 (NaN imputation), update assertion
    if i == 7:
        cell['source'] = [
            line.replace(
                'assert all(col in final_df.columns for col in FEATURE_COLUMNS)',
                'assert all(col in final_df.columns for col in BASE_FEATURE_COLUMNS)'
            ).replace(
                'print(f"\\n✓ All {len(FEATURE_COLUMNS)} features present")',
                'print(f"\\n✓ All {len(BASE_FEATURE_COLUMNS)} base features present")'
            )
            for line in cell['source']
        ]
    new_cells.append(cell)

# 8. NEW: Regime feature generation
new_cells.append(make_markdown_cell(['## Regime Feature Generation (NEW in Attempt 8)']))

new_cells.append(make_code_cell([
    'print("\\n" + "="*60)\n',
    'print("GENERATING REGIME-CONDITIONAL FEATURES")\n',
    'print("="*60)\n',
    '\n',
    'def generate_regime_features(df):\n',
    '    print("\\nGenerating regime features...")\n',
    "    high_vol = (df['vix_persistence'] > 0.7).astype(float)\n",
    "    df['real_rate_x_high_vol'] = df['real_rate_change'] * high_vol\n",
    "    df['dxy_x_high_vol'] = df['dxy_change'] * high_vol\n",
    '    print(f"  High-vol: {high_vol.sum()}/{len(df)} ({high_vol.mean()*100:.1f}%)")\n',
    '    \n',
    "    risk_off = (df['xasset_recession_signal'] > 0.5).astype(float)\n",
    "    df['etf_flow_x_risk_off'] = df['etf_capital_intensity'] * risk_off\n",
    "    df['yc_curvature_x_risk_off'] = df['yc_curvature_z'] * risk_off\n",
    '    print(f"  Risk-off: {risk_off.sum()}/{len(df)} ({risk_off.mean()*100:.1f}%)")\n',
    '    \n',
    "    trend_on = (df['tech_trend_regime_prob'] > 0.7).astype(float)\n",
    "    df['inflation_x_trend'] = df['inflation_exp_change'] * trend_on\n",
    "    df['temporal_x_trend'] = df['temporal_context_score'] * trend_on\n",
    '    print(f"  Trend: {trend_on.sum()}/{len(df)} ({trend_on.mean()*100:.1f}%)")\n',
    '    return df\n',
    '\n',
    'final_df = generate_regime_features(final_df)\n',
    'assert all(col in final_df.columns for col in FEATURE_COLUMNS)\n',
    'print(f"\\n✓ All {len(FEATURE_COLUMNS)} features present (24 base + 6 regime)")\n',
    'print(f"✓ Dataset shape: {final_df.shape}")\n',
    'regime_nans = final_df[REGIME_FEATURE_COLUMNS].isna().sum().sum()\n',
    'print(f"✓ Regime NaNs: {regime_nans}")'
]))

#  9-11: Train/val/test split, metrics (Cells 8-11)
for i in range(8, 12):
    cell = copy.deepcopy(nb7['cells'][i])
    # Update samples-per-feature
    if i == 9:
        cell['source'] = [
            line.replace('/ 24', '/ len(FEATURE_COLUMNS)')
            for line in cell['source']
        ]
    new_cells.append(cell)

# 12-13: XGBoost Optuna - UPDATE weights and max_depth
new_cells.append(make_markdown_cell(['## XGBoost Optuna HPO (100 trials) - ATTEMPT 8']))

cell13 = copy.deepcopy(nb7['cells'][13])
cell13['source'] = [
    line.replace(
        "'max_depth': trial.suggest_int('max_depth', 2, 4)",
        "'max_depth': trial.suggest_int('max_depth', 2, 5)  # CHANGED"
    ).replace(
        '0.40 * sharpe_norm +',
        '0.35 * sharpe_norm +'
    ).replace(
        '0.30 * da_norm +',
        '0.35 * da_norm +  # CHANGED'
    )
    for line in cell13['source']
]
new_cells.append(cell13)

# 14: XGBoost HPO execution (Cell 14 unchanged)
new_cells.append(nb7['cells'][14])

# 15-18: Fallback, final training, OLS, bootstrap (Cells 15-22)
# Keep cells 15-22 from attempt 7
for i in range(15, 23):
    new_cells.append(copy.deepcopy(nb7['cells'][i]))

# NOTE: This creates a partial notebook. The complete version requires adding:
# - LightGBM HPO (new cell)
# - CatBoost HPO (new cell)
# - Stacking meta-learner (new cell)
# - Stacking vs single XGBoost comparison (new cell)
# - Updated evaluation and results cells

# For now, save partial notebook and document what's missing
nb8['cells'] = new_cells

save_notebook(nb8, NB8_PATH)
print(f"\nGenerated partial notebook at: {NB8_PATH}")
print(f"Total cells: {len(new_cells)}")
print("\n⚠️  INCOMPLETE - Missing cells:")
print("  - LightGBM HPO")
print("  - CatBoost HPO")
print("  - Stacking meta-learner")
print("  - Stacking vs XGBoost comparison")
print("  - Updated training_result.json")
print("\nThis partial notebook is NOT ready for Kaggle submission.")
print("Manual completion required or use full generation script.")
