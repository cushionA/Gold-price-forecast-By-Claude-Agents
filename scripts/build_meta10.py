"""Build meta_model_10 notebook from meta_model_7 base."""
import json
import copy
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(BASE_DIR, 'notebooks', 'meta_model_7', 'train.ipynb')
dst_path = os.path.join(BASE_DIR, 'notebooks', 'meta_model_10', 'train.ipynb')

with open(src_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb_new = copy.deepcopy(nb)

# ========================================
# Cell 0: Update title/description
# ========================================
cell0_src = nb_new['cells'][0]['source']
new_cell0 = []
for line in cell0_src:
    line = line.replace('Attempt 7', 'Attempt 10')
    line = line.replace('+1 feature', '+2 features')
    line = line.replace('24 total features, was 23', '25 total features, was 24')
    new_cell0.append(line)
new_cell0.extend([
    '\n',
    '2. **+1 feature**: cny_demand_spread_z added (25 total features)\n',
    '   - CNY-CNH spread change z-score (capital control tension)\n',
    '   - Gate 3 PASS: DA +1.53%, Sharpe +0.217 (5/5 folds)\n',
])
nb_new['cells'][0]['source'] = new_cell0

# ========================================
# Cell 3: Add cny_demand_spread_z to FEATURE_COLUMNS
# ========================================
cell3_src = ''.join(nb_new['cells'][3]['source'])
cell3_src = cell3_src.replace(
    "    # Temporal context submodel (1) -- NEW in Attempt 7\n    'temporal_context_score',\n]",
    "    # Temporal context submodel (1)\n    'temporal_context_score',\n    # CNY demand submodel (1) -- NEW in Attempt 10\n    'cny_demand_spread_z',\n]"
)
cell3_src = cell3_src.replace('assert len(FEATURE_COLUMNS) == 24', 'assert len(FEATURE_COLUMNS) == 25')
cell3_src = cell3_src.replace('Expected 24 features', 'Expected 25 features')
nb_new['cells'][3]['source'] = [cell3_src]
assert 'cny_demand_spread_z' in cell3_src
assert '== 25' in cell3_src
print("Cell 3: FEATURE_COLUMNS updated (24->25)")

# ========================================
# Cell 5: Add cny_demand to submodel_files dict
# ========================================
cell5_src = ''.join(nb_new['cells'][5]['source'])

# Find the end of temporal_context entry and add cny_demand
# Look for the pattern: temporal_context dict followed by closing }
tc_marker = "'tz_aware': False,         # no timezone in dates\n    },\n}"
if tc_marker in cell5_src:
    replacement = (
        "'tz_aware': False,         # no timezone in dates\n    },\n"
        "    # NEW in Attempt 10\n"
        "    'cny_demand': {\n"
        "        'path': '../input/gold-prediction-submodels/cny_demand.csv',\n"
        "        'columns': ['cny_demand_spread_z'],\n"
        "        'date_col': 'Date',       # uppercase 'Date' (index column name)\n"
        "        'tz_aware': False,\n"
        "    },\n"
        "}"
    )
    cell5_src = cell5_src.replace(tc_marker, replacement)
    print("Cell 5: cny_demand added to submodel_files")
else:
    print("WARNING: temporal_context marker not found exactly")
    # Try alternative
    alt_marker = "'tz_aware': False,         # no timezone in dates"
    idx = cell5_src.rfind(alt_marker)
    if idx > 0:
        # Find the closing },\n} after this
        close_idx = cell5_src.find('\n}', idx + len(alt_marker))
        if close_idx > 0:
            insert = (
                "\n    },\n"
                "    # NEW in Attempt 10\n"
                "    'cny_demand': {\n"
                "        'path': '../input/gold-prediction-submodels/cny_demand.csv',\n"
                "        'columns': ['cny_demand_spread_z'],\n"
                "        'date_col': 'Date',\n"
                "        'tz_aware': False,"
            )
            cell5_src = cell5_src[:close_idx] + insert + cell5_src[close_idx:]
            print("Cell 5: cny_demand added (alt method)")

nb_new['cells'][5]['source'] = [cell5_src]
assert 'cny_demand_spread_z' in cell5_src
assert 'cny_demand.csv' in cell5_src

# ========================================
# Cell 7: Add cny_demand_spread_z to NaN imputation z_cols
# ========================================
cell7_src = ''.join(nb_new['cells'][7]['source'])
cell7_src = cell7_src.replace(
    "'ie_anchoring_z', 'ie_gold_sensitivity_z']",
    "'ie_anchoring_z', 'ie_gold_sensitivity_z',\n          'cny_demand_spread_z']  # NEW in Attempt 10"
)
nb_new['cells'][7]['source'] = [cell7_src]
assert 'cny_demand_spread_z' in cell7_src
print("Cell 7: cny_demand_spread_z added to z_cols imputation")

# ========================================
# Cell 15: Update fallback header
# ========================================
cell15_src = ''.join(nb_new['cells'][15]['source'])
cell15_src = cell15_src.replace('on 24 Features', 'on 25 Features')
nb_new['cells'][15]['source'] = [cell15_src]

# ========================================
# Cell 16: Update fallback title
# ========================================
cell16_src = ''.join(nb_new['cells'][16]['source'])
cell16_src = cell16_src.replace('on 24 Features', 'on 25 Features')
nb_new['cells'][16]['source'] = [cell16_src]
print("Cells 15-16: Fallback headers updated")

# ========================================
# Cell 26: Add cny_demand importance tracking
# ========================================
cell26_src = ''.join(nb_new['cells'][26]['source'])
cell26_src = cell26_src.replace('Rank {options_rank}/24', 'Rank {options_rank}/25')
cell26_src = cell26_src.replace('Rank {tc_rank}/24', 'Rank {tc_rank}/25')

# Add cny_demand tracking after temporal_context
cny_tracking = (
    '\n\n# Find cny_demand_spread_z rank (NEW in Attempt 10)\n'
    'cny_rank = (feature_ranking.reset_index(drop=True).reset_index()\n'
    "           .loc[feature_ranking['feature'] == 'cny_demand_spread_z', 'index'].values[0] + 1)\n"
    "cny_importance = feature_ranking.loc[feature_ranking['feature'] == 'cny_demand_spread_z', 'importance'].values[0]\n"
    'print(f"cny_demand_spread_z: Rank {cny_rank}/25, Importance {cny_importance:.4f}")'
)

# Insert after temporal_context print
tc_print_marker = 'print(f"temporal_context_score: Rank {tc_rank}/'
idx = cell26_src.find(tc_print_marker)
if idx > 0:
    # Find end of that line
    end_of_line = cell26_src.find('\n', idx)
    if end_of_line > 0:
        cell26_src = cell26_src[:end_of_line] + cny_tracking + cell26_src[end_of_line:]
        print("Cell 26: cny_demand importance tracking added")

nb_new['cells'][26]['source'] = [cell26_src]

# ========================================
# Cell 28: Update training_result.json
# ========================================
cell28_src = ''.join(nb_new['cells'][28]['source'])
cell28_src = cell28_src.replace("'attempt': 7", "'attempt': 10")
cell28_src = cell28_src.replace(
    "'architecture': 'XGBoost reg:squarederror + Bootstrap confidence + OLS scaling'",
    "'architecture': 'XGBoost reg:squarederror + Bootstrap confidence + OLS scaling (25 features, +cny_demand_spread_z)'"
)
cell28_src = cell28_src.replace("'n_features': 24", "'n_features': 25")
cell28_src = cell28_src.replace('len(X_train) / 24', 'len(X_train) / 25')

# Add cny_demand to feature_importance dict
cell28_src = cell28_src.replace(
    "        'temporal_context_score_importance': float(tc_importance),\n    },",
    "        'temporal_context_score_importance': float(tc_importance),\n"
    "        'cny_demand_spread_z_rank': int(cny_rank),\n"
    "        'cny_demand_spread_z_importance': float(cny_importance),\n    },"
)
nb_new['cells'][28]['source'] = [cell28_src]
print("Cell 28: attempt=10, n_features=25, cny_demand tracking")

# ========================================
# Verify and save
# ========================================
# Check all cells contain expected content
full_src = ''.join([''.join(c.get('source', [])) for c in nb_new['cells']])
checks = [
    ('cny_demand_spread_z in features', 'cny_demand_spread_z' in full_src),
    ('25 features assert', 'len(FEATURE_COLUMNS) == 25' in full_src),
    ('cny_demand.csv loading', 'cny_demand.csv' in full_src),
    ('z_cols imputation', "'cny_demand_spread_z']" in full_src),
    ('attempt 10', "'attempt': 10" in full_src),
    ('n_features 25', "'n_features': 25" in full_src),
]

print("\n=== Verification ===")
all_pass = True
for name, result in checks:
    status = "PASS" if result else "FAIL"
    print(f"  {name}: {status}")
    if not result:
        all_pass = False

if all_pass:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(nb_new, f, indent=1)
    print(f"\n=== Saved: {dst_path} ===")
else:
    print("\nERROR: Verification failed!")
