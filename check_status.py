import os
import subprocess
import json

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'

result = subprocess.run(
    ['kaggle', 'kernels', 'status', 'bigbigzabuton/gold-vix-1'],
    capture_output=True,
    text=True
)

print("=== Kaggle Status ===")
print(result.stdout)

with open('shared/state.json', 'r') as f:
    state = json.load(f)

print("\n=== state.json ===")
print(f"Status: {state.get('status')}")
print(f"Last updated: {state.get('last_updated')}")
