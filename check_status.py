import os
import subprocess
import json

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'

# Check Kaggle status
result = subprocess.run(
    ['kaggle', 'kernels', 'status', 'bigbigzabuton/gold-dxy-1'],
    capture_output=True,
    text=True
)

print("=== Kaggle Status ===")
print(result.stdout)

# Check state.json
with open('shared/state.json', 'r') as f:
    state = json.load(f)

print("\n=== state.json ===")
print(f"Status: {state.get('status')}")
print(f"Resume from: {state.get('resume_from')}")
print(f"Current feature: {state.get('current_feature')}")
print(f"Current attempt: {state.get('current_attempt')}")
print(f"Last updated: {state.get('last_updated')}")
