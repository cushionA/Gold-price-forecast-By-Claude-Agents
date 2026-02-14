import os, subprocess, json
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'

# Read state.json to get current kernel
with open('shared/state.json') as f:
    state = json.load(f)

kernel_id = state.get('kaggle_kernel', 'bigbigzabuton/gold-vix-1')

result = subprocess.run(['kaggle', 'kernels', 'status', kernel_id], capture_output=True, text=True)
print("=== Kaggle Status ===")
print(f"Kernel: {kernel_id}")
print(result.stdout)

print("\n=== state.json ===")
print(f"Status: {state.get('status')}")
print(f"Feature: {state.get('current_feature')}")
print(f"Attempt: {state.get('current_attempt')}")
print(f"Resume from: {state.get('resume_from')}")
print(f"Last updated: {state.get('last_updated')}")
