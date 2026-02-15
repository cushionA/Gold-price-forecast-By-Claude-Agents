import os, subprocess, json
from pathlib import Path

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set UTF-8 encoding
os.environ['PYTHONUTF8'] = '1'

# Find project root (where shared/ directory is)
project_root = Path(__file__).parent.parent.parent
state_path = project_root / 'shared' / 'state.json'

# Read state.json to get current kernel
with open(state_path) as f:
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
