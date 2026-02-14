import os
import subprocess
import json

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'

# Check Kaggle status
result = subprocess.run(
    ['kaggle', 'kernels', 'status', 'bigbigzabuton/gold-vix-1'],
    capture_output=True,
    text=True
)

print("=== Kaggle Status ===")
print(result.stdout)

# Download error log
subprocess.run(
    ['kaggle', 'kernels', 'output', 'bigbigzabuton/gold-vix-1', '-p', 'tmp/latest_error/'],
    capture_output=True,
    text=True
)

# Read error log
import glob
log_files = glob.glob('tmp/latest_error/*.log')
if log_files:
    with open(log_files[0], 'r', encoding='utf-8', errors='ignore') as f:
        error_log = f.read()

    # Show key errors
    print("\n=== Error Summary ===")
    for line in error_log.split('\n'):
        if 'Error' in line or 'Exception' in line or 'Traceback' in line:
            print(line)
