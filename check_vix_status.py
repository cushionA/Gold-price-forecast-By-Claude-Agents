import os
import subprocess

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'

# Check status
result = subprocess.run(
    ['kaggle', 'kernels', 'status', 'bigbigzabuton/gold-vix-submodel-attempt-1'],
    capture_output=True,
    text=True
)

print("=== Kaggle Status ===")
print(result.stdout)

# Try to fetch output/logs
result2 = subprocess.run(
    ['kaggle', 'kernels', 'output', 'bigbigzabuton/gold-vix-submodel-attempt-1', '-p', 'tmp/vix_error/'],
    capture_output=True,
    text=True
)

print("\n=== Output Download ===")
print(result2.stdout)
if result2.stderr:
    print("STDERR:", result2.stderr)
