import os
import subprocess
import sys
from pathlib import Path

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'

# Use absolute path
dataset_dir = Path(r'C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents\kaggle_datasets\gold_target')

# Create dataset
result = subprocess.run(
    ['kaggle', 'datasets', 'create', '-p', str(dataset_dir)],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace',
    cwd=r'C:\Users\tatuk\Desktop\Gold-price-forecast-By-Claude-Agents'
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print("\nReturn code:", result.returncode)

sys.exit(result.returncode)
