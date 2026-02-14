import os
import subprocess
import sys

# Set environment variables
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'

# Submit to Kaggle
result = subprocess.run(
    ['kaggle', 'kernels', 'push', '-p', 'notebooks/dxy_1'],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

print(result.stdout)
print(result.stderr, file=sys.stderr)
sys.exit(result.returncode)
