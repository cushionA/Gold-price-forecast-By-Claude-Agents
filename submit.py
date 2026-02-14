import os, subprocess, sys
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'
result = subprocess.run(['kaggle', 'kernels', 'push', '-p', 'notebooks/vix_1'], capture_output=True, text=True, encoding='utf-8', errors='replace')
print(result.stdout)
sys.exit(result.returncode)
