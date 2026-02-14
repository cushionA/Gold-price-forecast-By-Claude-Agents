import os
import subprocess
import json

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'

# Read current metadata
with open('tmp/dataset-metadata.json', 'r') as f:
    metadata = json.load(f)

# Update to public
metadata['info']['isPrivate'] = False

# Save updated metadata
kaggle_dir = 'kaggle_datasets/gold_target'
with open(f'{kaggle_dir}/dataset-metadata.json', 'w') as f:
    json.dump(metadata['info'], f, indent=2)

print("Updated metadata to public")
print(json.dumps(metadata['info'], indent=2))

# Try to update via version command
result = subprocess.run(
    ['kaggle', 'datasets', 'version', '-p', kaggle_dir, '-m', 'Set isPrivate=false', '--dir-mode', 'zip'],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

print("\n=== Update Result ===")
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
