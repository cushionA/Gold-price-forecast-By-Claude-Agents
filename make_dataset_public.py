import os
import subprocess

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_357794ad87b13a4ecd000b7ff9ac57ea'
os.environ['PYTHONUTF8'] = '1'

# Update dataset metadata to make it public
import json
metadata_path = 'kaggle_datasets/gold_target/dataset-metadata.json'

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Ensure it's public
metadata['isPrivate'] = False

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("Updated metadata to public")

# Update the dataset
result = subprocess.run(
    ['kaggle', 'datasets', 'version', '-p', 'kaggle_datasets/gold_target', '-m', 'Make dataset public'],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print("Return code:", result.returncode)
