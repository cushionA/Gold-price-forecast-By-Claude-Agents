"""
Kaggle Datasetを作成するスクリプト
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')

# Set Kaggle credentials
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME', '')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_API_TOKEN', '')

print(f"KAGGLE_USERNAME: {os.environ.get('KAGGLE_USERNAME')}")
print(f"KAGGLE_KEY: {'*' * len(os.environ.get('KAGGLE_KEY', ''))} ({len(os.environ.get('KAGGLE_KEY', ''))} chars)")

# Import after setting env vars
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Installing kaggle package...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
    from kaggle.api.kaggle_api_extended import KaggleApi

# Create dataset
dataset_path = project_root / 'data' / 'submodel_outputs'
print(f"\nDataset path: {dataset_path}")
print(f"Files to upload:")
for csv_file in dataset_path.glob('*.csv'):
    size_kb = csv_file.stat().st_size / 1024
    print(f"  - {csv_file.name} ({size_kb:.1f} KB)")

# Initialize API
api = KaggleApi()
api.authenticate()

print("\nCreating Kaggle Dataset...")
try:
    # Create new dataset
    api.dataset_create_new(
        folder=str(dataset_path),
        public=False,
        quiet=False,
        dir_mode='zip'
    )
    print("\n✅ Dataset created successfully!")
    print("Dataset URL: https://www.kaggle.com/datasets/bigbigzabuton/gold-prediction-submodels")
except Exception as e:
    print(f"\n❌ Error creating dataset: {e}")
    print("\nTrying alternative method...")

    # Alternative: use CLI command
    import subprocess
    result = subprocess.run(
        ['kaggle', 'datasets', 'create', '-p', str(dataset_path), '-r', 'zip'],
        capture_output=True,
        text=True,
        cwd=project_root
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)

    if result.returncode == 0:
        print("\n✅ Dataset created successfully via CLI!")
    else:
        print("\n❌ Dataset creation failed")
        sys.exit(1)
