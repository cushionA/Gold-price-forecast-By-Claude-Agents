"""
Submit Kaggle kernel and update state.json
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Kaggle credentials for API
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_API_TOKEN')

from kaggle import api

def submit_kernel(feature, attempt):
    """Submit kernel to Kaggle and update state"""
    notebook_dir = Path(f"notebooks/{feature}_{attempt}")

    if not notebook_dir.exists():
        print(f"Error: Notebook directory {notebook_dir} not found")
        return False

    # Read kernel metadata to get kernel ID
    metadata_path = notebook_dir / "kernel-metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    kernel_id = metadata['id']
    print(f"\nSubmitting kernel: {kernel_id}")
    print(f"Notebook directory: {notebook_dir}")

    try:
        # Push kernel to Kaggle using Python API
        # Kaggle API outputs to stderr which causes encoding issues, suppress it
        import sys
        import io
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            api.kernels_push(str(notebook_dir))
        finally:
            sys.stderr = old_stderr

        print(f"Kernel submitted successfully!")

        # Update state.json
        state_path = Path("shared/state.json")
        with open(state_path) as f:
            state = json.load(f)

        state.update({
            "status": "waiting_training",
            "phase": "1.5_smoke_test",
            "current_feature": feature,
            "current_attempt": attempt,
            "resume_from": "evaluator",
            "kaggle_kernel": kernel_id,
            "submitted_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        })

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"OK Updated state.json")
        print(f"\nKaggle kernel is now running in the cloud.")
        print(f"Check status: kaggle kernels status {kernel_id}")
        print(f"\nYou can now shut down your PC. Training will continue on Kaggle.")
        print(f"When ready to resume, run: 'Resume from where we left off'")

        return True

    except Exception as e:
        print(f"Error submitting kernel: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/kaggle_submit.py <feature> <attempt>")
        sys.exit(1)

    feature = sys.argv[1]
    attempt = int(sys.argv[2])

    success = submit_kernel(feature, attempt)
    sys.exit(0 if success else 1)
