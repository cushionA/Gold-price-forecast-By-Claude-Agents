"""
Check Kaggle kernel status and fetch results when complete
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Kaggle credentials for API
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_API_TOKEN')

from kaggle import api

def check_and_fetch_results(feature, attempt):
    """Check kernel status and fetch results if complete"""
    kernel_id = f"{os.getenv('KAGGLE_USERNAME').lower()}/gold-{feature}-{attempt}"

    print(f"Checking kernel: {kernel_id}")

    # Suppress stderr to avoid encoding issues
    import io
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()

    try:
        # Check status
        status_result = api.kernels_status(kernel_id)
        status = status_result['status']
        print(f"Kernel status: {status}")

        if status == "running" or status == "queued":
            print(f"\nKernel is still {status}. Training continues on Kaggle cloud.")
            print("Check again later with the same command.")
            return False

        elif status == "error":
            print(f"\nKernel failed with error!")
            failure_msg = status_result.get('failureMessage', 'Unknown error')
            print(f"Error message: {failure_msg}")

            # Update state
            state_path = Path("shared/state.json")
            with open(state_path) as f:
                state = json.load(f)

            state.update({
                "status": "in_progress",
                "resume_from": "builder_model",
                "error_context": {
                    "stage": "kaggle_training",
                    "message": failure_msg,
                    "timestamp": datetime.now().isoformat()
                },
                "last_updated": datetime.now().isoformat()
            })

            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

            print("State updated. Review error and resubmit if needed.")
            return False

        elif status == "complete":
            print(f"\nKernel completed successfully! Fetching results...")

            # Create output directory
            output_dir = Path(f"data/submodel_outputs/{feature}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Fetch output files
            temp_dir = Path(f"temp_{feature}_{attempt}")
            temp_dir.mkdir(exist_ok=True)

            api.kernels_output(kernel_id, str(temp_dir))

            # Move files to proper locations
            files_moved = []

            # submodel_output.csv -> data/submodel_outputs/{feature}.csv
            output_csv = temp_dir / "submodel_output.csv"
            if output_csv.exists():
                target_csv = output_dir / f"{feature}.csv"
                shutil.move(str(output_csv), str(target_csv))
                files_moved.append(str(target_csv))
                print(f"OK Moved: {target_csv}")

            # model.pt -> models/submodels/{feature}/model.pt
            model_pt = temp_dir / "model.pt"
            if model_pt.exists():
                model_dir = Path(f"models/submodels/{feature}")
                model_dir.mkdir(parents=True, exist_ok=True)
                target_pt = model_dir / "model.pt"
                shutil.move(str(model_pt), str(target_pt))
                files_moved.append(str(target_pt))
                print(f"OK Moved: {target_pt}")

            # training_result.json -> logs/training/{feature}_{attempt}.json
            result_json = temp_dir / "training_result.json"
            if result_json.exists():
                log_dir = Path("logs/training")
                log_dir.mkdir(parents=True, exist_ok=True)
                target_json = log_dir / f"{feature}_{attempt}.json"
                shutil.move(str(result_json), str(target_json))
                files_moved.append(str(target_json))
                print(f"OK Moved: {target_json}")

            # Clean up temp directory
            shutil.rmtree(temp_dir)

            print(f"\nFetched {len(files_moved)} files from Kaggle")

            # Update state
            state_path = Path("shared/state.json")
            with open(state_path) as f:
                state = json.load(f)

            state.update({
                "status": "in_progress",
                "phase": "1.5_smoke_test",
                "resume_from": "evaluator",
                "kaggle_kernel": None,
                "submitted_at": None,
                "last_updated": datetime.now().isoformat()
            })

            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)

            print("OK State updated. Ready for evaluator.")
            return True

        else:
            print(f"Unknown status: {status}")
            return False

    finally:
        sys.stderr = old_stderr

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/kaggle_fetch_results.py <feature> <attempt>")
        sys.exit(1)

    feature = sys.argv[1]
    attempt = int(sys.argv[2])

    success = check_and_fetch_results(feature, attempt)
    sys.exit(0 if success else 1)
