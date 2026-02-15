"""
Submit notebook to Kaggle with proper encoding handling
"""
import os
import sys
import subprocess
from pathlib import Path

def submit_notebook(notebook_dir):
    """Submit notebook to Kaggle"""
    # Set UTF-8 encoding for subprocess
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    # Run kaggle command
    result = subprocess.run(
        ['kaggle', 'kernels', 'push', '-p', str(notebook_dir)],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python submit_to_kaggle.py <notebook_dir>")
        sys.exit(1)

    notebook_dir = Path(sys.argv[1])
    exit_code = submit_notebook(notebook_dir)
    sys.exit(exit_code)
