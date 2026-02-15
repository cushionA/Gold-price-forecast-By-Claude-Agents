#!/bin/bash
export KAGGLE_USERNAME=BigBigZabuton
export KAGGLE_API_TOKEN=KGAT_1a02e5e1e2de5cf4f660694c73980042
export PYTHONUTF8=1

cd "$(dirname "$0")/.."
python scripts/auto_resume_after_kaggle.py
