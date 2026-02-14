"""
Utility Functions - Common helper functions
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict, filepath: str, indent: int = 2):
    """Save dictionary to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def update_state(updates: Dict[str, Any], state_path: str = "shared/state.json"):
    """Update state.json with new values"""
    state = load_json(state_path)
    state.update(updates)
    state['last_updated'] = datetime.now().isoformat()
    save_json(state, state_path)


def log_message(message: str, log_file: Optional[str] = None):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(log_entry + '\n')


def ensure_dir(path: str):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """Format metrics dictionary for display"""
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.{precision}f}"
        else:
            formatted[key] = str(value)
    return formatted


def load_csv_with_date_index(filepath: str, date_column: str = 'date') -> pd.DataFrame:
    """Load CSV with date index"""
    df = pd.read_csv(filepath, parse_dates=[date_column])
    df.set_index(date_column, inplace=True)
    return df


def save_csv_with_date_index(df: pd.DataFrame, filepath: str):
    """Save DataFrame with date index to CSV"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)


def check_data_leakage(train_index: pd.Index, test_index: pd.Index) -> bool:
    """Check for data leakage between train and test sets"""
    overlap = train_index.intersection(test_index)
    if len(overlap) > 0:
        print(f"WARNING: Data leakage detected! {len(overlap)} overlapping dates")
        return True
    return False


def compute_delay_days(series: pd.Series) -> int:
    """
    Estimate publication delay in days (for FRED data validation)

    Args:
        series: Time series data

    Returns:
        Estimated delay in days
    """
    # Check how far back the last data point is from today
    last_date = series.index[-1]
    today = pd.Timestamp.now()
    delay = (today - last_date).days
    return delay
