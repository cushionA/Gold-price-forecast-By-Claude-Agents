"""
Data Validation Script for Gold DOWN Classifier (Attempt 1)

Verifies all 18 features can be computed from accessible data sources.
Reports date ranges, data quality issues, and saves validated raw data.

Run this BEFORE builder_model generates the Kaggle notebook.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify FRED API key
FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY not found in .env file")

# Import libraries (install if needed)
try:
    import yfinance as yf
    from fredapi import Fred
except ImportError as e:
    print(f"Missing library: {e}")
    print("Install with: pip install yfinance fredapi")
    sys.exit(1)

# Configuration
START_DATE = "2014-01-01"  # 1 year extra for warmup
END_DATE = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = "logs/datacheck/classifier_data_validation.log"
OUTPUT_FILE = "data/raw/classifier_raw.csv"

# Create directories
os.makedirs("logs/datacheck", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# Initialize log
log_lines = []

def log(msg):
    """Log message to console and file."""
    # Windows cp932 fix: replace Unicode symbols with ASCII
    msg_clean = msg.replace('✓', '[OK]').replace('❌', '[ERROR]').replace('⚠', '[WARN]')
    print(msg_clean)
    log_lines.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg_clean}")

def fetch_yfinance_data(ticker, fields=['Close']):
    """Fetch yfinance data with error handling."""
    try:
        log(f"  Fetching {ticker} from yfinance...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if df.empty:
            log(f"  ❌ {ticker}: No data returned")
            return None

        # Flatten multi-level column names (yfinance sometimes returns tuples)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Select requested fields
        if isinstance(fields, list) and len(fields) > 1:
            df = df[fields]
        elif fields == ['Close']:
            df = df[['Close']]

        log(f"  ✓ {ticker}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        log(f"  ❌ {ticker}: Error - {e}")
        return None

def fetch_fred_data(series_id):
    """Fetch FRED data with error handling."""
    try:
        fred = Fred(api_key=FRED_API_KEY)
        log(f"  Fetching {series_id} from FRED...")
        s = fred.get_series(series_id, observation_start=START_DATE)
        if s.empty:
            log(f"  ❌ {series_id}: No data returned")
            return None

        df = pd.DataFrame({series_id: s})
        df.index = pd.to_datetime(df.index)
        log(f"  ✓ {series_id}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        log(f"  ❌ {series_id}: Error - {e}")
        return None

def main():
    log("=" * 80)
    log("GOLD DOWN CLASSIFIER DATA VALIDATION")
    log("=" * 80)
    log(f"Date range: {START_DATE} to {END_DATE}")
    log("")

    # ========================================
    # STEP 1: Fetch yfinance data
    # ========================================
    log("STEP 1: Fetching yfinance data")
    log("-" * 80)

    yf_data = {}

    # GC=F: Gold futures (OHLCV)
    gc = fetch_yfinance_data('GC=F', ['Open', 'High', 'Low', 'Close', 'Volume'])
    if gc is not None:
        yf_data['GC'] = gc

    # GLD: Gold ETF (OHLCV)
    gld = fetch_yfinance_data('GLD', ['Open', 'High', 'Low', 'Close', 'Volume'])
    if gld is not None:
        yf_data['GLD'] = gld

    # SI=F: Silver futures
    si = fetch_yfinance_data('SI=F', ['Close'])
    if si is not None:
        yf_data['SI'] = si

    # HG=F: Copper futures
    hg = fetch_yfinance_data('HG=F', ['Close'])
    if hg is not None:
        yf_data['HG'] = hg

    # DX-Y.NYB: Dollar Index
    dxy = fetch_yfinance_data('DX-Y.NYB', ['Close'])
    if dxy is not None:
        yf_data['DXY'] = dxy

    # ^GSPC: S&P 500
    spx = fetch_yfinance_data('^GSPC', ['Close'])
    if spx is not None:
        yf_data['SPX'] = spx

    log("")

    # ========================================
    # STEP 2: Fetch FRED data
    # ========================================
    log("STEP 2: Fetching FRED data")
    log("-" * 80)

    fred_data = {}

    # GVZCLS: Gold VIX
    gvz = fetch_fred_data('GVZCLS')
    if gvz is not None:
        fred_data['GVZ'] = gvz

    # VIXCLS: VIX
    vix = fetch_fred_data('VIXCLS')
    if vix is not None:
        fred_data['VIX'] = vix

    # DFII10: 10Y TIPS Yield
    dfii10 = fetch_fred_data('DFII10')
    if dfii10 is not None:
        fred_data['DFII10'] = dfii10

    # DGS10: 10Y Nominal Yield
    dgs10 = fetch_fred_data('DGS10')
    if dgs10 is not None:
        fred_data['DGS10'] = dgs10

    # DGS2: 2Y Nominal Yield (for yield curve)
    dgs2 = fetch_fred_data('DGS2')
    if dgs2 is not None:
        fred_data['DGS2'] = dgs2

    log("")

    # ========================================
    # STEP 3: Merge all data sources
    # ========================================
    log("STEP 3: Merging data sources")
    log("-" * 80)

    # Start with GC=F as the base (most important)
    if 'GC' not in yf_data:
        log("❌ CRITICAL: GC=F data is missing. Cannot proceed.")
        sys.exit(1)

    df = yf_data['GC'].copy()
    df.columns = [f'GC_{col}' for col in df.columns]
    log(f"  Base data (GC=F): {len(df)} rows")

    # Merge yfinance data
    for key, data in yf_data.items():
        if key == 'GC':
            continue
        data_copy = data.copy()
        data_copy.columns = [f'{key}_{col}' for col in data_copy.columns]
        df = df.join(data_copy, how='left')
        log(f"  Merged {key}: {len(df)} rows total")

    # Merge FRED data
    for key, data in fred_data.items():
        df = df.join(data, how='left')
        log(f"  Merged {key}: {len(df)} rows total")

    log(f"  Final merged dataset: {len(df)} rows, {len(df.columns)} columns")
    log("")

    # ========================================
    # STEP 4: Data quality check
    # ========================================
    log("STEP 4: Data quality analysis")
    log("-" * 80)

    # NaN counts before forward-fill
    log("  NaN counts (before forward-fill):")
    nan_counts = df.isna().sum()
    for col in df.columns:
        if nan_counts[col] > 0:
            pct = 100 * nan_counts[col] / len(df)
            log(f"    {col}: {nan_counts[col]} ({pct:.2f}%)")

    log("")

    # Forward-fill FRED data (max 5 days)
    log("  Applying forward-fill to FRED series (max 5 days)...")
    fred_cols = ['GVZCLS', 'VIXCLS', 'DFII10', 'DGS10', 'DGS2']
    for col in fred_cols:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].ffill(limit=5)
            after = df[col].isna().sum()
            if before > 0:
                log(f"    {col}: {before} -> {after} NaNs")

    # Forward-fill yfinance data (max 3 days for weekends)
    log("  Applying forward-fill to yfinance series (max 3 days)...")
    yf_cols = [col for col in df.columns if col not in fred_cols]
    for col in yf_cols:
        before = df[col].isna().sum()
        df[col] = df[col].ffill(limit=3)
        after = df[col].isna().sum()
        if before > 0:
            log(f"    {col}: {before} -> {after} NaNs")

    log("")

    # NaN counts after forward-fill
    log("  NaN counts (after forward-fill):")
    nan_counts_after = df.isna().sum()
    total_nans = nan_counts_after.sum()
    if total_nans == 0:
        log("    ✓ No NaN values remaining")
    else:
        for col in df.columns:
            if nan_counts_after[col] > 0:
                pct = 100 * nan_counts_after[col] / len(df)
                log(f"    {col}: {nan_counts_after[col]} ({pct:.2f}%)")

    log("")

    # ========================================
    # STEP 5: Date range analysis
    # ========================================
    log("STEP 5: Date range analysis")
    log("-" * 80)

    log(f"  Overall date range: {df.index.min()} to {df.index.max()}")
    log(f"  Total trading days: {len(df)}")

    # Find effective start date (after warmup for 60-day rolling windows)
    warmup_days = 60
    effective_start_idx = warmup_days
    if len(df) > warmup_days:
        effective_start = df.index[warmup_days]
        log(f"  Effective start (after {warmup_days}-day warmup): {effective_start}")
        effective_rows = len(df) - warmup_days
        log(f"  Effective rows for training: {effective_rows}")
    else:
        log(f"  ⚠ Warning: Only {len(df)} rows, less than {warmup_days}-day warmup needed")

    log("")

    # Check 2015-01-01 availability (design requirement)
    target_start = pd.to_datetime("2015-01-01")
    if df.index.min() <= target_start:
        log(f"  ✓ Data available from {df.index.min()}, covers 2015-01-01 requirement")
    else:
        log(f"  ⚠ Warning: Data starts {df.index.min()}, after 2015-01-01 target")

    log("")

    # ========================================
    # STEP 6: Critical data checks
    # ========================================
    log("STEP 6: Critical data checks")
    log("-" * 80)

    issues = []

    # Check GVZCLS availability
    if 'GVZCLS' not in df.columns:
        issues.append("❌ GVZCLS (Gold VIX) is missing")
    else:
        gvz_start = df['GVZCLS'].first_valid_index()
        log(f"  GVZCLS starts: {gvz_start}")
        if gvz_start > target_start:
            issues.append(f"⚠ GVZCLS starts {gvz_start}, after 2015-01-01")

    # Check key tickers
    required_tickers = ['GC_Close', 'GLD_Close', 'SI_Close', 'HG_Close', 'DXY_Close', 'SPX_Close']
    for ticker in required_tickers:
        if ticker not in df.columns:
            issues.append(f"❌ {ticker} is missing")

    # Check key FRED series
    required_fred = ['VIXCLS', 'DFII10', 'DGS10']
    for series in required_fred:
        if series not in df.columns:
            issues.append(f"❌ {series} is missing")

    if issues:
        log("  Issues found:")
        for issue in issues:
            log(f"    {issue}")
    else:
        log("  ✓ All required data sources present")

    log("")

    # ========================================
    # STEP 7: Sample statistics
    # ========================================
    log("STEP 7: Sample statistics")
    log("-" * 80)

    # Gold return distribution
    if 'GC_Close' in df.columns:
        gold_return = df['GC_Close'].pct_change() * 100
        log(f"  Gold daily return (%):")
        log(f"    Mean: {gold_return.mean():.4f}%")
        log(f"    Std: {gold_return.std():.4f}%")
        log(f"    Min: {gold_return.min():.4f}%")
        log(f"    Max: {gold_return.max():.4f}%")

        # UP/DOWN balance
        up_days = (gold_return > 0).sum()
        down_days = (gold_return <= 0).sum()
        total_valid = up_days + down_days
        up_pct = 100 * up_days / total_valid if total_valid > 0 else 0
        down_pct = 100 * down_days / total_valid if total_valid > 0 else 0
        log(f"    UP days: {up_days} ({up_pct:.2f}%)")
        log(f"    DOWN days: {down_days} ({down_pct:.2f}%)")

    log("")

    # GVZ/VIX ratio check
    if 'GVZCLS' in df.columns and 'VIXCLS' in df.columns:
        gvz_vix_ratio = df['GVZCLS'] / df['VIXCLS']
        log(f"  GVZ/VIX ratio:")
        log(f"    Mean: {gvz_vix_ratio.mean():.4f}")
        log(f"    Std: {gvz_vix_ratio.std():.4f}")
        log(f"    Min: {gvz_vix_ratio.min():.4f}")
        log(f"    Max: {gvz_vix_ratio.max():.4f}")

    log("")

    # ========================================
    # STEP 8: Save validated data
    # ========================================
    log("STEP 8: Saving validated raw data")
    log("-" * 80)

    # Save to CSV
    df.to_csv(OUTPUT_FILE, encoding='utf-8')
    log(f"  ✓ Saved to: {OUTPUT_FILE}")
    log(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    log("")

    # ========================================
    # STEP 9: Recommendations
    # ========================================
    log("STEP 9: Recommendations for builder_model")
    log("-" * 80)

    if issues:
        log("  ⚠ WARNINGS:")
        for issue in issues:
            log(f"    {issue}")
        log("")

    log("  Data fetching code for Kaggle notebook:")
    log("    1. Use yfinance.download() for all yfinance tickers")
    log("    2. Use Fred(api_key=os.environ['FRED_API_KEY']) for FRED series")
    log("    3. Forward-fill: FRED max 5 days, yfinance max 3 days")
    log("    4. Drop rows with remaining NaN after forward-fill")
    log("    5. Expect ~2,700-2,800 rows after warmup period drop")

    log("")
    log("  Expected effective date range: 2015-01-15 to present")
    log(f"  Expected effective rows: {len(df) - 60} (after 60-day warmup)")

    log("")

    # ========================================
    # FINAL SUMMARY
    # ========================================
    log("=" * 80)
    log("VALIDATION COMPLETE")
    log("=" * 80)

    if not issues:
        log("✓ All data sources accessible")
        log("✓ Date ranges overlap sufficiently")
        log("✓ Data quality acceptable for training")
        log("")
        log("READY FOR builder_model to generate Kaggle notebook")
    else:
        log("⚠ Some issues detected (see above)")
        log("Review and address before proceeding to builder_model")

    log("")
    log(f"Log saved to: {LOG_FILE}")

    # Save log
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

if __name__ == "__main__":
    main()
