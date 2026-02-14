"""
Fetch base data for gold prediction model
- Gold prices (GC=F)
- 9 key features raw data
- Generate target variable (next-day return %)
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from data_fetcher import (
    fetch_yahoo_data, fetch_fred_series,
    compute_returns
)
from utils import save_csv_with_date_index, log_message, ensure_dir

# Date range: 10 years of data (sufficient for train/val/test split)
START_DATE = '2015-01-01'
END_DATE = '2025-02-14'


def fetch_gold_price():
    """Fetch gold futures price (GC=F)"""
    log_message("Fetching gold futures (GC=F)...")
    gold = fetch_yahoo_data('GC=F', START_DATE, END_DATE)

    # Use close price
    gold_price = gold[['Close']].copy()
    gold_price.columns = ['gold_price']

    # Compute next-day return (target variable)
    gold_price['gold_return_next'] = gold_price['gold_price'].pct_change(1).shift(-1) * 100

    ensure_dir('data/raw')
    save_csv_with_date_index(gold_price, 'data/raw/gold.csv')
    log_message(f"  Saved: data/raw/gold.csv ({len(gold_price)} rows)")

    return gold_price


def fetch_feature_1_real_rate():
    """Feature 1: Real Interest Rate (10Y TIPS)"""
    log_message("Fetching Real Interest Rate (DFII10)...")
    real_rate = fetch_fred_series('DFII10', START_DATE, END_DATE)
    real_rate.columns = ['real_rate']

    ensure_dir('data/raw')
    save_csv_with_date_index(real_rate, 'data/raw/real_rate.csv')
    log_message(f"  Saved: data/raw/real_rate.csv ({len(real_rate)} rows)")

    return real_rate


def fetch_feature_2_dxy():
    """Feature 2: Dollar Index (DXY)"""
    log_message("Fetching Dollar Index (DX-Y.NYB)...")
    dxy = fetch_yahoo_data('DX-Y.NYB', START_DATE, END_DATE)
    dxy_close = dxy[['Close']].copy()
    dxy_close.columns = ['dxy']

    ensure_dir('data/raw')
    save_csv_with_date_index(dxy_close, 'data/raw/dxy.csv')
    log_message(f"  Saved: data/raw/dxy.csv ({len(dxy_close)} rows)")

    return dxy_close


def fetch_feature_3_vix():
    """Feature 3: VIX"""
    log_message("Fetching VIX (VIXCLS)...")
    vix = fetch_fred_series('VIXCLS', START_DATE, END_DATE)
    vix.columns = ['vix']

    ensure_dir('data/raw')
    save_csv_with_date_index(vix, 'data/raw/vix.csv')
    log_message(f"  Saved: data/raw/vix.csv ({len(vix)} rows)")

    return vix


def fetch_feature_4_technical():
    """Feature 4: Gold Technicals (GC=F and GLD)"""
    log_message("Fetching Gold Technicals (GC=F, GLD)...")

    # GC=F already fetched, get GLD
    gld = fetch_yahoo_data('GLD', START_DATE, END_DATE)

    # Combine OHLCV
    technical = gld[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    technical.columns = ['gld_open', 'gld_high', 'gld_low', 'gld_close', 'gld_volume']

    ensure_dir('data/raw')
    save_csv_with_date_index(technical, 'data/raw/technical.csv')
    log_message(f"  Saved: data/raw/technical.csv ({len(technical)} rows)")

    return technical


def fetch_feature_5_cross_asset():
    """Feature 5: Cross-Asset (Silver, Copper, S&P500)"""
    log_message("Fetching Cross-Asset data (SI=F, HG=F, ^GSPC)...")

    silver = fetch_yahoo_data('SI=F', START_DATE, END_DATE)
    copper = fetch_yahoo_data('HG=F', START_DATE, END_DATE)
    sp500 = fetch_yahoo_data('^GSPC', START_DATE, END_DATE)

    cross_asset = pd.DataFrame(index=silver.index)
    cross_asset['silver_close'] = silver['Close']
    cross_asset['copper_close'] = copper['Close']
    cross_asset['sp500_close'] = sp500['Close']

    ensure_dir('data/raw')
    save_csv_with_date_index(cross_asset, 'data/raw/cross_asset.csv')
    log_message(f"  Saved: data/raw/cross_asset.csv ({len(cross_asset)} rows)")

    return cross_asset


def fetch_feature_6_yield_curve():
    """Feature 6: Yield Curve (10Y, 2Y)"""
    log_message("Fetching Yield Curve (DGS10, DGS2)...")

    dgs10 = fetch_fred_series('DGS10', START_DATE, END_DATE)
    dgs2 = fetch_fred_series('DGS2', START_DATE, END_DATE)

    yield_curve = pd.DataFrame(index=dgs10.index)
    yield_curve['dgs10'] = dgs10['value']
    yield_curve['dgs2'] = dgs2['value']
    yield_curve['yield_spread'] = yield_curve['dgs10'] - yield_curve['dgs2']

    ensure_dir('data/raw')
    save_csv_with_date_index(yield_curve, 'data/raw/yield_curve.csv')
    log_message(f"  Saved: data/raw/yield_curve.csv ({len(yield_curve)} rows)")

    return yield_curve


def fetch_feature_7_etf_flow():
    """Feature 7: ETF Flow Proxy (GLD volume + shares outstanding change)"""
    log_message("Fetching ETF Flow Proxy (GLD)...")

    # Use GLD data (already fetched in feature 4)
    gld = fetch_yahoo_data('GLD', START_DATE, END_DATE)

    etf_flow = pd.DataFrame(index=gld.index)
    etf_flow['gld_volume'] = gld['Volume']
    etf_flow['gld_close'] = gld['Close']
    # Volume-weighted flow proxy
    etf_flow['volume_ma20'] = etf_flow['gld_volume'].rolling(20).mean()

    ensure_dir('data/raw')
    save_csv_with_date_index(etf_flow, 'data/raw/etf_flow.csv')
    log_message(f"  Saved: data/raw/etf_flow.csv ({len(etf_flow)} rows)")

    return etf_flow


def fetch_feature_8_inflation_expectation():
    """Feature 8: Inflation Expectation (10Y Breakeven)"""
    log_message("Fetching Inflation Expectation (T10YIE)...")

    inflation_exp = fetch_fred_series('T10YIE', START_DATE, END_DATE)
    inflation_exp.columns = ['inflation_expectation']

    ensure_dir('data/raw')
    save_csv_with_date_index(inflation_exp, 'data/raw/inflation_expectation.csv')
    log_message(f"  Saved: data/raw/inflation_expectation.csv ({len(inflation_exp)} rows)")

    return inflation_exp


def fetch_feature_9_cny_demand():
    """Feature 9: CNY Demand Proxy (CNY/USD)"""
    log_message("Fetching CNY Demand Proxy (CNY=X)...")

    cny = fetch_yahoo_data('CNY=X', START_DATE, END_DATE)
    cny_rate = cny[['Close']].copy()
    cny_rate.columns = ['cny_usd']

    ensure_dir('data/raw')
    save_csv_with_date_index(cny_rate, 'data/raw/cny_demand.csv')
    log_message(f"  Saved: data/raw/cny_demand.csv ({len(cny_rate)} rows)")

    return cny_rate


def generate_target_variable(gold_price: pd.DataFrame):
    """Generate and save target variable"""
    log_message("Generating target variable (next-day return)...")

    target = gold_price[['gold_return_next']].copy()
    target = target.dropna()  # Remove last row (no next-day return)

    ensure_dir('data/processed')
    save_csv_with_date_index(target, 'data/processed/target.csv')
    log_message(f"  Saved: data/processed/target.csv ({len(target)} rows)")

    return target


if __name__ == '__main__':
    log_message("=== Starting base data fetch ===")

    try:
        # Fetch gold price and generate target
        gold_price = fetch_gold_price()
        target = generate_target_variable(gold_price)

        # Fetch all 9 features
        fetch_feature_1_real_rate()
        fetch_feature_2_dxy()
        fetch_feature_3_vix()
        fetch_feature_4_technical()
        fetch_feature_5_cross_asset()
        fetch_feature_6_yield_curve()
        fetch_feature_7_etf_flow()
        fetch_feature_8_inflation_expectation()
        fetch_feature_9_cny_demand()

        log_message("=== Base data fetch completed successfully ===")

        # Summary
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"Date range: {START_DATE} to {END_DATE}")
        print(f"Gold price rows: {len(gold_price)}")
        print(f"Target variable rows: {len(target)}")
        print(f"Target mean: {target['gold_return_next'].mean():.4f}%")
        print(f"Target std: {target['gold_return_next'].std():.4f}%")
        print("="*60)

    except Exception as e:
        log_message(f"ERROR: {str(e)}")
        raise
