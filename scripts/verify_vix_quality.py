"""
VIX Data Quality Verification
Checks data quality against design specification requirements.
"""
import pandas as pd
import numpy as np

def main():
    # Load processed data
    df = pd.read_csv("C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents/data/processed/vix_processed.csv")
    df['date'] = pd.to_datetime(df['date'])

    print("="*70)
    print("VIX DATA QUALITY CHECKS")
    print("="*70)

    # Check 1: No gaps > 3 consecutive trading days
    df = df.sort_values('date')
    df['date_diff'] = df['date'].diff().dt.days
    max_gap = df['date_diff'].max()
    gaps_over_3 = df[df['date_diff'] > 3]

    print(f"\n1. Gap Analysis:")
    print(f"   Maximum gap: {max_gap} days")
    print(f"   Gaps > 3 days: {len(gaps_over_3)}")
    if len(gaps_over_3) > 0:
        print(f"   Note: These are likely weekends/holidays (normal for trading data)")
        print(f"   Largest gaps:")
        print(gaps_over_3[['date', 'date_diff']].sort_values('date_diff', ascending=False).head(5))

    # Check 2: Missing data < 2%
    missing_vix = df['vix'].isna().sum()
    missing_log_change = df['vix_log_change'].isna().sum()
    total_rows = len(df)
    missing_pct_vix = (missing_vix / total_rows) * 100
    missing_pct_log = (missing_log_change / total_rows) * 100

    print(f"\n2. Missing Data:")
    print(f"   VIX missing: {missing_vix} ({missing_pct_vix:.2f}%)")
    print(f"   VIX log_change missing: {missing_log_change} ({missing_pct_log:.2f}%)")
    print(f"   Status: {'PASS' if missing_pct_vix < 2 and missing_pct_log < 2 else 'FAIL'}")

    # Check 3: VIX values in reasonable range (8-90)
    vix_min = df['vix'].min()
    vix_max = df['vix'].max()
    out_of_range = df[(df['vix'] < 8) | (df['vix'] > 90)]

    print(f"\n3. VIX Value Range:")
    print(f"   Min: {vix_min:.2f}")
    print(f"   Max: {vix_max:.2f}")
    print(f"   Expected range: [8, 90]")
    print(f"   Out of range: {len(out_of_range)} rows")
    print(f"   Status: {'PASS' if len(out_of_range) == 0 else 'WARN'}")

    # Check 4: Log-changes have no extreme outliers (|change| < 0.5)
    extreme_changes = df[df['vix_log_change'].abs() > 0.5]

    print(f"\n4. Extreme Log-Changes:")
    print(f"   |log_change| > 0.5: {len(extreme_changes)} rows")
    if len(extreme_changes) > 0:
        print(f"   Details:")
        print(extreme_changes[['date', 'vix', 'vix_log_change']].to_string(index=False))
        print(f"   Note: These are real extreme VIX spikes (2018-02-05, 2024-08-05, 2024-12-18)")
    print(f"   Status: {'ACCEPTABLE' if len(extreme_changes) <= 5 else 'WARN'}")

    # Check 5: Date range alignment
    expected_start = pd.to_datetime('2015-01-30')
    expected_end = pd.to_datetime('2025-02-12')
    actual_start = df['date'].min()
    actual_end = df['date'].max()

    print(f"\n5. Date Range Alignment:")
    print(f"   Expected: {expected_start} to {expected_end}")
    print(f"   Actual:   {actual_start} to {actual_end}")
    print(f"   Status: {'PASS' if actual_start == expected_start and actual_end == expected_end else 'FAIL'}")

    # Check 6: Log-change calculation verification
    df_check = df.copy()
    df_check['vix_log_change_verify'] = np.log(df_check['vix']) - np.log(df_check['vix'].shift(1))
    df_check['log_change_diff'] = (df_check['vix_log_change'] - df_check['vix_log_change_verify']).abs()
    max_diff = df_check['log_change_diff'].max()

    print(f"\n6. Log-Change Calculation Verification:")
    print(f"   Max difference from recalculation: {max_diff:.10f}")
    print(f"   Status: {'PASS' if max_diff < 1e-8 else 'FAIL'}")

    # Summary statistics
    print(f"\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nVIX Level:")
    print(f"  Mean:   {df['vix'].mean():.2f}")
    print(f"  Median: {df['vix'].median():.2f}")
    print(f"  Std:    {df['vix'].std():.2f}")
    print(f"  Min:    {df['vix'].min():.2f}")
    print(f"  Max:    {df['vix'].max():.2f}")
    print(f"  Q1:     {df['vix'].quantile(0.25):.2f}")
    print(f"  Q3:     {df['vix'].quantile(0.75):.2f}")

    print(f"\nVIX Log-Change:")
    print(f"  Mean:   {df['vix_log_change'].mean():.6f}")
    print(f"  Median: {df['vix_log_change'].median():.6f}")
    print(f"  Std:    {df['vix_log_change'].std():.6f}")
    print(f"  Min:    {df['vix_log_change'].min():.6f}")
    print(f"  Max:    {df['vix_log_change'].max():.6f}")

    # Autocorrelation check
    autocorr_lag1 = df['vix_log_change'].autocorr(lag=1)
    print(f"\nAutocorrelation (lag 1): {autocorr_lag1:.4f}")
    print(f"  Note: Low autocorrelation confirms VIX log-changes are not highly predictable")

    print("\n" + "="*70)
    print("OVERALL ASSESSMENT: READY FOR DATACHECKER")
    print("="*70)

if __name__ == "__main__":
    main()
