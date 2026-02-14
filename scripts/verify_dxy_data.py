"""
Verify DXY data quality against design requirements
"""
import pandas as pd
import numpy as np
import json


def main():
    # Load data
    dxy_data = pd.read_csv('data/multi_country/dxy_constituents.csv', parse_dates=['Date'], index_col='Date')
    base_features = pd.read_csv('data/processed/base_features.csv', parse_dates=['Date'], index_col='Date')

    print("="*70)
    print("DXY DATA QUALITY VERIFICATION REPORT")
    print("="*70)

    # 1. Date range alignment
    print("\n1. DATE RANGE ALIGNMENT")
    print("-" * 70)
    print(f"Base features: {base_features.index[0]} to {base_features.index[-1]} ({len(base_features)} rows)")
    print(f"DXY data:      {dxy_data.index[0]} to {dxy_data.index[-1]} ({len(dxy_data)} rows)")

    # Check overlap
    overlap = dxy_data.index.intersection(base_features.index)
    print(f"Overlap:       {len(overlap)} rows ({len(overlap)/len(base_features)*100:.2f}%)")

    if len(overlap) != len(base_features):
        print("WARNING: Not all base_features dates are in DXY data")
        missing = base_features.index.difference(dxy_data.index)
        print(f"Missing dates: {len(missing)}")
    else:
        print("[OK] Perfect alignment with base_features")

    # 2. Missing values
    print("\n2. MISSING VALUES")
    print("-" * 70)
    missing_report = {}
    for col in dxy_data.columns:
        n_missing = dxy_data[col].isna().sum()
        pct_missing = n_missing / len(dxy_data) * 100
        missing_report[col] = {'count': n_missing, 'pct': pct_missing}
        status = "[OK]" if pct_missing < 1.0 else "[FAIL]"
        print(f"{col:15} {n_missing:5} ({pct_missing:6.2f}%)  {status}")

    # 3. Currency direction verification
    print("\n3. CURRENCY DIRECTION (verify values are reasonable)")
    print("-" * 70)
    print("Currency pair directions:")
    print("  EURUSD (inverse): EUR/USD - higher = weaker USD")
    print("  GBPUSD (inverse): GBP/USD - higher = weaker USD")
    print("  USDJPY (same):    USD/JPY - higher = stronger USD")
    print("  USDCAD (same):    USD/CAD - higher = stronger USD")
    print("  USDSEK (same):    USD/SEK - higher = stronger USD")
    print("  USDCHF (same):    USD/CHF - higher = stronger USD")

    print("\nTypical value ranges:")
    for col in dxy_data.columns:
        print(f"  {col:15} [{dxy_data[col].min():.4f}, {dxy_data[col].max():.4f}]")

    # 4. Data gaps
    print("\n4. DATA GAPS (consecutive missing trading days)")
    print("-" * 70)
    date_diff = dxy_data.index.to_series().diff()
    max_gap = date_diff.max().days
    gaps = date_diff[date_diff.dt.days > 3]
    print(f"Maximum gap: {max_gap} days")
    print(f"Number of gaps > 3 days: {len(gaps)}")
    if len(gaps) > 0 and len(gaps) <= 5:
        print("Gap details:")
        for date, gap in gaps.items():
            print(f"  {date}: {gap.days} days")
    status = "[OK]" if max_gap <= 5 else "[WARNING]"
    print(f"Status: {status} (threshold: <=5 days for weekends + holidays)")

    # 5. Basic statistics
    print("\n5. BASIC STATISTICS")
    print("-" * 70)
    print(dxy_data.describe().T[['mean', 'std', 'min', 'max']])

    # 6. Correlation with base DXY feature
    print("\n6. CORRELATION WITH BASE DXY FEATURE")
    print("-" * 70)
    if 'dxy_dxy' in base_features.columns:
        # Merge on dates
        merged = base_features[['dxy_dxy']].merge(dxy_data[['dxy_close']], left_index=True, right_index=True)
        corr = merged['dxy_dxy'].corr(merged['dxy_close'])
        print(f"Correlation between base dxy_dxy and dxy_close: {corr:.6f}")
        if corr > 0.99:
            print("[OK] Excellent correlation (same source)")
        else:
            print(f"[WARNING] Lower than expected correlation")

    # 7. Sample data inspection
    print("\n7. SAMPLE DATA (first 5 and last 5 rows)")
    print("-" * 70)
    print("\nFirst 5 rows:")
    print(dxy_data.head())
    print("\nLast 5 rows:")
    print(dxy_data.tail())

    # 8. Currency pair reasonableness checks
    print("\n8. CURRENCY PAIR REASONABLENESS CHECKS")
    print("-" * 70)

    checks = {
        'eur_usd': (0.9, 1.3, "EUR/USD typically 0.9-1.3"),
        'jpy': (90.0, 170.0, "USD/JPY typically 90-170"),
        'gbp_usd': (1.0, 1.7, "GBP/USD typically 1.0-1.7"),
        'usd_cad': (1.0, 1.6, "USD/CAD typically 1.0-1.6"),
        'usd_sek': (6.0, 12.0, "USD/SEK typically 6-12"),
        'usd_chf': (0.7, 1.1, "USD/CHF typically 0.7-1.1")
    }

    for col, (low, high, desc) in checks.items():
        within_range = ((dxy_data[col] >= low) & (dxy_data[col] <= high)).all()
        status = "[OK]" if within_range else "[WARNING]"
        actual_range = f"[{dxy_data[col].min():.4f}, {dxy_data[col].max():.4f}]"
        print(f"{col:10} {status}  {desc}")
        print(f"           Actual: {actual_range}")

    # 9. Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_checks_pass = True
    issues = []

    if len(overlap) < len(base_features):
        all_checks_pass = False
        issues.append(f"Missing {len(base_features) - len(overlap)} dates vs base_features")

    if any(v['pct'] >= 1.0 for v in missing_report.values()):
        all_checks_pass = False
        issues.append("Some columns have >1% missing data")

    if max_gap > 5:
        all_checks_pass = False
        issues.append(f"Large gap detected: {max_gap} days")

    if all_checks_pass:
        print("\n[OK] ALL CHECKS PASSED")
        print("\nData is ready for datachecker validation.")
    else:
        print("\n[FAIL] SOME ISSUES DETECTED")
        print("\nIssues:")
        for issue in issues:
            print(f"  - {issue}")

    print("\nFiles created:")
    print("  - data/multi_country/dxy_constituents.csv (processed data)")
    print("  - data/raw/dxy_currencies.csv (raw prices)")
    print("  - data/multi_country/dxy_metadata.json (metadata)")
    print("  - src/fetch_dxy.py (reproducible fetch function)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
