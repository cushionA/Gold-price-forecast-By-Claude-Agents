"""
DXY Submodel Data Quality Check - 7-Step Standardized
datachecker agent (Haiku)
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

BASE = "C:/Users/tatuk/Desktop/Gold-price-forecast-By-Claude-Agents"

DXY_PATH   = f"{BASE}/data/processed/dxy_features_input.csv"
TARGET_PATH = f"{BASE}/data/processed/target.csv"
BASE_FEAT_PATH = f"{BASE}/data/processed/base_features.csv"
LOG_PATH   = f"{BASE}/logs/datacheck/dxy_1_check.json"

all_results = []
critical_issues = []
warnings = []

# ── STEP 1: File existence ──────────────────────────────────────────────────
def step1_file_check():
    result = {"step": "file_check", "issues": []}
    required = [DXY_PATH, TARGET_PATH, BASE_FEAT_PATH]
    for f in required:
        if not os.path.exists(f):
            msg = f"CRITICAL: {f} が存在しない"
            result["issues"].append(msg)
    result["passed"] = not any("CRITICAL" in i for i in result["issues"])
    return result

# ── STEP 2: Basic stats ─────────────────────────────────────────────────────
def step2_basic_stats(df):
    result = {"step": "basic_stats", "issues": []}
    if len(df) < 1000:
        result["issues"].append(f"WARNING: 行数が少ない ({len(df)}行)")
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].std() == 0:
            result["issues"].append(f"CRITICAL: {col} の標準偏差が0")
        if abs(df[col].min()) > 1e6 or abs(df[col].max()) > 1e6:
            result["issues"].append(f"WARNING: {col} に極端な値 (min={df[col].min():.4f}, max={df[col].max():.4f})")
    # Domain-specific range checks
    if "dxy_close" in df.columns:
        out = df[(df["dxy_close"] < 70) | (df["dxy_close"] > 130)]
        if len(out) > 0:
            result["issues"].append(
                f"CRITICAL: dxy_close が70-130の範囲外の行が {len(out)} 件 "
                f"(min={df['dxy_close'].min():.2f}, max={df['dxy_close'].max():.2f})"
            )
    if "dxy_log_ret" in df.columns:
        out = df[(df["dxy_log_ret"] < -0.05) | (df["dxy_log_ret"] > 0.05)]
        if len(out) > 0:
            result["issues"].append(
                f"WARNING: dxy_log_ret が-0.05~0.05の範囲外の行が {len(out)} 件 "
                f"(min={df['dxy_log_ret'].min():.5f}, max={df['dxy_log_ret'].max():.5f})"
            )
    result["passed"] = not any("CRITICAL" in i for i in result["issues"])
    return result

# ── STEP 3: Missing values ──────────────────────────────────────────────────
def step3_missing_values(df):
    result = {"step": "missing_values", "issues": []}
    for col in df.columns:
        pct = df[col].isnull().mean() * 100
        if pct > 20:
            result["issues"].append(f"CRITICAL: {col} の欠損率が{pct:.1f}%")
        elif pct > 5:
            result["issues"].append(f"WARNING: {col} の欠損率が{pct:.1f}%")
    # Consecutive missing
    for col in df.select_dtypes(include=[np.number]).columns:
        groups = df[col].isnull().astype(int).groupby(
            df[col].notnull().astype(int).cumsum()
        ).sum()
        max_consec = groups.max() if len(groups) > 0 else 0
        if max_consec > 10:
            result["issues"].append(f"WARNING: {col} に{max_consec}日連続欠損")
    result["passed"] = not any("CRITICAL" in i for i in result["issues"])
    return result

# ── STEP 4: Future leak check ───────────────────────────────────────────────
def step4_future_leak(df, target_col="gold_return_next"):
    result = {"step": "future_leak", "issues": []}
    if target_col not in df.columns:
        result["passed"] = True
        result["note"] = f"{target_col} カラムなし → スキップ"
        return result
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue
        corr0 = df[col].corr(df[target_col])
        corr1 = df[col].shift(1).corr(df[target_col])
        if abs(corr0) > 0.8:
            result["issues"].append(
                f"CRITICAL: {col} とtargetの相関{corr0:.3f}（リーク疑い）"
            )
        elif abs(corr0) > abs(corr1) * 3 and abs(corr0) > 0.3:
            result["issues"].append(
                f"WARNING: {col} ラグ0相関({corr0:.3f})がラグ1({corr1:.3f})より大幅に高い"
            )
    result["passed"] = not any("CRITICAL" in i for i in result["issues"])
    return result

# ── STEP 5: Temporal consistency ────────────────────────────────────────────
def step5_temporal(df):
    result = {"step": "temporal", "issues": []}
    try:
        dates = pd.to_datetime(df.index)
    except Exception as e:
        result["issues"].append(f"CRITICAL: インデックスが日付変換できない: {e}")
        result["passed"] = False
        return result
    if not dates.is_monotonic_increasing:
        result["issues"].append("CRITICAL: 日付がソートされていない")
    dupes = dates.duplicated().sum()
    if dupes > 0:
        result["issues"].append(f"CRITICAL: {dupes}件の重複日付")
    gaps = dates.to_series().diff()
    large_gaps = gaps[gaps > pd.Timedelta(days=7)]
    for date, gap in large_gaps.items():
        result["issues"].append(f"WARNING: {date.date()}に{gap.days}日のギャップ")
    result["passed"] = not any("CRITICAL" in i for i in result["issues"])
    return result

# ── STEP 6: Date alignment & frequency ─────────────────────────────────────
def step6_alignment(dxy_df, target_df, base_df):
    result = {"step": "alignment_frequency", "issues": []}
    dxy_idx = pd.to_datetime(dxy_df.index)
    tgt_idx = pd.to_datetime(target_df.index)
    base_idx = pd.to_datetime(base_df.index)

    # Overlap with target
    overlap_tgt = dxy_idx.intersection(tgt_idx)
    if len(overlap_tgt) == 0:
        result["issues"].append("CRITICAL: target.csv との日付が一致しない（オーバーラップなし）")
    else:
        coverage = len(overlap_tgt) / len(tgt_idx) * 100
        result["overlap_with_target"] = {
            "count": int(len(overlap_tgt)),
            "target_total": int(len(tgt_idx)),
            "coverage_pct": round(coverage, 2)
        }
        if coverage < 80:
            result["issues"].append(
                f"WARNING: target.csv のカバレッジが低い ({coverage:.1f}%)"
            )

    # Coverage of base_features date range
    base_start = base_idx.min()
    base_end   = base_idx.max()
    dxy_start  = dxy_idx.min()
    dxy_end    = dxy_idx.max()
    result["date_ranges"] = {
        "dxy": [str(dxy_start.date()), str(dxy_end.date())],
        "base_features": [str(base_start.date()), str(base_end.date())],
        "target": [str(pd.to_datetime(tgt_idx.min()).date()), str(pd.to_datetime(tgt_idx.max()).date())]
    }
    if dxy_start > base_start:
        result["issues"].append(
            f"WARNING: dxy データがbase_featuresの開始({base_start.date()})より遅い({dxy_start.date()})"
        )
    if dxy_end < base_end:
        result["issues"].append(
            f"WARNING: dxy データがbase_featuresの終了({base_end.date()})より早く終わっている({dxy_end.date()})"
        )

    # Business-day gap check (>3 consecutive business day gaps)
    diffs = pd.Series(dxy_idx).diff().dt.days.dropna()
    large = diffs[diffs > 5]  # >5 calendar days roughly covers >3 business days
    if len(large) > 0:
        result["issues"].append(
            f"WARNING: 3営業日以上の連続ギャップが {len(large)} 件"
        )
        # Show first 3
        for i, (idx, val) in enumerate(large.items()):
            if i >= 3:
                break
            result["issues"].append(f"  ギャップ: インデックス{idx} で {int(val)} 日")

    result["passed"] = not any("CRITICAL" in i for i in result["issues"])
    return result

# ── STEP 7: Correlation with base_features & autocorrelation ───────────────
def step7_correlation_and_autocorr(dxy_df, base_df):
    result = {"step": "correlation_autocorr", "issues": []}

    # Autocorrelation of dxy_log_ret (lag 1)
    if "dxy_log_ret" in dxy_df.columns:
        ac = dxy_df["dxy_log_ret"].autocorr(lag=1)
        result["dxy_log_ret_autocorr_lag1"] = round(float(ac), 5)
        if abs(ac) >= 0.99:
            result["issues"].append(
                f"CRITICAL: dxy_log_ret の自己相関(lag=1)が{ac:.4f}（非定常の疑い）"
            )
        elif abs(ac) > 0.9:
            result["issues"].append(
                f"WARNING: dxy_log_ret の自己相関(lag=1)が高い ({ac:.4f})"
            )
        else:
            result["note_autocorr"] = f"dxy_log_ret autocorr(lag1)={ac:.5f} (OK)"

    # Correlation with base_features dxy_change
    if "dxy_log_ret" in dxy_df.columns and "dxy_change" in base_df.columns:
        merged = pd.merge(
            dxy_df[["dxy_log_ret"]].rename(columns={"dxy_log_ret": "dxy_lr"}),
            base_df[["dxy_change"]],
            left_index=True, right_index=True, how="inner"
        ).dropna()
        if len(merged) > 100:
            r = merged["dxy_lr"].corr(merged["dxy_change"])
            result["corr_dxy_log_ret_vs_base_dxy_change"] = round(float(r), 6)
            if r < 0.99:
                result["issues"].append(
                    f"WARNING: dxy_log_ret と base_features.dxy_change の相関が{r:.4f} (<0.99期待値)"
                )
            else:
                result["note_corr"] = f"dxy_log_ret vs base_features.dxy_change r={r:.6f} (OK)"
        else:
            result["issues"].append(
                f"WARNING: base_featuresとのマージ後サンプル数が少ない ({len(merged)}行)"
            )
    else:
        missing = []
        if "dxy_log_ret" not in dxy_df.columns:
            missing.append("dxy_log_ret (dxy_features_input)")
        if "dxy_change" not in base_df.columns:
            missing.append("dxy_change (base_features)")
        result["note_corr"] = f"相関チェックスキップ: {', '.join(missing)} が存在しない"

    result["passed"] = not any("CRITICAL" in i for i in result["issues"])
    return result

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=== DXY Submodel 7-Step Data Quality Check ===\n")

    # STEP 1
    s1 = step1_file_check()
    all_results.append(s1)
    status = "PASS" if s1["passed"] else "FAIL"
    print(f"STEP 1 File Check: {status}")
    for i in s1["issues"]:
        print(f"  {i}")
    if not s1["passed"]:
        # Cannot proceed without files
        report = build_report(all_results)
        save_report(report)
        return report

    # Load data
    dxy_df  = pd.read_csv(DXY_PATH,   index_col=0, parse_dates=True)
    tgt_df  = pd.read_csv(TARGET_PATH, index_col=0, parse_dates=True)
    base_df = pd.read_csv(BASE_FEAT_PATH, index_col=0, parse_dates=True)

    print(f"\nLoaded dxy_features_input: {dxy_df.shape}, columns={list(dxy_df.columns)}")
    print(f"Loaded target: {tgt_df.shape}, columns={list(tgt_df.columns)}")
    print(f"Loaded base_features: {base_df.shape}, columns={list(base_df.columns)[:10]}...")

    # Merge target into dxy_df for leak check
    tgt_col = "gold_return_next"
    if tgt_col in tgt_df.columns:
        dxy_with_tgt = dxy_df.join(tgt_df[[tgt_col]], how="left")
    else:
        dxy_with_tgt = dxy_df.copy()

    # STEP 2
    s2 = step2_basic_stats(dxy_df)
    all_results.append(s2)
    status = "PASS" if s2["passed"] else "FAIL"
    print(f"\nSTEP 2 Basic Stats: {status}")
    for i in s2["issues"]:
        print(f"  {i}")

    # STEP 3
    s3 = step3_missing_values(dxy_df)
    all_results.append(s3)
    status = "PASS" if s3["passed"] else "FAIL"
    print(f"\nSTEP 3 Missing Values: {status}")
    for i in s3["issues"]:
        print(f"  {i}")

    # STEP 4
    s4 = step4_future_leak(dxy_with_tgt, target_col=tgt_col)
    all_results.append(s4)
    status = "PASS" if s4["passed"] else "FAIL"
    print(f"\nSTEP 4 Future Leak: {status}")
    for i in s4["issues"]:
        print(f"  {i}")

    # STEP 5
    s5 = step5_temporal(dxy_df)
    all_results.append(s5)
    status = "PASS" if s5["passed"] else "FAIL"
    print(f"\nSTEP 5 Temporal Consistency: {status}")
    for i in s5["issues"]:
        print(f"  {i}")

    # STEP 6
    s6 = step6_alignment(dxy_df, tgt_df, base_df)
    all_results.append(s6)
    status = "PASS" if s6["passed"] else "FAIL"
    print(f"\nSTEP 6 Alignment & Frequency: {status}")
    for i in s6["issues"]:
        print(f"  {i}")

    # STEP 7
    s7 = step7_correlation_and_autocorr(dxy_df, base_df)
    all_results.append(s7)
    status = "PASS" if s7["passed"] else "FAIL"
    print(f"\nSTEP 7 Correlation & Autocorr: {status}")
    for i in s7["issues"]:
        print(f"  {i}")
    if "note_autocorr" in s7:
        print(f"  {s7['note_autocorr']}")
    if "note_corr" in s7:
        print(f"  {s7['note_corr']}")

    report = build_report(all_results)
    save_report(report)
    return report


def build_report(results):
    crit = [i for r in results for i in r.get("issues", []) if "CRITICAL" in i]
    warn = [i for r in results for i in r.get("issues", []) if "WARNING" in i]
    if crit:
        action = "REJECT"
    elif len(warn) > 5:
        action = "CONDITIONAL_PASS"
    else:
        action = "PASS"
    return {
        "feature": "dxy",
        "attempt": 1,
        "timestamp": datetime.now().isoformat(),
        "steps": results,
        "critical_issues": crit,
        "warnings": warn,
        "action": action,
        "overall_passed": action != "REJECT"
    }


def save_report(report):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nReport saved to: {LOG_PATH}")


if __name__ == "__main__":
    report = main()
    print("\n" + "="*50)
    print(f"FINAL VERDICT: {report['action']}")
    print(f"Critical issues: {len(report['critical_issues'])}")
    print(f"Warnings: {len(report['warnings'])}")
    if report["critical_issues"]:
        print("\nCritical issues:")
        for c in report["critical_issues"]:
            print(f"  {c}")
    if report["warnings"]:
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  {w}")
