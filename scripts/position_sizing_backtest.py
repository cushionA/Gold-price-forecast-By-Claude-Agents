"""
Position Sizing Backtest for Meta-Model Attempt 7
==================================================
Goal: Instead of predicting DOWN, reduce position size when confidence is low.
This avoids large losses without needing a DOWN classifier.

Strategies tested:
1. Baseline: Always full position following regression sign (= Attempt 7 as-is)
2. |prediction| magnitude sizing: position ∝ |pred|
3. Bootstrap std sizing: reduce when models disagree
4. Combined: both signals
5. Abstain on weak signals: skip when |pred| < threshold
6. Asymmetric sizing: full on UP, reduced on weak UP / skip near-zero
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ============================================================
# Data Loading
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "kaggle_results" / "meta_model_7"

def load_data():
    """Load test predictions from Attempt 7."""
    df = pd.read_csv(DATA_PATH / "test_predictions.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# ============================================================
# Metrics Calculation
# ============================================================
def calc_metrics(returns, actual_returns, positions, name, tc_bps=5):
    """
    Calculate strategy metrics.

    Args:
        returns: strategy returns (position * actual)
        actual_returns: raw actual returns
        positions: position sizes (-1 to +1)
        name: strategy name
        tc_bps: transaction cost in bps per trade (one-way)
    """
    # Transaction costs
    position_changes = np.abs(np.diff(positions, prepend=0))
    tc = position_changes * tc_bps / 10000 * 100  # convert to %
    net_returns = returns - tc

    # Direction accuracy (only where we have a position)
    active_mask = positions != 0
    if active_mask.sum() == 0:
        return {"name": name, "da": 0, "sharpe": 0, "mae": 999, "coverage": 0}

    active_actual = actual_returns[active_mask]
    active_positions = positions[active_mask]

    # DA: did we get the sign right?
    pred_sign = np.sign(active_positions)
    actual_sign = np.sign(active_actual)
    valid = (actual_sign != 0)  # exclude exact 0 returns
    if valid.sum() == 0:
        da = 0
    else:
        da = (pred_sign[valid] == actual_sign[valid]).mean()

    # Sharpe (annualized, after costs)
    if net_returns.std() == 0:
        sharpe = 0
    else:
        sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)

    # MAE (weighted by position size)
    mae = np.mean(np.abs(actual_returns[active_mask] -
                          active_positions * actual_returns[active_mask].mean()))

    # High-confidence DA (top 20% by |position * prediction|)
    active_pred_strength = np.abs(active_positions)
    hc_threshold = np.percentile(active_pred_strength, 80)
    hc_mask_inner = active_pred_strength > hc_threshold
    if hc_mask_inner.sum() > 0:
        hc_actual = active_actual[hc_mask_inner]
        hc_positions = active_positions[hc_mask_inner]
        hc_valid = np.sign(hc_actual) != 0
        if hc_valid.sum() > 0:
            hcda = (np.sign(hc_positions[hc_valid]) == np.sign(hc_actual[hc_valid])).mean()
        else:
            hcda = 0
    else:
        hcda = da

    # Cumulative return
    cum_return = (1 + net_returns / 100).prod() - 1

    # Max drawdown
    cumulative = (1 + net_returns / 100).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # DOWN day analysis
    down_days = actual_returns < 0
    if down_days.sum() > 0:
        down_day_returns = net_returns[down_days]
        avg_down_loss = down_day_returns.mean()
        # How many DOWN days did we avoid (position = 0) or reduce?
        down_day_positions = positions[down_days]
        avoided = (down_day_positions == 0).sum()
        reduced = ((down_day_positions > 0) & (down_day_positions < 1)).sum()
        full_exposed = (down_day_positions >= 1).sum()
        short = (down_day_positions < 0).sum()
    else:
        avg_down_loss = 0
        avoided = reduced = full_exposed = short = 0

    # Trade count
    n_trades = (position_changes > 0.01).sum()  # significant position changes

    return {
        "name": name,
        "da": da * 100,
        "hcda": hcda * 100,
        "sharpe": sharpe,
        "cum_return_pct": cum_return * 100,
        "max_drawdown_pct": max_dd * 100,
        "coverage": active_mask.mean() * 100,
        "n_trades": n_trades,
        "avg_down_day_loss": avg_down_loss,
        "down_days_avoided": avoided,
        "down_days_reduced": reduced,
        "down_days_full": full_exposed,
        "down_days_short": short,
        "total_down_days": down_days.sum(),
        "total_return_net": net_returns.sum(),
        "mean_position": np.mean(np.abs(positions)),
    }


# ============================================================
# Strategies
# ============================================================

def strategy_baseline(df):
    """Strategy 0: Attempt 7 as-is (always full position, follow sign)."""
    positions = np.sign(df["prediction_raw"].values).astype(float)
    returns = positions * df["actual"].values
    return positions, returns

def strategy_magnitude_sizing(df, scale_percentile=50):
    """Strategy 1: Position size ∝ |prediction| magnitude."""
    pred = df["prediction_raw"].values
    abs_pred = np.abs(pred)

    # Normalize: median |pred| = 1.0 position
    median_pred = np.percentile(abs_pred[abs_pred > 0], scale_percentile)
    if median_pred == 0:
        median_pred = 1e-6

    raw_size = abs_pred / median_pred
    positions = np.sign(pred) * np.clip(raw_size, 0, 2.0)  # cap at 2x
    returns = positions * df["actual"].values
    return positions, returns

def strategy_bootstrap_sizing(df, std_threshold_pct=75):
    """Strategy 2: Reduce position when bootstrap models disagree."""
    pred = df["prediction_raw"].values
    bstd = df["bootstrap_std"].values

    # High std → low confidence → reduce position
    # Normalize: anything above 75th percentile std → halved position
    threshold = np.percentile(bstd, std_threshold_pct)

    confidence = np.ones_like(pred)
    high_std_mask = bstd > threshold
    # Linear reduction: at threshold → 1.0, at 2x threshold → 0.0
    confidence[high_std_mask] = np.clip(1.0 - (bstd[high_std_mask] - threshold) / threshold, 0.1, 1.0)

    positions = np.sign(pred) * confidence
    returns = positions * df["actual"].values
    return positions, returns

def strategy_combined(df, pred_pct=50, std_pct=75):
    """Strategy 3: Combine magnitude + bootstrap confidence."""
    pred = df["prediction_raw"].values
    abs_pred = np.abs(pred)
    bstd = df["bootstrap_std"].values

    # Magnitude component
    median_pred = np.percentile(abs_pred[abs_pred > 0], pred_pct)
    if median_pred == 0:
        median_pred = 1e-6
    mag_score = np.clip(abs_pred / median_pred, 0, 2.0)

    # Bootstrap component
    std_threshold = np.percentile(bstd, std_pct)
    conf_score = np.ones_like(pred)
    high_std = bstd > std_threshold
    conf_score[high_std] = np.clip(1.0 - (bstd[high_std] - std_threshold) / std_threshold, 0.1, 1.0)

    # Combined: geometric mean
    combined = np.sqrt(mag_score * conf_score)
    positions = np.sign(pred) * np.clip(combined, 0, 2.0)
    returns = positions * df["actual"].values
    return positions, returns

def strategy_abstain(df, min_pred_pct=20):
    """Strategy 4: Abstain (no position) when |prediction| is too small."""
    pred = df["prediction_raw"].values
    abs_pred = np.abs(pred)

    threshold = np.percentile(abs_pred, min_pred_pct)

    positions = np.sign(pred).astype(float)
    positions[abs_pred <= threshold] = 0  # skip weak signals
    returns = positions * df["actual"].values
    return positions, returns

def strategy_abstain_sweep(df):
    """Strategy 4 with multiple thresholds."""
    results = []
    for pct in [10, 20, 30, 40, 50, 60]:
        positions, returns = strategy_abstain(df, min_pred_pct=pct)
        m = calc_metrics(returns, df["actual"].values, positions,
                         f"Abstain (skip bottom {pct}%)")
        results.append(m)
    return results

def strategy_asymmetric(df, down_scale=0.3, neutral_zone=0.005):
    """
    Strategy 5: Asymmetric sizing.
    - Strong UP prediction → full position
    - Weak UP (near zero) → reduced position
    - Prediction near zero → no position
    - DOWN prediction → small short position
    """
    pred = df["prediction_raw"].values

    positions = np.zeros_like(pred)
    for i, p in enumerate(pred):
        if p > neutral_zone:
            positions[i] = 1.0  # full long
        elif p > 0:
            positions[i] = p / neutral_zone  # linear scale in neutral zone
        elif p > -neutral_zone:
            positions[i] = 0.0  # skip
        else:
            positions[i] = down_scale * (p / neutral_zone)  # small short

    returns = positions * df["actual"].values
    return positions, returns

def strategy_regime_aware(df):
    """
    Strategy 6: Use bootstrap std as regime indicator.
    High disagreement among models → uncertain regime → reduce ALL positions.
    Rolling 20-day average of bootstrap std as regime signal.
    """
    pred = df["prediction_raw"].values
    bstd = df["bootstrap_std"].values

    # Rolling regime uncertainty (expanding window to avoid lookahead)
    regime_uncertainty = pd.Series(bstd).rolling(20, min_periods=5).mean().values

    # Normalize to [0, 1] using expanding percentile rank
    positions = np.sign(pred).astype(float)
    for i in range(len(positions)):
        if i < 5:
            continue
        # Expanding percentile rank
        historical = regime_uncertainty[5:i+1]
        current_pctile = (historical <= regime_uncertainty[i]).mean()
        # High uncertainty (top 25%) → reduce to 0.5
        if current_pctile > 0.75:
            positions[i] *= 0.5
        # Very high (top 10%) → reduce to 0.25
        if current_pctile > 0.90:
            positions[i] *= 0.5  # 0.25 total

    returns = positions * df["actual"].values
    return positions, returns


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 80)
    print("POSITION SIZING BACKTEST - Meta-Model Attempt 7")
    print("=" * 80)

    df = load_data()
    actual = df["actual"].values

    print(f"\nTest set: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total days: {len(df)}")
    print(f"UP days: {(actual > 0).sum()} ({(actual > 0).mean()*100:.1f}%)")
    print(f"DOWN days: {(actual < 0).sum()} ({(actual < 0).mean()*100:.1f}%)")
    print(f"Mean actual return: {actual.mean():.4f}%")
    print(f"Actual return std: {actual.std():.4f}%")

    # Prediction statistics
    pred = df["prediction_raw"].values
    bstd = df["bootstrap_std"].values
    print(f"\nPrediction stats:")
    print(f"  |pred| mean: {np.abs(pred).mean():.6f}")
    print(f"  |pred| median: {np.median(np.abs(pred)):.6f}")
    print(f"  |pred| std: {np.abs(pred).std():.6f}")
    print(f"  pred > 0: {(pred > 0).sum()} ({(pred > 0).mean()*100:.1f}%)")
    print(f"  pred < 0: {(pred < 0).sum()} ({(pred < 0).mean()*100:.1f}%)")
    print(f"  bootstrap_std mean: {bstd.mean():.6f}")
    print(f"  bootstrap_std median: {np.median(bstd):.6f}")
    print(f"  bootstrap_std range: [{bstd.min():.6f}, {bstd.max():.6f}]")
    print(f"  bootstrap_std 75th: {np.percentile(bstd, 75):.6f}")
    print(f"  bootstrap_std 90th: {np.percentile(bstd, 90):.6f}")

    # ---- Run all strategies ----
    all_results = []

    # 0. Baseline
    pos, ret = strategy_baseline(df)
    all_results.append(calc_metrics(ret, actual, pos, "0. Baseline (Att7 as-is)"))

    # 1. Magnitude sizing
    for pct in [30, 50, 70]:
        pos, ret = strategy_magnitude_sizing(df, scale_percentile=pct)
        all_results.append(calc_metrics(ret, actual, pos, f"1. Magnitude (p{pct})"))

    # 2. Bootstrap std sizing
    for pct in [50, 75, 90]:
        pos, ret = strategy_bootstrap_sizing(df, std_threshold_pct=pct)
        all_results.append(calc_metrics(ret, actual, pos, f"2. Bootstrap (p{pct})"))

    # 3. Combined
    pos, ret = strategy_combined(df)
    all_results.append(calc_metrics(ret, actual, pos, "3. Combined"))

    # 4. Abstain sweep
    all_results.extend(strategy_abstain_sweep(df))

    # 5. Asymmetric
    for ds in [0.0, 0.3, 0.5]:
        for nz in [0.003, 0.005, 0.01, 0.02]:
            pos, ret = strategy_asymmetric(df, down_scale=ds, neutral_zone=nz)
            all_results.append(calc_metrics(ret, actual, pos,
                                           f"5. Asym (ds={ds},nz={nz})"))

    # 6. Regime-aware
    pos, ret = strategy_regime_aware(df)
    all_results.append(calc_metrics(ret, actual, pos, "6. Regime-aware"))

    # ---- Display Results ----
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Sort by Sharpe
    all_results.sort(key=lambda x: x["sharpe"], reverse=True)

    header = f"{'Strategy':<35} {'DA%':>6} {'HCDA%':>6} {'Sharpe':>7} {'CumRet%':>8} {'MaxDD%':>7} {'Cov%':>5} {'DownAvoid':>9} {'DownReduc':>9} {'AvgPos':>6}"
    print(header)
    print("-" * len(header))

    baseline = None
    for r in all_results:
        if "Baseline" in r["name"]:
            baseline = r
            break

    for r in all_results:
        marker = " ***" if r["sharpe"] > baseline["sharpe"] and "Baseline" not in r["name"] else ""
        print(f"{r['name']:<35} {r['da']:>6.2f} {r['hcda']:>6.2f} {r['sharpe']:>7.3f} "
              f"{r['cum_return_pct']:>8.2f} {r['max_drawdown_pct']:>7.2f} "
              f"{r['coverage']:>5.1f} {r['down_days_avoided']:>9} "
              f"{r['down_days_reduced']:>9} {r['mean_position']:>6.3f}{marker}")

    # ---- DOWN day deep-dive ----
    print("\n" + "=" * 80)
    print("DOWN DAY ANALYSIS")
    print("=" * 80)

    down_mask = actual < 0
    down_df = df[down_mask].copy()

    print(f"\nTotal DOWN days: {down_mask.sum()}")
    print(f"Average DOWN return: {actual[down_mask].mean():.4f}%")
    print(f"Worst DOWN day: {actual[down_mask].min():.4f}%")

    # What did predictions look like on DOWN days?
    down_pred = pred[down_mask]
    down_bstd = bstd[down_mask]
    print(f"\nPredictions on DOWN days:")
    print(f"  pred > 0 (wrong): {(down_pred > 0).sum()} ({(down_pred > 0).mean()*100:.1f}%)")
    print(f"  pred < 0 (correct): {(down_pred < 0).sum()} ({(down_pred < 0).mean()*100:.1f}%)")
    print(f"  Mean |pred| on DOWN days: {np.abs(down_pred).mean():.6f}")
    print(f"  Mean |pred| on UP days: {np.abs(pred[~down_mask]).mean():.6f}")
    print(f"  Mean bootstrap_std on DOWN days: {down_bstd.mean():.6f}")
    print(f"  Mean bootstrap_std on UP days: {bstd[~down_mask].mean():.6f}")

    # Key question: Are DOWN days distinguishable by |pred| or bootstrap_std?
    print(f"\n--- Signal Discrimination Test ---")

    # |pred| discrimination
    abs_pred = np.abs(pred)
    for pct in [10, 20, 30, 40, 50]:
        threshold = np.percentile(abs_pred, pct)
        weak_mask = abs_pred <= threshold
        if weak_mask.sum() == 0:
            continue
        down_rate_weak = (actual[weak_mask] < 0).mean()
        down_rate_strong = (actual[~weak_mask] < 0).mean()
        avg_ret_weak = actual[weak_mask].mean()
        avg_ret_strong = actual[~weak_mask].mean()
        print(f"  Bottom {pct}% |pred| (n={weak_mask.sum()}): "
              f"DOWN rate={down_rate_weak*100:.1f}%, avg_ret={avg_ret_weak:.4f}%  | "
              f"Top {100-pct}%: DOWN rate={down_rate_strong*100:.1f}%, avg_ret={avg_ret_strong:.4f}%")

    # Bootstrap std discrimination
    print()
    for pct in [75, 80, 85, 90, 95]:
        threshold = np.percentile(bstd, pct)
        high_std_mask = bstd > threshold
        if high_std_mask.sum() == 0:
            continue
        down_rate_high = (actual[high_std_mask] < 0).mean()
        down_rate_low = (actual[~high_std_mask] < 0).mean()
        avg_ret_high = actual[high_std_mask].mean()
        avg_ret_low = actual[~high_std_mask].mean()
        print(f"  Top {100-pct}% bootstrap_std (n={high_std_mask.sum()}): "
              f"DOWN rate={down_rate_high*100:.1f}%, avg_ret={avg_ret_high:.4f}%  | "
              f"Rest: DOWN rate={down_rate_low*100:.1f}%, avg_ret={avg_ret_low:.4f}%")

    # ---- Worst days analysis ----
    print(f"\n--- Worst 10 Days ---")
    worst_idx = np.argsort(actual)[:10]
    print(f"{'Date':<12} {'Actual%':>8} {'Pred_raw':>10} {'Sign':>5} {'BootStd':>10} {'|Pred|_pct':>10}")
    for idx in worst_idx:
        pct_rank = (np.abs(pred) <= np.abs(pred[idx])).mean() * 100
        correct = "OK" if np.sign(pred[idx]) == np.sign(actual[idx]) else "NG"
        print(f"{str(df.iloc[idx]['date'])[:10]:<12} {actual[idx]:>8.4f} {pred[idx]:>10.6f} "
              f"{correct:>5} {bstd[idx]:>10.6f} {pct_rank:>10.1f}")

    # ---- Save results ----
    output = {
        "test_period": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "n_test_days": len(df),
        "baseline": baseline,
        "all_strategies": all_results,
        "best_strategy": all_results[0],
        "signal_discrimination": {
            "down_rate_bottom20_pred": float((actual[abs_pred <= np.percentile(abs_pred, 20)] < 0).mean()),
            "down_rate_top80_pred": float((actual[abs_pred > np.percentile(abs_pred, 20)] < 0).mean()),
            "avg_bootstrap_std_down_days": float(down_bstd.mean()),
            "avg_bootstrap_std_up_days": float(bstd[~down_mask].mean()),
        }
    }

    output_path = PROJECT_ROOT / "logs" / "evaluation" / "position_sizing_backtest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    best = all_results[0]
    print(f"\nBaseline Sharpe: {baseline['sharpe']:.3f}")
    print(f"Best Sharpe:     {best['sharpe']:.3f} ({best['name']})")
    delta = best['sharpe'] - baseline['sharpe']
    print(f"Delta:           {delta:+.3f}")

    if delta > 0.05:
        print(f"\n[PASS] Position sizing IMPROVED Sharpe by {delta:.3f}")
        print(f"   DOWN days avoided: {best['down_days_avoided']} / {best['total_down_days']}")
        print(f"   DOWN days reduced: {best['down_days_reduced']} / {best['total_down_days']}")
    elif delta > -0.05:
        print(f"\n[NEUTRAL] Marginal difference ({delta:+.3f}). Position sizing is roughly neutral.")
    else:
        print(f"\n[FAIL] Position sizing HURT Sharpe by {delta:.3f}. Baseline is better.")

    return output


if __name__ == "__main__":
    main()
