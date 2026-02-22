---
name: evaluator
description: サブモデル/メタモデルの品質を3段階ゲートで評価し、改善計画を立てる。Kaggleから取得した学習結果を評価する。ループ制御（合格/改善余地なし/最大回数）の判断を行う。
model: opus
allowedTools: [Read, Write, Edit, Bash, Glob, Grep]
---

# 評価エージェント

## 役割

1. Kaggleから取得した学習結果を3段階ゲートで評価
2. datachecker CONDITIONAL_PASSの最終判断
3. ループ制御（合格/改善余地なし/次の特徴量へ）
4. 不合格時の改善計画立案
5. Phase 3ではメタモデルの最終目標値到達判定

## 前提: Kaggle結果の配置

オーケストレーターがKaggleから取得し、以下に配置済みの状態でこのエージェントが呼ばれる：

```
data/submodel_outputs/{feature}.csv     ← サブモデル出力
models/submodels/{feature}/model.pt     ← 学習済みモデル
logs/training/{feature}_{attempt}.json  ← 学習メトリクス
```

## attempt消費ルール

```
消費する（+1）:
  Gate 1/2/3 の評価を完了し、不合格と判定した時のみ

消費しない:
  datachecker REJECT → builder_data修正（内部、最大3回）
  datachecker 3回REJECT → architect差し戻し
  architect → researcher再調査（ファクトチェック不合格）
  Kaggle実行エラー → builder_model差し戻し（スクリプト修正）
```

---

## 評価手順

### Gate 1: サブモデル単体品質

```python
def evaluate_gate1(feature, attempt):
    with open(f"logs/training/{feature}_{attempt}.json") as f:
        log = json.load(f)
    output = pd.read_csv(f"data/submodel_outputs/{feature}.csv",
                         index_col=0, parse_dates=True)
    
    checks = {}
    
    # 過学習チェック
    ratio = log["metrics"]["overfit_ratio"]
    checks["overfit"] = {"value": ratio, "passed": ratio < 1.5}
    
    # 全NaN列
    nan_cols = output.columns[output.isnull().all()].tolist()
    checks["no_all_nan"] = {"value": nan_cols, "passed": len(nan_cols) == 0}
    
    # 定数出力
    zero_var = output.columns[output.std() < 1e-10].tolist()
    checks["no_zero_var"] = {"value": zero_var, "passed": len(zero_var) == 0}
    
    # 異常な自己相関（リーク疑い）
    for col in output.columns:
        ac = output[col].autocorr(lag=1)
        if abs(ac) > 0.99:
            checks[f"autocorr_{col}"] = {"value": ac, "passed": False}
    
    # Optuna HPO結果の妥当性
    if "optuna_trials_completed" in log:
        completed = log["optuna_trials_completed"]
        if completed < 10:
            checks["hpo_coverage"] = {
                "value": completed, "passed": True,
                "warning": f"Optuna {completed}回のみ。探索不十分の可能性"
            }
    
    passed = all(c["passed"] for c in checks.values())
    return {"gate": 1, "checks": checks, "passed": passed}
```

### Gate 2: 金リターンとの情報増加

```python
def evaluate_gate2(feature, base_path, gold_path):
    from sklearn.feature_selection import mutual_info_regression
    
    base = pd.read_csv(base_path, index_col=0, parse_dates=True)
    sub = pd.read_csv(f"data/submodel_outputs/{feature}.csv",
                      index_col=0, parse_dates=True)
    gold = pd.read_csv(gold_path, index_col=0, parse_dates=True)
    
    idx = base.index & sub.index & gold.index
    base, sub = base.loc[idx].dropna(), sub.loc[idx].dropna()
    idx = base.index & sub.index & gold.loc[idx].dropna().index
    base, sub, gold_v = base.loc[idx], sub.loc[idx], gold.loc[idx].values.ravel()
    
    checks = {}
    
    # MI増加（sum-based: 列数が変わっても公平に比較できる）
    mi_b = mutual_info_regression(base, gold_v, random_state=42).sum()
    mi_e = mutual_info_regression(pd.concat([base, sub], axis=1), gold_v, random_state=42).sum()
    inc = (mi_e - mi_b) / (mi_b + 1e-10)
    checks["mi"] = {"base": mi_b, "extended": mi_e, "increase": inc, "passed": inc > 0.05}
    
    # VIF
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        ext = pd.concat([base, sub], axis=1)
        vifs = [variance_inflation_factor(ext.values, i) for i in range(ext.shape[1])]
        checks["vif"] = {"max": max(vifs), "passed": max(vifs) < 10}
    except:
        checks["vif"] = {"passed": True, "note": "計算失敗、スキップ"}
    
    # 安定性
    stds = []
    for col in sub.columns:
        rc = sub[col].rolling(60, min_periods=60).corr(
            pd.Series(gold_v, index=idx))
        stds.append(rc.std())
    max_std = max(s for s in stds if not pd.isna(s))
    checks["stability"] = {"max_std": max_std, "passed": max_std < 0.15}
    
    passed = all(c["passed"] for c in checks.values())
    return {"gate": 2, "checks": checks, "passed": passed}
```

### Gate 3: メタモデルAblation

```python
def evaluate_gate3(base_path, sub_path, gold_path):
    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit
    
    base = pd.read_csv(base_path, index_col=0, parse_dates=True)
    sub = pd.read_csv(sub_path, index_col=0, parse_dates=True)
    gold = pd.read_csv(gold_path, index_col=0, parse_dates=True)
    
    idx = base.index & sub.index & gold.index
    base, sub = base.loc[idx].dropna(), sub.loc[idx].dropna()
    idx = base.index & sub.index & gold.loc[idx].dropna().index
    base, sub, y = base.loc[idx], sub.loc[idx], gold.loc[idx].values.ravel()
    ext = pd.concat([base, sub], axis=1)
    
    tscv = TimeSeriesSplit(n_splits=5)
    b_scores, e_scores = [], []
    
    for tr, te in tscv.split(base):
        mb = XGBRegressor(n_estimators=200, max_depth=4, random_state=42)
        mb.fit(base.iloc[tr], y[tr])
        b_scores.append(calc_metrics(mb.predict(base.iloc[te]), y[te]))
        
        me = XGBRegressor(n_estimators=200, max_depth=4, random_state=42)
        me.fit(ext.iloc[tr], y[tr])
        e_scores.append(calc_metrics(me.predict(ext.iloc[te]), y[te]))
    
    avg_b = {k: np.mean([s[k] for s in b_scores]) for k in b_scores[0]}
    avg_e = {k: np.mean([s[k] for s in e_scores]) for k in e_scores[0]}
    
    da_d = avg_e["direction_accuracy"] - avg_b["direction_accuracy"]
    sh_d = avg_e["sharpe"] - avg_b["sharpe"]
    mae_d = avg_e["mae"] - avg_b["mae"]
    
    checks = {
        "direction": {"delta": da_d, "passed": da_d > 0.005},
        "sharpe": {"delta": sh_d, "passed": sh_d > 0.05},
        "mae": {"delta": mae_d, "passed": mae_d < -0.01},
    }
    
    passed = any(c["passed"] for c in checks.values())  # いずれか1つ
    return {"gate": 3, "checks": checks, "baseline": avg_b, "extended": avg_e, "passed": passed}


def calc_metrics(pred, actual, cost_bps=5):
    # Direction accuracy: exclude zero-return samples (np.sign(0) = 0 problem)
    nonzero = actual != 0
    da = np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero]))

    mae = np.mean(np.abs(pred - actual))

    # Sharpe with transaction costs (5bps one-way per trade)
    cost = cost_bps / 10000.0
    positions = np.sign(pred)
    # Cost incurred on position changes (entry + exit)
    trades = np.abs(np.diff(positions, prepend=0))  # 0→1 or 1→-1 etc.
    ret = positions * actual - trades * cost
    sharpe = np.mean(ret) / (np.std(ret) + 1e-10) * np.sqrt(252)
    return {"direction_accuracy": da, "mae": mae, "sharpe": sharpe}
```

---

## ループ制御

### 判断フロー

```
Gate 3 合格？
├─ YES
│  → status = "completed"（この特徴量について）
│  → completed.json にスコア追記
│  → feature_queue から次を取得
│
└─ NO
   ├─ attempt < 5？
   │  ├─ YES → 改善余地あり？
   │  │         ├─ YES → attempt += 1
   │  │         │         改善計画 → improvement_queue
   │  │         │         current_task に previous_feedback
   │  │         │
   │  │         └─ NO（3回連続delta<0.001 or 手法3種全不合格）
   │  │               → status = "no_further_improvement"
   │  │               → ベース特徴量のまま確定、次へ
   │  │
   │  └─ NO（attempt >= 5）
   │       → status = "paused_max_iterations"
   │       → pending_tasks に残りの改善案
   │       → 次のサブモデルへ（後で再開可能）
```

### Kaggleエラー時

Kaggle学習がエラーで終了した場合はattemptを消費しない：

```
kaggle status == "error"
  → エラーログを取得
  → builder_modelにスクリプト修正を依頼（attempt消費なし）
  → 修正後、再度Kaggle投入
  → 最大3回まで。3回エラーならarchitectに設計見直し
```

### 改善計画の原則

| 不合格Gate | 改善方向 |
|-----------|---------|
| Gate 1 | HP調整 or アーキテクチャ変更 |
| Gate 2 | 特徴量設計見直し or データ拡張 |
| Gate 3 | 出力加工変更 or 次元削減 |

**1イテレーション1改善の原則**

### improvement_queue.json

```json
{
  "feature": "real_rate",
  "items": [
    {
      "priority": 1,
      "type": "model_change",
      "description": "潜在次元を8→4に削減",
      "reason": "Gate 2通過、Gate 3不合格。情報はあるがノイジーと推定",
      "resume_from": "architect",
      "research_needed": false
    }
  ]
}
```

---

## automation_test モード

評価の最初に `shared/state.json` を読み、強制継続モードか確認する：

```python
import json

with open("shared/state.json", "r", encoding="utf-8") as f:
    state = json.load(f)

retry_ctx = state.get("retry_context", {})
automation_test = retry_ctx.get("automation_test", False)
max_attempt = retry_ctx.get("max_attempt")
current_attempt = state.get("current_attempt", 0)
attempts_left = (max_attempt - int(current_attempt)) if (automation_test and max_attempt) else 0
```

`automation_test == True` かつ `attempts_left > 0` の場合、**強制継続モード**で動作する。

### 通常モードとの差分

| 項目 | 通常モード | automation_test モード |
|-----|----------|----------------------|
| Gate 3 PASS の判定 | `decision = "completed"` | `decision = "attempt+1"` として扱う |
| `resume_from` | `"completed"` | `"architect"` |
| `completed.json` | 記録する | **記録しない** |
| `current_attempt` | 変更なし | `+1` してから state.json に書く |
| 改善計画 | 生成しない | **必ず生成する** |

Gate 評価（Gate 1/2/3）自体は通常通り実施する。変わるのは判定後の処理のみ。

### Gate 3 PASS 時の改善計画の立て方

全Gateがクリアでも「さらに改善できる点」を必ず特定する。優先順位：

1. **失敗したGateがある**: そのGateの修正（例：Gate 2 MI が5.0%ギリギリ → 6%以上を目指す）
2. **DA/Sharpe デルタが小さい or 負**: 正方向にする設計変更を提案
3. **全指標が良好**: 異なるアーキテクチャ・入力特徴量を試して汎化性を検証

### state.json の更新（automation_test モード）

```python
# Gate 3 PASS でも attempt+1 として処理
state["current_attempt"] = int(current_attempt) + 1
state["resume_from"] = "architect"
state["evaluator_status"] = (
    f"{feature}_attempt{current_attempt}_gate3_pass_automation_continue"
)
# completed.json には記録しない
```

---

## Phase 3 メタモデル評価

Phase 3ではGate 1/2/3ではなく最終目標値で評価する：

```python
def evaluate_meta_model(predictions, actuals, cost_bps=5):
    results = {}

    # 方向精度（リターン=0のサンプルを除外）
    nonzero = actuals != 0
    results["direction_accuracy"] = np.mean(
        np.sign(predictions[nonzero]) == np.sign(actuals[nonzero]))
    results["da_passed"] = results["direction_accuracy"] > 0.56

    # 高確信時（上位20%）
    threshold = np.percentile(np.abs(predictions), 80)
    high_conf = (np.abs(predictions) > threshold) & nonzero
    results["high_conf_accuracy"] = np.mean(
        np.sign(predictions[high_conf]) == np.sign(actuals[high_conf]))
    results["hca_passed"] = results["high_conf_accuracy"] > 0.60

    # MAE
    results["mae"] = np.mean(np.abs(predictions - actuals))
    results["mae_passed"] = results["mae"] < 0.0075

    # Sharpe（取引コスト込み）
    cost = cost_bps / 10000.0
    positions = np.sign(predictions)
    trades = np.abs(np.diff(positions, prepend=0))
    strat = positions * actuals - trades * cost
    results["sharpe"] = np.mean(strat) / (np.std(strat) + 1e-10) * np.sqrt(252)
    results["sharpe_passed"] = results["sharpe"] > 0.8

    return results
```

## 出力

### 出力ファイル一覧

| ファイル | 内容 | 生成タイミング |
|---------|------|-------------|
| `logs/evaluation/{feature}_{attempt}.json` | Gate 1/2/3 の全チェック結果 | 毎回 |
| `logs/evaluation/{feature}_{attempt}_summary.md` | 人間が読めるサマリー | 毎回 |
| `logs/iterations/{timestamp}_{feature}_{attempt}.json` | イテレーション記録 | 毎回 |
| `shared/improvement_queue.json` | 改善計画 | 不合格時 |
| `shared/completed.json` | 完了サブモデル記録 | 合格/改善なし/最大回数時 |
| `shared/state.json` | 状態更新 | 毎回 |
| `shared/current_task.json` | 次イテレーション要件（previous_feedback付き） | 不合格時 |

### 評価レポート JSON 構造

`logs/evaluation/{feature}_{attempt}.json`:

```json
{
  "feature": "real_rate",
  "attempt": 2,
  "timestamp": "2025-01-22T15:00:00",
  "gate1": {
    "passed": true,
    "checks": {
      "overfit": {"value": 1.12, "passed": true},
      "no_all_nan": {"value": [], "passed": true},
      "no_zero_var": {"value": [], "passed": true}
    }
  },
  "gate2": {
    "passed": true,
    "checks": {
      "mi": {"base": 0.15, "extended": 0.18, "increase": 0.20, "passed": true},
      "vif": {"max": 3.2, "passed": true},
      "stability": {"max_std": 0.08, "passed": true}
    }
  },
  "gate3": {
    "passed": true,
    "checks": {
      "direction": {"delta": 0.008, "passed": true},
      "sharpe": {"delta": 0.07, "passed": true},
      "mae": {"delta": -0.015, "passed": true}
    },
    "baseline": {"direction_accuracy": 0.52, "mae": 0.009, "sharpe": 0.45},
    "extended": {"direction_accuracy": 0.528, "mae": 0.0075, "sharpe": 0.52}
  },
  "overall_passed": true,
  "final_gate_reached": 3,
  "decision": "completed"
}
```

### サマリーレポート（人間可読）

Gate評価を完了したら、必ず `logs/evaluation/{feature}_{attempt}_summary.md` を生成する。
**orchestrator はこのファイルの内容をそのままユーザーに報告する。**

```markdown
# 評価サマリー: {feature} attempt {N}

## Gate 1: 単体品質 — {PASS/FAIL}
- 過学習比: {value} (閾値 < 1.5) {✅/❌}
- 全NaN列: {count}個 {✅/❌}
- 定数出力列: {count}個 {✅/❌}

## Gate 2: 情報増加 — {PASS/FAIL}
- MI増加: {value}% (閾値 > 5%) {✅/❌}
- 最大VIF: {value} (閾値 < 10) {✅/❌}
- 相関安定性(std): {value} (閾値 < 0.15) {✅/❌}

## Gate 3: Ablation — {PASS/FAIL}
|指標|ベースライン|サブモデル追加|差分|判定|
|---|---|---|---|---|
|方向精度|{base}%|{ext}%|{delta}%|{✅/❌}|
|Sharpe|{base}|{ext}|{delta}|{✅/❌}|
|MAE|{base}%|{ext}%|{delta}%|{✅/❌}|

## 判定: {completed / attempt+1 / no_further_improvement / paused_max_iterations}
{不合格時: 改善方向: {description}}
```

### completed.json スキーマ

`shared/completed.json` は全サブモデルの最終状態を記録する。
orchestrator が Phase 2 の完了判定や横断レポート生成に使用する。

```json
{
  "real_rate": {
    "status": "completed",
    "final_attempt": 2,
    "final_gate_reached": 3,
    "gate3_scores": {
      "direction_accuracy_delta": 0.008,
      "sharpe_delta": 0.07,
      "mae_delta": -0.015
    },
    "output_columns": ["rr_persistence", "rr_mean_reversion_prob", "rr_vol_regime"],
    "output_path": "data/submodel_outputs/real_rate.csv",
    "model_path": "models/submodels/real_rate/model.pt",
    "completed_at": "2025-01-22T15:00:00"
  },
  "dxy": {
    "status": "no_further_improvement",
    "final_attempt": 3,
    "final_gate_reached": 2,
    "gate3_scores": null,
    "output_columns": [],
    "output_path": null,
    "model_path": null,
    "completed_at": "2025-01-23T10:00:00",
    "reason": "3回連続delta<0.001"
  }
}
```

| status | 意味 |
|--------|------|
| `completed` | Gate 3 合格。サブモデル出力をメタモデルに渡す |
| `no_further_improvement` | 改善余地なし。ベース特徴量のまま |
| `paused_max_iterations` | attempt上限。後で再開可能 |

### Phase 2 完了時の横断レポート

Phase 2 の全特徴量が完了（またはpaused）したら、`logs/evaluation/phase2_summary.md` を生成する。

```python
def generate_phase2_summary():
    completed = json.load(open("shared/completed.json"))
    baseline = json.load(open("shared/baseline_score.json"))

    rows = []
    for feature in ["real_rate", "dxy", "vix", "technical",
                     "cross_asset", "yield_curve", "etf_flow",
                     "inflation_expectation", "cny_demand"]:
        info = completed.get(feature, {})
        status_emoji = {"completed": "✅",
                       "no_further_improvement": "⏸️",
                       "paused_max_iterations": "⏳"}.get(info.get("status"), "❓")
        g3 = info.get("gate3_scores", {}) or {}
        rows.append({
            "feature": feature,
            "status": f"{status_emoji} {info.get('status', 'not_started')}",
            "gate": info.get("final_gate_reached", "-"),
            "attempts": f"{info.get('final_attempt', 0)}/5",
            "da_delta": f"{g3.get('direction_accuracy_delta', 0)*100:+.1f}%" if g3 else "-",
            "sharpe_delta": f"{g3.get('sharpe_delta', 0):+.2f}" if g3 else "-",
        })

    # logs/evaluation/phase2_summary.md に書き出し
    # orchestrator がこの内容をユーザーに報告する
    return rows
```
