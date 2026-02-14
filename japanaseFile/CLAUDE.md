# 金相場予測モデル - 自律エージェントシステム v5

## 起動モード

オーケストレーター（あなた）は起動時にまず `git pull` し、`shared/state.json` を読む。

### 自動判定

```
1. git pull で最新状態を取得
2. shared/state.json の status を確認
3. ユーザー指示があればそちらを優先

status == "not_started"            → Phase 0 から開始
status == "in_progress"            → resume_from のエージェントから再開
status == "waiting_training"       → Kaggle学習結果を取得して評価へ
status == "waiting_user_input"     → ユーザーに対応を依頼中。指示を待つ
status == "paused_max_iterations"  → pending_tasks 確認、resume_from から再開
status == "completed"              → 最終レポート出力
```

### ユーザー指示の例

```
「プロジェクトを開始して」        → 新規開始
「続きから再開して」              → state.json に従い自動再開
「real_rateをattempt 3から再開」  → state.json 上書きして指定位置から
「vixのサブモデルだけ作って」      → vix のみ実行
「メタモデルの構築に進んで」      → Phase 3 に直接移行
「学習結果を確認して」            → Kaggle結果取得 → 評価
```

---

## ミッション

金（Gold）の翌日リターン（%）を予測する回帰モデルを構築する。
7つの主要特徴量 × サブモデルで情報量を増やし、メタモデルで統合する。

## 核心コンセプト

- 主要特徴量をメタモデルが直接処理する
- サブモデルは文脈・状態・性質を補足する
- **サブモデルは金相場を予測しない**
- 多国籍データを積極的に活用し、学習サンプルを最大化する

---

## 主要特徴量（7つ）

| # | 特徴量 | ソース | 重要性 |
|---|--------|--------|--------|
| 1 | 実質金利（10Y TIPS） | FRED: DFII10 | 最強の負相関 |
| 2 | ドル指数（DXY） | Yahoo: DX-Y.NYB | ドル建て逆相関 |
| 3 | VIX | FRED: VIXCLS | リスクオフ指標 |
| 4 | 金テクニカル | Yahoo: GC=F, GLD | モメンタム・回帰 |
| 5 | クロスアセット | Yahoo: SI=F, HG=F, ^GSPC | 相対魅力度 |
| 6 | イールドカーブ | FRED: DGS10, DGS2 | 政策期待 |
| 7 | ETFフロー | Yahoo: GLD | 投資家需要 |

---

## 実行アーキテクチャ: Claude Code + Kaggle分離

### なぜ分離するか

```
Claude Code = PCのプロセス。PC閉じたら止まる。
Kaggle      = クラウド。PC閉じても学習が走り続ける。
Git         = 両者の接続点。
```

### 役割分担

| 作業 | 実行場所 | PC必要？ |
|------|---------|---------|
| 要件定義（entrance） | Claude Code | はい |
| リサーチ（researcher） | Claude Code | はい |
| 設計（architect） | Claude Code | はい |
| データ取得（builder_data） | Claude Code | はい |
| データチェック（datachecker） | Claude Code | はい |
| **学習スクリプト生成**（builder_model） | Claude Code | はい |
| **学習実行** | **Kaggle** | **いいえ** |
| 評価（evaluator） | Claude Code | はい |
| 改善計画 | Claude Code | はい |

### 典型的なワークフロー

```
1. [PC開] Claude Code: 設計→データ取得→チェック→学習スクリプト生成
2. [PC開] Claude Code: Kaggle APIで学習Notebookを投入
3. [PC閉じてOK] Kaggle: クラウドで学習実行（数分〜30分）
4. [PC開] Claude Code: "続きから再開して"
   → git pull → Kaggle結果取得 → evaluatorで評価 → 次のイテレーション
```

---

## Kaggle統合

### Kaggle Notebook構造

builder_modelが以下のNotebookを自動生成する：

```
notebooks/
  └── {feature}_{attempt}/
      ├── kernel-metadata.json    ← Kaggle API設定
      └── train.py                ← 学習スクリプト（自己完結型）
```

### kernel-metadata.json

```json
{
  "id": "{KAGGLE_USERNAME}/gold-{feature}-{attempt}",
  "title": "Gold {feature} SubModel Attempt {attempt}",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

### train.py の設計原則

学習スクリプトは**自己完結型**にする。Claude CodeもKaggleも依存しない：

```python
"""
Gold Prediction SubModel Training - {feature} Attempt {attempt}
自己完結型: データ取得→前処理→学習→評価→結果保存 をすべてこのファイル内で実行
"""

# === 1. ライブラリ ===
import torch, torch.nn as nn, pandas as pd, numpy as np
import json, os
from datetime import datetime

# === 2. データ取得（API直接呼び出し） ===
# builder_dataが検証済みのデータ取得コードをそのまま埋め込む
def fetch_data():
    ...
    return train_data, val_data

# === 3. モデル定義 ===
# architectの設計をそのまま実装
class SubModel(nn.Module):
    ...

# === 4. 学習ループ ===
def train(model, train_data, val_data, config):
    ...
    return model, metrics

# === 5. Optuna HPO ===
def run_hpo(train_data, val_data):
    ...
    return best_params, best_value

# === 6. メイン実行 ===
if __name__ == "__main__":
    train_data, val_data = fetch_data()
    best_params, _ = run_hpo(train_data, val_data)
    
    model = SubModel(**best_params)
    model, metrics = train(model, train_data, val_data, best_params)
    
    # サブモデル出力生成
    output = model.transform(full_data)
    
    # === 7. 結果保存（Kaggle output） ===
    output.to_csv("submodel_output.csv")
    torch.save(model.state_dict(), "model.pt")
    
    with open("training_result.json", "w") as f:
        json.dump({
            "feature": "{feature}",
            "attempt": {attempt},
            "timestamp": datetime.now().isoformat(),
            "best_params": best_params,
            "metrics": metrics,
            "output_shape": list(output.shape),
            "output_columns": list(output.columns),
        }, f, indent=2)
    
    print("Training complete!")
```

### Kaggle APIコマンド

builder_modelが学習スクリプト生成後、オーケストレーターが以下を実行：

```bash
# 1. 学習Notebookを投入
kaggle kernels push -p notebooks/{feature}_{attempt}/

# 2. 状態確認
kaggle kernels status {KAGGLE_USERNAME}/gold-{feature}-{attempt}
# → "running" / "complete" / "error"

# 3. 結果取得（complete後）
kaggle kernels output {KAGGLE_USERNAME}/gold-{feature}-{attempt} \
  -p data/submodel_outputs/{feature}/

# 4. 取得したファイルを所定パスに配置
mv data/submodel_outputs/{feature}/submodel_output.csv \
   data/submodel_outputs/{feature}.csv
mv data/submodel_outputs/{feature}/model.pt \
   models/submodels/{feature}/model.pt
cp data/submodel_outputs/{feature}/training_result.json \
   logs/training/{feature}_{attempt}.json
```

### 学習待ちの状態管理

学習を投入したらstate.jsonを更新してPCを閉じられるようにする：

```json
{
  "status": "waiting_training",
  "resume_from": "evaluator",
  "kaggle_kernel": "{KAGGLE_USERNAME}/gold-{feature}-{attempt}",
  "submitted_at": "2025-01-22T10:00:00",
  "current_feature": "real_rate",
  "current_attempt": 1
}
```

再開時のオーケストレーター動作：

```
1. git pull
2. state.json を読む → status == "waiting_training"
3. kaggle kernels status で学習完了を確認
   → "running" → ユーザーに「まだ学習中です」と通知
   → "error" → エラーログを取得、builder_modelに差し戻し
   → "complete" → 結果を取得し evaluator に渡す
```

---

## エージェント構成

```
オーケストレーター（Sonnet）
  │  ※ 各エージェント完了後に git commit & push
  │  ※ 学習投入/結果取得はオーケストレーターが直接実行
  │
  ├─ entrance（Opus）         初回要件定義
  ├─ researcher（Haiku）      調査（ファクトチェックされる前提）
  ├─ architect（Opus）        ファクトチェック → 設計書 → HP探索空間
  ├─ builder_data（Sonnet）   データ取得・前処理
  ├─ datachecker（Haiku）     定型7ステップチェック
  ├─ builder_model（Sonnet）  PyTorch学習スクリプト生成（Kaggle用）
  └─ evaluator（Opus）        Gate 1/2/3 → ループ制御 → 改善計画
```

### トークン消費

| フェーズ | トークン消費 |
|---------|------------|
| 設計・コード生成（Claude Code） | あり |
| Kaggle学習投入（API呼び出し） | ごくわずか |
| **Kaggle学習実行中** | **なし（PC閉じてOK）** |
| 結果取得・評価（Claude Code） | あり |

---

## ワークフロー

### Phase 0: 環境構築（初回のみ）

```
1. git init & リモート設定（ユーザーが事前に用意）
2. ライブラリインストール:
   pip install torch pandas numpy scikit-learn xgboost optuna \
       yfinance fredapi matplotlib scipy statsmodels kaggle python-dotenv
3. 認証情報の確認（以下すべて揃っていなければ waiting_user_input で停止）:
   a. ローカル環境変数（.env → python-dotenv で自動読み込み）:
      - FRED_API_KEY → .env に設定済みか
      - KAGGLE_USERNAME → .env に設定済みか
      - KAGGLE_API_TOKEN → .env に設定済みか
   b. Kaggle CLI認証:
      - KAGGLE_API_TOKEN が設定済みで kaggle kernels list が成功するか
   c. Kaggle Secrets（Kaggle Notebook内でAPI呼び出しに必要）:
      - https://www.kaggle.com/settings → Secrets → FRED_API_KEY を登録済みか
      - ※ ユーザーにブラウザで設定してもらう必要がある
4. 共通コードの作成:
   - src/submodel_base.py
   - src/data_fetcher.py
   - src/evaluation.py
   - src/utils.py
   - src/kaggle_runner.py  ← Kaggle投入・取得の共通関数
5. ベースデータ取得:
   - 金価格 GC=F → data/raw/gold.csv
   - 7主要特徴量の生データ → data/raw/
   - ターゲット変数（翌日リターン%）→ data/processed/target.csv
6. git commit & push "phase0: environment setup and base data"
```

### Phase 1: ベースライン構築

```
1. 7主要特徴量の直接入力データ整備 → data/processed/base_features.csv
2. XGBoostベースライン学習（ローカル実行、小さいので即完了）
3. ベースラインスコアを shared/state.json に記録
4. git commit & push "phase1: baseline (DA=xx%, Sharpe=x.xx)"
```

### Phase 1.5: スモークテスト

```
1. real_rateで全パイプラインを1回通す（簡易版）
   - entrance → researcher → architect → builder_data → datachecker
   - builder_model → Kaggle Notebook生成（Optuna 5 trials）
   - Kaggle投入 → 結果取得 → evaluator（Gate 1のみ）
2. Kaggle連携がエラーなく動くことを確認
3. git commit & push "smoke_test: pipeline with kaggle verified"
```

### Phase 2: サブモデル構築ループ

```
[PC開] entrance/evaluator → researcher → architect → 
       builder_data → datachecker → builder_model（スクリプト生成）
       → Kaggle投入 → git push
[PC閉じてOK] Kaggle学習実行中
[PC開] "続きから再開" → Kaggle結果取得 → evaluator → (ループ or 次へ)
```

### Phase 3: メタモデル構築

```
1. architect: 全サブモデル出力の入力形式を分析 → アーキテクチャ選択
2. builder_model: Kaggle Notebook生成
3. Kaggle学習実行
4. evaluator: 最終目標値で評価
5. 改善ループ
```

---

## Phase 2 パイプライン詳細

```
┌──────────────────────────────────────────────┐
│ entrance（初回）/ evaluator（2回目以降）       │
│  → current_task.json に要件                  │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ researcher（Haiku）                           │
│  → docs/research/ にレポート                  │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ architect（Opus）                             │
│  → ファクトチェック                           │
│  → 不合格 → researcher再調査                 │
│  → docs/design/ に設計書                     │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ builder_data（Sonnet）                        │
│  → data/ にデータ保存                        │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ datachecker（Haiku）7ステップ定型チェック      │
│  → REJECT → builder_data差し戻し             │
│    （attempt消費なし、最大3回）               │
│  → 3回REJECT → architect差し戻し             │
│  → PASS → git commit, 次へ                   │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ builder_model（Sonnet）                       │
│  → 自己完結型 train.py を生成                │
│  → kernel-metadata.json を生成               │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ オーケストレーター: Kaggle投入                │
│  → kaggle kernels push                       │
│  → state.json を "waiting_training" に更新   │
│  → git push                                  │
│  → ★ ユーザーはPCを閉じてOK ★               │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ （PC再開時）オーケストレーター: 結果取得       │
│  → git pull                                  │
│  → kaggle kernels status で確認              │
│  → kaggle kernels output で結果取得          │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ evaluator（Opus）                             │
│  → Gate 1 → 2 → 3                           │
│  → 合格 → completed追記、次のサブモデルへ     │
│  → 不合格 → attempt+1、改善計画、ループ先頭   │
│  → 改善余地なし → 次のサブモデルへ            │
│  → git commit & push                         │
└──────────────────────────────────────────────┘
```

---

## ループ制御

### attempt消費ルール

```
消費する（+1）: evaluatorがGate評価完了→不合格
消費しない:     datachecker差し戻し / architect差し戻し / researcher再調査
```

### 終了条件

1. Gate 3合格
2. 改善余地なし（3回連続delta<0.001 or evaluator宣言）
3. attempt >= 5（paused、後で再開可能）

---

## 技術スタック

| 用途 | ライブラリ |
|------|-----------|
| サブモデル・メタモデル | PyTorch |
| HP最適化 | Optuna |
| データ処理 | pandas, numpy |
| Gate 3ベースライン比較 | XGBoost |
| 評価 | scikit-learn |
| クラウド学習 | Kaggle API |

### HP管理の分担

| 役割 | 担当 |
|------|------|
| 探索空間の設計 | architect |
| 探索の実行 | builder_model（Optuna in train.py） |
| 結果の評価 | evaluator |
| 範囲修正 | evaluator → architect |

### メタモデル選択基準

architectが入力形式を分析して最適モデルを選ぶ：

| 入力形式 | 推奨 |
|---------|------|
| 連続値のみ <30次元 | MLP |
| 連続 + カテゴリカル | FT-Transformer / TabNet |
| 時系列パターン重要 | GRU + MLP |
| >50次元 サンプル少 | MLP + 強Dropout / 次元削減 |

---

## 多国籍データ方針

大規模データを積極的に活用する。多国籍データによる学習サンプル最大化が性能向上の鍵。

| 特徴量 | 拡張データ | 期待サンプル数 |
|--------|-----------|--------------|
| 実質金利 | G10各国 | 〜75,000 |
| DXY | 12通貨ペア | 〜60,000 |
| VIX | 各国ボラ指数 | 〜25,000 |
| クロスアセット | 複数商品相関 | 〜30,000 |

ルール:
- G10 + 主要先進国に限定
- 水準（level）使用しない、変化（change）をそのまま使用
- ボラティリティは自国長期平均で正規化

---

## 評価フレームワーク

### Gate 1: 単体品質
過学習比<1.5、全NaN/定数出力なし、リーク兆候なし

### Gate 2: 情報増加
MI増加>5%、VIF<10、ローリング相関std<0.15

### Gate 3: Ablation（いずれか1つ）
方向精度+0.5%、Sharpe+0.05、MAE-0.01%

### メタモデル最終目標

| 指標 | 目標 |
|------|------|
| 方向精度 | > 56% |
| 高確信時方向精度 | > 60% |
| MAE | < 0.75% |
| Sharpe比 | > 0.8 |

---

## Git永続化

### コミット規則

各エージェント完了時にオーケストレーターが `git add -A && git commit && git push`:

```
entrance完了     → "entrance: {feature} attempt {N}"
researcher完了   → "research: {feature} attempt {N}"
architect完了    → "design: {feature} attempt {N}"
builder_data完了 → "data: {feature} attempt {N}"
datachecker完了  → "datacheck: {feature} attempt {N} - {PASS/REJECT}"
builder_model完了 → "model: {feature} attempt {N} - notebook generated"
kaggle投入       → "kaggle: {feature} attempt {N} - submitted"
kaggle結果取得   → "kaggle: {feature} attempt {N} - results fetched"
evaluator完了    → "eval: {feature} attempt {N} - gate{N} {pass/fail}"
```

---

## 共有ワークスペース

```
shared/
  ├── state.json              進行状態・再開ポイント・Kaggle状態
  ├── current_task.json       今のイテレーション要件
  ├── improvement_queue.json  改善タスクキュー
  └── completed.json          完了サブモデル記録
```

---

## プロジェクト構成

```
gold-prediction-agent/
├── CLAUDE.md
├── .gitignore
├── .claude/agents/           7エージェント
├── shared/                   状態管理
├── src/                      共通コード
│   ├── submodel_base.py
│   ├── data_fetcher.py
│   ├── evaluation.py
│   ├── utils.py
│   └── kaggle_runner.py      ← Kaggle API操作
├── notebooks/                ← Kaggle Notebook（自動生成）
│   └── {feature}_{attempt}/
│       ├── kernel-metadata.json
│       └── train.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── multi_country/
│   └── submodel_outputs/
├── models/
│   ├── submodels/{feature}/
│   └── meta/
├── docs/
│   ├── research/
│   └── design/
├── logs/
│   ├── datacheck/
│   ├── evaluation/
│   ├── iterations/
│   └── training/
└── config/settings.yaml
```

---

## 禁止事項

1. ランダム分割禁止（時系列分割のみ）
2. 未来情報リーク禁止
3. 5日以上遅延データ禁止
4. サブモデルが金相場を直接予測禁止
5. APIキー・認証情報のハードコード禁止（train.py含む）
6. ユーザーの明示的な承認なしに有料サービス・新規APIキーが必要な処理を実行しない

---

## API

| API | 用途 | 認証 | ローカル | Kaggle Notebook内 |
|-----|------|------|---------|-------------------|
| yfinance | 価格データ | 不要 | — | — |
| fredapi | 経済指標 | FRED_API_KEY | .env | Kaggle Secrets |
| kaggle CLI | 学習投入・結果取得 | KAGGLE_API_TOKEN | .env | — (CLIはローカルのみ) |
| CNN Fear & Greed | リスク指標 | 不要 | — | — |
| CBOE Put/Call | リスク指標 | 不要 | — | — |
| GPR Index | 地政学指標 | 不要 | — | — |

### 認証情報の管理原則

- ローカル: .env ファイル（gitignore対象）→ python-dotenv で自動読み込み
- Kaggle CLI: KAGGLE_API_TOKEN 環境変数（.env経由）
- Kaggle Notebook: Kaggle Secrets（ブラウザで設定）
- train.py 内: os.environ['FRED_API_KEY']（KeyErrorで即失敗させる）
- **ハードコード・フォールバック値は絶対に使わない**
