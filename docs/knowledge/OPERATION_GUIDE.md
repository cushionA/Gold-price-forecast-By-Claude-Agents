# 運用手順書

## 0. 初回セットアップ

### 0.1 前提条件

| 項目 | 要件 |
|------|------|
| Claude Code | Max plan ($200/月) で利用可能 |
| Node.js | v18+ (Claude Code実行に必要) |
| Python | 3.10+ |
| Git | 設定済み（SSH鍵 or Personal Access Token） |
| FRED API Key | https://fred.stlouisfed.org/docs/api/api_key.html で取得 |
| Kaggle API Token | https://www.kaggle.com/settings → API → Create New Token で取得 |

### 0.2 リポジトリ作成

```bash
# GitHub上でリポジトリ作成後
git clone git@github.com:<username>/gold-prediction-agent.git
cd gold-prediction-agent
```

### 0.3 ファイル配置

```bash
# エージェント定義を配置
mkdir -p .claude/agents
cp entrance.md researcher.md architect.md builder_data.md \
   datachecker.md builder_model.md evaluator.md orchestrator.md \
   .claude/agents/

# CLAUDE.md をルートに配置
cp CLAUDE.md .

# 設定・ユーティリティ
mkdir -p config src
cp settings.yaml config/
cp kaggle_runner.py src/

# ディレクトリ構造を作成
mkdir -p shared data/{raw,processed,multi_country,submodel_outputs} \
         models/{submodels,meta} docs/{research,design} \
         logs/{datacheck,evaluation,iterations,training} \
         notebooks

# 空ディレクトリ維持用
find data models logs notebooks -type d -empty -exec touch {}/.gitkeep \;

# .gitignore
cp gitignore .gitignore
```

### 0.4 環境変数

```bash
# .env ファイル（.gitignoreに含まれる）
cat > .env << 'EOF'
FRED_API_KEY=your_fred_api_key_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_TOKEN=your_kaggle_api_token_here
EOF
```

※ `KAGGLE_API_TOKEN` は Kaggle Settings → API → "Create New Token" で取得できる。
  `~/.kaggle/kaggle.json` は不要。python-dotenv が .env を自動読み込みし、
  kaggle CLI もこの環境変数を認識する。

### 0.5 Kaggle Secrets（Kaggle Notebook内の認証）

train.py は Kaggle 上で実行されるため、ローカルの .env は使えない。
Kaggle Secrets に FRED_API_KEY を登録する必要がある。

1. https://www.kaggle.com/settings にアクセス
2. 画面下部「Secrets」セクションで「Add Secret」
3. Label: FRED_API_KEY / Value: 自分のFREDキー
4. 「Save」

※ これを忘れると train.py が即エラーで終了する。
   Claude Code は Phase 0 でこの設定の有無を確認し、
   未設定なら停止してユーザーに通知する。

### 0.6 Python環境

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch pandas numpy scikit-learn xgboost optuna \
    yfinance fredapi matplotlib scipy statsmodels kaggle python-dotenv
```

### 0.7 初回コミット

```bash
git add -A
git commit -m "init: project structure and agent definitions"
git push origin main
```

### 0.8 state.json 初期化

```bash
cat > shared/state.json << 'EOF'
{
  "status": "not_started",
  "phase": null,
  "current_feature": null,
  "current_attempt": null,
  "resume_from": null,
  "feature_queue": ["real_rate", "dxy", "vix", "technical", "cross_asset", "yield_curve", "etf_flow"],
  "kaggle_kernel": null,
  "submitted_at": null,
  "last_updated": null,
  "error_context": null,
  "user_action_required": null
}
EOF

git add shared/state.json
git commit -m "init: state.json"
git push origin main
```

---

## 1. プロジェクト開始

### 1.1 Claude Code を起動

```bash
cd gold-prediction-agent
claude
```

### 1.2 開始指示

```
プロジェクトを開始して
```

Claude Code は CLAUDE.md を読み、`shared/state.json` の `status: "not_started"` を検出し、Phase 0 から自律的に開始する。

### 1.3 Phase 0 で起きること

Claude Code が以下を自動実行する（所要時間: 10-15分）:

1. ライブラリの確認・インストール
2. FRED_API_KEY, KAGGLE_USERNAME の確認
3. 共通コードの生成 (`src/` 配下)
4. ベースデータの取得 (金価格, 7主要特徴量)
5. ターゲット変数の計算
6. git commit & push

**ユーザーの操作**: 特になし。APIキーの入力を求められたら応答する。

---

## 2. Phase 1: ベースライン構築

### 2.1 自動実行

Phase 0 完了後、自動的に Phase 1 に遷移する。

1. 7主要特徴量のDataFrame整備
2. XGBoostベースラインの学習（ローカル実行、数分で完了）
3. スコア記録: `shared/baseline_score.json`
4. スキーマ凍結: `shared/schema_freeze.json`

**ユーザーの操作**: 特になし。

### 2.2 ベースライン結果の確認

```
ベースラインの結果を見せて
```

方向精度・Sharpe・MAEが表示される。この数値が今後の改善のベンチマーク。

---

## 3. Phase 1.5: スモークテスト

### 3.1 自動実行

`real_rate` 1つで全パイプラインを簡易版（Optuna 5 trials）で通す。

1. entrance → researcher → architect → builder_data → datachecker → builder_model
2. Kaggle Notebook投入
3. ここで **PCを閉じてよい**

### 3.2 Kaggle投入後の画面表示

```
🚀 Kaggle投入完了
   Kernel: username/gold-real_rate-smoke
   推定実行時間: 5-10分
   PCを閉じてOKです。再開時は「続きから再開して」と伝えてください。
```

### 3.3 再開

PCを開き直してClaude Codeを起動:

```bash
cd gold-prediction-agent
claude
```

```
続きから再開して
```

Claude Code は `state.json` → `status: "waiting_training"` を検出し、Kaggle結果を取得 → evaluator で評価。

---

## 4. Phase 2: サブモデル構築ループ（メインフェーズ）

### 4.1 自動実行

9つの特徴量を順に処理する。1特徴量あたりの流れ:

```
entrance → researcher → architect → builder_data → datachecker
→ builder_model → Kaggle投入 → [PC閉じてOK]
→ 再開 → Kaggle結果取得 → evaluator → 合格 or ループ
```

### 4.2 日常の操作パターン

#### パターンA: 1日1特徴量

```
朝: "続きから再開して"
  → Kaggle結果取得 → 評価 → 次の特徴量の設計 → Kaggle投入
夜: PCを閉じる
翌朝: "続きから再開して"
  → 繰り返し
```

#### パターンB: 半日集中

```
午前: "続きから再開して"
  → 結果取得 → 評価 → 次の特徴量 → 投入
  → 30分待つ or 一旦閉じる
  → "続きから再開して"
  → 結果取得 → 評価 → 次の特徴量 → 投入
  ...
```

#### パターンC: 特定の特徴量だけ

```
"vixのサブモデルだけ作って"
```

### 4.3 進捗確認

```
現在の進捗を見せて
```

`shared/completed.json` と `shared/state.json` から進捗を報告してくれる。

### 4.4 改善ループ中の判断

evaluator が Gate 不合格 → 改善計画を提示 → 自動的に次の attempt に入る。
ユーザーが介入したい場合:

```
改善計画を見せて。手動で方向を変えたい。
```

```
real_rateは一旦スキップして次に進んで
```

---

## 5. Phase 3: メタモデル構築

### 5.1 開始

Phase 2 の全特徴量が完了（または paused）した後:

```
メタモデルの構築に進んで
```

### 5.2 自動実行

1. architect: 全サブモデル出力の統合設計
2. builder_model: メタモデル学習スクリプト生成
3. Kaggle投入 → [PC閉じてOK]
4. evaluator: 最終目標値で評価

### 5.3 最終結果

```
最終結果を見せて
```

| 指標 | 目標 | 結果 |
|------|------|------|
| 方向精度 | > 56% | ?% |
| 高確信時精度 | > 60% | ?% |
| MAE | < 0.75% | ?% |
| Sharpe | > 0.8 | ? |

---

## 6. トラブルシューティング

### 6.1 「続きから再開して」が効かない

```bash
# state.json を手動確認
cat shared/state.json
```

`status` が壊れている場合は手動修正:

```bash
# 例: evaluatorから再開させたい
python3 -c "
import json
state = json.load(open('shared/state.json'))
state['status'] = 'in_progress'
state['resume_from'] = 'evaluator'
json.dump(state, open('shared/state.json', 'w'), indent=2)
"
```

### 6.2 Kaggleがエラーで止まった

```
学習結果を確認して
```

Claude Code がエラーログを取得し、builder_model にスクリプト修正を依頼する。
3回失敗したら architect に設計見直しが入る。

### 6.3 FRED APIが応答しない

```
FREDが落ちてるようです。後で再開します。
```

state.json に error_context が記録される。API復旧後:

```
続きから再開して
```

### 6.4 特定のattemptからやり直したい

```
real_rateをattempt 3から再開して
```

### 6.5 Claude Code のコンテキストが溢れた

長時間セッションでコンテキストが一杯になることがある:

```bash
# 一度終了して再起動
exit
claude
```

```
続きから再開して
```

state.json + git のおかげで、再起動しても状態は失われない。

---

## 7. コマンド一覧

### よく使う指示

| 指示 | 動作 |
|------|------|
| `プロジェクトを開始して` | 新規開始 (Phase 0〜) |
| `続きから再開して` | state.json に従い自動再開 |
| `現在の進捗を見せて` | completed.json + state.json を表示 |
| `ベースラインの結果を見せて` | Phase 1 のスコアを表示 |
| `学習結果を確認して` | Kaggle結果取得 → 評価 |
| `改善計画を見せて` | improvement_queue.json を表示 |
| `最終結果を見せて` | メタモデルの評価結果を表示 |

### 特定操作

| 指示 | 動作 |
|------|------|
| `{feature}のサブモデルだけ作って` | 指定特徴量のみ実行 |
| `{feature}をattempt {N}から再開` | 指定位置から再開 |
| `{feature}はスキップして次に進んで` | 現在の特徴量を paused にして次へ |
| `メタモデルの構築に進んで` | Phase 3 に直接移行 |

---

## 8. ファイル配置の確認

正常に動いている場合の共有ワークスペース:

```bash
# 状態確認
cat shared/state.json          # 現在の進行状態
cat shared/completed.json      # 完了サブモデル一覧
cat shared/current_task.json   # 今のイテレーション要件
cat shared/schema_freeze.json  # Phase 1で凍結したベースラインスキーマ

# ログ確認
ls logs/datacheck/             # datachecker結果
ls logs/evaluation/            # evaluator結果
ls logs/training/              # Kaggle学習メトリクス
ls logs/iterations/            # イテレーション履歴

# 成果物確認
ls data/submodel_outputs/      # サブモデル出力CSV
ls models/submodels/           # 学習済みモデル
ls docs/design/                # 設計書
ls docs/research/              # リサーチレポート
```

---

## 9. 想定スケジュール

| 日程 | 作業 | 所要時間 |
|------|------|---------|
| Day 1 | セットアップ + Phase 0/1/1.5 | 1-2時間 |
| Day 2 | Phase 2: real_rate, dxy, inflation_expectation | 朝投入→夜回収 × 3 |
| Day 3 | Phase 2: vix, technical, cny_demand | 同上 |
| Day 4 | Phase 2: cross_asset, yield_curve, etf_flow | 同上 |
| Day 5 | Phase 2: 改善ループ（必要分） | 適宜 |
| Day 6 | Phase 3: メタモデル構築 | 2-3時間 |
| Day 7 | 最終評価・チューニング | 2-3時間 |

※ 各日30分〜1時間のPC操作。残りはKaggle学習待ち。
