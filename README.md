# Gold Prediction Agent

Claude Code で金（Gold）の翌日リターンを予測するMLモデルを**自律的に**構築するプロジェクト。

7つのドメインモデル × 改善ループ × メタモデル統合を、人間は「続きから再開して」と言うだけで進行させる。

---

## コンセプト

```
人間: 「プロジェクトを開始して」
Claude Code: 要件定義 → 調査 → 設計 → データ取得 → 検証 → 学習 → 評価 → 改善
人間: (PCを閉じて寝る)
人間: 「続きから再開して」
Claude Code: Kaggle結果を回収 → 評価 → 次の特徴量へ → ...
```

**サブモデルは金相場を予測しない。** 各ドメインの「文脈・状態・性質」を抽出し、メタモデルだけが金リターンを予測する。

## アーキテクチャ

```
メタモデル（GRU + Attention）← 62次元入力
  │
  ├── 直接入力（12次元）         実質金利, DXY, VIX, GLD Flow, ...
  │
  └── ドメインモデル出力（50次元）
        ├── 金利短期特性     Autoencoder        8次元
        ├── 金利サイクル     専門家ラベル蒸留    4次元
        ├── 通貨トレンド     Autoencoder        8次元
        ├── 通貨マクロ       計算               3次元
        ├── リスク短期       専門家ラベル蒸留    4次元
        ├── リスクレジーム   外部指標            4次元
        ├── 需給            教師あり            4次元
        ├── テクニカル       GRU埋め込み         8次元
        ├── クロスアセット   計算               5次元
        ├── 地政学          計算               2次元
        └── 季節性          ルールベース        4次元
```

## エージェント構成

7つの専門エージェントが `.claude/agents/` に定義されている。Claude Code がオーケストレーターとして各エージェントを順に呼び出す。

| エージェント | モデル | 役割 |
|---|---|---|
| **orchestrator** | Sonnet | 進行管理、git操作、Kaggle連携 |
| **entrance** | Opus | 要件定義、調査事項の具体化 |
| **researcher** | Sonnet | 手法・データソース調査 |
| **architect** | Opus | ファクトチェック、設計書作成 |
| **builder_data** | Sonnet | データ取得・前処理 |
| **datachecker** | Sonnet | 品質検証（7ステップ定型チェック） |
| **builder_model** | Sonnet | PyTorch学習スクリプト生成 |
| **evaluator** | Opus | 3段階Gate評価、改善計画 |

## ワークフロー

```
Phase 0  環境構築
Phase 1  XGBoostベースライン
Phase 1.5 スモークテスト（real_rateでパイプライン1周）
Phase 2  サブモデル構築ループ（7特徴量 × 最大5回改善）
Phase 3  メタモデル統合・最終評価
```

### Phase 2 パイプライン

```
entrance → researcher → architect → builder_data → datachecker
  → builder_model → Kaggle投入 → ☕ PC閉じてOK ☕
  → 再開 → Kaggle結果取得 → evaluator
  → 合格 → 次の特徴量へ
  → 不合格 → 改善計画 → attempt+1 → ループ先頭
```

### 評価ゲート

| Gate | 基準 | 目的 |
|------|------|------|
| Gate 1 | 過学習比 < 1.5、NaN/定数なし | サブモデル単体の品質 |
| Gate 2 | MI増加 > 5%、VIF < 10 | 情報量の増加 |
| Gate 3 | 方向精度+0.5% or Sharpe+0.05 | メタモデルへの貢献 |

### 最終目標

| 指標 | 目標値 |
|------|--------|
| 方向精度 | > 56% |
| 高確信時精度 | > 60% |
| MAE | < 0.75% |
| Sharpe比 | > 0.8 |

## セットアップ

### 必要なもの

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Max plan)
- Python 3.10+
- [FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html)
- [Kaggle API](https://www.kaggle.com/docs/api) (`~/.kaggle/kaggle.json`)

### インストール

```bash
git clone git@github.com:<username>/gold-prediction-agent.git
cd gold-prediction-agent

python -m venv .venv
source .venv/bin/activate
pip install torch pandas numpy scikit-learn xgboost optuna \
    yfinance fredapi matplotlib scipy statsmodels kaggle

# 環境変数
cp .env.example .env
# .env を編集: FRED_API_KEY, KAGGLE_USERNAME を設定
```

### 起動

```bash
claude
```

```
プロジェクトを開始して
```

詳細な操作手順は [OPERATION_GUIDE.md](OPERATION_GUIDE.md) を参照。

## ディレクトリ構成

```
gold-prediction-agent/
├── CLAUDE.md                     # オーケストレーター指示書
├── OPERATION_GUIDE.md            # 運用手順書
├── .claude/agents/               # 8エージェント定義
│   ├── orchestrator.md
│   ├── entrance.md
│   ├── researcher.md
│   ├── architect.md
│   ├── builder_data.md
│   ├── datachecker.md
│   ├── builder_model.md
│   └── evaluator.md
│
├── config/
│   └── settings.yaml             # 全体設定
│
├── shared/                       # 状態管理（gitで共有）
│   ├── state.json                # 進行状態
│   ├── current_task.json         # 現在のタスク要件
│   ├── completed.json            # 完了記録
│   ├── improvement_queue.json    # 改善計画
│   └── schema_freeze.json        # ベースラインスキーマ
│
├── src/                          # 共通ユーティリティ
│   └── kaggle_runner.py          # Kaggle API操作
│
├── data/
│   ├── raw/                      # 生データ（gitignore）
│   ├── processed/                # 前処理済み
│   ├── multi_country/            # 多国籍データ
│   └── submodel_outputs/         # サブモデル出力
│
├── models/
│   ├── submodels/{feature}/      # 学習済みモデル
│   └── meta/                     # メタモデル
│
├── notebooks/                    # Kaggle Notebook（自動生成）
│   └── {feature}_{attempt}/
│       ├── kernel-metadata.json
│       └── train.py
│
├── docs/
│   ├── research/                 # リサーチレポート
│   └── design/                   # 設計書
│
└── logs/
    ├── datacheck/                # datachecker結果
    ├── evaluation/               # evaluator結果
    ├── iterations/               # イテレーション履歴
    └── training/                 # Kaggle学習メトリクス
```

## 設計原則

- **サブモデルは金相場を予測しない** — 文脈情報のみ抽出
- **5日遅延ルール** — 推論時にデータ遅延で破綻しないよう、5日以上遅れるデータは不採用
- **時系列分割のみ** — ランダム分割禁止、未来情報リーク防止
- **1イテレーション1改善** — 変更を小さく保ち、効果を測定可能に
- **attempt消費の厳格管理** — datachecker差し戻しやKaggleエラーはattemptを消費しない

## 制約

| 項目 | 制約 |
|------|------|
| Claude Code | PC起動中のみ動作。学習はKaggleに委譲してPC閉じてOK |
| Kaggle | CPU 12時間 / GPU 30時間(週)。通常のサブモデルはCPUで十分 |
| FRED API | 120リクエスト/分 |
| Max plan | Opus使用量に実質上限あり。1サイクルは問題なし |

## コスト

1回のメタモデル作成（7サブモデル × 平均2.5 attempt + メタモデル）:

| 項目 | 推定 |
|------|------|
| トークン消費 | 約2.7M tokens |
| 所要時間（PC操作） | 合計3-5時間（数日に分散） |
| Kaggle使用時間 | 合計3-5時間 |
| 金銭コスト | Max plan $200/月 に含まれる |

## ライセンス

Private repository — 個人利用。

## 関連ドキュメント

| ドキュメント | 内容 |
|---|---|
| [OPERATION_GUIDE.md](OPERATION_GUIDE.md) | 日常の操作手順 |
| [CLAUDE.md](CLAUDE.md) | オーケストレーター指示書（全体設計） |
| [config/settings.yaml](config/settings.yaml) | 全体設定 |
