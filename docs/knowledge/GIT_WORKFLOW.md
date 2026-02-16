# Git Workflow - Gold Price Prediction Project

## ブランチ戦略

### ブランチ構成

```
main (protected, production-ready milestones only)
  ├─ develop (daily development, all agent cycles)
  └─ feature/* (optional, per-submodel branches)
```

### ブランチの役割

#### `main` - プロダクションブランチ
**目的:** 完成したマイルストーンのみを記録
**マージ条件:**
- ✅ サブモデル完了（Gate 3 PASS確認済み）
- ✅ メタモデル完了
- ✅ 環境構築完了（Phase 0/1）
- ✅ 重要機能追加（自動化スクリプト等）

**禁止事項:**
- ❌ 試行錯誤中のコミット
- ❌ デバッグコード
- ❌ 実験的変更
- ❌ Gate失敗したサブモデル

#### `develop` - 開発ブランチ
**目的:** 日常的な開発作業全般
**許可事項:**
- ✅ entrance → evaluator の全サイクル
- ✅ 試行錯誤・デバッグ
- ✅ ファイル整理・リファクタリング
- ✅ Gate 1/2失敗時の改善試行

**コミット頻度:** エージェント完了毎（現行通り）

#### `feature/*` - フィーチャーブランチ（オプション）
**目的:** 大規模サブモデル開発の隔離
**使用例:**
- `feature/submodel-inflation_expectation`
- `feature/meta-model`
- `feature/auto-resume-v2`

**運用:** developから分岐 → 完了後developにマージ → developからmainにマージ

---

## コミットメッセージ規約（日本語）

### developブランチ

**現行ルールを継続（英語のままOK、説明部分は日本語推奨）:**
```bash
entrance: {feature} attempt {N}
research: {feature} attempt {N}
design: {feature} attempt {N}
data: {feature} attempt {N}
datacheck: {feature} attempt {N} - {PASS/REJECT}
model: {feature} attempt {N} - Notebook生成完了
kaggle: {feature} attempt {N} - {送信完了/結果取得完了}
eval: {feature} attempt {N} - gate{N} {pass/fail}
リファクタ: {説明}
整理: {説明}
状態: {説明}
```

### mainブランチ

**マージコミットのみ（日本語、簡潔に）:**
```bash
機能: {feature}サブモデル完了（Gate 3 PASS、メトリクス）
機能: メタモデル完了（最終メトリクス）
機能: {機能名}追加
リファクタ: {大規模変更の説明}
ドキュメント: {更新内容}
```

**例:**
```bash
機能: etf_flowサブモデル完了（Gate 3 PASS、Sharpe +0.377、MAE -0.0436）
機能: vixサブモデル完了（Gate 3 PASS、DA +0.96%、Sharpe +0.289）
機能: Kaggleトレーニング後の自動再開スクリプト追加
リファクタ: CLAUDE.md準拠のプロジェクト構造再構築
```

---

## ワークフロー

### 日常開発（サブモデル構築）

```bash
# 1. developブランチで作業
git checkout develop

# 2. エージェントサイクル実行
# entrance → researcher → ... → evaluator
# orchestratorが自動コミット（現行通り）

# 3. Gate 3 PASS確認後、mainにマージ
git checkout main
git merge develop --no-ff -m "feat: complete inflation_expectation submodel (Gate 3 PASS, Sharpe +0.12, MAE -0.02)"
git push origin main

# 4. developに戻る
git checkout develop
```

### リファクタリング・環境整備

```bash
# developで作業
git checkout develop

# ファイル整理、スクリプト改善など
git add -A
git commit -m "cleanup: remove redundant tmp files"
git push origin develop

# 重要な変更のみmainにマージ
git checkout main
git merge develop --no-ff -m "refactor: improve Kaggle error handling"
git push origin main
```

### フィーチャーブランチ使用（オプション）

```bash
# 大規模開発の場合
git checkout develop
git checkout -b feature/submodel-inflation_expectation

# 開発作業...
git commit -m "design: inflation_expectation attempt 1"
git commit -m "data: inflation_expectation attempt 1"

# 完了後developにマージ
git checkout develop
git merge feature/submodel-inflation_expectation --no-ff
git branch -d feature/submodel-inflation_expectation

# Gate 3 PASS後mainにマージ
git checkout main
git merge develop --no-ff -m "feat: complete inflation_expectation submodel"
```

---

## orchestratorの動作変更

### 現在の動作
```python
# 毎エージェント完了後
git add -A
git commit -m "{agent}: {feature} attempt {N}"
git push  # ← main に直接プッシュ
```

### 変更後の動作
```python
# 毎エージェント完了後
git add -A
git commit -m "{agent}: {feature} attempt {N}"
git push origin develop  # ← develop にプッシュ

# Gate 3 PASS時のみ
if gate3_pass:
    git checkout main
    git merge develop --no-ff -m "feat: complete {feature} submodel (metrics)"
    git push origin main
    git checkout develop
```

---

## 初回セットアップ

### 1. developブランチ作成

```bash
# 現在のmainから作成
git checkout -b develop
git push -u origin develop

# 以降の開発はdevelopで
git checkout develop
```

### 2. shared/state.json に現在のブランチを記録

```json
{
  "git_branch": "develop",
  "git_main_last_merge": "2026-02-15T15:30:00",
  ...
}
```

### 3. orchestratorスクリプト更新

`scripts/kaggle_ops.py` でブランチを意識した動作に変更

---

## メリット

✅ **mainブランチがクリーン:** 完成品のみ、レビュー可能
✅ **履歴が明確:** マイルストーン単位で進捗確認
✅ **ロールバック容易:** Gate失敗時もmainは影響なし
✅ **並行開発可能:** 複数サブモデルをfeatureブランチで隔離
✅ **CI/CD準備:** mainブランチにテスト・デプロイを自動化可能

---

## 注意事項

⚠️ **PCシャットダウン可能性:**
Kaggleトレーニング中にPCを閉じる場合、`git push origin develop`は完了している必要あり。
→ `kaggle_ops.py` の `submit()` がKaggle送信前に必ずgit commit & pushする。

⚠️ **コンフリクト対策:**
developで長時間作業後、mainにマージ前に`git pull origin main`で同期確認。

⚠️ **kaggle_ops.py monitor:**
バックグラウンド実行時、developブランチでcommit&pushする。

---

## 移行手順（今すぐ実行）

```bash
# 1. 現在のmainを保護（タグ付け）
git tag v0.1-before-branching
git push origin v0.1-before-branching

# 2. developブランチ作成
git checkout -b develop
git push -u origin develop

# 3. 以降の作業はdevelopで
# state.jsonをdevelopに更新して最初のコミット
git add shared/state.json
git commit -m "setup: switch to develop branch workflow"
git push origin develop

# 4. orchestratorスクリプト更新は次回作業で
```

---

## 今後の運用例

```
develop ブランチ:
  ├─ entrance: inflation_expectation attempt 1
  ├─ research: inflation_expectation attempt 1
  ├─ design: inflation_expectation attempt 1
  ├─ data: inflation_expectation attempt 1
  ├─ kaggle: inflation_expectation attempt 1 - submitted
  ├─ eval: inflation_expectation attempt 1 - gate3 pass
  └─ [merge to main]

main ブランチ:
  ├─ feat: complete etf_flow submodel (Gate 3 PASS)
  ├─ feat: complete inflation_expectation submodel (Gate 3 PASS)
  └─ feat: complete meta-model (final DA=58.2%, Sharpe=1.12)
```

これでプロジェクトの品質とトレーサビリティが大幅に向上します。
