# 完全自動化システム

Gold Price Prediction プロジェクトの完全自動化システムの使い方

---

## 概要

このシステムは、Kaggle学習の監視、結果取得、評価、コンテキストクリア、次回試行の開始を**完全自動化**します。

### 利点

✅ **手動介入ゼロ** - 一度起動すれば、全て自動で進行
✅ **PC off OK** - Kaggle学習中はPCを閉じても大丈夫
✅ **メモリ効率** - 評価後に自動でコンテキストクリア
✅ **エラー耐性** - タイムアウト、エラー検出、自動再試行
✅ **Git永続化** - 全ての状態がGitに保存され、いつでも再開可能

---

## 自動化フロー

```
[1] builder_model: Kaggle Notebook生成
      ↓
[2] orchestrator: Kaggle提出 + 自動監視開始
      - kaggle kernels push
      - auto_resume_after_kaggle.py を起動（バックグラウンド）
      - git commit & push
      - orchestrator終了（PCを閉じてOK）
      ↓
[3] auto_resume_after_kaggle.py: 5分ごとに監視（最大3時間）
      - Kaggle完了を検出
      - 結果をダウンロード
      - git commit & push
      - Claude Code CLIを自動起動
      ↓
[4] evaluator: Gate 1/2/3評価
      - 評価完了
      - 改善計画作成
      - auto_clean_and_resume.py を呼び出し
      ↓
[5] auto_clean_and_resume.py: コンテキストクリア + 再開
      - git commit & push
      - claude clean（コンテキスト破棄）
      - Claude Code CLIを新規起動
      - evaluator終了
      ↓
[6] 新しいClaude Codeセッション開始
      - 新鮮なコンテキストで次のattempt開始
      - [1]に戻る（ループ継続）
```

---

## 3つの自動化スクリプト

### 1. `scripts/auto_resume_after_kaggle.py`

**役割**: Kaggle学習完了を監視し、Claude Codeを自動再起動

**機能**:
- `shared/state.json`から現在のKaggle kernel IDを読み込み
- 5分ごとに`kaggle kernels status`をチェック（最大3時間）
- 完了を検出したら：
  - `kaggle kernels output`で結果取得
  - Git commit & push
  - `claude-code`を自動起動（evaluatorタスク付き）
- エラー検出時：
  - エラーログ取得
  - state.json更新
  - Claude Codeを起動（エラー修正を促す）

**使い方**:
```bash
# 自動（orchestratorが呼び出す）
# 手動で起動する必要はありません

# 手動起動（必要な場合のみ）
python scripts/auto_resume_after_kaggle.py
```

**出力例**:
```
======================================================================
[2026-02-14 20:00:00] 🔍 Kaggle Training Monitor Started
======================================================================
Kernel ID: username/gold-real-rate-1
Feature: real_rate, Attempt: 1
Check interval: 300s (5 minutes)
Max wait time: 3.0 hours
======================================================================

[2026-02-14 20:05:00] Check #1 (elapsed: 5.0 min)
⏳ Still running... (next check in 300s)

[2026-02-14 20:10:00] Check #2 (elapsed: 10.0 min)
✅✅✅ Training COMPLETE! ✅✅✅
✅ Results downloaded to data/submodel_outputs/real_rate/
✅ Git pushed
🚀 Resuming Claude Code...
✅ Claude Code launched
🎉 All done! Claude Code will resume evaluation.
```

---

### 2. `scripts/orchestrator_kaggle_handler.py`

**役割**: Kaggle提出 + 自動監視開始を統合

**機能**:
- `kaggle kernels push`でノートブック提出
- Kernel IDを抽出
- state.json更新（`status: "waiting_training"`）
- Git commit & push
- `auto_resume_after_kaggle.py`をバックグラウンド起動
- orchestratorセッションを終了

**使い方**:
```python
# orchestrator内での使用
from scripts.orchestrator_kaggle_handler import KaggleSubmissionHandler

handler = KaggleSubmissionHandler()
handler.submit_and_exit(
    notebook_path='notebooks/real_rate_1/',
    feature='real_rate',
    attempt=1
)
# → Kaggle提出
# → 自動監視開始
# → orchestrator終了（PCを閉じてOK）
```

**出力例**:
```
======================================================================
[2026-02-14 20:00:00] 🚀 Submitting to Kaggle
======================================================================
Feature: real_rate, Attempt: 1
Notebook path: notebooks/real_rate_1/
======================================================================

[Kaggle Output]
Successfully pushed to username/gold-real-rate-1

✅ Kernel ID: username/gold-real-rate-1
✅ state.json updated
✅ Git committed and pushed
✅ Auto-resume monitor started in background

======================================================================
🎉 Kaggle Training Submitted Successfully!
======================================================================
Kernel URL: https://www.kaggle.com/code/username/gold-real-rate-1

📊 Monitoring:
  - Auto-resume script is running in the background
  - It will check every 5 minutes for up to 3 hours
  - Claude Code will automatically restart when training completes

👋 You can now:
  - Close this terminal (monitoring continues in background)
  - Turn off your PC (monitoring will stop, but Kaggle continues)
  - Check Kaggle web UI for live training progress
======================================================================

🛑 Exiting orchestrator session...
(Auto-resume will handle the rest)
```

---

### 3. `scripts/auto_clean_and_resume.py`

**役割**: 評価完了後にコンテキストをクリアして再開

**機能**:
- 評価結果をGit commit & push
- `claude clean`でコンテキストクリア
- state.jsonから次のアクションを決定
- `claude-code`を新しいセッションで起動
- evaluatorセッションを終了

**使い方**:
```python
# evaluator内での使用
from scripts.auto_clean_and_resume import AutoCleanResume

handler = AutoCleanResume()
handler.execute_and_exit(
    feature='real_rate',
    attempt=1,
    decision='attempt+1'  # 'no_further_improvement', 'success'
)
# → 評価結果commit
# → コンテキストクリア
# → 新規Claude Code起動
# → evaluator終了
```

**出力例**:
```
======================================================================
[2026-02-14 21:00:00] 🧹 Auto Clean & Resume
======================================================================
Feature: real_rate, Attempt: 1
Decision: attempt+1
======================================================================
✅ Git pushed: eval: real_rate attempt 1 - attempt+1

🧹 Cleaning context...
✅ Context cleaned (claude clean)

📋 Next action: architect for real_rate attempt 2
Resume message: Continuing real_rate with attempt 2...

🚀 Resuming Claude Code with fresh context...
✅ Claude Code launched in new session

======================================================================
✅ Auto Clean & Resume Complete!
======================================================================
  - Context cleaned
  - Git pushed
  - Claude Code restarted with fresh context
  - Next: architect for real_rate
======================================================================

🛑 Exiting current session...
(New Claude Code session has been started)
```

---

## セットアップ

### 必要な環境

1. **Kaggle API認証**:
   - `~/.kaggle/kaggle.json`が設定済み
   - `KAGGLE_API_TOKEN`環境変数が設定済み（`.env`ファイル）

2. **Claude Code CLI**:
   - `claude-code`コマンドが実行可能
   - プロジェクトディレクトリで動作

3. **Git**:
   - リモートリポジトリ設定済み
   - Push権限あり

### 初回セットアップ

```bash
# 1. ライブラリインストール（既にインストール済みの場合はスキップ）
pip install kaggle

# 2. Kaggle認証確認
kaggle kernels list

# 3. Claude Code動作確認
claude-code --version

# 4. Git設定確認
git remote -v

# 完了！自動化システムは準備完了
```

---

## 使い方

### 通常の開始（自動化あり）

```bash
# プロジェクトディレクトリで
claude-code --message "Resume from where we left off"
```

orchestratorが自動的に：
1. `git pull`で最新状態を取得
2. `state.json`を読み込み
3. 適切なエージェントを起動
4. Kaggle提出時に自動監視を開始
5. 評価完了時に自動でコンテキストクリア + 再開

→ **あなたは何もする必要がありません**

---

## トラブルシューティング

### Q: 自動監視が起動しない

**A**: 手動で起動できます：
```bash
python scripts/auto_resume_after_kaggle.py
```

### Q: Kaggle学習が3時間以上かかる

**A**: タイムアウト後、手動で確認：
```bash
# Kaggle状態確認
kaggle kernels status <kernel_id>

# 完了していたら手動で結果取得
kaggle kernels output <kernel_id> -p data/submodel_outputs/<feature>/

# Claude Code再開
claude-code --message "Kaggle training complete. Please evaluate."
```

### Q: コンテキストクリアが失敗する

**A**: `claude clean`コマンドが使えない場合、自動的にスキップされます。問題ありません。

### Q: PC再起動後、どうすれば再開できる？

**A**: 自動監視スクリプトは停止しますが、Kaggleは継続中です：
```bash
# 1. 最新状態を取得
git pull

# 2. state.jsonを確認
cat shared/state.json

# 3. status == "waiting_training"なら、手動で監視再開
python scripts/auto_resume_after_kaggle.py

# または、Kaggle完了を確認してから手動で再開
claude-code --message "Resume from where we left off"
```

---

## カスタマイズ

### 監視間隔を変更

`scripts/auto_resume_after_kaggle.py`:
```python
self.check_interval = 300  # 5分 → 変更可能（秒単位）
self.max_wait = timedelta(hours=3)  # 3時間 → 変更可能
```

### Slack/Discord通知を追加

`scripts/auto_resume_after_kaggle.py`の`monitor()`メソッド内、完了検出時に：
```python
import requests

def notify_slack(message):
    webhook_url = "YOUR_SLACK_WEBHOOK_URL"
    requests.post(webhook_url, json={'text': message})

# 完了時
notify_slack(f"✅ {feature} attempt {attempt} training complete!")
```

---

## まとめ

この自動化システムにより：

✅ **24時間無人運転** - 寝ている間も学習が進行
✅ **PCリソース節約** - Kaggle学習中はPCを閉じてOK
✅ **メモリ効率** - 評価ごとにコンテキストリフレッシュ
✅ **完全永続化** - 全ての状態がGitに保存
✅ **柔軟な再開** - いつでもどこからでも再開可能

**質問・問題があれば**、`shared/state.json`と各スクリプトのログを確認してください。
