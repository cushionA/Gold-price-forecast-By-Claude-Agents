# Automation Guide: Kaggle Training Workflow

## Overview

Kaggle学習ワークフローの自動化ガイド。
全操作は `scripts/kaggle_ops.py` (Python API v2.0.0) に統一。

---

## Workflow

```
[PC on] builder_model: Generate Kaggle Notebook
  ↓
orchestrator: submit_and_monitor()
  - Python API: kernels_push
  - state.json → "waiting_training"
  - git commit & push
  - Start background monitor (detached process)
  ↓
[PC off OK] monitor polls every 60s (max 3 hours)
  ↓
[Kaggle complete] monitor detects completion
  - Python API: kernels_output (download results)
  - state.json → "in_progress", resume_from="evaluator"
  - git commit & push
  ↓
[PC on] User says "Resume from where we left off"
  - orchestrator reads state.json
  - evaluator runs Gate 1/2/3 (Claude Code agent)
  - evaluator decides next action
  ↓
[Loop continues with fresh context]
```

---

## Usage

### Python API

```python
from scripts.kaggle_ops import submit_and_monitor, submit, monitor

# Submit + background monitor (recommended)
result = submit_and_monitor(
    folder="notebooks/real_rate_1/",
    feature="real_rate",
    attempt=1,
)

# Submit only (no background monitor)
result = submit(
    folder="notebooks/real_rate_1/",
    feature="real_rate",
    attempt=1,
)

# Manual monitoring (reads kernel_id from state.json)
result = monitor()             # blocking: poll until complete/error
result = monitor(once=True)    # single check and return
```

### CLI

```bash
# Submit + background monitor
python scripts/kaggle_ops.py submit notebooks/real_rate_1/ real_rate 1 --monitor

# Submit only
python scripts/kaggle_ops.py submit notebooks/real_rate_1/ real_rate 1

# Monitor (blocking)
python scripts/kaggle_ops.py monitor

# Single status check
python scripts/kaggle_ops.py monitor --once

# Check interval and timeout
python scripts/kaggle_ops.py monitor --interval 120 --max-hours 5
```

### Basic Operations

```bash
# Notebook push (low-level)
python scripts/kaggle_ops.py notebook-push notebooks/real_rate_1/

# Kernel status
python scripts/kaggle_ops.py kernel-status bigbigzabuton/gold-real-rate-1

# Download kernel output
python scripts/kaggle_ops.py kernel-output bigbigzabuton/gold-real-rate-1 data/outputs/

# Create dataset
python scripts/kaggle_ops.py dataset-create data/submodel_outputs/

# Update dataset (new version)
python scripts/kaggle_ops.py dataset-update data/dataset_upload_clean/ "v3: added file"
```

---

## Error Handling

monitor が自動的にエラーを分類し、state.json に記録する。

| Error Type | Pattern | Action |
|------------|---------|--------|
| `network_timeout` | connection, timeout | resume_from=builder_model (retry) |
| `oom` | memory, killed | resume_from=builder_model (reduce model) |
| `pandas_compat` | fillna, append | resume_from=builder_model (fix deprecated API) |
| `missing_dep` | cannot import | resume_from=builder_model (add dependency) |
| `dataset_missing` | no such file | resume_from=builder_model (fix dataset ref) |
| `yfinance` | multiindex | resume_from=builder_model (fix data fetch) |
| `unknown` | (other) | resume_from=builder_model (manual review) |

エラー時の state.json:
```json
{
  "status": "in_progress",
  "resume_from": "builder_model",
  "error_type": "pandas_compat",
  "error_context": "... error message ..."
}
```

---

## State Management

### submit_and_monitor() 実行後

```json
{
  "status": "waiting_training",
  "resume_from": "evaluator",
  "kaggle_kernel": "bigbigzabuton/gold-real-rate-1",
  "submitted_at": "2026-02-16T10:00:00",
  "current_feature": "real_rate",
  "current_attempt": 1
}
```

### monitor 完了後 (success)

```json
{
  "status": "in_progress",
  "resume_from": "evaluator",
  "kaggle_kernel": "bigbigzabuton/gold-real-rate-1"
}
```

### monitor 完了後 (error)

```json
{
  "status": "in_progress",
  "resume_from": "builder_model",
  "error_type": "pandas_compat",
  "error_context": "..."
}
```

---

## Evaluator Execution

monitor 完了後の evaluator 実行は **Claude Code エージェント** が担当する。
自動インライン実行は行わない。

```
1. monitor completes → state.json updated → git push
2. User says "Resume from where we left off"
3. orchestrator reads state.json (resume_from="evaluator")
4. @evaluator agent runs Gate 1/2/3
5. evaluator decides: PASS / attempt+1 / no_further_improvement
6. state.json updated → git push
7. Next iteration or next feature
```

---

## Technical Details

### Windows cp932 Encoding Fix

Kaggle Python API v2.0.0 は `open(file, "w")` を encoding 指定なしで使用する。
Windows では cp932 がデフォルトとなり、Unicode文字でクラッシュする。
`kaggle_ops.py` は `builtins.open` をパッチして UTF-8 をデフォルトにする。

### Authentication

`.env` から自動ロード:
- `KAGGLE_API_TOKEN` → Python API が必要とする `KAGGLE_KEY` に自動マッピング
- `KAGGLE_USERNAME` → kernel ID の構築に使用

### Background Monitor (Windows)

```python
subprocess.Popen(
    [sys.executable, script, "monitor"],
    creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS,
)
```

PC シャットダウン後もプロセスは継続する（ただし再起動時は終了）。

---

## Troubleshooting

### Monitor が起動しない

```bash
# 手動で起動
python scripts/kaggle_ops.py monitor
```

### 3時間タイムアウト

```bash
# Kaggle Web UI でステータス確認
# まだ実行中なら、タイムアウトを延長して再実行
python scripts/kaggle_ops.py monitor --max-hours 6
```

### 409 Conflict エラー

既存カーネルが running/queued の場合に発生。
Kaggle Web UI で既存カーネルを削除してから再提出。
