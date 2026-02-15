# Automation Guide: Auto vs Manual Mode

## Overview

This project supports two execution modes for Kaggle training workflows:

- 🤖 **Full Auto Mode** (default): Fully automated evaluation and loop control
- 👤 **Manual Mode**: User-controlled evaluation and decision-making

---

## 🤖 Full Auto Mode (Recommended)

### Features

✅ Background monitoring (1-minute intervals, max 3 hours)
✅ Automatic evaluator execution (no Claude Code restart)
✅ Automatic decision-making (attempt+1 / next feature / done)
✅ Intelligent error handling (auto-retry / auto-skip)
✅ Git persistence (state.json auto-update)
✅ PC can be closed during Kaggle training

### Usage in Orchestrator

```python
from scripts.orchestrator_kaggle_handler import KaggleSubmissionHandler

handler = KaggleSubmissionHandler()
handler.submit_and_exit(
    notebook_path='notebooks/real_rate_1/',
    feature='real_rate',
    attempt=1,
    auto_mode=True  # ← Full auto mode (default)
)
```

### Command-line

```bash
# Full auto mode (default)
python scripts/orchestrator_kaggle_handler.py notebooks/real_rate_1/ real_rate 1

# Same as above (auto_mode=True is default)
python scripts/orchestrator_kaggle_handler.py notebooks/real_rate_1/ real_rate 1 --no-exit
```

### Workflow

```
1. builder_model generates Kaggle Notebook
   ↓
2. orchestrator calls handler.submit_and_exit(auto_mode=True)
   - Submits to Kaggle
   - Starts auto_resume_after_kaggle.py in background
   - Exits orchestrator
   ↓
3. auto_resume_after_kaggle.py monitors every 1 minute
   - Status check: kaggle kernels status <kernel_id>
   ↓
4. When training completes:
   - Downloads results: kaggle kernels output
   - Git commit & push
   - Runs evaluator INLINE (no Claude restart)
   - Gate 1 → Gate 2 → Gate 3 evaluation
   ↓
5. Automatic decision:
   - Gate 3 PASS → mark completed, move to next feature
   - Gate 3 FAIL → set resume_from=architect, increment attempt
   - No improvement → move to next feature
   ↓
6. state.json updated, git commit & push
   ↓
7. User says "Resume from where we left off"
   - orchestrator reads state.json
   - Resumes from designated agent
```

### Error Handling

| Error Type | Action |
|------------|--------|
| `network_timeout` | Auto-retry (resubmit same notebook) |
| `yfinance_multiindex` | Set resume_from=builder_model (code fix needed) |
| `pandas_api_change` | Set resume_from=builder_model (code fix needed) |
| `out_of_memory` | Skip to next feature (OOM is fatal) |
| `unknown` | Set resume_from=builder_model (manual review) |

---

## 👤 Manual Mode

### Features

✅ User controls evaluation timing
✅ Manual review of results before decisions
✅ No background processes
✅ Suitable for debugging or custom workflows

### Usage in Orchestrator

```python
from scripts.orchestrator_kaggle_handler import KaggleSubmissionHandler

handler = KaggleSubmissionHandler()
handler.submit_and_exit(
    notebook_path='notebooks/real_rate_1/',
    feature='real_rate',
    attempt=1,
    auto_mode=False  # ← Manual mode
)
```

### Command-line

```bash
# Manual mode
python scripts/orchestrator_kaggle_handler.py notebooks/real_rate_1/ real_rate 1 --manual
```

### Workflow

```
1. builder_model generates Kaggle Notebook
   ↓
2. orchestrator calls handler.submit_and_exit(auto_mode=False)
   - Submits to Kaggle
   - NO background monitoring
   - Prints kernel URL
   ↓
3. User manually checks Kaggle web UI
   - Wait for "complete" status
   ↓
4. User says "Resume from where we left off"
   - orchestrator fetches results
   - evaluator runs (Gate 1/2/3)
   ↓
5. User reviews evaluation results
   - Decide next action manually
   - Continue or adjust strategy
```

---

## Comparison

| Feature | Auto Mode | Manual Mode |
|---------|-----------|-------------|
| Background monitoring | ✅ Yes (1-min intervals) | ❌ No |
| Evaluator auto-run | ✅ Yes (inline) | ❌ No (user triggers) |
| Decision-making | ✅ Automatic | 👤 Manual |
| Error handling | ✅ Intelligent (7 types) | 👤 Manual review |
| PC can be closed | ✅ Yes (monitoring continues) | ⚠️ No effect (no monitor) |
| Git persistence | ✅ Auto commit/push | 👤 User commits |
| Best for | Production loops | Debugging, custom flows |

---

## Switching Modes

You can switch between modes at any time:

```python
# Start with auto mode
handler.submit_and_exit(..., auto_mode=True)

# Later, if auto-monitor fails, manually resume:
# 1. Check state.json → status="waiting_training"
# 2. Manually run: python scripts/auto_resume_after_kaggle.py
# Or manually fetch and evaluate

# Start next submission with manual mode
handler.submit_and_exit(..., auto_mode=False)
```

---

## Troubleshooting

### Auto mode not starting

**Symptom**: Notebook submitted but no background monitor

**Solution**:
```bash
# Check if monitor is running
ps aux | grep auto_resume  # Unix
tasklist | findstr python  # Windows

# Manually start if needed
python scripts/auto_resume_after_kaggle.py
```

### Monitor timeout (3 hours)

**Symptom**: state.json shows `status="timeout"`

**Solution**:
```bash
# Check Kaggle web UI for actual status
# If still running, wait and manually fetch:
python scripts/kaggle_fetch_results.py <kernel_id>

# If complete, resume:
# Say "Resume from where we left off"
```

### Evaluator decision unclear

**Symptom**: Not sure why auto-evaluator chose attempt+1

**Solution**:
```bash
# Check evaluation log
cat logs/evaluation/<feature>_<attempt>_auto.json

# Review Gate 1/2/3 failures
# Adjust improvement plan in current_task.json if needed
```

---

## Best Practices

### Use Auto Mode When:
- ✅ Running production submodel loops
- ✅ Overnight or multi-day training
- ✅ Consistent failure patterns (auto-retry helps)
- ✅ You want unattended operation

### Use Manual Mode When:
- 👤 Debugging new architectures
- 👤 Testing experimental features
- 👤 Need to review each result carefully
- 👤 Custom evaluation criteria

### Hybrid Approach:
```python
# Phase 2 (submodels): Auto mode for speed
handler.submit_and_exit(..., auto_mode=True)

# Phase 3 (meta-model): Manual mode for careful tuning
handler.submit_and_exit(..., auto_mode=False)
```

---

## Implementation Details

### Auto Mode Internals

1. **Monitor Script**: `scripts/auto_resume_after_kaggle.py`
   - Class: `KaggleMonitor`
   - Check interval: 60 seconds
   - Max wait: 3 hours
   - Background process (detached)

2. **Evaluator Inline**: `KaggleMonitor.run_evaluator_inline()`
   - Simplified Gate 1/2/3 logic
   - Reads `training_result.json`
   - Writes `logs/evaluation/<feature>_<attempt>_auto.json`
   - No Claude Code restart required

3. **Decision Handler**: `KaggleMonitor.handle_evaluation_decision()`
   - Reads evaluation result
   - Updates state.json
   - Git commit & push
   - Sets next action (resume_from)

### Manual Mode Internals

1. **Submission Only**: `orchestrator_kaggle_handler.py`
   - Submits to Kaggle
   - Updates state.json to `waiting_training`
   - Git commit & push
   - Prints kernel URL
   - NO background process

2. **User Triggers Resume**:
   - User says "Resume from where we left off"
   - orchestrator detects `status="waiting_training"`
   - Calls `kaggle kernels output` to fetch results
   - Launches evaluator agent (full Claude Code session)

---

## Migration from Old System

If you're upgrading from the old `auto_resume_after_kaggle_v2.py`:

**Old system**:
```python
# v2 script (deprecated)
handler.submit_and_exit(...)  # Always auto mode
```

**New system**:
```python
# Explicit mode selection
handler.submit_and_exit(..., auto_mode=True)   # Auto
handler.submit_and_exit(..., auto_mode=False)  # Manual
```

**What changed**:
- ✅ Single script: `auto_resume_after_kaggle.py` (v2 removed)
- ✅ Mode selection: `auto_mode` parameter
- ✅ Faster checks: 60s → 60s (was 300s in v1)
- ✅ Better errors: 7 error types with auto-actions
- ✅ No v2 suffix: Clean naming

---

## Summary

- **Default = Auto Mode** → Use unless you need control
- **Manual Mode** → Use for debugging or custom flows
- **Switch anytime** → Just change `auto_mode` parameter
- **State persists** → state.json tracks everything
- **Git is source of truth** → Always commit & push

Choose the mode that fits your workflow, and enjoy automated or manual control as needed!
