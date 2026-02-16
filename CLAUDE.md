# Gold Price Prediction Model - Autonomous Agent System v5

## Startup Mode

The orchestrator (you) first runs `git pull` and reads `shared/state.json` on startup.

### Auto-Detection

```
1. git pull to get latest state
2. Check shared/state.json status
3. User instructions take priority if provided

status == "not_started"            → Start from Phase 0
status == "in_progress"            → Resume from resume_from agent
status == "waiting_training"       → Fetch Kaggle training results → evaluate
status == "waiting_user_input"     → Waiting for user action. Await instructions
status == "paused_max_iterations"  → Check pending_tasks, resume from resume_from
status == "completed"              → Output final report
```

### User Instruction Examples

```
"Start the project"                → Fresh start
"Resume from where we left off"    → Auto-resume per state.json
"Resume real_rate from attempt 3"  → Overwrite state.json, resume from specified point
"Build only the vix submodel"      → Execute vix only
"Proceed to meta-model"            → Jump to Phase 3
"Check training results"           → Fetch Kaggle results → evaluate
```

---

## Mission

Build a regression model to predict next-day gold return (%).
Increase information via 9 key features × submodels, then integrate with a meta-model.

## Core Concepts

- Meta-model directly processes key features
- Submodels supplement context, state, and characteristics
- **Submodels do NOT predict gold prices** — they extract latent patterns (persistence, regime, mean-reversion probability, etc.) from the feature's underlying dynamics using unsupervised, semi-supervised, or supervised learning
- Submodel quality is measured by whether the output provides statistically significant information gain to the meta-model (Gate 2/3), not by the submodel's own loss
- Actively leverage multi-country data to maximize training samples

---

## Key Features (9)

| # | Feature | Source | Importance | Note |
|---|---------|--------|------------|------|
| 1 | Real Interest Rate (10Y TIPS) | FRED: DFII10 | Strongest negative correlation | Yield Curve(DGS10)と相関高い。VIF注意 |
| 2 | Dollar Index (DXY) | Yahoo: DX-Y.NYB | USD-denominated inverse correlation | |
| 3 | VIX | FRED: VIXCLS | Risk-off indicator | |
| 4 | Gold Technicals | Yahoo: GC=F, GLD | Momentum / mean-reversion | |
| 5 | Cross-Asset | Yahoo: SI=F, HG=F, ^GSPC | Relative attractiveness | |
| 6 | Yield Curve | FRED: DGS10, DGS2 | Policy expectations | Real Rate(DFII10)と相関高い。VIF注意 |
| 7 | ETF Flows (proxy) | Yahoo: GLD | Investor demand | 真のETF純流入出額は取得困難。GLD volume + shares outstanding変化率を代理指標として使用 |
| 8 | Inflation Expectation | FRED: T10YIE | Real asset demand driver | 10Y Breakeven Inflation Rate。TIPS利回りに含まれるが独立した変動パターンを持つ |
| 9 | CNY Demand Proxy | Yahoo: CNY=X | China demand (~30% of gold demand) | 上海プレミアムの代理としてCNY/USD為替を使用 |

---

## Execution Architecture: Claude Code + Kaggle Separation

### Why Separate?

```
Claude Code = Local PC process. Stops when PC shuts down.
Kaggle      = Cloud. Training continues even when PC is off.
Git         = Bridge between the two.
```

### Role Assignment

| Task | Execution | PC Required? |
|------|-----------|-------------|
| Requirements (entrance) | Claude Code | Yes |
| Research (researcher) | Claude Code | Yes |
| Design (architect) | Claude Code | Yes |
| Data fetching (builder_data) | Claude Code | Yes |
| Data check (datachecker) | Claude Code | Yes |
| **Training script generation** (builder_model) | Claude Code | Yes |
| **Training execution** | **Kaggle** | **No** |
| Evaluation (evaluator) | Claude Code | Yes |
| Improvement planning | Claude Code | Yes |

### Typical Workflow

```
1. [PC on] Claude Code: Design → Data fetch → Check → Training script generation
2. [PC on] Claude Code: Submit training Notebook via Kaggle API
3. [PC off OK] Kaggle: Cloud training execution (minutes to ~30min)
4. [PC on] Claude Code: "Resume from where we left off"
   → git pull → Fetch Kaggle results → Evaluate → Next iteration
```

---

## Kaggle Integration

### Kaggle Notebook Structure

builder_model auto-generates the following Notebook:

```
notebooks/
  └── {feature}_{attempt}/
      ├── kernel-metadata.json    ← Kaggle API config
      └── train.ipynb             ← Jupyter Notebook (self-contained training script)
```

**重要**: 全サブモデルで統一Notebook「**Gold Model Training**」（ID: `bigbigzabuton/gold-model-training`）を使用します。このNotebookにはFRED_API_KEYが有効化されています。

### kernel-metadata.json

**CRITICAL: All notebooks MUST include the submodel dataset reference**

```json
{
  "id": "bigbigzabuton/gold-{feature}-{attempt}",
  "title": "Gold {Feature} Model - Attempt {attempt}",
  "code_file": "train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["bigbigzabuton/gold-prediction-submodels"],
  "competition_sources": [],
  "kernel_sources": []
}
```

**Notes**:
- `dataset_sources` MUST include `"bigbigzabuton/gold-prediction-submodels"` for all submodels and meta-models
- This dataset contains all pre-computed submodel outputs (vix.csv, technical.csv, etc.)
- Kaggle mounts this at `/kaggle/input/gold-prediction-submodels/`
- Missing this dataset will cause immediate runtime failure

### train.ipynb Design Principles

Training notebooks must be **self-contained** Jupyter Notebooks. No dependency on Claude Code or Kaggle:

```python
"""
Gold Prediction SubModel Training - {feature} Attempt {attempt}
Self-contained: Data fetch → Preprocessing → Training → Evaluation → Save results, all in this file
"""

# === 1. Libraries ===
import torch, torch.nn as nn, pandas as pd, numpy as np
import json, os
from datetime import datetime

# === 2. Data Fetching (direct API calls) ===
# Embed builder_data's verified data fetching code as-is
def fetch_data():
    ...
    return train_data, val_data, test_data

# === 3. Model Definition ===
# Implement architect's design directly
class SubModel(nn.Module):
    ...

# === 4. Training Loop ===
def train(model, train_data, val_data, config):
    ...
    return model, metrics

# === 5. Optuna HPO ===
def run_hpo(train_data, val_data):
    ...
    return best_params, best_value

# === 6. Main Execution ===
if __name__ == "__main__":
    train_data, val_data, test_data = fetch_data()
    best_params, _ = run_hpo(train_data, val_data)

    model = SubModel(**best_params)
    model, metrics = train(model, train_data, val_data, best_params)

    # Generate submodel output
    output = model.transform(full_data)

    # === 7. Save Results (Kaggle output) ===
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

### Kaggle Operations (`scripts/kaggle_ops.py`)

All Kaggle operations use Python API v2.0.0 via a single unified module.
Includes Windows cp932 encoding fix (builtins.open monkey-patch).

#### Basic Operations

```python
from scripts.kaggle_ops import notebook_push, kernel_status, kernel_output
from scripts.kaggle_ops import dataset_create, dataset_update

# 1. Push notebook (new or update, auto-detected by Kaggle)
result = notebook_push("notebooks/real_rate_1/")

# 2. Check kernel status
result = kernel_status("bigbigzabuton/gold-real-rate-1")
# result.data["status"] → "queued" / "running" / "complete" / "error"

# 3. Download kernel output
result = kernel_output("bigbigzabuton/gold-real-rate-1", "data/outputs/")

# 4. Create new dataset
result = dataset_create("data/submodel_outputs/")

# 5. Update existing dataset (new version)
result = dataset_update("data/dataset_upload_clean/", "v3: added temporal_context")
```

#### Workflow Operations

```python
from scripts.kaggle_ops import submit, submit_and_monitor, monitor

# Submit: push + state.json update + git commit & push
result = submit("notebooks/real_rate_1/", feature="real_rate", attempt=1)

# Submit + start background monitor
result = submit_and_monitor("notebooks/real_rate_1/", feature="real_rate", attempt=1)

# Monitor: poll until complete, then download results + update state + git commit
result = monitor()  # reads kernel_id from state.json
```

#### CLI

```bash
# Basic operations
python scripts/kaggle_ops.py notebook-push notebooks/real_rate_1/
python scripts/kaggle_ops.py kernel-status bigbigzabuton/gold-real-rate-1
python scripts/kaggle_ops.py kernel-output bigbigzabuton/gold-real-rate-1 data/outputs/
python scripts/kaggle_ops.py dataset-create data/submodel_outputs/
python scripts/kaggle_ops.py dataset-update data/dataset_upload_clean/ "v3: notes"

# Workflow: submit + state + git
python scripts/kaggle_ops.py submit notebooks/real_rate_1/ real_rate 1

# Workflow: submit + background monitor
python scripts/kaggle_ops.py submit notebooks/real_rate_1/ real_rate 1 --monitor

# Workflow: monitor (reads from state.json)
python scripts/kaggle_ops.py monitor
python scripts/kaggle_ops.py monitor --once          # single check
python scripts/kaggle_ops.py monitor --interval 120  # check every 2 min
```

#### Execution Flow

```
[PC on] builder_model: Generate Kaggle Notebook
  ↓
orchestrator: submit_and_monitor()
  - Python API: kernels_push
  - state.json → "waiting_training"
  - git commit & push
  - Start background monitor
  ↓
[PC off OK] monitor polls every 1 min (max 3 hours)
  ↓
[Kaggle complete] monitor detects completion
  - Python API: kernels_output (download results)
  - state.json → "in_progress", resume_from="evaluator"
  - git commit & push
  ↓
[Continue] User says "Resume from where we left off"
  - orchestrator reads state.json
  - evaluator runs Gate 1/2/3 (Claude Code agent, not inline)
  - evaluator decides next action
  ↓
[Loop continues with fresh context]
```

#### Error Handling

Monitor auto-classifies errors:
- `network_timeout` → state resume_from=builder_model (retry same code)
- `oom` → state resume_from=builder_model (reduce model size)
- `pandas_compat` → state resume_from=builder_model (fix deprecated API)
- `dataset_missing` → state resume_from=builder_model (fix dataset ref)
- `unknown` → state resume_from=builder_model (manual investigation)

All errors update state.json with `error_type` and `error_context` fields.

---

## Agent Architecture

```
Orchestrator (Sonnet)
  │  * git commit & push after each agent completes
  │  * Training submission/result fetching done directly by orchestrator
  │
  ├─ entrance (Opus)          Initial requirements definition
  ├─ researcher (Sonnet)      Research (subject to fact-checking)
  ├─ architect (Opus)         Fact-check → Design doc → HP search space
  ├─ builder_data (Sonnet)    Data fetching & preprocessing
  ├─ datachecker (Haiku)      Standardized 7-step check
  ├─ builder_model (Sonnet)   PyTorch training script generation (for Kaggle) + Notebook validation
  └─ evaluator (Opus)         Gate 1/2/3 → Loop control → Improvement plan
```

### Token Consumption

| Phase | Token Consumption |
|-------|-------------------|
| Design & code generation (Claude Code) | Yes |
| Kaggle training submission (API call) | Minimal |
| **Kaggle training execution** | **None (PC can be off)** |
| Result fetching & evaluation (Claude Code) | Yes |

---

## Workflow

### Phase 0: Environment Setup (first time only)

```
1. git init & remote setup (user prepares in advance)
2. Library installation:
   pip install torch pandas numpy scikit-learn xgboost optuna \
       yfinance fredapi matplotlib scipy statsmodels kaggle python-dotenv
3. Credential verification (stop with waiting_user_input if any are missing):
   a. Local environment variables (.env → auto-loaded via python-dotenv):
      - FRED_API_KEY → set in .env?
      - KAGGLE_USERNAME → set in .env?
      - KAGGLE_API_TOKEN → set in .env? (environment variable format, DO NOT use ~/.kaggle/kaggle.json)
   b. Kaggle CLI authentication:
      - KAGGLE_API_TOKEN is set and kaggle kernels list succeeds?
      - Note: Use environment variables ONLY. Delete ~/.kaggle/kaggle.json if it exists.
   c. Kaggle Secrets (required for API calls inside Kaggle Notebooks):
      - https://www.kaggle.com/settings → Secrets → FRED_API_KEY registered?
      - * User must configure this in browser
4. Create shared code:
   - src/submodel_base.py
   - src/data_fetcher.py
   - src/evaluation.py
   - src/utils.py
   - src/kaggle_runner.py  ← Common Kaggle submission/retrieval functions
5. Fetch base data:
   - Gold price GC=F → data/raw/gold.csv
   - Raw data for 9 key features → data/raw/
   - Target variable (next-day return %) → data/processed/target.csv
6. git commit & push "phase0: environment setup and base data"
```

### Phase 1: Baseline Construction

```
1. Prepare direct input data for 9 key features → data/processed/base_features.csv
2. Split data into train/val/test (70/15/15, time-series order)
3. Train XGBoost baseline (local execution, small enough to finish immediately)
4. Record baseline score in shared/baseline_score.json
5. Freeze base_features schema → shared/schema_freeze.json
   (columns, dtypes, date_range, row_count — used for Gate 2/3 validation)
6. git commit & push "phase1: baseline (DA=xx%, Sharpe=x.xx)"
```

### Phase 1.5: Smoke Test

```
1. Run full pipeline once with real_rate (simplified version)
   - entrance → researcher → architect → builder_data → datachecker
   - builder_model → Generate Kaggle Notebook (Optuna 5 trials)
   - Kaggle submission → Fetch results → evaluator (Gate 1 only)
2. Verify Kaggle integration works without errors
3. git commit & push "smoke_test: pipeline with kaggle verified"
```

### Phase 2: Submodel Construction Loop

```
[PC on] entrance/evaluator → researcher → architect →
        builder_data → datachecker → builder_model (script generation)
        → Kaggle submission → git push
[PC off OK] Kaggle training in progress
[PC on] "Resume" → Fetch Kaggle results → evaluator → (loop or next)
```

### Phase 3: Meta-Model Construction

```
1. architect: Analyze all submodel output formats → Select architecture
2. builder_model: Generate Kaggle Notebook
3. Kaggle training execution
4. evaluator: Evaluate against final target metrics
5. Improvement loop
```

---

## Phase 2 Pipeline Detail

```
┌──────────────────────────────────────────────┐
│ entrance (first time) / evaluator (2nd+)     │
│  → Requirements in current_task.json         │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ researcher (Sonnet)                          │
│  → Report in docs/research/                  │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ architect (Opus)                             │
│  → Fact-check                                │
│  → Fail → researcher re-investigation        │
│  → Design doc in docs/design/                │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ builder_data (Sonnet)                        │
│  → Save data in data/                        │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ datachecker (Haiku) 7-step standardized check│
│  → REJECT → Return to builder_data           │
│    (no attempt consumed, max 3 times)        │
│  → 3 REJECTs → Return to architect           │
│  → PASS → git commit, proceed               │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ builder_model (Sonnet)                       │
│  → Generate self-contained train.py          │
│  → Generate kernel-metadata.json             │
│  → **RUN VALIDATION** (scripts/validate_notebook.py) │
│    • Syntax check                            │
│    • Typo detection (.UPPER() etc)           │
│    • Compatibility warnings (SHAP+XGBoost)   │
│    • Dataset reference check                 │
│    • kernel-metadata.json validation         │
│  → FAIL → Fix and re-validate                │
│  → PASS → git commit                         │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ Orchestrator: Kaggle Submission              │
│  → kaggle kernels push                       │
│  → Update state.json to "waiting_training"   │
│  → git push                                  │
│  → ★ User can shut down PC ★                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ (On PC restart) Orchestrator: Fetch Results  │
│  → git pull                                  │
│  → Check via kaggle kernels status           │
│  → Fetch via kaggle kernels output           │
│  → git commit                                │
└────────────────┬─────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────┐
│ evaluator (Opus)                             │
│  → Gate 1 → 2 → 3                           │
│  → Pass → Append to completed, next submodel │
│  → Fail → attempt+1, improvement plan, loop  │
│  → No improvement possible → next submodel   │
│  → git commit & push                         │
└──────────────────────────────────────────────┘
```

---

## Loop Control

### Attempt Consumption Rules

```
Consumed (+1): evaluator completes Gate evaluation → fail
Not consumed:  datachecker return / architect return / researcher re-investigation
```

### Termination Conditions

1. Gate 3 pass
2. No improvement possible (3 consecutive delta<0.001 or evaluator declaration)
3. attempt >= 5 (paused, can be resumed later)

---

## Tech Stack

| Purpose | Library |
|---------|---------|
| Submodels & meta-model | PyTorch |
| HP optimization | Optuna |
| Data processing | pandas, numpy |
| Gate 3 baseline comparison | XGBoost |
| Evaluation | scikit-learn |
| Cloud training | Kaggle API |

### HP Management Roles

| Role | Owner |
|------|-------|
| Search space design | architect |
| Search execution | builder_model (Optuna in train.py) |
| Result evaluation | evaluator |
| Range adjustment | evaluator → architect |

### Meta-Model Selection Criteria

architect analyzes input formats and selects optimal model:

| Input Format | Recommended |
|-------------|-------------|
| Continuous only <30 dims | MLP |
| Continuous + categorical | FT-Transformer / TabNet |
| Time-series patterns important | GRU + MLP |
| >50 dims, few samples | MLP + strong Dropout / dim reduction |

---

## Multi-Country Data Strategy

Actively leverage large-scale data. Maximizing training samples via multi-country data is key to performance improvement.

| Feature | Extended Data | Expected Samples | Note |
|---------|--------------|-----------------|------|
| Real Interest Rate | G10 countries | ~50,000 (realistic) | FREDでの利用可能シリーズは限定的。architect要確認 |
| DXY | 12 currency pairs | ~60,000 | |
| VIX | Country-specific vol indices | ~25,000 | |
| Cross-Asset | Multi-commodity correlations | ~30,000 | |

Rules:
- Limited to G10 + major developed countries
- Do not use levels; use changes directly
- Normalize volatility by domestic long-term average
- architect must verify actual FRED series availability before design (not all G10 have TIPS-equivalent data)

---

## Evaluation Framework

### Data Split
- train/val/test = 70/15/15 (time-series order, no shuffle)
- HPO and model selection use train+val only
- Gate 3 and final meta-model evaluation use test set

### Gate 1: Standalone Quality
Overfit ratio <1.5 (same epoch comparison), no all-NaN/constant output, no leak indicators

### Gate 2: Information Gain
MI total increase >5% (sum-based, not mean), VIF <10, rolling correlation std <0.15

### Gate 3: Ablation (any one of the following)
Direction accuracy +0.5%, Sharpe +0.05 (after transaction costs), MAE -0.01%

### Sharpe Calculation Rules
- Transaction cost: 5bps per trade (one-way) deducted from strategy returns
- Direction sign: returns exactly 0 are excluded from direction accuracy calculation (np.sign(0) = 0 problem)

### Meta-Model Final Targets

| Metric | Target |
|--------|--------|
| Direction Accuracy | > 56% |
| High-Confidence Direction Accuracy | > 60% |
| MAE | < 0.75% |
| Sharpe Ratio (after costs) | > 0.8 |

---

## Git Persistence

### Branch Strategy

**develop branch (daily work):**
- All agent cycles: entrance → researcher → ... → evaluator
- Trials, debugging, file organization
- Orchestrator auto-commits after each agent

**main branch (milestones only):**
- Submodel completion (Gate 3 PASS)
- Meta-model completion
- Environment setup completion
- Critical feature additions

### Commit Rules

**On develop branch** (orchestrator runs after each agent):

```bash
git add -A && git commit && git push origin develop
```

Commit messages:
```
entrance done     → "entrance: {feature} attempt {N}"
researcher done   → "research: {feature} attempt {N}"
architect done    → "design: {feature} attempt {N}"
builder_data done → "data: {feature} attempt {N}"
datachecker done  → "datacheck: {feature} attempt {N} - {PASS/REJECT}"
builder_model done → "model: {feature} attempt {N} - notebook generated"
kaggle submit     → "kaggle: {feature} attempt {N} - submitted"
kaggle fetch      → "kaggle: {feature} attempt {N} - results fetched"
evaluator done    → "eval: {feature} attempt {N} - gate{N} {pass/fail}"
cleanup/refactor  → "cleanup: {description}" or "refactor: {description}"
```

**On main branch** (manual or automated on Gate 3 PASS):

```bash
git checkout main
git merge develop --no-ff -m "feat: complete {feature} submodel (metrics)"
git push origin main
git checkout develop
```

Merge commit messages (Conventional Commits):
```
feat: complete {feature} submodel (Gate 3 PASS, Sharpe +X.XX, MAE -X.XX)
feat: complete meta-model (DA=XX%, Sharpe=X.XX)
feat: add {feature_name}
refactor: {major_refactoring}
```

**Current branch:** Check `shared/state.json` → `git_branch` field

See `docs/knowledge/GIT_WORKFLOW.md` for detailed workflow.

---

## Shared Workspace

```
shared/
  ├── state.json              Progress state, resume point, Kaggle state
  ├── current_task.json       Current iteration requirements
  ├── improvement_queue.json  Improvement task queue
  └── completed.json          Completed submodel records
```

---

## Project Structure

```
gold-prediction-agent/
├── CLAUDE.md
├── .gitignore
├── .claude/agents/           7 agents
├── shared/                   State management
├── src/                      Shared code
│   ├── submodel_base.py
│   ├── data_fetcher.py
│   ├── evaluation.py
│   ├── utils.py
│   └── kaggle_runner.py      ← Kaggle API operations
├── notebooks/                ← Kaggle Notebooks (auto-generated)
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

## Prohibited Actions

1. No random splits (time-series splits only)
2. No future information leakage
3. No data with >5 day delay
4. Submodels must NOT directly predict gold prices
5. No hardcoded API keys or credentials (including train.py)
6. Do not execute processes requiring paid services or new API keys without explicit user approval

## File Generation Policy

**DO NOT create the following files** (they are redundant and waste resources):

1. ❌ `notebooks/**/README.md` - Duplicates design documents in `docs/design/`
2. ❌ `data/**/metadata.json` or `*_metadata.json` - Not used after datachecker passes
3. ❌ `docs/data/*_summary.md` - Duplicates design documents
4. ❌ `temp_*/` directories - Use proper output directories instead

**DO create**:
- ✅ `logs/evaluation/*_summary.md` - User needs these for review
- ✅ Design docs in `docs/design/`
- ✅ Research reports in `docs/research/`
- ✅ Evaluation logs in `logs/evaluation/*.json`

All agents must follow this policy to avoid creating unnecessary files.

---

## APIs

| API | Purpose | Auth | Local | Inside Kaggle Notebook |
|-----|---------|------|-------|----------------------|
| yfinance | Price data | None | — | — |
| fredapi | Economic indicators | FRED_API_KEY | .env | Kaggle Secrets |
| kaggle CLI | Training submission/retrieval | KAGGLE_API_TOKEN | .env | — (CLI is local only) |
| CNN Fear & Greed | Risk indicator | None | — | — |
| CBOE Put/Call | Risk indicator | None | — | — |
| GPR Index | Geopolitical indicator | None | — | — |

### Credential Management Principles

- Local: .env file (gitignored) → auto-loaded via python-dotenv
- Kaggle CLI: KAGGLE_USERNAME + KAGGLE_API_TOKEN env vars (via .env)
- **CRITICAL**: DO NOT use ~/.kaggle/kaggle.json. Delete it if exists. Use environment variables ONLY.
- Kaggle Notebook: Kaggle Secrets (configured in browser)
- Inside train.py: os.environ['FRED_API_KEY'] (fail immediately with KeyError)
- **Never use hardcoded values or fallback defaults**
