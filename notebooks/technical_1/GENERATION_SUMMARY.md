# Technical Submodel - Attempt 1 - Kaggle Notebook Generation Summary

**Generated:** 2026-02-14
**Agent:** builder_model
**Feature:** Technical (GLD OHLC-based features)

## Files Generated

```
notebooks/technical_1/
  ├── kernel-metadata.json    ✓ Kaggle API configuration
  ├── train.py                ✓ Self-contained training script (489 lines)
  └── GENERATION_SUMMARY.md   ✓ This file
```

## Script Validation

- **Python syntax:** VALID (ast.parse passed)
- **Self-contained:** YES (no external file dependencies)
- **Kaggle-ready:** YES (pip install hmmlearn at start)

## Key Design Decisions

### 1. Data Source: GLD (NOT GC=F)

**Rationale:** GC=F has 137 flat-bar days (H==L==O==C) causing GK vol = 0. GLD has zero such issues.

### 2. HMM Implementation: Manual Multi-Restart Loop

**Critical correction from design doc:**
- hmmlearn does NOT have `n_init` parameter (researcher error)
- Implemented manual loop: `for seed in range(n_restarts): model = GaussianHMM(..., random_state=seed)`
- Keeps best model by log-likelihood score

### 3. HMM Input: 2D [returns, gk_vol]

**Rationale:** Returns-only HMM showed 91% single-state occupancy (weak separation). Adding GK vol leverages OHLC data advantage.

### 4. GK Vol Z-Score: Daily (NO Smoothing)

**Critical design choice:**
- Smoothed version: `gk_vol.rolling(20).mean()` has autocorr = 0.979 (Gate 1 risk)
- Daily version: autocorr = 0.206 (safe)
- Uses daily GK vol directly with baseline window z-score

### 5. Target Data: Computed Directly in Kaggle

**Following VIX v8 success pattern:**
- No dependency on external Gold dataset
- Computes `gold_return_next` from GLD close prices directly
- Self-contained approach eliminates dataset alignment issues

## Model Architecture

### Component 1: HMM Regime Detection
- Input: 2D [returns, GK_vol]
- States: 2 or 3 (Optuna selects)
- Covariance: full
- Training: EM with multi-restart
- Output: P(highest-covariance-trace state)

### Component 2: Mean-Reversion Z-Score
- Input: GLD daily returns
- Window: 15/20/30 days (Optuna selects)
- Computation: rolling z-score, clipped to [-4, 4]
- Output: overbought/oversold indicator

### Component 3: Garman-Klass Volatility Z-Score
- Input: GLD OHLC
- Formula: sqrt(0.5 * log(H/L)^2 - (2*ln2-1) * log(C/O)^2)
- Baseline: 40/60/90 days (Optuna selects)
- Output: volatility regime z-score

## Hyperparameter Search Space

| Parameter | Range | Type |
|-----------|-------|------|
| hmm_n_components | {2, 3} | categorical |
| hmm_n_restarts | {5, 10} | categorical |
| zscore_window | {15, 20, 30} | categorical |
| gk_baseline_window | {40, 60, 90} | categorical |

**Total combinations:** 2 × 2 × 3 × 3 = 36
**Optuna trials:** 30 (83% coverage)
**Timeout:** 600 seconds (10 minutes)
**Objective:** Maximize MI sum on validation set

## Output Format

```csv
date,tech_trend_regime_prob,tech_mean_reversion_z,tech_volatility_regime
2015-01-30,0.234,0.12,-0.45
2015-02-02,0.241,-0.34,0.12
...
```

**Columns:** 3 (matching VIX/DXY success pattern)
**Expected rows:** ~2,500 (matching base_features)

## Kaggle Execution Settings

| Setting | Value |
|---------|-------|
| enable_gpu | false (CPU-only) |
| enable_internet | true (yfinance data fetch) |
| Estimated time | 5-8 minutes |
| Estimated memory | <1 GB |

## Data Split

- Train: 70% (HMM fitting)
- Val: 15% (Optuna optimization)
- Test: 15% (evaluator Gate 3)
- Order: Time-series (no shuffle)

## Expected Gate Performance

### Gate 1: Standalone Quality
- **Overfit ratio:** N/A (non-neural, EM-based)
- **No constant output:** Expected PASS (features vary with market conditions)
- **Autocorrelation <0.99:** Expected PASS (regime ~0.45-0.83, zscore -0.017, vol_z 0.206)
- **No NaN:** Expected PASS (forward-fill after warmup)

**Expected:** PASS

### Gate 2: Information Gain
- **MI increase >5%:** High probability (individual MI = 0.082-0.089)
- **VIF <10:** High probability (measured corr <0.41 with base features)
- **Rolling corr std <0.15:** High probability (gold-specific dynamics)

**Expected:** PASS

### Gate 3: Ablation Test
- **Direction accuracy +0.5%:** Target (higher MI than VIX features)
- **OR Sharpe +0.05:** Secondary target
- **OR MAE -0.01%:** Tertiary target

**Confidence:** 7/10 (follows VIX success pattern, higher MI values)

## Critical Implementation Details

1. **NaN handling:** Forward-fill after warmup period (~60 days)
2. **Reproducibility:** Fixed random seeds for HMM (42-based)
3. **Zero-division protection:** Clip GK vol minimum to 1e-8
4. **Extreme outlier handling:** Clip z-scores to [-4, 4]
5. **HMM convergence:** 200 iterations, tol=1e-4

## Differences from Design Doc

| Design Doc | Actual Implementation | Reason |
|-----------|----------------------|--------|
| GaussianHMM(n_init=5) | Manual loop over random_state | hmmlearn has no n_init parameter |
| GC=F data source | GLD data source | GC=F has 137 flat-bar corruptions |
| 20d smoothed GK vol | Daily GK vol | Autocorr 0.979 → 0.206 (safety) |

All changes improve robustness and follow fact-check corrections.

## Next Steps (Orchestrator)

1. Submit to Kaggle: `kaggle kernels push -p notebooks/technical_1/`
2. Monitor status: `kaggle kernels status BigBigZabuton/gold-technical-1`
3. Fetch results: `kaggle kernels output BigBigZabuton/gold-technical-1 -p data/submodel_outputs/technical/`
4. Pass to evaluator for Gate 1/2/3 evaluation

## Notes

- **Self-contained design:** Zero external file dependencies
- **VIX v8 pattern:** Direct target computation eliminates dataset alignment issues
- **Fact-check corrections:** All 4 critical corrections from architect implemented
- **Syntax verified:** ast.parse passed without errors
