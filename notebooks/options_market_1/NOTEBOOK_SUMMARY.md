# Options Market Submodel - Kaggle Notebook Summary

## Generation Info

- **Feature**: options_market
- **Attempt**: 1
- **Generated**: 2026-02-15
- **Agent**: builder_model (Sonnet)
- **Notebook ID**: bigbigzabuton/gold-model-training

## Architecture

### Component 1: 2D HMM Regime Detection
- **Input**: [SKEW daily changes, GVZ daily changes]
- **Model**: hmmlearn.hmm.GaussianHMM
- **States**: {2, 3} (Optuna selects)
- **Covariance**: "full"
- **Output**: `options_risk_regime_prob` [0, 1]

### Component 2: SKEW Tail Risk Z-Score
- **Input**: SKEW closing levels
- **Window**: {40, 60, 90} days (Optuna selects)
- **Output**: `options_tail_risk_z` [-4, +4] (clipped)

### Component 3: SKEW Momentum Z-Score
- **Input**: SKEW closing levels
- **Momentum Window**: {5, 10, 15} days (Optuna selects)
- **Z-Score Window**: 60 days (fixed)
- **Output**: `options_skew_momentum_z` [-4, +4] (clipped)

## Hyperparameter Optimization

- **Algorithm**: Optuna with TPESampler
- **Objective**: Maximize MI sum of 3 outputs vs validation target
- **Trials**: 30
- **Timeout**: 300 seconds (5 minutes)
- **Random Seed**: 42

### Search Space

| Parameter | Range | Type |
|-----------|-------|------|
| hmm_n_components | {2, 3} | categorical |
| hmm_n_init | {3, 5, 10} | categorical |
| skew_zscore_window | {40, 60, 90} | categorical |
| skew_momentum_window | {5, 10, 15} | categorical |

## Data Sources

- **SKEW**: Yahoo Finance (^SKEW)
- **GVZ**: FRED (GVZCLS) with Yahoo (^GVZ) fallback
- **Date Range**: 2014-10-01 to 2026-02-15 (includes warmup buffer)
- **Expected Samples**: ~2,800 rows

## Kaggle Configuration

- **GPU**: false (CPU-only, no neural network)
- **Internet**: true (for API calls)
- **Estimated Runtime**: 3-5 minutes
- **Dependencies**: hmmlearn (auto-installed via pip)

## Output Files

1. **submodel_output.csv**
   - Columns: options_risk_regime_prob, options_tail_risk_z, options_skew_momentum_z
   - Index: date
   - Format: CSV with header

2. **training_result.json**
   - best_params (Optuna result)
   - optuna_trials_completed
   - optuna_best_value
   - output_shape, output_columns
   - data_info (train/val/test split info)
   - autocorrelation_metrics
   - output_statistics

## Design Rationale

This submodel captures options-derived risk context that complements the VIX submodel:

| Dimension | VIX Submodel | Options Market Submodel |
|-----------|-------------|------------------------|
| What it captures | Volatility LEVEL (how much uncertainty) | Tail risk SHAPE (how asymmetric the risk) |
| HMM input | 1D log-VIX changes | 2D [SKEW changes, GVZ changes] |
| Information type | "How scared is the market?" | "How much tail risk is priced in?" |

## Key Implementation Notes

1. **Self-Contained**: All code embedded in notebook (no external dependencies)
2. **API Key**: FRED_API_KEY from Kaggle Secrets (must be configured in browser)
3. **No Lookahead Bias**:
   - HMM: fit on training data only, predict_proba on full dataset
   - Z-score: rolling window (inherently backward-looking)
   - Momentum: rolling window change (inherently backward-looking)
4. **Reproducibility**: Fixed random seeds (HMM: 42, Optuna: 42)
5. **NaN Handling**: Forward/backward fill after feature generation

## Expected Performance

### Gate 1 (Standalone Quality)
- **Overfit ratio**: N/A (no neural network)
- **Autocorrelation < 0.99**: Expected 0.7-0.95 (PASS)
- **Constant output**: PASS (SKEW/GVZ have meaningful variation)

### Gate 2 (Information Gain)
- **MI increase > 5%**: UNCERTAIN (highest-risk gate)
- **VIF < 10**: HIGH PROBABILITY (measured correlations with VIX submodel < 0.27)
- **Rolling correlation std < 0.15**: HIGH PROBABILITY

### Gate 3 (Ablation)
- **Overall Pass Probability**: 5/10 (moderate confidence)
- **Risk Factor**: No P/C ratio, weak raw correlations (<0.06)
- **Rationale for Attempt**: HMM may capture nonlinear patterns invisible to raw correlation

## Next Steps

1. builder_model commits notebook to git
2. Orchestrator submits to Kaggle: `kaggle kernels push -p notebooks/options_market_1/`
3. Kaggle training executes (3-5 minutes)
4. Orchestrator fetches results
5. Evaluator runs Gates 1, 2, 3
