# Submodel Design Document: Regime Classification GMM (Attempt 1)

## 0. Fact-Check Results

| Claim / Source | Result | Detail |
|----------------|--------|--------|
| FRED: VIXCLS | CONFIRMED | 3 records retrieved. Earliest: 1990-01-02. Available daily, no delay. |
| FRED: DGS10 | CONFIRMED | 3 records retrieved. Earliest: 1962-01-02. Available daily. |
| FRED: DGS2 | CONFIRMED | 3 records retrieved. Earliest: 1976-06-01. Available daily. |
| FRED: T10YIE (optional) | CONFIRMED | 3 records retrieved. Earliest: 2003-01-02. Available daily. |
| Yahoo: ^GSPC | CONFIRMED | 5 records retrieved. Latest: 2026-02-17. |
| Yahoo: GC=F | CONFIRMED | 4 records retrieved. Latest: 2026-02-18. |
| sklearn.mixture.GaussianMixture API | CONFIRMED | All parameters verified: n_components, covariance_type='diag', n_init, reg_covar, random_state. predict_proba() returns probabilities summing to 1.0. BIC/AIC accessible. |
| Diagonal covariance safer than full | CONFIRMED | K=3, D=4: diag=26 params vs full=44 params (1.69x). Samples/param: diag=67.3 vs full=39.8. Both adequate but diag provides more margin. |
| Researcher param counts | CORRECTED | Researcher stated K=3,D=4 full = 42 params. Actual sklearn count = 44 (includes mixture weights). Researcher stated diag = 24. Actual = 26. Minor discrepancy due to weight parameters; does not change conclusions. |
| n_init=20 sufficient | CONFIRMED | Standard for financial data. Sklearn default is 1. 20 random restarts with kmeans init provide robust convergence. |
| fillna(method='bfill') | DEPRECATED | pandas 2.3.3 raises FutureWarning. Must use .bfill().ffill() instead. Researcher's code snippet needs correction. |
| Researcher correlation estimates | UNVERIFIABLE | Pairwise correlations (vix_z vs equity_return_z = -0.70, etc.) are plausible but stated as approximations. Will be computed empirically during training. |
| Researcher Gate pass probabilities | ACCEPTED WITH CAVEAT | Gate 1: 85-90% (reasonable, GMM has fewer failure modes). Gate 2: 30-40% (reasonable, VIF with existing 7 regime features is the key risk). Gate 3: 25-35% (conservative, consistent with project history). |

### Researcher Corrections Applied

1. **Parameter counts**: Researcher undercounted by 2 (missing mixture weights). Corrected to 26 (diag) and 44 (full) for K=3, D=4. No impact on design decisions.
2. **DXY exclusion**: Researcher recommended excluding dxy_momentum_z. I agree -- DXY is already well-represented in the meta-model (dxy_change base feature + 3 dxy submodel features). Keeping it would increase D to 5, raising param count and VIF risk with marginal benefit.
3. **pandas deprecation**: Researcher's smoothing code uses `fillna(method='bfill')` which is deprecated in pandas 2.x. Design specifies `.bfill().ffill()` instead.
4. **Smoothing window**: Researcher recommends 3-day rolling average with center=True. I change this to center=False (backward-looking only) to prevent future information leakage. The first 2 rows of smoothed output will use partial windows via min_periods=1.

---

## 1. Overview

- **Purpose**: Detect multi-dimensional macro regime states (Risk-Off, Risk-On, Calm) by jointly analyzing VIX, yield spread, equity returns, and gold realized volatility. This captures cross-asset synchronization patterns that individual single-feature HMMs cannot detect. The regime probability vector provides the meta-model with macro-state awareness, potentially improving direction accuracy during regime transitions.
- **Core method**: Gaussian Mixture Model (GMM) with K=2-4 components (selected by BIC via Optuna), diagonal covariance, fitted on D=4 z-scored input features.
- **Why GMM over HMM**: (1) Fewer failure modes -- yield_curve HMM collapsed to constant output. (2) With 1750 training samples and D=4, GMM has 67.3 samples/param (diag) vs HMM which adds transition matrix parameters. (3) XGBoost meta-model already learns temporal patterns via tree splits; HMM temporal structure is partially redundant. (4) GMM trains in seconds, enabling broader Optuna search.
- **Why this is different from existing regime features**: Existing 7 regime probabilities each detect regime in ONE asset's dynamics. This model detects regime in the JOINT distribution of 4 assets simultaneously. Example: vix_regime_prob=0.9 (high VIX) but regime_classification_prob_riskoff=0.3 (low) because yield curve is normal, equities are stable, and gold vol is low -- an isolated VIX spike, not a systemic risk-off event.
- **Expected effect**: Moderate probability (25-35%) of passing Gate 3, most likely via MAE improvement. This is a high-risk, moderate-reward feature.

---

## 2. Data Specification

### Primary Data Sources

| Data | Source | ID/Ticker | Frequency | Delay | Verified |
|------|--------|-----------|-----------|-------|----------|
| VIX | FRED | VIXCLS | Daily | 0 days | Yes |
| 10Y Treasury Yield | FRED | DGS10 | Daily | 0-1 days | Yes |
| 2Y Treasury Yield | FRED | DGS2 | Daily | 0-1 days | Yes |
| S&P 500 | Yahoo | ^GSPC | Daily | 0 days | Yes |
| Gold Futures | Yahoo | GC=F | Daily | 0 days | Yes |

### Feature Engineering

All input features are rolling z-scores to ensure stationarity and comparability:

| Feature | Formula | Lookback | Rationale |
|---------|---------|----------|-----------|
| vix_z | (VIX - rolling_mean) / rolling_std | 20d | VIX dynamics are fast-moving; 20d captures ~1 month of context |
| yield_spread_z | (spread - rolling_mean) / rolling_std, where spread = DGS10 - DGS2 | 60d | Yield curve dynamics are slow-moving; 60d captures ~3 months |
| equity_return_z | (spx_5d_ret - rolling_mean) / rolling_std | 60d | 5d return reduces daily noise; 60d normalization captures medium-term context |
| gold_rvol_z | (rvol_10d - rolling_mean) / rolling_std | 60d | 10d realized vol balances responsiveness vs noise; 60d normalization |

**Detailed computation**:

```python
# vix_z
vix_z = (vix - vix.rolling(20).mean()) / vix.rolling(20).std()

# yield_spread_z
spread = dgs10 - dgs2
yield_spread_z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()

# equity_return_z
spx_5d_ret = spx_close.pct_change(5)
equity_return_z = (spx_5d_ret - spx_5d_ret.rolling(60).mean()) / spx_5d_ret.rolling(60).std()

# gold_rvol_z
gold_log_ret = np.log(gold_close / gold_close.shift(1))
gold_rvol_10d = gold_log_ret.rolling(10).std() * np.sqrt(252)  # annualized
gold_rvol_z = (gold_rvol_10d - gold_rvol_10d.rolling(60).mean()) / gold_rvol_10d.rolling(60).std()
```

### Missing Data Handling

1. FRED data (VIXCLS, DGS10, DGS2): Forward-fill gaps up to 5 business days (weekends, holidays)
2. Yahoo data (^GSPC, GC=F): Forward-fill gaps up to 3 business days
3. Align all series on common trading dates (inner join on date index)
4. After computing z-scores, the first ~70 rows (60d rolling window + 10d rvol window) will be NaN
5. Drop all rows with any NaN in the 4 input features
6. For production output (full date range), impute remaining NaN with 0.0 (neutral z-score)

### Train/Val/Test Split

- Time-series order, no shuffle
- Train: 70% (~1750 samples)
- Val: 15% (~375 samples)
- Test: 15% (~375 samples)
- GMM is fit on training set ONLY
- Val set used for BIC comparison (Optuna objective)
- Test set used only for Gate 3 evaluation

### Expected Sample Count

- Raw data: ~2800 trading days (2014-01-01 to 2026-02-18)
- After z-score warmup (~70 rows dropped): ~2730
- After alignment with base_features (2015-01-30 to 2025-02-12): ~2523
- Output date range: matches other submodel outputs (2015-01-30 to latest available)

---

## 3. Model Architecture

### sklearn.mixture.GaussianMixture

This is NOT a PyTorch model. GMM is a classical statistical model implemented in scikit-learn. There is no neural network, no gradient descent, no GPU requirement.

**Base configuration** (defaults for Optuna to search around):

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,          # Optuna: 2-4
    covariance_type='diag',  # Optuna: ['diag', 'full']
    n_init=20,               # Fixed: 20 random restarts
    init_params='kmeans',    # Fixed: kmeans++ initialization
    max_iter=200,            # Fixed: sufficient for convergence
    tol=1e-4,                # Fixed: convergence threshold
    reg_covar=1e-5,          # Optuna: 1e-6 to 1e-3
    random_state=42,         # Fixed: reproducibility
)
```

**Fitting**:

```python
gmm.fit(X_train)  # X_train shape: [~1750, 4]
```

**Inference**:

```python
regime_probs = gmm.predict_proba(X_full)  # shape: [N, K]
# Each row sums to 1.0
```

### Parameter Count

| Config | K | D | Cov | Params | Samples/Param |
|--------|---|---|-----|--------|---------------|
| Primary | 3 | 4 | diag | 26 | 67.3 |
| Alt 1 | 2 | 4 | diag | 17 | 102.9 |
| Alt 2 | 4 | 4 | diag | 35 | 50.0 |
| Alt 3 | 3 | 4 | full | 44 | 39.8 |
| Alt 4 | 4 | 4 | full | NOT recommended | Too many params |

All configurations have samples/param > 20 (rule of thumb minimum), but diagonal covariance provides substantially more margin.

---

## 4. Hyperparameter Search

### Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_init | 20 | 20 random restarts ensure robust convergence; standard for financial data |
| init_params | 'kmeans' | K-means++ initialization is more stable than random |
| max_iter | 200 | Financial GMMs typically converge in 20-50 iterations; 200 provides ample margin |
| tol | 1e-4 | Stricter than default (1e-3) for precision; negligible computational cost |
| random_state | 42 | Reproducibility |
| Input features | 4 (vix_z, yield_spread_z, equity_return_z, gold_rvol_z) | Fixed by design; adding more increases VIF risk |

### Optuna Search Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| n_components | {2, 3, 4} | categorical | K=3 is prior from literature; K=2 (simpler) and K=4 (stagflation) are alternatives |
| covariance_type | {'diag', 'full'} | categorical | Diagonal is safer; full captures cross-feature correlations within regimes |
| reg_covar | [1e-6, 1e-3] | log-uniform | Regularization on covariance diagonal; prevents singular matrices |
| smoothing_window | {1, 3, 5, 7} | categorical | Post-hoc smoothing on regime probs; 1 = no smoothing |
| use_pca | {false, true} | categorical | PCA on 4 inputs before GMM; reduces collinearity (vix_z vs equity_return_z) |
| pca_components | {2, 3} | categorical (only if use_pca=true) | Number of principal components to retain |

### Optuna Objective

```python
def objective(trial):
    n_components = trial.suggest_categorical('n_components', [2, 3, 4])
    covariance_type = trial.suggest_categorical('covariance_type', ['diag', 'full'])
    reg_covar = trial.suggest_float('reg_covar', 1e-6, 1e-3, log=True)
    smoothing_window = trial.suggest_categorical('smoothing_window', [1, 3, 5, 7])
    use_pca = trial.suggest_categorical('use_pca', [False, True])

    if use_pca:
        pca_components = trial.suggest_categorical('pca_components', [2, 3])
        # Apply PCA to training data
        pca = PCA(n_components=pca_components)
        X_train_input = pca.fit_transform(X_train_scaled)
        X_val_input = pca.transform(X_val_scaled)
    else:
        X_train_input = X_train_scaled
        X_val_input = X_val_scaled

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=20,
        max_iter=200,
        tol=1e-4,
        random_state=42,
    )
    gmm.fit(X_train_input)

    # --- Quality checks (prune if degenerate) ---
    weights = gmm.weights_
    if np.min(weights) < 0.05:
        raise optuna.TrialPruned("Regime collapse: min weight < 5%")

    # Compute regime probs on validation set
    val_probs = gmm.predict_proba(X_val_input)

    # Apply smoothing (backward-looking)
    if smoothing_window > 1:
        val_probs_smooth = pd.DataFrame(val_probs).rolling(
            smoothing_window, min_periods=1
        ).mean().values
        # Re-normalize to sum to 1
        val_probs_smooth = val_probs_smooth / val_probs_smooth.sum(axis=1, keepdims=True)
    else:
        val_probs_smooth = val_probs

    # Check regime persistence on validation set
    dominant_regime = np.argmax(val_probs_smooth, axis=1)
    regime_changes = np.sum(np.diff(dominant_regime) != 0)
    avg_duration = len(dominant_regime) / max(regime_changes, 1)

    if avg_duration < 2 or avg_duration > 100:
        raise optuna.TrialPruned(f"Bad persistence: {avg_duration:.1f} days")

    # --- Objective: BIC on validation set (lower is better) ---
    # BIC penalizes model complexity, rewarding parsimony
    val_bic = gmm.bic(X_val_input)

    # Secondary: penalize if regime balance is poor
    min_weight_penalty = -100 * np.min(weights)  # reward higher min weight

    return val_bic + min_weight_penalty  # minimize
```

### Search Settings

- n_trials: 50
- timeout: 300 seconds (5 minutes)
- sampler: TPESampler(seed=42)
- pruner: MedianPruner (via TrialPruned for degenerate solutions)

**Rationale for 50 trials**: GMM fits in <3 seconds per trial (with n_init=20). 50 trials x 3 seconds = 2.5 minutes. Well within Kaggle time limits. The search space has 6 parameters with limited categorical ranges, so 50 trials provides good coverage.

---

## 5. Post-Processing Pipeline

### Step 1: Compute raw regime probabilities

```python
# Fit GMM on training data with best params from Optuna
gmm_best = GaussianMixture(**best_params_gmm)
gmm_best.fit(X_train_best)

# Predict on full dataset
regime_probs_raw = gmm_best.predict_proba(X_full_best)  # [N, K]
```

### Step 2: Smooth regime probabilities (backward-looking)

```python
if best_smoothing_window > 1:
    regime_probs_smooth = pd.DataFrame(regime_probs_raw).rolling(
        best_smoothing_window, min_periods=1
    ).mean().values
    # Re-normalize rows to sum to 1.0
    regime_probs_smooth = regime_probs_smooth / regime_probs_smooth.sum(axis=1, keepdims=True)
else:
    regime_probs_smooth = regime_probs_raw
```

**CRITICAL**: Use `center=False` (default for pandas rolling) to prevent future information leakage. The researcher's code used `center=True`, which is corrected here.

### Step 3: Compute transition velocity

```python
# Euclidean distance between consecutive regime probability vectors
diff = np.diff(regime_probs_smooth, axis=0)  # [N-1, K]
velocity = np.sqrt(np.sum(diff ** 2, axis=1))  # [N-1]
velocity = np.concatenate([[0.0], velocity])    # [N], prepend 0 for first day
```

**Interpretation**: High velocity = regime is transitioning rapidly (unstable). Low velocity = regime is stable. Range: [0, sqrt(2)] for K=2, [0, sqrt(2)] for K=3 (maximum when jumping from one regime to another with probability 1).

### Step 4: Regime labeling (post-hoc, for interpretability only)

After fitting, examine the component means to assign economic labels:

```python
means = gmm_best.means_  # [K, D] where D=4 (vix_z, yield_spread_z, equity_return_z, gold_rvol_z)
for k in range(K):
    print(f"Regime {k}: vix_z={means[k,0]:.2f}, yield_spread_z={means[k,1]:.2f}, "
          f"equity_return_z={means[k,2]:.2f}, gold_rvol_z={means[k,3]:.2f}")
# Expected pattern:
# Risk-Off:  high vix_z (+), negative equity_return_z (-), high gold_rvol_z (+)
# Risk-On:   low vix_z (-), positive equity_return_z (+), low gold_rvol_z (-)
# Calm:      near-zero across all dimensions
```

Labels are assigned post-hoc in training_result.json for interpretability. The meta-model does not use labels; it uses the raw probabilities.

### Step 5: Collapse detection

```python
# Check if any regime captures < 5% of observations
regime_assignments = np.argmax(regime_probs_smooth, axis=1)
for k in range(K):
    pct = np.mean(regime_assignments == k) * 100
    if pct < 5.0:
        print(f"WARNING: Regime {k} captures only {pct:.1f}% of observations (degenerate)")
```

---

## 6. Output Format

### Columns

For K=3 (primary):
- `Date`: Date index
- `regime_prob_0`: Probability of Regime 0 (labeled post-hoc)
- `regime_prob_1`: Probability of Regime 1 (labeled post-hoc)
- `regime_prob_2`: Probability of Regime 2 (labeled post-hoc)
- `regime_transition_velocity`: Rate of change in regime probability vector

For K=2: Drop `regime_prob_2` (output 3 columns total: 2 probs + velocity)
For K=4: Add `regime_prob_3` (output 5 columns total: 4 probs + velocity)

### Properties

| Property | Requirement |
|----------|-------------|
| Probability constraint | regime_prob_0 + regime_prob_1 + ... = 1.0 per row (within 1e-6 tolerance) |
| Range | Each probability in [0, 1] |
| Velocity range | [0, sqrt(2)] approximately |
| No NaN | Zero NaN in output |
| Smoothness | After smoothing, average regime duration should be 5-30 trading days |
| Date range | Must match base_features range (2015-01-30 to latest available) |

### Save Paths

- **Kaggle notebook output**: `submodel_output.csv` (Kaggle output directory)
- **Local integration**: `data/submodel_outputs/regime_classification.csv`
- **Training metadata**: `training_result.json` (Kaggle output directory)

**IMPORTANT**: The Kaggle notebook saves as `submodel_output.csv` (standard name for all submodels). The orchestrator renames this to `regime_classification.csv` when uploading to the Kaggle dataset.

---

## 7. Kaggle Notebook Requirements

### Self-Contained Structure

The notebook must be fully self-contained with no external dependencies beyond pip-installable packages:

```
1. Libraries: sklearn, pandas, numpy, optuna, json, os, datetime
   - fredapi (for FRED data inside Kaggle)
   - yfinance (for Yahoo data inside Kaggle)
2. Data fetching: FRED API (VIXCLS, DGS10, DGS2) + yfinance (^GSPC, GC=F)
3. Feature engineering: rolling z-scores
4. Optuna HPO: 50 trials
5. Best model training: GMM with best params
6. Post-processing: smoothing + velocity
7. Save: submodel_output.csv + training_result.json
```

### FRED API Key

```python
import os
FRED_API_KEY = os.environ['FRED_API_KEY']  # From Kaggle Secrets
# Fail immediately with KeyError if not set
```

### Dataset Reference

kernel-metadata.json MUST include:

```json
{
  "dataset_sources": ["bigbigzabuton/gold-prediction-submodels"]
}
```

This dataset is mounted at `/kaggle/input/gold-prediction-submodels/` (or alternative path `/kaggle/input/datasets/bigbigzabuton/gold-prediction-submodels/`). The notebook should check both paths.

### Output Files

```python
# Save regime classification output
output_df.to_csv("submodel_output.csv", index=False)

# Save training metadata
with open("training_result.json", "w") as f:
    json.dump({
        "feature": "regime_classification",
        "attempt": 1,
        "timestamp": datetime.now().isoformat(),
        "best_params": best_params,
        "metrics": {
            "train_bic": float(gmm.bic(X_train)),
            "val_bic": float(gmm.bic(X_val)),
            "train_log_likelihood": float(gmm.score(X_train)),
            "val_log_likelihood": float(gmm.score(X_val)),
            "n_components": int(gmm.n_components),
            "covariance_type": gmm.covariance_type,
            "component_weights": gmm.weights_.tolist(),
            "component_means": gmm.means_.tolist(),
            "regime_balance": regime_balance_dict,
            "avg_regime_duration_days": float(avg_duration),
            "regime_labels": regime_labels,
        },
        "output_shape": list(output_df.shape),
        "output_columns": list(output_df.columns),
        "optuna_n_trials": n_completed_trials,
        "optuna_best_value": float(study.best_value),
    }, f, indent=2)
```

---

## 8. Gate Criteria (Specific to This Submodel)

### Gate 1: Standalone Quality

| Check | Criterion | Expected | Risk |
|-------|-----------|----------|------|
| No constant output | std > 0.01 for each regime_prob column | PASS | Low (20 restarts + collapse check) |
| No all-NaN | Zero NaN in output | PASS | None (deterministic computation) |
| Regime balance | No regime > 80% of days | PASS (85-90%) | 15-25% risk of collapse |
| Regime persistence | Average duration 5-30 trading days | PASS | Low (K=3 typically 10-40 days) |
| Overfit ratio | train_log_likelihood / val_log_likelihood < 1.5 | PASS | Low (GMM generalizes well with diagonal cov) |

### Gate 2: Information Gain

| Check | Criterion | Expected | Risk |
|-------|-----------|----------|------|
| MI increase | > 5% total MI increase | Borderline (3-8%) | 40-50% fail |
| VIF | < 10 for each new column | Expected 5-7 | 30-40% fail |
| Stability | Rolling correlation std < 0.15 | Expected 0.10-0.14 | Low |

**Key risk**: VIF overlap with existing regime probabilities (vix_regime_probability, xasset_regime_prob, tech_trend_regime_prob, etc.). The multi-dimensional regime may correlate r=0.5-0.7 with vix_regime_probability.

### Gate 3: Ablation

| Metric | Threshold | Expected | Most Likely |
|--------|-----------|----------|-------------|
| Direction Accuracy | +0.5% | +0.3% to +0.6% | Marginal |
| Sharpe | +0.05 | -0.1 to +0.15 | Uncertain |
| MAE | -0.01% | -0.02 to -0.05% | **Most likely pass criterion** |

**Rationale for MAE as most likely pass**: Regime features help the meta-model distinguish high-volatility periods (predict smaller magnitude) from low-volatility periods (predict larger magnitude). This pattern is consistent with how technical, cross_asset, cny_demand, and options_market all passed Gate 3 via MAE.

---

## 9. Kaggle Execution Settings

- **enable_gpu**: false
- **Estimated execution time**: 3-5 minutes
- **Estimated memory**: < 2 GB
- **Required pip packages**: optuna, fredapi, yfinance (sklearn, pandas, numpy are pre-installed on Kaggle)

**Rationale for no GPU**: GMM is a CPU-only algorithm (sklearn does not use GPU). Optuna with 50 trials x ~3 seconds each = ~2.5 minutes. Data fetching adds ~1-2 minutes. Total well under 10 minutes.

---

## 10. Implementation Instructions

### For builder_data

This submodel fetches data INSIDE the Kaggle notebook (no separate data preparation step). However, builder_data should verify:

1. FRED VIXCLS is accessible with FRED_API_KEY
2. FRED DGS10, DGS2 are accessible
3. Yahoo ^GSPC, GC=F are downloadable
4. Date range coverage: 2014-01-01 to present (need buffer for rolling windows)

### For builder_model

**Notebook structure** (single self-contained train.ipynb):

```
Cell 1: Imports and configuration
Cell 2: Data fetching (FRED + yfinance)
Cell 3: Feature engineering (4 z-scores)
Cell 4: Data quality checks + pairwise correlation matrix
Cell 5: Train/val/test split
Cell 6: Optuna HPO (50 trials, BIC objective)
Cell 7: Best model training + post-processing
Cell 8: Regime labeling (post-hoc) + quality checks
Cell 9: Save output CSV + training_result.json
```

**Critical implementation details**:

1. **Z-score division by zero**: If rolling_std == 0 (constant values in window), set z-score to 0.0. Add explicit check.
2. **FRED NaN handling**: FRED returns NaN for holidays. Forward-fill before computing z-scores.
3. **Date alignment**: Inner join all 5 raw series on date index before computing features.
4. **Smoothing re-normalization**: After rolling mean smoothing, probabilities may not sum to 1.0. Explicitly re-normalize each row.
5. **Optuna pruning**: Prune trials where min(weights) < 0.05 or avg_duration outside [2, 100] range.
6. **BIC computation on validation set**: sklearn's `gmm.bic(X_val)` is the correct API. It uses X_val sample count for the penalty term.
7. **PCA inside Optuna**: If use_pca=True, fit PCA on training data only, transform both train and val with the same PCA.
8. **Output date column**: Must be a string in 'YYYY-MM-DD' format (no timezone info). Remove any timezone from Yahoo/FRED dates.
9. **Kaggle dataset paths**: Check both `/kaggle/input/gold-prediction-submodels/` and `/kaggle/input/datasets/bigbigzabuton/gold-prediction-submodels/` for existing submodel data (needed for validation alignment, not for training inputs).

### kernel-metadata.json

```json
{
  "id": "bigbigzabuton/gold-regime-classification-1",
  "title": "Gold Regime Classification - Attempt 1",
  "code_file": "train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": true,
  "dataset_sources": ["bigbigzabuton/gold-prediction-submodels"],
  "competition_sources": [],
  "kernel_sources": []
}
```

---

## 11. Risk Assessment and Mitigation

### Risk 1: VIF Overlap with Existing Regime Features (30-40% probability)

**Description**: regime_prob_0 (Risk-Off) may correlate r=0.6-0.8 with vix_regime_probability, leading to VIF > 10.

**Detection**: Gate 2 VIF check.

**Mitigation**:
- Attempt 1 uses RAW market data inputs (not submodel outputs), preventing circular dependencies
- If VIF > 10 for one column, attempt 2 drops the most correlated regime_prob column and outputs only the remaining K-1 probs + velocity
- If VIF > 10 for all columns, attempt 2 uses residualization: regress each regime_prob on existing regime features, output residuals

### Risk 2: Regime Collapse (15-25% probability)

**Description**: One GMM component captures >80% of observations. This happened with yield_curve HMM.

**Detection**: Optuna pruning (min weight < 5%) + Gate 1 regime balance check.

**Mitigation**:
- n_init=20 (multiple restarts reduce local minimum risk)
- reg_covar > 0 (prevents singular covariance)
- Optuna searches K={2,3,4}; if K=3 collapses, K=2 may not
- If collapse persists, attempt 2 uses Bayesian GMM (BayesianGaussianMixture) with Dirichlet prior

### Risk 3: Too Many Output Columns (20-30% probability)

**Description**: Adding 4 columns (K=3: 3 probs + velocity) to the 24-feature meta-model. Meta-model attempts 8-12 all regressed when features were added.

**Detection**: Gate 3 ablation.

**Mitigation**:
- Output column count depends on K: K=2 produces 3 columns, K=3 produces 4, K=4 produces 5
- Optuna's BIC objective naturally favors lower K (BIC penalizes complexity)
- If Gate 3 fails with K=3 (4 columns), attempt 2 reduces to 2-column output (most informative regime_prob + velocity)
- Pattern from options_market: attempt 1 (3 columns) failed Gate 3, attempt 2 (1 column) passed

### Risk 4: Non-Stationarity (25-35% probability)

**Description**: Regime structure may shift over the 2015-2025 period (COVID, inflation surge, gold rally).

**Detection**: Qualitative check of test-period regime assignments in training_result.json.

**Mitigation**:
- Static fitting on full training set (most robust with 1750 samples)
- GMM with diagonal covariance is more robust to structural breaks than full covariance
- If test-period regimes are nonsensical, attempt 2 uses expanded training window (including validation data) or Bayesian GMM with informative priors

### Risk 5: Pandas Compatibility (5% probability)

**Description**: Kaggle's pandas version may differ from local. Deprecated APIs may cause errors.

**Mitigation**:
- Use `.bfill().ffill()` instead of `fillna(method='bfill')`
- Avoid `.append()` (deprecated); use `pd.concat()` instead
- Test notebook locally before submission

---

## 12. Expected Outcome Summary

| Scenario | Probability | Gate Result | Description |
|----------|-------------|-------------|-------------|
| Full success | 20-30% | Gate 1+2+3 PASS | Multi-dim regime captures unique joint patterns, MAE improves |
| Partial success | 10-15% | Gate 1+2 PASS, Gate 3 FAIL | Information exists but not actionable for meta-model |
| VIF failure | 30-40% | Gate 1 PASS, Gate 2 FAIL (VIF) | Regime probs too correlated with existing features |
| MI failure | 15-20% | Gate 1 PASS, Gate 2 FAIL (MI) | Existing 7 regime features already capture most info |
| Collapse | 10-15% | Gate 1 FAIL | Degenerate GMM solution |

**Decision rules**:
- If attempt 1 fails Gate 2 (VIF): Attempt 2 with reduced output dimensionality
- If attempt 1 fails Gate 2 (MI): Consider abandoning (existing features are sufficient)
- If attempt 1 fails Gate 3: Attempt 2 with HMM (temporal smoothing) or K=4 (finer granularity)
- If attempt 1 passes: Accept and proceed
- Max attempts: 2 (if both fail, accept that existing 7 regime features capture most macro regime information)

---

**End of Design Document**
