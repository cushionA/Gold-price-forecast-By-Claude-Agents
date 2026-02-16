# Meta-Model Requirements: Attempt 1

## 1. Objective

Build a regression meta-model that integrates 19 base features and 20 submodel output columns (from 7 completed submodels) to predict next-day gold return (%). The meta-model must simultaneously achieve all four final targets:

| Metric | Target | Baseline (XGBoost, base features only) | Gap |
|--------|--------|---------------------------------------|-----|
| Direction Accuracy (DA) | > 56% | 43.54% | +12.46pp |
| High-Confidence DA (HC-DA) | > 60% | 42.74% | +17.26pp |
| MAE | < 0.75% | 0.7139% | Already met by baseline |
| Sharpe Ratio (after 5bps costs) | > 0.8 | -1.696 | +2.496 |

**Critical observation**: The baseline already meets the MAE target (0.7139 < 0.75). The primary challenge is DA, HC-DA, and Sharpe. Several submodels (technical, yield_curve, cross_asset, cny_demand, etf_flow) improved MAE but degraded Sharpe or DA individually. The meta-model must find the combination that improves directional accuracy and risk-adjusted returns without sacrificing MAE.

## 2. Input Data Inventory

### 2.1 Base Features (19 columns)

Source: `data/processed/base_features.csv` (2523 rows, 2015-01-30 to 2025-02-12)

| # | Column | Category | Note |
|---|--------|----------|------|
| 1 | real_rate_real_rate | Macro | Strongest neg. correlation with gold. No submodel succeeded. |
| 2 | dxy_dxy | FX | USD inverse correlation. No submodel output (auto-evaluated). |
| 3 | vix_vix | Volatility | Risk-off indicator. |
| 4 | technical_gld_open | Technical | GLD price level. |
| 5 | technical_gld_high | Technical | GLD intraday high. |
| 6 | technical_gld_low | Technical | GLD intraday low. |
| 7 | technical_gld_close | Technical | GLD closing price. |
| 8 | technical_gld_volume | Technical | GLD trading volume. |
| 9 | cross_asset_silver_close | Cross-Asset | Silver price. |
| 10 | cross_asset_copper_close | Cross-Asset | Copper price (economic indicator). |
| 11 | cross_asset_sp500_close | Cross-Asset | S&P 500 level. |
| 12 | yield_curve_dgs10 | Rates | 10Y Treasury yield. |
| 13 | yield_curve_dgs2 | Rates | 2Y Treasury yield. |
| 14 | yield_curve_yield_spread | Rates | 10Y-2Y spread. |
| 15 | etf_flow_gld_volume | ETF | GLD volume (demand proxy). |
| 16 | etf_flow_gld_close | ETF | GLD NAV close. |
| 17 | etf_flow_volume_ma20 | ETF | 20-day volume moving average. |
| 18 | inflation_expectation_inflation_expectation | Macro | 10Y Breakeven Inflation Rate. |
| 19 | cny_demand_cny_usd | FX | CNY/USD exchange rate. |

### 2.2 Submodel Outputs (20 columns from 7 submodels)

| # | Submodel | Column | Type | Gate 3 Role | Quality |
|---|----------|--------|------|-------------|---------|
| 1 | vix | vix_regime_probability | HMM regime prob [0,1] | DA +0.96%, Sharpe +0.289 | Strong |
| 2 | vix | vix_mean_reversion_z | z-score | DA +0.96%, Sharpe +0.289 | Strong |
| 3 | vix | vix_persistence | continuous state | DA +0.96%, Sharpe +0.289 | Strong |
| 4 | technical | tech_trend_regime_prob | HMM regime prob [0,1] | MAE -0.182 (18x) | Strong |
| 5 | technical | tech_mean_reversion_z | z-score | MAE -0.182 (18x) | Strong |
| 6 | technical | tech_volatility_regime | regime state | MAE -0.182 (18x) | Strong |
| 7 | cross_asset | xasset_regime_prob | HMM regime prob [0,1] | DA +0.76%, MAE -0.087 | Strong |
| 8 | cross_asset | xasset_recession_signal | binary signal | DA +0.76%, MAE -0.087 | Strong |
| 9 | cross_asset | xasset_divergence | continuous | DA +0.76%, MAE -0.087 | Strong |
| 10 | yield_curve | yc_regime_prob | HMM regime prob | EXCLUDED (constant) | Excluded |
| 11 | yield_curve | yc_spread_velocity_z | z-score | MAE -0.069 (6.9x) | Moderate |
| 12 | yield_curve | yc_curvature_z | z-score | MAE -0.069 (6.9x) | Moderate |
| 13 | etf_flow | etf_regime_prob | HMM regime prob [0,1] | Sharpe +0.377, MAE -0.044 | Strong (VIF warning) |
| 14 | etf_flow | etf_capital_intensity | z-score | Sharpe +0.377, MAE -0.044 | Strong |
| 15 | etf_flow | etf_pv_divergence | z-score | Sharpe +0.377, MAE -0.044 | Strong |
| 16 | inflation_expectation | ie_regime_prob | HMM regime prob [0,1] | DA +0.57%, Sharpe +0.152 | Strong |
| 17 | inflation_expectation | ie_anchoring_z | z-score | DA +0.57%, Sharpe +0.152 | Strong |
| 18 | inflation_expectation | ie_gold_sensitivity_z | z-score | DA +0.57%, Sharpe +0.152 | Strong |
| 19 | cny_demand | cny_regime_prob | HMM regime prob [0,1] | MAE -0.066 | Weak (DA/Sharpe degrade) |
| 20 | cny_demand | cny_momentum_z | z-score | MAE -0.066 | Weak (DA/Sharpe degrade) |
| 21 | cny_demand | cny_vol_regime_z | z-score | MAE -0.066 | Weak (DA/Sharpe degrade) |

**Usable columns**: 20 (yc_regime_prob excluded due to constant value from collapsed HMM).

**Total feature count**: 19 (base) + 20 (submodel) = 39 features.

### 2.3 Feature Type Taxonomy

The 39 features are heterogeneous and fall into distinct types that require different handling:

| Type | Count | Examples | Characteristics |
|------|-------|---------|-----------------|
| Raw price levels | 8 | gld_open, silver_close, sp500_close | Non-stationary, trend-dominated |
| Interest rates | 4 | dgs10, dgs2, real_rate, yield_spread | Slowly varying, bounded |
| Volume/flow | 3 | gld_volume, volume_ma20, capital_intensity | Fat-tailed, heteroskedastic |
| Exchange rates | 2 | dxy, cny_usd | Slowly varying, mean-reverting |
| Volatility indices | 1 | vix | Positively skewed, clustered |
| Inflation expectations | 1 | inflation_expectation | Slowly varying |
| HMM regime probs | 6 | vix_regime_probability, tech_trend_regime_prob, etc. | Bounded [0,1], often bimodal |
| Z-scores | 9 | vix_mean_reversion_z, tech_mean_reversion_z, etc. | Approx. standard normal, may be autocorrelated |
| Binary/categorical | 1 | xasset_recession_signal | 0/1 or small integer |
| Continuous states | 4 | vix_persistence, tech_volatility_regime, xasset_divergence, cny_vol_regime_z | Misc continuous |

### 2.4 Data Alignment

- Base features: 2523 rows (2015-01-30 to 2025-02-12)
- Submodel outputs have varying date ranges (some start 2014, some extend to 2026)
- Merge strategy: inner join on date index, aligned to base features date range
- Expected merged rows: approximately 2520 (minimal loss from alignment)
- Target: `data/processed/target.csv` (gold_return_next column)

### 2.5 Data Split (frozen from Phase 1)

| Split | Rows | Date Range |
|-------|------|-----------|
| Train | 1766 | 2015-01-30 to 2022-02-07 |
| Val | 378 | 2022-02-08 to 2023-08-10 |
| Test | 379 | 2023-08-11 to 2025-02-12 |

Note: Test period (Aug 2023 - Feb 2025) includes the 2024 gold rally. This is a challenging out-of-sample period.

## 3. Known Issues and Warnings

### 3.1 cny_demand Quality Concern

- Gate 3 passed only via MAE (-0.066), while DA degraded -2.06% and Sharpe degraded -0.593 (worst of any passing submodel)
- The CNY features may attenuate prediction magnitudes while adding directional noise
- cny_momentum_z ranked #4 in feature importance (5.57%) during Gate 3 testing, so it carries signal
- **Recommendation**: Researcher should investigate whether including/excluding cny_demand features improves or hurts the combined model. The meta-model should be tested with and without these 3 features. Partial inclusion (cny_momentum_z only) is also a candidate strategy.

### 3.2 etf_flow VIF Warning

- etf_regime_prob has VIF=12.47, exceeding the 10 threshold
- Likely collinear with other HMM regime probability features (vix, technical, cross_asset)
- Despite VIF, Gate 3 showed the strongest Sharpe improvement of any submodel (+0.377)
- **Recommendation**: Regularization or feature selection should address this. Tree-based models handle collinearity naturally but neural network approaches may need explicit treatment.

### 3.3 Submodel Complementarity Pattern

Individual submodel Gate 3 results reveal complementary strengths:

| Submodel | DA | Sharpe | MAE | Primary Contribution |
|----------|-----|--------|------|---------------------|
| vix | ++ | ++ | - | Direction + risk-adjusted returns |
| technical | 0 | - | +++ | Magnitude calibration |
| cross_asset | + | 0 | ++ | Direction + magnitude |
| yield_curve | 0 | - | + | Magnitude refinement |
| etf_flow | 0 | +++ | + | Risk-adjusted returns + magnitude |
| inflation_expectation | + | + | - | Direction + risk-adjusted returns |
| cny_demand | --- | --- | + | Magnitude only (direction noise) |

The DA/Sharpe-improving submodels (vix, inflation_expectation, etf_flow) and the MAE-improving submodels (technical, cross_asset, yield_curve) are complementary. The central challenge is combining them without the negatives canceling the positives.

### 3.4 Baseline Overfitting Pattern

The Phase 1 XGBoost baseline shows clear overfitting:
- Train DA: 54.5% vs Test DA: 43.5% (11pp drop)
- Train Sharpe: 2.18 vs Test Sharpe: -1.70 (sign flip)
- Train MAE: 0.64 vs Test MAE: 0.71 (modest increase)

The meta-model must incorporate strong regularization to avoid this pattern with 39 features.

## 4. Architecture Constraints

### 4.1 Input Dimensionality

- 39 features is moderate -- manageable for most architectures
- However, feature heterogeneity (price levels, regime probs, z-scores) demands careful normalization
- Tree-based models (XGBoost, LightGBM) handle heterogeneous features and collinearity naturally
- Neural approaches (MLP, TabNet, FT-Transformer) require feature-type-aware preprocessing

### 4.2 Sample Size

- Train: 1766 samples, Val: 378, Test: 379
- With 39 features, the feature-to-sample ratio is ~1:45 for training
- This is adequate for tree-based models but may constrain deep neural architectures
- Dropout and weight decay are essential for neural approaches

### 4.3 Time-Series Structure

- Strict temporal ordering: no shuffle, no future leakage
- Walk-forward validation is the appropriate paradigm
- Regime probabilities from HMMs already encode temporal state -- the meta-model does not need to explicitly model time
- However, if the meta-model uses sequential architecture (GRU/LSTM), it could capture regime transitions

### 4.4 Multi-Objective Challenge

All four targets must be met simultaneously. This is fundamentally different from optimizing a single loss:

- DA and HC-DA require good directional prediction (sign of return)
- MAE requires good magnitude prediction
- Sharpe requires profitable trading after costs (directional + magnitude + risk management)
- These objectives can conflict: a model that improves DA by being more decisive may worsen MAE

### 4.5 Transaction Cost Handling

- 5bps per trade (one-way) deducted from strategy returns
- The model must generate predictions with enough magnitude to overcome this cost barrier
- Predictions close to zero should ideally be classified as "no trade" for HC-DA

## 5. Research Questions for Researcher

### 5.1 Architecture Selection

1. **What are the strengths and weaknesses of candidate meta-model architectures for integrating 39 heterogeneous features (raw prices, HMM regime probabilities, z-scores) with 1766 training samples?**
   - Candidates: XGBoost/LightGBM, MLP, TabNet, FT-Transformer, stacking ensemble
   - Key consideration: The baseline XGBoost (base features only) already achieves MAE < 0.75 but badly fails DA and Sharpe. Is tree-based still the right paradigm?

2. **Should the meta-model use a single unified architecture or a stacking/blending ensemble?**
   - Option A: Single model (XGBoost or neural) on all 39 features
   - Option B: Two-level stack: Level-1 diverse models on subsets, Level-2 blender
   - Option C: Feature-group-aware architecture (separate processing for base vs submodel features, then fusion)
   - Evaluate trade-offs in complexity, overfitting risk, and interpretability

3. **Is there evidence that feature-group-aware architectures (e.g., separate embeddings for regime probs vs z-scores vs raw features) outperform flat input approaches for this type of heterogeneous input?**

### 5.2 Loss Function and Multi-Objective Optimization

4. **What loss function or training objective best balances DA, MAE, and Sharpe simultaneously?**
   - Standard MSE/MAE loss optimizes magnitude but not direction
   - Directional loss (e.g., sign-aware loss, Qini-style) may improve DA
   - Sharpe-based differentiable loss functions exist in portfolio optimization literature
   - Composite loss with learned weights?

5. **How should the high-confidence direction accuracy (HC-DA) target influence model design?**
   - HC-DA > 60% means the model must identify a subset of predictions where it is confident and correct 60%+ of the time
   - This may require calibrated uncertainty or explicit confidence scoring
   - Threshold selection for "high confidence" is a design choice

### 5.3 Feature Engineering and Selection

6. **What feature selection strategy is optimal given the mixed quality of submodel outputs?**
   - cny_demand features degrade DA/Sharpe individually -- should they be included?
   - etf_regime_prob VIF=12.47 -- how to handle?
   - Options: include all 39, drop cny_demand (36), drop cny_demand + yc_regime_prob (already excluded) + etf_regime_prob (35), use Optuna feature selection
   - Should the researcher test forward/backward feature selection or rely on model-internal regularization?

7. **Should raw price-level base features be transformed before meta-model input?**
   - Current base features include price levels (gld_open, silver_close, sp500_close) which are non-stationary
   - Tree-based models are invariant to monotonic transforms, but neural models benefit from normalization
   - Options: keep raw (for tree), convert to returns/log-returns (for neural), or z-score standardize

### 5.4 Regularization and Overfitting Prevention

8. **What regularization strategies are most effective for 39-feature meta-models on 1766 time-series samples?**
   - The baseline XGBoost overfits severely (train DA 54.5% vs test DA 43.5%)
   - With 20 additional features, overfitting risk is higher
   - Specific strategies: early stopping, dropout, L1/L2, subsampling, feature sampling, noise injection

9. **Is temporal cross-validation (expanding/sliding window) more appropriate than a single train/val/test split for meta-model HP optimization?**
   - The Phase 2 submodels used 5-fold time-series CV for Gate 3
   - Should the meta-model HP search use similar walk-forward CV?
   - Trade-off: more robust HP selection vs computational cost

### 5.5 Benchmark and Practical Considerations

10. **What is a realistic range of achievable DA improvement from adding 20 submodel features to a 19-feature baseline?**
    - The individual submodel ablation results showed DA gains of 0.05% to 0.96% each
    - Non-linear interactions may yield super-additive gains, or interference may yield sub-additive
    - Academic benchmarks for next-day commodity return DA typically range 52-58%

11. **Are there published examples of HMM regime features being combined in a meta-model for commodity prediction? What architectures were used?**

## 6. Constraints

1. **Data split**: train/val/test = 70/15/15, time-series order, no shuffle (frozen from Phase 1)
2. **No future leakage**: All features must be available at prediction time (T) for predicting T+1
3. **5-day delay rule**: All data sources have <= 5 business day delay
4. **Transaction cost**: 5bps per trade (one-way) for Sharpe calculation
5. **Direction sign**: Returns exactly 0 excluded from DA calculation
6. **Self-contained Kaggle Notebook**: All training in a single self-contained notebook
7. **Optuna HP optimization**: Minimum 50 trials
8. **No gold price prediction by submodels**: Submodel outputs are latent features, not predictions
9. **Gate 3 ablation is NOT required for meta-model**: Final evaluation against absolute targets only
10. **yc_regime_prob must be excluded**: Constant column from collapsed HMM

## 7. Success Criteria

### Primary (all must be met simultaneously on test set)

| Metric | Target | Measurement |
|--------|--------|-------------|
| DA | > 56% | np.mean(np.sign(pred) == np.sign(actual)) excluding zeros |
| HC-DA | > 60% | DA on predictions where abs(pred) > threshold |
| MAE | < 0.75% | np.mean(np.abs(pred - actual)) |
| Sharpe | > 0.8 | annualized Sharpe of strategy returns minus 5bps per trade |

### Secondary (diagnostic, not blocking)

- Overfit ratio (train/val metric) < 1.5
- No constant or all-NaN output columns in predictions
- Feature importance: submodel features should contribute measurably (not all importance on base features)
- Stability: consistent performance across val and test splits

## 8. Expected Artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| Research report | `docs/research/meta_model_attempt_1.md` | Architecture analysis |
| Design document | `docs/design/meta_model_attempt_1.md` | Architecture specification |
| Merged input data | `data/processed/meta_model_input.csv` | Training input |
| Training notebook | `notebooks/meta_model_1/train.ipynb` | Kaggle execution |
| Evaluation summary | `logs/evaluation/meta_model_attempt_1_summary.md` | Results |

## 9. Strategic Considerations

### 9.1 Why the Gap is Large

The baseline (DA 43.5%, Sharpe -1.70) represents a near-random (slightly worse than random) predictor. The submodels individually showed incremental improvements of 0.05-0.96% DA each. Reaching 56% DA requires a multiplicative or interaction effect beyond simple addition of submodel gains.

This suggests the meta-model architecture should:
- Exploit interaction effects between regime states and raw features
- Use regime probabilities as gating/weighting mechanisms (when VIX regime = X AND cross-asset regime = Y, weight certain features differently)
- Consider that regime-conditioned prediction may be more powerful than flat regression

### 9.2 Submodel Output Nature

All 7 submodels use HMM-based regime detection. The 6 regime probability features represent different market states. The z-scores represent normalized feature dynamics. This is conceptually similar to a mixture-of-experts setup where regime states identify the expert and z-scores provide the input signal.

### 9.3 Risk of Over-engineering

With 39 features and only 1766 training samples, there is real risk of overfitting. The simplest approach that works is preferred. The researcher should seriously consider whether a well-tuned gradient boosting model (XGBoost/LightGBM) with appropriate regularization can meet the targets before recommending complex neural architectures.
