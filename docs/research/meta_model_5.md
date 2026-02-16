# Research Report: Meta-Model Attempt 5

**Date**: 2026-02-16
**Researcher**: researcher (Sonnet)
**Subject**: Integrating options_risk_regime_prob feature and re-optimizing for HCDA bottleneck

---

## Executive Summary

Attempt 4's post-hoc calibration catastrophically failed (HCDA 42.86% vs 60% target, 22.6pp validation-test gap). Attempt 5 returns to improving the base XGBoost model itself by:
1. Adding `options_risk_regime_prob` (rank #2 feature, 7.55% importance, 15.6x MAE threshold)
2. Full Optuna re-optimization with 100 trials to exploit feature interactions
3. Investigating optimal objective weights given persistent HCDA bottleneck

This report answers 5 research questions to inform Attempt 5's design with actionable, evidence-based recommendations.

---

## Research Questions

### Q1: Optuna Objective Weight Rebalancing for HCDA Bottleneck

**Question**: Current weights are MAE=40%, Sharpe=30%, DA=20%, HCDA=10%. Given that HCDA is the primary bottleneck (42.86% vs 60% target in attempt 4, 55.26% vs 60% in attempt 2), should HCDA weight increase? What are the trade-offs?

#### Current Weight Distribution Analysis

**Attempt 2 weights** (best result: 3/4 targets):
- Sharpe: 50%
- DA: 30%
- MAE: 10%
- HCDA: 10%

**Result**: DA 57.26% (pass), Sharpe 1.58 (pass), MAE 0.688% (pass), HCDA 55.26% (fail by 4.74pp)

**Attempt 4 weights** (calibration approach, not directly comparable):
- Same base model weights as attempt 2
- Result after calibration: HCDA 42.86% (catastrophic collapse)

#### Web Research on Multi-Objective Weight Selection

According to the [Weighted Sum Scalarization Method](https://medium.com/@ugurcanuzunkaya1/at-the-heart-of-multi-criteria-optimization-the-weighted-sum-scalarization-method-e2e9efedccb0), the weighted sum approach creates a linear combination of objectives where each weight represents the relative importance of that objective. **Critical insight**: Higher weight on an objective increases the pressure to optimize it, but may degrade other objectives if they are in conflict.

[Multi-objective Optimization with Optuna](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html) documentation shows that for true multi-objective optimization, Optuna can return a Pareto front of solutions rather than using weighted scalarization. However, weighted scalarization is simpler for automated execution and has been effective for 3 of 4 targets.

Recent 2025 developments include [AutoSampler](https://optuna.org/) and [GPSampler](https://medium.com/optuna/an-introduction-to-moea-d-and-examples-of-multi-objective-optimization-comparisons-8630565a4e89) for constrained multi-objective optimization, but these require explicit constraint formulation rather than weight tuning.

#### Trade-Off Analysis: Increasing HCDA Weight

**Scenario A: Increase HCDA from 10% to 20-30%**

Pros:
- Directly addresses the binding constraint (HCDA is the only failing target)
- May incentivize Optuna to find hyperparameters that produce larger |predictions| on genuinely predictable patterns
- Aligns optimization pressure with the actual bottleneck

Cons:
- May degrade DA if HCDA and DA are in conflict (they're not always aligned: high-confidence predictions can be directionally wrong)
- Reducing Sharpe weight from 50% to 40% may sacrifice the 1.58 Sharpe achievement (target is only 0.8, so 0.8pp margin exists)
- MAE weight reduction may allow magnitude errors to increase

**Scenario B: Keep current weights (50/30/10/10)**

Pros:
- Attempt 2 achieved 3/4 targets with these weights, demonstrating they're fundamentally sound
- Adding options_risk_regime_prob may naturally improve HCDA without weight rebalancing
- Avoids risk of destabilizing DA and Sharpe (which were passing)

Cons:
- Does not explicitly address the HCDA bottleneck
- Assumes the new feature alone will close the 4.74pp HCDA gap

#### Empirical Evidence from Attempt 2 HCDA Profile

From meta_model_attempt_2_summary.md:
- Top 10% |prediction|: 60.53% DA (n=38) -- **PASSES 60% target**
- Top 15%: 59.65% (n=57)
- Top 20%: 55.26% (n=76) -- **FAILS**
- Top 25%: 57.89% (n=95)

**Critical observation**: The 15-20th percentile band has **~42% DA** (worse than random), while the 20-25th band has **~68% DA**. This suggests the model is "confidently wrong" on certain patterns in the 15-20th band.

**Implication**: HCDA is not uniformly distributed. The problem is not that the model lacks high-confidence accuracy (top 10% passes), but that the **20th percentile threshold includes too many false-confidence predictions**.

#### Recommended Weight Distribution

**Option 1 (Recommended): Moderate HCDA Increase**
- Sharpe: 40% (reduced from 50%)
- DA: 30% (unchanged)
- MAE: 10% (unchanged)
- HCDA: 20% (increased from 10%)

**Rationale**:
- 2x weight on HCDA directly addresses the bottleneck
- Sharpe has 0.78 margin (1.58 actual vs 0.8 target), so 10pp weight reduction is acceptable
- DA and MAE remain stable (DA has only 1.26pp margin, so must stay at 30%)
- This configuration signals to Optuna: "HCDA is as important as DA"

**Option 2 (Conservative): Minimal Adjustment**
- Sharpe: 45% (reduced from 50%)
- DA: 30% (unchanged)
- MAE: 5% (reduced from 10%)
- HCDA: 20% (increased from 10%)

**Rationale**:
- MAE has largest margin (0.688% vs 0.75% target), so can be deprioritized
- Sharpe retains dominant position (45%)
- DA protected at 30%

**Option 3 (Aggressive): Equal Weighting of Bottlenecks**
- Sharpe: 30%
- DA: 30%
- HCDA: 30%
- MAE: 10%

**Rationale**:
- Treats DA and HCDA as equally important (they both drive profitability)
- High risk: May degrade Sharpe significantly

#### Answer to Original Question

**Recommended**: Increase HCDA weight from 10% to 20% (Option 1: 40/30/10/20 distribution).

**Trade-offs**:
- Sharpe may decline from 1.58 to ~1.2-1.4 (still well above 0.8 target)
- HCDA optimization pressure doubles, incentivizing magnitude-accuracy alignment
- DA and MAE risks are minimal (DA protected at 30%, MAE has margin)

**Fallback**: If Sharpe drops below 1.0, revert to Option 2 (45/30/5/20) in a subsequent attempt.

**Alternative approach**: Instead of changing weights, increase `confidence_threshold` search range to [0.15, 0.25] (currently top 20% = 80th percentile). This may naturally find a threshold where HCDA passes without weight rebalancing. However, this requires `confidence_threshold` to be an Optuna hyperparameter (currently it's computed as `np.percentile(|pred|, 80)`).

---

### Q2: Feature Interaction Terms for Regime Alignment

**Question**: Can feature interaction terms (e.g., options_risk_regime_prob * vix_regime_prob, options_risk_regime_prob * tech_trend_regime_prob) improve the model's ability to distinguish genuine high-confidence situations from false ones?

#### Theoretical Framework

**Why interaction terms matter**: Tree-based models like XGBoost can implicitly learn interactions through hierarchical splits (e.g., "if vix_regime_prob > 0.6 AND options_risk_regime_prob > 0.7, then..."). However, explicit interaction features can:
1. Accelerate learning by pre-computing critical patterns
2. Improve shallow tree performance (attempt 2 used max_depth=2, which limits implicit interaction depth)
3. Provide stronger signal when multiple regimes align (e.g., high VIX regime + high options risk regime → flight-to-quality gold rally)

#### Web Research on Feature Interactions in XGBoost

According to [Feature Interaction Constraints — XGBoost documentation](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html), XGBoost can constrain which features are allowed to interact during tree construction. This is the **inverse** of what we need: we want to **encourage** specific interactions, not constrain them.

[Feature Engineering for XGBoost Models](https://www.geeksforgeeks.org/machine-learning/feature-engineering-for-xgboost-models/) emphasizes that "creating interaction features involves performing arithmetic operations on features that may have nonlinear relationships, using domain knowledge to create meaningful combinations."

[Research on Feature Interactions in XGBoost (ArXiv)](https://arxiv.org/pdf/2007.05758) shows that accurate identification of feature interactions can significantly improve baseline XGBoost model performance.

**Critical insight**: "Generating polynomial and interaction features manually is less critical because decision trees can inherently model non-linear relationships through their hierarchical structure" ([Feature Engineering for XGBoost](https://xgboosting.com/feature-engineering-for-xgboost/)). However, this assumes **sufficient tree depth**. With max_depth=2 (attempt 2 setting), explicit interactions become valuable.

#### Candidate Interaction Terms

**High-priority interactions** (regime alignment):
1. `options_vix_regime_alignment = options_risk_regime_prob * vix_regime_probability`
   - **Economic meaning**: When both options market (SKEW/GVZ) and equity options (VIX) signal high risk, gold demand surges (flight-to-quality)
   - **Expected behavior**: High values (>0.5) indicate strong risk-off alignment → gold up

2. `options_tech_regime_alignment = options_risk_regime_prob * tech_trend_regime_prob`
   - **Economic meaning**: Options risk regime + gold technical trend regime alignment
   - **Expected behavior**: High options risk + bullish gold technicals → reinforced upward momentum

3. `multi_regime_risk_signal = options_risk_regime_prob * vix_regime_probability * xasset_regime_prob`
   - **Economic meaning**: Triple alignment (options, VIX, cross-asset) signals systemic risk
   - **Expected behavior**: High values → strong gold demand

**Medium-priority interactions** (mean-reversion signals):
4. `options_tech_divergence = options_risk_regime_prob * tech_mean_reversion_z`
   - **Economic meaning**: High options risk + gold overbought (negative z) → conflicting signals
   - **Expected behavior**: May identify false confidence zones where regimes conflict

**Low-priority interactions** (inflation/rates):
5. `options_inflation_alignment = options_risk_regime_prob * ie_regime_prob`
   - **Economic meaning**: High risk regime + high inflation expectation regime
   - **Expected behavior**: Weak signal (inflation is slower-moving than options risk)

#### Empirical Evidence from Submodel Evaluations

From options_market_2_summary.md:
- `options_risk_regime_prob` achieved rank #2 importance (7.55%) in a 20-feature baseline
- This suggests it's highly predictive **on its own**, but we don't know if it's redundant with VIX regime

From meta_model_attempt_2:
- Feature importance top-3: tech_trend_regime_prob (15.28), etf_pv_divergence (13.79), ie_gold_sensitivity_z (13.23)
- VIX regime features were NOT in the top-10, suggesting VIX regime is less important than expected
- **Implication**: `options_risk_regime_prob` may capture information NOT redundant with VIX regime, making interaction terms less necessary

#### Risks of Interaction Terms

**Curse of dimensionality**:
- Adding 3-5 interaction terms increases feature count from 23 to 26-28
- Samples-per-feature ratio decreases from ~71:1 to ~59-62:1
- May increase overfitting risk despite providing signal

**Multicollinearity**:
- Interaction terms are by definition correlated with their components (e.g., `A*B` is correlated with `A` and `B`)
- XGBoost is robust to multicollinearity (unlike linear models), but high VIF may still degrade interpretability

**Shallow tree limitation**:
- With max_depth=2, each tree can only capture 2-level interactions implicitly (e.g., split on A, then split on B)
- Adding explicit A*B interaction allows single-level splits to capture the pattern
- **Benefit is real but may be small**

#### Answer to Original Question

**Recommended**: Add 2-3 high-priority interaction terms as **optional Optuna-tuned features**.

**Implementation strategy**:
1. Pre-compute interactions: `options_vix_regime_alignment`, `options_tech_regime_alignment`, `multi_regime_risk_signal`
2. Add Optuna categorical parameter: `use_interactions = trial.suggest_categorical('use_interactions', [True, False])`
3. If `True`, include interaction terms in feature set (26 features); if `False`, exclude (23 features)
4. Let Optuna decide if interactions improve validation performance

**Expected impact**:
- If interactions provide signal: HCDA +1-2pp (by identifying regime-aligned high-confidence zones)
- If interactions add noise: Optuna will select `use_interactions=False`
- Minimal downside risk (Optuna-gated feature selection)

**Alternative (lower risk)**: Do NOT add interaction terms in attempt 5. Reserve for attempt 6+ if attempt 5 fails to close HCDA gap. This avoids dimensionality increase and relies on options_risk_regime_prob's standalone 7.55% importance.

**Final recommendation**: **Skip interaction terms in attempt 5** to minimize moving parts. The addition of options_risk_regime_prob (1 feature, 23 total) is already a significant change. If attempt 5 achieves HCDA > 57-58% but < 60%, add interactions in attempt 6.

---

### Q3: Options-Gold Prediction Literature Review

**Question**: Is there academic or empirical evidence that options market indicators (SKEW, GVZ, put/call ratios) improve gold return prediction quality specifically in the high-confidence subset?

#### Web Research Findings

**CBOE GVZ and Gold Volatility Prediction**:

According to [academic research on gold futures volatility](https://www.sciencedirect.com/science/article/abs/pii/S1544612319305793), "CBOE GVZ and silver implied volatility indices can help forecast realized volatility of gold futures prices, with results showing significant better predictive performance in models incorporating these indices." The study found that the GVZ is an "extremely important predictor for future volatility of Shanghai gold futures contracts."

**Critical finding**: This research focused on **volatility prediction**, not return prediction. High volatility ≠ high confidence in direction. However, volatility regimes may help identify when magnitude predictions should be larger (high vol regime → larger predicted moves).

**VIX-Gold Interaction for Risk Regimes**:

[2025 Gold Strategy Analysis](https://acy.com/en/market-news/education/gold-strategy-using-vix-yields-dxy-2025-l-s-162409/) shows that "when the VIX rises above 20, risk-averse sentiment fuels demand for safe assets like gold" and "when VIX spikes, the market is likely in a risk-off regime, and that's when capital seeks refuge and gold demand rises."

**Key insight**: VIX > 20-25 signals risk-off regime, which historically drives gold rallies. If options_risk_regime_prob captures similar risk regime information via SKEW and GVZ, it may identify high-confidence gold-up predictions during risk-off periods.

**Options Market Regime Alignment**:

[Market outlook for gold in 2025](https://www.investing.com/analysis/gold-price-outlook-for-late-2025-what-could-drive-the-next-rally-200667956) and [gold strategy using VIX, yields, and DXY](https://acy.com/en/market-news/education/gold-strategy-using-vix-yields-dxy-2025-l-s-162409/) emphasize that "when DXY is topping out and real yields start to fall, especially if VIX is rising, gold often leads the next major move" and "when VIX is rising alongside gold and equities are breaking down, that's a high-probability zone for gold trading."

**Implication**: Multi-regime alignment (VIX rising + yields falling + DXY weakening) creates high-confidence gold-up environments. If options_risk_regime_prob captures the options-derived component of this alignment, it may improve HCDA by increasing |prediction| during these aligned regimes.

#### Literature Gap

**No direct research found** on using CBOE SKEW or GVZ specifically for **gold return direction prediction** or **high-confidence prediction subset improvement**. Existing research focuses on:
- Gold volatility forecasting (not return prediction)
- VIX-gold correlation during risk-off events (not predictive modeling)
- Multi-factor gold strategies (not confidence calibration)

**Implication**: The use of options_risk_regime_prob for HCDA improvement is **theoretically sound but empirically untested** in academic literature. We are pioneering this approach.

#### Empirical Evidence from This Project

From options_market_2_summary.md (Gate 3 ablation):
- **MAE improvement**: -0.156% (15.6x threshold, 4 of 5 folds improved)
- **DA change**: -0.24% (slight degradation, but 3 of 5 folds improved)
- **Sharpe change**: -0.141 (degradation)
- **Feature importance**: 7.55%, rank #2 of 20 features

**Interpretation**:
- options_risk_regime_prob improves magnitude accuracy (MAE -0.156%) significantly
- DA is neutral to slightly negative (-0.24%)
- Sharpe degradation suggests the feature may increase prediction magnitude on **wrong days** (high magnitude wrong predictions → larger losses → lower Sharpe)

**Critical question**: If MAE improved but Sharpe degraded, does this mean the feature increases magnitude on both correct AND incorrect predictions? If so, it may **worsen HCDA** by promoting false-confidence predictions.

**Counter-evidence**: Feature importance of 7.55% (rank #2) suggests XGBoost found it highly useful in the 5-fold CV. The Sharpe degradation may be fold-specific (fold 4 had -0.803 Sharpe delta, pulling average down).

#### Answer to Original Question

**Academic evidence**: Moderate. GVZ and SKEW predict gold **volatility** (well-supported), but not return **direction** or **high-confidence accuracy** (no direct research found).

**Empirical evidence from this project**: Promising. MAE improvement of 15.6x threshold with rank #2 importance suggests strong signal. Sharpe degradation is concerning but may be fold-specific noise.

**Mechanism hypothesis**: options_risk_regime_prob may improve HCDA by:
1. Identifying risk-off regimes where gold-up predictions should have higher magnitude (→ larger |pred| on correct days)
2. Identifying low-risk regimes where predictions should be more conservative (→ smaller |pred| on uncertain days)

**Risk**: If the feature increases magnitude on both correct and incorrect predictions uniformly, HCDA will not improve (false-confidence predictions remain in top 20%).

**Recommendation**: Proceed with integration in attempt 5. The 15.6x MAE improvement threshold and rank #2 importance outweigh the Sharpe concern. Monitor fold-level HCDA in Optuna trials to detect if the feature specifically improves high-confidence accuracy.

---

### Q4: Confidence Threshold Search Strategy

**Question**: Should the confidence_threshold be searched independently, or can Optuna's HCDA component naturally find the right balance? What threshold ranges are reasonable given attempt 2's prediction distribution (std ~0.167%)?

#### Current Approach Analysis

**Attempt 2 approach**: HCDA computed as DA on top 20% by |prediction| (80th percentile). This is a **fixed percentile**, not an optimized threshold.

**Attempt 2 results by percentile**:
- Top 10%: 60.53% DA (n=38) -- PASS
- Top 15%: 59.65% (n=57) -- marginal
- Top 20%: 55.26% (n=76) -- FAIL
- Top 25%: 57.89% (n=95) -- worse than 15%
- Top 30%: 57.89% (n=114) -- same as 25%

**Observation**: HCDA is **non-monotonic** in percentile threshold. It peaks at ~10-15%, declines at 20%, then partially recovers at 25-30%. This suggests the **15-20th percentile band contains low-quality predictions** (the "confidently wrong" zone identified in Q1).

#### Alternative Threshold Approaches

**Option A: Fixed Percentile (Current)**
- Compute `threshold = np.percentile(|predictions|, 80)` → top 20%
- HCDA = DA on predictions where |pred| > threshold
- Pros: Simple, interpretable, matches CLAUDE.md spec ("top 20%")
- Cons: Does not adapt to prediction distribution; includes low-quality 15-20th band

**Option B: Optuna-Optimized Percentile**
- Add `conf_percentile = trial.suggest_float('conf_percentile', 70, 95)` (top 30% to top 5%)
- Compute `threshold = np.percentile(|predictions|, conf_percentile)`
- HCDA = DA on predictions where |pred| > threshold
- Pros: Adapts to find optimal trade-off (may discover 85th percentile = top 15% is optimal)
- Cons: Violates CLAUDE.md spec ("top 20%"), changes evaluation metric definition

**Option C: Absolute Magnitude Threshold**
- Add `conf_threshold_abs = trial.suggest_float('conf_threshold_abs', 0.05, 0.30)` (5-30 bps)
- HCDA = DA on predictions where |pred| > conf_threshold_abs
- Pros: Economically interpretable (e.g., "only trade when |pred| > 10bps")
- Cons: Sample count varies (may have 5% or 40% of predictions above threshold depending on distribution)

**Option D: Hybrid (Percentile OR Absolute)**
- `threshold = max(np.percentile(|pred|, 80), conf_threshold_abs)`
- Ensures at least top 20%, but allows higher threshold if predictions are all small
- Pros: Adaptive while respecting CLAUDE.md minimum
- Cons: Complex, two tunable parameters

#### Web Research on Confidence Thresholds

According to [classification threshold balancing](https://www.evidentlyai.com/classification-metrics/classification-threshold), "how to use classification threshold to balance precision and recall" involves optimizing the threshold as a hyperparameter based on the evaluation metric. This is common in classification, but **regression confidence thresholds are less studied**.

[Research on quantiles vs percentiles](https://www.machinelearningplus.com/statistics/quantiles-and-percentiles/) confirms they are mathematically equivalent: percentiles divide by 100, quantiles are the general case. For our purposes, **percentile-based and quantile-based thresholds are identical**.

#### Prediction Distribution Constraints

From attempt 2:
- Prediction std: 0.167%
- Actual std: 0.896%
- Prediction range: roughly [-0.5%, +0.5%] (3 std deviations)

If we use absolute threshold:
- 0.05% (5bps): ~50-60% of predictions qualify (too broad)
- 0.10% (10bps): ~30-40% of predictions qualify (close to top 20%)
- 0.15% (15bps): ~15-20% of predictions qualify (close to top 10%)
- 0.20% (20bps): ~5-10% of predictions qualify (too selective)

**Observation**: With prediction std ~0.167%, an absolute threshold of 0.10-0.15% approximately corresponds to top 20-30% by percentile. Optimizing absolute threshold may not provide significantly different results than optimizing percentile.

#### Answer to Original Question

**Recommended**: Keep fixed percentile at 80th (top 20%) **in attempt 5**, matching CLAUDE.md spec. Do NOT add threshold as an Optuna hyperparameter.

**Rationale**:
1. **Spec compliance**: CLAUDE.md defines HCDA as "top 20% by |prediction|"
2. **Evaluation consistency**: Changing the threshold changes the metric definition, making cross-attempt comparison invalid
3. **Optuna HCDA objective is sufficient**: By including HCDA at 20% weight in the objective, Optuna will find hyperparameters that naturally improve top-20% DA without explicitly tuning the threshold
4. **Risk of overfitting**: Optimizing the threshold on validation set may not generalize to test set (attempt 4's calibration failure demonstrated this)

**Alternative for future attempts**: If attempt 5 achieves HCDA 57-59% (close but not passing), attempt 6 could explore:
- Reporting HCDA at multiple thresholds (10%, 15%, 20%) and selecting the best post-hoc
- Asymmetric thresholds (different for up vs down predictions)
- Regime-dependent thresholds (higher threshold in high-vol regimes)

**Threshold range recommendation** (if optimized in future): `conf_percentile = [75, 90]` (top 25% to top 10%), keeping 80th as midpoint.

---

### Q5: Rolling-Window Features for Temporal Context

**Question**: Would adding rolling-window feature engineering (e.g., 5-day average of options_risk_regime_prob, regime persistence duration) provide useful temporal context without introducing noise?

#### Theoretical Framework

**Why rolling windows matter for regimes**: Regime features like `options_risk_regime_prob` are semi-persistent but not constant. A single-day spike to 0.9 may be noise; a 5-day average >0.7 indicates sustained high-risk regime. Rolling windows smooth noise and capture persistence.

**Existing persistence features**: The project already includes `vix_persistence` (from VIX submodel), which measures how long VIX has been in the current regime. This demonstrates that persistence signals are valuable.

#### Web Research on Rolling Window Features

According to [Rolling window strategies for time series](https://www.emergentmind.com/topics/rolling-window-strategy), "rolling window strategies are the backbone of predictive modeling for nonstationary time series, especially in machine learning and econometrics."

[Practical guide for time series feature engineering](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/) emphasizes that "rolling window statistics—like moving averages and variances—help smooth out noise and highlight local trends. By computing these statistics over a sliding window, you capture the temporal dynamics and volatility of the data."

**Recent 2024-2025 research** ([Rolling analysis of time series](https://arxiv.org/html/2510.23150v2)) shows that "re-estimation frequency (stride) are the chief determinants of out-of-sample forecasting performance—even more than model complexity or feature set, in many regimes. Notably, simple models (e.g., HAR) with daily rolling windows achieve lower prediction error than complex ML models fitted with static windows."

**Key finding**: Rolling windows improve generalization by capturing local patterns, but window size and stride matter critically. Too short → noise; too long → lag.

#### Candidate Rolling-Window Features

**High-priority candidates**:
1. `options_regime_5d_avg = rolling_mean(options_risk_regime_prob, window=5)`
   - **Economic meaning**: 5-day average regime probability smooths daily noise
   - **Expected behavior**: Values >0.6 indicate sustained risk-off regime → high-confidence gold-up
   - **Window rationale**: 5 days = 1 trading week, balances responsiveness and smoothing

2. `options_regime_volatility = rolling_std(options_risk_regime_prob, window=10)`
   - **Economic meaning**: Regime instability (frequent regime switches)
   - **Expected behavior**: High volatility → uncertain environment → lower confidence
   - **Window rationale**: 10 days = 2 weeks, captures regime transition periods

3. `options_regime_persistence = days_since_regime_change(options_risk_regime_prob, threshold=0.5)`
   - **Economic meaning**: How many consecutive days has regime been above/below 0.5?
   - **Expected behavior**: Long persistence → stable regime → higher confidence
   - **Implementation**: Count consecutive days where regime > 0.5 (or < 0.5)

**Medium-priority candidates**:
4. `options_vix_corr_5d = rolling_correlation(options_risk_regime_prob, vix_regime_probability, window=5)`
   - **Economic meaning**: Short-term alignment between options and equity volatility regimes
   - **Expected behavior**: High correlation → aligned risk signal → higher confidence
   - **Risk**: Correlation on 5-day windows may be noisy with only 5 samples

#### Risks of Rolling-Window Features

**Look-ahead bias**: Rolling windows inherently use past data only (no future leakage), but must ensure window calculation is correct (e.g., `rolling_mean(x, window=5)` at day t uses days t-4 to t, not t to t+4).

**Data loss at start**: First 5-10 rows will have NaN for rolling features (warm-up period). This is acceptable given we already impute early NaNs with domain-specific values (Q5 from attempt 2 research).

**Correlation with existing features**: `options_regime_5d_avg` will be highly correlated with `options_risk_regime_prob` (r ~0.8-0.9). XGBoost is robust to this, but it increases feature count without necessarily adding information.

**Increased dimensionality**: Adding 2-3 rolling features increases total from 23 to 25-26. Samples-per-feature ratio decreases from ~71:1 to ~63-67:1.

#### Empirical Evidence from Existing Features

From attempt 2 feature importance:
- `vix_persistence`: Not in top-10 importance (likely rank 11-20)
- `tech_volatility_regime`: Not in top-10
- `etf_pv_divergence`: Rank #5 (13.79%) — this is a divergence/momentum feature, conceptually similar to rolling volatility

**Implication**: Persistence and volatility features exist in the feature set but are NOT top drivers of performance. This suggests rolling-window features may have **marginal utility**.

#### Answer to Original Question

**Recommended**: Do NOT add rolling-window features in attempt 5.

**Rationale**:
1. **Marginal expected value**: Existing persistence features (vix_persistence) are not top-10 important, suggesting temporal context is less critical than regime levels
2. **Dimensionality cost**: Adding 2-3 rolling features increases feature count to 25-26, reducing samples-per-feature ratio
3. **Correlation with source**: Rolling averages are highly correlated with source features (multicollinearity, even if XGBoost handles it)
4. **Minimize moving parts**: Attempt 5 already adds 1 new feature (options_risk_regime_prob) and potentially reweights Optuna objective. Adding 2-3 more features compounds risk.

**Alternative approach**: If attempt 5 achieves HCDA 57-59% but not 60%, attempt 6 can add **1 high-priority rolling feature** (`options_regime_5d_avg`) to test if smoothing improves signal.

**Exception**: If architect determines that regime persistence is critical for HCDA (based on analysis of attempt 2's confidently-wrong 15-20th percentile band), then add `options_regime_persistence` only (1 feature, not 2-3).

**Final recommendation**: **Skip rolling-window features in attempt 5**. Focus on the base options_risk_regime_prob feature and Optuna re-optimization. Reserve rolling features for later attempts if needed.

---

## Consolidated Recommendations for Attempt 5

### 1. Optuna Objective Weights (Q1)

**Recommended**: Moderate HCDA increase
- Sharpe: 40% (reduced from 50%)
- DA: 30% (unchanged)
- MAE: 10% (unchanged)
- HCDA: 20% (increased from 10%)

**Rationale**: HCDA is the binding constraint; Sharpe has 0.78 margin; DA has minimal margin and must be protected.

**Alternative**: 45/30/5/20 if Sharpe risk is high concern.

### 2. Feature Interaction Terms (Q2)

**Recommended**: Do NOT add interaction terms in attempt 5.

**Rationale**: Minimize moving parts; options_risk_regime_prob's standalone 7.55% importance is already strong; interaction terms increase dimensionality and may not provide benefit with max_depth=2-4.

**Reserve for**: Attempt 6 if HCDA is 57-59% (close but not passing).

### 3. Options-Gold Literature Support (Q3)

**Finding**: Moderate academic support for GVZ/SKEW predicting gold **volatility**; no direct support for **direction prediction** or **HCDA improvement**.

**Project empirical evidence**: Strong (MAE improvement 15.6x threshold, rank #2 importance).

**Recommendation**: Proceed with integration; the empirical project evidence outweighs lack of academic literature on this specific application.

### 4. Confidence Threshold Strategy (Q4)

**Recommended**: Keep fixed at 80th percentile (top 20%) per CLAUDE.md spec.

**Rationale**: Spec compliance; evaluation consistency; Optuna HCDA objective is sufficient; risk of overfitting threshold to validation set.

**Do NOT**: Add threshold as Optuna hyperparameter in attempt 5.

**Alternative**: Report HCDA at multiple thresholds (10%, 15%, 20%) in evaluation for diagnostic purposes, but use 20% for pass/fail decision.

### 5. Rolling-Window Features (Q5)

**Recommended**: Do NOT add rolling-window features in attempt 5.

**Rationale**: Marginal expected value (existing persistence features not top-10 important); dimensionality cost; correlation with source; minimize moving parts.

**Reserve for**: Attempt 6 if HCDA is close (57-59%) and temporal smoothing hypothesis is strong.

---

## Expected Outcomes for Attempt 5

| Change | Expected Impact on HCDA | Expected Impact on Other Metrics |
|--------|-------------------------|----------------------------------|
| Add options_risk_regime_prob | +2-3pp (rank #2 feature, may reduce confidently-wrong predictions in 15-20th band) | DA: ±0.5pp (neutral), MAE: -0.05% (improves), Sharpe: -0.1 to +0.2 (uncertain) |
| Reweight Optuna (40/30/10/20) | +1-2pp (explicit HCDA optimization pressure) | Sharpe: -0.2 to -0.4 (acceptable, still >1.0) |
| 100 Optuna trials (vs 80) | +0.5-1pp (better HP exploration) | All metrics: +0-5% (marginal improvement from better HP) |
| **Combined effect** | **+3.5-6pp → 58.8-61.3% HCDA** | DA: 56.5-58% (pass), Sharpe: 1.2-1.4 (pass), MAE: 0.64-0.68% (pass) |

**Probability of HCDA > 60%**: Moderate to High (60-70%)

**Primary risk**: options_risk_regime_prob increases magnitude on both correct AND incorrect predictions uniformly, providing no HCDA benefit (only MAE benefit). In this case, HCDA may remain at 55-57%.

**Fallback plan**: If HCDA is 57-59%, attempt 6 adds interaction terms or rolling features. If HCDA is <55%, investigate whether options feature is counterproductive and consider removing it.

---

## Limitations and Uncertainties

### 1. Options Feature Impact Uncertainty (Q3)

**Uncertainty**: options_risk_regime_prob improved MAE by 15.6x threshold but degraded Sharpe by -0.141. This suggests it may increase magnitude on wrong predictions as well as right predictions.

**Implication**: HCDA may not improve as much as expected if the feature increases |prediction| uniformly rather than selectively on high-accuracy predictions.

**Mitigation**: Monitor fold-level HCDA in Optuna trials. If early trials show HCDA degradation, add constraint or remove feature.

### 2. Objective Weight Trade-Off (Q1)

**Uncertainty**: Reducing Sharpe weight from 50% to 40% may cause Sharpe to drop below 1.0 (though still above 0.8 target).

**Implication**: If Sharpe drops to 0.9, we're narrowing the margin. If it drops below 0.8, we've traded one failing metric (HCDA) for another (Sharpe).

**Mitigation**: If Optuna trials show Sharpe <1.0 consistently, revert to 45/30/5/20 weights mid-training or in attempt 6.

### 3. Interaction Term Decision (Q2)

**Uncertainty**: Skipping interaction terms may leave performance on the table if regime alignment is the key to HCDA improvement.

**Implication**: If attempt 5 achieves 58-59% HCDA, we may regret not testing interactions.

**Mitigation**: Architect can override this recommendation if design analysis suggests interactions are critical. Alternatively, attempt 6 can add interactions as a follow-up.

### 4. No Academic Validation (Q3)

**Uncertainty**: The use of SKEW/GVZ for gold return direction prediction has no published academic validation. We are pioneering this approach.

**Implication**: Higher risk of feature being ineffective or even harmful.

**Mitigation**: Strong empirical validation from Phase 2 (Gate 3 pass with 15.6x MAE threshold) provides confidence. Trust the project's own ablation testing.

---

## Sources

### Web Research Sources

**Q1 (Optuna Multi-Objective Optimization)**:
- [Multi-objective Optimization with Optuna](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)
- [Weighted Sum Scalarization Method](https://medium.com/@ugurcanuzunkaya1/at-the-heart-of-multi-criteria-optimization-the-weighted-sum-scalarization-method-e2e9efedccb0)
- [MOEA/D and Multi-Objective Optimization](https://medium.com/optuna/an-introduction-to-moea-d-and-examples-of-multi-objective-optimization-comparisons-8630565a4e89)

**Q2 (Feature Interaction Engineering)**:
- [Feature Interaction Constraints — XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html)
- [Feature Engineering for XGBoost Models](https://www.geeksforgeeks.org/machine-learning/feature-engineering-for-xgboost-models/)
- [Feature Interactions in XGBoost (ArXiv)](https://arxiv.org/pdf/2007.05758)
- [Feature Engineering for XGBoost](https://xgboosting.com/feature-engineering-for-xgboost/)

**Q3 (Options-Gold Prediction Literature)**:
- [Can CBOE gold and silver implied volatility help forecast gold futures volatility](https://www.sciencedirect.com/science/article/abs/pii/S1544612319305793)
- [Gold Strategy Using VIX, Yields, DXY 2025](https://acy.com/en/market-news/education/gold-strategy-using-vix-yields-dxy-2025-l-s-162409/)
- [Gold Price Outlook for Late 2025](https://www.investing.com/analysis/gold-price-outlook-for-late-2025-what-could-drive-the-next-rally-200667956)
- [CBOE Gold ETF Volatility Index (GVZCLS) | FRED](https://fred.stlouisfed.org/series/GVZCLS)

**Q4 (Confidence Thresholds)**:
- [How to use classification threshold to balance precision and recall](https://www.evidentlyai.com/classification-metrics/classification-threshold)
- [Quantiles and Percentiles](https://www.machinelearningplus.com/statistics/quantiles-and-percentiles/)

**Q5 (Rolling Window Features)**:
- [Rolling Window Strategy](https://www.emergentmind.com/topics/rolling-window-strategy)
- [Practical Guide for Feature Engineering of Time Series Data](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/)
- [Revisiting the Structure of Trend Premia](https://arxiv.org/html/2510.23150v2)
- [Feature engineering for time-series data](https://www.statsig.com/perspectives/feature-engineering-timeseries)

### Empirical Sources (Project Data)

- shared/current_task.json
- logs/evaluation/meta_model_attempt_2_summary.md
- logs/evaluation/options_market_2_summary.md
- docs/research/meta_model_attempt_2.md

---

## Key Insights for Architect

1. **HCDA bottleneck is structural, not hyperparameter-driven**: Attempt 2's top 10% achieves 60.53% DA, but the 15-20th percentile band has ~42% DA. The problem is **specific low-quality predictions in the marginal confidence zone**, not overall model quality.

2. **options_risk_regime_prob has strong MAE signal but Sharpe concern**: The feature improved MAE by 15.6x threshold but degraded Sharpe by -0.141. This pattern suggests it may increase magnitude uniformly rather than selectively on high-accuracy predictions. Architect should investigate whether this is acceptable or requires mitigation.

3. **Objective weight rebalancing is low-risk**: Sharpe has 0.78 margin (1.58 vs 0.8 target), allowing 10pp weight reduction. Increasing HCDA from 10% to 20% directly addresses the bottleneck with minimal risk to other metrics.

4. **Interaction terms and rolling features should be reserved for attempt 6**: Adding too many features in attempt 5 increases complexity and dimensionality risk. Focus on the single strongest feature (options_risk_regime_prob) and let Optuna re-optimize around it.

5. **Confidence threshold should remain fixed at 80th percentile**: Optimizing the threshold risks overfitting to validation set (as demonstrated by attempt 4's calibration disaster). Let Optuna improve HCDA at the fixed threshold by finding better base model hyperparameters.

---

**End of Research Report**

**Next step**: Architect will fact-check this report and create design document for Attempt 5.
