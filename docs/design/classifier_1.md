# Classifier Design Document: Gold DOWN Detector (Attempt 1)

## 0. Fact-Check Results

### 0.1 Data Source Verification

| Series / Ticker | Source | Status | Detail |
|-----------------|--------|--------|--------|
| GVZCLS | FRED | PASS | Gold Volatility Index, starts 2008-06-03 (researcher said 2011 -- corrected) |
| VIXCLS | FRED | PASS | VIX, starts 1990-01-02 |
| DFII10 | FRED | PASS | 10Y TIPS Yield, starts 2003-01-02 |
| DGS10 | FRED | PASS | 10Y Nominal Yield |
| DGS2 | FRED | PASS | 2Y Nominal Yield |
| GC=F | yfinance | PASS | Gold futures OHLCV |
| GLD | yfinance | PASS | Gold ETF OHLCV |
| SI=F | yfinance | PASS | Silver futures |
| HG=F | yfinance | PASS | Copper futures |
| DX-Y.NYB | yfinance | PASS | Dollar Index |
| ^GSPC | yfinance | PASS | S&P 500 |
| ^VIX | yfinance | PASS | VIX (alternative to FRED) |
| ^SKEW | yfinance | PASS | CBOE Skew Index |
| ^GVZ | yfinance | PASS | Gold VIX (alternative to FRED) |
| ^TNX | yfinance | PASS | 10Y Treasury Yield |

### 0.2 Empirical Claim Verification

All key claims independently reproduced using 2015-2025 GC=F daily data (n=2,764):

| Claim | Researcher Value | Verified Value | Status |
|-------|-----------------|----------------|--------|
| Baseline UP/DOWN split | 54% / 46% | 53.0% / 46.6% | PASS (close, minor rounding) |
| RSI > 70 DOWN prob | 45.68% | 45.57% | PASS (no predictive power) |
| Streak >= 5 DOWN prob | 47.31% | 47.31% | PASS (exact match, minimal effect) |
| rv_ratio_10_30 > 1.2 DOWN prob | 51.60% (+5.07pp) | 51.71% (+5.15pp) | PASS (strongest signal confirmed) |
| GVZ/VIX ratio change > 5% DOWN prob | 43.72% | 43.72% | PASS (counterintuitive: lower DOWN) |
| GVZ/VIX correlation w/ next return | +0.0444 | +0.0446 | PASS (weak positive) |
| VIX High regime next-day DOWN prob | 45.77% | 45.72% | PASS (no significance) |
| Day-of-week ANOVA | p=0.972 | Range: 44.27%-49.29% | PASS (no significance) |
| Total DOWN days | 1,286 | 1,287 | PASS (1 day difference, negligible) |

### 0.3 Methodology Assessment

| Finding | Assessment | Design Implication |
|---------|-----------|-------------------|
| RSI/BB/Streak have zero power | CONFIRMED -- gold is macro-driven, not momentum-driven | Exclude from core features. Include RSI as low-priority only. |
| rv_ratio_10_30 is strongest | CONFIRMED -- +5.15pp is meaningful and consistent | Make this Tier 1 feature |
| GVZ/VIX counterintuitive | CONFIRMED -- GVZ spike = fear already priced in | Include ratio level AND change, let XGBoost learn non-linear interaction |
| Day-of-week no significance | CONFIRMED -- gold trades 24h globally | Include as zero-cost feature (integer encoding) |
| VIX regime no power | CONFIRMED -- simple regime is too coarse | Use VIX change and VIX acceleration instead of regime buckets |
| GVZCLS start date | Researcher: 2011. Actual: 2008. | Minor error. Using 2015+ data, no impact on design. |

### 0.4 Research Quality Summary

Research quality: HIGH. All 8 verified empirical claims match within rounding tolerance. The researcher correctly identified that traditional technical indicators fail for gold and that volatility-based features are more promising. The feature priority ranking is well-justified. One minor factual error (GVZ start date) with no design impact.

---

## 1. Overview

- **Purpose**: Build a standalone binary classifier (UP=1, DOWN=0) to detect gold DOWN days, then ensemble with the existing regression meta-model (attempt 7) to improve DOWN-day capture from ~10% to >25%.
- **Architecture**: XGBoost with binary:logistic objective + custom focal loss option
- **Key Insight**: The regression model (attempt 7) has a 96% bullish bias in 2026 YTD. It catches 20/21 UP days but only 1/10 DOWN days. The classifier uses DIFFERENT features than the regression model's 24 features to capture reversal signals, volatility regime shifts, and cross-asset stress that the regression model cannot detect.
- **Expected Effect**: Standalone DOWN recall >30%, ensemble DA improvement +1-2.5pp over regression-only (60.04% -> 61-62.5%), Sharpe maintained above 2.0.

---

## 2. Data Specification

### 2.1 Raw Data Sources

| Source | Ticker / Series | Fields Used | Period | Purpose |
|--------|----------------|-------------|--------|---------|
| yfinance | GC=F | OHLCV | 2014-01-01 to present | Gold futures (core features + target) |
| yfinance | GLD | OHLCV | 2014-01-01 to present | ETF volume features |
| yfinance | SI=F | Close | 2014-01-01 to present | Gold/Silver divergence |
| yfinance | HG=F | Close | 2014-01-01 to present | Gold/Copper divergence |
| yfinance | DX-Y.NYB | Close | 2014-01-01 to present | DXY acceleration |
| yfinance | ^GSPC | Close | 2014-01-01 to present | Risk-off composite, equity-gold beta |
| FRED | GVZCLS | Close | 2014-01-01 to present | Gold volatility index |
| FRED | VIXCLS | Close | 2014-01-01 to present | Equity volatility |
| FRED | DFII10 | Close | 2014-01-01 to present | Real rate surprise |
| FRED | DGS10 | Close | 2014-01-01 to present | 10Y yield for risk-off score |

Note: Data fetched from 2014-01-01 to allow 252-day (1-year) lookback windows. Effective feature start date: ~2015-01-15 after rolling window warmup.

### 2.2 Target Variable

```python
target = (gold_return_next > 0).astype(int)
# 1 = UP (next-day return > 0%)
# 0 = DOWN (next-day return <= 0%)
# Days with exactly 0% return classified as DOWN (conservative, anti-bullish-bias)
```

Class balance (2015-2025): ~53% UP / 47% DOWN (mild imbalance, ratio ~1.13:1)

### 2.3 Data Split

- Train: first 70% by date (~2015-01 to ~2022-08)
- Validation: next 15% (~2022-08 to ~2024-02)
- Test: final 15% (~2024-02 to ~2025-12)
- NO shuffle, strict time-series order

### 2.4 Expected Sample Count

- Total: ~2,750 trading days (2015-01 to 2025-12)
- Train: ~1,925
- Validation: ~413
- Test: ~413

### 2.5 Missing Data Handling

1. FRED series (GVZCLS, VIXCLS, DFII10, DGS10): forward-fill with max 5-day gap
2. yfinance series: forward-fill with max 3-day gap (weekends/holidays)
3. After forward-fill, drop any remaining NaN rows (should be <1% of data)
4. All features computed after forward-fill to avoid lookahead

---

## 3. Feature Specification (18 Features)

### 3.1 Category A: Volatility Regime Features (5 features)

These are the highest-priority features based on empirical evidence that volatility dynamics, not price momentum, drive gold reversals.

#### A1: rv_ratio_10_30 (Realized Volatility Ratio)
- **Source**: GC=F Close
- **Formula**: `rv_10 = std(daily_return, window=10); rv_30 = std(daily_return, window=30); rv_ratio = rv_10 / rv_30`
- **Normalization**: Raw ratio (typically 0.5-2.0 range, already bounded)
- **Expected Range**: [0.3, 3.0] with median ~1.0
- **Why it helps DOWN detection**: When short-term vol expands relative to medium-term vol (ratio > 1.2), DOWN probability increases by +5.15pp. This captures volatility regime shifts that precede reversals.
- **Evidence**: Strongest single signal in empirical analysis. 439 occurrences with 51.71% DOWN prob vs 46.56% baseline.

#### A2: rv_ratio_10_30_z (Realized Vol Ratio Z-Score)
- **Source**: Derived from A1
- **Formula**: `(rv_ratio_10_30 - rolling_mean(rv_ratio_10_30, 60)) / rolling_std(rv_ratio_10_30, 60)`
- **Normalization**: Z-scored (mean ~0, std ~1)
- **Expected Range**: [-3, 4]
- **Why it helps DOWN detection**: Captures how ABNORMAL the current vol ratio is relative to recent history. A z-score of +2 means vol expansion is unusually rapid, a stronger DOWN signal than the raw ratio alone.

#### A3: gvz_level_z (Gold VIX Z-Score)
- **Source**: FRED GVZCLS
- **Formula**: `(gvz - rolling_mean(gvz, 60)) / rolling_std(gvz, 60)`
- **Normalization**: Z-scored
- **Expected Range**: [-2, 5]
- **Why it helps DOWN detection**: GVZ spikes capture gold-specific options market fear. Unlike the binary options_risk_regime_prob in the regression model, this is a continuous intensity measure.

#### A4: gvz_vix_ratio (Gold Vol / Equity Vol Ratio)
- **Source**: FRED GVZCLS / VIXCLS
- **Formula**: `gvz / vix` (raw ratio)
- **Normalization**: Raw ratio (typically 0.5-3.0)
- **Expected Range**: [0.3, 5.0]
- **Why it helps DOWN detection**: The ratio level captures gold-specific fear relative to market fear. Research shows the relationship is non-linear -- XGBoost can learn the complex pattern (counterintuitively, high ratio = lower DOWN prob due to mean-reversion after panic).

#### A5: intraday_range_ratio (Daily Range vs Average)
- **Source**: GC=F High, Low, Close
- **Formula**: `daily_range = (high - low) / close; range_ratio = daily_range / rolling_mean(daily_range, 20)`
- **Normalization**: Ratio (centered around 1.0)
- **Expected Range**: [0.1, 5.0]
- **Why it helps DOWN detection**: Abnormally wide daily ranges (ratio > 2) signal uncertainty and potential directional change. Complements realized vol with intraday information.

### 3.2 Category B: Cross-Asset Stress Features (4 features)

Gold reversals often coincide with multi-asset risk-off cascades. These features capture cross-market synchronization that individual asset features miss.

#### B1: risk_off_score (Cross-Asset Stress Composite)
- **Source**: VIXCLS, DX-Y.NYB, ^GSPC, DGS10
- **Formula**:
  ```
  vix_z = (vix_change - rolling_mean(vix_change, 20)) / rolling_std(vix_change, 20)
  dxy_z = (dxy_change - rolling_mean(dxy_change, 20)) / rolling_std(dxy_change, 20)
  spx_z = (spx_return - rolling_mean(spx_return, 20)) / rolling_std(spx_return, 20)
  yield_z = (yield_change - rolling_mean(yield_change, 20)) / rolling_std(yield_change, 20)
  risk_off_score = vix_z + dxy_z - spx_z - yield_z
  ```
- **Normalization**: Composite z-scores (approximately mean 0, but can range widely)
- **Expected Range**: [-6, 8]
- **Why it helps DOWN detection**: When VIX rises, USD strengthens, equities fall, AND yields drop simultaneously, gold faces competing safe-haven vs USD-headwind forces. High scores indicate synchronized stress.

#### B2: gold_silver_ratio_change (Gold/Silver Relative Performance)
- **Source**: GC=F, SI=F
- **Formula**: `gold_5d_return = pct_change(gold_close, 5); silver_5d_return = pct_change(silver_close, 5); divergence = gold_5d_return - silver_5d_return`
- **Normalization**: Z-scored vs 60-day rolling window
- **Expected Range**: [-3, 3]
- **Why it helps DOWN detection**: When silver outperforms gold (negative divergence), it signals industrial demand > safe-haven premium, reducing gold's floor. When gold outperforms silver (positive divergence), safe-haven buying supports gold.

#### B3: equity_gold_beta_20d (Rolling Beta to S&P 500)
- **Source**: GC=F, ^GSPC
- **Formula**: Rolling 20-day OLS beta of gold daily returns on SPX daily returns. `beta = cov(gold_ret, spx_ret, 20d) / var(spx_ret, 20d)`
- **Normalization**: Raw (typically -0.5 to +0.5)
- **Expected Range**: [-1.0, 1.0]
- **Why it helps DOWN detection**: Normally gold has near-zero or negative beta to equities (safe-haven). When beta turns significantly positive, the safe-haven premium has collapsed, increasing DOWN risk. Using 20d window (not 5d as researcher suggested) for stability.

#### B4: gold_copper_ratio_change (Gold/Copper Relative Performance)
- **Source**: GC=F, HG=F
- **Formula**: `gold_5d_return - copper_5d_return`, z-scored vs 60-day rolling window
- **Normalization**: Z-scored
- **Expected Range**: [-3, 3]
- **Why it helps DOWN detection**: Copper is a pure cyclical/industrial metal. Gold underperforming copper signals risk-on rotation away from safe havens.

### 3.3 Category C: Rate and Currency Shock Features (3 features)

Rate surprises and DXY acceleration are direct fundamental drivers of gold price changes, distinct from the regression model's raw level/change features.

#### C1: rate_surprise (Real Rate Shock Magnitude)
- **Source**: FRED DFII10
- **Formula**: `rate_change = diff(dfii10); rate_std_20 = rolling_std(rate_change, 20); rate_surprise = abs(rate_change) / rate_std_20`
- **Normalization**: Ratio (number of standard deviations)
- **Expected Range**: [0, 6]
- **Why it helps DOWN detection**: Captures how UNEXPECTED a rate move is. A 2-sigma upward rate move causes sharper gold selloff than a gradual drift. The regression model uses raw rate_change but not the surprise magnitude.

#### C2: rate_surprise_signed (Directional Rate Surprise)
- **Source**: Derived from C1
- **Formula**: `rate_surprise_signed = sign(rate_change) * rate_surprise`. Positive = hawkish surprise (bad for gold), negative = dovish surprise (good for gold).
- **Normalization**: Signed ratio
- **Expected Range**: [-6, 6]
- **Why it helps DOWN detection**: Separates the direction from magnitude. Large positive values (hawkish surprise) are the primary DOWN driver from rates.

#### C3: dxy_acceleration (DXY Second Derivative)
- **Source**: DX-Y.NYB
- **Formula**: `dxy_change = pct_change(dxy); dxy_accel = dxy_change - dxy_change.shift(1)`
- **Normalization**: Z-scored vs 20-day rolling window
- **Expected Range**: [-4, 4]
- **Why it helps DOWN detection**: USD acceleration (increasingly positive changes) signals intensifying headwind for gold. Unlike raw dxy_change (in regression model), acceleration captures the rate-of-change of the headwind.

### 3.4 Category D: Volume and Flow Features (2 features)

Volume divergences from price signal institutional distribution/accumulation.

#### D1: gld_volume_z (GLD Volume Z-Score)
- **Source**: GLD Volume
- **Formula**: `(volume - rolling_mean(volume, 20)) / rolling_std(volume, 20)`
- **Normalization**: Z-scored
- **Expected Range**: [-2, 6]
- **Why it helps DOWN detection**: Abnormally high volume on flat/negative days signals institutional selling. Distinct from ETF submodel's dollar-weighted capital_intensity.

#### D2: volume_return_sign (Volume-Price Agreement)
- **Source**: GLD Volume and GC=F Return
- **Formula**: `sign(gold_return) * gld_volume_z`. Negative values = high volume + negative return = bearish. Positive = high volume + positive return = bullish.
- **Normalization**: Product of sign and z-score
- **Expected Range**: [-6, 6]
- **Why it helps DOWN detection**: Classic volume-price divergence signal. When volume spikes on DOWN days (large negative values), it confirms selling pressure and predicts DOWN continuation.

### 3.5 Category E: Momentum Context Features (2 features)

Although traditional momentum indicators have zero predictive power, momentum CONTEXT (divergences, position in range) may interact with other features.

#### E1: momentum_divergence (Short vs Medium Term)
- **Source**: GC=F Close
- **Formula**: `ret_5d = pct_change(close, 5); ret_20d = pct_change(close, 20); divergence = ret_5d - ret_20d`
- **Normalization**: Z-scored vs 60-day rolling window
- **Expected Range**: [-3, 3]
- **Why it helps DOWN detection**: When short-term momentum decouples from medium-term trend (e.g., 5d return negative but 20d positive), it signals potential reversal. Not the same as RSI (which is purely internal).

#### E2: distance_from_20d_high (Position in 20-Day Range)
- **Source**: GC=F Close, High
- **Formula**: `high_20d = rolling_max(close, 20); range_20d = high_20d - rolling_min(close, 20); distance = (close - high_20d) / range_20d` (0 = at high, -1 = at low)
- **Normalization**: Already normalized to [-1, 0]
- **Expected Range**: [-1.0, 0.0]
- **Why it helps DOWN detection**: Despite researcher finding minimal standalone effect, this feature may interact with volatility features (being at 20d high DURING vol expansion is different from 20d high during low vol).

### 3.6 Category F: Calendar and Auxiliary (2 features)

Zero-cost features that may capture interactions with other signals.

#### F1: day_of_week
- **Source**: Computed from date
- **Formula**: `date.dayofweek` (0=Monday, 4=Friday)
- **Normalization**: Integer encoding (not one-hot, XGBoost handles ordinal splits)
- **Expected Range**: [0, 4]
- **Why included**: Zero-cost categorical feature. While day-of-week alone has no significance, it may interact with other features (e.g., Friday + high GVZ may behave differently).

#### F2: month_of_year
- **Source**: Computed from date
- **Formula**: `date.month` (1-12)
- **Normalization**: Integer encoding
- **Expected Range**: [1, 12]
- **Why included**: Zero-cost seasonal feature. Gold has mild seasonal patterns. XGBoost can learn calendar-based splits if they exist.

### 3.7 Feature Summary Table

| # | Name | Category | Source | Lookback | Type |
|---|------|----------|--------|----------|------|
| 1 | rv_ratio_10_30 | Volatility | GC=F | 10d/30d | Continuous |
| 2 | rv_ratio_10_30_z | Volatility | GC=F | 10d/30d/60d | Continuous |
| 3 | gvz_level_z | Volatility | FRED GVZCLS | 60d | Continuous |
| 4 | gvz_vix_ratio | Volatility | FRED GVZCLS+VIXCLS | None | Continuous |
| 5 | intraday_range_ratio | Volatility | GC=F OHLC | 20d | Continuous |
| 6 | risk_off_score | Cross-Asset | Multi-source | 20d | Continuous |
| 7 | gold_silver_ratio_change | Cross-Asset | GC=F + SI=F | 5d/60d | Continuous |
| 8 | equity_gold_beta_20d | Cross-Asset | GC=F + ^GSPC | 20d | Continuous |
| 9 | gold_copper_ratio_change | Cross-Asset | GC=F + HG=F | 5d/60d | Continuous |
| 10 | rate_surprise | Rate/Currency | FRED DFII10 | 20d | Continuous |
| 11 | rate_surprise_signed | Rate/Currency | FRED DFII10 | 20d | Continuous |
| 12 | dxy_acceleration | Rate/Currency | DX-Y.NYB | 1d/20d | Continuous |
| 13 | gld_volume_z | Volume/Flow | GLD | 20d | Continuous |
| 14 | volume_return_sign | Volume/Flow | GLD + GC=F | 20d | Continuous |
| 15 | momentum_divergence | Momentum | GC=F | 5d/20d/60d | Continuous |
| 16 | distance_from_20d_high | Momentum | GC=F | 20d | Continuous |
| 17 | day_of_week | Calendar | Date | None | Ordinal |
| 18 | month_of_year | Calendar | Date | None | Ordinal |

**Total: 18 features** (15 continuous + 2 ordinal + 1 composite)

Samples-per-feature ratio (train): ~1,925 / 18 = 107:1 (well above the 10:1 minimum, and below the 25-feature overfitting zone)

### 3.8 Features Excluded and Why

| Feature | Reason for Exclusion |
|---------|---------------------|
| RSI (14d) | Zero predictive power for gold DOWN (45.57% vs 46.56% baseline). Gold is macro-driven. |
| Consecutive streak days | Minimal effect (47.31% vs 46.56%). Not worth a feature slot. |
| Bollinger Band width | NEGATIVELY associated with DOWN (wide bands = UP bias). Would confuse classifier. |
| VIX regime (categorical) | No significance (chi-sq p=0.828). VIX change captured via risk_off_score instead. |
| VIX acceleration | Redundant with risk_off_score which already includes z-scored VIX change. |
| GVZ/VIX ratio CHANGE | Weak correlation (+0.04) and counterintuitive direction. Ratio LEVEL (A4) is sufficient. |

---

## 4. Model Architecture

### 4.1 Primary Model: XGBoost Binary Classifier

```
Input: 18-dimensional feature vector
  |
  v
XGBoost Classifier
  - Objective: binary:logistic (default) or custom focal loss
  - Output: P(UP) probability [0, 1]
  |
  v
Derived Output:
  - P(DOWN) = 1 - P(UP)
  - predicted_direction = 1 if P(UP) > 0.5, else 0
```

### 4.2 Why XGBoost

1. **Same framework as regression model**: Consistent methodology, well-understood behavior.
2. **Handles mixed feature types**: Ordinal calendar features + continuous features natively.
3. **Built-in regularization**: max_depth, min_child_weight, colsample_bytree prevent overfitting on 18 features.
4. **Feature importance**: Gain-based importance reveals which features contribute to DOWN detection.
5. **Fast training**: 100 Optuna trials in <15 minutes on Kaggle.

### 4.3 Loss Function Selection

**Primary: Weighted binary cross-entropy (scale_pos_weight)**

Given the mild 53/47 imbalance, standard binary:logistic with scale_pos_weight is the appropriate starting point. The imbalance ratio is 1.13:1, far below the threshold where focal loss provides significant benefit.

```python
# scale_pos_weight = n_negative / n_positive
# Since we want to EMPHASIZE the DOWN class (label=0), and XGBoost's
# scale_pos_weight scales the POSITIVE class:
# We want DOWN (0) to matter more -> make scale_pos_weight < 1
# OR equivalently, set sample_weight manually for DOWN class
#
# For Optuna search: let scale_pos_weight range [0.8, 2.0]
# Values < 1.0 = emphasize DOWN class
# Values > 1.0 = emphasize UP class (default behavior)
# 1.0 = balanced
```

**Fallback: Custom Focal Loss (if WCE achieves DOWN recall < 25%)**

```python
def focal_loss(gamma):
    def custom_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
        grad = p - y_true  # standard gradient
        # Focal modulation: (1 - p_t)^gamma
        p_t = y_true * p + (1 - y_true) * (1 - p)
        focal_weight = (1 - p_t) ** gamma
        grad = focal_weight * grad
        hess = focal_weight * p * (1 - p)  # approximate Hessian
        hess = np.maximum(hess, 1e-7)  # numerical stability
        return grad, hess
    return custom_obj
```

The decision between WCE and focal loss will be made WITHIN the Optuna search: one hyperparameter (`use_focal`) selects the loss, and `focal_gamma` is only active when focal is selected.

---

## 5. Hyperparameter Specification

### 5.1 Fixed Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective | binary:logistic | Binary classification |
| eval_metric | logloss | Standard for binary classification |
| tree_method | hist | Fast histogram-based |
| verbosity | 0 | Suppress output |
| seed | 42 + trial.number | Reproducible |
| early_stopping_rounds | 50 | Prevent overfitting without excessive training |

### 5.2 Optuna Search Space

| Parameter | Range | Scale | Rationale |
|-----------|-------|-------|-----------|
| max_depth | [2, 5] | int | Shallow trees for small sample. 2=linear interactions, 5=moderate complexity |
| n_estimators | [100, 500] | int step 50 | Cap at 500 (early stopping further limits) |
| learning_rate | [0.005, 0.1] | log | Lower bound for fine-grained learning |
| min_child_weight | [5, 20] | int | Minimum samples per leaf. Higher = more conservative |
| subsample | [0.5, 0.95] | linear | Row subsampling for regularization |
| colsample_bytree | [0.4, 0.9] | linear | Column subsampling. With 18 features, 0.4 = use ~7 per tree |
| gamma | [0, 0.5] | linear | Minimum split loss. Prevents trivial splits |
| reg_lambda | [0.5, 10.0] | log | L2 regularization. Moderate to strong |
| reg_alpha | [0.0, 5.0] | log(1+x) | L1 regularization. Promotes feature sparsity |
| scale_pos_weight | [0.7, 1.8] | linear | Class weight. <1 emphasizes DOWN, >1 emphasizes UP |
| use_focal | [True, False] | categorical | Whether to use focal loss |
| focal_gamma | [0.5, 3.0] | linear | Focal loss gamma (only if use_focal=True) |

**Total: 12 hyperparameters** (10 always active + 2 conditional on use_focal)

### 5.3 Search Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| n_trials | 100 | Sufficient for 12-HP space with TPE |
| timeout | 3600 sec (1 hour) | Conservative; 100 trials should finish in ~15 min |
| sampler | TPESampler(seed=42) | Reproducible, efficient |
| pruner | MedianPruner(n_startup_trials=10) | Kill bad trials early to explore more |
| direction | maximize | Higher objective = better |

### 5.4 Optuna Objective Function

The objective maximizes a composite metric that balances DOWN recall (primary goal) with overall classifier quality:

```python
def objective(trial):
    params = suggest_params(trial)
    model = train_xgb(params, X_train, y_train, X_val, y_val)
    y_val_prob = model.predict_proba(X_val)[:, 0]  # P(DOWN)
    y_val_pred = (y_val_prob > 0.5).astype(int)  # 1=DOWN, 0=UP (inverted)

    # Actually: P(UP) from XGBoost, then derive
    y_val_pred_up = model.predict(dval)  # probability of UP (label=1)
    y_val_pred_direction = (y_val_pred_up > 0.5).astype(int)

    # Metrics
    balanced_acc = balanced_accuracy_score(y_val, y_val_pred_direction)
    down_recall = recall_score(y_val, y_val_pred_direction, pos_label=0)
    down_precision = precision_score(y_val, y_val_pred_direction, pos_label=0)
    f1_down = f1_score(y_val, y_val_pred_direction, pos_label=0)
    roc_auc = roc_auc_score(y_val, y_val_pred_up)

    # Composite objective:
    # 40% DOWN F1 (primary goal)
    # 30% ROC-AUC (overall discrimination)
    # 30% Balanced accuracy
    composite = 0.40 * f1_down + 0.30 * roc_auc + 0.30 * balanced_acc

    return composite
```

**Weight rationale**:
- 40% DOWN F1: The entire purpose of the classifier is DOWN detection. F1 balances recall and precision.
- 30% ROC-AUC: Overall discrimination ability. Prevents the model from achieving high DOWN F1 by predicting DOWN for everything.
- 30% Balanced accuracy: Ensures the model works for both classes.

---

## 6. Training Configuration

### 6.1 Training Algorithm

```
1. DATA PREPARATION:
   a. Fetch raw data from yfinance and FRED
   b. Forward-fill missing values (max 5d for FRED, 3d for yfinance)
   c. Compute all 18 features using formulas in Section 3
   d. Create target: binary UP=1/DOWN=0 from next-day gold return
   e. Drop rows with any NaN (warmup period for rolling windows)
   f. Split into train (70%), val (15%), test (15%) by date

2. FEATURE VALIDATION:
   a. Print feature statistics (mean, std, min, max, NaN count)
   b. Print feature correlations (check no pair > 0.85)
   c. Print class balance per split
   d. Assert no NaN in any feature

3. OPTUNA HPO (100 trials):
   a. For each trial: suggest params, train XGBoost, evaluate on val
   b. Objective: composite (40% DOWN_F1 + 30% AUC + 30% balanced_acc)
   c. Pruning: stop bad trials early
   d. Report best params and best composite score

4. RETRAIN BEST MODEL:
   a. Train with best params on full train set
   b. Evaluate on val (confirmation) and test (final)
   c. Report all metrics for both splits

5. THRESHOLD OPTIMIZATION:
   a. Get P(DOWN) on validation set from best model
   b. Grid search threshold [0.40, 0.70, step=0.01] for ensemble
   c. For each threshold: compute ensemble DA, Sharpe on val set
   d. Select threshold maximizing: val_sharpe + 0.5*(val_da - 0.50)
   e. Apply selected threshold to test set for final evaluation

6. FEATURE IMPORTANCE:
   a. Extract gain-based importance from best model
   b. Rank all 18 features by importance
   c. Report top 5 and bottom 3

7. SAVE RESULTS:
   a. classifier.csv: Date, p_up, p_down, predicted_direction
   b. training_result.json: all metrics, params, feature importance
   c. model.json: XGBoost model (save_model format)
```

### 6.2 Early Stopping

- Metric: logloss on validation set
- Patience: 50 rounds
- Maximum rounds: Optuna-controlled (100-500)

---

## 7. Ensemble Design

### 7.1 Ensemble Strategy: Threshold-Based DOWN Override

The classifier's P(DOWN) is used to override the regression model's direction when the classifier is confident a DOWN day is coming.

```python
def ensemble_predict(regression_pred, p_down, threshold):
    """
    regression_pred: regression model's return prediction (float, %)
    p_down: classifier's P(DOWN) [0, 1]
    threshold: optimized on validation set

    Returns: +1 (UP) or -1 (DOWN)
    """
    if p_down > threshold:
        return -1  # Classifier overrides to DOWN
    elif regression_pred > 0:
        return +1  # Regression says UP
    else:
        return -1  # Regression says DOWN
```

### 7.2 Threshold Tuning Procedure

1. Load regression model predictions on validation set (from meta_model attempt 7)
2. Load classifier P(DOWN) on validation set
3. For each threshold in [0.40, 0.70, step=0.01]:
   a. Apply ensemble_predict to all validation samples
   b. Compute direction accuracy vs actual next-day direction
   c. Compute Sharpe ratio (sign-based strategy with 5bps cost)
   d. Compute composite: `sharpe + 0.5 * (da - 0.50)`
4. Select threshold with highest composite score
5. Apply to test set (once, no re-tuning)

### 7.3 Fallback: Weighted Probability Combination

If threshold override sacrifices too many UP days (UP accuracy drops below 80%):

```python
def ensemble_weighted(regression_pred, p_up, alpha=0.5):
    """
    alpha: weight on regression (tuned on validation)
    """
    reg_score = np.clip(regression_pred, -2.0, 2.0) / 2.0  # Scale to [-1, 1]
    clf_score = 2 * p_up - 1  # [0,1] -> [-1, 1]
    final_score = alpha * reg_score + (1 - alpha) * clf_score
    return np.sign(final_score)
```

Tune alpha in [0.3, 0.7, step=0.05] on validation set.

### 7.4 Regression Model Predictions Access

The ensemble requires meta-model attempt 7 predictions. These are loaded from the Kaggle dataset or computed on-the-fly using the meta-model's saved parameters.

**Approach**: The classifier notebook will load the regression meta-model's submodel_output.csv from the Kaggle dataset (bigbigzabuton/gold-prediction-submodels or gold-prediction-complete). If regression predictions are not available as a pre-saved file, the notebook will train a minimal regression model using the same 24 features and attempt 7's best parameters (hardcoded as fallback).

### 7.5 Ensemble Output

```python
ensemble_output = pd.DataFrame({
    'Date': dates,
    'regression_pred': regression_predictions,   # from meta-model
    'classifier_p_down': p_down,                 # from classifier
    'classifier_p_up': p_up,                     # from classifier
    'ensemble_direction': ensemble_directions,    # +1 or -1
    'ensemble_method': 'threshold',              # or 'weighted'
    'threshold': optimal_threshold,
})
```

---

## 8. Evaluation Criteria

### 8.1 Standalone Classifier Metrics (on test set)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Balanced Accuracy | > 52% | Above random (50%), accounting for class imbalance |
| DOWN Recall | > 30% | Must catch significantly more DOWN days than regression-only (~10%) |
| DOWN Precision | > 42% | When predicting DOWN, right at least 42% (above baseline DOWN rate ~47% would be ideal) |
| DOWN F1 Score | > 0.35 | Harmonic mean ensures both recall and precision are reasonable |
| ROC-AUC | > 0.52 | Overall discrimination above random |

### 8.2 Ensemble Metrics (on test set, vs regression-only)

| Metric | Target | Current (Regression-Only) |
|--------|--------|--------------------------|
| Direction Accuracy | > 60.5% (+0.5pp) | 60.04% |
| Sharpe Ratio | > 2.0 | 2.46 |
| DOWN Day Capture | > 25% | ~10% (estimated) |
| UP Day Accuracy | > 85% | ~96% |
| Net DA Improvement | > 0 | (new metric) |

### 8.3 Pass/Fail Decision Rules

| Outcome | Decision |
|---------|----------|
| Ensemble DA > 60.5% AND Sharpe > 2.0 | PASS -- adopt classifier |
| Ensemble DA > 60.0% AND DOWN capture > 25% AND Sharpe > 1.8 | CONDITIONAL PASS -- acceptable tradeoff |
| Ensemble DA < 60.0% (regression worse) | FAIL -- abandon classifier |
| Ensemble Sharpe < 1.5 (too much turnover) | FAIL -- threshold too aggressive |
| Standalone DOWN F1 < 0.25 | FAIL -- classifier has no skill, ensemble cannot help |

### 8.4 Diagnostic Checks

| Check | Threshold | Purpose |
|-------|-----------|---------|
| Train-val balanced accuracy gap | < 8pp | Overfitting control |
| Feature importance concentration | Top feature < 30% | No single feature dominance |
| P(DOWN) distribution | std > 0.05 | Not collapsed to constant |
| Prediction balance | DOWN predictions between 20%-60% | Not degenerate |
| Overfit ratio (train/val logloss) | < 1.5 | Standard overfitting check |

---

## 9. Data Pipeline (Feature Engineering Order)

```
Step 1: Fetch raw data
  - yfinance: GC=F (OHLCV), GLD (OHLCV), SI=F, HG=F, DX-Y.NYB, ^GSPC
  - FRED: GVZCLS, VIXCLS, DFII10, DGS10
  - Date range: 2014-01-01 to present (1 year extra for warmup)

Step 2: Align dates and forward-fill
  - Merge all on Date (inner join for yfinance, left join for FRED with ffill)
  - Forward-fill FRED gaps (max 5d)
  - Drop any remaining NaN from alignment

Step 3: Compute gold return and target
  - gold_return = pct_change(GC=F Close) * 100
  - target = (gold_return.shift(-1) > 0).astype(int)  # 1=UP, 0=DOWN

Step 4: Category A features (Volatility)
  - rv_ratio_10_30: std(gold_return, 10) / std(gold_return, 30)
  - rv_ratio_10_30_z: z-score of rv_ratio vs 60d rolling
  - gvz_level_z: z-score of GVZCLS vs 60d rolling
  - gvz_vix_ratio: GVZCLS / VIXCLS
  - intraday_range_ratio: (H-L)/C / rolling_mean((H-L)/C, 20)

Step 5: Category B features (Cross-Asset)
  - risk_off_score: composite of z-scored changes (VIX, DXY, SPX, 10Y)
  - gold_silver_ratio_change: z-score of (gold_5d_ret - silver_5d_ret)
  - equity_gold_beta_20d: rolling cov/var
  - gold_copper_ratio_change: z-score of (gold_5d_ret - copper_5d_ret)

Step 6: Category C features (Rate/Currency)
  - rate_surprise: abs(rate_change) / rolling_std(rate_change, 20)
  - rate_surprise_signed: sign(rate_change) * rate_surprise
  - dxy_acceleration: z-score of (dxy_change - dxy_change.shift(1))

Step 7: Category D features (Volume)
  - gld_volume_z: z-score of GLD volume vs 20d rolling
  - volume_return_sign: sign(gold_return) * gld_volume_z

Step 8: Category E features (Momentum Context)
  - momentum_divergence: z-score of (5d_ret - 20d_ret) vs 60d rolling
  - distance_from_20d_high: (close - rolling_max(close, 20)) / rolling_range(close, 20)

Step 9: Category F features (Calendar)
  - day_of_week: date.dayofweek (0-4)
  - month_of_year: date.month (1-12)

Step 10: Drop warmup rows (first ~60 trading days for 60d rolling windows)
Step 11: Drop rows where target is NaN (last row)
Step 12: Verify: 18 features, 0 NaN, ~2,700 rows
Step 13: Split into train/val/test (70/15/15 by date order)
```

---

## 10. Notebook Structure (Cell-by-Cell)

### Cell 0: Markdown Header
```
# Gold DOWN Classifier - Attempt 1
Architecture: XGBoost binary:logistic + optional focal loss
Features: 18 NEW features (different from regression model's 24)
Purpose: Detect DOWN days to ensemble with regression meta-model (attempt 7)
```

### Cell 1: Imports and Configuration
- torch (not needed, but kept for consistency), xgboost, optuna, sklearn, pandas, numpy, etc.
- pip install xgboost optuna (if not pre-installed on Kaggle)
- Configuration constants: RANDOM_SEED, N_TRIALS, TIMEOUT, TRAIN_RATIO, VAL_RATIO

### Cell 2: Data Fetching
- yfinance downloads: GC=F, GLD, SI=F, HG=F, DX-Y.NYB, ^GSPC
- FRED downloads: GVZCLS, VIXCLS, DFII10, DGS10
- Merge into single DataFrame

### Cell 3: Feature Engineering
- Compute all 18 features per Section 3 formulas
- Drop warmup rows
- Create target variable

### Cell 4: Data Validation
- Print feature statistics
- Check for NaN
- Print class balance
- Print feature correlations
- Assert no pair > 0.85 correlation

### Cell 5: Train/Val/Test Split
- Time-series split (70/15/15)
- Print split boundaries and sizes

### Cell 6: XGBoost + Optuna HPO
- Define suggest_params function (per Section 5.2)
- Define objective function (per Section 5.4)
- Run 100 trials with TPE sampler
- Report best params and score

### Cell 7: Train Final Model
- Retrain with best params on train set
- Evaluate on val and test
- Print all standalone classifier metrics

### Cell 8: Feature Importance
- Extract gain-based importance
- Print ranked feature list
- Plot feature importance bar chart

### Cell 9: Threshold Optimization
- Load regression model predictions (from meta_model attempt 7)
  - Try loading from Kaggle dataset first
  - Fallback: train minimal regression model with hardcoded params
- Grid search threshold on validation set
- Report optimal threshold and val metrics

### Cell 10: Ensemble Evaluation (Test Set)
- Apply optimal threshold to test set
- Compute ensemble DA, Sharpe, DOWN capture, UP accuracy
- Compare vs regression-only baseline
- Print pass/fail assessment

### Cell 11: Save Results
- Save classifier.csv (Date, p_up, p_down, predicted_direction)
- Save training_result.json (comprehensive metrics)
- Save model via xgb.save_model('model.json')

### Cell 12: Diagnostic Plots
- P(DOWN) distribution histogram
- ROC curve
- Confusion matrix
- Feature importance chart
- Ensemble vs regression-only comparison

### Expected Runtime
- Data fetching: 2-3 min
- Feature engineering: 1 min
- Optuna 100 trials: 10-15 min
- Final training + evaluation: 2 min
- **Total: ~20 minutes (CPU sufficient)**

---

## 11. Kaggle Execution Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| enable_gpu | false | XGBoost with tree_method=hist on <3K rows is fast on CPU |
| Estimated execution time | 20-25 minutes | 100 Optuna trials on small dataset |
| Estimated memory usage | 0.5 GB | Tiny dataset, no neural networks |
| Required pip packages | [imbalanced-learn] | For focal loss if needed (optional) |
| Internet required | true | For data fetching (yfinance, FRED) |
| dataset_sources | ["bigbigzabuton/gold-prediction-submodels"] | For loading regression model predictions |

### kernel-metadata.json

```json
{
  "id": "bigbigzabuton/gold-classifier-attempt-1",
  "title": "Gold DOWN Classifier - Attempt 1",
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

## 12. Implementation Instructions

### 12.1 For builder_data

No separate data preparation step needed. The classifier notebook is self-contained and fetches all data directly via yfinance and FRED API.

However, the regression model predictions (meta_model attempt 7) must be accessible for the ensemble step. Two options:
1. **Preferred**: Upload meta_model_7 predictions as a CSV to the Kaggle dataset
2. **Fallback**: The notebook will train a minimal regression model inline with hardcoded attempt 7 params

### 12.2 For builder_model

**Task**: Generate `notebooks/classifier_1/train.ipynb`

#### Key Implementation Details:

1. **Feature engineering must be deterministic and vectorized** -- use pandas rolling/shift operations. No loops.

2. **Z-score function template** (reusable for multiple features):
```python
def rolling_zscore(series, window=60):
    """Z-score vs rolling window. Returns NaN for warmup period."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.clip(lower=1e-8)
```

3. **Focal loss implementation** (if use_focal=True in trial):
```python
def focal_loss_obj(gamma):
    def obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p_t = y_true * p + (1 - y_true) * (1 - p)
        alpha_t = y_true * 0.5 + (1 - y_true) * 0.5  # balanced
        grad = alpha_t * (1 - p_t)**gamma * (p - y_true)
        hess = alpha_t * (1 - p_t)**gamma * p * (1 - p)
        hess = np.maximum(hess, 1e-7)
        return grad, hess
    return obj
```

4. **Equity-gold beta computation**:
```python
def rolling_beta(gold_ret, spx_ret, window=20):
    cov = gold_ret.rolling(window).cov(spx_ret)
    var = spx_ret.rolling(window).var()
    return cov / var.clip(lower=1e-8)
```

5. **Ensemble threshold optimization**:
```python
# Load regression predictions
# Option 1: From dataset
# Option 2: Hardcode attempt 7 params and train inline
regression_preds = load_or_compute_regression_predictions()

thresholds = np.arange(0.40, 0.71, 0.01)
best_composite = -np.inf
best_threshold = 0.55  # default

for thresh in thresholds:
    ensemble_dir = np.where(p_down_val > thresh, -1,
                            np.where(regression_preds_val > 0, 1, -1))
    val_da = (ensemble_dir == np.sign(actual_returns_val)).mean()
    val_sharpe = compute_sharpe(ensemble_dir, actual_returns_val)
    composite = val_sharpe + 0.5 * (val_da - 0.50)
    if composite > best_composite:
        best_composite = composite
        best_threshold = thresh
```

6. **training_result.json structure**:
```python
training_result = {
    "feature": "classifier",
    "attempt": 1,
    "timestamp": datetime.now().isoformat(),
    "architecture": "XGBoost binary:logistic",
    "n_features": 18,
    "best_params": best_params,
    "standalone_metrics": {
        "train": {"balanced_acc": ..., "down_recall": ..., "down_precision": ..., "f1_down": ..., "roc_auc": ...},
        "val": {"balanced_acc": ..., "down_recall": ..., "down_precision": ..., "f1_down": ..., "roc_auc": ...},
        "test": {"balanced_acc": ..., "down_recall": ..., "down_precision": ..., "f1_down": ..., "roc_auc": ...},
    },
    "ensemble_metrics": {
        "optimal_threshold": ...,
        "val": {"da": ..., "sharpe": ..., "down_capture": ..., "up_accuracy": ...},
        "test": {"da": ..., "sharpe": ..., "down_capture": ..., "up_accuracy": ...},
    },
    "regression_only_metrics": {
        "test": {"da": 0.6004, "sharpe": 2.46},
    },
    "feature_importance": {
        "ranked": [...],
        "top5": [...],
    },
    "class_balance": {"train_up_pct": ..., "train_down_pct": ...},
    "p_down_distribution": {"mean": ..., "std": ..., "min": ..., "max": ...},
}
```

7. **classifier.csv output format**:
```python
classifier_output = pd.DataFrame({
    'Date': all_dates,        # YYYY-MM-DD format
    'p_up': p_up_all,         # P(UP) probability [0, 1]
    'p_down': p_down_all,     # P(DOWN) probability [0, 1]
    'predicted_direction': predicted_dir_all,  # 1=UP, 0=DOWN
})
classifier_output.to_csv('classifier.csv', index=False)
```

### 12.3 Notebook Validation Checklist

Before submission, validate_notebook.py must check:
- [ ] No syntax errors
- [ ] No typos in method names
- [ ] dataset_sources includes "bigbigzabuton/gold-prediction-submodels"
- [ ] kernel-metadata.json is valid
- [ ] enable_gpu is false (CPU sufficient)
- [ ] FRED_API_KEY accessed via os.environ (Kaggle Secrets)
- [ ] No hardcoded API keys
- [ ] Target variable uses shift(-1) for next-day prediction (no leakage)
- [ ] Train/val/test split is time-series ordered (no shuffle)

---

## 13. Risk Mitigation

### Risk 1: Classifier Has No Skill (HIGH -- 30-40% probability)

**Scenario**: With only ~2,750 samples and 18 features, the classifier may achieve balanced accuracy barely above 50%. Gold direction is inherently noisy, and even the best features (rv_ratio_10_30 at +5pp) provide weak signals.

**Mitigation**:
- Optuna explores 100 different configurations including focal loss.
- Feature selection within XGBoost (colsample_bytree) naturally prunes weak features.
- If standalone F1 < 0.25, the ensemble step is skipped and the classifier is abandoned.
- Fallback: The regression model continues as-is (no harm done).

### Risk 2: Ensemble Hurts Rather Than Helps (MODERATE -- 25-35% probability)

**Scenario**: The classifier's DOWN overrides miss-classify UP days more often than they correctly catch DOWN days, reducing ensemble DA below regression-only.

**Mitigation**:
- Threshold tuned on validation set to maximize composite (not just DOWN recall).
- High threshold (0.55-0.65) ensures only high-confidence DOWN overrides.
- Fallback threshold = 1.0 disables override entirely, degrading gracefully to regression-only.
- Weighted probability combination as secondary approach if threshold is too aggressive.

### Risk 3: Feature Overlap with Regression Model (LOW -- 10-15% probability)

**Scenario**: Despite using different transformations, some classifier features may be highly correlated with the regression model's 24 features, providing no new information for DOWN detection.

**Mitigation**:
- Features deliberately chosen to capture DIFFERENT aspects: reversal signals (not regime states), shock magnitudes (not levels), cross-asset synchronization (not individual asset changes).
- rv_ratio_10_30 is genuinely novel (no realized vol ratio in regression model).
- rate_surprise captures shock magnitude (regression uses raw rate change).
- Correlation check in Cell 4 validates orthogonality.

### Risk 4: Overfitting on Small Validation Set (MODERATE -- 20-30% probability)

**Scenario**: Threshold optimization on ~413 validation samples may overfit to validation period idiosyncrasies. Optimal threshold from val may not generalize to test.

**Mitigation**:
- Coarse threshold grid (0.01 step) limits overfitting risk (only 31 candidates).
- Composite metric (Sharpe + DA) is more robust than single metric optimization.
- Report both val and test metrics for comparison.
- If val-test gap on ensemble DA > 5pp, flag as potential overfitting.

### Risk 5: FRED Data Unavailable in Kaggle (LOW -- 5% probability)

**Scenario**: FRED_API_KEY not configured in Kaggle Secrets, causing data fetching failure.

**Mitigation**:
- Notebook checks FRED_API_KEY existence at startup, fails fast with clear error message.
- FRED_API_KEY has been configured for all previous submodel runs (confirmed working).

---

## 14. Comparison with Failed Classification Attempt

The entrance requirements document mentions a prior failed XGBoost classifier using the SAME 24 regression features. This design addresses every identified failure mode:

| Failure Mode | Prior Attempt | This Design |
|--------------|--------------|-------------|
| P(UP) range 0.46-0.51 for ALL samples | Same 24 features | 18 NEW features designed for reversal detection |
| Zero discrimination | Features predict magnitude, not direction | Features capture vol shifts, stress, shocks |
| Regime probs too slow-moving | HMM regime outputs (binary-ish) | Continuous z-scores, ratios, second derivatives |
| Z-scores symmetric around 0 | Mean-reversion z-scores | Directional surprise, signed rate shock |
| No reversal signals | No momentum exhaustion features | Vol expansion ratio, momentum divergence, range position |
| No cross-asset synchronization | Individual asset features | Composite risk-off score, multi-asset beta |

---

## 15. Expected Outcomes

### Standalone Classifier

| Metric | Expected Range | Confidence |
|--------|---------------|------------|
| Balanced Accuracy | 51-54% | Medium |
| DOWN Recall | 25-40% | Medium |
| DOWN Precision | 42-50% | Medium |
| DOWN F1 | 0.30-0.44 | Medium |
| ROC-AUC | 0.51-0.55 | Medium |

### Ensemble vs Regression-Only

| Metric | Regression-Only | Expected Ensemble | Confidence |
|--------|----------------|-------------------|------------|
| DA | 60.04% | 60.5-62.5% | Medium |
| Sharpe | 2.46 | 2.0-2.5 | Medium-High |
| DOWN Capture | ~10% | 25-40% | Medium |
| UP Accuracy | ~96% | 85-95% | High |

### Probability of Outcomes

| Outcome | Probability |
|---------|------------|
| Ensemble DA > 60.5% AND Sharpe > 2.0 (PASS) | 35-45% |
| Ensemble DA > 60.0% AND DOWN capture > 25% (CONDITIONAL PASS) | 55-65% |
| Classifier has no skill (standalone F1 < 0.25) | 15-25% |
| Ensemble hurts (DA < 60.0%) | 15-20% |

---

**End of Design Document**

**Architect**: architect (Opus)
**Date**: 2026-02-18
**Based on**: Research report `docs/research/classifier_features_research.md` + entrance requirements `shared/current_task.json`
**Context**: Regression meta-model attempt 7 (DA 60.04%, HCDA 64.13%, Sharpe 2.46) has 96% bullish bias. This classifier aims to improve DOWN-day detection via parallel ensemble.
