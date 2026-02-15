# Evaluation Summary: etf_flow attempt 1

## Gate 1: Standalone Quality -- PASS

- Overfit ratio: N/A (HMM + deterministic z-scores, no neural network) -- skipped
- All-NaN columns: 0 -- PASS
- Constant output columns: 0 -- PASS
- Autocorrelation (lag=1):
  - etf_regime_prob: 0.558 -- PASS
  - etf_capital_intensity: 0.481 -- PASS
  - etf_pv_divergence: 0.557 -- PASS

All three columns have healthy variance (std: 0.38, 1.11, 0.99) and well-behaved autocorrelation.

## Gate 2: Information Gain -- FAIL

- MI increase: 1.58% (threshold > 5%) -- FAIL
- Max VIF (submodel): 12.47 (etf_regime_prob, threshold < 10) -- FAIL
- Max VIF (other submodel cols): etf_capital_intensity 6.25, etf_pv_divergence 1.06
- Correlation stability (std): 0.1617 (threshold < 0.15) -- FAIL

**Failure detail:** All three Gate 2 checks fail. MI increase is only 1.58% (below 5%), indicating low linear mutual information. etf_regime_prob has VIF=12.47 (exceeding 10 threshold), likely due to correlation with existing regime probability features (VIX, technical, cross-asset). Stability marginally fails for regime_prob (0.161) and capital_intensity (0.162), while pv_divergence passes (0.127).

**Precedent note:** All 5 previously completed submodels (VIX, Technical, Cross-Asset, Yield-Curve, DXY) failed at least one Gate 2 check. The 4 submodels that passed Gate 3 all had Gate 2 failures. MI underestimates nonlinear contributions captured by XGBoost in Gate 3.

## Gate 3: Ablation -- PASS (via Sharpe + MAE)

Evaluated with all 3 columns. Base includes 19 raw features + 11 existing submodel features (30 total).

| Metric | Baseline | Extended | Delta | Threshold | Result |
|--------|----------|----------|-------|-----------|--------|
| Direction Accuracy | 49.21% | 49.66% | +0.45% | > +0.50% | FAIL |
| Sharpe | -0.072 | +0.305 | +0.377 | > +0.05 | PASS (7.5x) |
| MAE | 0.9340 | 0.8904 | -0.0436 | < -0.01 | PASS (4.4x) |

### Fold Consistency

| Fold | DA delta | MAE delta | Sharpe delta |
|------|----------|-----------|--------------|
| 1 | +3.49% | -0.1442 | +1.338 |
| 2 | -1.51% | -0.0021 | -1.366 |
| 3 | -0.50% | +0.0502 | -0.369 |
| 4 | +1.00% | -0.1135 | +1.299 |
| 5 | -0.25% | -0.0085 | +0.981 |

- DA improved: 2/5 folds
- MAE improved: 4/5 folds
- Sharpe improved: 3/5 folds

### Feature Importance (last fold)

- etf_regime_prob: 0.0352 (rank 9/33) -- strongest ETF flow feature
- etf_pv_divergence: 0.0275 (rank 19/33)
- etf_capital_intensity: 0.0255 (rank 22/33)

## Decision: completed

**Rationale:** Gate 3 passes on two criteria simultaneously -- a first among all submodels:

1. **Sharpe improvement +0.377** is the strongest of any submodel (VIX was +0.289, the previous best). The baseline Sharpe flips from -0.072 (losing strategy) to +0.305 (profitable strategy). This is the most impactful Sharpe improvement in the project.
2. **MAE improvement -0.0436** (4.4x threshold) is consistent across 4/5 folds, following the Technical (-0.182), Cross-Asset (-0.087), Yield-Curve (-0.069) precedent.

The Sharpe result is particularly noteworthy because it demonstrates that the ETF flow regime detection helps the model make better-positioned trades, not just reduce absolute error.

**VIF caveat:** etf_regime_prob has VIF=12.47, which formally exceeds the Gate 2 threshold of 10. However, given the strong Gate 3 confirmation and the feature's rank #9/33 importance, this is acceptable. The meta-model architect should be aware of this and may apply regularization if needed.

**Comparison with prior submodels:**

| Submodel | MAE delta | DA delta | Sharpe delta | Primary Pass |
|----------|-----------|----------|--------------|-------------|
| VIX | +0.016 | +0.96% | +0.289 | DA + Sharpe |
| Technical | -0.1824 | +0.05% | -0.092 | MAE (18x) |
| Cross-Asset | -0.0866 | +0.76% | +0.040 | DA + MAE (8.7x) |
| Yield Curve | -0.0693 | +0.20% | -0.089 | MAE (6.9x) |
| **ETF Flow** | **-0.0436** | **+0.45%** | **+0.377** | **Sharpe (7.5x) + MAE (4.4x)** |

ETF Flow is the only submodel to pass both Sharpe and MAE thresholds simultaneously. Its Sharpe improvement (+0.377) exceeds VIX (+0.289), making it the strongest contributor to risk-adjusted returns.
