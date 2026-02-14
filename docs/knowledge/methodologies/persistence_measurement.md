# Persistence Measurement Methodologies

## Q1: What is persistence and why does it matter?

**Answer**: Persistence measures whether a time series exhibits trending behavior (momentum) or mean-reversion.

**Financial Interpretation**:
- **Persistent** (H > 0.5): Trends continue, momentum exists
- **Random Walk** (H = 0.5): No memory, Brownian motion
- **Mean-Reverting** (H < 0.5): Tends to return to mean

**Why It Matters for Gold Prediction**:
- If real rates show persistence, recent trends likely continue
- If mean-reverting, recent spikes/drops likely reverse
- Submodel can learn regime-dependent persistence patterns

**Last Verified**: 2026-02-14

---

## Q2: What are the main methods to measure persistence?

**Answer**: Three main methods:
1. Hurst Exponent (H)
2. Variance Ratio Test
3. Autocorrelation Analysis

**Quick Comparison**:

| Method | Range | Interpretation | PyTorch-Friendly | Robustness |
|--------|-------|----------------|------------------|------------|
| Hurst Exponent | [0, 1] | H>0.5 = persistent | Medium | Low (sensitive to params) |
| Variance Ratio | [0, ∞) | VR>1 = persistent | High | High |
| Autocorrelation | [-1, 1] | AC>0 = persistent | Very High | High |

**Recommendation for Smoke Test**: Autocorrelation (simplest, most robust)

**Last Verified**: 2026-02-14

---

## Q3: How do I calculate the Hurst Exponent?

**Answer**: Multiple methods exist. R/S (Rescaled Range) is most common.

**Simple Implementation** (R/S method):
```python
import numpy as np

def hurst_exponent(series, max_lag=20):
    """
    Calculate Hurst exponent using R/S method
    Args:
        series: 1D array of values
        max_lag: maximum lag for calculation
    Returns:
        H: Hurst exponent
    """
    lags = range(2, max_lag)
    tau = [np.std(np.diff(series, lag)) for lag in lags]

    # Linear fit in log-log space
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] * 2.0

    return hurst
```

**Using hurst library**:
```python
# pip install hurst
from hurst import compute_Hc

H, c, data = compute_Hc(series, kind='price', simplified=True)
```

**Rolling Window Implementation**:
```python
import pandas as pd

# Rolling 60-day Hurst exponent
df['hurst_60d'] = df['value'].rolling(60).apply(
    lambda x: hurst_exponent(x.values, max_lag=20)
)
```

**Limitations**:
- ⚠️ Results sensitive to max_lag parameter choice
- ⚠️ Different implementations yield inconsistent results on real financial data
- ⚠️ Works well on synthetic data but diverges on actual time series

**Recommendation**: Use for Attempt 2+, not smoke test. Prefer autocorrelation for initial testing.

**Last Verified**: 2026-02-14

---

## Q4: How do I use autocorrelation to measure persistence?

**Answer**: Compute autocorrelation at multiple lags on rolling windows.

**Simple Implementation**:
```python
import pandas as pd
import numpy as np

# Single lag autocorrelation
def autocorr_lag_k(series, k=1):
    """Autocorrelation at lag k"""
    return series.autocorr(lag=k)

# Rolling autocorrelation
df['ac_lag1_60d'] = df['value'].rolling(60).apply(
    lambda x: x.autocorr(lag=1)
)
df['ac_lag5_60d'] = df['value'].rolling(60).apply(
    lambda x: x.autocorr(lag=5)
)
df['ac_lag10_60d'] = df['value'].rolling(60).apply(
    lambda x: x.autocorr(lag=10)
)

# Composite persistence score (mean of lags 1, 5, 10)
df['persistence_score'] = df[['ac_lag1_60d', 'ac_lag5_60d', 'ac_lag10_60d']].mean(axis=1)
```

**PyTorch Implementation**:
```python
import torch

def autocorrelation(x, lag=1):
    """
    Calculate autocorrelation at given lag
    Args:
        x: torch.Tensor of shape (N,)
        lag: int, lag for autocorrelation
    Returns:
        ac: float, autocorrelation coefficient
    """
    N = len(x)
    if N <= lag:
        return 0.0

    x_mean = x.mean()
    x_centered = x - x_mean

    numerator = (x_centered[lag:] * x_centered[:-lag]).sum()
    denominator = (x_centered ** 2).sum()

    return (numerator / denominator).item()
```

**Interpretation**:
- **AC > 0**: Persistence (positive values follow positive values)
- **AC ≈ 0**: Random walk (no memory)
- **AC < 0**: Mean reversion (positive values follow negative values)

**Advantages**:
- Simple, interpretable, fast
- Robust to parameter choices
- PyTorch-friendly
- Well-understood in time series analysis

**Recommendation**: ✓ Use for smoke test. Best balance of simplicity and information content.

**Last Verified**: 2026-02-14 (real_rate research)

---

## Q5: What is the Variance Ratio Test?

**Answer**: Compares variance of k-period returns to k times the variance of 1-period returns.

**Formula**:
```
VR(k) = Var(r_t + r_{t-1} + ... + r_{t-k+1}) / (k * Var(r_t))
```

**Under random walk hypothesis**: VR(k) = 1

**Interpretation**:
- **VR > 1**: Positive autocorrelation (persistence)
- **VR < 1**: Negative autocorrelation (mean reversion)

**Implementation**:
```python
import numpy as np

def variance_ratio(returns, k=5):
    """
    Calculate Variance Ratio at lag k
    Args:
        returns: 1D array of returns
        k: lag period
    Returns:
        vr: variance ratio
    """
    # k-period returns
    k_returns = np.add.reduceat(returns, np.arange(0, len(returns), k))

    # Variance of k-period returns
    var_k = np.var(k_returns, ddof=1)

    # Variance of 1-period returns
    var_1 = np.var(returns, ddof=1)

    # Variance ratio
    vr = var_k / (k * var_1)

    return vr
```

**Advantages**:
- More robust than Hurst in some applications
- Well-grounded in financial econometrics
- Statistical tests available (Lo-MacKinlay test)

**Limitations**:
- Requires choosing k (typical values: 2, 5, 10, 20)
- Adds complexity for smoke test

**Recommendation**: Consider for Attempt 2+ if autocorrelation proves insufficient. Skip for smoke test.

**Last Verified**: 2026-02-14

---

## Q6: Which persistence method should I use for the smoke test?

**Answer**: **Rolling Autocorrelation at lags 1, 5, 10**

**Rationale**:
1. **Simplest**: One-line pandas/PyTorch implementation
2. **Fast**: O(N) computation per lag
3. **Interpretable**: Direct measure of memory
4. **Robust**: Not sensitive to hyperparameter choices
5. **PyTorch-friendly**: Easy to integrate

**Implementation for Smoke Test**:
```python
# Compute on rolling 60-day windows
df['autocorr_20d'] = df['change_1d'].rolling(60).apply(
    lambda x: x.autocorr(lag=1) if len(x) >= 2 else 0
)
```

**When to Use Alternatives**:
- **Hurst**: If you need a single scalar persistence measure (Attempt 2+)
- **Variance Ratio**: If you need econometric rigor or hypothesis testing (Attempt 3+)

**Last Verified**: 2026-02-14 (real_rate design decision)
