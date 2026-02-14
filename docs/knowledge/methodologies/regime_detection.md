# Regime Detection Methodologies

## Q1: What are the main approaches for detecting regime changes in financial time series?

**Answer**: Three main approaches exist:
1. Hidden Markov Models (HMM)
2. Markov-Switching Regression Models
3. Self-Exciting Threshold Autoregression (SETAR)

**Best for PyTorch + Smoke Test**: Simple percentile-based thresholds
- Define regimes based on rolling percentiles (e.g., <33rd = low, 33-67th = mid, >67th = high)
- Fully PyTorch-compatible
- Fast to implement and debug
- Sufficient for initial verification

**Evidence**:
- Source: Research for real_rate submodel (2026-02-14)
- HMM/SETAR exist but require scikit-learn/statsmodels (not PyTorch native)
- Common practice in quantitative finance: percentile rank = regime indicator

**Code Example**:
```python
import torch

# Percentile-based regime detection (PyTorch)
def compute_regime_percentile(values, window=252):
    """
    Args:
        values: torch.Tensor of shape (N,)
        window: lookback period for percentile calculation
    Returns:
        regime_percentile: torch.Tensor of shape (N,) with values in [0, 1]
    """
    N = len(values)
    regime = torch.zeros(N)

    for i in range(window, N):
        window_data = values[i-window:i]
        rank = (window_data < values[i]).sum().float()
        regime[i] = rank / window

    return regime

# Alternative: pandas-based (for preprocessing)
import pandas as pd
regime_percentile = df['value'].rolling(252).apply(
    lambda x: (x < x.iloc[-1]).sum() / len(x)
).fillna(0.5)
```

**Limitations**:
- Percentile approach is retrospective (uses past data to classify current state)
- Requires choosing lookback window (252 days = 1 trading year is typical)
- Does not predict future regime changes

**When to Use Alternatives**:
- HMM: If you need probabilistic regime transitions or regime forecasting
- SETAR: If you need asymmetric dynamics modeling (different behavior in different regimes)

**Last Verified**: 2026-02-14 (real_rate research)

---

## Q2: How to implement Hidden Markov Models (HMM) for regime detection?

**Answer**: Use `hmmlearn` library (scikit-learn style), not PyTorch native.

**Implementation**:
```python
from hmmlearn.hmm import GaussianHMM

# Fit HMM (2-3 states typical for financial regimes)
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
model.fit(returns.reshape(-1, 1))

# Get regime probabilities
regime_probs = model.predict_proba(returns.reshape(-1, 1))
```

**PyTorch Integration**:
1. Fit HMM in preprocessing (outside PyTorch)
2. Convert regime probabilities to torch.Tensor
3. Use as input features to PyTorch model

**Strengths**:
- Well-established in finance for Bull/Bear or Volatile/Calm regimes
- Captures volatility persistence, time-varying correlations
- Probabilistic output (soft assignments)

**Limitations**:
- Not PyTorch native (requires separate training pipeline)
- Requires choosing number of states (2-3 recommended)
- Adds complexity to smoke tests

**Recommendation**: Skip for smoke test, use percentile method. Consider for Attempt 2+ if needed.

**Last Verified**: 2026-02-14

---

## Q3: What is SETAR and should I use it?

**Answer**: Self-Exciting Threshold Autoregression (SETAR) - a non-parametric regime detection method.

**Concept**:
- SETAR(k, p) uses threshold values to separate k+1 regimes
- Delay parameter d determines which lag is used for threshold comparison
- No distributional assumptions required

**Implementation**:
- `statsmodels` has SETAR module
- ⚠️ WARNING: `pip install setar` does NOT exist (hallucination found in some sources)
- Must use statsmodels or implement from scratch

**Strengths**:
- Captures asymmetric dynamics (different AR parameters in different regimes)
- No distributional assumptions

**Challenges**:
- Requires grid search over delay (d) and threshold (c) parameters
- Objective function has many local optima
- Computationally expensive
- Not PyTorch native

**Recommendation**: **Avoid for smoke test**. Too complex for initial verification. Use simple percentile method instead.

**Last Verified**: 2026-02-14 (architect fact-check found `setar` package does not exist)

---

## Q4: How many regimes should I use?

**Answer**: 2-3 regimes for financial time series

**Typical Configurations**:
- **2 regimes**: Bull/Bear, High/Low, Expansion/Contraction
- **3 regimes**: Low/Mid/High, Bear/Neutral/Bull
- 4+ regimes: Rarely used (overfitting risk)

**For Percentile Method**:
- 3 regimes: <33rd percentile, 33-67th percentile, >67th percentile
- Or 2 regimes: <50th percentile, >50th percentile

**Evidence**: Common practice in quantitative finance literature

**Last Verified**: 2026-02-14
