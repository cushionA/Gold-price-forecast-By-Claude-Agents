# Knowledge Base Structure Proposal

## Problem with Current Approach

**Current**: Each research report is feature-specific and narrative
- `docs/research/real_rate_attempt1.md` - 200+ lines of narrative text
- Hard to reuse findings for other features
- Same topics researched multiple times (e.g., regime detection for VIX, DXY, etc.)

**Example**: If we research "regime detection methods" for real_rate, we'll likely need the same info for VIX and DXY. Currently, we'd research it 3 times.

## Proposed: Q&A Knowledge Base

### Structure

```
docs/knowledge/
├── KNOWLEDGE_STRUCTURE.md          ← This file
├── methodologies/
│   ├── regime_detection.md         ← Q&A: How to detect regime changes?
│   ├── persistence_measurement.md  ← Q&A: How to measure persistence?
│   ├── multivariate_forecasting.md
│   └── autoencoder_design.md
├── data_sources/
│   ├── fred_apis.md                ← Q&A: What FRED series are available?
│   ├── multi_country_data.md       ← Q&A: Multi-country equivalents for X?
│   └── data_delays.md              ← Q&A: What's the delay for series X?
├── financial_concepts/
│   ├── real_rates_gold_relationship.md
│   ├── vix_gold_relationship.md
│   └── lag_structures.md
└── technical/
    ├── pytorch_time_series.md      ← Q&A: How to implement X in PyTorch?
    ├── optuna_best_practices.md
    └── kaggle_limitations.md
```

### Format: Question-Driven

Each knowledge file follows this structure:

```markdown
# Topic: [Topic Name]

## Question 1: [Specific Question]

**Answer**: [Concise answer]

**Evidence**:
- Source 1: [citation/URL]
- Source 2: [citation/URL]

**Code Example** (if applicable):
```python
# Minimal working example
```

**Limitations**:
- Limitation 1
- Limitation 2

**Last Verified**: 2026-02-14

---

## Question 2: [Next Question]
...
```

## Example: regime_detection.md

```markdown
# Regime Detection Methodologies

## Q: What are the main approaches for detecting regime changes in financial time series?

**Answer**: Three main approaches: HMM, Markov-Switching, and Threshold Autoregression (SETAR)

**Best for PyTorch**: Simple percentile-based thresholds or rolling window statistics
- Reason: HMM/SETAR require scikit-learn style APIs, not PyTorch native
- Smoke test: Use 33rd/67th percentile (3 regimes: low/mid/high)

**Evidence**:
- Research for real_rate (2026-02-14): HMM works but not PyTorch-native
- Common in quant finance: Regime = percentile rank over lookback window

**Code Example**:
```python
# PyTorch-friendly regime detection
regime_percentile = (df['value'].rolling(252).rank() / 252).fillna(0.5)
```

**Limitations**:
- Percentile approach is retrospective, not predictive
- Requires choosing lookback window (252 days = 1 year typical)

**Last Verified**: 2026-02-14

---

## Q: How to implement HMM in PyTorch for regime detection?

**Answer**: Not straightforward. Use hmmlearn (scikit-learn style) or implement from scratch.

**Recommendation**: For smoke test, avoid HMM complexity. Use simple approaches.

**Evidence**:
- hmmlearn library exists but not PyTorch
- PyTorch HMM requires custom implementation (100+ lines)

**Last Verified**: 2026-02-14
```

## Benefits

1. **Reusability**: "How to detect regime changes?" answer works for real_rate, VIX, DXY
2. **Incremental Updates**: Add new Q&A without rewriting entire reports
3. **Easy Lookup**: `grep -r "regime detection" docs/knowledge/`
4. **Fact-Checking**: Architect can quickly verify specific claims
5. **Avoid Duplication**: Check knowledge base before researching

## Integration with Current Flow

```
entrance → Define research questions
    ↓
researcher → Check docs/knowledge/ first
           → If answer exists: Cite it
           → If not: Research and ADD to knowledge base
           → Create minimal feature-specific summary in docs/research/
    ↓
architect → Fact-check against knowledge base
```

## Migration Plan

1. **Phase 1** (Now): Create knowledge/ structure, keep current flow
2. **Phase 2**: Researcher checks knowledge base before web search
3. **Phase 3**: Auto-update knowledge base with new findings
4. **Phase 4**: researcher primarily updates knowledge base, not per-feature reports

---

**Status**: Proposal - Not yet implemented
**Next Step**: Discuss with orchestrator for approval
