# Model & Feature Evaluation Knowledge Base

## Purpose

Record what works and what doesn't for each submodel:
- Architecture choices (MLP vs GRU vs Transformer)
- Feature engineering effectiveness
- Hyperparameter ranges that work/fail
- Failure patterns to avoid

## Structure

```
docs/knowledge/evaluations/
├── EVALUATION_STRUCTURE.md       ← This file
├── architectures/
│   ├── autoencoder_patterns.md   ← What works for autoencoders
│   ├── gru_patterns.md           ← What works for GRU
│   └── transformer_patterns.md   ← What works for Transformers
├── features/
│   ├── regime_features.md        ← Regime detection feature effectiveness
│   ├── velocity_features.md      ← Velocity/acceleration effectiveness
│   └── persistence_features.md   ← Persistence measurement effectiveness
└── submodels/
    ├── real_rate.md              ← Learnings from real_rate submodel
    ├── dxy.md                    ← Learnings from DXY submodel
    └── vix.md                    ← etc.
```

## Entry Format

Each evaluation entry follows:

```markdown
## Evaluation: [Architecture/Feature Name] - [Date]

**Context**: [Submodel, Attempt #, Phase]

**Approach**: [What was tried]

**Result**: [Gate 1/2/3 result or other metrics]

**Analysis**:
- What worked: [List]
- What didn't work: [List]
- Why (hypothesis): [Explanation]

**Recommendation**:
- ✓ Use for: [Use cases]
- ✗ Avoid for: [Anti-use cases]
- Alternative: [Better approach if this failed]

**Evidence**:
- Training log: logs/training/[feature]_[attempt].json
- Evaluation log: logs/evaluation/[feature]_[attempt].json

**Last Updated**: [Date]
```

## Example: Autoencoder Evaluation

```markdown
## Evaluation: MLP Autoencoder for real_rate - 2026-02-14

**Context**: real_rate submodel, Attempt 1, Smoke Test

**Approach**:
- Architecture: 3-layer MLP encoder/decoder
- Input: Sliding window of 60 timesteps × 8 features
- Latent: 4 dimensions with Tanh activation
- Loss: MSE reconstruction
- Optimizer: Adam with ReduceLROnPlateau

**Result**:
- Gate 1: PASS (overfit ratio 1.2, no leakage)
- Gate 2: FAIL (MI increase only 2.3%, target >5%)
- Gate 3: Not evaluated

**Analysis**:

What worked:
- Sliding window approach captured temporal patterns
- 4 latent dimensions were sufficient (not too many, not too few)
- GPU training completed in 3 minutes (efficient)

What didn't work:
- Information gain too low (2.3% < 5% target)
- Latent features showed high correlation with raw input (0.85)
- Model learned trivial identity mapping

Why (hypothesis):
- Window size (60) too small to capture meaningful regimes
- No explicit regime signal in loss function (purely reconstruction)
- Features too correlated (velocity derived from same base signal)

**Recommendation**:
✓ Use MLP autoencoder for: Quick smoke tests, simple feature sets
✗ Avoid for: Complex temporal dynamics requiring long-term memory
Alternative: Try GRU autoencoder with attention, or add contrastive loss

**Evidence**:
- Training log: logs/training/real_rate_1.json
- Evaluation log: logs/evaluation/real_rate_1.json

**Next Steps**:
- Attempt 2: Increase window size to 120 days
- Attempt 3: Add contrastive loss to force regime separation
- Attempt 4: Try GRU encoder if MLP still fails

**Last Updated**: 2026-02-14
```

## Integration with Workflow

### When to Update

1. **After Gate evaluation** (evaluator agent)
   - Add evaluation entry to relevant files
   - Link to training/evaluation logs

2. **When changing architecture** (architect agent)
   - Check existing evaluations first
   - Document why chosen approach differs from past failures

3. **When debugging** (any agent)
   - Search evaluations for similar failure patterns
   - Avoid repeating known dead ends

### Search Utilities

Extend `scripts/search_knowledge.py` to include evaluations:

```bash
# Search for MLP autoencoder learnings
python scripts/search_knowledge.py "MLP autoencoder" --type evaluation

# Search for real_rate specific learnings
python scripts/search_knowledge.py "real_rate" --submodel

# Search for Gate 2 failures
python scripts/search_knowledge.py "Gate 2 FAIL" --type evaluation
```

## Benefits

1. **Avoid repetition**: Don't try failed approaches again
2. **Faster iteration**: Know which hyperparameter ranges work
3. **Knowledge transfer**: Learnings from real_rate apply to DXY
4. **Explainability**: Understand why certain choices were made
5. **Continuous improvement**: Each attempt builds on previous knowledge

---

**Status**: Proposal - Ready to implement
**Next Step**: Create first evaluation entry after real_rate Attempt 1 completes
