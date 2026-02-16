# Error Log: temporal_context Attempt 1 - FOURTH ERROR

## Date
2026-02-16 21:41 (Kernel version 4)

## Error Type
**Optuna CategoricalDistribution Dynamic Value Space Error**

## Error Message
```
ValueError: CategoricalDistribution does not support dynamic value space.
```

## Location
Cell 13 (Optuna HPO function), objective function:
```python
d_model = trial.suggest_categorical('d_model', [16, 24, 32])

# n_heads depends on d_model (must be divisor)
if d_model == 16:
    n_heads = trial.suggest_categorical('n_heads', [2])
elif d_model == 24:
    n_heads = trial.suggest_categorical('n_heads', [2, 4])
else:  # d_model == 32
    n_heads = trial.suggest_categorical('n_heads', [2, 4])
```

## Root Cause
Optuna does not allow conditional/dynamic parameter spaces where the choices for one parameter depend on the value of another parameter within the same trial.

### Why This Happens
- `n_heads` choices change based on `d_model` value
- Optuna requires all parameter distributions to be static
- Dynamic value spaces violate Optuna's design principles

## Solution Options

### Option 1: Flatten Parameter Space (Recommended)
Define valid (d_model, n_heads) combinations as a single categorical parameter:
```python
model_config = trial.suggest_categorical('model_config', [
    (16, 2),
    (24, 2), (24, 4),
    (32, 2), (32, 4)
])
d_model, n_heads = model_config
```

### Option 2: Use Fixed n_heads
Always use `n_heads = 2` or `n_heads = 4` regardless of d_model:
```python
d_model = trial.suggest_categorical('d_model', [16, 24, 32])
n_heads = trial.suggest_categorical('n_heads', [2, 4])
# Add validation: if d_model % n_heads != 0, prune trial
if d_model % n_heads != 0:
    raise optuna.TrialPruned()
```

### Option 3: Use IntDistribution with Constraints
```python
d_model = trial.suggest_categorical('d_model', [16, 24, 32])
# Always suggest from full range, then validate
n_heads = trial.suggest_categorical('n_heads', [2, 4])
if d_model % n_heads != 0:
    raise optuna.TrialPruned()
```

## Decision
**Use Option 1 (Flatten Parameter Space)** - cleanest and most explicit solution that avoids trial pruning overhead.

## Implementation
Modify Cell 13 in train.ipynb to use flattened parameter space.

## Status
- 🔍 Error identified and documented
- ⏸️ Awaiting fix implementation
