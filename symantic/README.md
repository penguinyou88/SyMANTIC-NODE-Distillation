# <p align="center">SyMANTIC: An Efficient Symbolic Regression Method for Interpretable and Parsimonious Model Discovery in Science and Beyond

![](https://i.ibb.co/4Nmvj3B/symantic-toc.jpg)

SyMANTIC is a symbolic regression algorithm that discovers interpretable, parsimonious equations from data. It combines mutual information-based feature selection, adaptive feature expansion with mathematical operators, and sparse regression to produce a set of Pareto-optimal equations — each offering the best accuracy for a given complexity. SyMANTIC supports multiple regression strategies including exhaustive $\ell_0$ search (default), $\ell_1$ (Lasso), $\ell_2$ (Ridge), and ElasticNet regularization. Built on the PyTorch ecosystem with automatic GPU acceleration.

## Installation

```bash
pip install symantic
```

## Quick Start

```python
from symantic import SymanticModel
import numpy as np
import pandas as pd

# Create DataFrame: first column = target, remaining columns = features
data = np.column_stack((y, X))
df = pd.DataFrame(data, columns=["y", "x1", "x2", "x3"])

# Fit with auto-depth (default) — SyMANTIC adaptively expands features until convergence
model = SymanticModel(
    df,
    operators=['+', '-', '*', '/'],
    disp=True,
)
result = model.fit()

# Access results directly via attributes
print(f"Equation: {result.equation}")
print(f"RMSE: {result.rmse:.6f}")
print(f"R2: {result.r2:.6f}")

# Or use backward-compatible tuple unpacking
res, full_pareto = result       # auto-depth mode
# rmse, equation, r2 = result   # fixed-depth mode

# Plot the Pareto front
model.plot_pareto_front()

# Full Pareto table with Loss, Complexity, R2, Equation columns
print(result.pareto_front)
```

## Regularization

SyMANTIC provides four regression strategies. The choice affects **how sparse solutions are found**, not the feature expansion itself — all strategies use the same feature engineering pipeline and produce a Pareto front.

### L0 — Exhaustive Combinatorial Search (default)

```python
model = SymanticModel(df, operators=['+', '-', '*', '/'], regularization='l0')
```

Enumerates all $\binom{k}{n}$ feature combinations (where $k$ = screened features, $n$ = `n_term`) and solves OLS for each. Finds the global optimum within the screened feature set but scales combinatorially:

| `n_term` | Combinations (sis_features=20) |
|----------|-------------------------------|
| 2        | ~1,900 OLS solves             |
| 3        | ~11,400 OLS solves            |
| 4        | ~48,450 OLS solves            |
| 5        | ~155,040 OLS solves           |

**Best for**: `n_term` <= 3, or when you need guaranteed exhaustive search.

### L1 — Lasso (Recommended for n_term >= 4)

```python
model = SymanticModel(df, operators=['+', '-', '*', '/'], regularization='l1')
```

Uses coordinate descent via `sklearn.linear_model.lasso_path` to sweep a regularization path. The $\ell_1$ penalty naturally drives coefficients to exactly zero, producing sparse solutions without enumerating combinations. Solves ~100 penalized regressions regardless of `n_term`.

**Best for**: `n_term` >= 4, large feature spaces, fast iteration.

### L2 — Ridge

```python
model = SymanticModel(df, operators=['+', '-', '*', '/'], regularization='l2')
```

Shrinks all coefficients toward zero but does not set them exactly to zero. SyMANTIC applies a threshold (`reg_threshold`) to zero out small coefficients, creating approximate sparsity.

**Best for**: Highly correlated features where L1 is unstable.

### ElasticNet — L1 + L2 Mix

```python
model = SymanticModel(
    df, operators=['+', '-', '*', '/'],
    regularization='elastic_net',
    l1_ratio=0.7,  # 70% L1, 30% L2
)
```

Combines L1 sparsity with L2 stability. `l1_ratio=1.0` is pure Lasso; lower values add more Ridge.

**Best for**: Correlated features where you still want exact sparsity.

### How n_term works with regularization

Regardless of the regularization method, `n_term` is always enforced as a **hard upper bound** on the number of terms in the equation. For L1/L2/ElasticNet, if the regularization path produces a solution with more non-zero coefficients than `n_term`, only the top `n_term` largest-magnitude coefficients are kept.

### Pareto front with regularization

All regularization methods generate a Pareto front of (complexity, RMSE) trade-offs:
- **L0**: each feature combination is a candidate solution
- **L1/ElasticNet**: each alpha value in the regularization path produces a different sparsity pattern — the Pareto front emerges naturally from sweeping alpha
- **L2**: each alpha + thresholding combination produces a candidate

Access the Pareto front the same way regardless of method:
```python
result = model.fit()
print(result.pareto_front)   # DataFrame with Loss, Complexity, R2, Equation
model.plot_pareto_front()
```

## Memory Management: Level Pruning

In auto-depth mode (`n_expansion=None`), the feature space grows exponentially with each expansion level. With many features and operators, this can cause out-of-memory crashes because the full feature tensor is materialized before the `max_features` check runs.

**`level_pruning=True`** solves this by pruning the feature space between expansion levels using SIS (Sure Independence Screening — `|X^T @ y|`, absolute correlation with target). After regression at each level, it keeps:
- All **original base features** (always retained as building blocks)
- The **top `sis_features * n_term` derived features** by SIS score

This matches the regressor's internal screening budget and makes `max_features` still useful as a secondary cap on the expanded output:

| | After expansion | After pruning | Next expansion |
|---|-----------------|---------------|----------------|
| **Level 1** (3 base, 4 ops) | 50 | 3 + 60 = 63 | ~8,000 |
| **Level 2** | ~8,000 | 63 | ~8,000 |
| **Without pruning** | 50 -> 5,000 -> millions | - | OOM |

Additional stop conditions with pruning enabled:
- **Stagnation**: stops if RMSE does not improve between levels
- **Max depth**: hard cap of 10 expansion levels

**Controlling the pruning budget**: `sis_features` and `n_term` together determine the retention count (`sis_features * n_term`). The same `sis_features` parameter also controls SIS screening inside the regressor (top `sis_features * n_term` features fed to regression). Increase `sis_features` to carry more features forward (better coverage, more memory); decrease for tighter memory control.

```python
# Aggressive pruning — tight memory, fast
model = SymanticModel(df, operators=['+', '-', '*', '/'],
                      level_pruning=True, sis_features=10)

# Looser pruning — more features survive, bigger search space
model = SymanticModel(df, operators=['+', '-', '*', '/'],
                      level_pruning=True, sis_features=50)

# Combine with fast regularization for large problems
model = SymanticModel(df, operators=['+', '-', '*', '/'],
                      level_pruning=True, sis_features=20,
                      regularization='l1', n_term=4)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | *required* | First column = target, remaining = features |
| `operators` | list of str | *required* | Operators for feature expansion (see below) |
| `n_expansion` | int or None | None | Number of expansion levels. `None` = auto-depth |
| `n_term` | int or None | 3 | Max terms per equation (sparsity) |
| `sis_features` | int | 20 | Features to screen per iteration via SIS |
| `device` | str or None | None | `'cpu'`, `'cuda'`, or `None` (auto-detect GPU) |
| `regularization` | str | `'l0'` | `'l0'`, `'l1'`, `'l2'`, or `'elastic_net'` |
| `reg_alpha` | float or None | None | Regularization strength. `None` = auto-select |
| `l1_ratio` | float | 0.5 | L1/L2 mixing for elastic_net (1.0 = pure L1) |
| `reg_threshold` | float | 1e-4 | L2: zero out coefficients below this fraction of max |
| `n_alphas` | int | 100 | Number of alpha values in regularization path |
| `max_features` | int or None | 2000 | Max features before stopping expansion |
| `level_pruning` | bool | False | Prune features between auto-depth levels (see below) |
| `metrics` | list | [0.06, 0.995] | [RMSE, R2] thresholds for auto-depth convergence |
| `dimensionality` | list or None | None | Sympy dimension expressions for dimensional regression |
| `output_dim` | sympy expr | None | Dimension of the target variable |
| `relational_units` | list or None | None | Unit relationships for dimensional regression |
| `initial_screening` | tuple or None | None | (n_features, quantile) for initial feature screening |
| `multi_task` | tuple or None | None | (target_indices, feature_indices) for multi-task |
| `disp` | bool | False | Print progress information |

### Supported operators

Binary: `+`, `-`, `*`, `/`

Unary: `exp`, `exp(-1)`, `ln`, `log`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `^-1`, `+1`, `-1`, `/2`

Power: `^N` or `pow(N)` where N is any number (e.g., `'^2'`, `'^0.5'`, `'pow(3)'`, `'pow(1/3)'`)

## Return Values

`model.fit()` returns a `FitResult` object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `result.rmse` | float | RMSE of the best (utopia) model |
| `result.equation` | str | Discovered equation |
| `result.r2` | float | R-squared score |
| `result.complexity` | float | Complexity of the utopia model (auto-depth only) |
| `result.pareto_front` | DataFrame | Pareto frontier table (auto-depth only) |
| `result.all_equations` | list | Per-target equations (multi-task only) |

Backward-compatible tuple unpacking is supported:
```python
res, pareto_df = model.fit()              # auto-depth
rmse, equation, r2 = model.fit()          # fixed-depth
rmse, equation, r2, eqs = model.fit()     # multi-task
```

## Examples

### Basic: Auto-depth with L0 (default)

```python
from symantic import SymanticModel
import pandas as pd

df = pd.read_csv("data.csv")  # first column = target

model = SymanticModel(
    df,
    operators=['+', '-', '*', '/'],
    sis_features=20,
    disp=True,
)
result = model.fit()
print(result.equation)
model.plot_pareto_front()
```

### Fast regression with L1 (4+ terms)

```python
model = SymanticModel(
    df,
    operators=['+', '-', '*', '/', 'exp', 'sin', 'cos'],
    n_expansion=2,
    n_term=4,
    regularization='l1',
)
result = model.fit()
print(f"R2={result.r2:.4f}  Equation: {result.equation}")
```

### Dimensional regression

```python
from sympy import symbols

model = SymanticModel(
    df,
    operators=['+', '-', '*', '/'],
    dimensionality=[symbols('L'), symbols('T'), symbols('M')],
    relational_units=[(symbols('L') * symbols('T'), symbols('M'))],
    output_dim=symbols('L') * symbols('L'),
    disp=True,
)
result = model.fit()
```

### Multi-task regression

```python
model = SymanticModel(
    df,
    operators=['+', '-', '*', '/'],
    multi_task=([0, 1], [[2, 3, 4], [2, 3, 4]]),
)
result = model.fit()
print(result.all_equations)  # list of equations for each target
```

More examples can be found in the [examples/](examples/) folder and in the Colab Notebook [SyMANTIC Examples](https://colab.research.google.com/drive/1dBc2QJeEjW0T8iobFU8F54Y25pxR7isG#scrollTo=60564135).

## Citation

    Coming soon
