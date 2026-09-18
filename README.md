# Neural ODE Distillation via Symbolic Regression

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

A framework for **distilling Neural ODE (NODE) models into interpretable symbolic ordinary differential equations** using symbolic regression.
This repository provides the code for reproducing the experiments and results presented in the accompanying paper.

## Overview

Neural ODEs are powerful black-box models for learning continuous-time dynamics from noisy observational data, but they lack interpretability and generalize poorly outside their training domain.
This work proposes a two-stage pipeline that combines the noise-filtering strength of Neural ODEs with the interpretability and extrapolation guarantees of symbolic regression:

1. **Stage 1 — Neural ODE Training**: Fit a Neural ODE to noisy trajectory data, producing a smooth, continuous-time model of the system dynamics.
2. **Stage 2 — Symbolic Distillation**: Extract the NODE's learned gradients (derivatives) over the training state space and distill them into a closed-form symbolic equation using symbolic regression (SINDy or SyMANTIC).

The resulting symbolic ODE inherits the global mathematical properties of its constituent operators (e.g., rational saturation), enabling robust **out-of-distribution extrapolation** that neural networks cannot achieve.

## Method

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NODE Distillation Pipeline                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Noisy Trajectory       Neural ODE         Gradient         Symbolic   │
│       Data          ──►  Training     ──►  Extraction   ──►  Regression │
│    x(t) + ε              (Stage 1)          dx/dt             (Stage 2) │
│                                              │                          │
│                                              ▼                          │
│                                     ┌────────────────┐                  │
│                                     │   SINDy         │                  │
│                                     │   (Polynomial)  │                  │
│                                     ├────────────────┤                  │
│                                     │   SyMANTIC      │                  │
│                                     │   (Rational)    │                  │
│                                     └────────┬───────┘                  │
│                                              │                          │
│                                              ▼                          │
│                                     Interpretable ODE                   │
│                                     dx/dt = f(x)                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: By fitting symbolic regression to the NODE's *in-domain* gradients rather than raw noisy data, the pipeline:
- **Denoises** the observations through the Neural ODE's smooth trajectory fitting.
- **Discovers** the correct functional form (including rational terms) from clean gradient data.
- **Extrapolates** reliably because the symbolic expression's algebraic structure governs behavior outside the training domain.

## Repository Structure

```
SyMANTIC-NODE-Distillation/
├── benchmark/                  # Benchmark framework for multi-state systems
│   ├── __init__.py
│   ├── config.py               # Experiment configuration dataclasses
│   ├── runner.py               # Main experiment runner (derivative + symbolic methods)
│   ├── data.py                 # Data loading, normalization, and train/val splitting
│   ├── gradients.py            # Finite-difference gradient estimators
│   ├── node_gradients.py       # Neural ODE training and gradient extraction
│   ├── simulation.py           # ODE integration with piecewise inputs and fallbacks
│   ├── symbolic_sindy.py       # SINDy-based symbolic regression interface
│   ├── symbolic_symantic.py    # SyMANTIC-based symbolic regression interface
│   ├── complexity.py           # Equation complexity metrics
│   ├── metrics.py              # RMSE and R² evaluation metrics
│   └── plotting.py             # Publication-style trajectory and Pareto plots
│
├── casestudy/                  # Spruce-budworm noise impact study (paper case study)
│   ├── simulate.py             # Ground-truth Spruce-budworm ODE simulator
│   ├── run_noise_study_updated.py  # Full noise sweep: data → NODE → SR → evaluation
│   ├── generate_plots.py       # Standalone figure generation from saved checkpoints
│   ├── run_symantic_only.py    # Re-run SyMANTIC fitting and save Pareto fronts
│   ├── test_dynamics_toy.ipynb # Interactive notebook for the case study
│   └── results/                # Pre-computed results, figures, and model checkpoints
│       ├── spruce_budworm_noise_study/
│       │   ├── models/         # Saved NODE model checkpoints (.pt files)
│       │   ├── equations.json  # Discovered symbolic equations per noise level
│       │   ├── performance_table.csv
│       │   ├── equations_table.csv
│       │   └── *.png           # Trajectory, parity, and distribution plots
│       └── paper_figures/      # Publication-ready figures
│
├── symantic/                   # SyMANTIC symbolic regression library
│   ├── Feature_Space_Construction.py   # Iterative feature space expansion
│   ├── Regressor.py                    # Sparse regression with Pareto optimization
│   ├── pareto_new.py                   # Pareto front computation
│   ├── feature_selection.py            # Feature selection utilities
│   ├── cmi_feature_selection.py        # Conditional mutual information selection
│   ├── mic_torch.py                    # Mutual information computation (PyTorch)
│   ├── Feature_Space_Construction.ipynb # Interactive demonstration notebook
│   └── test_dynamics.ipynb             # SyMANTIC dynamics testing notebook
│
├── README.md
└── .gitignore
```

## Installation

### Prerequisites

- Python ≥ 3.9
- CUDA-capable GPU (optional, for faster NODE training)

### Setup

```bash
# Clone the repository
git clone https://github.com/penguinyou88/SyMANTIC-NODE-Distillation.git
cd SyMANTIC-NODE-Distillation

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy pandas scipy matplotlib scikit-learn torch torchdiffeq pysindy sympy
```

### Dependencies

| Package       | Purpose                                    |
| :------------ | :----------------------------------------- |
| `numpy`       | Array operations                           |
| `pandas`      | Data manipulation                          |
| `scipy`       | ODE integration (`solve_ivp`)              |
| `matplotlib`  | Plotting and visualization                 |
| `scikit-learn`| Metrics (RMSE, R²) and feature selection   |
| `torch`       | Neural ODE model definition and training   |
| `torchdiffeq` | Differentiable ODE solvers for PyTorch     |
| `pysindy`     | SINDy sparse symbolic regression           |
| `sympy`       | Symbolic mathematics                       |

## Usage

### Running the Noise Impact Case Study

The main experiment runs the full NODE distillation pipeline across multiple noise levels on the Spruce-budworm model:

```bash
python casestudy/run_noise_study_updated.py
```

This will:
1. Generate 80 Sobol-sampled trajectories from the Spruce-budworm ODE
2. Add observation noise at levels σ ∈ {0.0, 0.02, 0.05, 0.1}
3. Train a Neural ODE for each noise level
4. Distill symbolic equations using SINDy and SyMANTIC
5. Evaluate interpolation, time extrapolation, and state extrapolation performance
6. Save results, equations, and figures to `casestudy/results/`

### Regenerating Figures (Without Retraining)

To regenerate publication figures from saved NODE checkpoints and equation strings:

```bash
python casestudy/generate_plots.py
```

### Running the Benchmark Framework

For multi-state systems with exogenous inputs (e.g., industrial process data):

```python
from benchmark import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    data_path="path/to/your/data.csv",
    batch_id="YOUR_BATCH",
    node_epochs=300,
)
results_df = run_experiment(cfg)
```

## Case Study: Spruce-Budworm Outbreak Model

The paper case study uses the 1D Spruce-budworm outbreak model as a ground-truth benchmark:

$$\frac{dw}{dt} = r\,w\!\left(1 - \frac{w}{a}\right) - \frac{w^2}{1 + w^2}, \qquad r = 0.5,\ a = 10.0$$

This system features a **rational nonlinearity** ($w^2 / (1 + w^2)$) that cannot be captured by polynomial bases alone, making it an ideal test for comparing SINDy (polynomial library) against SyMANTIC (rational function search).

### Experimental Design

| Setting               | Value                          |
| :-------------------- | :----------------------------- |
| Initial conditions    | 80 Sobol-sampled, $w_0 \in [0.05, 16.0]$ |
| Training IC domain    | $w_0 \in [0.05, 8.0]$          |
| Test (OOD) IC domain  | $w_0 \in [8.0, 16.0]$          |
| Time window (train)   | $t \in [0, 5.0]$               |
| Time window (test)    | $t \in [5.0, 12.0]$            |
| Noise levels          | $\sigma \in \{0.0, 0.02, 0.05, 0.1\}$ |

## Key Results

Performance summary across noise levels (trajectory-level R² on clean ground truth):

| Noise (σ) | Method      | Train R²  | Time Extrap. R² | State Extrap. R² |
| :--------- | :---------- | :-------- | :-------------- | :--------------- |
| **0.00**   | NODE        | 0.9999    | 0.9999          | 0.3301           |
|            | SINDy       | 0.9989    | 0.9926          | 0.8590           |
|            | **SyMANTIC**| **0.9999**| **0.9993**      | **0.9999**       |
| **0.02**   | NODE        | 0.9999    | 0.9999          | 0.4773           |
|            | SINDy       | 0.9989    | 0.9927          | 0.8607           |
|            | **SyMANTIC**| **0.9999**| **0.9994**      | **0.9999**       |
| **0.05**   | NODE        | 0.9999    | 0.9999          | 0.5023           |
|            | SINDy       | 0.9988    | 0.9922          | 0.8678           |
|            | **SyMANTIC**| **0.9999**| **0.9992**      | **0.9999**       |
| **0.10**   | NODE        | 0.9999    | 0.9997          | 0.3421           |
|            | SINDy       | 0.9988    | 0.9924          | 0.8689           |
|            | **SyMANTIC**| **0.9999**| **0.9989**      | **0.9995**       |

**Key findings**:
- **SyMANTIC** achieves near-perfect out-of-distribution extrapolation (R² ≥ 0.999) by discovering the correct rational functional form.
- **Neural ODEs** excel at interpolation but fail under state extrapolation due to the lack of structure-preserving constraints outside the training domain.
- **SINDy** (polynomial basis) provides moderate extrapolation but cannot capture the rational saturation behavior, leading to divergence at large state values.
- The NODE distillation pipeline is **robust to observation noise** — SyMANTIC consistently recovers the true dynamics across all tested noise levels.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{node_distillation_2026,
  title     = {Neural ODE Distillation via Symbolic Regression for Interpretable and Extrapolable Dynamic Models},
  author    = {Peng, You and others},
  year      = {2026},
  note      = {Paper submitted}
}
```

## License

This project is released for academic and research use. Please see the accompanying paper for details.

## Acknowledgments

- **SyMANTIC** symbolic regression library by Madhav Muthyala ([muthyala.7](mailto:muthyala.7@osu.edu))
- Built with [PyTorch](https://pytorch.org/), [torchdiffeq](https://github.com/rtqichen/torchdiffeq), and [PySINDy](https://github.com/dynamicslab/pysindy)
