#!/usr/bin/env python3
"""
SyMANTIC Ablation Study on NODE Distillation
=============================================
Ablation study to vary data points generated from a pre-trained NODE model
and examine the full Pareto front of equations screened by SyMANTIC.

We load a pre-trained NODE model trained with 20% noise on 40 steps.
We then vary the number of data points evaluated from this NODE by changing
the integration steps.
For each try, we save the full Pareto front of equations screened by SyMANTIC.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint
from scipy.stats import qmc

# Suppress convergence warnings from pysindy/sklearn during sweeps
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path setup — ensure repo root is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Add symantic directory for local imports
symantic_path = str(REPO_ROOT / "symantic")
if symantic_path not in sys.path:
    sys.path.insert(0, symantic_path)

from casestudy.simulate import dynamic_system
from benchmark.complexity import calculate_complexity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Initial condition bounds
U0_BOUNDS = [0.5, 2.0]
V0_BOUNDS = [0.05, 0.2]
Y0_BOUNDS = [0.05, 0.2]

# Training IC subdomain (inner mask)
TRAIN_U0 = [0.6, 1.9]
TRAIN_V0 = [0.06, 0.19]
TRAIN_Y0 = [0.06, 0.19]

# Ablation settings
ABLATION_STEPS = [10, 20, 40, 80]
SIS_FEATURES = 30   # Screen for deeper models (default was 20)
METRICS = [0.01, 0.99]  # Restored metrics to allow simple variables to terminate early

# Output directories
ABLATION_DIR = SCRIPT_DIR / "results" / "SyMANTIC_ablation"
SCREENED_EQS_DIR = ABLATION_DIR / "screened_equations"
MODEL_PATH = ABLATION_DIR / "models" / "node_noise_pct_0.2.pt"

# Ground truth equations for reference
GROUND_TRUTH_EQUATIONS = {
    "du/dt": "1/(1 + v**2) - u",
    "dv/dt": "1/(1 + (u/(1+y)**2)**2) - v",
    "dy/dt": "-0.1*y",
}

# ---------------------------------------------------------------------------
# NODE Model Architecture
# ---------------------------------------------------------------------------
class ODEFunc(nn.Module):
    """Neural ODE right-hand side: 3-state MLP."""
    def __init__(self, state_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, y):
        return self.net(y)


def get_training_ics():
    """Get the exact same training initial conditions via Sobol sampling."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=SEED)
    sample = sampler.random(n=64)
    l_bounds = [U0_BOUNDS[0], V0_BOUNDS[0], Y0_BOUNDS[0]]
    u_bounds = [U0_BOUNDS[1], V0_BOUNDS[1], Y0_BOUNDS[1]]
    ics = qmc.scale(sample, l_bounds, u_bounds)

    eps = 1e-7
    train_mask = (
        (ics[:, 0] >= TRAIN_U0[0] - eps) & (ics[:, 0] <= TRAIN_U0[1] + eps) &
        (ics[:, 1] >= TRAIN_V0[0] - eps) & (ics[:, 1] <= TRAIN_V0[1] + eps) &
        (ics[:, 2] >= TRAIN_Y0[0] - eps) & (ics[:, 2] <= TRAIN_Y0[1] + eps)
    )
    return ics[train_mask]


def get_node_predictions(model, y0s, t_points):
    """Integrate NODE from multiple initial conditions."""
    t_tensor = torch.tensor(t_points, dtype=torch.float32).to(DEVICE)
    y0_tensor = torch.tensor(y0s, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = odeint(model, y0_tensor, t_tensor, method="rk4")  # (T, B, D)
        preds = preds.permute(1, 0, 2).cpu().numpy()  # (B, T, D)
    return preds


def run_symantic_sr(states, grads):
    """Fit SyMANTIC on NODE gradients using Feature_Space_Construction."""
    import Feature_Space_Construction as fsc
    from unittest.mock import patch

    feature_names = ["u", "v", "y"]
    target_names = ["dudt", "dvdt", "dydt"]
    operators = ["+", "/", "^-1", "pow(2)", "+1"]

    equations = []
    complexities = []
    pareto_fronts = []

    for k, target in enumerate(target_names):
        print(f"    Fitting {target}...")

        # Build DataFrame: first column = target, rest = features
        df = pd.DataFrame(states.astype(np.float32), columns=feature_names)
        df.insert(0, target, grads[:, k].astype(np.float32))

        # Construct the feature space
        model_fsc = fsc.feature_space_construction(
            operators,
            df,
            no_of_operators=None,
            metrics=METRICS,
            dimension=3,
            sis_features=SIS_FEATURES,
            disp=False
        )

        with patch("builtins.input", return_value="no"):
            _, _, _, df_sorted = model_fsc.feature_space()

        if df_sorted is None or len(df_sorted) == 0:
            print(f"    WARNING: SyMANTIC returned empty Pareto set for {target}")
            equations.append("0")
            complexities.append(0.0)
            pareto_fronts.append(None)
            continue

        # Combine terms to make the full equations
        df_sorted['final'] = df_sorted.apply(fsc.combine_equation, axis=1)

        # Select Pareto point (best accuracy with reasonable complexity)
        max_comp = 5.0 if target == "dydt" else 15.0
        df_filtered = df_sorted[df_sorted['Complexity'] <= max_comp]
        if len(df_filtered) == 0:
            df_filtered = df_sorted

        # Sort by complexity ascending
        df_filtered = df_filtered.sort_values(by='Complexity', ascending=True)

        df_above_thresh = df_filtered[df_filtered['Score'] >= 0.98]
        if len(df_above_thresh) > 0:
            selected_row = df_above_thresh.iloc[0]
        else:
            best_idx = df_filtered['Score'].idxmax()
            selected_row = df_filtered.loc[best_idx]

        eq = str(selected_row['final'])
        equations.append(eq)
        complexities.append(calculate_complexity(eq, method="operator_count"))
        pareto_fronts.append(df_sorted)

        print(f"    {target} selected: {eq}")

    return {
        "equations": equations,
        "complexities": complexities,
        "pareto_fronts": pareto_fronts,
    }


def main():
    start_time = time.time()
    SCREENED_EQS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SYMANTIC ABLATION STUDY")
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Varying steps: {ABLATION_STEPS}")
    print(f"Screening features size: {SIS_FEATURES}")
    print("=" * 70)

    # 1. Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: Pre-trained NODE model not found at {MODEL_PATH}!")
        sys.exit(1)

    print("Loading pre-trained NODE model...")
    model = ODEFunc().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 2. Get training ICs
    train_ics = get_training_ics()
    print(f"Number of training initial conditions: {len(train_ics)}")

    summary_records = []

    # 3. Loop over varying steps (ablation study)
    for n_steps in ABLATION_STEPS:
        total_pts = len(train_ics) * n_steps
        print(f"\n--- Running ablation trial: steps={n_steps} (Total points = {total_pts}) ---")
        
        t_eval = np.linspace(0.0, 15.0, n_steps)
        
        # Integrate NODE to get clean states
        pred_trajs = get_node_predictions(model, train_ics, t_eval)
        states = pred_trajs.reshape(-1, 3)

        # Get NODE gradients at these states
        states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            grads = model(0, states_tensor).cpu().numpy()

        # Run SyMANTIC SR
        sr_res = run_symantic_sr(states, grads)

        # Save the full Pareto fronts to CSV
        target_names = ["dudt", "dvdt", "dydt"]
        for idx, target in enumerate(target_names):
            df_pareto = sr_res["pareto_fronts"][idx]
            if df_pareto is not None:
                csv_path = SCREENED_EQS_DIR / f"pareto_front_{target}_n{total_pts}.csv"
                df_pareto.to_csv(csv_path, index=False)
                print(f"    Saved full Pareto front for {target} to {csv_path}")

        # Store summary details
        summary_records.append({
            "steps_per_traj": n_steps,
            "total_points": total_pts,
            "dudt_eq": sr_res["equations"][0],
            "dudt_complexity": sr_res["complexities"][0],
            "dvdt_eq": sr_res["equations"][1],
            "dvdt_complexity": sr_res["complexities"][1],
            "dydt_eq": sr_res["equations"][2],
            "dydt_complexity": sr_res["complexities"][2],
        })

    # Save summary table
    df_summary = pd.DataFrame(summary_records)
    summary_path = ABLATION_DIR / "ablation_summary_table.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("Ablation Study Summary:")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print(f"\nSummary table saved to {summary_path}")

    elapsed = time.time() - start_time
    print(f"\nAblation study completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
