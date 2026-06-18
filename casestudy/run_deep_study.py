#!/usr/bin/env python3
"""
Deep SyMANTIC Study on NODE Distillation
========================================
Runs SyMANTIC search to depth 8 using node_noise_pct_0.2.pt model.
Generates exactly 10 data points per trajectory.
Evaluates training and test trajectory fitting, identifies worst/best fitting trajectories,
and plots the distributions of R2 and MSE across test trajectories.
"""

import os
import sys
import json
import time
import warnings
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint
from scipy.integrate import solve_ivp
from scipy.stats import qmc
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
symantic_path = str(REPO_ROOT / "symantic")
if symantic_path not in sys.path:
    sys.path.insert(0, symantic_path)

from casestudy.simulate import dynamic_system, simulate_system
from benchmark.complexity import calculate_complexity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

U0_BOUNDS = [0.5, 2.0]
V0_BOUNDS = [0.05, 0.2]
Y0_BOUNDS = [0.05, 0.2]

TRAIN_U0 = [0.6, 1.9]
TRAIN_V0 = [0.06, 0.19]
TRAIN_Y0 = [0.06, 0.19]

N_STEPS_TRAIN = 10
SIS_FEATURES = 30
METRICS = [1e-6, 0.99999]  # Strict metrics to force deep search

ABLATION_DIR = SCRIPT_DIR / "results" / "SyMANTIC_ablation"
SCREENED_EQS_DIR = ABLATION_DIR / "screened_equations"
MODEL_PATH = ABLATION_DIR / "models" / "node_noise_pct_0.2.pt"

GROUND_TRUTH_EQUATIONS = {
    "du/dt": "1/(1 + v**2) - u",
    "dv/dt": "1/(1 + (u/(1+y)**2)**2) - v",
    "dy/dt": "-0.1*y",
}

# ---------------------------------------------------------------------------
# NODE Architecture
# ---------------------------------------------------------------------------
class ODEFunc(nn.Module):
    """Neural ODE right-hand side."""
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


# ---------------------------------------------------------------------------
# Custom Input Mock for Depth 8
# ---------------------------------------------------------------------------
class DeepInputMock:
    """Mock builtins.input to allow loop to continue to depth 8."""
    def __init__(self, max_calls=2):
        self.calls = 0
        self.max_calls = max_calls

    def __call__(self, prompt):
        self.calls += 1
        print(f"      [Mock Input Prompt] Call {self.calls}: returning 'yes' if {self.calls} < {self.max_calls} else 'no'")
        if self.calls < self.max_calls:
            return "yes"
        return "no"


def get_dataset_ics():
    """Sobol sample ICs and partition them."""
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
    return ics, train_mask


def get_node_predictions(model, y0s, t_points):
    t_tensor = torch.tensor(t_points, dtype=torch.float32).to(DEVICE)
    y0_tensor = torch.tensor(y0s, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = odeint(model, y0_tensor, t_tensor, method="rk4")  # (T, B, D)
        preds = preds.permute(1, 0, 2).cpu().numpy()  # (B, T, D)
    return preds


def run_symantic_sr(states, grads):
    import Feature_Space_Construction as fsc

    feature_names = ["u", "v", "y"]
    target_names = ["dudt", "dvdt", "dydt"]
    operators = ["+", "/", "^-1", "pow(2)", "+1"]

    equations = []
    complexities = []
    pareto_fronts = []

    for k, target in enumerate(target_names):
        print(f"    Fitting {target}...")

        df = pd.DataFrame(states.astype(np.float32), columns=feature_names)
        df.insert(0, target, grads[:, k].astype(np.float32))

        model_fsc = fsc.feature_space_construction(
            operators,
            df,
            no_of_operators=None,
            metrics=METRICS,
            dimension=3,
            sis_features=SIS_FEATURES,
            disp=False
        )

        # Force depth 8 (completed level 7 & 8)
        input_mock = DeepInputMock(max_calls=2)
        with patch("builtins.input", side_effect=input_mock):
            _, _, _, df_sorted = model_fsc.feature_space()

        if df_sorted is None or len(df_sorted) == 0:
            print(f"    WARNING: SyMANTIC returned empty Pareto set for {target}")
            equations.append("0")
            complexities.append(0.0)
            pareto_fronts.append(None)
            continue

        df_sorted['final'] = df_sorted.apply(fsc.combine_equation, axis=1)
        pareto_fronts.append(df_sorted)

        # Select model from Pareto frontier
        # Allow dy to be simpler, and others to be more complex
        max_comp = 5.0 if target == "dydt" else 15.0
        df_filtered = df_sorted[df_sorted['Complexity'] <= max_comp]
        if len(df_filtered) == 0:
            df_filtered = df_sorted
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
        print(f"    Selected {target} = {eq} (complexity: {complexities[-1]})")

    return {
        "equations": equations,
        "complexities": complexities,
        "pareto_fronts": pareto_fronts,
    }


def make_rhs_from_symantic_equations(equations, feature_names=("u", "v", "y")):
    processed_eqs = []
    for eq in equations:
        eq = re.sub(r"\^", "**", eq)
        eq = re.sub(r"\bexp\b", "np.exp", eq)
        eq = re.sub(r"\bln\b", "np.log", eq)
        eq = re.sub(r"\blog\b", "np.log10", eq)
        eq = re.sub(r"\babs\b", "np.abs", eq)
        processed_eqs.append(eq)

    def rhs(t, y):
        local_vars = {name: float(y[i]) for i, name in enumerate(feature_names)}
        local_vars["np"] = np
        local_vars["t"] = float(t)

        result = np.zeros(len(equations))
        for k, eq in enumerate(processed_eqs):
            try:
                result[k] = float(eval(eq, {"__builtins__": {}}, local_vars))
            except Exception:
                result[k] = 0.0
        return result

    return rhs


def integrate_from_ic(rhs_func, y0, t_points, method="RK45"):
    try:
        sol = solve_ivp(
            rhs_func,
            [t_points[0], t_points[-1]],
            y0,
            t_eval=t_points,
            method=method,
            max_step=0.1,
        )
        if sol.success:
            return sol.y.T
        else:
            return np.full((len(t_points), len(y0)), np.nan)
    except Exception:
        return np.full((len(t_points), len(y0)), np.nan)


def main():
    start_time = time.time()
    SCREENED_EQS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("DEEP SYMANTIC STUDY & TRAJECTORY EVALUATION")
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Steps: {N_STEPS_TRAIN}")
    print(f"Screening features size: {SIS_FEATURES}")
    print("=" * 70)

    # 1. Load NODE model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}!")
        sys.exit(1)

    print("Loading pre-trained NODE model...")
    model = ODEFunc().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 2. Get ICs
    ics, train_mask = get_dataset_ics()
    train_ics = ics[train_mask]
    test_ics = ics[~train_mask]

    print(f"Train ICs: {len(train_ics)}, Test ICs: {len(test_ics)}")

    # 3. Generate training data points from NODE
    t_train = np.linspace(0.0, 15.0, N_STEPS_TRAIN)
    pred_train_node = get_node_predictions(model, train_ics, t_train)
    states = pred_train_node.reshape(-1, 3)

    states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        grads = model(0, states_tensor).cpu().numpy()

    # 4. Run SyMANTIC SR
    sr_res = run_symantic_sr(states, grads)

    # Save Pareto fronts
    target_names = ["dudt", "dvdt", "dydt"]
    for idx, target in enumerate(target_names):
        df_pareto = sr_res["pareto_fronts"][idx]
        if df_pareto is not None:
            csv_path = SCREENED_EQS_DIR / f"pareto_front_pct_0.2_{target}_n{len(states)}.csv"
            df_pareto.to_csv(csv_path, index=False)
            print(f"Saved Pareto front for {target} to {csv_path}")

    # Build RHS function for integration
    rhs_symantic = make_rhs_from_symantic_equations(sr_res["equations"])

    # 5. Integrate and evaluate on Training ICs (0-15s, 41 points for smooth comparison)
    t_train_eval = np.linspace(0.0, 15.0, 41)
    train_mses = []
    train_r2s = []
    train_clean_trajs = []
    train_node_trajs = []
    train_symantic_trajs = []

    for ic in train_ics:
        # Ground truth clean trajectory
        sol_clean = simulate_system(ic, [0.0, 15.0], len(t_train_eval), integrator="solve_ivp", return_dataframe=False)
        y_clean = sol_clean["states"]
        train_clean_trajs.append(y_clean)

        # NODE trajectory
        y_node = get_node_predictions(model, [ic], t_train_eval)[0]
        train_node_trajs.append(y_node)

        # SyMANTIC trajectory
        y_sym = integrate_from_ic(rhs_symantic, ic, t_train_eval)
        train_symantic_trajs.append(y_sym)

        # Calculate metrics
        mse = np.mean((y_clean - y_sym) ** 2) if not np.isnan(y_sym).any() else float("inf")
        train_mses.append(mse)

    worst_train_idx = np.argmax(train_mses)
    print(f"\nWorst training trajectory index: {worst_train_idx} (MSE: {train_mses[worst_train_idx]:.6f})")

    # 6. Integrate and evaluate on Test ICs (0-20s, OOD, 41 points)
    t_test_eval = np.linspace(0.0, 20.0, 41)
    test_mses = []
    test_r2s = []
    test_clean_trajs = []
    test_node_trajs = []
    test_symantic_trajs = []

    for ic in test_ics:
        # Ground truth clean trajectory
        sol_clean = simulate_system(ic, [0.0, 20.0], len(t_test_eval), integrator="solve_ivp", return_dataframe=False)
        y_clean = sol_clean["states"]
        test_clean_trajs.append(y_clean)

        # NODE trajectory
        y_node = get_node_predictions(model, [ic], t_test_eval)[0]
        test_node_trajs.append(y_node)

        # SyMANTIC trajectory
        y_sym = integrate_from_ic(rhs_symantic, ic, t_test_eval)
        test_symantic_trajs.append(y_sym)

        # Calculate metrics
        if not np.isnan(y_sym).any():
            mse = np.mean((y_clean - y_sym) ** 2)
            ss_res = np.sum((y_clean - y_sym) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean, axis=0)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-8)
        else:
            mse = float("inf")
            r2 = -1e6

        test_mses.append(mse)
        test_r2s.append(r2)

    # Filter out failed integrations for index selection
    valid_test_mses = [m if m != float("inf") else 1e9 for m in test_mses]
    best_test_idx = np.argmin(valid_test_mses)
    worst_test_idx = np.argmax(valid_test_mses)

    print(f"Best test trajectory index: {best_test_idx} (MSE: {test_mses[best_test_idx]:.6f}, R2: {test_r2s[best_test_idx]:.4f})")
    print(f"Worst test trajectory index: {worst_test_idx} (MSE: {test_mses[worst_test_idx]:.6f}, R2: {test_r2s[worst_test_idx]:.4f})")

    # 7. Generate Trajectory Fit Plot (Worst Train, Best Test, Worst Test)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex='row')
    states_names = ["u", "v", "y"]

    # Trajectories to show
    show_configs = [
        ("Worst Train Traj", train_clean_trajs[worst_train_idx], train_node_trajs[worst_train_idx], train_symantic_trajs[worst_train_idx], t_train, train_ics[worst_train_idx]),
        ("Best Test Traj (OOD)", test_clean_trajs[best_test_idx], test_node_trajs[best_test_idx], test_symantic_trajs[best_test_idx], None, None),
        ("Worst Test Traj (OOD)", test_clean_trajs[worst_test_idx], test_node_trajs[worst_test_idx], test_symantic_trajs[worst_test_idx], None, None)
    ]

    for row_idx, (title, y_clean, y_node, y_sym, t_tr_pts, ic_val) in enumerate(show_configs):
        t_pts = t_train_eval if row_idx == 0 else t_test_eval
        for state_idx in range(3):
            ax = axes[row_idx, state_idx]
            
            # True Clean
            ax.plot(t_pts, y_clean[:, state_idx], 'k-', linewidth=2, label="True (Clean)")
            
            # NODE
            ax.plot(t_pts, y_node[:, state_idx], 'b--', linewidth=1.5, label="NODE")
            
            # SyMANTIC
            if not np.isnan(y_sym).any():
                ax.plot(t_pts, y_sym[:, state_idx], 'm:', linewidth=2, label="SyMANTIC")
            
            # For train, draw training points as red dots
            if row_idx == 0 and t_tr_pts is not None and ic_val is not None:
                # Retrieve the training points generated from NODE
                y_tr_node = get_node_predictions(model, [ic_val], t_tr_pts)[0]
                ax.scatter(t_tr_pts, y_tr_node[:, state_idx], color='red', alpha=0.6, s=15, label="NODE Train Data")

            if state_idx == 0:
                ax.set_ylabel(f"{title}\nState {states_names[state_idx]}", fontsize=12, weight='bold')
            else:
                ax.set_ylabel(f"State {states_names[state_idx]}", fontsize=10)
            
            if row_idx == 0:
                ax.set_title(f"State {states_names[state_idx]} Comparison", fontsize=12)
            if row_idx == 2:
                ax.set_xlabel("Time (s)", fontsize=10)
            
            if row_idx == 0 and state_idx == 0:
                ax.legend(fontsize=8, loc='upper right')

    plt.suptitle("SyMANTIC Distilled Model Trajectory Fitting Results", fontsize=16, weight='bold')
    plt.tight_layout()
    traj_plot_path = ABLATION_DIR / "trajectory_fits_pct_0.2.png"
    plt.savefig(traj_plot_path, dpi=150)
    plt.close()
    print(f"Saved trajectory fits figure to {traj_plot_path}")

    # 8. Plot distributions of R2 and MSE for test trajectories
    # Exclude failed integrations (MSE = inf, R2 = extremely negative) for plotting convenience
    filtered_mses = [m for m in test_mses if m != float("inf")]
    filtered_r2s = [r for r in test_r2s if r > -5.0]  # truncate extreme outliers for visualization

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE Distribution
    axes[0].hist(filtered_mses, bins=10, color='tab:blue', edgecolor='k', alpha=0.7)
    axes[0].set_title("Distribution of Trajectory MSE (Test OOD)", fontsize=14, weight='bold')
    axes[0].set_xlabel("Mean Squared Error (MSE)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # R2 Distribution
    axes[1].hist(filtered_r2s, bins=10, color='tab:orange', edgecolor='k', alpha=0.7)
    axes[1].set_title(r"Distribution of Trajectory $R^2$ (Test OOD)", fontsize=14, weight='bold')
    axes[1].set_xlabel(r"Coefficient of Determination ($R^2$)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("SyMANTIC Generalization Performance Metrics Distribution", fontsize=16, weight='bold')
    plt.tight_layout()
    dist_plot_path = ABLATION_DIR / "metrics_distribution_pct_0.2.png"
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"Saved metric distributions figure to {dist_plot_path}")

    # Save summary report text
    report = {
        "discovered_equations": {
            "du/dt": sr_res["equations"][0],
            "dv/dt": sr_res["equations"][1],
            "dy/dt": sr_res["equations"][2],
        },
        "training_worst_traj_idx": int(worst_train_idx),
        "training_worst_traj_mse": float(train_mses[worst_train_idx]),
        "test_best_traj_idx": int(best_test_idx),
        "test_best_traj_mse": float(test_mses[best_test_idx]),
        "test_best_traj_r2": float(test_r2s[best_test_idx]),
        "test_worst_traj_idx": int(worst_test_idx),
        "test_worst_traj_mse": float(test_mses[worst_test_idx]),
        "test_worst_traj_r2": float(test_r2s[worst_test_idx]),
        "average_test_mse": float(np.mean(filtered_mses)) if len(filtered_mses) > 0 else float("nan"),
        "average_test_r2": float(np.mean(filtered_r2s)) if len(filtered_r2s) > 0 else float("nan"),
    }
    
    report_path = ABLATION_DIR / "ablation_report_pct_0.2.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved analysis report to {report_path}")

    elapsed = time.time() - start_time
    print(f"\nExperiment completed successfully in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
