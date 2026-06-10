#!/usr/bin/env python3
"""
SyMANTIC Ablation Study on Equation Rediscovery — Phase 2
==========================================================
Studies targeted L0-regularization settings to recover the true genetic switch equations.
Introduces a constant feature column C = 1.0 to enable rational denominator discovery.
"""

import os
import sys
import json
import time
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint
from scipy.integrate import solve_ivp
from scipy.stats import qmc
import sympy as sp

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from casestudy.simulate import dynamic_system, simulate_system
from benchmark.complexity import calculate_complexity

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

U0_BOUNDS = [0.5, 2.0]
V0_BOUNDS = [0.05, 0.2]
Y0_BOUNDS = [0.05, 0.2]

TRAIN_U0 = [0.6, 1.9]
TRAIN_V0 = [0.06, 0.19]
TRAIN_Y0 = [0.06, 0.19]

T_SPAN = [0.0, 20.0]
N_STEPS = 101
T_SPLIT = 15.0
N_SOBOL_ICS = 64

NOISE_LEVEL = 0.01
NODE_HIDDEN_DIM = 64

RESULTS_DIR = SCRIPT_DIR / "results" / "ablation_phase2"
NODE_MODEL_PATH = SCRIPT_DIR / "results" / "2nd iteration" / "models" / f"node_noise_{NOISE_LEVEL}.pt"

GROUND_TRUTH_EQUATIONS = {
    "dudt": "1/(1 + v**2) - u",
    "dvdt": "1/(1 + (1 + u/(1+y)**2)**2) - v",
    "dydt": "0.1*y",
}

# Ablation Cases Definition (Phase 2)
ABLATION_CASES = {
    "4A": {
        "description": "Baseline: L0 depth=3",
        "operators": ["+", "-", "*", "/", "^2", "^-1"],
        "n_expansion": 3,
        "regularization": "l0",
        "sis_features": 100,
        "use_constant": False,
    },
    "4B": {
        "description": "L0 depth=3 + Constant C=1.0",
        "operators": ["+", "-", "*", "/", "^2", "^-1"],
        "n_expansion": 3,
        "regularization": "l0",
        "sis_features": 100,
        "use_constant": True,
    },
    "4C": {
        "description": "L0 depth=4 (No Constant)",
        "operators": ["+", "-", "*", "/", "^2", "^-1"],
        "n_expansion": 4,
        "regularization": "l0",
        "sis_features": 100,
        "use_constant": False,
    },
    "4D": {
        "description": "L0 depth=3 + sis_features=200",
        "operators": ["+", "-", "*", "/", "^2", "^-1"],
        "n_expansion": 3,
        "regularization": "l0",
        "sis_features": 200,
        "use_constant": False,
    },
    "4E": {
        "description": "Full Rediscovery: Constant C=1.0 + depth=4 + sis_features=150",
        "operators": ["+", "-", "*", "/", "^2", "^-1"],
        "n_expansion": 4,
        "regularization": "l0",
        "sis_features": 150,
        "use_constant": True,
    },
}


# NODE rhs
class ODEFunc(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, y):
        return self.net(y)


# Data Setup
def get_dataset():
    # Sobol sampling
    sampler = qmc.Sobol(d=3, scramble=True, seed=SEED)
    sample = sampler.random(n=N_SOBOL_ICS)
    l_bounds = [U0_BOUNDS[0], V0_BOUNDS[0], Y0_BOUNDS[0]]
    u_bounds = [U0_BOUNDS[1], V0_BOUNDS[1], Y0_BOUNDS[1]]
    ics = qmc.scale(sample, l_bounds, u_bounds)
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS)

    dataset_clean = []
    for ic in ics:
        res = simulate_system(
            initial_condition=ic,
            t_span=T_SPAN,
            n_steps=N_STEPS,
            integrator="solve_ivp",
            return_dataframe=False,
        )
        dataset_clean.append(res["states"])
    dataset_clean = np.array(dataset_clean)

    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, NOISE_LEVEL, size=dataset_clean.shape)
    dataset_noisy = dataset_clean + noise

    # Partitioning
    eps = 1e-7
    train_mask = (
        (ics[:, 0] >= TRAIN_U0[0] - eps) & (ics[:, 0] <= TRAIN_U0[1] + eps) &
        (ics[:, 1] >= TRAIN_V0[0] - eps) & (ics[:, 1] <= TRAIN_V0[1] + eps) &
        (ics[:, 2] >= TRAIN_Y0[0] - eps) & (ics[:, 2] <= TRAIN_Y0[1] + eps)
    )
    train_indices = np.where(train_mask)[0]
    test_x0_indices = np.where(~train_mask)[0]

    t_split_index = np.searchsorted(t_eval, T_SPLIT, side="left")
    t_train = t_eval[:t_split_index]

    splits = {
        "train_trajs": dataset_noisy[train_indices, :t_split_index, :],
        "train_ics": ics[train_indices],
        "clean_train": dataset_clean[train_indices, :t_split_index, :],
        "clean_ext_t": dataset_clean[train_indices, t_split_index:, :],
        "clean_ext_x0": dataset_clean[test_x0_indices, :, :],
        "test_ext_x0_ics": ics[test_x0_indices],
    }

    return ics, t_eval, t_split_index, t_train, splits


def get_node_gradients(splits):
    # Load NODE
    print(f"Loading NODE model from {NODE_MODEL_PATH}...")
    model = ODEFunc(hidden_dim=NODE_HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(NODE_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    train_trajs = splits["train_trajs"]
    states = train_trajs.reshape(-1, 3)

    states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        grads = model(0, states_tensor).cpu().numpy()

    return model, states, grads


# Equation Closeness
def compute_closeness(disc_eq_str, true_eq_str):
    disc_eq_str = disc_eq_str.replace("^", "**")
    true_eq_str = true_eq_str.replace("^", "**")

    u, v, y, C = sp.symbols('u v y C')
    try:
        disc_expr = sp.sympify(disc_eq_str)
        true_expr = sp.sympify(true_eq_str)

        # Substitute constant feature C with 1.0 for simplification check
        disc_expr_sub = disc_expr.subs(C, 1.0)
        
        diff = disc_expr_sub - true_expr
        is_equiv = bool(diff.simplify() == 0)

        # Numerical comparison over a grid
        rng = np.random.default_rng(SEED)
        u_vals = rng.uniform(0.6, 1.9, size=100)
        v_vals = rng.uniform(0.06, 0.19, size=100)
        y_vals = rng.uniform(0.06, 0.19, size=100)

        diff_vals = []
        for uv, vv, yv in zip(u_vals, v_vals, y_vals):
            d_val = float(disc_expr.evalf(subs={u: uv, v: vv, y: yv, C: 1.0}))
            t_val = float(true_expr.evalf(subs={u: uv, v: vv, y: yv}))
            diff_vals.append(abs(d_val - t_val))

        mad = float(np.mean(diff_vals))
        return is_equiv, mad
    except Exception as e:
        return False, float("inf")


# Integration RHS
def _make_rhs(equations, feature_names=("u", "v", "y")):
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
        local_vars["C"] = 1.0  # Provide constant feature value

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
            max_step=0.5,
        )
        if sol.success:
            return sol.y.T
        else:
            return np.full((len(t_points), len(y0)), np.nan)
    except Exception:
        return np.full((len(t_points), len(y0)), np.nan)


def evaluate_integration(rhs_func, splits, t_eval, t_split_index):
    t_train_pts = t_eval[:t_split_index]
    t_full = t_eval

    # Train
    train_preds = []
    for ic in splits["train_ics"]:
        pred = integrate_from_ic(rhs_func, ic, t_train_pts)
        train_preds.append(pred)
    train_preds = np.array(train_preds)

    valid_train = ~np.any(np.isnan(train_preds), axis=(1, 2))
    if valid_train.sum() > 0:
        y_true = splits["clean_train"][valid_train]
        y_pred = train_preds[valid_train]
        mse_train = float(np.mean((y_pred - y_true) ** 2))
        rmse_train = float(np.sqrt(mse_train))
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=(0, 1))) ** 2)
        r2_train = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        rmse_train, r2_train = float("nan"), float("nan")

    # Ext-T
    ext_t_preds = []
    for ic in splits["train_ics"]:
        pred = integrate_from_ic(rhs_func, ic, t_full)
        ext_t_preds.append(pred)
    ext_t_preds = np.array(ext_t_preds)

    ext_t_tail = ext_t_preds[:, t_split_index:, :]
    valid_ext_t = ~np.any(np.isnan(ext_t_tail), axis=(1, 2))
    if valid_ext_t.sum() > 0:
        y_true = splits["clean_ext_t"][valid_ext_t]
        y_pred = ext_t_tail[valid_ext_t]
        mse_ext_t = float(np.mean((y_pred - y_true) ** 2))
        rmse_ext_t = float(np.sqrt(mse_ext_t))
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=(0, 1))) ** 2)
        r2_ext_t = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        rmse_ext_t, r2_ext_t = float("nan"), float("nan")

    # Ext-X0
    ext_x0_preds = []
    for ic in splits["test_ext_x0_ics"]:
        pred = integrate_from_ic(rhs_func, ic, t_full)
        ext_x0_preds.append(pred)
    ext_x0_preds = np.array(ext_x0_preds)

    valid_ext_x0 = ~np.any(np.isnan(ext_x0_preds), axis=(1, 2))
    if valid_ext_x0.sum() > 0:
        y_true = splits["clean_ext_x0"][valid_ext_x0]
        y_pred = ext_x0_preds[valid_ext_x0]
        mse_ext_x0 = float(np.mean((y_pred - y_true) ** 2))
        rmse_ext_x0 = float(np.sqrt(mse_ext_x0))
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=(0, 1))) ** 2)
        r2_ext_x0 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        rmse_ext_x0, r2_ext_x0 = float("nan"), float("nan")

    return {
        "rmse_train": rmse_train,
        "r2_train": r2_train,
        "rmse_ext_t": rmse_ext_t,
        "r2_ext_t": r2_ext_t,
        "rmse_ext_x0": rmse_ext_x0,
        "r2_ext_x0": r2_ext_x0,
    }


def main():
    print("=" * 80)
    print("STARTING SyMANTIC ABLATION STUDY PHASE 2")
    print(f"Device: {DEVICE} | Noise Level: {NOISE_LEVEL}")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load Data and Gradients
    ics, t_eval, t_split_index, t_train, splits = get_dataset()
    node_model, states, grads = get_node_gradients(splits)

    from symantic.model import SymanticModel
    from unittest.mock import patch

    feature_names = ["u", "v", "y"]
    target_names = ["dudt", "dvdt", "dydt"]

    performance_rows = []
    equations_rows = []

    # Cache NODE Predictions for plotting
    t_tensor = torch.tensor(t_eval, dtype=torch.float32).to(DEVICE)
    y0_train_tensor = torch.tensor(splits["train_ics"][:1], dtype=torch.float32).to(DEVICE)
    y0_test_tensor = torch.tensor(splits["test_ext_x0_ics"][:1], dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        pred_train_node = odeint(node_model, y0_train_tensor, t_tensor, method="rk4").permute(1, 0, 2).cpu().numpy()[0]
        pred_test_node = odeint(node_model, y0_test_tensor, t_tensor, method="rk4").permute(1, 0, 2).cpu().numpy()[0]

    # Run Ablation Cases
    for case_name, case_conf in ABLATION_CASES.items():
        print("\n" + "=" * 60)
        print(f"RUNNING CASE {case_name}: {case_conf['description']}")
        print("=" * 60)

        case_utopia_eqs = []
        case_highest_acc_eqs = []
        case_utopia_complexities = []
        case_highest_acc_complexities = []

        # Fit each equation
        for k, target in enumerate(target_names):
            print(f"  Fitting {target}...")
            
            # Setup columns based on use_constant config
            if case_conf.get("use_constant", False):
                df = pd.DataFrame(states.astype(np.float32), columns=feature_names)
                df["C"] = 1.0
            else:
                df = pd.DataFrame(states.astype(np.float32), columns=feature_names)

            df.insert(0, target, grads[:, k].astype(np.float32))

            model = SymanticModel(
                df,
                operators=case_conf["operators"],
                n_term=3,
                n_expansion=case_conf["n_expansion"],
                sis_features=case_conf["sis_features"],
                level_pruning=True,
                regularization=case_conf["regularization"],
                disp=False,
                metrics=[0.1, 0.99],
                max_features=case_conf.get("max_features", 300000),
            )

            with patch("builtins.input", return_value="no"):
                result = model.fit()

            # Unpack
            _res, pareto = result

            # 1. Utopia Model selection
            utopia_eq = str(result.equation)
            case_utopia_eqs.append(utopia_eq)
            case_utopia_complexities.append(calculate_complexity(utopia_eq, method="operator_count"))

            # 2. Highest Accuracy selection
            if pareto is not None and len(pareto) > 0 and "Loss" in pareto.columns:
                row_idx = pareto["Loss"].idxmin()
                highest_acc_eq = str(pareto.iloc[row_idx]["Equation"])
            else:
                highest_acc_eq = utopia_eq
            
            case_highest_acc_eqs.append(highest_acc_eq)
            case_highest_acc_complexities.append(calculate_complexity(highest_acc_eq, method="operator_count"))

            # Ground Truth Closeness evaluation
            true_eq = GROUND_TRUTH_EQUATIONS[target]
            
            # Utopia Closeness
            u_equiv, u_mad = compute_closeness(utopia_eq, true_eq)
            # Highest Accuracy Closeness
            h_equiv, h_mad = compute_closeness(highest_acc_eq, true_eq)

            print(f"    Utopia:   {utopia_eq} | Complexity={case_utopia_complexities[-1]} | Equiv={u_equiv} | Closeness MAD={u_mad:.6f}")
            print(f"    Highest:  {highest_acc_eq} | Complexity={case_highest_acc_complexities[-1]} | Equiv={h_equiv} | Closeness MAD={h_mad:.6f}")

            # Record in Equations table
            equations_rows.append({
                "case": case_name,
                "selection": "Utopia",
                "state": target,
                "equation": utopia_eq,
                "complexity": case_utopia_complexities[-1],
                "is_equiv": u_equiv,
                "mad": u_mad,
            })
            equations_rows.append({
                "case": case_name,
                "selection": "Highest Accuracy",
                "state": target,
                "equation": highest_acc_eq,
                "complexity": case_highest_acc_complexities[-1],
                "is_equiv": h_equiv,
                "mad": h_mad,
            })

        # Integrate and evaluate both selections
        print(f"  Integrating and evaluating trajectories for Case {case_name}...")
        
        # Utopia Integration
        rhs_utopia = _make_rhs(case_utopia_eqs)
        utopia_perf = evaluate_integration(rhs_utopia, splits, t_eval, t_split_index)
        utopia_perf.update({"case": case_name, "selection": "Utopia"})
        performance_rows.append(utopia_perf)

        # Highest Accuracy Integration
        rhs_highest = _make_rhs(case_highest_acc_eqs)
        highest_perf = evaluate_integration(rhs_highest, splits, t_eval, t_split_index)
        highest_perf.update({"case": case_name, "selection": "Highest Accuracy"})
        performance_rows.append(highest_perf)

        print(f"    Utopia:  Train RMSE={utopia_perf['rmse_train']:.6f} | Ext-T RMSE={utopia_perf['rmse_ext_t']:.6f} | Ext-X0 RMSE={utopia_perf['rmse_ext_x0']:.6f}")
        print(f"    Highest: Train RMSE={highest_perf['rmse_train']:.6f} | Ext-T RMSE={highest_perf['rmse_ext_t']:.6f} | Ext-X0 RMSE={highest_perf['rmse_ext_x0']:.6f}")

        # Integrate trajectories for single trajectory plots
        pred_train_utopia = integrate_from_ic(rhs_utopia, splits["train_ics"][0], t_eval)
        pred_test_utopia = integrate_from_ic(rhs_utopia, splits["test_ext_x0_ics"][0], t_eval)

        pred_train_highest = integrate_from_ic(rhs_highest, splits["train_ics"][0], t_eval)
        pred_test_highest = integrate_from_ic(rhs_highest, splits["test_ext_x0_ics"][0], t_eval)

        # Plot comparison figure for the case
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
        states_names_lbl = ["u", "v", "y"]
        clean_full_train = np.concatenate([splits["clean_train"][0, :, :], splits["clean_ext_t"][0, :, :]], axis=0)

        for state_idx in range(3):
            # Train IC (Time extrapolation)
            ax = axes[state_idx, 0]
            ax.plot(t_eval, clean_full_train[:, state_idx], 'k-', linewidth=2, label='True (Clean)' if state_idx == 0 else "")
            ax.scatter(t_train, splits["train_trajs"][0, :, state_idx], color='red', alpha=0.4, s=12, label='Noisy Train Data' if state_idx == 0 else "")
            ax.plot(t_eval, pred_train_node[:, state_idx], 'b--', linewidth=1.5, label='NODE Pred' if state_idx == 0 else "")
            
            if not np.isnan(pred_train_utopia).all():
                ax.plot(t_eval, pred_train_utopia[:, state_idx], 'g-.', linewidth=1.5, label='SyMANTIC (Utopia) Pred' if state_idx == 0 else "")
            if not np.isnan(pred_train_highest).all():
                ax.plot(t_eval, pred_train_highest[:, state_idx], 'm:', linewidth=1.5, label='SyMANTIC (Highest Acc) Pred' if state_idx == 0 else "")
            
            ax.axvline(x=T_SPLIT, color='gray', linestyle=':')
            ax.set_ylabel(states_names_lbl[state_idx])
            if state_idx == 0:
                ax.set_title("Train IC (Time Extrapolation)")
                ax.legend()

            # OOD IC (Space extrapolation)
            ax = axes[state_idx, 1]
            ax.plot(t_eval, splits["clean_ext_x0"][0, :, state_idx], 'k-', linewidth=2, label='True (Clean)' if state_idx == 0 else "")
            ax.plot(t_eval, pred_test_node[:, state_idx], 'b--', linewidth=1.5, label='NODE Pred' if state_idx == 0 else "")
            
            if not np.isnan(pred_test_utopia).all():
                ax.plot(t_eval, pred_test_utopia[:, state_idx], 'g-.', linewidth=1.5, label='SyMANTIC (Utopia) Pred' if state_idx == 0 else "")
            if not np.isnan(pred_test_highest).all():
                ax.plot(t_eval, pred_test_highest[:, state_idx], 'm:', linewidth=1.5, label='SyMANTIC (Highest Acc) Pred' if state_idx == 0 else "")
            
            ax.set_ylabel(states_names_lbl[state_idx])
            if state_idx == 0:
                ax.set_title("OOD IC (Space Extrapolation)")
                ax.legend()

        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 1].set_xlabel("Time (s)")
        plt.suptitle(f"Ablation Phase 2 Case {case_name}: {case_conf['description']}", fontsize=14, weight='bold')
        plt.tight_layout()

        plot_path = RESULTS_DIR / f"case_{case_name}_trajectories.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"    Saved trajectory comparison figure to {plot_path}")

    # Compile and Save summary CSVs
    df_perf = pd.DataFrame(performance_rows)
    df_perf_path = RESULTS_DIR / "performance_table.csv"
    df_perf.to_csv(df_perf_path, index=False)

    df_eqs = pd.DataFrame(equations_rows)
    df_eqs_path = RESULTS_DIR / "equations_table.csv"
    df_eqs.to_csv(df_eqs_path, index=False)

    print("\n" + "=" * 80)
    print("ABLATION STUDY PHASE 2 COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
