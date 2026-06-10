#!/usr/bin/env python3
"""
SyMANTIC Full Time Horizon Study (t = 20s)
===========================================
Trains a new Neural ODE model using the entire 20s time span (noise 0.01).
Saves the new model separately and runs symbolic regressions (SINDy, SyMANTIC L1 and L0)
on the new NODE gradients to see if the full time series enables better rediscovery.
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
N_SOBOL_ICS = 64

NOISE_LEVEL = 0.01
NODE_HIDDEN_DIM = 64
NODE_N_EPOCHS = 1000
NODE_LR = 1e-3

RESULTS_DIR = SCRIPT_DIR / "results" / "full_time_study"
MODELS_DIR = RESULTS_DIR / "models"
NODE_MODEL_PATH = MODELS_DIR / "node_noise_0.01_t20.pt"

GROUND_TRUTH_EQUATIONS = {
    "dudt": "1/(1 + v**2) - u",
    "dvdt": "1/(1 + (1 + u/(1+y)**2)**2) - v",
    "dydt": "0.1*y",
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

    splits = {
        # Note training is over the FULL t span [0, 20]
        "train_trajs": dataset_noisy[train_indices, :, :],
        "train_ics": ics[train_indices],
        "clean_train": dataset_clean[train_indices, :, :],
        "clean_ext_x0": dataset_clean[test_x0_indices, :, :],
        "test_ext_x0_ics": ics[test_x0_indices],
    }

    return ics, t_eval, splits


# NODE training
def train_node_from_scratch(train_data, t_eval):
    func = ODEFunc(hidden_dim=NODE_HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(func.parameters(), lr=NODE_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5, min_lr=1e-5
    )

    n_traj = train_data.shape[0]
    n_val = max(1, int(n_traj * 0.15))
    n_train = n_traj - n_val

    train_trajs = train_data[:n_train]
    val_trajs = train_data[n_train:]

    t_tensor = torch.tensor(t_eval, dtype=torch.float32).to(DEVICE)
    
    obs_trajs = torch.tensor(train_trajs, dtype=torch.float32).to(DEVICE)
    y0s = obs_trajs[:, 0, :]

    val_obs_trajs = torch.tensor(val_trajs, dtype=torch.float32).to(DEVICE)
    val_y0s = val_obs_trajs[:, 0, :]

    best_val_loss = float("inf")
    best_state = None
    patience = 100
    epochs_no_improve = 0

    print("Training NODE model on full time horizon...")
    for epoch in range(1, NODE_N_EPOCHS + 1):
        func.train()
        optimizer.zero_grad()

        pred_trajs = odeint(func, y0s, t_tensor, method="rk4")
        pred_trajs = pred_trajs.permute(1, 0, 2)

        loss = torch.mean((pred_trajs - obs_trajs) ** 2)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()
        
        func.eval()
        with torch.no_grad():
            pred_val_trajs = odeint(func, val_y0s, t_tensor, method="rk4")
            pred_val_trajs = pred_val_trajs.permute(1, 0, 2)
            val_loss = torch.mean((pred_val_trajs - val_obs_trajs) ** 2).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in func.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.8f} | Val Loss: {val_loss:.8f} | Best Val: {best_val_loss:.8f}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.8f}")
            break

    if best_state is not None:
        func.load_state_dict(best_state)

    return func


def get_node_predictions(model, y0s, t_points):
    t_tensor = torch.tensor(t_points, dtype=torch.float32).to(DEVICE)
    y0_tensor = torch.tensor(y0s, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = odeint(model, y0_tensor, t_tensor, method="rk4")
        preds = preds.permute(1, 0, 2).cpu().numpy()
    return preds


def evaluate_node(model, splits, t_eval):
    print("Evaluating NODE model...")
    # Train set
    pred_train = get_node_predictions(model, splits["train_ics"], t_eval)
    mse_train = float(np.mean((pred_train - splits["clean_train"]) ** 2))
    rmse_train = float(np.sqrt(mse_train))
    ss_res = np.sum((pred_train - splits["clean_train"]) ** 2)
    ss_tot = np.sum((splits["clean_train"] - np.mean(splits["clean_train"], axis=(0, 1))) ** 2)
    r2_train = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Ext-X0
    pred_ext_x0 = get_node_predictions(model, splits["test_ext_x0_ics"], t_eval)
    mse_ext_x0 = float(np.mean((pred_ext_x0 - splits["clean_ext_x0"]) ** 2))
    rmse_ext_x0 = float(np.sqrt(mse_ext_x0))
    ss_res_x0 = np.sum((pred_ext_x0 - splits["clean_ext_x0"]) ** 2)
    ss_tot_x0 = np.sum((splits["clean_ext_x0"] - np.mean(splits["clean_ext_x0"], axis=(0, 1))) ** 2)
    r2_ext_x0 = float(1 - ss_res_x0 / ss_tot_x0) if ss_tot_x0 > 0 else float("nan")

    print(f"  Train:  RMSE={rmse_train:.6f} | R2={r2_train:.4f}")
    print(f"  Ext-X0: RMSE={rmse_ext_x0:.6f} | R2={r2_ext_x0:.4f}")

    return {
        "method": "NODE",
        "rmse_train": rmse_train,
        "r2_train": r2_train,
        "rmse_ext_x0": rmse_ext_x0,
        "r2_ext_x0": r2_ext_x0,
    }


# SINDy regression
def run_sindy_sr(states, grads):
    import pysindy as ps
    feature_names = ["u", "v", "y"]
    library = ps.PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
    optimizer = ps.STLSQ(threshold=0.01, alpha=1e-2)
    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    model.fit(states, t=0.2, x_dot=grads, feature_names=feature_names)
    equations = model.equations()
    complexities = [calculate_complexity(eq, method="operator_count") for eq in equations]
    return {
        "method": "SINDy",
        "model": model,
        "equations": equations,
        "complexities": complexities,
    }


# Equation Closeness
def compute_closeness(disc_eq_str, true_eq_str):
    disc_eq_str = disc_eq_str.replace("^", "**")
    true_eq_str = true_eq_str.replace("^", "**")
    u, v, y, C = sp.symbols('u v y C')
    try:
        disc_expr = sp.sympify(disc_eq_str)
        true_expr = sp.sympify(true_eq_str)
        disc_expr_sub = disc_expr.subs(C, 1.0)
        
        diff = disc_expr_sub - true_expr
        is_equiv = bool(diff.simplify() == 0)

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
def _make_rhs_sindy(sindy_model):
    def rhs(t, y):
        return sindy_model.predict(y.reshape(1, -1)).flatten()
    return rhs


def _make_rhs_symantic(equations, feature_names=("u", "v", "y")):
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
        local_vars["C"] = 1.0

        result = np.zeros(len(equations))
        for k, eq in enumerate(processed_eqs):
            try:
                result[k] = float(eval(eq, {"__builtins__": {}}, local_vars))
            except Exception:
                result[k] = 0.0
        return result
    return rhs


def integrate_from_ic(rhs_func, y0, t_points, method="RK45"):
    """Integrate an ODE RHS from initial condition y0 over t_points."""
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
            return sol.y.T  # (n_steps, state_dim)
        else:
            return np.full((len(t_points), len(y0)), np.nan)
    except Exception:
        return np.full((len(t_points), len(y0)), np.nan)


def evaluate_integration(rhs_func, splits, t_eval):
    # Train
    train_preds = []
    for ic in splits["train_ics"]:
        pred = integrate_from_ic(rhs_func, ic, t_eval)
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

    # Ext-X0
    ext_x0_preds = []
    for ic in splits["test_ext_x0_ics"]:
        pred = integrate_from_ic(rhs_func, ic, t_eval)
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
        "rmse_ext_x0": rmse_ext_x0,
        "r2_ext_x0": r2_ext_x0,
    }


def main():
    print("=" * 80)
    print("RUNNING STUDY: FULL TIME HORIZON (t = 20s)")
    print("=" * 80)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate/load data
    ics, t_eval, splits = get_dataset()

    # 2. Train NODE from scratch over 20s or load if exists
    node_model = ODEFunc(hidden_dim=NODE_HIDDEN_DIM).to(DEVICE)
    if NODE_MODEL_PATH.exists():
        print(f"Loading existing NODE model from {NODE_MODEL_PATH}...")
        node_model.load_state_dict(torch.load(NODE_MODEL_PATH, map_location=DEVICE))
    else:
        node_model = train_node_from_scratch(splits["train_trajs"], t_eval)
        torch.save(node_model.state_dict(), NODE_MODEL_PATH)
        print(f"NODE model saved to {NODE_MODEL_PATH}")

    # 3. Evaluate NODE
    node_metrics = evaluate_node(node_model, splits, t_eval)

    # 4. Extract gradients
    states = splits["train_trajs"].reshape(-1, 3)
    states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        grads = node_model(0, states_tensor).cpu().numpy()

    # 5. SINDy
    print("\nRunning SINDy...")
    sindy_res = run_sindy_sr(states, grads)
    print("SINDy equations:")
    for i, eq in enumerate(sindy_res["equations"]):
        print(f"  State {i}: {eq}")

    # 6. SyMANTIC (L1 vs L0)
    from symantic.model import SymanticModel
    from unittest.mock import patch

    feature_names = ["u", "v", "y"]
    target_names = ["dudt", "dvdt", "dydt"]
    operators = ["+", "-", "*", "/", "^2", "^-1"]

    performance_rows = []
    equations_rows = []

    # Cache NODE Predictions for plotting
    y0_train_tensor = torch.tensor(splits["train_ics"][:1], dtype=torch.float32).to(DEVICE)
    y0_test_tensor = torch.tensor(splits["test_ext_x0_ics"][:1], dtype=torch.float32).to(DEVICE)
    t_tensor = torch.tensor(t_eval, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_train_node = odeint(node_model, y0_train_tensor, t_tensor, method="rk4").permute(1, 0, 2).cpu().numpy()[0]
        pred_test_node = odeint(node_model, y0_test_tensor, t_tensor, method="rk4").permute(1, 0, 2).cpu().numpy()[0]

    # Run SyMANTIC Ablations (L1 vs L0)
    configurations = {
        "L1": {"regularization": "l1", "description": "L1 Regularization (distillation)"},
        "L0": {"regularization": "l0", "description": "L0 Regularization (distillation)"},
    }

    integrated_preds = {}

    for config_name, conf in configurations.items():
        print(f"\nRunning SyMANTIC ({config_name})...")
        case_utopia_eqs = []
        case_highest_acc_eqs = []

        for k, target in enumerate(target_names):
            print(f"  Fitting {target}...")
            # Inject constant feature C = 1.0
            df = pd.DataFrame(states.astype(np.float32), columns=feature_names)
            df["C"] = 1.0
            df.insert(0, target, grads[:, k].astype(np.float32))

            model = SymanticModel(
                df,
                operators=operators,
                n_term=3,
                n_expansion=4,
                sis_features=100,
                level_pruning=True,
                regularization=conf["regularization"],
                disp=False,
                metrics=[0.1, 0.99],
                max_features=300000,
            )

            with patch("builtins.input", return_value="no"):
                result = model.fit()

            _res, pareto = result
            
            # Utopia
            utopia_eq = str(result.equation)
            case_utopia_eqs.append(utopia_eq)
            u_comp = calculate_complexity(utopia_eq, method="operator_count")

            # Highest Accuracy
            if pareto is not None and len(pareto) > 0 and "Loss" in pareto.columns:
                row_idx = pareto["Loss"].idxmin()
                highest_acc_eq = str(pareto.iloc[row_idx]["Equation"])
            else:
                highest_acc_eq = utopia_eq
            h_comp = calculate_complexity(highest_acc_eq, method="operator_count")
            
            case_highest_acc_eqs.append(highest_acc_eq)

            # Closeness checks
            true_eq = GROUND_TRUTH_EQUATIONS[target]
            u_equiv, u_mad = compute_closeness(utopia_eq, true_eq)
            h_equiv, h_mad = compute_closeness(highest_acc_eq, true_eq)

            print(f"    Utopia:  {utopia_eq} | Complexity={u_comp} | Equiv={u_equiv} | MAD={u_mad:.6f}")
            print(f"    Highest: {highest_acc_eq} | Complexity={h_comp} | Equiv={h_equiv} | MAD={h_mad:.6f}")

            equations_rows.append({
                "method": f"SyMANTIC-{config_name}",
                "selection": "Utopia",
                "state": target,
                "equation": utopia_eq,
                "complexity": u_comp,
                "is_equiv": u_equiv,
                "mad": u_mad,
            })
            equations_rows.append({
                "method": f"SyMANTIC-{config_name}",
                "selection": "Highest Accuracy",
                "state": target,
                "equation": highest_acc_eq,
                "complexity": h_comp,
                "is_equiv": h_equiv,
                "mad": h_mad,
            })

        # Trajectory evaluation
        rhs_utopia = _make_rhs_symantic(case_utopia_eqs)
        utopia_perf = evaluate_integration(rhs_utopia, splits, t_eval)
        utopia_perf.update({"method": f"SyMANTIC-{config_name}", "selection": "Utopia"})
        performance_rows.append(utopia_perf)

        rhs_highest = _make_rhs_symantic(case_highest_acc_eqs)
        highest_perf = evaluate_integration(rhs_highest, splits, t_eval)
        highest_perf.update({"method": f"SyMANTIC-{config_name}", "selection": "Highest Accuracy"})
        performance_rows.append(highest_perf)

        print(f"  {config_name} Utopia Trajectory RMSE (Train)={utopia_perf['rmse_train']:.6f} | (Ext-X0)={utopia_perf['rmse_ext_x0']:.6f}")
        print(f"  {config_name} Highest Trajectory RMSE (Train)={highest_perf['rmse_train']:.6f} | (Ext-X0)={highest_perf['rmse_ext_x0']:.6f}")

        # Integrate for plotting
        pred_train_utopia = integrate_from_ic(rhs_utopia, splits["train_ics"][0], t_eval)
        pred_test_utopia = integrate_from_ic(rhs_utopia, splits["test_ext_x0_ics"][0], t_eval)

        pred_train_highest = integrate_from_ic(rhs_highest, splits["train_ics"][0], t_eval)
        pred_test_highest = integrate_from_ic(rhs_highest, splits["test_ext_x0_ics"][0], t_eval)

        integrated_preds[config_name] = {
            "utopia_train": pred_train_utopia,
            "utopia_test": pred_test_utopia,
            "highest_train": pred_train_highest,
            "highest_test": pred_test_highest,
        }

    # Evaluate SINDy integration
    print("\nIntegrating and evaluating SINDy...")
    rhs_sindy = _make_rhs_sindy(sindy_res["model"])
    sindy_perf = evaluate_integration(rhs_sindy, splits, t_eval)
    sindy_perf.update({"method": "SINDy", "selection": "Utopia"})  # placeholder Utopia selection
    performance_rows.append(sindy_perf)
    
    pred_train_sindy = integrate_from_ic(rhs_sindy, splits["train_ics"][0], t_eval)
    pred_test_sindy = integrate_from_ic(rhs_sindy, splits["test_ext_x0_ics"][0], t_eval)

    # 7. Add NODE row to performance results
    performance_rows.append({
        "method": "NODE",
        "selection": "Utopia",
        "rmse_train": node_metrics["rmse_train"],
        "r2_train": node_metrics["r2_train"],
        "rmse_ext_x0": node_metrics["rmse_ext_x0"],
        "r2_ext_x0": node_metrics["r2_ext_x0"],
    })

    # Save Tables
    df_perf = pd.DataFrame(performance_rows)
    df_perf.to_csv(RESULTS_DIR / "performance_table.csv", index=False)

    df_eqs = pd.DataFrame(equations_rows)
    # Add SINDy rows to equations table
    sindy_eq_rows = []
    for i, eq in enumerate(sindy_res["equations"]):
        true_eq = GROUND_TRUTH_EQUATIONS[target_names[i]]
        equiv, mad = compute_closeness(eq, true_eq)
        sindy_eq_rows.append({
            "method": "SINDy",
            "selection": "Utopia",
            "state": target_names[i],
            "equation": eq,
            "complexity": sindy_res["complexities"][i],
            "is_equiv": equiv,
            "mad": mad,
        })
    df_eqs = pd.concat([df_eqs, pd.DataFrame(sindy_eq_rows)], ignore_index=True)
    df_eqs.to_csv(RESULTS_DIR / "equations_table.csv", index=False)

    # Plot trajectories
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    states_names_lbl = ["u", "v", "y"]
    
    for state_idx in range(3):
        # Train IC
        ax = axes[state_idx, 0]
        ax.plot(t_eval, splits["clean_train"][0, :, state_idx], 'k-', linewidth=2, label='True (Clean)' if state_idx == 0 else "")
        ax.scatter(t_eval, splits["train_trajs"][0, :, state_idx], color='red', alpha=0.4, s=12, label='Noisy Train Data' if state_idx == 0 else "")
        ax.plot(t_eval, pred_train_node[:, state_idx], 'b--', linewidth=1.5, label='NODE Pred' if state_idx == 0 else "")
        ax.plot(t_eval, pred_train_sindy[:, state_idx], 'g-.', linewidth=1.5, label='SINDy Pred' if state_idx == 0 else "")
        
        # Utopia L1 & L0
        pred_L1 = integrated_preds["L1"]["utopia_train"]
        pred_L0 = integrated_preds["L0"]["utopia_train"]
        if not np.isnan(pred_L1).all():
            ax.plot(t_eval, pred_L1[:, state_idx], 'm:', linewidth=1.5, label='SyMANTIC-L1 (Utopia) Pred' if state_idx == 0 else "")
        if not np.isnan(pred_L0).all():
            ax.plot(t_eval, pred_L0[:, state_idx], 'c--', linewidth=1.5, label='SyMANTIC-L0 (Utopia) Pred' if state_idx == 0 else "")

        ax.set_ylabel(states_names_lbl[state_idx])
        if state_idx == 0:
            ax.set_title("Train IC Trajectories (t = 20s)")
            ax.legend()

        # Test IC
        ax = axes[state_idx, 1]
        ax.plot(t_eval, splits["clean_ext_x0"][0, :, state_idx], 'k-', linewidth=2, label='True (Clean)' if state_idx == 0 else "")
        ax.plot(t_eval, pred_test_node[:, state_idx], 'b--', linewidth=1.5, label='NODE Pred' if state_idx == 0 else "")
        ax.plot(t_eval, pred_test_sindy[:, state_idx], 'g-.', linewidth=1.5, label='SINDy Pred' if state_idx == 0 else "")
        
        # Utopia L1 & L0
        pred_test_L1 = integrated_preds["L1"]["utopia_test"]
        pred_test_L0 = integrated_preds["L0"]["utopia_test"]
        if not np.isnan(pred_test_L1).all():
            ax.plot(t_eval, pred_test_L1[:, state_idx], 'm:', linewidth=1.5, label='SyMANTIC-L1 (Utopia) Pred' if state_idx == 0 else "")
        if not np.isnan(pred_test_L0).all():
            ax.plot(t_eval, pred_test_L0[:, state_idx], 'c--', linewidth=1.5, label='SyMANTIC-L0 (Utopia) Pred' if state_idx == 0 else "")

        ax.set_ylabel(states_names_lbl[state_idx])
        if state_idx == 0:
            ax.set_title("OOD IC Trajectories (t = 20s)")
            ax.legend()

    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    plt.suptitle("Study: Full Time Horizon (t = 20s) Distillation Comparison", fontsize=14, weight='bold')
    plt.tight_layout()

    plot_path = RESULTS_DIR / "trajectories.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved comparison figure to {plot_path}")

    print("\n" + "=" * 80)
    print("FULL TIME HORIZON STUDY COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
