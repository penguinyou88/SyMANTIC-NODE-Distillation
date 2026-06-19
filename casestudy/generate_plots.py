#!/usr/bin/env python3
"""
Generate detailed plots for Spruce-budworm noise study without re-running regression.
Loads equations from equations.json and NODE checkpoints from models/.
Ensures independent y-axis scaling for subfigures.
"""

import os
import sys
import json
import re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Sibling SyMANTIC directory for import setup
sibling_symantic_path = str(REPO_ROOT.parent / "SyMANTIC")
if sibling_symantic_path not in sys.path:
    sys.path.insert(0, sibling_symantic_path)

# Import configurations & data generators from existing updated script
from casestudy.run_noise_study_updated import (
    ODEFunc,
    generate_data,
    partition_data,
    get_node_predictions,
    integrate_from_ic,
    DEVICE,
    NOISE_LEVELS,
    NODE_HIDDEN_DIM,
    TRAIN_W0,
    RESULTS_DIR,
    MODELS_DIR,
    T_SPLIT,
)

# ---------------------------------------------------------------------------
# Helper functions for RHS construction from strings
# ---------------------------------------------------------------------------

def _make_rhs_from_sindy_equations(equations, feature_names=("w",)):
    """Create an ODE RHS function from SINDy equation strings, converting 'coeff 1' and inserting '*'."""
    processed_eqs = []
    for eq in equations:
        # Convert SINDy's "coef 1" to "coef * 1"
        eq = re.sub(r"\b([0-9.eE-]+)\s+1\b", r"\1 * 1", eq)
        # Insert * between coefficient and variable (e.g. -0.095 w -> -0.095 * w)
        eq = re.sub(r"\b([0-9.eE-]+)\s+([a-zA-Z])", r"\1 * \2", eq)
        eq = re.sub(r"\^", "**", eq)
        processed_eqs.append(eq)

    def rhs(t, y):
        local_vars = {name: float(y[i]) for i, name in enumerate(feature_names)}
        local_vars["np"] = np
        local_vars["t"] = float(t)

        result = np.zeros(len(equations))
        for k, eq in enumerate(processed_eqs):
            try:
                result[k] = float(eval(eq, {"__builtins__": {}}, local_vars))
            except Exception as e:
                print(f"Error evaluating SINDy equation '{eq}': {e}")
                result[k] = 0.0
        return result

    return rhs


def _make_rhs_from_symantic_equations(equations, feature_names=("w",)):
    """Create an ODE RHS function from SyMANTIC equation strings."""
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
            except Exception as e:
                print(f"Error evaluating SyMANTIC equation '{eq}': {e}")
                result[k] = 0.0
        return result

    return rhs


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_16_trajectories_ext_t(model, sindy_eqs, symantic_eqs, splits, t_eval, noise_std):
    train_ics = splits["train_ics"]
    n_ics = len(train_ics)
    if n_ics < 16:
        indices = np.arange(n_ics)
    else:
        # Sample 16 evenly-spaced ICs
        sorted_indices = np.argsort(train_ics[:, 0])
        indices = [sorted_indices[int(i * (n_ics - 1) / 15)] for i in range(16)]
        
    ics_show = train_ics[indices]
    clean_show = splits["clean_train"][indices]
    clean_ext_show = splits["clean_ext_t"][indices]
    clean_full = np.concatenate([clean_show, clean_ext_show], axis=1)
    noisy_show = splits["train_trajs"][indices]
    
    pred_node = get_node_predictions(model, ics_show, t_eval)
    
    if sindy_eqs:
        rhs_sindy = _make_rhs_from_sindy_equations(sindy_eqs)
        pred_sindy = np.array([integrate_from_ic(rhs_sindy, ic, t_eval) for ic in ics_show])
    else:
        pred_sindy = np.full((len(ics_show), len(t_eval), 1), np.nan)
        
    if symantic_eqs:
        rhs_symantic = _make_rhs_from_symantic_equations(symantic_eqs)
        pred_symantic = np.array([integrate_from_ic(rhs_symantic, ic, t_eval) for ic in ics_show])
    else:
        pred_symantic = np.full((len(ics_show), len(t_eval), 1), np.nan)
        
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), sharex=True, sharey=False)
    axes = axes.flatten()
    
    t_train = t_eval[:len(clean_show[0])]
    
    for i in range(len(ics_show)):
        ax = axes[i]
        ax.plot(t_eval, clean_full[i, :, 0], 'k-', linewidth=2, label='True (Clean)' if i == 0 else "")
        ax.scatter(t_train, noisy_show[i, :, 0], color='red', alpha=0.4, s=8, label='Noisy Train' if i == 0 else "")
        ax.plot(t_eval, pred_node[i, :, 0], 'b--', linewidth=1.5, label='NODE' if i == 0 else "")
        
        if not np.isnan(pred_sindy[i]).all():
            ax.plot(t_eval, pred_sindy[i, :, 0], 'g-.', linewidth=1.5, label='SINDy' if i == 0 else "")
        if not np.isnan(pred_symantic[i]).all():
            ax.plot(t_eval, pred_symantic[i, :, 0], 'm:', linewidth=1.5, label='SyMANTIC' if i == 0 else "")
            
        ax.axvline(x=T_SPLIT, color='gray', linestyle=':')
        ax.set_title(f"IC = {ics_show[i, 0]:.3f}", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax.legend(fontsize=8)
            
    for i in range(12, 16):
        axes[i].set_xlabel("Time (s)", fontsize=10)
    for i in [0, 4, 8, 12]:
        axes[i].set_ylabel("w", fontsize=10)
        
    plt.suptitle(f"16 Trajectories: Time Extrapolation (Noise = {noise_std:.3f})", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = RESULTS_DIR / f"trajectories_16_ext_t_noise_{noise_std}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved 16 trajectories time-extrapolation plot to {plot_path}")


def plot_16_trajectories_ext_x0(model, sindy_eqs, symantic_eqs, splits, t_eval, noise_std):
    test_ics = splits["test_ext_x0_ics"]
    n_ics = len(test_ics)
    if n_ics < 16:
        indices = np.arange(n_ics)
    else:
        # Sample 16 evenly-spaced ICs
        sorted_indices = np.argsort(test_ics[:, 0])
        indices = [sorted_indices[int(i * (n_ics - 1) / 15)] for i in range(16)]
        
    ics_show = test_ics[indices]
    clean_show = splits["clean_ext_x0"][indices]
    
    pred_node = get_node_predictions(model, ics_show, t_eval)
    
    if sindy_eqs:
        rhs_sindy = _make_rhs_from_sindy_equations(sindy_eqs)
        pred_sindy = np.array([integrate_from_ic(rhs_sindy, ic, t_eval) for ic in ics_show])
    else:
        pred_sindy = np.full((len(ics_show), len(t_eval), 1), np.nan)
        
    if symantic_eqs:
        rhs_symantic = _make_rhs_from_symantic_equations(symantic_eqs)
        pred_symantic = np.array([integrate_from_ic(rhs_symantic, ic, t_eval) for ic in ics_show])
    else:
        pred_symantic = np.full((len(ics_show), len(t_eval), 1), np.nan)
        
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), sharex=True, sharey=False)
    axes = axes.flatten()
    
    for i in range(len(ics_show)):
        ax = axes[i]
        ax.plot(t_eval, clean_show[i, :, 0], 'k-', linewidth=2, label='True (Clean)' if i == 0 else "")
        ax.plot(t_eval, pred_node[i, :, 0], 'b--', linewidth=1.5, label='NODE' if i == 0 else "")
        
        if not np.isnan(pred_sindy[i]).all():
            ax.plot(t_eval, pred_sindy[i, :, 0], 'g-.', linewidth=1.5, label='SINDy' if i == 0 else "")
        if not np.isnan(pred_symantic[i]).all():
            ax.plot(t_eval, pred_symantic[i, :, 0], 'm:', linewidth=1.5, label='SyMANTIC' if i == 0 else "")
            
        ax.set_title(f"IC = {ics_show[i, 0]:.3f}", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax.legend(fontsize=8)
            
    for i in range(12, 16):
        axes[i].set_xlabel("Time (s)", fontsize=10)
    for i in [0, 4, 8, 12]:
        axes[i].set_ylabel("w", fontsize=10)
        
    plt.suptitle(f"16 Trajectories: State Extrapolation (Noise = {noise_std:.3f})", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = RESULTS_DIR / f"trajectories_16_ext_x0_noise_{noise_std}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved 16 trajectories state-extrapolation plot to {plot_path}")


def plot_metrics_distribution(model, sindy_eqs, symantic_eqs, splits, t_eval, t_split_index, noise_std):
    rhs_sindy = _make_rhs_from_sindy_equations(sindy_eqs) if sindy_eqs else None
    rhs_symantic = _make_rhs_from_symantic_equations(symantic_eqs) if symantic_eqs else None
    
    # 1. Ext-T metrics
    train_ics = splits["train_ics"]
    clean_ext_t = splits["clean_ext_t"]
    
    pred_node_full = get_node_predictions(model, train_ics, t_eval)
    pred_node_ext_t = pred_node_full[:, t_split_index:, :]
    
    mse_node_ext_t, r2_node_ext_t = [], []
    mse_sindy_ext_t, r2_sindy_ext_t = [], []
    mse_symantic_ext_t, r2_symantic_ext_t = [], []
    
    for i, ic in enumerate(train_ics):
        y_true = clean_ext_t[i, :, 0]
        y_true_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_true_mean) ** 2)
        
        # NODE
        y_pred_node = pred_node_ext_t[i, :, 0]
        mse = np.mean((y_true - y_pred_node) ** 2)
        mse_node_ext_t.append(mse)
        r2 = 1 - np.sum((y_true - y_pred_node) ** 2) / ss_tot if ss_tot > 0 else 0.0
        r2_node_ext_t.append(max(r2, -2.0))
        
        # SINDy
        if rhs_sindy is not None:
            pred_full = integrate_from_ic(rhs_sindy, ic, t_eval)
            y_pred_sindy = pred_full[t_split_index:, 0]
            if not np.isnan(y_pred_sindy).any():
                mse = np.mean((y_true - y_pred_sindy) ** 2)
                mse_sindy_ext_t.append(mse)
                r2 = 1 - np.sum((y_true - y_pred_sindy) ** 2) / ss_tot if ss_tot > 0 else 0.0
                r2_sindy_ext_t.append(max(r2, -2.0))
            else:
                mse_sindy_ext_t.append(np.nan)
                r2_sindy_ext_t.append(-2.0)
        else:
            mse_sindy_ext_t.append(np.nan)
            r2_sindy_ext_t.append(-2.0)
            
        # SyMANTIC
        if rhs_symantic is not None:
            pred_full = integrate_from_ic(rhs_symantic, ic, t_eval)
            y_pred_symantic = pred_full[t_split_index:, 0]
            if not np.isnan(y_pred_symantic).any():
                mse = np.mean((y_true - y_pred_symantic) ** 2)
                mse_symantic_ext_t.append(mse)
                r2 = 1 - np.sum((y_true - y_pred_symantic) ** 2) / ss_tot if ss_tot > 0 else 0.0
                r2_symantic_ext_t.append(max(r2, -2.0))
            else:
                mse_symantic_ext_t.append(np.nan)
                r2_symantic_ext_t.append(-2.0)
        else:
            mse_symantic_ext_t.append(np.nan)
            r2_symantic_ext_t.append(-2.0)
            
    # 2. Ext-X0 metrics
    test_ics = splits["test_ext_x0_ics"]
    clean_ext_x0 = splits["clean_ext_x0"]
    
    pred_node_ext_x0 = get_node_predictions(model, test_ics, t_eval)
    
    mse_node_ext_x0, r2_node_ext_x0 = [], []
    mse_sindy_ext_x0, r2_sindy_ext_x0 = [], []
    mse_symantic_ext_x0, r2_symantic_ext_x0 = [], []
    
    for i, ic in enumerate(test_ics):
        y_true = clean_ext_x0[i, :, 0]
        y_true_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_true_mean) ** 2)
        
        # NODE
        y_pred_node = pred_node_ext_x0[i, :, 0]
        mse = np.mean((y_true - y_pred_node) ** 2)
        mse_node_ext_x0.append(mse)
        r2 = 1 - np.sum((y_true - y_pred_node) ** 2) / ss_tot if ss_tot > 0 else 0.0
        r2_node_ext_x0.append(max(r2, -2.0))
        
        # SINDy
        if rhs_sindy is not None:
            pred_full = integrate_from_ic(rhs_sindy, ic, t_eval)
            y_pred_sindy = pred_full[:, 0]
            if not np.isnan(y_pred_sindy).any():
                mse = np.mean((y_true - y_pred_sindy) ** 2)
                mse_sindy_ext_x0.append(mse)
                r2 = 1 - np.sum((y_true - y_pred_sindy) ** 2) / ss_tot if ss_tot > 0 else 0.0
                r2_sindy_ext_x0.append(max(r2, -2.0))
            else:
                mse_sindy_ext_x0.append(np.nan)
                r2_sindy_ext_x0.append(-2.0)
        else:
            mse_sindy_ext_x0.append(np.nan)
            r2_sindy_ext_x0.append(-2.0)
            
        # SyMANTIC
        if rhs_symantic is not None:
            pred_full = integrate_from_ic(rhs_symantic, ic, t_eval)
            y_pred_symantic = pred_full[:, 0]
            if not np.isnan(y_pred_symantic).any():
                mse = np.mean((y_true - y_pred_symantic) ** 2)
                mse_symantic_ext_x0.append(mse)
                r2 = 1 - np.sum((y_true - y_pred_symantic) ** 2) / ss_tot if ss_tot > 0 else 0.0
                r2_symantic_ext_x0.append(max(r2, -2.0))
            else:
                mse_symantic_ext_x0.append(np.nan)
                r2_symantic_ext_x0.append(-2.0)
        else:
            mse_symantic_ext_x0.append(np.nan)
            r2_symantic_ext_x0.append(-2.0)
            
    # Filter out NaNs for plotting distributions
    def clean_data(arr):
        arr = np.array(arr)
        return arr[~np.isnan(arr)]
        
    data_mse_ext_t = [clean_data(mse_node_ext_t), clean_data(mse_sindy_ext_t), clean_data(mse_symantic_ext_t)]
    data_r2_ext_t = [clean_data(r2_node_ext_t), clean_data(r2_sindy_ext_t), clean_data(r2_symantic_ext_t)]
    
    data_mse_ext_x0 = [clean_data(mse_node_ext_x0), clean_data(mse_sindy_ext_x0), clean_data(mse_symantic_ext_x0)]
    data_r2_ext_x0 = [clean_data(r2_node_ext_x0), clean_data(r2_sindy_ext_x0), clean_data(r2_symantic_ext_x0)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    labels = ["NODE", "SINDy", "SyMANTIC"]
    
    # Top Left: Ext-T MSE
    axes[0, 0].boxplot(data_mse_ext_t, labels=labels)
    axes[0, 0].set_title("Time Extrapolation (Ext-T) MSE", fontsize=12, weight="bold")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, linestyle=":", alpha=0.5)
    
    # Top Right: Ext-T R2
    axes[0, 1].boxplot(data_r2_ext_t, labels=labels)
    axes[0, 1].set_title("Time Extrapolation (Ext-T) R² (clamped min=-2.0)", fontsize=12, weight="bold")
    axes[0, 1].set_ylabel("R²")
    axes[0, 1].set_ylim(-2.1, 1.1)
    axes[0, 1].grid(True, linestyle=":", alpha=0.5)
    
    # Bottom Left: Ext-X0 MSE
    axes[1, 0].boxplot(data_mse_ext_x0, labels=labels)
    axes[1, 0].set_title("State Extrapolation (Ext-X0) MSE", fontsize=12, weight="bold")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, linestyle=":", alpha=0.5)
    
    # Bottom Right: Ext-X0 R2
    axes[1, 1].boxplot(data_r2_ext_x0, labels=labels)
    axes[1, 1].set_title("State Extrapolation (Ext-X0) R² (clamped min=-2.0)", fontsize=12, weight="bold")
    axes[1, 1].set_ylabel("R²")
    axes[1, 1].set_ylim(-2.1, 1.1)
    axes[1, 1].grid(True, linestyle=":", alpha=0.5)
    
    plt.suptitle(f"Performance Metric Distributions per Trajectory (Noise = {noise_std:.3f})", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = RESULTS_DIR / f"metrics_distribution_noise_{noise_std}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved metric distributions plot to {plot_path}")


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("GENERATING NOISE STUDY PLOTS (WITHOUT SR SWEET RERUN)")
    print("Using independent y-axis scaling (sharey=False)")
    print("=" * 70)

    # 1. Load Equations from JSON
    eq_json_path = RESULTS_DIR / "equations.json"
    if not eq_json_path.exists():
        raise FileNotFoundError(f"equations.json not found at {eq_json_path}. Please run run_noise_study_updated.py first.")
    
    with open(eq_json_path, "r") as f:
        eq_json = json.load(f)
    print(f"Loaded equations from {eq_json_path}")

    # 2. Re-generate dataset partition to evaluate trajectories
    ics, t_eval, dataset_clean, dataset_noisy = generate_data()
    dataset_splits, train_indices, test_x0_indices, t_split_index, t_train, t_ext = \
        partition_data(ics, t_eval, dataset_clean, dataset_noisy)

    # 3. Load NODE checkpoints and plot figures per noise level
    for std in NOISE_LEVELS:
        print(f"\n--- Processing Noise Level: {std:.3f} ---")
        model_path = MODELS_DIR / f"node_noise_std_{std}.pt"
        if not model_path.exists():
            print(f"Warning: Checkpoint not found at {model_path}. Skipping...")
            continue
        
        # Load NODE model
        model = ODEFunc(state_dim=1, hidden_dim=NODE_HIDDEN_DIM, scale=TRAIN_W0[1]).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        print(f"Loaded NODE model from {model_path}")

        # Get SINDy and SyMANTIC equations for this noise level
        noise_key = str(std)
        sindy_eqs = eq_json["discovered"].get(noise_key, {}).get("sindy", [])
        symantic_eqs = eq_json["discovered"].get(noise_key, {}).get("symantic", [])

        # Generate plots
        splits = dataset_splits[std]
        plot_16_trajectories_ext_t(model, sindy_eqs, symantic_eqs, splits, t_eval, std)
        plot_16_trajectories_ext_x0(model, sindy_eqs, symantic_eqs, splits, t_eval, std)
        plot_metrics_distribution(model, sindy_eqs, symantic_eqs, splits, t_eval, t_split_index, std)

    print("\n" + "=" * 70)
    print("All plots generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
