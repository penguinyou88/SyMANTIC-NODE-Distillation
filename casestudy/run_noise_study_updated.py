#!/usr/bin/env python3
"""
Noise-Level Impact Study on NODE Distillation
==============================================
Studies how observation noise affects:
1. Neural ODE model quality
2. Symbolic regression (SINDy, SyMANTIC) distillation from NODE gradients

System: Spruce-budworm outbreak model (1-state)
    dw_dt = r * w * (1 - w/a) - w**2 / (1 + w**2)
    with r = 0.5, a = 10.0
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
from scipy.integrate import solve_ivp
from scipy.stats import qmc

# Suppress convergence warnings from pysindy/sklearn during sweeps
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path setup — ensure repo root and sibling SyMANTIC directory are importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add symantic directory for local imports
symantic_path = str(REPO_ROOT / "symantic")
if symantic_path not in sys.path:
    sys.path.insert(0, symantic_path)

# Add sibling SyMANTIC directory for SymanticModel imports
sibling_symantic_path = str(REPO_ROOT.parent / "SyMANTIC")
if sibling_symantic_path not in sys.path:
    sys.path.insert(0, sibling_symantic_path)

from casestudy.simulate import dynamic_system, simulate_system
from benchmark.complexity import calculate_complexity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Initial condition bounds (full domain for Sobol sampling, matching extrapolation range)
W0_BOUNDS = [0.05, 16.0]

# Training IC subdomain (inner mask, matching training state window)
TRAIN_W0 = [0.05, 8.0]

# Time configuration
T_SPAN = [0.0, 12.0]
N_STEPS = 61         # dt = 0.2s (61 steps from 0 to 12)
T_SPLIT = 5.0        # train on [0, 5.0], test-ext-t on [5.0, 12.0]

# Noise levels to study (absolute standard deviation)
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1]

# NODE configuration
NODE_HIDDEN_DIM = 96
NODE_N_EPOCHS = 1500
NODE_LR = 3e-3
NODE_LAMBDA_COLLOC = 0.0

# Number of Sobol ICs
N_SOBOL_ICS = 80

# Output directories
RESULTS_DIR = SCRIPT_DIR / "results" / "spruce_budworm_noise_study"
MODELS_DIR = RESULTS_DIR / "models"

# Ground truth equations for reference
GROUND_TRUTH_EQUATIONS = {
    "dw/dt": "0.5*w*(1 - w/10.0) - w**2/(1 + w**2)",
}


# ===========================================================================
# Section 1: Data Generation
# ===========================================================================

def generate_data():
    """Generate Sobol-sampled ICs, simulate clean trajectories, add noise."""
    print("=" * 70)
    print("SECTION 1: Data Generation")
    print("=" * 70)

    # Sobol sampling
    sampler = qmc.Sobol(d=1, scramble=True, seed=SEED)
    sample = sampler.random(n=N_SOBOL_ICS)
    l_bounds = [W0_BOUNDS[0]]
    u_bounds = [W0_BOUNDS[1]]
    ics = qmc.scale(sample, l_bounds, u_bounds)

    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS)

    # Simulate clean trajectories
    print(f"Simulating {len(ics)} clean trajectories...")
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
    dataset_clean = np.array(dataset_clean)  # (N_ICs, N_STEPS, 1)

    # Add absolute Gaussian noise and pin initial conditions at t=0
    rng = np.random.default_rng(SEED)
    dataset_noisy = {}
    for noise_std in NOISE_LEVELS:
        noise = rng.normal(0, noise_std, size=dataset_clean.shape)
        noisy = dataset_clean + noise
        noisy[:, 0, :] = ics # pin t=0 to exact IC
        noisy = np.maximum(noisy, 1e-8)
        dataset_noisy[noise_std] = noisy

    print(f"Clean dataset shape: {dataset_clean.shape}")
    print(f"Noise levels: {NOISE_LEVELS} (interpreted as absolute standard deviation)")

    return ics, t_eval, dataset_clean, dataset_noisy


# ===========================================================================
# Section 2: Dataset Partitioning
# ===========================================================================

def partition_data(ics, t_eval, dataset_clean, dataset_noisy):
    """Split into train/test by IC subdomain mask and time."""
    print("\n" + "=" * 70)
    print("SECTION 2: Dataset Partitioning")
    print("=" * 70)

    # Mask-based split: ICs inside training domain [0.05, 8.0] -> training
    eps = 1e-7
    train_mask = (
        (ics[:, 0] >= TRAIN_W0[0] - eps) & (ics[:, 0] <= TRAIN_W0[1] + eps)
    )

    train_indices = np.where(train_mask)[0]
    test_x0_indices = np.where(~train_mask)[0]

    # Time split index
    t_split_index = np.searchsorted(t_eval, T_SPLIT, side="left")
    t_train = t_eval[:t_split_index]
    t_ext = t_eval[t_split_index:]

    print(f"Training ICs: {len(train_indices)} / {len(ics)} (inner domain mask)")
    print(f"Test (OOD x0) ICs: {len(test_x0_indices)} / {len(ics)}")
    print(f"Time train steps: {len(t_train)} (t=[{t_train[0]:.1f}, {t_train[-1]:.1f}])")
    print(f"Time ext steps: {len(t_ext)} (t=[{t_ext[0]:.1f}, {t_ext[-1]:.1f}])")

    # Build splits for each noise level
    # NOTE: Training uses noisy data, but test ground truth is CLEAN
    dataset_splits = {}
    for std in NOISE_LEVELS:
        noisy = dataset_noisy[std]
        dataset_splits[std] = {
            # Training: noisy data, train ICs, t in [0, T_SPLIT)
            "train_trajs": noisy[train_indices, :t_split_index, :],
            "train_ics": ics[train_indices],
            # Clean ground truth for evaluation
            "clean_train": dataset_clean[train_indices, :t_split_index, :],
            "clean_ext_t": dataset_clean[train_indices, t_split_index:, :],
            "clean_ext_x0": dataset_clean[test_x0_indices, :, :],
            "test_ext_x0_ics": ics[test_x0_indices],
        }

    return dataset_splits, train_indices, test_x0_indices, t_split_index, t_train, t_ext


# ===========================================================================
# Section 3: NODE Model Definition and Training
# ===========================================================================

class ODEFunc(nn.Module):
    """Neural ODE right-hand side: scaled 1-state MLP with Tanh activation."""
    def __init__(self, state_dim=1, hidden_dim=96, scale=8.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, y):
        return self.net(y / self.scale)


def train_node(train_data, t_train, n_epochs=NODE_N_EPOCHS, lr=NODE_LR, use_val_split=False):
    """
    Train a NODE on multi-trajectory data with CosineAnnealingLR.
    """
    func = ODEFunc(state_dim=1, hidden_dim=NODE_HIDDEN_DIM, scale=TRAIN_W0[1]).to(DEVICE)
    optimizer = torch.optim.Adam(func.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    n_traj = train_data.shape[0]
    if use_val_split:
        n_val = max(1, int(n_traj * 0.15))
        n_train = n_traj - n_val
        train_trajs = train_data[:n_train]
        val_trajs = train_data[n_train:]
    else:
        train_trajs = train_data
        val_trajs = train_data

    t_tensor = torch.tensor(t_train, dtype=torch.float32).to(DEVICE)
    obs_trajs = torch.tensor(train_trajs, dtype=torch.float32).to(DEVICE)
    y0s = obs_trajs[:, 0, :]

    val_obs_trajs = torch.tensor(val_trajs, dtype=torch.float32).to(DEVICE)
    val_y0s = val_obs_trajs[:, 0, :]

    train_log = []
    best_val_loss = float("inf")
    best_state = None
    patience = 300
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        func.train()
        optimizer.zero_grad()

        # Batched integration (permute to B, T, D)
        pred_trajs = odeint(func, y0s, t_tensor, method="rk4").permute(1, 0, 2)

        # Trajectory MSE loss
        loss = torch.mean((pred_trajs - obs_trajs) ** 2)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation loss
        func.eval()
        with torch.no_grad():
            pred_val_trajs = odeint(func, val_y0s, t_tensor, method="rk4").permute(1, 0, 2)
            val_loss = torch.mean((pred_val_trajs - val_obs_trajs) ** 2).item()

        loss_val = loss.item()
        train_log.append({"epoch": epoch, "loss": loss_val, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in func.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 100 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:5d} | Loss: {loss_val:.8f} | Val Loss: {val_loss:.8f} | Best Val: {best_val_loss:.8f} | LR: {current_lr:.2e}")

        if use_val_split and epochs_no_improve >= patience:
            print(f"  Early stopping triggered at epoch {epoch} (no validation improvement for {patience} epochs). Best Val Loss: {best_val_loss:.8f}")
            break

    # Restore best model
    if best_state is not None and use_val_split:
        func.load_state_dict(best_state)

    return func, train_log


def train_or_load_nodes(dataset_splits, t_train):
    """Train NODE models for each noise level, or load from disk if available."""
    print("\n" + "=" * 70)
    print("SECTION 3: NODE Training")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    node_models = {}
    training_logs = {}

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    for std in NOISE_LEVELS:
        model_path = MODELS_DIR / f"node_noise_std_{std}.pt"

        if model_path.exists():
            print(f"\n--- Loading saved NODE for noise={std:.3f} from {model_path} ---")
            func = ODEFunc(state_dim=1, hidden_dim=NODE_HIDDEN_DIM, scale=TRAIN_W0[1]).to(DEVICE)
            func.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            func.eval()
            node_models[std] = func
            training_logs[std] = [{"epoch": 0, "loss": float("nan"), "note": "loaded from disk"}]
        else:
            print(f"\n--- Training NODE for noise={std:.3f} ---")
            train_data = dataset_splits[std]["train_trajs"]
            func, log = train_node(train_data, t_train, use_val_split=(std > 0.0))
            func.eval()

            # Save model
            torch.save(func.state_dict(), model_path)
            print(f"  Model saved to {model_path}")

            node_models[std] = func
            training_logs[std] = log

    # Save training logs
    log_path = RESULTS_DIR / "node_training_logs.json"
    serializable_logs = {
        str(std): [{"epoch": e["epoch"], "loss": e["loss"], "val_loss": e.get("val_loss", float("nan"))} for e in log]
        for std, log in training_logs.items()
    }
    log_path.write_text(json.dumps(serializable_logs, indent=2))

    return node_models, training_logs


# ===========================================================================
# Section 4: NODE Evaluation
# ===========================================================================

def get_node_predictions(model, y0s, t_points):
    """Integrate NODE from multiple initial conditions."""
    t_tensor = torch.tensor(t_points, dtype=torch.float32).to(DEVICE)
    y0_tensor = torch.tensor(y0s, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = odeint(model, y0_tensor, t_tensor, method="rk4")  # (T, B, D)
        preds = preds.permute(1, 0, 2).cpu().numpy()  # (B, T, D)
    return preds


def evaluate_nodes(node_models, dataset_splits, t_eval, t_split_index):
    """Evaluate NODE models on train, ext-t, and ext-x0 test sets with R2."""
    print("\n" + "=" * 70)
    print("SECTION 4: NODE Evaluation")
    print("=" * 70)

    node_metrics = []

    for std in NOISE_LEVELS:
        model = node_models[std]
        splits = dataset_splits[std]

        # --- Train set: integrate from train ICs over [0, T_SPLIT) ---
        t_train_pts = t_eval[:t_split_index]
        pred_train = get_node_predictions(model, splits["train_ics"], t_train_pts)
        mse_train = float(np.mean((pred_train - splits["clean_train"]) ** 2))
        rmse_train = float(np.sqrt(mse_train))
        
        # Compute R2 for train
        y_true_train = splits["clean_train"]
        ss_res_train = np.sum((pred_train - y_true_train) ** 2)
        ss_tot_train = np.sum((y_true_train - np.mean(y_true_train, axis=(0, 1))) ** 2)
        r2_train = float(1 - ss_res_train / ss_tot_train) if ss_tot_train > 0 else float("nan")

        # --- Ext-T: integrate from train ICs over full [0, 12], evaluate [T_SPLIT, 12] ---
        pred_full = get_node_predictions(model, splits["train_ics"], t_eval)
        pred_ext_t = pred_full[:, t_split_index:, :]
        mse_ext_t = float(np.mean((pred_ext_t - splits["clean_ext_t"]) ** 2))
        rmse_ext_t = float(np.sqrt(mse_ext_t))
        
        # Compute R2 for ext-t
        y_true_ext_t = splits["clean_ext_t"]
        ss_res_ext_t = np.sum((pred_ext_t - y_true_ext_t) ** 2)
        ss_tot_ext_t = np.sum((y_true_ext_t - np.mean(y_true_ext_t, axis=(0, 1))) ** 2)
        r2_ext_t = float(1 - ss_res_ext_t / ss_tot_ext_t) if ss_tot_ext_t > 0 else float("nan")

        # --- Ext-X0: integrate from OOD ICs over full [0, 12] ---
        pred_ext_x0 = get_node_predictions(model, splits["test_ext_x0_ics"], t_eval)
        mse_ext_x0 = float(np.mean((pred_ext_x0 - splits["clean_ext_x0"]) ** 2))
        rmse_ext_x0 = float(np.sqrt(mse_ext_x0))
        
        # Compute R2 for ext-x0
        y_true_ext_x0 = splits["clean_ext_x0"]
        ss_res_ext_x0 = np.sum((pred_ext_x0 - y_true_ext_x0) ** 2)
        ss_tot_ext_x0 = np.sum((y_true_ext_x0 - np.mean(y_true_ext_x0, axis=(0, 1))) ** 2)
        r2_ext_x0 = float(1 - ss_res_ext_x0 / ss_tot_ext_x0) if ss_tot_ext_x0 > 0 else float("nan")

        metrics = {
            "noise_level": std,
            "method": "NODE",
            "mse_train": mse_train,
            "rmse_train": rmse_train,
            "r2_train": r2_train,
            "mse_ext_t": mse_ext_t,
            "rmse_ext_t": rmse_ext_t,
            "r2_ext_t": r2_ext_t,
            "mse_ext_x0": mse_ext_x0,
            "rmse_ext_x0": rmse_ext_x0,
            "r2_ext_x0": r2_ext_x0,
        }
        node_metrics.append(metrics)
        print(f"  Noise={std:.3f} | Train RMSE={rmse_train:.6f} (R2={r2_train:.4f}) | "
              f"Ext-T RMSE={rmse_ext_t:.6f} (R2={r2_ext_t:.4f}) | "
              f"Ext-X0 RMSE={rmse_ext_x0:.6f} (R2={r2_ext_x0:.4f})")

    return node_metrics


def plot_gradient_parity(node_models, dataset_splits, t_eval, t_split_index):
    """
    Plot parity of true gradients vs. fitted NODE gradients.
    Shows dwdt in a 1x1 plot for each noise level.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, root_mean_squared_error
    
    for std in NOISE_LEVELS:
        model = node_models[std]
        splits = dataset_splits[std]
        
        # Train data (evaluated on clean states to compare against noise-free gradients)
        train_trajs = splits["clean_train"]  # (n_traj, n_steps_train, 1)
        train_states = train_trajs.reshape(-1, 1)
        
        # Test data (Ext-X0: OOD initial conditions, full time trajectory)
        ext_x0_trajs = splits["clean_ext_x0"]  # (n_traj, n_steps, 1)
        test_states = ext_x0_trajs.reshape(-1, 1)
        
        # Compute true gradients at these states
        true_train_grads = np.array([dynamic_system(0, s) for s in train_states])
        true_test_grads = np.array([dynamic_system(0, s) for s in test_states])
        
        # Compute NODE-predicted gradients at these states
        train_states_tensor = torch.tensor(train_states, dtype=torch.float32).to(DEVICE)
        test_states_tensor = torch.tensor(test_states, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            node_train_grads = model(0, train_states_tensor).cpu().numpy()
            node_test_grads = model(0, test_states_tensor).cpu().numpy()
            
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        
        # Scatter train points
        r2_tr = r2_score(true_train_grads[:, 0], node_train_grads[:, 0])
        ax.scatter(true_train_grads[:, 0], node_train_grads[:, 0], color="tab:blue", alpha=0.5, s=10,
                   label=f"Train (R2={r2_tr:.3f})")
        
        # Scatter test points
        r2_te = r2_score(true_test_grads[:, 0], node_test_grads[:, 0])
        ax.scatter(true_test_grads[:, 0], node_test_grads[:, 0], color="tab:orange", alpha=0.4, s=10,
                   label=f"OOD Test (R2={r2_te:.3f})")
        
        # Plot y = x line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, "k--", alpha=0.7, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel("True dw/dt", fontsize=12)
        ax.set_ylabel("NODE dw/dt", fontsize=12)
        ax.set_title("dw/dt Parity Plot", fontsize=14)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=10)
            
        plt.suptitle(f"NODE Gradient Parity Plot (Noise Level = {std:.3f})", fontsize=16, weight="bold")
        plt.tight_layout()
        
        plot_path = RESULTS_DIR / f"node_parity_noise_pct_{std}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved gradient parity figure to {plot_path}")
        
        # Print gradient parity metrics directly to the terminal
        print(f"  Gradient Parity R2 (Noise={std:.3f}):")
        r2_tr = r2_score(true_train_grads[:, 0], node_train_grads[:, 0])
        r2_te = r2_score(true_test_grads[:, 0], node_test_grads[:, 0])
        print(f"    dw/dt: Train R2 = {r2_tr:6.4f} | Test (OOD) R2 = {r2_te:6.4f}\n")


def plot_all_results(node_models, sr_results, dataset_splits, t_eval, t_split_index):
    """Plot fitting and testing trajectories for NODE, SINDy, and SyMANTIC models for multiple trajectories."""
    import matplotlib.pyplot as plt
    
    t_train = t_eval[:t_split_index]
    
    for std in NOISE_LEVELS:
        model = node_models[std]
        splits = dataset_splits[std]
        sr_res = sr_results.get(std, {})
        
        # NODE predictions for first 3 train ICs and first 3 test ICs
        n_show = 3
        train_ics_show = splits["train_ics"][:n_show]
        test_ics_show = splits["test_ext_x0_ics"][:n_show]
        
        pred_train_node = get_node_predictions(model, train_ics_show, t_eval)  # (3, N_STEPS, 1)
        pred_test_node = get_node_predictions(model, test_ics_show, t_eval)    # (3, N_STEPS, 1)
        
        # SINDy integration
        sindy_res = sr_res.get("sindy")
        pred_train_sindy = []
        pred_test_sindy = []
        if sindy_res is not None:
            rhs_sindy = _make_rhs_from_sindy_model(sindy_res["model"])
            for ic in train_ics_show:
                pred_train_sindy.append(integrate_from_ic(rhs_sindy, ic, t_eval))
            for ic in test_ics_show:
                pred_test_sindy.append(integrate_from_ic(rhs_sindy, ic, t_eval))
            pred_train_sindy = np.array(pred_train_sindy)
            pred_test_sindy = np.array(pred_test_sindy)
        else:
            pred_train_sindy = np.full((n_show, len(t_eval), 1), np.nan)
            pred_test_sindy = np.full((n_show, len(t_eval), 1), np.nan)
            
        # SyMANTIC integration
        symantic_res = sr_res.get("symantic")
        pred_train_symantic = []
        pred_test_symantic = []
        if symantic_res is not None and symantic_res.get("equations") is not None:
            rhs_symantic = _make_rhs_from_symantic_equations(symantic_res["equations"])
            for ic in train_ics_show:
                pred_train_symantic.append(integrate_from_ic(rhs_symantic, ic, t_eval))
            for ic in test_ics_show:
                pred_test_symantic.append(integrate_from_ic(rhs_symantic, ic, t_eval))
            pred_train_symantic = np.array(pred_train_symantic)
            pred_test_symantic = np.array(pred_test_symantic)
        else:
            pred_train_symantic = np.full((n_show, len(t_eval), 1), np.nan)
            pred_test_symantic = np.full((n_show, len(t_eval), 1), np.nan)
        
        # 1 row (w) and 6 columns (3 train, 3 test)
        fig, axes = plt.subplots(1, 6, figsize=(24, 5), sharex=True)
        states_names = ["w"]
        
        # Plot Train ICs (columns 0, 1, 2)
        for c in range(n_show):
            clean_full_train = np.concatenate([splits["clean_train"][c, :, :], splits["clean_ext_t"][c, :, :]], axis=0)
            ax = axes[c]
            ax.plot(t_eval, clean_full_train[:, 0], 'k-', linewidth=2, label='True (Clean)' if c == 0 else "")
            ax.scatter(t_train, splits["train_trajs"][c, :, 0], color='red', alpha=0.4, s=8, 
                       label='Noisy Train Data' if c == 0 else "")
            ax.plot(t_eval, pred_train_node[c, :, 0], 'b--', linewidth=1.5, label='NODE Pred' if c == 0 else "")
            
            if not np.isnan(pred_train_sindy[c]).all():
                ax.plot(t_eval, pred_train_sindy[c, :, 0], 'g-.', linewidth=1.5, label='SINDy Pred' if c == 0 else "")
            if not np.isnan(pred_train_symantic[c]).all():
                ax.plot(t_eval, pred_train_symantic[c, :, 0], 'm:', linewidth=1.5, label='SyMANTIC Pred' if c == 0 else "")
                
            ax.axvline(x=T_SPLIT, color='gray', linestyle=':', label='Time Boundary' if c == 0 else "")
            if c == 0:
                ax.set_ylabel(states_names[0], fontsize=14)
            ax.set_title(f"Train IC {c+1}", fontsize=14)
            if c == 0:
                ax.legend(fontsize=8)
                        
        # Plot OOD ICs (columns 3, 4, 5)
        for c in range(n_show):
            col_idx = c + 3
            ax = axes[col_idx]
            ax.plot(t_eval, splits["clean_ext_x0"][c, :, 0], 'k-', linewidth=2, label='True (Clean)' if c == 0 else "")
            ax.plot(t_eval, pred_test_node[c, :, 0], 'b--', linewidth=1.5, label='NODE Pred' if c == 0 else "")
            
            if not np.isnan(pred_test_sindy[c]).all():
                ax.plot(t_eval, pred_test_sindy[c, :, 0], 'g-.', linewidth=1.5, label='SINDy Pred' if c == 0 else "")
            if not np.isnan(pred_test_symantic[c]).all():
                ax.plot(t_eval, pred_test_symantic[c, :, 0], 'm:', linewidth=1.5, label='SyMANTIC Pred' if c == 0 else "")
                
            ax.set_title(f"OOD IC {c+1}", fontsize=14)
            if c == 0:
                ax.legend(fontsize=8)
                        
        for col_idx in range(6):
            axes[col_idx].set_xlabel("Time (s)", fontsize=12)
            
        plt.suptitle(f"NODE & Distilled SR Model Performance (Noise Level = {std:.3f})", fontsize=18, weight='bold')
        plt.tight_layout()
        
        plot_path = RESULTS_DIR / f"node_trajectories_noise_pct_{std}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved trajectory comparison figure to {plot_path}")


# ===========================================================================
# Section 5: Symbolic Regression (SINDy + SyMANTIC)
# ===========================================================================

def get_node_gradient_data(model, dataset_splits, std):
    """Extract (states, gradients) from NODE for symbolic regression."""
    train_trajs = dataset_splits[std]["train_trajs"]  # (n_traj, n_steps, 1)
    states = train_trajs.reshape(-1, 1)

    states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        grads = model(0, states_tensor).cpu().numpy()

    return states, grads


def run_sindy_sr(states, grads):
    """
    Fit SINDy on NODE gradients for the 1-state autonomous system.
    Uses degree-3 polynomial library to match test_dynamics_toy.ipynb
    """
    import pysindy as ps

    feature_names = ["w"]

    library = ps.PolynomialLibrary(degree=3)
    optimizer = ps.STLSQ(threshold=1e-4)

    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
    )

    # Fit: x_dot is the NODE-predicted gradients
    model.fit(states, t=1.0, x_dot=grads, feature_names=feature_names)

    # Get equations
    equations = model.equations()

    # Compute fit MSE
    grads_pred = model.predict(states)
    fit_mse = float(np.mean((grads_pred - grads) ** 2))

    # Complexity
    complexities = [calculate_complexity(eq, method="operator_count") for eq in equations]

    # Get coefficients for reporting
    coeffs = np.asarray(model.coefficients(), dtype=float)
    nonzero_terms = int(np.count_nonzero(np.abs(coeffs) > 1e-12))

    return {
        "method": "SINDy",
        "model": model,
        "equations": equations,
        "complexities": complexities,
        "fit_mse": fit_mse,
        "nonzero_terms": nonzero_terms,
    }


def run_symantic_sr(states, grads):
    """
    Fit SyMANTIC on NODE gradients for the 1-state system using the SymanticModel class.
    """
    from symantic.model import SymanticModel
    from unittest.mock import patch

    feature_names = ["w"]
    
    # Build DataFrame: first column = target, rest = features
    df = pd.DataFrame(states.astype(np.float64), columns=feature_names)
    df.insert(0, "dw", grads[:, 0].astype(np.float64))

    operators = ["pow(2)", "/", "+1"]
    symantic = SymanticModel(
        df,
        operators=operators,
        metrics=[0.001, 0.99],
        disp=True,
    )

    with patch("builtins.input", return_value="no"):
        res, pareto = symantic.fit()

    if pareto is None or len(pareto) == 0:
        print(f"    WARNING: SyMANTIC returned empty Pareto set")
        return {
            "method": "SyMANTIC",
            "models": [None],
            "equations": ["0"],
            "complexities": [0.0],
            "pareto_fronts": [None],
        }

    eq = str(res['utopia']['expression'])
    comp = float(res['utopia']['complexity'])

    print(f"    dw/dt = {eq}")

    return {
        "method": "SyMANTIC",
        "models": [res],
        "equations": [eq],
        "complexities": [comp],
        "pareto_fronts": [pareto],
    }


def run_symbolic_regression(node_models, dataset_splits):
    """Run SINDy and SyMANTIC symbolic regression for each noise level."""
    print("\n" + "=" * 70)
    print("SECTION 5: Symbolic Regression")
    print("=" * 70)

    sr_results = {}

    for std in NOISE_LEVELS:
        print(f"\n--- Noise level: {std:.3f} ---")
        model = node_models[std]
        states, grads = get_node_gradient_data(model, dataset_splits, std)

        # SINDy
        print("  Running SINDy...")
        try:
            sindy_result = run_sindy_sr(states, grads)
            print(f"  SINDy equations:")
            for i, eq in enumerate(sindy_result["equations"]):
                print(f"    State {i}: {eq}")
        except Exception as e:
            print(f"  SINDy FAILED: {e}")
            sindy_result = None

        # SyMANTIC
        print("  Running SyMANTIC...")
        try:
            symantic_result = run_symantic_sr(states, grads)
            print(f"  SyMANTIC equations:")
            for i, eq in enumerate(symantic_result["equations"]):
                print(f"    State {i}: {eq}")
        except Exception as e:
            print(f"  SyMANTIC FAILED: {e}")
            import traceback
            traceback.print_exc()
            symantic_result = None

        sr_results[std] = {
            "sindy": sindy_result,
            "symantic": symantic_result,
        }

    return sr_results


# ===========================================================================
# Section 6: Evaluate Symbolic Models via Trajectory Integration
# ===========================================================================

def _make_rhs_from_sindy_model(sindy_model):
    """Create an ODE RHS function from a fitted PySINDy model."""
    def rhs(t, y):
        return sindy_model.predict(y.reshape(1, -1)).flatten()
    return rhs


def _make_rhs_from_symantic_equations(equations, feature_names=("w",)):
    """Create an ODE RHS function from SyMANTIC equation strings."""
    import re

    # Preprocess equations: convert ^ to ** for Python eval
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
    """Integrate an ODE RHS from initial condition y0 over t_points."""
    try:
        sol = solve_ivp(
            rhs_func,
            [t_points[0], t_points[-1]],
            y0,
            t_eval=t_points,
            method=method,
            max_step=0.5,
            rtol=1e-10,
            atol=1e-10
        )
        if sol.success:
            return sol.y.T  # (n_steps, state_dim)
        else:
            return np.full((len(t_points), len(y0)), np.nan)
    except Exception:
        return np.full((len(t_points), len(y0)), np.nan)


def evaluate_sr_models(sr_results, dataset_splits, t_eval, t_split_index):
    """Evaluate symbolic regression models on all test sets with R2."""
    print("\n" + "=" * 70)
    print("SECTION 6: SR Model Evaluation")
    print("=" * 70)

    sr_metrics = []

    for std in NOISE_LEVELS:
        splits = dataset_splits[std]

        for sr_name, sr_result in [("SINDy", sr_results[std].get("sindy")),
                                    ("SyMANTIC", sr_results[std].get("symantic"))]:
            if sr_result is None:
                print(f"  Noise={std:.3f}, {sr_name}: SKIPPED (no model)")
                sr_metrics.append({
                    "noise_level": std,
                    "method": sr_name,
                    "mse_train": float("nan"),
                    "rmse_train": float("nan"),
                    "r2_train": float("nan"),
                    "mse_ext_t": float("nan"),
                    "rmse_ext_t": float("nan"),
                    "r2_ext_t": float("nan"),
                    "mse_ext_x0": float("nan"),
                    "rmse_ext_x0": float("nan"),
                    "r2_ext_x0": float("nan"),
                })
                continue

            # Build RHS function
            if sr_name == "SINDy":
                rhs_func = _make_rhs_from_sindy_model(sr_result["model"])
            else:
                rhs_func = _make_rhs_from_symantic_equations(sr_result["equations"])

            # --- Evaluate on training ICs, full trajectory ---
            train_ics = splits["train_ics"]
            t_train_pts = t_eval[:t_split_index]
            t_full = t_eval

            # Train set: integrate from ICs over [0, T_SPLIT)
            train_preds = []
            for ic in train_ics:
                pred = integrate_from_ic(rhs_func, ic, t_train_pts)
                train_preds.append(pred)
            train_preds = np.array(train_preds)

            valid_train = ~np.any(np.isnan(train_preds), axis=(1, 2))
            if valid_train.sum() > 0:
                y_true = splits["clean_train"][valid_train]
                y_pred = train_preds[valid_train]
                mse_train = float(np.mean((y_pred - y_true) ** 2))
                ss_res_train = np.sum((y_pred - y_true) ** 2)
                ss_tot_train = np.sum((y_true - np.mean(y_true, axis=(0, 1))) ** 2)
                r2_train = float(1 - ss_res_train / ss_tot_train) if ss_tot_train > 0 else float("nan")
            else:
                mse_train = float("nan")
                r2_train = float("nan")

            # Ext-T: integrate full [0, 12], evaluate [T_SPLIT, 12]
            ext_t_preds = []
            for ic in train_ics:
                pred = integrate_from_ic(rhs_func, ic, t_full)
                ext_t_preds.append(pred)
            ext_t_preds = np.array(ext_t_preds)

            ext_t_tail = ext_t_preds[:, t_split_index:, :]
            valid_ext_t = ~np.any(np.isnan(ext_t_tail), axis=(1, 2))
            if valid_ext_t.sum() > 0:
                y_true_ext_t = splits["clean_ext_t"][valid_ext_t]
                y_pred_ext_t = ext_t_tail[valid_ext_t]
                mse_ext_t = float(np.mean((y_pred_ext_t - y_true_ext_t) ** 2))
                ss_res_ext_t = np.sum((y_pred_ext_t - y_true_ext_t) ** 2)
                ss_tot_ext_t = np.sum((y_true_ext_t - np.mean(y_true_ext_t, axis=(0, 1))) ** 2)
                r2_ext_t = float(1 - ss_res_ext_t / ss_tot_ext_t) if ss_tot_ext_t > 0 else float("nan")
            else:
                mse_ext_t = float("nan")
                r2_ext_t = float("nan")

            # Ext-X0: integrate from OOD ICs over [0, 12]
            ext_x0_preds = []
            for ic in splits["test_ext_x0_ics"]:
                pred = integrate_from_ic(rhs_func, ic, t_full)
                ext_x0_preds.append(pred)
            ext_x0_preds = np.array(ext_x0_preds)

            valid_ext_x0 = ~np.any(np.isnan(ext_x0_preds), axis=(1, 2))
            if valid_ext_x0.sum() > 0:
                y_true_ext_x0 = splits["clean_ext_x0"][valid_ext_x0]
                y_pred_ext_x0 = ext_x0_preds[valid_ext_x0]
                mse_ext_x0 = float(np.mean((y_pred_ext_x0 - y_true_ext_x0) ** 2))
                ss_res_ext_x0 = np.sum((y_pred_ext_x0 - y_true_ext_x0) ** 2)
                ss_tot_ext_x0 = np.sum((y_true_ext_x0 - np.mean(y_true_ext_x0, axis=(0, 1))) ** 2)
                r2_ext_x0 = float(1 - ss_res_ext_x0 / ss_tot_ext_x0) if ss_tot_ext_x0 > 0 else float("nan")
            else:
                mse_ext_x0 = float("nan")
                r2_ext_x0 = float("nan")

            rmse_train = float(np.sqrt(mse_train)) if not np.isnan(mse_train) else float("nan")
            rmse_ext_t = float(np.sqrt(mse_ext_t)) if not np.isnan(mse_ext_t) else float("nan")
            rmse_ext_x0 = float(np.sqrt(mse_ext_x0)) if not np.isnan(mse_ext_x0) else float("nan")

            sr_metrics.append({
                "noise_level": std,
                "method": sr_name,
                "mse_train": mse_train,
                "rmse_train": rmse_train,
                "r2_train": r2_train,
                "mse_ext_t": mse_ext_t,
                "rmse_ext_t": rmse_ext_t,
                "r2_ext_t": r2_ext_t,
                "mse_ext_x0": mse_ext_x0,
                "rmse_ext_x0": rmse_ext_x0,
                "r2_ext_x0": r2_ext_x0,
            })

            print(f"  Noise={std:.3f}, {sr_name:>8s} | Train RMSE={rmse_train:.6f} (R2={r2_train:.4f}) | "
                  f"Ext-T RMSE={rmse_ext_t:.6f} (R2={r2_ext_t:.4f}) | "
                  f"Ext-X0 RMSE={rmse_ext_x0:.6f} (R2={r2_ext_x0:.4f})")

    return sr_metrics


# ===========================================================================
# Section 7: Results Summary
# ===========================================================================

def compile_results(node_metrics, sr_metrics, sr_results):
    """Compile all results into summary tables and save."""
    print("\n" + "=" * 70)
    print("SECTION 7: Results Summary")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Performance comparison table ---
    all_metrics = node_metrics + sr_metrics
    df_perf = pd.DataFrame(all_metrics)
    df_perf = df_perf.sort_values(["noise_level", "method"]).reset_index(drop=True)

    perf_path = RESULTS_DIR / "performance_table.csv"
    df_perf.to_csv(perf_path, index=False)
    print(f"\nPerformance table saved to {perf_path}")
    print("\n" + df_perf.to_string(index=False))

    # --- Equations table ---
    eq_rows = []
    for std in NOISE_LEVELS:
        for sr_name in ["sindy", "symantic"]:
            sr_result = sr_results.get(std, {}).get(sr_name)
            if sr_result is None:
                continue
            equations = sr_result["equations"]
            complexities = sr_result["complexities"]
            for i, (eq, comp) in enumerate(zip(equations, complexities)):
                state_names = ["dw/dt"]
                eq_rows.append({
                    "noise_level": std,
                    "method": sr_result["method"],
                    "state": state_names[i] if i < len(state_names) else f"state_{i}",
                    "equation": eq,
                    "complexity": comp,
                })

    df_eqs = pd.DataFrame(eq_rows)
    eq_path = RESULTS_DIR / "equations_table.csv"
    df_eqs.to_csv(eq_path, index=False)
    print(f"\nEquations table saved to {eq_path}")
    print("\n" + df_eqs.to_string(index=False))

    # --- Ground truth reference ---
    print("\n--- Ground Truth Equations ---")
    for state, eq in GROUND_TRUTH_EQUATIONS.items():
        print(f"  {state} = {eq}")

    # --- Save equations as JSON ---
    eq_json = {
        "ground_truth": GROUND_TRUTH_EQUATIONS,
        "discovered": {},
    }
    for std in NOISE_LEVELS:
        eq_json["discovered"][str(std)] = {}
        for sr_name in ["sindy", "symantic"]:
            sr_result = sr_results.get(std, {}).get(sr_name)
            if sr_result:
                eq_json["discovered"][str(std)][sr_name] = sr_result["equations"]

    json_path = RESULTS_DIR / "equations.json"
    json_path.write_text(json.dumps(eq_json, indent=2))
    print(f"\nEquations JSON saved to {json_path}")

    return df_perf, df_eqs


def plot_16_trajectories_ext_t(model, sr_res, splits, t_eval, noise_std):
    import matplotlib.pyplot as plt
    
    train_ics = splits["train_ics"]
    n_ics = len(train_ics)
    if n_ics < 16:
        indices = np.arange(n_ics)
    else:
        # Sample 16 evenly-spaced ICs
        sorted_indices = np.argsort(train_ics[:, 0])
        indices = [sorted_indices[int(i * (n_ics - 1) / 15)] for i in range(16)]
        
    ics_show = train_ics[indices]
    clean_show = splits["clean_train"][indices]  # (16, N_STEPS_TRAIN, 1)
    clean_ext_show = splits["clean_ext_t"][indices]  # (16, N_STEPS_EXT, 1)
    clean_full = np.concatenate([clean_show, clean_ext_show], axis=1)  # (16, N_STEPS, 1)
    noisy_show = splits["train_trajs"][indices]  # (16, N_STEPS_TRAIN, 1)
    
    pred_node = get_node_predictions(model, ics_show, t_eval)  # (16, N_STEPS, 1)
    
    sindy_res = sr_res.get("sindy")
    if sindy_res is not None:
        rhs_sindy = _make_rhs_from_sindy_model(sindy_res["model"])
        pred_sindy = np.array([integrate_from_ic(rhs_sindy, ic, t_eval) for ic in ics_show])
    else:
        pred_sindy = np.full((len(ics_show), len(t_eval), 1), np.nan)
        
    symantic_res = sr_res.get("symantic")
    if symantic_res is not None and symantic_res.get("equations") is not None:
        rhs_symantic = _make_rhs_from_symantic_equations(symantic_res["equations"])
        pred_symantic = np.array([integrate_from_ic(rhs_symantic, ic, t_eval) for ic in ics_show])
    else:
        pred_symantic = np.full((len(ics_show), len(t_eval), 1), np.nan)
        
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), sharex=True, sharey=True)
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


def plot_16_trajectories_ext_x0(model, sr_res, splits, t_eval, noise_std):
    import matplotlib.pyplot as plt
    
    test_ics = splits["test_ext_x0_ics"]
    n_ics = len(test_ics)
    if n_ics < 16:
        indices = np.arange(n_ics)
    else:
        # Sample 16 evenly-spaced ICs
        sorted_indices = np.argsort(test_ics[:, 0])
        indices = [sorted_indices[int(i * (n_ics - 1) / 15)] for i in range(16)]
        
    ics_show = test_ics[indices]
    clean_show = splits["clean_ext_x0"][indices]  # (16, N_STEPS, 1)
    
    pred_node = get_node_predictions(model, ics_show, t_eval)  # (16, N_STEPS, 1)
    
    sindy_res = sr_res.get("sindy")
    if sindy_res is not None:
        rhs_sindy = _make_rhs_from_sindy_model(sindy_res["model"])
        pred_sindy = np.array([integrate_from_ic(rhs_sindy, ic, t_eval) for ic in ics_show])
    else:
        pred_sindy = np.full((len(ics_show), len(t_eval), 1), np.nan)
        
    symantic_res = sr_res.get("symantic")
    if symantic_res is not None and symantic_res.get("equations") is not None:
        rhs_symantic = _make_rhs_from_symantic_equations(symantic_res["equations"])
        pred_symantic = np.array([integrate_from_ic(rhs_symantic, ic, t_eval) for ic in ics_show])
    else:
        pred_symantic = np.full((len(ics_show), len(t_eval), 1), np.nan)
        
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), sharex=True, sharey=True)
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


def plot_metrics_distribution(model, sr_res, splits, t_eval, t_split_index, noise_std):
    import matplotlib.pyplot as plt
    
    t_full = t_eval
    
    # Pre-calculate RHS functions
    sindy_res = sr_res.get("sindy")
    rhs_sindy = _make_rhs_from_sindy_model(sindy_res["model"]) if sindy_res is not None else None
    
    symantic_res = sr_res.get("symantic")
    rhs_symantic = _make_rhs_from_symantic_equations(symantic_res["equations"]) if (symantic_res is not None and symantic_res.get("equations") is not None) else None
    
    # ----------------------------------------------------
    # 1. Ext-T metrics per trajectory
    # ----------------------------------------------------
    train_ics = splits["train_ics"]
    clean_ext_t = splits["clean_ext_t"]  # (n_traj, n_steps_ext, 1)
    
    pred_node_full = get_node_predictions(model, train_ics, t_eval)
    pred_node_ext_t = pred_node_full[:, t_split_index:, :]
    
    mse_node_ext_t = []
    r2_node_ext_t = []
    
    mse_sindy_ext_t = []
    r2_sindy_ext_t = []
    
    mse_symantic_ext_t = []
    r2_symantic_ext_t = []
    
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
            
    # ----------------------------------------------------
    # 2. Ext-X0 metrics per trajectory
    # ----------------------------------------------------
    test_ics = splits["test_ext_x0_ics"]
    clean_ext_x0 = splits["clean_ext_x0"]  # (n_traj, N_STEPS, 1)
    
    pred_node_ext_x0 = get_node_predictions(model, test_ics, t_eval)
    
    mse_node_ext_x0 = []
    r2_node_ext_x0 = []
    
    mse_sindy_ext_x0 = []
    r2_sindy_ext_x0 = []
    
    mse_symantic_ext_x0 = []
    r2_symantic_ext_x0 = []
    
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


def main():
    start_time = time.time()

    print("\n" + "=" * 70)
    print("NOISE-LEVEL IMPACT STUDY ON NODE DISTILLATION")
    print("System: Spruce-budworm (1-state)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # 1. Generate data
    ics, t_eval, dataset_clean, dataset_noisy = generate_data()

    # 2. Partition into train/test
    dataset_splits, train_indices, test_x0_indices, t_split_index, t_train, t_ext = \
        partition_data(ics, t_eval, dataset_clean, dataset_noisy)

    # 3. Train/load NODE models
    node_models, training_logs = train_or_load_nodes(dataset_splits, t_train)

    # 4. Evaluate NODEs
    node_metrics = evaluate_nodes(node_models, dataset_splits, t_eval, t_split_index)
    plot_gradient_parity(node_models, dataset_splits, t_eval, t_split_index)

    # 5. Symbolic regression
    sr_results = run_symbolic_regression(node_models, dataset_splits)

    # 6. Evaluate SR models
    sr_metrics = evaluate_sr_models(sr_results, dataset_splits, t_eval, t_split_index)
    plot_all_results(node_models, sr_results, dataset_splits, t_eval, t_split_index)

    # Generate new requested figures per noise level
    for std in NOISE_LEVELS:
        print(f"\nGenerating detailed plots for noise level {std:.3f}...")
        splits = dataset_splits[std]
        sr_res = sr_results[std]
        plot_16_trajectories_ext_t(node_models[std], sr_res, splits, t_eval, std)
        plot_16_trajectories_ext_x0(node_models[std], sr_res, splits, t_eval, std)
        plot_metrics_distribution(node_models[std], sr_res, splits, t_eval, t_split_index, std)

    # 7. Compile results
    df_perf, df_eqs = compile_results(node_metrics, sr_metrics, sr_results)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Experiment completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
