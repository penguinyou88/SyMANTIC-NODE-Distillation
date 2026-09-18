#!/usr/bin/env python3
"""
Rerun just the SyMANTIC symbolic regression fitting part using the saved NODE checkpoints
and save the full Pareto front results to CSV files for each noise level.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

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
    get_node_gradient_data,
    run_symantic_sr,
    DEVICE,
    NOISE_LEVELS,
    NODE_HIDDEN_DIM,
    TRAIN_W0,
    RESULTS_DIR,
    MODELS_DIR,
)

def main():
    print("=" * 70)
    print("RERUNNING SyMANTIC FITTING ONLY & SAVING FULL PARETO FRONTS")
    print("=" * 70)

    # 1. Re-generate dataset partition to get gradient training data
    ics, t_eval, dataset_clean, dataset_noisy = generate_data()
    dataset_splits, train_indices, test_x0_indices, t_split_index, t_train, t_ext = \
        partition_data(ics, t_eval, dataset_clean, dataset_noisy)

    # 2. For each noise level, load the NODE model, run SyMANTIC and save pareto
    for std in NOISE_LEVELS:
        print(f"\n--- Noise level: {std:.3f} ---")
        model_path = MODELS_DIR / f"node_noise_std_{std}.pt"
        if not model_path.exists():
            print(f"Warning: Checkpoint not found at {model_path}. Skipping...")
            continue
        
        # Load NODE model
        model = ODEFunc(state_dim=1, hidden_dim=NODE_HIDDEN_DIM, scale=TRAIN_W0[1]).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        print(f"Loaded NODE model from {model_path}")

        # Get NODE gradients
        states, grads = get_node_gradient_data(model, dataset_splits, std)

        # Run SyMANTIC
        print("Running SyMANTIC fitting...")
        try:
            symantic_result = run_symantic_sr(states, grads)
            pareto = symantic_result["pareto_fronts"][0]
            
            if pareto is not None and isinstance(pareto, pd.DataFrame):
                pareto_path = RESULTS_DIR / f"pareto_front_noise_{std}.csv"
                pareto.to_csv(pareto_path, index=False)
                print(f"Successfully saved full Pareto front to {pareto_path}")
                print("\nPareto Front Preview:")
                print(pareto.to_string())
            else:
                print("Warning: SyMANTIC returned an empty or invalid Pareto front.")
        except Exception as e:
            print(f"SyMANTIC FAILED for noise {std:.3f}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All SyMANTIC runs completed.")
    print("=" * 70)

if __name__ == "__main__":
    main()
