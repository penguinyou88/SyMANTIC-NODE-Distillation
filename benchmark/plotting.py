from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 240,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
        }
    )



def plot_trajectories(time, y_true, y_pred, label, state_names, out_file: Path, xlabel: str = "CumulativeProp / Mlb") -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(time, y_true[:, i], color="#1f4e79", lw=2.2, label="Measured")
        ax.plot(time, y_pred[:, i], color="#d95f02", lw=1.9, ls="--", label=label)
        ax.set_ylabel(state_names[i])
        ax.legend(loc="best")

    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)



def plot_pareto(df_results: pd.DataFrame, out_file: Path) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in df_results.iterrows():
        complexity = row["complexity_1"] + row["complexity_2"]
        name = f"{row['derivative_method']} + {row['symbolic_method']}"
        ax.scatter(complexity, row["rmse_mean"], s=70, label=name)

    ax.set_xlabel("Equation complexity (state1 + state2)")
    ax.set_ylabel("Mean trajectory RMSE")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)



def plot_metric_bars(df_results: pd.DataFrame, out_file: Path) -> None:
    set_publication_style()
    labels = [f"{d}\n{s}" for d, s in zip(df_results["derivative_method"], df_results["symbolic_method"])]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, df_results["rmse_mean"], width, label="RMSE")
    ax.bar(x + width / 2, df_results["r2_mean"], width, label="R2")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Metric value")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
