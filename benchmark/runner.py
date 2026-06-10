from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig, RunRecord
from .data import inverse_scale, load_batch_data, train_val_split
from .gradients import central_difference_gradients, forward_euler_gradients
from .metrics import compute_state_metrics
from .node_gradients import train_node_and_estimate_gradients
from .plotting import plot_metric_bars, plot_pareto, plot_trajectories
from .simulation import integrate_with_piecewise_inputs
from .symbolic_sindy import fit_sindy_models, make_rhs_from_sindy
from .symbolic_symantic import fit_symantic_models, make_rhs_from_symantic



def _estimate_gradients(method: str, time, states, inputs, cfg: ExperimentConfig):
    if method == "forward_euler":
        return forward_euler_gradients(time, states)
    if method == "central_difference":
        return central_difference_gradients(time, states)
    if method == "sindy_default":
        import pysindy as ps

        try:
            diff = ps.SmoothedFiniteDifference(
                smoother_kws={
                    "window_length": cfg.differentiation_savgol_window,
                    "polyorder": cfg.differentiation_savgol_order,
                }
            )
            return diff._differentiate(states, time)  # pylint: disable=protected-access
        except Exception:
            try:
                diff = ps.FiniteDifference(order=2)
                return diff._differentiate(states, time)  # pylint: disable=protected-access
            except Exception:
                # Final fallback if pysindy internals differ in this environment.
                return np.gradient(states, time, axis=0)
    if method == "node":
        result = train_node_and_estimate_gradients(
            time_np=time,
            states_np=states,
            inputs_np=inputs,
            hidden_dim=cfg.node_hidden_dim,
            lr=cfg.node_lr,
            weight_decay=cfg.node_weight_decay,
            epochs=cfg.node_epochs,
            method=cfg.node_method,
        )
        return result["gradients"]

    raise ValueError(f"Unknown derivative method: {method}")



def _fit_symbolic(symbolic_method: str, time, states, inputs, gradients, cfg: ExperimentConfig):
    if symbolic_method == "sindy":
        return fit_sindy_models(
            time=time,
            states=states,
            inputs=inputs,
            gradients=gradients,
            poly_degree=cfg.sindy_poly_degree,
            threshold_grid=cfg.sindy_threshold_grid,
            alpha=cfg.sindy_alpha,
            complexity_weight=cfg.sindy_complexity_weight,
            include_interaction=cfg.sindy_include_interaction,
            include_bias=cfg.sindy_include_bias,
        )
    if symbolic_method == "symantic":
        return fit_symantic_models(time=time, states=states, inputs=inputs, gradients=gradients, return_point=cfg.symantic_return_point)
    raise ValueError(f"Unknown symbolic method: {symbolic_method}")



def _make_rhs(symbolic_method: str, fit_result):
    if symbolic_method == "sindy":
        return make_rhs_from_sindy(fit_result)
    if symbolic_method == "symantic":
        return make_rhs_from_symantic(fit_result)
    raise ValueError(f"Unknown symbolic method: {symbolic_method}")



def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    np.random.seed(cfg.random_seed)
    cfg.ensure_output_dirs()

    bundle = load_batch_data(cfg)
    split = train_val_split(bundle.time, bundle.states, bundle.inputs, cfg.train_fraction)

    train_t = split["train"]["time"]
    train_x = split["train"]["states"]
    train_u = split["train"]["inputs"]

    val_t = split["val"]["time"]
    val_x = split["val"]["states"]
    val_u = split["val"]["inputs"]

    # For NODE-based gradients, use full dataset
    full_t = bundle.time
    full_x = bundle.states
    full_u = bundle.inputs

    results: list[RunRecord] = []
    node_models_dir = cfg.output_dir / "models"
    node_models_dir.mkdir(parents=True, exist_ok=True)

    for d_method in cfg.run_derivative_methods:
        try:
            if d_method == "node":
                # Train NODE on full dataset
                print(f"\nTraining NODE on full {cfg.batch_id} dataset ({len(full_t)} points)...")
                node_result = train_node_and_estimate_gradients(
                    time_np=full_t,
                    states_np=full_x,
                    inputs_np=full_u,
                    hidden_dim=cfg.node_hidden_dim,
                    lr=cfg.node_lr,
                    weight_decay=cfg.node_weight_decay,
                    epochs=cfg.node_epochs,
                    method=cfg.node_method,
                    verbose=False,
                )
                grads = node_result["gradients"]

                # Save NODE model
                model_path = node_models_dir / f"node_model_{cfg.batch_id}.pth"
                torch.save(node_result["model"].state_dict(), model_path)
                print(f"NODE model saved to {model_path}")

                # Plot NODE fitted states vs actual (on full data)
                node_pred_full = node_result["pred_states"]
                plot_trajectories(
                    time=inverse_scale(full_t, cfg.time_col, bundle.scaler),
                    y_true=np.column_stack(
                        [
                            inverse_scale(full_x[:, 0], cfg.state_cols[0], bundle.scaler),
                            inverse_scale(full_x[:, 1], cfg.state_cols[1], bundle.scaler),
                        ]
                    ),
                    y_pred=np.column_stack(
                        [
                            inverse_scale(node_pred_full[:, 0], cfg.state_cols[0], bundle.scaler),
                            inverse_scale(node_pred_full[:, 1], cfg.state_cols[1], bundle.scaler),
                        ]
                    ),
                    label="NODE (fitted)",
                    state_names=cfg.state_cols,
                    out_file=cfg.output_dir / "figures" / f"node_fitted_{cfg.batch_id}.png",
                )
                print(f"NODE fit plot saved")

                # Use train data for symbolic regression
                _t = train_t
                _x = train_x
                _u = train_u
                eval_t = val_t
                eval_x = val_x
                eval_u = val_u
            else:
                # Other gradient methods use train/val split
                grads = _estimate_gradients(d_method, train_t, train_x, train_u, cfg)
                _t = train_t
                _x = train_x
                _u = train_u
                eval_t = val_t
                eval_x = val_x
                eval_u = val_u

        except Exception as exc:
            print(f"Skipping derivative method {d_method}: {exc}")
            continue

        for s_method in cfg.run_symbolic_methods:
            try:
                fit_result = _fit_symbolic(s_method, _t, _x, _u, grads, cfg)
                rhs = _make_rhs(s_method, fit_result)

                pred_val = integrate_with_piecewise_inputs(
                    rhs=rhs,
                    time=eval_t,
                    init_state=eval_x[0],
                    inputs=eval_u,
                )

                metrics = compute_state_metrics(eval_x, pred_val)

                eq_path = cfg.output_dir / "equations" / f"equations_{d_method}_{s_method}.json"
                eq_payload = {
                    "equation_state_1": fit_result["equations"][0],
                    "equation_state_2": fit_result["equations"][1],
                    "complexities": fit_result["complexities"],
                }
                eq_path.write_text(json.dumps(eq_payload, indent=2))

                metadata = {}
                if d_method == "node":
                    metadata["node_final_loss"] = str(node_result["final_loss"])
                    metadata["node_model_path"] = str(model_path)

                results.append(
                    RunRecord(
                        derivative_method=d_method,
                        symbolic_method=s_method,
                        state_1_equation=fit_result["equations"][0],
                        state_2_equation=fit_result["equations"][1],
                        complexity_1=int(fit_result["complexities"][0]),
                        complexity_2=int(fit_result["complexities"][1]),
                        rmse_state_1=metrics["rmse_state_1"],
                        rmse_state_2=metrics["rmse_state_2"],
                        rmse_mean=metrics["rmse_mean"],
                        r2_state_1=metrics["r2_state_1"],
                        r2_state_2=metrics["r2_state_2"],
                        r2_mean=metrics["r2_mean"],
                        metadata=metadata,
                    )
                )

                print(f"  {d_method} + {s_method}: RMSE={metrics['rmse_mean']:.6f}, R2={metrics['r2_mean']:.4f}")

                plot_trajectories(
                    time=inverse_scale(eval_t, cfg.time_col, bundle.scaler),
                    y_true=np.column_stack(
                        [
                            inverse_scale(eval_x[:, 0], cfg.state_cols[0], bundle.scaler),
                            inverse_scale(eval_x[:, 1], cfg.state_cols[1], bundle.scaler),
                        ]
                    ),
                    y_pred=np.column_stack(
                        [
                            inverse_scale(pred_val[:, 0], cfg.state_cols[0], bundle.scaler),
                            inverse_scale(pred_val[:, 1], cfg.state_cols[1], bundle.scaler),
                        ]
                    ),
                    label=f"{d_method}+{s_method}",
                    state_names=cfg.state_cols,
                    out_file=cfg.output_dir / "figures" / f"traj_{d_method}_{s_method}.png",
                )

            except Exception as exc:
                print(f"Skipping combo {d_method}+{s_method}: {exc}")
                continue

    if not results:
        raise RuntimeError("No experiment combinations completed successfully.")

    df_results = pd.DataFrame([r.__dict__ for r in results])
    df_results = df_results.sort_values(by=["rmse_mean", "r2_mean"], ascending=[True, False])
    df_results.to_csv(cfg.output_dir / "tables" / "benchmark_results.csv", index=False)

    plot_pareto(df_results, cfg.output_dir / "figures" / "pareto_rmse_complexity.png")
    plot_metric_bars(df_results, cfg.output_dir / "figures" / "metrics_bar_comparison.png")

    print("\n" + "=" * 60)
    print("Benchmark complete. Results:")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("\nOutputs saved to:", cfg.output_dir)

    return df_results
