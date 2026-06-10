from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig


ScaleParams = tuple[str, float, float]


@dataclass
class DataBundle:
    df_raw: pd.DataFrame
    df_processed: pd.DataFrame
    time: np.ndarray
    states: np.ndarray
    inputs: np.ndarray
    init_state: np.ndarray
    scaler: dict[str, ScaleParams]
    state_names: list[str]
    input_names: list[str]



def _zscore(series: pd.Series) -> tuple[pd.Series, ScaleParams]:
    mu = float(series.mean())
    sigma = float(series.std())
    sigma = sigma if sigma > 0 else 1.0
    return (series - mu) / sigma, ("zscore", mu, sigma)


def _minmax(series: pd.Series) -> tuple[pd.Series, ScaleParams]:
    min_value = float(series.min())
    max_value = float(series.max())
    span = max_value - min_value
    span = span if span > 0 else 1.0
    return (series - min_value) / span, ("minmax", min_value, span)



def load_batch_data(config: ExperimentConfig) -> DataBundle:
    df = pd.read_csv(config.data_path)

    # Restrict to high-quality rows when available.
    if "Data Quality" in df.columns:
        df = df[df["Data Quality"].eq("Good")].copy()

    df_batch = df[df["R1 Charge"].eq(config.batch_id)].copy()
    if df_batch.empty:
        raise ValueError(f"No rows found for batch_id={config.batch_id}")

    required_cols = [config.time_col, *config.state_cols, *config.input_cols]
    missing = [c for c in required_cols if c not in df_batch.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_batch = df_batch.sort_values(by=config.time_col)
    df_batch = df_batch.drop_duplicates(subset=config.time_col)
    df_batch = df_batch[required_cols]
    df_batch = df_batch.fillna(method="ffill").fillna(method="bfill")
    df_batch = df_batch.reset_index(drop=True)

    scaler: dict[str, ScaleParams] = {}
    df_proc = df_batch.copy()
    if config.normalization == "zscore":
        df_proc[config.time_col], scaler[config.time_col] = _minmax(df_proc[config.time_col])
        for col in [*config.state_cols, *config.input_cols]:
            df_proc[col], scaler[col] = _zscore(df_proc[col])
    else:
        for col in required_cols:
            scaler[col] = ("none", 0.0, 1.0)

    time = df_proc[config.time_col].to_numpy(dtype=np.float64)
    if not np.all(np.diff(time) >= 0):
        raise ValueError("Time column is not monotonic after sorting.")

    states = df_proc[list(config.state_cols)].to_numpy(dtype=np.float64)
    inputs = df_proc[list(config.input_cols)].to_numpy(dtype=np.float64)

    return DataBundle(
        df_raw=df_batch,
        df_processed=df_proc,
        time=time,
        states=states,
        inputs=inputs,
        init_state=states[0].copy(),
        scaler=scaler,
        state_names = config.state_cols,
        input_names = config.input_cols,
    )



def train_val_split(time: np.ndarray, states: np.ndarray, inputs: np.ndarray, train_fraction: float) -> dict[str, Any]:
    n = len(time)
    if n < 8:
        raise ValueError("Need at least 8 points for stable split and integration checks.")

    split_idx = max(3, min(n - 3, int(np.floor(train_fraction * n))))
    return {
        "train": {
            "time": time[:split_idx],
            "states": states[:split_idx],
            "inputs": inputs[:split_idx],
        },
        "val": {
            "time": time[split_idx - 1 :],
            "states": states[split_idx - 1 :],
            "inputs": inputs[split_idx - 1 :],
        },
        "split_idx": split_idx,
    }



def inverse_scale(values: np.ndarray, col: str, scaler: dict[str, ScaleParams]) -> np.ndarray:
    scale_kind, offset, scale = scaler[col]
    if scale_kind in {"zscore", "minmax", "none"}:
        return values * scale + offset
    raise ValueError(f"Unknown scaling kind for column {col}: {scale_kind}")
