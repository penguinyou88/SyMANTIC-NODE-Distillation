from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score



def compute_state_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse_1 = float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])))
    rmse_2 = float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])))
    r2_1 = float(r2_score(y_true[:, 0], y_pred[:, 0]))
    r2_2 = float(r2_score(y_true[:, 1], y_pred[:, 1]))

    return {
        "rmse_state_1": rmse_1,
        "rmse_state_2": rmse_2,
        "rmse_mean": float(np.mean([rmse_1, rmse_2])),
        "r2_state_1": r2_1,
        "r2_state_2": r2_2,
        "r2_mean": float(np.mean([r2_1, r2_2])),
    }
