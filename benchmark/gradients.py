from __future__ import annotations

import numpy as np



def forward_euler_gradients(time: np.ndarray, states: np.ndarray) -> np.ndarray:
    dt = np.diff(time)
    dx = np.diff(states, axis=0)
    g = dx / dt[:, None]
    g_last = g[-1:]
    return np.vstack([g, g_last])



def central_difference_gradients(time: np.ndarray, states: np.ndarray) -> np.ndarray:
    n = len(time)
    grad = np.zeros_like(states)

    grad[0] = (states[1] - states[0]) / (time[1] - time[0])
    grad[-1] = (states[-1] - states[-2]) / (time[-1] - time[-2])

    dt_prev = (time[1:-1] - time[:-2])[:, None]
    dt_next = (time[2:] - time[1:-1])[:, None]
    slope_prev = (states[1:-1] - states[:-2]) / dt_prev
    slope_next = (states[2:] - states[1:-1]) / dt_next

    grad[1:-1] = (dt_next * slope_prev + dt_prev * slope_next) / (dt_prev + dt_next)
    return grad
