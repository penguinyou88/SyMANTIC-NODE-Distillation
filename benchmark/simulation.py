from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d



def integrate_with_piecewise_inputs(
    rhs,
    time: np.ndarray,
    init_state: np.ndarray,
    inputs: np.ndarray,
    method: str = "RK45",
    strategy: str = "auto",
    rtol: float = 1e-6,
    atol: float = 1e-8,
):
    time = np.asarray(time, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    init_state = np.asarray(init_state, dtype=float)

    if time.ndim != 1:
        raise ValueError("`time` must be a 1D array.")
    if len(time) < 2:
        raise ValueError("`time` must contain at least two points.")
    if inputs.shape[0] != len(time):
        raise ValueError("`inputs` must have the same number of rows as `time`.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("`time` must be strictly increasing.")

    if strategy not in {"auto", "interval", "global"}:
        raise ValueError("`strategy` must be one of {'auto', 'interval', 'global'}.")

    if strategy == "auto":
        # Fast-first strategy: try a single global solve, then fall back to robust interval mode.
        try:
            return integrate_with_piecewise_inputs(
                rhs=rhs,
                time=time,
                init_state=init_state,
                inputs=inputs,
                method=method,
                strategy="global",
                rtol=rtol,
                atol=atol,
            )
        except Exception:
            return integrate_with_piecewise_inputs(
                rhs=rhs,
                time=time,
                init_state=init_state,
                inputs=inputs,
                method=method,
                strategy="interval",
                rtol=rtol,
                atol=atol,
            )

    if strategy == "global":
        # Fast path: single solve over full horizon using piecewise-constant interpolation.
        # This is typically much faster, but can fail on stiff/problematic trajectories.
        u_interp = interp1d(
            time,
            inputs,
            axis=0,
            kind="previous",
            bounds_error=False,
            fill_value=(inputs[0], inputs[-1]),
            assume_sorted=True,
        )

        def _f_global(t, x):
            u = np.asarray(u_interp(t), dtype=float)
            dx = np.asarray(rhs(t, x, u), dtype=float)
            if not np.all(np.isfinite(dx)):
                raise FloatingPointError(f"Non-finite RHS detected at t={t:.6g}.")
            return dx

        sol = solve_ivp(
            fun=_f_global,
            t_span=(float(time[0]), float(time[-1])),
            y0=init_state.astype(float),
            t_eval=time,
            method=method,
            vectorized=False,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(
                "ODE integration failed in global strategy: "
                f"{sol.message}. Try strategy='interval' or method='Radau'/'BDF'."
            )

        return sol.y.T

    pred = np.empty((len(time), init_state.shape[0]), dtype=float)
    pred[0] = init_state

    def _rk4_interval_step(rhs_func, t0: float, t1: float, y0: np.ndarray, u: np.ndarray, n_steps: int):
        # Fixed-step RK4 fallback for intervals where adaptive solvers underflow.
        y = y0.astype(float).copy()
        h = (t1 - t0) / float(n_steps)
        t = t0
        for _ in range(n_steps):
            k1 = np.asarray(rhs_func(t, y, u), dtype=float)
            k2 = np.asarray(rhs_func(t + 0.5 * h, y + 0.5 * h * k1, u), dtype=float)
            k3 = np.asarray(rhs_func(t + 0.5 * h, y + 0.5 * h * k2, u), dtype=float)
            k4 = np.asarray(rhs_func(t + h, y + h * k3, u), dtype=float)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                raise FloatingPointError("Non-finite state encountered during RK4 fallback.")
        return y

    # Try requested method first, then robust fallbacks for stiff/problematic intervals.
    fallback_order = [method, "Radau", "BDF", "LSODA", "RK45"]
    methods = []
    for m in fallback_order:
        if m not in methods:
            methods.append(m)

    for i in range(len(time) - 1):
        t0 = float(time[i])
        t1 = float(time[i + 1])
        u_interval = inputs[i]

        def _f(t, x):
            dx = np.asarray(rhs(t, x, u_interval), dtype=float)
            if not np.all(np.isfinite(dx)):
                raise FloatingPointError(
                    f"Non-finite RHS detected at t={t:.6g}, interval={i}."
                )
            return dx

        y0 = pred[i]
        interval_success = False
        last_error = ""

        # Try progressively relaxed tolerances when adaptive methods struggle.
        tol_schedule = [
            (rtol, atol),
            (max(rtol, 1e-5), max(atol, 1e-7)),
            (max(rtol, 1e-4), max(atol, 1e-6)),
        ]

        for solver_method in methods:
            for rtol_try, atol_try in tol_schedule:
                try:
                    sol = solve_ivp(
                        fun=_f,
                        t_span=(t0, t1),
                        y0=y0,
                        t_eval=[t1],
                        method=solver_method,
                        vectorized=False,
                        rtol=rtol_try,
                        atol=atol_try,
                    )
                    if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                        pred[i + 1] = sol.y[:, -1]
                        interval_success = True
                        break
                    last_error = sol.message
                except Exception as exc:
                    last_error = str(exc)
            if interval_success:
                break

        if not interval_success:
            # Last-resort fixed-step fallback on this interval.
            for n_steps in (8, 16, 32, 64):
                try:
                    y1 = _rk4_interval_step(rhs, t0, t1, y0, u_interval, n_steps=n_steps)
                    if np.all(np.isfinite(y1)):
                        pred[i + 1] = y1
                        interval_success = True
                        break
                except Exception as exc:
                    last_error = str(exc)
            if interval_success:
                continue

        if not interval_success:
            raise RuntimeError(
                "ODE integration failed on interval "
                f"{i} [{t0:.6g}, {t1:.6g}] after trying {methods}: {last_error}"
            )

    return pred
