import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def dynamic_system(t, state):
    """
    ODE system from the image:

        du/dt = 1 / (1 + v^2) - u
        dv/dt = 1 / (1 + (1 + u / (1 + y)^2)^2) - v
        dy/dt = 0.1 * y

    Parameters
    ----------
    t : float
        Time (included for solver compatibility).
    state : array-like of length 3
        Current state [u, v, y].

    Returns
    -------
    np.ndarray
        Derivatives [du_dt, dv_dt, dy_dt].
    """
    u, v, y = state

    du_dt = 1.0 / (1.0 + v**2) - u
    dv_dt = 1.0 / (1.0 + (1.0 + u / (1.0 + y)**2)**2) - v
    dy_dt = 0.1 * y

    return np.array([du_dt, dv_dt, dy_dt], dtype=float)


def rk4_step(f, t, x, dt):
    """
    One fixed-step classical RK4 step.

    Parameters
    ----------
    f : callable
        RHS function f(t, x).
    t : float
        Current time.
    x : np.ndarray
        Current state.
    dt : float
        Step size.

    Returns
    -------
    np.ndarray
        Next state after one RK4 step.
    """
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_rk4(f, x0, t_span, n_steps):
    """
    Fixed-step RK4 integrator over a uniform grid.

    Parameters
    ----------
    f : callable
        RHS function f(t, x).
    x0 : array-like
        Initial condition.
    t_span : tuple(float, float)
        (t0, tf)
    n_steps : int
        Number of time points including endpoints.

    Returns
    -------
    t_eval : np.ndarray, shape (n_steps,)
        Time grid.
    states : np.ndarray, shape (n_steps, n_states)
        Simulated states.
    """
    t0, tf = t_span
    t_eval = np.linspace(t0, tf, n_steps)
    dt = t_eval[1] - t_eval[0]

    x0 = np.asarray(x0, dtype=float)
    states = np.zeros((n_steps, len(x0)), dtype=float)
    states[0] = x0

    x = x0.copy()
    for i in range(1, n_steps):
        x = rk4_step(f, t_eval[i - 1], x, dt)
        states[i] = x

    return t_eval, states


def integrate_solve_ivp(f, x0, t_span, n_steps, method="RK45"):
    """
    Adaptive solver wrapper using scipy.solve_ivp, returning values on a uniform t_eval grid.
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    sol = solve_ivp(
        fun=f,
        t_span=t_span,
        y0=np.asarray(x0, dtype=float),
        t_eval=t_eval,
        method=method,
        vectorized=False
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.t, sol.y.T


def simulate_system(
    initial_condition,
    t_span,
    n_steps=200,
    noise_std=0.0,
    random_seed=None,
    integrator="rk4",
    ivp_method="RK45",
    noise_on=("u", "v", "y"),
    return_dataframe=True
):
    """
    Simulate the dynamic system and optionally add Gaussian noise.

    Parameters
    ----------
    initial_condition : list or array-like of length 3
        Initial values [u0, v0, y0].
    t_span : tuple(float, float)
        Time interval (t_start, t_end).
    n_steps : int, optional
        Number of time points in the output.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise.
    random_seed : int or None, optional
        Random seed for reproducibility.
    integrator : str, optional
        "rk4" for fixed-step RK4, or "solve_ivp" for scipy adaptive solver.
    ivp_method : str, optional
        Method passed to solve_ivp if integrator="solve_ivp".
    noise_on : tuple, optional
        Which variables receive additive noise. Any subset of ("u", "v", "y").
    return_dataframe : bool, optional
        Whether to include a pandas DataFrame in the returned dictionary.

    Returns
    -------
    results : dict
        Keys:
            - "t"
            - "states"
            - "states_noisy"
            - "noise"
            - "df" (optional)
    """
    initial_condition = np.asarray(initial_condition, dtype=float)

    if len(initial_condition) != 3:
        raise ValueError("initial_condition must have length 3: [u0, v0, y0].")

    if n_steps < 2:
        raise ValueError("n_steps must be at least 2.")

    integrator = integrator.lower()

    if integrator == "rk4":
        t, states = integrate_rk4(dynamic_system, initial_condition, t_span, n_steps)

    elif integrator == "solve_ivp":
        t, states = integrate_solve_ivp(
            dynamic_system,
            initial_condition,
            t_span,
            n_steps,
            method=ivp_method
        )
    else:
        raise ValueError("integrator must be either 'rk4' or 'solve_ivp'.")

    # Additive Gaussian noise
    rng = np.random.default_rng(random_seed)

    variable_to_index = {"u": 0, "v": 1, "y": 2}
    noise = np.zeros_like(states)

    for var in noise_on:
        if var not in variable_to_index:
            raise ValueError("noise_on entries must be chosen from ('u', 'v', 'y').")
        idx = variable_to_index[var]
        noise[:, idx] = rng.normal(loc=0.0, scale=noise_std, size=states.shape[0])

    states_noisy = states + noise

    results = {
        "t": t,
        "states": states,
        "states_noisy": states_noisy,
        "noise": noise
    }

    if return_dataframe:
        df = pd.DataFrame({
            "t": t,
            "u": states[:, 0],
            "v": states[:, 1],
            "y": states[:, 2],
            "u_noisy": states_noisy[:, 0],
            "v_noisy": states_noisy[:, 1],
            "y_noisy": states_noisy[:, 2],
            "u_noise": noise[:, 0],
            "v_noise": noise[:, 1],
            "y_noise": noise[:, 2],
        })
        results["df"] = df

    return results


def plot_results(results, use_noisy_scatter=True):
    """
    Plot clean and noisy trajectories.
    """
    t = results["t"]
    states = results["states"]
    states_noisy = results["states_noisy"]
    labels = ["u(t)", "v(t)", "y(t)"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(t, states[:, i], label=f"{labels[i]} true", linewidth=2)

        if use_noisy_scatter:
            ax.scatter(t, states_noisy[:, i], label=f"{labels[i]} noisy", s=12, alpha=0.6)
        else:
            ax.plot(t, states_noisy[:, i], "--", label=f"{labels[i]} noisy", alpha=0.8)

        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    initial_condition = [0.2, 0.1, 1.0]
    t_span = (0.0, 20.0)
    n_steps = 400
    noise_std = 0.02

    results = simulate_system(
        initial_condition=initial_condition,
        t_span=t_span,
        n_steps=n_steps,
        noise_std=noise_std,
        random_seed=42,
        integrator="rk4",          # or "solve_ivp"
        ivp_method="RK45",         # only used if integrator="solve_ivp"
        noise_on=("u", "v", "y"),  # or e.g. ("u", "v")
        return_dataframe=True
    )

    print(results["df"].head())
    plot_results(results)