import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def dynamic_system(t, state):
    """
    Spruce-budworm outbreak model:
        dw_dt = r * w * (1 - w/a) - w**2 / (1 + w**2)
    with r = 0.5, a = 10.0

    Parameters
    ----------
    t : float
        Time (included for solver compatibility).
    state : array-like of length 1
        Current state [w].

    Returns
    -------
    np.ndarray
        Derivatives [dw_dt].
    """
    w = state[0]
    r, a = 0.5, 10.0
    dw_dt = r * w * (1.0 - w / a) - w**2 / (1.0 + w**2)
    return np.array([dw_dt], dtype=float)


def rk4_step(f, t, x, dt):
    """
    One fixed-step classical RK4 step.
    """
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_rk4(f, x0, t_span, n_steps):
    """
    Fixed-step RK4 integrator over a uniform grid.
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
        vectorized=False,
        rtol=1e-10,
        atol=1e-10
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
    noise_on=("w",),
    return_dataframe=True
):
    """
    Simulate the dynamic system and optionally add Gaussian noise.
    """
    initial_condition = np.asarray(initial_condition, dtype=float)

    if len(initial_condition) != 1:
        raise ValueError("initial_condition must have length 1: [w0].")

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
    noise = np.zeros_like(states)

    if "w" in noise_on:
        noise[:, 0] = rng.normal(loc=0.0, scale=noise_std, size=states.shape[0])

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
            "w": states[:, 0],
            "w_noisy": states_noisy[:, 0],
            "w_noise": noise[:, 0],
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
    labels = ["w(t)"]

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(t, states[:, 0], label="w(t) true", linewidth=2)

    if use_noisy_scatter:
        ax.scatter(t, states_noisy[:, 0], label="w(t) noisy", s=12, alpha=0.6)
    else:
        ax.plot(t, states_noisy[:, 0], "--", label="w(t) noisy", alpha=0.8)

    ax.set_ylabel("w")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel("Time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    initial_condition = [3.0]
    t_span = (0.0, 12.0)
    n_steps = 61
    noise_std = 0.05

    results = simulate_system(
        initial_condition=initial_condition,
        t_span=t_span,
        n_steps=n_steps,
        noise_std=noise_std,
        random_seed=42,
        integrator="solve_ivp",
        ivp_method="RK45",
        noise_on=("w",),
        return_dataframe=True
    )

    print(results["df"].head())
    plot_results(results)