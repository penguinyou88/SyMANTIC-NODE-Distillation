from __future__ import annotations

import re

import numpy as np

from .complexity import calculate_complexity


def _rename_sindy_input_tokens(equations: list[str], input_feature_names: list[str]) -> list[str]:
    token_pattern = re.compile(r"\bu(\d+)\b")

    def _replace_token(match: re.Match[str]) -> str:
        token_index = int(match.group(1))
        if 0 <= token_index < len(input_feature_names):
            return input_feature_names[token_index]
        return match.group(0)

    return [token_pattern.sub(_replace_token, eq) for eq in equations]


def _bind_named_equation_accessor(model, input_feature_names: list[str]):
    original_equations = model.equations

    def _equations(*args, **kwargs):
        # Force default token output (`u0`, `u1`, ...) and apply our own mapping
        # so naming remains stable across pysindy versions.
        kwargs.pop("input_features", None)
        equations = original_equations(*args, **kwargs)
        return _rename_sindy_input_tokens(equations, input_feature_names)

    model.equations = _equations



def fit_sindy_models(
    time: np.ndarray,
    states: np.ndarray,
    inputs: np.ndarray,
    gradients: np.ndarray,
    poly_degree: int = 2,
    threshold_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 5e-2),
    alpha: float = 1e-2,
    complexity_weight: float = 1e-3,
    include_interaction: bool = True,
    include_bias: bool = True,
):
    try:
        import pysindy as ps
    except Exception as exc:
        raise ImportError(
            "pysindy is required for the SINDy branch. Install with: pip install pysindy"
        ) from exc

    input_feature_names = ["t"] + [f"u{i}" for i in range(inputs.shape[1])]
    augmented_inputs = np.column_stack((time.reshape(-1, 1), inputs))

    best = None
    for threshold in threshold_grid:
        library = ps.PolynomialLibrary(
            degree=poly_degree,
            include_interaction=include_interaction,
            include_bias=include_bias,
        )
        optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
        model = ps.SINDy(feature_library=library, optimizer=optimizer)
        model.fit(x=states, t=time, u=augmented_inputs, x_dot=gradients)

        xdot_hat = model.predict(x=states, u=augmented_inputs)
        mse = float(np.mean((xdot_hat[:, :2] - gradients) ** 2))
        coeffs = np.asarray(model.coefficients(), dtype=float)
        nonzero_terms = int(np.count_nonzero(np.abs(coeffs) > 1e-12))
        objective = mse + complexity_weight * float(nonzero_terms)
        if best is None or objective < best["objective"]:
            best = {
                "model": model,
                "mse": mse,
                "threshold": threshold,
                "nonzero_terms": nonzero_terms,
                "objective": objective,
            }

    assert best is not None
    model = best["model"]
    _bind_named_equation_accessor(model, input_feature_names)
    equations = model.equations()

    state_equations = equations[:2]
    complexities = [calculate_complexity(eq, method="operator_count") for eq in state_equations]

    return {
        "backend": "sindy",
        "model": model,
        "equations": state_equations,
        "complexities": complexities,
        "fit_mse": best["mse"],
        "selection_objective": best["objective"],
        "selected_threshold": best["threshold"],
        "nonzero_terms": best["nonzero_terms"],
        "feature_names": ["x0", "x1"] + input_feature_names,
    }



def make_rhs_from_sindy(fit_result):
    model = fit_result["model"]
    feature_names = fit_result.get("feature_names", [])
    expected_u_dim = max(0, len(feature_names) - 3) if feature_names else None

    def rhs(_t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        u_vec = np.asarray(u, dtype=float).reshape(-1)
        if expected_u_dim is not None and u_vec.size != expected_u_dim:
            raise ValueError(
                f"Expected {expected_u_dim} control inputs (u0..u{expected_u_dim - 1}), got {u_vec.size}."
            )

        u_augmented = np.concatenate(([float(_t)], u_vec))
        xdot = model.predict(x=x.reshape(1, -1), u=u_augmented.reshape(1, -1))
        return np.asarray(xdot, dtype=float).reshape(-1)[: x.shape[0]]

    return rhs
