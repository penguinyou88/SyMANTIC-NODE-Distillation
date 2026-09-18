from __future__ import annotations

import pandas as pd
import numpy as np
import importlib.util
import sys
from pathlib import Path

from .complexity import calculate_complexity



def _try_import_symantic_model():
    # SyMANTIC can be available via pip, editable installs, or a local repo clone.
    # Try robust import paths in order and return SymanticModel when found.
    import_errors: list[str] = []

    # 1) Installed package path.
    try:
        from symantic.model import SymanticModel  # type: ignore

        return SymanticModel
    except Exception as exc:
        import_errors.append(f"symantic.model import failed: {exc}")

    # 2) Local clone fallback: load <repo>/symantic/model.py directly.
    # This bypasses fragile absolute imports in symantic/__init__.py.
    try:
        repo_root = Path(__file__).resolve().parents[1]
        symantic_dir = repo_root / "symantic"
        model_file = symantic_dir / "model.py"
        if model_file.exists():
            # Ensure local support modules (FeatureSpaceConstruction, etc.) are importable.
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            if str(symantic_dir) not in sys.path:
                sys.path.insert(0, str(symantic_dir))

            spec = importlib.util.spec_from_file_location("symantic_local_model", str(model_file))
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                SymanticModel = getattr(module, "SymanticModel", None)
                if SymanticModel is not None:
                    return SymanticModel
    except Exception as exc:
        import_errors.append(f"local symantic/model.py load failed: {exc}")

    # 3) Last resort: top-level model module in PYTHONPATH/cwd.
    try:
        from model import SymanticModel  # type: ignore

        return SymanticModel
    except Exception as exc:
        import_errors.append(f"top-level model import failed: {exc}")
        raise ImportError(
            "SyMANTIC backend not available. Ensure one of these works: "
            "`from symantic.model import SymanticModel`, local `symantic/model.py`, "
            "or `from model import SymanticModel`. "
            "Root causes: " + " | ".join(import_errors)
        ) from exc



def fit_symantic_models(time: np.ndarray, states: np.ndarray, inputs: np.ndarray, gradients: np.ndarray, return_point: str = 'utopia'):
    SymanticModel = _try_import_symantic_model()

    cols = ["x0", "x1", "t", "u0", "u1", "u2", "u3", "u4", "u5", "u6"]
    base_df = pd.DataFrame(np.hstack([states, time[:, None], inputs]), columns=cols)

    equations = []
    complexities = []
    models = []

    for k, target in enumerate(["dx1", "dx2"]):
        df = base_df.copy()
        df.insert(0, target, gradients[:, k])

        operators = ['+','-','*','/','^2'] #'ln','^-1'

        model = SymanticModel(df,operators=operators,n_term=3, sis_features=100,level_pruning=True, regularization='l1',disp=True,metrics=[0.1,0.99])

        from unittest.mock import patch

        with patch("builtins.input", return_value="no"):
            _res, pareto = model.fit()

        if pareto is None or len(pareto) == 0:
            raise RuntimeError("SyMANTIC returned empty Pareto set.")

        # Select Pareto point: 'utopia' (closest to [0,0]) or 'most_accurate' (last/most accurate)
        if return_point == 'utopia':
            # Find the solution on Pareto front closest to ideal point (0, 0) in (complexity, RMSE) space
            # Compute euclidean distance from (0,0) to each Pareto point
            final = model.final_df
            row_idx = final['Distance_to_Utopia'].idxmin()
        elif return_point == 'most_accurate':
            row_idx = -1
        else:
            raise ValueError(f"Unknown return_point option: {return_point}. Choose 'utopia' or 'most_accurate'.")
        
        eq = str(pareto.iloc[row_idx]["Equation"])
        equations.append(eq)
        complexities.append(calculate_complexity(eq, method="operator_count"))
        models.append(model)

    return {
        "backend": "symantic",
        "models": models,
        "equations": equations,
        "complexities": complexities,
        "feature_names": cols,
        "return_point": return_point,
    }



def _eval_equation(eq: str, x: np.ndarray, u: np.ndarray, t: float) -> float:
    local_vars = {
        "x0": float(x[0]),
        "x1": float(x[1]),
        "u0": float(u[0]),
        "u1": float(u[1]),
        "u2": float(u[2]),
        "u3": float(u[3]),
        "u4": float(u[4]),
        "u5": float(u[5]),
        "u6": float(u[6]),
        "t": float(t),
        "np": np,
    }
    return float(eval(eq, {"__builtins__": {}}, local_vars))



def make_rhs_from_symantic(fit_result):
    eq1, eq2 = fit_result["equations"]

    def rhs(_t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([
            _eval_equation(eq1, x, u, t=_t),
            _eval_equation(eq2, x, u, t=_t),
        ])

    return rhs
