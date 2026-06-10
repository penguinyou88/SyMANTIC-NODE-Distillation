"""Result types for SyMANTIC model fitting."""

import dataclasses
from typing import Optional

import pandas as pd


@dataclasses.dataclass
class FitResult:
    """Unified result from SymanticModel.fit().

    Provides a consistent interface regardless of whether auto-depth or
    fixed-depth mode was used, or whether multi-task was enabled.

    Supports backward-compatible tuple unpacking:
        - Auto-depth:  ``res, pareto_df = model.fit()``
        - Fixed-depth: ``rmse, equation, r2 = model.fit()``
        - Multi-task:  ``rmse, equation, r2, equations = model.fit()``

    Attributes
    ----------
    rmse : float
        Root mean squared error of the best model.
    equation : str
        String representation of the discovered equation.
    r2 : float
        R-squared score of the best model.
    complexity : float or None
        Complexity of the utopia equation (auto-depth only).
    pareto_front : DataFrame or None
        Full Pareto frontier with columns [Loss, Complexity, R2, Equation].
        Populated only in auto-depth mode.
    all_equations : list or None
        List of equations from each target (multi-task only).
    """

    rmse: float
    equation: str
    r2: float
    complexity: Optional[float] = None
    pareto_front: Optional[pd.DataFrame] = None
    all_equations: Optional[list] = None

    def __iter__(self):
        """Support backward-compatible tuple unpacking.

        Auto-depth mode (pareto_front is set):
            yields (utopia_dict, pareto_df) for ``res, df = model.fit()``
        Multi-task mode (all_equations is set):
            yields (rmse, equation, r2, equations)
        Fixed-depth mode:
            yields (rmse, equation, r2)
        """
        if self.pareto_front is not None and self.all_equations is None:
            utopia_dict = {
                'utopia': {
                    'expression': self.equation,
                    'rmse': self.rmse,
                    'r2': self.r2,
                    'complexity': self.complexity,
                }
            }
            yield utopia_dict
            yield self.pareto_front
        elif self.all_equations is not None:
            yield self.rmse
            yield self.equation
            yield self.r2
            yield self.all_equations
        else:
            yield self.rmse
            yield self.equation
            yield self.r2

    def __len__(self):
        """Length for unpacking support."""
        if self.pareto_front is not None and self.all_equations is None:
            return 2
        elif self.all_equations is not None:
            return 4
        else:
            return 3
