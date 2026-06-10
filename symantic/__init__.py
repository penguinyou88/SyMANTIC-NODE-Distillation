"""
SyMANTIC: An Efficient Symbolic Regression Method for Interpretable
and Parsimonious Model Discovery in Science and Beyond.
"""

from .model import SymanticModel
from .pareto import pareto
from .results import FitResult
from .exceptions import FeatureSpaceLimitError, ValidationError

# Feature expansion - qualified names to avoid namespace collision
from .feature_expansion.nondimensional import feature_space_construction as NonDimensionalFeatureExpander
from .feature_expansion.dimensional import feature_space_construction as DimensionalFeatureExpander

# Regression - qualified names to avoid namespace collision
from .regression.l0_greedy import Regressor as NonDimensionalRegressor
from .regression.l0_greedy_dimensional import Regressor as DimensionalRegressor
from .regression.screening import Regressor as DimensionalScreeningRegressor
from .regression.penalized import PenalizedRegressor
from .regression.factory import get_regressor

# Backward-compatible default aliases (non-dimensional versions)
from .feature_expansion.nondimensional import feature_space_construction
from .regression.l0_greedy import Regressor

__version__ = "2.0.0"
