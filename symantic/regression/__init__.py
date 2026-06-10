"""Regression modules for SyMANTIC."""

from .l0_greedy import Regressor as NonDimensionalRegressor
from .l0_greedy_dimensional import Regressor as DimensionalRegressor
from .screening import Regressor as DimensionalScreeningRegressor
from .penalized import PenalizedRegressor
from .penalized_dimensional import PenalizedRegressor as DimensionalPenalizedRegressor
from .factory import get_regressor
