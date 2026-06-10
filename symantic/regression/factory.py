"""Factory function for selecting the appropriate regressor class."""


def get_regressor(regularization='l0', dimensional=False):
    """Return the Regressor class for the given regularization type.

    Parameters
    ----------
    regularization : str
        'l0', 'l1', 'l2', or 'elastic_net'.
    dimensional : bool
        Whether to use the dimensional (unit-aware) variant.

    Returns
    -------
    class
        The Regressor class (not an instance) to be instantiated by the caller.
    """
    if regularization == 'l0':
        if dimensional:
            from .l0_greedy_dimensional import Regressor
        else:
            from .l0_greedy import Regressor
        return Regressor

    # L1, L2, ElasticNet — all handled by PenalizedRegressor
    if dimensional:
        from .penalized_dimensional import PenalizedRegressor
        return PenalizedRegressor
    from .penalized import PenalizedRegressor
    return PenalizedRegressor
