"""Custom exceptions for SyMANTIC."""


class FeatureSpaceLimitError(RuntimeError):
    """Raised when the expanded feature space exceeds max_features.

    Parameters
    ----------
    n_features : int
        Current number of features.
    max_features : int
        The configured limit.
    """

    def __init__(self, n_features: int, max_features: int):
        self.n_features = n_features
        self.max_features = max_features
        super().__init__(
            f"Expanded feature space ({n_features} features) exceeds "
            f"max_features={max_features}. Stopping expansion. "
            f"Increase max_features to allow deeper expansion, or use "
            f"initial_screening to reduce the feature space earlier."
        )


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass
