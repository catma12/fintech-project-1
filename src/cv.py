"""Walk-forward cross-validation splitters for horizon-specific gap enforcement.

For a horizon-h target measured as CAR[t+1 .. t+h], a fold's last training
month t_train and first test month t_test have overlapping label windows
whenever t_test - t_train < h. A gap of at least h+1 months between train and
test prevents this leakage. The guide specifies gap=3, which is adequate for
h=1 but insufficient for h>=3; we use gap=max(3, h+1) so h=12 is protected.
"""
from __future__ import annotations

from sklearn.model_selection import TimeSeriesSplit


def walk_forward_splits(horizon: int, n_splits: int = 5) -> TimeSeriesSplit:
    """TimeSeriesSplit with horizon-aware gap.

    Parameters
    ----------
    horizon : int
        Target horizon in months (1, 3, 6, or 12).
    n_splits : int
        Number of walk-forward folds (default 5).
    """
    gap = max(3, horizon + 1)
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)
