"""Milestone E — 2026 out-of-sample prediction (STUB — blocked on data).

The panel currently ends 2025-12-31. When monthly 2026 rows are appended to
``master_panel_clean.csv``, the functions below should be implemented and run
to produce the assignment-required comparison of predicted vs. realized
sector CARs.

Design
------
For each month t in 2026 where a large shock lands:

  1. Refit the Kilian VAR on the panel subset ``[1986-01 .. t]`` so the
     structural identification uses only historically-available information.
  2. Extract that month's shock triplet from the VAR residuals.
  3. Build the month-t feature vector using ``src.features.build_feature_matrix``
     restricted to the same window.
  4. Load the 48 persisted regressors from ``outputs/models/``.
  5. Generate predictions at h ∈ {1, 3, 6} — h=12 is usually omitted because
     12-month realized returns are not yet observable for most of 2026.
  6. Once h months pass and realized CARs become observable, diff predicted
     vs. realized and append to the OOS comparison table.

The report template has a placeholder Section 5 that will auto-populate
once ``outputs/oos_2026/predictions.parquet`` is written.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import FF_INDUSTRIES, HORIZONS


def predict_for_month(panel_extended: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Generate predicted CARs at h ∈ {1, 3, 6} for one 2026 month t.

    Parameters
    ----------
    panel_extended : pd.DataFrame
        The full panel (1986-01 through t) with monthly index. Must contain
        all columns required by ``src.features`` and the VAR input columns.
    t : pd.Timestamp
        The month-end timestamp at which to make predictions.

    Returns
    -------
    pd.DataFrame with columns (sector, horizon) and a single row indexed by t.
    """
    raise NotImplementedError(
        "Milestone E is pending 2026 data append. See module docstring for "
        "the intended interface and workflow."
    )


def build_oos_comparison_table(
    predictions_path: Path, realized_panel: pd.DataFrame
) -> pd.DataFrame:
    """Diff predicted CARs against realized CARs as 2026 horizons close.

    When implemented, will produce a long-format table with columns
    [date, sector, horizon, predicted_car, realized_car, error, correct_sign].
    The report renderer picks this up automatically.
    """
    raise NotImplementedError("Pending Milestone E.")


def main() -> None:
    print(
        "Milestone E is currently a stub. The panel ends 2025-12-31 and no "
        "2026 rows are appended. See src/oos_2026.py module docstring for "
        "the full interface that will be implemented once data lands."
    )


if __name__ == "__main__":
    main()
