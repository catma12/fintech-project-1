"""v7 model specification.

LightGBM ranking model with Optuna tuning and global feature selection,
as developed in the notebook sequence and exported to report2 tables.
"""

from __future__ import annotations

V7_DESCRIPTION = {
    "version": "v7",
    "model_family": "LightGBMRanker",
    "selection": "Global feature selection",
    "tuning": "Optuna per horizon",
    "artifact_table": "outputs/report/tables/report2_per_horizon.csv",
}
