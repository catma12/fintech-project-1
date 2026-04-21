"""v8 model specifications.

v8a: weighted ensemble over candidate model OOF predictions.
v8b: LightGBM with horizon-specific feature selection (HFS).
"""

from __future__ import annotations

V8_1_DESCRIPTION = {
    "version": "v8a",
    "approach": "Weighted ensemble",
    "weight_constraint": "non-negative simplex (sum=1)",
    "artifact_table": "outputs/report/tables/report2_per_horizon.csv",
}

V8_2_DESCRIPTION = {
    "version": "v8b",
    "approach": "LightGBM + horizon-specific feature selection",
    "top_k_source": "v8_hfs_top_k column in report2_per_horizon.csv",
    "artifact_table": "outputs/report/tables/report2_per_horizon.csv",
}
