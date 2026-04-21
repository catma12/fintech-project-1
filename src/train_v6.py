"""v6 notebook-family model specifications (v6a/v6b/v6c).

These variants were developed in notebooks as part of the post-v5
algorithm-selection pass and are documented here for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class V6Variant:
    name: str
    objective: str
    notes: str


V6A = V6Variant(
    name="v6a",
    objective="Random Forest baseline selection",
    notes="Initial RF-based comparison against v5 ranker results in notebook workflow.",
)

V6B = V6Variant(
    name="v6b",
    objective="Random Forest + Optuna tuning",
    notes="Per-horizon RF hyperparameter tuning and feature-threshold sweeps.",
)

V6C = V6Variant(
    name="v6c",
    objective="Full algorithm shootout",
    notes="Cross-model comparison using report-consistent ranking metrics.",
)


V6_VARIANTS = (V6A, V6B, V6C)
