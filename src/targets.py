"""Milestone B — target matrix construction.

Builds the 48-column target matrix Y (12 Fama-French industries × 4 horizons)
of forward cumulative abnormal returns (CAR):

    CAR(s, t, h) = sum( FF_s_abn[t+1 .. t+h] )

The panel's FF_s_abn columns are already monthly industry-minus-market abnormal
returns, so a forward rolling sum delivers the h-month CAR directly. We keep
arithmetic sums (not compounded BHAR) for three reasons:

  1. The monthly building block is already arithmetic abnormal returns.
  2. Trees handle arithmetic-scaled targets cleanly.
  3. Arithmetic CAR is the standard event-study convention and additive across
     horizons.

Targets are NOT annualized: we want per-horizon-specific signal strength,
which annualizing would compress.

A parallel sign-classification target matrix Y_sign is produced for the binary
variant recommended by the implementation guide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FF_INDUSTRIES, HORIZONS, TARGETS_DIR
from src.io_utils import load_panel, update_manifest


def build_car_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Return Y with MultiIndex columns (sector, horizon) of forward CARs."""
    data: dict[tuple[str, int], pd.Series] = {}
    for s in FF_INDUSTRIES:
        abn = panel[f"FF_{s}_abn"]
        for h in HORIZONS:
            # rolling(h).sum() at index t gives sum(abn[t-h+1 .. t]).
            # Shift back by h so the value aligned at t becomes sum(abn[t+1 .. t+h]).
            data[(s, h)] = abn.rolling(window=h, min_periods=h).sum().shift(-h)
    Y = pd.DataFrame(data)
    Y.columns = pd.MultiIndex.from_tuples(Y.columns, names=["sector", "horizon"])
    return Y


def build_sign_targets(Y: pd.DataFrame) -> pd.DataFrame:
    """Binary-sign target: +1 (outperform), -1 (underperform), 0 (zero CAR / missing)."""
    return np.sign(Y).astype("Int8")  # nullable int so NaN rows stay NaN


def check_target_alignment(Y: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Spot-check: Y.loc[2020-02, ('Enrgy', 1)] should equal FF_Enrgy_abn[2020-03].

    If this passes, the forward-shift is wired correctly.
    """
    t = pd.Timestamp("2020-02-29")
    tp1 = pd.Timestamp("2020-03-31")
    actual = Y.loc[t, ("Enrgy", 1)]
    expected = panel.loc[tp1, "FF_Enrgy_abn"]
    if not np.isclose(actual, expected, atol=1e-12):
        raise ValueError(
            f"Target alignment failed at t=2020-02: Y=(Enrgy,1)={actual:.6f} "
            f"but FF_Enrgy_abn[2020-03]={expected:.6f}"
        )
    # Also verify h=3 sum
    h3_actual = Y.loc[t, ("Enrgy", 3)]
    h3_expected = panel.loc[tp1:"2020-05-31", "FF_Enrgy_abn"].iloc[:3].sum()
    if not np.isclose(h3_actual, h3_expected, atol=1e-12):
        raise ValueError(
            f"Target alignment failed at t=2020-02 h=3: Y={h3_actual:.6f} expected={h3_expected:.6f}"
        )
    return {"passed": True, "h1_check": float(actual), "h3_check": float(h3_actual)}


def main() -> None:
    panel = load_panel()

    Y = build_car_targets(panel)
    print(f"CAR targets: {Y.shape[0]} rows × {Y.shape[1]} cols "
          f"(12 sectors × 4 horizons, {Y.index.min().date()} -> {Y.index.max().date()})")

    check = check_target_alignment(Y, panel)
    print(f"check_target_alignment PASSED: h1={check['h1_check']:.4f}, h3={check['h3_check']:.4f}")

    Y_sign = build_sign_targets(Y)

    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    Y_path = TARGETS_DIR / "Y_v1.parquet"
    Y.to_parquet(Y_path)
    print(f"Wrote {Y_path}")

    Y_sign_path = TARGETS_DIR / "Y_sign_v1.parquet"
    Y_sign.to_parquet(Y_sign_path)
    print(f"Wrote {Y_sign_path}")

    update_manifest(
        "milestone_b_targets",
        {
            "targets": "outputs/targets/Y_v1.parquet",
            "targets_sign": "outputs/targets/Y_sign_v1.parquet",
            "n_targets": Y.shape[1],
            "sectors": FF_INDUSTRIES,
            "horizons": HORIZONS,
            "alignment_check": check,
        },
    )


if __name__ == "__main__":
    main()
