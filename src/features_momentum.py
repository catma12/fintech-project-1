"""Momentum features — own-sector history + cross-sectional rank.

Adds the predictors that v1-v3 were missing. Each feature is sector-specific
so it must be attached at the per-sector row level, not shared across sectors:

  own_ret_1m        FF_{s}_abn at t                 (trailing 1m own return;
                                                    trades short-horizon reversal)
  own_ret_3m        sum FF_{s}_abn[t-2..t]          (3m own-sector momentum)
  own_mom_12_1      sum FF_{s}_abn[t-12..t-2]       (Jegadeesh-Titman 12-1;
                                                    skips most recent month)
  own_vol_6m        rolling std FF_{s}_abn[t-5..t]  (idiosyncratic volatility)
  cross_rank_12_1   rank of this sector's          (cross-sectional momentum —
                    own_mom_12_1 among 12 sectors   direct input for rankers)

All features use information available at the close of month t; the CAR
target window starts at t+1. No look-ahead.
"""
from __future__ import annotations

import pandas as pd

from src.config import FF_INDUSTRIES


MOMENTUM_COLS = [
    "own_ret_1m",
    "own_ret_3m",
    "own_mom_12_1",
    "own_vol_6m",
    "cross_rank_12_1",
]


def compute_sector_momentum(panel: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Own-sector momentum features indexed by panel dates."""
    abn = panel[f"FF_{sector}_abn"]
    return pd.DataFrame({
        "own_ret_1m": abn,
        "own_ret_3m": abn.rolling(3, min_periods=3).sum(),
        "own_mom_12_1": abn.rolling(11, min_periods=11).sum().shift(1),
        "own_vol_6m": abn.rolling(6, min_periods=6).std(),
    })


def compute_cross_rank_12_1(panel: pd.DataFrame) -> pd.DataFrame:
    """For each month, rank the 12 sectors' 12-1 momentum. Higher = better."""
    mom = pd.DataFrame({
        s: panel[f"FF_{s}_abn"].rolling(11, min_periods=11).sum().shift(1)
        for s in FF_INDUSTRIES
    })
    # Rank within each row (date); method='average' handles occasional NaN rows
    return mom.rank(axis=1, method="average")


def attach_momentum_for_sector(
    X_base: pd.DataFrame, panel: pd.DataFrame, sector: str
) -> pd.DataFrame:
    """Return X_base copy with 5 momentum cols appended (per-cell training path)."""
    mom = compute_sector_momentum(panel, sector).reindex(X_base.index)
    cross = compute_cross_rank_12_1(panel)[sector].reindex(X_base.index)
    out = X_base.copy()
    for col in ["own_ret_1m", "own_ret_3m", "own_mom_12_1", "own_vol_6m"]:
        out[col] = mom[col]
    out["cross_rank_12_1"] = cross
    return out


def attach_momentum_to_long(X_long: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Inject sector-specific momentum into a pooled long-format frame.

    X_long has MultiIndex (date, sector). Rows are filtered to this sector's
    valid months; we just need to pull the right values by date for each
    sector and slot them into that sector's rows.
    """
    X = X_long.copy()
    for col in MOMENTUM_COLS:
        X[col] = 0.0

    cross = compute_cross_rank_12_1(panel)

    for s in FF_INDUSTRIES:
        mask = X.index.get_level_values("sector") == s
        if not mask.any():
            continue
        sector_dates = X.index[mask].get_level_values("date")
        mom = compute_sector_momentum(panel, s).reindex(sector_dates)
        for col in ["own_ret_1m", "own_ret_3m", "own_mom_12_1", "own_vol_6m"]:
            X.loc[mask, col] = mom[col].values
        X.loc[mask, "cross_rank_12_1"] = cross[s].reindex(sector_dates).values

    return X
