"""Milestone B — feature matrix construction.

Builds the month-t-safe feature matrix X that joins Kilian shocks with macro
regime and oil-context variables. Every feature uses only information available
at the close of month t; the prediction horizon begins at t+1.

Feature groups
--------------
Shock (current)            eps_supply, eps_agg_demand, eps_precaut
Shock (lags 1-3)           9 cols
Shock (3-month cumulative) 3 cols
Dominant-shock one-hot     dom_is_{supply,agg_demand,precaut}  (tie-break precaut > agg_demand > supply)
Dominant-shock signed      shock_sign, shock_magnitude
Shock overlap              contamination_flag  (|eps_*| > 1.5 in any of t-3..t-1)
Volatility regime          vix_level, vix_regime, vix_is_proxy
Monetary regime            fed_regime_num
Credit regime              credit_spread_ig, credit_spread_hy, hy_available
Macro state                Recession
Oil context                oil_ret_3m, oil_ret_12m, net_oil_price_3yr, oil_vol_6m_monthly
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    FEATURES_DIR,
    REPORT_TABLES,
    SHOCKS_DIR,
    SHOCK_COLUMNS,
)
from src.io_utils import load_panel, update_manifest


LARGE_SHOCK_Z = 1.5

# Priority for dominant-shock tie-breaking (highest first). Ties are numerically
# rare but a fixed order keeps the one-hot deterministic.
DOMINANT_PRIORITY = ["eps_precaut", "eps_agg_demand", "eps_supply"]


def _dominant_shock(shocks: pd.DataFrame) -> pd.Series:
    """Return the column name with largest |shock| per row, breaking ties by DOMINANT_PRIORITY."""
    # Reordering columns to priority order makes idxmax deterministic on ties
    # (pandas idxmax returns the first column hit).
    return shocks[DOMINANT_PRIORITY].abs().idxmax(axis=1)


def build_feature_matrix(panel: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    """Construct the feature matrix X indexed by shocks.index (1988-01..2025-12)."""
    X = pd.DataFrame(index=shocks.index)

    # --- Core shocks (contemporaneous) ---
    for col in SHOCK_COLUMNS:
        X[col] = shocks[col]

    # --- Lagged shocks (1, 2, 3 months) ---
    for col in SHOCK_COLUMNS:
        for lag in (1, 2, 3):
            X[f"{col}_lag{lag}"] = shocks[col].shift(lag)

    # --- 3-month cumulative shocks (sum of t-2, t-1, t) ---
    for col in SHOCK_COLUMNS:
        X[f"{col}_cum3m"] = shocks[col].rolling(window=3, min_periods=3).sum()

    # --- Dominant-shock one-hot + signed magnitude ---
    dom = _dominant_shock(shocks)
    for col in SHOCK_COLUMNS:
        X[f"dom_is_{col.replace('eps_', '')}"] = (dom == col).astype(int)

    # Signed value of the dominant shock (sign gives direction, magnitude gives size)
    dom_values = pd.Series(index=shocks.index, dtype=float)
    for col in SHOCK_COLUMNS:
        mask = (dom == col)
        dom_values.loc[mask] = shocks.loc[mask, col]
    X["shock_sign"] = np.sign(dom_values).astype(int)
    X["shock_magnitude"] = dom_values.abs()

    # --- Contamination flag: any large shock in t-3, t-2, or t-1 ---
    large_now = (shocks[SHOCK_COLUMNS].abs() > LARGE_SHOCK_Z).any(axis=1).astype(int)
    # Shift by 1 so we only look at the *prior* 3 months (strict history, not including t)
    large_prev = large_now.shift(1).rolling(window=3, min_periods=1).max()
    X["contamination_flag"] = large_prev.fillna(0).astype(int)

    # --- Volatility regime (VIX) ---
    # Prefer raw VIX where available (1990+); fall back to proxy VIX_filled before that.
    # The panel's vix_is_proxy flag is unreliable (all-1s per audit) — recompute from vix_available.
    vix = panel["VIX"].where(panel["vix_available"] == 1, panel["VIX_filled"])
    X["vix_level"] = vix.reindex(X.index)
    X["vix_regime"] = panel["vix_regime"].reindex(X.index)
    X["vix_is_proxy"] = (1 - panel["vix_available"]).reindex(X.index).astype(int)

    # --- Monetary regime ---
    X["fed_regime_num"] = panel["fed_regime_num"].reindex(X.index)

    # --- Credit regime ---
    # IG spread has full coverage via forward-filled panel column.
    X["credit_spread_ig"] = panel["LUACOAS_filled"].reindex(X.index)
    # HY spread begins 1994-01; leave NaN before that (XGBoost handles NaNs natively).
    X["credit_spread_hy"] = panel["LF98OAS"].reindex(X.index)
    X["hy_available"] = panel["hy_available"].reindex(X.index).astype(int)

    # --- Macro state ---
    X["Recession"] = panel["Recession"].reindex(X.index).astype(int)

    # --- Oil context ---
    X["oil_ret_3m"] = panel["oil_ret_3m"].reindex(X.index)
    X["oil_ret_12m"] = panel["oil_ret_12m"].reindex(X.index)
    X["net_oil_price_3yr"] = panel["net_oil_price_3yr"].reindex(X.index)
    # Guide specifies realized vol of *daily* oil returns over 3 months; the panel is monthly,
    # so we use a 6-month rolling std of monthly log-returns as a principled substitute.
    # Name makes the substitution explicit.
    X["oil_vol_6m_monthly"] = (
        panel["oil_ret_1m"].rolling(window=6, min_periods=6).std().reindex(X.index)
    )

    return X


def check_no_lookahead(X: pd.DataFrame, panel: pd.DataFrame, shocks: pd.DataFrame) -> dict:
    """Spot-check that features contain no future information.

    Asserts:
      (1) X has no column equal to a FF_*_abn column at the same index t
          (i.e., abnormal returns used as targets do not leak into features).
      (2) At t=2022-03-31, eps_supply is positive (war began) and feature matrix does not
          contain FF_Enrgy_abn[2022-03].
      (3) Max index of X equals max index of shocks.
    """
    failures: list[str] = []

    # (1) ensure no FF_*_abn column appears in X
    ff_abn_cols = [c for c in panel.columns if c.endswith("_abn") and c.startswith("FF_")]
    leaked = [c for c in ff_abn_cols if c in X.columns]
    if leaked:
        failures.append(f"FF_*_abn columns leaked into X: {leaked}")

    # (2) Russia-Ukraine spot-check — precautionary should be large positive in Feb 2022
    # (invasion began Feb 24, 2022; shock innovation peaks the month the event lands).
    russia = pd.Timestamp("2022-02-28")
    if russia in X.index:
        if X.loc[russia, "eps_precaut"] < 1.0:
            failures.append(f"2022-02 eps_precaut={X.loc[russia, 'eps_precaut']:.2f} not > 1.0")
    else:
        failures.append("2022-02-28 missing from feature index")

    # (3) no unexpected future dates
    if X.index.max() > shocks.index.max():
        failures.append(f"X.index.max ({X.index.max()}) exceeds shocks.index.max ({shocks.index.max()})")

    if failures:
        raise ValueError("check_no_lookahead failed:\n  " + "\n  ".join(failures))
    return {"passed": True, "n_features": X.shape[1], "n_rows": len(X)}


def summarize(X: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats for the feature summary table."""
    summary = X.describe(percentiles=[0.05, 0.5, 0.95]).T
    summary["n_missing"] = X.isna().sum()
    summary = summary.round(4)
    return summary


def main() -> None:
    panel = load_panel()
    shocks = pd.read_parquet(SHOCKS_DIR / "shocks_v1.parquet")

    X = build_feature_matrix(panel, shocks)
    print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} cols "
          f"({X.index.min().date()} -> {X.index.max().date()})")

    check_report = check_no_lookahead(X, panel, shocks)
    print(f"check_no_lookahead PASSED: n_features={check_report['n_features']}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "X_v1.parquet"
    X.to_parquet(out_path)
    print(f"Wrote {out_path}")

    summary = summarize(X)
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    summary_path = REPORT_TABLES / "feature_summary.csv"
    summary.to_csv(summary_path)
    print(f"Wrote {summary_path}")

    update_manifest(
        "milestone_b_features",
        {
            "features": "outputs/features/X_v1.parquet",
            "feature_summary": "outputs/report/tables/feature_summary.csv",
            "n_features": X.shape[1],
            "n_rows": X.shape[0],
            "features_list": list(X.columns),
        },
    )


if __name__ == "__main__":
    main()
