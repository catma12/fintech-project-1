"""Verification diagnostics for the Kilian VAR shock identification.

Three checks addressing methodology questions a reviewer (or teammate) might raise:

  1. ADF stationarity tests on VAR inputs. Documents the choice to feed the
     VAR first-differenced real oil prices (as in Kilian's robustness specs)
     rather than log levels.

  2. Sign-convention check around the 1990 Gulf War. Narrative: Iraq invasion
     Aug 2 suspends ~4.5 mb/d of supply (expect negative eps_supply); Saudi
     ramp-up hits production reports in Sept/Oct (expect positive eps_supply
     to follow); war-risk premium drives precautionary spike throughout.

  3. Forecast-error variance decomposition (FEVD) per sector: fit a 4-variable
     VAR [dprod, kilian_rea, real_oil_price_diff, FF_sector_abn] per industry
     and report the share of sector-return variance attributable to each
     structural oil shock at horizons 1/6/12/24 months. This quantifies the
     upper bound on predictability from oil shocks alone — distinct from our
     OOF R² because FEVD is in-sample and descriptive.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR

from src.config import FF_INDUSTRIES, REPORT_FIGURES, REPORT_TABLES, SHOCKS_DIR, VAR_LAGS
from src.io_utils import load_panel


# --- 1. ADF stationarity ----------------------------------------------------

def run_adf() -> pd.DataFrame:
    panel = load_panel()
    candidates = {
        "dprod": panel["dprod"],
        "kilian_rea": panel["kilian_rea"],
        "real_oil_price (log level)": panel["real_oil_price"],
        "real_oil_price_diff (log 1st diff)": panel["real_oil_price_diff"],
    }
    rows: list[dict] = []
    for name, series in candidates.items():
        s = series.dropna()
        result = adfuller(s, autolag="AIC", maxlag=24)
        rows.append({
            "variable": name,
            "adf_stat": float(result[0]),
            "p_value": float(result[1]),
            "crit_1pct": float(result[4]["1%"]),
            "crit_5pct": float(result[4]["5%"]),
            "stationary_at_5pct": bool(result[1] < 0.05),
            "n_obs": len(s),
        })
    return pd.DataFrame(rows)


# --- 2. Sign-convention check -----------------------------------------------

def sign_check() -> pd.DataFrame:
    panel = load_panel()
    shocks = pd.read_parquet(SHOCKS_DIR / "shocks_v1.parquet")
    window = shocks.loc["1990-06-30":"1990-12-31"].copy()
    window["world_oil_prod_tbpd"] = panel.loc[window.index, "world_oil_prod_tbpd"]
    window["dprod"] = panel.loc[window.index, "dprod"]
    return window.round(3)


# --- 3. Variance decomposition ----------------------------------------------

def fevd_per_sector(horizons=(1, 6, 12, 24)) -> pd.DataFrame:
    """Fit 4-variable VARs [dprod, rea, oil_diff, FF_s_abn] for each sector,
    extract forecast-error variance decomposition of the sector return at
    given horizons. Returns long-format table (sector, horizon, supply_share,
    agg_demand_share, precautionary_share, own_share)."""
    panel = load_panel()
    base_cols = ["dprod", "kilian_rea", "real_oil_price_diff"]

    rows: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in FF_INDUSTRIES:
            data = panel[base_cols + [f"FF_{s}_abn"]].dropna().rename(
                columns={f"FF_{s}_abn": "sector"}
            )
            model = VAR(data)
            results = model.fit(maxlags=VAR_LAGS, ic=None)
            fevd = results.fevd(periods=max(horizons) + 1)
            # fevd.decomp has shape (n_eqs, periods, n_eqs). Row = response variable, col = shock.
            # Variable order = base_cols + ['sector']; sector is index 3.
            # FEVD contribution of each shock to 'sector' response at period h (1-indexed in reporting).
            decomp = fevd.decomp[3]  # shape (periods, n_eqs)
            for h in horizons:
                shares = decomp[h - 1]  # 0-indexed period h-1
                rows.append({
                    "sector": s,
                    "horizon": h,
                    "supply_share": float(shares[0]),
                    "agg_demand_share": float(shares[1]),
                    "precautionary_share": float(shares[2]),
                    "own_share": float(shares[3]),
                    "oil_total_share": float(shares[0] + shares[1] + shares[2]),
                })
    return pd.DataFrame(rows)


def plot_fevd_heatmap(fevd: pd.DataFrame, out_path) -> None:
    """Horizon-12 FEVD heatmap: sectors × shock components."""
    h12 = fevd[fevd["horizon"] == 12].set_index("sector")
    m = h12[["supply_share", "agg_demand_share", "precautionary_share", "oil_total_share"]]
    m.columns = ["Supply", "Aggregate demand", "Precautionary", "All oil (sum)"]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        m * 100, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=30,
        cbar_kws={"label": "FEVD share (%)"}, ax=ax,
    )
    ax.set_title(
        "Variance decomposition at 12m: sector return variance attributable to oil shocks",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel(""); ax.set_ylabel("Sector")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --- Main --------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("1) ADF stationarity tests on VAR inputs")
    print("=" * 70)
    adf = run_adf()
    print(adf.round(4).to_string(index=False))
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    adf.to_csv(REPORT_TABLES / "adf_results.csv", index=False)
    print(f"\nWrote {REPORT_TABLES / 'adf_results.csv'}")

    print("\n" + "=" * 70)
    print("2) Sign-convention check — 1990 Gulf War window")
    print("=" * 70)
    print("Narrative: Iraq invasion 1990-08-02 (expect negative eps_supply that")
    print("month as 4.5 mb/d of supply is suspended). Saudi ramp-up hits Sep/Oct")
    print("(expect positive eps_supply). Precautionary should be broadly positive")
    print("throughout from war-risk premium.\n")
    window = sign_check()
    print(window[["eps_supply", "eps_agg_demand", "eps_precaut", "dprod", "world_oil_prod_tbpd"]]
          .to_string())
    window.to_csv(REPORT_TABLES / "sign_check_1990.csv")
    print(f"\nWrote {REPORT_TABLES / 'sign_check_1990.csv'}")

    print("\n" + "=" * 70)
    print("3) Forecast-error variance decomposition — per-sector 4-var VAR")
    print("=" * 70)
    fevd = fevd_per_sector()

    print("\n-- Mean across sectors, by horizon --")
    print(
        fevd.groupby("horizon")[
            ["supply_share", "agg_demand_share", "precautionary_share", "oil_total_share"]
        ].mean().round(3).to_string()
    )

    print("\n-- FEVD at h=12 per sector (% of sector variance explained by oil shocks) --")
    h12 = fevd[fevd["horizon"] == 12].copy()
    h12 = h12.set_index("sector")
    print(
        (h12[["supply_share", "agg_demand_share", "precautionary_share", "oil_total_share"]] * 100)
        .round(2).to_string()
    )

    fevd.to_csv(REPORT_TABLES / "variance_decomposition.csv", index=False)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    plot_fevd_heatmap(fevd, REPORT_FIGURES / "variance_decomposition_heatmap.png")
    print(f"\nWrote {REPORT_TABLES / 'variance_decomposition.csv'}")
    print(f"Wrote {REPORT_FIGURES / 'variance_decomposition_heatmap.png'}")


if __name__ == "__main__":
    main()
