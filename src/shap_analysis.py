"""Milestone D — SHAP interpretation.

Produces four deliverable figure types that connect the 48 trained XGBoost
regressors back to the three-channel economic framework from Kilian & Park
(2009): final demand, input cost, and risk-premium/inflation channels.

Outputs (all in outputs/report/figures/):
  shap_global_bar_h6.png        — 12-sector grid of top-10 feature bars
  shap_dependence_precaut_vix_Shops_h6.png
  shap_dependence_precaut_vix_Enrgy_h6.png
  shap_waterfall_Enrgy_h6_{event}.png  (4 events)
  shap_heatmap_h6.png           — 12 sectors × top-10 features, mean signed SHAP

Per-cell SHAP value arrays are saved as .npy to outputs/shap/ so they can be
re-loaded without rerunning the explainer.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb

from src.config import (
    FEATURES_DIR,
    FF_INDUSTRIES,
    HORIZONS,
    MODELS_DIR,
    REPORT_FIGURES,
    SHAP_DIR,
    TARGETS_DIR,
)
from src.io_utils import update_manifest


EVENT_DATES = {
    "2008-09_GFC": "2008-09-30",
    "2014-11_SaudiSupply": "2014-11-30",
    "2020-03_COVID": "2020-03-31",
    "2022-02_Russia": "2022-02-28",
}


def load_regressor(sector: str, horizon: int) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / f"{sector}_h{horizon}_reg.json"))
    return model


def compute_shap_all(X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    """Compute SHAP values for all 48 regression models and save to disk."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in FF_INDUSTRIES:
            for h in HORIZONS:
                y = Y[(s, h)]
                mask = y.notna()
                X_c = X.loc[mask]
                model = load_regressor(s, h)
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_c)
                base = float(explainer.expected_value)

                np.save(SHAP_DIR / f"shap_values_{s}_h{h}.npy", sv)
                out[(s, h)] = {"shap_values": sv, "base_value": base, "X": X_c}

    # Persist base values in one small parquet for easy lookup
    bases = pd.DataFrame(
        [(s, h, d["base_value"]) for (s, h), d in out.items()],
        columns=["sector", "horizon", "base_value"],
    )
    bases.to_csv(SHAP_DIR / "shap_base_values.csv", index=False)

    return out


def plot_global_bar_grid(shap_data: dict, horizon: int = 6, out_path=None) -> None:
    """4×3 grid, one panel per sector, showing top-10 |SHAP| features."""
    fig, axes = plt.subplots(4, 3, figsize=(16, 15))
    for ax, s in zip(axes.flat, FF_INDUSTRIES):
        entry = shap_data[(s, horizon)]
        sv = entry["shap_values"]
        X = entry["X"]
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-10:]
        ax.barh(range(len(top_idx)), mean_abs[top_idx], color="#1f3b66")
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([X.columns[i] for i in top_idx], fontsize=8)
        ax.set_title(f"FF_{s} (h={horizon})", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.25, axis="x")

    fig.suptitle(
        f"Global SHAP feature importance by sector (h={horizon}m, OOF-trained regressors)",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = REPORT_FIGURES / f"shap_global_bar_h{horizon}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_dependence(
    shap_data: dict,
    sector: str,
    horizon: int,
    feat_x: str,
    interaction_feat: str,
    out_path=None,
) -> None:
    """SHAP dependence plot for one (sector, horizon), colored by interaction feature."""
    entry = shap_data[(sector, horizon)]
    sv = entry["shap_values"]
    X = entry["X"]

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feat_x, sv, X,
        interaction_index=interaction_feat,
        ax=ax, show=False,
    )
    ax.set_title(
        f"SHAP dependence: {feat_x} × {interaction_feat}  |  FF_{sector} (h={horizon}m)",
        fontsize=11,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = REPORT_FIGURES / f"shap_dependence_{feat_x}_{interaction_feat}_{sector}_h{horizon}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_waterfall(
    shap_data: dict,
    sector: str,
    horizon: int,
    event_date: str,
    event_label: str,
    out_path=None,
) -> None:
    """SHAP waterfall for one (sector, horizon) prediction at a specific date."""
    entry = shap_data[(sector, horizon)]
    sv = entry["shap_values"]
    X = entry["X"]
    base = entry["base_value"]

    date = pd.Timestamp(event_date)
    if date not in X.index:
        raise KeyError(f"{event_date} missing from X (sector={sector}, horizon={horizon})")
    row_idx = X.index.get_loc(date)

    exp = shap.Explanation(
        values=sv[row_idx],
        base_values=base,
        data=X.iloc[row_idx].values,
        feature_names=X.columns.tolist(),
    )

    # shap.plots.waterfall creates its own figure; capture via gcf
    shap.plots.waterfall(exp, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.suptitle(
        f"Waterfall — FF_{sector} (h={horizon}m) prediction at {date.date()} ({event_label})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    if out_path is None:
        safe_label = event_label.replace(" ", "_").replace("/", "-")
        out_path = REPORT_FIGURES / f"shap_waterfall_{sector}_h{horizon}_{safe_label}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_cross_sector_heatmap(
    shap_data: dict,
    horizon: int = 6,
    top_k: int = 10,
    out_path=None,
) -> None:
    """12 sectors × top-k features, heatmap of mean signed SHAP.

    This is the Kilian & Park Figure 6 analogue: does the same oil-shock feature
    drive outperformance in some sectors and underperformance in others? Signed
    SHAP directly shows the cross-sectional rotation pattern.
    """
    # Feature names are identical across (s, horizon) cells
    feature_names = shap_data[(FF_INDUSTRIES[0], horizon)]["X"].columns.tolist()

    mean_shap = pd.DataFrame(index=FF_INDUSTRIES, columns=feature_names, dtype=float)
    mean_abs_shap = pd.DataFrame(index=FF_INDUSTRIES, columns=feature_names, dtype=float)

    for s in FF_INDUSTRIES:
        sv = shap_data[(s, horizon)]["shap_values"]
        mean_shap.loc[s] = sv.mean(axis=0)
        mean_abs_shap.loc[s] = np.abs(sv).mean(axis=0)

    agg_imp = mean_abs_shap.mean(axis=0).sort_values(ascending=False)
    top_features = agg_imp.head(top_k).index.tolist()
    hm = mean_shap[top_features].astype(float)

    fig, ax = plt.subplots(figsize=(13, 6))
    vmax = float(np.max(np.abs(hm.values)))
    sns.heatmap(
        hm,
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt="+.4f", annot_kws={"fontsize": 8},
        cbar_kws={"shrink": 0.7, "label": "Mean signed SHAP"},
        ax=ax,
    )
    ax.set_title(
        f"Cross-sector SHAP heatmap (h={horizon}m)  |  top-{top_k} features by aggregate |SHAP|",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Feature"); ax.set_ylabel("Sector")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()

    if out_path is None:
        out_path = REPORT_FIGURES / f"shap_heatmap_h{horizon}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    X = pd.read_parquet(FEATURES_DIR / "X_v1.parquet")
    Y = pd.read_parquet(TARGETS_DIR / "Y_v1.parquet")
    print(f"Loaded X {X.shape}, Y {Y.shape}")

    print("Computing SHAP values for all 48 regressors...")
    shap_data = compute_shap_all(X, Y)
    print(f"  Done. Saved {len(shap_data)} arrays to {SHAP_DIR}")

    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)

    print("Plotting global bar grid (h=6)...")
    plot_global_bar_grid(shap_data, horizon=6)

    print("Plotting dependence plots (eps_precaut × vix_regime)...")
    for s in ("Shops", "Enrgy"):
        plot_dependence(shap_data, s, horizon=6, feat_x="eps_precaut",
                        interaction_feat="vix_regime")

    print("Plotting event-month waterfalls for FF_Enrgy h=6...")
    for label, date in EVENT_DATES.items():
        plot_waterfall(shap_data, "Enrgy", horizon=6, event_date=date, event_label=label)

    print("Plotting cross-sector heatmap (h=6)...")
    plot_cross_sector_heatmap(shap_data, horizon=6, top_k=10)

    update_manifest(
        "milestone_d_shap",
        {
            "shap_dir": "outputs/shap/",
            "figures": [
                "outputs/report/figures/shap_global_bar_h6.png",
                "outputs/report/figures/shap_dependence_eps_precaut_vix_regime_Shops_h6.png",
                "outputs/report/figures/shap_dependence_eps_precaut_vix_regime_Enrgy_h6.png",
                "outputs/report/figures/shap_waterfall_Enrgy_h6_2008-09_GFC.png",
                "outputs/report/figures/shap_waterfall_Enrgy_h6_2014-11_SaudiSupply.png",
                "outputs/report/figures/shap_waterfall_Enrgy_h6_2020-03_COVID.png",
                "outputs/report/figures/shap_waterfall_Enrgy_h6_2022-02_Russia.png",
                "outputs/report/figures/shap_heatmap_h6.png",
            ],
            "n_shap_arrays": len(shap_data),
            "event_dates": EVENT_DATES,
        },
    )
    print("Milestone D SHAP analysis complete.")


if __name__ == "__main__":
    main()
