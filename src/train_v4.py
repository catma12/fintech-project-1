"""v4 — momentum features across three training paradigms.

Adds the 5 momentum features (see src/features_momentum.py) to each of v1, v2,
and v3 architectures and rebuilds OOF comparisons:

  v4a — per-cell XGBoost (v1 + momentum). 48 models, 33+5 = 38 features.
  v4b — pooled XGB regression (v2 + momentum). 4 models, 18+12+5 = 35 features.
  v4c — pooled XGBRanker (v3 + momentum). 4 models, 18+12+5 = 35 features.

Head-to-head comparison tables written for each paradigm. Long-short cumulative
plot and sector-selection frequency updated for v4c.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tqdm import tqdm

from src.config import (
    FEATURES_DIR,
    FF_INDUSTRIES,
    HORIZONS,
    MODELS_DIR,
    OOF_DIR,
    REPORT_FIGURES,
    REPORT_TABLES,
    TARGETS_DIR,
    TRIMMED_FEATURES,
    XGB_POOLED_PARAMS,
    XGB_REG_PARAMS,
)
from src.cv import walk_forward_splits
from src.features_momentum import (
    MOMENTUM_COLS,
    attach_momentum_for_sector,
    attach_momentum_to_long,
)
from src.io_utils import load_panel, update_manifest
from src.train_ranking import (
    RANKER_PARAMS,
    _date_walk_forward,
    _groups_for,
    long_short_backtest,
    plot_cumulative_ls,
    plot_sector_selection_freq,
    ranking_metrics_per_month,
    summarize_backtest,
)


# --- v4a: per-cell XGBoost with momentum ------------------------------------

def train_v4a_per_cell(panel: pd.DataFrame, X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    """Re-run v1 paradigm with per-sector momentum features added."""
    oof_reg: dict[tuple[str, int], pd.Series] = {}
    metrics_rows: list[dict] = []

    cells = [(s, h) for s in FF_INDUSTRIES for h in HORIZONS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s, h in tqdm(cells, desc="v4a per-cell"):
            X_s = attach_momentum_for_sector(X, panel, s)
            y = Y[(s, h)].reindex(X_s.index)
            mask = y.notna()
            X_clean = X_s.loc[mask]
            y_clean = y.loc[mask]

            oof = pd.Series(index=y_clean.index, dtype=float)
            for tr, te in walk_forward_splits(h).split(X_clean):
                model = xgb.XGBRegressor(**XGB_REG_PARAMS)
                model.fit(X_clean.iloc[tr], y_clean.iloc[tr], verbose=False)
                oof.iloc[te] = model.predict(X_clean.iloc[te])

            valid = oof.notna()
            y_val = y_clean.loc[valid]
            oof_val = oof.loc[valid]
            dir_true = (y_val > 0).astype(int)
            dir_pred = (oof_val > 0).astype(int)
            base_rate = float(dir_true.mean())
            naive = max(base_rate, 1 - base_rate)

            oof_reg[(s, h)] = oof
            metrics_rows.append({
                "sector": s, "horizon": h, "task": "reg",
                "r2": float(r2_score(y_val, oof_val)),
                "rmse": float(np.sqrt(mean_squared_error(y_val, oof_val))),
                "mae": float(mean_absolute_error(y_val, oof_val)),
                "dir_acc": float(accuracy_score(dir_true, dir_pred)),
                "base_rate": base_rate, "naive_baseline": naive,
                "edge": float(accuracy_score(dir_true, dir_pred)) - naive,
                "n_oof": int(valid.sum()),
            })

    oof_df = pd.DataFrame(oof_reg)
    oof_df.columns = pd.MultiIndex.from_tuples(oof_df.columns, names=["sector", "horizon"])
    return {"oof": oof_df, "metrics": pd.DataFrame(metrics_rows)}


# --- v4b: pooled regression with momentum ------------------------------------

def _pool_with_momentum(X: pd.DataFrame, Y: pd.DataFrame, panel: pd.DataFrame, horizon: int):
    """Pool + sector one-hot + sector-specific momentum in one pass."""
    X_trim = X[TRIMMED_FEATURES]
    frames: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    for s in FF_INDUSTRIES:
        y = Y[(s, horizon)].reindex(X_trim.index)
        mask = y.notna()
        X_s = X_trim.loc[mask].copy()
        for other in FF_INDUSTRIES:
            X_s[f"is_{other}"] = 1 if other == s else 0
        X_s.index = pd.MultiIndex.from_product([X_s.index, [s]], names=["date", "sector"])
        y_s = y.loc[mask].copy()
        y_s.index = X_s.index
        frames.append(X_s); targets.append(y_s)
    X_long = pd.concat(frames).sort_index()
    y_long = pd.concat(targets).sort_index()
    X_long = attach_momentum_to_long(X_long, panel)
    return X_long, y_long


def train_v4b_pooled_reg(panel: pd.DataFrame, X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    all_oof: dict[tuple[str, int], pd.Series] = {}
    rows: list[dict] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for h in tqdm(HORIZONS, desc="v4b pooled reg"):
            X_long, y_long = _pool_with_momentum(X, Y, panel, h)
            oof = pd.Series(index=X_long.index, dtype=float)

            for tr_dates, te_dates in _date_walk_forward(
                X_long.index.get_level_values("date"), h
            ):
                n_val = max(3, int(0.15 * len(tr_dates)))
                tr_inner = tr_dates[:-n_val]
                val = tr_dates[-n_val:]
                dates = X_long.index.get_level_values("date")
                tr_mask = dates.isin(tr_inner)
                val_mask = dates.isin(val)
                te_mask = dates.isin(te_dates)

                model = xgb.XGBRegressor(**XGB_POOLED_PARAMS)
                model.fit(
                    X_long[tr_mask], y_long[tr_mask],
                    eval_set=[(X_long[val_mask], y_long[val_mask])],
                    verbose=False,
                )
                oof.loc[te_mask] = model.predict(X_long[te_mask])

            # Per-cell metrics
            valid = oof.notna()
            for s in FF_INDUSTRIES:
                m = valid & (oof.index.get_level_values("sector") == s)
                y_true = y_long.loc[m]
                y_pred = oof.loc[m]
                dir_true = (y_true > 0).astype(int)
                dir_pred = (y_pred > 0).astype(int)
                base_rate = float(dir_true.mean())
                naive = max(base_rate, 1 - base_rate)
                rows.append({
                    "sector": s, "horizon": h, "task": "pooled_reg",
                    "r2": float(r2_score(y_true, y_pred)) if len(y_true) else np.nan,
                    "dir_acc": float(accuracy_score(dir_true, dir_pred)),
                    "base_rate": base_rate, "naive_baseline": naive,
                    "edge": float(accuracy_score(dir_true, dir_pred)) - naive,
                    "n_oof": int(m.sum()),
                })
                pred_s = oof.loc[m]
                pred_s.index = pred_s.index.get_level_values("date")
                all_oof[(s, h)] = pred_s

    oof_df = pd.DataFrame(all_oof)
    oof_df.columns = pd.MultiIndex.from_tuples(oof_df.columns, names=["sector", "horizon"])
    return {"oof": oof_df, "metrics": pd.DataFrame(rows)}


# --- v4c: ranker with momentum ------------------------------------------------

def _plot_v3_vs_v4c(v4c_ls: dict[int, pd.DataFrame]) -> None:
    """Four-panel comparison of cumulative L-S returns: v3 (macro only) vs v4c (macro + momentum)."""
    v3_ls = pd.read_parquet(OOF_DIR / "long_short_returns_v3.parquet").set_index("date")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, h in zip(axes.flat, sorted(v4c_ls.keys())):
        v3_h = v3_ls[v3_ls["horizon"] == h]["ls_ret"].cumsum() if not v3_ls.empty else pd.Series(dtype=float)
        v4_h = v4c_ls[h]["ls_ret"].cumsum()
        if len(v3_h):
            ax.plot(v3_h.index, v3_h.values, color="#c94a3a", lw=1.0, label="v3 (macro only)")
        ax.plot(v4_h.index, v4_h.values, color="#1f3b66", lw=1.2, label="v4c (macro + momentum)")
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(f"h = {h}m", loc="left", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Cumulative long-short abnormal return — v3 vs v4c (adding momentum features)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(REPORT_FIGURES / "v3_vs_v4c_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def train_v4c_ranker(panel: pd.DataFrame, X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    scores_by_h: dict[int, pd.Series] = {}
    cars_by_h: dict[int, pd.Series] = {}
    ls_by_h: dict[int, pd.DataFrame] = {}
    backtest_summaries: list[dict] = []
    per_month_metrics: list[pd.DataFrame] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for h in tqdm(HORIZONS, desc="v4c ranker"):
            X_long, y_long_car = _pool_with_momentum(X, Y, panel, h)
            y_long_rank = (
                y_long_car.groupby(level="date").rank(method="first", ascending=True) - 1
            ).astype(int)

            oof_scores = pd.Series(index=X_long.index, dtype=float)
            for tr_dates, te_dates in _date_walk_forward(
                X_long.index.get_level_values("date"), h
            ):
                n_val = max(3, int(0.15 * len(tr_dates)))
                tr_inner = tr_dates[:-n_val]
                val = tr_dates[-n_val:]
                dates = X_long.index.get_level_values("date")
                tr_mask = dates.isin(tr_inner)
                val_mask = dates.isin(val)
                te_mask = dates.isin(te_dates)

                groups_tr = _groups_for(X_long[tr_mask])
                groups_val = _groups_for(X_long[val_mask])

                ranker = xgb.XGBRanker(**RANKER_PARAMS, early_stopping_rounds=30)
                ranker.fit(
                    X_long[tr_mask], y_long_rank[tr_mask],
                    group=groups_tr,
                    eval_set=[(X_long[val_mask], y_long_rank[val_mask])],
                    eval_group=[groups_val.tolist()],
                    verbose=False,
                )
                oof_scores.loc[te_mask] = ranker.predict(X_long[te_mask])

            ranker.save_model(MODELS_DIR / f"ranker_h{h}_v4c.json")
            scores_by_h[h] = oof_scores
            cars_by_h[h] = y_long_car

            mon = ranking_metrics_per_month(oof_scores, y_long_car)
            mon["horizon"] = h
            per_month_metrics.append(mon.reset_index())

            ls = long_short_backtest(oof_scores, y_long_car, top_k=3)
            ls_by_h[h] = ls
            summary = summarize_backtest(ls, h, label="v4c_ranker")
            summary["mean_spearman"] = float(mon["spearman"].mean())
            summary["mean_top3_hit"] = float(mon["top3_hit_rate"].mean())
            summary["mean_bot3_hit"] = float(mon["bot3_hit_rate"].mean())
            backtest_summaries.append(summary)

    return {
        "scores": scores_by_h,
        "cars": cars_by_h,
        "ls": ls_by_h,
        "summaries": pd.DataFrame(backtest_summaries),
        "per_month": pd.concat(per_month_metrics, ignore_index=True),
    }


# --- Main -------------------------------------------------------------------

def main() -> None:
    panel = load_panel()
    X = pd.read_parquet(FEATURES_DIR / "X_v1.parquet")
    Y = pd.read_parquet(TARGETS_DIR / "Y_v1.parquet")
    print(f"Loaded panel {panel.shape}, X {X.shape}, Y {Y.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)

    # --- v4a ---
    print("\n=== v4a: per-cell XGB + momentum ===")
    v4a = train_v4a_per_cell(panel, X, Y)
    v4a["oof"].to_parquet(OOF_DIR / "oof_reg_v4a.parquet")
    v4a["metrics"].to_csv(REPORT_TABLES / "oof_metrics_v4a.csv", index=False)
    m = v4a["metrics"]
    print(f"  mean R² = {m['r2'].mean():+.4f}  | mean edge = {m['edge'].mean():+.4f}"
          f"  | +edge cells: {(m['edge']>0).sum()}/48")

    # --- v4b ---
    print("\n=== v4b: pooled XGB regression + momentum ===")
    v4b = train_v4b_pooled_reg(panel, X, Y)
    v4b["oof"].to_parquet(OOF_DIR / "oof_pooled_xgb_v4b.parquet")
    v4b["metrics"].to_csv(REPORT_TABLES / "oof_metrics_v4b.csv", index=False)
    m = v4b["metrics"]
    print(f"  mean R² = {m['r2'].mean():+.4f}  | mean edge = {m['edge'].mean():+.4f}"
          f"  | +edge cells: {(m['edge']>0).sum()}/48")

    # --- v4c ---
    print("\n=== v4c: pooled XGBRanker + momentum ===")
    v4c = train_v4c_ranker(panel, X, Y)
    # Save artifacts
    ranker_scores = pd.DataFrame({h: s for h, s in v4c["scores"].items()})
    ranker_scores.columns = [f"h{h}" for h in ranker_scores.columns]
    ranker_scores.to_parquet(OOF_DIR / "ranker_scores_v4c.parquet")
    ls_long = pd.concat(
        [df.assign(horizon=h) for h, df in v4c["ls"].items()], axis=0
    ).reset_index()
    ls_long.to_parquet(OOF_DIR / "long_short_returns_v4c.parquet")
    v4c["summaries"].to_csv(REPORT_TABLES / "ranking_summary_v4c.csv", index=False)
    v4c["per_month"].to_csv(REPORT_TABLES / "ranking_per_month_v4c.csv", index=False)
    print(v4c["summaries"].round(4).to_string(index=False))

    plot_cumulative_ls(
        v4c["ls"],
        REPORT_FIGURES / "v4c_long_short_cumulative.png",
        title="v4c ranker (macro + momentum) — long top-3 / short bottom-3 each month",
    )
    plot_sector_selection_freq(v4c["ls"], REPORT_FIGURES / "v4c_sector_selection_freq.png")

    # Side-by-side v3 vs v4c cumulative
    _plot_v3_vs_v4c(v4c["ls"])

    # --- Head-to-head comparison ---
    print("\n=== Head-to-head: baseline vs v4 ===")
    comp_rows: list[dict] = []

    # v1 vs v4a (per-cell regression; v1 had a classifier for edge)
    v1 = pd.read_csv(REPORT_TABLES / "oof_metrics.csv")
    v1_clf = v1[v1["task"] == "clf"].copy()
    v1_clf["naive_baseline"] = v1_clf["base_rate"].combine(1 - v1_clf["base_rate"], max)
    v1_clf["edge"] = v1_clf["accuracy"] - v1_clf["naive_baseline"]
    m4a = v4a["metrics"]
    comp_rows.append({
        "paradigm": "per-cell",
        "baseline": "v1 clf",
        "baseline_mean_edge": float(v1_clf["edge"].mean()),
        "v4_mean_edge": float(m4a["edge"].mean()),
        "delta_edge": float(m4a["edge"].mean() - v1_clf["edge"].mean()),
        "v4_mean_r2": float(m4a["r2"].mean()),
    })

    # v2 vs v4b (pooled regression)
    v2 = pd.read_csv(REPORT_TABLES / "oof_metrics_v2.csv")
    v2_xgb = v2[v2["model"] == "xgb_pooled_v2"]
    m4b = v4b["metrics"]
    comp_rows.append({
        "paradigm": "pooled-reg",
        "baseline": "v2 xgb",
        "baseline_mean_edge": float(v2_xgb["edge"].mean()),
        "v4_mean_edge": float(m4b["edge"].mean()),
        "delta_edge": float(m4b["edge"].mean() - v2_xgb["edge"].mean()),
        "v4_mean_r2": float(m4b["r2"].mean()),
        "baseline_mean_r2": float(v2_xgb["r2"].mean()),
    })

    # v3 vs v4c (ranker)
    v3 = pd.read_csv(REPORT_TABLES / "ranking_summary_v3.csv")
    v4c_sum = v4c["summaries"]
    comp_rows.append({
        "paradigm": "ranker",
        "baseline": "v3 ranker",
        "baseline_mean_sharpe_ann": float(v3["sharpe_ann"].mean()),
        "v4_mean_sharpe_ann": float(v4c_sum["sharpe_ann"].mean()),
        "delta_sharpe": float(v4c_sum["sharpe_ann"].mean() - v3["sharpe_ann"].mean()),
        "baseline_mean_top3_hit": float(v3["mean_top3_hit"].mean()),
        "v4_mean_top3_hit": float(v4c_sum["mean_top3_hit"].mean()),
    })

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(REPORT_TABLES / "v1v2v3_vs_v4_comparison.csv", index=False)
    print(comp_df.round(4).to_string(index=False))

    update_manifest(
        "milestone_c_v4_momentum",
        {
            "v4a_oof": "outputs/oof/oof_reg_v4a.parquet",
            "v4a_metrics": "outputs/report/tables/oof_metrics_v4a.csv",
            "v4a_mean_r2": float(v4a["metrics"]["r2"].mean()),
            "v4a_mean_edge": float(v4a["metrics"]["edge"].mean()),
            "v4b_oof": "outputs/oof/oof_pooled_xgb_v4b.parquet",
            "v4b_metrics": "outputs/report/tables/oof_metrics_v4b.csv",
            "v4b_mean_r2": float(v4b["metrics"]["r2"].mean()),
            "v4b_mean_edge": float(v4b["metrics"]["edge"].mean()),
            "v4c_scores": "outputs/oof/ranker_scores_v4c.parquet",
            "v4c_summary": "outputs/report/tables/ranking_summary_v4c.csv",
            "v4c_mean_sharpe": float(v4c["summaries"]["sharpe_ann"].mean()),
            "v4c_mean_top3_hit": float(v4c["summaries"]["mean_top3_hit"].mean()),
            "comparison": "outputs/report/tables/v1v2v3_vs_v4_comparison.csv",
            "momentum_features": MOMENTUM_COLS,
        },
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
