"""Milestone C v3 — cross-sectional ranking model.

Sector rotation is a relative decision: "which sector outperforms which?" not
"what's the absolute return?". v1 and v2 both trained on absolute CARs and
hit a directional-signal ceiling — monthly return *magnitude* is hard to
predict, but relative *ordering* under the same macro conditions may carry
more structure.

v3 reframes the task as learning-to-rank:

  - For each month t at horizon h, rank the 12 Fama-French industries by
    their forward CAR(s, t, h). Highest CAR → rank 11, lowest → 0.
  - Train an XGBRanker (pairwise objective) with a group-aware walk-forward
    CV. Each month is one group of 12 rows; the ranker learns to order
    sectors within a group based on the shared macro features plus the
    sector identity one-hot.
  - Evaluate with the metrics that actually matter for sector rotation:
    Spearman rank correlation between predicted and actual CARs, top-3 /
    bottom-3 hit rate, and a long-short portfolio (long top-3, short
    bottom-3 each month) with annualized Sharpe and t-statistic.

If ranking signal exists where absolute prediction failed, the long-short
Sharpe will be well above 0 and the t-stat will be significant — that's the
actionable outcome.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from src.config import (
    FEATURES_DIR,
    FF_INDUSTRIES,
    HORIZONS,
    MODELS_DIR,
    OOF_DIR,
    REPORT_FIGURES,
    REPORT_TABLES,
    SEED,
    TARGETS_DIR,
    TRIMMED_FEATURES,
)
from src.io_utils import update_manifest


RANKER_PARAMS = dict(
    objective="rank:pairwise",
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=10,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=SEED,
    tree_method="hist",
    n_jobs=4,
)


# --- Data assembly ----------------------------------------------------------

def pool_and_rank(X: pd.DataFrame, Y: pd.DataFrame, horizon: int):
    """Return long-format (date, sector) X, rank target y_rank, raw CAR y_car.

    Ranks use average method so ties don't distort; 0 = worst, 11 = best.
    """
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
        frames.append(X_s)
        targets.append(y_s)

    X_long = pd.concat(frames).sort_index()
    y_car = pd.concat(targets).sort_index()
    # Integer ranks 0..11 (higher = better). method='first' deterministically
    # breaks ties; XGBRanker requires integer labels for rank:pairwise.
    y_rank = (
        y_car.groupby(level="date").rank(method="first", ascending=True) - 1
    ).astype(int)
    return X_long, y_rank, y_car


def _date_walk_forward(dates, horizon: int, n_splits: int = 5):
    gap = max(3, horizon + 1)
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    unique_dates = dates.unique().sort_values()
    for tr_idx, te_idx in splitter.split(unique_dates):
        yield unique_dates[tr_idx], unique_dates[te_idx]


def _groups_for(mask_or_df) -> np.ndarray:
    """Number of rows per date, ordered by date. XGBRanker expects this format."""
    if isinstance(mask_or_df, pd.DataFrame):
        return mask_or_df.groupby(level="date").size().values
    raise ValueError("expected DataFrame")


# --- Training ---------------------------------------------------------------

def train_pooled_ranker(X_long: pd.DataFrame, y_rank: pd.Series, horizon: int) -> dict:
    """Walk-forward XGBRanker with group-aware splits and date-held-out val."""
    oof_scores = pd.Series(index=X_long.index, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr_dates, te_dates in _date_walk_forward(
            X_long.index.get_level_values("date"), horizon
        ):
            # 15% of training dates (time-ordered tail) for early-stopping val
            n_val = max(3, int(0.15 * len(tr_dates)))
            tr_inner_dates = tr_dates[:-n_val]
            val_dates = tr_dates[-n_val:]

            date_level = X_long.index.get_level_values("date")
            tr_mask = date_level.isin(tr_inner_dates)
            val_mask = date_level.isin(val_dates)
            te_mask = date_level.isin(te_dates)

            X_tr = X_long[tr_mask]
            X_val = X_long[val_mask]
            X_te = X_long[te_mask]

            groups_tr = _groups_for(X_tr)
            groups_val = _groups_for(X_val)

            ranker = xgb.XGBRanker(
                **RANKER_PARAMS,
                early_stopping_rounds=30,
            )
            ranker.fit(
                X_tr, y_rank[tr_mask],
                group=groups_tr,
                eval_set=[(X_val, y_rank[val_mask])],
                eval_group=[groups_val.tolist()],
                verbose=False,
            )
            oof_scores.loc[te_mask] = ranker.predict(X_te)

    # Final model refit on all data, no early stopping
    final_params = {k: v for k, v in RANKER_PARAMS.items()}
    final_params["n_estimators"] = 200  # fixed — conservative without val set
    final_ranker = xgb.XGBRanker(**final_params)
    final_ranker.fit(X_long, y_rank, group=_groups_for(X_long))

    return {"oof_scores": oof_scores, "model": final_ranker}


# --- Metrics + backtest -----------------------------------------------------

def ranking_metrics_per_month(scores: pd.Series, cars: pd.Series) -> pd.DataFrame:
    """Per-month Spearman correlation and top-3 / bottom-3 hit rates."""
    valid = scores.notna()
    df = pd.DataFrame({"score": scores[valid], "car": cars[valid]})
    scores_wide = df["score"].unstack("sector")
    cars_wide = df["car"].unstack("sector")

    rows: list[dict] = []
    for date in scores_wide.index:
        pred = scores_wide.loc[date]
        actual = cars_wide.loc[date]
        if pred.isna().any() or actual.isna().any():
            continue
        rho, _ = spearmanr(pred.values, actual.values)
        top3_pred = set(pred.nlargest(3).index)
        top3_actual = set(actual.nlargest(3).index)
        bot3_pred = set(pred.nsmallest(3).index)
        bot3_actual = set(actual.nsmallest(3).index)
        rows.append({
            "date": date,
            "spearman": rho,
            "top3_hit_rate": len(top3_pred & top3_actual) / 3,
            "bot3_hit_rate": len(bot3_pred & bot3_actual) / 3,
        })
    return pd.DataFrame(rows).set_index("date")


def long_short_backtest(scores: pd.Series, cars: pd.Series, top_k: int = 3) -> pd.DataFrame:
    """Long top-k predicted sectors, short bottom-k predicted sectors.
    Return the monthly long-short abnormal return (already net of market, since
    CARs are abnormal returns)."""
    valid = scores.notna()
    df = pd.DataFrame({"score": scores[valid], "car": cars[valid]})
    scores_wide = df["score"].unstack("sector")
    cars_wide = df["car"].unstack("sector")

    rows: list[dict] = []
    for date in scores_wide.index:
        pred = scores_wide.loc[date]
        actual = cars_wide.loc[date]
        if pred.isna().any() or actual.isna().any():
            continue
        long_sectors = pred.nlargest(top_k).index
        short_sectors = pred.nsmallest(top_k).index
        long_ret = float(actual[long_sectors].mean())
        short_ret = float(actual[short_sectors].mean())
        rows.append({
            "date": date,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": long_ret - short_ret,
            "long_sectors": ",".join(long_sectors),
            "short_sectors": ",".join(short_sectors),
        })
    return pd.DataFrame(rows).set_index("date")


def summarize_backtest(ls: pd.DataFrame, horizon: int, label: str = "v3") -> dict:
    """Mean, std, Sharpe (annualized), t-stat, hit rate."""
    returns = ls["ls_ret"].dropna()
    n = len(returns)
    if n < 5:
        return {"n": n, "mean": np.nan, "std": np.nan, "sharpe_ann": np.nan, "t_stat": np.nan, "hit_rate": np.nan}
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    # Annualize the Sharpe assuming monthly returns — but note for h>1 these are
    # overlapping h-month returns, so the sqrt(12) annualization is approximate.
    # For h=1 it's exact.
    sharpe_monthly = mean / std if std > 0 else 0
    sharpe_ann = sharpe_monthly * np.sqrt(12 / horizon) if horizon > 0 else sharpe_monthly
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else 0
    hit_rate = float((returns > 0).mean())
    return {
        "horizon": horizon,
        "label": label,
        "n_months": n,
        "mean_ls_ret": mean,
        "std_ls_ret": std,
        "sharpe_ann": float(sharpe_ann),
        "t_stat": float(t_stat),
        "hit_rate": hit_rate,
    }


# --- Plotting ----------------------------------------------------------------

def plot_cumulative_ls(
    all_ls: dict[int, pd.DataFrame],
    out_path: Path,
    title: str = "Cross-sectional ranker — long top-3 / short bottom-3 each month",
) -> None:
    """Cumulative L-S strategy return over time for each horizon."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, h in zip(axes.flat, sorted(all_ls.keys())):
        ls = all_ls[h]
        if ls.empty:
            continue
        cum = ls["ls_ret"].cumsum()
        ax.plot(cum.index, cum.values, color="#1f3b66", lw=1.1, label="L-S cumulative")
        ax.axhline(0, color="black", lw=0.4)
        ax.fill_between(cum.index, 0, cum.values, alpha=0.2, color="#1f3b66")
        ax.set_title(f"h = {h}m  |  cumulative long-short abnormal return", fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_sector_selection_freq(
    all_ls: dict[int, pd.DataFrame], out_path: Path
) -> None:
    """How often does each sector appear in the model's top-3 and bottom-3 picks?"""
    fig, axes = plt.subplots(1, len(all_ls), figsize=(16, 4), sharey=True)
    if len(all_ls) == 1:
        axes = [axes]

    for ax, h in zip(axes, sorted(all_ls.keys())):
        ls = all_ls[h]
        if ls.empty:
            continue
        n = len(ls)
        counts_long = pd.Series(
            {s: ls["long_sectors"].str.contains(s).sum() for s in FF_INDUSTRIES}
        ) / n
        counts_short = pd.Series(
            {s: ls["short_sectors"].str.contains(s).sum() for s in FF_INDUSTRIES}
        ) / n
        order = (counts_long - counts_short).sort_values(ascending=True).index
        x = np.arange(len(FF_INDUSTRIES))
        ax.barh(x, counts_long[order], color="#2e8b57", alpha=0.8, label="long")
        ax.barh(x, -counts_short[order], color="#c94a3a", alpha=0.8, label="short")
        ax.set_yticks(x)
        ax.set_yticklabels(order, fontsize=8)
        ax.axvline(0, color="black", lw=0.4)
        ax.axvline(3 / 12, color="gray", lw=0.4, ls="--", alpha=0.5)
        ax.axvline(-3 / 12, color="gray", lw=0.4, ls="--", alpha=0.5)
        ax.set_xlabel("Selection frequency")
        ax.set_title(f"h={h}m", fontsize=10)
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Sector selection frequency (dashed line = 3/12 chance-baseline)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --- Main --------------------------------------------------------------------

def main() -> None:
    X = pd.read_parquet(FEATURES_DIR / "X_v1.parquet")
    Y = pd.read_parquet(TARGETS_DIR / "Y_v1.parquet")
    print(f"Loaded X {X.shape}, Y {Y.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)

    all_oof_scores: dict[int, pd.Series] = {}
    all_metrics_frames: list[pd.DataFrame] = []
    all_ls: dict[int, pd.DataFrame] = {}
    backtest_summaries: list[dict] = []

    for h in tqdm(HORIZONS, desc="Ranking horizons"):
        X_long, y_rank, y_car = pool_and_rank(X, Y, h)

        res = train_pooled_ranker(X_long, y_rank, h)
        res["model"].save_model(MODELS_DIR / f"ranker_h{h}_v3.json")

        all_oof_scores[h] = res["oof_scores"]

        # Per-month metrics
        mon = ranking_metrics_per_month(res["oof_scores"], y_car)
        mon["horizon"] = h
        all_metrics_frames.append(mon.reset_index())

        # Long-short backtest
        ls = long_short_backtest(res["oof_scores"], y_car, top_k=3)
        all_ls[h] = ls

        summary = summarize_backtest(ls, h, label="v3_ranker")
        summary["mean_spearman"] = float(mon["spearman"].mean())
        summary["median_spearman"] = float(mon["spearman"].median())
        summary["mean_top3_hit"] = float(mon["top3_hit_rate"].mean())
        summary["mean_bot3_hit"] = float(mon["bot3_hit_rate"].mean())
        backtest_summaries.append(summary)

        print(
            f"\n[h={h}]  Spearman {summary['mean_spearman']:+.4f} | "
            f"top3 hit {summary['mean_top3_hit']:.3f} (chance 0.250) | "
            f"bot3 hit {summary['mean_bot3_hit']:.3f} (chance 0.250)"
        )
        print(
            f"       L-S monthly mean {summary['mean_ls_ret']:+.4f} | "
            f"Sharpe (ann, horizon-adj) {summary['sharpe_ann']:+.3f} | "
            f"t-stat {summary['t_stat']:+.2f} | hit rate {summary['hit_rate']:.3f}"
        )

    # Save artifacts
    scores_df = pd.DataFrame({h: s for h, s in all_oof_scores.items()})
    scores_df.columns = [f"h{h}" for h in scores_df.columns]
    scores_df.to_parquet(OOF_DIR / "ranker_scores_v3.parquet")

    ls_long = pd.concat(
        [df.assign(horizon=h) for h, df in all_ls.items()], axis=0
    ).reset_index()
    ls_long.to_parquet(OOF_DIR / "long_short_returns_v3.parquet")

    metrics_df = pd.concat(all_metrics_frames, ignore_index=True)
    metrics_df.to_csv(REPORT_TABLES / "ranking_metrics_per_month_v3.csv", index=False)

    summary_df = pd.DataFrame(backtest_summaries)
    summary_df.to_csv(REPORT_TABLES / "ranking_summary_v3.csv", index=False)
    print("\n=== v3 Ranking summary ===")
    print(summary_df.round(4).to_string(index=False))

    # Plots
    plot_cumulative_ls(all_ls, REPORT_FIGURES / "v3_long_short_cumulative.png")
    plot_sector_selection_freq(all_ls, REPORT_FIGURES / "v3_sector_selection_freq.png")
    print(f"\nPlots: {REPORT_FIGURES}/v3_*.png")

    update_manifest(
        "milestone_c_v3_ranking",
        {
            "oof_scores": "outputs/oof/ranker_scores_v3.parquet",
            "long_short_returns": "outputs/oof/long_short_returns_v3.parquet",
            "per_month_metrics": "outputs/report/tables/ranking_metrics_per_month_v3.csv",
            "summary": "outputs/report/tables/ranking_summary_v3.csv",
            "figures": [
                "outputs/report/figures/v3_long_short_cumulative.png",
                "outputs/report/figures/v3_sector_selection_freq.png",
            ],
            "ranker_params": {k: v for k, v in RANKER_PARAMS.items()},
            "summary_by_horizon": {
                str(r["horizon"]): {
                    "sharpe_ann": r["sharpe_ann"],
                    "t_stat": r["t_stat"],
                    "mean_spearman": r["mean_spearman"],
                    "top3_hit_rate": r["mean_top3_hit"],
                    "hit_rate": r["hit_rate"],
                }
                for r in backtest_summaries
            },
        },
    )


if __name__ == "__main__":
    main()
