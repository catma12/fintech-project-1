"""Milestone C v2 — pooled XGBoost + Ridge baseline.

Diagnostic evidence (outputs/diagnostics_cv.csv):
  - Mean train-vs-test R² gap = 0.74 in v1 (severe overfitting)
  - Shallower trees cut gap sharply (d1: 0.37, d2: 0.54, d3: 0.74)
  - Long horizons (h=6, h=12) hit a target-autocorrelation ceiling
    (lag-1 AC = 0.84 and 0.92) that structurally limits OOF R²
  - Per-sector models waste data: ~450 rows each with 33 features

v2 response:

  Pool 12 sectors into one XGBoost per horizon.
    Input features = 18 trimmed macro/shock/oil features + 12 sector one-hots.
    Training rows per horizon: ~450 × 12 ≈ 5,400, a 12× increase.

  Feature trim: drop eps_*_lag2, eps_*_lag3, eps_*_cum3m, dom_is_*,
    shock_sign, shock_magnitude, vix_is_proxy. Trees can recover lag-2/3 and
    cumulative effects from the continuous shock series if the signal is there.

  Early stopping: 15% of each training fold held out as a time-ordered val
    set; training stops at val-loss plateau (patience=30 rounds).

  Ridge baseline: linear regression on standardized features as a floor.
    If XGBoost can't beat Ridge, nonlinearity isn't earning its keep.

Walk-forward CV operates on DATES (not stacked rows) to prevent a given
month from appearing in train for sector A while simultaneously in test
for sector B.

v1 artifacts are untouched; v2 writes to separate files so cells can be
compared one-to-one.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import (
    FEATURES_DIR,
    FF_INDUSTRIES,
    HORIZONS,
    MODELS_DIR,
    OOF_DIR,
    REPORT_TABLES,
    SEED,
    TARGETS_DIR,
    TRIMMED_FEATURES,
    XGB_POOLED_PARAMS,
)
from src.io_utils import update_manifest


# --- Pooling -----------------------------------------------------------------

def pool_data(X: pd.DataFrame, Y: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Stack (X, Y[:, :, h]) into a long (date × sector) frame with sector one-hots.

    Returns
    -------
    X_long : pd.DataFrame
        MultiIndex rows (date, sector), columns = TRIMMED_FEATURES + sector one-hots.
    y_long : pd.Series
        Aligned CAR target.
    """
    X_trim = X[TRIMMED_FEATURES]

    frames: list[pd.DataFrame] = []
    targets: list[pd.Series] = []

    for s in FF_INDUSTRIES:
        # Reindex Y to X's row set first (X starts 1988-01 after VAR burn-in);
        # then filter rows where the CAR target is observable (last h rows NaN).
        y = Y[(s, horizon)].reindex(X_trim.index)
        mask = y.notna()
        X_s = X_trim.loc[mask].copy()
        y_s = y.loc[mask].copy()

        # Add sector one-hot
        for other in FF_INDUSTRIES:
            X_s[f"is_{other}"] = 1 if other == s else 0

        # MultiIndex: (date, sector)
        X_s.index = pd.MultiIndex.from_product([X_s.index, [s]], names=["date", "sector"])
        y_s.index = X_s.index

        frames.append(X_s)
        targets.append(y_s)

    X_long = pd.concat(frames).sort_index()
    y_long = pd.concat(targets).sort_index()
    return X_long, y_long


def _date_walk_forward(dates: pd.Index, horizon: int, n_splits: int = 5):
    """TimeSeriesSplit yielding (train_dates, test_dates) with horizon-aware gap."""
    gap = max(3, horizon + 1)
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    unique_dates = dates.unique().sort_values()
    for tr_idx, te_idx in splitter.split(unique_dates):
        yield unique_dates[tr_idx], unique_dates[te_idx]


def _split_rows_by_dates(X_long: pd.DataFrame, dates) -> np.ndarray:
    """Boolean mask selecting rows of X_long whose date level is in `dates`."""
    row_dates = X_long.index.get_level_values("date")
    return row_dates.isin(dates)


# --- XGBoost pooled ----------------------------------------------------------

def train_pooled_xgb(X_long: pd.DataFrame, y_long: pd.Series, horizon: int) -> dict:
    """Pooled XGBoost across 12 sectors for one horizon.

    Within each walk-forward fold, the last 15% of training dates are carved
    out as a validation set for early stopping.
    """
    oof = pd.Series(index=X_long.index, dtype=float)

    for tr_dates, te_dates in _date_walk_forward(X_long.index.get_level_values("date"), horizon):
        # Carve off last 15% of training dates as val (time-ordered)
        n_val = max(3, int(0.15 * len(tr_dates)))
        tr_inner_dates = tr_dates[:-n_val]
        val_dates = tr_dates[-n_val:]

        tr_mask = _split_rows_by_dates(X_long, tr_inner_dates)
        val_mask = _split_rows_by_dates(X_long, val_dates)
        te_mask = _split_rows_by_dates(X_long, te_dates)

        model = xgb.XGBRegressor(**XGB_POOLED_PARAMS)
        model.fit(
            X_long[tr_mask], y_long[tr_mask],
            eval_set=[(X_long[val_mask], y_long[val_mask])],
            verbose=False,
        )
        oof.loc[te_mask] = model.predict(X_long[te_mask])

    # Final model refit on all data with fixed best_iteration (median across folds)
    # Simpler: refit with a reasonable fixed n_estimators using late-fold's best
    final_model = xgb.XGBRegressor(**{**XGB_POOLED_PARAMS, "early_stopping_rounds": None})
    final_model.fit(X_long, y_long, verbose=False)

    return {"oof": oof, "model": final_model}


# --- Ridge baseline ----------------------------------------------------------

def train_pooled_classifier(
    X_long: pd.DataFrame, y_long: pd.Series, horizon: int
) -> dict:
    """Pooled XGBClassifier on binary sign target. Mirrors train_pooled_xgb but
    with log-loss so the model directly optimizes directional accuracy."""
    y_bin = (y_long > 0).astype(int)
    oof = pd.Series(index=X_long.index, dtype=float)

    for tr_dates, te_dates in _date_walk_forward(X_long.index.get_level_values("date"), horizon):
        n_val = max(3, int(0.15 * len(tr_dates)))
        tr_inner_dates = tr_dates[:-n_val]
        val_dates = tr_dates[-n_val:]

        tr_mask = _split_rows_by_dates(X_long, tr_inner_dates)
        val_mask = _split_rows_by_dates(X_long, val_dates)
        te_mask = _split_rows_by_dates(X_long, te_dates)

        params = {**XGB_POOLED_PARAMS, "eval_metric": "logloss"}
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_long[tr_mask], y_bin[tr_mask],
            eval_set=[(X_long[val_mask], y_bin[val_mask])],
            verbose=False,
        )
        oof.loc[te_mask] = model.predict(X_long[te_mask])

    final_model = xgb.XGBClassifier(
        **{**XGB_POOLED_PARAMS, "early_stopping_rounds": None, "eval_metric": "logloss"}
    )
    final_model.fit(X_long, y_bin, verbose=False)
    return {"oof": oof, "model": final_model}


def train_pooled_ridge(X_long: pd.DataFrame, y_long: pd.Series, horizon: int) -> dict:
    """Standardized Ridge with alpha selected via built-in CV inside each fold.

    HY spread and hy_available are dropped because Ridge can't handle NaN; IG
    spread carries the credit signal. NaN in other features (first 0-3 rows of
    lags) is median-imputed per-fold.
    """
    drop_cols = ["credit_spread_hy", "hy_available"]
    X_r = X_long.drop(columns=drop_cols)
    oof = pd.Series(index=X_r.index, dtype=float)

    for tr_dates, te_dates in _date_walk_forward(X_r.index.get_level_values("date"), horizon):
        tr_mask = _split_rows_by_dates(X_r, tr_dates)
        te_mask = _split_rows_by_dates(X_r, te_dates)

        X_tr = X_r[tr_mask].copy()
        X_te = X_r[te_mask].copy()
        y_tr = y_long[tr_mask]

        # Median-impute remaining NaN (lag rows at sample start)
        medians = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(medians)
        X_te = X_te.fillna(medians)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
        ridge.fit(X_tr_s, y_tr)
        oof.loc[te_mask] = ridge.predict(X_te_s)

    # Final model on all data
    X_all = X_r.fillna(X_r.median(numeric_only=True))
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)
    final_ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
    final_ridge.fit(X_all_s, y_long)

    return {"oof": oof, "model": (final_ridge, scaler, drop_cols)}


# --- Metrics -----------------------------------------------------------------

def score_per_cell(oof: pd.Series, y_long: pd.Series, horizon: int) -> pd.DataFrame:
    """Split pooled OOF predictions back into per-(sector, horizon) metrics."""
    rows: list[dict] = []
    valid = oof.notna()

    for s in FF_INDUSTRIES:
        mask = valid & (oof.index.get_level_values("sector") == s)
        y_true = y_long.loc[mask]
        y_pred = oof.loc[mask]
        if len(y_true) == 0:
            continue

        dir_true = (y_true > 0).astype(int)
        dir_pred = (y_pred > 0).astype(int)
        base_rate = float(dir_true.mean())
        naive_baseline = max(base_rate, 1 - base_rate)

        rows.append({
            "sector": s,
            "horizon": horizon,
            "n_oof": int(len(y_true)),
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "dir_acc": float(accuracy_score(dir_true, dir_pred)),
            "base_rate": base_rate,
            "naive_baseline": naive_baseline,
            "edge": float(accuracy_score(dir_true, dir_pred)) - naive_baseline,
        })
    return pd.DataFrame(rows)


# --- Main --------------------------------------------------------------------

def main() -> None:
    X = pd.read_parquet(FEATURES_DIR / "X_v1.parquet")
    Y = pd.read_parquet(TARGETS_DIR / "Y_v1.parquet")
    print(f"Loaded X {X.shape}, Y {Y.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TABLES.mkdir(parents=True, exist_ok=True)

    all_oof_xgb: dict[tuple[str, int], pd.Series] = {}
    all_oof_clf: dict[tuple[str, int], pd.Series] = {}
    all_oof_ridge: dict[tuple[str, int], pd.Series] = {}
    metrics_rows: list[dict] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for h in tqdm(HORIZONS, desc="Horizons"):
            X_long, y_long = pool_data(X, Y, h)

            # --- Pooled XGBoost ---
            res_xgb = train_pooled_xgb(X_long, y_long, h)
            res_xgb["model"].save_model(MODELS_DIR / f"pooled_h{h}_xgb_v2.json")

            # Split OOF by sector
            for s in FF_INDUSTRIES:
                mask = res_xgb["oof"].index.get_level_values("sector") == s
                pred_s = res_xgb["oof"].loc[mask]
                pred_s.index = pred_s.index.get_level_values("date")
                all_oof_xgb[(s, h)] = pred_s

            m_xgb = score_per_cell(res_xgb["oof"], y_long, h).assign(model="xgb_pooled_v2")
            metrics_rows.append(m_xgb)

            # --- Pooled XGBClassifier (direct direction learning) ---
            res_clf = train_pooled_classifier(X_long, y_long, h)
            res_clf["model"].save_model(MODELS_DIR / f"pooled_h{h}_clf_v2.json")
            for s in FF_INDUSTRIES:
                mask = res_clf["oof"].index.get_level_values("sector") == s
                pred_s = res_clf["oof"].loc[mask]
                pred_s.index = pred_s.index.get_level_values("date")
                all_oof_clf[(s, h)] = pred_s

            # For classifier, score via direct prediction == target
            clf_rows = []
            valid = res_clf["oof"].notna()
            for s in FF_INDUSTRIES:
                sector_mask = valid & (res_clf["oof"].index.get_level_values("sector") == s)
                y_true_bin = (y_long.loc[sector_mask] > 0).astype(int)
                y_pred_bin = res_clf["oof"].loc[sector_mask].astype(int)
                base_rate = float(y_true_bin.mean())
                naive_baseline = max(base_rate, 1 - base_rate)
                acc = float((y_true_bin == y_pred_bin).mean())
                clf_rows.append({
                    "sector": s, "horizon": h, "n_oof": int(sector_mask.sum()),
                    "r2": np.nan, "rmse": np.nan, "mae": np.nan,
                    "dir_acc": acc,
                    "base_rate": base_rate, "naive_baseline": naive_baseline,
                    "edge": acc - naive_baseline,
                })
            m_clf = pd.DataFrame(clf_rows).assign(model="clf_pooled_v2")
            metrics_rows.append(m_clf)

            # --- Ridge baseline ---
            res_ridge = train_pooled_ridge(X_long, y_long, h)
            with open(MODELS_DIR / f"pooled_h{h}_ridge_v2.pkl", "wb") as f:
                pickle.dump(res_ridge["model"], f)

            for s in FF_INDUSTRIES:
                mask = res_ridge["oof"].index.get_level_values("sector") == s
                pred_s = res_ridge["oof"].loc[mask]
                pred_s.index = pred_s.index.get_level_values("date")
                all_oof_ridge[(s, h)] = pred_s

            m_rid = score_per_cell(res_ridge["oof"], y_long, h).assign(model="ridge_pooled_v2")
            metrics_rows.append(m_rid)

    # Assemble OOF DataFrames
    def _assemble(d: dict[tuple[str, int], pd.Series]) -> pd.DataFrame:
        df = pd.DataFrame(d)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["sector", "horizon"])
        return df

    oof_xgb_df = _assemble(all_oof_xgb)
    oof_clf_df = _assemble(all_oof_clf)
    oof_rid_df = _assemble(all_oof_ridge)
    oof_xgb_df.to_parquet(OOF_DIR / "oof_pooled_xgb_v2.parquet")
    oof_clf_df.to_parquet(OOF_DIR / "oof_pooled_clf_v2.parquet")
    oof_rid_df.to_parquet(OOF_DIR / "oof_pooled_ridge_v2.parquet")

    metrics_df = pd.concat(metrics_rows, ignore_index=True)
    metrics_df.to_csv(REPORT_TABLES / "oof_metrics_v2.csv", index=False)
    print(f"\nWrote v2 metrics to {REPORT_TABLES / 'oof_metrics_v2.csv'}")

    # Print summary
    print("\n=== v2 pooled XGBoost regression — mean metrics by horizon ===")
    xgb_m = metrics_df[metrics_df["model"] == "xgb_pooled_v2"]
    print(xgb_m.groupby("horizon")[["r2", "dir_acc", "edge"]].mean().round(4).to_string())

    print("\n=== v2 pooled XGBoost classifier — mean metrics by horizon ===")
    clf_m = metrics_df[metrics_df["model"] == "clf_pooled_v2"]
    print(clf_m.groupby("horizon")[["dir_acc", "edge"]].mean().round(4).to_string())

    print("\n=== v2 pooled Ridge — mean metrics by horizon ===")
    rid_m = metrics_df[metrics_df["model"] == "ridge_pooled_v2"]
    print(rid_m.groupby("horizon")[["r2", "dir_acc", "edge"]].mean().round(4).to_string())

    print("\n=== Head-to-head vs v1 (classification edge) ===")
    v1 = pd.read_csv(REPORT_TABLES / "oof_metrics.csv")
    v1_clf = v1[v1["task"] == "clf"].copy()
    v1_clf["naive_baseline"] = v1_clf["base_rate"].combine(1 - v1_clf["base_rate"], max)
    v1_clf["v1_edge"] = v1_clf["accuracy"] - v1_clf["naive_baseline"]

    v2_clf_edge = clf_m[["sector", "horizon", "edge", "dir_acc"]].rename(
        columns={"edge": "v2_clf_edge", "dir_acc": "v2_clf_acc"}
    )
    v2_xgb_edge = xgb_m[["sector", "horizon", "edge", "dir_acc", "r2"]].rename(
        columns={"edge": "v2_xgb_edge", "dir_acc": "v2_xgb_acc", "r2": "v2_xgb_r2"}
    )
    comp = (
        v1_clf[["sector", "horizon", "v1_edge", "accuracy"]]
        .rename(columns={"accuracy": "v1_acc"})
        .merge(v2_xgb_edge, on=["sector", "horizon"])
        .merge(v2_clf_edge, on=["sector", "horizon"])
    )
    comp["xgb_reg_delta_edge"] = comp["v2_xgb_edge"] - comp["v1_edge"]
    comp["clf_delta_edge"] = comp["v2_clf_edge"] - comp["v1_edge"]

    print(f"Mean v1 edge          : {comp['v1_edge'].mean():+.4f}")
    print(f"Mean v2 XGB-reg edge  : {comp['v2_xgb_edge'].mean():+.4f}  (delta {comp['xgb_reg_delta_edge'].mean():+.4f})")
    print(f"Mean v2 XGB-clf edge  : {comp['v2_clf_edge'].mean():+.4f}  (delta {comp['clf_delta_edge'].mean():+.4f})")
    print(f"Cells improved (vs v1): XGB-reg {(comp['xgb_reg_delta_edge'] > 0).sum()}/48  "
          f"|  XGB-clf {(comp['clf_delta_edge'] > 0).sum()}/48")

    # Write comparison
    comp_cols = ["sector", "horizon", "v1_edge", "v1_acc",
                 "v2_xgb_edge", "v2_xgb_acc", "v2_xgb_r2",
                 "v2_clf_edge", "v2_clf_acc",
                 "xgb_reg_delta_edge", "clf_delta_edge"]
    comp[comp_cols].to_csv(REPORT_TABLES / "v1_v2_comparison.csv", index=False)
    print(f"Wrote comparison to {REPORT_TABLES / 'v1_v2_comparison.csv'}")

    update_manifest(
        "milestone_c_v2",
        {
            "oof_xgb": "outputs/oof/oof_pooled_xgb_v2.parquet",
            "oof_clf": "outputs/oof/oof_pooled_clf_v2.parquet",
            "oof_ridge": "outputs/oof/oof_pooled_ridge_v2.parquet",
            "metrics": "outputs/report/tables/oof_metrics_v2.csv",
            "comparison": "outputs/report/tables/v1_v2_comparison.csv",
            "xgb_mean_r2": float(xgb_m["r2"].mean()),
            "xgb_mean_edge": float(xgb_m["edge"].mean()),
            "clf_mean_edge": float(clf_m["edge"].mean()),
            "ridge_mean_r2": float(rid_m["r2"].mean()),
            "ridge_mean_edge": float(rid_m["edge"].mean()),
            "v1_mean_edge": float(comp["v1_edge"].mean()),
            "v2_xgb_reg_mean_edge": float(comp["v2_xgb_edge"].mean()),
            "v2_xgb_clf_mean_edge": float(comp["v2_clf_edge"].mean()),
            "pooled_xgb_params": {k: v for k, v in XGB_POOLED_PARAMS.items()
                                   if k != "early_stopping_rounds"},
            "trimmed_features": TRIMMED_FEATURES,
        },
    )


if __name__ == "__main__":
    main()
