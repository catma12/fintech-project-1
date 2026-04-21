"""Diagnostic for Milestone C — why is OOF R² so negative?

Fits XGBRegressor on four representative (sector, horizon) cells with three
model-complexity configs and records per-fold train R², test R², train-test
gap, and directional accuracy. The overfitting signature is train R² >> test R²,
especially in the smallest (early) folds.

Also measures autocorrelation of CAR targets across horizons to quantify the
effective-sample-size shrinkage from overlapping target windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

from src.config import FEATURES_DIR, OUTPUTS, TARGETS_DIR, XGB_REG_PARAMS
from src.cv import walk_forward_splits


CELLS = [
    ("Hlth", 12),    # strongest edge cell from milestone C
    ("Enrgy", 6),    # theoretically most oil-shock-sensitive
    ("Manuf", 6),    # middle-of-pack OOF performance
    ("Shops", 1),    # consumer-demand channel, short horizon
]


def _config(**overrides) -> dict:
    p = dict(XGB_REG_PARAMS)
    p.update(overrides)
    return p


CONFIGS = {
    "current (d3, n300)": _config(),
    "shallow (d2, n100)": _config(max_depth=2, n_estimators=100, learning_rate=0.05),
    "stumps (d1, n50)":   _config(max_depth=1, n_estimators=50,  learning_rate=0.1),
}


def per_fold_report() -> pd.DataFrame:
    X = pd.read_parquet(FEATURES_DIR / "X_v1.parquet")
    Y = pd.read_parquet(TARGETS_DIR / "Y_v1.parquet")

    rows: list[dict] = []
    for sector, horizon in CELLS:
        y = Y[(sector, horizon)]
        mask = y.notna()
        X_c = X.loc[mask]
        y_c = y.loc[mask]
        splits = list(walk_forward_splits(horizon).split(X_c))

        for cfg_name, params in CONFIGS.items():
            for fold_i, (tr, te) in enumerate(splits):
                model = xgb.XGBRegressor(**params)
                model.fit(X_c.iloc[tr], y_c.iloc[tr], verbose=False)
                pred_tr = model.predict(X_c.iloc[tr])
                pred_te = model.predict(X_c.iloc[te])

                train_r2 = r2_score(y_c.iloc[tr], pred_tr)
                test_r2 = r2_score(y_c.iloc[te], pred_te)
                train_dir = float((np.sign(y_c.iloc[tr]) == np.sign(pred_tr)).mean())
                test_dir = float((np.sign(y_c.iloc[te]) == np.sign(pred_te)).mean())

                rows.append({
                    "sector": sector,
                    "horizon": horizon,
                    "config": cfg_name,
                    "fold": fold_i + 1,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "rows_per_feature": round(len(tr) / X_c.shape[1], 1),
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "gap": train_r2 - test_r2,
                    "train_dir_acc": train_dir,
                    "test_dir_acc": test_dir,
                })

    return pd.DataFrame(rows)


def target_autocorr_report() -> pd.DataFrame:
    """How autocorrelated are CAR targets? High autocorr = low effective N."""
    Y = pd.read_parquet(TARGETS_DIR / "Y_v1.parquet")
    rows: list[dict] = []
    sectors = Y.columns.get_level_values("sector").unique()
    horizons = Y.columns.get_level_values("horizon").unique()
    for h in horizons:
        for s in sectors:
            y = Y[(s, h)].dropna()
            rows.append({
                "horizon": h,
                "sector": s,
                "ac_1": y.autocorr(lag=1),
                "ac_6": y.autocorr(lag=6),
                "ac_12": y.autocorr(lag=12),
                "n": len(y),
            })
    df = pd.DataFrame(rows)
    # aggregate by horizon
    agg = df.groupby("horizon")[["ac_1", "ac_6", "ac_12"]].mean()
    return agg.round(3)


def main() -> None:
    print("=" * 70)
    print("Milestone C diagnostic — per-fold train vs test R²")
    print("=" * 70)

    df = per_fold_report()
    out = OUTPUTS / "diagnostics_cv.csv"
    df.to_csv(out, index=False)

    print("\n--- Mean across cells, by config ---")
    agg = df.groupby("config").agg(
        train_r2=("train_r2", "mean"),
        test_r2=("test_r2", "mean"),
        gap=("gap", "mean"),
        train_dir=("train_dir_acc", "mean"),
        test_dir=("test_dir_acc", "mean"),
    ).round(3)
    print(agg.to_string())

    print("\n--- By fold for current config, averaged across 4 cells ---")
    cur = df[df["config"] == "current (d3, n300)"]
    by_fold = cur.groupby("fold").agg(
        n_train=("n_train", "mean"),
        rows_per_feat=("rows_per_feature", "mean"),
        train_r2=("train_r2", "mean"),
        test_r2=("test_r2", "mean"),
        gap=("gap", "mean"),
        test_dir=("test_dir_acc", "mean"),
    ).round(3)
    print(by_fold.to_string())

    print("\n--- Per-cell detail, current config ---")
    for (s, h), g in cur.groupby(["sector", "horizon"]):
        print(f"\n  {s} h={h}:")
        print(g[["fold", "n_train", "train_r2", "test_r2", "gap", "test_dir_acc"]]
              .round(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("Target autocorrelation by horizon (mean across 12 sectors)")
    print("=" * 70)
    ac = target_autocorr_report()
    print(ac.to_string())

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
