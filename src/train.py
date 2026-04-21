"""Milestone C — walk-forward XGBoost training.

For each of 12 Fama-French industries x 4 horizons (48 cells), we train an
XGBRegressor on CAR targets and an XGBClassifier on binary sign targets. CV
uses horizon-aware TimeSeriesSplit with gap=max(3, h+1) so h=12 targets do not
leak across fold boundaries.

Out-of-fold predictions are the only honest signal: we never report in-sample
metrics. Final models are retrained on all available data (for later 2026
inference) and persisted alongside OOF predictions and the metrics table.

Hyperparameters are fixed per the implementation guide's small-sample advice:
shallow trees (max_depth=3), strong regularization (min_child_weight=5,
reg_alpha=0.5, reg_lambda=1.0). With ~450 rows across 48 targets, grid search
risks overfitting to CV folds — we prefer fixed, defensible defaults.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tqdm import tqdm
import xgboost as xgb

from src.config import (
    FEATURES_DIR,
    FF_INDUSTRIES,
    HORIZONS,
    MODELS_DIR,
    OOF_DIR,
    REPORT_TABLES,
    TARGETS_DIR,
    XGB_CLF_PARAMS,
    XGB_REG_PARAMS,
)
from src.cv import walk_forward_splits
from src.io_utils import update_manifest


def _binarize(y: pd.Series) -> pd.Series:
    """Map continuous CAR to binary: 1 if CAR > 0 (outperformance), else 0."""
    return (y > 0).astype(int)


def train_one(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    task: str = "reg",
) -> dict:
    """Train one (sector, horizon) model with walk-forward CV.

    Returns
    -------
    dict with keys:
      - 'oof': pd.Series of out-of-fold predictions, indexed by original dates
      - 'model': final XGBoost model refit on all available (X, y)
      - 'metrics': dict of OOF metrics
    """
    # Drop rows where y is NaN (last h rows for horizon h have unobservable future)
    mask = y.notna()
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]

    splitter = walk_forward_splits(horizon)
    oof = pd.Series(index=y_clean.index, dtype=float)

    if task == "reg":
        model_cls = xgb.XGBRegressor
        params = dict(XGB_REG_PARAMS)
    elif task == "clf":
        model_cls = xgb.XGBClassifier
        params = dict(XGB_CLF_PARAMS)
        params["eval_metric"] = "logloss"
    else:
        raise ValueError(f"unknown task: {task}")

    for train_idx, test_idx in splitter.split(X_clean):
        model = model_cls(**params)
        if task == "reg":
            model.fit(X_clean.iloc[train_idx], y_clean.iloc[train_idx], verbose=False)
            preds = model.predict(X_clean.iloc[test_idx])
        else:
            y_train_bin = _binarize(y_clean.iloc[train_idx])
            model.fit(X_clean.iloc[train_idx], y_train_bin, verbose=False)
            preds = model.predict(X_clean.iloc[test_idx])
        oof.iloc[test_idx] = preds

    # Final model refit on all (X_clean, y_clean) for downstream inference
    final_model = model_cls(**params)
    if task == "reg":
        final_model.fit(X_clean, y_clean, verbose=False)
    else:
        final_model.fit(X_clean, _binarize(y_clean), verbose=False)

    # OOF metrics over rows that received a prediction
    valid = oof.notna()
    y_valid = y_clean.loc[valid]
    oof_valid = oof.loc[valid]

    if task == "reg":
        metrics = {
            "r2": float(r2_score(y_valid, oof_valid)),
            "rmse": float(np.sqrt(mean_squared_error(y_valid, oof_valid))),
            "mae": float(mean_absolute_error(y_valid, oof_valid)),
            "directional_accuracy": float(
                accuracy_score(y_valid > 0, oof_valid > 0)
            ),
            "n_oof": int(valid.sum()),
        }
    else:
        y_bin = _binarize(y_valid)
        metrics = {
            "accuracy": float(accuracy_score(y_bin, oof_valid.astype(int))),
            "f1": float(f1_score(y_bin, oof_valid.astype(int))),
            "n_oof": int(valid.sum()),
            "base_rate": float(y_bin.mean()),
        }

    return {"oof": oof, "model": final_model, "metrics": metrics}


def train_all(X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    """Fit regressors and classifiers for every (sector, horizon) cell."""
    oof_reg: dict[tuple[str, int], pd.Series] = {}
    oof_clf: dict[tuple[str, int], pd.Series] = {}
    rows: list[dict] = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    cells = [(s, h) for s in FF_INDUSTRIES for h in HORIZONS]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s, h in tqdm(cells, desc="Training"):
            y = Y[(s, h)]
            # Align X to Y index (intersection)
            X_a, y_a = X.align(y, join="inner", axis=0)

            # --- Regression ---
            result_reg = train_one(X_a, y_a, h, task="reg")
            oof_reg[(s, h)] = result_reg["oof"]
            result_reg["model"].save_model(MODELS_DIR / f"{s}_h{h}_reg.json")
            row = {"sector": s, "horizon": h, "task": "reg", **result_reg["metrics"]}
            rows.append(row)

            # --- Classification ---
            result_clf = train_one(X_a, y_a, h, task="clf")
            oof_clf[(s, h)] = result_clf["oof"]
            result_clf["model"].save_model(MODELS_DIR / f"{s}_h{h}_clf.json")
            row = {"sector": s, "horizon": h, "task": "clf", **result_clf["metrics"]}
            rows.append(row)

    oof_reg_df = pd.DataFrame(oof_reg)
    oof_reg_df.columns = pd.MultiIndex.from_tuples(
        oof_reg_df.columns, names=["sector", "horizon"]
    )
    oof_clf_df = pd.DataFrame(oof_clf)
    oof_clf_df.columns = pd.MultiIndex.from_tuples(
        oof_clf_df.columns, names=["sector", "horizon"]
    )
    metrics_df = pd.DataFrame(rows)

    return {"oof_reg": oof_reg_df, "oof_clf": oof_clf_df, "metrics": metrics_df}


def _print_summary(metrics: pd.DataFrame) -> None:
    reg = metrics[metrics["task"] == "reg"]
    clf = metrics[metrics["task"] == "clf"]

    print("\n=== Regression OOF R^2 (sector x horizon) ===")
    print(reg.pivot(index="sector", columns="horizon", values="r2").round(3).to_string())
    print(f"\nMean R^2: {reg['r2'].mean():+.4f}   Median R^2: {reg['r2'].median():+.4f}")
    print(f"Mean directional accuracy (reg): {reg['directional_accuracy'].mean():.3f}")

    print("\n=== Classification OOF accuracy (sector x horizon) ===")
    print(clf.pivot(index="sector", columns="horizon", values="accuracy").round(3).to_string())
    print(f"\nMean accuracy: {clf['accuracy'].mean():.3f}   Mean F1: {clf['f1'].mean():.3f}")
    print(f"Mean base rate (P[CAR>0]): {clf['base_rate'].mean():.3f}")


def main() -> None:
    X = pd.read_parquet(FEATURES_DIR / "X_v1.parquet")
    Y = pd.read_parquet(TARGETS_DIR / "Y_v1.parquet")
    print(f"Loaded X: {X.shape}, Y: {Y.shape}")

    results = train_all(X, Y)

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    results["oof_reg"].to_parquet(OOF_DIR / "oof_reg_v1.parquet")
    results["oof_clf"].to_parquet(OOF_DIR / "oof_clf_v1.parquet")
    print(f"Wrote OOF predictions to {OOF_DIR}")

    REPORT_TABLES.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_TABLES / "oof_metrics.csv"
    results["metrics"].to_csv(metrics_path, index=False)
    print(f"Wrote metrics to {metrics_path}")

    _print_summary(results["metrics"])

    reg_m = results["metrics"][results["metrics"]["task"] == "reg"]
    clf_m = results["metrics"][results["metrics"]["task"] == "clf"]

    update_manifest(
        "milestone_c_training",
        {
            "oof_reg": "outputs/oof/oof_reg_v1.parquet",
            "oof_clf": "outputs/oof/oof_clf_v1.parquet",
            "metrics": "outputs/report/tables/oof_metrics.csv",
            "n_models": len(reg_m) + len(clf_m),
            "reg_mean_r2": float(reg_m["r2"].mean()),
            "reg_mean_directional_acc": float(reg_m["directional_accuracy"].mean()),
            "clf_mean_accuracy": float(clf_m["accuracy"].mean()),
            "clf_mean_f1": float(clf_m["f1"].mean()),
            "xgb_params": dict(XGB_REG_PARAMS),
        },
    )


if __name__ == "__main__":
    main()
