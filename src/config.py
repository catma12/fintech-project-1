"""Project-wide configuration: paths, seeds, horizons, hyperparameters."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PANEL_PATH = ROOT / "master_panel_clean.csv"

OUTPUTS = ROOT / "outputs"
SHOCKS_DIR = OUTPUTS / "shocks"
FEATURES_DIR = OUTPUTS / "features"
TARGETS_DIR = OUTPUTS / "targets"
MODELS_DIR = OUTPUTS / "models"
OOF_DIR = OUTPUTS / "oof"
SHAP_DIR = OUTPUTS / "shap"
REPORT_DIR = OUTPUTS / "report"
REPORT_FIGURES = REPORT_DIR / "figures"
REPORT_TABLES = REPORT_DIR / "tables"
RUN_MANIFEST = OUTPUTS / "run_manifest.json"

SEED = 42

VAR_VARIABLES = ["dprod", "kilian_rea", "real_oil_price_diff"]
VAR_LAGS = 24

SHOCK_COLUMNS = ["eps_supply", "eps_agg_demand", "eps_precaut"]

HORIZONS = [1, 3, 6, 12]

FF_INDUSTRIES = [
    "NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
    "Telcm", "Utils", "Shops", "Hlth", "Money", "Other",
]

XGB_REG_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=SEED,
    tree_method="hist",
    n_jobs=4,
)

XGB_CLF_PARAMS = dict(XGB_REG_PARAMS)

# --- v2 (pooled model) configuration ---
#
# Diagnostic (outputs/diagnostics_cv.csv) showed the per-cell v1 models overfit
# badly (mean train-vs-test R² gap = 0.74, largely invariant to fold size).
# The v2 configuration pools all 12 sectors into one XGBoost model per horizon,
# trims features to 18 (drops redundant lag-2/3, cum_3m, and one-hot dominant-
# shock), and adds a 12-column sector one-hot so the tree can learn sector ×
# feature interactions from ~12x more training rows.

TRIMMED_FEATURES = [
    # 3 contemporaneous Kilian shocks
    "eps_supply", "eps_agg_demand", "eps_precaut",
    # 3 lag-1 shocks (drop lag-2/3 and cum_3m as redundant)
    "eps_supply_lag1", "eps_agg_demand_lag1", "eps_precaut_lag1",
    # overlap indicator
    "contamination_flag",
    # volatility regime (drop vix_is_proxy; rarely split)
    "vix_level", "vix_regime",
    # monetary regime
    "fed_regime_num",
    # credit regime
    "credit_spread_ig", "credit_spread_hy", "hy_available",
    # macro state
    "Recession",
    # oil context
    "oil_ret_3m", "oil_ret_12m", "net_oil_price_3yr", "oil_vol_6m_monthly",
]  # 18 features

XGB_POOLED_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=10,          # stronger than per-cell (we have 12x more rows)
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=SEED,
    tree_method="hist",
    n_jobs=4,
    early_stopping_rounds=30,
)

CRISIS_WINDOWS_PRECAUT = [
    ("1990-08-01", "1990-12-31", "Gulf War"),
    ("2002-10-01", "2003-03-31", "Iraq buildup"),
    ("2022-02-01", "2022-04-30", "Russia-Ukraine"),
]
CHINA_DEMAND_WINDOW = ("2003-07-01", "2007-12-31", "China demand boom")
