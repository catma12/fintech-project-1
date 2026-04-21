"""Auto-rendered HTML report — v1 → v4c experimental arc.

Reads every artifact produced across Milestones A-D and v2/v3/v4 iterations
and renders a self-contained HTML file (figures base64-embedded) to
outputs/report/report.html.

The report structure follows the four graded rubric sections
(completeness, novelty, readability, attribution) but the central narrative
is the v1 → v4c progression:

  v1    per-cell baseline               -> overfit
  v2    pooled + regularized            -> exposed signal ceiling
  v3    cross-sectional ranker          -> revealed anti-correctness
  v4c   ranker + momentum features      -> Sharpe flips positive

The 2026 out-of-sample section remains a placeholder until that data is
appended and src/oos_2026.py is run.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

from src.config import (
    HORIZONS,
    FF_INDUSTRIES,
    OOF_DIR,
    REPORT_DIR,
    REPORT_FIGURES,
    REPORT_TABLES,
    RUN_MANIFEST,
    SEED,
    TRIMMED_FEATURES,
    VAR_LAGS,
    VAR_VARIABLES,
    XGB_POOLED_PARAMS,
    XGB_REG_PARAMS,
)


# --- Helpers ----------------------------------------------------------------

def _b64_png(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _df_html(df: pd.DataFrame, float_fmt="{:+.4f}", classes="metrics-table", idx=False) -> str:
    if df.empty:
        return "<p><em>Not available.</em></p>"
    return df.to_html(classes=classes, index=idx, float_format=lambda x: float_fmt.format(x), border=0)


# --- Four-experiment summary figure -----------------------------------------

def build_four_experiment_summary(out_path: Path) -> None:
    """Two-panel bar chart: (top) regression mean edge by version, (bottom) ranker Sharpe by version."""
    v1 = _read(REPORT_TABLES / "oof_metrics.csv")
    v1_clf = v1[v1["task"] == "clf"].copy()
    v1_clf["naive"] = v1_clf["base_rate"].combine(1 - v1_clf["base_rate"], max)
    v1_clf["edge"] = v1_clf["accuracy"] - v1_clf["naive"]
    v1_edge = v1_clf["edge"].mean()

    v2 = _read(REPORT_TABLES / "oof_metrics_v2.csv")
    v2_xgb = v2[v2["model"] == "xgb_pooled_v2"]
    v2_edge = float(v2_xgb["edge"].mean()) if not v2_xgb.empty else 0.0

    v4a = _read(REPORT_TABLES / "oof_metrics_v4a.csv")
    v4a_edge = float(v4a["edge"].mean()) if not v4a.empty else 0.0

    v4b = _read(REPORT_TABLES / "oof_metrics_v4b.csv")
    v4b_edge = float(v4b["edge"].mean()) if not v4b.empty else 0.0

    v3 = _read(REPORT_TABLES / "ranking_summary_v3.csv")
    v3_sharpe = float(v3["sharpe_ann"].mean()) if not v3.empty else 0.0

    v4c = _read(REPORT_TABLES / "ranking_summary_v4c.csv")
    v4c_sharpe = float(v4c["sharpe_ann"].mean()) if not v4c.empty else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    # Classification edge
    ax = axes[0]
    labels = ["v1\nper-cell", "v2\npooled", "v4a\npc+mom", "v4b\npooled+mom"]
    values = [v1_edge, v2_edge, v4a_edge, v4b_edge]
    colors = ["#b0b7c0", "#7a90b0", "#486796", "#1f3b66"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Mean classification edge vs naive baseline")
    ax.set_title("Regression / classification paradigm", fontsize=11, loc="left")
    ax.grid(alpha=0.25, axis="y")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.003 if v >= 0 else -0.006),
                f"{v:+.3f}", ha="center", fontsize=9)

    # Ranker Sharpe
    ax = axes[1]
    labels = ["v3\nmacro only", "v4c\n+ momentum"]
    values = [v3_sharpe, v4c_sharpe]
    colors = ["#c94a3a", "#2e8b57"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Mean long-short Sharpe (annualized)")
    ax.set_title("Cross-sectional ranking paradigm", fontsize=11, loc="left")
    ax.grid(alpha=0.25, axis="y")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.012 if v >= 0 else -0.02),
                f"{v:+.3f}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Experimental arc — cumulative improvement across four training paradigms",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --- Table builders ---------------------------------------------------------

def _build_adf_table(adf: pd.DataFrame) -> str:
    if adf.empty:
        return "<p><em>Not available.</em></p>"
    a = adf.copy()
    a["p_value"] = a["p_value"].apply(lambda x: f"{x:.4f}")
    a["adf_stat"] = a["adf_stat"].apply(lambda x: f"{x:+.3f}")
    a["crit_5pct"] = a["crit_5pct"].apply(lambda x: f"{x:+.3f}")
    a["stationary_at_5pct"] = a["stationary_at_5pct"].map({True: "Yes", False: "No"})
    cols = ["variable", "adf_stat", "p_value", "crit_5pct", "stationary_at_5pct", "n_obs"]
    return a[cols].to_html(classes="metrics-table", index=False, border=0)


def _build_sign_check_table(window: pd.DataFrame) -> str:
    if window.empty:
        return "<p><em>Not available.</em></p>"
    w = window[["eps_supply", "eps_agg_demand", "eps_precaut", "dprod"]].copy()
    return w.to_html(classes="metrics-table", float_format=lambda x: f"{x:+.3f}", border=0)


def _build_fevd_table(fevd: pd.DataFrame) -> str:
    if fevd.empty:
        return "<p><em>Not available.</em></p>"
    h12 = fevd[fevd["horizon"] == 12].copy()
    h12["Supply"] = (h12["supply_share"] * 100).round(1)
    h12["Agg demand"] = (h12["agg_demand_share"] * 100).round(1)
    h12["Precautionary"] = (h12["precautionary_share"] * 100).round(1)
    h12["All oil (sum)"] = (h12["oil_total_share"] * 100).round(1)
    h12 = h12.sort_values("All oil (sum)", ascending=False)
    h12 = h12[["sector", "Supply", "Agg demand", "Precautionary", "All oil (sum)"]]
    return h12.to_html(classes="metrics-table", index=False, float_format=lambda x: f"{x:.1f}", border=0)


def _build_v4c_table(v4c: pd.DataFrame) -> str:
    if v4c.empty:
        return "<p><em>Not available.</em></p>"
    tbl = v4c.copy()
    tbl = tbl[["horizon", "n_months", "mean_ls_ret", "sharpe_ann", "t_stat",
               "hit_rate", "mean_spearman", "mean_top3_hit"]]
    tbl.columns = ["h (months)", "n_obs", "Mean L-S return", "Sharpe (ann)",
                   "t-stat", "Hit rate", "Mean Spearman", "Top-3 hit"]
    tbl["h (months)"] = tbl["h (months)"].astype(int)
    tbl["n_obs"] = tbl["n_obs"].astype(int)
    return tbl.to_html(
        classes="metrics-table", index=False,
        float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.0f}",
        border=0,
    )


def _build_v3_table(v3: pd.DataFrame) -> str:
    return _build_v4c_table(v3)


def _build_v1_r2_pivot(v1: pd.DataFrame) -> str:
    reg = v1[v1["task"] == "reg"]
    if reg.empty:
        return ""
    pv = reg.pivot(index="sector", columns="horizon", values="r2").round(3)
    pv.columns = [f"h={h}" for h in pv.columns]
    return pv.reset_index().to_html(
        classes="metrics-table", index=False,
        float_format=lambda x: f"{x:+.3f}", border=0,
    )


def _build_v2_r2_pivot(v2: pd.DataFrame) -> str:
    xgb = v2[v2["model"] == "xgb_pooled_v2"]
    if xgb.empty:
        return ""
    pv = xgb.pivot(index="sector", columns="horizon", values="r2").round(3)
    pv.columns = [f"h={h}" for h in pv.columns]
    return pv.reset_index().to_html(
        classes="metrics-table", index=False,
        float_format=lambda x: f"{x:+.3f}", border=0,
    )


def _build_headline_arc_table(v1, v2, v4a, v4b, v3, v4c) -> str:
    """Single table that summarizes the arc in one view."""
    rows = []
    # v1 classifier edge
    v1c = v1[v1["task"] == "clf"].copy()
    v1c["naive"] = v1c["base_rate"].combine(1 - v1c["base_rate"], max)
    v1c["edge"] = v1c["accuracy"] - v1c["naive"]
    rows.append({
        "Version": "v1",
        "Architecture": "per-cell (48 models)",
        "Features": "33",
        "Mean OOF R²": "—",
        "Mean class. edge": f"{v1c['edge'].mean():+.3f}",
        "Mean L-S Sharpe": "—",
        "+edge cells": f"{int((v1c['edge'] > 0).sum())}/48",
    })
    v2x = v2[v2["model"] == "xgb_pooled_v2"]
    rows.append({
        "Version": "v2",
        "Architecture": "pooled regression (4 models)",
        "Features": "18 + 12 one-hot",
        "Mean OOF R²": f"{v2x['r2'].mean():+.3f}",
        "Mean class. edge": f"{v2x['edge'].mean():+.3f}",
        "Mean L-S Sharpe": "—",
        "+edge cells": f"{int((v2x['edge'] > 0).sum())}/48",
    })
    if not v3.empty:
        rows.append({
            "Version": "v3",
            "Architecture": "pooled ranker (4 models)",
            "Features": "18 + 12 one-hot",
            "Mean OOF R²": "—",
            "Mean class. edge": "—",
            "Mean L-S Sharpe": f"{v3['sharpe_ann'].mean():+.3f}",
            "+edge cells": "—",
        })
    if not v4a.empty:
        rows.append({
            "Version": "v4a",
            "Architecture": "per-cell (48 models) + momentum",
            "Features": "33 + 5 = 38",
            "Mean OOF R²": f"{v4a['r2'].mean():+.3f}",
            "Mean class. edge": f"{v4a['edge'].mean():+.3f}",
            "Mean L-S Sharpe": "—",
            "+edge cells": f"{int((v4a['edge'] > 0).sum())}/48",
        })
    if not v4b.empty:
        rows.append({
            "Version": "v4b",
            "Architecture": "pooled regression + momentum",
            "Features": "18 + 12 + 5 = 35",
            "Mean OOF R²": f"{v4b['r2'].mean():+.3f}",
            "Mean class. edge": f"{v4b['edge'].mean():+.3f}",
            "Mean L-S Sharpe": "—",
            "+edge cells": f"{int((v4b['edge'] > 0).sum())}/48",
        })
    if not v4c.empty:
        rows.append({
            "Version": "v4c",
            "Architecture": "pooled ranker + momentum",
            "Features": "18 + 12 + 5 = 35",
            "Mean OOF R²": "—",
            "Mean class. edge": "—",
            "Mean L-S Sharpe": f"{v4c['sharpe_ann'].mean():+.3f}",
            "+edge cells": "—",
        })
    df = pd.DataFrame(rows)
    return df.to_html(classes="metrics-table arc-table", index=False, border=0, escape=False)


# --- Template ---------------------------------------------------------------

TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Oil-Shock Sector Rotation — MIT 15.C51 Project 1</title>
  <style>
    body { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; max-width: 1120px; margin: 2em auto; padding: 0 1.5em; color: #1a1a1a; line-height: 1.55; }
    h1 { border-bottom: 3px solid #1f3b66; padding-bottom: 0.3em; color: #1f3b66; font-size: 2em; }
    h2 { color: #1f3b66; margin-top: 2em; border-left: 4px solid #1f3b66; padding-left: 0.6em; }
    h3 { color: #2c4a7a; margin-top: 1.4em; }
    h4 { color: #3a567f; margin-top: 1em; }
    .meta { background: #f4f6fa; padding: 0.9em 1.1em; border-radius: 6px; font-size: 0.92em; }
    .figure { text-align: center; margin: 1.2em 0; }
    .figure img { max-width: 100%; border: 1px solid #d0d7e1; border-radius: 4px; }
    .figure .caption { font-size: 0.88em; color: #445; margin-top: 0.4em; }
    table.metrics-table { border-collapse: collapse; width: 100%; font-size: 0.9em; margin: 0.8em 0; }
    table.metrics-table th, table.metrics-table td { border-bottom: 1px solid #dde2ea; padding: 6px 10px; text-align: right; }
    table.metrics-table th { background: #1f3b66; color: white; text-align: left; font-weight: 500; }
    table.metrics-table td:first-child, table.metrics-table th:first-child { text-align: left; }
    table.arc-table td, table.arc-table th { font-size: 0.93em; }
    .callout { background: #fef9e7; border-left: 4px solid #d4a017; padding: 0.7em 1em; margin: 1em 0; border-radius: 3px; }
    .pending { background: #fdecea; border-left: 4px solid #c0392b; padding: 0.7em 1em; margin: 1em 0; border-radius: 3px; color: #6a1a15; }
    .key-finding { background: #eafaf1; border-left: 4px solid #1e8449; padding: 0.8em 1em; margin: 1em 0; border-radius: 3px; }
    .win { background: #e6f4ea; border-left: 4px solid #188038; padding: 0.8em 1em; margin: 1em 0; border-radius: 3px; }
    code { background: #f0f2f6; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }
    pre { background: #f6f8fa; padding: 10px; border-radius: 4px; font-size: 0.85em; overflow-x: auto; }
    .small { font-size: 0.85em; color: #555; }
    .version-label { display: inline-block; padding: 2px 8px; border-radius: 3px; font-weight: 500; font-size: 0.85em; color: white; }
    .v1 { background: #b0b7c0; }
    .v2 { background: #7a90b0; }
    .v3 { background: #c94a3a; }
    .v4 { background: #1f3b66; }
  </style>
</head>
<body>

<h1>Oil-Shock Sector Rotation — An ML Extension of Kilian &amp; Park (2009)</h1>

<div class="meta">
<strong>Project:</strong> 15.C51 Modeling with Machine Learning: Financial Technology · MIT Sloan · Spring 2026<br>
<strong>Research question:</strong> Given the Kilian-decomposed oil shock at month t and the prevailing macro regime, can a supervised model predict the cross-section of 12 Fama-French industry returns over horizons of 1, 3, 6, and 12 months — well enough to support a sector-rotation strategy?<br>
<strong>Sample:</strong> monthly, 1986-01 to 2025-12 (480 rows). VAR usable from 1988-01 after 24-lag burn-in; OOF evaluation from 1994-03 onward.<br>
<strong>Report generated:</strong> {{ generated_at }} · <strong>Git SHA:</strong> <code>{{ git_sha }}</code>
</div>

<h2>Executive summary</h2>

<p>
This project extends Kilian &amp; Park (2009) from average impulse-response description to
forward-looking prediction using XGBoost, and progressively tests whether the
result is tradable as a sector rotation. We iterated through four training paradigms,
each exposing a different obstacle to predictive signal:
</p>

<div class="figure">
  <img src="data:image/png;base64,{{ four_exp_b64 }}" alt="Four-experiment summary">
  <div class="caption"><strong>Figure 0.</strong> The arc at a glance. Left: mean classification edge vs naive majority-class baseline for the regression paradigm — small improvements with pooling and momentum but never positive. Right: mean long-short Sharpe for the ranking paradigm — v3 (macro only) is net negative; v4c (macro + momentum) flips positive across all four horizons.</div>
</div>

{{ headline_arc_table_html | safe }}

<div class="win">
<strong>Main finding.</strong> A pooled XGBoost learning-to-rank model built on Kilian structural shocks, macro-regime features, and sector-level momentum produces a long-top-3 / short-bottom-3 strategy with a mean annualized Sharpe of <strong>{{ '%+.3f' % v4c_mean_sharpe }}</strong> across four horizons. The improvement over the macro-only ranker (v3 → v4c) is a <strong>{{ '%+.3f' % (v4c_mean_sharpe - v3_mean_sharpe) }}</strong> Sharpe swing. Naive t-statistics at h=3 and h=6 are <strong>+{{ '%.2f' % v4c_h3_tstat }}</strong> and <strong>+{{ '%.2f' % v4c_h6_tstat }}</strong>; properly adjusted for overlapping-return autocorrelation these drop to ~1.0-1.2, so strictly speaking the strategy is not significant at the 5% level — but four independent metrics (Sharpe, Spearman, top-3 hit, cumulative L-S) all agree in direction on a previously anti-correct baseline.
</div>

<h2>1 · Financial analysis completeness</h2>

<h3>1.1 · Data panel</h3>
<p>
A 480-row monthly panel spanning 1986-01 through 2025-12. The Kilian VAR inputs —
global crude production (<code>dprod</code>), the Kilian global real-activity index
(<code>kilian_rea</code>), and real WTI (deflated by CPI) — are all present with zero
missing. Macro regime features (VIX, IG and HY credit spreads, NBER recession,
Fed funds) are gated by availability flags where needed (VIX real data from 1990;
HY spread from 1994). Fama-French 12-industry returns (raw, market-adjusted,
and risk-free-excess forms) plus Mkt and RF come from Ken French's data library
via the CRSP value-weighted universe. Full data dictionary lives in
<code>CLAUDE.md</code>.
</p>

<h3>1.2 · Kilian structural VAR identification</h3>
<p>
We estimate a {{ var_lags }}-lag block-recursive VAR on
<code>{{ var_variables | join(', ') }}</code> and recover three standardized
structural shock series via Cholesky decomposition: supply, aggregate demand,
and precautionary (oil-specific) demand. Ordering follows Kilian (2009) exactly.
</p>

<h4>Stationarity — ADF tests</h4>
<p>
The VAR requires stationary inputs. Log-level real oil price fails the ADF test,
so we feed the first difference (log monthly return). <code>kilian_rea</code> is
borderline (p = 0.13) but is standard in the Kilian literature and the VAR
accommodates it.
</p>
{{ adf_table_html | safe }}

<h4>Shock validation — the 1990 Gulf War sign-convention check</h4>
<p>
A common identification error produces wrong-signed shocks. Our VAR's August
1990 supply shock is decisively negative (Iraq invasion on 1990-08-02 suspends
~4.5 mb/d of supply); Saudi compensation hits the data in September, producing
a positive supply <em>surprise</em>; the war-risk premium generates the
largest precautionary shock in our entire sample. This is textbook Kilian:
</p>
{{ sign_check_html | safe }}

<div class="figure">
  <img src="data:image/png;base64,{{ hist_decomp_b64 }}" alt="Historical decomposition">
  <div class="caption"><strong>Figure 1.</strong> Three-channel structural shock decomposition, 1988-01 to 2025-12. Dashed red lines at ±1.5σ mark the large-shock threshold used by the contamination flag.</div>
</div>

<h3>1.3 · In-sample predictability ceiling — variance decomposition</h3>
<p>
Before asking how well oil shocks <em>predict</em> sector returns out-of-sample,
we quantify the in-sample ceiling: per sector, we fit a 4-variable VAR
<code>[dprod, kilian_rea, real_oil_price_diff, FF_sector_abn]</code> and
decompose the sector's forecast-error variance at the 12-month horizon.
</p>

{{ fevd_table_html | safe }}

<div class="figure">
  <img src="data:image/png;base64,{{ fevd_heatmap_b64 }}" alt="FEVD heatmap">
  <div class="caption"><strong>Figure 2.</strong> Share of sector-return variance attributable to oil shocks at h=12 (in-sample). Energy tops out at 37.2% — the structural ceiling on how much predictive power we could extract from oil shocks alone for this sector. For the 12-sector average the ceiling is only ~14%, and the precautionary channel dominates throughout.</div>
</div>

<h3>1.4 · Feature engineering progression</h3>
<p>The feature set evolved across versions:</p>
<ul>
  <li><strong>v1 (33 features):</strong> 3 contemporaneous Kilian shocks + 9 lagged (l1/l2/l3) + 3 cumulative-3-month + dominant-shock one-hot (3) + signed magnitude (2) + contamination flag + vix_level/vix_regime/vix_is_proxy + fed_regime_num + IG/HY credit spreads + hy_available + Recession + oil_ret_3m/12m + net_oil_price_3yr + oil_vol_6m_monthly.</li>
  <li><strong>v2 trim (18 features + 12 sector one-hot = 30):</strong> drops lag-2/3, cum_3m, dominant-shock one-hot, shock_sign, shock_magnitude, vix_is_proxy. Trees can still recover lag-2/3 effects from the continuous shock series; removing them lets the model use its tree-capacity budget on the interactions that matter.</li>
  <li><strong>v4 momentum (v2 + 5 sector-specific features):</strong> <code>own_ret_1m</code>, <code>own_ret_3m</code>, <code>own_mom_12_1</code> (Jegadeesh-Titman 12-1 skip-1), <code>own_vol_6m</code>, <code>cross_rank_12_1</code> (cross-sectional rank of 12-1 momentum — direct input for the ranker).</li>
</ul>

<h3>1.5 · Target construction</h3>
<p>
The target at month t for horizon h is the forward cumulative abnormal return
<code>CAR(s, t, h) = sum(FF_s_abn[t+1 .. t+h])</code>. Arithmetic sum, not
compounded BHAR — standard event-study convention and the monthly
<code>FF_*_abn</code> column is already an arithmetic abnormal return.
Three parallel target formulations:
</p>
<ul>
  <li><strong>Regression (v1, v2, v4a, v4b):</strong> continuous CAR.</li>
  <li><strong>Classification (v1 clf):</strong> binary <code>sign(CAR)</code>.</li>
  <li><strong>Ranking (v3, v4c):</strong> integer 0..11 rank within each month's 12-sector group; XGBRanker <code>rank:pairwise</code> objective.</li>
</ul>

<h3>1.6 · Walk-forward cross-validation</h3>
<p>
<code>TimeSeriesSplit(n_splits=5, gap=max(3, h+1))</code> prevents target-window
overlap between train and test. For h=12, gap=13 months — the guide's default
gap=3 would have leaked nine months of overlap. Feature matrix has a
machine-checked <code>check_no_lookahead</code> pass confirming no
<code>FF_*_abn</code> column appears in X, and that spot-checked events
(Russia 2022-02 precautionary spike, COVID 2020-03 sign, etc.) have the
expected direction.
</p>

<h2>2 · Novelty — the experimental arc</h2>

<h3>2.1 · <span class="version-label v1">v1</span> per-cell XGBoost baseline</h3>
<p>
Forty-eight regression models (12 sectors × 4 horizons), each fit on ~450
observations with 33 features, and 48 parallel binary classifiers. Hyperparameters
follow the implementation-guide defaults: max_depth=3, 300 trees, lr=0.03,
min_child_weight=5, L1/L2 = 0.5/1.0.
</p>

<p>
Mean OOF R² is <strong>−0.271</strong> — worse than a mean-prediction baseline
across every cell. Mean classification edge vs naive is <strong>−0.060</strong>;
only <strong>6/48 cells</strong> beat the naive majority-class baseline, and the
strongest (Hlth h=12) has a +4.1pt edge.
</p>

<h4>v1 regression R² (by sector × horizon)</h4>
{{ v1_r2_pivot_html | safe }}

<h3>2.2 · Diagnostic — why is v1 so bad?</h3>
<p>
Before iterating, we quantified whether the problem was complexity (overfit),
sample-size (tiny early folds), or feature-signal strength. The diagnostic
(<code>src/diagnostics.py</code>, output in <code>outputs/diagnostics_cv.csv</code>)
fits three model configurations on four representative cells:
</p>

<pre>Mean across 4 cells × 5 folds, by config:
                    train R²   test R²   gap
current (d3, n300)   +0.551    -0.190   0.741
shallow (d2, n100)   +0.386    -0.155   0.541
stumps  (d1, n50)    +0.218    -0.147   0.365
</pre>

<p>
Findings: (1) the gap is <strong>catastrophic</strong> — model explains 55% of
training variance but -19% of test; (2) gap doesn't shrink across folds
1→5, so it's not just small-sample overfit but a non-stationarity between
in-sample and out-of-sample regimes; (3) stumps (depth-1) nearly halve the gap
while test R² barely moves — depth is the lever. Also: CAR target lag-1
autocorrelation is 0.03 at h=1 but <strong>0.92 at h=12</strong>, meaning the
effective sample size shrinks to ~{{ '%d' % 31 }} independent observations at
h=12. No amount of model tuning can fix an AC=0.92 target.
</p>

<h3>2.3 · <span class="version-label v2">v2</span> pooled regression + regularization</h3>
<p>
Based on the diagnostic: pool all 12 sectors into one regressor per horizon
(4× one-hot sector encoding), trim to 18 features, keep max_depth=3 but use
early stopping on a 15% time-ordered held-out val slice of each fold's
training set, and strengthen min_child_weight to 10.
</p>

<p>
<strong>Effect:</strong> mean OOF R² improves from <strong>−0.271</strong> to
<strong>−0.034</strong> — an 8× reduction in the overfit gap. The model is no
longer actively worse than predicting the mean; it sits just below zero.
Classification edge is essentially unchanged (<strong>−0.090</strong>), which
tells us regularization cured <em>overfitting</em> but didn't unlock
<em>signal</em>. Ridge regression on the same features produces similar numbers,
confirming nonlinearity isn't the missing ingredient.
</p>

<h4>v2 pooled regression R² (by sector × horizon)</h4>
{{ v2_r2_pivot_html | safe }}

<h3>2.4 · <span class="version-label v3">v3</span> ranking reframe — reveals the true problem</h3>
<p>
Sector rotation is an inherently <em>relative</em> decision. We reframe as
learning-to-rank: for each month, target is the 0..11 rank of the 12 sectors
by forward CAR, and XGBoost's <code>rank:pairwise</code> objective optimizes
for within-group ordering. Long-short backtest: long top-3 predicted sectors,
short bottom-3, each month.
</p>

{{ v3_table_html | safe }}

<p>
<strong>Two horizons neutral, two significantly anti-correct.</strong> The model
is not random — at h=3 and h=12 it is systematically <em>wrong</em>, producing
monotonically declining cumulative long-short returns. The sector-selection
plot reveals the mechanism: the macro-only ranker consistently goes long the
defensives (Money, NoDur, Hlth) and short cyclicals (Durbl, Enrgy, Other) —
a "risk-off" bet conditioned on regime features. That bet wins weakly at h=1
but gets systematically reversed over longer horizons, which is the classic
Jegadeesh-Titman momentum / DeBondt-Thaler reversal pattern in equity
markets. Adjusted for overlapping-return autocorrelation the anti-correctness
is borderline rather than strictly significant, but the direction is
unambiguous.
</p>

<h3>2.5 · <span class="version-label v4">v4c</span> ranker + momentum features — the main result</h3>
<p>
The v3 diagnosis points squarely at a missing feature: the model has no
information about sector-level return continuation. We add five sector-specific
predictors (own-return lags, 12-1 Jegadeesh-Titman momentum, own-volatility,
and a cross-sectional rank of 12-1 momentum across the 12 sectors) and re-run
the ranker:
</p>

{{ v4c_table_html | safe }}

<div class="win">
<strong>Every horizon flips positive.</strong> Mean annualized Sharpe across
four horizons: <strong>{{ '%+.3f' % v4c_mean_sharpe }}</strong> (v3 was
<strong>{{ '%+.3f' % v3_mean_sharpe }}</strong>). At h=3, v3's −0.28 Sharpe
with −2.71 t-stat becomes v4c's +0.21 Sharpe with +2.05 t-stat — a complete
reversal. Naive t-stats at h=3 and h=6 reach +2.05 and +2.21; autocorrelation-
adjusted values drop to ~1.0-1.2, not strictly significant but consistently
directional across Sharpe, Spearman, top-3 hit rate, and cumulative return.
</div>

<h3>2.6 · Head-to-head: v1 → v4c across paradigms</h3>

<div class="figure">
  <img src="data:image/png;base64,{{ v3_vs_v4c_b64 }}" alt="v3 vs v4c cumulative comparison">
  <div class="caption"><strong>Figure 3.</strong> Cumulative long-short abnormal return by horizon — v3 (macro only, red) vs v4c (macro + momentum, blue). The h=3 panel is the cleanest story: v3 drops from 0 to −3 over the OOF window; v4c climbs from 0 to +2. Adding momentum features flips the strategy from mean-reversion loser to momentum-continuation winner.</div>
</div>

<h3>2.7 · v4c sector rotation — which sectors does the model pick?</h3>

<div class="figure">
  <img src="data:image/png;base64,{{ v4c_sector_sel_b64 }}" alt="v4c sector selection frequency">
  <div class="caption"><strong>Figure 4.</strong> Per-horizon selection frequency in the v4c top-3 long (green) and bottom-3 short (red) baskets. Dashed gray lines mark the 3/12 = 25% random baseline. The momentum-enriched ranker produces more differentiated picks than v3: Durbl and Other remain heavily shorted, but the long book diversifies across Hlth, NoDur, Enrgy, and Manuf depending on horizon, rather than concentrating on Money across the board.</div>
</div>

<div class="figure">
  <img src="data:image/png;base64,{{ v4c_ls_cum_b64 }}" alt="v4c cumulative L-S">
  <div class="caption"><strong>Figure 5.</strong> v4c cumulative long-short abnormal return. All four horizons end the OOF period positive; drawdowns are concentrated in 2008-2010 (GFC) and 2020-2021 (COVID).</div>
</div>

<h3>2.8 · Caveats</h3>
<ul>
  <li><strong>Statistical significance after AC adjustment.</strong> For h=3 targets, CAR at t and t+1 share 2/3 of their components; at h=12, 11/12. Effective independent sample sizes are ~125 and ~31. Adjusting the t-stats by √horizon brings them to ~1.0-1.2 — directionally robust but not strictly significant at conventional thresholds.</li>
  <li><strong>No transaction cost modeling.</strong> A monthly-rebalanced top-3 / bottom-3 rotation would incur turnover; the reported Sharpe is gross.</li>
  <li><strong>In-sample FEVD ceiling.</strong> Even if extracted perfectly, oil shocks alone explain only ~14% of sector-variance on average (37% for Enrgy). v4c's signal is necessarily a <em>combination</em> of oil-shock and momentum information.</li>
  <li><strong>Sample horizon.</strong> All conclusions rest on 1994-2024 OOF data (375 months). The 2026 out-of-sample test in Section 6 will be the cleanest forward validation.</li>
</ul>

<h2>3 · Interpretability — SHAP (v1)</h2>
<p>
SHAP panels below come from the <em>v1 regression</em> models, saved before the
signal-ceiling diagnosis. They show what the <em>per-cell</em> macro-regime
model was attending to at the h=6 horizon — a useful baseline view even though
v4c moved to a ranking objective with added momentum features.
</p>

<div class="figure">
  <img src="data:image/png;base64,{{ shap_global_bar_b64 }}" alt="Global SHAP bar grid">
  <div class="caption"><strong>Figure 6.</strong> Global SHAP importance by sector at h=6, v1 regression models. Across nearly every sector, <code>credit_spread_hy</code>, <code>credit_spread_ig</code>, <code>oil_vol_6m_monthly</code>, and <code>oil_ret_12m</code> dominate — forward-looking, market-priced indicators. The raw Kilian shocks (residuals by construction) rank below these, which is consistent with the v3 finding that macro-only features don't carry strong cross-sectional signal; the v4c improvement shows where the missing piece was.</div>
</div>

<div class="figure">
  <img src="data:image/png;base64,{{ shap_heatmap_b64 }}" alt="Cross-sector SHAP heatmap">
  <div class="caption"><strong>Figure 7.</strong> Cross-sector SHAP heatmap at h=6 — the ML analogue of Kilian &amp; Park Figure 6. Same feature, opposite signs across sectors reveals the latent rotation pattern.</div>
</div>

<div class="figure">
  <img src="data:image/png;base64,{{ shap_waterfall_russia_b64 }}" alt="Russia 2022 waterfall">
  <div class="caption"><strong>Figure 8.</strong> Waterfall for FF_Enrgy prediction at 2022-02 (Russia-Ukraine invasion month). Model predicts +0.20 six-month Energy CAR; <code>oil_ret_12m</code> (+0.09) and <code>oil_vol_6m</code> (+0.05) dominate contribution. <code>eps_precaut</code> is in the "22 other features" aggregate — its individual contribution is small relative to the trend-and-volatility features, consistent with the v1 global finding.</div>
</div>

<div class="figure">
  <img src="data:image/png;base64,{{ shap_dep_enrgy_b64 }}" alt="Dependence Enrgy">
  <div class="caption"><strong>Figure 9.</strong> SHAP dependence: <code>eps_precaut</code> SHAP × <code>vix_regime</code> interaction for FF_Enrgy at h=6. Visualizes the regime-conditional transmission that motivated the feature set.</div>
</div>

<p class="small">
Future work: rerun SHAP on the v4c ranker models to confirm whether
<code>cross_rank_12_1</code> and <code>own_mom_12_1</code> dominate there as
hypothesized. Out of scope for this delivery.
</p>

<h2>4 · Reproducibility &amp; readability</h2>

<p>
Complete reproduction from <code>requirements.txt</code> via <code>make all</code>
or any milestone target (<code>make shocks | features | train | shap |
report</code>). Every intermediate artifact is versioned in
<code>outputs/</code> with a <code>run_manifest.json</code> recording git SHA,
seed ({{ seed }}), panel SHA-256, and per-milestone metadata. Seven commits on
<code>main</code> capture the full arc (bootstrap + Milestone A through v4 +
this report regeneration).
</p>

<h3>Project layout</h3>
<pre>oil-prices/
  src/
    config.py                 paths, seeds, horizons, XGB params, TRIMMED_FEATURES
    io_utils.py               load_panel(), update_manifest()
    var_shocks.py             Milestone A — Kilian VAR + Cholesky shocks
    features.py               v1 33-feature matrix + no-lookahead asserts
    features_momentum.py      v4 add-ons — own-sector + cross-sectional momentum
    targets.py                CAR + sign targets
    cv.py                     horizon-aware TimeSeriesSplit
    train.py                  v1 per-cell (48 regressors + 48 classifiers)
    train_v2.py               v2 pooled regression + Ridge baseline
    train_ranking.py          v3 XGBRanker + L-S backtest + plotting helpers
    train_v4.py               v4a/v4b/v4c orchestration + comparison tables
    diagnostics.py            per-fold train/test gap diagnostic
    verify_shocks.py          ADF + 1990 sign check + per-sector FEVD
    shap_analysis.py          Milestone D SHAP panels
    report.py                 this file
    oos_2026.py               Milestone E stub
  notebooks/01..04_*.ipynb    thin consumer notebooks per milestone
  outputs/{shocks,features,targets,models,oof,shap,report}/
  Makefile, requirements.txt, CLAUDE.md
</pre>

<h3>Model hyperparameters</h3>
<p>
<strong>v1 per-cell:</strong> <code>{{ xgb_v1_json }}</code><br>
<strong>v2/v4 pooled:</strong> <code>{{ xgb_pooled_json }}</code><br>
<strong>v3/v4c ranker:</strong> rank:pairwise, n=500 with 30-round early
stopping, max_depth=3, lr=0.05, min_child_weight=10, reg_alpha=0.5,
reg_lambda=1.0, random_state=42.
</p>

<h2>5 · Source attribution</h2>

<h3>5.1 · Data sources</h3>
<ul>
  <li>WTI spot (<code>CO1 Comdty</code>) and CPI YoY (<code>CPI YOY Index</code>) — Bloomberg.</li>
  <li>Global crude production (<code>DOEWCRPW Index</code>) — Bloomberg / EIA.</li>
  <li>Kilian global real activity index — Lutz Kilian's website; Baltic Dry Index (Bloomberg) used to extend through 2025.</li>
  <li>VIX (<code>VIX Index</code>), IG spread (<code>LUACOAS Index</code>), HY spread (<code>LF98OAS Index</code>), Fed funds (<code>FDFD Index</code>) — Bloomberg.</li>
  <li>NBER recession indicator (<code>USREC</code>) — FRED.</li>
  <li>Fama-French 12-industry returns and factor series — Kenneth French's data library, via CRSP.</li>
</ul>

<h3>5.2 · Methodology citations</h3>
<ul>
  <li>Kilian, L. (2009). "Not All Oil Price Shocks Are Alike: Disentangling Demand and Supply Shocks in the Crude Oil Market." <em>American Economic Review</em>, 99(3), 1053-1069.</li>
  <li>Kilian, L. &amp; Park, C. (2009). "The Impact of Oil Price Shocks on the U.S. Stock Market." <em>International Economic Review</em>, 50(4), 1267-1287.</li>
  <li>Jegadeesh, N. &amp; Titman, S. (1993). "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." <em>Journal of Finance</em>, 48(1), 65-91.</li>
  <li>DeBondt, W. &amp; Thaler, R. (1985). "Does the Stock Market Overreact?" <em>Journal of Finance</em>, 40(3), 793-805.</li>
  <li>Lundberg, S. M. &amp; Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." <em>NeurIPS</em>.</li>
  <li>Chen, T. &amp; Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." <em>KDD</em>.</li>
  <li>Burges, C. (2010). "From RankNet to LambdaRank to LambdaMART: An Overview." <em>Microsoft Tech Report</em>.</li>
  <li>Fama, E. F. &amp; French, K. R. — 12-industry classification by SIC code.</li>
</ul>

<h3>5.3 · LLM usage</h3>
<p class="small">
This pipeline was developed in pair-programming collaboration with Claude
Opus 4.7 (Anthropic). Representative prompts used:
(1) "Analyze this codebase and create a CLAUDE.md ..."; (2) "Let's do Milestone
A / B / C / D / momentum"; (3) "Identify possible causes to this [negative R²]
problem and let's try some solutions. Ultrathink"; (4) "Let's proceed with
that" (approving each proposed experiment); (5) "Run the diagnostic, please";
(6) "Also, another team member is building his own model and these are his
conclusions. Check if something's applicable and worth exploring." All
substantive design decisions (VAR first-differencing, z ≥ 1.25 precautionary
threshold, arithmetic CAR, horizon-aware CV gap, pooled-model pivot, momentum
feature selection) were discussed in conversation and chosen explicitly; the
model did not make independent research decisions. Full prompt history is
preserved in the project's Claude Code session log.
</p>

<h3>5.4 · Software / libraries</h3>
<p class="small">
pandas, numpy, scipy, statsmodels (VAR, ADF, FEVD), scikit-learn
(TimeSeriesSplit, RidgeCV, metrics), xgboost (XGBRegressor, XGBClassifier,
XGBRanker), shap (TreeExplainer), matplotlib, seaborn, pyarrow (parquet),
jinja2 (this template), jupyter. Python 3.11. All packages unpinned in
<code>requirements.txt</code>; pipeline is deterministic with
<code>random_state=42</code> and <code>tree_method="hist"</code>.
</p>

<h2>6 · 2026 out-of-sample test (pending)</h2>
<div class="pending">
The panel ends <strong>2025-12-31</strong>. The assignment-required 2026
validation will auto-populate this section once 2026 rows are appended to
<code>master_panel_clean.csv</code> and <code>python -m src.oos_2026</code> is
run. The interface is designed to refit the Kilian VAR on data through each
2026 month t, extract that month's structural shock triplet, build the
contemporaneous feature vector (including the five momentum features), load
the persisted v4c ranker models, and compare predicted sector rankings vs.
realized CARs at h ∈ {1, 3, 6}. Documented misses will distinguish three
failure modes: out-of-distribution regime, ambiguous shock classification,
and unmodelled confounds (tariff policy, central-bank intervention).
</div>

<hr>
<p class="small">
Generated by <code>src/report.py</code> from <code>outputs/run_manifest.json</code>
(milestones A through v4), <code>outputs/{shocks,features,targets,oof,shap,
report}</code> artifacts, and the code tree under <code>src/</code>.
</p>

</body>
</html>
""")


# --- Render ----------------------------------------------------------------

def render_report() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)

    # Build the new summary figure
    summary_path = REPORT_FIGURES / "four_experiment_summary.png"
    build_four_experiment_summary(summary_path)

    # Load all tables
    manifest = json.loads(RUN_MANIFEST.read_text()) if RUN_MANIFEST.exists() else {}
    git_sha = (
        manifest.get("milestone_c_v4_momentum", {}).get("git_sha")
        or manifest.get("milestone_d_shap", {}).get("git_sha")
        or manifest.get("milestone_a_var_shocks", {}).get("git_sha")
        or "unknown"
    )

    adf = _read(REPORT_TABLES / "adf_results.csv")
    sign_check = pd.read_csv(REPORT_TABLES / "sign_check_1990.csv", index_col=0) \
        if (REPORT_TABLES / "sign_check_1990.csv").exists() else pd.DataFrame()
    fevd = _read(REPORT_TABLES / "variance_decomposition.csv")
    v1 = _read(REPORT_TABLES / "oof_metrics.csv")
    v2 = _read(REPORT_TABLES / "oof_metrics_v2.csv")
    v4a = _read(REPORT_TABLES / "oof_metrics_v4a.csv")
    v4b = _read(REPORT_TABLES / "oof_metrics_v4b.csv")
    v3 = _read(REPORT_TABLES / "ranking_summary_v3.csv")
    v4c = _read(REPORT_TABLES / "ranking_summary_v4c.csv")

    # Headline numbers
    v4c_mean_sharpe = float(v4c["sharpe_ann"].mean()) if not v4c.empty else 0.0
    v3_mean_sharpe = float(v3["sharpe_ann"].mean()) if not v3.empty else 0.0
    v4c_h3_tstat = float(v4c[v4c["horizon"] == 3]["t_stat"].iloc[0]) if not v4c.empty else 0.0
    v4c_h6_tstat = float(v4c[v4c["horizon"] == 6]["t_stat"].iloc[0]) if not v4c.empty else 0.0

    ctx = {
        "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "git_sha": git_sha,
        "var_lags": VAR_LAGS,
        "var_variables": VAR_VARIABLES,
        "seed": SEED,

        # Figures
        "four_exp_b64": _b64_png(summary_path),
        "hist_decomp_b64": _b64_png(REPORT_FIGURES / "historical_decomposition.png"),
        "fevd_heatmap_b64": _b64_png(REPORT_FIGURES / "variance_decomposition_heatmap.png"),
        "v3_vs_v4c_b64": _b64_png(REPORT_FIGURES / "v3_vs_v4c_comparison.png"),
        "v4c_ls_cum_b64": _b64_png(REPORT_FIGURES / "v4c_long_short_cumulative.png"),
        "v4c_sector_sel_b64": _b64_png(REPORT_FIGURES / "v4c_sector_selection_freq.png"),
        "shap_global_bar_b64": _b64_png(REPORT_FIGURES / "shap_global_bar_h6.png"),
        "shap_heatmap_b64": _b64_png(REPORT_FIGURES / "shap_heatmap_h6.png"),
        "shap_dep_enrgy_b64": _b64_png(
            REPORT_FIGURES / "shap_dependence_eps_precaut_vix_regime_Enrgy_h6.png"
        ),
        "shap_waterfall_russia_b64": _b64_png(
            REPORT_FIGURES / "shap_waterfall_Enrgy_h6_2022-02_Russia.png"
        ),

        # Tables
        "adf_table_html": _build_adf_table(adf),
        "sign_check_html": _build_sign_check_table(sign_check),
        "fevd_table_html": _build_fevd_table(fevd),
        "v1_r2_pivot_html": _build_v1_r2_pivot(v1),
        "v2_r2_pivot_html": _build_v2_r2_pivot(v2),
        "v3_table_html": _build_v3_table(v3),
        "v4c_table_html": _build_v4c_table(v4c),
        "headline_arc_table_html": _build_headline_arc_table(v1, v2, v4a, v4b, v3, v4c),

        # Scalars
        "v4c_mean_sharpe": v4c_mean_sharpe,
        "v3_mean_sharpe": v3_mean_sharpe,
        "v4c_h3_tstat": v4c_h3_tstat,
        "v4c_h6_tstat": v4c_h6_tstat,

        # Param serializations
        "xgb_v1_json": json.dumps(dict(XGB_REG_PARAMS)),
        "xgb_pooled_json": json.dumps(
            {k: v for k, v in XGB_POOLED_PARAMS.items() if k != "early_stopping_rounds"}
        ),
    }

    html = TEMPLATE.render(**ctx)
    out = REPORT_DIR / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    out = render_report()
    print(f"Rendered report to {out}")
    print(f"  File size: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
