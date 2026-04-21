# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

MIT Sloan 15.C51 (Spring 2026) Project #1 — Event Analysis track. The project extends Kilian & Park (2009) by combining a Kilian-style structural VAR (for oil-shock identification) with an XGBoost + SHAP supervised pipeline (for cross-sectional sector-return prediction). The 2026 assignment deliverable is a true out-of-sample comparison of model predictions vs. realized sector returns for 2026 shock events.

Authoritative specs live in the PDFs at the repo root — read them before making architectural decisions:
- `15.C51-2026-Project1.pdf` — assignment prompt and grading rubric.
- `research_overview.pdf` — research design, extensions over Kilian & Park, three-transmission-channels framework.
- `implementation_guide.pdf` — the 8-step pipeline (data → VAR → features → targets → overlap handling → XGBoost → SHAP → 2026 OOS). Includes Bloomberg/FRED/WRDS tickers and hyperparameter ranges.

## Current repo state

Data-and-spec only. No source code, no notebooks, no commits yet. The first commit will likely introduce the Python pipeline. There is no build / lint / test tooling to invoke until code is written — do not assume a framework; ask or propose one.

## The data panel (`master_panel_clean.csv`)

480 monthly rows, 1986-01-31 → 2025-12-31. Already-computed columns fall into groups:

- **VAR inputs (raw)** — `dprod` (% change global crude production), `kilian_rea` (Kilian real activity index), `WTI_spot`, `CPI`, `real_oil_price`, `real_oil_price_diff`, `world_oil_prod_tbpd`. These feed Step 2 of the pipeline; the **structural shocks themselves (supply / aggregate-demand / precautionary) are NOT in the panel** — extracting them via a 24-lag Cholesky-identified VAR is the first modeling step.
- **Macro regime features** — `VIX`, `VIX_filled`, `vix_is_proxy`, `vix_regime`, `LUACOAS`, `LUACOAS_filled`, `LF98OAS`, `FEDFUNDS`, `fed_regime`, `fed_regime_num`, `Recession`, `net_oil_price_3yr`, `net_oil_price_1yr`. VIX and HY pre-1990/1997 are backfilled with proxies — respect the `vix_is_proxy` / `hy_available` / `vix_available` flags when analyzing.
- **Oil context** — `oil_ret_1m`, `oil_ret_3m`, `oil_ret_12m`, `WTI_futures_BBG`.
- **Return targets** — 12 Fama-French industry returns in three forms: raw (`FF_NoDur` … `FF_Other`), abnormal (`FF_*_abn`), and excess (`FF_*_excess`), plus `FF_Mkt`, `FF_RF`.

### Fama-French 12 vs. GICS 11 — spec divergence

`research_overview.pdf` specifies 11 GICS sectors (from CRSP/WRDS), but the cleaned panel ships with **Fama-French 12 industry returns**. Treat this as an open question before building sector-level targets: either the FF-12 set is a deliberate substitution (document it, map FF↔GICS in the report), or GICS returns still need to be pulled. Don't silently proceed with one or the other — flag it.

## Pipeline invariants (non-negotiable)

These constraints come from the implementation guide and are easy to violate accidentally in feature/model code:

- **No look-ahead in features.** Every feature at month *t* must use only information available at the close of month *t*. The return window starts at *t+1*. Realized returns inside the prediction horizon must never enter `X`.
- **Walk-forward CV with a gap.** Use `sklearn.model_selection.TimeSeriesSplit(n_splits=5, gap=3)` (or larger gap for 12-month horizons) — never plain k-fold. The gap prevents target-window overlap between train and validation.
- **Keep contaminated months in the training set.** When a large shock (|shock| > 1.5 SD) sits within the prior 3 months, set `contamination_flag = 1` and include the observation. Dropping shrinks an already-small (~480-row) sample and discards the sequential-shock pattern the model is meant to learn.
- **Small-data regime.** ~480 observations × 44 (sector, horizon) target variables. Default to shallow trees (`max_depth` 2-3), strong regularization, and report **out-of-fold** R² / accuracy — never in-sample.
- **VAR identification order is fixed:** `[dprod, kilian_rea, real_oil_price]`, 24 lags, block-recursive Cholesky. The three resulting structural residuals, in order, are supply / aggregate-demand / precautionary shocks. Cross-check against Kilian's published historical decomposition (precautionary spikes in 1990, 2002-03, 2022; aggregate-demand positive 2003-07) before trusting downstream features.

## Target structure

44 models total: 11 sectors × 4 horizons (+1, +3, +6, +12 months). Target is the **cumulative abnormal return** `CAR(s, t, h) = cum_return(sector_s, t→t+h) − cum_return(market, t→t+h)`. The guide recommends starting with the 6-month horizon (best signal/noise per Kilian & Park's IRFs) and reporting both continuous-CAR and binary-direction versions.

## The 2026 out-of-sample test is the deliverable

Train strictly on pre-2026 data. For each 2026 month with a large shock: re-estimate the VAR through that month, extract the shock triplet, build the feature vector from contemporaneous regime data, predict +1/+3/+6 sector CARs, and document misses (out-of-distribution regime vs. ambiguous shock classification vs. unmodelled confound like tariff policy or central-bank intervention). This comparison table is a required deliverable.
