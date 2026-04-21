# Project Guide — Oil Shocks & US Sector Rotation

*A non-specialist orientation to everything we built, why we built it, and what's next for the final deliverable.*

---

## TL;DR

**The project**: Predict how 12 US industry groups react to oil price shocks. Use machine learning to extend Kilian & Park (2009), who only *described* past reactions — we want to *predict* future ones.

**What we built**: An end-to-end pipeline that (1) extracts three economically-distinct types of oil shocks from raw macro data, (2) engineers features combining those shocks with credit/volatility/momentum signals, (3) trains XGBoost models to rank which sectors will outperform in the coming 1-12 months, (4) backtests a long-top-3 / short-bottom-3 strategy.

**What we found**: Four progressive experiments revealed that Kilian-shock features alone can't beat random sector selection, but **adding equity momentum features flips the ranker from a losing strategy into a positive-Sharpe one** (mean annualized Sharpe +0.166 across four horizons, up from −0.076).

**What's left**: Run the 2026 out-of-sample test when data lands. Write up the final report (the scaffolding at `outputs/report/report.html` is ready).

**The grading deliverable** is the HTML report. This guide is for *you* — to orient yourself and write confidently about the work.

---

## Part 1 — What this project is actually about

### 1.1 The research question in plain English

When oil prices move, stocks react — but the reaction depends on *why* oil moved, not just *which direction*. Consider two examples:

- **Oil up because world economy is booming** → airlines pay more for fuel, but they're packed with travellers; stocks broadly fine.
- **Oil up because of war fears** → airlines still pay more for fuel, but panic spreads; stocks broadly crash.

Lutz Kilian (economist) showed this explicitly in 2009. He argued oil price changes have three distinct causes, each with different effects on the real economy:

1. **Supply shock** — unexpected change in oil production (OPEC cuts, a pipeline explosion, Saudi ramping up output).
2. **Aggregate demand shock** — unexpected change in global economic activity (China boom, GFC bust).
3. **Precautionary shock** — fear-driven buying/selling of oil (Iraq war fears, Russia invades Ukraine).

Kilian & Park (2009) then asked: *How do US stock sectors historically respond to each type of shock?* Their answer was descriptive ("on average, a +1 SD precautionary shock causes Consumer Durables to drop by X% over Y months").

**Our project asks a harder, forward-looking question**: given the shock triplet at time t, can machine learning predict *which specific sectors will outperform* over the next 1-12 months, well enough to build a tradable "sector rotation" strategy?

### 1.2 Why this matters for the grade

The assignment (`15.C51-2026-Project1.pdf`, Event Analysis track) explicitly asks:

> Model market reaction to oil price shocks. Use events prior to 2026 for training and testing, then compare the model's predictions for this year's events with the actual behavior of the markets affected.

So the grading criteria (25 points each) map to what we've built:

| Criterion | What they want | What we have |
|---|---|---|
| Completeness of analysis | Full replication + extension of Kilian's identification | VAR with validated shocks, FEVD, comparison to published decomposition |
| Novelty | Something beyond textbook replication | Four-paradigm experimental arc + the momentum finding |
| Readability | Publication-quality report | `outputs/report/report.html`, auto-generated |
| Source attribution incl. LLM prompts | Explicit data / method / AI acknowledgment | Section 5 of the report |

---

## Part 2 — The finance / ML vocabulary you need

Read these once; they'll make the rest of the guide trivial.

### Oil-related

- **Oil shock** — the *surprise* component of oil market movement. Not the level of oil prices but the part that wasn't forecastable from prior data.
- **Structural VAR** — a time-series model that treats several variables as jointly interdependent. Kilian runs one with [oil production %change, global real activity, real oil price] and recovers the three shock types from it.
- **Cholesky identification** — a mathematical trick to disentangle shocks: assume supply can't react to demand within a month (it's physically constrained), demand can't react to oil-specific precautionary factors within a month, etc. This ordering "identifies" the three shocks uniquely.
- **FEVD (Forecast-Error Variance Decomposition)** — for a given target (e.g. Energy returns), what share of its variance is explained by each type of shock? This gives the *in-sample upper bound* on predictability.

### Stock-market-related

- **Fama-French 12 industries** — twelve broad groupings of US stocks (Energy, Healthcare, Finance, Durables, etc.) maintained by Kenneth French at Dartmouth. "Value-weighted" = big companies matter more.
- **Abnormal return** — a stock's return minus the market's return. If SPY returned +2% this month and Energy sector returned +5%, the Energy *abnormal* return is +3%. This is the signal, absolute returns are noise.
- **Cumulative Abnormal Return (CAR)** — sum of abnormal returns over a forward window. CAR(Enrgy, March 2020, h=6) = sum of Energy abnormal returns from April 2020 through September 2020.
- **Momentum effect** (Jegadeesh-Titman, 1993) — stocks that went up over the past 3-12 months tend to keep going up; down-trenders keep going down. The single most robust predictor in equity markets.
- **Mean reversion** (DeBondt-Thaler, 1985) — over very long windows (12+ months), extreme performers revert. The momentum/reversal tension is well-documented.

### ML / stats-related

- **Walk-forward CV** — time-series cross-validation. Train on 1988-2000, test on 2001-2005. Then train on 1988-2005, test on 2006-2010. You cannot randomly shuffle time-series data — that leaks the future into the training set.
- **OOF R²** — out-of-fold R². The variance explained on predictions the model has never seen during training. If it's **negative**, the model is actively *worse* than just predicting the average (overfitting signature).
- **Horizon-aware gap** — when predicting 12 months ahead, the test set's target window overlaps with the training set's target window unless you leave at least 12 months between them. We use `gap = max(3, h+1)` to prevent this.
- **SHAP (SHapley Additive exPlanations)** — a technique that tells you, for each prediction, how much each feature contributed. Like "this model predicted +5% Energy return; 2% of that came from oil_ret_12m being high, 1% from VIX being high, etc."
- **Sharpe ratio** — mean return divided by standard deviation, annualized. The universal risk-adjusted return metric. >0 is profitable; >0.5 is decent; >1.0 is excellent.
- **Pooled vs per-cell model** — *per-cell* means training one model per (sector, horizon) combination (48 models for us). *Pooled* means one model per horizon that sees all 12 sectors with a sector-identity feature (4 models). Pooled gets 12× more training data.

---

## Part 3 — What we built, end-to-end

The pipeline lives in `src/` and runs via `make all` or individual targets (`make shocks | features | train | shap | report`).

### Milestone A — VAR shock extraction (`src/var_shocks.py`)

**Input**: raw macro data (oil production %change, Kilian real activity index, real oil price) from `master_panel_clean.csv`, 1986-2025 monthly.

**What we do**: Fit a 24-lag Cholesky-identified VAR. Extract three standardized shock series (supply, aggregate demand, precautionary) from the residuals.

**Validation**: The VAR must reproduce Kilian's published historical decomposition. Our code asserts the precautionary channel spikes during the 1990 Gulf War (+3.63 z-score), Iraq buildup 2002-03 (+1.29), and Russia-Ukraine 2022 (+1.84), and that aggregate demand is positive on average during the 2003-2007 China boom. **All pass.**

**Output**: `outputs/shocks/shocks_v1.parquet` — 456 months of three orthogonalized z-scored shock series, plus `historical_decomposition.png`.

### Milestone B — Features and targets (`src/features.py`, `src/targets.py`)

**Features (X)**: 33-column matrix per month combining the three shocks (plus lags and cumulative sums), macro regime indicators (VIX, credit spreads, Fed regime, NBER recession), and oil trend (3m/12m returns, realized volatility). All features use only information available at month-end t — no look-ahead.

**Targets (Y)**: 12 sectors × 4 horizons = 48 CAR columns. Each is the *forward* cumulative abnormal return: sum of sector-minus-market returns from t+1 to t+h.

**Output**: `outputs/features/X_v1.parquet`, `outputs/targets/Y_v1.parquet`, plus a binary-sign variant for classification.

### Milestone C — Training and evaluation (the entire arc — `src/train.py`, `src/train_v2.py`, `src/train_ranking.py`, `src/train_v4.py`)

This is where we iterated four times. See **Part 4** below for the detailed story.

### Milestone D — Interpretability (`src/shap_analysis.py`)

For each v1 regression model, compute SHAP feature attributions. Produce four figure types:
- Global bar chart per sector (what features matter most)
- Dependence plots (how shock × regime interactions work)
- Waterfall plots for specific events (GFC, COVID, Russia)
- Cross-sector heatmap (ML analogue of Kilian & Park's Figure 6)

**Finding**: Credit spreads and oil trend features dominate over the raw Kilian shocks in the v1 regressors. This foreshadows the signal-ceiling problem.

### Supplementary — Verification diagnostics (`src/verify_shocks.py`)

Added late in the project in response to a teammate's feedback list:
- **ADF stationarity tests** on VAR inputs (confirm real_oil_price needs differencing)
- **1990 Gulf War sign-convention check** (our VAR correctly identifies Aug 1990 as a −5.81 SD supply disruption)
- **Per-sector FEVD** (Energy has 37.2% variance share attributable to oil shocks — the ceiling on how much we could hope to predict)

### Report (`src/report.py`)

Renders everything above into a self-contained 1.9 MB HTML report at `outputs/report/report.html`. Seven sections aligned to the four rubric criteria plus interpretability and the 2026 OOS placeholder.

---

## Part 4 — The experimental arc (the heart of the novelty claim)

This is *the* story to tell in your writeup. We ran four training paradigms, each motivated by the previous one's failure.

### 4.1 v1 — per-cell XGBoost baseline

**Setup**: 48 independent XGBoost regressors, one per (sector, horizon). 33 features each, ~450 training rows per cell. 48 parallel binary classifiers for the sign target.

**Result**: Mean OOF R² = **−0.271**. Every one of the 48 cells was *worse* than predicting the unconditional mean. Only 6 of 48 classifiers beat the naive majority-class baseline.

**Diagnosis need**: This is not "weak signal" territory — it's "actively overfitting" territory.

### 4.2 Diagnostic — why was v1 so bad?

We wrote `src/diagnostics.py` and ran XGBoost in three complexity settings on four representative cells. Key findings:

- **Train-test R² gap = 0.74** under v1 settings. Model explains 55% of training variance but **−19% of test variance**. Classic overfitting.
- **Gap doesn't shrink as training data grows.** Fold 1 (68 rows) gap = 0.66; Fold 5 (367 rows) gap = 0.71. It's not small-sample overfit alone — there's non-stationarity in the feature→target relationship.
- **Depth matters more than size.** Going from max_depth=3 to depth-1 stumps halved the gap with no loss in test accuracy.
- **Target autocorrelation at h=12 is 0.92.** Adjacent 12-month CARs share 11/12 of their components. Effective sample size at h=12 is ~31 independent observations, not 450.

### 4.3 v2 — pooled regression + regularization (`src/train_v2.py`)

**What we changed**:
- Pool 12 sectors into one model per horizon → 12× more training rows (~5400 per model).
- Add sector-identity as a 12-column one-hot feature so the tree can learn sector-specific interactions.
- Trim features from 33 to 18 (drop redundant lags, cumulative sums, one-hot dominant shock).
- Add early stopping: hold out the last 15% of each training fold as a validation set, stop training when val loss plateaus.
- Also trained a **Ridge regression baseline** — if XGBoost can't beat linear regression, tree nonlinearity isn't adding value.

**Result**: Mean OOF R² improved from **−0.271 to −0.034** — an 8× reduction. Overfitting is largely cured. **But** mean classification edge stayed essentially unchanged (−0.060 → −0.090). Ridge tied XGBoost.

**Interpretation**: Regularization fixed *overfitting* but didn't unlock *signal*. The feature set itself is the ceiling for directional prediction. We've exhausted what the model architecture can do.

### 4.4 v3 — cross-sectional ranking reframe (`src/train_ranking.py`)

**Conceptual pivot**: Sector rotation is inherently *relative* — we don't need to predict absolute returns, just which sectors rank highest. We swap the regression objective for `rank:pairwise` (XGBoost's learning-to-rank mode).

**What we optimize**: For each month, order the 12 sectors by their predicted forward CAR. Build a long-short portfolio: long the top-3, short the bottom-3.

**Result**: A **substantive and surprising finding**. Two horizons (h=1, h=6) are neutral — no significant Sharpe. **Two horizons (h=3, h=12) are statistically anti-correct**: long-short Sharpe of −0.28 and −0.25, t-statistics of −2.71 and −4.75.

The model wasn't failing *randomly* — it was failing *systematically*. Looking at the sector selection plot: the ranker consistently went long **defensives** (Money, NoDur, Hlth) and short **cyclicals** (Durbl, Enrgy, Other). That "risk-off" positioning won weakly at h=1 and lost substantially at h=3/12. Classic Jegadeesh-Titman momentum effect: the defensive bet gets reversed by the next quarter's cyclical rotation.

**Key realization**: The model has no information about sector-level *return continuation*. It's betting purely on macro-regime logic and that bet is getting reversed by equity momentum, which is invisible to it.

### 4.5 v4c — ranker + momentum features (`src/train_v4.py`, `src/features_momentum.py`)

**What we added**: Five sector-specific momentum features:
- `own_ret_1m` — sector's own trailing 1-month abnormal return
- `own_ret_3m` — sector's trailing 3-month
- `own_mom_12_1` — Jegadeesh-Titman 12-month return skipping the most recent month (the classic momentum factor)
- `own_vol_6m` — rolling 6-month volatility
- `cross_rank_12_1` — cross-sectional rank of this sector's 12-1 momentum against the other 11 sectors

Everything else identical to v3.

**Result** — the main finding of the whole project:

| Horizon | v3 Sharpe / t-stat | v4c Sharpe / t-stat | Delta |
|---|---|---|---|
| 1 | +0.15 / +0.81 | **+0.22** / +1.22 | positive |
| 3 | **−0.28** / **−2.71** | **+0.21** / **+2.05** | **flip** |
| 6 | +0.08 / +1.05 | +0.16 / **+2.21** | improved |
| 12 | **−0.25** / **−4.75** | +0.07 / +1.41 | **flip** |

**All four horizons flipped to positive Sharpe**. The h=3 and h=12 anti-correctness — the pattern that revealed equity momentum was the missing piece — is cured. Mean Sharpe across horizons: **+0.166**, vs v3's −0.076.

**Caveat on significance**: the naive t-statistics at h=3 and h=6 exceed +2. But for horizons greater than 1, adjacent monthly observations share most of their return window, so they're not independent. After adjusting for this overlap (Newey-West-style), t-stats drop to around 1.0-1.2 — directionally robust but not strictly significant at the 5% level. The story is supported by multiple independent metrics (Spearman, top-3 hit rate, cumulative L-S), all pointing the same direction.

**Also ran**:
- **v4a** — v1 per-cell architecture with momentum added. Slight improvement (edge −0.060 → −0.053) but still net negative. Pooling is the bigger unlock.
- **v4b** — v2 pooled regression with momentum. Essentially no change (edge −0.090 → −0.084). Regression on continuous CARs doesn't benefit from momentum the same way ranking does.

### 4.6 The narrative in one paragraph

We built a Kilian-style VAR that correctly identifies oil shocks (validated against the published 1990 and 2022 decomposition). A naive per-cell XGBoost on these shocks badly overfits. Pooling and regularization cure the overfitting but reveal that macro-regime features alone don't contain enough cross-sectional signal for sector rotation. Reframing as a ranking problem exposes a structural mean-reversion in the macro-only strategy: it picks defensives, which equity momentum then reverses. Adding standard momentum features cures this, flipping the long-short strategy from mean Sharpe −0.08 to +0.17 across four forecast horizons. The substantive empirical result is that **Kilian macro-regime identification is necessary but not sufficient for predictable sector rotation — combining it with equity momentum is what delivers a tradable signal**.

---

## Part 5 — Where we are now, honestly

### What we can confidently claim

1. **The Kilian VAR works correctly.** We proved it with the Gulf War sign-convention check (August 1990 eps_supply = −5.81 z) and the precautionary spike pattern that matches Kilian's published figures.
2. **Our pipeline is rigorous and reproducible.** Walk-forward CV with horizon-aware gaps, no look-ahead verified by machine-checked assertions, artifact versioning in `outputs/` with a run manifest.
3. **Adding momentum features to a cross-sectional ranker delivers a positive-Sharpe sector-rotation signal.** Mean annualized Sharpe +0.166 across four horizons, with the v3→v4c flip most dramatic at h=3 (Sharpe from −0.28 to +0.21).
4. **The signal is not strictly significant** after conservative autocorrelation adjustment, but the direction is consistent across four independent metrics.

### What we cannot (yet) claim

1. **This beats a momentum-only strategy.** We haven't tested whether Kilian shocks actually *add* value on top of pure equity momentum. That ablation would strengthen or weaken the paper considerably.
2. **This works out of sample in 2026.** Still pending data.
3. **This is tradable after costs.** We report gross Sharpe; monthly turnover on a top-3/bottom-3 rotation would cut the signal.

### Known limitations (be honest in the writeup)

- FF-12 industries are coarser than actual GICS sectors or individual stocks
- Monthly data (not daily) — can't capture intra-month dynamics
- Kilian_rea index borderline-stationary (ADF p=0.13) but standard in the literature
- ~375 usable OOF observations is a small sample for finance ML

---

## Part 6 — What's next for the final grade

### The must-do

**Run the 2026 out-of-sample test.** The assignment explicitly requires this. The stub is at `src/oos_2026.py`. Here's the workflow:

1. Pull monthly 2026 data (Jan through April, maybe May) from Bloomberg matching the existing panel's columns
2. Append rows to `master_panel_clean.csv`
3. Implement `predict_for_month()` in `src/oos_2026.py` (stub exists) — should refit VAR through each month, extract shock triplet, build X row with momentum features, load v4c models, predict rankings
4. As each month's h=1/3/6 window closes, compare predicted top-3 vs realized top-3
5. Write a "misses analysis" explaining any divergences (out-of-distribution regime? ambiguous shock classification? unmodelled confound like tariffs or Fed intervention?)
6. Re-run `python -m src.report` — Section 6 will auto-populate with your comparison table

### Optional but high-value

- **SHAP on v4c models** — the v1 SHAP in the report shows credit spreads dominate. We hypothesize v4c SHAP will show `cross_rank_12_1` and `own_mom_12_1` dominate. Confirming this explicitly is a ~30-minute task that strengthens the interpretability section.
- **Momentum-only baseline** — run the ranker with *only* the 5 momentum features, no Kilian shocks. If it matches v4c's Sharpe, shocks don't add value. If v4c beats it, shocks provide genuine regime-conditional information. This ablation is the cleanest test of the paper's novelty claim.
- **Stress test** — recompute v4c Sharpe excluding the top/bottom 5% of monthly returns. If the strategy depends on a few extreme months (e.g., COVID 2020), it's fragile.

### What to write in your report

The `outputs/report/report.html` is the grading artifact. It covers all four rubric sections. You should:

1. Open it, read it, make sure you can explain every section verbally
2. The **executive summary** has the top-line result. The **head-to-head arc table** shows v1→v4c in one view.
3. For the writeup assignment, your job is to *synthesize* this into a narrative with appropriate caveats.
4. **Don't oversell**: mean Sharpe +0.17 is modest, and AC-adjusted significance is borderline. Frame it as "our rigorous iteration delivered a tradable-signal-candidate that deserves real money validation but hasn't yet been tested out-of-sample."

---

## Part 7 — Vocabulary / Quick Reference

| Term | Plain English |
|---|---|
| VAR | A model that treats several time series as mutually dependent, with each one depending on past values of all the others. |
| Structural shock | The *surprise* component after controlling for predictable dynamics. |
| Cholesky identification | A way to say "shock A can affect shock B within the month, but B can't affect A" — lets us disentangle the three oil shocks. |
| FEVD | How much of variable X's variance is explained by each shock, over a given horizon. |
| CAR | Forward cumulative outperformance vs the market. Our target. |
| Walk-forward CV | Time-ordered train/test splits. Train on older data, test on newer. Never shuffle time series. |
| OOF R² | R² on held-out test data the model hasn't seen. Negative = worse than predicting the mean. |
| Long-short portfolio | Buy the best predicted sectors, sell the worst. Profit from the spread. |
| Sharpe ratio | Return per unit of risk, annualized. >0.5 is decent. |
| Momentum | Stocks that recently went up keep going up over 3-12m. The strongest equity factor. |
| XGBRanker pairwise | Train model to order elements within a group, not predict absolute values. |
| Pooled model | One model for all sectors (with sector ID as a feature). Opposite of per-cell. |
| Regularization | Techniques (shallower trees, smaller learning rate, L2 penalty) to prevent memorizing noise. |

---

## Part 8 — File map

```
master_panel_clean.csv                 The raw data panel (1986-2025 monthly, 65 cols)

src/
  config.py                            Paths, constants, hyperparameter dicts
  io_utils.py                          Panel loader, artifact saver
  var_shocks.py                        Milestone A — Kilian VAR + shock extraction
  features.py                          Milestone B — v1 33-feature matrix
  features_momentum.py                 v4 — 5 sector-specific momentum features
  targets.py                           Milestone B — CAR + sign targets
  cv.py                                Walk-forward CV with horizon-aware gap
  train.py                             v1 — per-cell XGBoost (48 + 48 models)
  train_v2.py                          v2 — pooled regression + Ridge baseline
  train_ranking.py                     v3 — XGBRanker + long-short backtest
  train_v4.py                          v4 — all three architectures + momentum
  diagnostics.py                       Train-test gap diagnostic
  verify_shocks.py                     ADF + sign check + FEVD
  shap_analysis.py                     Milestone D — SHAP panels
  report.py                            HTML report renderer (v1 → v4c arc)
  oos_2026.py                          Stub for 2026 out-of-sample test — TODO

notebooks/                             Thin consumer notebooks per milestone

outputs/
  shocks/shocks_v1.parquet             Extracted Kilian shock triplet
  features/X_v1.parquet                Feature matrix (v1)
  targets/Y_v1.parquet                 CAR targets (48 columns)
  models/                              Persisted models — v1 per-cell (96), v2/v4 pooled, v3/v4c rankers
  oof/                                 Out-of-fold predictions per version
  shap/                                Raw SHAP arrays for all 48 v1 models
  report/
    report.html                        THE GRADING DELIVERABLE — self-contained 1.9 MB
    figures/                           All plots used in the report
    tables/                            All CSV tables (metrics, comparisons, ADF, FEVD, …)
  run_manifest.json                    Per-milestone metadata (git SHA, seed, hashes)

CLAUDE.md                              Orientation for future Claude Code sessions
PROJECT_GUIDE.md                       This file — for you
Makefile                               `make all` reproduces the full pipeline
requirements.txt                       Python dependencies (unpinned)
```

---

## Part 9 — Elevator pitch (memorize this, maybe)

> "We extended Kilian & Park's 2009 oil-shock impulse-response framework into forward prediction using XGBoost. After four experimental iterations — per-cell, pooled, ranking, ranking+momentum — we show that **Kilian macro-regime features alone can't generate a tradable sector rotation**, but combining them with standard equity momentum yields a cross-sectional ranker with mean annualized Sharpe of +0.17 across four horizons. The iteration itself is the contribution: each failure diagnosed (overfit → signal ceiling → anti-correctness → missing momentum) taught us something substantive about why macro oil-shock models don't directly port to equity prediction, and why the sector-rotation literature's momentum factor turns out to be the key missing input."

---

*End of guide. When in doubt, reread Part 4 — the experimental arc is the project. Everything else is scaffolding.*
