# Actual Models Inventory in Repository

## File Discovery
All OOF (out-of-fold) prediction files found in: `outputs/oof/`

---

## Models Loaded by Notebook

### Main Progression (v1 → v4c → v5)

| Model | OOF File | Status | Description |
|-------|----------|--------|-------------|
| **v1** | `oof_reg_v1.parquet` | ✅ | Baseline regression (per-cell) |
| **v2** | `oof_pooled_ridge_v2.parquet` | ✅ | Pooled regression + Ridge regularization |
| **v3** | `ranker_scores_v3.parquet` | ✅ | Ranking model (LGB classifier on ranks) |
| **v4c** | `ranker_scores_v4c.parquet` | ✅ | **PUBLISHED WINNER** - Ranker + momentum features |
| **v5** | `ranker_scores_v5.parquet` | ✅ | Post-v4c ranker variant (confirmed in repo) |

### Alternative v4 Variants

| Model | OOF File | Status | Description |
|-------|----------|--------|-------------|
| **v4a** | `oof_reg_v4a.parquet` | ✅ | v4 Regression variant (alternative to ranker) |
| **v4b** | `oof_pooled_xgb_v4b.parquet` | ✅ | v4 XGBoost variant (pooled) |

### Related Data Files

| File | Model | Description |
|------|-------|-------------|
| `long_short_returns_v3.parquet` | v3 | Long-short monthly returns |
| `long_short_returns_v4c.parquet` | v4c | Long-short monthly returns |
| `long_short_returns_v5.parquet` | v5 | Long-short monthly returns |
| `ranking_summary_v5.csv` | v5 | Summary statistics for v5 rankings |

---

## Summary

✅ **7 Total Models in Evaluation:**
- **5 Main Progression:** v1 → v2 → v3 → v4c → v5
- **2 v4 Variants:** v4a (regression), v4b (XGBoost)

✅ **Status:** All models have OOF predictions available and loadable

✅ **v5 Confirmed:** YES - v5 is an actual model in the repo with:
- `ranker_scores_v5.parquet` OOF predictions
- `long_short_returns_v5.parquet` portfolio returns
- `ranking_summary_v5.csv` summary statistics

---

## Notebook Implementation

**The notebook will evaluate all 7 models** across 4 horizons (h=1, 3, 6, 12):
- Total evaluation records: 7 models × 4 horizons = **28 records**
- Metrics computed: Sharpe, IC, t-stat, hit rate, top-3 overlap

**Data Availability:**
- ✅ OOF files: All 7 models loaded successfully
- ⚠️ Target data: Requires `h_data.parquet` + `targets_h{1,3,6,12}.parquet` to compute metrics
- ✅ Report baseline: v4c metrics hard-coded from published report for validation

---

## File Loading Map

```
Notebook Cell: Load Baseline Models (Cell 7)
├── v1  ← outputs/oof/oof_reg_v1.parquet
├── v2  ← outputs/oof/oof_pooled_ridge_v2.parquet
├── v3  ← outputs/oof/ranker_scores_v3.parquet
├── v4c ← outputs/oof/ranker_scores_v4c.parquet (REPORT BASELINE)
└── v5  ← outputs/oof/ranker_scores_v5.parquet

Notebook Cell: Load Alternative Variants (Cell 9)
├── v4a ← outputs/oof/oof_reg_v4a.parquet
└── v4b ← outputs/oof/oof_pooled_xgb_v4b.parquet
```

---

## Why This Mapping?

For each model version, the notebook selects the **canonical/best-performing variant**:

- **v1:** Used `oof_reg_v1.parquet` (regression is the baseline approach)
- **v2:** Used `oof_pooled_ridge_v2.parquet` (Ridge regularization is the main advancement)
- **v3:** Used `ranker_scores_v3.parquet` (ranking model for v3)
- **v4c:** Used `ranker_scores_v4c.parquet` (published winner - ranker with momentum)
- **v5:** Used `ranker_scores_v5.parquet` (confirms v5 exists and continues ranker approach)

**v4 Variants** (v4a, v4b) are treated as alternatives to understand what happened:
- v4a: What if we stuck with regression instead of ranker?
- v4b: What if we used XGBoost instead of LGB?

---

## Next Steps

When target data becomes available (`h_data.parquet`, `targets_h{1,3,6,12}.parquet`):

1. Place in: `master_panel_clean.csv/` subdirectory
2. Re-run notebook Cell 7 onwards
3. Notebook will automatically:
   - Compute metrics for all 7 models
   - Validate v4c against report baseline
   - Generate comparison tables
   - Create visualizations
   - Export CSV/JSON results

---

**Last Updated:** 2025-02-28  
**Notebook Status:** ✅ Ready for execution  
**Models Verified:** ✅ All 7 OOF files present and loadable
