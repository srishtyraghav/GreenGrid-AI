# Tier 1+2 and Tier 4 Results — Consolidated Report

**Date:** 2026-10-03 · **Series:** July 2022–2026, Delhi NCT, 30 m grid
**Validation protocol (identical for every experiment):** 5-fold adjacent-block GroupKFold CV; leave-one-block-out over all occupied blocks; deterministic locked geographic holdout (`occupied_block_ids[2::6]`, evaluated exactly once per experiment, never tuned on). Targets computed from training rows only (per-year quartiles / medians).

---

## 1. Primary task — 4-class per-year-quartile heat severity, PIXEL level

**This is the project's actual objective; only changes here count as "model improvement."**

| Model (XGBoost) | CV5 acc / F1 | LOBO acc / F1 | **Locked acc / F1** |
|---|---|---|---|
| Baseline 5-yr (90 feat, paired 445k sample) | 43.03 / 0.405 | 40.85 / 0.325 | 35.15 / 0.324 |
| Tier 1+2 (100 feat, NDMI/MNDWI/BSI/NDRE + unpaired 750k) | 50.26 / 0.482 | 51.63 / 0.476 | 49.59 / 0.473 |
| **+ LULC (Dynamic World `lulc_class`, 101 feat)** | **50.63 / 0.485** | **52.38 / 0.480** | **50.02 / 0.477** |

- Tier 1+2 over baseline: **+7.6 pt locked** (35.2 → 49.6). LULC adds **+0.4 pt** more (locked crosses 50%).
- Random Forest agrees (locked 36.1 → 48.0 → 48.5).
- Tier 1+2 = Tier 1 features (NDMI, MNDWI, BSI, NDRE from bands already on disk) + Tier 2 unpaired per-year sampling (150k/yr). LULC = Dynamic World 10 m land cover, nearest-valid gap-filled for monsoon holes, 2024 = Jun–Aug majority (provenance deviation).

## 2. Tier 4 experiments — DIFFERENT TARGETS/GRAINS

> ⚠ These answer different questions. Their numbers must **not** be presented as improving the primary 4-class pixel model.

### 2.1 Continuous LST regression (XGBoost)
| Validation | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| CV5 | 2.754 | 3.698 | 0.473 |
| LOBO | 2.651 | 3.610 | 0.389 |
| Locked | 2.664 | 3.556 | 0.421 |

Features explain ~40–47% of LST variance on held-out geography. Useful for Phase 8–9 (cooling estimation needs continuous LST), but regression R² is modest — LST has substantial atmospheric noise the surface features don't capture.

### 2.2 Binary heat classification (above/below per-year median) — XGBoost
| Validation | Accuracy | Macro-F1 |
|---|---|---|
| CV5 | 74.37% | 0.7362 |
| LOBO | 75.39% | 0.7221 |
| **Locked** | **75.28%** | **0.7406** |

**~75% on unseen geography** — meets the 75% figure, but honestly labeled: a 2-class target with one boundary is intrinsically easier than the 4-class severity task. It is a different claim, not a better model.

### 2.3 Spatial/block aggregation
Aggregating pixel OOF predictions to the 5×5-block grain (only ~95 block-year units): block **majority-vote** accuracy 58.0% (+7.7 pt over pixel 50.3%, but n=95 ⇒ ±10 pt noise); block **mean-score** quartile accuracy 45.0% (−5.3 pt — coarse quartiles on tiny n). Aggregation at this grain is **not a reliable path** to a headline number; district-grain evaluation would have even fewer units.

## 3. Honest positioning
- The primary 4-class pixel model improved from 35.1% → **50.0% locked geographic holdout** (+15 pt) through features + sampling + LULC. Its CV/LOBO/locked spread is tight — real generalization.
- The 75% target is met **only** by the binary (2-class) target at the same pixel grain — a different, coarser claim. The 4-class pixel task remains bounded by quartile-boundary noise; further gains there would need new information (meteorological covariates, thermal sharpening) rather than more tuning.
- No leakage: `lst_C`/coordinates/blocks never in predictors; locked holdout evaluated once per experiment; no iteration on it.

## 4. Validation status
- Phase 5 validator on the final LULC run: **56 PASS / 0 FAIL / 14 WARN** (WARNs = the intentionally skipped generalization-audit tables).
- Phase 3 (87/0/9) and Phase 4 (58/0/0) re-validated on the 5-year series earlier.

## 5. Artifacts index
| Artifact | Path |
|---|---|
| Comparison report (baseline vs Tier 1+2) | `reports/experiments/baseline_vs_tier12_comparison.md` |
| Baseline geoeval | `data/processed/experiments/baseline_5yr_4class_classification_geoeval.json` |
| Tier 1+2 geoeval | `data/processed/experiments/tier12_5yr_classification_geoeval.json` |
| LULC geoeval | `data/processed/experiments/lulc_5yr_classification_geoeval.json` |
| Tier 4 regression | `data/processed/experiments/tier4_regression_regression_geoeval.json` |
| Tier 4 binary | `data/processed/experiments/tier4_binary_binary_geoeval.json` |
| Tier 4 aggregation | `data/processed/experiments/tier4_aggregation_summary.json` + `_blocks.csv` |
| Uniform evaluator | `scripts/geographic_holdout_eval.py` |
| Snapshots | `data/processed/baseline_5yr_4class/`, `tier12_outputs/`, `lulc_outputs/` |
| Final model (primary task) | `data/processed/phase5/` (XGBoost, 101 predictors) |
