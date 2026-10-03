# Phase 5 Primary Production Model — Handoff Document

**Date:** 2026-10-03
**Status:** FROZEN — official Phase 5 primary production model
**Model ID:** `phase5_primary_xgb_3class`

## What this is

The exp_spatial 3-class configuration is promoted to the official Phase 5
production model per project directive (2026-10-03). All Phase 6–9 downstream
work must use this model and exactly this feature schema.

| Attribute | Value |
|---|---|
| Model family | XGBoost classifier (`multi:softprob`, `num_class=3`) |
| Hyperparameters | 200 trees, lr 0.05, max_depth 6, subsample 0.8, colsample_bytree 0.8, seed 42 (`models.config.XGBOOST_PARAMS`) |
| Execution policy | **Fitted single-threaded (`n_jobs=1`)** — see Determinism below |
| Encoded features | **178** (manifest: `feature_manifest.csv`) |
| Target | Low / Moderate / High — per-year LST tertiles, thresholds from training rows only |
| Dataset | 749,998 sampled pixels, July 2022–2026 |
| Locked holdout | Blocks [2, 9, 15, 23], rule `occupied[2::6]`, evaluated exactly once |
| **Locked accuracy** | **63.97%** (0.6396665686967289) |
| **Locked macro-F1** | **0.6315** (0.6314607179023212) |
| Full validation record | CV5 65.79% / LOBO 67.93% (acc.) — see `full_validation_record.json` |

## Frozen artifacts — `data/processed/phase5_production_3class/`

| File | Contents |
|---|---|
| `phase5_primary_xgb_3class.json` | Trained booster (XGBoost native JSON), 600 trees (200 × 3 classes) |
| `phase5_primary_xgb_3class_features.json` | Ordered 178-feature schema |
| `feature_manifest.csv` | Feature → source group (59 base, 52 morphology, 28 Tier-2 spatial means, 33 new spatial-context, 6 met) |
| `locked_validation.json` | Verified locked metrics, confusion matrix, per-class metrics, production tertile thresholds per year |
| `full_validation_record.json` | Complete exp_spatial validation record (CV5/LOBO/locked) |
| `production_model_metadata.json` | Machine-readable metadata/provenance |

Freeze/verify reproducibly via `scripts/freeze_phase5_production.py`.

## Feature schema (must be reproduced exactly for inference)

1. `data/processed/lulc_outputs/combined_urban_environmental_dataset.csv`
2. merge `phase5_tables/spatial_neighbourhood_features.csv` (row,col,year)
3. merge `phase5_tables/morphology_features.csv` (row,col,year)
4. merge `data/processed/experiments/exp_spatial_context_features.csv`
   (6 met covariates + 33 spatial-context features; built by
   `scripts/build_spatial_context_features.py`)
5. predictors = `PREDICTOR_VARS` (101) + 6 met + 33 spatial-context,
   encoded by `models.dataset.encode_predictors` → 178 columns

## Target & thresholds for downstream severity mapping

Production tertile thresholds (°C, from non-locked training blocks, per year —
these map predicted/observed LST to Low/Moderate/High):

| Year | T1 (Low→Mod) | T2 (Mod→High) |
|---|---|---|
| 2022 | 36.18 | 42.12 |
| 2023 | 35.19 | 40.47 |
| 2024 | 36.37 | 39.77 |
| 2025 | 38.10 | 41.16 |
| 2026 | 33.50 | 35.24 |

The model outputs classes directly; thresholds are documented for transparency
and for any LST-anchored post-processing.

## Verification (required before downstream use) — PASSED

`scripts/freeze_phase5_production.py` rebuilds the dataset with the
evaluator's own functions, refits XGBoost on the non-locked blocks, and
asserts the locked metrics match the experiment record to 1e-9:

```
[FREEZE] locked = {'accuracy': 0.6396665686967289, 'macro_f1': 0.6314607179023212}
[FREEZE] VERIFICATION PASSED: locked metrics reproduce the experiment record exactly
```

## Determinism note (important)

XGBoost's multi-threaded histogram construction is **not bit-reproducible
run-to-run** (XGBoost FAQ). A multi-threaded refit of this exact configuration
scored **64.2230%** locked vs the record **63.9667%** — a ±0.3 pp
thread-scheduling jitter, not a configuration difference. Diagnosis
(`scripts/diagnose_xgb_determinism.py`): single-threaded (`n_jobs=1`) fits are
fully deterministic, and the single-threaded fit reproduces the experiment
record **exactly** (identical predictions). The frozen production model was
therefore fitted with `n_jobs=1`; hyperparameters are unchanged. Downstream
re-fits for inference must use the saved booster (no retraining); any
retraining for comparison must state its thread policy.

## Lineage & preserved historical baselines (nothing overwritten)

| Generation | Locked acc. | Location | Status |
|---|---|---|---|
| 5-yr 4-class baseline | 35.15% | `data/processed/baseline_5yr_4class/` | historical |
| Tier 1+2 (4-class) | 49.59% | `data/processed/tier12_outputs/` | historical |
| 4-class XGB + LULC (frozen) | 50.02% | `data/processed/lulc_outputs/` | historical baseline |
| + met covariates (4-class) | 51.40% | `data/processed/experiments/` | experiment record |
| 3-class direct (145 feat) | 62.99% | `data/processed/experiments/` | experiment record |
| **3-class XGB + LULC + met + 33 spatial (178 feat)** | **63.97%** | **`data/processed/phase5_production_3class/`** | **PRODUCTION** |

The pre-2026-10-03 RF-C production lineage
(`data/processed/phase6/models/severity_rf_c.joblib`, built on the superseded
2-year/300k-row 4-class model) and all Phase 6/7 outputs derived from it are
**preserved unchanged** and flagged as superseded: they must be re-run against
this production model before being used for decisions.

## Standing rules carried forward

- No target-derived predictors (no LST statistics, no coordinates, no block IDs).
- Locked blocks [2, 9, 15, 23] remain reserved; no tuning against the holdout.
- The 75%-accuracy target discussion stays honest: 63.97% is a geographic
  holdout on an ordered 3-class task; per-class locked metrics are in
  `locked_validation.json` (Low F1 0.721, Moderate F1 0.515, High F1 0.658).
