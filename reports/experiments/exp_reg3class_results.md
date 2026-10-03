# Experiment: Continuous LST → 3-Class Severity (exp_reg_to_3class) — Final Report

**Date:** 2026-10-03
**Status:** COMPLETE (run log `exp_reg3class_eval.log`, EXIT: 0, ~57 min)

## Design

Same starting configuration as the direct 3-class experiment: XGBoost/RF +
LULC + 6 met covariates (145 encoded features), same 5-year dataset
(749,998 rows), same spatial blocks, same sampling, same locked holdout
**[2, 9, 15, 23]**, locked evaluated exactly once.

Only the model path changes: instead of a 3-class classifier, a **continuous
LST regressor** is trained on each training portion; its predictions are
converted to Low/Moderate/High with that same training portion's per-year
LST tertiles. No coordinates, block IDs, or target-derived features.

## Headline — XGBoost

| Metric | Direct 3-class (baseline) | Regression → threshold | Delta |
|---|---|---|---|
| CV5 accuracy | 63.42% | 59.62% | **−3.80 pp** |
| CV5 macro-F1 | 0.6225 | 0.5933 | −0.0292 |
| LOBO accuracy | 66.24% | 62.22% | **−4.02 pp** |
| LOBO macro-F1 | 0.6283 | 0.5829 | −0.0454 |
| **Locked accuracy** | **62.99%** | **60.45%** | **−2.54 pp** |
| **Locked macro-F1** | **0.6186** | **0.5966** | **−0.0220** |

Random Forest agrees: locked 61.15% / 0.6033 vs direct-3class RF 62.33% / 0.6141.

## Locked holdout — confusion matrix (XGBoost, rows = true)

|  | Pred Low | Pred Moderate | Pred High | Support |
|---|---|---|---|---|
| **True Low** | 47,948 | 18,123 | 2,014 | 68,085 |
| **True Moderate** | 16,451 | 27,148 | 6,942 | 50,541 |
| **True High** | 4,351 | 15,366 | 21,570 | 41,287 |

## Locked per-class metrics (XGBoost)

| Class | Precision | Recall | F1 | (direct 3-class F1) |
|---|---|---|---|---|
| Low | 0.697 | 0.704 | 0.701 | 0.715 |
| Moderate | 0.448 | 0.537 | 0.488 | 0.496 |
| High | 0.707 | **0.522** | 0.601 | **0.645** |

## Continuous regression metrics (XGBoost)

| Stage | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| CV5 | 2.68 | 3.58 | 0.504 |
| LOBO | 2.43 ± 0.37 | 3.34 ± 0.49 | 0.481 ± 0.143 |
| **Locked** | **2.52** | **3.35** | **0.487** |

These match the earlier Tier-4 regression experiment (locked R² ≈ 0.42,
MAE ≈ 2.66 °C): adding met covariates did not transform continuous-LST
predictability. A ~2.5 °C MAE against tertile boundaries that are sometimes
only ~3 °C apart (e.g. 2026: T1 33.50 / T2 35.24) inherently blurs classes.

## Thresholds (identical to exp_3class — same training sets, sanity check)

2022: 36.18 / 42.12 · 2023: 35.19 / 40.47 · 2024: 36.37 / 39.77 ·
2025: 38.10 / 41.16 · 2026: 33.50 / 35.24 (°C).
Full per-fold/per-block lists in
`data/processed/experiments/exp_reg3class_reg3class_geoeval.json`.

## Interpretation

1. **Direct 3-class classification wins on the identical target and protocol.**
   The regression path loses ~2.5–4 pp at every validation level, for both
   model families. The classes are defined by within-year LST ranks; a
   classifier models those ranks directly (and uses class weights), while the
   regressor must first nail absolute LST across years and blocks — half of
   that variance (locked R² ≈ 0.49) is unexplained — and then thresholding
   adds a second error source.
2. **The failure mode is visible in the confusion matrix:** High recall
   collapsed 62.7% → 52.2% — the regressor shrinks predictions toward the
   mean, so 15,366 true-High pixels (37%) land in Moderate.
3. **Same-target comparison is fair:** unlike the 4-class↔3-class comparison,
   the target here is identical to exp_3class — only the model path differs.
   This is a genuine head-to-head, and direct classification is the keeper.

## Conclusion

**Keep the direct 3-class XGBoost model (62.99% / 0.6186 locked).**
Regression → tertile-thresholding is a strictly worse path (−2.54 pp locked
accuracy). The continuous LST regression remains scientifically useful
(MAE 2.52 °C, R² 0.49) as a *different* product, but it should not replace
the severity classifier.

## Artifacts

- Results: `data/processed/experiments/exp_reg3class_reg3class_geoeval.json`
- Log: `data/processed/experiments/exp_reg3class_eval.log`
- Baseline: `data/processed/experiments/exp_3class_3class_geoeval.json`
- Evaluator: `scripts/geographic_holdout_eval.py` (task `reg3class`)

## Frozen assets (untouched)

`data/processed/lulc_outputs/`, `baseline_5yr_4class/`, `tier12_outputs/`,
primary 4-class model and validation protocol: **not modified**.
