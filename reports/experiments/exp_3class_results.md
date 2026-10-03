# Experiment: 3-Class UHI Severity Target (exp_3class) — Final Report

**Date:** 2026-10-03
**Status:** COMPLETE (run log `exp_3class_eval.log`, EXIT: 0, ~76 min)

## What changed

Only the target formulation. Instead of per-year LST **quartiles** (4 classes),
the target is per-year LST **tertiles** (3 classes: Low / Moderate / High).
Thresholds are computed **per year, from training rows only**, and applied to
validation/locked rows — identical leakage-safe machinery to all prior runs.

Everything else identical to the current best configuration:
XGBoost + LULC + 6 met covariates (145 encoded features), same 5-year dataset
(749,998 rows), same spatial blocks, same sampling, same XGBoost params,
same 5-fold adjacent-block CV, same LOBO, same locked blocks **[2, 9, 15, 23]**,
locked evaluated exactly once. Alignment audit PASS (`reports/experiments/alignment_audit.md`).

## Headline — XGBoost (same family as the 51.40% baseline)

| Metric | 4-class +Met (frozen best) | 3-class (this exp) | Delta |
|---|---|---|---|
| CV5 accuracy | 53.85% | **63.42%** ± 5.12 | +9.57 pp |
| CV5 macro-F1 | 0.5194 | **0.6225** ± 0.0442 | +0.1031 |
| LOBO accuracy | 55.16% | **66.24%** ± 6.11 | +11.08 pp |
| LOBO macro-F1 | 0.5113 | **0.6283** ± 0.0619 | +0.1170 |
| **Locked accuracy** | **51.40%** | **62.99%** | **+11.59 pp** |
| **Locked macro-F1** | **0.5068** | **0.6186** | **+0.1118** |

Random Forest agrees: locked 62.33% / 0.6141 (CV5 64.70%, LOBO 66.48%).

## Locked holdout — confusion matrix (XGBoost)

Rows = true class, columns = predicted (n = 159,913 locked pixels):

|  | Pred Low | Pred Moderate | Pred High | Support |
|---|---|---|---|---|
| **True Low** | 50,390 | 14,367 | 3,328 | 68,085 |
| **True Moderate** | 16,286 | 24,453 | 9,802 | 50,541 |
| **True High** | 6,132 | 9,266 | 25,889 | 41,287 |

## Locked holdout — per-class metrics (XGBoost)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.692 | 0.740 | 0.715 | 68,085 |
| Moderate | 0.509 | 0.484 | 0.496 | 50,541 |
| High | 0.663 | 0.627 | 0.645 | 41,287 |

Moderate is the hardest class (typical for the middle ordered class): it is
where Low/High confusions land. High recall 62.7% vs the 4-class model's
high+severe recall of 49.7% on the same locked blocks.

## Exact training-fold/year LST thresholds (tertiles, °C)

**Locked evaluation** (train = all non-locked blocks):

| Year | T1 (Low→Moderate) | T2 (Moderate→High) |
|---|---|---|
| 2022 | 36.18 | 42.12 |
| 2023 | 35.19 | 40.47 |
| 2024 | 36.37 | 39.77 |
| 2025 | 38.10 | 41.16 |
| 2026 | 33.50 | 35.24 |

Per-fold CV thresholds (5 folds × 5 years) and per-block LOBO thresholds are
stored in full in `data/processed/experiments/exp_3class_3class_geoeval.json`
under `thresholds.cv5` / `thresholds.lobo`. Fold-to-fold threshold variation is
small (e.g. 2022 T1 spans 35.08–36.75 °C), confirming the threshold estimate
is stable across training sets.

## Class balance per year (locked-train target definition)

| Year | n | Low | Moderate | High | (locked-test split) |
|---|---|---|---|---|---|
| 2022 | 150,000 | 35.2% | 32.8% | 32.0% | 44.0 / 30.1 / 25.9 |
| 2023 | 150,000 | 38.2% | 34.1% | 27.7% | 55.8 / 37.0 / 7.2 |
| 2024 | 150,000 | 32.1% | 33.1% | 34.8% | 27.9 / 32.2 / 39.9 |
| 2025 | 149,999 | 38.1% | 33.0% | 28.9% | 51.9 / 31.8 / 16.3 |
| 2026 | 149,999 | 32.8% | 31.9% | 35.3% | 30.6 / 26.0 / 43.4 |

Tertiles are near-balanced on the full years by construction; the locked-test
columns show the strong geographic distribution shift the model must
generalize across (e.g. block set is 55.8% Low in 2023 but 39.9% High in 2024).

## Interpretation — honest labeling

1. **The 3-class task is easier than the 4-class task.** Chance level is
   33.3% vs 25%, the top tertile absorbs the old High+Severe band, and the
   moderate band is wider. **62.99% locked on 3 classes must NOT be presented
   as the primary model improving from 51.40%** — it is a different target.
2. **Chance-adjusted view:** 4-class +Met = +26.4 pp over chance on locked;
   3-class = +29.7 pp over chance. A modest genuine gain remains after this
   adjustment, consistent across CV5/LOBO/locked and both model families.
3. **High-class recall improved 49.7% → 62.7%** on identical locked blocks —
   for heat-risk screening ("is this pixel in the hottest third?"), the 3-class
   formulation is substantially more useful at the same geographic validation.
4. Moderate-class F1 (~0.50 locked) remains the weak point — the middle
   tertile is spectrally ambiguous.

## Artifacts

- Results: `data/processed/experiments/exp_3class_3class_geoeval.json`
  (per-fold CV details, per-block LOBO details/pooled matrix, all thresholds)
- Log: `data/processed/experiments/exp_3class_eval.log`
- Evaluator: `scripts/geographic_holdout_eval.py` (task `3class`; all other
  tasks' code paths unchanged)

## Frozen assets (untouched)

`data/processed/lulc_outputs/`, `baseline_5yr_4class/`, `tier12_outputs/`,
primary 4-class model and validation protocol: **not modified**.
