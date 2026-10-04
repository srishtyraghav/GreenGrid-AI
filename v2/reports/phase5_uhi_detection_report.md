# V2 Phase 5 — UHI Detection Report (3-class, per-year LST tertiles)

Phase 5 detects relative urban-heat-island severity classes (Low / Moderate /
High) per 30 m pixel from the frozen 178-feature W4 (May 1 – Jun 30) stack,
years 2022–2026, 750,000 sampled rows (150k/yr). Protocol is frozen from V1's
production 3-class model: per-year LST tertiles computed from **training rows
only** inside every split (`searchsorted side="right"` boundary), 5-fold
adjacent-block CV, LOBO, and a deterministic final refit with the locked
geographic holdout (blocks {2, 9, 15, 23}) evaluated **exactly once**.

**DESIGNATION (user decision, Phase 5 FROZEN): the Random Forest is the
official V2 Phase 5 primary production model.** The XGB run is the preserved
historical baseline. See `phase5_production_model.md` for the freeze record.

> **Caveat:** all numbers below are W4 (May–Jun) data. V1's historical 63.97%
> was July-window — **not season-comparable**.

## 1. XGB run — `phase5_primary_xgb_v2` (now the historical baseline)

Frozen replication of V1 `phase5_primary_xgb_3class` params: `n_estimators=200,
learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
objective=multi:softprob, eval_metric=mlogloss, random_state=42, num_class=3`
(final refit single-threaded `n_jobs=1`; XGBoost threaded histograms are not
bit-reproducible). Full record: `v2/data/phase5/` (cv/lobo/oof/locked
artifacts, manifest, model JSON). Run wall 1223.2s (load 1.2 / CV 173.0 /
LOBO 891.9 / refit 154.2 / locked 1.3). Verification gate `v2.phase5.verify_run`:
10/10 PASS incl. bit-identical locked-prediction reload.

| stage | accuracy | macro-F1 |
|---|---|---|
| CV5 (mean±std) | 0.71853 ± 0.04776 | 0.70425 ± 0.03398 |
| LOBO (20 fits) | 0.73919 ± 0.06605 | 0.65738 ± 0.08099 |
| **Locked (once, n=140,088)** | **0.68770** | **0.67247** |

Per-fold CV: 0.77778 / 0.74415 / 0.63594 / 0.70300 / 0.73177.
Locked per-class: Low 0.7775/0.6376/0.7006 (45,049) · Moderate
0.5407/0.5126/0.5262 (41,644) · High 0.7268/0.8666/0.7906 (53,395);
confusion [[28721,12557,3771],[6676,21345,13623],[1544,5578,46273]].

## 2. RF comparison run — `v2/data/phase5/rf_compare/` (frozen record)

Same protocol machinery (`v2.phase5.train` loaders/folds/targets/scoring),
estimator swapped to RandomForest with pre-specified verbatim V1 params
(`v1/src/models/config.py::RANDOM_FOREST_PARAMS`, `RANDOM_SEED=42`):
`n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2,
max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1`.
Locked evaluated exactly once under the same banner. Run wall 3860.2s
(load 0.6 / CV 519.4 / LOBO 3206.5 / refit 131.2 / locked 0.8).

| stage | accuracy | macro-F1 |
|---|---|---|
| CV5 (mean±std) | 0.71653 ± 0.04189 | 0.70222 ± 0.02968 |
| LOBO (20 fits) | 0.74191 ± 0.06077 | 0.66378 ± 0.07419 |
| **Locked (once, n=140,088)** | **0.69902** | **0.68512** |

Per-fold CV: 0.76348 / 0.73872 / 0.64043 / 0.70855 / 0.73145.
Locked per-class: Low 0.8345/0.6233/0.7136 (45,049) · Moderate
0.5502/0.5417/0.5459 (41,644) · High 0.7226/0.8856/0.7958 (53,395);
confusion [[28079,12901,4069],[4999,22560,14085],[569,5541,47285]].

## 3. Side-by-side (XGB baseline vs RF, identical protocol & folds)

| metric | XGB (baseline) | RF (primary) |
|---|---|---|
| CV5 accuracy | 0.71853 ± 0.04776 | 0.71653 ± 0.04189 |
| CV5 macro-F1 | 0.70425 ± 0.03398 | 0.70222 ± 0.02968 |
| LOBO accuracy | 0.73919 ± 0.06605 | 0.74191 ± 0.06077 |
| LOBO macro-F1 | 0.65738 ± 0.08099 | 0.66378 ± 0.07419 |
| **Locked accuracy** | **0.68770** | **0.69902** |
| **Locked macro-F1** | **0.67247** | **0.68512** |
| Locked per-class F1 (L/M/H) | 0.7006 / 0.5262 / 0.7906 | 0.7136 / 0.5459 / 0.7958 |

RF edges the locked holdout by +1.13 pt accuracy / +1.27 pt macro-F1 and is
tied on CV5/LOBO within fold-to-fold spread. Both runs share the identical
fold mapping (F1 val {16,18,19} · F2 {3,6,12} · F3 {1,7,14} · F4 {13,17,22,24}
· F5 {8,10,11}) and the identical thresholds (2022: 42.285/45.186 · 2023:
38.073/40.182 · 2024: 45.254/48.720 · 2025: 40.286/43.268 · 2026:
42.157/44.738 °C). Class balance on train portions: 0.3332/0.3333/0.3334
every year (both). Moderate is the hard class in both models.

## 4. Promotion decision + frozen primary

The user promoted RF to the official primary. The deterministic final refit
was persisted via `v2.phase5.rf_compare --refit-only` (refit wall 137.0s —
identical code path, RF bit-reproducible at `random_state=42`), verified by
`v2.phase5.verify_rf_primary` (9/9 checks PASS incl. bit-identical locked
reproduction from the saved joblib), and frozen per
`phase5_production_model.md`. The XGB artifacts and the `rf_compare/` run
record are untouched historical evidence.

**Environment (both runs):** python 3.13.5, xgboost 3.4.1, scikit-learn
1.9.1, numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1.
**Logs:** `v2/logs/`.
