# V2 Phase 5 — Production Model Freeze Record

**Primary model: Random Forest (`phase5_primary_rf_v2`) — FROZEN.**
User decision at the close of Phase 5; the XGB run
(`phase5_primary_xgb_v2`, locked 0.68770 / 0.67247) is the preserved
historical baseline. Phase 6+ consumes the model ONLY through the marker
`v2/data/phase5/phase5_primary_model.json`:

```json
{
  "primary_model": "rf",
  "frozen": true,
  "frozen_at_utc": "<freeze timestamp>",
  "locked_accuracy": 0.6990177602649763,
  "locked_macro_f1": 0.6851219134292812,
  "artifact": "phase5_primary_rf_v2.joblib",
  "baseline_xgb": "phase5_primary_xgb_v2.json",
  "verification_passed": true,
  "verified_at_utc": "<gate timestamp>",
  "verified_by": "v2.phase5.verify_rf_primary"
}
```

## Artifacts (v2/data/phase5/, gitignored — never committed)

| file | content |
|---|---|
| `phase5_primary_rf_v2.joblib` | fitted production RF (1.77 GB) |
| `phase5_primary_rf_v2.json` | production metadata (params, locked metrics, thresholds, schema hash, refit wall) |
| `phase5_primary_model.json` | **the marker Phase 6+ reads** |
| `rf_compare/` | frozen comparison-run record (CV/LOBO/locked, evaluated exactly once) |
| `phase5_primary_xgb_v2.json` + XGB run artifacts | historical baseline, untouched |

## Verbatim pre-specified RF params (V1 `RANDOM_FOREST_PARAMS`, no tuning)

`n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2,
max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1`

## Frozen locked metrics (blocks {2,9,15,23}, n_train=609,912, n=140,088)

| metric | value |
|---|---|
| **accuracy** | **0.6990177602649763** |
| **macro-F1** | **0.6851219134292812** |
| confusion | [[28079,12901,4069],[4999,22560,14085],[569,5541,47285]] |
| per-class F1 (Low/Mod/High) | 0.7136 / 0.5459 / 0.7958 |

## Freeze verification gate (`v2.phase5.verify_rf_primary`)

Reloaded joblib → re-predicted locked rows from the phase4 parquets →
**9/9 checks PASS, exit 0**: bit-identical locked predictions (140,088, exact
array equality), exact accuracy/macro-F1/confusion/per-class reproduction vs
the frozen `rf_compare/` record, thresholds monotonic + train-only exact,
frozen schema hash `ddd43d04…c211`, banned columns absent, marker numbers ==
frozen record. Result written into the marker (`verification_passed: true`).

## Determinism note

scikit-learn RandomForest derives every tree's seed deterministically from
`random_state`, so the fit is bit-reproducible regardless of `n_jobs` —
evidenced in-run (smoke-probe fits reproduced exactly in the full run) and at
freeze (the persisted refit reproduces the comparison run's locked
predictions exactly). This contrasts with XGBoost's threaded histograms,
which required the single-threaded final-refit policy on the baseline model.

## Provenance

Refit: `PYTHONUTF8=1 PYTHONPATH=v2/src ./.venv/Scripts/python.exe -m
v2.phase5.rf_compare --features-dir v2/data/phase4 --out v2/data/phase5
--refit-only` (137.0s). Gate: `python -m v2.phase5.verify_rf_primary
--features-dir v2/data/phase4`. Full narrative: `phase5_uhi_detection_report.md`.
**Not season-comparable to V1's 63.97%** (W4 vs July window).
