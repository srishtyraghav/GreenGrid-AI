# V2 Phase 9 — Status Report (checkpoint)

**Date:** 2026-10-05
**State:** CODE COMPLETE, NOT YET RUN. No Phase 9 artifacts exist under `v2/data/phase9/`; no report generated.

## Goal

Supervised ML regression for `ΔLST = LST(year+1, W4) − LST(year, W4)` over the
four transitions 2022→2023, 2023→2024, 2024→2025, 2025→2026, using observed
Landsat-9 W4 V2 data; apply the validated model to the frozen 2026
`v2_constrained` Phase 8 planting zones at 400 / 1,000 / 2,500 trees/ha
(planning-density scenarios — none claimed optimal; optimality is a
post-Phase-9 question using cooling predictions).

## Work done (on disk, machine-verified where marked)

- `v2/src/v2/phase9/build.py` — Codex draft + gap closures:
  - Transition panel builder (50k samples/transition, deterministic seed) with
    exact `delta_lst_C` target, block IDs from the pinned V1 5×5 manifest grid.
  - **Met covariates added**: 6 Phase 4 met variables (`current_met_*`), loaded
    via the exact Phase 4 code path; contemporaneous external observations,
    value-identical to the 178-feature schema met columns.
  - RF + XGBoost regressors; adjacent-block CV, LOBO, locked holdout
    `[2,9,15,23]` (evaluated once per model), leave-one-transition-out.
  - 2026 scenario inference on Phase 8 `v2_constrained` plantable pixels;
    explicit assumed vegetation-change parameter (0.05 cover fraction per
    1,000 trees/ha — scenario assumption, no V2-observed density→canopy
    conversion exists); sensitivity values 0.03/0.05/0.08.
  - Leakage audit, input hashing, manifest, report writer with
    OBSERVED / ASSUMED / PREDICTED separation and non-causal wording.
- `v2/src/v2/phase9/verify.py` — verification gate, 7 check families: panel
  integrity (delta exactness, blocks, no-NaN, met presence), leakage protocol
  (locked ∉ CV folds), metrics sanity (R²∈[−1,1], MAE≤RMSE), scenario
  integrity (tree counts recomputed independently from Phase 8 tables, zone-ID
  match, raster contract 1768×1874 / EPSG:4326 / nodata −9999), wording/framing,
  Phase 8 input-hash immutability, artifact completeness.
- `v2/tests/test_phase9_temperature.py` — 7 unit tests (synthetic fixtures);
  **all 7 pass** (two pre-existing defects fixed: float32 non-invertibility in
  the synthetic panel fixture; missing `json` import).
- `v2/tests/test_phase5_primary_rf.py` — reverted to last pushed state; the
  Phase 9 report name must be re-added to `EXPECTED_REPORTS` **after** the
  Phase 9 report exists (the convention test requires exact set equality).
- Environment incident resolved: Miniconda3 was uninstalled mid-task; venv base
  repointed to conda env `C:\Users\anmol\.conda\envs\py313` (Python 3.13.5,
  exact match). All 215 packages working.

## Work to be done (resume checklist)

1. Re-add `"phase9_temperature_reduction_prediction_report.md"` to
   `EXPECTED_REPORTS` in `v2/tests/test_phase5_primary_rf.py`.
2. Run build:
   `PYTHONUTF8=1 PYTHONPATH=v2/src ./.venv/Scripts/python.exe -m v2.phase9.build`
   (panel ~2 min; ~30 fits on ~200k×15 → roughly 10–25 min).
3. Run verify gate → must end `[SUMMARY] N/N checks pass` /
   `[VERIFY] ALL PHASE 9 CHECKS PASS`.
4. Full test suite — expected **106 passed** (99 + 7 new).
5. Review the Phase 9 report numbers; commit + push.

## Constraints in force

- Phases 2–8 frozen/read-only (verify only hashes Phase 8 inputs).
- No causal tree-cooling claims; densities are planning scenarios, never
  "optimal"; LST predictions are not air-temperature guarantees.
- Locked holdout evaluated exactly once per model; no tuning against it.
