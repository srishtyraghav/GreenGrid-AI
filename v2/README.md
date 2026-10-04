# GreenGrid AI — V2 Pipeline

V2 is a clean rebuild of the GreenGrid-AI pipeline starting at Phase 2, for
final academic submission. **V1 is the frozen baseline**: every V1 artifact
(`data/raw/`, `data/processed/`, `src/`, `reports/`, models, `dataset_metadata.csv`)
is preserved unchanged for comparison. Nothing in V1 may be modified, moved,
or deleted by V2 work; V2 only *adds* parallel paths.

## Why V2 exists

V1 reached a frozen production model (Phase 5, 178-feature 3-class XGBoost,
63.97% locked geographic holdout) and completed Phases 6–8, but its **data
foundation has documented weaknesses** that bound downstream quality:

1. **July-only window**: the July 1–30 composite window is after monsoon
   onset in Delhi — 2025 Landsat thermal coverage = 41.9% and Sentinel-2 =
   24.6% of the study area (spatially clustered holes; 12 of 25 spatial
   blocks below 50% valid).
2. **LULC instability**: V1 Dynamic World July exports cover 6.1 / 34.2 /
   95.3 / 1.1 / 92.6% of the study area across 2022–2026 (2025 effectively
   empty) — export-time ingestion gaps with no guard.
3. **Missing constraint data**: V1 Phase 7 could not exclude water/building
   footprints/road surfaces (only OSM road lines + a buildings *sample*).
4. **Grid defects discovered in audit**: Sentinel-2 20 m rasters sit half a
   20 m pixel off the common lattice in y; S2 10 m and LULC have a 2 px
   extent mismatch vs the Landsat grid.

## V2 changes (decided, evidence-backed)

| Aspect | V1 | V2 | Evidence |
|---|---|---|---|
| Acquisition window | Jul 1–30 | **W4 = May 1–Jun 30, all 5 years** | `data/v2/phase2/window_investigation/decision_matrix.md` (5-window × 5-year comparison on real per-pixel reads of 702 scenes) |
| Landsat | L9 only | **L9 only** (L8 evaluated in the 702-scene investigation, excluded 2026-10-04 for exact sensor parity) | `window_investigation/window_metrics_l9only.csv` — W4 passes 99.95–99.98% all years with 4–6 obs/cell; rankings unchanged |
| Export guards | none | empty-collection + coverage guards | V1 queued invalid no-bands exports (Jul 2024/2025 S2) |
| Observation counts | none | per-pixel QA_CLEAR_COUNT / VALID_COUNT band | enables coverage-weighted Phase 3+ |
| Constraints | road lines, building sample | full buildings (417k), water (2,682), road surfaces (78) | `data/v2/phase2/constraints/` |
| Meteorology | July-only, 16-point grid | hourly superset 2022→2026, same grid | `data/v2/phase2/met/` |
| Everything else | — | **identical to V1 recipe** (masking, scaling, boundary, EPSG:4326, median compositing) | `gee/v2/*.js` headers |

## Layout (after the 2026-10-04 restructure)

This tree is self-contained; paths below are relative to `v2/`:

- `src/v2/phase2/` — V2 Phase-2 tooling (inventory, audits, coverage, window investigation, fetchers)
- `src/v2/phase3/`, `src/v2/phase4/` — tested preprocessing/feature code (see `src/v2/CONTRACT.md`)
- `data/phase2/` — V2 Phase-2 artifacts (audits, metrics, manifests, constraints, met, GEE console logs)
- `data/raw/` — the W4 GEE downloads (landsat9 / sentinel2 / lulc, + SHA256SUMS.txt)
- `data/gis/` — study-area boundary (copied from V1, ODbL)
- `reference/` — frozen V1 production schema + feature manifest (read-only inputs to Phase 4)
- `gee/` — GEE export scripts (runbook in `gee/README.md`)
- `reports/` — the V2 phase reports
- `tests/` — synthetic test suite

V1 lives entirely under `v1/` (content byte-identical, only location changed). Historical Phase-2 audit/manifest JSONs still embed pre-restructure paths — treated as records, not live references. Execute from the repo root with `PYTHONPATH=v2/src` and module paths `v2.<phase>.<module>`; from inside `v2/`, everything is relative to this directory.

## Comparison protocol (for later phases)

V1 and V2 results are **not season-comparable at face value**: V1 severity
classes are relative to July windows, V2 classes relative to May–Jun
windows. Any V1-vs-V2 comparison must go through identical validation
protocols (same spatial blocks, same locked-holdout discipline) on each
version's own data, reported side-by-side, never mixed in one training set.

## Status

- **V2 Phase 2: COMPLETE (pending GEE W4 exports + user review).**
  Audits, window decision (W4), met superset, constraints, GEE scripts done.
  Next action: user runs `gee/v2/` scripts → assistant ingests into
  `data/raw/v2/` → coverage audit of the W4 rasters closes Phase 2.
- **V2 Phase 3: NOT STARTED** — blocked on Phase-2 review by user.
