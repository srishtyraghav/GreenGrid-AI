# Phase 6/7 Re-run + Phase 8 — Production Build Spec (2026-10-03, auto mode)

## Production model (frozen, verified)
- Booster: `data/processed/phase5_production_3class/phase5_primary_xgb_3class.json`
- Schema: `data/processed/phase5_production_3class/phase5_primary_xgb_3class_features.json` (178 encoded features, ORDER MATTERS)
- Target: 3-class Low/Moderate/High, per-year LST tertiles, train-only thresholds
- Verified: locked [2,9,15,23] acc 0.6396665686967289 / macro-F1 0.6314607179023212
- Raw predictor list = `models.config.PREDICTOR_VARS` (101) + 6 met + 33 spatial-context
  (exact list in `scripts/freeze_phase5_production.py` PREDICTOR_COLS)

## Locked decisions
1. **Full-grid domain (per year)**: all pixels where all 178 features are finite
   (mirrors training dropna). LST rasters written where LST finite (separate count).
2. **Schema alignment**: after `encode_predictors`, reindex columns to the 178-schema
   order exactly; missing dummy columns (category absent that year) filled 0. Cast
   landuse/lulc columns to the SAME dtypes as the training table BEFORE get_dummies
   so dummy names match the schema byte-for-byte (training names may be float-style,
   e.g. `landuse_class_0.0` — inspect schema file first).
3. **Met features (6)**: IDW-interpolate the 16 grid points
   (`data/processed/experiments/exp_met_features_grid_provenance.csv`, lat/lon +
   per-year values) to pixel lat/lon (reference-raster transform). REVERSE-ENGINEER
   the exact IDW variant (power/exponent, distance metric) by reproducing
   `exp_met_features.csv` sampled values to <1e-6 max error; document the variant.
4. **33 spatial-context features**: nan-safe border-correct window stats at 3/5/11
   (reuse logic from `scripts/build_spatial_context_features.py`) computed on FULL
   grids, not sampled pixels.
5. **Spatial-cache means + morphology**: reuse `src/severity/fullgrid.py` helpers
   (block-aware focal means 3/5/11; morphology disks 50-500 m). NO spatial_block_id
   needed (not a predictor).
6. **Base predictors**: reuse `src/severity/fullgrid.py` raster loading /
   `src/models/spatial_features.py` raster map. LULC class from
   `data/processed/phase3/masks/lulc_{year}_30m.tif` (2022-2026).
7. **Storage**: per-year full-grid feature table in PARQUET (float32; fall back to
   .npz if pyarrow missing) under `data/processed/phase6_production/fullgrid/`.
8. **Hotspot definitions (3-class)**: A = severity High; B = High AND confidence>=0.60;
   C = severity_score >= 1.5 (top-half of High band) AND confidence>=0.60.
   MIN cluster 10 px. Document all three.
9. **Severity score** = probs @ [0,1,2] (range [0,2]) — 3-class replacement for the
   old 0-3 score.
10. **Years**: all five (2022-2026) through every stage.
11. **Phase 8 tree assumptions (documented, defensible)**:
    - Available planting space = suitability-eligible land (Phase 7 exclusion mask
      AND landuse eligibility) minus pixels with existing vegetation cover >= 0.30
      and minus water/built-excluded classes.
    - Tree density: **1,000 trees/ha** (~3.2 m spacing, mid-range urban plantation;
      sensitivity table at 400 / 1,000 / 2,500 trees/ha).
    - Zones: Phase 7 priority zones; per-zone: area ha, current vegetation %,
      UHI severity, suitable planting ha, recommended trees, priority.
    - Outputs: per-zone table, recommended-locations GeoJSON (top-ranked planting
      pixels with per-pixel tree counts), density map rasters, summary figures.

## Verification gates (each must PASS before next stage)
G1 (full-grid features): columns == 178 schema exactly (names+order); predict at
   the 749,998 sampled pixels reproduces locked acc/macro-F1 to 1e-9 AND full
   predictions identical to `locked_validation.json` expectations (metrics at least;
   predictions ideally) → proves feature parity with training.
G2 (Phase 6 rasters): raster shape == reference grid; nodata handling; class
   histogram plausible; per-year valid counts recorded.
G3 (Phase 7): class==classify(score) recompute; zone pixel accounting; 5-year loop.
G4 (Phase 8): area reconciliation (ha = px * 900 m2 / 1e4), tree-count arithmetic
   recomputed independently, totals consistent across tables.

## Directory plan
- `data/processed/phase6_production/` — fullgrid features, rasters, hotspots,
  analytics, figures, models, manifest
- `data/processed/phase7_production/` — suitability rasters, zones, tables, figures
- `data/processed/phase8_tree_requirement/` — planting-space rasters, zone tables,
  recommended locations GeoJSON, figures, manifest
- Reports: `reports/phase6_uhi_severity_mapping_report.md` (rewrite),
  `reports/phase7_tree_plantation_suitability_report.md` (rewrite),
  `reports/phase8_tree_requirement_report.md` (new)

## Honesty rules
- Old Phase 6/7 code frozen assumptions are NOT silently reused anywhere they
  conflict (4-class, 2-year, 300k rows, RF-C).
- Every number in reports must trace to a table/raster in the production dirs.
- If a gate fails, STOP that stage, document, move on only if the failure is
  provably cosmetic; mark failures clearly in the final summary.
