# GreenGrid AI V2 — Phase 6: UHI Severity Mapping (Production Run)

**Status: COMPLETE — verification gate 23/23 PASS (2026-10-05)**
**Primary model: V2 Phase 5 Random Forest (frozen)** — locked geographic holdout
accuracy 0.69902 / macro-F1 0.68512, blocks {2, 9, 15, 23}, per the marker
`v2/data/phase5/phase5_primary_model.json`.

---

## 1. Objective

Apply the frozen V2 Phase 5 primary model to the full valid 30 m Delhi grid for
every study year (2022–2026, W4 May–Jun window) and produce the production
relative heat-severity classification products (UHI hotspot proxies): relative
severity class rasters, class-probability rasters, confidence rasters, and
multi-definition hotspot inventories with per-block statistics — the direct
input to Phase 7 plantation suitability. These products describe relative
heat-severity classes within each year; they are not measurements of physical
UHI intensity.

## 2. Inputs and Model Lineage

| Input | Source |
|---|---|
| Primary model | `v2/data/phase5/phase5_primary_rf_v2.joblib` (resolved via marker, never hardcoded) |
| Model record | RF comparison run `v2/data/phase5/rf_compare/`; frozen 2026-10-05; verification gate 9/9 |
| Features | Phase 4 code path (`v2.phase4.assemble_features` / `_spatial`) at FULL grid (sampling off) — schema-identical to the 178-feature training tables; frozen schema sha256 `ddd43d04…c211` asserted before predicting |
| Valid domain | Phase 3 masks with the hard S2 B2>0 invariant; invalid/cloudy pixels are NoData everywhere |
| Grid | 1768×1874, EPSG:4326, 2026-reference affine (identical to V1) |
| Met covariates | W4-window ERA5-Land table (same 6 predictors used in training) |
| Hotspot tiers | Faithful port of V1 §8 definitions (see §6) |

The model is a DIRECT 3-class classifier (Low/Moderate/High from per-year LST
tertiles); no LST thresholding is applied at map time (`thresholds_by_year.json`
is provenance only). Phase 5 artifacts were read-only throughout.

## 3. Verification Gates

Independent re-derivation from the written rasters/tables (`v2.phase6.verify`):

- **[1] PASS** — marker resolves the frozen, verified RF primary artifact.
- **[2] PASS** — all per-year rasters match the authoritative grid: 1768×1874,
  EPSG:4326, exact 2026-reference affine, expected bands/dtypes.
- **[3] PASS (×5)** — NoData consistent with the valid domain: severity,
  probability, confidence, and all hotspot-ID rasters share the exact domain
  (1,908,519–1,910,308 px/year).
- **[4] PASS (×5)** — probabilities sum to 1 on every valid pixel
  (max |sum−1| = 1.19e-07, float32 rounding) and severity == argmax(stored
  probabilities) exactly.
- **[5] PASS (×5)** — class balance sane (all classes within [15%, 50%]).
- **[6] PASS (×5)** — tables agree with rasters (area, per-block, hotspot counts).
- **[7] PASS** — all artifacts + manifest provenance present, schema hash matches.

**Gate result: 23/23 PASS, 0 FAIL** (log: `v2/logs/_phase6_verify.log`).
One full rerun was required to reach this: the first run wrote 0 instead of the
declared −1 nodata outside the domain in the hotspot-ID rasters, and derived
severity from float64 probabilities while storing float32 bands (3 px/year
disagreement on exact ties). Both defects were fixed in `build.py`
(severity/confidence now derive from the same float32 values that are stored)
and every product was regenerated — no post-hoc output patching. Test suite:
**79/79 pass**.

## 4. Method

1. Preflight: all Phase 3/4/5 inputs, marker, frozen schema, study-area mask.
2. Per year: assemble the 178-feature frame at full grid on the valid domain
   (~1.91 M px), assert schema, predict with the RF (`predict_proba`),
   scatter to rasters.
3. Derive confidence = max class probability; severity score = probs·[0,1,2].
4. Hotspots: 8-connected components ≥ 10 px, computed inside the valid domain
   only (outside-domain pixels never join a cluster).
5. Write rasters (CloudGeoTIFF-free GeoTIFF), GeoJSON cluster polygons, per-year
   statistics tables, temporal comparison, coverage statistics, manifest +
   pipeline record.

Runtime: 2070 s total (≈ 410–419 s/year), single pass, Python 3.13.5.

## 5. Relative Severity Class Distribution

| Year | Valid px | Low % | Moderate % | High % | Mean confidence | Mean severity score |
|---|---|---|---|---|---|---|
| 2022 | 1,908,519 | 30.86 | 33.44 | 35.69 | 0.727 | 1.039 |
| 2023 | 1,910,057 | 32.01 | 31.45 | 36.54 | 0.703 | 1.015 |
| 2024 | 1,910,308 | 28.49 | 31.40 | **40.11** | 0.742 | 1.092 |
| 2025 | 1,910,308 | 30.17 | 32.99 | 36.84 | 0.772 | 1.050 |
| 2026 | 1,910,149 | 30.24 | 32.67 | 37.09 | 0.793 | 1.073 |

Under the per-year relative severity classification, 2024 has the largest
proportion of pixels classified as High (40.1%) and the highest mean severity
score (1.092). These values describe the spatial distribution of the relative
classes within each year's valid domain and should not be interpreted as
evidence that 2024 had the highest absolute UHI intensity or surface
temperature among the five years. Conversely, 2023 has the largest Low-class
share (32.0%) under its year-specific relative classification. Mean confidence
rises monotonically 2023→2026 (0.70→0.79). Class shares are stable across
years — consistent with the ~1/3 training balance and with inter-annual
variability in the relative LST distribution rather than domain shifts.

## 6. Hotspot Results

Tier definitions (V1 §8 port): **A** = severity High; **B** = High AND
confidence ≥ 0.60; **C** = severity score ≥ 1.5 AND confidence ≥ 0.60.

| Year | Def A area (ha) / clusters | Def B area (ha) / clusters | Def C area (ha) / clusters |
|---|---|---|---|
| 2022 | 60,914 / 589 | 50,390 / 421 | 49,982 / 420 |
| 2023 | 62,317 / 825 | 46,003 / 714 | 44,835 / 701 |
| 2024 | **68,548** / 648 | 56,299 / 515 | 56,076 / 504 |
| 2025 | 63,025 / 557 | 54,780 / 528 | 54,723 / 525 |
| 2026 | 63,398 / 578 | **56,832** / 487 | **56,802** / 486 |

Def-B hotspot extent varies ~15% across years (46.0k–56.8k ha), largest in
2026; 2023 has the most fragmented pattern (714 clusters on the smallest area).
All hotspot areas are UHI hotspot proxies derived from the relative
classification — not absolute heat-intensity measures. Full per-definition
tables (including per-cluster GeoJSON polygons and per-block breakdowns) are in
`v2/data/phase6/tables/`.

## 7. Analytics

- **Coverage**: valid domain ≈ 97.0% of the ~1.968 M study-area pixels in every
  year; L9 ≥ 0.90, S2-10m ≥ 0.85, LULC ≥ 0.80 gates all TRUE for all years
  (`tables/coverage_statistics.csv`).
- **Confidence structure**: mean max-probability 0.70–0.79; Low/High decisions
  carry higher confidence than Moderate (echoing the Moderate-class recall
  limitation documented in Phase 5).
- **Spatial structure**: per-block statistics (`tables/block_statistics_*.csv`)
  show the spatial distribution of High-class predictions across the study
  blocks; these tables are provided as inputs for Phase 7.

## 8. Figures

No figures in this phase (per plan); all products are rasters/GeoJSON/tables.

## 9. Output Files

Under `v2/data/phase6/` (all gitignored data):

- `rasters/` — severity_{year}.tif (uint8, nodata 255), probability_{year}.tif
  (float32 3-band), confidence_{year}.tif, severity_score_{year}.tif — 20 files, 188 MB
- `hotspots/` — hotspot_ids_def{A,B,C}_{year}.tif (int32, nodata −1) +
  hotspots_def{A,B,C}_{year}.geojson polygons — 50 MB
- `tables/` — area/block/hotspot statistics × 5 years + temporal_comparison.csv
  + coverage_statistics.csv
- `phase6_manifest.json`, `phase6_pipeline_record.json`
- Logs: `v2/logs/_phase6_train.log`, `_phase6_verify.log`

## 10. Honest Notes and Caveats

- Severity classes are per-year RELATIVE tertiles of LST (Low/Moderate/High
  within each July→W4 year) — maps compare severity *structure* across years,
  not absolute temperature thresholds. This is the frozen Phase 5 target
  definition; do not re-interpret class values as absolute heat levels.
- W4 (May–Jun) products are NOT directly comparable to V1's July-based severity
  maps; the same is true of the Phase 5 metrics.
- The 3-px float32-tie correction (§3) changes nothing at map-reading scale.
- Domain pixels (~1.91 M) exceed the 1,881,088 V1 study-cell count by ~1.5%:
  the V2 domain is defined by the Phase 3 valid masks, not the V1 cell count.
- Phase 7 has NOT been started (explicit user instruction).

## 11. Validation Summary

| Gate | Result |
|---|---|
| Verification gate (`v2.phase6.verify`) | **23/23 PASS** |
| Full test suite | **79/79 pass** |
| Primary model | V2 Phase 5 RF (frozen, marker-verified) |
| Downstream readiness | Phase 7 inputs complete: severity, confidence, hotspot tiers, per-block stats, constraint-ready rasters |
