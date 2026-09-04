# Phase 7 — Stage 0 Input Audit

**Project:** GreenGrid-AI — Tree Plantation Suitability Analysis
**Date:** 2026-09-04
**Scope:** Read-only verification of every candidate Phase 7 input against disk, using the project venv (`rasterio 1.5.1`, `pandas`). No implementation code; no existing file modified.

---

## 1. Grid facts (verified on disk)

All rasters below share one identical grid, opened and confirmed with `rasterio`:

- **Shape:** 1768 × 1874 (3,313,232 px full grid)
- **CRS:** EPSG:4326
- **Transform (all rasters, identical):** origin (76.83290625, 28.88469914), pixel 0.00026949° × −0.00026949°, north-up, no rotation
- **Resolution:** ≈0.00026949° (~30 m at the equator; ~26–30 m metric at Delhi's latitude, per Phase 3 report)
- **Valid mask:** `data/processed/phase3/masks/valid_mask_30m.tif` — uint8, nodata=0, value=1, **1,500,777 valid px (~45.3% of full grid)**; this is the analysis domain for every Phase 7 computation.
- `phase6_manifest.json` confirms: grid_spec height 1768 / width 1874, valid_cells_per_year 1,500,777, years [2022, 2026]. All statements below match the manifest.

## 2. Audit table

Transform abbreviated as `origin / res` since it is identical for every row. "Years" = which snapshot years the input exists for. Coverage = number of finite/non-nodata px (full grid = 3,313,232).

| Input | Source path | Shape / dims | CRS | Transform (origin / res) | Resolution | Years available | NoData semantics | Coverage (valid px) | Proposed role | Validation status |
|---|---|---|---|---|---|---|---|---|---|---|
| severity class | `data/processed/phase6/rasters/severity_{2022,2026}.tif` | 1768×1874, int16 | EPSG:4326 | (76.83290625, 28.88469914) / 0.00026949° | ~30 m | **2022, 2026** | −1 = NoData; classes 0–3 | 1,500,777/yr; unique {0,1,2,3} | Need (diagnostic cross-check) | **Verified** |
| severity_score | `data/processed/phase6/rasters/severity_score_{2022,2026}.tif` | 1768×1874, float32 | EPSG:4326 | same | ~30 m | **2022, 2026** | −1.0 = NoData | 1,500,777/yr; range 0.013–2.989 (2022), 0.013–2.951 (2026) | **Need** (primary heat term) | **Verified** |
| LST | `data/processed/phase6/rasters/lst_{2022,2026}.tif` (copy of `data/processed/phase3/aligned/l9_{year}_composite_lst_30m.tif`, float64, NaN nodata, ranges verified identical: 11.58–55.28 °C 2022, 16.16–53.05 °C 2026) | 1768×1874, float32 | EPSG:4326 | same | ~30 m | **2022, 2026** | −1.0 = NoData | 1,743,888 (2022), 1,894,985 (2026) — **wider than valid mask** | **Need** (reduced weight) | **Verified**; coverage ≠ valid mask, must intersect with valid mask before use |
| confidence | `data/processed/phase6/rasters/confidence_{2022,2026}.tif` | 1768×1874, float32 | EPSG:4326 | same | ~30 m | **2022, 2026** | −1.0 = NoData | 1,500,777/yr; 0.250–0.994 (2022), 0.250–0.989 (2026) | **Diagnostic** (priority stability, §j) | **Verified** |
| probability_low / moderate / high / severe | `data/processed/phase6/rasters/probability_{low,moderate,high,severe}_{2022,2026}.tif` (all 8 opened) | 1768×1874, float32 | EPSG:4326 | same | ~30 m | **2022, 2026** | −1.0 = NoData | 1,500,777 each; each in ~[0, 0.99] | **Diagnostic** (not used in baseline Need; severity_score already aggregates them) | **Verified** |
| uncertainty_zone | `data/processed/phase6/rasters/uncertainty_zone_{2022,2026}.tif` | 1768×1874, uint8 | EPSG:4326 | same | ~30 m | **2022, 2026** | 255 = NoData; 0/1 flag | 1,500,777/yr; mean 0.61 (2022), 0.40 (2026) | **Diagnostic** (flag priority zones landing in uncertain areas) | **Verified** |
| NDVI | `data/processed/phase3/aligned/s2_{2022,2026}_ndvi_30m.tif` | 1768×1874, float64 | EPSG:4326 | same | ~30 m | **2022, 2026** | **NaN = nodata** | 1,651,319 (2022, −0.33–0.88), 1,914,362 (2026, −0.98–1.00) — wider than valid mask | **Need + Opportunity** (heat stress & planting headroom) | **Verified** |
| NDBI | `data/processed/phase3/aligned/s2_{2022,2026}_ndbi_30m.tif` | 1768×1874, float64 | EPSG:4326 | same | ~30 m | **2022, 2026** | **NaN = nodata** | 1,652,453 (2022, −0.64–0.58), 1,915,823 (2026, −0.54–0.87) | **Need + Opportunity** (built-up intensity) | **Verified** |
| vegetation_cover | `data/processed/phase4/features/vegetation_cover_{2022,2026}_30m.tif` | 1768×1874, float64 | EPSG:4326 | same | ~30 m | **2022, 2026** | **NaN = nodata** | 1,651,319 (2022), 1,914,362 (2026); 0–1 | **Diagnostic** — baseline Need/Opportunity only; appears in Scenario D | **Verified** |
| dist_road_m | `data/processed/phase3/masks/roads_distance_30m.tif` | 1768×1874, float64 | EPSG:4326 | same | ~30 m | **STATIC (single file, no year variants)** | NaN = nodata | full grid 3,313,232; 0–8,410 m | **Opportunity** (road accessibility band) | **Verified** |
| dist_vegetation_m | `data/processed/phase3/masks/vegetation_distance_30m.tif` | 1768×1874, float64 | EPSG:4326 | same | ~30 m | **STATIC** | NaN = nodata | full grid; 0–17,132 m | **Opportunity** (green proximity) | **Verified** |
| dist_building_m | `data/processed/phase3/masks/buildings_distance_30m.tif` | 1768×1874, float64 | EPSG:4326 | same | ~30 m | **STATIC** | NaN = nodata | full grid; 0–44,690 m | **Diagnostic only** — NOT in baseline Need/Opportunity (buildings are an OSM *sample*, not full coverage; documented decision) | **Verified** |
| landuse raster | `data/processed/phase3/masks/landuse_raster_30m.tif` | 1768×1874, uint8 | EPSG:4326 | same | ~30 m | **STATIC** | 255 = nodata (no 255 present → full-grid coverage) | full grid; classes {0:2,784,500 (84.0%), 1:4, 2:24,245, 4:23,157, 5:106,826, 6:255,946, 7:5,462, 8:113,092} — **class 3 (grass) entirely absent; class 1 (park) = 4 px** | **Opportunity** (eligibility rules matrix) | **Verified** |
| valid mask | `data/processed/phase3/masks/valid_mask_30m.tif` | 1768×1874, uint8 | EPSG:4326 | same | ~30 m | **STATIC** (shared by both years) | 0 = nodata; 1 = valid | 1,500,777 | **Constraint** — the only hard exclusion | **Verified** |
| Phase 4 combined CSV | `data/processed/phase4/tables/combined_urban_environmental_dataset.csv` | 300,000 rows × 14 cols | n/a (lon/lat + row/col) | n/a | n/a | **2022, 2026** (year column) | n/a | 300k sampled rows (the Phase 5/6 training sample) | **Diagnostic only** (validation/sanity statistics; not a full-grid input) | **Verified** — cols: lon, lat, row, col, spatial_block_id, year, lst_C, ndvi, ndbi, vegetation_cover, landuse_class, dist_road_m, dist_vegetation_m, dist_building_m |

## 3. Static vs year-specific (critical for two-year Opportunity)

- **Year-specific (both 2022 AND 2026 available):** severity, severity_score, LST, confidence, probability_×4, uncertainty_zone, NDVI, NDBI, vegetation_cover. → Need and the dynamic parts of Opportunity can be computed per year.
- **Static / single-date (one file, reused for both years):** landuse raster, dist_road_m, dist_vegetation_m, dist_building_m, valid mask. → The landuse-eligibility, road-accessibility and green-proximity terms of Opportunity are **identical in 2022 and 2026**; year-to-year Opportunity variation comes only from NDBI and NDVI (built-up intensity, planting headroom). This must be stated in Phase 7 outputs: Opportunity change 2022→2026 reflects vegetation/index change on a static land-use/accessibility base, not observed land-use change.
- **Coverage mismatch warning:** NDVI/NDBI/vegetation_cover/LST raw rasters cover *more* px than the valid mask (e.g. NDVI-2026 1,914,362 vs 1,500,777). Phase 6 outputs are already masked to exactly 1,500,777. Phase 7 must intersect every raw input with the valid mask before any computation.

## 4. Water / hard-exclusion reality (honest report)

Searched `data/raw/gis/*`, `data/processed/phase3/*`, `data/processed/phase4/*`, source configs, and prior phase reports:

- **No NDWI or any water-index raster** exists (Phase 5/6 reports list NDWI explicitly as a *future* feature family).
- **No water class** in the landuse raster (classes 0–8, none is water).
- **No water polygons or waterways** in raw OSM data: `data/raw/gis/` contains only `administrative/`, `buildings/`, `landuse/`, `roads/`, `study_area/`, `vegetation/`. Grep for `natural=water` / `waterway` / reservoir tags in the OSM GeoJSONs returns **0 hits**.
- Water is visible only *indirectly*: strongly negative NDVI pixels (2026 min −0.98), the cool Yamuna corridor in LST, and LST-masked (NaN) areas flagged in Phase 3/4 reports as likely water/wetland edges — none of these is a usable, validated water exclusion layer.

**Conclusion:** the only implementable hard exclusion is the invalid mask (NoData). Water, building-footprint and road-surface exclusions are **NOT implementable with current project data** and are recorded as stated limitations in the design spec — not silent omissions.

## 5. Deviations / corrections to the brief's assumptions

1. Distance rasters are named `roads_distance_30m.tif` / `vegetation_distance_30m.tif` / `buildings_distance_30m.tif` (not `dist_*_30m.tif`); same layers, corrected paths.
2. `class 1 (park)` is not fully absent — 4 px exist; `class 3 (grass)` is fully absent. Rules matrix covers both anyway.
3. `lst_{2022,2026}.tif` in `phase6/rasters/` are re-noded copies of the Phase 3 aligned LST with **coverage wider than the valid mask** (1.74 M / 1.89 M px) — unlike the other Phase 6 rasters. Intersect with valid mask before use.
4. NDVI/NDBI/vegetation_cover are **float64 with NaN nodata**, not a numeric nodata value; Stage 1 must use NaN-aware masking.

All other claims in the brief (grid, CRS, res, valid px, dtypes, class meanings, Phase 5 frozen facts) **verified against disk**.
