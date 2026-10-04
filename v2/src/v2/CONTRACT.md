# GreenGrid AI — V2 Phase 3/4 Executable Contract

Status: **CODE COMPLETE, EXECUTION PENDING** (waits for GEE W4 exports into
`data/raw/` and per-phase user-triggered runs). Unit-tested only on
synthetic fixtures (`tests/v2/test_phase34_synthetic.py`, fixtures under
`data/_smoketest/p34/`). 36/36 tests pass; every module `py_compile`-clean.

Source-of-truth artifacts this contract codes against:
`data/phase2/` (audit/, coverage/, window_investigation/,
provenance_table.csv, manifest/), `reports/phase2_dataset_report.md` §5–8,
`gee/v2/01_export_landsat9_w4.js` + `02_export_sentinel2_w4.js` +
`03_export_lulc_w4.js`, `V2.md`, and the frozen V1 model semantics in
`scripts/build_production_fullgrid_features.py`,
`scripts/build_spatial_context_features.py`, `src/models/spatial_features.py`,
`src/models/morphology_features.py`, `src/models/dataset.py`,
`src/preprocessing/{indices,align,vector_raster}.py`,
`src/features/vegetation_cover.py`, `reports/experiments/exp_met_provenance.md`.

---

## 1. Input layout (`data/raw/`)

Years 2022–2026, ONE common window **W4 = May 1 – Jun 30** (tag `05_06`), no
per-year deviations.

| Product | Path per year | Bands (positional contract) | Native |
|---|---|---|---|
| Landsat 9 | `landsat9/<y>_05_06/landsat9_<y>_05_06_composite_30m.tif` | `SR_B2..SR_B7` (scaled ×0.0000275−0.2), `ST_B10` (RAW DN), `QA_PIXEL`, `QA_CLEAR_COUNT` | 30 m |
| Sentinel-2 10 m | `sentinel2/<y>_05_06/sentinel2_<y>_05_06_10m_composite.tif` | `B2,B3,B4,B8` (÷10000), `VALID_COUNT` | 10 m |
| Sentinel-2 20 m | `sentinel2/<y>_05_06/sentinel2_<y>_05_06_20m_composite.tif` | `B5,B6,B7,B8A,B11,B12` (÷10000), `SCL`, `VALID_COUNT` | 20 m |
| LULC (Dynamic World) | `lulc/<y>_05_06/lulc_<y>_05_06_10m.tif` | `lulc` float label, classes 0..8 | 10 m |

Class legend: 0 water, 1 trees, 2 grass, 3 flooded_vegetation, 4 crops,
5 shrub_and_scrub, 6 built, 7 bare, 8 snow_and_ice.

Context (vector) inputs, from V2's own Phase-2 fetch
(`data/phase2/constraints/`, `CONTEXT_LAYERS_PROVENANCE.md` — ODbL 1.0):

| Layer | File | Stored content | Feature use |
|---|---|---|---|
| Roads | `osm_roads_delhi.geojson` | FULL network, 238,194 features | `dist_road_m` via V1-parity subset (§5); `--roads-full` opt-in |
| Vegetation | `osm_vegetation_delhi.geojson` | 6,484 features | `dist_vegetation_m` via V1-parity union (§5) |
| OSM landuse | `osm_landuse_delhi.geojson` | 13,570 features | `landuse_class` via V1's 7 values (§5) |
| Buildings | `osm_buildings_delhi.geojson` | 417,343 footprints | `dist_building_m` (full) |
| Water | `osm_water_delhi.geojson` | 2,682 polygons | presence raster (later phases) |
| Road surfaces | `osm_road_surfaces_delhi.geojson` | 78 polygons | presence raster (later phases) |

No V1 file under `data/raw/gis/` is read anywhere in the V2 feature path.

GEE GeoTIFFs carry **no band descriptions and nodata attr = None**; the
positional layout above is authoritative (V1 convention,
`src/preprocessing/io.py::get_band_index`). If descriptions exist they are
cross-checked by preflight.

**Ambiguity (A1, resolved by convention):** the Phase-2 report §9 runbook
still names the L9 export `landsat89_YYYY_05_06_composite_30m.tif` (L8+L9
design); the updated single-sensor decision and `gee/v2/01_export_landsat9_w4.js`
produce `landsat9_...`. The code follows the script + the user's instruction
(L9-only). Flag for the Phase-2 report errata.

## 2. Grid, transforms, NoData

- **Authoritative grid**: read at runtime from
  `data/raw/landsat9/2026_05_06/landsat9_2026_05_06_composite_30m.tif`
  (2026-reference convention, same as V1). `--grid-file` overrides (tests).
  Verified identical to the V1 grid (1874×1768, EPSG:4326, origin
  76.8329062507/28.8846991402, res 0.00026949458523585647) — row/col pinning
  of the spatial-block geometry (§7.6) is geographically exact.
- **All L9 years must match the authoritative transform EXACTLY** (all six
  affine terms). S2/LULC are never assumed aligned: every product is
  reprojected ONTO the authoritative grid explicitly via
  `rasterio.warp.reproject` (`src/v2/common.py::reproject_to_grid`) —
  average for continuous, nearest for labels. This fixes the V1 F4 defects
  (S2 20 m half-pixel y-offset; S2 10 m/LULC 2 px extent mismatch) by
  construction.
- **NoData**: NaN is the universal masked sentinel; `0` is a REAL value
  everywhere **except** the known S2 B2 zero-fill artifact (F3) which the
  S2-10m validity rule guards against. `±Inf` is never legitimate → preflight
  failure.
- **NaN-safe average is implemented by construction** (GDAL's `average` does
  not reliably exclude interior NaN pixels, verified on GDAL 3.12.2):
  `reproject_to_grid` computes `sum(v·w)/sum(valid·w)` with a two-pass
  weighted scheme; cells with coverage `≤1e-6` become NaN. Nearest
  reprojection passes values through raw (NaN propagates — no invented
  labels).

## 3. Validity rules & quality gates

Pure functions in `src/v2/common.py`:

- L9 valid = `finite(ST_B10) & finite(SR_B2) & (QA_CLEAR_COUNT ≥ 1)`
- S2-10m valid = `finite(B2..B8) & (B2 > 0) & (VALID_COUNT ≥ 1)`
  **HARD INVARIANT (C2)**: a native pixel with `B2==0` must not enter the
  S2-derived value stack even if all other bands are finite. Phase 3 NaNs
  invalid native pixels *before* index computation/aggregation, so the
  NaN-safe area-weighted aggregation can never use them in NDVI/PVC or any
  30 m product. Phase-3 preflight reports the exclusion accounting per year
  (`finite == valid + B2==0`, exact identity, gate; `--crosscheck-audit`
  pins it against a prior audit JSON); Phase-4 verify asserts every feature
  row sits on a 30 m cell with ≥1 valid native B2>0 pixel (leakage gate).
- S2-20m valid = `finite(B5..B12)` (SCL-excluded classes already masked in GEE)
- LULC valid = `finite(label)` (class 0 water is real)

Per-year coverage gates inside the study area (Phase-2 contract): **L9 ST
≥ 90 %, S2-10m ≥ 85 %, LULC ≥ 80 %** (V1 2025 LULC was 1.1 %). Evaluated at
native resolution in preflight (`verify_inputs.py`) and re-checked from the
30 m masks in `verify_features.py`. S2-20m coverage is reported, not gated
(no numeric gate in the Phase-2 contract).

Valid masks are uint8: `1 valid / 0 invalid / 255 outside-study`
(study area = `data/raw/gis/study_area/study_area.geojson`, pixel-centre
rule, `all_touched=False`).

## 4. Index formulas (V1 conventions)

| Product | Formula | Source bands |
|---|---|---|
| LST °C | `ST_B10 × 0.00341802 + 149.0 − 273.15` | L9 ST_B10 raw DN |
| NDVI (authoritative) | `(B8−B4)/(B8+B4)` | S2 10 m |
| NDBI (authoritative) | `(B11−B8A)/(B11+B8A)` — 20 m bands because B8∉20 m export and B11∉10 m export | S2 20 m |
| NDRE | `(B8A−B5)/(B8A+B5)` | S2 20 m |
| NDMI | `(B5−B6)/(B5+B6)` | L9 |
| MNDWI | `(B3−B6)/(B3+B6)` | L9 |
| BSI | `((B6+B4)−(B5+B2))/((B6+B4)+(B5+B2))` | L9 |
| L9 NDVI/NDBI | `(B5−B4)/(B5+B4)`, `(B6−B5)/(B6+B5)` | diagnostics only (not predictors) |
| PVC | `clamp((NDVI−0.05)/0.75, 0, 1)`, NaN preserved | from authoritative S2 NDVI |

All division-by-zero → NaN; NaN in → NaN out. Indices are computed at native
resolution, then aggregated (average) to 30 m — exactly V1's
`indices.py` → `align.py` order.

## 5. Static constraint layers (`data/phase3/static/`, built once)

All context layers are V2's **own** Phase-2 GeoJSON fetch
(`data/phase2/constraints/`, provenance `CONTEXT_LAYERS_PROVENANCE.md`);
V1's Overpass JSON under `data/raw/gis/` is **never read** in the feature
path. Feature semantics replicate V1 exactly via explicit parity filters
(`build_core.py::filter_v1_*`), derived from the verbatim V1 queries in
`src/data_collection/download_gis_data.py` and verified against the actual
V1 files (read-only).

| Layer | V2 source (stored) | Feature-parity filter |
|---|---|---|
| `roads_distance_30m.tif` | `osm_roads_delhi.geojson` — FULL network, 238,194 features | **default**: `highway ∈ {motorway, trunk, primary}` (exact, case-insensitive) — V1's fetch query was `way["highway"~"^(motorway|trunk|primary)$"]` and its file contains exactly those three classes (708/1478/1434 ways; no `_link`, no secondary/tertiary). V1's preprocessing filter (motorway..tertiary) was a superset on that file. `--roads-full` burns the entire network (opt-in; changes `dist_road_m`/`road_pixel_fraction` vs. the frozen model) |
| `vegetation_distance_30m.tif` | `osm_vegetation_delhi.geojson` — 6,484 features | V1's query union: `leisure ∈ {park, garden, nature_reserve}` OR `landuse ∈ {forest, grass, meadow}` OR `natural ∈ {wood, scrub, grassland}`. V2's extra `recreation_ground` fetch excluded |
| `landuse_raster_30m.tif` | `osm_landuse_delhi.geojson` — 13,570 features | `landuse` tag ∈ V1's 7 fetched values {residential, commercial, industrial, retail, park, forest, farmland}. V1 fetched only `way["landuse"~"^(...)$"]`, so `classify_landuse`'s leisure/natural fallbacks could never fire on V1 data — natural-only and other-tagged rows are dropped (mirrors V1's query) |
| `buildings_distance_30m.tif` | `osm_buildings_delhi.geojson` (417,343 full-NCT footprints; invalid geometries repaired with `shapely.make_valid`, disclosed) | none |
| `water_presence_30m.tif` | `osm_water_delhi.geojson` | n/a (not a 178 predictor) |
| `road_surfaces_presence_30m.tif` | `osm_road_surfaces_delhi.geojson` | n/a (not a 178 predictor) |

Rasterization recipes (unchanged from the previous contract version):
class-coded landuse burn `all_touched=True`, uint8 nodata 255; distance
layers = binary burn (1 feature/0 bg, all_touched=True) then
`distance_transform_edt(binary==0) × 30 m` (feature pixels distance == 0).
`static_manifest.json` records stored-vs-burned feature counts, the applied
filter, source SHA-256, and repair counts.

The tolerant Overpass-JSON parser (`build_core.py::parse_overpass_json`) is
**retained but unused for features** (kept for provenance/audit of legacy
inputs; exercised only by a retention test).

## 6. Phase-3 outputs (per year under `data/phase3/<year>/`)

`lst_30m.tif`, `ndvi_30m.tif`, `ndbi_30m.tif`, `ndre_30m.tif`,
`ndmi_30m.tif`, `mndwi_30m.tif`, `bsi_30m.tif`, `l9_ndvi_30m.tif`,
`l9_ndbi_30m.tif`, `vegetation_cover_30m.tif`, `lulc_30m.tif` (float32
nearest label), `valid_{l9,s2_10m,s2_20m,lulc}_30m.tif`, `domain_30m.tif`
(all-product validity ∩ study), `build_manifest.json`. Static layers + per-year
manifests carry SHA-256 of every input/output.

LULC 10 m → 30 m uses `Resampling.nearest` on the raw label raster.
**V1 gap-fills (nearest-valid) before aggregating; V2 does NOT gap-fill by
default** (`--lulc-gap-fill` opt-in) because the Phase-2 GEE coverage guard
(≥50 % at export) plus the ≥80 % Phase-3 gate make fabrication unnecessary;
unfilled pixels are NaN and leave the ML domain through the finite-predictor
rule. Deviation recorded in §open-questions.

30 m S2/LULC valid masks = "any valid native pixel overlaps the cell"
(reprojected validity-average > 1e-6). Under (C2) the B2-zero artifact is
additionally excluded from the *values* feeding those products: a fully-
artifact 30 m cell becomes NaN (out of the domain), a mixed cell aggregates
only its valid native pixels. `build_manifest.json` records per-year
`{finite_px, b2_zero_px, valid_excluding_b2_zero, identity_holds}`.

## 7. Phase-4 feature assembly (all 178, grouped)

Schema source (confirmed): **`data/processed/phase5_production_3class/
phase5_primary_xgb_3class_features.json`** → `feature_names` (178, ordered);
identical order to `feature_manifest.csv`. SHA-256 of the JSON name list:
`ddd43d0468dcab8df1c173a2adf57c8e3aec9f386851f7031b77cb3f8dc2c211`.
Runtime gate: after one-hot encoding the assembled frame is reindexed to the
schema (`fill_value=0`) and `list(columns) == schema` is asserted — any
drift fails loudly.

Metadata columns precede the 178: `row, col, spatial_block_id, lst_C`.
`year` is **inside** the predictor block (frozen schema position 24); keeping
it in the metadata row too would duplicate the column name (parquet-illegal).
`lst_C` is target/metadata only — never a predictor (leakage gate).

### 7.1 Base predictors (27 after encoding)
ndvi, ndbi, vegetation_cover, ndmi, mndwi, bsi, ndre (as §4);
`landuse_class_{0,2,4,5,6,7,8}.0` — one-hot of the OSM landuse code (float64
dtype ⇒ `.0` dummy names, V1 parity; classes 1/3 unseen in V1 training ⇒
zero-filled); `lulc_class_0..8` — one-hot of the 30 m DW label (int64
dummies); dist_road_m, dist_vegetation_m, dist_building_m (float64);
year (float32).

### 7.2 Spatial means (24) — block-aware focal means, V1 `spatial_features.py`
For each of the 8 bases {ndvi, ndbi, vegetation_cover, ndmi, bsi,
dist_road_m, dist_vegetation_m, dist_building_m} and window {3,5,11}:
mean over the window restricted to pixels of the **same spatial block**
(other blocks treated as NaN). Port of `_focal_mean_block_aware`
(scipy `ndimage.convolve`, `mode="constant"`, cval=0).

### 7.3 Morphology (88 after encoding) — V1 `morphology_features.py`
Disk radii {50,100,250,500} m → px {2,3,8,17} (`round(r/30)`, min 1); exact
disk kernels; stripe prefix-sum engine (`focal_aggregate_block_aware`), same
block restriction. Per radius: `building/road/vegetation_pixel_fraction`
(fraction of `dist_* == 0` pixels), `vegetation_cover_mean`,
`landuse_frac_1..8` (per-class fraction; denominator = valid class pixels in
disk ∩ block; 0 where empty), `landuse_dominant_{r}` (argmax class, 0 when
empty → encoded as int64 one-hot `landuse_dominant_{r}_{0,2,4,5,6,7,8}`,
unseen classes zero-filled), `landuse_entropy` (Shannon, natural log, 0 when
empty), `ndvi_contrast` = ndvi − disk-mean(ndvi), `ndbi_contrast` likewise.

### 7.4 Met (6) — `exp_met` matched variant
Per point (16-point grid) per year: mean of hours **04:00–06:00 UTC** over
**May 1–Jun 30** from `data/phase2/met/met_superset_hourly_2022_2026.csv`.
IDW to pixel centres: `w_i = 1/(dlat² + (cos(28.64°)·dlon)² + 1e-6)`,
`pred = Σwᵢvᵢ / Σwᵢ` — reproduces V1's `exp_met_features` variant
(build_production_fullgrid_features.py header; power 2, degrees,
longitude scaled, epsilon INSIDE the squared distance). Names: met_t2m_c,
met_rh_pct, met_wind_kmh, met_precip_mm, met_ssr_wm2, met_swc_m3m3.
(V1 used the July window; V2 uses W4 per the Phase-2 contract — the method
is identical, the window differs by design.)

### 7.5 Window stats (33) — V1 `build_spatial_context_features.py`
On float32 casts of {ndvi, ndbi, ndre, ndmi, mndwi, bsi, vegetation_cover}:
`{band}_std{3,5,11}` (7×3); on {mndwi, ndre}: `{band}_mean{3,5,11}` (2×3);
on {ndvi, ndbi}: `{band}_range{3,5,11}` (2×3). `nan_window_stats`: border-
correct summed-area filters (`uniform_filter`, `mode="constant"` cval=0),
std via `sqrt(max(E[x²]−E[x]²,0))`, range = max−min ignoring NaN; a pixel is
valid iff ≥1 valid pixel lies in its window. No LST-derived statistics
(V1 leakage rule).

### 7.6 Sampling, blocks, domain
- Domain = pixels where ALL raw predictors are finite, **clipped to the
  study area** (`--study-area`, rasterized pixel-centre rule; with real GEE
  products the predictors are already NaN outside — the clip makes it
  explicit and matches V1's never-outside-study Phase-3 valid mask).
- Sampling (V1 Tier-2): per-year independent, up to `--max-samples`
  (default 150,000), `numpy.default_rng(42)` choice without replacement,
  then sorted. On the real grid this triggers only if a year exceeds the cap.
- `spatial_block_id`: **PINNED V1 geometry (C1)** — `src/v2/common.py`
  `PINNED_ROW_BINS = (1.0, 354.2, 707.4, 1060.6, 1413.8, 1767.0)`,
  `PINNED_COL_BINS = (0.0, 374.6, 749.2, 1123.8000000000002, 1498.4, 1873.0)`
  (the V1 `_compute_spatial_block_raster` formula `linspace(min, max+1, 6)`,
  `digitize−1`, clip, `id = row_band·5 + col_band` applied to the pooled
  extent of the table that trained the frozen model), `HOLDOUT_BLOCKS =
  (2, 9, 15, 23)` fixed. Bins NEVER derive from the V2 sample — block ids
  are identical for the full grid and any subset of rows (sampling
  independence, tested). Full provenance, per-block geographic bboxes and
  the honest formula-reproduction verification:
  `data/phase2/spatial_blocks_manifest.json`.
- dtypes: float32 everywhere except the 12 distance columns (float64, V1
  parity: ~4.5e4 m magnitudes would breach the V1 G1 1e-4 bound in float32).

## 8. Phase-4 verification gates (`verify_features.py`)

1. predictor columns == frozen 178 (names AND order; schema hash match);
2. no NaN/Inf in predictors;
3. leakage: {lst_C, lon, lat, row, col, spatial_block_id} ∉ predictors;
4. per-year coverage ≥ §3 gates (recomputed from Phase-3 masks);
4b. hard B2 invariant: every feature row's 30 m cell contains ≥ 1 valid
   native S2-10m pixel under the B2>0 guard (no S2-derived feature may
   contain a B2==0 pixel contribution);
5. spatial_block_id == PINNED V1 bins (independent of the sample);
6. met sanity: finite; spatially constant only if all 16 point means are
   equal (warning); adjacent-pixel |Δ| ≤ `--met-gradient-tol` (default 0.1 —
   IDW of a 16-point field is smooth at 30 m; reported per column);
7. count summary per year (rows, occupied blocks, coverage) →
   `v2_feature_verify.json`; exit non-zero on any failure.

Preflight (`verify_inputs.py`) writes `w4_input_audit.json` and exits
non-zero with per-check failure messages on: missing/empty files, wrong band
count/order/dtype, L9 transform mismatch, wrong native resolution, CRS
mismatch, label outside 0..8, Inf presence, coverage-gate breach. Runnable
standalone: `python -m v2.phase3.verify_inputs --data-root data/raw/v2`.

## 9. §open-questions / deviations / blockers

**Deviations from V1 semantics (deliberate, need review sign-off):**

1. **LULC gap-fill off by default.** V1 `process_lulc_layers` nearest-fills
   the 10 m label mosaic before 10→30 m (its July coverage was 1–6 %). V2's
   contract gates coverage at ≥80 % and GEE refuses <50 % exports, so filling
   would fabricate labels. Unfilled pixels → NaN → out of the domain.
   `--lulc-gap-fill` reproduces V1 exactly if reviewers prefer parity.
2. **`dist_building_m` uses V2 full-NCT buildings** (417,343 footprints) —
   V1 used an east-Delhi *sample* (F6 defect). Values will differ from V1's;
   semantics ("distance to nearest OSM building") are identical.
3. **Context layers: V2 fetch date ≈ V1 fetch date but not identical.**
   V2's roads/vegetation/landuse GeoJSONs were fetched 2026-10-04; V1's JSON
   files 2026-08-26 (OSM evolves). Feature *definitions* are parity-filtered
   (§5), but pixel-level values inherit the newer OSM state. Unavoidable;
   directionally an improvement (more complete mapping).
4. **Block bins provenance (RESOLVED by C1).** V2 pins the V1 pooled bins
   (§7.6) instead of deriving bins from its own sample. Verified anomaly
   (documented in `data/phase2/spatial_blocks_manifest.json`): the V1
   production table's own `spatial_block_id` column is per-year Phase-3
   output — 2022-2025 reproduce exactly from per-year sample extents, but
   2026 carries 581 rows from the original 2-year sample's bins, and 22,251
   pixels have year-inconsistent block ids. **No single bin set can
   reproduce that column 100%**; the pinned pooled bins are the frozen
   production semantics (the formula that generated the production fullgrid
   block-aware features), reproduce 88.3% of the column, and preserve
   91.7% / 97.9% / 90.6% / 98.4% of locked blocks 2/9/15/23 pixels (locked-set
   agreement 95.69%). Holdout ids are the fixed constants (2, 9, 15, 23) —
   NOT `occupied[2::6]` under the pooled bins (which would give [3,11,17]).
5. **Metadata columns**: `year` appears only inside the 178-predictor block
   (duplicate parquet columns are illegal); metadata = row, col,
   spatial_block_id, lst_C.
6. **Distance metric** is pixel-Euclidean × 30 m (V1 approximation, EW pixel
   ≈26.4 m at Delhi lat). A geographic (pyproj geodesic) distance would be
   more correct but breaks V1 parity of `dist_*` and all derived features.
7. **Met window** is W4 (May 1–Jun 30, hours 04–06 UTC) per the Phase-2
   contract; V1 used July 1–30. Same method, different window (by design).
8. **B2==0 pixels change value, not just coverage (C2).** V1 counted
   artifact pixels as valid (F3) and let them into 30 m means. V2 excludes
   them from the value stack by construction; mixed 30 m cells therefore
   differ slightly from a V1-style leaky mean. User-mandated invariant;
   residual difference vs V1 exists only where artifacts mix into cells
   (2022: 84,909 px, 2026: 7,867 px per the real audit).

**Residual semantic differences vs V1 (after the §5 parity filters):**

- **Roads**: none in definition — {motorway, trunk, primary} exactly matches
  V1's file content. Note the V1 preprocessing filter (motorway..tertiary)
  suggested secondary/tertiary, but V1's fetch never retrieved them; the
  user's hypothesis of `_link` variants is not borne out by the V1 file
  (verified: only the three exact classes).
- **Vegetation**: V2's fetch did not query `natural=grassland` standalone, so
  grassland ways that carry none of V2's other fetched tags are absent (~182
  such ways in V1's file, ≈2.8 % by count; V1 burned them). Grassland ways
  co-tagged with V2's leisure/landuse tags ARE present in the V2 layer and
  kept. Documented, not compensated (cannot recover unfetched features).
- **Landuse**: none in definition — the 7-value filter reproduces V1's query;
  classify fallbacks remain unreachable in the feature path.
- V2 layers include `relation` geometries (V1 fetched ways only) — a
  completeness improvement, kept.

**Blockers for review:**

- **B1 (RESOLVED 2026-10-04)**: V2 now owns its context layers
  (`osm_roads_delhi.geojson` full network, `osm_vegetation_delhi.geojson`,
  `osm_landuse_delhi.geojson`); no V1 input is read anywhere in the feature
  path. The stored roads layer is the full network; features default to the
  V1-parity subset for frozen-178 comparability (`--roads-full` opt-in).
- **B2**: `landuse_class_1.0` (park) and `landuse_class_3.0` (grass) and the
  corresponding `landuse_dominant_{r}_1/_3` columns are absent from the
  frozen schema (never sampled in V1 training). V2's OSM landuse raster *can*
  produce codes 1/3; the reindex zero-fills them, so the encoded table
  matches the schema exactly. Reviewers should confirm zero-fill (vs.
  retraining) is the intended production behavior.
- **B3**: No 178 feature required V1's July rasters, the QA_PIXEL/SCL median
  bands, or the paired-sampling table — nothing else was underivable from V2
  contract inputs alone.

**Ambiguities in the Phase-2 contract found while coding:**

- **A1**: report §9 runbook filename (`landsat89_...`) vs. script/final
  decision (`landsat9_...`) — code follows the script (see §1).
- **A2**: the met aggregation is defined for "hours 04:00–06:00 UTC" —
  interpreted as the three hourly timestamps 04:00, 05:00, 06:00 (matching
  V1's implementation), not a 3-hour integral mean.
- **A3**: `QA_CLEAR_COUNT`/`VALID_COUNT` are Float64/Float32 in the exports;
  the `≥1` test is exact (counts are small integers stored as floats).

## 10. Usage (real run, later session)

```
# 0. preflight on the W4 downloads (writes v2/data/phase2/w4_input_audit.json).
# Re-runs: add --crosscheck-audit v2/data/phase2/w4_input_audit.json to pin the
# B2==0 exclusion accounting against the recorded audit.
# Run all commands from the repository root:
PYTHONUTF8=1 PYTHONPATH=v2/src .venv/Scripts/python.exe -m v2.phase3.verify_inputs \
    --data-root v2/data/raw --out v2/data/phase2/w4_input_audit.json
# 1. core products (v2/data/phase3/<year>/ + static/):
PYTHONUTF8=1 PYTHONPATH=v2/src .venv/Scripts/python.exe -m v2.phase3.build_core \
    --data-root v2/data/raw --out-root v2/data/phase3
# 2. feature tables (v2/data/phase4/ + manifests; --study-area defaults to the
#    study-area GeoJSON and clips the sampling domain):
PYTHONUTF8=1 PYTHONPATH=v2/src .venv/Scripts/python.exe -m v2.phase4.assemble_features \
    --phase3-root v2/data/phase3 --out-root v2/data/phase4
# 3. gates:
PYTHONUTF8=1 PYTHONPATH=v2/src .venv/Scripts/python.exe -m v2.phase4.verify_features \
    --features-dir v2/data/phase4 --phase3-root v2/data/phase3
```

Tests: `PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest v2/tests -q`
(39 tests, synthetic fixtures only; the suite bootstraps its own path).
