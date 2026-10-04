# GreenGrid AI — V2 Phase 2: Dataset Rebuild Report

**Status:** COMPLETE — pending GEE W4 exports (user-run) and user review.
**Phase 3 has NOT been started.**
**Date:** 2026-10-04 · **Scope:** V2 data foundation, 2022–2026, Delhi NCT

---

## 1. Executive summary

This report documents the V2 Phase-2 data rebuild. V1 remains frozen and
byte-identical as the baseline (git-verified: V2 work adds only `src/v2/`,
`v2/data/`, `gee/v2/`, `v2/data/raw/`, `v2/README.md`, and this report).

Four things were established:

1. **The V1 July dataset has a real, quantified coverage defect.** In the
   July 1–30 composites, Landsat thermal validity is 88.4 / 86.6 / 69.2 /
   **41.9** / 96.4% and Sentinel-2 validity 82.1 / 73.5 / 40.4 / **24.6** /
   96.9% (2022→2026). The 2024/2025 holes are spatially clustered: in 2025,
   12 of 25 spatial blocks fall below 50% valid thermal coverage (block 7 =
   4.6%).
2. **A wider hot-season window fixes it, and the evidence picks W4
   (May 1–Jun 30).** A five-window × five-year comparison over real
   per-pixel reads of 702 satellite scenes shows every non-July candidate
   reaches 99.98% thermal coverage in all five years; W4 wins on temporal
   tightness (day-of-year σ 16.6 vs 25.0–34.0 for the others) and on
   phenomenon fit (mean composite LST 42.8 °C — squarely on Delhi's
   pre-monsoon heat peak; July is post-monsoon-onset, cloud-prone, and the
   coolest window, mean 38.0 °C).
3. **The V1 archive has further defects the rebuild corrects:** unstable
   LULC coverage (6.1→1.1% across years — Dynamic World ingestion gaps with
   no export guard), a half-pixel Sentinel-2 20 m grid offset, a Sentinel-2
   2026 B2 zero-fill swath, and missing planting-constraint data.
4. **V2 Phase 2 delivers:** full data inventory + SHA256 manifest,
   source/provenance table, acquisition & compositing methodology, per-year
   coverage statistics and maps, CRS/extent and NoData audits, a
   data-quality report, ready-to-run GEE W4 export scripts with
   empty-collection guards, an hourly meteorological superset, and full
   OSM water/building/road-surface constraint layers.

---

## 2. Temporal-window decision (required record)

### 2.1 Method

Candidate windows: **W0** Jul 1–30 (V1), **W1** Apr 1–Jun 30,
**W2** Mar 1–Jun 30, **W3** Apr 1–Jul 31, **W4** May 1–Jun 30 — evaluated
for all five years. Evidence source: Microsoft Planetary Computer STAC
(public, no auth) — 702 scenes (246 Landsat 8/9 C2 T1 L2, 456 Sentinel-2
L2A, after the V1 recipe's filters) read as COG windows over the study area;
masking cloned exactly from the verified V1 GEE recipe
(`v1/gee/export_data_2023_2025.js`; see
`window_investigation/masking_recipe.md`). The union of all windows
(Mar 1–Jul 31) was read once; each candidate window selects scenes by date.
0 genuine read failures. Full machine table: `window_investigation/window_metrics.csv`
(50 rows); per-scene list: `scene_list.csv` (702 rows).

Masking (identical to V1, verified against the GEE script): Landsat
QA_PIXEL **bits 3 (cloud) + 4 (cloud shadow) only** — V1 does not mask
dilated cloud/cirrus/snow and applies no scene-level cloud filter; valid ST
additionally requires raw ≠ 0 and a 250–350 K sanity range. Sentinel-2
scene filter `CLOUDY_PIXEL_PERCENTAGE < 60`; SCL classes **{3, 8, 9, 10}**
masked. Deviations from a *stricter-than-V1* recipe would lower all
coverages but hurt July most (cirrus + monsoon), strengthening the
conclusion.

### 2.2 Decision matrix (Landsat ST = primary thermal evidence; S2 corroborates)

Valid-ST coverage is the % of the 1,881,088 study cells (Delhi NCT on the
authoritative 30 m grid) with ≥1 usable thermal observation.

| Window | Valid-ST cov. mean / min across 5 yrs (%) | Coverage CV across yrs | Scenes/yr (mean, L / S2) | Seasonal spread (doy σ, mean) | ≥2-obs cov. (%) | Mean composite LST (°C) | Phenomenology fit |
|---|---|---|---|---|---|---|---|
| W0 Jul 1–30 (V1) | 92.7 / **80.4** | 8.4% | 8.4 / 7.4 | **8.0** | 62.5 | 38.0 | Post-onset: monsoon clouds; coolest LST |
| W1 Apr 1–Jun 30 | 99.98 / 99.98 | ~0.0 | 30.8 / 56.6 | 25.0 | 100 | 42.3 | Peak-heat plateau, includes cooler April |
| W2 Mar 1–Jun 30 | 99.98 / 99.98 | ~0.0 | 40.8 / 79.2 | 33.3 | 100 | 40.0 | Diluted by cool March |
| W3 Apr 1–Jul 31 | 99.98 / 99.98 | ~0.0 | 39.2 / 69.2 | 34.0 | 100 | 41.8 | Cloudy monsoon July tail |
| **W4 May 1–Jun 30** | **99.98 / 99.98** | **~0.0** | **20.6 / 39.4** | **16.6** | **100** | **42.8** | **Squarely on pre-monsoon heat peak** |

Per-year valid-ST coverage (Landsat): W0 = 90.2 / 94.4 / 99.1 / **80.4** /
99.3% (2022→2026); **every other window = 99.98% in all five years**.
Sentinel-2 valid coverage: W0 collapses in monsoon-affected years
(94.7 / 75.6 / 41.4 / **25.3** / 100.0%); W1–W4 = 99.97–100% every year.
Inter-scene LST spread (mean per-pixel σ): W1 4.08, W4 4.20, W3 4.57,
W2 5.92 °C. Median clear observations per cell under W4: 7–14 (Landsat),
8–10 (S2) — robust median composites everywhere.

### 2.3 Pre-registered criteria (applied in order)

1. **Hard screen: ≥90% valid thermal coverage in EVERY year.** W0 **fails**
   (2025 = 80.4%; 2022 = 90.2% is only marginal). W1–W4 pass.
2. **Among survivors, maximize cross-year temporal consistency.** Coverage
   CV is ~0 for all four, so consistency reduces to seasonal tightness:
   doy_std W4 (16.6) < W1 (25.0) < W2 (33.3) ≈ W3 (34.0); W2/W3 dilute the
   composite with cool-March or cloudy-July acquisitions (lower mean LST,
   larger inter-scene spread).
3. **Prefer the window whose mean composite LST is closest to the annual
   peak heat.** LST ordering W4 42.8 ≳ W1 42.3 > W3 41.8 > W2 40.0 > W0
   38.0 °C matches the independent air-temperature phenomenology.

### 2.4 Phenomenology (Open-Meteo / ERA5-Land, Delhi-mean, 4×4 point grid)

Monthly mean Tmax: May 36.2–41.5 °C, Jun 36.4–40.4 °C, **Jul 33.2–34.4 °C**.
Hottest 30-day period starts **May 22 / Jun 10 / Jun 17 / Jun 12 / Jun 10**
(2022–2026). Monsoon onset (10-day rolling precip ≥ 2.5 mm/day persisting):
**Jul 4 / Jun 21 / Jun 26 / Jun 17 / Jul 20**. Peak UHI heat in Delhi is
pre-monsoon May–June; July — the V1 window — is post-onset, cooling, and
cloud-contaminated. Figure: `window_investigation/delhi_tmax_precip_2022_2026.png`;
data: `phenomenology.csv`, `delhi_daily_met_2022_2026.csv`.

### 2.5 Decision and rationale

**W4 (May 1–Jun 30) is approved as the single common V2 window for all five
years** (user-approved 2026-10-04), with W1 documented as fallback. W4:
passes the hard screen at 99.98% thermal coverage in every year with 100%
≥2-observation coverage; tightest seasonal focus among survivors (~20.6
L8/L9 scenes/yr, ~10 valid obs/cell, doy σ 16.6); highest mean composite
LST (42.8 °C), consistent with the pre-monsoon Tmax peak and safely before
monsoon onset.

**Why W0/July was rejected:** it fails the coverage hard screen in 2025
(80.4% thermal; S2 25.3%) and is marginal in 2022 (90.2%); its few scenes
(median ~1–3 clear obs/cell in weak years) make V1 composites fragile to
inter-annual cloud luck; and it is the *coolest* candidate (38.0 °C mean LST,
3–7 °C below the pre-monsoon peak) — i.e., it samples Delhi after monsoon
onset, biased away from peak UHI season. The 2026 W0 coverage (99.3%) is
not evidence July is generally safe: 2026 monsoon onset was unusually late
(Jul 20).

**Why not W1/W2/W3:** all pass the screen, but W4 dominates on criteria 2
and 3 — W1 spreads 3 months (σ 25.0, cooler April dilution), W2 adds March
(σ 33.3, LST 40.0), W3 adds the cloudy July tail (σ 34.0, lower S2 quality
in monsoon years). **No year gets a different window** — one common window
preserves cross-year comparability (user instruction).

**Honest caveats:** 5 years is short; LST ≠ air Tmax (criterion 3 uses LST
as corroboration only); a stricter-than-V1 cloud mask would lower all
coverages but hurt W0 most; the study used L8+L9 (V1 exported L9 only —
L8-only would halve scene counts without changing rankings).

---

## 3. Data inventory

`inventory/inventory.csv`: all 34 files under `data/raw/` (Landsat 9,
Sentinel-2, LULC, OSM vectors) with SHA256, size, format.
`inventory/inventory_bands.csv`: 116 band rows — dtype, CRS, dimensions,
nodata attr, exact min/max/mean/std, % finite per band (windowed reads).
`manifest/manifest.json` + `manifest/SHA256SUMS.txt`: full artifact
manifest including all V2 outputs (237 files, 521 MB). Highlights:

- **Landsat 9** (7 files): SR_B2–B7, ST_B10, QA_PIXEL; 1874×1768;
  EPSG:4326; **all 7 byte-comparable on one identical grid**.
- **Sentinel-2 10 m** (5 files): B2/B3/B4/B8, float64, 5620×5300.
- **Sentinel-2 20 m** (5 files): B5/B6/B7/B8A/B11/B12 + SCL, float32, 5623×2651.
- **LULC** (5 files): single float band (Dynamic World label), 5620×5300.
- **Vectors**: study area (GeoJSON, readable); other OSM layers are
  **Overpass API JSON, not GeoJSON** (see §8, finding F6).

Band statistics, file hashes, and per-band validity are in the CSVs —
nothing here relies on filenames or assumptions.

---

## 4. Source & provenance table

`provenance_table.csv` (also §1 of `v2/README.md`); every layer lists source,
product, temporal window, cloud filtering, resolution, CRS, NoData rule,
license, path, and its provenance document. Summary:

| Layer | Source / product | Window | Filtering | Resolution | CRS |
|---|---|---|---|---|---|
| Landsat 9 SR+ST | USGS LC09 C2 T1 L2 via GEE (**L9 only**; L8 evaluated & excluded 2026-10-04) | W4 per year | QA_PIXEL bits 3+4; no scene filter | 30 m | EPSG:4326 |
| Sentinel-2 SR | COPERNICUS/S2_SR_HARMONIZED via GEE | W4 per year | scene cloud <60%; SCL {3,8,9,10} | 10/20 m | EPSG:4326 |
| LULC | GOOGLE/DYNAMICWORLD/V1 mode label via GEE | W4 per year | inherits DW pipeline | 10 m | EPSG:4326 |
| Meteorology | Open-Meteo archive (ERA5/ERA5-Land) | hourly superset 2022-01-01→2026-10-04 | n/a | 0.1° synoptic, 16-pt grid | EPSG:4326 |
| Water / roads / buildings | OpenStreetMap via Overpass API | fetched 2026-10-04 | n/a | vector | EPSG:4326 |
| Study area | OSM Relation 1942586 (V1, frozen) | 2026-08-27 | n/a | vector | EPSG:4326 |
| Window evidence | Microsoft Planetary Computer STAC (L8/9 + S2 COG reads) | Mar 1–Jul 31 per year | V1-masking replica | 30/10/20 m | per asset |

---

## 5. Acquisition & compositing methodology

**Satellite (production path):** `gee/v2/01_export_landsat89_w4.js`,
`02_export_sentinel2_w4.js`, `03_export_lulc_w4.js` — cloned from the
verified V1 recipes with only the documented V2 changes:

- W4 window (May 1–Jun 30) for every year 2022–2026 — no per-year changes.
- Landsat **9 only** (user decision 2026-10-04: exact sensor parity with V1;
  L8 was evaluated in the window investigation and excluded — L9-only evidence
  in `window_investigation/window_metrics_l9only.csv`).
- **Empty-collection + coverage guards:** if a year's collection is empty
  (or LULC coverage <50%), nothing is queued and an explicit `ERROR` prints —
  this is the failure mode that silently produced invalid V1 July-2024/2025
  S2 exports and the ~1%-coverage V1 2025 LULC raster.
- **Per-pixel observation-count bands** (`QA_CLEAR_COUNT`, `VALID_COUNT`)
  exported alongside composites for coverage-weighted processing.
- Unchanged from V1: Delhi boundary (FAO/GAUL), QA bits 3+4 / SCL
  {3,8,9,10} masking, S2 scene cloud <60%, SR ×0.0000275−0.2 scaling,
  S2 ÷10000, **ST_B10 raw DN** (Phase 3 applies ×0.00341802+149.0),
  per-pixel median compositing, EPSG:4326, scales 30/10/20 m, GeoTIFF.

**Meteorology:** `fetch_met_superset.py` — same Open-Meteo/ERA5 source and
16-point grid as V1, hourly superset (6 variables) so any window can be
aggregated later without refetching. 2026 runs through 2026-10-04 (the API
rejects future dates); all five W4 windows are fully covered. 0 NaN across
667,392 rows; 2023-07 sanity value reproduced (30.5 °C, 04–06 UTC mean).

**Constraints:** `fetch_osm_constraints.py` — Overpass mirror rotation as
V1; full-Delhi building footprints via 3×3 sub-bbox merge (9/9 cells
succeeded); verbatim queries, timestamps, counts, areas, and ODbL
attribution in `CONSTRAINTS_PROVENANCE.md`.

**Window evidence:** `window_investigation.py` — STAC query once per year
for the Mar 1–Jul 31 union; COG window reads via signed `/vsicurl` URLs;
per-pixel accumulation on the authoritative grid; metrics per §2.
`window_phenomenology.py` — daily Tmax/precip monsoon-onset analysis.

---

## 6. Per-year valid-coverage statistics (V1 July archive, finite-based)

% of study-area cells valid, from `coverage/per_year_coverage.csv`
(exact pixel counts in the CSV):

| Year | L9 ST_B10 | L9 QA-clear | S2 10 m | S2 20 m | LULC (excl. class 0) |
|---|---|---|---|---|---|
| 2022 | 88.4 | 80.3 | 82.1 | 82.1 | 6.1 |
| 2023 | 86.6 | 79.0 | 73.5 | 73.5 | 34.2 |
| 2024 | **69.2** | **62.0** | **40.4** | 40.4 | 95.3 |
| 2025 | **41.9** | **36.9** | **24.6** | 24.6 | **1.1** |
| 2026 | 96.4 | 93.9 | 96.9 (79.3 excl. artifact) | 97.0 | 92.6 |

Spatial structure (`coverage/block_coverage_l9_st.csv`, 25 V1 blocks,
tiling replicated exactly from V1): 2025 — 12/25 blocks <50% valid (blocks
1, 6, 7, 8, 9, 11, 12, 14, 15, 19, 20, 24; block 7 = 4.6%); 2024 — blocks
3, 9, 14, 19, 20, 24 (block 3 = 25.5%). Missing pixels are **spatially
clustered**, not random — concentrated in the N/NE in 2024 and widespread
(with the far-N worst) in 2025 (maps: `coverage/maps/*.png`, 10 PNGs;
machine-readable masks: `coverage/masks/`, 10 uint8 GeoTIFFs on native
grids, 1=valid / 0=invalid / 255=outside-study).

LULC class histograms: `coverage/lulc_class_histogram.csv`.

---

## 7. CRS / resolution / extent audit

`audit/grid_audit.csv|.json` — every raster vs the authoritative grid
(2026 L9 composite: EPSG:4326, 1874×1768, 1,881,088 study cells), with
numeric transform deltas and half-pixel/one-pixel shift tests:

- **Landsat 9 (7/7): EXACT-MATCH** — identical transform, dimensions, CRS.
- **Sentinel-2 20 m (5/5): RESAMPLING-REQUIRED** — top edge offset by
  *exactly half a 20 m pixel in y* (origin mod own pixel = 0.5); height
  2651 (odd) vs 2650 expected. Genuine sub-pixel misalignment vs both the
  L9 30 m and S2 10 m lattices.
- **Sentinel-2 10 m + LULC (10/10): SAME-GRID-DIFF-EXTENT** — origins align
  (integer × own pixel) but extent is 2 px short on each side
  (5620×5300 vs 5622×5304). Exact 3×3 aggregation onto the L9 grid remains
  possible.
- No rotation, no CRS mismatch, no x-offset anywhere.

**Phase-3 consequence (documented, not yet actioned):** V2 preprocessing
must register S2 20 m explicitly (half-pixel shift) and pad/align S2 10 m +
LULC to the Landsat lattice before any aggregation — V1 code paths hard-
code assumptions this audit contradicts.

## 8. NoData audit & data-quality report

`audit/nodata_audit.csv|.json` — all 116 bands: **nodata attribute = None
in every band; NaN is the universal sentinel** (0 Inf anywhere). No file
uses a declared nodata value. Classified NAN-SENTINEL. Notable usage
patterns: ST_B10 stored raw DN (37272–53275 ≈ 289–326 K; Phase 3 applies
×0.00341802+149.0); QA_PIXEL is a live bitmask (dominant clear value 21824);
L9 and S2 10 m rasters are float64, S2 20 m float32.

### Findings (severity-ordered)

- **F1 (critical, fixed in V2 design):** July 2024/2025 coverage collapse,
  spatially clustered (§6). Root cause: monsoon-onset cloud + few scenes.
  Fix: W4 window — 99.98% coverage all years.
- **F2 (critical, fixed in V2 design):** V1 LULC coverage 6.1 / 34.2 / 95.3 /
  1.1 / 92.6% across 2022–2026 — Dynamic World ingestion gaps at export
  time with no guard (2025 essentially empty). Fix: `03_export_lulc_w4.js`
  coverage guard (<50% → no export, explicit ERROR).
- **F3 (major, guarded for V2):** Sentinel-2 2026 B2 zero-fill swath —
  2,982,452 study pixels (17.6%) have B2==0 with all other bands normal;
  V1's finite test counts them valid, inflating 2026 S2 coverage 79.3→96.9%.
  V2 exports carry VALID_COUNT and Phase 3 must apply a B2>0 guard.
- **F4 (major, addressed in V2 Phase 3 scope):** S2 20 m half-pixel offset +
  S2 10 m/LULC 2 px extent mismatch (§7).
- **F5 (moderate, structural):** block 20 (SW sliver, 294 cells) 0% valid in
  all five years; block 24 (SE) <50% in all years — persistent dead/low
  zones independent of window; documented for Phase 3 masking decisions.
- **F6 (moderate, superseded by V2):** V1 OSM `.json` layers are Overpass
  JSON, not GeoJSON (geopandas-incompatible); V1 buildings layer is a
  sample covering only east Delhi (77.18–77.26 °E). V2 fetched full-NCT
  buildings (417,343 features, 103.75 km²; 2 invalid geometries disclosed,
  0.0005%), water (2,682 features, 51.05 km², incl. 181 canal centerlines),
  road surfaces (78 polygon features, 0.16 km² — sparse mapping, road
  lines remain the primary road layer). ODbL 1.0 attribution recorded.
- **F7 (minor, noted):** V1 exported median-of-QA_PIXEL/SCL — bitwise-
  meaningless but preserved in V2 exports for downstream parity.
- **F8 (minor, scheduled):** met superset 2026 truncated at 2026-10-04
  (API cannot serve future dates); all W4 windows fully covered; the fetch
  script completes 2026 automatically after 2026-12-31.

No invented or interpolated values exist anywhere in V2 data: composites
are per-pixel medians over real clear observations; NoData stays NoData.

---

## 9. GEE W4 acquisition — runbook and expected outputs

`gee/v2/README.md` is the operational runbook. Expected (≤30 tasks to
Drive folder `GreenGridAI_V2_Phase2`):

- `landsat9_YYYY_05_06_composite_30m.tif` ×5 — 9 bands (SR_B2–B7,
  ST_B10 raw, QA_PIXEL, QA_CLEAR_COUNT)
- `sentinel2_YYYY_05_06_10m_composite.tif` ×5 — B2,B3,B4,B8 (÷10000), VALID_COUNT
- `sentinel2_YYYY_05_06_20m_composite.tif` ×5 — B5–B12 (÷10000), SCL, VALID_COUNT
- `lulc_YYYY_05_06_10m.tif` ×5 — Dynamic World mode label (float)

After download: assistant ingests into `v2/data/raw/` per its README,
extends manifest/checksums, re-runs the coverage audit on the W4 rasters,
and closes Phase 2. Console logs (scene IDs, dates, cloud %, valid-pixel
estimates) are saved to `v2/data/phase2/gee_console_log_*.txt` as provenance.

### 9.1 As-delivered verification (2026-10-04) — Phase 2 closeout

All 20 exports were produced by the user's GEE run (console logs:
`v2/data/phase2/gee_logs/console_log_0{1,2,3}_*.txt`), downloaded, ingested
into `v2/data/raw/`, and audited by `v2/src/v2/phase3/verify_inputs.py`:
**PREFLIGHT PASS — all inputs satisfy the V2 W4 contract.**

Per-year valid coverage inside the study area (gates: L9 ≥90%, S2 ≥85%,
LULC ≥80%):

| Year | L9 ST_B10 | S2 10m | S2 20m | LULC | L9 transform = authoritative |
|---|---|---|---|---|---|
| 2022 | 96.99% | 96.44% | 96.96% | 96.93% | exact |
| 2023 | 96.98% | 96.93% | 96.96% | 96.93% | exact |
| 2024 | 96.99% | 96.93% | 96.96% | 96.93% | exact |
| 2025 | 96.99% | 96.93% | 96.96% | 96.93% | exact |
| 2026 | 96.99% | 96.89% | 96.96% | 96.93% | exact |

vs the V1 July archive (§6): L9 ST 88.4/86.6/69.2/41.9/96.4%, S2-10m
82.1/73.5/40.4/24.6/96.9%, LULC 6.1/34.2/95.3/1.1/92.6% — the V2 W4 dataset
is uniform across years at ~97% on every layer (the ~3% shortfall vs the
100% export region is the GAUL-export vs OSM-polygon boundary difference,
constant across years). Two WARN-level notes, both handled: the S2 10m
B2 zero-fill pattern exists in 2022 (84,909 px) and 2026 (7,867 px) at far
smaller magnitude than V1 2026 (3.0 M px), with the B2>0 guard active.
Checksums: `v2/data/raw/SHA256SUMS.txt` (20 files, independently re-hashed
and matched against the preflight audit).

---

## 10. Reproducibility

- Code: `v2/src/v2/phase2/` — `inventory.py`, `audit_grid.py`,
  `audit_nodata.py`, `coverage_v1_july.py`, `manifest.py`,
  `window_investigation.py`, `window_phenomenology.py`,
  `fetch_met_superset.py`, `fetch_osm_constraints.py`, `verify_spotchecks.py`
  (13/13 independent checks PASS).
- GEE: `gee/v2/01…03…js` (headers document every V1→V2 delta).
- Manifests: `v2/data/phase2/manifest/manifest.json`, `SHA256SUMS.txt`.
- Verification: grid fidelity of all window maps to the authoritative
  transform (diff 0.0); scene-level cloud vs per-pixel valid fraction
  anticorrelation confirmed (Spearman ρ = −0.777 over 456 S2 scenes);
  met grid values reproduce V1's published 2023-07 mean.

## 11. STOP statement

Phase 2 ends here. **Phase 3 has not been started** and no V2 model work
has occurred. Open items for the reviewer: (a) run the three GEE scripts;
(b) approve/adjust the F3 B2-guard and F4 grid-registration plan, which
become Phase-3 requirements; (c) confirm Dynamic World remains the V2 LULC
product (alternatives: ESA WorldCover, annual but 2020/2021-only epochs —
Dynamic World stays the scientifically consistent 5-year choice).

*V2 Phase 2 · GreenGrid AI · all artifacts under `v2/data/phase2/`*
