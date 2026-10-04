# GreenGrid AI — V2 Phase 2 GEE Runbook

These scripts rebuild the satellite/LULC dataset for V2 with the **approved
W4 window (May 1 – Jun 30, all five years 2022–2026)**. They are cloned from
the verified V1 recipes with only the documented V2 changes (W4 window,
empty-collection coverage guards, per-pixel observation-count bands).
**Landsat 9 only** (user decision 2026-10-04: exact sensor parity with V1;
Landsat 8 was evaluated and excluded). See the header comments of each script.

V1 scripts in `gee/` remain the frozen historical record — do not re-run
them; they would queue duplicate July exports.

## How to run

1. Open https://code.earthengine.google.com (the same account used for V1).
2. For each script, in order:
   - `01_export_landsat9_w4.js`
   - `02_export_sentinel2_w4.js`
   - `03_export_lulc_w4.js`
3. Paste → **Run**. Read the Console **before** opening Tasks:
   - any line starting with `ERROR` means that year's exports were **not**
     queued — that is the guard working. Tell me; do not work around it.
   - note the per-year scene IDs, dates, cloud percentages, and valid-pixel
     estimates — copy the console output to a file
     (`v2/data/phase2/gee_console_log_<script>.txt`) for provenance.
4. Tasks tab → RUN every queued task (up to 30 total: 5 L8/9, 10 S2, 5 LULC).
5. Exports land in Google Drive folder **`GreenGridAI_V2_Phase2`**.
6. Download the files (Drive → select all → Download) and leave them in
   `Downloads/`. Then tell me — I move them into `v2/data/raw/` per
   `v2/data/raw/README.md`, extend the manifest/checksums, and finish the
   Phase 2 audit.

## Expected outputs

| Script | Files | Bands |
|---|---|---|
| 01 | `landsat9_YYYY_05_06_composite_30m.tif` ×5 | SR_B2–B7, ST_B10 (raw DN), QA_PIXEL, QA_CLEAR_COUNT |
| 02 | `sentinel2_YYYY_05_06_10m_composite.tif` ×5 | B2,B3,B4,B8 (÷10000), VALID_COUNT |
| 02 | `sentinel2_YYYY_05_06_20m_composite.tif` ×5 | B5,B6,B7,B8A,B11,B12 (÷10000), SCL, VALID_COUNT |
| 03 | `lulc_YYYY_05_06_10m.tif` ×5 | lulc (Dynamic World mode label, float) |

## What does NOT change (comparability contract)

- Same Delhi boundary (FAO/GAUL Delhi), EPSG:4326, export scales (30/10/20 m),
  median compositing, identical cloud masking (Landsat QA bits 3+4; S2 SCL
  classes 3/8/9/10; S2 scene cloud <60%), identical scaling (SR ÷scale factor,
  S2 ÷10000, ST_B10 raw DN as in V1).
- QA_PIXEL/SCL medians are exported exactly as V1 did (bitwise-meaningless
  but preserved for downstream parity).
