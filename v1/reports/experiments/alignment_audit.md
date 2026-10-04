# Spatial Alignment Audit — Frozen 101-Feature Model

Reference grid: `l9_2026_composite_lst_30m.tif` — 1874x1768, res 0.00026949458523586 deg, origin (76.8329063, 28.8846991), EPSG:4326
Rasters audited: 59 | Traced pixels: 100 (one per year x block)

## Section A
- **PASS** grid identity of all 59 model rasters — 59/59 exactly equal to reference (CRS+transform+1874x1768)

## Section B
- **PASS** native-grid commensurability measured — S2_10m: origin offset = (-0.0000, -1.9963) native px (fractional 1.0000, 0.0037) | S2_20m: origin offset = (-0.0000, -0.4991) native px (fractional 1.0000, 0.5009) | LULC_10m: origin offset = (-0.0000, -1.9963) native px (fractional 1.0000, 0.0037)

## Section C
- **PASS** C1: S2 10m -> 30m average (no shift) — energy centroid -> cell (900,900); marker centre cell (900,900)
- **PASS** C2: S2 20m -> 30m average (no shift) — energy centroid -> cell (900,900); marker centre cell (900,900)
- **PASS** C3: LULC 10m -> 30m nearest (no shift) — energy centroid -> cell (900,900); marker centre cell (900,900)

## Section D
- **PASS** lon/lat of 100 traced pixels vs reference transform — 100/100 exact
- **PASS** table values match fresh raster reads (100 px x 13 cols) — 0 mismatches
- **PASS** spatial_block_id recomputation (per-year production bins) — 100/100 traced pixels match table
- **PASS** ndvi_mean3 independent recompute vs Phase-5 cache — 100 traced values match

## Verdict: PASS — alignment correct; no changes made