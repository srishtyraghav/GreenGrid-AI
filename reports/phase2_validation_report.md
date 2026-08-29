# GreenGrid AI — Phase 2 Data Validation Report

Generated: 2026-08-27 11:14:58

## Summary

| Status  | Count |
|---------|-------|
| PASSED  | 86     |
| FAILED  | 0     |
| WARNED  | 8     |
| SKIPPED | 0     |
| TOTAL   | 94     |

## Detailed Results

| Category | Check | Status | Message |
|----------|-------|--------|---------|
| Study Area | File exists | PASS | 76,990 bytes |
| Study Area | Valid JSON | PASS | File parses as valid JSON |
| Study Area | GeoJSON type | PASS | FeatureCollection |
| Study Area | Feature count | PASS | 1 features |
| Study Area | GeoPandas read | PASS | File opened successfully |
| Study Area | CRS | PASS | EPSG:4326 (WGS84) |
| Study Area | Geometry validity | PASS | All geometries are valid |
| Study Area | Spatial extent | PASS | (76.8388, 28.4046) → (77.3453, 28.8834) |
| Districts | File exists | PASS | 764,034 bytes |
| Districts | Valid JSON | PASS | File parses as valid JSON |
| Districts | GeoJSON type | PASS | Overpass JSON format (7,415 elements) |
| Districts | Element count | PASS | 7,415 OSM elements |
| Roads | File exists | PASS | 3,608,363 bytes |
| Roads | Valid JSON | PASS | File parses as valid JSON |
| Roads | GeoJSON type | PASS | Overpass JSON format (28,225 elements) |
| Roads | Element count | PASS | 28,225 OSM elements |
| Land Use | File exists | PASS | 5,515,020 bytes |
| Land Use | Valid JSON | PASS | File parses as valid JSON |
| Land Use | GeoJSON type | PASS | Overpass JSON format (49,999 elements) |
| Land Use | Element count | PASS | 49,999 OSM elements |
| Vegetation | File exists | PASS | 6,224,854 bytes |
| Vegetation | Valid JSON | PASS | File parses as valid JSON |
| Vegetation | GeoJSON type | PASS | Overpass JSON format (59,857 elements) |
| Vegetation | Element count | PASS | 59,857 OSM elements |
| Buildings | File exists | PASS | 11,153,262 bytes |
| Buildings | Valid JSON | PASS | File parses as valid JSON |
| Buildings | GeoJSON type | PASS | Overpass JSON format (107,645 elements) |
| Buildings | Element count | PASS | 107,645 OSM elements |
| Landsat9/L9 2022 Best | File exists | PASS | 43.5 MB |
| Landsat9/L9 2022 Best | File opens | PASS | GeoTIFF opens successfully |
| Landsat9/L9 2022 Best | CRS | PASS | EPSG:4326 |
| Landsat9/L9 2022 Best | Dimensions | PASS | 1874 × 1768 px, 8 band(s) |
| Landsat9/L9 2022 Best | Band count | PASS | 8 bands (min: 8) |
| Landsat9/L9 2022 Best | Extent | PASS | (76.8329, 28.4082) → (77.3379, 28.8847) |
| Landsat9/L9 2022 Best | NoData value | WARN | Not set (may be OK for raw data) |
| Landsat9/L9 2022 Best | Resolution | PASS | 0.000269 × 0.000269 degrees (~30m at equator) |
| Landsat9/L9 2022 Composite | File exists | PASS | 96.1 MB |
| Landsat9/L9 2022 Composite | File opens | PASS | GeoTIFF opens successfully |
| Landsat9/L9 2022 Composite | CRS | PASS | EPSG:4326 |
| Landsat9/L9 2022 Composite | Dimensions | PASS | 1874 × 1768 px, 8 band(s) |
| Landsat9/L9 2022 Composite | Band count | PASS | 8 bands (min: 8) |
| Landsat9/L9 2022 Composite | Extent | PASS | (76.8329, 28.4082) → (77.3379, 28.8847) |
| Landsat9/L9 2022 Composite | NoData value | WARN | Not set (may be OK for raw data) |
| Landsat9/L9 2022 Composite | Resolution | PASS | 0.000269 × 0.000269 degrees (~30m at equator) |
| Landsat9/L9 2026 Best | File exists | PASS | 50.5 MB |
| Landsat9/L9 2026 Best | File opens | PASS | GeoTIFF opens successfully |
| Landsat9/L9 2026 Best | CRS | PASS | EPSG:4326 |
| Landsat9/L9 2026 Best | Dimensions | PASS | 1874 × 1768 px, 8 band(s) |
| Landsat9/L9 2026 Best | Band count | PASS | 8 bands (min: 8) |
| Landsat9/L9 2026 Best | Extent | PASS | (76.8329, 28.4082) → (77.3379, 28.8847) |
| Landsat9/L9 2026 Best | NoData value | WARN | Not set (may be OK for raw data) |
| Landsat9/L9 2026 Best | Resolution | PASS | 0.000269 × 0.000269 degrees (~30m at equator) |
| Landsat9/L9 2026 Composite | File exists | PASS | 100.9 MB |
| Landsat9/L9 2026 Composite | File opens | PASS | GeoTIFF opens successfully |
| Landsat9/L9 2026 Composite | CRS | PASS | EPSG:4326 |
| Landsat9/L9 2026 Composite | Dimensions | PASS | 1874 × 1768 px, 8 band(s) |
| Landsat9/L9 2026 Composite | Band count | PASS | 8 bands (min: 8) |
| Landsat9/L9 2026 Composite | Extent | PASS | (76.8329, 28.4082) → (77.3379, 28.8847) |
| Landsat9/L9 2026 Composite | NoData value | WARN | Not set (may be OK for raw data) |
| Landsat9/L9 2026 Composite | Resolution | PASS | 0.000269 × 0.000269 degrees (~30m at equator) |
| Sentinel2/S2 2022 10m | File exists | PASS | 330.2 MB |
| Sentinel2/S2 2022 10m | File opens | PASS | GeoTIFF opens successfully |
| Sentinel2/S2 2022 10m | CRS | PASS | EPSG:4326 |
| Sentinel2/S2 2022 10m | Dimensions | PASS | 5620 × 5300 px, 4 band(s) |
| Sentinel2/S2 2022 10m | Band count | PASS | 4 bands (min: 4) |
| Sentinel2/S2 2022 10m | Extent | PASS | (76.8329, 28.4084) → (77.3378, 28.8845) |
| Sentinel2/S2 2022 10m | NoData value | WARN | Not set (may be OK for raw data) |
| Sentinel2/S2 2022 10m | Resolution | PASS | 0.000090 × 0.000090 degrees (~10m at equator) |
| Sentinel2/S2 2022 20m | File exists | PASS | 133.6 MB |
| Sentinel2/S2 2022 20m | File opens | PASS | GeoTIFF opens successfully |
| Sentinel2/S2 2022 20m | CRS | PASS | EPSG:4326 |
| Sentinel2/S2 2022 20m | Dimensions | PASS | 2810 × 2651 px, 7 band(s) |
| Sentinel2/S2 2022 20m | Band count | PASS | 7 bands (min: 7) |
| Sentinel2/S2 2022 20m | Extent | PASS | (76.8329, 28.4083) → (77.3378, 28.8846) |
| Sentinel2/S2 2022 20m | NoData value | WARN | Not set (may be OK for raw data) |
| Sentinel2/S2 2022 20m | Resolution | PASS | 0.000180 × 0.000180 degrees (~20m at equator) |
| Sentinel2/S2 2026 10m | File exists | PASS | 376.7 MB |
| Sentinel2/S2 2026 10m | File opens | PASS | GeoTIFF opens successfully |
| Sentinel2/S2 2026 10m | CRS | PASS | EPSG:4326 |
| Sentinel2/S2 2026 10m | Dimensions | PASS | 5620 × 5300 px, 4 band(s) |
| Sentinel2/S2 2026 10m | Band count | PASS | 4 bands (min: 4) |
| Sentinel2/S2 2026 10m | Extent | PASS | (76.8329, 28.4084) → (77.3378, 28.8845) |
| Sentinel2/S2 2026 10m | NoData value | WARN | Not set (may be OK for raw data) |
| Sentinel2/S2 2026 10m | Resolution | PASS | 0.000090 × 0.000090 degrees (~10m at equator) |
| Sentinel2/S2 2026 20m | File exists | PASS | 151.3 MB |
| Sentinel2/S2 2026 20m | File opens | PASS | GeoTIFF opens successfully |
| Sentinel2/S2 2026 20m | CRS | PASS | EPSG:4326 |
| Sentinel2/S2 2026 20m | Dimensions | PASS | 2810 × 2651 px, 7 band(s) |
| Sentinel2/S2 2026 20m | Band count | PASS | 7 bands (min: 7) |
| Sentinel2/S2 2026 20m | Extent | PASS | (76.8329, 28.4083) → (77.3378, 28.8846) |
| Sentinel2/S2 2026 20m | NoData value | WARN | Not set (may be OK for raw data) |
| Sentinel2/S2 2026 20m | Resolution | PASS | 0.000180 × 0.000180 degrees (~20m at equator) |
| Metadata | File exists | PASS | 7,031 bytes |
| Metadata | CSV readable | PASS | 14 rows |

## Overall Status

**✓ ALL CHECKS PASSED — Phase 2 data is valid.**