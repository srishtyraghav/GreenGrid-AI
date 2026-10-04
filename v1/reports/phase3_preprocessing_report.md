# GreenGrid AI — Phase 3 Preprocessing Report

**Project:** GreenGrid AI  
**Phase:** 3 — Data Preprocessing  
**Study Area:** National Capital Territory (NCT) of Delhi, India  
**Report Date:** 2026-09-01  
**Status:** ✅ COMPLETED — Pipeline executed successfully | Validation 43 PASS / 0 FAIL / 2 WARN

---

## 1. Objective

Phase 3 converts the raw, validated Phase 2 datasets into a clean, analysis-ready, machine-learning-ready spatial dataset. The objective is to:

- Calculate spectral indices (NDVI, NDBI) and Land Surface Temperature (LST) from the correct sensor bands.
- Align all raster products to a common 30 m grid.
- Preserve the existing Phase 2 cloud mask rather than introducing an unnecessary second masking step.
- Convert OpenStreetMap vector layers into 30 m raster features.
- Produce a tabular feature dataset with explicit spatial-block identifiers so later phases can perform spatially aware validation.

No model training is performed in Phase 3; that belongs to later phases.

---

## 2. Relationship to Phase 1 and Phase 2

- **Phase 1** defined the problem, study area, and overall 12-phase pipeline.
- **Phase 2** collected and validated the raw inputs: Landsat 9, Sentinel-2, OSM GIS layers, and study-area boundary.
- **Phase 3** builds directly on Phase 2 by applying the scale factors, index formulas, and spatial alignment steps that were deliberately deferred from Phase 2 (see `gee/export_data.js` and `reports/phase2_dataset_report.md`).

All Phase 1 and Phase 2 files remain untouched.

---

## 3. Input Datasets

| Dataset | Source | File | Resolution | CRS |
|---|---|---|---|---|
| Landsat 9 July 2022 best scene | USGS / GEE | `data/raw/landsat9/2022_07/landsat9_2022_07_best_scene_30m.tif` | 30 m | EPSG:4326 |
| Landsat 9 July 2022 composite | USGS / GEE | `data/raw/landsat9/2022_07/landsat9_2022_07_composite_30m.tif` | 30 m | EPSG:4326 |
| Landsat 9 July 2026 best scene | USGS / GEE | `data/raw/landsat9/2026_07/landsat9_2026_07_best_scene_30m.tif` | 30 m | EPSG:4326 |
| Landsat 9 July 2026 composite | USGS / GEE | `data/raw/landsat9/2026_07/landsat9_2026_07_composite_30m.tif` | 30 m | EPSG:4326 |
| Sentinel-2 July 2022 10 m | ESA / GEE | `data/raw/sentinel2/2022_07/sentinel2_2022_07_10m_composite.tif` | 10 m | EPSG:4326 |
| Sentinel-2 July 2022 20 m | ESA / GEE | `data/raw/sentinel2/2022_07/sentinel2_2022_07_20m_composite.tif` | 20 m | EPSG:4326 |
| Sentinel-2 July 2026 10 m | ESA / GEE | `data/raw/sentinel2/2026_07/sentinel2_2026_07_10m_composite.tif` | 10 m | EPSG:4326 |
| Sentinel-2 July 2026 20 m | ESA / GEE | `data/raw/sentinel2/2026_07/sentinel2_2026_07_20m_composite.tif` | 20 m | EPSG:4326 |
| Study area boundary | OSM | `data/raw/gis/study_area/study_area.geojson` | Vector | EPSG:4326 |
| Roads | OSM | `data/raw/gis/roads/delhi_roads_major_osm.json` | Vector | EPSG:4326 |
| Land use | OSM | `data/raw/gis/landuse/delhi_landuse_osm.json` | Vector | EPSG:4326 |
| Vegetation | OSM | `data/raw/gis/vegetation/delhi_vegetation_osm.json` | Vector | EPSG:4326 |
| Buildings sample | OSM | `data/raw/gis/buildings/delhi_buildings_sample_osm.json` | Vector | EPSG:4326 |

---

## 4. Band Mapping Verification

Before calculating indices, band names were verified against `dataset_metadata.csv`, the GEE export scripts, and the actual GeoTIFF band descriptions.

### Landsat 9 (8 bands)

| Band | Description | Phase 3 use |
|---|---|---|
| SR_B2 | Blue | — |
| SR_B3 | Green | — |
| SR_B4 | Red | NDVI denominator/numerator |
| SR_B5 | NIR | NDVI / NDBI |
| SR_B6 | SWIR1 | NDBI |
| SR_B7 | SWIR2 | — |
| ST_B10 | Thermal (raw DN) | LST |
| QA_PIXEL | Quality / cloud mask | Already used by GEE |

### Sentinel-2 10 m (4 bands)

| Band | Description | Phase 3 use |
|---|---|---|
| B2 | Blue | — |
| B3 | Green | — |
| B4 | Red | NDVI |
| B8 | NIR | NDVI |

### Sentinel-2 20 m (7 bands)

| Band | Description | Phase 3 use |
|---|---|---|
| B5 | Red Edge 1 | — |
| B6 | Red Edge 2 | — |
| B7 | Red Edge 3 | — |
| B8A | Narrow NIR | NDBI |
| B11 | SWIR1 | NDBI |
| B12 | SWIR2 | — |
| SCL | Scene Classification | Already used by GEE for cloud masking |

**NDBI band-choice note:** The 20 m Sentinel-2 composite exported from GEE does not contain B8 (10 m NIR). Therefore native-resolution S2 NDBI is computed from B11 (SWIR1) and B8A (narrow NIR), then aggregated to 30 m. B8 is reserved for S2 NDVI at 10 m.

---

## 5. Data Preprocessing

### 5.1 Cloud / Invalid-Pixel Handling

Phase 2 GEE scripts already applied cloud masking:

- **Landsat 9:** `QA_PIXEL` bits 3 (cloud shadow) and 4 (cloud) masked out.
- **Sentinel-2:** SCL classes 3 (cloud shadow), 8 (cloud medium prob), 9 (cloud high prob), and 10 (thin cirrus) masked out.

The exported GeoTIFFs have `nodata=None`, but masked pixels are represented as `NaN`. Phase 3 therefore treats `NaN` (and `Inf`) as invalid. It does **not** apply a second cloud mask.

Observed NaN fractions in the source bands (first band inspected):

| Source | Finite Pixels | NaN Pixels | NaN Fraction |
|---|---|---|---|
| L9 2022 best scene | 1,531,325 | 1,781,907 | 53.78 % |
| L9 2022 composite | 1,743,888 | 1,569,344 | 47.37 % |
| L9 2026 best scene | 1,880,166 | 1,433,066 | 43.25 % |
| L9 2026 composite | 1,894,985 | 1,418,247 | 42.81 % |
| S2 2022 10 m | 14,587,385 | 15,198,615 | 51.03 % |
| S2 2022 20 m | 3,649,755 | 3,799,555 | 51.01 % |
| S2 2026 10 m | 17,190,848 | 12,595,152 | 42.29 % |
| S2 2026 20 m | 4,301,592 | 3,147,718 | 42.26 % |

Legitimate zero or negative reflectance values were **not** converted to NoData.

### 5.2 Index Calculation

| Index | Formula | Bands | Output |
|---|---|---|---|
| NDVI | (NIR − Red) / (NIR + Red) | L9: SR_B5, SR_B4 | `data/processed/phase3/indices/l9_*_ndvi_30m.tif` |
| NDVI | (NIR − Red) / (NIR + Red) | S2: B8, B4 | `data/processed/phase3/indices/s2_*_ndvi_native_10m.tif` |
| NDBI | (SWIR − NIR) / (SWIR + NIR) | L9: SR_B6, SR_B5 | `data/processed/phase3/indices/l9_*_ndbi_30m.tif` |
| NDBI | (SWIR − NIR) / (SWIR + NIR) | S2: B11, B8A | `data/processed/phase3/indices/s2_*_ndbi_native_20m.tif` |
| LST (°C) | `(ST_B10 × 0.00341802 + 149.0) − 273.15` | L9: ST_B10 raw DN | `data/processed/phase3/indices/l9_*_lst_30m.tif` |

### 5.3 CRS, Spatial Alignment, and Resampling

All inputs are already in **EPSG:4326**, so **no reprojection** was required. Phase 3 performs only **resampling** and **aggregation/downsampling**.

- **Reference grid:** `landsat9_2026_07_composite_30m.tif` (1,874 × 1,768 pixels).
- **Landsat 9 indices:** Already on the reference grid; copied to `data/processed/phase3/aligned/` after verifying identical width, height, transform, and CRS.
- **Sentinel-2 NDVI (10 m):** Aggregated to the reference grid using **mean** resampling.
- **Sentinel-2 NDBI (20 m):** Aggregated to the reference grid using **mean** resampling.

The common grid was chosen because Landsat 9 LST is the primary target variable and is only available at the Landsat scale. Upsampling thermal data would falsely imply genuine sub-Landsat thermal detail.

**Important note on metric resolution:** EPSG:4326 uses angular units, so the metric pixel size varies with latitude. The GEE-exported Landsat 9 scale of `0.00026949458523585647°` corresponds to approximately 30 m at the equator, but at Delhi's latitude (~28.65°N) the actual metric size is:

| Direction | Metric size |
|---|---|
| North–South | ≈ 29.87 m |
| East–West | ≈ 26.35 m |

This is a property of the geographic CRS and is not a processing error. All aligned rasters were verified to share exactly the same width, height, CRS, transform, bounds, and pixel size.

### 5.4 Combined Valid-Pixel Mask

A single valid-pixel mask was built by intersecting the NaN masks of all aligned index layers:

| Metric | Value |
|---|---|
| Total pixels | 3,313,232 |
| Valid pixels | 1,500,777 |
| Invalid pixels | 1,812,455 |
| Valid fraction | 45.30 % |

Output: `data/processed/phase3/masks/valid_mask_30m.tif` (1 = valid, 0 = invalid).

### 5.5 Vector-to-Raster Conversion

OpenStreetMap Overpass JSON files were parsed into GeoDataFrames and rasterized onto the 30 m grid.

| Layer | Output | Notes |
|---|---|---|
| Land use | `masks/landuse_raster_30m.tif` | 8 classes + unclassified/background (code 0) |
| Roads distance | `masks/roads_distance_30m.tif` | Euclidean distance to nearest motorway/trunk/primary/secondary/tertiary road |
| Vegetation distance | `masks/vegetation_distance_30m.tif` | Euclidean distance to nearest park/forest/green area |
| Buildings distance | `masks/buildings_distance_30m.tif` | Euclidean distance to nearest building in the Central Delhi sample |

Land-use class mapping:

| Code | Class |
|---|---|
| 0 | unclassified_background |
| 1 | park |
| 2 | forest |
| 3 | grass |
| 4 | commercial |
| 5 | industrial |
| 6 | residential |
| 7 | retail |
| 8 | farmland |

**Semantics note:** Code **0** does **not** represent a real OSM land-use class. It means that no OSM landuse/leisure/natural polygon covered that 30 m pixel. This is preserved in `feature_metadata.json` so that later phases do not misinterpret 0 as a genuine class.

Pixel counts on the full 30 m grid:

| Class | Pixels |
|---|---|
| unclassified_background | 2,784,500 |
| residential | 255,946 |
| industrial | 106,826 |
| farmland | 113,092 |
| forest | 24,245 |
| commercial | 23,157 |
| retail | 5,462 |
| park | 4 |

Distance-layer statistics:

| Layer | Feature Count | Min Distance (m) | Max Distance (m) | Mean Distance (m) |
|---|---|---|---|---|
| Roads | 3,620 | 0.0 | 8,409.53 | 1,765.13 |
| Vegetation | 6,468 | 0.0 | 17,131.68 | 2,282.05 |
| Buildings (sample) | 16,656 | 0.0 | 44,690.31 | 18,970.67 |

---

## 6. Feature Table

A machine-learning-ready CSV was created by sampling valid 30 m pixels. To support spatially aware validation in later phases, each sample was assigned a `spatial_block_id` based on its row/column location in a 5 × 5 tile grid.

| Attribute | Value |
|---|---|
| Random seed | 42 |
| Max samples requested | 150,000 per year |
| Valid pixels available | 1,500,777 |
| Samples drawn per year | 150,000 |
| Total rows (combined long-format table) | 300,000 |
| Unique spatial blocks observed | 20 of 25 possible |

### Output files

| File | Rows | Purpose |
|---|---|---|
| `data/processed/phase3/ml/feature_table_2022.csv` | 150,000 | 2022 samples with year-specific column names |
| `data/processed/phase3/ml/feature_table_2026.csv` | 150,000 | 2026 samples with year-specific column names |
| `data/processed/phase3/ml/feature_table.csv` | 300,000 | Combined long-format table, one row per pixel-year |
| `data/processed/phase3/ml/feature_metadata.json` | — | Column list, dtypes, summary statistics, reproducibility metadata |

### Feature columns (combined table)

`lon`, `lat`, `row`, `col`, `spatial_block_id`, `lst_l9_C`, `ndvi_l9`, `ndbi_l9`, `ndvi_s2`, `ndbi_s2`, `landuse_class`, `dist_road_m`, `dist_vegetation_m`, `dist_building_m`, `year`

### Summary statistics from the feature table

| Feature | Year | Min | Mean | Max |
|---|---|---|---|---|
| LST (°C) | 2022 | 11.78 | 38.73 | 54.81 |
| LST (°C) | 2026 | 21.81 | 34.93 | 51.72 |
| NDVI (L9) | 2022 | −0.055 | 0.434 | 1.235 |
| NDVI (L9) | 2026 | 0.036 | 0.313 | 0.792 |
| NDBI (L9) | 2022 | −0.835 | −0.137 | 0.394 |
| NDBI (L9) | 2026 | −0.438 | −0.114 | 0.277 |
| NDVI (S2) | 2022 | −0.221 | 0.335 | 0.860 |
| NDVI (S2) | 2026 | −0.342 | 0.341 | 1.000 |
| NDBI (S2) | 2022 | −0.610 | −0.084 | 0.392 |
| NDBI (S2) | 2026 | −0.540 | −0.120 | 0.716 |

---

## 7. Validation

Validation script: `src/preprocessing/validate_preprocessing.py`

### Results

| Status | Count |
|---|---|
| PASS | 43 |
| FAIL | 0 |
| WARN | 2 |

### Warnings

1. **L9 2022 NDVI maximum = 1.468** on the full raster (44 pixels > 1 out of 1.74 M finite pixels). Investigation showed that **all 44 pixels have negative red reflectance** (range −0.0309 to −0.0002) while NIR remains below 1. This is a known artifact of Landsat Collection 2 atmospheric correction over very dark surfaces (dense vegetation, shadow, or water). It is not a calculation bug. These outliers were not silently clipped; they are documented here.
2. **L9 2022 LST minimum = 11.58 °C**. Only 15 pixels are below 15 °C, located in a small cluster adjacent to masked (NaN) areas, likely a water body or cloud-shadow edge. This is treated as a real but minor outlier.

### Checks performed

- All expected output files exist.
- All aligned rasters match the reference 30 m grid (width, height, transform, CRS).
- Index value ranges are physically plausible (NDVI/NDBI within ±1 except the documented outliers; LST within 15–65 °C except the documented outlier).
- Feature tables contain no NaN or Inf values in numeric columns.
- Feature tables contain all expected columns.
- Spatial blocks are present for spatially aware cross-validation.

### 7.3 Scientific Audit

A final scientific audit was performed before locking Phase 3.

#### 30 m grid verification

All 16 aligned rasters were verified to share exactly the same width (1,874), height (1,768), CRS (`EPSG:4326`), transform, bounds, and pixel size (`0.00026949458523585647°` square). Because EPSG:4326 is geographic, the metric pixel size at Delhi's latitude (~28.65°N) is approximately **29.87 m NS** and **26.35 m EW**, not a literal 30 m in both directions.

#### LST formula verification

The source band used for LST is `ST_B10` exported as raw Digital Numbers (DN range 41,050–51,844 for the 2026 composite). Spot-checks confirmed:

```
LST (°C) = ST_B10 × 0.00341802 + 149.0 − 273.15
```

matches the stored LST rasters to machine precision. The scale factor is applied exactly once.

#### Sentinel-2 aggregation verification

For a sample 30 m pixel, the aligned S2 NDVI value (`0.082193`) was identical to the mean of the corresponding native 10 m NDVI pixels, confirming that mean aggregation is working correctly and without spatial misalignment.

#### Feature-table ↔ raster spot check

Five randomly selected feature-table rows were compared against the underlying rasters. All 45 checked values (9 features × 5 rows) matched the raster values exactly.

#### Spatial-block geographic sanity

The 20 occupied spatial blocks form a regular geographic grid over Delhi NCT. Missing blocks (0, 4, 5, 20, 21) correspond to areas with no valid pixels, primarily outside the study-area boundary or fully masked. Block IDs therefore genuinely represent spatial separation and are suitable for spatial cross-validation.

#### OSM distance-raster verification

- **Roads:** max distance 8,409.53 m, located at the western edge of the study area; consistent with full Delhi road coverage.
- **Vegetation:** max distance 17,131.68 m, at the northwest corner; consistent with the distribution of mapped green areas.
- **Buildings (sample):** max distance 44,690.31 m, at the northwest corner. This large value is explained by the building layer being a **Central Delhi sample only**; distances grow as one moves away from the sampled area. The calculation is confirmed to be distance to the nearest mapped building polygon, not distance to the raster edge.

#### Reproducibility

The pipeline was rerun from a clean `data/processed/phase3/` directory. With random seed 42, the following reproduced exactly:

| Metric | Baseline | Rerun |
|---|---|---|
| Valid pixels | 1,500,777 | 1,500,777 |
| Samples per year | 150,000 | 150,000 |
| Spatial blocks observed | 20 | 20 |
| Mean LST 2022 (°C) | 38.73299 | 38.73299 |
| Mean LST 2026 (°C) | 34.93318 | 34.93318 |
| Mean NDVI (L9) 2022 | 0.43366 | 0.43366 |
| Mean NDVI (L9) 2026 | 0.31303 | 0.31303 |

---

## 8. Generated Outputs

```
data/processed/phase3/
├── indices/                    # Native-resolution indices
│   ├── l9_*_ndvi_30m.tif
│   ├── l9_*_ndbi_30m.tif
│   ├── l9_*_lst_30m.tif
│   ├── s2_*_ndvi_native_10m.tif
│   └── s2_*_ndbi_native_20m.tif
├── aligned/                    # All indices on the common 30 m grid
│   ├── l9_*_ndvi_30m.tif
│   ├── l9_*_ndbi_30m.tif
│   ├── l9_*_lst_30m.tif
│   ├── s2_*_ndvi_30m.tif
│   └── s2_*_ndbi_30m.tif
├── masks/                      # Masks and rasterized GIS layers
│   ├── valid_mask_30m.tif
│   ├── landuse_raster_30m.tif
│   ├── roads_distance_30m.tif
│   ├── vegetation_distance_30m.tif
│   └── buildings_distance_30m.tif
├── ml/                         # ML-ready feature tables
│   ├── feature_table.csv
│   ├── feature_table_2022.csv
│   ├── feature_table_2026.csv
│   └── feature_metadata.json
└── phase3_pipeline_record.json # Full reproducibility record
```

Total processed output size: ~997 MB (gitignored under `data/processed/`).

---

## 9. Important Technical Decisions

| Decision | Rationale |
|---|---|
| **30 m common grid** | Landsat 9 LST is the primary target variable and is only available at 30 m. Aggregating Sentinel-2 to 30 m preserves thermal integrity. |
| **NaN as invalid-pixel indicator** | Source GeoTIFFs have `nodata=None` but use `NaN` for masked pixels. This convention was preserved and documented. |
| **No second cloud mask** | Phase 2 GEE scripts already masked clouds using QA_PIXEL and SCL. The NaN pattern confirms this. |
| **Mean aggregation for S2 → 30 m** | Reflectance and index values are averaged over larger pixels, which is physically appropriate for continuous variables. Spot-checks confirmed exact agreement with native-resolution means. |
| **B11 + B8A for S2 NDBI** | The exported 20 m composite contains B11 (SWIR1) and B8A (narrow NIR) but not B8, so these are the only native-resolution NDBI bands available. |
| **Spatial block IDs instead of random train/val split** | Spatial autocorrelation can leak information across random pixel splits. Spatial blocks allow later phases to use leave-one-block-out or similar spatial cross-validation. |
| **Long-format combined feature table** | One row per pixel-year avoids artificial NaN values that would appear in a wide-format table. |

---

## 10. Problems Encountered and Solutions

| Problem | Solution |
|---|---|
| `nodata=None` but masked pixels exist as `NaN` | Documented the convention and treated `NaN`/`Inf` as invalid in all processing. |
| Integer rasters (e.g., valid mask, landuse) cannot use `NaN` as nodata | Used explicit nodata values (0 for valid-mask invalid, 255 for landuse nodata) and read masks without NaN conversion. |
| Sentinel-2 NDBI band ambiguity | Verified against `dataset_metadata.csv`, GEE export script, and actual band descriptions; used B11 + B8A. |
| Overpass JSON ordering | Implemented a two-pass parser (nodes first, then ways) to handle any element order. |
| Combined wide-format feature table created NaN values | Switched to long-format combined table with generic column names plus a `year` column. |

---

## 11. Limitations

1. **Building footprints are a Central Delhi sample only.** The `dist_building_m` feature is therefore most reliable in the sampled area and less meaningful farther away.
2. **OSM landuse coverage is sparse.** The majority of pixels are classified as `unclassified_background` because OSM landuse polygons do not cover the entire study area.
3. **L9 2022 best scene has 30.91 % cloud cover.** The composite mitigates this, but some residual cloud influence may remain in the 2022 best-scene indices.
4. **A small number of index outliers exist.** L9 2022 NDVI has 44 pixels > 1; L9 2022 LST has 15 pixels < 15 °C. These were retained and documented rather than arbitrarily clipped.
5. **Distance values are approximate.** Because the rasters are in EPSG:4326, the conversion from pixel distance to metres uses a nominal 30 m factor. Actual metric pixel size varies with latitude (≈ 29.87 m NS, ≈ 26.35 m EW at Delhi). Relative distances are correct; absolute values have a small latitude-dependent error.
6. **District boundaries (Overpass relations) were not rasterized.** Relation geometries require additional handling and are not required for the current Phase 3 feature set.

---

## 12. Reproducibility Instructions

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the full Phase 3 pipeline
python3 -m src.preprocessing.pipeline

# Validate the outputs
python3 -m src.preprocessing.validate_preprocessing
```

All paths in `src/preprocessing/config.py` are repository-relative. Required inputs are the Phase 2 datasets listed in Section 3.

---

## 13. Conclusion

Phase 3 successfully converted the raw Phase 2 satellite and GIS datasets into a clean, aligned, 30 m, ML-ready feature dataset. The pipeline executed in approximately 72 seconds, produced 43 validation passes with 2 documented warnings, and preserved all Phase 1 and Phase 2 work. The outputs are ready for Phase 4 feature extraction and subsequent machine-learning phases.

---

*GreenGrid AI — Phase 3 Preprocessing Report | Generated 2026-09-01*
