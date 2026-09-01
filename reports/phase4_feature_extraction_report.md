# GreenGrid AI — Phase 4 Feature Extraction Report

**Project:** GreenGrid AI  
**Phase:** 4 — Feature Extraction  
**Study Area:** National Capital Territory (NCT) of Delhi, India  
**Report Date:** 2026-09-01  
**Status:** ✅ COMPLETED — Pipeline executed successfully | Validation 31 PASS / 0 FAIL / 0 WARN

---

## 1. Objective

Phase 4 converts the validated, analysis-ready Phase 3 outputs into the final set of environmental features and visual products required by the project synopsis. The objective is to:

- Reuse the audited Phase 3 remote-sensing indices without recalculating them.
- Derive the one new methodological product: **proportional Vegetation Cover**.
- Make an explicit, documented choice about which sensor is authoritative for each feature when both Landsat 9 and Sentinel-2 versions exist.
- Preserve continuous NDVI separately from the derived Vegetation Cover layer.
- Keep the 2022 and 2026 time steps strictly separate.
- Retain `spatial_block_id` for spatially aware validation in later phases.
- Produce report-ready maps for LST, NDVI, NDBI and Vegetation Cover.
- Assemble a single **Combined Urban Environmental Dataset** with one row per pixel-year.

No UHI detection, no land-use classification, and no model training are performed in Phase 4; those belong to later phases.

---

## 2. Relationship to Phase 3

- **Phase 3** calculated NDVI, NDBI and LST, aligned all layers to a common 30 m grid, built a combined valid-pixel mask, rasterised OSM context layers, and produced a sampled feature table with spatial-block identifiers.
- **Phase 4** builds directly on those outputs. It does **not** re-derive NDVI, NDBI or LST; it treats the Phase 3 GeoTIFFs as authoritative inputs.
- The only new calculation in Phase 4 is **Vegetation Cover**, derived from the authoritative Sentinel-2 NDVI.

All Phase 3 files remain untouched.

---

## 3. Input Datasets

| Dataset | Source | Phase 3 Output Path | Resolution | CRS | Phase 4 Use |
|---|---|---|---|---|---|
| LST 2022 | Landsat 9 ST_B10 | `data/processed/phase3/aligned/l9_2022_composite_lst_30m.tif` | 30 m | EPSG:4326 | Authoritative thermal feature |
| LST 2026 | Landsat 9 ST_B10 | `data/processed/phase3/aligned/l9_2026_composite_lst_30m.tif` | 30 m | EPSG:4326 | Authoritative thermal feature |
| NDVI 2022 (S2) | Sentinel-2 B8/B4 | `data/processed/phase3/aligned/s2_2022_ndvi_30m.tif` | 30 m (mean-aggregated from 10 m) | EPSG:4326 | Authoritative NDVI; input to Vegetation Cover |
| NDVI 2026 (S2) | Sentinel-2 B8/B4 | `data/processed/phase3/aligned/s2_2026_ndvi_30m.tif` | 30 m (mean-aggregated from 10 m) | EPSG:4326 | Authoritative NDVI; input to Vegetation Cover |
| NDBI 2022 (S2) | Sentinel-2 B11/B8A | `data/processed/phase3/aligned/s2_2022_ndbi_30m.tif` | 30 m (mean-aggregated from 20 m) | EPSG:4326 | Authoritative built-up index |
| NDBI 2026 (S2) | Sentinel-2 B11/B8A | `data/processed/phase3/aligned/s2_2026_ndbi_30m.tif` | 30 m (mean-aggregated from 20 m) | EPSG:4326 | Authoritative built-up index |
| Valid-pixel mask | Phase 3 | `data/processed/phase3/masks/valid_mask_30m.tif` | 30 m | EPSG:4326 | Implicitly preserved via NaN masks |
| Land-use raster | OSM | `data/processed/phase3/masks/landuse_raster_30m.tif` | 30 m | EPSG:4326 | Context layer |
| Road distance | OSM | `data/processed/phase3/masks/roads_distance_30m.tif` | 30 m | EPSG:4326 | Context layer |
| Vegetation distance | OSM | `data/processed/phase3/masks/vegetation_distance_30m.tif` | 30 m | EPSG:4326 | Context layer |
| Building distance | OSM | `data/processed/phase3/masks/buildings_distance_30m.tif` | 30 m | EPSG:4326 | Context layer |
| Phase 3 feature table | Phase 3 | `data/processed/phase3/ml/feature_table.csv` | Tabular | — | Source of row/col/geometry/spatial_block_id |

Landsat 9 NDVI and NDBI rasters also exist from Phase 3 but are **not** used as the authoritative spectral features in the combined dataset. They remain available for comparison or sensitivity analysis if required.

---

## 4. Methodology

### 4.1 Authoritative Feature Selection

Both Landsat 9 and Sentinel-2 provide NDVI and NDBI. Phase 4 makes an explicit, documented choice:

| Feature | Authoritative Source | Rationale |
|---|---|---|
| LST | Landsat 9 (Band 10) | Only Landsat 9 carries a thermal band in the project archive. |
| NDVI | Sentinel-2 (B8/B4) | Higher native resolution (10 m) aggregated to 30 m gives a sharper vegetation signal than Landsat 9's native 30 m bands. |
| NDBI | Sentinel-2 (B11/B8A) | Aggregated from 20 m; chosen to match the authoritative NDVI sensor for internal consistency. |
| Vegetation Cover | Derived from Sentinel-2 NDVI | Same source as authoritative NDVI; avoids mixing sensors in the vegetation proxy. |

This choice is recorded in `data/processed/phase4/phase4_feature_metadata.json` under `authoritative_feature_sources`.

### 4.2 Vegetation Cover Derivation

Phase 4 introduces **proportional Vegetation Cover (PVC)** as a continuous 0–1 layer. Because no project-specific vegetation-cover threshold or end-member spectrum was defined in earlier phases, the following standard empirical approach is adopted and documented as a Phase 4 methodological assumption:

```text
PVC = (NDVI − NDVI_soil) / (NDVI_veg − NDVI_soil)
PVC = clamp(PVC, 0.0, 1.0)
```

with:

- `NDVI_soil = 0.05` — lower reference. Bare soil, built-up surfaces and very sparse vegetation in arid/sub-tropical urban settings commonly fall in the 0.0–0.1 range; 0.05 is a conservative centre-of-range value for "effectively non-vegetated".
- `NDVI_veg = 0.80` — upper reference. Dense, healthy green vegetation in the Delhi region during the growing season can reach 0.7–0.9; 0.80 represents near-full vegetation cover and is consistent with the upper bound commonly used in proportional-cover studies.

**Important caveats recorded in the metadata:**

- PVC is a **proportional proxy** for vegetation abundance, not a direct physical measurement of canopy cover percentage.
- The reference values are a **Phase 4 methodological assumption**, not derived from site-specific end-member spectra.
- The original continuous NDVI raster is retained separately; PVC does not replace it.

PVC is calculated for both 2022 and 2026 from the authoritative Sentinel-2 NDVI rasters.

### 4.3 Combined Urban Environmental Dataset

The dataset is assembled by extracting values from the authoritative rasters at the same sampled pixel locations used in Phase 3. It therefore inherits the Phase 3 valid-pixel mask and spatial-block identifiers.

Final columns:

| Column | Description |
|---|---|
| `lon` | Longitude (EPSG:4326) |
| `lat` | Latitude (EPSG:4326) |
| `row` | Pixel row in the 30 m reference grid |
| `col` | Pixel column in the 30 m reference grid |
| `spatial_block_id` | Spatial block for cross-block validation |
| `year` | 2022 or 2026 |
| `lst_C` | Land Surface Temperature (°C) from Landsat 9 |
| `ndvi` | NDVI from Sentinel-2 |
| `ndbi` | NDBI from Sentinel-2 |
| `vegetation_cover` | Proportional vegetation cover derived from Sentinel-2 NDVI |
| `landuse_class` | Integer land-use class from OSM, inherited from Phase 3 |
| `dist_road_m` | Euclidean distance to nearest motorway/trunk/primary/secondary/tertiary road (m), inherited from Phase 3 |
| `dist_vegetation_m` | Euclidean distance to nearest OSM park/forest/green-area feature (m), inherited from Phase 3 |
| `dist_building_m` | Euclidean distance to nearest building in the Central Delhi sample (m), inherited from Phase 3 |

The dataset contains 300,000 rows: 150,000 per year.

### 4.4 Map Generation

Eight report-ready PNG maps are generated, one for each variable–year combination:

| Map | Variable | Year | Colormap | Value range |
|---|---|---|---|---|
| `lst_2022.png` | LST | 2022 | `hot` | data-driven |
| `lst_2026.png` | LST | 2026 | `hot` | data-driven |
| `ndvi_2022.png` | NDVI | 2022 | `RdYlGn` | [-0.2, 1.0] |
| `ndvi_2026.png` | NDVI | 2026 | `RdYlGn` | [-0.2, 1.0] |
| `ndbi_2022.png` | NDBI | 2022 | `Spectral_r` | [-0.5, 0.5] |
| `ndbi_2026.png` | NDBI | 2026 | `Spectral_r` | [-0.5, 0.5] |
| `vegetation_cover_2022.png` | Vegetation Cover | 2022 | `YlGn` | [0.0, 1.0] |
| `vegetation_cover_2026.png` | Vegetation Cover | 2026 | `YlGn` | [0.0, 1.0] |

Maps are produced with Matplotlib using masked arrays so that invalid pixels appear transparent. The plotted rasters themselves are not altered.

---

## 5. Outputs

### 5.1 New Raster Feature

```text
data/processed/phase4/features/
├── vegetation_cover_2022_30m.tif
└── vegetation_cover_2026_30m.tif
```

Both rasters match the Phase 3 reference grid (EPSG:4326, 1,874 × 1,768 pixels).

### 5.2 Combined Dataset

```text
data/processed/phase4/tables/
└── combined_urban_environmental_dataset.csv
```

300,000 rows × 14 columns; one row per pixel-year.

### 5.3 Maps

```text
data/processed/phase4/maps/
├── lst_2022.png
├── lst_2026.png
├── ndvi_2022.png
├── ndvi_2026.png
├── ndbi_2022.png
├── ndbi_2026.png
├── vegetation_cover_2022.png
└── vegetation_cover_2026.png
```

### 5.4 Metadata and Records

```text
data/processed/phase4/
├── phase4_feature_metadata.json
├── phase4_pipeline_record.json
└── phase4_validation_report.json
```

---

## 6. Validation Results

Phase 4 validation was run with `src/features/validate_features.py`.

**Result: 31 PASS / 0 FAIL / 0 WARN**

Checked items include:

- All expected output files exist (rasters, CSV, metadata, pipeline record, 8 maps).
- Vegetation Cover rasters match the reference grid (CRS, shape, transform).
- Vegetation Cover values lie within [0, 1].
- Vegetation Cover formula is consistent with the input NDVI rasters.
- Combined dataset contains all required columns and has no NaN in feature columns.
- Year values are exactly {2022, 2026} and are balanced (150,000 each).
- No duplicate `(row, col, year)` combinations.
- All sampled `(row, col)` coordinates lie within the reference raster bounds.
- 50-row raster/table spot check passes.
- Metadata documents authoritative sources and vegetation-cover methodology.

---

## 7. Scientific Audit

A manual audit was performed on the Phase 4 outputs in addition to the automated validation.

### 7.1 Vegetation Cover

| Statistic | 2022 | 2026 |
|---|---|---|
| Valid pixels (full raster) | 1,651,319 | 1,914,362 |
| Min | 0.0000 | 0.0000 |
| Max | 1.0000 | 1.0000 |
| Mean (full raster) | 0.3779 | 0.3888 |
| Mean (sampled dataset) | 0.3820 | 0.3914 |
| Pearson r(PVC, NDVI) | 0.9993 | 0.9966 |

The very high correlation with NDVI is expected because PVC is a monotonic transformation of NDVI. The small difference between full-raster and sampled means reflects the spatial sampling design inherited from Phase 3.

Using a simple descriptive threshold of `PVC ≥ 0.5`:

- 2022: 35.8% of sampled pixels classified as "vegetated".
- 2026: 36.9% of sampled pixels classified as "vegetated".

These percentages describe the 150,000-pixel sampled dataset used for modelling, not a census of the entire NCT Delhi 30 m grid. The sampled dataset inherits the spatial sampling design from Phase 3 (stratified by valid pixels and spatial blocks), so the sample means differ slightly from the full-raster means (e.g., full-raster PVC mean 2022 = 0.3779; sampled mean = 0.3820).

This threshold is reported as an exploratory summary, not a formal land-cover classification or an official project threshold.

### 7.2 Feature Ranges (Sampled Dataset)

| Feature | Year | Min | Max | Mean |
|---|---|---|---|---|
| LST (°C) | 2022 | 11.78 | 54.81 | 38.73 |
| LST (°C) | 2026 | 21.81 | 51.72 | 34.93 |
| NDVI | 2022 | -0.2205 | 0.8598 | 0.3353 |
| NDVI | 2026 | -0.3417 | 0.9996 | 0.3407 |
| NDBI | 2022 | -0.6102 | 0.3915 | -0.0843 |
| NDBI | 2026 | -0.5398 | 0.7164 | -0.1204 |
| Vegetation Cover | 2022 | 0.0000 | 1.0000 | 0.3820 |
| Vegetation Cover | 2026 | 0.0000 | 1.0000 | 0.3914 |

- 2022 LST minimum ~11.8°C for two sampled pixels matches the Phase 3 anomaly (water/cloud-shadow edge) and is preserved, not silently corrected.
- Sentinel-2 NDVI contains no sampled values outside [-1, 1]. The known Landsat 9 2022 NDVI > 1 anomaly does not propagate into the authoritative Sentinel-2 NDVI used here.
- NDBI values are retained as computed from the authoritative Sentinel-2 B11/B8A rasters. Negative NDBI values are physically valid (vegetation/water). The 2026 sampled maximum of 0.7164 occurs in a sparsely vegetated western-Delhi pixel and was retained; more extreme full-raster values (>0.80) occur where LST is masked and NDVI is strongly negative (water/wetland edge pixels) and therefore do not enter the combined dataset.

### 7.3 Exploratory Correlations (Pooled 2022 + 2026)

|  | LST | NDVI | NDBI | Vegetation Cover |
|---|---|---|---|---|
| LST | 1.000 | -0.296 | 0.415 | -0.301 |
| NDVI | -0.296 | 1.000 | -0.815 | 0.999 |
| NDBI | 0.415 | -0.815 | 1.000 | -0.820 |
| Vegetation Cover | -0.301 | 0.999 | -0.820 | 1.000 |

Interpretation (descriptive only):

- Vegetation cover and NDVI are almost perfectly correlated because the former is derived from the latter.
- LST is negatively correlated with vegetation/NDVI and positively correlated with NDBI, consistent with the expected urban heat-island pattern.
- NDBI is strongly negatively correlated with NDVI (−0.815) and Vegetation Cover (−0.820), as expected for a vegetation vs. built-up contrast.

### 7.4 Map Visual Inspection

- **Vegetation Cover 2026**: Green tones align with known parks, forest patches and the river corridor; pale/yellow tones align with built-up and bare areas.
- **LST 2026**: Built-up central Delhi appears hot (yellow/red), while the Yamuna river corridor appears cooler (dark/black).
- **NDVI 2026**: Vegetated areas are green; built-up/water areas are red/orange.
- **NDBI 2026**: Built-up areas show high values (red/orange); vegetated and water areas show low values (blue/green). A visually prominent high-NDBI region in western Delhi (around 76.95°E, 28.72°N, near the Najafgarh wetland/drain area) was investigated: the strongest full-raster values coincide with strongly negative NDVI and masked LST (NaN), consistent with water/wetland pixels that fall outside the combined valid-pixel mask. The highest NDBI retained in the sampled dataset is 0.7164 at row 600, col 453 (LST = 31.25 °C, NDVI = 0.059), which is high but physically plausible for a sparsely vegetated wet/bare area.

All maps display spatial patterns consistent with the underlying July 2022 / July 2026 satellite datasets recorded in `dataset_metadata.csv` (e.g., cool river corridor, hot built-up core, green vegetated patches).

### 7.5 Year Separation and Spatial Blocks

- `year` takes only values 2022 and 2026.
- `spatial_block_id` is preserved from Phase 3; the same blocks appear in both years with identical sample counts per block. It is retained for future spatially aware validation (e.g., leave-one-block-out cross-validation) and must be used instead of naive random train/test splitting in later modelling phases.
- No pixel-level mixing of years occurs in the combined dataset.
- The uniqueness criterion `(row, col, year)` is correct because the same geographic pixel can legitimately appear once in 2022 and once in 2026, but should never appear twice within the same year.
- Phase 4 does not perform any train/test split, model training, or prediction.

### 7.6 Reproducibility

A clean-run test was performed: the entire `data/processed/phase4/` directory was deleted and the pipeline was rerun. All per-year feature means matched the first run to at least 1e-9 precision:

| Feature | 2022 | 2026 |
|---|---|---|
| LST | MATCH | MATCH |
| NDVI | MATCH | MATCH |
| NDBI | MATCH | MATCH |
| Vegetation Cover | MATCH | MATCH |

The pipeline is deterministic given the fixed random seed and the stable Phase 3 inputs.

---

## 8. Limitations and Assumptions

1. **Vegetation Cover reference values**: `NDVI_soil = 0.05` and `NDVI_veg = 0.80` are standard empirical values, not site-specific end-members derived from high-resolution reference imagery. PVC should be treated as a proportional proxy, not a literal canopy percentage.
2. **Sentinel-2 NDBI band choice**: Phase 3 used B11 (SWIR) and B8A (narrow NIR) for S2 NDBI. This choice is carried forward unchanged.
3. **No cloud mask beyond Phase 3**: The valid-pixel mask is inherited from Phase 3; no additional cloud-screening is applied in Phase 4.
4. **Static context layers**: Road, vegetation and building distance layers are treated as static (same for 2022 and 2026) because OSM data were collected once.
5. **Map colour scales**: Colour ranges are chosen for visual interpretability and do not alter the underlying analytical rasters.
6. **No causal modelling**: Correlations reported in this phase are exploratory and descriptive only.

---

## 9. How to Reproduce

From the repository root with the Phase 3 `.venv` environment activated:

```bash
# Run the full Phase 4 pipeline (includes map generation)
PYTHONPATH=src python3 -m features.pipeline

# Run validation
PYTHONPATH=src python3 -m features.validate_features

# Run the pipeline without maps (faster for testing)
PYTHONPATH=src python3 -m features.pipeline --skip-maps
```

Expected runtime on the reference machine is approximately 12 seconds for the full pipeline.

---

## 10. Files Added or Modified

### New source code

```text
src/features/
├── __init__.py
├── config.py
├── io.py
├── vegetation_cover.py
├── features.py
├── maps.py
├── pipeline.py
└── validate_features.py
```

### New reports

```text
reports/phase4_feature_extraction_report.md
```

### Modified documentation

```text
README.md
```

---

## 11. Next Steps

Phase 4 is complete. The outputs are ready for:

- **Phase 5 — UHI Detection / Heat Mapping**: Use `lst_C` as the response variable and the remaining features as predictors or stratification variables.
- **Phase 6+ — Modelling**: Use `spatial_block_id` for spatially aware train/validation splits; do not use naive random splits.

Before moving on, it is recommended that the project supervisor review the Vegetation Cover methodology assumptions documented in Section 4.2.
