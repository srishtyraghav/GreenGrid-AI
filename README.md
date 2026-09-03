# GreenGrid AI

> **Where is the city hot? Why is it hot? Where should trees be planted? How many trees are required? How much cooling can be expected?**

An AI-powered Urban Heat Island (UHI) detection and tree plantation planning system for the National Capital Territory of Delhi, India.

---

## Project Overview

GreenGrid AI is a 12-phase academic minor project that combines satellite remote sensing, GIS analysis, and machine learning to:

1. Detect and map Urban Heat Island zones across Delhi NCT
2. Identify areas with highest heat stress
3. Determine optimal locations for tree plantation
4. Estimate the number of trees required
5. Predict temperature reduction from proposed tree cover

---

## Study Area

**National Capital Territory of Delhi (Delhi NCT), India**

| Attribute        | Value                        |
|------------------|------------------------------|
| Area             | ≈ 1,484 km²                  |
| Population       | ≈ 34 million (2025)          |
| Longitude extent | 76.84°E – 77.35°E            |
| Latitude extent  | 28.40°N – 28.88°N            |
| CRS              | EPSG:4326 (WGS84)            |
| Boundary source  | OpenStreetMap (Relation 1942586) |

Delhi NCT was selected because it is one of Asia's most extreme Urban Heat Islands, with peak summer temperatures regularly exceeding 48°C and documented LST differences of 8–12°C between urban cores and green areas.

---

## Project Pipeline

```
Phase 1  → Problem Definition & Research        ✅ Complete
Phase 2  → Dataset Collection                   ✅ Complete
Phase 3  → Data Preprocessing                   ✅ Complete
Phase 4  → Feature Extraction                   ✅ Complete
Phase 5  → AI-Based UHI Detection               ✅ Complete
Phase 6  → UHI Severity Classification          ⬜ Pending Phase 5
Phase 7  → Tree Plantation Suitability          ⬜ Pending Phase 6
Phase 8  → Tree Requirement Estimation          ⬜ Pending Phase 7
Phase 9  → Temperature Reduction Prediction     ⬜ Pending Phase 8
Phase 10 → Visualization & Integration          ⬜ Pending Phase 9
Phase 11 → Testing & Validation                 ⬜ Pending Phase 10
Phase 12 → Documentation & Presentation        ⬜ Pending Phase 11
```

---

## Phase 2 — Dataset Collection

### Objective
Collect a validated, structured dataset of satellite imagery and GIS data for Delhi NCT, covering July 2022 (historical baseline) and July 2026 (current condition).

### Data Sources

| Source | Product | Purpose | Period |
|--------|---------|---------|--------|
| USGS Landsat 9 | Collection 2 Level-2 SR+ST | Thermal (LST), vegetation, urban bands | Jul 2022, Jul 2026 |
| ESA Sentinel-2 | Level-2A Surface Reflectance | High-res vegetation/land-cover | Jul 2022, Jul 2026 |
| OpenStreetMap | Various | Roads, land use, vegetation, buildings | 2026-08-27 |

### Phase 2 Status

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Delhi NCT study-area boundary | ✅ COMPLETED |
| Task 2 | GEE Landsat 9 collection script | ✅ COMPLETED |
| Task 3 | GEE Sentinel-2 collection script | ✅ COMPLETED |
| Task 4 | GEE export scripts | ✅ COMPLETED |
| Task 5 | GIS data download (OSM) | ✅ COMPLETED |
| Task 6 | Dataset metadata CSV | ✅ COMPLETED |
| Task 7 | Data validation scripts | ✅ COMPLETED |
| Task 8 | Phase 2 report | ✅ COMPLETED |
| Task 9 | GEE export → Google Drive | ✅ COMPLETED (local GeoTIFFs present) |
| Task 10 | Download GeoTIFFs from Drive | ✅ COMPLETED (local GeoTIFFs present) |

---

## Phase 3 — Data Preprocessing

### Objective
Convert the validated Phase 2 raw datasets into a clean, analysis-ready, machine-learning-ready spatial dataset on a common 30 m grid.

### Phase 3 Status

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Calculate NDVI, NDBI, and LST from correct sensor bands | ✅ COMPLETED |
| Task 2 | Align Landsat 9 and Sentinel-2 to 30 m common grid | ✅ COMPLETED |
| Task 3 | Preserve existing GEE cloud mask and handle NaN pixels | ✅ COMPLETED |
| Task 4 | Rasterize OSM vector layers (landuse, roads, vegetation, buildings) | ✅ COMPLETED |
| Task 5 | Build ML-ready feature table with spatial block IDs | ✅ COMPLETED |
| Task 6 | Validate preprocessing outputs | ✅ COMPLETED |
| Task 7 | Phase 3 report | ✅ COMPLETED |

### Run Phase 3

```bash
source .venv/bin/activate

# Run the full preprocessing pipeline
python3 -m src.preprocessing.pipeline

# Validate the outputs
python3 -m src.preprocessing.validate_preprocessing
```

### Phase 3 Outputs

- `data/processed/phase3/indices/` — native-resolution NDVI, NDBI, and LST rasters
- `data/processed/phase3/aligned/` — all indices aligned to the common 30 m grid
- `data/processed/phase3/masks/` — valid-pixel mask and rasterized GIS layers
- `data/processed/phase3/ml/` — ML-ready feature tables (`feature_table.csv`, `feature_table_2022.csv`, `feature_table_2026.csv`, `feature_metadata.json`)
- `reports/phase3_preprocessing_report.md` — detailed Phase 3 report

### Key Decisions

- **30 m common grid:** chosen because Landsat 9 LST is only available at 30 m; Sentinel-2 bands are aggregated to this resolution using mean resampling.
- **NaN handling:** source GeoTIFFs have `nodata=None` but use `NaN` for masked pixels. Phase 3 treats `NaN`/`Inf` as invalid and preserves legitimate zero/negative reflectance values.
- **No second cloud mask:** Phase 2 GEE scripts already masked clouds using `QA_PIXEL` (Landsat 9) and `SCL` (Sentinel-2).
- **Spatial blocks:** each feature-table row includes a `spatial_block_id` so later phases can perform spatially aware cross-validation instead of naive random splits.

---

## Phase 4 — Feature Extraction

### Objective

Derive the final environmental-feature set on top of the validated Phase 3 outputs. The only new calculation is proportional Vegetation Cover; all other indices are reused unchanged.

### Phase 4 Status

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Define authoritative feature sources (LST=L9, NDVI/NDBI/VegCover=S2) | ✅ COMPLETED |
| Task 2 | Derive proportional Vegetation Cover from Sentinel-2 NDVI | ✅ COMPLETED |
| Task 3 | Assemble combined urban environmental dataset (one row per pixel-year) | ✅ COMPLETED |
| Task 4 | Compute descriptive statistics and exploratory correlations | ✅ COMPLETED |
| Task 5 | Generate report-ready maps (LST, NDVI, NDBI, Vegetation Cover) | ✅ COMPLETED |
| Task 6 | Validate Phase 4 outputs | ✅ COMPLETED |
| Task 7 | Phase 4 report | ✅ COMPLETED |

### Run Phase 4

```bash
source .venv/bin/activate

# Run the full feature-extraction pipeline (includes map generation)
PYTHONPATH=src python3 -m features.pipeline

# Validate the outputs
PYTHONPATH=src python3 -m features.validate_features

# Run without maps (faster for testing)
PYTHONPATH=src python3 -m features.pipeline --skip-maps
```

### Phase 4 Outputs

- `data/processed/phase4/features/` — Vegetation Cover rasters (`vegetation_cover_2022_30m.tif`, `vegetation_cover_2026_30m.tif`)
- `data/processed/phase4/tables/` — `combined_urban_environmental_dataset.csv` (300,000 rows × 14 columns)
- `data/processed/phase4/maps/` — LST, NDVI, NDBI, and Vegetation Cover maps for 2022 and 2026
- `data/processed/phase4/phase4_feature_metadata.json` — dataset and statistics metadata
- `data/processed/phase4/phase4_pipeline_record.json` — reproducibility record
- `reports/phase4_feature_extraction_report.md` — detailed Phase 4 report

### Key Decisions

- **Authoritative sources:** LST comes from Landsat 9; NDVI, NDBI, and Vegetation Cover come from Sentinel-2.
- **Vegetation Cover:** proportional vegetation cover derived from Sentinel-2 NDVI using `PVC = (NDVI − 0.05) / (0.80 − 0.05)`, clamped to [0, 1]. The reference values are a Phase 4 methodological assumption documented in the report.
- **Continuous NDVI preserved:** Vegetation Cover is stored alongside NDVI, not as a replacement.
- **Spatial blocks retained:** `spatial_block_id` is carried into the combined dataset for later spatially aware validation.
- **No Phase 5 modelling:** UHI detection and ML training are intentionally out of scope for Phase 4.

---

## Phase 5 — AI-Based UHI Detection & Classification

### Objective

Implement the project’s first machine-learning component: classify 30 m pixels into relative LST heat-severity categories (Low, Moderate, High, Severe) using Phase 4 environmental features.

### Phase 5 Status

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Define per-year quartile target from LST | ✅ COMPLETED |
| Task 2 | Prevent target leakage (exclude `lst_C`, coordinates, `spatial_block_id`) | ✅ COMPLETED |
| Task 3 | Add block-aware spatial neighbourhood features | ✅ COMPLETED |
| Task 4 | Implement Random Forest classifier | ✅ COMPLETED |
| Task 5 | Implement XGBoost classifier | ✅ COMPLETED |
| Task 6 | Spatial GroupKFold validation | ✅ COMPLETED |
| Task 7 | Temporal generalization experiments | ✅ COMPLETED |
| Task 8 | Feature ablation and year-sensitivity experiments | ✅ COMPLETED |
| Task 9 | Generate predictions, rasters, and maps | ✅ COMPLETED |
| Task 10 | Validate Phase 5 outputs | ✅ COMPLETED |
| Task 11 | Clean-run reproducibility test | ✅ COMPLETED |
| Task 12 | Phase 5 report | ✅ COMPLETED |
| Task 13 | Urban morphology feature experiment | ✅ COMPLETED |
| Task 14 | OOF diagnostics (OOF predictions, confusion matrices, boundary analysis, permutation importance) | ✅ COMPLETED |
| Task 15 | Generalization & overfitting audit (gap, regularization, seeds, permutation, LOBO, distance stress test, nested CV, locked holdout) | ✅ COMPLETED |

### Run Phase 5

```bash
source .venv/bin/activate

# Run the full Phase 5 pipeline (includes spatial features, maps, and tables)
PYTHONPATH=src python3 -m models.pipeline

# Validate the outputs
PYTHONPATH=src python3 -m models.validate_models

# Run without map generation (faster for testing)
PYTHONPATH=src python3 -m models.pipeline --skip-maps

# Generalization / overfitting audit (development diagnostics, ~5 h)
PYTHONPATH=src python3 scripts/run_phase5_audit.py

# Locked geographic holdout — evaluated exactly once, after the
# development results are reviewed and the configuration is frozen
PYTHONPATH=src python3 scripts/run_phase5_audit.py --final

# Morphology implementation reference test (numerical exactness)
PYTHONPATH=src python3 tests/test_morphology_reference.py
```

### Phase 5 Outputs

- `data/processed/phase5/tables/predictions.csv` — final pixel-level predictions and probabilities
- `data/processed/phase5/tables/oof_predictions.csv` — out-of-fold predictions per model/fold
- `data/processed/phase5/tables/model_comparison.csv` — model comparison metrics
- `data/processed/phase5/tables/classification_metrics.csv` — per-fold metrics (incl. train/validation gap)
- `data/processed/phase5/tables/ablation_results.csv` — ablation experiment results
- `data/processed/phase5/tables/feature_importance.csv` — selected model feature importance
- `data/processed/phase5/tables/permutation_importance.csv` — RF permutation importance
- `data/processed/phase5/tables/confusion_matrices.csv` — per-fold and aggregated confusion matrices
- `data/processed/phase5/tables/boundary_case_analysis.csv` — accuracy vs distance-to-threshold
- `data/processed/phase5/tables/morphology_features.csv` — cached morphology features
- `data/processed/phase5/predictions/uhi_2022.tif` — 2022 UHI severity raster
- `data/processed/phase5/predictions/uhi_2026.tif` — 2026 UHI severity raster
- `data/processed/phase5/probabilities/probability_*.tif` — per-class probability rasters
- `data/processed/phase5/maps/uhi_2022.png` — 2022 UHI severity map
- `data/processed/phase5/maps/uhi_2026.png` — 2026 UHI severity map
- `data/processed/phase5/phase5_model_metadata.json` — model and target metadata
- `data/processed/phase5/phase5_pipeline_record.json` — reproducibility record
- `data/processed/phase5/phase5_validation_report.json` — validation report
- `reports/phase5_uhi_detection_report.md` — detailed Phase 5 report

Audit diagnostics (`data/processed/phase5/tables/`): `phase5_baseline_manifest.json`,
`train_validation_gap.csv` (+ `_summary`), `rf_regularization.csv`, `xgb_regularization.csv`,
`seed_stability.csv`, `prediction_stability.csv`, `target_permutation_test.csv`,
`learning_curves.csv`, `leave_one_block_out.csv`, `spatial_distance_stress_test.csv`,
`feature_permutation_sanity.csv`, `data_integrity_duplicate_audit.csv`, `leakage_audit.csv`,
`nested_spatial_cv_results.csv`, `compact_feature_results.csv`, `calibration_report.csv`,
`spatial_error_autocorrelation.csv`, `year_feature_investigation.csv`,
`locked_geographic_holdout.csv`, `phase5_generalization_summary.json`.

### Key Results

| Model | Mean Accuracy | Macro F1 | High Recall | Severe Recall |
|---|---|---|---|---|
| Majority Baseline | 24.76% | 0.0984 | 0.0000 | 0.0000 |
| Stratified Baseline | 24.97% | 0.2442 | 0.2508 | 0.2495 |
| NDVI/NDBI Rule Baseline | 36.33% | 0.3547 | 0.2801 | 0.4247 |
| Random Forest (no spatial context) | 42.65% | 0.3999 | 0.3692 | 0.4720 |
| XGBoost (no spatial context) | 42.66% | 0.3965 | 0.3383 | 0.4882 |
| Random Forest (spatial means) | 44.99% | 0.4232 | 0.4008 | 0.4867 |
| XGBoost (spatial means) | 45.32% | 0.4215 | 0.3750 | 0.5265 |
| **Random Forest (spatial + morphology)** | **46.28%** | **0.4306** | **0.4017** | **0.5104** |
| XGBoost (spatial + morphology) | 46.29% | 0.4266 | 0.3811 | 0.5308 |

Generalization-audit results (see report §20): regularized RF-C
(`max_depth=15`, `min_samples_leaf=10`) reaches **46.99% / 0.4310** with the
train/validation gap halved (28.9 pp vs 53.3 pp). Locked geographic holdout
(unseen blocks 3, 11, 17): **39.9% accuracy / 0.395 macro-F1** — consistent
with LOBO and far-band stress tests; no leakage (permutation test at chance);
seed macro-F1 range 0.0011; nested-CV outer macro-F1 identical to flat CV
(0.4306). **The 60% target is not met; the trustworthy ceiling is ~47%
(adjacent-block CV) / ~40% (unseen geography).**

### Key Decisions

- **Target:** Per-year LST quartiles produce balanced, interpretable relative heat-severity classes.
- **Leakage prevention:** `lst_C`, `spatial_block_id`, and coordinates are excluded from predictors.
- **Spatial validation:** GroupKFold on `spatial_block_id` prevents spatial leakage.
- **Spatial features:** Block-aware 3×3, 5×5, and 11×11 focal means improve macro F1 by ~0.023–0.025.
- **Morphology features:** Block-aware 50/100/250/500 m urban-form features (building/road/vegetation pixel fractions, vegetation-cover means, land-use composition/entropy, NDVI/NDBI contrast) add a further ~+0.007 macro F1 and +1 pp accuracy over the spatial-means benchmark.
- **Model selection:** Random Forest selected by deterministic hierarchy (macro F1 → hotspot recall → stability → interpretability); it essentially ties XGBoost.
- **Audit selection (Phase 5 extension):** RF-C (`max_depth=15`, `min_samples_leaf=10`) selected over the baseline RF — highest macro-F1, best hotspot recall, half the overfitting gap, better fold stability.
- **Scientific framing:** Outputs are described as relative heat-severity classification, not independently measured physical UHI intensity; the audit confirms the binding constraint is the four-class quartile target (40.8% accuracy within 0.5 °C of a boundary vs 52.3% on clear pixels), not overfitting.

---

## Folder Structure

```
GreenGrid-AI/
│
├── data/
│   ├── raw/
│   │   ├── landsat9/
│   │   │   ├── 2022_07/       ← Landsat 9 July 2022 (local GeoTIFFs)
│   │   │   └── 2026_07/       ← Landsat 9 July 2026 (local GeoTIFFs)
│   │   ├── sentinel2/
│   │   │   ├── 2022_07/       ← Sentinel-2 July 2022 (local GeoTIFFs)
│   │   │   └── 2026_07/       ← Sentinel-2 July 2026 (local GeoTIFFs)
│   │   └── gis/
│   │       ├── study_area/    ← Delhi NCT boundary (COMPLETED)
│   │       ├── administrative/← District boundaries (COMPLETED)
│   │       ├── roads/         ← OSM road network (COMPLETED)
│   │       ├── landuse/       ← OSM land use (COMPLETED)
│   │       ├── vegetation/    ← OSM green areas (COMPLETED)
│   │       └── buildings/     ← OSM building footprints (COMPLETED)
│   ├── processed/             ← Phase 3 & 4 outputs
│   │   ├── phase3/            ← Phase 3 outputs
│   │   └── phase4/            ← Phase 4 outputs
│   └── final/                 ← Phase 5+ outputs
│
├── notebooks/                 ← Jupyter notebooks (Phase 3+)
│
├── src/
│   ├── data_collection/       ← Phase 2 scripts
│   │   ├── create_study_area.py
│   │   ├── validate_study_area.py
│   │   ├── download_gis_data.py
│   │   └── validate_data.py
│   ├── preprocessing/         ← Phase 3 scripts
│   │   ├── config.py
│   │   ├── pipeline.py
│   │   ├── indices.py
│   │   ├── align.py
│   │   ├── cloud_mask.py
│   │   ├── vector_raster.py
│   │   ├── feature_table.py
│   │   ├── io.py
│   │   └── validate_preprocessing.py
│   ├── features/              ← Phase 4 scripts
│   │   ├── config.py
│   │   ├── io.py
│   │   ├── vegetation_cover.py
│   │   ├── features.py
│   │   ├── maps.py
│   │   ├── pipeline.py
│   │   └── validate_features.py
│   ├── models/                ← Phase 5–9 scripts
│   └── utils/                 ← Shared utilities
│
├── gee/                       ← Google Earth Engine scripts
│   ├── landsat9_collection.js ← Explore L9 imagery
│   ├── sentinel2_collection.js← Explore S2 imagery
│   └── export_data.js         ← Export to Google Drive
│
├── maps/                      ← Map outputs
├── models/                    ← Trained model files
├── reports/
│   ├── phase2_dataset_report.md
│   ├── phase2_validation_report.md
│   ├── phase3_preprocessing_report.md
│   └── phase4_feature_extraction_report.md
│
├── dataset_metadata.csv       ← Phase 2 dataset catalog
├── requirements.txt           ← Python dependencies
└── README.md
```

---

## Setup Instructions

### 1. Python Environment

```bash
# Install required Python packages
pip3 install geopandas rasterio shapely pyproj fiona \
             earthengine-api geemap requests pandas numpy
```

### 2. Create Study Area Boundary

```bash
# Creates data/raw/gis/study_area/study_area.geojson
python3 src/data_collection/create_study_area.py

# Validate the boundary
python3 src/data_collection/validate_study_area.py
```

### 3. Download GIS Data from OpenStreetMap

```bash
# Downloads roads, land use, vegetation, buildings, admin boundaries
python3 src/data_collection/download_gis_data.py
```

---

## Google Earth Engine Instructions

### Requirements
- Free Google account
- Sign up at: https://earthengine.google.com (academic use is free)

### Step 1 — Explore imagery

1. Go to: https://code.earthengine.google.com
2. Open `gee/landsat9_collection.js` → paste into editor → click **Run**
3. Open `gee/sentinel2_collection.js` → paste into editor → click **Run**
4. Check the **Console** panel for actual image dates and cloud cover
5. Record the actual image IDs, dates, and tile IDs in `dataset_metadata.csv`

### Step 2 — Export data

1. Open `gee/export_data.js` → paste into editor → click **Run**
2. Go to the **Tasks** tab (top-right in GEE editor)
3. Click **RUN** for each of the 8 export tasks:
   - L9_2022_07_Delhi_BestScene_30m
   - L9_2022_07_Delhi_Composite_30m
   - L9_2026_07_Delhi_BestScene_30m
   - L9_2026_07_Delhi_Composite_30m
   - S2_2022_07_Delhi_10m_Composite
   - S2_2022_07_Delhi_20m_Composite
   - S2_2026_07_Delhi_10m_Composite
   - S2_2026_07_Delhi_20m_Composite
4. Files export to your Google Drive folder: **GreenGridAI_Phase2**
5. Download them to the local project directories:
   - `data/raw/landsat9/2022_07/`
   - `data/raw/landsat9/2026_07/`
   - `data/raw/sentinel2/2022_07/`
   - `data/raw/sentinel2/2026_07/`

---

## Validation Instructions

```bash
# Validate all Phase 2 data (study area + GIS + satellite GeoTIFFs)
python3 src/data_collection/validate_data.py

# View the validation report
cat reports/phase2_validation_report.md
```

---

## Phase 2 Output

The Phase 2 pipeline produces:

```
Delhi NCT boundary (EPSG:4326)
        ↓
Landsat 9 — July 2022 (historical baseline)
        ↓
Landsat 9 — July 2026 (current condition)
        ↓
Sentinel-2 — July 2022 (historical baseline)
        ↓
Sentinel-2 — July 2026 (current condition)
        ↓
GIS layers (roads, land use, vegetation, buildings)
        ↓
Dataset metadata catalog (dataset_metadata.csv)
        ↓
Validated structured dataset → ready for Phase 3
```

---

## Technology Stack

| Component          | Technology                    |
|--------------------|-------------------------------|
| Satellite processing | Google Earth Engine          |
| GIS analysis       | QGIS + GeoPandas             |
| Python             | Python 3.12                   |
| Raster processing  | Rasterio                      |
| Vector processing  | GeoPandas                     |
| Data handling      | Pandas, NumPy                 |

---

## Data Licenses

| Dataset | License |
|---------|---------|
| Landsat 9 | USGS — Public Domain |
| Sentinel-2 | ESA Copernicus — Free and Open |
| OpenStreetMap | ODbL 1.0 (attribution required) |

---

## Faculty Q&A Reference

**Why Delhi NCT?** → Asia's most extreme documented UHI; complete data availability; 1,484 km² computationally tractable; active policy relevance for tree planting.

**Why July 2022 and July 2026?** → July = peak UHI season (post-heat-wave, monsoon onset); 4-year gap captures meaningful urban change; Landsat 9 launched Sep 2021 so July 2022 is the earliest valid July.

**Why Landsat 9?** → Improved thermal sensor (TIRS-2, reduced stray light); Collection 2 Level-2 provides atmospherically corrected SR + ST products; free access.

**Why Sentinel-2?** → 10 m spatial resolution (3× better than Landsat for vegetation mapping); 5-day revisit; essential for NDVI accuracy in urban environments.

**Why use two satellite sources?** → Landsat 9 provides thermal data (essential for LST/UHI); Sentinel-2 provides high-resolution vegetation mapping. Neither alone is sufficient.

**Why historical data?** → UHI analysis requires temporal comparison to quantify change and validate the model against known conditions.

**Why not calculate LST/NDVI yet?** → Phase 2 is data collection only. Index calculation requires preprocessing (Phase 3): scale factors, cloud masking, reprojection, co-registration — all of which must be applied systematically before analysis.

**What happens in Phase 3?** → Applied scale factors → computed LST from ST_B10 → computed NDVI/NDBI → aligned all rasters to a common 30 m grid → rasterized OSM GIS layers → produced analysis-ready feature tables. See `reports/phase3_preprocessing_report.md` and `src/preprocessing/`.

---

*GreenGrid AI | Phase 2 — Dataset Collection*

## Dataset Access

Large satellite GeoTIFF files are not stored in this GitHub repository
because of GitHub file-size limitations.

The following datasets are stored separately:

- Landsat 9 — July 2022
- Landsat 9 — July 2026
- Sentinel-2 — July 2022
- Sentinel-2 — July 2026

To reproduce the project:

1. Clone this repository.
2. Obtain the Phase 2 satellite datasets from the shared project storage.
3. Place the files in the corresponding `data/raw/` directories.
4. Run the validation scripts.

Google Drive Link: https://drive.google.com/drive/folders/1JzJqUIF4HcZT0-Z4HG9PWjnPxq0i9NQT?usp=sharing
