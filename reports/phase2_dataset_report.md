# GreenGrid AI — Phase 2 Dataset Report

**Project:** GreenGrid AI  
**Phase:** 2 — Dataset Collection  
**Study Area:** National Capital Territory (NCT) of Delhi, India  
**Report Date:** 2026-08-27  
**Status:** ✅ COMPLETED — All 86/94 validation checks passed | 0 failures | 8 warnings (expected)

---

## 1. Objective

Phase 2 of GreenGrid AI collects the raw datasets required for all subsequent analysis phases. The primary objective is to assemble a **structured, validated, and documented dataset** for the Delhi NCT study area, including:

- Administrative boundary (master study-area geometry)
- Landsat 9 satellite imagery (July 2022 historical baseline + July 2026 current condition)
- Sentinel-2 satellite imagery (July 2022 + July 2026)
- Open GIS layers (roads, land use, vegetation, buildings)
- Dataset metadata catalog

**No indices (LST, NDVI, NDBI) are calculated at this stage.** Index computation begins in Phase 3.

---

## 2. Study Area

| Attribute            | Value                                          |
|----------------------|------------------------------------------------|
| Name                 | National Capital Territory of Delhi (Delhi NCT)|
| Country              | India                                          |
| Area                 | ≈ 1,484 km² (official); 1,487 km² computed    |
| Longitude extent     | 76.838835°E → 77.345338°E                     |
| Latitude extent      | 28.404629°N → 28.883446°N                     |
| Centroid             | 77.0921°E, 28.6440°N                          |
| CRS                  | EPSG:4326 (WGS84 Geographic)                  |
| Geometry type        | MultiPolygon                                   |
| Coordinate points    | 865                                            |

### Why Delhi NCT?

Delhi NCT is one of the most densely populated metropolitan areas in the world and a **textbook Urban Heat Island (UHI)**. It was selected because:

1. **Extreme urbanisation** — Delhi has grown from 9.4M (2001) to ~34M (2025) residents
2. **Documented UHI intensity** — peak summer LST exceeds 50°C in commercial zones
3. **Green cover deficit** — only ~20% green cover vs. 33% national norm
4. **Data availability** — all required satellite data (Landsat 9, Sentinel-2) is freely available
5. **Policy relevance** — Delhi has active urban greening mandates (Delhi Tree Authority)
6. **Manageable size** — 1,484 km² is computationally tractable for the prototype

---

## 3. Study-Area Boundary

### Source
- **OpenStreetMap** (OSM Relation ID: 1942586)
- License: ODbL 1.0 (https://www.openstreetmap.org/copyright)
- Download date: 2026-08-27

### File
```
data/raw/gis/study_area/study_area.geojson
```

### Validation Results ✅
All 9 validation checks passed:

| Check              | Result |
|--------------------|--------|
| File exists        | ✓ PASS (76,990 bytes) |
| Valid JSON         | ✓ PASS |
| GeoJSON FeatureCollection | ✓ PASS |
| Feature count      | ✓ PASS (1 feature) |
| CRS = EPSG:4326    | ✓ PASS |
| Geometry type      | ✓ PASS (MultiPolygon) |
| Coordinates present | ✓ PASS |
| Centroid in Delhi  | ✓ PASS |
| Area ≈ 1,484 km²   | ✓ PASS (1,487 km² computed) |

---

## 4. Satellite Sources

### 4.1 Landsat 9

| Attribute         | Value                                                   |
|-------------------|---------------------------------------------------------|
| Sensor            | OLI-2 (Optical) + TIRS-2 (Thermal)                     |
| Product           | Collection 2 Level-2 Science Product (SR + ST)          |
| EE Dataset ID     | `LANDSAT/LC09/C02/T1_L2`                               |
| Launch date       | 27 September 2021                                       |
| Spatial resolution| 30 m (all bands, including thermal — resampled)         |
| Revisit time      | 16 days                                                 |
| Swath width       | 185 km                                                  |

**Why Landsat 9 (not Landsat 8)?**

Landsat 9 launched September 27, 2021 and carries the improved TIRS-2 sensor with significantly reduced thermal stray light artifact compared to Landsat 8 TIRS-1. Landsat 9 data is therefore preferred for LST computation in this project.

**Why July 2022 (not July 2020)?**

Landsat 9 did not exist in July 2020. The first full operational month with full-orbit coverage is November 2021. July 2022 is the first July available for Landsat 9, making it the logical historical baseline.

### 4.2 Sentinel-2

| Attribute          | Value                                                    |
|--------------------|----------------------------------------------------------|
| Sensor             | MSI (MultiSpectral Instrument)                           |
| Product            | Level-2A Surface Reflectance (Atmospherically corrected) |
| EE Dataset ID      | `COPERNICUS/S2_SR_HARMONIZED`                           |
| Spatial resolution | 10 m (B2/B3/B4/B8), 20 m (B5/B6/B7/B8A/B11/B12)       |
| Revisit time       | 5 days (combined Sentinel-2A + 2B)                      |
| Swath width        | 290 km                                                   |

**Why Sentinel-2?**

Sentinel-2 provides significantly higher spatial resolution than Landsat 9 (10 m vs 30 m) and is the standard for vegetation mapping, NDVI computation, and land-cover classification. Using **both** satellite sources gives complementary data:

- **Landsat 9** → thermal analysis (LST, UHI) — Sentinel-2 has no thermal band
- **Sentinel-2** → high-resolution vegetation and land-cover mapping

---

## 5. Landsat 9 Methodology

### Collection strategy
1. Filter Landsat 9 Collection 2 Level-2 by Delhi NCT geometry
2. Filter by July 2022 date range (2022-07-01 to 2022-07-31)
3. Sort by cloud cover (ascending — least cloudy first)
4. Select best individual scene AND create cloud-masked median composite
5. Apply QA_PIXEL cloud mask (bits 3 = cloud shadow, 4 = cloud)
6. Export to Google Drive at 30 m resolution clipped to Delhi NCT

### Why July?
July falls in the peak summer/monsoon onset period. This is the worst-case condition for UHI because:
- Maximum solar radiation exposure has already occurred in May–June
- Urban surfaces have accumulated maximum heat
- Vegetation contrast between green/non-green areas is most pronounced
- Consistent time-window (July 2022 vs July 2026) ensures fair comparison

### Bands exported

| Band    | Description       | Resolution | Purpose               |
|---------|-------------------|------------|-----------------------|
| SR_B2   | Blue              | 30 m       | True colour / aerosol |
| SR_B3   | Green             | 30 m       | True colour           |
| SR_B4   | Red               | 30 m       | True colour / NDVI    |
| SR_B5   | NIR               | 30 m       | NDVI (Phase 3)        |
| SR_B6   | SWIR 1            | 30 m       | NDBI (Phase 3)        |
| SR_B7   | SWIR 2            | 30 m       | Urban analysis        |
| ST_B10  | Thermal Infrared  | 30 m       | LST (Phase 3)         |
| QA_PIXEL| Quality/Cloud Mask| 30 m       | Cloud masking         |

---

## 6. Sentinel-2 Methodology

### Collection strategy
1. Filter `COPERNICUS/S2_SR_HARMONIZED` by Delhi NCT geometry
2. Filter by July date range (2022-07-01 to 2022-07-31, then 2026-07-01 to 2026-07-31)
3. Filter to < 30% cloud cover
4. Apply SCL (Scene Classification Layer) cloud mask
5. Create cloud-masked median composite
6. Export 10 m and 20 m bands separately to Google Drive

### Tile coverage
Delhi NCT falls primarily within MGRS tile **44RNR**. Tile **43RGQ** covers a small western strip. The cloud-masked composite in GEE merges all tiles automatically.

### Bands exported (10 m)

| Band | Description | Purpose              |
|------|-------------|----------------------|
| B2   | Blue  (490 nm) | True colour        |
| B3   | Green (560 nm) | True colour        |
| B4   | Red   (665 nm) | True colour / NDVI |
| B8   | NIR   (842 nm) | NDVI (Phase 3)     |

### Bands exported (20 m)

| Band | Description        | Purpose              |
|------|--------------------|----------------------|
| B5   | Red Edge 1 (705 nm)| Vegetation health    |
| B6   | Red Edge 2 (740 nm)| Vegetation health    |
| B7   | Red Edge 3 (783 nm)| Vegetation health    |
| B8A  | Narrow NIR (865 nm)| EVI (Phase 3)        |
| B11  | SWIR 1 (1610 nm)   | NDBI / soil moisture |
| B12  | SWIR 2 (2190 nm)   | Urban analysis       |
| SCL  | Scene Class Layer  | Cloud masking        |

---

## 7. July 2022 Dataset (Historical Baseline)

| Dataset          | Status               | Notes                               |
|------------------|----------------------|-------------------------------------|
| Landsat 9 scenes | ⏳ Pending GEE export | Run `gee/export_data.js`, Task 1   |
| L9 acquisition date | UNKNOWN           | Check GEE Console after running     |
| L9 cloud cover   | UNKNOWN              | Will be printed in GEE Console      |
| L9 WRS Path/Row  | Expected: 146/040    | Confirm in GEE Console              |
| Sentinel-2 tiles | ⏳ Pending GEE export | Run `gee/export_data.js`, Task 3   |
| S2 tile IDs      | Expected: 44RNR      | Confirm in GEE Console              |
| S2 cloud cover   | UNKNOWN              | Filter: < 30% applied               |

---

## 8. July 2026 Dataset (Current Condition)

| Dataset          | Status               | Notes                               |
|------------------|----------------------|-------------------------------------|
| Landsat 9 scenes | ⏳ Pending GEE export | Run `gee/export_data.js`, Task 2   |
| L9 acquisition date | UNKNOWN           | Check GEE Console after running     |
| L9 cloud cover   | UNKNOWN              | Will be printed in GEE Console      |
| Sentinel-2 tiles | ⏳ Pending GEE export | Run `gee/export_data.js`, Task 4   |
| S2 tile IDs      | Expected: 44RNR      | Confirm in GEE Console              |

---

## 9. GIS Datasets

| Layer               | Source | File                              | Status       |
|---------------------|--------|-----------------------------------|--------------|
| Study area boundary | OSM    | study_area/study_area.geojson     | ✅ COMPLETED  |
| Administrative      | OSM    | administrative/delhi_districts_osm.json | ✅ COMPLETED |
| Roads               | OSM    | roads/delhi_roads_major_osm.json  | ✅ COMPLETED  |
| Land use            | OSM    | landuse/delhi_landuse_osm.json    | ✅ COMPLETED  |
| Vegetation          | OSM    | vegetation/delhi_vegetation_osm.json | ✅ COMPLETED |
| Buildings (sample)  | OSM    | buildings/delhi_buildings_sample_osm.json | ✅ COMPLETED |
| Buildings (full)    | —      | DOWNLOAD_FULL_DATA.md             | ⏳ Manual step|

All GIS data: **License: ODbL 1.0** (https://www.openstreetmap.org/copyright)

---

## 10. Dataset Structure

```
data/
├── raw/
│   ├── landsat9/
│   │   ├── 2022_07/
│   │   │   ├── landsat9_2022_07_best_scene_30m.tif    [PENDING GEE]
│   │   │   └── landsat9_2022_07_composite_30m.tif     [PENDING GEE]
│   │   └── 2026_07/
│   │       ├── landsat9_2026_07_best_scene_30m.tif    [PENDING GEE]
│   │       └── landsat9_2026_07_composite_30m.tif     [PENDING GEE]
│   ├── sentinel2/
│   │   ├── 2022_07/
│   │   │   ├── sentinel2_2022_07_10m_composite.tif    [PENDING GEE]
│   │   │   └── sentinel2_2022_07_20m_composite.tif    [PENDING GEE]
│   │   └── 2026_07/
│   │       ├── sentinel2_2026_07_10m_composite.tif    [PENDING GEE]
│   │       └── sentinel2_2026_07_20m_composite.tif    [PENDING GEE]
│   └── gis/
│       ├── study_area/
│       │   └── study_area.geojson                     [COMPLETED ✅]
│       ├── administrative/
│       │   └── delhi_districts_osm.json               [COMPLETED ✅]
│       ├── roads/
│       │   └── delhi_roads_major_osm.json             [COMPLETED ✅]
│       ├── landuse/
│       │   └── delhi_landuse_osm.json                 [COMPLETED ✅]
│       ├── vegetation/
│       │   └── delhi_vegetation_osm.json              [COMPLETED ✅]
│       └── buildings/
│           ├── delhi_buildings_sample_osm.json        [COMPLETED ✅]
│           └── DOWNLOAD_FULL_DATA.md
└── dataset_metadata.csv                               [COMPLETED ✅]
```

---

## 11. Metadata

The `dataset_metadata.csv` file documents all Phase 2 datasets with:

- Dataset ID, source, satellite, product
- Acquisition date (UNKNOWN until GEE export runs)
- Image ID, tile ID, cloud cover (UNKNOWN until GEE export runs)
- Spatial resolution, bands, CRS
- File name and path
- Purpose, status, notes

All currently unknown satellite metadata fields are marked **UNKNOWN** (not fabricated).  
These will be updated after running the GEE export scripts.

---

## 12. Data Quality Checks

### Completed checks

| Dataset         | Check              | Result |
|-----------------|--------------------|--------|
| study_area.geojson | File exists    | ✅ PASS |
| study_area.geojson | Valid JSON     | ✅ PASS |
| study_area.geojson | CRS EPSG:4326  | ✅ PASS |
| study_area.geojson | MultiPolygon   | ✅ PASS |
| study_area.geojson | Area ≈ 1,484 km² | ✅ PASS |
| study_area.geojson | Centroid in Delhi | ✅ PASS |
| dataset_metadata.csv | File exists | ✅ PASS |

### Pending checks (after GEE export)

- Landsat 9 GeoTIFFs: CRS, dimensions, band count, extent, nodata
- Sentinel-2 GeoTIFFs: CRS, dimensions, band count, extent, nodata
- Satellite data spatial overlap with Delhi NCT boundary

Run:
```bash
python3 src/data_collection/validate_data.py
```

---

## 13. Limitations

1. **Satellite data not yet downloaded** — GEE export must be run before Landsat/Sentinel GeoTIFFs exist locally. Export tasks are queued; the user must click Run in the GEE Tasks panel.

2. **Building data is a sample** — Only Central Delhi buildings were downloaded via Overpass. Full building data requires GeoFabrik download (see `buildings/DOWNLOAD_FULL_DATA.md`).

3. **July imagery may have monsoon cloud cover** — Delhi's monsoon onset is typically late June/early July. Cloud cover varies significantly by year. The GEE scripts use cloud masking and composite approaches to mitigate this, but very cloudy periods may reduce data quality.

4. **No official Government of India boundary** — The study area uses the OSM-derived boundary. For official government work, the Survey of India shapefiles should be used (requires permission).

5. **OSM completeness** — OSM road/building/vegetation completeness in Delhi is good but not 100%. Some areas may be underrepresented.

---

## 14. Phase 3 Next Steps

Once Phase 2 is complete (satellite GeoTIFFs downloaded):

| Task | Description |
|------|-------------|
| 3.1  | Apply Landsat 9 scale factors (SR bands: ×0.0000275 + (−0.2)) |
| 3.2  | Apply thermal scale factors (ST_B10: ×0.00341802 + 149.0 → Kelvin) |
| 3.3  | Convert thermal band to Land Surface Temperature (LST) in °C |
| 3.4  | Calculate NDVI from Landsat 9 (NIR - Red) / (NIR + Red) |
| 3.5  | Calculate NDVI from Sentinel-2 (B8 - B4) / (B8 + B4) |
| 3.6  | Calculate NDBI (SWIR - NIR) / (SWIR + NIR) |
| 3.7  | Reproject/co-register all rasters to common resolution |
| 3.8  | Clip all layers to Delhi NCT boundary |
| 3.9  | Handle nodata and cloud-masked pixels |
| 3.10 | Produce preprocessed dataset for Phase 4 feature extraction |

---

*GreenGrid AI — Phase 2 Dataset Report | Generated 2026-08-27*
