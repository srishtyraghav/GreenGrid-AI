# Meteorological Covariates — Source & Alignment Provenance

**Experiment:** `exp_met` — adds 6 environmental covariates to the frozen primary model (XGBoost + LULC, 101 predictors) and evaluates under the unchanged protocol against the 50.02% locked baseline.

## Source
- **API:** Open-Meteo Historical Weather API — `https://archive-api.open-meteo.com/v1/archive`
- **Underlying data:** ECMWF **ERA5 / ERA5-Land** reanalysis (Copernicus Climate Data Store), served as open data without a key.
- **Variables (hourly):** `temperature_2m` (°C), `relative_humidity_2m` (%), `wind_speed_10m` (km/h), `precipitation` (mm), `shortwave_radiation` (W/m²), `soil_moisture_0_to_7cm` (m³/m³) — all six requested variables, available for every year 2022–2026 (verified by test fetch before the run).
- **Request window per grid point:** 2022-07-01 → 2026-07-31 (all five Julys in one call per point).

## Spatial resolution and processing
- ERA5-Land native grid ≈ **0.1° (~9–11 km)**; we sampled a **4×4 grid (16 points)** spanning the Delhi study bbox (lat 28.40–28.88, lon 76.83–77.35), one call per point.
- Per point and year: mean over the aligned hours (below) of all July days 1–30 → 16-point field per variable per year.
- Fields were **IDW-interpolated (power 2, longitude scaled by cos 28.64°)** to each sampled pixel's lon/lat. The covariates therefore vary only at synoptic (~10 km) scale — **no false 30 m precision** was introduced.
- Interpolation used pixel lon/lat solely to locate weather values; coordinates are **not** predictors.

## Temporal alignment
- Landsat 9 overpasses Delhi on the morning descending pass (~**04:30–05:00 UTC**, 10:00–10:30 IST), and each year's product is a median composite of all clear July scenes (acquisition dates listed in `dataset_metadata.csv`).
- For each year we take the mean of hours **04:00–06:00 UTC** (acquisition window) over **July 1–30** (matching the composites' `filterDate(YYYY-07-01, YYYY-07-31)` window).
- Documented approximation: scene-level exact timestamps were not used; the 2-hour bracket covers the Landsat morning pass for every listed July scene date.

## Grid-mean values per year (Delhi average, acquisition window, July 1–30)

| Year | T2m °C | RH % | Wind km/h | Precip mm | SSR W/m² | Soil moisture m³/m³ |
|---|---|---|---|---|---|---|
| 2022 | 30.88 | 72.0 | 10.97 | 0.21 | 472.8 | 0.34 |
| 2023 | 29.97 | 77.5 | 9.01 | 0.44 | 441.5 | 0.37 |
| 2024 | 31.30 | 73.7 | 7.94 | 0.25 | 471.7 | 0.33 |
| 2025 | 30.50 | 73.6 | 8.24 | 0.18 | 450.2 | 0.32 |
| 2026 | 31.23 | 71.0 | 9.72 | 0.63 | 451.5 | 0.29 |

(Physically consistent with the series: 2022 hot/dry, 2026 warmest mornings with the most precipitation; 2025 coolest air temperatures.)

## Leakage check (explicit)
- Predictor set = the frozen 101 predictors + 6 `met_*` columns.
- **Excluded:** `lst_C` (target), `lon`/`lat`, `row`/`col`, `spatial_block_id`, and anything derived from them.
- Weather covariates are **external reanalysis fields**, not derived from LST or other predictors; they describe acquisition-time atmospheric state — legitimately knowable at prediction time, no future information (all values are within the same July window as the imagery).
- Programmatic assertion in the fetch script: no banned columns among the experiment features — **PASS**.

## Artifacts
- `data/processed/experiments/exp_met_features.csv` — (row, col, year, 6 met columns), 749,998 rows, 0 NaN
- `data/processed/experiments/exp_met_features_grid_provenance.csv` — the raw 16-point × year grid means
- `data/processed/experiments/exp_met_classification_geoeval.json` — evaluation results (CV5 / LOBO / locked)
