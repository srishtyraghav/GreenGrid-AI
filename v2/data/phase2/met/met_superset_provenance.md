# Meteorological Superset (V2 Phase 2) — Fetch Provenance

**Script:** `src/v2/phase2/fetch_met_superset.py`
**Fetch timestamp (UTC):** 2026-10-04T07:52:17+00:00
**Requests issued this run:** 80 point-year fetches (served from raw_cache where already cached)

**IMPORTANT — 2026 is truncated at the fetch date:** today is 2026-10-04 (UTC) and the archive API rejects future end dates (verified HTTP 400), so the 2026 year runs only through **2026-10-04** (6,648 h). The nominal spec (43,824 h/point incl. a full 2026) is therefore not yet reachable; complete 2026 by re-running this script after 2026-12-31 — the 2022–2025 cache is reused and only the 2026 point-years refetch (cache filenames embed the end date).

## Source
- **API:** Open-Meteo Historical Weather API — `https://archive-api.open-meteo.com/v1/archive` (no key required)
- **Underlying data:** ECMWF **ERA5 / ERA5-Land** reanalysis (Copernicus Climate Data Store), served under **CC-BY 4.0** per Open-Meteo terms (https://open-meteo.com/en/terms, https://cds.climate.copernicus.eu).
- Mirrors V1 provenance in `reports/experiments/exp_met_provenance.md`, but fetches a **temporal superset**: full hourly 2022–2026 per point instead of July-only means.

## API template (verbatim)
```
https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={YYYY}-01-01&end_date={YYYY}-12-31 (clamped to fetch date for the current year)&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,shortwave_radiation,soil_moisture_0_to_7cm&timezone=UTC
```
One request per grid point per calendar year (16 points x 5 years = 80 point-year requests; each is a single point-year call against Open-Meteo, retried with exponential backoff, with a 1.5s polite delay between requests).

## Variables (hourly) & units (as returned by the API)
- `temperature_2m` — °C
- `relative_humidity_2m` — %
- `wind_speed_10m` — km/h
- `precipitation` — mm
- `shortwave_radiation` — W/m²
- `soil_moisture_0_to_7cm` — m³/m³

## Grid definition
- Same 16-point 4x4 grid as V1, spanning lat 28.40–28.88 x lon 76.83–77.35 (inclusive endpoints).
- Grid provenance: **reused from C:\GreenGrid-AI\data\processed\experiments\exp_met_features_grid_provenance.csv**.
- `point_id` p01–p16 ordered row-major: lat ascending outer, lon ascending inner
  (p01 = 28.40,76.83 … p16 = 28.88,77.35).

## Output
- `data/v2/phase2/met/met_superset_hourly_2022_2026.csv` — long format:
  `point_id, lat, lon, time_utc, temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation, shortwave_radiation, soil_moisture_0_to_7cm`
- Rows: 667,392 = 16 points x 41,712 hours
  (2022–2025 full years incl. leap 2024 = 35,040 h; 2026 through 2026-10-04 = 6,648 h).
  Sanity assertion **PASS** (0 duplicate point-time rows).
- Raw per-(point,year) JSON cached under `data/v2/phase2/met/raw_cache/` before parsing; reruns do not refetch.

## Spatial-resolution warning (explicit)
ERA5-Land native resolution is ~0.1° (~9–11 km). These fields are **synoptic-scale only**;
any aggregation to pixels must NOT claim 30 m precision (same rule as V1: interpolate, don't imply native resolution).

## NaN summary (this fetch)
```
{
  "temperature_2m": 0,
  "relative_humidity_2m": 0,
  "wind_speed_10m": 0,
  "precipitation": 0,
  "shortwave_radiation": 0,
  "soil_moisture_0_to_7cm": 0
}
```

## Sanity cross-check vs V1
2023-07, 04:00–06:00 UTC, grid-mean `temperature_2m` = **30.47 °C**
(V1 `exp_met_provenance.md` reports 29.97 °C for the same window; order-of-magnitude match ~30 °C).

## License note
Data delivered by Open-Meteo under **CC-BY 4.0**; underlying ERA5/ERA5-Land (C) Copernicus Climate Change Service. Attribution: "Weather data by Open-Meteo.com" + Copernicus CDS citation.
