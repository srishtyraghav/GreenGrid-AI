# v2/data/raw — V2 acquisition landing zone

V2 (W4 = May 1–Jun 30, 2022–2026) raw dataset, populated from the GEE
exports described in `v2/gee/README.md`. V1 data under `v1/data/` is frozen
and is never written by V2.

Layout:

```
landsat9/YYYY_05_06/landsat9_YYYY_05_06_composite_30m.tif      (9 bands)
sentinel2/YYYY_05_06/sentinel2_YYYY_05_06_10m_composite.tif     (5 bands)
sentinel2/YYYY_05_06/sentinel2_YYYY_05_06_20m_composite.tif     (8 bands)
lulc/YYYY_05_06/lulc_YYYY_05_06_10m.tif                         (1 band)
SHA256SUMS.txt                                                  (20 files, verified)
```

Companion data (see `v2/data/phase2/`): constraint + context vectors
(`constraints/`, OSM/Overpass), meteorology (`met/`, ERA5-Land via
Open-Meteo hourly superset), study area (`../gis/study_area.geojson`),
frozen reference schema (`../reference/`).
