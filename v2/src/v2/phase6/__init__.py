"""V2 Phase 6 — UHI severity mapping with the frozen primary model.

Applies the FROZEN V2 Phase 5 primary production model (Random Forest, per
the marker ``v2/data/phase5/phase5_primary_model.json`` — loaded via the
marker, never hardcoded) to the full valid 30 m grid for 2022–2026 and
builds the production severity products (output taxonomy mirrored from
``v1/scripts/run_phase6_production.py``, adapted to V2 paths and the frozen
3-class target):

  rasters/    severity_{year}.tif        uint8 0/1/2, nodata 255 (argmax)
              probability_{year}.tif     float32 3-band Low/Moderate/High,
                                         band sums = 1, nodata -1.0
              confidence_{year}.tif      float32 max-prob, nodata -1.0
              severity_score_{year}.tif  float32 probs @ [0,1,2], nodata -1.0
  hotspots/   hotspots_def{A,B,C}_{year}.geojson  polygons (>= 10 px clusters)
              hotspot_ids_def{A,B,C}_{year}.tif    int32 ids, nodata -1
  tables/     area_statistics_{year}.csv      class distribution (+area, %)
              block_statistics_{year}.csv   per spatial block: class hist +
                                            per-def hotspot px/ha
              hotspot_statistics_def{A,B,C}_{year}.csv  per-def summary
              temporal_comparison.csv     5-year summary
              coverage_statistics.csv     vs Phase-3 masks + Phase-2 gates
  phase6_manifest.json / phase6_pipeline_record.json

Hotspot tiers are a faithful port of V1's spec-section-8 definitions:
  A: severity == 2 (High)
  B: severity == 2 AND confidence >= 0.60
  C: severity_score >= 1.5 AND confidence >= 0.60
8-connected components >= 10 px; outside-domain pixels never join a cluster.
Deviations vs V1, by design: probabilities are written as ONE 3-band raster
(the brief's "3-band class-probability rasters"); no LST copy rasters
(Phase 3 owns LST); no figures; green/built & vegetation-temperature tables
omitted (not requested). The model is a DIRECT 3-class classifier — no LST
thresholding step (thresholds_by_year is provenance only).

Feature construction reuses the Phase 4 code path
(``v2.phase4.assemble_features`` / ``v2.phase4._spatial``) at FULL grid
(sampling off), so mapped features are schema-identical to the training
tables; the frozen 178-name schema + sha256 are asserted before predicting.

Usage:
    PYTHONPATH=v2/src .venv/Scripts/python.exe -m v2.phase6.build \
        [--years 2022,2023,...] [--out v2/data/phase6]
"""
