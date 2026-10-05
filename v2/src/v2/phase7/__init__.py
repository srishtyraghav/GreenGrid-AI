"""V2 Phase 7 — tree plantation SUITABILITY (where to plant + priority).

Ports the frozen V1 suitability formulation (``v1/src/suitability``,
``v1/scripts/run_phase7_production.py``; design spec 2026-09-04) to V2 with
two MANDATORY scenarios:

  v1_parity       no constraint exclusion — directly comparable to V1
  v2_constrained  water / building footprints / road surfaces excluded
                  (V2 owns these constraint layers; V1 explicitly documented
                  their absence as a limitation)

Need x Opportunity gating, class thresholds, zone rule and per-year
class-relative normalization are VERBATIM V1 ports (see
``suitability.py`` docstrings for the frozen constants). Tree-count
requirement estimation is Phase 8 — NOT built here.

Outputs: ``v2/data/phase7/{v1_parity,v2_constrained}/{rasters,zones,tables}/``
(scenario-tagged filenames), constraint rasters under
``v2/data/phase7/constraints_rasters/``, plus ``phase7_manifest.json`` and
``phase7_pipeline_record.json`` (the record carries per-year summary numbers
for the report).
"""
