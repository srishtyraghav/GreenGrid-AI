# GreenGrid AI V2 — Phase 7: Tree Plantation Suitability (Production Run)

**Status: COMPLETE — verification gate 30/30 PASS (2026-10-05)**
**Inputs: Phase 6 relative heat-severity products (frozen RF primary) + Phase 2 constraint layers**
**Formulation: frozen V1 suitability spec, verbatim port (weights predefined, not tuned)**

---

## 1. Objective

Rank every valid 30 m pixel in Delhi for potential plantation suitability for
each study year (2022–2026, W4 May–Jun window), and delimit priority planting
zones — the "where to plant" layer. Tree-requirement quantities are Phase 8
and are intentionally NOT part of this phase.

## 2. Inputs and Lineage

| Input | Source |
|---|---|
| Heat-need drivers | `v2/data/phase6/` severity_score / severity / confidence rasters + Phase 4 feature stack (NDVI, NDBI, LST) at full grid |
| Environmental drivers | Phase 4 static/context layers (landuse class, built-up, roads, vegetation) |
| Planting constraints | `v2/data/phase2/constraints/`: osm_water_delhi, osm_buildings_delhi, osm_road_surfaces_delhi (GeoJSON → 30 m grid burn; precedence water > buildings > road surfaces) |
| Formulation | `v1/src/suitability/*` frozen 2026-09-04 design spec — verbatim constants |
| Domain | Per-year Phase 6 `severity_score` valid pixels |

## 3. Method (frozen V1 formulation)

- **Need** = 0.40·severity_score + 0.25·(1−NDVI) + 0.20·NDBI + 0.15·LST
- **Opportunity** = 0.30·landuse_eligibility + 0.25·built_up_inverse +
  0.15·road_accessibility (piecewise: 50 m corridor → 500 m optimal → 2 km
  decline) + 0.15·green_proximity (≤500 m cap) + 0.15·planting_headroom
- **Suitability** = Need × Opportunity / 100 (gated product, 0–100)
- Inputs min-max normalized per year at the p1/p99 percentiles (V1 spec).
- Classes: Very Low <20, Low 20–40, Medium 40–60, High 60–80, Very High ≥80
  (boundaries belong to the lower class). Priority zones: 8-connected
  components ≥ 10 px of class ≥ 3 (High + Very High).
- **Two scenarios**: `v1_parity` (no exclusions — V1-comparable) and
  `v2_constrained` (water/buildings/road-surface pixels excluded, with
  per-cause exclusion accounting). Exclusions in `v2_constrained` are applied
  **after scoring**: suitability is computed once on the full valid domain and
  the constrained scenario subsets those scores (normalization percentiles are
  likewise computed on the full valid domain). Scenario differences are
  therefore pure exclusion effects, not re-normalization effects — a
  constrained-domain pixel carries exactly the score it would have in
  `v1_parity`.

## 4. Verification Gate

Independent re-derivation (`v2.phase7.verify`, log `_phase7_verify.log`):
**30/30 PASS** — grid/CRS/dims; NoData == Phase 6 domain in both scenarios;
constraint rasters exactly equal a fresh re-burn of the GeoJSONs (all 3
layers); class/priority consistency; tables↔rasters reconciliation for every
scenario-year; both scenarios present; manifest/record complete. Tests:
**90/90 pass** (11 new suitability/constraint tests).

## 5. Results — Priority Planting Area (class ≥ 3 = High+)

| Year | v1_parity class≥3 (ha) | zones | v2_constrained class≥3 (ha) | zones | Mean suitability (both) |
|---|---|---|---|---|---|
| 2022 | 915.3 | 114 | 619.9 | 75 | 27.3 / 26.8 |
| 2023 | 901.0 | 112 | 574.4 | 91 | 27.2 / 26.1 |
| 2024 | 109.1 | 19 | 88.3 | 17 | 28.8 / 28.9 |
| 2025 | 213.8 | 26 | 136.8 | 17 | 26.6 / 26.3 |
| 2026 | 44.9 | 11 | 43.2 | 11 | 27.2 / 27.3 |

Priority **High** tier (Very High class) is **0 ha in all years** — the frozen
gated product (Need×Opp/100 with these weights) rarely crosses 80; this is the
verbatim V1 formula's honest outcome, not a defect. "Priority Medium" area
equals the class≥3 area above.

**Constraint accounting** (v2_constrained): ~355.5–355.7k px/year excluded
(≈32.0k ha) — overwhelmingly buildings (~303k px) and water (~53k px), road
surfaces minor (~0.5k px); domain shrinks from ~1.91M to ~1.554M px.

## 6. Honest Notes and Caveats

1. **Snapshots, not trends (critical):** every input is min-max normalized
   *within each year* (p1/p99). Year-to-year differences in class≥3 area
   (915 → 109 → 45 ha) therefore reflect each year's internal score
   distribution, NOT a worsening/improving planting situation. Valid use:
   within-year spatial ranking. Invalid use: comparing priority area across
   years. This is the frozen V1 design ("snapshots, not trends").
2. Outputs are **potential suitability — a relative decision-support
   ranking**. They are not a statement of legal availability, land ownership,
   plantability guarantee, or survival probability.
3. Heat drivers are Phase 6 **relative** classes (per-year LST tertiles, W4
   window) — not absolute UHI intensity; W4 products are not comparable to
   V1's July-based maps.
4. V1's B/C/D sensitivity scenarios were not ported (out of brief); only the
   baseline (V1 Scenario A) composition is computed, in both V2 scenarios.
5. **Unclassified/background land-use is weakly identified**: OSM-untagged
   pixels (landuse class 0) inherit the frozen neutral eligibility value of 50
   — neither clearly eligible nor excluded. Absence of an OSM tag is not
   evidence of planting eligibility; suitability on background-class land
   should be read with low confidence. (Inherited verbatim from the frozen V1
   spec, where it is flagged CONDITIONAL/UNCERTAIN.)

## 7. Output Files

Under `v2/data/phase7/` (gitignored):

- `v1_parity/`, `v2_constrained/` — each with `rasters/` (suitability score
  float32 + class int8, per year), `zones/` (priority-zone rasters + GeoJSON),
  `tables/` (class statistics, zone inventories, per-block tables, exclusion
  accounting)
- `constraints_rasters/` — burned water/buildings/road-surface masks +
  attributed exclusion raster
- `phase7_manifest.json`, `phase7_pipeline_record.json`
- Logs: `v2/logs/_phase7_train.log`, `_phase7_verify.log`

Runtime: 67.8 s total (constraint burn ~26 s once + ~2–13 s per scenario-year).

## 8. Validation Summary

| Gate | Result |
|---|---|
| Verification gate (`v2.phase7.verify`) | **30/30 PASS** |
| Full test suite | **90/90 pass** |
| Downstream readiness | Phase 8 inputs complete: priority zones, suitability scores, exclusion accounting (constraint-aware tree counts can differ by scenario) |
| Phase 8 status | **NOT started — awaiting user review of Phase 7 results** |
