"""V2 Phase 8 — tree requirement estimation ("how many trees, and where").

VERBATIM port of ``v1/scripts/run_phase8_tree_requirement.py`` (production
build spec section 11) to V2, computed for BOTH Phase 7 scenarios
(``v1_parity`` / ``v2_constrained``). Consumes ``v2/data/phase7/{scenario}/``
priority zones, Phase 6 severity rasters, and the Phase 3/4
vegetation_cover + static landuse (Phase 4 code path).

Frozen locked decisions (ported — full text in ``build.DEVIATIONS``):
  1. plantable = Phase 7 priority-zone pixels (kept 8-connected suitability
     class >= 3 zones) AND NOT exclusion AND landuse-eligible AND
     vegetation_cover < 0.30. Excluded area per zone is attributed to exactly
     one cause with precedence constraint > landuse_ineligible > veg_threshold
     (constraint rasters REUSED from Phase 7, never re-derived; exclusions
     applied AFTER scoring on the same normalization as the full valid
     domain — scenario differences are exclusion effects, not
     re-normalization effects).
  2. Density 1,000 trees/ha -> **90** trees/px (0.09 ha/px; V1's spec-text
     "9" is a documented decimal slip); 400 / 1,000 / 2,500 trees/ha are
     PLANNING-DENSITY SCENARIOS — none is claimed scientifically optimal;
     the optimal density will be evaluated after Phase 9 using cooling
     predictions.
  3. Zone priority from Phase 7 ``priority_ranking`` order; High/Medium/Low
     by ranking thirds (ceil(n/3) each).
  4. Per-zone report: location (centroid), priority, zone area, excluded
     area (+cause breakdown), plantable area, vegetation %, UHI severity
     (mean severity_score + High-class %), recommended trees at ALL 3
     densities.
  5. All 5 years; **2026 is the primary planning snapshot** (relative,
     per-year-normalized products — NOT an absolute inter-annual heat
     claim); 2022-2025 are supplementary snapshots.

**v2_constrained is the PRIMARY scenario** (constraint-aware); v1_parity is
the V1-comparable BASELINE.

Landuse eligibility rule (extracted verbatim from the V1 runner):
``eligible = LANDUSE_RULE_STATUS != 'DISCOURAGED'`` — excluded codes
{5 industrial, 7 retail}; nodata (255) keeps the frozen neutral class-0
eligibility value 50 and remains a weakly identified uncertainty.

Outputs are a RELATIVE plantation-suitability decision-support ranking on
the Phase 7 relative heat-severity classification / UHI hotspot proxy — not
physical UHI intensity or planting-success predictions. Figures (matplotlib)
are SKIPPED in V2 per the phase6/7 briefs; vectors (recommended-plantation
polygons, 2026 focus points) ARE written.
"""
