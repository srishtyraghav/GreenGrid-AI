# V2 Phase 8 — Tree Requirement Estimation Report

**Product framing (read first):** all Phase 6/7/8 outputs are **relative
classifications / decision-support rankings**, not physical UHI intensity or
planting-success predictions. Phase 6 provides a **relative heat-severity
classification / UHI hotspot proxy**; Phase 7 provides a **relative
plantation-suitability decision-support ranking**; this phase turns those
rankings into **planning estimates of tree counts**. **2026 is the primary
planning snapshot** (relative, per-year-normalized products — NOT an absolute
inter-annual heat claim); 2022–2025 are supplementary snapshots.

- **Primary scenario: `v2_constrained`** (building/water/road-surface
  exclusions). **Baseline: `v1_parity`** (V1-comparable, no constraint
  exclusion). Exclusions are applied **after scoring**, on the same per-year
  normalization as the full valid domain — scenario differences are
  **exclusion effects, not re-normalization effects**.
- Densities **400 / 1,000 / 2,500 trees/ha are planning-density scenarios.
  None is claimed scientifically optimal**; the optimal density will be
  evaluated **after Phase 9** using cooling predictions.
- Planting space per locked rule: Phase 7 priority-zone pixels AND NOT
  exclusion AND landuse-eligible AND vegetation_cover < 0.30. Landuse
  eligibility = `LANDUSE_RULE_STATUS != 'DISCOURAGED'` (excluded: 5
  industrial, 7 retail); **unclassified/background land-use retains the frozen
  neutral eligibility value 50 and remains a weakly identified uncertainty**.
- Tree arithmetic: 0.09 ha/px × 1,000 trees/ha = **90 trees/px** (V1 spec
  text "9 trees/px" is a documented decimal slip — errata kept).
- Verification gate: **73/73 PASS** (`v2.phase8.verify`, exit 0; log
  `v2/logs/_phase8_verify.log`). Run wall 15.4s (build) + 7s (verify).
  Artifacts: `v2/data/phase8/{v2_constrained,v1_parity}/{rasters,tables,vectors}/`
  + manifest + pipeline record.

## 1. Headline — 2026 primary planning snapshot, v2_constrained (PRIMARY)

11 priority zones; total plantable **29.61 ha**; recommended trees:
**29,610 @1,000/ha** (11,844 @400/ha; 74,025 @2,500/ha).

| rank | zone | priority | area ha | excluded ha | plantable ha | veg % | severity (mean score / High %) | trees @400 | @1000 | @2500 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | High | 2.25 | 0.0 | 2.25 | 5.80 | 1.83 / 100% | 900 | 2,250 | 5,625 |
| 2 | 8 | High | 8.73 | 0.0 | 8.73 | 3.05 | 1.91 / 100% | 3,492 | 8,730 | 21,825 |
| 3 | 7 | High | 4.05 | 0.0 | 4.05 | 3.06 | 1.87 / 100% | 1,620 | 4,050 | 10,125 |
| 4 | 5 | High | 2.88 | 0.0 | 2.88 | 4.14 | 1.86 / 100% | 1,152 | 2,880 | 7,200 |
| 5 | 2 | Medium | 1.26 | 0.0 | 1.26 | 2.77 | 1.72 / 100% | 504 | 1,260 | 3,150 |
| 6 | 4 | Medium | 1.26 | 0.0 | 1.26 | 3.87 | 1.89 / 100% | 504 | 1,260 | 3,150 |
| 7 | 6 | Medium | 1.17 | 0.0 | 1.17 | 3.29 | 1.95 / 100% | 468 | 1,170 | 2,925 |
| 8 | 10 | Medium | 1.80 | 0.0 | 1.80 | 1.91 | 1.94 / 100% | 720 | 1,800 | 4,500 |
| 9 | 11 | Low | 3.78 | 0.0 | 3.78 | 2.63 | 1.93 / 100% | 1,512 | 3,780 | 9,450 |
| 10 | 1 | Low | 1.08 | 0.0 | 1.08 | 3.75 | 1.93 / 100% | 432 | 1,080 | 2,700 |
| 11 | 9 | Low | 1.35 | 0.0 | 1.35 | 3.96 | 1.94 / 100% | 540 | 1,350 | 3,375 |

Zone centroids: `tree_requirement_by_zone_v2_constrained_2026.csv` (all
zones sit in the 76.97–77.18°E / 28.61–28.80°N belt; recommended-plantation
polygons + 2026 focus points under `vectors/`).

## 2. 2026 baseline comparison — v1_parity (no constraint exclusion)

11 zones; plantable **31.68 ha**; **31,680 trees @1,000/ha** (12,672 @400;
79,200 @2,500). Zone set is near-identical (shared zones 1–7, 9–11; v2
re-ranks zone 10/11 because Phase 7 rebuilt zones on the constrained
domain). Constraint effect in 2026: −2.07 ha plantable, −2,070 trees
(−6.5%) — the exclusion effect is modest in 2026 because the constrained
Phase 7 domain had already removed constraint-covered area before zoning.

## 3. Totals per year per scenario (planning-density scenarios)

| scenario | year | zones | zone ha | plantable ha | trees @400 | **@1000** | @2500 |
|---|---|---|---|---|---|---|---|
| v2_constrained (PRIMARY) | 2022 | 75 | 557.7 | 557.73 | 223,092 | **557,730** | 1,394,325 |
| v2_constrained | 2023 | 91 | 496.5 | 496.53 | 198,612 | **496,530** | 1,241,325 |
| v2_constrained | 2024 | 17 | 69.9 | 69.93 | 27,972 | **69,930** | 174,825 |
| v2_constrained | 2025 | 17 | 110.3 | 110.34 | 44,136 | **110,340** | 275,850 |
| v2_constrained | **2026** | **11** | **29.6** | **29.61** | **11,844** | **29,610** | **74,025** |
| v1_parity (baseline) | 2022 | 114 | 831.0 | 830.97 | 332,388 | **830,970** | 2,077,425 |
| v1_parity | 2023 | 112 | 821.9 | 821.88 | 328,752 | **821,880** | 2,054,700 |
| v1_parity | 2024 | 19 | 89.9 | 89.91 | 35,964 | **89,910** | 224,775 |
| v1_parity | 2025 | 26 | 191.7 | 191.70 | 76,680 | **191,700** | 479,250 |
| v1_parity | 2026 | 11 | 31.7 | 31.68 | 12,672 | **31,680** | 79,200 |

Constraint effect (v1 − v2) at 1,000 trees/ha: 2022 −273,240; 2023 −325,350;
2024 −19,980; 2025 −81,360; 2026 −2,070. Constraint exclusions remove whole
priority zones in Phase 7 (2022: 114 → 75 zones) — the within-zone
`excluded_*` columns are structurally 0 because Phase 7's constrained domain
already excludes water/building/road-surface pixels before zoning, and kept
zones are ~100% residential-landuse, vegetation-cover < 0.30 pixels (the
within-zone causes are gated and verified disjoint regardless).

## 4. Method & provenance

Chain: frozen Phase 5 RF primary (marker-driven) → Phase 6 relative
heat-severity rasters → Phase 7 suitability + priority zones (both
scenarios) → this phase. Inputs hashed in `phase8_manifest.json`; constraint
rasters **reused** from `v2/data/phase7/constraints_rasters/` (water >
buildings > road-surfaces precedence established there — never re-derived).
Area accounting: zone ha = plantable + excluded exactly (0.09 ha/px);
excluded causes disjoint by precedence constraint > landuse > veg. Zone
priorities: Phase 7 ranking thirds. Verification: 73/73 incl. independent
plantable re-derivation, per-zone × per-density arithmetic exactness, zone-ID
identity with Phase 7, and v1 ≥ v2 plantable monotonicity.

## 5. Caveats

- Planning estimates on a relative ranking — not legal availability,
  ownership, field-verified plantability, or survival probability.
- Neutral landuse-50 background remains a weakly identified uncertainty.
- Densities are planning scenarios; optimal density deferred to post-Phase-9
  cooling-prediction evaluation.
- Per-year normalization makes years snapshots, not a trend; 2026-first
  framing is a planning choice, not a heat claim.
- Figures are not produced in V2 (phase6/7/8 convention); all numbers above
  reproduce from `v2/data/phase8/` tables.
