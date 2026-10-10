# V2 Phase 8 — Tree Requirement Report (2026-only plan builder)

**Framing:** planning estimates on the Phase 7 2026 relative
plantation-suitability decision-support ranking — not physical UHI intensity,
not planting-success predictions, no ownership/permission/field-feasibility/
survival claims, **no citywide planting target**. Tree density is a
**planning assumption, not optimal and not AI-predicted**. Single-year 2026
product. Verification gate **5/5 PASS**; Phase 7 gate 8/8 PASS; full suite
112/112 (incl. the zone-table JSON-serializability regression test).

## Usable area and tree arithmetic

Sites are the Phase 7 2026 candidates (S2 gates on feasible land, MMU ≥ 2 ha,
57 ha max-size cap). **usable_area_ha == plantable_ha** (the Phase 9 contract
column; zero-usable sites excluded). Reference tree counts =
**floor(usable_area_ha × density)**, exact — the same arithmetic the frontend
performs client-side at any slider density.

## The 2026 recommendation (v2_constrained PRIMARY)

- **286 candidate sites** (20+6 = 26 oversized flagged, excluded from ranking)
- **Shortlist = top-100 non-oversized sites by the 0–100 score = 100 sites /
  1,046.1 usable ha** — inside the explicit 800–1,200 ha planning band.
- **Shortlist-cut evidence** (cumulative usable-area vs rank): the band is
  entered at rank 91 (945 ha); top-100 = 1,046 ha with min score **72.97**;
  scores descend smoothly across the top 100 (≈74.8 → 73.0) with no score
  cliff — the cut crosses no discontinuity into clearly-lower-ranked land.
- **Trees (floor-exact): 418,428 @400/ha · 1,046,069 @1000/ha · 2,615,175
  @2500/ha.**

Ranking score = 0.50·heat_need + 0.25·vegetation_deficit +
0.25·planting_opportunity (each indicator once; fixed pooled references;
labeled a planning score, not a validated probability or proof of optimal
cooling). Priority classes are rank bands (Shortlist/Candidate). Top-3
shortlist sites: score 74.6/74.5/74.4 with component triples (need 80.1 /
deficit 96.3 / opportunity 49.4), (80.7 / 89.8 / 51.6), (86.7 / 91.7 / 37.4).

## Frontend plan builder (2026 shortlist only)

- **Density slider 100–2500** (default 400, quick picks 400/1000/2500),
  labeled a planning assumption; per-site count = floor(usable × density),
  live-updating; plan total = Σ selected counts (floor-exact).
- **Selection + optional capacity limit**: per-site select/deselect;
  "auto-select shortlist in priority order until the next site would exceed
  the limit"; clear separation of all candidates (behind the transparency
  toggle) vs selected sites vs plan totals (plan ha + plan trees at current
  density). Nothing silently included.
- Map + ranked table + clickable details (usable area, component explanation,
  rationale string, confidence badge, per-site tree estimate); priority pie;
  per-zone click highlight. No year selector, no annual columns, no capacity
  views — a single-year 2026 module.

## Phase 9 contract (unchanged; Phase 9 code untouched)

`tree_requirement_by_zone_v2_constrained_2026.csv` (`zone_id`, `plantable_ha`,
additive attribute columns), `available_planting_space_v2_constrained_2026.tif`
(uint8, nodata 255, authoritative grid), and the phase7-path
`priority_zone_ids_v2_constrained_2026.tif` mirror — all present and
bit-match gate-verified after every build.

## Verification

Phase 8 gate **5/5 PASS**: Phase 9 contract + mirror bit-match; 2026 site-rule
recompute (feasible ∧ S2 gates ∧ veg<0.30, pinned constants); score recompute
+ top-100 shortlist rule on sampled sites; usable-area exactness + floor
arithmetic + certainty rule; artifacts complete. Logs `_phase8_build.log` /
`_phase8_verify.log`; inputs hashed in `phase8_manifest.json`.

## Limitations

2026-only product — no inter-annual claims. Majority of shortlist area sits on
untagged OSM land (85/100 sites confidence low) — flagged, never silently
trusted. Density scenarios are planning assumptions; the supported optimal
density awaits post-Phase-9 cooling-prediction evaluation.
