# V2 Phase 7 — Tree Plantation Suitability Report (2026-only recommendation module)

**Framing:** a **relative plantation-suitability decision-support ranking** for
the 2026 W4 snapshot — not physical UHI intensity, not planting-success
predictions, not a statement of legal availability, ownership, or field
feasibility. Single-year 2026 product; no annual-comparison claims. Heat need
stays separate from feasibility throughout. Verification gate **8/8 PASS**.

## Fixed pooled references (documented exact values)

Pooled 2022–2026 valid W4 pixels (200k samples/year/variable): LST p1/p99 =
**34.359 / 55.228 °C**; NDVI 0.021 / 0.724; NDBI −0.273 / 0.160;
vegetation_cover 0.000 / 0.898. All scores clip to these fixed bounds — never
per-year min-max.

## Full-area components (2026 continuous layers, unchanged)

1. **Cooling need (0–1)** = 0.55·n(LST) + 0.25·n(1−NDVI) + 0.20·n(NDBI),
   LST-led, Phase-6-independent.
2. **Feasibility (hard mask)** = eligible landuse (frozen rule: excl.
   industrial 5 / retail 7; nodata neutral-eligible) ∧ ~water ∧ ~buildings ∧
   ~road-surfaces; per-reason exclusion accounting partitions the domain.
   vegetation_cover < 0.30 is the documented planting-space criterion.
3. **Suitability (0–1)** = cooling_need × [0.30·LU + 0.25·(1−n(NDBI)) +
   0.15·road + 0.15·green + 0.15·(1−n(NDVI))] (V1 weights, gated product).

## Candidate planting sites (2026)

Per scenario (v2_constrained PRIMARY; v1_parity also emitted):
**sites = feasible ∧ S2 gates ∧ veg<0.30**, where the S2 gates are
**priority ≥ pooled-p90 = 0.5926 ∧ cooling_need ≥ pooled-p75 = 0.7116** —
pinned constants, DOCUMENTED OPERATIONAL SELECTION CHOICES (not validated
ecological cutoffs). Zone-form: 8-connected, **MMU ≥ 2 ha**, **max-size cap
57 ha** (patch-size p99 evidence; kills mega-patches — 26 of 286 candidates
are oversized: flagged and excluded from shortlist ranking). Result:
**286 candidate sites** (2026 v2). Geometry = exact pixel unions with a 15 m
display-only simplify. Multi-year persistence machinery (h/K/MIN-veg/per-year
columns) was removed — this is a 2026-only module.

## 0–100 planning-priority score (each indicator used exactly once)

**score = 0.50·heat_need + 0.25·vegetation_deficit + 0.25·planting_opportunity**
(0–100 pooled-normalized components; a former persistence term was dropped and
the weights renormalized — documented). Class labels are rank bands
(Shortlist/Candidate). Every artifact carries: **"planning-priority score
(0–100), not a validated probability or proof of optimal cooling."**

## Per-site attributes

usable_area_ha (== plantable_ha, the Phase 9 contract column), score + the
three component scores, a rationale string ("score=… = 0.50·need(…) +
0.25·veg_deficit(…) + 0.25·opportunity(…); 2026 conditions; top-100
shortlist"), landuse composition + untagged_share, confidence status
(low/mixed/identified by untagged share >0.5/>0.2) — **untagged OSM land is
not confirmed available planting space** — constraint flags, and
reference_trees at 400/1000/2500 = floor(usable × density), exact. Zero-usable
sites are excluded. Shortlist confidence mix (2026): 85 low / 2 mixed /
13 identified — the uncertainty flag is load-bearing and never reorders ranks.

## Verification & provenance

Phase 7 gate **8/8 PASS** (pooled references recompute, fixed-threshold class
recompute, class shares, exclusion partition, artifacts). Lineage: Phase 3
products + Phase 2 constraint layers (hashed in the manifest). Logs
`_phase7_build.log` / `_phase7_verify.log`.
