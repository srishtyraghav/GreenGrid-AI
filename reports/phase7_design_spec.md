# Phase 7 — Stage 0 Design Specification (Frozen Decisions)

**Project:** GreenGrid-AI — Tree Plantation Suitability Analysis
**Date:** 2026-09-04
**Status:** FROZEN for Stage 1 implementation. Stage 1 implements these decisions **exactly**; any deviation requires an explicit written justification in the Phase 7 report.
**Input basis:** every fact cross-checked against disk — see `reports/phase7_input_audit.md`.

**Framing (inherited from Phase 6):** all outputs are a *relative* tree-plantation suitability proxy built on a *relative* heat-severity proxy. 2022 and 2026 are two snapshot composites, not a trend. Suitability classes are per-year normalized (class-relative cross-year comparison only). Nothing in this document claims planting success probability.

---

## a. Conceptual model

```
Heat Need  (0–100) ─┐
                    ├──►  Final = Need × Opportunity / 100   (multiplicative, gated)
Opportunity (0–100) ┘
```

- **Gated multiplicative formulation is the primary output.** Extreme heat **cannot** produce high suitability where opportunity is absent: if `Opportunity = 0`, `Final = 0` regardless of Need. Both factors are 0–100, so `Final ∈ [0, 100]`.
- An **additive baseline** `Final_add = 0.5×Need + 0.5×Opportunity` is computed **as a diagnostic comparison only** (reported side-by-side, never as the primary map). Rationale: additive forms let a Very-High-Need / Zero-Opportunity cell score 50, which the concept model explicitly rejects.

## b. Heat Need (0–100) — baseline

```
Need = 0.40·norm(severity_score)
     + 0.25·norm(1 − NDVI)
     + 0.20·norm(NDBI)
     + 0.15·norm(LST)
```

- `severity_score` (Phase 6, expected ordinal score 0–3) is the primary heat-need term.
- **`vegetation_cover` is EXCLUDED from the baseline** — Phase 5 froze NDVI–vegetation_cover redundancy at r ≈ 0.998–0.9995. It appears only in Scenario D (§h).
- **LST weight is deliberately reduced to 0.15** (and severity_score capped at 0.40): severity classes/scores are themselves trained on per-year LST quartiles, so severity_score and LST are strongly dependent. The 0.40/0.15 split limits this double-counting while keeping both signals. Documented, not tuned.
- `norm` = per-year robust min-max, §f.

## c. Plantation Opportunity (0–100) — baseline

```
Opportunity = 0.30·landuse_eligibility
            + 0.25·(1 − norm(NDBI))            # built-up intensity, inverted
            + 0.15·road_accessibility_band
            + 0.15·green_proximity
            + 0.15·(1 − norm(NDVI))            # planting headroom, inverted NDVI
```

- **Road accessibility is NON-monotonic** (operational bands, §9 Option A — documented, not site-calibrated):
  - `d < 50 m` (road corridor): poor — score ramps linearly **0 → 40** as d goes 0 → 50 m.
  - `50–500 m`: optimal — **100**.
  - `500–2,000 m`: linear decline **100 → 30**.
  - `> 2,000 m`: floor at **30**.
  - Implemented as a piecewise-linear function of `dist_road_m` (static raster, same both years).
- **Green proximity** = `100 · max(0, 1 − dist_vegetation_m / 500)` — full score within/alongside existing vegetation, linearly decaying to 0 at 500 m, 0 beyond (500 m cap). Operational, documented.
- **`dist_building_m` is NOT used** in Opportunity: buildings are an OSM *sample*, not full coverage (audit §2); a partial distance surface would bias feasibility near sampled buildings. Kept as a diagnostic table only.

## d. Land-use eligibility rules matrix (FROZEN)

Scores are 0–100 eligibility contributions; classes 1/3 absent or near-absent on disk (audit §2/§5) but rules defined for completeness. UNCERTAIN = low confidence in the mapping, flagged for the report, not guessed silently.

| Class | Meaning | Eligibility score | Rationale |
|---|---|---|---|
| 0 | unclassified_background | **50 (CONDITIONAL, UNCERTAIN)** | 84.0% of the full grid; absence of an OSM land-use tag, **not** a real land-use observation. Neutral score; large Phase 7 "background" suitability must be explicitly discussed in the report. |
| 1 | park | **40 (CONDITIONAL, UNCERTAIN)** | Already vegetated public land; planting headroom low (maintenance/infilling only). Only 4 px on disk — rule never meaningfully fires. |
| 2 | forest | **30 (CONDITIONAL)** | Already forested; headroom low (understorey/enrichment only). |
| 3 | grass | **50 (CONDITIONAL, UNCERTAIN)** | Open green land; possible but competes with existing open-space function. Absent from raster. |
| 4 | commercial | **35 (CONDITIONAL)** | Paved, high-traffic; limited verge/pocket planting. |
| 5 | industrial | **20 (DISCOURAGED)** | Contamination/operational constraints; low priority. |
| 6 | residential | **90 (ELIGIBLE)** | Yards, street trees, neighbourhood greening — primary planting target. Largest mapped class (255,946 px). |
| 7 | retail | **30 (DISCOURAGED)** | Dense impervious retail cores. |
| 8 | farmland | **60 (CONDITIONAL)** | Planting possible (agroforestry, bunds) but competes with crop land. |

## e. Hard exclusions — only what the data supports

The **only** hard exclusion is the **invalid mask** (`valid_mask_30m.tif`, nodata=0): NoData cells never participate and are never interpreted as "unsuitable" — they are unclassified.

Explicitly documented limitations (audit §4 found **no** supporting data):

- **No water exclusion** — no NDWI raster, no water landuse class, no water OSM polygons (0 tags found). Water is only indirectly visible (strongly negative NDVI, cool Yamuna corridor, masked LST edges); none of these is a validated exclusion layer.
- **No building-footprint exclusion** — buildings are an OSM sample, no footprint raster exists.
- **No road-surface exclusion** — only road *distance* exists; road surface pixels cannot be masked.

These are **stated limitations, not silent omissions**, and must appear in the Phase 7 report's limitations section.

## f. Normalization (FROZEN)

- Per-year robust min-max: clip each input at its **1st–99th percentile** (computed on valid-mask px for that year), then scale to 0–100.
- Per-year normalization means all scores/class boundaries are **class-relative across years** — consistent with the Phase 6 framing (snapshots, not trends); this is restated on every output.
- Every bound (p1, p99, per variable per year) is stored in **`data/processed/phase7/tables/normalization_parameters.csv`** and reproduced in the report.

## g. Suitability classes (FROZEN, operational thresholds)

| Final score | Class |
|---|---|
| 0–20 | 0 = Very Low |
| >20–40 | 1 = Low |
| >40–60 | 2 = Medium |
| >60–80 | 3 = High |
| >80–100 | 4 = Very High |

## h. Sensitivity scenarios (FROZEN weight vectors)

Scenario semantics: A–C vary the **relative emphasis** of Need vs Opportunity around the same gated concept; D varies the **internal composition**. The brief's "(60/30/10)" / "(30/50/20)" triples are realized as exponent emphasis `(α_need, α_opp, remainder)` in a weighted geometric mean that is exactly the gated product when α_need = α_opp = 1:

```
Final_s = 100^(1 − α_need − α_opp) · Need^(α_need) · Opportunity^(α_opp)
```

(this stays in [0,100]: at Need=Opp=100, Final=100. α_need+α_opp ≤ 1 by construction.)

| Scenario | Name | Need composition (sev_score, 1−NDVI, NDBI, LST) | Opportunity composition (landuse, 1−NDBI, road, greenprox, headroom) | Gating exponents (α_need, α_opp) |
|---|---|---|---|---|
| **A** | Balanced (baseline) | 0.40 / 0.25 / 0.20 / 0.15 | 0.30 / 0.25 / 0.15 / 0.15 / 0.15 | (1, 1) — pure gated product Need×Opp/100 |
| **B** | Heat-focused | unchanged | unchanged | (0.6, 0.3) → 100^0.1·Need^0.6·Opp^0.3 |
| **C** | Feasibility-focused | unchanged | unchanged | (0.3, 0.5) → 100^0.2·Need^0.3·Opp^0.5 |
| **D** | Vegetation-priority | 0.40 / **1−vegetation_cover 0.25** (NDVI dropped from Need) / 0.20 / 0.15 | 0.30 / 0.15 / 0.15 / 0.15 / **headroom 0.25** (headroom on 1−vegetation_cover, built-up downweighted) | (1, 1) |

Note: because NDVI–vegetation_cover r ≈ 0.999, Scenario D is expected to approximate A; that agreement is itself a useful redundancy check and will be reported.

## i. Priority zones (FROZEN)

- Connected components, **8-connectivity**, computed per year on `class ≥ 3 (High)` and separately on `class == 4 (Very High)`.
- Minimum cluster size **10 px (~0.8 ha)** — same operational noise rule as Phase 6 hotspots.
- Areas computed in **EPSG:32643** (UTM 43N); degrees are never used for area.

## j. Priority confidence / stability (FROZEN)

```
priority_confidence = 100 · mean_phase6_confidence(0–1) · scenario_agreement_rate
scenario_agreement_rate = fraction of scenarios A–D assigning class ≥ 3 (High)
```

Output 0–100. **Documented meaning:** an operational stability index combining model-confidence (Phase 6, itself an uncalibrated proxy) with cross-scenario robustness. It is **NOT** a probability that trees will grow or that the priority is "correct".

## k. Temporal persistence (FROZEN)

Computed at **paired valid cells** (the 1,500,777 shared valid-mask px), on Phase 7 classes:

| 2022 class | 2026 class | Persistence label |
|---|---|---|
| ≥ 3 (High/Very High) | ≥ 3 | **Persistent** |
| < 3 | ≥ 3 | **Emerging** |
| ≥ 3 | < 3 | **Declining** |
| < 3 | < 3 | **Stable Low** |

Residual combinations are impossible by construction (the table is exhaustive); the label set is fixed. Per-year-normalized classes mean persistence is class-relative, restated on the output.

## l. Feature-dependency table (§6)

| Variable | Role | Correlated variables | Treatment |
|---|---|---|---|
| severity_score | Need (0.40) | LST (trained on LST quartiles); NDVI/NDBI indirectly | Primary term; LST weight capped at 0.15 to limit double-count |
| LST | Need (0.15) | severity_score | Reduced weight; intersect valid mask first (coverage > valid mask) |
| NDVI | Need (0.25 as 1−NDVI); Opportunity (0.15 headroom) | vegetation_cover r≈0.999 | vegetation_cover excluded from baseline; double use across Need/Opportunity accepted & documented (different constructs: heat stress vs planting headroom) |
| vegetation_cover | Scenario D only | NDVI | Baseline exclusion is a frozen Phase 5 redundancy decision |
| NDBI | Need (0.20); Opportunity (0.25 as 1−NDBI) | — | Double use accepted & documented (heat-trapping vs built-up intensity) |
| dist_road_m | Opportunity (0.15, non-monotonic band) | — | Static; same both years |
| dist_vegetation_m | Opportunity (0.15, green proximity, 500 m cap) | NDVI | Static; proximity ≠ competition, documented |
| dist_building_m | None (diagnostic only) | dist_roads | Excluded: OSM sample, not full coverage |
| landuse raster | Opportunity (0.30 eligibility) | NDBI (association) | Static rules matrix (§d); class 0 = 84% of grid treated as neutral, flagged in report |
| valid mask | Constraint (only hard exclusion) | — | All stats restricted to valid px; NoData never interpreted |
| Phase 6 confidence / uncertainty_zone | Diagnostic (§j) | severity | Uncalibrated proxy, labelled as such |
| probability_low…severe | Diagnostic | severity_score | Not used in baseline (already aggregated into severity_score) |

## m. Leakage / circularity notes (§34)

- Phase 7 consumes **only** Phase 6 outputs + raw environmental rasters (audit table). **No Phase 7 output feeds back as an input** to itself, to Phase 6, or to any earlier phase.
- All weights are **predefined in this document** — they are not tuned, fitted, or selected against any evaluation target, ground truth, or desired outcome map.
- Phase 6 severity classes are per-year LST quartiles and Phase 5 froze OOF accuracy ≈ 0.46 / locked holdout ≈ 0.40; Phase 7 inherits this uncertainty and states it wherever severity-derived terms drive the result.
- Suitability is a decision-support ranking, not a causal or biological model; no claim of planting survival is made.

## Assumptions & deviations recorded

1. **Scenario B/C realization** (§h): the brief's "(60/30/10)" / "(30/50/20)" triples were not mathematically specified; realized as weighted-geometric-mean exponents `(α_need, α_opp, remainder)` so all scenarios stay in [0,100] and collapse exactly to the gated product for A. Any other reading would break the 0–100 range or the gating property.
2. Class 0 (84% of grid) receives a neutral 50 rather than exclusion — it is absence of an OSM tag, not evidence of ineligibility (§d); flagged UNCERTAIN.
3. `dist_building_m` deliberately unused (OSM sample coverage) — deviation from any "distance-to-building constraint" intuition, justified in §c/§l.
4. No exclusions beyond the valid mask are implementable (§e); recorded as limitations.
