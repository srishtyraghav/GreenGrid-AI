# GreenGrid AI — Phase 7: Tree Plantation Suitability

**Project:** GreenGrid AI  
**Phase:** 7 — Tree Plantation Suitability & Priority Zones  
**Study Area:** National Capital Territory (NCT) of Delhi, India  
**Report Date:** 2026-09-05  
**Status:** ✅ COMPLETED — Stages 0–5 executed successfully | Validation 137 PASS / 0 FAIL / 0 WARN (reproducibility run)

---

## 1. Executive Summary

Phase 7 converts the frozen Phase 5 model and Phase 6 heat-severity maps into a **GIS multi-criteria decision-support product**: a 0–100 **potential tree-plantation suitability** score per pixel, a five-class suitability map, and delineated **priority zones** for July 2022 and July 2026 over the full valid grid (1,500,777 px/yr, 118,119 ha).

Suitability is computed as a **gated product** of two 0–100 factors — Heat Need (severity_score, NDVI, NDBI, LST) and Plantation Opportunity (land-use eligibility, built-up intensity, road accessibility, green proximity, planting headroom): `Final = Need × Opportunity / 100`. Neither factor alone can produce a high score.

**Headline result (baseline Scenario A, gated):**

| Quantity | 2022 | 2026 |
|---|---|---|
| Mean suitability (0–100) | 25.28 | 24.88 |
| High + Very High area | **45.8 ha (0.039% of valid)** | **9.8 ha (0.008%)** |
| Priority zones (High, ≥10 px) | **11** | **3** |
| Very High zones | 0 | 0 |

All 2022 zones (11) and all 2026 zones (3) are **residential-dominant**; the largest 2022 zone is 8.97 ha (zone 7, mean suitability 62.0, priority confidence 59.5).

**Headline sensitivity finding.** Priority-zone extent is **highly sensitive to the need/opportunity weighting**, and the gated baseline is **conservative by construction**: the same Need and Opportunity re-gated with Scenario B (heat-focused) exponents yields **45,315 ha** of High+Very High in 2022 and **41,911 ha** in 2026 — roughly a **~1,000× spread (989× in 2022; ~4,300× in 2026)** across weightings that are all defensible. The baseline should be read as a deliberately conservative core priority set; recommendations are **scenario-conditional** (§22).

Validation: **137 PASS / 0 FAIL / 0 WARN**, including bit-identical reproduction of all tables, exact class-histogram matches, formula recomputation checks, and frozen-phase integrity checks (Phases 5/6 untouched).

---

## 2. Phase 7 Objective

Phase 7 is **decision support for tree-plantation planning**: it ranks every valid pixel by *potential plantation suitability* and delineates candidate priority zones where high heat need and real planting opportunity co-occur.

Phase 7 **is NOT**:

- a **planting authorization** or legal/tenure clearance — no ownership, land-availability, or permission layer exists in the project;
- a **biological or survival model** — nothing estimates whether trees will grow, survive, or mature;
- a **causal model** — no claim that planting at a priority zone will produce a specific cooling effect;
- an **administrative ranking** — ranking units are 8-connected raster zones, not wards or districts (no usable administrative boundaries exist in the project);
- a **calibrated probability product** — scores and the priority-confidence index are operational, uncalibrated rankings.

---

## 3. Relationship to Phase 5 and Phase 6

- **Phases 5 and 6 are frozen.** Phase 7 consumed their outputs read-only; the validator confirms no modification under any frozen path and that the Phase 5 (57/0/0) and Phase 6 (147/0/0) validation reports still hold.
- **Inputs consumed:** Phase 6 rasters (`severity_score`, `lst`, `confidence`, `probability_*`, `uncertainty_zone`, per-year severity), Phase 3/4 aligned rasters (NDVI, NDBI, vegetation_cover, landuse, road/vegetation/building distance, valid mask). Full audit: `reports/phase7_input_audit.md`.
- **The only ML in Phase 7 is Phase 6's heat severity.** The 0.40-weight severity_score term is the sole model-derived input; everything else is GIS arithmetic on observed rasters.
- **Phase 7 itself is GIS-MCDA (multi-criteria decision analysis), not a new ML model — and that is correct.** There is no planting-outcome ground truth anywhere in the project (no records of where trees were planted, survived, or failed), so no suitability model could be trained or honestly validated. Adding an ML model here would manufacture false precision. A transparent, fully specified weighted-combination scheme with documented weights and sensitivity scenarios is the defensible choice for an unsupervised ranking task.

---

## 4. Scientific Framing

All outputs describe **potential plantation suitability** — a *relative* decision-support ranking built on a *relative* heat-severity proxy (itself a per-year LST-quartile ML ranking with an ~46% accuracy ceiling). Suitability classes are per-year normalized; 2022 and 2026 are two snapshot composites, not a trend.

**What this analysis CAN say:**

- "Under the frozen, documented weighting scheme, location X ranks higher in *potential* plantation suitability than location Y, relative to this grid and these two snapshot dates."
- "These zones combine high heat need with high planting opportunity, and they recur (in containment) across all four weighting scenarios."
- "These zones are the most conservative, highest-confidence candidates for *field verification*."

**What this analysis CANNOT say:**

- "Trees will grow or survive here." No survival, growth, or establishment probability is estimated anywhere.
- "This land is available, legal, or authorized for planting." No tenure, ownership, or availability layer exists.
- "Planting here will cool the city by X °C." All temperature relationships reported are associations, not causal effects.
- Anything absolute about cross-year change or physical heat: classes are per-year and class-relative.

---

## 5. Input Data Audit

Every candidate input was verified on disk before implementation (`reports/phase7_input_audit.md`, Stage 0). Key facts:

- All rasters share one grid: **1768 × 1874**, EPSG:4326, identical transform, ~30 m pixels.
- **Valid mask** (`valid_mask_30m.tif`): 1,500,777 valid px (~45.3% of the full 3,313,232-px grid) — the analysis domain for every Phase 7 computation. NoData cells never participate and are never interpreted as "unsuitable".
- **Year-specific inputs** (2022 and 2026 rasters): severity, severity_score, LST, confidence, probability ×4, uncertainty_zone, NDVI, NDBI, vegetation_cover.
- **Static inputs** (one file reused for both years): landuse raster, dist_road_m, dist_vegetation_m, dist_building_m, valid mask.
- **Consequence:** the landuse-eligibility, road-accessibility, and green-proximity terms of Opportunity are **identical in 2022 and 2026**. Year-to-year Opportunity variation comes **only** from the NDBI (built-up intensity) and NDVI (planting headroom) terms. Opportunity change 2022→2026 reflects vegetation/index change on a static land-use/accessibility base — **not observed land-use change**.
- **Coverage warning handled:** NDVI/NDBI/vegetation_cover/LST raw rasters cover more px than the valid mask (e.g. NDVI-2026: 1,914,362 vs 1,500,777). Every raw input is intersected with the valid mask before use.
- Landuse raster (static): class 0 unclassified_background 2,784,500 px (84.0% of full grid); class 3 (grass) entirely absent; class 1 (park) = 4 px.
- No water polygons, no NDWI raster, no building-footprint raster, no road-surface raster exist anywhere in the project (audit §4) — see §10.

---

## 6. Spatial Reference and Resolution

| Property | Value |
|---|---|
| Grid shape | 1768 rows × 1874 cols (3,313,232 px full grid) |
| CRS | EPSG:4326 (WGS 84) |
| Origin / pixel size | (76.83290625, 28.88469914) / 0.00026949° × −0.00026949° |
| Nominal resolution | ~30 m (~26–30 m metric at Delhi's latitude, per Phase 3) |
| Valid pixels | 1,500,777 per year (~45.3% of grid) |
| Valid area | **118,119.088 ha (~1,181 km²)** per year |
| Area CRS | **EPSG:32643** (UTM 43N) — degrees are never used for area |

---

## 7. Heat Need Methodology

Heat Need (0–100) measures *where cooling benefit is most needed* — proxied by heat stress:

```
Need = 0.40·norm(severity_score)      ← Phase 6 ordinal severity score (primary term)
     + 0.25·norm(1 − NDVI)            ← sparse vegetation raises heat stress
     + 0.20·norm(NDBI)                ← built-up surfaces trap heat
     + 0.15·norm(LST)                 ← observed land-surface temperature
```

Design decisions (frozen, documented — not tuned):

- **`vegetation_cover` is excluded from the baseline.** Phase 5 froze NDVI–vegetation_cover redundancy at r ≈ 0.998–0.9995; including both would double-count one signal. It appears only in Scenario D.
- **LST is deliberately downweighted to 0.15** (severity_score capped at 0.40). Phase 6 severity classes/scores are themselves trained on per-year LST quartiles, so severity_score and LST are strongly dependent; the 0.40/0.15 split limits this double-counting while keeping both signals.
- `norm` = per-year robust min-max at p1–p99 (§11). 2022 Need mean/median: 53.30/55.26; 2026: 51.27/53.18.

---

## 8. Plantation Opportunity Methodology

Plantation Opportunity (0–100) measures *where planting is plausibly feasible*:

```
Opportunity = 0.30·landuse_eligibility          ← OSM land-use rules matrix (§9)
            + 0.25·(100 − norm(NDBI))           ← built-up intensity, inverted
            + 0.15·road_accessibility_band      ← non-monotonic (below)
            + 0.15·green_proximity              ← 500 m cap (below)
            + 0.15·(100 − norm(NDVI))           ← planting headroom, inverted NDVI
```

- **Road accessibility is NON-monotonic** (frozen piecewise-linear band, documented, not site-calibrated; `dist_road_m` is static):
  - `d < 50 m` (road corridor): ramps linearly **0 → 40** as d goes 0 → 50 m;
  - `50–500 m`: optimal, score **100**;
  - `500–2,000 m`: linear decline **100 → 30**;
  - `> 2,000 m`: floor at **30** (remote land still scores some opportunity).
- **Green proximity** = `100 · max(0, 1 − dist_vegetation_m / 500)`: full score within/alongside existing vegetation, linear decay to 0 at the **500 m cap**, 0 beyond. Operational: near-source planting (nursery stock, watering, connectivity); it is *not* a competition measure.
- **Planting headroom** = inverted normalized NDVI: sparse-vegetation land has room for trees; dense green land does not.
- **`dist_building_m` is excluded from Opportunity.** Buildings are an OSM *sample*, not full coverage (audit §2); a partial distance surface would systematically bias feasibility near sampled buildings. Kept as a diagnostic table only.

2022 Opportunity mean/median: 48.87/47.66; 2026: 49.34/48.12 — nearly identical across years because three of five components are static (§5).

---

## 9. Land-use Constraints

Eligibility is a **frozen rules matrix** on the static OSM land-use raster (full table: `landuse_suitability_rules.csv`):

| Class | Meaning | Eligibility | Status | Rationale |
|---|---|---|---|---|
| 0 | unclassified_background | **50** | CONDITIONAL (UNCERTAIN) | Absence of an OSM tag, not a land-use observation → neutral score (see below) |
| 1 | park | 40 | CONDITIONAL (UNCERTAIN) | Already vegetated; maintenance/infilling only; 4 px on disk |
| 2 | forest | 30 | CONDITIONAL | Already forested; understorey/enrichment only |
| 3 | grass | 50 | CONDITIONAL (UNCERTAIN) | Possible but competes with open-space function; absent from raster |
| 4 | commercial | 35 | CONDITIONAL | Paved, high-traffic; verge/pocket planting only |
| 5 | industrial | 20 | DISCOURAGED | Contamination/operational constraints |
| 6 | residential | **90** | ELIGIBLE | Yards, street trees, neighbourhood greening — primary target (255,946 px full grid) |
| 7 | retail | 30 | DISCOURAGED | Dense impervious retail cores |
| 8 | farmland | 60 | CONDITIONAL | Agroforestry/bunds possible; competes with crops |

- **Class 0 = neutral 50 (UNCERTAIN).** It is 84.0% of the full grid and its score reflects *absence of a tag*, not evidence of ineligibility — excluding it would silently delete most of the city from the analysis. It is flagged UNCERTAIN (not guessed silently) and its dominance is discussed honestly in §20. Nodata (255) maps to the same neutral 50.
- **Park/forest eligibility reasoning:** both are already vegetated, so planting *headroom* is low; they score below neutral to reflect maintenance/enrichment-only roles rather than new planting.

---

## 10. Hard Exclusions

The **only implementable hard exclusion is the invalid mask** (`valid_mask_30m.tif`, nodata=0): NoData cells never participate and are never interpreted as "unsuitable" — they are unclassified.

The following exclusions are **NOT implementable with current project data** (input audit §4; stated limitations, not silent omissions):

- **No water exclusion** — no NDWI raster, no water land-use class, and 0 water/waterway tags in the raw OSM GeoJSONs. Water is only indirectly visible (strongly negative NDVI, the cool Yamuna LST corridor, masked LST edges); none of these is a validated exclusion layer.
- **No building-footprint exclusion** — buildings are an OSM sample; no footprint raster exists.
- **No road-surface exclusion** — only road *distance* exists; road surface pixels cannot be masked.

---

## 11. Normalization

Per-year **robust min-max**: each input is clipped at its **1st–99th percentile** computed on that year's valid-mask px, then scaled to 0–100. Per-year normalization means all scores and class boundaries are **class-relative across years** (snapshots, not trends) — restated on every output. All bounds are stored in `normalization_parameters.csv` and reproduced here:

| Year | Variable | p1 | p99 |
|---|---|---|---|
| 2022 | severity_score | 0.1125 | 2.8307 |
| 2022 | ndvi | 0.0142 | 0.7544 |
| 2022 | ndbi | −0.3552 | 0.1328 |
| 2022 | lst | 27.663 | 50.210 |
| 2022 | vegetation_cover | 0.000 | 0.9391 |
| 2022 | one_minus_ndvi | 0.2456 | 0.9858 |
| 2022 | one_minus_vegetation_cover | 0.0609 | 1.000 |
| 2026 | severity_score | 0.0827 | 2.7070 |
| 2026 | ndvi | −0.0291 | 0.8156 |
| 2026 | ndbi | −0.3444 | 0.0744 |
| 2026 | lst | 30.163 | 44.058 |
| 2026 | vegetation_cover | 0.000 | 1.000 |
| 2026 | one_minus_ndvi | 0.1844 | 1.0291 |
| 2026 | one_minus_vegetation_cover | 0.000 | 1.000 |

(n_cells = 1,500,777 for every row.)

---

## 12. Weighting

All weights are **predefined in the frozen Stage 0 design spec** (`reports/phase7_design_spec.md`). They are **configurable in `src/suitability/config.py` but were NOT tuned, fitted, or selected against any evaluation target, ground truth, or desired outcome map** — there is no planting-outcome target to tune against (§3). Every weighting choice is documented with its rationale (§7–§9); the validator checks that the manifest weights match the frozen config exactly.

---

## 13. Suitability Formula

**Primary (gated, multiplicative):**

```
Final = Need × Opportunity / 100        (Scenario A / baseline)
```

Both factors are 0–100, so Final ∈ [0, 100]. Gating means extreme heat **cannot** produce high suitability where opportunity is absent: if Opportunity = 0, Final = 0 regardless of Need.

**Diagnostic additive comparison** (computed side-by-side, never the primary map):

```
Final_add = 0.5·Need + 0.5·Opportunity
```

| Form | 2022 mean | 2026 mean |
|---|---|---|
| Gated product (primary) | **25.28** | **24.88** |
| Additive 0.5/0.5 (diagnostic) | **51.09** | **50.31** |

**Why the gated form is preferred.** An additive form lets a Very-High-Need / Zero-Opportunity cell score 50 — ranking land as "moderately suitable" where planting is infeasible — and lets heat alone dominate on low-opportunity land (water margins, dense cores with no room, industrial sites). The concept model explicitly rejects that: need without opportunity is not a planting candidate. The gated product is therefore conservative by construction (§22) — a feature for a shortlist product, at the cost of a compressed score distribution.

---

## 14. Suitability Classes

Frozen **operational** thresholds on Final (0–100); encoded 0–4 (int16, nodata −1):

| Score | Class |
|---|---|
| 0–20 | 0 = Very Low |
| >20–40 | 1 = Low |
| >40–60 | 2 = Medium |
| >60–80 | 3 = High |
| >80–100 | 4 = Very High |

These thresholds are **operational cut-points for shortlisting, not universal biological or agronomic boundaries**; because inputs are per-year normalized, classes are class-relative across years.

---

## 15. Priority Zone Extraction

- Connected components with **8-connectivity** (diagonal neighbours count), computed per year on **class ≥ 3 (High)** and separately on **class == 4 (Very High)** — the same operational rule as Phase 6 hotspots.
- **Minimum cluster size 10 px (~0.8 ha at 30 m)**; smaller clusters are removed as operational noise. This is a pragmatic noise rule, not a physical minimum-patch-size claim.
- Zones are raster-derived analytical units — **not administrative areas**; no administrative ranking exists.

---

## 16. Area Calculation

All areas are **per-pixel geodesic areas**: each pixel's four corners are transformed to **EPSG:32643** (UTM 43N) and the exact polygon area is summed over unit cells (mean valid pixel ≈ 787 m²). Degree coordinates are never used for area.

**Reconciliation:** per-year class areas in `suitability_area_statistics.csv` sum to the valid area 118,119.088 ha within ±0.5% (validator checks PASS); priority-zone polygon areas agree with pixel-count × 787.11 m² within 5% for every year × scenario combination; transition-table counts/areas sum exactly to the paired valid total.

---

## 17. 2022 Results

Valid area 118,119.088 ha. Need mean 53.30, Opportunity mean 48.87, suitability mean 25.28 (median 25.63).

| Class | Pixels | % of valid | Area (ha) |
|---|---|---|---|
| 0 Very Low | 455,227 | 30.33% | 35,841.3 |
| 1 Low | 936,961 | 62.43% | 73,732.6 |
| 2 Medium | 108,007 | 7.20% | 8,499.4 |
| 3 High | 582 | 0.039% | **45.8** |
| 4 Very High | 0 | 0.0% | 0.0 |

**Baseline priority zones: 11 High, 0 Very High** — all residential-dominant. Top 3 by area:

| Zone | Area (ha) | Mean suit. | Priority conf. | Why here (one line) |
|---|---|---|---|---|
| 7 | 8.971 | 62.01 | 59.5 | Residential (elig. 90) with Q5 heat (sev 2.30, LST 46.2 °C), near-bare NDVI 0.036, optimal road band (298 m), close green (103 m), headroom 97/100 |
| 4 | 5.823 | 60.97 | 58.5 | Same signature: residential, Q5 sev 2.32 / Q1 NDVI, optimal roads (304 m), headroom 97/100 |
| 11 | 4.173 | 63.28 | **92.2** | Residential, highest need (sev 2.88, LST 45.7 °C), headroom 92/100 — highest-confidence zone |

Environment of the High class (`environmental_priority_statistics`): mean LST 45.51 °C, NDVI 0.044, NDBI 0.046, vegetation_cover 0.010, severity_score 2.37, Phase 6 confidence 0.632 — the class cleanly separates hot, bare, built land.

---

## 18. 2026 Results

Valid area 118,119.088 ha. Need mean 51.27, Opportunity mean 49.34, suitability mean 24.88 (median 24.60).

| Class | Pixels | % of valid | Area (ha) |
|---|---|---|---|
| 0 Very Low | 522,790 | 34.83% | 41,149.2 |
| 1 Low | 835,572 | 55.68% | 65,760.8 |
| 2 Medium | 142,291 | 9.48% | 11,199.3 |
| 3 High | 124 | 0.008% | **9.8** |
| 4 Very High | 0 | 0.0% | 0.0 |

**Baseline priority zones: 3 High, 0 Very High** — all residential:

| Zone | Area (ha) | Mean suit. | Priority conf. |
|---|---|---|---|
| 2 | 2.835 | 61.64 | **92.8** |
| 1 | 2.834 | 61.83 | 68.8 |
| 3 | 0.866 | 61.13 | 52.6 |

High-class environment 2026: mean LST 38.21 °C, NDVI 0.042, vegetation_cover 0.014, severity_score 2.63, confidence 0.744. The 2026 priority set is a near-subset in character (same residential, near-bare, hot signature) but smaller in extent — a class-relative observation under per-year normalization, not a physical decline (§21).

---

## 19. Green vs Built-up Analysis

Operational groups on observed NDVI/NDBI (`green_built_comparison.csv`): green_dominant = NDVI ≥ 0.3; built_dominant = NDBI ≥ 0.1 and NDVI < 0.3; else other_mixed. These are conventional remote-sensing thresholds, not site-calibrated.

| Year | Group | Pixels | Area (ha) | % valid | Mean suitability | Mean heat need | High+VH % |
|---|---|---|---|---|---|---|---|
| 2022 | green_dominant | 772,625 | 60,817.1 | 51.48% | 18.27 | 38.25 | 0.0% |
| 2022 | built_dominant | 53,541 | 4,213.1 | 3.57% | 31.21 | **83.91** | 0.0% |
| 2022 | other_mixed | 674,611 | 53,088.9 | 44.95% | 32.82 | 68.11 | 0.086% |
| 2026 | green_dominant | 777,356 | 61,191.4 | 51.80% | 18.16 | 36.91 | 0.0% |
| 2026 | built_dominant | 5,532 | 435.5 | 0.37% | 28.15 | 77.20 | 0.0% |
| 2026 | other_mixed | 717,889 | 56,492.2 | 47.83% | 32.13 | 66.62 | 0.017% |

Pattern: vegetated land has low heat need and low suitability (little headroom — correctly gated); the highest *suitability* sits in the mixed band where need is high but some headroom and eligibility remain; pure built cores score high need but low opportunity. **Association, not causation:** these groups are index thresholds and suitability differences describe association, not a causal planting effect.

---

## 20. Land-use Analysis

Per-class suitability (`landuse_suitability_summary.csv`, valid-mask px; the land-use raster is static so pixel counts are identical across years):

| Class | Valid px | Area (ha) | Mean suitability 2022 | 2026 | High area 2022 | 2026 |
|---|---|---|---|---|---|---|
| 0 unclassified_background | 1,223,885 | 96,328.4 | 24.25 | 23.88 | 0.0 | 0.0 |
| 2 forest | 14,590 | 1,149.8 | 13.82 | 15.80 | 0.0 | 0.0 |
| 4 commercial | 11,907 | 937.6 | 23.11 | 25.08 | 0.0 | 0.0 |
| 5 industrial | 47,735 | 3,755.5 | 23.23 | 23.29 | 0.0 | 0.0 |
| 6 residential | 134,612 | 10,593.8 | **37.79** | **36.93** | **45.8** | **9.8** |
| 7 retail | 2,822 | 222.2 | 26.94 | 30.72 | 0.0 | 0.0 |
| 8 farmland | 65,226 | 5,131.9 | 23.16 | 21.73 | 0.0 | 0.0 |

**Class-0 caveat, stated honestly:** 1,223,885 valid px — **81.55% of the valid domain (84.0% of the full grid)** — carry "unclassified_background". Much of the mapped city's background suitability is therefore driven by *absence of an OSM tag*, scored neutral 50, not by a positive land-use observation. This is why eligibility needs field verification (§29) and why the background mean suitability (~24) sits near the gated middle of the distribution. The genuinely informative signal is concentrated in the minority of tagged land — and it is coherent: **residential is the only class containing any High-suitability area** (all 45.8 ha in 2022, all 9.8 ha in 2026), consistent with its ELIGIBLE 90 score. Forest is correctly *low* (13.8–15.8): vegetated, little headroom.

---

## 21. Temporal Priority Analysis

Paired-cell persistence on the 1,500,777 cells valid in both years, baseline Scenario A classes (`temporal_priority_transition.csv`):

| Category | Pixels | Area (ha) | % of valid |
|---|---|---|---|
| Persistent (High/VH both years) | 49 | 3.858 | 0.003% |
| Emerging (Low 2022 → High/VH 2026) | 75 | 5.904 | 0.005% |
| Declining (High/VH 2022 → Low 2026) | 533 | 41.942 | 0.036% |
| Stable Low | 1,500,120 | 118,067.4 | 99.956% |

Scenario B/C persistence is also tabulated (e.g. B: Persistent 26.05%, Emerging 9.43%, Declining 12.32% of valid).

**Framing:** these are **relative snapshots, not a trend.** Classes are per-year normalized, so "Declining" means the 2026 within-year ranking no longer crosses the High threshold — not a measured physical loss of suitability. The mean-LST difference between the composites (Phase 6: 38.76 °C in July 2022 vs 34.94 °C in July 2026) is a **data observation about two images** (atmospheric, cloud-masking, and acquisition differences included), not a climate claim about Delhi's trajectory.

---

## 22. Sensitivity Analysis

Four frozen scenarios (`sensitivity_analysis.csv`; findings memo `data/processed/phase7/reports/sensitivity_findings.md`). Scenarios A–C re-gate the *same* Need and Opportunity via a weighted geometric mean `Final_s = 100^(1−αn−αo) · Need^αn · Opp^αo` (stays in [0,100]; αn+αo ≤ 1; collapses exactly to the gated product when αn = αo = 1); D varies internal composition (vegetation_cover replaces NDVI in Need; headroom upweighted 0.15→0.25, built-up downweighted 0.25→0.15).

| Scenario | Exponents (α_need, α_opp) | 2022 mean | 2022 High+VH (ha) | 2026 mean | 2026 High+VH (ha) |
|---|---|---|---|---|---|
| A Balanced (baseline) | (1, 1) — pure gated product | 25.28 | **45.8** | 24.88 | **9.8** |
| B Heat-focused | (0.6, 0.3) → 100^0.1·Need^0.6·Opp^0.3 | 53.39 | **45,315.0** | 52.39 | **41,910.5** |
| C Feasibility-focused | (0.3, 0.5) → 100^0.2·Need^0.3·Opp^0.5 | 55.93 | 36,987.5 | 55.62 | 38,928.5 |
| D Vegetation-priority | (1, 1), composition swap | 27.69 | 743.3 | 27.61 | 646.2 |

Overlap vs baseline (IoU / containment / top-10 rank stability):

| Scenario-year | IoU vs A | Baseline High+VH contained | Top-10 rank stability |
|---|---|---|---|
| B 2022 | 0.0010 | 100.0% | 1.00 |
| C 2022 | 0.0012 | 100.0% | 1.00 |
| D 2022 | 0.0616 | 100.0% | 1.00 |
| B 2026 | 0.0002 | 100.0% | 1.00 |
| C 2026 | 0.0003 | 100.0% | 1.00 |
| D 2026 | 0.0151 | 100.0% | 1.00 |

**The ~1000× extent finding.** Priority-zone extent is highly weight-sensitive: High+Very-High area spans **45.8 ha → 45,315 ha in 2022 (~989×)** and **9.8 ha → 41,911 ha in 2026 (~4,300×)** across reasonable weightings — a spread of roughly three to four orders of magnitude. The gated baseline is *conservative by construction* (its product compresses scores toward the low end), and 100% of its (very few) priority cells are reproduced by every relaxed scenario — but the near-zero IoUs and the rank-stability caveat (tiny baseline zones fall *inside* scenario zones up to ~4,300× larger; that is containment, not shape/rank preservation at comparable scales) mean the headline extent must not be read as a stable physical quantity.

**Honest statement:** because priority-zone extent is highly weight-sensitive, all recommendations in this report are **scenario-conditional**. The baseline is a deliberately conservative core priority set; scenarios B/C bracket how the priority map broadens when the gating is relaxed; Scenario D's near-agreement with A (IoU 0.06, similar means) confirms the NDVI–vegetation_cover redundancy decision.

---

## 23. Uncertainty

**Priority confidence** (per zone, 0–100):

```
priority_confidence = 100 · mean_phase6_confidence(0–1) · scenario_agreement_rate
scenario_agreement_rate = fraction of scenarios A–D assigning class ≥ 3 (High)
```

**Meaning:** an operational *stability index* combining model confidence (Phase 6, itself an uncalibrated proxy) with cross-scenario robustness. **It is explicitly NOT a probability that trees will grow or that the priority is "correct."** Baseline (Scenario A) zone confidences span 51.5–92.2 (2022) and 52.6–92.8 (2026); the lowest value anywhere in the analysis, 19.9, belongs to Scenario C zone 417, not to a baseline zone.

**Inherited model uncertainty.** The heat-need layer rests on the Phase 5/6 model, whose honest accuracy ceiling is **out-of-fold accuracy ≈ 0.46** (pooled OOF 0.4619; RF-C adjacent-block CV 46.99%; locked holdout 39.9% on unseen geography), with 61.1% (2022) / 40.4% (2026) of cells in Phase 6 uncertainty zones. Every severity-driven term inherits this ceiling; a zone with confidence 60 is not "60% likely to be right."

---

## 24. "Why Here?" Examples

Three worked examples (full detail: `why_here_examples.md` and `why_here_explanations.csv`; driver levels are quintiles of that year's valid-cell distribution). Each describes a *relative* ranking on a *relative* heat proxy — no survival claim.

**Zone A-2022-7 (High, 62.01, baseline rank 1 by area, 8.97 ha).** Heat need: severity 2.30 (Q5), LST 46.2 °C (Q5), NDVI 0.036 (Q1 — raises heat need *and* planting headroom), NDBI 0.056 (Q5). Opportunity: residential (eligibility 90), built-up Q5, road band optimal 50–500 m (mean 298 m), green proximity close (mean 103 m, score 79/100), headroom 97.0/100 (Q5). Constraints: none implementable beyond the invalid mask (§10). Confidence 59.5/100. → High Priority — candidate for phased planting assessment and field verification.

**Zone B-2022-214 (High, 67.14, Scenario B rank 1, 27,259.7 ha).** The dominant Scenario-B zone: need severity 2.10 (Q4), LST 43.5 °C (Q4), NDVI 0.156 (Q2), NDBI 0.020 (Q4); opportunity on *unclassified background* (eligibility 50 — the absence-of-tag caveat applies in full), road band declining 500–2,000 m (mean 1,664 m), green proximity beyond the 500 m cap (score 0), headroom 80.8/100 (Q4). Confidence 22.9/100 — low, reflecting background-landuse uncertainty and low scenario agreement. → shortlist only with field verification.

**Zone C-2022-417 (High, 66.94, Scenario C rank 1).** Need: severity 1.67 (Q3), LST 40.8 °C (Q4), NDVI 0.174 (Q2), NDBI −0.016 (Q4); opportunity: unclassified background (50), road declining (mean 923 m), green proximity distant 250–500 m (mean 337 m, score 33/100), headroom 78.3/100 (Q4). Confidence 19.9/100. → shortlist only with field verification.

The contrast is instructive: the baseline's top zone is verified-eligible residential land with strong access; the relaxed scenarios' top zones are vast background-land blocks whose opportunity rests on the neutral-50 assumption — exactly why the baseline is the recommended shortlist.

---

## 25. Validation

`data/processed/phase7/reports/phase7_validation_report.json` (reproducibility run): **137 PASS / 0 FAIL / 0 WARN**.

| Category | Checks |
|---|---|
| inputs | 8 |
| alignment | 15 |
| scores | 25 |
| classes | 10 |
| zones | 52 |
| temporal | 6 |
| sensitivity | 6 |
| provenance | 6 |
| leakage | 6 |
| reproducibility | 3 |

Coverage includes: output-suite completeness (15 rasters, 11 GeoJSONs, 13 canonical tables); grid/CRS/transform/nodata conformance of every raster; frozen-phase integrity (Phase 5 57/0/0, Phase 6 147/0/0, git-clean frozen paths); score-range and weight-sum checks; exact class-histogram matches; zone geometry validity, unique IDs, polygon-vs-pixel area agreement (±5%); transition tables summing to paired totals; sensitivity-findings presence; provenance and leakage guards (no module reads its own outputs as inputs; pipeline reads of Phase 7 rasters confined to read-back verification); and the reproducibility checks of §26.

---

## 26. Reproducibility

- Random seed: 42 (all stochastic components seeded).
- Stage commands (from the project root, venv active):
  ```bash
  PYTHONPATH=src .venv/bin/python -m suitability.pipeline
  PYTHONPATH=src .venv/bin/python -m suitability.zones
  PYTHONPATH=src .venv/bin/python -m suitability.figures
  PYTHONPATH=src .venv/bin/python -m suitability.validate_suitability --repro
  ```
- `--repro` re-runs the zones stage and confirms **bit-identical tables**;
- **class-histogram match:** the 2022 and 2026 class rasters' histograms match `suitability_summary.csv` exactly;
- **formula recomputation:** independent recomputation `Need × Opportunity / 100` on 50,000 sampled cells per year matches the Scenario A raster within float tolerance (2022 and 2026);
- **manifest integrity:** input hashes recompute bit-identically; manifest grid spec, weights, class thresholds, and scenario exponents all equal the frozen config;
- software stack identical to Phase 6 (recorded in `phase7_pipeline_record.json`).

---

## 27. Limitations

1. **Relative ranking on a relative proxy.** Suitability is a per-year-normalized decision-support ranking built on Phase 6 relative heat severity — not a physical plantability measure, not calibrated to any planting outcome.
2. **Inherited ML accuracy ceiling.** The heat-need layer rests on a model with OOF accuracy ≈ 0.46 (locked holdout ~0.40 on unseen geography); 61.1%/40.4% of cells lie in Phase 6 uncertainty zones. All severity-driven terms inherit this.
3. **No survival or growth claim.** Nothing estimates establishment, survival, growth rate, or cooling realized; priority_confidence is a stability index, not a probability (§23).
4. **No hard exclusions beyond the invalid mask.** No water, building-footprint, or road-surface layers exist in the project (§10); priority zones may overlap water bodies, structures, or road surfaces and must be screened in the field.
5. **Dominant unclassified background.** 84.0% of the full grid (81.55% of valid px) has no OSM land-use tag and receives a neutral, uncertain 50; background suitability partly reflects absence-of-tag, and classes 1 (park, 4 px) and 3 (grass, 0 px) are effectively absent.
6. **Static OSM context.** Land use, roads, vegetation distance, and buildings are a single snapshot reused for both years; no infrastructure or land-use change analysis is possible.
7. **Two snapshots only.** 2022 and 2026 composites; no trend, seasonality, or interannual variability; mean-LST differences are two-image observations.
8. **Class-relative cross-year comparability.** Per-year p1–p99 normalization and per-year class thresholds mean a 2026 High pixel is not directly comparable in absolute terms to a 2022 High pixel.
9. **Documented, not calibrated, weights.** Weights and the road/green-proximity curves are frozen expert-design choices; they were not tuned (no target exists) but they are also not site-calibrated — and extent is highly sensitive to them (§22).
10. **Building sample bias.** Building distance comes from an OSM Central Delhi sample; it is excluded from scoring for this reason, but it also means no building-proximity feasibility term exists at all.
11. **NDVI/PVC redundancy.** vegetation_cover is a near-duplicate of NDVI (r ≈ 0.998–0.9995); Scenario D's agreement with A corroborates this rather than adding independent evidence.
12. **Spatial autocorrelation.** 30 m pixels are strongly autocorrelated; 1.5 M px are not 1.5 M independent observations, and Phase 5 found strongly positive error-field Moran's I — apparent zone structure may partly reflect structured model error.
13. **Operational thresholds.** Class edges (20/40/60/80), the 10-px (~0.8 ha) zone rule, the road bands, the 500 m green cap, and the NDVI/NDBI green/built cut-offs are conventional operational choices, not site-calibrated science.
14. **No administrative units.** Zones are raster-derived; no ward/district-level reporting is possible with the data in this project, and no field-verification inputs (tenure, soil, water, permissions) exist in the analysis.

---

## 28. Scientific Interpretation

The Phase 7 patterns are internally coherent and physically plausible:

- **The gated concept behaves as designed.** High suitability requires both high heat need and high opportunity; bare, hot residential land qualifies (45.8 ha in 2022), while equally hot industrial land (eligibility 20) and cool forest land (no need, no headroom) do not. The exclusive concentration of High class in residential land (§20) is exactly what the rules matrix predicts.
- **Heat and opportunity are geographically distinct problems.** Mean Need is 53.3 vs mean Opportunity 48.9 (2022) — but their *co-occurrence* is rare; the product compresses the mean to 25.3 and only 0.039% of the domain crosses the High threshold. High-need/low-opportunity land (dense cores, water margins) and low-need/high-opportunity land (existing green) dominate the map; the planning-relevant band is the intersection.
- **The 2022→2026 contrast is class-relative.** The 2026 High set is smaller (9.8 vs 45.8 ha) with a cooler LST distribution (High-class mean 38.2 vs 45.5 °C), consistent with per-year re-ranking under a cooler composite — not evidence of a physical suitability decline.
- **Sensitivity is the honest headline.** A ~1,000× extent spread across defensible weightings (§22) means the *location of the core zones* (contained in all scenarios, residential, hot, bare) is more robust than any *extent figure*; the product is best read as a conservative shortlist plus bracketing scenarios.

---

## 29. Practical Use

**How a planner should read this:** the baseline map and its 11 (2022) / 3 (2026) priority zones are a **candidate shortlist for field verification** — the most conservative set of places where heat need and planting opportunity co-occur under the frozen scheme. They are **NOT planting authorization**; nothing here establishes that any pixel is available, legal, or plantable.

**Required verification before any planting decision** (none of these data exist in the project):

1. **Land tenure, ownership, and legal availability** per candidate site;
2. **Field surface inspection** — actual ground cover (the 81.55% unclassified background), encumbrances, utilities, underground infrastructure;
3. **Water availability and irrigation access** (no hydrological input exists);
4. **Soil quality and depth** (no soil layer exists);
5. **Administrative permissions** and municipal/stakeholder coordination (no administrative boundaries exist in-project);
6. **Species selection, planting stock, and maintenance commitment** — the biological questions this analysis explicitly does not answer;
7. **Ground-truthing of OSM land-use tags** and acceptance of the static-context assumption (§27.6).

Use the scenario maps as brackets: if the planning question is heat-focused, Scenario B broadens the candidate set to ~45,315 ha (2022); if feasibility-led, Scenario C gives ~36,988 ha — both at much lower per-zone confidence than the baseline core.

---

## 30. Future Improvements

1. **Water and exclusion layers:** NDWI/water-index raster, building-footprint raster, road-surface mask — converting §10's stated limitations into real exclusions.
2. **More dates:** a time series of seasonal composites to replace two snapshots and support genuine trend analysis.
3. **Field data:** planting-outcome records (plantings, survival) to enable calibration or validation of suitability weights, and ground-truth surveys of land use and availability.
4. **Administrative boundaries:** usable ward/district layers so priority can be reported for real planning units.
5. **Site factors:** soil, groundwater, and hydrology layers; species-specific suitability.
6. **Participatory weighting:** structured expert/stakeholder weight elicitation (e.g. AHP) to replace documented-not-calibrated weights, with the sensitivity machinery retained.
7. **Uncertainty propagation:** full Monte-Carlo propagation of Phase 6 posterior uncertainty through the gated product instead of the two-factor stability index.
8. **Alternative MCDA formulations and targets:** compare the gated product against outranking/fuzzy methods once any outcome data exist.

---

## 31. Conclusion

**What Phase 7 demonstrates:**

- A complete, fully specified **GIS-MCDA suitability pipeline** (Heat Need × Plantation Opportunity, gated) producing 0–100 suitability rasters, five-class maps, and delineated priority zones for July 2022 and July 2026 over the full valid grid (1,500,777 px/yr; 118,119 ha).
- A **deliberately conservative baseline shortlist**: 11 High zones / 45.8 ha (2022) and 3 High zones / 9.8 ha (2026), all residential-dominant, each with a worked "why here?" explanation and a stability-index confidence.
- **Honest sensitivity quantification**: priority-zone extent spans ~989× (2022) across defensible weightings; 100% of baseline priority cells are contained in all relaxed scenarios, but extents are scenario-conditional.
- **Full reproducibility and validation**: 137 PASS / 0 FAIL / 0 WARN, bit-identical tables, exact class-histogram and formula-recomputation checks, and frozen-phase integrity.

**What Phase 7 does NOT demonstrate:**

- Planting success, survival, or realized cooling (no biological or causal model).
- Land availability, tenure, or authorization (no such layers exist).
- Physical trends between 2022 and 2026 (two class-relative snapshots).
- Exact priority extents (highly weight-sensitive, §22) or administrative-level priorities.

Phase 7's correct use is as the **first filter** in a planting program: a transparent, reproducible, conservatively gated shortlist that tells planners where to look first — and, just as clearly, where this data cannot speak.

---

## Appendix — Key Figures

All figures are in `data/processed/phase7/figures/` (220 dpi; explicit NoData handling).

| Figure | Caption |
|---|---|
| `fig01_heat_need_2022.png` | Heat Need surface (0–100), July 2022 composite |
| `fig02_heat_need_2026.png` | Heat Need surface (0–100), July 2026 composite |
| `fig03_opportunity_2022.png` | Plantation Opportunity surface (0–100), 2022 (static base + NDBI/NDVI terms) |
| `fig04_opportunity_2026.png` | Plantation Opportunity surface (0–100), 2026 |
| `fig05_suitability_2022.png` | Baseline gated suitability (Scenario A), 2022 |
| `fig06_suitability_2026.png` | Baseline gated suitability (Scenario A), 2026 |
| `fig07_very_high_priority_zones.png` | Very High (class 4) priority zones, both years (empty under baseline — shown for completeness) |
| `fig08_high_priority_zones.png` | High (class ≥ 3) priority-zone boundaries over suitability, both years |
| `fig09_green_vs_built.png` | Green/built/mixed groups vs mean suitability (association, not causation) |
| `fig10_landuse_suitability.png` | Mean suitability per OSM land-use class (class-0 background caveat) |
| `fig11_temporal_priority_transition.png` | 2022→2026 priority persistence categories (class-relative snapshots) |
| `fig12_sensitivity_analysis.png` | Scenario A–D suitability comparison and High+VH extent spread |
| `fig13_need_vs_opportunity.png` | Need vs Opportunity scatter/density — the gated product's two-factor structure |

---

*GreenGrid AI — Phase 7 Tree Plantation Suitability Report | Generated 2026-09-05*
