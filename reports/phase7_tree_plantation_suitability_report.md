# GreenGrid AI — Phase 7: Tree Plantation Suitability (Production Re-run)

**Project:** GreenGrid AI
**Phase:** 7 — Tree Plantation Suitability & Priority Zones, production re-run
**Study Area:** National Capital Territory (NCT) of Delhi, India
**Report Date:** 2026-10-03
**Status:** ✅ COMPLETED — production re-run over all five years (2022–2026) | Validation gate G3: **87 PASS / 0 FAIL**

> **Supersedes notice.** This report **supersedes the original Phase 7 report of 2026-09-02** (2-year, 4-class baseline with 0–3 severity score). The old Phase 7 outputs were deleted in the production re-run; the superseded report and outputs remain in git history. All numbers below trace to `data/processed/phase7_production/`.

---

## 1. Executive Summary

Phase 7 converts the Phase 6 production heat-severity products into a **GIS multi-criteria decision-support layer**: a 0–100 **potential tree-plantation suitability** score per pixel, a five-class suitability map, and delineated **priority zones** for each of the five study years (2022–2026).

Suitability is a **gated product** of two 0–100 factors — Heat Need and Plantation Opportunity: `S = Need × Opportunity / 100`, so extreme heat alone cannot create high suitability on low-opportunity land.

**Headline results (baseline Scenario A, gated)** — from `tables/suitability_summary_{year}.csv` and `verification_g3.json`:

| Quantity | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| Domain px / ha | 1,512,679 / 136,141.11 | 1,370,704 / 123,363.36 | 583,573 / 52,521.57 | 312,400 / 28,116.00 | 1,894,980 / 170,548.20 |
| Mean suitability (0–100) | 24.93 | 24.43 | 24.56 | 24.03 | 24.66 |
| High class area (ha, %) | 324.27 (0.238%) | 813.96 (0.660%) | 173.88 (0.331%) | 22.68 (0.081%) | 51.93 (0.030%) |
| Very High area (ha) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Priority zones (class ≥ 3, ≥ 10 px, 8-conn) | 69 | 118 | 36 | 6 | 11 |

The 2026 priority set is dominated by one large zone: **zone 6, 17.73 ha, mean suitability 62.30** (`tables/priority_ranking_2026.csv`).

**Terminology note (from the production manifest).** Outputs describe **POTENTIAL PLANTATION SUITABILITY** — a relative, decision-support ranking on per-year-normalized inputs. They are NOT a statement of legal availability, land ownership, or a physical plantability guarantee, and make no claim of planting survival probability.

## 2. Objective

Rank every valid pixel by *potential plantation suitability* and delineate candidate priority zones where high heat need and real planting opportunity co-occur — the "where and why" input to Phase 8 tree-requirement estimation. Phase 7 is **decision support**, not planting authorization.

## 3. Inputs (Phase 6 production + static layers)

Per `phase7_production_manifest.json`:

| Input | Source |
|---|---|
| severity_score [0,2], LST (per year) | `data/processed/phase6_production/rasters/` (frozen 3-class production model output) |
| NDVI, NDBI (per year, 30 m aligned) | `data/processed/phase3/aligned/s2_{year}_{ndvi,ndbi}_30m.tif` |
| vegetation_cover (per year) | `data/processed/phase4/features/vegetation_cover_{year}_30m.tif` |
| Landuse raster, road distance, vegetation distance (static) | `data/processed/phase3/masks/` |
| Analysis domain | per-year Phase-6 valid domain (all 178 features finite; spec locked decision #1) |

Model lineage: `phase5_primary_xgb_3class` (SHA-256 recorded) → Phase 6 production rasters → this stage. Static Opportunity terms (landuse eligibility, road accessibility, green proximity) reuse the static Phase-3 rasters for every year; year-to-year Opportunity variation comes only from the NDBI/NDVI terms.

## 4. Method (frozen weights)

**Normalization.** Each dynamic input is normalized **per year** by the robust p1–p99 min-max rule (`frozen robust_minmax`, percentiles [1, 99] recorded in `tables/normalization_parameters_{year}.csv`; G3 recompute deviation ≤ 3.6e-15). The rule is range-agnostic, which is why the 0–2 severity score (3-class production model) replaces the old 0–3 score without any formula change.

**Heat Need (0–100):**

```
Need = 0.40·n(severity_score) + 0.25·n(1 − NDVI) + 0.20·n(NDBI) + 0.15·n(LST)
```

**Plantation Opportunity (0–100):**

```
Opportunity = 0.30·eligibility + 0.25·(100 − n(NDBI)) + 0.15·road_access
            + 0.15·green_proximity + 0.15·(100 − n(NDVI))        # planting headroom
```

with the frozen landuse eligibility scores {0:50, 1:40, 2:30, 3:50, 4:35, 5:20, 6:90, 7:30, 8:60}, road band (≤50 m corridor 40 pts; 500 m optimal 100 pts; 2000 m decline floor 30 pts), and 500 m green-proximity cap (`phase7_production_manifest.json` → `frozen_parameters`).

**Gated suitability and classes.** Baseline Scenario A: `S = Need × Opportunity / 100`. Sensitivity scenarios B/C/D re-gate the same Need/Opportunity with frozen exponents (A/D [1.0, 1.0]; B [0.6, 0.3] heat-focused; C [0.3, 0.5]); Scenario D additionally swaps in vegetation-cover-based need/headroom weights. All four scenario rasters ship per year. Five classes at frozen edges 20/40/60/80: 0 Very Low, 1 Low, 2 Medium, 3 High, 4 Very High; score nodata −1.

**Priority zones.** Pixels of suitability class ≥ 3 (0-indexed High + Very High), 8-connected clustering, **min zone size 10 px**; zone areas by `pixel_count × 900 m² / 1e4` (G3-enforced; polygons written in EPSG:4326). Ranked by mean suitability descending (`tables/priority_ranking_{year}.csv`).

## 5. Results

**Class distribution and area** (`tables/suitability_summary_{year}.csv`; G3-verified `summary_table_matches_raster` per year):

| Year | Very Low | Low | Medium | High | Very High |
|---|---|---|---|---|---|
| 2022 | 33.67% / 45,843.30 ha | 57.63% / 78,462.54 ha | 8.46% / 11,511.00 ha | 0.238% / 324.27 ha | 0.0 ha |
| 2023 | 41.70% / 51,437.61 ha | 46.30% / 57,118.86 ha | 11.34% / 13,992.93 ha | 0.660% / 813.96 ha | 0.0 ha |
| 2024 | 37.83% / 19,871.37 ha | 52.76% / 27,708.39 ha | 9.08% / 4,767.93 ha | 0.331% / 173.88 ha | 0.0 ha |
| 2025 | 40.19% / 11,300.94 ha | 49.18% / 13,827.42 ha | 10.55% / 2,964.96 ha | 0.081% / 22.68 ha | 0.0 ha |
| 2026 | 37.37% / 63,729.54 ha | 52.08% / 88,813.89 ha | 10.53% / 17,952.84 ha | 0.030% / 51.93 ha | 0.0 ha |

Mean Need / Opportunity / Suitability per year (`phase7_production_manifest.json` → `per_year`): Need 52.89 / 51.76 / 50.99 / 48.86 / 50.12; Opportunity 48.81 / 47.65 / 48.99 / 49.79 / 49.98; Suitability 24.93 / 24.43 / 24.56 / 24.03 / 24.66 (2022→2026). The near-constant means are expected — per-year p1/p99 normalization stretches each year's inputs to a common scale by construction (see §8).

**Top-5 priority zones, 2026** (`tables/priority_ranking_2026.csv`; S = mean suitability, 0–100):

| Rank | Zone | Pixels | Area (ha) | Mean S | Mean Need | Mean Opportunity |
|---|---|---|---|---|---|---|
| 1 | 6 | 197 | 17.73 | 62.30 | 83.97 | 74.21 |
| 2 | 10 | 26 | 2.34 | 61.63 | 80.65 | 76.42 |
| 3 | 5 | 34 | 3.06 | 61.54 | 89.76 | 68.58 |
| 4 | 8 | 27 | 2.43 | 61.34 | 81.38 | 75.39 |
| 5 | 4 | 11 | 0.99 | 61.26 | 82.58 | 74.19 |

All 11 zones of 2026 have mean class 3.0 (dominant class High, 0-indexed). Centroids and per-zone need/opportunity breakdowns are in `tables/priority_zone_statistics_2026.csv`.

**Temporal zone transition 2022 → 2026** (`tables/temporal_priority_transition.csv`; raster pixel-overlap rule: a pair is persistent if intersection px / min(px_2022, px_2026) ≥ 0.5). Of **78** zone records: **2 persistent, 67 declining, 9 emerging** (verified in `verification_g3.json` → `temporal_transition_table`).

- Persistent: 2022 zone 68 ↔ 2026 zone 1 (2.16 → 1.98 ha, overlap 0.864); 2022 zone 69 ↔ 2026 zone 5 (3.78 → 3.06 ha, overlap 0.971).
- Emerging 2026 zones include the year's dominant zone 6 (17.73 ha) and zone 2 (3.60 ha).

## 6. Validation (Gate G3)

`verification_g3.json`: **87 checks, 87 passed, 0 failed (PASS)**, covering per year: grid metadata for all 7 raster types (shape 1768×1874, dtypes, nodata −1/255); exclusion mask encodes exactly the domain complement; `class == classify(score)` on up to 1,000,000 sampled px with **0 mismatches** (full domain sampled in 2024/2025); value ranges; zone pixel accounting (independent cluster count == GeoJSON px sum == expected ha, rel err ≤ 2.1e-16, unique IDs, valid geometries); normalization-parameter recompute (≤ 3.6e-15); Need/Opportunity/S and scenarios B/C/D recomputation on 200,000 px (max dev ≤ 3.9e-06, tol 1e-4); summary-table-vs-raster match; ranking-table ordering; 5-year completeness; and the temporal transition table categories (67/9/2).

## 7. Figures and Files

- **Figures** (`figures/`): `suitability_class_map_{2022..2026}.png` (5), `need_vs_opportunity_{2022..2026}.png` (5), `suitability_5year_summary.png`
- **Rasters** (`rasters/`, per year): `suitability`, `suitability_class`, `heat_need`, `plantation_opportunity`, `exclusion_mask` (uint8, 1=excluded, 0=domain, nodata 255), `suitability_scenario_b/c/d`
- **Tables** (`tables/`): `suitability_summary_{year}`, `suitability_area_statistics_{year}`, `priority_zone_statistics_{year}`, `priority_ranking_{year}`, `normalization_parameters{,_year}`, `temporal_priority_transition.csv`
- **Vectors** (`zones/`): `priority_zones_{year}.geojson` (69/118/36/6/11 polygons)
- **Records**: `phase7_production_manifest.json`, `phase7_pipeline_record.json`, `verification_g3.json`

## 8. Honest Notes and Caveats

1. **Year-relative classes — do NOT read as trends.** Every input is p1/p99-normalized within its own year, so class *areas* are year-relative by construction: a year with a larger/smaller domain or different cloud-free geography (2024/2025 are cloud-hole subsets) will show different High-class hectares even under identical on-the-ground conditions. The 2022→2026 transition table's 67 "declining" zones largely reflect this relativity plus domain change, not verified on-the-ground loss of suitability.
2. **Very High class = 0 ha in all five years.** This is **gated-product compression**, identical to the old baseline build: with `S = Need × Opportunity / 100` and both factors realistically bounded away from 100, scores rarely exceed the 80-point Very High edge. It is a property of the frozen formula, not an absence of need.
3. **Frozen weights, documented reimplementations.** All weights, exponents, eligibility scores, and band parameters are the frozen `src/suitability` values; the frozen primitives (`robust_minmax`, `gated_product`, `weighted_geometric_mean`) are reused unchanged. The per-year loop, input loading, exclusion-mask writing, and zone-area accounting were reimplemented for the 5-year production domain — every deviation is listed verbatim in `phase7_production_manifest.json` → `deviations` (including: zone area now `px × 900 m² / 1e4` instead of UTM polygon reprojection; zone level = class ≥ 3 0-indexed; Scenario D exponents (1,1) make the weighted geometric mean identical to the gated product).
4. **Conservative baseline, scenario-conditional.** As in the original design, the gated baseline is deliberately conservative; the B/C/D scenario rasters per year allow heat-focused or opportunity-focused re-rankings without changing the frozen Need/Opportunity surfaces.
5. **Inherited caveats.** Outputs inherit Phase 6's per-year relative severity and the 2024/2025 cloud-hole domains (see the Phase 6 report §10); priority-zone extents in those years describe the cloud-free subset only.
