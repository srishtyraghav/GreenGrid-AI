# GreenGrid AI — Phase 8: Tree Requirement Estimation (Production)

**Project:** GreenGrid AI
**Phase:** 8 — Tree Plantation Requirement (how many trees, where)
**Study Area:** National Capital Territory (NCT) of Delhi, India
**Report Date:** 2026-10-03
**Status:** ✅ COMPLETED — production run over all five years (2022–2026) | Validation gate G4: **64 PASS / 0 FAIL**

---

## 1. Executive Summary — the synopsis answer

**How many trees, and where?** Taking the Phase 7 priority zones, removing pixels that are excluded, ineligible, or already vegetated, and planting the remainder at **1,000 trees/ha (~3.2 m spacing)**:

| Year | Priority zones | Plantable area (ha) | **Recommended trees** |
|---|---|---|---|
| 2022 | 69 | 272.52 | **272,520** |
| 2023 | 118 | 713.43 | **713,430** |
| 2024 | 36 | 147.87 | **147,870** |
| 2025 | 6 | 15.03 | **15,030** |
| **2026 (focus year)** | **11** | **44.28** | **44,280** |

Source: `tables/tree_requirement_citywide_5yr.csv` (G4-verified). **Where:** inside the Phase 7 priority zones — for 2026, eleven zones led by **zone 6 (17.73 ha, 17,730 trees, High priority)**, with per-zone polygons in `vectors/recommended_plantations_2026.geojson` and point locations in `vectors/recommended_locations_2026.geojson`.

These are **planning estimates on potential plantation suitability** — an upper bound assuming full conversion of plantable space at the stated density. They are not legal-availability, ownership, field-verified plantability, or survival-probability statements (`phase8_manifest.json` → `terminology_note`).

## 2. Objective

Quantify the tree-plantation requirement implied by the Phase 6/7 production chain: convert priority zones into plantable hectares, plantable hectares into tree counts at a documented density, and package the answer as maps, zone tables, polygons, and point locations for decision-makers.

## 3. Inputs and Lineage

Chain (`phase8_manifest.json` → `model_lineage`): frozen Phase-5 3-class XGBoost → `phase6_production` severity rasters → `phase7_production` suitability + priority zones → this stage. Concrete inputs, all checksum-listed in the manifest:

- Phase 7 priority zones (kept 8-connected class ≥ 3 clusters, ≥ 10 px) and `priority_ranking_{year}.csv`
- Phase 7 `suitability_{year}.tif`, `suitability_class_{year}.tif`, `exclusion_mask_{year}.tif`
- Phase 6 `severity_{year}.tif`, `severity_score_{year}.tif`
- Phase 4 `vegetation_cover_{year}_30m.tif`; Phase 3 `landuse_raster_30m.tif`

## 4. Locked Assumptions (from the manifest)

1. **Plantable pixel rule:** plantable = priority-zone pixel **AND** not excluded (Phase 7 exclusion mask) **AND** landuse eligible **AND** `vegetation_cover < 0.30`.
   - "Priority-zone pixels" = pixels of the **kept** Phase 7 zones only; class ≥ 3 pixels inside dropped sub-10-px noise clusters are removed first (2022: 575 px, 2023: 1,116 px, 2024: 289 px, 2025: 85 px, 2026: 85 px).
   - Landuse eligible = `LANDUSE_RULE_STATUS != 'DISCOURAGED'` (industrial class 5, retail class 7 excluded; nodata 255 treated as neutral-eligible). **~99% of priority-zone pixels are residential (class 6)**, so results are insensitive to this choice (`phase8_manifest.json` → deviations).
2. **Tree density:** **1,000 trees/ha** (~3.2 m spacing, mid-range urban plantation). Pixel arithmetic: 30 m × 30 m = 0.09 ha → **90 trees per plantable pixel** at 1,000/ha. Sensitivity at **400 and 2,500 trees/ha** (36 / 225 trees per px) ships in every table. (The task text's "9 trees/px" is a documented decimal slip; the self-consistent rule `trees = plantable_ha × density` is used everywhere — `phase8_manifest.json` deviations[0], confirmed by gate G4.)
3. **Area rule:** `area_ha = pixel_count × 900 m² / 1e4`.
4. **Priority labels:** zones ordered by Phase 7 priority ranking (mean suitability desc); **High = ranks 1..⌈n/3⌉, Medium = next ⌈n/3⌉, Low = remainder** (n=11 → 4/4/3; n=69 → 23/23/23).
5. **Zone statistics:** per-zone current vegetation (%) is the mean `vegetation_cover` over **all** zone pixels (the zone's existing vegetation state), not only plantable pixels.

## 5. Results

### 5.1 Citywide headline table

`tables/tree_requirement_citywide_5yr.csv` (G4 check `citywide_5yr_table` PASS):

| Year | Zones | Zone area (ha) | Plantable px | Plantable ha | Trees @400/ha | **Trees @1,000/ha** | Trees @2,500/ha |
|---|---|---|---|---|---|---|---|
| 2022 | 69 | 272.52 | 3,028 | 272.52 | 109,008 | **272,520** | 681,300 |
| 2023 | 118 | 713.52 | 7,927 | 713.43 | 285,372 | **713,430** | 1,783,575 |
| 2024 | 36 | 147.87 | 1,643 | 147.87 | 59,148 | **147,870** | 369,675 |
| 2025 | 6 | 15.03 | 167 | 15.03 | 6,012 | **15,030** | 37,575 |
| **2026** | **11** | **44.28** | **492** | **44.28** | **17,712** | **44,280** | **110,700** |

(2023: zone area 713.52 ha = 7,928 px, of which 1 px has vegetation_cover ≥ 0.30, leaving 7,927 plantable px / 713.43 ha — the only pixel removed by the vegetation filter in any year.)

### 5.2 Where — 2026 per-zone detail

`tables/tree_requirement_by_zone_2026.csv` (top 5 by priority ranking):

| Zone | Area (ha) | Current vegetation | UHI severity (score / High%) | Recommended trees | Priority |
|---|---|---|---|---|---|
| **6** | **17.73** | **0.50%** | **1.898 / 100%** | **17,730** | **High** |
| 10 | 2.34 | 0.00% | 1.751 / 100% | 2,340 | High |
| 5 | 3.06 | 1.92% | 1.946 / 100% | 3,060 | High |
| 8 | 2.43 | 0.01% | 1.760 / 100% | 2,430 | High |
| 4 | 0.99 | 3.24% | 1.925 / 100% | 990 | Medium |

(Remaining 2026 zones: 7 → 2,700 Medium; 2 → 3,600 Medium; 11 → 3,870 Medium; 3 → 990 Low; 9 → 4,590 Low; 1 → 1,980 Low. Severity = Phase-6 mean severity_score; every 2026 zone is 100% model-High severity.) Full per-zone fields including mean suitability and centroids are in the table; all 5 years have equivalent `tree_requirement_by_zone_{year}.csv` files, and per-year density breakdowns in `tree_requirement_summary_{year}.csv`.

### 5.3 Sensitivity (2026)

| Density (trees/ha) | Trees per px | Recommended trees (2026) |
|---|---|---|
| 400 (sparse) | 36 | 17,712 |
| **1,000 (primary)** | **90** | **44,280** |
| 2,500 (dense) | 225 | 110,700 |

## 6. Deliverables

- **Vectors** (`vectors/`): `recommended_plantations_{2022..2026}.geojson` — one polygon per 8-connected plantable cluster (69/118/36/6/11 polygons, EPSG:4326, per-polygon area ha + tree counts); `recommended_locations_2026.geojson` — 11 point locations (one per 2026 plantation cluster, area-centroid based) with per-point tree counts
- **Rasters** (`rasters/`, per year): `available_planting_space` (uint8: 1 plantable / 0 in-domain not plantable / 255 nodata) and `recommended_trees` (int16: 90 plantable / 0 / −1 nodata)
- **Tables** (`tables/`): `tree_requirement_citywide_5yr.csv`, `tree_requirement_by_zone_{year}.csv`, `tree_requirement_summary_{year}.csv`
- **Figures** (`figures/`): `plantable_ha_trees_5yr.png`, `recommended_plantations_2026_map.png`, `top10_zone_trees_2026.png`

## 7. Validation (Gate G4)

`verification_g4.json`: **64 checks, 64 passed, 0 failed (PASS)**. Highlights: both rasters per year have correct grid (1768×1874), dtype, and nodata; planting-space raster content matches recomputation with **0 mismatch px**; `recommended_trees` raster sums equal expected tree sums exactly; plantable is disjoint from exclusion, from vegetation_cover ≥ 0.30, and from discouraged landuse; zone area reconciliation to ≤ 2e-16 relative error with zone identity matching Phase 7; priority label counts (2026: 4 High / 4 Medium / 3 Low); summary-table tree arithmetic at all three densities; plantation GeoJSON accounting (cluster counts, area and tree sums, valid geometries); deep zone recompute of ranks 1/6/11 (zone 6 = 17,730 trees High ✓, zone 7 = 2,700 Medium ✓, zone 1 = 1,980 Low ✓); and manifest lineage.

## 8. Honest Caveats

1. **Year-relative classes → hectares differ by domain.** Recommended hectares inherit Phase 7's per-year p1/p99 normalization and Phase 6's per-year domains; 2024/2025 totals describe the cloud-free subset. **Do not read the 5-year column as a planting trend.**
2. **Tree counts are an upper bound.** They assume every plantable pixel converts to plantation at the stated density — no allowance for tenure, access, utilities, underground infrastructure, or survival.
3. **~99% residential.** Priority-zone pixels sit almost entirely in the residential landuse class; the product is an urban-greening plan, not a regional one.
4. **Point locations are area centroids of non-convex polygons.** Shapely area-centroids can fall just outside U/L-shaped pixel unions: G4 confirms all 11 points lie within 1 px diagonal of their polygon and 9/11 are strictly covered — treat points as zone markers, not planting spots; the polygons are the authoritative geometry.
5. **Sensitivity is linear in density** (±2.5× around the primary estimate); the larger uncertainty is the suitability model chain above it.

## 9. How to Read This (for decision-makers)

- Start from `recommended_plantations_2026.geojson` (or the map figure): the 11 polygons are *where*; per-polygon attributes give hectares and trees.
- **Zone 6 first.** It is 40% of the 2026 plantable area (17.73 of 44.28 ha), has the city's worst severity profile in this product (mean severity score 1.898, 100% High, only 0.5% existing vegetation), and alone accounts for 17,730 of 44,280 trees.
- The **High-priority tier (4 zones: 6, 10, 5, 8)** covers 25.56 ha / 25,560 trees — a natural first budget cycle; Medium and Low tiers follow the same ranking.
- If budget covers only part of a zone, scale linearly: trees ≈ hectares planted × 1,000 (use 400/ha for sparse or 2,500/ha for dense planting; the sensitivity table brackets the answer).
- 2022–2025 rows give historical context on the same relative scale; they are not additive across years and not directly comparable in hectares (see §8.1).
