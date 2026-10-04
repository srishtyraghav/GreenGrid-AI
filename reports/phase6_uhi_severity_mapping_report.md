# GreenGrid AI — Phase 6: UHI Severity Mapping (Production Re-run)

**Project:** GreenGrid AI
**Phase:** 6 — UHI Severity Mapping, full-grid production re-run
**Study Area:** National Capital Territory (NCT) of Delhi, India
**Report Date:** 2026-10-03
**Status:** ✅ COMPLETED — production re-run against the frozen Phase-5 3-class model | Gate G1 PASS (feature parity, bit-identical) | Gate G2 PASS (rasters + hotspots + anchor)

> **Supersedes notice.** This report **supersedes the original Phase 6 report of 2026-09-02** (2-year, 4-class, RF-C baseline). The old Phase 6 outputs were deleted in the production re-run; the superseded report and its outputs remain available in git history. All numbers below trace to `data/processed/phase6_production/` and nothing else.

---

## 1. Objective

Phase 6 converts the frozen Phase-5 production model into **full-grid decision-support analytics** for all five study years (July 2022–2026): per-pixel severity maps, delineated heat hotspots, area-wise statistics, green-vs-built-up comparisons, vegetation–temperature relationships, and a five-year temporal comparison.

**Scientific framing:** all outputs describe **ML-based relative heat severity** — an **operational UHI hotspot proxy**, not physical UHI intensity. Classes are per-year model outputs relative to each year's own distribution; they are not directly cross-year comparable (see §10).

## 2. Inputs and Model Lineage

| Item | Value | Source |
|---|---|---|
| Production model | `phase5_primary_xgb_3class` (XGBoost `multi:softprob`, 3-class) | `data/processed/phase5_production_3class/phase5_primary_xgb_3class.json` |
| Feature schema | 178 encoded features, order locked | `phase5_primary_xgb_3class_features.json` |
| Locked holdout metrics | accuracy **0.6396665686967289**, macro-F1 **0.6314607179023212** (blocks [2, 9, 15, 23]) | `fullgrid/verification_g1.json` |
| Training dataset | 749,998 sampled pixels, 5 years (2022–2026) | `data/processed/lulc_outputs/combined_urban_environmental_dataset.csv` |
| Spec | `reports/experiments/docs_production_build_spec.md` (2026-10-03, sha256 recorded in manifest) | `phase6_production_manifest.json` |

The booster SHA-256 (`2568cb79…57b1a4`) and schema SHA-256 are recorded in `phase6_production_manifest.json`; every input parquet, LST composite, and output raster is checksum-listed there.

## 3. Verification Gates

**Gate G1 — feature parity (PASS).** Full-grid 178-feature tables were built per year and checked two ways (`fullgrid/verification_g1.json`):

- Predicting at all **749,998 training pixels** from the full-grid features reproduces the locked accuracy 0.6396665686967289 / macro-F1 0.6314607179023212 **exactly**, with `predictions_identical_to_training_table: true`.
- Feature columns match the 178-schema names and order exactly; worst per-feature deviation vs the training table is 1.53e-05 (`met_ssr_wm2`); **0 columns over tolerance**.
- Sanity ranges pass (vegetation_cover ∈ [0,1], window stats ≥ 0, morphology fractions ∈ [0,1]); spectral indices outside [−1,1] (mndwi/bsi ratio artefacts at near-zero denominators) are recorded warnings, verified to be verbatim source-composite values.

**Gate G2 — raster and hotspot integrity (PASS, `verification_g2.json`).** Grid metadata (1768×1874, EPSG:4326, nodata −1) verified for every raster; class histograms sum exactly to the per-year domain; per-class probability row-sums deviate ≤ 8.94e-08; LST means match source composites to ≤ 2.8e-06 °C with identical valid-pixel counts. All 15 hotspot products (3 definitions × 5 years) recompute exactly (ID rasters, cluster counts, pixel counts, GeoJSON areas; min cluster 10 px).

**Anchor bit-identity.** A stratified anchor sample (year × spatial_block_id, 110/combo, seed 42; **10,859 pixels checked, 0 mismatches, 0 missing**) confirms grid predictions are identical in class to training-table predictions; `g1_predictions_identical_to_training_table_all_749998_px: true` (`verification_g2.json` → `anchor`).

## 4. Method

**Analysis domain (per year).** All grid pixels where **all 178 features are finite** (mirrors training dropna). Domain sizes (`fullgrid/domain_counts.json`, `phase6_production_manifest.json`):

| Year | Domain px | Grid px | LST-finite px |
|---|---|---|---|
| 2022 | 1,512,679 | 3,313,232 | 1,743,888 |
| 2023 | 1,370,704 | 3,313,232 | 1,716,180 |
| 2024 | 583,573 | 3,313,232 | 1,355,099 |
| 2025 | 312,400 | 3,313,232 | 831,154 |
| 2026 | 1,894,980 | 3,313,232 | 1,894,985 |

LST rasters are written wherever LST is finite (separate, larger count) so temperature analytics are not clipped to the feature domain.

**Feature construction.** The 178 features comprise 101 base predictors + 6 meteorological + 33 spatial-context features + encoded landuse/lulc dummies. Key engineering per the locked spec:

- **Met features (6):** IDW-interpolated from the 16 provenance grid points to pixel centres. The exact IDW variant was reverse-engineered against `data/processed/experiments/exp_met_features.csv` (all 749,998 rows) to **max abs error 5.12e-13**: *power-2 IDW over geographic degree distance with longitude scaled by cos(28.64°), squared-distance epsilon 1e-6* — i.e. weight `w = 1 / (dlat² + (cos(28.64°)·dlon)² + 1e-6)` (`verification_g1.json` → `met_idw`).
- **Spatial-context (33):** nan-safe, border-correct window statistics at 3/5/11 px scales plus spatial-cache focal means and morphology disks (50–500 m), computed on full grids.
- **Schema alignment:** after one-hot encoding, columns are reindexed to the locked 178-order; category dummies absent in a given year are zero-filled; landuse/lulc dtypes are cast to the training dtypes before encoding so dummy names match the schema byte-for-byte.

**Prediction and outputs.** The frozen Booster predicts via XGBoost `DMatrix` on the aligned full-grid table. Per pixel and year the pipeline writes (`rasters/`): `severity` (argmax class 0/1/2, int16), `probability_low/moderate/high` (float32), `confidence` (max class probability), `severity_score` = **probs · [0, 1, 2]** (range [0,2]; the 3-class replacement for the old 0–3 score), and `lst`. Nodata = −1 throughout; LZW compression.

**Hotspots.** Three documented definitions (G2-verified), all with **min cluster size 10 px, 8-connectivity**:

- **A** = severity High (class 2)
- **B** = severity High **AND confidence ≥ 0.60** (headline definition)
- **C** = severity_score ≥ 1.5 (top half of the High band) **AND confidence ≥ 0.60**

Areas use `area_ha = pixel_count × 900 m² / 1e4` (30 m px).

## 5. Severity Class Distribution

From `tables/area_statistics_{year}.csv` (percent of the year's valid domain):

| Year | Low | Moderate | High |
|---|---|---|---|
| 2022 | 30.68% (41,761.71 ha) | 33.23% (45,244.71 ha) | 36.09% (49,134.69 ha) |
| 2023 | 39.34% (48,531.60 ha) | 30.49% (37,612.08 ha) | 30.17% (37,219.68 ha) |
| 2024 | 32.66% (17,153.28 ha) | 32.97% (17,316.09 ha) | 34.37% (18,052.20 ha) |
| 2025 | 35.26% (9,913.14 ha) | 34.32% (9,649.53 ha) | 30.42% (8,553.33 ha) |
| 2026 | 34.00% (57,990.06 ha) | 31.56% (53,818.29 ha) | 34.44% (58,739.85 ha) |

## 6. Hotspot Results

Per-year cluster counts and area (from `tables/hotspot_statistics_def{A,B,C}_{year}.csv`; all G2-verified):

| Definition | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| A — clusters | 469 | 532 | 502 | 225 | 521 |
| A — px / ha | 543,396 / 48,905.64 | 410,667 / 36,960.03 | 198,668 / 17,880.12 | 93,989 / 8,459.01 | 649,672 / 58,470.48 |
| **B — clusters** | **487** | **437** | **438** | **172** | **567** |
| **B — px / ha** | **410,293 / 36,926.37** | **299,339 / 26,940.51** | **127,616 / 11,485.44** | **59,202 / 5,328.18** | **455,220 / 40,969.80** |
| C — clusters | 483 | 438 | 438 | 181 | 598 |
| C — px / ha | 409,050 / 36,814.50 | 282,316 / 25,408.44 | 123,148 / 11,083.32 | 56,193 / 5,057.37 | 427,008 / 38,430.72 |

Confidence gating (B vs A) removes 24–46% of High pixels depending on year; the score gate (C) removes only a little more, showing High-class probabilities concentrate near the top of the band (mean severity_score ≈ 1.74–1.78 in def-B/C clusters vs ≈ 1.56–1.69 in def A; mean confidence ≈ 0.74–0.83). Hotspot ID rasters (`hotspots/hotspot_ids_def{X}_{year}.tif`, int32, nodata −1) and cluster polygons (`hotspots/hotspots_def{X}_{year}.geojson`) accompany the tables.

## 7. Analytics

**Green vs built-up** (`tables/green_built_comparison_{year}.csv`; domain split into per-year tertiles of vegetation_cover and NDBI). Mean severity_score (High fraction) by tertile:

| Year | Veg tertile 1 (sparse) | Veg tertile 3 (dense) | NDBI tertile 1 (green) | NDBI tertile 3 (built) |
|---|---|---|---|---|
| 2022 | 1.293 (51.5%) | 0.676 (17.0%) | 0.633 (13.2%) | 1.390 (58.9%) |
| 2023 | 1.304 (58.4%) | 0.540 (6.4%) | 0.541 (7.2%) | 1.361 (60.5%) |
| 2024 | 1.280 (60.7%) | 0.715 (13.7%) | 0.659 (10.9%) | 1.324 (63.2%) |
| 2025 | 1.252 (54.4%) | 0.556 (10.0%) | 0.624 (11.4%) | 1.234 (53.4%) |
| 2026 | 1.364 (57.2%) | 0.599 (13.3%) | 0.563 (11.9%) | 1.406 (63.8%) |

The gradient is monotone and strong in every year: sparse-vegetation / built-up tertiles carry roughly **2× the mean severity_score and 3–5× the High fraction** of dense-vegetation / green tertiles.

**Vegetation–LST decile relationship** (`tables/vegetation_temperature_relationship_{year}.csv`). Mean LST falls monotonically from the lowest to the highest vegetation-cover decile:

- **2022:** 41.09 °C (decile 1) → 34.49 °C (decile 10) — a −6.6 °C gradient; mean severity_score 1.301 → 0.474.
- **2026:** 35.79 °C → 32.97 °C — a −2.8 °C gradient; mean severity_score 1.418 → 0.352.

**Temporal comparison** (`tables/temporal_comparison.csv`):

| Year | Valid domain px | Low % | Mod % | High % | Def-B area ha | Def-B clusters | Mean LST °C (domain) | Mean severity_score |
|---|---|---|---|---|---|---|---|---|
| 2022 | 1,512,679 | 30.68 | 33.23 | 36.09 | 36,926.37 | 487 | 38.76 | 1.0127 |
| 2023 | 1,370,704 | 39.34 | 30.49 | 30.17 | 26,940.51 | 437 | 36.32 | 0.9138 |
| 2024 | 583,573 | 32.66 | 32.97 | 34.37 | 11,485.44 | 438 | 38.12 | 0.9852 |
| 2025 | 312,400 | 35.26 | 34.32 | 30.42 | 5,328.18 | 172 | 39.19 | 0.9137 |
| 2026 | 1,894,980 | 34.00 | 31.56 | 34.44 | 40,969.80 | 567 | 34.87 | 0.9995 |

## 8. Figures

All in `data/processed/phase6_production/figures/`: `severity_map_{2022..2026}.png` (5), `hotspot_defB_map_{2022..2026}.png` (5), `temporal_comparison.png`.

## 9. Output Files

- `rasters/` — 35 GeoTIFFs: `severity`, `probability_low/moderate/high`, `confidence`, `severity_score`, `lst` × 5 years (1768×1874, EPSG:4326, nodata −1)
- `hotspots/` — 15 ID rasters + 15 cluster GeoJSONs (defs A/B/C × 5 years)
- `tables/` — `area_statistics_{year}.csv`, `hotspot_statistics_def{A,B,C}_{year}.csv`, `green_built_comparison_{year}.csv`, `vegetation_temperature_relationship_{year}.csv`, `temporal_comparison.csv`
- `fullgrid/` — `features_{2022..2026}.parquet` (float32, 178 cols), `domain_counts.json`, `verification_g1.json`
- `figures/` — 11 PNGs
- `phase6_production_manifest.json`, `phase6_pipeline_record.json`, `verification_g2.json`

## 10. Honest Notes and Caveats

1. **Small 2024/2025 domains are real, not error.** The 2024 (0.58M px) and 2025 (0.31M px) valid domains are small because of **monsoon cloud holes in the source composites** (2025 imagery was particularly affected). All 2024/2025 statistics describe the cloud-free subset only (`phase6_production_manifest.json` notes; G1 sanity-range warnings).
2. **Classes are per-year model outputs.** Severity classes come from a model trained on per-year LST-tertile targets; percentages and areas **cannot be read directly as cross-year trends**. The temporal table is provided for completeness and domain context; interpret year-to-year differences with the domain-size caveat (point 1) and per-year relative scales (point 4) in mind.
3. **Severity is a relative hotspot proxy, NOT physical UHI intensity.** No rural baseline or air-temperature validation exists in this project. "Hot" means "ranked hot by the model relative to that year's distribution".
4. **Per-year relative scales.** severity_score, class boundaries, and hotspot thresholds are all year-internal. A pixel scored 1.8 in 2023 and 1.8 in 2026 is hot *within its own year*; the numbers do not certify identical physical conditions across years.
5. **Inherited model ceiling.** Full-grid outputs inherit the frozen model's locked accuracy (63.97% / macro-F1 0.6315 on the untouched geographic holdout); mapped classes are model estimates, not ground truth.

## 11. Validation Summary

| Gate | Scope | Result |
|---|---|---|
| G1 | 178-feature schema parity; locked metrics + all 749,998 training-pixel predictions reproduced bit-identically; met-IDW variant reverse-engineered to <1e-12; sanity ranges | **PASS** (`fullgrid/verification_g1.json`) |
| G2 | Grid metadata, nodata, class histograms, probability row-sums, LST fidelity, all 15 hotspot products; stratified anchor 10,859/10,859 bit-identical | **PASS** (`verification_g2.json`, `overall_passed: true`, no failures) |
