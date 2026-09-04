# GreenGrid AI — Phase 6: UHI Severity Mapping

**Project:** GreenGrid AI  
**Phase:** 6 — UHI Severity Mapping  
**Study Area:** National Capital Territory (NCT) of Delhi, India  
**Report Date:** 2026-09-04  
**Status:** ✅ COMPLETED — Stages 1–5 executed successfully | Validation 147 PASS / 0 FAIL / 0 WARN (reproducibility run)

---

## 1. Objective

Phase 6 converts Phase 5's validated pixel-level classification into **decision-support analytics**. The goal is to answer, over the full valid satellite grid (not just the sampled training pixels): where are the heat-severity hotspots, how large are they, how do green and built-up areas differ in heat exposure, how do vegetation and urbanization relate to observed land-surface temperature (LST), and what changed between the July 2022 and July 2026 snapshot composites.

Phase 5 produced and validated a model; Phase 6 produces the **maps, delineated hotspot boundaries, area-wise statistics, group comparisons, relationship analyses, and uncertainty diagnostics** that make the model usable for tree-plantation planning.

**Scientific framing (unchanged from Phase 5):** all outputs describe **ML-based relative heat severity** — an **operational UHI hotspot proxy**, not physical UHI intensity. Severity classes are derived from per-year LST quartiles and reflect relative within-year ranking, not absolute temperature anomalies.

---

## 2. Relationship to Phase 5

- **Phase 5 is frozen.** No Phase 5 methodology, tables, rasters, or numbers were modified. Phase 6 consumes the frozen outputs (predictions, OOF predictions, spatial/morphology feature tables, frozen metrics).
- **Production model adoption.** Per the Phase 5 report §24 (Next Steps, item 4), the regularized **RF-C** configuration (`max_depth=15`, `min_samples_leaf=10`, all other parameters at the Phase 5 Random Forest baseline) is adopted as the Phase 6 production model. RF-C was retrained on all 300,000 sampled rows for full-grid prediction. The Phase 5 experiment results are unchanged by this adoption.
- **Frozen Phase 5 numbers restated for reference:**

| Result | Accuracy | Macro F1 |
|---|---|---|
| Random Forest baseline (no spatial context) | 42.65% | 0.3999 |
| Random Forest (spatial means + morphology) | 46.28% | 0.4306 |
| **RF-C (regularized, adopted for Phase 6)** | **46.99%** | **0.4310** |
| Locked geographic holdout (RF-C, unseen blocks 3/11/17) | 39.9% | 0.395 |

**The 60% accuracy target was NOT achieved.** The honest trustworthy ceiling is ~47% accuracy under adjacent-block spatial CV and ~40% on fully unseen geography. Phase 6 does not re-litigate this; all Phase 6 outputs inherit this accuracy ceiling.

---

## 3. Data Sources

| Source | What Phase 6 uses it for |
|---|---|
| `data/processed/phase4/tables/combined_urban_environmental_dataset.csv` (300,000 sampled rows, 14 columns; 150,000 per year) | Model training data for the RF-C production model |
| Phase 5 OOF tables (`oof_predictions.csv`, `spatial_neighbourhood_features.csv`, `morphology_features.csv`) | TRUE out-of-fold error diagnostics (never the in-sample `predictions.csv`) |
| Phase 3/4 full-grid rasters (valid mask, LST, NDVI, NDBI, vegetation cover, land use, distance layers) | Full-grid feature generation and continuous heat-intensity surfaces |

**Terminology note.** Throughout this report, "heat severity" means **ML-based relative heat severity**, and "hotspot" means an **operational UHI hotspot proxy**. Neither term denotes physical UHI intensity, which would require a defined rural baseline and air-temperature measurements that this project does not have.

---

## 4. Severity Methodology

**Full-grid prediction.** Phase 5 rasters covered only the ~4.5% of the grid represented by sampled training pixels. Phase 6 extends the RF-C model to the **full valid grid: 1,500,777 valid pixels per year** (EPSG:4326, 1,874 × 1,768 grid; masked pixels remain NoData and never participate in any statistic).

**Classes.** Identical to Phase 5: classes 0–3 (Low, Moderate, High, Severe) are **per-year LST quartiles**. Final descriptive thresholds:

| Year | Low→Moderate (°C) | Moderate→High (°C) | High→Severe (°C) |
|---|---|---|---|
| 2022 | 34.21 | 38.63 | 43.47 |
| 2026 | 32.94 | 34.47 | 36.20 |

The lower 2026 thresholds reflect the overall cooler July 2026 LST distribution.

**Feature generation.** Full-grid features replicate the Phase 5 machinery exactly: the block grid is derived from the same sampled row/col bins used in Phase 5, and the block-aware spatial-mean (3×3/5×5/11×11) and morphology (50/100/250/500 m) features are recomputed with the same code paths. Parity was verified at 10,000 sampled cells per year against the frozen Phase 5 feature tables: maximum absolute difference 0.0 for spatial features and ≤ 1.2 × 10⁻¹⁰ (2022) / ≤ 7.5 × 10⁻⁹ (2026) for morphology features (both within float32 tolerance; pipeline record: `phase5_feature_parity_at_sampled_cells`).

**Outputs per year:** severity class raster, four per-class probability rasters, a confidence raster (max class probability), and an ordinal severity-score raster (§5).

---

## 5. Heat-Intensity Methodology

For continuous heat exposure, Phase 6 uses the **observed LST rasters** (Phase 3 Landsat 9 composites), re-emitted on the Phase 6 grid. This is the most defensible continuous heat surface in the project: it is a direct satellite measurement, not a model output.

The model's `severity_score = 0·P(Low) + 1·P(Moderate) + 2·P(High) + 3·P(Severe)` is also provided as a raster, but it is an **ordinal model score, not a temperature** — it carries model error and uncalibrated probability mass, and it must not be read as °C or as a physical heat index.

**No physical UHI baseline is invented.** With no valid rural reference or air-temperature data, Phase 6 reports observed LST and relative severity only.

---

## 6. Hotspot Extraction Methodology

Hotspots are delineated from the full-grid severity rasters under three definitions:

| Definition | Membership rule | Role |
|---|---|---|
| **A (primary)** | severity ∈ {High, Severe} | Main decision-support product |
| **B** | severity = Severe only | Strictest |
| **C** | severity ∈ {High, Severe} **and** confidence ≥ 0.60 | Operational confidence filter |

Notes on the rules:

- **Minimum 10-pixel rule.** Connected clusters smaller than 10 pixels (~0.8 ha at 30 m) are removed as operational noise. This is a pragmatic noise-removal choice, not a physical minimum-patch-size claim.
- **8-connectivity.** Clusters use scipy.ndimage 8-connectivity (`structure=np.ones((3,3))`) — diagonal neighbours count.
- **Areas via EPSG:32643.** All areas are computed by reprojecting to UTM zone 43N. Degree coordinates are **not** equal-area; degree-based areas are never used. Mean valid pixel area is 787.05 m².
- Definition C's 0.60 threshold is an **operational filter on uncalibrated confidence** (§17), not a calibrated probability statement.

Sensitivity across definitions is reported in §13 (table reference: `hotspot_sensitivity.csv`) and definition A is used for all headline hotspot numbers.

---

## 7. Boundary Methodology

Hotspot boundaries are produced as GeoJSON polygons:

1. Raster → polygon conversion per hotspot cluster.
2. **Connectivity = 4 for polygon shapes, dissolved per cluster.** Because a polygonised 4-connected shape can split a diagonally-connected (8-connected) cluster into two touching shapes, the per-cluster shapes are dissolved so the boundary exactly preserves the 8-connected cluster semantics of §6.
3. Validation per year/definition: **0 invalid geometries, 0 zero-area polygons**, polygon-vs-pixel area accounting within **≤ 0.23%** relative error (tolerance 5%), no duplicate hotspot IDs, minimum cluster size 10 px confirmed.

Validation result: all six year × definition combinations pass (recorded in `phase6_validation_report.json` and `phase6_pipeline_record.json`).

---

## 8. Area Statistics Methodology

Area-wise statistics are computed on **spatial blocks**: the 5 × 5 grid from the Phase 5 sampling design, of which **20 of 25 blocks are occupied**. These are **sampling-design blocks, explicitly NOT administrative sectors**: no valid administrative boundaries exist in the project (the districts file collected in Phase 2 is unusable for analysis). Block-level results (`area_statistics_*.csv`) must not be labelled by district or any administrative name.

All areas are **per-pixel geodesic areas**: each pixel's four corners are transformed to EPSG:32643 and the exact polygon area (shoelace) is summed over unit cells. Total valid area is 118,118.923 ha (~1,181 km²) per year.

---

## 9. Green vs Built-Up Methodology

The green/built contrast uses an **operational classification** on the observed NDVI/NDBI rasters:

| Group | Rule |
|---|---|
| green_dominant | NDVI ≥ 0.3 |
| built_dominant | NDBI ≥ 0.1 **and** NDVI < 0.3 |
| other_mixed | everything else |

These are **conventional remote-sensing thresholds, not site-calibrated**. `landuse_class` (OSM) is reported **separately** (§16) and is not merged into this classification. All green/built contrasts are **associations, not causal effects**.

---

## 10. 2022 Results

Headline 2022 numbers (valid area 118,118.923 ha; mean observed LST **38.76 °C**, median 38.65 °C):

| Quantity | 2022 |
|---|---|
| Class distribution | Low 24.26% · Moderate 21.79% · High 21.69% · Severe 32.26% |
| High + Severe (class-based) | 809,752 px = **53.96%**, 63,708.5 ha |
| DefA hotspots (after 10-px rule) | **580 hotspots**, 804,862 px, **63,323.5 ha (53.63% of valid area)** |
| Largest DefA hotspot | 46,070.8 ha (hotspot 6; mean LST 43.08 °C) |
| Mean NDVI / NDBI / veg cover | 0.335 / −0.084 / 0.382 |
| Mean confidence | 0.496 |

Green vs built-up 2022: green_dominant 772,625 px (60,817.0 ha; mean LST 37.02 °C; High+Severe 34.96%); built_dominant 53,541 px (4,213.1 ha; mean LST **44.23 °C**; High+Severe **94.08%**); other_mixed 674,611 px (40.33 °C mean LST). Built-minus-green mean LST gap: **+7.22 °C**.

---

## 11. 2026 Results

Headline 2026 numbers (mean observed LST **34.94 °C**, median 34.47 °C):

| Quantity | 2026 |
|---|---|
| Class distribution | Low 23.63% · Moderate 23.29% · High 24.84% · Severe 28.25% |
| High + Severe (class-based) | 796,642 px = **53.08%**, 62,693.9 ha |
| DefA hotspots (after 10-px rule) | **495 hotspots**, 793,680 px, **62,460.7 ha (52.88% of valid area)** |
| Largest DefA hotspot | 49,000.6 ha (hotspot 1; mean LST 36.85 °C) |
| Mean NDVI / NDBI / veg cover | 0.341 / −0.120 / 0.392 |
| Mean confidence | 0.556 |

Green vs built-up 2026: green_dominant 777,356 px (34.30 °C; High+Severe 31.00%); built_dominant 5,532 px (435.4 ha; mean LST **36.93 °C**; High+Severe **81.36%**); other_mixed 717,889 px (35.62 °C). Built-minus-green mean LST gap: **+2.63 °C**.

---

## 12. Temporal Comparison (2022 vs 2026)

Side-by-side (from `temporal_comparison.csv`):

| Metric | 2022 | 2026 |
|---|---|---|
| Mean LST (°C) | 38.76 | 34.94 |
| Median LST (°C) | 38.65 | 34.47 |
| High+Severe % | 53.96% | 53.08% |
| High+Severe area (ha) | 63,708.5 | 62,693.9 |
| DefA hotspot count | 580 | 495 |
| DefA hotspot area (ha) | 63,323.5 | 62,460.7 |
| Largest hotspot (ha) | 46,070.8 | 49,000.6 |
| Mean confidence | 0.496 | 0.556 |

**Caveats (stated in every output table):**

- 2022 and 2026 are **two snapshot composite dates, NOT a long-term trend**. There is no monitored time series between them.
- Because classes are per-year quartiles, cross-year severity is **class-relative**: a 2026 Severe pixel can be cooler in °C than a 2022 Moderate one. The 2026 LST distribution is shifted cooler (lower mean/median and lower per-year quartile thresholds), but the two distributions overlap substantially in °C (2022 range 11.58–55.28 °C vs 2026 range 16.16–53.05 °C); per-year quartile classes mean cross-year severity comparisons remain class-relative.
- The ~53–54% High+Severe share is largely **built into the class design**: High and Severe are the top two of four balanced per-year quartile classes (≈ 50% of valid pixels by construction, before the 10-px hotspot noise rule), so the headline share is not an independently measured exposure rate.
- The observation that mean LST is lower in the July 2026 composite than in the July 2022 composite is a **data observation about two images**, not a climate claim about Delhi's trajectory. It may reflect atmospheric conditions, cloud-masking differences, and acquisition differences as much as any surface change.

---

## 13. Hotspot Persistence and Change

Paired-cell analysis over the 1,500,777 cells valid in both years (definition-A masks; `hotspot_change_statistics.csv`, `hotspot_persistence.csv`):

| Category | Pixels | % of paired | Area (ha) |
|---|---|---|---|
| Persistent hotspot (hot in both years) | 567,810 | 37.83% | 44,680.2 |
| New hotspot 2026 | 225,870 | 15.05% | 17,780.5 |
| Disappeared hotspot (2022 only) | 237,052 | 15.80% | 18,643.2 |
| Stable non-hotspot | 470,045 | 31.32% | 37,015.0 |
| **DefA mask IoU (2022 vs 2026)** | | **0.551** | |

Class-based (High+Severe) persistence: **P(HS₂₀₂₆ | HS₂₀₂₂) = 0.707**, **P(HS₂₀₂₂ | HS₂₀₂₆) = 0.719**; marginal shares 53.96% (2022) and 53.08% (2026). Because classes are per-year quartiles, this is class-relative persistence, not a physical trend.

Largest flows in the severity transition matrix (`temporal_transition_matrix.csv`): the diagonal dominates — Low→Low 155,060 px (42.6% of 2022 Low), Moderate→Moderate 141,404 (43.2%), High→High 135,065 (41.5%), Severe→Severe 266,075 (55.0%). Largest off-diagonal flows: Low→Moderate 131,177; Severe→High 91,420; Moderate→High 88,123; Severe→Low 82,376; High→Severe 79,905. These are consistent with per-year quartile re-ranking under a cooler 2026 distribution.

Definition sensitivity (`hotspot_sensitivity.csv`): in 2022, definition B (Severe only) covers 59.7% of definition A (IoU 0.597); definition C covers 25.6% of A (IoU 0.256). In 2026: B covers 52.9% of A (IoU 0.529); C covers 32.5% of A (IoU 0.325). All analyses here are **descriptive only**.

---

## 14. Vegetation–Temperature Relationship

Correlations (seed-42 subsample, n = 200,000 valid pixels per year; `vegetation_temperature_relationship.csv`):

| Relationship | 2022 Pearson r | 2022 Spearman r | 2026 Pearson r | 2026 Spearman r |
|---|---|---|---|---|
| NDVI vs LST | −0.368 | −0.357 | −0.286 | −0.361 |
| vegetation_cover vs LST | −0.369 | −0.357 | −0.288 | −0.361 |
| NDVI vs vegetation_cover | 0.9995 | 0.9999 | 0.9978 | 0.9996 |

Decile profiles (all valid pixels, per-year NDVI deciles): 2022 mean LST falls monotonically-ish from 40.66 °C (decile 1) to 34.61 °C (decile 10), with High+Severe share falling from 80.3% to 17.5%; 2026 falls from 35.97 °C to 33.18 °C, High+Severe from 88.3% to 10.5%.

**Two mandatory cautions:**

- **Redundancy:** NDVI and vegetation_cover are near-duplicates (r ≈ 0.998–0.9995). Their correlations with LST are **not independent evidence**; effectively one relationship is being measured twice.
- **Spatial autocorrelation:** 30 m pixels are strongly spatially autocorrelated; 300,000 (or 200,000) pixels are **not** 300,000 independent observations. The stated n overstates the effective sample size, so correlation magnitudes are optimistic in precision terms.

Interpretation: more vegetated areas are consistently and substantially cooler in observed LST, in both years — an association, not a measured causal cooling effect.

---

## 15. Urbanization–Temperature Relationship

NDBI–LST correlations (same subsample): 2022 Pearson r = **+0.422**, Spearman r = +0.424; 2026 Pearson r = **+0.336**, Spearman r = +0.392 (`vegetation_temperature_relationship.csv` / `urbanization_heat_relationship.csv`).

Decile profile (per-year NDBI deciles, all valid pixels): mean LST rises steadily with NDBI decile in both years — 2022 from 35.27 °C (decile 1) to 43.18 °C (decile 10), with High+Severe share rising from 25.8% to 90.8%; 2026 from 33.17 °C to 36.57 °C, High+Severe from 14.8% to 86.1%. Mean severity_score rises monotonically across deciles in both years. The relationship is monotonic, strong, and consistent across both snapshots — again an association, not a causal claim.

---

## 16. Land-Use Analysis

Mean LST and High+Severe share per OSM land-use class (`landuse_heat_statistics_*.csv`):

| Class | 2022 px (%) | 2022 mean LST (°C) | 2022 High+Severe % | 2026 mean LST (°C) | 2026 High+Severe % |
|---|---|---|---|---|---|
| unclassified_background | 1,223,885 (81.55%) | 38.68 | 54.04% | 34.91 | 52.36% |
| forest | 14,590 (0.97%) | 34.39 | 8.40% | 33.82 | 24.59% |
| commercial | 11,907 (0.79%) | 36.77 | 23.58% | 35.16 | 44.19% |
| industrial | 47,735 (3.18%) | 40.67 | 67.50% | 35.94 | 69.15% |
| residential | 134,612 (8.97%) | 39.32 | 53.51% | 34.95 | 61.38% |
| retail | 2,822 (0.19%) | 36.62 | 36.29% | 35.86 | 79.31% |
| farmland | 65,226 (4.35%) | 39.26 | 59.89% | 34.84 | 44.56% |

Caveats:

- **81.55% of valid pixels are unclassified_background** — the OSM land-use layer covers a small minority of the city, so class-level statistics rest on limited, spatially uneven coverage.
- **park and grass classes are entirely absent from the samples** (zero pixels in both years), so no statement can be made about them.
- The OSM snapshot is **static**: the same land-use raster is used for 2022 and 2026, so no land-use change analysis is possible.

---

## 17. Confidence and Uncertainty

**Confidence definition.** Confidence = max class probability. Per the Phase 5 calibration report, probabilities are reasonably but not perfectly calibrated (multiclass Brier 0.162, ECE ≈ 0.045 on development OOF); confidence is a **confidence proxy, not a calibrated probability**.

**Uncertainty zones.** A cell is flagged uncertain if top-2 margin < 0.10 **or** confidence < 0.50 (operational thresholds, not calibrated cut-offs):

| Year | Uncertain cells | Fraction of valid |
|---|---|---|
| 2022 | 916,269 | **61.1%** |
| 2026 | 605,866 | **40.4%** |

Uncertain cells also concentrate on class boundaries: 34.3% (2022) / 39.3% (2026) of uncertain cells touch a class boundary vs ~1.0% / 4.6% of certain cells.

**OOF error diagnostics** (TRUE out-of-fold predictions of the frozen Phase 5 baseline RF at the 300k sampled cells — not the Phase 6 RF-C model, and never the in-sample `predictions.csv`):

- OOF overall accuracy: 43.12% (2022), 49.25% (2026) sampled cells.
- **Error rises near quartile boundaries:** OOF accuracy in the ±0.5 °C near-boundary bands is ~36–48% (2022: 37.5% and 36.2%; 2026: 44.8% and 47.6%) versus ~55–56% for cells ≥ 2 °C from the nearest threshold (2022: 56.3%; 2026: 55.1%). Roughly speaking, a large share of model error is inherent label noise from the quartile target, not fixable model weakness.
- **Moderate is the weakest class** (2022 recall 26.1%; 2026 recall 43.6% — lowest or near-lowest per year), consistent with Phase 5.
- Per-block error varies substantially (`error_by_block.csv`); Phase 5 already flagged block 10's local threshold imbalance.

**Diagnostic-provenance note.** An early draft of the Phase 6 error diagnostics was computed from `predictions.csv`, which Phase 5 writes as the production model's **in-sample** predictions at the sampled cells (accuracy 0.9957, matching the frozen baseline RF's ~99.5% train accuracy). This was detected because it contradicted the frozen Phase 5 out-of-fold results, and the diagnostics were recomputed from the true out-of-fold rows in `oof_predictions.csv` (model "Random Forest", pooled accuracy 0.4619, consistent with frozen Phase 5). A validator guard now fails loudly if the in-sample table is reintroduced: the pooled OOF accuracy must fall in [0.42, 0.50].

---

## 18. Spatial Limitations

- **Blocks are sampling-design units, not places.** The 20 occupied 5 × 5 blocks carry no administrative meaning; they exist because Phase 4 sampled row/col bins.
- **Full-grid accuracy is unvalidated.** The honest accuracy ceiling is the frozen Phase 5 locked holdout (~39.9% / 0.395) for unseen geography; ~47% / 0.43 for adjacent-block CV. No new ground truth was created in Phase 6, so full-grid per-pixel accuracy cannot be claimed to exceed these numbers. All full-grid statistics inherit this ceiling.
- **Building-distance features come from an OSM Central Delhi building sample** (a zero-distance pixel proxy, not measured footprint-area fractions), treated as identical for both years.
- **OSM context is static** for 2022 and 2026 (roads, vegetation, land use, buildings), so no analysis here reflects 2022→2026 infrastructure change.
- **Spatially structured model error (frozen Phase 5).** Moran's I on the frozen baseline RF's error fields is strongly positive: classification-error Moran's I ≈ 0.53–0.54 (2022: 0.5434; 2026: 0.5321) and absolute-class-distance-error Moran's I ≈ 0.61–0.70 (2022: 0.6141; 2026: 0.7046), all with permutation p = 0.005 (`data/processed/phase5/tables/spatial_error_autocorrelation.csv`). Apparent hotspot/cluster structure in the Phase 6 maps could therefore partly reflect spatially structured model error rather than true surface pattern, and pixel-level statistics overstate the effective sample size.

---

## 19. Scientific Interpretation

The patterns Phase 6 shows are internally coherent and physically plausible:

- **Severity is spatially structured, not noise.** Hotspots form large contiguous regions (the largest ~46,000–49,000 ha) rather than speckle; the Phase 5-style agreement between the Phase 6 full-grid map and Phase 5 sampled predictions is 73.4% (2022) / 75.8% (2026).
- **Urban form tracks heat.** Built-dominant cells are far hotter than green-dominant cells (+7.22 °C in 2022, +2.63 °C in 2026), NDBI rises monotonically with LST across deciles, and industrial/residential land-use classes carry the highest High+Severe shares. This plausibly reflects impervious surface cover, reduced evaporative cooling, and anthropogenic heat — but Phase 6 measures **association only**.
- **Vegetated areas are cooler.** NDVI/vegetation_cover deciles show a strong, monotonic inverse relationship with LST in both years.
- **2022 is the hotter snapshot in absolute °C.** The 2026 LST distribution is shifted cooler (lower mean and median, and lower quartile thresholds), while the two distributions overlap substantially in °C (2022 range 11.58–55.28 °C vs 2026 range 16.16–53.05 °C; means 38.76 vs 34.94 °C), consistent with the threshold tables. This is a two-image observation, not a trend.
- **Hotspots are moderately persistent** (mask IoU 0.551; class-conditional persistence ~0.71), consistent with stable urban structure being the dominant control.

---

## 20. Limitations

1. **Proxy, not physical UHI:** outputs are ML-based relative heat severity / an operational hotspot proxy, not physical UHI intensity (no rural baseline, no air temperature).
2. **Two snapshots only:** 2022 and 2026 are composite dates; no trend, seasonality, or interannual variability can be assessed.
3. **Uncalibrated probabilities:** confidence is a proxy (Brier 0.162, ECE ≈ 0.045); the 0.60 definition-C cut-off is operational.
4. **Unvalidated full-grid accuracy:** the honest ceiling is ~47% (adjacent-block CV) / ~40% (locked holdout); no Phase 6 ground truth exists.
5. **Static OSM context:** roads, vegetation, buildings, and land use are a 2026 snapshot applied to both years.
6. **Sparse land-use coverage:** 81.55% unclassified_background; park/grass absent; class statistics are unevenly supported.
7. **Building sample:** building-distance features are a Central Delhi sample and a zero-distance pixel proxy.
8. **Block-edge kernel clipping:** block-aware morphology kernels are truncated near block boundaries (up to 15.8% of sampled pixels affected at 500 m) — a conservative leakage-control trade-off.
9. **NDVI/PVC redundancy:** vegetation_cover is a near-duplicate of NDVI (r ≈ 0.998–0.9995); their agreement is not independent corroboration.
10. **Class-relative cross-year comparability:** per-year quartile classes mean a 2026 Severe pixel can be cooler in °C than a 2022 Moderate one; cross-year statements are class-level only.
11. **Spatial autocorrelation:** correlation n overstates effective n — frozen Phase 5 found RF error-field Moran's I ≈ 0.53–0.54 (classification error) and ≈ 0.61–0.70 (class-distance error), p = 0.005, so apparent cluster structure may partly reflect spatially structured model error; no causal inference is supported anywhere in Phase 6.
12. **Operational thresholds:** the 10-px minimum hotspot rule and the NDVI/NDBI green/built thresholds are conventional operational choices, not site-calibrated science.
13. **Blocks are not administrative units:** area statistics cannot be reported per district/ward with the data currently in the project.

---

## 21. Conclusions

**What Phase 6 demonstrates:**

- Complete **full-grid maps of relative heat severity** for Delhi NCT for July 2022 and July 2026 (1,500,777 valid px/yr), built on the frozen, audited Phase 5 model with exact feature-generation parity.
- **Coherent, delineated hotspot boundaries** (definition A: 580 hotspots / 63,323 ha in 2022; 495 / 62,461 ha in 2026) with validated geometries and pixel-consistent areas.
- **Area-level (spatial-block) heat statistics** with exact geodesic areas.
- **Quantified high-risk exposure:** ~53–54% of the valid area is High+Severe in both snapshots.
- **Consistent green vs built-up differences:** built-dominant areas are 2.6–7.2 °C hotter in mean LST and carry 81–94% High+Severe shares.
- **Stable NDVI–LST (negative) and NDBI–LST (positive) relationships**, monotonic across deciles in both years.
- **Hotspot persistence and 2022↔2026 snapshot differences**, with explicit class-relativity caveats.
- **Uncertainty and error diagnostics** showing where (near quartile boundaries) and how much (61.1% / 40.4% of cells) the map should be trusted cautiously.

**What Phase 6 does NOT demonstrate:**

- Physical UHI intensity (no baseline, no air temperature).
- A causal vegetation or urbanization effect (association only).
- Long-term trends or climate trajectories (two snapshots).
- City-wide longitudinal change (static OSM context, two images).
- Exact physical boundaries of heat zones (30 m raster model output with ~40–47% class accuracy ceiling).
- Universal generalization beyond this grid, these dates, and this feature set.

---

## 22. Future Work

1. **Alternative targets** (Phase 5 audit's top recommendation): binary hotspot detection, 3-class, ordinal classification, or LST regression → quartiles — the quartile target's boundary noise (~28% of pixels within 0.5 °C of a threshold) is the dominant accuracy bottleneck.
2. **New feature families:** NDWI / water presence, impervious-surface fraction, surface albedo.
3. **Per-block recalibration** (flagged for block 10 in Phase 5 LOBO) to recover local hotspot recall.
4. **Ensembles:** combine RF and XGBoost probabilities (seed-stable, decorrelated errors).
5. **Administrative-boundary acquisition:** obtain usable district/ward boundaries so area statistics can be reported for real planning units instead of sampling-design blocks.
6. **Plantation-priority scoring (Phase 7 link):** combine hotspot severity, green/built contrast, and uncertainty layers into a tree-plantation suitability and prioritization product.

---

## Appendix — Key Figures

All figures are in `data/processed/phase6/figures/` (220 dpi; severity palette Low #2c7bb6 / Moderate #abd9e9 / High #fdae61 / Severe #d7191c; explicit NoData patch).

| Figure | Caption |
|---|---|
| `fig01_lst_2022.png` | Observed Landsat 9 LST, July 2022 composite (continuous heat surface) |
| `fig02_lst_2026.png` | Observed Landsat 9 LST, July 2026 composite (continuous heat surface) |
| `fig03_severity_2022.png` | Full-grid RF-C relative heat-severity map, 2022 (Low–Severe) |
| `fig04_severity_2026.png` | Full-grid RF-C relative heat-severity map, 2026 (Low–Severe) |
| `fig05_hotspots_2022.png` | Definition-A hotspot boundaries over severity, 2022 |
| `fig06_hotspots_2026.png` | Definition-A hotspot boundaries over severity, 2026 |
| `fig07_temporal_comparison.png` | 2022 vs 2026 side-by-side severity/hotspot comparison (snapshots, not a trend) |
| `fig08_green_vs_built.png` | Green vs built-dominant classification and mean LST contrast |
| `fig09_ndvi_vs_lst.png` | NDVI–LST relationship (subsampled scatter + decile profile) |
| `fig10_ndbi_vs_lst.png` | NDBI–LST relationship (subsampled scatter + decile profile) |
| `fig11_high_severe_by_block.png` | High+Severe share per spatial block (sampling-design blocks, not administrative) |
| `fig12_hotspot_size_distribution.png` | Definition-A hotspot size (area) distribution per year |
| `fig13_confidence_map.png` | Full-grid model confidence (max class probability) maps |
| `fig14_uncertainty.png` | Uncertainty zones (low top-2 margin / low confidence) per year |
| `fig15_transition_matrix.png` | 2022→2026 severity class transition matrix heatmap |
| `fig16_landuse_heat.png` | Mean LST / High+Severe share per OSM land-use class |

---

## Reproducibility

- Random seed: 42 (all stochastic components seeded).
- Stage commands (from the project root, venv active):
  ```bash
  PYTHONPATH=src .venv/bin/python -m severity.pipeline [--skip-model]
  PYTHONPATH=src .venv/bin/python -m severity.hotspots
  PYTHONPATH=src .venv/bin/python -m severity.analytics
  PYTHONPATH=src .venv/bin/python -m severity.figures
  PYTHONPATH=src .venv/bin/python -m severity.validate_severity [--repro]
  ```
- `severity.validate_severity --repro` re-runs the analytics stage and confirms bit-identical tables, then reports **147 PASS / 0 FAIL / 0 WARN**, including Phase 5 frozen-number integrity checks.
- Software: Python 3.13.3, numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, rasterio 1.5.1, scipy 1.18.1, shapely 2.1.2, geopandas 1.1.4, matplotlib 3.11.1 (recorded in `phase6_pipeline_record.json`).

---

*GreenGrid AI — Phase 6 UHI Severity Mapping Report | Generated 2026-09-04*
