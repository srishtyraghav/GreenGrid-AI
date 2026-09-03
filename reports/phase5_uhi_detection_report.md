# GreenGrid AI — Phase 5: AI-Based UHI Detection & Classification

**Project:** GreenGrid AI  
**Phase:** 5 — AI-Based UHI Detection & Classification  
**Study Area:** National Capital Territory (NCT) of Delhi, India  
**Report Date:** 2026-09-02  
**Status:** ✅ COMPLETED — Pipeline executed successfully | Validation 37 PASS / 0 FAIL / 0 WARN
**Extension:** Urban morphology experiment added (64 block-aware morphology features) with full OOF diagnostics

---

## 1. Objective

Phase 5 implements the project’s first genuine machine-learning component. The goal is to classify 30 m pixels into **relative LST heat-severity categories** using environmental and land-surface predictors derived in Phase 4.

The four output classes are:

| Class | Operational meaning |
|---|---|
| Low | Relatively cool within the sampled distribution |
| Moderate | Moderately warm within the sampled distribution |
| High | Hot within the sampled distribution |
| Severe | Hottest within the sampled distribution |

Two models are compared:

- **Random Forest**
- **XGBoost**

The final system produces AI-generated relative heat-severity maps for Delhi NCT for both 2022 and 2026.

**Important scientific framing:** The classes are derived from Landsat 9 LST. They are an **operational proxy for UHI hotspot detection**, not independently observed physical UHI ground-truth labels. This report therefore describes the work as **ML-based relative heat-severity classification** rather than claiming independent measurement of atmospheric UHI intensity.

---

## 2. Relationship to Phase 4

- **Phase 4** assembled the combined urban environmental dataset with `lst_C`, `ndvi`, `ndbi`, `vegetation_cover`, land-use class, and distance layers, plus `spatial_block_id` for spatially aware validation.
- **Phase 5** uses `lst_C` to define a relative heat-severity target and trains classifiers on the remaining features.
- Phase 5 also adds **block-aware spatial neighbourhood features** (3×3, 5×5, 11×11 focal means) to give the model urban spatial context.

No Phase 3 or Phase 4 methodology was modified.

---

## 3. Dataset

| Attribute | Value |
|---|---|
| Input | `data/processed/phase4/tables/combined_urban_environmental_dataset.csv` |
| Rows | 300,000 |
| Columns (Phase 4) | 14 |
| Years | 2022 (150,000), 2026 (150,000) |
| Spatial blocks | 20 occupied blocks (IDs 1–24) from a 5 × 5 grid |
| Added spatial-mean features | 18 neighbourhood-mean features |
| Added morphology features | 64 block-aware urban-morphology features |
| Final columns | 96 |

The spatial and morphology features are described in Section 6.

---

## 4. Target Definition

### 4.1 Methodology

Because the 2022 and 2026 LST distributions differ substantially, the target is defined **per year** using temperature quartiles:

| Class | Definition |
|---|---|
| Low | LST ≤ 25th percentile of that year |
| Moderate | 25th percentile < LST ≤ 50th percentile |
| High | 50th percentile < LST ≤ 75th percentile |
| Severe | LST > 75th percentile |

### 4.2 Final descriptive thresholds

| Year | Low→Moderate (°C) | Moderate→High (°C) | High→Severe (°C) |
|---|---|---|---|
| 2022 | 34.21 | 38.63 | 43.47 |
| 2026 | 32.94 | 34.47 | 36.20 |

The lower 2026 thresholds reflect the overall cooler July 2026 LST distribution.

### 4.3 Leakage prevention during cross-validation

For every spatial CV fold, quartile thresholds are computed using **only the training-block rows** of the relevant year. The same thresholds are then applied to the held-out validation rows. Validation rows never influence their own target thresholds.

---

## 5. Leakage Prevention

The following variables are explicitly excluded from the predictor matrix:

| Variable | Reason |
|---|---|
| `lst_C` | Target variable |
| `spatial_block_id` | Grouping variable for spatial CV |
| `lon`, `lat`, `row`, `col` | Identifiers / spatial references |

A programmatic assertion verifies that none of these variables enter `PREDICTOR_VARS`.

---

## 6. Predictor Selection

### 6.1 Base predictors

| Feature | Type | Notes |
|---|---|---|
| `ndvi` | Continuous | Sentinel-2 NDVI |
| `ndbi` | Continuous | Sentinel-2 NDBI |
| `vegetation_cover` | Continuous | Proportional vegetation cover derived from NDVI |
| `landuse_class` | Categorical | One-hot encoded; 0 = unclassified/background |
| `dist_road_m` | Continuous | Distance to nearest major road |
| `dist_vegetation_m` | Continuous | Distance to nearest OSM green area |
| `dist_building_m` | Continuous | Distance to nearest building in Central Delhi sample |
| `year` | Integer | 2022 or 2026 |

### 6.2 Spatial neighbourhood features

Block-aware focal means are computed at 3×3, 5×5, and 11×11 scales for the six continuous base predictors, adding 18 features:

- `ndvi_mean3`, `ndvi_mean5`, `ndvi_mean11`
- `ndbi_mean3`, `ndbi_mean5`, `ndbi_mean11`
- `vegetation_cover_mean3`, `vegetation_cover_mean5`, `vegetation_cover_mean11`
- `dist_road_m_mean3`, `dist_road_m_mean5`, `dist_road_m_mean11`
- `dist_vegetation_m_mean3`, `dist_vegetation_m_mean5`, `dist_vegetation_m_mean11`
- `dist_building_m_mean3`, `dist_building_m_mean5`, `dist_building_m_mean11`

**Block-aware computation:** For each spatial block, the focal mean is computed using only pixels inside that block. This prevents a validation pixel’s neighbourhood feature from incorporating training-block values.

### 6.3 NDVI / Vegetation Cover redundancy

`ndvi` and `vegetation_cover` are nearly perfectly correlated (`r ≈ 0.999`). Both are retained in the primary model because the project synopsis explicitly includes vegetation cover, but an ablation experiment shows whether PVC adds predictive information beyond NDVI.

### 6.4 Urban morphology features (Phase 5 extension)

To give the model a richer representation of the urban environment, 64 block-aware morphology features were derived from existing Phase 3/4 rasters only (no new external data, no LST). All features use circular neighbourhoods at radii of 50 m, 100 m, 250 m, and 500 m (~30 m pixels).

| Feature family | Definition | Source |
|---|---|---|
| `building_pixel_fraction_<R>m` | Fraction of neighbourhood pixels with `dist_building_m == 0` | `buildings_distance_30m.tif` |
| `road_pixel_fraction_<R>m` | Fraction of neighbourhood pixels with `dist_road_m == 0` | `roads_distance_30m.tif` |
| `vegetation_pixel_fraction_<R>m` | Fraction of neighbourhood pixels with `dist_vegetation_m == 0` | `vegetation_distance_30m.tif` |
| `vegetation_cover_mean_<R>m` | Mean proportional vegetation cover in the neighbourhood | `vegetation_cover_*.tif` |
| `landuse_frac_<class>_<R>m` | Fraction of neighbourhood pixels in each OSM land-use class | `landuse_raster_30m.tif` |
| `landuse_dominant_<R>m` | Dominant land-use class in the neighbourhood (one-hot encoded) | `landuse_raster_30m.tif` |
| `landuse_entropy_<R>m` | Shannon entropy of the neighbourhood land-use mixture | `landuse_raster_30m.tif` |
| `ndvi_contrast_<R>m` | Pixel NDVI minus neighbourhood-mean NDVI | `s2_*_ndvi_30m.tif` |
| `ndbi_contrast_<R>m` | Pixel NDBI minus neighbourhood-mean NDBI | `s2_*_ndbi_30m.tif` |

**Verified proxy semantics.** The Phase 3 distance rasters were generated by rasterizing features as value 1 and background as 0, then applying `distance_transform_edt(binary == 0)`. Pixels with distance 0 are therefore exactly the feature pixels. The `building_pixel_fraction` variables are a pixel-count proxy, not measured building-footprint area fractions.

**Block-aware computation and edge truncation.** As with the spatial-mean features, every neighbourhood is restricted to the pixel's own spatial block. This prevents cross-block leakage but clips kernels near block boundaries. The number of affected pixels grows with radius, as expected:

| Radius | Clipped pixels (full grid) | Clipped sampled pixels | Share of sampled |
|---|---|---|---|
| 50 m | 58,016 | 5,692 | 1.9% |
| 100 m | 86,832 | 8,738 | 2.9% |
| 250 m | 228,992 | 22,832 | 7.6% |
| 500 m | 476,816 | 47,272 | 15.8% |

The 500 m features are the most affected by this conservative leakage-control choice and should be interpreted with that caveat.

---

## 7. Spatial Validation

- **Strategy:** GroupKFold on `spatial_block_id`.
- **Folds:** 5.
- **Guarantee:** Training and validation blocks are disjoint in every fold. This is asserted programmatically.
- **Geographic composition:** The 20 occupied blocks form a regular 5 × 5 grid over Delhi NCT. Fold composition is recorded in the pipeline output.

---

## 8. Temporal Validation

Two temporal generalization experiments are performed:

- Train on 2022 → test on 2026
- Train on 2026 → test on 2022

Destination-year labels are constructed using the destination year’s own independently derived thresholds. The model never receives destination-year target information during training.

---

## 9. Models

### 9.1 Random Forest

```text
n_estimators=200
max_depth=None
min_samples_split=5
min_samples_leaf=2
max_features='sqrt'
class_weight='balanced'
random_state=42
```

### 9.2 XGBoost

```text
n_estimators=200
learning_rate=0.05
max_depth=6
subsample=0.8
colsample_bytree=0.8
objective='multi:softprob'
eval_metric='mlogloss'
random_state=42
```

Hyperparameters are fixed and documented for this baseline. No broad hyperparameter search was performed.

---

## 10. Evaluation Metrics

Reported metrics include:

- Accuracy
- Balanced accuracy
- Macro precision, recall, F1
- Weighted precision, recall, F1
- Per-class precision, recall, F1
- Confusion matrix

All reported spatial-CV metrics are computed from **out-of-fold predictions**. Every fold's OOF predictions (with identifiers, fold, model, actual/predicted class, class probabilities, and confidence) are saved to `data/processed/phase5/tables/oof_predictions.csv` for auditability.

---

## 11. Baseline Models

Three baselines are evaluated under the same spatial folds. The rule baseline uses a fixed, predetermined NDVI/NDBI decision rule (training-block thresholds only, no validation fitting):

1. Clip NDVI and NDBI to their training-block 5th/95th percentiles and min-max normalize.
2. `heat_score = normalized_ndbi - normalized_ndvi`.
3. Derive `heat_score` q25/q50/q75 thresholds from training blocks and assign the same four quartile classes.

| Baseline | Mean Accuracy | Mean Macro F1 | High Recall | Severe Recall |
|---|---|---|---|---|
| Majority class | 24.76% | 0.0984 | 0.0000 | 0.0000 |
| Stratified random | 24.97% | 0.2442 | 0.2508 | 0.2495 |
| NDVI/NDBI rule baseline | 36.33% | 0.3547 | 0.2801 | 0.4247 |
| **Random Forest (full features)** | **46.28%** | **0.4306** | **0.4017** | **0.5104** |
| **XGBoost (full features)** | **46.29%** | **0.4266** | **0.3811** | **0.5308** |

The ML models outperform the rule baseline by ~10 percentage points of accuracy and ~0.07 macro F1, confirming that the multivariate models capture structure beyond simple NDVI/NDBI thresholds.

---

## 12. Model Comparison

### 12.1 Base predictors only (no spatial context)

| Model | Mean Accuracy | Balanced Accuracy | Macro F1 | High Recall | Severe Recall |
|---|---|---|---|---|---|
| Random Forest | 42.65% | 0.4082 | 0.3999 | 0.3692 | 0.4720 |
| XGBoost | 42.66% | 0.4095 | 0.3965 | 0.3383 | 0.4882 |

### 12.2 Spatial means only (previous frozen benchmark)

| Model | Mean Accuracy | Balanced Accuracy | Macro F1 | High Recall | Severe Recall |
|---|---|---|---|---|---|
| Random Forest | 44.99% | 0.4302 | 0.4232 | 0.4008 | 0.4867 |
| XGBoost | 45.32% | 0.4323 | 0.4215 | 0.3750 | 0.5265 |

### 12.3 Spatial means + urban morphology (current full model)

| Model | Mean Accuracy | Balanced Accuracy | Macro F1 | High Recall | Severe Recall |
|---|---|---|---|---|---|
| Random Forest | 46.28% | 0.4376 | 0.4306 | 0.4017 | 0.5104 |
| XGBoost | 46.29% | 0.4375 | 0.4266 | 0.3811 | 0.5308 |

### 12.4 Improvement over the frozen spatial-means benchmark

| Model | Δ Accuracy | Δ Macro F1 | Δ High Recall | Δ Severe Recall |
|---|---|---|---|---|
| Random Forest | +1.29 pp | +0.0074 | +0.0009 | +0.0237 |
| XGBoost | +0.97 pp | +0.0051 | +0.0061 | +0.0043 |

The urban morphology features produce a **modest** improvement over the frozen spatial-means benchmark: about +1 percentage point of accuracy and +0.005–0.007 macro F1. The gain is consistent across both models but is not a dramatic jump, suggesting the existing predictors already capture much of the accessible signal and that further large gains will likely require new information (e.g., water/impervious-surface layers), alternative target formulations, or controlled model tuning rather than more morphology variants alone.

---

## 13. Feature Ablation

Controlled comparison of the morphology representation under identical folds (full table in `ablation_results.csv`):

| Experiment | Model | Mean Macro F1 | Mean High Recall | Mean Severe Recall |
|---|---|---|---|---|
| No spatial context | Random Forest | 0.3999 | 0.3692 | 0.4720 |
| No spatial context | XGBoost | 0.3965 | 0.3383 | 0.4882 |
| Spatial means only | Random Forest | 0.4232 | 0.4008 | 0.4867 |
| Spatial means only | XGBoost | 0.4215 | 0.3750 | 0.5265 |
| Urban morphology only | Random Forest | 0.4299 | 0.3899 | 0.5252 |
| Urban morphology only | XGBoost | 0.4242 | 0.3743 | 0.5289 |
| Full (spatial + morphology) | Random Forest | 0.4306 | 0.4017 | 0.5104 |
| Full (spatial + morphology) | XGBoost | 0.4266 | 0.3811 | 0.5308 |

**Findings:**

- Morphology-only already beats the spatial-means benchmark for both models, indicating the new urban-form features carry real predictive signal.
- Combining spatial means + morphology gives the best RF macro F1 (0.4306) and the best XGB accuracy (46.29%).
- Severe-class recall benefits most from morphology (RF +0.0237 over spatial-only), which is valuable for hotspot detection.
- The marginal gain of morphology over spatial means is small (~+0.007 macro F1 for RF), so feature-family attribution experiments (buildings vs roads vs vegetation vs land-use vs contrast, separately) are a sensible next step but were not required to establish that morphology helps.

---

## 14. Feature Importance

Top built-in (Gini) importances from the selected full-feature Random Forest
(spatial means + urban morphology).  One-hot `landuse_*` columns are excluded
from the table for brevity; the table shows the top 11 of 100+ encoded
features:

| Feature | Importance |
|---|---|
| `dist_building_m_mean11` | 0.0418 |
| `dist_building_m_mean5` | 0.0411 |
| `dist_building_m` | 0.0407 |
| `ndbi_mean11` | 0.0405 |
| `dist_building_m_mean3` | 0.0394 |
| `year` | 0.0360 |
| `vegetation_cover_mean_500m` | 0.0338 |
| `ndbi_mean5` | 0.0336 |
| `dist_vegetation_m_mean11` | 0.0274 |
| `dist_road_m_mean11` | 0.0269 |
| `vegetation_cover_mean11` | 0.0269 |

Building-distance features (raw and neighbourhood means) and NDBI means
dominate, with `year` and the 500 m vegetation-cover mean close behind.
Compared with the pre-morphology model, neighbourhood aggregates now occupy
most of the top ranks — the morphology representation is doing real work.
Feature importance is interpreted as **association / influence**, not
causation.

Permutation importance (independent stratified hold-out sample,
`permutation_importance.csv`) ranks `year` first by a wide margin
(mean macro-F1 drop ≈ 0.157), followed by `ndbi_mean11` and the
building-distance features; morphology features (`vegetation_cover_mean_500m`,
`landuse_frac_6_500m`, `road_pixel_fraction_500m`) also appear high.

---

## 15. Final Model Selection

**Selected model:** Random Forest (spatial means + urban morphology)

**Selection rationale (deterministic hierarchy):**

1. Mean spatial-CV macro F1: **0.4306** (highest; XGBoost 0.4266)
2. High + Severe recall: **0.9121** (XGBoost 0.9119)
3. Macro F1 fold-to-fold std: **0.0325** (XGBoost 0.0259)
4. Simpler/more interpretable than XGBoost

The margin over XGBoost is small. RF is retained as the primary model for interpretability, but XGBoost is essentially tied and is a valid alternative.

---

## 16. Temporal Generalization

| Experiment | Macro F1 |
|---|---|
| Random Forest 2022 → 2026 | 0.2903 |
| XGBoost 2022 → 2026 | 0.2828 |
| Random Forest 2026 → 2022 | 0.2910 |
| XGBoost 2026 → 2022 | 0.2904 |

Temporal generalization macro F1 (~0.28–0.29) remains substantially lower than within-year spatial CV (~0.43). Adding morphology features did not materially improve year-to-year transfer. This indicates that relationships learned from one July do not fully transfer to the other July, possibly due to different atmospheric conditions, cloud masking, or LST distribution shifts. Temporal generalization remains an important scientific caveat.

---

## 17. Final Predictions and Maps

The selected Random Forest model was trained on the full dataset and used to generate:

- `data/processed/phase5/tables/predictions.csv` — final production predictions
- `data/processed/phase5/tables/oof_predictions.csv` — out-of-fold predictions for every model and fold
- `data/processed/phase5/tables/confusion_matrices.csv` — per-fold and aggregated confusion matrices
- `data/processed/phase5/tables/boundary_case_analysis.csv` — accuracy vs distance-to-threshold and per-boundary errors
- `data/processed/phase5/tables/permutation_importance.csv` — RF permutation importance on a hold-out sample
- `data/processed/phase5/predictions/uhi_2022.tif`
- `data/processed/phase5/predictions/uhi_2026.tif`
- `data/processed/phase5/probabilities/probability_*.tif`
- `data/processed/phase5/maps/uhi_2022.png`
- `data/processed/phase5/maps/uhi_2026.png`

Class encoding:

| Value | Class |
|---|---|
| 0 | Low |
| 1 | Moderate |
| 2 | High |
| 3 | Severe |

Rasters match the Phase 4 reference grid (EPSG:4326, 1,874 × 1,768 pixels). Masked pixels remain NoData.

### 17.1 Map interpretation

- **2022:** More extensive High/Severe areas, consistent with the warmer overall 2022 LST distribution.
- **2026:** Relatively more Low/Moderate areas, consistent with the cooler overall 2026 LST distribution.
- The Yamuna river corridor in the east appears as cooler (Low/Moderate) areas in both years.
- Urban core and built-up areas tend toward High/Severe.
- Patterns are spatially coherent rather than random noise.

---

## 18. Validation

`src/models/validate_models.py` verifies:

- Dataset integrity (required columns, no NaNs, expected years, row count, uniqueness)
- Leakage prevention (`lst_C`, `spatial_block_id`, coordinates excluded from predictors)
- Spatial fold disjointness
- Prediction validity (class values, probability sums, confidence range)
- Raster alignment (CRS, shape, transform, valid class values, table/raster consistency)

**Result:** 37 PASS / 0 FAIL / 0 WARN (includes morphology feature checks and diagnostic output checks)

---

## 19. Scientific Audit

### 19.1 Target

- Per-year quartile classes are balanced (37,500 per class per year).
- LST increases monotonically across predicted classes, confirming target consistency.
- CV thresholds are derived from training blocks only.

### 19.2 Model

- Random Forest and XGBoost fold-mean results are stable (RF macro F1 std 0.0325; XGB 0.0259).
- No suspiciously high validation score suggesting leakage.
- Performance is well above all baselines but far from perfect, indicating a genuinely difficult classification problem.
- **Overfitting check:** RF training accuracy is ~99.5% vs ~46.3% spatial-validation accuracy — a large train–validation gap typical of an unconstrained Random Forest on a noisy four-class target. XGBoost shows a much smaller gap (train ~64.0% vs validation ~46.3%). RF is still preferred by the deterministic macro-F1 selection rule, but the gap is documented as a caveat.

### 19.3 Features

- Built-in RF importance remains dominated by `dist_building_m`, `dist_road_m`, NDBI, and NDVI, with neighbourhood means of the distance layers also important.
- Permutation importance (hold-out sample) ranks `year` first by a wide margin, followed by `ndbi_mean11` and the building-distance features. The strong `year` effect is consistent with the two Julies having different LST distributions; it is a distributional/period indicator, not a physical driver, and is interpreted cautiously.
- Morphology features contribute: vegetation-cover means, residential land-use fraction at 500 m, and road pixel fraction at 500 m all appear in the top permutation-importance features.
- PVC contributes almost nothing beyond NDVI.
- No coordinate features and no `spatial_block_id` are used.

### 19.4 Spatial results

- Hotspots form geographically coherent patterns.
- No isolated one-pixel artifacts.
- River corridor appears cooler; built-up areas appear warmer.
- Masked regions remain masked.

### 19.5 Boundary-case and OOF diagnostics

Aggregate out-of-fold (OOF) diagnostics for the full-feature models
(`boundary_case_analysis.csv`, `oof_predictions.csv`):

- **Accuracy vs distance to nearest quartile threshold (RF, all folds/years
  pooled):** clear cases (> 1.5 °C from the nearest of q25/q50/q75):
  **52.3%** accuracy (n = 115,838); moderate band (0.5–1.5 °C): **43.6%**
  (n = 100,595); boundary band (< 0.5 °C): **40.8%** (n = 83,567).
  Roughly 28% of validation pixels sit within 0.5 °C of a class boundary,
  where even a perfect environmental model would often disagree with the
  quartile label — this is inherent label noise in the target, not purely
  model weakness.
- **Per-boundary accuracy within ±0.5 °C of the threshold (RF):**
  Low↔Moderate 41.2% (n = 28,306), Moderate↔High 39.2% (n = 33,687),
  High↔Severe 43.0% (n = 21,574). The Moderate↔High boundary is the hardest.
- **OOF vs fold-mean metrics:** the pooled OOF RF macro-F1 is 0.4594, higher
  than the fold-mean 0.4306 because folds have unequal sizes and difficulty;
  the project convention is to report the fold-mean, which weights each
  spatial fold equally.
- OOF predictions are persisted for every model and fold and are the basis
  for the confusion matrices and the boundary analysis above.

---

## 20. Advanced Generalization and Overfitting Audit

This section reports the Phase 5 generalization/overfitting audit
(`src/models/generalization_audit.py`, driven by `scripts/run_phase5_audit.py`).
Frozen benchmark under test: full spatial + morphology model
(RF 46.28% / macro-F1 0.4306; XGB 46.29% / 0.4266). All diagnostics reuse the
same 5-fold spatial GroupKFold, seed 42, and training-only target thresholds.
Each diagnostic lists purpose → result → interpretation.

**1. Train/validation gap** (`train_validation_gap.csv`) — fold-level gap
diagnostic. RF: train 99.5% vs validation 46.3% (mean gap 53.3 pp; per-fold
47.6–58.3 pp). XGB: 64.0% vs 46.3% (17.7 pp). Confirms the baseline RF is a
high-variance memorizer; XGBoost is far less overfit. *Indicates overfitting:
yes (RF baseline).*

**2. Model regularization** (`rf_regularization.csv`, `xgb_regularization.csv`)
— controlled RF-A…D / X0…X3 candidates under identical folds. RF gap collapses
53.3 → 44.4 (B) → 28.9 (C) → 11.6 pp (D) with **no validation loss**; RF-C
even gains macro-F1 (0.4310) and accuracy (46.99%). Best XGB candidate (X3)
reaches macro-F1 0.4287 at a 9.2 pp gap. *Overfitting is controllable and is
not the performance bottleneck.*

**3. Seed stability** (`seed_stability.csv`) — 5 seeds × 2 models, identical
folds. Across-seed macro-F1 std 0.0005 (RF), 0.0006 (XGB); ranges ≤ 0.0012.
Results are not seed artifacts.

**4. Prediction stability** (`prediction_stability.csv`) — 79.1% of OOF pixels
get 5/5 seed agreement; hotspot (High+Severe vs rest) agreement 89.1% at 5/5.
Per-pixel predictions are stable where it matters operationally.

**5. Target permutation sanity test** (`target_permutation_test.csv`) — 3
seeds × 2 models trained on permuted training labels. Accuracy 23.6–26.0%
(chance ≈ 25%), macro-F1 ≈ 0.23–0.25. **No leakage, no duplicated-target
shortcut.** This is a sanity check, not a model-selection experiment.

**6. Learning curves** (`learning_curves.csv`) — deterministic block-level
training subsets (20→100%), validation untouched. RF train stays ~99.5% at
every fraction (memorization), while validation rises 40.0 → 46.3% and is
flattening (+0.5 pp for the last 20% increment). More data alone will not
reach 60%; the binding constraint is feature information, not sample size.

**7. Leave-one-block-out** (`leave_one_block_out.csv`) — 20 occupied blocks,
train on the rest. Mean accuracy 47.3% (RF) / 47.5% (XGB); per-block macro-F1
0.24–0.57. Worst block (10, both models) has near-zero High recall but 0.94
Severe recall — a local class-threshold imbalance, flagged for Phase 6. LOBO
matches GroupKFold means, so no fold-construction artifact.

**8. Spatial distance stress test** (`spatial_distance_stress_test.csv`) —
accuracy vs distance to nearest training block: 54.2% (0–500 m) → 50.7%
(500 m–1 km) → 47.4% (1–2 km) → 42.6% (> 2 km). The model relies partly on
local spatial similarity; with permutation at chance, this is spatial
autocorrelation, not leakage — but far-transfer claims must use the distant
bands, not the CV mean.

**9. Feature permutation (training-side)** (`feature_permutation_sanity.csv`)
— permute one training feature, retrain, evaluate untouched validation.
Degradations are tiny (year 0.011; all others ≤ 0.001) because the 90-feature
set is highly redundant (e.g. `dist_building_m` has three neighbourhood-mean
siblings). Consistent with the compact-feature result; this is *not* ordinary
validation-side permutation importance.

**10. Nested spatial CV** (`nested_spatial_cv_results.csv`) — outer 5 folds,
inner 4-fold GroupKFold on outer-training blocks only (outer validation blocks
asserted absent from inner data), 4 RF candidates. Inner selection picks
different candidates per fold (no universal winner); mean outer macro-F1
0.4306 — identical to flat CV. *The headline result has no optimistic
selection bias.*

**11. Locked geographic holdout** (`locked_geographic_holdout.csv`) — blocks
[3, 11, 17] locked by the deterministic rule `sorted_ids[2::6]`, never touched
during development; evaluated exactly once after freezing the configuration.
Selected RF-C: **39.9% accuracy, macro-F1 0.395** (XGB baseline 40.1% /
0.397) on 56,560 unseen pixels — consistent with LOBO (0.387) and the >2 km
band (0.383). The holdout *supports* the model (no collapse, no leakage) while
showing adjacent-block CV is ~7 pp optimistic for wholly unseen geography.

**12. Calibration** (`calibration_report.csv`) — multiclass Brier 0.162 (RF) /
0.164 (XGB); ECE ≈ 0.045 (RF) / 0.064 (XGB) over 10 confidence bins on
development OOF predictions. Probabilities are reasonably calibrated;
`prediction_confidence` remains a confidence proxy, not a calibrated
probability.

**13. Residual spatial autocorrelation** (`spatial_error_autocorrelation.csv`)
— Moran's I (k=8 NN weights on 30 m-scaled coordinates, 199 permutations):
classification error I ≈ 0.53–0.54, |class-distance error| I ≈ 0.61–0.70,
p = 0.005 (both years). Errors are strongly spatially clustered — evidence of
unmodelled spatial environmental structure (missing predictors), *not*
automatically overfitting.

**14. Data integrity & leakage audits** (`data_integrity_duplicate_audit.csv`,
`leakage_audit.csv`) — no duplicate (row, col, year); the 150k repeated
(row, col) pairs are legitimate 2022/2026 temporal observations; no
predictor-identical rows with conflicting labels; all 10 leakage-audit rows
PASS (`lst_C`, coordinates, `spatial_block_id`, and target-derived thresholds
are never predictors; morphology neighbourhoods never cross blocks).

### 20.1 Final model decision framework

| Candidate | Spatial-CV acc | Macro-F1 (std) | Gap | LOBO mean F1 | Seed range | Nested-CV | Locked holdout | Verdict |
|---|---|---|---|---|---|---|---|---|
| RF baseline (A) | 46.28% | 0.4306 (0.033) | 53.3 pp | 0.387 | 0.0011 | 0.4306 | 0.395* | YELLOW (overfit gap) |
| **RF-C (depth 15, leaf 10)** | **46.99%** | **0.4310 (0.028)** | **28.9 pp** | — | — | selected in 1/5 folds | **0.395** | **GREEN (selected)** |
| RF-D (depth 10, leaf 15) | 47.45% | 0.4301 (0.024) | 11.6 pp | — | — | selected in 2/5 folds | not evaluated | GREEN |
| XGB baseline | 46.29% | 0.4266 (0.026) | 17.7 pp | 0.377 | 0.0012 | — | 0.397 | YELLOW-GREEN |
| XGB X3 (regularized) | 46.86% | 0.4287 (0.026) | 9.2 pp | — | — | — | not evaluated | GREEN |

\* evaluated with RF-C parameters in the single locked-holdout run.

**Selected model: RF-C** — Random Forest, `max_depth=15`,
`min_samples_leaf=10`, other parameters at baseline, full spatial +
morphology feature set. It wins the mandated hierarchy: highest mean macro-F1
(0.4310), equal-best accuracy, best High+Severe recall (0.947), lower fold
std, gap halved. The margin over the baseline RF is small; the selection is
driven by generalization quality, not the primary metric alone. Production
rasters/maps in `data/processed/phase5/predictions` remain the frozen baseline
RF; adopting RF-C is a drop-in parameter change deferred to Phase 6.

### 20.2 Audit conclusions

- **60% is not achieved**, and the audit shows why: the trustworthy ceiling is
  **≈ 47% accuracy / 0.43 macro-F1** under adjacent-block spatial CV and
  **≈ 40% / 0.40** on fully held-out geography. No leakage, no seed
  instability, no selection bias explains the gap — the environmental
  predictors have limited separability for four quartile classes.
- **Dominant bottleneck: target formulation/feature information.** Boundary
  pixels (< 0.5 °C from a quartile threshold, ~28% of samples) are classified
  at 40.8% vs 52.3% for clear cases; errors are strongly spatially
  autocorrelated (missing spatial predictors); temporal transfer is stuck at
  ~0.29 macro-F1 regardless of the `year` feature.
- **Overfitting verdict:** the baseline RF was overfit but regularization
  removes most of it without validation loss; it is not the primary blocker.
- **Next experiment (Phase 6 direction):** alternative target formulations
  (binary hotspot, 3-class, ordinal, or LST-regression → quartiles), feature
  families not yet used (water/NDWI, impervious fraction, albedo), and
  hierarchical/ordinal classifiers — plus per-block threshold recalibration
  for blocks like #10.

---

## 21. Limitations

1. **Target is LST-derived:** Classes represent relative heat severity within each year, not independently measured physical UHI intensity.
2. **Temporal generalization is weak:** Models trained on one year perform poorly on the other year (~0.29 macro F1).
3. **Static context layers:** Road, vegetation, and building distance layers are treated as identical for 2022 and 2026.
4. **Building sample:** Distance-to-building is based on a Central Delhi sample only; `building_pixel_fraction` is a zero-distance pixel proxy, not a measured footprint-area fraction.
5. **PVC redundancy:** Vegetation Cover is mathematically derived from NDVI and adds little predictive information.
6. **Moderate recall:** The Moderate class has the lowest recall (~0.34 for RF), likely because it sits between two adjacent classes with overlapping predictor distributions; boundary-case analysis shows accuracy is much lower near quartile thresholds.
7. **Block-edge truncation:** Block-aware morphology kernels are clipped near spatial-block boundaries (15.8% of sampled pixels at 500 m), which is a conservative leakage-control trade-off.
8. **RF train–validation gap:** RF training accuracy (~99.5%) far exceeds validation accuracy (~46.3%); the model is not externally overfit to validation blocks (spatial CV is intact) but memorises training blocks.
9. **No hyperparameter tuning:** This is a fixed-parameter baseline; controlled tuning may improve results.
10. **Correlation is not causation:** Feature importance shows association, not causal relationships.
11. **Adjacent-block CV is optimistic for far transfer:** the locked geographic holdout (fully unseen blocks) gives ~40% accuracy / ~0.40 macro-F1 vs ~47% / 0.43 under GroupKFold — claims about unseen geography should use the former.
12. **The 60% accuracy target is not reached:** the trustworthy ceiling under the approved leakage-free methodology is ~47% (spatial CV) / ~40% (locked geography); the audit attributes the gap primarily to target formulation and missing environmental information, not to overfitting.

---

## 22. Reproducibility

- Random seed: 42
- All stochastic model components are seeded.
- Python and package versions recorded in `phase5_pipeline_record.json`.
- Clean-run test performed: Phase 5 outputs deleted and pipeline rerun.
- A second clean run (2026-09-02, from a fully deleted
  `data/processed/phase5/`) reproduced the frozen benchmark **exactly**:
  every value in `model_comparison.csv` and `ablation_results.csv` matched the
  pre-deletion snapshot to all recorded decimals, and the validator again
  reported 37 PASS / 0 FAIL / 0 WARN.
- **Clean-run reproducibility: PASS**
  - RF macro F1 matched to machine precision.
  - XGBoost macro F1 matched to machine precision.
  - Target thresholds matched to machine precision.
  - Class distributions matched exactly.
- Generalization audit: all diagnostics deterministic (fixed seeds, fixed fold
  lists, deterministic block-selection rules); the audit run and the single
  locked-holdout evaluation are recorded in `phase5_generalization_summary.json`
  and `phase5_pipeline_record.json`.

---

## 23. Files Added or Modified

### New source code

```text
src/models/
├── __init__.py
├── config.py
├── target.py
├── dataset.py
├── validation.py
├── baseline.py
├── random_forest.py
├── xgboost_model.py
├── evaluation.py
├── experiments.py
├── spatial_features.py
├── morphology_features.py   ← block-aware urban morphology
├── generalization_audit.py  ← new (generalization/overfitting audit)
├── maps.py
├── pipeline.py
└── validate_models.py

scripts/
└── run_phase5_audit.py      ← new (audit driver; --final = locked holdout, once)

tests/
├── test_morphology_reference.py      ← new (numerical reference test)
└── test_generalization_audit.py      ← new (audit smoke tests)
```

### New/modified reports and docs

```text
reports/phase5_uhi_detection_report.md
reports/phase5_methodology_proposal.md
README.md
requirements.txt
```

### Generated outputs (gitignored)

```text
data/processed/phase5/
```

---

## 24. Next Steps

Phase 5 (including the urban-morphology extension and the generalization/overfitting
audit) is complete and validated. The audit's conclusions redefine the priorities:
the four-class quartile target, not the model, is the primary bottleneck. Next
experiments, in priority order:

1. **Alternative target formulations** (highest priority): binary hotspot
   detection (High+Severe vs rest), 3-class (Cool/Moderate/Hot), ordinal
   classification, or LST-regression → quartile classes. The audit shows ~28% of
   pixels sit within 0.5 °C of a class boundary at 40.8% accuracy vs 52.3% for
   clear cases — a coarser or ordinal target should convert this directly into
   gains.
2. **Feature families not yet used:** NDWI / water presence, impervious-surface
   fraction, surface albedo — plus feature-family attribution ablations of the
   existing morphology gains (buildings-only, roads-only, vegetation-only).
3. **Per-block recalibration:** LOBO block 10 (macro-F1 0.24, near-zero High
   recall at 0.94 Severe recall) shows local threshold imbalance; a
   block-level calibration stage may recover hotspot recall in such regions.
4. **Adopt RF-C:** promote the regularized RF (`max_depth=15`,
   `min_samples_leaf=10`) selected by the audit as the Phase 6 base model.
5. **Stronger temporal generalization:** year-to-year transfer is ~0.29 macro-F1
   with or without the `year` feature; investigate distribution-shift-robust
   training if cross-year deployment matters.
6. **Ensemble methods:** combine RF and XGBoost probabilities; their errors
   differ and both are seed-stable.

---

*GreenGrid AI — Phase 5 UHI Detection Report | Generated 2026-09-02*
