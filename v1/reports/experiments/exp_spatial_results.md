# Experiment: Spatial Neighborhood / Context Features for 3-Class Severity (exp_spatial_context) — Final Report

**Date:** 2026-10-03
**Status:** COMPLETE (run log `exp_spatial_eval.log`, EXIT: 0, ~102 min)

## Design

Start from the best direct 3-class configuration: XGBoost/RF + LULC +
6 met covariates (145 encoded features), same 5-year dataset (749,998 rows),
same sampling, same per-year LST-tertile target (training-fold-only
thresholds), same spatial blocks, same locked holdout **[2, 9, 15, 23]**,
locked evaluated exactly once. Only the predictor set changes.

## Added features (33 new; 145 → 178 encoded features)

All statistics are nan-safe, border-correct functions of NON-TARGET
environmental rasters at 3×3, 5×5 and 11×11 windows (30 m), constructed
identically for every pixel regardless of block membership:

| Stat | Bands | Columns |
|---|---|---|
| `std` (local heterogeneity) | ndvi, ndbi, ndre, ndmi, mndwi, bsi, vegetation_cover | 21 |
| `mean` (bands missing from existing cache) | mndwi, ndre | 6 |
| `range` (max−min) | ndvi, ndbi | 6 |

Source rasters verified numerically against the modeling table (max err 0.0):
ndvi/ndbi/ndre → s2_{year}_*_30m.tif; ndmi/mndwi/bsi →
l9_{year}_composite_*_30m.tif; vegetation_cover →
phase4/features/vegetation_cover_{year}_30m.tif.

**Per the user's decision (2026-10-03), no LST-derived neighborhood features
were added** — they violate the standing rule against target-derived
predictors and would be deployment-circular. Note: 3×3/5×5/11×11 window
*means* for ndvi/ndbi/vegetation_cover/ndmi/bsi were already present in the
145 (Tier-2 spatial cache); this experiment adds heterogeneity (std/range)
and the missing mndwi/ndre means.

Feature construction: `scripts/build_spatial_context_features.py`; table:
`data/processed/experiments/exp_spatial_context_features.csv` (0 NaN).

## Headline — XGBoost

| Metric | Direct 3-class (145 feat) | +33 spatial (178 feat) | Delta |
|---|---|---|---|
| CV5 accuracy | 63.42% | **65.79%** | +2.37 pp |
| CV5 macro-F1 | 0.6225 | **0.6463** | +0.0238 |
| LOBO accuracy | 66.24% | **67.93%** | +1.69 pp |
| LOBO macro-F1 | 0.6283 | **0.6482** | +0.0199 |
| **Locked accuracy** | **62.99%** | **63.97%** | **+0.97 pp** |
| **Locked macro-F1** | **0.6186** | **0.63146** | **+0.0128** |

Random Forest agrees: locked **64.22% / 0.6318** (CV5 66.65%, LOBO 68.08%) —
with the spatial features RF marginally exceeds XGBoost.

## Locked confusion matrix (XGBoost, rows = true)

|  | Pred Low | Pred Moderate | Pred High | Support |
|---|---|---|---|---|
| **True Low** | 49,620 | 15,404 | 3,061 | 68,085 |
| **True Moderate** | 14,774 | 26,246 | 9,521 | 50,541 |
| **True High** | 5,207 | 9,655 | 26,425 | 41,287 |

## Locked per-class metrics — Moderate-class focus

| Class | Precision | Recall | F1 | Baseline F1 | Δ F1 |
|---|---|---|---|---|---|
| Low | 0.713 | 0.729 | 0.721 | 0.715 | +0.006 |
| **Moderate** | 0.512 | **0.519** | **0.515** | 0.496 | **+0.020** |
| High | 0.677 | 0.640 | 0.658 | 0.645 | +0.013 |

Moderate recall improved **48.4% → 51.9% (+3.5 pp)** — the middle class, the
historical weak point, benefits most, consistent with heterogeneity features
(NDVI/NDBI std, ranges) disambiguating spectrally mixed pixels. High recall
+1.3 pp; Low recall −1.1 pp (mild trade as confidences shift inward).

## Interpretation

1. **Consistent, honest gain.** Positive at every validation level (CV5,
   LOBO, locked) and for both model families — same signature as the met
   covariate result. Magnitude is modest (+0.97 pp locked), matching
   expectation: local context helps at the margins but cannot bridge the
   geographic-distribution gap.
2. **Mechanism fits the design:** std/range capture sub-window surface
   mixing (vegetation/built mosaics) invisible to single-pixel values —
   exactly the signal expected to separate Moderate from Low/High.
3. **No leakage:** features are pure functions of environmental rasters; no
   target, label, coordinate, or block-ID information. Construction is
   identical across train/validation/LOBO/locked by design (per-pixel raster
   statistics).
4. Thresholds identical to exp_3class (same training sets); full per-fold /
   per-block values in the JSON.

## Conclusion

**The +33 spatial context features are a keeper: new best configuration is
XGBoost 63.97% locked / 0.6315 macro-F1 (178 features)** — honestly labeled
as a modest, consistent improvement over 62.99% / 0.6186 on the identical
target and protocol.

## Artifacts

- Results: `data/processed/experiments/exp_spatial_3class_geoeval.json`
- Log: `data/processed/experiments/exp_spatial_eval.log`
- Feature build: `scripts/build_spatial_context_features.py` +
  `data/processed/experiments/exp_spatial_context_features.csv`
- Baseline: `data/processed/experiments/exp_3class_3class_geoeval.json`

## Frozen assets (untouched)

`data/processed/lulc_outputs/`, `baseline_5yr_4class/`, `tier12_outputs/`,
primary 4-class model and validation protocol: **not modified**.
