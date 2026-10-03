# Experiment: Meteorological Covariates (exp_met) — Final Report

**Date:** 2026-10-03
**Status:** COMPLETE
**Authorization:** Single experiment — add met covariates to frozen primary model, compare vs frozen 50.02% locked holdout. No other experiments launched.

## Hypothesis

Air temperature, humidity, wind, precipitation, solar radiation and soil moisture
explain LST variation beyond surface/spectral features, especially sub-city
synoptic-scale variation during the July observation window.

## Data

- **Source:** Open-Meteo archive API (ERA5 / ERA5-Land reanalysis), free, no key.
- **Variables (6):** `met_t2m_c` (2 m air temp), `met_rh_pct` (relative humidity),
  `met_wind_kmh` (wind), `met_precip_mm` (precipitation), `met_ssr_wm2` (solar
  radiation), `met_swc_m3m3` (soil moisture, top layer).
- **Alignment:** 16-point Delhi grid, July 1–30 mean over 04:00–06:00 UTC
  (Landsat acquisition window), IDW-interpolated to pixels. Exact source,
  grid and per-year means documented in `reports/experiments/exp_met_provenance.md`.
- **Leakage check:** PASS — no LST, no coordinates, no block IDs, no
  target-derived variables. 0 NaN across 749,998 rows.
- **Spatial alignment audit:** PASS (`reports/experiments/alignment_audit.md`) — all model
  rasters exactly on the reference 30 m grid; 100-pixel trace exact.

## Protocol (identical to frozen baseline)

- Same dataset (749,998 rows), same target (4-class severity), same spatial blocks.
- Same deterministic locked geographic holdout: blocks [2, 9, 15, 23].
- 5-fold adjacent-block CV (serial), LOBO (4 workers), locked evaluation.
- Baseline = frozen XGBoost + LULC (101 predictors, lulc_5yr run).
- exp_met = same 101 predictors + 6 met covariates (145 encoded feature columns).

## Results

### XGBoost (headline — same family as frozen primary)

| Metric | Baseline (frozen) | +Met | Delta |
|---|---|---|---|
| CV5 accuracy | 50.63% | 53.85% | **+3.22 pp** |
| CV5 macro-F1 | 0.4851 | 0.5194 | +0.0343 |
| LOBO accuracy | 52.38% | 55.16% | **+2.78 pp** |
| LOBO macro-F1 | 0.4800 | 0.5113 | +0.0313 |
| **Locked accuracy** | **50.02%** | **51.40%** | **+1.38 pp** |
| **Locked macro-F1** | **0.4769** | **0.5068** | **+0.0299** |
| Locked high+severe recall | 43.60% | 49.72% | **+6.12 pp** |

### Random Forest (secondary)

| Metric | Baseline | +Met | Delta |
|---|---|---|---|
| CV5 accuracy | 49.91% | 55.23% | +5.32 pp |
| CV5 macro-F1 | 0.4803 | 0.5365 | +0.0562 |
| LOBO accuracy | 51.81% | 56.32% | +4.51 pp |
| LOBO macro-F1 | 0.4767 | 0.5291 | +0.0524 |
| Locked accuracy | 48.50% | 51.39% | +2.89 pp |
| Locked macro-F1 | 0.4672 | 0.5078 | +0.0406 |

## Interpretation

1. **Consistent positive effect.** The improvement appears at every validation
   level (CV5, LOBO, locked) and for both model families — this consistency,
   not the size of any single number, is the evidence that the met covariates
   carry real signal.
2. **Locked gain is modest (+1.38 pp accuracy, +0.03 macro-F1).** Expected:
   the `year` predictor already absorbs between-year weather differences, so the
   met fields add mostly smooth sub-city synoptic variation. They cannot repair
   the core geographic-generalization gap.
3. **High+severe recall improved the most (+6.12 pp, XGB locked).** For a
   heat-risk application this is the most practically useful gain — the model
   misses fewer of the hottest pixels.
4. **No leakage, no protocol change, no holdout tuning.** Locked blocks
   [2, 9, 15, 23] are identical to the baseline run.

## Conclusion

- The met-covariate model **XGBoost 51.40% locked / 0.5068 macro-F1 is the new
  best configuration**, honestly labeled: a modest but consistent improvement
  over the frozen 50.02% / 0.4769 baseline.
- The 75% target remains far out of reach for pixel-level 4-class geographic
  holdout with this feature set; the gap is a data/generalization problem, not
  a missing-covariate problem.

## Artifacts

- Results: `data/processed/experiments/exp_met_classification_geoeval.json`
- Baseline: `data/processed/experiments/lulc_5yr_classification_geoeval.json`
- Run log: `data/processed/experiments/exp_met_eval4.log` (EXIT: 0)
- Features: `data/processed/experiments/exp_met_features.csv` (+ grid provenance CSV)
- Provenance: `reports/experiments/exp_met_provenance.md`; alignment: `reports/experiments/alignment_audit.md`

## Frozen assets (untouched)

- `data/processed/lulc_outputs/` — primary model, tables, pipeline records
- `data/processed/baseline_5yr_4class/`, `tier12_outputs/` — earlier generations
- Python pipeline and validation protocol: **not modified**
