# V2 Phase 4 Feature Assembly Verification — W4 Real Run

Date: 2026-10-04 (IST). Chain step: `v2.phase4.assemble_features` → `v2.phase4.verify_features`.
Inputs: `v2/data/phase3/` (verified, see `phase3_preprocessing_report.md`),
schema `v2/reference/phase5_primary_xgb_3class_features.json`, met
`v2/data/phase2/met/met_superset_hourly_2022_2026.csv`, study area
`v2/data/gis/study_area/study_area.geojson`.

## 1. Assembly

- Command: `--phase3-root v2/data/phase3 --out-root v2/data/phase4` (defaults:
  study-area clip on; sampling 150,000/yr, seed 42, without replacement, sorted;
  pinned V1 block bins; no CSV side output).
- Wall time: total 33 min 36.8 s. Per year: 2022 6 min 40 s, 2023 6 min 38 s,
  2024 6 min 43 s, 2025 6 min 39 s, 2026 6 min 49 s (sequential; single process).
- Exit 0. Benign `RuntimeWarning` (log(0) in landuse entropy, handled `nan=0.0`)
  — no effect on outputs.

Per-year result (from `v2_feature_manifest.json` / `v2_feature_verify.json`):

| Year | Domain px | Sampled rows | Rows written | Cols | Blocks occupied |
|---|---|---|---|---|---|
| 2022 | 1,821,324 | 150,000 | 150,000 | 182 | 20 |
| 2023 | 1,822,853 | 150,000 | 150,000 | 182 | 20 |
| 2024 | 1,823,104 | 150,000 | 150,000 | 182 | 20 |
| 2025 | 1,823,104 | 150,000 | 150,000 | 182 | 20 |
| 2026 | 1,822,999 | 150,000 | 150,000 | 182 | 20 |

182 cols = 4 metadata (row, col, spatial_block_id, lst_C) + 178 predictors.
Domain = all-finite predictors ∩ study clip; sampled 150k/yr because domain ≈ 1.82 M
> cap. All 5 years occupy the same 20 of 25 pinned blocks (block set identical by
construction — bins pinned, independent of sample).

## 2. Verification (`v2.phase4.verify_features`) — ALL GATES PASS

Command: `--features-dir v2/data/phase4 --phase3-root v2/data/phase3`.
Wall time ≈ 2.8 s; exit code 0 (verified with a direct un-piped re-run).
Full gate report: `v2/data/phase4/v2_feature_verify.json` → `"status": "PASS"`,
`"failures": []`.

Verbatim gate results:

1. **Predictor columns == frozen 178 (names and order)** — PASS.
   `frozen_schema_sha256 = ddd43d0468dcab8df1c173a2adf57c8e3aec9f386851f7031b77cb3f8dc2c211`
   (exact contract match). Assemble log: "frozen schema: 178 features, sha256 ddd43d0468dcab8d…".
2. **No NaN/Inf in predictors** — PASS (750,000 rows checked).
3. **Leakage** — PASS: {lst_C, lon, lat, row, col, spatial_block_id} ∉ predictors;
   `lst_C` metadata/target only.
4. **Per-year coverage gates recomputed from Phase-3 masks** — PASS all years
   (L9 ≥ 0.90, S2-10m ≥ 0.85, LULC ≥ 0.80):

   | Year | valid_l9 | valid_s2_10m | valid_lulc |
   |---|---|---|---|
   | 2022 | 0.969943 | 0.969330 | 0.970047 |
   | 2023 | 0.969810 | 0.970047 | 0.970047 |
   | 2024 | 0.969943 | 0.970047 | 0.970047 |
   | 2025 | 0.969943 | 0.970047 | 0.970047 |
   | 2026 | 0.969943 | 0.970005 | 0.970047 |

4b. **Hard B2 invariant** — PASS: every feature row's 30 m cell contains ≥ 1 valid
   native S2-10m pixel under the B2>0 guard (no B2==0 pixel contribution).
5. **spatial_block_id == pinned V1 bins** — PASS; independent of the sample
   (`block_geometry: "pinned V1 bins (HOLDOUT_BLOCKS=[2,9,15,23])"`).
6. **Met sanity** — PASS: all 6 met columns finite; not erroneously constant;
   adjacent-pixel |Δ| within `--met-gradient-tol 0.1` for all columns/years
   (max observed |Δ| = 0.0217, met_ssr_wm2 2025 ≪ 0.1).
7. **Count summary** — written to `v2_feature_verify.json` (table above).

Warnings (30, informational only — per-column met std/max-adjacent-diff statistics
emitted at WARN level for every year×column; all gradients far below tolerance):
2022–2026 × {met_t2m_c, met_rh_pct, met_wind_kmh, met_precip_mm, met_ssr_wm2,
met_swc_m3m3}; full list in `v2_feature_verify.json` → `warnings`.

## 3. Deviations / settings (per contract §7.6, §9)

- Sampling: per-year independent, 150,000 cap, `numpy.default_rng(42)`, sorted.
- Block geometry: PINNED V1 bins (C1); holdout blocks (2, 9, 15, 23).
- Met window: W4 (May 1–Jun 30), hours 04:00–06:00 UTC, V1-matched IDW variant.
- `year` appears only inside the 178-predictor block; metadata = row, col,
  spatial_block_id, lst_C.
- One-hot reindex zero-fill for unseen landuse classes 1/3 (B2 open item — schema-exact).

## 4. File inventory

- `v2/data/phase4/features_{2022..2026}.parquet` (150,000 rows × 182 cols each,
  sha256 in `v2_feature_manifest.json`).
- `v2/data/phase4/v2_feature_manifest.json`, `feature_metadata.json`,
  `v2_feature_verify.json` (full gate report).
- Run logs: `v2/logs/_assemble_run.log`, `_verify_features_run.log`,
  `_verify_features_exit.log`.
