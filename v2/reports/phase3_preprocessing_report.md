# V2 Phase 3 Preprocessing Verification — W4 Real Run

Date: 2026-10-04 (IST). Chain step: preflight → `v2.phase3.build_core`. Contract: `v2/src/v2/CONTRACT.md`.

## 1. Preflight (`v2.phase3.verify_inputs`)

- Command: `--data-root v2/data/raw --out v2/data/phase2/w4_input_audit.json --crosscheck-audit <same>`.
- Wall time: 91–92 s per run. Final result: **PASS** (exit 0), 0 failures.
- Note: the pre-move audit JSON lacked `b2_zero_guard` records for 2023/2024/2025, so the
  first crosscheck run reported 3 crosscheck FAILs ("no prior B2==0 count found"). The audit
  was regenerated fresh (old file superseded) and the crosscheck re-run pinned the new audit:
  all 5 years `match: true`. The 3 initial FAILs were an artifact of the stale file, not of
  the data.
- Known WARNs (expected, guard active): S2-10m B2==0 zero-fill 2022 = 84,909 px;
  2026 = 7,867 px excluded from the value stack.
- Input SHA-256 (per audit + build manifests): L9 2022 `8c66c741…`, 2023 `03dc3563…`,
  2024/2025/2026 recorded in `v2/data/phase2/w4_input_audit.json` and per-year
  `build_manifest.json` (full hashes there).

Preflight per-year native coverage vs gates (L9 ST ≥ 0.90, S2-10m ≥ 0.85, LULC ≥ 0.80):

| Year | L9 ST cov | S2-10m cov | S2-20m cov (report) | LULC cov | L9 transform exact | Inf |
|---|---|---|---|---|---|---|
| 2022 | 0.969943 | 0.964364 | 0.969631 | 0.969280 | yes | 0 |
| 2023 | 0.969810 | 0.969280 | 0.969631 | 0.969280 | yes | 0 |
| 2024 | 0.969943 | 0.969280 | 0.969631 | 0.969280 | yes | 0 |
| 2025 | 0.969943 | 0.969280 | 0.969631 | 0.969280 | yes | 0 |
| 2026 | 0.969943 | 0.968897 | 0.969631 | 0.969280 | yes | 0 |

B2==0 guard accounting (audit crosscheck, all `match: true`):

| Year | finite_px | b2_zero_px | valid_excl_b2_zero | identity_holds |
|---|---|---|---|---|
| 2022 | 17,193,173 | 84,909 | 17,108,264 | yes |
| 2023 | 17,193,173 | 0 | 17,193,173 | yes |
| 2024 | 17,193,173 | 0 | 17,193,173 | yes |
| 2025 | 17,193,173 | 0 | 17,193,173 | yes |
| 2026 | 17,193,173 | 7,867 | 17,185,306 | yes |

## 2. Phase 3 core (`v2.phase3.build_core`)

- Command: `--data-root v2/data/raw --out-root v2/data/phase3` (defaults: vectors built,
  roads = V1-parity subset, LULC gap-fill OFF).
- Total wall time: 2 min 49.9 s (static layers + 5 years). Per year ≈ 26–27 s
  (2022: 27 s incl. static rasterization start; 2023–2026: 26 s each).
- Exit 0; `build_summary.json` + per-year `build_manifest.json` written.

Per-year 30 m coverage inside study area (1,881,088 cells) vs gates:

| Year | valid_l9 | valid_s2_10m | valid_s2_20m | valid_lulc | domain px | PVC mean |
|---|---|---|---|---|---|---|
| 2022 | 0.969943 | 0.969330 | 0.970393 | 0.970047 | 1,822,769 | 0.3120 |
| 2023 | 0.969810 | 0.970047 | 0.970393 | 0.970047 | 1,824,298 | 0.3470 |
| 2024 | 0.969943 | 0.970047 | 0.970393 | 0.970047 | 1,824,549 | 0.2449 |
| 2025 | 0.969943 | 0.970047 | 0.970393 | 0.970047 | 1,824,549 | 0.3271 |
| 2026 | 0.969943 | 0.970005 | 0.970393 | 0.970047 | 1,824,444 | 0.3052 |

B2-exclusion identity on real data (per-year `build_manifest.json`,
`stats.s2_10m_b2_zero_guard`): identical to the preflight table above for all 5 years —
`identity_holds: true` and counts equal the pinned audit exactly.

Registration check (V1 F4 fix): all 80 per-year output rasters (16/year) carry the
authoritative grid transform exactly — affine
(0.00026949458523585647, 0, 76.83290625074268, 0, −0.00026949458523585647, 28.884699140164333),
EPSG:4326, 1768×1874 — verified programmatically raster-by-raster, including the S2-20m-derived
`ndbi_30m.tif` / `ndre_30m.tif` (explicit reprojection onto the authoritative grid; output
transform == authoritative grid transform exactly, all six affine terms).

## 3. Gates / checks summary

| Check | Result |
|---|---|
| Preflight (files, bands, dtypes, transforms, CRS, labels, Inf, coverage gates) | PASS |
| B2==0 audit crosscheck (5 years) | PASS (all match) |
| Per-year output inventory (16 tifs + build_manifest.json × 5) | complete |
| Output grid: transform/CRS/shape == authoritative | exact (all 80) |
| Coverage gates at 30 m (L9 ≥ 0.90, S2-10m ≥ 0.85, LULC ≥ 0.80) | PASS all years |
| B2-exclusion identity on real data | holds, counts == audit |
| S2-20m half-pixel registration | fixed by construction; transform exact |

## 4. Deviations applied (build-time defaults, per contract §9)

- Roads: V1-parity subset burned — 3,637 / 238,194 features
  (highway ∈ {motorway, trunk, primary}); `roads_full_network_burned: false`.
- Vegetation: V1-parity tag union, 6,423 / 6,484 features (recreation_ground excluded).
- Landuse: 7 V1 values, 7,546 classified / 13,570 stored; 1 geometry repaired.
- Buildings: full NCT, 417,343 footprints (2 repaired).
- LULC: NO gap-fill (opt-in flag off); unfilled pixels are NaN, outside the ML domain.
- Water 2,682 and road-surfaces 78 presence polygons burned.

## 5. File inventory

- `v2/data/phase3/{2022..2026}/`: 16 tifs each — lst, ndvi, ndbi, ndre, ndmi, mndwi, bsi,
  l9_ndvi, l9_ndbi, vegetation_cover, lulc, valid_{l9,s2_10m,s2_20m,lulc}, domain — plus
  `build_manifest.json` (input/output SHA-256, stats).
- `v2/data/phase3/static/`: roads/vegetation/buildings distance, landuse raster,
  water + road-surfaces presence, `static_manifest.json`.
- `v2/data/phase3/build_summary.json`; audit: `v2/data/phase2/w4_input_audit.json`.
- Run logs: `v2/logs/_preflight_run.log`, `_preflight_crosscheck.log`, `_build_core_run.log`.
