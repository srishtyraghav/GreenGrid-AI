# GreenGrid-AI V2 Phase 2 — Temporal-window decision matrix

Evidence: real per-pixel reads of Landsat 8/9 C2-L2 (`qa_pixel`, `ST_B10`) and
Sentinel-2 L2A (`SCL`) from Microsoft Planetary Computer, 2022–2026, Mar 1–Jul 31
union span, study area = Delhi NCT polygon rasterized onto the authoritative V1
30 m grid (`landsat9_2026_07_composite_30m.tif`, EPSG:4326, 1768×1874,
1,881,088 study cells). Masking cloned from `gee/export_data_2023_2025.js`
(see `masking_recipe.md`). Landsat valid-ST = QA bits 3+4 clear AND raw≠0 AND
250–350 K; S2 valid = SCL ∉ {3,8,9,10}, scene cloud < 60 %.

## Decision matrix (Landsat ST = primary thermal evidence; S2 corroborates coverage)

| Window | Valid-ST coverage: mean / min across 5 yrs (%) | Coverage CV (%) | Scenes/yr (mean) | doy_std (mean) | ≥2-obs coverage (%) | Mean composite LST (°C) | Phenomenology fit |
|---|---|---|---|---|---|---|---|
| W0 Jul 1–30 (V1) | 92.7 / **80.4** | 8.4 | 8.4 | 8.0 (tightest) | 62.5 | 38.0 | Post-onset: monsoon clouds; coolest LST |
| W1 Apr 1–Jun 30 | 99.98 / 99.98 | ~0.0 | 30.8 | 25.0 | 100 | 42.3 | Peak-heat plateau |
| W2 Mar 1–Jun 30 | 99.98 / 99.98 | ~0.0 | 40.8 | 33.3 | 100 | 40.0 | Starts too cool (March ~28–31 °C Tmax) |
| W3 Apr 1–Jul 31 | 99.98 / 99.98 | ~0.0 | 39.2 | 34.0 | 100 | 41.8 | Includes cloudy monsoon July tail |
| W4 May 1–Jun 30 | 99.98 / 99.98 | ~0.0 | 20.6 | **16.6** | 100 | **42.8** | Squarely on pre-monsoon heat peak |

Per-year valid-ST coverage (Landsat): W0 = 90.2 / 94.4 / 99.1 / **80.4** / 99.3 %
(2022→2026); every other window = 99.98 % in all five years.

Sentinel-2 corroboration (valid_coverage_pct): W0 collapses in monsoon years
(94.7 / 75.6 / 41.4 / **25.3** / 100.0 %), W1–W4 = 100 % every year.

Phenomenology (Open-Meteo, Delhi-mean): monthly mean Tmax peaks in May–June
(May 36.2–41.5 °C, Jun 36.4–40.4 °C, Jul drops to 33.2–34.4 °C); hottest 30-day
period starts May 22 / Jun 10 / Jun 17 / Jun 12 / Jun 10 (2022–2026, mean ≈ Jun 2,
37.5–43.4 °C); monsoon onset (10-day rolling precip ≥ 2.5 mm/day persisting) =
Jul 4 / Jun 21 / Jun 26 / Jun 17 / Jul 20. Peak UHI heat is **pre-monsoon
May–June**, and July — especially early–mid July — is cloud-contaminated and
cooling down.

## Criteria applied (pre-registered)

1. **Hard screen: valid-ST coverage ≥ 90 % of study area in EVERY year.**
   W0 **FAILS** (2025 = 80.4 %; 2022 = 90.2 % is only marginal). W1, W2, W3, W4 pass.
2. **Among survivors, maximize cross-year temporal consistency.** Coverage CV is
   ~0 for all four survivors, so consistency reduces to seasonal tightness:
   doy_std W4 (16.6) < W1 (25.0) < W2 (33.3) ≈ W3 (34.0); inter-scene ST spread
   W1 4.08 / W4 4.20 / W3 4.57 / W2 5.92 °C. W4 (and W1) keep the composite inside
   the heat peak; W2/W3 dilute it with cooler March or cloudy July acquisitions.
3. **Prefer mean composite LST closest to the annual peak heat.** LST means:
   W4 42.8 ≳ W1 42.3 > W3 41.8 > W2 40.0 > W0 38.0 °C — matches the Tmax
   phenomenology ordering (May–June hottest, July cooler).

## Recommendation: **W4 (May 1 – Jun 30)** — with W1 (Apr 1 – Jun 30) as fallback

- Passes the hard screen at 99.98 % valid-ST coverage in all five years
  (and 100 % ≥2-observation coverage — a robust median composite everywhere).
- Tightest seasonal focus of any survivor (mean doy_std 16.6; ~20.6 scenes/yr,
  ~10 valid observations per cell), so the median composite mixes acquisitions
  within a homogeneous 2-month heat plateau.
- Highest mean composite LST (42.8 °C), consistent with the pre-monsoon Tmax
  peak (hottest 30 d starts May 22–Jun 17 across years) and safely before
  monsoon onset (Jun 17–Jul 20).
- S2 evidence agrees: W4 = 100 % S2 coverage every year vs W0's 25–76 % in
  monsoon-affected years.

**The data contradicts the July-window prior.** W0's attraction was a tight
16-day span, but it sits after monsoon onset: in 2025 only 80.4 % of Delhi got a
single valid Landsat ST observation and median per-cell count was ~2, so the V1
July composite is fragile to inter-annual cloud luck and is the *coolest* window,
biased away from peak UHI season. W4 keeps two months of the year that are
consistently hot and consistently clear.

## Honest caveats

- Five years is short; 2025's W0 failure is one bad year, but it is exactly the
  kind of failure the hard screen is meant to catch.
- LST ≠ air Tmax: criterion 3 uses LST ordering only as corroboration of the
  air-temperature phenomenology; absolute LST levels include surface effects.
- V1's Landsat mask ignores dilated-cloud/cirrus/snow QA bits (only bits 3+4
  masked, per the GEE script); a stricter mask would lower all coverages, but
  July would suffer most (cirrus + monsoon), strengthening the conclusion.
- V1 exported Landsat 9 only; this study used L8+L9 (T1). L8-only years would
  halve scene counts but not change window rankings (each window-year still has
  ≥8 L9-class scenes' worth of revisit).
- 2026 monsoon onset was unusually late (Jul 20); W0 2026 coverage (99.3 %) is
  therefore not evidence that July is generally safe.
- S2 "native" valid fraction is computed at SCL's native 20 m (SCL has no 10 m
  product); reported as `s2_native10m_coverage_pct` per spec.
- `doy_std` uses raw day-of-year; leap year 2024 shifts July by one DOY — no
  material effect on rankings.

---

# L9-only sensitivity (2026-10-04 decision: production dataset = Landsat 9 ONLY)

Per-pixel QA_PIXEL/ST_B10 windows for exactly the 125 LC09 scenes in
`scene_list.csv` were re-read and re-accumulated (same grid, same masking, same
metrics; `src/v2/phase2/window_investigation_l9only.py` →
`window_metrics_l9only.csv`, 25 landsat rows). All 125 scenes processed, 0
read failures. Sentinel-2 rows are unchanged.

## L9-only decision matrix

| Window | Valid-ST coverage: mean / min (%) | Coverage CV (%) | Scenes/yr | doy_std | ≥2-obs coverage (%) | Median obs/cell | Mean LST (°C) |
|---|---|---|---|---|---|---|---|
| W0 Jul 1–30 | 78.1 / **42.5** | 28.6 | 4.6 | 6.8 | 23.0 | 1.2 | 37.6 |
| W1 Apr 1–Jun 30 | 99.98 / 99.98 | ~0.0 | 15.4 | 24.6 | 100 | 8.2 | 42.1 |
| W2 Mar 1–Jun 30 | 99.98 / 99.98 | ~0.0 | 20.4 | 33.3 | 100 | 11.0 | 39.7 |
| W3 Apr 1–Jul 31 | 99.98 / 99.98 | ~0.0 | 20.0 | 34.3 | 100 | 9.4 | 41.5 |
| W4 May 1–Jun 30 | 99.97 / **99.95** | 0.01 | 10.2 | **15.9** | 99.6 | 5.0 | **42.4** |

W0 L9-only per-year coverage: 88.9 / 89.0 / 70.9 / **42.5** / 99.2 %
(2022→2026) — fails the ≥90 % hard screen in **four of five years** (L8 was
masking July's fragility in the combined run).

## Headline answers (L9-only)

- **Does W4 still pass the hard screen? YES.** Per-year W4 valid-ST coverage:
  2022 = 99.98 %, 2023 = 99.95 %, 2024 = 99.98 %, 2025 = 99.98 %,
  2026 = 99.98 %; **minimum = 99.95 %** (2023). ≥90 % every year with huge margin.
- **W4 redundancy:** ≥2-observation coverage 99.98 / 98.18 / 99.76 / 99.96 /
  99.98 % (mean 99.6 %); median clear-observation count per cell = **5 / 4 / 5 /
  5 / 6** (2022→2026). Every cell sees ~4–6 valid L9 acquisitions — comfortably
  robust for a median composite.
- **Rankings: unchanged on every axis.** Coverage ordering among survivors is
  flat (~100 %); doy_std ordering W4 (15.9) < W1 (24.6) < W2 (33.3) ≈ W3 (34.3);
  LST ordering W4 (42.4) ≥ W1 (42.1) > W3 (41.5) > W2 (39.7) > W0 (37.6) °C —
  identical to the L8+L9 run. L8+L9 vs L9-only W4 LST differs by ≤1.5 °C/yr
  (mean −0.2 °C), i.e. sensor-mix bias is small.
- **Verdict: W4 STANDS.** L9-only halves W4's per-cell median (5 vs 10.4) and
  drops its ≥2-obs floor to 98.2 % (2023), but that floor is far above any
  fragility threshold and 4–6 obs/cell still yields a stable median. W1's extra
  redundancy (8.2 obs/cell, 100 % ≥2-obs) buys robustness against future L9
  outage/longevity risk at the cost of a 55 % wider seasonal spread (doy_std
  24.6 vs 15.9) and slightly lower LST (42.1 vs 42.4 °C, within inter-scene
  noise ~4.3 °C). Criteria 2 and 3 still both favour W4; recommend **W4 (May 1–
  Jun 30) with W1 (Apr 1–Jun 30) as the explicit fallback** if L9 revisit
  degrades (e.g., post-2026 constellation changes) below ~4 obs/cell median.

Sanity check (L9): study-area per-pixel valid-SR fraction vs scene
`eo:cloud_cover` for LC09 scenes — 0.378 at 34.8 % cloud vs 0.047 at 55.1 %
cloud (inverse relation holds, same pattern as the L8+L9 run).
