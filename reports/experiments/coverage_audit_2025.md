# 2025 Composite Spatial Coverage Audit

**Date:** 2026-10-02 · **Scope:** July 2025 Landsat 9 and Sentinel-2 composites (with 2022–2024/2026 as reference) · **Nature:** read-only — no rasters, pipeline code, or models were modified.

## Method

- **Spatial blocks:** exact replica of `src/preprocessing/feature_table.py:compute_spatial_block_ids` — a regular 5×5 binning on the 1768×1874 Landsat grid, `block_id = row_block * 5 + col_block`. Blocks {0, 4, 5, 20, 21} are at 0% valid in **every** year: they are bounding-box corners outside the Delhi boundary. The remaining **20 blocks are the "occupied" blocks** used by Phase 5/6 spatial cross-validation.
- **Landsat 9:** valid = finite `ST_B10` on the native 30 m grid.
- **Sentinel-2:** valid = finite `B8` at 10 m, aggregated to the 30 m grid by NaN-aware averaging — identical semantics to Phase 3 `align.py` (a 30 m cell is valid iff ≥1 underlying 10 m pixel is valid). Percentages on the 30 m grid are therefore slightly higher than native-resolution counts (more sub-pixel gaps tolerated).
- **Clustering metrics:** Moran's I of the valid indicator (8-neighbour), 8-connected missing-cluster count, largest missing cluster (px/ha at ≈787 m²/px).
- Audit script: temporary, outside the repo; outputs are this report, `coverage_audit_2025_blocks.csv`, `coverage_audit_2025_maps.png`, `coverage_audit_2025_blocks.png`.

## Table 1 — Overall coverage and clustering, all years

Blocks-below thresholds include the 5 structural outside-Delhi blocks; subtract 5 for occupied-block counts.

| Dataset-year | Valid % | Moran's I | Missing clusters | Largest missing (ha) | Blocks <50% | Blocks <10% |
|---|---|---|---|---|---|---|
| L9 2022 | 52.6 | 0.980 | 1,519 | 47,670 | 10 | 6 |
| L9 2023 | 51.8 | 0.984 | 841 | 47,162 | 13 | 6 |
| L9 2024 | 40.9 | 0.975 | 491 | 83,697 | 14 | 6 |
| **L9 2025** | **25.1** | 0.969 | 848 | **191,898** | **20** | **10** |
| L9 2026 | 57.2 | 0.995 | 72 | 37,159 | 10 | 5 |
| S2 2022 | 49.8 | 0.977 | 784 | 47,670 | 10 | 6 |
| S2 2023 | 44.7 | 0.984 | 366 | 63,349 | 13 | 7 |
| S2 2024 | 25.2 | 0.957 | 833 | 144,904 | 21 | 8 |
| **S2 2025** | **15.7** | **0.961** | 574 | **210,469** | **23** | **14** |
| S2 2026 | 57.8 | 0.996 | 5 | 36,655 | 10 | 5 |

Moran's I is high (0.96–1.00) in **all** years — valid coverage is always spatially smooth — so it does not discriminate; the discriminators are the largest-missing-cluster size and the per-block distribution.

## Table 2 — 2025 per-block valid coverage (%)

Occupied = block has Delhi-area cells. Reference: L9 2022/2026 per-block.

| Block | Cells | L9 2025 | S2 2025 | L9 2022 | L9 2026 |
|---|---|---|---|---|---|
| 0 | 132,702 | 0.0 (outside) | 0.0 | 0.0 | 0.0 |
| 1 | 132,724 | 16.7 | 5.3 | 42.1 | 42.5 |
| 2 | 132,724 | 47.4 | 61.9 | 84.6 | 84.6 |
| 3 | 132,681 | 28.8 | 38.2 | 50.9 | 50.4 |
| 4 | 132,681 | 0.0 (outside) | 0.0 | 0.0 | 0.0 |
| 5 | 132,681 | 0.0 (outside) | 0.0 | 0.0 | 0.0 |
| 6 | 132,681 | 7.7 | 0.0 | 87.4 | 87.1 |
| 7 | 132,681 | 4.7 | 36.7 | 99.2 | 100.0 |
| 8 | 132,681 | 37.2 | 21.4 | 92.8 | 93.7 |
| 9 | 132,681 | 2.0 | 0.1 | 27.1 | 32.1 |
| 10 | 132,681 | 15.0 | 0.0 | 27.3 | 26.8 |
| 11 | 132,681 | 31.0 | 0.0 | 93.5 | 97.4 |
| 12 | 132,681 | 30.2 | 44.4 | 95.6 | 100.0 |
| 13 | 132,681 | 59.3 | 20.5 | 92.1 | 99.5 |
| 14 | 132,681 | 28.9 | 10.2 | 75.3 | 84.4 |
| 15 | 132,681 | 23.5 | 0.0 | 59.1 | 68.8 |
| 16 | 132,681 | 50.6 | 0.0 | 69.2 | 78.5 |
| 17 | 132,681 | 88.5 | 47.4 | 80.3 | 92.5 |
| 18 | 132,681 | 76.7 | 25.8 | 76.5 | 100.0 |
| 19 | 132,681 | 4.4 | 5.7 | 52.7 | 73.6 |
| 20 | 132,702 | 0.0 (outside) | 0.0 | 0.0 | 0.0 |
| 21 | 132,702 | 0.0 (outside) | 0.0 | 0.0 | 0.0 |
| 22 | 132,702 | 16.6 | 15.0 | 18.5 | 19.7 |
| 23 | 132,702 | 56.7 | 58.3 | 83.6 | 88.1 |
| 24 | 132,702 | 1.2 | 1.9 | 7.8 | 12.3 |

## Findings

1. **Yes — the 2025 losses are strongly spatially concentrated, not randomly scattered.** L9 2025: **15 of 20 occupied blocks are below 50% valid** (2022: 5, 2026: 5), and 5 occupied blocks are nearly wiped out (<10%: blocks 6, 7, 9, 19, 24). S2 2025 is worse: 18 of 20 occupied blocks below 50%, 9 occupied blocks below 10%.
2. **Yes — missing pixels form one dominant contiguous deck.** The largest connected missing region is 191,898 ha (L9) / 210,469 ha (S2) — **5× the 2022/2026 norm (~37–48k ha)** and 74–81% of the entire grid in a single cluster. This is a coherent monsoon cloud deck, not dispersed dropout.
3. **The 2025 survivors are in the same place for both sensors:** valid pixels concentrate in the north-east/central-east — blocks 2, 13, 16, 17, 18, 23 (L9) and 2, 3, 7, 12, 17, 23 (S2). Both sensors independently agree, which confirms an atmospheric cause rather than a sensor artifact.
4. **West/south-west Delhi is effectively unobserved in 2025** (blocks 6, 9, 19, 24 at 0–8% for L9; 9 at 0.05% and 24 at 1.9% for S2). 2024 shows the same pattern at half strength (largest missing cluster 84–145k ha), so coverage degrades monotonically 2023 → 2024 → 2025.
5. Both 2025 composites are internally consistent with the GEE run's warning (~829k valid px for L9; local count 831,154) — the low coverage is real, not a download/processing defect.

## Implications for the 5-year dataset (recorded, no action taken)

- The 2025 annual sample (≤150k rows) will be drawn almost entirely from ~10–12 blocks in NE/central-east Delhi instead of 20. Spatial-block CV folds for 2025 will have little holdout power in W/SW Delhi, weakening exactly the geographic-generalization goal of adding years.
- Per-year quartile thresholds for 2025 will describe the observed (NE-biased) subset, not Delhi as a whole; cross-year class comparisons involving 2025 carry that bias.
- 2023 is usable (coverage ≈ 2022); 2024 is marginal; 2025 is the problem year. Alternatives for a later decision: accept 2025 with an explicit coverage caveat; widen the 2025 acquisition window (documented deviation); or drop 2025 and use 2022/2023/2024/2026. Data files are unchanged either way.

## Provenance notes

- **S2 20m 2023–2025 exports are Float32** (2022/2026 are Float64). Harmless: Phase 3 reads bands as Float64; value ranges verified sane. Recorded in `dataset_metadata.csv`.
- S2 2024/2025 20m rasters contain SCL class 2 (dark feature/shadow) — a valid clear-sky class; the mask only excludes 3/8/9/10.
- Native-resolution 2025 valid counts (L9 831,154 px; S2 4,428,045 px @10m / 1,107,815 px @20m) are recorded in `dataset_metadata.csv`; percentages in Table 1 use the 30 m-grid semantics defined above.

## Artifacts

- `reports/experiments/coverage_audit_2025_maps.png` — valid/missing maps per year with the 5×5 block grid and ids.
- `reports/experiments/coverage_audit_2025_blocks.png` — 2025 per-block valid % vs 2022/2026 overall reference lines.
- `reports/experiments/coverage_audit_2025_blocks.csv` — full per-block table for all 10 dataset-years.
