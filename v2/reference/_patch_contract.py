"""Update v2/src/v2/CONTRACT.md for the two contract changes."""
from pathlib import Path

p = Path("v2/src/v2/CONTRACT.md")
s = p.read_text(encoding="utf-8")

# ---------- 1) header status ----------
old = """Status: **CODE COMPLETE, EXECUTION PENDING** (waits for GEE W4 exports into
`data/raw/` and per-phase user-triggered runs). Unit-tested only on
synthetic fixtures (`tests/v2/test_phase34_synthetic.py`, fixtures under
`data/_smoketest/p34/`). 36/36 tests pass; every module `py_compile`-clean."""
new = """Status: **CODE COMPLETE, EXECUTION PENDING** (waits for per-phase user-triggered
runs; real W4 rasters are already ingested under `data/raw/`). Unit-tested on
synthetic fixtures (`tests/test_phase34_synthetic.py`, fixtures under
`data/_smoketest/p34/`). 49/49 tests pass; every module `py_compile`-clean.
Two user-approved contract changes are incorporated (2026-10-05):
**(C1) pinned V1 5×5 spatial-block geometry** (§7.6, §2/§9) and
**(C2) S2 B2>0 hard value-stack invariant** (§3, §6, §8)."""
assert old in s, "header anchor"
s = s.replace(old, new)

# ---------- 2) section 2: add pinned-blocks pointer after grid paragraph ----------
old = """- **Authoritative grid**: read at runtime from
  `data/raw/v2/landsat9/2026_05_06/landsat9_2026_05_06_composite_30m.tif`
  (2026-reference convention, same as V1). `--grid-file` overrides (tests)."""
new = """- **Authoritative grid**: read at runtime from
  `data/raw/landsat9/2026_05_06/landsat9_2026_05_06_composite_30m.tif`
  (2026-reference convention, same as V1). `--grid-file` overrides (tests).
  Verified identical to the V1 grid (1874×1768, EPSG:4326, origin
  76.8329062507/28.8846991402, res 0.00026949458523585647) — row/col pinning
  of the spatial-block geometry (§7.6) is therefore geographically exact."""
assert old in s, "grid anchor"
s = s.replace(old, new)

# ---------- 3) section 3: B2 hard invariant ----------
old = """- S2-10m valid = `finite(B2..B8) & (B2 > 0) & (VALID_COUNT ≥ 1)`"""
new = """- S2-10m valid = `finite(B2..B8) & (B2 > 0) & (VALID_COUNT ≥ 1)`
  **HARD INVARIANT (C2)**: a native pixel with `B2==0` must not enter the
  S2-derived value stack even if all other bands are finite. Phase 3 NaNs the
  invalid native pixels *before* index computation/aggregation (build_core),
  so NaN-safe area-weighted aggregation can never use them in NDVI/PVC or any
  30 m product. Phase 3 preflight reports the exclusion accounting per year
  (`finite == valid + B2==0`, exact identity, gate; crosscheckable against a
  prior audit via `--crosscheck-audit`); Phase 4 verify asserts every feature
  row sits on a 30 m cell with ≥1 valid native B2>0 pixel (leakage gate)."""
assert old in s, "b2 anchor"
s = s.replace(old, new)

# ---------- 4) section 6: value-stack exclusion note ----------
old = """30 m S2/LULC valid masks = "any valid native pixel overlaps the cell"
(reprojected validity-average > 1e-6). The B2-zero artifact therefore does
**not** NaN-out partial 30 m products; it is a coverage gate, per the
Phase-2 contract (F3 note: V1's finite test counted artifact pixels valid)."""
new = """30 m S2/LULC valid masks = "any valid native pixel overlaps the cell"
(reprojected validity-average > 1e-6). Under (C2) the B2-zero artifact is
additionally excluded from the *values* feeding those products: a fully-
artifact 30 m cell becomes NaN (out of the domain), a mixed cell aggregates
only its valid native pixels. `build_manifest.json` records per-year
`{finite_px, b2_zero_px, valid_excluding_b2_zero, identity_holds}`."""
assert old in s, "sec6 anchor"
s = s.replace(old, new)

# ---------- 5) section 7.6 rewrite ----------
old_start = s.index("### 7.6 Sampling, blocks, domain")
old_end = s.index("## 8. Phase-4 verification gates")
new76 = """### 7.6 Sampling, blocks, domain
- Domain = pixels where ALL raw predictors are finite, **clipped to the
  study area** (`--study-area`, rasterized pixel-centre rule; with real GEE
  products the predictors are already NaN outside, the clip makes it
  explicit and matches V1's never-outside-study Phase-3 valid mask).
- Sampling (V1 Tier-2): per-year independent, up to `--max-samples`
  (default 150,000), `numpy.default_rng(42)` choice without replacement,
  then sorted. On the real grid this triggers only if a year exceeds the cap.
- `spatial_block_id`: **PINNED V1 geometry (C1)** — `src/v2/common.py`
  `PINNED_ROW_BINS = (1.0, 354.2, 707.4, 1060.6, 1413.8, 1767.0)`,
  `PINNED_COL_BINS = (0.0, 374.6, 749.2, 1123.8000000000002, 1498.4, 1873.0)`
  (the V1 `_compute_spatial_block_raster` formula `linspace(min, max+1, 6)`,
  `digitize−1`, clip, `id = row_band·5 + col_band` applied to the pooled
  extent of the table that trained the frozen model), `HOLDOUT_BLOCKS =
  (2, 9, 15, 23)` fixed. Bins NEVER derive from the V2 sample — block ids
  are identical for the full grid and any subset of rows (sampling
  independence, tested). Full provenance + per-block geographic bboxes +
  the honest formula-reproduction verification:
  `data/phase2/spatial_blocks_manifest.json`.
- dtypes: float32 everywhere except the 12 distance columns (float64, V1
  parity: ~4.5e4 m magnitudes would breach the V1 G1 1e-4 bound in float32).

"""
s = s[:old_start] + new76 + s[old_end:]

# ---------- 6) section 8 gates list ----------
old = """  4. per-year coverage of the Phase-3 valid masks within the study area meets
     the Phase-2 gates (L9 ST >= 90%, S2-10m >= 85%, LULC >= 80%);
  5. spatial_block_id == V1 tiling recomputed from pooled sampled extent;"""
new = """  4. per-year coverage of the Phase-3 valid masks within the study area meets
     the Phase-2 gates (L9 ST >= 90%, S2-10m >= 85%, LULC >= 80%);
  4b. hard B2 invariant: every feature row's 30 m cell contains >= 1 valid
     native S2-10m pixel under the B2>0 guard (no S2-derived feature may
     contain a B2==0 pixel contribution);
  5. spatial_block_id == PINNED V1 bins (independent of the sample);"""
assert old in s, "gates anchor"
s = s.replace(old, new)

# ---------- 7) section 9: replace block-bins deviation, add C1/C2 notes ----
old = """4. **Block bins provenance.** V1's bins came from its Phase-4 *sampled* table
   extent; V2 recomputes the same formula on its own sampled extent
   (pooled across years). If the sampled extents differ, block IDs shift
   relative to V1's map — validation protocols that reuse V1's locked blocks
   {2,9,15,23} geographically would need the V1 bins pinned explicitly
   (one-line change: pass V1's sampled min/max to `compute_block_raster`)."""
new = """4. **Block bins provenance (RESOLVED by C1).** V2 pins the V1 pooled bins
   (§7.6) instead of deriving bins from its own sample. Verified anomaly
   (documented in `data/phase2/spatial_blocks_manifest.json`): the V1
   production table's own `spatial_block_id` column is per-year Phase-3
   output — 2022-2025 reproduce exactly from per-year sample extents, but
   2026 carries 581 rows from the original 2-year sample's bins, and 22,251
   pixels have year-inconsistent block ids. **No single bin set can
   reproduce that column 100%**; the pinned pooled bins are the frozen
   production semantics (the formula that generated the production fullgrid
   block-aware features), reproduce 88.3% of the column, and preserve
   91.7% / 97.9% / 90.6% / 98.4% of locked blocks 2/9/15/23 pixels (locked-set
   agreement 95.69%). Holdout ids are the fixed constants (2, 9, 15, 23) —
   NOT `occupied[2::6]` under the pooled bins (which would give [3,11,17])."""
assert old in s, "dev4 anchor"
s = s.replace(old, new)

# add C2 deviation note after deviation 7
old = """7. **Met window** is W4 (May 1–Jun 30, hours 04–06 UTC) per the Phase-2
   contract; V1 used July 1–30. Same method, different window (by design)."""
new = """7. **Met window** is W4 (May 1–Jun 30, hours 04–06 UTC) per the Phase-2
   contract; V1 used July 1–30. Same method, different window (by design).
8. **B2==0 pixels change value, not just coverage (C2).** V1 counted
   artifact pixels as valid (F3) and let them into 30 m means. V2 excludes
   them from the value stack by construction; mixed 30 m cells therefore
   differ slightly from a V1-style leaky mean. This is the user-mandated
   invariant; residual difference vs V1 exists only where artifacts mix
   into cells (2022: 84,909 px, 2026: 7,867 px — real audit numbers)."""
assert old in s, "dev7 anchor"
s = s.replace(old, new)

# ---------- 8) usage section flags ----------
old = """PYTHONPATH=src .venv/Scripts/python.exe -m src.v2.phase4.assemble_features \\
    --phase3-root data/v2/phase3 --out-root data/v2/phase4"""
if old not in s:
    old = """PYTHONPATH=src .venv/Scripts/python.exe -m src.v2.phase4.assemble_features \\
        --phase3-root data/v2/phase3 --out-root data/v2/phase4"""
    new = """PYTHONPATH=src .venv/Scripts/python.exe -m src.v2.phase4.assemble_features \\
        --phase3-root data/v2/phase3 --out-root data/v2/phase4
        # add --study-area data/gis/study_area/study_area.geojson (default)"""
    assert old in s, "usage assemble anchor"
    s = s.replace(old, new)

s = s.replace("""# 0. after GEE downloads land in data/raw/v2/:
PYTHONPATH=src .venv/Scripts/python.exe -m src.v2.phase3.verify_inputs \\
    --data-root data/raw/v2                       # writes w4_input_audit.json""",
"""# 0. preflight (writes data/phase2/w4_input_audit.json; add --crosscheck-audit
#    data/phase2/w4_input_audit.json on re-runs to pin the B2==0 accounting):
PYTHONPATH=src .venv/Scripts/python.exe -m src.v2.phase3.verify_inputs \\
    --data-root data/raw/v2""")

p.write_text(s, encoding="utf-8")
print("CONTRACT.md updated")
