"""Independent spot-checks of the V2 Phase-2 audit numbers.

Recomputes key figures directly from the raw rasters with plain rasterio/numpy
(no audit helper code) and compares them against the audit CSVs. Fails loudly
on mismatch.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "v2" / "phase2"

study = gpd.read_file(RAW / "gis" / "study_area" / "study_area.geojson")
geom = study.geometry.union_all()

inv = pd.read_csv(OUT / "inventory" / "inventory_bands.csv")
invf = pd.read_csv(OUT / "inventory" / "inventory.csv")
nod = pd.read_csv(OUT / "audit" / "nodata_audit.csv")
cov = pd.read_csv(OUT / "coverage" / "per_year_coverage.csv")
grid = pd.read_csv(OUT / "audit" / "grid_audit.csv")

checks = []


def check(name: str, ok: bool, detail: str) -> None:
    checks.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def study_mask(src) -> np.ndarray:
    return rasterio.features.rasterize(
        [(geom, 1)], out_shape=(src.height, src.width),
        transform=src.transform, fill=0, dtype="uint8",
    ).astype(bool)


# --- Check 1: L9 2025 band stats (min/max/mean/nan of ST_B10 & SR_B2) -------
p = RAW / "landsat9" / "2025_07" / "landsat9_2025_07_composite_30m.tif"
with rasterio.open(p) as src:
    st = src.read(7)
    sr2 = src.read(1)
    tf = src.transform
for band_idx, arr in ((7, st), (1, sr2)):
    row = inv[(inv.file_path.str.contains("2025_07")) & (inv.band_index == band_idx)].iloc[0]
    fin = arr[np.isfinite(arr)]
    ok = (
        row["nan"] == int(np.isnan(arr).sum())
        and abs(row["min"] - float(fin.min())) < 1e-9
        and abs(row["max"] - float(fin.max())) < 1e-9
        and abs(row["mean"] - float(fin.mean())) < 1e-6
        and row["zeros"] == int((fin == 0).sum())
    )
    check(f"L9-2025 {row.band_name} stats vs inventory_bands.csv", ok,
          f"csv nan={row["nan"]} min={row["min"]:.4f} max={row["max"]:.2f} mean={row["mean"]:.4f} | "
          f"raw nan={int(np.isnan(arr).sum())} min={float(fin.min()):.4f} "
          f"max={float(fin.max()):.2f} mean={float(fin.mean()):.4f}")

# --- Check 2: L9 2025 coverage % vs CSV --------------------------------------
with rasterio.open(p) as src:
    sm = study_mask(src)
valid = np.isfinite(st) & np.isfinite(sr2)
pct = 100.0 * (valid & sm).sum() / sm.sum()
csv_pct = float(cov.loc[cov.year == 2025, "l9_st_valid_pct"].iloc[0])
check("L9-2025 ST valid % of study vs per_year_coverage.csv",
      abs(pct - csv_pct) < 1e-6, f"raw={pct:.6f} csv={csv_pct:.6f}")

# --- Check 3: QA clear mask for L9 2024 vs CSV --------------------------------
p24 = RAW / "landsat9" / "2024_07" / "landsat9_2024_07_composite_30m.tif"
with rasterio.open(p24) as src:
    qa = src.read(8)
    sm24 = study_mask(src)
qai = np.where(np.isfinite(qa), qa, 0).astype(np.int64)
clear = np.isfinite(qa) & ((qai & 0b111110) == 0)
pct = 100.0 * (clear & sm24).sum() / sm24.sum()
csv_pct = float(cov.loc[cov.year == 2024, "l9_qa_clear_pct"].iloc[0])
check("L9-2024 QA clear % of study vs per_year_coverage.csv",
      abs(pct - csv_pct) < 1e-6, f"raw={pct:.6f} csv={csv_pct:.6f}")

# --- Check 4: S2 2024 10m valid % + B2-zero artifact vs CSV -------------------
p = RAW / "sentinel2" / "2024_07" / "sentinel2_2024_07_10m_composite.tif"
with rasterio.open(p) as src:
    names = {d: i + 1 for i, d in enumerate(src.descriptions)}
    bands = {b: src.read(i) for b, i in names.items()}
    sm2 = study_mask(src)
valid2 = np.ones(next(iter(bands.values())).shape, bool)
for b in ("B2", "B3", "B4", "B8"):
    valid2 &= np.isfinite(bands[b])
pct = 100.0 * (valid2 & sm2).sum() / sm2.sum()
csv_pct = float(cov.loc[cov.year == 2024, "s2_10m_valid_pct"].iloc[0])
check("S2-2024 10m valid % of study vs per_year_coverage.csv",
      abs(pct - csv_pct) < 1e-6, f"raw={pct:.6f} csv={csv_pct:.6f}")
b2z = np.isfinite(bands["B2"]) & (bands["B2"] == 0) & np.isfinite(bands["B3"]) \
    & np.isfinite(bands["B4"]) & np.isfinite(bands["B8"])
n_in = int((b2z & sm2).sum())
csv_n = int(cov.loc[cov.year == 2024, "s2_10m_b2_zero_all_finite_inside_study_n"].iloc[0])
check("S2-2024 B2-zero artifact count vs CSV", n_in == csv_n,
      f"raw={n_in} csv={csv_n}")

# --- Check 5: S2 20m half-pixel shift -----------------------------------------
with rasterio.open(RAW / "landsat9" / "2026_07" / "landsat9_2026_07_composite_30m.tif") as s:
    tref = s.transform
p20 = RAW / "sentinel2" / "2022_07" / "sentinel2_2022_07_20m_composite.tif"
with rasterio.open(p20) as src:
    t = src.transform
dy_own = (t.f - tref.f) / (-t.e)
row = grid[grid.file_path.str.contains("sentinel2_2022_07_20m")].iloc[0]
check("S2-2022 20m half-pixel y-shift vs grid_audit.csv",
      abs(dy_own - (-0.5)) < 1e-9 and row["origin_dy_own_pixels"] == dy_own
      and row["classification"] == "RESAMPLING-REQUIRED",
      f"raw dy_own={dy_own:.6f} csv={row["origin_dy_own_pixels"]} class={row["classification"]}")

# --- Check 6: LULC 2025 valid % (finite & != 0) --------------------------------
p = RAW / "lulc" / "2025_07" / "lulc_2025_10m.tif"
with rasterio.open(p) as src:
    arr = src.read(1)
    sml = study_mask(src)
valid_l = np.isfinite(arr) & (arr != 0)
pct = 100.0 * (valid_l & sml).sum() / sml.sum()
csv_pct = float(cov.loc[cov.year == 2025, "lulc_valid_pct"].iloc[0])
check("LULC-2025 valid % of study vs per_year_coverage.csv",
      abs(pct - csv_pct) < 1e-6, f"raw={pct:.6f} csv={csv_pct:.6f}")

# --- Check 7: nodata audit NaN count for a 20m band ----------------------------
row = nod[(nod.file_path.str.contains("sentinel2_2023_07_20m")) & (nod.band_name == "B11")].iloc[0]
with rasterio.open(RAW / "sentinel2" / "2023_07" / "sentinel2_2023_07_20m_composite.tif") as src:
    b11 = src.read(6)  # B11 is band 6 in the 20 m stack
check("S2-2023 20m B11 NaN count vs nodata_audit.csv",
      row["nan"] == int(np.isnan(b11).sum()),
      f"csv={row["nan"]} raw={int(np.isnan(b11).sum())}")

# --- Check 8: mask GeoTIFF round-trip ------------------------------------------
mt = OUT / "coverage" / "masks" / "L9_ST_valid_2025.tif"
with rasterio.open(mt) as src:
    m = src.read(1)
    nd = src.nodata
with rasterio.open(RAW / "landsat9" / "2025_07" / "landsat9_2025_07_composite_30m.tif") as src:
    st25 = src.read(7)
ok = (
    nd == 255
    and int((m == 1).sum()) == int((np.isfinite(st25) & (m != 255)).sum())
    and bool((m[np.isnan(st25)] != 1).all())
)
check("mask GeoTIFF L9_ST_valid_2025 nodata/semantics", ok,
      f"nodata={nd} valid_px={int((m == 1).sum())} raw_finite={int(np.isfinite(st25).sum())}")

# --- Check 9: sha256 spot re-hash of two files ----------------------------------
import hashlib
for fp in ["data/raw/gis/study_area/study_area.geojson",
           "data/raw/landsat9/2023_07/landsat9_2023_07_composite_30m.tif"]:
    h = hashlib.sha256()
    with open(ROOT / fp, "rb") as f:
        while (c := f.read(1 << 20)):
            h.update(c)
    row = invf[invf.file_path == fp].iloc[0]
    check(f"sha256 {Path(fp).name}", h.hexdigest() == row["sha256"],
          f"recomputed={h.hexdigest()[:16]}... csv={row["sha256"][:16]}...")

# --- Check 10: block coverage spot recompute (block 7, 2025) -------------------
blk = pd.read_csv(OUT / "coverage" / "block_coverage_l9_st.csv")
sampled = pd.read_csv(ROOT / "data" / "processed" / "phase4" / "tables"
                      / "combined_urban_environmental_dataset.csv", usecols=["row", "col"])
rmin, rmax, cmin, cmax = sampled.row.min(), sampled.row.max(), sampled.col.min(), sampled.col.max()
rbins = np.linspace(rmin, rmax + 1, 6)
cbins = np.linspace(cmin, cmax + 1, 6)
H, W = st25.shape
rid = np.clip(np.digitize(np.arange(H), rbins) - 1, 0, 4)
cid = np.clip(np.digitize(np.arange(W), cbins) - 1, 0, 4)
bid = rid[:, None] * 5 + cid[None, :]
with rasterio.open(RAW / "landsat9" / "2025_07" / "landsat9_2025_07_composite_30m.tif") as src:
    sm = study_mask(src)
sel = (bid == 7) & sm
pct = 100.0 * (np.isfinite(st25) & sel).sum() / sel.sum()
csv_pct = float(blk.loc[blk.block_id == 7, "valid_pct_2025"].iloc[0])
check("block 7 (2025) valid % vs block_coverage_l9_st.csv",
      abs(pct - csv_pct) < 1e-6, f"raw={pct:.6f} csv={csv_pct:.6f}")

n_fail = sum(1 for _, ok, _ in checks if not ok)
print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
raise SystemExit(1 if n_fail else 0)
