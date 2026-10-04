"""V2 Phase-2 audit 4/5: five-year July coverage audit (2022-2026).

Valid-pixel accounting per layer per year, masked by the study-area polygon on
each raster's native grid, plus V1 spatial-block coverage for L9 ST_B10,
valid-mask maps (PNG) and reusable uint8 mask GeoTIFFs.

Layer validity definitions (V1 semantics):
  L9 ST     finite ST_B10
  L9 SR     finite SR_B2
  L9 QA     QA_PIXEL finite AND bits 1-5 (dilated cloud, cirrus, cloud,
            cloud shadow, snow) all zero; bit 0 = fill is reported separately
  S2 10m    finite B2, B3, B4, B8 (all four)
  S2 20m    finite B5, B6, B7, B8A, B11, B12 (all six)
  LULC      finite AND != 0 (0 = nodata-like class; class histogram kept)

Spatial blocks replicate V1 exactly: bins from the Phase-4 combined dataset
row/col min/max (src/models/spatial_features.py::_compute_spatial_block_raster,
n_blocks=5). Block coverage is computed on the L9 30 m grid.

Outputs under data/v2/phase2/coverage/:
  per_year_coverage.csv      rows=year, valid_% and counts per layer
  block_coverage_l9_st.csv   rows=block, cols=year (+% and <50% flags)
  lulc_class_histogram.csv   class counts inside/outside study area
  maps/L9_ST_valid_YYYY.png, maps/S2_10m_valid_YYYY.png
  masks/*.tif                uint8 masks, 1=valid 0=invalid 255=outside study
  coverage.json              machine-readable summary + block bins + notes
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    N_BLOCKS,
    OUT_ROOT,
    RAW_ROOT,
    STUDY_AREA,
    V1_COMBINED_CSV,
    YEARS,
    dump_json,
    rel,
)

COV_DIR = OUT_ROOT / "coverage"
MAPS_DIR = COV_DIR / "maps"
MASKS_DIR = COV_DIR / "masks"

QA_CLOUD_BITS = 0b111110  # bits 1..5: dilated cloud, cirrus, cloud, shadow, snow
QA_FILL_BIT = 0b1

S2_10M_BANDS = ["B2", "B3", "B4", "B8"]
S2_20M_BANDS = ["B5", "B6", "B7", "B8A", "B11", "B12"]


def l9_path(year: int) -> Path:
    return RAW_ROOT / "landsat9" / f"{year}_07" / f"landsat9_{year}_07_composite_30m.tif"


def s2_path(year: int, res: str) -> Path:
    return RAW_ROOT / "sentinel2" / f"{year}_07" / f"sentinel2_{year}_07_{res}_composite.tif"


def lulc_path(year: int) -> Path:
    return RAW_ROOT / "lulc" / f"{year}_07" / f"lulc_{year}_10m.tif"


def study_mask(transform: Affine, height: int, width: int, geom) -> np.ndarray:
    mask = rasterio.features.rasterize(
        [(geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def finite_all_bands(src, band_indices) -> np.ndarray:
    """AND of finite() over the given 1-based band indices (band at a time)."""
    valid = np.ones((src.height, src.width), dtype=bool)
    for b in band_indices:
        arr = src.read(b)
        np.logical_and(valid, np.isfinite(arr), out=valid)
    return valid


def band_lookup(src) -> dict[str, int]:
    return {d: i + 1 for i, d in enumerate(src.descriptions)}


def valid_mask_for(path: Path, kind: str) -> tuple[np.ndarray, dict]:
    """Return (valid_mask, info) on the file's native grid."""
    with rasterio.open(path) as src:
        names = band_lookup(src)
        if kind == "l9_st":
            valid = np.isfinite(src.read(names["ST_B10"]))
            info = {}
        elif kind == "l9_sr":
            valid = np.isfinite(src.read(names["SR_B2"]))
            info = {}
        elif kind == "l9_qa":
            qa = src.read(names["QA_PIXEL"])
            fin = np.isfinite(qa)
            qai = np.where(fin, qa, 0).astype(np.int64)
            valid = fin & ((qai & QA_CLOUD_BITS) == 0)
            info = {
                "qa_finite": bool_array_stats(fin),
                "qa_fill_bit": bool_array_stats(fin & ((qai & QA_FILL_BIT) != 0)),
                "qa_clear_of_finite": float((valid & fin).sum() / max(fin.sum(), 1)),
            }
        elif kind == "s2_10m":
            valid = finite_all_bands(src, [names[b] for b in S2_10M_BANDS])
            b2 = src.read(names["B2"])
            b3 = src.read(names["B3"])
            b4 = src.read(names["B4"])
            b8 = src.read(names["B8"])
            others_fin = np.isfinite(b3) & np.isfinite(b4) & np.isfinite(b8)
            info = {
                "b2_zero_all_bands_finite": int(
                    (np.isfinite(b2) & (b2 == 0) & others_fin).sum()
                )
            }
        elif kind == "s2_20m":
            valid = finite_all_bands(src, [names[b] for b in S2_20M_BANDS])
            info = {}
        elif kind == "lulc":
            arr = src.read(1)
            valid = np.isfinite(arr) & (arr != 0)
            info = {"zero_count": int((np.isfinite(arr) & (arr == 0)).sum())}
        else:
            raise ValueError(kind)
    return valid, info


def bool_array_stats(mask: np.ndarray) -> int:
    return int(mask.sum())


def write_mask_tiff(out_path: Path, mask_valid: np.ndarray, study: np.ndarray,
                    transform: Affine, crs_str: str) -> None:
    """uint8 mask: 1=valid, 0=invalid, 255=outside study area."""
    arr = np.where(study, mask_valid.astype(np.uint8), 255).astype(np.uint8)
    crs = rasterio.crs.CRS.from_string(crs_str)
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="uint8", crs=crs, transform=transform, nodata=255,
        compress="deflate",
    ) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, "valid mask (1=valid, 0=invalid, 255=outside study area)")


def render_map(out_png: Path, mask_valid: np.ndarray, study: np.ndarray,
               transform: Affine, geom, title: str, stride: int = 3) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import mapping

    h, w = mask_valid.shape
    disp = np.full((h, w), 255, dtype=np.uint8)  # outside = white
    disp[study] = 1 - mask_valid[study].astype(np.uint8)  # valid->0 green, invalid->1 grey
    disp = disp[::stride, ::stride]
    extent = (
        transform.c,
        transform.c + w * transform.a,
        transform.f + h * transform.e,
        transform.f,
    )
    fig, ax = plt.subplots(figsize=(8, 8), dpi=130)
    cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#d9d9d9", "#ffffff"])
    bounds = [-0.5, 0.5, 1.5, 255.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(disp, cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
    for ring in _polygon_rings(mapping(geom)):
        xs, ys = ring
        ax.plot(xs, ys, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _polygon_rings(geom_mapping: dict):
    """Yield exterior ring coordinate arrays of a (multi)polygon mapping."""
    polys = []
    gt = geom_mapping["type"]
    if gt == "Polygon":
        polys = [geom_mapping]
    elif gt == "MultiPolygon":
        polys = [{"type": "Polygon", "coordinates": c} for c in geom_mapping["coordinates"]]
    for p in polys:
        for ring in p["coordinates"][:1]:  # exterior only
            arr = np.asarray(ring)
            yield arr[:, 0], arr[:, 1]


def v1_block_raster(height: int, width: int) -> tuple[np.ndarray, dict]:
    """Replicate src/models/spatial_features.py::_compute_spatial_block_raster."""
    sampled = pd.read_csv(V1_COMBINED_CSV, usecols=["row", "col"])
    rmin, rmax = int(sampled.row.min()), int(sampled.row.max())
    cmin, cmax = int(sampled.col.min()), int(sampled.col.max())
    row_bins = np.linspace(rmin, rmax + 1, N_BLOCKS + 1)
    col_bins = np.linspace(cmin, cmax + 1, N_BLOCKS + 1)
    all_rows = np.arange(height)
    all_cols = np.arange(width)
    row_block = np.clip(np.digitize(all_rows, row_bins) - 1, 0, N_BLOCKS - 1)
    col_block = np.clip(np.digitize(all_cols, col_bins) - 1, 0, N_BLOCKS - 1)
    block_id = row_block[:, None] * N_BLOCKS + col_block[None, :]
    meta = {
        "source": rel(Path(V1_COMBINED_CSV)),
        "sampled_row_min": rmin, "sampled_row_max": rmax,
        "sampled_col_min": cmin, "sampled_col_max": cmax,
        "row_bins": row_bins.tolist(),
        "col_bins": col_bins.tolist(),
        "n_blocks": N_BLOCKS,
        "block_id_layout": "block_id = row_block * n_blocks + col_block",
    }
    return block_id.astype(np.int32), meta


def main() -> int:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    study_gdf = gpd.read_file(STUDY_AREA)
    geom = study_gdf.geometry.union_all()
    crs_str = study_gdf.crs.to_string()

    # Cache study masks per native grid.
    grids: dict[str, dict] = {}
    for key, path in (("l9", l9_path(YEARS[0])), ("s2_10m", s2_path(YEARS[0], "10m")),
                      ("s2_20m", s2_path(YEARS[0], "20m"))):
        with rasterio.open(path) as src:
            grids[key] = {
                "transform": src.transform,
                "height": src.height,
                "width": src.width,
                "crs": src.crs.to_string(),
                "study": study_mask(src.transform, src.height, src.width, geom),
            }
            print(f"[coverage] study cells {key}: {int(grids[key]['study'].sum())}")

    # V1 block raster on the L9 grid.
    block_id, block_meta = v1_block_raster(grids["l9"]["height"], grids["l9"]["width"])

    per_year: list[dict] = []
    blocks: dict[int, dict] = {}
    notes: list[str] = []
    lulc_hist_rows: list[dict] = []

    for year in YEARS:
        rec = {"year": year}

        # ---- L9 ----
        p = l9_path(year)
        with rasterio.open(p) as src:
            l9_transform, l9_crs = src.transform, src.crs.to_string()
        study9 = grids["l9"]["study"]
        n_study9 = int(study9.sum())

        st_valid, _ = valid_mask_for(p, "l9_st")
        sr_valid, _ = valid_mask_for(p, "l9_sr")
        qa_valid, qa_info = valid_mask_for(p, "l9_qa")

        rec["l9_study_cells"] = n_study9
        rec["l9_st_valid_n"] = int((st_valid & study9).sum())
        rec["l9_st_valid_pct"] = 100.0 * rec["l9_st_valid_n"] / n_study9
        rec["l9_sr_valid_n"] = int((sr_valid & study9).sum())
        rec["l9_sr_valid_pct"] = 100.0 * rec["l9_sr_valid_n"] / n_study9
        rec["l9_qa_clear_n"] = int((qa_valid & study9).sum())
        rec["l9_qa_clear_pct"] = 100.0 * rec["l9_qa_clear_n"] / n_study9
        rec["l9_qa_fill_bit_n"] = qa_info["qa_fill_bit"]
        rec["l9_qa_clear_of_finite"] = qa_info["qa_clear_of_finite"]

        write_mask_tiff(MASKS_DIR / f"L9_ST_valid_{year}.tif", st_valid, study9,
                        l9_transform, l9_crs)
        render_map(MAPS_DIR / f"L9_ST_valid_{year}.png", st_valid, study9,
                   l9_transform, geom, f"Landsat-9 ST_B10 valid mask {year} July")

        # per-block coverage (denominator = study-area cells in block)
        for bid in range(N_BLOCKS * N_BLOCKS):
            b = blocks.setdefault(bid, {"block_id": bid})
            sel = (block_id == bid) & study9
            b["study_cells"] = int(sel.sum())
            b[f"valid_n_{year}"] = int((st_valid & sel).sum())
            b[f"valid_pct_{year}"] = (
                100.0 * b[f"valid_n_{year}"] / b["study_cells"]
                if b["study_cells"] else np.nan
            )

        # ---- S2 10m ----
        p = s2_path(year, "10m")
        with rasterio.open(p) as src:
            s2_transform, s2_crs = src.transform, src.crs.to_string()
        study2 = grids["s2_10m"]["study"]
        n_study2 = int(study2.sum())
        s2_valid, s2_info = valid_mask_for(p, "s2_10m")
        rec["s2_10m_study_cells"] = n_study2
        rec["s2_10m_valid_n"] = int((s2_valid & study2).sum())
        rec["s2_10m_valid_pct"] = 100.0 * rec["s2_10m_valid_n"] / n_study2
        rec["s2_10m_b2_zero_all_finite_n"] = s2_info["b2_zero_all_bands_finite"]
        # count zero-artifact pixels inside study area
        with rasterio.open(p) as src:
            names = band_lookup(src)
            b2 = src.read(names["B2"])
            others = np.isfinite(src.read(names["B3"])) & np.isfinite(src.read(names["B4"])) \
                & np.isfinite(src.read(names["B8"]))
        b2zero = np.isfinite(b2) & (b2 == 0) & others
        rec["s2_10m_b2_zero_all_finite_inside_study_n"] = int((b2zero & study2).sum())
        rec["s2_10m_valid_pct_excl_b2_zero"] = (
            100.0 * (rec["s2_10m_valid_n"] - rec["s2_10m_b2_zero_all_finite_inside_study_n"])
            / n_study2
        )
        if rec["s2_10m_b2_zero_all_finite_inside_study_n"] > 0:
            notes.append(
                f"{year} S2 10m B2: {rec['s2_10m_b2_zero_all_finite_inside_study_n']} pixels "
                f"({100.0*rec['s2_10m_b2_zero_all_finite_inside_study_n']/n_study2:.2f}% of study) "
                "have B2==0 with all other bands finite (zero-fill artifact swath)."
            )

        write_mask_tiff(MASKS_DIR / f"S2_10m_valid_{year}.tif", s2_valid, study2,
                        s2_transform, s2_crs)
        render_map(MAPS_DIR / f"S2_10m_valid_{year}.png", s2_valid, study2,
                   s2_transform, geom, f"Sentinel-2 10m valid mask {year} July")

        # ---- S2 20m ----
        p = s2_path(year, "20m")
        study20 = grids["s2_20m"]["study"]
        n_study20 = int(study20.sum())
        s220_valid, _ = valid_mask_for(p, "s2_20m")
        rec["s2_20m_study_cells"] = n_study20
        rec["s2_20m_valid_n"] = int((s220_valid & study20).sum())
        rec["s2_20m_valid_pct"] = 100.0 * rec["s2_20m_valid_n"] / n_study20

        # ---- LULC ----
        p = lulc_path(year)
        lulc_valid, lulc_info = valid_mask_for(p, "lulc")
        with rasterio.open(p) as src:
            arr = src.read(1)
        finite_arr = np.isfinite(arr)
        inside = finite_arr & study2
        outside = finite_arr & ~study2
        u_in, c_in = np.unique(arr[inside], return_counts=True)
        u_out, c_out = np.unique(arr[outside], return_counts=True)
        for cls, n in zip(u_in.tolist(), c_in.tolist()):
            lulc_hist_rows.append({"year": year, "class": cls, "region": "study",
                                   "count": int(n)})
        for cls, n in zip(u_out.tolist(), c_out.tolist()):
            lulc_hist_rows.append({"year": year, "class": cls, "region": "outside",
                                   "count": int(n)})
        rec["lulc_study_cells"] = n_study2
        rec["lulc_valid_n"] = int((lulc_valid & study2).sum())
        rec["lulc_valid_pct"] = 100.0 * rec["lulc_valid_n"] / n_study2
        rec["lulc_zero_in_study_n"] = int((finite_arr & (arr == 0) & study2).sum())
        rec["lulc_pct_if_zero_counted_valid"] = (
            100.0 * (rec["lulc_valid_n"] + rec["lulc_zero_in_study_n"]) / n_study2
        )

        per_year.append(rec)
        print(f"[coverage] {year} done: L9 ST {rec['l9_st_valid_pct']:.1f}%  "
              f"S2 10m {rec['s2_10m_valid_pct']:.1f}%", flush=True)

    per_year_df = pd.DataFrame(per_year)
    per_year_df.to_csv(COV_DIR / "per_year_coverage.csv", index=False)

    block_df = pd.DataFrame(blocks.values())
    block_df["row_block"] = block_df.block_id // N_BLOCKS
    block_df["col_block"] = block_df.block_id % N_BLOCKS
    for year in YEARS:
        block_df[f"below50_{year}"] = block_df[f"valid_pct_{year}"] < 50.0
    block_df.to_csv(COV_DIR / "block_coverage_l9_st.csv", index=False)

    pd.DataFrame(lulc_hist_rows).to_csv(COV_DIR / "lulc_class_histogram.csv", index=False)

    weak_2024 = sorted(block_df.loc[block_df.below50_2024, "block_id"].tolist())
    weak_2025 = sorted(block_df.loc[block_df.below50_2025, "block_id"].tolist())
    notes.append(
        f"Blocks below 50% L9 ST valid in 2024: {weak_2024}; in 2025: {weak_2025}."
    )

    dump_json(
        {
            "per_year": per_year,
            "blocks": block_df.to_dict("records"),
            "block_definition": block_meta,
            "notes": notes,
        },
        COV_DIR / "coverage.json",
    )
    print(f"[coverage] wrote {COV_DIR}/per_year_coverage.csv "
          f"({len(per_year_df)} rows), block_coverage_l9_st.csv ({len(block_df)} rows), "
          f"{len(list(MAPS_DIR.glob('*.png')))} maps, {len(list(MASKS_DIR.glob('*.tif')))} mask tifs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
