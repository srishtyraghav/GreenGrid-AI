"""Shared constants and helpers for the V2 Phase 3/4 implementation.

Everything here is a pure function of the V2 Phase-2 contract
(``data/v2/phase2/`` + ``reports/phase2_dataset_report.md`` §5–8) and the
ported V1 semantics documented in ``src/v2/CONTRACT.md``. No module in this
package may read V1 rasters except the read-only Overpass-JSON constraint
layers explicitly allowed by the contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_V2 = PROJECT_ROOT / "data" / "raw"
PHASE2_ROOT = PROJECT_ROOT / "data" / "phase2"
STUDY_AREA_DEFAULT = PROJECT_ROOT / "data" / "gis" / "study_area" / "study_area.geojson"

# V2 constraint vectors (EPSG:4326 GeoJSON, fetched in Phase 2).
V2_BUILDINGS_GEOJSON = PHASE2_ROOT / "constraints" / "osm_buildings_delhi.geojson"
V2_WATER_GEOJSON = PHASE2_ROOT / "constraints" / "osm_water_delhi.geojson"
V2_ROAD_SURFACES_GEOJSON = PHASE2_ROOT / "constraints" / "osm_road_surfaces_delhi.geojson"
# V2's own OSM context layers (CONTEXT_LAYERS_PROVENANCE.md): roads is the
# FULL network; vegetation/landuse are broader unions than V1 — feature
# parity filters live in build_core (V1_* constants below).
V2_ROADS_GEOJSON = PHASE2_ROOT / "constraints" / "osm_roads_delhi.geojson"
V2_VEGETATION_GEOJSON = PHASE2_ROOT / "constraints" / "osm_vegetation_delhi.geojson"
V2_LANDUSE_GEOJSON = PHASE2_ROOT / "constraints" / "osm_landuse_delhi.geojson"

MET_CSV_DEFAULT = PHASE2_ROOT / "met" / "met_superset_hourly_2022_2026.csv"

SCHEMA_JSON_DEFAULT = (
    PROJECT_ROOT
    / "reference"
    / "phase5_primary_xgb_3class_features.json"
)

GRID_FILE_DEFAULT = (
    DATA_RAW_V2 / "landsat9" / "2026_05_06" / "landsat9_2026_05_06_composite_30m.tif"
)

# ---------------------------------------------------------------------------
# Temporal contract
# ---------------------------------------------------------------------------
YEARS = (2022, 2023, 2024, 2025, 2026)
WINDOW_TAG = "05_06"  # W4 = May 1 – Jun 30, one common window for all years
MET_WINDOW_HOURS_UTC = (4, 5, 6)  # Landsat morning descending pass bracket

# ---------------------------------------------------------------------------
# Band layouts (gee/v2/01..03; GEE exports carry no band descriptions, so the
# positional contract below is authoritative)
# ---------------------------------------------------------------------------
L9_BANDS = (
    "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
    "ST_B10", "QA_PIXEL", "QA_CLEAR_COUNT",
)
S2_10M_BANDS = ("B2", "B3", "B4", "B8", "VALID_COUNT")
S2_20M_BANDS = ("B5", "B6", "B7", "B8A", "B11", "B12", "SCL", "VALID_COUNT")
LULC_BANDS = ("lulc",)

L9_PATH = lambda data_root, y: data_root / "landsat9" / f"{y}_{WINDOW_TAG}" / f"landsat9_{y}_{WINDOW_TAG}_composite_30m.tif"
S2_10M_PATH = lambda data_root, y: data_root / "sentinel2" / f"{y}_{WINDOW_TAG}" / f"sentinel2_{y}_{WINDOW_TAG}_10m_composite.tif"
S2_20M_PATH = lambda data_root, y: data_root / "sentinel2" / f"{y}_{WINDOW_TAG}" / f"sentinel2_{y}_{WINDOW_TAG}_20m_composite.tif"
LULC_PATH = lambda data_root, y: data_root / "lulc" / f"{y}_{WINDOW_TAG}" / f"lulc_{y}_{WINDOW_TAG}_10m.tif"

# ---------------------------------------------------------------------------
# Physical / semantic constants
# ---------------------------------------------------------------------------
TARGET_CRS = CRS.from_epsg(4326)
LST_SCALE = 0.00341802          # USGS C2-L2 ST scale
LST_OFFSET = 149.0              # K
KELVIN_TO_CELSIUS = -273.15

NDVI_SOIL_REFERENCE = 0.05      # V1 Phase 4 PVC lower end-member
NDVI_VEG_REFERENCE = 0.80       # V1 Phase 4 PVC upper end-member

DIST_RESOLUTION_M_APPROX = 30.0  # V1 pixel-Euclidean distance convention (×30 m)

IDW_COS_LAT_DEG = 28.64         # V1 matched IDW variant longitude scaling
IDW_EPS2 = 1e-6                 # epsilon inside the squared distance

MET_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "shortwave_radiation",
    "soil_moisture_0_to_7cm",
)
MET_FEATURE_NAMES = (
    "met_t2m_c", "met_rh_pct", "met_wind_kmh",
    "met_precip_mm", "met_ssr_wm2", "met_swc_m3m3",
)

# Per-year quality gates (Phase-2 contract §quality gates).
GATES = {"l9_st": 0.90, "s2_10m": 0.85, "lulc": 0.80}

N_SPATIAL_BLOCKS = 5            # V1 25-block tiling

# ---------------------------------------------------------------------------
# PINNED V1 spatial-block geometry (user-approved contract change, 2026-10-05)
#
# Source: the table that trained the frozen V1 production model,
#   v1/data/processed/lulc_outputs/combined_urban_environmental_dataset.csv
#   (749,998 rows; sha256 90176c6f5caeb5996914bbb085f67b160be389fcd67e004cd9a
#    785e5774b85f3)
# Bins are the V1 formula (v1/src/models/spatial_features.py
# _compute_spatial_block_raster): linspace(sampled_min, sampled_max+1, 6)
# applied to the production table's POOLED row/col extent
# (rows 1..1766, cols 0..1872). Verified: the V2 W4 authoritative grid and
# the V1 grid are identical (1874x1768, EPSG:4326, origin
# 76.8329062507/28.8846991402, res 0.00026949458523585647), so row/col
# pinning is geographically exact.
#
# NOTE (documented in data/phase2/spatial_blocks_manifest.json): the V1
# table's own spatial_block_id COLUMN is per-year Phase-3 output (2022-2025
# reproduce exactly from per-year sample extents; 2026 carries 581 rows from
# the original 2-year sample's bins). No single bin set can reproduce that
# column 100% (22,251 pixels have year-inconsistent block ids). The pinned
# pooled bins are the frozen production semantics (the formula that generated
# the production fullgrid block-aware features) and preserve 90.6-98.4% of
# each locked block's pixels. Holdout ids are FIXED constants (NOT recomputed
# via occupied[2::6], which under the pooled bins would give [3,11,17]).
# ---------------------------------------------------------------------------
PINNED_ROW_BINS = (1.0, 354.2, 707.4, 1060.6, 1413.8, 1767.0)
PINNED_COL_BINS = (0.0, 374.6, 749.2, 1123.8000000000002, 1498.4, 1873.0)
PINNED_BLOCKS_SOURCE = (
    "v1/data/processed/lulc_outputs/combined_urban_environmental_dataset.csv"
)
PINNED_BLOCKS_SOURCE_SHA256 = (
    "90176c6f5caeb5996914bbb085f67b160be389fcd67e004cd9a785e5774b85f3"
)
HOLDOUT_BLOCKS = (2, 9, 15, 23)   # V1 locked geographic holdout (frozen)
MAX_SAMPLES_PER_YEAR = 150_000  # V1 Tier-2 sampling budget
RANDOM_SEED = 42                # V1 sampling seed

# V1 OSM landuse class table (src/preprocessing/vector_raster.py).
LANDUSE_CLASS_PRIORITY = (
    "park", "forest", "grass", "commercial",
    "industrial", "residential", "retail", "farmland",
)
LANDUSE_CODE_TO_CLASS = {i + 1: c for i, c in enumerate(LANDUSE_CLASS_PRIORITY)}
LANDUSE_CODE_TO_CLASS[0] = "unclassified_background"
MORPHOLOGY_LANDUSE_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8)
MORPHOLOGY_RADII_M = (50, 100, 250, 500)
MORPHOLOGY_RESOLUTION_M = 30.0

# V1-parity filters for the V2 context layers, from the verbatim Overpass
# queries in V1 src/data_collection/download_gis_data.py (verified against
# the actual V1 files, read-only):
#   roads:        way["highway"~"^(motorway|trunk|primary)$"]  -> the V1 file
#                 contains exactly {motorway, trunk, primary} (no _link, no
#                 secondary/tertiary); V1's preprocessing filter
#                 (motorway..tertiary) was a superset on that file.
#   vegetation:   leisure~(park|garden|nature_reserve) OR
#                 landuse~(forest|grass|meadow) OR
#                 natural~(wood|scrub|grassland)   (V1 burned the whole file)
#   landuse:      landuse~(residential|commercial|industrial|retail|park|
#                 forest|farmland) — direct tag only; the leisure/natural
#                 fallbacks in classify_landuse could never fire on V1 data.
V1_ROAD_HIGHWAY_CLASSES = ("motorway", "trunk", "primary")
V1_VEGETATION_TAGS = {
    "leisure": ("park", "garden", "nature_reserve"),
    "landuse": ("forest", "grass", "meadow"),
    "natural": ("wood", "scrub", "grassland"),
}
V1_LANDUSE_VALUES = (
    "residential", "commercial", "industrial", "retail",
    "park", "forest", "farmland",
)

MASK_OUTSIDE = 255              # uint8 valid-mask value outside the study area


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2)


# ---------------------------------------------------------------------------
# Grid handling
# ---------------------------------------------------------------------------
class Grid:
    """Authoritative 30 m grid (2026-reference convention, same as V1)."""

    def __init__(self, transform, height: int, width: int, crs: CRS):
        self.transform = transform
        self.height = int(height)
        self.width = int(width)
        self.crs = CRS.from_user_input(crs)

    @classmethod
    def from_file(cls, path: Path) -> "Grid":
        with rasterio.open(path) as ds:
            return cls(ds.transform, ds.height, ds.width, ds.crs)

    @property
    def shape(self):
        return (self.height, self.width)

    def profile(self, count: int = 1, dtype: str = "float64", nodata=np.nan, compress: str = "lzw") -> dict:
        return dict(
            driver="GTiff", height=self.height, width=self.width, count=count,
            dtype=dtype, crs=self.crs, transform=self.transform, nodata=nodata,
            compress=compress, tiled=True, blockxsize=256, blockysize=256,
        )

    def pixel_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Full-grid pixel-centre lon/lat (row-major), via rasterio.transform.xy."""
        from rasterio.transform import xy

        rr, cc = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing="ij")
        xs, ys = xy(self.transform, rr.ravel(), cc.ravel(), offset="center")
        return np.asarray(xs), np.asarray(ys)


def transforms_equal(t1, t2, tol: float = 0.0) -> bool:
    """Exact (or tolerance-based) affine equality."""
    a = np.array([t1.a, t1.b, t1.c, t1.d, t1.e, t1.f], dtype=np.float64)
    b = np.array([t2.a, t2.b, t2.c, t2.d, t2.e, t2.f], dtype=np.float64)
    return bool(np.all(np.abs(a - b) <= tol))


def write_band(path: Path, grid: Grid, arr: np.ndarray, dtype: str = "float64",
               nodata=np.nan, band_name: str | None = None) -> Path:
    """Write a single-band GeoTIFF exactly on the authoritative grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = grid.profile(count=1, dtype=dtype, nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
        if band_name:
            dst.set_band_description(1, band_name)
    return path


def read_band_nan(ds: rasterio.DatasetReader, band: int) -> np.ndarray:
    """Read one band as float64 with masked pixels (incl. nodata) as NaN."""
    arr = ds.read(band, masked=True)
    return arr.astype(np.float64).filled(np.nan)


def reproject_to_grid(src: np.ndarray, src_transform, src_crs, grid: Grid,
                      resampling, src_nodata=np.nan, dst_nodata=np.nan) -> np.ndarray:
    """Reproject/aggregate a single band ONTO the authoritative grid.

    This is the explicit alignment step that fixes the V1 F4 grid defects
    (S2 20 m half-pixel y-offset; S2 10 m / LULC 2 px extent mismatch) by
    construction: the destination lattice is always the authoritative grid.

    For ``Resampling.average`` the aggregation is NaN-safe BY CONSTRUCTION
    (GDAL's average does not reliably exclude interior NaN pixels, verified
    on GDAL 3.12): a two-pass weighted scheme computes
    ``sum(v*w) / sum(valid*w)`` exactly, where ``w`` are the cell overlap
    weights. Cells with no valid source coverage become NaN. For
    ``Resampling.nearest`` values pass through raw (NaN propagates), which
    for categorical labels is the honest no-invention semantics.
    """
    if resampling == Resampling.average:
        valid = np.isfinite(src)
        data = np.where(valid, src, 0.0)
        weights = valid.astype(np.float64)
        dst_data = np.full(grid.shape, np.nan, dtype=np.float64)
        dst_w = np.full(grid.shape, np.nan, dtype=np.float64)
        kwargs = dict(src_transform=src_transform, src_crs=src_crs,
                      dst_transform=grid.transform, dst_crs=grid.crs,
                      resampling=Resampling.average)
        reproject(source=data, destination=dst_data, **kwargs)
        reproject(source=weights, destination=dst_w, **kwargs)
        with np.errstate(invalid="ignore", divide="ignore"):
            # GDAL's average leaves ~1e-11 numerical dust in fully-invalid
            # cells; 1e-6 separates real partial coverage (>= ~1%) from dust.
            dst = np.where(dst_w > 1e-6, dst_data / dst_w, np.nan)
        return dst

    dst = np.full(grid.shape, np.nan, dtype=np.float64)
    reproject(
        source=src.astype(np.float64),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return dst


def load_study_mask(grid: Grid, study_area_path: Path) -> np.ndarray:
    """Boolean mask of study-area cells (pixel centre rule, all_touched=False)."""
    import geopandas as gpd
    from rasterio import features

    gdf = gpd.read_file(study_area_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    gdf = gdf.to_crs(grid.crs)
    mask = features.rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return mask.astype(bool)


def valid_mask_uint8(valid: np.ndarray, study: np.ndarray) -> np.ndarray:
    """uint8 valid mask: 1 valid / 0 invalid / 255 outside the study area."""
    out = np.where(study, valid.astype(np.uint8), MASK_OUTSIDE).astype(np.uint8)
    return out


def block_bands_for_indices(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row_band, col_band) per index array via the PINNED V1 bins (clip 0..4)."""
    rb = np.clip(np.digitize(np.asarray(rows), PINNED_ROW_BINS) - 1, 0, N_SPATIAL_BLOCKS - 1)
    cb = np.clip(np.digitize(np.asarray(cols), PINNED_COL_BINS) - 1, 0, N_SPATIAL_BLOCKS - 1)
    return rb, cb


def compute_block_raster(height: int, width: int) -> np.ndarray:
    """Full-grid spatial_block_id raster using the PINNED V1 bin geometry.

    The extent-derived binning path is deleted by contract: bins never depend
    on the data being mapped, so block ids are identical for the full grid,
    any subset of rows, or any resampling of V2 (requirement: pinned
    geometry; sampling independence). Holdout blocks are
    ``HOLDOUT_BLOCKS`` — see the PINNED_* constants above for provenance.
    """
    rows = np.arange(int(height))
    cols = np.arange(int(width))
    rb, cb = block_bands_for_indices(rows, cols)
    return (rb[:, None] * N_SPATIAL_BLOCKS + cb[None, :]).astype(np.int64)


def write_spatial_blocks_manifest(out_path: Path, source_csv: Path,
                                  grid_file: Path = GRID_FILE_DEFAULT) -> dict:
    """Generate data/phase2/spatial_blocks_manifest.json (one-time provenance).

    Recomputes, from the read-only V1 production table: the pooled bins
    (verified identical to the PINNED_* constants), the formula reproduction
    against the table's spatial_block_id column (honest mismatch count with
    the per-year diagnosis), per-locked-block pixel preservation, and the
    per-block geographic bboxes on the authoritative grid.
    """
    import pandas as pd

    src = Path(source_csv)
    h = sha256_file(src)
    assert h == PINNED_BLOCKS_SOURCE_SHA256, (
        f"source table sha256 changed: {h} != {PINNED_BLOCKS_SOURCE_SHA256}")

    df = pd.read_csv(src, usecols=["row", "col", "year", "spatial_block_id"])
    r_min, r_max = int(df.row.min()), int(df.row.max())
    c_min, c_max = int(df.col.min()), int(df.col.max())
    rbins = np.linspace(r_min, r_max + 1, N_SPATIAL_BLOCKS + 1)
    cbins = np.linspace(c_min, c_max + 1, N_SPATIAL_BLOCKS + 1)
    assert np.array_equal(rbins, np.array(PINNED_ROW_BINS)), (rbins, PINNED_ROW_BINS)
    assert np.array_equal(cbins, np.array(PINNED_COL_BINS)), (cbins, PINNED_COL_BINS)

    rb, cb = block_bands_for_indices(df.row.values, df.col.values)
    pooled = rb * N_SPATIAL_BLOCKS + cb
    col = df.spatial_block_id.to_numpy()
    mism = int((pooled != col).sum())

    # per-year diagnosis of the V1 column
    per_year = {}
    for year, g in df.groupby("year"):
        grb = np.linspace(g.row.min(), g.row.max() + 1, N_SPATIAL_BLOCKS + 1)
        gcb = np.linspace(g.col.min(), g.col.max() + 1, N_SPATIAL_BLOCKS + 1)
        per_year[str(year)] = {
            "sampled_extent": {"row_min": int(g.row.min()), "row_max": int(g.row.max()),
                               "col_min": int(g.col.min()), "col_max": int(g.col.max())},
            "per_year_bins_row": [float(x) for x in grb],
            "per_year_bins_col": [float(x) for x in gcb],
            "mismatches_vs_column_under_per_year_bins": int(
                ((np.clip(np.digitize(g.row.values, grb) - 1, 0, 4) * 5
                  + np.clip(np.digitize(g.col.values, gcb) - 1, 0, 4))
                 != g.spatial_block_id.values).sum()),
        }

    # locked geography preservation under the pinned bins
    locked_rows_mask = np.isin(col, HOLDOUT_BLOCKS)
    preservation = {}
    for b in HOLDOUT_BLOCKS:
        v1_b = col == b
        preserved = int((pooled[v1_b] == b).sum())
        preservation[str(b)] = {
            "v1_column_pixels": int(v1_b.sum()),
            "preserved_under_pinned_bins": preserved,
            "fraction": round(preserved / int(v1_b.sum()), 4),
        }
    locked_set_agreement = float((np.isin(pooled, HOLDOUT_BLOCKS) == locked_rows_mask).mean())

    # per-block geographic records on the authoritative grid
    grid = Grid.from_file(Path(grid_file))
    assert (grid.height, grid.width) == (1768, 1874), (
        "pinned bins are defined for the 1874x1768 V1/V2 grid, got "
        f"{grid.width}x{grid.height}")
    blocks = []
    raster = compute_block_raster(grid.height, grid.width)
    for bid in range(N_SPATIAL_BLOCKS ** 2):
        m = raster == bid
        rr, cc = np.where(m)
        rb_i, cb_i = divmod(bid, N_SPATIAL_BLOCKS)
        blocks.append({
            "block_id": bid,
            "row_band": int(rb_i), "col_band": int(cb_i),
            "row_min": int(rr.min()), "row_max": int(rr.max()),
            "col_min": int(cc.min()), "col_max": int(cc.max()),
            "n_pixels": int(m.sum()),
            "lon_min": float(grid.transform.c + (cc.min() + 0.5) * grid.transform.a),
            "lon_max": float(grid.transform.c + (cc.max() + 0.5) * grid.transform.a),
            "lat_max": float(grid.transform.f - (rr.min() + 0.5) * abs(grid.transform.e)),
            "lat_min": float(grid.transform.f - (rr.max() + 0.5) * abs(grid.transform.e)),
            "is_holdout": bid in HOLDOUT_BLOCKS,
        })
    non_empty = all(b["n_pixels"] > 0 for b in blocks)

    manifest = {
        "generated_by": "src/v2/common.py::write_spatial_blocks_manifest",
        "pinned_constants": {
            "PINNED_ROW_BINS": list(PINNED_ROW_BINS),
            "PINNED_COL_BINS": list(PINNED_COL_BINS),
            "N_SPATIAL_BLOCKS": N_SPATIAL_BLOCKS,
            "HOLDOUT_BLOCKS": list(HOLDOUT_BLOCKS),
        },
        "provenance": {
            "source_table": PINNED_BLOCKS_SOURCE,
            "source_table_sha256": h,
            "source_table_rows": int(len(df)),
            "formula": "V1 _compute_spatial_block_raster pooled: "
                       "linspace(min, max+1, 6), digitize-1, clip(0,4), "
                       "block_id = row_band*5 + col_band",
            "grid": {
                "grid_file": str(grid_file),
                "width": grid.width, "height": grid.height,
                "transform": [grid.transform.a, grid.transform.b, grid.transform.c,
                              grid.transform.d, grid.transform.e, grid.transform.f],
                "crs": str(grid.crs),
                "identical_to_v1_grid": True,
            },
        },
        "verification": {
            "formula_reproduction_vs_column_mismatches": mism,
            "formula_reproduction_vs_column_total": int(len(df)),
            "note": "the V1 table's spatial_block_id COLUMN is per-year "
                    "Phase-3 output; no single bin set reproduces it 100% "
                    "(22,251 pixels have year-inconsistent ids). Pinned "
                    "pooled bins are the frozen production semantics.",
            "per_year_column_diagnosis": per_year,
            "locked_holdout_preservation": preservation,
            "locked_set_agreement_fraction": round(locked_set_agreement, 4),
            "all_25_blocks_non_empty_on_grid": bool(non_empty),
        },
        "blocks": blocks,
    }
    dump_json(manifest, Path(out_path))
    return manifest


# ---------------------------------------------------------------------------
# Valid-mask rules (V2 Phase-2 contract; unit-testable pure functions)
# ---------------------------------------------------------------------------
def valid_l9_mask(st_b10: np.ndarray, sr_b2: np.ndarray,
                  qa_clear_count: np.ndarray) -> np.ndarray:
    """L9 valid: finite ST_B10 & finite SR_B2 & QA_CLEAR_COUNT >= 1."""
    return np.isfinite(st_b10) & np.isfinite(sr_b2) & (qa_clear_count >= 1)


def valid_s2_10m_mask(b2: np.ndarray, b3: np.ndarray, b4: np.ndarray,
                      b8: np.ndarray, valid_count: np.ndarray) -> np.ndarray:
    """S2-10m valid: finite B2..B8 & B2 > 0 (zero-fill guard) & VALID_COUNT >= 1."""
    finite = np.isfinite(b2) & np.isfinite(b3) & np.isfinite(b4) & np.isfinite(b8)
    return finite & (b2 > 0) & (valid_count >= 1)


def valid_s2_20m_mask(b5: np.ndarray, b6: np.ndarray, b7: np.ndarray,
                      b8a: np.ndarray, b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """S2-20m valid: finite B5..B12 (SCL-excluded classes already masked in GEE)."""
    return (np.isfinite(b5) & np.isfinite(b6) & np.isfinite(b7)
            & np.isfinite(b8a) & np.isfinite(b11) & np.isfinite(b12))


def valid_lulc_mask(label: np.ndarray) -> np.ndarray:
    """LULC valid: finite Dynamic World label (classes 0..8)."""
    return np.isfinite(label)


# ---------------------------------------------------------------------------
# Index formulas (V1 conventions, src/preprocessing/indices.py)
# ---------------------------------------------------------------------------
def norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b); NaN preserved, division-by-zero -> NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def lst_from_dn(st_dn: np.ndarray) -> np.ndarray:
    """ST_B10 raw DN -> LST °C: (DN * 0.00341802 + 149.0) - 273.15."""
    with np.errstate(invalid="ignore"):
        return st_dn * LST_SCALE + LST_OFFSET + KELVIN_TO_CELSIUS


def vegetation_cover_from_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """PVC = clamp((NDVI - 0.05) / 0.75, 0, 1); NaN preserved (V1 Phase 4)."""
    with np.errstate(invalid="ignore"):
        pvc = (ndvi - NDVI_SOIL_REFERENCE) / (NDVI_VEG_REFERENCE - NDVI_SOIL_REFERENCE)
    pvc = np.clip(pvc, 0.0, 1.0)
    return np.where(np.isnan(ndvi), np.nan, pvc).astype(np.float64)


# ---------------------------------------------------------------------------
# Frozen 178-feature schema
# ---------------------------------------------------------------------------
def load_frozen_schema(schema_json: Path = SCHEMA_JSON_DEFAULT) -> list[str]:
    data = json.loads(Path(schema_json).read_text(encoding="utf-8"))
    schema = list(data["feature_names"])
    if len(schema) != 178:
        raise ValueError(f"frozen schema must have 178 features, got {len(schema)}")
    return schema


def schema_hash(feature_names: list[str]) -> str:
    return hashlib.sha256(json.dumps(list(feature_names)).encode()).hexdigest()
