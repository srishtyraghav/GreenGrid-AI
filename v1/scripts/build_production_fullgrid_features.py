"""Stage 1 production build: full-grid 178-feature tables for 2022-2026.

Builds, for every year, the complete encoded predictor matrix required by the
frozen Phase-5 production XGBoost model (178 features, schema order fixed by
data/processed/phase5_production_3class/phase5_primary_xgb_3class_features.json)
on the FULL reference grid (1768 x 1874), streaming year by year.

Construction mirrors the training table exactly (scripts/freeze_phase5_production.py):
  * base predictors from the Phase 3/4 source rasters (float64),
  * block-aware focal means 3/5/11 (src/models/spatial_features.py helpers),
  * morphology disk features 50-500 m (src/models/morphology_features.py helpers),
  * 33 nan-safe window stats at 3/5/11 (scripts/build_spatial_context_features.py
    nan_window_stats, computed on float32 casts of the same source rasters),
  * 6 met covariates via the exact IDW variant reverse-engineered from
    data/processed/experiments/exp_met_features.csv:
        w_i = 1 / (dlat^2 + (cos(28.64 deg) * dlon)^2 + 1e-6)   (power 2, degrees,
        longitude scaled by cos(28.64 deg), 1e-6 epsilon inside the squared distance)
    reproducing exp_met_features.csv to 5.1e-13 max abs error.
  * landuse/lulc/dominant columns cast to the SAME dtypes as the training table
    (landuse_class float64 -> float-style dummies, lulc_class / landuse_dominant_*
    int64 -> int-style dummies) before encode_predictors; after encoding the frame
    is reindexed to the 178-schema order, missing dummies zero-filled.

Prediction domain per year: pixels where all raw predictors are finite
(mirrors training dropna). Output: one parquet per year (float32,
row/col + 178 features) plus per-year domain counts JSON.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/build_production_fullgrid_features.py
      [--force] [--years 2022,2023,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))

from build_spatial_context_features import (  # noqa: E402
    MEAN_BANDS,
    RANGE_BANDS,
    STD_BANDS,
    WINDOWS,
    band_path,
    nan_window_stats,
)
from features.config import (  # noqa: E402
    BUILDINGS_DISTANCE,
    LANDUSE_RASTER,
    L9_MNDWI_RASTERS,
    ROADS_DISTANCE,
    S2_NDRE_RASTERS,
    VEGETATION_DISTANCE,
)
from freeze_phase5_production import (  # noqa: E402
    MET,
    NEW_SPATIAL,
    PREDICTOR_COLS,
)
from models.config import (  # noqa: E402
    CATEGORICAL_VARS,
    MORPHOLOGY_LANDUSE_CLASSES,
    MORPHOLOGY_RADII_M,
    REFERENCE_RASTER,
    SPATIAL_FEATURE_BASES,
    SPATIAL_WINDOW_SIZES,
    YEAR_VAR,
)
from models.dataset import encode_predictors  # noqa: E402
from models.morphology_features import (  # noqa: E402
    _disk_kernel,
    _focal_aggregate_block_aware,
    _focal_fraction_block_aware,
    _focal_mean_block_aware,
    _load_source_rasters,
    _radius_m_to_px,
)
from models.spatial_features import (  # noqa: E402
    _compute_spatial_block_raster,
    _focal_mean_block_aware as _spatial_focal_mean_block_aware,
    load_raster_for_feature,
)

OUT_DIR = PROJECT / "data" / "processed" / "phase6_production" / "fullgrid"
SCHEMA_JSON = PROJECT / "data" / "processed" / "phase5_production_3class" / "phase5_primary_xgb_3class_features.json"
COMBINED_CSV = PROJECT / "data" / "processed" / "lulc_outputs" / "combined_urban_environmental_dataset.csv"
GRID_CSV = PROJECT / "data" / "processed" / "experiments" / "exp_met_features_grid_provenance.csv"
YEARS = (2022, 2023, 2024, 2025, 2026)

IDW_COS_SCALE = np.cos(np.deg2rad(28.64))  # longitude scaling of the matched variant
IDW_EPS2 = 1e-6                            # epsilon added to squared distance

# Distance columns have magnitudes up to ~4.5e4 m; float32 storage would round
# them by up to ~2.7e-3, breaching the G1 parity bound (1e-4 max abs diff).
# They are therefore kept float64 in the parquet; every other feature column
# is float32 (stage-2 prediction casts to float32 anyway via DMatrix).
DIST_COLS = (
    ["dist_road_m", "dist_vegetation_m", "dist_building_m"]
    + [f"dist_{b}_mean{w}" for b in ("road_m", "vegetation_m", "building_m") for w in (3, 5, 11)]
)
assert len(DIST_COLS) == 12

# extra base predictors not served by load_raster_for_feature
_EXTRA_BASE_PATHS = {
    "mndwi": L9_MNDWI_RASTERS,
    "ndre": S2_NDRE_RASTERS,
    "dist_road_m": {y: ROADS_DISTANCE for y in YEARS},
    "dist_vegetation_m": {y: VEGETATION_DISTANCE for y in YEARS},
    "dist_building_m": {y: BUILDINGS_DISTANCE for y in YEARS},
}


def load_schema() -> list[str]:
    schema = json.loads(SCHEMA_JSON.read_text())["feature_names"]
    assert len(schema) == 178
    return schema


def read_band_float64(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.float64)


def load_base_rasters(year: int) -> dict[str, np.ndarray]:
    """All base-predictor rasters for one year as float64 full grids."""
    rasters: dict[str, np.ndarray] = {}
    for feature in ("ndvi", "ndbi", "vegetation_cover", "ndmi", "bsi"):
        arr, _ = load_raster_for_feature(feature, year)
        rasters[feature] = arr
    for feature, paths in _EXTRA_BASE_PATHS.items():
        rasters[feature] = read_band_float64(paths[year])
    with rasterio.open(LANDUSE_RASTER) as ds:
        lu = ds.read(1).astype(np.float64)
    lu[lu == 255.0] = np.nan  # nodata -> NaN (never sampled in training)
    rasters["landuse_class"] = lu
    from features.config import LULC_RASTERS
    rasters["lulc_class"] = read_band_float64(LULC_RASTERS[year])
    return rasters


def compute_spatial_mean_grids(year: int, block_id_raster: np.ndarray) -> dict[str, np.ndarray]:
    """Block-aware focal means 3/5/11 for the 8 spatial bases (full grids)."""
    out: dict[str, np.ndarray] = {}
    for feature in SPATIAL_FEATURE_BASES:
        arr, _ = load_raster_for_feature(feature, year)
        for window in SPATIAL_WINDOW_SIZES:
            out[f"{feature}_mean{window}"] = _spatial_focal_mean_block_aware(
                arr, block_id_raster, window
            )
        del arr
    return out


def compute_morphology_grids(year: int, block_id_raster: np.ndarray) -> dict[str, np.ndarray]:
    """Full-grid morphology features (disk radii 50-500 m), same recipe as
    src/severity/fullgrid.py::_morphology_features_for_year but keeping the
    full grids instead of sampling."""
    rasters = _load_source_rasters(year)
    lu_classes = np.array(MORPHOLOGY_LANDUSE_CLASSES)
    out: dict[str, np.ndarray] = {}

    for radius_m in MORPHOLOGY_RADII_M:
        radius_px = _radius_m_to_px(radius_m)
        kernel = _disk_kernel(radius_px)

        out[f"building_pixel_fraction_{radius_m}m"] = _focal_fraction_block_aware(
            (rasters["dist_building_m"] == 0).astype(np.float64), block_id_raster, kernel
        )
        out[f"road_pixel_fraction_{radius_m}m"] = _focal_fraction_block_aware(
            (rasters["dist_road_m"] == 0).astype(np.float64), block_id_raster, kernel
        )
        out[f"vegetation_pixel_fraction_{radius_m}m"] = _focal_fraction_block_aware(
            (rasters["dist_vegetation_m"] == 0).astype(np.float64), block_id_raster, kernel
        )
        out[f"vegetation_cover_mean_{radius_m}m"] = _focal_mean_block_aware(
            rasters["vegetation_cover"], block_id_raster, kernel
        )

        ndvi_mean = _focal_mean_block_aware(rasters["ndvi"], block_id_raster, kernel)
        out[f"ndvi_contrast_{radius_m}m"] = rasters["ndvi"] - ndvi_mean
        del ndvi_mean
        ndbi_mean = _focal_mean_block_aware(rasters["ndbi"], block_id_raster, kernel)
        out[f"ndbi_contrast_{radius_m}m"] = rasters["ndbi"] - ndbi_mean
        del ndbi_mean

        class_counts = np.stack(
            [
                _focal_aggregate_block_aware(
                    (rasters["landuse_class"] == c).astype(np.float64),
                    block_id_raster,
                    kernel,
                )[0]
                for c in lu_classes
            ],
            axis=0,
        )
        total_count = class_counts.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            class_fracs = np.where(total_count > 0, class_counts / total_count, 0.0)
        class_fracs = np.nan_to_num(np.clip(class_fracs, 0.0, 1.0), nan=0.0)
        for idx, c in enumerate(MORPHOLOGY_LANDUSE_CLASSES):
            out[f"landuse_frac_{c}_{radius_m}m"] = class_fracs[idx]

        dominant = np.where(
            total_count > 0,
            lu_classes[np.argmax(class_counts, axis=0)],
            0,
        )
        out[f"landuse_dominant_{radius_m}m"] = dominant.astype(np.float64)

        with np.errstate(invalid="ignore", divide="ignore"):
            probs = np.where(total_count > 0, class_counts / total_count, 0.0)
            entropy = np.where(
                total_count > 0,
                -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=0),
                0.0,
            )
        out[f"landuse_entropy_{radius_m}m"] = np.nan_to_num(entropy, nan=0.0)
        del class_counts, class_fracs, total_count

    return out


def compute_window_stat_grids(year: int) -> dict[str, np.ndarray]:
    """33 nan-safe window stats at 3/5/11 on full grids. Bit-exact replication
    of scripts/build_spatial_context_features.py: source rasters are cast to
    float32 before filtering."""
    out: dict[str, np.ndarray] = {}
    for band in dict.fromkeys(STD_BANDS + MEAN_BANDS + RANGE_BANDS):
        with rasterio.open(band_path(band, year)) as ds:
            arr = ds.read(1).astype(np.float32)
        for w in WINDOWS:
            mean, std, rng = nan_window_stats(arr, w)
            if band in STD_BANDS:
                out[f"{band}_std{w}"] = std.astype(np.float64)
            if band in MEAN_BANDS:
                out[f"{band}_mean{w}"] = mean.astype(np.float64)
            if band in RANGE_BANDS:
                out[f"{band}_range{w}"] = rng.astype(np.float64)
        del arr
        print(f"  [winstats] {year} {band} done", flush=True)
    return out


def compute_met_grids(weights: np.ndarray, wsum: np.ndarray, year: int) -> dict[str, np.ndarray]:
    grid = pd.read_csv(GRID_CSV)
    g = grid[grid.year == year]
    gvar = g[["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
              "precipitation", "shortwave_radiation", "soil_moisture_0_to_7cm"]].values
    preds = (weights @ gvar) / wsum[:, None]
    names = ["met_t2m_c", "met_rh_pct", "met_wind_kmh",
             "met_precip_mm", "met_ssr_wm2", "met_swc_m3m3"]
    return {n: preds[:, i] for i, n in enumerate(names)}


def build_year(year: int, schema: list[str], block_id_raster: np.ndarray,
               weights: np.ndarray, wsum: np.ndarray) -> tuple[pd.DataFrame, dict]:
    print(f"[BUILD] === year {year} ===", flush=True)

    cols: dict[str, np.ndarray] = {}
    domain = None

    def add(name: str, arr: np.ndarray) -> None:
        nonlocal domain
        cols[name] = arr
        finite = np.isfinite(arr)
        domain = finite if domain is None else (domain & finite)

    base = load_base_rasters(year)
    for feature, arr in base.items():
        add(feature, arr)
    year_grid = np.full(next(iter(base.values())).shape, year, dtype=np.float64)
    add(YEAR_VAR, year_grid)
    del base

    spatial = compute_spatial_mean_grids(year, block_id_raster)
    for name, arr in spatial.items():
        add(name, arr)
    del spatial

    morph = compute_morphology_grids(year, block_id_raster)
    for name, arr in morph.items():
        add(name, arr)
    del morph

    stats = compute_window_stat_grids(year)
    for name, arr in stats.items():
        add(name, arr)
    del stats

    met = compute_met_grids(weights, wsum, year)
    grid_shape = cols[YEAR_VAR].shape
    for name, arr in met.items():
        add(name, arr.reshape(grid_shape))
    del met

    assert set(cols) == set(PREDICTOR_COLS), (
        f"column mismatch: missing={set(PREDICTOR_COLS) - set(cols)}, "
        f"extra={set(cols) - set(PREDICTOR_COLS)}")

    rows, cl = np.where(domain)
    n_domain = len(rows)
    out = {"row": rows.astype(np.int64), "col": cl.astype(np.int64)}
    for name in PREDICTOR_COLS:
        arr = cols[name]
        out[name] = arr[domain].astype(np.float64 if name in DIST_COLS else np.float32)
    del cols, domain
    df = pd.DataFrame(out)

    # dtype parity with the training table BEFORE encoding (drives dummy names)
    df["landuse_class"] = df["landuse_class"].astype(np.float64)
    df["lulc_class"] = df["lulc_class"].astype(np.int64)
    for c in [c for c in df.columns if c.startswith("landuse_dominant_")]:
        df[c] = df[c].astype(np.int64)

    X, names = encode_predictors(df, predictor_cols=PREDICTOR_COLS,
                                 categorical_cols=CATEGORICAL_VARS)
    X = X.reindex(columns=schema, fill_value=0)
    assert list(X.columns) == list(schema)
    for c in X.columns:
        if c not in DIST_COLS:
            X[c] = X[c].astype(np.float32)

    result = pd.concat([df[["row", "col"]].reset_index(drop=True),
                        X.reset_index(drop=True)], axis=1)
    info = {
        "year": year,
        "grid_pixels": int(year_grid.size),
        "domain_pixels": int(n_domain),
        "encoded_columns_before_reindex": int(len(names)),
    }
    return result, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if parquet exists")
    parser.add_argument("--years", default=None, help="comma-separated subset of years")
    args = parser.parse_args()

    years = YEARS if not args.years else tuple(int(y) for y in args.years.split(","))
    schema = load_schema()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(REFERENCE_RASTER) as ds:
        height, width = ds.height, ds.width
        transform = ds.transform
    print(f"[BUILD] reference grid {height} x {width}")

    # Spatial block raster from the full sampled table (identical bins to Phase 5)
    sampled_df = pd.read_csv(COMBINED_CSV, usecols=["row", "col"])
    block_id_raster = _compute_spatial_block_raster(sampled_df, height, width)
    del sampled_df

    # Met IDW geometry (identical every year): pixel-centre lon/lat from the
    # reference transform, 16 grid points from the provenance CSV.
    grid = pd.read_csv(GRID_CSV)
    g_lon = grid[grid.year == YEARS[0]].lon.values  # same points every year
    g_lat = grid[grid.year == YEARS[0]].lat.values
    rr, cc = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    px_lon, px_lat = xy(transform, rr.ravel(), cc.ravel(), offset="center")
    px_lon = np.asarray(px_lon)
    px_lat = np.asarray(px_lat)
    dlon = px_lon[:, None] - g_lon[None, :]
    dlat = px_lat[:, None] - g_lat[None, :]
    d2 = dlat ** 2 + (IDW_COS_SCALE * dlon) ** 2 + IDW_EPS2
    weights = 1.0 / d2
    wsum = weights.sum(axis=1)
    del px_lon, px_lat, dlon, dlat, d2, rr, cc

    counts_path = OUT_DIR / "domain_counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}

    for year in years:
        parquet_path = OUT_DIR / f"features_{year}.parquet"
        if parquet_path.exists() and not args.force:
            print(f"[BUILD] {parquet_path} exists, skipping (use --force to rebuild)")
            if str(year) in counts:
                continue
        result, info = build_year(year, schema, block_id_raster, weights, wsum)
        result.to_parquet(parquet_path, index=False)
        counts[str(year)] = info
        counts_path.write_text(json.dumps(counts, indent=2))
        print(f"[BUILD] wrote {parquet_path} rows={len(result)} domain={info['domain_pixels']}", flush=True)
        del result

    print("[BUILD] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
