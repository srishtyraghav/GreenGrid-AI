"""V2 Phase 4 — spatial/morphology/meteorology feature engines.

Ports of the V1 semantics (documented in src/v2/CONTRACT.md §features):

* block-aware focal means 3/5/11   <- src/models/spatial_features.py
* disk morphology 50-500 m         <- src/models/morphology_features.py
* nan-safe window stats (33)       <- scripts/build_spatial_context_features.py
* met W4 aggregation + matched IDW <- reports/experiments/exp_met_provenance.md
  and scripts/build_production_fullgrid_features.py (matched variant)
* one-hot encoding to the frozen 178 schema <- src/models/dataset.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from ..common import (
    IDW_COS_LAT_DEG,
    IDW_EPS2,
    MET_VARS,
    MET_WINDOW_HOURS_UTC,
    MORPHOLOGY_RESOLUTION_M,
    N_SPATIAL_BLOCKS,
)

# 8 spatial bases for block-aware focal means (V1 SPATIAL_FEATURE_BASES).
SPATIAL_FEATURE_BASES = (
    "ndvi", "ndbi", "vegetation_cover", "ndmi", "bsi",
    "dist_road_m", "dist_vegetation_m", "dist_building_m",
)
SPATIAL_WINDOW_SIZES = (3, 5, 11)

# 33 nan-safe window stats (V1 STD/MEAN/RANGE_BANDS).
STD_BANDS = ("ndvi", "ndbi", "ndre", "ndmi", "mndwi", "bsi", "vegetation_cover")
MEAN_BANDS = ("mndwi", "ndre")
RANGE_BANDS = ("ndvi", "ndbi")
WINDOWS = (3, 5, 11)


# ---------------------------------------------------------------------------
# Block-aware focal means (port of spatial_features._focal_mean_block_aware)
# ---------------------------------------------------------------------------
def focal_mean_block_aware(arr: np.ndarray, block_id_raster: np.ndarray,
                           window_size: int) -> np.ndarray:
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    result = np.full(arr.shape, np.nan, dtype=np.float64)
    kernel = np.ones((window_size, window_size), dtype=np.float64)
    for bid in np.unique(block_id_raster):
        block_mask = block_id_raster == bid
        block_arr = arr.copy()
        block_arr[~block_mask] = np.nan
        valid = ~np.isnan(block_arr)
        value_sum = ndimage.convolve(np.where(valid, block_arr, 0.0), kernel,
                                     mode="constant", cval=0.0)
        valid_count = ndimage.convolve(valid.astype(np.float64), kernel,
                                       mode="constant", cval=0.0)
        with np.errstate(invalid="ignore"):
            block_mean = np.where(valid_count > 0, value_sum / valid_count, np.nan)
        result[block_mask] = block_mean[block_mask]
    return result


# ---------------------------------------------------------------------------
# Disk morphology (port of morphology_features.py)
# ---------------------------------------------------------------------------
def disk_kernel(radius_px: int) -> np.ndarray:
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    y, x = np.ogrid[-radius_px: radius_px + 1, -radius_px: radius_px + 1]
    return (x * x + y * y) <= radius_px * radius_px


def radius_m_to_px(radius_m: float, resolution_m: float = MORPHOLOGY_RESOLUTION_M) -> int:
    return max(1, int(round(radius_m / resolution_m)))


def _prefix_sum(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, ((1, 0), (1, 0)), mode="constant")
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r: np.ndarray, c_lo: np.ndarray, c_hi: np.ndarray) -> np.ndarray:
    n_rows, n_cols = prefix.shape[0] - 1, prefix.shape[1] - 1
    r = np.clip(r, 0, n_rows - 1)
    c_lo = np.clip(c_lo, 0, n_cols - 1)
    c_hi = np.clip(c_hi, 0, n_cols - 1)
    return (prefix[r + 1, c_hi + 1] - prefix[r, c_hi + 1]
            - prefix[r + 1, c_lo] + prefix[r, c_lo])


def _block_bounds(block_id_raster: np.ndarray):
    height, width = block_id_raster.shape
    r_min = np.full((height, width), height, dtype=int)
    r_max = np.full((height, width), -1, dtype=int)
    c_min = np.full((height, width), width, dtype=int)
    c_max = np.full((height, width), -1, dtype=int)
    for bid in np.unique(block_id_raster):
        block_mask = block_id_raster == bid
        rows = np.where(block_mask.any(axis=1))[0]
        cols = np.where(block_mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        r_min[block_mask] = rows.min()
        r_max[block_mask] = rows.max()
        c_min[block_mask] = cols.min()
        c_max[block_mask] = cols.max()
    return r_min, r_max, c_min, c_max


def _disk_half_widths(kernel_half: int) -> list[int]:
    return [int(np.floor(np.sqrt(max(kernel_half * kernel_half - dy * dy, 0))))
            for dy in range(-kernel_half, kernel_half + 1)]


def focal_aggregate_block_aware(arr: np.ndarray, block_id_raster: np.ndarray,
                                kernel: np.ndarray):
    """Block-aware focal (value_sum, valid_count) via stripe prefix sums."""
    height, width = arr.shape
    kernel_half = (kernel.shape[0] - 1) // 2
    valid = ~np.isnan(arr)
    value_prefix = _prefix_sum(np.where(valid, arr, 0.0))
    valid_prefix = _prefix_sum(valid.astype(np.float64))
    r_min, r_max, c_min, c_max = _block_bounds(block_id_raster)
    r_idx = np.arange(height)[:, None]
    c_idx = np.arange(width)[None, :]

    value_sum = np.zeros((height, width), dtype=np.float64)
    valid_count = np.zeros((height, width), dtype=np.float64)
    for dy, half_width in zip(range(-kernel_half, kernel_half + 1),
                              _disk_half_widths(kernel_half)):
        rr = r_idx + dy
        row_inside_block = (rr >= r_min) & (rr <= r_max)
        rr_clip = np.clip(rr, 0, height - 1)
        c_lo = np.clip(np.clip(c_idx - half_width, c_min, c_max), 0, width - 1)
        c_hi = np.clip(np.clip(c_idx + half_width, c_min, c_max), 0, width - 1)
        value_sum += np.where(row_inside_block,
                              _rect_sum(value_prefix, rr_clip, c_lo, c_hi), 0.0)
        valid_count += np.where(row_inside_block,
                                _rect_sum(valid_prefix, rr_clip, c_lo, c_hi), 0.0)
    return value_sum, valid_count


def focal_mean_block_aware_disk(arr: np.ndarray, block_id_raster: np.ndarray,
                                kernel: np.ndarray) -> np.ndarray:
    value_sum, valid_count = focal_aggregate_block_aware(arr, block_id_raster, kernel)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid_count > 0, value_sum / valid_count, np.nan)


def focal_fraction_block_aware(binary_arr: np.ndarray, block_id_raster: np.ndarray,
                               kernel: np.ndarray) -> np.ndarray:
    value_sum, valid_count = focal_aggregate_block_aware(
        binary_arr.astype(np.float64), block_id_raster, kernel)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(valid_count > 0, value_sum / valid_count, np.nan)
    return np.clip(fraction, 0.0, 1.0)


# ---------------------------------------------------------------------------
# nan-safe window stats (port of build_spatial_context_features.nan_window_stats)
# ---------------------------------------------------------------------------
def nan_window_stats(arr: np.ndarray, w: int):
    """Border-correct nan-safe window mean/std/range via summed-area filters."""
    valid = np.isfinite(arr)
    a = np.where(valid, arr, 0.0).astype(np.float32)
    w2 = float(w * w)
    c = ndimage.uniform_filter(valid.astype(np.float32), size=w, mode="constant", cval=0) * w2
    safe_c = np.where(c > 0, c, np.nan)
    s = ndimage.uniform_filter(a, size=w, mode="constant", cval=0) * w2
    s2 = ndimage.uniform_filter(a * a, size=w, mode="constant", cval=0) * w2
    mean = s / safe_c
    var = np.clip(s2 / safe_c - mean ** 2, 0, None)
    mx = ndimage.maximum_filter(np.where(valid, arr, -np.inf), size=w, mode="constant", cval=-np.inf)
    mn = ndimage.minimum_filter(np.where(valid, arr, np.inf), size=w, mode="constant", cval=np.inf)
    rng = np.where(c > 0, mx - mn, np.nan)
    return mean, np.sqrt(var), rng


# ---------------------------------------------------------------------------
# Meteorology: W4 aggregation + matched IDW
# ---------------------------------------------------------------------------
def aggregate_met_window(met_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Per point per year: mean over days May 1 - Jun 30, hours 04:00-06:00 UTC.

    Returns a DataFrame: point_id, lat, lon + the six MET_VARS.
    """
    t = pd.to_datetime(met_df["time_utc"])
    in_window = (
        (t.dt.year == year)
        & (t.dt.hour.isin(MET_WINDOW_HOURS_UTC))
        & (t.dt.month.isin((5, 6)))
    )
    sel = met_df.loc[in_window]
    if sel.empty:
        raise ValueError(f"met superset has no W4 rows for year {year} "
                         f"(hours {MET_WINDOW_HOURS_UTC} UTC, May-Jun)")
    vars_present = [v for v in MET_VARS if v in sel.columns]
    agg = sel.groupby("point_id", as_index=False)[vars_present].mean()
    pts = met_df[["point_id", "lat", "lon"]].drop_duplicates("point_id")
    out = pts.merge(agg, on="point_id", how="inner").sort_values("point_id")
    return out.reset_index(drop=True)


def idw_weights(px_lon: np.ndarray, px_lat: np.ndarray,
                g_lon: np.ndarray, g_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """V1 matched variant: w_i = 1 / (dlat^2 + (cos(28.64 deg) * dlon)^2 + 1e-6)."""
    cos_scale = np.cos(np.deg2rad(IDW_COS_LAT_DEG))
    dlon = px_lon[:, None] - g_lon[None, :]
    dlat = px_lat[:, None] - g_lat[None, :]
    d2 = dlat ** 2 + (cos_scale * dlon) ** 2 + IDW_EPS2
    weights = 1.0 / d2
    return weights, weights.sum(axis=1)


def met_fields_for_year(met_df: pd.DataFrame, year: int,
                        px_lon: np.ndarray, px_lat: np.ndarray) -> dict[str, np.ndarray]:
    """Full-grid met feature fields for one year."""
    agg = aggregate_met_window(met_df, year)
    missing = [v for v in MET_VARS if v not in agg.columns]
    if missing:
        raise ValueError(f"met superset missing variables {missing} (need all six)")
    g_lon = agg["lon"].to_numpy(dtype=np.float64)
    g_lat = agg["lat"].to_numpy(dtype=np.float64)
    g_var = agg[list(MET_VARS)].to_numpy(dtype=np.float64)
    weights, wsum = idw_weights(px_lon, px_lat, g_lon, g_lat)
    preds = (weights @ g_var) / wsum[:, None]
    from ..common import MET_FEATURE_NAMES
    return {name: preds[:, i] for i, name in enumerate(MET_FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# Encoding to the frozen 178 schema (port of models.dataset.encode_predictors)
# ---------------------------------------------------------------------------
CATEGORICAL_VARS = ("landuse_class", "lulc_class")


def encode_predictors(df: pd.DataFrame, predictor_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    feature_names = []
    categorical = set(CATEGORICAL_VARS) | {
        c for c in predictor_cols if c.startswith("landuse_dominant_")}
    for col in predictor_cols:
        if col in categorical:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            frames.append(dummies)
            feature_names.extend(dummies.columns.tolist())
        else:
            frames.append(df[[col]].copy())
            feature_names.append(col)
    return pd.concat(frames, axis=1), feature_names
