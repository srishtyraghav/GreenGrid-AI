"""Block-aware spatial neighbourhood features for Phase 5.

Computes focal means of existing predictors at multiple spatial scales
(3×3, 5×5, 11×11). To avoid leakage across spatial blocks during validation,
each pixel's neighbourhood mean is computed using only pixels that belong to
the same spatial block.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from .config import (
    GROUP_VAR,
    INPUT_DATASET_CSV,
    PHASE5_TABLES_DIR,
    REFERENCE_RASTER,
)

# Spatial scales (window sizes in pixels) to compute.
WINDOW_SIZES = [3, 5, 11]

# Base features for which neighbourhood means are computed.
SPATIAL_FEATURE_BASES = [
    "ndvi",
    "ndbi",
    "vegetation_cover",
    "dist_road_m",
    "dist_vegetation_m",
    "dist_building_m",
]


def _compute_spatial_block_raster(
    sampled_df: pd.DataFrame,
    height: int,
    width: int,
    n_blocks: int = 5,
) -> np.ndarray:
    """Compute a full-raster spatial_block_id grid using Phase 3 bins.

    The bins are derived from the min/max row/col of the sampled dataset so
    that block IDs are consistent with the Phase 4 table.
    """
    sampled_rows = sampled_df["row"].values
    sampled_cols = sampled_df["col"].values

    row_bins = np.linspace(sampled_rows.min(), sampled_rows.max() + 1, n_blocks + 1)
    col_bins = np.linspace(sampled_cols.min(), sampled_cols.max() + 1, n_blocks + 1)

    all_rows = np.arange(height)
    all_cols = np.arange(width)

    row_block = np.digitize(all_rows, row_bins) - 1
    col_block = np.digitize(all_cols, col_bins) - 1

    row_block = np.clip(row_block, 0, n_blocks - 1)
    col_block = np.clip(col_block, 0, n_blocks - 1)

    block_id_raster = row_block[:, None] * n_blocks + col_block[None, :]
    return block_id_raster


def _focal_mean_block_aware(
    arr: np.ndarray,
    block_id_raster: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Compute block-aware focal mean.

    For each spatial block, the focal mean is computed using only pixels within
    that block. Pixels outside the block are treated as NaN for that block's
    computation, so a pixel's neighbourhood feature never incorporates values
    from a different spatial block.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    result = np.full_like(arr, np.nan, dtype=np.float64)
    kernel = np.ones((window_size, window_size), dtype=np.float64)
    half = window_size // 2

    unique_blocks = np.unique(block_id_raster)
    for bid in unique_blocks:
        block_mask = block_id_raster == bid
        # Work on a copy where only this block is valid.
        block_arr = arr.copy()
        block_arr[~block_mask] = np.nan

        valid = ~np.isnan(block_arr)
        # Sum of valid values in window.
        value_sum = ndimage.convolve(
            np.where(valid, block_arr, 0.0),
            kernel,
            mode="constant",
            cval=0.0,
        )
        # Count of valid values in window.
        valid_count = ndimage.convolve(
            valid.astype(np.float64),
            kernel,
            mode="constant",
            cval=0.0,
        )
        # Mean where count > 0.
        with np.errstate(invalid="ignore"):
            block_mean = np.where(valid_count > 0, value_sum / valid_count, np.nan)

        # Assign only to pixels in this block.
        result[block_mask] = block_mean[block_mask]

    return result


def load_raster_for_feature(feature_name: str, year: int) -> Tuple[np.ndarray, rasterio.Affine]:
    """Load the Phase 4 raster corresponding to a feature name and year."""
    from features.config import (
        BUILDINGS_DISTANCE,
        LANDUSE_RASTER,
        ROADS_DISTANCE,
        S2_NDBI_2022,
        S2_NDBI_2026,
        S2_NDVI_2022,
        S2_NDVI_2026,
        VEGETATION_COVER_RASTER_2022,
        VEGETATION_COVER_RASTER_2026,
        VEGETATION_DISTANCE,
    )

    feature_to_path = {
        "ndvi": {2022: S2_NDVI_2022, 2026: S2_NDVI_2026},
        "ndbi": {2022: S2_NDBI_2022, 2026: S2_NDBI_2026},
        "vegetation_cover": {
            2022: VEGETATION_COVER_RASTER_2022,
            2026: VEGETATION_COVER_RASTER_2026,
        },
        "dist_road_m": {2022: ROADS_DISTANCE, 2026: ROADS_DISTANCE},
        "dist_vegetation_m": {2022: VEGETATION_DISTANCE, 2026: VEGETATION_DISTANCE},
        "dist_building_m": {2022: BUILDINGS_DISTANCE, 2026: BUILDINGS_DISTANCE},
    }

    path = feature_to_path[feature_name][year]
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        transform = src.transform
    return arr, transform


def build_spatial_features(
    sampled_df: pd.DataFrame,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Build block-aware neighbourhood features for all sampled pixels.

    Parameters
    ----------
    sampled_df : pd.DataFrame
        Phase 4 combined dataset with `row`, `col`, and `year` columns.
    output_csv : str, optional
        If provided, save the resulting feature table to this path.

    Returns
    -------
    pd.DataFrame
        One row per sampled pixel-year, with columns:
        `row`, `col`, `year`, and `<feature>_mean<W>` for each feature and window.
    """
    # Reference grid dimensions.
    with rasterio.open(REFERENCE_RASTER) as src:
        height, width = src.height, src.width

    # Full-raster block IDs consistent with sampled blocks.
    block_id_raster = _compute_spatial_block_raster(sampled_df, height, width)

    records = []
    years = sorted(sampled_df["year"].unique())

    for year in years:
        sub = sampled_df[sampled_df["year"] == year].copy()
        rows = sub["row"].values.astype(int)
        cols = sub["col"].values.astype(int)

        feat_dict: Dict[str, np.ndarray] = {"row": rows, "col": cols, "year": np.full(len(rows), year)}

        for feature in SPATIAL_FEATURE_BASES:
            arr, _ = load_raster_for_feature(feature, year)
            for window in WINDOW_SIZES:
                mean_raster = _focal_mean_block_aware(arr, block_id_raster, window)
                values = mean_raster[rows, cols]
                feat_dict[f"{feature}_mean{window}"] = values

        records.append(pd.DataFrame(feat_dict))

    spatial_df = pd.concat(records, ignore_index=True)

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        spatial_df.to_csv(output_csv, index=False)

    return spatial_df


def merge_spatial_features(
    df: pd.DataFrame,
    spatial_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge spatial neighbourhood features into the Phase 4 dataset."""
    if spatial_df is None:
        # Try to load cached spatial features.
        cache_path = PHASE5_TABLES_DIR / "spatial_neighbourhood_features.csv"
        if cache_path.exists():
            spatial_df = pd.read_csv(cache_path)
        else:
            spatial_df = build_spatial_features(df, output_csv=str(cache_path))

    return df.merge(spatial_df, on=["row", "col", "year"], how="left")


from pathlib import Path
