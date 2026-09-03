"""Block-aware urban morphology features for Phase 5.

Computes circular-neighbourhood urban-form features from existing Phase 3/4
rasters.  Every pixel's neighbourhood is restricted to pixels that belong to the
same spatial block, matching the leakage-control design of
`spatial_features.py`.

Distance-raster semantics (verified against `src/preprocessing/vector_raster.py`):
Phase 3 rasterizes features as value 1 and background as 0, then computes
`distance_transform_edt(binary == 0)`.  Therefore pixels with `distance == 0`
are exactly the feature pixels, and are used here to build binary masks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio

from .config import (
    GROUP_VAR,
    INPUT_DATASET_CSV,
    MORPHOLOGY_FEATURES_CSV,
    MORPHOLOGY_LANDUSE_CLASSES,
    MORPHOLOGY_RADII_M,
    MORPHOLOGY_RESOLUTION_M,
    REFERENCE_RASTER,
)
from .spatial_features import _compute_spatial_block_raster


def _disk_kernel(radius_px: int) -> np.ndarray:
    """Return a boolean disk-shaped footprint with the given integer radius."""
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    size = 2 * radius_px + 1
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (x * x + y * y) <= radius_px * radius_px


def _radius_m_to_px(radius_m: int, resolution_m: float = MORPHOLOGY_RESOLUTION_M) -> int:
    """Convert a metric radius to an integer pixel radius."""
    return max(1, int(round(radius_m / resolution_m)))


def _prefix_sum(arr: np.ndarray) -> np.ndarray:
    """Return a 2-D prefix sum (integral image) with a zero border."""
    padded = np.pad(arr, ((1, 0), (1, 0)), mode="constant")
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r: np.ndarray, c_lo: np.ndarray, c_hi: np.ndarray) -> np.ndarray:
    """Inclusive single-row rectangle sum using a prefix sum image.

    ``r``, ``c_lo``, and ``c_hi`` are H×W index arrays.  The rectangle is the
    single row ``r`` and the inclusive column range ``[c_lo, c_hi]``.
    """
    n_rows = prefix.shape[0] - 1
    n_cols = prefix.shape[1] - 1
    r = np.clip(r, 0, n_rows - 1)
    c_lo = np.clip(c_lo, 0, n_cols - 1)
    c_hi = np.clip(c_hi, 0, n_cols - 1)
    return (
        prefix[r + 1, c_hi + 1]
        - prefix[r, c_hi + 1]
        - prefix[r + 1, c_lo]
        + prefix[r, c_lo]
    )


def _block_bounds(block_id_raster: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-pixel (row_min, row_max, col_min, col_max) of each block.

    Spatial blocks are rectangular contiguous regions, so bounds are well-defined
    and cheap to compute once per raster.
    """
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


def _disk_half_widths(kernel_half: int) -> List[int]:
    """Horizontal half-widths of a discrete disk kernel for each row offset."""
    widths = []
    for dy in range(-kernel_half, kernel_half + 1):
        widths.append(int(np.floor(np.sqrt(max(kernel_half * kernel_half - dy * dy, 0)))))
    return widths


def _focal_aggregate_block_aware(
    arr: np.ndarray,
    block_id_raster: np.ndarray,
    kernel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute block-aware focal sum and valid-pixel count.

    The disk kernel is decomposed into horizontal stripes and evaluated with
    prefix sums, so the running time is independent of the number of spatial
    blocks and is much faster than per-block convolution for large kernels.

    Returns
    -------
    value_sum, valid_count : both same shape as ``arr``.
        ``value_sum / valid_count`` gives the block-aware focal mean; dividing
        by the count gives the block-aware focal fraction when ``arr`` is
        binary.
    """
    height, width = arr.shape
    kernel_half = (kernel.shape[0] - 1) // 2

    valid = ~np.isnan(arr)
    values = np.where(valid, arr, 0.0)
    value_prefix = _prefix_sum(values)
    valid_prefix = _prefix_sum(valid.astype(np.float64))

    r_min, r_max, c_min, c_max = _block_bounds(block_id_raster)
    r_idx = np.arange(height)[:, None]
    c_idx = np.arange(width)[None, :]

    value_sum = np.zeros((height, width), dtype=np.float64)
    valid_count = np.zeros((height, width), dtype=np.float64)

    for dy, half_width in zip(range(-kernel_half, kernel_half + 1), _disk_half_widths(kernel_half)):
        rr = r_idx + dy
        row_inside_block = (rr >= r_min) & (rr <= r_max)
        rr_clip = np.clip(rr, 0, height - 1)

        c_lo = np.clip(np.clip(c_idx - half_width, c_min, c_max), 0, width - 1)
        c_hi = np.clip(np.clip(c_idx + half_width, c_min, c_max), 0, width - 1)

        stripe_value = _rect_sum(value_prefix, rr_clip, c_lo, c_hi)
        stripe_valid = _rect_sum(valid_prefix, rr_clip, c_lo, c_hi)

        value_sum += np.where(row_inside_block, stripe_value, 0.0)
        valid_count += np.where(row_inside_block, stripe_valid, 0.0)

    return value_sum, valid_count


def _focal_mean_block_aware(
    arr: np.ndarray,
    block_id_raster: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Block-aware focal mean, ignoring NaN values."""
    value_sum, valid_count = _focal_aggregate_block_aware(arr, block_id_raster, kernel)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(valid_count > 0, value_sum / valid_count, np.nan)
    return mean


def _focal_fraction_block_aware(
    binary_arr: np.ndarray,
    block_id_raster: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Block-aware focal fraction of ones in a binary array."""
    value_sum, valid_count = _focal_aggregate_block_aware(
        binary_arr.astype(np.float64), block_id_raster, kernel
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(valid_count > 0, value_sum / valid_count, np.nan)
    # Clip to [0, 1] to guard against tiny floating-point drift.
    return np.clip(fraction, 0.0, 1.0)


def _focal_count_full(kernel: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Full (unclipped) focal valid-pixel count for a disk kernel."""
    height, width = shape
    kernel_half = (kernel.shape[0] - 1) // 2
    ones_prefix = _prefix_sum(np.ones((height, width), dtype=np.float64))
    r_idx = np.arange(height)[:, None]
    c_idx = np.arange(width)[None, :]

    count = np.zeros((height, width), dtype=np.float64)
    for dy, half_width in zip(range(-kernel_half, kernel_half + 1), _disk_half_widths(kernel_half)):
        rr = r_idx + dy
        row_inside = (rr >= 0) & (rr < height)
        rr_clip = np.clip(rr, 0, height - 1)
        c_lo = np.clip(c_idx - half_width, 0, width - 1)
        c_hi = np.clip(c_idx + half_width, 0, width - 1)
        count += np.where(row_inside, _rect_sum(ones_prefix, rr_clip, c_lo, c_hi), 0.0)
    return count


def _count_clipped_pixels(
    block_id_raster: np.ndarray,
    kernel: np.ndarray,
) -> int:
    """Count pixels whose block-aware kernel is smaller than the full kernel.

    A pixel is clipped if any part of its neighbourhood lies outside its spatial
    block.  We detect this by comparing the block-aware valid-pixel count to the
    count obtained without a block mask.
    """
    block_count = _focal_aggregate_block_aware(
        np.ones_like(block_id_raster, dtype=np.float64),
        block_id_raster,
        kernel,
    )[1]
    full_count = _focal_count_full(kernel, block_id_raster.shape)
    return int((block_count < full_count).sum())


def _load_raster_array(path) -> np.ndarray:
    """Load the first band of a GeoTIFF as a float64 array."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64)


def _load_source_rasters(year: int) -> Dict[str, np.ndarray]:
    """Load all Phase 3/4 rasters needed for morphology features."""
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

    return {
        "dist_building_m": _load_raster_array(BUILDINGS_DISTANCE),
        "dist_road_m": _load_raster_array(ROADS_DISTANCE),
        "dist_vegetation_m": _load_raster_array(VEGETATION_DISTANCE),
        "vegetation_cover": _load_raster_array(
            VEGETATION_COVER_RASTER_2022 if year == 2022 else VEGETATION_COVER_RASTER_2026
        ),
        "ndvi": _load_raster_array(S2_NDVI_2022 if year == 2022 else S2_NDVI_2026),
        "ndbi": _load_raster_array(S2_NDBI_2022 if year == 2022 else S2_NDBI_2026),
        "landuse_class": _load_raster_array(LANDUSE_RASTER),
    }


def compute_morphology_clipping_summary(
    sampled_df: pd.DataFrame,
) -> Dict:
    """Compute the block-edge kernel-clipping summary for all radii.

    This is intentionally separate from `build_morphology_features` so the
    summary can be recomputed quickly from the sample table without reloading
    all source rasters.
    """
    with rasterio.open(REFERENCE_RASTER) as src:
        height, width = src.height, src.width

    block_id_raster = _compute_spatial_block_raster(sampled_df, height, width)

    total_raster_pixels = int(height * width)
    clipped_counts = {}
    sampled_clipped_counts = {}
    sampled_total_counts = {}
    for radius_m in MORPHOLOGY_RADII_M:
        radius_px = _radius_m_to_px(radius_m)
        kernel = _disk_kernel(radius_px)
        block_count = _focal_aggregate_block_aware(
            np.ones((height, width), dtype=np.float64), block_id_raster, kernel
        )[1]
        full_count = _focal_count_full(kernel, (height, width))
        clipped_mask = block_count < full_count
        clipped_counts[radius_m] = int(clipped_mask.sum())

        # Count affected pixels among the Phase 4 sampled rows.
        sampled_rows = sampled_df["row"].values.astype(int)
        sampled_cols = sampled_df["col"].values.astype(int)
        sampled_clipped_counts[radius_m] = int(clipped_mask[sampled_rows, sampled_cols].sum())
        sampled_total_counts[radius_m] = int(len(sampled_df))

    return {
        "note": (
            "Neighbourhoods are restricted to the same spatial block; kernels near "
            "block boundaries are clipped.  Counts below are reported both for the "
            "full reference grid and for the Phase 4 sampled rows.  The 500 m radius "
            "is most affected."
        ),
        "clipped_pixel_counts_full_grid": clipped_counts,
        "total_raster_pixels": total_raster_pixels,
        "sampled_clipped_pixel_counts": sampled_clipped_counts,
        "sampled_total_pixels": sampled_total_counts,
    }


def build_morphology_features(
    sampled_df: pd.DataFrame,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Build block-aware urban morphology features for all sampled pixels.

    Parameters
    ----------
    sampled_df : pd.DataFrame
        Phase 4 combined dataset with ``row``, ``col``, ``year`` columns.
    output_csv : str, optional
        If provided, save the resulting feature table to this path.

    Returns
    -------
    pd.DataFrame
        One row per sampled pixel-year, with columns ``row``, ``col``, ``year``
        and all morphology features.
    """
    with rasterio.open(REFERENCE_RASTER) as src:
        height, width = src.height, src.width

    block_id_raster = _compute_spatial_block_raster(sampled_df, height, width)

    records = []
    clipped_counts: Dict[int, int] = {}
    years = sorted(sampled_df["year"].unique())

    lu_classes = np.array(MORPHOLOGY_LANDUSE_CLASSES)

    for year in years:
        sub = sampled_df[sampled_df["year"] == year].copy()
        rows = sub["row"].values.astype(int)
        cols = sub["col"].values.astype(int)

        rasters = _load_source_rasters(year)
        feat_dict: Dict[str, np.ndarray] = {
            "row": rows,
            "col": cols,
            "year": np.full(len(rows), year),
        }

        for radius_m in MORPHOLOGY_RADII_M:
            radius_px = _radius_m_to_px(radius_m)
            kernel = _disk_kernel(radius_px)

            # Binary pixel-fraction features using distance == 0 semantics.
            feat_dict[f"building_pixel_fraction_{radius_m}m"] = _focal_fraction_block_aware(
                (rasters["dist_building_m"] == 0).astype(np.float64),
                block_id_raster,
                kernel,
            )[rows, cols]
            feat_dict[f"road_pixel_fraction_{radius_m}m"] = _focal_fraction_block_aware(
                (rasters["dist_road_m"] == 0).astype(np.float64),
                block_id_raster,
                kernel,
            )[rows, cols]
            feat_dict[f"vegetation_pixel_fraction_{radius_m}m"] = _focal_fraction_block_aware(
                (rasters["dist_vegetation_m"] == 0).astype(np.float64),
                block_id_raster,
                kernel,
            )[rows, cols]

            # Mean vegetation cover in the neighbourhood.
            feat_dict[f"vegetation_cover_mean_{radius_m}m"] = _focal_mean_block_aware(
                rasters["vegetation_cover"],
                block_id_raster,
                kernel,
            )[rows, cols]

            # NDVI / NDBI contrast (pixel value minus neighbourhood mean).
            ndvi_mean = _focal_mean_block_aware(rasters["ndvi"], block_id_raster, kernel)
            feat_dict[f"ndvi_contrast_{radius_m}m"] = (
                rasters["ndvi"][rows, cols] - ndvi_mean[rows, cols]
            )

            ndbi_mean = _focal_mean_block_aware(rasters["ndbi"], block_id_raster, kernel)
            feat_dict[f"ndbi_contrast_{radius_m}m"] = (
                rasters["ndbi"][rows, cols] - ndbi_mean[rows, cols]
            )

            # Land-use context features (meaningful classes only; class 0 is
            # unclassified/background and is excluded from the mixture).
            class_counts = np.stack(
                [_focal_aggregate_block_aware(
                    (rasters["landuse_class"] == c).astype(np.float64),
                    block_id_raster,
                    kernel,
                )[0][rows, cols] for c in lu_classes],
                axis=0,
            )
            total_count = class_counts.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                class_fracs = np.where(total_count > 0, class_counts / total_count, 0.0)
            # Clip to [0, 1] and fill any remaining NaN with 0.
            class_fracs = np.nan_to_num(np.clip(class_fracs, 0.0, 1.0), nan=0.0)
            for idx, c in enumerate(MORPHOLOGY_LANDUSE_CLASSES):
                feat_dict[f"landuse_frac_{c}_{radius_m}m"] = class_fracs[idx]

            dominant = np.where(
                total_count > 0,
                lu_classes[np.argmax(class_counts, axis=0)],
                0,
            )
            feat_dict[f"landuse_dominant_{radius_m}m"] = dominant

            with np.errstate(invalid="ignore", divide="ignore"):
                probs = np.where(total_count > 0, class_counts / total_count, 0.0)
                entropy = np.where(
                    total_count > 0,
                    -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=0),
                    0.0,
                )
            feat_dict[f"landuse_entropy_{radius_m}m"] = np.nan_to_num(entropy, nan=0.0)

            # Track clipping for this radius (only once, since it depends only on
            # the block raster and kernel).
            if radius_m not in clipped_counts:
                clipped_counts[radius_m] = _count_clipped_pixels(block_id_raster, kernel)

        records.append(pd.DataFrame(feat_dict))

    morphology_df = pd.concat(records, ignore_index=True)

    if output_csv is not None:
        from pathlib import Path

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        morphology_df.to_csv(output_csv, index=False)

    return morphology_df


def merge_morphology_features(
    df: pd.DataFrame,
    morphology_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge precomputed urban morphology features into the Phase 4 dataset."""
    from .config import USE_MORPHOLOGY_FEATURES

    if not USE_MORPHOLOGY_FEATURES:
        return df

    if morphology_df is None:
        cache_path = MORPHOLOGY_FEATURES_CSV
        if cache_path.exists():
            morphology_df = pd.read_csv(cache_path)
        else:
            morphology_df = build_morphology_features(df, output_csv=str(cache_path))

    return df.merge(morphology_df, on=["row", "col", "year"], how="left")


if __name__ == "__main__":
    # Standalone execution for testing/diagnostics.
    df = pd.read_csv(INPUT_DATASET_CSV)
    summary = compute_morphology_clipping_summary(df)
    print("Clipped-pixel summary:", summary)
    morphology_df = build_morphology_features(df, output_csv=str(MORPHOLOGY_FEATURES_CSV))
    print("Morphology features built:", morphology_df.shape)
