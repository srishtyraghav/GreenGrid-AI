"""Full-grid feature generation for Phase 6.

Builds the complete Phase 5 predictor set (base + spatial-neighbourhood +
urban-morphology features) for every valid-mask cell and year, reusing the
frozen Phase 5 machinery from ``src/models/spatial_features.py`` and
``src/models/morphology_features.py`` so that features are bit-identical to
those seen by the Phase 5 models at sampled cells.

Key mechanics (all inherited from Phase 5):

* The spatial block grid is derived from the ORIGINAL Phase 4 sampled
  dataframe, so block IDs and bin edges are identical to Phase 5; full-grid
  pixels receive block IDs under the same bins (clipped to [0, 4]).
* Neighbourhood features are block-aware: focal windows never cross a
  spatial-block boundary (kernel truncation at block edges is expected and
  matches Phase 5).
* Morphology features use the distance == 0 feature-pixel proxy and treat
  landuse nodata (255) as "no class", exactly as in Phase 5.
* The CSV ``row`` / ``col`` columns index the raster array directly as
  ``arr[row, col]`` (verified empirically in the Stage 1 pipeline).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

from models.config import (
    BASE_PREDICTOR_VARS,
    MORPHOLOGY_LANDUSE_CLASSES,
    MORPHOLOGY_RADII_M,
    PREDICTOR_VARS,
    SPATIAL_FEATURE_BASES,
    SPATIAL_WINDOW_SIZES,
    YEAR_VAR,
)
from models.morphology_features import (
    _disk_kernel,
    _focal_aggregate_block_aware,
    _focal_fraction_block_aware,
    _focal_mean_block_aware,
    _load_source_rasters,
    _radius_m_to_px,
)
from models.spatial_features import (
    _compute_spatial_block_raster,
    _focal_mean_block_aware as _spatial_focal_mean_block_aware,
    load_raster_for_feature,
)

from .config import VALID_MASK_RASTER


def load_valid_cells() -> Tuple[np.ndarray, np.ndarray, rasterio.Affine, Dict]:
    """Load the valid mask and return per-cell indices plus grid metadata.

    Returns
    -------
    tuple
        (rows, cols, transform, profile) where ``rows`` / ``cols`` are 1-D
        integer index arrays with ``arr[rows[i], cols[i]]`` addressing the
        raster grid directly.
    """
    with rasterio.open(VALID_MASK_RASTER) as src:
        mask = src.read(1)
        transform = src.transform
        profile = src.profile.copy()

    rows, cols = np.where(mask == 1)
    return rows.astype(np.int64), cols.astype(np.int64), transform, profile


def cell_coordinates(
    rows: np.ndarray,
    cols: np.ndarray,
    transform: rasterio.Affine,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return pixel-centre lon/lat coordinates for cell indices."""
    lon, lat = xy(transform, rows, cols, offset="center")
    return np.asarray(lon), np.asarray(lat)


def compute_fullgrid_block_raster(
    sampled_df: pd.DataFrame,
    height: int,
    width: int,
) -> np.ndarray:
    """Compute the full-grid spatial block raster under Phase 5 bins.

    ``sampled_df`` must be the original Phase 4 sampled dataset so that the
    digitize bins (and therefore block IDs) match Phase 5 exactly.
    """
    return _compute_spatial_block_raster(sampled_df, height, width)


def _spatial_features_for_year(
    year: int,
    block_id_raster: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Block-aware focal means of the 6 base features at 3×3/5×5/11×11."""
    features: Dict[str, np.ndarray] = {}
    for feature in SPATIAL_FEATURE_BASES:
        arr, _ = load_raster_for_feature(feature, year)
        for window in SPATIAL_WINDOW_SIZES:
            mean_raster = _spatial_focal_mean_block_aware(arr, block_id_raster, window)
            features[f"{feature}_mean{window}"] = mean_raster[rows, cols].astype(np.float32)
    return features


def _morphology_features_for_year(
    year: int,
    block_id_raster: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Block-aware disk-neighbourhood morphology features (4 radii)."""
    rasters = _load_source_rasters(year)
    lu_classes = np.array(MORPHOLOGY_LANDUSE_CLASSES)
    features: Dict[str, np.ndarray] = {}

    for radius_m in MORPHOLOGY_RADII_M:
        radius_px = _radius_m_to_px(radius_m)
        kernel = _disk_kernel(radius_px)

        # Binary pixel-fraction features using distance == 0 semantics.
        features[f"building_pixel_fraction_{radius_m}m"] = _focal_fraction_block_aware(
            (rasters["dist_building_m"] == 0).astype(np.float64),
            block_id_raster,
            kernel,
        )[rows, cols].astype(np.float32)
        features[f"road_pixel_fraction_{radius_m}m"] = _focal_fraction_block_aware(
            (rasters["dist_road_m"] == 0).astype(np.float64),
            block_id_raster,
            kernel,
        )[rows, cols].astype(np.float32)
        features[f"vegetation_pixel_fraction_{radius_m}m"] = _focal_fraction_block_aware(
            (rasters["dist_vegetation_m"] == 0).astype(np.float64),
            block_id_raster,
            kernel,
        )[rows, cols].astype(np.float32)

        # Mean vegetation cover in the neighbourhood.
        features[f"vegetation_cover_mean_{radius_m}m"] = _focal_mean_block_aware(
            rasters["vegetation_cover"],
            block_id_raster,
            kernel,
        )[rows, cols].astype(np.float32)

        # NDVI / NDBI contrast (pixel value minus neighbourhood mean).
        ndvi_mean = _focal_mean_block_aware(rasters["ndvi"], block_id_raster, kernel)
        features[f"ndvi_contrast_{radius_m}m"] = (
            rasters["ndvi"][rows, cols] - ndvi_mean[rows, cols]
        ).astype(np.float32)

        ndbi_mean = _focal_mean_block_aware(rasters["ndbi"], block_id_raster, kernel)
        features[f"ndbi_contrast_{radius_m}m"] = (
            rasters["ndbi"][rows, cols] - ndbi_mean[rows, cols]
        ).astype(np.float32)

        # Land-use context features (classes 1-8 only; nodata 255 pixels
        # contribute to no class and fall out of the denominator, exactly
        # as in the Phase 5 implementation).
        class_counts = np.stack(
            [
                _focal_aggregate_block_aware(
                    (rasters["landuse_class"] == c).astype(np.float64),
                    block_id_raster,
                    kernel,
                )[0][rows, cols]
                for c in lu_classes
            ],
            axis=0,
        )
        total_count = class_counts.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            class_fracs = np.where(total_count > 0, class_counts / total_count, 0.0)
        class_fracs = np.nan_to_num(np.clip(class_fracs, 0.0, 1.0), nan=0.0)
        for idx, c in enumerate(MORPHOLOGY_LANDUSE_CLASSES):
            features[f"landuse_frac_{c}_{radius_m}m"] = class_fracs[idx].astype(np.float32)

        dominant = np.where(
            total_count > 0,
            lu_classes[np.argmax(class_counts, axis=0)],
            0,
        )
        features[f"landuse_dominant_{radius_m}m"] = dominant.astype(np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            probs = np.where(total_count > 0, class_counts / total_count, 0.0)
            entropy = np.where(
                total_count > 0,
                -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=0),
                0.0,
            )
        features[f"landuse_entropy_{radius_m}m"] = np.nan_to_num(entropy, nan=0.0).astype(np.float32)

    return features


def build_fullgrid_features_for_year(
    year: int,
    sampled_df: pd.DataFrame,
    rows: np.ndarray,
    cols: np.ndarray,
    transform: rasterio.Affine,
) -> pd.DataFrame:
    """Build the full predictor set for all valid cells in one year.

    Parameters
    ----------
    year : int
        Observation year (2022 or 2026).
    sampled_df : pd.DataFrame
        Original Phase 4 sampled dataset; used only to derive the Phase 5
        spatial-block bins.
    rows, cols : np.ndarray
        Valid-cell index arrays (direct ``arr[row, col]`` addressing).
    transform : rasterio.Affine
        Grid affine transform, used for pixel-centre lon/lat.

    Returns
    -------
    pd.DataFrame
        One row per valid cell with identifiers, ``year`` and every column
        in ``PREDICTOR_VARS`` (raw, pre-encoding).
    """
    # Grid dimensions come from the reference profile via the valid mask.
    with rasterio.open(VALID_MASK_RASTER) as src:
        grid_height, grid_width = src.height, src.width

    block_id_raster = compute_fullgrid_block_raster(sampled_df, grid_height, grid_width)

    lon, lat = cell_coordinates(rows, cols, transform)

    data: Dict[str, np.ndarray] = {
        "lon": lon.astype(np.float64),
        "lat": lat.astype(np.float64),
        "row": rows,
        "col": cols,
        YEAR_VAR: np.full(len(rows), year, dtype=np.int64),
    }

    # Base predictors straight from the Phase 3/4 source rasters.
    # landuse_class has no entry in load_raster_for_feature; load it from
    # the Phase 3 landuse raster directly (nodata 255 preserved, exactly as
    # the Phase 5 morphology machinery treats it).
    from features.config import LANDUSE_RASTER

    for feature in BASE_PREDICTOR_VARS:
        if feature == YEAR_VAR:
            continue
        if feature == "landuse_class":
            with rasterio.open(LANDUSE_RASTER) as src:
                arr = src.read(1).astype(np.float64)
        else:
            arr, _ = load_raster_for_feature(feature, year)
        data[feature] = arr[rows, cols].astype(np.float32)

    # Spatial-neighbourhood features.
    data.update(_spatial_features_for_year(year, block_id_raster, rows, cols))

    # Urban morphology features.
    data.update(_morphology_features_for_year(year, block_id_raster, rows, cols))

    df = pd.DataFrame(data)

    # Match Phase 5 dtype conventions so one-hot dummy names align with the
    # training matrix (integer landuse codes, not float).
    df["landuse_class"] = df["landuse_class"].astype(int)
    for c in [c for c in df.columns if c.startswith("landuse_dominant_")]:
        df[c] = df[c].astype(int)

    missing = [c for c in PREDICTOR_VARS if c not in df.columns]
    if missing:
        raise ValueError(f"Full-grid feature frame missing predictor columns: {missing}")

    return df


def align_features_to_model(
    X: pd.DataFrame,
    model_feature_names: List[str],
) -> pd.DataFrame:
    """Align an encoded full-grid design matrix to the model's training columns.

    The full-grid landuse raster can contain codes absent from the Phase 4
    sample (e.g. park code 1), producing extra dummy columns; conversely, a
    category present in training may never occur full-grid.  Extra dummies
    are dropped and missing ones zero-filled so the column set matches the
    training feature names and order exactly.
    """
    extra = [c for c in X.columns if c not in model_feature_names]
    missing = [c for c in model_feature_names if c not in X.columns]
    if extra or missing:
        print(f"  Feature alignment: dropping {len(extra)} extra dummies, zero-filling {len(missing)}")
    X_aligned = X.reindex(columns=list(model_feature_names), fill_value=0)
    if list(X_aligned.columns) != list(model_feature_names):
        raise AssertionError("Feature alignment failed: column names/order mismatch.")
    return X_aligned
