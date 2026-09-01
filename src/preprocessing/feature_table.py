"""
GreenGrid AI — Phase 3: Data Preprocessing
ML-ready feature table generation.

This module stacks the aligned 30 m raster products and extracts a sample of
valid pixels into a tabular CSV. It does NOT train any model. Instead, it
includes a ``spatial_block_id`` column so that later phases can perform
spatially aware cross-validation rather than a naive random pixel split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio

from . import config
from .io import open_raster, read_band


def compute_spatial_block_ids(rows: np.ndarray, cols: np.ndarray, n_blocks: int = config.N_SPATIAL_BLOCKS) -> np.ndarray:
    """
    Assign each pixel to a spatial block based on its row and column.

    Blocks are arranged in a regular grid. The block id allows later phases to
    perform spatial cross-validation (e.g., leave-one-block-out) instead of
    naive random splits, which can leak information due to spatial
    autocorrelation.
    """
    row_bins = np.linspace(rows.min(), rows.max() + 1, n_blocks + 1)
    col_bins = np.linspace(cols.min(), cols.max() + 1, n_blocks + 1)

    row_block = np.digitize(rows, row_bins) - 1
    col_block = np.digitize(cols, col_bins) - 1

    # Clip to valid range in case of edge values
    row_block = np.clip(row_block, 0, n_blocks - 1)
    col_block = np.clip(col_block, 0, n_blocks - 1)

    block_id = row_block * n_blocks + col_block
    return block_id


def get_row_col_from_index(idx: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flat array indices to row and column coordinates."""
    rows = idx // width
    cols = idx % width
    return rows, cols


def get_lon_lat(rows: np.ndarray, cols: np.ndarray, transform: rasterio.Affine) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pixel row/col to longitude/latitude using the affine transform."""
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return np.array(xs), np.array(ys)


def read_values_at_indices(path: Path, idx: np.ndarray) -> np.ndarray:
    """Read a single-band raster and return values at flat indices."""
    with open_raster(path) as ds:
        arr = read_band(ds, 1)
    return arr.ravel()[idx]


def build_feature_table(
    aligned_dir: Path = config.ALIGNED_DIR,
    masks_dir: Path = config.MASKS_DIR,
    output_dir: Path = config.FEATURES_DIR,
    max_samples: int = config.MAX_FEATURE_SAMPLES,
    random_seed: int = config.RANDOM_SEED,
    n_blocks: int = config.N_SPATIAL_BLOCKS,
) -> Dict:
    """
    Build the ML-ready feature table.

    Features (per pixel, 30 m):
      - lon, lat, row, col, spatial_block_id
      - LST from Landsat 9 (2022 and 2026)
      - NDVI from Landsat 9 (2022 and 2026)
      - NDBI from Landsat 9 (2022 and 2026)
      - NDVI from Sentinel-2 (2022 and 2026)
      - NDBI from Sentinel-2 (2022 and 2026)
      - Land-use class code
      - Distance to nearest major road (m)
      - Distance to nearest vegetation area (m)
      - Distance to nearest building in the sample (m)

    The feature table is one CSV per year plus a combined CSV. It is sampled
    from valid pixels only (where the combined valid mask is True).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference metadata
    with open_raster(config.REFERENCE_RASTER) as ref:
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    # Load valid mask
    # The valid mask is stored as uint8 with value 1 for valid pixels and 0 for
    # invalid pixels. We read it without NaN conversion so that 0 remains 0.
    valid_mask_path = masks_dir / "valid_mask_30m.tif"
    with open_raster(valid_mask_path) as ds:
        valid = ds.read(1).astype(bool)
    valid_flat = valid.ravel()
    valid_indices = np.flatnonzero(valid_flat)

    if len(valid_indices) == 0:
        raise ValueError("No valid pixels found. Cannot build feature table.")

    # Sample valid pixels
    rng = np.random.default_rng(random_seed)
    if len(valid_indices) > max_samples:
        sample_idx = rng.choice(valid_indices, size=max_samples, replace=False)
    else:
        sample_idx = valid_indices.copy()
    sample_idx = np.sort(sample_idx)

    rows, cols = get_row_col_from_index(sample_idx, ref_width)
    lons, lats = get_lon_lat(rows, cols, ref_transform)
    spatial_block_id = compute_spatial_block_ids(rows, cols, n_blocks)

    # Helper to collect a feature
    def add_feature(name: str, path: Path):
        values = read_values_at_indices(path, sample_idx)
        features[name] = values

    # ── Build 2022 feature table ────────────────────────────────────────────
    features = {
        "lon": lons,
        "lat": lats,
        "row": rows,
        "col": cols,
        "spatial_block_id": spatial_block_id,
    }
    add_feature("lst_l9_2022_C", aligned_dir / "l9_2022_composite_lst_30m.tif")
    add_feature("ndvi_l9_2022", aligned_dir / "l9_2022_composite_ndvi_30m.tif")
    add_feature("ndbi_l9_2022", aligned_dir / "l9_2022_composite_ndbi_30m.tif")
    add_feature("ndvi_s2_2022", aligned_dir / "s2_2022_ndvi_30m.tif")
    add_feature("ndbi_s2_2022", aligned_dir / "s2_2022_ndbi_30m.tif")
    add_feature("landuse_class", masks_dir / "landuse_raster_30m.tif")
    add_feature("dist_road_m", masks_dir / "roads_distance_30m.tif")
    add_feature("dist_vegetation_m", masks_dir / "vegetation_distance_30m.tif")
    add_feature("dist_building_m", masks_dir / "buildings_distance_30m.tif")

    df_2022 = pd.DataFrame(features)
    df_2022["year"] = 2022
    path_2022 = output_dir / "feature_table_2022.csv"
    df_2022.to_csv(path_2022, index=False)

    # ── Build 2026 feature table ────────────────────────────────────────────
    features = {
        "lon": lons,
        "lat": lats,
        "row": rows,
        "col": cols,
        "spatial_block_id": spatial_block_id,
    }
    add_feature("lst_l9_2026_C", aligned_dir / "l9_2026_composite_lst_30m.tif")
    add_feature("ndvi_l9_2026", aligned_dir / "l9_2026_composite_ndvi_30m.tif")
    add_feature("ndbi_l9_2026", aligned_dir / "l9_2026_composite_ndbi_30m.tif")
    add_feature("ndvi_s2_2026", aligned_dir / "s2_2026_ndvi_30m.tif")
    add_feature("ndbi_s2_2026", aligned_dir / "s2_2026_ndbi_30m.tif")
    add_feature("landuse_class", masks_dir / "landuse_raster_30m.tif")
    add_feature("dist_road_m", masks_dir / "roads_distance_30m.tif")
    add_feature("dist_vegetation_m", masks_dir / "vegetation_distance_30m.tif")
    add_feature("dist_building_m", masks_dir / "buildings_distance_30m.tif")

    df_2026 = pd.DataFrame(features)
    df_2026["year"] = 2026
    path_2026 = output_dir / "feature_table_2026.csv"
    df_2026.to_csv(path_2026, index=False)

    # ── Combined feature table (long format) ────────────────────────────────
    # Rename year-specific columns to generic names so the combined table has
    # one row per pixel-year with no NaN values.
    generic_cols = {
        "lst_l9_2022_C": "lst_l9_C",
        "lst_l9_2026_C": "lst_l9_C",
        "ndvi_l9_2022": "ndvi_l9",
        "ndvi_l9_2026": "ndvi_l9",
        "ndbi_l9_2022": "ndbi_l9",
        "ndbi_l9_2026": "ndbi_l9",
        "ndvi_s2_2022": "ndvi_s2",
        "ndvi_s2_2026": "ndvi_s2",
        "ndbi_s2_2022": "ndbi_s2",
        "ndbi_s2_2026": "ndbi_s2",
    }
    df_2022_long = df_2022.rename(columns=generic_cols)
    df_2026_long = df_2026.rename(columns=generic_cols)
    df_combined = pd.concat([df_2022_long, df_2026_long], ignore_index=True)
    path_combined = output_dir / "feature_table.csv"
    df_combined.to_csv(path_combined, index=False)

    # ── Metadata ────────────────────────────────────────────────────────────
    metadata = {
        "random_seed": random_seed,
        "max_samples_requested": max_samples,
        "valid_pixels_total": int(len(valid_indices)),
        "samples_drawn": int(len(sample_idx)),
        "n_spatial_blocks": n_blocks,
        "reference_raster": str(config.REFERENCE_RASTER),
        "landuse_class_note": "Code 0 = unclassified/background (not a real OSM landuse class); codes 1-8 are mapped OSM classes.",
        "landuse_class_mapping": {
            0: "unclassified_background",
            1: "park",
            2: "forest",
            3: "grass",
            4: "commercial",
            5: "industrial",
            6: "residential",
            7: "retail",
            8: "farmland",
        },
        "columns": list(df_combined.columns),
        "dtypes": {c: str(df_combined[c].dtype) for c in df_combined.columns},
        "output_files": {
            "2022": str(path_2022),
            "2026": str(path_2026),
            "combined": str(path_combined),
        },
        "summary": {
            "lst_l9_2022_C": df_2022["lst_l9_2022_C"].describe().to_dict(),
            "lst_l9_2026_C": df_2026["lst_l9_2026_C"].describe().to_dict(),
            "ndvi_l9_2022": df_2022["ndvi_l9_2022"].describe().to_dict(),
            "ndvi_l9_2026": df_2026["ndvi_l9_2026"].describe().to_dict(),
            "ndbi_l9_2022": df_2022["ndbi_l9_2022"].describe().to_dict(),
            "ndbi_l9_2026": df_2026["ndbi_l9_2026"].describe().to_dict(),
            "ndvi_s2_2022": df_2022["ndvi_s2_2022"].describe().to_dict(),
            "ndvi_s2_2026": df_2026["ndvi_s2_2026"].describe().to_dict(),
            "ndbi_s2_2022": df_2022["ndbi_s2_2022"].describe().to_dict(),
            "ndbi_s2_2026": df_2026["ndbi_s2_2026"].describe().to_dict(),
        },
    }

    metadata_path = output_dir / "feature_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        "feature_table_2022": str(path_2022),
        "feature_table_2026": str(path_2026),
        "feature_table": str(path_combined),
        "metadata": str(metadata_path),
        "samples": int(len(sample_idx)),
    }
