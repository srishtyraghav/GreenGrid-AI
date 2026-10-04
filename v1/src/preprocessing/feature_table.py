"""
GreenGrid AI — Phase 3: Data Preprocessing
ML-ready feature table generation.

This module stacks the aligned 30 m raster products and extracts samples of
valid pixels into tabular CSVs. It does NOT train any model. Instead, it
includes a ``spatial_block_id`` column so that later phases can perform
spatially aware cross-validation rather than a naive random pixel split.

Sampling design (Tier 2 hybrid, adopted with the 5-year extension):
  - Per-year INDEPENDENT sampling (up to ``max_samples`` per year): each
    year's table draws from pixels valid in that year. This preserves the
    full spatial diversity of every snapshot — the 2024/2025 monsoon holes
    no longer remove pixels from other years' training data. The combined
    long-format table is the modelling input.
  - PAIRED sampling (pixels valid in ALL years): written to
    ``feature_table_paired.csv`` and reserved for paired-cell temporal
    analytics (Phase 6/7 style comparisons), where the same pixel must be
    observed in every year.
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


def year_feature_paths(year: int, aligned_dir: Path = config.ALIGNED_DIR) -> Dict[str, Path]:
    """Aligned 30 m feature rasters for one year (Tier 1: 10 spectral layers)."""
    return {
        f"lst_l9_{year}_C": aligned_dir / f"l9_{year}_composite_lst_30m.tif",
        f"ndvi_l9_{year}": aligned_dir / f"l9_{year}_composite_ndvi_30m.tif",
        f"ndbi_l9_{year}": aligned_dir / f"l9_{year}_composite_ndbi_30m.tif",
        f"ndmi_l9_{year}": aligned_dir / f"l9_{year}_composite_ndmi_30m.tif",
        f"mndwi_l9_{year}": aligned_dir / f"l9_{year}_composite_mndwi_30m.tif",
        f"bsi_l9_{year}": aligned_dir / f"l9_{year}_composite_bsi_30m.tif",
        f"ndvi_s2_{year}": aligned_dir / f"s2_{year}_ndvi_30m.tif",
        f"ndbi_s2_{year}": aligned_dir / f"s2_{year}_ndbi_30m.tif",
        f"ndre_s2_{year}": aligned_dir / f"s2_{year}_ndre_30m.tif",
    }


def load_year_valid_mask(year: int, aligned_dir: Path = config.ALIGNED_DIR) -> np.ndarray:
    """Boolean mask of pixels where every feature layer of ``year`` is finite."""
    valid = None
    for path in year_feature_paths(year, aligned_dir).values():
        with open_raster(path) as ds:
            arr = read_band(ds, 1)
        layer = np.isfinite(arr)
        valid = layer if valid is None else (valid & layer)
    return valid


def build_feature_table(
    aligned_dir: Path = config.ALIGNED_DIR,
    masks_dir: Path = config.MASKS_DIR,
    output_dir: Path = config.FEATURES_DIR,
    max_samples: int = config.MAX_FEATURE_SAMPLES,
    random_seed: int = config.RANDOM_SEED,
    n_blocks: int = config.N_SPATIAL_BLOCKS,
) -> Dict:
    """
    Build the ML-ready feature tables.

    Per-year table columns:
      - lon, lat, row, col, spatial_block_id
      - L9: lst, ndvi, ndbi, ndmi, mndwi, bsi (per year)
      - S2: ndvi, ndbi, ndre (per year)
      - static: landuse_class, dist_road_m, dist_vegetation_m, dist_building_m

    Outputs:
      - feature_table_<year>.csv  (independent per-year samples)
      - feature_table.csv         (combined long format, modelling input)
      - feature_table_paired.csv  (same pixels in all years; temporal analytics)
      - feature_metadata.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference metadata
    with open_raster(config.REFERENCE_RASTER) as ref:
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    static_features = {
        "landuse_class": masks_dir / "landuse_raster_30m.tif",
        "dist_road_m": masks_dir / "roads_distance_30m.tif",
        "dist_vegetation_m": masks_dir / "vegetation_distance_30m.tif",
        "dist_building_m": masks_dir / "buildings_distance_30m.tif",
    }

    rng = np.random.default_rng(random_seed)

    def sample_from_mask(valid: np.ndarray) -> np.ndarray:
        indices = np.flatnonzero(valid.ravel())
        if len(indices) == 0:
            raise ValueError("No valid pixels found. Cannot build feature table.")
        if len(indices) > max_samples:
            indices = rng.choice(indices, size=max_samples, replace=False)
        return np.sort(indices)

    def build_frame(year: int, sample_idx: np.ndarray) -> pd.DataFrame:
        rows, cols = get_row_col_from_index(sample_idx, ref_width)
        lons, lats = get_lon_lat(rows, cols, ref_transform)
        features = {
            "lon": lons,
            "lat": lats,
            "row": rows,
            "col": cols,
            "spatial_block_id": compute_spatial_block_ids(rows, cols, n_blocks),
        }
        for name, path in year_feature_paths(year, aligned_dir).items():
            features[name] = read_values_at_indices(path, sample_idx)
        for name, path in static_features.items():
            features[name] = read_values_at_indices(path, sample_idx)
        df_year = pd.DataFrame(features)
        df_year["year"] = year
        return df_year

    # ── Per-year independent samples (modelling input) ─────────────────────
    year_frames: Dict[int, pd.DataFrame] = {}
    year_paths: Dict[int, Path] = {}
    year_masks: Dict[int, np.ndarray] = {}

    for year in config.YEARS:
        valid_year = load_year_valid_mask(year, aligned_dir)
        year_masks[year] = valid_year
        sample_idx = sample_from_mask(valid_year)
        df_year = build_frame(year, sample_idx)
        path_year = output_dir / f"feature_table_{year}.csv"
        df_year.to_csv(path_year, index=False)
        year_frames[year] = df_year
        year_paths[year] = path_year
        print(f"[FEATURE TABLE] Built {path_year.name}: {len(df_year)} rows "
              f"({int(valid_year.sum()):,} valid pixels available)")

    # ── Combined long-format table ─────────────────────────────────────────
    long_frames = []
    for year in config.YEARS:
        generic_cols = {
            f"lst_l9_{year}_C": "lst_l9_C",
            f"ndvi_l9_{year}": "ndvi_l9",
            f"ndbi_l9_{year}": "ndbi_l9",
            f"ndmi_l9_{year}": "ndmi_l9",
            f"mndwi_l9_{year}": "mndwi_l9",
            f"bsi_l9_{year}": "bsi_l9",
            f"ndvi_s2_{year}": "ndvi_s2",
            f"ndbi_s2_{year}": "ndbi_s2",
            f"ndre_s2_{year}": "ndre_s2",
        }
        long_frames.append(year_frames[year].rename(columns=generic_cols))
    df_combined = pd.concat(long_frames, ignore_index=True)
    path_combined = output_dir / "feature_table.csv"
    df_combined.to_csv(path_combined, index=False)

    # ── Paired sample (valid in ALL years; temporal analytics) ──────────────
    paired_mask = year_masks[config.YEARS[0]].copy()
    for year in config.YEARS[1:]:
        paired_mask &= year_masks[year]
    paired_idx = sample_from_mask(paired_mask)
    paired_frames = []
    for year in config.YEARS:
        df_paired = build_frame(year, paired_idx)
        generic_cols = {c: c.replace(f"_{year}", "") for c in df_paired.columns if c.endswith(f"_{year}")}
        # Rename year-suffixed spectral columns to generic names
        generic_cols = {}
        for col in df_paired.columns:
            if col.startswith(("lst_l9_", "ndvi_l9_", "ndbi_l9_", "ndmi_l9_", "mndwi_l9_", "bsi_l9_", "ndvi_s2_", "ndbi_s2_", "ndre_s2_")):
                generic_cols[col] = col[: col.rfind("_")]
        paired_frames.append(df_paired.rename(columns=generic_cols))
    df_paired_long = pd.concat(paired_frames, ignore_index=True)
    path_paired = output_dir / "feature_table_paired.csv"
    df_paired_long.to_csv(path_paired, index=False)
    print(f"[FEATURE TABLE] Built {path_paired.name}: {len(df_paired_long)} rows "
          f"({len(paired_idx):,} paired pixels x {len(config.YEARS)} years)")

    # ── Metadata ────────────────────────────────────────────────────────────
    summary = {}
    for year in config.YEARS:
        df_year = year_frames[year]
        for col in df_year.columns:
            if col.startswith(("lst_l9_", "ndvi_l9_", "ndbi_l9_", "ndmi_l9_", "mndwi_l9_", "bsi_l9_", "ndvi_s2_", "ndbi_s2_", "ndre_s2_")):
                summary[col] = df_year[col].describe().to_dict()

    metadata = {
        "random_seed": random_seed,
        "max_samples_requested_per_year": max_samples,
        "sampling_design": "per-year independent samples + all-year paired subset",
        "years": config.YEARS,
        "valid_pixels_per_year": {str(y): int(year_masks[y].sum()) for y in config.YEARS},
        "paired_valid_pixels": int(paired_mask.sum()),
        "samples_drawn_per_year": int(len(year_frames[config.YEARS[0]])),
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
            **{str(year): str(path) for year, path in year_paths.items()},
            "combined": str(path_combined),
            "paired": str(path_paired),
        },
        "summary": summary,
    }

    metadata_path = output_dir / "feature_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        **{f"feature_table_{year}": str(path) for year, path in year_paths.items()},
        "feature_table": str(path_combined),
        "feature_table_paired": str(path_paired),
        "metadata": str(metadata_path),
        "samples_per_year": int(len(year_frames[config.YEARS[0]])),
        "paired_pixels": int(len(paired_idx)),
    }
