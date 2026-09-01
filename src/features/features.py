"""Feature assembly, statistics, and combined-dataset creation.

This module:

1. Reads the authoritative Phase 4 feature rasters.
2. Extracts values at the Phase 3 sampled pixel locations using row/col indices.
3. Adds the newly derived Vegetation Cover.
4. Builds the combined urban environmental dataset (one row per pixel-year).
5. Computes descriptive statistics and optional correlation summaries.

The authoritative feature selections are documented in
``src/features/config.py``:

    LST  → Landsat 9
    NDVI → Sentinel-2
    NDBI → Sentinel-2
    Vegetation Cover → derived from Sentinel-2 NDVI
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from features.config import (
    AUTHORITATIVE_FEATURES,
    BUILDINGS_DISTANCE,
    COMBINED_DATASET_CSV,
    FEATURE_METADATA_JSON,
    L9_LST_2022,
    L9_LST_2026,
    LANDUSE_RASTER,
    PHASE3_FEATURE_TABLE,
    ROADS_DISTANCE,
    S2_NDBI_2022,
    S2_NDBI_2026,
    S2_NDVI_2022,
    S2_NDVI_2026,
    VEGETATION_COVER_RASTER_2022,
    VEGETATION_COVER_RASTER_2026,
    VEGETATION_DISTANCE,
)
from features.io import read_raster_array, read_feature_table, write_csv, write_json


# Mapping of year → authoritative raster paths
YEARLY_RASTERS = {
    2022: {
        "lst": L9_LST_2022,
        "ndvi": S2_NDVI_2022,
        "ndbi": S2_NDBI_2022,
        "vegetation_cover": VEGETATION_COVER_RASTER_2022,
    },
    2026: {
        "lst": L9_LST_2026,
        "ndvi": S2_NDVI_2026,
        "ndbi": S2_NDBI_2026,
        "vegetation_cover": VEGETATION_COVER_RASTER_2026,
    },
}

# Static context layers (same for both years)
STATIC_RASTERS = {
    "landuse_class": LANDUSE_RASTER,
    "dist_road_m": ROADS_DISTANCE,
    "dist_vegetation_m": VEGETATION_DISTANCE,
    "dist_building_m": BUILDINGS_DISTANCE,
}


def _extract_at_indices(
    raster_path: Path,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Extract pixel values from a raster at given row/col indices.

    Out-of-bounds indices are clipped to the valid range and returned as NaN.
    """
    arr = read_raster_array(raster_path)
    n_rows, n_cols = arr.shape

    # Clip to bounds for safe indexing; mark out-of-bounds as NaN.
    valid = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
    out = np.full(rows.shape, np.nan, dtype=np.float64)
    out[valid] = arr[rows[valid], cols[valid]]
    return out


def build_combined_dataset() -> pd.DataFrame:
    """Build the Phase 4 combined urban environmental dataset.

    Returns a DataFrame with one row per pixel-year containing authoritative
    remote-sensing features, the new vegetation cover, and static context
    layers. Continuous NDVI is retained separately from vegetation cover.
    """
    df_in = read_feature_table(PHASE3_FEATURE_TABLE)

    required = {"lon", "lat", "row", "col", "spatial_block_id", "year"}
    missing = required - set(df_in.columns)
    if missing:
        raise ValueError(f"Phase 3 feature table missing required columns: {missing}")

    frames: List[pd.DataFrame] = []
    for year, rasters in YEARLY_RASTERS.items():
        df_year = df_in[df_in["year"] == year].copy()
        if df_year.empty:
            raise ValueError(f"No rows found for year {year} in Phase 3 feature table")

        rows = df_year["row"].to_numpy(dtype=int)
        cols = df_year["col"].to_numpy(dtype=int)

        # Authoritative remote-sensing features
        df_year["lst_C"] = _extract_at_indices(rasters["lst"], rows, cols)
        df_year["ndvi"] = _extract_at_indices(rasters["ndvi"], rows, cols)
        df_year["ndbi"] = _extract_at_indices(rasters["ndbi"], rows, cols)
        df_year["vegetation_cover"] = _extract_at_indices(
            rasters["vegetation_cover"], rows, cols
        )

        # Static context layers (reuse Phase 3 values if available; otherwise
        # re-extract from rasters to ensure consistency).
        for col_name, raster_path in STATIC_RASTERS.items():
            if col_name in df_year.columns:
                # Keep existing values but validate via spot extraction
                continue
            df_year[col_name] = _extract_at_indices(raster_path, rows, cols)

        # Select final column order
        final_cols = [
            "lon",
            "lat",
            "row",
            "col",
            "spatial_block_id",
            "year",
            "lst_C",
            "ndvi",
            "ndbi",
            "vegetation_cover",
            "landuse_class",
            "dist_road_m",
            "dist_vegetation_m",
            "dist_building_m",
        ]
        frames.append(df_year[final_cols])

    df = pd.concat(frames, ignore_index=True)
    return df


def compute_feature_statistics(df: pd.DataFrame) -> Dict:
    """Compute descriptive statistics per feature and per year.

    Returns a nested dictionary with count, mean, std, min, 25%, 50%, 75%, max
    for each numeric feature, plus vegetation-cover class counts if a simple
    binary threshold is applied.
    """
    numeric_cols = [
        "lst_C",
        "ndvi",
        "ndbi",
        "vegetation_cover",
        "dist_road_m",
        "dist_vegetation_m",
        "dist_building_m",
    ]

    stats = {
        "overall": {"n_rows": int(len(df)), "n_years": int(df["year"].nunique())},
        "by_year": {},
        "by_feature": {},
    }

    for year, g in df.groupby("year"):
        year_stats = {}
        for col in numeric_cols:
            desc = g[col].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
            # Convert numpy types to Python scalars for JSON serialization.
            year_stats[col] = {k: float(v) for k, v in desc.items()}

        # Vegetation cover class counts using a simple descriptive threshold
        # (>= 0.5). This is reported as a summary, not as a hard classification.
        vc = g["vegetation_cover"].dropna()
        year_stats["vegetation_cover_classes"] = {
            "non_vegetated_lt_0.5": int((vc < 0.5).sum()),
            "vegetated_ge_0.5": int((vc >= 0.5).sum()),
            "vegetated_fraction": float((vc >= 0.5).mean()) if len(vc) else None,
        }
        stats["by_year"][str(year)] = year_stats

    # Overall statistics across both years
    for col in numeric_cols:
        desc = df[col].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        stats["by_feature"][col] = {k: float(v) for k, v in desc.items()}

    return stats


def compute_correlations(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute descriptive Pearson correlations between numeric features.

    These are exploratory associations, not causal or predictive models.
    """
    numeric_cols = [
        "lst_C",
        "ndvi",
        "ndbi",
        "vegetation_cover",
        "dist_road_m",
        "dist_vegetation_m",
        "dist_building_m",
    ]
    corr = df[numeric_cols].corr(method="pearson")
    result: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        result[col] = {
            other: float(corr.loc[col, other])
            for other in numeric_cols
            if col != other
        }
    return result


def create_combined_dataset_and_stats() -> Tuple[pd.DataFrame, Dict, Dict]:
    """Run the full feature assembly pipeline.

    Returns
    -------
    df : pd.DataFrame
        Combined urban environmental dataset.
    stats : dict
        Per-year and overall descriptive statistics.
    correlations : dict
        Descriptive Pearson correlations.
    """
    df = build_combined_dataset()
    stats = compute_feature_statistics(df)
    correlations = compute_correlations(df)

    write_csv(df, COMBINED_DATASET_CSV, index=False)

    metadata = {
        "authoritative_feature_sources": AUTHORITATIVE_FEATURES,
        "vegetation_cover_methodology": {
            "description": (
                "Proportional vegetation cover (PVC) derived from Sentinel-2 NDVI "
                "using a linear scaling between NDVI_soil=0.05 and NDVI_veg=0.80, "
                "clamped to [0, 1]."
            ),
            "note": (
                "PVC is a continuous proxy for vegetation abundance, not a direct "
                "physical canopy-fraction measurement. The reference values are a "
                "Phase 4 methodological assumption documented in the report."
            ),
        },
        "dataset": {
            "n_rows": int(len(df)),
            "n_years": int(df["year"].nunique()),
            "years": sorted(df["year"].unique().tolist()),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "output_csv": str(COMBINED_DATASET_CSV),
        },
        "statistics": stats,
        "correlations": correlations,
    }
    write_json(metadata, FEATURE_METADATA_JSON)

    return df, stats, correlations


if __name__ == "__main__":
    df, stats, corr = create_combined_dataset_and_stats()
    print(f"Combined dataset shape: {df.shape}")
    print(df.head().to_string(index=False))
