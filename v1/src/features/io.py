"""Raster and table I/O helpers for Phase 4.

These are thin wrappers around the audited Phase 3 I/O utilities so that the
features package stays self-contained while reusing the validated grid/profile
handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import rasterio

from preprocessing.io import (
    build_profile,
    describe_nan_pattern,
    open_raster,
    read_band,
    read_raster_metadata,
    write_raster,
)


def read_raster_array(path: Path, band: int | str = 1) -> np.ndarray:
    """Read a single-band raster as a float64 NumPy array.

    Preserves NaN values so downstream analyses see the same mask as Phase 3.
    """
    with open_raster(path) as ds:
        return read_band(ds, band, masked=True)


def read_raster_profile(path: Path) -> Tuple[Dict, rasterio.DatasetReader]:
    """Return (profile, dataset) for a raster."""
    ds = open_raster(path)
    return ds.profile.copy(), ds


def write_single_band_raster(
    path: Path,
    data: np.ndarray,
    reference_path: Path,
    dtype=None,
    nodata=None,
    description: str | None = None,
    compress: str = "lzw",
) -> None:
    """Write a single-band raster matching the reference grid/profile.

    Parameters
    ----------
    path : Path
        Destination path.
    data : np.ndarray
        2-D array to write.
    reference_path : Path
        Raster whose CRS, transform and shape should be copied.
    dtype : optional
        Output dtype. Defaults to the data array dtype.
    nodata : optional
        NoData value. Defaults to ``np.nan`` for float outputs.
    description : str, optional
        Band description to embed in the GeoTIFF.
    compress : str, default "lzw"
        Compression algorithm.
    """
    if dtype is None:
        dtype = data.dtype
    if nodata is None and np.issubdtype(dtype, np.floating):
        nodata = np.nan

    profile = build_profile(reference_path, dtype=dtype, nodata=nodata, count=1)
    profile.update(compress=compress)
    write_raster(path, data, profile, band_names=[description] if description else None)


def read_feature_table(path: Path) -> pd.DataFrame:
    """Read the Phase 3 feature table CSV."""
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write a DataFrame to CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def write_json(data: dict, path: Path) -> None:
    """Write a dictionary to a JSON file with readable formatting."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


__all__ = [
    "build_profile",
    "describe_nan_pattern",
    "open_raster",
    "read_band",
    "read_raster_array",
    "read_raster_metadata",
    "read_raster_profile",
    "write_raster",
    "write_single_band_raster",
    "read_feature_table",
    "write_csv",
    "write_json",
]
