"""
GreenGrid AI — Phase 3: Data Preprocessing
Raster input/output utilities with explicit NaN / masked-pixel handling.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.profiles import DefaultGTiffProfile

from . import config


def ensure_dir(path: Path) -> Path:
    """Create the directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_raster(path: Path):
    """Open a GeoTIFF with rasterio. Returns a dataset object (use as context manager)."""
    return rasterio.open(path)


def get_band_index(dataset, band_name: str) -> int:
    """
    Return the 1-based band index for a named band.

    The band description is checked first, then the configured band map.
    """
    descriptions = dataset.descriptions
    if descriptions and band_name in descriptions:
        return descriptions.index(band_name) + 1

    # Fallback to configured maps based on band count
    if dataset.count == 8 and band_name in config.L9_BANDS:
        return config.L9_BANDS[band_name]
    if dataset.count == 4 and band_name in config.S2_10M_BANDS:
        return config.S2_10M_BANDS[band_name]
    if dataset.count == 7 and band_name in config.S2_20M_BANDS:
        return config.S2_20M_BANDS[band_name]

    raise ValueError(
        f"Band '{band_name}' not found in {path.name}. "
        f"Descriptions: {descriptions}"
    )


def read_band(dataset, band: int | str, masked: bool = True) -> np.ndarray:
    """
    Read a single band as a NumPy array.

    Parameters
    ----------
    dataset : rasterio.DatasetReader
    band : int or str
        1-based band index or band name.
    masked : bool, default True
        If True, masked pixels are returned as NaN. This preserves the
        existing GEE cloud mask without introducing a second masking step.

    Returns
    -------
    np.ndarray
        2-D array. Masked pixels are NaN if masked=True.
    """
    idx = band if isinstance(band, int) else get_band_index(dataset, band)
    arr = dataset.read(idx, masked=masked)
    if masked:
        # Convert rasterio masked array to float array with NaN for missing data.
        # Cast before filling so NaN can be used for integer source dtypes.
        arr = arr.astype(np.float64).filled(np.nan)
    return arr.astype(np.float64)


def read_raster_metadata(path: Path) -> Dict:
    """Return a dictionary of essential raster metadata."""
    with rasterio.open(path) as ds:
        return {
            "path": str(path),
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "crs": str(ds.crs),
            "transform": ds.transform,
            "bounds": ds.bounds,
            "resolution": ds.res,
            "nodata": ds.nodata,
            "dtype": ds.dtypes[0],
            "descriptions": ds.descriptions,
        }


def build_profile(
    reference_path: Path,
    count: int = 1,
    dtype: str = "float64",
    nodata=None,
    compress: str = "lzw",
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
) -> Dict:
    """
    Build a GeoTIFF profile matching the reference raster's grid.

    Parameters
    ----------
    reference_path : Path
        Raster defining width, height, transform, CRS, and bounds.
    count : int
        Number of output bands.
    dtype : str
        Output data type.
    nodata : optional
        NoData value. Defaults to NaN for float dtypes; must be supplied
        explicitly for integer dtypes.
    compress : str
        Compression algorithm to keep output files small.

    Returns
    -------
    dict
        Rasterio-compatible profile.
    """
    if nodata is None:
        nodata = np.nan if dtype.startswith("float") else 0

    with rasterio.open(reference_path) as ref:
        profile = DefaultGTiffProfile(
            driver="GTiff",
            width=ref.width,
            height=ref.height,
            count=count,
            dtype=dtype,
            crs=ref.crs,
            transform=ref.transform,
            nodata=nodata,
            compress=compress,
            tiled=tiled,
            blockxsize=blockxsize,
            blockysize=blockysize,
        )
    return profile


def write_raster(
    path: Path,
    data: np.ndarray,
    profile: Dict,
    band_names: Optional[List[str]] = None,
) -> Path:
    """
    Write a NumPy array to a GeoTIFF.

    Parameters
    ----------
    path : Path
        Output file path.
    data : np.ndarray
        2-D array (single band) or 3-D array (bands, height, width).
    profile : dict
        Rasterio profile.
    band_names : list of str, optional
        Descriptions for each band.
    """
    ensure_dir(path.parent)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    # Ensure data shape matches profile count
    if data.shape[0] != profile["count"]:
        raise ValueError(
            f"Data has {data.shape[0]} bands but profile count is {profile['count']}"
        )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if band_names:
            for i, name in enumerate(band_names):
                dst.set_band_description(i + 1, name)
    return path


def load_study_area(path: Path = config.STUDY_AREA_PATH):
    """Load the study area GeoJSON as a GeoDataFrame."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(config.TARGET_CRS)
    return gdf


def describe_nan_pattern(arr: np.ndarray, label: str = "") -> Dict:
    """
    Return a dictionary describing the NaN / finite pattern of an array.

    Useful for documenting the existing GEE cloud mask without modifying it.
    """
    total = arr.size
    finite = np.isfinite(arr)
    finite_count = int(np.sum(finite))
    nan_count = int(np.sum(~finite))
    return {
        "label": label,
        "total_pixels": total,
        "finite_pixels": finite_count,
        "nan_pixels": nan_count,
        "nan_fraction": round(nan_count / total, 4) if total else None,
        "finite_min": float(np.min(arr[finite])) if finite_count else None,
        "finite_max": float(np.max(arr[finite])) if finite_count else None,
        "finite_mean": float(np.mean(arr[finite])) if finite_count else None,
    }
