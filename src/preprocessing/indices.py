"""
GreenGrid AI — Phase 3: Data Preprocessing
Index calculation: NDVI, NDBI, and Land Surface Temperature (LST).

All formulas are applied to the already-scaled reflectance values exported
from Google Earth Engine. ST_B10 is the only band kept as raw DN and must be
converted to Celsius in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from . import config
from .io import (
    build_profile,
    describe_nan_pattern,
    open_raster,
    read_band,
    write_raster,
)


def _check_bands(arr_a: np.ndarray, arr_b: np.ndarray, name_a: str, name_b: str):
    """Verify shape compatibility and that arrays are 2-D."""
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"Band shape mismatch: {name_a} {arr_a.shape} vs {name_b} {arr_b.shape}"
        )


def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    NaN is preserved. Division-by-zero locations become NaN.
    """
    _check_bands(nir, red, "NIR", "Red")
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    return ndvi


def calculate_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Built-up Index.

    NDBI = (SWIR - NIR) / (SWIR + NIR)

    For Landsat 9: SWIR1 = SR_B6, NIR = SR_B5.
    For Sentinel-2: SWIR1 = B11, NIR = B8A (narrow NIR), because the 20 m
    composite exported from GEE does not include B8. B8 is only available in
    the 10 m composite and is used for NDVI, not NDBI.

    NaN is preserved. Division-by-zero locations become NaN.
    """
    _check_bands(swir, nir, "SWIR", "NIR")
    with np.errstate(divide="ignore", invalid="ignore"):
        ndbi = (swir - nir) / (swir + nir)
    return ndbi


def calculate_lst(st_b10: np.ndarray) -> np.ndarray:
    """
    Convert Landsat 9 Collection 2 Level-2 ST_B10 DN to LST in Celsius.

    K  = DN * 0.00341802 + 149.0
    °C = K - 273.15

    NaN is preserved.
    """
    with np.errstate(invalid="ignore"):
        lst_c = st_b10 * config.LST_SCALE + config.LST_OFFSET + config.KELVIN_TO_CELSIUS
    return lst_c


def _process_landsat9_file(input_path: Path, output_dir: Path, product_key: str) -> Dict:
    """
    Calculate NDVI, NDBI, and LST from one Landsat 9 GeoTIFF.

    Returns a dictionary of output paths and NaN-pattern metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = build_profile(config.REFERENCE_RASTER, count=1, dtype="float64")

    with open_raster(input_path) as ds:
        red = read_band(ds, "SR_B4")
        nir = read_band(ds, "SR_B5")
        swir1 = read_band(ds, "SR_B6")
        st_b10 = read_band(ds, "ST_B10")

    ndvi = calculate_ndvi(nir, red)
    ndbi = calculate_ndbi(swir1, nir)
    lst = calculate_lst(st_b10)

    ndvi_path = output_dir / f"{product_key}_ndvi_30m.tif"
    ndbi_path = output_dir / f"{product_key}_ndbi_30m.tif"
    lst_path = output_dir / f"{product_key}_lst_30m.tif"

    write_raster(ndvi_path, ndvi, profile, band_names=["NDVI"])
    write_raster(ndbi_path, ndbi, profile, band_names=["NDBI"])
    write_raster(lst_path, lst, profile, band_names=["LST_C"])

    return {
        "input": str(input_path),
        "ndvi": str(ndvi_path),
        "ndbi": str(ndbi_path),
        "lst": str(lst_path),
        "nan_patterns": {
            "red": describe_nan_pattern(red, "SR_B4 Red"),
            "nir": describe_nan_pattern(nir, "SR_B5 NIR"),
            "swir1": describe_nan_pattern(swir1, "SR_B6 SWIR1"),
            "st_b10": describe_nan_pattern(st_b10, "ST_B10"),
            "ndvi": describe_nan_pattern(ndvi, "NDVI"),
            "ndbi": describe_nan_pattern(ndbi, "NDBI"),
            "lst": describe_nan_pattern(lst, "LST_C"),
        },
    }


def process_landsat9(output_dir: Path = config.INDICES_DIR) -> Dict:
    """
    Calculate NDVI, NDBI, and LST for all four Landsat 9 products.

    Returns a nested dictionary keyed by product name.
    """
    products = {
        "l9_2022_best": config.LANDSAT9_DIR / "2022_07" / "landsat9_2022_07_best_scene_30m.tif",
        "l9_2022_composite": config.LANDSAT9_DIR / "2022_07" / "landsat9_2022_07_composite_30m.tif",
        "l9_2026_best": config.LANDSAT9_DIR / "2026_07" / "landsat9_2026_07_best_scene_30m.tif",
        "l9_2026_composite": config.LANDSAT9_DIR / "2026_07" / "landsat9_2026_07_composite_30m.tif",
    }

    results = {}
    for name, path in products.items():
        print(f"[INDICES] Processing Landsat 9: {name}")
        results[name] = _process_landsat9_file(path, output_dir, name)
    return results


def process_sentinel2(output_dir: Path = config.INDICES_DIR) -> Dict:
    """
    Calculate NDVI and NDBI for Sentinel-2 10 m and 20 m composites.

    NDVI uses the 10 m composite (B8, B4). NDBI uses the 20 m composite
    (B11, B8A) because B8 is not included in the 20 m export and B11 is not
    included in the 10 m export. Both indices are later aggregated to the
    common 30 m grid.

    Returns a nested dictionary keyed by product name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = build_profile(config.REFERENCE_RASTER, count=1, dtype="float64")

    products = {
        "s2_2022": (
            config.SENTINEL2_DIR / "2022_07" / "sentinel2_2022_07_10m_composite.tif",
            config.SENTINEL2_DIR / "2022_07" / "sentinel2_2022_07_20m_composite.tif",
        ),
        "s2_2026": (
            config.SENTINEL2_DIR / "2026_07" / "sentinel2_2026_07_10m_composite.tif",
            config.SENTINEL2_DIR / "2026_07" / "sentinel2_2026_07_20m_composite.tif",
        ),
    }

    results = {}
    for name, (path_10m, path_20m) in products.items():
        print(f"[INDICES] Processing Sentinel-2: {name}")

        with open_raster(path_10m) as ds:
            red = read_band(ds, "B4")
            nir = read_band(ds, "B8")
        ndvi = calculate_ndvi(nir, red)

        with open_raster(path_20m) as ds:
            swir1 = read_band(ds, "B11")
            nir_narrow = read_band(ds, "B8A")
        ndbi = calculate_ndbi(swir1, nir_narrow)

        ndvi_path = output_dir / f"{name}_ndvi_native_10m.tif"
        ndbi_path = output_dir / f"{name}_ndbi_native_20m.tif"

        write_raster(ndvi_path, ndvi, build_profile(path_10m, count=1, dtype="float64"), band_names=["NDVI"])
        write_raster(ndbi_path, ndbi, build_profile(path_20m, count=1, dtype="float64"), band_names=["NDBI"])

        results[name] = {
            "input_10m": str(path_10m),
            "input_20m": str(path_20m),
            "ndvi_native": str(ndvi_path),
            "ndbi_native": str(ndbi_path),
            "nan_patterns": {
                "red": describe_nan_pattern(red, "B4 Red"),
                "nir": describe_nan_pattern(nir, "B8 NIR"),
                "swir1": describe_nan_pattern(swir1, "B11 SWIR1"),
                "nir_narrow": describe_nan_pattern(nir_narrow, "B8A NIR narrow"),
                "ndvi": describe_nan_pattern(ndvi, "NDVI"),
                "ndbi": describe_nan_pattern(ndbi, "NDBI"),
            },
        }
    return results
