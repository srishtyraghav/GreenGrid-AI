"""Full-grid raster writing for Phase 6.

Writes complete 1768×1874 EPSG:4326 rasters (valid-grid predictions,
per-class probabilities, confidence, ordinal severity score and observed
LST) on the Phase 3/4 reference grid, with LZW compression and explicit
NoData.  Pixels outside the valid mask remain NoData.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import rasterio

from models.maps import load_reference_profile

from .config import (
    LST_NODATA,
    LST_RASTERS,
    PROBABILITY_NODATA,
    SEVERITY_NODATA,
    confidence_raster_path,
    lst_raster_path,
    probability_raster_path,
    severity_raster_path,
    severity_score_raster_path,
)


def write_full_grid_raster(
    values: np.ndarray,
    output_path: Path,
    dtype: str,
    nodata,
) -> Dict:
    """Write a full-grid single-band raster, NoData outside valid cells.

    Parameters
    ----------
    values : np.ndarray
        Flat array of per-valid-cell values, in the same order as the
        valid-cell index arrays used upstream.
    output_path : Path
        Destination GeoTIFF path.
    dtype : str
        Output dtype ("int16" or "float32").
    nodata : int or float
        Explicit NoData value.

    Returns
    -------
    dict
        Output metadata (path, shape, crs, nodata, valid-pixel count).
    """
    profile = load_reference_profile()
    profile.update(
        {
            "dtype": dtype,
            "count": 1,
            "nodata": nodata,
            "compress": "lzw",
        }
    )

    raster = np.full((profile["height"], profile["width"]), nodata, dtype=dtype)
    valid_rows, valid_cols = _valid_cell_indices()
    raster[valid_rows, valid_cols] = values.astype(dtype)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(raster, 1)

    return {
        "output_path": str(output_path),
        "shape": list(raster.shape),
        "crs": str(profile["crs"]),
        "nodata": nodata,
        "n_valid_pixels": int(len(valid_rows)),
    }


_VALID_ROWS = None
_VALID_COLS = None


def _valid_cell_indices():
    """Cached valid-cell index arrays (direct arr[row, col] addressing)."""
    global _VALID_ROWS, _VALID_COLS
    if _VALID_ROWS is None:
        from .fullgrid import load_valid_cells

        _VALID_ROWS, _VALID_COLS, _, _ = load_valid_cells()
    return _VALID_ROWS, _VALID_COLS


def write_year_rasters(
    year: int,
    predicted_class: np.ndarray,
    probabilities: np.ndarray,
    confidence: np.ndarray,
    severity_score: np.ndarray,
) -> Dict:
    """Write the full prediction raster set for one year.

    Parameters
    ----------
    year : int
        Observation year.
    predicted_class : np.ndarray
        Integer class (0-3) per valid cell.
    probabilities : np.ndarray
        (n_valid, 4) per-class probabilities, columns ordered
        Low, Moderate, High, Severe.
    confidence : np.ndarray
        Max class probability per valid cell.
    severity_score : np.ndarray
        Expected ordinal score per valid cell.

    Returns
    -------
    dict
        Paths and metadata for every raster written.
    """
    from .config import CLASS_LABELS

    results: Dict[str, Dict] = {}

    results["severity"] = write_full_grid_raster(
        predicted_class, severity_raster_path(year), "int16", SEVERITY_NODATA
    )

    for i, label in enumerate(CLASS_LABELS):
        results[f"probability_{label.lower()}"] = write_full_grid_raster(
            probabilities[:, i], probability_raster_path(label, year), "float32", PROBABILITY_NODATA
        )

    results["confidence"] = write_full_grid_raster(
        confidence, confidence_raster_path(year), "float32", PROBABILITY_NODATA
    )

    # Ordinal model score: expected class index 0*P(Low)+...+3*P(Severe).
    # This is a relative ML severity score, NOT a physical temperature.
    results["severity_score"] = write_full_grid_raster(
        severity_score, severity_score_raster_path(year), "float32", PROBABILITY_NODATA
    )

    return results


def write_lst_rasters() -> Dict:
    """Write full-grid observed LST rasters for both years.

    Values are copied from the Phase 3 30 m-aligned Landsat 9 composite LST
    rasters; NaN (masked) pixels become NoData.  These are the most
    defensible continuous heat-intensity surfaces in the project.
    """
    results: Dict[int, Dict] = {}
    profile = load_reference_profile()

    for year, src_path in LST_RASTERS.items():
        with rasterio.open(src_path) as src:
            arr = src.read(1).astype(np.float32)
        out = np.where(np.isfinite(arr), arr, LST_NODATA).astype(np.float32)

        out_profile = profile.copy()
        out_profile.update(
            {
                "dtype": "float32",
                "count": 1,
                "nodata": LST_NODATA,
                "compress": "lzw",
            }
        )
        out_path = lst_raster_path(year)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(out, 1)

        results[year] = {
            "output_path": str(out_path),
            "source_path": str(src_path),
            "shape": list(out.shape),
            "crs": str(out_profile["crs"]),
            "nodata": LST_NODATA,
        }

    return results
