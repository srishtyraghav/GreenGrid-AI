"""Input loading and grid alignment for Phase 7 Stage 1.

Loads every Phase 7 input on the common 1768x1874 EPSG:4326 grid and
intersects EVERYTHING with the Phase 3 valid mask
(``data/processed/phase3/masks/valid_mask_30m.tif``).  Per the input audit,
the raw NDVI / NDBI / vegetation_cover / LST rasters cover MORE pixels than
the valid mask and use NaN (or -1.0) nodata; computations are therefore
restricted to valid-mask cells only, and every dynamic input is asserted
finite at every valid cell.  All per-cell values are returned as flat
arrays addressed by the valid-cell index order (``np.where(valid_mask)``).

Phases 5 and 6 are FROZEN: their outputs are used read-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import rasterio

from .config import (
    EXPECTED_VALID_PIXELS,
    LANDUSE_RASTER_PATH,
    LST_RASTERS,
    NDBI_RASTERS,
    NDVI_RASTERS,
    ROADS_DISTANCE_RASTER,
    SEVERITY_SCORE_RASTERS,
    VALID_MASK_RASTER,
    VEGETATION_COVER_RASTERS,
    VEGETATION_DISTANCE_RASTER,
    YEARS,
)


def _read_aligned(path: Path, reference: Dict) -> np.ndarray:
    """Read a raster and assert CRS/transform/shape match the reference."""
    with rasterio.open(path) as src:
        if src.crs != reference["crs"]:
            raise AssertionError(f"CRS mismatch: {path}")
        if src.transform != reference["transform"]:
            raise AssertionError(f"Transform mismatch: {path}")
        if (src.height, src.width) != (reference["height"], reference["width"]):
            raise AssertionError(f"Shape mismatch: {path}")
        return src.read(1)


def _finite_at_valid(arr: np.ndarray, valid: np.ndarray, name: str) -> np.ndarray:
    """Return flat per-valid-cell values, asserting finiteness at valid cells.

    Raw inputs whose coverage is WIDER than the valid mask are intersected
    with it here (audit sections 3/5): cells outside the valid mask are
    dropped entirely, never interpreted.
    """
    values = arr[valid].astype(np.float64)
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise AssertionError(f"{name}: {n_bad} non-finite values inside the valid mask")
    return values


def load_inputs() -> Dict:
    """Load and align every Phase 7 input, intersected with the valid mask.

    Returns
    -------
    dict
        ``valid_mask`` (bool full grid), ``rows``/``cols`` (valid-cell index
        arrays), ``profile`` (reference raster profile copied from a Phase 6
        severity_score raster), ``transform``, ``n_valid``, per-year flat
        float64 arrays (``severity_score``, ``lst``, ``ndvi``, ``ndbi``,
        ``vegetation_cover``) and static flat arrays (``landuse`` int,
        ``dist_road_m``, ``dist_vegetation_m``).

    Notes
    -----
    The reference profile (CRS / transform / shape) is copied from the Phase
    6 ``severity_score_2022`` raster and every input is asserted identical
    to it.  The valid-pixel count must equal 1,500,777 exactly.
    """
    with rasterio.open(VALID_MASK_RASTER) as src:
        valid_mask = src.read(1) == 1

    n_valid = int(valid_mask.sum())
    if n_valid != EXPECTED_VALID_PIXELS:
        raise AssertionError(
            f"valid mask has {n_valid} px, expected {EXPECTED_VALID_PIXELS}"
        )

    # Reference grid: a Phase 6 severity_score raster (Phase 6 outputs are
    # already masked to exactly the valid grid).
    ref_path = SEVERITY_SCORE_RASTERS[YEARS[0]]
    with rasterio.open(ref_path) as src:
        reference = {
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
        }
        profile = src.profile.copy()

    rows, cols = np.where(valid_mask)

    data: Dict = {
        "valid_mask": valid_mask,
        "rows": rows,
        "cols": cols,
        "profile": profile,
        "transform": reference["transform"],
        "n_valid": n_valid,
        "yearly": {},
        "static": {},
    }

    for year in YEARS:
        yearly: Dict[str, np.ndarray] = {}
        for name, path in (
            ("severity_score", SEVERITY_SCORE_RASTERS[year]),
            ("lst", LST_RASTERS[year]),
            ("ndvi", NDVI_RASTERS[year]),
            ("ndbi", NDBI_RASTERS[year]),
            ("vegetation_cover", VEGETATION_COVER_RASTERS[year]),
        ):
            arr = _read_aligned(path, reference)
            # Phase 6 rasters carry -1.0 outside valid cells; raw float64
            # rasters carry NaN.  Both are intersected with the valid mask
            # and additionally guarded against sentinel leakage.
            finite = np.isfinite(arr.astype(np.float64)) & (arr != -1.0)
            yearly[name] = _finite_at_valid(
                np.where(finite, arr, np.nan), valid_mask, f"{name}_{year}"
            )
        data["yearly"][year] = yearly

    landuse = _read_aligned(LANDUSE_RASTER_PATH, reference)
    data["static"]["landuse"] = landuse[valid_mask].astype(np.int64)
    for name, path in (
        ("dist_road_m", ROADS_DISTANCE_RASTER),
        ("dist_vegetation_m", VEGETATION_DISTANCE_RASTER),
    ):
        arr = _read_aligned(path, reference)
        data["static"][name] = _finite_at_valid(arr, valid_mask, name)

    return data


__all__ = ["load_inputs"]
