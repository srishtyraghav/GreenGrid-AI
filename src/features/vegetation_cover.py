"""Vegetation Cover derivation for Phase 4.

This module implements the single new methodological product introduced in
Phase 4: proportional Vegetation Cover (PVC). It is derived from the
authoritative Sentinel-2 NDVI using an empirical linear scaling between two
reference NDVI end-members.

See ``src/features/config.py`` for the documented rationale behind the
reference values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from features.config import (
    NDVI_SOIL_REFERENCE,
    NDVI_VEG_REFERENCE,
    REFERENCE_RASTER_PATH,
    S2_NDVI_2022,
    S2_NDVI_2026,
    VEGETATION_COVER_DESCRIPTION,
    VEGETATION_COVER_RASTER_2022,
    VEGETATION_COVER_RASTER_2026,
)
from features.io import read_raster_array, write_single_band_raster


def calculate_proportional_vegetation_cover(
    ndvi: np.ndarray,
    ndvi_soil: float = NDVI_SOIL_REFERENCE,
    ndvi_veg: float = NDVI_VEG_REFERENCE,
) -> np.ndarray:
    """Derive proportional vegetation cover from NDVI.

    Formula
    -------
        PVC = (NDVI - ndvi_soil) / (ndvi_veg - ndvi_soil)
        PVC = clamp(PVC, 0.0, 1.0)

    NaN pixels remain NaN. Values outside the reference range are clamped to
    [0, 1], so the output is always interpretable as a vegetation abundance
    fraction.

    Parameters
    ----------
    ndvi : np.ndarray
        NDVI array (may contain NaN).
    ndvi_soil : float, default ``NDVI_SOIL_REFERENCE``
        NDVI value interpreted as "bare / non-vegetated" reference.
    ndvi_veg : float, default ``NDVI_VEG_REFERENCE``
        NDVI value interpreted as "dense vegetation" reference.

    Returns
    -------
    np.ndarray
        Proportional vegetation cover in [0, 1] (float64), with NaN preserved.
    """
    if ndvi_veg <= ndvi_soil:
        raise ValueError("ndvi_veg must be greater than ndvi_soil")

    denom = ndvi_veg - ndvi_soil
    pvc = (ndvi - ndvi_soil) / denom
    pvc = np.clip(pvc, 0.0, 1.0)
    # Preserve the original NaN mask explicitly (clip turns NaN into NaN, but
    # re-apply to be safe against any future changes).
    pvc = np.where(np.isnan(ndvi), np.nan, pvc)
    return pvc.astype(np.float64)


def derive_vegetation_cover_for_year(
    ndvi_path: Path,
    output_path: Path,
    reference_path: Path = REFERENCE_RASTER_PATH,
    ndvi_soil: float = NDVI_SOIL_REFERENCE,
    ndvi_veg: float = NDVI_VEG_REFERENCE,
    year: int | None = None,
) -> dict:
    """Read NDVI, compute PVC, and write a GeoTIFF.

    Returns a metadata dictionary with formula parameters and summary stats.
    """
    ndvi = read_raster_array(ndvi_path)
    pvc = calculate_proportional_vegetation_cover(ndvi, ndvi_soil, ndvi_veg)

    description = VEGETATION_COVER_DESCRIPTION
    if year is not None:
        description = f"Vegetation Cover {year} — {description}"

    write_single_band_raster(
        output_path,
        pvc,
        reference_path,
        dtype=np.float64,
        nodata=np.nan,
        description=description,
    )

    valid = pvc[~np.isnan(pvc)]
    stats = {
        "year": year,
        "input_ndvi": str(ndvi_path),
        "output_raster": str(output_path),
        "formula": "(NDVI - ndvi_soil) / (ndvi_veg - ndvi_soil), clamped [0, 1]",
        "ndvi_soil": float(ndvi_soil),
        "ndvi_veg": float(ndvi_veg),
        "description": VEGETATION_COVER_DESCRIPTION,
        "n_pixels_total": int(pvc.size),
        "n_valid_pixels": int(valid.size),
        "n_nan_pixels": int(np.isnan(pvc).sum()),
        "min": float(np.nanmin(pvc)) if valid.size else None,
        "max": float(np.nanmax(pvc)) if valid.size else None,
        "mean": float(np.nanmean(pvc)) if valid.size else None,
        "median": float(np.nanmedian(pvc)) if valid.size else None,
        "std": float(np.nanstd(pvc)) if valid.size else None,
    }
    return stats


def derive_vegetation_cover_all_years() -> Tuple[dict, dict]:
    """Derive PVC for both years and return their metadata dicts."""
    stats_2022 = derive_vegetation_cover_for_year(
        S2_NDVI_2022, VEGETATION_COVER_RASTER_2022, year=2022
    )
    stats_2026 = derive_vegetation_cover_for_year(
        S2_NDVI_2026, VEGETATION_COVER_RASTER_2026, year=2026
    )
    return stats_2022, stats_2026


if __name__ == "__main__":
    s2022, s2026 = derive_vegetation_cover_all_years()
    import json

    print(json.dumps({"2022": s2022, "2026": s2026}, indent=2, default=str))
