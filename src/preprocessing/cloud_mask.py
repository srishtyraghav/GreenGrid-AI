"""
GreenGrid AI — Phase 3: Data Preprocessing
Valid-pixel mask creation.

Phase 2 GEE scripts already applied cloud masking:
  - Landsat 9: QA_PIXEL bits 3 (cloud shadow) and 4 (cloud) set to mask.
  - Sentinel-2: SCL classes 3, 8, 9, 10 (cloud shadow / cloud / cirrus) masked.

The exported GeoTIFFs represent masked pixels as NaN (nodata=None). This
module inspects that pattern and builds a single valid-pixel mask from the
existing NaN pattern. It does NOT re-apply cloud masking unless later evidence
shows it is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from . import config
from .io import build_profile, describe_nan_pattern, open_raster, read_band, write_raster


def build_valid_mask_from_nan(arr: np.ndarray) -> np.ndarray:
    """Return a boolean mask where True = finite (valid) pixel."""
    return np.isfinite(arr)


def create_combined_valid_mask(
    reference_path: Path = config.REFERENCE_RASTER,
    mask_paths: Dict[str, Path] | None = None,
    output_path: Path = config.MASKS_DIR / "valid_mask_30m.tif",
) -> Dict:
    """
    Build a combined valid-pixel mask for the common 30 m grid.

    The mask is True only where ALL input layers have finite values. If
    ``mask_paths`` is None, the mask is derived from the reference raster's
    first band (assumed to already contain the GEE NaN mask).

    Parameters
    ----------
    reference_path : Path
        Reference 30 m raster.
    mask_paths : dict, optional
        Mapping of layer names to aligned raster paths. If provided, the
        combined mask requires finiteness across every layer.
    output_path : Path
        Destination for the mask GeoTIFF.

    Returns
    -------
    dict
        Metadata including valid pixel count and NaN pattern summary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_raster(reference_path) as ref:
        ref_arr = read_band(ref, 1)
        valid = build_valid_mask_from_nan(ref_arr)
        pattern = describe_nan_pattern(ref_arr, "reference")

    if mask_paths:
        for name, path in mask_paths.items():
            with open_raster(path) as ds:
                arr = read_band(ds, 1)
            layer_valid = build_valid_mask_from_nan(arr)
            valid = valid & layer_valid
            pattern["additional_layers"] = pattern.get("additional_layers", [])
            pattern["additional_layers"].append(describe_nan_pattern(arr, name))

    valid_count = int(np.sum(valid))
    invalid_count = int(np.sum(~valid))

    # Use 1=valid, 0=invalid for a compact mask file
    profile = build_profile(reference_path, count=1, dtype="uint8", nodata=0)
    write_raster(output_path, valid.astype(np.uint8), profile, band_names=["valid_pixel"])

    return {
        "output": str(output_path),
        "valid_pixels": valid_count,
        "invalid_pixels": invalid_count,
        "valid_fraction": round(valid_count / valid.size, 4),
        "reference_pattern": pattern,
    }


def inspect_source_nan_patterns() -> Dict:
    """
    Inspect and document the NaN pattern in each source GeoTIFF.

    This is a read-only diagnostic used to justify the decision to keep the
    existing GEE cloud mask.
    """
    sources = {
        "l9_2022_best": config.LANDSAT9_DIR / "2022_07" / "landsat9_2022_07_best_scene_30m.tif",
        "l9_2022_composite": config.LANDSAT9_DIR / "2022_07" / "landsat9_2022_07_composite_30m.tif",
        "l9_2026_best": config.LANDSAT9_DIR / "2026_07" / "landsat9_2026_07_best_scene_30m.tif",
        "l9_2026_composite": config.LANDSAT9_DIR / "2026_07" / "landsat9_2026_07_composite_30m.tif",
        "s2_2022_10m": config.SENTINEL2_DIR / "2022_07" / "sentinel2_2022_07_10m_composite.tif",
        "s2_2022_20m": config.SENTINEL2_DIR / "2022_07" / "sentinel2_2022_07_20m_composite.tif",
        "s2_2026_10m": config.SENTINEL2_DIR / "2026_07" / "sentinel2_2026_07_10m_composite.tif",
        "s2_2026_20m": config.SENTINEL2_DIR / "2026_07" / "sentinel2_2026_07_20m_composite.tif",
    }

    results = {}
    for name, path in sources.items():
        with open_raster(path) as ds:
            arr = read_band(ds, 1)
        results[name] = describe_nan_pattern(arr, name)
    return results
