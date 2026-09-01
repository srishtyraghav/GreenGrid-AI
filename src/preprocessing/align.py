"""
GreenGrid AI — Phase 3: Data Preprocessing
Raster alignment, resampling, and aggregation.

Terminology used in this project:
- Reprojection: changing the coordinate reference system (CRS).
- Resampling: changing the pixel grid (resolution or alignment) without
  changing the CRS.
- Aggregation/downsampling: a special case of resampling where the output
  resolution is coarser than the input resolution and pixel values are
  combined (e.g., averaged) over larger areas.

All Phase 2 satellite products are already in EPSG:4326, so Phase 3 performs
resampling/aggregation only — no reprojection. The common analysis grid is
30 m, matching the native resolution of Landsat 9 LST.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from . import config
from .io import build_profile, open_raster, write_raster


def get_reference_profile(reference_path: Path = config.REFERENCE_RASTER) -> Dict:
    """Return the profile of the reference 30 m raster."""
    with open_raster(reference_path) as ref:
        return {
            "driver": "GTiff",
            "width": ref.width,
            "height": ref.height,
            "count": 1,
            "dtype": "float64",
            "crs": ref.crs,
            "transform": ref.transform,
            "nodata": np.nan,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }


def needs_resampling(src_path: Path, ref_path: Path = config.REFERENCE_RASTER) -> bool:
    """Check whether a raster has the same width/height/transform as the reference."""
    with open_raster(src_path) as src, open_raster(ref_path) as ref:
        same_dims = (src.width == ref.width) and (src.height == ref.height)
        same_transform = src.transform == ref.transform
        same_crs = src.crs == ref.crs
    return not (same_dims and same_transform and same_crs)


def resample_to_reference(
    src_path: Path,
    output_path: Path,
    ref_path: Path = config.REFERENCE_RASTER,
    resampling: Resampling = Resampling.average,
    band_name: Optional[str] = None,
) -> Path:
    """
    Resample/aggregate a raster to the reference 30 m grid.

    Parameters
    ----------
    src_path : Path
        Input raster.
    output_path : Path
        Output raster.
    ref_path : Path
        Reference raster defining the target grid.
    resampling : rasterio.enums.Resampling
        Resampling method. Default is ``average`` for downsampling
        reflectance/index rasters. For categorical data, use ``nearest``.
    band_name : str, optional
        Band description for the output.

    Returns
    -------
    Path
        Output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref_profile = get_reference_profile(ref_path)

    with open_raster(src_path) as src:
        src_data = src.read(1, masked=True).filled(np.nan)
        src_crs = src.crs
        src_transform = src.transform

    dst_data = np.empty((ref_profile["height"], ref_profile["width"]), dtype="float64")
    dst_data.fill(np.nan)

    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    write_raster(
        output_path,
        dst_data,
        ref_profile,
        band_names=[band_name] if band_name else None,
    )
    return output_path


def align_landsat9_indices(indices_dir: Path = config.INDICES_DIR, aligned_dir: Path = config.ALIGNED_DIR) -> Dict:
    """
    Ensure all Landsat 9 index rasters are on the common 30 m grid.

    Because Landsat 9 is already 30 m and shares the reference transform,
    this step typically copies/verifies rather than resamples.
    """
    aligned_dir.mkdir(parents=True, exist_ok=True)
    products = ["l9_2022_best", "l9_2022_composite", "l9_2026_best", "l9_2026_composite"]
    index_types = ["ndvi", "ndbi", "lst"]

    results = {}
    for prod in products:
        results[prod] = {}
        for idx in index_types:
            src = indices_dir / f"{prod}_{idx}_30m.tif"
            dst = aligned_dir / f"{prod}_{idx}_30m.tif"
            if not src.exists():
                raise FileNotFoundError(f"Expected index raster not found: {src}")

            if needs_resampling(src):
                print(f"[ALIGN] Resampling {src.name} → 30 m grid")
                resample_to_reference(src, dst, band_name=idx.upper())
            else:
                print(f"[ALIGN] {src.name} already on 30 m grid")
                # Copy to aligned dir to keep a clean output set
                import shutil
                shutil.copy(src, dst)
            results[prod][idx] = str(dst)
    return results


def align_sentinel2_indices(indices_dir: Path = config.INDICES_DIR, aligned_dir: Path = config.ALIGNED_DIR) -> Dict:
    """
    Aggregate Sentinel-2 NDVI (10 m) and NDBI (20 m) to the 30 m grid.

    Method:
    - NDVI native 10 m → 30 m using ``average`` aggregation.
    - NDBI native 20 m → 30 m using ``average`` aggregation.

    This is downsampling (aggregation), not reprojection, because the CRS is
    unchanged.
    """
    aligned_dir.mkdir(parents=True, exist_ok=True)
    products = ["s2_2022", "s2_2026"]
    results = {}

    for prod in products:
        ndvi_src = indices_dir / f"{prod}_ndvi_native_10m.tif"
        ndbi_src = indices_dir / f"{prod}_ndbi_native_20m.tif"

        ndvi_dst = aligned_dir / f"{prod}_ndvi_30m.tif"
        ndbi_dst = aligned_dir / f"{prod}_ndbi_30m.tif"

        print(f"[ALIGN] Aggregating {prod} NDVI 10 m → 30 m (average)")
        resample_to_reference(ndvi_src, ndvi_dst, resampling=Resampling.average, band_name="NDVI")

        print(f"[ALIGN] Aggregating {prod} NDBI 20 m → 30 m (average)")
        resample_to_reference(ndbi_src, ndbi_dst, resampling=Resampling.average, band_name="NDBI")

        results[prod] = {"ndvi": str(ndvi_dst), "ndbi": str(ndbi_dst)}
    return results


def align_all(indices_dir: Path = config.INDICES_DIR, aligned_dir: Path = config.ALIGNED_DIR) -> Dict:
    """Align Landsat 9 and Sentinel-2 index rasters to the 30 m common grid."""
    results = {}
    results["landsat9"] = align_landsat9_indices(indices_dir, aligned_dir)
    results["sentinel2"] = align_sentinel2_indices(indices_dir, aligned_dir)
    return results
