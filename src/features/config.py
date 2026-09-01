"""Central configuration for Phase 4 feature extraction.

Phase 4 builds directly on the validated Phase 3 outputs and adds the one
new methodological component: proportional Vegetation Cover. All other
remote-sensing indices are reused unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve project root robustly whether this file is imported directly or
# via a package import.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
src_dir = PROJECT_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from preprocessing.config import (  # noqa: E402
    PROCESSED_DIR,
    REFERENCE_RASTER,
    TARGET_CRS,
    TARGET_RESOLUTION_M_APPROX,
)

# ---------------------------------------------------------------------------
# Phase 4 directories
# ---------------------------------------------------------------------------
# Phase 4 outputs live at data/processed/phase4, alongside data/processed/phase3.
PHASE4_DIR = PROJECT_ROOT / "data" / "processed" / "phase4"
PHASE4_FEATURES_DIR = PHASE4_DIR / "features"
PHASE4_MAPS_DIR = PHASE4_DIR / "maps"
PHASE4_TABLES_DIR = PHASE4_DIR / "tables"

for _d in (PHASE4_FEATURES_DIR, PHASE4_MAPS_DIR, PHASE4_TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Phase 3 inputs reused by Phase 4
# ---------------------------------------------------------------------------
# NOTE: preprocessing.config.PROCESSED_DIR already points to data/processed/phase3.
PHASE3_ALIGNED_DIR = PROCESSED_DIR / "aligned"
PHASE3_MASKS_DIR = PROCESSED_DIR / "masks"
PHASE3_ML_DIR = PROCESSED_DIR / "ml"

# Landsat 9 features (thermal only; LST is the authoritative thermal product)
L9_LST_2022 = PHASE3_ALIGNED_DIR / "l9_2022_composite_lst_30m.tif"
L9_LST_2026 = PHASE3_ALIGNED_DIR / "l9_2026_composite_lst_30m.tif"

# Sentinel-2 features (authoritative for spectral vegetation/built-up indices)
S2_NDVI_2022 = PHASE3_ALIGNED_DIR / "s2_2022_ndvi_30m.tif"
S2_NDVI_2026 = PHASE3_ALIGNED_DIR / "s2_2026_ndvi_30m.tif"
S2_NDBI_2022 = PHASE3_ALIGNED_DIR / "s2_2022_ndbi_30m.tif"
S2_NDBI_2026 = PHASE3_ALIGNED_DIR / "s2_2026_ndbi_30m.tif"

# Optional Landsat 9 NDVI/NDBI for reference/comparison only.
L9_NDVI_2022 = PHASE3_ALIGNED_DIR / "l9_2022_composite_ndvi_30m.tif"
L9_NDVI_2026 = PHASE3_ALIGNED_DIR / "l9_2026_composite_ndvi_30m.tif"
L9_NDBI_2022 = PHASE3_ALIGNED_DIR / "l9_2022_composite_ndbi_30m.tif"
L9_NDBI_2026 = PHASE3_ALIGNED_DIR / "l9_2026_composite_ndbi_30m.tif"

# Masks / context layers
VALID_MASK = PHASE3_MASKS_DIR / "valid_mask_30m.tif"
LANDUSE_RASTER = PHASE3_MASKS_DIR / "landuse_raster_30m.tif"
ROADS_DISTANCE = PHASE3_MASKS_DIR / "roads_distance_30m.tif"
VEGETATION_DISTANCE = PHASE3_MASKS_DIR / "vegetation_distance_30m.tif"
BUILDINGS_DISTANCE = PHASE3_MASKS_DIR / "buildings_distance_30m.tif"

# Phase 3 feature table (reused for row/col/geometry/spatial_block_id)
PHASE3_FEATURE_TABLE = PHASE3_ML_DIR / "feature_table.csv"

# ---------------------------------------------------------------------------
# Authoritative feature selections for the combined dataset
# ---------------------------------------------------------------------------
# Rationale:
#   * LST comes from Landsat 9 Band 10, which is the only thermal measurement
#     available in the Phase 3 archive; therefore LST is Landsat 9.
#   * NDVI, NDBI and Vegetation Cover are derived from Sentinel-2 because the
#     native 10/20 m resolution is aggregated to the common 30 m grid, giving
#     a sharper input than Landsat 9's 30 m native bands.
# Landsat 9 NDVI/NDBI rasters are still read for diagnostics, comparison maps,
# and validation but are not the authoritative values used in the combined
# environmental dataset.
AUTHORITATIVE_FEATURES = {
    "lst": "landsat9",
    "ndvi": "sentinel2",
    "ndbi": "sentinel2",
    "vegetation_cover": "sentinel2_ndvi",
}

# ---------------------------------------------------------------------------
# Vegetation Cover methodology
# ---------------------------------------------------------------------------
# Phase 4 derives proportional Vegetation Cover (PVC) from the authoritative
# Sentinel-2 NDVI using a simple linear unmixing / scaling between two
# reference NDVI values. This is a standard empirical approximation, NOT a
# direct physical measurement of canopy cover.
#
#   PVC = clamp((NDVI - NDVI_SOIL) / (NDVI_VEG - NDVI_SOIL), 0, 1)
#
# NDVI_SOIL = 0.05  -- Lower reference. Bare soil, built-up surfaces and very
#                      sparse vegetation in arid/sub-tropical urban settings
#                      commonly fall near 0.0-0.1. 0.05 is a conservative
#                      centre-of-range value for "effectively non-vegetated".
# NDVI_VEG  = 0.80  -- Upper reference. Dense, healthy green vegetation in the
#                      Delhi region during the growing season can reach 0.7-0.9.
#                      0.80 represents near-full vegetation cover and matches
#                      the typical upper bound used in proportional-cover
#                      studies.
#
# These values are a Phase 4 methodological assumption. They are NOT derived
# from a site-specific end-member extraction (e.g., pixel-pure spectra from
# high-resolution reference imagery) and therefore PVC should be interpreted
# as a proportional proxy, not a literal canopy percentage.
# ---------------------------------------------------------------------------
NDVI_SOIL_REFERENCE = 0.05
NDVI_VEG_REFERENCE = 0.80

VEGETATION_COVER_DESCRIPTION = (
    "Proportional vegetation cover (PVC) derived from Sentinel-2 NDVI using "
    f"NDVI_soil={NDVI_SOIL_REFERENCE} and NDVI_veg={NDVI_VEG_REFERENCE}. "
    "Values are clamped to [0, 1] and represent a continuous proxy for "
    "vegetation abundance, not a direct physical canopy fraction."
)

# Naming conventions for outputs
VEGETATION_COVER_RASTER_2022 = PHASE4_FEATURES_DIR / "vegetation_cover_2022_30m.tif"
VEGETATION_COVER_RASTER_2026 = PHASE4_FEATURES_DIR / "vegetation_cover_2026_30m.tif"

COMBINED_DATASET_CSV = PHASE4_TABLES_DIR / "combined_urban_environmental_dataset.csv"
FEATURE_METADATA_JSON = PHASE4_DIR / "phase4_feature_metadata.json"
PIPELINE_RECORD_JSON = PHASE4_DIR / "phase4_pipeline_record.json"

# ---------------------------------------------------------------------------
# Reproducibility / randomness
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# CRS / grid metadata
# ---------------------------------------------------------------------------
CRS = TARGET_CRS
REFERENCE_RASTER_PATH = REFERENCE_RASTER
NOMINAL_RESOLUTION_M = TARGET_RESOLUTION_M_APPROX

__all__ = [
    "PROJECT_ROOT",
    "PHASE4_DIR",
    "PHASE4_FEATURES_DIR",
    "PHASE4_MAPS_DIR",
    "PHASE4_TABLES_DIR",
    "PHASE3_ALIGNED_DIR",
    "PHASE3_MASKS_DIR",
    "PHASE3_ML_DIR",
    "L9_LST_2022",
    "L9_LST_2026",
    "S2_NDVI_2022",
    "S2_NDVI_2026",
    "S2_NDBI_2022",
    "S2_NDBI_2026",
    "L9_NDVI_2022",
    "L9_NDVI_2026",
    "L9_NDBI_2022",
    "L9_NDBI_2026",
    "VALID_MASK",
    "LANDUSE_RASTER",
    "ROADS_DISTANCE",
    "VEGETATION_DISTANCE",
    "BUILDINGS_DISTANCE",
    "PHASE3_FEATURE_TABLE",
    "AUTHORITATIVE_FEATURES",
    "NDVI_SOIL_REFERENCE",
    "NDVI_VEG_REFERENCE",
    "VEGETATION_COVER_DESCRIPTION",
    "VEGETATION_COVER_RASTER_2022",
    "VEGETATION_COVER_RASTER_2026",
    "COMBINED_DATASET_CSV",
    "FEATURE_METADATA_JSON",
    "PIPELINE_RECORD_JSON",
    "RANDOM_SEED",
    "CRS",
    "REFERENCE_RASTER_PATH",
    "NOMINAL_RESOLUTION_M",
]
