"""Central configuration for Phase 6 severity mapping.

Phase 6 builds on the frozen Phase 5 outputs and produces complete
full-valid-grid prediction rasters (Phase 5 rasters cover only the ~4.5%
of the grid represented by the sampled training pixels).  The production
model is the Random Forest RF-C configuration selected by the Phase 5
generalization audit, retrained here on all 300,000 sampled rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Resolve project root robustly whether this file is imported directly or
# via a package import.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
src_dir = PROJECT_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from features.config import (  # noqa: E402
    L9_LST_2022,
    L9_LST_2026,
    LANDUSE_RASTER,
    S2_NDBI_2022,
    S2_NDBI_2026,
    S2_NDVI_2022,
    S2_NDVI_2026,
    VALID_MASK,
    VEGETATION_COVER_RASTER_2022,
    VEGETATION_COVER_RASTER_2026,
)
from models.config import (  # noqa: E402
    CLASS_LABELS,
    CLASS_VALUES,
    INPUT_DATASET_CSV,
    RANDOM_FOREST_PARAMS,
    RANDOM_SEED,
    REFERENCE_RASTER,
)

# ---------------------------------------------------------------------------
# Phase 6 directories
# ---------------------------------------------------------------------------
PHASE6_DIR = PROJECT_ROOT / "data" / "processed" / "phase6"
PHASE6_MODELS_DIR = PHASE6_DIR / "models"
PHASE6_RASTERS_DIR = PHASE6_DIR / "rasters"
PHASE6_TABLES_DIR = PHASE6_DIR / "tables"
PHASE6_FIGURES_DIR = PHASE6_DIR / "figures"
PHASE6_REPORTS_DIR = PHASE6_DIR / "reports"
PHASE6_HOTSPOTS_DIR = PHASE6_DIR / "hotspots"

for _d in (
    PHASE6_MODELS_DIR,
    PHASE6_RASTERS_DIR,
    PHASE6_TABLES_DIR,
    PHASE6_FIGURES_DIR,
    PHASE6_REPORTS_DIR,
    PHASE6_HOTSPOTS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Inputs reused from frozen phases
# ---------------------------------------------------------------------------
VALID_MASK_RASTER = VALID_MASK
LST_RASTERS = {2022: L9_LST_2022, 2026: L9_LST_2026}
PHASE5_PREDICTIONS_CSV = (
    PROJECT_ROOT / "data" / "processed" / "phase5" / "tables" / "predictions.csv"
)
# TRUE out-of-fold predictions from the frozen Phase 5 spatial-CV: every
# sampled cell scored by a model trained on folds excluding its block.
# NOTE: predictions.csv above holds the production model's IN-SAMPLE
# predictions and must NOT be used for error diagnostics.
PHASE5_OOF_PREDICTIONS_CSV = (
    PROJECT_ROOT / "data" / "processed" / "phase5" / "tables" / "oof_predictions.csv"
)
PHASE5_SPATIAL_FEATURES_CSV = (
    PROJECT_ROOT / "data" / "processed" / "phase5" / "tables" / "spatial_neighbourhood_features.csv"
)
PHASE5_MORPHOLOGY_FEATURES_CSV = (
    PROJECT_ROOT / "data" / "processed" / "phase5" / "tables" / "morphology_features.csv"
)
PHASE5_UHI_RASTER_2022 = (
    PROJECT_ROOT / "data" / "processed" / "phase5" / "predictions" / "uhi_2022.tif"
)

# ---------------------------------------------------------------------------
# Class definitions (identical to Phase 5; CLASS_LABELS / CLASS_VALUES are
# imported above from models.config)
# ---------------------------------------------------------------------------
CLASS_MAPPING = dict(zip(CLASS_LABELS, CLASS_VALUES))

# ---------------------------------------------------------------------------
# Production model: Random Forest RF-C (adopted per the Phase 5 report)
# ---------------------------------------------------------------------------
# The Phase 5 generalization audit recommended the regularized RF-C
# configuration (max_depth=15, min_samples_leaf=10) for production use,
# with all other parameters identical to the Phase 5 Random Forest.
RF_C_PARAMS = {
    **RANDOM_FOREST_PARAMS,
    "max_depth": 15,
    "min_samples_leaf": 10,
}

# ---------------------------------------------------------------------------
# Raster output nodata conventions
# ---------------------------------------------------------------------------
SEVERITY_NODATA = -1  # int16 severity class rasters
PROBABILITY_NODATA = -1.0  # float32 probability / confidence / score rasters
LST_NODATA = -1.0  # float32 observed LST rasters

# ---------------------------------------------------------------------------
# Prediction chunk size (rows per predict_proba call) for memory discipline
# ---------------------------------------------------------------------------
PREDICT_CHUNK_ROWS = 200_000

# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
MODEL_PATH = PHASE6_MODELS_DIR / "severity_rf_c.joblib"
MANIFEST_JSON = PHASE6_DIR / "phase6_manifest.json"
PIPELINE_RECORD_JSON = PHASE6_DIR / "phase6_pipeline_record.json"
FULLGRID_SUMMARY_CSV = PHASE6_TABLES_DIR / "fullgrid_prediction_summary.csv"

# ---------------------------------------------------------------------------
# Stage 2: hotspot delineation
# ---------------------------------------------------------------------------
# NOTE: these are ML-based RELATIVE heat-severity hotspots — an operational
# UHI hotspot proxy, NOT physical UHI intensity.  NoData cells never
# participate in hotspot membership; "not a hotspot" is asserted only for
# VALID cells that fail a definition.
HOTSPOT_DEFINITIONS = {
    # A (primary): High or Severe severity class.
    "A": {"severity_values": (2, 3), "min_confidence": None},
    # B: Severe only (strictest).
    "B": {"severity_values": (3,), "min_confidence": None},
    # C: A + operational confidence criterion.  NOTE: probabilities are
    # uncalibrated confidence proxies (per the Phase 5 calibration report),
    # so the 0.60 threshold is an operational filter, not a calibrated
    # probability statement.
    "C": {"severity_values": (2, 3), "min_confidence": 0.60},
}
MIN_HOTSPOT_PIXELS = 10  # 8-connected; ~0.8 ha at 30 m — operational noise removal
CONNECTIVITY_STRUCTURE = "8"  # scipy.ndimage 8-connectivity (structure=np.ones((3, 3)))
HOTSPOT_NODATA = -1  # int32 hotspot-ID rasters (0 = valid but not hotspot)

# Equal-area projection for area computation: UTM zone 43N (Delhi).
# Degree widths are NOT equal-area — areas must never be computed from
# degree coordinates.
UTM_CRS = "EPSG:32643"
WGS84_CRS = "EPSG:4326"

# Operational green/built classification.  Conventional remote-sensing
# thresholds (NDVI >= 0.3 vegetated; NDBI >= 0.1 built), applied
# operationally — they are NOT site-calibrated.  ``landuse_class`` is
# reported separately as its own attribute.
GREEN_NDVI_THRESHOLD = 0.3
BUILT_NDBI_THRESHOLD = 0.1
BUILT_MAX_NDVI = 0.3  # built cells must also satisfy NDVI < 0.3

LANDUSE_CLASS_NAMES = {
    0: "unclassified_background",
    1: "park",
    2: "forest",
    3: "grass",
    4: "commercial",
    5: "industrial",
    6: "residential",
    7: "retail",
    8: "farmland",
}
LANDUSE_NODATA = 255

SEVERITY_CLASS_NAMES = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}

NDVI_RASTERS = {2022: S2_NDVI_2022, 2026: S2_NDVI_2026}
NDBI_RASTERS = {2022: S2_NDBI_2022, 2026: S2_NDBI_2026}
VEGETATION_COVER_RASTERS = {
    2022: VEGETATION_COVER_RASTER_2022,
    2026: VEGETATION_COVER_RASTER_2026,
}
LANDUSE_RASTER_PATH = LANDUSE_RASTER

HOTSPOT_STATISTICS_CSV = PHASE6_TABLES_DIR / "hotspot_statistics_def{definition}_{year}.csv"
HOTSPOT_SENSITIVITY_CSV = PHASE6_TABLES_DIR / "hotspot_sensitivity.csv"
POLYGON_AREA_TOLERANCE = 0.05  # |polygon area / (pixel_count * mean pixel area) - 1|


def classify_green_built(ndvi: np.ndarray, ndbi: np.ndarray) -> np.ndarray:
    """Classify valid cells as green-dominant / built-dominant / other-mixed.

    Parameters
    ----------
    ndvi, ndbi : np.ndarray
        Arrays of equal shape with NDVI / NDBI values at valid cells.

    Returns
    -------
    np.ndarray of str
        One of "green_dominant", "built_dominant", "other_mixed" per cell.
        Thresholds are conventional remote-sensing operational values, NOT
        site-calibrated (see GREEN_NDVI_THRESHOLD / BUILT_NDBI_THRESHOLD).
    """
    green = ndvi >= GREEN_NDVI_THRESHOLD
    built = (ndbi >= BUILT_NDBI_THRESHOLD) & (ndvi < BUILT_MAX_NDVI)
    out = np.full(ndvi.shape, "other_mixed", dtype=object)
    out[built] = "built_dominant"
    out[green] = "green_dominant"
    return out


def hotspot_mask_path(definition: str, year: int) -> Path:
    """Return the hotspot-ID raster path for a definition and year."""
    return PHASE6_HOTSPOTS_DIR / f"hotspot_mask_def{definition}_{year}.tif"


def hotspot_boundaries_path(definition: str, year: int) -> Path:
    """Return the hotspot-boundary GeoJSON path for a definition and year."""
    return PHASE6_HOTSPOTS_DIR / f"hotspot_boundaries_def{definition}_{year}.geojson"


def hotspot_statistics_path(definition: str, year: int) -> Path:
    """Return the per-hotspot statistics CSV path for a definition and year."""
    return PHASE6_TABLES_DIR / f"hotspot_statistics_def{definition}_{year}.csv"


def severity_raster_path(year: int) -> Path:
    """Return the full-grid severity class raster path for a year."""
    return PHASE6_RASTERS_DIR / f"severity_{year}.tif"


def probability_raster_path(label: str, year: int) -> Path:
    """Return the full-grid per-class probability raster path."""
    return PHASE6_RASTERS_DIR / f"probability_{label.lower()}_{year}.tif"


def confidence_raster_path(year: int) -> Path:
    """Return the full-grid confidence raster path for a year."""
    return PHASE6_RASTERS_DIR / f"confidence_{year}.tif"


def severity_score_raster_path(year: int) -> Path:
    """Return the full-grid ordinal severity-score raster path for a year."""
    return PHASE6_RASTERS_DIR / f"severity_score_{year}.tif"


def lst_raster_path(year: int) -> Path:
    """Return the full-grid observed LST raster path for a year."""
    return PHASE6_RASTERS_DIR / f"lst_{year}.tif"


__all__ = [
    "PROJECT_ROOT",
    "PHASE6_DIR",
    "PHASE6_MODELS_DIR",
    "PHASE6_RASTERS_DIR",
    "PHASE6_TABLES_DIR",
    "PHASE6_FIGURES_DIR",
    "PHASE6_REPORTS_DIR",
    "PHASE6_HOTSPOTS_DIR",
    "VALID_MASK_RASTER",
    "LST_RASTERS",
    "INPUT_DATASET_CSV",
    "REFERENCE_RASTER",
    "PHASE5_PREDICTIONS_CSV",
    "PHASE5_OOF_PREDICTIONS_CSV",
    "PHASE5_SPATIAL_FEATURES_CSV",
    "PHASE5_MORPHOLOGY_FEATURES_CSV",
    "PHASE5_UHI_RASTER_2022",
    "CLASS_LABELS",
    "CLASS_VALUES",
    "CLASS_MAPPING",
    "RANDOM_SEED",
    "RF_C_PARAMS",
    "SEVERITY_NODATA",
    "PROBABILITY_NODATA",
    "LST_NODATA",
    "PREDICT_CHUNK_ROWS",
    "MODEL_PATH",
    "MANIFEST_JSON",
    "PIPELINE_RECORD_JSON",
    "FULLGRID_SUMMARY_CSV",
    "HOTSPOT_DEFINITIONS",
    "MIN_HOTSPOT_PIXELS",
    "CONNECTIVITY_STRUCTURE",
    "HOTSPOT_NODATA",
    "UTM_CRS",
    "WGS84_CRS",
    "GREEN_NDVI_THRESHOLD",
    "BUILT_NDBI_THRESHOLD",
    "BUILT_MAX_NDVI",
    "LANDUSE_CLASS_NAMES",
    "LANDUSE_NODATA",
    "SEVERITY_CLASS_NAMES",
    "NDVI_RASTERS",
    "NDBI_RASTERS",
    "VEGETATION_COVER_RASTERS",
    "LANDUSE_RASTER_PATH",
    "HOTSPOT_STATISTICS_CSV",
    "HOTSPOT_SENSITIVITY_CSV",
    "POLYGON_AREA_TOLERANCE",
    "classify_green_built",
    "hotspot_mask_path",
    "hotspot_boundaries_path",
    "hotspot_statistics_path",
    "severity_raster_path",
    "probability_raster_path",
    "confidence_raster_path",
    "severity_score_raster_path",
    "lst_raster_path",
]
