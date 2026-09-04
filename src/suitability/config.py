"""Central configuration for Phase 7 tree-plantation suitability scoring.

Every parameter here is FROZEN by ``reports/phase7_design_spec.md``
(Stage 0, 2026-09-04).  Stage 1 implements these decisions exactly; any
deviation requires an explicit written justification in the Phase 7 report.
All weights are predefined by the design spec — they are not tuned, fitted,
or selected against any evaluation target.
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
    LANDUSE_RASTER,
    ROADS_DISTANCE,
    S2_NDBI_2022,
    S2_NDBI_2026,
    S2_NDVI_2022,
    S2_NDVI_2026,
    VALID_MASK,
    VEGETATION_COVER_RASTER_2022,
    VEGETATION_COVER_RASTER_2026,
)

# ---------------------------------------------------------------------------
# Phase 7 directories
# ---------------------------------------------------------------------------
PHASE7_DIR = PROJECT_ROOT / "data" / "processed" / "phase7"
PHASE7_RASTERS_DIR = PHASE7_DIR / "rasters"
PHASE7_VECTORS_DIR = PHASE7_DIR / "vectors"
PHASE7_TABLES_DIR = PHASE7_DIR / "tables"
PHASE7_FIGURES_DIR = PHASE7_DIR / "figures"
PHASE7_REPORTS_DIR = PHASE7_DIR / "reports"

for _d in (
    PHASE7_RASTERS_DIR,
    PHASE7_VECTORS_DIR,
    PHASE7_TABLES_DIR,
    PHASE7_FIGURES_DIR,
    PHASE7_REPORTS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Grid / domain facts (frozen by the input audit)
# ---------------------------------------------------------------------------
YEARS = (2022, 2026)
EXPECTED_VALID_PIXELS = 1_500_777  # valid_mask == 1, identical for both years
RANDOM_SEED = 42
UTM_CRS = "EPSG:32643"  # equal-area area CRS (UTM 43N, Delhi)
WGS84_CRS = "EPSG:4326"

# Priority-zone noise rule (used by Stage 2; frozen here for the record).
MIN_ZONE_PIXELS = 10  # 8-connected; ~0.8 ha at 30 m

# ---------------------------------------------------------------------------
# Inputs reused from frozen phases (Phases 3-6 are read-only)
# ---------------------------------------------------------------------------
PHASE6_RASTERS_DIR = PROJECT_ROOT / "data" / "processed" / "phase6" / "rasters"
VALID_MASK_RASTER = VALID_MASK

SEVERITY_SCORE_RASTERS = {
    year: PHASE6_RASTERS_DIR / f"severity_score_{year}.tif" for year in YEARS
}
LST_RASTERS = {year: PHASE6_RASTERS_DIR / f"lst_{year}.tif" for year in YEARS}
NDVI_RASTERS = {2022: S2_NDVI_2022, 2026: S2_NDVI_2026}
NDBI_RASTERS = {2022: S2_NDBI_2022, 2026: S2_NDBI_2026}
VEGETATION_COVER_RASTERS = {
    2022: VEGETATION_COVER_RASTER_2022,
    2026: VEGETATION_COVER_RASTER_2026,
}
LANDUSE_RASTER_PATH = LANDUSE_RASTER
ROADS_DISTANCE_RASTER = ROADS_DISTANCE
VEGETATION_DISTANCE_RASTER = VALID_MASK.parent / "vegetation_distance_30m.tif"

# ---------------------------------------------------------------------------
# Normalization (FROZEN, design spec section f)
# ---------------------------------------------------------------------------
# Per-year robust min-max: clip at the 1st-99th percentile over valid-mask
# cells for that year, then scale to 0-100.  All scores/class boundaries are
# therefore class-relative across years (snapshots, not trends).
NORM_PERCENTILES = (1.0, 99.0)

# ---------------------------------------------------------------------------
# Heat Need weights (FROZEN, design spec section b)
# ---------------------------------------------------------------------------
# Need = 0.40*n(severity_score) + 0.25*n(1-NDVI) + 0.20*n(NDBI) + 0.15*n(LST)
# vegetation_cover is EXCLUDED from the baseline (Phase 5 froze NDVI-
# vegetation_cover redundancy at r ~ 0.998-0.9995); it appears only in
# Scenario D.  The LST weight is deliberately small (0.15) because the
# severity score is itself trained on per-year LST quartiles (limits
# double-counting; documented, not tuned).
NEED_WEIGHTS = {
    "severity_score": 0.40,
    "one_minus_ndvi": 0.25,
    "ndbi": 0.20,
    "lst": 0.15,
}

# ---------------------------------------------------------------------------
# Plantation Opportunity weights (FROZEN, design spec section c)
# ---------------------------------------------------------------------------
# Opportunity = 0.30*landuse_eligibility
#             + 0.25*(100 - n(NDBI))          built-up intensity, inverted
#             + 0.15*road_accessibility_band  non-monotonic, see below
#             + 0.15*green_proximity          500 m cap, see below
#             + 0.15*(100 - n(NDVI))          planting headroom, inverted
OPPORTUNITY_WEIGHTS = {
    "landuse_eligibility": 0.30,
    "built_up_inverse": 0.25,
    "road_accessibility": 0.15,
    "green_proximity": 0.15,
    "planting_headroom": 0.15,
}

# Road accessibility band (FROZEN piecewise-linear curve, design spec
# section c, Option A — documented, not site-calibrated):
#   d < 50 m        road corridor: ramps linearly 0 -> 40 as d goes 0 -> 50 m
#   50-500 m        optimal: 100
#   500-2,000 m     linear decline 100 -> 30
#   > 2,000 m       floor at 30
ROAD_BAND = {
    "corridor_max_m": 50.0,
    "corridor_score_max": 40.0,  # score at d = corridor_max_m (from below)
    "optimal_max_m": 500.0,
    "optimal_score": 100.0,
    "decline_max_m": 2000.0,
    "decline_score": 30.0,  # score reached at d = decline_max_m and beyond
}

# Green proximity (FROZEN, design spec section c):
#   green_proximity = 100 * max(0, 1 - dist_vegetation_m / 500)
GREEN_PROXIMITY_CAP_M = 500.0

# ---------------------------------------------------------------------------
# Land-use eligibility rules matrix (FROZEN, design spec section d)
# ---------------------------------------------------------------------------
# Scores are 0-100 eligibility contributions.  Class 0 is "absence of an
# OSM land-use tag", NOT a real land-use observation, and receives a NEUTRAL
# 50 (UNCERTAIN — flagged for the report, not guessed silently).  nodata
# (255) is mapped to the same neutral 50.
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
LANDUSE_ELIGIBILITY = {
    0: 50,
    1: 40,
    2: 30,
    3: 50,
    4: 35,
    5: 20,
    6: 90,
    7: 30,
    8: 60,
}
LANDUSE_RULE_STATUS = {
    0: "CONDITIONAL (UNCERTAIN)",
    1: "CONDITIONAL (UNCERTAIN)",
    2: "CONDITIONAL",
    3: "CONDITIONAL (UNCERTAIN)",
    4: "CONDITIONAL",
    5: "DISCOURAGED",
    6: "ELIGIBLE",
    7: "DISCOURAGED",
    8: "CONDITIONAL",
}
LANDUSE_RULE_RATIONALE = {
    0: "84.0% of the full grid; absence of an OSM land-use tag, not a real "
       "land-use observation. Neutral score; large background suitability "
       "must be explicitly discussed in the report.",
    1: "Already vegetated public land; planting headroom low "
       "(maintenance/infilling only). Only 4 px on disk - rule never "
       "meaningfully fires.",
    2: "Already forested; headroom low (understorey/enrichment only).",
    3: "Open green land; possible but competes with existing open-space "
       "function. Absent from raster.",
    4: "Paved, high-traffic; limited verge/pocket planting.",
    5: "Contamination/operational constraints; low priority.",
    6: "Yards, street trees, neighbourhood greening - primary planting "
       "target. Largest mapped class (255,946 px).",
    7: "Dense impervious retail cores.",
    8: "Planting possible (agroforestry, bunds) but competes with crop land.",
}
LANDUSE_NODATA = 255
LANDUSE_NODATA_ELIGIBILITY = 50  # neutral, same as class 0

# ---------------------------------------------------------------------------
# Suitability classes (FROZEN, design spec section g)
# ---------------------------------------------------------------------------
# Documented encoding: classes 0-4 (int16, nodata -1):
#   score in [0, 20]     -> 0 = Very Low
#   score in (20, 40]    -> 1 = Low
#   score in (40, 60]    -> 2 = Medium
#   score in (60, 80]    -> 3 = High
#   score in (80, 100]   -> 4 = Very High
# Boundary cells exactly on 20/40/60/80 belong to the LOWER class.
CLASS_EDGES = (20.0, 40.0, 60.0, 80.0)
CLASS_LABELS = {0: "Very Low", 1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
HIGH_VERY_HIGH_CLASSES = (3, 4)

# ---------------------------------------------------------------------------
# Sensitivity scenarios (FROZEN, design spec section h)
# ---------------------------------------------------------------------------
# Weighted geometric mean:
#   Final_s = 100^(1 - a_need - a_opp) * Need^a_need * Opportunity^a_opp
# which stays in [0, 100] and collapses exactly to the gated product
# Need * Opportunity / 100 when a_need = a_opp = 1.
SCENARIO_EXPONENTS = {
    "A": (1.0, 1.0),  # Balanced (baseline): pure gated product
    "B": (0.6, 0.3),  # Heat-focused:  100^0.1 * Need^0.6 * Opp^0.3
    "C": (0.3, 0.5),  # Feasibility-focused: 100^0.2 * Need^0.3 * Opp^0.5
    "D": (1.0, 1.0),  # Vegetation-priority (composition varies, see below)
}
# Scenario D internal composition (FROZEN): NDVI is dropped from Need and
# replaced by 1-vegetation_cover (0.25); Opportunity downweights built-up
# (0.25 -> 0.15) and upweights planting headroom computed on inverted
# normalized vegetation_cover (0.15 -> 0.25).
SCENARIO_D_NEED_WEIGHTS = {
    "severity_score": 0.40,
    "one_minus_vegetation_cover": 0.25,
    "ndbi": 0.20,
    "lst": 0.15,
}
SCENARIO_D_OPPORTUNITY_WEIGHTS = {
    "landuse_eligibility": 0.30,
    "built_up_inverse": 0.15,
    "road_accessibility": 0.15,
    "green_proximity": 0.15,
    "planting_headroom": 0.25,
}

# ---------------------------------------------------------------------------
# Nodata conventions
# ---------------------------------------------------------------------------
SCORE_NODATA = -1.0  # float32 score rasters (need/opportunity/suitability)
CLASS_NODATA = -1  # int16 class rasters

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
TABLE_PATHS = {
    "summary": PHASE7_TABLES_DIR / "suitability_summary.csv",
    "area_statistics": PHASE7_TABLES_DIR / "suitability_area_statistics.csv",
    "feature_dependency": PHASE7_TABLES_DIR / "feature_dependency_analysis.csv",
    "normalization_parameters": PHASE7_TABLES_DIR / "normalization_parameters.csv",
    "landuse_rules": PHASE7_TABLES_DIR / "landuse_suitability_rules.csv",
    "sensitivity": PHASE7_TABLES_DIR / "sensitivity_analysis.csv",
}
MANIFEST_JSON = PHASE7_DIR / "phase7_manifest.json"
PIPELINE_RECORD_JSON = PHASE7_DIR / "phase7_pipeline_record.json"
EXCLUSION_MASK_RASTER = PHASE7_RASTERS_DIR / "exclusion_mask.tif"

TERMINOLOGY_NOTE = (
    "Outputs describe POTENTIAL PLANTATION SUITABILITY — a relative, "
    "decision-support ranking. They are NOT a statement of legal "
    "availability, land ownership, or a physical plantability guarantee, "
    "and make no claim of planting survival probability."
)


def heat_need_raster_path(year: int) -> Path:
    """Return the Heat Need raster path for a year."""
    return PHASE7_RASTERS_DIR / f"heat_need_{year}.tif"


def opportunity_raster_path(year: int) -> Path:
    """Return the Plantation Opportunity raster path for a year."""
    return PHASE7_RASTERS_DIR / f"plantation_opportunity_{year}.tif"


def suitability_raster_path(year: int) -> Path:
    """Return the baseline (Scenario A) suitability raster path."""
    return PHASE7_RASTERS_DIR / f"tree_plantation_suitability_{year}.tif"


def suitability_class_raster_path(year: int) -> Path:
    """Return the baseline suitability class raster path."""
    return PHASE7_RASTERS_DIR / f"tree_plantation_suitability_class_{year}.tif"


def scenario_raster_path(scenario: str, year: int) -> Path:
    """Return the suitability raster path for a scenario (B/C/D) and year."""
    return PHASE7_RASTERS_DIR / f"suitability_scenario_{scenario}_{year}.tif"


def road_accessibility_score(dist_m: np.ndarray) -> np.ndarray:
    """Piecewise-linear road-accessibility band score (0-100).

    Parameters
    ----------
    dist_m : np.ndarray
        Distance to nearest road (m), finite values.

    Returns
    -------
    np.ndarray
        Accessibility score: linear ramp 0 -> 40 over [0, 50) m, 100 over
        [50, 500] m, linear decline 100 -> 30 over (500, 2000) m, floor 30
        beyond 2000 m.  FROZEN curve (design spec section c); documented,
        not site-calibrated.
    """
    spec = ROAD_BAND
    score = np.full(dist_m.shape, spec["decline_score"], dtype=np.float64)

    decline = (dist_m > spec["optimal_max_m"]) & (dist_m < spec["decline_max_m"])
    score[decline] = spec["optimal_score"] - (
        (spec["optimal_score"] - spec["decline_score"])
        * (dist_m[decline] - spec["optimal_max_m"])
        / (spec["decline_max_m"] - spec["optimal_max_m"])
    )

    optimal = (dist_m >= spec["corridor_max_m"]) & (dist_m <= spec["optimal_max_m"])
    score[optimal] = spec["optimal_score"]

    corridor = dist_m < spec["corridor_max_m"]
    score[corridor] = (
        spec["corridor_score_max"] * dist_m[corridor] / spec["corridor_max_m"]
    )
    return score


def green_proximity_score(dist_m: np.ndarray) -> np.ndarray:
    """Green-proximity score: 100*max(0, 1 - dist/500).  FROZEN (spec c)."""
    return 100.0 * np.clip(1.0 - dist_m / GREEN_PROXIMITY_CAP_M, 0.0, 1.0)


def landuse_eligibility(landuse: np.ndarray) -> np.ndarray:
    """Map landuse class codes to 0-100 eligibility scores.

    nodata (255) maps to the same neutral 50 as class 0 (absence of an OSM
    tag is not evidence of ineligibility; FROZEN decision, spec section d).
    """
    lookup = np.full(256, LANDUSE_NODATA_ELIGIBILITY, dtype=np.float64)
    for code, score in LANDUSE_ELIGIBILITY.items():
        lookup[code] = score
    return lookup[landuse.astype(np.int64)]


__all__ = [
    "PROJECT_ROOT",
    "PHASE7_DIR",
    "PHASE7_RASTERS_DIR",
    "PHASE7_VECTORS_DIR",
    "PHASE7_TABLES_DIR",
    "PHASE7_FIGURES_DIR",
    "PHASE7_REPORTS_DIR",
    "YEARS",
    "EXPECTED_VALID_PIXELS",
    "RANDOM_SEED",
    "UTM_CRS",
    "WGS84_CRS",
    "MIN_ZONE_PIXELS",
    "VALID_MASK_RASTER",
    "SEVERITY_SCORE_RASTERS",
    "LST_RASTERS",
    "NDVI_RASTERS",
    "NDBI_RASTERS",
    "VEGETATION_COVER_RASTERS",
    "LANDUSE_RASTER_PATH",
    "ROADS_DISTANCE_RASTER",
    "VEGETATION_DISTANCE_RASTER",
    "NORM_PERCENTILES",
    "NEED_WEIGHTS",
    "OPPORTUNITY_WEIGHTS",
    "ROAD_BAND",
    "GREEN_PROXIMITY_CAP_M",
    "LANDUSE_CLASS_NAMES",
    "LANDUSE_ELIGIBILITY",
    "LANDUSE_RULE_STATUS",
    "LANDUSE_RULE_RATIONALE",
    "LANDUSE_NODATA",
    "LANDUSE_NODATA_ELIGIBILITY",
    "CLASS_EDGES",
    "CLASS_LABELS",
    "HIGH_VERY_HIGH_CLASSES",
    "SCENARIO_EXPONENTS",
    "SCENARIO_D_NEED_WEIGHTS",
    "SCENARIO_D_OPPORTUNITY_WEIGHTS",
    "SCORE_NODATA",
    "CLASS_NODATA",
    "TABLE_PATHS",
    "MANIFEST_JSON",
    "PIPELINE_RECORD_JSON",
    "EXCLUSION_MASK_RASTER",
    "TERMINOLOGY_NOTE",
    "heat_need_raster_path",
    "opportunity_raster_path",
    "suitability_raster_path",
    "suitability_class_raster_path",
    "scenario_raster_path",
    "road_accessibility_score",
    "green_proximity_score",
    "landuse_eligibility",
]
