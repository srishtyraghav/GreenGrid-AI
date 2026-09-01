"""
GreenGrid AI — Phase 3: Data Preprocessing
Configuration and constants.

All paths are repository-relative so the project can be cloned and run
on another machine provided the required local data is available.
"""

from pathlib import Path

# ─── Project Root ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Input Directories ──────────────────────────────────────────────────────
RAW_DIR = PROJECT_ROOT / "data" / "raw"
GIS_DIR = RAW_DIR / "gis"
LANDSAT9_DIR = RAW_DIR / "landsat9"
SENTINEL2_DIR = RAW_DIR / "sentinel2"

# ─── Output Directories ─────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "phase3"
INDICES_DIR = PROCESSED_DIR / "indices"
ALIGNED_DIR = PROCESSED_DIR / "aligned"
MASKS_DIR = PROCESSED_DIR / "masks"
FEATURES_DIR = PROCESSED_DIR / "ml"

# ─── Study Area ─────────────────────────────────────────────────────────────
STUDY_AREA_PATH = GIS_DIR / "study_area" / "study_area.geojson"

# ─── Coordinate Reference System ────────────────────────────────────────────
# All Phase 2 data were exported from GEE in EPSG:4326 (WGS84).
TARGET_CRS = "EPSG:4326"

# ─── Common Analysis Grid ───────────────────────────────────────────────────
# Landsat 9 LST is only available at 30 m. We therefore use the Landsat 30 m
# angular resolution as the common grid and aggregate Sentinel-2 products to it.
# IMPORTANT: EPSG:4326 uses angular units, so the metric pixel size depends on
# latitude. The value below corresponds to ~30 m at the equator; at Delhi's
# latitude (~28.65°N) the actual metric size is approximately:
#   NS ≈ 29.9 m, EW ≈ 26.4 m.
TARGET_RESOLUTION_DEG = 0.00026949458523585647  # GEE-exported Landsat 9 scale
TARGET_RESOLUTION_M_APPROX = 30

# Reference raster used to define the common 30 m grid.
# The 2026 Landsat 9 composite covers the full study area and has the most
# recent, least-cloudy data, making it a stable reference.
REFERENCE_RASTER = LANDSAT9_DIR / "2026_07" / "landsat9_2026_07_composite_30m.tif"

# ─── Landsat 9 Band Mapping ─────────────────────────────────────────────────
# Matches dataset_metadata.csv and the band descriptions in the GeoTIFFs.
L9_BANDS = {
    "SR_B2": 1,   # Blue
    "SR_B3": 2,   # Green
    "SR_B4": 3,   # Red
    "SR_B5": 4,   # NIR
    "SR_B6": 5,   # SWIR1
    "SR_B7": 6,   # SWIR2
    "ST_B10": 7,  # Thermal (raw DN in the exported GeoTIFFs)
    "QA_PIXEL": 8,
}

# ─── Sentinel-2 Band Mapping ────────────────────────────────────────────────
# 10 m composite: B2, B3, B4, B8
# 20 m composite: B5, B6, B7, B8A, B11, B12, SCL
S2_10M_BANDS = {"B2": 1, "B3": 2, "B4": 3, "B8": 4}
S2_20M_BANDS = {"B5": 1, "B6": 2, "B7": 3, "B8A": 4, "B11": 5, "B12": 6, "SCL": 7}

# ─── Index Formulas ─────────────────────────────────────────────────────────
# Landsat 9 indices use already-scaled surface reflectance from GEE.
# Sentinel-2 indices use already-scaled surface reflectance (0–1).
#
# NDBI for Landsat 9 follows the Phase 2 report: SR_B6 (SWIR1) and SR_B5 (NIR).
# NDBI for Sentinel-2 is computed from the 20 m bands available in the
# exported composite: B11 (SWIR1) and B8A (narrow NIR). B8 (10 m NIR) is not
# present in the 20 m composite, so native-resolution S2 NDBI must use B8A.
# The 20 m NDBI is then aggregated to the common 30 m grid.
L9_NDVI_BANDS = ("SR_B5", "SR_B4")   # (NIR, Red)
L9_NDBI_BANDS = ("SR_B6", "SR_B5")   # (SWIR1, NIR)
S2_NDVI_BANDS = ("B8", "B4")         # (NIR, Red)
S2_NDBI_BANDS = ("B11", "B8A")       # (SWIR1, narrow NIR)

# ─── Landsat 9 Collection 2 Level-2 Thermal Scaling ─────────────────────────
# ST_B10 is exported as raw DN. USGS scale factors:
#   K  = DN * 0.00341802 + 149.0
#   °C = K - 273.15
LST_SCALE = 0.00341802
LST_OFFSET = 149.0
KELVIN_TO_CELSIUS = -273.15

# ─── Sentinel-2 Reflectance Scale ───────────────────────────────────────────
# Already applied in GEE (divide by 10000). Kept here for documentation.
S2_REFLECTANCE_SCALE = 1.0  # GEE export already scaled the values

# ─── Missing-Data Handling ──────────────────────────────────────────────────
# The source GeoTIFFs have nodata=None but use NaN to represent masked pixels
# (outside study area / clouds / cloud shadows / SCL-excluded classes).
# We treat NaN and Inf as invalid and do NOT treat legitimate zero or negative
# reflectance values as NoData.
INVALID_FLAGS = {"nan", "inf", "-inf"}

# ─── Spatial Blocking for Future Validation ─────────────────────────────────
# Number of tiles used to assign a spatial_block_id to each sample. This lets
# later phases perform spatial cross-validation instead of naive random splits.
N_SPATIAL_BLOCKS = 5

# ─── Sampling ───────────────────────────────────────────────────────────────
# Maximum number of valid pixels to extract for the ML-ready feature table.
# Using a sample keeps the CSV tractable while preserving spatial coverage.
MAX_FEATURE_SAMPLES = 150_000
RANDOM_SEED = 42
