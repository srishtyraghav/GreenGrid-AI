"""
GreenGrid AI — Phase 2: Dataset Collection
Task 9 (partial): Validate Study Area Boundary

This script validates the Delhi NCT study_area.geojson to ensure it meets
all requirements for Phase 2.

Checks performed:
  1. File exists
  2. Valid JSON / GeoJSON structure
  3. CRS is EPSG:4326
  4. Geometry type is Polygon or MultiPolygon
  5. Geometry is non-empty
  6. Spatial extent matches known Delhi NCT bounding box
  7. Approximate area is realistic for Delhi NCT (≈ 1,484 km²)
  8. Feature count

Usage:
    python3 src/data_collection/validate_study_area.py
"""

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STUDY_AREA_FILE = PROJECT_ROOT / "data" / "raw" / "gis" / "study_area" / "study_area.geojson"

# Known approximate values for Delhi NCT
EXPECTED_AREA_KM2 = 1484          # Official Delhi NCT area
AREA_TOLERANCE    = 0.30           # ±30% tolerance on area estimate
DELHI_CENTROID    = (77.22, 28.64) # Approximate centroid (lon, lat)

# ─── Helpers ────────────────────────────────────────────────────────────────

def haversine_area_deg2_to_km2(area_deg2: float, lat_center: float) -> float:
    """
    Rough conversion of area in square degrees to km².
    1° lat ≈ 111.32 km, 1° lon ≈ 111.32 * cos(lat) km
    """
    km_per_lat = 111.32
    km_per_lon = 111.32 * math.cos(math.radians(lat_center))
    return area_deg2 * km_per_lat * km_per_lon


def shoelace(ring: list) -> float:
    """Compute signed area of a polygon ring using the shoelace formula (degrees²)."""
    n = len(ring)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ring[i][0] * ring[j][1]
        area -= ring[j][0] * ring[i][1]
    return abs(area) / 2.0


def polygon_area_deg2(poly_coords: list) -> float:
    """Area of a GeoJSON Polygon (outer ring minus holes)."""
    outer = shoelace(poly_coords[0])
    holes = sum(shoelace(ring) for ring in poly_coords[1:])
    return outer - holes


def geometry_area_deg2(geometry: dict) -> float:
    """Total area of a Polygon or MultiPolygon geometry in deg²."""
    gtype = geometry["type"]
    coords = geometry["coordinates"]
    if gtype == "Polygon":
        return polygon_area_deg2(coords)
    elif gtype == "MultiPolygon":
        return sum(polygon_area_deg2(poly) for poly in coords)
    return 0.0


def get_all_points(geometry: dict) -> list:
    """Flatten all coordinate points from a Polygon or MultiPolygon."""
    gtype = geometry["type"]
    coords = geometry["coordinates"]
    points = []
    if gtype == "Polygon":
        for ring in coords:
            points.extend(ring)
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                points.extend(ring)
    return points


# ─── Validation ─────────────────────────────────────────────────────────────

def run_validation():
    passed  = []
    failed  = []
    warning = []

    print("=" * 60)
    print("  GreenGrid AI — Phase 2 | Study Area Validation Report")
    print("=" * 60)
    print(f"\n  File: {STUDY_AREA_FILE}\n")

    # CHECK 1: File existence
    if STUDY_AREA_FILE.exists():
        passed.append("File exists")
        print(f"[PASS] File exists ({STUDY_AREA_FILE.stat().st_size:,} bytes)")
    else:
        failed.append("File does NOT exist")
        print(f"[FAIL] File does NOT exist at:\n       {STUDY_AREA_FILE}")
        print("\nRun: python3 src/data_collection/create_study_area.py first.")
        summarise(passed, failed, warning)
        sys.exit(1)

    # CHECK 2: Valid JSON
    try:
        with open(STUDY_AREA_FILE, "r") as f:
            geojson = json.load(f)
        passed.append("Valid JSON")
        print("[PASS] Valid JSON")
    except json.JSONDecodeError as e:
        failed.append(f"Invalid JSON: {e}")
        print(f"[FAIL] Invalid JSON: {e}")
        summarise(passed, failed, warning)
        sys.exit(1)

    # CHECK 3: GeoJSON FeatureCollection structure
    if geojson.get("type") == "FeatureCollection" and "features" in geojson:
        passed.append("Valid GeoJSON FeatureCollection")
        print("[PASS] Valid GeoJSON FeatureCollection")
    else:
        failed.append(f"Not a FeatureCollection (type={geojson.get('type')})")
        print(f"[FAIL] Expected FeatureCollection, got: {geojson.get('type')}")

    # CHECK 4: Feature count
    n_features = len(geojson.get("features", []))
    if n_features >= 1:
        passed.append(f"Feature count: {n_features}")
        print(f"[PASS] Feature count: {n_features}")
    else:
        failed.append("No features in file")
        print("[FAIL] No features found")
        summarise(passed, failed, warning)
        sys.exit(1)

    # CHECK 5: CRS metadata
    crs_ok = False
    # Check GeoJSON CRS field (legacy but present in our file)
    crs_field = geojson.get("crs", {})
    crs_name  = crs_field.get("properties", {}).get("name", "")
    if "CRS84" in crs_name or "4326" in crs_name:
        crs_ok = True
    # Also check feature properties
    feature_crs = geojson["features"][0].get("properties", {}).get("crs", "")
    if "4326" in feature_crs:
        crs_ok = True
    if crs_ok:
        passed.append("CRS = EPSG:4326")
        print("[PASS] CRS = EPSG:4326 (longitude/latitude, WGS84)")
    else:
        warning.append(f"CRS field not found; assuming EPSG:4326 (GeoJSON default)")
        print(f"[WARN] CRS field missing — GeoJSON default is EPSG:4326 (WGS84)")

    # CHECK 6: Geometry type
    geometry = geojson["features"][0]["geometry"]
    gtype    = geometry.get("type", "")
    if gtype in ("Polygon", "MultiPolygon"):
        passed.append(f"Geometry type: {gtype}")
        print(f"[PASS] Geometry type: {gtype}")
    else:
        failed.append(f"Unexpected geometry type: {gtype}")
        print(f"[FAIL] Unexpected geometry type: {gtype}")

    # CHECK 7: Non-empty coordinates
    if geometry.get("coordinates"):
        passed.append("Coordinates present")
        print("[PASS] Coordinates present")
    else:
        failed.append("No coordinates")
        print("[FAIL] Geometry has no coordinates")
        summarise(passed, failed, warning)
        sys.exit(1)

    # CHECK 8: Spatial extent
    all_pts  = get_all_points(geometry)
    lons     = [p[0] for p in all_pts]
    lats     = [p[1] for p in all_pts]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    centroid_lon = (min_lon + max_lon) / 2
    centroid_lat = (min_lat + max_lat) / 2

    print(f"\n  [SPATIAL EXTENT]")
    print(f"    Longitude : {min_lon:.6f}° → {max_lon:.6f}°  (span: {max_lon - min_lon:.4f}°)")
    print(f"    Latitude  : {min_lat:.6f}° → {max_lat:.6f}°  (span: {max_lat - min_lat:.4f}°)")
    print(f"    Centroid  : ({centroid_lon:.4f}°E, {centroid_lat:.4f}°N)")
    print(f"    Points    : {len(all_pts):,}")

    # Validate centroid is near Delhi
    lon_ok = 76.8 < centroid_lon < 77.4
    lat_ok = 28.4 < centroid_lat < 28.9
    if lon_ok and lat_ok:
        passed.append("Centroid within Delhi NCT region")
        print("[PASS] Centroid is within expected Delhi NCT region")
    else:
        failed.append(f"Centroid ({centroid_lon:.4f}, {centroid_lat:.4f}) outside Delhi")
        print(f"[FAIL] Centroid outside expected Delhi region")

    # CHECK 9: Area estimate
    area_deg2  = geometry_area_deg2(geometry)
    area_km2   = haversine_area_deg2_to_km2(area_deg2, centroid_lat)
    low_bound  = EXPECTED_AREA_KM2 * (1 - AREA_TOLERANCE)
    high_bound = EXPECTED_AREA_KM2 * (1 + AREA_TOLERANCE)

    print(f"\n  [AREA ESTIMATE]")
    print(f"    Computed area : {area_km2:.0f} km²")
    print(f"    Expected range: {low_bound:.0f} – {high_bound:.0f} km²")
    print(f"    Official area : {EXPECTED_AREA_KM2} km² (Delhi NCT)")

    if low_bound <= area_km2 <= high_bound:
        passed.append(f"Area estimate ({area_km2:.0f} km²) within expected range")
        print("[PASS] Area within acceptable range of Delhi NCT")
    else:
        warning.append(f"Area estimate {area_km2:.0f} km² is outside expected range")
        print(f"[WARN] Area estimate outside expected range (shoelace approximation may differ)")

    summarise(passed, failed, warning)
    return len(failed) == 0


def summarise(passed, failed, warning):
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  PASSED  : {len(passed)}")
    print(f"  WARNINGS: {len(warning)}")
    print(f"  FAILED  : {len(failed)}")
    if warning:
        print("\n  Warnings:")
        for w in warning:
            print(f"    ⚠  {w}")
    if failed:
        print("\n  Failures:")
        for f_ in failed:
            print(f"    ✗  {f_}")
        print("\n  [STATUS] VALIDATION FAILED — fix issues before proceeding")
    else:
        print("\n  [STATUS] VALIDATION PASSED ✓")
        print("  The study_area.geojson is ready for Phase 2 data filtering.")


if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
