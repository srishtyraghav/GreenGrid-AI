"""
GreenGrid AI — Phase 2: Dataset Collection
Task 1: Create Delhi NCT Study Area Boundary

This script fetches the authoritative Delhi NCT administrative boundary
from OpenStreetMap (relation ID: 1942586) via the polygons.openstreetmap.fr
service and saves it as a clean, validated GeoJSON in EPSG:4326.

Source: OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)
OSM Relation: https://www.openstreetmap.org/relation/1942586

Usage:
    python3 src/data_collection/create_study_area.py
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

# ─── Project Root (2 levels up from this file) ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "raw" / "gis" / "study_area"
OUTPUT_FILE  = OUTPUT_DIR / "study_area.geojson"
CACHE_FILE   = OUTPUT_DIR / "study_area_raw.geojson"

# OSM Relation ID for Delhi (NCT) state boundary
OSM_RELATION_ID = 1942586
OSM_POLYGON_URL = f"https://polygons.openstreetmap.fr/get_geojson.py?id={OSM_RELATION_ID}&params=0"

# Approximate bounding box for Delhi NCT sanity check
DELHI_BBOX = {
    "min_lon": 76.80,
    "max_lon": 77.40,
    "min_lat": 28.40,
    "max_lat": 28.95,
}


def fetch_boundary():
    """Download the Delhi NCT geometry from OSM polygon service."""
    print(f"[INFO] Fetching Delhi NCT boundary from OSM (relation {OSM_RELATION_ID})…")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use cached file if it already exists (avoids repeated downloads)
    if CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 1000:
        print(f"[INFO] Using cached raw file: {CACHE_FILE}")
        with open(CACHE_FILE, "r") as f:
            return json.load(f)

    req = urllib.request.Request(OSM_POLYGON_URL, headers={"User-Agent": "GreenGridAI/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = json.loads(response.read().decode("utf-8"))

    with open(CACHE_FILE, "w") as f:
        json.dump(raw, f)
    print(f"[OK]   Raw boundary cached at: {CACHE_FILE}")
    return raw


def build_feature_collection(geometry: dict) -> dict:
    """
    Wrap a geometry in a proper GeoJSON FeatureCollection with metadata.
    This is the canonical output format for study_area.geojson.
    """
    return {
        "type": "FeatureCollection",
        "name": "Delhi_NCT_Study_Area",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": [
            {
                "type": "Feature",
                "id": "delhi_nct",
                "properties": {
                    "name": "Delhi NCT",
                    "full_name": "National Capital Territory of Delhi",
                    "osm_relation_id": OSM_RELATION_ID,
                    "osm_url": f"https://www.openstreetmap.org/relation/{OSM_RELATION_ID}",
                    "country": "India",
                    "admin_level": "state",
                    "crs": "EPSG:4326",
                    "source": "OpenStreetMap contributors",
                    "license": "ODbL 1.0 — https://www.openstreetmap.org/copyright",
                    "download_date": "2026-08-27",
                    "project": "GreenGrid AI — Phase 2",
                    "purpose": "Master study-area geometry for all data filtering and clipping",
                },
                "geometry": geometry,
            }
        ],
    }


def validate_geometry(geometry: dict) -> bool:
    """Run basic sanity checks on the Delhi NCT geometry."""
    errors = []

    # 1. Type check
    allowed_types = {"Polygon", "MultiPolygon"}
    if geometry.get("type") not in allowed_types:
        errors.append(f"  Unexpected geometry type: {geometry.get('type')}")

    # 2. Coordinate presence
    if not geometry.get("coordinates"):
        errors.append("  Geometry has no coordinates")

    # 3. Rough bounding-box check (Delhi NCT should be within known extents)
    def flatten_coords(coords, depth=0):
        """Recursively flatten nested coordinate lists to (lon, lat) pairs."""
        if depth == 0 and isinstance(coords[0], (int, float)):
            return [coords]
        return [pt for sub in coords for pt in flatten_coords(sub, depth - 1)]

    # Determine nesting depth
    coords = geometry["coordinates"]
    # For MultiPolygon: [[ [ [lon,lat], … ] ]] — depth 3 to get points
    depth = 3 if geometry["type"] == "MultiPolygon" else 2
    try:
        all_points = flatten_coords(coords, depth)
        lons = [p[0] for p in all_points]
        lats = [p[1] for p in all_points]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        if not (DELHI_BBOX["min_lon"] <= min_lon and max_lon <= DELHI_BBOX["max_lon"]):
            errors.append(f"  Longitude range {min_lon:.4f}–{max_lon:.4f} outside Delhi extent")
        if not (DELHI_BBOX["min_lat"] <= min_lat and max_lat <= DELHI_BBOX["max_lat"]):
            errors.append(f"  Latitude range {min_lat:.4f}–{max_lat:.4f} outside Delhi extent")

        # 4. Approximate area check (Delhi NCT ≈ 1484 km²)
        # Rough area estimate using bounding box (actual area will be less)
        bbox_area_deg2 = (max_lon - min_lon) * (max_lat - min_lat)
        # 1° lat ≈ 111 km, 1° lon ≈ 98 km at 28.6°N
        bbox_area_km2 = bbox_area_deg2 * 111 * 98
        if bbox_area_km2 < 500 or bbox_area_km2 > 5000:
            errors.append(f"  Bounding-box area {bbox_area_km2:.0f} km² seems wrong for Delhi")

        print(f"\n[GEOMETRY REPORT]")
        print(f"  Type             : {geometry['type']}")
        print(f"  Longitude extent : {min_lon:.6f} → {max_lon:.6f}")
        print(f"  Latitude extent  : {min_lat:.6f} → {max_lat:.6f}")
        print(f"  BBox area (est.) : {bbox_area_km2:.0f} km² (rough; Delhi NCT ≈ 1,484 km²)")
        print(f"  Point count      : {len(all_points)}")

    except Exception as e:
        errors.append(f"  Could not parse coordinates: {e}")

    if errors:
        print("\n[VALIDATION ERRORS]")
        for e in errors:
            print(e)
        return False

    print("  All geometry checks PASSED ✓")
    return True


def main():
    print("=" * 60)
    print("  GreenGrid AI — Phase 2 | Task 1: Study Area Boundary")
    print("=" * 60)

    # Step 1: Fetch or load raw geometry
    raw_geometry = fetch_boundary()

    # The OSM polygon service returns a bare geometry (Polygon/MultiPolygon)
    # Wrap it into a proper FeatureCollection
    geojson = build_feature_collection(raw_geometry)

    # Step 2: Validate
    geometry = geojson["features"][0]["geometry"]
    print("\n[VALIDATION]")
    valid = validate_geometry(geometry)

    if not valid:
        print("\n[ABORT] Geometry validation failed. Do not use this boundary.")
        sys.exit(1)

    # Step 3: Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"\n[OK] Study area saved: {OUTPUT_FILE}")
    print(f"     File size       : {OUTPUT_FILE.stat().st_size:,} bytes")
    print(f"     CRS             : EPSG:4326")
    print(f"     Feature count   : {len(geojson['features'])}")
    print("\n[NEXT STEP] Run validate_study_area.py to confirm the boundary.")


if __name__ == "__main__":
    main()
