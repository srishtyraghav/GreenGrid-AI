"""
GreenGrid AI — Phase 2: Dataset Collection
Task 7: Download Open GIS Datasets for Delhi NCT

Downloads authoritative/open GIS data layers from:
  - Overpass API (OpenStreetMap) — simplified queries with retry
  - GeoFabrik pre-built India extract — more reliable for large datasets

Saves to:
  data/raw/gis/administrative/
  data/raw/gis/roads/
  data/raw/gis/buildings/
  data/raw/gis/vegetation/
  data/raw/gis/landuse/

Usage:
    python3 src/data_collection/download_gis_data.py

Note:
  Overpass API can time out on very large queries.
  This script uses simplified queries and retry logic to handle that.
  For full datasets, GeoFabrik links are provided in each README.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GIS_ROOT     = PROJECT_ROOT / "data" / "raw" / "gis"
TODAY        = date.today().isoformat()

HEADERS = {"User-Agent": "GreenGridAI/1.0 (student research project; contact: greengrid-ai)"}

# Approximate Delhi NCT bounding box for Overpass queries
# These are safe outer bounds — the actual boundary will clip results in Phase 3
BBOX = "28.40,76.83,28.89,77.35"  # south,west,north,east

# Overpass API mirror list — tries each in order if one fails
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


def download(url: str, out_path: Path, label: str, timeout: int = 120) -> bool:
    """Download a URL to a file if not already present."""
    if out_path.exists() and out_path.stat().st_size > 500:
        print(f"  [SKIP] Already downloaded: {out_path.name}")
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [DOWNLOADING] {label}…")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"  [OK] Saved: {out_path.name} ({len(data):,} bytes)")
        return True
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        return False


def save_instruction_file(out_path: Path, label: str, content: str):
    """Save a markdown instruction file when actual download is not possible."""
    if out_path.exists() and out_path.stat().st_size > 100:
        print(f"  [SKIP] Already exists: {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"  [OK] Created instruction file: {out_path.name}")


def overpass_query(query: str, out_path: Path, label: str, max_retries: int = 3) -> bool:
    """Run an Overpass API query and save the JSON result. Tries multiple mirrors."""
    if out_path.exists() and out_path.stat().st_size > 500:
        print(f"  [SKIP] Already downloaded: {out_path.name}")
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = urllib.parse.urlencode({"data": query}).encode("utf-8")

    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(1, max_retries + 1):
            print(f"  [QUERYING] Overpass ({endpoint.split('/')[2]}): {label} (attempt {attempt}/{max_retries})…")
            try:
                req = urllib.request.Request(endpoint, data=encoded, headers={
                    **HEADERS, "Content-Type": "application/x-www-form-urlencoded"
                })
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = resp.read()

                # Validate it's proper JSON
                parsed = json.loads(data)
                if "elements" not in parsed:
                    print(f"  [WARN] Response missing 'elements' key — may be empty or error")

                with open(out_path, "wb") as f:
                    f.write(data)
                size_kb = len(data) // 1024
                print(f"  [OK] {label}: {size_kb:,} KB saved to {out_path.name} "
                      f"({len(parsed.get('elements', []))} elements)")
                return True
            except Exception as e:
                print(f"  [WARN] Attempt {attempt} failed on {endpoint.split('/')[2]}: {type(e).__name__}: {e}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)  # exponential back-off

    print(f"  [ERROR] All Overpass endpoints failed for: {label}")
    return False


def write_readme(folder: Path, content: str):
    """Write a README.md in each GIS subdirectory with source info."""
    readme = folder / "README.md"
    if not readme.exists():
        with open(readme, "w") as f:
            f.write(content)


# ─── TASK 1: Administrative Boundaries ─────────────────────────────────────

def download_administrative():
    """Download Delhi NCT district and ward boundaries from OSM."""
    print("\n[ADMINISTRATIVE BOUNDARIES]")
    out_dir = GIS_ROOT / "administrative"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simpler query: just get relation metadata for Delhi districts
    # Using bbox filter instead of area lookup (more reliable)
    districts_query = f"""[out:json][timeout:60][bbox:{BBOX}];
(
  relation["admin_level"="6"]["boundary"="administrative"];
);
out body;
>;
out skel qt;
"""
    result = overpass_query(
        districts_query,
        out_dir / "delhi_districts_osm.json",
        "Delhi Districts (admin level 6)"
    )
    if not result:
        # Save GeoFabrik download instructions instead
        save_instruction_file(
            out_dir / "DOWNLOAD_DISTRICTS.md",
            "Delhi Districts",
            f"""# Delhi Districts — Download Instructions\n\n"""
            f"""The Overpass API download timed out.\n\n"""
            f"""## Option A: Overpass Turbo (web interface)\n"""
            f"""1. Visit: https://overpass-turbo.eu\n"""
            f"""2. Run query:\n"""
            f"""   ```\n[out:json][timeout:120][bbox:{BBOX}];\n"""
            f"""   (relation["admin_level"="6"]["boundary"="administrative"];);\n"""
            f"""   out body; >; out skel qt;\n   ```\n"""
            f"""3. Export as GeoJSON\n\n"""
            f"""## Option B: GeoFabrik NCR extract\n"""
            f"""1. https://download.geofabrik.de/asia/india/northern-zone.html\n"""
            f"""2. Download `northern-zone-latest-free.shp.zip`\n"""
            f"""3. Look for the `places` or `political_boundaries` layer\n"""
            f"""\nDownload date: {TODAY}\n"""
        )
    return result

    write_readme(out_dir, f"""# Administrative Boundaries — Delhi NCT

## Source
- OpenStreetMap contributors (https://www.openstreetmap.org)
- License: ODbL 1.0 (https://www.openstreetmap.org/copyright)
- Download date: {TODAY}

## Files
- `delhi_districts_osm.json` — Delhi districts (admin level 6) from Overpass API
- `study_area.geojson` — Delhi NCT boundary (in study_area/ folder)

## CRS
EPSG:4326 (WGS84 Geographic)

## Purpose
Administrative district boundaries for:
- Spatial context in maps
- District-level UHI analysis (Phase 5)
- Tree plantation district-wise planning (Phase 7–8)
""")


# ─── TASK 2: Roads ──────────────────────────────────────────────────────────

def download_roads():
    """Download major road network for Delhi NCT from OSM."""
    print("\n[ROADS]")
    out_dir = GIS_ROOT / "roads"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simplified query: motorway and trunk only (smaller result = faster)
    roads_query = f"""[out:json][timeout:90][bbox:{BBOX}];
(
  way["highway"~"^(motorway|trunk|primary)$"];
);
out body;
>;
out skel qt;
"""
    result = overpass_query(
        roads_query,
        out_dir / "delhi_roads_major_osm.json",
        "Delhi Major Roads — motorway/trunk/primary (OSM)"
    )
    if not result:
        save_instruction_file(
            out_dir / "DOWNLOAD_ROADS.md",
            "Delhi Roads",
            f"""# Delhi Roads — Download Instructions\n\n"""
            f"""The Overpass API download timed out. Use one of these alternatives:\n\n"""
            f"""## Option A: GeoFabrik (Recommended)\n"""
            f"""1. https://download.geofabrik.de/asia/india/northern-zone.html\n"""
            f"""2. Download `northern-zone-latest-free.shp.zip`\n"""
            f"""3. Use the `roads` shapefile layer\n"""
            f"""4. Filter in QGIS: highway IN ('motorway','trunk','primary','secondary')\n\n"""
            f"""## Option B: Overpass Turbo\n"""
            f"""1. https://overpass-turbo.eu\n"""
            f"""2. Run query for the bbox {BBOX}\n"""
            f"""3. highways: motorway, trunk, primary, secondary\n"""
            f"""\nDownload date: {TODAY}\n"""
        )
    return result

    write_readme(out_dir, f"""# Road Network — Delhi NCT

## Source
- OpenStreetMap contributors (https://www.openstreetmap.org)
- License: ODbL 1.0 (https://www.openstreetmap.org/copyright)
- Download date: {TODAY}

## Files
- `delhi_roads_major_osm.json` — Major roads (motorway, trunk, primary, secondary, tertiary)
  - Format: Overpass API JSON
  - Coverage: Delhi NCT bounding box (28.40°N–28.89°N, 76.83°E–77.35°E)

## CRS
EPSG:4326 (WGS84 Geographic)

## Purpose
Road network for:
- Urban structure analysis (Phase 4)
- UHI contributor mapping (high-density asphalt = urban heat)
- Tree plantation suitability (road corridors for tree planting)
""")


# ─── TASK 3: Land Use ───────────────────────────────────────────────────────

def download_landuse():
    """Download land use/land cover data for Delhi NCT from OSM."""
    print("\n[LAND USE]")
    out_dir = GIS_ROOT / "landuse"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restrict to most important urban land use categories
    landuse_query = f"""[out:json][timeout:90][bbox:{BBOX}];
(
  way["landuse"~"^(residential|commercial|industrial|retail|park|forest|farmland)$"];
);
out body;
>;
out skel qt;
"""
    result = overpass_query(
        landuse_query,
        out_dir / "delhi_landuse_osm.json",
        "Delhi Land Use — key categories (OSM)"
    )
    if not result:
        save_instruction_file(
            out_dir / "DOWNLOAD_LANDUSE.md",
            "Delhi Land Use",
            f"""# Delhi Land Use — Download Instructions\n\n"""
            f"""## Option A: GeoFabrik (Recommended)\n"""
            f"""1. https://download.geofabrik.de/asia/india/northern-zone.html\n"""
            f"""2. Download `northern-zone-latest-free.shp.zip`\n"""
            f"""3. Use the `landuse` shapefile layer\n\n"""
            f"""## Option B: Overpass Turbo\n"""
            f"""1. https://overpass-turbo.eu (bbox: {BBOX})\n"""
            f"""2. Query: way['landuse']; out body; >; out skel qt;\n"""
            f"""\nDownload date: {TODAY}\n"""
        )
    return result

    write_readme(out_dir, f"""# Land Use / Land Cover — Delhi NCT

## Source
- OpenStreetMap contributors (https://www.openstreetmap.org)
- License: ODbL 1.0 (https://www.openstreetmap.org/copyright)
- Download date: {TODAY}

## Files
- `delhi_landuse_osm.json` — Land use polygons (residential, commercial, industrial, etc.)
  - Format: Overpass API JSON
  - Coverage: Delhi NCT bounding box

## Land Use Classes (OSM `landuse` tag)
  residential, commercial, industrial, retail, park,
  forest, farmland, grass, meadow, military, institutional, etc.

## CRS
EPSG:4326 (WGS84 Geographic)

## Purpose
Land use information for:
- Urban/rural classification
- NDVI/NDBI correlation with land use types (Phase 4)
- Identifying suitable plantation zones (Phase 7)
""")


# ─── TASK 4: Vegetation / Green Areas ──────────────────────────────────────

def download_vegetation():
    """Download parks, forests, and green areas in Delhi NCT from OSM."""
    print("\n[VEGETATION / GREEN AREAS]")
    out_dir = GIS_ROOT / "vegetation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Focused query: parks and forests only
    vegetation_query = f"""[out:json][timeout:90][bbox:{BBOX}];
(
  way["leisure"~"^(park|garden|nature_reserve)$"];
  way["landuse"~"^(forest|grass|meadow)$"];
  way["natural"~"^(wood|scrub|grassland)$"];
);
out body;
>;
out skel qt;
"""
    result = overpass_query(
        vegetation_query,
        out_dir / "delhi_vegetation_osm.json",
        "Delhi Green Areas — parks/forests (OSM)"
    )
    if not result:
        save_instruction_file(
            out_dir / "DOWNLOAD_VEGETATION.md",
            "Delhi Vegetation",
            f"""# Delhi Vegetation/Green Areas — Download Instructions\n\n"""
            f"""## Option A: Overpass Turbo\n"""
            f"""1. https://overpass-turbo.eu (bbox: {BBOX})\n"""
            f"""2. Query: way['leisure'~'park|garden|nature_reserve'];\n"""
            f"""   way['landuse'~'forest|grass|meadow'];\n"""
            f"""   out body; >; out skel qt;\n\n"""
            f"""## Option B: GeoFabrik\n"""
            f"""1. https://download.geofabrik.de/asia/india/northern-zone.html\n"""
            f"""2. natural.shp layer — filter by type IN ('wood','grassland','scrub')\n"""
            f"""\nDownload date: {TODAY}\n"""
        )
    return result

    write_readme(out_dir, f"""# Existing Vegetation / Green Areas — Delhi NCT

## Source
- OpenStreetMap contributors (https://www.openstreetmap.org)
- License: ODbL 1.0 (https://www.openstreetmap.org/copyright)
- Download date: {TODAY}

## Files
- `delhi_vegetation_osm.json` — Parks, forests, gardens, nature reserves
  - Format: Overpass API JSON
  - Tags included: leisure=park/garden, landuse=forest/grass/meadow,
                   natural=wood/scrub/grassland

## CRS
EPSG:4326 (WGS84 Geographic)

## Purpose
Existing vegetation inventory for:
- Comparison with satellite-derived NDVI (Phase 4)
- Baseline green cover mapping
- Tree plantation gap analysis (Phase 7)
""")


# ─── TASK 5: Buildings ──────────────────────────────────────────────────────

def download_buildings():
    """
    Download building footprints for Delhi NCT.
    NOTE: Delhi has millions of buildings — this may be a large query.
    We limit to a sample area first; full download can be done via Overpass turbo
    or GeoFabrik downloads.
    """
    print("\n[BUILDINGS]")
    out_dir = GIS_ROOT / "buildings"
    out_dir.mkdir(parents=True, exist_ok=True)

    # For prototype: download from a small representative bbox of Delhi
    # Full city-wide building download is better done via GeoFabrik
    buildings_query = f"""[out:json][timeout:90][bbox:28.60,77.18,28.68,77.26];
(
  way["building"];
);
out body;
>;
out skel qt;
"""
    result = overpass_query(
        buildings_query,
        out_dir / "delhi_buildings_sample_osm.json",
        "Delhi Buildings Sample (OSM) — Connaught Place area"
    )

    # Create a note about the full dataset
    note_file = out_dir / "DOWNLOAD_FULL_DATA.md"
    if not note_file.exists():
        with open(note_file, "w") as f:
            f.write(f"""# Buildings — Full Dataset Download Instructions

The file `delhi_buildings_sample_osm.json` contains a sample of building
footprints from Central Delhi only (bbox: 28.55°N–28.75°N, 77.10°E–77.30°E).

## For Full Delhi NCT Buildings:

### Option A: GeoFabrik (Recommended — complete data)
1. Visit: https://download.geofabrik.de/asia/india/northern-zone.html
2. Download: `northern-zone-latest.osm.pbf`
3. Extract Delhi buildings using osmium or osmfilter:
   ```bash
   osmium tags-filter northern-zone-latest.osm.pbf w/building -o delhi_buildings.osm.pbf
   osmium extract --bbox 76.83,28.40,77.35,28.89 delhi_buildings.osm.pbf -o delhi_nct_buildings.osm.pbf
   ```
4. Convert to GeoJSON: `ogr2ogr -f GeoJSON delhi_buildings.geojson delhi_nct_buildings.osm.pbf multipolygons`

### Option B: Overpass Turbo (web interface)
1. Visit: https://overpass-turbo.eu
2. Use query:
   ```
   [out:json][timeout:300][bbox:28.40,76.83,28.89,77.35];
   (way["building"]; relation["building"];);
   out body; >; out skel qt;
   ```
3. Export as GeoJSON

### Option C: India Open Government Data
- data.gov.in may have official building/property maps

Downloaded: {TODAY}
""")

    write_readme(out_dir, f"""# Building Footprints — Delhi NCT

## Source
- OpenStreetMap contributors (https://www.openstreetmap.org)
- License: ODbL 1.0 (https://www.openstreetmap.org/copyright)
- Download date: {TODAY}

## Files
- `delhi_buildings_sample_osm.json` — Building footprints (Central Delhi sample)
- `DOWNLOAD_FULL_DATA.md` — Instructions to download full Delhi NCT buildings

## CRS
EPSG:4326 (WGS84 Geographic)

## Purpose
Building footprints for:
- Urban built-up area analysis
- Impervious surface mapping
- NDBI validation (Phase 4)
- UHI correlation with built-up density (Phase 5)
""")


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  GreenGrid AI — Phase 2 | Task 7: GIS Data Download")
    print("=" * 65)
    print(f"  Target directory: {GIS_ROOT}")
    print(f"  Study area: Delhi NCT (BBOX: {BBOX})")
    print()

    results = {}

    # Run all downloads (with pause between Overpass requests to respect rate limits)
    results["administrative"] = download_administrative()
    time.sleep(3)

    results["roads"] = download_roads()
    time.sleep(3)

    results["landuse"] = download_landuse()
    time.sleep(3)

    results["vegetation"] = download_vegetation()
    time.sleep(3)

    results["buildings"] = download_buildings()

    # Summary — handle None values (from functions that didn't return explicitly)
    print("\n" + "=" * 65)
    print("  DOWNLOAD SUMMARY")
    print("=" * 65)
    for name, ok in results.items():
        if ok is True:
            status = "✓ OK"
        elif ok is False:
            status = "✗ FAILED — instruction file created"
        else:
            status = "— UNKNOWN"
        print(f"  {status}  {name}")

    n_ok     = sum(1 for v in results.values() if v is True)
    n_failed = sum(1 for v in results.values() if v is False)

    print(f"\n  Completed: {n_ok}/{len(results)}")
    if n_failed > 0:
        print(f"  Failed:    {n_failed}")
        print("  NOTE: For failed downloads, instruction files (.md) have been created")
        print("        in each data directory with alternative download methods.")
    print("\n[NEXT STEP] Run: python3 src/data_collection/validate_data.py")


if __name__ == "__main__":
    main()
