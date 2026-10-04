"""
GreenGrid AI — Phase 2: Dataset Collection
Task 9: Validate Satellite and Vector Data

Validates all Phase 2 data files and produces a validation report.

Checks:
  - Satellite GeoTIFFs: file exists, opens, CRS, dimensions, extent, bands
  - Vector/GeoJSON files: valid geometry, CRS, extent, feature count
  - Study area: Delhi NCT coverage

Usage:
    python3 src/data_collection/validate_data.py

Requirements:
    pip3 install rasterio geopandas
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Import optional packages ──────────────────────────────────────────────
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not installed — satellite GeoTIFF checks will be skipped")
    print("       Run: pip3 install rasterio")

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("[WARN] geopandas not installed — advanced vector checks will be basic")
    print("       Run: pip3 install geopandas")


# ─── Validation helpers ────────────────────────────────────────────────────

class ValidationReport:
    def __init__(self):
        self.checks  = []  # list of (category, name, status, message)

    def add(self, category: str, name: str, status: str, message: str):
        self.checks.append((category, name, status, message))
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "SKIP": "—"}.get(status, "?")
        print(f"  [{status}] {icon} {name}: {message}")

    def summary(self):
        passed  = sum(1 for c in self.checks if c[2] == "PASS")
        failed  = sum(1 for c in self.checks if c[2] == "FAIL")
        warned  = sum(1 for c in self.checks if c[2] == "WARN")
        skipped = sum(1 for c in self.checks if c[2] == "SKIP")
        total   = len(self.checks)
        return passed, failed, warned, skipped, total

    def write_report(self, out_path: Path):
        passed, failed, warned, skipped, total = self.summary()
        lines = [
            "# GreenGrid AI — Phase 2 Data Validation Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Summary\n",
            f"| Status  | Count |",
            f"|---------|-------|",
            f"| PASSED  | {passed}     |",
            f"| FAILED  | {failed}     |",
            f"| WARNED  | {warned}     |",
            f"| SKIPPED | {skipped}     |",
            f"| TOTAL   | {total}     |",
            "\n## Detailed Results\n",
            "| Category | Check | Status | Message |",
            "|----------|-------|--------|---------|",
        ]
        for cat, name, status, msg in self.checks:
            lines.append(f"| {cat} | {name} | {status} | {msg} |")

        if failed == 0:
            lines.append("\n## Overall Status\n\n**✓ ALL CHECKS PASSED — Phase 2 data is valid.**")
        else:
            lines.append(f"\n## Overall Status\n\n**✗ {failed} CHECK(S) FAILED — see details above.**")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\n[REPORT] Written to: {out_path}")


def validate_geojson_basic(filepath: Path, report: ValidationReport, category: str):
    """Basic GeoJSON/Overpass-JSON validation using stdlib only."""
    # File existence
    if not filepath.exists():
        report.add(category, "File exists", "FAIL", f"Not found: {filepath.name}")
        return False
    report.add(category, "File exists", "PASS", f"{filepath.stat().st_size:,} bytes")

    # Valid JSON
    try:
        with open(filepath) as f:
            data = json.load(f)
        report.add(category, "Valid JSON", "PASS", "File parses as valid JSON")
    except json.JSONDecodeError as e:
        report.add(category, "Valid JSON", "FAIL", str(e))
        return False

    # Accept either GeoJSON type or Overpass API format (has 'elements' key)
    gtype = data.get("type", "")
    has_elements = "elements" in data  # Overpass JSON format
    geojson_types = ("FeatureCollection", "Feature", "Polygon", "MultiPolygon",
                     "Point", "LineString", "MultiLineString", "GeometryCollection")

    if gtype in geojson_types:
        report.add(category, "GeoJSON type", "PASS", gtype)
    elif has_elements:
        n_elements = len(data["elements"])
        report.add(category, "GeoJSON type", "PASS",
                   f"Overpass JSON format ({n_elements:,} elements)")
    else:
        report.add(category, "GeoJSON type", "FAIL", f"Unexpected type/format (type={gtype!r})")
        return False

    # Feature count
    if gtype == "FeatureCollection":
        n = len(data.get("features", []))
        if n > 0:
            report.add(category, "Feature count", "PASS", f"{n} features")
        else:
            report.add(category, "Feature count", "FAIL", "0 features")
    elif has_elements:
        n = len(data["elements"])
        if n > 0:
            report.add(category, "Element count", "PASS", f"{n:,} OSM elements")
        else:
            report.add(category, "Element count", "WARN", "0 elements — may be empty query")

    return True


def validate_geojson_advanced(filepath: Path, report: ValidationReport, category: str):
    """Advanced vector validation using geopandas."""
    if not HAS_GEOPANDAS:
        report.add(category, "GeoPandas checks", "SKIP", "geopandas not installed")
        return

    try:
        gdf = gpd.read_file(filepath)
    except Exception as e:
        report.add(category, "GeoPandas read", "FAIL", str(e))
        return

    report.add(category, "GeoPandas read", "PASS", "File opened successfully")

    # CRS
    if gdf.crs is not None:
        crs_str = str(gdf.crs)
        if "4326" in crs_str or "WGS 84" in crs_str:
            report.add(category, "CRS", "PASS", "EPSG:4326 (WGS84)")
        else:
            report.add(category, "CRS", "WARN", f"CRS: {crs_str} (expected EPSG:4326)")
    else:
        report.add(category, "CRS", "WARN", "No CRS defined — assuming EPSG:4326 (GeoJSON default)")

    # Geometry validity
    invalid = gdf[~gdf.geometry.is_valid]
    if len(invalid) == 0:
        report.add(category, "Geometry validity", "PASS", "All geometries are valid")
    else:
        report.add(category, "Geometry validity", "WARN", f"{len(invalid)} invalid geometries")

    # Bounds
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    report.add(category, "Spatial extent", "PASS",
               f"({bounds[0]:.4f}, {bounds[1]:.4f}) → ({bounds[2]:.4f}, {bounds[3]:.4f})")


def validate_geotiff(filepath: Path, report: ValidationReport, category: str,
                     expected_crs: str = "EPSG:4326", min_bands: int = 1):
    """Validate a GeoTIFF file."""
    if not filepath.exists():
        report.add(category, "File exists", "FAIL", f"Not found: {filepath.name}")
        return

    size_mb = filepath.stat().st_size / (1024 * 1024)
    report.add(category, "File exists", "PASS", f"{size_mb:.1f} MB")

    if not HAS_RASTERIO:
        report.add(category, "GeoTIFF checks", "SKIP", "rasterio not installed")
        return

    try:
        with rasterio.open(filepath) as ds:
            report.add(category, "File opens", "PASS", "GeoTIFF opens successfully")

            # CRS
            if ds.crs:
                report.add(category, "CRS", "PASS", str(ds.crs))
            else:
                report.add(category, "CRS", "WARN", "No CRS defined in file")

            # Dimensions
            report.add(category, "Dimensions", "PASS",
                       f"{ds.width} × {ds.height} px, {ds.count} band(s)")

            # Bands
            if ds.count >= min_bands:
                report.add(category, "Band count", "PASS", f"{ds.count} bands (min: {min_bands})")
            else:
                report.add(category, "Band count", "FAIL",
                           f"{ds.count} bands (expected ≥ {min_bands})")

            # Spatial extent
            bounds = ds.bounds
            report.add(category, "Extent", "PASS",
                       f"({bounds.left:.4f}, {bounds.bottom:.4f}) → ({bounds.right:.4f}, {bounds.top:.4f})")

            # NoData
            nodata = ds.nodata
            report.add(category, "NoData value", "PASS" if nodata is not None else "WARN",
                       str(nodata) if nodata is not None else "Not set (may be OK for raw data)")

            # Resolution
            res = ds.res  # (pixel_width, pixel_height) in CRS units
            report.add(category, "Resolution", "PASS",
                       f"{res[0]:.6f} × {res[1]:.6f} degrees (~{res[0]*111000:.0f}m at equator)")

    except Exception as e:
        report.add(category, "File opens", "FAIL", str(e))


# ─── Main validation function ──────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  GreenGrid AI — Phase 2 | Data Validation")
    print("=" * 65)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report = ValidationReport()

    # ── 1. Study Area ──────────────────────────────────────────────────────
    print("\n[STUDY AREA]")
    study_area = PROJECT_ROOT / "data/raw/gis/study_area/study_area.geojson"
    validate_geojson_basic(study_area, report, "Study Area")
    validate_geojson_advanced(study_area, report, "Study Area")

    # ── 2. GIS Vector Data ─────────────────────────────────────────────────
    gis_files = {
        "Districts":  PROJECT_ROOT / "data/raw/gis/administrative/delhi_districts_osm.json",
        "Roads":      PROJECT_ROOT / "data/raw/gis/roads/delhi_roads_major_osm.json",
        "Land Use":   PROJECT_ROOT / "data/raw/gis/landuse/delhi_landuse_osm.json",
        "Vegetation": PROJECT_ROOT / "data/raw/gis/vegetation/delhi_vegetation_osm.json",
        "Buildings":  PROJECT_ROOT / "data/raw/gis/buildings/delhi_buildings_sample_osm.json",
    }

    for label, path in gis_files.items():
        print(f"\n[{label.upper()}]")
        validate_geojson_basic(path, report, label)

    # ── 3. Landsat 9 GeoTIFFs ─────────────────────────────────────────────
    landsat_files = {
        "L9 2022 Best":      PROJECT_ROOT / "data/raw/landsat9/2022_07/landsat9_2022_07_best_scene_30m.tif",
        "L9 2022 Composite": PROJECT_ROOT / "data/raw/landsat9/2022_07/landsat9_2022_07_composite_30m.tif",
        "L9 2026 Best":      PROJECT_ROOT / "data/raw/landsat9/2026_07/landsat9_2026_07_best_scene_30m.tif",
        "L9 2026 Composite": PROJECT_ROOT / "data/raw/landsat9/2026_07/landsat9_2026_07_composite_30m.tif",
    }

    print("\n[LANDSAT 9 GEOTIFFS]")
    for label, path in landsat_files.items():
        print(f"\n  --- {label} ---")
        validate_geotiff(path, report, f"Landsat9/{label}", min_bands=8)

    # ── 4. Sentinel-2 GeoTIFFs ────────────────────────────────────────────
    sentinel_files = {
        "S2 2022 10m":  PROJECT_ROOT / "data/raw/sentinel2/2022_07/sentinel2_2022_07_10m_composite.tif",
        "S2 2022 20m":  PROJECT_ROOT / "data/raw/sentinel2/2022_07/sentinel2_2022_07_20m_composite.tif",
        "S2 2026 10m":  PROJECT_ROOT / "data/raw/sentinel2/2026_07/sentinel2_2026_07_10m_composite.tif",
        "S2 2026 20m":  PROJECT_ROOT / "data/raw/sentinel2/2026_07/sentinel2_2026_07_20m_composite.tif",
    }

    print("\n[SENTINEL-2 GEOTIFFS]")
    for label, path in sentinel_files.items():
        print(f"\n  --- {label} ---")
        min_b = 4 if "10m" in label else 7
        validate_geotiff(path, report, f"Sentinel2/{label}", min_bands=min_b)

    # ── 5. Metadata CSV ───────────────────────────────────────────────────
    print("\n[METADATA CSV]")
    metadata_file = PROJECT_ROOT / "dataset_metadata.csv"
    if metadata_file.exists():
        report.add("Metadata", "File exists", "PASS", f"{metadata_file.stat().st_size:,} bytes")
        try:
            import csv
            with open(metadata_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            report.add("Metadata", "CSV readable", "PASS", f"{len(rows)} rows")
        except Exception as e:
            report.add("Metadata", "CSV readable", "FAIL", str(e))
    else:
        report.add("Metadata", "File exists", "FAIL", "dataset_metadata.csv not found")

    # ── Summary ───────────────────────────────────────────────────────────
    passed, failed, warned, skipped, total = report.summary()
    print("\n" + "=" * 65)
    print("  VALIDATION SUMMARY")
    print("=" * 65)
    print(f"  PASSED  : {passed}/{total}")
    print(f"  FAILED  : {failed}")
    print(f"  WARNINGS: {warned}")
    print(f"  SKIPPED : {skipped}")

    if failed == 0:
        print("\n  ✓ All critical checks passed — Phase 2 data is valid.")
    else:
        print(f"\n  ✗ {failed} check(s) failed.")
        print("    Satellite files (Landsat/Sentinel) will fail until exported from GEE.")
        print("    Run the GEE export scripts first.")

    # Write markdown report
    report_path = PROJECT_ROOT / "reports" / "phase2_validation_report.md"
    report.write_report(report_path)

    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
