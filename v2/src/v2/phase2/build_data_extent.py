"""Build the data-extent study-area boundary for display.

The GEE export scripts (V1 and V2) clipped every raster to the
FAO/GAUL/2015 level-1 "Delhi" polygon. The geojson previously served as the
study-area boundary (OSM relation 1942586) does not match that clip geometry,
so the dashboard outline and the colored rasters diverge along the NCT fringe.

This script derives the honest display boundary from the data itself: the
vectorized envelope of every pixel that holds valid data in at least one of
the five W4 years (Phase 6 severity mask UNION Phase 3 LST mask). Disconnected
regions are preserved as separate features and interior never-valid gaps
become polygon holes. No raster or model output is modified.

Output: v2/data/gis/study_area/study_area_data_extent.geojson
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import mapping, shape

PROJECT_ROOT = Path(__file__).resolve().parents[3]
YEARS = (2022, 2023, 2024, 2025, 2026)
OUT = PROJECT_ROOT / "data" / "gis" / "study_area" / "study_area_data_extent.geojson"

# Douglas-Peucker tolerance in degrees (~30 m at Delhi's latitude), applied
# after polygonization to smooth pixel staircase edges without moving the
# boundary meaningfully.
SIMPLIFY_TOLERANCE_DEG = 0.0003


def load_union_mask() -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    union = None
    transform = crs = None
    for year in YEARS:
        for path in (
            PROJECT_ROOT / "data" / "phase6" / "rasters" / f"severity_{year}.tif",
            PROJECT_ROOT / "data" / "phase3" / str(year) / "lst_30m.tif",
        ):
            with rasterio.open(path) as ds:
                arr = ds.read(1)
                valid = np.isfinite(arr) & (arr != ds.nodata) if ds.nodata is not None else np.isfinite(arr)
                if union is None:
                    union = np.zeros(valid.shape, dtype=bool)
                    transform, crs = ds.transform, ds.crs
                elif valid.shape != union.shape:
                    raise AssertionError(f"grid mismatch for {path}")
                union |= valid
    if union is None or not union.any():
        raise AssertionError("union mask is empty")
    return union, transform, crs


def main() -> int:
    union, transform, crs = load_union_mask()

    # Polygonize: contiguous valid regions become separate polygon features;
    # enclosed never-valid gaps become interior rings (holes).
    feats = [
        shape(geom)
        for geom, val in shapes(union.astype(np.uint8), mask=union, transform=transform)
        if val == 1
    ]
    if not feats:
        raise AssertionError("polygonization produced no features")

    features = []
    for i, geom in enumerate(sorted(feats, key=lambda g: g.area, reverse=True), start=1):
        simplified = geom.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
        features.append({
            "type": "Feature",
            "properties": {"region_id": i, "area_ha": round(simplified.area, 1)},
            "geometry": mapping(simplified),
        })

    n_regions = len(features)
    n_holes = sum(len(f["geometry"].get("coordinates", [[]])[0]) - 1 for f in features
                  if f["geometry"]["type"] == "Polygon")
    multi = [f for f in features if f["geometry"]["type"] == "MultiPolygon"]
    for f in multi:  # count holes across multipolygon parts
        n_holes += sum(len(p) - 1 for p in f["geometry"]["coordinates"])

    out = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
        "metadata": {
            "name": "Delhi study area — data extent",
            "description": (
                "Vectorized envelope of all pixels with valid data in at least one of the "
                "five W4 years (Phase 6 severity UNION Phase 3 LST). This is the geometry "
                "the displayed rasters actually cover."
            ),
            "operational_clip_geometry": (
                "FAO/GAUL/2015 level-1 ADM1_NAME=Delhi (used by all V1/V2 GEE export "
                "scripts). The OSM relation 1942586 geojson is kept separately as the "
                "reference 'Delhi NCT (OSM)' boundary and differs from this extent along "
                "the NCT fringe by ~7,850 ha outside and ~5,230 ha inside."
            ),
            "pixel_size_m": 30,
            "grid": "1768 x 1874, EPSG:4326, origin 76.8329062507 / 28.8846991402",
            "union_rule": "valid in >= 1 of 5 years (2022-2026)",
            "simplify_tolerance_deg": SIMPLIFY_TOLERANCE_DEG,
            "n_regions": n_regions,
            "n_interior_holes": n_holes,
            "generated_by": "v2/src/v2/phase2/build_data_extent.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"regions: {n_regions} | holes: {n_holes} | "
          f"union area: {union.sum() * 0.09:,.1f} ha | wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
