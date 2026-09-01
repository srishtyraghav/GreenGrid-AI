"""
GreenGrid AI — Phase 3: Data Preprocessing
Vector-to-raster conversion for OpenStreetMap GIS layers.

The OSM data downloaded in Phase 2 are in Overpass API JSON format, which
contains 'node', 'way', and 'relation' elements. This module parses the
'way' elements into GeoDataFrames and rasterizes them onto the common 30 m
Landsat grid. Relation-based district boundaries are not required for Phase 3
and are therefore skipped to keep the scope focused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Polygon

from . import config
from .io import build_profile, write_raster


def parse_overpass_json(path: Path) -> gpd.GeoDataFrame:
    """
    Parse an Overpass API JSON file into a GeoDataFrame of way geometries.

    Nodes are stored by id; ways reference node ids to form LineStrings or
    Polygons (closed ways). Tags from each way are preserved as columns.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    elements = data.get("elements", [])

    # First pass: collect all node coordinates.
    # Overpass JSON usually lists nodes before ways, but a two-pass approach
    # makes the parser robust to any element ordering.
    nodes = {}
    for element in elements:
        if element.get("type") == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])

    # Second pass: build way geometries.
    ways = []
    for element in elements:
        if element.get("type") != "way":
            continue

        node_ids = element.get("nodes", [])
        coords = [nodes.get(nid) for nid in node_ids]
        coords = [c for c in coords if c is not None]

        if len(coords) < 2:
            continue

        is_closed = (len(coords) >= 4) and (coords[0] == coords[-1])
        if is_closed:
            geom = Polygon(coords)
        else:
            geom = LineString(coords)

        record = {"geometry": geom, **element.get("tags", {})}
        ways.append(record)

    if not ways:
        # Return an empty GeoDataFrame with the correct CRS and geometry column.
        return gpd.GeoDataFrame({"geometry": []}, crs=config.TARGET_CRS)

    gdf = gpd.GeoDataFrame(ways, crs=config.TARGET_CRS)
    return gdf


def rasterize_shapes(
    shapes: List[Tuple],
    reference_path: Path,
    fill_value: float = 0.0,
    dtype: str = "float64",
    all_touched: bool = False,
) -> np.ndarray:
    """
    Rasterize a list of (geometry, value) tuples onto the reference grid.

    Parameters
    ----------
    shapes : list of (geometry, value)
        Geometries to burn.
    reference_path : Path
        Reference raster defining width, height, transform, CRS.
    fill_value : float
        Background value.
    dtype : str
        Output dtype.
    all_touched : bool
        If True, all pixels touched by a geometry receive its value.

    Returns
    -------
    np.ndarray
        2-D raster.
    """
    with rasterio.open(reference_path) as ref:
        out_shape = (ref.height, ref.width)
        transform = ref.transform
        crs = ref.crs

    arr = features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=fill_value,
        dtype=dtype,
        all_touched=all_touched,
    )
    return arr


def rasterize_landuse(
    input_path: Path = config.GIS_DIR / "landuse" / "delhi_landuse_osm.json",
    output_path: Path = config.MASKS_DIR / "landuse_raster_30m.tif",
    reference_path: Path = config.REFERENCE_RASTER,
) -> Dict:
    """
    Rasterize OSM landuse polygons onto the 30 m grid.

    Returns a class-coded raster and a class mapping dictionary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = parse_overpass_json(input_path)

    # Select meaningful landuse classes for urban/vegetation analysis
    class_priority = [
        "park",
        "forest",
        "grass",
        "commercial",
        "industrial",
        "residential",
        "retail",
        "farmland",
    ]

    def classify(row) -> Optional[str]:
        lu = str(row.get("landuse", "")).lower()
        # Direct match
        if lu in class_priority:
            return lu
        # Leisure fallback
        leisure = str(row.get("leisure", "")).lower()
        if leisure in ("park", "garden", "nature_reserve"):
            return "park"
        if leisure in ("pitch", "playground"):
            return "grass"
        # Natural fallback
        natural = str(row.get("natural", "")).lower()
        if natural in ("wood", "scrub", "grassland"):
            return "forest" if natural == "wood" else "grass"
        return None

    gdf["class"] = gdf.apply(classify, axis=1)
    gdf = gdf.dropna(subset=["class"]).copy()

    class_to_code = {cls: i + 1 for i, cls in enumerate(class_priority)}
    code_to_class = {v: k for k, v in class_to_code.items()}
    code_to_class[0] = "unclassified_background"  # fill value is a valid class
    gdf["code"] = gdf["class"].map(class_to_code)

    shapes = [(geom, code) for geom, code in zip(gdf.geometry, gdf["code"])]
    arr = rasterize_shapes(shapes, reference_path, fill_value=0.0, dtype="uint8", all_touched=True)

    # Use 255 as nodata so that the background class 0 remains a valid value
    profile = build_profile(reference_path, count=1, dtype="uint8", nodata=255)
    write_raster(output_path, arr, profile, band_names=["landuse_class"])

    return {
        "output": str(output_path),
        "class_mapping": code_to_class,
        "pixel_counts": {code_to_class.get(int(k), "other"): int(v) for k, v in zip(*np.unique(arr, return_counts=True))},
    }


def rasterize_distance_layer(
    input_path: Path,
    output_path: Path,
    reference_path: Path,
    tag_filter: Optional[Dict[str, List[str]]] = None,
    buffer_m: Optional[float] = None,
) -> Dict:
    """
    Rasterize vector features as a binary mask and compute Euclidean distance
    to the nearest feature (in pixels, then converted to metres).

    Parameters
    ----------
    input_path : Path
        Overpass JSON file.
    output_path : Path
        Destination for the distance raster.
    reference_path : Path
        Reference 30 m grid.
    tag_filter : dict, optional
        e.g. {"highway": ["motorway", "trunk", "primary"]}. If None, all
        ways with non-empty tags are used.
    buffer_m : float, optional
        If provided, features are buffered by this distance (metres) before
        rasterization.

    Returns
    -------
    dict
        Output path and summary statistics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = parse_overpass_json(input_path)

    if tag_filter:
        mask = np.zeros(len(gdf), dtype=bool)
        for key, values in tag_filter.items():
            mask |= gdf.get(key, "").str.lower().isin([v.lower() for v in values])
        gdf = gdf[mask].copy()

    if gdf.empty:
        # No features: output all-no-data distance raster
        with rasterio.open(reference_path) as ref:
            arr = np.full((ref.height, ref.width), np.nan, dtype="float64")
    else:
        if buffer_m is not None:
            gdf["geometry"] = gdf.to_crs(epsg=32643).buffer(buffer_m).to_crs(config.TARGET_CRS)

        shapes = [(geom, 1) for geom in gdf.geometry if geom.is_valid]
        binary = rasterize_shapes(shapes, reference_path, fill_value=0.0, dtype="uint8", all_touched=True)

        # Euclidean distance in pixels; convert to metres using 30 m resolution
        distances_px = distance_transform_edt(binary == 0)
        distances_m = distances_px * config.TARGET_RESOLUTION_M_APPROX
        arr = distances_m.astype("float64")

    profile = build_profile(reference_path, count=1, dtype="float64")
    write_raster(output_path, arr, profile, band_names=["distance_m"])

    finite = np.isfinite(arr)
    return {
        "output": str(output_path),
        "feature_count": len(gdf),
        "distance_min_m": float(np.min(arr[finite])) if np.any(finite) else None,
        "distance_max_m": float(np.max(arr[finite])) if np.any(finite) else None,
        "distance_mean_m": float(np.mean(arr[finite])) if np.any(finite) else None,
    }


def process_all_vector_layers() -> Dict:
    """Convert all relevant OSM layers to 30 m rasters."""
    results = {}

    print("[VECTOR] Rasterizing landuse polygons")
    results["landuse"] = rasterize_landuse()

    print("[VECTOR] Rasterizing roads distance")
    results["roads_distance"] = rasterize_distance_layer(
        config.GIS_DIR / "roads" / "delhi_roads_major_osm.json",
        config.MASKS_DIR / "roads_distance_30m.tif",
        config.REFERENCE_RASTER,
        tag_filter={"highway": ["motorway", "trunk", "primary", "secondary", "tertiary"]},
    )

    print("[VECTOR] Rasterizing vegetation distance")
    results["vegetation_distance"] = rasterize_distance_layer(
        config.GIS_DIR / "vegetation" / "delhi_vegetation_osm.json",
        config.MASKS_DIR / "vegetation_distance_30m.tif",
        config.REFERENCE_RASTER,
    )

    print("[VECTOR] Rasterizing buildings mask")
    results["buildings"] = rasterize_distance_layer(
        config.GIS_DIR / "buildings" / "delhi_buildings_sample_osm.json",
        config.MASKS_DIR / "buildings_distance_30m.tif",
        config.REFERENCE_RASTER,
    )

    return results
