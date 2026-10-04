"""V2 Phase 3 core-product builder.

Builds, per year, on the AUTHORITATIVE 30 m grid (default: the 2026 L9 W4
composite; ``--grid-file`` overrides for tests):

  valid masks (uint8, 1 valid / 0 invalid / 255 outside-study)
    L9       finite ST_B10 & finite SR_B2 & QA_CLEAR_COUNT >= 1
    S2-10m   finite B2..B8 & B2 > 0 (zero-fill guard) & VALID_COUNT >= 1
    S2-20m   finite B5..B12
    LULC     finite Dynamic World label (classes 0..8)
  products (float64 NaN-masked, V1 naming conventions)
    lst_30m.tif                     LST °C from raw ST_B10 DN
    ndvi_30m.tif / ndbi_30m.tif / ndre_30m.tif   Sentinel-2 authoritative
    l9_ndvi_30m.tif / l9_ndbi_30m.tif            Landsat diagnostics
    ndmi_30m.tif / mndwi_30m.tif / bsi_30m.tif   Landsat 9 Tier-1 indices
    vegetation_cover_30m.tif        PVC = clamp((NDVI-0.05)/0.75, 0, 1)
    lulc_30m.tif                    DW label, nearest 10 m -> 30 m (+ valid mask)
  static constraint rasters (once, under ``static/``)
    landuse_raster_30m.tif          OSM landuse classes 0..8 (255 nodata)
    roads_distance_30m.tif          OSM major-road lines (V1-parity classes)
    vegetation_distance_30m.tif     OSM vegetation (V1-parity tag union)
    buildings_distance_30m.tif      V2 full-NCT OSM buildings GeoJSON
    water_presence_30m.tif          V2 OSM water polygons (0/1)
    road_surfaces_presence_30m.tif  V2 OSM road-surface polygons (0/1)
  All context layers come from V2's own GeoJSON fetch
  (data/v2/phase2/constraints/, CONTEXT_LAYERS_PROVENANCE.md); V1's Overpass
  JSON is no longer read anywhere in the feature path. The stored roads
  layer is the FULL network; features use the V1-equivalent subset
  (motorway/trunk/primary) unless --roads-full is given.

All Sentinel-2 / LULC products are reprojected ONTO the authoritative grid
explicitly (``rasterio.warp.reproject``; average for continuous, nearest for
SCL / labels), fixing the V1 F4 grid defects (S2 20 m half-pixel y-offset,
S2 10 m / LULC 2 px extent mismatch) by construction.

Usage:
    PYTHONPATH=src .venv/Scripts/python.exe -m src.v2.phase3.build_core \
        --data-root data/raw/v2 --out-root data/v2/phase3 \
        [--years 2022,...] [--grid-file ...] [--skip-vectors]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features as rio_features
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage

from ..common import (
    DATA_RAW_V2,
    DIST_RESOLUTION_M_APPROX,
    GRID_FILE_DEFAULT,
    L9_BANDS,
    L9_PATH,
    LANDUSE_CLASS_PRIORITY,
    LANDUSE_CODE_TO_CLASS,
    LULC_PATH,
    MET_WINDOW_HOURS_UTC,
    PROJECT_ROOT,
    S2_10M_BANDS,
    S2_10M_PATH,
    S2_20M_BANDS,
    S2_20M_PATH,
    STUDY_AREA_DEFAULT,
    TARGET_CRS,
    V1_LANDUSE_VALUES,
    V1_ROAD_HIGHWAY_CLASSES,
    V1_VEGETATION_TAGS,
    V2_BUILDINGS_GEOJSON,
    V2_LANDUSE_GEOJSON,
    V2_ROAD_SURFACES_GEOJSON,
    V2_ROADS_GEOJSON,
    V2_VEGETATION_GEOJSON,
    V2_WATER_GEOJSON,
    WINDOW_TAG,
    YEARS,
    Grid,
    dump_json,
    lst_from_dn,
    norm_diff,
    read_band_nan,
    reproject_to_grid,
    sha256_file,
    valid_l9_mask,
    valid_lulc_mask,
    valid_mask_uint8,
    valid_s2_10m_mask,
    valid_s2_20m_mask,
    vegetation_cover_from_ndvi,
    write_band,
    load_study_mask,
)


# ---------------------------------------------------------------------------
# Vector layers
# ---------------------------------------------------------------------------
def parse_overpass_json(path: Path) -> gpd.GeoDataFrame:
    """Tolerant port of V1 ``parse_overpass_json`` (src/preprocessing/vector_raster.py).

    RETAINED BUT NO LONGER USED FOR FEATURES: V2 reads its own GeoJSON
    context layers (data/v2/phase2/constraints/). Kept for provenance/audit
    of legacy Overpass JSON inputs only.

    V1 OSM .json files are Overpass JSON (node/way/relation elements), NOT
    GeoJSON. Ways with missing nodes, degenerate rings or invalid geometries
    are skipped or repaired instead of raising.
    """
    import json as _json
    from shapely.geometry import LineString, Polygon

    data = _json.loads(Path(path).read_text(encoding="utf-8"))
    elements = data.get("elements", [])

    nodes = {}
    for el in elements:
        if el.get("type") == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    ways = []
    n_skipped = 0
    for el in elements:
        if el.get("type") != "way":
            continue
        coords = [nodes.get(nid) for nid in el.get("nodes", [])]
        coords = [c for c in coords if c is not None]
        if len(coords) < 2:
            n_skipped += 1
            continue
        is_closed = len(coords) >= 4 and coords[0] == coords[-1]
        try:
            geom = Polygon(coords) if is_closed else LineString(coords)
            if not geom.is_valid:
                from shapely import make_valid
                geom = make_valid(geom)
            if geom.is_empty:
                n_skipped += 1
                continue
        except Exception:
            n_skipped += 1
            continue
        ways.append({"geometry": geom, **el.get("tags", {})})

    gdf = gpd.GeoDataFrame(ways, crs=TARGET_CRS) if ways else \
        gpd.GeoDataFrame({"geometry": []}, crs=TARGET_CRS, geometry="geometry")
    gdf.attrs["n_skipped"] = n_skipped
    return gdf


def read_constraints_geojson(path: Path) -> gpd.GeoDataFrame:
    """Read a V2 constraint GeoJSON, dropping (or repairing) invalid geometries."""
    from shapely import make_valid

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    n_bad = int((~gdf.geometry.is_valid).sum())
    if n_bad:
        fixed = gdf.geometry.apply(lambda g: make_valid(g) if not g.is_valid else g)
        gdf = gdf.set_geometry(fixed)
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    gdf.attrs["n_repaired"] = n_bad
    return gdf


def _tag_in(gdf: gpd.GeoDataFrame, column: str, values) -> np.ndarray:
    """Case-insensitive NaN-safe membership test on a tag column."""
    col = gdf.get(column)
    if col is None:
        return np.zeros(len(gdf), dtype=bool)
    lowered = col.astype("string").str.lower()
    return lowered.isin([v.lower() for v in values]).fillna(False).to_numpy(dtype=bool)


def filter_v1_road_classes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """V1-parity road subset: highway in {motorway, trunk, primary} (exact).

    V1's fetch query was way["highway"~"^(motorway|trunk|primary)$"] and its
    file contains exactly those three classes (verified read-only); V1's
    preprocessing filter (motorway..tertiary) was a superset on that file.
    """
    return gdf[_tag_in(gdf, "highway", V1_ROAD_HIGHWAY_CLASSES)].copy()


def filter_v1_vegetation(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """V1-parity vegetation subset: V1's verbatim query union.

    leisure in {park, garden, nature_reserve} OR landuse in
    {forest, grass, meadow} OR natural in {wood, scrub, grassland}.
    Excludes V2's extra recreation_ground fetch (not in V1's query).
    """
    mask = np.zeros(len(gdf), dtype=bool)
    for column, values in V1_VEGETATION_TAGS.items():
        mask |= _tag_in(gdf, column, values)
    return gdf[mask].copy()


def filter_v1_landuse(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """V1-parity landuse subset: landuse tag in V1's 7 fetched values.

    V1 fetched ONLY way["landuse"~"^(residential|commercial|industrial|
    retail|park|forest|farmland)$"]; classify_landuse's leisure/natural
    fallbacks could never fire on V1 data, so rows without one of these
    landuse values are dropped (mirrors V1's query, not a semantic change).
    """
    return gdf[_tag_in(gdf, "landuse", V1_LANDUSE_VALUES)].copy()


def rasterize_binary(grid: Grid, gdf: gpd.GeoDataFrame, all_touched: bool = True) -> np.ndarray:
    """Rasterize vector features as 1 on the feature, 0 background (V1 recipe)."""
    if gdf.empty:
        return np.zeros(grid.shape, dtype=np.uint8)
    shapes = [(geom, 1) for geom in gdf.to_crs(grid.crs).geometry]
    return rio_features.rasterize(
        shapes, out_shape=grid.shape, transform=grid.transform,
        fill=0, dtype="uint8", all_touched=all_touched,
    )


def distance_raster(binary: np.ndarray) -> np.ndarray:
    """V1 semantics: EDT on (binary == 0), Euclidean px distance * 30 m.

    Feature pixels therefore have distance == 0 exactly.
    """
    if not binary.any():
        return np.full(binary.shape, np.nan, dtype=np.float64)
    dist_px = ndimage.distance_transform_edt(binary == 0)
    return (dist_px * DIST_RESOLUTION_M_APPROX).astype(np.float64)


def classify_landuse(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """V1 landuse classification (src/preprocessing/vector_raster.py)."""

    def classify(row):
        lu = str(row.get("landuse", "")).lower()
        if lu in LANDUSE_CLASS_PRIORITY:
            return lu
        leisure = str(row.get("leisure", "")).lower()
        if leisure in ("park", "garden", "nature_reserve"):
            return "park"
        if leisure in ("pitch", "playground"):
            return "grass"
        natural = str(row.get("natural", "")).lower()
        if natural in ("wood", "scrub", "grassland"):
            return "forest" if natural == "wood" else "grass"
        return None

    gdf = gdf.copy()
    gdf["class"] = gdf.apply(classify, axis=1)
    gdf = gdf.dropna(subset=["class"]).copy()
    class_to_code = {cls: i + 1 for i, cls in enumerate(LANDUSE_CLASS_PRIORITY)}
    gdf["code"] = gdf["class"].map(class_to_code)
    return gdf


def build_static_layers(grid: Grid, out_dir: Path, study_area: Path,
                        roads_geojson: Path, vegetation_geojson: Path,
                        landuse_geojson: Path, buildings_geojson: Path,
                        water_geojson: Path, road_surfaces_geojson: Path,
                        roads_full: bool = False) -> dict:
    """Rasterize all static constraint layers once, on the authoritative grid.

    All context layers are V2's own GeoJSON fetch
    (data/v2/phase2/constraints/). Feature semantics follow V1 exactly:
    roads default to the V1-parity major-road subset (the stored layer is the
    full network; ``roads_full=True`` burns everything), vegetation to V1's
    query union, landuse to V1's 7 fetched landuse values.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"grid_transform": [grid.transform.a, grid.transform.b, grid.transform.c,
                                     grid.transform.d, grid.transform.e, grid.transform.f],
                  "grid_shape": [grid.height, grid.width], "layers": {}}

    def _dist_layer(name: str, gdf: gpd.GeoDataFrame, source: Path,
                    n_stored: int | None = None, filter_note: str | None = None) -> None:
        binary = rasterize_binary(grid, gdf)
        dist = distance_raster(binary)
        path = out_dir / f"{name}_30m.tif"
        write_band(path, grid, dist, dtype="float64", nodata=np.nan, band_name="distance_m")
        finite = np.isfinite(dist)
        meta["layers"][name] = {
            "path": str(path), "source": str(source),
            "n_features_burned": int(len(gdf)),
            "sha256_source": sha256_file(source) if source.exists() else None,
            "distance_min_m": float(dist[finite].min()) if finite.any() else None,
            "distance_max_m": float(dist[finite].max()) if finite.any() else None,
        }
        if n_stored is not None:
            meta["layers"][name]["n_features_stored_in_source"] = int(n_stored)
        if filter_note:
            meta["layers"][name]["feature_filter"] = filter_note

    # ---- OSM context layers (V2 GeoJSON, V1-parity feature semantics) ------
    gdf_roads_all = read_constraints_geojson(roads_geojson)
    if roads_full:
        gdf_roads = gdf_roads_all
        note = "none (--roads-full: full network burned)"
    else:
        gdf_roads = filter_v1_road_classes(gdf_roads_all)
        note = ("V1-parity subset: highway in "
                f"{list(V1_ROAD_HIGHWAY_CLASSES)} (V1 query "
                "way[\"highway\"~\"^(motorway|trunk|primary)$\"])")
    _dist_layer("roads_distance", gdf_roads, roads_geojson,
                n_stored=len(gdf_roads_all), filter_note=note)
    meta["layers"]["roads_distance"]["n_repaired"] = int(gdf_roads_all.attrs.get("n_repaired", 0))

    gdf_veg_all = read_constraints_geojson(vegetation_geojson)
    gdf_veg = filter_v1_vegetation(gdf_veg_all)
    _dist_layer("vegetation_distance", gdf_veg, vegetation_geojson,
                n_stored=len(gdf_veg_all),
                filter_note="V1-parity union: leisure(park|garden|nature_reserve) "
                            "OR landuse(forest|grass|meadow) OR natural(wood|scrub|"
                            "grassland); V2's extra recreation_ground excluded")
    meta["layers"]["vegetation_distance"]["n_repaired"] = int(gdf_veg_all.attrs.get("n_repaired", 0))

    gdf_lu_all = read_constraints_geojson(landuse_geojson)
    gdf_lu = classify_landuse(filter_v1_landuse(gdf_lu_all))
    lu_arr = rasterize_binary(grid, gdf_lu)  # presence (diagnostic)
    coded = np.zeros(grid.shape, dtype=np.uint8)
    if not gdf_lu.empty:
        shapes = [(geom, int(code)) for geom, code in
                  zip(gdf_lu.to_crs(grid.crs).geometry, gdf_lu["code"])]
        coded = rio_features.rasterize(
            shapes, out_shape=grid.shape, transform=grid.transform,
            fill=0, dtype="uint8", all_touched=True,
        )
    lu_path = out_dir / "landuse_raster_30m.tif"
    write_band(lu_path, grid, coded, dtype="uint8", nodata=255, band_name="landuse_class")
    meta["layers"]["landuse_raster"] = {
        "path": str(lu_path), "source": str(landuse_geojson),
        "sha256_source": sha256_file(landuse_geojson) if landuse_geojson.exists() else None,
        "class_mapping": {str(k): v for k, v in LANDUSE_CODE_TO_CLASS.items()},
        "n_features_stored_in_source": int(len(gdf_lu_all)),
        "n_classified_features": int(len(gdf_lu)),
        "feature_filter": "V1-parity: landuse tag in V1's 7 fetched values "
                          f"{list(V1_LANDUSE_VALUES)}",
        "n_repaired": int(gdf_lu_all.attrs.get("n_repaired", 0)),
        "presence_pixels": int(lu_arr.sum()),
    }

    # ---- V2 Phase-2 constraint GeoJSON layers -------------------------------
    gdf_bld = read_constraints_geojson(buildings_geojson)
    _dist_layer("buildings_distance", gdf_bld, buildings_geojson)
    meta["layers"]["buildings_distance"]["n_repaired"] = int(gdf_bld.attrs.get("n_repaired", 0))

    for name, src in (("water_presence", water_geojson),
                      ("road_surfaces_presence", road_surfaces_geojson)):
        gdf = read_constraints_geojson(src)
        arr = rasterize_binary(grid, gdf)
        path = out_dir / f"{name}_30m.tif"
        write_band(path, grid, arr, dtype="uint8", nodata=None, band_name=name)
        meta["layers"][name] = {
            "path": str(path), "source": str(src),
            "n_features_burned": int(len(gdf)),
            "sha256_source": sha256_file(src) if src.exists() else None,
            "presence_pixels": int(arr.sum()),
            "n_repaired": int(gdf.attrs.get("n_repaired", 0)),
        }

    meta["study_area"] = str(study_area)
    meta["roads_full_network_burned"] = bool(roads_full)
    dump_json(meta, out_dir / "static_manifest.json")
    return meta


# ---------------------------------------------------------------------------
# Per-year core products
# ---------------------------------------------------------------------------
def _gap_fill_nearest(arr: np.ndarray) -> np.ndarray:
    """V1 LULC gap-fill: nearest valid label (categorical nearest-fill)."""
    valid = np.isfinite(arr)
    if valid.all():
        return arr
    _, inds = ndimage.distance_transform_edt(~valid, return_indices=True)
    return arr[inds[0], inds[1]]


def _reproject_valid_mask(native_valid: np.ndarray, src_transform, src_crs,
                          grid: Grid) -> np.ndarray:
    """30 m validity = any valid native pixel contributing to the cell."""
    avg = reproject_to_grid(
        native_valid.astype(np.float64), src_transform, src_crs, grid,
        resampling=Resampling.average,
    )
    with np.errstate(invalid="ignore"):
        return avg > 0


def build_year(data_root: Path, year: int, grid: Grid, study: np.ndarray,
               out_dir: Path, lulc_gap_fill: bool) -> dict:
    """Build all per-year core products; returns the per-year manifest."""
    year_dir = out_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    inputs: dict = {}
    stats: dict = {"year": year}

    # ---- Landsat 9 ----------------------------------------------------------
    l9_path = L9_PATH(data_root, year)
    with rasterio.open(l9_path) as ds:
        band = {name: read_band_nan(ds, i + 1) for i, name in enumerate(L9_BANDS)}
    inputs["landsat9"] = {"path": str(l9_path), "sha256": sha256_file(l9_path)}

    valid_l9 = valid_l9_mask(band["ST_B10"], band["SR_B2"], band["QA_CLEAR_COUNT"])
    lst = lst_from_dn(band["ST_B10"])
    l9_ndvi = norm_diff(band["SR_B5"], band["SR_B4"])
    l9_ndbi = norm_diff(band["SR_B6"], band["SR_B5"])
    ndmi = norm_diff(band["SR_B5"], band["SR_B6"])
    mndwi = norm_diff(band["SR_B3"], band["SR_B6"])
    with np.errstate(divide="ignore", invalid="ignore"):
        bsi = ((band["SR_B6"] + band["SR_B4"]) - (band["SR_B5"] + band["SR_B2"])) / \
              ((band["SR_B6"] + band["SR_B4"]) + (band["SR_B5"] + band["SR_B2"]))

    write_band(year_dir / "lst_30m.tif", grid, lst, band_name="LST_C")
    write_band(year_dir / "l9_ndvi_30m.tif", grid, l9_ndvi, band_name="NDVI")
    write_band(year_dir / "l9_ndbi_30m.tif", grid, l9_ndbi, band_name="NDBI")
    write_band(year_dir / "ndmi_30m.tif", grid, ndmi, band_name="NDMI")
    write_band(year_dir / "mndwi_30m.tif", grid, mndwi, band_name="MNDWI")
    write_band(year_dir / "bsi_30m.tif", grid, bsi, band_name="BSI")
    write_band(year_dir / "valid_l9_30m.tif", grid,
               valid_mask_uint8(valid_l9, study), dtype="uint8", nodata=255,
               band_name="valid_l9")
    stats["coverage_l9_st"] = round(float(valid_l9[study].mean()), 6)

    # ---- Sentinel-2 10 m -> NDVI (authoritative) ----------------------------
    s210_path = S2_10M_PATH(data_root, year)
    with rasterio.open(s210_path) as ds:
        s2t, s2crs = ds.transform, ds.crs
        b = {name: read_band_nan(ds, i + 1) for i, name in enumerate(S2_10M_BANDS)}
    inputs["s2_10m"] = {"path": str(s210_path), "sha256": sha256_file(s210_path)}

    valid_s2_10m_native = valid_s2_10m_mask(
        b["B2"], b["B3"], b["B4"], b["B8"], b["VALID_COUNT"])
    # HARD B2>0 INVARIANT (user-approved contract change): native pixels that
    # fail the S2-10m validity rule -- including the B2==0 zero-fill artifact
    # (F3) -- are excluded from the VALUE STACK before aggregation, so the
    # NaN-safe area-weighted mean can never use them in NDVI/PVC or any 30 m
    # aggregate. Not merely a coverage exclusion.
    b4v = np.where(valid_s2_10m_native, b["B4"], np.nan)
    b8v = np.where(valid_s2_10m_native, b["B8"], np.nan)
    ndvi_native = norm_diff(b8v, b4v)
    ndvi_30m = reproject_to_grid(ndvi_native, s2t, s2crs, grid, Resampling.average)
    finite_stack = (np.isfinite(b["B2"]) & np.isfinite(b["B3"])
                    & np.isfinite(b["B4"]) & np.isfinite(b["B8"])
                    & (b["VALID_COUNT"] >= 1))
    b2_zero_native = finite_stack & (b["B2"] == 0)
    stats["s2_10m_b2_zero_guard"] = {
        "finite_px": int(finite_stack.sum()),
        "b2_zero_px": int(b2_zero_native.sum()),
        "valid_excluding_b2_zero": int(valid_s2_10m_native.sum()),
        "identity_holds": bool(int(valid_s2_10m_native.sum())
                               == int(finite_stack.sum()) - int(b2_zero_native.sum())),
    }

    # ---- Sentinel-2 20 m -> NDBI / NDRE (authoritative) ----------------------
    s220_path = S2_20M_PATH(data_root, year)
    with rasterio.open(s220_path) as ds:
        s2t20, s2crs20 = ds.transform, ds.crs
        b20 = {name: read_band_nan(ds, i + 1) for i, name in enumerate(S2_20M_BANDS)}
    inputs["s2_20m"] = {"path": str(s220_path), "sha256": sha256_file(s220_path)}

    valid_s2_20m_native = valid_s2_20m_mask(
        b20["B5"], b20["B6"], b20["B7"], b20["B8A"], b20["B11"], b20["B12"])
    ndbi_native = norm_diff(b20["B11"], b20["B8A"])
    ndre_native = norm_diff(b20["B8A"], b20["B5"])
    ndbi_30m = reproject_to_grid(ndbi_native, s2t20, s2crs20, grid, Resampling.average)
    ndre_30m = reproject_to_grid(ndre_native, s2t20, s2crs20, grid, Resampling.average)

    write_band(year_dir / "ndvi_30m.tif", grid, ndvi_30m, band_name="NDVI")
    write_band(year_dir / "ndbi_30m.tif", grid, ndbi_30m, band_name="NDBI")
    write_band(year_dir / "ndre_30m.tif", grid, ndre_30m, band_name="NDRE")
    valid_s2_10m_30m = _reproject_valid_mask(valid_s2_10m_native, s2t, s2crs, grid)
    valid_s2_20m_30m = _reproject_valid_mask(valid_s2_20m_native, s2t20, s2crs20, grid)
    write_band(year_dir / "valid_s2_10m_30m.tif", grid,
               valid_mask_uint8(valid_s2_10m_30m, study), dtype="uint8", nodata=255,
               band_name="valid_s2_10m")
    write_band(year_dir / "valid_s2_20m_30m.tif", grid,
               valid_mask_uint8(valid_s2_20m_30m, study), dtype="uint8", nodata=255,
               band_name="valid_s2_20m")
    stats["coverage_s2_10m_30m_any"] = round(float(valid_s2_10m_30m[study].mean()), 6)
    stats["coverage_s2_20m_30m_any"] = round(float(valid_s2_20m_30m[study].mean()), 6)

    # ---- Vegetation cover ----------------------------------------------------
    pvc = vegetation_cover_from_ndvi(ndvi_30m)
    write_band(year_dir / "vegetation_cover_30m.tif", grid, pvc, band_name="PVC")
    finite = pvc[np.isfinite(pvc)]
    stats["pvc"] = {"min": float(finite.min()), "max": float(finite.max()),
                    "mean": float(finite.mean())} if finite.size else None

    # ---- LULC ----------------------------------------------------------------
    lulc_path = LULC_PATH(data_root, year)
    with rasterio.open(lulc_path) as ds:
        lulc_t, lulc_crs = ds.transform, ds.crs
        lulc_native = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
    inputs["lulc"] = {"path": str(lulc_path), "sha256": sha256_file(lulc_path)}

    valid_lulc_native = valid_lulc_mask(lulc_native)
    n_valid_native = int(valid_lulc_native.sum())
    lulc_src = _gap_fill_nearest(lulc_native) if lulc_gap_fill else lulc_native
    lulc_30m = reproject_to_grid(lulc_src, lulc_t, lulc_crs, grid, Resampling.nearest)
    lulc_30m = lulc_30m.astype(np.float32)
    write_band(year_dir / "lulc_30m.tif", grid, lulc_30m, dtype="float32",
               nodata=np.nan, band_name="LULC")
    valid_lulc_30m = _reproject_valid_mask(valid_lulc_native, lulc_t, lulc_crs, grid)
    write_band(year_dir / "valid_lulc_30m.tif", grid,
               valid_mask_uint8(valid_lulc_30m, study), dtype="uint8", nodata=255,
               band_name="valid_lulc")
    stats["coverage_lulc_30m_any"] = round(float(valid_lulc_30m[study].mean()), 6)
    stats["lulc_gap_fill"] = bool(lulc_gap_fill)
    stats["lulc_valid_10m_px"] = n_valid_native

    # ---- Combined domain mask -------------------------------------------------
    domain = (valid_l9 & valid_s2_10m_30m & valid_s2_20m_30m & valid_lulc_30m
              & study & np.isfinite(ndvi_30m) & np.isfinite(ndbi_30m)
              & np.isfinite(ndre_30m) & np.isfinite(pvc) & np.isfinite(lst))
    write_band(year_dir / "domain_30m.tif", grid,
               valid_mask_uint8(domain, study), dtype="uint8", nodata=255,
               band_name="domain")
    stats["domain_pixels"] = int(domain.sum())
    stats["study_pixels"] = int(study.sum())

    manifest = {"year": year, "inputs": inputs, "stats": stats,
                "outputs": sorted(p.name for p in year_dir.glob("*.tif")),
                "window": f"W4 {WINDOW_TAG}; met hours UTC {MET_WINDOW_HOURS_UTC}"}
    dump_json(manifest, year_dir / "build_manifest.json")
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(DATA_RAW_V2))
    p.add_argument("--out-root", default=str(PROJECT_ROOT / "data" / "phase3"))
    p.add_argument("--years", default=",".join(str(y) for y in YEARS))
    p.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    p.add_argument("--study-area", default=str(STUDY_AREA_DEFAULT))
    p.add_argument("--roads-geojson", default=str(V2_ROADS_GEOJSON),
                   help="V2 full road network GeoJSON; features burn the "
                        "V1-parity subset unless --roads-full")
    p.add_argument("--roads-full", action="store_true",
                   help="burn the FULL road network instead of the V1-parity "
                        "major-road subset (changes dist_road_m semantics "
                        "relative to the frozen V1 model)")
    p.add_argument("--vegetation-geojson", default=str(V2_VEGETATION_GEOJSON))
    p.add_argument("--landuse-geojson", default=str(V2_LANDUSE_GEOJSON))
    p.add_argument("--buildings-geojson", default=str(V2_BUILDINGS_GEOJSON))
    p.add_argument("--water-geojson", default=str(V2_WATER_GEOJSON))
    p.add_argument("--road-surfaces-geojson", default=str(V2_ROAD_SURFACES_GEOJSON))
    p.add_argument("--lulc-gap-fill", action="store_true",
                   help="V1-style nearest-label gap-fill of the 10 m LULC layer "
                        "before aggregation (OFF by default in V2; the Phase-2 "
                        "coverage guard makes it unnecessary)")
    p.add_argument("--skip-vectors", action="store_true",
                   help="skip static constraint layers (already built)")
    args = p.parse_args(argv)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    years = tuple(int(y) for y in args.years.split(","))
    grid_file = Path(args.grid_file)

    if not grid_file.exists():
        print(f"[BUILD] ERROR: authoritative grid file not found: {grid_file}", file=sys.stderr)
        return 1
    grid = Grid.from_file(grid_file)
    print(f"[BUILD] authoritative grid {grid.height} x {grid.width} from {grid_file}")

    study = load_study_mask(grid, Path(args.study_area))
    print(f"[BUILD] study-area cells: {int(study.sum())} / {study.size}")

    if not args.skip_vectors:
        print("[BUILD] rasterizing static constraint layers ...")
        build_static_layers(
            grid, out_root / "static", Path(args.study_area),
            Path(args.roads_geojson), Path(args.vegetation_geojson),
            Path(args.landuse_geojson),
            Path(args.buildings_geojson), Path(args.water_geojson),
            Path(args.road_surfaces_geojson),
            roads_full=args.roads_full,
        )
    else:
        print("[BUILD] --skip-vectors: reusing existing static layers")

    summaries = []
    for year in years:
        print(f"[BUILD] === year {year} ===", flush=True)
        summaries.append(build_year(data_root, year, grid, study, out_root,
                                    lulc_gap_fill=args.lulc_gap_fill))

    dump_json({"years": [s["year"] for s in summaries], "summaries": summaries},
              out_root / "build_summary.json")
    print("[BUILD] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
