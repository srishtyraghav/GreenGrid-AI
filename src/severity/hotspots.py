"""Phase 6 Stage 2: heat-severity hotspot delineation.

Builds connected hotspot clusters from the Stage 1 full-grid severity
rasters under three predefined, documented definitions (A/B/C — frozen
operational choices, not tuned to make maps look good), removes noise
clusters smaller than 10 pixels (8-connectivity), and derives per-hotspot
polygons plus a full attribute table.

Scientific framing: these are ML-based RELATIVE heat-severity hotspots —
an operational UHI hotspot proxy, NOT physical UHI intensity.  NoData
cells never participate in hotspot membership and are never interpreted
as "not a hotspot"; only valid cells can become hotspot members.

Area accounting is done by reprojecting polygons to EPSG:32643 (UTM
zone 43N, Delhi) because degree coordinates are NOT equal-area; hectares
must never be derived from degree widths.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as raster_shapes
from scipy import ndimage
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import (
    BUILT_MAX_NDVI,
    BUILT_NDBI_THRESHOLD,
    CONNECTIVITY_STRUCTURE,
    GREEN_NDVI_THRESHOLD,
    HOTSPOT_DEFINITIONS,
    HOTSPOT_NODATA,
    LANDUSE_CLASS_NAMES,
    LANDUSE_NODATA,
    LANDUSE_RASTER_PATH,
    MIN_HOTSPOT_PIXELS,
    NDBI_RASTERS,
    NDVI_RASTERS,
    POLYGON_AREA_TOLERANCE,
    PROJECT_ROOT,
    RANDOM_SEED,
    SEVERITY_CLASS_NAMES,
    SEVERITY_NODATA,
    UTM_CRS,
    VEGETATION_COVER_RASTERS,
    WGS84_CRS,
    classify_green_built,
    confidence_raster_path,
    hotspot_boundaries_path,
    hotspot_mask_path,
    hotspot_statistics_path,
    lst_raster_path,
    severity_raster_path,
    severity_score_raster_path,
)
from .fullgrid import load_valid_cells

MASK_STRUCTURE = np.ones((3, 3), dtype=int)  # 8-connectivity

LANDUSE_CLASS_CODES = sorted(LANDUSE_CLASS_NAMES)


# ---------------------------------------------------------------------------
# Raster loading
# ---------------------------------------------------------------------------
def load_year_rasters(year: int) -> Dict[str, np.ndarray]:
    """Load every full-grid raster needed for hotspot attribution in a year.

    All arrays are on the common 1768x1874 EPSG:4326 grid; per-cell nodata
    (NaN for float sources, 255 for landuse, -1 for Phase 6 rasters) is
    preserved so cluster-attribute aggregations can mask it explicitly.
    """
    out: Dict[str, np.ndarray] = {}
    with rasterio.open(severity_raster_path(year)) as src:
        out["severity"] = src.read(1)
    with rasterio.open(confidence_raster_path(year)) as src:
        out["confidence"] = src.read(1)
    with rasterio.open(severity_score_raster_path(year)) as src:
        out["severity_score"] = src.read(1)
    with rasterio.open(lst_raster_path(year)) as src:
        out["lst"] = src.read(1)
    for key, mapping in (
        ("ndvi", NDVI_RASTERS),
        ("ndbi", NDBI_RASTERS),
        ("vegetation_cover", VEGETATION_COVER_RASTERS),
    ):
        with rasterio.open(mapping[year]) as src:
            out[key] = src.read(1)
    with rasterio.open(LANDUSE_RASTER_PATH) as src:
        out["landuse"] = src.read(1)
    return out


# ---------------------------------------------------------------------------
# Mask building and cluster labelling
# ---------------------------------------------------------------------------
def build_hotspot_mask(
    severity: np.ndarray,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    definition: str,
) -> np.ndarray:
    """Return the binary hotspot membership mask for one definition/year.

    Membership is evaluated ONLY at valid cells; NoData is never
    interpreted as "not a hotspot" — it simply does not participate.
    """
    spec = HOTSPOT_DEFINITIONS[definition]
    member = np.isin(severity, spec["severity_values"])
    if spec["min_confidence"] is not None:
        # Definition C operational confidence criterion.  Confidence is an
        # UNCALIBRATED max-class-probability proxy (Phase 5 calibration
        # report); the 0.60 threshold is operational, not probabilistic.
        member &= confidence >= spec["min_confidence"]
    return member & valid_mask


def label_hotspot_clusters(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label 8-connected clusters and drop clusters below the min-size rule.

    Returns
    -------
    tuple
        (labelled int32 raster with dense IDs 1..n_kept over valid cells,
        n_kept).  Removed clusters are an operational noise-removal rule
        (MIN_HOTSPOT_PIXELS, ~0.8 ha at 30 m): isolated small clusters are
        discarded, coherent clusters retained.
    """
    labels, n_found = ndimage.label(binary_mask, structure=MASK_STRUCTURE)
    if n_found == 0:
        return labels.astype(np.int32), 0

    sizes = np.bincount(labels.ravel())
    keep = sizes >= MIN_HOTSPOT_PIXELS
    keep[0] = False  # background is never a cluster
    remap = np.zeros(n_found + 1, dtype=np.int32)
    remap[np.where(keep)[0]] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    labelled = remap[labels]
    return labelled.astype(np.int32), int(keep.sum())


def write_hotspot_mask_raster(
    labelled: np.ndarray,
    valid_mask: np.ndarray,
    profile: Dict,
    output_path: Path,
) -> Dict:
    """Write the int32 hotspot-ID raster (0 = valid non-hotspot, -1 nodata)."""
    raster = np.full(labelled.shape, HOTSPOT_NODATA, dtype=np.int32)
    raster[valid_mask] = labelled[valid_mask]

    out_profile = profile.copy()
    out_profile.update(
        {
            "dtype": "int32",
            "count": 1,
            "nodata": HOTSPOT_NODATA,
            "compress": "lzw",
        }
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(raster, 1)

    return {
        "output_path": str(output_path),
        "n_hotspot_pixels": int((raster > 0).sum()),
        "n_hotspots": int(raster.max()),
        "nodata": HOTSPOT_NODATA,
    }


# ---------------------------------------------------------------------------
# Cluster -> polygon boundaries
# ---------------------------------------------------------------------------
def polygons_per_cluster(
    labelled: np.ndarray,
    transform,
) -> Dict[int, BaseGeometry]:
    """Dissolve pixel polygons into one boundary polygon per hotspot ID.

    ``raster_shapes`` is called with 4-connectivity so that cells touching
    only at a corner are emitted as separate rings; corner-pinched single
    rings are NOT valid OGC polygons, so dissolving per ID (8-connected
    cluster semantics are already encoded in the IDs) yields exact,
    valid Polygon/MultiPolygon boundaries.  All geometries are in
    EPSG:4326.  No silent buffer-fixing: validity is checked and reported
    by the caller.
    """
    mask = labelled > 0
    per_id: Dict[int, List[BaseGeometry]] = {}
    for geom, value in raster_shapes(
        labelled,
        mask=mask,
        transform=transform,
        connectivity=4,
    ):
        per_id.setdefault(int(value), []).append(shapely_shape(geom))
    return {hid: unary_union(geoms) for hid, geoms in per_id.items()}


def validate_geometries(polygons: Dict[int, BaseGeometry]) -> Dict:
    """Check boundary polygons for invalid or zero-area geometries.

    Geometries are NOT silently buffer(0)-fixed; problems are reported so
    downstream consumers know the boundaries are exact pixel unions.
    """
    invalid_ids = [hid for hid, g in polygons.items() if not g.is_valid]
    zero_area_ids = [hid for hid, g in polygons.items() if g.area <= 0.0]
    return {
        "n_polygons": len(polygons),
        "n_invalid": len(invalid_ids),
        "invalid_ids": invalid_ids,
        "n_zero_area": len(zero_area_ids),
        "zero_area_ids": zero_area_ids,
        "all_valid": len(invalid_ids) == 0 and len(zero_area_ids) == 0,
    }


# ---------------------------------------------------------------------------
# Per-hotspot attribute table
# ---------------------------------------------------------------------------
def compute_hotspot_attributes(
    labelled: np.ndarray,
    rasters: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
    polygons_4326: Dict[int, BaseGeometry],
    year: int,
    definition: str,
    mean_pixel_area_m2: float,
) -> Tuple[pd.DataFrame, Dict]:
    """Compute the per-hotspot attribute table and validation summaries.

    Returns
    -------
    tuple
        (attribute DataFrame, one row per hotspot; validation dict with
        geometry checks and pixel/polygon area-consistency results).
    """
    geom_report = validate_geometries(polygons_4326)

    # Duplicate-ID guard: cluster IDs are dense 1..n by construction.
    ids = sorted(int(i) for i in np.unique(labelled[labelled > 0]))
    duplicate_ids = len(ids) != len(set(ids))

    # Equal-area areas: reproject to UTM 43N.  Degree coordinates are NOT
    # equal-area, so hectares are never derived from degree widths.
    gdf = gpd.GeoDataFrame(
        {"hotspot_id": ids},
        geometry=[polygons_4326[i] for i in ids],
        crs=WGS84_CRS,
    )
    gdf_utm = gdf.to_crs(UTM_CRS)

    rows: List[Dict] = []
    for idx, hid in enumerate(ids):
        cluster = (labelled == hid) & valid_mask
        n_px = int(cluster.sum())
        if n_px == 0:
            raise AssertionError(f"Hotspot {hid} has no pixels")

        geom_4326 = polygons_4326[hid]
        geom_utm = gdf_utm.geometry.iloc[idx]
        area_m2 = float(geom_utm.area)

        cx, cy = geom_4326.centroid.x, geom_4326.centroid.y
        cx_u, cy_u = geom_utm.centroid.x, geom_utm.centroid.y
        minx, miny, maxx, maxy = geom_4326.bounds

        severity_vals = rasters["severity"][cluster]
        conf_vals = rasters["confidence"][cluster]
        lst_vals = rasters["lst"][cluster].astype(np.float64)
        lst_vals = lst_vals[np.isfinite(lst_vals) & (lst_vals != -1.0)]
        ndvi_vals = rasters["ndvi"][cluster]
        ndvi_vals = ndvi_vals[np.isfinite(ndvi_vals)]
        ndbi_vals = rasters["ndbi"][cluster]
        ndbi_vals = ndbi_vals[np.isfinite(ndbi_vals)]
        veg_vals = rasters["vegetation_cover"][cluster]
        veg_vals = veg_vals[np.isfinite(veg_vals)]
        score_vals = rasters["severity_score"][cluster]
        score_vals = score_vals[np.isfinite(score_vals) & (score_vals != -1.0)]
        lu_vals = rasters["landuse"][cluster]
        lu_vals = lu_vals[lu_vals != LANDUSE_NODATA]

        counts = np.bincount(severity_vals.astype(int), minlength=4)
        sev_frac = counts / n_px

        # Operational green/built classification of cluster cells.
        gb = classify_green_built(ndvi_vals, ndbi_vals)
        green_frac = float((gb == "green_dominant").mean())
        built_frac = float((gb == "built_dominant").mean())

        # Landuse composition over classified cells only (nodata 255 falls
        # out of the denominator).  Operational mode, not site-calibrated.
        lu_unique, lu_counts = np.unique(lu_vals, return_counts=True) if lu_vals.size else (np.array([]), np.array([]))
        lu_frac_total = float(lu_counts.sum()) if lu_vals.size else 0.0
        lu_comp = {}
        if lu_frac_total > 0:
            order = np.argsort(-lu_counts)
            for rank, o in enumerate(order[:3], start=1):
                code = int(lu_unique[o])
                lu_comp[f"landuse_top{rank}_code"] = code
                lu_comp[f"landuse_top{rank}_name"] = LANDUSE_CLASS_NAMES.get(code, "unknown")
                lu_comp[f"landuse_top{rank}_fraction"] = round(float(lu_counts[o]) / lu_frac_total, 6)
        dominant_code = int(lu_unique[np.argmax(lu_counts)]) if lu_vals.size else 0

        row: Dict = {
            "hotspot_id": hid,
            "year": year,
            "definition": definition,
            "definition_spec": (
                f"severity in {HOTSPOT_DEFINITIONS[definition]['severity_values']}"
                + (
                    f" AND confidence >= {HOTSPOT_DEFINITIONS[definition]['min_confidence']}"
                    if HOTSPOT_DEFINITIONS[definition]["min_confidence"] is not None
                    else ""
                )
            ),
            "pixel_count": n_px,
            "area_m2": round(area_m2, 1),
            "area_ha": round(area_m2 / 1e4, 3),
            "centroid_lon": round(cx, 7),
            "centroid_lat": round(cy, 7),
            "centroid_utm_easting_m": round(cx_u, 1),
            "centroid_utm_northing_m": round(cy_u, 1),
            "bbox_min_lon": round(minx, 7),
            "bbox_min_lat": round(miny, 7),
            "bbox_max_lon": round(maxx, 7),
            "bbox_max_lat": round(maxy, 7),
            "frac_low": round(float(sev_frac[0]), 6),
            "frac_moderate": round(float(sev_frac[1]), 6),
            "frac_high": round(float(sev_frac[2]), 6),
            "frac_severe": round(float(sev_frac[3]), 6),
            "high_fraction": round(float(sev_frac[2]), 6),
            "severe_fraction": round(float(sev_frac[3]), 6),
            "high_severe_fraction": round(float(sev_frac[2] + sev_frac[3]), 6),
            "high_severe_area_ha": round(float(sev_frac[2] + sev_frac[3]) * area_m2 / 1e4, 3),
            "mean_lst_C": round(float(lst_vals.mean()), 3) if lst_vals.size else np.nan,
            "median_lst_C": round(float(np.median(lst_vals)), 3) if lst_vals.size else np.nan,
            "max_lst_C": round(float(lst_vals.max()), 3) if lst_vals.size else np.nan,
            "mean_ndvi": round(float(ndvi_vals.mean()), 6) if ndvi_vals.size else np.nan,
            "mean_ndbi": round(float(ndbi_vals.mean()), 6) if ndbi_vals.size else np.nan,
            "mean_vegetation_cover": round(float(veg_vals.mean()), 6) if veg_vals.size else np.nan,
            "mean_confidence": round(float(conf_vals.mean()), 6),
            "mean_severity_score": round(float(score_vals.mean()), 6) if score_vals.size else np.nan,
            "dominant_landuse_code": dominant_code,
            "dominant_landuse_name": LANDUSE_CLASS_NAMES.get(dominant_code, "unknown"),
            "green_fraction": green_frac,
            "built_fraction": built_frac,
            "green_ndvi_threshold": GREEN_NDVI_THRESHOLD,
            "built_ndbi_threshold": BUILT_NDBI_THRESHOLD,
            "built_max_ndvi": BUILT_MAX_NDVI,
        }
        row.update(lu_comp)
        rows.append(row)

    df = pd.DataFrame(rows)
    for rank in (1, 2, 3):
        col = f"landuse_top{rank}_code"
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    if df.empty:
        return df, {**geom_report, "duplicate_ids": duplicate_ids}

    # Ranking per year/definition.  Primary rank: high_severe_area (the
    # hotspot's own High+Severe area).  No arbitrary composite score.
    df["rank_by_high_severe_area"] = (
        df["high_severe_area_ha"].rank(ascending=False, method="min").astype("Int64")
    )
    df["rank_by_mean_lst"] = df["mean_lst_C"].rank(ascending=False, method="min").astype("Int64")
    df["rank_by_area"] = df["area_ha"].rank(ascending=False, method="min").astype("Int64")

    # Area consistency: polygon area (UTM) vs pixel_count * mean pixel area.
    rel_errs = np.abs(df["area_m2"].values / (df["pixel_count"].values * mean_pixel_area_m2) - 1.0)
    consistency = {
        "sum_cluster_pixel_counts": int(df["pixel_count"].sum()),
        "hotspot_mask_pixel_count": int((labelled > 0).sum()),
        "pixel_counts_match": bool(int(df["pixel_count"].sum()) == int((labelled > 0).sum())),
        "mean_pixel_area_m2": round(mean_pixel_area_m2, 2),
        "max_polygon_area_rel_err": round(float(np.max(rel_errs)), 6),
        "polygon_area_within_tolerance": bool(np.max(rel_errs) <= POLYGON_AREA_TOLERANCE),
        "duplicate_ids": duplicate_ids,
    }
    return df, {**geom_report, **consistency}


def _mean_pixel_area_m2(transform, height: int, width: int) -> float:
    """Mean per-pixel area (m2) in UTM 43N over the full grid extent."""
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    ring = [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
        (left, top),
    ]
    from shapely.geometry import Polygon

    poly = Polygon(ring)
    gdf = gpd.GeoDataFrame(geometry=[poly], crs=WGS84_CRS).to_crs(UTM_CRS)
    return float(gdf.geometry.iloc[0].area) / float(height * width)


# ---------------------------------------------------------------------------
# Per year/definition driver
# ---------------------------------------------------------------------------
def delineate_hotspots(
    year: int,
    definition: str,
    rasters: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
    transform,
    mask_profile: Dict,
) -> Dict:
    """Delineate hotspots for one year/definition and write all outputs."""
    print(f"  [{year} def {definition}] Building hotspot mask ...")
    binary = build_hotspot_mask(
        rasters["severity"], rasters["confidence"], valid_mask, definition
    )
    labelled, n_kept = label_hotspot_clusters(binary)
    print(f"  [{year} def {definition}] Clusters kept: {n_kept} "
          f"(raw members {int(binary.sum()):,} px)")

    mask_meta = write_hotspot_mask_raster(
        labelled, valid_mask, mask_profile, hotspot_mask_path(definition, year)
    )

    print(f"  [{year} def {definition}] Building polygons ...")
    polygons = polygons_per_cluster(labelled, transform)
    mean_px_area = _mean_pixel_area_m2(transform, labelled.shape[0], labelled.shape[1])

    print(f"  [{year} def {definition}] Computing attributes ...")
    stats_df, validation = compute_hotspot_attributes(
        labelled, rasters, valid_mask, polygons, year, definition, mean_px_area
    )

    # Boundary GeoJSON (EPSG:4326), one feature per hotspot with its stats.
    if not stats_df.empty:
        gdf = gpd.GeoDataFrame(
            stats_df,
            geometry=[polygons[i] for i in stats_df["hotspot_id"]],
            crs=WGS84_CRS,
        )
    else:
        gdf = gpd.GeoDataFrame(
            {"hotspot_id": pd.Series(dtype="int64"), "year": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries(dtype="geometry"),
            crs=WGS84_CRS,
        )
    gdf.to_file(hotspot_boundaries_path(definition, year), driver="GeoJSON")

    stats_csv = hotspot_statistics_path(definition, year)
    stats_df.to_csv(stats_csv, index=False)

    valid_px = int(valid_mask.sum())
    hotspot_px = int((labelled > 0).sum())
    summary = {
        "year": year,
        "definition": definition,
        "n_hotspots": n_kept,
        "hotspot_pixels": hotspot_px,
        "valid_pixels": valid_px,
        "total_area_ha": round(float(stats_df["area_ha"].sum()), 3) if not stats_df.empty else 0.0,
        "pct_of_valid_area": round(100.0 * hotspot_px / valid_px, 4),
        "min_area_ha": round(float(stats_df["area_ha"].min()), 3) if not stats_df.empty else 0.0,
        "median_area_ha": round(float(stats_df["area_ha"].median()), 3) if not stats_df.empty else 0.0,
        "max_area_ha": round(float(stats_df["area_ha"].max()), 3) if not stats_df.empty else 0.0,
    }

    return {
        "binary_mask": binary,
        "labelled": labelled,
        "stats_df": stats_df,
        "polygons": polygons,
        "validation": validation,
        "summary": summary,
        "outputs": {
            "mask_raster": mask_meta["output_path"],
            "boundaries": str(hotspot_boundaries_path(definition, year)),
            "statistics_csv": str(stats_csv),
        },
    }


def load_grid_context() -> Tuple[np.ndarray, np.ndarray, object, Dict, np.ndarray]:
    """Load valid-cell indices and the reference grid profile/transform."""
    rows, cols, transform, mask_profile = load_valid_cells()
    valid_mask = np.zeros((mask_profile["height"], mask_profile["width"]), dtype=bool)
    valid_mask[rows, cols] = True
    return rows, cols, transform, mask_profile, valid_mask


__all__ = [
    "load_year_rasters",
    "build_hotspot_mask",
    "label_hotspot_clusters",
    "write_hotspot_mask_raster",
    "polygons_per_cluster",
    "validate_geometries",
    "compute_hotspot_attributes",
    "delineate_hotspots",
    "load_grid_context",
]


# ---------------------------------------------------------------------------
# Stage 2 pipeline entry point
# ---------------------------------------------------------------------------
def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _convert_for_json(obj: object) -> object:
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert_for_json(obj), f, indent=2, default=str)


def _update_manifest(record: Dict) -> Dict:
    """Read-modify-write the Phase 6 manifest with hotspot metadata."""
    from .config import MANIFEST_JSON

    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)

    manifest["hotspots"] = {
        "stage": 2,
        "generated_utc": _now(),
        "framing": (
            "ML-based RELATIVE heat-severity hotspots; operational UHI "
            "hotspot proxy, NOT physical UHI intensity. NoData cells never "
            "participate and are never interpreted as 'not a hotspot'."
        ),
        "definitions": {
            k: {
                "description": {
                    "A": "severity in {High=2, Severe=3} (primary definition)",
                    "B": "severity == Severe=3 only",
                    "C": "severity in {2,3} AND confidence >= 0.60 (operational "
                         "confidence criterion; probabilities are uncalibrated "
                         "confidence proxies per the Phase 5 calibration report)",
                }[k],
                **v,
            }
            for k, v in HOTSPOT_DEFINITIONS.items()
        },
        "min_hotspot_pixels": MIN_HOTSPOT_PIXELS,
        "min_area_rule": (
            "clusters smaller than 10 connected pixels (8-connectivity, "
            "~0.8 ha at 30 m) removed as operational noise"
        ),
        "connectivity": CONNECTIVITY_STRUCTURE,
        "area_crs": UTM_CRS,
        "area_note": (
            "areas computed by reprojecting polygons to EPSG:32643 (UTM 43N, "
            "Delhi); degrees are NOT equal-area"
        ),
        "green_built_classification": {
            "green_dominant": f"NDVI >= {GREEN_NDVI_THRESHOLD}",
            "built_dominant": f"NDBI >= {BUILT_NDBI_THRESHOLD} AND NDVI < {BUILT_MAX_NDVI}",
            "else": "other_mixed",
            "note": (
                "conventional remote-sensing thresholds, operational not "
                "site-calibrated; landuse_class reported separately"
            ),
        },
        "ranking": {
            "primary": "rank_by_high_severe_area",
            "also_provided": ["rank_by_mean_lst", "rank_by_area"],
            "note": "no arbitrary composite score",
        },
        "outputs": record["outputs"],
        "validation": record["validation"],
    }
    _save_json(manifest, MANIFEST_JSON)
    return manifest


def _append_pipeline_record(record: Dict) -> None:
    """Append the Stage 2 entry to the Phase 6 pipeline record.

    The existing Stage 1 record is preserved verbatim inside the
    ``stage_records`` list (read-modify-write).
    """
    from .config import PIPELINE_RECORD_JSON

    with open(PIPELINE_RECORD_JSON) as f:
        existing = json.load(f)

    container = {"phase": 6, "project": "GreenGrid-AI", "stage_records": []}
    if "stage_records" in existing:
        container["stage_records"] = existing["stage_records"]
    else:
        # First append: wrap the legacy single-stage record unmodified.
        container["stage_records"] = [existing]
    # Idempotent re-runs replace the previous Stage 2 entry.
    container["stage_records"] = [r for r in container["stage_records"] if r.get("stage") != 2]
    container["stage_records"].append(record)
    _save_json(container, PIPELINE_RECORD_JSON)


def run_stage2() -> Dict:
    """Run the complete Phase 6 Stage 2 hotspot-delineation pipeline."""
    import platform

    from .config import HOTSPOT_SENSITIVITY_CSV, MANIFEST_JSON, PIPELINE_RECORD_JSON

    record: Dict = {
        "phase": 6,
        "stage": 2,
        "project": "GreenGrid-AI",
        "project_root": str(PROJECT_ROOT),
        "started_utc": _now(),
        "random_seed": RANDOM_SEED,
        "steps": {},
    }

    # ------------------------------------------------------------------
    # 1. Grid context and per-year source rasters
    # ------------------------------------------------------------------
    _, _, transform, mask_profile, valid_mask = load_grid_context()
    n_valid = int(valid_mask.sum())
    print(f"  Valid cells per year: {n_valid:,} (grid {mask_profile['height']}x{mask_profile['width']})")

    masks: Dict[int, Dict[str, np.ndarray]] = {}
    summaries: Dict[int, Dict[str, Dict]] = {}
    validations: Dict[int, Dict[str, Dict]] = {}
    outputs: Dict[str, Dict] = {}

    for year in (2022, 2026):
        rasters = load_year_rasters(year)
        # NoData never participates: membership masks are restricted to the
        # valid grid (build_hotspot_mask ANDs with the valid mask).
        nodata_ok = bool(
            ((rasters["severity"] != SEVERITY_NODATA) == valid_mask).all()
        )
        print(f"  [{year}] severity/valid-mask nodata alignment: {nodata_ok}")
        masks[year] = {}
        summaries[year] = {}
        validations[year] = {}
        outputs[str(year)] = {}
        for definition in ("A", "B", "C"):
            result = delineate_hotspots(
                year, definition, rasters, valid_mask, transform, mask_profile
            )
            masks[year][definition] = result["labelled"] > 0  # final, post min-size rule
            summaries[year][definition] = result["summary"]
            validations[year][definition] = result["validation"]
            outputs[str(year)][definition] = result["outputs"]
            print(
                f"  [{year} def {definition}] n={result['summary']['n_hotspots']}, "
                f"area={result['summary']['total_area_ha']} ha "
                f"({result['summary']['pct_of_valid_area']}% of valid), "
                f"geom valid={result['validation']['all_valid']}"
            )

    record["steps"]["delineation"] = {
        "status": "success",
        "valid_cells_per_year": n_valid,
        "summaries": summaries,
        "validations": validations,
    }

    # ------------------------------------------------------------------
    # 2. Threshold-sensitivity comparison
    # ------------------------------------------------------------------
    from .hotspot_stats import build_sensitivity_table, write_sensitivity_csv

    sensitivity_df = build_sensitivity_table(
        masks, {y: n_valid for y in (2022, 2026)}, summaries
    )
    sensitivity_path = write_sensitivity_csv(sensitivity_df, HOTSPOT_SENSITIVITY_CSV)
    print(f"  Sensitivity table: {sensitivity_path}")
    record["steps"]["sensitivity"] = {
        "status": "success",
        "output_path": sensitivity_path,
    }

    # ------------------------------------------------------------------
    # 3. NoData-handling confirmation
    # ------------------------------------------------------------------
    nodata_checks = {}
    for year in (2022, 2026):
        with rasterio.open(hotspot_mask_path("A", year)) as src:
            hm = src.read(1)
        nodata_checks[year] = {
            "mask_is_nodata_where_severity_is_nodata": bool(
                (hm[valid_mask] != HOTSPOT_NODATA).all()
                and (hm[~valid_mask] == HOTSPOT_NODATA).all()
            ),
        }
    record["steps"]["nodata_handling"] = {"status": "success", **nodata_checks}
    print(f"  NoData handling: {nodata_checks}")

    # ------------------------------------------------------------------
    # 4. Manifest + pipeline record
    # ------------------------------------------------------------------
    record["outputs"] = outputs
    record["validation"] = {"per_year_definition": validations}
    record["finished_utc"] = _now()
    record["status"] = "success"
    record["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
        "scipy": ndimage.__name__ and __import__("scipy").__version__,
        "shapely": __import__("shapely").__version__,
        "geopandas": gpd.__version__,
    }
    _update_manifest(record)
    _append_pipeline_record(record)
    print(f"  Manifest updated: {MANIFEST_JSON}")
    print(f"  Pipeline record appended: {PIPELINE_RECORD_JSON}")

    return {"record": record, "summaries": summaries, "sensitivity": sensitivity_df}


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for the Phase 6 Stage 2 hotspot delineation."""
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 6 Stage 2 hotspot delineation."
    )
    parser.parse_args(argv)

    try:
        results = run_stage2()
        print(json.dumps(_convert_for_json(results["record"]), indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Phase 6 Stage 2 pipeline failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
