"""Phase 7 Stage 2: priority-zone extraction and zone-level analytics.

Extracts connected priority zones from the Stage 1 suitability classes
(FROZEN rule, design spec section i): 8-connected components computed per
year on ``class >= 3 (High)`` and separately on ``class == 4 (Very High)``,
with clusters below ``MIN_ZONE_PIXELS = 10`` (~0.8 ha at 30 m) removed as
operational noise — the same rule Phase 6 used for hotspots.  Zone
boundaries are dissolved pixel unions (``rasterio.features.shapes`` with
connectivity=4 + per-ID dissolve, the Phase 6-validated pattern) in
EPSG:4326; areas are computed by reprojecting to EPSG:32643 (UTM 43N) —
degrees are never used for area.

Stage 2 products:

1. Priority-zone GeoJSONs (baseline A High / Very High / combined, plus
   scenarios B/C) with full attribute tables and ``priority_confidence``
   (spec section j: 100 x Phase 6 confidence x scenario agreement rate —
   an operational stability index, NOT a probability trees will grow).
2. ``priority_zone_statistics.csv`` and ``priority_ranking.csv``.
3. "Why here?" explanations (spec section 26) — CSV + worked-examples MD.
4. Spatial sensitivity extension of ``sensitivity_analysis.csv`` plus a
   plain-language ``sensitivity_findings.md``.
5. Temporal persistence products (spec sections 14/15) — transition table
   plus persistent/emerging/declining zone vectors.
6. Area reconciliation (class areas vs valid area; zone pixel counts vs
   mask counts; polygon validation counts — reported, never silently
   repaired).

The Stage 1 baseline is deliberately very sparse (gated product: High
582 px in 2022, 124 px in 2026, Very High 0 both years).  Few or zero
baseline zones is the REAL result of the frozen formulation, not a bug;
scenarios B/C (which relax the gating) are zoned as well so the
sensitivity discussion has spatial substance.

Run with::

    PYTHONPATH=src .venv/bin/python -m suitability.zones
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
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
    CLASS_LABELS,
    HIGH_VERY_HIGH_CLASSES,
    LANDUSE_CLASS_NAMES,
    LANDUSE_ELIGIBILITY,
    MANIFEST_JSON,
    MIN_ZONE_PIXELS,
    PHASE7_REPORTS_DIR,
    PHASE7_TABLES_DIR,
    PHASE7_VECTORS_DIR,
    PIPELINE_RECORD_JSON,
    PROJECT_ROOT,
    RANDOM_SEED,
    TABLE_PATHS,
    UTM_CRS,
    WGS84_CRS,
    YEARS,
)
from .statistics import (
    ALL_SCENARIOS,
    PRIORITY_CONFIDENCE_MEANING,
    SENSITIVITY_FINDINGS_MD,
    SPATIAL_SCENARIOS,
    TEMPORAL_TRANSITION_CSV,
    ENVIRONMENTAL_PRIORITY_CSV,
    GREEN_BUILT_CSV,
    LANDUSE_SUITABILITY_CSV,
    load_stage2_context,
    persistence_categories,
    quintile_label,
    quintile_thresholds,
    build_temporal_transition_table,
    build_environmental_priority_table,
    build_green_built_table,
    build_landuse_suitability_table,
)

MASK_STRUCTURE = np.ones((3, 3), dtype=int)  # 8-connectivity (spec i)

PRIORITY_ZONE_STATISTICS_CSV = PHASE7_TABLES_DIR / "priority_zone_statistics.csv"
PRIORITY_RANKING_CSV = PHASE7_TABLES_DIR / "priority_ranking.csv"
WHY_HERE_CSV = PHASE7_TABLES_DIR / "why_here_explanations.csv"
WHY_HERE_MD = PHASE7_REPORTS_DIR / "why_here_examples.md"

BLOCK_NOTE = (
    "Ranking units are 8-connected priority zones derived from the 30 m "
    "grid. Spatial blocks elsewhere in the project are sampling-design "
    "blocks, NOT administrative units; no administrative ranking exists."
)

CONSTRAINTS_TEXT = (
    "None implementable beyond the invalid mask: no water-exclusion, "
    "building-footprint or road-surface exclusion layers exist (input "
    "audit section 4; spec section e) - a stated limitation, not a "
    "silent omission."
)

LANDUSE_STATIC_NOTE = (
    "landuse raster is STATIC and reused for both years; class 0 is the "
    "absence of an OSM tag, not a real land-use observation"
)

RECOMMENDATION_BY_LEVEL = {
    "Very High": (
        "Very High Priority - requires field verification before any "
        "planting commitment."
    ),
    "High": (
        "High Priority - candidate for phased planting assessment and "
        "field verification."
    ),
    "High+Very High": (
        "High Priority - candidate for phased planting assessment and "
        "field verification."
    ),
}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _now() -> str:
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


def recommendation_for(class_level: str) -> str:
    """Recommendation sentence for a zone (spec section 26 style)."""
    return RECOMMENDATION_BY_LEVEL[class_level]


# ---------------------------------------------------------------------------
# Zone labelling and polygons (Phase 6-validated pattern)
# ---------------------------------------------------------------------------
def label_zones(binary_grid: np.ndarray, min_pixels: int = MIN_ZONE_PIXELS) -> Tuple[np.ndarray, int]:
    """Label 8-connected zones and drop clusters below the min-size rule.

    Parameters
    ----------
    binary_grid : np.ndarray
        Full-grid boolean membership mask (only valid cells can be True).
    min_pixels : int
        Operational noise rule (FROZEN, spec section i): clusters smaller
        than 10 connected pixels (~0.8 ha at 30 m) are removed — the same
        rule Phase 6 used for hotspots.

    Returns
    -------
    tuple
        (labelled int32 raster with dense IDs 1..n_kept, n_kept).
    """
    labels, n_found = ndimage.label(binary_grid, structure=MASK_STRUCTURE)
    if n_found == 0:
        return labels.astype(np.int32), 0
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False  # background is never a zone
    remap = np.zeros(n_found + 1, dtype=np.int32)
    remap[np.where(keep)[0]] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[labels].astype(np.int32), int(keep.sum())


def polygons_per_zone(labelled: np.ndarray, transform) -> Dict[int, BaseGeometry]:
    """Dissolve pixel polygons into one boundary polygon per zone ID.

    ``raster_shapes`` is called with 4-connectivity so corner-touching
    cells are emitted as separate rings; dissolving per ID (8-connected
    zone semantics are already encoded in the IDs) yields exact, valid
    Polygon/MultiPolygon boundaries.  Geometries are in EPSG:4326.  No
    silent buffer-fixing: validity is checked and reported by the caller.
    """
    per_id: Dict[int, List[BaseGeometry]] = {}
    for geom, value in raster_shapes(
        labelled, mask=labelled > 0, transform=transform, connectivity=4
    ):
        per_id.setdefault(int(value), []).append(shapely_shape(geom))
    return {zid: unary_union(geoms) for zid, geoms in per_id.items()}


def validate_geometries(polygons: Dict[int, BaseGeometry]) -> Dict:
    """Check zone polygons for invalid / zero-area / duplicate geometries.

    Problems are REPORTED, never silently repaired (exact pixel unions are
    the intended boundaries).
    """
    invalid_ids = [z for z, g in polygons.items() if not g.is_valid]
    zero_area_ids = [z for z, g in polygons.items() if g.area <= 0.0]
    ids = list(polygons)
    duplicate_ids = len(ids) != len(set(ids))
    return {
        "n_polygons": len(ids),
        "n_invalid": len(invalid_ids),
        "invalid_ids": invalid_ids,
        "n_zero_area": len(zero_area_ids),
        "zero_area_ids": zero_area_ids,
        "duplicate_ids": duplicate_ids,
        "all_valid": (
            not invalid_ids and not zero_area_ids and not duplicate_ids
        ),
    }


# ---------------------------------------------------------------------------
# Zone extraction with full attribute table
# ---------------------------------------------------------------------------
def extract_zones(
    ctx: Dict,
    year: int,
    scenario: str,
    level: str,
) -> Dict:
    """Extract priority zones for one scenario/year/class level.

    Parameters
    ----------
    ctx : dict
        Stage 2 context from ``statistics.load_stage2_context``.
    year : int
        Snapshot year.
    scenario : str
        Scenario letter (A baseline / B / C / D).
    level : str
        ``"high"`` (class >= 3) or ``"very_high"`` (class == 4), per the
        FROZEN rule (spec section i) — the two masks are labelled
        separately.

    Returns
    -------
    dict
        ``frame`` (attribute DataFrame), ``flat_ids`` (per-valid-cell zone
        IDs), ``labelled`` (full-grid ID raster), ``polygons`` (EPSG:4326),
        ``binary`` (membership mask), ``validation`` (geometry + pixel
        accounting checks).
    """
    cls_grid = ctx["class_grids"][scenario][year]
    if level == "high":
        binary = cls_grid >= min(HIGH_VERY_HIGH_CLASSES)
    elif level == "very_high":
        binary = cls_grid == max(HIGH_VERY_HIGH_CLASSES)
    else:
        raise AssertionError(f"unknown zone level: {level}")

    labelled, n_kept = label_zones(binary)
    flat_ids = labelled[ctx["rows"], ctx["cols"]].astype(np.int32)
    polygons = polygons_per_zone(labelled, ctx["transform"]) if n_kept else {}
    frame = compute_zone_attributes(ctx, year, scenario, level, flat_ids, polygons)

    validation = validate_geometries(polygons)
    validation.update(
        {
            "mask_pixels_raw": int(binary.sum()),
            "mask_pixels_kept": int((labelled > 0).sum()),
            "noise_pixels_removed": int(binary.sum() - (labelled > 0).sum()),
            "zone_pixel_count_sum": int(frame["pixel_count"].sum()) if not frame.empty else 0,
            "pixel_counts_match": bool(
                (frame["pixel_count"].sum() if not frame.empty else 0)
                == int((labelled > 0).sum())
            ),
        }
    )
    return {
        "frame": frame,
        "flat_ids": flat_ids,
        "labelled": labelled,
        "polygons": polygons,
        "binary": binary,
        "validation": validation,
        "level": level,
        "n_zones": n_kept,
    }


def compute_zone_attributes(
    ctx: Dict,
    year: int,
    scenario: str,
    level: str,
    flat_ids: np.ndarray,
    polygons: Dict[int, BaseGeometry],
) -> pd.DataFrame:
    """Build the per-zone attribute table (one row per zone).

    Area is the EPSG:32643 polygon area (degrees are never used).
    ``priority_confidence`` is the FROZEN spec-section-j index:
    100 x mean Phase 6 confidence x scenario agreement rate.  Need /
    Opportunity are the baseline composition inputs shared by scenarios
    A-C (only the gating exponents differ); the landuse layer is static
    and reused for both years (documented in ``landuse_note``).
    """
    class_level = {"high": "High", "very_high": "Very High"}[level]
    ids = sorted(int(i) for i in np.unique(flat_ids[flat_ids > 0]))
    if not ids:
        return pd.DataFrame(
            columns=[
                "zone_id", "year", "scenario", "class_level", "pixel_count",
                "area_m2", "area_ha", "centroid_lon", "centroid_lat",
                "mean_suitability", "median_suitability", "mean_heat_need",
                "mean_opportunity", "mean_lst_C", "median_lst_C", "mean_ndvi",
                "mean_ndbi", "mean_vegetation_cover", "mean_severity_score",
                "mean_phase6_confidence", "dominant_landuse_code",
                "dominant_landuse_name", "landuse_note", "priority_confidence",
                "recommendation",
            ]
        )

    gdf = gpd.GeoDataFrame(
        {"zone_id": ids},
        geometry=[polygons[z] for z in ids],
        crs=WGS84_CRS,
    ).to_crs(UTM_CRS)

    src = ctx["yearly"][year]
    landuse = ctx["static"]["landuse"]
    rows: List[Dict] = []
    for idx, zid in enumerate(ids):
        sel = flat_ids == zid
        n_px = int(sel.sum())
        geom_utm = gdf.geometry.iloc[idx]
        area_m2 = float(geom_utm.area)
        geom_4326 = polygons[zid]
        cx, cy = geom_4326.centroid.x, geom_4326.centroid.y

        lu_vals, lu_counts = np.unique(landuse[sel], return_counts=True)
        dom_code = int(lu_vals[np.argmax(lu_counts)])

        mean_conf = float(ctx["confidence"][year][sel].mean())
        mean_agr = float(ctx["agreement"][year][sel].mean())
        priority_confidence = 100.0 * mean_conf * mean_agr

        suitability = ctx["scores"][scenario][year][sel]
        lst = src["lst"][sel]
        rows.append(
            {
                "zone_id": zid,
                "year": year,
                "scenario": scenario,
                "class_level": class_level,
                "pixel_count": n_px,
                "area_m2": round(area_m2, 1),
                "area_ha": round(area_m2 / 1e4, 3),
                "centroid_lon": round(cx, 7),
                "centroid_lat": round(cy, 7),
                "mean_suitability": round(float(suitability.mean()), 4),
                "median_suitability": round(float(np.median(suitability)), 4),
                "mean_heat_need": round(float(ctx["need"][year][sel].mean()), 4),
                "mean_opportunity": round(float(ctx["opportunity"][year][sel].mean()), 4),
                "mean_lst_C": round(float(lst.mean()), 3),
                "median_lst_C": round(float(np.median(lst)), 3),
                "mean_ndvi": round(float(src["ndvi"][sel].mean()), 6),
                "mean_ndbi": round(float(src["ndbi"][sel].mean()), 6),
                "mean_vegetation_cover": round(float(src["vegetation_cover"][sel].mean()), 6),
                "mean_severity_score": round(float(src["severity_score"][sel].mean()), 6),
                "mean_phase6_confidence": round(mean_conf, 6),
                "dominant_landuse_code": dom_code,
                "dominant_landuse_name": LANDUSE_CLASS_NAMES.get(dom_code, "unknown"),
                "landuse_note": LANDUSE_STATIC_NOTE,
                "priority_confidence": round(priority_confidence, 2),
                "recommendation": recommendation_for(class_level),
            }
        )
    return pd.DataFrame(rows)


def write_zones_geojson(frame: pd.DataFrame, polygons: Dict[int, BaseGeometry], path: Path) -> None:
    """Write zone boundaries + attributes to GeoJSON (EPSG:4326)."""
    if frame.empty:
        gdf = gpd.GeoDataFrame(
            {
                "zone_id": pd.Series(dtype="int64"),
                "year": pd.Series(dtype="int64"),
                "scenario": pd.Series(dtype="object"),
            },
            geometry=gpd.GeoSeries(dtype="geometry"),
            crs=WGS84_CRS,
        )
    else:
        gdf = gpd.GeoDataFrame(
            frame,
            geometry=[polygons[int(z)] for z in frame["zone_id"]],
            crs=WGS84_CRS,
        )
    gdf.to_file(path, driver="GeoJSON")


# ---------------------------------------------------------------------------
# Ranked tables (priority_zone_statistics.csv, priority_ranking.csv)
# ---------------------------------------------------------------------------
def rank_zones(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate zone frames and rank by area, then mean suitability.

    Ranking is within (year, scenario) — the FROZEN order is area
    descending, ties broken by mean suitability descending.
    """
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        df["rank_by_area"] = pd.Series(dtype="Int64")
        return df
    df = df.sort_values(
        ["year", "scenario", "area_ha", "mean_suitability"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    df["rank_by_area"] = (
        df.groupby(["year", "scenario"]).cumcount() + 1
    ).astype("Int64")
    return df


def build_priority_ranking_table(zones_df: pd.DataFrame) -> pd.DataFrame:
    """Build ``priority_ranking.csv`` — the ranked decision-support list."""
    if zones_df.empty:
        return pd.DataFrame(
            columns=[
                "rank", "year", "scenario", "zone_id", "class_level", "area_ha",
                "mean_suitability", "mean_heat_need", "mean_opportunity",
                "priority_confidence", "recommendation", "note",
            ]
        )
    df = zones_df.sort_values(
        ["year", "scenario", "rank_by_area"]
    ).reset_index(drop=True)
    out = pd.DataFrame(
        {
            "rank": df["rank_by_area"],
            "year": df["year"],
            "scenario": df["scenario"],
            "zone_id": df["zone_id"],
            "class_level": df["class_level"],
            "area_ha": df["area_ha"],
            "mean_suitability": df["mean_suitability"],
            "mean_heat_need": df["mean_heat_need"],
            "mean_opportunity": df["mean_opportunity"],
            "priority_confidence": df["priority_confidence"],
            "recommendation": df["recommendation"],
            "note": BLOCK_NOTE,
        }
    )
    return out


# ---------------------------------------------------------------------------
# "Why here?" explanations (design spec section 26)
# ---------------------------------------------------------------------------
def _road_band(mean_dist_m: float) -> str:
    if mean_dist_m < 50.0:
        return "road corridor (<50 m; low score)"
    if mean_dist_m <= 500.0:
        return "optimal (50-500 m)"
    if mean_dist_m <= 2000.0:
        return "declining (500-2000 m)"
    return "far (>2000 m; floor score)"


def _green_proximity_band(mean_dist_m: float) -> str:
    if mean_dist_m <= 50.0:
        return "adjacent (<=50 m)"
    if mean_dist_m <= 150.0:
        return "close (50-150 m)"
    if mean_dist_m <= 250.0:
        return "moderate (150-250 m)"
    if mean_dist_m <= 500.0:
        return "distant (250-500 m)"
    return "beyond 500 m cap (score 0)"


def build_why_here_rows(ctx: Dict, zone_sets: Dict, n_top: int = 10) -> pd.DataFrame:
    """Build the machine-readable "why here?" table for A/B/C.

    For the top ``n_top`` zones per year per scenario (rank by area, then
    mean suitability): suitability class + score, heat-need drivers
    (severity score, LST, NDVI, NDBI — quintile labels against that
    year's valid-cell distribution), opportunity drivers (landuse class +
    eligibility, built-up intensity level, road band, green proximity
    band, planting headroom), the constraint statement, the recommendation
    sentence and priority_confidence with its documented meaning.

    NDVI is interpreted twice, deliberately: low NDVI raises heat NEED
    (sparse vegetation -> heat stress) and raises planting headroom
    (opportunity) — the direction note makes this explicit.
    """
    rows: List[Dict] = []
    for year in YEARS:
        # Quintile cut points for this year's valid-cell distribution.
        src = ctx["yearly"][year]
        thr = {
            "severity_score": quintile_thresholds(src["severity_score"]),
            "lst": quintile_thresholds(src["lst"]),
            "ndvi": quintile_thresholds(src["ndvi"]),
            "ndbi": quintile_thresholds(src["ndbi"]),
        }
        p1, p99 = ctx["ndvi_bounds"][year]
        ndvi_norm = np.clip((src["ndvi"] - p1) / (p99 - p1), 0.0, 1.0) * 100.0
        headroom = 100.0 - ndvi_norm
        thr["headroom"] = quintile_thresholds(headroom)

        for scenario in SPATIAL_SCENARIOS:
            zs = zone_sets[scenario][year]
            frame = zs["frame"].sort_values(
                ["area_ha", "mean_suitability"], ascending=[False, False]
            ).head(n_top)
            flat_ids = zs["flat_ids"]
            for rank, rec in enumerate(frame.itertuples(), start=1):
                sel = flat_ids == int(rec.zone_id)
                sev = float(src["severity_score"][sel].mean())
                lst = float(src["lst"][sel].mean())
                ndvi = float(src["ndvi"][sel].mean())
                ndbi = float(src["ndbi"][sel].mean())
                hr = float(headroom[sel].mean())
                dist_road = float(ctx["static"]["dist_road_m"][sel].mean())
                dist_veg = float(ctx["static"]["dist_vegetation_m"][sel].mean())
                green_score = 100.0 * max(0.0, 1.0 - dist_veg / 500.0)
                rows.append(
                    {
                        "rank": rank,
                        "zone_id": int(rec.zone_id),
                        "year": year,
                        "scenario": scenario,
                        "class_level": rec.class_level,
                        "suitability_class": rec.class_level,
                        "mean_suitability": rec.mean_suitability,
                        "severity_score_mean": round(sev, 4),
                        "severity_score_level": quintile_label(sev, thr["severity_score"]),
                        "lst_mean_C": round(lst, 3),
                        "lst_level": quintile_label(lst, thr["lst"]),
                        "ndvi_mean": round(ndvi, 6),
                        "ndvi_level": quintile_label(ndvi, thr["ndvi"]),
                        "ndvi_direction_note": (
                            "low NDVI raises heat need (sparse vegetation) "
                            "AND planting headroom (opportunity)"
                        ),
                        "ndbi_mean": round(ndbi, 6),
                        "ndbi_level": quintile_label(ndbi, thr["ndbi"]),
                        "landuse_code": int(rec.dominant_landuse_code),
                        "landuse_name": rec.dominant_landuse_name,
                        "landuse_eligibility_score": LANDUSE_ELIGIBILITY.get(
                            int(rec.dominant_landuse_code), 50
                        ),
                        "built_up_intensity_level": quintile_label(ndbi, thr["ndbi"]),
                        "mean_dist_road_m": round(dist_road, 1),
                        "road_accessibility_band": _road_band(dist_road),
                        "mean_dist_vegetation_m": round(dist_veg, 1),
                        "green_proximity_band": _green_proximity_band(dist_veg),
                        "mean_green_proximity_score": round(green_score, 2),
                        "planting_headroom_mean": round(hr, 2),
                        "planting_headroom_level": quintile_label(hr, thr["headroom"]),
                        "constraints": CONSTRAINTS_TEXT,
                        "recommendation": rec.recommendation,
                        "priority_confidence": rec.priority_confidence,
                        "priority_confidence_meaning": PRIORITY_CONFIDENCE_MEANING,
                    }
                )
    return pd.DataFrame(rows)


def write_why_here_examples_md(why_df: pd.DataFrame, path: Path, n_examples: int = 3) -> None:
    """Write the worked "why here?" examples in the spec section 26 format."""
    lines: List[str] = [
        "# Phase 7 Stage 2 - 'Why here?' worked examples",
        "",
        "Format follows design spec section 26.  Driver levels are quintiles",
        "of that year's valid-cell distribution (Q1 = lowest 20%, Q5 =",
        "highest 20%).  These explanations describe a RELATIVE suitability",
        "ranking on a relative heat-severity proxy - nothing here claims",
        "planting success probability.",
        "",
    ]
    if why_df.empty:
        lines += [
            "No priority zones exist to explain: the baseline Scenario A",
            "produced zero High/Very High cells in one or both years (a",
            "real result of the gated formulation, not a bug).",
            "",
        ]
    else:
        # One worked example per scenario (A/B/C) for coverage; each is the
        # rank-1 zone of that scenario's first year with zones.
        picks: List = []
        seen = set()
        for rec in why_df.itertuples():
            if rec.scenario not in seen:
                picks.append(rec)
                seen.add(rec.scenario)
            if len(picks) == n_examples:
                break
        for rec in picks:
            lines += [
                f"## Zone "
                f"{rec.scenario}-{rec.year}-{rec.zone_id} "
                f"({rec.class_level}, {rec.mean_suitability:.1f}, "
                f"zone rank {rec.rank} by area)",
                "",
                f"- **Suitability:** class {rec.suitability_class}; mean "
                f"suitability score {rec.mean_suitability:.2f}.",
                f"- **Heat-need drivers:** Phase 6 severity score "
                f"{rec.severity_score_mean:.2f} ({rec.severity_score_level}); "
                f"LST {rec.lst_mean_C:.1f} C ({rec.lst_level}); NDVI "
                f"{rec.ndvi_mean:.3f} ({rec.ndvi_level} - {rec.ndvi_direction_note}); "
                f"NDBI {rec.ndbi_mean:.3f} ({rec.ndbi_level}).",
                f"- **Opportunity drivers:** landuse "
                f"{rec.landuse_name} (class {rec.landuse_code}, eligibility "
                f"{rec.landuse_eligibility_score}); built-up intensity "
                f"{rec.built_up_intensity_level}; road accessibility "
                f"{rec.road_accessibility_band} (mean {rec.mean_dist_road_m:.0f} m); "
                f"green proximity {rec.green_proximity_band} (mean "
                f"{rec.mean_dist_vegetation_m:.0f} m, score "
                f"{rec.mean_green_proximity_score:.0f}/100); planting headroom "
                f"{rec.planting_headroom_mean:.1f}/100 ({rec.planting_headroom_level}).",
                f"- **Constraints:** {CONSTRAINTS_TEXT}",
                f"- **Priority confidence:** {rec.priority_confidence:.1f}/100.  "
                f"{PRIORITY_CONFIDENCE_MEANING}",
                f"- **Recommendation:** {rec.recommendation}",
                "",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Spatial sensitivity extension (spec section h + Stage 2 spatial stats)
# ---------------------------------------------------------------------------
def build_sensitivity_spatial_rows(
    ctx: Dict, zone_sets: Dict
) -> pd.DataFrame:
    """Build the Stage 2 spatial-overlap rows for ``sensitivity_analysis.csv``.

    Per scenario B/C/D per year: High+VH area and % of valid, mask IoU vs
    baseline A, % of baseline's High+VH contained in the scenario's,
    rank stability of the top-10 baseline zones (fraction appearing in the
    scenario's top-20 zones by area, matched by pixel overlap) and the
    High+VH mask class-transition summary (persisted / lost / gained).
    """
    rows: List[Dict] = []
    for year in YEARS:
        hvh_a = ctx["classes"]["A"][year] >= min(HIGH_VERY_HIGH_CLASSES)
        a_top = _top_zone_masks(zone_sets["A"][year], 10)
        for scenario in ("B", "C", "D"):
            hvh_s = ctx["classes"][scenario][year] >= min(HIGH_VERY_HIGH_CLASSES)
            inter = int((hvh_s & hvh_a).sum())
            union = int((hvh_s | hvh_a).sum())
            n_s = int(hvh_s.sum())
            n_a = int(hvh_a.sum())

            s_top = _top_zone_masks(zone_sets[scenario][year], 20)
            if a_top:
                matched = sum(
                    1
                    for a_mask in a_top
                    if any(
                        np.count_nonzero(a_mask & s_mask) > 0 for s_mask in s_top
                    )
                )
                rank_stability = matched / len(a_top)
            else:
                rank_stability = np.nan

            flat_s = zone_sets[scenario][year]["flat_ids"]
            rows.append(
                {
                    "section": "spatial_overlap_stage2",
                    "scenario": scenario,
                    "year": year,
                    "high_very_high_area_ha": round(
                        float(ctx["pixel_areas_m2"][hvh_s].sum()) / 1e4, 3
                    ),
                    "high_very_high_pct": round(100.0 * n_s / ctx["n_valid"], 4),
                    "mask_iou_vs_baseline_A": round(inter / union, 6) if union else np.nan,
                    "pct_of_baseline_hvh_contained": (
                        round(100.0 * inter / n_a, 4) if n_a else np.nan
                    ),
                    "rank_stability_top10_in_top20": (
                        round(rank_stability, 4) if np.isfinite(rank_stability) else np.nan
                    ),
                    "n_baseline_zones_top10_considered": len(a_top),
                    "n_scenario_zones_top20": len(s_top),
                    "transition_persist_px": inter,
                    "transition_lost_px": int((hvh_a & ~hvh_s).sum()),
                    "transition_gained_px": int((~hvh_a & hvh_s).sum()),
                }
            )
    return pd.DataFrame(rows)


def _top_zone_masks(zone_set: Dict, n: int) -> List[np.ndarray]:
    """Boolean per-valid-cell masks of the top-n zones by area."""
    frame = zone_set["frame"]
    if frame.empty:
        return []
    flat_ids = zone_set["flat_ids"]
    top = frame.sort_values("area_ha", ascending=False).head(n)["zone_id"]
    return [flat_ids == int(z) for z in top]


def extend_sensitivity_csv(rows_df: pd.DataFrame) -> Path:
    """Append the Stage 2 spatial rows to the Stage 1 sensitivity CSV.

    Idempotent: any previous ``spatial_overlap_stage2`` rows are dropped
    first, so re-runs never duplicate the Stage 2 extension.
    """
    path = TABLE_PATHS["sensitivity"]
    existing = pd.read_csv(path)
    existing = existing[existing["section"] != "spatial_overlap_stage2"]
    combined = pd.concat([existing, rows_df], ignore_index=True, sort=False)
    for col in [c for c in combined.columns if c.endswith(("_px", "_count"))]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").astype("Int64")
    combined.to_csv(path, index=False)
    return path


def write_sensitivity_findings(
    ctx: Dict, spatial_df: pd.DataFrame, path: Path = SENSITIVITY_FINDINGS_MD
) -> None:
    """Write the plain-language sensitivity findings (headline: gating)."""
    lines: List[str] = [
        "# Phase 7 Stage 2 - sensitivity findings (spatial)",
        "",
        "Headline finding: **priority-zone extent is highly sensitive to the",
        "need/opportunity weighting; the gated baseline is conservative by",
        "construction.**",
        "",
        "The baseline Scenario A applies the FROZEN gated product",
        "Final = Need x Opportunity / 100.  Because both factors are 0-100",
        "and their product is divided by 100, the baseline distribution is",
        "strongly compressed toward low scores (mean "
        f"{ctx['scores']['A'][2022].mean():.2f} in 2022, "
        f"{ctx['scores']['A'][2026].mean():.2f} in 2026), so almost no cell",
        "crosses the High (60) or Very High (80) thresholds.  Scenarios B",
        "(heat-focused) and C (feasibility-focused) re-gate the SAME Need",
        "and Opportunity with weighted-geometric-mean exponents, which",
        "lifts the score mass and yields two to four orders of magnitude",
        "more High+Very High area:",
        "",
        "| scenario | year | mean suitability | High+VH area (ha) | % of valid |",
        "|---|---|---|---|---|",
    ]
    for scenario in ("A", "B", "C", "D"):
        for year in YEARS:
            hvh = ctx["classes"][scenario][year] >= min(HIGH_VERY_HIGH_CLASSES)
            lines.append(
                f"| {scenario} | {year} | {ctx['scores'][scenario][year].mean():.2f} "
                f"| {ctx['pixel_areas_m2'][hvh].sum() / 1e4:,.1f} "
                f"| {100.0 * hvh.sum() / ctx['n_valid']:.4f} |"
            )
    lines += [
        "",
        "Mask overlap against the baseline (full detail in the "
        "`spatial_overlap_stage2` section of `sensitivity_analysis.csv`):",
        "",
    ]
    for rec in spatial_df.itertuples():
        lines.append(
            f"- Scenario {rec.scenario} {rec.year}: IoU vs A = "
            f"{rec.mask_iou_vs_baseline_A:.4f}; "
            f"{rec.pct_of_baseline_hvh_contained:.1f}% of A's High+VH cells are "
            f"contained in {rec.scenario}'s mask; top-10 baseline rank stability "
            f"= {rec.rank_stability_top10_in_top20:.2f} "
            f"({rec.n_baseline_zones_top10_considered} baseline zones vs "
            f"{rec.n_scenario_zones_top20} scenario top-20 zones)."
        )
    lines += [
        "",
        "Interpretation: 100% of the (very few) baseline priority cells are",
        "reproduced by scenarios B/C/D — the baseline is a strict subset of",
        "the relaxed scenarios — but the baseline's tiny extent means the IoU",
        "is near zero.  Top-10 baseline rank stability is 1.0 in the",
        "containment sense (every baseline zone overlaps a scenario top-20",
        "zone), but that reflects the tiny baseline zones falling inside",
        "scenario zones that are hundreds of times larger — not shape- or",
        "rank-preservation at comparable scales.  The baseline should",
        "therefore be read as a deliberately conservative core priority set;",
        "scenarios B/C bracket how the priority map broadens when the",
        "need/opportunity gating is relaxed.  All classes are per-year",
        "normalized (class-relative, snapshots not trends).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Temporal priority zones (persistent / emerging / declining vectors)
# ---------------------------------------------------------------------------
def build_temporal_zone_vectors(
    ctx: Dict,
    scenario: str,
    path: Path,
) -> Dict:
    """Delineate persistent/emerging/declining priority zones for a scenario.

    Zones are 8-connected components (MIN_ZONE_PIXELS noise rule) on each
    temporal category mask at paired valid cells.  A thin attribute set is
    attached (category, pixel count, UTM area, centroid).
    """
    hvh = {
        year: ctx["classes"][scenario][year] >= min(HIGH_VERY_HIGH_CLASSES)
        for year in YEARS
    }
    cats = persistence_categories(hvh[YEARS[0]], hvh[YEARS[1]])
    rows, cols = ctx["rows"], ctx["cols"]

    frames: List[pd.DataFrame] = []
    geometries: List[BaseGeometry] = []
    validations: Dict[str, Dict] = {}
    for category in ("Persistent", "Emerging", "Declining"):
        grid_mask = np.zeros(ctx["valid_mask"].shape, dtype=bool)
        grid_mask[rows, cols] = cats[category]
        labelled, n_kept = label_zones(grid_mask)
        polygons = polygons_per_zone(labelled, ctx["transform"]) if n_kept else {}
        validations[category] = validate_geometries(polygons)
        validations[category]["n_zones"] = n_kept
        validations[category]["category_pixels"] = int(cats[category].sum())
        if n_kept:
            ids = sorted(int(i) for i in np.unique(labelled[labelled > 0]))
            gdf = (
                gpd.GeoDataFrame(
                    {"zone_id": ids},
                    geometry=[polygons[z] for z in ids],
                    crs=WGS84_CRS,
                )
                .to_crs(UTM_CRS)
            )
            for idx, zid in enumerate(ids):
                geom_4326 = polygons[zid]
                cx, cy = geom_4326.centroid.x, geom_4326.centroid.y
                frames.append(
                    {
                        "zone_id": zid,
                        "year_pair": f"{YEARS[0]}-{YEARS[1]}",
                        "scenario": scenario,
                        "persistence_category": category,
                        "pixel_count": int((labelled == zid).sum()),
                        "area_ha": round(float(gdf.geometry.iloc[idx].area) / 1e4, 3),
                        "centroid_lon": round(cx, 7),
                        "centroid_lat": round(cy, 7),
                    }
                )
                geometries.append(geom_4326)

    frame = pd.DataFrame(frames)
    write_zones_geojson(frame, dict(zip(frame["zone_id"], geometries)) if not frame.empty else {}, path)
    return {"frame": frame, "validation": validations, "output_path": str(path)}


# ---------------------------------------------------------------------------
# Records: manifest + pipeline record (idempotent)
# ---------------------------------------------------------------------------
def _update_manifest(record: Dict) -> None:
    """Read-modify-write the Phase 7 manifest with the Stage 2 section."""
    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)

    manifest["priority_zones"] = {
        "stage": 2,
        "generated_utc": _now(),
        "framing": (
            "Relative priority zones of the relative suitability ranking; "
            "not a statement of legal availability, ownership, or planting "
            "survival probability.  NoData never participates."
        ),
        "rule": {
            "connectivity": "8 (scipy.ndimage, structure=ones((3,3)))",
            "masks": "class >= 3 (High) and class == 4 (Very High) labelled "
                     "separately per year (spec section i)",
            "min_zone_pixels": MIN_ZONE_PIXELS,
            "min_area_rule": (
                "clusters smaller than 10 connected 8-connected pixels "
                "(~0.8 ha at 30 m) removed as operational noise - the same "
                "rule Phase 6 used for hotspots"
            ),
            "polygons": "rasterio.features.shapes connectivity=4 + per-ID "
                        "dissolve (Phase 6-validated pattern); EPSG:4326",
            "area_crs": UTM_CRS,
            "area_note": "areas computed by reprojecting polygons to "
                         "EPSG:32643 (UTM 43N, Delhi); degrees are NOT "
                         "equal-area",
        },
        "priority_confidence": {
            "formula": "100 * mean_phase6_confidence(0-1) * scenario_agreement_rate",
            "scenario_agreement_rate": "fraction of scenarios A-D assigning "
                                       "class >= 3 (High) at the pixel",
            "meaning": PRIORITY_CONFIDENCE_MEANING,
        },
        "outputs": record["outputs"],
        "validation": record["validation"],
        "deviations": record.get("deviations", []),
    }
    _save_json(manifest, MANIFEST_JSON)


def _append_pipeline_record(record: Dict) -> None:
    """Append the Stage 2 entry to the Phase 7 pipeline record (idempotent)."""
    with open(PIPELINE_RECORD_JSON) as f:
        existing = json.load(f)

    container = {"phase": 7, "project": "GreenGrid-AI", "stage_records": []}
    if "stage_records" in existing:
        container["stage_records"] = existing["stage_records"]
    else:
        container["stage_records"] = [existing]
    # Idempotent re-runs replace the previous Stage 2 entry.
    container["stage_records"] = [
        r for r in container["stage_records"] if r.get("stage") != 2
    ]
    container["stage_records"].append(record)
    _save_json(container, PIPELINE_RECORD_JSON)


# ---------------------------------------------------------------------------
# Stage 2 driver
# ---------------------------------------------------------------------------
def run_stage2() -> Dict:
    """Run the complete Phase 7 Stage 2 priority-zone pipeline."""
    PHASE7_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE7_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    record: Dict = {
        "phase": 7,
        "stage": 2,
        "project": "GreenGrid-AI",
        "project_root": str(PROJECT_ROOT),
        "started_utc": _now(),
        "random_seed": RANDOM_SEED,
        "steps": {},
        "deviations": [],
    }

    # ------------------------------------------------------------------
    # 1. Shared context (inputs, scenario scores/classes, areas)
    # ------------------------------------------------------------------
    t0 = time.time()
    ctx = load_stage2_context()
    print(
        f"  Valid cells per year: {ctx['n_valid']:,}; "
        f"valid area {ctx['valid_area_ha']:,.1f} ha (EPSG:32643)"
    )
    record["steps"]["context"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "valid_cells_per_year": int(ctx["n_valid"]),
        "valid_area_ha": round(ctx["valid_area_ha"], 3),
    }

    # ------------------------------------------------------------------
    # 2. Zone extraction: baseline A (High + Very High) + scenarios B/C/D
    # ------------------------------------------------------------------
    t0 = time.time()
    zone_sets: Dict[str, Dict[int, Dict]] = {s: {} for s in ALL_SCENARIOS}
    for year in YEARS:
        for scenario in ALL_SCENARIOS:
            zone_sets[scenario][year] = extract_zones(ctx, year, scenario, "high")
        zone_sets["A"][year]["very_high"] = extract_zones(ctx, year, "A", "very_high")
        for scenario in ("A", "B", "C"):
            zs = zone_sets[scenario][year]
            print(
                f"  [{year} scenario {scenario}] High+VH zones: {zs['n_zones']} "
                f"({zs['validation']['mask_pixels_kept']:,} px of "
                f"{zs['validation']['mask_pixels_raw']:,} mask px; "
                f"geom valid={zs['validation']['all_valid']})"
            )
        vh = zone_sets["A"][year]["very_high"]
        print(
            f"  [{year} scenario A] Very High zones: {vh['n_zones']} "
            f"({vh['validation']['mask_pixels_kept']:,} px)"
        )
    record["steps"]["zone_extraction"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "min_zone_pixels": MIN_ZONE_PIXELS,
        "connectivity": "8",
        "zone_counts": {
            s: {
                str(y): zone_sets[s][y]["n_zones"] for y in YEARS
            }
            for s in ("A", "B", "C", "D")
        },
        "very_high_zone_counts": {
            str(y): zone_sets["A"][y]["very_high"]["n_zones"] for y in YEARS
        },
        "note": (
            "baseline A is deliberately sparse (gated product): few/zero "
            "zones is the REAL result, not a bug; B/C/D zones are extracted "
            "for the sensitivity discussion"
        ),
    }

    # ------------------------------------------------------------------
    # 3. GeoJSON outputs
    # ------------------------------------------------------------------
    t0 = time.time()
    outputs: Dict[str, str] = {}
    for year in YEARS:
        zs = zone_sets["A"][year]
        for level, fname in (
            ("high", f"high_priority_zones_{year}.geojson"),
            ("very_high", f"very_high_priority_zones_{year}.geojson"),
        ):
            if level == "high":
                frame, polys = zs["frame"], zs["polygons"]
            else:
                frame, polys = zs["very_high"]["frame"], zs["very_high"]["polygons"]
            path = PHASE7_VECTORS_DIR / fname
            write_zones_geojson(frame, polys, path)
            outputs[fname] = str(path)
        combined = pd.concat(
            [zs["frame"], zs["very_high"]["frame"]], ignore_index=True
        )
        combined_path = PHASE7_VECTORS_DIR / f"priority_zones_combined_{year}.geojson"
        write_zones_geojson(
            combined,
            {**zs["polygons"], **zs["very_high"]["polygons"]},
            combined_path,
        )
        outputs[combined_path.name] = str(combined_path)

        for scenario in ("B", "C"):
            frame = zone_sets[scenario][year]["frame"]
            path = PHASE7_VECTORS_DIR / f"scenario_priority_zones_{scenario}_{year}.geojson"
            write_zones_geojson(frame, zone_sets[scenario][year]["polygons"], path)
            outputs[path.name] = str(path)
    record["steps"]["vectors"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "outputs": outputs,
    }

    # ------------------------------------------------------------------
    # 4. Ranked tables
    # ------------------------------------------------------------------
    t0 = time.time()
    frames = []
    for year in YEARS:
        frames.append(zone_sets["A"][year]["frame"])
        frames.append(zone_sets["A"][year]["very_high"]["frame"])
        for scenario in ("B", "C"):
            frames.append(zone_sets[scenario][year]["frame"])
    zones_df = rank_zones(frames)
    zones_df.to_csv(PRIORITY_ZONE_STATISTICS_CSV, index=False)
    ranking_df = build_priority_ranking_table(zones_df)
    ranking_df.to_csv(PRIORITY_RANKING_CSV, index=False)
    print(f"  Priority zones total: {len(zones_df)} "
          f"(baseline A: {len(zones_df[zones_df['scenario'] == 'A'])})")
    record["steps"]["ranked_tables"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "priority_zone_statistics": str(PRIORITY_ZONE_STATISTICS_CSV),
        "priority_ranking": str(PRIORITY_RANKING_CSV),
        "ranking_rule": "area descending, ties broken by mean suitability; "
                        "within (year, scenario)",
    }

    # ------------------------------------------------------------------
    # 5. "Why here?" explanations
    # ------------------------------------------------------------------
    t0 = time.time()
    why_df = build_why_here_rows(ctx, zone_sets)
    why_df.to_csv(WHY_HERE_CSV, index=False)
    write_why_here_examples_md(why_df, WHY_HERE_MD)
    print(f"  Why-here explanations: {len(why_df)} rows")
    record["steps"]["why_here"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "csv": str(WHY_HERE_CSV),
        "examples_md": str(WHY_HERE_MD),
        "n_rows": int(len(why_df)),
        "format": "design spec section 26",
    }

    # ------------------------------------------------------------------
    # 6. Spatial sensitivity extension + findings
    # ------------------------------------------------------------------
    t0 = time.time()
    spatial_df = build_sensitivity_spatial_rows(ctx, zone_sets)
    extend_sensitivity_csv(spatial_df)
    write_sensitivity_findings(ctx, spatial_df)
    print(f"  Sensitivity spatial rows: {len(spatial_df)} "
          f"(sensitivity_analysis.csv extended)")
    record["steps"]["sensitivity_spatial"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "extended_csv": str(TABLE_PATHS["sensitivity"]),
        "findings_md": str(SENSITIVITY_FINDINGS_MD),
        "rows": _convert_for_json(spatial_df.to_dict(orient="records")),
    }

    # ------------------------------------------------------------------
    # 7. Temporal analysis: transition table + zone vectors
    # ------------------------------------------------------------------
    t0 = time.time()
    transition_df = build_temporal_transition_table(ctx)
    transition_df.to_csv(TEMPORAL_TRANSITION_CSV, index=False)
    print(f"  Temporal transition table: {TEMPORAL_TRANSITION_CSV.name} written")

    # Baseline A temporal zones if they exist; otherwise scenario B with a
    # documented deviation (spec: honest fallback, not a silent choice).
    temporal_scenario = "A"
    temporal = build_temporal_zone_vectors(
        ctx, "A", PHASE7_VECTORS_DIR / "temporal_priority_zones_A_2022_2026.geojson"
    )
    n_a_temporal = len(temporal["frame"])
    if n_a_temporal == 0:
        temporal_scenario = "B"
        record["deviations"].append(
            "Baseline A yielded zero persistent/emerging/declining zones "
            "(gated baseline too sparse: High 582 px 2022, 124 px 2026). "
            "Temporal zone vectors computed for Scenario B instead; "
            "transition COUNTS for A are still reported in "
            "temporal_priority_transition.csv."
        )
        temporal = build_temporal_zone_vectors(
            ctx, "B", PHASE7_VECTORS_DIR / "temporal_priority_zones_B_2022_2026.geojson"
        )
    outputs[f"temporal_priority_zones_{temporal_scenario}_2022_2026.geojson"] = (
        temporal["output_path"]
    )
    print(
        f"  Temporal zones: scenario {temporal_scenario}, "
        f"{len(temporal['frame'])} zones "
        f"(A baseline zones: {n_a_temporal})"
    )
    record["steps"]["temporal"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "transition_csv": str(TEMPORAL_TRANSITION_CSV),
        "zone_vectors_scenario": temporal_scenario,
        "zone_vectors": temporal["output_path"],
        "zone_counts_by_category": {
            k: v["n_zones"] for k, v in temporal["validation"].items()
        },
        "validation": temporal["validation"],
    }

    # ------------------------------------------------------------------
    # 8. Environmental / green-built / landuse statistics tables
    # ------------------------------------------------------------------
    t0 = time.time()
    env_df = build_environmental_priority_table(ctx)
    env_df.to_csv(ENVIRONMENTAL_PRIORITY_CSV, index=False)
    gb_df = build_green_built_table(ctx)
    gb_df.to_csv(GREEN_BUILT_CSV, index=False)
    lu_df = build_landuse_suitability_table(ctx)
    lu_df.to_csv(LANDUSE_SUITABILITY_CSV, index=False)
    record["steps"]["statistics_tables"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "outputs": {
            "environmental_priority_statistics": str(ENVIRONMENTAL_PRIORITY_CSV),
            "green_built_comparison": str(GREEN_BUILT_CSV),
            "landuse_suitability_summary": str(LANDUSE_SUITABILITY_CSV),
        },
    }

    # ------------------------------------------------------------------
    # 9. Area reconciliation + geometry validation
    # ------------------------------------------------------------------
    t0 = time.time()
    reconciliation: Dict = {"valid_area_ha": round(ctx["valid_area_ha"], 3)}
    for year in YEARS:
        classes = ctx["classes"]["A"][year]
        class_area = float(ctx["pixel_areas_m2"].sum()) / 1e4  # partition check below
        per_class = {
            int(c): round(float(ctx["pixel_areas_m2"][classes == c].sum()) / 1e4, 3)
            for c in range(len(CLASS_LABELS))
        }
        total = sum(per_class.values())
        rel_err = abs(total - ctx["valid_area_ha"]) / ctx["valid_area_ha"]
        reconciliation[str(year)] = {
            "per_class_area_ha": per_class,
            "sum_class_area_ha": round(total, 3),
            "valid_area_ha": round(ctx["valid_area_ha"], 3),
            "rel_err": round(rel_err, 8),
            "within_0.5pct": bool(rel_err <= 0.005),
        }
        assert rel_err <= 0.005, f"{year}: class areas do not reconcile to valid area"
    reconciliation["zone_pixel_checks"] = {}
    for scenario in ("A", "B", "C"):
        for year in YEARS:
            zs = zone_sets[scenario][year]
            ok = zs["validation"]["pixel_counts_match"]
            reconciliation["zone_pixel_checks"][f"{scenario}_{year}"] = {
                "match": ok,
                "zone_pixel_count_sum": zs["validation"]["zone_pixel_count_sum"],
                "mask_pixels_kept": zs["validation"]["mask_pixels_kept"],
            }
            assert ok, f"{scenario} {year}: zone pixel counts != mask counts"
    reconciliation["geometry_validation"] = {
        f"{s}_{y}": zone_sets[s][y]["validation"]
        for s in ("A", "B", "C")
        for y in YEARS
    }
    reconciliation["all_geometry_valid"] = all(
        zone_sets[s][y]["validation"]["all_valid"]
        for s in ("A", "B", "C")
        for y in YEARS
    )
    print(
        f"  Reconciliation: class areas == valid area within 0.5% both years; "
        f"zone pixel counts match masks; "
        f"all geometry valid={reconciliation['all_geometry_valid']}"
    )
    record["steps"]["reconciliation"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        **reconciliation,
    }
    record["outputs"] = outputs
    record["validation"] = reconciliation

    # ------------------------------------------------------------------
    # 10. Manifest + pipeline record
    # ------------------------------------------------------------------
    record["finished_utc"] = _now()
    record["total_elapsed_s"] = round(
        sum(s.get("elapsed_s", 0) for s in record["steps"].values() if isinstance(s, dict)),
        3,
    )
    record["status"] = "success"
    record["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rasterio": rasterio.__version__,
        "geopandas": gpd.__version__,
        "shapely": __import__("shapely").__version__,
        "scipy": ndimage.__name__ and __import__("scipy").__version__,
    }
    _update_manifest(record)
    _append_pipeline_record(record)
    print(f"  Manifest updated: {MANIFEST_JSON}")
    print(f"  Pipeline record appended: {PIPELINE_RECORD_JSON}")

    return {
        "record": record,
        "zones_df": zones_df,
        "ranking_df": ranking_df,
        "why_df": why_df,
        "spatial_df": spatial_df,
        "transition_df": transition_df,
        "ctx": ctx,
        "zone_sets": zone_sets,
    }


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for the Phase 7 Stage 2 priority-zone pipeline."""
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 7 Stage 2 priority-zone extraction."
    )
    parser.parse_args(argv)

    try:
        results = run_stage2()
        print(json.dumps(_convert_for_json(results["record"]), indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Phase 7 Stage 2 pipeline failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
