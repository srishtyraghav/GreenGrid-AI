"""Phase 8 PRODUCTION build: Tree Requirement Estimation.

Answers "how many trees should be planted, and where?" per Phase 7 priority
zone for all five years (2022-2026), per docs_production_build_spec.md §11.

Locked design decisions (spec section 11) implemented here:
  1. Available planting space = priority-zone pixels (Phase 7 priority
     zones, i.e. suitability class >= 3 on the frozen 0-indexed 0-4 scheme)
     AND NOT exclusion_mask AND landuse eligible AND vegetation_cover < 0.30.
  2. Tree density 1,000 trees/ha (~3.2 m spacing) -> 9 trees/px (0.09 ha/px);
     sensitivity at 400 and 2,500 trees/ha in the summary tables.
  3. Zone priority from Phase 7 priority_ranking_{year}.csv order (mean
     suitability desc); Priority = High/Medium/Low by ranking thirds.
  4. Per-zone report: zone id, Area (ha), Current vegetation (%), UHI
     severity (mean severity_score + High-class %), suitable planting space
     (ha), Recommended trees (= planting ha x 1000), Priority (+ planting px).
  5. Planning focus year 2026 (largest domain); all 5 years computed.

REUSED from ``src/`` (read-only; ``src/`` is NOT modified):
  - ``suitability.config``  — LANDUSE_ELIGIBILITY, LANDUSE_RULE_STATUS,
                              LANDUSE_NODATA, MIN_ZONE_PIXELS
  - ``suitability.zones``   — label_zones, polygons_per_zone,
                              validate_geometries
  - ``models.config``       — REFERENCE_RASTER (production grid definition)

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/run_phase8_tree_requirement.py
      [--years 2022,2023,...] [--skip-figures]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from models.config import REFERENCE_RASTER  # noqa: E402
from suitability.config import (  # noqa: E402
    LANDUSE_ELIGIBILITY,
    LANDUSE_NODATA,
    LANDUSE_RULE_STATUS,
    MIN_ZONE_PIXELS,
)
from suitability.zones import (  # noqa: E402
    label_zones,
    polygons_per_zone,
    validate_geometries,
)

# ---------------------------------------------------------------------------
# Constants (spec docs_production_build_spec.md section 11)
# ---------------------------------------------------------------------------
YEARS = (2022, 2023, 2024, 2025, 2026)
FOCUS_YEAR = 2026
PX_AREA_M2 = 900.0        # 30 m x 30 m reference grid
M2_PER_HA = 10_000.0
HA_PER_PX = PX_AREA_M2 / M2_PER_HA          # 0.09
TREES_PER_HA_PRIMARY = 1_000                # locked decision 2
SENSITIVITY_DENSITIES = (400, 2_500)        # trees/ha, locked decision 2
# trees per plantable px = 0.09 ha/px * 1000 trees/ha = 90 (0.09*1000; the
# spec text's "9" is a decimal slip - see DEVIATIONS[0]).
TREES_PER_PX_PRIMARY = int(round(TREES_PER_HA_PRIMARY * HA_PER_PX))   # 90
VEG_COVER_THRESHOLD = 0.30                  # locked decision 1

OUT_ROOT = PROJECT / "data" / "processed" / "phase8_tree_requirement"
RASTERS_DIR = OUT_ROOT / "rasters"
TABLES_DIR = OUT_ROOT / "tables"
VECTORS_DIR = OUT_ROOT / "vectors"
FIGURES_DIR = OUT_ROOT / "figures"
MANIFEST_JSON = OUT_ROOT / "phase8_manifest.json"
PIPELINE_RECORD_JSON = OUT_ROOT / "phase8_pipeline_record.json"

P7_ROOT = PROJECT / "data" / "processed" / "phase7_production"
P7_RASTERS = P7_ROOT / "rasters"
P7_ZONES = P7_ROOT / "zones"
P7_TABLES = P7_ROOT / "tables"
P6_RASTERS = PROJECT / "data" / "processed" / "phase6_production" / "rasters"
P4_FEATURES = PROJECT / "data" / "processed" / "phase4" / "features"
P3_MASKS = PROJECT / "data" / "processed" / "phase3" / "masks"
P6_MANIFEST = PROJECT / "data" / "processed" / "phase6_production" / "phase6_production_manifest.json"
P7_MANIFEST = P7_ROOT / "phase7_production_manifest.json"
SPEC_PATH = PROJECT / "docs_production_build_spec.md"

PLANT_NODATA = 255          # available_planting_space nodata (= out of domain)
TREES_NODATA = -1           # recommended_trees nodata (= out of domain)
SEVERITY_HIGH_CLASS = 2     # phase6 3-class encoding: 0 Low / 1 Moderate / 2 High

PRIORITY_COLORS = {"High": "#d73027", "Medium": "#fc8d59", "Low": "#fee08b"}

DEVIATIONS: List[str] = [
    "Tree arithmetic: the task text states 'per-pixel trees = 9 (0.09 ha x "
    "1000)' and gate G4 'recommended_trees == plantable_px x 9', but "
    "0.09 ha/px x 1000 trees/ha = 90 trees/px, and the gate's own "
    "sensitivity rule 'px x 0.09 x density' also yields 90 at 1000/ha. "
    "The '9' is a decimal slip; the self-consistent rule trees = "
    "plantable_ha x density (90 trees/px at 1,000/ha) is used everywhere, "
    "matching locked decisions 11.2 (1,000 trees/ha, ~3.2 m spacing) and "
    "11.4 (Recommended trees = planting ha x 1000). At 400/2,500 trees/ha: "
    "36 / 225 trees per px.",
    "Spec section 11.1 'suitability_class >= 4 priority zone pixels' is read "
    "in the spec's own 1-indexed class numbering, i.e. 0-indexed class >= 3 "
    "(High + Very High) - exactly the Phase 7 priority-zone definition "
    "(same convention as spec section 10: 'classes 4-5 in 1-indexed terms "
    "== class >= 3 (0-indexed)'). The literal 0-indexed reading (class == 4 "
    "Very High) is impossible: the production suitability tables show 0 "
    "Very High pixels in all five years, which would zero out every Phase 8 "
    "product.",
    "'Priority zone pixels' = pixels of the KEPT Phase 7 priority zones "
    "(8-connected class >= 3 clusters after the frozen MIN_ZONE_PIXELS=10 "
    "noise rule). Class >= 3 pixels in dropped noise clusters are not part "
    "of any zone and are excluded from planting space (2022: 575 px, 2023: "
    "1116 px, 2024: 289 px, 2025: 85 px, 2026: 85 px removed at this stage "
    "before the other filters).",
    "'landuse eligible' = LANDUSE_RULE_STATUS != 'DISCOURAGED' (industrial "
    "class 5 and retail class 7 excluded). Landuse nodata (255) receives the "
    "frozen neutral treatment of class 0 (absence of an OSM tag is not "
    "evidence of ineligibility) and is ELIGIBLE. LANDUSE_ELIGIBILITY scores "
    "are imported and recorded in the manifest; the priority-zone pixels sit "
    "~99% in class 6 (residential, the only ELIGIBLE class), so the results "
    "are insensitive to this threshold choice.",
    "No minimum cluster size is applied when polygonizing plantable clusters "
    "(spec section 11 states none); every 8-connected plantable cluster "
    "becomes one recommended-plantation polygon. Cluster counts are recorded "
    "per year.",
    "Per-zone 'Current vegetation (%)' is the mean vegetation_cover over ALL "
    "zone pixels (the zone's existing vegetation state), not only over "
    "plantable pixels.",
    "Priority thirds: with n zones ordered by priority_ranking "
    "(mean suitability desc), ranks 1..ceil(n/3) = High, the next ceil(n/3) "
    "ranks = Medium, the remainder = Low (e.g. n=11 -> 4/4/3; n=69 -> "
    "23/23/23).",
    "available_planting_space nodata (255) marks cells OUTSIDE the per-year "
    "analysis domain (exclusion_mask == 1); inside the domain 0 = not "
    "plantable, 1 = plantable. recommended_trees uses -1 outside the domain, "
    "0 inside not plantable, 90 (= 0.09 ha x 1000 trees/ha) on plantable px.",
]


def log(msg: str) -> None:
    print(f"[P8] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _convert(obj):
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert(obj), f, indent=2, default=str)


def read_band(path: Path) -> Tuple[np.ndarray, float]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.profile.get("nodata")
        nodata = float(nodata) if nodata is not None else np.nan
    return arr, nodata


# ---------------------------------------------------------------------------
# Core per-year computation
# ---------------------------------------------------------------------------
def landuse_eligible_mask(landuse: np.ndarray) -> np.ndarray:
    """Boolean eligibility per frozen LANDUSE_RULE_STATUS (nodata -> neutral)."""
    eligible = np.ones(256, dtype=bool)
    for code, status in LANDUSE_RULE_STATUS.items():
        if status == "DISCOURAGED":
            eligible[int(code)] = False
    # LANDUSE_NODATA (255) keeps the neutral class-0 treatment: eligible.
    return eligible[landuse.astype(np.int64)]


def compute_year(year: int, landuse: np.ndarray, profile: Dict,
                 input_hashes: Dict[str, str]) -> Dict:
    """Full Phase 8 computation for one year."""
    cls_path = P7_RASTERS / f"suitability_class_{year}.tif"
    excl_path = P7_RASTERS / f"exclusion_mask_{year}.tif"
    suit_path = P7_RASTERS / f"suitability_{year}.tif"
    sev_path = P6_RASTERS / f"severity_{year}.tif"
    sevscore_path = P6_RASTERS / f"severity_score_{year}.tif"
    veg_path = P4_FEATURES / f"vegetation_cover_{year}_30m.tif"
    for p in (cls_path, excl_path, suit_path, sev_path, sevscore_path, veg_path):
        input_hashes[str(p.relative_to(PROJECT))] = sha256(p)

    cls, _ = read_band(cls_path)
    excl, _ = read_band(excl_path)
    suitability, _ = read_band(suit_path)
    sev, _ = read_band(sev_path)
    sevscore, _ = read_band(sevscore_path)
    veg, veg_nd = read_band(veg_path)

    domain = excl == 0
    if not domain.any():
        raise AssertionError(f"{year}: empty analysis domain")

    # Priority zones re-derived with the frozen Phase 7 rule (8-conn,
    # MIN_ZONE_PIXELS) on the written class raster -> IDs match Phase 7.
    zone_labelled, n_zones = label_zones((cls >= 3) & domain)
    ranking = pd.read_csv(P7_TABLES / f"priority_ranking_{year}.csv")
    p7_stats = pd.read_csv(P7_TABLES / f"priority_zone_statistics_{year}.csv")
    input_hashes[str((P7_TABLES / f"priority_ranking_{year}.csv").relative_to(PROJECT))] = sha256(
        P7_TABLES / f"priority_ranking_{year}.csv")
    input_hashes[str((P7_TABLES / f"priority_zone_statistics_{year}.csv").relative_to(PROJECT))] = sha256(
        P7_TABLES / f"priority_zone_statistics_{year}.csv")
    input_hashes[str((P7_ZONES / f"priority_zones_{year}.geojson").relative_to(PROJECT))] = sha256(
        P7_ZONES / f"priority_zones_{year}.geojson")

    # Cross-check re-derived zones against the Phase 7 tables (identity).
    if int(n_zones) != len(p7_stats):
        raise AssertionError(f"{year}: zone count mismatch vs Phase 7 tables")
    merged = p7_stats[["zone_id", "pixel_count"]].merge(
        pd.DataFrame({"zone_id": [int(z) for z in np.unique(zone_labelled[zone_labelled > 0])],
                      "rederived_px": [int((zone_labelled == z).sum())
                                       for z in np.unique(zone_labelled[zone_labelled > 0])]}),
        on="zone_id", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all() or not (merged["pixel_count"] == merged["rederived_px"]).all():
        raise AssertionError(f"{year}: re-derived zones do not match Phase 7 pixel counts")

    veg_ok = np.isfinite(veg) & (veg != veg_nd) & (veg < VEG_COVER_THRESHOLD)
    eligible_lu = landuse_eligible_mask(landuse)

    # Locked decision 1 (all conditions explicit).
    plantable = ((cls >= 3) & (zone_labelled > 0) & domain
                 & (excl == 0) & eligible_lu & veg_ok)

    # ---- rasters ------------------------------------------------------
    space_raster = np.where(domain, plantable.astype(np.uint8), PLANT_NODATA).astype(np.uint8)
    space_profile = profile.copy()
    space_profile.update({"dtype": "uint8", "count": 1, "nodata": PLANT_NODATA,
                          "compress": "lzw"})
    space_path = RASTERS_DIR / f"available_planting_space_{year}.tif"
    with rasterio.open(space_path, "w", **space_profile) as dst:
        dst.write(space_raster, 1)

    trees_raster = np.where(domain, plantable.astype(np.int16) * TREES_PER_PX_PRIMARY,
                            TREES_NODATA).astype(np.int16)
    trees_profile = profile.copy()
    trees_profile.update({"dtype": "int16", "count": 1, "nodata": TREES_NODATA,
                          "compress": "lzw"})
    trees_path = RASTERS_DIR / f"recommended_trees_{year}.tif"
    with rasterio.open(trees_path, "w", **trees_profile) as dst:
        dst.write(trees_raster, 1)

    # ---- per-zone table ------------------------------------------------
    n_ranked = len(ranking)
    third = int(np.ceil(n_ranked / 3.0))

    def priority_for_rank(rank: int) -> str:
        if rank <= third:
            return "High"
        if rank <= 2 * third:
            return "Medium"
        return "Low"

    rank_order = ranking.sort_values("rank")
    zone_rows: List[Dict] = []
    zone_priority: Dict[int, str] = {}
    for rec in rank_order.itertuples():
        zid = int(rec.zone_id)
        zmask = zone_labelled == zid
        n_px = int(zmask.sum())
        z_plant = int((zmask & plantable).sum())
        zone_priority[zid] = priority_for_rank(int(rec.rank))
        zone_rows.append({
            "year": year,
            "rank": int(rec.rank),
            "zone_id": zid,
            "area_ha": round(n_px * HA_PER_PX, 3),
            "pixel_count": n_px,
            "current_vegetation_pct": round(float(veg[zmask].mean()) * 100.0, 2),
            "mean_severity_score": round(float(sevscore[zmask].mean()), 4),
            "high_severity_pct": round(float((sev[zmask] == SEVERITY_HIGH_CLASS).mean()) * 100.0, 2),
            "mean_suitability": float(rec.mean_suitability),
            "plantable_px": z_plant,
            "plantable_ha": round(z_plant * HA_PER_PX, 3),
            "recommended_trees": z_plant * TREES_PER_PX_PRIMARY,
            "priority": zone_priority[zid],
            "centroid_lon": float(rec.centroid_lon),
            "centroid_lat": float(rec.centroid_lat),
        })
    zone_df = pd.DataFrame(zone_rows)
    zone_df.to_csv(TABLES_DIR / f"tree_requirement_by_zone_{year}.csv", index=False)

    # ---- citywide summary (primary + sensitivity densities) ------------
    total_zone_px = int(zone_df["pixel_count"].sum())
    total_zone_ha = total_zone_px * HA_PER_PX
    plantable_px = int(plantable.sum())
    plantable_ha = plantable_px * HA_PER_PX
    summary_rows = []
    for density in (TREES_PER_HA_PRIMARY, *SENSITIVITY_DENSITIES):
        trees = int(round(plantable_ha * density))
        summary_rows.append({
            "year": year,
            "density_trees_per_ha": density,
            "is_primary_density": density == TREES_PER_HA_PRIMARY,
            "n_zones": int(n_zones),
            "total_zone_area_ha": round(total_zone_ha, 3),
            "plantable_px": plantable_px,
            "plantable_ha": round(plantable_ha, 3),
            "recommended_trees": trees,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / f"tree_requirement_summary_{year}.csv", index=False)

    # ---- recommended-plantation polygons (8-conn clusters) -------------
    cluster_labelled, n_clusters = ndimage.label(
        plantable, structure=np.ones((3, 3), dtype=int))
    transform = profile["transform"]
    polygons = polygons_per_zone(cluster_labelled, transform) if n_clusters else {}
    validation = validate_geometries(polygons)

    cluster_rows: List[Dict] = []
    cluster_geoms: List[object] = []
    for cid in sorted(int(c) for c in np.unique(cluster_labelled[cluster_labelled > 0])):
        cmask = cluster_labelled == cid
        n_px = int(cmask.sum())
        zids = np.unique(zone_labelled[cmask])
        zids = zids[zids > 0]
        if zids.size != 1:
            raise AssertionError(f"{year}: cluster {cid} spans {zids.size} zones")
        zid = int(zids[0])
        cluster_rows.append({
            "zone_id": zid,
            "area_ha": round(n_px * HA_PER_PX, 3),
            "trees": n_px * TREES_PER_PX_PRIMARY,
            "mean_suitability": round(float(suitability[cmask].mean()), 4),
            "mean_severity_score": round(float(sevscore[cmask].mean()), 4),
            "priority": zone_priority[zid],
        })
        cluster_geoms.append(polygons[cid])
    cluster_df = pd.DataFrame(cluster_rows)
    if len(cluster_df):
        gdf = gpd.GeoDataFrame(cluster_df, geometry=cluster_geoms, crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(
            {c: pd.Series(dtype="float64") for c in
             ("zone_id", "area_ha", "trees", "mean_suitability",
              "mean_severity_score", "priority")},
            geometry=gpd.GeoSeries(dtype="geometry"), crs="EPSG:4326")
    gdf.to_file(VECTORS_DIR / f"recommended_plantations_{year}.geojson", driver="GeoJSON")

    points_path = None
    if year == FOCUS_YEAR:
        pts = cluster_df.copy()
        pts["geometry"] = [g.centroid for g in cluster_geoms]
        pts_gdf = gpd.GeoDataFrame(pts, geometry="geometry", crs="EPSG:4326")
        pts_gdf.to_file(VECTORS_DIR / "recommended_locations_2026.geojson", driver="GeoJSON")
        points_path = VECTORS_DIR / "recommended_locations_2026.geojson"

    noise_zone_px = int(((cls >= 3) & domain & (zone_labelled == 0)).sum())
    return {
        "year": year,
        "domain_px": int(domain.sum()),
        "n_zones": int(n_zones),
        "zone_pixel_sum": total_zone_px,
        "noise_zone_px_excluded": noise_zone_px,
        "plantable_px": plantable_px,
        "plantable_ha": round(plantable_ha, 3),
        "recommended_trees_primary": plantable_px * TREES_PER_PX_PRIMARY,
        "n_clusters": int(n_clusters),
        "cluster_px_sum": int(round(float(cluster_df["area_ha"].sum()) / HA_PER_PX)),
        "cluster_trees_sum": int(cluster_df["trees"].sum()) if len(cluster_df) else 0,
        "cluster_geom_valid": bool(validation["all_valid"]),
        "zone_df": zone_df,
        "summary_df": summary_df,
        "cluster_df": cluster_df,
        "plantable": plantable,
        "domain": domain,
        "zone_labelled": zone_labelled,
        "suitability": suitability,
        "sevscore": sevscore,
        "priority": zone_priority,
        "space_path": space_path,
        "trees_path": trees_path,
        "points_path": points_path,
        "veg_threshold_excluded_px": int(((zone_labelled > 0) & domain & eligible_lu
                                          & ~(veg < VEG_COVER_THRESHOLD)).sum()),
        "landuse_excluded_px": int(((zone_labelled > 0) & domain & ~eligible_lu).sum()),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def map_figure_2026(result: Dict, path: Path) -> None:
    """Severity basemap + plantable polygons colored by priority (2026)."""
    sevscore = result["sevscore"]
    plantable = result["plantable"]
    shown = np.where(np.isfinite(sevscore), sevscore, np.nan)
    cluster_gdf = gpd.read_file(VECTORS_DIR / f"recommended_plantations_{FOCUS_YEAR}.geojson")

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(shown, cmap="hot_r", vmin=0.0, vmax=2.0, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7, label="UHI severity score (0-2)")
    for prio in ("High", "Medium", "Low"):
        sub = cluster_gdf[cluster_gdf["priority"] == prio]
        if len(sub):
            sub.plot(ax=ax, facecolor=PRIORITY_COLORS[prio], edgecolor="black",
                     linewidth=0.4, alpha=0.85, label=f"{prio} priority")
    ax.set_title(f"Phase 8 {FOCUS_YEAR}: recommended plantations on Phase 7 priority zones\n"
                 f"(plantable px where suitability class >= 3, landuse eligible, "
                 f"vegetation cover < {VEG_COVER_THRESHOLD:.2f})")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    leg = ax.legend(loc="lower left", title="Zone priority")
    leg.set_zorder(5)
    n_trees = result["recommended_trees_primary"]
    ax.annotate(f"Citywide recommended trees ({FOCUS_YEAR}): {n_trees:,}\n"
                f"plantable area: {result['plantable_ha']:.2f} ha "
                f"({result['n_clusters']} plantation clusters)",
                xy=(0.99, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def five_year_panel_figure(per_year: Dict[int, Dict], path: Path) -> None:
    years = list(YEARS)
    ha = [per_year[y]["plantable_ha"] for y in years]
    trees = [per_year[y]["recommended_trees_primary"] for y in years]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bars = ax1.bar([str(y) for y in years], ha, color="#2c7fb8")
    ax1.bar_label(bars, fmt="%.2f", fontsize=9)
    ax1.set_ylabel("Plantable area (ha)")
    ax1.set_title("Available planting space per year\n(Phase 7 priority zones, ha = px x 900/1e4)")
    bars = ax2.bar([str(y) for y in years], trees, color="#41ab5d")
    ax2.bar_label(bars, fmt="%d", fontsize=9)
    ax2.set_ylabel("Recommended trees (1,000 trees/ha)")
    ax2.set_title(f"Recommended trees per year ({TREES_PER_HA_PRIMARY:,} trees/ha;\n"
                  f"{TREES_PER_PX_PRIMARY} trees per plantable px)")
    for ax in (ax1, ax2):
        ax.set_xlabel("year")
    fig.suptitle("Phase 8 tree requirement - 5-year summary")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def top10_zone_figure_2026(result: Dict, path: Path) -> None:
    zone_df = result["zone_df"].sort_values("recommended_trees", ascending=False).head(10)
    zone_df = zone_df.iloc[::-1]  # largest on top
    labels = [f"zone {int(r.zone_id)} ({r.priority})" for r in zone_df.itertuples()]
    colors = [PRIORITY_COLORS[p] for p in zone_df["priority"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, zone_df["recommended_trees"], color=colors, edgecolor="black",
                   linewidth=0.4)
    ax.bar_label(bars, fmt="%d", fontsize=9)
    ax.set_xlabel("Recommended trees (1,000 trees/ha)")
    ax.set_title(f"Phase 8 {FOCUS_YEAR}: top-10 priority zones by recommended trees\n"
                 f"(total {result['recommended_trees_primary']:,} trees citywide)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run(years: Tuple[int, ...], skip_figures: bool) -> Dict:
    for d in (RASTERS_DIR, TABLES_DIR, VECTORS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    record: Dict = {
        "phase": 8, "variant": "production", "project": "GreenGrid-AI",
        "project_root": str(PROJECT), "started_utc": _now(),
        "years": list(years), "steps": {}, "deviations": list(DEVIATIONS),
    }
    steps: Dict[str, Dict] = {}
    input_hashes: Dict[str, str] = {}

    with rasterio.open(REFERENCE_RASTER) as ref:
        profile = ref.profile.copy()
    log(f"Reference grid: {profile['width']}x{profile['height']} {ref.crs}")
    input_hashes[str(Path(REFERENCE_RASTER).relative_to(PROJECT))] = sha256(Path(REFERENCE_RASTER))

    landuse, _ = read_band(P3_MASKS / "landuse_raster_30m.tif")
    input_hashes[str((P3_MASKS / "landuse_raster_30m.tif").relative_to(PROJECT))] = sha256(
        P3_MASKS / "landuse_raster_30m.tif")

    per_year: Dict[int, Dict] = {}
    for year in years:
        t0 = time.time()
        log(f"--- {year}")
        res = compute_year(year, landuse, profile, input_hashes)
        per_year[year] = res
        steps[str(year)] = {
            "status": "success",
            "elapsed_s": round(time.time() - t0, 3),
            "domain_px": res["domain_px"],
            "n_zones": res["n_zones"],
            "total_zone_area_ha": round(res["zone_pixel_sum"] * HA_PER_PX, 3),
            "noise_zone_px_excluded": res["noise_zone_px_excluded"],
            "landuse_excluded_zone_px": res["landuse_excluded_px"],
            "veg_threshold_excluded_zone_px": res["veg_threshold_excluded_px"],
            "plantable_px": res["plantable_px"],
            "plantable_ha": res["plantable_ha"],
            "recommended_trees_1000_per_ha": res["recommended_trees_primary"],
            "n_plantation_clusters": res["n_clusters"],
            "cluster_trees_sum": res["cluster_trees_sum"],
            "cluster_geometry_valid": res["cluster_geom_valid"],
        }
        log(f"--- {year}: zones={res['n_zones']} plantable={res['plantable_px']:,} px "
            f"({res['plantable_ha']:.2f} ha) trees={res['recommended_trees_primary']:,} "
            f"clusters={res['n_clusters']} ({steps[str(year)]['elapsed_s']}s)")

    # ---- 5-year citywide table ------------------------------------------
    citywide_rows = []
    for year in years:
        res = per_year[year]
        px = res["plantable_px"]
        row = {
            "year": year,
            "n_zones": res["n_zones"],
            "total_zone_area_ha": round(res["zone_pixel_sum"] * HA_PER_PX, 3),
            "plantable_px": px,
            "plantable_ha": round(px * HA_PER_PX, 3),
            "recommended_trees_1000_per_ha": px * TREES_PER_PX_PRIMARY,
            "recommended_trees_400_per_ha": int(round(px * HA_PER_PX * 400)),
            "recommended_trees_2500_per_ha": int(round(px * HA_PER_PX * 2500)),
        }
        citywide_rows.append(row)
    citywide_df = pd.DataFrame(citywide_rows)
    citywide_df.to_csv(TABLES_DIR / "tree_requirement_citywide_5yr.csv", index=False)

    # ---- figures ----------------------------------------------------------
    if not skip_figures:
        t0 = time.time()
        map_figure_2026(per_year[FOCUS_YEAR],
                        FIGURES_DIR / f"recommended_plantations_{FOCUS_YEAR}_map.png")
        five_year_panel_figure(per_year, FIGURES_DIR / "plantable_ha_trees_5yr.png")
        top10_zone_figure_2026(per_year[FOCUS_YEAR],
                               FIGURES_DIR / f"top10_zone_trees_{FOCUS_YEAR}.png")
        steps["figures"] = {"status": "success",
                            "elapsed_s": round(time.time() - t0, 3)}

    # ---- manifest + pipeline record ---------------------------------------
    with open(P6_MANIFEST) as f:
        p6_manifest = json.load(f)
    with open(P7_MANIFEST) as f:
        p7_manifest = json.load(f)

    outputs: Dict[str, str] = {}
    output_hashes: Dict[str, str] = {}
    for sub in ("rasters", "tables", "vectors", "figures"):
        for p in sorted((OUT_ROOT / sub).rglob("*")):
            if p.is_file():
                outputs[str(p.relative_to(OUT_ROOT))] = str(p.relative_to(PROJECT))
                output_hashes[str(p.relative_to(OUT_ROOT))] = sha256(p)

    manifest = {
        "stage": "phase8_tree_requirement",
        "created_at_utc": _now(),
        "spec": {"path": "docs_production_build_spec.md",
                 "sha256": sha256(SPEC_PATH),
                 "version": "2026-10-03 production build spec, section 11"},
        "model_lineage": {
            "phase5_model_id": p6_manifest.get("model_id"),
            "phase5_model_sha256": p6_manifest.get("model_sha256"),
            "phase5_schema": p6_manifest.get("schema"),
            "phase6_manifest": str(P6_MANIFEST.relative_to(PROJECT)),
            "phase7_manifest": str(P7_MANIFEST.relative_to(PROJECT)),
            "chain": ("phase5 production 3-class XGBoost -> phase6_production "
                      "severity rasters -> phase7_production suitability + "
                      "priority zones -> phase8 tree requirement"),
        },
        "assumptions": {
            "planting_space_rule": (
                "plantable = Phase 7 priority-zone pixels (suitability class >= 3 "
                "0-indexed, kept 8-connected zones, MIN_ZONE_PIXELS=10) AND NOT "
                "exclusion_mask AND landuse eligible AND vegetation_cover < 0.30"),
            "class_numbering_note": (
                "spec 11.1 'suitability_class >= 4' read as 1-indexed (0-indexed "
                ">= 3); see deviations[0]"),
            "tree_density_trees_per_ha": TREES_PER_HA_PRIMARY,
            "trees_per_plantable_px": TREES_PER_PX_PRIMARY,
            "sensitivity_densities_trees_per_ha": list(SENSITIVITY_DENSITIES),
            "vegetation_cover_threshold": VEG_COVER_THRESHOLD,
            "landuse_eligibility_source": {
                "module": "src.suitability.config",
                "imported": ["LANDUSE_ELIGIBILITY", "LANDUSE_RULE_STATUS"],
                "rule": "eligible = LANDUSE_RULE_STATUS != 'DISCOURAGED'; "
                        "nodata (255) neutral-eligible (same as class 0)",
                "excluded_classes": [int(c) for c, s in LANDUSE_RULE_STATUS.items()
                                     if s == "DISCOURAGED"],
                "eligibility_scores": {int(k): v for k, v in LANDUSE_ELIGIBILITY.items()},
                "nodata_code": LANDUSE_NODATA,
            },
            "priority_rule": (
                "Phase 7 priority_ranking order (mean suitability desc); "
                "High = ranks 1..ceil(n/3), Medium = next ceil(n/3), Low = rest"),
            "area_rule": "area_ha = pixel_count * 900 m2 / 1e4 (px = 30 m x 30 m)",
            "focus_year": FOCUS_YEAR,
            "plantation_clusters": ("all 8-connected plantable clusters polygonized "
                                    "(no min-size rule in spec 11); EPSG:4326"),
            "raster_encodings": {
                "available_planting_space": ("uint8; 1 = plantable, 0 = in-domain not "
                                             "plantable, 255 = nodata (outside "
                                             "analysis domain)"),
                "recommended_trees": ("int16; 90 = plantable px (0.09 ha x "
                                      "1000 trees/ha), "
                                      "0 = in-domain not plantable, -1 = nodata"),
            },
            "severity_encoding": ("phase6 3-class: 0 Low / 1 Moderate / 2 High; "
                                  "severity_score in [0,2]"),
        },
        "inputs": [{"path": k, "sha256": v} for k, v in sorted(input_hashes.items())],
        "outputs": [{"path": k, "sha256": v} for k, v in sorted(output_hashes.items())],
        "per_year": {str(y): steps[str(y)] for y in years},
        "citywide_5yr": _convert(citywide_df.to_dict(orient="records")),
        "deviations": record["deviations"],
        "terminology_note": (
            "Tree numbers are PLANNING ESTIMATES on potential plantation "
            "suitability - a relative, decision-support ranking. They are NOT a "
            "statement of legal availability, land ownership, field-verified "
            "plantability, or planting survival probability."),
    }
    save_json(manifest, MANIFEST_JSON)
    outputs["phase8_manifest.json"] = str(MANIFEST_JSON.relative_to(PROJECT))
    outputs["phase8_pipeline_record.json"] = str(PIPELINE_RECORD_JSON.relative_to(PROJECT))

    record["steps"].update(steps)
    record["citywide_5yr"] = _convert(citywide_df.to_dict(orient="records"))
    record["outputs"] = outputs
    record["finished_utc"] = _now()
    record["total_elapsed_s"] = round(
        sum(s.get("elapsed_s", 0) for s in steps.values() if isinstance(s, dict)), 3)
    record["status"] = "success"
    record["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "rasterio": rasterio.__version__, "geopandas": gpd.__version__,
        "matplotlib": matplotlib.__version__,
        "scipy": __import__("scipy").__version__,
    }
    save_json(record, PIPELINE_RECORD_JSON)
    log(f"Manifest: {MANIFEST_JSON}")
    log(f"Pipeline record: {PIPELINE_RECORD_JSON}")
    return {"record": record, "manifest": manifest, "per_year": per_year,
            "citywide_df": citywide_df}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 8 tree-requirement build.")
    parser.add_argument("--years", default=",".join(str(y) for y in YEARS),
                        help="comma-separated years")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)
    years = tuple(int(y) for y in args.years.split(","))

    try:
        result = run(years, args.skip_figures)
        print(json.dumps(_convert(result["record"]), indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Phase 8 pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
