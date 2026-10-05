"""V2 Phase 8 build — tree requirement estimation (see package docstring).

Usage:
    PYTHONPATH=v2/src .venv/Scripts/python.exe -m v2.phase8.build \
        [--years 2022,2023,...] [--out v2/data/phase8]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from ..common import (
    GRID_FILE_DEFAULT,
    PROJECT_ROOT,
    YEARS,
    Grid,
    dump_json,
    sha256_file,
)
from ..phase4.assemble_features import load_phase3_year, load_static
from ..phase7.build import read_band
from ..phase7.suitability import (
    MIN_ZONE_PIXELS,
    M2_PER_HA,
    PX_AREA_M2,
    ZONE_MIN_CLASS,
    label_zones,
    polygons_per_zone,
    validate_geometries,
)

FOCUS_YEAR = 2026
HA_PER_PX = PX_AREA_M2 / M2_PER_HA          # 0.09
TREES_PER_HA_PRIMARY = 1_000
SENSITIVITY_DENSITIES = (400, 2_500)
TREES_PER_PX_PRIMARY = int(round(TREES_PER_HA_PRIMARY * HA_PER_PX))   # 90
VEG_COVER_THRESHOLD = 0.30
PLANT_NODATA = 255
TREES_NODATA = -1
SEVERITY_HIGH_CLASS = 2

# eligibility rule extracted verbatim from the V1 runner:
# eligible = LANDUSE_RULE_STATUS != "DISCOURAGED"  (v1/src/suitability/config.py)
DISCOURAGED_LANDUSE_CODES = (5, 7)        # industrial, retail

# v2_constrained is the PRIMARY result (constraint-aware); v1_parity is the
# V1-comparable BASELINE. Exclusions are applied AFTER scoring, on the same
# per-year normalization as the full valid domain — scenario differences are
# exclusion effects, not re-normalization effects.
SCENARIOS = ("v2_constrained", "v1_parity")
PRIMARY_SCENARIO = "v2_constrained"
BASELINE_SCENARIO = "v1_parity"
PRIMARY_SNAPSHOT_YEAR = 2026              # primary planning snapshot (NOT a
                                          # "hottest year" claim — relative,
                                          # per-year-normalized products)

# excluded-cause attribution precedence (each excluded px counted ONCE)
EXCLUSION_PRECEDENCE = ("constraint", "landuse_ineligible", "veg_threshold")
DENSITY_LABEL = ("planning-density scenarios (not scientifically optimal; "
                 "the optimal density will be evaluated after Phase 9 using "
                 "cooling predictions)")
PRODUCT_LABEL = ("relative plantation-suitability decision-support ranking "
                 "(Phase 7); tree numbers are planning estimates on that "
                 "ranking, not planting-success predictions")

PHASE6_DIR_DEFAULT = PROJECT_ROOT / "data" / "phase6"
PHASE7_DIR_DEFAULT = PROJECT_ROOT / "data" / "phase7"
PHASE3_ROOT_DEFAULT = PROJECT_ROOT / "data" / "phase3"
OUT_DEFAULT = PROJECT_ROOT / "data" / "phase8"

DEVIATIONS = [
    "Tree arithmetic: V1 spec text says '9 trees/px' but 0.09 ha/px x 1000 "
    "trees/ha = 90 trees/px; V1's own gate formula (px x 0.09 x density) also "
    "yields 90 at 1,000/ha. The '9' is a documented decimal slip; 90 is used "
    "(36 at 400/ha, 225 at 2,500/ha) — identical to the V1 production build.",
    "Spec 11.1 'suitability_class >= 4' is read 1-indexed = 0-indexed class "
    ">= 3 (High + Very High), exactly the Phase 7 priority-zone definition; "
    "a literal 0-indexed reading (class 4 only) is impossible (0 Very High "
    "pixels in all V2 years too).",
    "'Priority zone pixels' = pixels of the KEPT Phase 7 zones (8-connected, "
    "MIN_ZONE_PIXELS=10); class >= 3 pixels in dropped noise clusters are not "
    "planting space.",
    "landuse eligible = LANDUSE_RULE_STATUS != 'DISCOURAGED' -> excluded "
    "codes {5 industrial, 7 retail}; nodata (255) neutral-eligible (class 0 "
    "treatment, frozen V1 decision).",
    "No minimum cluster size when polygonizing plantable clusters (spec 11 "
    "states none); every 8-connected plantable cluster is one recommended "
    "plantation polygon (EPSG:4326).",
    "Per-zone 'Current vegetation (%)' is the mean vegetation_cover over ALL "
    "zone pixels, not only plantable pixels (V1 rule).",
    "Priority thirds: ranks 1..ceil(n/3) High, next ceil(n/3) Medium, rest "
    "Low from the Phase 7 priority_ranking order (mean suitability desc).",
    "Figures (matplotlib) are SKIPPED in V2 per the phase6/7 briefs; every "
    "numeric/vector product is still produced.",
    "Zone ids are taken from the Phase 7 priority_zone_ids rasters (written "
    "by v2.phase7.build); identity with the Phase 7 zone statistics tables is "
    "asserted per year (same guarantee as V1's re-derivation cross-check).",
]


def log(msg: str) -> None:
    print(f"[P8] {msg}", flush=True)


def landuse_eligible_mask(landuse: np.ndarray) -> np.ndarray:
    """Boolean eligibility per the frozen rule; nodata (255) neutral-eligible."""
    eligible = np.ones(256, dtype=bool)
    for code in DISCOURAGED_LANDUSE_CODES:
        eligible[int(code)] = False
    return eligible[landuse.astype(np.int64)]


def build_cause_grid(veg_ok: np.ndarray, eligible_lu: np.ndarray,
                     attributed: np.ndarray | None) -> np.ndarray:
    """Disjoint excluded-cause grid (uint8): 0 none / 1 constraint /
    2 landuse_ineligible / 3 veg_threshold. LOWEST precedence applied first
    so constraint > landuse > veg wins on overlaps — every excluded pixel is
    counted exactly once (no double counting). ``attributed`` is the REUSED
    Phase 7 constraint raster (>0 = some constraint class); pass None for the
    v1_parity baseline (no constraint exclusion)."""
    cause = np.zeros(veg_ok.shape, dtype=np.uint8)
    cause[~veg_ok] = 3
    cause[~eligible_lu] = 2
    if attributed is not None:
        cause[attributed > 0] = 1
    return cause


def preflight(years, phase7_dir, phase6_dir, phase3_root, grid_file):
    # constraint rasters are REUSED from Phase 7 (water > buildings >
    # road-surfaces precedence already established there) — never re-derived
    # from the Phase 2 raw GeoJSONs here.
    for name in ("constraint_water_30m.tif", "constraint_buildings_30m.tif",
                 "constraint_road_surfaces_30m.tif",
                 "constraint_attributed_30m.tif"):
        if not (phase7_dir / "constraints_rasters" / name).exists():
            raise FileNotFoundError(
                f"missing reused Phase 7 constraint raster: {name}")
    for scenario in SCENARIOS:
        for y in years:
            rdir = phase7_dir / scenario / "rasters"
            tdir = phase7_dir / scenario / "tables"
            for name in (f"suitability_class_{scenario}_{y}.tif",
                         f"suitability_{scenario}_{y}.tif",
                         f"exclusion_mask_{scenario}_{y}.tif",
                         f"priority_zone_ids_{scenario}_{y}.tif"):
                if not (rdir / name).exists():
                    raise FileNotFoundError(f"missing phase7 input: {rdir / name}")
            for name in (f"priority_ranking_{scenario}_{y}.csv",
                         f"priority_zone_statistics_{scenario}_{y}.csv"):
                if not (tdir / name).exists():
                    raise FileNotFoundError(f"missing phase7 table: {tdir / name}")
            for name in (f"severity_{y}.tif", f"severity_score_{y}.tif"):
                if not (phase6_dir / "rasters" / name).exists():
                    raise FileNotFoundError(f"missing phase6 input: {name}")
            if not (phase3_root / str(y) / "vegetation_cover_30m.tif").exists():
                raise FileNotFoundError(f"missing phase3 vegetation_cover for {y}")
    if not (phase3_root / "static" / "landuse_raster_30m.tif").exists():
        raise FileNotFoundError("missing static landuse raster")
    grid = Grid.from_file(Path(grid_file))
    assert (grid.height, grid.width) == (1768, 1874), "authoritative grid mismatch"


def compute_year(year, scenario, landuse, constraints, profile, input_hashes,
                 phase7_dir, phase6_dir, phase3_root, out_root):
    """Full Phase 8 computation for one scenario-year.

    ``constraints`` carries the REUSED Phase 7 constraint rasters
    (``attributed`` int grid with water>buildings>road precedence + the three
    layer masks). Excluded-cause accounting applies the precedence
    constraint > landuse_ineligible > veg_threshold so every excluded zone
    pixel is counted exactly once.
    """
    rdir7 = phase7_dir / scenario / "rasters"
    tdir7 = phase7_dir / scenario / "tables"
    rasters_dir = out_root / scenario / "rasters"
    tables_dir = out_root / scenario / "tables"
    vectors_dir = out_root / scenario / "vectors"
    for d in (rasters_dir, tables_dir, vectors_dir):
        d.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario}_{year}"

    paths = {
        "cls": rdir7 / f"suitability_class_{scenario}_{year}.tif",
        "excl": rdir7 / f"exclusion_mask_{scenario}_{year}.tif",
        "suit": rdir7 / f"suitability_{scenario}_{year}.tif",
        "zids": rdir7 / f"priority_zone_ids_{scenario}_{year}.tif",
        "sev": phase6_dir / "rasters" / f"severity_{year}.tif",
        "sevscore": phase6_dir / "rasters" / f"severity_score_{year}.tif",
        "veg": phase3_root / str(year) / "vegetation_cover_30m.tif",
        "ranking": tdir7 / f"priority_ranking_{scenario}_{year}.csv",
        "zstats": tdir7 / f"priority_zone_statistics_{scenario}_{year}.csv",
    }
    for k, p in paths.items():
        input_hashes[str(p)] = sha256_file(p)

    cls, _ = read_band(paths["cls"])
    excl, _ = read_band(paths["excl"])
    suitability, _ = read_band(paths["suit"])
    zone_labelled, _ = read_band(paths["zids"])
    zone_labelled = zone_labelled.astype(np.int32)
    sev, _ = read_band(paths["sev"])
    sevscore, _ = read_band(paths["sevscore"])
    veg, veg_nd = read_band(paths["veg"])

    domain = excl == 0
    shape = domain.shape
    if not domain.any():
        raise AssertionError(f"{scenario} {year}: empty analysis domain")

    # zone ids must be 0 outside the scenario domain
    assert not (zone_labelled[~domain] > 0).any(), \
        f"{tag}: zone ids present outside scenario domain"

    ranking = pd.read_csv(paths["ranking"])
    p7_stats = pd.read_csv(paths["zstats"])
    # identity cross-check vs Phase 7 tables (same guarantee as V1)
    zpresent = [(int(z), int((zone_labelled == z).sum()))
                for z in np.unique(zone_labelled[zone_labelled > 0])]
    merged = p7_stats[["zone_id", "pixel_count"]].merge(
        pd.DataFrame(zpresent, columns=["zone_id", "raster_px"]),
        on="zone_id", how="outer", indicator=True)
    if not (merged["_merge"] == "both").all() or \
            not (merged["pixel_count"] == merged["raster_px"]).all():
        raise AssertionError(f"{tag}: zone ids raster does not match Phase 7 tables")
    if int(len(zpresent)) != len(ranking):
        raise AssertionError(f"{tag}: ranking/zone count mismatch")

    veg_ok = np.isfinite(veg) & (veg != veg_nd) & (veg < VEG_COVER_THRESHOLD)
    eligible_lu = landuse_eligible_mask(landuse)

    # locked decision 1 (all conditions explicit; zone px are in-domain)
    plantable = ((cls >= ZONE_MIN_CLASS) & (zone_labelled > 0) & domain
                 & (excl == 0) & eligible_lu & veg_ok)

    # ---- excluded-cause grid (disjoint by construction) ----------------------
    attributed = constraints["attributed"] if scenario == PRIMARY_SCENARIO else None
    cause = build_cause_grid(veg_ok, eligible_lu, attributed)
    # consistency: plantable zone px are exactly zone-domain px with no cause
    assert bool(np.array_equal(
        plantable, (zone_labelled > 0) & domain & (cause == 0))), \
        f"{tag}: plantable != zone-domain-without-exclusion-causes"

    # ---- rasters -----------------------------------------------------------
    space = np.where(domain, plantable.astype(np.uint8), PLANT_NODATA).astype(np.uint8)
    space_path = rasters_dir / f"available_planting_space_{tag}.tif"
    prof = profile.copy()
    prof.update({"dtype": "uint8", "count": 1, "nodata": PLANT_NODATA,
                 "compress": "lzw", "crs": profile["crs"],
                 "transform": profile["transform"], "height": profile["height"],
                 "width": profile["width"]})
    with rasterio.open(space_path, "w", **prof) as dst:
        dst.write(space, 1)
        dst.set_band_description(1, "1 plantable / 0 not / 255 outside domain")

    trees = np.where(domain, plantable.astype(np.int16) * TREES_PER_PX_PRIMARY,
                     TREES_NODATA).astype(np.int16)
    trees_path = rasters_dir / f"recommended_trees_{tag}.tif"
    prof2 = dict(prof)
    prof2.update({"dtype": "int16", "nodata": TREES_NODATA})
    with rasterio.open(trees_path, "w", **prof2) as dst:
        dst.write(trees, 1)
        dst.set_band_description(
            1, f"{TREES_PER_PX_PRIMARY} trees per plantable px")

    # ---- per-zone table ------------------------------------------------------
    n_ranked = len(ranking)
    third = int(np.ceil(n_ranked / 3.0))

    def priority_for_rank(rank: int) -> str:
        return "High" if rank <= third else ("Medium" if rank <= 2 * third else "Low")

    zone_rows = []
    zone_priority = {}
    for rec in ranking.sort_values("rank").itertuples():
        zid = int(rec.zone_id)
        zmask = zone_labelled == zid
        n_px = int(zmask.sum())
        z_plant = int((zmask & plantable).sum())
        zone_cause = cause[zmask]
        z_excl = int((zone_cause != 0).sum())
        zone_priority[zid] = priority_for_rank(int(rec.rank))
        zone_rows.append({
            "year": year, "scenario": scenario,
            "rank": int(rec.rank), "zone_id": zid,
            "area_ha": round(n_px * HA_PER_PX, 3),
            "pixel_count": n_px,
            "excluded_px": z_excl,
            "excluded_ha": round(z_excl * HA_PER_PX, 3),
            "excluded_constraint_px": int((zone_cause == 1).sum()),
            "excluded_landuse_px": int((zone_cause == 2).sum()),
            "excluded_veg_px": int((zone_cause == 3).sum()),
            "plantable_px": z_plant,
            "plantable_ha": round(z_plant * HA_PER_PX, 3),
            "current_vegetation_pct": round(float(veg[zmask].mean()) * 100.0, 2),
            "mean_severity_score": round(float(sevscore[zmask].mean()), 4),
            "high_severity_pct": round(
                float((sev[zmask] == SEVERITY_HIGH_CLASS).mean()) * 100.0, 2),
            "mean_suitability": float(rec.mean_suitability),
            "recommended_trees": z_plant * TREES_PER_PX_PRIMARY,
            "recommended_trees_400": int(round(z_plant * HA_PER_PX * 400)),
            "recommended_trees_1000": z_plant * TREES_PER_PX_PRIMARY,
            "recommended_trees_2500": int(round(z_plant * HA_PER_PX * 2500)),
            "priority": zone_priority[zid],
            "centroid_lon": float(rec.centroid_lon),
            "centroid_lat": float(rec.centroid_lat),
        })
    zone_df = pd.DataFrame(zone_rows)
    zone_df.to_csv(tables_dir / f"tree_requirement_by_zone_{tag}.csv", index=False)

    # ---- citywide summary (primary + sensitivity planning densities) ---------
    total_zone_px = int(zone_df["pixel_count"].sum())
    total_zone_ha = total_zone_px * HA_PER_PX
    total_excl_px = int(zone_df["excluded_px"].sum())
    total_excl_ha = total_excl_px * HA_PER_PX
    plantable_px = int(plantable.sum())
    plantable_ha = plantable_px * HA_PER_PX
    summary_rows = []
    for density in (TREES_PER_HA_PRIMARY, *SENSITIVITY_DENSITIES):
        summary_rows.append({
            "year": year, "scenario": scenario,
            "density_trees_per_ha": density,
            "is_primary_density": density == TREES_PER_HA_PRIMARY,
            "n_zones": int(len(zpresent)),
            "total_zone_area_ha": round(total_zone_ha, 3),
            "excluded_px": total_excl_px,
            "excluded_ha": round(total_excl_ha, 3),
            "plantable_px": plantable_px,
            "plantable_ha": round(plantable_ha, 3),
            "recommended_trees": int(round(plantable_ha * density)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / f"tree_requirement_summary_{tag}.csv", index=False)

    # ---- recommended-plantation polygons (8-conn clusters, no min size) -------
    cluster_labelled, n_clusters = ndimage.label(
        plantable, structure=np.ones((3, 3), dtype=int))
    polygons = polygons_per_zone(cluster_labelled, profile["transform"]) \
        if n_clusters else {}
    validation = validate_geometries(polygons)
    cluster_rows, cluster_geoms = [], []
    for cid in sorted(int(c) for c in np.unique(cluster_labelled[cluster_labelled > 0])):
        cmask = cluster_labelled == cid
        n_px = int(cmask.sum())
        zids = np.unique(zone_labelled[cmask])
        zids = zids[zids > 0]
        if zids.size != 1:
            raise AssertionError(f"{tag}: cluster {cid} spans {zids.size} zones")
        zid = int(zids[0])
        cluster_rows.append({
            "zone_id": zid, "area_ha": round(n_px * HA_PER_PX, 3),
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
            cluster_df, geometry=gpd.GeoSeries(dtype="geometry"), crs="EPSG:4326")
    gdf.to_file(vectors_dir / f"recommended_plantations_{tag}.geojson",
                driver="GeoJSON")

    points_path = None
    if year == FOCUS_YEAR:
        pts = cluster_df.copy()
        pts["geometry"] = [g.centroid for g in cluster_geoms]
        pts_gdf = gpd.GeoDataFrame(pts, geometry="geometry", crs="EPSG:4326")
        points_path = vectors_dir / f"recommended_locations_{scenario}_2026.geojson"
        pts_gdf.to_file(points_path, driver="GeoJSON")

    noise_zone_px = int(((cls >= ZONE_MIN_CLASS) & domain & (zone_labelled == 0)).sum())
    return {
        "year": year, "scenario": scenario, "domain_px": int(domain.sum()),
        "n_zones": int(len(zpresent)),
        "zone_pixel_sum": total_zone_px,
        "excluded_px": total_excl_px,
        "excluded_ha": round(total_excl_ha, 3),
        "noise_zone_px_excluded": noise_zone_px,
        "plantable_px": plantable_px,
        "plantable_ha": round(plantable_ha, 3),
        "recommended_trees_primary": plantable_px * TREES_PER_PX_PRIMARY,
        "recommended_trees_400": int(round(plantable_ha * 400)),
        "recommended_trees_2500": int(round(plantable_ha * 2500)),
        "n_clusters": int(n_clusters),
        "cluster_trees_sum": int(cluster_df["trees"].sum()) if len(cluster_df) else 0,
        "cluster_geom_valid": bool(validation["all_valid"]),
        "veg_threshold_excluded_px": int(((zone_labelled > 0) & domain & eligible_lu
                                          & ~(veg < VEG_COVER_THRESHOLD)).sum()),
        "landuse_excluded_px": int(((zone_labelled > 0) & domain & ~eligible_lu).sum()),
        "paths": {"space": str(space_path), "trees": str(trees_path),
                  "points": str(points_path) if points_path else None},
        "wall_s": None,   # filled by caller
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase7-dir", default=str(PHASE7_DIR_DEFAULT))
    ap.add_argument("--phase6-dir", default=str(PHASE6_DIR_DEFAULT))
    ap.add_argument("--phase3-root", default=str(PHASE3_ROOT_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--years", default=",".join(str(y) for y in YEARS))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    args = ap.parse_args(argv)

    years = tuple(int(y) for y in args.years.split(","))
    phase7_dir = Path(args.phase7_dir)
    phase6_dir = Path(args.phase6_dir)
    phase3_root = Path(args.phase3_root)
    out_root = Path(args.out)
    t_start = time.perf_counter()

    preflight(years, phase7_dir, phase6_dir, phase3_root, Path(args.grid_file))
    grid = Grid.from_file(Path(args.grid_file))
    profile = grid.profile(count=1, dtype="float32")
    profile.update({"height": grid.height, "width": grid.width,
                    "transform": grid.transform, "crs": grid.crs})
    log(f"grid {grid.width}x{grid.height}; years={years}; scenarios={SCENARIOS}")

    # static landuse (phase4 code path) — full grid, NaN->255
    static = load_static(phase3_root)
    lu = static["landuse_class"]
    landuse = np.where(np.isfinite(lu), lu, 255.0).astype(np.int64)

    # REUSED Phase 7 constraint rasters (water > buildings > road-surfaces
    # precedence established there; never re-derived from Phase 2 raw inputs)
    with rasterio.open(phase7_dir / "constraints_rasters" / "constraint_attributed_30m.tif") as ds:
        attributed = ds.read(1).astype(np.int32)
    constraints = {"attributed": attributed}

    input_hashes: dict[str, str] = {}
    steps: dict[str, dict] = {}
    for year in years:
        t0 = time.perf_counter()
        for scenario in SCENARIOS:
            res = compute_year(year, scenario, landuse, constraints, profile,
                               input_hashes, phase7_dir, phase6_dir,
                               phase3_root, out_root)
            res["wall_s"] = round(time.perf_counter() - t0, 3)
            steps[f"{scenario}_{year}"] = {k: v for k, v in res.items()
                                           if k not in ("paths",)}
            steps[f"{scenario}_{year}"]["paths"] = res["paths"]
            log(f"[{('PRIMARY' if scenario == PRIMARY_SCENARIO else 'baseline')}] "
                f"{scenario} {year}: zones={res['n_zones']} "
                f"plantable={res['plantable_px']:,} px ({res['plantable_ha']:.2f} ha) "
                f"excluded={res['excluded_px']:,} px ({res['excluded_ha']:.2f} ha) "
                f"trees(1000/ha)={res['recommended_trees_primary']:,} "
                f"clusters={res['n_clusters']} ({res['wall_s']}s)")

    # ---- per-scenario citywide tables ----------------------------------------
    for scenario in SCENARIOS:
        rows = []
        for year in years:
            s = steps[f"{scenario}_{year}"]
            rows.append({
                "year": year, "scenario": scenario,
                "n_zones": s["n_zones"],
                "total_zone_area_ha": round(s["zone_pixel_sum"] * HA_PER_PX, 3),
                "plantable_px": s["plantable_px"],
                "plantable_ha": s["plantable_ha"],
                "recommended_trees_1000_per_ha": s["recommended_trees_primary"],
                "recommended_trees_400_per_ha": s["recommended_trees_400"],
                "recommended_trees_2500_per_ha": s["recommended_trees_2500"],
            })
        tdir = out_root / scenario / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            tdir / f"tree_requirement_citywide_5yr_{scenario}.csv", index=False)

    manifest = {
        "stage": "phase8_tree_requirement",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": list(SCENARIOS),
        "primary_scenario": PRIMARY_SCENARIO,
        "baseline_scenario": BASELINE_SCENARIO,
        "primary_snapshot_year": PRIMARY_SNAPSHOT_YEAR,
        "product_label": PRODUCT_LABEL,
        "density_label": DENSITY_LABEL,
        "assumptions": {
            "planting_space_rule": (
                "plantable = Phase 7 priority-zone pixels (kept 8-connected "
                "suitability class >= 3 zones) AND NOT exclusion AND landuse "
                "eligible AND vegetation_cover < 0.30; per-zone excluded area "
                "is attributed to exactly one cause with precedence "
                "constraint > landuse_ineligible > veg_threshold"),
            "constraint_source": (
                "REUSED v2/data/phase7/constraints_rasters/ (attributed raster "
                "with water > buildings > road-surfaces precedence, written by "
                "v2.phase7); exclusions applied AFTER scoring on the same "
                "per-year normalization as the full valid domain — scenario "
                "differences are exclusion effects, not re-normalization"),
            "tree_density_trees_per_ha": TREES_PER_HA_PRIMARY,
            "trees_per_plantable_px": TREES_PER_PX_PRIMARY,
            "sensitivity_densities_trees_per_ha": list(SENSITIVITY_DENSITIES),
            "density_disclaimer": DENSITY_LABEL,
            "vegetation_cover_threshold": VEG_COVER_THRESHOLD,
            "landuse_eligibility_rule": {
                "rule": "eligible = LANDUSE_RULE_STATUS != 'DISCOURAGED' "
                        "(V1 src/suitability/config.py)",
                "excluded_codes": list(DISCOURAGED_LANDUSE_CODES),
                "nodata_255": ("neutral-eligible (class 0 treatment, frozen "
                               "eligibility value 50); unclassified background "
                               "remains a weakly identified uncertainty"),
            },
            "priority_rule": ("Phase 7 priority_ranking order (mean suitability "
                              "desc); High = ranks 1..ceil(n/3), Medium = next "
                              "ceil(n/3), Low = rest"),
            "area_rule": "area_ha = pixel_count * 900 m2 / 1e4",
            "focus_year": ("2026 = primary planning snapshot (relative, "
                           "per-year-normalized products; NOT an absolute "
                           "inter-annual heat claim)"),
            "severity_encoding": ("phase6 relative heat-severity "
                                  "classification / UHI hotspot proxy: "
                                  "0 Low / 1 Moderate / 2 High; "
                                  "severity_score in [0,2]"),
        },
        "inputs": [{"path": k, "sha256": v} for k, v in sorted(input_hashes.items())],
        "per_year": {k: {kk: vv for kk, vv in v.items() if kk != "paths"}
                     for k, v in steps.items()},
        "deviations": DEVIATIONS,
        "terminology_note": (
            "Tree numbers are PLANNING ESTIMATES on the relative "
            "plantation-suitability decision-support ranking (Phase 7) — not "
            "physical UHI intensity, not planting-success predictions, and "
            "not statements of legal availability, land ownership, or "
            "field-verified plantability. Densities are planning-density "
            "scenarios; the scientifically supported optimal density will be "
            "evaluated after Phase 9 using cooling predictions."),
    }
    dump_json(manifest, out_root / "phase8_manifest.json")
    dump_json({
        "phase": 8, "variant": "v2", "project": "GreenGrid-AI",
        "started_utc": manifest["created_at_utc"],
        "years": list(years), "scenarios": list(SCENARIOS),
        "primary_scenario": PRIMARY_SCENARIO,
        "baseline_scenario": BASELINE_SCENARIO,
        "primary_snapshot_year": PRIMARY_SNAPSHOT_YEAR,
        "total_wall_s": time.perf_counter() - t_start,
        "steps": steps,
        "software_versions": {"python": platform.python_version()},
        "status": "success",
    }, out_root / "phase8_pipeline_record.json")
    log(f"total {time.perf_counter() - t_start:.1f}s; manifest + record written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
