"""Gate G4 verification for the Phase 8 tree-requirement build.

Independently recomputes every Phase 8 quantity from the raw phase 3/4/6/7
inputs (implemented inline, NOT imported from the runner) and checks the
phase8_tree_requirement outputs:

  1. Area reconciliation: every area_ha == pixel_count * 900/1e4 (rel dev
     <= 1e-9); zone Area == zone px count * 900/1e4.
  2. Tree arithmetic: recommended_trees == plantable_px * 0.09 * 1000
     (= px * 90; the spec text's "9" is a decimal slip, documented in the
     manifest deviations); sensitivity rows == px * 0.09 * density (+-1);
     per-zone tree sum == citywide total.
  3. Raster/grid/nodata/CRS checks on all new rasters vs the reference grid.
  4. Plantable mask never intersects exclusion_mask, discouraged landuse, or
     vegetation_cover >= 0.30 (assert 0 px each).
  5. GeoJSON properties consistent with zone tables: 3 zones of the focus
     year (top / middle / bottom by rank) fully recomputed from raw rasters
     and compared against both the zone table and every cluster row.
  6. Priority labels == ranking thirds of the Phase 7 priority_ranking CSV.

Writes data/processed/phase8_tree_requirement/verification_g4.json and prints
PASS/FAIL.  Exit code 0 on PASS, 1 on FAIL.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/verify_phase8_tree_requirement.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from models.config import REFERENCE_RASTER  # noqa: E402
from suitability.config import (  # noqa: E402  (frozen constants only)
    LANDUSE_RULE_STATUS,
    MIN_ZONE_PIXELS,
)

YEARS = (2022, 2023, 2024, 2025, 2026)
FOCUS_YEAR = 2026
OUT_ROOT = PROJECT / "data" / "processed" / "phase8_tree_requirement"
RASTERS_DIR = OUT_ROOT / "rasters"
TABLES_DIR = OUT_ROOT / "tables"
VECTORS_DIR = OUT_ROOT / "vectors"
G4_JSON = OUT_ROOT / "verification_g4.json"

P7_RASTERS = PROJECT / "data" / "processed" / "phase7_production" / "rasters"
P7_TABLES = PROJECT / "data" / "processed" / "phase7_production" / "tables"
P6_RASTERS = PROJECT / "data" / "processed" / "phase6_production" / "rasters"
P4_FEATURES = PROJECT / "data" / "processed" / "phase4" / "features"
P3_MASKS = PROJECT / "data" / "processed" / "phase3" / "masks"

PX_AREA_M2 = 900.0
M2_PER_HA = 10_000.0
HA_PER_PX = PX_AREA_M2 / M2_PER_HA
TREES_PER_HA_PRIMARY = 1_000
TREES_PER_PX_PRIMARY = int(round(TREES_PER_HA_PRIMARY * HA_PER_PX))  # 90
SENSITIVITY_DENSITIES = (400, 2_500)
VEG_THRESHOLD = 0.30
PLANT_NODATA = 255
TREES_NODATA = -1
SEVERITY_HIGH = 2
AREA_TOL = 1e-9


def log(msg: str) -> None:
    print(f"[G4] {msg}", flush=True)


def read_band(path: Path) -> Tuple[np.ndarray, float]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.profile.get("nodata")
    return arr, (float(nodata) if nodata is not None else np.nan)


# ---------------------------------------------------------------------------
# Independent reimplementations (frozen constants imported only)
# ---------------------------------------------------------------------------
def label_kept_zones(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected labelling + >= MIN_ZONE_PIXELS noise rule (independent)."""
    labels, n_found = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    if n_found == 0:
        return labels.astype(np.int32), 0
    sizes = np.bincount(labels.ravel())
    keep = sizes >= MIN_ZONE_PIXELS
    keep[0] = False
    remap = np.zeros(n_found + 1, dtype=np.int32)
    remap[np.where(keep)[0]] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[labels].astype(np.int32), int(keep.sum())


def landuse_eligible(landuse: np.ndarray) -> np.ndarray:
    eligible = np.ones(256, dtype=bool)
    for code, status in LANDUSE_RULE_STATUS.items():
        if status == "DISCOURAGED":
            eligible[int(code)] = False
    return eligible[landuse.astype(np.int64)]


def recompute_year(year: int) -> Dict:
    cls, _ = read_band(P7_RASTERS / f"suitability_class_{year}.tif")
    excl, _ = read_band(P7_RASTERS / f"exclusion_mask_{year}.tif")
    suit, _ = read_band(P7_RASTERS / f"suitability_{year}.tif")
    sev, _ = read_band(P6_RASTERS / f"severity_{year}.tif")
    sevscore, _ = read_band(P6_RASTERS / f"severity_score_{year}.tif")
    veg, veg_nd = read_band(P4_FEATURES / f"vegetation_cover_{year}_30m.tif")
    lu, _ = read_band(P3_MASKS / "landuse_raster_30m.tif")

    domain = excl == 0
    zone_lab, n_zones = label_kept_zones((cls >= 3) & domain)
    veg_ok = np.isfinite(veg) & (veg != veg_nd) & (veg < VEG_THRESHOLD)
    eligible = landuse_eligible(lu)
    plantable = ((cls >= 3) & (zone_lab > 0) & domain & (excl == 0)
                 & eligible & veg_ok)
    clusters, n_clusters = ndimage.label(plantable,
                                         structure=np.ones((3, 3), dtype=int))
    return {
        "domain": domain, "zone_lab": zone_lab, "n_zones": n_zones,
        "plantable": plantable, "clusters": clusters,
        "n_clusters": int(n_clusters), "suit": suit, "sev": sev,
        "sevscore": sevscore, "veg": veg, "eligible": eligible, "lu": lu,
    }


def priority_thirds(n: int) -> Dict[int, str]:
    third = int(np.ceil(n / 3.0))
    return {r: ("High" if r <= third else "Medium" if r <= 2 * third else "Low")
            for r in range(1, n + 1)}


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self) -> None:
        self.checks: List[Dict] = []

    def add(self, name: str, passed: bool, detail: Dict) -> bool:
        self.checks.append({"check": name, "passed": bool(passed), **detail})
        log(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")
        return passed

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def grid_check(name: str, path: Path, ref_profile: Dict, ref_transform,
               ref_crs, dtype: str, nodata: float, checker: Checker) -> np.ndarray:
    with rasterio.open(path) as src:
        prof = src.profile
        ok = (prof["width"] == ref_profile["width"]
              and prof["height"] == ref_profile["height"]
              and src.transform == ref_transform
              and src.crs == ref_crs
              and prof["dtype"] == dtype
              and float(prof["nodata"]) == nodata)
        arr = src.read(1)
    checker.add(f"grid:{name}", ok, {
        "path": str(path.relative_to(PROJECT)),
        "shape": [int(prof["height"]), int(prof["width"])],
        "dtype": prof["dtype"], "nodata": prof.get("nodata")})
    return arr


def rel_dev(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------
def main() -> int:
    t_start = time.time()
    checker = Checker()
    year_results: Dict[str, Dict] = {}

    with rasterio.open(REFERENCE_RASTER) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs

    for year in YEARS:
        log(f"=== {year} ===")
        yr: Dict = {}
        rc = recompute_year(year)
        domain = rc["domain"]
        plantable = rc["plantable"]
        n_plant = int(plantable.sum())

        # ---- rasters: presence + grid conformance ----------------------
        space = grid_check(f"{year}:available_planting_space",
                           RASTERS_DIR / f"available_planting_space_{year}.tif",
                           ref_profile, ref_transform, ref_crs, "uint8",
                           float(PLANT_NODATA), checker)
        trees_r = grid_check(f"{year}:recommended_trees",
                             RASTERS_DIR / f"recommended_trees_{year}.tif",
                             ref_profile, ref_transform, ref_crs, "int16",
                             float(TREES_NODATA), checker)

        # ---- raster content vs independent recomputation ---------------
        exp_space = np.where(domain, plantable.astype(np.uint8), PLANT_NODATA)
        space_ok = bool((space == exp_space).all())
        checker.add(f"{year}:planting_space_raster_content", space_ok, {
            "plantable_px": n_plant,
            "mismatch_px": int((space != exp_space).sum())})
        exp_trees = np.where(domain, plantable.astype(np.int16) * TREES_PER_PX_PRIMARY,
                             TREES_NODATA)
        trees_ok = bool((trees_r == exp_trees).all())
        checker.add(f"{year}:recommended_trees_raster_content", trees_ok, {
            "trees_per_plantable_px": TREES_PER_PX_PRIMARY,
            "raster_tree_sum": int(trees_r[trees_r > 0].sum()),
            "expected_tree_sum": n_plant * TREES_PER_PX_PRIMARY})

        # ---- exclusion intersection assertions (0 px) -------------------
        excl, _ = read_band(P7_RASTERS / f"exclusion_mask_{year}.tif")
        bad_excl = int((plantable & (excl == 1)).sum())
        veg = rc["veg"]
        veg_fin = np.isfinite(veg)
        bad_veg = int((plantable & veg_fin & (veg >= VEG_THRESHOLD)).sum())
        bad_veg_nan = int((plantable & ~veg_fin).sum())
        bad_lu = int((plantable & ~rc["eligible"]).sum())
        checker.add(f"{year}:plantable_exclusion_disjoint", bad_excl == 0, {
            "intersection_px": bad_excl})
        checker.add(f"{year}:plantable_vegetation_disjoint",
                    bad_veg == 0 and bad_veg_nan == 0, {
                        "veg_ge_threshold_px": bad_veg,
                        "veg_nonfinite_px": bad_veg_nan,
                        "threshold": VEG_THRESHOLD})
        checker.add(f"{year}:plantable_landuse_eligible", bad_lu == 0, {
            "discouraged_lu_px": bad_lu})

        # ---- zone table area + tree arithmetic --------------------------
        zone_df = pd.read_csv(TABLES_DIR / f"tree_requirement_by_zone_{year}.csv")
        ranking = pd.read_csv(P7_TABLES / f"priority_ranking_{year}.csv")
        p7_stats = pd.read_csv(P7_TABLES / f"priority_zone_statistics_{year}.csv")
        n = len(zone_df)
        yr["n_zones"] = n

        area_devs = np.abs(zone_df["area_ha"].to_numpy(float)
                           - zone_df["pixel_count"].to_numpy(float) * HA_PER_PX)
        plant_ha_devs = np.abs(zone_df["plantable_ha"].to_numpy(float)
                               - zone_df["plantable_px"].to_numpy(float) * HA_PER_PX)
        rel_devs = [d / max(v, 1e-12) for d, v in
                    zip(area_devs, zone_df["area_ha"].to_numpy(float))]
        rel_devs += [d / max(v, 1e-12) for d, v in
                     zip(plant_ha_devs, zone_df["plantable_ha"].to_numpy(float))]
        max_area_rel = float(max(rel_devs)) if rel_devs else 0.0
        trees_match = bool((zone_df["recommended_trees"].to_numpy(int)
                            == zone_df["plantable_px"].to_numpy(int)
                            * TREES_PER_PX_PRIMARY).all())
        sum_zone_px = int(zone_df["pixel_count"].sum())
        sum_zone_plant = int(zone_df["plantable_px"].sum())
        sum_zone_trees = int(zone_df["recommended_trees"].sum())
        kept_px = int((rc["zone_lab"] > 0).sum())
        identity_ok = (set(zone_df["zone_id"]) == set(p7_stats["zone_id"])
                       == set(ranking["zone_id"]))
        px_match = bool((zone_df.sort_values("zone_id")["pixel_count"].to_numpy(int)
                         == p7_stats.sort_values("zone_id")["pixel_count"].to_numpy(int)).all())
        checker.add(f"{year}:zone_area_reconciliation",
                    max_area_rel <= AREA_TOL and trees_match
                    and sum_zone_px == kept_px and identity_ok and px_match, {
                        "max_rel_dev_ha": f"{max_area_rel:.2e}",
                        "zone_px_sum": sum_zone_px, "kept_zone_px": kept_px,
                        "trees_eq_px_x90": trees_match,
                        "zone_identity_matches_phase7": identity_ok and px_match})

        # zone stats recompute for ALL zones (mean values, from raw rasters)
        zone_lab = rc["zone_lab"]
        suit, sev, sevscore, veg = rc["suit"], rc["sev"], rc["sevscore"], rc["veg"]
        max_stat_dev = 0.0
        stat_ok = True
        for rec in zone_df.itertuples():
            zm = zone_lab == int(rec.zone_id)
            checks = {
                "current_vegetation_pct": float(veg[zm].mean()) * 100.0,
                "mean_severity_score": float(sevscore[zm].mean()),
                "high_severity_pct": float((sev[zm] == SEVERITY_HIGH).mean()) * 100.0,
                "plantable_px": int((zm & plantable).sum()),
            }
            for key, val in checks.items():
                dev = abs(val - float(getattr(rec, key)))
                tol = 0.006 if key.endswith("pct") else (1e-4 if "severity" in key else 0.0)
                if key == "plantable_px":
                    stat_ok &= dev == 0.0
                else:
                    stat_ok &= dev <= tol
                max_stat_dev = max(max_stat_dev, dev)
        checker.add(f"{year}:zone_stats_recompute", stat_ok, {
            "zones_checked": n, "max_abs_dev": f"{max_stat_dev:.2e}"})

        # priority labels vs ranking thirds
        expect_prio = priority_thirds(len(ranking))
        prio_ok = bool(all(expect_prio[int(r.rank)] == str(p)
                           for r, p in zip(zone_df.itertuples(), zone_df["priority"])))
        rank_order_ok = list(zone_df["rank"]) == sorted(zone_df["rank"])
        checker.add(f"{year}:priority_labels", prio_ok and rank_order_ok, {
            "n_zones": n,
            "label_counts": zone_df["priority"].value_counts().to_dict()})

        # ---- summary tables ---------------------------------------------
        summary = pd.read_csv(TABLES_DIR / f"tree_requirement_summary_{year}.csv")
        summ_ok = True
        summ_detail: Dict[str, int] = {}
        for row in summary.itertuples():
            density = int(row.density_trees_per_ha)
            expected = int(round(n_plant * HA_PER_PX * density))
            ok = abs(int(row.recommended_trees) - expected) <= 1
            summ_ok &= ok
            summ_ok &= int(row.plantable_px) == n_plant
            summ_ok &= rel_dev(float(row.plantable_ha), n_plant * HA_PER_PX) <= AREA_TOL
            summ_ok &= int(row.n_zones) == n
            summ_detail[f"trees@{density}"] = int(row.recommended_trees)
            if not ok:
                summ_detail[f"expected@{density}"] = expected
        primary_row = summary[summary["is_primary_density"] == True]  # noqa: E712
        summ_ok &= (len(primary_row) == 1
                    and int(primary_row["recommended_trees"].iloc[0]) == sum_zone_trees)
        checker.add(f"{year}:summary_table_tree_arithmetic", bool(summ_ok),
                    {"plantable_px": n_plant, **summ_detail,
                     "zone_trees_sum": sum_zone_trees})

        # ---- citywide 5yr table row (checked after loop for all years) ---
        yr["plantable_px"] = n_plant
        yr["plantable_ha"] = round(n_plant * HA_PER_PX, 3)
        yr["recommended_trees_primary"] = n_plant * TREES_PER_PX_PRIMARY
        yr["zone_trees_sum"] = sum_zone_trees
        yr["n_clusters"] = rc["n_clusters"]
        year_results[str(year)] = yr

        # ---- GeoJSON cluster accounting ----------------------------------
        gdf = gpd.read_file(VECTORS_DIR / f"recommended_plantations_{year}.geojson")
        cl_ok = len(gdf) == rc["n_clusters"]
        cl_area_dev = rel_dev(float(gdf["area_ha"].sum()), n_plant * HA_PER_PX)
        cl_trees = int(gdf["trees"].sum())
        cl_trees_ok = cl_trees == sum_zone_trees == n_plant * TREES_PER_PX_PRIMARY
        props = {"zone_id", "area_ha", "trees", "mean_suitability",
                 "mean_severity_score", "priority"}
        props_ok = props.issubset(set(gdf.columns))
        geom_ok = bool(gdf.geometry.is_valid.all()) and len(gdf) > 0
        # per-cluster property recompute vs raw rasters
        clusters = rc["clusters"]
        zlab = rc["zone_lab"]
        max_cl_dev = 0.0
        cl_stat_ok = True
        for rec in gdf.itertuples():
            cid = None
            # match cluster by (zone_id, trees) then verify geometry overlap
            cand = [int(c) for c in np.unique(clusters[clusters > 0])
                    if int((clusters == c).sum()) * TREES_PER_PX_PRIMARY == int(rec.trees)
                    and int(zlab[clusters == c][0]) == int(rec.zone_id)]
            matched = False
            for c in cand:
                cmask = clusters == c
                devs = [
                    rel_dev(float(rec.area_ha), int(cmask.sum()) * HA_PER_PX),
                    abs(float(rec.mean_suitability) - float(suit[cmask].mean())),
                    abs(float(rec.mean_severity_score) - float(sevscore[cmask].mean())),
                ]
                if devs[0] <= AREA_TOL and devs[1] <= 1e-4 and devs[2] <= 1e-4:
                    max_cl_dev = max(max_cl_dev, *devs[1:])
                    matched = True
                    break
            cl_stat_ok &= matched
        checker.add(f"{year}:plantation_geojson_accounting",
                    cl_ok and cl_area_dev <= AREA_TOL and cl_trees_ok
                    and props_ok and geom_ok and cl_stat_ok, {
                        "clusters_geojson": int(len(gdf)),
                        "clusters_recomputed": rc["n_clusters"],
                        "area_ha_sum": round(float(gdf["area_ha"].sum()), 4),
                        "trees_sum": cl_trees,
                        "props_present": props_ok,
                        "geom_valid": geom_ok,
                        "all_clusters_matched": cl_stat_ok,
                        "max_mean_dev": f"{max_cl_dev:.2e}"})

    # ---- focus-year deep check: 3 zones fully recomputed ------------------
    log(f"=== {FOCUS_YEAR} deep zone recompute (3 zones) ===")
    focus = recompute_year(FOCUS_YEAR)
    zone_lab, plantable = focus["zone_lab"], focus["plantable"]
    suit, sev, sevscore, veg = focus["suit"], focus["sev"], focus["sevscore"], focus["veg"]
    zone_df = pd.read_csv(TABLES_DIR / f"tree_requirement_by_zone_{FOCUS_YEAR}.csv")
    ranking = pd.read_csv(P7_TABLES / f"priority_ranking_{FOCUS_YEAR}.csv")
    gdf = gpd.read_file(VECTORS_DIR / f"recommended_plantations_{FOCUS_YEAR}.geojson")
    pts = gpd.read_file(VECTORS_DIR / "recommended_locations_2026.geojson")
    n_zones_f = len(zone_df)
    picks = [0, n_zones_f // 2, n_zones_f - 1]
    deep_ok = True
    deep_detail: Dict[str, Dict] = {}
    expect_prio = priority_thirds(len(ranking))
    for pos in picks:
        rec = zone_df.sort_values("rank").iloc[pos]
        zid = int(rec["zone_id"])
        zm = zone_lab == zid
        zp = int((zm & plantable).sum())
        third = int(np.ceil(len(ranking) / 3.0))
        rank = int(rec["rank"])
        expected = {
            "area_ha": round(int(zm.sum()) * HA_PER_PX, 3),
            "pixel_count": int(zm.sum()),
            "current_vegetation_pct": round(float(veg[zm].mean()) * 100.0, 2),
            "mean_severity_score": round(float(sevscore[zm].mean()), 4),
            "high_severity_pct": round(float((sev[zm] == SEVERITY_HIGH).mean()) * 100.0, 2),
            "plantable_px": zp,
            "plantable_ha": round(zp * HA_PER_PX, 3),
            "recommended_trees": zp * TREES_PER_PX_PRIMARY,
            "priority": expect_prio[rank],
        }
        row_ok = True
        for key, val in expected.items():
            got = rec[key]
            if isinstance(val, str):
                row_ok &= str(got) == val
            elif key.endswith("px") or key in ("pixel_count", "recommended_trees"):
                row_ok &= int(got) == int(val)
            elif key.endswith("_ha"):
                row_ok &= rel_dev(float(got), float(val)) <= AREA_TOL
            else:
                row_ok &= abs(float(got) - float(val)) <= (0.006 if key.endswith("pct") else 1e-4)
        # every cluster row of this zone vs full recompute
        sub = gdf[gdf["zone_id"] == zid]
        clusters = focus["clusters"]
        for crow in sub.itertuples():
            cand = [int(c) for c in np.unique(clusters[clusters > 0])
                    if int(zlab[clusters == c][0]) == zid
                    and int((clusters == c).sum()) * TREES_PER_PX_PRIMARY == int(crow.trees)]
            matched = False
            for c in cand:
                cmask = clusters == c
                matched = (rel_dev(float(crow.area_ha), int(cmask.sum()) * HA_PER_PX) <= AREA_TOL
                           and abs(float(crow.mean_suitability) - float(suit[cmask].mean())) <= 1e-4
                           and abs(float(crow.mean_severity_score)
                                   - float(sevscore[cmask].mean())) <= 1e-4
                           and str(crow.priority) == expect_prio[rank])
                if matched:
                    break
            row_ok &= matched
        deep_ok &= bool(row_ok)
        deep_detail[f"rank{rank}_zone{zid}"] = {
            "ok": bool(row_ok),
            "plantable_px": zp,
            "recommended_trees": expected["recommended_trees"],
            "priority": expected["priority"],
        }
    checker.add(f"{FOCUS_YEAR}:deep_zone_recompute_3zones", deep_ok, deep_detail)

    # ---- points file (2026) -----------------------------------------------
    pts_ok = len(pts) == len(gdf) and len(pts) == focus["n_clusters"]
    pts_props = {"zone_id", "area_ha", "trees", "mean_suitability",
                 "mean_severity_score", "priority"}.issubset(set(pts.columns))
    poly_by_key = {}
    for rec in gdf.itertuples():
        poly_by_key[(int(rec.zone_id), int(rec.trees),
                     round(float(rec.area_ha), 3))] = rec.geometry
    pt_match = 0
    pt_covered = 0
    # pixel diagonal in degree units (centroid of a non-convex pixel union
    # can fall outside the polygon by at most ~1 px)
    px_diag_deg = float(np.hypot(0.00026949458523585647, 0.00026949458523585647))
    for rec in pts.itertuples():
        key = (int(rec.zone_id), int(rec.trees), round(float(rec.area_ha), 3))
        poly = poly_by_key.get(key)
        if poly is None:
            continue
        if poly.covers(rec.geometry):
            pt_covered += 1
        if poly.distance(rec.geometry) <= px_diag_deg + 1e-12:
            pt_match += 1
    pts_ok &= pts_props and pt_match == len(pts)
    checker.add(f"{FOCUS_YEAR}:recommended_locations_points", pts_ok, {
        "points": int(len(pts)), "polygons": int(len(gdf)),
        "properties_present": pts_props,
        "centroid_within_1px_of_polygon": pt_match,
        "centroid_covered_by_polygon": pt_covered,
        "note": ("shapely area-centroid can exit non-convex pixel unions "
                 "(U/L-shapes); <= 1 pixel-diagonal distance is the tight "
                 "geometric bound")})

    # ---- citywide 5-year table -------------------------------------------
    cw = pd.read_csv(TABLES_DIR / "tree_requirement_citywide_5yr.csv")
    cw_ok = len(cw) == len(YEARS) and list(cw["year"]) == list(YEARS)
    for row in cw.itertuples():
        yr_res = year_results[str(int(row.year))]
        cw_ok &= int(row.plantable_px) == yr_res["plantable_px"]
        cw_ok &= rel_dev(float(row.plantable_ha), yr_res["plantable_ha"]) <= AREA_TOL
        cw_ok &= int(row.recommended_trees_1000_per_ha) == yr_res["recommended_trees_primary"]
        cw_ok &= int(row.recommended_trees_400_per_ha) == int(
            round(yr_res["plantable_px"] * HA_PER_PX * 400))
        cw_ok &= int(row.recommended_trees_2500_per_ha) == int(
            round(yr_res["plantable_px"] * HA_PER_PX * 2500))
        cw_ok &= int(row.n_zones) == yr_res["n_zones"]
    checker.add("citywide_5yr_table", bool(cw_ok),
                {"rows": int(len(cw))})

    # ---- manifest ----------------------------------------------------------
    manifest_path = OUT_ROOT / "phase8_manifest.json"
    man_ok = manifest_path.exists()
    if man_ok:
        with open(manifest_path) as f:
            man = json.load(f)
        man_ok &= all(k in man.get("assumptions", {}) for k in
                      ("tree_density_trees_per_ha", "vegetation_cover_threshold",
                       "landuse_eligibility_source", "planting_space_rule"))
        man_ok &= all(k in man.get("model_lineage", {}) for k in
                      ("phase5_model_id", "phase6_manifest", "phase7_manifest"))
    checker.add("manifest_assumptions_lineage", man_ok,
                {"path": str(manifest_path.relative_to(PROJECT))})

    # ------------------------------------------------------------------------
    n_pass = sum(1 for c in checker.checks if c["passed"])
    result = {
        "gate": "G4",
        "stage": "phase8_tree_requirement",
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t_start, 3),
        "n_checks": len(checker.checks),
        "n_passed": n_pass,
        "n_failed": len(checker.checks) - n_pass,
        "status": "PASS" if checker.all_passed else "FAIL",
        "tree_arithmetic_note": (
            f"trees = plantable_px * {HA_PER_PX} ha/px * density; at "
            f"{TREES_PER_HA_PRIMARY}/ha = {TREES_PER_PX_PRIMARY} trees/px "
            "(spec text's '9' documented as decimal slip in manifest "
            "deviations[0])"),
        "years": year_results,
        "checks": checker.checks,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(G4_JSON, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log(f"{result['status']}: {n_pass}/{len(checker.checks)} checks passed "
        f"({result['elapsed_s']}s)")
    if not checker.all_passed:
        for c in checker.checks:
            if not c["passed"]:
                log(f"  FAILED: {c['check']} {c}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
