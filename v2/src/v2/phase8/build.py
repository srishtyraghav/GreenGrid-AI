"""V2 Phase 8 final — usable-area planning on the STABLE ACADEMIC SHORTLIST.

The zone set is FIXED (Phase 7: pooled h>=K evidence, MIN-veg persistent
plantability, feasibility, MMU 2 ha / max 57 ha, fixed 0-100 score, shortlist
= score >= 71.3 & not oversized). Phase 8 emits:
  - contract zone CSV (2026): zone_id + plantable_ha (== usable area = full
    zone, persistent plantability by construction) + additive columns;
  - per-year annual metric tables (qualifying/plantable px per zone);
  - rasters: priority_zone_ids (stable ids, identical every year) and
    available_planting_space_{y} (zone px plantable THAT year);
  - reference tree columns = floor(usable_area_ha x density) — the exact
    frontend arithmetic; density selection is frontend-owned (default 400,
    slider 100-2500, planning assumption, not optimal / not AI-predicted);
  - theoretical capacity envelope (unchanged semantics, transparency only).
Phase 9 contract: unchanged code; 2026 files + mirror bit-match verified.
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

from ..common import (
    GRID_FILE_DEFAULT, PROJECT_ROOT, YEARS, Grid, dump_json, sha256_file,
)
from ..phase4.assemble_features import load_static
from ..phase7.build import DISCOURAGED_LU_CODES, eligible_lu_grid
from ..phase7.suitability import label_zones

DENSITY_LABEL = ("planning assumption, not optimal and not AI-predicted")
SCORE_LABEL = ("planning-priority score (0-100), not a validated probability "
               "or proof of optimal cooling")
SCENARIOS = ("v1_parity", "v2_constrained")
PRIMARY_SCENARIO = "v2_constrained"
HA_PER_PX = 0.09
PLANT_NODATA = 255
MIN_PLANTING_PATCH_PX = 25

PHASE7_DIR_DEFAULT = PROJECT_ROOT / "data" / "phase7"
PHASE3_ROOT_DEFAULT = PROJECT_ROOT / "data" / "phase3"
OUT_DEFAULT = PROJECT_ROOT / "data" / "phase8"
LU_NAMES = {0: "untagged", 1: "park", 2: "forest", 3: "grass", 4: "commercial",
            5: "industrial", 6: "residential", 7: "retail", 8: "farmland",
            255: "nodata"}


def log(msg: str) -> None:
    print(f"[P8] {msg}", flush=True)


def band(p):
    with rasterio.open(p) as ds:
        return ds.read(1)


def preflight(years, phase7_dir, phase3_root, grid_file):
    for scenario in SCENARIOS:
        for name in (f"stable_zones_{scenario}.csv", f"stable_zones_{scenario}.geojson"):
            if not (phase7_dir / scenario / "zones" / name).exists():
                raise FileNotFoundError(f"missing stable zones: {name}")
        if not (phase7_dir / scenario / "rasters" / f"stable_zone_ids_{scenario}.tif").exists():
            raise FileNotFoundError("missing stable zone ids raster")
        for y in years:
            if not (phase3_root / str(y) / "vegetation_cover_30m.tif").exists():
                raise FileNotFoundError(f"missing veg {y}")
        for name in ("priority_class",):
            for y in years:
                if not (phase7_dir / scenario / "rasters" / f"{name}_{scenario}_{y}.tif").exists():
                    raise FileNotFoundError(f"missing phase7 {name} {y}")
    grid = Grid.from_file(Path(grid_file))
    assert (grid.height, grid.width) == (1768, 1874), "authoritative grid mismatch"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase7-dir", default=str(PHASE7_DIR_DEFAULT))
    ap.add_argument("--phase3-root", default=str(PHASE3_ROOT_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--years", default=",".join(str(y) for y in YEARS))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    args = ap.parse_args(argv)

    years = (2026,)   # 2026-only planning module
    phase7_dir = Path(args.phase7_dir)
    phase3_root = Path(args.phase3_root)
    out_root = Path(args.out)
    t_start = time.perf_counter()

    preflight(years, phase7_dir, phase3_root, Path(args.grid_file))
    grid = Grid.from_file(Path(args.grid_file))
    prof_u8 = grid.profile(count=1, dtype="uint8", nodata=PLANT_NODATA)
    prof_i32 = grid.profile(count=1, dtype="int32", nodata=-1)
    static = load_static(phase3_root)
    lu = np.where(np.isfinite(static["landuse_class"]),
                  static["landuse_class"], 255.0).astype(np.int64)
    p7rec = json.loads((phase7_dir / "phase7_pipeline_record.json").read_text())
    stable_meta = p7rec["sites_2026"]

    steps = {}
    for scenario in SCENARIOS:
        zlab = band(phase7_dir / scenario / "rasters" / f"stable_zone_ids_{scenario}.tif")
        zdf = pd.read_csv(phase7_dir / scenario / "zones" / f"stable_zones_{scenario}.csv")
        # landuse composition + certainty per zone (disclosure only)
        certs, shares, lu_ha_list = [], [], []
        for z in zdf["zone_id"]:
            m = zlab == z
            codes, counts = np.unique(lu[m], return_counts=True)
            d = {LU_NAMES.get(int(c_), str(c_)): int(n_) * HA_PER_PX
                 for c_, n_ in zip(codes, counts)}
            share = (float(counts[list(codes).index(0)] / counts.sum())
                     if 0 in list(codes) else 0.0)
            shares.append(round(share, 3))
            lu_ha_list.append({k: round(v, 3) for k, v in d.items()})
            certs.append("low" if share > 0.5 else "mixed" if share > 0.2 else "identified")
        zdf["untagged_share"] = shares
        zdf["landuse_certainty"] = certs
        zdf["landuse_ha"] = lu_ha_list
        zdf["has_water"] = False
        zdf["has_buildings"] = False
        zdf["has_road_surfaces"] = False
        zdf["score_label"] = SCORE_LABEL
        zdf["density_label"] = DENSITY_LABEL
        for d in (400, 1000, 2500):
            zdf[f"reference_trees_{d}"] = (zdf["usable_area_ha"] * d).astype(int)
        zdf["plantable_px"] = zdf["zone_px"]
        zdf["plantable_ha"] = zdf["usable_area_ha"]
        zdf["rationale"] = zdf["rationale"]

        rasters_dir = out_root / scenario / "rasters"
        tables_dir = out_root / scenario / "tables"
        vectors_dir = out_root / scenario / "vectors"
        for d in (rasters_dir, tables_dir, vectors_dir):
            d.mkdir(parents=True, exist_ok=True)

        # stable zone ids raster (identical every year) + mirror
        with rasterio.open(rasters_dir / f"priority_zone_ids_{scenario}_2026.tif", "w", **prof_i32) as dst:
            dst.write(zlab.astype(np.int32), 1)
        mirror = phase7_dir / scenario / "rasters"
        mirror.mkdir(parents=True, exist_ok=True)
        with rasterio.open(mirror / f"priority_zone_ids_{scenario}_2026.tif", "w", **prof_i32) as dst:
            dst.write(zlab.astype(np.int32), 1)

        y = 2026
        with rasterio.open(phase3_root / str(y) / "vegetation_cover_30m.tif") as ds:
            vg = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
        vok = np.isfinite(vg) & (vg < 0.30)
        space = np.where(zlab > 0, (vok & (zlab > 0)).astype(np.uint8),
                         PLANT_NODATA).astype(np.uint8)
        with rasterio.open(rasters_dir / f"available_planting_space_{scenario}_{y}.tif", "w", **prof_u8) as dst:
            dst.write(space, 1)
        plant_ha = round(float((space == 1).sum()) * HA_PER_PX, 1)
        short = zdf[zdf["shortlist"]]
        pd.DataFrame([{
            "year": y, "scenario": scenario,
            "n_zones": len(zdf), "shortlist_zones": len(short),
            "usable_area_ha": round(float(zdf["usable_area_ha"].sum()), 1),
            "annual_plantable_ha": plant_ha,
            "annual_trees_400": int(plant_ha * 400),
            "annual_trees_1000": int(plant_ha * 1000),
            "annual_trees_2500": int(plant_ha * 2500),
            "shortlist_ha": round(float(short["usable_area_ha"].sum()), 1),
            "density_label": DENSITY_LABEL,
            "label": ("2026-only recommendation module; density is a planning "
                      "assumption, not optimal and not AI-predicted"),
        }]).to_csv(tables_dir / f"tree_requirement_summary_{scenario}_{y}.csv", index=False)

        # contract CSV (2026 filename) + candidate/shortlist tables
        contract = zdf.copy()
        contract["year"] = 2026
        contract["scenario"] = scenario
        contract.drop(columns=["landuse_ha"]).to_csv(
            tables_dir / f"tree_requirement_by_zone_{scenario}_2026.csv", index=False)
        contract.drop(columns=["landuse_ha"]).to_csv(
            tables_dir / f"candidate_sites_{scenario}_2026.csv", index=False)
        zdf[zdf["shortlist"]].drop(columns=["landuse_ha"]).to_csv(
            tables_dir / f"recommended_sites_{scenario}_2026.csv", index=False)
        zdf[["zone_id", "landuse_ha"]].to_json(
            tables_dir / f"site_landuse_{scenario}_2026.json", orient="records")
        gj = gpd.read_file(phase7_dir / scenario / "zones" / f"stable_zones_{scenario}.geojson")
        for y in years:    # same geometry all years; per-year plantable attr
            gjy = gj.copy()
            gjy["usable_area_ha"] = zdf["usable_area_ha"].to_numpy()
            gjy.to_file(vectors_dir / f"recommended_plantations_{scenario}_{y}.geojson",
                        driver="GeoJSON")
        pts = zdf[zdf["shortlist"]].copy()
        if len(pts):
            pts["geometry"] = [gj.set_index("zone_id").loc[int(z)].geometry.centroid
                               for z in pts["zone_id"]]
            gpd.GeoDataFrame(pts.drop(columns=["landuse_ha"], errors="ignore"),
                             geometry="geometry", crs="EPSG:4326").to_file(
                vectors_dir / f"recommended_locations_{scenario}_2026.geojson",
                driver="GeoJSON")

        s2026 = pd.read_csv(tables_dir / f"tree_requirement_summary_{scenario}_2026.csv").iloc[0]
        log(f"{scenario}: zones={len(zdf)} shortlist={len(short)} "
            f"({float(short['usable_area_ha'].sum()):,.0f} ha) "
            f"2026-plantable={s2026['annual_plantable_ha']:,.0f} ha")
        steps[scenario] = {"n_zones": len(zdf), "n_shortlist": len(short),
                           "shortlist_ha": round(float(short["usable_area_ha"].sum()), 1),
                           "usable_ha": round(float(zdf["usable_area_ha"].sum()), 1)}

    record = {
        "phase": 8, "variant": "final-stable-shortlist", "project": "GreenGrid-AI",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "years": list(years), "scenarios": list(SCENARIOS),
        "primary_scenario": PRIMARY_SCENARIO, "primary_snapshot_year": 2026,
        "stable_shortlist_source": "phase7 record stable_shortlist (constants pinned)",
        "density_label": DENSITY_LABEL, "score_label": SCORE_LABEL,
        "usable_area_rule": ("usable_area_ha = full zone (zones form on MIN-veg "
                             "persistent plantability); per-year plantable px in "
                             "annual columns/rasters"),
        "total_wall_s": time.perf_counter() - t_start,
        "steps": steps,
        "software_versions": {"python": platform.python_version()},
        "status": "success",
    }
    dump_json(record, out_root / "phase8_pipeline_record.json")
    manifest = {
        "stage": "phase8_tree_requirement",
        "variant": "final-stable-shortlist",
        "created_at_utc": record["started_utc"],
        "scenarios": list(SCENARIOS),
        "inputs": [{"path": str(phase7_dir / "phase7_manifest.json"),
                    "sha256": sha256_file(phase7_dir / "phase7_manifest.json")}],
        "assumptions": {
            "zone_set": "FIXED across years (Phase 7 stable shortlist); year selector switches annual metric columns only",
            "usable_area": record["usable_area_rule"],
            "tree_arithmetic": "floor(usable_area_ha x density); density frontend-owned (default 400, 100-2500)",
            "density_label": DENSITY_LABEL,
            "shortlist": "score >= 71.3 and not oversized; OPERATIONAL PLANNING cut, not validated ecological cutoffs",
        },
        "per_scenario": steps,
        "terminology_note": (
            "Tree numbers are PLANNING ESTIMATES on the relative "
            "plantation-suitability decision-support ranking - not physical UHI "
            "intensity, not planting-success predictions, and not statements of "
            "legal availability, land ownership, or field-verified plantability."),
    }
    dump_json(manifest, out_root / "phase8_manifest.json")
    log(f"total {record['total_wall_s']:.1f}s; manifest + record written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
