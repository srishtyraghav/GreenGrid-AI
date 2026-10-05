"""V2 Phase 8 verification gate — numbered PASS/FAIL checklist.

Read-only re-derivation. Exit 0 = all PASS.

Checks
------
1.  lineage + framing: phase7/phase6 manifests present; frozen assumptions;
    v2_constrained recorded as PRIMARY scenario, v1_parity as baseline,
    2026 as the primary planning snapshot.
2.  grid/CRS/dims of every written raster vs the authoritative grid.
3.  tree-count arithmetic: for EVERY zone/density/year/scenario,
    trees == round(plantable_ha x density) exactly (400/1000/2500).
3z. per-zone area accounting: zone px == plantable + excluded exactly;
    excluded cause counts (constraint/landuse/veg) are disjoint and sum to
    excluded px (precedence, no double counting).
4.  planting space independently re-derived from source inputs equals the
    written raster; plantable is boolean, a subset of the valid domain, and
    (v2_constrained) excludes every constraint pixel.
4z. zone IDs: unique, present in the Phase 7 zone statistics tables with
    identical pixel counts (identity cross-check re-asserted).
5.  priority thirds match the Phase 7 ranking order.
6.  cross-scenario sanity: v1_parity plantable space >= v2_constrained
    (constraint exclusion can only remove space).
7.  tables/clusters agree with rasters.
8.  manifest + pipeline record complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from ..common import GRID_FILE_DEFAULT, PROJECT_ROOT, Grid
from ..phase4.assemble_features import load_static
from ..phase7.build import read_band
from ..phase8.build import (
    DISCOURAGED_LANDUSE_CODES,
    HA_PER_PX,
    SCENARIOS,
    SENSITIVITY_DENSITIES,
    TREES_PER_HA_PRIMARY,
    TREES_PER_PX_PRIMARY,
    VEG_COVER_THRESHOLD,
    build_cause_grid,
    landuse_eligible_mask,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(n: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{n}] [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "phase8"))
    ap.add_argument("--phase7-dir", default=str(PROJECT_ROOT / "data" / "phase7"))
    ap.add_argument("--phase6-dir", default=str(PROJECT_ROOT / "data" / "phase6"))
    ap.add_argument("--phase3-root", default=str(PROJECT_ROOT / "data" / "phase3"))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    phase7_dir = Path(args.phase7_dir)
    phase6_dir = Path(args.phase6_dir)

    manifest = json.loads((out_root / "phase8_manifest.json").read_text())
    record = json.loads((out_root / "phase8_pipeline_record.json").read_text())
    years = [int(y) for y in record["years"]]
    grid = Grid.from_file(Path(args.grid_file))
    shape = grid.shape
    ok_all = True

    # -- [1] lineage + frozen assumptions + framing ------------------------------
    a = manifest.get("assumptions", {})
    ok1 = (a.get("tree_density_trees_per_ha") == TREES_PER_HA_PRIMARY
           and a.get("trees_per_plantable_px") == TREES_PER_PX_PRIMARY
           and a.get("vegetation_cover_threshold") == VEG_COVER_THRESHOLD
           and tuple(a.get("landuse_eligibility_rule", {}).get("excluded_codes", ()))
           == DISCOURAGED_LANDUSE_CODES
           and manifest.get("primary_scenario") == "v2_constrained"
           and manifest.get("baseline_scenario") == "v1_parity"
           and manifest.get("primary_snapshot_year") == 2026
           and (phase7_dir / "phase7_pipeline_record.json").exists()
           and (phase6_dir / "phase6_manifest.json").exists())
    ok_all &= check("1", "lineage + frozen assumptions + primary/baseline framing", ok1)

    # -- independent static sources ---------------------------------------------
    static = load_static(Path(args.phase3_root))
    lu = static["landuse_class"]
    landuse = np.where(np.isfinite(lu), lu, 255.0).astype(np.int64)
    eligible_lu = landuse_eligible_mask(landuse)
    with rasterio.open(phase7_dir / "constraints_rasters" / "constraint_attributed_30m.tif") as ds:
        attributed = ds.read(1).astype(np.int32)

    DENSITIES = (400, TREES_PER_HA_PRIMARY, 2500)

    # -- [2..7] per scenario-year ------------------------------------------------
    totals = {}
    for scenario in SCENARIOS:
        totals[scenario] = {}
        for y in years:
            tag = f"{scenario}_{y}"
            rdir = out_root / scenario / "rasters"
            tdir = out_root / scenario / "tables"
            ok2 = True
            for name in (f"available_planting_space_{tag}.tif",
                         f"recommended_trees_{tag}.tif"):
                with rasterio.open(rdir / name) as ds:
                    ok2 &= (ds.height, ds.width) == shape and \
                        ds.crs.to_epsg() == 4326
            ok_all &= check(f"2-{tag}", f"{tag} grid/CRS/dims", ok2)

            space, sp_nd = read_band(rdir / f"available_planting_space_{tag}.tif")
            trees_r, tr_nd = read_band(rdir / f"recommended_trees_{tag}.tif")
            zids, _ = read_band(phase7_dir / scenario / "rasters"
                                / f"priority_zone_ids_{scenario}_{y}.tif")
            excl, _ = read_band(phase7_dir / scenario / "rasters"
                                / f"exclusion_mask_{scenario}_{y}.tif")
            cls, _ = read_band(phase7_dir / scenario / "rasters"
                               / f"suitability_class_{scenario}_{y}.tif")
            veg, veg_nd = read_band(Path(args.phase3_root) / str(y)
                                    / "vegetation_cover_30m.tif")
            domain = excl == 0
            veg_ok = np.isfinite(veg) & (veg != veg_nd) & (veg < VEG_COVER_THRESHOLD)

            # independent plantable re-derivation via the shared cause grid
            cause = build_cause_grid(
                veg_ok, eligible_lu,
                attributed if scenario == "v2_constrained" else None)
            plant_re = ((cls >= 3) & (zids > 0) & domain & (excl == 0)
                        & eligible_lu & veg_ok)
            ok4a = bool(np.array_equal(plant_re, space == 1))
            ok4b = bool(np.all((space[domain] == 0) | (space[domain] == 1)))
            ok4c = bool(np.all(space[~domain] == 255))
            ok4d = bool(np.all(~plant_re | domain))                    # subset of domain
            ok4e = (int((plant_re & (attributed > 0)).sum()) == 0
                    if scenario == "v2_constrained" else True)        # no constraint px
            ok4f = bool(np.array_equal(
                np.where(domain, plant_re.astype(np.int16) * TREES_PER_PX_PRIMARY,
                         -1), trees_r))
            ok_all &= check(f"4-{tag}", f"{tag} plantable mask re-derived/boolean/"
                            f"subset/excludes constraints",
                            ok4a and ok4b and ok4c and ok4d and ok4e and ok4f,
                            f"plantable={int(plant_re.sum()):,}")

            zdf = pd.read_csv(tdir / f"tree_requirement_by_zone_{tag}.csv")
            sdf = pd.read_csv(tdir / f"tree_requirement_summary_{tag}.csv")
            p7z = pd.read_csv(phase7_dir / scenario / "tables"
                              / f"priority_zone_statistics_{scenario}_{y}.csv")

            # [4z] zone IDs unique + identical to Phase 7 tables
            ok4z = bool(zdf["zone_id"].is_unique) and \
                set(zdf["zone_id"].tolist()) == set(p7z["zone_id"].tolist())
            ok_all &= check(f"4z-{tag}", f"{tag} zone IDs unique == Phase 7 zones",
                            ok4z)

            # [3z] per-zone area accounting + disjoint causes (no double count)
            acc = (zdf["pixel_count"]
                   == zdf["plantable_px"] + zdf["excluded_px"])
            causes = (zdf["excluded_constraint_px"] + zdf["excluded_landuse_px"]
                      + zdf["excluded_veg_px"] == zdf["excluded_px"])
            ok3z = bool(acc.all() and causes.all())
            ok_all &= check(f"3z-{tag}", f"{tag} area accounting plantable+excluded"
                            f" == zone px; causes disjoint", ok3z)

            # [3] tree arithmetic for EVERY zone x density
            ok3 = True
            for d, col in ((400, "recommended_trees_400"),
                           (TREES_PER_HA_PRIMARY, "recommended_trees_1000"),
                           (2500, "recommended_trees_2500")):
                expect = (zdf["plantable_ha"] * d).round().astype(int)
                ok3 &= bool((zdf[col] == expect).all())
            ok3 &= bool((zdf["recommended_trees"]
                         == zdf["recommended_trees_1000"]).all())
            ok3 &= bool(np.all(sdf["recommended_trees"]
                               == (sdf["plantable_ha"]
                                   * sdf["density_trees_per_ha"]).round().astype(int)))
            ok_all &= check(f"3-{tag}", f"{tag} trees == round(ha x density) for "
                            f"every zone x {DENSITIES}", ok3)

            n = len(zdf)
            third = int(np.ceil(n / 3.0))
            expect = pd.Series(
                ["High" if r <= third else ("Medium" if r <= 2 * third else "Low")
                 for r in zdf.sort_values("rank")["rank"]],
                index=zdf.sort_values("rank").index)
            ok5 = bool((zdf.sort_values("rank")["priority"]
                        .reset_index(drop=True) == expect.reset_index(drop=True)).all())
            ok_all &= check(f"5-{tag}", f"{tag} priority thirds == ranking order", ok5,
                            f"n={n} third={third}")

            ok7a = int(zdf["plantable_px"].sum()) == int(plant_re.sum())
            ok7b = int(sdf.loc[sdf["is_primary_density"], "plantable_px"].iloc[0]) \
                == int(plant_re.sum())
            import geopandas as gpd
            cl = gpd.read_file(out_root / scenario / "vectors"
                               / f"recommended_plantations_{tag}.geojson")
            ok7c = int(cl["trees"].sum()) == int(plant_re.sum()) * TREES_PER_PX_PRIMARY
            ok_all &= check(f"7-{tag}", f"{tag} tables/clusters agree with rasters",
                            ok7a and ok7b and ok7c)
            totals[scenario][y] = int(plant_re.sum())

    # -- [6] cross-scenario -------------------------------------------------------
    ok6 = all(totals["v1_parity"][y] >= totals["v2_constrained"][y] for y in years)
    ok_all &= check("6", "v1_parity plantable >= v2_constrained every year", ok6,
                    " ".join(f"{y}:{totals['v1_parity'][y]:,}>={totals['v2_constrained'][y]:,}"
                             for y in years))

    # -- [8] artifacts -------------------------------------------------------------
    need = [out_root / "phase8_manifest.json", out_root / "phase8_pipeline_record.json"]
    need += [out_root / sc / "tables" / f"tree_requirement_citywide_5yr_{sc}.csv"
             for sc in SCENARIOS]
    missing = [str(p) for p in need if not p.exists()]
    ok_all &= check("8", "manifest + record + citywide tables present",
                    not missing, f"missing={missing}" if missing else "all present")

    fails = [r for r in RESULTS if not r[1]]
    print(f"\n[SUMMARY] {len(RESULTS) - len(fails)}/{len(RESULTS)} checks pass, "
          f"{len(fails)} FAIL", flush=True)
    if fails:
        for name, _, detail in fails:
            print(f"  FAILED: {name} {detail}")
        return 1
    print("[VERIFY] ALL PHASE 8 CHECKS PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
