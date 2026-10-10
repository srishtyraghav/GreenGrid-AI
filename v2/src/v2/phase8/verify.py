"""V2 Phase 8 FINAL (2026-only) verification gate. Exit 0 = all PASS.

1. Phase 9 contract (zone CSV cols + plantable raster + mirror bit-match).
2. Site rule recompute: 2026 feasible & S2 gates & veg<0.30, MMU>=2ha
   (constants pinned from the phase7 record).
3. Score recompute (0.50/0.25/0.25) + shortlist rule (top-100 non-oversized).
4. Usable-area exactness (usable == site px) + floor arithmetic + certainty.
5. Artifacts complete.
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
from ..phase4.assemble_features import load_static, load_phase3_year
from ..phase7.build import eligible_lu_grid
from ..phase7.suitability import (
    green_proximity_score, label_zones, landuse_eligibility, pooled_normalize,
    road_accessibility_score,
)

RESULTS = []


def check(n, name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{n}] [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return bool(ok)


def band(p):
    with rasterio.open(p) as ds:
        return ds.read(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "phase8"))
    ap.add_argument("--phase7-dir", default=str(PROJECT_ROOT / "data" / "phase7"))
    ap.add_argument("--phase3-root", default=str(PROJECT_ROOT / "data" / "phase3"))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    args = ap.parse_args(argv)
    out_root, p7 = Path(args.out), Path(args.phase7_dir)
    grid = Grid.from_file(Path(args.grid_file))
    record = json.loads((out_root / "phase8_pipeline_record.json").read_text())
    p7rec = json.loads((p7 / "phase7_pipeline_record.json").read_text())
    consts = p7rec["sites_2026"]["constants"]
    ok_all = True

    # [1] Phase 9 contract
    zcsv = out_root / "v2_constrained" / "tables" / "tree_requirement_by_zone_v2_constrained_2026.csv"
    cols = set(pd.read_csv(zcsv, nrows=1).columns)
    ok1 = {"zone_id", "plantable_ha", "usable_area_ha", "planning_priority_score",
           "shortlist"} <= cols
    with rasterio.open(out_root / "v2_constrained" / "rasters"
                       / "available_planting_space_v2_constrained_2026.tif") as ds:
        ok1 &= (ds.width, ds.height) == (grid.width, grid.height) and ds.nodata == 255.0
    a = band(out_root / "v2_constrained" / "rasters" / "priority_zone_ids_v2_constrained_2026.tif")
    b = band(p7 / "v2_constrained" / "rasters" / "priority_zone_ids_v2_constrained_2026.tif")
    ok1 &= bool(np.array_equal(a, b))
    ok_all &= check("1", "Phase 9 contract (cols, raster, mirror bit-match)", ok1)

    # [2] site rule recompute (2026)
    assert consts["year"] == 2026 and consts["mmu_min_px"] == 23 and \
        abs(consts["max_size_ha"] - 57.0) < 1e-9 and consts["shortlist_top_n"] == 100
    static = load_static(Path(args.phase3_root))
    lu = np.where(np.isfinite(static["landuse_class"]),
                  static["landuse_class"], 255.0).astype(np.int64)
    elig = eligible_lu_grid(lu)
    att = band(p7 / "constraints_rasters" / "constraint_attributed_30m.tif") > 0
    need = band(p7 / "v2_constrained" / "rasters" / "cooling_need_v2_constrained_2026.tif")
    suit = band(p7 / "v2_constrained" / "rasters" / "suitability_v2_constrained_2026.tif")
    dom = need != -1
    pr = np.full(dom.shape, np.nan)
    feas = elig & ~att & dom
    pr[feas] = 0.5 * need[feas].astype(float) + 0.5 * suit[feas].astype(float)
    P90 = float(p7rec["selection_constants"]["priority_p90"])
    ND75 = float(p7rec["selection_constants"]["need_p75"])
    vg = load_phase3_year(Path(args.phase3_root), 2026)["vegetation_cover"]
    vok = np.isfinite(vg) & (vg < 0.30)
    sites = feas & np.isfinite(pr) & (pr >= P90) & (need >= ND75) & vok
    zexp, _ = label_zones(sites, min_pixels=23)
    zgot = band(p7 / "v2_constrained" / "rasters" / "stable_zone_ids_v2_constrained.tif")
    ok2 = bool(np.array_equal(zexp > 0, zgot > 0))
    ok_all &= check("2", "site set == 2026 feasible & S2 gates & veg<0.30 "
                    "recompute (constants pinned)", ok2)

    # [3] score + shortlist rule on sampled sites
    zdf = pd.read_csv(p7 / "v2_constrained" / "zones" / "stable_zones_v2_constrained.csv")
    R = p7rec["pooled_references"]
    prod = load_phase3_year(Path(args.phase3_root), 2026)
    n_lst = 100.0 * pooled_normalize(prod["lst_C"], R["lst"]["p1"], R["lst"]["p99"])
    n_1v = 100.0 * pooled_normalize(1.0 - prod["vegetation_cover"],
                                    1.0 - R["veg"]["p99"], 1.0 - R["veg"]["p1"])
    n_ndvi = 100.0 * pooled_normalize(prod["ndvi"], R["ndvi"]["p1"], R["ndvi"]["p99"])
    n_ndbi = 100.0 * pooled_normalize(prod["ndbi"], R["ndbi"]["p1"], R["ndbi"]["p99"])
    og = np.clip((0.30 * landuse_eligibility(lu) / 100.0
                  + 0.25 * (100.0 - n_ndbi) / 100.0
                  + 0.15 * road_accessibility_score(static["dist_road_m"]) / 100.0
                  + 0.15 * green_proximity_score(static["dist_vegetation_m"]) / 100.0
                  + 0.15 * (100.0 - n_ndvi) / 100.0) * 100.0, 0, 100)
    rng = np.random.default_rng(5)
    ok3 = True
    zsorted = zdf.sort_values("rank").reset_index(drop=True)
    rank_excl = (~zsorted["oversized"]).cumsum()
    exp_short = ((rank_excl <= 100) & (~zsorted["oversized"])).to_numpy()
    zdf = zsorted
    for z in rng.choice(zdf["zone_id"].to_numpy(), size=25, replace=False):
        m = zgot == int(z)
        exp = (0.50 * float(np.nanmean(n_lst[m])) + 0.25 * float(np.nanmean(n_1v[m]))
               + 0.25 * float(np.nanmean(og[m])))
        got = float(zdf.loc[zdf["zone_id"] == int(z), "planning_priority_score"].iloc[0])
        ok3 &= abs(exp - got) <= 0.06
        row = zdf[zdf["zone_id"] == int(z)].iloc[0]
        ok3 &= bool(row["shortlist"]) == bool(exp_short[int(row.name)])
    ok_all &= check("3", "score recompute (0.50/0.25/0.25) + shortlist rule "
                    "(top-100 non-oversized)", ok3)

    # [4] usable exactness + floor arithmetic + certainty
    zc = pd.read_csv(out_root / "v2_constrained" / "tables"
                     / "tree_requirement_by_zone_v2_constrained_2026.csv")
    ok4 = bool((zc["usable_area_ha"] == (zc["zone_px"] * 0.09).round(3)).all())
    for d, col in ((400, "reference_trees_400"), (1000, "reference_trees_1000"),
                   (2500, "reference_trees_2500")):
        ok4 &= bool((zc[col] == (zc["usable_area_ha"] * d).astype(int)).all())
    exp_cert = np.where(zc["untagged_share"] > 0.5, "low",
                        np.where(zc["untagged_share"] > 0.2, "mixed", "identified"))
    ok4 &= bool((zc["landuse_certainty"].to_numpy() == exp_cert).all())
    ok_all &= check("4", "usable == site px; floor arithmetic exact; certainty rule", ok4)

    # [5] artifacts
    need_files = [out_root / "phase8_manifest.json", out_root / "phase8_pipeline_record.json",
                  out_root / "v2_constrained" / "tables" / "recommended_sites_v2_constrained_2026.csv"]
    missing = [str(f) for f in need_files if not f.exists()]
    ok_all &= check("5", "artifacts complete", not missing,
                    f"missing={missing}" if missing else "all present")

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
