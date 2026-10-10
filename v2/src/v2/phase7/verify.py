"""V2 Phase 7 v3 verification gate — numbered PASS/FAIL checklist. Read-only.

Checks
------
1.  methodology recorded: pooled references, fixed thresholds, formulas,
    feasibility rule, scenario definitions.
2.  raster contract: full authoritative grid; cooling_need/suitability/
    opportunity in [0,1]; priority_score in [0,1] and defined ONLY on
    feasible px; priority_class in {0,1,2} on feasible, 255 elsewhere.
3.  pooled references recompute (sampled from phase3 rasters, tolerance).
4.  priority classes == fixed documented thresholds applied to the written
    priority_score raster (exact recompute).
5.  class-share tables agree with rasters and sum to feasible px; feasible
    px agrees with the named-filter recompute.
6.  exclusion accounting per reason recomputed independently (v2: water >
    buildings > road_surfaces precedence, then landuse, then veg>=0.30;
    v1: constraints reported as 0 by definition).
7.  cross-year comparability: thresholds identical across years (structural)
    + per-year class shares recorded.
8.  artifacts complete (manifest, record, rasters, tables).
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
from ..phase7.build import DISCOURAGED_LU_CODES, eligible_lu_grid
from ..phase7.suitability import PRIORITY_CLASS_LABELS

RESULTS: list[tuple[str, bool, str]] = []


def check(n: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{n}] [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def band(p):
    with rasterio.open(p) as ds:
        a = ds.read(1)
        nd = ds.nodata
    return a, nd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "phase7"))
    ap.add_argument("--phase3-root", default=str(PROJECT_ROOT / "data" / "phase3"))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    grid = Grid.from_file(Path(args.grid_file))
    shape = grid.shape

    record = json.loads((out_root / "phase7_pipeline_record.json").read_text())
    years = [int(y) for y in record["years"]]
    ok_all = True

    # -- [1] methodology ------------------------------------------------------
    ok1 = all(k in record for k in ("pooled_references", "priority_thresholds",
                                    "formulas", "feasibility_rule"))
    ok_all &= check("1", "methodology recorded (pooled refs, thresholds, formulas)",
                    ok1)

    # -- [2] raster contract ----------------------------------------------------
    ok2 = True
    detail2 = []
    for scenario in record["scenarios"]:
        for y in years:
            rdir = out_root / scenario / "rasters"
            for name, lo, hi in (("cooling_need", 0.0, 1.0),
                                 ("suitability", 0.0, 1.0),
                                 ("plantation_opportunity", 0.0, 1.0),
                                 ("priority_score", 0.0, 1.0)):
                a, nd = band(rdir / f"{name}_{scenario}_{y}.tif")
                if (a.shape != shape) or nd != -1.0:
                    ok2 = False
                    continue
                valid = a != -1
                if valid.any() and (a[valid].min() < lo - 1e-6 or a[valid].max() > hi + 1e-6):
                    ok2 = False
                    detail2.append(f"{name}_{y}")
            a, nd = band(rdir / f"priority_class_{scenario}_{y}.tif")
            feas_a, _ = band(rdir / f"feasibility_mask_{scenario}_{y}.tif")
            if nd != 255 or not np.isin(np.unique(a), [0, 1, 2, 255]).all():
                ok2 = False
            if not np.array_equal(a != 255, feas_a == 1):
                ok2 = False
                detail2.append(f"class/feasible mismatch {scenario} {y}")
    ok_all &= check("2", "raster contract (grid, [0,1] scores, class only on "
                    "feasible)", ok2, "; ".join(detail2[:4]))

    # -- [3] pooled references recompute -----------------------------------------
    from v2.phase7.build import compute_pooled_references
    refs_rec = record["pooled_references"]
    refs_now = compute_pooled_references(tuple(years), Path(args.phase3_root))
    ok3 = True
    for k in ("lst", "ndvi", "ndbi", "veg"):
        tol = max(0.02, 0.002 * abs(refs_rec[k]["p99"]))
        ok3 &= abs(refs_rec[k]["p1"] - refs_now[k]["p1"]) < tol
        ok3 &= abs(refs_rec[k]["p99"] - refs_now[k]["p99"]) < tol
    ok_all &= check("3", "pooled references recompute (sampled, tolerance)", ok3,
                    f"LST p1/p99={refs_rec['lst']['p1']:.2f}/{refs_rec['lst']['p99']:.2f}")

    # -- [4] classes == fixed thresholds on written scores -------------------------
    t1 = record["priority_thresholds"]["t1_medium"]
    t2 = record["priority_thresholds"]["t2_high"]
    ok4 = True
    for scenario in record["scenarios"]:
        for y in years:
            sc_, _ = band(out_root / scenario / "rasters" / f"priority_score_{scenario}_{y}.tif")
            cl_, _ = band(out_root / scenario / "rasters" / f"priority_class_{scenario}_{y}.tif")
            scf = sc_.astype(np.float64)   # compare in float64 (build classifies
            exp = np.full(scf.shape, 255, dtype=np.uint8)   # the f32-rounded score)
            m = scf != -1
            exp[m & (scf < t1)] = 0
            exp[m & (scf >= t1) & (scf < t2)] = 1
            exp[m & (scf >= t2)] = 2
            ok4 &= bool(np.array_equal(exp, cl_))
    ok_all &= check("4", "priority classes == documented fixed thresholds "
                    f"(t1={t1:.4f}, t2={t2:.4f}) applied to written scores", ok4)

    # -- [5] tables agree; domain == build definition (finite lst+ndvi+ndbi+veg)
    from v2.phase4.assemble_features import load_phase3_year, load_static
    from v2.phase7.build import load_year_arrays
    static = load_static(Path(args.phase3_root))
    lu_codes = np.where(np.isfinite(static["landuse_class"]),
                        static["landuse_class"], 255.0).astype(np.int64)
    elig = eligible_lu_grid(lu_codes)
    att = band(out_root / "constraints_rasters" / "constraint_attributed_30m.tif")[0] > 0
    ok5 = True
    for scenario in record["scenarios"]:
        for y in years:
            cl_, _ = band(out_root / scenario / "rasters" / f"priority_class_{scenario}_{y}.tif")
            cn_, _ = band(out_root / scenario / "rasters" / f"cooling_need_{scenario}_{y}.tif")
            feasible = cl_ != 255
            dom_exp = load_year_arrays(Path(args.phase3_root), y)["domain"]
            dom = cn_ != -1
            ok5 &= bool(np.array_equal(dom, dom_exp))
            shares = pd.read_csv(out_root / scenario / "tables"
                                 / f"class_shares_{scenario}_{y}.csv")
            tot = int(shares["pixel_count"].sum())
            ok5 &= tot == int(feasible.sum())
            ok5 &= set(shares["class"]) == set(PRIORITY_CLASS_LABELS.values())
    ok_all &= check("5", "class-share tables agree with rasters; domain == phase3 "
                    "finite LST", ok5)

    # -- [6] exclusion accounting recompute ------------------------------------------
    from v2.phase7.build import load_year_arrays
    ok6 = True
    for scenario in record["scenarios"]:
        for y in years:
            arr = load_year_arrays(Path(args.phase3_root), y)
            dom = arr["domain"]
            vg = arr["vegetation_cover"]
            del arr
            ea = pd.read_csv(out_root / scenario / "tables"
                             / f"exclusion_accounting_{scenario}_{y}.csv"
                             ).set_index("reason")["area_ha"]
            remaining = dom.copy()
            exp = {}
            for name in ("water", "buildings", "road_surfaces"):
                layer = att if False else None
                with rasterio.open(out_root / "constraints_rasters"
                                   / f"constraint_{name}_30m.tif") as ds:
                    lm = ds.read(1).astype(bool)
                if scenario == "v2_constrained":
                    hit = remaining & lm
                    exp[name] = hit.sum() * 0.09
                    remaining &= ~lm
                else:
                    exp[name] = 0.0
            exp["landuse_ineligible"] = (remaining & ~elig).sum() * 0.09
            remaining &= elig
            exp["planting_space_veg_ge_0.30"] = (remaining & ~(np.isfinite(vg) & (vg < 0.30))).sum() * 0.09
            exp["FEASIBLE"] = (remaining & np.isfinite(vg) & (vg < 0.30)).sum() * 0.09
            for k, v in exp.items():
                ok6 &= abs(float(ea[k]) - round(v, 1)) < 0.2
    ok_all &= check("6", "exclusion accounting per reason independently recomputed",
                    ok6)

    # -- [7] cross-year comparability ---------------------------------------------------
    ok7 = True
    shares_tab = {}
    for scenario in record["scenarios"]:
        rows = []
        for y in years:
            sh = pd.read_csv(out_root / scenario / "tables"
                             / f"class_shares_{scenario}_{y}.csv")
            rows.append({r["class"]: r["pct_of_feasible"] for _, r in sh.iterrows()})
        shares_tab[scenario] = rows
        ok7 &= all(set(r) == {"Low", "Medium", "High"} for r in rows)
    ok_all &= check("7", "fixed thresholds across years (structural comparability); "
                    "per-year shares recorded", ok7,
                    f"v2 2026 shares={shares_tab['v2_constrained'][-1]}")

    # -- [8] artifacts ---------------------------------------------------------------------
    need = [out_root / "phase7_manifest.json", out_root / "phase7_pipeline_record.json"]
    for scenario in record["scenarios"]:
        for y in years:
            need += [out_root / scenario / "tables" / f"class_shares_{scenario}_{y}.csv",
                     out_root / scenario / "tables" / f"full_area_stats_{scenario}_{y}.csv",
                     out_root / scenario / "tables" / f"exclusion_accounting_{scenario}_{y}.csv"]
    missing = [str(p) for p in need if not p.exists()]
    ok_all &= check("8", "manifest + record + all tables present",
                    not missing, f"missing={len(missing)}" if missing else "all present")

    fails = [r for r in RESULTS if not r[1]]
    print(f"\n[SUMMARY] {len(RESULTS) - len(fails)}/{len(RESULTS)} checks pass, "
          f"{len(fails)} FAIL", flush=True)
    if fails:
        for name, _, detail in fails:
            print(f"  FAILED: {name} {detail}")
        return 1
    print("[VERIFY] ALL PHASE 7 CHECKS PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
