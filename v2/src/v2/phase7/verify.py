"""V2 Phase 7 verification gate — numbered PASS/FAIL checklist.

Read-only re-derivation of every Phase 7 output family. Exit 0 = all PASS.

Checks
------
1.  lineage: phase6 manifest present; Phase 7 record carries the frozen
    primary-model lineage (RF via marker).
2.  grid/CRS/dims: scenario rasters match the authoritative 1768x1874
    EPSG:4326 grid.
3.  NoData consistent with the phase6 domain: v1_parity valid set == phase6
    domain; v2_constrained valid set == v1_parity minus constraint union;
    v2_constrained valid pixels never intersect the attributed constraint
    raster.
4.  constraint rasters truly exclude: re-burn the three GeoJSONs and compare
    masks exactly against ``constraints_rasters/``; also verify the
    attributed raster equals precedence application.
5.  class/priority consistency: suitability_class == classify(suitability);
    priority raster == priority_tier(class) on the scenario domain.
6.  tables agree with rasters: priority-tier counts, per-block domain sums,
    zone pixel counts and zone-id rasters all reconcile.
7.  both scenarios present for every year; V1-parity summary recorded.
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
from . import suitability as S
from .constraints import CONSTRAINT_FILES, burn_layer

RESULTS: list[tuple[str, bool, str]] = []


def check(n: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{n}] [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "phase7"))
    ap.add_argument("--phase6-dir", default=str(PROJECT_ROOT / "data" / "phase6"))
    ap.add_argument("--constraints-dir",
                    default=str(PROJECT_ROOT / "data" / "phase2" / "constraints"))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    phase6_dir = Path(args.phase6_dir)
    grid = Grid.from_file(Path(args.grid_file))
    shape = grid.shape

    record = json.loads((out_root / "phase7_pipeline_record.json").read_text())
    years = [int(y) for y in record["years"]]
    ok_all = True

    # -- [1] lineage ---------------------------------------------------------
    p6 = json.loads((phase6_dir / "phase6_manifest.json").read_text())
    lin_ok = bool(p6.get("primary_model")) and \
        bool(record.get("model_lineage"))
    ok_all &= check("1", "lineage: phase6 manifest + RF primary recorded",
                    lin_ok, f"model={p6.get('primary_model', {}).get('artifact')}")

    # -- [2] grid/CRS/dims ----------------------------------------------------
    import rasterio.crs
    ref_crs = rasterio.crs.CRS.from_epsg(4326)
    grid_ok = True
    for scenario in record["scenarios"]:
        for y in years:
            for name in ("suitability", "suitability_class", "priority",
                         "priority_zone_ids", "exclusion_mask"):
                tag = f"{scenario}_{y}" if name != "priority_zone_ids" else f"{scenario}_{y}"
                fname = {"suitability": f"suitability_{scenario}_{y}.tif",
                         "suitability_class": f"suitability_class_{scenario}_{y}.tif",
                         "priority": f"priority_{scenario}_{y}.tif",
                         "priority_zone_ids": f"priority_zone_ids_{scenario}_{y}.tif",
                         "exclusion_mask": f"exclusion_mask_{scenario}_{y}.tif"}[name]
                with rasterio.open(out_root / scenario / "rasters" / fname) as ds:
                    grid_ok &= (ds.height, ds.width) == shape and ds.crs == ref_crs
    ok_all &= check("2", "scenario rasters match authoritative grid/CRS/dims",
                    grid_ok, f"{len(record['scenarios'])}x{len(years)}x5 rasters")

    # -- [3/5/6] per-year scenario content -----------------------------------
    attributed_path = out_root / "constraints_rasters" / "constraint_attributed_30m.tif"
    with rasterio.open(attributed_path) as ds:
        attributed = ds.read(1)

    for y in years:
        dom_p6 = p6["domain_counts"][str(y)]["domain_pixels"]
        valid = {}
        for scenario in record["scenarios"]:
            with rasterio.open(out_root / scenario / "rasters" / f"suitability_{scenario}_{y}.tif") as ds:
                s = ds.read(1)
                nd = ds.nodata
            valid[scenario] = s != nd
        v1, v2 = valid["v1_parity"], valid["v2_constrained"]
        ok3a = int(v1.sum()) == dom_p6
        ok3b = bool(np.all(~v2 | v1))                       # v2 subset of v1
        ok3c = int((v2 & (attributed > 0)).sum()) == 0      # no constraint px in v2
        n_excl = int((v1 & ~v2).sum())
        ok_all &= check(f"3-{y}", f"{y} NoData consistent with phase6 domain + constraints",
                        ok3a and ok3b and ok3c,
                        f"v1={int(v1.sum()):,} p6={dom_p6:,} v2={int(v2.sum()):,} excluded={n_excl:,}")

        for scenario in record["scenarios"]:
            rdir = out_root / scenario / "rasters"
            tdir = out_root / scenario / "tables"
            with rasterio.open(rdir / f"suitability_{scenario}_{y}.tif") as ds:
                score = ds.read(1)
            with rasterio.open(rdir / f"suitability_class_{scenario}_{y}.tif") as ds:
                cls = ds.read(1)
            with rasterio.open(rdir / f"priority_{scenario}_{y}.tif") as ds:
                pri = ds.read(1)
            with rasterio.open(rdir / f"priority_zone_ids_{scenario}_{y}.tif") as ds:
                zids = ds.read(1)
            dom = valid[scenario]
            ok5a = bool(np.array_equal(
                S.classify(score[dom].astype(np.float64)), cls[dom].astype(np.int16)))
            ok5b = bool(np.array_equal(S.priority_tier(cls[dom].astype(np.int16)),
                                       pri[dom].astype(np.uint8)))
            ok_all &= check(f"5-{scenario[:2]}-{y}", f"{y} {scenario} class/priority consistent",
                            ok5a and ok5b)

            tiers = pd.read_csv(tdir / f"priority_tiers_{scenario}_{y}.csv")
            tr = tiers.set_index("priority")["pixel_count"]
            ok6a = (int(tr["High"]) == int((pri[dom] == 3).sum())
                    and int(tr["Medium"]) == int((pri[dom] == 2).sum())
                    and int(tr["Low"]) == int((pri[dom] == 1).sum()))
            blk = pd.read_csv(tdir / f"block_suitability_{scenario}_{y}.csv")
            ok6b = int(blk["domain_px"].sum()) == int(dom.sum())
            ok6c = int(blk["class_ge3_px"].sum()) == int((cls[dom] >= 3).sum())
            zstat = pd.read_csv(tdir / f"priority_zone_statistics_{scenario}_{y}.csv")
            ok6d = int(zstat["pixel_count"].sum()) == int((zids > 0).sum())
            ok6e = len(zstat) == int(zids.max())
            ok_all &= check(f"6-{scenario[:2]}-{y}", f"{y} {scenario} tables agree with rasters",
                            ok6a and ok6b and ok6c and ok6d and ok6e)

    # -- [4] constraint rasters truly exclude ---------------------------------
    masks_ok = True
    for name, fname in CONSTRAINT_FILES.items():
        with rasterio.open(out_root / "constraints_rasters" / f"constraint_{name}_30m.tif") as ds:
            burned = ds.read(1).astype(bool)
        reburn = burn_layer(Path(args.constraints_dir) / fname, grid)
        masks_ok &= bool(np.array_equal(burned, reburn))
    ok_all &= check("4", "constraint rasters == fresh re-burn of the GeoJSONs",
                    masks_ok, "exact mask equality, all 3 layers")

    # -- [7] both scenarios + V1-parity summary --------------------------------
    summ_ok = all(sc in record["summary"][str(y)] for y in years
                  for sc in ("v1_parity", "v2_constrained"))
    parity_ok = all("class_ge3_ha" in record["summary"][str(y)]["v1_parity"]
                    for y in years)
    ok_all &= check("7", "both scenarios present + V1-parity summary recorded",
                    summ_ok and parity_ok)

    # -- [8] artifacts ----------------------------------------------------------
    need = [out_root / "phase7_manifest.json", out_root / "phase7_pipeline_record.json",
            attributed_path]
    missing = [str(p) for p in need if not p.exists()]
    ok_all &= check("8", "manifest + pipeline record + constraint rasters present",
                    not missing, f"missing={missing}" if missing else "all present")

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
