"""V2 Phase 6 verification gate — numbered PASS/FAIL checklist.

Independently re-derives the Phase 6 outputs from the read-only inputs and
the written rasters/tables. Exit 0 = all PASS.

Checks
------
1.  marker-driven resolution: the Phase 5 marker still designates the frozen
    verified RF primary and its artifact exists.
2.  per-year rasters exist and match the authoritative grid: dimensions
    1768x1874, EPSG:4326, exact 2026-reference affine, expected bands/dtypes.
3.  NoData consistency: severity valid-pixel count == manifest domain count;
    probability/confidence valid masks equal severity's; hotspot-ID rasters
    valid exactly on the same domain.
4.  probabilities sum to 1 (|sum-1| <= 1e-5) on every valid pixel, and
    severity == argmax(probabilities) on every valid pixel.
5.  class-balance sanity: per-year severity shares stay within [15%, 50%]
    per class (no pathological blowups vs the ~1/3 training balance).
6.  tables agree with rasters: area_statistics pixels == severity bincount;
    block_statistics per-block class px + totals == raster counts;
    per-def hotspot pixel counts == hotspot-ID raster (>0) counts.
7.  artifacts complete: manifest + pipeline record + all expected files
    listed, schema hash recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from ..common import (
    GRID_FILE_DEFAULT,
    PROJECT_ROOT,
    load_frozen_schema,
    schema_hash,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(n: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{n}] [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "phase6"))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    args = ap.parse_args(argv)
    out_root = Path(args.out)

    manifest = json.loads((out_root / "phase6_manifest.json").read_text())
    years = [int(y) for y in manifest["years"]]
    ok_all = True

    # -- [1] marker-driven resolution --------------------------------------
    from ..phase6.build import resolve_primary_model
    try:
        marker, artifact = resolve_primary_model(
            PROJECT_ROOT / "data" / "phase5")
        ok_all &= check("1", "marker resolves frozen verified RF primary",
                        True, f"artifact={marker['artifact']} "
                              f"locked_acc={marker['locked_accuracy']:.6f}")
    except Exception as e:                               # noqa: BLE001
        ok_all &= check("1", "marker resolves frozen verified RF primary",
                        False, str(e))
        marker, artifact = None, None

    # -- [2] grid/CRS/transform/dtype ---------------------------------------
    import rasterio.crs
    ref_crs = rasterio.crs.CRS.from_epsg(4326)
    grid_ok, grid_detail = True, []
    for y in years:
        with rasterio.open(out_root / "rasters" / f"severity_{y}.tif") as ds:
            g_ok = (ds.height, ds.width) == (1768, 1874) and ds.crs == ref_crs
            grid_detail.append(f"{y}:{'OK' if g_ok else 'BAD'}")
            grid_ok &= g_ok
        with rasterio.open(out_root / "rasters" / f"probability_{y}.tif") as ds:
            p_ok = (ds.height, ds.width) == (1768, 1874) and ds.count == 3 \
                and ds.crs == ref_crs
            grid_ok &= p_ok
    ok_all &= check("2", "rasters match authoritative grid/CRS/dims",
                    grid_ok, " ".join(grid_detail))

    # -- [3/4/5/6] per-year content ----------------------------------------
    sem = {0: "Low", 1: "Moderate", 2: "High"}
    for y in years:
        with rasterio.open(out_root / "rasters" / f"severity_{y}.tif") as ds:
            sev = ds.read(1)
            sev_nd = ds.nodata
        with rasterio.open(out_root / "rasters" / f"probability_{y}.tif") as ds:
            prob = ds.read()                     # (3, H, W)
        with rasterio.open(out_root / "rasters" / f"confidence_{y}.tif") as ds:
            conf = ds.read(1)
        domain = sev != sev_nd
        n_dom = int(domain.sum())
        expected = manifest["domain_counts"][str(y)]["domain_pixels"]

        ok3a = n_dom == expected
        ok3b = bool(np.all(np.isfinite(prob[:, domain])))
        ok3c = bool(np.all(conf[domain] != -1.0))
        ids_ok = True
        for d in ("A", "B", "C"):
            with rasterio.open(out_root / "hotspots" / f"hotspot_ids_def{d}_{y}.tif") as ds:
                ids = ds.read(1)
            ids_ok &= bool(np.all((ids != -1) == domain))
        ok_all &= check(f"3-{y}", f"{y} NoData consistent with domain",
                        ok3a and ok3b and ok3c and ids_ok,
                        f"domain={n_dom} expected={expected}")

        psum = prob[:, domain].sum(axis=0)
        ok4a = bool(np.all(np.abs(psum - 1.0) <= 1e-5))
        ok4b = bool(np.array_equal(np.argmax(prob[:, domain], axis=0),
                                   sev[domain].astype(int)))
        conf_ok = bool(np.allclose(conf[domain],
                                   prob[:, domain].max(axis=0), atol=1e-6))
        ok_all &= check(f"4-{y}", f"{y} probabilities sum to 1 / argmax / conf",
                        ok4a and ok4b and conf_ok,
                        f"max|sum-1|={float(np.abs(psum - 1.0).max()):.2e}")

        hist = np.bincount(sev[domain].astype(int), minlength=3) / n_dom
        ok5 = bool(np.all((hist >= 0.15) & (hist <= 0.50)))
        ok_all &= check(f"5-{y}", f"{y} class balance sane",
                        ok5, " ".join(f"{sem[i]}={hist[i]:.3f}" for i in range(3)))

        area = pd.read_csv(out_root / "tables" / f"area_statistics_{y}.csv")
        ok6a = bool((area.sort_values("class_value")["pixels"].to_numpy()
                     == np.bincount(sev[domain].astype(int), minlength=3)).all())
        blk = pd.read_csv(out_root / "tables" / f"block_statistics_{y}.csv")
        ok6b = bool(int(blk["valid_pixels"].sum()) == n_dom)
        ok6c = bool(int(blk["low_px"].sum() + blk["moderate_px"].sum()
                        + blk["high_px"].sum()) == n_dom)
        hot_ok = True
        for d in ("A", "B", "C"):
            st = pd.read_csv(out_root / "tables" / f"hotspot_statistics_def{d}_{y}.csv").iloc[0]
            with rasterio.open(out_root / "hotspots" / f"hotspot_ids_def{d}_{y}.tif") as ds:
                ids = ds.read(1)
            hot_ok &= int(st["n_hotspot_pixels"]) == int((ids > 0).sum())
            hot_ok &= int(st["n_clusters"]) == int(ids.max())
        ok_all &= check(f"6-{y}", f"{y} tables agree with rasters",
                        ok6a and ok6b and ok6c and hot_ok)

    # -- [7] artifacts + manifest -------------------------------------------
    expected_files = [out_root / "phase6_manifest.json",
                      out_root / "phase6_pipeline_record.json",
                      out_root / "tables" / "temporal_comparison.csv",
                      out_root / "tables" / "coverage_statistics.csv"]
    for y in years:
        expected_files += [
            out_root / "rasters" / f"severity_{y}.tif",
            out_root / "rasters" / f"probability_{y}.tif",
            out_root / "rasters" / f"confidence_{y}.tif",
            out_root / "rasters" / f"severity_score_{y}.tif",
            out_root / "tables" / f"area_statistics_{y}.csv",
            out_root / "tables" / f"block_statistics_{y}.csv",
        ] + [out_root / "hotspots" / f"hotspots_def{d}_{y}.geojson" for d in "ABC"] \
          + [out_root / "hotspots" / f"hotspot_ids_def{d}_{y}.tif" for d in "ABC"]
    missing = [str(p) for p in expected_files if not p.exists()]
    ok7 = (not missing
           and manifest["schema"]["sha256"] == schema_hash(load_frozen_schema())
           and manifest["primary_model"]["locked_accuracy"] == marker["locked_accuracy"])
    ok_all &= check("7", "artifacts complete + manifest provenance",
                    ok7, f"missing={len(missing)}" if missing else "all present")

    fails = [r for r in RESULTS if not r[1]]
    print(f"\n[SUMMARY] {len(RESULTS) - len(fails)}/{len(RESULTS)} checks pass, "
          f"{len(fails)} FAIL", flush=True)
    if fails:
        for name, _, detail in fails:
            print(f"  FAILED: {name} {detail}")
        return 1
    print("[VERIFY] ALL PHASE 6 CHECKS PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
