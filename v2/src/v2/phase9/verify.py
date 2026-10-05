"""V2 Phase 9 verification gate — numbered PASS/FAIL checklist.

Read-only re-derivation of every Phase 9 output family. Exit 0 = all PASS.

Checks
------
1.  panel integrity: delta_lst_C == next - current exactly; spatial_block_id
    within 0..24; transitions exactly the 4 expected; no NaN in
    features/target; met covariates present.
2.  leakage protocol: locked blocks {2,9,15,23} present in the panel;
    adjacent-block CV val_blocks contain no locked id; locked predictions
    file exists per model; leakage audit json exists and names the
    intervention variable + met covariates.
3.  metrics sanity: R2 within [-1, 1] across all eval tables; MAE <= RMSE
    (tolerance 1e-6); n counts positive and CV/LOBO n sums reconcile with
    the panel block/row structure.
4.  scenario integrity: zone table tree counts == phase8 plantable_ha x
    density exactly (recomputed independently from the phase8 CSV); zone ids
    match phase8; raster contract (1768x1874, EPSG:4326, nodata -9999,
    plantable-pixel support only).
5.  wording/framing: manifest carries non-causal framing, planning-density
    disclaimer, and the 0.05 scenario-assumption note; report exists.
6.  immutability: phase8 input file sha256 match the manifest input_hashes.
7.  artifacts complete: manifest, report, all tables/rasters/models listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from ..common import GRID_FILE_DEFAULT, HOLDOUT_BLOCKS, PROJECT_ROOT, Grid

RESULTS: list[tuple[str, bool, str]] = []


def check(n: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{n}] [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "phase9"))
    ap.add_argument("--phase8-root", default=str(PROJECT_ROOT / "data" / "phase8"))
    ap.add_argument("--report", default=str(
        PROJECT_ROOT / "reports" / "phase9_temperature_reduction_prediction_report.md"))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    phase8_root = Path(args.phase8_root)
    grid = Grid.from_file(Path(args.grid_file))

    manifest = json.loads((out_dir / "phase9_manifest.json").read_text())
    ok_all = True

    # -- [1] panel integrity ----------------------------------------------------
    panel = pd.read_parquet(out_dir / "tables" / "training_panel.parquet")
    ok1a = bool(np.allclose(
        panel["delta_lst_C"].to_numpy(),
        (panel["next_lst_C"] - panel["current_lst_C"]).to_numpy(),
        rtol=0, atol=0, equal_nan=False))
    ok1b = bool(panel["spatial_block_id"].between(0, 24).all())
    ok1c = set(panel["transition"].unique()) == {
        "2022->2023", "2023->2024", "2024->2025", "2025->2026"}
    feat_cols = list(manifest["features"])
    ok1d = not panel[feat_cols + ["delta_lst_C"]].isna().any().any()
    ok1e = all(c.startswith("current_met_") for c in feat_cols
               if c.startswith("current_met"))
    ok1f = sum(c.startswith("current_met_") for c in feat_cols) == 6
    ok_all &= check("1", "panel integrity (delta exact / blocks / transitions /"
                    " no-NaN / met covariates)",
                    ok1a and ok1b and ok1c and ok1d and ok1e and ok1f,
                    f"rows={len(panel):,}")

    # -- [2] leakage protocol -----------------------------------------------------
    locked_ids = set(int(b) for b in HOLDOUT_BLOCKS)
    ok2a = locked_ids.issubset(set(panel["spatial_block_id"].unique().tolist()))
    ok2b = True
    cv_n_sum = 0
    for name in manifest["models"]:
        cv = pd.read_csv(out_dir / "models" / f"{name}_adjacent_block_cv.csv")
        for vb in cv["val_blocks"]:
            if locked_ids & {int(x) for x in str(vb).split(",")}:
                ok2b = False
        cv_n_sum = int(cv["n"].sum())
    panel_nonlocked = int((~panel["spatial_block_id"].isin(sorted(locked_ids))).sum())
    ok2c = cv_n_sum == panel_nonlocked
    ok2d = all((out_dir / "models" / f"{name}_locked_predictions.csv").exists()
               for name in manifest["models"])
    audit = json.loads((out_dir / "diagnostics" / "leakage_audit.json").read_text())
    audit_txt = json.dumps(audit)
    ok2e = "intervention_variable" in audit_txt and "current_met_" in audit_txt
    ok_all &= check("2", "leakage protocol (CV excludes locked / locked files /"
                    " audit complete)", ok2a and ok2b and ok2c and ok2d and ok2e,
                    f"cv_n={cv_n_sum:,} nonlocked={panel_nonlocked:,}")

    # -- [3] metrics sanity --------------------------------------------------------
    ok3 = True
    detail3 = []
    for name, res in manifest["models"].items():
        tables = [res.get("adjacent_block_cv"), res.get("lobo"),
                  res.get("time_holdout")]
        lk = res.get("locked_holdout")
        if lk:
            tables.append({"folds": 1, "mae_C_mean": lk["mae_C"],
                           "rmse_C_mean": lk["rmse_C"], "r2_mean": lk["r2"],
                           "n_total_eval_rows": lk["n"]})
        for agg in tables:
            if not agg:
                continue
            if not (-1.0 - 1e-9 <= agg["r2_mean"] <= 1.0 + 1e-9):
                ok3 = False
                detail3.append(f"{name}:r2={agg['r2_mean']:.3f}")
            if agg["mae_C_mean"] - agg["rmse_C_mean"] > 1e-6:
                ok3 = False
                detail3.append(f"{name}:mae>rmse")
            if agg["n_total_eval_rows"] <= 0:
                ok3 = False
        lobo = pd.read_csv(out_dir / "models" / f"{name}_lobo.csv")
        if int(lobo["n"].sum()) != int(len(panel)):
            ok3 = False
            detail3.append(f"{name}:lobo_n")
    ok_all &= check("3", "metrics sanity (R2 in [-1,1], MAE<=RMSE, n reconcile)",
                    ok3, "; ".join(detail3))

    # -- [4] scenario integrity ------------------------------------------------------
    pred = pd.read_csv(out_dir / "tables" / "phase9_zone_cooling_predictions_2026.csv")
    p8 = pd.read_csv(phase8_root / "v2_constrained" / "tables"
                     / "tree_requirement_by_zone_v2_constrained_2026.csv")
    p8_area = p8.set_index("zone_id")["plantable_ha"].to_dict()
    ok4a = set(pred["zone_id"].unique()) == set(p8["zone_id"].unique())
    ok4b = bool(np.all([
        int(round(float(p8_area[int(r.zone_id)]) * int(r.tree_density_trees_per_ha)))
        == int(r.estimated_tree_count)
        for r in pred.itertuples()]))
    rast_ok = True
    plant = None
    with rasterio.open(phase8_root / "v2_constrained" / "rasters"
                       / "available_planting_space_v2_constrained_2026.tif") as ds:
        plant = ds.read(1) == 1
    for density in (400, 1000, 2500):
        for kind in ("predicted_delta_lst", "predicted_cooling", "predicted_post_lst"):
            with rasterio.open(out_dir / "rasters"
                               / f"{kind}_v2_constrained_2026_{density}trees_ha.tif") as ds:
                arr = ds.read(1)
                ok = ((ds.height, ds.width) == (grid.height, grid.width)
                      and ds.crs.to_epsg() == 4326 and ds.nodata == -9999.0)
                ok &= bool(np.array_equal(np.isfinite(arr) & (arr != -9999.0), plant))
                ok &= bool(np.isfinite(arr[plant]).all())
                rast_ok &= ok
    ok_all &= check("4", "scenario integrity (trees==ha x density, zone ids, "
                    "raster contract + plantable-only support)",
                    ok4a and ok4b and rast_ok)

    # -- [5] wording / framing ---------------------------------------------------------
    mtxt = json.dumps(manifest)
    rt = Path(args.report).read_text(encoding="utf-8")
    ok5 = ("assumed" in mtxt.lower() and "not proof of causal" in mtxt.lower()
           and "planning" in rt.lower()
           and "not air-temperature guarantees" in rt
           and "0.05" in rt
           and "not scientifically optimal" in rt
           and Path(args.report).exists())
    ok_all &= check("5", "wording/framing (non-causal, planning-density disclaimer,"
                    " 0.05 assumption, LST-not-air-temp)", ok5)

    # -- [6] phase8 immutability ----------------------------------------------------------
    ok6 = True
    bad = []
    for path_str, expected in manifest.get("input_hashes", {}).items():
        from ..common import sha256_file
        got = sha256_file(Path(path_str))
        if got != expected:
            ok6 = False
            bad.append(Path(path_str).name)
    ok_all &= check("6", "phase8 input immutability (sha256 == manifest)", ok6,
                    f"checked={len(manifest.get('input_hashes', {}))} "
                    f"mismatch={bad}" if bad else "")

    # -- [7] artifacts -----------------------------------------------------------------------
    need = [out_dir / "phase9_manifest.json", Path(args.report),
            out_dir / "tables" / "training_panel.parquet",
            out_dir / "tables" / "panel_summary.csv",
            out_dir / "tables" / "phase9_zone_cooling_predictions_2026.csv",
            out_dir / "diagnostics" / "leakage_audit.json",
            out_dir / "diagnostics" / "sanity_checks.json",
            out_dir / "diagnostics" / "vegetation_change_sensitivity.csv"]
    for name in manifest["models"]:
        need += [out_dir / "models" / f"{name}_model.joblib",
                 out_dir / "models" / f"{name}_feature_importance.csv"]
    missing = [str(p) for p in need if not p.exists()]
    ok_all &= check("7", "manifest + report + tables/rasters/models present",
                    not missing, f"missing={len(missing)}" if missing else "all present")

    fails = [r for r in RESULTS if not r[1]]
    print(f"\n[SUMMARY] {len(RESULTS) - len(fails)}/{len(RESULTS)} checks pass, "
          f"{len(fails)} FAIL", flush=True)
    if fails:
        for name, _, detail in fails:
            print(f"  FAILED: {name} {detail}")
        return 1
    print("[VERIFY] ALL PHASE 9 CHECKS PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
