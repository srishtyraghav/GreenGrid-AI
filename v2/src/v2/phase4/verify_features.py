"""V2 Phase 4 verification gates for the assembled feature tables.

Gates (all must pass; exits non-zero otherwise):

  1. schema exact: parquet predictor columns == frozen 178 (names AND order),
     SHA-256 of the name list matches the frozen schema file,
  2. no NaN / no Inf in any of the 178 predictors,
  3. leakage: banned columns {lst_C, lon, lat, row, col, spatial_block_id}
     do NOT appear among the 178 predictors (row/col/spatial_block_id/lst_C
     exist only as leading metadata columns; year is a schema predictor),
  4. per-year coverage of the Phase-3 valid masks within the study area meets
     the Phase-2 gates (L9 ST >= 90%, S2-10m >= 85%, LULC >= 80%),
  5. spatial_block_id matches the PINNED V1 25-block geometry
     (src/v2/common.py PINNED_* bins; independent of the sample),
  5b. hard B2 invariant: every feature row's 30 m cell contains >= 1 valid
     native S2-10m pixel under the B2>0 guard (no S2-derived feature may
     contain a B2==0 pixel contribution),
  6. met columns: finite; not erroneously constant when the per-year point
     means differ; adjacent-pixel gradients bounded by a synoptic threshold
     (IDW of a 16-point field is smooth at 30 m),
  7. count summary per year (rows, blocks occupied, coverage) written to
     v2_feature_verify.json.

Usage:
    PYTHONPATH=src .venv/Scripts/python.exe -m src.v2.phase4.verify_features \
        --features-dir data/v2/phase4 [--phase3-root data/v2/phase3] \
        [--schema-json ...] [--met-gradient-tol 0.1]
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
    GATES,
    PROJECT_ROOT,
    SCHEMA_JSON_DEFAULT,
    block_bands_for_indices,
    dump_json,
    load_frozen_schema,
    schema_hash,
)

BANNED_IN_PREDICTORS = ("lst_C", "lon", "lat", "row", "col", "spatial_block_id")
METADATA_COLS = ["row", "col", "spatial_block_id", "lst_C"]
MET_FEATURES = ["met_t2m_c", "met_rh_pct", "met_wind_kmh",
                "met_precip_mm", "met_ssr_wm2", "met_swc_m3m3"]


class GateReport:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _check_schema(report: GateReport, schema: list[str], df: pd.DataFrame,
                  year: int) -> list[str]:
    cols = list(df.columns)
    predictors = cols[len(METADATA_COLS):]
    if cols[: len(METADATA_COLS)] != METADATA_COLS:
        report.fail(f"[{year}] metadata columns must be exactly {METADATA_COLS}, "
                    f"got {cols[: len(METADATA_COLS)]}")
    if predictors != schema:
        n_diff = sum(1 for a, b in zip(predictors, schema) if a != b)
        report.fail(
            f"[{year}] predictor schema mismatch vs frozen 178: "
            f"len {len(predictors)} vs {len(schema)}, {n_diff} positional diffs, "
            f"first_diff="
            f"{next(((a, b) for a, b in zip(predictors, schema) if a != b), None)}")
    leaked = [c for c in predictors if c in BANNED_IN_PREDICTORS]
    if leaked:
        report.fail(f"[{year}] LEAKAGE: banned columns among predictors: {leaked}")
    return predictors


def _check_finite(report: GateReport, df: pd.DataFrame, predictors: list[str],
                  year: int) -> None:
    arr = df[predictors].to_numpy(dtype=np.float64)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan or n_inf:
        report.fail(f"[{year}] predictors contain {n_nan} NaN and {n_inf} Inf values")


def _check_blocks(report: GateReport, df: pd.DataFrame, year: int) -> None:
    """Pinned V1 geometry: block id from PINNED_* bins only (never from data)."""
    rb, cb = block_bands_for_indices(df["row"].to_numpy(), df["col"].to_numpy())
    expected = rb * 5 + cb
    got = df["spatial_block_id"].to_numpy()
    if not np.array_equal(expected, got):
        n_bad = int((expected != got).sum())
        report.fail(f"[{year}] spatial_block_id mismatch on {n_bad} rows vs the "
                    f"pinned V1 bins")


def _check_met(report: GateReport, df: pd.DataFrame, year: int,
               gradient_tol: float) -> None:
    for col in MET_FEATURES:
        if col not in df.columns:
            report.fail(f"[{year}] missing met column {col}")
            return
        v = df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(v).all():
            report.fail(f"[{year}] met column {col} has non-finite values")
            continue
        std = float(v.std())
        if std == 0.0:
            report.warn(f"[{year}] met column {col} is spatially constant "
                        f"(legitimate only if all 16 point means are equal)")
        # adjacent-pixel gradient in row-major grid order within each row
        rows, cols = df["row"].to_numpy(), df["col"].to_numpy()
        order = np.lexsort((cols, rows))
        v_sorted = v[order]
        row_sorted, col_sorted = rows[order], cols[order]
        adjacent = np.flatnonzero(np.diff(col_sorted) == 1)
        adjacent = adjacent[row_sorted[adjacent] == row_sorted[adjacent + 1]]
        if len(adjacent):
            max_grad = float(np.abs(np.diff(v_sorted)[adjacent]).max())
        else:
            max_grad = 0.0
        report.warnings.append(
            f"[{year}] met {col}: std={std:.6g}, max adjacent-pixel |diff|={max_grad:.6g}")
        if max_grad > gradient_tol:
            report.fail(f"[{year}] met column {col} adjacent-pixel gradient "
                        f"{max_grad:.6g} > synoptic tolerance {gradient_tol} "
                        f"(IDW fields must vary only at ~10 km scale)")


def _check_b2_exclusion(report: GateReport, df: pd.DataFrame, year: int,
                        phase3_root: Path) -> None:
    """Hard invariant: every feature-table row sits on a 30 m cell that has
    at least one valid native S2-10m pixel UNDER THE B2>0 GUARD — i.e. no
    S2-derived feature value can contain a B2==0 native pixel contribution."""
    path = phase3_root / str(year) / "valid_s2_10m_30m.tif"
    if not path.exists():
        report.fail(f"[{year}] missing S2 validity mask {path}")
        return
    with rasterio.open(path) as ds:
        mask = ds.read(1)
    rows, cols = df["row"].to_numpy(), df["col"].to_numpy()
    ok = mask[rows, cols] == 1
    if not ok.all():
        bad = np.flatnonzero(~ok)[:5]
        examples = [(int(rows[i]), int(cols[i])) for i in bad]
        report.fail(
            f"[{year}] B2==0 LEAKAGE: {int((~ok).sum())} feature rows sit on cells "
            f"with no valid native B2>0 S2 pixel (examples row/col: {examples})")


def _check_coverage(report: GateReport, phase3_root: Path, year: int) -> dict:
    cov = {}
    ydir = phase3_root / str(year)
    gate_map = {"valid_l9": "l9_st", "valid_s2_10m": "s2_10m", "valid_lulc": "lulc"}
    for mask_name, gate_key in gate_map.items():
        path = ydir / f"{mask_name}_30m.tif"
        if not path.exists():
            report.fail(f"[{year}] missing coverage mask {path}")
            continue
        with rasterio.open(path) as ds:
            m = ds.read(1)
        inside = m != 255
        frac = float((m == 1)[inside].mean()) if inside.any() else float("nan")
        cov[mask_name] = round(frac, 6)
        gate = GATES[gate_key]
        if not (frac >= gate):
            report.fail(f"[{year}] coverage {mask_name} {frac:.4f} < gate {gate:.2f}")
    return cov


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-dir", default=str(PROJECT_ROOT / "data" / "phase4"))
    p.add_argument("--phase3-root", default=str(PROJECT_ROOT / "data" / "phase3"))
    p.add_argument("--schema-json", default=str(SCHEMA_JSON_DEFAULT))
    p.add_argument("--met-gradient-tol", type=float, default=0.1)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    report = GateReport()
    features_dir = Path(args.features_dir)
    schema = load_frozen_schema(Path(args.schema_json))
    frozen_sha = schema_hash(schema)

    parquets = sorted(features_dir.glob("features_*.parquet"))
    if not parquets:
        report.fail(f"no features_*.parquet under {features_dir}")

    summary: dict = {"parquets": [], "frozen_schema_sha256": frozen_sha,
                     "block_geometry": "pinned V1 bins (HOLDOUT_BLOCKS=[2,9,15,23])"}
    frames: dict[str, pd.DataFrame] = {}
    for pq in parquets:
        year = pq.stem.split("_")[-1]
        frames[year] = pd.read_parquet(pq)
    for pq in parquets:
        year = pq.stem.split("_")[-1]
        df = frames[year]
        predictors = _check_schema(report, schema, df, year)
        if not any(f.startswith(f"[{year}]") for f in report.failures):
            _check_finite(report, df, predictors, year)
        _check_blocks(report, df, year)
        _check_b2_exclusion(report, df, year, Path(args.phase3_root))
        _check_met(report, df, year, args.met_gradient_tol)
        cov = _check_coverage(report, Path(args.phase3_root), year) \
            if Path(args.phase3_root).exists() else {}
        summary["parquets"].append({
            "path": str(pq), "year": year, "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "blocks_occupied": sorted(int(b) for b in df["spatial_block_id"].unique()),
            "coverage": cov,
        })
        print(f"[VERIFY] {pq.name}: rows={len(df)} cols={len(df.columns)} "
              f"blocks={len(summary['parquets'][-1]['blocks_occupied'])} cov={cov}")

    summary["warnings"] = report.warnings
    summary["failures"] = report.failures
    summary["status"] = "FAIL" if report.failures else "PASS"
    out_path = Path(args.out) if args.out else features_dir / "v2_feature_verify.json"
    dump_json(summary, out_path)

    for w in report.warnings:
        print(f"[VERIFY][WARN] {w}")
    if report.failures:
        print(f"[VERIFY] FAIL — {len(report.failures)} gate failure(s):", file=sys.stderr)
        for f in report.failures:
            print(f"[VERIFY][FAIL] {f}", file=sys.stderr)
        return 1
    print(f"[VERIFY] PASS — all Phase-4 gates satisfied (report: {out_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
