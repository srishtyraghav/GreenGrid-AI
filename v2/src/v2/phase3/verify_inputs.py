"""V2 Phase 3 preflight: verify every expected W4 input before processing.

Checks, per year 2022–2026 (source of truth: ``data/v2/phase2/`` contract):

  * existence of the per-year L9 / S2-10m / S2-20m / LULC exports
  * band count, positional band-name contract and dtype per product
  * per-file transform vs the authoritative grid: EXACT for every L9 year;
    for S2/LULC the reprojection target derived from the authoritative grid
    (CRS equality + expected native pixel size within tolerance)
  * dimensions, NaN/Inf sanity, count-band (QA_CLEAR_COUNT / VALID_COUNT)
  * study-area rasterization consistency on the authoritative grid
  * per-year coverage gates computed at native resolution inside the study
    area: L9 ST >= 90%, S2-10m >= 85% (with the B2>0 zero-fill guard),
    LULC >= 80%
  * HARD B2>0 accounting gate per year: valid == finite - B2==0 exactly,
    with the exclusion counts reported structurally; optional cross-
    validation of the counts against a prior audit JSON
    (--crosscheck-audit, e.g. data/phase2/w4_input_audit.json)

Writes ``w4_input_audit.json`` next to the data root (or ``--out``) and exits
non-zero with a precise failure report on any mismatch.

Usage (from the project root):
    PYTHONPATH=src .venv/Scripts/python.exe -m src.v2.phase3.verify_inputs \
        --data-root data/raw/v2 [--grid-file ...] [--years 2022,2023]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from ..common import (
    DATA_RAW_V2,
    GATES,
    GRID_FILE_DEFAULT,
    L9_BANDS,
    L9_PATH,
    LULC_BANDS,
    LULC_PATH,
    MET_WINDOW_HOURS_UTC,
    S2_10M_BANDS,
    S2_10M_PATH,
    S2_20M_BANDS,
    S2_20M_PATH,
    STUDY_AREA_DEFAULT,
    WINDOW_TAG,
    YEARS,
    Grid,
    dump_json,
    sha256_file,
    transforms_equal,
    valid_l9_mask,
    valid_lulc_mask,
    valid_s2_10m_mask,
    valid_s2_20m_mask,
)

S2_10M_NATIVE_RES_REL = 1.0 / 3.0   # 10 m vs 30 m (in degrees)
S2_20M_NATIVE_RES_REL = 2.0 / 3.0   # 20 m vs 30 m
LULC_NATIVE_RES_REL = 1.0 / 3.0
RES_TOL_REL = 0.02                  # 2% tolerance on native pixel size


class FailureReport:
    """Accumulates precise preflight failures."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _check_file(report: FailureReport, path: Path, role: str, year: int) -> bool:
    if not path.exists():
        report.fail(f"[{year}] MISSING {role}: {path}")
        return False
    if path.stat().st_size == 0:
        report.fail(f"[{year}] EMPTY FILE {role}: {path}")
        return False
    return True


def _nan_inf_stats(ds: rasterio.DatasetReader, band: int) -> dict:
    arr = ds.read(band, masked=True).astype(np.float64).filled(np.nan)
    finite = np.isfinite(arr)
    return {
        "finite_fraction": round(float(finite.sum()) / arr.size, 6),
        "n_inf": int(np.isinf(arr).sum()),
        "nan_or_masked_fraction": round(float((~finite).sum()) / arr.size, 6),
    }


def _check_bands(report: FailureReport, ds, role: str, year: int,
                 expected_names, expected_min_count: int) -> None:
    if ds.count < expected_min_count:
        report.fail(
            f"[{year}] {role}: expected >= {expected_min_count} bands "
            f"({list(expected_names)}), got {ds.count}"
        )
    desc = [d for d in ds.descriptions if d]
    if desc and len(desc) >= len(expected_names) and list(desc[: len(expected_names)]) != list(expected_names):
        report.fail(
            f"[{year}] {role}: band descriptions {desc[: len(expected_names)]} "
            f"!= contract order {list(expected_names)}"
        )


def _check_native_res(report: FailureReport, ds, role: str, year: int,
                      grid: Grid, expected_rel: float) -> None:
    if not ds.crs or ds.crs != grid.crs:
        report.fail(f"[{year}] {role}: CRS {ds.crs} != authoritative {grid.crs}")
        return
    px_x = abs(ds.transform.a)
    gx = abs(grid.transform.a)
    rel = px_x / gx
    if abs(rel - expected_rel) > RES_TOL_REL:
        report.fail(
            f"[{year}] {role}: native pixel size {px_x:.12g} deg is "
            f"{rel:.4f}x the authoritative 30 m cell (expected ~{expected_rel})"
        )


def _study_mask_native(ds, study_geoms) -> np.ndarray:
    from rasterio import features

    return features.rasterize(
        study_geoms, out_shape=(ds.height, ds.width),
        transform=ds.transform, fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)


def _coverage(arr_valid: np.ndarray, study_native: np.ndarray) -> float:
    n_study = int(study_native.sum())
    if n_study == 0:
        return float("nan")
    return float((arr_valid & study_native).sum() / n_study)


def verify_year(report: FailureReport, data_root: Path, year: int,
                grid: Grid, study_geoms) -> dict:
    """Verify one year; returns the audit dict for that year."""
    audit: dict = {"year": year, "products": {}}

    # ---- Landsat 9 (must match the authoritative grid EXACTLY) -------------
    l9_path = L9_PATH(data_root, year)
    if not _check_file(report, l9_path, "Landsat 9", year):
        return audit
    with rasterio.open(l9_path) as ds:
        _check_bands(report, ds, "L9", year, L9_BANDS, len(L9_BANDS))
        n_bands_ok = ds.count >= len(L9_BANDS)
        if not transforms_equal(ds.transform, grid.transform):
            report.fail(
                f"[{year}] L9 transform mismatch vs authoritative grid: "
                f"{tuple(ds.transform)} != {tuple(grid.transform)} "
                f"(L9 years must share ONE identical transform)"
            )
        if (ds.height, ds.width) != grid.shape:
            report.fail(
                f"[{year}] L9 dimensions {(ds.height, ds.width)} != "
                f"authoritative {grid.shape}"
            )
        if n_bands_ok:
            nan = {name: _nan_inf_stats(ds, i + 1) for i, name in enumerate(L9_BANDS)}
            bad = [k for k, v in nan.items() if v["n_inf"] > 0]
            if bad:
                report.fail(f"[{year}] L9 Inf values in bands {bad}")
            st, sr2, cc = (ds.read(i, masked=True).astype(np.float64).filled(np.nan)
                           for i in (7, 1, 9))
            valid_l9 = valid_l9_mask(st, sr2, cc)
            study_native = _study_mask_native(ds, study_geoms)
            cov = _coverage(valid_l9, study_native)
            if cov < GATES["l9_st"]:
                report.fail(
                    f"[{year}] L9 ST coverage {cov:.4f} < gate {GATES['l9_st']:.2f}"
                )
        else:
            nan, cov = None, None
        audit["products"]["landsat9"] = {
            "path": str(l9_path), "sha256": sha256_file(l9_path),
            "transform_exact_match": transforms_equal(ds.transform, grid.transform),
            "coverage_st": round(cov, 6) if cov is not None else None,
            "gate": GATES["l9_st"],
            "nan_inf": nan,
        }

    # ---- Sentinel-2 10 m ----------------------------------------------------
    s210_path = S2_10M_PATH(data_root, year)
    if _check_file(report, s210_path, "S2 10m", year):
        with rasterio.open(s210_path) as ds:
            _check_bands(report, ds, "S2-10m", year, S2_10M_BANDS, len(S2_10M_BANDS))
            _check_native_res(report, ds, "S2-10m", year, grid, S2_10M_NATIVE_RES_REL)
            b2_guard = None
            if ds.count >= len(S2_10M_BANDS):
                b2 = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
                b3 = ds.read(2, masked=True).astype(np.float64).filled(np.nan)
                b4 = ds.read(3, masked=True).astype(np.float64).filled(np.nan)
                b8 = ds.read(4, masked=True).astype(np.float64).filled(np.nan)
                vc = ds.read(5, masked=True).astype(np.float64).filled(np.nan)
                valid_s2 = valid_s2_10m_mask(b2, b3, b4, b8, vc)
                finite_stack = (np.isfinite(b2) & np.isfinite(b3)
                                & np.isfinite(b4) & np.isfinite(b8) & (vc >= 1))
                b2_zero = finite_stack & (b2 == 0)
                n_finite = int(finite_stack.sum())
                n_zero = int(b2_zero.sum())
                n_valid = int(valid_s2.sum())
                b2_guard = {
                    "finite_px": n_finite,
                    "b2_zero_px": n_zero,
                    "valid_excluding_b2_zero": n_valid,
                    "identity_holds": bool(n_valid == n_finite - n_zero),
                }
                # HARD GATE (user-approved): the B2>0 exclusion accounting
                # must be exact -- a pixel with B2==0 must not be counted
                # valid under any circumstance.
                if not b2_guard["identity_holds"]:
                    report.fail(
                        f"[{year}] S2-10m B2==0 accounting violated: valid "
                        f"({n_valid}) != finite ({n_finite}) - B2==0 ({n_zero})"
                    )
                if n_zero > 0:
                    report.warn(
                        f"[{year}] S2-10m: B2==0 zero-fill pattern present "
                        f"({n_zero} px excluded from the value stack)"
                    )
                study_native = _study_mask_native(ds, study_geoms)
                cov = _coverage(valid_s2, study_native)
                if cov < GATES["s2_10m"]:
                    report.fail(
                        f"[{year}] S2-10m coverage {cov:.4f} < gate {GATES['s2_10m']:.2f} "
                        f"(finite B2..B8 & B2>0 & VALID_COUNT>=1)"
                    )
            else:
                cov = None
            audit["products"]["sentinel2_10m"] = {
                "path": str(s210_path), "sha256": sha256_file(s210_path),
                "coverage": round(cov, 6) if cov is not None else None,
                "gate": GATES["s2_10m"],
                "b2_zero_guard": b2_guard,
            }

    # ---- Sentinel-2 20 m ----------------------------------------------------
    s220_path = S2_20M_PATH(data_root, year)
    if _check_file(report, s220_path, "S2 20m", year):
        with rasterio.open(s220_path) as ds:
            _check_bands(report, ds, "S2-20m", year, S2_20M_BANDS, len(S2_20M_BANDS))
            _check_native_res(report, ds, "S2-20m", year, grid, S2_20M_NATIVE_RES_REL)
            if ds.count >= len(S2_20M_BANDS):
                bands20 = {i: ds.read(i, masked=True).astype(np.float64).filled(np.nan)
                           for i in range(1, 7)}
                finite_all = valid_s2_20m_mask(bands20[1], bands20[2], bands20[3],
                                               bands20[4], bands20[5], bands20[6])
                study_native = _study_mask_native(ds, study_geoms)
                cov = _coverage(finite_all, study_native)
            else:
                cov = None
            audit["products"]["sentinel2_20m"] = {
                "path": str(s220_path), "sha256": sha256_file(s220_path),
                "coverage_finite_b5_b12": round(cov, 6) if cov is not None else None,
            }

    # ---- LULC ---------------------------------------------------------------
    lulc_path = LULC_PATH(data_root, year)
    if _check_file(report, lulc_path, "LULC", year):
        with rasterio.open(lulc_path) as ds:
            _check_bands(report, ds, "LULC", year, LULC_BANDS, 1)
            _check_native_res(report, ds, "LULC", year, grid, LULC_NATIVE_RES_REL)
            lab = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
            finite = valid_lulc_mask(lab)
            bad = finite & ((lab < 0) | (lab > 8) | (np.round(lab) != lab))
            if bad.any():
                report.fail(
                    f"[{year}] LULC: {int(bad.sum())} pixels with labels "
                    f"outside Dynamic World classes 0..8"
                )
            study_native = _study_mask_native(ds, study_geoms)
            cov = _coverage(finite, study_native)
            if cov < GATES["lulc"]:
                report.fail(
                    f"[{year}] LULC coverage {cov:.4f} < gate {GATES['lulc']:.2f} "
                    f"(V1 2025 defect was 1.1%)"
                )
            audit["products"]["lulc"] = {
                "path": str(lulc_path), "sha256": sha256_file(lulc_path),
                "coverage": round(cov, 6), "gate": GATES["lulc"],
            }

    return audit


def run(data_root: Path, grid_file: Path, years, study_area: Path,
        out_path: Path, crosscheck_audit: Path | None = None) -> int:
    import geopandas as gpd

    report = FailureReport()
    audit: dict = {
        "data_root": str(data_root),
        "grid_file": str(grid_file),
        "years": list(years),
        "window": f"W4 May 1 - Jun 30 (tag {WINDOW_TAG}); met hours UTC {MET_WINDOW_HOURS_UTC}",
        "gates": dict(GATES),
        "checks": [],
    }

    if not data_root.exists():
        report.fail(f"data root does not exist: {data_root}")
    if not grid_file.exists():
        report.fail(f"authoritative grid file does not exist: {grid_file}")

    grid = None
    if grid_file.exists():
        try:
            grid = Grid.from_file(grid_file)
        except Exception as exc:
            report.fail(f"cannot open authoritative grid {grid_file}: {exc}")

    study_geoms = None
    if study_area.exists():
        gdf = gpd.read_file(study_area)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        study_geoms = [(geom, 1) for geom in gdf.to_crs(grid.crs if grid else "EPSG:4326").geometry]
        audit["study_area"] = {"path": str(study_area), "n_features": len(gdf)}
    else:
        report.fail(f"study area not found: {study_area}")

    if grid is not None:
        audit["grid"] = {
            "height": grid.height, "width": grid.width,
            "transform": [grid.transform.a, grid.transform.b, grid.transform.c,
                          grid.transform.d, grid.transform.e, grid.transform.f],
            "crs": str(grid.crs),
        }
        # The 2026 L9 reference file doubles as the grid definition; when the
        # grid file IS the 2026 L9 export it must satisfy the L9 band contract.
        if grid_file == L9_PATH(data_root, 2026) and grid_file.exists():
            with rasterio.open(grid_file) as ds:
                _check_bands(report, ds, "L9-grid(2026)", 2026, L9_BANDS, len(L9_BANDS))

    if grid is not None and study_geoms is not None and data_root.exists():
        for year in years:
            audit["checks"].append(
                verify_year(report, data_root, int(year), grid, study_geoms)
            )

    if crosscheck_audit is not None:
        _crosscheck_b2_counts(report, audit, Path(crosscheck_audit))

    audit["warnings"] = report.warnings
    audit["failures"] = report.failures
    audit["status"] = "FAIL" if report.failures else "PASS"
    dump_json(audit, out_path)

    print(f"[PREFLIGHT] audit written to {out_path}")
    for w in report.warnings:
        print(f"[PREFLIGHT][WARN] {w}")
    if report.failures:
        print(f"[PREFLIGHT] FAIL — {len(report.failures)} problem(s):", file=sys.stderr)
        for f in report.failures:
            print(f"[PREFLIGHT][FAIL] {f}", file=sys.stderr)
        return 1
    print("[PREFLIGHT] PASS — all inputs satisfy the V2 W4 contract")
    return 0


_B2_WARNING_RE = None


def _crosscheck_b2_counts(report: FailureReport, audit: dict, prior_path: Path) -> None:
    """Cross-validate this run's B2==0 exclusion counts against a prior audit.

    Accepts the current structured format (products.*.b2_zero_guard) and the
    legacy warning-string format ("B2==0 zero-fill pattern present (N px)").
    Counts must match exactly -- the audit is over the same rasters, so any
    drift means the inputs changed underneath the pipeline.
    """
    import re

    if not prior_path.exists():
        report.fail(f"crosscheck audit not found: {prior_path}")
        return
    prior = json.loads(Path(prior_path).read_text(encoding="utf-8"))
    prior_by_year = {c["year"]: c for c in prior.get("checks", [])}
    for chk in audit.get("checks", []):
        year = chk["year"]
        guard = chk.get("products", {}).get("sentinel2_10m", {}).get("b2_zero_guard")
        if guard is None:
            continue
        p_chk = prior_by_year.get(year)
        prior_count = None
        if p_chk is not None:
            p_guard = (p_chk.get("products", {}).get("sentinel2_10m", {})
                       .get("b2_zero_guard"))
            if p_guard is not None:
                prior_count = p_guard.get("b2_zero_px")
            if prior_count is None:
                for w in prior.get("warnings", []):
                    m = re.search(rf"^\[{year}\] S2-10m: B2==0.*\((\d+) px\)", w)
                    if m:
                        prior_count = int(m.group(1))
                        break
        if prior_count is None:
            report.fail(f"[{year}] crosscheck: no prior B2==0 count found in {prior_path}")
        elif int(guard["b2_zero_px"]) != int(prior_count):
            report.fail(
                f"[{year}] crosscheck: recomputed B2==0 px {guard['b2_zero_px']} "
                f"!= prior audit {prior_count} ({prior_path})"
            )
        else:
            audit.setdefault("crosschecks", []).append({
                "year": year, "kind": "b2_zero_px", "recomputed": int(guard["b2_zero_px"]),
                "prior": int(prior_count), "prior_path": str(prior_path), "match": True,
            })


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(DATA_RAW_V2))
    p.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    p.add_argument("--years", default=",".join(str(y) for y in YEARS))
    p.add_argument("--study-area", default=str(STUDY_AREA_DEFAULT))
    p.add_argument("--out", default=None,
                   help="audit JSON path (default: data/phase2/w4_input_audit.json)")
    p.add_argument("--crosscheck-audit", default=None,
                   help="optional prior w4_input_audit.json; recomputed B2==0 "
                        "counts must match it exactly")
    args = p.parse_args(argv)

    data_root = Path(args.data_root)
    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[2] / "data" / "phase2" / "w4_input_audit.json"
    )
    years = tuple(int(y) for y in args.years.split(","))
    return run(data_root, Path(args.grid_file), years, Path(args.study_area), out_path,
                 crosscheck_audit=Path(args.crosscheck_audit) if args.crosscheck_audit else None)


if __name__ == "__main__":
    sys.exit(main())
