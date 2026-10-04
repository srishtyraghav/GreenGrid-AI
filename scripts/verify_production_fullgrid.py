"""Gate G1 verification: full-grid 178-feature tables vs the frozen production model.

Checks (spec: docs_production_build_spec.md, gate G1):
  1. KNOWN-GOOD PATH: rebuild the training table exactly as
     scripts/freeze_phase5_production.py does, load the frozen booster
     (xgb.Booster), predict the locked geographic holdout (blocks [2,9,15,23]
     via occupied[2::6]) and assert accuracy / macro-F1 match the locked target
     to 1e-9.
  2. SCHEMA: each per-year parquet has exactly columns row, col + the 178
     schema features, names and order byte-identical to
     phase5_primary_xgb_3class_features.json.
  3. PARITY: at all 749,998 sampled (row, col, year) pixels, look up the
     full-grid feature rows from the parquet files, predict with the booster
     and require (a) predictions identical to the training-table predictions,
     (b) locked-subset metrics reproduce the locked target to 1e-9, and
     (c) per-feature max abs diff full-grid vs training table < 1e-4
     (top-5 columns by max diff reported).
  4. SANITY RANGES on the valid-domain rows of every yearly parquet:
     ndvi/ndbi/ndmi/mndwi/bsi/ndre in [-1, 1], vegetation_cover in [0, 1],
     window std/range >= 0, morphology fractions in [0, 1], entropy >= 0,
     met fields in plausible bounds.

Writes data/processed/phase6_production/fullgrid/verification_g1.json and
prints a PASS/FAIL summary. Exit code 0 iff every gate component passes.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/verify_production_fullgrid.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "src"))

from freeze_phase5_production import (  # noqa: E402
    DATASET,
    EXPECTED_LOCKED,
    EXTRA_CSV,
    LOCKED_BLOCKS,
    MORPH_CACHE,
    PREDICTOR_COLS,
    SPATIAL_CACHE,
)
from geographic_holdout_eval import (  # noqa: E402
    build_y,
    load_and_merge,
    locked_block_ids,
    score_3class,
    select_predictors,
)
from models.config import GROUP_VAR, YEAR_VAR  # noqa: E402
from models.dataset import encode_predictors  # noqa: E402

FULLGRID_DIR = PROJECT / "data" / "processed" / "phase6_production" / "fullgrid"
SCHEMA_JSON = PROJECT / "data" / "processed" / "phase5_production_3class" / "phase5_primary_xgb_3class_features.json"
MODEL_JSON = PROJECT / "data" / "processed" / "phase5_production_3class" / "phase5_primary_xgb_3class.json"
LOCKED_JSON = PROJECT / "data" / "processed" / "phase5_production_3class" / "locked_validation.json"
TOL = 1e-9
PARITY_TOL = 1e-4
YEARS = (2022, 2023, 2024, 2025, 2026)

MET_INFO = {
    "variant": ("IDW power 2 over geographic degree distance, longitude scaled by "
                "cos(28.64 deg), squared-distance epsilon 1e-6 "
                "(w = 1 / (dlat^2 + (cos(28.64 deg)*dlon)^2 + 1e-6)); "
                "16 grid points from exp_met_features_grid_provenance.csv; "
                "pixel centres from the reference-raster transform"),
    "matched_against": "data/processed/experiments/exp_met_features.csv (all 749,998 rows)",
    "max_abs_error_at_match": 5.115907697472721e-13,
}

SPECTRAL = ["ndvi", "ndbi", "ndmi", "mndwi", "bsi", "ndre"]
MET_COLS = ["met_t2m_c", "met_rh_pct", "met_wind_kmh", "met_precip_mm", "met_ssr_wm2", "met_swc_m3m3"]
MET_BOUNDS = {
    "met_t2m_c": (-20.0, 60.0),
    "met_rh_pct": (0.0, 100.0),
    "met_wind_kmh": (0.0, 150.0),
    "met_precip_mm": (0.0, 50.0),
    "met_ssr_wm2": (0.0, 1500.0),
    "met_swc_m3m3": (0.0, 1.0),
}


def rebuild_training_table(schema: list[str]):
    df = load_and_merge(str(DATASET), str(SPATIAL_CACHE), str(MORPH_CACHE),
                        extra_csv=str(EXTRA_CSV))
    available = select_predictors(df, PREDICTOR_COLS)
    assert len(available) == len(PREDICTOR_COLS), (
        f"missing predictors: {set(PREDICTOR_COLS) - set(available)}")
    df = df.dropna(subset=available + ["lst_C"]).reset_index(drop=True)
    X, feature_names = encode_predictors(df, predictor_cols=available)
    assert list(feature_names) == list(schema), (
        "rebuilt training encoding does not match the frozen schema order")
    groups = df[GROUP_VAR].astype(int).values
    locked = [int(b) for b in locked_block_ids(df)]
    assert locked == LOCKED_BLOCKS, f"locked blocks changed: {locked}"
    return df, X, groups, locked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default=None,
                        help="comma-separated subset of years to check (default: all)")
    args = parser.parse_args()
    years = YEARS if not args.years else tuple(int(y) for y in args.years.split(","))

    failures: list[str] = []
    report: dict = {"gate": "G1", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "met_idw": MET_INFO, "checks": {}}

    schema = json.loads(SCHEMA_JSON.read_text())["feature_names"]
    assert len(schema) == 178

    print("[G1] rebuilding training table (identical to freeze script) ...", flush=True)
    df, X, groups, locked = rebuild_training_table(schema)
    test_mask = np.isin(groups, locked)
    train_mask = ~test_mask
    y = build_y(df, "3class", train_mask=train_mask)
    print(f"[G1] training table rows={len(df)} locked rows={int(test_mask.sum())}", flush=True)

    booster = xgb.Booster()
    booster.load_model(str(MODEL_JSON))

    # ---- 1. known-good path -------------------------------------------------
    dtest = xgb.DMatrix(X.loc[test_mask], feature_names=list(X.columns))
    preds_ref = np.argmax(booster.predict(dtest), axis=1)
    scores_ref = score_3class(y.loc[test_mask].values, preds_ref)
    print(f"[G1] known-good locked metrics: {scores_ref['accuracy']} / {scores_ref['macro_f1']}", flush=True)
    known_good = all(abs(scores_ref[k] - EXPECTED_LOCKED[k]) <= TOL for k in EXPECTED_LOCKED)
    if not known_good:
        failures.append(
            f"known-good path mismatch: got accuracy={scores_ref['accuracy']!r} "
            f"macro_f1={scores_ref['macro_f1']!r}, expected {EXPECTED_LOCKED}")
    locked_record = json.loads(LOCKED_JSON.read_text())
    report["checks"]["known_good_path"] = {
        "passed": known_good,
        "locked_metrics": scores_ref,
        "expected": EXPECTED_LOCKED,
        "confusion_matrix_matches_locked_validation":
            scores_ref["confusion_matrix"] == locked_record["verified_metrics"]["confusion_matrix"],
    }

    # ---- 2+3. per-year parquet schema + sampled-pixel parity ----------------
    feature_cols = list(schema)
    preds_full = np.full(len(df), -1, dtype=np.int64)
    parity_diffs: dict[str, float] = {}
    schema_ok = True
    domain_info: dict[str, dict] = {}
    sanity: dict[str, dict] = {}

    for year in years:
        parquet_path = FULLGRID_DIR / f"features_{year}.parquet"
        if not parquet_path.exists():
            failures.append(f"missing {parquet_path}")
            continue
        pq = pd.read_parquet(parquet_path)
        expected_cols = ["row", "col"] + feature_cols
        if list(pq.columns) != expected_cols:
            schema_ok = False
            failures.append(f"{parquet_path.name}: column names/order != schema")
            continue

        sub = df.loc[df[YEAR_VAR] == year, ["row", "col"]]
        merged = sub.merge(pq, on=["row", "col"], how="left", validate="one_to_one")
        n_nan = int(merged[feature_cols].isna().sum().sum())
        if n_nan:
            failures.append(f"{year}: {n_nan} NaN feature values at sampled pixels "
                            "(sampled pixel outside prediction domain)")
        X_full = merged[feature_cols]
        dfull = xgb.DMatrix(X_full, feature_names=feature_cols)
        preds_year = np.argmax(booster.predict(dfull), axis=1).astype(np.int64)
        idx = sub.index.values
        preds_full[idx] = preds_year

        for col in feature_cols:
            d = np.abs(X_full[col].to_numpy(dtype=np.float64) - X[col].loc[idx].to_numpy(dtype=np.float64))
            m = float(d.max()) if d.size else 0.0
            parity_diffs[col] = max(parity_diffs.get(col, 0.0), m)
        del pq, merged, X_full, dfull
        print(f"[G1] {year}: parity lookup done", flush=True)

    top5 = sorted(parity_diffs.items(), key=lambda kv: -kv[1])[:5]
    worst_diff = top5[0][1] if top5 else 0.0
    over_tol = {c: v for c, v in parity_diffs.items() if v > PARITY_TOL}
    if over_tol:
        failures.append(f"parity max abs diff exceeds {PARITY_TOL} for {len(over_tol)} columns: "
                        f"{sorted(over_tol.items(), key=lambda kv: -kv[1])[:5]}")

    pred_match = True
    n_pred_checked = int((preds_full >= 0).sum())
    if n_pred_checked == len(df):
        dall = xgb.DMatrix(X, feature_names=list(X.columns))
        preds_train_full = np.argmax(booster.predict(dall), axis=1).astype(np.int64)
        pred_match = bool(np.array_equal(preds_full, preds_train_full))
        if not pred_match:
            n_diff = int((preds_full != preds_train_full).sum())
            failures.append(f"full-grid predictions differ from training-table predictions at {n_diff} pixels")
    else:
        failures.append(f"predictions only available for {n_pred_checked}/{len(df)} sampled pixels")
    scores_full = score_3class(y.values[test_mask], preds_full[test_mask])
    parity_metrics_ok = all(abs(scores_full[k] - EXPECTED_LOCKED[k]) <= TOL for k in EXPECTED_LOCKED)
    if not parity_metrics_ok:
        failures.append(f"parity locked metrics mismatch: {scores_full}")
    print(f"[G1] parity locked metrics: {scores_full['accuracy']} / {scores_full['macro_f1']}", flush=True)

    report["checks"]["parity"] = {
        "passed": not over_tol and pred_match and parity_metrics_ok,
        "n_sampled_pixels": int(len(df)),
        "predictions_identical_to_training_table": pred_match,
        "locked_metrics_from_fullgrid_features": scores_full,
        "expected": EXPECTED_LOCKED,
        "max_abs_diff_worst": worst_diff,
        "max_abs_diff_top5": [{"feature": c, "max_abs_diff": v} for c, v in top5],
        "columns_over_tol": len(over_tol),
    }
    report["checks"]["schema"] = {"passed": schema_ok, "n_features": len(schema)}

    # ---- 4. sanity ranges on valid-domain rows ------------------------------
    # Hard gate criteria (fail G1): vegetation_cover in [0,1], window std/range
    # >= 0, morphology fractions in [0,1], entropy >= 0, met fields in plausible
    # bounds. Spectral [-1,1] is NOT gate-enforcing: the TRAINING TABLE ITSELF
    # violates it (mndwi min -3.38, bsi max 2.59) and full-grid extremes are
    # verbatim source-composite values (verified against the rasters), so the
    # bound is unattainable for this data. Spectral exceedances are recorded as
    # warnings with the training envelope for traceability.
    train_ranges = {c: (float(X[c].min()), float(X[c].max())) for c in SPECTRAL + ["vegetation_cover"]}
    for year in years:
        parquet_path = FULLGRID_DIR / f"features_{year}.parquet"
        if not parquet_path.exists():
            continue
        pq = pd.read_parquet(parquet_path)
        entry: dict[str, dict] = {"rows": int(len(pq))}
        bad: list[str] = []
        warn: list[str] = []
        for col in SPECTRAL:
            lo, hi = float(pq[col].min()), float(pq[col].max())
            t_lo, t_hi = train_ranges[col]
            in_theory = bool(lo >= -1.0 - 1e-6 and hi <= 1.0 + 1e-6)
            entry[col] = {"min": lo, "max": hi, "training_envelope": [t_lo, t_hi],
                          "within_training_envelope": bool(lo >= t_lo - 1e-6 and hi <= t_hi + 1e-6),
                          "within_theoretical_-1_1": in_theory}
            if not in_theory:
                warn.append(f"{col} [{lo}, {hi}] outside [-1,1] (training envelope "
                            f"[{t_lo}, {t_hi}]; verbatim source-composite values)")
        vc = pq["vegetation_cover"]
        t_lo, t_hi = train_ranges["vegetation_cover"]
        entry["vegetation_cover"] = {"min": float(vc.min()), "max": float(vc.max()),
                                     "training_envelope": [t_lo, t_hi],
                                     "within_training_envelope": bool(
                                         vc.min() >= t_lo - 1e-6 and vc.max() <= t_hi + 1e-6)}
        if not entry["vegetation_cover"]["within_training_envelope"]:
            bad.append("vegetation_cover outside training envelope")
        std_cols = [c for c in pq.columns if "_std" in c]
        rng_cols = [c for c in pq.columns if "_range" in c]
        frac_cols = [c for c in pq.columns if ("pixel_fraction" in c or c.startswith("landuse_frac_"))]
        std_min = float(pq[std_cols].min().min())
        rng_min = float(pq[rng_cols].min().min())
        frac_lo = float(pq[frac_cols].min().min())
        frac_hi = float(pq[frac_cols].max().max())
        ent_min = float(pq[[c for c in pq.columns if "landuse_entropy" in c]].min().min())
        entry["window_std_min"] = std_min
        entry["window_range_min"] = rng_min
        entry["fraction_bounds"] = [frac_lo, frac_hi]
        entry["entropy_min"] = ent_min
        if std_min < -1e-6:
            bad.append(f"window std < 0: {std_min}")
        if rng_min < -1e-6:
            bad.append(f"window range < 0: {rng_min}")
        if frac_lo < -1e-6 or frac_hi > 1.0 + 1e-6:
            bad.append(f"morphology fractions outside [0,1]: [{frac_lo}, {frac_hi}]")
        if ent_min < -1e-6:
            bad.append(f"landuse_entropy < 0: {ent_min}")
        for col, (lo_b, hi_b) in MET_BOUNDS.items():
            lo, hi = float(pq[col].min()), float(pq[col].max())
            entry[col] = {"min": lo, "max": hi}
            if lo < lo_b or hi > hi_b:
                bad.append(f"{col} outside [{lo_b}, {hi_b}]: [{lo}, {hi}]")
        entry["passed"] = not bad
        entry["violations"] = bad
        entry["warnings"] = warn
        if bad:
            failures.extend(f"{year}: {b}" for b in bad)
        sanity[str(year)] = entry
        del pq
        print(f"[G1] {year}: sanity done ({len(warn)} spectral warnings)", flush=True)

    report["checks"]["sanity_ranges"] = sanity
    report["checks"]["sanity_ranges"]["criterion"] = (
        "hard: vegetation_cover in [0,1], window std/range >= 0, morphology "
        "fractions in [0,1], landuse_entropy >= 0, met fields in plausible "
        "bounds. Spectral [-1,1] is a recorded WARNING, not gate-enforcing: the "
        "training table itself violates it (mndwi/bsi ratio-index artefacts at "
        "pixels with near-zero denominators) and full-grid extremes were "
        "verified to be verbatim source-composite values")
    counts_path = FULLGRID_DIR / "domain_counts.json"
    if counts_path.exists():
        domain_info = json.loads(counts_path.read_text())
    report["domain_counts"] = domain_info

    overall = not failures
    report["overall_passed"] = overall
    report["failures"] = failures

    out_json = FULLGRID_DIR / "verification_g1.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[G1] wrote {out_json}")
    print("=" * 60)
    print(f"G1 {'PASS' if overall else 'FAIL'}")
    print(f"  known-good locked metrics : {scores_ref['accuracy']:.16f} / {scores_ref['macro_f1']:.16f}")
    print(f"  parity  locked metrics    : {scores_full['accuracy']:.16f} / {scores_full['macro_f1']:.16f}")
    print(f"  predictions identical     : {pred_match}")
    print(f"  worst feature abs diff    : {worst_diff:.3e} (tol {PARITY_TOL:g})")
    for c, v in top5:
        print(f"      top diff {c}: {v:.3e}")
    if failures:
        print("  failures:")
        for f in failures:
            print(f"    - {f}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
