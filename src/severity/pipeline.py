"""End-to-end Phase 6 Stage 1 pipeline: full-grid RF-C prediction rasters.

Orchestrates:

1. Empirical verification of CSV ``row``/``col`` -> ``arr[row, col]``
   raster indexing against the observed LST rasters.
2. Production RF-C model training on all 300,000 Phase 4 sampled rows
   (full spatial + morphology predictor set), or reuse of a saved model.
3. Full-valid-grid feature generation for both years, reusing the frozen
   Phase 5 spatial/morphology machinery with identical spatial-block bins.
4. Full-grid class + probability prediction (chunked, deterministic).
5. Full-grid raster outputs: severity, per-class probabilities, confidence,
   ordinal severity score, and observed LST for both years.
6. Sanity checks, summary table, manifest, and pipeline record.

Run with::

    PYTHONPATH=src python3 -m severity.pipeline
    PYTHONPATH=src python3 -m severity.pipeline --skip-model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio

from models.config import CLASS_LABELS, INPUT_DATASET_CSV, PREDICTOR_VARS, RANDOM_SEED
from models.dataset import (
    encode_predictors,
    load_phase4_dataset,
    merge_morphology_features,
    merge_spatial_features,
)

from .config import (
    FULLGRID_SUMMARY_CSV,
    LST_RASTERS,
    MANIFEST_JSON,
    MODEL_PATH,
    PHASE5_MORPHOLOGY_FEATURES_CSV,
    PHASE5_PREDICTIONS_CSV,
    PHASE5_SPATIAL_FEATURES_CSV,
    PHASE5_UHI_RASTER_2022,
    PHASE6_DIR,
    PIPELINE_RECORD_JSON,
    PROJECT_ROOT,
    RF_C_PARAMS,
    VALID_MASK_RASTER,
)
from .fullgrid import (
    align_features_to_model,
    build_fullgrid_features_for_year,
    load_valid_cells,
)
from .model import load_production_model, predict_full_grid, train_production_model
from .rasters import write_lst_rasters, write_year_rasters


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _convert_for_json(obj: object) -> object:
    """Recursively convert numpy types and dict keys for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert_for_json(obj), f, indent=2, default=str)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_row_col_indexing(n_check: int = 1000) -> Dict:
    """Verify empirically that CSV ``row``/``col`` index rasters as arr[row, col]."""
    df = load_phase4_dataset()
    rng = np.random.default_rng(RANDOM_SEED)

    matched_total = 0
    max_diff = 0.0
    checked = 0
    for year, lst_path in LST_RASTERS.items():
        with rasterio.open(lst_path) as src:
            lst = src.read(1)
        sub = df[df["year"] == year]
        idx = rng.choice(len(sub), min(n_check, len(sub)), replace=False)
        s = sub.iloc[idx]
        values = lst[s["row"].values, s["col"].values]
        diff = np.abs(values - s["lst_C"].values)
        matched_total += int((diff < 0.01).sum())
        max_diff = max(max_diff, float(diff.max()))
        checked += len(s)

    result = {
        "method": "compare lst raster arr[row, col] to CSV lst_C",
        "n_checked": checked,
        "n_matched": matched_total,
        "max_abs_diff_C": max_diff,
        "conclusion": "direct arr[row, col] indexing confirmed" if matched_total == checked else "INDEXING MISMATCH",
    }
    print(f"  row/col indexing: {matched_total}/{checked} matched (max diff {max_diff:.2e})")
    return result


def _class_distribution(predicted_class: np.ndarray) -> Dict:
    counts = np.bincount(predicted_class.astype(int), minlength=4)
    total = counts.sum()
    return {
        label: {"class_value": i, "count": int(counts[i]), "pct": round(100.0 * counts[i] / total, 4)}
        for i, label in enumerate(CLASS_LABELS)
    }


def run_stage1(skip_model: bool = False) -> Dict:
    """Run the complete Phase 6 Stage 1 pipeline."""
    PHASE6_DIR.mkdir(parents=True, exist_ok=True)
    record: Dict = {
        "phase": 6,
        "stage": 1,
        "project": "GreenGrid-AI",
        "project_root": str(PROJECT_ROOT),
        "started_utc": _now(),
        "random_seed": RANDOM_SEED,
        "steps": {},
    }

    # ------------------------------------------------------------------
    # 1. Empirical row/col indexing verification
    # ------------------------------------------------------------------
    t0 = time.time()
    indexing_check = verify_row_col_indexing()
    record["steps"]["indexing_verification"] = {
        "status": "success" if indexing_check["n_matched"] == indexing_check["n_checked"] else "failed",
        "elapsed_s": round(time.time() - t0, 3),
        **indexing_check,
    }

    # ------------------------------------------------------------------
    # 2. Load sampled dataset with cached Phase 5 features
    # ------------------------------------------------------------------
    t0 = time.time()
    df = load_phase4_dataset()
    df = merge_spatial_features(df)
    df = merge_morphology_features(df)
    record["steps"]["load_sampled_dataset"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "input_path": str(INPUT_DATASET_CSV),
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }

    # ------------------------------------------------------------------
    # 3. Train (or load) the production RF-C model
    # ------------------------------------------------------------------
    t0 = time.time()
    if skip_model and MODEL_PATH.exists():
        model = load_production_model()
        feature_names = list(model.feature_names_in_)
        print(f"  Loaded saved model: {MODEL_PATH} ({len(feature_names)} features)")
        record["steps"]["train_model"] = {
            "status": "success",
            "elapsed_s": round(time.time() - t0, 3),
            "mode": "loaded",
            "model_path": str(MODEL_PATH),
            "n_features": len(feature_names),
        }
    else:
        train_out = train_production_model(df)
        model = train_out["model"]
        feature_names = train_out["feature_names"]
        record["steps"]["train_model"] = {
            "status": "success",
            "elapsed_s": round(time.time() - t0, 3),
            "mode": "trained",
            "model_path": train_out["model_path"],
            "model": "RandomForestClassifier",
            "params": RF_C_PARAMS,
            "n_features": len(feature_names),
            "n_training_rows": len(train_out["y"]),
            "target_thresholds_C": train_out["thresholds"],
        }
    print(f"  Model feature count: {len(feature_names)}")

    # ------------------------------------------------------------------
    # 4. Valid cells and per-year full-grid prediction
    # ------------------------------------------------------------------
    t0 = time.time()
    rows, cols, transform, mask_profile = load_valid_cells()
    n_valid = len(rows)
    print(f"  Valid cells per year: {n_valid:,} (grid {mask_profile['height']}x{mask_profile['width']})")
    record["steps"]["valid_grid"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "valid_cells_per_year": int(n_valid),
        "grid_height": int(mask_profile["height"]),
        "grid_width": int(mask_profile["width"]),
        "crs": str(mask_profile["crs"]),
        "valid_mask": str(VALID_MASK_RASTER),
    }

    per_year: Dict[int, Dict] = {}
    for year in (2022, 2026):
        t0 = time.time()
        print(f"  Building full-grid features for {year} ...")
        fg = build_fullgrid_features_for_year(year, df, rows, cols, transform)
        print(f"  Encoding + aligning features for {year} ...")
        X_fg, _ = encode_predictors(fg, predictor_cols=PREDICTOR_VARS)
        X_fg = align_features_to_model(X_fg, feature_names)
        X_fg = X_fg.astype(np.float32)

        print(f"  Predicting full grid for {year} ...")
        preds = predict_full_grid(model, X_fg)

        print(f"  Writing rasters for {year} ...")
        raster_results = write_year_rasters(
            year,
            preds["predicted_class"],
            preds["probabilities"],
            preds["confidence"],
            preds["severity_score"],
        )

        per_year[year] = {
            "features": fg,
            "preds": preds,
            "raster_results": raster_results,
            "feature_elapsed_s": round(time.time() - t0, 3),
        }
        # Free the encoded matrix; only the raw feature frame is kept for
        # downstream parity/agreement checks.
        del X_fg
        record["steps"][f"fullgrid_prediction_{year}"] = {
            "status": "success",
            "elapsed_s": round(time.time() - t0, 3),
            "n_valid_pixels": int(n_valid),
            "class_distribution": _class_distribution(preds["predicted_class"]),
            "rasters": {k: v["output_path"] for k, v in raster_results.items()},
        }
        print(f"  {year} class distribution: {record['steps'][f'fullgrid_prediction_{year}']['class_distribution']}")

    # ------------------------------------------------------------------
    # 5. Observed LST rasters
    # ------------------------------------------------------------------
    t0 = time.time()
    lst_results = write_lst_rasters()
    record["steps"]["lst_rasters"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "outputs": {str(k): v["output_path"] for k, v in lst_results.items()},
    }

    # ------------------------------------------------------------------
    # 6. Sanity checks
    # ------------------------------------------------------------------
    t0 = time.time()
    sanity = _run_sanity_checks(per_year, rows, cols)
    record["steps"]["sanity_checks"] = {
        "status": "success" if sanity["all_passed"] else "failed",
        "elapsed_s": round(time.time() - t0, 3),
        **sanity,
    }

    # ------------------------------------------------------------------
    # 7. Full-grid prediction summary table
    # ------------------------------------------------------------------
    t0 = time.time()
    summary_rows = []
    for year in (2022, 2026):
        preds = per_year[year]["preds"]
        probs = preds["probabilities"]
        dist = _class_distribution(preds["predicted_class"])
        row = {
            "year": year,
            "valid_pixel_count": int(n_valid),
        }
        for i, label in enumerate(CLASS_LABELS):
            row[f"class_{i}_{label.lower()}_count"] = dist[label]["count"]
            row[f"class_{i}_{label.lower()}_pct"] = dist[label]["pct"]
        row["mean_probability"] = round(float(probs.mean()), 6)
        row["max_probability"] = round(float(probs.max()), 6)
        row["mean_confidence"] = round(float(preds["confidence"].mean()), 6)
        row["mean_severity_score"] = round(float(preds["severity_score"].mean()), 6)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    FULLGRID_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(FULLGRID_SUMMARY_CSV, index=False)
    record["steps"]["summary_table"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "output_path": str(FULLGRID_SUMMARY_CSV),
    }

    # ------------------------------------------------------------------
    # 8. Manifest
    # ------------------------------------------------------------------
    t0 = time.time()
    manifest = _build_manifest(df, feature_names, per_year, mask_profile, n_valid)
    _save_json(manifest, MANIFEST_JSON)
    record["steps"]["manifest"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "output_path": str(MANIFEST_JSON),
    }

    # ------------------------------------------------------------------
    # 9. Pipeline record
    # ------------------------------------------------------------------
    record["finished_utc"] = _now()
    record["total_elapsed_s"] = round(
        sum(s.get("elapsed_s", 0) for s in record["steps"].values() if isinstance(s, dict)),
        3,
    )
    record["status"] = "success"
    _save_json(record, PIPELINE_RECORD_JSON)

    return {
        "record": record,
        "manifest": manifest,
        "summary": summary_df,
        "per_year": per_year,
    }


def _run_sanity_checks(per_year: Dict[int, Dict], rows: np.ndarray, cols: np.ndarray) -> Dict:
    """Run Stage 1 sanity checks and return a report dictionary."""
    checks: Dict[str, object] = {}

    # (a) Probability sum tolerance at valid cells.
    prob_checks = {}
    for year in (2022, 2026):
        probs = per_year[year]["preds"]["probabilities"]
        prob_sums = probs.sum(axis=1)
        prob_checks[year] = {
            "max_abs_deviation_from_1": float(np.abs(prob_sums - 1.0).max()),
            "min_probability": float(probs.min()),
            "max_probability": float(probs.max()),
            "within_1e-4": bool(np.all(np.abs(prob_sums - 1.0) < 1e-4)),
        }
    checks["probability_sums"] = prob_checks
    print(f"  Prob-sum deviations: " + ", ".join(f"{y}: {v['max_abs_deviation_from_1']:.2e}" for y, v in prob_checks.items()))

    # (b) Agreement with frozen Phase 5 sampled predictions at sampled cells.
    p5 = pd.read_csv(PHASE5_PREDICTIONS_CSV)
    agreement = {}
    for year in (2022, 2026):
        fg = per_year[year]["features"]
        fg_key = fg[["row", "col", "year"]].copy()
        fg_key["fg_class"] = per_year[year]["preds"]["predicted_class"].astype(int)
        merged = p5[p5["year"] == year].merge(fg_key, on=["row", "col", "year"], how="inner")
        if len(merged) == 0:
            raise AssertionError(f"No overlapping sampled cells for {year}")
        rate = float((merged["predicted_class"] == merged["fg_class"]).mean())
        agreement[year] = {
            "n_sampled_cells": int(len(merged)),
            "agreement_rate": round(rate, 4),
        }
        print(f"  Agreement with Phase 5 sampled predictions ({year}): {rate:.4f} over {len(merged):,} cells")
    checks["phase5_sampled_agreement"] = agreement

    # (c) Raster checks: value domains, nodata preservation, grid parity with Phase 5.
    from .config import severity_raster_path

    raster_checks = {}
    with rasterio.open(PHASE5_UHI_RASTER_2022) as ref:
        ref_profile = {"crs": str(ref.crs), "transform": ref.transform, "height": ref.height, "width": ref.width}

    with rasterio.open(severity_raster_path(2022)) as src:
        sev = src.read(1)
        raster_checks["severity_2022"] = {
            "valid_count": int((sev != -1).sum()),
            "unique_values": sorted(np.unique(sev).tolist()),
            "values_in_{0,1,2,3}": bool(set(np.unique(sev)) <= {-1, 0, 1, 2, 3}),
            "crs_matches_phase5": str(src.crs) == ref_profile["crs"],
            "transform_matches_phase5": src.transform == ref_profile["transform"],
            "dims_match_phase5": (src.height, src.width) == (ref_profile["height"], ref_profile["width"]),
        }
    nodata_rows, nodata_cols = mask_nodata_cells(rows, cols, ref_profile["height"], ref_profile["width"])
    raster_checks["nodata_initialised"] = bool((sev[nodata_rows, nodata_cols] == -1).all())
    checks["rasters"] = raster_checks
    print(f"  Raster checks: {raster_checks['severity_2022']}")

    # (d) Feature parity at sampled cells vs cached Phase 5 feature tables.
    parity = _check_feature_parity(per_year)
    checks["phase5_feature_parity_at_sampled_cells"] = parity
    print(f"  Feature parity at sampled cells: {parity}")

    all_passed = (
        all(v["within_1e-4"] for v in prob_checks.values())
        and raster_checks["severity_2022"]["values_in_{0,1,2,3}"]
        and raster_checks["severity_2022"]["crs_matches_phase5"]
        and raster_checks["severity_2022"]["transform_matches_phase5"]
        and raster_checks["severity_2022"]["dims_match_phase5"]
        and raster_checks["nodata_initialised"]
        and parity["all_match"]
    )
    checks["all_passed"] = bool(all_passed)
    return checks


def mask_nodata_cells(rows: np.ndarray, cols: np.ndarray, height: int, width: int) -> tuple:
    """Return index arrays of all grid cells NOT in the valid set."""
    valid = np.zeros((height, width), dtype=bool)
    valid[rows, cols] = True
    return np.where(~valid)


def _check_feature_parity(per_year: Dict[int, Dict], n_sample: int = 10000) -> Dict:
    """Verify full-grid features equal the cached Phase 5 tables at sampled cells.

    Full-grid features are stored as float32 for memory discipline, so the
    comparison is performed in the float32 domain (rtol/atol appropriate for
    float32 rounding on features whose magnitudes reach ~4e4 m).
    """
    p5_spatial = pd.read_csv(PHASE5_SPATIAL_FEATURES_CSV)
    p5_morph = pd.read_csv(PHASE5_MORPHOLOGY_FEATURES_CSV)

    results = {}
    for year in (2022, 2026):
        fg = per_year[year]["features"]
        sub = fg[["row", "col", "year"]].merge(
            p5_spatial[p5_spatial["year"] == year], on=["row", "col", "year"], how="inner"
        )
        sub = sub.merge(p5_morph[p5_morph["year"] == year], on=["row", "col", "year"], how="inner", suffixes=("", "_morph"))
        take = sub.sample(n=min(n_sample, len(sub)), random_state=RANDOM_SEED)
        fg_sub = fg.set_index(["row", "col", "year"]).loc[
            list(zip(take["row"], take["col"], take["year"]))
        ]

        def _max_diff(cols: List[str]) -> float:
            worst = 0.0
            for c in cols:
                ref = take[c].values.astype(np.float32)
                got = fg_sub[c].values.astype(np.float32)
                ok = np.isclose(got, ref, rtol=1e-5, atol=1e-6, equal_nan=True)
                if not ok.all():
                    return float("inf")
                worst = max(worst, float(np.nanmax(np.abs(got - ref))))
            return worst

        spatial_cols = [c for c in p5_spatial.columns if c not in ("row", "col", "year")]
        morph_cols = [c for c in p5_morph.columns if c not in ("row", "col", "year")]
        max_spatial_diff = _max_diff(spatial_cols)
        max_morph_diff = _max_diff(morph_cols)
        results[year] = {
            "n_compared": int(len(take)),
            "max_abs_diff_spatial": max_spatial_diff,
            "max_abs_diff_morphology": max_morph_diff,
            "tolerance": "float32 rtol=1e-5, atol=1e-6",
            "match": bool(np.isfinite(max_spatial_diff) and np.isfinite(max_morph_diff)),
        }
    results["all_match"] = all(v["match"] for v in results.values() if isinstance(v, dict))
    return results


def _build_manifest(
    df: pd.DataFrame,
    feature_names: List[str],
    per_year: Dict[int, Dict],
    mask_profile: Dict,
    n_valid: int,
) -> Dict:
    """Build the Phase 6 manifest."""
    import sklearn

    input_paths = [INPUT_DATASET_CSV, VALID_MASK_RASTER, PHASE5_PREDICTIONS_CSV] + list(LST_RASTERS.values())
    inputs = []
    for p in input_paths:
        p = Path(p)
        inputs.append({"path": str(p), "size_bytes": p.stat().st_size, "sha256": _sha256(p)})

    return {
        "phase": 6,
        "stage": 1,
        "project": "GreenGrid-AI",
        "generated_utc": _now(),
        "model": {
            "name": "Random Forest RF-C (production, Phase 6 adoption)",
            "library": "scikit-learn",
            "params": RF_C_PARAMS,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "training_rows": int(len(df)),
            "artifact": str(MODEL_PATH),
        },
        "inputs": inputs,
        "grid_spec": {
            "crs": str(mask_profile["crs"]),
            "height": int(mask_profile["height"]),
            "width": int(mask_profile["width"]),
            "valid_cells_per_year": int(n_valid),
            "years": [2022, 2026],
        },
        "class_definitions": {
            "labels": CLASS_LABELS,
            "values": [0, 1, 2, 3],
            "method": "per-year LST quartiles (inherited from Phase 5)",
            "note": (
                "Classes are defined relative to each year's own LST distribution; "
                "Severe-2022 and Severe-2026 do not necessarily represent the same "
                "absolute LST range. Cross-year comparison is class-level only."
            ),
        },
        "terminology_note": (
            "ML-based relative heat severity; operational UHI hotspot proxy — "
            "not physical UHI intensity. Severity classes are derived from "
            "satellite-observed LST quartiles and reflect relative within-year "
            "ranking, not absolute temperature anomalies."
        ),
        "outputs": {
            "rasters_dir": str(PHASE6_DIR / "rasters"),
            "summary_csv": str(FULLGRID_SUMMARY_CSV),
            "severity_score_definition": (
                "expected ordinal score 0*P(Low)+1*P(Moderate)+2*P(High)+3*P(Severe); "
                "an ordinal model score, NOT a temperature"
            ),
        },
        "software_versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
    }


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for the Phase 6 Stage 1 pipeline."""
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 6 Stage 1 full-grid severity prediction."
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Reuse the saved production model instead of retraining RF-C.",
    )
    args = parser.parse_args(argv)

    try:
        results = run_stage1(skip_model=args.skip_model)
        print(json.dumps(_convert_for_json(results["record"]), indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Phase 6 Stage 1 pipeline failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
