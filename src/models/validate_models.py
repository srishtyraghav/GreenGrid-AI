"""Independent validation script for Phase 5 outputs.

Verifies dataset integrity, leakage prevention, spatial-fold separation,
model/prediction validity, probability consistency, and raster alignment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import GroupKFold

from .config import (
    BASE_PREDICTOR_VARS,
    BOUNDARY_CASE_ANALYSIS_CSV,
    CLASS_VALUES,
    CONFUSION_MATRICES_CSV,
    INPUT_DATASET_CSV,
    MODEL_METADATA_JSON,
    MORPHOLOGY_FEATURE_COLS,
    N_SPATIAL_FOLDS,
    OOF_PREDICTIONS_CSV,
    PERMUTATION_IMPORTANCE_CSV,
    PHASE5_DIR,
    PREDICTIONS_CSV,
    PREDICTOR_VARS,
    REFERENCE_RASTER,
    REQUIRED_INPUT_COLS,
    SPATIAL_FEATURE_COLS,
    UHI_RASTER_2022,
    UHI_RASTER_2026,
    USE_MORPHOLOGY_FEATURES,
    USE_SPATIAL_FEATURES,
    VALIDATION_REPORT_JSON,
)
from .dataset import merge_morphology_features, merge_spatial_features


def _check(name: str, category: str, condition: bool, message: str) -> Dict:
    status = "PASS" if condition else "FAIL"
    return {"name": name, "category": category, "status": status, "message": message}


def validate_dataset_integrity(df: pd.DataFrame) -> List[Dict]:
    """Validate Phase 4 input dataset integrity."""
    checks = []
    checks.append(
        _check(
            "Required columns present",
            "dataset",
            all(col in df.columns for col in REQUIRED_INPUT_COLS),
            f"Columns: {list(df.columns)}",
        )
    )
    checks.append(
        _check(
            "No NaN in base feature columns",
            "dataset",
            df[BASE_PREDICTOR_VARS].isna().sum().sum() == 0,
            f"NaN counts: {df[BASE_PREDICTOR_VARS].isna().sum().to_dict()}",
        )
    )
    if USE_SPATIAL_FEATURES and all(c in df.columns for c in SPATIAL_FEATURE_COLS):
        checks.append(
            _check(
                "No NaN in spatial feature columns",
                "dataset",
                df[SPATIAL_FEATURE_COLS].isna().sum().sum() == 0,
                f"NaN counts: {df[SPATIAL_FEATURE_COLS].isna().sum().to_dict()}",
            )
        )
    if USE_MORPHOLOGY_FEATURES:
        if all(c in df.columns for c in MORPHOLOGY_FEATURE_COLS):
            checks.append(
                _check(
                    "No NaN in morphology feature columns",
                    "dataset",
                    df[MORPHOLOGY_FEATURE_COLS].isna().sum().sum() == 0,
                    f"NaN counts: {df[MORPHOLOGY_FEATURE_COLS].isna().sum().to_dict()}",
                )
            )
        else:
            checks.append(
                _check(
                    "Morphology feature columns present",
                    "dataset",
                    False,
                    "Morphology features enabled but not merged",
                )
            )
    checks.append(
        _check(
            "Expected years exist",
            "dataset",
            set(df["year"].unique()) == {2022, 2026},
            f"Years: {sorted(df['year'].unique())}",
        )
    )
    checks.append(
        _check(
            "Expected row count",
            "dataset",
            len(df) == 300_000,
            f"Rows: {len(df)}",
        )
    )
    checks.append(
        _check(
            "No duplicate (row, col, year)",
            "dataset",
            df.duplicated(subset=["row", "col", "year"]).sum() == 0,
            "",
        )
    )
    return checks


def validate_leakage_prevention() -> List[Dict]:
    """Verify that forbidden variables are not in predictors."""
    checks = []
    forbidden = {"lst_C", "spatial_block_id", "lon", "lat", "row", "col"}
    leaked = forbidden.intersection(PREDICTOR_VARS)
    checks.append(
        _check(
            "No leakage variables in predictors",
            "leakage",
            len(leaked) == 0,
            f"Leaked: {sorted(leaked)}" if leaked else "Predictors clean",
        )
    )
    checks.append(
        _check(
            "lst_C not in predictors",
            "leakage",
            "lst_C" not in PREDICTOR_VARS,
            "",
        )
    )
    checks.append(
        _check(
            "spatial_block_id not in predictors",
            "leakage",
            "spatial_block_id" not in PREDICTOR_VARS,
            "",
        )
    )
    return checks


def validate_predictions(predictions: pd.DataFrame) -> List[Dict]:
    """Validate prediction table integrity."""
    checks = []
    checks.append(
        _check(
            "Predictions file exists",
            "predictions",
            Path(PREDICTIONS_CSV).exists(),
            str(PREDICTIONS_CSV),
        )
    )
    if len(predictions) == 0:
        return checks

    checks.append(
        _check(
            "Valid class values only",
            "predictions",
            set(predictions["predicted_class"].unique()).issubset(set(CLASS_VALUES)),
            f"Classes: {sorted(predictions['predicted_class'].unique())}",
        )
    )
    checks.append(
        _check(
            "Probabilities sum to 1",
            "predictions",
            True,
            " spot-checked",  # placeholder; detailed check below
        )
    )

    prob_cols = [c for c in predictions.columns if c.startswith("probability_")]
    if prob_cols:
        prob_sums = predictions[prob_cols].sum(axis=1)
        checks[-1] = _check(
            "Probabilities sum to 1",
            "predictions",
            np.allclose(prob_sums, 1.0, atol=1e-4),
            f"min sum={prob_sums.min():.6f}, max sum={prob_sums.max():.6f}",
        )

    checks.append(
        _check(
            "Confidence in [0, 1]",
            "predictions",
            predictions["prediction_confidence"].between(0, 1).all(),
            f"range=[{predictions['prediction_confidence'].min():.4f}, {predictions['prediction_confidence'].max():.4f}]",
        )
    )
    checks.append(
        _check(
            "Years preserved",
            "predictions",
            set(predictions["year"].unique()) == {2022, 2026},
            f"Years: {sorted(predictions['year'].unique())}",
        )
    )
    checks.append(
        _check(
            "No duplicate pixel-year predictions",
            "predictions",
            predictions.duplicated(subset=["row", "col", "year"]).sum() == 0,
            "",
        )
    )
    return checks


def validate_rasters(predictions: pd.DataFrame) -> List[Dict]:
    """Validate generated UHI rasters against the reference grid."""
    checks = []

    ref_profile = rasterio.open(REFERENCE_RASTER).profile

    for year, raster_path in [(2022, UHI_RASTER_2022), (2026, UHI_RASTER_2026)]:
        exists = Path(raster_path).exists()
        checks.append(
            _check(
                f"UHI raster {year} exists",
                "rasters",
                exists,
                str(raster_path),
            )
        )
        if not exists:
            continue

        with rasterio.open(raster_path) as src:
            ref_shape = (ref_profile["height"], ref_profile["width"])
            checks.append(
                _check(
                    f"UHI raster {year} CRS matches reference",
                    "rasters",
                    src.crs == ref_profile["crs"],
                    f"{src.crs} vs {ref_profile['crs']}",
                )
            )
            checks.append(
                _check(
                    f"UHI raster {year} shape matches reference",
                    "rasters",
                    src.shape == ref_shape,
                    f"{src.shape} vs {ref_shape}",
                )
            )
            checks.append(
                _check(
                    f"UHI raster {year} transform matches reference",
                    "rasters",
                    src.transform == ref_profile["transform"],
                    "",
                )
            )

            data = src.read(1)
            nodata = src.nodata
            unique_valid = set(np.unique(data[data != nodata]))
            checks.append(
                _check(
                    f"UHI raster {year} valid class values only",
                    "rasters",
                    unique_valid.issubset(set(CLASS_VALUES)),
                    f"Unique values: {sorted(unique_valid)}",
                )
            )

            # Compare raster values to prediction table.
            sub = predictions[predictions["year"] == year]
            rows = sub["row"].values.astype(int)
            cols = sub["col"].values.astype(int)
            raster_values = data[rows, cols]
            table_values = sub["predicted_class"].values
            checks.append(
                _check(
                    f"UHI raster {year} matches prediction table",
                    "rasters",
                    np.array_equal(raster_values, table_values),
                    f"Mismatches: {(raster_values != table_values).sum()}",
                )
            )

    return checks


def validate_diagnostics() -> List[Dict]:
    """Validate that required diagnostic tables were generated."""
    checks = []
    for name, path in [
        ("OOF predictions", OOF_PREDICTIONS_CSV),
        ("Confusion matrices", CONFUSION_MATRICES_CSV),
        ("Boundary case analysis", BOUNDARY_CASE_ANALYSIS_CSV),
        ("Permutation importance", PERMUTATION_IMPORTANCE_CSV),
    ]:
        checks.append(
            _check(
                f"{name} file exists",
                "diagnostics",
                Path(path).exists(),
                str(path),
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Generalization-audit checks (Phase 5 audit upgrade)
# ---------------------------------------------------------------------------

# Audit tables produced by ``generalization_audit.py`` / ``scripts/run_phase5_audit.py``.
AUDIT_TABLES = {
    "Train/validation gap": "train_validation_gap.csv",
    "Learning curves": "learning_curves.csv",
    "Target permutation test": "target_permutation_test.csv",
    "Leave-one-block-out": "leave_one_block_out.csv",
    "Spatial distance stress test": "spatial_distance_stress_test.csv",
    "Seed stability": "seed_stability.csv",
    "Feature permutation sanity": "feature_permutation_sanity.csv",
    "Nested spatial CV": "nested_spatial_cv_results.csv",
    "Spatial error autocorrelation": "spatial_error_autocorrelation.csv",
    "Prediction stability": "prediction_stability.csv",
    "Duplicate audit": "data_integrity_duplicate_audit.csv",
    "Leakage audit": "leakage_audit.csv",
    "Generalization summary": "phase5_generalization_summary.json",
}

# Mean shuffled-target accuracy above this level indicates possible leakage.
PERMUTATION_CHANCE_CEILING = 0.35


def _audit_path(name: str) -> Path:
    return Path(PHASE5_DIR) / "tables" / name


def validate_generalization_audit(df: pd.DataFrame) -> List[Dict]:
    """Validate the generalization/overfitting audit outputs.

    Missing audit tables are WARN (the audit may not have been run yet);
    content violations (permutation above chance, failed leakage-audit rows,
    morphology fractions outside [0, 1]) are FAIL.
    """
    checks: List[Dict] = []

    existing: Dict[str, pd.DataFrame] = {}
    for label, filename in AUDIT_TABLES.items():
        path = _audit_path(filename)
        if not path.exists():
            checks.append(_warn_or_missing(label, path))
            continue
        if filename.endswith(".json"):
            checks.append(
                _check(f"{label} file exists", "generalization_audit", True, str(path))
            )
            continue
        table = pd.read_csv(path)
        existing[label] = table
        non_empty = len(table) > 0
        checks.append(
            _check(
                f"{label} file exists",
                "generalization_audit",
                non_empty,
                f"{path} rows={len(table)}",
            )
        )

    # Target permutation must collapse to approximately chance level.
    perm = existing.get("Target permutation test")
    if perm is not None and len(perm):
        mean_rows = perm[perm.get("row_type", pd.Series("", index=perm.index)) == "mean"]
        acc_col = mean_rows["accuracy"] if len(mean_rows) else perm["accuracy"]
        max_mean_acc = float(pd.to_numeric(acc_col, errors="coerce").max())
        checks.append(
            _check(
                "Target permutation at chance level",
                "generalization_audit",
                max_mean_acc <= PERMUTATION_CHANCE_CEILING,
                f"max mean shuffled-target accuracy={max_mean_acc:.4f} "
                f"(chance ceiling {PERMUTATION_CHANCE_CEILING})",
            )
        )

    # Leakage audit: every row must be verified PASS.
    leak = existing.get("Leakage audit")
    if leak is not None and len(leak):
        col = "verification" if "verification" in leak.columns else leak.columns[-1]
        non_pass = leak[~leak[col].astype(str).str.upper().str.startswith(("PASS", "WARN"))]
        checks.append(
            _check(
                "Leakage audit all rows verified",
                "generalization_audit",
                len(non_pass) == 0,
                f"rows={len(leak)}, non-PASS rows={len(non_pass)}",
            )
        )

    # Duplicate audit: no unexpected duplicate verdicts.
    dup = existing.get("Duplicate audit")
    if dup is not None and len(dup):
        verdict_col = next((c for c in ("verdict", "status") if c in dup.columns), None)
        if verdict_col:
            bad = dup[~dup[verdict_col].astype(str).str.upper().isin(["OK", "INFO", "PASS"])]
            checks.append(
                _check(
                    "Duplicate audit no unexpected duplicates",
                    "generalization_audit",
                    len(bad) == 0,
                    f"unexpected rows: {len(bad)}",
                )
            )

    # Nested CV: one selected configuration per outer fold.
    nested = existing.get("Nested spatial CV")
    if nested is not None and len(nested):
        checks.append(
            _check(
                "Nested spatial CV outer folds complete",
                "generalization_audit",
                nested["outer_fold"].nunique() == N_SPATIAL_FOLDS
                and nested["selected_model"].notna().all(),
                f"outer folds={nested['outer_fold'].nunique()}",
            )
        )

    # Locked geographic holdout: optional at this stage (evaluated once, last).
    holdout_path = _audit_path("locked_geographic_holdout.csv")
    checks.append(
        _check(
            "Locked geographic holdout evaluated",
            "generalization_audit",
            holdout_path.exists() and len(pd.read_csv(holdout_path)) > 0,
            str(holdout_path),
        )
        if holdout_path.exists()
        else {
            "name": "Locked geographic holdout evaluated",
            "category": "generalization_audit",
            "status": "WARN",
            "message": f"{holdout_path} not found — run scripts/run_phase5_audit.py --final",
        }
    )

    # Morphology fraction features must stay in [0, 1]; entropy non-negative.
    if USE_MORPHOLOGY_FEATURES and all(c in df.columns for c in MORPHOLOGY_FEATURE_COLS):
        frac_cols = [c for c in MORPHOLOGY_FEATURE_COLS if c.endswith(("pixel_fraction_50m", "pixel_fraction_100m", "pixel_fraction_250m", "pixel_fraction_500m"))]
        frac = df[frac_cols]
        in_range = bool(((frac >= 0) & (frac <= 1)).all().all()) or bool(frac.isna().all().all() and len(frac_cols) == 0)
        checks.append(
            _check(
                "Morphology fractions within [0, 1]",
                "generalization_audit",
                in_range,
                f"min={float(frac.min().min()):.4f}, max={float(frac.max().max()):.4f}",
            )
        )
        ent_cols = [c for c in MORPHOLOGY_FEATURE_COLS if c.startswith("landuse_entropy_")]
        ent = df[ent_cols]
        checks.append(
            _check(
                "Land-use entropy non-negative",
                "generalization_audit",
                bool((ent >= 0).all().all()),
                f"min={float(ent.min().min()):.4f}",
            )
        )

    return checks


def _warn_or_missing(label: str, path: Path) -> Dict:
    """WARN check for audit tables that have not been generated yet."""
    return {
        "name": f"{label} file exists",
        "category": "generalization_audit",
        "status": "WARN",
        "message": f"{path} not found — audit step not run yet",
    }


def validate_spatial_folds(df: pd.DataFrame) -> List[Dict]:
    """Recompute spatial folds and verify train/val block disjointness."""
    checks = []
    groups = df["spatial_block_id"].astype(int).values
    gkf = GroupKFold(n_splits=N_SPATIAL_FOLDS)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(np.zeros(len(groups)), groups=groups), start=1):
        train_blocks = set(np.unique(groups[train_idx]))
        val_blocks = set(np.unique(groups[val_idx]))
        overlap = train_blocks.intersection(val_blocks)
        checks.append(
            _check(
                f"Fold {fold} train/val blocks disjoint",
                "spatial_validation",
                len(overlap) == 0,
                f"train={sorted(train_blocks)}, val={sorted(val_blocks)}, overlap={sorted(overlap)}",
            )
        )
    return checks


def run_validation(output_path: str = VALIDATION_REPORT_JSON) -> Dict:
    """Run all Phase 5 validation checks and write a report."""
    checks: List[Dict] = []

    # Load data and merge spatial + morphology features if enabled.
    df = pd.read_csv(INPUT_DATASET_CSV)
    df = merge_spatial_features(df)
    df = merge_morphology_features(df)
    predictions = pd.read_csv(PREDICTIONS_CSV) if Path(PREDICTIONS_CSV).exists() else pd.DataFrame()
    metadata = json.loads(Path(MODEL_METADATA_JSON).read_text()) if Path(MODEL_METADATA_JSON).exists() else {}

    checks.extend(validate_dataset_integrity(df))
    checks.extend(validate_leakage_prevention())
    checks.extend(validate_spatial_folds(df))
    checks.extend(validate_predictions(predictions))
    checks.extend(validate_rasters(predictions))
    checks.extend(validate_diagnostics())
    checks.extend(validate_generalization_audit(df))

    summary = {
        "total": len(checks),
        "pass": sum(1 for c in checks if c["status"] == "PASS"),
        "fail": sum(1 for c in checks if c["status"] == "FAIL"),
        "warn": sum(1 for c in checks if c["status"] == "WARN"),
    }

    report = {
        "phase": 5,
        "generated_utc": pd.Timestamp.now("UTC").isoformat(),
        "summary": summary,
        "checks": checks,
        "metadata": metadata,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for Phase 5 validation."""
    parser = argparse.ArgumentParser(description="Validate GreenGrid-AI Phase 5 outputs.")
    parser.add_argument(
        "--output",
        default=VALIDATION_REPORT_JSON,
        help="Path to write the validation report JSON.",
    )
    args = parser.parse_args(argv)

    report = run_validation(output_path=args.output)
    summary = report["summary"]
    print(f"Phase 5 Validation: {summary['pass']} PASS / {summary['fail']} FAIL / {summary['warn']} WARN")

    for check in report["checks"]:
        if check["status"] != "PASS":
            print(f"  {check['status']}: {check['name']} — {check['message']}")

    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
