"""End-to-end Phase 5 pipeline for UHI relative heat-severity classification.

Orchestrates:

1. Load Phase 4 dataset.
2. Define per-year relative heat-severity target.
3. Spatial cross-validation with baselines, Random Forest, and XGBoost.
4. Feature-ablation and year-sensitivity experiments.
5. Temporal generalization experiments.
6. Model selection using deterministic hierarchy.
7. Final model training and production prediction generation.
8. Raster/map generation.
9. Metadata, records, and tables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from . import maps as maps_module
from .config import (
    ABLATION_RESULTS_CSV,
    BASE_PREDICTOR_VARS,
    BOUNDARY_CASE_ANALYSIS_CSV,
    CLASSIFICATION_METRICS_CSV,
    CONFUSION_MATRICES_CSV,
    FEATURE_IMPORTANCE_CSV,
    INPUT_DATASET_CSV,
    MODEL_COMPARISON_CSV,
    MODEL_METADATA_JSON,
    MORPHOLOGY_FEATURE_COLS,
    OOF_PREDICTIONS_CSV,
    PERMUTATION_IMPORTANCE_CSV,
    PHASE5_DIR,
    PIPELINE_RECORD_JSON,
    PREDICTIONS_CSV,
    PREDICTOR_VARS,
    PROJECT_ROOT,
    RANDOM_SEED,
    SPATIAL_FEATURE_COLS,
    UHI_RASTER_2022,
    UHI_RASTER_2026,
    USE_MORPHOLOGY_FEATURES,
)
from .dataset import (
    encode_predictors,
    load_phase4_dataset,
    merge_morphology_features,
    merge_spatial_features,
)
from .evaluation import (
    aggregate_fold_metrics,
    boundary_case_records,
    confusion_matrix_records,
    format_metrics_table,
)
from .experiments import (
    run_ablations,
    run_baselines,
    run_random_forest,
    run_rule_baseline,
    run_temporal_validation,
    run_xgboost,
    select_final_model,
)
from .maps import generate_all_maps
from .random_forest import (
    predict_random_forest,
    random_forest_feature_importance,
    random_forest_permutation_importance,
    train_random_forest,
)
from .target import build_target, format_threshold_table, get_final_descriptive_thresholds
from .xgboost_model import predict_xgboost, train_xgboost, xgboost_feature_importance


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


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_phase5(
    skip_maps: bool = False,
    save_models: bool = True,
) -> Dict:
    """Run the complete Phase 5 pipeline."""
    PHASE5_DIR.mkdir(parents=True, exist_ok=True)
    record: Dict = {
        "phase": 5,
        "project": "GreenGrid-AI",
        "project_root": str(PROJECT_ROOT),
        "started_utc": _now(),
        "random_seed": RANDOM_SEED,
        "steps": {},
    }

    # ------------------------------------------------------------------
    # 1. Load dataset and merge spatial + morphology features
    # ------------------------------------------------------------------
    t0 = time.time()
    df = load_phase4_dataset()
    df = merge_spatial_features(df)
    df = merge_morphology_features(df)
    n_spatial_features = len([c for c in df.columns if c in SPATIAL_FEATURE_COLS])
    n_morphology_features = len([c for c in df.columns if c in MORPHOLOGY_FEATURE_COLS])

    morphology_clipping_summary = None
    if USE_MORPHOLOGY_FEATURES:
        from .morphology_features import compute_morphology_clipping_summary

        morphology_clipping_summary = compute_morphology_clipping_summary(df)

    record["steps"]["load_dataset"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "input_path": str(INPUT_DATASET_CSV),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "years": sorted(df["year"].unique().tolist()),
        "spatial_features_merged": n_spatial_features,
        "morphology_features_merged": n_morphology_features,
        "morphology_block_edge_clipping": morphology_clipping_summary,
    }

    # ------------------------------------------------------------------
    # 2. Final descriptive target thresholds
    # ------------------------------------------------------------------
    t0 = time.time()
    final_target, final_thresholds = build_target(df, row_mask=None)
    final_thresholds_df = format_threshold_table(final_thresholds)
    record["steps"]["target_definition"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "method": "per-year quartiles",
        "thresholds": final_thresholds,
    }

    # ------------------------------------------------------------------
    # 3. Baselines (majority, stratified, and NDVI/NDBI rule baseline)
    # ------------------------------------------------------------------
    t0 = time.time()
    baseline_results = run_baselines(df, predictor_cols=PREDICTOR_VARS)
    rule_baseline_result = run_rule_baseline(df)
    baseline_results.append(rule_baseline_result)
    record["steps"]["baselines"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "models": [r["model_name"] for r in baseline_results],
    }

    # ------------------------------------------------------------------
    # 4. Random Forest and XGBoost spatial CV
    # ------------------------------------------------------------------
    t0 = time.time()
    rf_result = run_random_forest(df, predictor_cols=PREDICTOR_VARS)
    record["steps"]["random_forest_cv"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "aggregated_metrics": rf_result["aggregated_metrics"],
    }

    t0 = time.time()
    xgb_result = run_xgboost(df, predictor_cols=PREDICTOR_VARS)
    record["steps"]["xgboost_cv"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "aggregated_metrics": xgb_result["aggregated_metrics"],
    }

    # ------------------------------------------------------------------
    # 5. Ablation experiments
    # ------------------------------------------------------------------
    t0 = time.time()
    ablation_results = run_ablations(df)
    record["steps"]["ablations"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "n_experiments": len(ablation_results),
    }

    # ------------------------------------------------------------------
    # 6. OOF diagnostics (OOF predictions, confusion matrices, boundary analysis)
    # ------------------------------------------------------------------
    t0 = time.time()
    all_cv_results = baseline_results + [rf_result, xgb_result]

    oof_df = pd.concat([r["oof_predictions"] for r in all_cv_results], ignore_index=True)
    _save_csv(oof_df, OOF_PREDICTIONS_CSV)

    confusion_records = []
    for r in all_cv_results:
        confusion_records.extend(confusion_matrix_records(r))
    _save_csv(pd.DataFrame(confusion_records), CONFUSION_MATRICES_CSV)

    boundary_records = []
    for r in [rf_result, xgb_result]:
        boundary_records.extend(boundary_case_records(r, df))
    _save_csv(pd.DataFrame(boundary_records), BOUNDARY_CASE_ANALYSIS_CSV)

    record["steps"]["oof_diagnostics"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "oof_predictions_csv": str(OOF_PREDICTIONS_CSV),
        "confusion_matrices_csv": str(CONFUSION_MATRICES_CSV),
        "boundary_case_analysis_csv": str(BOUNDARY_CASE_ANALYSIS_CSV),
        "n_oob_rows": len(oof_df),
    }

    # ------------------------------------------------------------------
    # 7. Temporal validation
    # ------------------------------------------------------------------
    t0 = time.time()
    temporal_results = run_temporal_validation(df, predictor_cols=PREDICTOR_VARS)
    record["steps"]["temporal_validation"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "experiments": [
            {
                "model": r["model_name"],
                "train_year": r["train_year"],
                "test_year": r["test_year"],
                "macro_f1": r["metrics"]["macro_f1"],
            }
            for r in temporal_results
        ],
    }

    # ------------------------------------------------------------------
    # 7. Model selection
    # ------------------------------------------------------------------
    t0 = time.time()
    selected_result, selection_rationale = select_final_model([rf_result, xgb_result])
    selected_model_name = selected_result["model_name"]
    record["steps"]["model_selection"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "selected_model": selected_model_name,
        "rationale": selection_rationale,
    }

    # ------------------------------------------------------------------
    # 8. Final production model training and predictions
    # ------------------------------------------------------------------
    t0 = time.time()
    X_full, feature_names = encode_predictors(df, predictor_cols=PREDICTOR_VARS)
    y_full = final_target.astype(int)

    if selected_model_name == "Random Forest":
        final_model = train_random_forest(X_full, y_full)
        final_predict_fn = predict_random_forest
    else:
        final_model = train_xgboost(X_full, y_full)
        final_predict_fn = predict_xgboost

    final_preds = final_predict_fn(final_model, X_full)

    predictions = df[["lon", "lat", "row", "col", "year", "spatial_block_id"]].copy()
    predictions["actual_class"] = y_full.values
    predictions["predicted_class"] = final_preds["y_pred"]
    for i, label in enumerate(["Low", "Moderate", "High", "Severe"]):
        predictions[f"probability_{label.lower()}"] = final_preds["y_prob"][:, i]
    predictions["prediction_confidence"] = np.max(final_preds["y_prob"], axis=1)

    _save_csv(predictions, PREDICTIONS_CSV)

    if save_models:
        model_path = PHASE5_DIR / "models" / f"{selected_model_name.lower().replace(' ', '_')}_final.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_path)

    # Feature importance on full model.
    if selected_model_name == "Random Forest":
        feat_imp = random_forest_feature_importance(final_model, feature_names)
    else:
        feat_imp = xgboost_feature_importance(final_model, feature_names)
    _save_csv(feat_imp, FEATURE_IMPORTANCE_CSV)

    # Permutation importance on an independent stratified hold-out sample.
    perm_imp = None
    if selected_model_name == "Random Forest":
        from sklearn.model_selection import train_test_split

        X_perm_train, X_perm_val, y_perm_train, y_perm_val = train_test_split(
            X_full,
            y_full,
            test_size=0.10,
            random_state=RANDOM_SEED,
            stratify=y_full,
        )
        perm_model = train_random_forest(X_perm_train, y_perm_train)
        perm_val = X_perm_val.sample(n=min(3_000, len(X_perm_val)), random_state=RANDOM_SEED)
        perm_val_labels = y_perm_val.loc[perm_val.index]
        perm_imp = random_forest_permutation_importance(
            perm_model, perm_val, perm_val_labels, feature_names, random_state=RANDOM_SEED, n_repeats=3
        )
        _save_csv(perm_imp, PERMUTATION_IMPORTANCE_CSV)

    record["steps"]["final_predictions"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "model": selected_model_name,
        "predictions_csv": str(PREDICTIONS_CSV),
        "n_predictions": len(predictions),
        "feature_importance_csv": str(FEATURE_IMPORTANCE_CSV),
        "permutation_importance_csv": str(PERMUTATION_IMPORTANCE_CSV) if perm_imp is not None else None,
    }

    # ------------------------------------------------------------------
    # 9. Maps and rasters
    # ------------------------------------------------------------------
    if not skip_maps:
        t0 = time.time()
        map_results = generate_all_maps(predictions)
        record["steps"]["maps"] = {
            "status": "success",
            "elapsed_s": round(time.time() - t0, 3),
            "outputs": [r["output_path"] for r in map_results],
        }
    else:
        record["steps"]["maps"] = {"status": "skipped"}

    # ------------------------------------------------------------------
    # 10. Save tables
    # ------------------------------------------------------------------
    t0 = time.time()

    # Model comparison table.
    comparison_records = []
    all_cv_results = baseline_results + [rf_result, xgb_result]
    for r in all_cv_results:
        m = r["aggregated_metrics"]
        comparison_records.append(
            {
                "model": r["model_name"],
                "mean_accuracy": round(m.get("accuracy_mean", np.nan), 4),
                "mean_balanced_accuracy": round(m.get("balanced_accuracy_mean", np.nan), 4),
                "mean_macro_f1": round(m.get("macro_f1_mean", np.nan), 4),
                "std_macro_f1": round(m.get("macro_f1_std", np.nan), 4),
                "mean_high_recall": round(m.get("high_recall_mean", np.nan), 4),
                "mean_severe_recall": round(m.get("severe_recall_mean", np.nan), 4),
            }
        )
    _save_csv(pd.DataFrame(comparison_records), MODEL_COMPARISON_CSV)

    # Per-fold classification metrics.
    metrics_records = []
    for r in all_cv_results:
        for fold_res in r["fold_results"]:
            rec = {"model": r["model_name"], "fold": fold_res["fold"]}
            for k, v in fold_res.items():
                if k != "confusion_matrix" and isinstance(v, (int, float, np.floating, np.integer)):
                    rec[k] = round(float(v), 5)
            metrics_records.append(rec)
    _save_csv(pd.DataFrame(metrics_records), CLASSIFICATION_METRICS_CSV)

    # Ablation results.
    ablation_records = []
    for r in ablation_results:
        m = r["aggregated_metrics"]
        ablation_records.append(
            {
                "experiment": r["experiment_name"],
                "model": r["model_name"],
                "features": ", ".join(r["predictor_cols"]),
                "mean_macro_f1": round(m.get("macro_f1_mean", np.nan), 4),
                "std_macro_f1": round(m.get("macro_f1_std", np.nan), 4),
                "mean_high_recall": round(m.get("high_recall_mean", np.nan), 4),
                "mean_severe_recall": round(m.get("severe_recall_mean", np.nan), 4),
            }
        )
    _save_csv(pd.DataFrame(ablation_records), ABLATION_RESULTS_CSV)

    record["steps"]["save_tables"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "outputs": [
            str(MODEL_COMPARISON_CSV),
            str(CLASSIFICATION_METRICS_CSV),
            str(ABLATION_RESULTS_CSV),
        ],
    }

    # ------------------------------------------------------------------
    # 11. Metadata
    # ------------------------------------------------------------------
    metadata = {
        "target": {
            "variable": "lst_C",
            "methodology": "per-year quartiles",
            "description": "Relative LST heat-severity classification: Low, Moderate, High, Severe",
            "caveat": "Classes are derived from LST and are not independently observed physical UHI ground truth.",
            "final_thresholds": final_thresholds,
        },
        "predictors": {
            "feature_names": feature_names,
            "excluded_features": ["lst_C", "spatial_block_id", "lon", "lat", "row", "col"],
            "categorical_encoding": {"landuse_class": "one-hot"},
            "spatial_means": {"enabled": True, "window_sizes_px": [3, 5, 11], "feature_count": len(SPATIAL_FEATURE_COLS)},
            "urban_morphology": {
                "enabled": USE_MORPHOLOGY_FEATURES,
                "radii_m": [50, 100, 250, 500],
                "feature_count": len(MORPHOLOGY_FEATURE_COLS),
                "block_edge_clipping": morphology_clipping_summary,
                "note": (
                    "Building/road/vegetation pixel fractions use the verified "
                    "distance == 0 feature-pixel proxy; they are not measured "
                    "footprint-area fractions."
                ),
            },
            "diagnostic_outputs": {
                "oof_predictions": str(OOF_PREDICTIONS_CSV),
                "confusion_matrices": str(CONFUSION_MATRICES_CSV),
                "boundary_case_analysis": str(BOUNDARY_CASE_ANALYSIS_CSV),
                "permutation_importance": str(PERMUTATION_IMPORTANCE_CSV),
            },
        },
        "models": {
            "random_forest": {
                "library": "scikit-learn",
                "params": rf_result.get("model_params", "fixed baseline params"),
            },
            "xgboost": {
                "library": "xgboost",
                "params": xgb_result.get("model_params", "fixed baseline params"),
            },
        },
        "validation": {
            "spatial_strategy": "GroupKFold on spatial_block_id",
            "n_folds": 5,
            "temporal_experiments": ["2022->2026", "2026->2022"],
            "target_leakage_prevention": "CV thresholds derived from training blocks only",
        },
        "model_selection": {
            "primary_metric": "mean spatial-CV macro F1",
            "tie_breakers": ["High + Severe recall", "macro F1 stability", "simpler model"],
            "selected_model": selected_model_name,
            "rationale": selection_rationale,
        },
        "results": {
            "model_comparison": comparison_records,
            "selected_model_macro_f1_mean": selected_result["aggregated_metrics"]["macro_f1_mean"],
        },
    }
    _save_json(metadata, MODEL_METADATA_JSON)

    # ------------------------------------------------------------------
    # 12. Pipeline record
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
        "metadata": metadata,
        "predictions": predictions,
        "selected_model_name": selected_model_name,
        "rf_result": rf_result,
        "xgb_result": xgb_result,
        "baseline_results": baseline_results,
        "ablation_results": ablation_results,
        "temporal_results": temporal_results,
    }


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for the Phase 5 pipeline."""
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 5 UHI detection and classification."
    )
    parser.add_argument(
        "--skip-maps",
        action="store_true",
        help="Skip PNG map and raster generation (faster for testing).",
    )
    args = parser.parse_args(argv)

    try:
        results = run_phase5(skip_maps=args.skip_maps)
        print(json.dumps(_convert_for_json(results["record"]), indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Phase 5 pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
