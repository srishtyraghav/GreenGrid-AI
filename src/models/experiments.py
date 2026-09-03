"""Experiment runners for Phase 5: spatial CV, temporal validation, ablations.

Each runner is model-agnostic and accepts a train/predict callable pair.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from .baseline import (
    baseline_model_name,
    predict_baseline,
    predict_rule_baseline,
    train_majority_baseline,
    train_rule_baseline,
    train_stratified_baseline,
)
from .config import (
    CLASS_LABELS,
    CLASS_VALUES,
    INV_CLASS_MAPPING,
    N_SPATIAL_FOLDS,
    PREDICTOR_VARS,
    RANDOM_FOREST_PARAMS,
    RANDOM_SEED,
    XGBOOST_PARAMS,
)
from .dataset import get_feature_columns_for_model, load_phase4_dataset
from .evaluation import aggregate_fold_metrics, compute_metrics
from .random_forest import predict_random_forest, train_random_forest
from .target import build_target, get_final_descriptive_thresholds
from .validation import create_spatial_folds, get_fold_blocks, temporal_train_test_split
from .xgboost_model import predict_xgboost, train_xgboost


def run_spatial_cv(
    df: pd.DataFrame,
    predictor_cols: List[str],
    model_name: str,
    train_fn: Callable,
    predict_fn: Callable,
    n_splits: int = N_SPATIAL_FOLDS,
) -> Dict:
    """Run spatial cross-validation for a single model.

    Parameters
    ----------
    df : pd.DataFrame
        Phase 4 input dataset.
    predictor_cols : list[str]
        Predictor columns to use.
    model_name : str
        Display name for this experiment.
    train_fn : callable
        Function(model_name ignored, X_train, y_train) -> fitted_model.
    predict_fn : callable
        Function(fitted_model, X) -> dict with 'y_pred' and 'y_prob'.
    n_splits : int
        Number of spatial folds.

    Returns
    -------
    dict with fold results, aggregated metrics, OOF predictions, and fold info.
    """
    from .dataset import encode_predictors
    from .target import build_target

    X_full, feature_names = encode_predictors(df, predictor_cols=predictor_cols)
    groups_full = df["spatial_block_id"].astype(int)

    fold_results = []
    oof_predictions = []
    fold_records = []

    for fold, (train_idx, val_idx) in enumerate(
        create_spatial_folds(groups_full, n_splits=n_splits), start=1
    ):
        train_blocks, val_blocks = get_fold_blocks(groups_full, train_idx, val_idx)

        if set(train_blocks).intersection(val_blocks):
            raise ValueError(f"Fold {fold}: train/val block overlap detected.")

        # Training-only thresholds for this fold.
        row_mask = np.zeros(len(df), dtype=bool)
        row_mask[train_idx] = True
        y_series, thresholds = build_target(df, row_mask=row_mask)

        X_train = X_full.iloc[train_idx]
        X_val = X_full.iloc[val_idx]
        y_train = y_series.iloc[train_idx].astype(int)
        y_val = y_series.iloc[val_idx].astype(int)

        model = train_fn(model_name, X_train, y_train)
        preds = predict_fn(model, X_val)

        metrics = compute_metrics(y_val.values, preds["y_pred"], preds.get("y_prob"))

        # Training-set metrics for overfitting/gap diagnostics.
        train_preds = predict_fn(model, X_train)
        train_metrics = compute_metrics(
            y_train.values, train_preds["y_pred"], train_preds.get("y_prob")
        )

        # Store OOF predictions with identifiers.
        identifiers = df[["lon", "lat", "row", "col", "year", "spatial_block_id"]].iloc[val_idx].copy()
        identifiers["actual_class"] = y_val.values
        identifiers["predicted_class"] = preds["y_pred"]
        for i, label in enumerate(CLASS_LABELS):
            identifiers[f"probability_{label.lower()}"] = preds["y_prob"][:, i]
        identifiers["prediction_confidence"] = np.max(preds["y_prob"], axis=1)
        identifiers["fold"] = fold
        identifiers["model"] = model_name
        oof_predictions.append(identifiers)

        fold_records.append(
            {
                "fold": fold,
                "train_blocks": train_blocks,
                "val_blocks": val_blocks,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "thresholds": thresholds,
                "train_class_counts": y_train.value_counts().sort_index().to_dict(),
                "val_class_counts": y_val.value_counts().sort_index().to_dict(),
            }
        )
        train_metric_records = {
            f"train_{k}": v for k, v in train_metrics.items() if k != "confusion_matrix"
        }
        fold_results.append({"fold": fold, **train_metric_records, **metrics})

    oof_df = pd.concat(oof_predictions, ignore_index=True)

    return {
        "model_name": model_name,
        "feature_names": feature_names,
        "predictor_cols": predictor_cols,
        "fold_results": fold_results,
        "aggregated_metrics": aggregate_fold_metrics(fold_results),
        "oof_predictions": oof_df,
        "fold_records": fold_records,
    }


def run_baselines(df: pd.DataFrame, predictor_cols: List[str] = PREDICTOR_VARS) -> List[Dict]:
    """Run majority and stratified baselines under spatial CV."""
    results = []

    def train_baseline(strategy: str, model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
        if strategy == "most_frequent":
            return train_majority_baseline(y_train)
        return train_stratified_baseline(y_train, random_state=RANDOM_SEED)

    for strategy in ["most_frequent", "stratified"]:
        name = baseline_model_name(strategy)

        def make_train(s):
            return lambda n, X, y: train_baseline(s, n, X, y)

        res = run_spatial_cv(
            df,
            predictor_cols,
            name,
            train_fn=make_train(strategy),
            predict_fn=predict_baseline,
        )
        results.append(res)
    return results


def run_rule_baseline(
    df: pd.DataFrame,
    n_splits: int = N_SPATIAL_FOLDS,
) -> Dict:
    """Run the fixed NDVI/NDBI rule baseline under spatial cross-validation.

    The rule baseline operates on the raw environmental columns rather than the
    one-hot encoded predictor matrix, so it is implemented as a small wrapper
    around the same fold generation logic.
    """
    from .target import build_target

    groups_full = df["spatial_block_id"].astype(int)
    fold_results = []
    oof_predictions = []
    fold_records = []

    for fold, (train_idx, val_idx) in enumerate(
        create_spatial_folds(groups_full, n_splits=n_splits), start=1
    ):
        train_blocks, val_blocks = get_fold_blocks(groups_full, train_idx, val_idx)
        if set(train_blocks).intersection(val_blocks):
            raise ValueError(f"Fold {fold}: train/val block overlap detected.")

        row_mask = np.zeros(len(df), dtype=bool)
        row_mask[train_idx] = True
        y_series, thresholds = build_target(df, row_mask=row_mask)

        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
        y_train = y_series.iloc[train_idx].astype(int)
        y_val = y_series.iloc[val_idx].astype(int)

        model = train_rule_baseline(df_train)
        preds = predict_rule_baseline(model, df_val)

        metrics = compute_metrics(y_val.values, preds["y_pred"], preds["y_prob"])

        # Store OOF predictions with identifiers.
        identifiers = df_val[["lon", "lat", "row", "col", "year", "spatial_block_id"]].copy()
        identifiers["actual_class"] = y_val.values
        identifiers["predicted_class"] = preds["y_pred"]
        for i, label in enumerate(CLASS_LABELS):
            identifiers[f"probability_{label.lower()}"] = preds["y_prob"][:, i]
        identifiers["prediction_confidence"] = np.max(preds["y_prob"], axis=1)
        identifiers["fold"] = fold
        identifiers["model"] = "Rule Baseline"
        oof_predictions.append(identifiers)

        fold_records.append(
            {
                "fold": fold,
                "train_blocks": train_blocks,
                "val_blocks": val_blocks,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "thresholds": thresholds,
                "train_class_counts": y_train.value_counts().sort_index().to_dict(),
                "val_class_counts": y_val.value_counts().sort_index().to_dict(),
            }
        )
        fold_results.append({"fold": fold, **metrics})

    oof_df = pd.concat(oof_predictions, ignore_index=True)

    return {
        "model_name": "Rule Baseline",
        "feature_names": ["ndvi", "ndbi", "year"],
        "predictor_cols": ["ndvi", "ndbi", "year"],
        "fold_results": fold_results,
        "aggregated_metrics": aggregate_fold_metrics(fold_results),
        "oof_predictions": oof_df,
        "fold_records": fold_records,
    }


def run_random_forest(df: pd.DataFrame, predictor_cols: List[str] = PREDICTOR_VARS) -> Dict:
    """Run Random Forest under spatial CV."""
    result = run_spatial_cv(
        df,
        predictor_cols,
        "Random Forest",
        train_fn=lambda n, X, y: train_random_forest(X, y),
        predict_fn=predict_random_forest,
    )
    result["model_params"] = RANDOM_FOREST_PARAMS.copy()
    return result


def run_xgboost(df: pd.DataFrame, predictor_cols: List[str] = PREDICTOR_VARS) -> Dict:
    """Run XGBoost under spatial CV."""
    result = run_spatial_cv(
        df,
        predictor_cols,
        "XGBoost",
        train_fn=lambda n, X, y: train_xgboost(X, y),
        predict_fn=predict_xgboost,
    )
    result["model_params"] = XGBOOST_PARAMS.copy()
    return result


def run_temporal_validation(
    df: pd.DataFrame,
    predictor_cols: List[str] = PREDICTOR_VARS,
) -> List[Dict]:
    """Run temporal generalization experiments in both directions."""
    results = []

    def run_one_direction(train_year: int, test_year: int, model_name: str) -> Dict:
        split = temporal_train_test_split(df, predictor_cols, train_year, test_year)
        if model_name == "Random Forest":
            model = train_random_forest(split["X_train"], split["y_train"])
            preds = predict_random_forest(model, split["X_test"])
        else:
            model = train_xgboost(split["X_train"], split["y_train"])
            preds = predict_xgboost(model, split["X_test"])

        metrics = compute_metrics(
            split["y_test"].values, preds["y_pred"], preds.get("y_prob")
        )

        predictions = split["identifiers_test"].copy()
        predictions["actual_class"] = split["y_test"].values
        predictions["predicted_class"] = preds["y_pred"]
        for i, label in enumerate(CLASS_LABELS):
            predictions[f"probability_{label.lower()}"] = preds["y_prob"][:, i]
        predictions["prediction_confidence"] = np.max(preds["y_prob"], axis=1)
        predictions["model"] = model_name
        predictions["train_year"] = train_year
        predictions["test_year"] = test_year

        return {
            "model_name": model_name,
            "train_year": train_year,
            "test_year": test_year,
            "metrics": metrics,
            "predictions": predictions,
            "thresholds_train": split["thresholds_train"],
            "thresholds_test": split["thresholds_test"],
        }

    for train_year, test_year in [(2022, 2026), (2026, 2022)]:
        for model_name in ["Random Forest", "XGBoost"]:
            results.append(run_one_direction(train_year, test_year, model_name))

    return results


def run_ablations(df: pd.DataFrame) -> List[Dict]:
    """Run feature-ablation and year-sensitivity experiments under spatial CV.

    The morphology variants let us compare the frozen spatial-mean benchmark
    against the new urban-morphology representation under identical folds.
    """
    from .config import BASE_PREDICTOR_VARS, USE_MORPHOLOGY_FEATURES, USE_SPATIAL_FEATURES

    # Core controlled comparison for the morphology experiment.  Additional
    # feature-family attributions are deferred until we know whether the full
    # morphology representation helps at all.
    experiments = [
        ("Full features (spatial + morphology)",
         get_feature_columns_for_model(df, include_year=True, include_pvc=True,
                                       include_spatial=True, include_morphology=True)),
        ("Spatial means only (frozen benchmark)",
         get_feature_columns_for_model(df, include_year=True, include_pvc=True,
                                       include_spatial=True, include_morphology=False)),
        ("Urban morphology only",
         get_feature_columns_for_model(df, include_year=True, include_pvc=True,
                                       include_spatial=False, include_morphology=True)),
        ("No spatial context", BASE_PREDICTOR_VARS),
    ]

    results = []
    for name, cols in experiments:
        rf_res = run_random_forest(df, predictor_cols=cols)
        rf_res["experiment_name"] = name
        rf_res["model_name"] = "Random Forest"
        results.append(rf_res)

        xgb_res = run_xgboost(df, predictor_cols=cols)
        xgb_res["experiment_name"] = name
        xgb_res["model_name"] = "XGBoost"
        results.append(xgb_res)

    return results


def select_final_model(
    cv_results: List[Dict],
) -> Tuple[Dict, str]:
    """Select the best model using the deterministic hierarchy.

    Hierarchy:
      1. Highest mean spatial-CV macro F1.
      2. Tie-breaker: highest mean High + Severe recall.
      3. Second tie-breaker: lowest fold-to-fold std of macro F1.
      4. Third tie-breaker: prefer simpler model (Random Forest over XGBoost).

    Returns
    -------
    tuple of (selected result dict, selection rationale string).
    """
    # Filter out baselines if they are included.
    candidate_results = [r for r in cv_results if r["model_name"] in ("Random Forest", "XGBoost")]

    def score(r):
        m = r["aggregated_metrics"]
        return (
            -m.get("macro_f1_mean", -999),  # higher is better -> negate
            -m.get("high_recall_mean", 0) - m.get("severe_recall_mean", 0),
            m.get("macro_f1_std", 999),  # lower is better
            0 if r["model_name"] == "Random Forest" else 1,  # prefer RF
        )

    ranked = sorted(candidate_results, key=score)
    selected = ranked[0]

    rationale = (
        f"Selected {selected['model_name']} based on: "
        f"mean macro F1 = {selected['aggregated_metrics']['macro_f1_mean']:.4f}, "
        f"High+Severe recall = {selected['aggregated_metrics']['high_recall_mean'] + selected['aggregated_metrics']['severe_recall_mean']:.4f}, "
        f"macro F1 std = {selected['aggregated_metrics']['macro_f1_std']:.4f}."
    )
    return selected, rationale
