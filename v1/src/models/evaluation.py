"""Evaluation metrics and aggregation for Phase 5 models."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import CLASS_LABELS, CLASS_VALUES, INV_CLASS_MAPPING


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute a comprehensive set of classification metrics.

    Returns
    -------
    dict containing scalar metrics, per-class metrics, and confusion matrix.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    per_class_precision = precision_score(y_true, y_pred, labels=CLASS_VALUES, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=CLASS_VALUES, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=CLASS_VALUES, average=None, zero_division=0)

    for i, label in enumerate(CLASS_LABELS):
        metrics[f"{label.lower()}_precision"] = float(per_class_precision[i])
        metrics[f"{label.lower()}_recall"] = float(per_class_recall[i])
        metrics[f"{label.lower()}_f1"] = float(per_class_f1[i])

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=CLASS_VALUES).tolist()

    if y_prob is not None:
        metrics["mean_confidence"] = float(np.max(y_prob, axis=1).mean())

    return metrics


def metrics_to_dataframe(metrics_list: List[Dict]) -> pd.DataFrame:
    """Convert a list of per-fold metrics into a DataFrame."""
    records = []
    for m in metrics_list:
        record = {k: v for k, v in m.items() if k != "confusion_matrix"}
        records.append(record)
    return pd.DataFrame(records)


def aggregate_fold_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
    """Aggregate per-fold metrics by mean and std."""
    df = metrics_to_dataframe(metrics_list)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    aggregated = {}
    for col in numeric_cols:
        aggregated[f"{col}_mean"] = float(df[col].mean())
        aggregated[f"{col}_std"] = float(df[col].std())
    return aggregated


def format_metrics_table(aggregated: Dict[str, float]) -> pd.DataFrame:
    """Format aggregated metrics as a readable table."""
    rows = []
    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "high_recall",
        "severe_recall",
    ]:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        if mean_key in aggregated:
            rows.append(
                {
                    "metric": metric,
                    "mean": round(aggregated[mean_key], 4),
                    "std": round(aggregated.get(std_key, 0.0), 4),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Confusion-matrix export helpers
# ---------------------------------------------------------------------------


def confusion_matrix_records(cv_result: Dict) -> List[Dict]:
    """Return per-fold and aggregated confusion-matrix rows for ``cv_result``."""
    records = []
    model_name = cv_result["model_name"]

    # Per-fold matrices.
    for fold_res in cv_result["fold_results"]:
        fold = fold_res["fold"]
        cm = np.asarray(fold_res["confusion_matrix"])
        for actual_idx, actual_label in enumerate(CLASS_LABELS):
            for pred_idx, pred_label in enumerate(CLASS_LABELS):
                records.append({
                    "model": model_name,
                    "fold": fold,
                    "actual_class": actual_label,
                    "predicted_class": pred_label,
                    "count": int(cm[actual_idx, pred_idx]),
                })

    # Aggregated matrix across all OOF predictions.
    oof = cv_result.get("oof_predictions")
    if oof is not None and len(oof) > 0:
        cm = confusion_matrix(
            oof["actual_class"].values,
            oof["predicted_class"].values,
            labels=CLASS_VALUES,
        )
        for actual_idx, actual_label in enumerate(CLASS_LABELS):
            for pred_idx, pred_label in enumerate(CLASS_LABELS):
                records.append({
                    "model": model_name,
                    "fold": "aggregated",
                    "actual_class": actual_label,
                    "predicted_class": pred_label,
                    "count": int(cm[actual_idx, pred_idx]),
                })

    return records


# ---------------------------------------------------------------------------
# Boundary-case analysis helpers
# ---------------------------------------------------------------------------


def _distance_to_nearest_threshold(lst: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """Return min absolute distance of each LST value to q25/q50/q75."""
    t_vals = np.array([
        thresholds["low_to_moderate"],
        thresholds["moderate_to_high"],
        thresholds["high_to_severe"],
    ])
    return np.min(np.abs(lst[:, None] - t_vals[None, :]), axis=1)


def _boundary_bin(distance: np.ndarray) -> np.ndarray:
    """Classify distance-to-nearest-threshold into diagnostic bins."""
    return np.select(
        [
            distance < 0.5,
            (distance >= 0.5) & (distance <= 1.5),
            distance > 1.5,
        ],
        ["boundary (<0.5C)", "moderate (0.5-1.5C)", "clear (>1.5C)"],
        default="unknown",
    )


def boundary_case_records(
    cv_result: Dict,
    df: pd.DataFrame,
) -> List[Dict]:
    """Return boundary-case analysis records for a CV result.

    For each fold, computes distance-to-nearest-threshold for validation pixels
    using that fold's training-derived thresholds, then reports accuracy in
    distance bins and per-boundary error rates.
    """
    records = []
    model_name = cv_result["model_name"]
    oof = cv_result.get("oof_predictions")
    if oof is None or len(oof) == 0:
        return records

    fold_records = cv_result.get("fold_records", [])

    for fold_res, fold_rec in zip(cv_result["fold_results"], fold_records):
        fold = fold_res["fold"]
        thresholds = fold_rec["thresholds"]
        val_idx = oof[oof["fold"] == fold].index
        # Re-align OOF rows to original dataframe rows by identifiers.
        val_df = oof[oof["fold"] == fold].copy()
        # Merge LST from original dataframe via identifiers.
        merged = val_df.merge(
            df[["row", "col", "year", "lst_C"]],
            on=["row", "col", "year"],
            how="left",
        )

        for year, year_thresholds in thresholds.items():
            year_mask = merged["year"] == year
            if year_mask.sum() == 0:
                continue
            sub = merged[year_mask].copy()
            distances = _distance_to_nearest_threshold(
                sub["lst_C"].values, year_thresholds
            )
            sub["distance_to_nearest_threshold"] = distances
            sub["boundary_bin"] = _boundary_bin(distances)

            # Overall accuracy per bin.
            for bin_name in ["boundary (<0.5C)", "moderate (0.5-1.5C)", "clear (>1.5C)"]:
                bin_mask = sub["boundary_bin"] == bin_name
                if bin_mask.sum() == 0:
                    continue
                bin_sub = sub[bin_mask]
                acc = accuracy_score(
                    bin_sub["actual_class"].values,
                    bin_sub["predicted_class"].values,
                )
                records.append({
                    "model": model_name,
                    "fold": fold,
                    "year": year,
                    "boundary": bin_name,
                    "n_pixels": int(bin_mask.sum()),
                    "accuracy": round(float(acc), 5),
                    "boundary_pair": "all",
                })

            # Per-boundary error rates for adjacent classes.
            boundary_defs = [
                ("Low↔Moderate", "low_to_moderate", 0, 1),
                ("Moderate↔High", "moderate_to_high", 1, 2),
                ("High↔Severe", "high_to_severe", 2, 3),
            ]
            for pair_name, thresh_key, class_a, class_b in boundary_defs:
                t = year_thresholds[thresh_key]
                near_mask = np.abs(sub["lst_C"].values - t) < 0.5
                near_mask &= sub["actual_class"].isin([class_a, class_b]).values
                if near_mask.sum() == 0:
                    continue
                near_sub = sub[near_mask]
                acc = accuracy_score(
                    near_sub["actual_class"].values,
                    near_sub["predicted_class"].values,
                )
                records.append({
                    "model": model_name,
                    "fold": fold,
                    "year": year,
                    "boundary": "<0.5C from threshold",
                    "n_pixels": int(near_mask.sum()),
                    "accuracy": round(float(acc), 5),
                    "boundary_pair": pair_name,
                })

    return records


# ---------------------------------------------------------------------------
# Train/validation gap helpers
# ---------------------------------------------------------------------------


def train_validation_gap_records(cv_result: Dict) -> List[Dict]:
    """Return train-vs-validation accuracy/macro-F1 gap per fold."""
    records = []
    for fold_res in cv_result["fold_results"]:
        records.append({
            "model": cv_result["model_name"],
            "fold": fold_res["fold"],
            "train_accuracy": round(fold_res.get("train_accuracy", np.nan), 5),
            "train_macro_f1": round(fold_res.get("train_macro_f1", np.nan), 5),
            "val_accuracy": round(fold_res.get("accuracy", np.nan), 5),
            "val_macro_f1": round(fold_res.get("macro_f1", np.nan), 5),
            "accuracy_gap": round(
                fold_res.get("train_accuracy", np.nan) - fold_res.get("accuracy", np.nan), 5
            ),
            "macro_f1_gap": round(
                fold_res.get("train_macro_f1", np.nan) - fold_res.get("macro_f1", np.nan), 5
            ),
        })
    return records
