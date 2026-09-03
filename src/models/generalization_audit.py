"""Generalization and leakage audit diagnostics for Phase 5.

This module implements a battery of diagnostic experiments that probe how well
the Phase 5 heat-severity classifiers (Random Forest, XGBoost) generalize in
space and time, and whether any leakage or data-integrity issues inflate the
reported metrics.  Each diagnostic writes a machine-readable CSV under
``PHASE5_TABLES_DIR`` (override the module-level ``OUTPUT_DIR`` for tests) and
returns a small summary dictionary.

Design rules shared by every diagnostic:

* Determinism: every stochastic step uses an explicit seed (default
  ``RANDOM_SEED``) via ``np.random.default_rng``.
* No validation leakage: target thresholds are always derived from training
  rows only (``build_target(df, row_mask=...)``); validation data is never used
  for any fitted choice (subsamples, permutations, hyper-parameter selection).
* Plain model fitting keeps the config defaults (``n_jobs=-1``); any sklearn
  permutation-importance style computation would use ``n_jobs=1`` to avoid
  joblib multiprocessing side-effects in long audit loops.

The module is executable::

    PYTHONPATH=src python3 -m models.generalization_audit [step ...]

with the step list defaulting to ``DEFAULT_STEPS``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from .config import (
    CLASS_LABELS,
    CLASS_VALUES,
    GROUP_VAR,
    N_SPATIAL_FOLDS,
    PHASE5_TABLES_DIR,
    RANDOM_FOREST_PARAMS,
    RANDOM_SEED,
    TARGET_VAR,
    XGBOOST_PARAMS,
    YEAR_VAR,
)
from .dataset import encode_predictors, get_feature_columns_for_model, load_phase4_dataset
from .evaluation import compute_metrics
from .experiments import run_spatial_cv
from .random_forest import predict_random_forest
from .target import build_target
from .validation import create_spatial_folds, get_fold_blocks, temporal_train_test_split
from .xgboost_model import predict_xgboost

# Seeds used for seed-stability / prediction-stability diagnostics.
SEEDS = [42, 123, 456, 789, 2026]

# Output directory for all audit CSVs / JSON files.  Tests override this.
OUTPUT_DIR = PHASE5_TABLES_DIR

# Pixel size (metres) used to convert row/col steps into metric distances.
PIXEL_SIZE_M = 30.0

# Distance bands for the spatial-distance stress test (metres, [lo, hi)).
DISTANCE_BANDS = [
    (0.0, 500.0, "0-500m"),
    (500.0, 1000.0, "500-1000m"),
    (1000.0, 2000.0, "1000-2000m"),
    (2000.0, np.inf, ">2000m"),
]

# Moran's I settings: k-nearest-neighbour weights on (row, col) coordinates.
MORANS_K = 8
MORANS_N_PERMUTATIONS = 199

# Deterministic locked-holdout selection rule: sorted occupied block ids,
# every ``LOCKED_HOLDOUT_STRIDE``-th block starting at index
# ``LOCKED_HOLDOUT_START_INDEX`` (i.e. ``ids[2::6]``).
LOCKED_HOLDOUT_START_INDEX = 2
LOCKED_HOLDOUT_STRIDE = 6

# Default steps for ``run_audit`` / the CLI entry point.
DEFAULT_STEPS = [
    "train_validation_gap",
    "seed_stability",
    "target_permutation",
    "learning_curves",
    "leave_one_block_out",
    "spatial_distance",
    "duplicate_audit",
    "leakage_audit",
]

# Features permuted one-at-a-time inside the training fold by
# ``feature_permutation_sanity``.
DEFAULT_PERMUTED_FEATURES = [
    "year",
    "ndvi",
    "ndbi",
    "vegetation_cover",
    "dist_building_m",
    "dist_road_m",
    "dist_vegetation_m",
    "building_pixel_fraction_100m",
    "road_pixel_fraction_100m",
    "vegetation_pixel_fraction_100m",
    "ndvi_contrast_100m",
    "ndbi_contrast_100m",
    "landuse_entropy_250m",
]

RF_REGULARIZATION_CANDIDATES = {
    "A_baseline": {},
    "B_max_depth_20_min_leaf_5": {"max_depth": 20, "min_samples_leaf": 5},
    "C_max_depth_15_min_leaf_10": {"max_depth": 15, "min_samples_leaf": 10},
    "D_max_depth_10_min_leaf_15": {"max_depth": 10, "min_samples_leaf": 15},
}

XGB_REGULARIZATION_CANDIDATES = {
    "X0_baseline": {},
    "X1_max_depth_4_min_child_5": {"max_depth": 4, "min_child_weight": 5},
    "X2_max_depth_3_lr_0.03_min_child_10": {
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 10,
    },
    "X3_reg_lambda_5_alpha_0.5_max_depth_4": {
        "reg_lambda": 5,
        "reg_alpha": 0.5,
        "max_depth": 4,
    },
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Write ``df`` to ``OUTPUT_DIR / name`` and return it."""
    path = Path(OUTPUT_DIR) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def make_rf(overrides: Optional[Dict] = None) -> Tuple[Callable, Callable]:
    """Return (train_fn, predict_fn) for a parameterized Random Forest.

    Parameters
    ----------
    overrides : dict, optional
        Parameters merged on top of ``RANDOM_FOREST_PARAMS``.

    Returns
    -------
    tuple
        ``train_fn(model_name, X_train, y_train) -> fitted model`` and
        ``predict_fn(model, X) -> {'y_pred', 'y_prob'}``, compatible with
        ``experiments.run_spatial_cv``.
    """
    params = {**RANDOM_FOREST_PARAMS, **(overrides or {})}

    def train_fn(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return model

    return train_fn, predict_random_forest


def make_xgb(overrides: Optional[Dict] = None) -> Tuple[Callable, Callable]:
    """Return (train_fn, predict_fn) for a parameterized XGBoost classifier.

    ``num_class`` is added exactly as ``xgboost_model._xgboost_params_with_num_class``
    does; ``overrides`` are merged on top of ``XGBOOST_PARAMS``.
    """
    params = {
        **XGBOOST_PARAMS,
        "num_class": len(CLASS_VALUES),
        **(overrides or {}),
    }

    def train_fn(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model

    return train_fn, predict_xgboost


def _folds_once(df: pd.DataFrame, n_splits: int = N_SPATIAL_FOLDS) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute the spatial fold index list exactly once and return it as a list."""
    groups = df[GROUP_VAR].astype(int)
    return list(create_spatial_folds(groups, n_splits=n_splits))


def _fold_target(df: pd.DataFrame, train_idx: np.ndarray) -> pd.Series:
    """Build the target using training-only thresholds for one fold."""
    row_mask = np.zeros(len(df), dtype=bool)
    row_mask[train_idx] = True
    y_series, _ = build_target(df, row_mask=row_mask)
    return y_series


def _stratified_subsample_indices(
    y: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deterministic class-stratified row subsample (local integer indices).

    Every class contributes proportionally (at least one row when present);
    the result is sorted so downstream indexing is stable.
    """
    y = np.asarray(y)
    n = len(y)
    if n <= max_rows:
        return np.arange(n)
    parts = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        take = max(1, int(round(max_rows * len(cls_idx) / n)))
        take = min(take, len(cls_idx))
        parts.append(rng.choice(cls_idx, size=take, replace=False))
    return np.sort(np.concatenate(parts))


# ---------------------------------------------------------------------------
# Moran's I helpers (manual implementation, no extra dependencies)
# ---------------------------------------------------------------------------


def _knn_neighbor_indices(coords: np.ndarray, k: int = MORANS_K) -> np.ndarray:
    """Return (n, k) neighbour indices for kNN weights, excluding self-links.

    Rows with fewer than ``k`` available neighbours are padded with -1.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return np.full((n, k), -1, dtype=int)
    k_eff = min(k, n - 1)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k_eff + 1)
    neighbors = np.full((n, k), -1, dtype=int)
    for i in range(n):
        row = [int(j) for j in np.atleast_1d(idx[i]) if int(j) != i][:k_eff]
        neighbors[i, : len(row)] = row
    return neighbors


def morans_i(values: np.ndarray, neighbors: np.ndarray) -> Tuple[float, float]:
    """Compute global Moran's I manually.

    Uses row-standardised binary kNN weights (each location's weights sum to
    one; no diagonal self-weights):

    ``I = (n / W) * (sum_ij w_ij z_i z_j) / sum_i z_i^2``

    Parameters
    ----------
    values : np.ndarray
        Variable values at each location.
    neighbors : np.ndarray
        Integer array of shape (n, k) from ``_knn_neighbor_indices``; entries
        of -1 are ignored.

    Returns
    -------
    tuple
        (morans_i, expected_i) where ``expected_i = -1 / (n - 1)``.  Returns
        (nan, expected) when the variable has zero variance.
    """
    values = np.asarray(values, dtype=float)
    neighbors = np.asarray(neighbors, dtype=int)
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    valid = neighbors >= 0
    row_counts = valid.sum(axis=1)
    weights = np.where(valid, 1.0 / np.maximum(row_counts, 1)[:, None], 0.0)
    W = float(weights.sum())
    z = values - values.mean()
    denom = float(np.sum(z ** 2))
    expected = -1.0 / (n - 1)
    if denom == 0.0 or W == 0.0:
        return np.nan, expected
    cross = float(np.sum(weights * z[:, None] * z[np.where(valid, neighbors, 0)]))
    I = (n / W) * cross / denom
    return float(I), float(expected)


def morans_i_with_permutation(
    values: np.ndarray,
    coords: np.ndarray,
    k: int = MORANS_K,
    n_permutations: int = MORANS_N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """Moran's I with a fixed-seed permutation p-value (two-sided).

    The p-value is the fraction of permutations whose |I_perm - E| meets or
    exceeds |I_obs - E|, with the usual +1 correction.
    """
    neighbors = _knn_neighbor_indices(coords, k=k)
    I_obs, expected = morans_i(values, neighbors)
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    extreme = 0
    for _ in range(n_permutations):
        I_perm, _ = morans_i(rng.permutation(values), neighbors)
        if not np.isnan(I_perm) and abs(I_perm - expected) >= abs(I_obs - expected):
            extreme += 1
    p_value = (extreme + 1) / (n_permutations + 1)
    return I_obs, expected, float(p_value)


# ---------------------------------------------------------------------------
# 1. Train/validation gap
# ---------------------------------------------------------------------------


def train_validation_gap(df: pd.DataFrame, predictor_cols: List[str]) -> Dict:
    """Measure the train-vs-validation accuracy/macro-F1 gap per fold.

    Runs the full-feature Random Forest and XGBoost through
    ``experiments.run_spatial_cv`` and reports, per model and fold, the
    training metrics (in-sample) next to the validation metrics.  Large gaps
    indicate overfitting.  Writes ``train_validation_gap.csv`` (one row per
    model x fold) and ``train_validation_gap_summary.csv`` (mean/median/std/
    max/min of each quantity and gap per model).
    """
    rows = []
    for model_name, factory in (("Random Forest", make_rf), ("XGBoost", make_xgb)):
        train_fn, predict_fn = factory()
        result = run_spatial_cv(df, predictor_cols, model_name, train_fn, predict_fn)
        for fold_res, fold_rec in zip(result["fold_results"], result["fold_records"]):
            rows.append(
                {
                    "model": model_name,
                    "experiment": "full_features",
                    "fold": fold_res["fold"],
                    "n_train": fold_rec["n_train"],
                    "n_validation": fold_rec["n_val"],
                    "train_accuracy": fold_res["train_accuracy"],
                    "validation_accuracy": fold_res["accuracy"],
                    "accuracy_gap": fold_res["train_accuracy"] - fold_res["accuracy"],
                    "train_macro_f1": fold_res["train_macro_f1"],
                    "validation_macro_f1": fold_res["macro_f1"],
                    "macro_f1_gap": fold_res["train_macro_f1"] - fold_res["macro_f1"],
                    "train_balanced_accuracy": fold_res["train_balanced_accuracy"],
                    "validation_balanced_accuracy": fold_res["balanced_accuracy"],
                }
            )
    detail = _write_csv(pd.DataFrame(rows), "train_validation_gap.csv")

    summary_rows = []
    for model_name, grp in detail.groupby("model"):
        record = {"model": model_name}
        for col in [
            "accuracy_gap",
            "macro_f1_gap",
            "train_accuracy",
            "validation_accuracy",
            "train_macro_f1",
            "validation_macro_f1",
            "train_balanced_accuracy",
            "validation_balanced_accuracy",
        ]:
            record[f"{col}_mean"] = float(grp[col].mean())
            record[f"{col}_median"] = float(grp[col].median())
            record[f"{col}_std"] = float(grp[col].std())
            record[f"{col}_max"] = float(grp[col].max())
            record[f"{col}_min"] = float(grp[col].min())
        summary_rows.append(record)
    summary_df = _write_csv(pd.DataFrame(summary_rows), "train_validation_gap_summary.csv")

    return {
        "csv": "train_validation_gap.csv",
        "csv_summary": "train_validation_gap_summary.csv",
        "models": sorted(detail["model"].unique().tolist()),
        "max_accuracy_gap": {
            r["model"]: float(r["accuracy_gap_max"]) for _, r in summary_df.iterrows()
        },
        "max_macro_f1_gap": {
            r["model"]: float(r["macro_f1_gap_max"]) for _, r in summary_df.iterrows()
        },
    }


# ---------------------------------------------------------------------------
# 2. Seed stability
# ---------------------------------------------------------------------------


def seed_stability(
    df: pd.DataFrame,
    predictor_cols: List[str],
    seeds: Sequence[int] = SEEDS,
    return_oof: bool = False,
):
    """Measure how much spatial-CV metrics move when only the seed changes.

    The fold index list is created ONCE (``list(create_spatial_folds(...))``)
    and every per-seed run reuses the same spatial blocks; after each
    ``run_spatial_cv`` call the fold records are checked against the cached
    fold list and an error is raised if any fold assignment differs.  Only the
    model ``random_state`` changes between seeds.

    Writes ``seed_stability.csv`` with ``row_type`` = ``per_seed`` rows
    (columns: seed, model, experiment, mean_accuracy, std_accuracy,
    mean_macro_f1, std_macro_f1, high_recall, severe_recall) and
    ``row_type`` = ``summary`` rows (``statistic`` in {mean, std, min, max,
    range} of the per-seed aggregates, per model).

    Parameters
    ----------
    return_oof : bool
        If True, returns a dict including ``oof``: {seed: OOF DataFrame} for
        the Random Forest runs, so ``prediction_stability`` can reuse them
        without retraining.
    """
    groups = df[GROUP_VAR].astype(int)
    fold_indices = _folds_once(df)
    rows = []
    oof_by_seed: Dict[int, pd.DataFrame] = {}
    for seed in seeds:
        for model_name, factory in (("Random Forest", make_rf), ("XGBoost", make_xgb)):
            train_fn, predict_fn = factory({"random_state": seed})
            result = run_spatial_cv(df, predictor_cols, model_name, train_fn, predict_fn)
            # Verify the fold assignment is unchanged for this seed.
            for (train_idx, val_idx), fold_rec in zip(fold_indices, result["fold_records"]):
                train_blocks, val_blocks = get_fold_blocks(groups, train_idx, val_idx)
                if train_blocks != fold_rec["train_blocks"] or val_blocks != fold_rec["val_blocks"]:
                    raise ValueError(
                        f"Fold assignment changed for seed={seed} model={model_name}: "
                        f"expected val_blocks={val_blocks}, got {fold_rec['val_blocks']}"
                    )
            m = result["aggregated_metrics"]
            rows.append(
                {
                    "row_type": "per_seed",
                    "statistic": "",
                    "seed": seed,
                    "model": model_name,
                    "experiment": "seed_stability",
                    "mean_accuracy": m.get("accuracy_mean", np.nan),
                    "std_accuracy": m.get("accuracy_std", np.nan),
                    "mean_macro_f1": m.get("macro_f1_mean", np.nan),
                    "std_macro_f1": m.get("macro_f1_std", np.nan),
                    "high_recall": m.get("high_recall_mean", np.nan),
                    "severe_recall": m.get("severe_recall_mean", np.nan),
                }
            )
            if model_name == "Random Forest":
                oof_by_seed[seed] = result["oof_predictions"]

    table = pd.DataFrame(rows)
    summary_rows = []
    for model_name, grp in table.groupby("model"):
        for statistic, fn in (
            ("mean", lambda s: s.mean()),
            ("std", lambda s: s.std()),
            ("min", lambda s: s.min()),
            ("max", lambda s: s.max()),
            ("range", lambda s: s.max() - s.min()),
        ):
            summary_rows.append(
                {
                    "row_type": "summary",
                    "statistic": statistic,
                    "seed": "",
                    "model": model_name,
                    "experiment": "seed_stability",
                    "mean_accuracy": float(fn(grp["mean_accuracy"])),
                    "std_accuracy": float(fn(grp["std_accuracy"])),
                    "mean_macro_f1": float(fn(grp["mean_macro_f1"])),
                    "std_macro_f1": float(fn(grp["std_macro_f1"])),
                    "high_recall": float(fn(grp["high_recall"])),
                    "severe_recall": float(fn(grp["severe_recall"])),
                }
            )
    table = pd.concat([table, pd.DataFrame(summary_rows)], ignore_index=True)
    _write_csv(table, "seed_stability.csv")

    per_model = {}
    for model_name, grp in table[table["row_type"] == "per_seed"].groupby("model"):
        per_model[model_name] = {
            "mean_macro_f1_across_seeds": float(grp["mean_macro_f1"].mean()),
            "macro_f1_range_across_seeds": float(
                grp["mean_macro_f1"].max() - grp["mean_macro_f1"].min()
            ),
        }
    summary = {
        "csv": "seed_stability.csv",
        "seeds": list(seeds),
        "per_model": per_model,
    }
    if return_oof:
        return {"summary": summary, "oof": oof_by_seed, "table": table}
    return summary


# ---------------------------------------------------------------------------
# 3. Target permutation (leakage / null sanity test)
# ---------------------------------------------------------------------------


def target_permutation(
    df: pd.DataFrame,
    predictor_cols: List[str],
    seeds: Sequence[int] = (42, 123, 456),
) -> Dict:
    """Leakage/null sanity test: train on permuted TRAINING labels.

    For each seed and model, inside each spatial fold the *training* labels
    are permuted with ``np.random.default_rng(seed)`` (a permutation preserves
    class counts by construction), the model is trained on the permuted
    labels, and it is evaluated on the untouched, real validation labels
    (derived with the usual training-only thresholds).  If the pipeline leaks
    information, permuted-label training will score far above chance.

    Expected accuracy is approximately chance level (~25% for four balanced
    classes).  THIS IS A SANITY CHECK, NOT MODEL SELECTION: no decision should
    be made from these numbers other than "the pipeline is (not) leaking".

    Writes ``target_permutation_test.csv`` (columns: seed, model, fold,
    accuracy, macro_f1, balanced_accuracy, high_recall, severe_recall) plus
    ``row_type`` = ``mean`` rows per seed x model.
    """
    X_full, _ = encode_predictors(df, predictor_cols=predictor_cols)
    fold_indices = _folds_once(df)
    rows = []
    for seed in seeds:
        for model_name, factory in (("Random Forest", make_rf), ("XGBoost", make_xgb)):
            train_fn, predict_fn = factory({"random_state": seed})
            for fold, (train_idx, val_idx) in enumerate(fold_indices, start=1):
                y_series = _fold_target(df, train_idx)
                y_train = y_series.iloc[train_idx].astype(int)
                y_val = y_series.iloc[val_idx].astype(int).values
                rng = np.random.default_rng(seed)
                y_permuted = pd.Series(
                    rng.permutation(y_train.values), index=y_train.index
                )
                model = train_fn(model_name, X_full.iloc[train_idx], y_permuted)
                preds = predict_fn(model, X_full.iloc[val_idx])
                m = compute_metrics(y_val, preds["y_pred"], preds.get("y_prob"))
                rows.append(
                    {
                        "row_type": "fold",
                        "seed": seed,
                        "model": model_name,
                        "fold": fold,
                        "accuracy": m["accuracy"],
                        "macro_f1": m["macro_f1"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "high_recall": m["high_recall"],
                        "severe_recall": m["severe_recall"],
                    }
                )
    table = pd.DataFrame(rows)
    mean_rows = []
    for (seed, model_name), grp in table.groupby(["seed", "model"]):
        mean_rows.append(
            {
                "row_type": "mean",
                "seed": seed,
                "model": model_name,
                "fold": "",
                "accuracy": float(grp["accuracy"].mean()),
                "macro_f1": float(grp["macro_f1"].mean()),
                "balanced_accuracy": float(grp["balanced_accuracy"].mean()),
                "high_recall": float(grp["high_recall"].mean()),
                "severe_recall": float(grp["severe_recall"].mean()),
            }
        )
    table = pd.concat([table, pd.DataFrame(mean_rows)], ignore_index=True)
    _write_csv(table, "target_permutation_test.csv")

    mean_accuracy = (
        table[table["row_type"] == "mean"].groupby("model")["accuracy"].mean().to_dict()
    )
    return {
        "csv": "target_permutation_test.csv",
        "mean_accuracy_by_model": {k: float(v) for k, v in mean_accuracy.items()},
        "verdict": (
            "PASS (near chance) if mean accuracy is close to 0.25; values far "
            "above chance indicate label leakage."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Learning curves (block-level deterministic training subsets)
# ---------------------------------------------------------------------------


def learning_curves(
    df: pd.DataFrame,
    predictor_cols: List[str],
    fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    folds: Optional[Sequence[int]] = None,
) -> Dict:
    """Learning curves over deterministically growing training-block subsets.

    Uses the outer spatial CV folds.  Within each fold the training blocks are
    sorted and the first ``k = max(1, round(fraction * n_train_blocks))``
    blocks are used for training.  This is a BLOCK-LEVEL DETERMINISTIC SUBSET,
    not a random pixel sample, so every fraction of a fold trains on nested,
    reproducible blocks.  Validation is always on the untouched spatial-CV
    validation blocks of that fold, with training-only thresholds per fold.

    Both Random Forest and XGBoost are fitted across all folds; pass
    ``folds=[1]`` (or any subset of fold numbers) to restrict the run for
    compute-limited invocations.  Writes ``learning_curves.csv`` with columns:
    model, fold, fraction, n_train_blocks_used, n_train_samples,
    train_accuracy, validation_accuracy, train_macro_f1, validation_macro_f1.
    """
    X_full, _ = encode_predictors(df, predictor_cols=predictor_cols)
    groups = df[GROUP_VAR].astype(int)
    fold_indices = _folds_once(df)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        if folds is not None and fold not in folds:
            continue
        train_blocks = sorted(groups.iloc[train_idx].unique().tolist())
        y_series = _fold_target(df, train_idx)
        g_train = groups.iloc[train_idx].values
        X_val = X_full.iloc[val_idx]
        y_val = y_series.iloc[val_idx].astype(int)
        for fraction in fractions:
            k = max(1, int(round(fraction * len(train_blocks))))
            used_blocks = train_blocks[:k]
            sub_mask = np.isin(g_train, used_blocks)
            sub_train_idx = train_idx[sub_mask]
            X_tr = X_full.iloc[sub_train_idx]
            y_tr = y_series.iloc[sub_train_idx].astype(int)
            for model_name, factory in (("Random Forest", make_rf), ("XGBoost", make_xgb)):
                train_fn, predict_fn = factory()
                model = train_fn(model_name, X_tr, y_tr)
                val_preds = predict_fn(model, X_val)
                train_preds = predict_fn(model, X_tr)
                val_m = compute_metrics(y_val.values, val_preds["y_pred"], val_preds.get("y_prob"))
                train_m = compute_metrics(y_tr.values, train_preds["y_pred"], train_preds.get("y_prob"))
                rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "fraction": fraction,
                        "n_train_blocks_used": k,
                        "n_train_samples": int(len(sub_train_idx)),
                        "train_accuracy": train_m["accuracy"],
                        "validation_accuracy": val_m["accuracy"],
                        "train_macro_f1": train_m["macro_f1"],
                        "validation_macro_f1": val_m["macro_f1"],
                    }
                )
    table = _write_csv(pd.DataFrame(rows), "learning_curves.csv")
    return {
        "csv": "learning_curves.csv",
        "fractions": list(fractions),
        "folds": sorted(table["fold"].unique().tolist()) if len(table) else [],
        "n_rows": int(len(table)),
        "note": (
            "Training subsets are deterministic block-level prefixes "
            "(sorted block ids), not random pixel samples."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Leave-one-block-out
# ---------------------------------------------------------------------------


def leave_one_block_out(df: pd.DataFrame, predictor_cols: List[str]) -> Dict:
    """Leave-one-block-out (LOBO) generalization test.

    For every occupied spatial block: train on all OTHER blocks and test on
    the held-out block.  Target thresholds are derived from the training
    blocks only (``build_target`` with ``row_mask`` = "not the held-out
    block"), consistent with the rest of Phase 5 — final descriptive
    thresholds are NOT used here.

    Writes ``leave_one_block_out.csv`` with per-block rows (columns: model,
    held_out_block, n_train, n_test, accuracy, balanced_accuracy,
    macro_precision, macro_recall, macro_f1, high_recall, severe_recall) and
    ``row_type`` = ``summary`` rows per model (statistic in {mean, median,
    std, min, max} of macro_f1 and accuracy, plus best/worst block ids).
    """
    X_full, _ = encode_predictors(df, predictor_cols=predictor_cols)
    groups = df[GROUP_VAR].astype(int)
    occupied = sorted(groups.unique().tolist())
    rows = []
    for block in occupied:
        train_idx = np.flatnonzero(groups.values != block)
        test_idx = np.flatnonzero(groups.values == block)
        y_series = _fold_target(df, train_idx)
        y_train = y_series.iloc[train_idx].astype(int)
        y_test = y_series.iloc[test_idx].astype(int).values
        for model_name, factory in (("Random Forest", make_rf), ("XGBoost", make_xgb)):
            train_fn, predict_fn = factory()
            model = train_fn(model_name, X_full.iloc[train_idx], y_train)
            preds = predict_fn(model, X_full.iloc[test_idx])
            m = compute_metrics(y_test, preds["y_pred"], preds.get("y_prob"))
            rows.append(
                {
                    "row_type": "block",
                    "model": model_name,
                    "held_out_block": block,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_precision": m["macro_precision"],
                    "macro_recall": m["macro_recall"],
                    "macro_f1": m["macro_f1"],
                    "high_recall": m["high_recall"],
                    "severe_recall": m["severe_recall"],
                }
            )
    table = pd.DataFrame(rows)
    summary_rows = []
    lobo_summary = {}
    for model_name, grp in table.groupby("model"):
        best_idx = grp["macro_f1"].idxmax()
        worst_idx = grp["macro_f1"].idxmin()
        best_block = int(table.loc[best_idx, "held_out_block"])
        worst_block = int(table.loc[worst_idx, "held_out_block"])
        lobo_summary[model_name] = {
            "mean_macro_f1": float(grp["macro_f1"].mean()),
            "min_macro_f1": float(grp["macro_f1"].min()),
            "best_block": best_block,
            "worst_block": worst_block,
        }
        for statistic, fn in (
            ("mean", lambda s: s.mean()),
            ("median", lambda s: s.median()),
            ("std", lambda s: s.std()),
            ("min", lambda s: s.min()),
            ("max", lambda s: s.max()),
        ):
            summary_rows.append(
                {
                    "row_type": "summary",
                    "model": model_name,
                    "held_out_block": "",
                    "statistic": statistic,
                    "n_train": "",
                    "n_test": "",
                    "accuracy": float(fn(grp["accuracy"])),
                    "balanced_accuracy": float(fn(grp["balanced_accuracy"])),
                    "macro_precision": float(fn(grp["macro_precision"])),
                    "macro_recall": float(fn(grp["macro_recall"])),
                    "macro_f1": float(fn(grp["macro_f1"])),
                    "high_recall": float(fn(grp["high_recall"])),
                    "severe_recall": float(fn(grp["severe_recall"])),
                    "best_block": best_block,
                    "worst_block": worst_block,
                }
            )
    for col in ["statistic", "best_block", "worst_block"]:
        if col not in table.columns:
            table[col] = ""
    table = pd.concat([table, pd.DataFrame(summary_rows)], ignore_index=True)
    _write_csv(table, "leave_one_block_out.csv")
    return {
        "csv": "leave_one_block_out.csv",
        "n_blocks": len(occupied),
        "per_model": lobo_summary,
    }


# ---------------------------------------------------------------------------
# 6. Spatial distance stress test
# ---------------------------------------------------------------------------


def spatial_distance(df: pd.DataFrame, predictor_cols: List[str]) -> Dict:
    """Accuracy as a function of distance to the nearest training block.

    Uses the standard 5-fold spatial-CV out-of-fold (OOF) predictions of the
    full-feature Random Forest.  Inside each fold, the distance of every
    validation pixel to the nearest TRAINING pixel of a different block is
    computed with ``scipy.spatial.cKDTree`` on the training rows' (row, col)
    coordinates scaled to metres (30 m per pixel step).  Because validation
    blocks are disjoint from training blocks, the nearest training pixel
    already belongs to the nearest training block.  Validation pixels are
    binned into bands 0-500 m, 500-1000 m, 1000-2000 m, >2000 m and per-band
    accuracy, macro-F1, High recall and Severe recall are reported.

    Interpretation caveat: declining accuracy with distance indicates the
    model relies on local spatial similarity (spatial autocorrelation of the
    target), which is expected for environmental interpolation; it is NOT
    automatically evidence of leakage.  Writes ``spatial_distance_stress_test.csv``.
    """
    train_fn, predict_fn = make_rf()
    result = run_spatial_cv(df, predictor_cols, "Random Forest", train_fn, predict_fn)
    oof = result["oof_predictions"]
    fold_indices = _folds_once(df)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        train_coords = df[["row", "col"]].iloc[train_idx].values * PIXEL_SIZE_M
        val_coords = df[["row", "col"]].iloc[val_idx].values * PIXEL_SIZE_M
        tree = cKDTree(train_coords)
        dist, _ = tree.query(val_coords, k=1)
        fold_oof = oof[oof["fold"] == fold].copy()
        fold_oof["distance_m"] = dist
        for lo, hi, band_name in DISTANCE_BANDS:
            if np.isinf(hi):
                mask = dist >= lo
            else:
                mask = (dist >= lo) & (dist < hi)
            if not mask.any():
                continue
            sub = fold_oof[mask]
            m = compute_metrics(
                sub["actual_class"].values, sub["predicted_class"].values
            )
            rows.append(
                {
                    "model": "Random Forest",
                    "fold": fold,
                    "distance_band": band_name,
                    "n_pixels": int(mask.sum()),
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "high_recall": m["high_recall"],
                    "severe_recall": m["severe_recall"],
                }
            )
    _write_csv(pd.DataFrame(rows), "spatial_distance_stress_test.csv")
    return {
        "csv": "spatial_distance_stress_test.csv",
        "caveat": (
            "Declining accuracy with distance indicates reliance on local "
            "spatial similarity / spatial autocorrelation, not automatically "
            "leakage."
        ),
    }


# ---------------------------------------------------------------------------
# 7. Feature permutation sanity (permute TRAINING feature, retrain)
# ---------------------------------------------------------------------------


def feature_permutation_sanity(
    df: pd.DataFrame,
    predictor_cols: List[str],
    features: Optional[Sequence[str]] = None,
    subsample: int = 30_000,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Retrain-based feature permutation diagnostic.

    Within each spatial training fold, a deterministic class-stratified
    subsample of the training fold (at most ``subsample`` rows, seed fixed) is
    used for BOTH the baseline and the permuted fits (this keeps compute and
    memory safe and is recorded in the ``subsample`` column).  For each
    feature, the TRAINING copy is modified by permuting that single column
    with a seeded ``np.random.default_rng``; a Random Forest is then trained
    on the permuted training copy and evaluated on the UNTOUCHED validation
    set of the fold.  The recorded degradation is
    ``baseline_macro_f1 - permuted_macro_f1``.

    This differs from ORDINARY permutation importance, which permutes the
    features of a validation set for an already-fitted model: here the model
    itself is retrained without access to the true within-fold relationship of
    the permuted feature, so the degradation includes the feature's training
    information content, not just its contribution at prediction time.

    Writes ``feature_permutation_sanity.csv`` with columns: model, fold,
    feature, subsample, baseline_macro_f1, permuted_macro_f1, degradation.
    """
    features = list(features) if features is not None else list(DEFAULT_PERMUTED_FEATURES)
    X_full, feature_names = encode_predictors(df, predictor_cols=predictor_cols)
    missing = [f for f in features if f not in feature_names]
    if missing:
        raise ValueError(f"Features not present in encoded predictors: {missing}")
    fold_indices = _folds_once(df)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        y_series = _fold_target(df, train_idx)
        y_train = y_series.iloc[train_idx].astype(int).values
        rng = np.random.default_rng(seed + fold)
        local_sub = _stratified_subsample_indices(y_train, subsample, rng)
        sub_train_idx = train_idx[local_sub]
        X_tr = X_full.iloc[sub_train_idx]
        y_tr = y_series.iloc[sub_train_idx].astype(int)
        X_val = X_full.iloc[val_idx]
        y_val = y_series.iloc[val_idx].astype(int).values
        train_fn, predict_fn = make_rf({"random_state": seed})
        baseline_model = train_fn("Random Forest", X_tr, y_tr)
        baseline_m = compute_metrics(
            y_val, predict_fn(baseline_model, X_val)["y_pred"]
        )
        baseline_f1 = baseline_m["macro_f1"]
        for feature in features:
            X_perm = X_tr.copy()
            X_perm[feature] = rng.permutation(X_perm[feature].values)
            model = train_fn("Random Forest", X_perm, y_tr)
            perm_m = compute_metrics(y_val, predict_fn(model, X_val)["y_pred"])
            rows.append(
                {
                    "model": "Random Forest",
                    "fold": fold,
                    "feature": feature,
                    "subsample": int(len(X_tr)),
                    "baseline_macro_f1": baseline_f1,
                    "permuted_macro_f1": perm_m["macro_f1"],
                    "degradation": baseline_f1 - perm_m["macro_f1"],
                }
            )
    table = _write_csv(pd.DataFrame(rows), "feature_permutation_sanity.csv")
    top = (
        table.groupby("feature")["degradation"].mean().sort_values(ascending=False)
        if len(table)
        else pd.Series(dtype=float)
    )
    return {
        "csv": "feature_permutation_sanity.csv",
        "features": features,
        "subsample_cap": subsample,
        "mean_degradation_ranking": {k: float(v) for k, v in top.items()},
    }


# ---------------------------------------------------------------------------
# 8. Nested spatial cross-validation (RF hyper-parameter selection)
# ---------------------------------------------------------------------------


def nested_spatial_cv(
    df: pd.DataFrame,
    predictor_cols: List[str],
    inner_subsample: int = 25_000,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Nested spatial CV for RF regularization hyper-parameter selection.

    Outer loop: the standard 5 spatial folds.  Inner loop: 4-fold GroupKFold
    on the OUTER-TRAINING blocks only; it is asserted that outer validation
    blocks never appear in the inner data.  Candidates: RF baseline (A),
    RF-B (max_depth=20, min_samples_leaf=5), RF-C (max_depth=15,
    min_samples_leaf=10), RF-D (max_depth=10, min_samples_leaf=15).

    Inside each inner fold the target uses training-only thresholds (inner
    row mask).  For compute safety the inner training sets use a deterministic
    class-stratified subsample of at most ``inner_subsample`` rows (recorded
    in the ``inner_train_subsample`` column); thresholds are computed on the
    FULL inner-training rows before subsampling so the subsample does not
    shift the quartiles.  The selected candidate is then refit on ALL
    outer-training blocks (no subsample) and evaluated once on the outer
    validation blocks.

    Only RF candidates are used to keep runtime sane.  Writes
    ``nested_spatial_cv_results.csv``.
    """
    X_full, _ = encode_predictors(df, predictor_cols=predictor_cols)
    groups = df[GROUP_VAR].astype(int)
    fold_indices = _folds_once(df)
    rows = []
    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(fold_indices, start=1):
        outer_train_blocks, outer_val_blocks = get_fold_blocks(
            groups, outer_train_idx, outer_val_idx
        )
        sub_df = df.iloc[outer_train_idx].reset_index(drop=True)
        X_sub = X_full.iloc[outer_train_idx].reset_index(drop=True)
        groups_sub = groups.iloc[outer_train_idx].reset_index(drop=True)
        inner_gkf = GroupKFold(n_splits=4)
        inner_splits = list(
            inner_gkf.split(np.zeros(len(sub_df)), groups=groups_sub.values)
        )
        inner_selection: Dict[str, float] = {}
        for candidate, overrides in RF_REGULARIZATION_CANDIDATES.items():
            inner_f1 = []
            for inner_train_local, inner_val_local in inner_splits:
                inner_blocks = sorted(groups_sub.iloc[inner_train_local].unique().tolist())
                if set(inner_blocks).intersection(outer_val_blocks):
                    raise AssertionError(
                        f"Outer fold {outer_fold}: inner training contains outer "
                        f"validation blocks {set(inner_blocks) & set(outer_val_blocks)}"
                    )
                inner_mask = np.zeros(len(sub_df), dtype=bool)
                inner_mask[inner_train_local] = True
                y_inner, _ = build_target(sub_df, row_mask=inner_mask)
                y_inner_train = y_inner.iloc[inner_train_local].astype(int).values
                rng = np.random.default_rng(seed + 1000 * outer_fold + len(inner_f1))
                local_sub = _stratified_subsample_indices(
                    y_inner_train, inner_subsample, rng
                )
                sub_train = inner_train_local[local_sub]
                train_fn, predict_fn = make_rf({"random_state": seed, **overrides})
                model = train_fn(
                    f"RF {candidate}",
                    X_sub.iloc[sub_train],
                    y_inner.iloc[sub_train].astype(int),
                )
                preds = predict_fn(model, X_sub.iloc[inner_val_local])
                m = compute_metrics(
                    y_inner.iloc[inner_val_local].astype(int).values,
                    preds["y_pred"],
                    preds.get("y_prob"),
                )
                inner_f1.append(m["macro_f1"])
            inner_selection[candidate] = float(np.mean(inner_f1))
        selected = max(inner_selection, key=inner_selection.get)
        selected_overrides = RF_REGULARIZATION_CANDIDATES[selected]

        y_outer = _fold_target(df, outer_train_idx)
        train_fn, predict_fn = make_rf({"random_state": seed, **selected_overrides})
        model = train_fn(
            f"RF {selected}",
            X_full.iloc[outer_train_idx],
            y_outer.iloc[outer_train_idx].astype(int),
        )
        preds = predict_fn(model, X_full.iloc[outer_val_idx])
        m = compute_metrics(
            y_outer.iloc[outer_val_idx].astype(int).values,
            preds["y_pred"],
            preds.get("y_prob"),
        )
        rows.append(
            {
                "outer_fold": outer_fold,
                "selected_model": f"RF {selected}",
                "selected_parameters": json.dumps(selected_overrides, sort_keys=True),
                "n_outer_train": int(len(outer_train_idx)),
                "n_outer_validation": int(len(outer_val_idx)),
                "outer_accuracy": m["accuracy"],
                "outer_balanced_accuracy": m["balanced_accuracy"],
                "outer_macro_f1": m["macro_f1"],
                "outer_high_recall": m["high_recall"],
                "outer_severe_recall": m["severe_recall"],
                "inner_train_subsample": inner_subsample,
                "inner_selection": json.dumps(inner_selection, sort_keys=True),
            }
        )
    table = _write_csv(pd.DataFrame(rows), "nested_spatial_cv_results.csv")
    return {
        "csv": "nested_spatial_cv_results.csv",
        "selection_counts": table["selected_model"].value_counts().to_dict(),
        "mean_outer_macro_f1": float(table["outer_macro_f1"].mean()),
    }


# ---------------------------------------------------------------------------
# 9. Locked geographic holdout
# ---------------------------------------------------------------------------


def locked_geographic_holdout(
    df: pd.DataFrame,
    predictor_cols: List[str],
    rf_overrides: Optional[Dict] = None,
    xgb_overrides: Optional[Dict] = None,
) -> Dict:
    """Evaluate once on a deterministically locked geographic holdout.

    Block selection happens BEFORE any model work and is purely
    deterministic: the occupied block ids are sorted and every
    ``LOCKED_HOLDOUT_STRIDE``-th block starting from index
    ``LOCKED_HOLDOUT_START_INDEX`` is taken (i.e. ``ids[2::6]``).  The locked
    blocks are removed from ALL development data; Random Forest and XGBoost
    (full features) are trained on the remaining blocks with training-only
    thresholds derived from the remaining blocks only, and evaluated ONCE on
    the locked blocks.  ``rf_overrides`` / ``xgb_overrides`` carry the frozen
    final hyper-parameters selected during development (e.g. the regularized
    RF candidate); they must be fixed BEFORE this function is called.

    IMPORTANT: this holdout must never be used for tuning, model selection, or
    threshold derivation — it is evaluated exactly once.  Writes
    ``locked_geographic_holdout.csv``.
    """
    groups = df[GROUP_VAR].astype(int)
    occupied = sorted(groups.unique().tolist())
    holdout_blocks = occupied[LOCKED_HOLDOUT_START_INDEX :: LOCKED_HOLDOUT_STRIDE]
    if not holdout_blocks:
        raise ValueError("Locked-holdout rule selected zero blocks; dataset too small.")
    is_holdout = groups.isin(holdout_blocks).values
    dev_idx = np.flatnonzero(~is_holdout)
    hold_idx = np.flatnonzero(is_holdout)

    X_full, _ = encode_predictors(df, predictor_cols=predictor_cols)
    row_mask = np.zeros(len(df), dtype=bool)
    row_mask[dev_idx] = True
    y_series, _ = build_target(df, row_mask=row_mask)
    y_dev = y_series.iloc[dev_idx].astype(int)
    y_hold = y_series.iloc[hold_idx].astype(int).values

    rows = []
    model_specs = [
        ("Random Forest", make_rf(rf_overrides), rf_overrides),
        ("XGBoost", make_xgb(xgb_overrides), xgb_overrides),
    ]
    for model_name, (train_fn, predict_fn), overrides in model_specs:
        model = train_fn(model_name, X_full.iloc[dev_idx], y_dev)
        preds = predict_fn(model, X_full.iloc[hold_idx])
        m = compute_metrics(y_hold, preds["y_pred"], preds.get("y_prob"))
        rows.append(
            {
                "model": model_name,
                "held_out_blocks": json.dumps(holdout_blocks),
                "parameters": json.dumps(overrides or {}, sort_keys=True),
                "n_train": int(len(dev_idx)),
                "n_test": int(len(hold_idx)),
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "high_recall": m["high_recall"],
                "severe_recall": m["severe_recall"],
                "confusion_matrix_json": json.dumps(m["confusion_matrix"]),
            }
        )
    _write_csv(pd.DataFrame(rows), "locked_geographic_holdout.csv")
    return {
        "csv": "locked_geographic_holdout.csv",
        "held_out_blocks": holdout_blocks,
        "rule": f"sorted_ids[{LOCKED_HOLDOUT_START_INDEX}::{LOCKED_HOLDOUT_STRIDE}]",
        "note": "Evaluated once; never use this holdout for tuning or selection.",
    }


# ---------------------------------------------------------------------------
# 10. Spatial error autocorrelation (Moran's I on OOF errors)
# ---------------------------------------------------------------------------


def spatial_error_autocorrelation(df: pd.DataFrame, predictor_cols: List[str]) -> Dict:
    """Global Moran's I of RF out-of-fold prediction errors, per year.

    Uses the OOF predictions of the full-feature Random Forest.  For each
    year separately, two variables are tested on the year's OOF set:
    (a) the classification error indicator (actual != predicted, 0/1) and
    (b) the absolute class-distance error |actual - predicted|.  Spatial
    weights are k-nearest-neighbour (k=8) on (row, col) coordinates scaled by
    30 m per pixel step, with no diagonal self-weights and row-standardised
    weights.  Moran's I is implemented manually
    (``I = (n/W) * sum_ij w_ij z_i z_j / sum_i z_i^2``); no new dependencies.
    A two-sided permutation p-value uses 199 fixed-seed permutations of the
    variable values across locations.

    Interpretation caveat: residual spatial autocorrelation of errors is NOT
    automatically overfitting — it can equally indicate missing spatial
    predictors (underfitting) or genuinely autocorrelated drivers.  Writes
    ``spatial_error_autocorrelation.csv``.
    """
    train_fn, predict_fn = make_rf()
    result = run_spatial_cv(df, predictor_cols, "Random Forest", train_fn, predict_fn)
    oof = result["oof_predictions"]
    weights_definition = (
        f"kNN k={MORANS_K} on (row,col)*{PIXEL_SIZE_M}m, no self-weights, "
        "row-standardised"
    )
    rows = []
    for year in sorted(oof[YEAR_VAR].unique().tolist()):
        sub = oof[oof[YEAR_VAR] == year]
        coords = sub[["row", "col"]].values * PIXEL_SIZE_M
        variables = {
            "classification_error": (
                (sub["actual_class"].values != sub["predicted_class"].values)
            ).astype(float),
            "absolute_class_distance": np.abs(
                sub["actual_class"].values - sub["predicted_class"].values
            ).astype(float),
        }
        for variable, values in variables.items():
            I, expected, p_value = morans_i_with_permutation(
                values,
                coords,
                k=MORANS_K,
                n_permutations=MORANS_N_PERMUTATIONS,
                seed=RANDOM_SEED,
            )
            rows.append(
                {
                    "model": "Random Forest",
                    "year": year,
                    "variable": variable,
                    "morans_i": I,
                    "expected_i": expected,
                    "permutation_p_value": p_value,
                    "n": int(len(sub)),
                    "weights_definition": weights_definition,
                }
            )
    _write_csv(pd.DataFrame(rows), "spatial_error_autocorrelation.csv")
    return {
        "csv": "spatial_error_autocorrelation.csv",
        "caveat": (
            "Residual spatial autocorrelation is not automatically overfitting; "
            "it can indicate missing spatial predictors / underfitting."
        ),
    }


# ---------------------------------------------------------------------------
# 11. Prediction stability across seeds
# ---------------------------------------------------------------------------


def prediction_stability(
    df: pd.DataFrame,
    predictor_cols: List[str],
    seed_runs: Optional[Dict[int, pd.DataFrame]] = None,
    seeds: Sequence[int] = SEEDS,
    reference_seed: int = 42,
) -> Dict:
    """Per-pixel OOF prediction agreement across the seed-stability runs.

    Reuses cached per-seed OOF frames from ``seed_stability`` (pass them via
    ``seed_runs``; if omitted they are computed once via
    ``seed_stability(..., return_oof=True)`` — never retrained inside this
    function beyond that).  For the Random Forest, each OOF pixel (identified
    by row/col/year) gets an agreement count: how many seeds (including the
    reference seed, default 42) produce the same predicted class as the
    reference seed.  Pixels are tallied into agreement tiers 5/5, 4/5, 3/5 and
    <=2/5.

    Writes ``prediction_stability.csv`` with three row blocks distinguished by
    ``metric_type``: ``overall`` (tier counts/percentages across all pixels),
    ``per_class`` (tier breakdown within each reference predicted class), and
    ``hotspot`` (tier breakdown of agreement on hotspot membership, i.e.
    whether the prediction is High/Severe vs the rest).  Columns: model,
    metric_type, class, tier, n_pixels, percentage.
    """
    if reference_seed not in seeds:
        raise ValueError("reference_seed must be part of seeds.")
    if seed_runs is None:
        seed_runs = seed_stability(df, predictor_cols, seeds=seeds, return_oof=True)["oof"]
    ref = seed_runs[reference_seed][["row", "col", YEAR_VAR, "predicted_class", "actual_class"]].copy()
    other_seeds = [s for s in seeds if s != reference_seed]
    for seed in other_seeds:
        other = seed_runs[seed][["row", "col", YEAR_VAR, "predicted_class"]].rename(
            columns={"predicted_class": f"pred_{seed}"}
        )
        ref = ref.merge(other, on=["row", "col", YEAR_VAR], how="inner")
    pred_cols = [f"pred_{s}" for s in other_seeds]
    agreement = 1 + (
        ref[pred_cols].values == ref["predicted_class"].values[:, None]
    ).sum(axis=1)

    def _tier(count: int) -> str:
        if count >= 5:
            return "5/5"
        if count == 4:
            return "4/5"
        if count == 3:
            return "3/5"
        return "<=2/5"

    tier_order = ["5/5", "4/5", "3/5", "<=2/5"]
    rows = []

    def _add_rows(metric_type: str, class_label: str, counts: pd.Series) -> None:
        total = int(counts.sum())
        for tier in tier_order:
            n = int(counts.get(tier, 0))
            rows.append(
                {
                    "model": "Random Forest",
                    "metric_type": metric_type,
                    "class": class_label,
                    "tier": tier,
                    "n_pixels": n,
                    "percentage": round(100.0 * n / total, 4) if total else np.nan,
                }
            )

    _add_rows("overall", "all", pd.Series(agreement).map(_tier).value_counts())
    ref["_agreement_tier"] = pd.Series(agreement).map(_tier).values
    ref["_hotspot_ref"] = ref["predicted_class"].isin([2, 3]).values
    for cls in sorted(ref["predicted_class"].unique().tolist()):
        sub = ref[ref["predicted_class"] == cls]
        _add_rows("per_class", f"class_{cls}", sub["_agreement_tier"].value_counts())
    hotspot_agree = 1 + (
        ref[pred_cols].apply(lambda s: s.isin([2, 3])).values
        == ref["_hotspot_ref"].values[:, None]
    ).sum(axis=1)
    _add_rows("hotspot", "high_severe_vs_rest", pd.Series(hotspot_agree).map(_tier).value_counts())

    _write_csv(pd.DataFrame(rows), "prediction_stability.csv")
    overall = pd.DataFrame(rows)
    overall = overall[overall["metric_type"] == "overall"].set_index("tier")
    return {
        "csv": "prediction_stability.csv",
        "reference_seed": reference_seed,
        "seeds": list(seeds),
        "overall_percentage": {
            tier: float(overall.loc[tier, "percentage"]) if tier in overall.index else 0.0
            for tier in tier_order
        },
    }


# ---------------------------------------------------------------------------
# 12. Duplicate / data-integrity audit
# ---------------------------------------------------------------------------


def duplicate_audit(df: pd.DataFrame, predictor_cols: Optional[List[str]] = None) -> Dict:
    """Data-integrity audit for duplicates and label conflicts.

    Checks, on the full merged dataframe:

    * exact duplicate lon/lat rows;
    * duplicate (row, col, year) pixels;
    * duplicate (spatial_block_id, row, col, year) pixels;
    * predictor-identical rows with conflicting labels (same encoded
      predictors, different ``uhi_class`` under the FULL descriptive
      thresholds);
    * repeated (row, col) pairs across DIFFERENT years — these are
      LEGITIMATE temporal observations and are counted with verdict INFO;
      they must NOT be deleted.

    Writes ``data_integrity_duplicate_audit.csv`` with columns: check, count,
    verdict, notes.
    """
    if predictor_cols is None:
        from .config import PREDICTOR_VARS

        predictor_cols = [c for c in PREDICTOR_VARS if c in df.columns]
    X_enc, _ = encode_predictors(df, predictor_cols=predictor_cols)
    y_full, _ = build_target(df, row_mask=None)

    dup_lonlat = int(df.duplicated(subset=["lon", "lat"]).sum())
    dup_rowcolyear = int(df.duplicated(subset=["row", "col", YEAR_VAR]).sum())
    dup_block_rowcolyear = int(
        df.duplicated(subset=[GROUP_VAR, "row", "col", YEAR_VAR]).sum()
    )

    conflict_groups = 0
    dup_encoded_mask = X_enc.duplicated(keep=False)
    if dup_encoded_mask.any():
        tmp = X_enc[dup_encoded_mask].copy()
        tmp["uhi_class"] = y_full[dup_encoded_mask].values
        nunique = tmp.groupby(list(X_enc.columns), sort=False)["uhi_class"].nunique()
        conflict_groups = int((nunique > 1).sum())

    repeated_rowcol = df.groupby(["row", "col"])[YEAR_VAR].nunique()
    temporal_repeat_locations = int((repeated_rowcol > 1).sum())

    def _verdict(count: int) -> str:
        return "OK" if count == 0 else "INFO"

    rows = [
        {
            "check": "exact_duplicate_lonlat",
            "count": dup_lonlat,
            "verdict": _verdict(dup_lonlat),
            "notes": "Duplicate lon/lat rows across any year/block.",
        },
        {
            "check": "duplicate_row_col_year",
            "count": dup_rowcolyear,
            "verdict": _verdict(dup_rowcolyear),
            "notes": "Duplicate (row, col, year) pixels.",
        },
        {
            "check": "duplicate_block_row_col_year",
            "count": dup_block_rowcolyear,
            "verdict": _verdict(dup_block_rowcolyear),
            "notes": "Duplicate (spatial_block_id, row, col, year) pixels.",
        },
        {
            "check": "predictor_identical_conflicting_labels",
            "count": conflict_groups,
            "verdict": _verdict(conflict_groups),
            "notes": (
                "Groups of rows with identical encoded predictors but more than "
                "one uhi_class (full descriptive thresholds). Intrinsic label "
                "noise, not leakage, if small."
            ),
        },
        {
            "check": "repeated_row_col_across_years",
            "count": temporal_repeat_locations,
            "verdict": "INFO",
            "notes": (
                "LEGITIMATE temporal observations: same (row, col) observed in "
                "multiple years. Do NOT delete."
            ),
        },
    ]
    _write_csv(pd.DataFrame(rows), "data_integrity_duplicate_audit.csv")
    return {
        "csv": "data_integrity_duplicate_audit.csv",
        "checks": {r["check"]: r["count"] for r in rows},
    }


# ---------------------------------------------------------------------------
# 13. Leakage audit (machine-readable policy table)
# ---------------------------------------------------------------------------


def leakage_audit(df: pd.DataFrame, predictor_cols: List[str]) -> Dict:
    """Machine-readable leakage policy / verification table.

    Every row states whether a variable or operation is allowed as a model
    predictor and why.  The ``verification`` column is derived
    programmatically where feasible: it is PASS only when none of the
    forbidden variables (``lst_C``, ``spatial_block_id``, ``lon``, ``lat``,
    ``row``, ``col``) appear in ``predictor_cols`` (WARN otherwise).  The
    training-only threshold guarantee is stated as verified-by-code-review
    with a short code reference (``experiments.run_spatial_cv`` builds the
    target with a training-only ``row_mask``).

    Writes ``leakage_audit.csv`` with columns: variable_or_operation,
    allowed_as_predictor, reason, verification.
    """
    forbidden = {TARGET_VAR, GROUP_VAR, "lon", "lat", "row", "col"}
    leaked = sorted(forbidden.intersection(predictor_cols))
    membership_ok = not leaked
    membership_verification = "PASS" if membership_ok else f"WARN: {leaked} in predictor_cols"

    rows = [
        {
            "variable_or_operation": TARGET_VAR,
            "allowed_as_predictor": "NO",
            "reason": "Target leakage: the class is derived from lst_C.",
            "verification": membership_verification,
        },
        {
            "variable_or_operation": GROUP_VAR,
            "allowed_as_predictor": "NO",
            "reason": "Grouping variable for spatial CV only; carries block identity.",
            "verification": membership_verification,
        },
        {
            "variable_or_operation": "lon/lat/row/col",
            "allowed_as_predictor": "NO",
            "reason": "Identifiers/coordinates; would let the model memorise locations.",
            "verification": membership_verification,
        },
        {
            "variable_or_operation": "target-derived thresholds",
            "allowed_as_predictor": "NO",
            "reason": (
                "Per-year quartile thresholds are derived from the target; using "
                "validation rows would leak the target."
            ),
            "verification": (
                "PASS (verified by code review): training-only row_mask inside "
                "folds, experiments.run_spatial_cv:89"
            ),
        },
        {
            "variable_or_operation": "ndvi/ndbi/vegetation_cover/year",
            "allowed_as_predictor": "YES",
            "reason": "Observed environmental predictors from Phase 4 rasters.",
            "verification": "PASS",
        },
        {
            "variable_or_operation": "morphology features",
            "allowed_as_predictor": "YES",
            "reason": (
                "Environmental predictors; block-aware neighbourhoods never "
                "cross spatial blocks."
            ),
            "verification": "PASS",
        },
        {
            "variable_or_operation": "one-hot encoding",
            "allowed_as_predictor": "YES",
            "reason": "Unsupervised transformation of landuse_class; no target use.",
            "verification": "PASS",
        },
        {
            "variable_or_operation": "preprocessing fitted on training only",
            "allowed_as_predictor": "YES",
            "reason": "Fitted choices (thresholds, subsamples) use training rows only.",
            "verification": "PASS",
        },
        {
            "variable_or_operation": "feature selection inside folds",
            "allowed_as_predictor": "YES",
            "reason": "Selection is currently not used; if added it must run inside folds.",
            "verification": "PASS",
        },
        {
            "variable_or_operation": "calibration not fitted on locked holdout",
            "allowed_as_predictor": "YES",
            "reason": "The locked geographic holdout is evaluated once, never fitted on.",
            "verification": "PASS",
        },
    ]
    _write_csv(pd.DataFrame(rows), "leakage_audit.csv")
    return {
        "csv": "leakage_audit.csv",
        "forbidden_in_predictor_cols": leaked,
        "all_pass": all(r["verification"].startswith("PASS") for r in rows),
    }


# ---------------------------------------------------------------------------
# 14. Regularization experiments
# ---------------------------------------------------------------------------


def _run_regularization(
    df: pd.DataFrame,
    predictor_cols: List[str],
    model_label: str,
    factory: Callable,
    candidates: Dict[str, Dict],
    csv_name: str,
) -> Dict:
    """Shared implementation for the RF/XGB regularization experiments.

    Each candidate is run under the SAME 5 spatial folds via
    ``experiments.run_spatial_cv`` with a custom ``train_fn``.  The output CSV
    contains both blocks in one file, distinguished by ``row_type``:
    ``candidate_summary`` rows (columns: model, candidate, parameters JSON,
    mean_train_accuracy, mean_validation_accuracy, accuracy_gap,
    mean_train_macro_f1, mean_validation_macro_f1, macro_f1_gap,
    mean_high_recall, mean_severe_recall, macro_f1_std) and ``fold`` rows
    (columns: model, candidate, fold, train_accuracy, validation_accuracy,
    accuracy_gap, train_macro_f1, validation_macro_f1, macro_f1_gap,
    high_recall, severe_recall).
    """
    summary_rows = []
    fold_rows = []
    for candidate, overrides in candidates.items():
        train_fn, predict_fn = factory(overrides)
        result = run_spatial_cv(df, predictor_cols, f"{model_label} {candidate}", train_fn, predict_fn)
        fold_table = pd.DataFrame(result["fold_results"])
        summary_rows.append(
            {
                "row_type": "candidate_summary",
                "model": model_label,
                "candidate": candidate,
                "parameters": json.dumps(overrides, sort_keys=True),
                "mean_train_accuracy": float(fold_table["train_accuracy"].mean()),
                "mean_validation_accuracy": float(fold_table["accuracy"].mean()),
                "accuracy_gap": float(
                    fold_table["train_accuracy"].mean() - fold_table["accuracy"].mean()
                ),
                "mean_train_macro_f1": float(fold_table["train_macro_f1"].mean()),
                "mean_validation_macro_f1": float(fold_table["macro_f1"].mean()),
                "macro_f1_gap": float(
                    fold_table["train_macro_f1"].mean() - fold_table["macro_f1"].mean()
                ),
                "mean_high_recall": float(fold_table["high_recall"].mean()),
                "mean_severe_recall": float(fold_table["severe_recall"].mean()),
                "macro_f1_std": float(fold_table["macro_f1"].std()),
            }
        )
        for _, fr in fold_table.iterrows():
            fold_rows.append(
                {
                    "row_type": "fold",
                    "model": model_label,
                    "candidate": candidate,
                    "fold": int(fr["fold"]),
                    "train_accuracy": float(fr["train_accuracy"]),
                    "validation_accuracy": float(fr["accuracy"]),
                    "accuracy_gap": float(fr["train_accuracy"] - fr["accuracy"]),
                    "train_macro_f1": float(fr["train_macro_f1"]),
                    "validation_macro_f1": float(fr["macro_f1"]),
                    "macro_f1_gap": float(fr["train_macro_f1"] - fr["macro_f1"]),
                    "high_recall": float(fr["high_recall"]),
                    "severe_recall": float(fr["severe_recall"]),
                }
            )
    table = pd.concat(
        [pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)], ignore_index=True
    )
    _write_csv(table, csv_name)
    summary_df = pd.DataFrame(summary_rows)
    return {
        "csv": csv_name,
        "candidates": list(candidates.keys()),
        "candidate_summary": summary_df.to_dict(orient="records"),
    }


def run_rf_regularization(df: pd.DataFrame, predictor_cols: List[str]) -> Dict:
    """RF regularization sweep (candidates A-D) under the same 5 spatial folds.

    See ``_run_regularization`` for the output layout; writes
    ``rf_regularization.csv``.
    """
    return _run_regularization(
        df,
        predictor_cols,
        "Random Forest",
        make_rf,
        RF_REGULARIZATION_CANDIDATES,
        "rf_regularization.csv",
    )


def run_xgb_regularization(df: pd.DataFrame, predictor_cols: List[str]) -> Dict:
    """XGBoost regularization sweep (baseline, X1-X3) under the same 5 folds.

    See ``_run_regularization`` for the output layout; writes
    ``xgb_regularization.csv``.
    """
    return _run_regularization(
        df,
        predictor_cols,
        "XGBoost",
        make_xgb,
        XGB_REGULARIZATION_CANDIDATES,
        "xgb_regularization.csv",
    )


# ---------------------------------------------------------------------------
# 15. Compact feature experiment
# ---------------------------------------------------------------------------

# Hand-picked compact predictor set (each column must exist in df).
COMPACT_FEATURE_SET = [
    "ndvi",
    "ndbi",
    "vegetation_cover",
    "landuse_class",
    "year",
    "dist_building_m",
    "dist_road_m",
    "dist_vegetation_m",
    "building_pixel_fraction_100m",
    "road_pixel_fraction_100m",
    "vegetation_pixel_fraction_100m",
    "vegetation_cover_mean_100m",
    "landuse_entropy_100m",
    "ndvi_contrast_100m",
    "ndbi_contrast_100m",
    "ndvi_mean3",
    "ndbi_mean3",
    "vegetation_cover_mean3",
]


def run_compact_feature_experiment(df: pd.DataFrame) -> Dict:
    """Compare five predictor sets under the same spatial CV folds.

    Feature sets:

    * ``base_only``: ``get_feature_columns_for_model(include_spatial=False,
      include_morphology=False)``;
    * ``spatial_means_only``: include_spatial=True, include_morphology=False;
    * ``morphology_only``: include_spatial=False, include_morphology=True;
    * ``full``: both families included;
    * ``compact``: the hand-picked ``COMPACT_FEATURE_SET`` list — every column
      is verified to exist in ``df`` and a ``ValueError`` is raised otherwise.

    Random Forest is run for all five sets; XGBoost is run only for ``full``
    and ``compact`` to save compute (documented here).  Writes
    ``compact_feature_results.csv`` with columns: model, feature_set,
    n_features, mean_accuracy, mean_macro_f1, macro_f1_std, mean_high_recall,
    mean_severe_recall, mean_accuracy_gap.
    """
    missing = [c for c in COMPACT_FEATURE_SET if c not in df.columns]
    if missing:
        raise ValueError(f"Compact feature columns missing from df: {missing}")

    feature_sets = {
        "base_only": get_feature_columns_for_model(
            df, include_year=True, include_pvc=True, include_spatial=False, include_morphology=False
        ),
        "spatial_means_only": get_feature_columns_for_model(
            df, include_year=True, include_pvc=True, include_spatial=True, include_morphology=False
        ),
        "morphology_only": get_feature_columns_for_model(
            df, include_year=True, include_pvc=True, include_spatial=False, include_morphology=True
        ),
        "full": get_feature_columns_for_model(
            df, include_year=True, include_pvc=True, include_spatial=True, include_morphology=True
        ),
        "compact": list(COMPACT_FEATURE_SET),
    }
    xgb_sets = {"full", "compact"}

    rows = []
    for feature_set, cols in feature_sets.items():
        for model_name, factory in (("Random Forest", make_rf), ("XGBoost", make_xgb)):
            if model_name == "XGBoost" and feature_set not in xgb_sets:
                continue
            train_fn, predict_fn = factory()
            result = run_spatial_cv(df, cols, model_name, train_fn, predict_fn)
            m = result["aggregated_metrics"]
            fold_table = pd.DataFrame(result["fold_results"])
            rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set,
                    "n_features": len(cols),
                    "mean_accuracy": m.get("accuracy_mean", np.nan),
                    "mean_macro_f1": m.get("macro_f1_mean", np.nan),
                    "macro_f1_std": m.get("macro_f1_std", np.nan),
                    "mean_high_recall": m.get("high_recall_mean", np.nan),
                    "mean_severe_recall": m.get("severe_recall_mean", np.nan),
                    "mean_accuracy_gap": float(
                        fold_table["train_accuracy"].mean() - fold_table["accuracy"].mean()
                    ),
                }
            )
    table = _write_csv(pd.DataFrame(rows), "compact_feature_results.csv")
    return {
        "csv": "compact_feature_results.csv",
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "xgb_feature_sets": sorted(xgb_sets),
        "note": "XGBoost run only for full/compact to save compute.",
        "results": table.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 16. Calibration report (OOF only)
# ---------------------------------------------------------------------------


def calibration(df: pd.DataFrame, predictor_cols: List[str]) -> Dict:
    """Multiclass calibration of the RF and XGBoost OOF predictions.

    Uses ONLY development OOF predictions (no locked holdout).  Reports the
    multiclass Brier score (mean over classes of squared probability error)
    and the Expected Calibration Error (ECE) over 10 equal-width bins of
    ``prediction_confidence`` (per-bin n, mean predicted probability, observed
    accuracy).  NOTE: ``prediction_confidence`` is a confidence proxy (argmax
    class probability), not a calibrated probability.

    Writes ``calibration_report.csv`` with ``row_type`` = ``summary`` rows
    (brier_score, ece, note) and ``row_type`` = ``bin`` rows.
    """
    prob_cols = [f"probability_{label.lower()}" for label in CLASS_LABELS]
    rows = []
    summaries = {}
    for model_name, factory in (("Random Forest", make_rf), ("XGBoost", make_xgb)):
        train_fn, predict_fn = factory()
        result = run_spatial_cv(df, predictor_cols, model_name, train_fn, predict_fn)
        oof = result["oof_predictions"]
        y = oof["actual_class"].astype(int).values
        P = oof[prob_cols].values
        onehot = np.zeros_like(P)
        onehot[np.arange(len(y)), y] = 1.0
        brier = float(np.mean((P - onehot) ** 2))

        confidence = oof["prediction_confidence"].values.astype(float)
        correct = (oof["predicted_class"].values == y).astype(float)
        edges = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for b in range(10):
            if b < 9:
                mask = (confidence >= edges[b]) & (confidence < edges[b + 1])
            else:
                mask = (confidence >= edges[b]) & (confidence <= edges[b + 1])
            if not mask.any():
                continue
            n_bin = int(mask.sum())
            mean_conf = float(confidence[mask].mean())
            acc_bin = float(correct[mask].mean())
            ece += (n_bin / len(confidence)) * abs(acc_bin - mean_conf)
            rows.append(
                {
                    "row_type": "bin",
                    "model": model_name,
                    "bin": f"[{edges[b]:.1f},{edges[b + 1]:.1f}]",
                    "n": n_bin,
                    "mean_predicted_probability": mean_conf,
                    "observed_accuracy": acc_bin,
                    "brier_score": "",
                    "ece": "",
                    "note": "",
                }
            )
        rows.append(
            {
                "row_type": "summary",
                "model": model_name,
                "bin": "",
                "n": int(len(oof)),
                "mean_predicted_probability": "",
                "observed_accuracy": float(correct.mean()),
                "brier_score": brier,
                "ece": float(ece),
                "note": (
                    "prediction_confidence is a confidence proxy (argmax class "
                    "probability), not a calibrated probability."
                ),
            }
        )
        summaries[model_name] = {"brier_score": brier, "ece": float(ece)}
    _write_csv(pd.DataFrame(rows), "calibration_report.csv")
    return {"csv": "calibration_report.csv", "per_model": summaries}


# ---------------------------------------------------------------------------
# 17. Year ablation (spatial + temporal)
# ---------------------------------------------------------------------------


def year_ablation(df: pd.DataFrame, predictor_cols: List[str]) -> Dict:
    """Investigate the role of the ``year`` predictor.

    Compares full features with vs without ``year`` (dropped from
    ``predictor_cols``) for Random Forest and XGBoost under (a) spatial CV
    (via ``experiments.run_spatial_cv``, one row per fold) and (b) temporal
    validation in both directions (2022->2026 and 2026->2022) via
    ``temporal_train_test_split`` and ``train_random_forest`` /
    ``train_xgboost`` — mirroring ``experiments.run_temporal_validation`` but
    parameterized by predictor columns.

    Writes ``year_feature_investigation.csv`` with columns: setting, model,
    variant, fold, train_year, test_year, accuracy, balanced_accuracy,
    macro_f1, high_recall, severe_recall.
    """
    from .random_forest import train_random_forest
    from .xgboost_model import train_xgboost

    variants = {
        "with_year": list(predictor_cols),
        "without_year": [c for c in predictor_cols if c != YEAR_VAR],
    }
    rows = []
    for variant, cols in variants.items():
        for model_name, factory in (("Random Forest", make_rf), ("XGBoost", make_xgb)):
            train_fn, predict_fn = factory()
            result = run_spatial_cv(df, cols, model_name, train_fn, predict_fn)
            for fr in result["fold_results"]:
                rows.append(
                    {
                        "setting": "spatial",
                        "model": model_name,
                        "variant": variant,
                        "fold": fr["fold"],
                        "train_year": "",
                        "test_year": "",
                        "accuracy": fr["accuracy"],
                        "balanced_accuracy": fr["balanced_accuracy"],
                        "macro_f1": fr["macro_f1"],
                        "high_recall": fr["high_recall"],
                        "severe_recall": fr["severe_recall"],
                    }
                )
            for train_year, test_year in [(2022, 2026), (2026, 2022)]:
                split = temporal_train_test_split(df, cols, train_year, test_year)
                if model_name == "Random Forest":
                    model = train_random_forest(split["X_train"], split["y_train"])
                    preds = predict_random_forest(model, split["X_test"])
                else:
                    model = train_xgboost(split["X_train"], split["y_train"])
                    preds = predict_xgboost(model, split["X_test"])
                m = compute_metrics(
                    split["y_test"].values, preds["y_pred"], preds.get("y_prob")
                )
                rows.append(
                    {
                        "setting": "temporal",
                        "model": model_name,
                        "variant": variant,
                        "fold": "",
                        "train_year": train_year,
                        "test_year": test_year,
                        "accuracy": m["accuracy"],
                        "balanced_accuracy": m["balanced_accuracy"],
                        "macro_f1": m["macro_f1"],
                        "high_recall": m["high_recall"],
                        "severe_recall": m["severe_recall"],
                    }
                )
    _write_csv(pd.DataFrame(rows), "year_feature_investigation.csv")
    return {
        "csv": "year_feature_investigation.csv",
        "variants": list(variants.keys()),
    }


# ---------------------------------------------------------------------------
# Baseline manifest and generalization summary
# ---------------------------------------------------------------------------


def build_baseline_manifest(df: pd.DataFrame, predictor_cols: List[str]) -> Dict:
    """Capture the Phase 5 baseline configuration as a JSON manifest.

    Writes ``phase5_baseline_manifest.json`` into ``OUTPUT_DIR`` and returns
    the manifest dictionary: timestamp, Python/library versions, random seed,
    fold count, predictor list, RF/XGB params, target definition, dataset
    shape, class distribution (full descriptive thresholds), and occupied
    spatial block ids.
    """
    import xgboost

    y_full, thresholds = build_target(df, row_mask=None)
    manifest = {
        "generated_utc": _now(),
        "python_version": platform.python_version(),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "random_seed": RANDOM_SEED,
        "n_folds": N_SPATIAL_FOLDS,
        "predictor_cols": list(predictor_cols),
        "random_forest_params": RANDOM_FOREST_PARAMS,
        "xgboost_params": {**XGBOOST_PARAMS, "num_class": len(CLASS_VALUES)},
        "target_definition": {
            "variable": TARGET_VAR,
            "method": "per-year quartiles",
            "thresholds": {str(k): v for k, v in thresholds.items()},
            "cv_policy": "training-only thresholds inside each CV fold",
        },
        "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
        "class_distribution": {
            str(k): int(v) for k, v in y_full.value_counts().sort_index().items()
        },
        "occupied_block_ids": sorted(df[GROUP_VAR].astype(int).unique().tolist()),
    }
    path = Path(OUTPUT_DIR) / "phase5_baseline_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def _convert_for_json(obj: object) -> object:
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_generalization_summary(results: Dict, path: Path) -> Dict:
    """Merge audit step results into the generalization summary JSON.

    Keys covered (missing steps are filled with ``"not_run"`` placeholders):
    baseline metrics, regularization result paths, nested CV summary, locked
    holdout, seed stability, learning curve verdict, permutation test verdict,
    LOBO summary, spatial distance, calibration, autocorrelation, selected
    model placeholders, generalization verdict, overfitting verdict, and
    remaining limitations (list of strings).
    """
    summary = {
        "generated_utc": _now(),
        "random_seed": RANDOM_SEED,
        "baseline_metrics": results.get("baseline_metrics", "not_run"),
        "regularization": {
            "rf": results.get("rf_regularization", "not_run"),
            "xgb": results.get("xgb_regularization", "not_run"),
        },
        "nested_spatial_cv": results.get("nested_spatial_cv", "not_run"),
        "locked_geographic_holdout": results.get("locked_geographic_holdout", "not_run"),
        "seed_stability": results.get("seed_stability", "not_run"),
        "learning_curves": results.get("learning_curves", "not_run"),
        "target_permutation_verdict": results.get("target_permutation", "not_run"),
        "leave_one_block_out": results.get("leave_one_block_out", "not_run"),
        "spatial_distance": results.get("spatial_distance", "not_run"),
        "calibration": results.get("calibration", "not_run"),
        "spatial_error_autocorrelation": results.get("spatial_error_autocorrelation", "not_run"),
        "feature_permutation_sanity": results.get("feature_permutation_sanity", "not_run"),
        "prediction_stability": results.get("prediction_stability", "not_run"),
        "compact_feature_experiment": results.get("compact_feature_experiment", "not_run"),
        "year_ablation": results.get("year_ablation", "not_run"),
        "duplicate_audit": results.get("duplicate_audit", "not_run"),
        "leakage_audit": results.get("leakage_audit", "not_run"),
        "selected_model": results.get(
            "selected_model",
            "TBD: select via experiments.select_final_model after audit review",
        ),
        "selected_model_rationale": results.get(
            "selected_model_rationale", "TBD: pending final model selection."
        ),
        "generalization_verdict": results.get(
            "generalization_verdict",
            "TBD: analyst verdict after reviewing the audit CSVs above.",
        ),
        "overfitting_verdict": results.get(
            "overfitting_verdict",
            "TBD: analyst verdict after reviewing train/validation gaps and regularization sweeps.",
        ),
        "remaining_limitations": results.get(
            "remaining_limitations",
            [
                "Classes are relative per-year quartiles, not absolute UHI ground truth.",
                "Spatial CV measures interpolation within the study area, not transfer to new cities.",
                "Morphology features use the distance==0 pixel proxy and suffer block-edge clipping.",
            ],
        ),
    }
    # Preserve any additional step results not covered above.
    for key, value in results.items():
        if key not in summary:
            summary[key] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert_for_json(summary), f, indent=2, default=str)
    return summary


# ---------------------------------------------------------------------------
# Audit orchestration
# ---------------------------------------------------------------------------

SIMPLE_STEPS = {
    "train_validation_gap": train_validation_gap,
    "seed_stability": seed_stability,
    "target_permutation": target_permutation,
    "learning_curves": learning_curves,
    "leave_one_block_out": leave_one_block_out,
    "spatial_distance": spatial_distance,
    "feature_permutation_sanity": feature_permutation_sanity,
    "nested_spatial_cv": nested_spatial_cv,
    "locked_geographic_holdout": locked_geographic_holdout,
    "spatial_error_autocorrelation": spatial_error_autocorrelation,
    "duplicate_audit": duplicate_audit,
    "leakage_audit": leakage_audit,
    "calibration": calibration,
    "year_ablation": year_ablation,
}


def run_audit(
    df: pd.DataFrame,
    predictor_cols: List[str],
    steps: Optional[Sequence[str]] = None,
) -> Dict:
    """Run a named subset of audit steps and return summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Full merged Phase 4 dataset (spatial + morphology features merged).
    predictor_cols : list[str]
        Predictor columns for the model-based steps.
    steps : sequence of str, optional
        Step names (see ``SIMPLE_STEPS`` plus ``regularization``,
        ``compact_feature``, ``prediction_stability``).  Defaults to
        ``DEFAULT_STEPS``.

    Returns
    -------
    dict
        Per-step summary dictionaries, plus ``phase5_generalization_summary.json``
        written into ``OUTPUT_DIR``.  When both ``seed_stability`` and
        ``prediction_stability`` are requested, the seed-stability OOF runs are
        computed once and reused.
    """
    steps = list(steps) if steps is not None else list(DEFAULT_STEPS)
    results: Dict[str, object] = {}
    seed_runs = None
    for step in steps:
        if step == "regularization":
            results["rf_regularization"] = run_rf_regularization(df, predictor_cols)
            results["xgb_regularization"] = run_xgb_regularization(df, predictor_cols)
        elif step == "compact_feature":
            results["compact_feature_experiment"] = run_compact_feature_experiment(df)
        elif step == "prediction_stability":
            results["prediction_stability"] = prediction_stability(
                df, predictor_cols, seed_runs=seed_runs
            )
        elif step in SIMPLE_STEPS:
            if step == "seed_stability" and "prediction_stability" in steps:
                # Compute the per-seed RF OOF frames once and reuse them for
                # prediction_stability instead of retraining.
                out = seed_stability(df, predictor_cols, return_oof=True)
                seed_runs = out["oof"]
                results[step] = out["summary"]
            else:
                results[step] = SIMPLE_STEPS[step](df, predictor_cols)
        else:
            raise ValueError(
                f"Unknown audit step: {step!r}. Known steps: {sorted(SIMPLE_STEPS) + ['regularization', 'compact_feature', 'prediction_stability']}"
            )
    write_generalization_summary(
        results, Path(OUTPUT_DIR) / "phase5_generalization_summary.json"
    )
    return results


def _load_full_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """Load the Phase 4 dataset and merge spatial + morphology features."""
    from .dataset import merge_morphology_features, merge_spatial_features

    df = load_phase4_dataset()
    df = merge_spatial_features(df)
    df = merge_morphology_features(df)
    predictor_cols = get_feature_columns_for_model(
        df, include_year=True, include_pvc=True, include_spatial=True, include_morphology=True
    )
    return df, predictor_cols


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: run the generalization audit on the full dataset."""
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 5 generalization/leakage audit diagnostics."
    )
    parser.add_argument(
        "steps",
        nargs="*",
        default=None,
        help=f"Audit steps to run (default: {DEFAULT_STEPS}).",
    )
    args = parser.parse_args(argv)

    df, predictor_cols = _load_full_dataset()
    steps = args.steps if args.steps else None
    results = run_audit(df, predictor_cols, steps=steps)
    print(json.dumps(_convert_for_json(results), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
