"""Spatial and temporal validation split generation for Phase 5.

All splits respect the spatial_block_id grouping variable. Target thresholds
for cross-validation are derived from training blocks only to prevent leakage.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import GROUP_VAR, N_SPATIAL_FOLDS, TARGET_VAR, YEAR_VAR
from .dataset import encode_predictors, prepare_modeling_data
from .target import assign_classes_from_thresholds, build_target, compute_year_thresholds


def create_spatial_folds(
    groups: pd.Series,
    n_splits: int = N_SPATIAL_FOLDS,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield train/validation indices using GroupKFold on spatial blocks."""
    # Use placeholder y since GroupKFold only needs groups.
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(np.zeros(len(groups)), groups=groups.values):
        yield train_idx, val_idx


def get_fold_blocks(
    groups: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[List[int], List[int]]:
    """Return sorted lists of training and validation block IDs."""
    train_blocks = sorted(groups.iloc[train_idx].unique().tolist())
    val_blocks = sorted(groups.iloc[val_idx].unique().tolist())
    return train_blocks, val_blocks


def spatial_cv_splits(
    df: pd.DataFrame,
    predictor_cols: List[str],
    n_splits: int = N_SPATIAL_FOLDS,
) -> Iterator[Dict]:
    """Generate spatial cross-validation folds with training-only thresholds.

    For each fold:
      1. Determine training and validation blocks.
      2. Compute per-year target thresholds using training rows only.
      3. Assign classes to all rows using those thresholds.
      4. Return train/val predictor matrices and labels.

    Yields
    ------
    dict containing:
        fold : int
        train_blocks : list[int]
        val_blocks : list[int]
        n_train : int
        n_val : int
        X_train : pd.DataFrame
        y_train : pd.Series
        X_val : pd.DataFrame
        y_val : pd.Series
        identifiers_train : pd.DataFrame
        identifiers_val : pd.DataFrame
        thresholds : dict[int, dict[str, float]]
        train_class_counts : pd.Series
        val_class_counts : pd.Series
    """
    # Encode predictors once using the full dataset so column order is stable.
    X_full, feature_names = encode_predictors(df, predictor_cols=predictor_cols)

    groups_full = df[GROUP_VAR].astype(int)
    years = sorted(df[YEAR_VAR].unique())

    for fold, (train_idx, val_idx) in enumerate(
        create_spatial_folds(groups_full, n_splits=n_splits), start=1
    ):
        train_blocks, val_blocks = get_fold_blocks(groups_full, train_idx, val_idx)

        # Verify disjointness.
        if set(train_blocks).intersection(val_blocks):
            raise ValueError(
                f"Fold {fold}: training and validation blocks overlap: "
                f"{train_blocks} vs {val_blocks}"
            )

        # Training-only row mask for threshold computation.
        row_mask = np.zeros(len(df), dtype=bool)
        row_mask[train_idx] = True

        # Build target using training-only thresholds.
        y_series, thresholds = build_target(df, row_mask=row_mask)

        # Select train/val subsets.
        X_train = X_full.iloc[train_idx]
        X_val = X_full.iloc[val_idx]
        y_train = y_series.iloc[train_idx].astype(int)
        y_val = y_series.iloc[val_idx].astype(int)

        identifiers = df[[]].copy()  # placeholder to ensure index alignment
        identifiers["lon"] = df["lon"]
        identifiers["lat"] = df["lat"]
        identifiers["row"] = df["row"]
        identifiers["col"] = df["col"]
        identifiers[YEAR_VAR] = df[YEAR_VAR]
        identifiers[GROUP_VAR] = df[GROUP_VAR]

        identifiers_train = identifiers.iloc[train_idx].copy()
        identifiers_val = identifiers.iloc[val_idx].copy()

        yield {
            "fold": fold,
            "train_blocks": train_blocks,
            "val_blocks": val_blocks,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "identifiers_train": identifiers_train,
            "identifiers_val": identifiers_val,
            "thresholds": thresholds,
            "feature_names": feature_names,
            "train_class_counts": y_train.value_counts().sort_index(),
            "val_class_counts": y_val.value_counts().sort_index(),
        }


def temporal_train_test_split(
    df: pd.DataFrame,
    predictor_cols: List[str],
    train_year: int,
    test_year: int,
) -> Dict:
    """Create a temporal train/test split using independent target thresholds.

    The model is trained on `train_year` features and that year's thresholds,
    then applied to `test_year` features. Test labels are constructed using
    `test_year`'s own thresholds.

    Returns
    -------
    dict with keys: X_train, y_train, X_test, y_test, identifiers_train,
    identifiers_test, thresholds_train, thresholds_test.
    """
    X_full, feature_names = encode_predictors(df, predictor_cols=predictor_cols)

    train_mask = df[YEAR_VAR] == train_year
    test_mask = df[YEAR_VAR] == test_year

    # Training thresholds from training year only.
    thresholds_train = {
        train_year: compute_year_thresholds(df, train_year, row_mask=train_mask.values)
    }
    # Test thresholds from test year only.
    thresholds_test = {
        test_year: compute_year_thresholds(df, test_year, row_mask=test_mask.values)
    }

    # Assign classes for the training year using training thresholds.
    y_series = pd.Series(np.nan, index=df.index, name="uhi_class")
    train_lst = df.loc[train_mask, TARGET_VAR]
    y_series.loc[train_mask] = assign_classes_from_thresholds(
        train_lst, thresholds_train[train_year]
    )
    # Assign classes for the test year using test thresholds.
    test_lst = df.loc[test_mask, TARGET_VAR]
    y_series.loc[test_mask] = assign_classes_from_thresholds(
        test_lst, thresholds_test[test_year]
    )

    train_idx = np.flatnonzero(train_mask.values)
    test_idx = np.flatnonzero(test_mask.values)

    identifiers = df[["lon", "lat", "row", "col", YEAR_VAR, GROUP_VAR]].copy()

    return {
        "train_year": train_year,
        "test_year": test_year,
        "X_train": X_full.iloc[train_idx],
        "y_train": y_series.iloc[train_idx].astype(int),
        "X_test": X_full.iloc[test_idx],
        "y_test": y_series.iloc[test_idx].astype(int),
        "identifiers_train": identifiers.iloc[train_idx].copy(),
        "identifiers_test": identifiers.iloc[test_idx].copy(),
        "thresholds_train": thresholds_train,
        "thresholds_test": thresholds_test,
        "feature_names": feature_names,
    }
