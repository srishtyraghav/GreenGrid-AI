"""Dataset preparation for Phase 5 modelling.

Loads the Phase 4 combined dataset, validates required columns, builds the
predictor matrix (including one-hot encoding of categorical variables), and
exposes identifiers, groups, targets, and feature names.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    BASE_PREDICTOR_VARS,
    CATEGORICAL_VARS,
    GROUP_VAR,
    IDENTIFIER_VARS,
    INPUT_DATASET_CSV,
    MORPHOLOGY_FEATURE_COLS,
    PREDICTOR_VARS,
    REQUIRED_INPUT_COLS,
    SPATIAL_FEATURE_COLS,
    TARGET_VAR,
    USE_MORPHOLOGY_FEATURES,
    USE_SPATIAL_FEATURES,
    YEAR_VAR,
)


def load_phase4_dataset(path: str = INPUT_DATASET_CSV) -> pd.DataFrame:
    """Load and validate the Phase 4 combined dataset."""
    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_INPUT_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Phase 4 dataset missing required columns: {missing}")

    # Ensure year is integer.
    df[YEAR_VAR] = df[YEAR_VAR].astype(int)

    # landuse_class is categorical; keep as integer-like for one-hot encoding.
    for col in CATEGORICAL_VARS:
        df[col] = df[col].astype(int)

    return df


def encode_predictors(
    df: pd.DataFrame,
    predictor_cols: List[str] = PREDICTOR_VARS,
    categorical_cols: List[str] = CATEGORICAL_VARS,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the model-ready predictor matrix.

    Continuous predictors are used as-is. Categorical predictors are one-hot
    encoded with explicit column names.

    Returns
    -------
    tuple
        (predictor_df, feature_names)
    """
    frames = []
    feature_names = []

    # Neighbourhood-dominant land-use columns are also categorical.
    categorical_col_set = set(categorical_cols) | {
        c for c in predictor_cols if c.startswith("landuse_dominant_")
    }

    for col in predictor_cols:
        if col in categorical_col_set:
            # One-hot encode.
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            frames.append(dummies)
            feature_names.extend(dummies.columns.tolist())
        else:
            frames.append(df[[col]].copy())
            feature_names.append(col)

    X = pd.concat(frames, axis=1)
    return X, feature_names


def prepare_modeling_data(
    df: pd.DataFrame,
    target_series: pd.Series,
) -> Dict[str, object]:
    """Prepare all data structures needed for modelling.

    Returns
    -------
    dict with keys:
        X : pd.DataFrame
            Predictor matrix (one-hot encoded where needed).
        y : pd.Series
            Integer class labels.
        groups : pd.Series
            Spatial block IDs.
        identifiers : pd.DataFrame
            lon, lat, row, col, year.
        feature_names : list[str]
            Names of columns in X.
        target_var : str
            Name of target variable (for audit).
    """
    # Drop rows where target is missing.
    valid_idx = target_series.notna()
    df_valid = df.loc[valid_idx].copy()
    y = target_series.loc[valid_idx].astype(int)

    X, feature_names = encode_predictors(df_valid)

    groups = df_valid[GROUP_VAR].astype(int)
    identifiers = df_valid[IDENTIFIER_VARS + [YEAR_VAR, GROUP_VAR]].copy()

    # Leakage sanity check.
    forbidden_in_X = {TARGET_VAR, GROUP_VAR} | set(IDENTIFIER_VARS)
    leaked = forbidden_in_X.intersection(X.columns)
    if leaked:
        raise ValueError(f"Leakage detected: forbidden columns in X: {leaked}")

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "identifiers": identifiers,
        "feature_names": feature_names,
        "target_var": TARGET_VAR,
    }


def merge_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge precomputed spatial neighbourhood features into the dataset."""
    if not USE_SPATIAL_FEATURES:
        return df

    from .spatial_features import merge_spatial_features as _merge_spatial
    return _merge_spatial(df)


def merge_morphology_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge precomputed urban morphology features into the dataset."""
    if not USE_MORPHOLOGY_FEATURES:
        return df

    from .morphology_features import merge_morphology_features as _merge_morph
    return _merge_morph(df)


def get_feature_columns_for_model(
    df: pd.DataFrame,
    include_year: bool = True,
    include_pvc: bool = True,
    include_spatial: bool = True,
    include_morphology: bool = True,
) -> List[str]:
    """Return a list of predictor column names for a specific model variant.

    Useful for ablation experiments.
    """
    cols = ["ndvi", "ndbi"]
    if include_pvc:
        cols.append("vegetation_cover")
    cols.append("landuse_class")
    cols.extend(["dist_road_m", "dist_vegetation_m", "dist_building_m"])
    if include_year:
        cols.append(YEAR_VAR)
    if include_spatial and USE_SPATIAL_FEATURES:
        cols.extend(SPATIAL_FEATURE_COLS)
    if include_morphology and USE_MORPHOLOGY_FEATURES:
        cols.extend(MORPHOLOGY_FEATURE_COLS)
    return cols
