"""Random Forest classifier for Phase 5 relative heat-severity classification."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from .config import RANDOM_FOREST_PARAMS


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train a Random Forest classifier with fixed reproducible parameters."""
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    return model


def predict_random_forest(
    model: RandomForestClassifier,
    X: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Return class predictions and probabilities from a trained RF model."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return {"y_pred": y_pred, "y_prob": y_prob}


def random_forest_feature_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
) -> pd.DataFrame:
    """Return built-in feature importance from a trained RF model."""
    importance = model.feature_importances_
    df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)
    return df


def random_forest_permutation_importance(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
    random_state: int = 42,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Compute permutation importance on a held-out validation set."""
    # Run single-threaded: parallel permutation importance pickles the whole
    # training matrix for each worker, which is slow and can exhaust temp space.
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="f1_macro",
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    return df
