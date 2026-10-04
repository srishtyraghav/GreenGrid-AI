"""XGBoost classifier for Phase 5 relative heat-severity classification."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import CLASS_VALUES, RANDOM_SEED, XGBOOST_PARAMS


def _xgboost_params_with_num_class(params: Dict) -> Dict:
    """Add num_class parameter required by XGBoost for multi-class objectives."""
    p = params.copy()
    p["num_class"] = len(CLASS_VALUES)
    return p


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """Train an XGBoost classifier with fixed reproducible parameters."""
    params = _xgboost_params_with_num_class(XGBOOST_PARAMS)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def predict_xgboost(
    model: XGBClassifier,
    X: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Return class predictions and probabilities from a trained XGB model."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return {"y_pred": y_pred, "y_prob": y_prob}


def xgboost_feature_importance(
    model: XGBClassifier,
    feature_names: List[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Return XGBoost feature importance."""
    importance = model.get_booster().get_score(importance_type=importance_type)
    # XGBoost may omit unused features; ensure all features appear.
    full = {name: importance.get(name, 0.0) for name in feature_names}
    df = pd.DataFrame(
        {"feature": list(full.keys()), "importance": list(full.values())}
    ).sort_values("importance", ascending=False)
    return df
