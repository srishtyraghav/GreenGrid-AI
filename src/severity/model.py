"""Production model training and prediction for Phase 6.

Retrains the Random Forest RF-C configuration (adopted per the Phase 5
report's deferral of RF-C to Phase 6) on all 300,000 Phase 4 sampled rows
with the full spatial + morphology predictor set, then applies it to the
full valid grid in fixed-size chunks for memory discipline.
"""

from __future__ import annotations

from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from models.dataset import encode_predictors
from models.target import build_target

from .config import MODEL_PATH, PREDICT_CHUNK_ROWS, RF_C_PARAMS


def train_production_model(
    df: pd.DataFrame,
    model_path=MODEL_PATH,
) -> Dict:
    """Train the RF-C production model on all sampled rows.

    Parameters
    ----------
    df : pd.DataFrame
        Phase 4 sampled dataset with spatial and morphology features
        already merged.
    model_path : Path
        Destination for the serialized model.

    Returns
    -------
    dict
        Model, encoded training matrix, feature names and target series.
    """
    target, thresholds = build_target(df, row_mask=None)
    valid_idx = target.notna()
    df_valid = df.loc[valid_idx].copy()
    y = target.loc[valid_idx].astype(int)

    X, feature_names = encode_predictors(df_valid)

    model = RandomForestClassifier(**RF_C_PARAMS)
    model.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return {
        "model": model,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "thresholds": thresholds,
        "model_path": str(model_path),
    }


def load_production_model(model_path=MODEL_PATH) -> RandomForestClassifier:
    """Load a previously saved production model."""
    return joblib.load(model_path)


def predict_full_grid(
    model: RandomForestClassifier,
    X_full: pd.DataFrame,
    chunk_rows: int = PREDICT_CHUNK_ROWS,
) -> Dict[str, np.ndarray]:
    """Predict classes and probabilities over the full grid in chunks.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained production model.
    X_full : pd.DataFrame
        Encoded full-grid design matrix, aligned to the model's training
        feature names and order.
    chunk_rows : int
        Rows per predict call.

    Returns
    -------
    dict
        ``predicted_class`` (n,), ``probabilities`` (n, 4), ``confidence``
        (n,) and ``severity_score`` (n,) arrays over all rows.
    """
    n = len(X_full)
    classes = np.empty(n, dtype=np.int16)
    probs = np.empty((n, 4), dtype=np.float32)

    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk = X_full.iloc[start:end]
        classes[start:end] = model.predict(chunk)
        probs[start:end] = model.predict_proba(chunk).astype(np.float32)
        print(f"  Predicted rows {start:,}-{end:,} of {n:,}")

    confidence = probs.max(axis=1)
    # Ordinal model score: expected class index.  This is a relative ML
    # severity score, NOT a physical temperature.
    severity_score = (probs * np.arange(4, dtype=np.float32)).sum(axis=1)

    return {
        "predicted_class": classes,
        "probabilities": probs,
        "confidence": confidence.astype(np.float32),
        "severity_score": severity_score.astype(np.float32),
    }
