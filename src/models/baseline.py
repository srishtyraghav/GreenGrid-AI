"""Naïve and rule-based baseline models for Phase 5.

Provides majority-class, stratified-random, and NDVI/NDBI rule baselines
evaluated under the same spatial folds as Random Forest and XGBoost.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier


# ---------------------------------------------------------------------------
# Dummy baselines
# ---------------------------------------------------------------------------


def train_majority_baseline(y_train: pd.Series) -> DummyClassifier:
    """Train a majority-class baseline."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(np.zeros((len(y_train), 1)), y_train)
    return model


def train_stratified_baseline(y_train: pd.Series, random_state: int = 42) -> DummyClassifier:
    """Train a stratified-random baseline."""
    model = DummyClassifier(strategy="stratified", random_state=random_state)
    model.fit(np.zeros((len(y_train), 1)), y_train)
    return model


def predict_baseline(model: DummyClassifier, X: pd.DataFrame) -> Dict[str, object]:
    """Return predictions and probabilities from a baseline model."""
    # DummyClassifier ignores feature dimensions, so any compatible shape works.
    X_arr = np.zeros((len(X), 1))
    y_pred = model.predict(X_arr)
    y_prob = model.predict_proba(X_arr)
    return {"y_pred": y_pred, "y_prob": y_prob}


def baseline_model_name(strategy: str) -> str:
    """Return a display name for a baseline strategy."""
    return {"most_frequent": "Majority Baseline", "stratified": "Stratified Baseline"}[strategy]


# ---------------------------------------------------------------------------
# Rule-based NDVI / NDBI environmental baseline
# ---------------------------------------------------------------------------


class RuleBaselineModel:
    """Fixed-threshold NDVI/NDBI heat-severity baseline.

    Decision logic is determined from training blocks only, per year:

      1. Clip ``ndbi`` and ``ndvi`` to their training-block 5th/95th percentiles.
      2. Min-max normalize the clipped values to [0, 1].
      3. ``heat_score = normalized_ndbi - normalized_ndvi``.
      4. Derive ``heat_score`` q25/q50/q75 thresholds from training blocks.
      5. Apply those thresholds to validation rows of the same year.

    No threshold is chosen or optimized against validation data.
    """

    def __init__(self) -> None:
        self.year_params: Dict[int, Dict[str, float]] = {}

    def fit(self, df_train: pd.DataFrame) -> "RuleBaselineModel":
        """Compute normalization and heat-score thresholds from training blocks."""
        for year in sorted(df_train["year"].unique()):
            sub = df_train[df_train["year"] == year]
            ndbi = sub["ndbi"].values
            ndvi = sub["ndvi"].values

            p5_ndbi, p95_ndbi = np.percentile(ndbi, [5, 95])
            p5_ndvi, p95_ndvi = np.percentile(ndvi, [5, 95])

            norm_ndbi = self._normalize(ndbi, p5_ndbi, p95_ndbi)
            norm_ndvi = self._normalize(ndvi, p5_ndvi, p95_ndvi)
            heat_score = norm_ndbi - norm_ndvi

            q25, q50, q75 = np.quantile(heat_score, [0.25, 0.50, 0.75])

            self.year_params[int(year)] = {
                "p5_ndbi": float(p5_ndbi),
                "p95_ndbi": float(p95_ndbi),
                "p5_ndvi": float(p5_ndvi),
                "p95_ndvi": float(p95_ndvi),
                "q25": float(q25),
                "q50": float(q50),
                "q75": float(q75),
            }
        return self

    @staticmethod
    def _normalize(values: np.ndarray, p5: float, p95: float) -> np.ndarray:
        """Clip to [p5, p95] and min-max normalize to [0, 1]."""
        denom = p95 - p5
        if denom == 0:
            return np.zeros_like(values)
        clipped = np.clip(values, p5, p95)
        return (clipped - p5) / denom

    def predict(self, df_val: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Assign quartile classes using the training-derived thresholds."""
        y_pred = np.empty(len(df_val), dtype=int)
        y_prob = np.zeros((len(df_val), 4), dtype=np.float64)

        for year, params in self.year_params.items():
            mask = df_val["year"] == year
            if not mask.any():
                continue

            ndbi = df_val.loc[mask, "ndbi"].values
            ndvi = df_val.loc[mask, "ndvi"].values

            norm_ndbi = self._normalize(ndbi, params["p5_ndbi"], params["p95_ndbi"])
            norm_ndvi = self._normalize(ndvi, params["p5_ndvi"], params["p95_ndvi"])
            heat_score = norm_ndbi - norm_ndvi

            q25, q50, q75 = params["q25"], params["q50"], params["q75"]
            classes = np.select(
                [
                    heat_score <= q25,
                    (heat_score > q25) & (heat_score <= q50),
                    (heat_score > q50) & (heat_score <= q75),
                    heat_score > q75,
                ],
                [0, 1, 2, 3],
                default=-1,
            )
            y_pred[mask.values] = classes

            # One-hot probability for consistency with model predictors.
            for c in range(4):
                y_prob[mask.values, c] = (classes == c).astype(np.float64)

        return {"y_pred": y_pred, "y_prob": y_prob}


def train_rule_baseline(df_train: pd.DataFrame) -> RuleBaselineModel:
    """Train the NDVI/NDBI rule baseline on training-block data."""
    model = RuleBaselineModel()
    model.fit(df_train)
    return model


def predict_rule_baseline(model: RuleBaselineModel, df_val: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Return predictions and probabilities from the rule baseline."""
    return model.predict(df_val)
