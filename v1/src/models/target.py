"""Target definition for Phase 5 relative heat-severity classification.

The target is constructed from Landsat 9 LST using per-year quartiles.
Two distinct threshold sets are supported:

* Final descriptive thresholds: derived from the full per-year dataset.
* CV thresholds: derived from training blocks only to prevent target leakage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    CLASS_LABELS,
    CLASS_MAPPING,
    CLASS_VALUES,
    TARGET_QUANTILES,
    TARGET_VAR,
    YEAR_VAR,
)


def compute_year_thresholds(
    df: pd.DataFrame,
    year: int,
    row_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute LST quartile thresholds for a single year.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing at least `lst_C` and `year`.
    year : int
        Year for which to compute thresholds.
    row_mask : np.ndarray of bool, optional
        If provided, only rows where `row_mask` is True are used to compute
        thresholds. This is used during cross-validation to ensure validation
        rows do not influence their own target thresholds.

    Returns
    -------
    dict
        Mapping of boundary names to LST values:
        {'low_to_moderate': ..., 'moderate_to_high': ..., 'high_to_severe': ...}
    """
    sub = df[df[YEAR_VAR] == year]
    if row_mask is not None:
        sub = sub[row_mask[sub.index]]
    values = sub[TARGET_VAR].dropna()
    if len(values) == 0:
        raise ValueError(f"No LST values available for year {year} with given mask.")

    q25, q50, q75 = values.quantile([0.25, 0.50, 0.75]).values
    return {
        "low_to_moderate": float(q25),
        "moderate_to_high": float(q50),
        "high_to_severe": float(q75),
    }


def assign_classes_from_thresholds(
    lst_values: pd.Series,
    thresholds: Dict[str, float],
) -> pd.Series:
    """Assign integer class labels from LST values and thresholds.

    Boundaries are inclusive on the upper side:
        Low:       lst <= q25
        Moderate:  q25 < lst <= q50
        High:      q50 < lst <= q75
        Severe:    lst > q75
    """
    q25 = thresholds["low_to_moderate"]
    q50 = thresholds["moderate_to_high"]
    q75 = thresholds["high_to_severe"]

    conditions = [
        lst_values <= q25,
        (lst_values > q25) & (lst_values <= q50),
        (lst_values > q50) & (lst_values <= q75),
        lst_values > q75,
    ]
    classes = np.select(conditions, CLASS_VALUES, default=np.nan).astype(float)
    # Any NaN LST should remain NaN in the class assignment.
    classes[np.isnan(lst_values.values)] = np.nan
    return pd.Series(classes, index=lst_values.index, name="uhi_class")


def build_target(
    df: pd.DataFrame,
    row_mask: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, Dict[int, Dict[str, float]]]:
    """Assign UHI classes for all rows using per-year thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    row_mask : np.ndarray of bool, optional
        If provided, thresholds are computed only from rows where the mask is
        True, but classes are assigned to all rows of each year using those
        thresholds.

    Returns
    -------
    tuple
        (uhi_class_series, thresholds_by_year)
    """
    years = sorted(df[YEAR_VAR].unique())
    thresholds_by_year: Dict[int, Dict[str, float]] = {}
    class_series = pd.Series(np.nan, index=df.index, name="uhi_class")

    for year in years:
        thresholds = compute_year_thresholds(df, year, row_mask=row_mask)
        thresholds_by_year[year] = thresholds
        year_mask = df[YEAR_VAR] == year
        lst_values = df.loc[year_mask, TARGET_VAR]
        class_series.loc[year_mask] = assign_classes_from_thresholds(
            lst_values, thresholds
        )

    return class_series, thresholds_by_year


def get_final_descriptive_thresholds(df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """Compute final descriptive thresholds from the full per-year dataset.

    These thresholds are used for the final production maps and report tables,
    not for cross-validation.
    """
    _, thresholds = build_target(df, row_mask=None)
    return thresholds


def format_threshold_table(thresholds_by_year: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """Format thresholds as a readable table for reports."""
    records = []
    for year, thresholds in sorted(thresholds_by_year.items()):
        records.append(
            {
                "year": year,
                "low_to_moderate_C": thresholds["low_to_moderate"],
                "moderate_to_high_C": thresholds["moderate_to_high"],
                "high_to_severe_C": thresholds["high_to_severe"],
            }
        )
    return pd.DataFrame(records)


def class_distribution(series: pd.Series) -> pd.Series:
    """Return counts and percentages for each class label."""
    counts = series.value_counts().sort_index()
    counts.index = [INV_CLASS_MAPPING.get(int(i), i) for i in counts.index]
    percentages = 100.0 * counts / counts.sum()
    return pd.DataFrame({"count": counts, "percentage": percentages.round(2)})
