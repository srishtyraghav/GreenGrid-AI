"""Per-year robust min-max normalization (FROZEN, design spec section f).

Each input quantity is clipped at its 1st-99th percentile computed over
valid-mask cells for that year, then scaled to 0-100.  Per-year
normalization means all scores and class boundaries are class-relative
across years — consistent with the Phase 6 framing (2022 and 2026 are two
snapshot composites, not a trend).

Every bound (p1, p99, per variable per year) is recorded in
``data/processed/phase7/tables/normalization_parameters.csv`` and must be
reproduced in the Phase 7 report.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import NORM_PERCENTILES, YEARS

# Quantities normalized per year.  ``one_minus_ndvi`` / ``one_minus_veg`` are
# the inverted raw quantities normalized as-is (design spec section b writes
# n(1-NDVI) literally); Opportunity's inverted terms instead invert the
# normalized base variable (100 - n(NDBI), 100 - n(NDVI), spec section c).
BASE_QUANTITIES = (
    "severity_score",
    "ndvi",
    "ndbi",
    "lst",
    "vegetation_cover",
)
DERIVED_QUANTITIES = (
    "one_minus_ndvi",
    "one_minus_vegetation_cover",
)


def robust_minmax(values: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Clip to the 1st/99th percentile and scale to 0-100.

    Parameters
    ----------
    values : np.ndarray
        Finite per-valid-cell values for one variable and year.

    Returns
    -------
    normalized : np.ndarray
        Values clipped to [p1, p99] and linearly rescaled to [0, 100].
    bounds : dict
        ``p1``, ``p99``, ``n_cells`` actually used.
    """
    p_low, p_high = NORM_PERCENTILES
    p1, p99 = np.percentile(values, (p_low, p_high))
    if not p99 > p1:
        raise AssertionError(f"degenerate percentile range: p1={p1}, p99={p99}")
    scaled = (np.clip(values, p1, p99) - p1) / (p99 - p1) * 100.0
    scaled = np.clip(scaled, 0.0, 100.0)
    bounds = {
        "p1": float(p1),
        "p99": float(p99),
        "n_cells": int(values.size),
    }
    return scaled, bounds


def normalize_inputs(data: Dict) -> Tuple[Dict, List[Dict]]:
    """Normalize every input quantity per year.

    Parameters
    ----------
    data : dict
        Output of :func:`suitability.inputs.load_inputs`.

    Returns
    -------
    normalized : dict
        ``normalized[year][quantity]`` = 0-100 float64 array per valid cell.
    bounds_rows : list of dict
        One row per year/variable for ``normalization_parameters.csv``.
    """
    normalized: Dict[int, Dict[str, np.ndarray]] = {}
    bounds_rows: List[Dict] = []

    for year in YEARS:
        yearly = data["yearly"][year]
        out: Dict[str, np.ndarray] = {}
        for name in BASE_QUANTITIES:
            scaled, bounds = robust_minmax(yearly[name])
            out[name] = scaled
            bounds_rows.append({"year": year, "variable": name, **bounds})
        for name, base in (
            ("one_minus_ndvi", "ndvi"),
            ("one_minus_vegetation_cover", "vegetation_cover"),
        ):
            scaled, bounds = robust_minmax(1.0 - yearly[base])
            out[name] = scaled
            bounds_rows.append({"year": year, "variable": name, **bounds})
        normalized[year] = out

    return normalized, bounds_rows


__all__ = ["BASE_QUANTITIES", "DERIVED_QUANTITIES", "robust_minmax", "normalize_inputs"]
