"""Suitability scoring, classes, scenarios, and raster outputs.

FROZEN concept model (design spec section a)::

    Final (Scenario A, baseline) = Need * Opportunity / 100

Gated multiplicative formulation: extreme heat CANNOT produce high
suitability where opportunity is absent (Opportunity = 0 -> Final = 0
regardless of Need).  An additive diagnostic (0.5*Need + 0.5*Opportunity)
is computed for comparison only — additive forms let a Very-High-Need /
Zero-Opportunity cell score 50, which the concept model rejects; it is
reported in tables and is never the primary map.

Scenarios B/C/D use the FROZEN weighted geometric mean (spec section h)::

    Final_s = 100^(1 - a_need - a_opp) * Need^a_need * Opportunity^a_opp

which stays in [0, 100] and collapses exactly to the gated product when
a_need = a_opp = 1.  Scenario D varies the internal composition
(vegetation_cover replaces NDVI in Need; headroom computed on inverted
normalized vegetation_cover, upweighted in Opportunity) rather than the
gating exponents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import rasterio

from .config import (
    CLASS_EDGES,
    CLASS_LABELS,
    CLASS_NODATA,
    SCORE_NODATA,
    SCENARIO_EXPONENTS,
)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def gated_product(need: np.ndarray, opportunity: np.ndarray) -> np.ndarray:
    """Baseline suitability: Need x Opportunity / 100 (both in [0, 100])."""
    final = need.astype(np.float64) * opportunity.astype(np.float64) / 100.0
    return np.clip(final, 0.0, 100.0)


def additive_diagnostic(need: np.ndarray, opportunity: np.ndarray) -> np.ndarray:
    """Additive comparison score 0.5*Need + 0.5*Opp — diagnostic ONLY."""
    diag = 0.5 * need.astype(np.float64) + 0.5 * opportunity.astype(np.float64)
    return np.clip(diag, 0.0, 100.0)


def weighted_geometric_mean(
    need: np.ndarray, opportunity: np.ndarray, a_need: float, a_opp: float
) -> np.ndarray:
    """100^(1-an-ao) * Need^an * Opp^ao — FROZEN scenario form (spec h).

    The normalization factor 100^(1-an-ao) keeps the result in [0, 100]
    for any non-negative exponents: the function is monotone increasing in
    each argument, equals 0 when either factor is 0, and equals exactly 100
    at Need = Opp = 100.  For A/D (an = ao = 1) it reduces exactly to the
    gated product Need * Opportunity / 100.
    """
    if a_need < 0.0 or a_opp < 0.0:
        raise AssertionError(f"scenario exponents must be non-negative: {(a_need, a_opp)}")
    final = (
        100.0 ** (1.0 - a_need - a_opp)
        * need.astype(np.float64) ** a_need
        * opportunity.astype(np.float64) ** a_opp
    )
    return np.clip(final, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Classes (FROZEN thresholds, spec section g)
# ---------------------------------------------------------------------------
def classify(score: np.ndarray) -> np.ndarray:
    """Map a 0-100 suitability score to classes 0-4.

    Boundaries 20/40/60/80 belong to the LOWER class (score == 20 is
    class 0 "Very Low", consistent with the spec's "0-20" interval).
    Classes: 0 Very Low, 1 Low, 2 Medium, 3 High, 4 Very High.
    """
    classes = np.searchsorted(CLASS_EDGES, score, side="left")
    return np.clip(classes, 0, len(CLASS_LABELS) - 1).astype(np.int16)


def assert_class_partition() -> None:
    """Assert the class thresholds strictly partition [0, 100] as documented."""
    edges = CLASS_EDGES
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise AssertionError(f"class edges not strictly increasing: {edges}")
    probe = np.array([0.0, 20.0, 20.0001, 40.0, 60.0, 80.0, 100.0])
    expected = np.array([0, 0, 1, 1, 2, 3, 4], dtype=np.int16)
    got = classify(probe)
    if not (got == expected).all():
        raise AssertionError(f"class partition mismatch: {got} vs {expected}")


def assert_class_consistency(score: np.ndarray, classes: np.ndarray) -> None:
    """Assert a class raster equals classify(score) at every cell."""
    if not (classify(score) == classes).all():
        raise AssertionError("class raster inconsistent with score raster")


# ---------------------------------------------------------------------------
# Raster writing
# ---------------------------------------------------------------------------
def write_score_raster(
    flat_values: np.ndarray,
    output_path: Path,
    profile: Dict,
    rows: np.ndarray,
    cols: np.ndarray,
) -> Dict:
    """Write a float32 0-100 score raster, NoData-initialised outside valid cells."""
    values = flat_values.astype(np.float32)
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 100.0:
        raise AssertionError(f"score raster {output_path} values outside [0, 100]")

    raster = np.full((profile["height"], profile["width"]), SCORE_NODATA, dtype=np.float32)
    raster[rows, cols] = values

    out_profile = profile.copy()
    out_profile.update(
        {
            "dtype": "float32",
            "count": 1,
            "nodata": SCORE_NODATA,
            "compress": "lzw",
        }
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(raster, 1)

    return {
        "output_path": str(output_path),
        "dtype": "float32",
        "nodata": SCORE_NODATA,
        "n_valid_pixels": int(len(rows)),
    }


def write_class_raster(
    flat_scores: np.ndarray,
    output_path: Path,
    profile: Dict,
    rows: np.ndarray,
    cols: np.ndarray,
) -> Dict:
    """Write the int16 class raster derived from float32-cast scores.

    Classification is performed on the float32-cast score values so the
    written class raster is bit-consistent with the written score raster at
    every valid cell (asserted).
    """
    scores_f32 = flat_scores.astype(np.float32)
    classes = classify(scores_f32.astype(np.float64))
    assert_class_consistency(scores_f32.astype(np.float64), classes)

    raster = np.full((profile["height"], profile["width"]), CLASS_NODATA, dtype=np.int16)
    raster[rows, cols] = classes

    out_profile = profile.copy()
    out_profile.update(
        {
            "dtype": "int16",
            "count": 1,
            "nodata": CLASS_NODATA,
            "compress": "lzw",
        }
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(raster, 1)

    return {
        "output_path": str(output_path),
        "dtype": "int16",
        "nodata": CLASS_NODATA,
        "n_valid_pixels": int(len(rows)),
    }


def compute_scenario_scores(
    need: Dict[str, np.ndarray], opportunity: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute all scenario suitability scores per year.

    Parameters
    ----------
    need, opportunity : dict
        ``need[variant]`` / ``opportunity[variant]`` per-valid-cell arrays
        (variants ``"baseline"`` and ``"D"``).

    Returns
    -------
    dict
        ``scores[scenario][year]`` = 0-100 suitability (scenario A is the
        gated baseline product; B/C re-gate the baseline composition with
        FROZEN exponents; D applies its own composition at exponents (1, 1)).
    """
    scores: Dict[str, Dict[int, np.ndarray]] = {
        s: {} for s in SCENARIO_EXPONENTS
    }
    for year in need:
        for scenario, (a_need, a_opp) in SCENARIO_EXPONENTS.items():
            variant = "D" if scenario == "D" else "baseline"
            if scenario == "A":
                final = gated_product(need[year][variant], opportunity[year][variant])
            else:
                final = weighted_geometric_mean(
                    need[year][variant], opportunity[year][variant], a_need, a_opp
                )
            if final.min() < 0.0 or final.max() > 100.0:
                raise AssertionError(
                    f"scenario {scenario} {year} outside [0, 100]: "
                    f"[{final.min()}, {final.max()}]"
                )
            scores[scenario][year] = final
    return scores


__all__ = [
    "gated_product",
    "additive_diagnostic",
    "weighted_geometric_mean",
    "classify",
    "assert_class_partition",
    "assert_class_consistency",
    "write_score_raster",
    "write_class_raster",
    "compute_scenario_scores",
]
