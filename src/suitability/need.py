"""Heat Need (0-100) — baseline and Scenario D variants.

FROZEN formulation (design spec section b)::

    Need = 0.40*n(severity_score) + 0.25*n(1-NDVI) + 0.20*n(NDBI)
         + 0.15*n(LST)

``severity_score`` (Phase 6, expected ordinal score 0-3) is the primary
heat-need term.  ``vegetation_cover`` is EXCLUDED from the baseline — Phase
5 froze NDVI-vegetation_cover redundancy at r ~ 0.998-0.9995; it appears
only in Scenario D, where the 1-NDVI term is replaced by
1-vegetation_cover.  The LST weight is deliberately small (0.15) because
the severity score is itself trained on per-year LST quartiles (limits
double-counting; documented, not tuned).

Heat Need is a RELATIVE heat-stress construct within the plantation
suitability model — not a physical heat measure.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import NEED_WEIGHTS, SCENARIO_D_NEED_WEIGHTS

#: Variant -> (weights dict).  "baseline" is Scenario A/B/C Need; "D" swaps
#: the 1-NDVI term for 1-vegetation_cover (FROZEN, spec section h).
VARIANT_WEIGHTS = {
    "baseline": NEED_WEIGHTS,
    "D": SCENARIO_D_NEED_WEIGHTS,
}


def compute_heat_need(normalized_year: Dict[str, np.ndarray], variant: str = "baseline") -> np.ndarray:
    """Compute Heat Need (0-100) at valid cells for one year.

    Parameters
    ----------
    normalized_year : dict
        0-100 normalized quantities for one year
        (:func:`suitability.normalization.normalize_inputs`).
    variant : str
        ``"baseline"`` (design spec section b) or ``"D"`` (Scenario D
        composition; identical gating exponents to A).

    Returns
    -------
    np.ndarray
        Heat Need per valid cell, asserted within [0, 100].

    Notes
    -----
    NDVI double use across Need (heat stress) and Opportunity (planting
    headroom) is accepted and documented (design spec section l): the two
    uses represent different constructs.
    """
    weights = VARIANT_WEIGHTS[variant]
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise AssertionError(f"Need weights do not sum to 1: {weights}")

    need = np.zeros_like(normalized_year["severity_score"])
    for term, weight in weights.items():
        need += weight * normalized_year[term]

    need = np.clip(need, 0.0, 100.0)
    if need.min() < 0.0 or need.max() > 100.0:
        raise AssertionError(f"Heat Need outside [0, 100]: [{need.min()}, {need.max()}]")
    return need


__all__ = ["VARIANT_WEIGHTS", "compute_heat_need"]
