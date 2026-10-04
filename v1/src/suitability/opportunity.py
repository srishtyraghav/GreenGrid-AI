"""Plantation Opportunity (0-100) — baseline and Scenario D variants.

FROZEN formulation (design spec section c)::

    Opportunity = 0.30*landuse_eligibility
                + 0.25*(100 - n(NDBI))            built-up intensity, inverted
                + 0.15*road_accessibility_band    non-monotonic, FROZEN curve
                + 0.15*green_proximity            500 m cap
                + 0.15*(100 - n(NDVI))            planting headroom, inverted

Static vs dynamic terms (input audit section 3): the landuse-eligibility,
road-accessibility and green-proximity terms come from STATIC rasters and
are identical in 2022 and 2026; year-to-year Opportunity variation comes
ONLY from the NDBI and NDVI terms.  Opportunity change 2022 -> 2026
therefore reflects vegetation/index change on a static land-use /
accessibility base, not observed land-use change — this is stated on every
output.

``dist_building_m`` is deliberately NOT used (buildings are an OSM sample,
not full coverage; a partial distance surface would bias feasibility).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .config import (
    OPPORTUNITY_WEIGHTS,
    SCENARIO_D_OPPORTUNITY_WEIGHTS,
    green_proximity_score,
    landuse_eligibility,
    road_accessibility_score,
)

#: Variant -> (weights dict, headroom quantity).  "baseline" headroom is
#: inverted normalized NDVI; Scenario D upweights headroom computed on
#: inverted normalized vegetation_cover and downweights built-up (FROZEN,
#: spec section h).
VARIANTS = {
    "baseline": (OPPORTUNITY_WEIGHTS, "ndvi"),
    "D": (SCENARIO_D_OPPORTUNITY_WEIGHTS, "vegetation_cover"),
}


def compute_opportunity(
    normalized_year: Dict[str, np.ndarray],
    static: Dict[str, np.ndarray],
    variant: str = "baseline",
) -> Dict[str, np.ndarray]:
    """Compute Plantation Opportunity (0-100) at valid cells.

    Parameters
    ----------
    normalized_year : dict
        0-100 normalized quantities for one year.
    static : dict
        Static per-valid-cell arrays: ``landuse`` (class codes),
        ``dist_road_m``, ``dist_vegetation_m`` (FROZEN static inputs,
        identical for both years).
    variant : str
        ``"baseline"`` (design spec section c) or ``"D"``.

    Returns
    -------
    dict
        ``opportunity`` (0-100, asserted), plus every component term
        (landuse_eligibility, built_up_inverse, road_accessibility,
        green_proximity, planting_headroom) for table provenance.

    Notes
    -----
    Landuse nodata (255) receives the neutral 50, identical to class 0
    (absence of an OSM tag is not evidence of ineligibility; FROZEN
    decision, spec section d).  NDVI/NDBI double use across Need and
    Opportunity is accepted and documented (spec section l): different
    constructs (heat stress vs planting headroom; heat trapping vs built-up
    intensity).  Proximity to existing vegetation is an operational
    accessibility proxy, not a competition measure.
    """
    weights, headroom_var = VARIANTS[variant]
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise AssertionError(f"Opportunity weights do not sum to 1: {weights}")

    components: Dict[str, np.ndarray] = {
        "landuse_eligibility": landuse_eligibility(static["landuse"]),
        "built_up_inverse": 100.0 - normalized_year["ndbi"],
        "road_accessibility": road_accessibility_score(static["dist_road_m"]),
        "green_proximity": green_proximity_score(static["dist_vegetation_m"]),
        "planting_headroom": 100.0 - normalized_year[headroom_var],
    }
    for name, values in components.items():
        if values.min() < 0.0 or values.max() > 100.0:
            raise AssertionError(f"Opportunity component {name} outside [0, 100]")

    opportunity = np.zeros_like(components["landuse_eligibility"])
    for name, weight in weights.items():
        opportunity += weight * components[name]

    opportunity = np.clip(opportunity, 0.0, 100.0)
    if opportunity.min() < 0.0 or opportunity.max() > 100.0:
        raise AssertionError(
            f"Opportunity outside [0, 100]: [{opportunity.min()}, {opportunity.max()}]"
        )
    return {"opportunity": opportunity, **components}


__all__ = ["VARIANTS", "compute_opportunity"]
