"""Frozen V1 suitability formulation — VERBATIM ports.

Every constant and formula below is copied from ``v1/src/suitability/config.py``
/ ``need.py`` / ``opportunity.py`` / ``normalization.py`` / ``scoring.py`` /
``zones.py`` (frozen by the V1 design spec, 2026-09-04; weights predefined,
not tuned). Only the baseline composition is computed (V1 Scenario A);
V1's B/C/D sensitivity scenarios are outside the V2 brief.

Domain rule (V1 spec locked decision #1): per-year Phase 6 ``severity_score``
valid pixels. Scores are class-relative per-year (p1/p99 min-max) —
snapshots, not trends.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# FROZEN constants (v1/src/suitability/config.py)
# ---------------------------------------------------------------------------
NORM_PERCENTILES = (1.0, 99.0)

NEED_WEIGHTS = {                     # design spec section b
    "severity_score": 0.40,
    "one_minus_ndvi": 0.25,
    "ndbi": 0.20,
    "lst": 0.15,
}
OPPORTUNITY_WEIGHTS = {              # design spec section c
    "landuse_eligibility": 0.30,
    "built_up_inverse": 0.25,
    "road_accessibility": 0.15,
    "green_proximity": 0.15,
    "planting_headroom": 0.15,
}
ROAD_BAND = {                        # piecewise-linear (spec c, Option A)
    "corridor_max_m": 50.0,
    "corridor_score_max": 40.0,
    "optimal_max_m": 500.0,
    "optimal_score": 100.0,
    "decline_max_m": 2000.0,
    "decline_score": 30.0,
}
GREEN_PROXIMITY_CAP_M = 500.0        # 100 * max(0, 1 - dist/500)
LANDUSE_ELIGIBILITY = {              # spec section d (0=untagged -> 50 neutral)
    0: 50, 1: 40, 2: 30, 3: 50, 4: 35, 5: 20, 6: 90, 7: 30, 8: 60,
}
LANDUSE_NODATA = 255
LANDUSE_NODATA_ELIGIBILITY = 50

CLASS_EDGES = (20.0, 40.0, 60.0, 80.0)   # boundaries belong to the LOWER class
CLASS_LABELS = {0: "Very Low", 1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
MIN_ZONE_PIXELS = 10                 # 8-connected, ~0.8 ha at 30 m (spec i)
ZONE_MIN_CLASS = 3                   # High + Very High (1-indexed classes 4-5)
SCORE_NODATA = -1.0
CLASS_NODATA = -1

PX_AREA_M2 = 900.0
M2_PER_HA = 10_000.0

TERMINOLOGY_NOTE = (
    "Outputs describe POTENTIAL PLANTATION SUITABILITY - a relative, "
    "decision-support ranking on per-year-normalized inputs. They are NOT a "
    "statement of legal availability, land ownership, or a physical "
    "plantability guarantee, and make no claim of planting survival "
    "probability."
)


# ---------------------------------------------------------------------------
# Normalization (normalization.py)
# ---------------------------------------------------------------------------
def robust_minmax(values: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Clip to p1/p99 over the year's domain cells, scale to 0-100."""
    p_low, p_high = NORM_PERCENTILES
    p1, p99 = np.percentile(values, (p_low, p_high))
    if not p99 > p1:
        raise AssertionError(f"degenerate percentile range: p1={p1}, p99={p99}")
    scaled = (np.clip(values, p1, p99) - p1) / (p99 - p1) * 100.0
    return np.clip(scaled, 0.0, 100.0), {"p1": float(p1), "p99": float(p99),
                                         "n_cells": int(values.size)}


def pooled_p1_p99(samples: np.ndarray) -> Tuple[float, float]:
    """Fixed pooled reference from concatenated valid samples (all years)."""
    p1, p99 = np.percentile(samples, NORM_PERCENTILES)
    if not p99 > p1:
        raise AssertionError(f"degenerate pooled range: p1={p1}, p99={p99}")
    return float(p1), float(p99)


def pooled_normalize(values: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """Clip to the FIXED pooled reference and scale to [0, 1] (NaN preserved)."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    m = np.isfinite(values)
    out[m] = np.clip((np.clip(values[m], p1, p99) - p1) / (p99 - p1), 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Component scores (config.py)
# ---------------------------------------------------------------------------
def road_accessibility_score(dist_m: np.ndarray) -> np.ndarray:
    """0->40 ramp over [0,50) m; 100 over [50,500]; 100->30 over (500,2000];
    floor 30 beyond. FROZEN curve, documented not site-calibrated."""
    s = ROAD_BAND
    score = np.full(dist_m.shape, s["decline_score"], dtype=np.float64)
    decline = (dist_m > s["optimal_max_m"]) & (dist_m < s["decline_max_m"])
    score[decline] = s["optimal_score"] - (
        (s["optimal_score"] - s["decline_score"])
        * (dist_m[decline] - s["optimal_max_m"])
        / (s["decline_max_m"] - s["optimal_max_m"]))
    optimal = (dist_m >= s["corridor_max_m"]) & (dist_m <= s["optimal_max_m"])
    score[optimal] = s["optimal_score"]
    corridor = dist_m < s["corridor_max_m"]
    score[corridor] = s["corridor_score_max"] * dist_m[corridor] / s["corridor_max_m"]
    return score


def green_proximity_score(dist_m: np.ndarray) -> np.ndarray:
    """100 * max(0, 1 - dist/500). FROZEN (spec c)."""
    return 100.0 * np.clip(1.0 - dist_m / GREEN_PROXIMITY_CAP_M, 0.0, 1.0)


def landuse_eligibility(landuse: np.ndarray) -> np.ndarray:
    """Class codes -> 0-100 eligibility; nodata(255) neutral 50 like class 0."""
    lookup = np.full(256, LANDUSE_NODATA_ELIGIBILITY, dtype=np.float64)
    for code, score in LANDUSE_ELIGIBILITY.items():
        lookup[code] = score
    return lookup[landuse.astype(np.int64)]


# ---------------------------------------------------------------------------
# Need / Opportunity (need.py / opportunity.py, baseline variants)
# ---------------------------------------------------------------------------
def compute_heat_need(norm: Dict[str, np.ndarray]) -> np.ndarray:
    """Need = 0.40*n(severity_score) + 0.25*n(1-NDVI) + 0.20*n(NDBI)
    + 0.15*n(LST); clipped to [0, 100]. vegetation_cover EXCLUDED from the
    baseline (Phase 5 froze NDVI-vegetation_cover redundancy)."""
    w = NEED_WEIGHTS
    if abs(sum(w.values()) - 1.0) > 1e-12:
        raise AssertionError("Need weights do not sum to 1")
    need = np.zeros_like(norm["severity_score"])
    for term, weight in w.items():
        need += weight * norm[term]
    need = np.clip(need, 0.0, 100.0)
    if need.min() < 0.0 or need.max() > 100.0:
        raise AssertionError("Heat Need outside [0, 100]")
    return need


def compute_opportunity(norm: Dict[str, np.ndarray],
                        static: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Opportunity = 0.30*landuse_eligibility + 0.25*(100-n(NDBI))
    + 0.15*road_accessibility + 0.15*green_proximity + 0.15*(100-n(NDVI));
    clipped to [0, 100]. Static terms are year-invariant by construction."""
    w = OPPORTUNITY_WEIGHTS
    if abs(sum(w.values()) - 1.0) > 1e-12:
        raise AssertionError("Opportunity weights do not sum to 1")
    components = {
        "landuse_eligibility": landuse_eligibility(static["landuse"]),
        "built_up_inverse": 100.0 - norm["ndbi"],
        "road_accessibility": road_accessibility_score(static["dist_road_m"]),
        "green_proximity": green_proximity_score(static["dist_vegetation_m"]),
        "planting_headroom": 100.0 - norm["ndvi"],
    }
    for name, values in components.items():
        if values.min() < 0.0 or values.max() > 100.0:
            raise AssertionError(f"Opportunity component {name} outside [0, 100]")
    opportunity = np.zeros_like(components["landuse_eligibility"])
    for name, weight in w.items():
        opportunity += weight * components[name]
    opportunity = np.clip(opportunity, 0.0, 100.0)
    if opportunity.min() < 0.0 or opportunity.max() > 100.0:
        raise AssertionError("Opportunity outside [0, 100]")
    return {"opportunity": opportunity, **components}


# ---------------------------------------------------------------------------
# Scoring / classes (scoring.py)
# ---------------------------------------------------------------------------
def gated_product(need: np.ndarray, opportunity: np.ndarray) -> np.ndarray:
    """Baseline suitability = Need x Opportunity / 100 (gated: Opp=0 -> 0)."""
    return np.clip(need.astype(np.float64) * opportunity.astype(np.float64)
                   / 100.0, 0.0, 100.0)


def classify(score: np.ndarray) -> np.ndarray:
    """0-100 score -> classes 0-4; edges 20/40/60/80 belong to the LOWER
    class (searchsorted side='left')."""
    classes = np.searchsorted(CLASS_EDGES, score, side="left")
    return np.clip(classes, 0, len(CLASS_LABELS) - 1).astype(np.int16)


def priority_tier(cls: np.ndarray) -> np.ndarray:
    """V2 priority tiers from the frozen 5-class scheme: 3=High (class 4 Very
    High), 2=Medium (class 3 High), 1=Low (class 2 Medium), 0=not priority
    (classes 0-1)."""
    return np.where(cls >= 4, 3, np.where(cls == 3, 2,
                                          np.where(cls == 2, 1, 0))).astype(np.uint8)


# ---------------------------------------------------------------------------
# Zones (zones.py)
# ---------------------------------------------------------------------------
_MASK8 = np.ones((3, 3), dtype=int)


def label_zones(binary_grid: np.ndarray,
                min_pixels: int = MIN_ZONE_PIXELS) -> Tuple[np.ndarray, int]:
    """8-connected zones; clusters < min_pixels dropped; dense IDs 1..n."""
    labels, n_found = ndimage.label(binary_grid, structure=_MASK8)
    if n_found == 0:
        return labels.astype(np.int32), 0
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    remap = np.zeros(n_found + 1, dtype=np.int32)
    remap[np.where(keep)[0]] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[labels].astype(np.int32), int(keep.sum())


def polygons_per_zone(labelled: np.ndarray, transform) -> Dict[int, object]:
    """Dissolve pixel polygons per zone ID (4-connectivity shapes + union);
    EPSG:4326. Exact unions; validity is checked, never silently repaired."""
    from rasterio.features import shapes as raster_shapes
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    per_id: Dict[int, list] = {}
    for geom, value in raster_shapes(labelled, mask=labelled > 0,
                                     transform=transform, connectivity=4):
        per_id.setdefault(int(value), []).append(shapely_shape(geom))
    return {zid: unary_union(gs) for zid, gs in per_id.items()}


def validate_geometries(polygons: Dict[int, object]) -> Dict:
    invalid = [z for z, g in polygons.items() if not g.is_valid]
    zero_area = [z for z, g in polygons.items() if g.area <= 0.0]
    ids = list(polygons)
    return {
        "n_polygons": len(ids),
        "n_invalid": len(invalid), "invalid_ids": invalid,
        "n_zero_area": len(zero_area), "zero_area_ids": zero_area,
        "duplicate_ids": len(ids) != len(set(ids)),
        "all_valid": (not invalid and not zero_area
                      and len(ids) == len(set(ids))),
    }


# ---------------------------------------------------------------------------
# v3 methodology (fixed pooled references — cross-year comparable)
# ---------------------------------------------------------------------------
# Cooling need (0-1), LST-led, INDEPENDENT of Phase 6: the V1 need's
# severity_score term (itself trained on per-year LST tertiles) is replaced
# by direct pooled-LST normalization; the V1 non-severity weights (0.25
# 1-NDVI, 0.20 NDBI) are kept and renormalized to sum to 1.
COOLING_NEED_WEIGHTS = {"lst": 0.55, "one_minus_ndvi": 0.25, "ndbi": 0.20}

# Plantation opportunity component (0-1): V1 opportunity weights VERBATIM
# (v1/src/suitability/config.py OPPORTUNITY_WEIGHTS), rebased from 0-100 to
# 0-1 with pooled references for NDVI/NDBI and frozen static curves.
OPPORTUNITY_WEIGHTS_01 = {
    "landuse_eligibility": 0.30,
    "built_up_inverse": 0.25,
    "road_accessibility": 0.15,
    "green_proximity": 0.15,
    "planting_headroom": 0.15,
}

# Priority (separate from suitability): equal-weight documented combination,
# computed over feasible land only.
PRIORITY_WEIGHTS = {"cooling_need": 0.5, "suitability": 0.5}
PRIORITY_CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High"}


def cooling_need(lst_n, one_minus_ndvi_n, ndbi_n) -> np.ndarray:
    """0-1 cooling-need score from POOLED-normalized inputs (NaN preserved)."""
    w = COOLING_NEED_WEIGHTS
    out = (w["lst"] * lst_n + w["one_minus_ndvi"] * one_minus_ndvi_n
           + w["ndbi"] * ndbi_n)
    return np.clip(out, 0.0, 1.0)


def opportunity_component_01(norm: Dict[str, np.ndarray],
                             static: Dict[str, np.ndarray]) -> np.ndarray:
    """V1 opportunity weights verbatim, 0-1: 0.30*landuse_eligibility +
    0.25*(1-n(NDBI)) + 0.15*road_band + 0.15*green_proximity +
    0.15*(1-n(NDVI)); landuse nodata(255) neutral 50/100=0.5 (class 0)."""
    w = OPPORTUNITY_WEIGHTS_01
    components = {
        "landuse_eligibility": landuse_eligibility(static["landuse"]) / 100.0,
        "built_up_inverse": 1.0 - norm["ndbi"],
        "road_accessibility": road_accessibility_score(static["dist_road_m"]) / 100.0,
        "green_proximity": green_proximity_score(static["dist_vegetation_m"]) / 100.0,
        "planting_headroom": 1.0 - norm["ndvi"],
    }
    for name, values in components.items():
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size == 0 or finite_vals.min() < 0.0 or finite_vals.max() > 1.0:
            raise AssertionError(f"opportunity component {name} outside [0,1]")
    opp = np.zeros_like(components["landuse_eligibility"])
    for name, weight in w.items():
        opp += weight * components[name]
    return np.clip(opp, 0.0, 1.0)


def suitability_product(need01: np.ndarray, opp01: np.ndarray) -> np.ndarray:
    """V1 gated product rebased: suitability = cooling_need x opportunity
    (0-1). Multiplicative gate: zero opportunity -> zero suitability."""
    return np.clip(need01 * opp01, 0.0, 1.0)


def classify_priority(priority_score: np.ndarray, t1: float, t2: float
                      ) -> np.ndarray:
    """FIXED thresholds (pooled terciles over feasible land): Low < t1 <=
    Medium < t2 <= High. NaN stays -1 (excluded pixels carry no priority)."""
    cls = np.full(priority_score.shape, -1, dtype=np.int16)
    m = np.isfinite(priority_score)
    cls[m & (priority_score < t1)] = 0
    cls[m & (priority_score >= t1) & (priority_score < t2)] = 1
    cls[m & (priority_score >= t2)] = 2
    return cls


def priority_rationale(cls: int, need: float, suit: float,
                       veg: float) -> str:
    band = PRIORITY_CLASS_LABELS.get(cls, "None")
    return (f"priority={band} (pooled fixed bands; need={need:.2f}, "
            f"suitability={suit:.2f}, vegetation_cover={veg:.2f})")
