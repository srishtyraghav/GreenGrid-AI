"""Phase 7 Stage 2 raster-level analytics tables.

Holds everything that operates directly on the per-valid-cell flat arrays
(rather than on extracted zone objects): the shared Stage 2 context loader,
quintile level labelling (for the "why here?" explanations), the temporal
priority-transition table, the environmental per-class statistics, the
Phase 7 green/built comparison, the per-landuse suitability summary, and
the spatial extension of the Stage 1 sensitivity table.

All statistics are restricted to valid-mask cells (NoData never
participates and is never interpreted).  Areas use the corner-reprojected
per-pixel EPSG:32643 areas from Stage 1 — degree coordinates are NOT
equal-area.  Classes are per-year normalized (snapshots, not trends).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio

from severity.config import classify_green_built

from .config import (
    CLASS_LABELS,
    HIGH_VERY_HIGH_CLASSES,
    LANDUSE_CLASS_NAMES,
    LANDUSE_ELIGIBILITY,
    PHASE6_RASTERS_DIR,
    PHASE7_TABLES_DIR,
    TABLE_PATHS,
    YEARS,
    heat_need_raster_path,
    opportunity_raster_path,
    scenario_raster_path,
    suitability_raster_path,
)
from .inputs import load_inputs
from .pipeline import per_pixel_area_m2
from .scoring import classify

#: Scenarios that receive full Stage 2 spatial products (A is the baseline).
SPATIAL_SCENARIOS = ("A", "B", "C")
#: Scenarios compared spatially against the baseline (sensitivity, spec h).
ALL_SCENARIOS = ("A", "B", "C", "D")

TEMPORAL_TRANSITION_CSV = PHASE7_TABLES_DIR / "temporal_priority_transition.csv"
ENVIRONMENTAL_PRIORITY_CSV = PHASE7_TABLES_DIR / "environmental_priority_statistics.csv"
GREEN_BUILT_CSV = PHASE7_TABLES_DIR / "green_built_comparison.csv"
LANDUSE_SUITABILITY_CSV = PHASE7_TABLES_DIR / "landuse_suitability_summary.csv"
SENSITIVITY_FINDINGS_MD = PHASE7_TABLES_DIR.parent / "reports" / "sensitivity_findings.md"

QUINTILE_LABELS = (
    "Q1_lowest_20pct",
    "Q2_lower_middle",
    "Q3_middle",
    "Q4_upper_middle",
    "Q5_highest_20pct",
)

GREEN_BUILT_NOTE = (
    "ASSOCIATION NOT CAUSATION: green/built groups are operational "
    "remote-sensing thresholds (NDVI>=0.3 green; NDBI>=0.1 & NDVI<0.3 "
    "built; else mixed), not site-calibrated; suitability differences "
    "between groups describe association, not a causal planting effect."
)

LANDUSE_EXCLUSION_NOTE = (
    "excluded_area_ha = 0 by construction: the only hard exclusion is the "
    "invalid mask (valid_mask_30m); no landuse class is excluded from "
    "suitability scoring (spec section e)."
)

PRIORITY_CONFIDENCE_MEANING = (
    "priority_confidence = 100 * mean Phase 6 confidence (0-1, itself an "
    "uncalibrated proxy) * scenario agreement rate (fraction of scenarios "
    "A-D assigning class >= High).  It is an operational stability index, "
    "NOT a probability that trees will grow or that the priority is "
    "'correct' (spec section j)."
)


# ---------------------------------------------------------------------------
# Stage 2 shared context
# ---------------------------------------------------------------------------
def _read_valid_flat(path, valid_mask: np.ndarray, name: str) -> np.ndarray:
    """Read a full-grid raster and return finite values at valid cells."""
    with rasterio.open(path) as src:
        grid = src.read(1)
    flat = grid[valid_mask].astype(np.float64)
    if not np.isfinite(flat).all():
        raise AssertionError(f"{name}: non-finite values inside the valid mask")
    return flat


def load_stage2_context() -> Dict:
    """Load every flat per-valid-cell array the Stage 2 products need.

    Returns
    -------
    dict
        ``data`` (Stage 1 ``load_inputs`` result), per-year flat arrays
        (``scores[scenario][year]``, ``classes[scenario][year]``,
        ``need``, ``opportunity``, ``confidence``), full-grid class
        rasters per scenario/year, per-pixel areas (m2, EPSG:32643
        corner-reprojected), the valid area (ha), the scenario-agreement
        rate per valid cell, and the per-year NDVI normalization bounds
        (for planting-headroom levels).

    Notes
    -----
    Scenario scores are read back from the Stage 1 written rasters and
    classified on the float32-cast values, exactly as Stage 1 wrote the
    class rasters (bit consistency).  Heat Need / Plantation Opportunity
    are read back from the Stage 1 baseline rasters.  The Phase 6
    confidence raster is intersected with the valid mask here.
    """
    data = load_inputs()
    valid_mask = data["valid_mask"]
    rows, cols = data["rows"], data["cols"]
    profile = data["profile"]

    scores: Dict[str, Dict[int, np.ndarray]] = {s: {} for s in ALL_SCENARIOS}
    classes: Dict[str, Dict[int, np.ndarray]] = {s: {} for s in ALL_SCENARIOS}
    class_grids: Dict[str, Dict[int, np.ndarray]] = {s: {} for s in ALL_SCENARIOS}
    need: Dict[int, np.ndarray] = {}
    opportunity: Dict[int, np.ndarray] = {}
    confidence: Dict[int, np.ndarray] = {}

    for year in YEARS:
        for scenario in ALL_SCENARIOS:
            score_path = (
                suitability_raster_path(year)
                if scenario == "A"
                else scenario_raster_path(scenario, year)
            )
            flat = _read_valid_flat(
                score_path, valid_mask,
                f"suitability_scenario_{scenario}_{year}",
            )
            scores[scenario][year] = flat
            cls = classify(flat.astype(np.float32).astype(np.float64))
            classes[scenario][year] = cls
            cls_grid = np.full((profile["height"], profile["width"]), -1, dtype=np.int16)
            cls_grid[rows, cols] = cls
            class_grids[scenario][year] = cls_grid

        need[year] = _read_valid_flat(
            heat_need_raster_path(year), valid_mask, f"heat_need_{year}"
        )
        opportunity[year] = _read_valid_flat(
            opportunity_raster_path(year), valid_mask, f"plantation_opportunity_{year}"
        )
        confidence[year] = _read_valid_flat(
            PHASE6_RASTERS_DIR / f"confidence_{year}.tif", valid_mask,
            f"confidence_{year}",
        )

    print("  Computing per-pixel EPSG:32643 areas ...")
    area_grid = per_pixel_area_m2(profile["transform"], profile["height"], profile["width"])
    pixel_areas_m2 = area_grid[valid_mask]
    valid_area_ha = float(pixel_areas_m2.sum()) / 1e4

    # Scenario agreement rate (spec section j): fraction of scenarios A-D
    # assigning class >= High at each valid cell.
    agreement: Dict[int, np.ndarray] = {}
    for year in YEARS:
        acc = np.zeros(data["n_valid"], dtype=np.float64)
        for scenario in ALL_SCENARIOS:
            acc += (classes[scenario][year] >= min(HIGH_VERY_HIGH_CLASSES)).astype(np.float64)
        agreement[year] = acc / len(ALL_SCENARIOS)

    # Per-year NDVI robust min-max bounds (Stage 1 table) for the planting
    # headroom score (100 - norm(NDVI)) used in "why here?" levels.
    norm_df = pd.read_csv(TABLE_PATHS["normalization_parameters"])
    ndvi_bounds = {
        int(r.year): (float(r.p1), float(r.p99))
        for r in norm_df[norm_df["variable"] == "ndvi"].itertuples()
    }

    return {
        "data": data,
        "valid_mask": valid_mask,
        "rows": rows,
        "cols": cols,
        "profile": profile,
        "transform": profile["transform"],
        "n_valid": data["n_valid"],
        "scores": scores,
        "classes": classes,
        "class_grids": class_grids,
        "need": need,
        "opportunity": opportunity,
        "confidence": confidence,
        "pixel_areas_m2": pixel_areas_m2,
        "valid_area_ha": valid_area_ha,
        "agreement": agreement,
        "ndvi_bounds": ndvi_bounds,
        "yearly": data["yearly"],
        "static": data["static"],
    }


# ---------------------------------------------------------------------------
# Quintile level labelling ("why here?" driver levels)
# ---------------------------------------------------------------------------
def quintile_thresholds(values: np.ndarray) -> np.ndarray:
    """Return the 20/40/60/80th percentile cut points of a valid-cell array."""
    return np.percentile(values, (20.0, 40.0, 60.0, 80.0))


def quintile_level(value: float, thresholds: np.ndarray) -> int:
    """Map a value to a 1-5 quintile index against precomputed cut points."""
    return int(np.searchsorted(thresholds, value, side="right")) + 1


def quintile_label(value: float, thresholds: np.ndarray) -> str:
    """Quintile label (Q1 lowest 20% .. Q5 highest 20%) for a value."""
    return QUINTILE_LABELS[quintile_level(value, thresholds) - 1]


# ---------------------------------------------------------------------------
# Temporal priority transition (spec sections 14/15)
# ---------------------------------------------------------------------------
PERSISTENCE_LABELS = ("Persistent", "Emerging", "Declining", "Stable Low")


def persistence_categories(hvh_2022: np.ndarray, hvh_2026: np.ndarray) -> Dict[str, np.ndarray]:
    """Split paired valid cells into the four FROZEN persistence categories.

    The label set is exhaustive by construction (spec section k): a cell is
    Persistent (>= High both years), Emerging (< High 2022, >= High 2026),
    Declining (>= High 2022, < High 2026) or Stable Low (< High both).
    Per-year-normalized classes make persistence class-relative (snapshots,
    not trends).
    """
    return {
        "Persistent": hvh_2022 & hvh_2026,
        "Emerging": (~hvh_2022) & hvh_2026,
        "Declining": hvh_2022 & (~hvh_2026),
        "Stable Low": (~hvh_2022) & (~hvh_2026),
    }


def build_temporal_transition_table(ctx: Dict) -> pd.DataFrame:
    """Build ``temporal_priority_transition.csv``.

    One row per scenario (A/B/C) per persistence category: paired-cell
    count, area (ha, EPSG:32643 per-pixel areas) and share of valid cells.
    """
    pixel_areas = ctx["pixel_areas_m2"]
    n_valid = ctx["n_valid"]
    rows: List[Dict] = []
    for scenario in SPATIAL_SCENARIOS:
        hvh = {
            year: ctx["classes"][scenario][year] >= min(HIGH_VERY_HIGH_CLASSES)
            for year in YEARS
        }
        cats = persistence_categories(hvh[YEARS[0]], hvh[YEARS[1]])
        for label in PERSISTENCE_LABELS:
            mask = cats[label]
            rows.append(
                {
                    "scenario": scenario,
                    "year_pair": f"{YEARS[0]}-{YEARS[1]}",
                    "persistence_category": label,
                    "pixel_count": int(mask.sum()),
                    "area_ha": round(float(pixel_areas[mask].sum()) / 1e4, 3),
                    "pct_of_valid": round(100.0 * mask.sum() / n_valid, 4),
                    "note": (
                        "per-year normalized classes; persistence is "
                        "class-relative (snapshots, not trends)"
                    ),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Environmental priority statistics (baseline A rasters + source rasters)
# ---------------------------------------------------------------------------
def build_environmental_priority_table(ctx: Dict) -> pd.DataFrame:
    """Build ``environmental_priority_statistics.csv``.

    Per year x suitability class (baseline Scenario A): mean/median LST,
    NDVI, NDBI, vegetation_cover, Phase 6 severity_score and Phase 6
    confidence over the class's valid cells.
    """
    rows: List[Dict] = []
    for year in YEARS:
        classes = ctx["classes"]["A"][year]
        src = ctx["yearly"][year]
        arrays = {
            "lst_C": src["lst"],
            "ndvi": src["ndvi"],
            "ndbi": src["ndbi"],
            "vegetation_cover": src["vegetation_cover"],
            "severity_score": src["severity_score"],
            "confidence": ctx["confidence"][year],
        }
        for cls, label in CLASS_LABELS.items():
            sel = classes == cls
            row: Dict = {
                "year": year,
                "class_code": cls,
                "class_label": label,
                "pixel_count": int(sel.sum()),
            }
            for name, arr in arrays.items():
                vals = arr[sel]
                if vals.size:
                    row[f"mean_{name}"] = round(float(vals.mean()), 6)
                    row[f"median_{name}"] = round(float(np.median(vals)), 6)
                else:
                    row[f"mean_{name}"] = np.nan
                    row[f"median_{name}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Green / built comparison (Phase 6 operational classification, reused)
# ---------------------------------------------------------------------------
def build_green_built_table(ctx: Dict) -> pd.DataFrame:
    """Build the Phase 7 ``green_built_comparison.csv``.

    Reuses the FROZEN Phase 6 operational classification
    (``severity.config.classify_green_built``: NDVI >= 0.3 green-dominant;
    NDBI >= 0.1 and NDVI < 0.3 built-dominant; else other-mixed) computed on
    valid cells, per year x group: pixel count, area, mean suitability /
    need / opportunity and the High+Very High share (baseline A classes).
    """
    pixel_areas = ctx["pixel_areas_m2"]
    n_valid = ctx["n_valid"]
    rows: List[Dict] = []
    for year in YEARS:
        gb = classify_green_built(ctx["yearly"][year]["ndvi"], ctx["yearly"][year]["ndbi"])
        hvh = ctx["classes"]["A"][year] >= min(HIGH_VERY_HIGH_CLASSES)
        for group in ("green_dominant", "built_dominant", "other_mixed"):
            sel = gb == group
            rows.append(
                {
                    "year": year,
                    "group": group,
                    "pixel_count": int(sel.sum()),
                    "area_ha": round(float(pixel_areas[sel].sum()) / 1e4, 3),
                    "pct_of_valid": round(100.0 * sel.sum() / n_valid, 4),
                    "mean_suitability": round(float(ctx["scores"]["A"][year][sel].mean()), 4),
                    "mean_heat_need": round(float(ctx["need"][year][sel].mean()), 4),
                    "mean_opportunity": round(float(ctx["opportunity"][year][sel].mean()), 4),
                    "high_very_high_pct": round(100.0 * hvh[sel].mean(), 4),
                    "note": GREEN_BUILT_NOTE,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-landuse suitability summary (Phase 7 report section 28)
# ---------------------------------------------------------------------------
def build_landuse_suitability_table(ctx: Dict) -> pd.DataFrame:
    """Build ``landuse_suitability_summary.csv``.

    Per landuse class per year (baseline Scenario A, valid cells only):
    pixel count, area, mean/median suitability, High and Very High area,
    excluded area (0 — no hard landuse exclusions exist, spec section e),
    mean opportunity and mean heat need.  The landuse raster is STATIC and
    is reused for both years (documented; the class-0 background is the
    absence of an OSM tag, not a real observation).
    """
    pixel_areas = ctx["pixel_areas_m2"]
    landuse = ctx["static"]["landuse"]
    rows: List[Dict] = []
    for year in YEARS:
        classes = ctx["classes"]["A"][year]
        for code in sorted(LANDUSE_CLASS_NAMES):
            sel = landuse == code
            n_px = int(sel.sum())
            if n_px == 0:
                continue
            high = sel & (classes == 3)
            very_high = sel & (classes == 4)
            rows.append(
                {
                    "year": year,
                    "landuse_code": code,
                    "landuse_name": LANDUSE_CLASS_NAMES[code],
                    "eligibility_score": LANDUSE_ELIGIBILITY[code],
                    "pixel_count": n_px,
                    "area_ha": round(float(pixel_areas[sel].sum()) / 1e4, 3),
                    "mean_suitability": round(float(ctx["scores"]["A"][year][sel].mean()), 4),
                    "median_suitability": round(
                        float(np.median(ctx["scores"]["A"][year][sel])), 4
                    ),
                    "high_area_ha": round(float(pixel_areas[high].sum()) / 1e4, 3),
                    "very_high_area_ha": round(float(pixel_areas[very_high].sum()) / 1e4, 3),
                    "excluded_area_ha": 0.0,
                    "mean_opportunity": round(float(ctx["opportunity"][year][sel].mean()), 4),
                    "mean_heat_need": round(float(ctx["need"][year][sel].mean()), 4),
                    "note": LANDUSE_EXCLUSION_NOTE if code == 0 else "",
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "SPATIAL_SCENARIOS",
    "ALL_SCENARIOS",
    "TEMPORAL_TRANSITION_CSV",
    "ENVIRONMENTAL_PRIORITY_CSV",
    "GREEN_BUILT_CSV",
    "LANDUSE_SUITABILITY_CSV",
    "SENSITIVITY_FINDINGS_MD",
    "QUINTILE_LABELS",
    "GREEN_BUILT_NOTE",
    "LANDUSE_EXCLUSION_NOTE",
    "PRIORITY_CONFIDENCE_MEANING",
    "PERSISTENCE_LABELS",
    "load_stage2_context",
    "quintile_thresholds",
    "quintile_level",
    "quintile_label",
    "persistence_categories",
    "build_temporal_transition_table",
    "build_environmental_priority_table",
    "build_green_built_table",
    "build_landuse_suitability_table",
]
