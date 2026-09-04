"""End-to-end Phase 7 Stage 1 pipeline: core suitability scoring.

Orchestrates:

1. Input loading and valid-mask intersection (every raw raster covers more
   pixels than the valid mask — audit sections 3/5).
2. Per-year robust min-max normalization (FROZEN, spec section f) with a
   recorded bounds table.
3. Heat Need and Plantation Opportunity per year (baseline composition,
   plus the Scenario D internal composition).
4. Hard-exclusion provenance raster (valid mask only — spec section e).
5. Baseline suitability (Scenario A gated product) + Scenarios B/C/D
   (FROZEN weighted geometric means), with class rasters.
6. Summary / area / dependency / rules / sensitivity tables.
7. Assertions (score ranges, class-score consistency, valid-pixel count,
   NoData never scored), manifest and pipeline record.

The additive diagnostic 0.5*Need + 0.5*Opportunity is stored in the summary
table only — it is a comparison diagnostic, never the primary map (spec
section a).  Per-component diagnostic rasters are deliberately NOT written;
components live in the tables.

Run with::

    PYTHONPATH=src .venv/bin/python -m suitability.pipeline
    PYTHONPATH=src .venv/bin/python -m suitability.pipeline --skip-rasters --skip-tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio

from .config import (
    CLASS_EDGES,
    CLASS_LABELS,
    EXPECTED_VALID_PIXELS,
    HIGH_VERY_HIGH_CLASSES,
    LANDUSE_CLASS_NAMES,
    LANDUSE_ELIGIBILITY,
    LANDUSE_RULE_RATIONALE,
    LANDUSE_RULE_STATUS,
    LANDUSE_RASTER_PATH,
    LST_RASTERS,
    MANIFEST_JSON,
    NEED_WEIGHTS,
    NDBI_RASTERS,
    NDVI_RASTERS,
    NORM_PERCENTILES,
    OPPORTUNITY_WEIGHTS,
    PHASE7_DIR,
    PIPELINE_RECORD_JSON,
    PROJECT_ROOT,
    RANDOM_SEED,
    ROADS_DISTANCE_RASTER,
    SCENARIO_D_NEED_WEIGHTS,
    SCENARIO_D_OPPORTUNITY_WEIGHTS,
    SCENARIO_EXPONENTS,
    SEVERITY_SCORE_RASTERS,
    TABLE_PATHS,
    TERMINOLOGY_NOTE,
    UTM_CRS,
    VALID_MASK_RASTER,
    VEGETATION_COVER_RASTERS,
    VEGETATION_DISTANCE_RASTER,
    YEARS,
    heat_need_raster_path,
    opportunity_raster_path,
    scenario_raster_path,
    suitability_class_raster_path,
    suitability_raster_path,
)
from .constraints import build_exclusion_mask
from .inputs import load_inputs
from .need import compute_heat_need
from .normalization import normalize_inputs
from .opportunity import compute_opportunity
from .scoring import (
    additive_diagnostic,
    assert_class_consistency,
    assert_class_partition,
    classify,
    compute_scenario_scores,
    write_class_raster,
    write_score_raster,
)

#: Feature-dependency table (FROZEN, design spec section l).
FEATURE_DEPENDENCY_ROWS = [
    {
        "variable": "severity_score",
        "role": "Need (0.40)",
        "correlated_variables": "LST (trained on LST quartiles); NDVI/NDBI indirectly",
        "treatment": "Primary term; LST weight capped at 0.15 to limit double-count",
    },
    {
        "variable": "LST",
        "role": "Need (0.15)",
        "correlated_variables": "severity_score",
        "treatment": "Reduced weight; intersect valid mask first (coverage > valid mask)",
    },
    {
        "variable": "NDVI",
        "role": "Need (0.25 as 1-NDVI); Opportunity (0.15 headroom)",
        "correlated_variables": "vegetation_cover r~0.999",
        "treatment": (
            "vegetation_cover excluded from baseline; double use across "
            "Need/Opportunity accepted & documented (different constructs: "
            "heat stress vs planting headroom)"
        ),
    },
    {
        "variable": "vegetation_cover",
        "role": "Scenario D only",
        "correlated_variables": "NDVI",
        "treatment": "Baseline exclusion is a frozen Phase 5 redundancy decision",
    },
    {
        "variable": "NDBI",
        "role": "Need (0.20); Opportunity (0.25 as 1-NDBI)",
        "correlated_variables": "-",
        "treatment": (
            "Double use accepted & documented (heat-trapping vs built-up intensity)"
        ),
    },
    {
        "variable": "dist_road_m",
        "role": "Opportunity (0.15, non-monotonic band)",
        "correlated_variables": "-",
        "treatment": "Static; same both years",
    },
    {
        "variable": "dist_vegetation_m",
        "role": "Opportunity (0.15, green proximity, 500 m cap)",
        "correlated_variables": "NDVI",
        "treatment": "Static; proximity != competition, documented",
    },
    {
        "variable": "dist_building_m",
        "role": "None (diagnostic only)",
        "correlated_variables": "dist_roads",
        "treatment": "Excluded: OSM sample, not full coverage",
    },
    {
        "variable": "landuse raster",
        "role": "Opportunity (0.30 eligibility)",
        "correlated_variables": "NDBI (association)",
        "treatment": (
            "Static rules matrix (spec d); class 0 = 84% of grid treated as "
            "neutral, flagged in report"
        ),
    },
    {
        "variable": "valid mask",
        "role": "Constraint (only hard exclusion)",
        "correlated_variables": "-",
        "treatment": "All stats restricted to valid px; NoData never interpreted",
    },
    {
        "variable": "Phase 6 confidence / uncertainty_zone",
        "role": "Diagnostic (spec j)",
        "correlated_variables": "severity",
        "treatment": "Uncalibrated proxy, labelled as such",
    },
    {
        "variable": "probability_low...severe",
        "role": "Diagnostic",
        "correlated_variables": "severity_score",
        "treatment": "Not used in baseline (already aggregated into severity_score)",
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _convert_for_json(obj: object) -> object:
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert_for_json(obj), f, indent=2, default=str)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Equal-area per-pixel areas (EPSG:32643)
# ---------------------------------------------------------------------------
def per_pixel_area_m2(transform, height: int, width: int) -> np.ndarray:
    """Per-pixel areas (m2) in UTM 43N, computed by corner reprojection.

    Degree coordinates are NOT equal-area, so hectares are never derived
    from degree widths (same convention as Phase 6).  Each pixel's four
    geographic corners are reprojected to EPSG:32643 and the planar
    quadrilateral area is taken by the shoelace formula.

    Returns
    -------
    np.ndarray
        (height, width) float64 per-pixel area in m2.
    """
    import pyproj

    corner_cols, corner_rows = np.meshgrid(
        np.arange(width + 1, dtype=np.float64),
        np.arange(height + 1, dtype=np.float64),
    )
    lons, lats = rasterio.transform.xy(transform, corner_rows, corner_cols)
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", UTM_CRS, always_xy=True
    )
    xs, ys = transformer.transform(np.asarray(lons), np.asarray(lats))
    xs = np.asarray(xs).reshape(height + 1, width + 1)
    ys = np.asarray(ys).reshape(height + 1, width + 1)

    x00, x10 = xs[:-1, :-1], xs[1:, :-1]
    x01, x11 = xs[:-1, 1:], xs[1:, 1:]
    y00, y10 = ys[:-1, :-1], ys[1:, :-1]
    y01, y11 = ys[:-1, 1:], ys[1:, 1:]
    # Shoelace over the corner order (00, 10, 11, 01).
    area = 0.5 * np.abs(
        x00 * y10 - x10 * y00
        + x10 * y11 - x11 * y10
        + x11 * y01 - x01 * y11
        + x01 * y00 - x00 * y01
    )
    return area


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _class_stats(classes: np.ndarray, pixel_areas_m2: np.ndarray, n_valid: int) -> Dict[int, Dict]:
    """Per-class pixel count, area (ha) and share of valid cells."""
    stats: Dict[int, Dict] = {}
    for cls in range(len(CLASS_LABELS)):
        mask = classes == cls
        n_px = int(mask.sum())
        area_ha = float(pixel_areas_m2[mask].sum()) / 1e4
        stats[cls] = {
            "count": n_px,
            "area_ha": round(area_ha, 3),
            "pct_valid": round(100.0 * n_px / n_valid, 4),
        }
    return stats


def build_summary_table(
    per_year: Dict[int, Dict],
    pixel_areas_flat_m2: np.ndarray,
    n_valid: int,
) -> pd.DataFrame:
    """Build ``suitability_summary.csv`` — one wide row per year."""
    rows: List[Dict] = []
    for year in YEARS:
        py = per_year[year]
        classes = py["baseline_classes"]
        cls_stats = _class_stats(classes, pixel_areas_flat_m2, n_valid)
        hvh_mask = np.isin(classes, HIGH_VERY_HIGH_CLASSES)
        row: Dict = {
            "year": year,
            "valid_pixel_count": int(n_valid),
            "valid_area_ha": round(float(pixel_areas_flat_m2.sum()) / 1e4, 3),
            "need_mean": round(float(py["need"].mean()), 4),
            "need_median": round(float(np.median(py["need"])), 4),
            "opportunity_mean": round(float(py["opportunity"].mean()), 4),
            "opportunity_median": round(float(np.median(py["opportunity"])), 4),
            "suitability_mean": round(float(py["baseline"].mean()), 4),
            "suitability_median": round(float(np.median(py["baseline"])), 4),
            # Additive diagnostic — comparison ONLY, never the primary map.
            "additive_diagnostic_mean": round(float(py["additive"].mean()), 4),
            "additive_diagnostic_median": round(float(np.median(py["additive"])), 4),
        }
        for cls in range(len(CLASS_LABELS)):
            label = CLASS_LABELS[cls].lower().replace(" ", "_")
            row[f"class_{cls}_{label}_count"] = cls_stats[cls]["count"]
            row[f"class_{cls}_{label}_pct"] = cls_stats[cls]["pct_valid"]
            row[f"class_{cls}_{label}_area_ha"] = cls_stats[cls]["area_ha"]
        row["high_very_high_count"] = int(hvh_mask.sum())
        row["high_very_high_pct"] = round(100.0 * hvh_mask.sum() / n_valid, 4)
        row["high_very_high_area_ha"] = round(
            float(pixel_areas_flat_m2[hvh_mask].sum()) / 1e4, 3
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_area_statistics_table(
    per_year: Dict[int, Dict], pixel_areas_flat_m2: np.ndarray, n_valid: int
) -> pd.DataFrame:
    """Build ``suitability_area_statistics.csv`` — one row per class per year."""
    rows: List[Dict] = []
    for year in YEARS:
        classes = per_year[year]["baseline_classes"]
        for cls, stats in _class_stats(classes, pixel_areas_flat_m2, n_valid).items():
            rows.append(
                {
                    "year": year,
                    "class_code": cls,
                    "class_label": CLASS_LABELS[cls],
                    "pixel_count": stats["count"],
                    "area_ha": stats["area_ha"],
                    "pct_valid": stats["pct_valid"],
                }
            )
    return pd.DataFrame(rows)


def build_sensitivity_table(
    scores: Dict[str, Dict[int, np.ndarray]],
    pixel_areas_flat_m2: np.ndarray,
    n_valid: int,
) -> pd.DataFrame:
    """Build ``sensitivity_analysis.csv``.

    One ``per_scenario`` row per scenario/year (class counts/areas,
    High+Very-High area, mean suitability) and one ``overlap_vs_baseline``
    row per scenario B/C/D per year (High+VH mask IoU vs Scenario A at
    valid cells; full spatial zone statistics are a Stage 2 product).
    """
    rows: List[Dict] = []
    for year in YEARS:
        baseline_hvh = classify(scores["A"][year]) >= min(HIGH_VERY_HIGH_CLASSES)
        for scenario in ("A", "B", "C", "D"):
            final = scores[scenario][year]
            classes = classify(final)
            cls_stats = _class_stats(classes, pixel_areas_flat_m2, n_valid)
            hvh_mask = np.isin(classes, HIGH_VERY_HIGH_CLASSES)
            row: Dict = {
                "section": "per_scenario",
                "scenario": scenario,
                "year": year,
                "mean_suitability": round(float(final.mean()), 4),
                "high_very_high_px": int(hvh_mask.sum()),
                "high_very_high_area_ha": round(
                    float(pixel_areas_flat_m2[hvh_mask].sum()) / 1e4, 3
                ),
                "high_very_high_pct": round(100.0 * hvh_mask.sum() / n_valid, 4),
            }
            for cls in range(len(CLASS_LABELS)):
                row[f"class_{cls}_count"] = cls_stats[cls]["count"]
                row[f"class_{cls}_area_ha"] = cls_stats[cls]["area_ha"]
            rows.append(row)

            if scenario != "A":
                inter = int((hvh_mask & baseline_hvh).sum())
                union = int((hvh_mask | baseline_hvh).sum())
                n_scen = int(hvh_mask.sum())
                n_base = int(baseline_hvh.sum())
                rows.append(
                    {
                        "section": "overlap_vs_baseline",
                        "scenario": scenario,
                        "year": year,
                        "mean_suitability": np.nan,
                        "high_very_high_px": None,
                        "high_very_high_area_ha": None,
                        "high_very_high_pct": None,
                        "intersection_px": inter,
                        "union_px": union,
                        "iou": round(inter / union, 6) if union else np.nan,
                        "pct_of_scenario_hvh_covered_by_baseline": (
                            round(100.0 * inter / n_scen, 4) if n_scen else np.nan
                        ),
                        "pct_of_baseline_hvh_covered_by_scenario": (
                            round(100.0 * inter / n_base, 4) if n_base else np.nan
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    for col in [c for c in df.columns if c.endswith(("_px", "_count"))]:
        df[col] = df[col].astype("Int64")
    return df


def build_landuse_rules_table() -> pd.DataFrame:
    """Build ``landuse_suitability_rules.csv`` (FROZEN matrix, spec section d)."""
    rows = [
        {
            "class_code": code,
            "meaning": LANDUSE_CLASS_NAMES[code],
            "eligibility_score": LANDUSE_ELIGIBILITY[code],
            "status": LANDUSE_RULE_STATUS[code],
            "rationale": LANDUSE_RULE_RATIONALE[code],
        }
        for code in sorted(LANDUSE_CLASS_NAMES)
    ]
    rows.append(
        {
            "class_code": 255,
            "meaning": "nodata",
            "eligibility_score": 50,
            "status": "CONDITIONAL (UNCERTAIN)",
            "rationale": (
                "Mapped to the same neutral 50 as class 0 (absence of an OSM "
                "tag is not evidence of ineligibility)."
            ),
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_written_rasters(
    per_year: Dict[int, Dict], scores: Dict[str, Dict[int, np.ndarray]], n_check: int = 5000
) -> Dict:
    """Re-open written rasters and assert score/class/nodata consistency.

    - score rasters: flat values equal the float32-cast source within
      exact float32 equality; NoData exactly outside the valid mask.
    - class rasters: equal ``classify(score_raster)`` at every valid cell
      (bit consistency with the written float32 score raster).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    checks: Dict[str, object] = {}

    for year in YEARS:
        py = per_year[year]
        with rasterio.open(suitability_raster_path(year)) as src:
            score_grid = src.read(1)
        with rasterio.open(suitability_class_raster_path(year)) as src:
            class_grid = src.read(1)
        valid_mask = py["valid_mask"]

        # NoData never scored: outside the valid mask everything is nodata.
        nodata_ok = bool(
            (score_grid[~valid_mask] == -1.0).all()
            and (class_grid[~valid_mask] == -1).all()
        )

        flat_score = score_grid[valid_mask].astype(np.float64)
        flat_class = class_grid[valid_mask]
        assert_class_consistency(flat_score, flat_class)

        idx = rng.choice(flat_score.size, min(n_check, flat_score.size), replace=False)
        source_f32 = scores["A"][year].astype(np.float32)
        roundtrip_exact = bool((flat_score[idx] == source_f32[idx].astype(np.float64)).all())

        checks[str(year)] = {
            "nodata_never_scored": nodata_ok,
            "class_consistent_with_score": True,
            "subsample_roundtrip_exact_f32": roundtrip_exact,
            "n_subsampled": int(idx.size),
        }
    checks["all_passed"] = all(
        v["nodata_never_scored"] and v["subsample_roundtrip_exact_f32"]
        for k, v in checks.items()
        if isinstance(v, dict)
    )
    print(f"  Written-raster verification: {checks}")
    return checks


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def _build_manifest(
    data: Dict,
    record: Dict,
) -> Dict:
    """Build the Phase 7 manifest (inputs+hashes, weights, thresholds)."""
    import pyproj

    input_paths = [VALID_MASK_RASTER]
    for mapping in (
        data["input_paths"]["severity_score"],
        data["input_paths"]["lst"],
        data["input_paths"]["ndvi"],
        data["input_paths"]["ndbi"],
        data["input_paths"]["vegetation_cover"],
    ):
        input_paths.extend(mapping.values())
    input_paths.extend(data["input_paths"]["static"])

    inputs = []
    for p in input_paths:
        p = Path(p)
        inputs.append({"path": str(p), "size_bytes": p.stat().st_size, "sha256": _sha256(p)})

    return {
        "phase": 7,
        "stage": 1,
        "project": "GreenGrid-AI",
        "generated_utc": _now(),
        "random_seed": RANDOM_SEED,
        "inputs": inputs,
        "grid_spec": {
            "crs": str(data["profile"]["crs"]),
            "height": int(data["profile"]["height"]),
            "width": int(data["profile"]["width"]),
            "valid_cells_per_year": int(data["n_valid"]),
            "years": list(YEARS),
        },
        "concept_model": {
            "baseline": "Final = Need * Opportunity / 100 (gated product)",
            "scenarios": {
                s: {
                    "exponents": list(exps),
                    "formula": (
                        f"100^{{1-{exps[0]}-{exps[1]}}} * "
                        f"Need^{exps[0]} * Opportunity^{exps[1]}"
                    ),
                }
                for s, exps in SCENARIO_EXPONENTS.items()
            },
            "scenario_D_composition": {
                "need_weights": SCENARIO_D_NEED_WEIGHTS,
                "opportunity_weights": SCENARIO_D_OPPORTUNITY_WEIGHTS,
                "note": (
                    "NDVI dropped from Need, replaced by 1-vegetation_cover "
                    "(0.25); Opportunity headroom computed on inverted "
                    "normalized vegetation_cover (0.15 -> 0.25) and built-up "
                    "downweighted (0.25 -> 0.15)."
                ),
            },
            "additive_diagnostic": (
                "0.5*Need + 0.5*Opportunity, stored in suitability_summary.csv "
                "only; comparison diagnostic, never the primary map"
            ),
        },
        "need_weights": NEED_WEIGHTS,
        "opportunity_weights": OPPORTUNITY_WEIGHTS,
        "normalization": {
            "method": "per-year robust min-max (clip at p1/p99 over valid cells, scale to 0-100)",
            "percentiles": list(NORM_PERCENTILES),
            "bounds_table": str(TABLE_PATHS["normalization_parameters"]),
            "per_year_note": (
                "Per-year normalization: all scores/class boundaries are "
                "class-relative across years (snapshots, not trends)."
            ),
        },
        "class_thresholds": {
            "encoding": "classes 0-4 (int16, nodata -1)",
            "edges": list(CLASS_EDGES),
            "labels": CLASS_LABELS,
            "boundary_rule": "scores exactly on 20/40/60/80 belong to the lower class",
        },
        "landuse_rules_table": str(TABLE_PATHS["landuse_rules"]),
        "hard_exclusions": {
            "only": "invalid mask (valid_mask_30m.tif)",
            "limitations": [
                "no water exclusion layer exists (audit section 4)",
                "no building-footprint exclusion (buildings are an OSM sample)",
                "no road-surface exclusion (only road distance exists)",
            ],
        },
        "opportunity_static_note": (
            "landuse/road/green-proximity Opportunity terms are STATIC: the "
            "2022 and 2026 Opportunity rasters differ only via the NDBI/NDVI "
            "terms. Opportunity change 2022->2026 reflects vegetation/index "
            "change on a static land-use/accessibility base, not observed "
            "land-use change."
        ),
        "terminology_note": TERMINOLOGY_NOTE,
        "framing": (
            "Relative tree-plantation suitability proxy built on a relative "
            "heat-severity proxy (Phase 6). 2022/2026 are two snapshot "
            "composites, not a trend. Suitability classes are per-year "
            "normalized (class-relative cross-year comparison only). Nothing "
            "claims planting success probability."
        ),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rasterio": rasterio.__version__,
            "pyproj": pyproj.__version__,
        },
        "stage_record": record["steps"],
    }


# ---------------------------------------------------------------------------
# Stage 1 driver
# ---------------------------------------------------------------------------
def run_stage1(skip_rasters: bool = False, skip_tables: bool = False) -> Dict:
    """Run the complete Phase 7 Stage 1 suitability-scoring pipeline."""
    PHASE7_DIR.mkdir(parents=True, exist_ok=True)
    record: Dict = {
        "phase": 7,
        "stage": 1,
        "project": "GreenGrid-AI",
        "project_root": str(PROJECT_ROOT),
        "started_utc": _now(),
        "random_seed": RANDOM_SEED,
        "steps": {},
    }

    # ------------------------------------------------------------------
    # 1. Load + align inputs, intersected with the valid mask
    # ------------------------------------------------------------------
    t0 = time.time()
    data = load_inputs()
    data["input_paths"] = {
        "severity_score": dict(SEVERITY_SCORE_RASTERS),
        "lst": dict(LST_RASTERS),
        "ndvi": dict(NDVI_RASTERS),
        "ndbi": dict(NDBI_RASTERS),
        "vegetation_cover": dict(VEGETATION_COVER_RASTERS),
        "static": [LANDUSE_RASTER_PATH, ROADS_DISTANCE_RASTER, VEGETATION_DISTANCE_RASTER],
    }
    n_valid = data["n_valid"]
    assert n_valid == EXPECTED_VALID_PIXELS, n_valid
    print(f"  Valid cells per year: {n_valid:,} (grid {data['profile']['height']}x{data['profile']['width']})")
    record["steps"]["load_inputs"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "valid_cells_per_year": int(n_valid),
        "valid_mask": str(VALID_MASK_RASTER),
        "note": (
            "every raw input intersected with the valid mask; raw NDVI/NDBI/"
            "vegetation_cover/LST cover more pixels than the valid mask "
            "(audit sections 3/5) and were asserted finite at all valid cells"
        ),
    }

    # ------------------------------------------------------------------
    # 2. Per-year robust min-max normalization
    # ------------------------------------------------------------------
    t0 = time.time()
    normalized, bounds_rows = normalize_inputs(data)
    norm_df = pd.DataFrame(bounds_rows)
    if not skip_tables:
        norm_df.to_csv(TABLE_PATHS["normalization_parameters"], index=False)
    record["steps"]["normalization"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "method": "per-year robust min-max, clip p1/p99 over valid cells, scale 0-100",
        "output_path": str(TABLE_PATHS["normalization_parameters"]),
        "n_variables_per_year": int(norm_df[norm_df["year"] == YEARS[0]].shape[0]),
    }
    print(f"  Normalized quantities per year: {record['steps']['normalization']['n_variables_per_year']}")

    # ------------------------------------------------------------------
    # 3. Heat Need and Plantation Opportunity per year (+ Scenario D)
    # ------------------------------------------------------------------
    t0 = time.time()
    need: Dict[int, Dict[str, np.ndarray]] = {}
    opportunity: Dict[int, Dict[str, np.ndarray]] = {}
    for year in YEARS:
        need[year] = {
            "baseline": compute_heat_need(normalized[year], "baseline"),
            "D": compute_heat_need(normalized[year], "D"),
        }
        opp = compute_opportunity(normalized[year], data["static"], "baseline")
        opp_d = compute_opportunity(normalized[year], data["static"], "D")
        opportunity[year] = {
            "baseline": opp["opportunity"],
            "D": opp_d["opportunity"],
        }
        # Park/forest presence at valid cells (spec section d provenance).
        lu = data["static"]["landuse"]
        lu_counts = {int(c): int(n) for c, n in zip(*np.unique(lu, return_counts=True))}
        if year == YEARS[0]:
            record["steps"]["landuse_composition_valid_cells"] = lu_counts
        print(
            f"  [{year}] Need mean={need[year]['baseline'].mean():.2f}, "
            f"Opportunity mean={opportunity[year]['baseline'].mean():.2f} "
            f"(park px={lu_counts.get(1, 0)}, forest px={lu_counts.get(2, 0)} at valid cells)"
        )
    record["steps"]["need_opportunity"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "need_weights": NEED_WEIGHTS,
        "opportunity_weights": OPPORTUNITY_WEIGHTS,
        "scenario_D_need_weights": SCENARIO_D_NEED_WEIGHTS,
        "scenario_D_opportunity_weights": SCENARIO_D_OPPORTUNITY_WEIGHTS,
        "score_range_assertions": "all Need/Opportunity scores asserted in [0, 100]",
    }

    # ------------------------------------------------------------------
    # 4. Hard exclusion provenance raster
    # ------------------------------------------------------------------
    t0 = time.time()
    exclusion = build_exclusion_mask(data["valid_mask"], data["profile"])
    record["steps"]["constraints"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        **exclusion,
    }

    # ------------------------------------------------------------------
    # 5. Suitability scores (A baseline + B/C/D scenarios)
    # ------------------------------------------------------------------
    t0 = time.time()
    need_by_variant = {y: need[y] for y in YEARS}
    scores = compute_scenario_scores(need_by_variant, opportunity)
    per_year: Dict[int, Dict] = {}
    for year in YEARS:
        per_year[year] = {
            "valid_mask": data["valid_mask"],
            "need": need[year]["baseline"],
            "opportunity": opportunity[year]["baseline"],
            "additive": additive_diagnostic(need[year]["baseline"], opportunity[year]["baseline"]),
            "baseline": scores["A"][year],
            "baseline_classes": classify(scores["A"][year].astype(np.float32).astype(np.float64)),
        }
    record["steps"]["scoring"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "scenarios": {
            s: {"exponents": list(e), "formula": "100^(1-an-ao) * Need^an * Opp^ao"}
            for s, e in SCENARIO_EXPONENTS.items()
        },
        "baseline_formula": "Need * Opportunity / 100 (gated product)",
        "additive_diagnostic": "0.5*Need + 0.5*Opp; table-only comparison diagnostic",
    }

    # ------------------------------------------------------------------
    # 6. Equal-area per-pixel areas (EPSG:32643)
    # ------------------------------------------------------------------
    t0 = time.time()
    print("  Computing per-pixel EPSG:32643 areas ...")
    area_grid = per_pixel_area_m2(
        data["transform"], data["profile"]["height"], data["profile"]["width"]
    )
    pixel_areas_flat_m2 = area_grid[data["valid_mask"]]
    record["steps"]["pixel_areas"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "area_crs": UTM_CRS,
        "mean_valid_pixel_area_m2": round(float(pixel_areas_flat_m2.mean()), 2),
        "note": "corner-reprojected per-pixel areas; degrees never used for area",
    }

    # ------------------------------------------------------------------
    # 7. Raster outputs
    # ------------------------------------------------------------------
    raster_outputs: Dict[str, Dict] = {}
    if not skip_rasters:
        t0 = time.time()
        rows, cols = data["rows"], data["cols"]
        profile = data["profile"]
        for year in YEARS:
            raster_outputs[f"heat_need_{year}"] = write_score_raster(
                need[year]["baseline"], heat_need_raster_path(year), profile, rows, cols
            )
            raster_outputs[f"plantation_opportunity_{year}"] = write_score_raster(
                opportunity[year]["baseline"], opportunity_raster_path(year), profile, rows, cols
            )
            raster_outputs[f"tree_plantation_suitability_{year}"] = write_score_raster(
                scores["A"][year], suitability_raster_path(year), profile, rows, cols
            )
            raster_outputs[f"tree_plantation_suitability_class_{year}"] = write_class_raster(
                scores["A"][year], suitability_class_raster_path(year), profile, rows, cols
            )
            for scenario in ("B", "C", "D"):
                raster_outputs[f"suitability_scenario_{scenario}_{year}"] = write_score_raster(
                    scores[scenario][year], scenario_raster_path(scenario, year), profile, rows, cols
                )
        record["steps"]["rasters"] = {
            "status": "success",
            "elapsed_s": round(time.time() - t0, 3),
            "outputs": {k: v["output_path"] for k, v in raster_outputs.items()},
            "note": (
                "Opportunity uses static landuse/distances -> its 2022 and "
                "2026 rasters differ only via the NDBI/NDVI terms (expected, "
                "documented). Per-component diagnostic rasters intentionally "
                "not written; components are in the tables."
            ),
        }
        print(f"  Wrote {len(raster_outputs)} score/class rasters")
    else:
        record["steps"]["rasters"] = {"status": "skipped", "outputs": {}}

    # ------------------------------------------------------------------
    # 8. Assertions on written rasters
    # ------------------------------------------------------------------
    t0 = time.time()
    assert_class_partition()
    if not skip_rasters:
        raster_checks = verify_written_rasters(per_year, scores)
    else:
        raster_checks = {"all_passed": True, "note": "raster writing skipped"}
    record["steps"]["assertions"] = {
        "status": "success" if raster_checks["all_passed"] else "failed",
        "elapsed_s": round(time.time() - t0, 3),
        "class_partition": "class thresholds verified to partition [0, 100]",
        "valid_pixel_count": int(n_valid),
        "score_range": "all scores asserted in [0, 100] at computation time",
        **raster_checks,
    }

    # ------------------------------------------------------------------
    # 9. Tables
    # ------------------------------------------------------------------
    if not skip_tables:
        t0 = time.time()
        summary_df = build_summary_table(per_year, pixel_areas_flat_m2, n_valid)
        summary_df.to_csv(TABLE_PATHS["summary"], index=False)
        area_df = build_area_statistics_table(per_year, pixel_areas_flat_m2, n_valid)
        area_df.to_csv(TABLE_PATHS["area_statistics"], index=False)
        sens_df = build_sensitivity_table(scores, pixel_areas_flat_m2, n_valid)
        sens_df.to_csv(TABLE_PATHS["sensitivity"], index=False)
        pd.DataFrame(FEATURE_DEPENDENCY_ROWS).to_csv(
            TABLE_PATHS["feature_dependency"], index=False
        )
        build_landuse_rules_table().to_csv(TABLE_PATHS["landuse_rules"], index=False)
        record["steps"]["tables"] = {
            "status": "success",
            "elapsed_s": round(time.time() - t0, 3),
            "outputs": {k: str(v) for k, v in TABLE_PATHS.items()},
        }
        for year in YEARS:
            row = summary_df[summary_df["year"] == year].iloc[0]
            print(
                f"  [{year}] suitability mean={row['suitability_mean']}, "
                f"High+VH {row['high_very_high_pct']}% "
                f"({row['high_very_high_area_ha']} ha)"
            )
    else:
        record["steps"]["tables"] = {"status": "skipped"}

    # ------------------------------------------------------------------
    # 10. Manifest + pipeline record (stage_records container)
    # ------------------------------------------------------------------
    record["finished_utc"] = _now()
    record["total_elapsed_s"] = round(
        sum(s.get("elapsed_s", 0) for s in record["steps"].values() if isinstance(s, dict)),
        3,
    )
    record["status"] = "success"
    record["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
    }

    manifest = _build_manifest(data, record)
    _save_json(manifest, MANIFEST_JSON)

    container = {"phase": 7, "project": "GreenGrid-AI", "stage_records": [record]}
    _save_json(container, PIPELINE_RECORD_JSON)
    print(f"  Manifest: {MANIFEST_JSON}")
    print(f"  Pipeline record: {PIPELINE_RECORD_JSON}")

    return {
        "record": record,
        "manifest": manifest,
        "scores": scores,
        "per_year": per_year,
    }


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for the Phase 7 Stage 1 pipeline."""
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 7 Stage 1 suitability scoring."
    )
    parser.add_argument(
        "--skip-rasters",
        action="store_true",
        help="Skip score/class raster writing (tables and JSON are still written).",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip CSV table writing (rasters and JSON are still written).",
    )
    args = parser.parse_args(argv)

    try:
        results = run_stage1(skip_rasters=args.skip_rasters, skip_tables=args.skip_tables)
        print(json.dumps(_convert_for_json(results["record"]), indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Phase 7 Stage 1 pipeline failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
