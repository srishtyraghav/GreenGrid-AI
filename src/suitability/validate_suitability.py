"""Independent validation script for Phase 7 outputs.

Verifies upstream integrity (Phases 5/6 frozen), raster/grid conformance,
score/class consistency (seed-42 50k samples + full-grid class compares),
priority-zone quality, temporal statistics, sensitivity/baseline
reproducibility, manifest provenance, the leakage/circularity audit
(design spec section 34), and (optionally, via ``--repro``) the
reproducibility of the Stage 2 tables and the Stage 1 class histograms.

The check record pattern is cloned from ``severity.validate_severity``:
``_check(name, category, condition, message)`` with categories
inputs / alignment / scores / classes / zones / temporal / sensitivity /
provenance / leakage / reproducibility.

Run with::

    PYTHONPATH=src .venv/bin/python -m suitability.validate_suitability
    PYTHONPATH=src .venv/bin/python -m suitability.validate_suitability --repro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from . import zones as zones_module
from .config import (
    CLASS_EDGES,
    CLASS_LABELS,
    EXCLUSION_MASK_RASTER,
    EXPECTED_VALID_PIXELS,
    MANIFEST_JSON,
    NEED_WEIGHTS,
    OPPORTUNITY_WEIGHTS,
    PHASE7_RASTERS_DIR,
    PHASE7_REPORTS_DIR,
    PHASE7_TABLES_DIR,
    PHASE7_VECTORS_DIR,
    PIPELINE_RECORD_JSON,
    PROJECT_ROOT,
    RANDOM_SEED,
    SCENARIO_D_NEED_WEIGHTS,
    SCENARIO_D_OPPORTUNITY_WEIGHTS,
    SCENARIO_EXPONENTS,
    TABLE_PATHS,
    UTM_CRS,
    VALID_MASK_RASTER,
    YEARS,
    heat_need_raster_path,
    opportunity_raster_path,
    scenario_raster_path,
    suitability_class_raster_path,
    suitability_raster_path,
)
from .scoring import classify

PHASE7_VALIDATION_REPORT_JSON = PHASE7_REPORTS_DIR / "phase7_validation_report.json"

PHASE5_VALIDATION_REPORT_JSON = (
    PROJECT_ROOT / "data" / "processed" / "phase5" / "phase5_validation_report.json"
)
PHASE6_VALIDATION_REPORT_JSON = (
    PROJECT_ROOT / "data" / "processed" / "phase6" / "reports" / "phase6_validation_report.json"
)

# Frozen grid facts (Phase 7 manifest grid_spec and raster census).
GRID_SHAPE = (1768, 1874)
VALID_PIXELS_PER_YEAR = 1_500_777
SCORE_RASTER_COUNT = 15
VECTOR_COUNT = 11
SCENARIOS = ("A", "B", "C", "D")
CONSISTENCY_SAMPLE_N = 50_000

# Nominal mean valid-pixel area (m2, EPSG:32643) for the zone polygon check.
NOMINAL_PIXEL_AREA_M2 = 787.11
POLYGON_AREA_TOLERANCE = 0.05
AREA_SUM_TOLERANCE = 0.005

# Frozen upstream validation summaries (do-not-break checks).
PHASE5_EXPECTED_SUMMARY = {"total": 57, "pass": 57, "fail": 0, "warn": 0}
PHASE6_EXPECTED_SUMMARY = {"total": 147, "pass": 147, "fail": 0, "warn": 0}

# Paths that must stay untouched by Phase 7 work (git status audit).
FROZEN_GIT_PREFIXES = (
    "src/models/",
    "src/features/",
    "src/preprocessing/",
    "src/severity/",
    "data/processed/phase5/",
    "data/processed/phase6/",
)

# Canonical Stage 1-3 tables (~15 expected).
EXPECTED_TABLES = [
    "suitability_summary.csv",
    "suitability_area_statistics.csv",
    "feature_dependency_analysis.csv",
    "normalization_parameters.csv",
    "landuse_suitability_rules.csv",
    "sensitivity_analysis.csv",
    "priority_zone_statistics.csv",
    "priority_ranking.csv",
    "why_here_explanations.csv",
    "temporal_priority_transition.csv",
    "environmental_priority_statistics.csv",
    "green_built_comparison.csv",
    "landuse_suitability_summary.csv",
]

EXPECTED_VECTORS = [
    f"high_priority_zones_{year}.geojson" for year in YEARS
] + [
    f"very_high_priority_zones_{year}.geojson" for year in YEARS
] + [
    f"priority_zones_combined_{year}.geojson" for year in YEARS
] + [
    f"scenario_priority_zones_{sc}_{year}.geojson"
    for sc in ("B", "C") for year in YEARS
] + ["temporal_priority_zones_A_2022_2026.geojson"]

MIN_ZONE_PIXELS = 10  # FROZEN noise rule (spec section i)


def _check(name: str, category: str, condition: bool, message: str) -> Dict:
    status = "PASS" if condition else "FAIL"
    return {"name": name, "category": category, "status": status, "message": message}


def _warn(name: str, category: str, message: str) -> Dict:
    return {"name": name, "category": category, "status": "WARN", "message": message}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _all_rasters() -> List[Path]:
    files: List[Path] = [EXCLUSION_MASK_RASTER]
    for year in YEARS:
        files.append(heat_need_raster_path(year))
        files.append(opportunity_raster_path(year))
        files.append(suitability_raster_path(year))
        files.append(suitability_class_raster_path(year))
        for scenario in ("B", "C", "D"):
            files.append(scenario_raster_path(scenario, year))
    return files


def _read_valid_mask() -> np.ndarray:
    with rasterio.open(VALID_MASK_RASTER) as src:
        return src.read(1) == 1


def _sample_cells(valid_mask: np.ndarray, n: int = CONSISTENCY_SAMPLE_N) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    rows, cols = np.where(valid_mask)
    sel = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return rows[sel], cols[sel]


def _kept_zone_pixels(binary_grid: np.ndarray) -> int:
    """Recompute the Stage 2 noise rule: 8-connected clusters >= MIN_ZONE_PIXELS."""
    labels, n_found = ndimage.label(binary_grid, structure=np.ones((3, 3), dtype=int))
    if n_found == 0:
        return 0
    sizes = np.bincount(labels.ravel())
    keep = sizes >= MIN_ZONE_PIXELS
    keep[0] = False
    return int(sizes[keep].sum())


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def validate_inputs() -> List[Dict]:
    """Validate frozen upstream phases and the presence of all stage outputs."""
    checks: List[Dict] = []
    print("Validating inputs ...")

    rasters = sorted(PHASE7_RASTERS_DIR.glob("*.tif"))
    checks.append(
        _check(
            "Phase 7 raster suite complete (15 rasters)",
            "inputs",
            len(rasters) == SCORE_RASTER_COUNT and all(p.stat().st_size > 0 for p in rasters),
            f"found={len(rasters)}, expected={SCORE_RASTER_COUNT}",
        )
    )

    vectors = sorted(PHASE7_VECTORS_DIR.glob("*.geojson"))
    missing_vectors = [v for v in EXPECTED_VECTORS if not (PHASE7_VECTORS_DIR / v).exists()]
    checks.append(
        _check(
            "Phase 7 vector suite complete (11 GeoJSONs)",
            "inputs",
            len(vectors) == VECTOR_COUNT and not missing_vectors,
            f"found={len(vectors)}, expected={VECTOR_COUNT}, missing={missing_vectors}",
        )
    )

    missing_tables = [t for t in EXPECTED_TABLES if not (PHASE7_TABLES_DIR / t).exists()]
    n_tables = len(list(PHASE7_TABLES_DIR.glob("*.csv")))
    checks.append(
        _check(
            "Phase 7 table suite complete (13 canonical Stage 1-3 tables)",
            "inputs",
            not missing_tables,
            f"found={n_tables}, expected~15, missing={missing_tables}",
        )
    )

    for label, path in [
        ("Phase 7 manifest", MANIFEST_JSON),
        ("Phase 7 pipeline record", PIPELINE_RECORD_JSON),
    ]:
        checks.append(
            _check(
                f"{label} exists and non-empty",
                "inputs",
                Path(path).exists() and Path(path).stat().st_size > 0,
                str(path),
            )
        )

    # Frozen upstream validation reports must be untouched.
    for label, path, expected in [
        ("Phase 5", PHASE5_VALIDATION_REPORT_JSON, PHASE5_EXPECTED_SUMMARY),
        ("Phase 6", PHASE6_VALIDATION_REPORT_JSON, PHASE6_EXPECTED_SUMMARY),
    ]:
        exists = Path(path).exists()
        summary = {}
        if exists:
            summary = json.loads(Path(path).read_text()).get("summary", {})
        checks.append(
            _check(
                f"{label} validation report still {expected['pass']}/{expected['fail']}/{expected['warn']}",
                "inputs",
                exists and summary == expected,
                f"summary={summary}",
            )
        )

    # Git status: no modifications anywhere under the frozen paths.  Only
    # new untracked phase7 files (src/suitability/, reports/phase7_*.md,
    # data/processed/phase7/) are acceptable.
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--short"],
        capture_output=True, text=True, check=True,
    )
    violations = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        status, path = line[:2], line[3:]
        if " -> " in path:  # rename: check both endpoints
            path = path.split(" -> ", 1)[1]
        if any(path.startswith(prefix) for prefix in FROZEN_GIT_PREFIXES):
            violations.append(line.strip())
    checks.append(
        _check(
            "git status clean under frozen paths (models/features/preprocessing/severity/phase5/phase6)",
            "inputs",
            len(violations) == 0,
            f"violations={violations}" if violations else "no modifications under frozen paths",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def validate_alignment() -> List[Dict]:
    """Verify CRS/transform/dims/nodata/compression of every Phase 7 raster."""
    checks: List[Dict] = []
    print("Validating raster alignment ...")

    ref_path = PROJECT_ROOT / "data" / "processed" / "phase6" / "rasters" / "severity_score_2022.tif"
    with rasterio.open(ref_path) as ref:
        ref_transform, ref_shape = ref.transform, (ref.height, ref.width)

    for path in _all_rasters():
        label = path.name
        facts_ok = False
        msg = "missing"
        if path.exists():
            with rasterio.open(path) as src:
                ok_crs = src.crs is not None and src.crs.to_epsg() == 4326
                ok_shape = (src.height, src.width) == ref_shape == GRID_SHAPE
                ok_transform = src.transform == ref_transform
                ok_nodata = src.nodata is not None
                ok_compress = src.compression is not None and "lzw" in str(src.compression).lower()
                facts_ok = ok_crs and ok_shape and ok_transform and ok_nodata and ok_compress
                msg = (
                    f"crs_epsg={src.crs.to_epsg() if src.crs else None}, "
                    f"shape={(src.height, src.width)}, transform_match={ok_transform}, "
                    f"nodata={src.nodata}, compression={src.compression}"
                )
        checks.append(_check(f"{label} grid conformance", "alignment", facts_ok, msg))
    return checks


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
def validate_scores(valid_mask: np.ndarray) -> List[Dict]:
    """Verify score ranges, NoData discipline and valid-pixel counts."""
    checks: List[Dict] = []
    print("Validating scores ...")
    sr, sc = _sample_cells(valid_mask)

    def _score_check(path_fn, label: str) -> None:
        for year in YEARS:
            with rasterio.open(path_fn(year)) as src:
                arr = src.read(1)
                nodata = src.nodata
            valid = valid_mask
            flat = arr[valid]
            n_valid = int((arr != nodata).sum())
            checks.append(
                _check(
                    f"{label} {year} valid pixel count",
                    "scores",
                    n_valid == VALID_PIXELS_PER_YEAR,
                    f"n_valid={n_valid:,}, expected={VALID_PIXELS_PER_YEAR:,}",
                )
            )
            checks.append(
                _check(
                    f"{label} {year} finite and in [0, 100] at all valid cells",
                    "scores",
                    bool(np.isfinite(flat).all()) and bool((flat >= 0).all()) and bool((flat <= 100).all()),
                    f"min={flat.min():.6f}, max={flat.max():.6f}, n_sampled={len(sr):,}",
                )
            )

    _score_check(heat_need_raster_path, "heat_need")
    _score_check(opportunity_raster_path, "plantation_opportunity")
    _score_check(suitability_raster_path, "suitability")
    for scenario in ("B", "C", "D"):
        _score_check(lambda y, s=scenario: scenario_raster_path(s, y), f"scenario_{scenario}")

    with rasterio.open(EXCLUSION_MASK_RASTER) as src:
        excl = src.read(1)
    checks.append(
        _check(
            "exclusion mask == 0 at all valid cells (only invalid mask excluded)",
            "scores",
            bool((excl[valid_mask] == 0).all()) and bool((excl[~valid_mask] == 1).all()),
            f"excluded={int((excl == 1).sum()):,}, domain={int((excl == 0).sum()):,}, "
            f"values={np.unique(excl).tolist()}",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
def validate_classes(valid_mask: np.ndarray) -> List[Dict]:
    """Verify class domains, exact thresholds (both boundary sides) and tables."""
    checks: List[Dict] = []
    print("Validating classes (full-grid score/class compare) ...")

    summary = pd.read_csv(TABLE_PATHS["summary"]).set_index("year")
    area_stats = pd.read_csv(TABLE_PATHS["area_statistics"])

    for year in YEARS:
        with rasterio.open(suitability_raster_path(year)) as src:
            score = src.read(1).astype(np.float64)
        with rasterio.open(suitability_class_raster_path(year)) as src:
            classes = src.read(1)
        nodata = -1

        vals = set(np.unique(classes)).difference({nodata})
        checks.append(
            _check(
                f"{year} class values subset of {{0..4}} + nodata",
                "classes",
                vals.issubset({0, 1, 2, 3, 4}),
                f"unique={sorted(vals)}",
            )
        )

        flat_score = score[valid_mask]
        flat_class = classes[valid_mask]
        mismatches = int((classify(flat_score) != flat_class).sum())
        checks.append(
            _check(
                f"{year} class raster == classify(score raster) at every valid cell",
                "classes",
                mismatches == 0,
                f"mismatches={mismatches:,} of {flat_class.size:,}",
            )
        )

        # Exact threshold semantics: boundary scores 20/40/60/80 belong to
        # the LOWER class; verify both sides of each boundary from the score
        # raster (float32 spacing is << the 1e-3 side windows).
        boundary_ok = True
        boundary_msg = []
        for idx, edge in enumerate(CLASS_EDGES):
            exact = flat_score == edge
            below = (flat_score > edge - 1e-3) & (flat_score < edge)
            above = (flat_score > edge) & (flat_score <= edge + 1e-3)
            ok = True
            if exact.any():
                ok &= bool((flat_class[exact] == idx).all())
            if below.any():
                ok &= bool((flat_class[below] <= idx).all())
            if above.any():
                ok &= bool((flat_class[above] >= idx + 1).all())
            boundary_ok &= ok
            boundary_msg.append(
                f"{edge}: n_exact={int(exact.sum())}, n_below={int(below.sum())}, "
                f"n_above={int(above.sum())}, ok={ok}"
            )
        checks.append(
            _check(
                f"{year} thresholds exact (20/40/60/80 in lower class, both sides)",
                "classes",
                boundary_ok,
                "; ".join(boundary_msg),
            )
        )

        # Class raster histogram vs summary counts (exact).
        counts = np.bincount(flat_class.astype(np.int64), minlength=5)
        expected = np.array(
            [int(summary.loc[year, f"class_{k}_{CLASS_LABELS[k].lower().replace(' ', '_')}_count"]) for k in range(5)]
        )
        checks.append(
            _check(
                f"{year} class histogram == suitability_summary counts",
                "classes",
                bool((counts == expected).all()),
                f"raster={counts.tolist()}, summary={expected.tolist()}",
            )
        )

        sub = area_stats[area_stats["year"] == year]
        sum_px = int(sub["pixel_count"].sum())
        sum_area = float(sub["area_ha"].sum())
        valid_area = float(summary.loc[year, "valid_area_ha"])
        rel_err = abs(sum_area - valid_area) / valid_area
        checks.append(
            _check(
                f"{year} suitability_area_statistics sums to valid area (+-0.5%)",
                "classes",
                sum_px == VALID_PIXELS_PER_YEAR and rel_err <= AREA_SUM_TOLERANCE,
                f"sum_px={sum_px:,}, sum_area={sum_area:,.3f} ha, valid_area={valid_area:,.3f} ha, rel_err={rel_err:.2e}",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------
def _scenario_class_grid(scenario: str, year: int, valid_mask: np.ndarray) -> np.ndarray:
    """Full-grid class raster for a scenario, recomputed from the score raster."""
    path = suitability_raster_path(year) if scenario == "A" else scenario_raster_path(scenario, year)
    with rasterio.open(path) as src:
        score = src.read(1)
    grid = np.full(score.shape, -1, dtype=np.int16)
    grid[valid_mask] = classify(score[valid_mask].astype(np.float64))
    return grid


def _geojson_checks(
    checks: List[Dict],
    path: Path,
    label: str,
    expected_pixel_sum: int,
    id_columns: List[str],
    check_attributes: bool,
) -> None:
    """Shared geometry/pixel/attribute checks for one zone GeoJSON."""
    try:
        gdf = gpd.read_file(path)
        loads = True
    except Exception as exc:  # pragma: no cover - defensive
        checks.append(_check(f"{label} GeoJSON loads", "zones", False, f"raised: {exc}"))
        return
    checks.append(
        _check(
            f"{label} GeoJSON loads",
            "zones",
            loads,
            f"n_features={len(gdf)}",
        )
    )

    if len(gdf):
        invalid = int((~gdf.geometry.is_valid).sum())
        gdf_utm = gdf.to_crs(UTM_CRS)
        zero_area = int((gdf_utm.geometry.area <= 0).sum())
        dup = int(gdf.duplicated(subset=id_columns).sum())
        px = pd.to_numeric(gdf["pixel_count"], errors="coerce")
        sum_px = int(px.sum())
        rel_err = np.abs(gdf_utm.geometry.area.values / (px.values * NOMINAL_PIXEL_AREA_M2) - 1.0)
        max_rel = float(np.nanmax(rel_err))
        checks.append(
            _check(
                f"{label} geometries valid, non-zero-area (EPSG:32643), unique IDs",
                "zones",
                invalid == 0 and zero_area == 0 and dup == 0,
                f"n={len(gdf)}, invalid={invalid}, zero_area={zero_area}, "
                f"dup_{'+'.join(id_columns)}={dup}",
            )
        )
        checks.append(
            _check(
                f"{label} sum(pixel_count) == extraction mask count",
                "zones",
                sum_px == expected_pixel_sum and not px.isna().any(),
                f"sum={sum_px:,}, mask_kept={expected_pixel_sum:,}",
            )
        )
        checks.append(
            _check(
                f"{label} polygon area within 5% of pixel_count x {NOMINAL_PIXEL_AREA_M2} m2",
                "zones",
                max_rel <= POLYGON_AREA_TOLERANCE,
                f"max rel err={max_rel:.6f}",
            )
        )
        if check_attributes:
            cols = ["area_ha", "mean_suitability", "priority_confidence"]
            num = gdf[cols].apply(pd.to_numeric, errors="coerce")
            nulls = int(num.isna().sum().sum())
            pc = num["priority_confidence"]
            checks.append(
                _check(
                    f"{label} attributes non-null (area_ha, mean_suitability, priority_confidence in [0,100])",
                    "zones",
                    nulls == 0 and bool((pc >= 0).all()) and bool((pc <= 100).all()),
                    f"nulls={nulls}, priority_confidence range=[{pc.min():.2f}, {pc.max():.2f}]",
                )
            )
    else:
        checks.append(
            _check(
                f"{label} empty extraction consistent (0 zones)",
                "zones",
                expected_pixel_sum == 0,
                f"n_features=0, mask_kept={expected_pixel_sum}",
            )
        )


def validate_zones(valid_mask: np.ndarray) -> List[Dict]:
    """Validate zone GeoJSONs, pixel accounting, areas and attributes."""
    checks: List[Dict] = []
    print("Validating priority zones ...")

    stats = pd.read_csv(PHASE7_TABLES_DIR / "priority_zone_statistics.csv")

    # Recompute the extraction masks independently (class grids + noise rule).
    kept: Dict[str, Dict[int, int]] = {}
    class_grids: Dict[str, Dict[int, np.ndarray]] = {}
    for scenario in ("A", "B", "C"):
        kept[scenario] = {}
        class_grids[scenario] = {}
        for year in YEARS:
            grid = _scenario_class_grid(scenario, year, valid_mask)
            class_grids[scenario][year] = grid
            kept[scenario][year] = _kept_zone_pixels(grid >= 3)
    kept_vh = {
        year: _kept_zone_pixels(class_grids["A"][year] == 4) for year in YEARS
    }

    for year in YEARS:
        _geojson_checks(
            checks, PHASE7_VECTORS_DIR / f"high_priority_zones_{year}.geojson",
            f"A-high {year}", kept["A"][year], ["zone_id"], True,
        )
        _geojson_checks(
            checks, PHASE7_VECTORS_DIR / f"very_high_priority_zones_{year}.geojson",
            f"A-vh {year}", kept_vh[year], ["zone_id"], True,
        )
        _geojson_checks(
            checks, PHASE7_VECTORS_DIR / f"priority_zones_combined_{year}.geojson",
            f"A-combined {year}", kept["A"][year] + kept_vh[year],
            ["zone_id", "class_level"], True,
        )
        for scenario in ("B", "C"):
            _geojson_checks(
                checks,
                PHASE7_VECTORS_DIR / f"scenario_priority_zones_{scenario}_{year}.geojson",
                f"{scenario}-high {year}", kept[scenario][year], ["zone_id"], True,
            )

    # Temporal zone vectors: per-category kept counts from the persistence
    # masks of the baseline (Scenario A) High+VH classes.
    hvh = {year: class_grids["A"][year] >= 3 for year in YEARS}
    cats = {
        "Persistent": valid_mask & hvh[YEARS[0]] & hvh[YEARS[1]],
        "Emerging": valid_mask & (~hvh[YEARS[0]]) & hvh[YEARS[1]],
        "Declining": valid_mask & hvh[YEARS[0]] & (~hvh[YEARS[1]]),
        "Stable Low": valid_mask & (~hvh[YEARS[0]]) & (~hvh[YEARS[1]]),
    }
    temporal_path = PHASE7_VECTORS_DIR / "temporal_priority_zones_A_2022_2026.geojson"
    try:
        tgdf = gpd.read_file(temporal_path)
        t_ok = True
        t_msg = f"n_features={len(tgdf)}"
    except Exception as exc:  # pragma: no cover - defensive
        tgdf = None
        t_ok = False
        t_msg = f"raised: {exc}"
    checks.append(_check("temporal zones GeoJSON loads", "zones", t_ok, t_msg))
    if tgdf is not None and len(tgdf):
        invalid = int((~tgdf.geometry.is_valid).sum())
        gdf_utm = tgdf.to_crs(UTM_CRS)
        zero_area = int((gdf_utm.geometry.area <= 0).sum())
        dup = int(tgdf.duplicated(subset=["zone_id", "persistence_category"]).sum())
        checks.append(
            _check(
                "temporal zones geometries valid, non-zero-area, unique (zone_id, category)",
                "zones",
                invalid == 0 and zero_area == 0 and dup == 0,
                f"n={len(tgdf)}, invalid={invalid}, zero_area={zero_area}, dup={dup}",
            )
        )
        # The temporal vectors only cover the three CHANGE categories
        # (Persistent/Emerging/Declining); Stable Low is counts-only in the
        # transition table and is deliberately not zoned.
        for category in ("Persistent", "Emerging", "Declining"):
            mask = cats[category]
            expected_kept = _kept_zone_pixels(mask)
            got = int(pd.to_numeric(
                tgdf.loc[tgdf["persistence_category"] == category, "pixel_count"], errors="coerce"
            ).sum())
            checks.append(
                _check(
                    f"temporal {category} sum(pixel_count) == recomputed category mask count",
                    "zones",
                    got == expected_kept,
                    f"sum={got:,}, recomputed_kept={expected_kept:,} (raw mask={int(mask.sum()):,})",
                )
            )

    # Why-here explanations: rows exist for the top zones of every
    # (scenario, year) with zones, in area order, with the confidence note.
    why_path = PHASE7_TABLES_DIR / "why_here_explanations.csv"
    why_exists = why_path.exists() and why_path.stat().st_size > 0
    checks.append(
        _check(
            "why_here_explanations.csv exists and non-empty",
            "zones",
            why_exists,
            str(why_path),
        )
    )
    if why_exists:
        why = pd.read_csv(why_path)
        note_ok = bool(why["priority_confidence_meaning"].notna().all()) and bool(
            (why["priority_confidence"] >= 0).all() and (why["priority_confidence"] <= 100).all()
        )
        checks.append(
            _check(
                "why-here rows carry priority_confidence and its meaning note",
                "zones",
                note_ok,
                f"rows={len(why)}, meaning_note_nulls={int(why['priority_confidence_meaning'].isna().sum())}",
            )
        )
        top_ok = True
        top_msg = []
        for scenario in ("A", "B", "C"):
            for year in YEARS:
                frame = stats[
                    (stats["scenario"] == scenario) & (stats["year"] == year)
                ].sort_values(["area_ha", "mean_suitability"], ascending=[False, False])
                n_top = min(10, len(frame))
                if n_top == 0:
                    continue
                expected_ids = set(frame.head(n_top)["zone_id"].astype(int))
                got_ids = set(
                    why.loc[(why["scenario"] == scenario) & (why["year"] == year), "zone_id"].astype(int)
                )
                ok = expected_ids == got_ids
                top_ok &= ok
                top_msg.append(f"{scenario}{year}:{'ok' if ok else 'MISMATCH'}")
        checks.append(
            _check(
                "why-here rows cover exactly the top zones (top 10 by area) per scenario-year",
                "zones",
                top_ok,
                " ".join(top_msg),
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------
def validate_temporal(valid_mask: np.ndarray) -> List[Dict]:
    """Validate the temporal priority transition table against the rasters."""
    checks: List[Dict] = []
    print("Validating temporal statistics ...")

    transition = pd.read_csv(PHASE7_TABLES_DIR / "temporal_priority_transition.csv")
    summary = pd.read_csv(TABLE_PATHS["summary"]).set_index("year")

    for scenario in ("A", "B", "C"):
        sub = transition[transition["scenario"] == scenario]
        hvh = {
            year: _scenario_class_grid(scenario, year, valid_mask) >= 3
            for year in YEARS
        }
        cats = {
            "Persistent": valid_mask & hvh[YEARS[0]] & hvh[YEARS[1]],
            "Emerging": valid_mask & (~hvh[YEARS[0]]) & hvh[YEARS[1]],
            "Declining": valid_mask & hvh[YEARS[0]] & (~hvh[YEARS[1]]),
            "Stable Low": valid_mask & (~hvh[YEARS[0]]) & (~hvh[YEARS[1]]),
        }
        sum_px = int(sub["pixel_count"].sum())
        sum_area = float(sub["area_ha"].sum())
        valid_area = float(summary.loc[YEARS[0], "valid_area_ha"])
        rel_err = abs(sum_area - valid_area) / valid_area
        checks.append(
            _check(
                f"{scenario} transition counts/areas sum to paired valid total",
                "temporal",
                sum_px == VALID_PIXELS_PER_YEAR and rel_err <= AREA_SUM_TOLERANCE,
                f"sum_px={sum_px:,}, sum_area={sum_area:,.3f} ha, valid_area={valid_area:,.3f} ha",
            )
        )
        recompute_ok = True
        recompute_msg = []
        for label, mask in cats.items():
            table_px = int(sub.loc[sub["persistence_category"] == label, "pixel_count"].iloc[0])
            raw = int(mask.sum())
            # Noise pixels (< MIN_ZONE_PIXELS clusters) are not zones but stay
            # in their category counts in the transition table; the table must
            # equal the raw category mask, and categories are disjoint by
            # construction (each cell has exactly one label).
            ok = table_px == raw
            recompute_ok &= ok
            recompute_msg.append(f"{label}:{table_px:,}{'==raw' if ok else f'!=raw:{raw:,}'}")
        checks.append(
            _check(
                f"{scenario} categories mutually exclusive and match raster recompute",
                "temporal",
                recompute_ok,
                " ".join(recompute_msg),
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------
def validate_sensitivity(valid_mask: np.ndarray) -> List[Dict]:
    """Validate scenario coverage, the gated-baseline identity and findings."""
    checks: List[Dict] = []
    print("Validating sensitivity analysis ...")

    sens = pd.read_csv(TABLE_PATHS["sensitivity"])
    per_scenario = sens[sens["section"] == "per_scenario"]
    for year in YEARS:
        present = set(per_scenario.loc[per_scenario["year"] == year, "scenario"])
        checks.append(
            _check(
                f"{year} all 4 scenarios present in sensitivity_analysis.csv",
                "sensitivity",
                present == set(SCENARIOS),
                f"present={sorted(present)}",
            )
        )
    count_cols = [f"class_{k}_count" for k in range(5)]
    sums_ok = bool((per_scenario[count_cols].sum(axis=1) == VALID_PIXELS_PER_YEAR).all())
    checks.append(
        _check(
            "per-scenario class counts sum to 1,500,777 per row",
            "sensitivity",
            sums_ok,
            f"row_sums_ok={sums_ok}",
        )
    )

    # Baseline reproducibility: Scenario A == Need x Opportunity / 100 from
    # the stored Need/Opportunity rasters (seed-42 50k sample per year).
    sr, sc = _sample_cells(valid_mask)
    for year in YEARS:
        with rasterio.open(heat_need_raster_path(year)) as src:
            need = src.read(1)[sr, sc].astype(np.float64)
        with rasterio.open(opportunity_raster_path(year)) as src:
            opp = src.read(1)[sr, sc].astype(np.float64)
        with rasterio.open(suitability_raster_path(year)) as src:
            stored = src.read(1)[sr, sc].astype(np.float64)
        expected = need * opp / 100.0
        max_abs = float(np.abs(expected - stored).max())
        checks.append(
            _check(
                f"{year} Scenario A == Need x Opp / 100 (50k sample, float tolerance)",
                "sensitivity",
                bool(np.allclose(expected, stored, rtol=1e-4, atol=1e-4)),
                f"max |diff|={max_abs:.2e} over {len(sr):,} cells",
            )
        )

    findings_path = PHASE7_REPORTS_DIR / "sensitivity_findings.md"
    exists = findings_path.exists() and findings_path.stat().st_size > 0
    quantifies = False
    msg = str(findings_path)
    if exists:
        text = findings_path.read_text()
        # The findings must quantify the High+VH extent spread: every
        # per-scenario area (rounded to 0.1 ha, thousands separator) from the
        # sensitivity CSV must appear in the findings table.
        missing = []
        for rec in per_scenario.itertuples():
            formatted = f"{rec.high_very_high_area_ha:,.1f}"
            plain = f"{rec.high_very_high_area_ha:.1f}"
            if formatted not in text and plain not in text:
                missing.append(f"{rec.scenario}{rec.year}:{formatted}")
        quantifies = not missing
        msg = f"missing_extent_values={missing}" if missing else "all 8 scenario-year High+VH extents quantified"
    checks.append(
        _check(
            "sensitivity_findings.md exists and quantifies the extent spread",
            "sensitivity",
            exists and quantifies,
            msg,
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def validate_provenance(manifest: Dict) -> List[Dict]:
    """Recompute input hashes and compare manifest weights to config."""
    checks: List[Dict] = []
    print("Validating provenance ...")

    inputs = manifest.get("inputs", [])
    mismatched = []
    missing = []
    for entry in inputs:
        path = Path(entry["path"])
        if not path.exists():
            missing.append(path.name)
            continue
        if _sha256(path) != entry["sha256"]:
            mismatched.append(path.name)
    checks.append(
        _check(
            "manifest input hashes recompute bit-identically",
            "provenance",
            inputs and not missing and not mismatched,
            f"n_inputs={len(inputs)}, missing={missing}, mismatched={mismatched}",
        )
    )

    checks.append(
        _check(
            "manifest need/opportunity weights == config (FROZEN)",
            "provenance",
            manifest.get("need_weights") == NEED_WEIGHTS
            and manifest.get("opportunity_weights") == OPPORTUNITY_WEIGHTS,
            f"need={manifest.get('need_weights')}, opportunity={manifest.get('opportunity_weights')}",
        )
    )
    d_comp = manifest.get("concept_model", {}).get("scenario_D_composition", {})
    checks.append(
        _check(
            "manifest Scenario D composition == config",
            "provenance",
            d_comp.get("need_weights") == SCENARIO_D_NEED_WEIGHTS
            and d_comp.get("opportunity_weights") == SCENARIO_D_OPPORTUNITY_WEIGHTS,
            f"d_need={d_comp.get('need_weights')}, d_opp={d_comp.get('opportunity_weights')}",
        )
    )
    scen = manifest.get("concept_model", {}).get("scenarios", {})
    exponents_ok = all(
        tuple(scen.get(s, {}).get("exponents", [])) == tuple(SCENARIO_EXPONENTS[s]) for s in SCENARIOS
    )
    checks.append(
        _check(
            "manifest scenario exponents == config",
            "provenance",
            exponents_ok,
            f"manifest={ {s: scen.get(s, {}).get('exponents') for s in SCENARIOS} }, config={SCENARIO_EXPONENTS}",
        )
    )
    grid = manifest.get("grid_spec", {})
    checks.append(
        _check(
            "manifest grid spec == frozen grid facts",
            "provenance",
            grid.get("valid_cells_per_year") == VALID_PIXELS_PER_YEAR
            and grid.get("height") == GRID_SHAPE[0]
            and grid.get("width") == GRID_SHAPE[1]
            and grid.get("crs") == "EPSG:4326",
            f"grid_spec={grid}",
        )
    )
    checks.append(
        _check(
            "manifest class thresholds == config CLASS_EDGES",
            "provenance",
            list(manifest.get("class_thresholds", {}).get("edges", [])) == list(CLASS_EDGES),
            f"edges={manifest.get('class_thresholds', {}).get('edges')}",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Leakage / circularity (design spec section 34) — static code audit
# ---------------------------------------------------------------------------
def validate_leakage() -> List[Dict]:
    """Static audit: no module consumes a phase7 output as an input."""
    checks: List[Dict] = []
    print("Validating leakage/circularity (static code audit) ...")

    src_dir = Path(__file__).resolve().parent
    sources = {p.name: p.read_text() for p in src_dir.glob("*.py")}

    # 1. inputs.py must only read phase3/4/5/6 + raw sources: no phase7 refs.
    inputs_src = sources.get("inputs.py", "")
    leaks = re.findall(r"phase7", inputs_src, flags=re.IGNORECASE)
    checks.append(
        _check(
            "inputs.py references no data/processed/phase7 path",
            "leakage",
            len(leaks) == 0,
            f"phase7_refs={len(leaks)}",
        )
    )

    # 2. Phase 7 raster paths may only be READ by statistics.py, zones.py,
    #    figures.py (Stage 1/2 raster consumers) and pipeline.py (read-back
    #    verification).  Writers (constraints.py, scoring.py via pipeline)
    #    reference the paths only in write mode.
    raster_helpers = [
        "heat_need_raster_path", "opportunity_raster_path",
        "suitability_raster_path", "suitability_class_raster_path",
        "scenario_raster_path", "EXCLUSION_MASK_RASTER", "PHASE7_RASTERS_DIR",
    ]
    allowed_modules = {"statistics.py", "zones.py", "figures.py", "pipeline.py", "config.py"}
    offenders = {}
    for name, text in sources.items():
        if name in allowed_modules or name == "validate_suitability.py":
            continue
        for line in text.splitlines():
            if "rasterio.open(" not in line or not any(h in line for h in raster_helpers):
                continue
            if '"w"' in line or "'w'" in line:  # write mode is a producer, not a consumer
                continue
            offenders.setdefault(name, []).append(line.strip())
    checks.append(
        _check(
            "phase7 raster paths read only by statistics/zones/figures/pipeline",
            "leakage",
            not offenders,
            f"offenders={offenders}" if offenders else "clean",
        )
    )

    # 3. pipeline.py may read back ONLY inside verify_written_rasters
    #    (read-only verification of values just written; never feeds scoring).
    pipe_src = sources.get("pipeline.py", "")
    verify_block = ""
    m = re.search(r"def verify_written_rasters.*?(?=\ndef |\n# -{10})", pipe_src, flags=re.S)
    if m:
        verify_block = m.group(0)
    open_lines = [
        line for line in pipe_src.splitlines()
        if "rasterio.open(" in line and any(h in line for h in raster_helpers)
    ]
    verify_lines = [
        line for line in verify_block.splitlines()
        if "rasterio.open(" in line and any(h in line for h in raster_helpers)
    ]
    outside = [line.strip() for line in open_lines if line.strip() not in [l.strip() for l in verify_lines]]
    checks.append(
        _check(
            "pipeline.py phase7 raster reads confined to verify_written_rasters read-back",
            "leakage",
            len(outside) == 0,
            f"reads_outside_verify={outside}" if outside else f"{len(verify_lines)} read-back opens, all inside verify_written_rasters",
        )
    )

    # 4. statistics.py / zones.py read only Stage 1 outputs — never their own.
    #    Allowed read constants: Stage 1 tables via TABLE_PATHS (normalization
    #    parameters, sensitivity_analysis).  Any read of their own outputs
    #    (their CSV/MD/GeoJSON products) is circularity.
    own_outputs = {
        "statistics.py": {
            "TEMPORAL_TRANSITION_CSV", "ENVIRONMENTAL_PRIORITY_CSV",
            "GREEN_BUILT_CSV", "LANDUSE_SUITABILITY_CSV", "SENSITIVITY_FINDINGS_MD",
        },
        "zones.py": {
            "PRIORITY_ZONE_STATISTICS_CSV", "PRIORITY_RANKING_CSV",
            "WHY_HERE_CSV", "WHY_HERE_MD",
        },
    }
    read_call = re.compile(r"(?:pd\.read_csv|gpd\.read_file|rasterio\.open|json\.load|\.read_text)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)")
    for module, own in own_outputs.items():
        text = sources.get(module, "")
        reads = set(read_call.findall(text))
        own_reads = sorted(r for r in reads if r in own)
        checks.append(
            _check(
                f"{module} never reads its own outputs as inputs",
                "leakage",
                not own_reads,
                f"own_output_reads={own_reads}" if own_reads else "clean",
            )
        )

    # 5. Full inventory note: every phase7 reference outside statistics.py /
    #    zones.py / pipeline.py is a write (figures.py reads Stage 1/2 tables
    #    and vectors to render PNGs — never its own figure outputs).
    fig_src = sources.get("figures.py", "")
    fig_reads = set(read_call.findall(fig_src))
    fig_phase7 = sorted(r for r in fig_reads if r.isupper() and "PHASE7" in r)
    checks.append(
        _check(
            "figures.py consumes only Stage 1/2 products (never its own figures)",
            "leakage",
            set(fig_phase7) <= {"PHASE7_TABLES_DIR", "PHASE7_VECTORS_DIR", "PRIORITY_ZONE_STATISTICS_CSV"},
            f"phase7_read_consts={fig_phase7}",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def run_reproducibility_checks() -> List[Dict]:
    """Re-run suitability.zones (idempotent) and demand bit-identical tables."""
    checks: List[Dict] = []
    print("Running reproducibility checks ...")

    # 1-2. Hash all Phase 7 tables, snapshot, re-run Stage 2, compare hashes.
    table_files = sorted(PHASE7_TABLES_DIR.glob("*.csv"))
    before = {p.name: _sha256(p) for p in table_files}
    tmp_dir = Path(tempfile.mkdtemp(prefix="phase7_repro_"))
    backup = tmp_dir / "tables_backup"
    shutil.copytree(PHASE7_TABLES_DIR, backup)

    print("  Re-running suitability.zones (Stage 2) for idempotence ...")
    try:
        zones_module.run_stage2()
        rerun_ok = True
        rerun_msg = "run_stage2 completed"
    except Exception as exc:  # pragma: no cover - defensive
        rerun_ok = False
        rerun_msg = f"run_stage2 raised: {exc}"

    after = {p.name: _sha256(p) for p in sorted(PHASE7_TABLES_DIR.glob("*.csv"))}
    missing_after = sorted(set(before) - set(after))
    added_after = sorted(set(after) - set(before))
    mismatched = sorted(name for name in before if name in after and before[name] != after[name])
    tables_identical = rerun_ok and not missing_after and not mismatched
    checks.append(
        _check(
            "suitability.zones re-run produces bit-identical tables",
            "reproducibility",
            tables_identical,
            f"{rerun_msg}; mismatched={mismatched}, missing={missing_after}, added={added_after}",
        )
    )
    if not tables_identical:
        for p in PHASE7_TABLES_DIR.glob("*.csv"):
            p.unlink()
        shutil.copytree(backup, PHASE7_TABLES_DIR, dirs_exist_ok=True)
        checks.append(
            _warn(
                "pre-repro table state restored after mismatch",
                "reproducibility",
                "tables restored from snapshot; investigate Stage 2 determinism",
            )
        )

    # 3. Lightweight raster-consistency repro: suitability class histograms
    #    vs suitability_summary.csv (stands in for re-running
    #    suitability.pipeline, which Stage 1 already verified bit-identical
    #    across two runs).
    summary = pd.read_csv(TABLE_PATHS["summary"]).set_index("year")
    valid_mask = _read_valid_mask()
    for year in YEARS:
        with rasterio.open(suitability_class_raster_path(year)) as src:
            classes = src.read(1)
        counts = np.bincount(classes[classes != -1].astype(np.int64), minlength=5)
        expected = np.array(
            [int(summary.loc[year, f"class_{k}_{CLASS_LABELS[k].lower().replace(' ', '_')}_count"]) for k in range(5)]
        )
        checks.append(
            _check(
                f"{year} class raster histogram matches suitability_summary (exact)",
                "reproducibility",
                bool((counts == expected).all()),
                f"raster={counts.tolist()}, summary={expected.tolist()}",
            )
        )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return checks


# ---------------------------------------------------------------------------
# Pipeline record (idempotent Stage 4 entry)
# ---------------------------------------------------------------------------
def _append_stage4_record(report: Dict, repro: bool) -> None:
    """Append/replace the Stage 4 entry in the Phase 7 pipeline record."""
    from datetime import datetime, timezone

    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    with open(PIPELINE_RECORD_JSON) as f:
        existing = json.load(f)

    container = {"phase": 7, "project": "GreenGrid-AI", "stage_records": []}
    if "stage_records" in existing:
        container["stage_records"] = existing["stage_records"]
    else:
        container["stage_records"] = [existing]
    container["stage_records"] = [
        r for r in container["stage_records"] if r.get("stage") != 4
    ]
    summary = report["summary"]
    container["stage_records"].append(
        {
            "phase": 7,
            "stage": 4,
            "project": "GreenGrid-AI",
            "project_root": str(PROJECT_ROOT),
            "started_utc": report["generated_utc"],
            "finished_utc": _now(),
            "random_seed": RANDOM_SEED,
            "status": "success" if summary["fail"] == 0 else "failed_validation",
            "validator": "suitability.validate_suitability",
            "repro_run": repro,
            "validation_summary": summary,
            "validation_report": str(PHASE7_VALIDATION_REPORT_JSON),
            "categories": {
                category: {
                    "pass": sum(1 for c in report["checks"] if c["category"] == category and c["status"] == "PASS"),
                    "fail": sum(1 for c in report["checks"] if c["category"] == category and c["status"] == "FAIL"),
                    "warn": sum(1 for c in report["checks"] if c["category"] == category and c["status"] == "WARN"),
                }
                for category in sorted({c["category"] for c in report["checks"]})
            },
            "failed_checks": [c["name"] for c in report["checks"] if c["status"] == "FAIL"],
            "note": (
                "Independent validator (design spec section 33/34): verifies "
                "upstream integrity, raster alignment, score/class "
                "consistency, zones, temporal, sensitivity, provenance and "
                "leakage; --repro additionally re-runs suitability.zones "
                "demanding bit-identical tables and recomputes class "
                "histograms.  Idempotent: re-running replaces this entry."
            ),
        }
    )
    with open(PIPELINE_RECORD_JSON, "w") as f:
        json.dump(container, f, indent=2, default=str)
    print(f"  Pipeline record updated: {PIPELINE_RECORD_JSON}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_validation(output_path: str = str(PHASE7_VALIDATION_REPORT_JSON), repro: bool = False) -> Dict:
    """Run all Phase 7 validation checks and write a report."""
    checks: List[Dict] = []

    manifest = json.loads(Path(MANIFEST_JSON).read_text()) if Path(MANIFEST_JSON).exists() else {}
    valid_mask = _read_valid_mask()

    checks.extend(validate_inputs())
    checks.extend(validate_alignment())
    checks.extend(validate_scores(valid_mask))
    checks.extend(validate_classes(valid_mask))
    checks.extend(validate_zones(valid_mask))
    checks.extend(validate_temporal(valid_mask))
    checks.extend(validate_sensitivity(valid_mask))
    checks.extend(validate_provenance(manifest))
    checks.extend(validate_leakage())
    if repro:
        checks.extend(run_reproducibility_checks())
    else:
        checks.append(
            _warn(
                "reproducibility checks not run (use --repro)",
                "reproducibility",
                "zones idempotence and class-histogram repro are only executed with --repro",
            )
        )

    summary = {
        "total": len(checks),
        "pass": sum(1 for c in checks if c["status"] == "PASS"),
        "fail": sum(1 for c in checks if c["status"] == "FAIL"),
        "warn": sum(1 for c in checks if c["status"] == "WARN"),
    }

    report = {
        "phase": 7,
        "generated_utc": pd.Timestamp.now("UTC").isoformat(),
        "repro_run": repro,
        "summary": summary,
        "checks": checks,
        "grid_spec": manifest.get("grid_spec", {}),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    _append_stage4_record(report, repro)
    return report


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for Phase 7 validation."""
    parser = argparse.ArgumentParser(description="Validate GreenGrid-AI Phase 7 outputs.")
    parser.add_argument(
        "--output",
        default=str(PHASE7_VALIDATION_REPORT_JSON),
        help="Path to write the validation report JSON.",
    )
    parser.add_argument(
        "--repro",
        action="store_true",
        help="Also run reproducibility checks (re-runs suitability.zones; ~1 min).",
    )
    args = parser.parse_args(argv)

    report = run_validation(output_path=args.output, repro=args.repro)
    summary = report["summary"]
    print(f"Phase 7 Validation: {summary['pass']} PASS / {summary['fail']} FAIL / {summary['warn']} WARN")

    for check in report["checks"]:
        if check["status"] != "PASS":
            print(f"  {check['status']}: {check['name']} — {check['message']}")

    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
