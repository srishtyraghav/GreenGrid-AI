"""Independent validation script for Phase 6 outputs.

Verifies input integrity, raster/grid conformance, severity/probability
consistency, hotspot delineation quality, analytics-table internal
consistency, temporal statistics, NoData discipline, OOF-based error
diagnostics, and (optionally, via ``--repro``) the reproducibility of the
Stage 3 analytics tables and the Stage 1 severity rasters.

The check record pattern is cloned from ``models.validate_models``:
``_check(name, category, condition, message)`` with categories
inputs / rasters / severity / probabilities / hotspots /
area_statistics / green_built / temporal / nodata / error_diagnostics /
phase5_integrity / reproducibility.

Run with::

    PYTHONPATH=src python3 -m severity.validate_severity
    PYTHONPATH=src python3 -m severity.validate_severity --repro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from models.config import PHASE5_DIR, REQUIRED_INPUT_COLS

from . import analytics
from .analytics import area_statistics_path, compute_pixel_area_m2, oof_error_map_path, uncertainty_zone_path
from .config import (
    FULLGRID_SUMMARY_CSV,
    HOTSPOT_NODATA,
    HOTSPOT_SENSITIVITY_CSV,
    INPUT_DATASET_CSV,
    MANIFEST_JSON,
    MIN_HOTSPOT_PIXELS,
    MODEL_PATH,
    PHASE5_OOF_PREDICTIONS_CSV,
    PHASE5_PREDICTIONS_CSV,
    PHASE5_UHI_RASTER_2022,
    PHASE6_FIGURES_DIR,
    PHASE6_REPORTS_DIR,
    PHASE6_TABLES_DIR,
    PIPELINE_RECORD_JSON,
    POLYGON_AREA_TOLERANCE,
    SEVERITY_NODATA,
    UTM_CRS,
    VALID_MASK_RASTER,
    confidence_raster_path,
    hotspot_boundaries_path,
    hotspot_mask_path,
    hotspot_statistics_path,
    lst_raster_path,
    probability_raster_path,
    severity_raster_path,
    severity_score_raster_path,
)
from .fullgrid import load_valid_cells

PHASE6_VALIDATION_REPORT_JSON = PHASE6_REPORTS_DIR / "phase6_validation_report.json"

# Frozen grid facts (from the Phase 6 manifest grid_spec and raster census).
VALID_PIXELS_PER_YEAR = 1_500_777
LST_VALID_PIXELS = {2022: 1_743_888, 2026: 1_894_985}
OOF_VALID_PIXELS = 150_000  # per year: sampled cells only
GRID_SHAPE = (1768, 1874)
YEARS = (2022, 2026)
DEFINITIONS = ("A", "B", "C")

# Sampled consistency-check size (seed 42, per year).
CONSISTENCY_SAMPLE_N = 50_000

# Empirical band for the pooled TRUE-OOF accuracy of the frozen Phase 5
# baseline Random Forest.  Guards against silently switching to the
# production model's IN-SAMPLE predictions.csv (accuracy ~0.9957).
OOF_ACCURACY_BAND = (0.42, 0.50)

# Frozen Phase 5 headline numbers (do-not-break checks).
PHASE5_EXPECTED_SUMMARY = {"total": 57, "pass": 57, "fail": 0, "warn": 0}
PHASE5_RF_SPATIAL_ACC = 0.4628
PHASE5_RF_SPATIAL_F1 = 0.4306
PHASE5_RFC_ACC = 0.4699
PHASE5_RFC_F1 = 0.4310
PHASE5_HOLDOUT_ACC = 0.3985
PHASE5_HOLDOUT_F1 = 0.3952

PHASE5_TABLES = PHASE5_DIR / "tables"
PHASE5_GENERALIZATION_JSON = PHASE5_TABLES / "phase5_generalization_summary.json"
PHASE5_VALIDATION_REPORT_JSON = PHASE5_DIR / "phase5_validation_report.json"


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


def _raster_profile_facts(path: Path) -> Dict:
    """Read (array, crs, transform, nodata, compression) for one raster."""
    with rasterio.open(path) as src:
        return {
            "array": src.read(1),
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
            "compression": src.compression,
            "shape": (src.height, src.width),
        }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def validate_inputs(manifest: Dict) -> List[Dict]:
    """Validate frozen upstream inputs and the presence of all stage outputs."""
    checks: List[Dict] = []
    print("Validating inputs ...")

    # Phase 4 dataset.
    df = pd.read_csv(INPUT_DATASET_CSV)
    checks.append(
        _check(
            "Phase 4 dataset shape 300000x14",
            "inputs",
            df.shape == (300_000, 14),
            f"shape={df.shape}",
        )
    )
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    checks.append(
        _check(
            "Phase 4 required columns present",
            "inputs",
            len(missing) == 0,
            f"missing={missing}" if missing else f"columns={list(df.columns)}",
        )
    )
    checks.append(
        _check(
            "Phase 4 years and per-year row counts",
            "inputs",
            set(df["year"].unique()) == set(YEARS)
            and int((df["year"] == 2022).sum()) == 150_000
            and int((df["year"] == 2026).sum()) == 150_000,
            f"year counts={df['year'].value_counts().to_dict()}",
        )
    )
    del df

    # Phase 5 frozen predictions.  predictions.csv holds the production
    # model's 300,000 in-sample rows; oof_predictions.csv holds every
    # model's out-of-fold rows (5 models x 300,000 = 1,500,000).
    for name, path, expected in [
        ("Phase 5 predictions", PHASE5_PREDICTIONS_CSV, 300_000),
        ("Phase 5 OOF predictions", PHASE5_OOF_PREDICTIONS_CSV, 1_500_000),
    ]:
        exists = Path(path).exists()
        n_rows = len(pd.read_csv(path)) if exists else 0
        checks.append(
            _check(
                f"{name} exist with expected rows",
                "inputs",
                exists and n_rows == expected,
                f"{path} rows={n_rows:,}, expected={expected:,}",
            )
        )

    # Every output registered in the manifest (stages 1-4) must exist.
    def _manifest_files() -> List[Path]:
        files: List[Path] = [FULLGRID_SUMMARY_CSV, MODEL_PATH, PIPELINE_RECORD_JSON]
        for year in YEARS:
            files.append(severity_raster_path(year))
            files.append(confidence_raster_path(year))
            files.append(severity_score_raster_path(year))
            files.append(lst_raster_path(year))
            files.append(oof_error_map_path(year))
            files.append(uncertainty_zone_path(year))
            for label in ("low", "moderate", "high", "severe"):
                files.append(probability_raster_path(label, year))
        for year_s, defs in manifest.get("hotspots", {}).get("outputs", {}).items():
            for def_s, entry in defs.items():
                files.extend(Path(p) for p in entry.values())
        tables = manifest.get("analytics", {}).get("tables", {})
        for value in tables.values():
            if isinstance(value, str):
                files.append(Path(value))
            elif isinstance(value, dict):
                files.extend(Path(p) for p in value.values())
        files.extend(Path(p) for p in manifest.get("figures", {}).get("figures", {}).values())
        return files

    files = _manifest_files()
    missing_files = [str(p) for p in files if not p.exists()]
    empty_files = [str(p) for p in files if p.exists() and p.stat().st_size == 0]
    checks.append(
        _check(
            "All manifest-registered Phase 6 outputs exist",
            "inputs",
            len(missing_files) == 0,
            f"checked={len(files)}, missing={missing_files[:5]}",
        )
    )
    checks.append(
        _check(
            "All manifest-registered Phase 6 outputs non-empty",
            "inputs",
            len(empty_files) == 0,
            f"empty={empty_files[:5]}",
        )
    )
    n_figures = len(manifest.get("figures", {}).get("figures", {}))
    checks.append(
        _check(
            "Stage 4 figure suite complete (16 figures)",
            "inputs",
            n_figures == 16
            and all((PHASE6_FIGURES_DIR / f"fig{i:02d}_{suffix}").exists() for i, suffix in [
                (1, "lst_2022.png"), (2, "lst_2026.png"), (3, "severity_2022.png"),
                (4, "severity_2026.png"), (5, "hotspots_2022.png"), (6, "hotspots_2026.png"),
                (7, "temporal_comparison.png"), (8, "green_vs_built.png"),
                (9, "ndvi_vs_lst.png"), (10, "ndbi_vs_lst.png"),
                (11, "high_severe_by_block.png"), (12, "hotspot_size_distribution.png"),
                (13, "confidence_map.png"), (14, "uncertainty.png"),
                (15, "transition_matrix.png"), (16, "landuse_heat.png"),
            ]),
            f"figures listed={n_figures}",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Raster integrity
# ---------------------------------------------------------------------------
def validate_rasters() -> List[Dict]:
    """Verify CRS/transform/dims/nodata/compression/valid-counts of all rasters."""
    checks: List[Dict] = []
    print("Validating rasters ...")

    with rasterio.open(VALID_MASK_RASTER) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs

    def _grid_check(raster_path: Path, label: str, expected_valid: int, expect_nodata) -> None:
        exists = Path(raster_path).exists()
        if not exists:
            checks.append(_check(f"{label} exists", "rasters", False, str(raster_path)))
            return
        facts = _raster_profile_facts(raster_path)
        ok_crs = facts["crs"] == ref_crs and facts["crs"].to_epsg() == 4326
        ok_shape = facts["shape"] == GRID_SHAPE
        ok_transform = facts["transform"] == ref_transform
        ok_nodata = facts["nodata"] is not None
        if expect_nodata is not None and ok_nodata:
            ok_nodata = float(facts["nodata"]) == float(expect_nodata)
        ok_compress = facts["compression"] is not None and str(facts["compression"]).lower() in (
            "lzw", "deflate", "compression.lzw", "compression.deflate",
        )
        n_valid = int((facts["array"] != facts["nodata"]).sum()) if ok_nodata else -1
        ok_count = n_valid == expected_valid
        checks.append(
            _check(
                f"{label} grid conformance",
                "rasters",
                ok_crs and ok_shape and ok_transform and ok_nodata and ok_compress,
                f"crs={facts['crs']}, shape={facts['shape']}, transform_match={ok_transform}, "
                f"nodata={facts['nodata']}, compression={facts['compression']}",
            )
        )
        checks.append(
            _check(
                f"{label} valid pixel count",
                "rasters",
                ok_count,
                f"n_valid={n_valid:,}, expected={expected_valid:,}",
            )
        )
        if label.startswith("severity "):
            vals = set(np.unique(facts["array"])).difference({facts["nodata"]})
            checks.append(
                _check(
                    f"{label} values subset of {{0,1,2,3}}+nodata",
                    "rasters",
                    vals.issubset({0, 1, 2, 3}),
                    f"unique={sorted(vals)}",
                )
            )

    for year in YEARS:
        _grid_check(severity_raster_path(year), f"severity {year}", VALID_PIXELS_PER_YEAR, SEVERITY_NODATA)
        for label in ("low", "moderate", "high", "severe"):
            _grid_check(
                probability_raster_path(label, year),
                f"probability_{label} {year}",
                VALID_PIXELS_PER_YEAR,
                -1.0,
            )
        _grid_check(confidence_raster_path(year), f"confidence {year}", VALID_PIXELS_PER_YEAR, -1.0)
        _grid_check(severity_score_raster_path(year), f"severity_score {year}", VALID_PIXELS_PER_YEAR, -1.0)
        _grid_check(lst_raster_path(year), f"lst {year}", LST_VALID_PIXELS[year], -1.0)
        _grid_check(oof_error_map_path(year), f"oof_error_map {year}", OOF_VALID_PIXELS, -1)
        _grid_check(uncertainty_zone_path(year), f"uncertainty_zone {year}", VALID_PIXELS_PER_YEAR, 255)
        for definition in DEFINITIONS:
            _grid_check(
                hotspot_mask_path(definition, year),
                f"hotspot_mask def{definition} {year}",
                VALID_PIXELS_PER_YEAR,
                HOTSPOT_NODATA,
            )
    return checks


# ---------------------------------------------------------------------------
# Severity / probability consistency (seed-42 50k sample per year)
# ---------------------------------------------------------------------------
def validate_severity_probabilities() -> List[Dict]:
    """Cross-validate probabilities, confidence, score and class on a sample."""
    checks: List[Dict] = []
    print("Validating severity/probability consistency (50k seed-42 sample per year) ...")

    for year in YEARS:
        rng = np.random.default_rng(42)
        probs = []
        for label in ("low", "moderate", "high", "severe"):
            with rasterio.open(probability_raster_path(label, year)) as src:
                probs.append(src.read(1))
        with rasterio.open(confidence_raster_path(year)) as src:
            conf = src.read(1)
        with rasterio.open(severity_score_raster_path(year)) as src:
            score = src.read(1)
        with rasterio.open(severity_raster_path(year)) as src:
            sev = src.read(1)

        valid = sev != SEVERITY_NODATA
        rows, cols = np.where(valid)
        sel = rng.choice(len(rows), size=min(CONSISTENCY_SAMPLE_N, len(rows)), replace=False)
        sr, sc = rows[sel], cols[sel]
        p = np.stack([pr[sr, sc] for pr in probs], axis=1).astype(np.float64)
        c = conf[sr, sc].astype(np.float64)
        s = score[sr, sc].astype(np.float64)
        k = sev[sr, sc].astype(np.int64)

        checks.append(
            _check(
                f"{year} per-class probabilities in [0, 1]",
                "probabilities",
                bool((p >= 0).all() and (p <= 1).all()),
                f"min={p.min():.6f}, max={p.max():.6f}",
            )
        )
        sums = p.sum(axis=1)
        checks.append(
            _check(
                f"{year} probability sums ~ 1 (±1e-3)",
                "probabilities",
                bool(np.all(np.abs(sums - 1.0) <= 1e-3)),
                f"max |sum-1|={np.abs(sums - 1.0).max():.2e}",
            )
        )
        checks.append(
            _check(
                f"{year} confidence == max class probability",
                "probabilities",
                bool(np.allclose(c, p.max(axis=1), atol=1e-5)),
                f"max |diff|={np.abs(c - p.max(axis=1)).max():.2e}",
            )
        )
        expected_score = p @ np.array([0.0, 1.0, 2.0, 3.0])
        checks.append(
            _check(
                f"{year} severity_score == sum k*P(k)",
                "probabilities",
                bool(np.allclose(s, expected_score, atol=1e-4)),
                f"max |diff|={np.abs(s - expected_score).max():.2e}",
            )
        )
        checks.append(
            _check(
                f"{year} severity class == argmax probability",
                "severity",
                bool((k == p.argmax(axis=1)).all()),
                f"mismatches={(k != p.argmax(axis=1)).sum()}",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Hotspots
# ---------------------------------------------------------------------------
def validate_hotspots() -> List[Dict]:
    """Validate hotspot masks, GeoJSON boundaries and attribute tables."""
    checks: List[Dict] = []
    print("Validating hotspots ...")

    _, _, transform, mask_profile = load_valid_cells()
    pixel_area = compute_pixel_area_m2(transform, mask_profile["height"], mask_profile["width"])
    mean_px_area = float(pixel_area.mean())

    seen_keys = set()
    dup_keys: List[str] = []
    for year in YEARS:
        for definition in DEFINITIONS:
            label = f"{year} def{definition}"
            mask_path = hotspot_mask_path(definition, year)
            boundaries_path = hotspot_boundaries_path(definition, year)
            stats_path = hotspot_statistics_path(definition, year)

            stats = pd.read_csv(stats_path)
            with rasterio.open(mask_path) as src:
                mask_arr = src.read(1)
            mask_hotspot_px = int((mask_arr > 0).sum())

            gdf = gpd.read_file(boundaries_path)
            invalid = int((~gdf.geometry.is_valid).sum())
            nulls = int(stats[["area_ha", "mean_lst_C", "pixel_count"]].isna().sum().sum())

            keys = list(zip(stats["hotspot_id"], stats["year"], stats["definition"]))
            for key in keys:
                if key in seen_keys:
                    dup_keys.append(str(key))
                seen_keys.add(key)

            sum_px = int(stats["pixel_count"].sum())
            min_px = int(stats["pixel_count"].min())

            # Areas are only meaningful in an equal-area CRS (UTM 43N).
            gdf_utm = gdf.to_crs(UTM_CRS)
            zero_area = int((gdf_utm.geometry.area <= 0).sum())
            rel_err = np.abs(
                gdf_utm.geometry.area.values / (stats["pixel_count"].values * mean_px_area) - 1.0
            )
            max_rel_err = float(rel_err.max())

            checks.append(
                _check(
                    f"{label} GeoJSON loads, geometries valid and non-zero-area",
                    "hotspots",
                    len(gdf) == len(stats) and invalid == 0 and zero_area == 0,
                    f"n_features={len(gdf)}, n_stats_rows={len(stats)}, invalid={invalid}, zero_area={zero_area}",
                )
            )
            checks.append(
                _check(
                    f"{label} attribute completeness (area_ha, mean_lst_C, pixel_count)",
                    "hotspots",
                    nulls == 0,
                    f"nulls={nulls}",
                )
            )
            checks.append(
                _check(
                    f"{label} sum(pixel_count) == mask hotspot pixels",
                    "hotspots",
                    sum_px == mask_hotspot_px,
                    f"sum={sum_px:,}, mask={mask_hotspot_px:,}",
                )
            )
            checks.append(
                _check(
                    f"{label} polygon area within 5% of pixel_count x mean pixel area",
                    "hotspots",
                    max_rel_err <= POLYGON_AREA_TOLERANCE,
                    f"max rel err={max_rel_err:.6f} (mean_px_area={mean_px_area:.2f} m2)",
                )
            )
            checks.append(
                _check(
                    f"{label} min cluster >= {MIN_HOTSPOT_PIXELS} px",
                    "hotspots",
                    min_px >= MIN_HOTSPOT_PIXELS,
                    f"min pixel_count={min_px}",
                )
            )

    checks.append(
        _check(
            "hotspot (hotspot_id, year, definition) unique across all files",
            "hotspots",
            len(dup_keys) == 0,
            f"duplicates={dup_keys[:5]}",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Area statistics
# ---------------------------------------------------------------------------
def validate_area_statistics() -> List[Dict]:
    """Validate per-block area statistics internal consistency."""
    checks: List[Dict] = []
    print("Validating area statistics ...")
    pct_cols = ["low_percent", "moderate_percent", "high_percent", "severe_percent"]
    count_cols = ["low_count", "moderate_count", "high_count", "severe_count"]

    for year in YEARS:
        tbl = pd.read_csv(area_statistics_path(year))
        n_valid = VALID_PIXELS_PER_YEAR

        per_row_ok = bool(
            (tbl[count_cols].sum(axis=1).values == tbl["valid_pixel_count"].values).all()
        )
        checks.append(
            _check(
                f"{year} per-block class counts sum to block valid pixels",
                "area_statistics",
                per_row_ok and int(tbl["valid_pixel_count"].sum()) == n_valid,
                f"blocks={len(tbl)}, total_valid={int(tbl['valid_pixel_count'].sum()):,}",
            )
        )
        pct_sum = tbl[pct_cols].sum(axis=1)
        checks.append(
            _check(
                f"{year} per-block class percents sum to 100 ±0.1",
                "area_statistics",
                bool(((pct_sum - 100.0).abs() <= 0.1).all()),
                f"max |sum-100|={(pct_sum - 100.0).abs().max():.4f}",
            )
        )
        hs_recompute = 100.0 * (tbl["high_count"] + tbl["severe_count"]) / tbl["valid_pixel_count"]
        checks.append(
            _check(
                f"{year} high_severe_percent == (high+severe)/valid x100",
                "area_statistics",
                bool((np.abs(tbl["high_severe_percent"] - hs_recompute) <= 0.01).all()),
                f"max diff={np.abs(tbl['high_severe_percent'] - hs_recompute).max():.4f}",
            )
        )
        hotspot_le_hs = tbl["hotspot_percentage"] <= tbl["high_severe_percent"] + 0.01
        checks.append(
            _check(
                f"{year} hotspot_percentage <= high_severe_percent per block",
                "area_statistics",
                bool(hotspot_le_hs.all()),
                f"violations={(~hotspot_le_hs).sum()}",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Green / built
# ---------------------------------------------------------------------------
def validate_green_built() -> List[Dict]:
    """Validate green-vs-built grouping; LST ordering is an empirical WARN."""
    checks: List[Dict] = []
    print("Validating green/built comparison ...")

    gb = pd.read_csv(PHASE6_TABLES_DIR / "green_built_comparison.csv")
    for year in YEARS:
        sub = gb[gb["year"] == year]
        total = int(sub["pixel_count"].sum())
        counts_sum = int(sub[["low_count", "moderate_count", "high_count", "severe_count"]].sum().sum())
        checks.append(
            _check(
                f"{year} green/built group counts sum to valid total",
                "green_built",
                total == VALID_PIXELS_PER_YEAR and counts_sum == total,
                f"groups={len(sub)}, sum={total:,}",
            )
        )
        hs_recompute = 100.0 * (sub["high_count"] + sub["severe_count"]) / sub["pixel_count"]
        checks.append(
            _check(
                f"{year} green/built high_severe fractions consistent with counts",
                "green_built",
                bool((np.abs(sub["high_severe_percent"] - hs_recompute) <= 0.01).all()),
                f"max diff={np.abs(sub['high_severe_percent'] - hs_recompute).max():.4f}",
            )
        )
        by_group = sub.set_index("group")
        if "built_dominant" in by_group.index and "green_dominant" in by_group.index:
            built_mean = float(by_group.loc["built_dominant", "mean_lst_C"])
            green_mean = float(by_group.loc["green_dominant", "mean_lst_C"])
            holds = built_mean >= green_mean
            msg = f"built={built_mean:.4f} C, green={green_mean:.4f} C, gap={built_mean - green_mean:+.4f} C"
            # Empirical association, not an invariant: report as WARN-level
            # observation when violated, PASS when the expected ordering holds.
            checks.append(
                _check(f"{year} built mean LST >= green mean LST", "green_built", holds, msg)
                if holds
                else _warn(f"{year} built mean LST >= green mean LST", "green_built", msg)
            )
    return checks


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------
def validate_temporal() -> List[Dict]:
    """Validate temporal comparison, transition matrix and hotspot change."""
    checks: List[Dict] = []
    print("Validating temporal statistics ...")

    transition = pd.read_csv(PHASE6_TABLES_DIR / "temporal_transition_matrix.csv")
    total = int(transition["count"].sum())
    paired = int(transition["paired_cell_count"].iloc[0])
    checks.append(
        _check(
            "transition matrix total == paired valid count",
            "temporal",
            total == VALID_PIXELS_PER_YEAR and paired == VALID_PIXELS_PER_YEAR,
            f"matrix_total={total:,}, paired={paired:,}",
        )
    )
    row_totals = transition.groupby("severity_2022")["count"].sum()
    declared = transition.groupby("severity_2022")["row_total_2022_class"].first()
    prob_ok = bool(
        (
            np.round(transition["count"] / transition["row_total_2022_class"], 6)
            == transition["row_probability_2026_given_2022"]
        ).all()
    )
    checks.append(
        _check(
            "transition matrix row totals and row probabilities consistent",
            "temporal",
            bool((row_totals.values == declared.values).all()) and prob_ok,
            "",
        )
    )

    change = pd.read_csv(PHASE6_TABLES_DIR / "hotspot_change_statistics.csv")
    cat_rows = change[change["change_category"] != "mask_iou_2022_vs_2026"]
    cat_sum = int(cat_rows["pixel_count"].sum())
    checks.append(
        _check(
            "hotspot change category counts sum to paired count",
            "temporal",
            cat_sum == int(cat_rows["paired_cell_count"].iloc[0]) == VALID_PIXELS_PER_YEAR,
            f"sum={cat_sum:,}, paired={int(cat_rows['paired_cell_count'].iloc[0]):,}",
        )
    )

    # IoU row must agree with a direct recomputation from the defA masks.
    m22 = None
    m26 = None
    with rasterio.open(hotspot_mask_path("A", 2022)) as src:
        m22 = src.read(1) > 0
    with rasterio.open(hotspot_mask_path("A", 2026)) as src:
        m26 = src.read(1) > 0
    inter = int((m22 & m26).sum())
    union = int((m22 | m26).sum())
    iou_row = change[change["change_category"] == "mask_iou_2022_vs_2026"].iloc[0]
    checks.append(
        _check(
            "hotspot change IoU consistent with defA masks",
            "temporal",
            inter == int(iou_row["intersection_px"])
            and union == int(iou_row["union_px"])
            and abs(inter / union - float(iou_row["mask_iou"])) < 1e-4,
            f"recomputed IoU={inter / union:.6f}, table={float(iou_row['mask_iou']):.6f}",
        )
    )

    # 2022-only / 2026-only stats on correct-year data: spot-check mean LST
    # between severity_summary, temporal_comparison and the LST rasters.
    sev_sum = pd.read_csv(PHASE6_TABLES_DIR / "severity_summary.csv").set_index("year")
    temp_cmp = pd.read_csv(PHASE6_TABLES_DIR / "temporal_comparison.csv").set_index("year")
    for year in YEARS:
        with rasterio.open(lst_raster_path(year)) as src:
            lst = src.read(1).astype(np.float64)
        with rasterio.open(severity_raster_path(year)) as src:
            sev = src.read(1)
        valid = sev != SEVERITY_NODATA
        raster_mean = float(lst[valid & np.isfinite(lst)].mean())
        tbl_mean = float(sev_sum.loc[year, "mean_lst_C"])
        cmp_mean = float(temp_cmp.loc[year, "mean_lst_C"])
        checks.append(
            _check(
                f"{year} mean LST consistent (raster vs tables, ±0.1 C)",
                "temporal",
                abs(raster_mean - tbl_mean) <= 0.1 and abs(tbl_mean - cmp_mean) <= 0.1,
                f"raster={raster_mean:.4f}, severity_summary={tbl_mean:.4f}, temporal_comparison={cmp_mean:.4f}",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# NoData discipline
# ---------------------------------------------------------------------------
def validate_nodata() -> List[Dict]:
    """Verify NoData never enters statistics and never masquerades as Low."""
    checks: List[Dict] = []
    print("Validating NoData discipline ...")

    fullgrid = pd.read_csv(FULLGRID_SUMMARY_CSV).set_index("year")
    high_risk = pd.read_csv(PHASE6_TABLES_DIR / "high_risk_summary.csv").set_index("year")
    grid_total = GRID_SHAPE[0] * GRID_SHAPE[1]

    for year in YEARS:
        with rasterio.open(severity_raster_path(year)) as src:
            sev = src.read(1)
        nodata_mask = sev == SEVERITY_NODATA
        valid = ~nodata_mask
        low_count = int((sev == 0).sum())  # value 0 only; nodata (-1) excluded
        checks.append(
            _check(
                f"{year} severity nodata count and alignment",
                "nodata",
                int(nodata_mask.sum()) == grid_total - VALID_PIXELS_PER_YEAR,
                f"nodata={int(nodata_mask.sum()):,}, expected={grid_total - VALID_PIXELS_PER_YEAR:,}",
            )
        )
        checks.append(
            _check(
                f"{year} Low count from raster equals table counts (NoData != Low)",
                "nodata",
                low_count == int(fullgrid.loc[year, "class_0_low_count"])
                and low_count == int(high_risk.loc[year, "low_count"]),
                f"raster={low_count:,}, fullgrid={int(fullgrid.loc[year, 'class_0_low_count']):,}, "
                f"high_risk={int(high_risk.loc[year, 'low_count']):,}",
            )
        )

        with rasterio.open(uncertainty_zone_path(year)) as src:
            unc = src.read(1)
        checks.append(
            _check(
                f"{year} uncertainty_zone nodata outside declared domain",
                "nodata",
                set(np.unique(unc[valid])).issubset({0, 1}) and bool((unc[~valid] == 255).all()),
                f"valid values={sorted(np.unique(unc[valid]))}",
            )
        )

        with rasterio.open(oof_error_map_path(year)) as src:
            err = src.read(1)
        err_nodata = err == -1
        checks.append(
            _check(
                f"{year} oof_error_map nodata outside sampled-cell domain",
                "nodata",
                int((~err_nodata).sum()) == OOF_VALID_PIXELS
                and set(np.unique(err[~err_nodata])).issubset({0, 1}),
                f"valid={int((~err_nodata).sum()):,}, expected={OOF_VALID_PIXELS:,}",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Error diagnostics (TRUE OOF only)
# ---------------------------------------------------------------------------
def validate_error_diagnostics() -> List[Dict]:
    """Validate OOF-based error diagnostics against oof_predictions.csv."""
    checks: List[Dict] = []
    print("Validating error diagnostics (TRUE OOF) ...")

    oof = pd.read_csv(
        PHASE5_OOF_PREDICTIONS_CSV,
        usecols=[
            "row", "col", "year", "spatial_block_id", "actual_class",
            "predicted_class", "prediction_confidence", "model",
        ],
    )
    rf = oof[oof["model"] == "Random Forest"].copy()
    checks.append(
        _check(
            "OOF Random Forest rows == 300000",
            "error_diagnostics",
            len(rf) == 300_000,
            f"rows={len(rf):,}",
        )
    )
    rf["correct"] = rf["actual_class"].values == rf["predicted_class"].values
    pooled_acc = float(rf["correct"].mean())
    lo, hi = OOF_ACCURACY_BAND
    checks.append(
        _check(
            "OOF pooled accuracy in empirical band (guards in-sample predictions)",
            "error_diagnostics",
            lo <= pooled_acc <= hi,
            f"pooled={pooled_acc:.6f}, band=[{lo}, {hi}]",
        )
    )

    summary = pd.read_csv(PHASE6_TABLES_DIR / "error_diagnostics_summary.csv")
    overall = summary[summary["analysis"] == "overall_accuracy"].set_index("year")
    per_year_ok = True
    for year in YEARS:
        acc = float(rf.loc[rf["year"] == year, "correct"].mean())
        per_year_ok &= round(acc, 6) == float(overall.loc[year, "accuracy"])
    checks.append(
        _check(
            "error_diagnostics_summary overall accuracy matches OOF recompute",
            "error_diagnostics",
            per_year_ok,
            f"table={overall['accuracy'].to_dict()}",
        )
    )

    by_block = pd.read_csv(PHASE6_TABLES_DIR / "error_by_block.csv")
    checks.append(
        _check(
            "error_by_block has 40 rows (20 blocks x 2 years)",
            "error_diagnostics",
            len(by_block) == 40,
            f"rows={len(by_block)}",
        )
    )
    recomputed = (
        rf.groupby(["year", "spatial_block_id"])["correct"]
        .agg(n_sampled="size", error_rate=lambda c: round(1.0 - float(c.mean()), 6))
        .reset_index()
    )
    merged = by_block.merge(
        recomputed,
        left_on=["year", "block_id"],
        right_on=["year", "spatial_block_id"],
        how="left",
    )
    block_ok = bool(
        (merged["n_sampled_x"] == merged["n_sampled_y"]).all()
        and (merged["error_rate_x"] == merged["error_rate_y"]).all()
    )
    checks.append(
        _check(
            "error_by_block matches OOF recompute",
            "error_diagnostics",
            block_ok,
            f"blocks={len(by_block)}, n_sampled per year={by_block.groupby('year')['n_sampled'].sum().to_dict()}",
        )
    )

    # Error-map raster must carry the OOF correctness flags at sampled cells.
    map_ok = True
    map_msg = ""
    for year in YEARS:
        with rasterio.open(oof_error_map_path(year)) as src:
            err = src.read(1)
        sub = rf[rf["year"] == year]
        flags = err[sub["row"].values, sub["col"].values]
        ok = bool((flags == sub["correct"].values.astype(np.int16)).all())
        map_ok &= ok
        map_msg += f"{year}: {'ok' if ok else 'MISMATCH'}; "
    checks.append(
        _check(
            "oof_error_map rasters match OOF correctness flags",
            "error_diagnostics",
            map_ok,
            map_msg.strip(),
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Phase 5 integrity (do-not-break checks)
# ---------------------------------------------------------------------------
def validate_phase5_integrity() -> List[Dict]:
    """Verify the frozen Phase 5 outputs are present and numerically unchanged."""
    checks: List[Dict] = []
    print("Validating Phase 5 integrity ...")

    frozen_files = [
        PHASE5_PREDICTIONS_CSV,
        PHASE5_OOF_PREDICTIONS_CSV,
        PHASE5_TABLES / "spatial_neighbourhood_features.csv",
        PHASE5_TABLES / "morphology_features.csv",
        PHASE5_TABLES / "model_comparison.csv",
        PHASE5_TABLES / "locked_geographic_holdout.csv",
        PHASE5_GENERALIZATION_JSON,
        PHASE5_VALIDATION_REPORT_JSON,
        PHASE5_UHI_RASTER_2022,
    ]
    missing = [str(p) for p in frozen_files if not Path(p).exists()]
    checks.append(
        _check(
            "Phase 5 frozen files present",
            "phase5_integrity",
            len(missing) == 0,
            f"missing={missing}",
        )
    )

    if PHASE5_VALIDATION_REPORT_JSON.exists():
        p5_report = json.loads(Path(PHASE5_VALIDATION_REPORT_JSON).read_text())
        summary = p5_report.get("summary", {})
        checks.append(
            _check(
                "Phase 5 validation report still 57 PASS / 0 FAIL / 0 WARN",
                "phase5_integrity",
                summary == PHASE5_EXPECTED_SUMMARY,
                f"summary={summary}",
            )
        )

    if (PHASE5_TABLES / "model_comparison.csv").exists():
        mc = pd.read_csv(PHASE5_TABLES / "model_comparison.csv").set_index("model")
        rf_row = mc.loc["Random Forest"]
        checks.append(
            _check(
                "Phase 5 model_comparison RF spatial numbers frozen",
                "phase5_integrity",
                round(float(rf_row["mean_accuracy"]), 4) == PHASE5_RF_SPATIAL_ACC
                and round(float(rf_row["mean_macro_f1"]), 4) == PHASE5_RF_SPATIAL_F1,
                f"mean_accuracy={rf_row['mean_accuracy']}, mean_macro_f1={rf_row['mean_macro_f1']}",
            )
        )

    if PHASE5_GENERALIZATION_JSON.exists():
        gen = json.loads(Path(PHASE5_GENERALIZATION_JSON).read_text())
        selected = gen.get("selected_model", {})
        cv = selected.get("spatial_cv", {})
        checks.append(
            _check(
                "Phase 5 generalization summary RF-C numbers frozen",
                "phase5_integrity",
                "RF-C" in str(selected.get("model", ""))
                and round(float(cv.get("accuracy", -1)), 4) == PHASE5_RFC_ACC
                and round(float(cv.get("macro_f1", -1)), 4) == PHASE5_RFC_F1,
                f"selected={selected.get('model')}, spatial_cv={cv}",
            )
        )
        holdout = gen.get("locked_geographic_holdout", {}).get("random_forest_selected_rf_c", {})
        checks.append(
            _check(
                "Phase 5 locked holdout RF-C numbers frozen",
                "phase5_integrity",
                round(float(holdout.get("accuracy", -1)), 4) == PHASE5_HOLDOUT_ACC
                and round(float(holdout.get("macro_f1", -1)), 4) == PHASE5_HOLDOUT_F1,
                f"holdout={holdout}",
            )
        )

    if (PHASE5_TABLES / "locked_geographic_holdout.csv").exists():
        hold = pd.read_csv(PHASE5_TABLES / "locked_geographic_holdout.csv")
        rf_c = hold[hold["parameters"].str.contains("max_depth") & hold["parameters"].str.contains("15")]
        if len(rf_c):
            row = rf_c.iloc[0]
            checks.append(
                _check(
                    "Phase 5 locked holdout CSV RF-C numbers frozen",
                    "phase5_integrity",
                    round(float(row["accuracy"]), 4) == PHASE5_HOLDOUT_ACC
                    and round(float(row["macro_f1"]), 4) == PHASE5_HOLDOUT_F1,
                    f"accuracy={row['accuracy']}, macro_f1={row['macro_f1']}",
                )
            )
    return checks


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def run_reproducibility_checks() -> List[Dict]:
    """Re-run cheap deterministic stages and demand bit-identical outputs.

    Heavy stages are not re-run: ``severity.pipeline`` (~15 min model-based
    prediction) is replaced by a lightweight raster-histogram comparison of
    the severity rasters against the frozen fullgrid summary, and
    ``severity.figures`` (rendering) is verified only by presence/size in the
    inputs checks.  ``severity.hotspot_stats`` logic IS re-executed: its
    pairwise-overlap metrics are recomputed directly from the stored hotspot
    mask rasters and compared against ``hotspot_sensitivity.csv``.
    ``severity.analytics`` IS re-run in full (idempotent) and every
    ``tables/*.csv`` must come back bit-identical; on any mismatch the
    pre-repro table copies are restored.
    """
    checks: List[Dict] = []
    print("Running reproducibility checks ...")

    # 1-2. Hash all analytics tables, snapshot them, re-run analytics, compare.
    table_files = sorted(PHASE6_TABLES_DIR.glob("*.csv"))
    before = {p.name: _sha256(p) for p in table_files}
    tmp_dir = Path(tempfile.mkdtemp(prefix="phase6_repro_"))
    backup = tmp_dir / "tables_backup"
    shutil.copytree(PHASE6_TABLES_DIR, backup)

    print("  Re-running severity.analytics (Stage 3) for idempotence ...")
    try:
        analytics.run_stage3()
        rerun_ok = True
        rerun_msg = "run_stage3 completed"
    except Exception as exc:  # pragma: no cover - defensive
        rerun_ok = False
        rerun_msg = f"run_stage3 raised: {exc}"

    after = {p.name: _sha256(p) for p in sorted(PHASE6_TABLES_DIR.glob("*.csv"))}
    missing_after = sorted(set(before) - set(after))
    added_after = sorted(set(after) - set(before))
    mismatched = sorted(
        name for name in before if name in after and before[name] != after[name]
    )
    tables_identical = rerun_ok and not missing_after and not mismatched
    checks.append(
        _check(
            "severity.analytics re-run produces bit-identical tables",
            "reproducibility",
            tables_identical,
            f"{rerun_msg}; mismatched={mismatched}, missing={missing_after}, added={added_after}",
        )
    )
    if not tables_identical:
        # Restore the pre-repro state so a failed repro leaves no damage.
        for p in PHASE6_TABLES_DIR.glob("*.csv"):
            p.unlink()
        shutil.copytree(backup, PHASE6_TABLES_DIR, dirs_exist_ok=True)
        checks.append(
            _warn(
                "pre-repro table state restored after mismatch",
                "reproducibility",
                "tables restored from snapshot; investigate analytics determinism",
            )
        )

    # 3. Lightweight raster-consistency repro: severity class histograms vs
    #    the frozen fullgrid prediction summary (stands in for re-running
    #    severity.pipeline, which is too heavy for validation).
    fullgrid = pd.read_csv(FULLGRID_SUMMARY_CSV).set_index("year")
    for year in YEARS:
        with rasterio.open(severity_raster_path(year)) as src:
            sev = src.read(1)
        counts = np.bincount(sev[sev != SEVERITY_NODATA].astype(np.int64), minlength=4)
        expected = np.array(
            [int(fullgrid.loc[year, f"class_{k}_{label.lower()}_count"]) for k, label in [
                (0, "Low"), (1, "Moderate"), (2, "High"), (3, "Severe"),
            ]]
        )
        checks.append(
            _check(
                f"{year} severity raster histogram matches fullgrid summary",
                "reproducibility",
                bool((counts == expected).all()),
                f"raster={counts.tolist()}, summary={expected.tolist()}",
            )
        )

    # 4. Re-execute the hotspot_stats overlap logic from the stored masks.
    sensitivity = pd.read_csv(HOTSPOT_SENSITIVITY_CSV)
    masks: Dict[int, Dict[str, np.ndarray]] = {}
    for year in YEARS:
        masks[year] = {}
        for definition in DEFINITIONS:
            with rasterio.open(hotspot_mask_path(definition, year)) as src:
                masks[year][definition] = src.read(1) > 0
    overlap_ok = True
    overlap_msg = []
    for year in YEARS:
        pairwise = sensitivity[
            (sensitivity["section"] == "pairwise_overlap") & (sensitivity["year"] == year)
        ]
        for _, row in pairwise.iterrows():
            first, second = row["pair"].split("_vs_")
            mx, my = masks[year][first], masks[year][second]
            inter = int((mx & my).sum())
            union = int((mx | my).sum())
            ok = (
                inter == int(row["intersection_px"])
                and union == int(row["union_px"])
                and abs(inter / union - float(row["iou"])) < 1e-4
            )
            overlap_ok &= ok
            if not ok:
                overlap_msg.append(f"{year} {row['pair']}")
    checks.append(
        _check(
            "hotspot_stats overlaps recompute bit-consistently from masks",
            "reproducibility",
            overlap_ok,
            f"mismatched={overlap_msg}" if overlap_msg else "all pairwise IoU rows match",
        )
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return checks


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_validation(output_path: str = str(PHASE6_VALIDATION_REPORT_JSON), repro: bool = False) -> Dict:
    """Run all Phase 6 validation checks and write a report."""
    checks: List[Dict] = []

    manifest = json.loads(Path(MANIFEST_JSON).read_text()) if Path(MANIFEST_JSON).exists() else {}

    checks.extend(validate_inputs(manifest))
    checks.extend(validate_rasters())
    checks.extend(validate_severity_probabilities())
    checks.extend(validate_hotspots())
    checks.extend(validate_area_statistics())
    checks.extend(validate_green_built())
    checks.extend(validate_temporal())
    checks.extend(validate_nodata())
    checks.extend(validate_error_diagnostics())
    checks.extend(validate_phase5_integrity())
    if repro:
        checks.extend(run_reproducibility_checks())
    else:
        checks.append(
            _warn(
                "reproducibility checks not run (use --repro)",
                "reproducibility",
                "analytics idempotence, raster histograms and hotspot_stats "
                "recompute are only executed with --repro",
            )
        )

    summary = {
        "total": len(checks),
        "pass": sum(1 for c in checks if c["status"] == "PASS"),
        "fail": sum(1 for c in checks if c["status"] == "FAIL"),
        "warn": sum(1 for c in checks if c["status"] == "WARN"),
    }

    report = {
        "phase": 6,
        "generated_utc": pd.Timestamp.now("UTC").isoformat(),
        "repro_run": repro,
        "summary": summary,
        "checks": checks,
        "grid_spec": manifest.get("grid_spec", {}),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for Phase 6 validation."""
    parser = argparse.ArgumentParser(description="Validate GreenGrid-AI Phase 6 outputs.")
    parser.add_argument(
        "--output",
        default=str(PHASE6_VALIDATION_REPORT_JSON),
        help="Path to write the validation report JSON.",
    )
    parser.add_argument(
        "--repro",
        action="store_true",
        help="Also run reproducibility checks (re-runs severity.analytics; ~minutes).",
    )
    args = parser.parse_args(argv)

    report = run_validation(output_path=args.output, repro=args.repro)
    summary = report["summary"]
    print(f"Phase 6 Validation: {summary['pass']} PASS / {summary['fail']} FAIL / {summary['warn']} WARN")

    for check in report["checks"]:
        if check["status"] != "PASS":
            print(f"  {check['status']}: {check['name']} — {check['message']}")

    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
