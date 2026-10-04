"""Phase 6 Stage 3: urban heat analytics tables.

Turns the Stage 1 full-grid severity products and the Stage 2 defA hotspot
masks into the analysis tables that feed the Phase 6 report: per-spatial-block
area statistics, high-risk summaries, green-vs-built comparisons, landuse heat
statistics, vegetation/urbanization-temperature relationships, temporal
comparison, hotspot change/persistence, model-error diagnostics, uncertainty
zones and a compact per-year severity summary.

Framing and method notes (repeated in the outputs):

* ALL severity analytics are restricted to the shared valid mask of the
  severity rasters.  NoData cells never enter any statistic.  The LST rasters
  cover more cells than the valid mask; LST statistics use LST-finite cells
  within the valid mask only.
* Areas are exact sums of per-pixel areas in EPSG:32643 (UTM 43N, Delhi),
  computed once by transforming pixel corners; degrees are NOT equal-area.
  (The Stage 2 full-grid mean of 787.14 m2/px is recovered as a sanity check.)
* Severity classes are per-year LST quartiles: the 2022 and 2026 snapshots are
  two dates, NOT a long-term trend, and a 2026 Severe pixel can be cooler in
  deg-C than a 2022 Moderate one.  Cross-year statements are class-relative.
* The green/built grouping uses the conventional operational thresholds from
  ``config.classify_green_built``; group differences are ASSOCIATIONS, not
  causal effects.
* Correlation tables use a deterministic seed-42 subsample of at most 200,000
  valid pixels; spatially autocorrelated pixels are NOT independent
  observations, so n overstates the effective sample size.  NDVI and
  vegetation_cover are near-duplicates (Phase 5 found r ~ 0.999); their
  correlation must not be read as independent evidence.
* Error diagnostics use the Phase 5 TRUE out-of-fold (OOF) predictions of the
  frozen baseline Random Forest at the SAMPLED cells only (150,000 per year,
  model == "Random Forest" rows of oof_predictions.csv); they are not
  full-grid statistics and NOT in-sample fits.  predictions.csv holds the
  production model's in-sample predictions and is deliberately not used here.
  The Phase 6 production model (RF-C) is a different model; its honest accuracy
  estimates are the frozen Phase 5 numbers (RF-C spatial-CV 46.99% accuracy /
  0.4310 macro-F1; locked holdout ~39.9% / 0.395), not anything recomputed here.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy import stats as scipy_stats

from .config import (
    INPUT_DATASET_CSV,
    LANDUSE_CLASS_NAMES,
    LANDUSE_RASTER_PATH,
    MANIFEST_JSON,
    NDBI_RASTERS,
    NDVI_RASTERS,
    PHASE5_OOF_PREDICTIONS_CSV,
    PHASE6_RASTERS_DIR,
    PHASE6_TABLES_DIR,
    PIPELINE_RECORD_JSON,
    PROJECT_ROOT,
    RANDOM_SEED,
    SEVERITY_CLASS_NAMES,
    SEVERITY_NODATA,
    UTM_CRS,
    VEGETATION_COVER_RASTERS,
    WGS84_CRS,
    classify_green_built,
    confidence_raster_path,
    hotspot_mask_path,
    lst_raster_path,
    probability_raster_path,
    severity_raster_path,
    severity_score_raster_path,
)
from .fullgrid import compute_fullgrid_block_raster, load_valid_cells

# ---------------------------------------------------------------------------
# Stage 3 operational constants
# ---------------------------------------------------------------------------
# Per-year LST quartile thresholds inherited from Phase 5 metadata; used to
# recompute each sampled cell's distance to the nearest severity boundary.
BOUNDARY_THRESHOLDS_C = {
    2022: (34.2137, 38.6264, 43.4697),
    2026: (32.9354, 34.4735, 36.1996),
}
BOUNDARY_BANDS_C = (0.0, 0.25, 0.5, 1.0, 2.0, np.inf)
BOUNDARY_BAND_LABELS = ("0-0.25", "0.25-0.5", "0.5-1.0", "1.0-2.0", ">=2.0")

# Correlation subsample: deterministic seed-42 draw of at most 200k valid
# pixels per year (speed only; full-grid correlations are also feasible).
CORRELATION_SUBSAMPLE_N = 200_000

# Uncertainty-zone operational thresholds (documented operational choices,
# not calibrated cut-offs): top-2 class-probability margin < 0.10 marks the
# uncertain transition zone, as does confidence < 0.50.
UNCERTAINTY_MARGIN_THRESHOLD = 0.10
UNCERTAINTY_CONFIDENCE_THRESHOLD = 0.50

TEMPORAL_CAVEAT = (
    "Two snapshot dates (2022 and 2026 composites), NOT a long-term trend. "
    "Severity classes are per-year LST quartiles, so cross-year severity is "
    "class-relative: a 2026 Severe pixel can be cooler in deg-C than a 2022 "
    "Moderate one."
)

UNIT_LABEL = "spatial_block"


def area_statistics_path(year: int) -> Path:
    """Return the per-block area-statistics CSV path for a year."""
    return PHASE6_TABLES_DIR / f"area_statistics_{year}.csv"


def landuse_heat_statistics_path(year: int) -> Path:
    """Return the landuse heat-statistics CSV path for a year."""
    return PHASE6_TABLES_DIR / f"landuse_heat_statistics_{year}.csv"


def oof_error_map_path(year: int) -> Path:
    """Return the sampled-cell error-map raster path for a year."""
    return PHASE6_RASTERS_DIR / f"oof_error_map_{year}.tif"


def uncertainty_zone_path(year: int) -> Path:
    """Return the uncertainty-zone raster path for a year."""
    return PHASE6_RASTERS_DIR / f"uncertainty_zone_{year}.tif"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    from datetime import datetime, timezone

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


def _class_count_percent(severity: np.ndarray, denom: int) -> Dict[str, float]:
    """Class counts (0-3) plus percents of ``denom`` for one severity vector."""
    counts = np.bincount(severity.astype(np.int64), minlength=4)
    out: Dict[str, float] = {}
    for k in range(4):
        out[f"{SEVERITY_CLASS_NAMES[k].lower()}_count"] = int(counts[k])
    for k in range(4):
        out[f"{SEVERITY_CLASS_NAMES[k].lower()}_percent"] = (
            round(100.0 * counts[k] / denom, 4) if denom > 0 else np.nan
        )
    return out


def compute_pixel_area_m2(transform, height: int, width: int) -> np.ndarray:
    """Per-pixel area (m2) in EPSG:32643 computed from transformed corners.

    Every pixel corner of the EPSG:4326 grid is transformed to UTM 43N once
    and the shoelace formula is applied per pixel.  Summing this grid over any
    cell set gives its equal-area area; degree widths are never used.
    """
    from pyproj import Transformer

    jj, ii = np.meshgrid(
        np.arange(width + 1, dtype=np.float64),
        np.arange(height + 1, dtype=np.float64),
    )
    lon, lat = transform * (jj, ii)  # pixel corner coordinates (upper-left)
    transformer = Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True)
    x, y = transformer.transform(lon, lat)

    x1, x2, x3, x4 = x[:-1, :-1], x[:-1, 1:], x[1:, 1:], x[1:, :-1]
    y1, y2, y3, y4 = y[:-1, :-1], y[:-1, 1:], y[1:, 1:], y[1:, :-1]
    area = 0.5 * np.abs(
        x1 * y2 - x2 * y1
        + x2 * y3 - x3 * y2
        + x3 * y4 - x4 * y3
        + x4 * y1 - x1 * y4
    )
    return area


def load_year_cells(
    year: int,
    sampled_df: pd.DataFrame,
    valid_mask: np.ndarray,
    pixel_area_m2: np.ndarray,
) -> pd.DataFrame:
    """Load one year's full-grid products as a per-valid-cell DataFrame.

    All arrays are sampled at valid-mask cells only, so NoData can never enter
    any downstream statistic.  LST keeps a finite-value guard (LST rasters
    cover cells outside the valid mask; within it, values are finite — the
    guard documents the rule rather than fixing data).
    """
    height, width = valid_mask.shape
    rows, cols = np.where(valid_mask)

    with rasterio.open(severity_raster_path(year)) as src:
        severity_full = src.read(1)
    if not ((severity_full != SEVERITY_NODATA) == valid_mask).all():
        raise AssertionError(
            f"[{year}] severity nodata does not align with the valid mask"
        )

    block_raster = compute_fullgrid_block_raster(sampled_df, height, width)

    def _at(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read(1)[rows, cols]

    lst = _at(lst_raster_path(year)).astype(np.float64)
    lst_finite = np.isfinite(lst)
    lst = np.where(lst_finite, lst, np.nan)

    with rasterio.open(hotspot_mask_path("A", year)) as src:
        hotspot_id = src.read(1)[rows, cols]

    with rasterio.open(LANDUSE_RASTER_PATH) as src:
        landuse = src.read(1)[rows, cols]

    data: Dict[str, np.ndarray] = {
        "row": rows,
        "col": cols,
        "block_id": block_raster[rows, cols],
        "severity": severity_full[rows, cols].astype(np.int64),
        "lst_C": lst,
        "lst_finite": lst_finite,
        "ndvi": _at(NDVI_RASTERS[year]).astype(np.float64),
        "ndbi": _at(NDBI_RASTERS[year]).astype(np.float64),
        "vegetation_cover": _at(VEGETATION_COVER_RASTERS[year]).astype(np.float64),
        "confidence": _at(confidence_raster_path(year)).astype(np.float64),
        "severity_score": _at(severity_score_raster_path(year)).astype(np.float64),
        "hotspot": hotspot_id > 0,
        "hotspot_id": hotspot_id,
        "landuse": landuse.astype(np.int64),
        "pixel_area_m2": pixel_area_m2[rows, cols],
    }
    cells = pd.DataFrame(data)
    cells["green_built"] = classify_green_built(cells["ndvi"].values, cells["ndbi"].values)
    return cells


# ---------------------------------------------------------------------------
# 1. Area-wise statistics by spatial block
# ---------------------------------------------------------------------------
def build_area_statistics(cells: pd.DataFrame, year: int) -> pd.DataFrame:
    """One row per occupied 5x5 spatial block for one year.

    These blocks are the Phase 5 sampling-design grid cells (5x5 over the
    Phase 4 sampled row/col extent), NOT administrative sectors.  Only the 20
    occupied blocks (non-zero valid pixels) are reported.
    """
    rows_out: List[Dict] = []
    for block_id, grp in cells.groupby("block_id"):
        n = len(grp)
        lst = grp["lst_C"].dropna().values
        row: Dict = {
            "block_id": int(block_id),
            "year": year,
            "unit": UNIT_LABEL,
            "unit_note": (
                "5x5 sampling-design grid block from Phase 5; NOT an "
                "administrative sector"
            ),
            "valid_pixel_count": n,
            "area_ha": round(float(grp["pixel_area_m2"].sum()) / 1e4, 3),
            "mean_lst_C": round(float(lst.mean()), 4),
            "median_lst_C": round(float(np.median(lst)), 4),
            "min_lst_C": round(float(lst.min()), 4),
            "max_lst_C": round(float(lst.max()), 4),
            "std_lst_C": round(float(lst.std(ddof=1)), 4),
            "mean_ndvi": round(float(grp["ndvi"].mean()), 6),
            "mean_ndbi": round(float(grp["ndbi"].mean()), 6),
            "mean_vegetation_cover": round(float(grp["vegetation_cover"].mean()), 6),
            "mean_confidence": round(float(grp["confidence"].mean()), 6),
            "mean_severity_score": round(float(grp["severity_score"].mean()), 6),
            "hotspot_pixel_count": int(grp["hotspot"].sum()),
            "hotspot_area_ha": round(float(grp.loc[grp["hotspot"], "pixel_area_m2"].sum()) / 1e4, 3),
            "hotspot_percentage": round(100.0 * grp["hotspot"].mean(), 4),
        }
        row.update(_class_count_percent(grp["severity"].values, n))
        row["high_severe_percent"] = row["high_percent"] + row["severe_percent"]
        rows_out.append(row)
    df = pd.DataFrame(rows_out).sort_values("block_id").reset_index(drop=True)
    class_pct_cols = [f"{SEVERITY_CLASS_NAMES[k].lower()}_percent" for k in range(4)]
    df["class_percent_sum"] = round(df[class_pct_cols].sum(axis=1), 4)
    return df


# ---------------------------------------------------------------------------
# 2. High-risk summary
# ---------------------------------------------------------------------------
def build_high_risk_summary(
    cells_by_year: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Per-year high-risk headline: High+Severe share and class area breakdown.

    Areas are exact EPSG:32643 per-pixel area sums (never degrees squared).
    Both the raw severity-class High+Severe area and the defA mask area are
    reported; the defA mask additionally drops clusters below the 10-pixel
    noise rule, so it is slightly smaller.
    """
    rows_out: List[Dict] = []
    for year, cells in cells_by_year.items():
        n = len(cells)
        hs = cells["severity"] >= 2
        row: Dict = {
            "year": year,
            "valid_pixels": n,
            "valid_area_ha": round(float(cells["pixel_area_m2"].sum()) / 1e4, 3),
            "high_severe_pixels": int(hs.sum()),
            "high_severe_pixel_percentage": round(100.0 * hs.mean(), 4),
            "high_severe_area_ha": round(float(cells.loc[hs, "pixel_area_m2"].sum()) / 1e4, 3),
            "defA_hotspot_pixels": int(cells["hotspot"].sum()),
            "defA_hotspot_area_ha": round(
                float(cells.loc[cells["hotspot"], "pixel_area_m2"].sum()) / 1e4, 3
            ),
            "defA_mask_note": (
                "defA mask = severity in {2,3} minus clusters < 10 px "
                "(Stage 2 noise rule); subset of high_severe above"
            ),
        }
        row.update(_class_count_percent(cells["severity"].values, n))
        for k in range(4):
            sel = cells["severity"] == k
            row[f"{SEVERITY_CLASS_NAMES[k].lower()}_area_ha"] = round(
                float(cells.loc[sel, "pixel_area_m2"].sum()) / 1e4, 3
            )
        rows_out.append(row)
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# 3. Green vs built-up comparison
# ---------------------------------------------------------------------------
def build_green_built_comparison(
    cells_by_year: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Per year x green/built/mixed group heat statistics.

    Grouping uses the conventional operational thresholds in
    ``config.classify_green_built`` (NDVI >= 0.3 green; NDBI >= 0.1 AND
    NDVI < 0.3 built).  These are ASSOCIATIONS between surface composition
    and LST/severity, not causal effects.
    """
    rows_out: List[Dict] = []
    for year, cells in cells_by_year.items():
        for group in ("green_dominant", "built_dominant", "other_mixed"):
            grp = cells[cells["green_built"] == group]
            n = len(grp)
            if n == 0:
                continue
            lst = grp["lst_C"].dropna().values
            row: Dict = {
                "year": year,
                "group": group,
                "pixel_count": n,
                "area_ha": round(float(grp["pixel_area_m2"].sum()) / 1e4, 3),
                "mean_lst_C": round(float(lst.mean()), 4),
                "median_lst_C": round(float(np.median(lst)), 4),
                "mean_ndvi": round(float(grp["ndvi"].mean()), 6),
                "mean_ndbi": round(float(grp["ndbi"].mean()), 6),
                "mean_vegetation_cover": round(float(grp["vegetation_cover"].mean()), 6),
                "mean_confidence": round(float(grp["confidence"].mean()), 6),
                "association_note": (
                    "association only; thresholds operational, not "
                    "site-calibrated; no causal claim"
                ),
            }
            row.update(_class_count_percent(grp["severity"].values, n))
            row["high_severe_percent"] = row["high_percent"] + row["severe_percent"]
            rows_out.append(row)
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# 4. Landuse heat statistics
# ---------------------------------------------------------------------------
def build_landuse_heat_statistics(cells: pd.DataFrame, year: int) -> pd.DataFrame:
    """Per landuse class heat statistics for one year.

    All landuse codes present at valid cells are reported, including code 0
    (unclassified_background); landuse nodata (255) never occurs at valid
    cells and is excluded by construction.
    """
    rows_out: List[Dict] = []
    n_valid = len(cells)
    for code in sorted(cells["landuse"].unique()):
        grp = cells[cells["landuse"] == code]
        n = len(grp)
        lst = grp["lst_C"].dropna().values
        row: Dict = {
            "year": year,
            "landuse_code": int(code),
            "landuse_name": LANDUSE_CLASS_NAMES.get(int(code), "unknown"),
            "pixel_count": n,
            "percent_of_valid": round(100.0 * n / n_valid, 4),
            "mean_lst_C": round(float(lst.mean()), 4),
            "mean_ndvi": round(float(grp["ndvi"].mean()), 6),
            "mean_ndbi": round(float(grp["ndbi"].mean()), 6),
            "mean_vegetation_cover": round(float(grp["vegetation_cover"].mean()), 6),
        }
        row.update(_class_count_percent(grp["severity"].values, n))
        row["high_severe_percent"] = row["high_percent"] + row["severe_percent"]
        rows_out.append(row)
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# 5. Vegetation-temperature relationship
# ---------------------------------------------------------------------------
def _decile_assign(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Assign values to 10 quantile bins; return (decile 0-9, 11 bin edges)."""
    edges = np.quantile(values, np.linspace(0.0, 1.0, 11))
    decile = np.clip(np.digitize(values, edges[1:-1]), 0, 9).astype(np.int64)
    return decile, edges


def _corr_rows(year: int, x: np.ndarray, y: np.ndarray, name: str) -> Dict:
    """Pearson + Spearman row for one variable pair (subsampled cells)."""
    pearson_r, _ = scipy_stats.pearsonr(x, y)
    spearman_r, _ = scipy_stats.spearmanr(x, y)
    note = (
        "seed-42 subsample of valid pixels; spatially autocorrelated pixels "
        "are NOT independent observations, so n overstates effective n; "
        "frozen Phase 5 Moran's I on RF error fields: 0.53-0.54 "
        "(classification error) / 0.61-0.70 (class-distance error), p=0.005"
    )
    if name == "ndvi_vs_vegetation_cover":
        note += (
            "; REDUNDANCY: NDVI and vegetation_cover are near-duplicates "
            "(Phase 5 found r ~ 0.999) — not independent evidence"
        )
    return {
        "year": year,
        "analysis": "correlation",
        "variable": name,
        "decile": np.nan,
        "decile_min": np.nan,
        "decile_max": np.nan,
        "n": int(len(x)),
        "pearson_r": round(float(pearson_r), 6),
        "spearman_r": round(float(spearman_r), 6),
        "mean_lst_C": np.nan,
        "high_severe_percent": np.nan,
        "note": note,
    }


def build_vegetation_temperature_relationship(
    cells_by_year: Dict[int, pd.DataFrame],
    subsample_n: int = CORRELATION_SUBSAMPLE_N,
) -> pd.DataFrame:
    """Vegetation/urbanization vs LST correlations and decile profiles.

    Correlations use a deterministic seed-42 subsample of at most 200,000
    valid pixels per year (documented operational choice for speed).  Decile
    profiles use ALL valid pixels.  NDVI vs vegetation_cover is reported with
    an explicit redundancy note (near-duplicate variables).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows_out: List[Dict] = []
    for year, cells in cells_by_year.items():
        n_take = min(subsample_n, len(cells))
        idx = rng.choice(len(cells), size=n_take, replace=False)
        sub = cells.iloc[idx]
        x_lst = sub["lst_C"].values
        rows_out.append(_corr_rows(year, sub["ndvi"].values, x_lst, "ndvi_vs_lst"))
        rows_out.append(
            _corr_rows(year, sub["vegetation_cover"].values, x_lst, "vegetation_cover_vs_lst")
        )
        rows_out.append(_corr_rows(year, sub["ndbi"].values, x_lst, "ndbi_vs_lst"))
        rows_out.append(
            _corr_rows(year, sub["ndvi"].values, sub["vegetation_cover"].values,
                       "ndvi_vs_vegetation_cover")
        )

        # Decile profiles on all valid pixels.
        for var in ("ndvi", "ndbi"):
            decile, edges = _decile_assign(cells[var].values)
            for d in range(10):
                sel = decile == d
                rows_out.append({
                    "year": year,
                    "analysis": f"{var}_decile",
                    "variable": var,
                    "decile": d + 1,
                    "decile_min": round(float(edges[d]), 6),
                    "decile_max": round(float(edges[d + 1]), 6),
                    "n": int(sel.sum()),
                    "pearson_r": np.nan,
                    "spearman_r": np.nan,
                    "mean_lst_C": round(float(cells.loc[sel, "lst_C"].mean()), 4),
                    "high_severe_percent": round(
                        100.0 * (cells.loc[sel, "severity"] >= 2).mean(), 4
                    ),
                    "note": "all valid pixels; deciles are per-year quantiles",
                })
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# 6. Severity by urbanization (NDBI deciles)
# ---------------------------------------------------------------------------
def build_urbanization_heat_relationship(
    cells_by_year: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Mean LST, mean severity score and High+Severe share by NDBI decile."""
    rows_out: List[Dict] = []
    for year, cells in cells_by_year.items():
        decile, edges = _decile_assign(cells["ndbi"].values)
        for d in range(10):
            sel = decile == d
            rows_out.append({
                "year": year,
                "ndbi_decile": d + 1,
                "ndbi_decile_min": round(float(edges[d]), 6),
                "ndbi_decile_max": round(float(edges[d + 1]), 6),
                "pixel_count": int(sel.sum()),
                "mean_lst_C": round(float(cells.loc[sel, "lst_C"].mean()), 4),
                "mean_severity_score": round(float(cells.loc[sel, "severity_score"].mean()), 6),
                "high_severe_percent": round(
                    100.0 * (cells.loc[sel, "severity"] >= 2).mean(), 4
                ),
                "note": "all valid pixels; deciles are per-year NDBI quantiles",
            })
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# 7. 2022 vs 2026 temporal comparison
# ---------------------------------------------------------------------------
def build_temporal_comparison(
    cells_by_year: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Side-by-side per-year headline comparison.

    CRITICAL framing (also written to the ``caveat`` column and a companion
    notes JSON): two snapshot dates, NOT a long-term trend; per-year quartile
    classes make cross-year severity class-relative.
    """
    rows_out: List[Dict] = []
    for year, cells in cells_by_year.items():
        lst = cells["lst_C"].dropna().values
        counts = np.bincount(cells["severity"].values.astype(np.int64), minlength=4)
        n = len(cells)
        hotspot_areas = (
            cells.loc[cells["hotspot"]]
            .groupby("hotspot_id")["pixel_area_m2"]
            .sum()
        )
        row: Dict = {
            "year": year,
            "valid_pixels": n,
            "valid_area_ha": round(float(cells["pixel_area_m2"].sum()) / 1e4, 3),
            "mean_lst_C": round(float(lst.mean()), 4),
            "median_lst_C": round(float(np.median(lst)), 4),
            "high_severe_area_ha": round(
                float(cells.loc[cells["severity"] >= 2, "pixel_area_m2"].sum()) / 1e4, 3
            ),
            "high_severe_percent": round(100.0 * (cells["severity"] >= 2).mean(), 4),
            "hotspot_count_defA": int(cells.loc[cells["hotspot"], "hotspot_id"].nunique()),
            "hotspot_area_ha_defA": round(
                float(cells.loc[cells["hotspot"], "pixel_area_m2"].sum()) / 1e4, 3
            ),
            "largest_hotspot_area_ha_defA": (
                round(float(hotspot_areas.max()) / 1e4, 3) if len(hotspot_areas) else np.nan
            ),
            "mean_ndvi": round(float(cells["ndvi"].mean()), 6),
            "mean_ndbi": round(float(cells["ndbi"].mean()), 6),
            "mean_vegetation_cover": round(float(cells["vegetation_cover"].mean()), 6),
            "mean_confidence": round(float(cells["confidence"].mean()), 6),
            "caveat": TEMPORAL_CAVEAT,
        }
        for k in range(4):
            row[f"{SEVERITY_CLASS_NAMES[k].lower()}_percent"] = round(
                100.0 * counts[k] / n, 4
            )
        rows_out.append(row)
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# 8. Hotspot change + severity transition matrix (paired cells)
# ---------------------------------------------------------------------------
def build_hotspot_change_statistics(
    cells_by_year: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """defA hotspot change categories on cells valid in BOTH years.

    The valid mask is identical across years, so paired cells = valid cells.
    NoData never means "not a hotspot"; categories describe mask membership
    only.  Descriptive, NOT a climate trend.
    """
    h22 = cells_by_year[2022]["hotspot"].values
    h26 = cells_by_year[2026]["hotspot"].values
    area = cells_by_year[2022]["pixel_area_m2"].values
    n_paired = len(area)

    categories = {
        "persistent_hotspot": h22 & h26,
        "new_hotspot_2026": ~h22 & h26,
        "disappeared_hotspot_2022": h22 & ~h26,
        "stable_non_hotspot": ~h22 & ~h26,
    }
    rows_out: List[Dict] = []
    for name, sel in categories.items():
        rows_out.append({
            "change_category": name,
            "pixel_count": int(sel.sum()),
            "area_ha": round(float(area[sel].sum()) / 1e4, 3),
            "percent_of_paired": round(100.0 * sel.mean(), 4),
            "paired_cell_count": n_paired,
            "note": (
                "paired cells valid in both years; NoData != 'not a "
                "hotspot'; descriptive only, not a climate trend"
            ),
            "intersection_px": np.nan,
            "union_px": np.nan,
            "mask_iou": np.nan,
        })

    inter = int((h22 & h26).sum())
    union = int((h22 | h26).sum())
    rows_out.append({
        "change_category": "mask_iou_2022_vs_2026",
        "pixel_count": np.nan,
        "area_ha": np.nan,
        "percent_of_paired": np.nan,
        "paired_cell_count": n_paired,
        "note": "IoU of the two defA hotspot masks over paired valid cells",
        "intersection_px": inter,
        "union_px": union,
        "mask_iou": round(inter / union, 6),
    })
    return pd.DataFrame(rows_out)


def build_transition_matrix(cells_by_year: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """2022 -> 2026 severity transition counts and row-normalized probabilities."""
    s22 = cells_by_year[2022]["severity"].values.astype(np.int64)
    s26 = cells_by_year[2026]["severity"].values.astype(np.int64)
    joint = np.bincount(s22 * 4 + s26, minlength=16).reshape(4, 4)
    rows_out: List[Dict] = []
    for i in range(4):
        row_total = int(joint[i].sum())
        for j in range(4):
            rows_out.append({
                "severity_2022": SEVERITY_CLASS_NAMES[i],
                "severity_2026": SEVERITY_CLASS_NAMES[j],
                "count": int(joint[i, j]),
                "row_probability_2026_given_2022": (
                    round(joint[i, j] / row_total, 6) if row_total > 0 else np.nan
                ),
                "row_total_2022_class": row_total,
                "paired_cell_count": int(joint.sum()),
                "note": (
                    "paired cells valid in both years; classes are per-year "
                    "quartiles (class-relative, not deg-C comparable)"
                ),
            })
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# 9. Hotspot persistence
# ---------------------------------------------------------------------------
def build_hotspot_persistence(cells_by_year: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """P(High/Severe 2026 | High/Severe 2022) and the reverse, on paired cells."""
    hs22 = cells_by_year[2022]["severity"].values >= 2
    hs26 = cells_by_year[2026]["severity"].values >= 2
    n = len(hs22)

    def _row(condition_name: str, numerator: np.ndarray, denominator: np.ndarray) -> Dict:
        return {
            "quantity": condition_name,
            "numerator": int(numerator.sum()),
            "denominator": int(denominator.sum()),
            "probability": (
                round(float(numerator.sum()) / float(denominator.sum()), 6)
                if denominator.sum() > 0
                else np.nan
            ),
            "paired_cell_count": n,
            "note": (
                "severity class in {High=2, Severe=3}; paired valid cells; "
                "per-year quartile classes — class-relative persistence, "
                "not a physical trend"
            ),
        }

    return pd.DataFrame([
        _row("P(HighSevere_2026 | HighSevere_2022)", hs22 & hs26, hs22),
        _row("P(HighSevere_2022 | HighSevere_2026)", hs22 & hs26, hs26),
        _row("P(HighSevere_2022)", hs22, np.ones(n, dtype=bool)),
        _row("P(HighSevere_2026)", hs26, np.ones(n, dtype=bool)),
    ])


# ---------------------------------------------------------------------------
# 10. Model error / confidence diagnostics (SAMPLED cells, TRUE OOF only)
# ---------------------------------------------------------------------------
# The frozen Phase 5 spatial-CV wrote every model's out-of-fold predictions
# for every sampled cell; "Random Forest" is the frozen baseline RF whose OOF
# accuracy (pooled 0.4619; 2022 0.4312, 2026 0.4925) matches the Phase 5
# spatial-CV report.  predictions.csv is the production model's IN-SAMPLE
# output (0.9957) and is deliberately NOT used here.
OOF_MODEL_NAME = "Random Forest"
OOF_EXPECTED_ROWS = 300_000
OOF_ACCURACY_GUARD = (0.45, 0.48)  # pooled OOF accuracy must fall in this band


def load_sampled_predictions() -> pd.DataFrame:
    """TRUE OOF predictions of the frozen Phase 5 baseline Random Forest.

    Selects the ``model == "Random Forest"`` rows (exactly one OOF prediction
    per sampled cell per year) from oof_predictions.csv and joins the Phase 4
    observed LST on (row, col, year); verified to match 300,000/300,000 rows.
    Guards assert the row count and the pooled OOF accuracy band so that an
    accidental switch to in-sample predictions (accuracy ~0.996) fails loudly.
    """
    oof = pd.read_csv(PHASE5_OOF_PREDICTIONS_CSV)
    preds = oof[oof["model"] == OOF_MODEL_NAME].copy()
    if len(preds) != OOF_EXPECTED_ROWS:
        raise AssertionError(
            f"Expected {OOF_EXPECTED_ROWS} '{OOF_MODEL_NAME}' OOF rows, got {len(preds)}"
        )
    pooled_acc = float((preds["actual_class"] == preds["predicted_class"]).mean())
    lo, hi = OOF_ACCURACY_GUARD
    if not (lo <= pooled_acc <= hi):
        raise AssertionError(
            f"Pooled OOF accuracy {pooled_acc:.4f} outside guard band "
            f"[{lo}, {hi}] — this does not look like true OOF predictions"
        )
    p4 = pd.read_csv(INPUT_DATASET_CSV, usecols=["row", "col", "year", "lst_C"])
    merged = preds.merge(p4, on=["row", "col", "year"], how="left")
    if int(merged["lst_C"].isna().sum()) > 0:
        raise AssertionError("Sampled predictions could not be joined to Phase 4 lst_C")
    return merged


def boundary_distance_band(lst_c: np.ndarray, year: int) -> np.ndarray:
    """Distance (deg-C) from each cell's LST to the nearest per-year quartile."""
    thresholds = np.array(BOUNDARY_THRESHOLDS_C[year])
    dist = np.min(np.abs(lst_c[:, None] - thresholds[None, :]), axis=1)
    return np.clip(np.digitize(dist, BOUNDARY_BANDS_C[1:-1]), 0, len(BOUNDARY_BANDS_C) - 2)


def write_error_map_rasters(
    sampled: pd.DataFrame,
    mask_profile: Dict,
) -> Dict[int, str]:
    """Write int16 error rasters (1=correct, 0=incorrect, nodata -1).

    Values exist ONLY at the Phase 5 sampled cells; every other cell is
    nodata.  These are TRUE out-of-fold diagnostics of the frozen Phase 5
    baseline Random Forest (the "Random Forest" rows of oof_predictions.csv),
    NOT in-sample fits and NOT the Phase 6 RF-C production model, whose honest
    accuracy estimates are the frozen Phase 5 numbers.
    """
    written: Dict[int, str] = {}
    height, width = mask_profile["height"], mask_profile["width"]
    for year in (2022, 2026):
        sub = sampled[sampled["year"] == year]
        correct = (sub["actual_class"].values == sub["predicted_class"].values).astype(np.int16)
        raster = np.full((height, width), -1, dtype=np.int16)
        raster[sub["row"].values, sub["col"].values] = correct
        out_path = oof_error_map_path(year)
        profile = mask_profile.copy()
        profile.update({"dtype": "int16", "count": 1, "nodata": -1, "compress": "lzw"})
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(raster, 1)
        written[year] = str(out_path)
    return written


def build_error_diagnostics_summary(sampled: pd.DataFrame) -> pd.DataFrame:
    """Per-year TRUE-OOF error diagnostics on Phase 5 sampled cells (150k/yr)."""
    rows_out: List[Dict] = []
    for year in (2022, 2026):
        sub = sampled[sampled["year"] == year].copy()
        sub["correct"] = sub["actual_class"].values == sub["predicted_class"].values
        sub["band"] = boundary_distance_band(sub["lst_C"].values.astype(np.float64), year)
        n = len(sub)
        base = {
            "year": year,
            "sample_note": (
                "TRUE out-of-fold predictions of the frozen Phase 5 baseline "
                "Random Forest at sampled cells (150,000/yr), not full grid; "
                "NOT the Phase 6 RF-C production model (RF-C frozen spatial-CV "
                "46.99% acc / 0.4310 macro-F1; locked holdout ~39.9% / 0.395)"
            ),
        }
        rows_out.append({
            **base,
            "analysis": "overall_accuracy",
            "group": "all",
            "n": n,
            "accuracy": round(float(sub["correct"].mean()), 6),
            "error_rate": round(1.0 - float(sub["correct"].mean()), 6),
            "mean_confidence": round(float(sub["prediction_confidence"].mean()), 6),
        })
        for cls in range(4):
            sel = sub["actual_class"] == cls
            rows_out.append({
                **base,
                "analysis": "per_class_recall",
                "group": SEVERITY_CLASS_NAMES[cls],
                "n": int(sel.sum()),
                "accuracy": round(float(sub.loc[sel, "correct"].mean()), 6),
                "error_rate": round(1.0 - float(sub.loc[sel, "correct"].mean()), 6),
                "mean_confidence": round(float(sub.loc[sel, "prediction_confidence"].mean()), 6),
            })
        for band_idx, label in enumerate(BOUNDARY_BAND_LABELS):
            sel = sub["band"] == band_idx
            rows_out.append({
                **base,
                "analysis": "boundary_distance_band_error_rate",
                "group": f"dist_C_{label}",
                "n": int(sel.sum()),
                "accuracy": np.nan,
                "error_rate": round(1.0 - float(sub.loc[sel, "correct"].mean()), 6)
                if sel.sum() > 0 else np.nan,
                "mean_confidence": np.nan,
            })
        for flag, label in ((True, "correct"), (False, "incorrect")):
            sel = sub["correct"] == flag
            rows_out.append({
                **base,
                "analysis": "confidence_by_outcome",
                "group": label,
                "n": int(sel.sum()),
                "accuracy": np.nan,
                "error_rate": np.nan,
                "mean_confidence": round(float(sub.loc[sel, "prediction_confidence"].mean()), 6),
            })
    return pd.DataFrame(rows_out)


def build_error_by_block(sampled: pd.DataFrame) -> pd.DataFrame:
    """TRUE-OOF sampled-cell error rate per occupied spatial block and year."""
    rows_out: List[Dict] = []
    for year in (2022, 2026):
        sub = sampled[sampled["year"] == year]
        for block_id, grp in sub.groupby("spatial_block_id"):
            rows_out.append({
                "year": year,
                "block_id": int(block_id),
                "unit": UNIT_LABEL,
                "n_sampled": len(grp),
                "n_errors": int((grp["actual_class"] != grp["predicted_class"]).sum()),
                "error_rate": round(
                    float((grp["actual_class"] != grp["predicted_class"]).mean()), 6
                ),
                "sample_note": (
                    "TRUE OOF, frozen Phase 5 baseline Random Forest, sampled "
                    "cells only, not full grid"
                ),
            })
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# 11. Uncertainty zones
# ---------------------------------------------------------------------------
def build_uncertainty_zones(
    cells_by_year: Dict[int, pd.DataFrame],
    mask_profile: Dict,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Flag uncertain cells and write the uint8 uncertainty-zone rasters.

    A valid cell is UNCERTAIN (value 1) when its top-2 class-probability
    margin is below 0.10 (the operational uncertain-transition-zone
    threshold, documented, not calibrated) OR its confidence is below 0.50.
    All other valid cells are 0; everything outside the valid mask is 255.
    """
    height, width = mask_profile["height"], mask_profile["width"]
    rows_out: List[Dict] = []
    written: Dict[int, str] = {}
    for year in (2022, 2026):
        cells = cells_by_year[year]
        rows, cols = cells["row"].values, cells["col"].values
        probs = np.stack(
            [
                _read_band_at(probability_raster_path(lbl, year), rows, cols)
                for lbl in ("low", "moderate", "high", "severe")
            ],
            axis=1,
        )
        top2 = np.partition(probs, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        uncertain = (margin < UNCERTAINTY_MARGIN_THRESHOLD) | (
            cells["confidence"].values < UNCERTAINTY_CONFIDENCE_THRESHOLD
        )

        raster = np.full((height, width), 255, dtype=np.uint8)
        raster[rows, cols] = uncertain.astype(np.uint8)
        out_path = uncertainty_zone_path(year)
        profile = mask_profile.copy()
        profile.update({"dtype": "uint8", "count": 1, "nodata": 255, "compress": "lzw"})
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(raster, 1)
        written[year] = str(out_path)

        # Overlap with severity boundaries: a cell sits on a class boundary
        # when any of its 4-neighbours has a different severity class.
        boundary = _severity_boundary_mask(cells, height, width)
        boundary_at_cells = boundary[rows, cols]
        on_b_unc = float(boundary_at_cells[uncertain].mean()) if uncertain.any() else np.nan
        on_b_cer = float(boundary_at_cells[~uncertain].mean()) if (~uncertain).any() else np.nan
        rows_out.append({
            "year": year,
            "valid_cells": len(cells),
            "uncertain_cells": int(uncertain.sum()),
            "uncertain_fraction": round(float(uncertain.mean()), 6),
            "margin_threshold": UNCERTAINTY_MARGIN_THRESHOLD,
            "confidence_threshold": UNCERTAINTY_CONFIDENCE_THRESHOLD,
            "threshold_note": (
                "operational thresholds (top-2 margin < 0.10 OR confidence < "
                "0.50), not calibrated cut-offs; probabilities are "
                "uncalibrated confidence proxies (Phase 5 calibration report)"
            ),
            "mean_confidence_uncertain": round(
                float(cells.loc[uncertain, "confidence"].mean()), 6
            ),
            "mean_confidence_certain": round(
                float(cells.loc[~uncertain, "confidence"].mean()), 6
            ),
            "uncertain_on_class_boundary_fraction": round(on_b_unc, 6),
            "certain_on_class_boundary_fraction": round(on_b_cer, 6),
            "boundary_definition": (
                ">=1 of 4-neighbours has a different severity class (valid "
                "neighbours only)"
            ),
        })
    return pd.DataFrame(rows_out), written


def _read_band_at(path: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Read one raster band at explicit cell indices."""
    with rasterio.open(path) as src:
        return src.read(1)[rows, cols].astype(np.float64)


def _severity_boundary_mask(
    cells: pd.DataFrame, height: int, width: int
) -> np.ndarray:
    """Boolean mask of valid cells with a different-class 4-neighbour."""
    sev = np.full((height, width), -1, dtype=np.int64)
    sev[cells["row"].values, cells["col"].values] = cells["severity"].values
    boundary = np.zeros((height, width), dtype=bool)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.full((height, width), -1, dtype=np.int64)
        r_src = slice(max(0, dr), height + min(0, dr))
        c_src = slice(max(0, dc), width + min(0, dc))
        r_dst = slice(max(0, -dr), height + min(0, -dr))
        c_dst = slice(max(0, -dc), width + min(0, -dc))
        shifted[r_dst, c_dst] = sev[r_src, c_src]
        boundary |= (shifted >= 0) & (shifted != sev) & (sev >= 0)
    return boundary


# ---------------------------------------------------------------------------
# 12. Severity summary
# ---------------------------------------------------------------------------
def build_severity_summary(cells_by_year: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Compact per-year headline table feeding the report and figures."""
    rows_out: List[Dict] = []
    for year, cells in cells_by_year.items():
        lst = cells["lst_C"].dropna().values
        counts = np.bincount(cells["severity"].values.astype(np.int64), minlength=4)
        n = len(cells)
        row: Dict = {
            "year": year,
            "valid_pixels": n,
            "valid_area_ha": round(float(cells["pixel_area_m2"].sum()) / 1e4, 3),
            "mean_lst_C": round(float(lst.mean()), 4),
            "median_lst_C": round(float(np.median(lst)), 4),
            "high_severe_percent": round(100.0 * (cells["severity"] >= 2).mean(), 4),
            "defA_hotspot_count": int(cells.loc[cells["hotspot"], "hotspot_id"].nunique()),
            "defA_hotspot_area_ha": round(
                float(cells.loc[cells["hotspot"], "pixel_area_m2"].sum()) / 1e4, 3
            ),
            "defA_hotspot_percent_of_valid": round(100.0 * cells["hotspot"].mean(), 4),
            "mean_confidence": round(float(cells["confidence"].mean()), 6),
            "mean_ndvi": round(float(cells["ndvi"].mean()), 6),
            "mean_ndbi": round(float(cells["ndbi"].mean()), 6),
            "mean_vegetation_cover": round(float(cells["vegetation_cover"].mean()), 6),
        }
        for k in range(4):
            row[f"{SEVERITY_CLASS_NAMES[k].lower()}_percent"] = round(
                100.0 * counts[k] / n, 4
            )
        rows_out.append(row)
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def run_validations(
    cells_by_year: Dict[int, pd.DataFrame],
    area_tables: Dict[int, pd.DataFrame],
    transition_df: pd.DataFrame,
) -> Dict:
    """Run the Stage 3 validation battery and return a report dict."""
    report: Dict = {}

    # 1. Class percents sum to ~100 (+-0.1) and counts sum to valid pixels.
    pct_issues = []
    count_issues = []
    for year, tbl in area_tables.items():
        n_valid = len(cells_by_year[year])
        pct_sum = tbl["class_percent_sum"].max()
        count_sum = int(
            tbl[[f"{SEVERITY_CLASS_NAMES[k].lower()}_count" for k in range(4)]]
            .sum()
            .sum()
        )
        if abs(pct_sum - 100.0) > 0.1:
            pct_issues.append({"year": year, "max_class_percent_sum": pct_sum})
        if count_sum != n_valid or int(tbl["valid_pixel_count"].sum()) != n_valid:
            count_issues.append({"year": year, "count_sum": count_sum, "n_valid": n_valid})
    report["class_percent_sums_within_0p1"] = not pct_issues
    report["class_count_sums_equal_valid_pixels"] = not count_issues
    report["class_percent_issues"] = pct_issues
    report["class_count_issues"] = count_issues

    # 2. Cross-check: defA mask vs severity classes at block level.
    mask_rel = {}
    for year, cells in cells_by_year.items():
        mask_is_hs = cells.loc[cells["hotspot"], "severity"] >= 2
        mask_rel[year] = {
            "mask_subset_of_high_severe": bool(mask_is_hs.all()),
            "mask_pixels": int(cells["hotspot"].sum()),
            "high_severe_pixels": int((cells["severity"] >= 2).sum()),
            "mask_minus_high_severe_px": int(
                cells["hotspot"].sum() - (cells["hotspot"] & (cells["severity"] >= 2)).sum()
            ),
            "high_severe_minus_mask_px": int(
                (cells["severity"] >= 2).sum()
                - (cells["hotspot"] & (cells["severity"] >= 2)).sum()
            ),
            "note": (
                "defA = severity {2,3} minus clusters < 10 px; mask_minus = "
                "should be 0 (mask never outside {2,3}); hs_minus = small "
                "clusters removed by the noise rule"
            ),
        }
    report["defA_mask_vs_severity_classes"] = mask_rel

    # 3. Transition matrix row sums = paired class counts.
    paired = int(transition_df["paired_cell_count"].iloc[0])
    g_counts = transition_df.groupby("severity_2022")["count"].sum()
    g_totals = transition_df.groupby("severity_2022")["row_total_2022_class"].first()
    row_ok = bool((g_counts.values == g_totals.values).all())
    total = int(transition_df["count"].sum())
    report["transition_matrix"] = {
        "paired_cell_count": paired,
        "matrix_count_total": total,
        "row_sums_match_paired_counts": bool(row_ok and total == paired),
    }

    # 4. LST sanity: 2022 snapshot warmer than 2026.
    m22 = float(cells_by_year[2022]["lst_C"].mean())
    m26 = float(cells_by_year[2026]["lst_C"].mean())
    report["lst_means"] = {"2022": round(m22, 4), "2026": round(m26, 4)}
    report["lst_2022_warmer_than_2026"] = bool(m22 > m26)

    # 5. NoData never enters any statistic.
    nodata_ok = True
    for year, cells in cells_by_year.items():
        nodata_ok &= bool(
            cells["lst_finite"].all()
            and np.isfinite(cells[["ndvi", "ndbi", "vegetation_cover", "confidence"]]
                            .to_numpy()).all()
        )
    report["no_nodata_in_statistics"] = bool(nodata_ok)
    report["all_passed"] = bool(
        report["class_percent_sums_within_0p1"]
        and report["class_count_sums_equal_valid_pixels"]
        and all(v["mask_subset_of_high_severe"] for v in mask_rel.values())
        and report["transition_matrix"]["row_sums_match_paired_counts"]
        and report["lst_2022_warmer_than_2026"]
        and nodata_ok
    )
    return report


# ---------------------------------------------------------------------------
# Stage 3 driver
# ---------------------------------------------------------------------------
def run_stage3() -> Dict:
    """Run the complete Phase 6 Stage 3 urban-heat-analytics pipeline."""
    record: Dict = {
        "phase": 6,
        "stage": 3,
        "project": "GreenGrid-AI",
        "project_root": str(PROJECT_ROOT),
        "started_utc": _now(),
        "random_seed": RANDOM_SEED,
        "steps": {},
    }

    print("Stage 3: loading grid context ...")
    rows, cols, transform, mask_profile = load_valid_cells()
    valid_mask = np.zeros(
        (mask_profile["height"], mask_profile["width"]), dtype=bool
    )
    valid_mask[rows, cols] = True
    n_valid = int(valid_mask.sum())
    print(f"  Valid cells per year: {n_valid:,}")

    pixel_area = compute_pixel_area_m2(
        transform, mask_profile["height"], mask_profile["width"]
    )
    mean_valid_area = float(pixel_area[valid_mask].mean())
    total_valid_area_ha = float(pixel_area[valid_mask].sum()) / 1e4
    print(
        f"  Per-pixel UTM43N area: mean {mean_valid_area:.2f} m2 "
        f"(cf. Stage 2 full-grid 787.14), valid total {total_valid_area_ha:,.1f} ha"
    )

    print("Stage 3: loading Phase 4 sampled dataset (block bins) ...")
    sampled_df = pd.read_csv(
        INPUT_DATASET_CSV, usecols=["row", "col"]
    )

    cells_by_year: Dict[int, pd.DataFrame] = {}
    for year in (2022, 2026):
        print(f"  [{year}] Loading full-grid stack at valid cells ...")
        cells_by_year[year] = load_year_cells(year, sampled_df, valid_mask, pixel_area)
    record["steps"]["data"] = {
        "status": "success",
        "valid_cells_per_year": n_valid,
        "area_method": (
            "exact per-pixel EPSG:32643 areas from transformed pixel corners "
            "(shoelace); summed over unit cells; degrees never used"
        ),
        "mean_valid_pixel_area_m2": round(mean_valid_area, 2),
        "total_valid_area_ha": round(total_valid_area_ha, 3),
    }

    # ------------------------------------------------------------------
    # Tables 1-7, 12: full-grid analytics
    # ------------------------------------------------------------------
    outputs: Dict[str, str] = {}

    area_tables: Dict[int, pd.DataFrame] = {}
    for year in (2022, 2026):
        tbl = build_area_statistics(cells_by_year[year], year)
        area_tables[year] = tbl
        p = area_statistics_path(year)
        tbl.drop(columns=["class_percent_sum"]).to_csv(p, index=False)
        outputs[f"area_statistics_{year}"] = str(p)
        print(f"  [{year}] area_statistics: {len(tbl)} blocks -> {p.name}")

    high_risk = build_high_risk_summary(cells_by_year)
    p = PHASE6_TABLES_DIR / "high_risk_summary.csv"
    high_risk.to_csv(p, index=False)
    outputs["high_risk_summary"] = str(p)
    print(f"  high_risk_summary -> {p.name}")

    gb = build_green_built_comparison(cells_by_year)
    p = PHASE6_TABLES_DIR / "green_built_comparison.csv"
    gb.to_csv(p, index=False)
    outputs["green_built_comparison"] = str(p)
    print(f"  green_built_comparison: {len(gb)} rows -> {p.name}")

    for year in (2022, 2026):
        tbl = build_landuse_heat_statistics(cells_by_year[year], year)
        p = landuse_heat_statistics_path(year)
        tbl.to_csv(p, index=False)
        outputs[f"landuse_heat_statistics_{year}"] = str(p)
        print(f"  [{year}] landuse_heat_statistics: {len(tbl)} classes -> {p.name}")

    veg_temp = build_vegetation_temperature_relationship(cells_by_year)
    p = PHASE6_TABLES_DIR / "vegetation_temperature_relationship.csv"
    veg_temp.to_csv(p, index=False)
    outputs["vegetation_temperature_relationship"] = str(p)
    print(f"  vegetation_temperature_relationship: {len(veg_temp)} rows -> {p.name}")

    urban = build_urbanization_heat_relationship(cells_by_year)
    p = PHASE6_TABLES_DIR / "urbanization_heat_relationship.csv"
    urban.to_csv(p, index=False)
    outputs["urbanization_heat_relationship"] = str(p)
    print(f"  urbanization_heat_relationship: {len(urban)} rows -> {p.name}")

    temporal = build_temporal_comparison(cells_by_year)
    p = PHASE6_TABLES_DIR / "temporal_comparison.csv"
    temporal.to_csv(p, index=False)
    outputs["temporal_comparison"] = str(p)
    notes_path = PHASE6_TABLES_DIR / "temporal_comparison_notes.json"
    _save_json({
        "caveat": TEMPORAL_CAVEAT,
        "detail": (
            "2022 and 2026 are two snapshot composite dates, not endpoints of "
            "a monitored trend.  Severity classes are per-year LST quartiles "
            "(2022 thresholds 34.2137/38.6264/43.4697 deg-C; 2026 thresholds "
            "32.9354/34.4735/36.1996 deg-C), so a 2026 Severe pixel can be "
            "cooler in deg-C than a 2022 Moderate one.  Any cross-year "
            "statement is class-relative."
        ),
    }, notes_path)
    outputs["temporal_comparison_notes"] = str(notes_path)
    print(f"  temporal_comparison + notes -> {p.name}")

    # ------------------------------------------------------------------
    # Tables 8-9: paired-cell hotspot change and persistence
    # ------------------------------------------------------------------
    change = build_hotspot_change_statistics(cells_by_year)
    p = PHASE6_TABLES_DIR / "hotspot_change_statistics.csv"
    change.to_csv(p, index=False)
    outputs["hotspot_change_statistics"] = str(p)
    print(f"  hotspot_change_statistics -> {p.name}")

    transition = build_transition_matrix(cells_by_year)
    p = PHASE6_TABLES_DIR / "temporal_transition_matrix.csv"
    transition.to_csv(p, index=False)
    outputs["temporal_transition_matrix"] = str(p)
    print(f"  temporal_transition_matrix: {len(transition)} rows -> {p.name}")

    persistence = build_hotspot_persistence(cells_by_year)
    p = PHASE6_TABLES_DIR / "hotspot_persistence.csv"
    persistence.to_csv(p, index=False)
    outputs["hotspot_persistence"] = str(p)
    print(f"  hotspot_persistence -> {p.name}")

    # ------------------------------------------------------------------
    # Table 10: sampled-cell error diagnostics
    # ------------------------------------------------------------------
    print("Stage 3: loading Phase 5 sampled predictions ...")
    sampled = load_sampled_predictions()
    error_maps = write_error_map_rasters(sampled, mask_profile)
    outputs["oof_error_maps"] = error_maps
    err_summary = build_error_diagnostics_summary(sampled)
    p = PHASE6_TABLES_DIR / "error_diagnostics_summary.csv"
    err_summary.to_csv(p, index=False)
    outputs["error_diagnostics_summary"] = str(p)
    err_block = build_error_by_block(sampled)
    p = PHASE6_TABLES_DIR / "error_by_block.csv"
    err_block.to_csv(p, index=False)
    outputs["error_by_block"] = str(p)
    print(
        f"  error diagnostics (TRUE OOF, sampled cells): {len(err_summary)} summary rows, "
        f"{len(err_block)} block rows"
    )

    # ------------------------------------------------------------------
    # Table 11: uncertainty zones
    # ------------------------------------------------------------------
    uncertainty, uncertainty_maps = build_uncertainty_zones(cells_by_year, mask_profile)
    outputs["uncertainty_zone_rasters"] = uncertainty_maps
    p = PHASE6_TABLES_DIR / "uncertainty_summary.csv"
    uncertainty.to_csv(p, index=False)
    outputs["uncertainty_summary"] = str(p)
    print(f"  uncertainty_summary -> {p.name}")

    # ------------------------------------------------------------------
    # Table 12: severity summary
    # ------------------------------------------------------------------
    summary = build_severity_summary(cells_by_year)
    p = PHASE6_TABLES_DIR / "severity_summary.csv"
    summary.to_csv(p, index=False)
    outputs["severity_summary"] = str(p)
    print(f"  severity_summary -> {p.name}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    print("Stage 3: running validations ...")
    validation = run_validations(cells_by_year, area_tables, transition)
    print(f"  Validation passed: {validation['all_passed']}")

    record["outputs"] = outputs
    record["validation"] = validation
    record["headline"] = _headline_numbers(
        cells_by_year, gb, veg_temp, change, persistence, err_summary
    )
    record["finished_utc"] = _now()
    record["status"] = "success" if validation["all_passed"] else "completed_with_validation_issues"
    record["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rasterio": rasterio.__version__,
        "scipy": scipy_stats.__name__ and __import__("scipy").__version__,
    }

    _update_manifest_analytics(record, outputs)
    _append_pipeline_record_stage3(record)
    print(f"  Manifest updated: {MANIFEST_JSON}")
    print(f"  Pipeline record appended: {PIPELINE_RECORD_JSON}")

    return {"record": record, "tables": outputs, "validation": validation}


def _headline_numbers(
    cells_by_year: Dict[int, pd.DataFrame],
    green_built: pd.DataFrame,
    veg_temp: pd.DataFrame,
    change: pd.DataFrame,
    persistence: pd.DataFrame,
    err_summary: pd.DataFrame,
) -> Dict:
    """Extract a compact headline block for the pipeline record."""
    out: Dict = {}
    for year in (2022, 2026):
        sub = green_built[green_built["year"] == year].set_index("group")
        if "green_dominant" in sub.index and "built_dominant" in sub.index:
            out[f"lst_gap_built_minus_green_{year}_C"] = round(
                float(sub.loc["built_dominant", "mean_lst_C"]
                      - sub.loc["green_dominant", "mean_lst_C"]), 4
            )
        hs = cells_by_year[year]
        out[f"high_severe_percent_{year}"] = round(
            100.0 * float((hs["severity"] >= 2).mean()), 4
        )
    corr = veg_temp[veg_temp["analysis"] == "correlation"]
    out["correlations"] = {
        f"{r.year}_{r.variable}": {
            "pearson_r": r.pearson_r,
            "spearman_r": r.spearman_r,
            "n": r.n,
        }
        for r in corr.itertuples()
    }
    cats = change.set_index("change_category")
    for cat in ("persistent_hotspot", "new_hotspot_2026", "disappeared_hotspot_2022"):
        if cat in cats.index:
            out[f"{cat}_px"] = int(cats.loc[cat, "pixel_count"])
    if "mask_iou_2022_vs_2026" in cats.index:
        out["defA_mask_iou"] = float(cats.loc["mask_iou_2022_vs_2026", "mask_iou"])
    out["persistence"] = {
        r.quantity: r.probability for r in persistence.itertuples()
    }
    bands = err_summary[
        (err_summary["analysis"] == "boundary_distance_band_error_rate")
    ]
    out["boundary_band_error_rates"] = {
        f"{r.year}_{r.group}": r.error_rate for r in bands.itertuples()
    }
    overall = err_summary[err_summary["analysis"] == "overall_accuracy"]
    out["sampled_overall_accuracy"] = {
        int(r.year): r.accuracy for r in overall.itertuples()
    }
    return out


def _update_manifest_analytics(record: Dict, outputs: Dict) -> None:
    """Read-modify-write the Phase 6 manifest with the analytics block."""
    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)

    manifest["analytics"] = {
        "stage": 3,
        "generated_utc": _now(),
        "framing": (
            "All statistics restricted to the shared valid mask; NoData never "
            "participates.  2022/2026 are two snapshot dates, not a trend; "
            "per-year quartile classes make cross-year severity "
            "class-relative.  Green/built and landuse contrasts are "
            "associations, not causal effects."
        ),
        "area_method": record["steps"]["data"]["area_method"],
        "mean_valid_pixel_area_m2": record["steps"]["data"]["mean_valid_pixel_area_m2"],
        "total_valid_area_ha": record["steps"]["data"]["total_valid_area_ha"],
        "boundary_thresholds_C": BOUNDARY_THRESHOLDS_C,
        "boundary_bands_C": list(BOUNDARY_BANDS_C),
        "correlation_subsample_n": CORRELATION_SUBSAMPLE_N,
        "correlation_subsample_seed": RANDOM_SEED,
        "uncertainty_thresholds": {
            "top2_margin_below": UNCERTAINTY_MARGIN_THRESHOLD,
            "confidence_below": UNCERTAINTY_CONFIDENCE_THRESHOLD,
            "note": "operational thresholds, not calibrated cut-offs",
        },
        "units": {
            UNIT_LABEL: (
                "5x5 grid blocks from the Phase 5 sampling design (Phase 4 "
                "sampled row/col bins); 20 of 25 occupied; NOT administrative "
                "sectors"
            )
        },
        "oof_error_diagnostics": {
            "source": (
                "data/processed/phase5/tables/oof_predictions.csv, model == "
                "'Random Forest' rows (300,000); TRUE out-of-fold predictions "
                "of the frozen Phase 5 baseline Random Forest"
            ),
            "pooled_oof_accuracy_guard": list(OOF_ACCURACY_GUARD),
            "not_used": (
                "data/processed/phase5/tables/predictions.csv is the "
                "production model's IN-SAMPLE output (accuracy 0.9957) and "
                "must never be used for error diagnostics"
            ),
            "phase6_rfc_note": (
                "the Phase 6 production model (RF-C) is a different model; its "
                "honest accuracy estimates are the frozen Phase 5 numbers "
                "(RF-C spatial-CV 46.99% accuracy / 0.4310 macro-F1; locked "
                "holdout ~39.9% / 0.395), not anything recomputed in Stage 3"
            ),
        },
        "tables": outputs,
        "validation": record["validation"],
    }
    _save_json(manifest, MANIFEST_JSON)


def _append_pipeline_record_stage3(record: Dict) -> None:
    """Append the Stage 3 entry to the Phase 6 pipeline record (idempotent)."""
    with open(PIPELINE_RECORD_JSON) as f:
        existing = json.load(f)

    container = {"phase": 6, "project": "GreenGrid-AI", "stage_records": []}
    if "stage_records" in existing:
        container["stage_records"] = existing["stage_records"]
    else:
        container["stage_records"] = [existing]
    # Idempotent re-runs replace the previous Stage 3 entry.
    container["stage_records"] = [r for r in container["stage_records"] if r.get("stage") != 3]
    container["stage_records"].append(record)
    _save_json(container, PIPELINE_RECORD_JSON)


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for the Phase 6 Stage 3 urban heat analytics."""
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 6 Stage 3 urban heat analytics."
    )
    parser.parse_args(argv)

    try:
        results = run_stage3()
        print(json.dumps(_convert_for_json(results["record"]), indent=2, default=str))
        return 0 if results["record"]["status"] == "success" else 1
    except Exception as exc:
        print(f"Phase 6 Stage 3 pipeline failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
