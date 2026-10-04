"""Phase 7 PRODUCTION re-run: plantation suitability scoring + priority zones.

Re-runs the frozen Phase 7 suitability model for ALL five years (2022-2026)
against the new ``data/processed/phase6_production/`` severity outputs and the
frozen production model semantics (docs_production_build_spec.md).

REUSED from ``src/suitability`` (pure functions of arrays / frozen constants;
``src/`` is NOT modified):
  - ``suitability.config``        — NORM_PERCENTILES, NEED_WEIGHTS,
                                    OPPORTUNITY_WEIGHTS, ROAD_BAND,
                                    GREEN_PROXIMITY_CAP_M, LANDUSE_ELIGIBILITY,
                                    CLASS_EDGES, CLASS_LABELS,
                                    SCENARIO_EXPONENTS (+D weights),
                                    SCORE_NODATA, CLASS_NODATA, MIN_ZONE_PIXELS,
                                    RANDOM_SEED, road_accessibility_score,
                                    green_proximity_score, landuse_eligibility
  - ``suitability.normalization.robust_minmax``
  - ``suitability.need.compute_heat_need``          (variants baseline / D)
  - ``suitability.opportunity.compute_opportunity`` (variants baseline / D)
  - ``suitability.scoring``       — gated_product, weighted_geometric_mean,
                                    classify, write_score_raster,
                                    write_class_raster
  - ``suitability.zones``         — label_zones, polygons_per_zone,
                                    validate_geometries

REIMPLEMENTED here (old code is config-path-coupled to the 2022/2026 phase6
raster layout; every case is recorded in ``DEVIATIONS`` and the manifest):
  - input loading             (old ``suitability.inputs`` reads phase6 2022/2026)
  - per-year normalization loop (old ``normalize_inputs`` iterates YEARS=(2022,2026))
  - scenario score assembly   (old ``compute_scenario_scores`` keyed on 2-yr dicts)
  - per-year exclusion masks  (old ``constraints.build_exclusion_mask`` writes a
    single mask to the old config path; same math/encoding, per-year domain)
  - zone area accounting      (production gate G3 fixes area = px * 900 m2; the
    old Stage-2 code used UTM-reprojected polygon areas)

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/run_phase7_production.py
      [--years 2022,2023,...] [--skip-figures]
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
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from models.config import REFERENCE_RASTER  # noqa: E402
from suitability.config import (  # noqa: E402
    CLASS_EDGES,
    CLASS_LABELS,
    CLASS_NODATA,
    GREEN_PROXIMITY_CAP_M,
    LANDUSE_ELIGIBILITY,
    LANDUSE_NODATA,
    MIN_ZONE_PIXELS,
    NEED_WEIGHTS,
    NORM_PERCENTILES,
    OPPORTUNITY_WEIGHTS,
    RANDOM_SEED,
    ROAD_BAND,
    SCENARIO_D_NEED_WEIGHTS,
    SCENARIO_D_OPPORTUNITY_WEIGHTS,
    SCENARIO_EXPONENTS,
    SCORE_NODATA,
)
from suitability.normalization import robust_minmax  # noqa: E402
from suitability.need import compute_heat_need  # noqa: E402
from suitability.opportunity import compute_opportunity  # noqa: E402
from suitability.scoring import (  # noqa: E402
    classify,
    gated_product,
    weighted_geometric_mean,
    write_class_raster,
    write_score_raster,
)
from suitability.zones import (  # noqa: E402
    label_zones,
    polygons_per_zone,
    validate_geometries,
)

# ---------------------------------------------------------------------------
# Constants (spec: docs_production_build_spec.md + frozen suitability config)
# ---------------------------------------------------------------------------
YEARS = (2022, 2023, 2024, 2025, 2026)
PX_AREA_M2 = 900.0   # 30 m x 30 m
M2_PER_HA = 10_000.0

OUT_ROOT = PROJECT / "data" / "processed" / "phase7_production"
RASTERS_DIR = OUT_ROOT / "rasters"
ZONES_DIR = OUT_ROOT / "zones"
TABLES_DIR = OUT_ROOT / "tables"
FIGURES_DIR = OUT_ROOT / "figures"
MANIFEST_JSON = OUT_ROOT / "phase7_production_manifest.json"
PIPELINE_RECORD_JSON = OUT_ROOT / "phase7_pipeline_record.json"

P6_RASTERS = PROJECT / "data" / "processed" / "phase6_production" / "rasters"
P3_ALIGNED = PROJECT / "data" / "processed" / "phase3" / "aligned"
P3_MASKS = PROJECT / "data" / "processed" / "phase3" / "masks"
P4_FEATURES = PROJECT / "data" / "processed" / "phase4" / "features"
P6_MANIFEST = (PROJECT / "data" / "processed" / "phase6_production" /
               "phase6_production_manifest.json")
SPEC_PATH = PROJECT / "docs_production_build_spec.md"

# Zone level: "top suitability class" = classes 4-5 in 1-indexed terms ==
# class >= 3 (0-indexed High + Very High) of the frozen 5-class scheme.
ZONE_MIN_CLASS = 3

CLASS_COLORS = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]

DEVIATIONS: List[str] = [
    "Domain: per-year phase6_production valid domain (severity_score valid "
    "pixels, spec locked decision #1) replaces the old static phase3 "
    "valid_mask domain; all five years 2022-2026 processed.",
    "Severity score is 0-2 (3-class production model) vs 0-3 in the old "
    "Phase 7 build; handled automatically by the frozen class-relative "
    "p1/p99 min-max normalization (no formula change needed, range-agnostic).",
    "Reimplemented: input loading (old suitability.inputs reads "
    "phase6/ 2022+2026 rasters via config paths).",
    "Reimplemented: per-year normalization loop (old normalize_inputs "
    "iterates config YEARS=(2022,2026) and the old data structure); the "
    "frozen robust_minmax function itself is reused unchanged.",
    "Reimplemented: scenario score assembly (old compute_scenario_scores "
    "iterates year-keyed dicts); the frozen primitives gated_product / "
    "weighted_geometric_mean are reused unchanged. Scenario D exponents "
    "(1,1) make the weighted geometric mean identical to the gated "
    "product, so gated_product is applied directly for D.",
    "Reimplemented: exclusion masks per year (old "
    "constraints.build_exclusion_mask writes ONE mask to the old config "
    "path); same math and encoding (uint8, 1=excluded, 0=domain, nodata "
    "255 declared-never-used), one mask per year because the domain is "
    "now per-year.",
    "Zone area accounting fixed to pixel_count * 900 m2 / 1e4 (production "
    "gate G3); the old Stage-2 code computed UTM-reprojected polygon "
    "areas. Polygons are still written in EPSG:4326.",
    "Zone level = class >= 3 (0-indexed High + Very High == 1-indexed "
    "classes 4-5 per the production spec), one zone set per year; the old "
    "build labelled High and Very High as separate products.",
    "LST is read from phase6_production/rasters/lst_{year}.tif (the "
    "production copy of the aligned Phase-3 LST composite).",
    "Static Opportunity terms (landuse eligibility, road accessibility, "
    "green proximity) reuse the static Phase-3 rasters for every year - "
    "year-to-year Opportunity variation comes only from NDBI/NDVI terms "
    "(same static-vs-dynamic statement as the frozen design).",
    "Not carried over from the old Stage 2 (outside production scope): "
    "priority_confidence index, why-here explanations, spatial "
    "sensitivity extension, environmental/green-built/landuse statistics "
    "tables.",
    "Zone-level temporal transition uses raster pixel overlap "
    "(intersection px / min(px_2022, px_2026) >= 0.5 => persistent pair); "
    "the old build used pixel-level persistence categories instead.",
]


def log(msg: str) -> None:
    print(f"[P7] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _convert(obj):
    if isinstance(obj, dict):
        return {str(k): _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert(obj), f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Raster loading (reimplemented; see DEVIATIONS)
# ---------------------------------------------------------------------------
def read_band(path: Path) -> Tuple[np.ndarray, float]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.profile.get("nodata")
        nodata = float(nodata) if nodata is not None else np.nan
    return arr, nodata


def finite_mask(arr: np.ndarray, nodata: float) -> np.ndarray:
    if np.isnan(nodata):
        return np.isfinite(arr)
    return np.isfinite(arr) & (arr != nodata)


def load_year(year: int) -> Dict:
    """Load all per-year inputs and derive the analysis domain.

    Domain = phase6 severity_score valid pixels (spec locked decision #1).
    Verified upstream to be a subset of the finite NDVI/NDBI/LST/
    vegetation_cover cells, so no further intersection is required; the
    subset relation is re-asserted here cheaply via finiteness.
    """
    sev, sev_nd = read_band(P6_RASTERS / f"severity_score_{year}.tif")
    domain = finite_mask(sev, sev_nd)
    yearly = {"severity_score": sev}
    for name, path in (
        ("ndvi", P3_ALIGNED / f"s2_{year}_ndvi_30m.tif"),
        ("ndbi", P3_ALIGNED / f"s2_{year}_ndbi_30m.tif"),
        ("lst", P6_RASTERS / f"lst_{year}.tif"),
        ("vegetation_cover", P4_FEATURES / f"vegetation_cover_{year}_30m.tif"),
    ):
        arr, nd = read_band(path)
        yearly[name] = arr
        domain &= finite_mask(arr, nd)
    if not domain.any():
        raise AssertionError(f"{year}: empty analysis domain")
    return {"yearly": yearly, "domain": domain}


def load_static() -> Dict:
    """Load the static Opportunity inputs (full-grid, identical every year)."""
    lu, lu_nd = read_band(P3_MASKS / "landuse_raster_30m.tif")
    rd, _ = read_band(P3_MASKS / "roads_distance_30m.tif")
    vd, _ = read_band(P3_MASKS / "vegetation_distance_30m.tif")
    if not (np.isfinite(rd).all() and np.isfinite(vd).all()):
        raise AssertionError("static distance rasters contain non-finite cells")
    return {"landuse": lu, "dist_road_m": rd, "dist_vegetation_m": vd,
            "landuse_nodata": lu_nd}


# ---------------------------------------------------------------------------
# Scoring (frozen formulas via src/suitability; normalization loop reimplemented)
# ---------------------------------------------------------------------------
BASE_QUANTITIES = ("severity_score", "ndvi", "ndbi", "lst", "vegetation_cover")
DERIVED_QUANTITIES = (("one_minus_ndvi", "ndvi"),
                      ("one_minus_vegetation_cover", "vegetation_cover"))


def normalize_year(year: int, yearly_flat: Dict[str, np.ndarray],
                   bounds_rows: List[Dict]) -> Dict[str, np.ndarray]:
    """Per-year p1/p99 min-max over domain cells (frozen robust_minmax reused)."""
    out: Dict[str, np.ndarray] = {}
    for name in BASE_QUANTITIES:
        scaled, bounds = robust_minmax(yearly_flat[name])
        out[name] = scaled
        bounds_rows.append({"year": year, "variable": name, **bounds})
    for name, base in DERIVED_QUANTITIES:
        scaled, bounds = robust_minmax(1.0 - yearly_flat[base])
        out[name] = scaled
        bounds_rows.append({"year": year, "variable": name, **bounds})
    return out


def score_year(norm: Dict[str, np.ndarray], static_flat: Dict[str, np.ndarray]):
    """Need / Opportunity / baseline + scenario suitability for one year."""
    need = {
        "baseline": compute_heat_need(norm, "baseline"),
        "D": compute_heat_need(norm, "D"),
    }
    opp = {
        "baseline": compute_opportunity(norm, static_flat, "baseline"),
        "D": compute_opportunity(norm, static_flat, "D"),
    }
    if tuple(SCENARIO_EXPONENTS["A"]) != (1.0, 1.0):
        raise AssertionError("frozen Scenario A exponents changed")
    if tuple(SCENARIO_EXPONENTS["D"]) != (1.0, 1.0):
        raise AssertionError("frozen Scenario D exponents changed")
    scores = {
        # Baseline (Scenario A): gated product Need x Opportunity / 100.
        "A": gated_product(need["baseline"], opp["baseline"]["opportunity"]),
    }
    for scenario in ("B", "C"):
        e_need, e_opp = SCENARIO_EXPONENTS[scenario]
        scores[scenario] = weighted_geometric_mean(
            need["baseline"], opp["baseline"]["opportunity"], e_need, e_opp
        )
    # Scenario D exponents (1,1): weighted geometric mean == gated product.
    scores["D"] = gated_product(need["D"], opp["D"]["opportunity"])
    return need, opp, scores


# ---------------------------------------------------------------------------
# Exclusion mask (reimplemented per year; encoding per frozen constraints.py)
# ---------------------------------------------------------------------------
def write_exclusion_mask(domain: np.ndarray, profile: Dict, path: Path) -> Dict:
    raster = np.where(domain, 0, 1).astype(np.uint8)
    out_profile = profile.copy()
    out_profile.update({"dtype": "uint8", "count": 1, "nodata": 255,
                        "compress": "lzw"})
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(raster, 1)
    return {
        "output_path": str(path),
        "encoding": "1 = excluded (invalid/NoData), 0 = analysis domain",
        "excluded_px": int((raster == 1).sum()),
        "analysis_domain_px": int((raster == 0).sum()),
        "note": ("Only hard exclusion is the per-year invalid domain. No "
                 "water / building-footprint / road-surface exclusion layer "
                 "exists (stated limitation, not a silent omission)."),
    }


# ---------------------------------------------------------------------------
# Priority zones (frozen rule: 8-conn, >= MIN_ZONE_PIXELS, class >= 3)
# ---------------------------------------------------------------------------
def extract_zones(year: int, cls_grid: np.ndarray, domain: np.ndarray,
                  rows: np.ndarray, cols: np.ndarray,
                  scores_A: np.ndarray, need_base: np.ndarray,
                  opp_base: np.ndarray, transform) -> Dict:
    """Zones from class >= ZONE_MIN_CLASS; areas = px * 900 m2 (gate G3)."""
    binary = (cls_grid >= ZONE_MIN_CLASS) & domain
    labelled, n_kept = label_zones(binary)
    polygons = polygons_per_zone(labelled, transform) if n_kept else {}
    validation = validate_geometries(polygons)
    validation.update({
        "mask_pixels_raw": int(binary.sum()),
        "mask_pixels_kept": int((labelled > 0).sum()),
        "noise_pixels_removed": int(binary.sum() - (labelled > 0).sum()),
    })

    flat_ids = labelled[rows, cols].astype(np.int32)
    cls_flat = cls_grid[rows, cols]
    zone_rows: List[Dict] = []
    for zid in sorted(int(i) for i in np.unique(flat_ids[flat_ids > 0])):
        sel = flat_ids == zid
        n_px = int(sel.sum())
        geom = polygons[zid]
        cx, cy = geom.centroid.x, geom.centroid.y
        vals, counts = np.unique(cls_flat[sel], return_counts=True)
        zone_rows.append({
            "zone_id": zid,
            "year": year,
            "pixel_count": n_px,
            "area_ha": round(n_px * PX_AREA_M2 / M2_PER_HA, 3),
            "centroid_lon": round(cx, 7),
            "centroid_lat": round(cy, 7),
            "mean_suitability": round(float(scores_A[sel].mean()), 4),
            "median_suitability": round(float(np.median(scores_A[sel])), 4),
            "mean_class": round(float(cls_flat[sel].mean()), 3),
            "dominant_class": int(vals[np.argmax(counts)]),
            "mean_heat_need": round(float(need_base[sel].mean()), 4),
            "mean_opportunity": round(float(opp_base[sel].mean()), 4),
        })
    frame = pd.DataFrame(zone_rows)
    validation["zone_pixel_count_sum"] = int(frame["pixel_count"].sum()) if n_kept else 0
    validation["pixel_counts_match"] = bool(
        validation["zone_pixel_count_sum"] == validation["mask_pixels_kept"]
    )
    return {"frame": frame, "labelled": labelled, "flat_ids": flat_ids,
            "polygons": polygons, "binary": binary, "validation": validation,
            "n_zones": n_kept}


def write_zones_geojson(frame: pd.DataFrame, polygons: Dict[int, object], path: Path) -> None:
    if frame.empty:
        gdf = gpd.GeoDataFrame(
            {"zone_id": pd.Series(dtype="int64"), "year": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries(dtype="geometry"), crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(
            frame,
            geometry=[polygons[int(z)] for z in frame["zone_id"]],
            crs="EPSG:4326",
        )
    gdf.to_file(path, driver="GeoJSON")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def class_summary_table(year: int, cls_flat: np.ndarray) -> pd.DataFrame:
    n_dom = cls_flat.size
    rows = []
    for c in range(len(CLASS_LABELS)):
        n = int((cls_flat == c).sum())
        rows.append({
            "year": year, "class": c, "label": CLASS_LABELS[c],
            "pixel_count": n,
            "area_ha": round(n * PX_AREA_M2 / M2_PER_HA, 3),
            "pct_of_domain": round(100.0 * n / n_dom, 4),
        })
    rows.append({
        "year": year, "class": -1, "label": "TOTAL (domain)",
        "pixel_count": n_dom,
        "area_ha": round(n_dom * PX_AREA_M2 / M2_PER_HA, 3),
        "pct_of_domain": 100.0,
    })
    return pd.DataFrame(rows)


def area_statistics_table(year: int, cls_flat: np.ndarray,
                          scenario_cls: Dict[str, np.ndarray]) -> pd.DataFrame:
    n_dom = cls_flat.size
    rows = [{"year": year, "category": "analysis_domain", "pixel_count": n_dom,
             "area_ha": round(n_dom * PX_AREA_M2 / M2_PER_HA, 3),
             "pct_of_domain": 100.0}]
    for c in range(len(CLASS_LABELS)):
        n = int((cls_flat == c).sum())
        rows.append({
            "year": year,
            "category": f"class_{c}_{CLASS_LABELS[c].replace(' ', '_').lower()}",
            "pixel_count": n,
            "area_ha": round(n * PX_AREA_M2 / M2_PER_HA, 3),
            "pct_of_domain": round(100.0 * n / n_dom, 4),
        })
    for scenario in ("B", "C", "D"):
        n = int((scenario_cls[scenario] >= ZONE_MIN_CLASS).sum())
        rows.append({
            "year": year,
            "category": f"scenario_{scenario.lower()}_class_ge_{ZONE_MIN_CLASS}",
            "pixel_count": n,
            "area_ha": round(n * PX_AREA_M2 / M2_PER_HA, 3),
            "pct_of_domain": round(100.0 * n / n_dom, 4),
        })
    return pd.DataFrame(rows)


def build_temporal_transition(zones_2022: Dict, zones_2026: Dict,
                              scores_2022: np.ndarray, scores_2026: np.ndarray) -> pd.DataFrame:
    """Zone-level 2022->2026 persistence.

    A 2022 zone and a 2026 zone form a persistent pair when their raster
    pixel overlap is >= 50% of the smaller of the two zones; unmatched 2022
    zones are declining, unmatched 2026 zones are emerging.
    """
    lab22 = zones_2022["labelled"]
    lab26 = zones_2026["labelled"]
    both = (lab22 > 0) & (lab26 > 0)
    pairs: Dict[Tuple[int, int], int] = {}
    for a, b in zip(lab22[both].ravel(), lab26[both].ravel()):
        key = (int(a), int(b))
        pairs[key] = pairs.get(key, 0) + 1
    ids22 = sorted(int(i) for i in np.unique(lab22[lab22 > 0]))
    ids26 = sorted(int(i) for i in np.unique(lab26[lab26 > 0]))
    px22 = {i: int((lab22 == i).sum()) for i in ids22}
    px26 = {i: int((lab26 == i).sum()) for i in ids26}
    sc22 = {i: float(scores_2022[zones_2022["flat_ids"] == i].mean()) for i in ids22}
    sc26 = {i: float(scores_2026[zones_2026["flat_ids"] == i].mean()) for i in ids26}

    matched22: Dict[int, Tuple[int, int, float]] = {}
    matched26: Dict[int, Tuple[int, int, float]] = {}
    for (a, b), inter in sorted(pairs.items()):
        frac = inter / min(px22[a], px26[b])
        if frac >= 0.5:
            if a not in matched22 or frac > matched22[a][2]:
                matched22[a] = (b, inter, frac)
            if b not in matched26 or frac > matched26[b][2]:
                matched26[b] = (a, inter, frac)

    rows: List[Dict] = []
    for a in ids22:
        if a in matched22:
            b, inter, frac = matched22[a]
            rows.append({
                "zone_id_2022": a, "zone_id_2026": b, "persistence": "persistent",
                "pixel_count_2022": px22[a], "pixel_count_2026": px26[b],
                "overlap_px": inter, "overlap_fraction": round(frac, 4),
                "area_ha_2022": round(px22[a] * PX_AREA_M2 / M2_PER_HA, 3),
                "area_ha_2026": round(px26[b] * PX_AREA_M2 / M2_PER_HA, 3),
                "mean_suitability_2022": round(sc22[a], 4),
                "mean_suitability_2026": round(sc26[b], 4),
            })
        else:
            rows.append({
                "zone_id_2022": a, "zone_id_2026": "", "persistence": "declining",
                "pixel_count_2022": px22[a], "pixel_count_2026": 0,
                "overlap_px": 0, "overlap_fraction": 0.0,
                "area_ha_2022": round(px22[a] * PX_AREA_M2 / M2_PER_HA, 3),
                "area_ha_2026": 0.0,
                "mean_suitability_2022": round(sc22[a], 4),
                "mean_suitability_2026": np.nan,
            })
    for b in ids26:
        if b not in matched26:
            rows.append({
                "zone_id_2022": "", "zone_id_2026": b, "persistence": "emerging",
                "pixel_count_2022": 0, "pixel_count_2026": px26[b],
                "overlap_px": 0, "overlap_fraction": 0.0,
                "area_ha_2022": 0.0,
                "area_ha_2026": round(px26[b] * PX_AREA_M2 / M2_PER_HA, 3),
                "mean_suitability_2022": np.nan,
                "mean_suitability_2026": round(sc26[b], 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def class_map_figure(cls_grid: np.ndarray, year: int, path: Path) -> None:
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
    shown = np.where(cls_grid < 0, np.nan, cls_grid).astype(np.float64)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(shown, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, ticks=range(5), shrink=0.75)
    cbar.ax.set_yticklabels([CLASS_LABELS[c] for c in range(5)])
    ax.set_title(f"Tree-plantation suitability class {year} "
                 f"(baseline Scenario A, per-year normalized)")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def need_opportunity_figure(need: np.ndarray, opp: np.ndarray, year: int,
                            path: Path, max_points: int = 200_000) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    n = need.size
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        x, y = opp[idx], need[idx]
    else:
        x, y = opp, need
    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(x, y, gridsize=80, cmap="viridis", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_xlabel("Plantation Opportunity (0-100)")
    ax.set_ylabel("Heat Need (0-100)")
    ax.set_title(f"Heat Need vs Plantation Opportunity {year} "
                 f"(baseline; {x.size:,} of {n:,} domain px shown)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summary_panel_figure(cls_grids: Dict[int, np.ndarray],
                         top_class_ha: Dict[int, float], path: Path) -> None:
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
    fig, axes = plt.subplots(2, 5, figsize=(25, 11))
    last_im = None
    for ax, year in zip(axes[0], YEARS):
        shown = np.where(cls_grids[year] < 0, np.nan,
                         cls_grids[year]).astype(np.float64)
        last_im = ax.imshow(shown, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{year}")
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(last_im, ax=list(axes[0]), ticks=range(5), shrink=0.75)
    cbar.ax.set_yticklabels([CLASS_LABELS[c] for c in range(5)])
    cbar.set_label("suitability class")
    axes[1, 0].axis("off")
    axes[1, 4].axis("off")
    ax = axes[1, 1]
    ax.bar([str(y) for y in YEARS], [top_class_ha[y] for y in YEARS], color="#d73027")
    ax.set_ylabel("High + Very High area (ha)")
    ax.set_title(f"Priority area (class >= {ZONE_MIN_CLASS}) per year")
    ax = axes[1, 2]
    n_dom = {y: int((cls_grids[y] >= 0).sum()) for y in YEARS}
    ax.bar([str(y) for y in YEARS],
           [n_dom[y] * PX_AREA_M2 / M2_PER_HA for y in YEARS], color="#4575b4")
    ax.set_ylabel("Analysis domain (ha)")
    ax.set_title("Per-year analysis domain (snapshot extents differ)")
    axes[1, 3].axis("off")
    fig.suptitle("Phase 7 production - 5-year suitability summary "
                 "(class-relative per-year normalization; snapshots, not trends)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run(years: Tuple[int, ...], skip_figures: bool) -> Dict:
    for d in (RASTERS_DIR, ZONES_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    record: Dict = {
        "phase": 7, "variant": "production", "project": "GreenGrid-AI",
        "project_root": str(PROJECT), "started_utc": _now(),
        "random_seed": RANDOM_SEED, "years": list(years),
        "steps": {}, "deviations": list(DEVIATIONS),
    }
    steps: Dict[str, Dict] = {}
    input_hashes: Dict[str, str] = {}

    with rasterio.open(REFERENCE_RASTER) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
    log(f"Reference grid: {profile['width']}x{profile['height']} {ref.crs}")

    static = load_static()
    for p in (P3_MASKS / "landuse_raster_30m.tif",
              P3_MASKS / "roads_distance_30m.tif",
              P3_MASKS / "vegetation_distance_30m.tif",
              REFERENCE_RASTER):
        input_hashes[str(p.relative_to(PROJECT))] = sha256(p)

    bounds_rows: List[Dict] = []
    per_year: Dict[int, Dict] = {}
    cls_grids: Dict[int, np.ndarray] = {}

    for year in years:
        t0 = time.time()
        log(f"--- {year}: loading inputs")
        data = load_year(year)
        domain = data["domain"]
        yearly_flat = {k: v[domain] for k, v in data["yearly"].items()}
        static_flat = {
            "landuse": static["landuse"][domain].astype(np.int64),
            "dist_road_m": static["dist_road_m"][domain],
            "dist_vegetation_m": static["dist_vegetation_m"][domain],
        }
        for p in (P6_RASTERS / f"severity_score_{year}.tif",
                  P6_RASTERS / f"lst_{year}.tif",
                  P3_ALIGNED / f"s2_{year}_ndvi_30m.tif",
                  P3_ALIGNED / f"s2_{year}_ndbi_30m.tif",
                  P4_FEATURES / f"vegetation_cover_{year}_30m.tif"):
            input_hashes[str(p.relative_to(PROJECT))] = sha256(p)

        log(f"--- {year}: normalizing ({int(domain.sum()):,} domain px)")
        norm = normalize_year(year, yearly_flat, bounds_rows)
        need, opp, scores = score_year(norm, static_flat)

        log(f"--- {year}: writing rasters")
        rows, cols = np.nonzero(domain)
        rast_meta: Dict[str, Dict] = {}
        rast_meta["heat_need"] = write_score_raster(
            need["baseline"], RASTERS_DIR / f"heat_need_{year}.tif",
            profile, rows, cols)
        rast_meta["plantation_opportunity"] = write_score_raster(
            opp["baseline"]["opportunity"],
            RASTERS_DIR / f"plantation_opportunity_{year}.tif",
            profile, rows, cols)
        rast_meta["suitability"] = write_score_raster(
            scores["A"], RASTERS_DIR / f"suitability_{year}.tif",
            profile, rows, cols)
        for scenario in ("B", "C", "D"):
            rast_meta[f"scenario_{scenario.lower()}"] = write_score_raster(
                scores[scenario],
                RASTERS_DIR / f"suitability_scenario_{scenario.lower()}_{year}.tif",
                profile, rows, cols)
        cls_meta = write_class_raster(
            scores["A"].astype(np.float32),
            RASTERS_DIR / f"suitability_class_{year}.tif", profile, rows, cols)
        excl_meta = write_exclusion_mask(
            domain, profile, RASTERS_DIR / f"exclusion_mask_{year}.tif")

        # In-memory class grid, identical to the written raster: the writer
        # classifies the float32-cast scores, so reproduce exactly that.
        scores_f32 = scores["A"].astype(np.float32)
        cls_grid = np.full((profile["height"], profile["width"]), CLASS_NODATA,
                           dtype=np.int16)
        cls_grid[rows, cols] = classify(scores_f32.astype(np.float64))
        cls_grids[year] = cls_grid
        scenario_cls = {s: classify(v.astype(np.float32).astype(np.float64))
                        for s, v in scores.items()}

        log(f"--- {year}: priority zones (class >= {ZONE_MIN_CLASS})")
        zones = extract_zones(year, cls_grid, domain, rows, cols,
                              scores["A"], need["baseline"],
                              opp["baseline"]["opportunity"], transform)
        geojson_path = ZONES_DIR / f"priority_zones_{year}.geojson"
        write_zones_geojson(zones["frame"], zones["polygons"], geojson_path)

        log(f"--- {year}: tables")
        cls_flat = cls_grid[rows, cols]
        class_summary_table(year, cls_flat).to_csv(
            TABLES_DIR / f"suitability_summary_{year}.csv", index=False)
        zones["frame"].to_csv(
            TABLES_DIR / f"priority_zone_statistics_{year}.csv", index=False)
        ranking = zones["frame"].sort_values(
            ["mean_suitability", "area_ha"], ascending=[False, False]
        ).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        ranking.to_csv(TABLES_DIR / f"priority_ranking_{year}.csv", index=False)
        area_statistics_table(year, cls_flat, scenario_cls).to_csv(
            TABLES_DIR / f"suitability_area_statistics_{year}.csv", index=False)

        per_year[year] = {
            "domain_px": int(domain.sum()), "scores": scores, "zones": zones,
            "rast_meta": rast_meta, "cls_meta": cls_meta, "excl_meta": excl_meta,
        }
        hvh = int(((cls_grid >= ZONE_MIN_CLASS) & domain).sum())
        steps[str(year)] = {
            "status": "success",
            "elapsed_s": round(time.time() - t0, 3),
            "domain_px": int(domain.sum()),
            "mean_need": round(float(need["baseline"].mean()), 4),
            "mean_opportunity": round(float(opp["baseline"]["opportunity"].mean()), 4),
            "mean_suitability_A": round(float(scores["A"].mean()), 4),
            "class_ge_3_px": hvh,
            "class_ge_3_ha": round(hvh * PX_AREA_M2 / M2_PER_HA, 3),
            "n_zones": zones["n_zones"],
            "zone_geometry_valid": bool(zones["validation"]["all_valid"]),
            "zone_pixel_counts_match": bool(zones["validation"]["pixel_counts_match"]),
        }
        log(f"--- {year}: done in {steps[str(year)]['elapsed_s']}s "
            f"(zones={zones['n_zones']}, High+VH={hvh:,} px)")

    pd.DataFrame(bounds_rows).to_csv(
        TABLES_DIR / "normalization_parameters.csv", index=False)
    for year in years:
        pd.DataFrame([r for r in bounds_rows if r["year"] == year]).to_csv(
            TABLES_DIR / f"normalization_parameters_{year}.csv", index=False)

    log("Temporal transition table (2022 -> 2026)")
    if 2022 in per_year and 2026 in per_year:
        # rebuild flat score arrays the transition table indexes by flat ids
        transition_df = build_temporal_transition(
            per_year[2022]["zones"], per_year[2026]["zones"],
            per_year[2022]["scores"]["A"], per_year[2026]["scores"]["A"])
    else:
        transition_df = pd.DataFrame()
        record["deviations"].append(
            "Temporal transition skipped: requires both 2022 and 2026.")
    transition_df.to_csv(TABLES_DIR / "temporal_priority_transition.csv", index=False)

    if not skip_figures:
        log("Figures")
        t0 = time.time()
        top_ha: Dict[int, float] = {}
        for year in years:
            cls_map_path = FIGURES_DIR / f"suitability_class_map_{year}.png"
            class_map_figure(cls_grids[year], year, cls_map_path)
            no_path = FIGURES_DIR / f"need_vs_opportunity_{year}.png"
            # flat need/opportunity are recomputable from written rasters;
            # keep them from the scoring step instead (memory-light: 2 flats)
            need_opp = _load_flat_pair(year, domain_unused=None)
            need_opportunity_figure(need_opp["need"], need_opp["opp"], year, no_path)
            top_ha[year] = steps[str(year)]["class_ge_3_ha"]
        panel_path = FIGURES_DIR / "suitability_5year_summary.png"
        summary_panel_figure(cls_grids, top_ha, panel_path)
        steps["figures"] = {"status": "success",
                            "elapsed_s": round(time.time() - t0, 3)}

    log("Manifest + pipeline record")
    with open(P6_MANIFEST) as f:
        p6_manifest = json.load(f)

    outputs: Dict[str, str] = {}
    output_hashes: Dict[str, str] = {}
    for sub in ("rasters", "zones", "tables", "figures"):
        for p in sorted((OUT_ROOT / sub).rglob("*")):
            if p.is_file():
                outputs[str(p.relative_to(OUT_ROOT))] = str(p.relative_to(PROJECT))
                output_hashes[str(p.relative_to(OUT_ROOT))] = sha256(p)

    manifest = {
        "stage": "phase7_production",
        "created_at_utc": _now(),
        "spec": {"path": "docs_production_build_spec.md",
                 "sha256": sha256(SPEC_PATH),
                 "version": "2026-10-03 production build spec"},
        "model_lineage": {
            "phase6_manifest": str(P6_MANIFEST.relative_to(PROJECT)),
            "model_id": p6_manifest.get("model_id"),
            "model_sha256": p6_manifest.get("model_sha256"),
            "schema": p6_manifest.get("schema"),
            "note": ("Suitability scores consume the phase6_production "
                     "severity_score / lst rasters produced by the frozen "
                     "3-class production model (severity_score in [0,2])."),
        },
        "frozen_parameters": {
            "norm_percentiles": list(NORM_PERCENTILES),
            "need_weights": NEED_WEIGHTS,
            "opportunity_weights": OPPORTUNITY_WEIGHTS,
            "scenario_exponents": {k: list(v) for k, v in SCENARIO_EXPONENTS.items()},
            "scenario_D_need_weights": SCENARIO_D_NEED_WEIGHTS,
            "scenario_D_opportunity_weights": SCENARIO_D_OPPORTUNITY_WEIGHTS,
            "road_band": ROAD_BAND,
            "green_proximity_cap_m": GREEN_PROXIMITY_CAP_M,
            "landuse_eligibility": {int(k): v for k, v in LANDUSE_ELIGIBILITY.items()},
            "landuse_nodata": LANDUSE_NODATA,
            "class_edges": list(CLASS_EDGES),
            "class_labels": {int(k): v for k, v in CLASS_LABELS.items()},
            "zone_rule": {"min_class": ZONE_MIN_CLASS,
                          "min_zone_pixels": MIN_ZONE_PIXELS,
                          "connectivity": "8"},
            "area_rule": "area_ha = pixel_count * 900 m2 / 1e4",
            "score_nodata": SCORE_NODATA, "class_nodata": CLASS_NODATA,
            "random_seed": RANDOM_SEED,
        },
        "domain_rule": ("per-year phase6_production severity_score valid "
                        "pixels (spec locked decision #1)"),
        "inputs": [{"path": k, "sha256": v} for k, v in sorted(input_hashes.items())],
        "outputs": [{"path": k, "sha256": v} for k, v in sorted(output_hashes.items())],
        "per_year": {str(y): steps[str(y)] for y in years},
        "deviations": record["deviations"],
        "terminology_note": (
            "Outputs describe POTENTIAL PLANTATION SUITABILITY - a relative, "
            "decision-support ranking on per-year-normalized inputs. They are "
            "NOT a statement of legal availability, land ownership, or a "
            "physical plantability guarantee, and make no claim of planting "
            "survival probability."),
    }
    save_json(manifest, MANIFEST_JSON)
    outputs["phase7_production_manifest.json"] = str(MANIFEST_JSON.relative_to(PROJECT))
    outputs["phase7_pipeline_record.json"] = str(PIPELINE_RECORD_JSON.relative_to(PROJECT))

    record["steps"].update(steps)
    record["outputs"] = outputs
    record["finished_utc"] = _now()
    record["total_elapsed_s"] = round(
        sum(s.get("elapsed_s", 0) for s in steps.values() if isinstance(s, dict)),
        3)
    record["status"] = "success"
    record["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "rasterio": rasterio.__version__, "geopandas": gpd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    save_json(record, PIPELINE_RECORD_JSON)
    log(f"Manifest: {MANIFEST_JSON}")
    log(f"Pipeline record: {PIPELINE_RECORD_JSON}")
    return {"record": record, "manifest": manifest, "per_year": per_year}


def _load_flat_pair(year: int, domain_unused) -> Dict[str, np.ndarray]:
    """Re-read the written need/opportunity rasters for figure use.

    Keeps the figures honest w.r.t. what is on disk (values identical to the
    in-memory flats; nodata cells are excluded).
    """
    out = {}
    for key, name in (("need", f"heat_need_{year}.tif"),
                      ("opp", f"plantation_opportunity_{year}.tif")):
        arr, nd = read_band(RASTERS_DIR / name)
        out[key] = arr[np.isfinite(arr) & (arr != nd)]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 7 production suitability build.")
    parser.add_argument("--years", default=",".join(str(y) for y in YEARS),
                        help="comma-separated years")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)
    years = tuple(int(y) for y in args.years.split(","))

    try:
        result = run(years, args.skip_figures)
        print(json.dumps(_convert(result["record"]), indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"Phase 7 production pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
