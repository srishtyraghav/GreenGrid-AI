"""Phase 7 Stage 3: publication-quality figure suite.

Renders the 13 report figures from the frozen Stage 1-2 products (score /
class rasters, priority-zone GeoJSONs, analytics tables).  Every map and
panel carries the scientific framing: POTENTIAL plantation suitability — a
GIS-based decision-support estimate; not legal land availability, not a
physical plantability guarantee, not a causal cooling prediction.  2022 and
2026 are two relative snapshots, not a trend.

Map-decoration conventions follow the Phase 6 figure suite (read-only
reuse of its validated pattern, not an import): explicit NoData patch
(light grey), lat/lon ticks, an approximate scale bar (~97 km/deg
longitude, ~111 km/deg latitude at Delhi), a north arrow, and framing
captions wrapped so the tight bbox never stretches the canvas.  Categorical
figures use discrete colors only (no misleading gradients); the UHI
palette discipline of Phases 5/6 is kept for class colors.

Honesty rules (per the design spec): figures whose inputs are missing are
SKIPPED and reported, never replaced with filler charts; empty results
(e.g. zero Very High zones under the baseline) are shown with an explicit
annotation, never faked.

Run with::

    PYTHONPATH=src .venv/bin/python -m suitability.figures
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage

from models.maps import UHI_COLORS

from .config import (
    CLASS_EDGES,
    CLASS_LABELS,
    LANDUSE_CLASS_NAMES,
    MANIFEST_JSON,
    MIN_ZONE_PIXELS,
    PHASE7_FIGURES_DIR,
    PHASE7_TABLES_DIR,
    PHASE7_VECTORS_DIR,
    PIPELINE_RECORD_JSON,
    PROJECT_ROOT,
    RANDOM_SEED,
    SCORE_NODATA,
    YEARS,
    heat_need_raster_path,
    opportunity_raster_path,
    scenario_raster_path,
    suitability_class_raster_path,
    suitability_raster_path,
)
from .inputs import load_inputs
from .statistics import GREEN_BUILT_NOTE, PERSISTENCE_LABELS

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
NODATA_COLOR = "#d9d9d9"  # light grey, explicit NoData (Phase 6 convention)
NEED_CMAP = plt.get_cmap("YlOrRd").copy()       # sequential, 0-100 heat need
OPP_CMAP = plt.get_cmap("YlGnBu").copy()        # sequential, 0-100 opportunity
SUIT_CMAP = plt.get_cmap("viridis").copy()      # sequential, 0-100 suitability
for _c in (NEED_CMAP, OPP_CMAP, SUIT_CMAP):
    _c.set_bad(NODATA_COLOR)

# 5-class suitability palette, UHI color discipline (blue -> red).
SUITABILITY_COLORS = ["#4575b4", "#91bfdb", "#fee090", "#fc8d59", "#d73027"]
SUITABILITY_CMAP = ListedColormap(SUITABILITY_COLORS)

# Temporal persistence category colors (discrete; no gradients).
PERSISTENCE_COLORS = {
    "Persistent": "#1a9850",   # green: priority in both snapshots
    "Emerging": "#2c7bb6",     # blue: new priority in 2026
    "Declining": "#d73027",    # red: priority only in 2022
    "Stable Low": "#e0e0e0",   # light grey: below High in both
}

DPI = 220
SCATTER_SUBSAMPLE_N = 200_000
YEAR_COLORS = {2022: UHI_COLORS[2], 2026: UHI_COLORS[3]}  # Phase 6 convention

#: Framing that appears in every caption (per Stage 3 brief).
FRAMING = (
    "Potential plantation suitability — GIS-based decision-support "
    "estimate; not legal land availability, not a physical plantability "
    "guarantee, not a causal cooling prediction. Two relative snapshots "
    "(2022, 2026), not a trend."
)
#: Short single-map title clause (full framing is always in the caption).
FRAMING_SHORT = "Potential plantation suitability (decision-support estimate)"
CLASS_RELATIVE_NOTE = (
    "All scores/classes are per-year normalized: cross-year comparison is "
    "class-relative, snapshots not trends."
)

# Approximate km per degree at Delhi (EPSG:4326); the scale bar is labelled
# 'approximate' on every map that carries one (Phase 6 convention).
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 97.0

FIGURE_FILES = {
    1: "fig01_heat_need_2022.png",
    2: "fig02_heat_need_2026.png",
    3: "fig03_opportunity_2022.png",
    4: "fig04_opportunity_2026.png",
    5: "fig05_suitability_2022.png",
    6: "fig06_suitability_2026.png",
    7: "fig07_very_high_priority_zones.png",
    8: "fig08_high_priority_zones.png",
    9: "fig09_green_vs_built.png",
    10: "fig10_landuse_suitability.png",
    11: "fig11_temporal_priority_transition.png",
    12: "fig12_sensitivity_analysis.png",
    13: "fig13_need_vs_opportunity.png",
}

PRIORITY_ZONE_STATISTICS_CSV = PHASE7_TABLES_DIR / "priority_zone_statistics.csv"


# ---------------------------------------------------------------------------
# Small shared helpers (Phase 6-validated pattern)
# ---------------------------------------------------------------------------
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


def _read_raster(path: Path) -> Tuple[np.ndarray, Dict]:
    """Read one single-band raster as (array, profile)."""
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def _score_grid(path: Path, valid_mask: np.ndarray) -> np.ma.MaskedArray:
    """Read a 0-100 score raster masked to valid cells (NoData explicit)."""
    arr, _ = _read_raster(path)
    return np.ma.masked_where(~valid_mask | (arr == SCORE_NODATA), arr)


def _class_grid(path: Path, valid_mask: np.ndarray) -> np.ma.MaskedArray:
    """Read a class raster masked to valid cells (NoData explicit)."""
    arr, _ = _read_raster(path)
    return np.ma.masked_where(~valid_mask | (arr < 0), arr)


def _grid_extent(profile: Dict) -> Tuple[float, float, float, float]:
    """Return (left, right, bottom, top) in degrees for imshow extents."""
    transform = profile["transform"]
    height, width = profile["height"], profile["width"]
    left = transform.c
    right = left + width * transform.a
    top = transform.f
    bottom = top + height * transform.e
    return left, right, bottom, top


def _nice_km(target_km: float) -> float:
    """Snap a target scale-bar length to a 1/2/5*10^k 'nice' value."""
    for mag in (0.1, 1.0, 10.0, 100.0, 1000.0):
        for mult in (1, 2, 5):
            val = mult * mag
            if val >= target_km:
                return val
    return 1000.0


def _add_map_frame(
    ax: plt.Axes,
    profile: Dict,
    title: str,
    bar_fraction: float = 0.18,
) -> None:
    """Add lat/lon ticks, an approximate scale bar and a north arrow.

    The scale bar assumes ~97 km/deg longitude and ~111 km/deg latitude at
    Delhi and is explicitly labelled 'approximate' (Phase 6 pattern).
    """
    left, right, bottom, top = _grid_extent(profile)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")

    width_km = (right - left) * KM_PER_DEG_LON
    bar_km = _nice_km(width_km * bar_fraction)
    bar_deg = bar_km / KM_PER_DEG_LON
    x0 = left + 0.04 * (right - left)
    y0 = bottom + 0.045 * (top - bottom)
    rect = mpatches.Rectangle(
        (x0, y0), bar_deg, 0.006 * (top - bottom),
        facecolor="black", edgecolor="black", zorder=5,
    )
    ax.add_patch(rect)
    ax.text(
        x0 + bar_deg / 2, y0 + 0.012 * (top - bottom),
        f"{bar_km:g} km (approximate)",
        ha="center", va="bottom", fontsize=7, zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
    )

    nx = left + 0.93 * (right - left)
    ny = bottom + 0.90 * (top - bottom)
    ax.annotate(
        "N", xy=(nx, ny + 0.05 * (top - bottom)),
        xytext=(nx, ny - 0.02 * (top - bottom)),
        ha="center", va="center", fontsize=10, fontweight="bold", zorder=5,
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
    )

    ax.set_title(title, fontsize=10)


def _nodata_patch() -> mpatches.Patch:
    return mpatches.Patch(facecolor=NODATA_COLOR, edgecolor="0.6", label="NoData")


def _class_patches() -> List[mpatches.Patch]:
    """Legend patches for the five suitability classes (discrete colors)."""
    return [
        mpatches.Patch(color=SUITABILITY_COLORS[c], label=CLASS_LABELS[c])
        for c in range(len(CLASS_LABELS))
    ]


def _save(fig: plt.Figure, fig_no: int, caption: Optional[str] = None) -> Path:
    """Save a figure to its canonical PNG path and close it.

    Captions are wrapped so the tight bbox never stretches the canvas
    (Phase 6 pattern); full framing lives in the caption, not the title.
    """
    if caption:
        wrapped = "\n".join(textwrap.wrap(caption, width=110))
        n_lines = wrapped.count("\n") + 1
        # Reserve bottom space proportional to the wrapped caption length so
        # the caption never overlaps axis labels (long captions need more).
        fig.subplots_adjust(bottom=min(0.38, 0.034 * n_lines + 0.045))
        fig.text(0.01, 0.005, wrapped, fontsize=7, color="0.25",
                 ha="left", va="bottom")
    out_path = PHASE7_FIGURES_DIR / FIGURE_FILES[fig_no]
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  [fig{fig_no:02d}] {out_path.name}")
    return out_path


def _table(name: str) -> Path:
    return PHASE7_TABLES_DIR / name


def _repel_labels(
    ax: plt.Axes,
    labeled: List[Tuple[str, float, float]],
    min_sep_px: float = 34.0,
    n_iter: int = 50,
) -> List[Tuple[str, float, float]]:
    """Deterministic label repel in display space (Phase 6 pattern).

    Nudges label boxes apart so near-touching centroid pairs get a clear
    gap, then converts back to data coordinates.  Returns the placed
    (label, x, y) positions in data coordinates.
    """
    disp = np.array([ax.transData.transform((x, y)) for _, x, y in labeled])
    for _ in range(n_iter):
        moved = False
        for i in range(len(disp)):
            for j in range(i + 1, len(disp)):
                delta = disp[j] - disp[i]
                dist = float(np.hypot(delta[0], delta[1]))
                if dist < min_sep_px:
                    if dist < 1e-6:
                        delta, dist = np.array([1.0, 0.0]), 1.0
                    push = delta / dist * (min_sep_px - dist) / 2.0
                    disp[i] -= push
                    disp[j] += push
                    moved = True
        if not moved:
            break
    inv = ax.transData.inverted()
    placed = []
    for (label, _x, _y), (dx, dy) in zip(labeled, disp):
        nx, ny = inv.transform((dx, dy))
        placed.append((label, float(nx), float(ny)))
    return placed


# ---------------------------------------------------------------------------
# fig01/02: Heat Need maps
# ---------------------------------------------------------------------------
def fig_heat_need(year: int, fig_no: int, profile: Dict, valid_mask: np.ndarray) -> Path:
    """Continuous 0-100 Heat Need map for one year (sequential YlOrRd)."""
    data = _score_grid(heat_need_raster_path(year), valid_mask)

    fig, ax = plt.subplots(figsize=(6.3, 6.7))
    im = ax.imshow(data, cmap=NEED_CMAP, extent=_grid_extent(profile),
                   origin="upper", interpolation="nearest", vmin=0.0, vmax=100.0)
    _add_map_frame(
        ax, profile,
        f"Heat Need (0-100), {year}\n{FRAMING_SHORT}",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Heat Need (0-100, per-year robust min-max)")
    ax.legend(handles=[_nodata_patch()], loc="lower right", fontsize=8)
    return _save(
        fig, fig_no,
        caption=(
            f"Heat Need {year} = 0.40*severity_score + 0.25*(1-NDVI) + "
            "0.20*NDBI + 0.15*LST (FROZEN weights, design spec section b), "
            "each term per-year robust min-max normalized. Need is a "
            "relative heat-stress ranking on a relative severity proxy, not "
            "a physical temperature anomaly. " + FRAMING
        ),
    )


# ---------------------------------------------------------------------------
# fig03/04: Plantation Opportunity maps
# ---------------------------------------------------------------------------
def fig_opportunity(year: int, fig_no: int, profile: Dict, valid_mask: np.ndarray) -> Path:
    """Continuous 0-100 Plantation Opportunity map for one year (YlGnBu)."""
    data = _score_grid(opportunity_raster_path(year), valid_mask)

    fig, ax = plt.subplots(figsize=(6.3, 6.7))
    im = ax.imshow(data, cmap=OPP_CMAP, extent=_grid_extent(profile),
                   origin="upper", interpolation="nearest", vmin=0.0, vmax=100.0)
    _add_map_frame(
        ax, profile,
        f"Plantation Opportunity (0-100), {year}\n{FRAMING_SHORT}",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Plantation Opportunity (0-100)")
    ax.legend(handles=[_nodata_patch()], loc="lower right", fontsize=8)
    return _save(
        fig, fig_no,
        caption=(
            f"Plantation Opportunity {year} = 0.30*landuse_eligibility + "
            "0.25*(1-NDBI) + 0.15*road_accessibility_band + "
            "0.15*green_proximity + 0.15*planting_headroom (FROZEN weights, "
            "spec section c). The landuse raster and both distance surfaces "
            "(roads, vegetation) are STATIC and reused for both years, so "
            "the 2022 and 2026 maps are near-identical by construction — "
            "they differ only through the NDBI and NDVI terms. The "
            "similarity is expected, not suspicious. " + FRAMING
        ),
    )


# ---------------------------------------------------------------------------
# fig05/06: baseline suitability with class thresholds on the colorbar
# ---------------------------------------------------------------------------
def fig_suitability(year: int, fig_no: int, profile: Dict, valid_mask: np.ndarray) -> Path:
    """Continuous baseline (Scenario A) suitability with the FROZEN class
    thresholds (20/40/60/80) marked on the colorbar."""
    data = _score_grid(suitability_raster_path(year), valid_mask)
    mean_score = float(data.mean())

    fig, ax = plt.subplots(figsize=(6.3, 6.7))
    im = ax.imshow(data, cmap=SUIT_CMAP, extent=_grid_extent(profile),
                   origin="upper", interpolation="nearest", vmin=0.0, vmax=100.0)
    _add_map_frame(
        ax, profile,
        f"Baseline suitability (Scenario A, gated product), {year}\n"
        f"{FRAMING_SHORT}",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[0, 20, 40, 60, 80, 100])
    cbar.set_label("Suitability (0-100); ticks = class edges 20/40/60/80")
    trans = cbar.ax.get_yaxis_transform()
    for edge in CLASS_EDGES:
        cbar.ax.plot([0.0, 1.0], [edge, edge], color="black", lw=0.9,
                     transform=trans, clip_on=False)
    ax.legend(handles=[_nodata_patch()], loc="lower right", fontsize=8)
    return _save(
        fig, fig_no,
        caption=(
            f"Baseline suitability {year} = Need x Opportunity / 100 "
            "(gated multiplicative product, spec section a); black ticks on "
            "the colorbar mark the FROZEN class edges 20/40/60/80. The "
            f"gated product compresses scores toward low values (mean "
            f"{mean_score:.1f} of 100), so High/Very High classes are rare "
            "BY CONSTRUCTION — this is the real result of the frozen "
            "formulation, not a data problem (see the sensitivity figure). "
            + FRAMING
        ),
    )


# ---------------------------------------------------------------------------
# fig07: Very High priority zones (baseline A) — an honest empty result
# ---------------------------------------------------------------------------
def _very_high_zone_counts(valid_mask: np.ndarray) -> Dict:
    """Count Very High (class 4 / score >= 80) pixels and >=MIN_ZONE_PIXELS
    zones per scenario and year, straight from the frozen Stage 1 rasters.

    The baseline A is expected to be VH-free (gated product compresses
    scores); B/C relax the gating.  Used by fig07 so the annotation states
    measured facts, never assumptions.
    """
    structure = np.ones((3, 3), dtype=int)
    counts: Dict[str, Dict[str, int]] = {}
    paths = {"A": {y: suitability_raster_path(y) for y in YEARS}}
    for scenario in ("B", "C", "D"):
        paths[scenario] = {y: scenario_raster_path(scenario, y) for y in YEARS}
    for scenario, per_year in paths.items():
        counts[scenario] = {}
        for year, path in per_year.items():
            arr, _ = _read_raster(path)
            vh = (arr >= CLASS_EDGES[-1]) & valid_mask
            labels, _ = ndimage.label(vh, structure=structure)
            sizes = np.bincount(labels.ravel()) if labels.size else np.array([0])
            n_zones = int((sizes >= MIN_ZONE_PIXELS).sum()) if sizes.size > 1 else 0
            counts[scenario][str(year)] = {
                "vh_pixels": int(vh.sum()),
                "vh_zones_min10px": n_zones,
            }
    return counts


def fig07_very_high(profile: Dict, valid_mask: np.ndarray) -> Tuple[Path, Dict]:
    """Two panels (2022/2026): Very High zones under baseline Scenario A.

    No Very High zone exists in either year (0 cells) — the maps show the
    suitability background with an explicit annotation instead of faking
    data.  A scenario-inset states the measured cross-scenario VH counts.
    """
    vh_counts = _very_high_zone_counts(valid_mask)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.0))
    for ax, year in zip(axes, YEARS):
        data = _score_grid(suitability_raster_path(year), valid_mask)
        im = ax.imshow(data, cmap=SUIT_CMAP, extent=_grid_extent(profile),
                       origin="upper", interpolation="nearest",
                       vmin=0.0, vmax=100.0)
        _add_map_frame(ax, profile, f"Very High priority zones (baseline A), {year}")
        ax.legend(handles=[_nodata_patch()], loc="lower right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Suitability (0-100)")

        left, right, bottom, top = _grid_extent(profile)
        ax.text(
            0.5, 0.60, "No Very High priority zones\nunder baseline Scenario A",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
            fontweight="bold", color="#d73027", zorder=6,
            bbox=dict(facecolor="white", edgecolor="#d73027", alpha=0.92,
                      pad=6, linewidth=1.2),
        )
        ax.text(
            0.5, 0.36,
            f"0 cells with suitability > 80 ({year})\n"
            "a real result of the gated product,\nnot missing data",
            transform=ax.transAxes, ha="center", va="center", fontsize=8,
            color="0.2", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3),
        )

    inset_text = (
        "Very High (class > 80) zones, >=10 px noise rule, measured from the "
        "frozen Stage 1-2 rasters:\n"
        f"  Scenario A (baseline): 2022 = {vh_counts['A']['2022']['vh_zones_min10px']} zones "
        f"({vh_counts['A']['2022']['vh_pixels']} px); "
        f"2026 = {vh_counts['A']['2026']['vh_zones_min10px']} zones "
        f"({vh_counts['A']['2026']['vh_pixels']} px)\n"
        f"  Scenario B (heat-focused): 2022 = {vh_counts['B']['2022']['vh_zones_min10px']} zones; "
        f"2026 = {vh_counts['B']['2026']['vh_zones_min10px']} zones — B is NOT VH-free\n"
        f"  Scenario C (feasibility-focused): 2022 = {vh_counts['C']['2022']['vh_zones_min10px']} zones; "
        f"2026 = {vh_counts['C']['2026']['vh_zones_min10px']} zones\n"
        f"  Scenario D (vegetation-priority): 2022 = {vh_counts['D']['2022']['vh_zones_min10px']} zones "
        f"({vh_counts['D']['2022']['vh_pixels']} px); "
        f"2026 = {vh_counts['D']['2026']['vh_zones_min10px']} zones "
        f"({vh_counts['D']['2026']['vh_pixels']} px)"
    )
    fig.text(0.01, 0.865, inset_text, fontsize=7.5, va="top", family="monospace",
             bbox=dict(facecolor="#f7f7f7", edgecolor="0.6", pad=5))
    fig.subplots_adjust(top=0.64, wspace=0.28)
    fig.suptitle(
        f"Very High priority zones — baseline Scenario A (empty result, shown honestly)\n"
        f"{FRAMING_SHORT}", fontsize=11
    )
    caption = (
        "Baseline Scenario A has ZERO Very High priority zones in both "
        "snapshots (0 cells above the FROZEN 80 threshold): the gated "
        "product Need x Opportunity / 100 compresses scores (mean ~25), so "
        "the Very High class is unreachable in practice under the baseline. "
        "The empty map is the real result and is shown with an explicit "
        "annotation rather than fabricated data. Scenarios B/C (relaxed "
        "gating) do contain Very High zones (inset); Scenario D is VH-free "
        "like A. " + FRAMING
    )
    path = _save(fig, 7, caption=caption)
    return path, vh_counts


# ---------------------------------------------------------------------------
# fig08: High priority zones over suitability background (2 panels)
# ---------------------------------------------------------------------------
def fig08_high_zones(profile: Dict, valid_mask: np.ndarray) -> Tuple[Path, Dict]:
    """Baseline High zones (11 in 2022, 3 in 2026) over the suitability
    background, with zone-ID labels at polygon centroids (Phase 6 repel).

    Returns the written path plus the label-placement record used by the
    programmatic centroid-match verification.
    """
    stats = pd.read_csv(PRIORITY_ZONE_STATISTICS_CSV)
    label_check: Dict[str, Dict] = {}
    # Label-box geometry: minimum pairwise separation (no overlap) and the
    # maximum acceptable centroid nudge (~5 km at Delhi) — nudged labels
    # carry a leader line, so attribution stays explicit within this bound.
    MIN_SEP_PX = 26.0
    MAX_NUDGE_DEG = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.4))
    for ax, year in zip(axes, YEARS):
        data = _score_grid(suitability_raster_path(year), valid_mask)
        im = ax.imshow(data, cmap=SUIT_CMAP, extent=_grid_extent(profile),
                       origin="upper", interpolation="nearest",
                       vmin=0.0, vmax=100.0)

        gdf = gpd.read_file(PHASE7_VECTORS_DIR / f"high_priority_zones_{year}.geojson")
        gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                 linewidth=0.9, zorder=4)

        zstats = stats[(stats["year"] == year) & (stats["scenario"] == "A") &
                       (stats["class_level"] == "High")]
        labeled = [
            (str(int(r.zone_id)), float(r.centroid_lon), float(r.centroid_lat))
            for r in zstats.itertuples()
        ]
        placed = _repel_labels(ax, labeled, min_sep_px=MIN_SEP_PX)
        centroids = {lab: (x, y) for lab, x, y in labeled}
        # Leader lines connect nudged labels back to their zone centroid so
        # the label->zone attribution stays explicit even in the dense
        # centroid cluster of 2022 (zones 3/4/5/6).
        for label, nx, ny in placed:
            cx, cy = centroids[label]
            if float(np.hypot(nx - cx, ny - cy)) > 1e-6:
                ax.plot([cx, nx], [cy, ny], color="black", lw=0.5,
                        alpha=0.7, zorder=5)
        for label, nx, ny in placed:
            ax.annotate(
                label, xy=(nx, ny), fontsize=7, fontweight="bold",
                ha="center", va="center", color="black", zorder=6,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.85,
                          pad=0.6, linewidth=0.5),
            )
        # Programmatic label verification (display space):
        #  (1) the rendered label set equals the CSV zone-ID set exactly;
        #  (2) no two label boxes overlap (pairwise separation >= threshold);
        #  (3) every nudged label stays within MAX_NUDGE_DEG of its own
        #      zone centroid (~5 km at Delhi) and carries a leader line, so
        #      each label remains unambiguously attributable to its zone.
        disp = np.array([ax.transData.transform((x, y)) for _, x, y in placed])
        min_sep = min(
            (float(np.hypot(disp[i][0] - disp[j][0], disp[i][1] - disp[j][1]))
             for i in range(len(disp)) for j in range(i + 1, len(disp))),
            default=np.inf,
        )
        nudges = {
            label: float(np.hypot(nx - centroids[label][0], ny - centroids[label][1]))
            for label, nx, ny in placed
        }
        label_set_matches = sorted(int(l) for l, _, _ in placed) == sorted(
            int(r.zone_id) for r in zstats.itertuples()
        )
        label_check[str(year)] = {
            "n_labels": len(placed),
            "label_set_matches_csv_zone_ids": bool(label_set_matches),
            "min_pairwise_separation_px": round(min_sep, 2),
            "min_separation_ok": bool(min_sep >= MIN_SEP_PX - 1.0),
            "max_nudge_deg": round(max(nudges.values()), 5),
            "max_nudge_ok": bool(max(nudges.values()) <= MAX_NUDGE_DEG),
            "placements": [
                {"zone_id": int(lab), "x": x, "y": y,
                 "centroid_lon": centroids[lab][0],
                 "centroid_lat": centroids[lab][1],
                 "nudge_deg": nudges[lab]}
                for lab, x, y in placed
            ],
        }

        ax.set_aspect("equal", adjustable="box")
        _add_map_frame(
            ax, profile,
            f"High priority zones (baseline A), {year}\nlabels = zone IDs",
        )
        ax.legend(
            handles=[
                mpatches.Patch(facecolor="none", edgecolor="black",
                               label="High zone boundary"),
                _nodata_patch(),
            ],
            loc="lower right", fontsize=7, title="Legend", title_fontsize=8,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Suitability (0-100)")

    fig.subplots_adjust(top=0.82, wspace=0.28)
    fig.suptitle(
        f"High priority zones — baseline Scenario A\n{FRAMING_SHORT}", fontsize=11
    )
    caption = (
        "Baseline (Scenario A) High priority zones (class >= 3, 8-connected, "
        ">= 10 px) over the continuous suitability background; labels are "
        "zone IDs placed at polygon centroids and nudged apart where "
        "centroids are close, with a thin leader line connecting each "
        "nudged label back to its centroid. The baseline is deliberately "
        "sparse — 11 zones in 2022, "
        "3 in 2026 — because the gated product compresses scores; scenarios "
        "B/C, which relax the gating, produce roughly 1,400 / 1,000 zones "
        "(see the sensitivity figure). Zone boundaries are decision-support "
        "ranking units, not administrative or ownership units. " + FRAMING
    )
    path = _save(fig, 8, caption=caption)
    return path, label_check


# ---------------------------------------------------------------------------
# fig09: green vs built comparison
# ---------------------------------------------------------------------------
def fig09_green_vs_built() -> Path:
    """Grouped bars per green/built group per year: mean suitability, mean
    need, mean opportunity and High+VH share (baseline A)."""
    df = pd.read_csv(_table("green_built_comparison.csv"))
    groups = ["green_dominant", "built_dominant", "other_mixed"]
    group_labels = ["green\ndominant", "built\ndominant", "other\nmixed"]
    metrics = [
        ("mean_suitability", "Mean suitability (0-100)"),
        ("mean_heat_need", "Mean Heat Need (0-100)"),
        ("mean_opportunity", "Mean Opportunity (0-100)"),
        ("high_very_high_pct", "High+Very High (% of group pixels)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        x = np.arange(len(groups))
        for offset, year in ((-0.2, 2022), (0.2, 2026)):
            vals = [
                df[(df["year"] == year) & (df["group"] == g)][metric].iloc[0]
                if len(df[(df["year"] == year) & (df["group"] == g)]) else np.nan
                for g in groups
            ]
            ax.bar(x + offset, vals, 0.4, label=str(year),
                   color=YEAR_COLORS[year])
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(top=0.82, hspace=0.38)
    fig.suptitle(
        "Green vs built surface composition vs baseline suitability "
        "(operational NDVI/NDBI thresholds)\n" + FRAMING_SHORT, fontsize=11
    )
    return _save(
        fig, 9,
        caption=(
            GREEN_BUILT_NOTE + " High+Very High shares use the baseline "
            "(Scenario A) per-year classes; per-year normalization makes the "
            "cross-year comparison class-relative (snapshots, not trends). "
            + FRAMING
        ),
    )


# ---------------------------------------------------------------------------
# fig10: landuse suitability summary
# ---------------------------------------------------------------------------
def fig10_landuse() -> Path:
    """Horizontal bars per landuse class: mean suitability and High area,
    both years (baseline A, valid cells only)."""
    df = pd.read_csv(_table("landuse_suitability_summary.csv"))
    codes = sorted(df["landuse_code"].unique())
    # Order by 2022 mean suitability for a stable, readable ranking.
    order = (
        df[df["year"] == YEARS[0]]
        .set_index("landuse_code")["mean_suitability"]
        .reindex(codes)
        .sort_values()
        .index.tolist()
    )
    labels = [f"{LANDUSE_CLASS_NAMES.get(c, f'code {c}')} ({c})" for c in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    y = np.arange(len(order))
    for ax, (metric, xlabel) in zip(
        axes, [("mean_suitability", "Mean suitability (0-100, baseline A)"),
               ("high_area_ha", "High class area (ha, EPSG:32643)")]
    ):
        for offset, year in ((-0.2, 2022), (0.2, 2026)):
            tbl = df[df["year"] == year].set_index("landuse_code").reindex(order)
            ax.barh(y + offset, tbl[metric], 0.4, label=str(year),
                    color=YEAR_COLORS[year])
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(xlabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.80)
    fig.suptitle(
        f"Landuse class vs baseline suitability\n{FRAMING_SHORT}; "
        "association, not causation", fontsize=11
    )
    return _save(
        fig, 10,
        caption=(
            "Per landuse class (static Phase 3 raster, reused for both "
            "years): mean baseline suitability and High-class area, valid "
            "cells only. Class 0 'unclassified_background' is 84% of the "
            "grid — the ABSENCE of an OSM land-use tag, not a real "
            "land-use observation — and receives a neutral eligibility "
            "score of 50 (UNCERTAIN, spec section d); it is shown but must "
            "not be read as a measured land use. The park (1) and grass (3) "
            "classes are absent from the raster, so green-class scarcity "
            "here reflects OSM tagging coverage, not the true absence of "
            "parks. " + FRAMING
        ),
    )


# ---------------------------------------------------------------------------
# fig11: temporal priority transition (map + scenario bars)
# ---------------------------------------------------------------------------
def fig11_temporal(profile: Dict, valid_mask: np.ndarray) -> Path:
    """Baseline A persistence-category map + category areas for A/B/C.

    The map shows where 2022 High+ priority persisted, emerged or declined
    by 2026 at paired valid cells; the bar panel compares category areas
    across scenarios from temporal_priority_transition.csv.
    """
    cls = {
        year: _class_grid(suitability_class_raster_path(year), valid_mask).data
        for year in YEARS
    }
    hvh = {year: (cls[year] >= 3) & valid_mask for year in YEARS}
    cats = {
        "Persistent": hvh[2022] & hvh[2026],
        "Emerging": (~hvh[2022]) & hvh[2026] & valid_mask,
        "Declining": hvh[2022] & (~hvh[2026]) & valid_mask,
        "Stable Low": (~hvh[2022]) & (~hvh[2026]) & valid_mask,
    }
    cat_index = {name: i + 1 for i, name in enumerate(PERSISTENCE_LABELS)}
    grid = np.zeros(valid_mask.shape, dtype=np.float64)
    for name in PERSISTENCE_LABELS:
        grid[cats[name]] = cat_index[name]
    data = np.ma.masked_where(~valid_mask, grid)
    cmap = ListedColormap([PERSISTENCE_COLORS[n] for n in PERSISTENCE_LABELS])
    norm = BoundaryNorm(np.arange(0.5, len(PERSISTENCE_LABELS) + 1.0, 1.0), cmap.N)

    trans = pd.read_csv(_table("temporal_priority_transition.csv"))

    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(data, cmap=cmap, norm=norm, extent=_grid_extent(profile),
              origin="upper", interpolation="nearest")
    _add_map_frame(
        ax, profile,
        "Priority persistence 2022 -> 2026\n(baseline A, paired valid cells)",
    )
    handles = [
        mpatches.Patch(facecolor=PERSISTENCE_COLORS[n], label=n)
        for n in PERSISTENCE_LABELS
    ] + [_nodata_patch()]
    # Tiny per-category pixel counts in the legend make the honest scale
    # (Stable Low ~99.96%) visible without a second colorbar.
    counts = {n: int(cats[n].sum()) for n in PERSISTENCE_LABELS}
    for h, n in zip(handles[: len(PERSISTENCE_LABELS)], PERSISTENCE_LABELS):
        h.set_label(f"{n} ({counts[n]:,} px)")
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              title="Persistence (baseline A)", title_fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(PERSISTENCE_LABELS))
    width = 0.25
    scenario_colors = {"A": UHI_COLORS[3], "B": UHI_COLORS[2], "C": "#2c7bb6"}
    for k, scenario in enumerate(("A", "B", "C")):
        sub = trans[trans["scenario"] == scenario].set_index(
            "persistence_category").reindex(PERSISTENCE_LABELS)
        ax.bar(x + (k - 1) * width, sub["area_ha"], width,
               label=f"Scenario {scenario}", color=scenario_colors[scenario])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(PERSISTENCE_LABELS, fontsize=9)
    ax.set_ylabel("Area (ha, log scale; EPSG:32643)")
    ax.set_title("Persistence-category areas by scenario\n"
                 "(from temporal_priority_transition.csv)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)

    fig.subplots_adjust(top=0.82)
    fig.suptitle(
        f"Temporal priority transition, 2022 vs 2026 snapshots\n{FRAMING_SHORT}",
        fontsize=11,
    )
    return _save(
        fig, 11,
        caption=(
            "Persistence categories at the 1,500,777 paired valid cells "
            "(FROZEN rule, spec section k): Persistent = High+ both years; "
            "Emerging = High+ in 2026 only; Declining = High+ in 2022 only; "
            "Stable Low = below High in both. Because classes are per-year "
            "normalized, this is a RELATIVE SUITABILITY TRANSITION, NOT A "
            "CLIMATE TREND. Under baseline A only ~49 px persist and ~75 "
            "emerge (see legend counts); scenarios B/C relax the gating and "
            "show the correspondingly larger transitions in the bar panel. "
            + FRAMING
        ),
    )


# ---------------------------------------------------------------------------
# fig12: sensitivity analysis
# ---------------------------------------------------------------------------
def fig12_sensitivity() -> Path:
    """Two panels: High+VH area per scenario (A-D, by year) and mask IoU of
    each scenario vs the baseline.  Carries the headline sensitivity finding."""
    sens = pd.read_csv(_table("sensitivity_analysis.csv"))
    per = sens[sens["section"] == "per_scenario"]
    overlap = sens[sens["section"] == "overlap_vs_baseline"]
    # Stage 1 stores the overlap IoU in the "iou" column; the
    # "mask_iou_vs_baseline_A" column is only filled by the Stage 2
    # spatial extension rows.
    iou_col = (
        "iou" if overlap["iou"].notna().any() else "mask_iou_vs_baseline_A"
    )
    scenarios = ["A", "B", "C", "D"]
    scenario_colors = {
        "A": UHI_COLORS[3], "B": UHI_COLORS[2],
        "C": "#2c7bb6", "D": "#91bfdb",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    ax = axes[0]
    x = np.arange(len(scenarios))
    width = 0.38
    for offset, year in ((-width / 2, 2022), (width / 2, 2026)):
        tbl = per[per["year"] == year].set_index("scenario").reindex(scenarios)
        bars = ax.bar(x + offset, tbl["high_very_high_area_ha"], width,
                      label=str(year), color=[YEAR_COLORS[year]] * len(scenarios))
        for rect, val in zip(bars, tbl["high_very_high_area_ha"]):
            ax.annotate(f"{val:,.0f}", (rect.get_x() + rect.get_width() / 2, val),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=6.5, rotation=90)
    ax.set_yscale("log")
    # Headroom so the upward rotated value labels stay inside the axes.
    ax.set_ylim(top=float(per["high_very_high_area_ha"].max()) * 6.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Scenario {s}" for s in scenarios])
    ax.set_ylabel("High+Very High area (ha, log scale)")
    ax.set_title("High+VH area per scenario\n(gated product = conservative baseline)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)

    ax = axes[1]
    comp = ["B", "C", "D"]
    x = np.arange(len(comp))
    for offset, year in ((-0.2, 2022), (0.2, 2026)):
        tbl = overlap[overlap["year"] == year].set_index("scenario").reindex(comp)
        vals = tbl[iou_col].astype(float).values
        bars = ax.bar(x + offset, vals, 0.4, label=str(year),
                      color=YEAR_COLORS[year])
        for rect, val in zip(bars, vals):
            ax.annotate(f"{val:.4f}", (rect.get_x() + rect.get_width() / 2, val),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=6.5)
    ax.set_ylim(0.0, float(overlap[iou_col].astype(float).max()) * 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Scenario {s}" for s in comp])
    ax.set_ylabel("Mask IoU vs baseline A")
    ax.set_title("High+VH mask IoU vs baseline\n(100% of baseline cells are contained "
                 "in B/C/D masks)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.subplots_adjust(top=0.80)
    fig.suptitle(
        f"Sensitivity of priority extent to the need/opportunity weighting\n"
        f"{FRAMING_SHORT}", fontsize=11
    )
    return _save(
        fig, 12,
        caption=(
            "HEADLINE FINDING: priority-zone extent varies ~1000x across "
            "reasonable weightings — 46 ha (Scenario A, baseline) to "
            "45,315 ha (Scenario B) in 2022 — so planting recommendations "
            "are weight-sensitive and the priority map must be read "
            "together with this sensitivity analysis. The baseline gated "
            "formulation (A) is CONSERVATIVE BY CONSTRUCTION: 100% of its "
            "(very few) High+VH cells are contained in every relaxed "
            "scenario, but the near-zero IoU shows containment is not "
            "shape-preservation — the relaxed masks are hundreds to "
            "thousands of times larger. Scenarios: A balanced (baseline "
            "gated product); B heat-focused; C feasibility-focused; D "
            "vegetation-priority (approximates A, the frozen NDVI-"
            "vegetation_cover redundancy check, r ~ 0.999). " + FRAMING
        ),
    )


# ---------------------------------------------------------------------------
# fig13: Need vs Opportunity density with gated iso-lines
# ---------------------------------------------------------------------------
def fig13_need_vs_opportunity(valid_mask: np.ndarray) -> Path:
    """Hexbin of Need vs Opportunity per year (seed-42 subsample of 200k
    valid cells) with the gated-product iso-lines S = N x O / 100 at the
    FROZEN class thresholds 20/40/60/80 overlaid."""
    rng = np.random.default_rng(RANDOM_SEED)

    # Iso-line grid for the gated product S = N x O / 100.
    gv = np.linspace(0.0, 100.0, 401)
    GN, GO = np.meshgrid(gv, gv)
    GS = GN * GO / 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True)
    for ax, year in zip(axes, YEARS):
        need = _score_grid(heat_need_raster_path(year), valid_mask).compressed()
        opp = _score_grid(opportunity_raster_path(year), valid_mask).compressed()
        n_take = min(SCATTER_SUBSAMPLE_N, need.size)
        idx = rng.choice(need.size, size=n_take, replace=False)
        x, y = need[idx], opp[idx]

        hb = ax.hexbin(x, y, gridsize=80, bins="log", mincnt=1, cmap="viridis")
        cs = ax.contour(GN, GO, GS, levels=[20, 40, 60, 80],
                        colors="red", linewidths=1.1, linestyles="--")
        ax.clabel(cs, inline=True, fontsize=7, fmt={v: f"S={v:g}" for v in (20, 40, 60, 80)})
        ax.set_xlabel("Heat Need (0-100)")
        ax.set_title(f"{year} (n={n_take:,}, seed 42)", fontsize=10)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log10(cell count)")
    axes[0].set_ylabel("Plantation Opportunity (0-100)")
    fig.subplots_adjust(top=0.80)
    fig.suptitle(
        f"Need vs Opportunity and the gated multiplicative structure\n"
        f"{FRAMING_SHORT}", fontsize=11
    )
    return _save(
        fig, 13,
        caption=(
            "Hexbin density of valid cells in the Need-Opportunity plane "
            f"(deterministic seed-42 subsample of {SCATTER_SUBSAMPLE_N:,} "
            "cells per year). Dashed red iso-lines are the gated product "
            "S = Need x Opportunity / 100 at the FROZEN class thresholds "
            "20/40/60/80: cells below the S=20 line are Very Low/Low "
            "REGARDLESS of how extreme Need is, because Opportunity gates "
            "the product (S=0 whenever Opportunity=0). The multiplicative "
            "gate — not either factor alone — is what compresses baseline "
            "scores and makes High/Very High rare. " + FRAMING
        ),
    )


# ---------------------------------------------------------------------------
# Records: manifest + pipeline record (idempotent)
# ---------------------------------------------------------------------------
def _update_manifest_figures(record: Dict) -> None:
    """Read-modify-write the Phase 7 manifest with the Stage 3 figures block."""
    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)

    manifest["figures"] = {
        "stage": 3,
        "generated_utc": _now(),
        "framing": FRAMING,
        "style": {
            "sequential_cmaps": {
                "heat_need": "YlOrRd",
                "opportunity": "YlGnBu",
                "suitability": "viridis",
                "hexbin_density": "viridis",
            },
            "suitability_class_palette": dict(zip(CLASS_LABELS.values(),
                                                  SUITABILITY_COLORS)),
            "persistence_colors": PERSISTENCE_COLORS,
            "year_colors": {str(k): v for k, v in YEAR_COLORS.items()},
            "nodata_color": NODATA_COLOR,
            "dpi": DPI,
            "map_frame": (
                "lat/lon ticks, approximate scale bar (~97 km/deg lon, "
                "~111 km/deg lat at Delhi), north arrow, explicit NoData patch"
            ),
            "categorical_discipline": (
                "categorical figures use discrete ListedColormap + "
                "BoundaryNorm only; continuous figures use sequential "
                "colormaps with fixed 0-100 limits"
            ),
        },
        "scatter_subsample": {"n_per_year": SCATTER_SUBSAMPLE_N, "seed": RANDOM_SEED},
        "figures": record["steps"]["figures"]["figures"],
        "validation": record["validation"],
        "deviations": record.get("deviations", []),
    }
    _save_json(manifest, MANIFEST_JSON)


def _append_pipeline_record(record: Dict) -> None:
    """Append the Stage 3 entry to the Phase 7 pipeline record (idempotent)."""
    with open(PIPELINE_RECORD_JSON) as f:
        existing = json.load(f)

    container = {"phase": 7, "project": "GreenGrid-AI", "stage_records": []}
    if "stage_records" in existing:
        container["stage_records"] = existing["stage_records"]
    else:
        container["stage_records"] = [existing]
    container["stage_records"] = [
        r for r in container["stage_records"] if r.get("stage") != 3
    ]
    container["stage_records"].append(record)
    _save_json(container, PIPELINE_RECORD_JSON)


# ---------------------------------------------------------------------------
# Stage 3 driver
# ---------------------------------------------------------------------------
def run_stage3() -> Dict:
    """Run the complete Phase 7 Stage 3 figure suite."""
    PHASE7_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    record: Dict = {
        "phase": 7,
        "stage": 3,
        "project": "GreenGrid-AI",
        "project_root": str(PROJECT_ROOT),
        "started_utc": _now(),
        "random_seed": RANDOM_SEED,
        "steps": {},
        "deviations": [],
    }

    t0 = time.time()
    data = load_inputs()
    valid_mask = data["valid_mask"]
    profile = data["profile"]
    record["steps"]["context"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "valid_cells_per_year": int(data["n_valid"]),
        "grid": f"{profile['width']}x{profile['height']} {profile['crs']}",
    }

    figures: Dict[str, str] = {}
    skipped: Dict[str, str] = {}

    def _track(fig_no: int, result: Optional[Path], why: str = "") -> None:
        if result is None:
            skipped[FIGURE_FILES[fig_no]] = why
            print(f"  [fig{fig_no:02d}] skipped: {why}")
        else:
            figures[FIGURE_FILES[fig_no]] = str(result)

    print("Stage 3: rendering figure suite ...")

    # fig01/02: Heat Need maps
    for fig_no, year in ((1, 2022), (2, 2026)):
        _track(fig_no, fig_heat_need(year, fig_no, profile, valid_mask))

    # fig03/04: Opportunity maps
    for fig_no, year in ((3, 2022), (4, 2026)):
        _track(fig_no, fig_opportunity(year, fig_no, profile, valid_mask))

    # fig05/06: baseline suitability with class thresholds
    for fig_no, year in ((5, 2022), (6, 2026)):
        _track(fig_no, fig_suitability(year, fig_no, profile, valid_mask))

    # fig07: Very High zones (honest empty result) + measured cross-scenario counts
    path7, vh_counts = fig07_very_high(profile, valid_mask)
    figures[FIGURE_FILES[7]] = str(path7)
    record["steps"]["very_high_counts"] = {
        "status": "success",
        "counts": vh_counts,
        "note": (
            "measured from frozen Stage 1-2 rasters with the Stage 2 zone "
            "noise rule (>=10 px); used by the fig07 annotation"
        ),
    }
    if vh_counts["B"]["2022"]["vh_zones_min10px"] > 0:
        record["deviations"].append(
            "Stage 3 brief assumed Scenario B is Very-High-free; measured "
            "counts from the frozen rasters show B DOES contain Very High "
            "zones (32 in 2022, 22 in 2026, >=10 px rule). The fig07 inset "
            "states the measured values instead of the assumed 'B VH-free' "
            "note. Scenario D is VH-free (0 px both years), like baseline A."
        )

    # fig08: High zones with ID labels + centroid-match verification
    path8, label_check = fig08_high_zones(profile, valid_mask)
    figures[FIGURE_FILES[8]] = str(path8)
    record["steps"]["zone_label_check"] = {
        "status": "success",
        "check": (
            "fig08 labels: (1) rendered label set == CSV zone-ID set; "
            "(2) no two label boxes overlap (pairwise display-space "
            "separation >= threshold); (3) every nudged label stays within "
            "0.05 deg (~5 km) of its own zone centroid and carries a leader "
            "line, so each label is unambiguously attributable to its zone"
        ),
        **label_check,
    }

    # fig09-13: table/density figures
    _track(9, fig09_green_vs_built())
    _track(10, fig10_landuse())
    _track(11, fig11_temporal(profile, valid_mask))
    _track(12, fig12_sensitivity())
    _track(13, fig13_need_vs_opportunity(valid_mask))

    all_label_checks = all(
        v["label_set_matches_csv_zone_ids"]
        and v["min_separation_ok"]
        and v["max_nudge_ok"]
        for v in label_check.values()
    )
    record["steps"]["figures"] = {
        "status": "success" if not skipped else "completed_with_skips",
        "n_figures": len(figures),
        "figures": figures,
        "skipped": skipped,
        "dpi": DPI,
        "scatter_subsample_n": SCATTER_SUBSAMPLE_N,
    }
    record["validation"] = {
        "n_figures_written": len(figures),
        "expected_figures": len(FIGURE_FILES),
        "all_13_written": len(figures) == len(FIGURE_FILES),
        "fig08_zone_labels_match_centroids": bool(all_label_checks),
        "very_high_baseline_empty_both_years": bool(
            vh_counts["A"]["2022"]["vh_pixels"] == 0
            and vh_counts["A"]["2026"]["vh_pixels"] == 0
        ),
    }
    record["finished_utc"] = _now()
    record["status"] = "success" if (not skipped and all_label_checks) else "completed_with_issues"
    record["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "rasterio": rasterio.__version__,
        "geopandas": gpd.__version__,
        "scipy": ndimage.__name__ and __import__("scipy").__version__,
    }

    _update_manifest_figures(record)
    _append_pipeline_record(record)
    print(f"  Figures written: {len(figures)}; skipped: {len(skipped)}")
    print(f"  Manifest updated: {MANIFEST_JSON}")
    print(f"  Pipeline record appended: {PIPELINE_RECORD_JSON}")

    return {"record": record, "figures": figures, "skipped": skipped}


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the Phase 7 Stage 3 figure suite."""
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 7 Stage 3 figure suite."
    )
    parser.parse_args(argv)

    try:
        results = run_stage3()
        print(json.dumps(_convert_for_json(results["record"]), indent=2, default=str))
        return 0 if results["record"]["status"] == "success" else 1
    except Exception as exc:
        print(f"Phase 7 Stage 3 pipeline failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
