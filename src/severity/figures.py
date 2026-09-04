"""Phase 6 Stage 4: publication-quality figure suite.

Renders the 16 report figures from the frozen Stage 1-3 products
(full-grid rasters, hotspot boundaries, analytics tables).  Every map and
panel carries the scientific framing: ML-based RELATIVE heat severity /
operational UHI hotspot proxy — NOT physical UHI intensity; two snapshot
dates (2022, 2026), not a long-term trend; association, not causation.

Conventions (match the rest of ``severity``):

* Severity palette identical to Phase 5 ``UHI_COLORS`` (Low #2c7bb6,
  Moderate #abd9e9, High #fdae61, Severe #d7191c); sequential colormaps
  for continuous LST/confidence.
* Every map shows NoData explicitly (light grey), lat/lon ticks, an
  approximate scale bar (EPSG:4326 degrees -> km at Delhi, ~111 km/deg
  lat / ~97 km/deg lon) and a north arrow.
* Figures whose inputs are missing are SKIPPED and reported, never
  replaced with filler charts.
* Deterministic seed-42 subsampling for the NDVI/NDBI scatter figures.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.features import rasterize as rasterio_rasterize

from models.config import RANDOM_SEED
from models.maps import UHI_COLORS

from .analytics import BOUNDARY_BAND_LABELS, uncertainty_zone_path
from .config import (
    INPUT_DATASET_CSV,
    MANIFEST_JSON,
    PHASE6_FIGURES_DIR,
    PHASE6_TABLES_DIR,
    PIPELINE_RECORD_JSON,
    PROJECT_ROOT,
    SEVERITY_CLASS_NAMES,
    SEVERITY_NODATA,
    confidence_raster_path,
    hotspot_boundaries_path,
    lst_raster_path,
    severity_raster_path,
)
from .fullgrid import load_valid_cells

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
YEARS = (2022, 2026)
CLASS_LABELS = [SEVERITY_CLASS_NAMES[k] for k in range(4)]  # Low..Severe
SEVERITY_CMAP = ListedColormap(UHI_COLORS)
NODATA_COLOR = "#d9d9d9"  # light grey, explicit NoData
LST_CMAP = plt.get_cmap("inferno").copy()
LST_CMAP.set_bad(NODATA_COLOR)
CONF_CMAP = plt.get_cmap("cividis").copy()  # colorblind-safe sequential
CONF_CMAP.set_bad(NODATA_COLOR)
DPI = 220

FRAMING = (
    "ML-based RELATIVE heat severity (operational UHI hotspot proxy) — "
    "NOT physical UHI intensity"
)
TEMPORAL_FRAMING = (
    "Two snapshot dates (2022 / 2026 composites), not a long-term trend; "
    "severity classes are per-year LST quartiles (class-relative comparison only)"
)
ASSOCIATION_FRAMING = "Association only, not causation"
OOF_NOTE = (
    "TRUE out-of-fold predictions of the frozen Phase 5 baseline Random "
    "Forest at sampled cells only; NOT the Phase 6 RF-C production model"
)
# Short framing clause for single-map titles (full framing is always in the
# figure caption; a full-width title line would overflow the tight bbox).
FRAMING_SHORT = "ML-based RELATIVE heat severity (operational UHI hotspot proxy)"

# Approximate km per degree at Delhi (EPSG:4326); the scale bar is labelled
# 'approximate' on every map that carries one.
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 97.0

SCATTER_SUBSAMPLE_N = 100_000

FIGURE_FILES = {
    1: "fig01_lst_2022.png",
    2: "fig02_lst_2026.png",
    3: "fig03_severity_2022.png",
    4: "fig04_severity_2026.png",
    5: "fig05_hotspots_2022.png",
    6: "fig06_hotspots_2026.png",
    7: "fig07_temporal_comparison.png",
    8: "fig08_green_vs_built.png",
    9: "fig09_ndvi_vs_lst.png",
    10: "fig10_ndbi_vs_lst.png",
    11: "fig11_high_severe_by_block.png",
    12: "fig12_hotspot_size_distribution.png",
    13: "fig13_confidence_map.png",
    14: "fig14_uncertainty.png",
    15: "fig15_transition_matrix.png",
    16: "fig16_landuse_heat.png",
}


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


def _read_raster(path: Path) -> Tuple[np.ndarray, Dict]:
    """Read one single-band raster as (array, profile)."""
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


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
    Delhi and is explicitly labelled 'approximate'.
    """
    left, right, bottom, top = _grid_extent(profile)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")

    # Approximate scale bar (lower-left inside the axes).
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

    # Simple north arrow (upper-right).
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


def _severity_patches() -> List[mpatches.Patch]:
    """Legend patches for the four severity classes."""
    return [
        mpatches.Patch(color=UHI_COLORS[i], label=CLASS_LABELS[i])
        for i in range(4)
    ]


def _nodata_patch() -> mpatches.Patch:
    return mpatches.Patch(facecolor=NODATA_COLOR, edgecolor="0.6", label="NoData")


def _save(fig: plt.Figure, fig_no: int, caption: Optional[str] = None) -> Path:
    """Save a figure to its canonical PNG path and close it.

    Captions are wrapped so the tight bbox never stretches the canvas.
    """
    if caption:
        import textwrap

        wrapped = "\n".join(textwrap.wrap(caption, width=110))
        # Reserve bottom space so the caption never overlaps axis labels.
        fig.subplots_adjust(bottom=0.16)
        fig.text(0.01, 0.005, wrapped, fontsize=7, color="0.25",
                 ha="left", va="bottom")
    out_path = PHASE6_FIGURES_DIR / FIGURE_FILES[fig_no]
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  [fig{fig_no:02d}] {out_path.name}")
    return out_path


def _table(name: str) -> Path:
    return PHASE6_TABLES_DIR / name


# ---------------------------------------------------------------------------
# fig01/02: observed LST heat-intensity maps
# ---------------------------------------------------------------------------
def fig_lst(year: int, fig_no: int, profile: Dict) -> Path:
    """Observed LST map for one year (continuous deg-C, inferno)."""
    arr, _ = _read_raster(lst_raster_path(year))
    data = np.ma.masked_where(~np.isfinite(arr) | (arr == -1.0), arr)

    fig, ax = plt.subplots(figsize=(6.3, 6.7))
    im = ax.imshow(data, cmap=LST_CMAP, extent=_grid_extent(profile),
                   origin="upper", interpolation="nearest")
    _add_map_frame(
        ax, profile,
        f"Observed LST (deg C), {year}\n{FRAMING_SHORT}",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("LST (deg C)")
    ax.legend(handles=[_nodata_patch()], loc="lower right", fontsize=8)
    return _save(
        fig, fig_no,
        caption=(
            f"Observed LST {year}; nodata grey. {TEMPORAL_FRAMING}. "
            "LST is the observed surface temperature input, not a UHI anomaly."
        ),
    )


# ---------------------------------------------------------------------------
# fig03/04: categorical severity maps
# ---------------------------------------------------------------------------
def fig_severity(year: int, fig_no: int, profile: Dict) -> Path:
    """Categorical severity map for one year (Phase 5 palette)."""
    arr, _ = _read_raster(severity_raster_path(year))
    data = np.ma.masked_where(arr == SEVERITY_NODATA, arr)
    cmap = SEVERITY_CMAP.copy()
    cmap.set_bad(NODATA_COLOR)
    norm = BoundaryNorm(np.arange(-0.5, 4.0, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(6.3, 6.7))
    ax.imshow(data, cmap=cmap, norm=norm, extent=_grid_extent(profile),
              origin="upper", interpolation="nearest")
    _add_map_frame(
        ax, profile,
        f"ML-based heat-severity classes (per-year LST quartiles), {year}\n"
        f"{FRAMING_SHORT}",
    )
    ax.legend(
        handles=[*_severity_patches(), _nodata_patch()],
        loc="lower right", fontsize=8, title="Severity", title_fontsize=8,
    )
    return _save(
        fig, fig_no,
        caption=(
            f"Severity classes {year}: per-year LST quartiles "
            "(Low/Moderate/High/Severe); nodata grey. " + TEMPORAL_FRAMING
        ),
    )


# ---------------------------------------------------------------------------
# fig05/06: hotspot boundaries over severity background
# ---------------------------------------------------------------------------
def fig_hotspots(year: int, fig_no: int, profile: Dict) -> Path:
    """defA hotspot boundaries (polygons) over the severity background.

    Labels the top-10 hotspots by rank_by_high_severe_area (the Stage 2
    primary ranking) at their polygon centroids.
    """
    arr, _ = _read_raster(severity_raster_path(year))
    data = np.ma.masked_where(arr == SEVERITY_NODATA, arr)
    cmap = SEVERITY_CMAP.copy()
    cmap.set_bad(NODATA_COLOR)
    norm = BoundaryNorm(np.arange(-0.5, 4.0, 1.0), cmap.N)

    gdf = gpd.read_file(hotspot_boundaries_path("A", year))
    stats = pd.read_csv(_table(f"hotspot_statistics_defA_{year}.csv"))
    rank_col = "rank_by_high_severe_area"
    top10 = (
        stats.nsmallest(min(10, len(stats)), rank_col)["hotspot_id"]
        .tolist()
    )
    rank_map = dict(zip(stats["hotspot_id"], stats[rank_col]))

    fig, ax = plt.subplots(figsize=(6.3, 6.7))
    ax.imshow(data, cmap=cmap, norm=norm, extent=_grid_extent(profile),
              origin="upper", interpolation="nearest")
    gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.7, zorder=4)

    # Minimal deterministic label repel: nudge top-10 label boxes apart in
    # display space so near-touching pairs (e.g. #9/#10, #2/#8) get a clear
    # gap, then convert back to data coordinates for annotation.
    MIN_SEP_PX = 34.0
    labeled = [
        (int(row["hotspot_id"]), row.geometry.centroid.x, row.geometry.centroid.y)
        for _, row in gdf.iterrows()
        if int(row["hotspot_id"]) in top10
    ]
    disp = np.array([ax.transData.transform((x, y)) for _, x, y in labeled])
    for _ in range(50):
        moved = False
        for i in range(len(disp)):
            for j in range(i + 1, len(disp)):
                delta = disp[j] - disp[i]
                dist = float(np.hypot(delta[0], delta[1]))
                if dist < MIN_SEP_PX:
                    if dist < 1e-6:
                        delta, dist = np.array([1.0, 0.0]), 1.0
                    push = delta / dist * (MIN_SEP_PX - dist) / 2.0
                    disp[i] -= push
                    disp[j] += push
                    moved = True
        if not moved:
            break
    inv = ax.transData.inverted()
    for (hid, _x, _y), (dx, dy) in zip(labeled, disp):
        nx, ny = inv.transform((dx, dy))
        ax.annotate(
            f"#{rank_map[hid]}", xy=(nx, ny), fontsize=6, fontweight="bold",
            ha="center", va="center", color="black", zorder=6,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.85,
                      pad=0.6, linewidth=0.5),
        )
    # gdf.plot resets the axes aspect; restore equal degree scaling so the
    # overlay matches the severity background exactly.
    ax.set_aspect("equal", adjustable="box")
    _add_map_frame(
        ax, profile,
        f"Operational UHI hotspots (definition A: severity High/Severe), {year}\n"
        f"{FRAMING_SHORT}; labels = top-10 by high+severe area",
    )
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="none", edgecolor="black",
                           label="defA hotspot boundary"),
            *_severity_patches(), _nodata_patch(),
        ],
        loc="lower right", fontsize=7, title="Legend", title_fontsize=8,
    )
    return _save(
        fig, fig_no,
        caption=(
            f"defA hotspots {year} (severity in {{High, Severe}}, clusters "
            ">= 10 px) over severity background; labels show the top-10 by "
            f"high+severe area. {FRAMING}. " + TEMPORAL_FRAMING
        ),
    )


# ---------------------------------------------------------------------------
# fig07: 2x2 temporal comparison
# ---------------------------------------------------------------------------
def fig07_temporal_comparison(profile: Dict) -> Optional[Path]:
    """2x2 panel: severity maps both years + block-level + class bars."""
    paths_needed = [
        _table(f"area_statistics_{y}.csv") for y in YEARS
    ] + [severity_raster_path(y) for y in YEARS]
    if not all(p.exists() for p in paths_needed):
        print("  [fig07] skipped: missing severity rasters or area_statistics CSVs")
        return None

    area = {y: pd.read_csv(_table(f"area_statistics_{y}.csv")) for y in YEARS}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
    norm = BoundaryNorm(np.arange(-0.5, 4.0, 1.0), SEVERITY_CMAP.N)
    for col, year in enumerate(YEARS):
        arr, _ = _read_raster(severity_raster_path(year))
        data = np.ma.masked_where(arr == SEVERITY_NODATA, arr)
        cmap = SEVERITY_CMAP.copy()
        cmap.set_bad(NODATA_COLOR)
        ax = axes[0, col]
        ax.imshow(data, cmap=cmap, norm=norm, extent=_grid_extent(profile),
                  origin="upper", interpolation="nearest")
        _add_map_frame(ax, profile, f"Severity classes, {year}")
        if col == 0:
            ax.legend(handles=[*_severity_patches(), _nodata_patch()],
                      loc="lower right", fontsize=6)

    # Panel 3: High+Severe percent per spatial block, 2022 vs 2026.
    ax = axes[1, 0]
    blocks = sorted(set(area[2022]["block_id"]) | set(area[2026]["block_id"]))
    x = np.arange(len(blocks))
    width = 0.4
    for offset, year, color in ((-width / 2, 2022, UHI_COLORS[2]),
                                (width / 2, 2026, UHI_COLORS[3])):
        tbl = area[year].set_index("block_id").reindex(blocks)
        ax.bar(x + offset, tbl["high_severe_percent"], width,
               label=str(year), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{b}" for b in blocks], rotation=60, fontsize=7)
    ax.set_ylabel("High+Severe (% of valid pixels)")
    ax.set_title("High+Severe share per spatial block\n"
                 "(5x5 sampling-design blocks, not administrative sectors)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: class distribution comparison (grouped bars).
    ax = axes[1, 1]
    x = np.arange(4)
    for offset, year, color in ((-0.2, 2022, UHI_COLORS[2]),
                                (0.2, 2026, UHI_COLORS[3])):
        tbl = area[year]
        tot = tbl[[f"{c.lower()}_count" for c in CLASS_LABELS]].sum()
        pct = [100.0 * tot[f"{c.lower()}_count"] / tot.sum() for c in CLASS_LABELS]
        ax.bar(x + offset, pct, 0.4, label=str(year), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylabel("% of valid pixels")
    ax.set_title("Severity class distribution (class-relative)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Temporal comparison, 2022 vs 2026 snapshots — {FRAMING}", fontsize=11
    )
    # Extra vertical room so the bottom-row panel titles cannot collide with
    # the top-row maps' x-axis labels (e.g. "Longitude (deg)").
    fig.subplots_adjust(hspace=0.34)
    return _save(
        fig, 7,
        caption=(
            "Classes are per-year LST quartiles: cross-year comparison is "
            "class-relative only, NOT a climate trend. Two snapshot dates, "
            "not a long-term trend."
        ),
    )


# ---------------------------------------------------------------------------
# fig08: green vs built
# ---------------------------------------------------------------------------
def fig08_green_vs_built() -> Optional[Path]:
    """Grouped bars: LST / high_severe % / NDVI for green vs built vs mixed."""
    path = _table("green_built_comparison.csv")
    if not path.exists():
        print("  [fig08] skipped: missing green_built_comparison.csv")
        return None
    df = pd.read_csv(path)

    groups = ["green_dominant", "built_dominant", "other_mixed"]
    group_labels = ["green\ndominant", "built\ndominant", "other\nmixed"]
    metrics = [
        ("mean_lst_C", "Mean LST (deg C)"),
        ("median_lst_C", "Median LST (deg C)"),
        ("high_severe_percent", "High+Severe (%)"),
        ("mean_ndvi", "Mean NDVI"),
    ]
    colors = {2022: UHI_COLORS[2], 2026: UHI_COLORS[3]}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        x = np.arange(len(groups))
        for offset, year in ((-0.2, 2022), (0.2, 2026)):
            vals = [
                df[(df["year"] == year) & (df["group"] == g)][metric].iloc[0]
                if len(df[(df["year"] == year) & (df["group"] == g)]) else np.nan
                for g in groups
            ]
            ax.bar(x + offset, vals, 0.4, label=str(year), color=colors[year])
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Surface composition vs heat severity (operational NDVI/NDBI "
        "thresholds) — " + FRAMING, fontsize=11
    )
    return _save(
        fig, 8,
        caption=(
            "Green/built grouping uses conventional operational thresholds "
            "(NDVI >= 0.3 green; NDBI >= 0.1 and NDVI < 0.3 built), not "
            "site-calibrated. " + ASSOCIATION_FRAMING + "."
        ),
    )


# ---------------------------------------------------------------------------
# fig09/10: NDVI/NDBI vs LST scatter
# ---------------------------------------------------------------------------
def fig_scatter(
    variable: str, fig_no: int, xlabel: str, expectation: str
) -> Optional[Path]:
    """Seed-42 subsampled scatter + binned mean line per year, 2 panels.

    Parameters
    ----------
    variable : str
        Column name in the Phase 4 sampled dataset ("ndvi" or "ndbi").
    expectation : str
        One-line expectation note for the caption.
    """
    if not INPUT_DATASET_CSV.exists():
        print(f"  [fig{fig_no:02d}] skipped: missing {INPUT_DATASET_CSV.name}")
        return None

    df = pd.read_csv(INPUT_DATASET_CSV, usecols=["year", "lst_C", variable])
    rng = np.random.default_rng(RANDOM_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    r_values: Dict[int, float] = {}
    for ax, year in zip(axes, YEARS):
        sub = df[df["year"] == year]
        n_take = min(SCATTER_SUBSAMPLE_N, len(sub))
        idx = rng.choice(len(sub), size=n_take, replace=False)
        sub = sub.iloc[idx]
        x = sub[variable].values.astype(np.float64)
        y = sub["lst_C"].values.astype(np.float64)
        ax.scatter(x, y, s=1, alpha=0.08, color=UHI_COLORS[0], rasterized=True)

        # Binned mean line (20 quantile bins).
        edges = np.quantile(x, np.linspace(0, 1, 21))
        edges[0], edges[-1] = -np.inf, np.inf
        bin_id = np.clip(np.digitize(x, edges[1:-1]), 0, 19)
        centers, means = [], []
        for b in range(20):
            sel = bin_id == b
            if sel.sum() >= 10:
                centers.append(float(np.mean(x[sel])))
                means.append(float(np.mean(y[sel])))
        ax.plot(centers, means, color=UHI_COLORS[3], lw=2, label="binned mean (20 quantile bins)")

        pearson_r = float(np.corrcoef(x, y)[0, 1])
        r_values[year] = pearson_r
        ax.plot([], [], " ", label=f"Pearson r = {pearson_r:.3f} (n={n_take:,})")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{year}", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("LST (deg C)")
    fig.subplots_adjust(top=0.80)
    fig.suptitle(
        f"{xlabel} vs observed LST (seed-42 subsample, {SCATTER_SUBSAMPLE_N:,}/yr)\n"
        f"{FRAMING_SHORT}; {ASSOCIATION_FRAMING}", fontsize=11
    )
    return _save(
        fig, fig_no,
        caption=(
            f"{expectation}. Seed-42 subsample of {SCATTER_SUBSAMPLE_N:,} "
            "sampled pixels per year; spatially autocorrelated pixels are NOT "
            "independent observations, so n overstates the effective sample "
            "size. " + ASSOCIATION_FRAMING + "."
        ),
    )


# ---------------------------------------------------------------------------
# fig11: high+severe percent per spatial block
# ---------------------------------------------------------------------------
def fig11_high_severe_by_block() -> Optional[Path]:
    """Grouped bar of high_severe_percent per spatial block, 2022 vs 2026."""
    paths = [_table(f"area_statistics_{y}.csv") for y in YEARS]
    if not all(p.exists() for p in paths):
        print("  [fig11] skipped: missing area_statistics CSVs")
        return None
    area = {y: pd.read_csv(p) for y, p in zip(YEARS, paths)}

    blocks = sorted(set(area[2022]["block_id"]) | set(area[2026]["block_id"]))
    x = np.arange(len(blocks))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for offset, year, color in ((-width / 2, 2022, UHI_COLORS[2]),
                                (width / 2, 2026, UHI_COLORS[3])):
        tbl = area[year].set_index("block_id").reindex(blocks)
        ax.bar(x + offset, tbl["high_severe_percent"], width,
               label=str(year), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"block {b}" for b in blocks], rotation=60, fontsize=8)
    ax.set_ylabel("High+Severe (% of valid pixels)")
    ax.set_title(
        "High+Severe heat-severity share per spatial block, 2022 vs 2026\n"
        f"{FRAMING}", fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _save(
        fig, 11,
        caption=(
            "Blocks are 5x5 sampling-design grid blocks from the Phase 5 "
            "sampling design (Phase 4 sampled row/col bins), NOT "
            "administrative sectors. Per-year quartile classes: the "
            "comparison is class-relative, not a climate trend."
        ),
    )


# ---------------------------------------------------------------------------
# fig12: hotspot size distribution + sensitivity inset
# ---------------------------------------------------------------------------
def fig12_hotspot_sizes() -> Optional[Path]:
    """defA hotspot area histogram both years + defA/B/C sensitivity inset."""
    paths = [
        _table(f"hotspot_statistics_defA_{y}.csv") for y in YEARS
    ] + [_table("hotspot_sensitivity.csv")]
    if not all(p.exists() for p in paths):
        print("  [fig12] skipped: missing hotspot statistics / sensitivity CSVs")
        return None
    stats = {y: pd.read_csv(_table(f"hotspot_statistics_defA_{y}.csv")) for y in YEARS}
    sens = pd.read_csv(_table("hotspot_sensitivity.csv"))

    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = {2022: UHI_COLORS[2], 2026: UHI_COLORS[3]}
    bins = np.logspace(0, np.log10(max(stats[2022]["area_ha"].max(),
                                       stats[2026]["area_ha"].max()) * 1.2), 31)
    for year in YEARS:
        ax.hist(stats[year]["area_ha"], bins=bins, histtype="step",
                lw=2, color=colors[year], label=f"{year} (n={len(stats[year])})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hotspot area (ha, EPSG:32643 equal-area)")
    ax.set_ylabel("Number of hotspots (log scale)")
    ax.set_title(
        "defA hotspot size distribution — " + FRAMING, fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # Sensitivity summary inset: n hotspots + total area per definition.
    # Placed in the sparse lower-right of the log-log histogram, clear of
    # the upper-right legend.
    inset = ax.inset_axes([0.42, 0.10, 0.56, 0.40])
    inset.axis("off")
    per_def = sens[sens["section"] == "per_definition"]
    lines = ["Definition sensitivity:", ""]
    for definition in ("A", "B", "C"):
        sub = per_def[per_def["definition"] == definition]
        parts = []
        for year in YEARS:
            row = sub[sub["year"] == year]
            if len(row):
                parts.append(
                    f"{year}: n={int(row['n_hotspots'].iloc[0])}, "
                    f"{row['total_area_ha'].iloc[0]:,.0f} ha"
                )
        lines.append(f"  def {definition}:  " + " | ".join(parts))
    lines += [
        "",
        "A: severity High/Severe (primary)",
        "B: Severe only",
        "C: A + confidence >= 0.60 (operational,",
        "    uncalibrated)",
    ]
    inset.text(0, 1, "\n".join(lines), fontsize=7.5, va="top", family="monospace")
    return _save(
        fig, 12,
        caption=(
            "Areas computed in EPSG:32643 (UTM 43N, Delhi); degrees are not "
            "equal-area. Clusters below 10 connected pixels removed as "
            f"operational noise. {FRAMING}."
        ),
    )


# ---------------------------------------------------------------------------
# fig13: confidence maps
# ---------------------------------------------------------------------------
def fig13_confidence(profile: Dict) -> Optional[Path]:
    """Confidence maps (max class probability) for both years, 2 panels."""
    if not all(confidence_raster_path(y).exists() for y in YEARS):
        print("  [fig13] skipped: missing confidence rasters")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2))
    for ax, year in zip(axes, YEARS):
        arr, _ = _read_raster(confidence_raster_path(year))
        data = np.ma.masked_where(~np.isfinite(arr) | (arr == -1.0), arr)
        im = ax.imshow(data, cmap=CONF_CMAP, extent=_grid_extent(profile),
                       origin="upper", interpolation="nearest", vmin=0.25, vmax=1.0)
        _add_map_frame(ax, profile, f"Confidence (max class probability), {year}")
        ax.legend(handles=[_nodata_patch()], loc="lower right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Model confidence maps — {FRAMING}", fontsize=11
    )
    return _save(
        fig, 13,
        caption=(
            "Confidence is the maximum per-class probability of the Phase 6 "
            "RF-C model and is an UNCALIBRATED confidence proxy (per the "
            "Phase 5 calibration report); it is not a calibrated probability "
            "statement. Nodata grey."
        ),
    )


# ---------------------------------------------------------------------------
# fig14: uncertainty zones + boundary-band OOF error rates
# ---------------------------------------------------------------------------
def fig14_uncertainty(profile: Dict) -> Optional[Path]:
    """2 uncertainty-zone panels + boundary-band OOF error-rate panel."""
    paths = (
        [uncertainty_zone_path(y) for y in YEARS]
        + [_table("error_diagnostics_summary.csv")]
    )
    if not all(p.exists() for p in paths):
        print("  [fig14] skipped: missing uncertainty rasters or error diagnostics")
        return None

    err = pd.read_csv(_table("error_diagnostics_summary.csv"))
    band_rows = err[err["analysis"] == "boundary_distance_band_error_rate"]

    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    for ax, year in zip(axes, YEARS):
        arr, _ = _read_raster(uncertainty_zone_path(year))
        # 255 = nodata, 1 = uncertain, 0 = certain
        data = np.ma.masked_where(arr == 255, arr)
        cmap = ListedColormap(["#f0f0f0", "#e6550d"])
        cmap.set_bad(NODATA_COLOR)
        ax.imshow(data, cmap=cmap, extent=_grid_extent(profile),
                  origin="upper", interpolation="nearest", vmin=0, vmax=1)
        _add_map_frame(ax, profile, f"Uncertainty zone, {year}")
        ax.legend(
            handles=[
                mpatches.Patch(facecolor="#e6550d",
                               label="uncertain (top-2 margin < 0.10 OR conf < 0.50)"),
                mpatches.Patch(facecolor="#f0f0f0", edgecolor="0.6",
                               label="certain"),
                _nodata_patch(),
            ],
            loc="lower right", fontsize=6,
        )

    # Third panel: boundary-band OOF error rates (Moderate<->High transition).
    ax = fig.add_subplot(gs[0, 2])
    labels = list(BOUNDARY_BAND_LABELS)
    x = np.arange(len(labels))
    for offset, year, color in ((-0.2, 2022, UHI_COLORS[2]),
                                (0.2, 2026, UHI_COLORS[3])):
        sub = band_rows[band_rows["year"] == year].set_index("group").reindex(
            [f"dist_C_{l}" for l in labels]
        )
        ax.bar(x + offset, sub["error_rate"].values * 100.0, 0.4,
               label=str(year), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Distance to nearest per-year quartile boundary (deg C)")
    ax.set_ylabel("OOF error rate (%)")
    ax.set_title("OOF error rate vs boundary distance\n"
                 "(Moderate<->High transition uncertainty)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Uncertainty analysis — {FRAMING}", fontsize=11
    )
    return _save(
        fig, 14,
        caption=(
            "Uncertainty zones use operational thresholds (top-2 margin "
            "< 0.10 OR confidence < 0.50), not calibrated cut-offs. "
            "Error rates are " + OOF_NOTE + ". Uncertainty concentrates near "
            "per-year severity class boundaries (e.g. Moderate<->High "
            "transitions)."
        ),
    )


# ---------------------------------------------------------------------------
# fig15: transition matrix heatmap
# ---------------------------------------------------------------------------
def fig15_transition_matrix() -> Optional[Path]:
    """Row-normalized 2022->2026 severity transition heatmap."""
    path = _table("temporal_transition_matrix.csv")
    if not path.exists():
        print("  [fig15] skipped: missing temporal_transition_matrix.csv")
        return None
    df = pd.read_csv(path)
    mat_prob = np.zeros((4, 4))
    mat_count = np.zeros((4, 4), dtype=np.int64)
    for _, row in df.iterrows():
        i = CLASS_LABELS.index(row["severity_2022"])
        j = CLASS_LABELS.index(row["severity_2026"])
        mat_prob[i, j] = row["row_probability_2026_given_2022"]
        mat_count[i, j] = int(row["count"])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat_prob, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            color = "white" if mat_prob[i, j] > 0.6 else "black"
            ax.text(j, i, f"{mat_prob[i, j]:.2f}\n(n={mat_count[i, j]:,})",
                    ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f"2026 {c}" for c in CLASS_LABELS])
    ax.set_yticklabels([f"2022 {c}" for c in CLASS_LABELS])
    ax.set_xlabel("Severity 2026")
    ax.set_ylabel("Severity 2022")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="P(2026 class | 2022 class), row-normalized")
    ax.set_title(
        "Severity transition matrix, 2022 -> 2026 (row-normalized)\n" + FRAMING,
        fontsize=11,
    )
    return _save(
        fig, 15,
        caption=(
            "Paired cells valid in both years; rows sum to 1. Classes are "
            "per-year quartiles (class-relative, not deg-C comparable). "
            "Descriptive only, NOT a climate trend — two snapshot dates."
        ),
    )


# ---------------------------------------------------------------------------
# fig16: landuse heat
# ---------------------------------------------------------------------------
def fig16_landuse_heat() -> Optional[Path]:
    """Mean LST and High+Severe % per landuse class, grouped by year."""
    paths = [_table(f"landuse_heat_statistics_{y}.csv") for y in YEARS]
    if not all(p.exists() for p in paths):
        print("  [fig16] skipped: missing landuse_heat_statistics CSVs")
        return None
    tbls = {y: pd.read_csv(p) for y, p in zip(YEARS, paths)}
    codes = sorted(set(tbls[2022]["landuse_code"]) | set(tbls[2026]["landuse_code"]))
    labels = [
        tbls[2022].set_index("landuse_code")["landuse_name"].get(c, f"code {c}")
        if c in set(tbls[2022]["landuse_code"]) else
        tbls[2026].set_index("landuse_code")["landuse_name"].get(c, f"code {c}")
        for c in codes
    ]
    # Shorten the long background label but keep it explicit.
    labels = [("unclassified\nbackground" if l == "unclassified_background" else l)
              for l in labels]
    colors = {2022: UHI_COLORS[2], 2026: UHI_COLORS[3]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(codes))
    for ax, (metric, ylabel) in zip(
        axes, [("mean_lst_C", "Mean LST (deg C)"),
               ("high_severe_percent", "High+Severe (%)")]
    ):
        for offset, year in ((-0.2, 2022), (0.2, 2026)):
            tbl = tbls[year].set_index("landuse_code").reindex(codes)
            ax.bar(x + offset, tbl[metric], 0.4, label=str(year),
                   color=colors[year])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by landuse class", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Landuse vs heat severity — {FRAMING}; {ASSOCIATION_FRAMING}", fontsize=11
    )
    return _save(
        fig, 16,
        caption=(
            "Landuse classes from the Phase 3 raster; unclassified_background "
            "(code 0) is included but labelled. Classes are per-year "
            "quartiles (class-relative). " + ASSOCIATION_FRAMING + "."
        ),
    )


# ---------------------------------------------------------------------------
# Overlay sanity check (hotspot polygons vs severity background)
# ---------------------------------------------------------------------------
def verify_hotspot_overlay(max_hotspots: int = 10) -> Dict:
    """Programmatic alignment check: top-N defA polygons must sit on
    High/Severe pixels.

    Rasterizes each polygon onto the severity grid and measures the fraction
    of rasterized cells whose severity is High or Severe (definition A
    guarantees 1.0 at mask cells; a value materially below 1 would indicate
    a misaligned overlay).
    """
    print("Stage 4: hotspot overlay sanity check ...")
    _, _, transform, mask_profile = load_valid_cells()
    height, width = mask_profile["height"], mask_profile["width"]
    report: Dict[str, Dict] = {}
    for year in YEARS:
        with rasterio.open(severity_raster_path(year)) as src:
            severity = src.read(1)
        gdf = gpd.read_file(hotspot_boundaries_path("A", year))
        gdf = gdf.sort_values("area_ha", ascending=False).head(max_hotspots)
        burned = rasterio_rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        inside = burned == 1
        hs_frac = float((severity[inside] >= 2).mean()) if inside.any() else np.nan
        report[str(year)] = {
            "n_polygons_checked": int(len(gdf)),
            "polygon_cells": int(inside.sum()),
            "fraction_high_or_severe_inside_polygons": round(hs_frac, 6),
            "check_passed": bool(inside.any() and hs_frac >= 0.999),
        }
        print(
            f"  [{year}] top-{len(gdf)} polygons: {int(inside.sum()):,} cells, "
            f"{100 * hs_frac:.3f}% High/Severe inside"
        )
    report["all_passed"] = all(v["check_passed"] for v in report.values())
    return report


# ---------------------------------------------------------------------------
# Stage 4 driver
# ---------------------------------------------------------------------------
def run_stage4() -> Dict:
    """Run the complete Phase 6 Stage 4 figure suite."""
    record: Dict = {
        "phase": 6,
        "stage": 4,
        "project": "GreenGrid-AI",
        "project_root": str(PROJECT_ROOT),
        "started_utc": _now(),
        "random_seed": RANDOM_SEED,
        "steps": {},
    }

    PHASE6_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Raster prerequisites shared by the map figures.
    if not severity_raster_path(2022).exists():
        print("Stage 4: severity rasters missing — map figures will be skipped.")
        profile: Dict = {}
    else:
        _, _, _, profile = load_valid_cells()

    figures: Dict[str, str] = {}
    skipped: Dict[str, str] = {}

    def _track(fig_no: int, result: Optional[Path], why: str) -> None:
        if result is None:
            skipped[FIGURE_FILES[fig_no]] = why
        else:
            figures[FIGURE_FILES[fig_no]] = str(result)

    print("Stage 4: rendering figure suite ...")

    # fig01/02: LST maps
    for fig_no, year in ((1, 2022), (2, 2026)):
        if profile and lst_raster_path(year).exists():
            _track(fig_no, fig_lst(year, fig_no, profile), "")
        else:
            print(f"  [fig{fig_no:02d}] skipped: missing LST raster {year}")
            _track(fig_no, None, f"missing lst_{year}.tif")

    # fig03/04: severity maps
    for fig_no, year in ((3, 2022), (4, 2026)):
        if profile and severity_raster_path(year).exists():
            _track(fig_no, fig_severity(year, fig_no, profile), "")
        else:
            print(f"  [fig{fig_no:02d}] skipped: missing severity raster {year}")
            _track(fig_no, None, f"missing severity_{year}.tif")

    # fig05/06: hotspot overlays
    for fig_no, year in ((5, 2022), (6, 2026)):
        needed = [
            hotspot_boundaries_path("A", year),
            _table(f"hotspot_statistics_defA_{year}.csv"),
        ]
        if profile and all(p.exists() for p in needed):
            _track(fig_no, fig_hotspots(year, fig_no, profile), "")
        else:
            missing = [p.name for p in needed if not p.exists()]
            print(f"  [fig{fig_no:02d}] skipped: missing {missing}")
            _track(fig_no, None, f"missing {missing}")

    # fig07-16: table-driven figures
    _track(7, fig07_temporal_comparison(profile) if profile else None,
           "missing severity rasters")
    _track(8, fig08_green_vs_built(), "")
    _track(9, fig_scatter("ndvi", 9, "NDVI",
                          "Negative NDVI-LST relationship expected"),
           "")
    _track(10, fig_scatter("ndbi", 10, "NDBI",
                           "Positive NDBI-LST relationship expected"),
           "")
    _track(11, fig11_high_severe_by_block(), "")
    _track(12, fig12_hotspot_sizes(), "")
    _track(13, fig13_confidence(profile) if profile else None,
           "missing severity rasters")
    _track(14, fig14_uncertainty(profile) if profile else None,
           "missing severity rasters")
    _track(15, fig15_transition_matrix(), "")
    _track(16, fig16_landuse_heat(), "")

    # Overlay sanity check.
    overlay_check = {"all_passed": False, "note": "severity rasters unavailable"}
    if profile and all(
        p.exists()
        for p in [severity_raster_path(y) for y in YEARS]
        + [hotspot_boundaries_path("A", y) for y in YEARS]
    ):
        overlay_check = verify_hotspot_overlay()

    record["steps"]["figures"] = {
        "status": "success",
        "n_figures": len(figures),
        "figures": figures,
        "skipped": skipped,
        "dpi": DPI,
        "scatter_subsample_n": SCATTER_SUBSAMPLE_N,
        "overlay_sanity_check": overlay_check,
    }
    record["validation"] = {"hotspot_overlay": overlay_check}
    record["finished_utc"] = _now()
    record["status"] = "success" if not skipped else "completed_with_skips"
    record["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "rasterio": rasterio.__version__,
        "geopandas": gpd.__version__,
    }

    _update_manifest_figures(record)
    _append_pipeline_record_stage4(record)
    print(f"  Figures written: {len(figures)}; skipped: {len(skipped)}")
    print(f"  Manifest updated: {MANIFEST_JSON}")
    print(f"  Pipeline record appended: {PIPELINE_RECORD_JSON}")

    return {"record": record, "figures": figures, "skipped": skipped}


def _update_manifest_figures(record: Dict) -> None:
    """Read-modify-write the Phase 6 manifest with the figures block."""
    with open(MANIFEST_JSON) as f:
        manifest = json.load(f)

    manifest["figures"] = {
        "stage": 4,
        "generated_utc": _now(),
        "framing": (
            "All figures describe ML-based RELATIVE heat severity / an "
            "operational UHI hotspot proxy, NOT physical UHI intensity. "
            "2022/2026 are two snapshot composites, not a long-term trend; "
            "severity classes are per-year LST quartiles (class-relative "
            "comparison only).  Group contrasts are associations, not causal "
            "effects."
        ),
        "style": {
            "severity_palette": dict(zip(CLASS_LABELS, UHI_COLORS)),
            "continuous_cmaps": {"lst": "inferno", "confidence": "cividis"},
            "nodata_color": NODATA_COLOR,
            "dpi": DPI,
            "map_frame": (
                "lat/lon ticks, approximate scale bar (~97 km/deg lon, "
                "~111 km/deg lat at Delhi), north arrow, explicit NoData patch"
            ),
        },
        "scatter_subsample": {
            "n_per_year": SCATTER_SUBSAMPLE_N,
            "seed": RANDOM_SEED,
            "source": "data/processed/phase4/tables/combined_urban_environmental_dataset.csv",
        },
        "figures": record["steps"]["figures"]["figures"],
        "skipped": record["steps"]["figures"]["skipped"],
        "validation": record["validation"],
    }
    _save_json(manifest, MANIFEST_JSON)


def _append_pipeline_record_stage4(record: Dict) -> None:
    """Append the Stage 4 entry to the Phase 6 pipeline record (idempotent)."""
    with open(PIPELINE_RECORD_JSON) as f:
        existing = json.load(f)

    container = {"phase": 6, "project": "GreenGrid-AI", "stage_records": []}
    if "stage_records" in existing:
        container["stage_records"] = existing["stage_records"]
    else:
        container["stage_records"] = [existing]
    # Idempotent re-runs replace the previous Stage 4 entry.
    container["stage_records"] = [r for r in container["stage_records"] if r.get("stage") != 4]
    container["stage_records"].append(record)
    _save_json(container, PIPELINE_RECORD_JSON)


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for the Phase 6 Stage 4 figure suite."""
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 6 Stage 4 figure suite."
    )
    parser.parse_args(argv)

    try:
        results = run_stage4()
        print(json.dumps(_convert_for_json(results["record"]), indent=2, default=str))
        return 0 if results["record"]["status"] == "success" else 1
    except Exception as exc:
        print(f"Phase 6 Stage 4 pipeline failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
