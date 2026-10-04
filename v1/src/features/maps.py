"""Map generation for Phase 4 visual outputs.

Creates report-ready static maps for the four key environmental variables
required by the synopsis, for both 2022 and 2026. Maps are generated from
the authoritative rasters and the newly derived Vegetation Cover layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from features.config import (
    L9_LST_RASTERS,
    PHASE4_MAPS_DIR,
    S2_NDBI_RASTERS,
    S2_NDVI_RASTERS,
    VEGETATION_COVER_RASTERS,
    YEARS,
)
from features.io import read_raster_array

matplotlib.use("Agg")


@dataclass(frozen=True)
class MapSpec:
    """Specification for a single map product."""

    variable: str
    year: int
    raster_path: Path
    title: str
    cmap: str
    vmin: float | None = None
    vmax: float | None = None
    unit: str = ""
    colorbar_label: str = ""


# Authoritative map catalogue — built for every year in YEARS.
# (variable, raster set, title, cmap, vmin, vmax, unit, colorbar label)
_MAP_DEFS = [
    ("lst", L9_LST_RASTERS, "Land Surface Temperature (LST)", "hot", None, None, "°C", "LST (°C)"),
    ("ndvi", S2_NDVI_RASTERS, "Normalised Difference Vegetation Index (NDVI)", "RdYlGn", -0.2, 1.0, "unitless", "NDVI"),
    ("ndbi", S2_NDBI_RASTERS, "Normalised Difference Built-up Index (NDBI)", "Spectral_r", -0.5, 0.5, "unitless", "NDBI"),
    ("vegetation_cover", VEGETATION_COVER_RASTERS, "Proportional Vegetation Cover", "YlGn", 0.0, 1.0, "fraction", "Vegetation cover fraction"),
]

MAP_CATALOGUE: List[MapSpec] = [
    MapSpec(
        variable=variable,
        year=year,
        raster_path=raster_set[year],
        title=f"{title} — {year}",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        unit=unit,
        colorbar_label=colorbar_label,
    )
    for variable, raster_set, title, cmap, vmin, vmax, unit, colorbar_label in _MAP_DEFS
    for year in YEARS
]


def _plot_raster_map(
    arr: np.ndarray,
    spec: MapSpec,
    output_path: Path,
    dpi: int = 150,
    figsize: Tuple[float, float] = (8, 7),
) -> Dict:
    """Render a single map and save it as PNG."""
    fig, ax = plt.subplots(figsize=figsize)

    # Mask NaN for display (matplotlib handles NaN as transparent when using
    # masked arrays, but explicit masking is clearer).
    plot_arr = np.ma.masked_where(np.isnan(arr), arr)

    im = ax.imshow(
        plot_arr,
        cmap=spec.cmap,
        vmin=spec.vmin,
        vmax=spec.vmax,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title(spec.title, fontsize=12)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(spec.colorbar_label, fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    valid = arr[~np.isnan(arr)]
    return {
        "variable": spec.variable,
        "year": spec.year,
        "output_path": str(output_path),
        "cmap": spec.cmap,
        "vmin": spec.vmin,
        "vmax": spec.vmax,
        "n_valid_pixels": int(valid.size),
        "min": float(np.nanmin(arr)) if valid.size else None,
        "max": float(np.nanmax(arr)) if valid.size else None,
        "mean": float(np.nanmean(arr)) if valid.size else None,
    }


def generate_maps(dpi: int = 150) -> List[Dict]:
    """Generate all Phase 4 maps and return a list of metadata dicts."""
    results: List[Dict] = []
    for spec in MAP_CATALOGUE:
        arr = read_raster_array(spec.raster_path)
        output_path = PHASE4_MAPS_DIR / f"{spec.variable}_{spec.year}.png"
        result = _plot_raster_map(arr, spec, output_path, dpi=dpi)
        results.append(result)
    return results


def list_generated_map_paths() -> List[Path]:
    """Return the expected output map paths."""
    return [PHASE4_MAPS_DIR / f"{spec.variable}_{spec.year}.png" for spec in MAP_CATALOGUE]


if __name__ == "__main__":
    results = generate_maps()
    for r in results:
        print(r)
