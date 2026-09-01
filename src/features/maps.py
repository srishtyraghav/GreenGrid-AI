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
    L9_LST_2022,
    L9_LST_2026,
    PHASE4_MAPS_DIR,
    S2_NDBI_2022,
    S2_NDBI_2026,
    S2_NDVI_2022,
    S2_NDVI_2026,
    VEGETATION_COVER_RASTER_2022,
    VEGETATION_COVER_RASTER_2026,
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


# Authoritative map catalogue
MAP_CATALOGUE: List[MapSpec] = [
    MapSpec(
        variable="lst",
        year=2022,
        raster_path=L9_LST_2022,
        title="Land Surface Temperature (LST) — 2022",
        cmap="hot",
        vmin=None,
        vmax=None,
        unit="°C",
        colorbar_label="LST (°C)",
    ),
    MapSpec(
        variable="lst",
        year=2026,
        raster_path=L9_LST_2026,
        title="Land Surface Temperature (LST) — 2026",
        cmap="hot",
        vmin=None,
        vmax=None,
        unit="°C",
        colorbar_label="LST (°C)",
    ),
    MapSpec(
        variable="ndvi",
        year=2022,
        raster_path=S2_NDVI_2022,
        title="Normalised Difference Vegetation Index (NDVI) — 2022",
        cmap="RdYlGn",
        vmin=-0.2,
        vmax=1.0,
        unit="unitless",
        colorbar_label="NDVI",
    ),
    MapSpec(
        variable="ndvi",
        year=2026,
        raster_path=S2_NDVI_2026,
        title="Normalised Difference Vegetation Index (NDVI) — 2026",
        cmap="RdYlGn",
        vmin=-0.2,
        vmax=1.0,
        unit="unitless",
        colorbar_label="NDVI",
    ),
    MapSpec(
        variable="ndbi",
        year=2022,
        raster_path=S2_NDBI_2022,
        title="Normalised Difference Built-up Index (NDBI) — 2022",
        cmap="Spectral_r",
        vmin=-0.5,
        vmax=0.5,
        unit="unitless",
        colorbar_label="NDBI",
    ),
    MapSpec(
        variable="ndbi",
        year=2026,
        raster_path=S2_NDBI_2026,
        title="Normalised Difference Built-up Index (NDBI) — 2026",
        cmap="Spectral_r",
        vmin=-0.5,
        vmax=0.5,
        unit="unitless",
        colorbar_label="NDBI",
    ),
    MapSpec(
        variable="vegetation_cover",
        year=2022,
        raster_path=VEGETATION_COVER_RASTER_2022,
        title="Proportional Vegetation Cover — 2022",
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
        unit="fraction",
        colorbar_label="Vegetation cover fraction",
    ),
    MapSpec(
        variable="vegetation_cover",
        year=2026,
        raster_path=VEGETATION_COVER_RASTER_2026,
        title="Proportional Vegetation Cover — 2026",
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
        unit="fraction",
        colorbar_label="Vegetation cover fraction",
    ),
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
