"""Map and raster generation for Phase 5 predictions.

Converts pixel-level predictions and probabilities into GeoTIFF rasters and
report-ready PNG maps on the Phase 3/4 reference grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap

from .config import (
    CLASS_LABELS,
    CLASS_VALUES,
    CONFIDENCE_RASTER_2022,
    CONFIDENCE_RASTER_2026,
    INV_CLASS_MAPPING,
    PROBABILITY_RASTERS,
    REFERENCE_RASTER,
    UHI_MAP_2022,
    UHI_MAP_2026,
    UHI_RASTER_2022,
    UHI_RASTER_2026,
)

# Categorical colormap for UHI classes.
UHI_COLORS = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"]  # Low, Moderate, High, Severe
UHI_CMAP = ListedColormap(UHI_COLORS)


def load_reference_profile(path: Path = REFERENCE_RASTER) -> Dict:
    """Load the reference raster profile (CRS, transform, shape, dtype)."""
    with rasterio.open(path) as src:
        profile = src.profile.copy()
    return profile


def predictions_to_raster(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    year: int,
    nodata: int = -1,
    dtype: np.dtype = np.int16,
) -> Dict:
    """Write a single-band raster from a prediction table.

    The raster matches the reference grid. Pixels not present in `df` are
    assigned NoData.
    """
    profile = load_reference_profile()
    profile.update(
        {
            "dtype": dtype,
            "count": 1,
            "nodata": nodata,
            "compress": "lzw",
        }
    )

    # Initialize with NoData.
    raster = np.full((profile["height"], profile["width"]), nodata, dtype=dtype)

    sub = df[df["year"] == year]
    rows = sub["row"].values.astype(int)
    cols = sub["col"].values.astype(int)
    values = sub[value_col].values

    # Clip to raster bounds for safety.
    valid = (rows >= 0) & (rows < profile["height"]) & (cols >= 0) & (cols < profile["width"])
    rows = rows[valid]
    cols = cols[valid]
    values = values[valid]

    raster[rows, cols] = values

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(raster, 1)

    return {
        "output_path": str(output_path),
        "year": year,
        "n_valid_pixels": int(valid.sum()),
        "shape": raster.shape,
        "crs": str(profile["crs"]),
    }


def probabilities_to_rasters(
    df: pd.DataFrame,
    year: int,
    output_paths: Dict[str, Path],
    nodata: float = -1.0,
) -> List[Dict]:
    """Write per-class probability rasters for a single year."""
    profile = load_reference_profile()
    profile.update(
        {
            "dtype": "float32",
            "count": 1,
            "nodata": nodata,
            "compress": "lzw",
        }
    )

    sub = df[df["year"] == year]
    rows = sub["row"].values.astype(int)
    cols = sub["col"].values.astype(int)

    valid = (rows >= 0) & (rows < profile["height"]) & (cols >= 0) & (cols < profile["width"])
    rows = rows[valid]
    cols = cols[valid]

    results = []
    for label in CLASS_LABELS:
        raster = np.full((profile["height"], profile["width"]), nodata, dtype=np.float32)
        values = sub[f"probability_{label.lower()}"].values[valid]
        raster[rows, cols] = values

        out_path = output_paths[label]
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(raster, 1)

        results.append(
            {
                "output_path": str(out_path),
                "year": year,
                "class": label,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        )

    # Confidence raster = max probability.
    confidence_raster = np.full((profile["height"], profile["width"]), nodata, dtype=np.float32)
    confidence = sub["prediction_confidence"].values[valid]
    confidence_raster[rows, cols] = confidence
    conf_path = CONFIDENCE_RASTER_2022 if year == 2022 else CONFIDENCE_RASTER_2026
    with rasterio.open(conf_path, "w", **profile) as dst:
        dst.write(confidence_raster, 1)

    results.append(
        {
            "output_path": str(conf_path),
            "year": year,
            "class": "confidence",
            "min": float(confidence.min()),
            "max": float(confidence.max()),
            "mean": float(confidence.mean()),
        }
    )

    return results


def generate_uhi_map(
    df: pd.DataFrame,
    year: int,
    output_path: Path,
    title: str = None,
) -> Dict:
    """Generate a categorical UHI map PNG for a single year."""
    profile = load_reference_profile()
    raster = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float32)

    sub = df[df["year"] == year]
    rows = sub["row"].values.astype(int)
    cols = sub["col"].values.astype(int)
    values = sub["predicted_class"].values

    valid = (rows >= 0) & (rows < profile["height"]) & (cols >= 0) & (cols < profile["width"])
    raster[rows[valid], cols[valid]] = values[valid]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        raster,
        cmap=UHI_CMAP,
        vmin=CLASS_VALUES[0] - 0.5,
        vmax=CLASS_VALUES[-1] + 0.5,
        interpolation="nearest",
    )
    ax.set_title(title or f"UHI Relative Heat Severity — {year}")
    ax.axis("off")

    patches = [
        mpatches.Patch(color=UHI_COLORS[i], label=CLASS_LABELS[i])
        for i in range(len(CLASS_LABELS))
    ]
    ax.legend(handles=patches, loc="lower right", title="Severity")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    return {
        "output_path": str(output_path),
        "year": year,
        "n_pixels_plotted": int(valid.sum()),
    }


def generate_all_maps(df: pd.DataFrame) -> List[Dict]:
    """Generate UHI rasters and maps for both years."""
    results = []

    for year in (2022, 2026):
        results.append(
            predictions_to_raster(df, "predicted_class", UHI_RASTER_2022 if year == 2022 else UHI_RASTER_2026, year)
        )
        results.extend(probabilities_to_rasters(df, year, PROBABILITY_RASTERS[year]))
        results.append(
            generate_uhi_map(df, year, UHI_MAP_2022 if year == 2022 else UHI_MAP_2026)
        )

    return results
