"""Phase 6 Stage 2: hotspot threshold-sensitivity comparison.

For each year, compare the predefined hotspot definitions A (primary:
High+Severe), B (Severe only) and C (A + confidence >= 0.60) in terms of
hotspot counts, areas, coverage of the valid grid, and pairwise mask
overlap (intersection/union IoU and the share of the first mask covered
by the second).

These are ML-based RELATIVE heat-severity hotspots — an operational UHI
hotspot proxy, NOT physical UHI intensity.  Comparisons are restricted to
valid cells; NoData never participates and is never read as "not a
hotspot".

Writes ``data/processed/phase6/tables/hotspot_sensitivity.csv``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import HOTSPOT_SENSITIVITY_CSV

PAIRWISE = [("A", "B"), ("A", "C"), ("B", "C")]


def _overlap_metrics(mask_x: np.ndarray, mask_y: np.ndarray) -> Dict[str, float]:
    """Return IoU and overlap fractions between two binary masks."""
    inter = int((mask_x & mask_y).sum())
    union = int((mask_x | mask_y).sum())
    px_x = int(mask_x.sum())
    px_y = int(mask_y.sum())
    return {
        "intersection_px": inter,
        "union_px": union,
        "iou": round(inter / union, 6) if union > 0 else np.nan,
        "pct_of_first_covered_by_second": round(100.0 * inter / px_x, 4) if px_x > 0 else np.nan,
        "pct_of_second_covered_by_first": round(100.0 * inter / px_y, 4) if px_y > 0 else np.nan,
    }


def build_sensitivity_table(
    masks: Dict[int, Dict[str, np.ndarray]],
    valid_pixels: Dict[int, int],
    summaries: Dict[int, Dict[str, Dict]],
) -> pd.DataFrame:
    """Build the per-year sensitivity table.

    Parameters
    ----------
    masks : dict
        ``masks[year][definition]`` = FINAL binary hotspot mask over the
        full grid (True only at valid cells, after the min-size rule).
    valid_pixels : dict
        ``valid_pixels[year]`` = number of valid cells that year.
    summaries : dict
        Delineation summaries (``summaries[year][definition]``) carrying
        cluster counts and UTM-based areas.

    Returns
    -------
    pd.DataFrame
        One ``per_definition`` row per year/definition (counts and UTM
        areas) and one ``pairwise_overlap`` row per year/pair (mask IoU
        and coverage fractions).
    """
    rows: List[Dict] = []

    for year in sorted(masks):
        for definition in ("A", "B", "C"):
            s = summaries[year][definition]
            n_px = int(s["hotspot_pixels"])
            rows.append(
                {
                    "section": "per_definition",
                    "year": year,
                    "definition": definition,
                    "pair": "",
                    "n_hotspots": int(s["n_hotspots"]),
                    "hotspot_px": n_px,
                    "total_area_ha": s["total_area_ha"],
                    "largest_hotspot_area_ha": s["max_area_ha"],
                    "pct_of_valid_area": round(100.0 * n_px / valid_pixels[year], 4),
                    "intersection_px": None,
                    "union_px": None,
                    "iou": None,
                    "pct_of_first_covered_by_second": None,
                    "pct_of_second_covered_by_first": None,
                }
            )

        for first, second in PAIRWISE:
            rows.append(
                {
                    "section": "pairwise_overlap",
                    "year": year,
                    "definition": "",
                    "pair": f"{first}_vs_{second}",
                    "n_hotspots": None,
                    "hotspot_px": None,
                    "total_area_ha": None,
                    "largest_hotspot_area_ha": None,
                    "pct_of_valid_area": None,
                    **_overlap_metrics(masks[year][first], masks[year][second]),
                }
            )

    return pd.DataFrame(rows)


def write_sensitivity_csv(df: pd.DataFrame, output_path=HOTSPOT_SENSITIVITY_CSV) -> str:
    """Write the sensitivity table to CSV and return the path."""
    int_cols = ["n_hotspots", "hotspot_px", "intersection_px", "union_px"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    df.to_csv(output_path, index=False)
    return str(output_path)


__all__ = [
    "PAIRWISE",
    "build_sensitivity_table",
    "write_sensitivity_csv",
]
