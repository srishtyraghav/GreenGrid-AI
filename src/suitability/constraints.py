"""Hard exclusions for Phase 7 — only what the data supports.

The ONLY hard exclusion is the invalid mask (``valid_mask_30m.tif``,
nodata=0): NoData cells never participate in any computation and are never
interpreted as "unsuitable" — they are unclassified.  An exclusion-mask
raster is produced for provenance: value 1 = excluded (invalid/NoData),
value 0 = analysis domain.

Explicitly documented limitations (input audit section 4 — no supporting
data exists for any of these):

- **No water exclusion** — no NDWI raster, no water landuse class, no
  water OSM polygons (0 tags found).  Water is only indirectly visible
  (strongly negative NDVI, cool Yamuna corridor, masked LST edges); none
  of these is a validated exclusion layer.
- **No building-footprint exclusion** — buildings are an OSM sample, no
  footprint raster exists.
- **No road-surface exclusion** — only road *distance* exists; road
  surface pixels cannot be masked.

These are stated limitations, not silent omissions, and must appear in the
Phase 7 report's limitations section.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import rasterio

from .config import EXCLUSION_MASK_RASTER


def build_exclusion_mask(valid_mask: np.ndarray, profile: Dict) -> Dict:
    """Write the provenance exclusion raster and return its metadata.

    Parameters
    ----------
    valid_mask : np.ndarray
        Full-grid bool array (True = analysis domain, from the Phase 3
        valid mask).
    profile : dict
        Reference raster profile (copied from a Phase 6 severity raster).

    Returns
    -------
    dict
        Output path, pixel counts and the framing note.  Encoding: 1 =
        excluded (invalid / NoData), 0 = valid analysis domain; the raster
        covers the FULL grid — every pixel is classified as excluded or not.
        A nodata value (255) is declared in the profile for conformance with
        the project raster contract (explicit nodata on every raster); it is
        never used because no pixel carries it.
    """
    raster = np.where(valid_mask, 0, 1).astype(np.uint8)

    out_profile = profile.copy()
    out_profile.update(
        {
            "dtype": "uint8",
            "count": 1,
            "nodata": 255,  # declared, never used: full-grid classification
            "compress": "lzw",
        }
    )
    with rasterio.open(EXCLUSION_MASK_RASTER, "w", **out_profile) as dst:
        dst.write(raster, 1)

    return {
        "output_path": str(EXCLUSION_MASK_RASTER),
        "encoding": "1 = excluded (invalid/NoData), 0 = analysis domain",
        "excluded_px": int((raster == 1).sum()),
        "analysis_domain_px": int((raster == 0).sum()),
        "note": (
            "Only hard exclusion is the invalid mask. No water / building-"
            "footprint / road-surface exclusion is implementable with "
            "current project data (audit section 4); stated limitations, "
            "not silent omissions."
        ),
    }


__all__ = ["build_exclusion_mask"]
