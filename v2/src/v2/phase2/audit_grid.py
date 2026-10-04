"""V2 Phase-2 audit 2/5: grid audit of every raster vs the V1 reference grid.

Authoritative grid: data/raw/landsat9/2026_07/landsat9_2026_07_composite_30m.tif
(EPSG:4326, 30 m, V1 2026-reference grid).

Classification of each raster:
  EXACT-MATCH           same CRS, same resolution, same origin, same dims
  SAME-GRID-DIFF-EXTENT origin delta is an integer multiple of the file's own
                        pixel size and the resolution divides the reference
                        resolution, so exact scale aggregation onto the
                        reference grid is possible (only extent differs)
  RESAMPLING-REQUIRED   origin delta is not an integer multiple of the file's
                        own pixel size (half-pixel / sub-pixel shift, flagged
                        in half_pixel_shift_own) or the resolution does not
                        divide the reference resolution
  MISMATCH              CRS differs or the geotransform has rotation

Outputs: audit/grid_audit.csv, audit/grid_audit.json
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUT_ROOT,
    RAW_ROOT,
    REFERENCE_RASTER,
    dump_json,
    rel,
)

AUDIT_DIR = OUT_ROOT / "audit"
ANG_TOL = 1e-9  # degrees


def pixel_mod(delta: float, res: float) -> float:
    """Fractional part of delta in units of res, in (-0.5, 0.5]."""
    r = (delta / res) % 1.0
    if r > 0.5:
        r -= 1.0
    return r


def audit_one(path: Path, ref) -> dict:
    row = {"file_path": rel(path)}
    with rasterio.open(path) as src:
        t = src.transform
        width, height = src.width, src.height
        row.update({
            "width": width, "height": height,
            "crs": src.crs.to_string() if src.crs else None,
            "res_x": t.a, "res_y": -t.e,
            "rotation_b": t.b, "rotation_d": t.d,
            "origin_x": t.c, "origin_y": t.f,
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
            "pixel_size_ratio_vs_ref": ref.res / t.a,
        })

    row["crs_match"] = row["crs"] == ref.crs
    row["resolution_match"] = (
        abs(t.a - ref.res) < 1e-12 and abs(-t.e - ref.res) < 1e-12
    )
    row["no_rotation"] = abs(t.b) < 1e-15 and abs(t.d) < 1e-15

    dx = t.c - ref.origin_x
    dy = t.f - ref.origin_y
    row["origin_dx_deg"] = dx
    row["origin_dy_deg"] = dy
    # Shift expressed in reference pixels and in the file's own pixels.
    row["origin_dx_ref_pixels"] = dx / ref.res
    row["origin_dy_ref_pixels"] = dy / ref.res
    row["origin_dx_own_pixels"] = dx / t.a
    row["origin_dy_own_pixels"] = dy / -t.e
    # Fractional residual modulo pixel size (half-pixel shift test):
    # 0.0 == aligned, 0.5 == exactly half a pixel off.
    row["origin_dx_mod_own"] = pixel_mod(dx, t.a)
    row["origin_dy_mod_own"] = pixel_mod(dy, -t.e)
    row["origin_dx_mod_ref"] = pixel_mod(dx, ref.res)
    row["origin_dy_mod_ref"] = pixel_mod(dy, ref.res)

    # Integer-divisor check: can ref pixels be exactly aggregated to this res?
    ratio = ref.res / t.a
    row["res_ratio_is_integer"] = abs(ratio - round(ratio)) < 1e-9

    aligned_own = (
        abs(row["origin_dx_mod_own"]) < ANG_TOL
        and abs(row["origin_dy_mod_own"]) < ANG_TOL
    )
    aligned_ref = (
        abs(row["origin_dx_mod_ref"]) < ANG_TOL
        and abs(row["origin_dy_mod_ref"]) < ANG_TOL
    )
    row["aligned_own_lattice"] = aligned_own
    row["aligned_ref_lattice"] = aligned_ref
    row["half_pixel_shift_own"] = (
        abs(abs(row["origin_dx_mod_own"]) - 0.5) < 1e-6
        or abs(abs(row["origin_dy_mod_own"]) - 0.5) < 1e-6
    )

    dims_match = width == ref.width and height == ref.height
    row["dims_match"] = dims_match
    row["width_delta_vs_ref"] = width - ref.width
    row["height_delta_vs_ref"] = height - ref.height
    # Expected dims if the file covered the reference extent at its own res.
    row["expected_width_aligned"] = math.floor(ref.width * ref.res / t.a + 0.5)
    row["expected_height_aligned"] = math.floor(ref.height * ref.res / t.a + 0.5)

    if not row["crs_match"] or not row["no_rotation"]:
        cls = "MISMATCH"
    elif row["resolution_match"] and dims_match and abs(dx) < ANG_TOL and abs(dy) < ANG_TOL:
        cls = "EXACT-MATCH"
    elif aligned_own and row["res_ratio_is_integer"]:
        # Origin is an integer multiple of the file's own pixel size away from
        # the reference origin and the resolution divides the reference res:
        # exact scale aggregation onto the reference grid is possible.
        cls = "SAME-GRID-DIFF-EXTENT"
    else:
        cls = "RESAMPLING-REQUIRED"
    row["classification"] = cls

    # Extent deltas (degrees) for context.
    row["left_delta_deg"] = (t.c) - ref.origin_x
    row["top_delta_deg"] = (t.f) - ref.origin_y
    row["right_delta_deg"] = (t.c + width * t.a) - ref.right
    row["bottom_delta_deg"] = (t.f + height * t.e) - ref.bottom
    return row


class RefGrid:
    def __init__(self) -> None:
        with rasterio.open(REFERENCE_RASTER) as src:
            t = src.transform
            self.res = t.a
            self.origin_x = t.c
            self.origin_y = t.f
            self.width = src.width
            self.height = src.height
            self.right = t.c + src.width * t.a
            self.bottom = t.f + src.height * t.e
            self.crs = src.crs.to_string()


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ref = RefGrid()
    rasters = sorted(RAW_ROOT.rglob("*.tif"))
    rows = [audit_one(p, ref) for p in rasters]
    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_DIR / "grid_audit.csv", index=False)
    payload = {
        "reference_raster": rel(REFERENCE_RASTER),
        "reference": {
            "crs": ref.crs,
            "res_x": ref.res, "res_y": ref.res,
            "origin_x": ref.origin_x, "origin_y": ref.origin_y,
            "width": ref.width, "height": ref.height,
            "right": ref.right, "bottom": ref.bottom,
        },
        "tolerance_deg": ANG_TOL,
        "files": df.to_dict("records"),
    }
    dump_json(payload, AUDIT_DIR / "grid_audit.json")
    print(df[["file_path", "classification"]].to_string(index=False))
    print(f"[grid_audit] {len(df)} rasters -> {AUDIT_DIR}/grid_audit.csv,.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
