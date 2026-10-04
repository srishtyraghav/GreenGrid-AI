"""V2 Phase-2 audit 3/5: NoData / sentinel audit of every raster band.

For every band of every raster (recomputed exactly via streaming windows):
  - nodata attribute value vs actual sentinel usage
  - NaN count, +/-Inf count, zero count, min/max finite
  - classification:
      NODATA-ATTR-SET   the file declares a nodata value (usage also counted)
      NAN-SENTINEL      no nodata attr; NaNs are the de-facto mask
      ZERO-USED-AS-NODATA?  zeros coincide with the file's own masked area
                            (zero pixels only/mostly where other bands are
                            masked, not in the valid region)
      NO-SENTINEL       dense array, no nodata attr, no NaNs

Outputs: audit/nodata_audit.csv, audit/nodata_audit.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT_ROOT, RAW_ROOT, BandStats, dump_json, rel  # noqa: E402

AUDIT_DIR = OUT_ROOT / "audit"
WIN_ROWS = 256


def audit_raster(path: Path) -> list[dict]:
    rows = []
    with rasterio.open(path) as src:
        n = src.count
        height, width = src.height, src.width

        # Pass 1: per-band stats (exact) + whole-file valid mask counts.
        stats = {i: BandStats() for i in range(1, n + 1)}
        # any_finite: pixels where at least one band is finite
        any_finite = np.zeros((height, width), dtype=np.uint32)
        # For zero-coincidence: accumulate zero counts inside/outside the
        # file's own valid region (valid = finite in EVERY band).
        zero_in_valid = {i: 0 for i in range(1, n + 1)}
        zero_total = {i: 0 for i in range(1, n + 1)}
        nodata_usage = {i: 0 for i in range(1, n + 1)}

        for row0 in range(0, height, WIN_ROWS):
            h = min(WIN_ROWS, height - row0)
            w = Window(0, row0, width, h)
            bands = [src.read(i, window=w) for i in range(1, n + 1)]
            allfin = np.ones((h, width), dtype=bool)
            for i, arr in enumerate(bands, 1):
                stats[i].update(arr)
                fin = np.isfinite(arr)
                allfin &= fin
                any_finite[row0:row0 + h] += fin
                z = fin & (arr == 0)
                zero_total[i] += int(z.sum())
                nd = src.nodatavals[i - 1]
                if nd is not None and np.issubdtype(arr.dtype, np.floating):
                    nodata_usage[i] += int((fin & (arr == nd)).sum())
            # now that allfin is final for this window, split zeros into
            # valid-region vs masked-region occurrences
            for i, arr in enumerate(bands, 1):
                fin = np.isfinite(arr)
                z = fin & (arr == 0)
                zero_in_valid[i] += int((z & allfin).sum())
            del bands

        any_finite_mask = any_finite > 0

        for i in range(1, n + 1):
            s = stats[i]
            d = s.as_dict()
            nd_attr = src.nodatavals[i - 1]
            zeros_outside_valid = zero_total[i] - zero_in_valid[i]
            if nd_attr is not None:
                cls = "NODATA-ATTR-SET"
            elif s.nan > 0:
                cls = "NAN-SENTINEL"
            elif zero_total[i] > 0 and zero_in_valid[i] == 0:
                cls = "ZERO-USED-AS-NODATA?"
            elif zero_total[i] > 0:
                cls = "ZERO-USED-AS-NODATA? (mixed with valid-region zeros)"
            else:
                cls = "NO-SENTINEL"
            row = {
                "file_path": rel(path),
                "band_index": i,
                "band_name": src.descriptions[i - 1] or "",
                "dtype": src.dtypes[i - 1],
                "nodata_attr": nd_attr,
                "nodata_attr_usage_count": nodata_usage[i],
                "nan": s.nan,
                "posinf": s.posinf,
                "neginf": s.neginf,
                "zeros": zero_total[i],
                "zeros_in_valid_region": zero_in_valid[i],
                "zeros_outside_valid_region": zeros_outside_valid,
                "min_finite": d["min"],
                "max_finite": d["max"],
                "mean_finite": d["mean"],
                "std_finite": d["std"],
                "pct_finite": d["pct_finite"],
                "pixels_outside_valid_region": int((~any_finite_mask).sum()),
                "classification": cls,
            }
            rows.append(row)
    return rows


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rasters = sorted(RAW_ROOT.rglob("*.tif"))
    rows: list[dict] = []
    for p in rasters:
        rows.extend(audit_raster(p))
        print(f"[nodata] {len(rows)} bands done; last {p.name}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(AUDIT_DIR / "nodata_audit.csv", index=False)
    dump_json({"bands": df.to_dict("records")}, AUDIT_DIR / "nodata_audit.json")
    print(f"[nodata] {len(df)} bands -> {AUDIT_DIR}/nodata_audit.csv,.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
