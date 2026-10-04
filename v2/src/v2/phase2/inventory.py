"""V2 Phase-2 audit 1/5: full inventory of data/raw/ with hashes and stats.

Outputs (under data/v2/phase2/inventory/):
  inventory.csv        one row per file (file-level fields)
  inventory_bands.csv  one row per (raster, band) with exact streaming stats
  inventory.json       machine-readable merge of both tables

Read-only on data/raw. Band statistics are exact (full-resolution streaming
accumulation over row windows), not downsampled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUT_ROOT,
    RAW_ROOT,
    band_stats_windowed,
    category_from_path,
    dump_json,
    rel,
    sha256_file,
    year_from_path,
)

INV_DIR = OUT_ROOT / "inventory"
RASTER_EXT = {".tif", ".tiff"}
VECTOR_EXT = {".geojson", ".json"}


def overpass_json_info(path: Path) -> dict:
    """Parse Overpass API JSON (OSM elements) for inventory purposes."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    elements = doc.get("elements", [])
    type_counts: dict[str, int] = {}
    lons, lats = [], []
    geom_kinds: set[str] = set()
    for el in elements:
        t = el.get("type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1
        if t == "node":
            if "lon" in el and "lat" in el:
                lons.append(el["lon"])
                lats.append(el["lat"])
        elif t == "way":
            geom = el.get("geometry") or []
            if geom:
                geom_kinds.add("linestring-ish")
            for g in geom:
                if "lon" in g and "lat" in g:
                    lons.append(g["lon"])
                    lats.append(g["lat"])
            if "center" in el:
                lons.append(el["center"]["lon"])
                lats.append(el["center"]["lat"])
        elif t == "relation" and "center" in el:
            lons.append(el["center"]["lon"])
            lats.append(el["center"]["lat"])
    bbox = [min(lons), min(lats), max(lons), max(lats)] if lons else []
    return {
        "geometry_type": f"osm-overpass-json({'/'.join(sorted(type_counts)) or 'empty'})",
        "feature_count": int(len(elements)),
        "crs": "EPSG:4326 (implied, OSM lon/lat)",
        "bbox": bbox,
        "element_type_counts": type_counts,
    }


def vector_info(path: Path) -> dict:
    import geopandas as gpd

    try:
        gdf = gpd.read_file(path)
        geom_types = sorted(gdf.geom_type.unique().tolist())
        bbox = list(gdf.total_bounds) if len(gdf) else []
        return {
            "geometry_type": ";".join(geom_types),
            "feature_count": int(len(gdf)),
            "crs": gdf.crs.to_string() if gdf.crs is not None else None,
            "bbox": bbox,
        }
    except Exception:
        return overpass_json_info(path)


def raster_info(path: Path) -> tuple[dict, list[dict]]:
    with rasterio.open(path) as src:
        file_row = {
            "driver": src.driver,
            "dtype": ";".join(sorted(set(src.dtypes))),
            "crs": src.crs.to_string() if src.crs else None,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "band_names": [d if d else "" for d in src.descriptions],
            "nodata_vals": list(src.nodatavals),
            "transform": [src.transform.a, src.transform.b, src.transform.c,
                          src.transform.d, src.transform.e, src.transform.f],
        }
        stats = band_stats_windowed(src)
        band_rows = []
        names = file_row["band_names"]
        for i in range(1, src.count + 1):
            d = stats[i].as_dict()
            d.update({
                "band_index": i,
                "band_name": names[i - 1],
                "nodata_attr": src.nodatavals[i - 1],
                "dtype": src.dtypes[i - 1],
            })
            band_rows.append(d)
    return file_row, band_rows


def main() -> int:
    INV_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW_ROOT.rglob("*") if p.is_file())
    print(f"[inventory] {len(files)} files under {RAW_ROOT}")

    file_rows, band_rows = [], []
    for k, p in enumerate(files, 1):
        ext = p.suffix.lower()
        row = {
            "file_path": rel(p),
            "category": category_from_path(p),
            "year": year_from_path(p),
            "file_size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "format": ext.lstrip("."),
        }
        if ext in RASTER_EXT:
            frow, brows = raster_info(p)
            row.update(frow)
            for br in brows:
                br = dict(br)
                br["file_path"] = row["file_path"]
                band_rows.append(br)
        elif ext in VECTOR_EXT:
            try:
                row.update(vector_info(p))
            except Exception as e:  # noqa: BLE001
                row["vector_error"] = f"{type(e).__name__}: {e}"
        file_rows.append(row)
        print(f"[inventory] {k}/{len(files)} {p.name}", flush=True)

    df = pd.DataFrame(file_rows)
    bdf = pd.DataFrame(band_rows)
    df.to_csv(INV_DIR / "inventory.csv", index=False)
    bdf.to_csv(INV_DIR / "inventory_bands.csv", index=False)
    dump_json(
        {"files": df.to_dict("records"), "bands": bdf.to_dict("records")},
        INV_DIR / "inventory.json",
    )
    print(f"[inventory] wrote {INV_DIR}/inventory.csv ({len(df)} rows), "
          f"inventory_bands.csv ({len(bdf)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
