"""V2 Phase 7 constraint layers — burn water / buildings / road surfaces to
the 30 m grid and build the V2-constrained scenario exclusion mask.

Sources (V2 Phase 2 fetch, ODbL 1.0; see CONSTRAINTS_PROVENANCE.md):
  ``v2/data/phase2/constraints/osm_water_delhi.geojson``         2,682 features
  ``v2/data/phase2/constraints/osm_buildings_delhi.geojson``   417,343 footprints
  ``v2/data/phase2/constraints/osm_road_surfaces_delhi.geojson``    78 polygons

Burn rule: presence/absence, ``all_touched=True`` (a pixel touched by a
constraint geometry is excluded — conservative, matches the V1 static-layer
burn convention). Water LineStrings (canals/riverbanks, 181) are burned too:
a water line crossing a pixel marks it excluded. These layers did NOT exist
in V1 — V1 documented their absence as a limitation; the v2_constrained
scenario closes it.

Exclusion accounting attributes each excluded domain pixel to exactly ONE
class with precedence water > buildings > road_surfaces (union accounting
plus per-layer attributed counts are both reported).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from ..common import Grid

CONSTRAINT_FILES = {
    "water": "osm_water_delhi.geojson",
    "buildings": "osm_buildings_delhi.geojson",
    "road_surfaces": "osm_road_surfaces_delhi.geojson",
}
# attribution precedence for pixels hit by more than one layer
PRECEDENCE = ("water", "buildings", "road_surfaces")


def burn_layer(geojson_path: Path, grid: Grid) -> np.ndarray:
    """Rasterize one constraint GeoJSON to a boolean presence grid."""
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(grid.crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros(grid.shape, dtype=bool)
    burned = rasterize(shapes, out_shape=grid.shape, transform=grid.transform,
                       fill=0, dtype="uint8", all_touched=True)
    return burned.astype(bool)


def build_constraint_stack(constraints_dir: Path, grid: Grid,
                           out_dir: Path | None = None) -> dict:
    """Burn all three layers; return masks + attributed exclusion accounting.

    If ``out_dir`` is given, write per-layer uint8 presence rasters
    (1 = constraint present, 0 = absent, full grid) and the combined
    attributed-class raster (0 none / 1 water / 2 buildings / 3 road
    surfaces; uint8, nodata 255 declared-never-used) for provenance.
    """
    masks = {}
    for name, fname in CONSTRAINT_FILES.items():
        masks[name] = burn_layer(Path(constraints_dir) / fname, grid)

    attributed = np.zeros(grid.shape, dtype=np.uint8)      # 0 = none
    # precedence water > buildings > road_surfaces: apply LOWEST first so the
    # highest-precedence layer wins on overlaps
    for code, name in ((3, "road_surfaces"), (2, "buildings"), (1, "water")):
        attributed[masks[name]] = code

    result = {
        "masks": masks,
        "attributed": attributed,
        "counts_full_grid": {name: int(m.sum()) for name, m in masks.items()},
        "counts_union_full_grid": int((attributed > 0).sum()),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prof = grid.profile(count=1, dtype="uint8", nodata=None)
        prof.pop("nodata", None)
        for name, m in masks.items():
            with rasterio.open(out_dir / f"constraint_{name}_30m.tif", "w",
                               **prof) as dst:
                dst.write(m.astype(np.uint8), 1)
                dst.set_band_description(1, f"constraint presence: {name}")
        with rasterio.open(out_dir / "constraint_attributed_30m.tif", "w",
                           **prof) as dst:
            dst.write(attributed, 1)
            dst.set_band_description(
                1, "0 none / 1 water / 2 buildings / 3 road_surfaces")
        result["rasters_dir"] = str(out_dir)
    return result
