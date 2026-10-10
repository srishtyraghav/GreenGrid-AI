"""V2 Phase 7 v3.2 — per-year candidate planting SITES with stable IDs.

Sites (per year) = contiguous (8-conn, MMU >= 2 ha) patches of land passing
the S2 gates (priority >= pooled-p90 AND cooling_need >= pooled-p75) within
the valid domain — heat-relevant cores. Feasibility/usable area is computed
in Phase 8 (heat != plantable); here we form geometry and STABLE IDs only.

STABLE ID SCHEME (documented): year-sites are chained across years through
an ID ACCUMULATOR raster. Years are processed in order; each year-site takes
the id of the previous-years site it overlaps most (>=30% of its pixels) --
so an id is STABLE whenever the site's geometry persists across years, even
across gaps. Within one year, if several sites claim the same prior id, the
largest keeps it and the others receive documented ephemeral ids >= 90000
(stable_id=False). IDs are positive integers; the accumulator is updated to
the current year's ids after each year (latest-year wins at overlaps).

Boundaries are exact pixel unions (rasterio 4-conn shapes + unary_union);
a 15 m (0.0005 deg) simplify tolerance is applied ONLY to the GeoJSON
display geometry and recorded here; all area accounting uses pixel counts.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

from ..common import Grid
from .suitability import label_zones

MMU_PX = 23                 # >= 2 ha
SIMPLIFY_TOL_DEG = 0.0005   # ~15 m display-only boundary tolerance
EPHEMERAL_BASE = 90000


class IdChainer:
    """Cross-year stable id assignment via an accumulator raster."""

    def __init__(self, shape):
        self.accum = np.zeros(shape, dtype=np.int32)   # latest-year ids per px
        self.next_id = 1

    def assign(self, year_mask) -> np.ndarray:
        zlab, n = label_zones(year_mask, min_pixels=MMU_PX)
        out = np.zeros_like(zlab)
        claims: dict[int, list[int]] = {}
        for z in [int(v) for v in np.unique(zlab[zlab > 0])]:
            m = zlab == z
            n_px = int(m.sum())
            prior = self.accum[m]
            prior = prior[prior > 0]
            best, best_frac = 0, 0.0
            if prior.size:
                vals, counts = np.unique(prior, return_counts=True)
                i = int(np.argmax(counts))
                best, best_frac = int(vals[i]), float(counts[i]) / n_px
            if best and best_frac >= 0.30:
                claims.setdefault(best, []).append(z)
            else:
                claims.setdefault(-1, []).append(z)
        for prior_id, zs in claims.items():
            if prior_id == -1:
                for z in zs:
                    out[zlab == z] = self.next_id
                    self.next_id += 1
                continue
            if len(zs) == 1:
                out[zlab == zs[0]] = prior_id
                continue
            sizes = {z: int((zlab == z).sum()) for z in zs}
            keep = max(sizes, key=sizes.get)
            out[zlab == keep] = prior_id
            eph = EPHEMERAL_BASE + self.next_id
            for z in zs:
                if z != keep:
                    out[zlab == z] = eph
                    eph += 1
        self.accum = out
        return out


def write_year_sites(year, scenario, gate_mask, chainer, priority, grid,
                     out_dir: Path) -> dict:
    """Form year sites, assign stable cross-year ids, write raster + geojson."""
    ids = chainer.assign(gate_mask)
    rasters_dir = out_dir / scenario / "rasters"
    zones_dir = out_dir / scenario / "zones"
    rasters_dir.mkdir(parents=True, exist_ok=True)
    zones_dir.mkdir(parents=True, exist_ok=True)
    prof = grid.profile(count=1, dtype="int32", nodata=-1)
    with rasterio.open(rasters_dir / f"site_ids_{scenario}_{year}.tif", "w", **prof) as dst:
        dst.write(ids.astype(np.int32), 1)
        dst.set_band_description(1, "candidate site ids (stable where persistent)")
    # geometry: exact pixel unions; display-only simplify
    from .suitability import polygons_per_zone, validate_geometries
    polys = polygons_per_zone(ids, grid.transform) if ids.max() > 0 else {}
    validation = validate_geometries(polys)
    rows = []
    for z in [int(v) for v in np.unique(ids[ids > 0])]:
        m = ids == z
        n_px = int(m.sum())
        rr, cc = np.nonzero(m)
        rows.append({
            "site_id": z, "year": year, "scenario": scenario,
            "stable_id": bool(z < EPHEMERAL_BASE),
            "pixel_count": n_px,
            "total_area_ha": round(n_px * 0.09, 3),
            "mean_priority_score": round(float(priority[m].mean()), 4),
            "centroid_lon": round(float(grid.transform.c + (cc.mean() + 0.5) * grid.transform.a), 7),
            "centroid_lat": round(float(grid.transform.f + (rr.mean() + 0.5) * grid.transform.e), 7),
        })
    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(
        df, geometry=[polys[int(z)].simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)
                      for z in df["site_id"]], crs="EPSG:4326") if len(df) else \
        gpd.GeoDataFrame(df, geometry=gpd.GeoSeries(dtype="geometry"), crs="EPSG:4326")
    gdf.to_file(zones_dir / f"candidate_sites_{scenario}_{year}.geojson", driver="GeoJSON")
    df.drop(columns=[]).to_csv(zones_dir / f"candidate_sites_{scenario}_{year}.csv", index=False)
    return {"n_sites": len(df), "stable_ids": int(df["stable_id"].sum()),
            "geom_valid": bool(validation["all_valid"]),
            "simplify_tol_deg": SIMPLIFY_TOL_DEG, "ids_raster": str(rasters_dir / f"site_ids_{scenario}_{year}.tif")}
