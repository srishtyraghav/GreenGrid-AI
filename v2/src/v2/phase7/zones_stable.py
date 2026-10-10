"""V2 Phase 7 FINAL (2026-only) -- candidate planting sites from 2026 conditions.

Sites (2026 only) = feasible land (eligible landuse, ~water/buildings/road
surfaces) that passes the S2 gates (priority >= pooled-p90 0.5926 AND
cooling_need >= pooled-p75 0.7116 -- pinned operational constants) AND has
vegetation_cover < 0.30; 8-connected, MMU >= 2 ha, max-size cap 57 ha
(distribution p99; kills mega-patches). Multi-year persistence machinery
(h/K/MIN-veg/per-year columns) was REMOVED by design (2026 planning module).

RANKING SCORE (0-100, 2026 components only, each indicator once):
  score = 0.50*heat_need + 0.25*vegetation_deficit + 0.25*planting_opportunity
(0-100 pooled-normalized components; the old persistence term was dropped and
the weights renormalized to sum to 1 -- documented).

SHORTLIST = top 100 sites by score, oversized excluded. Cut evidence (from
the cumulative usable-area curve): band 800-1,200 ha is entered at rank 91
(945 ha); top 100 = 1,046 ha (min score 72.97); scores descend smoothly
(74.8 -> 73.0 across the top 100, no score cliff); oversized (>57 ha) sites
are excluded from the shortlist but kept as candidates.

Site IDs = score rank (1 = highest). Everything here is a 2026 product.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from ..common import Grid
from .suitability import (
    green_proximity_score, label_zones, landuse_eligibility, pooled_normalize,
    polygons_per_zone, road_accessibility_score, validate_geometries,
)

MMU_MIN_PX = 23
MAX_SIZE_HA = 57.0
SCORE_WEIGHTS = {"need": 0.50, "veg_deficit": 0.25, "opportunity": 0.25}
SHORTLIST_TOP_N = 100
YEAR = 2026
HA = 0.09
SIMPLIFY_TOL_DEG = 0.0005


def build_stable_zones(scored, ctx, years, grid: Grid, out_root: Path) -> dict:
    y = YEAR
    static = ctx["static"]
    lu = ctx["lu_codes"]
    elig = ctx["eligible_lu"]
    constraint = ctx["constraint"]
    R = ctx["pooled_refs"]
    prod = ctx["year_products"][y]

    scy = scored[y]
    dom = scy["domain"]
    need_v = scy["need"]
    pr_v = np.full(dom.shape, np.nan)
    feas = elig & ~constraint & dom
    pr_v[feas] = 0.5 * need_v[feas] + 0.5 * scy["suitability"][feas]
    vok = np.isfinite(prod["vegetation_cover"]) & (prod["vegetation_cover"] < 0.30)
    sites = feas & np.isfinite(pr_v) & (pr_v >= ctx["sel_p90"]) & (need_v >= ctx["sel_nd75"]) & vok

    zlab, _ = label_zones(sites, min_pixels=MMU_MIN_PX)
    n_lst = 100.0 * pooled_normalize(prod["lst_C"], R["lst"]["p1"], R["lst"]["p99"])
    n_1v = 100.0 * pooled_normalize(1.0 - prod["vegetation_cover"],
                                    1.0 - R["veg"]["p99"], 1.0 - R["veg"]["p1"])
    n_ndvi = 100.0 * pooled_normalize(prod["ndvi"], R["ndvi"]["p1"], R["ndvi"]["p99"])
    n_ndbi = 100.0 * pooled_normalize(prod["ndbi"], R["ndbi"]["p1"], R["ndbi"]["p99"])
    opp = np.clip((0.30 * landuse_eligibility(lu) / 100.0
                   + 0.25 * (100.0 - n_ndbi) / 100.0
                   + 0.15 * road_accessibility_score(static["dist_road_m"]) / 100.0
                   + 0.15 * green_proximity_score(static["dist_vegetation_m"]) / 100.0
                   + 0.15 * (100.0 - n_ndvi) / 100.0) * 100.0, 0.0, 100.0)

    rows = []
    for z in [int(v) for v in np.unique(zlab[zlab > 0])]:
        m = zlab == z
        n_px = int(m.sum())
        ha = n_px * HA
        need_c = float(np.nanmean(n_lst[m]))
        vdef_c = float(np.nanmean(n_1v[m]))
        opp_c = float(np.nanmean(opp[m]))
        score = (SCORE_WEIGHTS["need"] * need_c + SCORE_WEIGHTS["veg_deficit"] * vdef_c
                 + SCORE_WEIGHTS["opportunity"] * opp_c)
        codes, counts = np.unique(lu[m], return_counts=True)
        lu_ha = {int(c_): int(n_) * HA for c_, n_ in zip(codes, counts)}
        untagged = (float(counts[list(codes).index(0)] / n_px) if 0 in list(codes) else 0.0)
        rows.append({"raw_id": z, "zone_px": n_px, "usable_px": n_px,
                     "usable_area_ha": round(ha, 3),
                     "mean_lst_C": round(float(np.nanmean(prod["lst_C"][m])), 3),
                     "mean_ndvi": round(float(np.nanmean(prod["ndvi"][m])), 4),
                     "mean_ndbi": round(float(np.nanmean(prod["ndbi"][m])), 4),
                     "mean_vegetation_cover": round(float(np.nanmean(prod["vegetation_cover"][m])), 4),
                     "heat_need_comp": round(need_c, 2),
                     "vegetation_deficit_comp": round(vdef_c, 2),
                     "opportunity_comp": round(opp_c, 2),
                     "planning_priority_score": round(score, 2),
                     "oversized": ha > MAX_SIZE_HA,
                     "untagged_share": round(untagged, 3),
                     "landuse_certainty": ("low" if untagged > 0.5 else
                                           "mixed" if untagged > 0.2 else "identified")})
    zdf = pd.DataFrame(rows).sort_values("planning_priority_score",
                                         ascending=False).reset_index(drop=True)
    zdf.insert(0, "rank", np.arange(1, len(zdf) + 1))
    zdf.insert(0, "zone_id", zdf["rank"])
    # top-N among NON-oversized sites in score order (oversized excluded from
    # the shortlist but keep their rank ids)
    zdf["rank_excl"] = zdf.loc[~zdf["oversized"], "rank"].rank(method="first")
    # oversized sites carry no exclusion-rank; 0 is the documented sentinel
    # (NaN is not JSON-serializable through the API layer)
    zdf["rank_excl"] = zdf["rank_excl"].fillna(0).astype(int)
    zdf["shortlist"] = (~zdf["oversized"]) & (zdf["rank_excl"] <= SHORTLIST_TOP_N)
    zdf["priority_class"] = np.where(zdf["shortlist"], "Shortlist", "Candidate")
    zdf["rationale"] = [
        f"score={r.planning_priority_score:.1f} = 0.50*need({r.heat_need_comp:.1f}) + "
        f"0.25*veg_deficit({r.vegetation_deficit_comp:.1f}) + "
        f"0.25*opportunity({r.opportunity_comp:.1f}); 2026 conditions; "
        f"{'top-100 shortlist' if r.shortlist else ('oversized (>57 ha) - excluded from shortlist' if r.oversized else 'candidate')}"
        for r in zdf.itertuples()]
    idmap = dict(zip(zdf["raw_id"], zdf["zone_id"]))
    zlab_stable = np.vectorize(lambda v: idmap.get(int(v), 0))(zlab).astype(np.int32)

    out = {}
    for scenario in ("v1_parity", "v2_constrained"):
        sdir = out_root / scenario
        (sdir / "rasters").mkdir(parents=True, exist_ok=True)
        (sdir / "zones").mkdir(parents=True, exist_ok=True)
        with rasterio.open(sdir / "rasters" / f"stable_zone_ids_{scenario}.tif", "w",
                           **grid.profile(count=1, dtype="int32", nodata=-1)) as dst:
            dst.write(zlab_stable, 1)
            dst.set_band_description(1, "2026 candidate site ids (score rank)")
        polys = polygons_per_zone(zlab_stable, grid.transform) if len(zdf) else {}
        validation = validate_geometries(polys)
        gdf = gpd.GeoDataFrame(
            zdf, geometry=[polys[int(z)].simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)
                           for z in zdf["zone_id"]], crs="EPSG:4326") if len(zdf) else \
            gpd.GeoDataFrame(zdf, geometry=gpd.GeoSeries(dtype="geometry"), crs="EPSG:4326")
        gdf.to_file(sdir / "zones" / f"stable_zones_{scenario}.geojson", driver="GeoJSON")
        zdf.to_csv(sdir / "zones" / f"stable_zones_{scenario}.csv", index=False)
        short = zdf[zdf["shortlist"]]
        out[scenario] = {"n_sites": len(zdf), "n_shortlist": len(short),
                         "candidate_ha": round(float(zdf["usable_area_ha"].sum()), 1),
                         "shortlist_ha": round(float(short["usable_area_ha"].sum()), 1),
                         "shortlist_min_score": float(short["planning_priority_score"].min()) if len(short) else None,
                         "oversized_zones": int(zdf["oversized"].sum()),
                         "geom_valid": bool(validation["all_valid"])}
    return {"zones": out, "frame": zdf, "label_raster_v2": zlab_stable,
            "constants": {
                "year": YEAR, "mmu_min_px": MMU_MIN_PX, "max_size_ha": MAX_SIZE_HA,
                "score_weights": SCORE_WEIGHTS,
                "score_note": ("0.50 need + 0.25 veg_deficit + 0.25 opportunity "
                               "(persistence term dropped, weights renormalized)"),
                "shortlist_top_n": SHORTLIST_TOP_N,
                "shortlist_evidence": ("cumulative usable curve: 800-1,200 ha band "
                                       "entered at rank 91 (945 ha); top 100 = "
                                       "1,046 ha, min score 72.97, smooth score "
                                       "descent (no cliff); oversized excluded"),
                "score_label": ("planning-priority score (0-100), not a validated "
                                "probability or proof of optimal cooling"),
                "s2_gates": "priority >= 0.5926 AND cooling_need >= 0.7116 (pinned)"}}
