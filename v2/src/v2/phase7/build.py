"""V2 Phase 7 build — plantation suitability for both scenarios, all years.

Reuses the Phase 4 code path (``v2.phase4.assemble_features`` loaders) for
the environmental rasters, the frozen V1 suitability formulation
(``v2.phase7.suitability``), and the V2 constraint stack
(``v2.phase7.constraints``). Consumes Phase 6 severity products (V2 Phase 5
RF primary lineage, recorded from the Phase 6 manifest).

Usage:
    PYTHONPATH=v2/src .venv/Scripts/python.exe -m v2.phase7.build \
        [--years 2022,2023,...] [--out v2/data/phase7]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from ..common import (
    GRID_FILE_DEFAULT,
    PROJECT_ROOT,
    YEARS,
    Grid,
    compute_block_raster,
    dump_json,
    sha256_file,
)
from ..phase4.assemble_features import load_phase3_year, load_static
from . import suitability as S
from .constraints import CONSTRAINT_FILES, PRECEDENCE, build_constraint_stack

PHASE6_DIR_DEFAULT = PROJECT_ROOT / "data" / "phase6"
PHASE3_ROOT_DEFAULT = PROJECT_ROOT / "data" / "phase3"
CONSTRAINTS_DIR_DEFAULT = PROJECT_ROOT / "data" / "phase2" / "constraints"
OUT_DEFAULT = PROJECT_ROOT / "data" / "phase7"

SCENARIOS = ("v1_parity", "v2_constrained")

# per-year Phase 6 inputs consumed (V2 layout; severity_score owns the domain)
P6_REQUIRED = ("severity_score", "severity", "confidence")


def log(msg: str) -> None:
    print(f"[P7] {msg}", flush=True)


def read_band(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float64)
        nodata = ds.profile.get("nodata")
        nodata = float(nodata) if nodata is not None else np.nan
    return arr, nodata


def finite_mask(arr, nodata):
    if np.isnan(nodata):
        return np.isfinite(arr)
    return np.isfinite(arr) & (arr != nodata)


def preflight(years, phase6_dir, phase3_root, constraints_dir, grid_file):
    for y in years:
        for name in P6_REQUIRED:
            p = phase6_dir / "rasters" / f"{name}_{y}.tif"
            if not p.exists():
                raise FileNotFoundError(f"missing phase6 input: {p}")
        if not (phase3_root / str(y)).is_dir():
            raise FileNotFoundError(f"missing phase3 year: {y}")
    for f in CONSTRAINT_FILES.values():
        if not (constraints_dir / f).exists():
            raise FileNotFoundError(f"missing constraint layer: {f}")
    if not (phase6_dir / "phase6_manifest.json").exists():
        raise FileNotFoundError("missing phase6 manifest")
    if not Path(grid_file).exists():
        raise FileNotFoundError(f"missing grid file: {grid_file}")
    grid = Grid.from_file(Path(grid_file))
    assert (grid.height, grid.width) == (1768, 1874), "authoritative grid mismatch"


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------
def write_score(flat, path, griddef, shape, rows, cols):
    values = flat.astype(np.float32)
    assert np.isfinite(values).all() and values.min() >= 0.0 and values.max() <= 100.0
    raster = np.full(shape, S.SCORE_NODATA, dtype=np.float32)
    raster[rows, cols] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = griddef.profile(count=1, dtype="float32", nodata=S.SCORE_NODATA)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(raster, 1)
        dst.set_band_description(1, path.stem)
    return {"path": str(path), "bytes": path.stat().st_size}


def write_class(flat_cls, path, griddef, shape, rows, cols, dtype, nodata):
    raster = np.full(shape, nodata, dtype=dtype)
    raster[rows, cols] = flat_cls.astype(dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = griddef.profile(count=1, dtype=dtype.name if hasattr(dtype, "name") else str(dtype),
                           nodata=nodata)
    prof["dtype"] = np.dtype(dtype).name
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(raster, 1)
        dst.set_band_description(1, path.stem)
    return {"path": str(path), "bytes": path.stat().st_size}


# ---------------------------------------------------------------------------
# Year pipeline
# ---------------------------------------------------------------------------
def run_year(year, ctx, out_root):
    grid: Grid = ctx["grid"]
    shape = grid.shape
    t0 = time.perf_counter()
    log(f"=== {year} ===")

    sev, sev_nd = read_band(ctx["phase6_dir"] / "rasters" / f"severity_score_{year}.tif")
    domain = finite_mask(sev, sev_nd)
    products = load_phase3_year(ctx["phase3_root"], year)
    yearly = {"severity_score": sev,
              "ndvi": products["ndvi"], "ndbi": products["ndbi"],
              "lst": products["lst_C"], "vegetation_cover": products["vegetation_cover"]}
    del products
    for name, arr in yearly.items():
        domain &= np.isfinite(arr)
    if not domain.any():
        raise AssertionError(f"{year}: empty analysis domain")
    rows, cols = np.nonzero(domain)
    flat = {k: v[domain] for k, v in yearly.items()}
    del yearly
    lu = ctx["static"]["landuse_class"]          # float64; NaN == nodata(255)
    assert np.isfinite(ctx["static"]["dist_road_m"][domain]).all() and \
        np.isfinite(ctx["static"]["dist_vegetation_m"][domain]).all(), \
        "static distance rasters contain non-finite domain cells"
    static_flat = {
        "landuse": np.where(np.isfinite(lu), lu, 255.0)[domain].astype(np.int64),
        "dist_road_m": ctx["static"]["dist_road_m"][domain],
        "dist_vegetation_m": ctx["static"]["dist_vegetation_m"][domain],
    }

    # per-year p1/p99 normalization (frozen V1 rule) + bounds record
    norm = {}
    bounds_rows = []
    for name in ("severity_score", "ndvi", "ndbi", "lst", "vegetation_cover"):
        scaled, bounds = S.robust_minmax(flat[name])
        norm[name] = scaled
        bounds_rows.append({"year": year, "variable": name, **bounds})
    scaled, bounds = S.robust_minmax(1.0 - flat["ndvi"])
    norm["one_minus_ndvi"] = scaled
    bounds_rows.append({"year": year, "variable": "one_minus_ndvi", **bounds})

    need = S.compute_heat_need(norm)
    opp = S.compute_opportunity(norm, static_flat)
    suitability = S.gated_product(need, opp["opportunity"])
    cls_flat = S.classify(suitability.astype(np.float32).astype(np.float64))
    tiers_flat = S.priority_tier(cls_flat)

    year_info = {"year": year, "domain_px": int(domain.sum()),
                 "normalization_bounds": bounds_rows, "scenarios": {}}
    excluded = ctx["constraints"]["attributed"] > 0

    for scenario in SCENARIOS:
        ts = time.perf_counter()
        sdir = out_root / scenario
        rasters_dir, zones_dir, tables_dir = sdir / "rasters", sdir / "zones", sdir / "tables"
        for d in (rasters_dir, zones_dir, tables_dir):
            d.mkdir(parents=True, exist_ok=True)
        tag = scenario
        if scenario == "v2_constrained":
            sdomain = domain & ~excluded
        else:
            sdomain = domain
        srows, scols = np.nonzero(sdomain)
        sidx = sdomain[domain]              # index of scenario cells in domain flats
        n_s = int(sdomain.sum())

        out = {}
        out["suitability"] = write_score(suitability[sidx], rasters_dir / f"suitability_{tag}_{year}.tif", grid, shape, srows, scols)
        out["heat_need"] = write_score(need[sidx], rasters_dir / f"heat_need_{tag}_{year}.tif", grid, shape, srows, scols)
        out["opportunity"] = write_score(opp["opportunity"][sidx], rasters_dir / f"plantation_opportunity_{tag}_{year}.tif", grid, shape, srows, scols)
        s_cls = cls_flat[sidx]
        out["suitability_class"] = write_class(s_cls, rasters_dir / f"suitability_class_{tag}_{year}.tif", grid, shape, srows, scols, np.int16, S.CLASS_NODATA)
        out["priority"] = write_class(tiers_flat[sidx], rasters_dir / f"priority_{tag}_{year}.tif", grid, shape, srows, scols, np.uint8, 255)
        # exclusion mask: full-grid classification (V1 encoding: 1=excluded, 0=domain)
        excl_raster = np.where(sdomain, 0, 1).astype(np.uint8)
        excl_path = rasters_dir / f"exclusion_mask_{tag}_{year}.tif"
        excl_path.parent.mkdir(parents=True, exist_ok=True)
        prof = grid.profile(count=1, dtype="uint8", nodata=255)
        with rasterio.open(excl_path, "w", **prof) as dst:
            dst.write(excl_raster, 1)
            dst.set_band_description(1, "1=excluded, 0=analysis domain")
        out["exclusion_mask"] = {"path": str(excl_path), "bytes": excl_path.stat().st_size,
                                 "excluded_px": int((excl_raster == 1).sum()),
                                 "analysis_domain_px": int((excl_raster == 0).sum())}

        # ---- priority zones (class >= 3, V1 rule) --------------------------
        cls_grid = np.full(shape, S.CLASS_NODATA, dtype=np.int16)
        cls_grid[srows, scols] = s_cls
        binary = (cls_grid >= S.ZONE_MIN_CLASS) & sdomain
        labelled, n_zones = S.label_zones(binary)
        polygons = S.polygons_per_zone(labelled, grid.transform) if n_zones else {}
        validation = S.validate_geometries(polygons)
        zids = labelled[srows, scols]
        zone_rows = []
        for zid in sorted(int(i) for i in np.unique(zids[zids > 0])):
            sel = zids == zid
            n_px = int(sel.sum())
            geom = polygons[zid]
            vals, counts = np.unique(s_cls[sel], return_counts=True)
            zone_rows.append({
                "zone_id": zid, "year": year, "scenario": scenario,
                "pixel_count": n_px,
                "area_ha": round(n_px * S.PX_AREA_M2 / S.M2_PER_HA, 3),
                "centroid_lon": round(geom.centroid.x, 7),
                "centroid_lat": round(geom.centroid.y, 7),
                "mean_suitability": round(float(suitability[sidx][sel].mean()), 4),
                "mean_class": round(float(s_cls[sel].mean()), 3),
                "dominant_class": int(vals[np.argmax(counts)]),
                "mean_heat_need": round(float(need[sidx][sel].mean()), 4),
                "mean_opportunity": round(float(opp["opportunity"][sidx][sel].mean()), 4),
            })
        zframe = pd.DataFrame(zone_rows)
        zframe.to_csv(tables_dir / f"priority_zone_statistics_{tag}_{year}.csv", index=False)
        ranking = zframe.sort_values(["mean_suitability", "area_ha"],
                                     ascending=[False, False]).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        ranking.to_csv(tables_dir / f"priority_ranking_{tag}_{year}.csv", index=False)
        # zone ids raster: full grid, 0 = not a zone, -1 never occurs inside grid
        zpath = rasters_dir / f"priority_zone_ids_{tag}_{year}.tif"
        prof = grid.profile(count=1, dtype="int32", nodata=-1)
        with rasterio.open(zpath, "w", **prof) as dst:
            dst.write(labelled.astype(np.int32), 1)
            dst.set_band_description(1, "priority zone ids (0 = not a zone)")
        import geopandas as gpd
        if not zframe.empty:
            gdf = gpd.GeoDataFrame(
                zframe, geometry=[polygons[int(z)] for z in zframe["zone_id"]],
                crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame(zframe, geometry=gpd.GeoSeries(dtype="geometry"),
                                   crs="EPSG:4326")
        gdf.to_file(zones_dir / f"priority_zones_{tag}_{year}.geojson", driver="GeoJSON")

        # ---- tables ---------------------------------------------------------
        n_dom = n_s
        tier_rows = []
        for tier, label in ((3, "High"), (2, "Medium"), (1, "Low")):
            n = int((tiers_flat[sidx] == tier).sum())
            tier_rows.append({"year": year, "scenario": scenario,
                              "priority": label, "pixel_count": n,
                              "area_ha": round(n * S.PX_AREA_M2 / S.M2_PER_HA, 3),
                              "pct_of_domain": round(100.0 * n / n_dom, 4)})
        hist = np.bincount(s_cls, minlength=5)
        for c in range(5):
            tier_rows.append({"year": year, "scenario": scenario,
                              "priority": f"class_{c}_{S.CLASS_LABELS[c].replace(' ', '_').lower()}",
                              "pixel_count": int(hist[c]),
                              "area_ha": round(int(hist[c]) * S.PX_AREA_M2 / S.M2_PER_HA, 3),
                              "pct_of_domain": round(100.0 * hist[c] / n_dom, 4)})
        pd.DataFrame(tier_rows).to_csv(
            tables_dir / f"priority_tiers_{tag}_{year}.csv", index=False)

        blk = ctx["block_raster"]
        blk_rows = []
        for bid in range(25):
            bmask = (blk == bid) & sdomain
            n_b = int(bmask.sum())
            if n_b == 0:
                continue
            bidx = sdomain[domain] & (blk[domain] == bid)
            b_cls = cls_flat[bidx]
            blk_rows.append({
                "year": year, "scenario": scenario, "spatial_block_id": bid,
                "domain_px": n_b,
                "mean_suitability": round(float(suitability[bidx].mean()), 4),
                "mean_heat_need": round(float(need[bidx].mean()), 4),
                "mean_opportunity": round(float(opp["opportunity"][bidx].mean()), 4),
                "class_ge3_px": int((b_cls >= S.ZONE_MIN_CLASS).sum()),
                "priority_high_px": int((b_cls == 4).sum()),
                "priority_medium_px": int((b_cls == 3).sum()),
                "priority_low_px": int((b_cls == 2).sum()),
                "zone_ha": round(float(sum(
                    r["area_ha"] for r in zone_rows
                    if bid in np.unique(blk[labelled == r["zone_id"]]))), 3),
            })
        pd.DataFrame(blk_rows).to_csv(
            tables_dir / f"block_suitability_{tag}_{year}.csv", index=False)

        excl_rows = []
        if scenario == "v2_constrained":
            in_dom_excl = excluded & domain
            total = int(in_dom_excl.sum())
            for code, name in ((1, "water"), (2, "buildings"), (3, "road_surfaces")):
                raw = ctx["constraints"]["masks"][name]
                excl_rows.append({
                    "year": year, "scenario": scenario, "layer": name,
                    "burned_px_full_grid": int(raw.sum()),
                    "excluded_domain_px": int((raw & domain).sum()),
                    "attributed_px": int(((ctx["constraints"]["attributed"] == code) & domain).sum()),
                    "area_ha": round(int((raw & domain).sum()) * S.PX_AREA_M2 / S.M2_PER_HA, 3),
                })
            excl_rows.append({"year": year, "scenario": scenario, "layer": "UNION",
                              "burned_px_full_grid": total,
                              "excluded_domain_px": total,
                              "attributed_px": total,
                              "area_ha": round(total * S.PX_AREA_M2 / S.M2_PER_HA, 3)})
            pd.DataFrame(excl_rows).to_csv(
                tables_dir / f"exclusion_accounting_{tag}_{year}.csv", index=False)

        hvh = int(((s_cls >= S.ZONE_MIN_CLASS)).sum())
        year_info["scenarios"][scenario] = {
            "domain_px": n_dom,
            "mean_need": round(float(need[sidx].mean()), 4),
            "mean_opportunity": round(float(opp["opportunity"][sidx].mean()), 4),
            "mean_suitability": round(float(suitability[sidx].mean()), 4),
            "class_ge3_px": hvh,
            "class_ge3_ha": round(hvh * S.PX_AREA_M2 / S.M2_PER_HA, 3),
            "priority_high_ha": round(int((tiers_flat[sidx] == 3).sum()) * S.PX_AREA_M2 / S.M2_PER_HA, 3),
            "priority_medium_ha": round(int((tiers_flat[sidx] == 2).sum()) * S.PX_AREA_M2 / S.M2_PER_HA, 3),
            "priority_low_ha": round(int((tiers_flat[sidx] == 1).sum()) * S.PX_AREA_M2 / S.M2_PER_HA, 3),
            "n_zones": n_zones,
            "zone_geometry_valid": bool(validation["all_valid"]),
            "zone_pixel_counts_match": bool(
                (int(zframe["pixel_count"].sum()) if n_zones else 0) == int((labelled > 0).sum())),
            "exclusion_rows": excl_rows if scenario == "v2_constrained" else [],
            "wall_s": time.perf_counter() - ts,
            "outputs": out,
        }
        log(f"{year} {scenario}: domain={n_dom:,} zones={n_zones} "
            f"High+VH={hvh:,}px wall={year_info['scenarios'][scenario]['wall_s']:.1f}s")

    year_info["wall_s"] = time.perf_counter() - t0
    return year_info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase6-dir", default=str(PHASE6_DIR_DEFAULT))
    ap.add_argument("--phase3-root", default=str(PHASE3_ROOT_DEFAULT))
    ap.add_argument("--constraints-dir", default=str(CONSTRAINTS_DIR_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--years", default=",".join(str(y) for y in YEARS))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    ap.add_argument("--skip-constraint-burn", action="store_true",
                    help="reuse existing v2/data/phase7/constraints_rasters")
    args = ap.parse_args(argv)

    years = tuple(int(y) for y in args.years.split(","))
    phase6_dir = Path(args.phase6_dir)
    phase3_root = Path(args.phase3_root)
    constraints_dir = Path(args.constraints_dir)
    out_root = Path(args.out)
    t_start = time.perf_counter()

    preflight(years, phase6_dir, phase3_root, constraints_dir, Path(args.grid_file))
    grid = Grid.from_file(Path(args.grid_file))
    log(f"grid {grid.width}x{grid.height}; years={years}")

    t = time.perf_counter()
    burn_dir = out_root / "constraints_rasters"
    if args.skip_constraint_burn and (burn_dir / "constraint_attributed_30m.tif").exists():
        import rasterio as _rio
        masks = {}
        for name in CONSTRAINT_FILES:
            with _rio.open(burn_dir / f"constraint_{name}_30m.tif") as ds:
                masks[name] = ds.read(1).astype(bool)
        with _rio.open(burn_dir / "constraint_attributed_30m.tif") as ds:
            attributed = ds.read(1)
        constraints = {"masks": masks, "attributed": attributed,
                       "counts_full_grid": {n: int(m.sum()) for n, m in masks.items()},
                       "counts_union_full_grid": int((attributed > 0).sum())}
        log(f"constraint stack reloaded from {burn_dir}")
    else:
        constraints = build_constraint_stack(constraints_dir, grid, burn_dir)
    log(f"constraint stack ready in {time.perf_counter() - t:.1f}s: "
        f"{constraints['counts_full_grid']} union={constraints['counts_union_full_grid']:,}")

    ctx = {
        "grid": grid, "phase6_dir": phase6_dir, "phase3_root": phase3_root,
        "static": load_static(phase3_root),
        "block_raster": compute_block_raster(grid.height, grid.width),
        "constraints": constraints,
    }

    year_infos = [run_year(y, ctx, out_root) for y in years]

    p6_manifest = json.loads((phase6_dir / "phase6_manifest.json").read_text())
    summary = {str(yi["year"]): {sc: {k: v for k, v in yi["scenarios"][sc].items()
                                      if k not in ("outputs", "exclusion_rows")}
                                 for sc in SCENARIOS} for yi in year_infos}
    record = {
        "phase": 7, "variant": "v2", "project": "GreenGrid-AI",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "years": list(years), "scenarios": list(SCENARIOS),
        "total_wall_s": time.perf_counter() - t_start,
        "per_year_wall_s": {str(yi["year"]): round(yi["wall_s"], 3) for yi in year_infos},
        "summary": summary,
        "normalization_bounds": [b for yi in year_infos
                                 for b in yi["normalization_bounds"]],
        "constraint_counts_full_grid": constraints["counts_full_grid"],
        "constraint_union_full_grid": constraints["counts_union_full_grid"],
        "frozen_parameters": {
            "norm_percentiles": list(S.NORM_PERCENTILES),
            "need_weights": S.NEED_WEIGHTS, "opportunity_weights": S.OPPORTUNITY_WEIGHTS,
            "road_band": S.ROAD_BAND, "green_proximity_cap_m": S.GREEN_PROXIMITY_CAP_M,
            "landuse_eligibility": S.LANDUSE_ELIGIBILITY,
            "class_edges": list(S.CLASS_EDGES), "class_labels": S.CLASS_LABELS,
            "zone_rule": {"min_class": S.ZONE_MIN_CLASS,
                          "min_zone_pixels": S.MIN_ZONE_PIXELS, "connectivity": "8"},
            "priority_tiers": {"High": "class 4 (Very High)", "Medium": "class 3 (High)",
                               "Low": "class 2 (Medium)"},
            "score_nodata": S.SCORE_NODATA, "class_nodata": S.CLASS_NODATA,
            "attribution_precedence": list(PRECEDENCE),
        },
        "model_lineage": p6_manifest.get("primary_model"),
        "terminology_note": S.TERMINOLOGY_NOTE,
        "software_versions": {"python": platform.python_version()},
        "status": "success",
    }
    dump_json(record, out_root / "phase7_pipeline_record.json")
    manifest = {
        "stage": "phase7_plantation_suitability",
        "created_at_utc": record["started_utc"],
        "scenarios": list(SCENARIOS),
        "scenario_definitions": {
            "v1_parity": "no constraint exclusion (V1-comparable)",
            "v2_constrained": "water/buildings/road-surfaces excluded (V2 owns "
                              "these layers; V1 documented their absence)"},
        "inputs": [{"path": str(phase6_dir / "phase6_manifest.json"),
                    "sha256": sha256_file(phase6_dir / "phase6_manifest.json")}
                   ] + [{"path": str(constraints_dir / f),
                         "sha256": sha256_file(constraints_dir / f)}
                        for f in CONSTRAINT_FILES.values()],
        "years": [int(y) for y in years],
        "per_year": summary,
        "notes": [
            "Formulation is a VERBATIM port of the frozen V1 suitability spec "
            "(baseline Scenario A: gated product Need x Opportunity / 100).",
            "Domain = per-year Phase 6 severity_score valid pixels (V1 locked "
            "decision #1).",
            "Class-relative per-year normalization (p1/p99): snapshots, not trends.",
            "Tree-count requirement estimation is Phase 8 - NOT built here.",
        ],
    }
    dump_json(manifest, out_root / "phase7_manifest.json")
    log(f"total {record['total_wall_s']:.1f}s; manifest + record written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
