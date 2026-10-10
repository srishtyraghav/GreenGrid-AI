"""V2 Phase 7 build v3 — plantation suitability over the FULL valid study area.

Three explicitly separate components (a hot pixel != plantable; a plantable
pixel != high priority):

  1. COOLING NEED (0-1): LST-led, INDEPENDENT of Phase 6. Fixed POOLED
     references (p1/p99 of the pooled 2022-2026 W4 valid pixels per
     variable) — never per-year min-max. Formula (documented):
       cooling_need = 0.55*n_pool(LST) + 0.25*n_pool(1-NDVI) + 0.20*n_pool(NDBI)
     (V1 need backbone with the Phase-6-severity term — itself LST-trained —
     replaced by direct pooled LST; V1's non-severity weights kept and
     renormalized.)
  2. PLANTING FEASIBILITY: HARD mask, independent of any score:
     eligible landuse (frozen rule: excl. industrial 5 / retail 7; nodata 255
     neutral-eligible) AND NOT water AND NOT buildings AND NOT road surfaces
     (Phase 2/3 constraint rasters). vegetation_cover < 0.30 is a DOCUMENTED
     planting-space criterion used by Phase 8 zone formation (and the
     scoring's planting-headroom term covers vegetation via NDVI) — it is NOT
     part of this feasibility mask. Exclusion accounting per reason per year.
  3. SUITABILITY (0-1) over the full valid domain: V1 opportunity weights
     VERBATIM rebased to 0-1 (0.30*landuse_eligibility + 0.25*(1-n(NDBI)) +
     0.15*road_band + 0.15*green_proximity + 0.15*(1-n(NDVI))), multiplicatively
     gated by cooling need (V1's gated product, rebased):
       suitability = cooling_need * opportunity.
  4. PRIORITY (separate from suitability): priority = 0.5*cooling_need +
     0.5*suitability, computed ONLY over feasible land (excluded pixels carry
     no priority). Classes by FIXED pooled terciles of the priority score
     over feasible land (pooled across all 5 years): Low < t1 <= Medium <
     t2 <= High. Medium stays eligible everywhere; no High-only gate exists.
     Because every reference is fixed, class maps are directly comparable
     across years (relative-to-pooled-reference — not absolute inter-annual
     heat claims).

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
    dump_json,
    sha256_file,
)
from ..phase4.assemble_features import load_phase3_year, load_static
from . import suitability as S
from .constraints import CONSTRAINT_FILES, build_constraint_stack
from .zones_stable import build_stable_zones

PHASE3_ROOT_DEFAULT = PROJECT_ROOT / "data" / "phase3"
CONSTRAINTS_DIR_DEFAULT = PROJECT_ROOT / "data" / "phase2" / "constraints"
OUT_DEFAULT = PROJECT_ROOT / "data" / "phase7"

SCENARIOS = ("v1_parity", "v2_constrained")
DISCOURAGED_LU_CODES = (5, 7)          # frozen V1 eligibility rule
POOL_SAMPLE_PER_YEAR = 200_000

LU_NAMES = {0: "untagged", 1: "park", 2: "forest", 3: "grass", 4: "commercial",
            5: "industrial", 6: "residential", 7: "retail", 8: "farmland",
            255: "nodata"}


def log(msg: str) -> None:
    print(f"[P7] {msg}", flush=True)


def read_band(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
        nodata = ds.nodata
    return arr, nodata


def write_full(arr, path, griddef, dtype, nodata, descr):
    import rasterio as rio
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = griddef.profile(count=1, dtype=np.dtype(dtype).name, nodata=nodata)
    with rio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
        dst.set_band_description(1, descr)
    return {"path": str(path), "bytes": path.stat().st_size}


def eligible_lu_grid(lu_codes: np.ndarray) -> np.ndarray:
    eligible = np.ones(256, dtype=bool)
    for c in DISCOURAGED_LU_CODES:
        eligible[int(c)] = False
    return eligible[lu_codes.astype(np.int64)]


def preflight(years, phase3_root, constraints_dir, grid_file):
    for y in years:
        if not (phase3_root / str(y)).is_dir():
            raise FileNotFoundError(f"missing phase3 year: {y}")
    for name in ("landuse_raster_30m.tif", "roads_distance_30m.tif",
                 "vegetation_distance_30m.tif"):
        if not (phase3_root / "static" / name).exists():
            raise FileNotFoundError(f"missing phase3 static: {name}")
    for f in CONSTRAINT_FILES.values():
        if not (constraints_dir / f).exists():
            raise FileNotFoundError(f"missing constraint layer: {f}")
    grid = Grid.from_file(Path(grid_file))
    assert (grid.height, grid.width) == (1768, 1874), "authoritative grid mismatch"


def load_year_arrays(phase3_root, year):
    d = load_phase3_year(phase3_root, year)
    out = {k: d[k] for k in ("lst_C", "ndvi", "ndbi", "vegetation_cover")}
    out["domain"] = (np.isfinite(out["lst_C"]) & np.isfinite(out["ndvi"])
                     & np.isfinite(out["ndbi"]) & np.isfinite(out["vegetation_cover"]))
    return out


def compute_pooled_references(years, phase3_root, seed=42):
    """Fixed p1/p99 per variable over the POOLED 2022-2026 valid pixels."""
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ("lst", "ndvi", "ndbi", "veg")}
    for y in years:
        arr = load_year_arrays(phase3_root, y)
        dom = arr["domain"]
        for k, src in (("lst", "lst_C"), ("ndvi", "ndvi"),
                       ("ndbi", "ndbi"), ("veg", "vegetation_cover")):
            vals = arr[src][dom]
            take = min(POOL_SAMPLE_PER_YEAR, vals.size)
            samples[k].append(rng.choice(vals, size=take, replace=False))
        del arr
    refs = {}
    for k in samples:
        pooled = np.concatenate(samples[k])
        p1, p99 = S.pooled_p1_p99(pooled)
        refs[k] = {"p1": p1, "p99": p99, "n_samples": int(pooled.size)}
    refs["one_minus_ndvi"] = dict(refs["ndvi"])
    log(f"pooled references: {json.dumps({k: (round(v['p1'],4), round(v['p99'],4)) for k,v in refs.items()})}")
    return refs


def score_year(year, refs, ctx):
    """Full-grid 0-1 scores for one year (all components)."""
    arr = load_year_arrays(ctx["phase3_root"], year)
    dom = arr["domain"]
    n_lst = S.pooled_normalize(arr["lst_C"], refs["lst"]["p1"], refs["lst"]["p99"])
    n_ndvi = S.pooled_normalize(arr["ndvi"], refs["ndvi"]["p1"], refs["ndvi"]["p99"])
    n_ndbi = S.pooled_normalize(arr["ndbi"], refs["ndbi"]["p1"], refs["ndbi"]["p99"])
    n_1mndvi = S.pooled_normalize(1.0 - arr["ndvi"],
                                  refs["one_minus_ndvi"]["p1"],
                                  refs["one_minus_ndvi"]["p99"])
    need = S.cooling_need(n_lst, n_1mndvi, n_ndbi)
    lu = ctx["lu_codes"]
    static = {"landuse": lu, "dist_road_m": ctx["dist_road_m"],
              "dist_vegetation_m": ctx["dist_vegetation_m"]}
    opp = S.opportunity_component_01(
        {"ndvi": n_ndvi, "ndbi": n_ndbi}, static)
    suit = S.suitability_product(need, opp)
    feasible = ctx["eligible_lu"] & ~ctx["constraint"] & dom
    priority = np.full(dom.shape, np.nan)
    priority[feasible] = (S.PRIORITY_WEIGHTS["cooling_need"] * need[feasible]
                          + S.PRIORITY_WEIGHTS["suitability"] * suit[feasible])
    return {"year": year, "domain": dom, "feasible": feasible,
            "need": need, "opportunity": opp, "suitability": suit,
            "priority": priority, "raw": arr}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase3-root", default=str(PHASE3_ROOT_DEFAULT))
    ap.add_argument("--constraints-dir", default=str(CONSTRAINTS_DIR_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--years", default=",".join(str(y) for y in YEARS))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    ap.add_argument("--skip-constraint-burn", action="store_true")
    args = ap.parse_args(argv)

    years = tuple(int(y) for y in args.years.split(","))
    phase3_root = Path(args.phase3_root)
    constraints_dir = Path(args.constraints_dir)
    out_root = Path(args.out)
    t_start = time.perf_counter()

    preflight(years, phase3_root, constraints_dir, Path(args.grid_file))
    grid = Grid.from_file(Path(args.grid_file))
    log(f"grid {grid.width}x{grid.height}; years={years}; v3 full-area methodology")

    burn_dir = out_root / "constraints_rasters"
    if args.skip_constraint_burn and (burn_dir / "constraint_attributed_30m.tif").exists():
        masks = {}
        for name in CONSTRAINT_FILES:
            with rasterio.open(burn_dir / f"constraint_{name}_30m.tif") as ds:
                masks[name] = ds.read(1).astype(bool)
        with rasterio.open(burn_dir / "constraint_attributed_30m.tif") as ds:
            attributed = ds.read(1)
        constraints = {"masks": masks, "attributed": attributed}
        log("constraint stack reloaded")
    else:
        constraints = build_constraint_stack(constraints_dir, grid, burn_dir)

    static = load_static(phase3_root)
    lu_codes = np.where(np.isfinite(static["landuse_class"]),
                        static["landuse_class"], 255.0).astype(np.int64)
    ctx = {
        "phase3_root": phase3_root, "grid": grid,
        "lu_codes": lu_codes,
        "eligible_lu": eligible_lu_grid(lu_codes),
        "dist_road_m": static["dist_road_m"],
        "dist_vegetation_m": static["dist_vegetation_m"],
        "constraint": constraints["attributed"] > 0,
        "constraint_masks": constraints["masks"],
    }

    refs = compute_pooled_references(years, phase3_root)
    scored = {y: score_year(y, refs, ctx) for y in years}

    # FIXED priority thresholds: pooled terciles of the priority score over
    # feasible land, pooled across ALL years.
    rng = np.random.default_rng(42)
    pr_samples = []
    for y in years:
        pv = scored[y]["priority"][scored[y]["feasible"]]
        pr_samples.append(rng.choice(pv, size=min(400_000, pv.size), replace=False))
    t1, t2 = (float(v) for v in np.quantile(np.concatenate(pr_samples), [1 / 3, 2 / 3]))
    sel_p90 = float(np.quantile(np.concatenate(pr_samples), 0.90))
    nd_samples = []
    for y in years:
        nv = scored[y]["need"][scored[y]["feasible"]]
        nd_samples.append(rng.choice(nv, size=min(400_000, nv.size), replace=False))
    nd_p75 = float(np.quantile(np.concatenate(nd_samples), 0.75))
    log(f"fixed priority bands (pooled terciles over feasible land): t1={t1:.4f} t2={t2:.4f}; "
        f"selection constants: priority_p90={sel_p90:.4f} need_p75={nd_p75:.4f}")

    per_year = {}
    input_hashes = {}
    for y in years:
        sc = scored[y]
        dom, feas = sc["domain"], sc["feasible"]
        cls = S.classify_priority(sc["priority"], t1, t2)
        cls = S.classify_priority(sc["priority"].astype(np.float32).astype(np.float64), t1, t2)
        per_year[y] = {"domain_px": int(dom.sum()), "feasible_px": int(feas.sum()),
                       "class_px": {int(c): int((cls[feas] == c).sum()) for c in (0, 1, 2)}}
        for scenario in SCENARIOS:
            sdir = out_root / scenario
            (sdir / "rasters").mkdir(parents=True, exist_ok=True)
            (sdir / "tables").mkdir(parents=True, exist_ok=True)
            tag = scenario
            if scenario == "v2_constrained":
                feas_s = feas            # feasibility already excludes constraints
                pr_s = sc["priority"]
            else:
                # v1_parity: feasibility WITHOUT constraint exclusion (V1-comparable)
                feas_s = ctx["eligible_lu"] & dom
                pr_s = np.full(dom.shape, np.nan)
                pr_s[feas_s] = (S.PRIORITY_WEIGHTS["cooling_need"] * sc["need"][feas_s]
                                + S.PRIORITY_WEIGHTS["suitability"] * sc["suitability"][feas_s])
            # classify from the float32-rounded score so the written class
            # raster is the exact threshold application of the written
            # priority_score raster (single source of truth)
            cls_s = S.classify_priority(pr_s.astype(np.float32).astype(np.float64), t1, t2)
            write_full(np.where(dom, sc["need"], -1).astype(np.float32),
                       sdir / "rasters" / f"cooling_need_{tag}_{y}.tif",
                       grid, np.float32, -1.0, "cooling need 0-1 (pooled LST-led)")
            write_full(np.where(dom, sc["suitability"], -1).astype(np.float32),
                       sdir / "rasters" / f"suitability_{tag}_{y}.tif",
                       grid, np.float32, -1.0, "suitability 0-1 (gated product)")
            write_full(np.where(dom, sc["opportunity"], -1).astype(np.float32),
                       sdir / "rasters" / f"plantation_opportunity_{tag}_{y}.tif",
                       grid, np.float32, -1.0, "opportunity component 0-1 (V1 weights)")
            pr_write = np.where(feas_s, pr_s, -1).astype(np.float32)
            write_full(pr_write, sdir / "rasters" / f"priority_score_{tag}_{y}.tif",
                       grid, np.float32, -1.0, "priority score 0-1 (feasible land only)")
            cls_write = np.full(dom.shape, 255, dtype=np.uint8)
            cls_write[feas_s] = cls_s[feas_s].astype(np.uint8)
            write_full(cls_write, sdir / "rasters" / f"priority_class_{tag}_{y}.tif",
                       grid, np.uint8, 255, "0 Low / 1 Medium / 2 High (fixed pooled bands)")
            feas_write = np.full(dom.shape, 255, dtype=np.uint8)
            feas_write[dom] = feas_s[dom].astype(np.uint8)
            write_full(feas_write, sdir / "rasters" / f"feasibility_mask_{tag}_{y}.tif",
                       grid, np.uint8, 255, "1 feasible / 0 not (hard mask)")

            # tables
            n_feas = int(feas_s.sum())
            cls_counts = {int(c): int((cls_s[feas_s] == c).sum()) for c in (0, 1, 2)}
            shares = pd.DataFrame([
                {"year": y, "scenario": scenario,
                 "class": S.PRIORITY_CLASS_LABELS[c], "class_value": c,
                 "pixel_count": cls_counts[c],
                 "area_ha": round(cls_counts[c] * 0.09, 1),
                 "pct_of_feasible": round(100.0 * cls_counts[c] / n_feas, 3)}
                for c in (0, 1, 2)])
            shares.to_csv(sdir / "tables" / f"class_shares_{tag}_{y}.csv", index=False)
            stats = pd.DataFrame([{
                "year": y, "scenario": scenario,
                "domain_px": int(dom.sum()), "feasible_px": n_feas,
                "feasible_ha": round(n_feas * 0.09, 1),
                "mean_cooling_need": round(float(np.nanmean(
                    np.where(feas_s, sc["need"], np.nan))), 4),
                "mean_suitability_feasible": round(float(np.nanmean(
                    np.where(feas_s, sc["suitability"], np.nan))), 4),
                "mean_priority_feasible": round(float(np.nanmean(
                    np.where(feas_s, pr_s, np.nan))), 4),
            }])
            stats.to_csv(sdir / "tables" / f"full_area_stats_{tag}_{y}.csv", index=False)
            # exclusion accounting per reason (within domain; precedence
            # water > buildings > road_surfaces > landuse for attribution)
            excl_rows = []
            remaining = dom.copy()
            for name, layer in (("water", ctx["constraint_masks"]["water"]),
                                ("buildings", ctx["constraint_masks"]["buildings"]),
                                ("road_surfaces", ctx["constraint_masks"]["road_surfaces"])):
                if scenario == "v2_constrained":
                    hit = remaining & layer
                    excl_rows.append({"year": y, "scenario": scenario, "reason": name,
                                      "area_ha": round(int(hit.sum()) * 0.09, 1)})
                    remaining &= ~layer
                else:
                    excl_rows.append({"year": y, "scenario": scenario, "reason": name,
                                      "area_ha": 0.0})
            lu_hit = remaining & ~ctx["eligible_lu"]
            excl_rows.append({"year": y, "scenario": scenario, "reason": "landuse_ineligible",
                              "area_ha": round(int(lu_hit.sum()) * 0.09, 1)})
            veg_hit = (remaining & ctx["eligible_lu"]
                       & ~(np.isfinite(sc["raw"]["vegetation_cover"])
                           & (sc["raw"]["vegetation_cover"] < 0.30)))
            excl_rows.append({"year": y, "scenario": scenario,
                              "reason": "planting_space_veg_ge_0.30",
                              "area_ha": round(int(veg_hit.sum()) * 0.09, 1)})
            # FEASIBLE row = planting space after ALL listed filters (the
            # rows partition the domain); landuse/constraints-only feasible
            # land is reported separately in full_area_stats
            plant_space = remaining & ctx["eligible_lu"] \
                & np.isfinite(sc["raw"]["vegetation_cover"]) \
                & (sc["raw"]["vegetation_cover"] < 0.30)
            excl_rows.append({"year": y, "scenario": scenario, "reason": "FEASIBLE",
                              "area_ha": round(int(plant_space.sum()) * 0.09, 1)})
            pd.DataFrame(excl_rows).to_csv(
                sdir / "tables" / f"exclusion_accounting_{tag}_{y}.csv", index=False)
        log(f"{y}: domain={int(dom.sum()):,} feasible_v2={int(feas.sum()):,} "
            f"classes_v2={ {S.PRIORITY_CLASS_LABELS[c]: per_year[y]['class_px'][c] for c in (0,1,2)} }")

    # ---- per-year candidate SITES with stable persistent IDs (S2 gates) ----
    # Sites form from FEASIBLE land (eligible landuse, ~constraints) that also
    # passes the S2 gates (priority >= pooled-p90 AND need >= pooled-p75);
    # usable area (veg<0.30 planting-space) is computed in Phase 8.
    # ---- STABLE ACADEMIC SHORTLIST: one fixed zone set from pooled evidence ----
    ctx["pooled_refs"] = refs
    ctx["sel_p90"] = sel_p90
    ctx["sel_nd75"] = nd_p75
    ctx["static"] = static
    ctx["lu_codes"] = lu_codes
    ctx["year_products"] = {y: load_phase3_year(phase3_root, y) for y in years}
    stable = build_stable_zones(scored, ctx, years, grid, out_root)
    log(f"2026 sites: {stable['zones']['v2_constrained']['n_sites']} candidates | "
        f"{stable['zones']['v2_constrained']['n_shortlist']} shortlist "
        f"({stable['zones']['v2_constrained']['shortlist_ha']:,.0f} ha)")

    record = {
        "phase": 7, "variant": "v3-full-area-pooled", "project": "GreenGrid-AI",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "years": list(years), "scenarios": list(SCENARIOS),
        "total_wall_s": time.perf_counter() - t_start,
        "pooled_references": refs,
        "priority_thresholds": {"t1_medium": t1, "t2_high": t2,
                                "rule": "pooled terciles of priority score over "
                                        "feasible land, pooled across all years"},
        "selection_constants": {
            "priority_p90": sel_p90, "need_p75": nd_p75,
            "source": "pooled v2_constrained feasible-land distributions "
                      "(priority p90 / cooling-need p75), fixed constants",
            "status": ("DOCUMENTED OPERATIONAL SELECTION CHOICES derived from "
                       "the pooled score distributions (multiplier IQR 0.18; "
                       "99.1% pass-through of the Medium gate; mega-patch "
                       "aggregation), PENDING FIELD VALIDATION - not "
                       "validated ecological cutoffs"),
        },
        "formulas": {
            "cooling_need": "0.55*n_pool(LST) + 0.25*n_pool(1-NDVI) + 0.20*n_pool(NDBI)",
            "opportunity": "0.30*landuse_eligibility + 0.25*(1-n_pool(NDBI)) + "
                           "0.15*road_band + 0.15*green_proximity + 0.15*(1-n_pool(NDVI))",
            "suitability": "cooling_need * opportunity (gated product)",
            "priority": "0.5*cooling_need + 0.5*suitability (feasible land only)",
            "classes": "Low < t1 <= Medium < t2 <= High (fixed pooled terciles)",
        },
        "feasibility_rule": "eligible landuse (excl. 5 industrial, 7 retail; nodata "
                            "neutral-eligible) AND NOT water/buildings/road_surfaces; "
                            "veg<0.30 is a Phase-8 planting-space criterion, not a "
                            "feasibility filter",
        "per_year": {str(y): per_year[y] for y in years},
        "sites_2026": {
            "method": ("2026-only candidate sites: feasible & S2 gates "
                       "(priority>=0.5926, need>=0.7116) & veg<0.30; MMU>=2ha; "
                       "max 57 ha; score = 0.50 need + 0.25 veg_deficit + "
                       "0.25 opportunity (0-100); shortlist = top 100 by score, "
                       "oversized excluded"),
            "constants": stable["constants"],
            "per_scenario": stable["zones"],
        },
        "constraint_counts_full_grid": {n: int(m.sum()) for n, m in constraints["masks"].items()},
        "software_versions": {"python": platform.python_version()},
        "status": "success",
    }
    dump_json(record, out_root / "phase7_pipeline_record.json")
    manifest = {
        "stage": "phase7_plantation_suitability",
        "variant": "v3-full-area-pooled",
        "created_at_utc": record["started_utc"],
        "scenarios": list(SCENARIOS),
        "scenario_definitions": {
            "v1_parity": "feasibility without constraint exclusion (V1-comparable)",
            "v2_constrained": "feasibility excludes water/buildings/road-surfaces"},
        "inputs": [{"path": str(constraints_dir / f),
                    "sha256": sha256_file(constraints_dir / f)}
                   for f in CONSTRAINT_FILES.values()],
        "notes": [
            "FULL valid study area scored; three separate components (cooling "
            "need / feasibility / suitability) and a separate priority layer.",
            "All normalizations use FIXED pooled 2022-2026 references; class "
            "maps are cross-year comparable (relative-to-pooled-reference, "
            "not absolute heat claims).",
            "Independent of Phase 6. Zone formation (planting-land patches) is "
            "Phase 8. Tree counts appear only in Phase 8 as planning-density "
            "scenarios (never 'optimal').",
        ],
    }
    dump_json(manifest, out_root / "phase7_manifest.json")
    log(f"total {record['total_wall_s']:.1f}s; manifest + record written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
