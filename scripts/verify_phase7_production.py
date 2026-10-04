"""Gate G3 verification for the Phase 7 PRODUCTION build.

Independently recomputes every frozen formula (implemented inline here, NOT
imported from the runner) and checks the phase7_production outputs:

  1. class raster == classify(score raster, CLASS_EDGES) at >= 1,000,000
     sampled px/year (full domain when the domain is smaller - 2024/2025).
  2. Zone accounting: sum(zone ha) == clustered_px * 900/1e4 within 0.1%,
     with clustered px re-derived from the class raster by independent
     labelling; zone IDs unique; polygons valid.
  3. Need / Opportunity / baseline-S formula recompute on sampled px
     (max deviation < 1e-4), incl. p1/p99 bounds vs normalization_parameters.
  4. Scenario B/C/D recompute matches the written rasters (< 1e-4).
  5. Grid conformance: every raster == reference grid (CRS EPSG:4326,
     transform, width/height); nodata conventions; all 5 years present.

Writes data/processed/phase7_production/verification_g3.json and prints
PASS/FAIL.  Exit code 0 on PASS, 1 on FAIL.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/verify_phase7_production.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from models.config import REFERENCE_RASTER  # noqa: E402
from suitability.config import (  # noqa: E402  (frozen constants only)
    CLASS_EDGES,
    GREEN_PROXIMITY_CAP_M,
    LANDUSE_ELIGIBILITY,
    LANDUSE_NODATA_ELIGIBILITY,
    NEED_WEIGHTS,
    NORM_PERCENTILES,
    OPPORTUNITY_WEIGHTS,
    ROAD_BAND,
    SCENARIO_D_NEED_WEIGHTS,
    SCENARIO_D_OPPORTUNITY_WEIGHTS,
    SCENARIO_EXPONENTS,
)

YEARS = (2022, 2023, 2024, 2025, 2026)
OUT_ROOT = PROJECT / "data" / "processed" / "phase7_production"
RASTERS_DIR = OUT_ROOT / "rasters"
ZONES_DIR = OUT_ROOT / "zones"
TABLES_DIR = OUT_ROOT / "tables"
G3_JSON = OUT_ROOT / "verification_g3.json"

P6_RASTERS = PROJECT / "data" / "processed" / "phase6_production" / "rasters"
P3_ALIGNED = PROJECT / "data" / "processed" / "phase3" / "aligned"
P3_MASKS = PROJECT / "data" / "processed" / "phase3" / "masks"
P4_FEATURES = PROJECT / "data" / "processed" / "phase4" / "features"

PX_AREA_M2 = 900.0
M2_PER_HA = 10_000.0
MIN_ZONE_PIXELS = 10
SCORE_NODATA = -1.0
CLASS_NODATA = -1
EXCLUSION_NODATA = 255
SEED = 42
N_CLASS_SAMPLE = 1_000_000
N_FORMULA_SAMPLE = 200_000
TOL = 1e-4


def log(msg: str) -> None:
    print(f"[G3] {msg}", flush=True)


def read_band(path: Path) -> Tuple[np.ndarray, float]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.profile.get("nodata")
    return arr, (float(nodata) if nodata is not None else np.nan)


# ---------------------------------------------------------------------------
# Independent reimplementations of the frozen math (constants imported only)
# ---------------------------------------------------------------------------
def p1p99(values: np.ndarray) -> Tuple[float, float]:
    lo, hi = NORM_PERCENTILES
    p1, p99 = np.percentile(values, (lo, hi))
    return float(p1), float(p99)


def minmax(values: np.ndarray, p1: float, p99: float) -> np.ndarray:
    return np.clip((np.clip(values, p1, p99) - p1) / (p99 - p1) * 100.0, 0.0, 100.0)


def road_score(d: np.ndarray) -> np.ndarray:
    spec = ROAD_BAND
    out = np.full(d.shape, spec["decline_score"], dtype=np.float64)
    decline = (d > spec["optimal_max_m"]) & (d < spec["decline_max_m"])
    out[decline] = spec["optimal_score"] - (
        (spec["optimal_score"] - spec["decline_score"])
        * (d[decline] - spec["optimal_max_m"])
        / (spec["decline_max_m"] - spec["optimal_max_m"]))
    optimal = (d >= spec["corridor_max_m"]) & (d <= spec["optimal_max_m"])
    out[optimal] = spec["optimal_score"]
    corridor = d < spec["corridor_max_m"]
    out[corridor] = spec["corridor_score_max"] * d[corridor] / spec["corridor_max_m"]
    return out


def green_score(d: np.ndarray) -> np.ndarray:
    return 100.0 * np.clip(1.0 - d / GREEN_PROXIMITY_CAP_M, 0.0, 1.0)


def lu_eligibility(codes: np.ndarray) -> np.ndarray:
    lut = np.full(256, LANDUSE_NODATA_ELIGIBILITY, dtype=np.float64)
    for k, v in LANDUSE_ELIGIBILITY.items():
        lut[int(k)] = v
    return lut[codes.astype(np.int64)]


def classify(score: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(CLASS_EDGES), score, side="left"),
                   0, 4).astype(np.int64)


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self) -> None:
        self.checks: List[Dict] = []

    def add(self, name: str, passed: bool, detail: Dict) -> bool:
        self.checks.append({"check": name, "passed": bool(passed), **detail})
        log(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}")
        return passed

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def grid_check(name: str, path: Path, ref_profile: Dict, ref_transform,
               ref_crs, dtype: str, nodata: float, checker: Checker) -> np.ndarray:
    with rasterio.open(path) as src:
        prof = src.profile
        ok = (prof["width"] == ref_profile["width"]
              and prof["height"] == ref_profile["height"]
              and src.transform == ref_transform
              and src.crs == ref_crs
              and prof["dtype"] == dtype
              and float(prof["nodata"]) == nodata)
        arr = src.read(1)
    checker.add(f"grid:{name}", ok, {
        "path": str(path.relative_to(PROJECT)),
        "crs_ok": str(src.crs) == str(ref_crs),
        "shape": [int(prof["height"]), int(prof["width"])],
        "dtype": prof["dtype"], "nodata": prof.get("nodata"),
    })
    return arr


def sample_indices(n: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    k = min(k, n)
    return rng.choice(n, size=k, replace=False)


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------
def main() -> int:
    t_start = time.time()
    checker = Checker()
    year_results: Dict[str, Dict] = {}

    with rasterio.open(REFERENCE_RASTER) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs

    # ---- per-year checks ----------------------------------------------
    for year in YEARS:
        log(f"=== {year} ===")
        yr: Dict = {}

        # -- raster presence + grid conformance (also loads arrays)
        score = grid_check(f"{year}:suitability", RASTERS_DIR / f"suitability_{year}.tif",
                           ref_profile, ref_transform, ref_crs, "float32",
                           SCORE_NODATA, checker)
        need_r = grid_check(f"{year}:heat_need", RASTERS_DIR / f"heat_need_{year}.tif",
                            ref_profile, ref_transform, ref_crs, "float32",
                            SCORE_NODATA, checker)
        opp_r = grid_check(f"{year}:opportunity", RASTERS_DIR / f"plantation_opportunity_{year}.tif",
                           ref_profile, ref_transform, ref_crs, "float32",
                           SCORE_NODATA, checker)
        cls_r = grid_check(f"{year}:class", RASTERS_DIR / f"suitability_class_{year}.tif",
                           ref_profile, ref_transform, ref_crs, "int16",
                           CLASS_NODATA, checker)
        excl_r = grid_check(f"{year}:exclusion", RASTERS_DIR / f"exclusion_mask_{year}.tif",
                            ref_profile, ref_transform, ref_crs, "uint8",
                            EXCLUSION_NODATA, checker)
        scen_r = {}
        for s in ("b", "c", "d"):
            scen_r[s] = grid_check(
                f"{year}:scenario_{s}",
                RASTERS_DIR / f"suitability_scenario_{s}_{year}.tif",
                ref_profile, ref_transform, ref_crs, "float32",
                SCORE_NODATA, checker)

        domain = excl_r == 0
        n_dom = int(domain.sum())
        yr["domain_px"] = n_dom
        checker.add(f"{year}:exclusion_encodes_domain",
                    bool((excl_r[domain] == 0).all() and n_dom > 0), {
                        "domain_px": n_dom, "excluded_px": int((excl_r == 1).sum())})

        # -- class == classify(score) on sampled px
        flat_score = score[domain].astype(np.float64)
        flat_cls = cls_r[domain].astype(np.int64)
        idx = sample_indices(n_dom, N_CLASS_SAMPLE)
        recomputed = classify(flat_score[idx])
        n_sampled = int(idx.size)
        mismatches = int((recomputed != flat_cls[idx]).sum())
        checker.add(f"{year}:class_matches_score", mismatches == 0, {
            "sampled_px": n_sampled,
            "requested_px": N_CLASS_SAMPLE,
            "full_domain_sampled": n_sampled == n_dom,
            "mismatches": mismatches})

        # -- value ranges
        checker.add(f"{year}:value_ranges",
                    bool(np.isfinite(flat_score).all() and flat_score.min() >= 0.0
                         and flat_score.max() <= 100.0
                         and flat_cls.min() >= 0 and flat_cls.max() <= 4), {
                        "score_range": [float(flat_score.min()), float(flat_score.max())],
                        "class_range": [int(flat_cls.min()), int(flat_cls.max())]})

        # -- zone accounting (independent labelling)
        binary = (cls_r >= 3) & domain
        labels, n_found = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
        sizes = np.bincount(labels.ravel())
        clustered = int(sizes[1:][sizes[1:] >= MIN_ZONE_PIXELS].sum()) if n_found else 0
        geojson_path = ZONES_DIR / f"priority_zones_{year}.geojson"
        gdf = gpd.read_file(geojson_path)
        yr["n_zones"] = int(len(gdf))
        if len(gdf):
            ids_unique = bool(gdf["zone_id"].is_unique)
            geom_valid = bool(gdf.geometry.is_valid.all())
            sum_px = int(gdf["pixel_count"].sum())
            sum_ha = float(gdf["area_ha"].sum())
            per_zone_dev = float(np.max(np.abs(
                gdf["area_ha"].to_numpy(float)
                - gdf["pixel_count"].to_numpy(float) * PX_AREA_M2 / M2_PER_HA)))
        else:
            ids_unique, geom_valid, sum_px, sum_ha, per_zone_dev = True, True, 0, 0.0, 0.0
        expected_ha = clustered * PX_AREA_M2 / M2_PER_HA
        if clustered > 0:
            rel = abs(sum_ha - expected_ha) / expected_ha
            ha_ok = rel <= 0.001
        else:
            rel, ha_ok = 0.0, (sum_ha == 0.0 and sum_px == 0)
        checker.add(f"{year}:zone_accounting",
                    ha_ok and ids_unique and geom_valid and sum_px == clustered, {
                        "clustered_px_independent": clustered,
                        "geojson_px_sum": sum_px,
                        "geojson_ha_sum": round(sum_ha, 4),
                        "expected_ha": round(expected_ha, 4),
                        "rel_err": f"{rel:.2e}",
                        "per_zone_max_ha_dev": f"{per_zone_dev:.2e}",
                        "ids_unique": ids_unique, "geometries_valid": geom_valid,
                        "n_zones": int(len(gdf))})

        # -- normalization params vs tables
        bounds = pd.read_csv(TABLES_DIR / f"normalization_parameters_{year}.csv")
        raw = {
            "severity_score": read_band(P6_RASTERS / f"severity_score_{year}.tif")[0],
            "ndvi": read_band(P3_ALIGNED / f"s2_{year}_ndvi_30m.tif")[0],
            "ndbi": read_band(P3_ALIGNED / f"s2_{year}_ndbi_30m.tif")[0],
            "lst": read_band(P6_RASTERS / f"lst_{year}.tif")[0],
            "vegetation_cover": read_band(P4_FEATURES / f"vegetation_cover_{year}_30m.tif")[0],
        }
        max_bp_dev = 0.0
        bp_ok = True
        for var in ("severity_score", "ndvi", "ndbi", "lst", "vegetation_cover"):
            vals = raw[var][domain].astype(np.float64)
            p1, p99 = p1p99(vals)
            row = bounds[bounds["variable"] == var].iloc[0]
            dev = max(abs(p1 - row["p1"]), abs(p99 - row["p99"]))
            max_bp_dev = max(max_bp_dev, dev)
            bp_ok &= dev < 1e-9 and int(row["n_cells"]) == n_dom
        checker.add(f"{year}:normalization_params", bp_ok, {
            "max_p1p99_dev_vs_table": f"{max_bp_dev:.2e}"})

        # -- Need / Opportunity / S / scenarios recompute on sampled px
        fidx = sample_indices(n_dom, N_FORMULA_SAMPLE)
        flat = {k: v[domain].astype(np.float64) for k, v in raw.items()}
        nrm: Dict[str, np.ndarray] = {}
        for var in ("severity_score", "ndvi", "ndbi", "lst", "vegetation_cover"):
            row = bounds[bounds["variable"] == var].iloc[0]
            nrm[var] = minmax(flat[var], float(row["p1"]), float(row["p99"]))
        # derived quantities normalized from raw 1-x (independent of table)
        nrm["one_minus_ndvi"] = minmax(1.0 - flat["ndvi"], *p1p99(1.0 - flat["ndvi"]))
        nrm["one_minus_vegetation_cover"] = minmax(
            1.0 - flat["vegetation_cover"], *p1p99(1.0 - flat["vegetation_cover"]))

        need_rc = (NEED_WEIGHTS["severity_score"] * nrm["severity_score"]
                   + NEED_WEIGHTS["one_minus_ndvi"] * nrm["one_minus_ndvi"]
                   + NEED_WEIGHTS["ndbi"] * nrm["ndbi"]
                   + NEED_WEIGHTS["lst"] * nrm["lst"])
        lu, _ = read_band(P3_MASKS / "landuse_raster_30m.tif")
        rd, _ = read_band(P3_MASKS / "roads_distance_30m.tif")
        vd, _ = read_band(P3_MASKS / "vegetation_distance_30m.tif")
        opp_rc = (OPPORTUNITY_WEIGHTS["landuse_eligibility"] * lu_eligibility(lu[domain])
                  + OPPORTUNITY_WEIGHTS["built_up_inverse"] * (100.0 - nrm["ndbi"])
                  + OPPORTUNITY_WEIGHTS["road_accessibility"] * road_score(rd[domain])
                  + OPPORTUNITY_WEIGHTS["green_proximity"] * green_score(vd[domain])
                  + OPPORTUNITY_WEIGHTS["planting_headroom"] * (100.0 - nrm["ndvi"]))
        s_rc = np.clip(need_rc * opp_rc / 100.0, 0.0, 100.0)

        dev_need = float(np.max(np.abs(need_rc[fidx] - need_r[domain][fidx].astype(np.float64))))
        dev_opp = float(np.max(np.abs(opp_rc[fidx] - opp_r[domain][fidx].astype(np.float64))))
        dev_s = float(np.max(np.abs(s_rc[fidx] - score[domain][fidx].astype(np.float64))))
        checker.add(f"{year}:need_opp_s_recompute",
                    dev_need < TOL and dev_opp < TOL and dev_s < TOL, {
                        "sampled_px": int(fidx.size),
                        "max_dev_need": f"{dev_need:.2e}",
                        "max_dev_opp": f"{dev_opp:.2e}",
                        "max_dev_suitability": f"{dev_s:.2e}", "tol": TOL})

        need_d = (SCENARIO_D_NEED_WEIGHTS["severity_score"] * nrm["severity_score"]
                  + SCENARIO_D_NEED_WEIGHTS["one_minus_vegetation_cover"]
                  * nrm["one_minus_vegetation_cover"]
                  + SCENARIO_D_NEED_WEIGHTS["ndbi"] * nrm["ndbi"]
                  + SCENARIO_D_NEED_WEIGHTS["lst"] * nrm["lst"])
        opp_d = (SCENARIO_D_OPPORTUNITY_WEIGHTS["landuse_eligibility"]
                 * lu_eligibility(lu[domain])
                 + SCENARIO_D_OPPORTUNITY_WEIGHTS["built_up_inverse"]
                 * (100.0 - nrm["ndbi"])
                 + SCENARIO_D_OPPORTUNITY_WEIGHTS["road_accessibility"]
                 * road_score(rd[domain])
                 + SCENARIO_D_OPPORTUNITY_WEIGHTS["green_proximity"]
                 * green_score(vd[domain])
                 + SCENARIO_D_OPPORTUNITY_WEIGHTS["planting_headroom"]
                 * (100.0 - nrm["vegetation_cover"]))
        scen_rc = {
            "b": np.clip(100.0 ** (1 - sum(SCENARIO_EXPONENTS["B"]))
                         * need_rc ** SCENARIO_EXPONENTS["B"][0]
                         * opp_rc ** SCENARIO_EXPONENTS["B"][1], 0.0, 100.0),
            "c": np.clip(100.0 ** (1 - sum(SCENARIO_EXPONENTS["C"]))
                         * need_rc ** SCENARIO_EXPONENTS["C"][0]
                         * opp_rc ** SCENARIO_EXPONENTS["C"][1], 0.0, 100.0),
            "d": np.clip(need_d * opp_d / 100.0, 0.0, 100.0),
        }
        scen_devs = {}
        scen_ok = True
        for s in ("b", "c", "d"):
            dev = float(np.max(np.abs(
                scen_rc[s][fidx] - scen_r[s][domain][fidx].astype(np.float64))))
            scen_devs[f"max_dev_scenario_{s}"] = f"{dev:.2e}"
            scen_ok &= dev < TOL
        checker.add(f"{year}:scenarios_bcd_recompute", scen_ok, {
            "sampled_px": int(fidx.size), **scen_devs, "tol": TOL})

        # -- tables cross-check: summary table vs class raster
        summ = pd.read_csv(TABLES_DIR / f"suitability_summary_{year}.csv")
        summ_ok = True
        for c in range(5):
            n_table = int(summ.loc[summ["class"] == c, "pixel_count"].iloc[0])
            summ_ok &= n_table == int((flat_cls == c).sum())
        checker.add(f"{year}:summary_table_matches_raster", summ_ok, {})

        # -- ranking table ordering
        ranking = pd.read_csv(TABLES_DIR / f"priority_ranking_{year}.csv")
        if len(ranking):
            ms = ranking["mean_suitability"].to_numpy(float)
            ranking_ok = bool((np.diff(ms) <= 1e-12).all()) and list(
                ranking["rank"]) == list(range(1, len(ranking) + 1))
        else:
            ranking_ok = True
        checker.add(f"{year}:ranking_table_ordered", ranking_ok,
                    {"n_rows": int(len(ranking))})

        yr["mean_suitability"] = float(flat_score.mean())
        yr["class_ge_3_px"] = int((flat_cls >= 3).sum())
        yr["class_ge_3_ha"] = round(yr["class_ge_3_px"] * PX_AREA_M2 / M2_PER_HA, 3)
        year_results[str(year)] = yr

    # ---- global checks ---------------------------------------------------
    all_years = all((RASTERS_DIR / f"suitability_{y}.tif").exists() for y in YEARS)
    checker.add("all_5_years_present", all_years, {"years": list(YEARS)})

    transition_path = TABLES_DIR / "temporal_priority_transition.csv"
    trans_ok = transition_path.exists()
    trans_detail: Dict = {"path": str(transition_path.relative_to(PROJECT))}
    if trans_ok:
        tr = pd.read_csv(transition_path)
        cats = tr["persistence"].value_counts().to_dict()
        matched22 = set(tr.loc[tr["persistence"] == "persistent", "zone_id_2022"].dropna())
        matched26 = set(tr.loc[tr["persistence"] == "persistent", "zone_id_2026"].dropna())
        trans_detail["categories"] = {str(k): int(v) for k, v in cats.items()}
        trans_detail["n_rows"] = int(len(tr))
        # sanity: declining rows carry 2022-only ids, emerging 2026-only ids
        trans_ok = bool(
            tr.loc[tr["persistence"] == "declining", "zone_id_2026"].isna().all()
            and tr.loc[tr["persistence"] == "emerging", "zone_id_2022"].isna().all())
    checker.add("temporal_transition_table", trans_ok, trans_detail)

    n_pass = sum(1 for c in checker.checks if c["passed"])
    result = {
        "gate": "G3",
        "stage": "phase7_production",
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t_start, 3),
        "n_checks": len(checker.checks),
        "n_passed": n_pass,
        "n_failed": len(checker.checks) - n_pass,
        "status": "PASS" if checker.all_passed else "FAIL",
        "sample_note": ("Classify check samples min(1,000,000, domain) px; "
                        "2024/2025 domains are smaller than 1M px, so the "
                        "full domain is sampled there."),
        "years": year_results,
        "checks": checker.checks,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(G3_JSON, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log(f"{result['status']}: {n_pass}/{len(checker.checks)} checks passed "
        f"({result['elapsed_s']}s)")
    if not checker.all_passed:
        for c in checker.checks:
            if not c["passed"]:
                log(f"  FAILED: {c['check']} {c}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
