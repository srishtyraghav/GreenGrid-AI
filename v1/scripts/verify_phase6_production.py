"""Gate G2 verification: Phase 6 production rasters / hotspots / analytics.

Checks (spec: docs_production_build_spec.md, gate G2; stage-2 build by
scripts/run_phase6_production.py):

  1. GRID / CRS / NODATA: every raster in
     data/processed/phase6_production/rasters/ matches the reference grid
     shape (read from REFERENCE_RASTER), CRS EPSG:4326, and the expected
     nodata (-1 int16 severity / -1.0 float32 everything else).
  2. DOMAIN / HISTOGRAM: per year, the severity class histogram sums to the
     stage-1 domain count and the number of non-nodata pixels equals it.
  3. PROBABILITIES / SCORE: at domain pixels the three probability rasters
     are in [0,1] and their row sums deviate from 1 by < 1e-5; confidence
     in [0,1]; severity_score in [0,2].
  4. ANCHOR: >= 10,000 training-table pixels sampled across years and
     spatial blocks; at each pixel the severity raster value equals
     argmax(booster(fullgrid parquet features)) (stage-1 parity lookup,
     sampled). Chained with stage-1's recorded
     predictions_identical_to_training_table (all 749,998 sampled pixels),
     this proves full-grid severity == training-table predictions.
  5. LST: lst_{year}.tif mean within 0.5 degC of the source aligned LST
     mean over valid pixels.
  6. HOTSPOTS: hotspot-ID rasters are reproducible from severity /
     confidence / severity_score rasters via the spec section-8 definitions
     (8-connectivity, >= 10 px); stats CSV pixel counts and GeoJSON
     per-cluster pixel counts match the ID rasters.

Writes data/processed/phase6_production/verification_g2.json and prints a
PASS/FAIL summary. Exit code 0 iff every check passes.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/verify_phase6_production.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from scipy import ndimage

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from models.config import REFERENCE_RASTER  # noqa: E402

OUT_ROOT = PROJECT / "data" / "processed" / "phase6_production"
FULLGRID_DIR = OUT_ROOT / "fullgrid"
RASTERS_DIR = OUT_ROOT / "rasters"
HOTSPOTS_DIR = OUT_ROOT / "hotspots"
TABLES_DIR = OUT_ROOT / "tables"
MODEL_JSON = PROJECT / "data" / "processed" / "phase5_production_3class" / "phase5_primary_xgb_3class.json"
SCHEMA_JSON = PROJECT / "data" / "processed" / "phase5_production_3class" / "phase5_primary_xgb_3class_features.json"
DOMAIN_COUNTS_JSON = FULLGRID_DIR / "domain_counts.json"
G1_JSON = FULLGRID_DIR / "verification_g1.json"
COMBINED_CSV = PROJECT / "data" / "processed" / "lulc_outputs" / "combined_urban_environmental_dataset.csv"
LST_SRC = {y: PROJECT / f"data/processed/phase3/aligned/l9_{y}_composite_lst_30m.tif"
           for y in (2022, 2023, 2024, 2025, 2026)}
YEARS = (2022, 2023, 2024, 2025, 2026)

SEVERITY_NODATA = -1
PROB_NODATA = -1.0
LST_NODATA = -1.0
HOTSPOT_NODATA = -1
MIN_CLUSTER_PX = 10
CONFIDENCE_T = 0.60
SCORE_T = 1.5
PROB_SUM_TOL = 1e-5
LST_MEAN_TOL = 0.5
ANCHOR_N_MIN = 10_000
ANCHOR_QUOTA_TOTAL = 11_000

SEVERITY_RASTER = {y: RASTERS_DIR / f"severity_{y}.tif" for y in YEARS}
PROB_RASTER = {(y, k): RASTERS_DIR / f"probability_{k}_{y}.tif"
               for y in YEARS for k in ("low", "moderate", "high")}
CONF_RASTER = {y: RASTERS_DIR / f"confidence_{y}.tif" for y in YEARS}
SCORE_RASTER = {y: RASTERS_DIR / f"severity_score_{y}.tif" for y in YEARS}
LST_RASTER = {y: RASTERS_DIR / f"lst_{y}.tif" for y in YEARS}
HOTSPOT_ID_RASTER = {(y, d): HOTSPOTS_DIR / f"hotspot_ids_def{d}_{y}.tif"
                     for y in YEARS for d in "ABC"}
HOTSPOT_GEOJSON = {(y, d): HOTSPOTS_DIR / f"hotspots_def{d}_{y}.geojson"
                   for y in YEARS for d in "ABC"}
HOTSPOT_STATS_CSV = {(y, d): TABLES_DIR / f"hotspot_statistics_def{d}_{y}.csv"
                     for y in YEARS for d in "ABC"}


def log(msg: str) -> None:
    print(f"[G2] {msg}", flush=True)


def read_grid(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        meta = {"shape": (ds.height, ds.width), "crs": ds.crs.to_string(),
                "nodata": ds.nodata, "dtype": ds.dtypes[0]}
    return arr, meta


def relabel(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=int)
    lab, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return np.zeros(mask.shape, dtype=np.int32)
    sizes = np.bincount(lab.ravel())
    kept = np.nonzero(sizes[1:] >= MIN_CLUSTER_PX)[0] + 1
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[kept] = np.arange(1, len(kept) + 1, dtype=np.int32)
    return remap[lab]


def hotspot_mask(def_id: str, cls: np.ndarray, conf: np.ndarray,
                 score: np.ndarray) -> np.ndarray:
    if def_id == "A":
        return (cls == 2) & (cls >= 0)
    if def_id == "B":
        return (cls == 2) & (conf >= CONFIDENCE_T) & (cls >= 0)
    if def_id == "C":
        return (score >= SCORE_T) & (conf >= CONFIDENCE_T) & (cls >= 0)
    raise ValueError(def_id)


def check_grid_meta(year: int, failures: list[str]) -> None:
    files = [SEVERITY_RASTER[year], PROB_RASTER[(year, "low")],
             PROB_RASTER[(year, "moderate")], PROB_RASTER[(year, "high")],
             CONF_RASTER[year], SCORE_RASTER[year], LST_RASTER[year]]
    for f in files:
        if not f.exists():
            failures.append(f"missing raster {f}")
            continue
        with rasterio.open(f) as ds:
            if (ds.height, ds.width) != REF_SHAPE:
                failures.append(f"{f.name}: shape {(ds.height, ds.width)} != {REF_SHAPE}")
            if ds.crs.to_string() != "EPSG:4326":
                failures.append(f"{f.name}: crs {ds.crs} != EPSG:4326")
            expected = (SEVERITY_NODATA if "severity_" in f.name and "score" not in f.name
                        else LST_NODATA if f.name.startswith("lst_") else PROB_NODATA)
            if ds.nodata != expected:
                failures.append(f"{f.name}: nodata {ds.nodata} != {expected}")


def main() -> int:
    failures: list[str] = []
    report: dict = {"gate": "G2",
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "years": list(YEARS), "checks": {}}

    global REF_SHAPE
    with rasterio.open(REFERENCE_RASTER) as ds:
        REF_SHAPE = (ds.height, ds.width)
    log(f"reference grid {REF_SHAPE[0]} x {REF_SHAPE[1]}")
    domain_counts = json.loads(DOMAIN_COUNTS_JSON.read_text())

    schema = json.loads(SCHEMA_JSON.read_text())["feature_names"]
    booster = xgb.Booster()
    booster.load_model(str(MODEL_JSON))

    # ---- 1. grid / crs / nodata ------------------------------------------------
    for year in YEARS:
        check_grid_meta(year, failures)
    report["checks"]["grid_meta"] = {
        "passed": not any("shape" not in f and "crs" not in f and "nodata" not in f
                          for f in failures),
        "reference_shape": list(REF_SHAPE),
        "expected_crs": "EPSG:4326",
        "expected_nodata": {"severity": SEVERITY_NODATA,
                            "probability/confidence/score": PROB_NODATA,
                            "lst": LST_NODATA},
    }
    log("grid/crs/nodata check done")

    # ---- 2+3+5. per-year raster content ----------------------------------------
    per_year: dict[str, dict] = {}
    for year in YEARS:
        info: dict = {}
        sev, _ = read_grid(SEVERITY_RASTER[year])
        dom = sev >= 0
        info["n_valid_pixels"] = int(dom.sum())
        info["domain_expected"] = int(domain_counts[str(year)]["domain_pixels"])
        hist = np.bincount(sev[dom].astype(np.int64), minlength=3)
        info["class_histogram"] = hist.tolist()
        info["histogram_sums_to_domain"] = bool(int(hist.sum()) == info["domain_expected"]
                                                == info["n_valid_pixels"])
        if not info["histogram_sums_to_domain"]:
            failures.append(f"{year}: histogram sum {int(hist.sum())} / valid px "
                            f"{info['n_valid_pixels']} != domain {info['domain_expected']}")

        pl, _ = read_grid(PROB_RASTER[(year, "low")])
        pm, _ = read_grid(PROB_RASTER[(year, "moderate")])
        ph, _ = read_grid(PROB_RASTER[(year, "high")])
        conf, _ = read_grid(CONF_RASTER[year])
        score, _ = read_grid(SCORE_RASTER[year])
        prob_sum = pl[dom].astype(np.float64) + pm[dom].astype(np.float64) + ph[dom].astype(np.float64)
        info["prob_rowsum_max_dev"] = float(np.max(np.abs(prob_sum - 1.0)))
        info["prob_min"] = float(min(pl[dom].min(), pm[dom].min(), ph[dom].min()))
        info["prob_max"] = float(max(pl[dom].max(), pm[dom].max(), ph[dom].max()))
        info["confidence_range"] = [float(conf[dom].min()), float(conf[dom].max())]
        info["severity_score_range"] = [float(score[dom].min()), float(score[dom].max())]
        ok = (info["prob_rowsum_max_dev"] < PROB_SUM_TOL
              and info["prob_min"] >= 0.0 and info["prob_max"] <= 1.0
              and conf[dom].min() >= 0.0 and conf[dom].max() <= 1.0
              and score[dom].min() >= -1e-6 and score[dom].max() <= 2.0 + 1e-6)
        info["passed"] = bool(ok)
        if not ok:
            failures.append(f"{year}: probability/score bounds failed: {info}")

        with rasterio.open(LST_SRC[year]) as src:
            lst_src = src.read(1)
        lst_out, _ = read_grid(LST_RASTER[year])
        src_mean = float(np.nanmean(lst_src))
        out_mean = float(lst_out[lst_out != LST_NODATA].mean())
        info["lst_mean_source"] = src_mean
        info["lst_mean_output"] = out_mean
        info["lst_mean_abs_diff"] = abs(src_mean - out_mean)
        info["lst_valid_px_source"] = int(np.isfinite(lst_src).sum())
        info["lst_valid_px_output"] = int((lst_out != LST_NODATA).sum())
        if info["lst_mean_abs_diff"] > LST_MEAN_TOL:
            failures.append(f"{year}: LST mean diff {info['lst_mean_abs_diff']} > {LST_MEAN_TOL}")
        if info["lst_valid_px_source"] != info["lst_valid_px_output"]:
            failures.append(f"{year}: LST valid px {info['lst_valid_px_output']} != "
                            f"source {info['lst_valid_px_source']}")

        per_year[str(year)] = info
        log(f"{year}: content checks done "
            f"(rowsum dev {info['prob_rowsum_max_dev']:.2e}, "
            f"lst diff {info['lst_mean_abs_diff']:.2e} C)")

    report["checks"]["raster_content"] = per_year

    # ---- 4. anchor: sampled training pixels vs raster severity -----------------
    log("sampling training pixels for anchor check ...")
    samples = pd.read_csv(COMBINED_CSV, usecols=["row", "col", "year", "spatial_block_id"])
    grouped = samples.groupby(["year", "spatial_block_id"])
    quota = int(np.ceil(ANCHOR_QUOTA_TOTAL / len(grouped)))
    rng = np.random.default_rng(42)
    parts = []
    for (y, b), g in grouped:
        take = min(len(g), quota)
        idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
        parts.append(samples.loc[idx])
    samp = pd.concat(parts).reset_index(drop=True)
    n_sampled = len(samp)
    log(f"sampled {n_sampled} pixels across {len(grouped)} year x block combos "
        f"(quota {quota}/combo)")
    if n_sampled < ANCHOR_N_MIN:
        failures.append(f"anchor sample {n_sampled} < {ANCHOR_N_MIN}")

    mismatches = 0
    missing = 0
    checked = 0
    per_year_anchor: dict[str, dict] = {}
    for year in YEARS:
        sub = samp[samp.year == year][["row", "col"]]
        pq = pd.read_parquet(FULLGRID_DIR / f"features_{year}.parquet")
        merged = sub.merge(pq, on=["row", "col"], how="left", validate="one_to_one")
        del pq
        n_miss = int(merged[schema].isna().any(axis=1).sum())
        missing += n_miss
        good = merged.dropna(subset=schema)
        dmat = xgb.DMatrix(good[schema], feature_names=schema)
        pred = np.argmax(booster.predict(dmat), axis=1).astype(np.int64)
        del dmat
        sev, _ = read_grid(SEVERITY_RASTER[year])
        got = sev[good["row"].to_numpy(np.int64), good["col"].to_numpy(np.int64)]
        n_mm = int((got != pred).sum())
        mismatches += n_mm
        checked += len(good)
        per_year_anchor[str(year)] = {"n_sampled": int(len(sub)), "n_checked": int(len(good)),
                                      "n_missing_from_domain": n_miss, "n_mismatch": n_mm}
        log(f"{year}: anchor {len(good)} px, {n_mm} mismatch")
        del merged, good

    g1 = json.loads(G1_JSON.read_text())
    g1_identity = bool(g1["checks"]["parity"]["predictions_identical_to_training_table"])
    anchor_ok = (mismatches == 0 and missing == 0 and checked >= ANCHOR_N_MIN and g1_identity)
    if not anchor_ok:
        failures.append(f"anchor: mismatches={mismatches} missing={missing} "
                        f"checked={checked} g1_identity={g1_identity}")
    report["checks"]["anchor"] = {
        "passed": bool(anchor_ok),
        "n_sampled": int(n_sampled),
        "n_checked": int(checked),
        "n_mismatch": int(mismatches),
        "n_missing_from_domain": int(missing),
        "sampling": f"stratified by year x spatial_block_id ({len(grouped)} combos), "
                    f"{quota}/combo, seed 42",
        "g1_predictions_identical_to_training_table_all_749998_px": g1_identity,
        "per_year": per_year_anchor,
    }

    # ---- 6. hotspots reproducible from rasters ----------------------------------
    hot_report: dict[str, dict] = {}
    for year in YEARS:
        sev, _ = read_grid(SEVERITY_RASTER[year])
        conf, _ = read_grid(CONF_RASTER[year])
        score, _ = read_grid(SCORE_RASTER[year])
        for def_id in "ABC":
            key = f"{def_id}_{year}"
            expected = relabel(hotspot_mask(def_id, sev, conf, score))
            got, _ = read_grid(HOTSPOT_ID_RASTER[(year, def_id)])
            identical = bool(np.array_equal(expected, got))
            counts = np.bincount(got[got > 0].astype(np.int64))
            n_id_clusters = int(counts.size - 1)          # ids are dense 1..K
            min_size = int(counts[1:].min()) if n_id_clusters else 0
            stats_csv = pd.read_csv(HOTSPOT_STATS_CSV[(year, def_id)])
            csv_px = int(stats_csv["n_hotspot_pixels"].iloc[0])
            csv_clusters = int(stats_csv["n_clusters"].iloc[0])
            import geopandas as gpd
            gdf = gpd.read_file(HOTSPOT_GEOJSON[(year, def_id)])
            geo_px = int(gdf["pixel_count"].sum()) if len(gdf) else 0
            geo_clusters = len(gdf)
            ok = (identical and csv_px == int((got > 0).sum()) == geo_px
                  and csv_clusters == geo_clusters == n_id_clusters
                  and min_size >= MIN_CLUSTER_PX)
            hot_report[key] = {
                "passed": bool(ok), "id_raster_matches_recompute": identical,
                "n_clusters": csv_clusters, "n_hotspot_pixels": csv_px,
                "geojson_pixels": geo_px, "min_cluster_size": min_size,
            }
            if not ok:
                failures.append(f"hotspots {key}: {hot_report[key]}")
        log(f"{year}: hotspot reproducibility done")
    report["checks"]["hotspots"] = hot_report

    overall = not failures
    report["overall_passed"] = overall
    report["failures"] = failures

    out_json = OUT_ROOT / "verification_g2.json"
    out_json.write_text(json.dumps(report, indent=2))
    log(f"wrote {out_json}")
    print("=" * 60)
    print(f"G2 {'PASS' if overall else 'FAIL'}")
    for year in YEARS:
        iy = per_year[str(year)]
        print(f"  {year}: domain {iy['n_valid_pixels']:>9,} px | "
              f"class% L/M/H {iy['class_histogram'][0]/iy['n_valid_pixels']*100:5.2f}/"
              f"{iy['class_histogram'][1]/iy['n_valid_pixels']*100:5.2f}/"
              f"{iy['class_histogram'][2]/iy['n_valid_pixels']*100:5.2f} | "
              f"rowsum dev {iy['prob_rowsum_max_dev']:.2e} | "
              f"LST diff {iy['lst_mean_abs_diff']:.2e} C")
    print(f"  anchor: {checked:,} px checked, {mismatches} mismatch "
          f"(G1 all-pixel identity: {g1_identity})")
    if failures:
        print("  failures:")
        for f in failures:
            print(f"    - {f}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
