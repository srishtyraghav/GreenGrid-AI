"""Stage 2 production build: Phase 6 severity products, hotspots, analytics.

Turns the verified (G1) full-grid 178-feature tables into production
severity products for ALL five years (2022-2026) with the frozen 3-class
XGBoost production model:

  1. Predict per year on the full valid domain: class (argmax proba),
     per-class probabilities (Low/Moderate/High), confidence (max proba),
     severity_score = probs @ [0,1,2] in [0,2].
  2. Write full-grid GeoTIFFs (reference-raster profile, LZW, explicit
     NoData) to data/processed/phase6_production/rasters/:
     severity / probability_low|moderate|high / confidence /
     severity_score / lst (copy of the aligned Phase-3 LST composite,
     NaN -> NoData). Valid-domain values are scattered into the full
     1768 x 1874 EPSG:4326 grid.
  3. Hotspots (spec section 8) to .../hotspots/:
       A: severity == 2 (High)
       B: severity == 2 AND confidence >= 0.60
       C: severity_score >= 1.5 AND confidence >= 0.60
     8-connected components >= 10 px; polygonized to GeoJSON with pixel
     count + area ha (900 m^2/px); hotspot-ID rasters + stats CSV per def
     per year (cluster count, total ha, mean/max severity_score, mean
     confidence).
  4. Analytics tables to .../tables/: area_statistics,
     green_built_comparison, vegetation_temperature_relationship,
     temporal_comparison, hotspot_statistics_def{A,B,C}.
  5. Figures to .../figures/: per-year 3-class severity map, per-year
     def-B hotspot map, 5-year temporal comparison panel.
  6. Manifest (input sha256, per-year domain counts, output list, model
     id, spec version) and pipeline record (timings, params, environment).

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/run_phase6_production.py
      [--years 2022,2023,...] [--skip-figures]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from models.config import REFERENCE_RASTER  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (spec: docs_production_build_spec.md)
# ---------------------------------------------------------------------------
YEARS = (2022, 2023, 2024, 2025, 2026)
CLASS_NAMES = ["Low", "Moderate", "High"]          # model column order 0/1/2
SCORE_WEIGHTS = np.array([0.0, 1.0, 2.0], dtype=np.float64)

OUT_ROOT = PROJECT / "data" / "processed" / "phase6_production"
FULLGRID_DIR = OUT_ROOT / "fullgrid"
RASTERS_DIR = OUT_ROOT / "rasters"
HOTSPOTS_DIR = OUT_ROOT / "hotspots"
TABLES_DIR = OUT_ROOT / "tables"
FIGURES_DIR = OUT_ROOT / "figures"

MODEL_JSON = PROJECT / "data" / "processed" / "phase5_production_3class" / "phase5_primary_xgb_3class.json"
SCHEMA_JSON = PROJECT / "data" / "processed" / "phase5_production_3class" / "phase5_primary_xgb_3class_features.json"
MODEL_ID = "phase5_primary_xgb_3class"
SPEC_PATH = PROJECT / "docs_production_build_spec.md"
COMBINED_CSV = PROJECT / "data" / "processed" / "lulc_outputs" / "combined_urban_environmental_dataset.csv"
DOMAIN_COUNTS_JSON = FULLGRID_DIR / "domain_counts.json"
G1_JSON = FULLGRID_DIR / "verification_g1.json"

LST_SRC = {y: PROJECT / f"data/processed/phase3/aligned/l9_{y}_composite_lst_30m.tif" for y in YEARS}

SEVERITY_NODATA = -1          # int16 class raster
PROB_NODATA = -1.0            # float32 prob / confidence / score rasters
LST_NODATA = -1.0             # float32 observed LST raster
HOTSPOT_NODATA = -1           # int32 hotspot-ID raster (0 = valid, not hotspot)

MIN_CLUSTER_PX = 10           # spec section 8
CONFIDENCE_T = 0.60
SCORE_T = 1.5
PX_AREA_M2 = 900.0            # 30 m x 30 m
M2_PER_HA = 10_000.0

HOTSPOT_DEFS = {
    "A": "severity == 2 (High)",
    "B": "severity == 2 AND confidence >= 0.60",
    "C": "severity_score >= 1.5 AND confidence >= 0.60",
}

CLASS_COLORS = ["#2c7bb6", "#fdae61", "#d7191c"]  # Low, Moderate, High


def log(msg: str) -> None:
    print(f"[P6] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def load_schema() -> list[str]:
    schema = json.loads(SCHEMA_JSON.read_text())["feature_names"]
    assert len(schema) == 178
    return schema


def reference_profile() -> dict:
    with rasterio.open(REFERENCE_RASTER) as ds:
        profile = ds.profile.copy()
    return profile


def scatter(rows: np.ndarray, cols: np.ndarray, values, dtype, nodata,
            shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, nodata, dtype=dtype)
    grid[rows, cols] = values.astype(dtype) if hasattr(values, "astype") else values
    return grid


def write_grid(grid: np.ndarray, path: Path, base_profile: dict, nodata) -> dict:
    profile = base_profile.copy()
    profile.update({"dtype": grid.dtype.name, "count": 1, "nodata": nodata,
                    "compress": "lzw"})
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(grid, 1)
    return {"path": rel(path), "dtype": grid.dtype.name, "nodata": nodata,
            "shape": list(grid.shape), "bytes": path.stat().st_size}


def hotspot_mask(def_id: str, cls: np.ndarray, conf: np.ndarray,
                 score: np.ndarray) -> np.ndarray:
    if def_id == "A":
        return cls == 2
    if def_id == "B":
        return (cls == 2) & (conf >= CONFIDENCE_T)
    if def_id == "C":
        return (score >= SCORE_T) & (conf >= CONFIDENCE_T)
    raise ValueError(def_id)


def label_clusters(mask: np.ndarray, min_px: int) -> tuple[np.ndarray, np.ndarray]:
    """8-connected component labelling; returns (cluster_id_grid, kept_ids)."""
    structure = np.ones((3, 3), dtype=int)
    lab, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return np.zeros(mask.shape, dtype=np.int32), np.zeros(0, dtype=np.int64)
    sizes = np.bincount(lab.ravel())
    kept_ids = np.nonzero(sizes[1:] >= min_px)[0] + 1
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[kept_ids] = np.arange(1, len(kept_ids) + 1, dtype=np.int32)
    return remap[lab], kept_ids


def polygonize(cluster_grid: np.ndarray, transform) -> dict[int, object]:
    out: dict[int, list] = {}
    for geom, val in shapes(cluster_grid, mask=cluster_grid > 0,
                            transform=transform):
        out.setdefault(int(val), []).append(shape(geom))
    return {cid: unary_union(gs) for cid, gs in out.items()}


def build_hotspots(year: int, def_id: str, cls_grid: np.ndarray,
                   conf_grid: np.ndarray, score_grid: np.ndarray,
                   base_profile: dict, transform, outputs: list[dict]) -> dict:
    mask = hotspot_mask(def_id, cls_grid, conf_grid, score_grid)
    # outside-domain (nodata) pixels must never join a cluster
    mask &= cls_grid >= 0
    cluster_grid, kept = label_clusters(mask, MIN_CLUSTER_PX)
    counts = np.bincount(cluster_grid.ravel())

    geojson_path = HOTSPOTS_DIR / f"hotspots_def{def_id}_{year}.geojson"
    polys = polygonize(cluster_grid, transform)
    import geopandas as gpd
    records = []
    for cid in range(1, len(kept) + 1):
        px = int(counts[cid]) if cid < len(counts) else 0
        records.append({
            "cluster_id": cid,
            "year": year,
            "definition": def_id,
            "definition_rule": HOTSPOT_DEFS[def_id],
            "pixel_count": px,
            "area_ha": round(px * PX_AREA_M2 / M2_PER_HA, 4),
            "geometry": polys[cid],
        })
    if records:
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        gdf.to_file(geojson_path, driver="GeoJSON")
    else:
        geojson_path.write_text(json.dumps(
            {"type": "FeatureCollection", "features": [],
             "crs": {"type": "name",
                     "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}}}))
    outputs.append({"path": rel(geojson_path), "bytes": geojson_path.stat().st_size})

    hot = cluster_grid > 0
    n_hot_px = int(hot.sum())
    stats = {
        "year": year,
        "definition": def_id,
        "definition_rule": HOTSPOT_DEFS[def_id],
        "min_cluster_px": MIN_CLUSTER_PX,
        "n_clusters": int(len(kept)),
        "n_hotspot_pixels": n_hot_px,
        "total_area_ha": round(n_hot_px * PX_AREA_M2 / M2_PER_HA, 4),
        "mean_severity_score": float(score_grid[hot].mean()) if n_hot_px else np.nan,
        "max_severity_score": float(score_grid[hot].max()) if n_hot_px else np.nan,
        "mean_confidence": float(conf_grid[hot].mean()) if n_hot_px else np.nan,
    }
    stats_path = TABLES_DIR / f"hotspot_statistics_def{def_id}_{year}.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    outputs.append({"path": rel(stats_path), "bytes": stats_path.stat().st_size})

    id_path = HOTSPOTS_DIR / f"hotspot_ids_def{def_id}_{year}.tif"
    outputs.append(write_grid(cluster_grid, id_path, base_profile, HOTSPOT_NODATA))
    return stats


def classify_tertiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return np.searchsorted(edges, values, side="right").astype(np.int8), edges


def run_year(year: int, schema: list[str], booster: xgb.Booster,
             base_profile: dict, transform, skip_figures: bool,
             outputs: list[dict], timings: dict) -> dict:
    t0 = time.perf_counter()
    log(f"=== {year} ===")

    parquet_path = FULLGRID_DIR / f"features_{year}.parquet"
    df = pd.read_parquet(parquet_path)
    rows = df["row"].to_numpy(np.int64)
    cols = df["col"].to_numpy(np.int64)
    assert list(df.columns) == ["row", "col"] + schema
    assert not df[schema].isna().any().any()
    n = len(df)

    t = time.perf_counter()
    X = df[schema]
    dmat = xgb.DMatrix(X, feature_names=schema)
    probs = booster.predict(dmat).astype(np.float64)      # (n, 3)
    del dmat, X, df
    cls = np.argmax(probs, axis=1).astype(np.int16)
    conf = probs.max(axis=1)
    score = probs @ SCORE_WEIGHTS
    timings[f"predict_{year}"] = round(time.perf_counter() - t, 3)
    log(f"{year}: predicted {n} px  class hist={np.bincount(cls, minlength=3).tolist()}")

    shape = (base_profile["height"], base_profile["width"])

    # ---- rasters ------------------------------------------------------------
    t = time.perf_counter()
    r = {}
    r["severity"] = write_grid(
        scatter(rows, cols, cls, np.int16, SEVERITY_NODATA, shape),
        RASTERS_DIR / f"severity_{year}.tif", base_profile, SEVERITY_NODATA)
    for i, name in enumerate(CLASS_NAMES):
        r[f"probability_{name.lower()}"] = write_grid(
            scatter(rows, cols, probs[:, i], np.float32, PROB_NODATA, shape),
            RASTERS_DIR / f"probability_{name.lower()}_{year}.tif",
            base_profile, PROB_NODATA)
    r["confidence"] = write_grid(
        scatter(rows, cols, conf, np.float32, PROB_NODATA, shape),
        RASTERS_DIR / f"confidence_{year}.tif", base_profile, PROB_NODATA)
    r["severity_score"] = write_grid(
        scatter(rows, cols, score, np.float32, PROB_NODATA, shape),
        RASTERS_DIR / f"severity_score_{year}.tif", base_profile, PROB_NODATA)

    with rasterio.open(LST_SRC[year]) as src:
        lst = src.read(1)
    lst_valid = np.isfinite(lst)
    lst_grid = np.where(lst_valid, lst, LST_NODATA).astype(np.float32)
    r["lst"] = write_grid(lst_grid, RASTERS_DIR / f"lst_{year}.tif",
                          base_profile, LST_NODATA)
    outputs.extend(r.values())
    timings[f"rasters_{year}"] = round(time.perf_counter() - t, 3)

    cls_grid = np.full(shape, SEVERITY_NODATA, dtype=np.int16)
    cls_grid[rows, cols] = cls
    conf_grid = np.full(shape, PROB_NODATA, dtype=np.float32)
    conf_grid[rows, cols] = conf
    score_grid = np.full(shape, PROB_NODATA, dtype=np.float32)
    score_grid[rows, cols] = score

    # ---- hotspots ------------------------------------------------------------
    t = time.perf_counter()
    hot_stats = {}
    for def_id in ("A", "B", "C"):
        hot_stats[def_id] = build_hotspots(
            year, def_id, cls_grid, conf_grid, score_grid,
            base_profile, transform, outputs)
        log(f"{year}: hotspots def {def_id}: {hot_stats[def_id]['n_clusters']} clusters, "
            f"{hot_stats[def_id]['total_area_ha']:.1f} ha")
    timings[f"hotspots_{year}"] = round(time.perf_counter() - t, 3)

    # ---- analytics tables ------------------------------------------------------
    t = time.perf_counter()
    hist = np.bincount(cls, minlength=3)
    area_rows = []
    for i, name in enumerate(CLASS_NAMES):
        area_rows.append({
            "year": year, "class": name, "class_value": i,
            "pixels": int(hist[i]),
            "area_ha": round(hist[i] * PX_AREA_M2 / M2_PER_HA, 4),
            "pct_of_valid_domain": round(100.0 * hist[i] / n, 6),
        })
    p = TABLES_DIR / f"area_statistics_{year}.csv"
    pd.DataFrame(area_rows).to_csv(p, index=False)
    outputs.append({"path": rel(p), "bytes": p.stat().st_size})

    # green vs built comparison (full-grid domain values)
    veg = pd.read_parquet(parquet_path, columns=["vegetation_cover", "ndbi"])
    gb_rows = []
    for axis in ("vegetation_cover", "ndbi"):
        vals = veg[axis].to_numpy(np.float64)
        tert, edges = classify_tertiles(vals)
        v_min, v_max = float(vals.min()), float(vals.max())
        ranges = [(v_min, float(edges[0])), (float(edges[0]), float(edges[1])),
                  (float(edges[1]), v_max)]
        for t_i in range(3):
            m = tert == t_i
            lo, hi = ranges[t_i]
            gb_rows.append({
                "year": year, "axis": axis, "tertile": t_i + 1,
                "range_lo": lo, "range_hi": hi,
                "n_pixels": int(m.sum()),
                "area_ha": round(float(m.sum()) * PX_AREA_M2 / M2_PER_HA, 4),
                "mean_severity_score": float(score[m].mean()),
                "high_fraction": float((cls[m] == 2).mean()),
            })
    p = TABLES_DIR / f"green_built_comparison_{year}.csv"
    pd.DataFrame(gb_rows).to_csv(p, index=False)
    outputs.append({"path": rel(p), "bytes": p.stat().st_size})

    # vegetation-temperature relationship (fixed decile bins of [0,1])
    vc = veg["vegetation_cover"].to_numpy(np.float32)
    del veg
    dec = np.minimum((vc * 10.0).astype(np.int32), 9)
    lst_at_domain = lst_grid[rows, cols]
    vt_rows = []
    for d in range(10):
        m = dec == d
        lst_m = lst_at_domain[m]
        lst_finite = np.isfinite(lst_m)
        vt_rows.append({
            "year": year, "vegetation_cover_decile": d + 1,
            "vc_range_lo": round(d / 10.0, 1), "vc_range_hi": round((d + 1) / 10.0, 1),
            "n_pixels": int(m.sum()),
            "mean_lst_c": float(lst_m[lst_finite].mean()) if lst_finite.any() else np.nan,
            "mean_severity_score": float(score[m].mean()),
        })
    p = TABLES_DIR / f"vegetation_temperature_relationship_{year}.csv"
    pd.DataFrame(vt_rows).to_csv(p, index=False)
    outputs.append({"path": rel(p), "bytes": p.stat().st_size})

    lst_dom = lst_at_domain[np.isfinite(lst_at_domain)]
    year_info = {
        "year": year,
        "domain_pixels": n,
        "grid_pixels": int(shape[0] * shape[1]),
        "lst_finite_pixels": int(lst_valid.sum()),
        "lst_finite_in_domain": int(lst_dom.size),
        "class_histogram": {"Low": int(hist[0]), "Moderate": int(hist[1]),
                            "High": int(hist[2])},
        "class_pct": {name: round(100.0 * hist[i] / n, 4)
                      for i, name in enumerate(CLASS_NAMES)},
        "mean_lst_c_domain": float(lst_dom.mean()) if lst_dom.size else np.nan,
        "mean_severity_score": float(score.mean()),
        "hotspots": hot_stats,
        "rasters": r,
    }
    timings[f"tables_{year}"] = round(time.perf_counter() - t, 3)

    if not skip_figures:
        t = time.perf_counter()
        make_figures(year, cls_grid, conf_grid, score_grid, transform,
                     hot_stats["B"], base_profile)
        timings[f"figures_{year}"] = round(time.perf_counter() - t, 3)

    timings[f"total_{year}"] = round(time.perf_counter() - t0, 3)
    log(f"{year}: done in {timings[f'total_{year}']:.1f}s")
    return year_info


def grid_extent(transform, height: int, width: int) -> tuple[float, float, float, float]:
    left = transform.c
    top = transform.f
    right = left + width * transform.a
    bottom = top + height * transform.e
    return left, right, bottom, top


def make_figures(year: int, cls_grid: np.ndarray, conf_grid: np.ndarray,
                 score_grid: np.ndarray, transform, hot_b: dict,
                 base_profile: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    ext = grid_extent(transform, base_profile["height"], base_profile["width"])
    cmap = ListedColormap(CLASS_COLORS)

    fig, ax = plt.subplots(figsize=(8, 8))
    data = np.ma.masked_where(cls_grid < 0, cls_grid)
    ax.imshow(data, cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest",
              extent=ext, aspect="equal")
    ax.set_title(f"Delhi UHI severity (3-class) — {year}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(handles=[mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
                       for i in range(3)],
              loc="lower right", title="Severity")
    fig.tight_layout()
    p = FIGURES_DIR / f"severity_map_{year}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)

    mask = hotspot_mask("B", cls_grid, conf_grid, score_grid)
    mask &= cls_grid >= 0
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.ma.masked_where(cls_grid < 0, cls_grid),
              cmap=ListedColormap(["#d9d9d9", "#d9d9d9", "#d9d9d9"]),
              vmin=-0.5, vmax=2.5, interpolation="nearest", extent=ext)
    hot = np.ma.masked_where(~mask, mask)
    ax.imshow(hot, cmap=ListedColormap(["#d7191c"]), vmin=0, vmax=1,
              interpolation="nearest", extent=ext, alpha=0.9)
    ax.set_title(f"Delhi UHI hotspots (def B: High & confidence ≥ 0.60) — {year}\n"
                 f"{hot_b['n_clusters']} clusters ≥ {MIN_CLUSTER_PX} px, "
                 f"{hot_b['total_area_ha']:.1f} ha")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(handles=[mpatches.Patch(color="#d7191c", label="Hotspot (def B)"),
                       mpatches.Patch(color="#d9d9d9", label="Valid domain")],
              loc="lower right")
    fig.tight_layout()
    p = FIGURES_DIR / f"hotspot_defB_map_{year}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_temporal_figure(year_infos: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = [yi["year"] for yi in year_infos]
    low = [yi["class_pct"]["Low"] for yi in year_infos]
    mod = [yi["class_pct"]["Moderate"] for yi in year_infos]
    high = [yi["class_pct"]["High"] for yi in year_infos]
    ha_b = [yi["hotspots"]["B"]["total_area_ha"] for yi in year_infos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.bar(years, low, label="Low", color=CLASS_COLORS[0])
    ax1.bar(years, mod, bottom=low, label="Moderate", color=CLASS_COLORS[1])
    ax1.bar(years, high, bottom=[l + m for l, m in zip(low, mod)],
            label="High", color=CLASS_COLORS[2])
    ax1.set_title("UHI severity class share of valid domain (%)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("% of valid domain")
    ax1.set_xticks(years)
    ax1.legend(loc="upper right")
    for x, t in zip(years, [yi["domain_pixels"] for yi in year_infos]):
        ax1.annotate(f"{t/1e6:.2f}M px", (x, 101), ha="center", fontsize=8,
                     color="#555555", annotation_clip=False)
    ax1.set_ylim(0, 112)

    ax2.plot(years, ha_b, marker="o", color="#d7191c", linewidth=2)
    for x, v in zip(years, ha_b):
        ax2.annotate(f"{v:,.0f}", (x, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    ax2.margins(y=0.18)
    ax2.set_title("Def-B hotspot area (High & confidence ≥ 0.60)")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Hotspot area (ha)")
    ax2.set_xticks(years)
    ax2.grid(alpha=0.3)

    fig.suptitle("Delhi UHI severity — 5-year production comparison (2022–2026)")
    fig.tight_layout()
    p = FIGURES_DIR / "temporal_comparison.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default=None, help="comma-separated subset of years")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    years = YEARS if not args.years else tuple(int(y) for y in args.years.split(","))

    for d in (RASTERS_DIR, HOTSPOTS_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    schema = load_schema()
    booster = xgb.Booster()
    booster.load_model(str(MODEL_JSON))
    base_profile = reference_profile()
    transform = base_profile["transform"]
    log(f"reference grid {base_profile['height']} x {base_profile['width']} "
        f"crs={base_profile['crs']}")

    timings: dict[str, float] = {}
    outputs: list[dict] = []
    year_infos: list[dict] = []
    for year in years:
        info = run_year(year, schema, booster, base_profile, transform,
                        args.skip_figures, outputs, timings)
        year_infos.append(info)

    # ---- temporal comparison table + figure -----------------------------------
    trows = []
    for yi in year_infos:
        trows.append({
            "year": yi["year"],
            "valid_domain_pixels": yi["domain_pixels"],
            "low_pct": yi["class_pct"]["Low"],
            "moderate_pct": yi["class_pct"]["Moderate"],
            "high_pct": yi["class_pct"]["High"],
            "hotspot_defB_area_ha": yi["hotspots"]["B"]["total_area_ha"],
            "hotspot_defB_clusters": yi["hotspots"]["B"]["n_clusters"],
            "mean_lst_c_domain": round(yi["mean_lst_c_domain"], 4),
            "mean_severity_score": round(yi["mean_severity_score"], 6),
        })
    p = TABLES_DIR / "temporal_comparison.csv"
    pd.DataFrame(trows).to_csv(p, index=False)
    outputs.append({"path": rel(p), "bytes": p.stat().st_size})

    if not args.skip_figures:
        make_temporal_figure(year_infos)
    fig_files = sorted(FIGURES_DIR.glob("*.png"))
    outputs.extend({"path": rel(f), "bytes": f.stat().st_size} for f in fig_files)

    # ---- manifest ---------------------------------------------------------------
    input_files = [FULLGRID_DIR / f"features_{y}.parquet" for y in years]
    input_files += [MODEL_JSON, SCHEMA_JSON, SPEC_PATH, DOMAIN_COUNTS_JSON,
                    G1_JSON, COMBINED_CSV, REFERENCE_RASTER]
    input_files += [LST_SRC[y] for y in years]
    spec_sha = sha256(SPEC_PATH)
    manifest = {
        "stage": "phase6_production_stage2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_sha256": sha256(MODEL_JSON),
        "spec": {"path": rel(SPEC_PATH), "sha256": spec_sha,
                 "version": "2026-10-03 production build spec"},
        "schema": {"path": rel(SCHEMA_JSON), "n_features": 178,
                   "sha256": sha256(SCHEMA_JSON)},
        "inputs": [{"path": rel(f), "sha256": sha256(f)} for f in input_files],
        "years": [int(y) for y in years],
        "domain_counts": {str(yi["year"]): {
            "domain_pixels": yi["domain_pixels"],
            "grid_pixels": yi["grid_pixels"],
            "lst_finite_pixels": yi["lst_finite_pixels"],
            "lst_finite_in_domain": yi["lst_finite_in_domain"],
        } for yi in year_infos},
        "class_distribution": {str(yi["year"]): yi["class_pct"] for yi in year_infos},
        "hotspot_definitions": HOTSPOT_DEFS,
        "min_cluster_px": MIN_CLUSTER_PX,
        "pixel_area_m2": PX_AREA_M2,
        "nodata": {"severity": SEVERITY_NODATA, "probability": PROB_NODATA,
                   "confidence": PROB_NODATA, "severity_score": PROB_NODATA,
                   "lst": LST_NODATA, "hotspot_ids": HOTSPOT_NODATA},
        "notes": [
            "2024/2025 valid domains are small (0.58M / 0.31M px) due to monsoon "
            "cloud holes in the source composites; this is real, not an error.",
            "Domain = pixels where all 178 features are finite (mirrors training "
            "dropna); LST rasters are written wherever LST is finite (separate count).",
            "Every table number is reproducible from the rasters/parquets: hotspot "
            "areas = pixel_count * 900 m2 / 1e4; class counts from severity rasters.",
        ],
        "outputs": outputs,
    }
    mp = OUT_ROOT / "phase6_production_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2))
    log(f"wrote {rel(mp)}")

    record = {
        "stage": "phase6_production_stage2",
        "run_at_utc": manifest["created_at_utc"],
        "script": rel(Path(__file__)),
        "timings_s": timings,
        "params": {
            "model_id": MODEL_ID,
            "n_features": 178,
            "score_weights": SCORE_WEIGHTS.tolist(),
            "hotspot_defs": HOTSPOT_DEFS,
            "min_cluster_px": MIN_CLUSTER_PX,
            "confidence_threshold": CONFIDENCE_T,
            "score_threshold": SCORE_T,
            "pixel_area_m2": PX_AREA_M2,
            "years": [int(y) for y in years],
            "raster_dtypes": {"severity": "int16",
                              "probability/confidence/score": "float32",
                              "lst": "float32", "hotspot_ids": "int32"},
            "compress": "lzw",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": (os_cpu_count() or "unknown"),
            "packages": env_versions(),
        },
    }
    rp = OUT_ROOT / "phase6_pipeline_record.json"
    rp.write_text(json.dumps(record, indent=2))
    log(f"wrote {rel(rp)}")
    log("done")
    return 0


def os_cpu_count():
    try:
        return int(__import__("os").cpu_count() or 0)
    except Exception:
        return None


def env_versions() -> dict:
    import importlib
    out = {}
    for m in ("numpy", "pandas", "pyarrow", "rasterio", "xgboost", "scipy",
              "shapely", "geopandas", "matplotlib"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            out[m] = "missing"
    return out


if __name__ == "__main__":
    sys.exit(main())
