"""V2 Phase 6 — UHI severity mapping with the frozen primary model.

Applies the FROZEN V2 Phase 5 primary production model (Random Forest, per
the marker ``v2/data/phase5/phase5_primary_model.json`` — loaded via the
marker, never hardcoded) to the full valid 30 m grid for 2022–2026 and
builds the production severity products (output taxonomy mirrored from
``v1/scripts/run_phase6_production.py``, adapted to V2 paths and the frozen
3-class target):

  rasters/    severity_{year}.tif        uint8 0/1/2, nodata 255 (argmax)
              probability_{year}.tif     float32 3-band Low/Moderate/High,
                                         band sums = 1, nodata -1.0
              confidence_{year}.tif      float32 max-prob, nodata -1.0
              severity_score_{year}.tif  float32 probs @ [0,1,2], nodata -1.0
  hotspots/   hotspots_def{A,B,C}_{year}.geojson  polygons (>= 10 px clusters)
              hotspot_ids_def{A,B,C}_{year}.tif    int32 ids, nodata -1
  tables/     area_statistics_{year}.csv      class distribution (+area, %)
              block_statistics_{year}.csv   per spatial block: class hist +
                                            per-def hotspot px/ha
              hotspot_statistics_def{A,B,C}_{year}.csv  per-def summary
              temporal_comparison.csv     5-year summary
              coverage_statistics.csv     vs Phase-3 masks + Phase-2 gates
  phase6_manifest.json / phase6_pipeline_record.json

Hotspot tiers are a faithful port of V1's spec-section-8 definitions:
  A: severity == 2 (High)
  B: severity == 2 AND confidence >= 0.60
  C: severity_score >= 1.5 AND confidence >= 0.60
8-connected components >= 10 px; outside-domain pixels never join a cluster.
Deviations vs V1, by design: probabilities are written as ONE 3-band raster
(the brief's "3-band class-probability rasters"); no LST copy rasters
(Phase 3 owns LST); no figures; green/built & vegetation-temperature tables
omitted (not requested). The model is a DIRECT 3-class classifier — no LST
thresholding step (thresholds_by_year is provenance only).

Feature construction reuses the Phase 4 code path
(``v2.phase4.assemble_features`` / ``v2.phase4._spatial``) at FULL grid
(sampling off), so mapped features are schema-identical to the training
tables; the frozen 178-name schema + sha256 are asserted before predicting.

Usage:
    PYTHONPATH=v2/src .venv/Scripts/python.exe -m v2.phase6.build \
        [--years 2022,2023,...] [--out v2/data/phase6]
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from ..common import (
    GRID_FILE_DEFAULT,
    HOLDOUT_BLOCKS,
    MET_CSV_DEFAULT,
    PROJECT_ROOT,
    SCHEMA_JSON_DEFAULT,
    STUDY_AREA_DEFAULT,
    YEARS,
    Grid,
    compute_block_raster,
    dump_json,
    load_frozen_schema,
    schema_hash,
    sha256_file,
)
from ..phase4.assemble_features import (
    DIST_COLS,
    build_year_columns,
    load_phase3_year,
    load_static,
    predictor_col_order,
)
from ..phase4._spatial import encode_predictors, met_fields_for_year

CLASS_NAMES = ["Low", "Moderate", "High"]          # model column order 0/1/2
SCORE_WEIGHTS = np.array([0.0, 1.0, 2.0], dtype=np.float64)

SEVERITY_NODATA = 255          # uint8 class raster
PROB_NODATA = -1.0             # float32 prob / confidence / score
HOTSPOT_NODATA = -1            # int32 hotspot-ID raster (0 = valid, not hotspot)

MIN_CLUSTER_PX = 10            # V1 spec section 8
CONFIDENCE_T = 0.60
SCORE_T = 1.5
PX_AREA_M2 = 900.0             # 30 m x 30 m
M2_PER_HA = 10_000.0

HOTSPOT_DEFS = {
    "A": "severity == 2 (High)",
    "B": "severity == 2 AND confidence >= 0.60",
    "C": "severity_score >= 1.5 AND confidence >= 0.60",
}

PHASE5_DIR_DEFAULT = PROJECT_ROOT / "data" / "phase5"
PHASE3_ROOT_DEFAULT = PROJECT_ROOT / "data" / "phase3"
OUT_DEFAULT = PROJECT_ROOT / "data" / "phase6"

REQUIRED_PHASE3 = ("lst_30m.tif", "ndvi_30m.tif", "ndbi_30m.tif", "ndre_30m.tif",
                   "vegetation_cover_30m.tif", "ndmi_30m.tif", "mndwi_30m.tif",
                   "bsi_30m.tif", "lulc_30m.tif")
REQUIRED_MASKS = ("valid_l9_30m.tif", "valid_s2_10m_30m.tif",
                  "valid_s2_20m_30m.tif", "valid_lulc_30m.tif")
REQUIRED_STATIC = ("landuse_raster_30m.tif", "roads_distance_30m.tif",
                   "vegetation_distance_30m.tif", "buildings_distance_30m.tif")


def log(msg: str) -> None:
    print(f"[P6] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Marker-driven model resolution (never hardcode the model)
# ---------------------------------------------------------------------------

def resolve_primary_model(phase5_dir: Path = PHASE5_DIR_DEFAULT):
    """Read the Phase 5 marker and return (marker, artifact_path). Loud failure."""
    marker_path = Path(phase5_dir) / "phase5_primary_model.json"
    if not marker_path.exists():
        raise FileNotFoundError(f"Phase 5 primary marker missing: {marker_path}")
    marker = json.loads(marker_path.read_text())
    if marker.get("primary_model") != "rf":
        raise AssertionError(
            f"marker primary_model is {marker.get('primary_model')!r}, expected 'rf'")
    if marker.get("frozen") is not True:
        raise AssertionError("Phase 5 marker is not frozen — refusing to map")
    if marker.get("verification_passed") is not True:
        raise AssertionError(
            "Phase 5 marker verification_passed is not true — run "
            "v2.phase5.verify_rf_primary first")
    artifact = Path(phase5_dir) / marker["artifact"]
    if not artifact.exists():
        raise FileNotFoundError(f"primary model artifact missing: {artifact}")
    return marker, artifact


def preflight(years, phase3_root: Path, phase5_dir: Path, grid_file: Path,
              met_csv: Path, study_area: Path, schema_json: Path) -> None:
    """Fail loudly before any heavy work."""
    resolve_primary_model(phase5_dir)
    for y in years:
        ydir = phase3_root / str(y)
        if not ydir.is_dir():
            raise FileNotFoundError(f"missing phase3 year dir: {ydir}")
        for f in REQUIRED_PHASE3 + REQUIRED_MASKS:
            if not (ydir / f).exists():
                raise FileNotFoundError(f"missing phase3 product: {ydir / f}")
    for f in REQUIRED_STATIC:
        if not (phase3_root / "static" / f).exists():
            raise FileNotFoundError(f"missing phase3 static: {f}")
    for p in (grid_file, met_csv, study_area, schema_json):
        if not Path(p).exists():
            raise FileNotFoundError(f"missing input: {p}")
    grid = Grid.from_file(Path(grid_file))
    assert (grid.height, grid.width) == (1768, 1874), (
        f"authoritative grid must be 1874x1768, got {grid.width}x{grid.height}")
    schema = load_frozen_schema(Path(schema_json))
    assert len(schema) == 178


# ---------------------------------------------------------------------------
# Feature stack (Phase 4 code path, full grid, sampling off)
# ---------------------------------------------------------------------------

def encode_domain_frame(df: pd.DataFrame, predictor_cols: list[str],
                        schema: list[str]) -> tuple[pd.DataFrame, int]:
    """V1 dtype-parity + one-hot encode + frozen-schema reindex. Shared by
    the full-grid mapping (assemble_year_frame) and the unit tests.

    Unseen categorical values (e.g. landuse code 1/3 present in the V2 OSM
    raster but absent from the frozen V1 schema) produce dummies OUTSIDE the
    schema; per the Phase-4 zero-fill convention (CONTRACT B2) they are
    dropped here (those pixels' one-hots zero-fill). Returns (X, n_extra) so
    callers can report the count loudly without crashing the mapping.
    """
    df = df.copy()
    df["landuse_class"] = df["landuse_class"].astype(np.float64)
    df["lulc_class"] = df["lulc_class"].astype(np.int64)
    for c in [c for c in df.columns if c.startswith("landuse_dominant_")]:
        df[c] = df[c].astype(np.int64)
    X, names = encode_predictors(df, predictor_cols)
    extras = [c for c in names if c not in schema]
    X = X.reindex(columns=schema, fill_value=0)
    if list(X.columns) != list(schema):
        raise AssertionError("schema order violation after reindex")
    for c in X.columns:
        if c not in DIST_COLS:
            X[c] = X[c].astype(np.float32)
    return X, len(extras)


def assemble_year_frame(year: int, ctx: dict) -> dict:
    """Full-grid Phase-4 feature stack -> domain subset, schema-exact X.

    Mirrors v2.phase4.assemble_features (same loaders, same derived-feature
    builders, same dtype parity policy) with sampling disabled: the domain is
    every pixel where all raw predictors are finite.
    """
    grid: Grid = ctx["grid"]
    merged = load_phase3_year(ctx["phase3_root"], year)
    merged.update(ctx["static"])
    met = met_fields_for_year(ctx["met_df"], year, ctx["px_lon"], ctx["px_lat"])
    predictor_cols = ctx["predictor_cols"]
    cols = build_year_columns(year, merged, ctx["block_raster"], met,
                              ctx["phase3_root"], predictor_cols)
    del merged
    domain = np.ones(grid.shape, dtype=bool)
    for arr in cols.values():
        domain &= np.isfinite(arr)
    flat = np.flatnonzero(domain.ravel())
    if len(flat) == 0:
        raise AssertionError(f"{year}: empty prediction domain")
    rows = flat // grid.width
    cl = flat % grid.width

    data = {"row": rows.astype(np.int64), "col": cl.astype(np.int64)}
    for name in predictor_cols:
        data[name] = cols[name].ravel()[flat]
    del cols, flat
    gc.collect()
    df = pd.DataFrame(data)
    del data
    X, n_extra = encode_domain_frame(df, predictor_cols, ctx["schema"])
    if n_extra:
        log(f"{year}: WARNING {n_extra} encoded dummy column(s) outside the "
            f"frozen schema (unseen landuse codes) zero-filled per CONTRACT B2")
    out = {"X": X, "domain": domain, "n_extra_dummies": n_extra,
           "rows": df["row"].to_numpy(), "cols": df["col"].to_numpy()}
    del df
    gc.collect()
    return out


def predict_year(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = time.perf_counter()
    # float32 = the precision stored in probability_*.tif; deriving
    # severity/confidence from the SAME rounded values keeps every product
    # internally consistent (float64 argmax can disagree with the stored
    # bands on exact-probability ties, ~3 px per 1.9 M).
    probs = model.predict_proba(X).astype(np.float32)       # (n, 3)
    cls = np.argmax(probs, axis=1).astype(np.uint8)
    log(f"predict: {len(X)} px in {time.perf_counter() - t:.1f}s "
        f"hist={np.bincount(cls, minlength=3).tolist()}")
    return cls, probs


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------

def scatter(shape, rows, cols, values, dtype, nodata) -> np.ndarray:
    grid = np.full(shape, nodata, dtype=dtype)
    grid[rows, cols] = values.astype(dtype) if hasattr(values, "astype") else values
    return grid


def write_single(grid: np.ndarray, path: Path, griddef: Grid, nodata) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = griddef.profile(count=1, dtype=grid.dtype.name, nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(grid, 1)
        dst.set_band_description(1, path.stem)
    return {"path": str(path), "dtype": grid.dtype.name, "nodata": nodata,
            "bytes": path.stat().st_size}


def write_probability(band_grids: list[np.ndarray], path: Path,
                      griddef: Grid, nodata) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = griddef.profile(count=3, dtype="float32", nodata=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        for i, (band, name) in enumerate(zip(band_grids, CLASS_NAMES), start=1):
            dst.write(band.astype(np.float32), i)
            dst.set_band_description(i, f"probability_{name.lower()}")
    return {"path": str(path), "dtype": "float32", "count": 3, "nodata": nodata,
            "bytes": path.stat().st_size}


# ---------------------------------------------------------------------------
# Hotspots (faithful V1 port)
# ---------------------------------------------------------------------------

def hotspot_mask(def_id: str, cls, conf, score) -> np.ndarray:
    if def_id == "A":
        return cls == 2
    if def_id == "B":
        return (cls == 2) & (conf >= CONFIDENCE_T)
    if def_id == "C":
        return (score >= SCORE_T) & (conf >= CONFIDENCE_T)
    raise ValueError(def_id)


def label_clusters(mask: np.ndarray, min_px: int):
    structure = np.ones((3, 3), dtype=int)
    lab, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return np.zeros(mask.shape, dtype=np.int32), np.zeros(0, dtype=np.int64)
    sizes = np.bincount(lab.ravel())
    kept_ids = np.nonzero(sizes[1:] >= min_px)[0] + 1
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[kept_ids] = np.arange(1, len(kept_ids) + 1, dtype=np.int32)
    return remap[lab], kept_ids


def build_hotspots(year: int, def_id: str, cls_grid, conf_grid, score_grid,
                   domain: np.ndarray, griddef: Grid,
                   hotspots_dir: Path, tables_dir: Path) -> dict:
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    mask = hotspot_mask(def_id, cls_grid, conf_grid, score_grid)
    mask &= domain                                # nodata px never join a cluster
    cluster_grid, kept = label_clusters(mask, MIN_CLUSTER_PX)
    # -1 (HOTSPOT_NODATA) everywhere outside the valid domain; 0 = valid but
    # not in any kept cluster.
    cluster_grid = np.where(domain, cluster_grid, HOTSPOT_NODATA).astype(np.int32)
    counts = np.bincount(cluster_grid[cluster_grid > 0].ravel(),
                         minlength=len(kept) + 1)

    geojson_path = hotspots_dir / f"hotspots_def{def_id}_{year}.geojson"
    polys: dict[int, object] = {}
    for geom, val in shapes(cluster_grid, mask=cluster_grid > 0,
                            transform=griddef.transform):
        polys.setdefault(int(val), []).append(shape(geom))
    feats = []
    for cid in range(1, len(kept) + 1):
        px = int(counts[cid]) if cid < len(counts) else 0
        feats.append({
            "type": "Feature",
            "properties": {
                "cluster_id": cid, "year": year, "definition": def_id,
                "definition_rule": HOTSPOT_DEFS[def_id],
                "pixel_count": px,
                "area_ha": round(px * PX_AREA_M2 / M2_PER_HA, 4),
            },
            "geometry": unary_union(polys[cid]).__geo_interface__,
        })
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    geojson_path.write_text(json.dumps(
        {"type": "FeatureCollection", "features": feats}))
    id_path = hotspots_dir / f"hotspot_ids_def{def_id}_{year}.tif"
    write_single(cluster_grid.astype(np.int32), id_path, griddef, HOTSPOT_NODATA)

    hot = cluster_grid > 0
    n_hot = int(hot.sum())
    stats = {
        "year": year, "definition": def_id, "definition_rule": HOTSPOT_DEFS[def_id],
        "min_cluster_px": MIN_CLUSTER_PX,
        "n_clusters": int(len(kept)),
        "n_hotspot_pixels": n_hot,
        "total_area_ha": round(n_hot * PX_AREA_M2 / M2_PER_HA, 4),
        "mean_severity_score": float(score_grid[hot].mean()) if n_hot else None,
        "max_severity_score": float(score_grid[hot].max()) if n_hot else None,
        "mean_confidence": float(conf_grid[hot].mean()) if n_hot else None,
    }
    pd.DataFrame([stats]).to_csv(
        tables_dir / f"hotspot_statistics_def{def_id}_{year}.csv", index=False)
    log(f"{year} hotspots {def_id}: {stats['n_clusters']} clusters, "
        f"{stats['total_area_ha']:.1f} ha")
    return {"stats": stats, "cluster_grid": cluster_grid}


# ---------------------------------------------------------------------------
# Year pipeline
# ---------------------------------------------------------------------------

def run_year(year: int, model, ctx: dict, out_root: Path) -> dict:
    t0 = time.perf_counter()
    grid: Grid = ctx["grid"]
    log(f"=== {year} ===")
    rasters_dir = out_root / "rasters"
    hotspots_dir = out_root / "hotspots"
    tables_dir = out_root / "tables"

    frame = assemble_year_frame(year, ctx)
    X, domain, rows, cols = frame["X"], frame["domain"], frame["rows"], frame["cols"]
    n_extra_dummies = frame["n_extra_dummies"]
    n = len(rows)
    cls, probs = predict_year(model, X)
    del X
    gc.collect()
    conf = probs.max(axis=1)
    score = probs @ SCORE_WEIGHTS

    shape = grid.shape
    cls_grid = scatter(shape, rows, cols, cls, np.uint8, SEVERITY_NODATA)
    conf_grid = scatter(shape, rows, cols, conf, np.float32, PROB_NODATA)
    score_grid = scatter(shape, rows, cols, score, np.float32, PROB_NODATA)
    prob_bands = [scatter(shape, rows, cols, probs[:, i], np.float32, PROB_NODATA)
                  for i in range(3)]
    del probs
    outputs = {
        "severity": write_single(cls_grid, rasters_dir / f"severity_{year}.tif",
                                 grid, SEVERITY_NODATA),
        "probability": write_probability(prob_bands, rasters_dir / f"probability_{year}.tif",
                                         grid, PROB_NODATA),
        "confidence": write_single(conf_grid, rasters_dir / f"confidence_{year}.tif",
                                   grid, PROB_NODATA),
        "severity_score": write_single(score_grid, rasters_dir / f"severity_score_{year}.tif",
                                       grid, PROB_NODATA),
    }

    # ---- hotspots ----------------------------------------------------------
    hot = {}
    for def_id in ("A", "B", "C"):
        hot[def_id] = build_hotspots(year, def_id, cls_grid, conf_grid,
                                     score_grid, domain, grid,
                                     hotspots_dir, tables_dir)

    # ---- tables -------------------------------------------------------------
    hist = np.bincount(cls, minlength=3)
    area_rows = [{
        "year": year, "class": name, "class_value": i,
        "pixels": int(hist[i]),
        "area_ha": round(hist[i] * PX_AREA_M2 / M2_PER_HA, 4),
        "pct_of_valid_domain": round(100.0 * hist[i] / n, 6),
    } for i, name in enumerate(CLASS_NAMES)]
    pd.DataFrame(area_rows).to_csv(tables_dir / f"area_statistics_{year}.csv",
                                   index=False)

    # per-block class distribution + hotspot px/ha per def
    block_raster = ctx["block_raster"]
    blk_rows = []
    for bid in range(25):
        bmask = block_raster == bid
        valid_b = bmask & domain
        n_b = int(valid_b.sum())
        if n_b == 0:
            continue
        cls_b = cls_grid[valid_b]
        rec = {"year": year, "spatial_block_id": bid,
               "valid_pixels": n_b,
               "low_px": int((cls_b == 0).sum()),
               "moderate_px": int((cls_b == 1).sum()),
               "high_px": int((cls_b == 2).sum()),
               "high_pct": round(100.0 * (cls_b == 2).mean(), 4)}
        for def_id in ("A", "B", "C"):
            hp = int((hot[def_id]["cluster_grid"][valid_b] > 0).sum())
            rec[f"hotspot_def{def_id}_px"] = hp
            rec[f"hotspot_def{def_id}_ha"] = round(hp * PX_AREA_M2 / M2_PER_HA, 4)
        blk_rows.append(rec)
    pd.DataFrame(blk_rows).to_csv(tables_dir / f"block_statistics_{year}.csv",
                                  index=False)

    # coverage vs phase-3 masks (+ Phase-2 gates)
    cov = {"year": year, "grid_pixels": int(shape[0] * shape[1]),
           "domain_pixels": n}
    for mask_name in ("valid_l9", "valid_s2_10m", "valid_s2_20m", "valid_lulc"):
        with rasterio.open(ctx["phase3_root"] / str(year) / f"{mask_name}_30m.tif") as ds:
            m = ds.read(1)
        inside = m != 255
        frac = round(float((m == 1)[inside].mean()), 6) if inside.any() else None
        cov[f"{mask_name}_coverage"] = frac
    cov["l9_gate_0.90"] = bool(cov["valid_l9_coverage"] >= 0.90)
    cov["s2_gate_0.85"] = bool(cov["valid_s2_10m_coverage"] >= 0.85)
    cov["lulc_gate_0.80"] = bool(cov["valid_lulc_coverage"] >= 0.80)

    info = {
        "year": year, "domain_pixels": n,
        "n_extra_dummies_zero_filled": int(n_extra_dummies),
        "grid_pixels": int(shape[0] * shape[1]),
        "class_histogram": {name: int(hist[i]) for i, name in enumerate(CLASS_NAMES)},
        "class_pct": {name: round(100.0 * hist[i] / n, 4)
                      for i, name in enumerate(CLASS_NAMES)},
        "mean_confidence": float(conf.mean()),
        "mean_severity_score": float(score.mean()),
        "hotspots": {d: hot[d]["stats"] for d in ("A", "B", "C")},
        "coverage": cov,
        "rasters": outputs,
        "wall_s": time.perf_counter() - t0,
    }
    log(f"{year}: done in {info['wall_s']:.1f}s")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase3-root", default=str(PHASE3_ROOT_DEFAULT))
    ap.add_argument("--phase5-dir", default=str(PHASE5_DIR_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--years", default=",".join(str(y) for y in YEARS))
    ap.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    ap.add_argument("--met-csv", default=str(MET_CSV_DEFAULT))
    ap.add_argument("--study-area", default=str(STUDY_AREA_DEFAULT))
    ap.add_argument("--schema-json", default=str(SCHEMA_JSON_DEFAULT))
    args = ap.parse_args(argv)

    years = tuple(int(y) for y in args.years.split(","))
    phase3_root = Path(args.phase3_root)
    phase5_dir = Path(args.phase5_dir)
    out_root = Path(args.out)
    for d in ("rasters", "hotspots", "tables"):
        (out_root / d).mkdir(parents=True, exist_ok=True)
    log(f"years={years} out={out_root}")

    t_start = time.perf_counter()
    preflight(years, phase3_root, phase5_dir, Path(args.grid_file),
              Path(args.met_csv), Path(args.study_area), Path(args.schema_json))
    marker, artifact = resolve_primary_model(phase5_dir)
    log(f"primary model via marker: {marker['artifact']} "
        f"(locked acc {marker['locked_accuracy']:.6f})")
    t = time.perf_counter()
    model = joblib.load(artifact)
    log(f"joblib loaded in {time.perf_counter() - t:.1f}s")

    schema = load_frozen_schema(Path(args.schema_json))
    grid = Grid.from_file(Path(args.grid_file))
    px_lon, px_lat = grid.pixel_centers()
    from ..common import load_study_mask
    _study = load_study_mask(grid, Path(args.study_area))   # gate: study loads
    met_df = pd.read_csv(args.met_csv)
    for c in ("point_id", "lat", "lon", "time_utc"):
        if c not in met_df.columns:
            raise AssertionError(f"met CSV missing column {c}")

    ctx = {
        "grid": grid, "phase3_root": phase3_root, "schema": schema,
        "predictor_cols": predictor_col_order(),
        "block_raster": compute_block_raster(grid.height, grid.width),
        "px_lon": px_lon, "px_lat": px_lat, "met_df": met_df,
        "static": load_static(phase3_root),
    }

    year_infos = []
    for year in years:
        year_infos.append(run_year(year, model, ctx, out_root))
        gc.collect()

    # ---- temporal comparison ------------------------------------------------
    trows = [{
        "year": yi["year"], "valid_domain_pixels": yi["domain_pixels"],
        "low_pct": yi["class_pct"]["Low"], "moderate_pct": yi["class_pct"]["Moderate"],
        "high_pct": yi["class_pct"]["High"],
        "hotspot_defB_area_ha": yi["hotspots"]["B"]["total_area_ha"],
        "hotspot_defB_clusters": yi["hotspots"]["B"]["n_clusters"],
        "mean_confidence": round(yi["mean_confidence"], 6),
        "mean_severity_score": round(yi["mean_severity_score"], 6),
    } for yi in year_infos]
    pd.DataFrame(trows).to_csv(out_root / "tables" / "temporal_comparison.csv",
                               index=False)
    cov_df = pd.DataFrame([yi["coverage"] for yi in year_infos])
    cov_df.to_csv(out_root / "tables" / "coverage_statistics.csv", index=False)

    # ---- manifest + record ----------------------------------------------------
    manifest = {
        "stage": "phase6_uhi_severity_mapping",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_model": {
            "source": "marker v2/data/phase5/phase5_primary_model.json",
            "artifact": marker["artifact"],
            "model_sha256": sha256_file(artifact),
            "locked_accuracy": marker["locked_accuracy"],
            "locked_macro_f1": marker["locked_macro_f1"],
        },
        "schema": {"n_features": 178, "sha256": schema_hash(schema),
                   "source": str(args.schema_json)},
        "feature_construction": "v2.phase4.assemble_features code path, full "
                                "grid, sampling off; domain = all predictors "
                                "finite",
        "inputs": [{"path": str(Path(args.grid_file)),
                    "sha256": sha256_file(Path(args.grid_file))},
                   {"path": str(args.met_csv), "sha256": sha256_file(Path(args.met_csv))},
                   {"path": str(args.schema_json), "sha256": sha256_file(Path(args.schema_json))}],
        "years": [int(y) for y in years],
        "hotspot_definitions": HOTSPOT_DEFS,
        "min_cluster_px": MIN_CLUSTER_PX,
        "confidence_threshold": CONFIDENCE_T,
        "score_threshold": SCORE_T,
        "pixel_area_m2": PX_AREA_M2,
        "nodata": {"severity": SEVERITY_NODATA, "probability": PROB_NODATA,
                   "confidence": PROB_NODATA, "severity_score": PROB_NODATA,
                   "hotspot_ids": HOTSPOT_NODATA},
        "domain_counts": {str(yi["year"]): {
            "domain_pixels": yi["domain_pixels"],
            "grid_pixels": yi["grid_pixels"]} for yi in year_infos},
        "class_distribution": {str(yi["year"]): yi["class_pct"]
                               for yi in year_infos},
        "coverage": {str(yi["year"]): yi["coverage"] for yi in year_infos},
        "per_year": {str(yi["year"]): yi for yi in year_infos},
        "notes": [
            "Model is a DIRECT 3-class classifier; no LST thresholding "
            "(thresholds_by_year is provenance only).",
            "Hotspot tiers A/B/C are a faithful port of V1 spec section 8.",
            "Probability rasters are single 3-band GeoTIFFs (bands "
            "Low/Moderate/High).",
            "Phase 7 was NOT started.",
        ],
    }
    dump_json(manifest, out_root / "phase6_manifest.json")
    dump_json({
        "run_at_utc": manifest["created_at_utc"],
        "script": "v2.phase6.build",
        "timings_s": {str(yi["year"]): round(yi["wall_s"], 3)
                      for yi in year_infos},
        "total_wall_s": time.perf_counter() - t_start,
        "params": {"hotspot_defs": HOTSPOT_DEFS, "min_cluster_px": MIN_CLUSTER_PX,
                   "confidence_threshold": CONFIDENCE_T,
                   "score_threshold": SCORE_T,
                   "score_weights": SCORE_WEIGHTS.tolist(),
                   "years": [int(y) for y in years]},
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
    }, out_root / "phase6_pipeline_record.json")
    log(f"total {time.perf_counter() - t_start:.1f}s; manifest written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
