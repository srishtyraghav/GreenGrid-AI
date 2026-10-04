"""V2 Phase 4 — assemble the ML feature table on the frozen 178 schema.

Per year 2022–2026, on the authoritative 30 m grid:

  * base predictors from the Phase-3 products (V1 authoritative-source rules:
    NDVI/NDBI/VegetationCover from Sentinel-2, NDMI/MNDWI/BSI from Landsat 9,
    LST from Landsat 9, lulc_class from Dynamic World, landuse_class from the
    V1 OSM landuse raster, dist_* from the constraint distance rasters),
  * block-aware focal means 3/5/11 over the 8 V1 spatial bases,
  * disk morphology features at 50/100/250/500 m (block-aware),
  * the 33 nan-safe window stats (V1 build_spatial_context_features),
  * 6 met covariates: W4 (May 1-Jun 30, 04:00-06:00 UTC) point means from the
    Phase-2 hourly superset, IDW-interpolated with the V1 matched variant
    (power 2, longitude scaled by cos(28.64 deg), 1e-6 inside the squared
    distance).

Deterministic sampling replicates V1's Tier-2 rule (per-year independent,
up to 150,000 domain pixels, numpy default_rng(42), sorted indices). The
prediction domain mirrors the V1 production build: pixels where every raw
predictor column is finite.

The assembled frame carries metadata columns (row, col, spatial_block_id,
lst_C) followed by the EXACT 178 frozen-schema columns (the frozen schema
includes ``year`` as a predictor, so it appears inside the predictor block
at its schema position — keeping it in the metadata row as well would create
a duplicate column name)
(name order + SHA-256 both asserted at runtime — fail loudly on any drift).
Outputs per year: features_<year>.parquet (+ optional CSV),
feature_metadata.json, v2_feature_manifest.json (input hashes, schema hash,
row counts, coverage stats).

Usage:
    PYTHONPATH=src .venv/Scripts/python.exe -m src.v2.phase4.assemble_features \
        --phase3-root data/v2/phase3 --out-root data/v2/phase4 \
        [--years ...] [--grid-file ...] [--met-csv ...] [--csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from ..common import (
    GRID_FILE_DEFAULT,
    MAX_SAMPLES_PER_YEAR,
    STUDY_AREA_DEFAULT,
    MET_CSV_DEFAULT,
    MET_FEATURE_NAMES,
    MORPHOLOGY_LANDUSE_CLASSES,
    MORPHOLOGY_RADII_M,
    PROJECT_ROOT,
    RANDOM_SEED,
    SCHEMA_JSON_DEFAULT,
    YEARS,
    Grid,
    compute_block_raster,
    dump_json,
    load_frozen_schema,
    schema_hash,
    sha256_file,
)
from ._spatial import (
    MEAN_BANDS,
    RANGE_BANDS,
    SPATIAL_FEATURE_BASES,
    SPATIAL_WINDOW_SIZES,
    STD_BANDS,
    WINDOWS,
    disk_kernel,
    encode_predictors,
    focal_fraction_block_aware,
    focal_mean_block_aware,
    focal_mean_block_aware_disk,
    met_fields_for_year,
    nan_window_stats,
    radius_m_to_px,
)

# Raw predictor column set, exactly as V1 (scripts/freeze_phase5_production.py
# PREDICTOR_COLS = models.config.PREDICTOR_VARS + MET + NEW_SPATIAL). Column
# ORDER here is irrelevant — the frozen 178-schema reindex governs output —
# but the SET must match, and it is asserted at runtime.
def predictor_col_order() -> list[str]:
    base = [
        "ndvi", "ndbi", "vegetation_cover", "ndmi", "mndwi", "bsi", "ndre",
        "landuse_class", "lulc_class",
        "dist_road_m", "dist_vegetation_m", "dist_building_m", "year",
    ]
    spatial = [f"{b}_mean{w}" for b in SPATIAL_FEATURE_BASES for w in SPATIAL_WINDOW_SIZES]
    morphology = []
    for radius in MORPHOLOGY_RADII_M:
        morphology += [
            f"building_pixel_fraction_{radius}m",
            f"road_pixel_fraction_{radius}m",
            f"vegetation_pixel_fraction_{radius}m",
            f"vegetation_cover_mean_{radius}m",
        ]
        morphology += [f"landuse_frac_{c}_{radius}m" for c in MORPHOLOGY_LANDUSE_CLASSES]
        morphology += [
            f"landuse_dominant_{radius}m",
            f"landuse_entropy_{radius}m",
            f"ndvi_contrast_{radius}m",
            f"ndbi_contrast_{radius}m",
        ]
    met = list(MET_FEATURE_NAMES)
    new_spatial = (
        [f"{b}_std{w}" for b in STD_BANDS for w in WINDOWS]
        + [f"{b}_mean{w}" for b in MEAN_BANDS for w in WINDOWS]
        + [f"{b}_range{w}" for b in RANGE_BANDS for w in WINDOWS]
    )
    return base + spatial + morphology + met + new_spatial


DIST_COLS = (["dist_road_m", "dist_vegetation_m", "dist_building_m"]
             + [f"dist_{b}_mean{w}" for b in ("road_m", "vegetation_m", "building_m")
                for w in (3, 5, 11)])

PRODUCT_FILES = {
    "lst_C": "lst_30m.tif",
    "ndvi": "ndvi_30m.tif",
    "ndbi": "ndbi_30m.tif",
    "ndre": "ndre_30m.tif",
    "vegetation_cover": "vegetation_cover_30m.tif",
    "ndmi": "ndmi_30m.tif",
    "mndwi": "mndwi_30m.tif",
    "bsi": "bsi_30m.tif",
    "lulc_class": "lulc_30m.tif",
}


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1, masked=True).astype(np.float64).filled(np.nan)


def load_phase3_year(phase3_root: Path, year: int) -> dict[str, np.ndarray]:
    ydir = phase3_root / str(year)
    return {k: read_tif(ydir / fname) for k, fname in PRODUCT_FILES.items()}


def load_static(phase3_root: Path) -> dict[str, np.ndarray]:
    sdir = phase3_root / "static"
    out = {
        "landuse_class": read_tif(sdir / "landuse_raster_30m.tif"),
        "dist_road_m": read_tif(sdir / "roads_distance_30m.tif"),
        "dist_vegetation_m": read_tif(sdir / "vegetation_distance_30m.tif"),
        "dist_building_m": read_tif(sdir / "buildings_distance_30m.tif"),
    }
    lu = out["landuse_class"]
    out["landuse_class"] = np.where(lu == 255.0, np.nan, lu)  # nodata -> NaN (V1)
    return out


def compute_spatial_means(merged: dict[str, np.ndarray],
                          block_raster: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for feature in SPATIAL_FEATURE_BASES:
        for w in SPATIAL_WINDOW_SIZES:
            out[f"{feature}_mean{w}"] = focal_mean_block_aware(
                merged[feature], block_raster, w)
    return out


def compute_morphology(merged: dict[str, np.ndarray],
                       block_raster: np.ndarray) -> dict[str, np.ndarray]:
    """Full-grid morphology; landuse_dominant_* stays integer-coded (V1)."""
    from ._spatial import focal_aggregate_block_aware

    lu_classes = np.array(MORPHOLOGY_LANDUSE_CLASSES)
    landuse = merged["landuse_class"]
    out = {}
    for radius_m in MORPHOLOGY_RADII_M:
        kernel = disk_kernel(radius_m_to_px(radius_m))
        out[f"building_pixel_fraction_{radius_m}m"] = focal_fraction_block_aware(
            (merged["dist_building_m"] == 0), block_raster, kernel)
        out[f"road_pixel_fraction_{radius_m}m"] = focal_fraction_block_aware(
            (merged["dist_road_m"] == 0), block_raster, kernel)
        out[f"vegetation_pixel_fraction_{radius_m}m"] = focal_fraction_block_aware(
            (merged["dist_vegetation_m"] == 0), block_raster, kernel)
        out[f"vegetation_cover_mean_{radius_m}m"] = focal_mean_block_aware_disk(
            merged["vegetation_cover"], block_raster, kernel)
        out[f"ndvi_contrast_{radius_m}m"] = (
            merged["ndvi"] - focal_mean_block_aware_disk(merged["ndvi"], block_raster, kernel))
        out[f"ndbi_contrast_{radius_m}m"] = (
            merged["ndbi"] - focal_mean_block_aware_disk(merged["ndbi"], block_raster, kernel))

        class_counts = np.stack(
            [focal_aggregate_block_aware(
                (landuse == c).astype(np.float64), block_raster, kernel)[0]
             for c in lu_classes], axis=0)
        total_count = class_counts.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            probs = np.where(total_count > 0, class_counts / total_count, 0.0)
        class_fracs = np.nan_to_num(np.clip(probs, 0.0, 1.0), nan=0.0)
        for idx, c in enumerate(MORPHOLOGY_LANDUSE_CLASSES):
            out[f"landuse_frac_{c}_{radius_m}m"] = class_fracs[idx]
        out[f"landuse_dominant_{radius_m}m"] = np.where(
            total_count > 0, lu_classes[np.argmax(class_counts, axis=0)], 0).astype(np.float64)
        out[f"landuse_entropy_{radius_m}m"] = np.nan_to_num(np.where(
            total_count > 0,
            -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=0), 0.0), nan=0.0)
    return out


def compute_window_stats(phase3_root: Path, year: int) -> dict[str, np.ndarray]:
    """33 nan-safe window stats on float32 casts of the V1 source rasters."""
    ydir = phase3_root / str(year)
    band_paths = {
        "ndvi": ydir / "ndvi_30m.tif",
        "ndbi": ydir / "ndbi_30m.tif",
        "ndre": ydir / "ndre_30m.tif",
        "ndmi": ydir / "ndmi_30m.tif",
        "mndwi": ydir / "mndwi_30m.tif",
        "bsi": ydir / "bsi_30m.tif",
        "vegetation_cover": ydir / "vegetation_cover_30m.tif",
    }
    out = {}
    for band in dict.fromkeys(STD_BANDS + MEAN_BANDS + RANGE_BANDS):
        with rasterio.open(band_paths[band]) as ds:
            arr = ds.read(1).astype(np.float32)
        for w in WINDOWS:
            mean, std, rng = nan_window_stats(arr, w)
            if band in STD_BANDS:
                out[f"{band}_std{w}"] = std.astype(np.float64)
            if band in MEAN_BANDS:
                out[f"{band}_mean{w}"] = mean.astype(np.float64)
            if band in RANGE_BANDS:
                out[f"{band}_range{w}"] = rng.astype(np.float64)
    return out


def build_year_columns(year: int, merged: dict[str, np.ndarray],
                       block_raster: np.ndarray, met: dict[str, np.ndarray],
                       phase3_root: Path, predictor_cols: list[str]
                       ) -> dict[str, np.ndarray]:
    """All raw predictor columns for one year (full grids), V1 set-exact."""
    cols: dict[str, np.ndarray] = {name: merged[name] for name in (
        "ndvi", "ndbi", "vegetation_cover", "ndmi", "mndwi", "bsi", "ndre",
        "landuse_class", "lulc_class",
        "dist_road_m", "dist_vegetation_m", "dist_building_m")}
    cols["year"] = np.full(merged["ndvi"].shape, year, dtype=np.float64)
    cols.update(compute_spatial_means(merged, block_raster))
    cols.update(compute_morphology(merged, block_raster))
    cols.update(compute_window_stats(phase3_root, year))
    for name, arr in met.items():
        cols[name] = arr.reshape(merged["ndvi"].shape)

    missing = set(predictor_cols) - set(cols)
    extra = set(cols) - set(predictor_cols)
    if missing or extra:
        raise AssertionError(
            f"[{year}] predictor column mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}")
    return cols


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase3-root", default=str(PROJECT_ROOT / "data" / "phase3"))
    p.add_argument("--out-root", default=str(PROJECT_ROOT / "data" / "phase4"))
    p.add_argument("--years", default=",".join(str(y) for y in YEARS))
    p.add_argument("--grid-file", default=str(GRID_FILE_DEFAULT))
    p.add_argument("--met-csv", default=str(MET_CSV_DEFAULT))
    p.add_argument("--study-area", default=str(STUDY_AREA_DEFAULT),
                   help="study-area GeoJSON; the sampling domain is clipped to "
                        "its rasterized extent (with real GEE data the products "
                        "are already NaN outside, this makes it explicit)")
    p.add_argument("--schema-json", default=str(SCHEMA_JSON_DEFAULT))
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_YEAR)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--csv", action="store_true", help="also write per-year CSV")
    args = p.parse_args(argv)

    phase3_root = Path(args.phase3_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    years = tuple(int(y) for y in args.years.split(","))
    predictor_cols = predictor_col_order()

    schema = load_frozen_schema(Path(args.schema_json))
    schema_sha = schema_hash(schema)
    print(f"[ASSEMBLE] frozen schema: 178 features, sha256 {schema_sha[:16]}...")

    grid = Grid.from_file(Path(args.grid_file))
    if (grid.height, grid.width) != (1768, 1874):
        print(f"[ASSEMBLE] WARNING: grid is {grid.width}x{grid.height}; the pinned "
              f"block bins are defined for the 1874x1768 V1/V2 grid", file=sys.stderr)
    px_lon, px_lat = grid.pixel_centers()

    print("[ASSEMBLE] loading met superset ...")
    met_df = pd.read_csv(args.met_csv)
    missing_cols = {"point_id", "lat", "lon", "time_utc"} - set(met_df.columns)
    if missing_cols:
        print(f"[ASSEMBLE] ERROR: met CSV missing columns {missing_cols}", file=sys.stderr)
        return 1

    static = load_static(phase3_root)
    static_files = sorted((phase3_root / "static").glob("*.tif"))

    # Study-area clip: with real GEE products every predictor is already NaN
    # outside the study area; the explicit clip makes the domain provably
    # study-internal for any input (and matches V1, whose Phase-3 valid mask
    # could never leave the study area).
    from ..common import load_study_mask
    study = load_study_mask(grid, Path(args.study_area))
    print(f"[ASSEMBLE] study-area cells: {int(study.sum())} / {study.size}")

    # ---- pass 1: base domain + deterministic sampling (V1 Tier-2 rule) ------
    sampled: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    domain_stats: dict[int, dict] = {}
    base_names = ("ndvi", "ndbi", "vegetation_cover", "ndmi", "mndwi", "bsi",
                  "ndre", "lulc_class", "landuse_class",
                  "dist_road_m", "dist_vegetation_m", "dist_building_m")
    for year in years:
        merged = load_phase3_year(phase3_root, year)
        merged.update(static)
        domain = study & np.isfinite(merged["ndvi"])
        for name in base_names[1:]:
            domain &= np.isfinite(merged[name])
        idx = np.flatnonzero(domain.ravel())
        if len(idx) == 0:
            print(f"[ASSEMBLE] ERROR: {year}: empty domain", file=sys.stderr)
            return 1
        rng = np.random.default_rng(args.seed)
        if len(idx) > args.max_samples:
            idx = np.sort(rng.choice(idx, size=args.max_samples, replace=False))
        rows, cl = idx // grid.width, idx % grid.width
        sampled[year] = (rows, cl)
        domain_stats[year] = {
            "domain_pixels": int(domain.sum()), "sampled_pixels": int(len(idx)),
            "grid_pixels": int(domain.size),
        }
        print(f"[ASSEMBLE] {year}: domain {domain_stats[year]['domain_pixels']:,} "
              f"-> sampled {len(idx):,}")

    # ---- 25-block tiling: PINNED V1 geometry (independent of sampling) ------
    block_raster = compute_block_raster(grid.height, grid.width)

    input_hashes: dict[str, str] = {
        "met_csv": sha256_file(Path(args.met_csv)),
        "schema_json": sha256_file(Path(args.schema_json)),
        "grid_file": sha256_file(Path(args.grid_file)),
    }
    for f in static_files:
        input_hashes[f"static/{f.name}"] = sha256_file(f)

    manifest: dict = {
        "schema_sha256": schema_sha,
        "schema_source": str(args.schema_json),
        "n_features": 178,
        "years": list(years),
        "sampling": {
            "rule": "per-year independent, up to --max-samples domain pixels, "
                    "numpy default_rng(seed) choice without replacement, sorted",
            "max_samples_per_year": args.max_samples,
            "seed": args.seed,
        },
        "block_tiling": {
            "n_blocks": 5,
            "rule": "PINNED V1 bins (src/v2/common.py PINNED_ROW_BINS/"
                    "PINNED_COL_BINS; data/phase2/spatial_blocks_manifest.json) "
                    "- independent of the V2 sample extent by contract",
            "holdout_blocks": [2, 9, 15, 23],
        },
        "met": {
            "source": str(args.met_csv),
            "aggregation": "per point per year mean over May 1 - Jun 30, "
                           "hours 04:00-06:00 UTC",
            "idw": "power 2, longitude scaled by cos(28.64 deg), 1e-6 "
                   "epsilon inside squared distance (V1 matched variant)",
        },
        "per_year": {},
    }

    for year in years:
        print(f"[ASSEMBLE] === year {year} ===", flush=True)
        merged = load_phase3_year(phase3_root, year)
        for name, fname in PRODUCT_FILES.items():
            input_hashes[f"{year}/{fname}"] = sha256_file(phase3_root / str(year) / fname)
        merged.update(static)

        met = met_fields_for_year(met_df, year, px_lon, px_lat)
        cols = build_year_columns(year, merged, block_raster, met,
                                  phase3_root, predictor_cols)

        # V1 production domain: pixels where EVERY raw predictor is finite.
        domain = np.ones(grid.shape, dtype=bool)
        for arr in cols.values():
            domain &= np.isfinite(arr)
        rows, cl = sampled[year]
        flat = rows * grid.width + cl
        keep = domain.ravel()[flat]
        rows, cl, flat = rows[keep], cl[keep], flat[keep]

        data = {
            "row": rows.astype(np.int64),
            "col": cl.astype(np.int64),
            "spatial_block_id": block_raster[rows, cl].astype(np.int64),
            "lst_C": merged["lst_C"].ravel()[flat].astype(np.float64),
        }
        for name in predictor_cols:
            data[name] = cols[name].ravel()[flat]
        df = pd.DataFrame(data)

        # dtype parity with the V1 training table BEFORE encoding (drives
        # float-style landuse dummies and int-style lulc/dominant dummies).
        df["landuse_class"] = df["landuse_class"].astype(np.float64)
        df["lulc_class"] = df["lulc_class"].astype(np.int64)
        for c in [c for c in df.columns if c.startswith("landuse_dominant_")]:
            df[c] = df[c].astype(np.int64)

        X, names = encode_predictors(df, predictor_cols)
        X = X.reindex(columns=schema, fill_value=0)
        assert list(X.columns) == list(schema), (
            f"schema mismatch after encode; encoded extras="
            f"{[c for c in names if c not in schema]}")
        for c in X.columns:
            if c not in DIST_COLS:
                X[c] = X[c].astype(np.float32)

        result = pd.concat(
            [df[["row", "col", "spatial_block_id", "lst_C"]].reset_index(drop=True),
             X.reset_index(drop=True)], axis=1)
        assert list(result.columns[4:]) == list(schema), "178-schema order violated"

        parquet_path = out_root / f"features_{year}.parquet"
        result.to_parquet(parquet_path, index=False)
        if args.csv:
            result.to_csv(out_root / f"features_{year}.csv", index=False)

        ydir = phase3_root / str(year)
        cov = {}
        for mask_name in ("valid_l9", "valid_s2_10m", "valid_s2_20m", "valid_lulc"):
            with rasterio.open(ydir / f"{mask_name}_30m.tif") as ds:
                m = ds.read(1)
            inside = m != 255
            cov[mask_name] = (round(float((m == 1)[inside].mean()), 6)
                              if inside.any() else None)

        manifest["per_year"][str(year)] = {
            **domain_stats[year],
            "rows_written": int(len(result)),
            "domain_rows_after_derived_filters": int(keep.sum()),
            "encoded_columns_before_reindex": int(len(names)),
            "coverage_phase3_masks": cov,
            "output_parquet": str(parquet_path),
            "output_sha256": sha256_file(parquet_path),
        }
        print(f"[ASSEMBLE] {year}: wrote {parquet_path} rows={len(result)}", flush=True)
        del merged, cols, df, X, result

    manifest["input_hashes"] = input_hashes
    dump_json(manifest, out_root / "v2_feature_manifest.json")
    dump_json({
        "n_features": 178,
        "schema_sha256": schema_sha,
        "feature_names": schema,
        "metadata_columns": ["row", "col", "spatial_block_id", "lst_C"],
        "note": "year is a predictor inside the 178-schema block "
                "(frozen schema position), not a duplicate metadata column",
        "dtype_policy": "float32 except distance columns float64 (V1 parity)",
        "generated_by": "src/v2/phase4/assemble_features.py",
    }, out_root / "feature_metadata.json")
    print("[ASSEMBLE] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
