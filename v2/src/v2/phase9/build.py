"""Build V2 Phase 9: supervised LST-change regression and 2026 scenarios.

This phase reads frozen V2 products from Phases 3 and 8, but only writes new
Phase 9 artifacts.  The scientific interpretation is deliberately narrow:
predicted LST reduction under an assumed vegetation/tree-change scenario,
learned from observed multi-year LST and vegetation changes.

Usage from repository root:
    PYTHONPATH=v2/src .venv/Scripts/python.exe -m v2.phase9.build
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from xgboost import XGBRegressor

from ..common import (
    GRID_FILE_DEFAULT,
    HOLDOUT_BLOCKS,
    MET_CSV_DEFAULT,
    MET_FEATURE_NAMES,
    PROJECT_ROOT,
    YEARS,
    Grid,
    compute_block_raster,
    dump_json,
    sha256_file,
)
from ..phase4.assemble_features import load_phase3_year, load_static
from ..phase4._spatial import met_fields_for_year

PHASE3_ROOT_DEFAULT = PROJECT_ROOT / "data" / "phase3"
PHASE8_ROOT_DEFAULT = PROJECT_ROOT / "data" / "phase8"
OUT_DEFAULT = PROJECT_ROOT / "data" / "phase9"
REPORT_DEFAULT = PROJECT_ROOT / "reports" / "phase9_temperature_reduction_prediction_report.md"

TRANSITIONS = ((2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026))
SCENARIO_YEAR = 2026
PRIMARY_SCENARIO = "v2_constrained"
DENSITIES = (400, 1000, 2500)
RANDOM_SEED = 42
HA_PER_PX = 900.0 / 10_000.0          # 0.09 ha per 30 m pixel

FEATURE_COLS = (
    "current_lst_C",
    "current_ndvi",
    "current_ndbi",
    "current_vegetation_cover",
    "vegetation_change",
    "dist_road_m",
    "dist_vegetation_m",
    "dist_building_m",
    "landuse_class",
    # meteorological / context covariates: per-year external W4 observations
    # (hours 04-06 UTC, May 1 - Jun 30), IDW-interpolated with the exact V1
    # matched variant by v2.phase4._spatial.met_fields_for_year. They are
    # CONTEMPORANEOUS EXTERNAL OBSERVATIONS of the transition start year,
    # not target-derived (see leakage_audit).
    "current_met_t2m_c",
    "current_met_rh_pct",
    "current_met_wind_kmh",
    "current_met_precip_mm",
    "current_met_ssr_wm2",
    "current_met_swc_m3m3",
)


@dataclass(frozen=True)
class VegetationScenarioConfig:
    """Explicit scenario assumption, not observed tree-density evidence."""

    vegetation_change_per_1000_trees_ha: float = 0.05
    sensitivity_values: tuple[float, ...] = (0.03, 0.05, 0.08)
    source: str = (
        "No V2-observed tree-density-to-canopy conversion exists; Phase 9 "
        "therefore treats this as an explicit scenario parameter. Values are "
        "vegetation-cover fraction added per 1,000 trees/ha."
    )


def log(msg: str) -> None:
    print(f"[P9] {msg}", flush=True)


def read_band(path: Path) -> tuple[np.ndarray, float | None, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
        nodata = ds.nodata
        profile = ds.profile
    return arr, nodata, profile


def metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae_C": float(mean_absolute_error(y_true, y_pred)),
        "rmse_C": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def preflight(phase3_root: Path, phase8_root: Path, grid_file: Path) -> None:
    for y in YEARS:
        ydir = phase3_root / str(y)
        for name in ("lst_30m.tif", "ndvi_30m.tif", "ndbi_30m.tif", "vegetation_cover_30m.tif"):
            p = ydir / name
            if not p.exists():
                raise FileNotFoundError(f"missing Phase 3 input: {p}")
    for name in (
        "landuse_raster_30m.tif",
        "roads_distance_30m.tif",
        "vegetation_distance_30m.tif",
        "buildings_distance_30m.tif",
    ):
        p = phase3_root / "static" / name
        if not p.exists():
            raise FileNotFoundError(f"missing Phase 3 static input: {p}")
    for density in DENSITIES:
        # Tables are density-agnostic, but this loop documents the planned scenario set.
        _ = density
    p8 = phase8_root / PRIMARY_SCENARIO / "tables" / f"tree_requirement_by_zone_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}.csv"
    if not p8.exists():
        raise FileNotFoundError(f"missing Phase 8 zone table: {p8}")
    pspace = phase8_root / PRIMARY_SCENARIO / "rasters" / f"available_planting_space_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}.tif"
    if not pspace.exists():
        raise FileNotFoundError(f"missing Phase 8 plantable raster: {pspace}")
    grid = Grid.from_file(grid_file)
    assert (grid.height, grid.width) == (1768, 1874), "authoritative grid mismatch"


def load_met_context(phase3_root: Path):
    """Meteorological context fields on the full grid, per year, via the
    EXACT Phase 4 code path (``met_fields_for_year``: W4, hours 04-06 UTC,
    V1 matched IDW variant) — value-identical to the 178-schema met columns.
    """
    import pandas as pd
    grid = Grid.from_file(GRID_FILE_DEFAULT)
    px_lon, px_lat = grid.pixel_centers()
    met_df = pd.read_csv(MET_CSV_DEFAULT)
    missing = {"point_id", "lat", "lon", "time_utc"} - set(met_df.columns)
    if missing:
        raise AssertionError(f"met CSV missing columns {missing}")
    out = {}
    for year in YEARS:
        fields = met_fields_for_year(met_df, year, px_lon, px_lat)
        for name in MET_FEATURE_NAMES:
            if name not in fields:
                raise AssertionError(f"met field {name} missing for {year}")
        out[year] = fields
    return out


def build_transition_panel(
    phase3_root: Path,
    out_dir: Path,
    max_samples_per_transition: int,
    seed: int,
) -> pd.DataFrame:
    """Construct sampled annual W4 transition panel.

    The intervention/change variable is observed vegetation-cover change
    during the transition.  It is allowed only because scenario inference
    replaces it with an assumed future change; the leakage audit records this
    distinction.  Met covariates are current-year (transition start)
    contemporaneous external observations (not target-derived).
    """
    rng = np.random.default_rng(seed)
    static = load_static(phase3_root)
    met_by_year = load_met_context(phase3_root)
    block_raster = compute_block_raster(static["dist_road_m"].shape[0], static["dist_road_m"].shape[1])
    rows = []
    summaries = []
    for start, end in TRANSITIONS:
        cur = load_phase3_year(phase3_root, start)
        nxt = load_phase3_year(phase3_root, end)
        met = met_by_year[start]
        domain = (
            np.isfinite(cur["lst_C"])
            & np.isfinite(nxt["lst_C"])
            & np.isfinite(cur["ndvi"])
            & np.isfinite(cur["ndbi"])
            & np.isfinite(cur["vegetation_cover"])
            & np.isfinite(nxt["vegetation_cover"])
            & np.isfinite(static["dist_road_m"])
            & np.isfinite(static["dist_vegetation_m"])
            & np.isfinite(static["dist_building_m"])
        )
        flat = np.flatnonzero(domain.ravel())
        n_domain = int(flat.size)
        if n_domain == 0:
            raise AssertionError(f"empty transition domain for {start}->{end}")
        n_take = min(max_samples_per_transition, n_domain)
        sample = rng.choice(flat, size=n_take, replace=False)
        sample.sort()
        rr = sample // domain.shape[1]
        cc = sample % domain.shape[1]
        frame = pd.DataFrame({
            "transition": f"{start}->{end}",
            "start_year": start,
            "end_year": end,
            "row": rr.astype(np.int32),
            "col": cc.astype(np.int32),
            "spatial_block_id": block_raster.ravel()[sample].astype(np.int16),
            "current_lst_C": cur["lst_C"].ravel()[sample].astype(np.float32),
            "next_lst_C": nxt["lst_C"].ravel()[sample].astype(np.float32),
            "delta_lst_C": (nxt["lst_C"].ravel()[sample] - cur["lst_C"].ravel()[sample]).astype(np.float32),
            "current_ndvi": cur["ndvi"].ravel()[sample].astype(np.float32),
            "current_ndbi": cur["ndbi"].ravel()[sample].astype(np.float32),
            "current_vegetation_cover": cur["vegetation_cover"].ravel()[sample].astype(np.float32),
            "vegetation_change": (
                nxt["vegetation_cover"].ravel()[sample] - cur["vegetation_cover"].ravel()[sample]
            ).astype(np.float32),
            "dist_road_m": static["dist_road_m"].ravel()[sample].astype(np.float32),
            "dist_vegetation_m": static["dist_vegetation_m"].ravel()[sample].astype(np.float32),
            "dist_building_m": static["dist_building_m"].ravel()[sample].astype(np.float32),
            "landuse_class": np.where(
                np.isfinite(static["landuse_class"].ravel()[sample]),
                static["landuse_class"].ravel()[sample],
                255.0,
            ).astype(np.float32),
        })
        for name in MET_FEATURE_NAMES:
            frame[f"current_{name}"] = (
                met[name].ravel()[sample].astype(np.float32)
            )
        rows.append(frame)
        summaries.append({
            "transition": f"{start}->{end}",
            "domain_pixels": n_domain,
            "sampled_rows": int(len(frame)),
            "blocks": int(frame["spatial_block_id"].nunique()),
            "target_mean_delta_lst_C": float(frame["delta_lst_C"].mean()),
            "target_std_delta_lst_C": float(frame["delta_lst_C"].std()),
            "vegetation_change_mean": float(frame["vegetation_change"].mean()),
        })
        log(f"panel {start}->{end}: domain={n_domain:,}, sampled={len(frame):,}")
    panel = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "training_panel.parquet", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "panel_summary.csv", index=False)
    return panel


def model_specs() -> dict:
    return {
        "rf": RandomForestRegressor(
            n_estimators=180,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=260,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def evaluate_model(name: str, base_model, panel: pd.DataFrame, out_dir: Path) -> dict:
    X = panel[list(FEATURE_COLS)]
    y = panel["delta_lst_C"].to_numpy()
    groups = panel["spatial_block_id"].to_numpy()
    transitions = panel["transition"].to_numpy()
    result = {
        "model": name,
        "features": list(FEATURE_COLS),
        "sample_count": int(len(panel)),
        "block_count": int(panel["spatial_block_id"].nunique()),
        "transition_count": int(panel["transition"].nunique()),
    }

    # Adjacent-block CV on non-locked blocks.
    non_locked = ~np.isin(groups, HOLDOUT_BLOCKS)
    sub_idx = np.flatnonzero(non_locked)
    sub_groups = groups[sub_idx]
    cv_rows = []
    n_splits = min(5, len(np.unique(sub_groups)))
    if n_splits >= 2:
        for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(sub_idx, groups=sub_groups), start=1):
            tr_idx = sub_idx[tr]
            va_idx = sub_idx[va]
            model = clone(base_model)
            pred = fit_predict(model, X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx])
            row = {"fold": fold, **metrics(y[va_idx], pred),
                   "val_blocks": ",".join(str(int(b)) for b in sorted(np.unique(groups[va_idx])))}
            cv_rows.append(row)
        pd.DataFrame(cv_rows).to_csv(out_dir / f"{name}_adjacent_block_cv.csv", index=False)
        result["adjacent_block_cv"] = aggregate_metric_rows(cv_rows)

    # LOBO on occupied blocks.
    lobo_rows = []
    for bid in sorted(np.unique(groups)):
        test = groups == bid
        train = ~test
        if test.sum() == 0 or train.sum() == 0:
            continue
        model = clone(base_model)
        pred = fit_predict(model, X.loc[train], y[train], X.loc[test])
        lobo_rows.append({"block": int(bid), **metrics(y[test], pred)})
    pd.DataFrame(lobo_rows).to_csv(out_dir / f"{name}_lobo.csv", index=False)
    result["lobo"] = aggregate_metric_rows(lobo_rows)

    # Locked geographic holdout.
    locked = np.isin(groups, HOLDOUT_BLOCKS)
    if locked.any() and (~locked).any():
        model = clone(base_model)
        pred = fit_predict(model, X.loc[~locked], y[~locked], X.loc[locked])
        locked_df = panel.loc[locked, ["transition", "start_year", "end_year", "row", "col", "spatial_block_id", "delta_lst_C", "current_lst_C"]].copy()
        locked_df["predicted_delta_lst_C"] = pred
        locked_df["residual_C"] = locked_df["delta_lst_C"] - locked_df["predicted_delta_lst_C"]
        locked_df.to_csv(out_dir / f"{name}_locked_predictions.csv", index=False)
        result["locked_holdout"] = metrics(y[locked], pred)
        resid = locked_df["residual_C"].to_numpy()
        result["locked_residual_summary"] = {
            "mean_C": float(resid.mean()),
            "std_C": float(resid.std()),
            "min_C": float(resid.min()),
            "p01_C": float(np.percentile(resid, 1)),
            "p99_C": float(np.percentile(resid, 99)),
            "max_C": float(resid.max()),
            "frac_abs_gt_2C": float((np.abs(resid) > 2.0).mean()),
        }
        result["current_lst_prediction_correlation_locked"] = float(np.corrcoef(panel.loc[locked, "current_lst_C"], pred)[0, 1])
        by_transition = []
        for t in sorted(np.unique(transitions[locked])):
            m = locked & (transitions == t)
            by_transition.append({"transition": t, **metrics(y[m], pred[transitions[locked] == t])})
        pd.DataFrame(by_transition).to_csv(out_dir / f"{name}_locked_by_transition.csv", index=False)
        result["locked_by_transition"] = by_transition

    # Time-aware leave-one-transition-out.
    time_rows = []
    for t in sorted(np.unique(transitions)):
        test = transitions == t
        train = ~test
        model = clone(base_model)
        pred = fit_predict(model, X.loc[train], y[train], X.loc[test])
        time_rows.append({"transition": t, **metrics(y[test], pred)})
    pd.DataFrame(time_rows).to_csv(out_dir / f"{name}_time_holdout.csv", index=False)
    result["time_holdout"] = aggregate_metric_rows(time_rows)

    # Final model on all panel rows.
    final = clone(base_model)
    final.fit(X, y)
    joblib.dump(final, out_dir / f"{name}_model.joblib")
    result["model_path"] = str(out_dir / f"{name}_model.joblib")
    importance = feature_importance(final, X, y, name)
    importance.to_csv(out_dir / f"{name}_feature_importance.csv", index=False)
    result["top_features"] = importance.head(10).to_dict(orient="records")
    return result


def aggregate_metric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out = {"folds": len(rows)}
    for col in ("mae_C", "rmse_C", "r2"):
        vals = np.array([r[col] for r in rows], dtype=float)
        out[f"{col}_mean"] = float(vals.mean())
        out[f"{col}_std"] = float(vals.std())
    out["n_total_eval_rows"] = int(sum(r["n"] for r in rows))
    return out


def feature_importance(model, X, y, name: str) -> pd.DataFrame:
    built_in = getattr(model, "feature_importances_", None)
    df = pd.DataFrame({"feature": list(X.columns)})
    if built_in is not None:
        df["importance"] = np.asarray(built_in, dtype=float)
    else:
        df["importance"] = 0.0
    # Small permutation sample: diagnostic, not model selection.
    n = min(5000, len(X))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(np.arange(len(X)), size=n, replace=False)
    try:
        perm = permutation_importance(
            model,
            X.iloc[idx],
            np.asarray(y)[idx],
            n_repeats=4,
            random_state=RANDOM_SEED,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        df["permutation_importance_mae"] = perm.importances_mean
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        log(f"{name}: permutation importance skipped: {exc}")
        df["permutation_importance_mae"] = np.nan
    return df.sort_values(["permutation_importance_mae", "importance"], ascending=False)


def select_model(results: dict) -> str:
    """Prefer locked MAE, with time-holdout MAE as tie-breaker."""
    def key(item):
        name, res = item
        locked = res.get("locked_holdout", {}).get("mae_C", np.inf)
        time_mae = res.get("time_holdout", {}).get("mae_C_mean", np.inf)
        return (locked, time_mae, name)
    return sorted(results.items(), key=key)[0][0]


def assumed_veg_change(density: int, cfg: VegetationScenarioConfig) -> float:
    return float(density / 1000.0 * cfg.vegetation_change_per_1000_trees_ha)


def build_scenario_predictions(
    selected_model,
    phase3_root: Path,
    phase8_root: Path,
    out_dir: Path,
    veg_cfg: VegetationScenarioConfig,
) -> pd.DataFrame:
    grid = Grid.from_file(GRID_FILE_DEFAULT)
    cur = load_phase3_year(phase3_root, SCENARIO_YEAR)
    static = load_static(phase3_root)
    met = load_met_context(phase3_root)[SCENARIO_YEAR]
    plant_path = phase8_root / PRIMARY_SCENARIO / "rasters" / f"available_planting_space_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}.tif"
    zids_path = phase8_root.parent / "phase7" / PRIMARY_SCENARIO / "rasters" / f"priority_zone_ids_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}.tif"
    # phase8_root normally points at v2/data/phase8; parent is v2/data.
    if not zids_path.exists():
        zids_path = PROJECT_ROOT / "data" / "phase7" / PRIMARY_SCENARIO / "rasters" / f"priority_zone_ids_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}.tif"
    plant, plant_nd, profile = read_band(plant_path)
    zids, _, _ = read_band(zids_path)
    plantable = plant == 1
    flat = np.flatnonzero(plantable.ravel())
    if flat.size == 0:
        raise AssertionError("2026 Phase 8 primary scenario has no plantable pixels")
    rr = flat // grid.width
    cc = flat % grid.width
    zone_table = pd.read_csv(
        phase8_root / PRIMARY_SCENARIO / "tables" / f"tree_requirement_by_zone_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}.csv"
    )
    rasters_dir = out_dir / "rasters"
    rasters_dir.mkdir(parents=True, exist_ok=True)
    zone_rows = []
    pixel_base = pd.DataFrame({
        "current_lst_C": cur["lst_C"].ravel()[flat].astype(np.float32),
        "current_ndvi": cur["ndvi"].ravel()[flat].astype(np.float32),
        "current_ndbi": cur["ndbi"].ravel()[flat].astype(np.float32),
        "current_vegetation_cover": cur["vegetation_cover"].ravel()[flat].astype(np.float32),
        "dist_road_m": static["dist_road_m"].ravel()[flat].astype(np.float32),
        "dist_vegetation_m": static["dist_vegetation_m"].ravel()[flat].astype(np.float32),
        "dist_building_m": static["dist_building_m"].ravel()[flat].astype(np.float32),
        "landuse_class": np.where(
            np.isfinite(static["landuse_class"].ravel()[flat]),
            static["landuse_class"].ravel()[flat],
            255.0,
        ).astype(np.float32),
        "zone_id": zids.ravel()[flat].astype(np.int32),
    })
    for name in MET_FEATURE_NAMES:
        pixel_base[f"current_{name}"] = (
            met[name].ravel()[flat].astype(np.float32)
        )
    for density in DENSITIES:
        veg_delta = assumed_veg_change(density, veg_cfg)
        X = pixel_base.copy()
        X["vegetation_change"] = veg_delta
        pred_delta = selected_model.predict(X[list(FEATURE_COLS)])
        cooling = -pred_delta
        post = X["current_lst_C"].to_numpy() + pred_delta
        write_scenario_raster(pred_delta, rr, cc, profile, rasters_dir / f"predicted_delta_lst_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}_{density}trees_ha.tif")
        write_scenario_raster(cooling, rr, cc, profile, rasters_dir / f"predicted_cooling_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}_{density}trees_ha.tif")
        write_scenario_raster(post, rr, cc, profile, rasters_dir / f"predicted_post_lst_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}_{density}trees_ha.tif")
        pixel_df = pixel_base[["zone_id", "current_lst_C"]].copy()
        pixel_df["predicted_delta_lst_C"] = pred_delta
        pixel_df["predicted_cooling_C"] = cooling
        pixel_df["predicted_post_lst_C"] = post
        grouped = pixel_df.groupby("zone_id", as_index=False).agg(
            current_lst_C=("current_lst_C", "mean"),
            predicted_delta_lst_C=("predicted_delta_lst_C", "mean"),
            predicted_cooling_C=("predicted_cooling_C", "mean"),
            predicted_post_lst_C=("predicted_post_lst_C", "mean"),
        )
        for rec in grouped.to_dict(orient="records"):
            zid = int(rec["zone_id"])
            zrow = zone_table.loc[zone_table["zone_id"] == zid].iloc[0]
            zone_rows.append({
                "year": SCENARIO_YEAR,
                "zone_id": zid,
                "plantable_area_ha": float(zrow["plantable_ha"]),
                "tree_density_trees_per_ha": density,
                "estimated_tree_count": int(round(float(zrow["plantable_ha"]) * density)),
                "current_lst_C": float(rec["current_lst_C"]),
                "assumed_vegetation_change": veg_delta,
                "predicted_delta_lst_C": float(rec["predicted_delta_lst_C"]),
                "predicted_cooling_C": float(rec["predicted_cooling_C"]),
                "predicted_post_lst_C": float(rec["predicted_post_lst_C"]),
            })
    out = pd.DataFrame(zone_rows).sort_values(["tree_density_trees_per_ha", "zone_id"])
    out.to_csv(out_dir / "tables" / "phase9_zone_cooling_predictions_2026.csv", index=False)
    return out


def write_scenario_raster(values, rows, cols, profile, path: Path) -> None:
    arr = np.full((profile["height"], profile["width"]), -9999.0, dtype=np.float32)
    arr[rows, cols] = np.asarray(values, dtype=np.float32)
    prof = dict(profile)
    prof.update({"dtype": "float32", "count": 1, "nodata": -9999.0, "compress": "lzw"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, path.stem)


def sanity_checks(selected_model, panel: pd.DataFrame, scenario: pd.DataFrame, out_dir: Path, veg_cfg: VegetationScenarioConfig) -> dict:
    X = panel[list(FEATURE_COLS)].sample(min(15000, len(panel)), random_state=RANDOM_SEED)
    base = X.copy()
    checks = {
        "predicted_cooling_distribution_2026": {
            "min": float(scenario["predicted_cooling_C"].min()),
            "p05": float(scenario["predicted_cooling_C"].quantile(0.05)),
            "median": float(scenario["predicted_cooling_C"].median()),
            "p95": float(scenario["predicted_cooling_C"].quantile(0.95)),
            "max": float(scenario["predicted_cooling_C"].max()),
        },
        "extreme_prediction_count_abs_delta_gt_10C": int((scenario["predicted_delta_lst_C"].abs() > 10).sum()),
        "density_monotonicity_not_forced": True,
    }
    sensitivity = []
    for val in veg_cfg.sensitivity_values:
        probe = base.copy()
        probe["vegetation_change"] = val
        pred = selected_model.predict(probe)
        sensitivity.append({"assumed_vegetation_change": val,
                            "mean_predicted_delta_lst_C": float(np.mean(pred)),
                            "mean_predicted_cooling_C": float(np.mean(-pred))})
    checks["vegetation_change_sensitivity"] = sensitivity
    pd.DataFrame(sensitivity).to_csv(out_dir / "diagnostics" / "vegetation_change_sensitivity.csv", index=False)
    dump_json(checks, out_dir / "diagnostics" / "sanity_checks.json")
    return checks


def write_report(path: Path, manifest: dict, zone_predictions: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = manifest["selected_model"]
    lines = [
        "# V2 Phase 9 — Temperature Reduction Prediction",
        "",
        "**Status:** COMPLETE — ML-based LST-change regression + 2026 scenario inference.",
        "",
        "All outputs are **predicted LST changes under assumed vegetation-change",
        " scenarios** — LST predictions, not air-temperature guarantees, and not",
        " causal tree-cooling claims. Three layers must not be conflated:",
        "",
        "1. **OBSERVED** — historical W4 LST and vegetation-cover transitions",
        "   2022->2023, 2023->2024, 2024->2025, 2025->2026 (the training panel).",
        "2. **ASSUMED** — future vegetation-cover change per planting-density",
        "   scenario (below); an explicit scenario parameter, not an observed",
        "   tree-density->canopy conversion (none exists in V2).",
        "3. **PREDICTED** — ML model estimates of delta LST given the assumed",
        "   change; empirical associations, not proven causal cooling.",
        "",
        "## Data and target (OBSERVED)",
        "",
        f"Transitions: {', '.join(t['transition'] for t in manifest['panel_summary'])}. "
        f"Training rows: {manifest['training_sample_size']:,}. Blocks: "
        f"{manifest['block_count']}. Target: `delta_lst_C = LST(year+1, W4) - "
        "LST(year, W4)` (observed same-season annual change).",
        "",
        "Predictors: current LST, NDVI, NDBI, vegetation cover, the observed "
        "vegetation-change variable (intervention variable; replaced by the "
        "ASSUMED value at scenario time), static distances, OSM landuse class, "
        "and 6 meteorological context covariates (per-year W4 external "
        "observations via the exact Phase 4 code path — contemporaneous, not "
        "target-derived; see the leakage audit). Phase 6/7/8 products are not "
        "training targets.",
        "",
        "## Assumed scenario parameter (ASSUMED)",
        "",
        f"`assumed_vegetation_change = density / 1000 x "
        f"{manifest['vegetation_scenario']['vegetation_change_per_1000_trees_ha']}` "
        "(vegetation-cover fraction added per 1,000 trees/ha).",
        "",
        manifest["vegetation_scenario"]["source"],
        "",
        "Densities 400 / 1,000 / 2,500 trees/ha are **planning-density "
        "scenarios — none is claimed scientifically optimal**; the supported "
        "optimal density will be evaluated after Phase 9 model improvements "
        "using cooling predictions.",
        "",
        "## Validation (PREDICTED — empirical skill only)",
        "",
        "Adjacent-block CV on non-locked blocks, LOBO over occupied blocks, "
        "locked geographic holdout [2, 9, 15, 23] (evaluated exactly once per "
        "model), and leave-one-transition-out temporal holdout.",
        "",
        "| model | eval | n | MAE (C) | RMSE (C) | R2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, res in manifest["models"].items():
        for eval_name, agg in (("CV5 (mean+/-std)", res.get("adjacent_block_cv")),
                               ("LOBO (mean+/-std)", res.get("lobo")),
                               ("time-holdout (mean+/-std)", res.get("time_holdout"))):
            if agg:
                lines.append(
                    f"| {name} | {eval_name} | {agg['n_total_eval_rows']:,} | "
                    f"{agg['mae_C_mean']:.3f} +/- {agg['mae_C_std']:.3f} | "
                    f"{agg['rmse_C_mean']:.3f} +/- {agg['rmse_C_std']:.3f} | "
                    f"{agg['r2_mean']:.3f} +/- {agg['r2_std']:.3f} |")
        lk = res.get("locked_holdout", {})
        if lk:
            lines.append(
                f"| {name} | LOCKED (once) | {lk['n']:,} | {lk['mae_C']:.3f} | "
                f"{lk['rmse_C']:.3f} | {lk['r2']:.3f} |")
    lines += [
        "",
        f"Selected model: **{selected}** (locked-holdout MAE, time-holdout MAE "
        "tie-breaker).",
        "",
        "## Feature importance (PREDICTED — diagnostic only)",
        "",
    ]
    for name, res in manifest["models"].items():
        lines.append(f"Top-10 (`{name}`; built-in + permutation MAE increase):")
        lines.append("")
        lines.append("| feature | built-in | permutation dMAE |")
        lines.append("|---|---:|---:|")
        for r in res.get("top_features", []):
            lines.append(f"| {r['feature']} | {r.get('importance', float('nan')):.4f} | "
                         f"{r.get('permutation_importance_mae', float('nan')):.4f} |")
        lines.append("")
        rs = res.get("locked_residual_summary")
        if rs:
            lines.append(
                f"Locked residuals (`{name}`): mean {rs['mean_C']:+.3f} C, "
                f"std {rs['std_C']:.3f} C, tails [{rs['p01_C']:+.2f}, "
                f"{rs['p99_C']:+.2f}] C (min {rs['min_C']:+.2f} / "
                f"max {rs['max_C']:+.2f}), |resid| > 2 C on "
                f"{100*rs['frac_abs_gt_2C']:.2f}% of locked px.")
            lines.append("")
    checks = manifest.get("sanity_checks", {})
    sens = checks.get("vegetation_change_sensitivity", [])
    if sens:
        lines += [
            "## Vegetation-change sensitivity (PREDICTED)",
            "",
            "Mean predicted delta LST over a fixed 15k-px sample as the "
            "ASSUMED vegetation change varies (informational; the model is "
            "not forced to be monotone in density):",
            "",
            "| assumed veg change | mean predicted delta LST (C) | mean predicted cooling (C) |",
            "|---:|---:|---:|",
        ]
        for r in sens:
            lines.append(f"| {r['assumed_vegetation_change']:.2f} | "
                         f"{r['mean_predicted_delta_lst_C']:+.4f} | "
                         f"{r['mean_predicted_cooling_C']:+.4f} |")
        dist = checks.get("predicted_cooling_distribution_2026", {})
        if dist:
            lines += [
                "",
                f"2026 predicted-cooling distribution (zone means): min "
                f"{dist['min']:.3f}, p05 {dist['p05']:.3f}, median "
                f"{dist['median']:.3f}, p95 {dist['p95']:.3f}, max "
                f"{dist['max']:.3f} C; extreme pixels (|delta| > 10 C): "
                f"{checks.get('extreme_prediction_count_abs_delta_gt_10C', 0)}.",
            ]
    lines += [
        "",
        "## 2026 primary scenario (ASSUMED change -> PREDICTED LST)",
        "",
        "Primary scenario: 2026 `v2_constrained` priority zones (Phase 8 "
        "PRIMARY). Predicted cooling = -predicted delta LST; positive = "
        "predicted LST reduction under the assumed change.",
        "",
        "| density trees/ha | zones | plantable ha | trees | mean predicted cooling C | mean predicted post LST C |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    summary = zone_predictions.groupby("tree_density_trees_per_ha").agg(
        zones=("zone_id", "count"),
        trees=("estimated_tree_count", "sum"),
        plantable_ha=("plantable_area_ha", "sum"),
        cooling=("predicted_cooling_C", "mean"),
        post_lst=("predicted_post_lst_C", "mean"),
    ).reset_index()
    for r in summary.itertuples():
        lines.append(
            f"| {int(r.tree_density_trees_per_ha)} | {int(r.zones)} | {r.plantable_ha:.2f} | "
            f"{int(r.trees):,} | {r.cooling:.3f} | {r.post_lst:.3f} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- Only four annual transitions are available in the current V2 panel.",
        "- `vegetation_change` is observed historically but ASSUMED for the scenario.",
        "- The model is empirical; spatial/temporal holdout skill bounds the "
        "confidence for planning use. No causal tree-cooling claim is made.",
        "- No V2-observed tree-density-to-canopy conversion exists; the 0.05 "
        "per-1,000-trees/ha vegetation-change parameter is an explicit "
        "scenario assumption.",
        "- Results are LST predictions, not air-temperature guarantees.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_leakage_audit() -> list[dict]:
    """Predictor-by-predictor leakage audit (testable; written to diagnostics)."""
    return [
        {"predictor": "current_lst_C", "status": "allowed", "reason": "current transition baseline LST; target is future minus current LST"},
        {"predictor": "current_ndvi", "status": "allowed", "reason": "current-year vegetation state"},
        {"predictor": "current_ndbi", "status": "allowed", "reason": "current-year built-up spectral index"},
        {"predictor": "current_vegetation_cover", "status": "allowed", "reason": "current-year vegetation fraction"},
        {"predictor": "vegetation_change", "status": "intervention_variable", "reason": "observed during historical transition; replaced by explicit assumed change in 2026 scenario"},
        {"predictor": "dist_road_m/dist_vegetation_m/dist_building_m/landuse_class", "status": "allowed", "reason": "static context available before target LST"},
        {"predictor": "current_met_* (6 W4 met covariates)", "status": "allowed", "reason": "contemporaneous external meteorological observations of the transition start year (per-year point means, hours 04-06 UTC, May 1 - Jun 30, V1 matched IDW via the exact Phase 4 code path); NOT derived from the target delta LST"},
        {"excluded": "Phase 6 severity / Phase 7 suitability / Phase 8 tree counts", "status": "excluded_from_training_target", "reason": "derived downstream decision products, not observed cooling labels"},
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase3-root", default=str(PHASE3_ROOT_DEFAULT))
    ap.add_argument("--phase8-root", default=str(PHASE8_ROOT_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--report", default=str(REPORT_DEFAULT))
    ap.add_argument("--max-samples-per-transition", type=int, default=50_000)
    ap.add_argument("--vegetation-change-per-1000-trees-ha", type=float, default=0.05)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    phase3_root = Path(args.phase3_root)
    phase8_root = Path(args.phase8_root)
    out_dir = Path(args.out)
    for d in ("tables", "models", "diagnostics", "rasters"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    preflight(phase3_root, phase8_root, GRID_FILE_DEFAULT)

    veg_cfg = VegetationScenarioConfig(
        vegetation_change_per_1000_trees_ha=args.vegetation_change_per_1000_trees_ha
    )
    panel = build_transition_panel(phase3_root, out_dir / "tables", args.max_samples_per_transition, RANDOM_SEED)
    model_dir = out_dir / "models"
    results = {}
    for name, spec in model_specs().items():
        log(f"training/evaluating {name}")
        results[name] = evaluate_model(name, spec, panel, model_dir)
    selected_name = select_model(results)
    selected_model = joblib.load(model_dir / f"{selected_name}_model.joblib")
    zone_predictions = build_scenario_predictions(selected_model, phase3_root, phase8_root, out_dir, veg_cfg)
    checks = sanity_checks(selected_model, panel, zone_predictions, out_dir, veg_cfg)

    panel_summary = pd.read_csv(out_dir / "tables" / "panel_summary.csv").to_dict(orient="records")
    leakage_audit = build_leakage_audit()
    dump_json(leakage_audit, out_dir / "diagnostics" / "leakage_audit.json")

    input_hashes = {
        str(phase8_root / PRIMARY_SCENARIO / "tables" / f"tree_requirement_by_zone_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}.csv"):
            sha256_file(phase8_root / PRIMARY_SCENARIO / "tables" / f"tree_requirement_by_zone_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}.csv"),
        str(phase8_root / PRIMARY_SCENARIO / "rasters" / f"available_planting_space_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}.tif"):
            sha256_file(phase8_root / PRIMARY_SCENARIO / "rasters" / f"available_planting_space_{PRIMARY_SCENARIO}_{SCENARIO_YEAR}.tif"),
    }
    manifest = {
        "stage": "phase9_temperature_reduction_prediction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_framing": "Predicted LST reduction under an assumed planting/vegetation scenario; not proof of causal tree cooling. LST predictions, not air-temperature guarantees.",
        "density_framing": ("Densities 400/1000/2500 trees/ha are planning-density "
                            "scenarios, none scientifically optimal; optimal "
                            "density will be evaluated after Phase 9 using "
                            "cooling predictions."),
        "target": {
            "name": "delta_lst_C",
            "definition": "LST(year+1, W4) - LST(year, W4)",
            "cooling_definition": "predicted_cooling_C = -predicted_delta_lst_C; positive means predicted LST reduction",
        },
        "features": list(FEATURE_COLS),
        "excluded_training_targets": ["phase6 severity", "phase7 suitability", "phase8 tree counts"],
        "training_sample_size": int(len(panel)),
        "block_count": int(panel["spatial_block_id"].nunique()),
        "transition_count": int(panel["transition"].nunique()),
        "panel_summary": panel_summary,
        "vegetation_scenario": asdict(veg_cfg),
        "models": results,
        "selected_model": selected_name,
        "scenario_prediction_table": str(out_dir / "tables" / "phase9_zone_cooling_predictions_2026.csv"),
        "sanity_checks": checks,
        "input_hashes": input_hashes,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "total_wall_s": time.perf_counter() - t0,
    }
    dump_json(manifest, out_dir / "phase9_manifest.json")
    write_report(Path(args.report), manifest, zone_predictions)
    log(f"complete in {manifest['total_wall_s']:.1f}s; selected={selected_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
