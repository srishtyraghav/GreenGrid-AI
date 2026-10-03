"""Freeze the exp_spatial 3-class XGBoost model as the official Phase 5
primary production model, after verifying it reproduces the reported locked
metrics (63.97% accuracy / 0.6315 macro-F1) under the exact experiment
protocol.

What this script does (read-only w.r.t. all historical artifacts):
  1. Rebuilds the exp_spatial dataset with the evaluator's own functions
     (identical merge, encoding, predictors -> 178 features).
  2. Refits XGBoost on the non-locked training blocks and re-evaluates on the
     locked holdout [2, 9, 15, 23].
  3. ASSERTS the locked accuracy/macro-F1 match the experiment record to
     1e-9. Aborts without writing anything if not.
  4. Writes frozen production artifacts to data/processed/phase5_production_3class/:
     - phase5_primary_xgb_3class.json      (XGBoost booster, native format)
     - phase5_primary_xgb_3class_features.json (178-feature schema, ordered)
     - feature_manifest.csv                (feature -> source group mapping)
     - locked_validation.json              (verified locked metrics + CM + thresholds)
     - full_validation_record.json         (copy of exp_spatial CV5/LOBO/locked record)
     - production_model_metadata.json      (production metadata / provenance)

Historical artifacts (lulc_outputs/, baseline_5yr_4class/, tier12_outputs/,
all experiment JSONs) are NOT modified.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from geographic_holdout_eval import (  # noqa: E402
    build_y,
    load_and_merge,
    locked_block_ids,
    score_3class,
    select_predictors,
    tercile_thresholds,
)
from models.config import PREDICTOR_VARS, XGBOOST_PARAMS  # noqa: E402
from models.dataset import encode_predictors  # noqa: E402

DATASET = PROJECT / "data/processed/lulc_outputs/combined_urban_environmental_dataset.csv"
SPATIAL_CACHE = PROJECT / "data/processed/lulc_outputs/phase5_tables/spatial_neighbourhood_features.csv"
MORPH_CACHE = PROJECT / "data/processed/lulc_outputs/phase5_tables/morphology_features.csv"
EXTRA_CSV = PROJECT / "data/processed/experiments/exp_spatial_context_features.csv"
EXP_RECORD = PROJECT / "data/processed/experiments/exp_spatial_3class_geoeval.json"
OUT_DIR = PROJECT / "data/processed/phase5_production_3class"

MET = ["met_t2m_c", "met_rh_pct", "met_wind_kmh", "met_precip_mm", "met_ssr_wm2", "met_swc_m3m3"]
NEW_SPATIAL = (
    [f"{b}_std{w}" for b in ("ndvi", "ndbi", "ndre", "ndmi", "mndwi", "bsi", "vegetation_cover") for w in (3, 5, 11)]
    + [f"{b}_mean{w}" for b in ("mndwi", "ndre") for w in (3, 5, 11)]
    + [f"{b}_range{w}" for b in ("ndvi", "ndbi") for w in (3, 5, 11)]
)
PREDICTOR_COLS = PREDICTOR_VARS + MET + NEW_SPATIAL

# Ground truth from data/processed/experiments/exp_spatial_eval.log
EXPECTED_LOCKED = {"accuracy": 0.6396665686967289, "macro_f1": 0.6314607179023212}
LOCKED_BLOCKS = [2, 9, 15, 23]
TOL = 1e-9


def source_group(feat: str) -> str:
    if feat.startswith("met_"):
        return "meteorological (Open-Meteo/ERA5, exp_met)"
    if any(feat.startswith(p) for p in ("ndvi_", "ndbi_", "ndre_", "ndmi_", "mndwi_", "bsi_", "vegetation_cover_")) and (
            "_std" in feat or "_range" in feat or feat.startswith(("mndwi_mean", "ndre_mean"))):
        return "spatial-context new (33, build_spatial_context_features.py)"
    if feat.startswith(("landuse_frac", "building_pixel_fraction", "road_pixel_fraction",
                        "vegetation_pixel_fraction", "ndvi_contrast", "ndbi_contrast")):
        return "morphology cache (50-500m)"
    if "_mean" in feat and any(feat.startswith(p) for p in ("ndvi", "ndbi", "vegetation_cover", "ndmi", "bsi", "dist_")):
        return "spatial cache 3/5/11 means (Tier 2)"
    return "base predictor (PREDICTOR_VARS)"


def main() -> int:
    print("[FREEZE] rebuilding exp_spatial dataset ...")
    df = load_and_merge(str(DATASET), str(SPATIAL_CACHE), str(MORPH_CACHE), extra_csv=str(EXTRA_CSV))
    available = select_predictors(df, PREDICTOR_COLS)
    assert len(available) == len(PREDICTOR_COLS), f"missing predictors: {set(PREDICTOR_COLS) - set(available)}"
    df = df.dropna(subset=available + ["lst_C"]).reset_index(drop=True)
    X, feature_names = encode_predictors(df, predictor_cols=available)
    assert len(feature_names) == 178, f"expected 178 encoded features, got {len(feature_names)}"
    groups = df["spatial_block_id"].astype(int).values
    locked = [int(b) for b in locked_block_ids(df)]
    assert locked == LOCKED_BLOCKS, f"locked blocks changed: {locked}"
    print(f"[FREEZE] rows={len(df)} features={len(feature_names)} locked={locked}")

    test_mask = np.isin(groups, locked)
    train_mask = ~test_mask
    y = build_y(df, "3class", train_mask=train_mask)

    print("[FREEZE] fitting production XGBoost (identical params to experiment; "
          "n_jobs=1 single-thread for bit-exact reproducibility) ...")
    # Determinism (verified by scripts/diagnose_xgb_determinism.py, 2026-10-03):
    # XGBoost's multi-threaded histogram construction is documented as NOT
    # bit-reproducible run-to-run; a multithreaded refit scored 64.2230% vs the
    # experiment record 63.9667%. Single-threaded (n_jobs=1) fits are fully
    # deterministic and reproduce the experiment record EXACTLY
    # (accuracy 0.6396665687, macro_f1 0.6314607179, predictions identical).
    params = {**XGBOOST_PARAMS, "num_class": 3, "n_jobs": 1}
    model = XGBClassifier(**params)
    model.fit(X[train_mask], y[train_mask])
    preds = model.predict(X[test_mask])
    locked_scores = score_3class(y[test_mask].values, preds)
    print(f"[FREEZE] locked = {{'accuracy': {locked_scores['accuracy']}, 'macro_f1': {locked_scores['macro_f1']}}}")

    for k, expected in EXPECTED_LOCKED.items():
        got = locked_scores[k]
        if abs(got - expected) > TOL:
            print(f"[FREEZE] VERIFICATION FAILED: {k} expected {expected}, got {got}")
            return 1
    print("[FREEZE] VERIFICATION PASSED: locked metrics reproduce the experiment record exactly")

    thresholds = tercile_thresholds(df, train_mask)
    exp_record = json.loads(EXP_RECORD.read_text())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUT_DIR / "phase5_primary_xgb_3class.json"))
    (OUT_DIR / "phase5_primary_xgb_3class_features.json").write_text(
        json.dumps({"n_features": len(feature_names), "feature_names": feature_names}, indent=2))
    pd.DataFrame({"feature": feature_names,
                  "source_group": [source_group(f) for f in feature_names]}
                 ).to_csv(OUT_DIR / "feature_manifest.csv", index=False)
    (OUT_DIR / "locked_validation.json").write_text(json.dumps({
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verified_metrics": locked_scores,
        "expected_metrics_from_experiment_record": EXPECTED_LOCKED,
        "tolerance": TOL,
        "locked_blocks": locked,
        "locked_rule": "occupied_block_ids[2::6]",
        "production_thresholds_lst_c_tertiles": thresholds,
        "n_train_rows": int(train_mask.sum()), "n_locked_rows": int(test_mask.sum()),
    }, indent=2))
    shutil.copyfile(EXP_RECORD, OUT_DIR / "full_validation_record.json")

    metadata = {
        "model_id": "phase5_primary_xgb_3class",
        "status": "PRODUCTION - official Phase 5 primary model",
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "XGBoost classifier",
        "model_params": {**XGBOOST_PARAMS, "num_class": 3, "n_jobs": 1},
        "execution_policy": ("Fitted single-threaded (n_jobs=1): XGBoost multi-thread "
                             "histogram building is not bit-reproducible run-to-run "
                             "(multithreaded refit scored 64.2230% vs record 63.9667%); "
                             "the single-threaded fit reproduces the experiment record "
                             "exactly. See scripts/diagnose_xgb_determinism.py."),
        "task": "3class",
        "target_definition": ("Per-year LST tertiles (Low/Moderate/High); thresholds computed from "
                              "training rows only, per year. Locked evaluation uses thresholds from "
                              "non-locked training blocks (see locked_validation.json)."),
        "n_features_encoded": len(feature_names),
        "dataset": {
            "table": "data/processed/lulc_outputs/combined_urban_environmental_dataset.csv",
            "spatial_cache": str(SPATIAL_CACHE.relative_to(PROJECT)),
            "morphology_cache": str(MORPH_CACHE.relative_to(PROJECT)),
            "extra_features": str(EXTRA_CSV.relative_to(PROJECT)),
            "n_rows": len(df), "years": sorted(df["year"].unique().tolist()),
        },
        "validation": {
            "protocol": "5-fold adjacent-block CV + LOBO + deterministic locked geographic holdout",
            "locked_blocks": locked,
            "locked_accuracy": locked_scores["accuracy"],
            "locked_macro_f1": locked_scores["macro_f1"],
            "full_record": "full_validation_record.json (exp_spatial_3class_geoeval.json copy)",
        },
        "historical_baselines_preserved": [
            "data/processed/lulc_outputs/ (frozen 4-class XGB+LULC model, 50.02% locked)",
            "data/processed/baseline_5yr_4class/",
            "data/processed/tier12_outputs/",
            "data/processed/experiments/ (all experiment records)",
        ],
        "replaces_as_primary": "lulc_outputs 4-class model (50.02% locked) - retained as historical baseline",
        "downstream_use": ("Phase 6+ severity mapping and suitability work must use this model with the "
                           "178-feature schema in phase5_primary_xgb_3class_features.json and the feature "
                           "construction documented in reports/experiments/exp_spatial_results.md."),
    }
    (OUT_DIR / "production_model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"[FREEZE] wrote production artifacts to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
