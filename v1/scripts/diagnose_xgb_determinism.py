"""Diagnose XGBoost fit-to-fit reproducibility on the production locked split.

The first freeze attempt refit the exp_spatial configuration and got
64.2230% locked accuracy vs the experiment-record 63.9667% — identical data,
identical params, identical code path. Suspected cause: XGBoost multi-thread
histogram construction is not bit-reproducible run-to-run (XGBoost FAQ:
results are only reproducible with fixed nthread; even then reduction order
can vary). This script verifies that hypothesis:

  A1/A2: exact experiment params (n_jobs=-1), twice  -> expect DIFFERENT preds
  B1/B2: n_jobs=1 (single thread), twice             -> expect IDENTICAL preds
  C1/C2: n_jobs=-1 + deterministic_histogram=True    -> expect IDENTICAL preds
                                                  (param is accepted in XGB 3.x?)

Prints locked accuracy/macro-F1 for every fit. Read-only: no artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from geographic_holdout_eval import (  # noqa: E402
    build_y, load_and_merge, locked_block_ids, score_3class, select_predictors,
)
from models.config import PREDICTOR_VARS, XGBOOST_PARAMS  # noqa: E402
from models.dataset import encode_predictors  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

DATASET = PROJECT / "data/processed/lulc_outputs/combined_urban_environmental_dataset.csv"
SPATIAL_CACHE = PROJECT / "data/processed/lulc_outputs/phase5_tables/spatial_neighbourhood_features.csv"
MORPH_CACHE = PROJECT / "data/processed/lulc_outputs/phase5_tables/morphology_features.csv"
EXTRA_CSV = PROJECT / "data/processed/experiments/exp_spatial_context_features.csv"
MET = ["met_t2m_c", "met_rh_pct", "met_wind_kmh", "met_precip_mm", "met_ssr_wm2", "met_swc_m3m3"]
NEW_SPATIAL = (
    [f"{b}_std{w}" for b in ("ndvi", "ndbi", "ndre", "ndmi", "mndwi", "bsi", "vegetation_cover") for w in (3, 5, 11)]
    + [f"{b}_mean{w}" for b in ("mndwi", "ndre") for w in (3, 5, 11)]
    + [f"{b}_range{w}" for b in ("ndvi", "ndbi") for w in (3, 5, 11)]
)
PREDICTOR_COLS = PREDICTOR_VARS + MET + NEW_SPATIAL


def fit_and_score(Xtr, ytr, Xte, yte, tag, **overrides):
    params = {**XGBOOST_PARAMS, "num_class": 3}
    params.pop("n_jobs", None)
    model = XGBClassifier(**params, **overrides)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    s = score_3class(yte, preds)
    print(f"[DIAG] {tag}: acc={s['accuracy']:.10f} f1={s['macro_f1']:.10f}", flush=True)
    return preds


def main() -> int:
    df = load_and_merge(str(DATASET), str(SPATIAL_CACHE), str(MORPH_CACHE), extra_csv=str(EXTRA_CSV))
    available = select_predictors(df, PREDICTOR_COLS)
    df = df.dropna(subset=available + ["lst_C"]).reset_index(drop=True)
    X, feature_names = encode_predictors(df, predictor_cols=available)
    assert len(feature_names) == 178
    groups = df["spatial_block_id"].astype(int).values
    assert locked_block_ids(df) == [2, 9, 15, 23]
    test_mask = np.isin(groups, [2, 9, 15, 23])
    y = build_y(df, "3class", train_mask=~test_mask)
    Xtr, ytr = X[~test_mask], y[~test_mask]
    Xte, yte = X[test_mask], y[test_mask].values
    print(f"[DIAG] rows={len(df)} train={len(Xtr)} test={len(Xte)}", flush=True)

    a1 = fit_and_score(Xtr, ytr, Xte, yte, "A1 n_jobs=-1 (experiment exact)", n_jobs=-1)
    a2 = fit_and_score(Xtr, ytr, Xte, yte, "A2 n_jobs=-1 (repeat)          ", n_jobs=-1)
    print(f"[DIAG] A1 vs A2 identical predictions: {np.array_equal(a1, a2)}", flush=True)

    b1 = fit_and_score(Xtr, ytr, Xte, yte, "B1 n_jobs=1  (single thread)   ", n_jobs=1)
    b2 = fit_and_score(Xtr, ytr, Xte, yte, "B2 n_jobs=1  (repeat)          ", n_jobs=1)
    print(f"[DIAG] B1 vs B2 identical predictions: {np.array_equal(b1, b2)}", flush=True)

    try:
        c1 = fit_and_score(Xtr, ytr, Xte, yte, "C1 n_jobs=-1 + deterministic_histogram=True",
                           n_jobs=-1, deterministic_histogram=True)
        c2 = fit_and_score(Xtr, ytr, Xte, yte, "C2 repeat                      ",
                           n_jobs=-1, deterministic_histogram=True)
        print(f"[DIAG] C1 vs C2 identical predictions: {np.array_equal(c1, c2)}", flush=True)
        print(f"[DIAG] C1 vs B1 identical predictions: {np.array_equal(c1, b1)}", flush=True)
    except Exception as e:
        print(f"[DIAG] deterministic_histogram not usable: {type(e).__name__}: {e}", flush=True)

    print(f"[DIAG] experiment record:            acc=0.6396665687 f1=0.6314607179", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
