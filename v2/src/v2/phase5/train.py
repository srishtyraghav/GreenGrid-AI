"""V2 Phase 5 training — UHI 3-class XGBoost model, frozen V1 protocol.

Protocol is FROZEN from V1's production model; there is no experiment search.
Sources (read-only):
  - v1/src/models/config.py           XGBOOST_PARAMS (verbatim below)
  - v1/scripts/geographic_holdout_eval.py  build_y 3class branch, tercile_thresholds,
                                      score_3class, per_class_metrics
  - v1/src/models/validation.py       create_spatial_folds / get_fold_blocks
                                      (GroupKFold on spatial_block_id)
  - v1/scripts/freeze_phase5_production.py  single-threaded final refit + one
                                      locked evaluation, n_jobs=1 determinism policy
  - v1/data/processed/phase5_production_3class/production_model_metadata.json
                                      locked blocks [2, 9, 15, 23], model_params

FROZEN XGBOOST PARAMS (verbatim V1, extracted 2026-10 from v1/src/models/config.py
::XGBOOST_PARAMS; the frozen V1 booster JSON confirms them: num_trees=600 =
200 estimators x 3 classes, num_class=3, objective multi:softprob, depth-6
trees). V1 set no min_child_weight / reg_alpha / reg_lambda -> XGBoost
defaults (min_child_weight=1, reg_alpha=0, reg_lambda=1); those defaults are
part of the frozen protocol:

    n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
    random_state=42, n_jobs=-1 (CV/LOBO default threading)
    + num_class=3

Determinism policy (V1-diagnosed via scripts/diagnose_xgb_determinism.py):
XGBoost multi-thread histogram building is NOT bit-reproducible run-to-run.
The FINAL refit and the LOCKED evaluation therefore run single-threaded
(n_jobs=1) and the locked set is evaluated EXACTLY ONCE, under a loud banner;
no metric-based decisions are made after that point. CV/LOBO fits use the
verbatim default threading (n_jobs=-1).

Stages (order fixed):
  a. load 5 parquets, schema-assert the 178 (names + hash), banned-column assert
  b. target: per-year LST tertiles from TRAINING rows only, inside every split
  c. 5-fold adjacent-block CV over the non-locked OCCUPIED blocks (V1 fold
     mapping; the V2 sample occupies 20 of 25 blocks, 16 of them non-locked)
  d. LOBO over all 25 pinned blocks (24 train / 1 test)
  e. FINAL single-threaded refit on all non-locked rows; LOCKED eval once
  f. artifacts + manifest
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from v2.common import (
    HOLDOUT_BLOCKS,
    PROJECT_ROOT,
    dump_json,
    json_safe,
    load_frozen_schema,
    schema_hash,
    sha256_file,
)

# ---------------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------------

METADATA_COLS = ("row", "col", "spatial_block_id", "lst_C")
BANNED_PREDICTOR_COLS = {"lst_C", "row", "col", "spatial_block_id"}
CLASS_LABELS = ("Low", "Moderate", "High")          # classes 0, 1, 2
N_CV_FOLDS = 5
N_SPATIAL_BLOCKS = 25                                # 5x5 pinned V1 tiling
TARGET_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)            # per-year LST tertiles

# Verbatim V1 XGBOOST_PARAMS (v1/src/models/config.py) + num_class=3.
# min_child_weight / reg_alpha / reg_lambda are absent in V1 -> XGBoost
# defaults (1 / 0 / 1), which are part of the frozen protocol.
XGB_PARAMS_V1 = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
    "num_class": 3,
}
# The final refit + locked evaluation pin single-threaded execution
# (V1 determinism diagnosis; multithreaded refit is not bit-reproducible).
FINAL_REFIT_NJOBS = 1

LOCKED_BANNER = """
================================================================================
  LOCKED GEOGRAPHIC HOLDOUT EVALUATION - BLOCKS {locked}
  FINAL MODEL (single-threaded n_jobs=1 refit on the non-locked blocks)
  EVALUATED EXACTLY ONCE. NO METRIC-BASED DECISIONS ARE MADE AFTER THIS POINT.
================================================================================
"""


# ---------------------------------------------------------------------------
# Target machinery (port of V1 geographic_holdout_eval.build_y / tercile_thresholds)
# ---------------------------------------------------------------------------

def tercile_thresholds(df: pd.DataFrame, train_mask: np.ndarray) -> dict:
    """Exact per-year LST tertiles (1/3, 2/3) from TRAIN ROWS ONLY.

    Faithful port of V1 ``tercile_thresholds``: ``base = df[train_mask]``,
    ``vals_train.quantile([1/3, 2/3])`` per year (pandas linear interpolation).
    Returns ``{year_int: (t1, t2)}``. Validation/locked LST never enters.
    """
    base = df[train_mask]
    out = {}
    for year in sorted(df["year"].unique()):
        vals_train = base.loc[base["year"] == year, "lst_C"]
        if len(vals_train) == 0:
            raise ValueError(f"No training LST values for year {year}.")
        q = vals_train.quantile(list(TARGET_QUANTILES)).values
        out[int(year)] = (float(q[0]), float(q[1]))
    return out


def assign_classes(lst_values: np.ndarray, years: np.ndarray,
                   thresholds: dict) -> np.ndarray:
    """LST -> 0/1/2 via per-year train tertiles, ``searchsorted(side="right")``
    — the exact frozen V1 3class code path (V1 build_y: ``np.searchsorted(q,
    vals, side="right")``). With side="right", class == number of thresholds
    <= lst, i.e. Low: lst < t1; Moderate: t1 <= lst < t2; High: lst >= t2.
    (Exact ties have measure zero on continuous LST; the replication is exact,
    not a reinterpretation of the 4-class inclusive-upper rule.)"""
    lst_values = np.asarray(lst_values, dtype=float)
    years = np.asarray(years)
    cls = np.zeros(len(lst_values), dtype=int)
    for year, (t1, t2) in thresholds.items():
        m = years == int(year)
        if m.any():
            cls[m] = np.searchsorted(np.array([t1, t2]), lst_values[m],
                                     side="right")
    return cls


def build_target(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Targets for ALL rows of df, thresholds from train rows only (no leakage).

    Returns (y int64 ndarray aligned to df.index, thresholds {year: (t1, t2)}).
    """
    thresholds = tercile_thresholds(df, train_mask)
    y = assign_classes(df["lst_C"].to_numpy(), df["year"].to_numpy(), thresholds)
    return y, thresholds


# ---------------------------------------------------------------------------
# Fold mapping (port of V1 validation.py create_spatial_folds / get_fold_blocks)
# ---------------------------------------------------------------------------

def adjacent_block_folds(df: pd.DataFrame, n_splits: int = N_CV_FOLDS,
                         locked_blocks: tuple = HOLDOUT_BLOCKS) -> list[dict]:
    """V1 adjacent-block CV mapping over the NON-LOCKED OCCUPIED blocks.

    Faithful port of V1 ``create_spatial_folds`` (GroupKFold(n_splits=5) on
    ``spatial_block_id``) + ``get_fold_blocks`` (sorted unique train/val block
    ids per fold), restricted to the blocks present in ``df`` minus the locked
    holdout. The training universe is dynamic: all 25 minus 4 locked = 21
    possible blocks; the real V2 sample occupies 16 of them (5 pinned blocks
    have no rows). Deviation vs V1, frozen by the V2 Phase-5 brief: V1's own
    CV5 swept all occupied blocks INCLUDING the locked ones; V2 excludes the
    locked blocks from CV (they are evaluated only once, in stage e).

    Returns a list of dicts: fold (1-based), train_blocks, val_blocks,
    train_idx, val_idx (positions into ``df``).
    """
    non_locked = ~df["spatial_block_id"].isin(locked_blocks)
    sub_idx = np.flatnonzero(non_locked.to_numpy())
    groups = df["spatial_block_id"].astype(int).to_numpy()[sub_idx]
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for fold, (tr, va) in enumerate(
            gkf.split(np.zeros(len(sub_idx)), groups=groups), start=1):
        tr_blocks = sorted(np.unique(groups[tr]).tolist())
        va_blocks = sorted(np.unique(groups[va]).tolist())
        if set(tr_blocks) & set(va_blocks):
            raise ValueError(f"Fold {fold}: train/val block overlap.")
        folds.append({
            "fold": fold,
            "train_blocks": tr_blocks,
            "val_blocks": va_blocks,
            "train_idx": sub_idx[tr],
            "val_idx": sub_idx[va],
        })
    return folds


# ---------------------------------------------------------------------------
# Metrics (port of V1 score_3class / per_class_metrics)
# ---------------------------------------------------------------------------

def per_class_metrics(cm: list) -> dict:
    """precision/recall/F1/support per class from a 3x3 confusion matrix
    (rows = true, cols = pred, order Low/Moderate/High) — V1 convention."""
    out = {}
    for i, name in enumerate(CLASS_LABELS):
        tp = cm[i][i]
        support = sum(cm[i])
        pred_k = sum(row[i] for row in cm)
        prec = tp / pred_k if pred_k else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[name] = {"precision": float(prec), "recall": float(rec),
                     "f1": float(f1), "support": int(support)}
    return out


def score_3class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """accuracy + macro-F1 + 3x3 confusion (labels [0,1,2]) + per-class."""
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "confusion_matrix": cm,
        "per_class": per_class_metrics(cm),
    }


def class_balance_per_year(y: np.ndarray, years: np.ndarray) -> dict:
    """Per-year class counts/fractions for the sanity gate."""
    out = {}
    for year in sorted(np.unique(years)):
        m = years == year
        n = int(m.sum())
        counts = np.bincount(y[m], minlength=3)
        out[int(year)] = {
            "n": n,
            "counts": {CLASS_LABELS[k]: int(counts[k]) for k in range(3)},
            "fractions": {CLASS_LABELS[k]: float(counts[k] / n) for k in range(3)},
        }
    return out


# ---------------------------------------------------------------------------
# Data loading + gates
# ---------------------------------------------------------------------------

def load_features(features_dir: Path, schema: list[str]) -> pd.DataFrame:
    """Load the 5 yearly parquets; schema-assert the 178 predictors (names +
    order + hash) and assert banned columns are not predictors."""
    expected_meta = list(METADATA_COLS)
    frames = []
    for year in range(2022, 2027):
        path = Path(features_dir) / f"features_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing feature parquet: {path}")
        df = pd.read_parquet(path)
        meta = [c for c in df.columns if c in METADATA_COLS]
        if meta != expected_meta or len(df.columns) != 4 + 178:
            raise AssertionError(
                f"{path.name}: expected metadata cols {expected_meta} + 178 "
                f"predictors (182 cols), got {list(df.columns)[:8]}... "
                f"({len(df.columns)} cols)")
        predictors = [c for c in df.columns if c not in METADATA_COLS]
        if predictors != schema:
            raise AssertionError(
                f"{path.name}: predictor columns do not match the frozen 178 "
                f"schema (names/order).")
        banned = BANNED_PREDICTOR_COLS.intersection(predictors)
        if banned:
            raise AssertionError(f"{path.name}: banned columns in predictors: {banned}")
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    if full[list(schema)].isna().any().any():
        raise AssertionError("NaN in predictors (features must be NaN-free).")
    return full


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_cv(df: pd.DataFrame, X: pd.DataFrame, out_dir: Path) -> dict:
    """Stage c: 5-fold adjacent-block CV (default threading), OOF predictions."""
    t0 = time.perf_counter()
    folds = adjacent_block_folds(df)
    oof_pred = np.full(len(df), -1, dtype=int)
    oof_true = np.full(len(df), -1, dtype=int)
    oof_fold = np.full(len(df), -1, dtype=int)
    rows, cms = [], []
    for f in folds:
        tr, va = f["train_idx"], f["val_idx"]
        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[tr] = True
        y, _thr = build_target(df, train_mask)          # train-only thresholds
        model = XGBClassifier(**XGB_PARAMS_V1)
        model.fit(X.iloc[tr], y[tr])
        preds = model.predict(X.iloc[va])
        s = score_3class(y[va], preds)
        oof_pred[va] = preds
        oof_true[va] = y[va]                            # this fold's own target
        oof_fold[va] = f["fold"]
        rows.append({"fold": f["fold"], "n_train": int(len(tr)),
                     "n_val": int(len(va)),
                     "train_blocks": f["train_blocks"], "val_blocks": f["val_blocks"],
                     "accuracy": s["accuracy"], "macro_f1": s["macro_f1"]})
        for i in range(3):
            for j in range(3):
                cms.append({"fold": f["fold"], "true_class": CLASS_LABELS[i],
                            "pred_class": CLASS_LABELS[j],
                            "count": int(s["confusion_matrix"][i][j])})
        print(f"[CV ] fold {f['fold']}: acc={s['accuracy']:.6f} "
              f"macro_f1={s['macro_f1']:.6f} n_val={len(va)}", flush=True)

    acc = np.array([r["accuracy"] for r in rows])
    f1 = np.array([r["macro_f1"] for r in rows])
    cv = {
        "per_fold": rows,
        "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std()),
        "macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std()),
        "wall_s": time.perf_counter() - t0,
    }
    pd.DataFrame(rows).drop(columns=["train_blocks", "val_blocks"]).to_csv(
        out_dir / "cv_metrics.csv", index=False)
    pd.DataFrame(cms).to_csv(out_dir / "cv_confusion.csv", index=False)
    oof = df.loc[oof_fold > 0, ["row", "col", "year"]].copy()
    oof["fold"] = oof_fold[oof_fold > 0]
    oof["y_true"] = oof_true[oof_fold > 0]
    oof["y_pred"] = oof_pred[oof_fold > 0]
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    return cv


def run_lobo(df: pd.DataFrame, X: pd.DataFrame, out_dir: Path) -> dict:
    """Stage d: leave-one-block-out over ALL 25 pinned blocks (24 train / 1 test).

    Blocks with no sampled rows are recorded with null metrics and no fit
    (documented in the manifest); the real V2 sample occupies 20 blocks and
    5 pinned blocks are empty.
    """
    t0 = time.perf_counter()
    rows = []
    groups = df["spatial_block_id"].astype(int).to_numpy()
    for bid in range(N_SPATIAL_BLOCKS):
        test_mask = groups == bid
        n_test = int(test_mask.sum())
        if n_test == 0:
            rows.append({"block": bid, "n_train": int((~test_mask).sum()),
                         "n_test": 0, "accuracy": None, "macro_f1": None,
                         "empty": True})
            print(f"[LOBO] block {bid:2d}: EMPTY (no rows) - recorded, no fit",
                  flush=True)
            continue
        train_mask = ~test_mask
        y, _thr = build_target(df, train_mask)
        model = XGBClassifier(**XGB_PARAMS_V1)
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        s = score_3class(y[test_mask], preds)
        rows.append({"block": bid, "n_train": int(train_mask.sum()),
                     "n_test": n_test, "accuracy": s["accuracy"],
                     "macro_f1": s["macro_f1"], "empty": False})
        print(f"[LOBO] block {bid:2d}: acc={s['accuracy']:.6f} "
              f"macro_f1={s['macro_f1']:.6f} n_test={n_test}", flush=True)
    real = [r for r in rows if not r["empty"]]
    acc = np.array([r["accuracy"] for r in real])
    f1 = np.array([r["macro_f1"] for r in real])
    lobo = {
        "per_block": rows,
        "n_blocks_empty": sum(1 for r in rows if r["empty"]),
        "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std()),
        "macro_f1_mean": float(f1.mean()), "macro_f1_std": float(f1.std()),
        "wall_s": time.perf_counter() - t0,
    }
    pd.DataFrame(rows).to_csv(out_dir / "lobo_metrics.csv", index=False)
    return lobo


def run_final_fit(df: pd.DataFrame, X: pd.DataFrame, out_dir: Path) -> dict:
    """Stage e: single-threaded refit on all non-locked rows; ONE locked eval."""
    t0 = time.perf_counter()
    groups = df["spatial_block_id"].astype(int).to_numpy()
    locked_mask = np.isin(groups, HOLDOUT_BLOCKS)
    train_mask = ~locked_mask
    y, thresholds = build_target(df, train_mask)
    # Class-balance sanity on the final train target (per year ~1/3 each).
    balance = class_balance_per_year(y[train_mask],
                                     df["year"].to_numpy()[train_mask])
    for year, entry in balance.items():
        fr = entry["fractions"]
        if not all(0.25 <= fr[k] <= 0.42 for k in CLASS_LABELS):
            raise AssertionError(f"class balance off for {year}: {fr}")

    print(f"[FINAL] refit on {int(train_mask.sum())} non-locked rows, "
          f"n_jobs=1 (single-thread, bit-reproducible) ...", flush=True)
    params = {**XGB_PARAMS_V1, "n_jobs": FINAL_REFIT_NJOBS}
    model = XGBClassifier(**params)
    model.fit(X[train_mask], y[train_mask])
    fit_s = time.perf_counter() - t0
    print(f"[FINAL] refit done in {fit_s:.1f}s", flush=True)

    n_locked = int(locked_mask.sum())
    locked = None
    if n_locked == 0:
        print("[LOCKED] WARNING: no locked-block rows in the data; "
              "locked evaluation SKIPPED (expected only for synthetic smoke).")
    else:
        banner = LOCKED_BANNER.format(locked=list(HOLDOUT_BLOCKS))
        print(banner, flush=True)
        t1 = time.perf_counter()
        preds = model.predict(X[locked_mask])          # single-threaded model
        eval_s = time.perf_counter() - t1
        s = score_3class(y[locked_mask], preds)
        locked = {**s, "locked_blocks": list(HOLDOUT_BLOCKS),
                  "n_train": int(train_mask.sum()), "n_locked": n_locked,
                  "evaluated_exactly_once": True, "eval_wall_s": eval_s}
        print(f"[LOCKED] accuracy={s['accuracy']:.6f} "
              f"macro_f1={s['macro_f1']:.6f}  <-- SINGLE NUMBER, EVALUATED ONCE",
              flush=True)
        pd.DataFrame({"row": df["row"].to_numpy()[locked_mask],
                      "col": df["col"].to_numpy()[locked_mask],
                      "year": df["year"].to_numpy()[locked_mask],
                      "y_true": y[locked_mask], "y_pred": preds}).to_csv(
            out_dir / "locked_predictions.csv", index=False)
        pd.DataFrame(s["confusion_matrix"],
                     index=CLASS_LABELS, columns=CLASS_LABELS).to_csv(
            out_dir / "locked_confusion.csv")
        pd.DataFrame([{"class": k, **v} for k, v in s["per_class"].items()]).to_csv(
            out_dir / "per_class_locked.csv", index=False)
        dump_json(locked, out_dir / "locked_metrics.json")

    model.save_model(str(out_dir / "phase5_primary_xgb_v2.json"))
    dump_json({str(k): {"t1": v[0], "t2": v[1]} for k, v in thresholds.items()},
              out_dir / "thresholds_by_year.json")
    for year, (t1, t2) in thresholds.items():
        assert t1 < t2, f"thresholds not monotonic for {year}: {t1} !< {t2}"
    return {"fit_wall_s": fit_s, "locked": locked, "thresholds": thresholds,
            "train_class_balance": balance}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema-json", default=None,
                    help="frozen 178-feature schema JSON (default: v2/reference)")
    ap.add_argument("--skip-lobo", action="store_true",
                    help="skip stage d (LOBO) — smoke/dry-run only")
    ap.add_argument("--skip-locked", action="store_true",
                    help="skip the locked evaluation — smoke/dry-run only")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = (load_frozen_schema(Path(args.schema_json)) if args.schema_json
              else load_frozen_schema())
    assert schema_hash(schema) == schema_hash(load_frozen_schema()), \
        "schema hash mismatch vs the frozen reference"
    assert not BANNED_PREDICTOR_COLS.intersection(schema), \
        f"banned columns inside the frozen schema: {BANNED_PREDICTOR_COLS & set(schema)}"

    print("[LOAD] reading 5 feature parquets + schema gates ...", flush=True)
    t0 = time.perf_counter()
    df = load_features(Path(args.features_dir), schema)
    load_s = time.perf_counter() - t0
    X = df[list(schema)]
    print(f"[LOAD] {len(df)} rows x {len(schema)} predictors "
          f"({load_s:.1f}s)", flush=True)

    wall = {"load_s": load_s}
    cv = run_cv(df, X, out_dir)
    wall["cv_s"] = cv["wall_s"]
    print(f"[CV ] mean acc={cv['accuracy_mean']:.6f}+-{cv['accuracy_std']:.6f} "
          f"mean macro_f1={cv['macro_f1_mean']:.6f}+-{cv['macro_f1_std']:.6f} "
          f"({cv['wall_s']:.1f}s)", flush=True)

    lobo = None
    if args.skip_lobo:
        print("[LOBO] SKIPPED (--skip-lobo, dry-run only)", flush=True)
        wall["lobo_s"] = None
    else:
        lobo = run_lobo(df, X, out_dir)
        wall["lobo_s"] = lobo["wall_s"]
        print(f"[LOBO] mean acc={lobo['accuracy_mean']:.6f}"
              f"+-{lobo['accuracy_std']:.6f} "
              f"mean macro_f1={lobo['macro_f1_mean']:.6f}"
              f"+-{lobo['macro_f1_std']:.6f} "
              f"({lobo['wall_s']:.1f}s, "
              f"{lobo['n_blocks_empty']} empty blocks)", flush=True)

    if args.skip_locked:
        print("[FINAL] SKIPPED (--skip-locked, dry-run only)", flush=True)
        final = None
        wall["final_fit_s"] = None
    else:
        final = run_final_fit(df, X, out_dir)
        wall["final_fit_s"] = final["fit_wall_s"]
        if final["locked"]:
            wall["locked_eval_s"] = final["locked"]["eval_wall_s"]

    # ---- manifest ------------------------------------------------------
    import pyarrow
    import sklearn
    import xgboost
    feat_dir = Path(args.features_dir)
    input_hashes = {p.name: sha256_file(p) for p in
                    sorted(feat_dir.glob("features_*.parquet"))}
    schema_path = Path(args.schema_json) if args.schema_json else \
        PROJECT_ROOT / "reference" / "phase5_primary_xgb_3class_features.json"
    input_hashes[schema_path.name] = sha256_file(schema_path)
    blocks_manifest = PROJECT_ROOT / "data" / "phase2" / "spatial_blocks_manifest.json"
    input_hashes[blocks_manifest.name] = sha256_file(blocks_manifest)

    y_final, thr_final = (build_target(df, ~df["spatial_block_id"].isin(HOLDOUT_BLOCKS))
                          if final else (None, None))

    manifest = {
        "model_id": "phase5_primary_xgb_v2",
        "protocol": "FROZEN replication of V1 phase5_primary_xgb_3class "
                    "(no experiment search)",
        "xgboost_params": XGB_PARAMS_V1,
        "final_refit_njobs": FINAL_REFIT_NJOBS,
        "xgb_unspecified_defaults": {
            "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1,
            "note": "absent in V1 XGBOOST_PARAMS -> XGBoost defaults are "
                    "part of the frozen protocol"},
        "frozen_v1_provenance": {
            "config": "v1/src/models/config.py::XGBOOST_PARAMS",
            "booster_json": "v1/data/processed/phase5_production_3class/"
                            "phase5_primary_xgb_3class.json "
                            "(num_trees=600 = 200x3 classes, num_class=3, "
                            "objective multi:softprob)",
            "metadata": "v1/data/processed/phase5_production_3class/"
                        "production_model_metadata.json",
            "protocol_scripts": [
                "v1/scripts/geographic_holdout_eval.py",
                "v1/src/models/validation.py",
                "v1/scripts/freeze_phase5_production.py"],
            "v1_locked_accuracy": 0.6396665686967289,
            "v1_locked_macro_f1": 0.6314607179023212,
            "season_note": "V1's 63.97% locked number used July-window "
                           "features/met; V2 uses W4 (May 1 - Jun 30). The V2 "
                           "locked number is NOT season-comparable to V1."},
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "seed": 42,
        "locked_blocks": list(HOLDOUT_BLOCKS),
        "n_spatial_blocks": N_SPATIAL_BLOCKS,
        "schema_hash": schema_hash(schema),
        "input_hashes": input_hashes,
        "row_counts": {
            "total": int(len(df)),
            "per_year": {int(k): int(v) for k, v in
                         df.groupby("year").size().items()},
            "per_block": {int(k): int(v) for k, v in
                          df.groupby("spatial_block_id").size().items()},
            "per_year_block_class": (
                {str(int(y)): {int(b): {CLASS_LABELS[c]: int(n) for c, n in
                        enumerate(np.bincount(y_final[(df["year"].to_numpy() == y) &
                                 (df["spatial_block_id"].to_numpy() == b)],
                                 minlength=3))}
                               for b in sorted(df["spatial_block_id"].unique())}
                 for y in sorted(df["year"].unique())}
                if y_final is not None else None),
        },
        "cv": json_safe(cv),
        "lobo": json_safe(lobo),
        "final": json_safe({k: v for k, v in (final or {}).items()
                            if k != "locked"}),
        "locked_metrics": json_safe(final["locked"]) if final else None,
        "thresholds_by_year": ({str(k): {"t1": v[0], "t2": v[1]}
                                for k, v in thr_final.items()}
                               if thr_final else None),
        "fold_mapping": [{"fold": f["fold"], "train_blocks": f["train_blocks"],
                          "val_blocks": f["val_blocks"]}
                         for f in adjacent_block_folds(df)],
        "wall_times_s": wall,
        "total_wall_s": time.perf_counter() - t_start,
        "stages_skipped": {"lobo": bool(args.skip_lobo),
                           "locked": bool(args.skip_locked)},
        "occupied_blocks": sorted(df["spatial_block_id"].unique().tolist()),
    }
    dump_json(manifest, out_dir / "phase5_manifest.json")
    print(f"[DONE] artifacts in {out_dir} "
          f"(total {manifest['total_wall_s']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
