"""V2 Phase 5 — pre-specified Random Forest experiment (comparison run + primary refit).

USER DECISION (Phase 5 FROZEN): the Random Forest is the OFFICIAL V2 Phase 5
primary production model; the XGB run (locked acc 0.6877034435497688 /
macro-F1 0.6724671997129691) is the preserved historical baseline. The
comparison run record in ``v2/data/phase5/rf_compare/`` (CV/LOBO/locked
metrics, evaluated exactly once) is historical evidence and is NOT modified;
this module reuses the frozen protocol machinery from ``v2.phase5.train``
(data loading + schema assert, per-year train-only tertile targets, the
non-locked occupied-block fold mapping, scoring helpers) so everything is
protocol-identical:

  - same 5 phase4 parquets, same 178-feature frozen schema assert
    (sha256 ddd43d04...c211), same banned-predictor assert
  - per-year LST tertiles from TRAINING rows only inside every split;
    ``searchsorted(side="right")`` boundary; classes Low/Moderate/High
  - 5-fold adjacent-block CV over the same non-locked occupied blocks,
    same GroupKFold mapping as the XGB run
  - LOBO over all 25 pinned blocks (24 train / 1 test)
  - locked evaluation on blocks {2, 9, 15, 23} EXACTLY ONCE, same banner

``--refit-only`` (no CV/LOBO/locked eval) reproduces the deterministic final
refit of the comparison run and persists the production model
(``phase5_primary_rf_v2.joblib`` + production metadata + the
``phase5_primary_model.json`` marker Phase 6+ reads).

RF hyperparameters are pre-specified verbatim from v1/src/models/config.py
::RANDOM_FOREST_PARAMS (RANDOM_SEED = 42 verified in v1/src/features/
config.py) — NO tuning:

    n_estimators=200, max_depth=None, min_samples_split=5,
    min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
    random_state=42, n_jobs=-1

Determinism note: scikit-learn RandomForest derives every tree's seed
deterministically from the estimator's random_state, so with random_state
fixed the fit is bit-reproducible regardless of n_jobs — unlike XGBoost's
threaded histogram construction. The evaluate-locked-exactly-once protocol
was kept for the comparison record anyway.
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
from sklearn.ensemble import RandomForestClassifier

from v2.common import (
    HOLDOUT_BLOCKS,
    PROJECT_ROOT,
    dump_json,
    json_safe,
    load_frozen_schema,
    schema_hash,
    sha256_file,
)
from v2.phase5.train import (
    CLASS_LABELS,
    LOCKED_BANNER,
    N_CV_FOLDS,
    N_SPATIAL_BLOCKS,
    adjacent_block_folds,
    build_target,
    class_balance_per_year,
    load_features,
    per_class_metrics,
    score_3class,
    tercile_thresholds,
)

# Verbatim V1 RANDOM_FOREST_PARAMS (v1/src/models/config.py; RANDOM_SEED=42).
RF_PARAMS_V1 = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS_V1)


# ---------------------------------------------------------------------------
# Stages (mirror v2.phase5.train, RF model swapped in)
# ---------------------------------------------------------------------------

def run_cv(df: pd.DataFrame, X: pd.DataFrame, out_dir: Path) -> dict:
    """5-fold adjacent-block CV with the XGB run's fold mapping + OOF."""
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
        model = make_rf()
        t1 = time.perf_counter()
        model.fit(X.iloc[tr], y[tr])
        fit_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        preds = model.predict(X.iloc[va])
        pred_s = time.perf_counter() - t2
        s = score_3class(y[va], preds)
        oof_pred[va] = preds
        oof_true[va] = y[va]
        oof_fold[va] = f["fold"]
        rows.append({"fold": f["fold"], "n_train": int(len(tr)),
                     "n_val": int(len(va)),
                     "train_blocks": f["train_blocks"], "val_blocks": f["val_blocks"],
                     "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                     "fit_wall_s": fit_s, "predict_wall_s": pred_s})
        for i in range(3):
            for j in range(3):
                cms.append({"fold": f["fold"], "true_class": CLASS_LABELS[i],
                            "pred_class": CLASS_LABELS[j],
                            "count": int(s["confusion_matrix"][i][j])})
        print(f"[CV ] fold {f['fold']}: acc={s['accuracy']:.6f} "
              f"macro_f1={s['macro_f1']:.6f} n_val={len(va)} "
              f"(fit {fit_s:.1f}s, predict {pred_s:.1f}s)", flush=True)

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
    """LOBO over all 25 pinned blocks; empty blocks recorded without fits."""
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
        model = make_rf()
        t1 = time.perf_counter()
        model.fit(X[train_mask], y[train_mask])
        fit_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        preds = model.predict(X[test_mask])
        pred_s = time.perf_counter() - t2
        s = score_3class(y[test_mask], preds)
        rows.append({"block": bid, "n_train": int(train_mask.sum()),
                     "n_test": n_test, "accuracy": s["accuracy"],
                     "macro_f1": s["macro_f1"], "empty": False,
                     "fit_wall_s": fit_s, "predict_wall_s": pred_s})
        print(f"[LOBO] block {bid:2d}: acc={s['accuracy']:.6f} "
              f"macro_f1={s['macro_f1']:.6f} n_test={n_test} "
              f"(fit {fit_s:.1f}s, predict {pred_s:.1f}s)", flush=True)
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
    """Refit on all non-locked rows; locked evaluation EXACTLY ONCE (banner)."""
    t0 = time.perf_counter()
    groups = df["spatial_block_id"].astype(int).to_numpy()
    locked_mask = np.isin(groups, HOLDOUT_BLOCKS)
    train_mask = ~locked_mask
    y, thresholds = build_target(df, train_mask)
    balance = class_balance_per_year(y[train_mask],
                                     df["year"].to_numpy()[train_mask])
    for year, entry in balance.items():
        fr = entry["fractions"]
        if not all(0.25 <= fr[k] <= 0.42 for k in CLASS_LABELS):
            raise AssertionError(f"class balance off for {year}: {fr}")

    print(f"[FINAL] RF refit on {int(train_mask.sum())} non-locked rows ...",
          flush=True)
    model = make_rf()
    model.fit(X[train_mask], y[train_mask])
    fit_s = time.perf_counter() - t0
    print(f"[FINAL] refit done in {fit_s:.1f}s", flush=True)

    n_locked = int(locked_mask.sum())
    locked = None
    if n_locked == 0:
        print("[LOCKED] WARNING: no locked-block rows (synthetic smoke).")
    else:
        print(LOCKED_BANNER.format(locked=list(HOLDOUT_BLOCKS)), flush=True)
        t1 = time.perf_counter()
        preds = model.predict(X[locked_mask])
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

    dump_json({str(k): {"t1": v[0], "t2": v[1]} for k, v in thresholds.items()},
              out_dir / "thresholds_by_year.json")
    for year, (t1, t2) in thresholds.items():
        assert t1 < t2, f"thresholds not monotonic for {year}: {t1} !< {t2}"
    return {"fit_wall_s": fit_s, "locked": locked, "thresholds": thresholds,
            "train_class_balance": balance}


# ---------------------------------------------------------------------------
# --refit-only: persist the deterministic final refit as the production model
# ---------------------------------------------------------------------------

def refit_only(df: pd.DataFrame, X: pd.DataFrame, out_dir: Path) -> dict:
    """Refit the final RF (identical code path as the comparison run's stage e,
    minus the locked evaluation) and persist the production artifacts.

    The RF is bit-reproducible at fixed random_state, so this model equals the
    one whose locked metrics are recorded in rf_compare/locked_metrics.json;
    ``verify_rf_primary`` asserts that reproduction from the saved joblib.
    """
    import joblib

    groups = df["spatial_block_id"].astype(int).to_numpy()
    locked_mask = np.isin(groups, HOLDOUT_BLOCKS)
    train_mask = ~locked_mask
    y, thresholds = build_target(df, train_mask)
    for year, entry in class_balance_per_year(y[train_mask],
                                              df["year"].to_numpy()[train_mask]).items():
        fr = entry["fractions"]
        if not all(0.25 <= fr[k] <= 0.42 for k in CLASS_LABELS):
            raise AssertionError(f"class balance off for {year}: {fr}")

    print(f"[REFIT] production RF refit on {int(train_mask.sum())} non-locked "
          f"rows (deterministic, random_state=42) ...", flush=True)
    t0 = time.perf_counter()
    model = make_rf()
    model.fit(X[train_mask], y[train_mask])
    fit_s = time.perf_counter() - t0
    print(f"[REFIT] done in {fit_s:.1f}s", flush=True)

    # cross-check against the frozen comparison record (no new evaluation:
    # these are the recorded metrics of the identical deterministic model)
    rec_dir = out_dir / "rf_compare"
    rec_path = rec_dir / "locked_metrics.json"
    recorded = json.loads(rec_path.read_text())
    rec_thr = {int(k): (v["t1"], v["t2"]) for k, v in
               json.loads((rec_dir / "thresholds_by_year.json").read_text()).items()}
    assert thresholds == rec_thr, (
        f"refit thresholds diverge from the comparison record: "
        f"{thresholds} vs {rec_thr}")
    assert int(train_mask.sum()) == recorded["n_train"], (
        f"train row count changed: {int(train_mask.sum())} vs "
        f"{recorded['n_train']}")

    joblib.dump(model, out_dir / "phase5_primary_rf_v2.joblib")
    metadata = {
        "model_id": "phase5_primary_rf_v2",
        "model": "random_forest",
        "status": "PRODUCTION - official V2 Phase 5 primary model "
                  "(USER DECISION; Phase 5 FROZEN)",
        "promoted_from": "v2/data/phase5/rf_compare/ comparison run "
                         "(locked metrics evaluated exactly once there)",
        "historical_baseline": "phase5_primary_xgb_v2.json (XGB, locked "
                               "acc 0.6877034435497688 / macro-F1 "
                               "0.6724671997129691)",
        "rf_params": {**RF_PARAMS_V1, "max_depth": "None (unlimited)"},
        "rf_determinism_note": "tree seeds derive deterministically from "
                               "random_state=42 -> bit-reproducible fit "
                               "regardless of n_jobs; this refit reproduces "
                               "the comparison run's final model",
        "locked_metrics": {
            "accuracy": recorded["accuracy"],
            "macro_f1": recorded["macro_f1"],
            "locked_blocks": recorded["locked_blocks"],
            "n_train": recorded["n_train"],
            "n_locked": recorded["n_locked"],
            "evaluated_exactly_once": True,
            "source": "rf_compare/locked_metrics.json (frozen record)",
        },
        "thresholds_by_year": {str(k): {"t1": v[0], "t2": v[1]}
                               for k, v in thresholds.items()},
        "schema_hash": schema_hash(load_frozen_schema()),
        "n_train_rows": int(train_mask.sum()),
        "refit_wall_s": fit_s,
        "frozen_at_utc": _utc_now(),
        "verify_gate": "v2.phase5.verify_rf_primary (must set "
                       "verification_passed=true in phase5_primary_model.json)",
    }
    dump_json(metadata, out_dir / "phase5_primary_rf_v2.json")
    marker = {
        "primary_model": "rf",
        "frozen": True,
        "frozen_at_utc": metadata["frozen_at_utc"],
        "locked_accuracy": recorded["accuracy"],
        "locked_macro_f1": recorded["macro_f1"],
        "artifact": "phase5_primary_rf_v2.joblib",
        "baseline_xgb": "phase5_primary_xgb_v2.json",
        "verification_passed": False,
        "verification_note": "set by v2.phase5.verify_rf_primary",
    }
    dump_json(marker, out_dir / "phase5_primary_model.json")
    print(f"[REFIT] wrote phase5_primary_rf_v2.joblib/.json + "
          f"phase5_primary_model.json to {out_dir}", flush=True)
    return {"fit_wall_s": fit_s}


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Smoke probe: exactly 1 CV fold + 1 LOBO fit, no locked eval, no artifacts
# ---------------------------------------------------------------------------

def smoke_probe(df: pd.DataFrame, X: pd.DataFrame) -> None:
    folds = adjacent_block_folds(df)
    f = folds[0]
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[f["train_idx"]] = True
    y, thr = build_target(df, train_mask)
    t1 = time.perf_counter()
    model = make_rf()
    model.fit(X.iloc[f["train_idx"]], y[f["train_idx"]])
    fit_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    preds = model.predict(X.iloc[f["val_idx"]])
    pred_s = time.perf_counter() - t2
    s = score_3class(y[f["val_idx"]], preds)
    print(f"[SMOKE] CV fold 1/{len(folds)}: n_train={len(f['train_idx'])} "
          f"n_val={len(f['val_idx'])} acc={s['accuracy']:.6f} "
          f"fit={fit_s:.1f}s predict={pred_s:.1f}s", flush=True)

    groups = df["spatial_block_id"].astype(int).to_numpy()
    bid = next(b for b in range(N_SPATIAL_BLOCKS) if (groups == b).any())
    test_mask = groups == bid
    train_mask = ~test_mask
    y, _ = build_target(df, train_mask)
    t1 = time.perf_counter()
    model = make_rf()
    model.fit(X[train_mask], y[train_mask])
    fit_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    preds = model.predict(X[test_mask])
    pred_s = time.perf_counter() - t2
    s = score_3class(y[test_mask], preds)
    print(f"[SMOKE] LOBO block {bid}: n_train={int(train_mask.sum())} "
          f"n_test={int(test_mask.sum())} acc={s['accuracy']:.6f} "
          f"fit={fit_s:.1f}s predict={pred_s:.1f}s", flush=True)
    print("[SMOKE] probe complete (no artifacts written, locked eval skipped)",
          flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema-json", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="probe only: 1 CV fold + 1 LOBO fit, no artifacts")
    ap.add_argument("--refit-only", action="store_true",
                    help="production refit: deterministic final RF -> joblib + "
                         "metadata + primary marker (no CV/LOBO/locked eval)")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = (load_frozen_schema(Path(args.schema_json)) if args.schema_json
              else load_frozen_schema())
    assert schema_hash(schema) == schema_hash(load_frozen_schema()), \
        "schema hash mismatch vs the frozen reference"

    print("[LOAD] reading 5 feature parquets + schema gates ...", flush=True)
    t0 = time.perf_counter()
    df = load_features(Path(args.features_dir), schema)
    load_s = time.perf_counter() - t0
    X = df[list(schema)]
    print(f"[LOAD] {len(df)} rows x {len(schema)} predictors "
          f"({load_s:.1f}s)", flush=True)

    if args.smoke:
        smoke_probe(df, X)
        return 0

    if args.refit_only:
        refit_only(df, X, out_dir)
        return 0

    wall = {"load_s": load_s}
    cv = run_cv(df, X, out_dir)
    wall["cv_s"] = cv["wall_s"]
    print(f"[CV ] mean acc={cv['accuracy_mean']:.6f}+-{cv['accuracy_std']:.6f} "
          f"mean macro_f1={cv['macro_f1_mean']:.6f}+-{cv['macro_f1_std']:.6f} "
          f"({cv['wall_s']:.1f}s)", flush=True)

    lobo = run_lobo(df, X, out_dir)
    wall["lobo_s"] = lobo["wall_s"]
    print(f"[LOBO] mean acc={lobo['accuracy_mean']:.6f}"
          f"+-{lobo['accuracy_std']:.6f} "
          f"mean macro_f1={lobo['macro_f1_mean']:.6f}"
          f"+-{lobo['macro_f1_std']:.6f} "
          f"({lobo['wall_s']:.1f}s, {lobo['n_blocks_empty']} empty blocks)",
          flush=True)

    final = run_final_fit(df, X, out_dir)
    wall["final_fit_s"] = final["fit_wall_s"]
    if final["locked"]:
        wall["locked_eval_s"] = final["locked"]["eval_wall_s"]

    import pyarrow
    import sklearn
    feat_dir = Path(args.features_dir)
    input_hashes = {p.name: sha256_file(p) for p in
                    sorted(feat_dir.glob("features_*.parquet"))}
    schema_path = Path(args.schema_json) if args.schema_json else \
        PROJECT_ROOT / "reference" / "phase5_primary_xgb_3class_features.json"
    input_hashes[schema_path.name] = sha256_file(schema_path)
    blocks_manifest = PROJECT_ROOT / "data" / "phase2" / "spatial_blocks_manifest.json"
    input_hashes[blocks_manifest.name] = sha256_file(blocks_manifest)

    y_final, thr_final = build_target(df, ~df["spatial_block_id"].isin(HOLDOUT_BLOCKS))
    manifest = {
        "model_id": "phase5_rf_compare_v2",
        "status": "SECONDARY COMPARISON MODEL - the frozen XGB primary "
                  "(phase5_primary_xgb_v2) is unchanged and remains the "
                  "official Phase 5 model",
        "protocol": "identical to the frozen XGB run (v2.phase5.train "
                    "machinery); RF params pre-specified verbatim from V1, "
                    "no tuning",
        "rf_params": {**RF_PARAMS_V1, "max_depth": "None (unlimited)"},
        "rf_unspecified_defaults": {
            "note": "all other RandomForestClassifier args are sklearn "
                    "defaults (bootstrap=True, criterion=gini, ...)"},
        "rf_determinism_note": "sklearn RandomForest derives every tree's "
                               "seed deterministically from random_state, so "
                               "with random_state=42 the fit is bit-"
                               "reproducible regardless of n_jobs (unlike "
                               "XGBoost threaded histograms). Locked "
                               "evaluation still performed exactly once per "
                               "protocol.",
        "frozen_xgb_primary": {
            "artifacts": "../ (v2/data/phase5/)",
            "locked_accuracy": 0.6877034435497688,
            "locked_macro_f1": 0.6724671997129691},
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "seed": 42,
        "locked_blocks": list(HOLDOUT_BLOCKS),
        "n_spatial_blocks": N_SPATIAL_BLOCKS,
        "n_cv_folds": N_CV_FOLDS,
        "schema_hash": schema_hash(schema),
        "input_hashes": input_hashes,
        "row_counts": {
            "total": int(len(df)),
            "per_year": {int(k): int(v) for k, v in
                         df.groupby("year").size().items()},
            "per_block": {int(k): int(v) for k, v in
                          df.groupby("spatial_block_id").size().items()},
        },
        "cv": json_safe(cv),
        "lobo": json_safe(lobo),
        "final": json_safe({k: v for k, v in final.items() if k != "locked"}),
        "locked_metrics": json_safe(final["locked"]),
        "thresholds_by_year": {str(k): {"t1": v[0], "t2": v[1]}
                               for k, v in thr_final.items()},
        "fold_mapping": [{"fold": f["fold"], "train_blocks": f["train_blocks"],
                          "val_blocks": f["val_blocks"]}
                         for f in adjacent_block_folds(df)],
        "wall_times_s": wall,
        "total_wall_s": time.perf_counter() - t_start,
        "occupied_blocks": sorted(df["spatial_block_id"].unique().tolist()),
    }
    dump_json(manifest, out_dir / "phase5_manifest.json")
    print(f"[DONE] artifacts in {out_dir} "
          f"(total {manifest['total_wall_s']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
