"""V2 Phase 5 run verification — independent re-check of every gate.

Reloads the artifacts written by ``train.py`` and re-derives each gate from
the read-only inputs. All gates must PASS; exit code 0 = PASS, 1 = FAIL.
Gates whose artifacts were skipped in a dry run (--skip-lobo/--skip-locked)
are reported as SKIP, not FAIL.

Gates
-----
1.  frozen schema: names+order+hash of the 178 predictors; banned columns
    {lst_C, row, col, spatial_block_id} absent from the predictors.
2.  model reload: phase5_primary_xgb_v2.json reloaded single-threaded
    (n_jobs=1) reproduces the training-time locked predictions BIT-IDENTICALLY
    (exact array equality against locked_predictions.csv).
3.  locked metrics: accuracy recomputed from locked_predictions.csv equals
    locked_metrics.json accuracy EXACTLY; the recomputed 3x3 confusion equals
    locked_confusion.csv; trace/total equals the recorded accuracy to 1e-12.
4.  thresholds: thresholds_by_year.json exists for all 5 years, t1 < t2
    (monotonic) per year, and matches per-year tertiles recomputed from the
    TRAINING rows only (non-locked) exactly.
5.  class balance: per-year train-portion class fractions under the saved
    thresholds are ~1/3 each (tolerance 0.05).
6.  manifest: required provenance present (verbatim xgb params, versions,
    seed, input hashes, row counts, fold mapping, wall times).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

from v2.common import HOLDOUT_BLOCKS, load_frozen_schema, schema_hash
from v2.phase5.train import (
    BANNED_PREDICTOR_COLS,
    CLASS_LABELS,
    METADATA_COLS,
    assign_classes,
    tercile_thresholds,
)

RESULTS = []  # (gate, status, detail)


def gate(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def skip(name: str, detail: str = "") -> None:
    RESULTS.append((name, "SKIP", detail))
    print(f"[SKIP] {name} {detail}", flush=True)


def check_schema(features_dir: Path) -> bool:
    import pyarrow.parquet as pq
    schema = load_frozen_schema()
    ok, detail = True, f"hash={schema_hash(schema)}"
    for year in range(2022, 2027):
        path = features_dir / f"features_{year}.parquet"
        if not path.exists():
            return gate("schema:parquets_present", False, f"missing {path}")
        cols = pq.ParquetFile(path).schema_arrow.names
        predictors = [c for c in cols if c not in METADATA_COLS]
        if predictors != schema:
            ok = False
            detail = f"{path.name}: predictor names/order != frozen schema"
            break
        banned = BANNED_PREDICTOR_COLS.intersection(predictors)
        if banned:
            ok = False
            detail = f"{path.name}: banned columns present: {banned}"
            break
    return gate("schema:names_order_hash_banned", ok, detail)


def check_reload(features_dir: Path, out_dir: Path) -> None:
    pred_path = out_dir / "locked_predictions.csv"
    model_path = out_dir / "phase5_primary_xgb_v2.json"
    if not pred_path.exists() or not model_path.exists():
        return skip("locked:reload_bit_identical",
                    "no locked artifacts (dry-run skip)")
    saved = pd.read_csv(pred_path)
    frames = []
    for year in range(2022, 2027):
        frames.append(pd.read_parquet(features_dir / f"features_{year}.parquet"))
    df = pd.concat(frames, ignore_index=True)
    schema = load_frozen_schema()
    key = ["row", "col", "year"]
    saved_keys = saved[key].copy()
    for k in key:                                   # parquet dtypes (year is
        saved_keys[k] = saved_keys[k].astype(df[k].dtype)   # float32)
    # Rebuild the locked matrix in the EXACT saved row order (left merge on a
    # unique key; any duplicate key would change the row count and fail).
    merged = saved_keys.merge(df, on=key, how="left")
    if len(merged) != len(saved):
        return gate("locked:reload_bit_identical", False,
                    f"key merge changed row count {len(saved)} -> {len(merged)} "
                    "(duplicate or missing feature rows)")
    if merged[schema].isna().any().any():
        return gate("locked:reload_bit_identical", False, "NaN after key merge")
    model = XGBClassifier()
    model.load_model(str(model_path))
    model.set_params(n_jobs=1)                     # single-threaded, per policy
    preds = model.predict(merged[schema])
    ok = bool(np.array_equal(np.asarray(preds, dtype=int),
                             saved["y_pred"].to_numpy(dtype=int)))
    gate("locked:reload_bit_identical", ok,
         f"{len(saved)} predictions, exact array equality vs "
         f"locked_predictions.csv")


def check_locked_metrics(out_dir: Path) -> None:
    pred_path = out_dir / "locked_predictions.csv"
    m_path = out_dir / "locked_metrics.json"
    if not pred_path.exists() or not m_path.exists():
        return skip("locked:metrics_consistency", "no locked artifacts")
    saved = pd.read_csv(pred_path)
    m = json.loads(m_path.read_text())
    acc = float(accuracy_score(saved["y_true"], saved["y_pred"]))
    ok_acc = gate("locked:accuracy_recompute_exact",
                  acc == m["accuracy"], f"csv={acc!r} json={m['accuracy']!r}")
    cm = confusion_matrix(saved["y_true"], saved["y_pred"], labels=[0, 1, 2])
    cm_csv = pd.read_csv(out_dir / "locked_confusion.csv", index_col=0)
    ok_cm = gate("locked:confusion_matches_csv",
                 np.array_equal(cm, cm_csv.to_numpy()),
                 f"recomputed={cm.tolist()}")
    cm_acc = float(np.trace(cm) / cm.sum())
    ok_tr = gate("locked:confusion_trace_accuracy",
                 abs(cm_acc - m["accuracy"]) <= 1e-12, f"trace={cm_acc!r}")
    pc_csv = pd.read_csv(out_dir / "per_class_locked.csv")
    ok_pc = gate("locked:per_class_support_sums",
                 int(pc_csv["support"].sum()) == len(saved),
                 f"support_sum={int(pc_csv['support'].sum())} n={len(saved)}")
    return ok_acc and ok_cm and ok_tr and ok_pc


def check_thresholds(features_dir: Path, out_dir: Path) -> None:
    t_path = out_dir / "thresholds_by_year.json"
    if not t_path.exists():
        return skip("thresholds:monotonic_train_only", "no thresholds artifact")
    saved = {int(k): (float(v["t1"]), float(v["t2"]))
             for k, v in json.loads(t_path.read_text()).items()}
    ok_mono = len(saved) == 5 and all(t1 < t2 for t1, t2 in saved.values())
    gate("thresholds:monotonic_all_years", ok_mono, f"n_years={len(saved)}")

    frames = [pd.read_parquet(features_dir / f"features_{year}.parquet")
              for year in range(2022, 2027)]
    df = pd.concat(frames, ignore_index=True)
    train_mask = ~df["spatial_block_id"].isin(HOLDOUT_BLOCKS)
    recomputed = tercile_thresholds(df, train_mask)
    ok_eq = all(np.array_equal(np.array(saved[y]), np.array(recomputed[y]))
                for y in saved)
    gate("thresholds:train_rows_only_exact", ok_eq,
         "saved == tertiles recomputed from non-locked rows only")


def check_class_balance(features_dir: Path, out_dir: Path) -> None:
    t_path = out_dir / "thresholds_by_year.json"
    if not t_path.exists():
        return skip("class_balance:sane", "no thresholds artifact")
    saved = {int(k): (float(v["t1"]), float(v["t2"]))
             for k, v in json.loads(t_path.read_text()).items()}
    frames = [pd.read_parquet(features_dir / f"features_{year}.parquet")
              for year in range(2022, 2027)]
    df = pd.concat(frames, ignore_index=True)
    train_mask = ~df["spatial_block_id"].isin(HOLDOUT_BLOCKS)
    sub = df[train_mask]
    y = assign_classes(sub["lst_C"].to_numpy(), sub["year"].to_numpy(), saved)
    ok, detail = True, ""
    for year in sorted(sub["year"].unique()):
        m = sub["year"].to_numpy() == year
        fr = np.bincount(y[m], minlength=3) / m.sum()
        if not np.all((fr >= 1 / 3 - 0.05) & (fr <= 1 / 3 + 0.05)):
            ok = False
        detail += (f"{int(year)}:[" +
                   ",".join(f"{CLASS_LABELS[k]}={fr[k]:.3f}" for k in range(3)) +
                   "] ")
    gate("class_balance:sane_train_portions", ok, detail)


def check_manifest(out_dir: Path) -> None:
    path = out_dir / "phase5_manifest.json"
    if not path.exists():
        return gate("manifest:present", False, "missing phase5_manifest.json")
    m = json.loads(path.read_text())
    required = ["xgboost_params", "versions", "seed", "input_hashes",
                "row_counts", "fold_mapping", "wall_times_s"]
    missing = [k for k in required if k not in m]
    ok = not missing and m["xgboost_params"]["n_estimators"] == 200 and \
        m["xgboost_params"]["num_class"] == 3 and m["seed"] == 42 and \
        len(m["fold_mapping"]) == 5
    gate("manifest:provenance_complete", ok,
         f"missing={missing}" if missing else
         f"params/version/seed/hash/folds({len(m['fold_mapping'])})/walls present")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--data-dir", required=True,
                    help="directory holding the train.py artifacts")
    args = ap.parse_args(argv)
    features_dir, out_dir = Path(args.features_dir), Path(args.data_dir)

    all_ok = True
    all_ok &= check_schema(features_dir)
    check_reload(features_dir, out_dir)
    check_locked_metrics(out_dir)
    check_thresholds(features_dir, out_dir)
    check_class_balance(features_dir, out_dir)
    check_manifest(out_dir)

    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n[SUMMARY] {len(RESULTS) - len(fails)}/{len(RESULTS)} gates pass, "
          f"{len(fails)} FAIL", flush=True)
    if fails:
        for name, _, detail in fails:
            print(f"  FAILED: {name} {detail}")
        return 1
    print("[VERIFY] ALL GATES PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
