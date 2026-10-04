"""V2 Phase 5 primary-model (Random Forest) freeze verification gate.

Mirrors ``v2.phase5.verify_run`` conventions: reload the persisted production
model and independently re-derive every fact Phase 6+ will rely on, from the
read-only inputs and the frozen rf_compare record. Numbered PASS/FAIL
checklist; exit 0 = all PASS. The result (and timestamp) is written into
``phase5_primary_model.json`` (``verification_passed``).

Checks
------
1.  marker designates RF, is frozen, and points at an existing artifact.
2.  joblib reload -> re-predicted locked rows (features rebuilt from the
    phase4 parquets) are BIT-IDENTICAL to rf_compare/locked_predictions.csv.
3.  recomputed accuracy / macro-F1 equal rf_compare/locked_metrics.json
    EXACTLY; confusion equals rf_compare/locked_confusion.csv; trace/total
    equals accuracy.
4.  per-class precision/recall/F1/support equal rf_compare/per_class_locked.csv.
5.  thresholds exist for all 5 years, t1 < t2, and match train-only tertiles
    recomputed from the phase4 features (non-locked rows) exactly.
6.  frozen 178-schema hash + banned-column absence in the features.
7.  the marker's locked numbers equal the frozen rf_compare record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from v2.common import HOLDOUT_BLOCKS, dump_json, load_frozen_schema, schema_hash
from v2.phase5.train import (
    BANNED_PREDICTOR_COLS,
    CLASS_LABELS,
    METADATA_COLS,
    per_class_metrics,
    tercile_thresholds,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(n: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{n}] [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--data-dir", default=None,
                    help="phase5 dir holding the marker/artifacts "
                         "(default: v2/data/phase5)")
    args = ap.parse_args(argv)
    phase5_dir = Path(args.data_dir) if args.data_dir else \
        Path(__file__).resolve().parents[3] / "data" / "phase5"
    features_dir = Path(args.features_dir)
    rec_dir = phase5_dir / "rf_compare"                    # frozen run record
    marker_path = phase5_dir / "phase5_primary_model.json"

    all_ok = True
    # -- [1] marker ------------------------------------------------------
    marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
    all_ok &= check("1", "marker designates frozen RF primary",
                    bool(marker) and marker.get("primary_model") == "rf"
                    and marker.get("frozen") is True
                    and (phase5_dir / marker.get("artifact", "")).exists(),
                    f"marker={marker_path.name} artifact="
                    f"{marker.get('artifact')}")

    # -- shared loads ------------------------------------------------------
    schema = load_frozen_schema()
    pred = pd.read_csv(rec_dir / "locked_predictions.csv")
    recorded = json.loads((rec_dir / "locked_metrics.json").read_text())
    frames = [pd.read_parquet(features_dir / f"features_{y}.parquet")
              for y in range(2022, 2027)]
    df = pd.concat(frames, ignore_index=True)
    key = ["row", "col", "year"]
    saved_keys = pred[key].copy()
    for k in key:                                       # parquet dtypes
        saved_keys[k] = saved_keys[k].astype(df[k].dtype)
    merged = saved_keys.merge(df, on=key, how="left")
    shape_ok = len(merged) == len(pred) and not merged[schema].isna().any().any()

    # -- [2] bit-identical reload ----------------------------------------
    preds = None
    if not shape_ok:
        all_ok &= check("2", "joblib reload reproduces locked predictions",
                        False, f"key merge changed rows {len(pred)} -> "
                               f"{len(merged)} (or NaN)")
    else:
        model = joblib.load(phase5_dir / marker["artifact"])
        t0 = __import__("time").perf_counter()
        preds = model.predict(merged[schema])
        dt = __import__("time").perf_counter() - t0
        all_ok &= check("2", "joblib reload reproduces locked predictions",
                        bool(np.array_equal(np.asarray(preds, dtype=int),
                                            pred["y_pred"].to_numpy(int))),
                        f"{len(pred)} predictions, exact array equality, "
                        f"predict {dt:.1f}s")

    # -- [3] metrics reproduction ----------------------------------------
    acc = float(accuracy_score(pred["y_true"], pred["y_pred"]))
    f1 = float(f1_score(pred["y_true"], pred["y_pred"], labels=[0, 1, 2],
                        average="macro"))
    cm = confusion_matrix(pred["y_true"], pred["y_pred"], labels=[0, 1, 2])
    cm_rec = pd.read_csv(rec_dir / "locked_confusion.csv", index_col=0).to_numpy()
    all_ok &= check("3a", "locked accuracy/macro-F1 reproduce exactly",
                    acc == recorded["accuracy"] and f1 == recorded["macro_f1"],
                    f"acc={acc!r} f1={f1!r}")
    all_ok &= check("3b", "locked confusion matches frozen record",
                    np.array_equal(cm, cm_rec), f"trace={int(np.trace(cm))}")
    all_ok &= check("3c", "confusion trace accuracy consistent",
                    abs(float(np.trace(cm) / cm.sum()) - acc) <= 1e-12)

    # -- [4] per-class reproduction ---------------------------------------
    pc_rec = pd.read_csv(rec_dir / "per_class_locked.csv")
    pc_now = per_class_metrics(cm.tolist())
    ok_pc = all(
        pc_rec.loc[pc_rec["class"] == name, "precision"].iloc[0]
        == pc_now[name]["precision"]
        and pc_rec.loc[pc_rec["class"] == name, "recall"].iloc[0]
        == pc_now[name]["recall"]
        and pc_rec.loc[pc_rec["class"] == name, "f1"].iloc[0]
        == pc_now[name]["f1"]
        and int(pc_rec.loc[pc_rec["class"] == name, "support"].iloc[0])
        == pc_now[name]["support"]
        for name in CLASS_LABELS)
    all_ok &= check("4", "per-class P/R/F1/support reproduce exactly", ok_pc,
                    f"support_sum={int(pc_rec['support'].sum())}")

    # -- [5] thresholds ----------------------------------------------------
    thr_rec = {int(k): (v["t1"], v["t2"]) for k, v in
               json.loads((rec_dir / "thresholds_by_year.json")
                          .read_text()).items()}
    thr_now = tercile_thresholds(df, ~df["spatial_block_id"].isin(HOLDOUT_BLOCKS))
    all_ok &= check("5", "thresholds monotonic + train-only exact",
                    len(thr_rec) == 5 and all(t1 < t2 for t1, t2 in thr_rec.values())
                    and all(np.array_equal(np.array(thr_rec[y]), np.array(thr_now[y]))
                            for y in thr_rec))

    # -- [6] schema + banned ----------------------------------------------
    banned_hit = False
    for y in range(2022, 2027):
        cols = pd.read_parquet(features_dir / f"features_{y}.parquet",
                               columns=None).columns.tolist()
        predictors = [c for c in cols if c not in METADATA_COLS]
        if predictors != schema or BANNED_PREDICTOR_COLS.intersection(predictors):
            banned_hit = True
    all_ok &= check("6", "frozen schema hash + banned columns absent",
                    schema_hash(schema) == schema_hash(load_frozen_schema())
                    and not banned_hit,
                    f"hash={schema_hash(schema)}")

    # -- [7] marker numbers == frozen record ------------------------------
    all_ok &= check("7", "marker locked numbers == frozen rf_compare record",
                    marker.get("locked_accuracy") == recorded["accuracy"]
                    and marker.get("locked_macro_f1") == recorded["macro_f1"]
                    and marker.get("baseline_xgb") == "phase5_primary_xgb_v2.json")

    # -- verdict into marker ----------------------------------------------
    marker["verification_passed"] = bool(all_ok)
    from datetime import datetime, timezone
    marker["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    marker["verified_by"] = "v2.phase5.verify_rf_primary"
    dump_json(marker, marker_path)

    fails = [r for r in RESULTS if not r[1]]
    print(f"\n[SUMMARY] {len(RESULTS) - len(fails)}/{len(RESULTS)} checks pass, "
          f"{len(fails)} FAIL", flush=True)
    if fails:
        for name, _, detail in fails:
            print(f"  FAILED: {name} {detail}")
        print("[VERIFY] FREEZE GATE FAILED", flush=True)
        return 1
    print("[VERIFY] ALL CHECKS PASS - RF primary frozen and verified",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
