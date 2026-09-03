"""Driver for the Phase 5 generalization / overfitting audit.

Executes the audit diagnostics in the approved priority order
(see the Phase 5 audit request, execution-order section):

  1.  Baseline experiment manifest (machine-readable benchmark snapshot).
  2.  Development diagnostics: train/validation gap, RF + XGBoost
      regularization, seed stability, prediction stability, target
      permutation, learning curves, leave-one-block-out, spatial-distance
      stress test, feature-permutation sanity, duplicate audit, leakage
      audit, nested spatial CV, compact-feature experiment, calibration,
      residual spatial autocorrelation, year investigation.
  3.  Locked geographic holdout — only with ``--final``, after all
      development results have been reviewed and the final configuration is
      frozen.  The holdout is evaluated exactly once and is never used for
      tuning or selection.

Usage:

    PYTHONPATH=src python3 scripts/run_phase5_audit.py             # development
    PYTHONPATH=src python3 scripts/run_phase5_audit.py --final     # holdout, once
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.generalization_audit import (  # noqa: E402
    _load_full_dataset,
    build_baseline_manifest,
    locked_geographic_holdout,
    run_audit,
    write_generalization_summary,
    OUTPUT_DIR,
)

# Development steps in approved priority order.  The locked geographic holdout
# is deliberately NOT in this list.
DEV_STEPS = [
    "train_validation_gap",
    "regularization",
    "seed_stability",
    "prediction_stability",
    "target_permutation",
    "learning_curves",
    "leave_one_block_out",
    "spatial_distance",
    "feature_permutation_sanity",
    "duplicate_audit",
    "leakage_audit",
    "nested_spatial_cv",
    "compact_feature",
    "calibration",
    "spatial_error_autocorrelation",
    "year_ablation",
]


def main() -> int:
    final = "--final" in sys.argv[1:]

    t0 = time.time()
    df, predictor_cols = _load_full_dataset()
    print(f"Dataset loaded: {df.shape}, {len(predictor_cols)} predictors "
          f"({time.time() - t0:.1f}s)", flush=True)

    if final:
        print("Running LOCKED GEOGRAPHIC HOLDOUT evaluation (once).", flush=True)
        print("Frozen development-time selection: RF-C "
              "(max_depth=15, min_samples_leaf=10; best spatial-CV macro-F1 "
              "0.4310, gap 28.9 pp). XGBoost evaluated at baseline params.",
              flush=True)
        rf_overrides = {"max_depth": 15, "min_samples_leaf": 10}
        result = locked_geographic_holdout(df, predictor_cols, rf_overrides=rf_overrides)
        print(json.dumps(result, indent=2), flush=True)
        return 0

    print("Building baseline experiment manifest ...", flush=True)
    manifest = build_baseline_manifest(df, predictor_cols)
    print(f"Manifest written: {manifest.get('output_path', 'phase5_baseline_manifest.json')}",
          flush=True)

    print(f"Running development audit steps: {DEV_STEPS}", flush=True)
    results = run_audit(df, predictor_cols, steps=DEV_STEPS)
    summary_path = Path(OUTPUT_DIR) / "phase5_generalization_summary.json"
    print(f"Development audit complete ({time.time() - t0:.1f}s). "
          f"Summary: {summary_path}", flush=True)
    print(json.dumps({k: str(v)[:200] for k, v in results.items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
