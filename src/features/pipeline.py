"""End-to-end Phase 4 feature-extraction pipeline.

Orchestrates:

1. Vegetation Cover derivation from Sentinel-2 NDVI.
2. Combined urban environmental dataset assembly.
3. Descriptive statistics and correlations.
4. Report-ready map generation.
5. Pipeline record JSON for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from features.config import (
    COMBINED_DATASET_CSV,
    FEATURE_METADATA_JSON,
    PHASE4_DIR,
    PIPELINE_RECORD_JSON,
    PROJECT_ROOT,
)
from features.features import create_combined_dataset_and_stats
from features.io import write_json
from features.maps import generate_maps
from features.vegetation_cover import derive_vegetation_cover_all_years


def run_phase4(maps: bool = True) -> Dict:
    """Run the complete Phase 4 pipeline.

    Parameters
    ----------
    maps : bool, default True
        If True, generate report-ready PNG maps (can be slow for large grids).

    Returns
    -------
    dict
        Pipeline record containing timings, output paths, and status.
    """
    PHASE4_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "phase": 4,
        "project": "GreenGrid-AI",
        "project_root": str(PROJECT_ROOT),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    # 1. Vegetation Cover
    t0 = time.time()
    vc_stats_2022, vc_stats_2026 = derive_vegetation_cover_all_years()
    record["steps"]["vegetation_cover"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "outputs": [
            vc_stats_2022["output_raster"],
            vc_stats_2026["output_raster"],
        ],
        "2022": vc_stats_2022,
        "2026": vc_stats_2026,
    }

    # 2. Combined dataset + statistics
    t0 = time.time()
    df, stats, correlations = create_combined_dataset_and_stats()
    record["steps"]["combined_dataset"] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 3),
        "output_csv": str(COMBINED_DATASET_CSV),
        "n_rows": int(len(df)),
        "n_years": int(df["year"].nunique()),
        "years": sorted(df["year"].unique().tolist()),
    }

    # 3. Maps
    if maps:
        t0 = time.time()
        map_results = generate_maps()
        record["steps"]["maps"] = {
            "status": "success",
            "elapsed_s": round(time.time() - t0, 3),
            "n_maps": len(map_results),
            "outputs": [r["output_path"] for r in map_results],
            "summaries": [
                {
                    "variable": r["variable"],
                    "year": r["year"],
                    "min": r["min"],
                    "max": r["max"],
                    "mean": r["mean"],
                }
                for r in map_results
            ],
        }
    else:
        record["steps"]["maps"] = {"status": "skipped"}

    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["total_elapsed_s"] = round(
        sum(s.get("elapsed_s", 0) for s in record["steps"].values() if isinstance(s, dict)),
        3,
    )
    record["status"] = "success"

    write_json(record, PIPELINE_RECORD_JSON)
    return record


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for the Phase 4 pipeline."""
    parser = argparse.ArgumentParser(
        description="Run GreenGrid-AI Phase 4 feature extraction."
    )
    parser.add_argument(
        "--skip-maps",
        action="store_true",
        help="Skip PNG map generation (faster; useful for testing).",
    )
    args = parser.parse_args(argv)

    try:
        record = run_phase4(maps=not args.skip_maps)
        print(json.dumps(record, indent=2, default=str))
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"Phase 4 pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
