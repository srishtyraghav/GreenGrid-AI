"""
GreenGrid AI — Phase 3: Data Preprocessing
End-to-end Phase 3 pipeline.

This script orchestrates the conversion of raw Phase 2 data into a clean,
ML-ready dataset on a common 30 m grid.

Usage:
    python3 src/preprocessing/pipeline.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from . import config
from .align import align_all
from .cloud_mask import create_combined_valid_mask, inspect_source_nan_patterns
from .feature_table import build_feature_table
from .indices import process_landsat9, process_sentinel2
from .io import ensure_dir
from .vector_raster import process_all_vector_layers


def run_phase3() -> Dict:
    """Execute the full Phase 3 preprocessing pipeline."""
    start_time = time.time()
    print("=" * 70)
    print("  GreenGrid AI — Phase 3 | Data Preprocessing Pipeline")
    print("=" * 70)
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Reference grid: {config.REFERENCE_RASTER.name}")
    print(f"  Target CRS: {config.TARGET_CRS}")
    print(f"  Target resolution: ~{config.TARGET_RESOLUTION_M_APPROX} m nominal (EPSG:4326, angular)")
    print()

    ensure_dir(config.PROCESSED_DIR)
    ensure_dir(config.INDICES_DIR)
    ensure_dir(config.ALIGNED_DIR)
    ensure_dir(config.MASKS_DIR)
    ensure_dir(config.FEATURES_DIR)

    record = {
        "started": datetime.now().isoformat(),
        "reference_raster": str(config.REFERENCE_RASTER),
        "target_crs": config.TARGET_CRS,
        "target_resolution_m_approx": config.TARGET_RESOLUTION_M_APPROX,
    }

    # 1. Inspect existing NaN / masked-pixel patterns
    print("[PIPELINE] Step 1/6: Inspecting source NaN patterns")
    record["source_nan_patterns"] = inspect_source_nan_patterns()

    # 2. Calculate indices
    print("\n[PIPELINE] Step 2/6: Calculating spectral indices")
    record["landsat9_indices"] = process_landsat9(config.INDICES_DIR)
    record["sentinel2_indices"] = process_sentinel2(config.INDICES_DIR)

    # 3. Align / aggregate to common 30 m grid
    print("\n[PIPELINE] Step 3/6: Aligning rasters to 30 m grid")
    record["aligned"] = align_all(config.INDICES_DIR, config.ALIGNED_DIR)

    # 4. Build valid-pixel mask
    print("\n[PIPELINE] Step 4/6: Building combined valid-pixel mask")
    mask_paths = {
        "l9_2022_ndvi": config.ALIGNED_DIR / "l9_2022_composite_ndvi_30m.tif",
        "l9_2026_ndvi": config.ALIGNED_DIR / "l9_2026_composite_ndvi_30m.tif",
        "l9_2022_ndbi": config.ALIGNED_DIR / "l9_2022_composite_ndbi_30m.tif",
        "l9_2026_ndbi": config.ALIGNED_DIR / "l9_2026_composite_ndbi_30m.tif",
        "l9_2022_lst": config.ALIGNED_DIR / "l9_2022_composite_lst_30m.tif",
        "l9_2026_lst": config.ALIGNED_DIR / "l9_2026_composite_lst_30m.tif",
        "s2_2022_ndvi": config.ALIGNED_DIR / "s2_2022_ndvi_30m.tif",
        "s2_2026_ndvi": config.ALIGNED_DIR / "s2_2026_ndvi_30m.tif",
        "s2_2022_ndbi": config.ALIGNED_DIR / "s2_2022_ndbi_30m.tif",
        "s2_2026_ndbi": config.ALIGNED_DIR / "s2_2026_ndbi_30m.tif",
    }
    record["valid_mask"] = create_combined_valid_mask(
        reference_path=config.REFERENCE_RASTER,
        mask_paths=mask_paths,
        output_path=config.MASKS_DIR / "valid_mask_30m.tif",
    )

    # 5. Convert vector GIS layers to 30 m rasters
    print("\n[PIPELINE] Step 5/6: Rasterizing vector GIS layers")
    record["vector_rasters"] = process_all_vector_layers()

    # 6. Build feature table
    print("\n[PIPELINE] Step 6/6: Building ML-ready feature table")
    record["feature_table"] = build_feature_table(
        aligned_dir=config.ALIGNED_DIR,
        masks_dir=config.MASKS_DIR,
        output_dir=config.FEATURES_DIR,
    )

    elapsed = time.time() - start_time
    record["finished"] = datetime.now().isoformat()
    record["elapsed_seconds"] = round(elapsed, 2)

    # Save pipeline record
    record_path = config.PROCESSED_DIR / "phase3_pipeline_record.json"
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  Phase 3 pipeline completed successfully")
    print(f"  Elapsed time: {elapsed:.1f} seconds")
    print(f"  Output directory: {config.PROCESSED_DIR}")
    print(f"  Pipeline record: {record_path}")
    print("=" * 70)

    return record


def main():
    record = run_phase3()
    # Print concise summary
    print("\nOutputs:")
    for key in ["feature_table"]:
        print(f"  {key}: {record[key]}")


if __name__ == "__main__":
    main()
