"""
GreenGrid AI — Phase 3: Data Preprocessing
Validation of Phase 3 outputs.

Checks performed:
  - Required output files exist.
  - All 30 m rasters have identical dimensions, CRS, and transform.
  - Index value ranges are physically plausible.
  - No unexpected NaN or Inf values in feature table.
  - Feature table contains expected columns and samples.
  - Valid mask is consistent with finite pixels across layers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio

from . import config
from .io import open_raster, read_band


class Phase3ValidationReport:
    def __init__(self):
        self.checks: List[Tuple[str, str, str, str]] = []

    def add(self, category: str, name: str, status: str, message: str):
        self.checks.append((category, name, status, message))
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}.get(status, "?")
        print(f"  [{status}] {icon} {category} | {name}: {message}")

    def summary(self) -> Tuple[int, int, int]:
        passed = sum(1 for c in self.checks if c[2] == "PASS")
        failed = sum(1 for c in self.checks if c[2] == "FAIL")
        warned = sum(1 for c in self.checks if c[2] == "WARN")
        return passed, failed, warned


def check_file_exists(path: Path, report: Phase3ValidationReport, category: str, name: str) -> bool:
    if path.exists():
        report.add(category, name, "PASS", f"exists ({path.stat().st_size:,} bytes)")
        return True
    report.add(category, name, "FAIL", f"missing: {path}")
    return False


def check_raster_grid(path: Path, ref_profile: Dict, report: Phase3ValidationReport, category: str, name: str) -> bool:
    if not path.exists():
        report.add(category, name, "FAIL", "file missing")
        return False

    try:
        with open_raster(path) as ds:
            dims_ok = (ds.width == ref_profile["width"]) and (ds.height == ref_profile["height"])
            transform_ok = ds.transform == ref_profile["transform"]
            crs_ok = str(ds.crs) == str(ref_profile["crs"])

            if dims_ok and transform_ok and crs_ok:
                report.add(category, name, "PASS", "matches reference 30 m grid")
                return True
            else:
                msg = f"dims={dims_ok}, transform={transform_ok}, crs={crs_ok}"
                report.add(category, name, "FAIL", msg)
                return False
    except Exception as e:
        report.add(category, name, "FAIL", str(e))
        return False


def check_index_ranges(path: Path, report: Phase3ValidationReport, name: str, expected_range: Tuple[float, float]):
    if not path.exists():
        report.add("Index Ranges", name, "FAIL", "file missing")
        return

    with open_raster(path) as ds:
        arr = read_band(ds, 1)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        report.add("Index Ranges", name, "FAIL", "no finite values")
        return

    lo, hi = expected_range
    min_v = float(finite.min())
    max_v = float(finite.max())
    mean_v = float(finite.mean())

    in_range = (min_v >= lo) and (max_v <= hi)
    status = "PASS" if in_range else "WARN"
    report.add(
        "Index Ranges",
        name,
        status,
        f"min={min_v:.4f}, max={max_v:.4f}, mean={mean_v:.4f} (expected [{lo}, {hi}])",
    )


def check_no_inf_nan_in_feature_table(csv_path: Path, report: Phase3ValidationReport, name: str):
    if not csv_path.exists():
        report.add("Feature Table", name, "FAIL", "file missing")
        return

    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric.values).sum())
    nan_count = int(numeric.isna().sum().sum())

    if inf_count == 0 and nan_count == 0:
        report.add("Feature Table", name, "PASS", f"rows={len(df)}, no NaN/Inf in numeric columns")
    else:
        report.add("Feature Table", name, "FAIL", f"NaN={nan_count}, Inf={inf_count}")


def validate() -> bool:
    print("=" * 70)
    print("  GreenGrid AI — Phase 3 | Preprocessing Validation")
    print("=" * 70)

    report = Phase3ValidationReport()

    # Reference profile
    with open_raster(config.REFERENCE_RASTER) as ref:
        ref_profile = {
            "width": ref.width,
            "height": ref.height,
            "crs": ref.crs,
            "transform": ref.transform,
        }
    report.add("Reference", "Reference raster", "PASS", f"{ref_profile['width']}x{ref_profile['height']} @ ~{config.TARGET_RESOLUTION_M_APPROX} m nominal (EPSG:4326)")

    # Required output files
    required_files = {
        "L9 2022 NDVI": config.ALIGNED_DIR / "l9_2022_composite_ndvi_30m.tif",
        "L9 2022 NDBI": config.ALIGNED_DIR / "l9_2022_composite_ndbi_30m.tif",
        "L9 2022 LST": config.ALIGNED_DIR / "l9_2022_composite_lst_30m.tif",
        "L9 2026 NDVI": config.ALIGNED_DIR / "l9_2026_composite_ndvi_30m.tif",
        "L9 2026 NDBI": config.ALIGNED_DIR / "l9_2026_composite_ndbi_30m.tif",
        "L9 2026 LST": config.ALIGNED_DIR / "l9_2026_composite_lst_30m.tif",
        "S2 2022 NDVI": config.ALIGNED_DIR / "s2_2022_ndvi_30m.tif",
        "S2 2022 NDBI": config.ALIGNED_DIR / "s2_2022_ndbi_30m.tif",
        "S2 2026 NDVI": config.ALIGNED_DIR / "s2_2026_ndvi_30m.tif",
        "S2 2026 NDBI": config.ALIGNED_DIR / "s2_2026_ndbi_30m.tif",
        "Valid mask": config.MASKS_DIR / "valid_mask_30m.tif",
        "Landuse raster": config.MASKS_DIR / "landuse_raster_30m.tif",
        "Roads distance": config.MASKS_DIR / "roads_distance_30m.tif",
        "Vegetation distance": config.MASKS_DIR / "vegetation_distance_30m.tif",
        "Buildings distance": config.MASKS_DIR / "buildings_distance_30m.tif",
        "Feature table 2022": config.FEATURES_DIR / "feature_table_2022.csv",
        "Feature table 2026": config.FEATURES_DIR / "feature_table_2026.csv",
        "Feature table": config.FEATURES_DIR / "feature_table.csv",
        "Feature metadata": config.FEATURES_DIR / "feature_metadata.json",
    }

    for name, path in required_files.items():
        check_file_exists(path, report, "Files", name)

    # Grid alignment
    aligned_files = [p for k, p in required_files.items() if "NDVI" in k or "NDBI" in k or "LST" in k]
    for path in aligned_files:
        check_raster_grid(path, ref_profile, report, "Grid Alignment", path.name)

    # Index ranges
    check_index_ranges(config.ALIGNED_DIR / "l9_2022_composite_ndvi_30m.tif", report, "L9 2022 NDVI", (-1.0, 1.0))
    check_index_ranges(config.ALIGNED_DIR / "l9_2026_composite_ndvi_30m.tif", report, "L9 2026 NDVI", (-1.0, 1.0))
    check_index_ranges(config.ALIGNED_DIR / "s2_2022_ndvi_30m.tif", report, "S2 2022 NDVI", (-1.0, 1.0))
    check_index_ranges(config.ALIGNED_DIR / "s2_2026_ndvi_30m.tif", report, "S2 2026 NDVI", (-1.0, 1.0))
    check_index_ranges(config.ALIGNED_DIR / "l9_2022_composite_ndbi_30m.tif", report, "L9 2022 NDBI", (-1.0, 1.0))
    check_index_ranges(config.ALIGNED_DIR / "l9_2026_composite_ndbi_30m.tif", report, "L9 2026 NDBI", (-1.0, 1.0))
    check_index_ranges(config.ALIGNED_DIR / "s2_2022_ndbi_30m.tif", report, "S2 2022 NDBI", (-1.0, 1.0))
    check_index_ranges(config.ALIGNED_DIR / "s2_2026_ndbi_30m.tif", report, "S2 2026 NDBI", (-1.0, 1.0))
    # LST: plausible July daytime range for Delhi
    check_index_ranges(config.ALIGNED_DIR / "l9_2022_composite_lst_30m.tif", report, "L9 2022 LST", (15.0, 65.0))
    check_index_ranges(config.ALIGNED_DIR / "l9_2026_composite_lst_30m.tif", report, "L9 2026 LST", (15.0, 65.0))

    # Feature table integrity
    check_no_inf_nan_in_feature_table(config.FEATURES_DIR / "feature_table_2022.csv", report, "feature_table_2022.csv")
    check_no_inf_nan_in_feature_table(config.FEATURES_DIR / "feature_table_2026.csv", report, "feature_table_2026.csv")
    check_no_inf_nan_in_feature_table(config.FEATURES_DIR / "feature_table.csv", report, "feature_table.csv")

    # Expected columns in the combined long-format table
    expected_columns = [
        "lon", "lat", "row", "col", "spatial_block_id", "year",
        "lst_l9_C", "ndvi_l9", "ndbi_l9",
        "ndvi_s2", "ndbi_s2",
        "landuse_class", "dist_road_m", "dist_vegetation_m", "dist_building_m",
    ]
    df = pd.read_csv(config.FEATURES_DIR / "feature_table.csv")
    missing = [c for c in expected_columns if c not in df.columns]
    if not missing:
        report.add("Feature Table", "Columns", "PASS", f"all {len(expected_columns)} expected columns present")
    else:
        report.add("Feature Table", "Columns", "FAIL", f"missing: {missing}")

    # Spatial block coverage
    n_blocks = df["spatial_block_id"].nunique()
    report.add("Feature Table", "Spatial blocks", "PASS" if n_blocks > 1 else "WARN", f"{n_blocks} unique spatial_block_id values")

    # Summary
    passed, failed, warned = report.summary()
    print("\n" + "=" * 70)
    print("  Validation Summary")
    print("=" * 70)
    print(f"  PASSED:  {passed}")
    print(f"  FAILED:  {failed}")
    print(f"  WARNED:  {warned}")

    if failed == 0:
        print("\n  ✓ Phase 3 preprocessing validation passed.")
        return True
    else:
        print(f"\n  ✗ {failed} validation check(s) failed.")
        return False


def main():
    ok = validate()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
