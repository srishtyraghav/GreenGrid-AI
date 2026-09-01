"""Independent validation of Phase 4 feature-extraction outputs.

Checks file existence, grid consistency, value ranges, raster/table agreement,
year separation, map generation, and reproducibility metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio

from features.config import (
    COMBINED_DATASET_CSV,
    FEATURE_METADATA_JSON,
    L9_LST_2022,
    L9_LST_2026,
    PHASE4_MAPS_DIR,
    PIPELINE_RECORD_JSON,
    REFERENCE_RASTER_PATH,
    S2_NDBI_2022,
    S2_NDBI_2026,
    S2_NDVI_2022,
    S2_NDVI_2026,
    VEGETATION_COVER_RASTER_2022,
    VEGETATION_COVER_RASTER_2026,
)
from features.io import read_raster_array
from features.vegetation_cover import calculate_proportional_vegetation_cover


# Year-to-raster lookup for spot checks
YEARLY_RASTERS = {
    2022: {
        "lst": L9_LST_2022,
        "ndvi": S2_NDVI_2022,
        "ndbi": S2_NDBI_2022,
        "vegetation_cover": VEGETATION_COVER_RASTER_2022,
    },
    2026: {
        "lst": L9_LST_2026,
        "ndvi": S2_NDVI_2026,
        "ndbi": S2_NDBI_2026,
        "vegetation_cover": VEGETATION_COVER_RASTER_2026,
    },
}


class ValidationReport:
    """Collect validation checks and emit a structured report."""

    def __init__(self):
        self.checks: List[Dict] = []

    def add(self, category: str, name: str, status: str, message: str = "") -> None:
        self.checks.append(
            {
                "category": category,
                "name": name,
                "status": status,
                "message": message,
            }
        )

    def summary(self) -> Dict:
        counts = {"PASS": 0, "FAIL": 0, "WARN": 0}
        for c in self.checks:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        return {
            "total": len(self.checks),
            "pass": counts["PASS"],
            "fail": counts["FAIL"],
            "warn": counts["WARN"],
        }

    def to_dict(self) -> Dict:
        return {"checks": self.checks, "summary": self.summary()}


def _grid_matches_reference(raster_path: Path, ref_path: Path) -> Tuple[bool, str]:
    """Return (ok, message) comparing a raster's grid to the reference."""
    try:
        with rasterio.open(raster_path) as src, rasterio.open(ref_path) as ref:
            if src.crs != ref.crs:
                return False, f"CRS mismatch: {src.crs} vs {ref.crs}"
            if src.shape != ref.shape:
                return False, f"Shape mismatch: {src.shape} vs {ref.shape}"
            if not np.allclose(src.transform, ref.transform, atol=1e-9):
                return False, "Affine transform mismatch"
            return True, "Grid matches reference"
    except Exception as exc:
        return False, f"Error reading raster: {exc}"


def _check_file_exists(report: ValidationReport, path: Path, label: str) -> bool:
    if path.exists():
        report.add("files", f"{label} exists", "PASS", str(path))
        return True
    report.add("files", f"{label} exists", "FAIL", f"Missing: {path}")
    return False


def validate_phase4() -> ValidationReport:
    """Run the full Phase 4 validation suite."""
    report = ValidationReport()

    # 1. Output files
    required_files = {
        "vegetation_cover_2022": VEGETATION_COVER_RASTER_2022,
        "vegetation_cover_2026": VEGETATION_COVER_RASTER_2026,
        "combined_dataset_csv": COMBINED_DATASET_CSV,
        "feature_metadata_json": FEATURE_METADATA_JSON,
        "pipeline_record_json": PIPELINE_RECORD_JSON,
    }
    all_files_exist = True
    for label, path in required_files.items():
        if not _check_file_exists(report, path, label):
            all_files_exist = False

    expected_maps = [
        "lst_2022.png",
        "lst_2026.png",
        "ndvi_2022.png",
        "ndvi_2026.png",
        "ndbi_2022.png",
        "ndbi_2026.png",
        "vegetation_cover_2022.png",
        "vegetation_cover_2026.png",
    ]
    for map_name in expected_maps:
        map_path = PHASE4_MAPS_DIR / map_name
        _check_file_exists(report, map_path, f"map_{map_name}")

    # 2. Vegetation cover rasters
    for year, path in (
        (2022, VEGETATION_COVER_RASTER_2022),
        (2026, VEGETATION_COVER_RASTER_2026),
    ):
        if not path.exists():
            report.add("vegetation_cover", f"{year} raster loadable", "FAIL", "File missing")
            continue

        ok, msg = _grid_matches_reference(path, REFERENCE_RASTER_PATH)
        report.add(
            "vegetation_cover",
            f"{year} grid matches reference",
            "PASS" if ok else "FAIL",
            msg,
        )

        arr = read_raster_array(path)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            report.add("vegetation_cover", f"{year} has valid pixels", "FAIL", "No valid pixels")
            continue
        report.add("vegetation_cover", f"{year} has valid pixels", "PASS", f"{valid.size} valid")

        if np.nanmin(arr) >= -1e-9 and np.nanmax(arr) <= 1.0 + 1e-9:
            report.add(
                "vegetation_cover",
                f"{year} values in [0, 1]",
                "PASS",
                f"min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}",
            )
        else:
            report.add(
                "vegetation_cover",
                f"{year} values in [0, 1]",
                "FAIL",
                f"min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}",
            )

    # 3. Vegetation cover formula consistency
    for year, ndvi_path, pvc_path in (
        (2022, S2_NDVI_2022, VEGETATION_COVER_RASTER_2022),
        (2026, S2_NDVI_2026, VEGETATION_COVER_RASTER_2026),
    ):
        if not (ndvi_path.exists() and pvc_path.exists()):
            report.add(
                "vegetation_cover",
                f"{year} formula consistency",
                "FAIL",
                "Input rasters missing",
            )
            continue
        ndvi = read_raster_array(ndvi_path)
        pvc = read_raster_array(pvc_path)
        expected = calculate_proportional_vegetation_cover(ndvi)
        # Compare only valid pixels
        mask = ~np.isnan(ndvi) & ~np.isnan(pvc)
        if np.allclose(expected[mask], pvc[mask], atol=1e-6):
            report.add(
                "vegetation_cover",
                f"{year} formula consistency",
                "PASS",
                f"{mask.sum()} valid pixels match",
            )
        else:
            diff = np.abs(expected[mask] - pvc[mask])
            report.add(
                "vegetation_cover",
                f"{year} formula consistency",
                "FAIL",
                f"Max absolute difference: {np.nanmax(diff):.6f}",
            )

    # 4. Combined dataset
    if COMBINED_DATASET_CSV.exists():
        df = pd.read_csv(COMBINED_DATASET_CSV)
        required_cols = [
            "lon",
            "lat",
            "row",
            "col",
            "spatial_block_id",
            "year",
            "lst_C",
            "ndvi",
            "ndbi",
            "vegetation_cover",
            "landuse_class",
            "dist_road_m",
            "dist_vegetation_m",
            "dist_building_m",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if not missing:
            report.add("dataset", "Required columns present", "PASS", str(required_cols))
        else:
            report.add("dataset", "Required columns present", "FAIL", f"Missing: {missing}")

        # No NaN in required feature columns
        feature_cols = ["lst_C", "ndvi", "ndbi", "vegetation_cover"]
        nan_counts = {c: int(df[c].isna().sum()) for c in feature_cols}
        if all(v == 0 for v in nan_counts.values()):
            report.add("dataset", "No NaN in feature columns", "PASS", str(nan_counts))
        else:
            report.add("dataset", "No NaN in feature columns", "FAIL", str(nan_counts))

        # Year separation
        years = sorted(df["year"].unique().tolist())
        if years == [2022, 2026]:
            report.add("dataset", "Year values", "PASS", f"Years: {years}")
        else:
            report.add("dataset", "Year values", "FAIL", f"Unexpected years: {years}")

        # Equal counts per year
        year_counts = df["year"].value_counts().to_dict()
        if len(set(year_counts.values())) == 1:
            report.add("dataset", "Balanced years", "PASS", str(year_counts))
        else:
            report.add("dataset", "Balanced years", "WARN", str(year_counts))

        # No duplicate (row, col, year)
        dupes = df.duplicated(subset=["row", "col", "year"]).sum()
        if dupes == 0:
            report.add("dataset", "No duplicate (row, col, year)", "PASS")
        else:
            report.add("dataset", "No duplicate (row, col, year)", "FAIL", f"{dupes} duplicates")

        # All sample coordinates lie inside the reference raster bounds
        with rasterio.open(REFERENCE_RASTER_PATH) as ref:
            n_rows, n_cols = ref.height, ref.width
        out_of_bounds = (
            (df["row"] < 0)
            | (df["row"] >= n_rows)
            | (df["col"] < 0)
            | (df["col"] >= n_cols)
        ).sum()
        if out_of_bounds == 0:
            report.add(
                "dataset",
                "Sample coordinates within raster bounds",
                "PASS",
                f"Reference shape: ({n_rows}, {n_cols})",
            )
        else:
            report.add(
                "dataset",
                "Sample coordinates within raster bounds",
                "FAIL",
                f"{out_of_bounds} rows outside reference shape ({n_rows}, {n_cols})",
            )

        # Vegetation cover range in table
        if df["vegetation_cover"].min() >= -1e-9 and df["vegetation_cover"].max() <= 1.0 + 1e-9:
            report.add(
                "dataset",
                "Table vegetation_cover in [0, 1]",
                "PASS",
                f"min={df['vegetation_cover'].min():.4f}, max={df['vegetation_cover'].max():.4f}",
            )
        else:
            report.add(
                "dataset",
                "Table vegetation_cover in [0, 1]",
                "FAIL",
                f"min={df['vegetation_cover'].min():.4f}, max={df['vegetation_cover'].max():.4f}",
            )

        # Spot check: table values match raster values for a random sample
        sample = df.sample(n=min(50, len(df)), random_state=42)
        mismatches = 0
        for year, group in sample.groupby("year"):
            lst = read_raster_array(YEARLY_RASTERS[year]["lst"])
            ndvi = read_raster_array(YEARLY_RASTERS[year]["ndvi"])
            ndbi = read_raster_array(YEARLY_RASTERS[year]["ndbi"])
            pvc = read_raster_array(YEARLY_RASTERS[year]["vegetation_cover"])
            for _, row in group.iterrows():
                r, c = int(row["row"]), int(row["col"])
                if not np.isclose(row["lst_C"], lst[r, c], atol=1e-4):
                    mismatches += 1
                if not np.isclose(row["ndvi"], ndvi[r, c], atol=1e-6):
                    mismatches += 1
                if not np.isclose(row["ndbi"], ndbi[r, c], atol=1e-6):
                    mismatches += 1
                if not np.isclose(row["vegetation_cover"], pvc[r, c], atol=1e-6):
                    mismatches += 1
        if mismatches == 0:
            report.add("dataset", "Raster/table spot check", "PASS", "50 sampled rows match")
        else:
            report.add("dataset", "Raster/table spot check", "FAIL", f"{mismatches} mismatches")
    else:
        report.add("dataset", "Combined dataset loadable", "FAIL", "CSV missing")

    # 5. Metadata
    if FEATURE_METADATA_JSON.exists():
        with open(FEATURE_METADATA_JSON, encoding="utf-8") as f:
            meta = json.load(f)
        if "authoritative_feature_sources" in meta:
            report.add("metadata", "Authoritative sources documented", "PASS")
        else:
            report.add("metadata", "Authoritative sources documented", "FAIL")
        if "vegetation_cover_methodology" in meta:
            report.add("metadata", "Vegetation cover methodology documented", "PASS")
        else:
            report.add("metadata", "Vegetation cover methodology documented", "FAIL")
    else:
        report.add("metadata", "Metadata loadable", "FAIL", "JSON missing")

    return report


def main(argv: List[str] | None = None) -> int:
    """CLI entry point for Phase 4 validation."""
    parser = argparse.ArgumentParser(description="Validate Phase 4 outputs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the validation report JSON.",
    )
    args = parser.parse_args(argv)

    report = validate_phase4()
    summary = report.summary()

    print(json.dumps(report.to_dict(), indent=2, default=str))
    print(f"\nSummary: {summary['pass']} PASS / {summary['fail']} FAIL / {summary['warn']} WARN")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

    return 1 if summary["fail"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
