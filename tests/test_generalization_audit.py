"""Smoke tests for ``models.generalization_audit``.

Runs fast, pure/light diagnostics on a tiny synthetic Phase-4-like DataFrame
(200 rows, 2 spatial blocks) writing into a temporary output directory:

* duplicate_audit: no duplicates -> OK verdicts; injected duplicate -> INFO;
  repeated (row, col) across years is reported as legitimate (INFO).
* leakage_audit: PASS when forbidden columns are absent from predictor_cols;
  WARN when ``lst_C`` is (incorrectly) included.
* Moran's I helper: strong positive autocorrelation for a spatial gradient;
  NaN for a constant variable; neighbour padding for tiny inputs.
* run_audit end-to-end with the two light steps writes the generalization
  summary JSON.

No model training is performed.  Run from the repository root:

    PYTHONPATH=src python3 tests/test_generalization_audit.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import models.generalization_audit as ga


def make_synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a tiny synthetic Phase-4-like dataset (2 blocks, 2 years)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    row = idx % 10
    col = (idx // 10) % 10
    # First half of rows = 2022, second half = 2026, so every (row, col)
    # location is observed in BOTH years (legitimate temporal repeats).
    years = np.where(idx < n // 2, 2022, 2026)
    lst = rng.normal(30.0, 3.0, n)
    return pd.DataFrame(
        {
            "lon": 85.0 + col * 0.0003,
            "lat": 27.0 + row * 0.0003,
            "row": row,
            "col": col,
            "year": years,
            "spatial_block_id": np.where(np.arange(n) < n // 2, 1, 2),
            "lst_C": lst,
            "ndvi": rng.random(n),
            "ndbi": rng.random(n) * 0.5,
            "vegetation_cover": rng.random(n),
            "landuse_class": rng.integers(1, 4, n),
            "dist_road_m": rng.random(n) * 500,
            "dist_vegetation_m": rng.random(n) * 300,
            "dist_building_m": rng.random(n) * 400,
        }
    )


PREDICTOR_COLS = [
    "ndvi",
    "ndbi",
    "vegetation_cover",
    "landuse_class",
    "dist_road_m",
    "dist_vegetation_m",
    "dist_building_m",
    "year",
]


def test_duplicate_audit(tmp: Path) -> None:
    df = make_synthetic_df()
    summary = ga.duplicate_audit(df)
    # lon/lat repeat across the two years (same locations), so exact
    # lon/lat duplicates are expected; (row, col, year) stays unique.
    assert summary["checks"]["exact_duplicate_lonlat"] == 100
    assert summary["checks"]["duplicate_row_col_year"] == 0
    assert summary["checks"]["duplicate_block_row_col_year"] == 0
    # Each (row, col) appears twice (two years) -> legitimate temporal repeats.
    assert summary["checks"]["repeated_row_col_across_years"] == 100
    out = pd.read_csv(tmp / "data_integrity_duplicate_audit.csv")
    verdicts = dict(zip(out["check"], out["verdict"]))
    assert verdicts["repeated_row_col_across_years"] == "INFO"
    assert verdicts["exact_duplicate_lonlat"] == "INFO"
    assert all(v in {"OK", "INFO"} for v in verdicts.values())

    # Inject an extra copy of one row -> one more duplicate lon/lat row.
    dup = df.copy()
    dup.loc[len(dup)] = dup.iloc[0]
    summary2 = ga.duplicate_audit(dup)
    assert summary2["checks"]["exact_duplicate_lonlat"] == 101
    assert summary2["checks"]["duplicate_row_col_year"] == 1


def test_leakage_audit(tmp: Path) -> None:
    df = make_synthetic_df()
    ok = ga.leakage_audit(df, PREDICTOR_COLS)
    assert ok["all_pass"], ok
    assert ok["forbidden_in_predictor_cols"] == []

    bad = ga.leakage_audit(df, PREDICTOR_COLS + ["lst_C"])
    assert not bad["all_pass"]
    assert bad["forbidden_in_predictor_cols"] == ["lst_C"]
    out = pd.read_csv(tmp / "leakage_audit.csv")
    lst_row = out[out["variable_or_operation"] == "lst_C"].iloc[0]
    assert lst_row["allowed_as_predictor"] == "NO"
    assert str(lst_row["verification"]).startswith("WARN")
    thresh_row = out[out["variable_or_operation"] == "target-derived thresholds"].iloc[0]
    assert "experiments.run_spatial_cv" in str(thresh_row["verification"])


def test_morans_i() -> None:
    # Strong positive spatial autocorrelation: values increase along x.
    coords = np.array([[i, 0] for i in range(20)], dtype=float) * 30.0
    gradient = np.arange(20, dtype=float)
    neighbors = ga._knn_neighbor_indices(coords, k=8)
    I, expected = ga.morans_i(gradient, neighbors)
    assert I > 0.5, f"expected strong positive I, got {I}"
    assert abs(expected - (-1.0 / 19)) < 1e-12

    # Constant variable -> zero variance -> NaN I, finite expected value.
    I_const, expected_const = ga.morans_i(np.ones(20), neighbors)
    assert np.isnan(I_const)
    assert expected_const == expected

    # Tiny input (fewer points than k) must not crash.
    small_coords = np.array([[0.0, 0.0], [30.0, 0.0]])
    small_neighbors = ga._knn_neighbor_indices(small_coords, k=8)
    assert small_neighbors.shape == (2, 8)
    I_small, _ = ga.morans_i(np.array([1.0, 2.0]), small_neighbors)
    assert np.isfinite(I_small)

    # Permutation p-value helper runs and returns sane values.
    I_p, E_p, p_p = ga.morans_i_with_permutation(
        gradient, coords, k=8, n_permutations=19, seed=42
    )
    assert 0.0 < p_p <= 1.0
    assert abs(I_p - I) < 1e-12


def test_run_audit_light_steps(tmp: Path) -> None:
    df = make_synthetic_df()
    results = ga.run_audit(df, PREDICTOR_COLS, steps=["duplicate_audit", "leakage_audit"])
    assert set(results) == {"duplicate_audit", "leakage_audit"}
    summary_path = tmp / "phase5_generalization_summary.json"
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary["leakage_audit"]["all_pass"] is True
    assert summary["seed_stability"] == "not_run"
    assert isinstance(summary["remaining_limitations"], list)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="generalization_audit_test_"))
    ga.OUTPUT_DIR = tmp
    print(f"OUTPUT_DIR -> {tmp}")
    test_duplicate_audit(tmp)
    print("duplicate_audit OK")
    test_leakage_audit(tmp)
    print("leakage_audit OK")
    test_morans_i()
    print("morans_i OK")
    test_run_audit_light_steps(tmp)
    print("run_audit light steps OK")
    print("All generalization_audit smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
