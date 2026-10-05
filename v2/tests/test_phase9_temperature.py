"""Synthetic unit tests for V2 Phase 9 (temperature reduction prediction).

Fast fixtures only: transition-panel construction invariants (delta
exactness, block assignment), metrics sanity, scenario arithmetic
(trees = ha x density; assumed veg change), leakage-audit completeness,
scenario-raster contract. No real rasters are trained on here.

Run from the project root:
    PYTHONUTF8=1 PYTHONPATH=v2/src .venv/Scripts/python.exe -m pytest v2/tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # the v2/ tree
SRC = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from v2.phase9 import build as B  # noqa: E402


# ---------------------------------------------------------------------------
# transition-panel invariants (synthetic fixture)
# ---------------------------------------------------------------------------

def _synthetic_panel(n=200, seed=7):
    rng = np.random.default_rng(seed)
    cur_lst = rng.normal(40, 3, n).astype(np.float32)
    next_lst = rng.normal(40, 3, n).astype(np.float32)
    # Mirror build.py: delta is the float32 subtraction next - current, so
    # the stored target is bit-exact with the stored endpoint columns.
    delta = next_lst - cur_lst
    return pd.DataFrame({
        "transition": ["2024->2025"] * n,
        "start_year": 2024,
        "end_year": 2025,
        "row": rng.integers(0, 1768, n),
        "col": rng.integers(0, 1874, n),
        "spatial_block_id": rng.integers(0, 25, n).astype(np.int16),
        "current_lst_C": cur_lst,
        "next_lst_C": next_lst,
        "delta_lst_C": delta,
    })


def test_panel_delta_exactness_and_blocks():
    df = _synthetic_panel()
    assert (df["delta_lst_C"].to_numpy()
            == (df["next_lst_C"] - df["current_lst_C"]).to_numpy()).all()
    assert df["spatial_block_id"].between(0, 24).all()
    assert set(B.TRANSITIONS) == {(2022, 2023), (2023, 2024),
                                  (2024, 2025), (2025, 2026)}
    assert B.SCENARIO_YEAR == 2026 and B.PRIMARY_SCENARIO == "v2_constrained"
    assert B.DENSITIES == (400, 1000, 2500)


def test_feature_columns_have_met_and_single_intervention():
    met = [c for c in B.FEATURE_COLS if c.startswith("current_met_")]
    assert len(met) == 6
    assert "vegetation_change" in B.FEATURE_COLS     # the ONLY change variable
    change_like = [c for c in B.FEATURE_COLS if "change" in c]
    assert change_like == ["vegetation_change"]


# ---------------------------------------------------------------------------
# metrics sanity + scenario arithmetic
# ---------------------------------------------------------------------------

def test_metrics_sanity_mae_le_rmse_and_r2_bounds():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred_good = np.array([1.1, 2.1, 2.9, 4.2])
    m = B.metrics(y, pred_good)
    assert m["mae_C"] <= m["rmse_C"] + 1e-12
    assert -1.0 - 1e-9 <= m["r2"] <= 1.0 + 1e-9
    assert m["n"] == 4
    pred_bad = np.array([10.0, -5.0, 0.0, 12.0])
    mb = B.metrics(y, pred_bad)
    assert mb["r2"] < m["r2"]
    const = B.metrics(y, np.array([2.5, 2.5, 2.5, 2.5]))
    assert const["mae_C"] <= const["rmse_C"] + 1e-12


def test_assumed_veg_change_and_tree_arithmetic():
    cfg = B.VegetationScenarioConfig()
    assert B.assumed_veg_change(400, cfg) == pytest.approx(0.02)
    assert B.assumed_veg_change(1000, cfg) == pytest.approx(0.05)
    assert B.assumed_veg_change(2500, cfg) == pytest.approx(0.125)
    # scenario assumption is explicit, not observed
    assert "No V2-observed tree-density-to-canopy conversion" in cfg.source
    # tree count = round(plantable_ha x density) exactly
    for ha_px, density in ((999, 400), (329, 1000), (1234, 2500)):
        ha = ha_px * B.HA_PER_PX
        assert int(round(ha * density)) == ha_px * int(round(density * B.HA_PER_PX))


def test_select_model_prefers_locked_mae():
    results = {
        "rf": {"locked_holdout": {"mae_C": 1.2}, "time_holdout": {"mae_C_mean": 1.0}},
        "xgb": {"locked_holdout": {"mae_C": 1.0}, "time_holdout": {"mae_C_mean": 1.5}},
    }
    assert B.select_model(results) == "xgb"
    results2 = {
        "rf": {"locked_holdout": {"mae_C": 1.0}, "time_holdout": {"mae_C_mean": 0.9}},
        "xgb": {"locked_holdout": {"mae_C": 1.0}, "time_holdout": {"mae_C_mean": 1.1}},
    }
    assert B.select_model(results2) == "rf"      # tie -> lower time MAE


# ---------------------------------------------------------------------------
# leakage audit completeness + raster contract
# ---------------------------------------------------------------------------

def test_leakage_audit_completeness():
    audit = B.build_leakage_audit()
    covered = " ".join(str(a.get("predictor", a.get("excluded"))) for a in audit)
    assert "intervention_variable" in json.dumps(audit)
    assert "current_met_" in covered
    assert "excluded_from_training_target" in json.dumps(audit)
    for c in B.FEATURE_COLS:
        if c.startswith("current_met_"):
            assert "current_met_*" in covered, c
        elif c in ("dist_road_m", "dist_vegetation_m", "dist_building_m",
                   "landuse_class"):
            assert c in covered, c
        else:
            assert c in covered, c


def json_str(a):
    import json
    return json.dumps(a)


def test_scenario_raster_contract(tmp_path):
    profile = {"height": 6, "width": 8, "transform": Affine(0.001, 0, 76.8,
                                                            0, -0.001, 28.9),
               "crs": CRS.from_epsg(4326), "count": 1, "dtype": "float32",
               "nodata": np.nan, "compress": "lzw"}
    rows = np.array([0, 3, 5])
    cols = np.array([0, 4, 7])
    vals = np.array([-0.5, 0.2, -1.1])
    p = tmp_path / "pred.tif"
    B.write_scenario_raster(vals, rows, cols, profile, p)
    with rasterio.open(p) as ds:
        arr = ds.read(1)
        assert (ds.height, ds.width) == (6, 8)
        assert ds.crs.to_epsg() == 4326
        assert ds.nodata == -9999.0
    support = np.isfinite(arr) & (arr != -9999.0)
    assert int(support.sum()) == 3
    assert arr[0, 0] == pytest.approx(-0.5)
    assert arr[5, 7] == pytest.approx(-1.1)
    assert arr[2, 2] == -9999.0
