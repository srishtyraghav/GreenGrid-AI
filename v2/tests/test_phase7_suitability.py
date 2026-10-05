"""Synthetic unit tests for V2 Phase 7 (plantation suitability).

Fast fixtures only: frozen-formula units (need/opportunity/components/class
boundaries/gating), constraint burning + attribution precedence, raster
NoData semantics, scenario tagging. No real rasters burned here.

Run from the project root:
    PYTHONUTF8=1 PYTHONPATH=v2/src .venv/Scripts/python.exe -m pytest v2/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # the v2/ tree
SRC = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from v2.common import Grid  # noqa: E402
from v2.phase7 import build, constraints as C, suitability as S  # noqa: E402


def tiny_grid(h=6, w=8):
    return Grid(Affine(0.001, 0, 76.8, 0, -0.001, 28.9), h, w,
                CRS.from_epsg(4326))


# ---------------------------------------------------------------------------
# frozen formula units
# ---------------------------------------------------------------------------

def test_need_formula_and_weights():
    norm = {"severity_score": np.array([100.0, 0.0]),
            "one_minus_ndvi": np.array([0.0, 100.0]),
            "ndbi": np.array([50.0, 50.0]),
            "lst": np.array([0.0, 0.0])}
    need = S.compute_heat_need(norm)
    # 0.40*100 + 0.25*0 + 0.20*50 + 0.15*0 = 50 ; second row = 25+10+0=35
    assert np.allclose(need, [50.0, 35.0])
    assert abs(sum(S.NEED_WEIGHTS.values()) - 1.0) <= 1e-12


def test_opportunity_formula_and_components():
    norm = {"ndbi": np.array([80.0]), "ndvi": np.array([30.0])}
    static = {"landuse": np.array([6]), "dist_road_m": np.array([300.0]),
              "dist_vegetation_m": np.array([100.0])}
    out = S.compute_opportunity(norm, static)
    # 0.30*90 + 0.25*(100-80) + 0.15*100 + 0.15*(100*(1-100/500)) + 0.15*(100-30)
    hand = 0.30 * 90 + 0.25 * 20 + 0.15 * 100 + 0.15 * 80 + 0.15 * 70
    assert np.allclose(out["opportunity"], [hand])
    assert abs(sum(S.OPPORTUNITY_WEIGHTS.values()) - 1.0) <= 1e-12


def test_road_accessibility_piecewise():
    d = np.array([0.0, 25.0, 50.0, 500.0, 1250.0, 2000.0, 5000.0])
    s = S.road_accessibility_score(d)
    assert np.allclose(s, [0.0, 20.0, 100.0, 100.0, 65.0, 30.0, 30.0])


def test_green_proximity_and_landuse_eligibility():
    assert np.allclose(S.green_proximity_score(np.array([0.0, 250.0, 500.0, 900.0])),
                       [100.0, 50.0, 0.0, 0.0])
    lu = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])
    assert S.landuse_eligibility(lu).tolist() == [50, 40, 30, 50, 35, 20,
                                                  90, 30, 60, 50]


def test_classify_boundaries_belong_to_lower_class():
    probe = np.array([0.0, 20.0, 20.0001, 40.0, 60.0, 80.0, 100.0])
    assert S.classify(probe).tolist() == [0, 0, 1, 1, 2, 3, 4]


def test_robust_minmax_clips_at_percentiles():
    rng = np.random.default_rng(3)
    vals = np.concatenate([rng.normal(50, 5, 1000), [1e6, -1e6]])
    scaled, bounds = S.robust_minmax(vals)
    assert scaled.min() >= 0.0 and scaled.max() <= 100.0
    assert bounds["p1"] > -1e6 and bounds["p99"] < 1e6


def test_gated_product_and_priority_tiers():
    need = np.array([100.0, 100.0, 0.0])
    opp = np.array([0.0, 50.0, 100.0])
    assert np.allclose(S.gated_product(need, opp), [0.0, 50.0, 0.0])
    cls = np.array([0, 1, 2, 3, 4], dtype=np.int16)
    assert S.priority_tier(cls).tolist() == [0, 0, 1, 2, 3]


# ---------------------------------------------------------------------------
# constraint burning + attribution
# ---------------------------------------------------------------------------

@pytest.fixture()
def constraint_dir(tmp_path):
    grid = tiny_grid()
    g = grid.transform
    water = gpd.GeoDataFrame(
        {"v": [1]}, geometry=[box(g.c + 0.5 * g.a, g.f - 3.5 * abs(g.e),
                                  g.c + 3.5 * g.a, g.f - 0.5 * abs(g.e))],
        crs="EPSG:4326")
    buildings = gpd.GeoDataFrame(
        {"v": [1]}, geometry=[box(g.c + 2.5 * g.a, g.f - 2.5 * abs(g.e),
                                  g.c + 4.5 * g.a, g.f - 1.5 * abs(g.e))],
        crs="EPSG:4326")
    roads = gpd.GeoDataFrame(
        {"v": [1]}, geometry=[box(g.c + 6.5 * g.a, g.f - 5.5 * abs(g.e),
                                  g.c + 7.5 * g.a, g.f - 4.5 * abs(g.e))],
        crs="EPSG:4326")
    d = tmp_path / "constraints"
    d.mkdir()
    water.to_file(d / "osm_water_delhi.geojson", driver="GeoJSON")
    buildings.to_file(d / "osm_buildings_delhi.geojson", driver="GeoJSON")
    roads.to_file(d / "osm_road_surfaces_delhi.geojson", driver="GeoJSON")
    return d


def test_constraint_burn_and_attribution(tmp_path, constraint_dir):
    grid = tiny_grid()
    stack = C.build_constraint_stack(constraint_dir, grid,
                                     tmp_path / "burn")
    w, b, r = stack["masks"]["water"], stack["masks"]["buildings"], \
        stack["masks"]["road_surfaces"]
    assert w[1, 2] and w.sum() >= 8            # 3x3-ish block at top-left
    assert b[2, 3] and b[1, 3]                 # building block
    assert r[5, 7]                             # road-surface cell
    # precedence: overlap px (rows1-2, col3 is water+building) -> water (code 1)
    assert stack["attributed"][1, 3] == 1
    assert stack["attributed"][2, 4] == 2      # building-only -> 2
    assert stack["attributed"][5, 7] == 3      # road-only -> 3
    assert int((stack["attributed"] > 0).sum()) <= int((w | b | r).sum())


# ---------------------------------------------------------------------------
# NoData semantics of the writers
# ---------------------------------------------------------------------------

def test_write_score_and_class_nodata(tmp_path):
    grid = tiny_grid()
    rows = np.array([0, 1, 5])
    cols = np.array([0, 3, 7])
    shape = grid.shape
    meta = build.write_score(np.array([10.0, 50.0, 90.0]),
                             tmp_path / "s.tif", grid, shape, rows, cols)
    with rasterio.open(meta["path"]) as ds:
        s = ds.read(1)
        assert ds.nodata == -1.0
    assert s[0, 0] == 10.0 and s[5, 7] == 90.0
    assert s[2, 2] == -1.0                     # outside domain -> NoData
    meta = build.write_class(np.array([0, 3, 4]),
                             tmp_path / "c.tif", grid, shape, rows, cols,
                             np.int16, S.CLASS_NODATA)
    with rasterio.open(meta["path"]) as ds:
        c = ds.read(1)
        assert ds.nodata == -1
    assert c[1, 3] == 3 and c[2, 2] == -1


# ---------------------------------------------------------------------------
# scenario tagging (constants + smoke-artifact layout if present)
# ---------------------------------------------------------------------------

def test_scenario_constants_and_tags():
    assert build.SCENARIOS == ("v1_parity", "v2_constrained")
    for scenario in build.SCENARIOS:
        for name in ("suitability_{t}_{y}.tif", "priority_{t}_{y}.tif",
                     "priority_tiers_{t}_{y}.csv",
                     "priority_zone_statistics_{t}_{y}.csv"):
            assert "{t}" in name and "{y}" in name


def test_smoke_outputs_tagged_both_scenarios():
    out = PROJECT_ROOT / "data" / "phase7"
    rec = out / "phase7_pipeline_record.json"
    if not rec.exists():
        pytest.skip("phase7 smoke not run here")
    import json
    years = json.loads(rec.read_text())["years"]
    for scenario in ("v1_parity", "v2_constrained"):
        for y in years:
            assert (out / scenario / "rasters" / f"suitability_{scenario}_{y}.tif").exists()
            assert (out / scenario / "tables" / f"priority_tiers_{scenario}_{y}.csv").exists()
