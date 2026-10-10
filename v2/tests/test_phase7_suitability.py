"""Synthetic + artifact tests for V2 Phase 7 v3 (full-area pooled suitability).

Covers: pooled normalization, the documented v3 formulas (cooling need,
opportunity weights verbatim-V1, gated suitability, fixed-threshold priority
classes), raster contracts on the real outputs (skip if not built), class
shares, exclusion-accounting partition, and backend endpoint <-> disk
consistency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # the v2/ tree
SRC = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from v2.phase7 import build as B  # noqa: E402
from v2.phase7 import suitability as S  # noqa: E402


# ---------------------------------------------------------------------------
# formula units (synthetic)
# ---------------------------------------------------------------------------

def test_cooling_need_formula_and_weights():
    w = S.COOLING_NEED_WEIGHTS
    assert abs(sum(w.values()) - 1.0) <= 1e-12
    need = S.cooling_need(np.array([1.0, 0.0, 0.5]),
                          np.array([0.0, 1.0, 0.5]),
                          np.array([0.0, 0.0, 0.5]))
    assert np.allclose(need, [0.55, 0.25, 0.50], atol=1e-12)
    assert need.min() >= 0.0 and need.max() <= 1.0


def test_opportunity_weights_verbatim_v1_and_components():
    w = S.OPPORTUNITY_WEIGHTS_01
    assert w == {"landuse_eligibility": 0.30, "built_up_inverse": 0.25,
                 "road_accessibility": 0.15, "green_proximity": 0.15,
                 "planting_headroom": 0.15}
    norm = {"ndvi": np.array([0.2, 0.8]), "ndbi": np.array([0.9, 0.1])}
    static = {"landuse": np.array([6, 5]),      # residential 0.9 / industrial 0.2
              "dist_road_m": np.array([300.0, 300.0]),
              "dist_vegetation_m": np.array([100.0, 100.0])}
    opp = S.opportunity_component_01(norm, static)
    # residential: 0.30*0.9 + 0.25*0.1 + 0.15*1.0 + 0.15*0.8 + 0.15*0.8
    assert np.allclose(opp[0], 0.27 + 0.025 + 0.15 + 0.12 + 0.12, atol=1e-9)
    assert opp.min() >= 0.0 and opp.max() <= 1.0


def test_pooled_normalize_fixed_reference():
    p1, p99 = 10.0, 50.0
    vals = np.array([5.0, 10.0, 30.0, 50.0, 60.0, np.nan])
    n = S.pooled_normalize(vals, p1, p99)
    assert np.isnan(n[-1])
    assert np.allclose(n[:5], [0.0, 0.0, 0.5, 1.0, 1.0])


def test_classify_priority_fixed_thresholds_and_nan():
    score = np.array([0.1, 0.4, 0.4, 0.6, np.nan])
    cls = S.classify_priority(score, t1=0.4, t2=0.6)
    assert cls.tolist() == [0, 1, 1, 2, -1]


def test_suitability_gated_product():
    out = S.suitability_product(np.array([1.0, 0.0, 0.5]), np.array([0.5, 1.0, 0.5]))
    assert np.allclose(out, [0.5, 0.0, 0.25])


def test_road_band_and_green_proximity_unchanged():
    d = np.array([0.0, 25.0, 50.0, 500.0, 1250.0, 2000.0, 5000.0])
    assert np.allclose(S.road_accessibility_score(d) / 100.0,
                       [0.0, 0.2, 1.0, 1.0, 0.65, 0.3, 0.3])
    assert np.allclose(S.green_proximity_score(np.array([0, 250, 500, 900])) / 100.0,
                       [1.0, 0.5, 0.0, 0.0])


def test_scenario_constants():
    assert B.SCENARIOS == ("v1_parity", "v2_constrained")
    assert B.DISCOURAGED_LU_CODES == (5, 7)
    assert S.PRIORITY_WEIGHTS == {"cooling_need": 0.5, "suitability": 0.5}
    assert S.PRIORITY_CLASS_LABELS == {0: "Low", 1: "Medium", 2: "High"}


# ---------------------------------------------------------------------------
# artifact contracts (skip if phase7 v3 not built)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def record():
    rec = PROJECT_ROOT / "data" / "phase7" / "phase7_pipeline_record.json"
    if not rec.exists():
        pytest.skip("phase7 v3 not built here")
    import json
    return json.loads(rec.read_text())


def test_raster_contracts_and_priority_on_feasible_only(record):
    import rasterio
    for scenario in record["scenarios"]:
        for y in record["years"]:
            rdir = PROJECT_ROOT / "data" / "phase7" / scenario / "rasters"
            with rasterio.open(rdir / f"priority_class_{scenario}_{y}.tif") as ds:
                cls = ds.read(1)
                assert (ds.width, ds.height) == (1874, 1768)
                assert ds.nodata == 255
            with rasterio.open(rdir / f"feasibility_mask_{scenario}_{y}.tif") as ds:
                feas = ds.read(1)
            assert np.array_equal(cls != 255, feas == 1), \
                "priority class must exist exactly on feasible px"
            with rasterio.open(rdir / f"cooling_need_{scenario}_{y}.tif") as ds:
                cn = ds.read(1)
            valid = cn != -1
            assert cn[valid].min() >= 0.0 and cn[valid].max() <= 1.0


def test_class_shares_and_thresholds(record):
    t1 = record["priority_thresholds"]["t1_medium"]
    t2 = record["priority_thresholds"]["t2_high"]
    assert 0.0 < t1 < t2 < 1.0
    for scenario in record["scenarios"]:
        for y in record["years"]:
            tdir = PROJECT_ROOT / "data" / "phase7" / scenario / "tables"
            sh = pd.read_csv(tdir / f"class_shares_{scenario}_{y}.csv")
            assert set(sh["class"]) == {"Low", "Medium", "High"}
            assert int(sh["pixel_count"].sum()) > 0
            ea = pd.read_csv(tdir / f"exclusion_accounting_{scenario}_{y}.csv")
            assert "FEASIBLE" in set(ea["reason"])


def test_backend_paths_match_disk(record):
    """Every data file the backend serves must exist on disk (v3 names), and
    the removed opportunity endpoints must be gone from the API."""
    main_py = (PROJECT_ROOT / "backend/app/main.py").read_text()
    data = PROJECT_ROOT / "data"
    served = [
        data / "phase8/v2_constrained/vectors/recommended_plantations_v2_constrained_2026.geojson",
        data / "phase8/v2_constrained/tables/tree_requirement_summary_v2_constrained_2026.csv",
        data / "phase8/v2_constrained/tables/tree_requirement_by_zone_v2_constrained_2026.csv",
        data / "phase7/v2_constrained/tables/class_shares_v2_constrained_2026.csv",
        data / "phase7/v2_constrained/tables/exclusion_accounting_v2_constrained_2026.csv",
        data / "phase7/v2_constrained/rasters/priority_class_v2_constrained_2026.tif",
        data / "gis/study_area/study_area.geojson",
    ]
    for p in served:
        assert p.exists(), f"backend-served file missing: {p}"
    assert "phase7/opportunity" not in main_py, "stale opportunity endpoint"
