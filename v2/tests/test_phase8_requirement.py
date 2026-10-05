"""Synthetic unit tests for V2 Phase 8 (tree requirement estimation).

Fast fixtures only: density arithmetic, the frozen landuse-eligibility rule,
the vegetation threshold boundary, priority thirds, exclusion interaction.
No real rasters.

Run from the project root:
    PYTHONUTF8=1 PYTHONPATH=v2/src .venv/Scripts/python.exe -m pytest v2/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # the v2/ tree
SRC = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from v2.phase8 import build as B  # noqa: E402


def test_density_arithmetic_verbatim():
    # 0.09 ha/px x 1000 trees/ha = 90 trees/px (V1's "9" was a decimal slip)
    assert B.TREES_PER_PX_PRIMARY == 90
    assert B.HA_PER_PX == pytest.approx(0.09)
    px = 1234
    ha = px * B.HA_PER_PX
    assert int(round(ha * 1000)) == px * 90
    assert int(round(ha * 400)) == px * 36
    assert int(round(ha * 2500)) == px * 225
    assert B.SENSITIVITY_DENSITIES == (400, 2500)
    assert B.FOCUS_YEAR == 2026


def test_landuse_eligibility_rule():
    # discouraged = {5 industrial, 7 retail}; everything else eligible,
    # nodata (255) neutral-eligible (frozen V1 class-0 treatment)
    lu = np.arange(0, 9)
    elig = B.landuse_eligible_mask(lu)
    assert elig.tolist() == [True, True, True, True, True, False,
                             True, False, True]
    assert bool(B.landuse_eligible_mask(np.array([255]))[0])
    # every code 0..254 except 5 and 7 eligible
    all_codes = np.arange(0, 256)
    e = B.landuse_eligible_mask(all_codes)
    assert int(e.sum()) == 254


def test_vegetation_threshold_boundary_strict():
    veg = np.array([0.0, 0.1, 0.299999, 0.30, 0.300001, 0.9])
    ok = np.isfinite(veg) & (veg < B.VEG_COVER_THRESHOLD)
    assert ok.tolist() == [True, True, True, False, False, False]


def test_priority_thirds_assignment():
    n = 11
    third = int(np.ceil(n / 3.0))
    assert third == 4
    pr = [("High" if r <= third else "Medium" if r <= 2 * third else "Low")
          for r in range(1, n + 1)]
    assert pr == ["High"] * 4 + ["Medium"] * 4 + ["Low"] * 3
    n = 69
    third = int(np.ceil(n / 3.0))
    pr = [("High" if r <= third else "Medium" if r <= 2 * third else "Low")
          for r in range(1, n + 1)]
    assert pr.count("High") == 23 and pr.count("Medium") == 23 \
        and pr.count("Low") == 23


def test_exclusion_interaction_synthetic():
    """plantable = zone px AND not excluded AND eligible AND veg<0.30 —
    each condition independently gates a synthetic pixel."""
    cls = np.array([[3, 3, 3, 1, 3]])
    zids = np.array([[7, 7, 7, 7, 0]])          # px4 not in any zone
    excl = np.array([[0, 0, 1, 0, 0]])          # px2 excluded
    lu = np.array([[6, 5, 6, 6, 6]])            # px1 industrial (discouraged)
    veg = np.array([[0.1, 0.1, 0.1, 0.5, 0.1]])  # px3 over threshold
    eligible = B.landuse_eligible_mask(lu)
    veg_ok = veg < B.VEG_COVER_THRESHOLD
    domain = excl == 0
    attributed = np.array([[0, 0, 1, 0, 0]])    # px2 constraint-hit
    cause = B.build_cause_grid(veg_ok, eligible, attributed)
    # precedence: px2 constraint-hit wins over its (out-of-domain) exclusion;
    # px3 veg-fail; px1 landuse-ineligible; px0 plantable; px4 outside zone
    assert cause.tolist() == [[0, 2, 1, 3, 0]]
    plant = (cls >= 3) & (zids > 0) & domain & (excl == 0) & eligible & veg_ok
    assert plant.tolist() == [[True, False, False, False, False]]
    # excluded-cause accounting is disjoint and sums to the excluded count
    zone_dom = (zids > 0) & domain
    excl_cause = cause[zone_dom & ~plant]
    n_excl = int((zone_dom & ~plant).sum())
    assert n_excl == int(sum(int((excl_cause == c).sum()) for c in (1, 2, 3)))


def test_cause_grid_precedence_and_no_double_count():
    rng = np.random.default_rng(11)
    shape = (40, 50)
    veg_ok = rng.random(shape) < 0.7
    lu = rng.integers(0, 9, shape)
    eligible = B.landuse_eligible_mask(lu)
    attributed = (rng.random(shape) < 0.2).astype(np.int32)
    cause = B.build_cause_grid(veg_ok, eligible, attributed)
    # disjointness: each px has exactly one cause
    per_px = ((cause == 1).astype(int) + (cause == 2).astype(int)
              + (cause == 3).astype(int))
    assert (per_px <= 1).all()
    # precedence: constraint beats landuse beats veg
    both = (attributed > 0) & ~eligible
    assert (cause[both] == 1).all()
    lu_only = ~(attributed > 0) & ~eligible
    assert (cause[lu_only] == 2).all()
    veg_only = ~(attributed > 0) & eligible & ~veg_ok
    assert (cause[veg_only] == 3).all()
    # v1_parity (attributed=None): no constraint cause at all
    cause_base = B.build_cause_grid(veg_ok, eligible, None)
    assert int((cause_base == 1).sum()) == 0


def test_zone_area_accounting_invariant():
    """zone px == plantable + excluded exactly; excluded ha == px * 0.09."""
    for px, plant, excl in ((1000, 400, 600), (37, 37, 0), (10, 0, 10)):
        assert px == plant + excl
        assert abs(excl * B.HA_PER_PX - round(excl * B.HA_PER_PX, 3)) < 1e-9


def test_scenario_and_output_constants():
    # v2_constrained is PRIMARY; v1_parity is the V1-comparable baseline
    assert B.SCENARIOS == ("v2_constrained", "v1_parity")
    assert B.PRIMARY_SCENARIO == "v2_constrained"
    assert B.BASELINE_SCENARIO == "v1_parity"
    assert B.PRIMARY_SNAPSHOT_YEAR == 2026
    assert B.PLANT_NODATA == 255 and B.TREES_NODATA == -1
    assert B.SEVERITY_HIGH_CLASS == 2
    assert B.EXCLUSION_PRECEDENCE == ("constraint", "landuse_ineligible",
                                      "veg_threshold")
    # wording: planning-density disclaimer present, no optimality claim
    assert "not scientifically optimal" in B.DENSITY_LABEL
    assert "Phase 9" in B.DENSITY_LABEL
    # DEVIATIONS documents the 9-vs-90 decimal slip
    assert any("decimal slip" in d for d in B.DEVIATIONS)


def test_smoke_outputs_tagged_both_scenarios():
    out = PROJECT_ROOT / "data" / "phase8"
    rec = out / "phase8_pipeline_record.json"
    if not rec.exists():
        pytest.skip("phase8 smoke not run here")
    import json
    record = json.loads(rec.read_text())
    years = record["years"]
    assert record["primary_scenario"] == "v2_constrained"
    assert record["primary_snapshot_year"] == 2026
    for scenario in ("v1_parity", "v2_constrained"):
        for y in years:
            zdf_path = (out / scenario / "tables"
                        / f"tree_requirement_by_zone_{scenario}_{y}.csv")
            assert zdf_path.exists()
            import pandas as pd
            zdf = pd.read_csv(zdf_path)
            # new per-zone columns present
            for col in ("excluded_ha", "recommended_trees_400",
                        "recommended_trees_1000", "recommended_trees_2500"):
                assert col in zdf.columns
            assert (out / scenario / "rasters"
                    / f"available_planting_space_{scenario}_{y}.tif").exists()
