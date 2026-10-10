"""Tests for the FINAL 2026-only tree-planting recommendation module (phase 8)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from v2.phase7.suitability import label_zones  # noqa: E402
from v2.phase8 import build as B  # noqa: E402


def test_floor_arithmetic_and_labels():
    assert B.HA_PER_PX == pytest.approx(0.09)
    assert "not optimal" in B.DENSITY_LABEL and "AI-predicted" in B.DENSITY_LABEL
    for ha, d in ((8.19, 400), (72.54, 1000), (3.87, 2500), (11.1, 1750)):
        assert int(ha * d) == np.floor(ha * d)


def test_mmu_and_max_size_rule():
    mask = np.zeros((30, 40), dtype=bool)
    mask[2:10, 2:12] = True
    mask[20, 20] = True
    labelled, n = label_zones(mask, min_pixels=23)
    assert n == 1
    sizes = pd.Series(labelled[labelled > 0]).value_counts() * 0.09
    assert (sizes > 2).all() and (sizes <= 57).all()


@pytest.fixture(scope="module")
def record():
    rec = PROJECT_ROOT / "data" / "phase8" / "phase8_pipeline_record.json"
    if not rec.exists():
        pytest.skip("phase8 not built")
    import json
    return json.loads(rec.read_text())


def test_contract_and_2026_only_outputs(record):
    out = PROJECT_ROOT / "data" / "phase8"
    z = pd.read_csv(out / "v2_constrained" / "tables"
                    / "tree_requirement_by_zone_v2_constrained_2026.csv")
    assert {"zone_id", "plantable_ha", "usable_area_ha", "planning_priority_score",
            "shortlist", "heat_need_comp", "vegetation_deficit_comp",
            "opportunity_comp", "rationale", "landuse_certainty"} <= set(z.columns)
    assert (z["usable_area_ha"] == (z["zone_px"] * 0.09).round(3)).all()
    assert record["years"] == [2026]
    # 2026-only: no per-year columns, no capacity artifacts
    assert not any(c.startswith("plantable_px_20") for c in z.columns)
    assert not (out / "v2_constrained" / "tables" / "theoretical_capacity_v2_constrained_2026.csv").exists()


def test_shortlist_top100_rule(record):
    import json
    p7 = json.loads((PROJECT_ROOT / "data" / "phase7"
                     / "phase7_pipeline_record.json").read_text())
    c = p7["sites_2026"]["constants"]
    assert c["year"] == 2026 and c["shortlist_top_n"] == 100 and c["mmu_min_px"] == 23
    z = pd.read_csv(PROJECT_ROOT / "data" / "phase7" / "v2_constrained" / "zones"
                    / "stable_zones_v2_constrained.csv")
    zs = z.sort_values("rank").reset_index(drop=True)
    exp = ((~zs["oversized"]).cumsum() <= 100) & (~zs["oversized"])
    assert (zs["shortlist"].to_numpy() == exp.to_numpy()).all()
    n_short = int(zs["shortlist"].sum())
    ha = float(zs.loc[zs["shortlist"], "usable_area_ha"].sum())
    assert 800 <= ha <= 1200, f"shortlist band violated: {n_short} sites {ha} ha"
    assert "0.50*need(" in zs.iloc[0]["rationale"]


def test_floor_reference_columns(record):
    out = PROJECT_ROOT / "data" / "phase8"
    z = pd.read_csv(out / "v2_constrained" / "tables"
                    / "tree_requirement_by_zone_v2_constrained_2026.csv")
    for d, col in ((400, "reference_trees_400"), (1000, "reference_trees_1000"),
                   (2500, "reference_trees_2500")):
        assert (z[col] == (z["usable_area_ha"] * d).astype(int)).all()
    short = z[z["shortlist"]]
    # per-site floor is exact; totals are sums of per-site floors (not floor of the sum)
    assert (short["reference_trees_1000"] == (short["usable_area_ha"] * 1000).astype(int)).all()


def test_phase9_contract_files(record):
    import rasterio
    out = PROJECT_ROOT / "data" / "phase8"
    for p in [out / "v2_constrained" / "tables" / "tree_requirement_by_zone_v2_constrained_2026.csv",
              out / "v2_constrained" / "rasters" / "available_planting_space_v2_constrained_2026.tif",
              PROJECT_ROOT / "data" / "phase7" / "v2_constrained" / "rasters"
              / "priority_zone_ids_v2_constrained_2026.tif"]:
        assert p.exists(), p
    with rasterio.open(out / "v2_constrained" / "rasters"
                       / "priority_zone_ids_v2_constrained_2026.tif") as ds:
        a = ds.read(1)
    with rasterio.open(PROJECT_ROOT / "data" / "phase7" / "v2_constrained" / "rasters"
                       / "priority_zone_ids_v2_constrained_2026.tif") as ds:
        b = ds.read(1)
    assert (a == b).all()


def test_zone_table_json_serializable():
    """The API serves df.to_dict(records) through FastAPI's JSONResponse
    (allow_nan=False): any NaN in the contract CSV 500s every frontend page
    that fetches it. Enforce serializability of every emitted zone CSV."""
    import math
    from fastapi.encoders import jsonable_encoder
    out = PROJECT_ROOT / "data" / "phase8"
    for scen in ("v1_parity", "v2_constrained"):
        p = out / scen / "tables" / f"tree_requirement_by_zone_{scen}_2026.csv"
        if not p.exists():
            continue
        recs = pd.read_csv(p).to_dict(orient="records")
        enc = jsonable_encoder(recs)

        def _walk(o):
            if isinstance(o, float):
                assert math.isfinite(o), "non-finite float in zone table"
            elif isinstance(o, dict):
                for v in o.values():
                    _walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    _walk(v)

        _walk(enc)
