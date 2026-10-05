"""Synthetic unit tests for V2 Phase 6 (UHI severity mapping).

Fast fixtures only: marker resolution, schema binding of the mapping
feature stack, NoData scatter semantics, and probability normalization with
a stub model. No real rasters, no real joblib.

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # the v2/ tree
SRC = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from v2.common import load_frozen_schema, schema_hash  # noqa: E402
from v2.phase4.assemble_features import predictor_col_order  # noqa: E402
from v2.phase6 import build  # noqa: E402


# ---------------------------------------------------------------------------
# marker-driven model resolution
# ---------------------------------------------------------------------------

def _write_marker(tmp_path, **over):
    marker = {
        "primary_model": "rf", "frozen": True,
        "locked_accuracy": 0.6990177602649763,
        "locked_macro_f1": 0.6851219134292812,
        "artifact": "model.joblib",
        "baseline_xgb": "phase5_primary_xgb_v2.json",
        "verification_passed": True,
    }
    marker.update(over)
    (tmp_path / "phase5_primary_model.json").write_text(json.dumps(marker))
    (tmp_path / "model.joblib").write_text("stub")
    return tmp_path


def test_marker_resolution_ok(tmp_path):
    d = _write_marker(tmp_path)
    marker, artifact = build.resolve_primary_model(d)
    assert marker["primary_model"] == "rf" and artifact == d / "model.joblib"


@pytest.mark.parametrize("over", [
    {"primary_model": "xgb"},
    {"frozen": False},
    {"verification_passed": False},
])
def test_marker_resolution_rejects(tmp_path, over):
    d = _write_marker(tmp_path, **over)
    with pytest.raises(AssertionError):
        build.resolve_primary_model(d)


def test_marker_missing_artifact(tmp_path):
    d = _write_marker(tmp_path)
    (d / "model.joblib").unlink()
    with pytest.raises(FileNotFoundError):
        build.resolve_primary_model(d)


# ---------------------------------------------------------------------------
# schema binding: mapping feature stack == Phase 4 frozen schema
# ---------------------------------------------------------------------------

def _synthetic_domain_frame(n=60, seed=5):
    rng = np.random.default_rng(seed)
    cols = predictor_col_order()
    # categorical values restricted to the frozen-schema dummy sets
    landuse_vals = np.array([0, 2, 4, 5, 6, 7, 8], dtype=float)
    data = {}
    for name in cols:
        if name == "year":
            data[name] = np.full(n, 2022.0)
        elif name in ("landuse_class",) or name.startswith("landuse_dominant_"):
            data[name] = landuse_vals[rng.integers(0, len(landuse_vals), n)]
        elif name == "lulc_class":
            data[name] = rng.integers(0, 9, n).astype(float)
        else:
            data[name] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_encode_domain_frame_matches_frozen_schema():
    schema = load_frozen_schema()
    X, n_extra = build.encode_domain_frame(_synthetic_domain_frame(),
                                           predictor_col_order(), schema)
    assert n_extra == 0
    assert list(X.columns) == list(schema), "mapping stack != frozen 178"
    assert schema_hash(list(X.columns)) == schema_hash(schema)
    non_dist = [c for c in schema if not c.startswith("dist_")]
    assert all(X[c].dtype == np.float32 for c in non_dist)


def test_encode_domain_frame_zero_fills_unseen_codes():
    """Unseen categorical values (landuse 1/3 exist in the V2 OSM raster per
    CONTRACT B2) must zero-fill, not crash — the mapping keeps running and
    the stack still matches the frozen schema exactly."""
    schema = load_frozen_schema()
    df = _synthetic_domain_frame()
    df.loc[:4, "landuse_class"] = 1.0                  # unseen in frozen schema
    X, n_extra = build.encode_domain_frame(df, predictor_col_order(), schema)
    assert n_extra > 0
    assert list(X.columns) == list(schema)
    assert schema_hash(list(X.columns)) == schema_hash(schema)
    # the unseen-code rows have every landuse one-hot zero-filled
    lu_cols = [c for c in schema if c.startswith("landuse_class_")]
    assert (X.loc[:4, lu_cols].to_numpy() == 0).all()


# ---------------------------------------------------------------------------
# NoData semantics + probability math (stub model)
# ---------------------------------------------------------------------------

class StubModel:
    """predict_proba cycling fixed near-simplex rows."""

    def predict_proba(self, X):
        n = len(X)
        base = np.tile(np.array([[0.5, 0.3, 0.2]]), (n, 1))
        tweak = (np.arange(n) % 3)[:, None] * np.array([[0.0, 0.0, 0.0]])
        return base + tweak


def test_scatter_nodata_semantics():
    shape = (4, 5)
    rows = np.array([0, 1, 3])
    cols = np.array([0, 2, 4])
    vals = np.array([2, 1, 0], dtype=np.uint8)
    grid = build.scatter(shape, rows, cols, vals, np.uint8,
                         build.SEVERITY_NODATA)
    assert grid.dtype == np.uint8
    assert grid[rows, cols].tolist() == [2, 1, 0]
    assert int(grid[2, :].sum()) == 0 + 255 * 5          # row 2 all NoData
    assert grid[0, 1] == 255 and grid[1, 1] == 255
    valid = grid != build.SEVERITY_NODATA
    assert int(valid.sum()) == 3


def test_predict_year_probabilities_normalized():
    X = pd.DataFrame({"a": np.ones(7)})
    cls, probs = build.predict_year(StubModel(), X)
    assert probs.shape == (7, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    assert set(np.unique(cls)) <= {0, 1, 2}
    assert np.array_equal(cls, np.argmax(probs, axis=1))


def test_hotspot_mask_tiers():
    cls = np.array([[2, 2, 1], [2, 0, 2], [1, 2, 2]], dtype=np.uint8)
    conf = np.array([[0.9, 0.5, 0.7],
                     [0.8, 0.9, 0.4],
                     [0.6, 0.95, 0.7]])
    score = np.array([[1.8, 1.2, 0.7],
                      [1.9, 0.1, 1.4],
                      [1.0, 1.7, 1.8]])
    a = build.hotspot_mask("A", cls, conf, score)
    b = build.hotspot_mask("B", cls, conf, score)
    c = build.hotspot_mask("C", cls, conf, score)
    assert a.sum() == 6                                # all High
    assert b.sum() == 4                                # High & conf>=0.6
    assert c.sum() == 4                                # score>=1.5 & conf>=0.6
    assert build.MIN_CLUSTER_PX == 10 and build.CONFIDENCE_T == 0.60 \
        and build.SCORE_T == 1.5                         # V1 spec values
