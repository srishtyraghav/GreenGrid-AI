"""Synthetic-only plumbing tests for the Phase-5 RF comparison module.

The RF experiment must reuse the frozen XGB protocol machinery (fold
mapping, train-only tertile thresholds, schema/banned asserts). Tiny
fixtures + a 2-tree RF keep this under ~30 s. No real data is trained.

Run from the project root:
    PYTHONUTF8=1 PYTHONPATH=v2/src .venv/Scripts/python.exe -m pytest v2/tests -q
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # the v2/ tree
SRC = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from v2.common import HOLDOUT_BLOCKS, load_frozen_schema  # noqa: E402
from v2.phase5 import rf_compare, train  # noqa: E402


def _tiny_frame(rng, years=(2022, 2023), blocks=(1, 3, 6, 7), per=8):
    rows = []
    for year in years:
        for i in range(per * len(blocks)):
            rows.append({"row": i, "col": i,
                         "lst_C": rng.uniform(20, 45),
                         "year": float(year),
                         "spatial_block_id": blocks[i % len(blocks)]})
    return pd.DataFrame(rows)


def test_rf_params_verbatim_v1():
    """Pre-specified verbatim from v1/src/models/config.py::
    RANDOM_FOREST_PARAMS (RANDOM_SEED=42, v1/src/features/config.py)."""
    assert rf_compare.RF_PARAMS_V1 == {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }


def test_rf_reuses_xgb_fold_mapping():
    rng = np.random.default_rng(3)
    df = _tiny_frame(rng, blocks=tuple(range(25)))
    rf_folds = rf_compare.adjacent_block_folds(df)
    xgb_folds = train.adjacent_block_folds(df)     # same function object
    assert [{k: f[k] for k in ("fold", "train_blocks", "val_blocks")}
            for f in rf_folds] == [
        {k: f[k] for k in ("fold", "train_blocks", "val_blocks")}
        for f in xgb_folds]
    assert len(rf_folds) == 5
    assert all(not (set(f["val_blocks"]) & set(HOLDOUT_BLOCKS)) for f in rf_folds)


def test_rf_thresholds_train_only():
    rng = np.random.default_rng(5)
    df = _tiny_frame(rng)
    train_mask = df.groupby("year").cumcount() < 24
    t_clean = rf_compare.tercile_thresholds(df, train_mask.to_numpy())
    df_shifted = df.copy()
    df_shifted.loc[~train_mask, "lst_C"] += 100.0
    assert rf_compare.tercile_thresholds(df_shifted,
                                         train_mask.to_numpy()) == t_clean
    y, thr = rf_compare.build_target(df, train_mask.to_numpy())
    assert set(np.unique(y)) == {0, 1, 2}
    assert thr == t_clean


def test_banned_columns_and_bad_frames_rejected(tmp_path):
    """The frozen schema contains no banned predictor, and any frame whose
    predictor block deviates from the frozen 178 (e.g. a smuggled extra
    column replacing a schema slot) fails the loud load gate. (Parquet itself
    also refuses duplicate column names, so a banned-name duplicate cannot
    even be written.)"""
    schema = load_frozen_schema()
    assert not train.BANNED_PREDICTOR_COLS.intersection(schema)
    rng = np.random.default_rng(7)
    n = 40
    rec = {"row": np.arange(n), "col": np.arange(n) % 13,
           "spatial_block_id": np.repeat([1, 3, 6, 7, 2, 9, 15, 23, 10, 11],
                                         n // 10),
           "lst_C": rng.uniform(20, 45, n).astype(np.float32)}
    for name in schema[:-1]:                       # last schema slot replaced
        rec[name] = rng.normal(size=n).astype(np.float32)
    rec["smuggled"] = rng.normal(size=n).astype(np.float32)
    df = pd.DataFrame(rec)[["row", "col", "spatial_block_id", "lst_C"]
                           + schema[:-1] + ["smuggled"]]
    assert len(df.columns) == 182 and df.columns.is_unique
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    for year in range(2022, 2027):
        df.assign(year=float(year)).to_parquet(
            feat_dir / f"features_{year}.parquet", index=False)
    with pytest.raises(AssertionError):
        rf_compare.load_features(feat_dir, schema)


def test_smoke_runs_one_cv_fold_and_one_lobo(monkeypatch, tmp_path):
    monkeypatch.setattr(rf_compare, "RF_PARAMS_V1",
                        {**rf_compare.RF_PARAMS_V1, "n_estimators": 2})
    schema = load_frozen_schema()
    rng = np.random.default_rng(11)
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    n_blocks, per = 25, 4
    n = n_blocks * per
    for year in range(2022, 2027):
        rec = {"row": np.arange(n), "col": np.arange(n) % 17,
               "spatial_block_id": np.repeat(np.arange(n_blocks), per),
               "lst_C": rng.uniform(20, 45, n).astype(np.float32)}
        for name in schema:
            rec[name] = rng.normal(size=n).astype(np.float32)
            if name == "year":
                rec[name] = np.full(n, float(year), np.float32)
        pd.DataFrame(rec)[["row", "col", "spatial_block_id", "lst_C"] + schema] \
            .to_parquet(feat_dir / f"features_{year}.parquet", index=False)
    out = tmp_path / "out"
    t0 = time.perf_counter()
    rc = rf_compare.main(["--features-dir", str(feat_dir), "--out", str(out),
                          "--smoke"])
    elapsed = time.perf_counter() - t0
    assert rc == 0
    assert elapsed < 30
    assert not any(out.iterdir()), "smoke must not write artifacts"
