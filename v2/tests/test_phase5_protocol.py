"""Synthetic-only protocol tests for V2 Phase 5 (UHI 3-class model).

Nothing here trains on real V2/V1 data. Fabricated LST frames exercise the
tertile-threshold target machinery (including a leakage trap), the V1
adjacent-block fold mapping, the frozen-params extraction, the documented
determinism policy, and an end-to-end tiny dry-run of train.py + verify_run.py
on a synthetic 182-column feature set in tmp_path.

Run from the project root:
    PYTHONUTF8=1 PYTHONPATH=v2/src .venv/Scripts/python.exe -m pytest v2/tests -q
"""

from __future__ import annotations

import json
import shutil
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

from v2.common import HOLDOUT_BLOCKS, load_frozen_schema  # noqa: E402
from v2.phase5 import train, verify_run  # noqa: E402

V1_PRODUCTION_DIR = PROJECT_ROOT.parent / "v1" / "data" / "processed" / \
    "phase5_production_3class"

TRAIN_BLOCKS = sorted(set(range(25)) - set(HOLDOUT_BLOCKS))


def _lst_frame(rng: np.random.Generator, years=(2022, 2023, 2024, 2025, 2026),
               per_year=300, blocks=TRAIN_BLOCKS) -> pd.DataFrame:
    """Fabricated frame: row/col/spatial_block_id/lst_C/year only."""
    rows = []
    for year in years:
        for i in range(per_year):
            rows.append({
                "row": i, "col": i, "lst_C": rng.uniform(20, 45),
                "year": float(year),
                "spatial_block_id": blocks[i % len(blocks)],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tertile thresholds (target machinery)
# ---------------------------------------------------------------------------

def test_tercile_thresholds_equal_hand_computed_quantiles():
    rng = np.random.default_rng(7)
    df = _lst_frame(rng)
    # deterministic train mask: first 200 rows of each year
    train_mask = df.groupby("year").cumcount() < 200
    thresholds = train.tercile_thresholds(df, train_mask.to_numpy())
    assert set(thresholds) == {2022, 2023, 2024, 2025, 2026}
    for year in thresholds:
        vals = df.loc[train_mask & (df["year"] == year), "lst_C"]
        hand = vals.quantile([1.0 / 3.0, 2.0 / 3.0]).values
        assert np.allclose(thresholds[year], hand, rtol=0, atol=1e-12)


def test_thresholds_use_training_rows_only_leakage_trap():
    """Drastically shifting VALIDATION LST must not move the thresholds, and
    train-only thresholds must differ from full-data tertiles on a skewed
    fabrication (a leak would make them coincide)."""
    rng = np.random.default_rng(11)
    df = _lst_frame(rng)
    train_mask = df.groupby("year").cumcount() < 200
    t_clean = train.tercile_thresholds(df, train_mask.to_numpy())

    df_shifted = df.copy()
    df_shifted.loc[~train_mask, "lst_C"] += 100.0     # absurd validation shift
    t_shifted = train.tercile_thresholds(df_shifted, train_mask.to_numpy())
    assert t_shifted == t_clean

    t_leaky = train.tercile_thresholds(df, np.ones(len(df), dtype=bool))
    assert any(not np.allclose(t_leaky[y], t_clean[y]) for y in t_clean)


def test_assign_classes_searchsorted_right_convention():
    thresholds = {2022: (30.0, 35.0)}
    lst = np.array([30.0, 30.0001, 35.0, 35.0001, 20.0, 40.0])
    years = np.full(len(lst), 2022.0)
    cls = train.assign_classes(lst, years, thresholds)
    # frozen V1 code path: np.searchsorted([t1, t2], lst, side="right") ->
    # class == #(thresholds <= lst): Low: lst < t1; Moderate: t1 <= lst < t2;
    # High: lst >= t2 (exact ties land in the HIGHER class).
    assert cls.tolist() == [1, 1, 2, 2, 0, 2]


def test_build_target_per_year_independent():
    rng = np.random.default_rng(13)
    df = _lst_frame(rng)
    df.loc[df["year"] == 2023, "lst_C"] += 8.0        # different year, shifted
    train_mask = np.ones(len(df), dtype=bool)
    y, thr = train.build_target(df, train_mask)
    for year in (2022, 2023):
        m = df["year"].to_numpy() == year
        hand = pd.Series(df.loc[m, "lst_C"]).quantile([1 / 3, 2 / 3]).values
        assert np.allclose(thr[year], hand, atol=1e-12)
        assert set(np.unique(y[m])) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Fold mapping (V1 adjacent-block assignment)
# ---------------------------------------------------------------------------

def test_fold_mapping_sanity():
    rng = np.random.default_rng(17)
    # rows across ALL 25 pinned blocks, locked included
    df = _lst_frame(rng, per_year=50, blocks=list(range(25)))
    folds = train.adjacent_block_folds(df)
    assert len(folds) == 5
    assert len(TRAIN_BLOCKS) == 21          # 25 pinned - 4 locked
    val_union, seen = set(), set()
    for f in folds:
        assert set(f["train_blocks"]).isdisjoint(f["val_blocks"])
        assert not (set(f["val_blocks"]) & set(HOLDOUT_BLOCKS))
        assert set(f["train_blocks"]) == set(TRAIN_BLOCKS) - set(f["val_blocks"])
        val_union |= set(f["val_blocks"])
        seen |= set(f["val_blocks"]) | set(f["train_blocks"])
    assert val_union == set(TRAIN_BLOCKS)   # union = all non-locked blocks
    assert seen == set(TRAIN_BLOCKS)        # locked never appear
    # each non-locked block is validation exactly once (GroupKFold property)
    assert sum(len(f["val_blocks"]) for f in folds) == len(TRAIN_BLOCKS)


def test_fold_mapping_deterministic():
    rng = np.random.default_rng(19)
    df = _lst_frame(rng, per_year=50, blocks=list(range(25)))
    m1 = [{k: f[k] for k in ("train_blocks", "val_blocks")}
          for f in train.adjacent_block_folds(df)]
    m2 = [{k: f[k] for k in ("train_blocks", "val_blocks")}
          for f in train.adjacent_block_folds(df)]
    assert m1 == m2


# ---------------------------------------------------------------------------
# Frozen params + determinism policy (documentation, not a runtime fit)
# ---------------------------------------------------------------------------

def test_frozen_params_match_v1_production_metadata():
    meta_path = V1_PRODUCTION_DIR / "production_model_metadata.json"
    if not meta_path.exists():
        pytest.skip("V1 production metadata not present")
    meta = json.loads(meta_path.read_text())
    v1 = meta["model_params"]
    for k, v in train.XGB_PARAMS_V1.items():
        if k in ("num_class", "n_jobs"):
            continue                                  # added/pinned for the task
        assert v1[k] == v, f"param drift on {k}: v1={v1[k]!r} v2={v!r}"
    assert train.XGB_PARAMS_V1["num_class"] == 3
    # booster JSON cross-check: 600 trees = 200 estimators x 3 classes
    booster_path = V1_PRODUCTION_DIR / "phase5_primary_xgb_3class.json"
    if booster_path.exists():
        booster = json.loads(booster_path.read_text())
        n_trees = int(booster["learner"]["gradient_booster"]["model"]
                      ["gbtree_model_param"]["num_trees"])
        assert n_trees == v1["n_estimators"] * 3
    # V1 set no regularisation keys -> XGBoost defaults are frozen
    for k in ("min_child_weight", "reg_alpha", "reg_lambda"):
        assert k not in train.XGB_PARAMS_V1


def test_determinism_policy_documented():
    """Not a runtime fit: the single-threaded final-refit policy and the
    exact-once locked evaluation must be documented in the module."""
    assert train.FINAL_REFIT_NJOBS == 1
    doc = (train.__doc__ or "").lower()
    assert "bit-reproducib" in doc or "bit-reproducible" in doc
    assert "exactly once" in doc
    assert "n_jobs=1" in doc


# ---------------------------------------------------------------------------
# End-to-end dry-run on a synthetic 182-column feature set (tmp_path)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_features(tmp_path_factory):
    rng = np.random.default_rng(23)
    schema = load_frozen_schema()
    base = tmp_path_factory.mktemp("phase5_smoke")
    feat_dir = base / "features"
    feat_dir.mkdir()
    n_blocks, per = 25, 20                       # 20 rows/block -> 500/year
    for year in range(2022, 2027):
        rec = {
            # globally unique (row, col) so (row, col, year) is a true key —
            # mirrors the real fullgrid sample where pixels are unique
            "row": np.arange(n_blocks * per, dtype=np.int32),
            "col": (np.arange(n_blocks * per) % 97).astype(np.int32),
            "spatial_block_id": np.repeat(np.arange(n_blocks), per),
            "lst_C": rng.uniform(20, 45, n_blocks * per).astype(np.float32),
        }
        for name in schema:
            rec[name] = rng.normal(size=n_blocks * per).astype(np.float32)
            if name == "year":
                rec[name] = np.full(n_blocks * per, float(year), np.float32)
        df = pd.DataFrame(rec)[["row", "col", "spatial_block_id", "lst_C"]
                               + schema]
        df.to_parquet(feat_dir / f"features_{year}.parquet", index=False)
    return feat_dir


@pytest.fixture(scope="module")
def trained_out(synthetic_features, tmp_path_factory):
    base = tmp_path_factory.mktemp("phase5_smoke_out")
    out = base / "phase5"
    assert train.main(["--features-dir", str(synthetic_features),
                       "--out", str(out)]) == 0
    return out


def test_end_to_end_dry_run(synthetic_features, trained_out):
    out = trained_out
    for name in ("phase5_primary_xgb_v2.json", "cv_metrics.csv",
                 "cv_confusion.csv", "oof_predictions.csv", "lobo_metrics.csv",
                 "locked_metrics.json", "locked_confusion.csv",
                 "per_class_locked.csv", "thresholds_by_year.json",
                 "phase5_manifest.json", "locked_predictions.csv"):
        assert (out / name).exists(), f"missing artifact {name}"
    thr = json.loads((out / "thresholds_by_year.json").read_text())
    assert all(v["t1"] < v["t2"] for v in thr.values())
    locked = json.loads((out / "locked_metrics.json").read_text())
    assert locked["locked_blocks"] == list(HOLDOUT_BLOCKS)
    assert locked["n_locked"] > 0
    lobo = pd.read_csv(out / "lobo_metrics.csv")
    assert len(lobo) == 25 and (lobo["n_test"] > 0).all()

    rc = verify_run.main(["--features-dir", str(synthetic_features),
                          "--data-dir", str(out)])
    assert rc == 0


def test_lobo_records_empty_blocks_without_fits(monkeypatch, tmp_path):
    """Blocks absent from the sample are recorded with n_test=0 and empty
    metrics (the real V2 sample leaves 5 of 25 pinned blocks empty)."""
    monkeypatch.setattr(train, "XGB_PARAMS_V1",
                        {**train.XGB_PARAMS_V1, "n_estimators": 1})
    rng = np.random.default_rng(29)
    df = _lst_frame(rng, years=(2022,), per_year=40, blocks=[0, 1, 3])
    X = pd.DataFrame({"f": rng.normal(size=len(df))})
    res = train.run_lobo(df, X, tmp_path)
    by_block = {r["block"]: r for r in res["per_block"]}
    assert len(by_block) == 25
    for b in (0, 1, 3):
        assert not by_block[b]["empty"] and by_block[b]["n_test"] > 0
    empty = [r["block"] for r in res["per_block"] if r["empty"]]
    assert len(empty) == 22 and all(by_block[b]["n_test"] == 0 for b in empty)
    csv = pd.read_csv(tmp_path / "lobo_metrics.csv")
    assert len(csv) == 25
    assert int((csv["n_test"] == 0).sum()) == 22
    assert int(csv["accuracy"].isna().sum()) == 22
    assert int(csv["macro_f1"].isna().sum()) == 22


def test_verify_run_fails_on_tampered_predictions(synthetic_features,
                                                  trained_out, tmp_path):
    out = tmp_path / "tampered"
    shutil.copytree(trained_out, out)
    pred_path = out / "locked_predictions.csv"
    pred = pd.read_csv(pred_path)
    pred.loc[pred.index[0], "y_pred"] = (int(pred["y_pred"].iloc[0]) + 1) % 3
    pred.to_csv(pred_path, index=False)
    rc = verify_run.main(["--features-dir", str(synthetic_features),
                          "--data-dir", str(out)])
    assert rc == 1
