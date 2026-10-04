"""Uniform geographic-validation evaluator for model experiments.

Every experiment (baseline 4-class, Tier 1+2, Tier 4 regression/binary) is
evaluated with the SAME protocol so numbers are directly comparable:

  1. Adjacent-block spatial CV  — GroupKFold(5) on spatial_block_id
     (skippable with --skip-cv when the pipeline already produced the same
     fold metrics under the identical protocol)
  2. LOBO                        — leave-one-block-out over occupied blocks,
     run in parallel across --workers processes (16 GB RAM: use 2)
  3. Locked geographic holdout   — deterministic blocks occupied[2::6]
                                   (the same rule as the Phase 5 audit),
                                   evaluated EXACTLY ONCE per experiment

Discipline:
  - Targets are built from TRAINING rows only (per-year thresholds/medians
    computed on train, applied to validation) — no target leakage.
  - The locked holdout is never used for tuning; this script reports it and
    nothing here iterates on it.
  - Tier 4 tasks (regression, binary) are DIFFERENT prediction targets/grains
    than the primary 4-class pixel task; their numbers must not be presented
    as improvements of the primary model.
  - The 3class task is a DIFFERENT target formulation (per-year LST tertiles
    from training folds only -> Low/Moderate/High); its accuracy is not
    directly comparable to the 4-class primary task.
  - The reg3class task keeps the 3class target for scoring but trains a
    continuous LST regressor and tertile-thresholds its predictions
    (training-fold thresholds only); it reports MAE/RMSE/R2 alongside.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/geographic_holdout_eval.py \
      --dataset <combined_dataset.csv> \
      --spatial-cache <spatial_neighbourhood_features.csv> \
      --morph-cache <morphology_features.csv> \
      --label <experiment-name> --task {classification,regression,binary} \
      [--skip-cv] [--workers 2] [--out-dir data/processed/experiments]
"""

from __future__ import annotations

import os

# Prevent BLAS/thread oversubscription: each joblib worker runs single-threaded
# math, which is markedly faster for many-core RF/XGB fitting than 16 procs x N
# threads fighting each other. Must be set before numpy/sklearn import.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.config import (  # noqa: E402
    GROUP_VAR,
    MORPHOLOGY_FEATURE_COLS,
    PREDICTOR_VARS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    YEAR_VAR,
)
from models.dataset import encode_predictors  # noqa: E402

LOCKED_BLOCK_RULE = "occupied_block_ids[2::6]"

# ---------------------------------------------------------------------------
# Parallel LOBO workers (module-level globals shared via initializer)
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _lobo_worker_init(X, df_full, task, n_classes):
    _WORKER["X"] = X
    _WORKER["df"] = df_full
    _WORKER["task"] = task
    _WORKER["n_classes"] = n_classes


def _lobo_one(job):
    bid, model_name = job
    X, df, task = _WORKER["X"], _WORKER["df"], _WORKER["task"]
    n_classes = _WORKER["n_classes"]
    groups = df[GROUP_VAR].astype(int).values
    test_mask = groups == bid
    train_mask = ~test_mask
    y = build_y(df, task, train_mask=train_mask)
    model = make_models(task, n_classes)[model_name]()
    if hasattr(model, "n_jobs"):
        # 4 threads per fit x 4 workers = 16 = core count (16 GB RAM machine).
        model.set_params(n_jobs=4)
    model.fit(X[train_mask], y[train_mask])
    preds = model.predict(X[test_mask])
    if task == "reg3class":
        # regression -> threshold predictions with THIS leave-out's train tertiles
        thr = tercile_thresholds(df, train_mask)
        years_test = df[YEAR_VAR].values[test_mask]
        return bid, model_name, score_reg3class(y[test_mask].values, preds, years_test, thr)
    return bid, model_name, score(task, y[test_mask].values, preds)


# ---------------------------------------------------------------------------
# Parallel CV workers (module-level globals shared via initializer)
# ---------------------------------------------------------------------------
_CV_WORKER: dict = {}


def _cv_worker_init(X, df_full, task, n_classes, fold_splits):
    _CV_WORKER["X"] = X
    _CV_WORKER["df"] = df_full
    _CV_WORKER["task"] = task
    _CV_WORKER["n_classes"] = n_classes
    _CV_WORKER["splits"] = fold_splits


def _cv_one(job):
    fold_idx, model_name = job
    X, df, task = _CV_WORKER["X"], _CV_WORKER["df"], _CV_WORKER["task"]
    train_idx, val_idx = _CV_WORKER["splits"][fold_idx]
    mask = np.zeros(len(df), dtype=bool)
    mask[train_idx] = True
    y = build_y(df, task, train_mask=mask)
    model = make_models(task, _CV_WORKER["n_classes"])[model_name]()
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=4)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[val_idx])
    if task == "reg3class":
        thr = tercile_thresholds(df, mask)
        years_val = df[YEAR_VAR].values[val_idx]
        return fold_idx, model_name, score_reg3class(y.iloc[val_idx].values, preds, years_val, thr)
    return fold_idx, model_name, score(task, y.iloc[val_idx].values, preds)


def load_and_merge(dataset_csv: str, spatial_csv: str | None, morph_csv: str | None,
                   extra_csv: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if spatial_csv and Path(spatial_csv).exists():
        df = df.merge(pd.read_csv(spatial_csv), on=["row", "col", "year"], how="left")
    if morph_csv and Path(morph_csv).exists():
        df = df.merge(pd.read_csv(morph_csv), on=["row", "col", "year"], how="left")
    if extra_csv and Path(extra_csv).exists():
        df = df.merge(pd.read_csv(extra_csv), on=["row", "col", "year"], how="left")
    return df


def select_predictors(df: pd.DataFrame, predictor_cols: list[str] | None) -> list[str]:
    """Resolve the predictor set: explicit override (experiment ablations) or
    the intersection of PREDICTOR_VARS with available columns."""
    if predictor_cols:
        available = [c for c in predictor_cols if c in df.columns]
        missing = [c for c in predictor_cols if c not in df.columns]
        if missing:
            print(f"[EVAL] warning: requested predictors missing and skipped: {missing}")
        return available
    return [c for c in PREDICTOR_VARS if c in df.columns]


def locked_block_ids(df: pd.DataFrame) -> list[int]:
    occupied = sorted(df[GROUP_VAR].unique())
    return [occupied[i] for i in range(2, len(occupied), 6)]


def build_y(df: pd.DataFrame, task: str, train_mask: np.ndarray | None = None):
    """Return target series for all rows. Classification uses per-year quartiles
    of lst_C; 3class uses per-year tertiles; binary uses per-year medians;
    regression uses lst_C directly. Thresholds/medians come from train rows
    only when a mask is provided."""
    base = df if train_mask is None else df[train_mask]
    y = pd.Series(index=df.index, dtype=float)
    for year in sorted(df[YEAR_VAR].unique()):
        vals_all = df.loc[df[YEAR_VAR] == year, "lst_C"]
        vals_train = base.loc[base[YEAR_VAR] == year, "lst_C"]
        if task in ("regression", "reg3class"):
            # continuous LST target; reg3class thresholds the predictions later
            y.loc[vals_all.index] = vals_all
        elif task == "binary":
            med = float(vals_train.median())
            y.loc[vals_all.index] = (vals_all > med).astype(int)
        elif task == "3class":
            # per-year LST tertiles from TRAINING rows only -> Low/Moderate/High
            q = vals_train.quantile([1.0 / 3.0, 2.0 / 3.0]).values
            y.loc[vals_all.index] = np.searchsorted(q, vals_all.values, side="right")
        else:
            # classification AND ordinal share the identical target definition:
            # per-year quartile classes from TRAINING rows only.
            q = vals_train.quantile([0.25, 0.5, 0.75]).values
            y.loc[vals_all.index] = np.searchsorted(q, vals_all.values, side="right")
    return y.astype(float)


def tercile_thresholds(df: pd.DataFrame, train_mask: np.ndarray) -> dict:
    """Exact per-year LST tertiles (1/3, 2/3) used by the 3class target,
    computed from train rows only. Returns {year: [t1, t2]}."""
    base = df[train_mask]
    out = {}
    for year in sorted(df[YEAR_VAR].unique()):
        vals_train = base.loc[base[YEAR_VAR] == year, "lst_C"]
        out[str(year)] = [float(v) for v in
                          vals_train.quantile([1.0 / 3.0, 2.0 / 3.0]).values]
    return out


def lst_classes(lst_values, years, thresholds: dict) -> np.ndarray:
    """Convert continuous LST values to 0/1/2 classes using per-year train
    tertiles. ``thresholds`` is {year_str: [t1, t2]} from tercile_thresholds;
    the same searchsorted('right') convention as build_y's 3class branch."""
    lst_values = np.asarray(lst_values, dtype=float)
    years = np.asarray(years)
    cls = np.zeros(len(lst_values), dtype=int)
    for year_str, (t1, t2) in thresholds.items():
        m = years == int(year_str)
        if m.any():
            cls[m] = np.searchsorted(np.array([t1, t2]), lst_values[m], side="right")
    return cls


def score_reg3class(y_true_lst, y_pred_lst, years, thresholds: dict) -> dict:
    """Regression-then-threshold evaluation: continuous LST predictions are
    converted to the 3 severity classes with training-fold-only tertiles and
    scored as classification, with the continuous MAE/RMSE/R2 alongside."""
    y_true_cls = lst_classes(y_true_lst, years, thresholds)
    y_pred_cls = lst_classes(y_pred_lst, years, thresholds)
    out = score_3class(y_true_cls, y_pred_cls)
    out["mae"] = float(mean_absolute_error(y_true_lst, y_pred_lst))
    out["rmse"] = float(mean_squared_error(y_true_lst, y_pred_lst) ** 0.5)
    out["r2"] = float(r2_score(y_true_lst, y_pred_lst))
    return out


class _OrdinalEnsemble:
    """Cumulative-link ordinal classifier for K ordered classes.

    Trains K-1 binary models P(y >= k), k = 1..K-1, and combines them into
    class probabilities: P(y=0) = 1 - p1; P(y=k) = p_k - p_{k+1};
    P(y=K-1) = p_{K-1}. Probabilities are clipped at 0 and renormalised.
    The target definition is unchanged (same per-year quartile classes);
    only the model respects the ordering. Prediction = argmax class.
    """

    def __init__(self, base_factory, n_classes: int = 4):
        self.base_factory = base_factory
        self.n_classes = n_classes

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.models_ = []
        for k in range(1, self.n_classes):
            yb = (y >= k).astype(int)
            model = self.base_factory()
            model.fit(X, yb)
            self.models_.append(model)
        return self

    def predict_proba(self, X):
        n = len(X)
        probs = np.zeros((n, self.n_classes))
        prev = np.ones(n)
        for k, model in enumerate(self.models_, start=1):
            p = model.predict_proba(X)[:, 1]
            probs[:, k - 1] = prev - p
            prev = p
        probs[:, self.n_classes - 1] = prev
        probs = np.clip(probs, 0.0, None)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def make_models(task: str, n_classes: int = 4):
    if task == "ordinal":
        xgb_params = {k: v for k, v in XGBOOST_PARAMS.items()
                      if k not in ("objective", "eval_metric")}
        return {
            "Random Forest": lambda: _OrdinalEnsemble(
                lambda: RandomForestClassifier(**RANDOM_FOREST_PARAMS), n_classes),
            "XGBoost": lambda: _OrdinalEnsemble(
                lambda: XGBClassifier(**xgb_params), n_classes),
        }
    if task in ("regression", "reg3class"):
        # reg3class: same regressors; predictions are tertile-thresholded later
        rf_params = {k: v for k, v in RANDOM_FOREST_PARAMS.items() if k != "class_weight"}
        xgb_params = dict(XGBOOST_PARAMS)
        xgb_params.pop("class_weight", None)
        xgb_params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
        return {
            "Random Forest": lambda: RandomForestRegressor(**rf_params),
            "XGBoost": lambda: XGBRegressor(**xgb_params),
        }
    if task == "binary":
        # Binary task: use XGBoost's default binary:logistic (multi:softprob
        # with num_class=2 returns an (n,2) indicator from predict, which
        # breaks sklearn's binary metrics).
        xgb_params = {k: v for k, v in XGBOOST_PARAMS.items()
                      if k not in ("objective", "eval_metric")}
        return {
            "Random Forest": lambda: RandomForestClassifier(**RANDOM_FOREST_PARAMS),
            "XGBoost": lambda: XGBClassifier(**xgb_params),
        }
    return {
        "Random Forest": lambda: RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        "XGBoost": lambda: XGBClassifier(**{**XGBOOST_PARAMS, "num_class": n_classes}),
    }


def _recall_per_class(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    recalls = []
    for c in labels:
        mask = np.asarray(y_true) == c
        recalls.append((c, float(np.mean(np.asarray(y_pred)[mask] == c)) if mask.any() else np.nan))
    return [c for c, _ in recalls], [r for _, r in recalls]


CLASS3_LABELS = ("Low", "Moderate", "High")


def per_class_metrics(cm: list) -> dict:
    """precision/recall/F1/support per class from a 3x3 confusion matrix
    (rows = true, cols = pred, order Low/Moderate/High)."""
    out = {}
    for i, name in enumerate(CLASS3_LABELS):
        tp = cm[i][i]
        support = sum(cm[i])
        pred_k = sum(row[i] for row in cm)
        prec = tp / pred_k if pred_k else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[name] = {"precision": float(prec), "recall": float(rec),
                     "f1": float(f1), "support": int(support)}
    return out


def score_3class(y_true, y_pred) -> dict:
    """Full report for the 3-class task: headline metrics + confusion matrix
    (always 3x3, labels=[0,1,2]) + per-class precision/recall/F1."""
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "confusion_matrix": cm,
        "per_class": per_class_metrics(cm),
    }


def score(task: str, y_true, y_pred) -> dict:
    if task == "3class":
        return score_3class(y_true, y_pred)
    if task == "regression":
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "r2": float(r2_score(y_true, y_pred)),
        }
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if task in ("classification", "ordinal"):
        out["high_severe_recall"] = float(
            np.nanmean([r for c, r in zip(*_recall_per_class(y_true, y_pred)) if c in (2, 3)])
        )
    return out


def run_evaluation(df: pd.DataFrame, task: str, label: str,
                   skip_cv: bool = False, workers: int = 1,
                   predictor_cols: list[str] | None = None) -> dict:
    # Resolve the predictor set: an explicit experiment override, or the
    # intersection with PREDICTOR_VARS (e.g. the pre-Tier-1 baseline dataset
    # lacks ndmi/mndwi/bsi/ndre — each model is scored on its training features).
    available_predictors = select_predictors(df, predictor_cols)
    df = df.dropna(subset=available_predictors + ["lst_C"]).reset_index(drop=True)
    X, feature_names = encode_predictors(df, predictor_cols=available_predictors)
    groups = df[GROUP_VAR].astype(int).values
    occupied = sorted(df[GROUP_VAR].unique())
    locked = locked_block_ids(df)
    detailed = task in ("3class", "reg3class")  # per-class/confusion reporting
    regress = task == "reg3class"               # regression -> tertile-threshold scoring
    # reg3class fits regressors (n_classes unused there); keep 3 for the record
    n_classes = 3 if regress else int(build_y(df, task).nunique())
    models = make_models(task, n_classes)
    results = {"label": label, "task": task, "n_rows": len(df), "n_features": len(feature_names),
               "occupied_blocks": occupied, "locked_blocks": locked, "locked_rule": LOCKED_BLOCK_RULE,
               "cv5": "reused from pipeline classification_metrics.csv (identical protocol)" if skip_cv else None,
               "models": {}}
    if detailed:
        results["thresholds"] = {}
        results["class_balance_per_year"] = {}

    # 1. Adjacent-block CV — SERIAL (proven; parallel CV5 exhausted 16 GB RAM
    #    at full scale and crashed the orchestrator on 2026-10-03)
    if not skip_cv:
        splits = list(GroupKFold(n_splits=5).split(X, groups=groups))
        if detailed:
            # thresholds are model-independent: compute once per fold
            thr = []
            for i, (train_idx, _) in enumerate(splits):
                mask = np.zeros(len(df), dtype=bool)
                mask[train_idx] = True
                thr.append({"fold": i, "thresholds": tercile_thresholds(df, mask)})
            results["thresholds"]["cv5"] = thr
        for name, factory in models.items():
            cv_scores = []
            cv_details = []
            for i, (train_idx, val_idx) in enumerate(splits):
                mask = np.zeros(len(df), dtype=bool)
                mask[train_idx] = True
                y = build_y(df, task, train_mask=mask)
                model = factory()
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[val_idx])
                if regress:
                    s = score_reg3class(y.iloc[val_idx].values, preds,
                                        df[YEAR_VAR].values[val_idx],
                                        results["thresholds"]["cv5"][i]["thresholds"])
                else:
                    s = score(task, y.iloc[val_idx].values, preds)
                if detailed:
                    cv_scores.append({k: v for k, v in s.items()
                                      if isinstance(v, (int, float))})
                    det = {"fold": i, "n_val": int(len(val_idx)),
                           "confusion_matrix": s["confusion_matrix"],
                           "per_class": s["per_class"]}
                    if regress:
                        det.update({"mae": s["mae"], "rmse": s["rmse"], "r2": s["r2"]})
                    cv_details.append(det)
                else:
                    cv_scores.append(s)
            keys = cv_scores[0].keys()
            results["models"].setdefault(name, {})["cv5"] = {
                k: {"mean": float(np.mean([s[k] for s in cv_scores])),
                    "std": float(np.std([s[k] for s in cv_scores]))} for k in keys
            }
            if detailed:
                results["models"][name]["cv5_details"] = cv_details

    # 2. LOBO (parallel across workers) — computed ONCE for all models
    jobs = [(bid, mname) for bid in occupied for mname in models.keys()]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_lobo_worker_init,
                                 initargs=(X, df, task, n_classes)) as pool:
            lobo_results = list(pool.map(_lobo_one, jobs))
    else:
        _lobo_worker_init(X, df, task, n_classes)
        lobo_results = [_lobo_one(j) for j in jobs]

    if detailed:
        results["thresholds"]["lobo"] = [
            {"block": bid, "thresholds": tercile_thresholds(df, ~(groups == bid))}
            for bid in occupied]

    for mname in models.keys():
        entries = [(bid, s) for bid, m, s in lobo_results if m == mname]
        if detailed:
            scal_keys = [k for k in entries[0][1]
                         if isinstance(entries[0][1][k], (int, float))]
            results["models"].setdefault(mname, {})["lobo"] = {
                k: {"mean": float(np.mean([s[k] for _, s in entries])),
                    "std": float(np.std([s[k] for _, s in entries]))} for k in scal_keys
            }
            cm_sum = np.sum([s["confusion_matrix"] for _, s in entries], axis=0).tolist()
            results["models"][mname]["lobo_pooled"] = {
                "confusion_matrix": cm_sum,
                "per_class": per_class_metrics(cm_sum),
            }
            details = []
            for bid, s in entries:
                det = {"block": bid, "confusion_matrix": s["confusion_matrix"],
                       "per_class": s["per_class"]}
                if regress:
                    det.update({"mae": s["mae"], "rmse": s["rmse"], "r2": s["r2"]})
                details.append(det)
            results["models"][mname]["lobo_details"] = details
        else:
            scores = [s for _, s in entries]
            keys = scores[0].keys()
            results["models"].setdefault(mname, {})["lobo"] = {
                k: {"mean": float(np.mean([s[k] for s in scores])),
                    "std": float(np.std([s[k] for s in scores]))} for k in keys
            }

    # 3. Locked geographic holdout — evaluated ONCE per model (all cores)
    for name, factory in models.items():
        test_mask = np.isin(groups, locked)
        train_mask = ~test_mask
        y = build_y(df, task, train_mask=train_mask)
        model = factory()
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        if regress:
            locked_scores = score_reg3class(
                y[test_mask].values, preds, df[YEAR_VAR].values[test_mask],
                tercile_thresholds(df, train_mask))
        else:
            locked_scores = score(task, y[test_mask].values, preds)
        results["models"][name]["locked_holdout"] = locked_scores
        print(f"[EVAL] {label} | {task} | {name}: locked = {locked_scores}", flush=True)

    if detailed:
        locked_train_mask = ~np.isin(groups, locked)
        results["thresholds"]["locked"] = tercile_thresholds(df, locked_train_mask)
        # class balance per year under the locked-train target definition
        y_def = build_y(df, "3class", train_mask=locked_train_mask)
        balance = {}
        for year in sorted(df[YEAR_VAR].unique()):
            yr_mask = (df[YEAR_VAR].values == year)
            sub = y_def[yr_mask]
            n = int(len(sub))
            counts = sub.value_counts()
            entry = {
                "n": n,
                "low_pct": float(counts.get(0, 0) / n),
                "moderate_pct": float(counts.get(1, 0) / n),
                "high_pct": float(counts.get(2, 0) / n),
            }
            test_yr = yr_mask & np.isin(groups, locked)
            n_test = int(test_yr.sum())
            if n_test:
                ct = y_def[test_yr].value_counts()
                entry["locked_test"] = {
                    "n": n_test,
                    "low_pct": float(ct.get(0, 0) / n_test),
                    "moderate_pct": float(ct.get(1, 0) / n_test),
                    "high_pct": float(ct.get(2, 0) / n_test),
                }
            balance[str(year)] = entry
        results["class_balance_per_year"] = balance
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--spatial-cache", default=None)
    parser.add_argument("--morph-cache", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--task", choices=["classification", "regression", "binary", "ordinal", "3class", "reg3class"], required=True)
    parser.add_argument("--skip-cv", action="store_true",
                        help="Reuse the pipeline's identical fold CV metrics; run LOBO + locked only.")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Optional seeded smoke-test subsample size (e.g. 30000).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel LOBO worker processes; each fit uses 4 threads "
                             "(4x4 = 16 threads on a 16 GB / 16-core machine).")
    parser.add_argument("--out-dir", default="data/processed/experiments")
    parser.add_argument("--extra-features-csv", default=None,
                        help="Optional experiment feature table (row,col,year + columns) merged in.")
    parser.add_argument("--predictor-cols", default=None,
                        help="Comma-separated explicit predictor list (experiment ablations); "
                             "default: PREDICTOR_VARS intersected with available columns.")
    args = parser.parse_args()

    df = load_and_merge(args.dataset, args.spatial_cache, args.morph_cache,
                        extra_csv=args.extra_features_csv)
    if args.subsample and len(df) > args.subsample:
        df = df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        print(f"[EVAL] smoke-test subsample: {len(df)} rows")
    predictor_cols = [c.strip() for c in args.predictor_cols.split(",")] if args.predictor_cols else None
    results = run_evaluation(df, args.task, args.label, skip_cv=args.skip_cv,
                             workers=args.workers, predictor_cols=predictor_cols)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.label}_{args.task}_geoeval.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"[EVAL] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
