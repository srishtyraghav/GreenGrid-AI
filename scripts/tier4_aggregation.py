"""Tier 4 experiment 3 — spatial/block AGGREGATION analysis.

⚠ LABEL: This is a DIFFERENT EVALUATION GRAIN, not an improvement of the
primary 4-class pixel-level model. It asks: "if decisions are made at the
level of ~600 ha spatial blocks (e.g. district-scale planning), how stable is
the aggregated severity?" Pixel-level accuracy of the primary model is
unchanged by anything computed here.

Method:
  - Uses the out-of-fold (OOF) predictions of the Tier 1+2 run — honest
    held-out pixel predictions (no in-sample inflation).
  - Severity score per pixel = expected ordinal class (sum k * P(k)), which
    is exactly the project's Phase 6 severity-score convention.
  - Block-level severity = mean of pixel severity scores within a
    spatial_block_id x year.
  - Block truth = same aggregation on actual classes.
  - Block classes: per-year quartiles of block means (same relative-target
    convention as the pixel task).
  - Aggregation uplift = block-level accuracy − pixel-level accuracy.

Outputs:
  data/processed/experiments/tier4_aggregation_blocks.csv
  data/processed/experiments/tier4_aggregation_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

EXPERIMENTS_DIR = Path("data/processed/experiments")
OOF_CSV = Path("data/processed/tier12_outputs/phase5_tables/oof_predictions.csv")
CLASS_LABELS = ["Low", "Moderate", "High", "Severe"]


def block_classes(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Per-year quartile classes of block-level means (relative convention)."""
    out = pd.Series(index=df.index, dtype=int)
    for year, idx in df.groupby("year").groups.items():
        means = df.loc[idx, value_col]
        q = means.quantile([0.25, 0.5, 0.75]).values
        out.loc[idx] = np.searchsorted(q, means.values, side="right")
    return out


def main() -> int:
    oof = pd.read_csv(OOF_CSV)
    prob_cols = [f"probability_{l.lower()}" for l in CLASS_LABELS]
    oof["severity_score"] = sum(oof[f"probability_{l.lower()}"].to_numpy() * k
                                for k, l in enumerate(CLASS_LABELS))
    oof["actual_score"] = oof["actual_class"].astype(float)

    rows = []
    for model in ["Random Forest", "XGBoost"]:
        sub = oof[oof["model"] == model]
        pixel_acc = accuracy_score(sub["actual_class"], sub["predicted_class"])
        pixel_f1 = f1_score(sub["actual_class"], sub["predicted_class"], average="macro")

        block = (sub.groupby(["spatial_block_id", "year"], as_index=False)
                 .agg(pred_score=("severity_score", "mean"),
                      actual_score=("actual_score", "mean"),
                      n_pixels=("actual_class", "size")))
        block["pred_class"] = block_classes(block, "pred_score")
        block["actual_class_block"] = block_classes(block, "actual_score")

        block_acc = accuracy_score(block["actual_class_block"], block["pred_class"])
        block_f1 = f1_score(block["actual_class_block"], block["pred_class"], average="macro")
        block_kappa = cohen_kappa_score(block["actual_class_block"], block["pred_class"])

        # Majority-vote variant (discrete vote, not score mean)
        vote = (sub.groupby(["spatial_block_id", "year"])["predicted_class"]
                .agg(lambda s: s.mode().iloc[0]).rename("vote_class").reset_index())
        truth_vote = (sub.groupby(["spatial_block_id", "year"])["actual_class"]
                      .agg(lambda s: s.mode().iloc[0]).rename("truth_class").reset_index())
        vote = vote.merge(truth_vote, on=["spatial_block_id", "year"])
        vote_acc = accuracy_score(vote["truth_class"], vote["vote_class"])

        rows.append({
            "model": model,
            "n_pixel_rows": int(len(sub)),
            "n_blocks_years": int(len(block)),
            "pixel_accuracy": round(float(pixel_acc), 4),
            "pixel_macro_f1": round(float(pixel_f1), 4),
            "block_score_accuracy": round(float(block_acc), 4),
            "block_score_macro_f1": round(float(block_f1), 4),
            "block_score_kappa": round(float(block_kappa), 4),
            "block_vote_accuracy": round(float(vote_acc), 4),
            "aggregation_uplift_score": round(float(block_acc - pixel_acc), 4),
            "aggregation_uplift_vote": round(float(vote_acc - pixel_acc), 4),
        })

    summary = {
        "label": "tier4_aggregation",
        "warning": ("DIFFERENT EVALUATION GRAIN (block-level aggregation of OOF pixel "
                    "predictions). NOT an improvement of the primary 4-class pixel task."),
        "oof_source": str(OOF_CSV),
        "block_definition": "spatial_block_id (5x5 grid) x year",
        "results": rows,
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "tier4_aggregation_summary.json").write_text(
        json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(EXPERIMENTS_DIR / "tier4_aggregation_blocks.csv", index=False)

    for r in rows:
        print(f"{r['model']}: pixel {r['pixel_accuracy']*100:.2f}% → "
              f"block-score {r['block_score_accuracy']*100:.2f}% "
              f"(uplift {r['aggregation_uplift_score']*100:+.2f} pt, "
              f"kappa {r['block_score_kappa']:.3f}), "
              f"block-vote {r['block_vote_accuracy']*100:.2f}%")
    print("[TIER4] wrote tier4_aggregation_summary.json + tier4_aggregation_blocks.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
