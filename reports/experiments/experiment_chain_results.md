# Experiment Chain Report — Ordinal Gate Decision

**Date:** 2026-10-03 · **Frozen baseline:** XGBoost + LULC, locked geographic holdout **50.02% / 0.4769**
**Protocol (unchanged):** 5-fold adjacent-block GroupKFold CV, LOBO over all occupied blocks, deterministic locked holdout `occupied_block_ids[2::6]` = blocks [2, 9, 15, 23] (identical for both runs — same dataset), evaluated exactly once, never tuned on. Targets from training rows only; identical per-year-quartile class definition.

## Experiment 1 — Ordinal formulation (cumulative-link ensemble: 3 binary P(y≥k) models)

| Model | CV5 acc / F1 | LOBO acc / F1 | **Locked acc / F1** |
|---|---|---|---|
| Frozen baseline XGB+LULC | 50.63 / 0.4851 | 52.38 / 0.4800 | **50.02 / 0.4769** |
| **Ordinal XGB** | 50.24 / 0.4892 | 52.91 / 0.4877 | **49.63 / 0.4841** |
| Frozen baseline RF | 49.91 / 0.4803 | 51.81 / 0.4767 | 48.50 / 0.4672 |
| Ordinal RF | 48.14 / 0.4145 | 49.38 / 0.4195 | 47.55 / 0.4113 |

## Gate decision: FAIL — chain stops

The protocol gate — "proceed only if the ordinal locked result exceeds 50.02%" — is **not met**:
- Locked accuracy **49.63% < 50.02%** (−0.39 pt). Both models agree (ordinal RF also below its baseline).
- Recorded for honesty: ordinal macro-F1 (0.4841) and LOBO accuracy (52.91%) are both *above* baseline (0.4769 / 52.38%) — the ordinal structure helps class balance and average-block generalization slightly, but not unseen-geography accuracy on the pre-registered gate metric.
- **Experiment 2 is NOT launched** per the explicit gating rule. Experiments 3 and 4 remain conditional and unstarted.

## Interpretation
With the same features and identical target, respecting class ordering trades a small amount of boundary-decision freedom for structural consistency — and that trade is slightly net-negative on the locked metric. Combined with the earlier finding (quartile-boundary noise dominates), this further confirms the 4-class ceiling is **informational** (feature–LST relationship), not architectural (loss/label structure). Per the earlier Tier 4 evidence, the remaining honest levers for the primary task are new data types (meteorological covariates at acquisition time, thermal sharpening), not further reformulations of the same inputs.

## Artifacts
- `data/processed/experiments/exp1_ordinal_ordinal_geoeval.json` — Experiment 1 full results
- `data/processed/experiments/lulc_5yr_classification_geoeval.json` — frozen baseline results
- `scripts/geographic_holdout_eval.py` — evaluator (now also supports `--task ordinal`, `--extra-features-csv`, `--predictor-cols` for any future experiment)
- The primary model (XGB+LULC, locked 50.02%) remains the production model, untouched.
