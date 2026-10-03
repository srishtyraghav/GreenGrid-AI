# Baseline vs Tier 1+2 — Uniform Geographic Comparison

**Date:** 2026-10-03 · **Protocol:** identical for both — 5-fold adjacent-block GroupKFold CV (from the Phase 5 pipeline runs), leave-one-block-out over all occupied blocks, and the deterministic locked geographic holdout (`occupied_block_ids[2::6]`, evaluated exactly once, never tuned on). Targets use per-year quartiles computed on training rows only.

## Headline (XGBoost — selected model in both runs)

| Validation | Baseline (5-yr, paired sample) | Tier 1+2 (750k, unpaired + new features) | Δ |
|---|---|---|---|
| Adjacent-block CV accuracy | 43.03% | 50.26% | **+7.2 pt** |
| Adjacent-block CV macro-F1 | 0.4051 | 0.4815 | +0.076 |
| LOBO accuracy (19 blocks) | 40.85% | 51.63% | **+10.8 pt** |
| LOBO macro-F1 | 0.3254 | 0.4757 | +0.150 |
| **Locked holdout accuracy** | **35.15%** | **49.59%** | **+14.4 pt** |
| Locked holdout macro-F1 | 0.3243 | 0.4726 | +0.148 |

Random Forest shows the same pattern (locked 36.1% → 48.0%).

## What changed between the runs
- **Tier 1 features:** NDMI, MNDWI, BSI (Landsat 9 existing bands), NDRE (Sentinel-2 red-edge); spatial means extended over NDMI/BSI.
- **Tier 2 sampling:** per-year independent 150k samples (750k rows total) instead of the all-years paired intersection (88,994 pixels × 5); paired table retained separately for temporal analytics.

## Caveats (honest)
1. **Locked blocks differ between datasets** — the rule is deterministic per dataset's occupied-block set (baseline: [2, 11, 18]; Tier 1+2: [2, 9, 15, 23]). LOBO — computed over every occupied block — is the fully like-for-like geographic metric; the locked numbers are indicative, not perfectly paired geographies.
2. Baseline evaluated with its 90 training-era predictors; Tier 1+2 with 100. This is deliberate: each model is scored on the features it was actually trained on.
3. The original 2-year model's published locked holdout (39.9%) came from the old audit on different data; the comparison above is within the new 5-year series.

## Verdict
Tier 1+2 delivers a large, consistent gain across all three geographic validations — the improvement is real generalization, not a fold artifact. Locked-holdout spread CV→LOBO→locked (50.4 → 51.6 → 49.6) shows no validation-strategy luck.

## Provenance artifacts
- `data/processed/experiments/baseline_5yr_4class_classification_geoeval.json`
- `data/processed/experiments/tier12_5yr_classification_geoeval.json`
- Snapshots: `data/processed/baseline_5yr_4class/`, `data/processed/tier12_outputs/`
