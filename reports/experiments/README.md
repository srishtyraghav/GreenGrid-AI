# Experiment & Audit Records (archive)

This folder holds the experiment reports, audits, and comparison notes that
fed the phase-wise reports. It is kept separate so that `reports/` itself
contains only phase-wise reports (source material for the research report).

| Document | Contents |
|---|---|
| `experiment_chain_results.md` | Tier 1+2 / LULC / ordinal experiment chain and gates |
| `baseline_vs_tier12_comparison.md` | 5-yr baseline vs Tier 1+2 |
| `tier1_2_tier4_results.md` | Tier 1+2 summary + Tier 4 (binary/regression) results |
| `exp_met_results.md` / `exp_met_provenance.md` | Met-covariate experiment (51.40% locked 4-class) + data provenance |
| `exp_3class_results.md` | 3-class tertile target experiment (62.99% locked) |
| `exp_reg3class_results.md` | Regression → tertile-threshold experiment (60.45% locked; negative result) |
| `exp_spatial_results.md` | Spatial-context features experiment (63.97% locked) — features promoted to production |
| `alignment_audit.md` | Label/predictor spatial-alignment audit (PASS) |
| `coverage_audit_2025.*` | 2025 Landsat/Sentinel-2 valid-pixel coverage audit (figures + block table) |

The official Phase 5 production model and its handoff document live outside
this archive: `../phase5_production_model.md` and
`../../data/processed/phase5_production_3class/`.
