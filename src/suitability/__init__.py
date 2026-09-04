"""Phase 7: tree-plantation suitability analysis.

Stage 1 (core suitability scoring) builds on the frozen Phase 6 severity
outputs and produces a RELATIVE tree-plantation suitability proxy as a
gated product of Heat Need (0-100) and Plantation Opportunity (0-100):

    Final = Need * Opportunity / 100

2022 and 2026 are two snapshot composites, not a trend; all normalizations
and classes are per-year (class-relative cross-year comparison only).
Nothing in this package claims planting success probability.
"""
