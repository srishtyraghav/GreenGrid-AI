"""V2 Phase 5 — UHI 3-class detection model (frozen V1 protocol replication).

Protocol is FROZEN from V1's production model (v1/data/processed/
phase5_production_3class/): per-year LST tertile targets computed from
training rows only, 5-fold adjacent-block CV (GroupKFold on
spatial_block_id), LOBO over all 25 pinned blocks, and a single-threaded
final refit on the 20 non-locked blocks with one locked evaluation on
blocks {2, 9, 15, 23}. No experiment search.
"""
