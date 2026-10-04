# GreenGrid AI

> **Where is the city hot? Why is it hot? Where should trees be planted? How many trees are required? How much cooling can be expected?**

An AI-powered Urban Heat Island (UHI) detection and tree plantation planning
system for the National Capital Territory of Delhi, India.

This repository is split into two frozen-baseline pipelines:

| Tree | Contents | Status |
|---|---|---|
| [`v1/`](v1/README.md) | The original 12-phase project: July 2022–2026 dataset, experiment campaign (frozen 4-class → 3-class lineage), production Phases 5–8. **Baseline — preserved unchanged.** | Phases 1–8 complete |
| [`v2/`](v2/README.md) | Clean rebuild for final academic submission: W4 (May 1–Jun 30) hot-season window, coverage-guarded GEE acquisition, self-contained data + code. | Phase 2 complete; Phase 3+ pending review |

Shared infrastructure at the root: `.venv/` (Python environment used by both
trees), `.gitignore`, and this index. V1 content is byte-identical after the
2026-10-04 relocation — only its path prefix changed (`v1/`). V1-vs-V2
results are **not season-comparable at face value** (July vs May–Jun
windows); comparisons must use each version's own validation protocol. See
`v2/README.md` for the V2 comparison protocol and phase status.
