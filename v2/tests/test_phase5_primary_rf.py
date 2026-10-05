"""Tests for the frozen RF-primary designation and the reports convention.

(a) the primary marker designates RF, the persisted artifacts exist, and the
    production params are the verbatim pre-specified V1 set;
(b) v2/reports/ holds ONLY the expected phase-wise .md files and no logs —
    the user's reports convention (V1 rule), guarded against regression.

Real-artifact checks skip gracefully when v2/data artifacts are absent
(v2/data is gitignored); the reports-convention check always runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # the v2/ tree
SRC = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from v2.phase5.rf_compare import RF_PARAMS_V1  # noqa: E402

PHASE5_DIR = PROJECT_ROOT / "data" / "phase5"
REPORTS_DIR = PROJECT_ROOT / "reports"

EXPECTED_REPORTS = {
    "phase2_dataset_report.md",
    "phase3_preprocessing_report.md",
    "phase4_feature_extraction_report.md",
    "phase5_production_model.md",
    "phase5_uhi_detection_report.md",
    "phase6_uhi_severity_mapping_report.md",
    "phase7_tree_plantation_suitability_report.md",
}


@pytest.fixture(scope="module")
def marker():
    path = PHASE5_DIR / "phase5_primary_model.json"
    if not path.exists():
        pytest.skip("production marker absent (v2/data not built here)")
    return json.loads(path.read_text())


def test_marker_designates_rf(marker):
    assert marker["primary_model"] == "rf"
    assert marker["frozen"] is True
    assert marker["locked_accuracy"] == 0.6990177602649763
    assert marker["locked_macro_f1"] == 0.6851219134292812
    assert marker["baseline_xgb"] == "phase5_primary_xgb_v2.json"
    assert (PHASE5_DIR / marker["artifact"]).exists()
    assert (PHASE5_DIR / "phase5_primary_rf_v2.json").exists()


def test_production_params_verbatim_v1():
    path = PHASE5_DIR / "phase5_primary_rf_v2.json"
    if not path.exists():
        pytest.skip("production metadata absent (v2/data not built here)")
    meta = json.loads(path.read_text())
    assert meta["model"] == "random_forest"
    stored = dict(meta["rf_params"])
    stored["max_depth"] = None          # JSON carries "None (unlimited)"
    assert stored == RF_PARAMS_V1


def test_marker_verified_by_freeze_gate(marker):
    assert marker.get("verification_passed") is True, (
        "run: python -m v2.phase5.verify_rf_primary --features-dir "
        "v2/data/phase4")


def test_reports_dir_phase_only_convention():
    files = {p.name for p in REPORTS_DIR.iterdir()}
    assert files == EXPECTED_REPORTS, (
        f"reports/ must hold exactly the phase-wise .md files; got "
        f"{sorted(files)}")
    assert not list(REPORTS_DIR.rglob("*.log")), \
        "no .log files anywhere under reports/ (logs live in v2/logs/)"
