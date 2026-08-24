"""Known-answer tests for the prospective full Stage-1 reducer."""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/summarize_codec_free_stage1_full.py"


def load_namespace():
    namespace = {"__name__": "stage1_reducer_test"}
    exec(compile(TOOL.read_text(encoding="utf-8"), str(TOOL), "exec"), namespace)
    return namespace


def test_numeric_summary_reports_seed_median_and_range() -> None:
    summary = load_namespace()["numeric_summary"]([3.0, 1.0, 2.0])
    assert summary == {"median": 2.0, "minimum": 1.0, "maximum": 3.0}
    with pytest.raises(ValueError, match="nonempty"):
        load_namespace()["numeric_summary"]([])


def test_reducer_freezes_exact_arm_and_block_identities() -> None:
    namespace = load_namespace()
    assert namespace["FAMILIES"] == ("c5p", "e6b")
    assert namespace["SEEDS"] == (1701, 1702, 1703)
    assert namespace["BLOCKS"] == ("V00", "V01", "V02")
    assert namespace["E6B_FIELDS"] == ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort")


def test_reducer_source_locks_prospective_thresholds_and_scope() -> None:
    text = TOOL.read_text(encoding="utf-8")
    assert "aggregate_ratio <= 1.10" in text
    assert "ratio <= 1.25" in text
    assert '"held_out_85606_read": False' in text
    assert '"physics_derived_loss_used": False' in text
