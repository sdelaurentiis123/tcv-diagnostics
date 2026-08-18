"""Contract tests for the matched O1 finalizer CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/finalize_matched_o1_codec.py"
SPEC = importlib.util.spec_from_file_location("finalize_matched_o1_codec", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _reconstruction() -> dict:
    split = lambda interval: {
        "frames": list(interval),
        "frame_count": interval[1] - interval[0],
        "candidate_native81": {"path": "/tmp/candidate.h5", "sha256": "abc"},
    }
    return {
        "scope": "phase2_matched_o1_codec_reconstruction",
        "status": "completed_pending_exact_elliptic_and_transport",
        "development_run": "85604",
        "held_out_85606_read": False,
        "evaluation_commit": "commit",
        "checkpoint_reload_bitwise_exact_on_evaluation_worker": True,
        "identity": {
            "family": "c5p",
            "codec": "dcae_l20",
            "seed": 1701,
            "checkpoint_sha256": "checkpoint",
        },
        "training": split((0, 432)),
        "validation": split((496, 624)),
    }


def test_reconstruction_contract_keeps_the_guard_gap() -> None:
    identity = MODULE.validate_reconstruction_result(_reconstruction())
    assert identity["family"] == "c5p"
    record = _reconstruction()
    record["validation"]["frames"] = [432, 560]
    with pytest.raises(ValueError, match="validation interval"):
        MODULE.validate_reconstruction_result(record)


def test_truth_replay_contract_requires_all_624_frames() -> None:
    record = {
        "scope": "phase2_matched_e6b_zero_seed_truth_replay",
        "status": "pass",
        "development_run": "85604",
        "held_out_85606_read": False,
        "coverage": [0, 624],
        "frame_count": 624,
        "all_frames_passed": True,
        "boundary_only_zero_interior_seed": True,
        "zperiod": 5,
    }
    MODULE.validate_truth_replay_summary(record)
    record["frame_count"] = 623
    with pytest.raises(ValueError, match="did not pass"):
        MODULE.validate_truth_replay_summary(record)


def test_json_safe_rejects_nonfinite_results() -> None:
    assert MODULE._json_safe({"array": np.asarray([1.0, 2.0])}) == {
        "array": [1.0, 2.0]
    }
    with pytest.raises(ValueError, match="non-finite"):
        MODULE._json_safe({"bad": np.nan})


def test_finalizer_does_not_import_training_dependencies() -> None:
    text = TOOL.read_text(encoding="utf-8")
    assert "tcv_diagnostics.codec_training" not in text
    assert "sha256_file as sha256_path" in text
