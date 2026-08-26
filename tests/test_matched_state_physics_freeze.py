"""Known-answer checks for the matched state-view physics freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paper0.tools.assemble_matched_state_exact_phi import SCOPE as PHI_SCOPE
from paper0.tools.freeze_matched_state_physics_scoring import freeze_manifest
from paper0.tools.generate_matched_state_bounded_forecasts import (
    SCOPE as GENERATION_SCOPE,
)


def _write(path: Path, record: dict) -> str:
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_lock(path: Path, content: bytes) -> dict[str, str]:
    path.write_bytes(content)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_freeze_locks_pair_exact_phi_blocks_and_decision(tmp_path: Path) -> None:
    generation_manifest = tmp_path / "generation_manifest.json"
    generation_manifest_sha = _write(
        generation_manifest,
        {
            "scope": GENERATION_SCOPE,
            "status": (
                "frozen_after_paired_transition_reduction_before_inference"
            ),
            "development_run": "85604",
            "held_out_85606_access_allowed": False,
            "new_nersc_data_access_allowed": False,
            "guard_frames_read_allowed": False,
            "training_allowed": False,
            "checkpoint_selection_allowed": False,
        },
    )
    generations = {}
    for family in ("c5p", "e6b"):
        forecast = _file_lock(tmp_path / f"{family}.h5", family.encode())
        result_path = tmp_path / f"{family}.json"
        result_sha = _write(
            result_path,
            {
                "scope": GENERATION_SCOPE,
                "status": "completed",
                "development_run": "85604",
                "family": family,
                "held_out_85606_read": False,
                "new_nersc_data_read": False,
                "guard_frames_read": False,
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "physics_evaluation_performed": False,
                "target_truth_used_during_generation": False,
                "manifest": {
                    "path": str(generation_manifest),
                    "sha256": generation_manifest_sha,
                },
                "forecast": forecast,
                "elliptic_candidates": (
                    [{"path": str(tmp_path / f"candidate_{index}")}
                     for index in range(7)]
                    if family == "e6b"
                    else []
                ),
            },
        )
        generations[family] = (result_path, result_sha)

    exact = tmp_path / "exact.json"
    exact_sha = _write(
        exact,
        {
            "scope": PHI_SCOPE,
            "status": "completed",
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "target_truth_phi_read": False,
            "truth_layout": False,
            "candidate_count": 7,
            "paired_common_view_physics_scoring_authorized": True,
            "generation_result": {
                "path": str(generations["e6b"][0]),
                "sha256": generations["e6b"][1],
            },
        },
    )
    dependencies = {
        "known": _file_lock(tmp_path / "known.dat", b"known")
    }
    result = freeze_manifest(
        generation_manifest=generation_manifest,
        generation_manifest_sha256=generation_manifest_sha,
        generation_results=generations,
        exact_phi_result=exact,
        exact_phi_result_sha256=exact_sha,
        paper0_root=tmp_path,
        paper0_commit="a" * 40,
        dependencies=dependencies,
    )
    assert result["state_views"] == ["c5p", "e6b"]
    assert result["evaluation"]["target_frame_blocks"]["4"] == [
        [500, 541],
        [541, 582],
        [582, 624],
    ]
    assert result["decision"][
        "separatrix_transport_e6b_over_c5p_max"
    ] == 0.9
    assert not result["held_out_85606_access_allowed"]
