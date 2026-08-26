"""Known-answer checks for exact-phi assembly of predicted E6B forecasts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from paper0.tools.assemble_matched_state_exact_phi import (
    EXPECTED_CANDIDATES,
    assemble,
)


ROOT = Path(__file__).resolve().parents[1]
EXACT_PHI_LAUNCHER = (
    ROOT / "cluster/post_ecrd_old_85604_matched_state_exact_phi.sbatch"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return _sha(path)


def test_assembly_accepts_only_truth_free_predicted_boundary_outputs(
    tmp_path,
) -> None:
    candidates = []
    phi_root = tmp_path / "phi"
    phi_root.mkdir()
    for name in EXPECTED_CANDIDATES:
        candidate = tmp_path / name
        start = 500 if name.startswith("h4_") else 504
        with h5py.File(candidate, "w") as handle:
            handle.attrs["target_truth_used_during_generation"] = False
            handle.attrs["boundary_policy"] = "predicted_Bphi_no_truth_bypass"
            coordinates = handle.create_group("coordinates")
            coordinates.create_dataset(
                "frame_index", data=np.arange(start, start + 2)
            )
            handle.create_group("candidate").create_dataset(
                "Ne", data=np.zeros((2, 1))
            )
        candidate_sha = _sha(candidate)
        candidates.append({"path": str(candidate), "sha256": candidate_sha})

        task = phi_root / candidate.stem
        task.mkdir()
        phi = task / "derived_phi.h5"
        phi.write_bytes(b"known-phi")
        phi_sha = _sha(phi)
        result = {
            "scope": "phase2_matched_e6b_elliptic_output",
            "status": "completed",
            "development_run": "85604",
            "held_out_85606_read": False,
            "truth_layout": False,
            "truth_replay_gate": None,
            "frame_interval": [start, start + 2],
            "source_input": {
                "path": str(candidate.resolve()),
                "sha256": candidate_sha,
            },
            "derived_phi": {"path": str(phi), "sha256": phi_sha},
        }
        _json(task / "result.json", result)

    generation = {
        "scope": "post_ecrd_old_85604_matched_state_bounded_generation",
        "status": "completed",
        "development_run": "85604",
        "family": "e6b",
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "target_truth_used_during_generation": False,
        "elliptic_candidates": candidates,
    }
    generation_path = tmp_path / "generation.json"
    generation_sha = _json(generation_path, generation)

    result = assemble(
        generation_result=generation_path,
        generation_result_sha256=generation_sha,
        phi_root=phi_root,
        paper0_commit="a" * 40,
        slurm_job_id="known-answer",
    )
    assert result["candidate_count"] == len(EXPECTED_CANDIDATES)
    assert not result["target_truth_phi_read"]
    assert result["paired_common_view_physics_scoring_authorized"]
    assert all(
        record["candidate"]["frame_interval"][0] in (500, 504)
        for record in result["outputs"]
    )


def test_exact_phi_launcher_separates_test_and_data_environments() -> None:
    text = EXACT_PHI_LAUNCHER.read_text(encoding="utf-8")
    assert 'readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"' in text
    assert 'readonly TEST_PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"' in text
    assert '"${TEST_PYTHON}" -m pytest' in text
    assert '"${PYTHON}" -m pytest' not in text
