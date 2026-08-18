"""Regression lock for the bounded C5P O2 GPU smoke on job 6894971."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase2_o2_gpu_smoke_6894971.json"


def _load() -> tuple[bytes, dict]:
    raw = RESULT.read_bytes()
    return raw, json.loads(raw)


def test_o2_smoke_compact_result_is_immutable() -> None:
    raw, _ = _load()
    assert hashlib.sha256(raw).hexdigest() == (
        "a812a5642c5d3d9dedc25711af6d92994bcebf5432d9dd7a99911fc1f5e53451"
    )


def test_o2_smoke_execution_identity_and_hardware_are_exact() -> None:
    _, result = _load()
    assert result["status"] == "passed"
    assert result["slurm"] == {
        "elapsed": "00:02:28",
        "exit_code": "0:0",
        "job_id": "6894971",
        "partition": "gpupreempt",
        "state": "COMPLETED",
    }
    assert result["paper0_commit"] == (
        "8abdddafcabd24f3f5b0774253c9a6cfd2fc4419"
    )
    assert result["rocky_major"] == 9
    assert result["accelerator"]["name"] == "NVIDIA H200"
    assert result["accelerator"]["count"] == 1
    assert result["test_suite"] == {
        "passed": 562,
        "skipped": 1,
        "subtests_passed": 29,
    }


def test_both_o2_smoke_arms_passed_engineering_checks() -> None:
    _, result = _load()
    assert [(run["arm"], run["seed"]) for run in result["runs"]] == [
        ("C5P-H1", 1701),
        ("C5P-H2", 1701),
    ]
    for run in result["runs"]:
        assert run["completed_epochs"] == 2
        assert run["completed_optimizer_steps"] == 2
        assert run["training_target_count"] == 16
        assert run["validation_target_count"] == 4
        assert run["finite_gradient_norms"] is True
        assert run["checkpoint_reload_bitwise_exact"] is True
        assert run["physics_derived_loss_used"] is False
        assert run["target_truth_used_as_model_input"] is False
        assert run["wandb"]["epochs_logged"] == 2
        assert run["wandb"]["remote_state_after_finish"] == "finished"


def test_smoke_is_not_a_scientific_result_or_o2_acceptance() -> None:
    _, result = _load()
    assert result["development_run"] == "85604"
    assert result["held_out_85606_read"] is False
    assert result["scientific_result"] is False
    assert result["O2_scientific_gate_evaluated"] is False
    assert result["O3_launch_allowed"] is False
    assert result["full_O2_training_allowed"] is True


def test_raw_h100_label_misnomer_is_disclosed_without_rewriting_history() -> None:
    _, result = _load()
    raw = result["raw_artifacts"]
    assert raw["raw_smoke_summary_sha256"] == (
        "6b3701e1d67bbed0ced3dd8fee31dc284e158a3f825f61697e20027921d884c2"
    )
    assert "stale label" in raw["raw_summary_label_note"]
    assert "NVIDIA H200" in raw["raw_summary_label_note"]
