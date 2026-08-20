"""Prospective tests for the bounded ECRD smoke finalizer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tcv_diagnostics.ecrd_training import (
    ECRDTrainingConfig,
    frozen_parameter_counts,
    model_config_record,
)
from tcv_diagnostics.models.field_residual_edm import B5_RESIDUAL_SCALES


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/summarize_ecrd_engineering_smoke.py"
SPEC = importlib.util.spec_from_file_location("summarize_ecrd_smoke", TOOL)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path, payload: bytes) -> dict[str, str]:
    path.write_bytes(payload)
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def _synthetic_b5_run(
    tmp_path: Path, *, commit: str, manifest_sha256: str
) -> tuple[Path, dict[str, object]]:
    run = tmp_path / "b5"
    run.mkdir()
    artifacts = {
        name: _artifact(run / f"{name}.bin", name.encode("utf-8"))
        for name in (
            "config",
            "training_order",
            "validation_seed_bank",
            "history",
            "selected_checkpoint",
        )
    }
    candidate = _artifact(run / "candidate.pt", b"candidate")
    candidate.update({"completed_epoch": 1, "global_optimizer_step": 2})
    config = ECRDTrainingConfig(arm="B5", seed=1701, mode="smoke")
    run_config = {
        "scope": "ECRD_matched_model_development_85604",
        "paper0_commit": commit,
        "training": config.to_record(),
        "model": model_config_record("B5"),
        "parameter_count": frozen_parameter_counts()["B5"],
        "residual_scales": list(B5_RESIDUAL_SCALES),
        "authority": {
            "authorized": True,
            "scope": "ECRD_smoke_B5_seed1701_85604",
            "mode": "smoke",
            "arm": "B5",
            "seed": 1701,
            "development_run": "85604",
            "target_truth_used_as_condition": False,
            "guard_frames_read": False,
            "held_out_85606_read": False,
            "manifest_sha256": manifest_sha256,
            "protocol_sha256": SUMMARY.EXPECTED_PROTOCOL_SHA256,
        },
    }
    _write_json(Path(artifacts["config"]["path"]), run_config)
    artifacts["config"]["sha256"] = hashlib.sha256(
        Path(artifacts["config"]["path"]).read_bytes()
    ).hexdigest()
    result = {
        "scope": "ECRD_matched_model_development_training_85604",
        "status": "training_completed_checkpoint_selected",
        "mode": "smoke",
        "arm": "B5",
        "seed": 1701,
        "paper0_commit": commit,
        "development_run": "85604",
        "training": config.to_record(),
        "model": model_config_record("B5"),
        "parameter_count": frozen_parameter_counts()["B5"],
        "completed_epochs": 1,
        "completed_optimizer_steps": 2,
        "target_presentations": 4,
        "candidate_count": 1,
        "selected_completed_epoch": 1,
        "checkpoint_reload_bitwise_exact": True,
        "selected_validation": {
            "target_frames": [498, 502],
            "target_count": 4,
            "probes_per_target": 4,
            "blocks": {"SMOKE": {}},
            "checkpoint_score": 1.0,
        },
        "artifacts": {**artifacts, "candidate_checkpoints": [candidate]},
        "training_performed": True,
        "validation_frames_read": True,
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "target_truth_used_as_condition": False,
        "absolute_time_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    probe = {
        "scope": "bounded_non_scientific_ECRD_full_volume_mechanical_probe",
        "arm": "B5",
        "optimizer_steps": 2,
        "canonical_field_shape": [1, 2, 1, 5, 64, 32, 88],
        "ensemble_members": 2,
        "sampler_steps": 18,
        "network_evaluations_per_member": 35,
        "member_seeds": [67_540, 67_541],
        "finite": True,
        "member_diversity": 0.1,
        "equivariance_shifts": [1, 2, 3, 7, 17],
        "max_generator_equivariance_error": 0.2,
        "max_mean_head_equivariance_error": 0.0,
        "equivariance_required": False,
        "peak_cuda_GiB": 2.0,
        "gates": {
            "finite": True,
            "canonical_shape": True,
            "member_diversity": True,
            "network_evaluations": True,
            "peak_memory": True,
            "required_equivariance": True,
        },
        "all_mechanical_gates_passed": True,
        "scientific_result": False,
        "physics_metric_evaluated": False,
        "held_out_85606_read": False,
    }
    tracking = {
        "required": True,
        "mode": "online",
        "remote_presence_verified_after_finish": True,
        "remote_state_after_finish": "finished",
        "checkpoints_uploaded": False,
        "samples_uploaded": False,
        "epochs_logged": 1,
        "run_url": "https://wandb.ai/example/project/runs/run-id",
        "spec": {
            "entity": "example",
            "project": "project",
            "group": "ecrd-smoke",
            "job_type": "ecrd_smoke_training",
        },
    }
    _write_json(run / "result.json", result)
    _write_json(run / "smoke_probe.json", probe)
    _write_json(run / "wandb.json", tracking)
    manifest = {
        "execution": {
            "wandb_entity": "example",
            "wandb_project": "project",
            "wandb_group": "ecrd-smoke",
        }
    }
    return run, manifest


def test_validate_smoke_run_is_mechanical_and_hash_checked(tmp_path: Path) -> None:
    manifest_sha256 = "f" * 64
    run, manifest = _synthetic_b5_run(
        tmp_path, commit="abc", manifest_sha256=manifest_sha256
    )
    record = SUMMARY.validate_smoke_run(
        arm="B5",
        run=run,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        paper0_commit="abc",
    )
    assert record["mechanical_probe"]["all_mechanical_gates_passed"] is True
    assert record["checkpoint_reload_bitwise_exact"] is True
    assert record["scientific_result"] is False


def test_smoke_run_rejects_a_held_out_value(tmp_path: Path) -> None:
    manifest_sha256 = "f" * 64
    run, manifest = _synthetic_b5_run(
        tmp_path, commit="abc", manifest_sha256=manifest_sha256
    )
    tracking_path = run / "wandb.json"
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    tracking["run_url"] = "https://example.invalid/85606"
    _write_json(tracking_path, tracking)
    with pytest.raises(RuntimeError, match="held-out"):
        SUMMARY.validate_smoke_run(
            arm="B5",
            run=run,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            paper0_commit="abc",
        )


def test_run_arguments_require_all_arms_in_frozen_order(tmp_path: Path) -> None:
    values = [
        f"B5={tmp_path / 'b5'}",
        f"B5-Context={tmp_path / 'context'}",
        f"ECRD={tmp_path / 'ecrd'}",
        f"ECRD-History={tmp_path / 'history'}",
    ]
    assert tuple(SUMMARY.parse_run_arguments(values)) == (
        "B5",
        "B5-Context",
        "ECRD",
        "ECRD-History",
    )
    with pytest.raises(ValueError, match="frozen arm order"):
        SUMMARY.parse_run_arguments(list(reversed(values)))


def test_finalizer_source_has_no_scientific_metric_dependency() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "from tcv_diagnostics.transport" not in source
    assert "from tcv_diagnostics.spect" not in source
    assert "from tcv_diagnostics.assimilat" not in source
    assert '"physics_metric_evaluated": False' in source
    assert '"full_training_authorized": False' in source
