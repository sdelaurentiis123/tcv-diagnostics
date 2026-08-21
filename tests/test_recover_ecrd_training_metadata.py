"""Tests for the narrow ECRD terminal-metadata recovery path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tcv_diagnostics.ecrd_training import frozen_parameter_counts


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/recover_ecrd_training_metadata.py"
LAUNCHER = ROOT / "cluster/ecrd_training_metadata_recovery.sbatch"
SPEC = importlib.util.spec_from_file_location("recover_ecrd_training_metadata", TOOL)
assert SPEC is not None and SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def _result() -> dict:
    return {
        "selected_completed_epoch": 75,
        "selected_validation": {"checkpoint_score": 1.707327701742687},
        "artifacts": {
            "selected_checkpoint": {"sha256": "a" * 64},
            "history": {"sha256": "b" * 64},
        },
    }


def test_recovery_constants_are_exactly_one_completed_source_task() -> None:
    assert RECOVERY.SOURCE_SLURM_JOB_ID == "6913340_8"
    assert RECOVERY.ARM == "ECRD-History"
    assert RECOVERY.SEED == 1701
    assert RECOVERY.WANDB_RUN_ID == "p0ecrdfull-6913340-8-s1701"
    assert len(RECOVERY.expected_artifact_relatives()) == 27


def test_write_or_verify_json_never_replaces_differing_metadata(tmp_path: Path) -> None:
    path = tmp_path / "wandb.json"
    first = {"state": "finished", "epochs": 100}
    digest = RECOVERY.write_or_verify_json(path, first)
    assert len(digest) == 64
    assert json.loads(path.read_text(encoding="utf-8")) == first
    assert RECOVERY.write_or_verify_json(path, first) == digest
    with pytest.raises(FileExistsError, match="refusing to replace"):
        RECOVERY.write_or_verify_json(path, {"state": "running"})
    assert json.loads(path.read_text(encoding="utf-8")) == first


def test_live_tracking_audit_requires_exact_finished_run_and_summary() -> None:
    result = _result()
    summary = {
        "final/training_completed": True,
        "final/completed_epochs": 100,
        "final/completed_optimizer_steps": 10_800,
        "final/candidate_count": 20,
        "final/selected_completed_epoch": 75,
        "final/selected_validation_objective": 1.707327701742687,
        "final/checkpoint_reload_bitwise_exact": True,
        "compute/parameter_count": frozen_parameter_counts()["ECRD-History"],
        "provenance/paper0_commit": RECOVERY.SOURCE_TRAINING_COMMIT,
        "provenance/selected_checkpoint_sha256": "a" * 64,
        "provenance/history_sha256": "b" * 64,
        "scope/physics_derived_loss_used": False,
        "scope/held_out_85606_read": False,
        "scope/scientific_forecast_generated": False,
    }
    remote = SimpleNamespace(
        id=RECOVERY.WANDB_RUN_ID,
        state="finished",
        url="https://wandb.ai/example/run",
        summary=summary,
    )
    api = SimpleNamespace(
        api_key="configured",
        viewer=SimpleNamespace(
            entity=RECOVERY.WANDB_ENTITY,
            username="sdelaurentiis123",
        ),
        run=lambda path: remote,
    )
    module = SimpleNamespace(__version__="test", Api=lambda timeout: api)
    record = RECOVERY.build_tracking_record(module=module, result=result)
    assert record["remote_state_after_finish"] == "finished"
    assert record["epochs_logged"] == 100
    summary["final/completed_optimizer_steps"] = 10_799
    with pytest.raises(RuntimeError, match="optimizer_steps"):
        RECOVERY.build_tracking_record(module=module, result=result)


def test_recovery_launcher_is_cpu_only_narrow_and_hash_locked() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --mem=16G" in source
    assert 'readonly SOURCE_TASK_ID="8"' in source
    assert "recover_ecrd_training_metadata.py" in source
    assert "evaluate_ecrd_checkpoint.py" not in source
    assert "train_ecrd.py" not in source
    assert "REPLACE_ENTRYPOINT_SHA256" not in source
    assert "sha256sum -c \"${SOURCE_MODEL_ROOT}/artifact_sha256.txt\"" in source


def test_recovery_tool_contains_no_training_or_evaluation_entrypoint() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "train_ecrd_arm(" not in source
    assert "generate_selected_ecrd_forecasts(" not in source
    assert "score_ecrd_forecast(" not in source
    assert '"training_rerun": False' in source
    assert '"checkpoint_bytes_modified": False' in source
    assert '"held_out_85606_read": False' in source
