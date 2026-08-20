"""Contract tests for the prospective ECRD evaluation-freeze builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from paper0.tools.freeze_ecrd_scientific_evaluation import (
    CODE_LOCKS,
    NEW_RUNS,
    TRAINING_ARRAY_JOB_ID,
    TRAINING_COMMIT,
    audit_training_finalization,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/freeze_ecrd_scientific_evaluation.py"
LAUNCHER = ROOT / "cluster/ecrd_freeze_scientific_evaluation.sbatch"


def _finalization() -> dict[str, object]:
    return {
        "scope": "ECRD_matched_full_training_finalization_85604",
        "status": "all_eleven_new_training_tasks_verified",
        "training_commit": TRAINING_COMMIT,
        "array_job_id": TRAINING_ARRAY_JOB_ID,
        "development_run": "85604",
        "total_ladder_runs": 12,
        "all_training_artifact_indices_verified": True,
        "all_wandb_runs_finished": True,
        "paired_training_order_verified": True,
        "paired_validation_seed_bank_verified": True,
        "scientific_result": False,
        "physics_metric_evaluated": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "new_runs": [
            {"array_index": index, "arm": arm, "seed": seed}
            for index, arm, seed in NEW_RUNS
        ],
    }


def test_freeze_requires_the_exact_verified_training_matrix() -> None:
    result = _finalization()
    assert len(audit_training_finalization(result)) == 11
    result["new_runs"] = list(result["new_runs"])[::-1]
    with pytest.raises(RuntimeError, match="matrix"):
        audit_training_finalization(result)


def test_freeze_rejects_partial_or_scientifically_scored_training() -> None:
    result = _finalization()
    result["all_wandb_runs_finished"] = False
    with pytest.raises(RuntimeError, match="contract"):
        audit_training_finalization(result)
    result = _finalization()
    result["physics_metric_evaluated"] = True
    with pytest.raises(RuntimeError, match="contract"):
        audit_training_finalization(result)


def test_freeze_locks_all_existing_evaluation_implementations() -> None:
    assert set(CODE_LOCKS) == {
        "paper0/tools/evaluate_ecrd_checkpoint.py",
        "paper0/tools/summarize_ecrd_model_ladder.py",
        "src/tcv_diagnostics/ecrd_forecast.py",
        "src/tcv_diagnostics/ecrd_scoring.py",
        "src/tcv_diagnostics/ecrd_acceptance.py",
        "src/tcv_diagnostics/ecrd_training.py",
        "src/tcv_diagnostics/models/ecrd.py",
    }
    assert all((ROOT / relative).is_file() for relative in CODE_LOCKS)


def test_freeze_is_metadata_only_and_keeps_downstream_closed() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "load_official_catalog" not in source
    assert "NativeTruthCatalog" not in source
    assert "score_ecrd_forecast" not in source
    assert '"held_out_85606_access_allowed": False' in source
    assert '"scientific_forecast_generated": False' in source
    assert '"assimilation_authorized": False' in source
    assert '"diagnostic_ranking_authorized": False' in source


def test_freeze_launcher_is_cpu_only_and_dependency_ready() -> None:
    import hashlib

    source = LAUNCHER.read_text(encoding="utf-8")
    tool_sha = hashlib.sha256(TOOL.read_bytes()).hexdigest()
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --gres" not in source
    assert 'readonly TRAINING_FINALIZER_JOB_ID="6913439"' in source
    assert tool_sha in source
    assert "freeze_ecrd_scientific_evaluation.py" in source
    assert "score_ecrd_forecast" not in source
    assert "evaluate_ecrd_checkpoint.py" not in source
