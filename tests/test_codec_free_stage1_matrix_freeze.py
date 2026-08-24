"""Known-answer tests for freezing the completed Stage-1 result matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper0.tools.freeze_codec_free_stage1_matrix import freeze_matrix


TRAINING_COMMIT = "3" * 40
EVALUATION_COMMIT = "4" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_arm(root: Path, *, family: str, seed: int) -> None:
    arm = root / family
    arm.mkdir()
    checkpoint = arm / "checkpoint_epoch_012.pt"
    checkpoint.write_bytes(f"{family}-{seed}".encode())
    checkpoint_sha = _sha(checkpoint)
    result = {
        "scope": "post_ecrd_old_85604_stage1_codec_free_full",
        "development_run": "85604",
        "held_out_85606_read": False,
        "physics_derived_loss_used": False,
        "family": family,
        "seed": seed,
        "paper0_commit": TRAINING_COMMIT,
        "status": "passed",
        "training_gate": {"passed": True},
        "best_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "selection_metric": 0.1,
        },
        "history": [
            {
                "epoch": 12,
                "validation": {
                    "shared_field_mean_model_derivative_mse": 0.1
                },
            }
        ],
    }
    (arm / "result.json").write_text(json.dumps(result), encoding="utf-8")
    tracking = {
        "required": True,
        "mode": "online",
        "run_url": f"https://wandb.invalid/{family}/{seed}",
        "remote_path": f"entity/project/{family}-{seed}",
        "remote_state_after_finish": "finished",
    }
    (arm / "wandb.json").write_text(json.dumps(tracking), encoding="utf-8")


def _matrix(tmp_path: Path) -> Path:
    root = tmp_path / "development_85604_array"
    root.mkdir()
    for task, seed in ((0, 1701), (1, 1702), (2, 1703)):
        task_root = root / f"task_{task}_seed_{seed}_job_{7000 + task}"
        task_root.mkdir()
        (task_root / "artifact_sha256.txt").write_text(
            "bounded fixture\n", encoding="utf-8"
        )
        for family in ("c5p", "e6b"):
            _write_arm(task_root, family=family, seed=seed)
    return root


def test_freeze_matrix_locks_exactly_six_finished_arms(tmp_path: Path) -> None:
    frozen = freeze_matrix(
        array_root=_matrix(tmp_path),
        array_job_id="6933635",
        training_commit=TRAINING_COMMIT,
        evaluation_commit=EVALUATION_COMMIT,
    )
    assert len(frozen["arms"]) == 6
    assert {(arm["family"], arm["seed"]) for arm in frozen["arms"]} == {
        (family, seed)
        for family in ("c5p", "e6b")
        for seed in (1701, 1702, 1703)
    }
    assert frozen["held_out_85606_read"] is False
    assert frozen["blocks"]["V02"]["target_interval"] == [582, 624]


def test_freeze_matrix_rejects_unfinished_wandb_arm(tmp_path: Path) -> None:
    root = _matrix(tmp_path)
    tracking_path = next(root.glob("task_0_seed_1701_job_*/c5p/wandb.json"))
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    tracking["remote_state_after_finish"] = "running"
    tracking_path.write_text(json.dumps(tracking), encoding="utf-8")
    with pytest.raises(ValueError, match="W&B"):
        freeze_matrix(
            array_root=root,
            array_job_id="6933635",
            training_commit=TRAINING_COMMIT,
            evaluation_commit=EVALUATION_COMMIT,
        )


def test_freeze_matrix_rejects_checkpoint_hash_drift(tmp_path: Path) -> None:
    root = _matrix(tmp_path)
    checkpoint = next(root.glob("task_1_seed_1702_job_*/e6b/checkpoint_epoch_012.pt"))
    checkpoint.write_bytes(b"drift")
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        freeze_matrix(
            array_root=root,
            array_job_id="6933635",
            training_commit=TRAINING_COMMIT,
            evaluation_commit=EVALUATION_COMMIT,
        )
