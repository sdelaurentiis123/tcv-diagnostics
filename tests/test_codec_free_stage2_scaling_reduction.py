"""Known-answer checks for the Stage-2 three-seed reducer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paper0.tools.reduce_codec_free_stage2_scaling import reduce_scaling
from paper0.tools.train_codec_free_stage2_multilead import FIELDS, LEADS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return _sha(path)


def _validation(offset: float) -> dict:
    return {
        "mean_shared_persistence_normalized_mse_ratio": 0.5 + offset,
        "per_lead": {
            str(lead): {
                "shared_field_mean_model_derivative_mse": 0.01 / lead + offset,
                "shared_field_persistence_relative_skill": 0.4 + offset,
                "per_field": {
                    field: {"persistence_relative_skill": 0.3 + offset}
                    for field in FIELDS
                },
            }
            for lead in LEADS
        },
    }


def _result(
    *,
    seed: int,
    scope: str,
    checkpoint: Path,
    checkpoint_sha: str,
    manifest: Path | None,
    manifest_sha: str | None,
    commit: str,
    confirmation: bool,
) -> dict:
    offset = (seed - 1701) * 0.01
    record = {
        "scope": scope,
        "status": "passed",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "physics_derived_loss_used": False,
        "family": "c5p",
        "seed": seed,
        "paper0_commit": commit,
        "training_pair_count": 2129,
        "validation_pair_count": 609,
        "epochs": 4,
        "optimizer_updates": 2132,
        "expected_optimizer_updates": 2132,
        "training_gate": {"passed": True},
        "best_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "epoch": 4,
            "selection_metric": 0.5 + offset,
        },
        "lead1_shared_mse": 0.005 + offset / 100,
        "parent_improvement_fraction": 0.9,
        "best_validation": _validation(offset),
    }
    if scope.endswith("screen"):
        record.update(
            {
                "advance_to_three_seed_scaling": confirmation,
                "screen_gates": {},
            }
        )
    else:
        record.update(
            {
                "manifest": str(manifest),
                "manifest_sha256": manifest_sha,
                "advance_to_three_seed_scaling": None,
                "prospective_gate_passed": confirmation,
                "seed_confirmation_passed": confirmation,
            }
        )
    return record


def _write_index(path: Path, root: Path, relatives: list[str]) -> None:
    path.write_text(
        "".join(
            f"{_sha(root / relative)}  {(root / relative).resolve()}\n"
            for relative in relatives
        ),
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
    *,
    failed_seed: int | None = None,
    aliased_recorded_manifest: bool = False,
) -> dict:
    root = tmp_path / "development_85604_reduction"
    root.mkdir()
    commit = "e" * 40
    manifest = root / "manifest.json"
    manifest_sha = _write_json(
        manifest,
        {
            "scope": "post_ecrd_old_85604_stage2_multilead_scaling",
            "development_run": "85604",
            "held_out_85606_access_allowed": False,
            "new_nersc_data_access_allowed": False,
            "all_seed_confirmation_required": True,
            "conditional_bounded_rollout_authorized": True,
            "paper0_commit_at_freeze": commit,
        },
    )
    recorded_manifest = manifest.resolve()
    if aliased_recorded_manifest:
        recorded_manifest = root / "manifest_alias.json"
        recorded_manifest.symlink_to(manifest.name)
    screen_checkpoint = root / "seed1701.pt"
    screen_checkpoint.write_bytes(b"seed1701")
    screen = root / "seed1701.json"
    screen_sha = _write_json(
        screen,
        _result(
            seed=1701,
            scope="post_ecrd_old_85604_stage2_multilead_screen",
            checkpoint=screen_checkpoint,
            checkpoint_sha=_sha(screen_checkpoint),
            manifest=None,
            manifest_sha=None,
            commit="old",
            confirmation=True,
        ),
    )
    array = root / "array_100"
    array.mkdir()
    for task, seed in ((1, 1702), (2, 1703)):
        task_root = array / f"task_{task}_seed_{seed}_job_{100 + task}"
        run = task_root / "run"
        run.mkdir(parents=True)
        for epoch in range(1, 5):
            (run / f"checkpoint_epoch_{epoch:03d}.pt").write_bytes(
                f"seed={seed},epoch={epoch}".encode()
            )
        selected = run / "checkpoint_epoch_004.pt"
        result = _result(
            seed=seed,
            scope="post_ecrd_old_85604_stage2_multilead_scaling",
            checkpoint=selected.resolve(),
            checkpoint_sha=_sha(selected),
            manifest=recorded_manifest,
            manifest_sha=manifest_sha,
            commit=commit,
            confirmation=seed != failed_seed,
        )
        _write_json(run / "result.json", result)
        _write_json(
            run / "wandb.json",
            {
                "required": True,
                "mode": "online",
                "remote_state_after_finish": "finished",
                "local_artifacts_are_scientific_authority": True,
                "run_url": f"https://wandb.invalid/{seed}",
            },
        )
        _write_json(run / "derivative_rms.json", {"seed": seed})
        _write_json(run / "parent_multilead_evaluation.json", {"seed": seed})
        run_relatives = [
            *(f"checkpoint_epoch_{epoch:03d}.pt" for epoch in range(1, 5)),
            "derivative_rms.json",
            "parent_multilead_evaluation.json",
            "result.json",
            "wandb.json",
        ]
        _write_index(run / "artifact_sha256.txt", run, run_relatives)
        for name in ("command.sh", "environment.txt", "slurm_job.txt", "test_output.txt"):
            (task_root / name).write_text(name, encoding="utf-8")
        task_relatives = [
            "command.sh",
            "environment.txt",
            "slurm_job.txt",
            "test_output.txt",
            "run/artifact_sha256.txt",
            "run/parent_multilead_evaluation.json",
            "run/result.json",
            "run/wandb.json",
        ]
        _write_index(task_root / "artifact_sha256.txt", task_root, task_relatives)
    return {
        "manifest_path": manifest,
        "manifest_sha256": manifest_sha,
        "seed1701_result_path": screen,
        "seed1701_result_sha256": screen_sha,
        "array_root": array,
        "training_commit": commit,
    }


def test_reducer_authorizes_rollout_only_when_all_seeds_pass(tmp_path: Path) -> None:
    result = reduce_scaling(**_fixture(tmp_path))
    assert result["three_seed_mechanism_confirmed"] is True
    assert result["bounded_rollout_authorized"] is True
    assert result["decision"] == "freeze_bounded_direct_vs_autoregressive_validation"
    assert set(result["by_seed"]) == {"1701", "1702", "1703"}
    assert result["aggregates"]["selection_metric"]["median"] == 0.51


def test_reducer_does_not_average_away_one_failed_seed(tmp_path: Path) -> None:
    result = reduce_scaling(**_fixture(tmp_path, failed_seed=1703))
    assert result["seed_confirmation_passed"]["1703"] is False
    assert result["three_seed_mechanism_confirmed"] is False
    assert result["bounded_rollout_authorized"] is False
    assert result["decision"] == "stop_multilead_schedule_and_freeze_operator_experiment"


def test_reducer_accepts_an_alias_to_the_same_frozen_manifest(
    tmp_path: Path,
) -> None:
    result = reduce_scaling(
        **_fixture(tmp_path, aliased_recorded_manifest=True)
    )
    assert result["three_seed_mechanism_confirmed"] is True
    assert result["bounded_rollout_authorized"] is True
