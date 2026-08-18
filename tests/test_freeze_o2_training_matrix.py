"""Known-answer tests for freezing the completed six-run O2 matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

import pytest

from tcv_diagnostics.codec_training import sha256_path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/freeze_o2_training_matrix.py"
SPEC = importlib.util.spec_from_file_location("freeze_o2_matrix", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rewrite_index(run: Path) -> None:
    lines = [
        f"{sha256_path(run / name)}  {run / name}\n"
        for name in MODULE.ARTIFACTS
    ]
    (run / "artifact_sha256.txt").write_text("".join(lines), encoding="utf-8")


def _write_run(root: Path) -> tuple[Path, Path, str]:
    codec = root / "codec.pt"
    codec.write_bytes(b"codec")
    codec_sha = sha256_path(codec)
    run = root / "task_0_c5p_h1_seed_1701"
    run.mkdir()
    config = MODULE._expected_config(
        arm="C5P-H1",
        seed=1701,
        codec_checkpoint_path=str(codec),
        codec_checkpoint_sha256=codec_sha,
    )
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")

    history = []
    for epoch in range(200):
        channels = {
            "Ne": 0.5,
            "Pe": 0.4,
            "Pi": 0.3,
            "phi": 0.2,
            "Vi": 0.1,
        }
        history.append(
            {
                "epoch": epoch,
                "examples": 430,
                "global_step": (epoch + 1) * 27,
                "learning_rate": 2.0e-4,
                "train_equal_channel_mae": 1.1 - epoch / 1000.0,
                "validation_equal_channel_mae": 1.0 - epoch / 1000.0,
                "validation_mae_by_channel": channels,
                "mean_preclip_gradient_norm": 0.2,
                "maximum_preclip_gradient_norm": 0.3,
                "selected_so_far": epoch,
                "epoch_wall_seconds": 1.0,
            }
        )
    (run / "history.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in history),
        encoding="utf-8",
    )
    latent = {
        "kind": "per_latent_channel_training_only_population_moments",
        "fit_frames": [0, 432],
        "sample_count_per_channel": 1_216_512,
        "codec_checkpoint_sha256": codec_sha,
        "scientific_authority": True,
        "held_out_85606_read": False,
    }
    (run / "latent_normalization.json").write_text(
        json.dumps(latent), encoding="utf-8"
    )
    (run / "selected.pt").write_bytes(b"selected")
    (run / "final_training_state.pt").write_bytes(b"final")
    result = {
        "scope": "O2_teacher_forced_one_step_full",
        "paper0_commit": "training-commit",
        "slurm_job_id": "6894980:run0:gpu0",
        "development_run": "85604",
        "held_out_85606_read": False,
        "completed_epochs": 200,
        "completed_optimizer_steps": 5400,
        "physics_derived_loss_used": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "checkpoint_reload_bitwise_exact": True,
        "O2_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
        "config": config,
        "selected_epoch": 199,
        "selected_validation_equal_channel_mae": history[199][
            "validation_equal_channel_mae"
        ],
        "final_validation_equal_channel_mae": history[199][
            "validation_equal_channel_mae"
        ],
        "final_validation_mae_by_channel": history[199][
            "validation_mae_by_channel"
        ],
        "selected_checkpoint": {"sha256": sha256_path(run / "selected.pt")},
        "final_training_state": {
            "sha256": sha256_path(run / "final_training_state.pt")
        },
        "history": {"sha256": sha256_path(run / "history.jsonl")},
        "latent_normalization": {
            "sha256": sha256_path(run / "latent_normalization.json")
        },
        "parameter_count": 51_612_800,
        "peak_cuda_bytes": 1234,
        "wall_seconds": 100.0,
    }
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    wandb = {
        "required": True,
        "mode": "online",
        "epochs_logged": 200,
        "remote_presence_verified_after_finish": True,
        "remote_state_after_finish": "finished",
        "local_artifacts_are_scientific_authority": True,
        "spec": {
            "run_id": "p0o2full-6894980-0",
            "group": "o2-c5p-l10-full",
        },
        "run_url": "https://wandb.invalid/run",
    }
    (run / "wandb.json").write_text(json.dumps(wandb), encoding="utf-8")
    _rewrite_index(run)
    return run, codec, codec_sha


def _freeze(run: Path, codec: Path, codec_sha: str) -> dict:
    return MODULE.freeze_run(
        run,
        run_index=0,
        arm="C5P-H1",
        slug="c5p_h1",
        seed=1701,
        context_frames=1,
        gpu_index=0,
        codec_checkpoint_path=str(codec),
        codec_checkpoint_sha256=codec_sha,
        training_commit="training-commit",
        training_slurm_job_id="6894980",
    )


def test_freeze_o2_run_rederives_checkpoint_and_hashes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run, codec, codec_sha = _write_run(Path(temporary))
        frozen = _freeze(run, codec, codec_sha)
        assert frozen["selected_epoch"] == 199
        assert frozen["selected_global_step"] == 5400
        assert frozen["selected_checkpoint"]["sha256"] == sha256_path(
            run / "selected.pt"
        )
        assert frozen["codec_checkpoint"]["trainable_during_O2"] is False
        assert frozen["wandb"]["remote_state"] == "finished"


def test_freeze_o2_run_refuses_nonminimal_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run, codec, codec_sha = _write_run(Path(temporary))
        result_path = run / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["selected_epoch"] = 198
        result_path.write_text(json.dumps(result), encoding="utf-8")
        _rewrite_index(run)
        with pytest.raises(ValueError, match="checkpoint selection"):
            _freeze(run, codec, codec_sha)


def test_freeze_o2_run_refuses_future_truth_or_physics_loss() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run, codec, codec_sha = _write_run(Path(temporary))
        result_path = run / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["target_truth_used_as_model_input"] = True
        result_path.write_text(json.dumps(result), encoding="utf-8")
        _rewrite_index(run)
        with pytest.raises(ValueError, match="completion contract"):
            _freeze(run, codec, codec_sha)


def test_freeze_o2_run_requires_authoritative_training_only_latent_moments() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run, codec, codec_sha = _write_run(Path(temporary))
        latent_path = run / "latent_normalization.json"
        latent = json.loads(latent_path.read_text(encoding="utf-8"))
        latent["scientific_authority"] = False
        latent_path.write_text(json.dumps(latent), encoding="utf-8")
        result_path = run / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["latent_normalization"]["sha256"] = sha256_path(latent_path)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        _rewrite_index(run)
        with pytest.raises(ValueError, match="latent normalization"):
            _freeze(run, codec, codec_sha)
