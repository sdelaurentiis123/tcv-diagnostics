"""Known-answer tests for freezing the completed six-run O1 matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

from tcv_diagnostics.codec_training import CodecRunConfig, sha256_path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/freeze_o1_codec_training_matrix.py"
SPEC = importlib.util.spec_from_file_location("freeze_o1_matrix", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_run(root: Path) -> Path:
    run = root / "task_0_c5p_seed_1701"
    run.mkdir()
    config = CodecRunConfig.frozen(
        mode="full", codec="dcae_l20", family="c5p", seed=1701
    ).to_record()
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    history = []
    for epoch in range(200):
        history.append(
            {
                "epoch": epoch,
                "examples": 432,
                "validation_equal_channel_mae": 1.0 - epoch / 1000.0,
            }
        )
    (run / "history.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in history),
        encoding="utf-8",
    )
    (run / "selected.pt").write_bytes(b"selected")
    (run / "final_training_state.pt").write_bytes(b"final")
    result = {
        "scope": "O1_codec_full",
        "paper0_commit": "training-commit",
        "development_run": "85604",
        "held_out_85606_read": False,
        "completed_epochs": 200,
        "physics_derived_loss_used": False,
        "checkpoint_reload_bitwise_exact": True,
        "config": config,
        "selected_epoch": 199,
        "selected_global_step": 5400,
        "selected_validation_equal_channel_mae": history[199][
            "validation_equal_channel_mae"
        ],
        "selected_checkpoint": {"sha256": sha256_path(run / "selected.pt")},
        "final_training_state": {
            "sha256": sha256_path(run / "final_training_state.pt")
        },
    }
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    wandb = {
        "required": True,
        "mode": "online",
        "epochs_logged": 200,
        "remote_presence_verified_after_finish": True,
        "remote_state_after_finish": "finished",
        "local_artifacts_are_scientific_authority": True,
        "spec": {"run_id": "p0o1r1-6893802-0"},
        "run_url": "https://wandb.invalid/run",
    }
    (run / "wandb.json").write_text(json.dumps(wandb), encoding="utf-8")
    lines = [
        f"{sha256_path(run / name)}  {run / name}\n"
        for name in MODULE.ARTIFACTS
    ]
    (run / "artifact_sha256.txt").write_text("".join(lines), encoding="utf-8")
    return run


def test_freeze_run_rederives_earliest_best_epoch_and_hashes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run = _write_run(Path(temporary))
        result = MODULE.freeze_run(
            run,
            run_index=0,
            family="c5p",
            seed=1701,
            training_commit="training-commit",
            training_slurm_job_id="6893802",
        )
        assert result["selected_epoch"] == 199
        assert result["selected_checkpoint"]["sha256"] == sha256_path(
            run / "selected.pt"
        )
        assert result["wandb"]["remote_state"] == "finished"


def test_freeze_run_refuses_a_late_nonminimal_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run = _write_run(Path(temporary))
        result_path = run / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["selected_epoch"] = 198
        result_path.write_text(json.dumps(result), encoding="utf-8")
        lines = [
            f"{sha256_path(run / name)}  {run / name}\n"
            for name in MODULE.ARTIFACTS
        ]
        (run / "artifact_sha256.txt").write_text("".join(lines), encoding="utf-8")
        try:
            MODULE.freeze_run(
                run,
                run_index=0,
                family="c5p",
                seed=1701,
                training_commit="training-commit",
                training_slurm_job_id="6893802",
            )
        except ValueError as error:
            assert "checkpoint selection" in str(error)
        else:
            raise AssertionError("a nonminimal selected epoch was accepted")
