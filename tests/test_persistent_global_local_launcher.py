from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "cluster/post_ecrd_old_85604_persistent_global_local_smoke.sbatch"
)
PILOT_LAUNCHER = (
    ROOT / "cluster/post_ecrd_old_85604_persistent_global_local_pilot.sbatch"
)


def test_launcher_is_one_right_sized_preemptible_gpu_job() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in text
    assert "#SBATCH --qos=gpupreempt" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --mem=20G" in text
    assert "#SBATCH --time=01:00:00" in text
    assert "#SBATCH --no-requeue" in text
    assert "h100" not in text.lower()


def test_launcher_is_fail_closed_and_requires_online_tracking() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "PAPER0_EXPECTED_COMMIT" in text
    assert "PAPER0_PGL_MANIFEST_SHA256" in text
    assert "status --porcelain --untracked-files=all" in text
    assert "WANDB_MODE=online" in text
    assert 'remote_state_after_finish == "finished"' in text
    assert "Refusing to overwrite" in text


def test_launcher_runs_only_smoke_and_keeps_closed_scope_assertions() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "--mode smoke" in text
    assert "train_persistent_global_local_pilot.py" in text
    assert ".physics_derived_loss_used == false" in text
    assert ".held_out_85606_read == false" in text
    assert ".new_nersc_data_read == false" in text
    assert ".physics_evaluation_authorized == false" in text
    assert "sbatch" not in text


def test_pilot_launcher_is_one_generic_preemptible_gpu_with_bounded_budget() -> None:
    text = PILOT_LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in text
    assert "#SBATCH --qos=gpupreempt" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --mem=24G" in text
    assert "#SBATCH --time=06:00:00" in text
    assert "h100" not in text.lower()
    assert "--mode pilot" in text
    assert ".completed_optimizer_steps == 4280" in text


def test_pilot_launcher_is_fail_closed_and_does_not_self_submit() -> None:
    text = PILOT_LAUNCHER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "PAPER0_EXPECTED_COMMIT" in text
    assert "PAPER0_PGL_MANIFEST_SHA256" in text
    assert "WANDB_MODE=online" in text
    assert 'remote_state_after_finish == "finished"' in text
    assert ".physics_derived_loss_used == false" in text
    assert ".held_out_85606_read == false" in text
    assert ".new_nersc_data_read == false" in text
    assert ".confirmation_seed_training_authorized == false" in text
    assert "sbatch" not in text
