"""Static safety checks for the old-85604 phi-plus-Vi launch."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "cluster/post_ecrd_old_85604_exact_state_derived_coordinate_screen.sbatch"
)


def test_launcher_is_one_small_gpu_job_with_online_tracking() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=12G" in source
    assert "#SBATCH --time=00:45:00" in source
    assert "#SBATCH --array" not in source
    assert "--constraint=" not in source
    assert "WANDB_MODE=online" in source
    assert "WANDB_REQUIRE_SERVICE=true" in source


def test_launcher_is_fail_closed_and_runs_only_frozen_phi_vi_arm() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "PAPER0_COORDINATE_MANIFEST_SHA256" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "Refusing to overwrite existing coordinate-screen output" in source
    assert 'readonly ARCHITECTURE="local_current_phi_vi"' in source
    assert "train_exact_state_phi_repair_screen.py" in source
    assert "remote_state_after_finish" in source
    assert "advance_to_three_seed_scaling" in source
