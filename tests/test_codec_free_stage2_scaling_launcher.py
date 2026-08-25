"""Static safety checks for the Stage-2 seed-confirmation launch."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/post_ecrd_old_85604_stage2_multilead_scaling.sbatch"


def test_launcher_uses_two_independent_small_gpu_tasks() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=12G" in source
    assert "#SBATCH --time=01:00:00" in source
    assert "#SBATCH --array=1-2%2" in source
    assert "--constraint=" not in source
    assert 'readonly SEEDS=(1702 1703)' in source
    assert "WANDB_MODE=online" in source


def test_launcher_is_fail_closed_and_does_not_gate_job_exit_on_science() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "PAPER0_SCALING_MANIFEST_SHA256" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "Refusing to overwrite existing multi-lead-scaling output" in source
    assert "Array seed is absent from frozen scaling manifest" in source
    assert "remote_state_after_finish" in source
    assert "seed_confirmation_passed" in source
    assert "jq -e '.seed_confirmation_passed" not in source
