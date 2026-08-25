"""Static safety checks for the old-85604 C5P multi-lead launch."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/post_ecrd_old_85604_stage2_multilead_screen.sbatch"


def test_launcher_is_one_right_sized_gpu_job_with_online_tracking() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=12G" in source
    assert "#SBATCH --time=01:00:00" in source
    assert "#SBATCH --array" not in source
    assert "--constraint=" not in source
    assert "WANDB_MODE=online" in source
    assert "WANDB_REQUIRE_SERVICE=true" in source


def test_launcher_is_fail_closed_and_c5p_only() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "PAPER0_MULTILEAD_MANIFEST_SHA256" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "Refusing to overwrite existing multi-lead-screen output" in source
    assert "train_codec_free_stage2_multilead.py" in source
    assert 'echo "family=c5p"' in source
    assert "remote_state_after_finish" in source
    assert "advance_to_three_seed_scaling" in source
