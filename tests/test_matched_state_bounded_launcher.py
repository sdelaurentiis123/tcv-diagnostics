"""Static safety checks for matched-state bounded generation on Rusty."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "cluster/post_ecrd_old_85604_matched_state_bounded_generation.sbatch"
)


def test_launcher_is_a_right_sized_two_arm_inference_array() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=24G" in source
    assert "#SBATCH --time=04:00:00" in source
    assert "#SBATCH --array=0-1%2" in source
    assert "--constraint=" not in source
    assert 'readonly FAMILIES=(c5p e6b)' in source


def test_launcher_prohibits_training_truth_and_new_data() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "PAPER0_MATCHED_GENERATION_MANIFEST_SHA256" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "generate_matched_state_bounded_forecasts.py" in source
    assert "WANDB_MODE=online" in source
    assert ".training_performed == false" in source
    assert ".checkpoint_selection_performed == false" in source
    assert ".target_truth_used_during_generation == false" in source
    assert ".held_out_85606_read == false" in source
    assert ".new_nersc_data_read == false" in source
