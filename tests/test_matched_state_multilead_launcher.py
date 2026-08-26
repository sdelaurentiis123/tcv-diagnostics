"""Static safety checks for the matched old-85604 state-view launch."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/post_ecrd_old_85604_matched_state_multilead.sbatch"


def test_launcher_uses_two_small_independent_gpu_tasks() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=12G" in source
    assert "#SBATCH --time=04:00:00" in source
    assert "#SBATCH --array=0-1%2" in source
    assert "--constraint=" not in source
    assert 'readonly FAMILIES=(c5p e6b)' in source


def test_launcher_is_fail_closed_and_online() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "PAPER0_MATCHED_STATE_MANIFEST_SHA256" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "Refusing to overwrite existing matched-state output" in source
    assert "train_matched_state_multilead.py" in source
    assert "WANDB_MODE=online" in source
    assert "WANDB_REQUIRE_SERVICE=true" in source
    assert ".optimizer_updates == 6396" in source
    assert ".held_out_85606_read == false" in source
    assert ".new_nersc_data_read == false" in source
    assert ".physics_derived_loss_used == false" in source
