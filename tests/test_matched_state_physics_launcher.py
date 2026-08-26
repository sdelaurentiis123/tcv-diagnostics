"""Static safety checks for matched state-view physics scoring on Rusty."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/post_ecrd_old_85604_matched_state_physics.sbatch"


def test_launcher_is_cpu_only_and_has_a_bounded_allocation() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --ntasks=1" in source
    assert "#SBATCH --cpus-per-task=8" in source
    assert "#SBATCH --mem=96G" in source
    assert "#SBATCH --time=08:00:00" in source
    assert "--gres=gpu" not in source
    assert "--array=" not in source


def test_launcher_is_hash_locked_evaluation_only_and_online() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "PAPER0_MATCHED_PHYSICS_MANIFEST_SHA256" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "score_matched_state_bounded_physics.py" in source
    assert "WANDB_MODE=online" in source
    assert '.training_performed == false' in source
    assert '.checkpoint_selection_performed == false' in source
    assert '.physics_derived_loss_used == false' in source
    assert '.held_out_85606_read == false' in source
    assert '.new_nersc_data_read == false' in source
    assert '.assimilation_performed == false' in source
    assert '.diagnostic_ranking_performed == false' in source
    assert '.steering_performed == false' in source
    assert "/85606/" not in source
