"""Static safety checks for matched state-view reduction on Rusty."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "cluster/post_ecrd_old_85604_matched_state_reduce_freeze.sbatch"
)


def test_launcher_is_short_cpu_only_reduction() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --cpus-per-task=2" in source
    assert "#SBATCH --mem=8G" in source
    assert "#SBATCH --time=00:20:00" in source
    assert "--gres=gpu" not in source
    assert "--array=" not in source


def test_launcher_keeps_training_and_reduction_commits_separate() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'readonly TRAINING_COMMIT="6bb5e8ed75540287dd917f1b130aa4102051a9d7"' in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert 'status --porcelain --untracked-files=all' in source
    assert "reduce_matched_state_multilead.py" in source
    assert "freeze_matched_state_bounded_generation.py" in source


def test_launcher_does_not_turn_scientific_failure_into_generation() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '.paired_physics_evaluation_authorized == true' in source
    assert (
        '.decision == "stop_before_paired_physics_and_record_transition_failure"'
        in source
    )
    assert "no generation manifest created" in source
    assert '.held_out_85606_access_allowed == false' in source
    assert '.new_nersc_data_access_allowed == false' in source
    assert "/85606/" not in source
