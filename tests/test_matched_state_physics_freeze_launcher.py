"""Static safety checks for the matched-state physics freeze launcher."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "cluster/post_ecrd_old_85604_matched_state_physics_freeze.sbatch"
)


def test_freeze_launcher_is_small_cpu_only_and_prospective() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --ntasks=1" in source
    assert "#SBATCH --cpus-per-task=2" in source
    assert "#SBATCH --mem=8G" in source
    assert "#SBATCH --time=00:20:00" in source
    assert "--gres=gpu" not in source
    assert "score_matched_state_bounded_physics.py" not in source
    assert "freeze_matched_state_physics_scoring.py" in source


def test_freeze_launcher_locks_inputs_and_prohibits_expansion() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "PAPER0_GENERATION_MANIFEST_SHA256",
        "PAPER0_C5P_GENERATION_RESULT_SHA256",
        "PAPER0_E6B_GENERATION_RESULT_SHA256",
        "PAPER0_EXACT_PHI_RESULT_SHA256",
        "status --porcelain --untracked-files=all",
        '.held_out_85606_access_allowed == false',
        '.new_nersc_data_access_allowed == false',
        '.training_allowed == false',
        '.physics_derived_training_loss_allowed == false',
        '.assimilation_allowed == false',
        '.diagnostic_ranking_allowed == false',
        '.steering_allowed == false',
    ):
        assert required in source
    assert "/85606/" not in source
