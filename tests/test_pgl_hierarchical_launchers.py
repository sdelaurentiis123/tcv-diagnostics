from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "cluster/post_ecrd_old_85604_pgl_hierarchical_preflight.sbatch"
SMOKE = ROOT / "cluster/post_ecrd_old_85604_pgl_hierarchical_smoke.sbatch"
SCREEN = ROOT / "cluster/post_ecrd_old_85604_pgl_hierarchical_screen.sbatch"
FORECAST = ROOT / "cluster/post_ecrd_old_85604_pgl_hierarchical_forecast.sbatch"
SCORE = ROOT / "cluster/post_ecrd_old_85604_pgl_hierarchical_score.sbatch"
REDUCE = ROOT / "cluster/post_ecrd_old_85604_pgl_hierarchical_reduce.sbatch"


def _text(path: Path) -> str:
    subprocess.run(["bash", "-n", str(path)], check=True)
    return path.read_text(encoding="utf-8")


def test_launchers_are_bounded_one_gpu_jobs() -> None:
    preflight = _text(PREFLIGHT)
    smoke = _text(SMOKE)
    screen = _text(SCREEN)
    for source in (preflight, smoke, screen):
        assert "#SBATCH --partition=gpu" in source
        assert "#SBATCH --qos=gen" in source
        assert "#SBATCH --gres=gpu:h100_pcie:1" in source
        assert "#SBATCH --mem=32G" in source
        assert "#SBATCH --no-requeue" in source
        assert "sbatch" not in source
    assert "#SBATCH --time=03:00:00" in preflight
    assert "#SBATCH --time=01:00:00" in smoke
    assert "#SBATCH --time=06:00:00" in screen
    assert "#SBATCH --array=0-1%2" in smoke
    assert "#SBATCH --array=0-1%2" in screen


def test_launchers_fail_closed_and_keep_the_data_scope_closed() -> None:
    for path in (PREFLIGHT, SMOKE, SCREEN):
        source = _text(path)
        assert "set -euo pipefail" in source
        assert "PAPER0_EXPECTED_COMMIT" in source
        assert "status --porcelain --untracked-files=all" in source
        assert "PAPER0_PGL_EVIDENCE_MANIFEST_SHA256" in source
        assert ".held_out_85606_read == false" in source
        assert ".new_nersc_data_read == false" in source
        assert "Refusing to overwrite" in source
        assert 'PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"' in source


def test_training_chain_requires_preflight_smoke_and_online_wandb() -> None:
    preflight = _text(PREFLIGHT)
    smoke = _text(SMOKE)
    screen = _text(SCREEN)
    assert "PAPER0_PGL_HIERARCHICAL_PROTOCOL_SHA256" in preflight
    assert "PAPER0_PGL_PRIOR_PREFLIGHT_SHA256" in preflight
    assert ".gradient_calibration.observed_ratio >= 0.2499" in preflight
    assert "PAPER0_PGL_HIERARCHICAL_PREFLIGHT_SHA256" in smoke
    assert "PAPER0_PGL_HIERARCHICAL_SMOKE_JOB_ID" in screen
    for source in (smoke, screen):
        assert "WANDB_MODE=online" in source
        assert 'remote_state_after_finish == "finished"' in source
        assert "ARMS=(CONTROL TRANSPORT)" in source
    assert ".completed_optimizer_updates == 1" in smoke
    assert ".completed_optimizer_updates == 428" in screen
    assert "[107,214,428]" in screen


def test_evaluation_is_all_six_fixed_checkpoints_and_truth_separated() -> None:
    forecast = _text(FORECAST)
    score = _text(SCORE)
    assert "#SBATCH --array=0-5%3" in forecast
    assert "#SBATCH --array=0-5%3" in score
    for source in (forecast, score):
        assert "ARMS=(CONTROL TRANSPORT)" in source
        assert "UPDATES=(107 214 428)" in source
        assert "PAPER0_PGL_HIERARCHICAL_SCREEN_JOB_ID" in source
        assert ".held_out_85606_read == false" in source
        assert ".new_nersc_data_read == false" in source
        assert "WANDB_MODE=online" in source
        assert "status --porcelain --untracked-files=all" in source
        assert "Refusing to overwrite" in source
        assert "sbatch" not in source
    assert "#SBATCH --gres=gpu:h100_pcie:1" in forecast
    assert "#SBATCH --mem=32G" in score
    assert "#SBATCH --time=06:00:00" in score
    assert ".target_truth_read == false" in forecast
    assert ".physics_diagnostics_scored == false" in forecast
    assert "PAPER0_PGL_HIERARCHICAL_GENERATION_JOB_IDS" in score
    assert "PAPER0_PGL_HIERARCHICAL_GENERATION_COMMIT" in score
    assert '--generation-paper0-commit "${PAPER0_PGL_HIERARCHICAL_GENERATION_COMMIT}"' in score
    assert "Expected exactly six generation source job IDs" in score
    assert 'readonly GENERATION_JOB_ID="${GENERATION_JOB_IDS[${SLURM_ARRAY_TASK_ID}]}"' in score
    assert ".target_truth_used_during_generation == false" in score


def test_reducer_is_tiny_cpu_only_and_cannot_train_or_select() -> None:
    source = _text(REDUCE)
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --cpus-per-task=1" in source
    assert "#SBATCH --mem=2G" in source
    assert "#SBATCH --time=00:15:00" in source
    assert "PAPER0_PGL_HIERARCHICAL_SCORE_JOB_ID" in source
    assert ".checkpoint_selection_performed == false" in source
    assert ".held_out_85606_read == false" in source
    assert ".new_nersc_data_read == false" in source
    assert "sbatch" not in source
