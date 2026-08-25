from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "cluster/post_ecrd_old_85604_persistent_global_local_forecast.sbatch"
SCORE = ROOT / "cluster/post_ecrd_old_85604_persistent_global_local_score.sbatch"


def test_forecast_launcher_is_one_unconstrained_preemptible_gpu_and_truth_free():
    source = FORECAST.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --constraint=" not in source
    assert "generate_persistent_global_local_forecast.py" in source
    assert "score_persistent_global_local_physics.py" not in source
    assert "target_truth_read == false" in source
    assert "physics_diagnostics_scored == false" in source
    assert "/85606" not in source


def test_score_launcher_is_cpu_only_and_consumes_closed_generation_dependency():
    source = SCORE.read_text(encoding="utf-8")
    assert "#SBATCH --partition=preempt" in source
    assert "#SBATCH --gres=gpu" not in source
    assert "PGL_GENERATION_JOB_ID" in source
    assert "forecast_M32_four_frame.h5" in source
    assert "sha256sum \"${FORECAST}\"" in source
    assert "score_persistent_global_local_physics.py" in source
    assert "target_truth_used_during_generation == false" in source
    assert "/85606" not in source
