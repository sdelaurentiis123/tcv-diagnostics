"""Static checks for the frozen full B5 M32 evaluation launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b5_residual_edm_evaluation_full.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_B5_full_evaluation_launcher_requires_training_and_smoke_chain() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --constraint=h100" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "B5_TRAINING_JOB_ID" in source
    assert "B5_SMOKE_JOB_ID" in source
    assert 'sha256sum -c "${TRAINING_OUTPUT}/artifact_sha256.txt"' in source
    assert 'sha256sum -c "${SMOKE_OUTPUT}/artifact_sha256.txt"' in source
    assert "--mode full" in source
    assert '--smoke-result "${SMOKE_RESULT}"' in source
    assert "--member-batch-size 8" in source
    assert "WANDB_MODE=online" in source
    assert "pytest -p no:cacheprovider -q" in source


def test_B5_full_evaluation_launcher_pins_identical_evaluator_and_metrics() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    paths = (
        "paper0/tools/evaluate_b5_residual_edm_checkpoint.py",
        "paper0/tools/run_b5_residual_edm_evaluation_wandb.py",
        "src/tcv_diagnostics/b5_residual_edm_forecast.py",
        "src/tcv_diagnostics/b5_residual_edm_scoring.py",
        "src/tcv_diagnostics/b5_residual_edm_full_training.py",
        "src/tcv_diagnostics/models/field_residual_edm.py",
        "src/tcv_diagnostics/b2_scoring.py",
        "src/tcv_diagnostics/b2_probabilistic_metrics.py",
        "src/tcv_diagnostics/b2_field_metrics.py",
        "src/tcv_diagnostics/b2_spectral_metrics.py",
        "src/tcv_diagnostics/b2_transport_metrics.py",
        "src/tcv_diagnostics/geometry.py",
        "src/tcv_diagnostics/codec_transport.py",
        "paper0/manifests/phase3_b5_full_training_evaluation_85604.json",
        "paper0/protocol/PHASE3_B5_FULL_TRAINING_EVALUATION_PROTOCOL.md",
    )
    for relative in paths:
        assert sha256(ROOT / relative) in source, relative


def test_B5_full_evaluation_launcher_uses_same_frozen_scientific_inputs() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "phase2_potential_vorticity_all_frame_6893033.json" in source
    assert "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" in source
    assert "phase2_85604_geometry_units.json" in source
    assert "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" in source
    assert "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" in source
    assert "job_6897493" in source
    assert "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb" in source
    assert "job_6896117/task_0_c5p_h1_seed_1701/forecast.h5" in source
    assert "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14" in source


def test_B5_full_evaluation_launcher_verifies_complete_truth_separated_result() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'result["target_frames"] != [498, 624]' in source
    assert 'result["target_count"] != 126' in source
    assert 'result["ensemble_members"] != 32' in source
    assert 'result["truth_opened_only_after_forecast_hash"] is not True' in source
    assert 'generation["inference"]["network_evaluations_per_member"] != 35' in source
    assert 'result["scientific_acceptance_evaluated"] is not False' in source
    assert 'result["O3_launch_allowed"] is not False' in source
    assert "The frozen one-seed gate remains unevaluated" in source

