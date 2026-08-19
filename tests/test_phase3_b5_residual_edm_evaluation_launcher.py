"""Static checks for the frozen B5 four-target evaluator launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b5_residual_edm_evaluator_smoke.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_B5_evaluator_smoke_launcher_scope_and_cluster_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --constraint=h100" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "B5_TRAINING_JOB_ID" in source
    assert "TRAINING_COMMIT" in source
    assert "--mode smoke" in source
    assert "--seed 1701" in source
    assert "--member-batch-size 8" in source
    assert "WANDB_MODE=online" in source
    assert "pytest -p no:cacheprovider -q" in source
    assert "phase3_b5_residual_edm_evaluator_smoke" in source
    assert "the gate, O3, and 85606 remain closed" in source


def test_B5_evaluator_smoke_launcher_pins_new_sources_and_metric_engine() -> None:
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


def test_B5_evaluator_smoke_launcher_pins_truth_geometry_threshold_mean_and_seed() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "phase2_potential_vorticity_all_frame_6893033.json" in source
    assert "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" in source
    assert "phase2_85604_geometry_units.json" in source
    assert "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" in source
    assert "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" in source
    assert "job_6897493" in source
    assert "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb" in source
    assert "job_6896117/task_0_c5p_h1_seed_1701/forecast.h5" in source
    assert "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed" in source
    assert "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14" in source


def test_B5_evaluator_smoke_launcher_verifies_truth_separation_and_35_NFE() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'result["target_frames"] != [498, 502]' in source
    assert 'result["truth_opened_only_after_forecast_hash"] is not True' in source
    assert 'generation["inference"]["network_evaluations_per_member"] != 35' in source
    assert 'result["scientific_acceptance_evaluated"] is not False' in source
    assert 'result["O3_launch_allowed"] is not False' in source
    assert "artifact_sha256.txt" in source

