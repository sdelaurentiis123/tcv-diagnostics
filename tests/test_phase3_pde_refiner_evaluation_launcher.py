"""Static fail-closed checks for the Rocky 9 B4 evaluator smoke launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b4_pde_refiner_evaluator_smoke.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_b4_evaluator_smoke_launcher_scope_and_cluster_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --constraint=h100|h200" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "B4_TRAINING_JOB_ID" in source
    assert 'TRAINING_COMMIT="0350b063a1f7e5c6a00b76bf4d6aeaec71d511ef"' in source
    assert "--mode smoke" in source
    assert "--seed 1701" in source
    assert "--member-batch-size 8" in source
    assert "WANDB_MODE=online" in source
    assert "pytest -p no:cacheprovider -q" in source
    assert "phase3_b4_pde_refiner_evaluator_smoke" in source
    assert "O3, assimilation, and 85606 remain closed" in source


def test_b4_launcher_pins_new_sources_and_all_frozen_metric_sources() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    paths = (
        "paper0/tools/evaluate_b4_pde_refiner_checkpoint.py",
        "paper0/tools/run_b4_pde_refiner_evaluation_wandb.py",
        "src/tcv_diagnostics/pde_refiner_forecast.py",
        "src/tcv_diagnostics/pde_refiner_scoring.py",
        "src/tcv_diagnostics/pde_refiner_full_training.py",
        "src/tcv_diagnostics/pde_refiner_training.py",
        "src/tcv_diagnostics/models/pde_refiner.py",
        "src/tcv_diagnostics/b2_scoring.py",
        "src/tcv_diagnostics/b2_probabilistic_metrics.py",
        "src/tcv_diagnostics/b2_field_metrics.py",
        "src/tcv_diagnostics/b2_spectral_metrics.py",
        "src/tcv_diagnostics/b2_transport_metrics.py",
        "src/tcv_diagnostics/geometry.py",
        "src/tcv_diagnostics/codec_transport.py",
    )
    for relative in paths:
        assert _sha256(ROOT / relative) in source, relative
    assert _sha256(
        ROOT / "paper0/manifests/phase3_b4_full_evaluation_85604.json"
    ) in source
    assert _sha256(
        ROOT / "paper0/protocol/PHASE3_B4_FULL_TRAINING_EVALUATION_PROTOCOL.md"
    ) in source


def test_b4_launcher_pins_truth_geometry_threshold_data_and_seed_bank() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "phase2_potential_vorticity_all_frame_6893033.json" in source
    assert "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" in source
    assert "phase2_85604_geometry_units.json" in source
    assert "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" in source
    assert "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" in source
    assert "job_6897493" in source
    assert "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb" in source
    assert "a1871e069bce6244073bfe1aa835a53c1d7a59302b01f6a366b3dc88297b6205" in source
    assert "job_6893525" in source


def test_b4_launcher_requires_both_artifacts_prefix_and_no_scientific_gate() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "forecast_final_M32.h5" not in source  # paths are owned by the evaluator
    assert 'generation["final_forecast"]["shape"]' in source
    assert 'generation["stage_forecast"]["shape"]' in source
    assert "M4_stage3_bitwise_prefix_of_M32" in source
    assert "level0_bitwise_shared_across_members" in source
    assert 'result["H_det_evaluated"] is not False' in source
    assert 'result["H_prob_evaluated"] is not False' in source
