"""Static fail-closed checks for the Rocky 9 B3 evaluator smoke launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b3_fgn_evaluator_smoke.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_b3_evaluator_smoke_launcher_scope_and_cluster_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --constraint=h100|h200" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "B3_TRAINING_JOB_ID" in source
    assert 'TRAINING_COMMIT="a2a17cf3fc30fd504bc3eee3274e78623bf15e2b"' in source
    assert "--mode smoke" in source
    assert "--seed 1701" in source
    assert "--member-batch-size 8" in source
    assert "WANDB_MODE=online" in source
    assert "pytest -p no:cacheprovider -q" in source
    assert "phase3_b3_fgn_evaluator_smoke" in source
    assert "phase3_b3_fgn_full" in source
    assert "85606" in source and "remain closed" in source


def test_b3_evaluator_launcher_pins_every_new_source_and_frozen_metric() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    paths = (
        "paper0/tools/evaluate_b3_fgn_checkpoint.py",
        "paper0/tools/run_b3_fgn_evaluation_wandb.py",
        "src/tcv_diagnostics/fgn_forecast.py",
        "src/tcv_diagnostics/fgn_scoring.py",
        "src/tcv_diagnostics/fgn_training.py",
        "src/tcv_diagnostics/models/functional_noise.py",
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
        ROOT / "paper0/manifests/phase3_b3_full_evaluation_85604.json"
    ) in source
    assert _sha256(
        ROOT / "paper0/protocol/PHASE3_B3_FULL_EVALUATION_PROTOCOL.md"
    ) in source


def test_b3_evaluator_launcher_pins_shared_truth_geometry_threshold_and_noise() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "phase2_potential_vorticity_all_frame_6893033.json" in source
    assert "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" in source
    assert "phase2_85604_geometry_units.json" in source
    assert "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" in source
    assert "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" in source
    assert "job_6897493" in source
    assert "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb" in source
    assert "1449777a61d40af49ccb3bd6bed5edcba0fd8afe24d113e6175218c04865aa9c" in source
