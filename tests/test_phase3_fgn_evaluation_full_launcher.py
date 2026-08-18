"""Static fail-closed checks for the full Rocky 9 B3 evaluator launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b3_fgn_evaluation_full.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_b3_launcher_requires_training_and_passing_smoke() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --constraint=h100|h200" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "B3_TRAINING_JOB_ID" in source
    assert "B3_SMOKE_JOB_ID" in source
    assert "sha256sum -c \"${SMOKE_OUTPUT}/artifact_sha256.txt\"" in source
    assert "sha256sum -c \"${SMOKE_ROOT}/artifact_sha256.txt\"" in source
    assert "--mode full" in source
    assert "--smoke-result \"${SMOKE_RESULT}\"" in source
    assert "--member-batch-size 8" in source
    assert "WANDB_MODE=online" in source
    assert "pytest -p no:cacheprovider -q" in source
    assert "probabilistic_scientific_gate_evaluated" in source
    assert "O3 and 85606 remain closed" in source


def test_full_b3_launcher_pins_exact_evaluator_and_metric_sources() -> None:
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


def test_full_b3_launcher_preserves_scientific_noise_and_scope() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "--seed 1701" in source
    assert "job_6897493" in source
    assert "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb" in source
    assert "1449777a61d40af49ccb3bd6bed5edcba0fd8afe24d113e6175218c04865aa9c" in source
    assert "B3_FGN_H1_full_probabilistic_evaluation_85604" in source
    assert "B3_FGN_H1_truth_separated_probabilistic_scoring_85604" in source
    assert "target_count\"] != 126" in source
    assert "network_evaluations_per_member\"] != 1" in source
