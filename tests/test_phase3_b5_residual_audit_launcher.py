"""Static fail-closed checks for the Rocky 9 B5 residual-audit launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b5_h1_residual_audit.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_b5_launcher_is_one_h100_rocky9_online_audit_not_training() -> None:
    source = LAUNCHER.read_text()
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --constraint=h100|h200" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "Rocky Linux 9" in source
    assert "WANDB_MODE=online" in source
    assert "pytest -p no:cacheprovider -q" in source
    assert "training_performed" in source
    assert "B5_training_authorized" in source
    assert "No B5 model was trained" in source


def test_b5_launcher_pins_implementation_protocol_and_evidence_sources() -> None:
    source = LAUNCHER.read_text()
    paths = (
        "paper0/manifests/phase3_b5_residual_audit_85604.json",
        "paper0/protocol/PHASE3_B5_RESIDUAL_AUDIT_PROTOCOL.md",
        "paper0/tools/audit_b5_h1_training_residual.py",
        "paper0/tools/run_b5_residual_audit_wandb.py",
        "src/tcv_diagnostics/b5_residual_audit.py",
        "src/tcv_diagnostics/b5_residual_forecast.py",
        "src/tcv_diagnostics/o2_forecast.py",
        "src/tcv_diagnostics/o2_training_data.py",
        "src/tcv_diagnostics/model_training_data.py",
        "src/tcv_diagnostics/models/o2.py",
        "src/tcv_diagnostics/matched_o1_transport.py",
        "src/tcv_diagnostics/geometry.py",
        "src/tcv_diagnostics/codec_transport.py",
        "src/tcv_diagnostics/b2_field_metrics.py",
        "paper0/results/phase1_85604_profile_6890606.json",
        "paper0/results/phase3_b4_pde_refiner_one_seed_gate_6901285.json",
        "paper0/manifests/phase2_85604_geometry_units.json",
    )
    for relative in paths:
        assert _sha256(ROOT / relative) in source, relative


def test_b5_launcher_pins_parent_data_geometry_and_truth_separation() -> None:
    source = LAUNCHER.read_text()
    assert "job_6894980/task_0_c5p_h1_seed_1701/selected.pt" in source
    assert "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e" in source
    assert "job_6894463/task_0_c5p_seed_1701/selected.pt" in source
    assert "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d" in source
    assert "job_6893525" in source
    assert "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" in source
    assert 'result.get("forecast_closed_and_hashed_before_truth_read") is not True' in source
    assert 'audit.get("canonical_shape") != [430, 5, 64, 32, 88]' in source
    assert 'get("mode_mapping") != "n=5k"' in source


def test_b5_launcher_keeps_all_downstream_scopes_closed() -> None:
    source = LAUNCHER.read_text()
    assert "validation, 85606, O3, assimilation, and ranking remain closed" in source
    assert "--mode full" not in source
    assert "--mode smoke" not in source
    assert "--decorrelation-result" in source
    assert "--audit-manifest" in source
    assert "--audit-protocol" in source
