"""Static fail-closed checks for the Rocky 9 B5 residual-EDM launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b5_field_residual_edm_smoke.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_B5_EDM_launcher_is_one_H100_Rocky9_online_bounded_smoke() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --constraint=h100" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "Rocky Linux 9" in source
    assert "WANDB_MODE=online" in source
    assert "wandb_preflight.json" in source
    assert "pytest -p no:cacheprovider -q" in source
    assert "--mode smoke" in source
    assert "--seed 1701" in source
    assert "--mode full" not in source


def test_B5_EDM_launcher_pins_protocol_implementation_and_data_readers() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    paths = (
        "paper0/manifests/phase3_b5_field_residual_edm_smoke_85604.json",
        "paper0/protocol/PHASE3_B5_FIELD_RESIDUAL_EDM_SMOKE_PROTOCOL.md",
        "paper0/tools/train_b5_field_residual_edm_smoke.py",
        "src/tcv_diagnostics/models/field_residual_edm.py",
        "src/tcv_diagnostics/b5_residual_edm_training.py",
        "src/tcv_diagnostics/b5_residual_edm_wandb_tracking.py",
        "src/tcv_diagnostics/b5_residual_forecast.py",
        "src/tcv_diagnostics/o2_training_data.py",
        "src/tcv_diagnostics/model_training_data.py",
        "src/tcv_diagnostics/wandb_tracking.py",
        "src/tcv_diagnostics/models/layers.py",
        "pyproject.toml",
    )
    for relative in paths:
        assert _sha256(ROOT / relative) in source, relative


def test_B5_EDM_launcher_pins_and_reverifies_parent_forecast_and_audit() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "job_6901393/audit/h1_training_forecast.h5" in source
    assert "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18" in source
    assert "job_6901393/audit/residual_audit.json" in source
    assert "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67" in source
    assert source.count('check_sha256 "${H1_FORECAST_SHA}"') >= 4
    assert source.count('check_sha256 "${RESIDUAL_AUDIT_SHA}"') >= 4
    assert "job_6893525" in source
    assert 'if [[ "${verified_shards}" -ne 8 ]]' in source


def test_B5_EDM_launcher_verifies_mechanics_and_keeps_science_closed() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'result.get("completed_optimizer_steps") != 64' in source
    assert 'result.get("parameter_count") != 11604709' in source
    assert 'get("canonical_field_shape") != [1, 2, 1, 5, 64, 32, 88]' in source
    assert 'get("network_evaluations_per_member") != 35' in source
    assert 'tracking.get("steps_logged") != 64' in source
    assert "frozen_H1_inputs_reverified_after_smoke" in source
    assert "No scientific checkpoint was selected" in source
    assert "validation, 85606, O3, assimilation, and ranking remain closed" in source
    assert "artifact_sha256.txt" in source
