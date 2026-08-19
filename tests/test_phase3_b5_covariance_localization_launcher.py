"""Static fail-closed checks for the Rocky 9 B5 localization launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b5_covariance_localization.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_is_Rocky9_CPU_only_online_read_only_analysis() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --cpus-per-task=16" in source
    assert "#SBATCH --mem=160G" in source
    assert "#SBATCH --gres" not in source
    assert "nvidia-smi" not in source
    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    assert "export OMP_NUM_THREADS=12" in source
    assert "export OPENBLAS_NUM_THREADS=12" in source
    assert "export MKL_NUM_THREADS=12" in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "Rocky Linux 9" in source
    assert "WANDB_MODE=online" in source
    assert "pytest -p no:cacheprovider -q" in source
    assert "No training, inference, retuning" in source


def test_launcher_pins_protocol_wrapper_evaluator_and_direct_modules() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    paths = (
        "paper0/manifests/phase3_b5_covariance_localization_85604.json",
        "paper0/protocol/PHASE3_B5_COVARIANCE_LOCALIZATION_PROTOCOL.md",
        "paper0/tools/localize_b5_covariance.py",
        "paper0/tools/run_b5_covariance_localization_wandb.py",
        "src/tcv_diagnostics/b5_covariance_localization.py",
        "src/tcv_diagnostics/b2_field_metrics.py",
        "src/tcv_diagnostics/b5_residual_edm_forecast.py",
        "src/tcv_diagnostics/b5_residual_edm_full_training.py",
        "src/tcv_diagnostics/b5_residual_forecast.py",
        "src/tcv_diagnostics/b5_residual_audit.py",
        "src/tcv_diagnostics/codec_training.py",
        "src/tcv_diagnostics/codec_transport.py",
        "src/tcv_diagnostics/matched_o1_transport.py",
        "src/tcv_diagnostics/model_data.py",
        "src/tcv_diagnostics/model_training_data.py",
        "src/tcv_diagnostics/o2_forecast.py",
        "src/tcv_diagnostics/resampling.py",
        "src/tcv_diagnostics/geometry.py",
        "src/tcv_diagnostics/wandb_tracking.py",
        "paper0/results/phase2_potential_vorticity_all_frame_6893033.json",
        "paper0/manifests/phase2_85604_geometry_units.json",
    )
    for relative in paths:
        assert _sha256(ROOT / relative) in source, relative


def test_launcher_pins_every_immutable_forecast_and_data_authority() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    expected = (
        "job_6901587/b5_joint_field_residual_edm_seed_1701/forecast_M32.h5",
        "1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b",
        "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14",
        "job_6896117/task_0_c5p_h1_seed_1701/forecast.h5",
        "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed",
        "c81c0e06313c652816be77025c2b42bbfce10728df7ac14787e00edf7d978ba6",
        "a1d9cf00de0a2b0b3cc0c13d31c727420214040dcbf575afa67c6ae64015974b",
        "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67",
        "50c54a8e9dd0f0983cb8360f598bdf00eae22854de2ab471cd7385e767f3058b",
        "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18",
        "job_6893525",
        "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
        "n=5k",
    )
    for value in expected:
        assert value in source


def test_launcher_keeps_artifacts_local_and_downstream_scopes_closed() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "run_b5_covariance_localization_wandb.py" in source
    assert 'wandb.get("remote_state_after_finish") != "finished"' in source
    assert '"raw_accumulators_uploaded"' in source
    assert 'localization.get("variogram_scores"' in source
    assert "--checkpoint" not in source
    assert "--mode full" not in source
    assert "--mode smoke" not in source
    assert 'len(result.get("tables", {})) != 6' in source
    assert 'len(result.get("figures", [])) != 12' in source
