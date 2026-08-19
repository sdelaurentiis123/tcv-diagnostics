"""Static fail-closed checks for the Rocky 9 residual-KL launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_residual_kl_oracle.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_is_one_Rocky9_CPU_job_within_frozen_budget() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --cpus-per-task=24" in source
    assert "#SBATCH --mem=128G" in source
    assert "#SBATCH --time=06:00:00" in source
    assert "#SBATCH --gres" not in source
    assert "nvidia-smi" not in source
    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    assert "Rocky Linux 9" in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "pytest -p no:cacheprovider -q" in source
    assert "WANDB_MODE=online" in source
    assert 'NODE_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"' in source
    assert "Writable node scratch is required" in source


def test_launcher_hash_pins_protocol_programs_and_direct_modules() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    paths = (
        "paper0/PHASE3_POST_LOCALIZATION_DECISION.md",
        "paper0/protocol/PHASE3_RESIDUAL_KL_ORACLE_PROTOCOL.md",
        "paper0/manifests/phase3_residual_kl_oracle_85604.json",
        "paper0/tools/build_residual_kl_pretruth.py",
        "paper0/tools/evaluate_residual_kl_oracle.py",
        "paper0/tools/run_residual_kl_oracle_wandb.py",
        "src/tcv_diagnostics/residual_kl_oracle.py",
        "src/tcv_diagnostics/residual_kl_metrics.py",
        "src/tcv_diagnostics/b2_field_metrics.py",
        "src/tcv_diagnostics/b2_field_scoring.py",
        "src/tcv_diagnostics/b2_forecast.py",
        "src/tcv_diagnostics/b2_spectral_metrics.py",
        "src/tcv_diagnostics/b5_covariance_localization.py",
        "src/tcv_diagnostics/b5_residual_audit.py",
        "src/tcv_diagnostics/b5_residual_forecast.py",
        "src/tcv_diagnostics/codec_training.py",
        "src/tcv_diagnostics/codec_transport.py",
        "src/tcv_diagnostics/matched_o1_transport.py",
        "src/tcv_diagnostics/model_data.py",
        "src/tcv_diagnostics/model_training_data.py",
        "src/tcv_diagnostics/o2_forecast.py",
        "src/tcv_diagnostics/resampling.py",
        "src/tcv_diagnostics/geometry.py",
        "src/tcv_diagnostics/wandb_tracking.py",
        "src/tcv_diagnostics/b2_probabilistic_metrics.py",
        "src/tcv_diagnostics/metrics.py",
    )
    for relative in paths:
        assert _sha256(ROOT / relative) in source, relative


def test_launcher_pins_every_immutable_85604_authority() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    expected = (
        "job_6901393/audit/h1_training_forecast.h5",
        "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18",
        "job_6896117/task_0_c5p_h1_seed_1701/forecast.h5",
        "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed",
        "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67",
        "50c54a8e9dd0f0983cb8360f598bdf00eae22854de2ab471cd7385e767f3058b",
        "331e7f3ff5d221d0d3720d9112ce90436d8330647501a2268f974867bbc140d2",
        "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18",
        "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7",
        "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
        "n=5k",
    )
    for value in expected:
        assert value in source


def test_launcher_keeps_all_scientific_artifacts_local_and_scope_closed() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "run_residual_kl_oracle_wandb.py" in source
    assert 'wandb.get("remote_state_after_finish") != "finished"' in source
    assert '"basis_arrays_uploaded"' in source
    assert '"raw_accumulators_uploaded"' in source
    assert '"model_training_performed"' in source
    assert '"diagnostic_ranking_performed"' in source
    assert "No checkpoint, inference, training, 85606, O3/O4/O5" in source
