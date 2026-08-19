"""Static fail-closed tests for the frozen B5 full-training launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b5_field_residual_edm_full.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_B5_full_launcher_is_one_H100_Rocky9_online_and_full_only() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --constraint=h100" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "Rocky Linux 9" in source
    assert "WANDB_MODE=online" in source
    assert "wandb_preflight.json" in source
    assert "pytest -p no:cacheprovider -q" in source
    assert "--mode full" in source
    assert "--seed 1701" in source
    assert "--mode smoke" not in source


def test_B5_full_launcher_pins_training_and_predeclared_evaluation_chain() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    paths = (
        "paper0/manifests/phase3_b5_full_training_evaluation_85604.json",
        "paper0/protocol/PHASE3_B5_FULL_TRAINING_EVALUATION_PROTOCOL.md",
        "paper0/tools/train_b5_field_residual_edm_full.py",
        "src/tcv_diagnostics/b5_residual_edm_full_training.py",
        "src/tcv_diagnostics/b5_residual_edm_full_wandb_tracking.py",
        "src/tcv_diagnostics/b5_residual_edm_forecast.py",
        "src/tcv_diagnostics/models/field_residual_edm.py",
        "paper0/tools/evaluate_b5_residual_edm_checkpoint.py",
        "paper0/tools/run_b5_residual_edm_evaluation_wandb.py",
        "paper0/tools/finalize_b5_residual_edm_one_seed.py",
        "src/tcv_diagnostics/b5_residual_edm_scoring.py",
        "src/tcv_diagnostics/b5_residual_edm_acceptance_gate.py",
    )
    for relative in paths:
        assert sha256(ROOT / relative) in source, relative
    assert "The scientific evaluator and gate are byte-pinned before training" in source


def test_B5_full_launcher_pins_both_H1_means_audit_and_85604_data() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "job_6901393/audit/h1_training_forecast.h5" in source
    assert "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18" in source
    assert "job_6896117/task_0_c5p_h1_seed_1701/forecast.h5" in source
    assert "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed" in source
    assert "job_6901393/audit/residual_audit.json" in source
    assert "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67" in source
    assert "job_6893525" in source
    assert 'if [[ "${verified_shards}" -ne 8 ]]' in source


def test_B5_full_launcher_checks_complete_budget_and_keeps_science_closed() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '"completed_epochs": 100' in source
    assert '"completed_optimizer_steps": 10800' in source
    assert '"EMA_updates": 10800' in source
    assert '"candidate_count": 20' in source
    assert '"checkpoint_reload_bitwise_exact": True' in source
    assert '"physics_metric_used_for_checkpoint_selection": False' in source
    assert '"scientific_forecast_generated": False' in source
    assert '"scientific_acceptance_evaluated": False' in source
    assert "No scientific forecast or gate has run" in source
    assert "artifact_sha256.txt" in source

