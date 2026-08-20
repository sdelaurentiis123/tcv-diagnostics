"""Static checks for the frozen ECRD 85604 scientific-evaluation launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/ecrd_scientific_evaluation_full.sbatch"


def test_launcher_requests_matched_h100_array() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --constraint=h100" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --array=0-11%2" in source
    assert "#SBATCH --time=12:00:00" in source
    assert "ECRD_EVALUATION_FREEZE_JOB_ID" in source


def test_launcher_matrix_contains_each_arm_seed_once() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for arm, safe in (
        ("B5", "b5"),
        ("B5-Context", "b5_context"),
        ("ECRD", "ecrd"),
        ("ECRD-History", "ecrd_history"),
    ):
        for seed in (1701, 1702, 1703):
            needle = f'readonly ARM="{arm}"; readonly SAFE_ARM="{safe}"; readonly MODEL_SEED="{seed}"'
            assert source.count(needle) == 1


def test_launcher_locks_the_frozen_evaluator_and_metric_code() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for relative in (
        "paper0/tools/evaluate_ecrd_checkpoint.py",
        "src/tcv_diagnostics/ecrd_forecast.py",
        "src/tcv_diagnostics/ecrd_scoring.py",
        "src/tcv_diagnostics/ecrd_acceptance.py",
    ):
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert digest in source


def test_launcher_preserves_truth_separation_and_scope() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "evaluate_ecrd_checkpoint.py" in source
    assert "--member-batch-size 8" in source
    assert "--historical-b5-forecast" in source
    assert "85604" in source
    assert "85606" not in source
    assert "wandb" not in source.lower()
    assert "assimilation" not in source.lower()
    assert "diagnostic_ranking" not in source.lower()
