"""Checks for the data-free ECRD model-ladder reduction launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/ecrd_scientific_evaluation_finalize.sbatch"


def test_reducer_launcher_is_cpu_only_and_dependency_ready() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --gres" not in source
    assert "ECRD_EVALUATION_FREEZE_JOB_ID" in source
    assert "ECRD_EVALUATION_ARRAY_JOB_ID" in source
    assert "summarize_ecrd_model_ladder.py" in source


def test_reducer_launcher_locks_reducer_and_acceptance_code() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for relative in (
        "paper0/tools/summarize_ecrd_model_ladder.py",
        "src/tcv_diagnostics/ecrd_acceptance.py",
    ):
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert digest in source


def test_reducer_launcher_consumes_the_complete_score_matrix() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for arm, safe in (
        ("B5", "b5"),
        ("B5-Context", "b5_context"),
        ("ECRD", "ecrd"),
        ("ECRD-History", "ecrd_history"),
    ):
        for seed in (1701, 1702, 1703):
            assert source.count(f":{arm}:{safe}:{seed}\"") == 1
    assert "SCORE_ARGS+=(--score" in source


def test_reducer_launcher_is_data_free_and_cannot_release_downstream() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "load_official_catalog" not in source
    assert "forecast_M32.h5" not in source
    assert "85606" not in source
    assert "assimilation" not in source.lower()
    assert "diagnostic_ranking" not in source.lower()
