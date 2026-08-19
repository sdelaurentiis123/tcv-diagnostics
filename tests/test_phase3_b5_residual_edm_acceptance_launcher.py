"""Static checks for the CPU-only frozen B5 gate launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b5_residual_edm_acceptance.sbatch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_B5_gate_launcher_is_cpu_only_Rocky9_and_requires_complete_inputs() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --qos=gen" in source
    assert "#SBATCH --gres" not in source
    assert "Rocky Linux 9" in source
    assert "B5_TRAINING_JOB_ID" in source
    assert "B5_EVALUATION_JOB_ID" in source
    assert "job_6899063" in source
    assert 'sha256sum -c "${TRAINING_OUTPUT}/artifact_sha256.txt"' in source
    assert 'sha256sum -c "${EVALUATION_OUTPUT}/artifact_sha256.txt"' in source
    assert 'sha256sum -c "${COMPARATOR_ROOT}/comparators/artifact_sha256.txt"' in source


def test_B5_gate_launcher_pins_finalizer_reducers_thresholds_and_comparator() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    paths = (
        "paper0/tools/finalize_b5_residual_edm_one_seed.py",
        "src/tcv_diagnostics/b5_residual_edm_acceptance_gate.py",
        "src/tcv_diagnostics/b2_acceptance_gate.py",
        "src/tcv_diagnostics/b2_acceptance_gate_event_eligibility.py",
        "paper0/manifests/phase3_b5_full_training_evaluation_85604.json",
        "paper0/manifests/phase3_b3_full_evaluation_85604.json",
    )
    for relative in paths:
        assert sha256(ROOT / relative) in source, relative
    assert "2b04c10971e6d38ee439e33aa0b5331305acf16b38a96e7952fb26046049b5d2" in source


def test_B5_gate_launcher_is_pure_and_cannot_launch_downstream_work() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        '"raw_forecast_changed"',
        '"raw_score_changed"',
        '"metrics_recomputed"',
        '"training_performed"',
        '"inference_performed"',
        '"truth_scoring_performed"',
        '"additional_seed_training_authorized"',
        '"O3_launch_allowed"',
        '"held_out_85606_access_allowed"',
        '"assimilation_allowed"',
        '"diagnostic_ranking_allowed"',
    ):
        assert fragment in source
    assert "No training, inference, truth scoring, O3 launch" in source


def test_B5_gate_launcher_preserves_conditional_O3_protocol_disposition() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'result["O3_protocol_may_be_written"]' in source
    assert 'acceptance["passes_complete_one_seed_gate"]' in source
    assert "truth_event_count_eligible_v1" in source
    assert "artifact_sha256.txt" in source

