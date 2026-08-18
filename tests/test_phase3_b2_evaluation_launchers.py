from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = ROOT / "cluster/phase3_b2_transport_event_thresholds.sbatch"
COMPARATOR = ROOT / "cluster/phase3_b2_deterministic_comparators.sbatch"
FREEZE = ROOT / "cluster/phase3_b2_freeze_training_matrix.sbatch"
SMOKE = ROOT / "cluster/phase3_b2_evaluator_smoke.sbatch"
EVALUATION = ROOT / "cluster/phase3_b2_evaluation_full.sbatch"
FINALIZER = ROOT / "cluster/phase3_b2_finalize_evaluation.sbatch"


@pytest.mark.parametrize(
    "launcher", (THRESHOLD, COMPARATOR, FREEZE, SMOKE, EVALUATION, FINALIZER)
)
def test_b2_evaluation_launcher_has_valid_bash_and_fail_closed_checkout(
    launcher: Path,
) -> None:
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text()
    assert "set -euo pipefail" in text
    assert "PAPER0_EXPECTED_COMMIT" in text
    assert "status --porcelain --untracked-files=all" in text
    assert "Rocky Linux 9" in text
    assert "--no-requeue" in text


def test_threshold_launcher_is_training_only_cpu_and_runs_complete_suite() -> None:
    text = THRESHOLD.read_text()
    assert "#SBATCH --partition=gen" in text
    assert "#SBATCH --gres=gpu" not in text
    assert "build_b2_transport_event_thresholds.py" in text
    assert "phase2_potential_vorticity_all_frame_6893033.json" in text
    assert '"${PYTHON}" -m pytest -p no:cacheprovider -q' in text
    assert "--output-directory \"${OUTDIR}/thresholds\"" in text
    command = text.split("COMMAND=(", 1)[1].split("\n)", 1)[0]
    assert "validation" not in command.lower()


def test_deterministic_comparator_is_cpu_only_and_cannot_inspect_b2_results() -> None:
    text = COMPARATOR.read_text()
    assert "#SBATCH --partition=gen" in text
    assert "#SBATCH --gres=gpu" not in text
    assert "build_b2_deterministic_comparators.py" in text
    assert "phase2_o2_evaluation_full/job_6896117/final_matrix.json" in text
    assert 'result["B2_forecasts_or_scores_read"] is not False' in text
    assert 'result["held_out_85606_read"] is not False' in text
    assert 'result["scientific_acceptance_evaluated"] is not False' in text


def test_full_evaluation_launcher_is_exact_three_seed_m32_gpu_matrix() -> None:
    text = EVALUATION.read_text()
    assert "#SBATCH --array=0-2" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --constraint=h100|h200" in text
    assert "#SBATCH --mem=128G" in text
    assert "declare -ar SEEDS=(1701 1702 1703)" in text
    assert "B2_TRAINING_JOB_ID" in text
    assert "B2_TRAINING_FREEZE_JOB_ID" in text
    assert "B2_THRESHOLD_JOB_ID" in text
    assert "B2_SMOKE_JOB_ID" in text
    assert "sha256sum -c \"${TRAINING_OUTPUT}/artifact_sha256.txt\"" in text
    assert "sha256sum -c \"${THRESHOLD_ROOT}/thresholds/artifact_sha256.txt\"" in text
    assert "run_b2_evaluation_wandb.py" in text
    assert "WANDB_MODE=online" in text
    assert "--member-batch-size 4" in text
    assert "--mode full" in text
    assert "--training-matrix" in text
    assert "--smoke-result" in text
    assert 'result["forecast"]["bytes"] < 14_000_000_000' in text
    assert '"probabilistic_scientific_gate_evaluated": False' in text
    assert '"O3_launch_allowed": False' in text
    assert '"assimilation_allowed": False' in text
    assert '"diagnostic_ranking_allowed": False' in text


def test_training_freeze_and_four_target_smoke_precede_full_evaluation() -> None:
    freeze = FREEZE.read_text()
    smoke = SMOKE.read_text()
    assert "#SBATCH --gres=gpu" not in freeze
    assert "freeze_b2_training_matrix.py" in freeze
    assert "declare -ar SEEDS=(1701 1702 1703)" in freeze
    assert "artifact_sha256.txt" in freeze
    assert "#SBATCH --gres=gpu:1" in smoke
    assert "--mode smoke" in smoke
    assert "--seed 1701" in smoke
    assert "bounded_non_scientific_B2_evaluator_smoke_85604" in smoke
    assert "[498, 502]" in smoke
    assert 'result["O3_launch_allowed"] is not False' in smoke


def test_finalizer_is_cpu_only_and_keeps_downstream_scope_closed() -> None:
    text = FINALIZER.read_text()
    assert "#SBATCH --partition=gen" in text
    assert "#SBATCH --gres=gpu" not in text
    assert "B2_EVALUATION_JOB_ID" in text
    assert "B2_COMPARATOR_JOB_ID" in text
    assert "finalize_b2_evaluation.py" in text
    assert 'result["O3_launch_allowed"] is not False' in text
    assert 'result["assimilation_allowed"] is not False' in text
    assert 'result["diagnostic_ranking_allowed"] is not False' in text


def test_launchers_lock_current_local_evaluation_implementations() -> None:
    threshold = THRESHOLD.read_text()
    evaluation = EVALUATION.read_text()
    expected = {
        "paper0/tools/build_b2_transport_event_thresholds.py": threshold,
        "paper0/tools/build_b2_deterministic_comparators.py": COMPARATOR.read_text(),
        "paper0/tools/freeze_b2_training_matrix.py": FREEZE.read_text(),
        "paper0/tools/evaluate_b2_checkpoint.py": evaluation,
        "paper0/tools/run_b2_evaluation_wandb.py": evaluation,
        "src/tcv_diagnostics/b2_forecast.py": evaluation,
        "src/tcv_diagnostics/b2_scoring.py": evaluation,
        "src/tcv_diagnostics/b2_field_metrics.py": evaluation,
        "src/tcv_diagnostics/b2_field_scoring.py": evaluation,
        "src/tcv_diagnostics/b2_spectral_metrics.py": evaluation,
        "src/tcv_diagnostics/b2_transport_metrics.py": evaluation,
        "src/tcv_diagnostics/b2_probabilistic_metrics.py": evaluation,
        "src/tcv_diagnostics/b2_acceptance.py": COMPARATOR.read_text(),
        "paper0/tools/finalize_b2_evaluation.py": FINALIZER.read_text(),
        "src/tcv_diagnostics/b2_acceptance_gate.py": FINALIZER.read_text(),
    }
    for relative, launcher_text in expected.items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert digest in launcher_text, relative
    evaluator_digest = hashlib.sha256(
        (ROOT / "paper0/tools/evaluate_b2_checkpoint.py").read_bytes()
    ).hexdigest()
    assert evaluator_digest in SMOKE.read_text()
