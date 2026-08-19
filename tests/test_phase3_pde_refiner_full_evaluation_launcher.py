"""Static safety checks for the full B4 evaluation/decision launcher."""

from __future__ import annotations

from pathlib import Path
import hashlib


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b4_pde_refiner_evaluation_full.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_launcher_is_Rusty_Rocky9_H100_H200_only() -> None:
    text = LAUNCHER.read_text()
    assert "#SBATCH --partition=gpupreempt" in text
    assert "#SBATCH --constraint=h100|h200" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert '"${ID}" != "rocky"' in text
    assert '"${VERSION_ID%%.*}" != "9"' in text
    assert "GPU outside" not in text


def test_full_launcher_pins_passed_smoke_and_closed_85606_scope() -> None:
    text = LAUNCHER.read_text()
    assert "B4_SMOKE_JOB_ID" in text
    assert "029f6d9d425fd9bbac11aebf82466588a97ac658" in text
    assert "--mode full" in text
    assert "--smoke-result" in text
    assert "held_out_85606_read" in text
    assert "O3_launch_allowed" in text
    assert "assimilation_allowed" in text
    assert "No B4 outcome authorizes O3 execution" in text


def test_full_launcher_hash_locks_evaluator_metrics_and_gate() -> None:
    text = LAUNCHER.read_text()
    assert "PLACEHOLDER" not in text
    for relative in (
        "paper0/tools/evaluate_b4_pde_refiner_checkpoint.py",
        "paper0/tools/run_b4_pde_refiner_evaluation_wandb.py",
        "paper0/tools/finalize_b4_pde_refiner_one_seed.py",
        "src/tcv_diagnostics/pde_refiner_acceptance_gate.py",
        "src/tcv_diagnostics/pde_refiner_forecast.py",
        "src/tcv_diagnostics/pde_refiner_scoring.py",
        "src/tcv_diagnostics/b2_probabilistic_metrics.py",
        "src/tcv_diagnostics/b2_transport_metrics.py",
    ):
        assert relative in text
        assert _sha256(ROOT / relative) in text, relative
    assert "2b04c10971e6d38ee439e33aa0b5331305acf16b38a96e7952fb26046049b5d2" in text


def test_full_launcher_runs_tests_before_generation_and_gate_after_scoring() -> None:
    text = LAUNCHER.read_text()
    tests = text.index('"${PYTHON}" -m pytest')
    evaluator = text.index('"${COMMAND[@]}"')
    finalizer = text.index('"${GATE_COMMAND[@]}"')
    assert tests < evaluator < finalizer
    assert "--evaluation-wandb" in text
    assert "--training-wandb" in text
    assert "--gate-execution-commit" in text


def test_full_launcher_uses_scoped_nonoverwriting_outputs() -> None:
    text = LAUNCHER.read_text()
    assert 'if [[ -e "${OUTDIR}" || -e "${NODE_LOCAL_ROOT}" ]]' in text
    assert 'rm -rf "${NODE_LOCAL_ROOT}"' in text
    assert 'rm -rf "${OUTDIR}"' not in text
    assert "--no-requeue" in text
