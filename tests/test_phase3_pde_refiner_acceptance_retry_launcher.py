"""Static safety checks for the CPU-only B4 acceptance retry launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b4_pde_refiner_acceptance_retry.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_B4_acceptance_retry_is_CPU_only_and_immutable_input_only() -> None:
    text = LAUNCHER.read_text()
    assert "#SBATCH --partition=gen" in text
    assert "#SBATCH --qos=gen" in text
    assert "--gres" not in text
    assert "nvidia-smi" not in text
    assert 'readonly EVALUATION_JOB_ID="6901015"' in text
    assert "0988f71aa0749044e51ded92b9ea594563232df746415dccbbc6031443ca7e92" in text
    assert "evaluate_b4_pde_refiner_checkpoint.py" not in text
    assert "run_b4_pde_refiner_evaluation_wandb.py" not in text
    assert "forecast mutation" in text
    assert "held_out_85606_read=false" in text


def test_B4_acceptance_retry_hash_locks_pure_reducer_sources() -> None:
    text = LAUNCHER.read_text()
    for relative in (
        "paper0/tools/finalize_b4_pde_refiner_one_seed.py",
        "src/tcv_diagnostics/pde_refiner_acceptance_gate.py",
        "src/tcv_diagnostics/b2_acceptance_gate.py",
        "src/tcv_diagnostics/b2_acceptance_gate_event_eligibility.py",
        "paper0/manifests/phase3_b4_full_evaluation_85604.json",
    ):
        assert relative in text
        assert _sha256(ROOT / relative) in text, relative


def test_B4_acceptance_retry_runs_complete_suite_before_reduction() -> None:
    text = LAUNCHER.read_text()
    test_position = text.index('"${PYTHON}" -m pytest -p no:cacheprovider -q')
    reduction_position = text.index('"${PYTHON}" -u "${ENTRYPOINT}"')
    assert test_position < reduction_position
