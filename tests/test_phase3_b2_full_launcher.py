"""Static safety checks for the full B2 Rocky 9 Slurm array."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b2_ldm_full.sbatch"


def test_full_b2_launcher_is_three_seed_rocky9_hopper_array() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --qos=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --constraint=h100|h200" in source
    assert "#SBATCH --array=0-2" in source
    assert "#SBATCH --no-requeue" in source
    assert "--mode full" in source
    assert "SEEDS=(1701 1702 1703)" in source
    assert "Rocky Linux 9" in source


def test_full_b2_launcher_is_hash_locked_clean_and_85604_only() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "phase3_b2_full_evaluation_85604.json" in source
    assert "phase3_b2_ldm_gpu_smoke_6896402.json" in source
    assert "phase2_model_dataset/job_6893525" in source
    assert '"held_out_85606_read",' in source
    assert "result[flag]" in source
    assert "85606" not in source.split("MODEL_DATA_SOURCE=", 1)[1].splitlines()[0]
    checks = "\n".join(
        line for line in source.splitlines() if line.startswith("check_sha256")
    )
    assert re.search(r"__[A-Z_]+__", checks) is None


def test_full_b2_launcher_enforces_budget_wandb_and_nonacceptance() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "-m pytest -p no:cacheprovider -q" in source
    assert "WANDB_MODE=online" in source
    assert 'result["completed_epochs"] != 200' in source
    assert 'result["completed_optimizer_steps"] != 5400' in source
    assert 'tracking["epochs_logged"] != 200' in source
    assert "remote_state_after_finish" in source
    assert "training_complete_is_scientific_acceptance" in source
    assert "probabilistic_scientific_gate_evaluated" in source
    assert "O3_launch_allowed" in source
