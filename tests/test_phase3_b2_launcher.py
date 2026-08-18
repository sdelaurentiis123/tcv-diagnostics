"""Static safety checks for the bounded B2 Rocky 9 launcher."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b2_ldm_gpu_smoke.sbatch"


def test_b2_launcher_is_rocky9_single_hopper_and_bounded() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "Rocky Linux 9" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --constraint=h100|h200" in source
    assert "--mode smoke" in source
    assert "--seed 1701" in source
    assert 'result["completed_epochs"] != 2' in source
    assert 'len(range(*result["config"]["train_targets"])) > 16' in source
    assert "full_B2_training_authorized" in source


def test_b2_launcher_requires_clean_commit_full_suite_and_online_wandb() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "-m pytest -p no:cacheprovider -q" in source
    assert "WANDB_MODE=online" in source
    assert "remote_state_after_finish" in source
    assert "checkpoint_reload_bitwise_exact" in source


def test_b2_launcher_has_no_unresolved_hash_placeholders() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    checks = "\n".join(
        line for line in source.splitlines() if line.startswith("check_sha256")
    )
    assert re.search(r"__[A-Z_]+__", checks) is None
