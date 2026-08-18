from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_freeze_o1_codec_r2.sbatch"
LOCKED = (
    ROOT / "paper0/tools/freeze_o1_codec_training_matrix.py",
    ROOT / "src/tcv_diagnostics/codec_training.py",
    ROOT / "src/tcv_diagnostics/model_data.py",
)


def test_r2_freeze_launcher_is_short_cpu_only_and_locks_training_job() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "--partition=gen",
        "--qos=gen",
        "--time=00:10:00",
        "9ba2a59ee94708a64e0255b25c29653740389fbe",
        'TRAINING_JOB_ID="6894463"',
        "PAPER0_EXPECTED_COMMIT",
        "status --porcelain --untracked-files=all",
        "frozen_training_matrix.json",
        "--stage R2",
        "--codec dcae_l10",
    ):
        assert required in text
    assert "--gres=gpu" not in text
    assert "85606" not in text


def test_r2_freeze_launcher_hash_locks_local_dependencies() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text
