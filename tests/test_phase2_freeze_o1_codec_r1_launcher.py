from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_freeze_o1_codec_r1.sbatch"
LOCKED = (
    ROOT / "paper0/tools/freeze_o1_codec_training_matrix.py",
    ROOT / "src/tcv_diagnostics/codec_training.py",
    ROOT / "src/tcv_diagnostics/model_data.py",
)


def test_freeze_launcher_is_short_cpu_only_and_locks_training_job() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "--partition=gen",
        "--qos=gen",
        "--time=00:10:00",
        "a5adc57499239702a8b28b661357b3dcfcbaa167",
        'TRAINING_JOB_ID="6893802"',
        "PAPER0_EXPECTED_COMMIT",
        "status --porcelain --untracked-files=all",
        "frozen_training_matrix.json",
    ):
        assert required in text
    assert "--gres=gpu" not in text
    assert "85606" not in text


def test_freeze_launcher_hash_locks_local_dependencies() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text
