"""Static safety locks for the completed O2 training freeze job."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_freeze_o2_training.sbatch"
LOCKED = (
    ROOT / "paper0/tools/freeze_o2_training_matrix.py",
    ROOT / "src/tcv_diagnostics/o2_training.py",
    ROOT / "src/tcv_diagnostics/models/o2.py",
    ROOT / "src/tcv_diagnostics/models/vit.py",
    ROOT / "src/tcv_diagnostics/model_data.py",
    ROOT / "src/tcv_diagnostics/codec_training.py",
    ROOT / "paper0/manifests/phase2_c5p_o2_full_runs_85604.json",
)


def test_o2_freeze_is_short_cpu_only_rocky9_and_non_overwriting() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=gen",
        "#SBATCH --qos=gen",
        "#SBATCH --time=00:15:00",
        'TRAINING_JOB_ID="6894980"',
        "9035bc3ce9d2351cd17586f4429af8116d43a47e",
        "PAPER0_EXPECTED_COMMIT",
        "status --porcelain --untracked-files=all",
        '"${VERSION_ID%%.*}" != "9"',
        "Refusing to overwrite existing result directory",
        "frozen_training_matrix.json",
    ):
        assert required in text
    assert "--gres=gpu" not in text


def test_o2_freeze_locks_code_manifest_and_raw_training_summary() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text
    assert "6d539b540a61d35df34d368499f7468f6931342abf512097f5dbd4774f8597df" in text
    assert "c0336239c94d204328e17a2b05822f210937a5e935e8857d0375664d94e2690a" in text


def test_o2_freeze_does_not_run_inference_or_open_later_stages() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "train_o2.py" not in text
    assert "--mode full" not in text
    assert "85606" not in text
    assert "O2 remains scientifically undecided" in text
