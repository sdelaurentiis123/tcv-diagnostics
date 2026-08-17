from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_matched_e6b_compile.sbatch"
LOCKED = (
    ROOT / "paper0/oracles/matched_e6b_elliptic/CMakeLists.txt",
    ROOT / "paper0/oracles/matched_e6b_elliptic/matched_e6b_elliptic_oracle.cxx",
)


def test_compile_launcher_is_short_cpu_only_rocky9_and_clean() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "--partition=gen",
        "--qos=gen",
        "--time=00:10:00",
        "PAPER0_EXPECTED_COMMIT",
        "status --porcelain --untracked-files=all",
        "VERSION_ID%%.*",
        "matched_e6b_elliptic_oracle",
        "artifact_sha256.txt",
    ):
        assert required in text
    assert "--gres=gpu" not in text
    assert "85606" not in text


def test_compile_launcher_hash_locks_local_sources() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text
