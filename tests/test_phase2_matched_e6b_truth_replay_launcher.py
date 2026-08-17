from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_matched_e6b_truth_replay.sbatch"
LOCKED = (
    ROOT / "paper0/oracles/matched_e6b_elliptic/CMakeLists.txt",
    ROOT / "paper0/oracles/matched_e6b_elliptic/matched_e6b_elliptic_oracle.cxx",
    ROOT / "paper0/tools/extract_matched_e6b_phi.py",
    ROOT / "paper0/tools/summarize_matched_e6b_truth_replay.py",
    ROOT / "paper0/oracles/potential_elliptic/BOUT.inp",
    ROOT / "paper0/results/phase2_potential_vorticity_all_frame_6893033.json",
)


def test_truth_replay_launcher_is_cpu_only_clean_and_all_frame() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "--partition=gen",
        "--qos=gen",
        "--ntasks=4",
        "--time=02:00:00",
        "PAPER0_EXPECTED_COMMIT",
        "status --porcelain --untracked-files=all",
        "paper0:truth_layout=true",
        "for ((shard = 0; shard < 8; shard++))",
        "truth_replay_summary.json",
        "--atol 5e-10",
        "--rtol 5e-10",
    ):
        assert required in text
    assert "--gres=gpu" not in text
    assert "85606" not in text


def test_truth_replay_launcher_hash_locks_every_local_dependency() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text


def test_truth_replay_postprocessors_do_not_import_training_dependencies() -> None:
    for path in LOCKED[2:4]:
        text = path.read_text(encoding="utf-8")
        assert "tcv_diagnostics.codec_training" not in text
        assert "sha256_file as sha256_path" in text
