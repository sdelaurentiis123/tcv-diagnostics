"""Safety and static gates for the ECRD parent CPU timing probe."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/ecrd_sym_h1_parent_cpu_probe.sbatch"
ENTRYPOINT = ROOT / "paper0/tools/probe_ecrd_sym_h1_parent_cpu.py"
NOTE = ROOT / "paper0/protocol/ECRD_PARENT_CPU_TIMING_PROBE_2026-08-20.md"


def test_probe_files_parse() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))


def test_launcher_is_short_cpu_only_rocky9_job() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --qos=gen" in source
    assert "#SBATCH --time=01:00:00" in source
    assert "#SBATCH --gres" not in source
    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    assert '"${VERSION_ID%%.*}" != "9"' in source


def test_probe_is_commit_locked_nonoverwriting_and_85604_only() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source
    assert 'if [[ -e "${JOB_ROOT}" ]]' in source
    assert '"85606" in str(path).lower()' in entrypoint
    assert '"held_out_85606_read": False' in entrypoint
    assert "PROBE_TARGET_FRAME = 2" in entrypoint
    assert "target_frames=(PROBE_TARGET_FRAME,)" in entrypoint


def test_probe_cannot_write_parent_or_compute_science() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert "ECRDParentMeanWriter" not in source
    assert "generate_symmetrized_h1_parent(" not in source
    assert '"scientific_parent_artifact_written": False' in source
    assert '"physics_metric_evaluated": False' in source
    assert '"training_performed": False' in source
    assert '"saved": False' in source
    assert "optimizer" not in source.lower()
    assert ".backward(" not in source


def test_prospective_note_keeps_h100_training_rule() -> None:
    source = NOTE.read_text(encoding="utf-8")
    assert "Prospectively frozen before executing the probe" in source
    assert "does not amend the ECRD scientific model-development protocol" in source
    assert "The H100 requirement for ECRD training is unchanged" in source
