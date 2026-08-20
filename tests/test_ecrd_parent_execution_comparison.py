"""Tests for the narrow CPU-smoke/H100 ECRD parent comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/compare_ecrd_parent_executions.py"
LAUNCHER = ROOT / "cluster/ecrd_parent_execution_comparison.sbatch"
SPEC = importlib.util.spec_from_file_location("compare_ecrd_parents", TOOL)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def test_streamed_parent_difference_summary_is_exact() -> None:
    reference = np.arange(2 * 5 * 2 * 3 * 4, dtype=np.float32).reshape(
        2, 5, 2, 3, 4
    )
    cpu = reference.copy()
    cpu[0, 2, 1, 1, 1] += np.float32(0.25)
    accumulator = COMPARE.ParentDifferenceAccumulator()
    accumulator.update(cpu[:1], reference[:1])
    accumulator.update(cpu[1:], reference[1:])
    record = accumulator.to_record()

    assert record["frame_count"] == 2
    assert record["element_count"] == reference.size
    assert record["maximum_absolute_difference"] == 0.25
    assert record["mean_absolute_difference"] == 0.25 / reference.size
    assert record["by_field"]["Pi"]["maximum_absolute_difference"] == 0.25
    assert record["by_field"]["Ne"]["relative_RMS_difference"] == 0.0
    assert record["engineering_consistency_guard"]["diagnostic_only"] is True
    assert record["engineering_consistency_guard"]["may_promote_CPU_parent"] is False


def test_exact_parents_pass_the_engineering_consistency_guard() -> None:
    values = np.ones((1, 5, 2, 2, 3), dtype=np.float32)
    accumulator = COMPARE.ParentDifferenceAccumulator()
    accumulator.update(values, values.copy())
    record = accumulator.to_record()
    assert record["exact_fraction"] == 1.0
    assert record["relative_RMS_difference"] == 0.0
    assert record["engineering_consistency_guard"]["passed"] is True


def test_parent_comparison_has_no_scientific_or_training_dependency() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "scientific_H100_parent_only" in source
    assert '"CPU_parent_promoted": False' in source
    assert '"scientific_result": False' in source
    assert '"physics_metric_evaluated": False' in source
    assert "from tcv_diagnostics.transport" not in source
    assert "from tcv_diagnostics.spect" not in source
    assert "from tcv_diagnostics.assimilat" not in source


def test_parent_comparison_launcher_is_hash_locked_and_fail_closed() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --gres" not in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "930f3b1f8e759afa0e30b0a57b51472505dd683920deb49076e6c81bca2ecb82" in source
    assert "d238d055c3f1da9e3096a81cac67176f90365c99dfb423a1a0629f85b61f9532" in source
    assert "21725a5ae29832068676840aabea37a2e59eac861e8b9243bf45414de6811cbd" in source
    assert 'if not guard["passed"]' in source
    assert '"CPU_parent_promoted": False' in source
    assert '"held_out_85606_read": False' in source
