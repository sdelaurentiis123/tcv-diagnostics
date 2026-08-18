"""Static and file-contract tests for the B3 gate-only finalizer."""

from __future__ import annotations

import inspect

import pytest

from paper0.tools import finalize_b3_fgn_one_seed as finalizer
from tcv_diagnostics.codec_training import sha256_path


ROOT = finalizer.ROOT
LAUNCHER = ROOT / "cluster/phase3_b3_fgn_acceptance.sbatch"


def test_verified_reference_requires_the_declared_content_hash(tmp_path) -> None:
    path = tmp_path / "record.json"
    path.write_text('{"scope":"development"}\n', encoding="utf-8")
    parent = {"item": {"path": str(path), "sha256": sha256_path(path)}}
    resolved, record = finalizer._verified_reference(parent, "item")
    assert resolved == path.resolve()
    assert record["scope"] == "development"

    parent["item"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        finalizer._verified_reference(parent, "item")


def test_family_summary_keeps_each_family_separate() -> None:
    acceptance = {
        "families": {
            "field": {
                "passes": True,
                "check_count": 10,
                "failed_check_count": 0,
                "blocks_passing": 6,
                "blocks_required": 5,
            },
            "spectral": {
                "passes": False,
                "check_count": 20,
                "failed_check_count": 2,
                "blocks_passing": 4,
                "blocks_required": 5,
            },
            "transport": {
                "passes": True,
                "check_count": 30,
                "failed_check_count": 0,
                "blocks_passing": 5,
                "blocks_required": 5,
            },
        }
    }
    summary = finalizer._family_summary(acceptance)
    assert summary["field"]["passes"] is True
    assert summary["spectral"]["passes"] is False
    assert summary["transport"]["blocks_passing"] == 5


def test_finalizer_is_gate_only_and_cannot_authorize_downstream_work() -> None:
    source = inspect.getsource(finalizer)
    assert "score_b2_forecast" not in source
    assert '"metrics_recomputed": False' in source
    assert '"training_performed": False' in source
    assert '"inference_performed": False' in source
    assert '"truth_scoring_performed": False' in source
    assert '"seed1702_1703_training_authorized": False' in source
    assert '"O3_launch_allowed": False' in source
    assert '"held_out_85606_access_allowed": False' in source


def test_launcher_is_cpu_only_and_locks_the_frozen_gate_sources() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in text
    assert "#SBATCH --qos=gen" in text
    assert "#SBATCH --gres" not in text
    assert "B3_TRAINING_JOB_ID" in text
    assert "B3_EVALUATION_JOB_ID" in text
    assert "B3_COMPARATOR_JOB_ID" in text
    assert "36640742fea989cc5e94604fc6a2970cf431802ca32f176ca06302158a454737" in text
    assert "53ece13a1f3bb520ad52056a970b76a362b8319405af842d1bfb12b7bbee5896" in text
    assert "6bb5d825b30c9c8292cda020d3bec824d9b04198617dc89afafa264daab44ea5" in text
    assert "899bf42d7d709badd2631e5b94db11e6c6603635d2ce5f0cf341e747216f86d3" in text
    assert '"seed1702_1703_training_authorized"' in text
    assert '"O3_launch_allowed"' in text
