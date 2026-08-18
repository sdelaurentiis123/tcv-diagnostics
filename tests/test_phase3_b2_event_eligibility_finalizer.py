from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_b2_event_eligibility_amendment",
    ROOT / "paper0/tools/finalize_b2_event_eligibility_amendment.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _original() -> dict[str, object]:
    return {
        "scope": "phase3_B2_LDM_H2_full_probabilistic_evaluation_matrix_85604",
        "status": "completed_failed_frozen_one_step_gate",
        "scientific_authority": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "slurm_job_id": "6897564",
        "paper0_commit": "361f0f27a9ece3b56f529a72c2fcfa19aa0be719",
        "architecture_acceptance": {
            "architecture_passes_one_step_B2_gate": False
        },
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "evaluation_inputs": [
            {"seed": seed} for seed in (1701, 1702, 1703)
        ],
    }


def test_A016_finalizer_locks_original_failed_matrix() -> None:
    MODULE._validate_original_matrix(
        _original(), digest=MODULE.ORIGINAL_MATRIX_SHA256
    )
    changed = _original()
    changed["status"] = "completed_passed_frozen_one_step_gate"
    with pytest.raises(ValueError, match="contract differs"):
        MODULE._validate_original_matrix(
            changed, digest=MODULE.ORIGINAL_MATRIX_SHA256
        )
    with pytest.raises(ValueError, match="hash differs"):
        MODULE._validate_original_matrix(_original(), digest="0" * 64)


def test_A016_finalizer_accepts_only_frozen_gate_only_amendment() -> None:
    manifest = load_strict_json(
        ROOT
        / "paper0/manifests/phase3_b2_event_eligibility_amendment_85604.json"
    )
    MODULE._validate_amendment(
        manifest,
        original_digest=MODULE.ORIGINAL_MATRIX_SHA256,
        protocol_digest=manifest["protocol"]["sha256"],
    )
    contaminated = dict(manifest)
    contaminated["held_out_85606_access_allowed"] = True
    with pytest.raises(ValueError, match="manifest contract differs"):
        MODULE._validate_amendment(
            contaminated,
            original_digest=MODULE.ORIGINAL_MATRIX_SHA256,
            protocol_digest=manifest["protocol"]["sha256"],
        )
