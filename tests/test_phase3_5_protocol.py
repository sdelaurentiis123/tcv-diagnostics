from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper0" / "PHASE3_5_PROTOCOL_AMENDMENT.md"
CLARIFICATION = (
    ROOT / "paper0" / "PHASE3_5_PROTOCOL_AMENDMENT_2026-08-19A.md"
)
MANIFEST = (
    ROOT / "paper0" / "manifests" / "phase3_5_cause_localization_85604.json"
)
AUDIT = ROOT / "paper0" / "phase3_5" / "DATA_STATE_AUDIT.md"
LAUNCHER = ROOT / "cluster" / "phase3_5_cause_localization.sbatch"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_phase3_5_protocol_files_exist_and_hash_lock() -> None:
    assert PROTOCOL.is_file()
    assert AUDIT.is_file()
    record = _manifest()
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert record["protocol"] == {
        "path": "paper0/PHASE3_5_PROTOCOL_AMENDMENT.md",
        "sha256": digest,
    }
    assert record["clarifying_amendments"] == [
        {
            "path": "paper0/PHASE3_5_PROTOCOL_AMENDMENT_2026-08-19A.md",
            "sha256": hashlib.sha256(CLARIFICATION.read_bytes()).hexdigest(),
        }
    ]


def test_phase3_5_scope_keeps_guard_and_held_out_closed() -> None:
    record = _manifest()
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert record["data"]["guard_frames"] == [432, 496]
    forbidden = set(record["forbidden_scope"])
    assert "guard_reads" in forbidden
    assert "85606_discovery_or_access" in forbidden
    assert "production_scale_neural_training_or_finetuning" in forbidden
    assert "assimilation" in forbidden
    assert "diagnostic_ranking" in forbidden


def test_phase3_5_blocks_are_disjoint_ordered_and_matched() -> None:
    blocks = _manifest()["blocks"]
    training = blocks["training"]
    validation = blocks["validation"]
    assert len(training) == 10
    assert len(validation) == 3
    assert training[0]["range"] == [2, 45]
    assert training[-1]["range"] == [389, 432]
    assert validation[0]["range"] == [498, 540]
    assert validation[-1]["range"] == [582, 624]
    assert all(
        current["range"][1] == following["range"][0]
        for current, following in zip(training, training[1:])
    )
    assert all(
        current["range"][1] == following["range"][0]
        for current, following in zip(validation, validation[1:])
    )
    matched = training + validation
    assert all(
        block["matched_range"][1] - block["matched_range"][0]
        == blocks["matched_sample_count"]
        for block in matched
    )
    assert training[-1]["range"][1] < validation[0]["range"][0]


def test_phase3_5_only_toroidal_axis_is_periodic() -> None:
    record = _manifest()
    assert record["data"]["periodic_axes_xyz"] == [False, False, True]
    assert record["translation"]["axis"] == "z"
    assert record["equivariance"]["nonperiodic_roll_allowed"] is False
    assert record["data"]["zperiod"] == 5
    assert record["data"]["mode_mapping"] == "n=5k"


def test_phase3_5_analysis_budget_and_no_automatic_next_action() -> None:
    record = _manifest()
    assert record["representations"]["coefficient_budgets_real"] == [
        32,
        64,
        128,
        256,
        416,
    ]
    assert record["uncertainty"]["replicates"] == 200
    assert record["uncertainty"]["primary_block_length"] == 12
    assert record["equivariance"]["all_integer_z_shifts"] is True
    assert record["post_phase"]["architecture_training_authorized"] is False
    assert record["post_phase"]["recommended_action_automatically_authorized"] is False
    assert record["post_phase"]["85606_access_authorized"] is False


def test_phase3_5_protocol_records_narrow_k4_statement() -> None:
    text = PROTOCOL.read_text()
    normalized = " ".join(text.split())
    required = (
        "A single, fixed, condition-independent, global linear residual "
        "distribution\n> learned from adjacent 85604 training frames does not "
        "describe later 85604\n> residuals well."
    )
    assert required in text
    assert "stochastic emulation in general" in normalized
    assert "No production-scale neural model" in normalized


def test_phase3_5_launcher_is_bounded_and_commit_locked() -> None:
    text = LAUNCHER.read_text()
    assert "#SBATCH --constraint=h100" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=24" in text
    assert "#SBATCH --mem=128G" in text
    assert "#SBATCH --time=08:00:00" in text
    assert "PAPER0_EXPECTED_COMMIT" in text
    assert "status --porcelain --untracked-files=all" in text
    assert "WANDB_MODE=online" in text
    assert "run_phase3_5_cause_localization.py" in text
    assert "tests/test_phase3_5_protocol.py" in text
    assert "tests/test_phase3_5_analysis.py" in text
