"""Contract checks for parent-versus-feedback state and physics evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import h5py
import pytest

from paper0.tools.evaluate_four_step_feedback_pilot import (
    BANDS,
    FIELDS,
    HORIZONS,
    METHODS,
    QUANTITIES,
    authorize_evaluation_manifest,
    create_forecast_file,
    physics_preservation_decision,
)
from tcv_diagnostics.codec_training import sha256_path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/post_ecrd_old_85604_four_step_feedback_evaluation.sbatch"


def _manifest() -> dict:
    return {
        "scope": "post_ecrd_old_85604_four_step_feedback_evaluation",
        "status": "frozen_after_training_before_physics_evaluation",
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "new_nersc_data_access_allowed": False,
        "guard_frames_read_allowed": False,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "steering_allowed": False,
        "physics_derived_loss_used": False,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "fields": list(FIELDS),
        "horizons": list(HORIZONS),
        "methods": list(METHODS),
        "inference_batch_size": 4,
        "wandb_required": True,
        "physics_preservation_gates": {
            "maximum_absolute_log_power_ratio_error_increase_fraction": 0.10,
            "maximum_mean_separatrix_relative_l2_increase_fraction": 0.05,
            "strict_face_transport_is_report_only": True,
            "cross_field_coherence_change_is_report_only": True,
        },
    }


def test_evaluation_manifest_is_hash_and_scope_locked(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    authorize_evaluation_manifest(
        _manifest(), manifest_path=path, manifest_sha256=sha256_path(path)
    )
    broken = copy.deepcopy(_manifest())
    broken["training_allowed"] = True
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="scope"):
        authorize_evaluation_manifest(
            broken, manifest_path=path, manifest_sha256=sha256_path(path)
        )


def test_evaluation_launcher_is_one_gpu_and_evaluation_only() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=32G" in source
    assert "#SBATCH --time=01:00:00" in source
    assert "export WANDB_MODE=online" in source
    assert "--array" not in source
    assert "85606" not in source


def test_forecast_file_has_only_two_models_and_two_horizons(tmp_path: Path) -> None:
    path = tmp_path / "forecast.h5"
    with create_forecast_file(
        path,
        paper0_commit="a" * 40,
        manifest_sha256="b" * 64,
        training_result_sha256="c" * 64,
    ) as handle:
        assert set(handle) == {"horizon_4", "horizon_8"}
        assert json.loads(handle.attrs["methods"]) == list(METHODS)
        assert json.loads(handle.attrs["fields"]) == list(FIELDS)
        for horizon in HORIZONS:
            group = handle[f"horizon_{horizon}"]
            assert set(group) == {"current_frame", "target_frame", *METHODS}
            for method in METHODS:
                assert group[method].shape == (
                    624 - horizon - 496,
                    len(FIELDS),
                    64,
                    32,
                    88,
                )
                assert group[method].compression == "gzip"
    with h5py.File(path, "r") as handle:
        assert handle.attrs["held_out_85606_read"] == 0


def _field_physics(power_ratio: float) -> dict:
    return {
        "field_band_summaries": {
            field: {band: {"power_ratio": power_ratio} for band in BANDS}
            for field in FIELDS
        }
    }


def _transport(relative_l2: float) -> dict:
    return {
        "comparisons": {
            f"truth_vs_{method}": {
                "quantities": {
                    quantity: {
                        "separatrix": {"metrics": {"relative_l2": relative_l2}}
                    }
                    for quantity in QUANTITIES
                }
            }
            for method in METHODS
        }
    }


def _horizon_record(
    *, parent_power: float, candidate_power: float, transport_l2: float
) -> dict:
    return {
        "field_spectral_cross": {
            "pre_feedback_parent": _field_physics(parent_power),
            "four_step_feedback_finetuned": _field_physics(candidate_power),
        },
        "transport": _transport(transport_l2),
    }


def test_physics_gate_cannot_rescue_a_failed_state_gate() -> None:
    records = {
        str(horizon): _horizon_record(
            parent_power=0.5,
            candidate_power=0.51,
            transport_l2=0.4,
        )
        for horizon in HORIZONS
    }
    gates = _manifest()["physics_preservation_gates"]
    passed = physics_preservation_decision(
        by_horizon=records, state_pilot_passed=True, gates=gates
    )
    assert passed["advance_to_confirmation_seeds"] is True
    failed = physics_preservation_decision(
        by_horizon=records, state_pilot_passed=False, gates=gates
    )
    assert failed["advance_to_confirmation_seeds"] is False
    assert failed["gates"]["state_pilot_passed"] is False
