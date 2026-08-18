"""Contract tests for the frozen B3 FGN truth-separated evaluator."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

import paper0.tools.evaluate_b3_fgn_checkpoint as evaluator
from paper0.tools.evaluate_b3_fgn_checkpoint import (
    _write_index,
    audit_full_training_result,
    audit_history,
    validate_bounded_smoke_result,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.fgn_training import FGNRunConfig, ParentArtifacts


TRAINING_COMMIT = "a2a17cf3fc30fd504bc3eee3274e78623bf15e2b"
FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")


def _validation(value: float) -> dict:
    return {
        "examples": 126,
        "equal_channel_fair_crps": value,
        "fair_crps_by_channel": {field: value + 0.01 for field in FIELDS},
        "accuracy_by_channel": {field: value + 0.02 for field in FIELDS},
        "spread_by_channel": {field: value + 0.03 for field in FIELDS},
    }


def _artifacts() -> ParentArtifacts:
    return ParentArtifacts(
        checkpoint_path=Path("/tmp/parent.pt"),
        checkpoint_sha256="1" * 64,
        codec_path=Path("/tmp/codec.pt"),
        codec_sha256="2" * 64,
        latent_normalization_path=Path("/tmp/norm.json"),
        latent_normalization_sha256="3" * 64,
    )


def _training_record() -> dict:
    artifacts = _artifacts()
    return {
        "scope": "B3_FGN_H1_seed1701_full_training_85604",
        "paper0_commit": TRAINING_COMMIT,
        "completed_epochs": 100,
        "completed_optimizer_steps": 2700,
        "checkpoint_reload_bitwise_exact": True,
        "codec_bitwise_unchanged": True,
        "common_parameter_gradient_seen": True,
        "new_parameter_gradient_seen": True,
        "cudnn_deterministic_requested": True,
        "tf32_allowed": False,
        "physics_derived_loss_used": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "scientific_result": False,
        "training_complete_is_scientific_acceptance": False,
        "full_B3_training_authorized": True,
        "probabilistic_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "config": FGNRunConfig.frozen(mode="full", seed=1701).to_record(),
        "selected_epoch": 37,
        "selected_validation": _validation(0.1),
        "final_validation": _validation(0.2),
        "preoptimization_parent_identity": {"bitwise_exact": True},
        "deterministic_parent_load_audit": {"passed": True},
        "member_probe": {
            "target_frame_index": 498,
            "ensemble_size": 2,
            "canonical_forecast_shape": [1, 2, 1, 5, 64, 32, 88],
            "finite": True,
            "reload_latent_bitwise_exact": True,
            "reload_forecast_bitwise_exact": True,
            "nonzero_latent_diversity": True,
            "nonzero_field_diversity": True,
        },
        "deterministic_parent": {
            "path": str(artifacts.checkpoint_path),
            "sha256": artifacts.checkpoint_sha256,
        },
        "codec_checkpoint": {
            "path": str(artifacts.codec_path),
            "sha256": artifacts.codec_sha256,
            "trainable": False,
        },
        "selected_checkpoint": {"path": "/tmp/selected.pt", "sha256": "4" * 64},
        "final_training_state": {"path": "/tmp/final.pt", "sha256": "5" * 64},
        "history": {"path": "/tmp/history.jsonl", "sha256": "6" * 64},
        "validation_noise_bank": {
            "path": "/tmp/selection.npy",
            "sha256": "7" * 64,
            "seed": 31003,
            "shape": [126, 2, 32],
        },
        "parameter_count": 51_700_000,
    }


def _history(path: Path, minimum_epoch: int = 37) -> tuple[dict, dict]:
    values = [1.0 + abs(epoch - minimum_epoch) / 1000.0 for epoch in range(100)]
    records = []
    selected = 0
    for epoch, value in enumerate(values):
        if value < values[selected]:
            selected = epoch
        records.append(
            {
                "epoch": epoch,
                "examples": 430,
                "ensemble_members": 2,
                "global_step": 27 * (epoch + 1),
                "common_learning_rate": 1.0e-5,
                "new_learning_rate": 2.0e-5,
                "train_equal_channel_fair_crps": value + 0.1,
                "validation_equal_channel_fair_crps": value,
                "validation_fair_crps_by_channel": {
                    field: value + 0.01 for field in FIELDS
                },
                "validation_accuracy_by_channel": {
                    field: value + 0.02 for field in FIELDS
                },
                "validation_spread_by_channel": {
                    field: value + 0.03 for field in FIELDS
                },
                "mean_preclip_total_gradient_norm": 0.2,
                "maximum_preclip_total_gradient_norm": 0.3,
                "mean_preclip_common_gradient_norm": 0.1,
                "mean_preclip_new_gradient_norm": 0.1,
                "epoch_wall_seconds": 1.0,
                "selected_so_far": selected,
            }
        )
    path.write_text("".join(json.dumps(item) + "\n" for item in records))
    return _validation(values[minimum_epoch]), _validation(values[-1])


def test_training_audit_accepts_only_exact_completed_seed1701_contract() -> None:
    record = json.loads(json.dumps(_training_record()))
    audited = audit_full_training_result(
        record,
        training_commit=TRAINING_COMMIT,
        artifacts=_artifacts(),
    )
    assert audited["selected_epoch"] == 37
    assert audited["parameter_count"] == 51_700_000

    forged = copy.deepcopy(record)
    forged["completed_optimizer_steps"] = 2699
    with pytest.raises(ValueError, match="completed_optimizer_steps"):
        audit_full_training_result(
            forged,
            training_commit=TRAINING_COMMIT,
            artifacts=_artifacts(),
        )


def test_history_audit_recomputes_earliest_fixed_noise_minimum(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    selected, final = _history(path)
    result = audit_history(
        path,
        expected_sha256=sha256_path(path),
        selected_epoch=37,
        selected_validation=selected,
        final_validation=final,
    )
    assert result["epochs"] == 100
    assert result["optimizer_steps"] == 2700
    assert result["earliest_validation_minimum_epoch"] == 37

    with pytest.raises(ValueError, match="earliest fixed-noise"):
        audit_history(
            path,
            expected_sha256=sha256_path(path),
            selected_epoch=38,
            selected_validation=selected,
            final_validation=final,
        )


def test_full_mode_requires_exact_passing_four_target_smoke() -> None:
    smoke = {
        "scope": "bounded_non_scientific_B3_FGN_H1_evaluator_smoke_85604",
        "status": "bounded_evaluator_smoke_completed",
        "paper0_commit": "a" * 40,
        "seed": 1701,
        "target_frames": [498, 502],
        "target_count": 4,
        "ensemble_members": 32,
        "held_out_85606_read": False,
        "truth_opened_only_after_forecast_hash": True,
        "full_probabilistic_evaluation_preconditions_passed": True,
        "probabilistic_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
        "training_result": {"sha256": "b" * 64},
    }
    validate_bounded_smoke_result(
        smoke,
        paper0_commit="a" * 40,
        training_result_sha256="b" * 64,
    )
    smoke["target_count"] = 126
    with pytest.raises(RuntimeError, match="smoke contract"):
        validate_bounded_smoke_result(
            smoke,
            paper0_commit="a" * 40,
            training_result_sha256="b" * 64,
        )


def test_artifact_index_does_not_rehash_verified_large_forecast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "forecast.h5"
    artifact.write_bytes(b"large-forecast-placeholder")

    def forbidden(path: Path) -> str:
        raise AssertionError(f"unexpected redundant hash of {path}")

    monkeypatch.setattr(evaluator, "sha256_path", forbidden)
    index = _write_index(
        tmp_path,
        [artifact],
        verified_sha256={artifact: "f" * 64},
    )
    assert index.read_text() == f"{'f' * 64}  {artifact.resolve()}\n"


def test_entrypoint_opens_truth_only_after_closed_hashed_generation() -> None:
    source = inspect.getsource(evaluator.main)
    assert source.index("generation = generate_selected_fgn_forecasts") < source.index(
        "native_truth = NativeTruthCatalog"
    )
    assert source.index("write_strict_json_atomic(generation_path, generation)") < (
        source.index("native_truth = NativeTruthCatalog")
    )
    assert 'context_frames=1' in source
    assert 'tuple(range(498, 502))' in source
    assert 'tuple(range(498, 624))' in source
