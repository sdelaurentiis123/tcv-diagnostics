"""Contract tests for the frozen B4 truth-separated evaluator entrypoint."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import paper0.tools.evaluate_b4_pde_refiner_checkpoint as evaluator
from paper0.tools.evaluate_b4_pde_refiner_checkpoint import (
    _write_index,
    audit_full_training_result,
    audit_history,
    validate_bounded_smoke_result,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.models.o2 import O2ViTConfig
from tcv_diagnostics.models.pde_refiner import PDERefinerConfig
from tcv_diagnostics.pde_refiner_full_training import (
    B4_FULL_LEVEL_COUNTS,
    B4_FULL_LEVEL_RAW_SHA256,
    B4_VALIDATION_BANK_NPY_SHA256,
    PDERefinerFullConfig,
    full_learning_rate,
    full_training_levels,
)
from tcv_diagnostics.pde_refiner_training import RefinerParentArtifacts


TRAINING_COMMIT = "0" * 40
FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")


def _validation(value: float) -> dict:
    levels = [value + 0.01 * index for index in range(4)]
    by_level = [
        {field: level + 0.001 * channel for channel, field in enumerate(FIELDS)}
        for level in levels
    ]
    return {
        "target_count": 126,
        "ensemble_members": 2,
        "checkpoint_weights": "EMA",
        "refinement_levels": [0, 1, 2, 3],
        "equal_channel_MAE_by_level": levels,
        "MAE_by_level_and_channel": by_level,
        "ensemble_mean_equal_channel_decoded_standardized_field_MAE": levels[3],
        "final_MAE_by_channel": by_level[3],
        "physics_metrics_used": False,
    }


def _artifacts() -> RefinerParentArtifacts:
    return RefinerParentArtifacts(
        checkpoint_path=Path("/tmp/parent.pt"),
        checkpoint_sha256="1" * 64,
        codec_path=Path("/tmp/codec.pt"),
        codec_sha256="2" * 64,
        latent_normalization_path=Path("/tmp/latent.json"),
        latent_normalization_sha256="3" * 64,
    )


def _training_record(selected_epoch: int = 9) -> dict:
    artifacts = _artifacts()
    config = PDERefinerFullConfig.frozen(seed=1701)
    groups = {
        "parent_parameter_tensor_count": 2,
        "refinement_parameter_tensor_count": 2,
        "parent_parameter_count": 51_612_800,
        "refinement_parameter_count": 9_606_144,
        "total_parameter_count": 61_218_944,
        "parent_parameter_names": ["parent.weight", "parent.bias"],
        "refinement_parameter_names": ["refiner.weight", "refiner.bias"],
    }
    parent_identity = {"bitwise_exact": True}
    load_audit = {"passed": True}
    run_config = config.to_record()
    run_config.update(
        {
            "model": O2ViTConfig().to_record(),
            "pde_refiner": PDERefinerConfig().to_record(),
            "parameter_groups": groups,
            "deterministic_parent": {
                "path": str(artifacts.checkpoint_path),
                "sha256": artifacts.checkpoint_sha256,
                "load_audit": load_audit,
                "preoptimization_identity": parent_identity,
            },
            "codec_checkpoint": {
                "path": str(artifacts.codec_path),
                "sha256": artifacts.codec_sha256,
                "trainable": False,
            },
            "latent_normalization": {
                "path": str(artifacts.latent_normalization_path),
                "sha256": artifacts.latent_normalization_sha256,
                "refit": False,
            },
            "training_levels": {
                "seed": 41001,
                "shape": [100, 430],
                "counts": list(B4_FULL_LEVEL_COUNTS),
                "raw_C_order_sha256": B4_FULL_LEVEL_RAW_SHA256,
                "npy_sha256": "7" * 64,
            },
            "validation_seed_bank": {
                "seed": 41003,
                "shape": [126, 2, 3],
                "dtype": "uint64",
                "npy_sha256": B4_VALIDATION_BANK_NPY_SHA256,
            },
        }
    )
    selected_validation = _validation(0.10)
    final_validation = _validation(0.20)
    return {
        "scope": "B4_PDE_Refiner_H1_seed1701_full_training_85604",
        "paper0_commit": TRAINING_COMMIT,
        "completed_epochs": 100,
        "completed_optimizer_steps": 2700,
        "EMA_updates": 2700,
        "validation_candidates_evaluated": 20,
        "validation_completed_epochs": list(config.validation_completed_epochs),
        "checkpoint_reload_bitwise_exact": True,
        "checkpoint_reload": {
            "latent_bitwise_exact": True,
            "forecast_bitwise_exact": True,
        },
        "parent_parameter_gradient_seen": True,
        "refinement_parameter_gradient_seen": True,
        "training_level_counts": list(B4_FULL_LEVEL_COUNTS),
        "all_four_training_levels_exercised": True,
        "codec_state_sha256_before": "8" * 64,
        "codec_state_sha256_after": "8" * 64,
        "codec_bitwise_unchanged": True,
        "training_dtype": "float32",
        "validation_dtype": "float32",
        "torch_float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_deterministic_requested": True,
        "physics_derived_loss_used": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "scientific_result": False,
        "training_complete_is_scientific_acceptance": False,
        "full_B4_training_authorized": True,
        "scientific_B4_evaluation_performed": False,
        "H_det_evaluated": False,
        "H_prob_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "config": run_config,
        "parameter_count": 61_218_944,
        "parameter_groups": groups,
        "selected_epoch": selected_epoch,
        "selected_completed_epoch": selected_epoch + 1,
        "selected_optimizer_step": (selected_epoch + 1) * 27,
        "selected_validation": selected_validation,
        "final_validation": final_validation,
        "preoptimization_parent_identity": parent_identity,
        "deterministic_parent_load_audit": load_audit,
        "deterministic_parent": {
            "path": str(artifacts.checkpoint_path),
            "sha256": artifacts.checkpoint_sha256,
        },
        "codec_checkpoint": {
            "path": str(artifacts.codec_path),
            "sha256": artifacts.codec_sha256,
            "trainable": False,
        },
        "latent_normalization": {
            "path": "/tmp/copied-latent.json",
            "sha256": artifacts.latent_normalization_sha256,
            "refit": False,
        },
        "selected_checkpoint": {"path": "/tmp/selected.pt", "sha256": "4" * 64},
        "final_training_state": {"path": "/tmp/final.pt", "sha256": "5" * 64},
        "history": {"path": "/tmp/history.jsonl", "sha256": "6" * 64},
        "validation_seed_bank": {
            "path": "/tmp/selection.npy",
            "sha256": B4_VALIDATION_BANK_NPY_SHA256,
            "seed": 41003,
            "shape": [126, 2, 3],
        },
        "training_levels": {
            "path": "/tmp/levels.npy",
            "sha256": "7" * 64,
            "raw_C_order_sha256": B4_FULL_LEVEL_RAW_SHA256,
            "seed": 41001,
            "shape": [100, 430],
        },
    }


def _history(path: Path, selected_epoch: int = 9) -> tuple[dict, dict]:
    config = PDERefinerFullConfig.frozen(seed=1701)
    levels = full_training_levels(config)
    records = []
    running_epoch = None
    running_value = float("inf")
    selected_validation = None
    final_validation = None
    for epoch in range(100):
        validation = None
        if epoch + 1 in config.validation_completed_epochs:
            value = 0.10 if epoch == selected_epoch else 0.20 + epoch / 1000.0
            validation = _validation(value)
            if validation[
                "ensemble_mean_equal_channel_decoded_standardized_field_MAE"
            ] < running_value:
                running_value = validation[
                    "ensemble_mean_equal_channel_decoded_standardized_field_MAE"
                ]
                running_epoch = epoch
            if epoch == selected_epoch:
                selected_validation = validation
            final_validation = validation
        counts = np.bincount(levels[epoch], minlength=4)
        global_step = 27 * (epoch + 1)
        records.append(
            {
                "epoch": epoch,
                "completed_epoch": epoch + 1,
                "examples": 430,
                "global_step": global_step,
                "learning_rate": full_learning_rate(config, global_step - 1),
                "EMA_decay": 0.995,
                "EMA_updates": global_step,
                "train_standardized_latent_MSE": 0.5,
                "train_MSE_by_level": {str(level): 0.5 for level in range(4)},
                "train_count_by_level": {
                    str(level): int(counts[level]) for level in range(4)
                },
                "mean_preclip_total_gradient_norm": 0.2,
                "maximum_preclip_total_gradient_norm": 0.3,
                "mean_preclip_parent_gradient_norm": 0.1,
                "mean_preclip_refinement_gradient_norm": 0.1,
                "validation_performed": validation is not None,
                "validation": validation,
                "selected_so_far": running_epoch,
                "epoch_wall_seconds": 1.0,
            }
        )
    assert selected_validation is not None and final_validation is not None
    path.write_text("".join(json.dumps(item) + "\n" for item in records))
    return selected_validation, final_validation


def test_training_audit_accepts_only_exact_full_seed1701_contract() -> None:
    record = _training_record()
    audited = audit_full_training_result(
        record,
        training_commit=TRAINING_COMMIT,
        artifacts=_artifacts(),
    )
    assert audited["selected_epoch"] == 9
    assert audited["parameter_count"] == 61_218_944

    forged = copy.deepcopy(record)
    forged["completed_optimizer_steps"] = 2699
    with pytest.raises(ValueError, match="completed_optimizer_steps"):
        audit_full_training_result(
            forged,
            training_commit=TRAINING_COMMIT,
            artifacts=_artifacts(),
        )
    forged = copy.deepcopy(record)
    forged["config"]["pde_refiner"]["refinement_steps"] = 4
    with pytest.raises(ValueError, match="refiner configuration"):
        audit_full_training_result(
            forged,
            training_commit=TRAINING_COMMIT,
            artifacts=_artifacts(),
        )


def test_history_audit_recomputes_sparse_earliest_minimum_and_schedule(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.jsonl"
    selected, final = _history(path)
    result = audit_history(
        path,
        expected_sha256=sha256_path(path),
        selected_epoch=9,
        selected_validation=selected,
        final_validation=final,
    )
    assert result["epochs"] == 100
    assert result["optimizer_steps"] == 2700
    assert result["validation_candidates"] == 20
    assert result["earliest_validation_minimum_epoch"] == 9

    with pytest.raises(ValueError, match="earliest fixed-seed"):
        audit_history(
            path,
            expected_sha256=sha256_path(path),
            selected_epoch=14,
            selected_validation=selected,
            final_validation=final,
        )


def test_full_mode_requires_exact_passing_four_target_b4_smoke() -> None:
    smoke = {
        "scope": "bounded_non_scientific_B4_PDE_Refiner_H1_evaluator_smoke_85604",
        "status": "bounded_evaluator_smoke_completed",
        "paper0_commit": "a" * 40,
        "seed": 1701,
        "target_frames": [498, 502],
        "target_count": 4,
        "final_ensemble_members": 32,
        "stage_prefix_members": 4,
        "held_out_85606_read": False,
        "truth_opened_only_after_both_forecast_hashes": True,
        "full_evaluation_preconditions_passed": True,
        "H_det_evaluated": False,
        "H_prob_evaluated": False,
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


def test_artifact_index_does_not_rehash_verified_large_forecasts(
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


def test_entrypoint_opens_truth_only_after_both_closed_hashed_forecasts() -> None:
    source = inspect.getsource(evaluator.main)
    assert source.index(
        "generation = generate_selected_pde_refiner_forecasts"
    ) < source.index("native_truth = NativeTruthCatalog")
    assert source.index(
        "write_strict_json_atomic(generation_path, generation)"
    ) < source.index("native_truth = NativeTruthCatalog")
    assert source.index("artifact_integrity = _artifact_integrity") < source.index(
        "native_truth = NativeTruthCatalog"
    )
    assert "PDERefinerFinalForecastArtifact" in source
    assert "PDERefinerStageForecastArtifact" in source
    assert "tuple(range(498, 502))" in source
    assert "tuple(range(498, 624))" in source
