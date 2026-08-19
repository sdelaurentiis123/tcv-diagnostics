"""Network-free checks for residual-KL compact W&B orchestration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from paper0.tools.run_residual_kl_oracle_wandb import (
    commands,
    compact_metrics,
    validate_completed,
    validate_scope,
)
from tcv_diagnostics.b5_residual_audit import B5_FIELDS
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES


def _args(tmp_path: Path) -> Namespace:
    builder = tmp_path / "builder.py"
    evaluator = tmp_path / "evaluator.py"
    builder.write_text("print('builder')\n", encoding="utf-8")
    evaluator.write_text("print('evaluator')\n", encoding="utf-8")
    common = tmp_path / "input"
    common.write_text("input\n", encoding="utf-8")
    return Namespace(
        builder=builder,
        builder_sha256=sha256_path(builder),
        evaluator=evaluator,
        evaluator_sha256=sha256_path(evaluator),
        oracle_manifest=common,
        oracle_manifest_sha256="1" * 64,
        oracle_protocol=common,
        oracle_protocol_sha256="2" * 64,
        decision_memo=common,
        decision_memo_sha256="3" * 64,
        artifact_root=tmp_path / "model_data",
        h1_training_forecast=common,
        h1_training_forecast_sha256="4" * 64,
        h1_validation_forecast=common,
        h1_validation_forecast_sha256="5" * 64,
        training_audit=common,
        training_audit_sha256="6" * 64,
        training_raw=common,
        training_raw_sha256="7" * 64,
        native_truth_result=common,
        native_truth_result_sha256="8" * 64,
        geometry_manifest=common,
        geometry_manifest_sha256="9" * 64,
        geometry=common,
        geometry_sha256="a" * 64,
        b5_localization_result=common,
        b5_localization_result_sha256="b" * 64,
        pretruth_output=tmp_path / "pretruth_output",
        evaluation_output=tmp_path / "evaluation_output",
        scratch_root=tmp_path / "scratch",
        paper0_commit="c" * 40,
        slurm_job_id="321",
    )


def test_wrapper_closes_pretruth_before_constructing_evaluator(tmp_path: Path) -> None:
    args = _args(tmp_path)
    builder, evaluator = validate_scope(args)
    build, evaluate = commands(args, builder=builder, evaluator=evaluator)
    assert evaluate is None
    assert "--native-truth-result" not in build
    assert "--h1-training-forecast" in build
    build_again, evaluate = commands(
        args,
        builder=builder,
        evaluator=evaluator,
        closure_sha256="d" * 64,
    )
    assert build_again == build
    assert evaluate is not None
    assert "--pretruth-closure-sha256" in evaluate
    assert evaluate[evaluate.index("--pretruth-closure-sha256") + 1] == "d" * 64
    assert "--native-truth-result" in evaluate
    assert "--h1-training-forecast" not in evaluate


def test_wrapper_rejects_heldout_or_existing_output(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.geometry = tmp_path / "forbidden_85606.nc"
    with pytest.raises(ValueError, match="held-out"):
        validate_scope(args)
    args = _args(tmp_path)
    args.pretruth_output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        validate_scope(args)


def _records() -> tuple[dict, dict, dict]:
    pretruth = {
        "status": "completed_and_closed_before_validation_truth",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "paper0_commit": "c" * 40,
        "slurm_job_id": "321",
    }
    result = {
        "status": "completed_without_model_training_or_downstream_opening",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "paper0_commit": "c" * 40,
        "slurm_job_id": "321",
        "positive_rank": 429,
        "selected_static_rank": 64,
        "tier_A_minimum_passing_rank": 32,
        "tier_B_static_covariance_useful": True,
        "wall_seconds": 12.0,
    }
    field_records = {
        field: {
            "ensemble_mean_rmse": 1.0,
            "fair_crps": 0.5,
            "corrected_spread_skill_ratio": 1.0,
        }
        for field in B5_FIELDS
    }
    covariance = {
        "local_corrected_spread_skill_ratio": 1.0,
        "integrated_corrected_spread_skill_ratio": 0.9,
        "ensemble_to_error_coherence_multiplier_ratio": 0.8,
    }
    static = {
        "field_and_marginal_calibration": {
            "regions": {
                "eligible_union": {
                    "aggregate": {
                        "equal_channel_corrected_spread_skill_ratio": 1.0,
                        "equal_channel_ensemble_mean_rmse": 1.0,
                        "equal_channel_fair_crps": 0.5,
                    },
                    "fields": field_records,
                }
            },
            "chronological_blocks_eligible_union": [{} for _ in range(6)],
        },
        "transport_covariance": {
            "quantities": {
                quantity: {"covariance_decomposition": covariance}
                for quantity in TRANSPORT_QUANTITIES
            }
        },
    }
    rank = {
        "variance_capture": {"total": 0.9},
        "representation_gate": {"passes": True},
        "dependence": {"identity_pass_count": 10},
        "material_power": {"in_range_count": 13},
        "chronological_blocks": {f"b{i}": {} for i in range(6)},
    }
    boundaries = {
        flag: False
        for flag in (
            "checkpoint_loaded",
            "model_inference_performed",
            "optimizer_or_trainable_parameter_created",
            "model_training_performed",
            "physics_metric_used_as_training_loss",
            "O3_launched",
            "O4_launched",
            "O5_launched",
            "assimilation_performed",
            "diagnostic_ranking_performed",
            "steering_performed",
            "held_out_85606_read",
        )
    }
    scientific = {
        "status": "completed_without_model_training_or_downstream_opening",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "paper0_commit": "c" * 40,
        "slurm_job_id": "321",
        "pretruth_closure": {"sha256": "d" * 64},
        "validation_truth_opened_only_after_pretruth_closure_verified": True,
        "scientific_boundaries": boundaries,
        "training_basis_summary": {
            "effective_rank": {
                "participation_ratio": 11.0,
                "entropy_effective_rank": 13.0,
            }
        },
        "tier_A_truth_projected_representation_oracle": {"rank_32": rank},
        "tier_B_static_Gaussian_KL": static,
    }
    return pretruth, result, scientific


def test_completed_scope_and_compact_metric_projection() -> None:
    pretruth, result, scientific = _records()
    validate_completed(
        pretruth=pretruth,
        result=result,
        scientific=scientific,
        paper0_commit="c" * 40,
        slurm_job_id="321",
        closure_sha256="d" * 64,
    )
    metrics = compact_metrics(result, scientific)
    assert metrics["oracle/tier_A_minimum_passing_rank"] == 32
    assert metrics["tier_A/rank_32/gate_passed"] is True
    assert metrics["static_transport/particle/integrated_spread_skill"] == 0.9
    scientific["scientific_boundaries"]["model_training_performed"] = True
    with pytest.raises(RuntimeError, match="forbidden scope"):
        validate_completed(
            pretruth=pretruth,
            result=result,
            scientific=scientific,
            paper0_commit="c" * 40,
            slurm_job_id="321",
            closure_sha256="d" * 64,
        )


def test_wrapper_never_uploads_local_scientific_artifacts() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "paper0/tools/run_residual_kl_oracle_wandb.py"
    ).read_text(encoding="utf-8")
    assert "wandb.log_artifact" not in source
    for record in (
        '"forecasts_uploaded": False',
        '"simulation_fields_uploaded": False',
        '"basis_arrays_uploaded": False',
        '"raw_accumulators_uploaded": False',
        '"figures_uploaded": False',
        '"tables_uploaded": False',
    ):
        assert record in source
