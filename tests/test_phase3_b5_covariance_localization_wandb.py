"""Network-free checks for the B5 covariance-localization W&B wrapper."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from paper0.tools.run_b5_covariance_localization_wandb import (
    localization_metrics,
    validate_completed_localization,
    validate_localization_command,
)
from tcv_diagnostics.codec_training import sha256_path


def _args(tmp_path: Path) -> Namespace:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("print('ok')\n", encoding="utf-8")
    output = tmp_path / "output"
    return Namespace(
        evaluator=evaluator,
        evaluator_sha256=sha256_path(evaluator),
        output_directory=output,
        paper0_commit="a" * 40,
        slurm_job_id="123",
        evaluation_args=[
            "--",
            "--output-directory",
            str(output),
            "--paper0-commit",
            "a" * 40,
            "--slurm-job-id",
            "123",
        ],
    )


def test_localization_wrapper_locks_scope_commit_job_and_heldout(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    command = validate_localization_command(args)
    assert command[-len(args.evaluation_args[1:]) :] == args.evaluation_args[1:]
    args.evaluation_args.append("/tmp/85606_forbidden")
    with pytest.raises(ValueError, match="held-out"):
        validate_localization_command(args)

    args = _args(tmp_path)
    args.evaluation_args.extend(["--checkpoint", "/tmp/model.pt"])
    with pytest.raises(ValueError, match="training or inference"):
        validate_localization_command(args)


def _completed_records() -> tuple[dict, dict]:
    result = {
        "scope": "B5_read_only_covariance_localization_85604",
        "status": "completed_without_retraining_or_downstream_opening",
        "paper0_commit": "a" * 40,
        "slurm_job_id": "123",
        "target_count": 126,
        "ensemble_size": 32,
        "wall_seconds": 10.0,
        "checkpoint_loaded": False,
        "model_inference_performed": False,
        "model_training_performed": False,
        "forecast_mutated": False,
        "additional_seed_trained": False,
        "O3_launched": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "held_out_85606_read": False,
    }
    covariance = {
        "local_corrected_spread_skill_ratio": 1.0,
        "integrated_corrected_spread_skill_ratio": 0.5,
        "ensemble_to_error_coherence_multiplier_ratio": 0.25,
        "counterfactual_local_spread_skill_after_same_factor": 2.0,
    }
    labels = {
        "L1_predominantly_amplitude_limited": {"supported": False},
        "L2_covariance_organization_limited": {"supported": True},
        "L4_explicit_residual_history_signal": {"supported": False},
        "L5_unresolved_by_one_realized_trajectory": {"supported": False},
    }
    localization = {
        "scope": "B5_read_only_covariance_localization_85604",
        "paper0_commit": "a" * 40,
        "slurm_job_id": "123",
        "integrity_anchors": {
            "legacy_training_reconstruction": {"passed": True},
            "B5_marginal_recomputation": {
                "passed": True,
                "recomputed": {
                    "equal_channel_ensemble_mean_RMSE": 0.1,
                    "equal_channel_corrected_spread_skill_ratio": 0.8,
                },
            },
        },
        "history_probe": {
            "AR1_vs_H1_equal_field_RMSE_improvement_fraction": 0.01,
            "AR1_improved_chronological_comparison_count": 4,
        },
        "interpretation_labels": labels,
        "blockwise_L3": {
            "systematic_identity_count": 5,
            "L3_field_dependence_mismatch_beyond_within_run_drift_supported": True,
        },
        "variogram_scores": {
            "field": {"aggregate_region_mean": {"global": 0.2}},
            "transport": {
                "aggregate_equal_lag_mean": {
                    "particle": 0.3,
                    "electron_internal_energy": 0.4,
                }
            },
            "used_as_training_loss": False,
        },
        "transport_covariance": {
            "quantities": {
                "particle": {"covariance_decomposition": covariance},
                "electron_internal_energy": {"covariance_decomposition": covariance},
            }
        },
        "spread_error_association": {
            "integrated_transport": {
                "particle": {"pearson": 0.2, "spearman": None},
                "electron_internal_energy": {"pearson": -0.1, "spearman": 0.1},
            }
        },
    }
    return result, localization


def test_localization_metric_projection_is_compact_and_scope_explicit() -> None:
    result, localization = _completed_records()
    metrics = localization_metrics(result, localization)
    assert metrics["marginal/equal_channel_corrected_spread_skill_ratio"] == 0.8
    assert metrics["transport/particle/integrated_spread_skill"] == 0.5
    assert metrics["label/L2_covariance_organization_limited"] is True
    assert "transport/particle/spread_error_spearman" not in metrics
    assert metrics["scope/held_out_85606_read"] is False


def test_completed_localization_validation_is_fail_closed() -> None:
    result, localization = _completed_records()
    validate_completed_localization(
        result=result,
        localization=localization,
        paper0_commit="a" * 40,
        slurm_job_id="123",
    )
    result["model_inference_performed"] = True
    with pytest.raises(RuntimeError, match="forbidden scope"):
        validate_completed_localization(
            result=result,
            localization=localization,
            paper0_commit="a" * 40,
            slurm_job_id="123",
        )


def test_wrapper_source_never_uploads_local_scientific_artifacts() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "paper0/tools/run_b5_covariance_localization_wandb.py"
    ).read_text(encoding="utf-8")
    assert "wandb.log_artifact" not in source
    assert '"forecasts_uploaded": False' in source
    assert '"simulation_fields_uploaded": False' in source
    assert '"raw_accumulators_uploaded": False' in source
    assert '"held_out_85606_read"' in source
