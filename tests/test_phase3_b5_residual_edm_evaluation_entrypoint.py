"""Fail-closed tests for the frozen B5 scientific evaluator entrypoint."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tcv_diagnostics.b5_residual_edm_full_training import B5EDMFullConfig
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import load_strict_json
from tcv_diagnostics.models.field_residual_edm import FieldResidualUNetConfig


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/evaluate_b5_residual_edm_checkpoint.py"
MANIFEST = ROOT / "paper0/manifests/phase3_b5_full_training_evaluation_85604.json"
PROTOCOL = ROOT / "paper0/protocol/PHASE3_B5_FULL_TRAINING_EVALUATION_PROTOCOL.md"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "evaluate_b5_residual_edm_checkpoint", ENTRYPOINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validation(loss: float, wall: float = 1.0) -> dict:
    return {
        "target_frames": [498, 624],
        "target_count": 126,
        "probes_per_target": 4,
        "probe_count": 504,
        "precision": "float32_no_autocast_TF32_disabled",
        "mean_EDM_loss": loss,
        "mean_unweighted_MSE": loss / 2,
        "minimum_sigma": 0.01,
        "maximum_sigma": 10.0,
        "wall_seconds": wall,
    }


def training_result() -> dict:
    selected = validation(0.2, 30.0)
    final = validation(0.4, 100.0)
    return {
        "scope": "B5_seed1701_full_training_and_data_only_selection_85604",
        "status": "training_completed_checkpoint_selected",
        "paper0_commit": "a" * 40,
        "development_run": "85604",
        "sequestered_run": "85606",
        "config": B5EDMFullConfig().to_record(),
        "model_config": FieldResidualUNetConfig().to_record(),
        "completed_epochs": 100,
        "target_presentations": 43_000,
        "completed_optimizer_steps": 10_800,
        "EMA_updates": 10_800,
        "candidate_count": 20,
        "candidate_completed_epochs": list(range(5, 101, 5)),
        "selected_completed_epoch": 30,
        "selected_optimizer_step": 3_240,
        "selected_validation": selected,
        "final_candidate_validation": final,
        "checkpoint_reload_bitwise_exact": True,
        "all_losses_and_gradients_finite": True,
        "parameter_count": 11_604_709,
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "sampled_forecast_metric_used_for_checkpoint_selection": False,
        "target_truth_used_as_condition": False,
        "absolute_time_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "scientific_acceptance_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "artifacts": {
            name: {"path": f"/{name}", "sha256": "b" * 64}
            for name in (
                "config",
                "training_order",
                "validation_seed_bank",
                "history",
                "selected_checkpoint",
                "selected_source_candidate",
                "final_training_state",
            )
        }
        | {"candidate_checkpoints": [{} for _ in range(20)]},
    }


def test_B5_evaluator_accepts_exact_frozen_manifest_and_training_identity() -> None:
    module = load_entrypoint()
    module.validate_manifest(
        load_strict_json(MANIFEST),
        manifest_path=MANIFEST,
        protocol_path=PROTOCOL,
    )
    audit = module.audit_full_training_result(
        training_result(), training_commit="a" * 40
    )
    assert audit["selected_completed_epoch"] == 30
    assert audit["parameter_count"] == 11_604_709

    contaminated = training_result()
    contaminated["physics_metric_used_for_checkpoint_selection"] = True
    with pytest.raises(ValueError, match="physics_metric"):
        module.audit_full_training_result(contaminated, training_commit="a" * 40)


def test_B5_evaluator_reaudits_complete_history_and_earliest_minimum(
    tmp_path: Path,
) -> None:
    module = load_entrypoint()
    records = []
    for epoch in range(1, 101):
        is_candidate = epoch % 5 == 0
        loss = 0.2 if epoch == 30 else 0.4 if epoch == 100 else 1.0 + epoch / 100
        candidate_validation = validation(loss, float(epoch)) if is_candidate else None
        candidate = (
            {
                "completed_epoch": epoch,
                "global_optimizer_step": epoch * 108,
                "validation": candidate_validation,
                "path": f"/candidate-{epoch}",
                "sha256": "c" * 64,
            }
            if is_candidate
            else None
        )
        records.append(
            {
                "completed_epoch": epoch,
                "global_optimizer_step": epoch * 108,
                "EMA_updates": epoch * 108,
                "train_target_count": 430,
                "train_mean_EDM_loss": 2.0,
                "train_mean_unweighted_MSE": 1.0,
                "mean_preclip_gradient_norm": 0.5,
                "maximum_preclip_gradient_norm": 1.0,
                "first_learning_rate": 1.0e-4,
                "last_learning_rate": 1.0e-5,
                "epoch_wall_seconds": 10.0,
                "validation_candidate": is_candidate,
                "validation": candidate_validation,
                "candidate": candidate,
            }
        )
    path = tmp_path / "history.jsonl"
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    audit = module.audit_history(
        path,
        expected_sha256=sha256_path(path),
        selected_completed_epoch=30,
        selected_validation=records[29]["validation"],
        final_validation=records[99]["validation"],
    )
    assert audit["candidate_count"] == 20
    assert audit["earliest_validation_minimum_completed_epoch"] == 30

    records[49]["global_optimizer_step"] += 1
    path2 = tmp_path / "corrupt.jsonl"
    path2.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="epoch/update"):
        module.audit_history(
            path2,
            expected_sha256=sha256_path(path2),
            selected_completed_epoch=30,
            selected_validation=records[29]["validation"],
            final_validation=records[99]["validation"],
        )


def test_B5_deterministic_mean_view_is_an_exact_contiguous_parent_slice() -> None:
    module = load_entrypoint()

    class Parent:
        target_frames = tuple(range(498, 624))

        def read(self, start: int, stop: int) -> np.ndarray:
            return np.arange(start, stop, dtype=np.float32)[:, None]

    view = module.DeterministicMeanView(Parent(), tuple(range(500, 504)))
    assert view.target_frames == tuple(range(500, 504))
    np.testing.assert_array_equal(view.read(1, 3)[:, 0], [3.0, 4.0])
    with pytest.raises(ValueError, match="validation"):
        module.DeterministicMeanView(Parent(), tuple(range(620, 625)))


def test_B5_full_evaluation_requires_exact_bounded_smoke() -> None:
    module = load_entrypoint()
    smoke = {
        "scope": "bounded_non_scientific_B5_residual_EDM_evaluator_smoke_85604",
        "status": "bounded_evaluator_smoke_completed",
        "paper0_commit": "a" * 40,
        "seed": 1701,
        "target_frames": [498, 502],
        "target_count": 4,
        "ensemble_members": 32,
        "held_out_85606_read": False,
        "truth_opened_only_after_forecast_hash": True,
        "full_evaluation_preconditions_passed": True,
        "scientific_acceptance_evaluated": False,
        "O3_launch_allowed": False,
        "training_result": {"sha256": "d" * 64},
    }
    module.validate_bounded_smoke_result(
        smoke,
        paper0_commit="a" * 40,
        training_result_sha256="d" * 64,
    )
    contaminated = deepcopy(smoke)
    contaminated["truth_opened_only_after_forecast_hash"] = False
    with pytest.raises(RuntimeError, match="smoke contract"):
        module.validate_bounded_smoke_result(
            contaminated,
            paper0_commit="a" * 40,
            training_result_sha256="d" * 64,
        )


def test_B5_evaluator_source_enforces_truth_separation_and_no_time_condition() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'choices=("smoke", "full")' in source
    assert "OneStepContextDataset" in source
    assert "OneStepWindowDataset" not in source
    assert "O2ForecastArtifact" in source
    assert "generate_selected_b5_forecasts" in source
    assert "load_selected_b5_model" in source
    assert "scientific_sampler_seed_bank" in source
    assert '"absolute_time_input": False' in source
    assert '"posthoc_calibration": False' in source
    assert '"member_prefixes_regenerated": False' in source
    assert source.index(
        "write_strict_json_atomic(generation_path, generation)"
    ) < source.index("native_truth = NativeTruthCatalog")
    assert source.index("native_truth = NativeTruthCatalog") < source.index(
        "score = scorer("
    )
    assert "85606_access" in source
