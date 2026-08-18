from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from paper0.tools.evaluate_b2_checkpoint import (
    _write_index,
    audit_full_training_result,
    audit_history,
    frozen_training_run,
    validate_bounded_smoke_result,
)
from tcv_diagnostics.b2_training import B2RunConfig
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.models.latent_diffusion import LatentDiffusionViTConfig


TRAINING_COMMIT = "46e2ca07e15c7114aace18202b26a9756489a3f0"


def _validation(complete: float) -> dict[str, float | int]:
    return {
        "complete": complete,
        "context": complete + 0.1,
        "target": complete + 0.2,
        "examples": 126,
    }


def _training_record(seed: int = 1701) -> dict[str, object]:
    config = B2RunConfig.frozen(mode="full", seed=seed).to_record()
    codec = {"path": "/tmp/codec.pt", "sha256": "b" * 64, "trainable": False}
    config.update(
        {
            "model": LatentDiffusionViTConfig().to_record(),
            "codec_checkpoint": codec,
        }
    )
    return {
        "scope": "B2_LDM_H2_full_training_85604",
        "paper0_commit": TRAINING_COMMIT,
        "completed_epochs": 200,
        "completed_optimizer_steps": 5400,
        "checkpoint_reload_bitwise_exact": True,
        "reload_identity_same_process_same_device": True,
        "cudnn_deterministic_requested": True,
        "tf32_allowed": False,
        "physics_derived_loss_used": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "scientific_result": False,
        "full_B2_training_authorized": True,
        "training_complete_is_scientific_acceptance": False,
        "probabilistic_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "config": config,
        "selected_epoch": 37,
        "selected_validation": _validation(0.1),
        "final_validation": _validation(0.2),
        "sampler_probe": {
            "target_frame_index": 498,
            "ensemble_size": 2,
            "canonical_forecast_shape": [1, 2, 1, 5, 64, 32, 88],
            "finite": True,
            "nonzero_latent_diversity": True,
            "nonzero_decoded_diversity": True,
        },
        "selected_checkpoint": {"path": "/tmp/selected.pt", "sha256": "a" * 64},
        "final_training_state": {"path": "/tmp/final.pt", "sha256": "c" * 64},
        "history": {"path": "/tmp/history.jsonl", "sha256": "d" * 64},
        "latent_normalization": {"path": "/tmp/norm.json", "sha256": "e" * 64},
        "codec_checkpoint": codec,
    }


def _history(path: Path, minimum_epoch: int = 37) -> tuple[dict, dict]:
    values = [1.0 + abs(epoch - minimum_epoch) / 1000.0 for epoch in range(200)]
    records = []
    selected = 0
    for epoch, value in enumerate(values):
        if value < values[selected]:
            selected = epoch
        records.append(
            {
                "epoch": epoch,
                "global_step": 27 * (epoch + 1),
                "learning_rate": 1.0e-4,
                "maximum_preclip_gradient_norm": 1.0,
                "mean_preclip_gradient_norm": 0.5,
                "train_complete_denoising_loss": value + 0.3,
                "train_context_denoising_loss": value + 0.4,
                "train_target_denoising_loss": value + 0.5,
                "train_examples": 430,
                "validation_complete_denoising_loss": value,
                "validation_context_denoising_loss": value + 0.1,
                "validation_target_denoising_loss": value + 0.2,
                "validation_examples": 126,
                "selected_so_far": selected,
            }
        )
    path.write_text("".join(json.dumps(item) + "\n" for item in records))
    return _validation(values[minimum_epoch]), _validation(values[-1])


def test_full_training_audit_accepts_only_the_frozen_complete_contract() -> None:
    record = _training_record()
    audited = audit_full_training_result(
        record,
        seed=1701,
        training_commit=TRAINING_COMMIT,
    )
    assert audited["selected_epoch"] == 37

    forged = copy.deepcopy(record)
    forged["completed_epochs"] = 199
    with pytest.raises(ValueError, match="completed_epochs"):
        audit_full_training_result(
            forged,
            seed=1701,
            training_commit=TRAINING_COMMIT,
        )


def test_full_training_audit_accepts_the_persisted_json_wire_format() -> None:
    record = json.loads(json.dumps(_training_record()))
    assert record["config"]["betas"] == [0.9, 0.99]
    audited = audit_full_training_result(
        record,
        seed=1701,
        training_commit=TRAINING_COMMIT,
    )
    assert audited["selected_epoch"] == 37


def test_history_audit_recomputes_earliest_minimum_and_matches_summaries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.jsonl"
    selected, final = _history(path)
    audited = audit_history(
        path,
        expected_sha256=sha256_path(path),
        selected_epoch=37,
        selected_validation=selected,
        final_validation=final,
    )
    assert audited["earliest_validation_loss_minimum_epoch"] == 37
    assert audited["epochs"] == 200
    assert audited["optimizer_steps"] == 5400


def test_history_audit_rejects_wrong_selection_truncation_and_tampering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.jsonl"
    selected, final = _history(path)
    digest = sha256_path(path)
    with pytest.raises(ValueError, match="earliest validation-loss minimum"):
        audit_history(
            path,
            expected_sha256=digest,
            selected_epoch=38,
            selected_validation=selected,
            final_validation=final,
        )

    selected["complete"] = 999.0
    with pytest.raises(ValueError, match="selected validation"):
        audit_history(
            path,
            expected_sha256=digest,
            selected_epoch=37,
            selected_validation=selected,
            final_validation=final,
        )

    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="exactly 200"):
        audit_history(
            path,
            expected_sha256=sha256_path(path),
            selected_epoch=37,
            selected_validation=_validation(1.0),
            final_validation=final,
        )


def test_artifact_index_reuses_an_already_independently_verified_large_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "forecast.h5"
    artifact.write_bytes(b"large-forecast-placeholder")

    def forbidden_rehash(path: Path) -> str:
        raise AssertionError(f"unexpected redundant hash of {path}")

    monkeypatch.setattr(
        "paper0.tools.evaluate_b2_checkpoint.sha256_path",
        forbidden_rehash,
    )
    index = _write_index(
        tmp_path,
        [artifact],
        verified_sha256={artifact: "f" * 64},
    )
    assert index.read_text() == f"{'f' * 64}  {artifact.resolve()}\n"


def test_full_evaluator_is_bound_to_frozen_matrix_and_passing_bounded_smoke(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training.json"
    training.write_text("{}\n")
    digest = sha256_path(training)
    matrix = {
        "scope": "phase3_B2_LDM_H2_full_training_matrix_frozen",
        "status": "completed_pending_bounded_evaluator_smoke",
        "paper0_commit": "a" * 40,
        "training_commit": TRAINING_COMMIT,
        "development_run": "85604",
        "held_out_85606_read": False,
        "seed_count": 3,
        "seeds": [1701, 1702, 1703],
        "all_training_histories_complete": True,
        "all_checkpoint_choices_frozen_before_probabilistic_metrics": True,
        "bounded_evaluator_smoke_required": True,
        "bounded_evaluator_smoke_completed": False,
        "full_probabilistic_evaluation_allowed": False,
        "O3_launch_allowed": False,
        "runs": [
            {
                "seed": seed,
                "training_result": {
                    "path": str(training),
                    "sha256": digest,
                },
            }
            for seed in (1701, 1702, 1703)
        ],
    }
    selected = frozen_training_run(
        matrix,
        seed=1702,
        training_result=training,
        training_result_sha256=digest,
        training_commit=TRAINING_COMMIT,
        paper0_commit="a" * 40,
    )
    assert selected["seed"] == 1702

    smoke = {
        "scope": "bounded_non_scientific_B2_evaluator_smoke_85604",
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
        "training_matrix": {"sha256": "m" * 64},
    }
    validate_bounded_smoke_result(
        smoke,
        paper0_commit="a" * 40,
        training_matrix_sha256="m" * 64,
    )
    smoke["target_count"] = 126
    with pytest.raises(RuntimeError, match="smoke contract"):
        validate_bounded_smoke_result(
            smoke,
            paper0_commit="a" * 40,
            training_matrix_sha256="m" * 64,
        )
