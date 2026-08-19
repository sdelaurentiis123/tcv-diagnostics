#!/usr/bin/env python3
"""Generate and truth-separately score frozen B4-PDE-Refiner-H1 seed 1701."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b2_field_metrics import (  # noqa: E402
    B2_FIELDS,
    B2_PRIMARY_REGIONS,
)
from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.matched_o1_transport import (  # noqa: E402
    NativeTruthCatalog,
    load_transport_geometry,
)
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.model_training_data import load_official_catalog  # noqa: E402
from tcv_diagnostics.models.o2 import O2ViTConfig  # noqa: E402
from tcv_diagnostics.models.pde_refiner import PDERefinerConfig  # noqa: E402
from tcv_diagnostics.o2_context_data import OneStepContextDataset  # noqa: E402
from tcv_diagnostics.pde_refiner_forecast import (  # noqa: E402
    B4_SCIENTIFIC_SEED_NPY_SHA256,
    PDERefinerFinalForecastArtifact,
    PDERefinerStageForecastArtifact,
    generate_selected_pde_refiner_forecasts,
    load_selected_pde_refiner_model,
    save_scientific_refiner_seed_bank,
    scientific_refiner_seed_bank,
)
from tcv_diagnostics.pde_refiner_full_training import (  # noqa: E402
    B4_FULL_LEVEL_COUNTS,
    B4_FULL_LEVEL_RAW_SHA256,
    B4_VALIDATION_BANK_NPY_SHA256,
    PDERefinerFullConfig,
    full_learning_rate,
    full_training_levels,
)
from tcv_diagnostics.pde_refiner_scoring import (  # noqa: E402
    score_pde_refiner_final,
    score_pde_refiner_final_smoke,
    score_pde_refiner_stages,
    score_pde_refiner_stages_smoke,
    verify_locked_b4_metric_sources,
)
from tcv_diagnostics.pde_refiner_training import (  # noqa: E402
    RefinerParentArtifacts,
    validation_seed_bank,
)


EXPECTED_MANIFEST_SHA256 = (
    "e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229"
)
EXPECTED_PROTOCOL_SHA256 = (
    "ffa56b2111074253a70c7453f1e36f91ca747ec59a68d632288764d60387aad1"
)
B4_PASSING_EVALUATOR_SMOKE_COMMIT = (
    "029f6d9d425fd9bbac11aebf82466588a97ac658"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", type=int, choices=(1701,), required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-result-sha256", required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--event-threshold-result", type=Path, required=True)
    parser.add_argument("--event-threshold-result-sha256", required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--evaluation-protocol-sha256", required=True)
    parser.add_argument("--smoke-result", type=Path)
    parser.add_argument("--smoke-result-sha256")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--member-batch-size", type=int, default=8)
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != str(expected_commit):
        raise RuntimeError(f"Paper 0 commit {actual} differs from {expected_commit}")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"Paper 0 checkout is dirty:\n{dirty}")


def verify_input(path: Path, expected_sha256: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    actual = sha256_path(resolved)
    if actual != str(expected_sha256):
        raise ValueError(f"SHA-256 mismatch for {resolved}: {actual}")
    return resolved


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"B4 {name} is non-finite")
    return result


def _strict_json_line(line: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token {value}")

    value = json.loads(line, parse_constant=reject_constant)
    if not isinstance(value, Mapping):
        raise ValueError("B4 history line is not a JSON object")
    return value


def _json_equal(first: Any, second: Any) -> bool:
    return json.loads(json.dumps(first)) == json.loads(json.dumps(second))


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    protocol_path: Path,
) -> RefinerParentArtifacts:
    if (
        sha256_path(manifest_path) != EXPECTED_MANIFEST_SHA256
        or sha256_path(protocol_path) != EXPECTED_PROTOCOL_SHA256
        or manifest.get("protocol_status")
        != (
            "frozen_after_passing_B4_smoke_before_full_training_checkpoint_"
            "selection_or_scientific_evaluation_implementation"
        )
        or manifest.get("development_run") != "85604"
        or manifest.get("sequestered_run") != "85606"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("full_training_authorized") is not True
        or manifest.get("scientific_one_step_evaluation_authorized") is not True
        or manifest.get("protocol", {}).get("sha256")
        != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("B4 full evaluation manifest contract differs")
    authorized = manifest.get("authorized_scope", [])
    for scope in (
        "B4_PDE_Refiner_H1_four_target_evaluator_smoke_85604",
        "B4_PDE_Refiner_H1_seed1701_one_step_M32_final_generation_85604_validation",
        "B4_PDE_Refiner_H1_seed1701_M4_all_stage_generation_85604_validation",
        "B4_PDE_Refiner_H1_H_det_and_H_prob_evaluation_85604_validation",
    ):
        if scope not in authorized:
            raise RuntimeError(f"B4 authorized scope omits {scope}")
    data = manifest.get("data", {})
    ensemble = manifest.get("scientific_ensemble", {})
    model = manifest.get("model", {})
    if (
        data.get("fields") != ["Ne", "Pe", "Pi", "phi", "Vi"]
        or data.get("context_frames") != 1
        or data.get("future_frames") != 1
        or data.get("guard_frames") != [432, 496]
        or data.get("validation_targets") != [498, 624]
        or data.get("zperiod") != 5
        or data.get("mode_mapping") != "n=5k"
        or data.get("absolute_time_input_allowed") is not False
        or data.get("guard_frames_read_allowed") is not False
        or model.get("arm") != "B4-PDE-Refiner-H1"
        or model.get("refinement_levels") != [0, 1, 2, 3]
        or model.get("refinement_steps") != 3
        or ensemble.get("seed_bank_shape") != [126, 32, 3]
        or ensemble.get("seed_bank_npy_sha256")
        != B4_SCIENTIFIC_SEED_NPY_SHA256
        or ensemble.get("final_forecast_shape")
        != [126, 32, 1, 5, 64, 32, 88]
        or ensemble.get("stage_forecast_shape")
        != [126, 4, 4, 5, 64, 32, 88]
        or ensemble.get("M4_stage3_bitwise_prefix_of_M32_required") is not True
        or ensemble.get("level0_bitwise_shared_across_members_required") is not True
        or ensemble.get("context_only_generation_required") is not True
        or ensemble.get("truth_opened_after_both_forecast_hashes_only") is not True
        or ensemble.get("regeneration_or_posthoc_calibration_allowed") is not False
    ):
        raise RuntimeError("B4 frozen data/model/scientific-ensemble contract differs")
    if manifest.get("metric_engine", {}).get(
        "source_sha256"
    ) != verify_locked_b4_metric_sources():
        raise RuntimeError("B4 manifest/source metric locks differ")
    codec = manifest.get("codec", {})
    parent = manifest.get("deterministic_parent", {})
    artifacts = RefinerParentArtifacts(
        checkpoint_path=Path(str(parent.get("checkpoint_path", ""))),
        checkpoint_sha256=str(parent.get("checkpoint_sha256", "")),
        codec_path=Path(str(codec.get("checkpoint_path", ""))),
        codec_sha256=str(codec.get("checkpoint_sha256", "")),
        latent_normalization_path=Path(
            str(codec.get("latent_normalization_path", ""))
        ),
        latent_normalization_sha256=str(
            codec.get("latent_normalization_sha256", "")
        ),
    )
    for path, digest in (
        (artifacts.checkpoint_path, artifacts.checkpoint_sha256),
        (artifacts.codec_path, artifacts.codec_sha256),
        (
            artifacts.latent_normalization_path,
            artifacts.latent_normalization_sha256,
        ),
    ):
        verify_input(path, digest)
    return artifacts


def _validate_validation_record(values: Mapping[str, Any], name: str) -> None:
    if (
        int(values.get("target_count", -1)) != 126
        or int(values.get("ensemble_members", -1)) != 2
        or values.get("refinement_levels") != [0, 1, 2, 3]
        or values.get("checkpoint_weights") != "EMA"
        or values.get("physics_metrics_used") is not False
    ):
        raise ValueError(f"B4 {name} validation identity differs")
    _finite(
        values.get("ensemble_mean_equal_channel_decoded_standardized_field_MAE"),
        f"{name}.selection_MAE",
    )
    levels = values.get("equal_channel_MAE_by_level", [])
    channels = values.get("MAE_by_level_and_channel", [])
    if len(levels) != 4 or len(channels) != 4:
        raise ValueError(f"B4 {name} validation level count differs")
    for level, by_channel in zip(levels, channels, strict=True):
        _finite(level, f"{name}.level_MAE")
        if set(by_channel) != set(B2_FIELDS):
            raise ValueError(f"B4 {name} validation fields differ")
        for field, value in by_channel.items():
            _finite(value, f"{name}.{field}")
    final = values.get("final_MAE_by_channel", {})
    if set(final) != set(B2_FIELDS):
        raise ValueError(f"B4 {name} final validation fields differ")
    for field, value in final.items():
        _finite(value, f"{name}.final.{field}")


def audit_full_training_result(
    record: Mapping[str, Any],
    *,
    training_commit: str,
    artifacts: RefinerParentArtifacts,
) -> dict[str, Any]:
    config = PDERefinerFullConfig.frozen(seed=1701)
    expected = {
        "scope": "B4_PDE_Refiner_H1_seed1701_full_training_85604",
        "paper0_commit": str(training_commit),
        "completed_epochs": 100,
        "completed_optimizer_steps": 2700,
        "EMA_updates": 2700,
        "validation_candidates_evaluated": 20,
        "validation_completed_epochs": list(config.validation_completed_epochs),
        "checkpoint_reload_bitwise_exact": True,
        "parent_parameter_gradient_seen": True,
        "refinement_parameter_gradient_seen": True,
        "training_level_counts": list(B4_FULL_LEVEL_COUNTS),
        "all_four_training_levels_exercised": True,
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
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise ValueError(f"B4 training result field {name!r} differs")

    observed_config = json.loads(json.dumps(record.get("config", {})))
    frozen_config = json.loads(json.dumps(config.to_record()))
    runtime_keys = {
        "model",
        "pde_refiner",
        "parameter_groups",
        "deterministic_parent",
        "codec_checkpoint",
        "latent_normalization",
        "training_levels",
        "validation_seed_bank",
    }
    if set(observed_config) != set(frozen_config) | runtime_keys:
        raise ValueError("B4 training expanded configuration keys differ")
    for name, value in frozen_config.items():
        if observed_config.get(name) != value:
            raise ValueError(f"B4 training frozen configuration field {name!r} differs")
    if observed_config["model"] != O2ViTConfig().to_record():
        raise ValueError("B4 training model configuration differs")
    if observed_config["pde_refiner"] != PDERefinerConfig().to_record():
        raise ValueError("B4 training refiner configuration differs")

    groups = record.get("parameter_groups", {})
    if (
        observed_config["parameter_groups"] != groups
        or int(groups.get("parent_parameter_count", -1)) != 51_612_800
        or int(groups.get("refinement_parameter_count", -1)) != 9_606_144
        or int(groups.get("total_parameter_count", -1)) != 61_218_944
        or int(record.get("parameter_count", -1)) != 61_218_944
    ):
        raise ValueError("B4 training parameter accounting differs")
    config_parent = observed_config["deterministic_parent"]
    if (
        Path(str(config_parent.get("path", ""))) != artifacts.checkpoint_path
        or config_parent.get("sha256") != artifacts.checkpoint_sha256
        or config_parent.get("load_audit")
        != record.get("deterministic_parent_load_audit")
        or config_parent.get("preoptimization_identity")
        != record.get("preoptimization_parent_identity")
    ):
        raise ValueError("B4 training expanded parent configuration differs")
    if observed_config["codec_checkpoint"] != {
        "path": str(artifacts.codec_path),
        "sha256": artifacts.codec_sha256,
        "trainable": False,
    }:
        raise ValueError("B4 training expanded codec configuration differs")
    if observed_config["latent_normalization"] != {
        "path": str(artifacts.latent_normalization_path),
        "sha256": artifacts.latent_normalization_sha256,
        "refit": False,
    }:
        raise ValueError("B4 training expanded normalization configuration differs")
    config_levels = observed_config["training_levels"]
    result_levels = record.get("training_levels", {})
    if (
        config_levels.get("seed") != 41_001
        or config_levels.get("shape") != [100, 430]
        or config_levels.get("counts") != list(B4_FULL_LEVEL_COUNTS)
        or config_levels.get("raw_C_order_sha256") != B4_FULL_LEVEL_RAW_SHA256
        or config_levels.get("npy_sha256") != result_levels.get("sha256")
    ):
        raise ValueError("B4 training level-matrix provenance differs")
    config_bank = observed_config["validation_seed_bank"]
    result_bank = record.get("validation_seed_bank", {})
    if (
        config_bank.get("seed") != 41_003
        or config_bank.get("shape") != [126, 2, 3]
        or config_bank.get("dtype") != "uint64"
        or config_bank.get("npy_sha256") != B4_VALIDATION_BANK_NPY_SHA256
        or result_bank.get("sha256") != B4_VALIDATION_BANK_NPY_SHA256
    ):
        raise ValueError("B4 checkpoint-selection seed-bank provenance differs")

    selected_epoch = int(record.get("selected_epoch", -1))
    if selected_epoch + 1 not in config.validation_completed_epochs:
        raise ValueError("B4 selected epoch differs")
    if (
        int(record.get("selected_completed_epoch", -1)) != selected_epoch + 1
        or int(record.get("selected_optimizer_step", -1))
        != (selected_epoch + 1) * config.optimizer_steps_per_epoch
    ):
        raise ValueError("B4 selected epoch/step accounting differs")
    _validate_validation_record(record.get("selected_validation", {}), "selected")
    _validate_validation_record(record.get("final_validation", {}), "final")
    if record.get("preoptimization_parent_identity", {}).get("bitwise_exact") is not True:
        raise ValueError("B4 deterministic-parent identity differs")
    if record.get("deterministic_parent_load_audit", {}).get("passed") is not True:
        raise ValueError("B4 deterministic-parent load audit differs")
    reload_record = record.get("checkpoint_reload", {})
    if (
        reload_record.get("latent_bitwise_exact") is not True
        or reload_record.get("forecast_bitwise_exact") is not True
    ):
        raise ValueError("B4 selected-checkpoint reload record differs")
    if record.get("codec_state_sha256_before") != record.get(
        "codec_state_sha256_after"
    ):
        raise ValueError("B4 codec state hashes differ")
    for observed, path, digest in (
        (record.get("deterministic_parent", {}), artifacts.checkpoint_path, artifacts.checkpoint_sha256),
        (record.get("codec_checkpoint", {}), artifacts.codec_path, artifacts.codec_sha256),
    ):
        if Path(str(observed.get("path", ""))) != path or observed.get(
            "sha256"
        ) != digest:
            raise ValueError("B4 parent/codec training provenance differs")
    latent = record.get("latent_normalization", {})
    if (
        latent.get("sha256") != artifacts.latent_normalization_sha256
        or latent.get("refit") is not False
    ):
        raise ValueError("B4 latent-normalization training provenance differs")
    for name in (
        "selected_checkpoint",
        "final_training_state",
        "history",
        "validation_seed_bank",
        "training_levels",
    ):
        item = record.get(name, {})
        if not item.get("path") or len(str(item.get("sha256", ""))) != 64:
            raise ValueError(f"B4 training artifact {name!r} is malformed")
    return {
        "config": config,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": dict(record["selected_checkpoint"]),
        "history": dict(record["history"]),
        "selection_seed_bank": dict(record["validation_seed_bank"]),
        "training_levels": dict(record["training_levels"]),
        "parameter_count": int(record["parameter_count"]),
    }


def audit_history(
    path: Path,
    *,
    expected_sha256: str,
    selected_epoch: int,
    selected_validation: Mapping[str, Any],
    final_validation: Mapping[str, Any],
) -> dict[str, Any]:
    history_path = verify_input(path, expected_sha256)
    lines = history_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 100:
        raise ValueError("B4 full history does not contain exactly 100 epochs")
    records = [_strict_json_line(line) for line in lines]
    config = PDERefinerFullConfig.frozen(seed=1701)
    level_matrix = full_training_levels(config)
    candidate_losses: list[tuple[int, float]] = []
    running_selected: int | None = None
    running_loss = math.inf
    for epoch, record in enumerate(records):
        expected_step = config.optimizer_steps_per_epoch * (epoch + 1)
        if (
            int(record.get("epoch", -1)) != epoch
            or int(record.get("completed_epoch", -1)) != epoch + 1
            or int(record.get("global_step", -1)) != expected_step
            or int(record.get("EMA_updates", -1)) != expected_step
            or int(record.get("examples", -1)) != 430
            or float(record.get("EMA_decay", -1.0)) != config.ema_decay
        ):
            raise ValueError("B4 history epoch/example/step contract differs")
        expected_lr = full_learning_rate(config, expected_step - 1)
        if not math.isclose(
            _finite(record.get("learning_rate"), f"history[{epoch}].learning_rate"),
            expected_lr,
            rel_tol=0.0,
            abs_tol=1.0e-18,
        ):
            raise ValueError("B4 history learning-rate schedule differs")
        for name in (
            "train_standardized_latent_MSE",
            "mean_preclip_total_gradient_norm",
            "maximum_preclip_total_gradient_norm",
            "mean_preclip_parent_gradient_norm",
            "mean_preclip_refinement_gradient_norm",
            "epoch_wall_seconds",
        ):
            _finite(record.get(name), f"history[{epoch}].{name}")
        by_level = record.get("train_MSE_by_level", {})
        counts = record.get("train_count_by_level", {})
        if set(by_level) != {"0", "1", "2", "3"} or set(counts) != {
            "0",
            "1",
            "2",
            "3",
        }:
            raise ValueError("B4 history per-level keys differ")
        expected_counts = np.bincount(level_matrix[epoch], minlength=4)
        for level in range(4):
            _finite(by_level[str(level)], f"history[{epoch}].level{level}")
            if int(counts[str(level)]) != int(expected_counts[level]):
                raise ValueError("B4 history per-level count differs")
        candidate = epoch + 1 in config.validation_completed_epochs
        if bool(record.get("validation_performed")) != candidate:
            raise ValueError("B4 history validation schedule differs")
        validation = record.get("validation")
        if candidate:
            if not isinstance(validation, Mapping):
                raise ValueError("B4 validation candidate record is absent")
            _validate_validation_record(validation, f"history[{epoch}]")
            loss = _finite(
                validation[
                    "ensemble_mean_equal_channel_decoded_standardized_field_MAE"
                ],
                f"history[{epoch}].selection_MAE",
            )
            candidate_losses.append((epoch, loss))
            if loss < running_loss:
                running_loss = loss
                running_selected = epoch
        elif validation is not None:
            raise ValueError("B4 non-candidate history contains validation")
        if record.get("selected_so_far") != running_selected:
            raise ValueError("B4 history running checkpoint selection differs")
    if len(candidate_losses) != 20:
        raise ValueError("B4 history validation-candidate count differs")
    earliest = min(candidate_losses, key=lambda item: item[1])[0]
    if int(selected_epoch) != earliest:
        raise ValueError("B4 checkpoint is not earliest fixed-seed validation minimum")
    if not _json_equal(selected_validation, records[earliest]["validation"]):
        raise ValueError("B4 selected validation does not match history")
    if not _json_equal(final_validation, records[-1]["validation"]):
        raise ValueError("B4 final validation does not match history")
    return {
        "epochs": 100,
        "optimizer_steps": 2700,
        "validation_candidates": 20,
        "selection_metric": (
            "fixed_seed_M2_all126_level3_ensemble_mean_equal_channel_decoded_"
            "standardized_field_MAE"
        ),
        "earliest_validation_minimum_epoch": earliest,
        "minimum_validation_MAE": min(value for _, value in candidate_losses),
        "finite": True,
    }


def validate_bounded_smoke_result(
    record: Mapping[str, Any],
    *,
    paper0_commit: str,
    training_result_sha256: str,
) -> None:
    if (
        record.get("scope")
        != "bounded_non_scientific_B4_PDE_Refiner_H1_evaluator_smoke_85604"
        or record.get("status") != "bounded_evaluator_smoke_completed"
        or record.get("paper0_commit") != paper0_commit
        or record.get("seed") != 1701
        or record.get("target_frames") != [498, 502]
        or record.get("target_count") != 4
        or record.get("final_ensemble_members") != 32
        or record.get("stage_prefix_members") != 4
        or record.get("held_out_85606_read") is not False
        or record.get("truth_opened_only_after_both_forecast_hashes") is not True
        or record.get("full_evaluation_preconditions_passed") is not True
        or record.get("H_det_evaluated") is not False
        or record.get("H_prob_evaluated") is not False
        or record.get("O3_launch_allowed") is not False
        or record.get("training_result", {}).get("sha256")
        != training_result_sha256
    ):
        raise RuntimeError("B4 bounded evaluator smoke contract differs")


def _write_index(
    output: Path,
    artifacts: list[Path],
    *,
    verified_sha256: Mapping[Path, str] | None = None,
) -> Path:
    index = output / "artifact_sha256.txt"
    if index.exists():
        raise FileExistsError(index)
    known = {} if verified_sha256 is None else dict(verified_sha256)
    lines = []
    for path in artifacts:
        digest = known[path] if path in known else sha256_path(path)
        lines.append(f"{digest}  {path.resolve(strict=True)}")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def _artifact_integrity(
    *,
    final_artifact: PDERefinerFinalForecastArtifact,
    stage_artifact: PDERefinerStageForecastArtifact,
) -> dict[str, Any]:
    if final_artifact.target_frames != stage_artifact.target_frames:
        raise ValueError("B4 final/stage target intervals differ")
    nonzero = np.zeros(5, dtype=bool)
    for position in range(len(final_artifact.target_frames)):
        final = final_artifact.read(position, position + 1)[0]
        stages = stage_artifact.read(position, position + 1)[0]
        if not np.array_equal(final[:4, 0], stages[:, 3]):
            raise RuntimeError("B4 stored M4 stage-three/M32 prefix differs")
        if any(
            not np.array_equal(stages[0, 0], stages[member, 0])
            for member in range(1, 4)
        ):
            raise RuntimeError("B4 stored level zero is not member-shared")
        spread = np.var(final[:, 0], axis=0, ddof=1)
        nonzero |= np.any(spread > 0.0, axis=(1, 2, 3))
    if not np.all(nonzero):
        raise RuntimeError("B4 final M32 has collapsed global field spread")
    return {
        "target_count": len(final_artifact.target_frames),
        "M4_stage3_bitwise_prefix_of_M32": True,
        "level0_bitwise_shared_across_members": True,
        "nonzero_global_spread_by_field": {
            field: bool(nonzero[channel])
            for channel, field in enumerate(B2_FIELDS)
        },
    }


def _validate_primary_region_spread(score: Mapping[str, Any]) -> dict[str, bool]:
    regions = score.get("field_and_marginal_calibration", {}).get("regions", {})
    result = {}
    for region in B2_PRIMARY_REGIONS:
        passed = regions.get(region, {}).get("aggregate", {}).get(
            "all_fields_nonzero_spread"
        )
        if passed is not True:
            raise RuntimeError(f"B4 final spread collapsed in primary region {region}")
        result[region] = True
    return result


def main() -> None:
    args = parse_args()
    paths = (
        args.training_result,
        args.artifact_root,
        args.native_truth_result,
        args.geometry_manifest,
        args.geometry,
        args.event_threshold_result,
        args.evaluation_manifest,
        args.evaluation_protocol,
        args.output_directory,
    )
    for path in paths:
        assert_development_path(path)
    if args.smoke_result is not None:
        assert_development_path(args.smoke_result)
    verify_checkout(args.paper0_commit)
    training_path = verify_input(args.training_result, args.training_result_sha256)
    native_path = verify_input(
        args.native_truth_result, args.native_truth_result_sha256
    )
    geometry_manifest_path = verify_input(
        args.geometry_manifest, args.geometry_manifest_sha256
    )
    geometry_path = verify_input(args.geometry, args.geometry_sha256)
    threshold_path = verify_input(
        args.event_threshold_result, args.event_threshold_result_sha256
    )
    manifest_path = verify_input(
        args.evaluation_manifest, args.evaluation_manifest_sha256
    )
    protocol_path = verify_input(
        args.evaluation_protocol, args.evaluation_protocol_sha256
    )
    if (
        args.evaluation_manifest_sha256 != EXPECTED_MANIFEST_SHA256
        or args.evaluation_protocol_sha256 != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("B4 evaluator manifest/protocol hash arguments differ")
    if args.mode == "smoke":
        if args.smoke_result is not None or args.smoke_result_sha256 is not None:
            raise RuntimeError("bounded B4 smoke cannot consume a smoke result")
        smoke_path = None
    else:
        if args.smoke_result is None or args.smoke_result_sha256 is None:
            raise RuntimeError("full B4 evaluation requires the bounded smoke result")
        smoke_path = verify_input(args.smoke_result, args.smoke_result_sha256)

    manifest = load_strict_json(manifest_path)
    parent_artifacts = validate_manifest(
        manifest,
        manifest_path=manifest_path,
        protocol_path=protocol_path,
    )
    training_record = load_strict_json(training_path)
    training = audit_full_training_result(
        training_record,
        training_commit=args.training_commit,
        artifacts=parent_artifacts,
    )
    checkpoint_path = verify_input(
        Path(training["selected_checkpoint"]["path"]),
        training["selected_checkpoint"]["sha256"],
    )
    selection_bank_path = verify_input(
        Path(training["selection_seed_bank"]["path"]),
        training["selection_seed_bank"]["sha256"],
    )
    selected_bank = np.load(selection_bank_path, allow_pickle=False)
    if not np.array_equal(selected_bank, validation_seed_bank()):
        raise RuntimeError("B4 checkpoint-selection seed-bank bytes differ")
    del selected_bank
    training_levels_path = verify_input(
        Path(training["training_levels"]["path"]),
        training["training_levels"]["sha256"],
    )
    observed_levels = np.load(training_levels_path, allow_pickle=False)
    if not np.array_equal(
        observed_levels, full_training_levels(training["config"])
    ):
        raise RuntimeError("B4 full training-level matrix bytes differ")
    del observed_levels
    final_training_state = verify_input(
        Path(training_record["final_training_state"]["path"]),
        training_record["final_training_state"]["sha256"],
    )
    history_audit = audit_history(
        Path(training["history"]["path"]),
        expected_sha256=training["history"]["sha256"],
        selected_epoch=training["selected_epoch"],
        selected_validation=training_record["selected_validation"],
        final_validation=training_record["final_validation"],
    )
    if smoke_path is not None:
        validate_bounded_smoke_result(
            load_strict_json(smoke_path),
            paper0_commit=B4_PASSING_EVALUATOR_SMOKE_COMMIT,
            training_result_sha256=args.training_result_sha256,
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("B4 evaluation requires exactly one allocated CUDA device")
    device = torch.device("cuda", 0)
    accelerator = torch.cuda.get_device_name(device)
    if "H100" not in accelerator and "H200" not in accelerator:
        raise RuntimeError(f"B4 evaluation requires H100/H200, found {accelerator!r}")

    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B4 evaluation {output}")
    output.mkdir(parents=True)
    seed_bank = scientific_refiner_seed_bank()
    seed_bank_path = output / "scientific_seed_bank_M32x3.npy"
    seed_bank_sha256 = save_scientific_refiner_seed_bank(
        seed_bank_path, seed_bank
    )
    if seed_bank_sha256 == training["selection_seed_bank"]["sha256"]:
        raise RuntimeError("scientific seed bank equals checkpoint-selection bank")
    catalog = load_official_catalog(args.artifact_root)
    bounded_smoke = args.mode == "smoke"
    targets = tuple(range(498, 502)) if bounded_smoke else tuple(range(498, 624))
    model = load_selected_pde_refiner_model(
        checkpoint=checkpoint_path,
        expected_checkpoint_sha256=training["selected_checkpoint"]["sha256"],
        artifacts=parent_artifacts,
        device=device,
        training_commit=args.training_commit,
        expected_selected_epoch=training["selected_epoch"],
    )
    context = OneStepContextDataset(
        catalog,
        target_frames=targets,
        context_frames=1,
        return_physical=False,
    )
    final_forecast_path = output / "forecast_final_M32.h5"
    stage_forecast_path = output / "forecast_stages_M4.h5"
    metadata = {
        "source_kind": "selected_B4_PDE_Refiner",
        "arm": "B4-PDE-Refiner-H1",
        "seed": 1701,
        "context_frames": 1,
        "checkpoint_sha256": training["selected_checkpoint"]["sha256"],
        "codec_checkpoint_sha256": parent_artifacts.codec_sha256,
        "training_commit": args.training_commit,
        "evaluation_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "evaluation_mode": args.mode,
        "bounded_non_scientific_smoke": bounded_smoke,
        "target_truth_read": False,
        "absolute_time_input": False,
        "member_prefixes_regenerated": False,
        "posthoc_calibration": False,
        "evaluation_manifest": {
            "path": str(manifest_path),
            "sha256": args.evaluation_manifest_sha256,
        },
        "evaluation_protocol": {
            "path": str(protocol_path),
            "sha256": args.evaluation_protocol_sha256,
        },
    }
    try:
        generation = generate_selected_pde_refiner_forecasts(
            model=model,
            dataset=context,
            target_frames=targets,
            seed_bank=seed_bank,
            seed_bank_path=seed_bank_path,
            seed_bank_sha256=seed_bank_sha256,
            final_output=final_forecast_path,
            stage_output=stage_forecast_path,
            metadata=metadata,
            device=device,
            member_batch_size=args.member_batch_size,
            bounded_smoke=bounded_smoke,
        )
    finally:
        context.close()
    generation_path = output / "generation.json"
    write_strict_json_atomic(generation_path, generation)
    del model, seed_bank
    torch.cuda.empty_cache()

    # Both forecasts are first closed, hashed, reopened, and cross-checked here.
    # Validation target truth is not constructed or opened until after this block.
    with PDERefinerFinalForecastArtifact(
        final_forecast_path,
        expected_sha256=generation["final_forecast"]["sha256"],
        target_frames=targets,
        seed_bank_path=seed_bank_path,
        seed_bank_sha256=seed_bank_sha256,
    ) as final_artifact:
        with PDERefinerStageForecastArtifact(
            stage_forecast_path,
            expected_sha256=generation["stage_forecast"]["sha256"],
            target_frames=targets,
            seed_bank_path=seed_bank_path,
            seed_bank_sha256=seed_bank_sha256,
        ) as stage_artifact:
            artifact_integrity = _artifact_integrity(
                final_artifact=final_artifact,
                stage_artifact=stage_artifact,
            )

    # Validation truth is first opened below, after both forecast hashes close.
    native_truth = NativeTruthCatalog(load_strict_json(native_path))
    geometry = load_transport_geometry(
        geometry_path=geometry_path,
        geometry_manifest=load_strict_json(geometry_manifest_path),
    )
    threshold_record = load_strict_json(threshold_path)
    with PDERefinerFinalForecastArtifact(
        final_forecast_path,
        expected_sha256=generation["final_forecast"]["sha256"],
        target_frames=targets,
        seed_bank_path=seed_bank_path,
        seed_bank_sha256=seed_bank_sha256,
    ) as final_artifact:
        final_scorer = (
            score_pde_refiner_final_smoke
            if bounded_smoke
            else score_pde_refiner_final
        )
        final_score = final_scorer(
            catalog=catalog,
            forecast_artifact=final_artifact,
            native_truth=native_truth,
            geometry=geometry,
            event_threshold_record=threshold_record,
            target_frames=targets,
        )
    with PDERefinerStageForecastArtifact(
        stage_forecast_path,
        expected_sha256=generation["stage_forecast"]["sha256"],
        target_frames=targets,
        seed_bank_path=seed_bank_path,
        seed_bank_sha256=seed_bank_sha256,
    ) as stage_artifact:
        stage_scorer = (
            score_pde_refiner_stages_smoke
            if bounded_smoke
            else score_pde_refiner_stages
        )
        stage_score = stage_scorer(
            catalog=catalog,
            stage_artifact=stage_artifact,
            native_truth=native_truth,
            geometry=geometry,
            event_threshold_record=threshold_record,
            target_frames=targets,
        )
    primary_region_spread = _validate_primary_region_spread(final_score)
    final_score_path = output / "score_final_M32.json"
    stage_score_path = output / "score_stages_M4.json"
    write_strict_json_atomic(final_score_path, final_score)
    write_strict_json_atomic(stage_score_path, stage_score)

    result = {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_B4_PDE_Refiner_H1_evaluator_smoke_85604"
            if bounded_smoke
            else "B4_PDE_Refiner_H1_full_one_step_evaluation_85604"
        ),
        "status": (
            "bounded_evaluator_smoke_completed"
            if bounded_smoke
            else "completed_pending_frozen_H_det_H_prob_reduction"
        ),
        "scientific_authority": not bounded_smoke,
        "bounded_non_scientific_smoke": bounded_smoke,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "truth_opened_only_after_both_forecast_hashes": True,
        "absolute_time_used_as_model_input": False,
        "target_frames": [targets[0], targets[-1] + 1],
        "target_count": len(targets),
        "final_ensemble_members": 32,
        "stage_prefix_members": 4,
        "member_prefixes_regenerated": False,
        "posthoc_calibration_applied": False,
        "physics_derived_training_loss_used": False,
        "full_evaluation_preconditions_passed": True,
        "H_det_evaluated": False,
        "H_prob_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "paper0_commit": args.paper0_commit,
        "training_commit": args.training_commit,
        "slurm_job_id": args.slurm_job_id,
        "seed": 1701,
        "selected_epoch": training["selected_epoch"],
        "parameter_count": training["parameter_count"],
        "accelerator": accelerator,
        "training_result": {
            "path": str(training_path),
            "sha256": args.training_result_sha256,
        },
        "bounded_smoke_result": (
            None
            if smoke_path is None
            else {"path": str(smoke_path), "sha256": args.smoke_result_sha256}
        ),
        "training_history_audit": history_audit,
        "selected_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": training["selected_checkpoint"]["sha256"],
        },
        "final_training_state": {
            "path": str(final_training_state),
            "sha256": training_record["final_training_state"]["sha256"],
            "used_for_evaluation": False,
        },
        "checkpoint_selection_seed_bank": {
            "path": str(selection_bank_path),
            "sha256": training["selection_seed_bank"]["sha256"],
            "used_for_scientific_ensemble": False,
        },
        "scientific_seed_bank": {
            "path": str(seed_bank_path.resolve(strict=True)),
            "sha256": seed_bank_sha256,
            "seed": 41032,
            "shape": [126, 32, 3],
        },
        "artifact_integrity": artifact_integrity,
        "primary_region_nonzero_spread": primary_region_spread,
        "generation": {
            "path": str(generation_path.resolve(strict=True)),
            "sha256": sha256_path(generation_path),
            "peak_cuda_memory_bytes": generation["peak_cuda_memory_bytes"],
        },
        "final_forecast": {
            "path": str(final_forecast_path.resolve(strict=True)),
            "sha256": generation["final_forecast"]["sha256"],
            "bytes": final_forecast_path.stat().st_size,
        },
        "stage_forecast": {
            "path": str(stage_forecast_path.resolve(strict=True)),
            "sha256": generation["stage_forecast"]["sha256"],
            "bytes": stage_forecast_path.stat().st_size,
        },
        "final_score": {
            "path": str(final_score_path.resolve(strict=True)),
            "sha256": sha256_path(final_score_path),
        },
        "stage_score": {
            "path": str(stage_score_path.resolve(strict=True)),
            "sha256": sha256_path(stage_score_path),
            "stage_repair_gate_evaluated": bool(not bounded_smoke),
            "stage_repair_passes": (
                None
                if bounded_smoke
                else stage_score["stagewise_repair"]["passes"]
            ),
        },
        "event_threshold_result": {
            "path": str(threshold_path),
            "sha256": args.event_threshold_result_sha256,
        },
        "evaluation_manifest": {
            "path": str(manifest_path),
            "sha256": args.evaluation_manifest_sha256,
        },
        "evaluation_protocol": {
            "path": str(protocol_path),
            "sha256": args.evaluation_protocol_sha256,
        },
        "metric_source_sha256": verify_locked_b4_metric_sources(),
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    index = _write_index(
        output,
        [
            seed_bank_path,
            generation_path,
            final_forecast_path,
            stage_forecast_path,
            final_score_path,
            stage_score_path,
            result_path,
        ],
        verified_sha256={
            seed_bank_path: seed_bank_sha256,
            final_forecast_path: generation["final_forecast"]["sha256"],
            stage_forecast_path: generation["stage_forecast"]["sha256"],
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "seed": 1701,
                "result": str(result_path.resolve(strict=True)),
                "result_sha256": sha256_path(result_path),
                "artifact_index": str(index.resolve(strict=True)),
                "artifact_index_sha256": sha256_path(index),
                "held_out_85606_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
