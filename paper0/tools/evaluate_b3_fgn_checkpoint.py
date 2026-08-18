#!/usr/bin/env python3
"""Generate and truth-separately score the frozen B3-FGN-H1 seed-1701 model."""

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

from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.fgn_forecast import (  # noqa: E402
    FGNForecastArtifact,
    generate_selected_fgn_forecasts,
    load_selected_fgn_model,
    save_scientific_noise_bank,
    scientific_noise_bank,
)
from tcv_diagnostics.fgn_scoring import (  # noqa: E402
    score_fgn_forecast,
    score_fgn_forecast_smoke,
    verify_locked_metric_sources,
)
from tcv_diagnostics.fgn_training import (  # noqa: E402
    FGNRunConfig,
    ParentArtifacts,
    validation_noise_bank,
)
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
from tcv_diagnostics.o2_context_data import OneStepContextDataset  # noqa: E402


EXPECTED_MANIFEST_SHA256 = (
    "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2"
)
EXPECTED_PROTOCOL_SHA256 = (
    "db717c5605ad9653d2b051ec13254b43bf230f514cb173d295e95d3c68af8030"
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
        raise ValueError(f"B3 training {name} is non-finite")
    return result


def _strict_json_line(line: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token {value}")

    value = json.loads(line, parse_constant=reject_constant)
    if not isinstance(value, Mapping):
        raise ValueError("B3 history line is not a JSON object")
    return value


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    protocol_path: Path,
) -> ParentArtifacts:
    if (
        sha256_path(manifest_path) != EXPECTED_MANIFEST_SHA256
        or sha256_path(protocol_path) != EXPECTED_PROTOCOL_SHA256
        or manifest.get("protocol_status")
        != (
            "frozen_after_passing_B3_smoke_before_full_training_or_scientific_"
            "evaluation_implementation"
        )
        or manifest.get("development_run") != "85604"
        or manifest.get("sequestered_run") != "85606"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("probabilistic_evaluation_authorized") is not True
        or manifest.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("B3 full evaluation manifest contract differs")
    if "B3_FGN_H1_seed1701_one_step_M32_generation_85604_validation" not in (
        manifest.get("authorized_scope", [])
    ):
        raise RuntimeError("B3 M32 evaluation is absent from authorized scope")
    data = manifest.get("data", {})
    ensemble = manifest.get("scientific_ensemble", {})
    model = manifest.get("model", {})
    if (
        data.get("fields") != ["Ne", "Pe", "Pi", "phi", "Vi"]
        or data.get("guard_frames") != [432, 496]
        or data.get("validation_targets") != [498, 624]
        or data.get("zperiod") != 5
        or data.get("mode_mapping") != "n=5k"
        or data.get("absolute_time_input_allowed") is not False
        or data.get("guard_frames_read_allowed") is not False
        or model.get("arm") != "B3-FGN-H1"
        or model.get("context_frames") != 1
        or model.get("future_frames") != 1
        or ensemble.get("seed") != 31032
        or ensemble.get("noise_shape") != [126, 32, 32]
        or ensemble.get("forecast_shape") != [126, 32, 1, 5, 64, 32, 88]
        or ensemble.get("independent_of_checkpoint_selection_noise") is not True
        or ensemble.get("truth_loaded_after_forecast_hash_only") is not True
        or ensemble.get("regeneration_for_member_prefixes_allowed") is not False
        or ensemble.get("posthoc_calibration_allowed") is not False
    ):
        raise RuntimeError("B3 frozen data/model/scientific-ensemble contract differs")
    if manifest.get("locked_metric_sources") != verify_locked_metric_sources():
        raise RuntimeError("B3 manifest/source metric locks differ")
    codec = manifest.get("codec", {})
    parent = manifest.get("deterministic_parent", {})
    artifacts = ParentArtifacts(
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


def audit_full_training_result(
    record: Mapping[str, Any],
    *,
    training_commit: str,
    artifacts: ParentArtifacts,
) -> dict[str, Any]:
    config = FGNRunConfig.frozen(mode="full", seed=1701)
    expected = {
        "scope": "B3_FGN_H1_seed1701_full_training_85604",
        "paper0_commit": str(training_commit),
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
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise ValueError(f"B3 training result field {name!r} differs")
    if json.loads(json.dumps(record.get("config", {}))) != json.loads(
        json.dumps(config.to_record())
    ):
        raise ValueError("B3 training frozen configuration differs")
    selected_epoch = int(record.get("selected_epoch", -1))
    if selected_epoch not in range(100):
        raise ValueError("B3 selected epoch differs")
    for section in ("selected_validation", "final_validation"):
        values = record.get(section, {})
        if int(values.get("examples", -1)) != 126:
            raise ValueError(f"B3 {section} example count differs")
        _finite(values.get("equal_channel_fair_crps"), f"{section}.fair_crps")
        for metric in (
            "fair_crps_by_channel",
            "accuracy_by_channel",
            "spread_by_channel",
        ):
            by_channel = values.get(metric, {})
            if set(by_channel) != {"Ne", "Pe", "Pi", "phi", "Vi"}:
                raise ValueError(f"B3 {section}.{metric} fields differ")
            for field, value in by_channel.items():
                _finite(value, f"{section}.{metric}.{field}")
    if record.get("preoptimization_parent_identity", {}).get("bitwise_exact") is not True:
        raise ValueError("B3 deterministic-parent identity differs")
    if record.get("deterministic_parent_load_audit", {}).get("passed") is not True:
        raise ValueError("B3 deterministic-parent load audit differs")
    probe = record.get("member_probe", {})
    if (
        probe.get("target_frame_index") != 498
        or probe.get("ensemble_size") != 2
        or probe.get("canonical_forecast_shape") != [1, 2, 1, 5, 64, 32, 88]
        or probe.get("finite") is not True
        or probe.get("reload_latent_bitwise_exact") is not True
        or probe.get("reload_forecast_bitwise_exact") is not True
        or probe.get("nonzero_latent_diversity") is not True
        or probe.get("nonzero_field_diversity") is not True
    ):
        raise ValueError("B3 selected-checkpoint member probe differs")
    expected_provenance = (
        (record.get("deterministic_parent", {}), artifacts.checkpoint_path, artifacts.checkpoint_sha256),
        (record.get("codec_checkpoint", {}), artifacts.codec_path, artifacts.codec_sha256),
    )
    for observed, path, digest in expected_provenance:
        if Path(str(observed.get("path", ""))) != path or observed.get("sha256") != digest:
            raise ValueError("B3 parent/codec training provenance differs")
    bank = record.get("validation_noise_bank", {})
    if bank.get("seed") != 31003 or bank.get("shape") != [126, 2, 32]:
        raise ValueError("B3 checkpoint-selection noise provenance differs")
    for name in (
        "selected_checkpoint",
        "final_training_state",
        "history",
        "validation_noise_bank",
    ):
        item = record.get(name, {})
        if not item.get("path") or len(str(item.get("sha256", ""))) != 64:
            raise ValueError(f"B3 training artifact {name!r} is malformed")
    return {
        "config": config,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": dict(record["selected_checkpoint"]),
        "history": dict(record["history"]),
        "selection_noise": dict(record["validation_noise_bank"]),
        "parameter_count": int(record["parameter_count"]),
    }


def _validation_from_history(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "examples": 126,
        "equal_channel_fair_crps": record["validation_equal_channel_fair_crps"],
        "fair_crps_by_channel": record["validation_fair_crps_by_channel"],
        "accuracy_by_channel": record["validation_accuracy_by_channel"],
        "spread_by_channel": record["validation_spread_by_channel"],
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
        raise ValueError("B3 full history does not contain exactly 100 epochs")
    records = [_strict_json_line(line) for line in lines]
    losses: list[float] = []
    selected_so_far: int | None = None
    for epoch, record in enumerate(records):
        if (
            int(record.get("epoch", -1)) != epoch
            or int(record.get("global_step", -1)) != 27 * (epoch + 1)
            or int(record.get("examples", -1)) != 430
            or int(record.get("ensemble_members", -1)) != 2
        ):
            raise ValueError("B3 history epoch/example/step contract differs")
        for name in (
            "common_learning_rate",
            "new_learning_rate",
            "train_equal_channel_fair_crps",
            "validation_equal_channel_fair_crps",
            "mean_preclip_total_gradient_norm",
            "maximum_preclip_total_gradient_norm",
            "mean_preclip_common_gradient_norm",
            "mean_preclip_new_gradient_norm",
            "epoch_wall_seconds",
        ):
            _finite(record.get(name), f"history[{epoch}].{name}")
        loss = float(record["validation_equal_channel_fair_crps"])
        if selected_so_far is None or loss < losses[selected_so_far]:
            selected_so_far = epoch
        if int(record.get("selected_so_far", -1)) != selected_so_far:
            raise ValueError("B3 history running checkpoint selection differs")
        losses.append(loss)
    earliest = min(range(100), key=losses.__getitem__)
    if selected_epoch != earliest:
        raise ValueError("B3 checkpoint is not the earliest fixed-noise fCRPS minimum")
    if json.loads(json.dumps(selected_validation)) != json.loads(
        json.dumps(_validation_from_history(records[earliest]))
    ):
        raise ValueError("B3 selected validation does not match history")
    if json.loads(json.dumps(final_validation)) != json.loads(
        json.dumps(_validation_from_history(records[-1]))
    ):
        raise ValueError("B3 final validation does not match history")
    return {
        "epochs": 100,
        "optimizer_steps": 2700,
        "selection_metric": "fixed_M2_all126_equal_channel_decoded_field_fair_CRPS",
        "earliest_validation_minimum_epoch": earliest,
        "minimum_validation_equal_channel_fair_crps": losses[earliest],
        "finite": True,
    }


def validate_bounded_smoke_result(
    record: Mapping[str, Any],
    *,
    paper0_commit: str,
    training_result_sha256: str,
) -> None:
    if (
        record.get("scope") != "bounded_non_scientific_B3_FGN_H1_evaluator_smoke_85604"
        or record.get("status") != "bounded_evaluator_smoke_completed"
        or record.get("paper0_commit") != paper0_commit
        or record.get("seed") != 1701
        or record.get("target_frames") != [498, 502]
        or record.get("target_count") != 4
        or record.get("ensemble_members") != 32
        or record.get("held_out_85606_read") is not False
        or record.get("truth_opened_only_after_forecast_hash") is not True
        or record.get("full_probabilistic_evaluation_preconditions_passed") is not True
        or record.get("probabilistic_scientific_gate_evaluated") is not False
        or record.get("O3_launch_allowed") is not False
        or record.get("training_result", {}).get("sha256")
        != training_result_sha256
    ):
        raise RuntimeError("B3 bounded evaluator smoke contract differs")


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
    native_path = verify_input(args.native_truth_result, args.native_truth_result_sha256)
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
        raise RuntimeError("B3 evaluator manifest/protocol hash arguments differ")
    if args.mode == "smoke":
        if args.smoke_result is not None or args.smoke_result_sha256 is not None:
            raise RuntimeError("bounded B3 smoke cannot consume a smoke result")
        smoke_path = None
    else:
        if args.smoke_result is None or args.smoke_result_sha256 is None:
            raise RuntimeError("full B3 evaluation requires the bounded smoke result")
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
    selection_noise_path = verify_input(
        Path(training["selection_noise"]["path"]),
        training["selection_noise"]["sha256"],
    )
    selected_training_noise = np.load(selection_noise_path, allow_pickle=False)
    if not np.array_equal(selected_training_noise, validation_noise_bank()):
        raise RuntimeError("B3 checkpoint-selection noise bytes differ")
    del selected_training_noise
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
            paper0_commit=args.paper0_commit,
            training_result_sha256=args.training_result_sha256,
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("B3 evaluation requires exactly one allocated CUDA device")
    device = torch.device("cuda", 0)
    accelerator = torch.cuda.get_device_name(device)
    if "H100" not in accelerator and "H200" not in accelerator:
        raise RuntimeError(f"B3 evaluation requires H100/H200, found {accelerator!r}")
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B3 evaluation {output}")
    output.mkdir(parents=True)

    bank = scientific_noise_bank()
    bank_path = output / "scientific_noise_M32.npy"
    bank_sha256 = save_scientific_noise_bank(bank_path, bank)
    if bank_sha256 == training["selection_noise"]["sha256"]:
        raise RuntimeError("scientific noise unexpectedly equals checkpoint-selection noise")
    catalog = load_official_catalog(args.artifact_root)
    bounded_smoke = args.mode == "smoke"
    targets = tuple(range(498, 502)) if bounded_smoke else tuple(range(498, 624))
    model = load_selected_fgn_model(
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
    forecast_path = output / "forecast_M32.h5"
    metadata = {
        "source_kind": "selected_B3_FGN",
        "arm": "B3-FGN-H1",
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
        "posthoc_calibration_applied": False,
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
        generation = generate_selected_fgn_forecasts(
            model=model,
            dataset=context,
            target_frames=targets,
            noise_bank=bank,
            noise_bank_path=bank_path,
            noise_bank_sha256=bank_sha256,
            output=forecast_path,
            metadata=metadata,
            device=device,
            member_batch_size=args.member_batch_size,
            bounded_smoke=bounded_smoke,
        )
    finally:
        context.close()
    generation_path = output / "generation.json"
    write_strict_json_atomic(generation_path, generation)
    del model, bank
    torch.cuda.empty_cache()

    # Validation truth is first opened below, after the forecast is closed and hashed.
    native_truth = NativeTruthCatalog(load_strict_json(native_path))
    geometry = load_transport_geometry(
        geometry_path=geometry_path,
        geometry_manifest=load_strict_json(geometry_manifest_path),
    )
    threshold_record = load_strict_json(threshold_path)
    with FGNForecastArtifact(
        forecast_path,
        expected_sha256=generation["forecast"]["sha256"],
        target_frames=targets,
        noise_bank_path=bank_path,
        noise_bank_sha256=bank_sha256,
    ) as artifact:
        scorer = score_fgn_forecast_smoke if bounded_smoke else score_fgn_forecast
        score = scorer(
            catalog=catalog,
            forecast_artifact=artifact,
            native_truth=native_truth,
            geometry=geometry,
            event_threshold_record=threshold_record,
            target_frames=targets,
        )
    score_path = output / "score.json"
    write_strict_json_atomic(score_path, score)
    result = {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_B3_FGN_H1_evaluator_smoke_85604"
            if bounded_smoke
            else "B3_FGN_H1_full_probabilistic_evaluation_85604"
        ),
        "status": (
            "bounded_evaluator_smoke_completed"
            if bounded_smoke
            else "completed_pending_frozen_acceptance_gate"
        ),
        "scientific_authority": not bounded_smoke,
        "bounded_non_scientific_smoke": bounded_smoke,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "truth_opened_only_after_forecast_hash": True,
        "absolute_time_used_as_model_input": False,
        "target_frames": [targets[0], targets[-1] + 1],
        "target_count": len(targets),
        "ensemble_members": 32,
        "member_prefixes_regenerated": False,
        "posthoc_calibration_applied": False,
        "physics_derived_training_loss_used": False,
        "full_probabilistic_evaluation_preconditions_passed": True,
        "probabilistic_scientific_gate_evaluated": False,
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
        "checkpoint_selection_noise": {
            "path": str(selection_noise_path),
            "sha256": training["selection_noise"]["sha256"],
            "used_for_scientific_ensemble": False,
        },
        "scientific_noise": {
            "path": str(bank_path.resolve(strict=True)),
            "sha256": bank_sha256,
            "seed": 31032,
            "shape": [126, 32, 32],
        },
        "generation": {
            "path": str(generation_path.resolve(strict=True)),
            "sha256": sha256_path(generation_path),
            "peak_cuda_memory_bytes": generation["peak_cuda_memory_bytes"],
        },
        "forecast": {
            "path": str(forecast_path.resolve(strict=True)),
            "sha256": generation["forecast"]["sha256"],
            "bytes": forecast_path.stat().st_size,
        },
        "score": {
            "path": str(score_path.resolve(strict=True)),
            "sha256": sha256_path(score_path),
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
        "metric_source_sha256": verify_locked_metric_sources(),
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    index = _write_index(
        output,
        [bank_path, generation_path, forecast_path, score_path, result_path],
        verified_sha256={
            bank_path: bank_sha256,
            forecast_path: generation["forecast"]["sha256"],
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
