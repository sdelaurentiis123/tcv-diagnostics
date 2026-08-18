#!/usr/bin/env python3
"""Generate and truth-separately score one frozen full B2 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b2_forecast import (  # noqa: E402
    B2ForecastArtifact,
    generate_selected_b2_forecasts,
    load_selected_b2_model,
)
from tcv_diagnostics.b2_scoring import score_b2_forecast  # noqa: E402
from tcv_diagnostics.b2_training import B2RunConfig  # noqa: E402
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
from tcv_diagnostics.models.latent_diffusion import (  # noqa: E402
    LatentDiffusionViTConfig,
)
from tcv_diagnostics.o2_context_data import OneStepContextDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(1701, 1702, 1703), required=True)
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
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--member-batch-size", type=int, default=4)
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
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
        raise ValueError(f"B2 training {name} is non-finite")
    return result


def _strict_json_line(line: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token {value}")

    value = json.loads(line, parse_constant=reject_constant)
    if not isinstance(value, Mapping):
        raise ValueError("B2 history line is not a JSON object")
    return value


def audit_full_training_result(
    record: Mapping[str, Any],
    *,
    seed: int,
    training_commit: str,
) -> dict[str, Any]:
    """Verify full completion and earliest-minimum checkpoint selection."""

    config = B2RunConfig.frozen(mode="full", seed=int(seed))
    expected_flags = {
        "scope": "B2_LDM_H2_full_training_85604",
        "paper0_commit": str(training_commit),
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
    }
    for name, expected in expected_flags.items():
        if record.get(name) != expected:
            raise ValueError(f"B2 training result field {name!r} differs")
    run_config = record.get("config", {})
    expected_config = config.to_record()
    if any(run_config.get(name) != value for name, value in expected_config.items()):
        raise ValueError("B2 training frozen configuration differs")
    if run_config.get("model") != LatentDiffusionViTConfig().to_record():
        raise ValueError("B2 training model configuration differs")
    codec_run = run_config.get("codec_checkpoint", {})
    codec_result = record.get("codec_checkpoint", {})
    if codec_run != codec_result or codec_result.get("trainable") is not False:
        raise ValueError("B2 training codec provenance differs")
    selected_epoch = int(record.get("selected_epoch", -1))
    if selected_epoch not in range(200):
        raise ValueError("B2 selected epoch differs")
    for section in ("selected_validation", "final_validation"):
        values = record.get(section, {})
        for name in ("complete", "context", "target"):
            _finite(values.get(name), f"{section}.{name}")
        if int(values.get("examples", -1)) != 126:
            raise ValueError(f"B2 training {section} example count differs")
    sampler = record.get("sampler_probe", {})
    if (
        sampler.get("target_frame_index") != 498
        or sampler.get("ensemble_size") != 2
        or sampler.get("canonical_forecast_shape") != [1, 2, 1, 5, 64, 32, 88]
        or sampler.get("finite") is not True
        or sampler.get("nonzero_latent_diversity") is not True
        or sampler.get("nonzero_decoded_diversity") is not True
    ):
        raise ValueError("B2 training sampler probe differs")
    for artifact in (
        "selected_checkpoint",
        "final_training_state",
        "history",
        "latent_normalization",
    ):
        item = record.get(artifact, {})
        if not item.get("path") or len(str(item.get("sha256", ""))) != 64:
            raise ValueError(f"B2 training artifact {artifact!r} is malformed")
    return {
        "config": config,
        "selected_epoch": selected_epoch,
        "selected_checkpoint": dict(record["selected_checkpoint"]),
        "codec_checkpoint": dict(codec_result),
        "history": dict(record["history"]),
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
    if len(lines) != 200:
        raise ValueError("B2 full history does not contain exactly 200 epochs")
    records = [_strict_json_line(line) for line in lines]
    validation_losses: list[float] = []
    selected_so_far: int | None = None
    for epoch, record in enumerate(records):
        if int(record.get("epoch", -1)) != epoch:
            raise ValueError("B2 history epoch order differs")
        if int(record.get("global_step", -1)) != 27 * (epoch + 1):
            raise ValueError("B2 history optimizer-step count differs")
        if int(record.get("train_examples", -1)) != 430:
            raise ValueError("B2 history training example count differs")
        if int(record.get("validation_examples", -1)) != 126:
            raise ValueError("B2 history validation example count differs")
        for name in (
            "learning_rate",
            "maximum_preclip_gradient_norm",
            "mean_preclip_gradient_norm",
            "train_complete_denoising_loss",
            "train_context_denoising_loss",
            "train_target_denoising_loss",
            "validation_complete_denoising_loss",
            "validation_context_denoising_loss",
            "validation_target_denoising_loss",
        ):
            _finite(record.get(name), f"history[{epoch}].{name}")
        current_loss = float(record["validation_complete_denoising_loss"])
        if selected_so_far is None or current_loss < validation_losses[selected_so_far]:
            selected_so_far = epoch
        if int(record.get("selected_so_far", -1)) != selected_so_far:
            raise ValueError("B2 history running checkpoint selection differs")
        validation_losses.append(current_loss)
    earliest_minimum = min(range(200), key=validation_losses.__getitem__)
    if int(selected_epoch) != earliest_minimum:
        raise ValueError("B2 checkpoint is not the earliest validation-loss minimum")
    for name in ("complete", "context", "target"):
        selected_value = float(
            records[earliest_minimum][f"validation_{name}_denoising_loss"]
        )
        final_value = float(records[-1][f"validation_{name}_denoising_loss"])
        if float(selected_validation.get(name, math.nan)) != selected_value:
            raise ValueError("B2 selected validation does not match history")
        if float(final_validation.get(name, math.nan)) != final_value:
            raise ValueError("B2 final validation does not match history")
    if int(selected_validation.get("examples", -1)) != 126:
        raise ValueError("B2 selected-validation example count differs")
    if int(final_validation.get("examples", -1)) != 126:
        raise ValueError("B2 final-validation example count differs")
    return {
        "epochs": 200,
        "optimizer_steps": 5400,
        "earliest_validation_loss_minimum_epoch": earliest_minimum,
        "minimum_validation_complete_denoising_loss": validation_losses[
            earliest_minimum
        ],
        "finite": True,
    }


def _write_index(output: Path, artifacts: list[Path]) -> Path:
    index = output / "artifact_sha256.txt"
    if index.exists():
        raise FileExistsError(index)
    index.write_text(
        "\n".join(
            f"{sha256_path(path)}  {path.resolve(strict=True)}" for path in artifacts
        )
        + "\n",
        encoding="utf-8",
    )
    return index


def main() -> None:
    args = parse_args()
    for path in (
        args.training_result,
        args.artifact_root,
        args.native_truth_result,
        args.geometry_manifest,
        args.geometry,
        args.event_threshold_result,
        args.evaluation_manifest,
        args.evaluation_protocol,
        args.output_directory,
    ):
        assert_development_path(path)
    verify_checkout(args.paper0_commit)
    training_path = verify_input(
        args.training_result, args.training_result_sha256
    )
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
    manifest = load_strict_json(manifest_path)
    if (
        manifest.get("protocol_status")
        != "frozen_before_B2_full_training_or_scientific_metric_implementation"
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("probabilistic_evaluation_authorized") is not True
        or manifest.get("protocol", {}).get("sha256")
        != args.evaluation_protocol_sha256
    ):
        raise RuntimeError("B2 full evaluation manifest contract differs")
    training_record = load_strict_json(training_path)
    training = audit_full_training_result(
        training_record,
        seed=args.seed,
        training_commit=args.training_commit,
    )
    checkpoint_path = verify_input(
        Path(training["selected_checkpoint"]["path"]),
        training["selected_checkpoint"]["sha256"],
    )
    codec_path = verify_input(
        Path(training["codec_checkpoint"]["path"]),
        training["codec_checkpoint"]["sha256"],
    )
    history_audit = audit_history(
        Path(training["history"]["path"]),
        expected_sha256=training["history"]["sha256"],
        selected_epoch=training["selected_epoch"],
        selected_validation=training_record["selected_validation"],
        final_validation=training_record["final_validation"],
    )
    threshold_record = load_strict_json(threshold_path)
    if threshold_record.get("paper0_commit") != args.paper0_commit:
        raise RuntimeError("B2 threshold artifact evaluation commit differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("B2 evaluation requires exactly one allocated CUDA device")
    device = torch.device("cuda")
    accelerator = torch.cuda.get_device_name(device)
    if "H100" not in accelerator and "H200" not in accelerator:
        raise RuntimeError(f"B2 evaluation requires H100/H200, found {accelerator!r}")
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B2 evaluation {output}")
    output.mkdir(parents=True)

    catalog = load_official_catalog(args.artifact_root)
    native_truth = NativeTruthCatalog(load_strict_json(native_path))
    geometry = load_transport_geometry(
        geometry_path=geometry_path,
        geometry_manifest=load_strict_json(geometry_manifest_path),
    )
    targets = tuple(range(498, 624))
    model = load_selected_b2_model(
        checkpoint=checkpoint_path,
        expected_checkpoint_sha256=training["selected_checkpoint"]["sha256"],
        codec_checkpoint=codec_path,
        expected_codec_sha256=training["codec_checkpoint"]["sha256"],
        seed=args.seed,
        device=device,
        training_commit=args.training_commit,
        expected_selected_epoch=training["selected_epoch"],
    )
    context = OneStepContextDataset(
        catalog,
        target_frames=targets,
        context_frames=2,
        return_physical=False,
    )
    forecast_path = output / "forecast_M32.h5"
    metadata = {
        "source_kind": "selected_B2_LDM",
        "arm": "B2-LDM-H2",
        "seed": args.seed,
        "context_frames": 2,
        "checkpoint_sha256": training["selected_checkpoint"]["sha256"],
        "codec_checkpoint_sha256": training["codec_checkpoint"]["sha256"],
        "training_commit": args.training_commit,
        "evaluation_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "target_truth_read": False,
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
        generation = generate_selected_b2_forecasts(
            model=model,
            model_seed=args.seed,
            dataset=context,
            target_frames=targets,
            output=forecast_path,
            metadata=metadata,
            device=device,
            member_batch_size=args.member_batch_size,
        )
    finally:
        context.close()
    generation_path = output / "generation.json"
    write_strict_json_atomic(generation_path, generation)
    parameter_count = sum(
        parameter.numel() for parameter in model.denoiser.parameters()
    )
    del model
    torch.cuda.empty_cache()

    with B2ForecastArtifact(
        forecast_path,
        expected_sha256=generation["forecast"]["sha256"],
        target_frames=targets,
        model_seed=args.seed,
    ) as artifact:
        score = score_b2_forecast(
            catalog=catalog,
            forecast_artifact=artifact,
            native_truth=native_truth,
            geometry=geometry,
            event_threshold_record=threshold_record,
            target_frames=targets,
            model_seed=args.seed,
        )
    score_path = output / "score.json"
    write_strict_json_atomic(score_path, score)
    result = {
        "schema_version": 1,
        "scope": "B2_LDM_H2_full_probabilistic_evaluation_85604",
        "status": "completed_pending_frozen_acceptance_gate",
        "scientific_authority": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "truth_opened_only_after_forecast_hash": True,
        "physics_derived_training_loss_used": False,
        "probabilistic_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "paper0_commit": args.paper0_commit,
        "training_commit": args.training_commit,
        "slurm_job_id": args.slurm_job_id,
        "seed": args.seed,
        "selected_epoch": training["selected_epoch"],
        "parameter_count": int(parameter_count),
        "accelerator": accelerator,
        "training_result": {
            "path": str(training_path),
            "sha256": args.training_result_sha256,
        },
        "training_history_audit": history_audit,
        "selected_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": training["selected_checkpoint"]["sha256"],
        },
        "codec_checkpoint": {
            "path": str(codec_path),
            "sha256": training["codec_checkpoint"]["sha256"],
            "trainable": False,
        },
        "generation": {
            "path": str(generation_path.resolve(strict=True)),
            "sha256": sha256_path(generation_path),
            "peak_cuda_memory_bytes": generation["peak_cuda_memory_bytes"],
        },
        "forecast": {
            "path": str(forecast_path.resolve(strict=True)),
            "sha256": sha256_path(forecast_path),
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
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    index = _write_index(
        output,
        [generation_path, forecast_path, score_path, result_path],
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "seed": args.seed,
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
