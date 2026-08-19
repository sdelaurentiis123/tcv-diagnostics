#!/usr/bin/env python3
"""Generate and truth-separately audit the frozen H1 training residual."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b2_field_metrics import (  # noqa: E402
    B2_OVERLAPPING_REGIONS,
    B2_PRIMARY_REGIONS,
)
from tcv_diagnostics.b5_residual_audit import (  # noqa: E402
    audit_training_residual,
    write_residual_audit_figures,
)
from tcv_diagnostics.b5_residual_forecast import (  # noqa: E402
    B5_TRAINING_TARGETS,
    B5TrainingContextDataset,
    B5TrainingForecastArtifact,
    generate_frozen_h1_training_forecast,
)
from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.matched_o1_transport import load_transport_geometry  # noqa: E402
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.model_training_data import (  # noqa: E402
    CodecFrameDataset,
    VOLUME_SHAPE,
    load_official_catalog,
)
from tcv_diagnostics.o2_forecast import load_selected_o2_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint-sha256", required=True)
    parser.add_argument("--latent-normalization", type=Path, required=True)
    parser.add_argument("--latent-normalization-sha256", required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest-sha256", required=True)
    parser.add_argument("--audit-protocol", type=Path, required=True)
    parser.add_argument("--audit-protocol-sha256", required=True)
    parser.add_argument("--decorrelation-result", type=Path, required=True)
    parser.add_argument("--decorrelation-result-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
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
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {resolved}: {actual}")
    return resolved


def _verify_audit_authority(
    manifest: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> None:
    if (
        manifest.get("status")
        != "frozen_after_B4_failure_before_B5_residual_audit_implementation_or_execution"
        or manifest.get("development_run") != "85604"
        or manifest.get("sequestered_run") != "85606"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("execution", {}).get("training_performed") is not False
        or manifest.get("post_audit", {}).get("B5_training_authorized") is not False
    ):
        raise RuntimeError("B5 residual-audit authority differs")
    data = manifest["data"]
    if (
        data.get("fields") != ["Ne", "Pe", "Pi", "phi", "Vi"]
        or data.get("training_targets") != [2, 432]
        or data.get("training_target_count") != 430
        or data.get("guard_frames_read_allowed") is not False
        or data.get("validation_frames_read_allowed") is not False
        or data.get("zperiod") != 5
        or data.get("mode_mapping") != "n=5k"
    ):
        raise RuntimeError("B5 residual-audit data contract differs")
    parent = manifest["deterministic_mean"]
    codec = manifest["codec"]
    if (
        parent.get("arm") != "C5P-H1"
        or parent.get("seed") != 1701
        or Path(parent["checkpoint_path"]) != Path(args.checkpoint)
        or parent.get("checkpoint_sha256") != args.checkpoint_sha256
        or parent.get("training_commit") != args.training_commit
        or parent.get("retraining_allowed") is not False
        or Path(codec["checkpoint_path"]) != Path(args.codec_checkpoint)
        or codec.get("checkpoint_sha256") != args.codec_checkpoint_sha256
        or Path(codec["latent_normalization_path"]) != Path(args.latent_normalization)
        or codec.get("latent_normalization_sha256")
        != args.latent_normalization_sha256
    ):
        raise RuntimeError("B5 H1 parent or codec authority differs")


def _decorrelation_frames(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> float:
    lock = manifest["evidence_locks"]["training_decorrelation_reference"]
    representative = record.get("decorrelation", {}).get("representative", {})
    value = float(representative.get("median_one_over_e_frames", -1.0))
    if (
        record.get("run_id") != "85604"
        or record.get("phase") != "phase1_immutable_data_protocol"
        or record.get("blind_test_accessed") is not False
        or record.get("decorrelation", {}).get("training_indices") != [0, 432]
        or record.get("decorrelation", {}).get("status")
        != "diagnostic_only_under_nonstationarity"
        or value != float(lock["frames"])
        or float(representative.get("median_one_over_e_microseconds", -1.0))
        != float(lock["microseconds"])
    ):
        raise RuntimeError("frozen training decorrelation reference differs")
    return value


def _verify_b4_stop_gate(record: Mapping[str, Any]) -> None:
    acceptance = record.get("acceptance", {})
    if (
        acceptance.get("H_det", {}).get("passes") is not False
        or acceptance.get("H_prob", {}).get("passes") is not False
        or record.get("post_gate_instruction")
        != "stop_B4_before_replication_O3_or_assimilation"
    ):
        raise RuntimeError("B4 stop decision differs")


def _region_masks_xy(geometry: Any) -> dict[str, np.ndarray]:
    masks = geometry.region_masks
    eligible = np.asarray(
        masks.strict_wall_interior & masks.operator_interior, dtype=bool
    )
    result = {"eligible_union": eligible}
    for name in (*B2_PRIMARY_REGIONS, *B2_OVERLAPPING_REGIONS):
        result[name] = np.asarray(getattr(masks, name), dtype=bool)
    if any(mask.shape != (64, 32) or not np.any(mask) for mask in result.values()):
        raise RuntimeError("B5 authoritative region mask differs or is empty")
    primary_count = sum(result[name].astype(np.int8) for name in B2_PRIMARY_REGIONS)
    if not np.array_equal(primary_count == 1, eligible) or np.any(primary_count > 1):
        raise RuntimeError("B5 primary regions do not partition eligible cells")
    return result


def _write_npz_atomic(path: Path, values: Mapping[str, np.ndarray]) -> Path:
    destination = Path(path)
    partial = destination.with_name(f".{destination.name}.partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    with partial.open("xb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    return destination


def _write_index(output: Path, artifacts: list[Path]) -> Path:
    index = output / "artifact_sha256.txt"
    if index.exists():
        raise FileExistsError(index)
    lines = [f"{sha256_path(path)}  {path.resolve(strict=True)}" for path in artifacts]
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def main() -> None:
    args = parse_args()
    for path in (
        args.checkpoint,
        args.codec_checkpoint,
        args.latent_normalization,
        args.artifact_root,
        args.geometry_manifest,
        args.geometry,
        args.audit_manifest,
        args.audit_protocol,
        args.decorrelation_result,
        args.output_directory,
    ):
        assert_development_path(path)
    verify_checkout(args.paper0_commit)
    checkpoint = verify_input(args.checkpoint, args.checkpoint_sha256)
    codec_checkpoint = verify_input(
        args.codec_checkpoint, args.codec_checkpoint_sha256
    )
    verify_input(args.latent_normalization, args.latent_normalization_sha256)
    geometry_manifest_path = verify_input(
        args.geometry_manifest, args.geometry_manifest_sha256
    )
    geometry_path = verify_input(args.geometry, args.geometry_sha256)
    audit_manifest_path = verify_input(
        args.audit_manifest, args.audit_manifest_sha256
    )
    verify_input(args.audit_protocol, args.audit_protocol_sha256)
    decorrelation_path = verify_input(
        args.decorrelation_result, args.decorrelation_result_sha256
    )
    manifest = load_strict_json(audit_manifest_path)
    _verify_audit_authority(manifest, args=args)
    b4_lock = manifest["evidence_locks"]["B4_final_gate"]
    b4_gate_path = verify_input(ROOT / b4_lock["path"], b4_lock["sha256"])
    b4_gate = load_strict_json(b4_gate_path)
    _verify_b4_stop_gate(b4_gate)
    decorrelation_record = load_strict_json(decorrelation_path)
    decorrelation_frames = _decorrelation_frames(decorrelation_record, manifest)

    if not torch.cuda.is_available():
        raise RuntimeError("B5 H1 residual audit requires one CUDA worker")
    device = torch.device("cuda")
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B5 residual audit {output}")
    output.mkdir(parents=True)
    wall_started = time.monotonic()
    catalog = load_official_catalog(args.artifact_root)
    geometry = load_transport_geometry(
        geometry_path=geometry_path,
        geometry_manifest=load_strict_json(geometry_manifest_path),
    )
    region_masks = _region_masks_xy(geometry)
    model = load_selected_o2_model(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        codec_checkpoint=codec_checkpoint,
        expected_codec_sha256=args.codec_checkpoint_sha256,
        arm="C5P-H1",
        seed=1701,
        training_commit=args.training_commit,
        device=device,
    )
    transition_parameter_count = sum(
        parameter.numel() for parameter in model.transition.parameters()
    )
    if transition_parameter_count != int(manifest["deterministic_mean"]["parameter_count"]):
        raise RuntimeError("B5 H1 transition parameter count differs")

    contexts = B5TrainingContextDataset(catalog)
    forecast_path = output / "h1_training_forecast.h5"
    metadata = {
        "source_kind": "frozen_selected_O2_transition",
        "arm": "C5P-H1",
        "seed": 1701,
        "context_frames": 1,
        "checkpoint_sha256": args.checkpoint_sha256,
        "codec_checkpoint_sha256": args.codec_checkpoint_sha256,
        "training_commit": args.training_commit,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "target_truth_read": False,
        "training_performed": False,
        "audit_manifest_sha256": args.audit_manifest_sha256,
    }
    try:
        generation = generate_frozen_h1_training_forecast(
            model=model,
            dataset=contexts,
            output=forecast_path,
            metadata=metadata,
            device=device,
        )
    finally:
        contexts.close()
    generation["forecast_closed_and_hashed_utc"] = _utc_now()
    generation_path = output / "generation.json"
    write_strict_json_atomic(generation_path, generation)
    generation_sha256 = sha256_path(generation_path)
    forecast_sha256 = generation["forecast"]["sha256"]
    del model
    torch.cuda.empty_cache()

    # The complete forecast is closed, hashed, and independently reopened
    # before the target-only dataset is even constructed below.
    forecast = np.empty((430, 5, *VOLUME_SHAPE), dtype=np.float32)
    with B5TrainingForecastArtifact(
        forecast_path,
        expected_sha256=forecast_sha256,
    ) as artifact:
        for start in range(0, len(B5_TRAINING_TARGETS), 16):
            stop = min(start + 16, len(B5_TRAINING_TARGETS))
            forecast[start:stop] = artifact.read(start, stop)
        timing = artifact.timing_record()

    truth_read_started_utc = _utc_now()
    truth = np.empty_like(forecast)
    truth_dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split="train",
        frames=B5_TRAINING_TARGETS,
        augment=False,
        seed=1701,
        return_physical=False,
    )
    try:
        for position, target in enumerate(B5_TRAINING_TARGETS):
            item = truth_dataset[position]
            if int(item["frame_index"]) != target:
                raise RuntimeError("B5 target-truth order differs")
            truth[position] = np.asarray(item["volume"], dtype=np.float32)
    finally:
        truth_dataset.close()
    truth_read_completed_utc = _utc_now()

    audit_started = time.monotonic()
    product = audit_training_residual(
        truth=truth,
        forecast=forecast,
        region_masks_xy=region_masks,
        cadence_microseconds=float(manifest["data"]["cadence_microseconds"]),
        training_decorrelation_frames=decorrelation_frames,
        target_start=2,
        target_stop=432,
    )
    audit = dict(product.record)
    audit.update(
        {
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "forecast_sha256": forecast_sha256,
            "generation_sha256": generation_sha256,
            "forecast_closed_and_hashed_before_truth_read": True,
            "forecast_closed_and_hashed_utc": generation[
                "forecast_closed_and_hashed_utc"
            ],
            "truth_read_started_utc": truth_read_started_utc,
            "truth_read_completed_utc": truth_read_completed_utc,
            "audit_wall_seconds": time.monotonic() - audit_started,
        }
    )
    audit_path = output / "residual_audit.json"
    write_strict_json_atomic(audit_path, audit)
    raw_path = _write_npz_atomic(
        output / "raw_accumulators.npz", product.raw_accumulators
    )
    figure_paths = write_residual_audit_figures(
        audit, output_directory=output / "figures"
    )

    result = {
        "schema_version": 1,
        "scope": "B5_frozen_H1_training_residual_audit_85604",
        "status": "completed_architecture_sizing_audit_only",
        "scientific_authority": True,
        "claim_scope": "in_sample_training_residual_architecture_sizing_only",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "validation_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "forecast_closed_and_hashed_before_truth_read": True,
        "training_performed": False,
        "B5_training_authorized": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "target_frames": [2, 432],
        "target_count": 430,
        "fields": ["Ne", "Pe", "Pi", "phi", "Vi"],
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "deterministic_mean": {
            "arm": "C5P-H1",
            "seed": 1701,
            "checkpoint": {"path": str(checkpoint), "sha256": args.checkpoint_sha256},
            "codec_checkpoint": {
                "path": str(codec_checkpoint),
                "sha256": args.codec_checkpoint_sha256,
                "trainable": False,
            },
            "transition_parameter_count": transition_parameter_count,
        },
        "accelerator": torch.cuda.get_device_name(device),
        "generation": {
            "path": str(generation_path.resolve(strict=True)),
            "sha256": generation_sha256,
            "wall_seconds": generation["wall_seconds"],
            "peak_cuda_memory_bytes": generation["peak_cuda_memory_bytes"],
            "inference_timing": timing,
        },
        "forecast": {"path": str(forecast_path.resolve(strict=True)), "sha256": forecast_sha256},
        "residual_audit": {
            "path": str(audit_path.resolve(strict=True)),
            "sha256": sha256_path(audit_path),
        },
        "raw_accumulators": {
            "path": str(raw_path.resolve(strict=True)),
            "sha256": sha256_path(raw_path),
            "keys": sorted(product.raw_accumulators),
        },
        "figures": [
            {"path": str(path.resolve(strict=True)), "sha256": sha256_path(path)}
            for path in figure_paths
        ],
        "audit_manifest": {
            "path": str(audit_manifest_path),
            "sha256": args.audit_manifest_sha256,
        },
        "audit_protocol": {
            "path": str(Path(args.audit_protocol).resolve(strict=True)),
            "sha256": args.audit_protocol_sha256,
        },
        "decorrelation_reference": {
            "path": str(decorrelation_path),
            "sha256": args.decorrelation_result_sha256,
            "frames": decorrelation_frames,
            "status": "diagnostic_only_under_nonstationarity",
        },
        "wall_seconds": time.monotonic() - wall_started,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    artifacts = [
        forecast_path,
        generation_path,
        audit_path,
        raw_path,
        *figure_paths,
        result_path,
    ]
    index = _write_index(output, artifacts)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": sha256_path(result_path),
                "forecast_sha256": forecast_sha256,
                "residual_audit_sha256": sha256_path(audit_path),
                "artifact_index": str(index),
                "artifact_index_sha256": sha256_path(index),
                "B5_training_authorized": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
