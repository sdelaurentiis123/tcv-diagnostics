#!/usr/bin/env python3
"""Time one truth-free four-phase H1 parent evaluation on a Rusty CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from tcv_diagnostics.b5_residual_forecast import B5TrainingContextDataset
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.ecrd_data import ECRD_TRAIN_TARGETS
from tcv_diagnostics.model_data import write_strict_json_atomic
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_ROOT,
    load_official_catalog,
)
from tcv_diagnostics.models.ecrd import symmetrized_h1_mean
from tcv_diagnostics.o2_forecast import load_selected_o2_model


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MANIFEST_SHA256 = (
    "6fa7c02499cb94dac13d29d797b3f06693d2b07a922e0812a033079ea7185fa5"
)
EXPECTED_PROTOCOL_SHA256 = (
    "74028e90568a4cfea0721c7fd7a28297a230672c538b3e7908784603c3b2fea4"
)
EXPECTED_H1_SHA256 = "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
EXPECTED_CODEC_SHA256 = (
    "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
)
EXPECTED_H1_TRAINING_COMMIT = "9035bc3ce9d2351cd17586f4429af8116d43a47e"
PROBE_TARGET_FRAME = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--h1-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"Paper 0 commit mismatch: {actual} != {expected_commit}")
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"Paper 0 checkout is dirty:\n{dirty}")


def require_rocky9_cpu() -> dict[str, Any]:
    release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip().strip('"')
    if (
        release.get("ID") != "rocky"
        or release.get("VERSION_ID", "").split(".")[0] != "9"
    ):
        raise RuntimeError("ECRD CPU timing probe requires Rocky Linux 9")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "NoDevFiles"):
        raise RuntimeError("ECRD CPU timing probe must hide CUDA devices")
    return {
        "os_id": release["ID"],
        "os_version": release["VERSION_ID"],
        "device": "cpu",
        "torch_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def verify_locked_inputs(args: argparse.Namespace) -> None:
    paths = (args.artifact_root, args.h1_checkpoint, args.codec_checkpoint, args.output)
    if any("85606" in str(path).lower() for path in paths):
        raise ValueError("held-out paths are prohibited during the CPU timing probe")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = ROOT / "paper0/manifests/ecrd_sym_h1_parent_85604.json"
    protocol = ROOT / "paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md"
    locked = {
        manifest: EXPECTED_MANIFEST_SHA256,
        protocol: EXPECTED_PROTOCOL_SHA256,
        args.h1_checkpoint: EXPECTED_H1_SHA256,
        args.codec_checkpoint: EXPECTED_CODEC_SHA256,
    }
    for path, expected in locked.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"locked input SHA-256 differs for {path}")


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    environment = require_rocky9_cpu()
    verify_locked_inputs(args)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cpu")

    load_started = time.perf_counter()
    catalog = load_official_catalog(args.artifact_root)
    model = load_selected_o2_model(
        checkpoint=args.h1_checkpoint,
        expected_checkpoint_sha256=EXPECTED_H1_SHA256,
        codec_checkpoint=args.codec_checkpoint,
        expected_codec_sha256=EXPECTED_CODEC_SHA256,
        arm="C5P-H1",
        seed=1701,
        training_commit=EXPECTED_H1_TRAINING_COMMIT,
        device=device,
    )
    model.eval()
    model.requires_grad_(False)
    model_load_seconds = time.perf_counter() - load_started

    dataset = B5TrainingContextDataset(
        catalog,
        target_frames=(PROBE_TARGET_FRAME,),
        context_frames=1,
    )
    try:
        read_started = time.perf_counter()
        item = dataset[0]
        context_read_seconds = time.perf_counter() - read_started
        if (
            int(item["target_frame_index"]) != PROBE_TARGET_FRAME
            or item.get("target_truth_read") is not False
            or "target" in item
        ):
            raise RuntimeError("CPU timing probe context scope differs")
        context = torch.from_numpy(np.asarray(item["context"]))[None].to(
            device=device,
            dtype=torch.float32,
        )
        inference_started = time.perf_counter()
        with torch.inference_mode():
            parent = symmetrized_h1_mean(model, context)
        inference_seconds = time.perf_counter() - inference_started
        parent_cpu = parent.detach().contiguous().to("cpu", torch.float32)
        if tuple(parent_cpu.shape) != (1, 5, 64, 32, 88):
            raise RuntimeError("CPU timing probe output shape differs")
        if not bool(torch.isfinite(parent_cpu).all()):
            raise FloatingPointError("CPU timing probe output is non-finite")
        output_digest = hashlib.sha256(parent_cpu.numpy().tobytes()).hexdigest()
        del parent_cpu, parent, context
    finally:
        dataset.close()

    estimated_full_seconds = inference_seconds * (len(ECRD_TRAIN_TARGETS) + 126)
    result = {
        "schema_version": 1,
        "status": "completed",
        "scope": "non_scientific_ECRD_four_phase_H1_CPU_timing_probe",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "development_run": "85604",
        "probe_target_frame": PROBE_TARGET_FRAME,
        "context_frames_read": [PROBE_TARGET_FRAME - 1],
        "target_truth_read": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "training_performed": False,
        "physics_metric_evaluated": False,
        "scientific_parent_artifact_written": False,
        "environment": environment,
        "model_load_seconds": model_load_seconds,
        "context_read_seconds": context_read_seconds,
        "four_phase_inference_seconds": inference_seconds,
        "conservative_linear_extrapolation": {
            "parent_frames": len(ECRD_TRAIN_TARGETS) + 126,
            "seconds": estimated_full_seconds,
            "hours": estimated_full_seconds / 3600.0,
            "excludes_model_load_and_file_writes": True,
        },
        "transient_output": {
            "shape": [1, 5, 64, 32, 88],
            "dtype": "float32",
            "finite": True,
            "sha256": output_digest,
            "saved": False,
        },
        "maximum_resident_set_size_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }
    write_strict_json_atomic(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
