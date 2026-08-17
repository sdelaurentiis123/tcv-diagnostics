#!/usr/bin/env python3
"""Run one frozen O1 deterministic codec experiment on a CUDA worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from tcv_diagnostics.codec_training import CodecRunConfig, train_codec
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_ROOT,
    load_official_catalog,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--codec", choices=("dcae_l20", "dcae_l10"), required=True)
    parser.add_argument("--family", choices=("c5p", "e6b"), required=True)
    parser.add_argument("--seed", type=int, choices=(1701, 1702, 1703), required=True)
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
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


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA worker is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    config = CodecRunConfig.frozen(
        mode=args.mode,
        codec=args.codec,
        family=args.family,
        seed=args.seed,
    )
    catalog = load_official_catalog(args.artifact_root)
    print(
        json.dumps(
            {
                "config": config.to_record(),
                "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "artifact_root": str(args.artifact_root),
                "output": str(args.output),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    result = train_codec(
        config=config,
        catalog=catalog,
        output_directory=args.output,
        paper0_commit=args.paper0_commit,
        slurm_job_id=args.slurm_job_id,
        device=device,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
