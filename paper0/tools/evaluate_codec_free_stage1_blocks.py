#!/usr/bin/env python3
"""Evaluate one selected full Stage-1 checkpoint in frozen 85604 blocks."""

from __future__ import annotations

import argparse
from dataclasses import fields as dataclass_fields
import json
from pathlib import Path
import subprocess

import torch

from paper0.tools.train_codec_free_stage1_pilot import atomic_json, evaluate
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.models.codec_free_operator import (
    CodecFreeIncrementOperator3D,
    CodecFreeOperatorConfig,
)
from tcv_diagnostics.state_operator_data import LeadTimeStateDataset


BLOCKS = {
    "V00": (498, 540),
    "V01": (540, 582),
    "V02": (582, 624),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-result-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--family", choices=("c5p", "e6b"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    return parser.parse_args()


def repository_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def restore_config(record: dict) -> CodecFreeOperatorConfig:
    allowed = {field.name for field in dataclass_fields(CodecFreeOperatorConfig)}
    values = {key: value for key, value in record.items() if key in allowed}
    if "channel_multipliers" in values:
        values["channel_multipliers"] = tuple(values["channel_multipliers"])
    return CodecFreeOperatorConfig(**values)


def main() -> None:
    args = parse_args()
    for path in (
        args.artifact_root,
        args.training_result,
        args.checkpoint,
        args.output,
        args.paper0_root,
    ):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 evaluation commit differs")
    if sha256_path(args.training_result) != args.training_result_sha256:
        raise ValueError("training result SHA-256 differs")
    if sha256_path(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("selected checkpoint SHA-256 differs")
    result = json.loads(args.training_result.read_text(encoding="utf-8"))
    if result.get("scope") != "post_ecrd_old_85604_stage1_codec_free_full":
        raise ValueError("training result scope differs")
    if result.get("development_run") != "85604":
        raise ValueError("training result development run differs")
    if result.get("held_out_85606_read") is not False:
        raise ValueError("training result held-out flag differs")
    if result.get("physics_derived_loss_used") is not False:
        raise ValueError("training result physics-loss flag differs")
    if result.get("family") != args.family or int(result.get("seed")) != args.seed:
        raise ValueError("training result arm identity differs")
    selected = result.get("best_checkpoint", {})
    if selected.get("sha256") != args.checkpoint_sha256:
        raise ValueError("training result selected checkpoint differs")

    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 block evaluation requires an allocated CUDA GPU")
    device = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if payload.get("family") != args.family or int(payload.get("seed")) != args.seed:
        raise ValueError("checkpoint arm identity differs")
    config = restore_config(payload["config"])
    model = CodecFreeIncrementOperator3D(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    model.eval()

    catalog = load_official_catalog(args.artifact_root)
    block_records = {}
    for name, (target_start, target_stop) in BLOCKS.items():
        dataset = LeadTimeStateDataset(
            catalog,
            family=args.family,
            split="validation",
            lead_steps=(1,),
            history_frames=1,
            augment=False,
            seed=args.seed,
            current_interval=(target_start - 1, target_stop - 1),
        )
        try:
            if len(dataset) != 42:
                raise ValueError(f"{name} does not contain 42 targets")
            block_records[name] = {
                "target_interval": [target_start, target_stop],
                "metrics": evaluate(
                    model, dataset, family=args.family, device=device
                ),
            }
        finally:
            dataset.close()

    output = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_stage1_chronological_block_evaluation",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "physics_derived_loss_used": False,
        "family": args.family,
        "seed": args.seed,
        "paper0_evaluation_commit": args.paper0_commit,
        "paper0_training_commit": result["paper0_commit"],
        "training_result": {
            "path": str(args.training_result),
            "sha256": args.training_result_sha256,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": args.checkpoint_sha256,
            "epoch": int(payload["epoch"]),
            "selection_metric": float(payload["selection_metric"]),
        },
        "blocks": block_records,
    }
    atomic_json(args.output, output)
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
