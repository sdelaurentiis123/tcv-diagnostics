#!/usr/bin/env python3
"""Evaluate the frozen six-arm old-85604 Stage-1 matrix on one GPU."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json


FAMILIES = ("c5p", "e6b")
SEEDS = (1701, 1702, 1703)
BLOCKS = ("V00", "V01", "V02")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--frozen-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    return parser.parse_args()


def validate_frozen_manifest(
    manifest: dict[str, Any], *, paper0_commit: str
) -> list[dict[str, Any]]:
    if manifest.get("scope") != (
        "post_ecrd_old_85604_stage1_block_evaluation_input_freeze"
    ):
        raise ValueError("frozen matrix scope differs")
    if manifest.get("development_run") != "85604":
        raise ValueError("frozen matrix development run differs")
    if manifest.get("held_out_85606_read") is not False:
        raise ValueError("frozen matrix held-out flag differs")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise ValueError("frozen matrix held-out access rule differs")
    if manifest.get("physics_derived_loss_used") is not False:
        raise ValueError("frozen matrix physics-loss flag differs")
    if manifest.get("evaluation_commit") != paper0_commit:
        raise ValueError("frozen matrix evaluation commit differs")
    if tuple(sorted(manifest.get("blocks", {}))) != BLOCKS:
        raise ValueError("frozen matrix block identities differ")
    arms = manifest.get("arms", [])
    identities = {
        (str(arm.get("family")), int(arm.get("seed", -1))) for arm in arms
    }
    expected = {(family, seed) for family in FAMILIES for seed in SEEDS}
    if identities != expected or len(arms) != 6:
        raise ValueError("frozen matrix must contain exactly six arms")
    return sorted(arms, key=lambda arm: (int(arm["seed"]), arm["family"]))


def evaluation_command(
    *,
    python: str,
    evaluator: Path,
    artifact_root: Path,
    arm: dict[str, Any],
    output: Path,
    paper0_root: Path,
    paper0_commit: str,
) -> list[str]:
    training = arm["training_result"]
    checkpoint = arm["checkpoint"]
    return [
        python,
        "-u",
        str(evaluator),
        "--artifact-root",
        str(artifact_root),
        "--training-result",
        str(training["path"]),
        "--training-result-sha256",
        str(training["sha256"]),
        "--checkpoint",
        str(checkpoint["path"]),
        "--checkpoint-sha256",
        str(checkpoint["sha256"]),
        "--family",
        str(arm["family"]),
        "--seed",
        str(arm["seed"]),
        "--output",
        str(output),
        "--paper0-root",
        str(paper0_root),
        "--paper0-commit",
        paper0_commit,
    ]


def validate_block_result(
    result: dict[str, Any], *, family: str, seed: int, commit: str
) -> None:
    if result.get("scope") != (
        "post_ecrd_old_85604_stage1_chronological_block_evaluation"
    ):
        raise ValueError("block result scope differs")
    if result.get("development_run") != "85604":
        raise ValueError("block result development run differs")
    if result.get("held_out_85606_read") is not False:
        raise ValueError("block result held-out flag differs")
    if result.get("guard_frames_read") is not False:
        raise ValueError("block result guard flag differs")
    if result.get("physics_derived_loss_used") is not False:
        raise ValueError("block result physics-loss flag differs")
    if result.get("family") != family or int(result.get("seed", -1)) != seed:
        raise ValueError("block result arm identity differs")
    if result.get("paper0_evaluation_commit") != commit:
        raise ValueError("block result evaluation commit differs")
    if tuple(sorted(result.get("blocks", {}))) != BLOCKS:
        raise ValueError("block result identities differ")


def main() -> None:
    args = parse_args()
    for path in (
        args.artifact_root,
        args.frozen_manifest,
        args.output_dir,
        args.paper0_root,
    ):
        assert_development_path(path)
    if sha256_path(args.frozen_manifest) != args.frozen_manifest_sha256:
        raise ValueError("frozen matrix SHA-256 differs")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    evaluator = args.paper0_root / "paper0/tools/evaluate_codec_free_stage1_blocks.py"
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)

    frozen = load_strict_json(args.frozen_manifest)
    arms = validate_frozen_manifest(frozen, paper0_commit=args.paper0_commit)
    evaluations = args.output_dir / "evaluations"
    logs = args.output_dir / "logs"
    evaluations.mkdir(parents=True, exist_ok=False)
    logs.mkdir()

    completed = copy.deepcopy(frozen)
    completed["scope"] = (
        "post_ecrd_old_85604_stage1_block_evaluation_completed_matrix"
    )
    completed["source_frozen_manifest"] = {
        "path": str(args.frozen_manifest),
        "sha256": args.frozen_manifest_sha256,
    }
    completed["paper0_evaluation_commit"] = args.paper0_commit
    commands: list[dict[str, Any]] = []
    completed_by_identity = {
        (str(arm["family"]), int(arm["seed"])): arm
        for arm in completed["arms"]
    }

    for arm in arms:
        family = str(arm["family"])
        seed = int(arm["seed"])
        stem = f"{family}_seed_{seed}"
        result_path = evaluations / f"{stem}.json"
        log_path = logs / f"{stem}.log"
        command = evaluation_command(
            python=sys.executable,
            evaluator=evaluator,
            artifact_root=args.artifact_root,
            arm=arm,
            output=result_path,
            paper0_root=args.paper0_root,
            paper0_commit=args.paper0_commit,
        )
        with log_path.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command,
                check=True,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        result = load_strict_json(result_path)
        validate_block_result(
            result, family=family, seed=seed, commit=args.paper0_commit
        )
        locked = completed_by_identity[(family, seed)]
        locked["block_evaluation"] = {
            "path": str(result_path),
            "sha256": sha256_path(result_path),
        }
        locked["block_evaluation_log"] = {
            "path": str(log_path),
            "sha256": sha256_path(log_path),
        }
        commands.append({"family": family, "seed": seed, "argv": command})

    commands_path = args.output_dir / "commands.json"
    completed_path = args.output_dir / "complete_matrix.json"
    atomic_json(
        commands_path,
        {
            "schema_version": 1,
            "development_run": "85604",
            "held_out_85606_read": False,
            "paper0_evaluation_commit": args.paper0_commit,
            "commands": commands,
        },
    )
    completed["commands"] = {
        "path": str(commands_path),
        "sha256": sha256_path(commands_path),
    }
    atomic_json(completed_path, completed)
    print(
        json.dumps(
            {
                "status": "passed",
                "development_run": "85604",
                "held_out_85606_read": False,
                "arm_count": len(arms),
                "complete_matrix": {
                    "path": str(completed_path),
                    "sha256": sha256_path(completed_path),
                },
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
