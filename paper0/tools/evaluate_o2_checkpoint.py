#!/usr/bin/env python3
"""Generate, truth-separately score, and gate one frozen O2 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
from tcv_diagnostics.o2_context_data import OneStepContextDataset  # noqa: E402
from tcv_diagnostics.o2_evaluation import build_o2_seed_gate  # noqa: E402
from tcv_diagnostics.o2_forecast import (  # noqa: E402
    O2ForecastArtifact,
    generate_selected_o2_forecasts,
    load_selected_o2_model,
)
from tcv_diagnostics.o2_scoring import score_o2_forecast  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--arm", choices=("C5P-H1", "C5P-H2"), required=True)
    parser.add_argument("--seed", type=int, choices=(1701, 1702, 1703), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint-sha256", required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--training-freeze-result", type=Path, required=True)
    parser.add_argument("--training-freeze-result-sha256", required=True)
    parser.add_argument("--references-result", type=Path, required=True)
    parser.add_argument("--references-result-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
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
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {resolved}: {actual}")
    return resolved


def frozen_run(
    freeze: dict[str, Any],
    *,
    arm: str,
    seed: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    codec_checkpoint: Path,
    codec_checkpoint_sha256: str,
    training_commit: str,
) -> dict[str, Any]:
    if (
        freeze.get("scope") != "phase2_C5P_O2_full_training_matrix_frozen"
        or freeze.get("status") != "completed_pending_scientific_O2_evaluation"
        or freeze.get("development_run") != "85604"
        or freeze.get("held_out_85606_read") is not False
        or freeze.get("training_commit") != training_commit
        or freeze.get("checkpoint_choice_frozen_before_reference_or_physics_metrics")
        is not True
        or freeze.get("O2_scientific_evaluation_completed") is not False
        or freeze.get("O3_launch_allowed") is not False
    ):
        raise RuntimeError("O2 training freeze contract differs")
    matches = [
        run
        for run in freeze.get("runs", [])
        if run.get("arm") == arm and int(run.get("seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise RuntimeError("O2 freeze does not identify exactly one arm/seed")
    run = matches[0]
    if (
        Path(run["selected_checkpoint"]["path"]) != checkpoint
        or run["selected_checkpoint"]["sha256"] != checkpoint_sha256
        or Path(run["codec_checkpoint"]["path"]) != codec_checkpoint
        or run["codec_checkpoint"]["sha256"] != codec_checkpoint_sha256
        or run["codec_checkpoint"]["trainable_during_O2"] is not False
    ):
        raise RuntimeError("O2 arm/seed artifacts differ from frozen matrix")
    return run


def _load_reference_scores(
    result: dict[str, Any],
    *,
    mode: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if (
        result.get("scope") != "O2_frozen_uncompressed_references"
        or result.get("status") != "completed"
        or result.get("mode") != mode
        or result.get("scientific_authority") is not (mode == "full")
        or result.get("development_run") != "85604"
        or result.get("held_out_85606_read") is not False
        or result.get("validation_tuning_used") is not False
        or result.get("target_truth_read_during_forecast_generation") is not False
        or result.get("O2_seed_gate_evaluated") is not False
        or result.get("O3_launch_allowed") is not False
    ):
        raise RuntimeError("O2 reference result contract differs")
    scores = {}
    for name in ("persistence", "spectral_ar1", "linear_extrapolation"):
        record = result["references"][name]["score"]
        path = verify_input(Path(record["path"]), record["sha256"])
        scores[name] = load_strict_json(path)
        metadata = scores[name].get("forecast_artifact", {}).get("metadata", {})
        if (
            metadata.get("reference_name") != name
            or metadata.get("source_kind") != "uncompressed_reference"
            or metadata.get("target_truth_read") is not False
        ):
            raise RuntimeError(f"O2 reference score {name} identity differs")
    materiality_record = result["training_materiality"]
    materiality_path = verify_input(
        Path(materiality_record["path"]), materiality_record["sha256"]
    )
    materiality = load_strict_json(materiality_path)
    if (
        materiality.get("scope") != "O2_training_truth_materiality"
        or materiality.get("training_frames") != [0, 432]
        or materiality.get("validation_truth_used_to_select_bands") is not False
        or materiality.get("held_out_85606_read") is not False
    ):
        raise RuntimeError("O2 training materiality contract differs")
    return scores, materiality["materiality"]


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
        args.training_freeze_result,
        args.references_result,
        args.artifact_root,
        args.native_truth_result,
        args.geometry_manifest,
        args.geometry,
        args.evaluation_manifest,
        args.output_directory,
    ):
        assert_development_path(path)
    verify_checkout(args.paper0_commit)
    freeze_path = verify_input(
        args.training_freeze_result, args.training_freeze_result_sha256
    )
    references_path = verify_input(
        args.references_result, args.references_result_sha256
    )
    native_path = verify_input(
        args.native_truth_result, args.native_truth_result_sha256
    )
    geometry_manifest_path = verify_input(
        args.geometry_manifest, args.geometry_manifest_sha256
    )
    evaluation_manifest_path = verify_input(
        args.evaluation_manifest, args.evaluation_manifest_sha256
    )
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    codec_path = Path(args.codec_checkpoint).resolve(strict=True)
    verify_input(checkpoint_path, args.checkpoint_sha256)
    verify_input(codec_path, args.codec_checkpoint_sha256)
    run = frozen_run(
        load_strict_json(freeze_path),
        arm=args.arm,
        seed=args.seed,
        checkpoint=checkpoint_path,
        checkpoint_sha256=args.checkpoint_sha256,
        codec_checkpoint=codec_path,
        codec_checkpoint_sha256=args.codec_checkpoint_sha256,
        training_commit=args.training_commit,
    )
    evaluation_manifest = load_strict_json(evaluation_manifest_path)
    if (
        evaluation_manifest.get("status")
        != "frozen_before_O2_scientific_evaluation"
        or evaluation_manifest.get("development_run") != "85604"
        or evaluation_manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise RuntimeError("O2 evaluation manifest contract differs")
    reference_result = load_strict_json(references_path)
    reference_scores, materiality = _load_reference_scores(
        reference_result,
        mode=args.mode,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("O2 checkpoint evaluation requires CUDA")
    device = torch.device("cuda")
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite O2 evaluation {output}")
    output.mkdir(parents=True)

    catalog = load_official_catalog(args.artifact_root)
    native_truth = NativeTruthCatalog(load_strict_json(native_path))
    geometry = load_transport_geometry(
        geometry_path=Path(args.geometry).resolve(strict=True),
        geometry_manifest=load_strict_json(geometry_manifest_path),
    )
    targets = tuple(range(498, 624 if args.mode == "full" else 502))
    context_frames = 1 if args.arm == "C5P-H1" else 2
    model = load_selected_o2_model(
        checkpoint=checkpoint_path,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        codec_checkpoint=codec_path,
        expected_codec_sha256=args.codec_checkpoint_sha256,
        arm=args.arm,
        seed=args.seed,
        training_commit=args.training_commit,
        device=device,
    )
    context = OneStepContextDataset(
        catalog,
        target_frames=targets,
        context_frames=context_frames,
        return_physical=False,
    )
    forecast_path = output / "forecast.h5"
    metadata = {
        "source_kind": "selected_O2_transition",
        "mode": args.mode,
        "arm": args.arm,
        "seed": args.seed,
        "context_frames": context_frames,
        "checkpoint_sha256": args.checkpoint_sha256,
        "codec_checkpoint_sha256": args.codec_checkpoint_sha256,
        "training_commit": args.training_commit,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "target_truth_read": False,
        "evaluation_manifest": {
            "path": str(evaluation_manifest_path),
            "sha256": args.evaluation_manifest_sha256,
        },
    }
    try:
        generation = generate_selected_o2_forecasts(
            model=model,
            dataset=context,
            target_frames=targets,
            output=forecast_path,
            metadata=metadata,
            device=device,
        )
    finally:
        context.close()
    generation_path = output / "generation.json"
    write_strict_json_atomic(generation_path, generation)
    with O2ForecastArtifact(
        forecast_path,
        expected_sha256=generation["forecast"]["sha256"],
        target_frames=targets,
    ) as artifact:
        score = score_o2_forecast(
            catalog=catalog,
            forecast_artifact=artifact,
            native_truth=native_truth,
            geometry=geometry,
            target_frames=targets,
            scientific_authority=args.mode == "full",
        )
    score_path = output / "score.json"
    write_strict_json_atomic(score_path, score)
    gate = (
        build_o2_seed_gate(
            arm=args.arm,
            candidate_score=score,
            reference_scores=reference_scores,
            materiality=materiality,
        )
        if args.mode == "full"
        else None
    )
    parameter_count = sum(parameter.numel() for parameter in model.transition.parameters())
    result = {
        "schema_version": 1,
        "scope": "O2_selected_checkpoint_scientific_evaluation",
        "status": "completed" if args.mode == "full" else "smoke_passed",
        "scientific_authority": args.mode == "full",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "physics_derived_training_loss_used": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "mode": args.mode,
        "arm": args.arm,
        "seed": args.seed,
        "context_frames": context_frames,
        "training_run_index": int(run["run_index"]),
        "selected_epoch": int(run["selected_epoch"]),
        "selected_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": args.checkpoint_sha256,
        },
        "codec_checkpoint": {
            "path": str(codec_path),
            "sha256": args.codec_checkpoint_sha256,
            "trainable": False,
        },
        "parameter_count": parameter_count,
        "accelerator": torch.cuda.get_device_name(device),
        "generation": {
            "path": str(generation_path.resolve(strict=True)),
            "sha256": sha256_path(generation_path),
            "peak_cuda_memory_bytes": generation["peak_cuda_memory_bytes"],
        },
        "forecast": {
            "path": str(forecast_path.resolve(strict=True)),
            "sha256": sha256_path(forecast_path),
        },
        "score": {
            "path": str(score_path.resolve(strict=True)),
            "sha256": sha256_path(score_path),
        },
        "references_result": {
            "path": str(references_path),
            "sha256": args.references_result_sha256,
        },
        "training_freeze_result": {
            "path": str(freeze_path),
            "sha256": args.training_freeze_result_sha256,
        },
        "evaluation_manifest": {
            "path": str(evaluation_manifest_path),
            "sha256": args.evaluation_manifest_sha256,
        },
        "gate": gate,
        "O2_seed_accepted": bool(gate["passes"]) if gate is not None else False,
        "O3_launch_allowed": False,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    artifacts = [forecast_path, generation_path, score_path, result_path]
    index = _write_index(output, artifacts)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": sha256_path(result_path),
                "artifact_index": str(index),
                "artifact_index_sha256": sha256_path(index),
                "O2_seed_accepted": result["O2_seed_accepted"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
