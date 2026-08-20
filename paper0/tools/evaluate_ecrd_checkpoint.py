#!/usr/bin/env python3
"""Generate and truth-separately score one frozen ECRD model on 85604."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b5_residual_edm_forecast import (  # noqa: E402
    B5ForecastArtifact,
    load_scientific_sampler_seed_bank,
)
from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.ecrd_data import (  # noqa: E402
    ECRDParentMeanArtifact,
    FrozenH1ParentAdapter,
)
from tcv_diagnostics.ecrd_forecast import (  # noqa: E402
    ECRDForecastArtifact,
    HistoricalB5ForecastAdapter,
    generate_selected_ecrd_forecasts,
    load_selected_ecrd_model,
)
from tcv_diagnostics.ecrd_scoring import score_ecrd_forecast  # noqa: E402
from tcv_diagnostics.ecrd_training import (  # noqa: E402
    ECRD_ARMS,
    ECRDTrainingConfig,
    frozen_parameter_counts,
    model_config_record,
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
from tcv_diagnostics.o2_forecast import O2ForecastArtifact  # noqa: E402


BASE_PROTOCOL_SHA256 = (
    "74028e90568a4cfea0721c7fd7a28297a230672c538b3e7908784603c3b2fea4"
)
HISTORICAL_B5_FORECAST_SHA256 = (
    "1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b"
)
HISTORICAL_B5_SEED_BANK_SHA256 = (
    "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ECRD_ARMS, required=True)
    parser.add_argument("--seed", choices=(1701, 1702, 1703), type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--h1-validation-parent", type=Path, required=True)
    parser.add_argument("--h1-validation-parent-sha256", required=True)
    parser.add_argument("--sym-h1-validation-parent", type=Path, required=True)
    parser.add_argument("--sym-h1-validation-parent-sha256", required=True)
    parser.add_argument("--seed-bank", type=Path, required=True)
    parser.add_argument("--seed-bank-sha256", required=True)
    parser.add_argument("--training-result", type=Path)
    parser.add_argument("--training-result-sha256")
    parser.add_argument("--historical-b5-forecast", type=Path)
    parser.add_argument("--historical-b5-forecast-sha256")
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--event-threshold-result", type=Path, required=True)
    parser.add_argument("--event-threshold-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def verify_input(path: Path, expected_sha256: str, label: str) -> Path:
    source = Path(path).resolve(strict=True)
    assert_development_path(source)
    observed = sha256_path(source)
    if observed != str(expected_sha256):
        raise RuntimeError(f"{label} SHA-256 differs: {observed}")
    return source


def require_rocky9_h100() -> dict[str, Any]:
    release = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip().strip('"')
    if release.get("ID") != "rocky" or release.get("VERSION_ID", "").split(".")[0] != "9":
        raise RuntimeError("ECRD forecast generation requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ECRD forecast generation requires one allocated GPU")
    accelerator = torch.cuda.get_device_name(0)
    if "H100" not in accelerator:
        raise RuntimeError(f"ECRD forecast generation requires H100, found {accelerator!r}")
    return {
        "os_id": release["ID"],
        "os_version": release["VERSION_ID"],
        "host": platform.node(),
        "accelerator": accelerator,
        "cuda_device_count": torch.cuda.device_count(),
    }


def _manifest_lock(manifest: Mapping[str, Any], name: str) -> str:
    value = str(manifest.get("evidence_locks", {}).get(name, {}).get("sha256", ""))
    if len(value) != 64:
        raise RuntimeError(f"ECRD evaluation evidence lock {name!r} differs")
    return value


def authorize_evaluation_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    arm: str,
    seed: int,
    input_hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    """Validate the post-training, pre-evaluation result boundary."""

    freeze = ROOT / "paper0/protocol/ECRD_EVALUATION_IMPLEMENTATION_FREEZE.md"
    if (
        sha256_path(manifest_path) != str(manifest_sha256)
        or manifest.get("status")
        != "frozen_after_ECRD_training_before_85604_scientific_evaluation"
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("physics_derived_training_loss_allowed") is not False
        or manifest.get("base_protocol", {}).get("sha256")
        != BASE_PROTOCOL_SHA256
        or sha256_path(ROOT / "paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md")
        != BASE_PROTOCOL_SHA256
        or manifest.get("evaluation_freeze", {}).get("sha256")
        != sha256_path(freeze)
    ):
        raise RuntimeError("ECRD scientific-evaluation manifest scope differs")
    authorized = {
        (str(item["arm"]), int(item["seed"]))
        for item in manifest.get("authorized_runs", ())
    }
    if (arm, int(seed)) not in authorized:
        raise RuntimeError("ECRD arm/seed evaluation is not authorized")
    for name, observed in input_hashes.items():
        if _manifest_lock(manifest, name) != observed:
            raise RuntimeError(f"ECRD evaluation input lock {name!r} differs")
    run = manifest.get("runs", {}).get(arm, {}).get(str(seed), {})
    if not isinstance(run, Mapping):
        raise RuntimeError("ECRD evaluation run record is absent")
    return run


def audit_full_training_result(
    result: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    expected_sha256: str,
    manifest_run: Mapping[str, Any],
) -> dict[str, str]:
    config = ECRDTrainingConfig(arm=arm, seed=seed, mode="full")
    expected_training = json.loads(json.dumps(config.to_record()))
    if (
        result.get("scope") != "ECRD_matched_model_development_training_85604"
        or result.get("status") != "training_completed_checkpoint_selected"
        or result.get("mode") != "full"
        or result.get("arm") != arm
        or result.get("seed") != seed
        or result.get("development_run") != "85604"
        or result.get("training") != expected_training
        or result.get("model") != model_config_record(arm)
        or result.get("parameter_count") != frozen_parameter_counts()[arm]
        or result.get("completed_epochs") != 100
        or result.get("completed_optimizer_steps") != 10_800
        or result.get("target_presentations") != 43_000
        or result.get("candidate_count") != 20
        or result.get("checkpoint_reload_bitwise_exact") is not True
        or result.get("physics_derived_loss_used") is not False
        or result.get("physics_metric_used_for_checkpoint_selection") is not False
        or result.get("target_truth_used_as_condition") is not False
        or result.get("guard_frames_read") is not False
        or result.get("held_out_85606_read") is not False
        or result.get("scientific_forecast_generated") is not False
    ):
        raise RuntimeError("ECRD full-training result contract differs")
    checkpoint = result.get("artifacts", {}).get("selected_checkpoint", {})
    training_commit = str(result.get("paper0_commit", ""))
    if (
        manifest_run.get("kind") != "new_full_training"
        or manifest_run.get("training_result_sha256") != expected_sha256
        or manifest_run.get("training_commit") != training_commit
        or manifest_run.get("selected_checkpoint_sha256")
        != checkpoint.get("sha256")
    ):
        raise RuntimeError("ECRD manifest/training identity differs")
    checkpoint_path = verify_input(
        Path(checkpoint["path"]), checkpoint["sha256"], "ECRD selected checkpoint"
    )
    return {
        "training_commit": training_commit,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": str(checkpoint["sha256"]),
    }


def _write_index(
    output: Path,
    artifacts: Sequence[Path],
    *,
    known_hashes: Mapping[Path, str],
) -> Path:
    destination = output / "artifact_sha256.txt"
    if destination.exists():
        raise FileExistsError(destination)
    lines = []
    for path in artifacts:
        digest = known_hashes[path] if path in known_hashes else sha256_path(path)
        lines.append(f"{digest}  {path.resolve(strict=True)}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "H1_validation_parent": verify_input(
            args.h1_validation_parent,
            args.h1_validation_parent_sha256,
            "unsymmetrized H1 validation parent",
        ),
        "sym_H1_validation_parent": verify_input(
            args.sym_h1_validation_parent,
            args.sym_h1_validation_parent_sha256,
            "symmetrized H1 validation parent",
        ),
        "scientific_seed_bank": verify_input(
            args.seed_bank, args.seed_bank_sha256, "scientific sampler seed bank"
        ),
        "native_truth_result": verify_input(
            args.native_truth_result,
            args.native_truth_result_sha256,
            "native truth result",
        ),
        "geometry_manifest": verify_input(
            args.geometry_manifest,
            args.geometry_manifest_sha256,
            "geometry manifest",
        ),
        "geometry": verify_input(
            args.geometry, args.geometry_sha256, "native geometry"
        ),
        "event_threshold_result": verify_input(
            args.event_threshold_result,
            args.event_threshold_result_sha256,
            "transport event thresholds",
        ),
    }


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    runtime_paths = (
        args.manifest,
        args.artifact_root,
        args.h1_validation_parent,
        args.sym_h1_validation_parent,
        args.seed_bank,
        args.training_result,
        args.historical_b5_forecast,
        args.native_truth_result,
        args.geometry_manifest,
        args.geometry,
        args.event_threshold_result,
        args.output,
    )
    if any(path is not None and "85606" in str(path).lower() for path in runtime_paths):
        raise ValueError("held-out paths are prohibited during ECRD evaluation")
    output = Path(args.output)
    assert_development_path(output)
    if output.exists():
        raise FileExistsError(output)

    manifest_path = verify_input(
        args.manifest, args.manifest_sha256, "ECRD evaluation manifest"
    )
    inputs = _input_paths(args)
    input_hashes = {
        "H1_validation_parent": args.h1_validation_parent_sha256,
        "sym_H1_validation_parent": args.sym_h1_validation_parent_sha256,
        "scientific_seed_bank": args.seed_bank_sha256,
        "native_truth_result": args.native_truth_result_sha256,
        "geometry_manifest": args.geometry_manifest_sha256,
        "geometry": args.geometry_sha256,
        "event_threshold_result": args.event_threshold_result_sha256,
    }
    manifest = load_strict_json(manifest_path)
    run_authority = authorize_evaluation_manifest(
        manifest,
        manifest_path=manifest_path,
        manifest_sha256=args.manifest_sha256,
        arm=args.arm,
        seed=args.seed,
        input_hashes=input_hashes,
    )
    model_lock = manifest["evidence_locks"]["model_dataset"]
    for filename, key in (
        ("model_dataset_manifest.json", "manifest_sha256"),
        ("normalization.json", "normalization_sha256"),
        ("artifact_sha256.txt", "artifact_index_sha256"),
    ):
        verify_input(
            args.artifact_root / filename,
            model_lock[key],
            f"ECRD model data {filename}",
        )
    if args.seed_bank_sha256 != HISTORICAL_B5_SEED_BANK_SHA256:
        raise RuntimeError("ECRD scientific seed-bank identity differs")
    seed_bank = load_scientific_sampler_seed_bank(
        inputs["scientific_seed_bank"], args.seed_bank_sha256
    )
    catalog = load_official_catalog(args.artifact_root)
    native_truth = NativeTruthCatalog(load_strict_json(inputs["native_truth_result"]))
    geometry = load_transport_geometry(
        geometry_path=inputs["geometry"],
        geometry_manifest=load_strict_json(inputs["geometry_manifest"]),
    )
    threshold_record = load_strict_json(inputs["event_threshold_result"])
    targets = tuple(range(498, 624))
    output.mkdir(parents=True)
    generation_path: Path | None = None
    forecast_path: Path
    environment: Mapping[str, Any]
    training_provenance: Mapping[str, Any]

    historical = args.arm == "B5" and args.seed == 1701
    if historical:
        if (
            args.training_result is not None
            or args.training_result_sha256 is not None
            or args.historical_b5_forecast is None
            or args.historical_b5_forecast_sha256 != HISTORICAL_B5_FORECAST_SHA256
            or run_authority.get("kind") != "historical_B5_seed1701_forecast_reuse"
            or run_authority.get("forecast_sha256") != HISTORICAL_B5_FORECAST_SHA256
        ):
            raise RuntimeError("historical B5 reuse contract differs")
        forecast_path = verify_input(
            args.historical_b5_forecast,
            args.historical_b5_forecast_sha256,
            "historical B5 M32 forecast",
        )
        environment = {
            "os": platform.platform(),
            "host": platform.node(),
            "forecast_generation_reused": True,
            "GPU_required": False,
        }
        training_provenance = dict(run_authority)
        with B5ForecastArtifact(
            forecast_path,
            expected_sha256=args.historical_b5_forecast_sha256,
            target_frames=targets,
            seed_bank_path=inputs["scientific_seed_bank"],
            seed_bank_sha256=args.seed_bank_sha256,
        ) as historical_artifact:
            artifact = HistoricalB5ForecastAdapter(historical_artifact)
            score = score_ecrd_forecast(
                catalog=catalog,
                forecast_artifact=artifact,
                native_truth=native_truth,
                geometry=geometry,
                event_threshold_record=threshold_record,
                target_frames=targets,
                arm=args.arm,
                model_seed=args.seed,
            )
    else:
        if (
            args.training_result is None
            or args.training_result_sha256 is None
            or args.historical_b5_forecast is not None
            or args.historical_b5_forecast_sha256 is not None
        ):
            raise RuntimeError("new ECRD evaluation training inputs differ")
        training_path = verify_input(
            args.training_result,
            args.training_result_sha256,
            "ECRD full-training result",
        )
        training_provenance = audit_full_training_result(
            load_strict_json(training_path),
            arm=args.arm,
            seed=args.seed,
            expected_sha256=args.training_result_sha256,
            manifest_run=run_authority,
        )
        environment = require_rocky9_h100()
        torch.cuda.set_device(0)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda", 0)
        model = load_selected_ecrd_model(
            checkpoint=Path(training_provenance["checkpoint_path"]),
            expected_checkpoint_sha256=training_provenance["checkpoint_sha256"],
            arm=args.arm,
            seed=args.seed,
            training_commit=training_provenance["training_commit"],
            device=device,
        )
        context_frames = 2 if args.arm == "ECRD-History" else 1
        context = OneStepContextDataset(
            catalog,
            target_frames=targets,
            context_frames=context_frames,
            return_physical=False,
        )
        forecast_path = output / "forecast_M32.h5"
        metadata = {
            "source_kind": "selected_ECRD_model_development_checkpoint",
            "arm": args.arm,
            "seed": args.seed,
            "context_frames": context_frames,
            "checkpoint_sha256": training_provenance["checkpoint_sha256"],
            "training_commit": training_provenance["training_commit"],
            "evaluation_commit": args.paper0_commit,
            "slurm_job_id": str(args.slurm_job_id),
            "target_truth_read": False,
            "held_out_85606_read": False,
            "posthoc_calibration": False,
            "evaluation_manifest_sha256": args.manifest_sha256,
        }
        try:
            with ExitStack() as stack:
                if args.arm in ("B5", "B5-Context"):
                    raw_parent = stack.enter_context(
                        O2ForecastArtifact(
                            inputs["H1_validation_parent"],
                            expected_sha256=args.h1_validation_parent_sha256,
                            target_frames=targets,
                        )
                    )
                    parent = FrozenH1ParentAdapter(raw_parent, split="validation")
                else:
                    parent = stack.enter_context(
                        ECRDParentMeanArtifact(
                            inputs["sym_H1_validation_parent"],
                            split="validation",
                            expected_sha256=args.sym_h1_validation_parent_sha256,
                        )
                    )
                generation = generate_selected_ecrd_forecasts(
                    model=model,
                    arm=args.arm,
                    model_seed=args.seed,
                    dataset=context,
                    parent_artifact=parent,
                    target_frames=targets,
                    seed_bank=seed_bank,
                    seed_bank_path=inputs["scientific_seed_bank"],
                    seed_bank_sha256=args.seed_bank_sha256,
                    output=forecast_path,
                    metadata=metadata,
                    device=device,
                    member_batch_size=args.member_batch_size,
                )
        finally:
            context.close()
        generation_path = output / "generation.json"
        write_strict_json_atomic(generation_path, generation)
        forecast_sha = generation["forecast"]["sha256"]
        del model
        torch.cuda.empty_cache()
        with ECRDForecastArtifact(
            forecast_path,
            expected_sha256=forecast_sha,
            target_frames=targets,
            arm=args.arm,
            model_seed=args.seed,
            seed_bank_path=inputs["scientific_seed_bank"],
            seed_bank_sha256=args.seed_bank_sha256,
        ) as artifact:
            score = score_ecrd_forecast(
                catalog=catalog,
                forecast_artifact=artifact,
                native_truth=native_truth,
                geometry=geometry,
                event_threshold_record=threshold_record,
                target_frames=targets,
                arm=args.arm,
                model_seed=args.seed,
            )

    score_path = output / "score.json"
    write_strict_json_atomic(score_path, score)
    forecast_sha = (
        args.historical_b5_forecast_sha256
        if historical
        else sha256_path(forecast_path)
    )
    result = {
        "schema_version": 1,
        "scope": "ECRD_full_M32_evaluation_85604",
        "status": "completed_pending_three_seed_acceptance_reduction",
        "development_run": "85604",
        "arm": args.arm,
        "model_seed": args.seed,
        "target_frames": [498, 624],
        "target_count": 126,
        "ensemble_members": 32,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "environment": dict(environment),
        "training_provenance": dict(training_provenance),
        "historical_forecast_reused": historical,
        "forecast": {
            "path": str(forecast_path.resolve(strict=True)),
            "sha256": forecast_sha,
            "bytes": forecast_path.stat().st_size,
        },
        "generation": (
            None
            if generation_path is None
            else {
                "path": str(generation_path.resolve(strict=True)),
                "sha256": sha256_path(generation_path),
            }
        ),
        "score": {
            "path": str(score_path.resolve(strict=True)),
            "sha256": sha256_path(score_path),
        },
        "evaluation_manifest": {
            "path": str(manifest_path),
            "sha256": args.manifest_sha256,
        },
        "truth_opened_only_after_forecast_hash": True,
        "target_truth_used_during_forecast_generation": False,
        "physics_derived_training_loss_used": False,
        "posthoc_calibration_used": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "acceptance_gate_evaluated": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    artifacts = [score_path, result_path]
    if generation_path is not None:
        artifacts.insert(0, generation_path)
        artifacts.insert(1, forecast_path)
    index = _write_index(
        output,
        artifacts,
        known_hashes={forecast_path: str(forecast_sha)},
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": args.arm,
                "seed": args.seed,
                "result": str(result_path.resolve(strict=True)),
                "result_sha256": sha256_path(result_path),
                "score_sha256": sha256_path(score_path),
                "artifact_index": str(index.resolve(strict=True)),
                "artifact_index_sha256": sha256_path(index),
                "held_out_85606_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
