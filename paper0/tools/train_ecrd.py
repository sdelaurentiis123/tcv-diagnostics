#!/usr/bin/env python3
"""Train one matched B5/ECRD arm on frozen 85604 development splits."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch

from tcv_diagnostics.b5_residual_forecast import B5TrainingForecastArtifact
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.ecrd_data import (
    ECRDParentMeanArtifact,
    ECRDResidualDataset,
    FrozenH1ParentAdapter,
    validation_sigma_and_noise_from_uint64,
)
from tcv_diagnostics.ecrd_training import (
    ECRD_ARMS,
    ECRD_AUGMENTATION_SEED,
    ECRD_MODEL_SEEDS,
    ECRDTrainingConfig,
    build_model,
    frozen_parameter_counts,
    train_ecrd_arm,
)
from tcv_diagnostics.ecrd_wandb_tracking import ECRDOnlineWandbTracker
from tcv_diagnostics.model_data import load_strict_json, write_strict_json_atomic
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_ROOT,
    VOLUME_SHAPE,
    load_official_catalog,
)
from tcv_diagnostics.models.ecrd import ECRDTransition, MultiscaleNoiseConfig
from tcv_diagnostics.models.field_residual_edm import JointFieldResidualEDM
from tcv_diagnostics.o2_forecast import O2ForecastArtifact
from tcv_diagnostics.o2_training_data import OneStepWindowDataset
from tcv_diagnostics.wandb_tracking import WandbRunSpec


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PROTOCOL_SHA256 = (
    "74028e90568a4cfea0721c7fd7a28297a230672c538b3e7908784603c3b2fea4"
)
EXPECTED_H1_TRAIN_SHA256 = (
    "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
)
EXPECTED_H1_VALIDATION_SHA256 = (
    "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
)
SMOKE_MEMBER_SEEDS = (67_540, 67_541)
SMOKE_EQUIVARIANCE_SHIFTS = (1, 2, 3, 7, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--arm", choices=ECRD_ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=ECRD_MODEL_SEEDS, required=True)
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--h1-training-parent", type=Path, required=True)
    parser.add_argument("--h1-training-parent-sha256", required=True)
    parser.add_argument("--h1-validation-parent", type=Path, required=True)
    parser.add_argument("--h1-validation-parent-sha256", required=True)
    parser.add_argument("--sym-h1-training-parent", type=Path, required=True)
    parser.add_argument("--sym-h1-training-parent-sha256", required=True)
    parser.add_argument("--sym-h1-validation-parent", type=Path, required=True)
    parser.add_argument("--sym-h1-validation-parent-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
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


def require_rocky9_h100() -> dict[str, Any]:
    release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip().strip('"')
    if release.get("ID") != "rocky" or release.get("VERSION_ID", "").split(".")[0] != "9":
        raise RuntimeError("ECRD training requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ECRD training requires exactly one CUDA GPU")
    accelerator = torch.cuda.get_device_name(0)
    if "H100" not in accelerator:
        raise RuntimeError(f"ECRD training requires H100, found {accelerator!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the allocated H100 does not report bfloat16 support")
    return {
        "os_id": release["ID"],
        "os_version": release["VERSION_ID"],
        "accelerator": accelerator,
        "cuda_device_count": torch.cuda.device_count(),
        "bfloat16_supported": True,
    }


def verify_input(path: Path, expected_sha256: str, label: str) -> Path:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    observed = sha256_path(source)
    if observed != str(expected_sha256):
        raise RuntimeError(f"{label} SHA-256 differs: {observed} != {expected_sha256}")
    return source


def _lock_sha(locks: Mapping[str, Any], name: str) -> str:
    record = locks.get(name, {})
    value = str(record.get("sha256", ""))
    if len(value) != 64:
        raise RuntimeError(f"ECRD evidence lock {name!r} lacks SHA-256")
    return value


def authorize_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    mode: str,
    arm: str,
    seed: int,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if sha256_path(manifest_path) != manifest_sha256:
        raise RuntimeError("ECRD execution manifest bytes differ")
    if (
        manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256
        or sha256_path(
            ROOT / "paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md"
        )
        != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("ECRD execution protocol or scope differs")
    expected_status = (
        "frozen_before_ECRD_engineering_smoke"
        if mode == "smoke"
        else "frozen_after_passing_ECRD_smoke_before_full_training"
    )
    if manifest.get("status") != expected_status:
        raise RuntimeError("ECRD execution manifest status differs")
    if arm not in manifest.get("authorized_arms", ()):
        raise RuntimeError("ECRD arm is not authorized")
    if int(seed) not in manifest.get("authorized_seeds", ()):
        raise RuntimeError("ECRD seed is not authorized")
    if mode == "smoke" and int(seed) != 1701:
        raise RuntimeError("the bounded ECRD smoke is seed 1701 only")
    if bool(manifest.get("full_training_authorized")) != (mode == "full"):
        raise RuntimeError("ECRD full-training authorization differs")
    parent_use = manifest.get("symmetrized_parent_use", {})
    if mode == "smoke":
        if (
            parent_use.get("artifact_authority")
            != "bounded_non_scientific_engineering_smoke_only"
            or parent_use.get("execution_device") != "cpu-smoke"
            or parent_use.get("authorized_modes") != ["smoke"]
            or parent_use.get("H100_comparison_required_before_full_training")
            is not True
        ):
            raise RuntimeError("ECRD smoke parent-use restriction differs")
    elif (
        parent_use.get("artifact_authority") != "scientific_H100_parent"
        or parent_use.get("execution_device") != "h100"
        or parent_use.get("authorized_modes") != ["smoke", "full"]
    ):
        raise RuntimeError("ECRD full-training parent authority differs")
    locks = manifest.get("evidence_locks", {})
    for name, observed in input_hashes.items():
        if _lock_sha(locks, name) != observed:
            raise RuntimeError(f"ECRD evidence hash {name!r} differs")
    data = manifest.get("data", {})
    if (
        data.get("training_targets") != [2, 432]
        or data.get("guard_frames") != [432, 496]
        or data.get("validation_targets") != [498, 624]
        or data.get("fields") != ["Ne", "Pe", "Pi", "phi", "Vi"]
        or data.get("periodic_axes_xyz") != [False, False, True]
        or data.get("zperiod") != 5
        or data.get("mode_mapping") != "n=5k"
    ):
        raise RuntimeError("ECRD data contract differs")
    if mode == "smoke":
        smoke = manifest.get("smoke", {})
        if (
            smoke.get("training_targets") != [2, 6]
            or smoke.get("validation_targets") != [498, 502]
            or smoke.get("epochs") != 1
            or smoke.get("optimizer_steps") != 2
            or smoke.get("scientific_result") is not False
        ):
            raise RuntimeError("ECRD bounded-smoke budget differs")
    else:
        exact = manifest.get("exact_implementation", {})
        if exact.get("parameter_counts") != frozen_parameter_counts():
            raise RuntimeError("ECRD frozen parameter counts differ")
        if exact.get("multiscale_noise") != MultiscaleNoiseConfig().to_record():
            raise RuntimeError("ECRD frozen multiscale-noise constants differ")
        smoke = manifest.get("smoke_evidence", {})
        if smoke.get("all_four_arms_passed") is not True:
            raise RuntimeError("ECRD full training lacks a passing four-arm smoke")
    return {
        "authorized": True,
        "scope": f"ECRD_{mode}_{arm}_seed{seed}_85604",
        "mode": mode,
        "arm": arm,
        "seed": int(seed),
        "development_run": "85604",
        "target_truth_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }


def _load_selected_model(
    *, output: Path, arm: str, device: torch.device
) -> JointFieldResidualEDM | ECRDTransition:
    try:
        payload = torch.load(
            output / "selected.pt", map_location="cpu", weights_only=False
        )
    except TypeError:
        payload = torch.load(output / "selected.pt", map_location="cpu")
    model = build_model(arm).to(device, torch.float32)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model


def _relative_rms_error(observed: torch.Tensor, expected: torch.Tensor) -> float:
    difference = torch.mean((observed.float() - expected.float()).square()).sqrt()
    scale = torch.mean(expected.float().square()).sqrt()
    if float(scale) == 0.0:
        return float(difference)
    return float(difference / scale)


@torch.no_grad()
def run_smoke_probe(
    *,
    model: JointFieldResidualEDM | ECRDTransition,
    dataset: ECRDResidualDataset,
    arm: str,
    output: Path,
    device: torch.device,
    optimizer_steps: int,
) -> dict[str, Any]:
    """Exercise full-volume sampling and symmetry without scientific scoring."""

    item = dataset[0]
    condition = torch.from_numpy(np.asarray(item["condition"]))[None].to(
        device=device, dtype=torch.float32
    )
    parent = torch.from_numpy(np.asarray(item["parent_mean"]))[None].to(
        device=device, dtype=torch.float32
    )
    multiscale = arm in ("ECRD", "ECRD-History")
    noises = [
        validation_sigma_and_noise_from_uint64(seed, multiscale=multiscale)[1]
        for seed in SMOKE_MEMBER_SEEDS
    ]
    initial = torch.from_numpy(np.stack(noises))[None].to(
        device=device, dtype=torch.float32
    )
    torch.cuda.reset_peak_memory_stats(device)
    sampled = model.sample_normalized(
        condition,
        initial,
        steps=18,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
    )
    if isinstance(model, ECRDTransition):
        fields = model.compose_fields(parent, condition, sampled)
    else:
        fields = model.compose_fields(parent, sampled)
    torch.cuda.synchronize(device)

    noisy = initial[:, 0]
    sigma = torch.tensor([0.7], device=device, dtype=torch.float32)
    reference = model.denoise(noisy, condition, sigma)
    generator_errors: list[float] = []
    mean_errors: list[float] = []
    for shift in SMOKE_EQUIVARIANCE_SHIFTS:
        shifted = model.denoise(
            torch.roll(noisy, shift, -1),
            torch.roll(condition, shift, -1),
            sigma,
        )
        generator_errors.append(
            _relative_rms_error(shifted, torch.roll(reference, shift, -1))
        )
        if isinstance(model, ECRDTransition) and model.mean_head is not None:
            mean_reference = model.mean_correction_normalized(condition)
            mean_shifted = model.mean_correction_normalized(
                torch.roll(condition, shift, -1)
            )
            mean_errors.append(
                _relative_rms_error(
                    mean_shifted, torch.roll(mean_reference, shift, -1)
                )
            )
    maximum_generator = max(generator_errors)
    maximum_mean = max(mean_errors, default=0.0)
    finite = bool(torch.isfinite(fields).all() and torch.isfinite(sampled).all())
    diversity = float(torch.mean(torch.abs(fields[:, 0] - fields[:, 1])))
    canonical_shape = list(fields.shape)
    symmetry_required = arm in ("ECRD", "ECRD-History")
    symmetry_passed = (not symmetry_required) or (
        maximum_generator <= 1.0e-4 and maximum_mean <= 1.0e-4
    )
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    gates = {
        "finite": finite,
        "canonical_shape": canonical_shape == [1, 2, 1, 5, *VOLUME_SHAPE],
        "member_diversity": math.isfinite(diversity) and diversity > 1.0e-8,
        "network_evaluations": True,
        "peak_memory": peak_bytes < 75 * 1024**3,
        "required_equivariance": symmetry_passed,
    }
    probe = {
        "schema_version": 1,
        "scope": "bounded_non_scientific_ECRD_full_volume_mechanical_probe",
        "arm": arm,
        "optimizer_steps": int(optimizer_steps),
        "canonical_field_shape": canonical_shape,
        "ensemble_members": 2,
        "sampler_steps": 18,
        "network_evaluations_per_member": 35,
        "member_seeds": list(SMOKE_MEMBER_SEEDS),
        "finite": finite,
        "member_diversity": diversity,
        "equivariance_shifts": list(SMOKE_EQUIVARIANCE_SHIFTS),
        "generator_equivariance_errors": generator_errors,
        "mean_head_equivariance_errors": mean_errors,
        "max_generator_equivariance_error": maximum_generator,
        "max_mean_head_equivariance_error": maximum_mean,
        "equivariance_required": symmetry_required,
        "peak_cuda_bytes": peak_bytes,
        "peak_cuda_GiB": float(peak_bytes / 1024**3),
        "gates": gates,
        "all_mechanical_gates_passed": all(gates.values()),
        "scientific_result": False,
        "physics_metric_evaluated": False,
        "held_out_85606_read": False,
    }
    probe_path = output / "smoke_probe.json"
    write_strict_json_atomic(probe_path, probe)
    sample_path = output / "smoke_probe_standardized_fields.npz"
    if sample_path.exists():
        raise FileExistsError(sample_path)
    np.savez_compressed(
        sample_path,
        standardized_fields=fields.to("cpu", torch.float32).numpy(),
        target_frame_index=np.asarray([item["target_frame_index"]], dtype=np.int64),
    )
    if not probe["all_mechanical_gates_passed"]:
        raise RuntimeError(f"{arm} failed its bounded ECRD mechanical smoke")
    return probe


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
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

    manifest_path = args.manifest.resolve()
    try:
        manifest_relative = manifest_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("ECRD execution manifest must be inside the repository") from error
    runtime_paths = (
        args.artifact_root,
        args.h1_training_parent,
        args.h1_validation_parent,
        args.sym_h1_training_parent,
        args.sym_h1_validation_parent,
        args.output,
    )
    if any("85606" in str(path).lower() for path in runtime_paths):
        raise ValueError("held-out paths are prohibited during ECRD training")
    if args.output.exists():
        raise FileExistsError(args.output)

    input_paths = {
        "H1_training_parent": verify_input(
            args.h1_training_parent,
            args.h1_training_parent_sha256,
            "unsymmetrized H1 training parent",
        ),
        "H1_validation_parent": verify_input(
            args.h1_validation_parent,
            args.h1_validation_parent_sha256,
            "unsymmetrized H1 validation parent",
        ),
        "sym_H1_training_parent": verify_input(
            args.sym_h1_training_parent,
            args.sym_h1_training_parent_sha256,
            "symmetrized H1 training parent",
        ),
        "sym_H1_validation_parent": verify_input(
            args.sym_h1_validation_parent,
            args.sym_h1_validation_parent_sha256,
            "symmetrized H1 validation parent",
        ),
    }
    if (
        args.h1_training_parent_sha256 != EXPECTED_H1_TRAIN_SHA256
        or args.h1_validation_parent_sha256 != EXPECTED_H1_VALIDATION_SHA256
    ):
        raise RuntimeError("historical unsymmetrized H1 parent hashes differ")
    manifest = load_strict_json(manifest_path)
    input_hashes = {
        name: sha256_path(path) for name, path in input_paths.items()
    }
    authorization = authorize_manifest(
        manifest,
        manifest_path=manifest_path,
        manifest_sha256=args.manifest_sha256,
        mode=args.mode,
        arm=args.arm,
        seed=args.seed,
        input_hashes=input_hashes,
    )
    model_data_lock = manifest["evidence_locks"]["model_dataset"]
    for filename, key in (
        ("model_dataset_manifest.json", "manifest_sha256"),
        ("normalization.json", "normalization_sha256"),
        ("artifact_sha256.txt", "artifact_index_sha256"),
    ):
        verify_input(
            args.artifact_root / filename,
            model_data_lock[key],
            f"ECRD model-data {filename}",
        )
    catalog = load_official_catalog(args.artifact_root)
    config = ECRDTrainingConfig(arm=args.arm, seed=args.seed, mode=args.mode)
    noise_config = MultiscaleNoiseConfig()
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type=f"ecrd_{args.mode}_training",
        tags=(
            "paper0",
            "ecrd",
            args.mode,
            args.arm.lower().replace("-", "_"),
            "85604-only",
            f"seed{args.seed}",
        ),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": authorization["scope"],
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "authorization": authorization,
        "manifest": {
            "path": str(manifest_relative),
            "sha256": sha256_path(manifest_path),
        },
        "inputs": {
            **input_hashes,
            "model_data_root": str(args.artifact_root),
            "development_run": "85604",
            "target_truth_used_as_condition": False,
            "held_out_85606_read": False,
        },
        "training": config.to_record(),
        "parameter_counts": frozen_parameter_counts(),
        "multiscale_noise": noise_config.to_record(),
        "environment": environment,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "ecrd_model_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/models/ecrd.py"
            ),
            "ecrd_data_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/ecrd_data.py"
            ),
            "ecrd_training_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/ecrd_training.py"
            ),
            "entrypoint_sha256": sha256_path(Path(__file__).resolve()),
        },
    }
    tracker = ECRDOnlineWandbTracker.start(
        spec=spec,
        config=tracking_config,
        tracking_directory=args.output.parent / f".{args.output.name}.wandb",
    )
    training_windows = None
    validation_windows = None
    try:
        history = 2 if args.arm == "ECRD-History" else 1
        augment = args.arm != "B5"
        training_windows = OneStepWindowDataset(
            catalog,
            split="train",
            target_frames=config.train_targets,
            context_frames=history,
            augment=augment,
            seed=ECRD_AUGMENTATION_SEED,
            return_physical=False,
        )
        validation_windows = OneStepWindowDataset(
            catalog,
            split="validation",
            target_frames=config.validation_targets,
            context_frames=history,
            augment=False,
            seed=ECRD_AUGMENTATION_SEED,
            return_physical=False,
        )
        with ExitStack() as stack:
            if args.arm in ("B5", "B5-Context"):
                train_source = stack.enter_context(
                    B5TrainingForecastArtifact(
                        input_paths["H1_training_parent"],
                        expected_sha256=input_hashes["H1_training_parent"],
                    )
                )
                validation_source = stack.enter_context(
                    O2ForecastArtifact(
                        input_paths["H1_validation_parent"],
                        expected_sha256=input_hashes["H1_validation_parent"],
                        target_frames=tuple(range(498, 624)),
                    )
                )
                train_parent = FrozenH1ParentAdapter(train_source, split="train")
                validation_parent = FrozenH1ParentAdapter(
                    validation_source, split="validation"
                )
            else:
                train_parent = stack.enter_context(
                    ECRDParentMeanArtifact(
                        input_paths["sym_H1_training_parent"],
                        split="train",
                        expected_sha256=input_hashes["sym_H1_training_parent"],
                    )
                )
                validation_parent = stack.enter_context(
                    ECRDParentMeanArtifact(
                        input_paths["sym_H1_validation_parent"],
                        split="validation",
                        expected_sha256=input_hashes["sym_H1_validation_parent"],
                    )
                )
                expected_parent_authority = manifest["symmetrized_parent_use"][
                    "artifact_authority"
                ]
                expected_execution_device = manifest["symmetrized_parent_use"][
                    "execution_device"
                ]
                for label, parent_artifact in (
                    ("training", train_parent),
                    ("validation", validation_parent),
                ):
                    if (
                        parent_artifact.artifact_authority
                        != expected_parent_authority
                        or parent_artifact.execution_device
                        != expected_execution_device
                    ):
                        raise RuntimeError(
                            f"ECRD {label} parent execution authority differs"
                        )
            training_dataset = ECRDResidualDataset(
                training_windows,
                train_parent,
                split="train",
                history_frames=history,
                augment=augment,
            )
            validation_dataset = ECRDResidualDataset(
                validation_windows,
                validation_parent,
                split="validation",
                history_frames=history,
                augment=False,
            )
            result = train_ecrd_arm(
                training_dataset=training_dataset,
                validation_dataset=validation_dataset,
                output=args.output,
                device=device,
                paper0_commit=args.paper0_commit,
                slurm_job_id=args.slurm_job_id,
                authority=authorization,
                config=config,
                noise_config=noise_config,
                on_epoch=tracker.log_epoch,
            )
            if args.mode == "smoke":
                selected = _load_selected_model(
                    output=args.output, arm=args.arm, device=device
                )
                probe = run_smoke_probe(
                    model=selected,
                    dataset=validation_dataset,
                    arm=args.arm,
                    output=args.output,
                    device=device,
                    optimizer_steps=config.total_optimizer_steps,
                )
                tracker.log_smoke_probe(probe)
            tracking = tracker.finish_success(result)
            write_strict_json_atomic(args.output / "wandb.json", tracking)
    except BaseException:
        tracker.finish_failure()
        raise
    finally:
        if training_windows is not None:
            training_windows.close()
        if validation_windows is not None:
            validation_windows.close()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
