#!/usr/bin/env python3
"""Generate and truth-separately score the frozen B5 residual-EDM model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
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
    generate_selected_b5_forecasts,
    load_selected_b5_model,
    save_scientific_sampler_seed_bank,
)
from tcv_diagnostics.b5_residual_edm_full_training import (  # noqa: E402
    B5EDMFullConfig,
    B5_FULL_VALIDATION_TARGETS,
    B5_SCIENTIFIC_BANK_NPY_SHA256,
    full_training_order,
    full_validation_seed_bank,
    scientific_sampler_seed_bank,
)
from tcv_diagnostics.b5_residual_edm_scoring import (  # noqa: E402
    score_b5_forecast,
    score_b5_forecast_smoke,
    verify_locked_metric_sources,
)
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
from tcv_diagnostics.models.field_residual_edm import (  # noqa: E402
    FieldResidualUNetConfig,
)
from tcv_diagnostics.o2_context_data import OneStepContextDataset  # noqa: E402
from tcv_diagnostics.o2_forecast import O2ForecastArtifact  # noqa: E402


EXPECTED_MANIFEST_SHA256 = (
    "61f1fa565e2bcff008cbe72909daa97362dabe96d160a9beee4a3d5aa87d1334"
)
EXPECTED_PROTOCOL_SHA256 = (
    "faab336bf3ae1a49008eff0e6604d48d9c475aa83732184668c4c2e444c928b9"
)
EXPECTED_H1_FORECAST_SHA256 = (
    "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
)
EXPECTED_NATIVE_TRUTH_SHA256 = (
    "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
)
EXPECTED_GEOMETRY_MANIFEST_SHA256 = (
    "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
)
EXPECTED_GEOMETRY_SHA256 = (
    "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
)
EXPECTED_EVENT_THRESHOLD_SHA256 = (
    "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb"
)


class DeterministicMeanView:
    """Expose a contiguous target view without copying the frozen H1 artifact."""

    def __init__(
        self,
        parent: O2ForecastArtifact,
        target_frames: Sequence[int],
    ) -> None:
        targets = tuple(int(frame) for frame in target_frames)
        if parent.target_frames != B5_FULL_VALIDATION_TARGETS:
            raise ValueError(
                "B5 H1 parent forecast does not cover all validation targets"
            )
        if not targets or targets != tuple(range(targets[0], targets[-1] + 1)):
            raise ValueError("B5 H1 mean view targets must be contiguous")
        if targets[0] < 498 or targets[-1] >= 624:
            raise ValueError("B5 H1 mean view leaves frozen validation")
        self.parent = parent
        self.target_frames = targets
        self.offset = targets[0] - 498

    def read(self, start: int, stop: int) -> np.ndarray:
        if start < 0 or stop > len(self.target_frames) or stop <= start:
            raise ValueError("B5 H1 mean view interval is invalid")
        return self.parent.read(self.offset + start, self.offset + stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", type=int, choices=(1701,), required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-result-sha256", required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--h1-validation-forecast", type=Path, required=True)
    parser.add_argument("--h1-validation-forecast-sha256", required=True)
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


def verify_input(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    actual = sha256_path(resolved)
    if actual != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return resolved


def require_rocky9_h100() -> tuple[torch.device, str]:
    release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip().strip('"')
    if (
        release.get("ID") != "rocky"
        or release.get("VERSION_ID", "").split(".")[0] != "9"
    ):
        raise RuntimeError("B5 evaluation requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("B5 evaluation requires exactly one allocated CUDA GPU")
    device = torch.device("cuda", 0)
    accelerator = torch.cuda.get_device_name(device)
    if "H100" not in accelerator:
        raise RuntimeError(f"B5 evaluation requires H100, found {accelerator!r}")
    return device, accelerator


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    protocol_path: Path,
) -> None:
    if (
        sha256_path(manifest_path) != EXPECTED_MANIFEST_SHA256
        or sha256_path(protocol_path) != EXPECTED_PROTOCOL_SHA256
        or manifest.get("protocol_status")
        != (
            "frozen_after_passing_job_6901469_before_B5_full_training_"
            "validation_or_evaluation_implementation"
        )
        or manifest.get("development_run") != "85604"
        or manifest.get("sequestered_run") != "85606"
        or manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise RuntimeError("B5 evaluation manifest identity differs")
    required_scope = {
        "one_four_target_M32_evaluator_smoke",
        "one_full_126_target_M32_one_step_85604_validation_evaluation",
        "one_prospective_B5_one_seed_acceptance_gate",
    }
    if not required_scope.issubset(set(manifest.get("authorized_scope", ()))):
        raise RuntimeError("B5 evaluation authorization differs")
    for forbidden in (
        "sampler_or_noise_retuning",
        "O3_fixed_block_forecast",
        "O4_autonomous_rollout",
        "assimilation",
        "diagnostic_ranking",
        "85606_access",
    ):
        if forbidden not in manifest.get("forbidden_scope", ()):
            raise RuntimeError(f"B5 forbidden scope is absent: {forbidden}")
    data = manifest.get("data", {})
    expected_data = {
        "fields": ["Ne", "Pe", "Pi", "phi", "Vi"],
        "context_frames": 1,
        "future_frames": 1,
        "guard_frames": [432, 496],
        "validation_targets": [498, 624],
        "volume_shape": [5, 64, 32, 88],
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "absolute_time_input_allowed": False,
        "future_truth_condition_allowed": False,
        "guard_frames_read_allowed": False,
    }
    for name, expected in expected_data.items():
        if data.get(name) != expected:
            raise RuntimeError(f"B5 evaluation data field {name!r} differs")
    model = manifest.get("model", {})
    if (
        model.get("name") != FieldResidualUNetConfig().to_record()["name"]
        or model.get("parameter_count") != 11_604_709
        or model.get("DCAE_or_latent_representation_used") is not False
        or model.get("physics_derived_training_loss_allowed") is not False
    ):
        raise RuntimeError("B5 evaluation model identity differs")
    ensemble = manifest.get("scientific_forecast", {})
    if (
        ensemble.get("target_frames") != [498, 624]
        or ensemble.get("ensemble_size") != 32
        or ensemble.get("seed_bank_seed") != 67_532
        or ensemble.get("seed_bank_shape") != [126, 32]
        or ensemble.get("seed_bank_npy_sha256") != B5_SCIENTIFIC_BANK_NPY_SHA256
        or ensemble.get("canonical_shape") != [126, 32, 1, 5, 64, 32, 88]
        or ensemble.get("truth_separated_generation") is not True
        or any(
            ensemble.get(name) is not False
            for name in (
                "recentring_allowed",
                "inflation_allowed",
                "clipping_allowed",
                "member_rejection_allowed",
                "member_sorting_allowed",
                "regeneration_allowed",
            )
        )
    ):
        raise RuntimeError("B5 scientific forecast contract differs")
    locks = manifest.get("evidence_locks", {})
    if (
        locks.get("H1_validation_forecast", {}).get("sha256")
        != EXPECTED_H1_FORECAST_SHA256
        or locks.get("H1_validation_forecast", {}).get("target_frames") != [498, 624]
        or locks.get("H1_validation_forecast", {}).get("truth_separated") is not True
    ):
        raise RuntimeError("B5 H1 validation-mean lock differs")
    verify_locked_metric_sources()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"B5 {name} is non-finite")
    return result


def _strict_json_line(line: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token {value}")

    record = json.loads(line, parse_constant=reject_constant)
    if not isinstance(record, Mapping):
        raise ValueError("B5 history line is not an object")
    return record


def audit_full_training_result(
    record: Mapping[str, Any],
    *,
    training_commit: str,
) -> dict[str, Any]:
    expected = {
        "scope": "B5_seed1701_full_training_and_data_only_selection_85604",
        "status": "training_completed_checkpoint_selected",
        "paper0_commit": str(training_commit),
        "development_run": "85604",
        "sequestered_run": "85606",
        "completed_epochs": 100,
        "target_presentations": 43_000,
        "completed_optimizer_steps": 10_800,
        "EMA_updates": 10_800,
        "candidate_count": 20,
        "candidate_completed_epochs": list(range(5, 101, 5)),
        "checkpoint_reload_bitwise_exact": True,
        "all_losses_and_gradients_finite": True,
        "parameter_count": 11_604_709,
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "sampled_forecast_metric_used_for_checkpoint_selection": False,
        "target_truth_used_as_condition": False,
        "absolute_time_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "scientific_acceptance_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
    for name, expected_value in expected.items():
        if record.get(name) != expected_value:
            raise ValueError(f"B5 training result field {name!r} differs")
    if record.get("config") != B5EDMFullConfig().to_record():
        raise ValueError("B5 training configuration differs")
    if record.get("model_config") != FieldResidualUNetConfig().to_record():
        raise ValueError("B5 training model configuration differs")
    selected_epoch = int(record.get("selected_completed_epoch", -1))
    if selected_epoch not in range(5, 101, 5):
        raise ValueError("B5 selected candidate epoch differs")
    if record.get("selected_optimizer_step") != selected_epoch * 108:
        raise ValueError("B5 selected candidate optimizer step differs")
    for name in ("selected_validation", "final_candidate_validation"):
        validation = record.get(name, {})
        expected_identity = {
            "target_frames": [498, 624],
            "target_count": 126,
            "probes_per_target": 4,
            "probe_count": 504,
            "precision": "float32_no_autocast_TF32_disabled",
        }
        for key, expected_value in expected_identity.items():
            if validation.get(key) != expected_value:
                raise ValueError(f"B5 {name}.{key} differs")
        for key in (
            "mean_EDM_loss",
            "mean_unweighted_MSE",
            "minimum_sigma",
            "maximum_sigma",
            "wall_seconds",
        ):
            _finite(validation.get(key), f"{name}.{key}")
    artifacts = record.get("artifacts", {})
    required = (
        "config",
        "training_order",
        "validation_seed_bank",
        "history",
        "selected_checkpoint",
        "selected_source_candidate",
        "final_training_state",
        "candidate_checkpoints",
    )
    if any(name not in artifacts for name in required):
        raise ValueError("B5 training artifact inventory differs")
    return {
        "selected_completed_epoch": selected_epoch,
        "selected_checkpoint": dict(artifacts["selected_checkpoint"]),
        "history": dict(artifacts["history"]),
        "training_order": dict(artifacts["training_order"]),
        "validation_seed_bank": dict(artifacts["validation_seed_bank"]),
        "final_training_state": dict(artifacts["final_training_state"]),
        "parameter_count": int(record["parameter_count"]),
    }


def audit_history(
    path: Path,
    *,
    expected_sha256: str,
    selected_completed_epoch: int,
    selected_validation: Mapping[str, Any],
    final_validation: Mapping[str, Any],
) -> dict[str, Any]:
    history_path = verify_input(path, expected_sha256, "B5 training history")
    lines = history_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 100:
        raise ValueError("B5 history does not contain exactly 100 epochs")
    candidates: list[tuple[int, Mapping[str, Any]]] = []
    for expected_epoch, line in enumerate(lines, start=1):
        record = _strict_json_line(line)
        if (
            record.get("completed_epoch") != expected_epoch
            or record.get("global_optimizer_step") != expected_epoch * 108
            or record.get("EMA_updates") != expected_epoch * 108
            or record.get("train_target_count") != 430
        ):
            raise ValueError("B5 history epoch/update/example contract differs")
        for name in (
            "train_mean_EDM_loss",
            "train_mean_unweighted_MSE",
            "mean_preclip_gradient_norm",
            "maximum_preclip_gradient_norm",
            "first_learning_rate",
            "last_learning_rate",
            "epoch_wall_seconds",
        ):
            _finite(record.get(name), f"history[{expected_epoch}].{name}")
        should_validate = expected_epoch % 5 == 0
        if record.get("validation_candidate") is not should_validate:
            raise ValueError("B5 history validation cadence differs")
        if should_validate:
            validation = record.get("validation")
            candidate = record.get("candidate", {})
            if not isinstance(validation, Mapping):
                raise ValueError("B5 history validation record is absent")
            _finite(validation.get("mean_EDM_loss"), "history validation loss")
            if (
                candidate.get("completed_epoch") != expected_epoch
                or candidate.get("global_optimizer_step") != expected_epoch * 108
                or candidate.get("validation") != validation
                or not candidate.get("path")
                or len(str(candidate.get("sha256", ""))) != 64
            ):
                raise ValueError("B5 history candidate record differs")
            candidates.append((expected_epoch, validation))
        elif (
            record.get("validation") is not None or record.get("candidate") is not None
        ):
            raise ValueError("B5 non-candidate epoch contains validation")
    if [epoch for epoch, _ in candidates] != list(range(5, 101, 5)):
        raise ValueError("B5 history candidate epochs differ")
    earliest_epoch, earliest_validation = min(
        candidates,
        key=lambda item: (float(item[1]["mean_EDM_loss"]), item[0]),
    )
    if earliest_epoch != selected_completed_epoch:
        raise ValueError("B5 checkpoint is not earliest fixed-bank loss minimum")
    if dict(earliest_validation) != dict(selected_validation):
        raise ValueError("B5 selected validation does not match history")
    if dict(candidates[-1][1]) != dict(final_validation):
        raise ValueError("B5 final validation does not match history")
    return {
        "epochs": 100,
        "optimizer_steps": 10_800,
        "candidate_count": 20,
        "candidate_completed_epochs": list(range(5, 101, 5)),
        "selection_metric": "fixed_seed_validation_EDM_loss",
        "earliest_validation_minimum_completed_epoch": earliest_epoch,
        "minimum_validation_EDM_loss": float(earliest_validation["mean_EDM_loss"]),
        "finite": True,
    }


def validate_bounded_smoke_result(
    record: Mapping[str, Any],
    *,
    paper0_commit: str,
    training_result_sha256: str,
) -> None:
    if (
        record.get("scope")
        != "bounded_non_scientific_B5_residual_EDM_evaluator_smoke_85604"
        or record.get("status") != "bounded_evaluator_smoke_completed"
        or record.get("paper0_commit") != paper0_commit
        or record.get("seed") != 1701
        or record.get("target_frames") != [498, 502]
        or record.get("target_count") != 4
        or record.get("ensemble_members") != 32
        or record.get("held_out_85606_read") is not False
        or record.get("truth_opened_only_after_forecast_hash") is not True
        or record.get("full_evaluation_preconditions_passed") is not True
        or record.get("scientific_acceptance_evaluated") is not False
        or record.get("O3_launch_allowed") is not False
        or record.get("training_result", {}).get("sha256") != training_result_sha256
    ):
        raise RuntimeError("B5 bounded evaluator smoke contract differs")


def _write_index(
    output: Path,
    artifacts: Sequence[Path],
    *,
    known_hashes: Mapping[Path, str],
) -> Path:
    index = output / "artifact_sha256.txt"
    if index.exists():
        raise FileExistsError(index)
    lines = []
    for path in artifacts:
        digest = known_hashes[path] if path in known_hashes else sha256_path(path)
        lines.append(f"{digest}  {path.resolve(strict=True)}")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def main() -> None:
    args = parse_args()
    paths = (
        args.training_result,
        args.artifact_root,
        args.h1_validation_forecast,
        args.native_truth_result,
        args.geometry_manifest,
        args.geometry,
        args.event_threshold_result,
        args.evaluation_manifest,
        args.evaluation_protocol,
        args.output_directory,
    )
    if any("85606" in str(path).lower() for path in paths):
        raise ValueError("B5 evaluation paths may not mention held-out 85606")
    for path in paths:
        assert_development_path(path)
    if args.smoke_result is not None:
        assert_development_path(args.smoke_result)
    verify_checkout(args.paper0_commit)

    training_path = verify_input(
        args.training_result, args.training_result_sha256, "B5 training result"
    )
    h1_path = verify_input(
        args.h1_validation_forecast,
        args.h1_validation_forecast_sha256,
        "B5 frozen H1 validation forecast",
    )
    native_path = verify_input(
        args.native_truth_result,
        args.native_truth_result_sha256,
        "B5 native truth result",
    )
    geometry_manifest_path = verify_input(
        args.geometry_manifest,
        args.geometry_manifest_sha256,
        "B5 geometry manifest",
    )
    geometry_path = verify_input(args.geometry, args.geometry_sha256, "B5 geometry")
    threshold_path = verify_input(
        args.event_threshold_result,
        args.event_threshold_result_sha256,
        "B5 event threshold",
    )
    manifest_path = verify_input(
        args.evaluation_manifest,
        args.evaluation_manifest_sha256,
        "B5 evaluation manifest",
    )
    protocol_path = verify_input(
        args.evaluation_protocol,
        args.evaluation_protocol_sha256,
        "B5 evaluation protocol",
    )
    exact_inputs = (
        (args.h1_validation_forecast_sha256, EXPECTED_H1_FORECAST_SHA256),
        (args.native_truth_result_sha256, EXPECTED_NATIVE_TRUTH_SHA256),
        (args.geometry_manifest_sha256, EXPECTED_GEOMETRY_MANIFEST_SHA256),
        (args.geometry_sha256, EXPECTED_GEOMETRY_SHA256),
        (args.event_threshold_result_sha256, EXPECTED_EVENT_THRESHOLD_SHA256),
        (args.evaluation_manifest_sha256, EXPECTED_MANIFEST_SHA256),
        (args.evaluation_protocol_sha256, EXPECTED_PROTOCOL_SHA256),
    )
    if any(observed != expected for observed, expected in exact_inputs):
        raise RuntimeError("B5 evaluator frozen input identity differs")

    manifest = load_strict_json(manifest_path)
    validate_manifest(
        manifest,
        manifest_path=manifest_path,
        protocol_path=protocol_path,
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
            f"B5 model-data {filename}",
        )

    training_record = load_strict_json(training_path)
    training = audit_full_training_result(
        training_record,
        training_commit=args.training_commit,
    )
    checkpoint_path = verify_input(
        Path(training["selected_checkpoint"]["path"]),
        training["selected_checkpoint"]["sha256"],
        "B5 selected checkpoint",
    )
    history_audit = audit_history(
        Path(training["history"]["path"]),
        expected_sha256=training["history"]["sha256"],
        selected_completed_epoch=training["selected_completed_epoch"],
        selected_validation=training_record["selected_validation"],
        final_validation=training_record["final_candidate_validation"],
    )
    order_path = verify_input(
        Path(training["training_order"]["path"]),
        training["training_order"]["sha256"],
        "B5 training order",
    )
    if not np.array_equal(
        np.load(order_path, allow_pickle=False), full_training_order()
    ):
        raise RuntimeError("B5 training order values differ")
    selection_bank_path = verify_input(
        Path(training["validation_seed_bank"]["path"]),
        training["validation_seed_bank"]["sha256"],
        "B5 checkpoint-selection seed bank",
    )
    selection_bank = np.load(selection_bank_path, allow_pickle=False)
    if not np.array_equal(selection_bank, full_validation_seed_bank()):
        raise RuntimeError("B5 checkpoint-selection seed-bank values differ")
    final_state_path = verify_input(
        Path(training["final_training_state"]["path"]),
        training["final_training_state"]["sha256"],
        "B5 final training state",
    )
    if args.mode == "smoke":
        if args.smoke_result is not None or args.smoke_result_sha256 is not None:
            raise RuntimeError("bounded B5 smoke cannot consume a smoke result")
        smoke_path = None
    else:
        if args.smoke_result is None or args.smoke_result_sha256 is None:
            raise RuntimeError("full B5 evaluation requires the bounded smoke result")
        smoke_path = verify_input(
            args.smoke_result,
            args.smoke_result_sha256,
            "B5 evaluator smoke result",
        )
        validate_bounded_smoke_result(
            load_strict_json(smoke_path),
            paper0_commit=args.paper0_commit,
            training_result_sha256=args.training_result_sha256,
        )

    device, accelerator = require_rocky9_h100()
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B5 evaluation {output}")
    output.mkdir(parents=True)

    bank = scientific_sampler_seed_bank()
    if np.array_equal(bank[:, :4], selection_bank):
        raise RuntimeError("B5 scientific seeds equal checkpoint-selection seeds")
    bank_path = output / "scientific_sampler_seeds_M32.npy"
    bank_sha256 = save_scientific_sampler_seed_bank(bank_path, bank)
    catalog = load_official_catalog(args.artifact_root)
    bounded_smoke = args.mode == "smoke"
    targets = tuple(range(498, 502)) if bounded_smoke else B5_FULL_VALIDATION_TARGETS
    model = load_selected_b5_model(
        checkpoint=checkpoint_path,
        expected_checkpoint_sha256=training["selected_checkpoint"]["sha256"],
        device=device,
        training_commit=args.training_commit,
    )
    context = OneStepContextDataset(
        catalog,
        target_frames=targets,
        context_frames=1,
        return_physical=False,
    )
    forecast_path = output / "forecast_M32.h5"
    metadata = {
        "source_kind": "selected_B5_residual_EDM",
        "arm": "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI",
        "seed": 1701,
        "context_frames": 1,
        "checkpoint_sha256": training["selected_checkpoint"]["sha256"],
        "deterministic_mean_sha256": args.h1_validation_forecast_sha256,
        "training_commit": args.training_commit,
        "evaluation_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "evaluation_mode": args.mode,
        "bounded_non_scientific_smoke": bounded_smoke,
        "target_truth_read": False,
        "absolute_time_input": False,
        "member_prefixes_regenerated": False,
        "posthoc_calibration": False,
        "evaluation_manifest_sha256": args.evaluation_manifest_sha256,
        "evaluation_protocol_sha256": args.evaluation_protocol_sha256,
    }
    try:
        with O2ForecastArtifact(
            h1_path,
            expected_sha256=args.h1_validation_forecast_sha256,
            target_frames=B5_FULL_VALIDATION_TARGETS,
        ) as parent_mean:
            mean_view = DeterministicMeanView(parent_mean, targets)
            generation = generate_selected_b5_forecasts(
                model=model,
                dataset=context,
                deterministic_mean_artifact=mean_view,
                target_frames=targets,
                seed_bank=bank,
                seed_bank_path=bank_path,
                seed_bank_sha256=bank_sha256,
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
    del model, bank, selection_bank
    torch.cuda.empty_cache()

    # Validation truth is first constructed below, after the forecast is closed,
    # hashed, and its immutable generation record has been persisted.
    native_truth = NativeTruthCatalog(load_strict_json(native_path))
    geometry = load_transport_geometry(
        geometry_path=geometry_path,
        geometry_manifest=load_strict_json(geometry_manifest_path),
    )
    threshold_record = load_strict_json(threshold_path)
    with B5ForecastArtifact(
        forecast_path,
        expected_sha256=generation["forecast"]["sha256"],
        target_frames=targets,
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_sha256,
    ) as artifact:
        scorer = score_b5_forecast_smoke if bounded_smoke else score_b5_forecast
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
    metric_sources = verify_locked_metric_sources()
    result = {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_B5_residual_EDM_evaluator_smoke_85604"
            if bounded_smoke
            else "B5_residual_EDM_full_one_step_evaluation_85604"
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
        "full_evaluation_preconditions_passed": True,
        "scientific_acceptance_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "paper0_commit": args.paper0_commit,
        "training_commit": args.training_commit,
        "slurm_job_id": args.slurm_job_id,
        "seed": 1701,
        "selected_completed_epoch": training["selected_completed_epoch"],
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
            "path": str(final_state_path),
            "sha256": training["final_training_state"]["sha256"],
            "used_for_evaluation": False,
        },
        "checkpoint_selection_seed_bank": {
            "path": str(selection_bank_path),
            "sha256": training["validation_seed_bank"]["sha256"],
            "used_for_scientific_ensemble": False,
        },
        "deterministic_mean_forecast": {
            "path": str(h1_path),
            "sha256": args.h1_validation_forecast_sha256,
            "trainable": False,
            "regenerated": False,
        },
        "scientific_sampler_seed_bank": {
            "path": str(bank_path.resolve(strict=True)),
            "sha256": bank_sha256,
            "seed": 67_532,
            "shape": [126, 32],
            "dtype": "uint64",
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
        "metric_source_sha256": metric_sources,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    index = _write_index(
        output,
        (bank_path, generation_path, forecast_path, score_path, result_path),
        known_hashes={
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
