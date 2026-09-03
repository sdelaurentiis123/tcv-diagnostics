#!/usr/bin/env python3
"""Freeze controls and gradient scaling for hierarchical PGL transport training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from paper0.tools.generate_persistent_global_local_forecast import load_selected_models
from paper0.tools.train_persistent_global_local_pilot import load_parent
from tcv_diagnostics.autoregressive_training import AutoregressiveStateWindowDataset
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.matched_o1_transport import load_transport_geometry
from tcv_diagnostics.model_data import (
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.persistent_global_local_authority import (
    authorize_pgl_evaluation_manifest,
)
from tcv_diagnostics.persistent_global_local_training import (
    PGL_NOISE_BASE_SEED,
    mean_forecast_trajectory,
    keyed_sigma_and_noise,
    tensor_window,
    weighted_mean_state_loss,
)
from tcv_diagnostics.pgl_hierarchical_training import (
    PGL_HIERARCHICAL_CONTROL_STARTS,
    PGL_HIERARCHICAL_GRADIENT_STARTS,
    PGL_HIERARCHICAL_MEAN_LR,
    PGL_HIERARCHICAL_STOCHASTIC_LR,
    PGL_HIERARCHICAL_TARGET_GRADIENT_RATIO,
    HierarchicalControlMagnitudes,
    parameter_branches,
    score_hierarchical_terms,
)
from tcv_diagnostics.pgl_hierarchical_transport import (
    PGL_HIERARCHICAL_LOW_N,
    PGL_HIERARCHICAL_TRANSPORT_N,
    fair_crps_score,
    fair_energy_score,
    global_sum_from_k0,
    global_transport_sum,
    regional_transport_sums,
)
from tcv_diagnostics.pgl_torch_transport import (
    PGL_TRANSPORT_QUANTITIES,
    TorchSeparatrixTransport,
    decoder_records_from_normalization,
)
from tcv_diagnostics.pgl_variogram import differentiable_sample_normalized
from tcv_diagnostics.pgl_variogram_training import (
    keyed_sampler_initial_noise,
    load_pair_banks,
    load_training_transport_truth,
    training_transport_window,
)


PARENT_CHECKPOINT_SHA256 = (
    "4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb"
)
PROTOCOL_NAME = (
    "POST_ECRD_OLD_85604_PGL_HIERARCHICAL_TRANSPORT_TRAINING_2026-09-02.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--prior-preflight", type=Path, required=True)
    parser.add_argument("--prior-preflight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def verify_checkout(root: Path, expected: str) -> None:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if commit != str(expected) or dirty:
        raise RuntimeError("hierarchical preflight requires the locked clean checkout")


def _locked_artifact(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    assert_development_path(path)
    digest = str(record.get("sha256", ""))
    if len(digest) != 64 or not path.is_file() or sha256_path(path) != digest:
        raise ValueError(f"{label} path or SHA-256 differs")
    return path.resolve(strict=True)


def authorize_prior_preflight(path: Path, digest: str) -> tuple[dict[str, Any], dict[str, Path]]:
    if sha256_path(path) != str(digest):
        raise ValueError("prior variogram preflight SHA-256 differs")
    result = load_strict_json(path)
    if (
        result.get("scope") != "post_ecrd_old_85604_pgl_variogram_preflight"
        or result.get("status") != "passed"
        or result.get("development_run") != "85604"
        or result.get("known_answer_gates", {}).get("passed") is not True
        or result.get("sampler_regression_gate", {}).get("passed") is not True
        or result.get("transport_equivalence_gate", {}).get("passed") is not True
        or result.get("parent", {}).get("selected_checkpoint", {}).get("sha256")
        != PARENT_CHECKPOINT_SHA256
        or result.get("held_out_85606_read") is not False
        or result.get("new_nersc_data_read") is not False
    ):
        raise ValueError("prior variogram preflight contract differs")
    return result, {
        "pair_banks": _locked_artifact(result["pair_banks"], label="pair banks"),
        "native_transport_truth": _locked_artifact(
            result["native_transport_truth"], label="native transport truth"
        ),
    }


def hierarchy_known_answer_gates() -> dict[str, Any]:
    truth = torch.tensor([[0.0, 1.0, -2.0]], dtype=torch.float32)
    exact = truth[:, None].repeat(1, 4, 1)
    biased = exact + 0.5
    energy_exact = fair_energy_score(exact, truth)
    energy_biased = fair_energy_score(biased, truth)
    crps_exact = fair_crps_score(exact, truth)
    crps_biased = fair_crps_score(biased, truth)
    generator = torch.Generator().manual_seed(856040902)
    local = torch.randn((2, 4, 4, 16, 81), generator=generator)
    region_error = float(
        torch.max(
            torch.abs(
                regional_transport_sums(local).sum(dim=-1)
                - global_transport_sum(local)
            )
        )
    )
    k0_error = float(
        torch.max(torch.abs(global_sum_from_k0(local) - global_transport_sum(local)))
    )
    gates = {
        "energy_truth_like_zero": abs(float(energy_exact.fair)) <= 1.0e-12,
        "energy_bias_positive": float(energy_biased.fair) > 0.0,
        "crps_truth_like_zero": abs(float(crps_exact.fair)) <= 1.0e-12,
        "crps_bias_positive": float(crps_biased.fair) > 0.0,
        "regional_partition_count": 12,
        "regional_sum_maximum_absolute_error": region_error,
        "regional_sum_passed": region_error <= 2.0e-4,
        "fourier_k0_maximum_absolute_error": k0_error,
        "fourier_k0_passed": k0_error <= 2.0e-4,
        "low_physical_modes": list(PGL_HIERARCHICAL_LOW_N),
        "transport_physical_modes": list(PGL_HIERARCHICAL_TRANSPORT_N),
        "mode_mapping_passed": (
            PGL_HIERARCHICAL_LOW_N == (5, 10, 15)
            and PGL_HIERARCHICAL_TRANSPORT_N == (20, 25, 30, 35)
        ),
    }
    gates["passed"] = bool(
        all(
            value
            for name, value in gates.items()
            if name.endswith("_passed")
            or name in (
                "energy_truth_like_zero",
                "energy_bias_positive",
                "crps_truth_like_zero",
                "crps_bias_positive",
            )
        )
    )
    return gates


@torch.no_grad()
def compute_controls(
    *,
    mean_model: nn.Module,
    edm: nn.Module,
    transport: TorchSeparatrixTransport,
    dataset: Any,
    native_truth: np.ndarray,
    spatial_bank: Any,
    temporal_bank: Any,
    device: torch.device,
) -> tuple[HierarchicalControlMagnitudes, dict[str, Any]]:
    names = (
        "local_spatial",
        "local_temporal",
        "regional",
        "fourier_low",
        "fourier_transport_band",
        "global_crps",
    )
    accumulated = {
        name: {quantity: [] for quantity in PGL_TRANSPORT_QUANTITIES}
        for name in names
    }
    keys: list[dict[str, Any]] = []
    dataset.set_epoch(0)
    mean_model.eval()
    edm.eval()
    for current in PGL_HIERARCHICAL_CONTROL_STARTS:
        item = dataset[current]
        context, targets, observed = tensor_window(item, device)
        if observed != current:
            raise ValueError("hierarchical control frame mapping differs")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mean = mean_forecast_trajectory(mean_model, context).float()
            initial, seeds = keyed_sampler_initial_noise(
                model=edm,
                reference=mean,
                epoch_zero_based=0,
                current_frame=current,
            )
            normalized = edm.sample_normalized(
                context[:, -1], mean, initial, steps=18
            )
            members = edm.compose_fields(mean, normalized).float()
        truth = training_transport_window(
            native_truth,
            current_frame=current,
            model_roll=int(item["toroidal_roll"]),
        )[None].to(device=device, dtype=torch.float32)
        terms = score_hierarchical_terms(
            mean_loss=torch.zeros((), device=device),
            edm_loss=torch.zeros((), device=device),
            members=members,
            transport=transport,
            transport_truth=truth,
            spatial_bank=spatial_bank,
            temporal_bank=temporal_bank,
        )
        for component in names:
            values = getattr(terms.scores, component)
            for index, quantity in enumerate(PGL_TRANSPORT_QUANTITIES):
                accumulated[component][quantity].append(float(values[index].cpu()))
        keys.append(
            {
                "current_frame": current,
                "toroidal_roll": int(item["toroidal_roll"]),
                "sampler_seeds": list(seeds),
            }
        )

    def values(component: str) -> tuple[float, ...]:
        return tuple(
            float(np.mean(accumulated[component][quantity]))
            for quantity in PGL_TRANSPORT_QUANTITIES
        )

    controls = HierarchicalControlMagnitudes(
        local_spatial=values("local_spatial"),
        local_temporal=values("local_temporal"),
        regional=values("regional"),
        fourier_low=values("fourier_low"),
        fourier_transport_band=values("fourier_transport_band"),
        global_crps=values("global_crps"),
    )
    return controls, {
        "per_control_start_keys": keys,
        "arithmetic_mean_of_fair_scores": True,
    }


def _gradient_norm(gradients: Sequence[Tensor]) -> float:
    squared = sum(float(value.double().square().sum()) for value in gradients)
    return math.sqrt(squared)


def _dot(first: Sequence[Tensor], second: Sequence[Tensor]) -> float:
    return sum(float(torch.sum(a.double() * b.double())) for a, b in zip(first, second))


def _gradient_summary(
    gradients: Sequence[Tensor],
    branches: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    offset = 0
    for name, values in branches.items():
        selected = gradients[offset : offset + len(values)]
        offset += len(values)
        norm = _gradient_norm(selected)
        learning_rate = (
            PGL_HIERARCHICAL_MEAN_LR
            if name == "mean"
            else PGL_HIERARCHICAL_STOCHASTIC_LR
        )
        result[name] = {
            "gradient_norm": norm,
            "learning_rate_scaled_norm": learning_rate * norm,
            "parameter_count": sum(parameter.numel() for _, parameter in values),
            "nonzero_parameter_tensors": sum(
                int(torch.count_nonzero(value) > 0) for value in selected
            ),
        }
    if offset != len(gradients):
        raise AssertionError("gradient branch coverage differs")
    return result


def calibrate_gradient_weight(
    *,
    mean_model: nn.Module,
    edm: nn.Module,
    transport: TorchSeparatrixTransport,
    dataset: Any,
    derivative_rms: Tensor,
    native_truth: np.ndarray,
    spatial_bank: Any,
    temporal_bank: Any,
    controls: HierarchicalControlMagnitudes,
    device: torch.device,
) -> dict[str, Any]:
    mean_model.train()
    edm.train()
    branches = parameter_branches(mean_model, edm)
    parameters = [parameter for values in branches.values() for _, parameter in values]
    totals: dict[str, list[Tensor]] = {
        name: [torch.zeros_like(parameter, device="cpu", dtype=torch.float64) for parameter in parameters]
        for name in ("original", "local", "regional", "global")
    }
    loss_values = {name: [] for name in totals}
    key_rows: list[dict[str, Any]] = []
    dataset.set_epoch(0)
    for current in PGL_HIERARCHICAL_GRADIENT_STARTS:
        item = dataset[current]
        context, targets, observed = tensor_window(item, device)
        if observed != current:
            raise ValueError("hierarchical gradient frame mapping differs")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mean = mean_forecast_trajectory(mean_model, context).float()
            mean_loss, _ = weighted_mean_state_loss(mean, targets, derivative_rms)
            clean = edm.normalize_residual(targets - mean.detach())
            sigma, noise, noise_seed = keyed_sigma_and_noise(
                base_seed=PGL_NOISE_BASE_SEED,
                epoch_zero_based=0,
                current_frame=current,
                probe=0,
                reference=clean,
                noise_config=edm.noise_config,
            )
            edm_loss = edm.training_loss(
                clean,
                context[:, -1],
                mean.detach(),
                sigma=sigma,
                noise=noise,
            ).loss
            initial, member_seeds = keyed_sampler_initial_noise(
                model=edm,
                reference=mean,
                epoch_zero_based=0,
                current_frame=current,
            )
            normalized = differentiable_sample_normalized(
                edm,
                context[:, -1],
                mean,
                initial,
                steps=18,
                activation_checkpointing=True,
            )
            members = edm.compose_fields(mean, normalized).float()
        truth = training_transport_window(
            native_truth,
            current_frame=current,
            model_roll=int(item["toroidal_roll"]),
        )[None].to(device=device, dtype=torch.float32)
        terms = score_hierarchical_terms(
            mean_loss=mean_loss,
            edm_loss=edm_loss,
            members=members,
            transport=transport,
            transport_truth=truth,
            spatial_bank=spatial_bank,
            temporal_bank=temporal_bank,
        )
        local, regional, global_score = controls.normalize(terms.scores)
        losses = {
            "original": terms.original,
            "local": local,
            "regional": regional,
            "global": global_score,
        }
        for loss_index, (name, loss) in enumerate(losses.items()):
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=loss_index < len(losses) - 1,
                allow_unused=True,
                create_graph=False,
            )
            for index, gradient in enumerate(gradients):
                if gradient is not None:
                    totals[name][index].add_(gradient.detach().double().cpu())
            loss_values[name].append(float(loss.detach().cpu()))
        key_rows.append(
            {
                "current_frame": current,
                "toroidal_roll": int(item["toroidal_roll"]),
                "edm_noise_seed": int(noise_seed),
                "sampler_seeds": list(member_seeds),
            }
        )
        del members, normalized, terms, losses, mean
        torch.cuda.empty_cache()

    count = float(len(PGL_HIERARCHICAL_GRADIENT_STARTS))
    averaged = {
        name: [value / count for value in values] for name, values in totals.items()
    }
    averaged["auxiliary_unscaled"] = [
        averaged["local"][index]
        + averaged["regional"][index]
        + averaged["global"][index]
        for index in range(len(parameters))
    ]
    original_norm = _gradient_norm(averaged["original"])
    auxiliary_norm = _gradient_norm(averaged["auxiliary_unscaled"])
    if not math.isfinite(original_norm) or not math.isfinite(auxiliary_norm) or min(
        original_norm, auxiliary_norm
    ) <= 0.0:
        raise FloatingPointError("hierarchical calibration gradients are invalid")
    multiplier = PGL_HIERARCHICAL_TARGET_GRADIENT_RATIO * original_norm / auxiliary_norm
    averaged["auxiliary_scaled"] = [
        multiplier * value for value in averaged["auxiliary_unscaled"]
    ]
    observed_ratio = _gradient_norm(averaged["auxiliary_scaled"]) / original_norm
    summaries = {
        name: {
            "total_gradient_norm": _gradient_norm(values),
            "branches": _gradient_summary(values, branches),
        }
        for name, values in averaged.items()
    }
    names = list(averaged)
    cosines: dict[str, float] = {}
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            denominator = (
                summaries[first]["total_gradient_norm"]
                * summaries[second]["total_gradient_norm"]
            )
            cosines[f"{first}__{second}"] = (
                _dot(averaged[first], averaged[second]) / denominator
                if denominator > 0.0
                else 0.0
            )
    passed = bool(
        math.isfinite(multiplier)
        and multiplier > 0.0
        and abs(observed_ratio - 0.25) <= 1.0e-4
        and all(
            summaries[name]["total_gradient_norm"] > 0.0
            for name in ("original", "local", "regional", "global")
        )
    )
    return {
        "schema_version": 1,
        "scope": "old_85604_pgl_hierarchical_gradient_calibration",
        "status": "passed" if passed else "failed",
        "development_run": "85604",
        "current_frames": list(PGL_HIERARCHICAL_GRADIENT_STARTS),
        "loss_arithmetic_means": {
            name: float(np.mean(values)) for name, values in loss_values.items()
        },
        "target_ratio": PGL_HIERARCHICAL_TARGET_GRADIENT_RATIO,
        "observed_ratio": observed_ratio,
        "ratio_tolerance": 1.0e-4,
        "auxiliary_lambda": multiplier,
        "gradient_summaries": summaries,
        "cosine_similarity": cosines,
        "key_audit": key_rows,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }


def main() -> int:
    args = parse_args()
    for path in (
        args.artifact_root,
        args.evidence_manifest,
        args.protocol,
        args.prior_preflight,
        args.output,
        args.paper0_root,
    ):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_checkout(args.paper0_root, args.paper0_commit)
    if args.protocol.name != PROTOCOL_NAME or sha256_path(args.protocol) != args.protocol_sha256:
        raise ValueError("hierarchical protocol identity or SHA-256 differs")
    prior, prior_paths = authorize_prior_preflight(
        args.prior_preflight, args.prior_preflight_sha256
    )
    manifest = load_strict_json(args.evidence_manifest)
    paths = authorize_pgl_evaluation_manifest(
        manifest,
        manifest_path=args.evidence_manifest,
        manifest_sha256=args.evidence_manifest_sha256,
        paper0_root=args.paper0_root,
    )
    if paths["model_dataset"] != args.artifact_root.resolve(strict=True):
        raise ValueError("hierarchical preflight model-dataset root differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("hierarchical preflight requires one allocated GPU")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    args.output.mkdir(parents=True)
    started = time.perf_counter()

    selected_mean, _, model, provenance = load_selected_models(
        paths=paths, artifact_root=args.artifact_root, device=device
    )
    if provenance["selected_checkpoint"]["sha256"] != PARENT_CHECKPOINT_SHA256:
        raise ValueError("hierarchical parent checkpoint differs")
    training_manifest = load_strict_json(paths["training_manifest"])
    _, _, derivative_rms = load_parent(training_manifest, device=device)
    catalog = load_official_catalog(args.artifact_root)
    geometry = load_transport_geometry(
        geometry_path=paths["geometry"],
        geometry_manifest=load_strict_json(paths["geometry_manifest"]),
    )
    transport = TorchSeparatrixTransport(
        geometry, decoder_records_from_normalization(catalog.normalization)
    ).to(device, torch.float32).eval().requires_grad_(False)
    banks = load_pair_banks(
        prior_paths["pair_banks"], expected_sha256=prior["pair_banks"]["sha256"]
    )
    native_truth = load_training_transport_truth(
        prior_paths["native_transport_truth"],
        expected_sha256=prior["native_transport_truth"]["sha256"],
    )
    dataset = AutoregressiveStateWindowDataset(
        catalog,
        family="c5p",
        split="train",
        horizon=4,
        augment=True,
        seed=1702,
    )
    try:
        known = hierarchy_known_answer_gates()
        if not known["passed"]:
            raise RuntimeError("hierarchical known-answer gates failed")
        controls, control_keys = compute_controls(
            mean_model=selected_mean,
            edm=model,
            transport=transport,
            dataset=dataset,
            native_truth=native_truth,
            spatial_bank=banks["transport_spatial"],
            temporal_bank=banks["transport_temporal"],
            device=device,
        )
        control_record = controls.to_record()
        control_record.update(
            {
                "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
                "prior_preflight_sha256": args.prior_preflight_sha256,
                "pair_banks_sha256": prior["pair_banks"]["sha256"],
                "native_transport_truth_sha256": prior["native_transport_truth"]["sha256"],
                "key_audit": control_keys,
            }
        )
        write_strict_json_atomic(args.output / "control_magnitudes.json", control_record)
        gradient = calibrate_gradient_weight(
            mean_model=selected_mean,
            edm=model,
            transport=transport,
            dataset=dataset,
            derivative_rms=derivative_rms,
            native_truth=native_truth,
            spatial_bank=banks["transport_spatial"],
            temporal_bank=banks["transport_temporal"],
            controls=controls,
            device=device,
        )
        if gradient["status"] != "passed":
            raise RuntimeError("hierarchical gradient calibration failed")
        write_strict_json_atomic(args.output / "gradient_calibration.json", gradient)
    finally:
        dataset.close()

    result = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_pgl_hierarchical_transport_preflight",
        "status": "passed",
        "development_run": "85604",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "protocol": {"path": str(args.protocol), "sha256": args.protocol_sha256},
        "prior_preflight": {
            "path": str(args.prior_preflight),
            "sha256": args.prior_preflight_sha256,
            "sampler_regression_passed": True,
            "transport_equivalence_passed": True,
        },
        "parent": provenance,
        "known_answer_gates": known,
        "pair_banks": prior["pair_banks"],
        "native_transport_truth": prior["native_transport_truth"],
        "control_magnitudes": {
            "path": str(args.output / "control_magnitudes.json"),
            "sha256": sha256_path(args.output / "control_magnitudes.json"),
        },
        "gradient_calibration": {
            "path": str(args.output / "gradient_calibration.json"),
            "sha256": sha256_path(args.output / "gradient_calibration.json"),
            "auxiliary_lambda": gradient["auxiliary_lambda"],
            "observed_ratio": gradient["observed_ratio"],
        },
        "production_smoke_required_next": list(("CONTROL", "TRANSPORT")),
        "screen_training_authorized": False,
        "peak_cuda_memory_GiB": float(torch.cuda.max_memory_allocated(device) / 2**30),
        "wall_seconds": float(time.perf_counter() - started),
        "gpu": torch.cuda.get_device_name(device),
        "physics_derived_training_loss_used": False,
        "future_truth_used_by_sampler": False,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
    write_strict_json_atomic(args.output / "preflight.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "auxiliary_lambda": gradient["auxiliary_lambda"],
                "gradient_ratio": gradient["observed_ratio"],
                "known_answer_gates": known["passed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
