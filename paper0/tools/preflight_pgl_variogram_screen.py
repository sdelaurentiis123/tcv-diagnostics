#!/usr/bin/env python3
"""Hash-close data, scores, sampler, and transport for the PGL variogram screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from paper0.tools.generate_persistent_global_local_forecast import load_selected_models
from tcv_diagnostics.b5_covariance_localization import exact_separatrix_local_contributions
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.codec_transport import (
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from tcv_diagnostics.matched_o1_transport import NativeTruthCatalog, load_transport_geometry
from tcv_diagnostics.model_data import assert_development_path, load_strict_json, write_strict_json_atomic
from tcv_diagnostics.model_training_data import CodecFrameDataset, load_official_catalog
from tcv_diagnostics.persistent_global_local_authority import authorize_pgl_evaluation_manifest
from tcv_diagnostics.persistent_global_local_training import (
    PGL_NOISE_BASE_SEED,
    keyed_sigma_and_noise,
    mean_forecast_trajectory,
    tensor_window,
)
from tcv_diagnostics.autoregressive_training import AutoregressiveStateWindowDataset
from tcv_diagnostics.pgl_torch_transport import (
    PGL_TRANSPORT_QUANTITIES,
    TorchSeparatrixTransport,
    apply_periodic_resample,
    decoder_records_from_normalization,
    periodic_resample_matrix,
    resample_matrix_sha256,
)
from tcv_diagnostics.pgl_variogram import (
    IndexedPairBank,
    build_spatial_pair_bank,
    build_temporal_pair_bank,
    differentiable_sample_normalized,
    fair_variogram_score,
)
from tcv_diagnostics.pgl_variogram_training import (
    PGL_VARIOGRAM_CONTROL_STARTS,
    VariogramControlMagnitudes,
    keyed_sampler_initial_noise,
    save_pair_banks,
    save_training_transport_truth,
    score_variogram_terms,
    training_transport_window,
)
from tcv_diagnostics.resampling import periodic_resample_float32


PARENT_CHECKPOINT_SHA256 = "4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb"
EXPECTED_PROTOCOL_NAMES = (
    "POST_ECRD_OLD_85604_PGL_BOUNDED_VARIOGRAM_FINETUNE_2026-08-28.md",
    "POST_ECRD_OLD_85604_PGL_BOUNDED_VARIOGRAM_FINETUNE_AMENDMENT_2026-08-28.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest-sha256", required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--base-protocol-sha256", required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def verify_checkout(root: Path, expected: str) -> None:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if commit != str(expected) or dirty:
        raise RuntimeError("variogram preflight requires the locked clean checkout")


def verify_protocol(path: Path, expected_name: str, digest: str) -> dict[str, str]:
    if path.name != expected_name or sha256_path(path) != str(digest):
        raise ValueError("variogram protocol path or SHA-256 differs")
    return {"path": str(path), "sha256": str(digest)}


def _geometry_positions(
    geometry_path: Path,
    geometry_manifest: Mapping[str, Any],
    geometry: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = geometry_manifest["grid"]
    offset = int(grid["model_x_to_grid_x_offset"])
    n_x, n_y = map(int, grid["model_shape"])
    rows = np.arange(8, 24, dtype=np.int64)
    with h5py.File(geometry_path, "r") as handle:
        r_cell = np.asarray(handle["Rxy"][offset : offset + n_x], dtype=np.float64)
        z_cell = np.asarray(handle["Zxy"][offset : offset + n_x], dtype=np.float64)
        face = int(grid["topology"]["ixseps1"])
        r_face = np.asarray(handle["Rxy_xlow"][face, rows], dtype=np.float64)
        z_face = np.asarray(handle["Zxy_xlow"][face, rows], dtype=np.float64)
    if r_cell.shape != (64, 32) or z_cell.shape != (64, 32):
        raise ValueError("field-position geometry crop differs")
    phi88 = 2.0 * np.pi * np.arange(88, dtype=np.float64) / (5.0 * 88.0)
    r88 = np.broadcast_to(r_cell[..., None], (64, 32, 88))
    zz88 = np.broadcast_to(z_cell[..., None], (64, 32, 88))
    p88 = np.broadcast_to(phi88, (64, 32, 88))
    field_positions = np.stack((r88, zz88, p88), axis=-1).reshape(-1, 3)
    strict = geometry.region_masks.strict_wall_interior & geometry.region_masks.operator_interior
    field_eligible = np.flatnonzero(np.broadcast_to(strict[..., None], (64, 32, 88)).reshape(-1))

    phi81 = 2.0 * np.pi * np.arange(81, dtype=np.float64) / (5.0 * 81.0)
    r81 = np.broadcast_to(r_face[:, None], (16, 81))
    zz81 = np.broadcast_to(z_face[:, None], (16, 81))
    p81 = np.broadcast_to(phi81, (16, 81))
    transport_positions = np.stack((r81, zz81, p81), axis=-1).reshape(-1, 3)
    transport_eligible = np.arange(16 * 81, dtype=np.int64)
    if field_eligible.size < 1024 or transport_eligible.size < 1024:
        raise ValueError("variogram eligible geometry is too small")
    return field_positions, field_eligible, transport_positions, transport_eligible


def build_pair_banks_from_geometry(
    field_positions: np.ndarray,
    field_eligible: np.ndarray,
    transport_positions: np.ndarray,
    transport_eligible: np.ndarray,
) -> dict[str, IndexedPairBank]:
    return {
        "field_spatial": build_spatial_pair_bank(
            field_positions, field_eligible, future_times=4, variables=5
        ),
        "field_temporal": build_temporal_pair_bank(
            field_eligible, cells=64 * 32 * 88, trajectory_times=5, variables=5
        ),
        "transport_spatial": build_spatial_pair_bank(
            transport_positions,
            transport_eligible,
            future_times=4,
            variables=1,
        ),
        "transport_temporal": build_temporal_pair_bank(
            transport_eligible,
            cells=16 * 81,
            trajectory_times=5,
            variables=1,
        ),
    }


def known_answer_gates() -> dict[str, Any]:
    bank = IndexedPairBank(
        left=np.asarray([0, 2], dtype=np.int64),
        right=np.asarray([1, 3], dtype=np.int64),
        weight=np.asarray([0.5, 0.5], dtype=np.float64),
        group=np.asarray([0, 1], dtype=np.int64),
        group_name="synthetic_pair",
        group_values=(1.0, 2.0),
        metadata={"scope": "preflight_known_answer"},
    )
    truth = torch.tensor([[0.0, 2.0, 1.0, 5.0]])
    exact = truth[:, None].repeat(1, 4, 1)
    exact_result = fair_variogram_score(exact, truth, bank)
    shuffled = exact[..., torch.tensor([2, 0, 3, 1])]
    shuffled_result = fair_variogram_score(shuffled, truth, bank)
    biased_result = fair_variogram_score(exact + 7.0, truth, bank)
    perturbed = (exact + 0.1 * torch.randn_like(exact)).requires_grad_(True)
    gradient_result = fair_variogram_score(perturbed, truth, bank)
    gradient_result.fair.backward()
    identity_error = float(
        torch.abs(
            gradient_result.ordinary
            - gradient_result.fair
            - gradient_result.finite_member_correction
        )
    )
    gates = {
        "truth_like_zero": abs(float(exact_result.fair)) <= 1.0e-12,
        "spatial_shuffle_positive": float(shuffled_result.fair) > 0.0,
        "constant_bias_invariant": abs(float(biased_result.fair)) <= 1.0e-12,
        "fair_correction_identity_error": identity_error,
        "fair_correction_identity_passed": identity_error <= 2.0e-7,
        "finite_nonzero_gradient": bool(
            perturbed.grad is not None
            and torch.isfinite(perturbed.grad).all()
            and torch.count_nonzero(perturbed.grad) > 0
        ),
    }
    # A time permutation is a pair-index permutation on a nondegenerate path.
    temporal_truth = torch.tensor([[0.0, 1.0, 4.0, 9.0]])
    temporal_shuffled = temporal_truth[:, None, torch.tensor([0, 2, 1, 3])].repeat(1, 4, 1)
    gates["temporal_shuffle_positive"] = bool(
        fair_variogram_score(temporal_shuffled, temporal_truth, bank).fair > 0
    )
    gates["passed"] = bool(
        all(value for name, value in gates.items() if name not in ("fair_correction_identity_error", "passed"))
    )
    return gates


def precompute_native_transport(native_truth: NativeTruthCatalog, geometry: Any) -> tuple[np.ndarray, float]:
    result = np.empty((432, 4, 16, 81), dtype=np.float64)
    maximum_closure = 0.0
    for start in range(0, 432, 24):
        stop = min(start + 24, 432)
        native = native_truth.read(start, stop, fields=("Ne", "Pe", "Pi", "phi"))
        evaluated = evaluate_transport_state(
            direct_pressure_transport_state(native["Ne"], native["Pe"], native["Pi"], native["phi"]),
            geometry,
        )
        local, closure = exact_separatrix_local_contributions(
            evaluated,
            strict_face_mask=geometry.strict_face_mask,
            separatrix_face_mask=geometry.separatrix_face_mask,
        )
        result[start:stop] = np.stack(
            [local[name] for name in PGL_TRANSPORT_QUANTITIES], axis=1
        )
        maximum_closure = max(maximum_closure, float(closure))
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("native training transport artifact is non-finite")
    return result, maximum_closure


def transport_equivalence_gates(
    *,
    catalog: Any,
    geometry: Any,
    operator: TorchSeparatrixTransport,
    device: torch.device,
) -> dict[str, Any]:
    matrix = periodic_resample_matrix(88, 81)
    rng = np.random.default_rng(856040829)
    probes = rng.normal(size=(8, 88)).astype(np.float32)
    torch_resampled = apply_periodic_resample(
        torch.from_numpy(probes).to(device), operator.resample
    ).cpu().numpy()
    scipy_resampled = periodic_resample_float32(probes, 81, axis=-1)
    resample_max = float(np.max(np.abs(torch_resampled - scipy_resampled)))

    frames = (0, 143, 286, 431)
    dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split="train",
        frames=range(0, 432),
        augment=False,
        seed=1702,
        return_physical=True,
    )
    records: list[dict[str, float | int]] = []
    maximum_closure = 0.0
    try:
        for frame in frames:
            item = dataset[frame]
            standardized = torch.from_numpy(item["volume"])[None, None, None].to(
                device=device, dtype=torch.float32
            )
            observed = operator(standardized)[0, 0, 0].detach().cpu().numpy()
            physical = np.asarray(item["physical_volume"], dtype=np.float32)
            native = periodic_resample_float32(physical[:4], 81, axis=-1).astype(np.float64)
            evaluated = evaluate_transport_state(
                direct_pressure_transport_state(native[0:1], native[1:2], native[2:3], native[3:4]),
                geometry,
            )
            local, closure = exact_separatrix_local_contributions(
                evaluated,
                strict_face_mask=geometry.strict_face_mask,
                separatrix_face_mask=geometry.separatrix_face_mask,
            )
            expected = np.stack([local[name][0] for name in PGL_TRANSPORT_QUANTITIES], axis=0)
            difference = observed.astype(np.float64) - expected
            relative = float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1.0e-30))
            integrated_expected = np.sum(expected, axis=(1, 2))
            integrated_observed = np.sum(observed, axis=(1, 2), dtype=np.float64)
            integrated_relative = float(
                np.linalg.norm(integrated_observed - integrated_expected)
                / max(np.linalg.norm(integrated_expected), 1.0e-30)
            )
            records.append(
                {
                    "frame": frame,
                    "relative_L2": relative,
                    "integrated_relative_error": integrated_relative,
                    "maximum_absolute_error": float(np.max(np.abs(difference))),
                }
            )
            maximum_closure = max(maximum_closure, float(closure))
    finally:
        dataset.close()
    maximum_l2 = max(float(value["relative_L2"]) for value in records)
    maximum_integrated = max(float(value["integrated_relative_error"]) for value in records)
    return {
        "resample_matrix_sha256": resample_matrix_sha256(matrix),
        "resample_torch_vs_scipy_maximum_absolute_error": resample_max,
        "authorized_training_frames": records,
        "maximum_relative_L2": maximum_l2,
        "maximum_integrated_relative_error": maximum_integrated,
        "float64_authority_separatrix_closure": maximum_closure,
        "threshold_relative_L2": 2.0e-5,
        "threshold_integrated_relative_error": 2.0e-5,
        "threshold_float64_authority_closure": 2.0e-10,
        "passed": bool(
            resample_max <= 2.0e-6
            and maximum_l2 <= 2.0e-5
            and maximum_integrated <= 2.0e-5
            and maximum_closure <= 2.0e-10
        ),
    }


def sampler_regression_gate(
    *,
    mean_model: torch.nn.Module,
    model: Any,
    dataset: Any,
    device: torch.device,
) -> dict[str, Any]:
    item = dataset[0]
    context, _, current = tensor_window(item, device)
    with torch.no_grad():
        mean = mean_forecast_trajectory(mean_model, context)
        initial, seeds = keyed_sampler_initial_noise(
            model=model, reference=mean, epoch_zero_based=0, current_frame=current
        )
        expected = model.sample_normalized(context[:, -1], mean, initial, steps=18)
        observed = differentiable_sample_normalized(
            model,
            context[:, -1],
            mean,
            initial,
            steps=18,
            activation_checkpointing=True,
        )
    difference = (observed - expected).float()
    relative = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected.float()).clamp_min(1.0e-30)
    )
    maximum = float(torch.max(torch.abs(difference)))
    return {
        "current_frame": current,
        "member_seeds": list(seeds),
        "members": 4,
        "steps": 18,
        "network_evaluations_per_member": 35,
        "relative_L2": relative,
        "maximum_absolute_error": maximum,
        "relative_L2_tolerance": 2.0e-6,
        "maximum_absolute_tolerance": 2.0e-5,
        "activation_checkpointing": True,
        "passed": bool(relative <= 2.0e-6 and maximum <= 2.0e-5),
    }


@torch.no_grad()
def compute_controls(
    *,
    mean_model: torch.nn.Module,
    model: Any,
    transport: TorchSeparatrixTransport,
    dataset: Any,
    native_transport: np.ndarray,
    pair_banks: Mapping[str, IndexedPairBank],
    device: torch.device,
) -> tuple[VariogramControlMagnitudes, dict[str, Any]]:
    accumulators: dict[str, list[float]] = {
        "edm": [],
        "field_spatial": [],
        "field_temporal": [],
    }
    for name in PGL_TRANSPORT_QUANTITIES:
        accumulators[f"transport_spatial/{name}"] = []
        accumulators[f"transport_temporal/{name}"] = []
    dataset.set_epoch(0)
    rows: list[dict[str, Any]] = []
    for current in PGL_VARIOGRAM_CONTROL_STARTS:
        item = dataset[current]
        context, targets, observed_current = tensor_window(item, device)
        if observed_current != current:
            raise ValueError("control dataset index/current-frame mapping differs")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mean = mean_forecast_trajectory(mean_model, context).float()
        clean = model.normalize_residual(targets - mean)
        sigma, noise, _ = keyed_sigma_and_noise(
            base_seed=PGL_NOISE_BASE_SEED,
            epoch_zero_based=0,
            current_frame=current,
            probe=0,
            reference=clean,
            noise_config=model.noise_config,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            edm = model.training_loss(
                clean, context[:, -1], mean, sigma=sigma, noise=noise
            ).loss
            initial_noise, seeds = keyed_sampler_initial_noise(
                model=model, reference=mean, epoch_zero_based=0, current_frame=current
            )
            normalized = model.sample_normalized(
                context[:, -1], mean, initial_noise, steps=18
            )
            members = model.compose_fields(mean, normalized).float()
        truth = training_transport_window(
            native_transport,
            current_frame=current,
            model_roll=int(item["toroidal_roll"]),
        )[None].to(device)
        terms = score_variogram_terms(
            edm_loss=edm,
            members=members,
            truth=targets,
            current=context[:, -1],
            transport=transport,
            transport_truth=truth,
            pair_banks=pair_banks,
        )
        accumulators["edm"].append(float(terms.edm.cpu()))
        accumulators["field_spatial"].append(float(terms.field_spatial.cpu()))
        accumulators["field_temporal"].append(float(terms.field_temporal.cpu()))
        for index, name in enumerate(PGL_TRANSPORT_QUANTITIES):
            accumulators[f"transport_spatial/{name}"].append(
                float(terms.transport_spatial[index].cpu())
            )
            accumulators[f"transport_temporal/{name}"].append(
                float(terms.transport_temporal[index].cpu())
            )
        rows.append({"current_frame": current, "roll": int(item["toroidal_roll"]), "sampler_seeds": list(seeds)})
    means = {name: float(np.mean(values)) for name, values in accumulators.items()}
    controls = VariogramControlMagnitudes(
        edm=means["edm"],
        field_spatial=means["field_spatial"],
        field_temporal=means["field_temporal"],
        transport_spatial=tuple(means[f"transport_spatial/{name}"] for name in PGL_TRANSPORT_QUANTITIES),
        transport_temporal=tuple(means[f"transport_temporal/{name}"] for name in PGL_TRANSPORT_QUANTITIES),
    )
    return controls, {"per_control_start_keys": rows, "arithmetic_mean_of_fair_scores": True}


def main() -> int:
    args = parse_args()
    for path in (
        args.artifact_root,
        args.evidence_manifest,
        args.base_protocol,
        args.amendment,
        args.output,
        args.paper0_root,
    ):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_checkout(args.paper0_root, args.paper0_commit)
    protocols = {
        "base": verify_protocol(args.base_protocol, EXPECTED_PROTOCOL_NAMES[0], args.base_protocol_sha256),
        "amendment": verify_protocol(args.amendment, EXPECTED_PROTOCOL_NAMES[1], args.amendment_sha256),
    }
    manifest = load_strict_json(args.evidence_manifest)
    paths = authorize_pgl_evaluation_manifest(
        manifest,
        manifest_path=args.evidence_manifest,
        manifest_sha256=args.evidence_manifest_sha256,
        paper0_root=args.paper0_root,
    )
    if paths["model_dataset"] != args.artifact_root.resolve(strict=True):
        raise ValueError("variogram preflight model-dataset root differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("variogram preflight requires one allocated CUDA GPU")
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
        raise ValueError("variogram selected parent checkpoint differs")
    selected_mean.eval().requires_grad_(False)
    model.eval()
    catalog = load_official_catalog(args.artifact_root)
    geometry_manifest = load_strict_json(paths["geometry_manifest"])
    geometry = load_transport_geometry(
        geometry_path=paths["geometry"], geometry_manifest=geometry_manifest
    )
    torch_transport = TorchSeparatrixTransport(
        geometry, decoder_records_from_normalization(catalog.normalization)
    ).to(device, torch.float32).eval()
    torch_transport.requires_grad_(False)

    known = known_answer_gates()
    if not known["passed"]:
        raise RuntimeError("variogram known-answer gate failed")
    positions = _geometry_positions(paths["geometry"], geometry_manifest, geometry)
    banks = build_pair_banks_from_geometry(*positions)
    pair_record = save_pair_banks(args.output / "pair_banks.npz", banks)
    if any(
        not np.allclose(
            bank.to_record()["group_total_weights"],
            np.full(6 if "spatial" in name else 4, 1.0 / (6 if "spatial" in name else 4)),
            rtol=0.0,
            atol=1.0e-12,
        )
        for name, bank in banks.items()
    ):
        raise RuntimeError("variogram pair-bank equal-group weighting failed")

    native_catalog = NativeTruthCatalog(load_strict_json(paths["native_truth_result"]))
    native_transport, truth_closure = precompute_native_transport(native_catalog, geometry)
    truth_record = save_training_transport_truth(
        args.output / "native_transport_truth_train_0_432.npz", native_transport
    )
    equivalence = transport_equivalence_gates(
        catalog=catalog, geometry=geometry, operator=torch_transport, device=device
    )
    if not equivalence["passed"] or truth_closure > 2.0e-10:
        raise RuntimeError("differentiable authoritative transport equivalence failed")

    dataset = AutoregressiveStateWindowDataset(
        catalog,
        family="c5p",
        split="train",
        horizon=4,
        augment=True,
        seed=1702,
    )
    try:
        sampler = sampler_regression_gate(
            mean_model=selected_mean, model=model, dataset=dataset, device=device
        )
        if not sampler["passed"]:
            raise RuntimeError("differentiable sampler forward regression failed")
        controls, control_keys = compute_controls(
            mean_model=selected_mean,
            model=model,
            transport=torch_transport,
            dataset=dataset,
            native_transport=native_transport,
            pair_banks=banks,
            device=device,
        )
    finally:
        dataset.close()
    controls_record = controls.to_record()
    controls_record.update(
        {
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "pair_banks_sha256": pair_record["sha256"],
            "native_transport_truth_sha256": truth_record["sha256"],
            "key_audit": control_keys,
        }
    )
    write_strict_json_atomic(args.output / "control_magnitudes.json", controls_record)
    controls_artifact = {
        "path": str(args.output / "control_magnitudes.json"),
        "sha256": sha256_path(args.output / "control_magnitudes.json"),
    }
    result = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_pgl_variogram_preflight",
        "status": "passed",
        "development_run": "85604",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "protocols": protocols,
        "parent": provenance,
        "known_answer_gates": known,
        "sampler_regression_gate": sampler,
        "transport_equivalence_gate": equivalence,
        "native_truth_maximum_separatrix_closure": truth_closure,
        "pair_banks": pair_record,
        "native_transport_truth": truth_record,
        "control_magnitudes": controls_artifact,
        "production_size_optimizer_smoke_required_next": ["A", "B", "C", "D"],
        "screen_training_authorized": False,
        "future_truth_used_by_sampler": False,
        "held_out_85606_read": False,
        "held_out_run_read": False,
        "new_nersc_data_read": False,
        "new_segment_read": False,
        "physics_derived_loss_used": False,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_cuda_memory_GiB": float(torch.cuda.max_memory_allocated(device) / 2**30),
        "gpu": torch.cuda.get_device_name(device),
    }
    write_strict_json_atomic(args.output / "preflight.json", result)
    print(json.dumps({"status": result["status"], "artifacts": {"pair_banks": pair_record, "transport_truth": truth_record, "controls": controls_artifact}, "gates": {"known": known["passed"], "sampler": sampler["passed"], "transport": equivalence["passed"]}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
