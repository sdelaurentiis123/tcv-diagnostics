"""Bounded warm-start training mechanics for the PGL variogram screen.

This module implements only the prospective 2026-08-28 amendment.  It never
discovers data or checkpoints and contains no held-out-run path.  Entrypoints
must supply hash-closed 85604 evidence and the preflight artifacts.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from .b5_residual_edm_full_training import update_ema_model
from .b5_residual_edm_training import module_state_sha256, parameter_count
from .codec_training import save_torch_atomic, sha256_path
from .model_data import assert_development_path, write_strict_json_atomic
from .models.persistent_global_local import PersistentGlobalLocalEDM
from .persistent_global_local_forecast import initial_noise_from_uint64
from .persistent_global_local_training import (
    PGL_HORIZON,
    PGL_NOISE_BASE_SEED,
    PGL_SEED,
    mean_forecast_trajectory,
    keyed_sigma_and_noise,
    tensor_window,
)
from .pgl_torch_transport import PGL_TRANSPORT_QUANTITIES, TorchSeparatrixTransport
from .pgl_variogram import (
    IndexedPairBank,
    differentiable_sample_normalized,
    fair_variogram_score,
    gauge_fix_phi,
    prepend_observed_current,
)


PGL_VARIOGRAM_ARMS = ("A", "B", "C", "D")
PGL_VARIOGRAM_CONTROL_STARTS = tuple(
    int(value) for value in np.floor(np.linspace(0, 427, 32)).astype(np.int64)
)
PGL_VARIOGRAM_LAMBDA = 0.10
PGL_VARIOGRAM_SAMPLER_MEMBERS = 4
PGL_VARIOGRAM_SAMPLER_STEPS = 18
PGL_VARIOGRAM_EXPECTED_WINDOWS = 428
PGL_VARIOGRAM_EXPECTED_UPDATES = 214
PGL_VARIOGRAM_LR = 1.0e-6
PGL_VARIOGRAM_CONTROL_SCHEMA = 1
_PAIR_BANK_NAMES = (
    "field_spatial",
    "field_temporal",
    "transport_spatial",
    "transport_temporal",
)


class _WindowDataset(Protocol):
    split: str
    horizon: int

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...

    def set_epoch(self, epoch: int) -> None: ...


@dataclass(frozen=True)
class VariogramScreenConfig:
    """Frozen one-update smoke or one-epoch screen budget."""

    mode: str
    arm: str
    seed: int = PGL_SEED
    learning_rate: float = PGL_VARIOGRAM_LR
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    accumulation_windows: int = 2
    sampler_members: int = PGL_VARIOGRAM_SAMPLER_MEMBERS
    sampler_steps: int = PGL_VARIOGRAM_SAMPLER_STEPS
    auxiliary_lambda: float = PGL_VARIOGRAM_LAMBDA

    def __post_init__(self) -> None:
        if self.mode not in ("smoke", "screen"):
            raise ValueError("variogram mode must be smoke or screen")
        if self.arm not in PGL_VARIOGRAM_ARMS:
            raise ValueError("variogram arm must be A, B, C, or D")
        expected = {
            "seed": 1702,
            "learning_rate": 1.0e-6,
            "betas": (0.9, 0.99),
            "weight_decay": 1.0e-4,
            "gradient_clip": 1.0,
            "ema_decay": 0.999,
            "accumulation_windows": 2,
            "sampler_members": 4,
            "sampler_steps": 18,
            "auxiliary_lambda": 0.10,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("variogram optimization contract differs")

    @property
    def optimizer_updates(self) -> int:
        return 1 if self.mode == "smoke" else PGL_VARIOGRAM_EXPECTED_UPDATES

    @property
    def training_windows(self) -> int:
        return 2 if self.mode == "smoke" else PGL_VARIOGRAM_EXPECTED_WINDOWS

    @property
    def field_variogram_enabled(self) -> bool:
        return self.arm in ("B", "D")

    @property
    def transport_variogram_enabled(self) -> bool:
        return self.arm in ("C", "D")

    @property
    def physics_derived_training_loss_used(self) -> bool:
        return self.transport_variogram_enabled

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["betas"] = list(self.betas)
        record.update(
            {
                "epochs": 1,
                "training_windows": self.training_windows,
                "optimizer_updates": self.optimizer_updates,
                "constant_learning_rate": True,
                "deterministic_mean_frozen": True,
                "fixed_final_ema_no_checkpoint_selection": True,
                "activation_checkpoint_every_denoiser_evaluation": True,
                "network_evaluations_per_member": 35,
                "future_truth_used_by_sampler": False,
                "held_out_85606_read": False,
                "held_out_run_read": False,
                "new_nersc_data_read": False,
                "new_segment_read": False,
                "physics_derived_training_loss_used": (
                    self.physics_derived_training_loss_used
                ),
            }
        )
        return record


@dataclass(frozen=True)
class VariogramControlMagnitudes:
    edm: float
    field_spatial: float
    field_temporal: float
    transport_spatial: tuple[float, float, float, float]
    transport_temporal: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        values = (
            self.edm,
            self.field_spatial,
            self.field_temporal,
            *self.transport_spatial,
            *self.transport_temporal,
        )
        if len(self.transport_spatial) != 4 or len(self.transport_temporal) != 4:
            raise ValueError("transport controls require all four quantities")
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            raise ValueError("every initial control magnitude must be finite and positive")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "VariogramControlMagnitudes":
        if (
            int(record.get("schema_version", -1)) != PGL_VARIOGRAM_CONTROL_SCHEMA
            or record.get("development_run") != "85604"
            or record.get("current_frames") != list(PGL_VARIOGRAM_CONTROL_STARTS)
            or record.get("sampler_members") != 4
            or record.get("sampler_steps") != 18
            or record.get("fair_order") != 1
            or record.get("future_truth_used_by_sampler") is not False
            or record.get("held_out_85606_read") is not False
            or record.get("held_out_run_read") is not False
            or record.get("new_nersc_data_read") is not False
            or record.get("new_segment_read") is not False
        ):
            raise ValueError("variogram control artifact contract differs")
        spatial = record.get("transport_spatial", {})
        temporal = record.get("transport_temporal", {})
        if set(spatial) != set(PGL_TRANSPORT_QUANTITIES) or set(temporal) != set(PGL_TRANSPORT_QUANTITIES):
            raise ValueError("variogram transport control quantities differ")
        return cls(
            edm=float(record["edm"]),
            field_spatial=float(record["field_spatial"]),
            field_temporal=float(record["field_temporal"]),
            transport_spatial=tuple(float(spatial[name]) for name in PGL_TRANSPORT_QUANTITIES),
            transport_temporal=tuple(float(temporal[name]) for name in PGL_TRANSPORT_QUANTITIES),
        )

    def to_record(self, *, sample_count: int = 32) -> dict[str, Any]:
        if int(sample_count) != len(PGL_VARIOGRAM_CONTROL_STARTS):
            raise ValueError("initial controls require exactly 32 chronological starts")
        return {
            "schema_version": PGL_VARIOGRAM_CONTROL_SCHEMA,
            "scope": "old_85604_pgl_variogram_initial_controls",
            "development_run": "85604",
            "current_frames": list(PGL_VARIOGRAM_CONTROL_STARTS),
            "sample_count": sample_count,
            "edm": float(self.edm),
            "field_spatial": float(self.field_spatial),
            "field_temporal": float(self.field_temporal),
            "transport_spatial": {
                name: float(value)
                for name, value in zip(PGL_TRANSPORT_QUANTITIES, self.transport_spatial)
            },
            "transport_temporal": {
                name: float(value)
                for name, value in zip(PGL_TRANSPORT_QUANTITIES, self.transport_temporal)
            },
            "sampler_members": 4,
            "sampler_steps": 18,
            "fair_order": 1,
            "ordinary_scores_logged_separately": True,
            "future_truth_used_by_sampler": False,
            "held_out_85606_read": False,
            "held_out_run_read": False,
            "new_nersc_data_read": False,
            "new_segment_read": False,
        }


@dataclass(frozen=True)
class VariogramTerms:
    edm: Tensor
    field_spatial: Tensor
    field_temporal: Tensor
    transport_spatial: tuple[Tensor, ...]
    transport_temporal: tuple[Tensor, ...]
    ordinary: Mapping[str, Tensor]

    def normalized_field(self, controls: VariogramControlMagnitudes) -> Tensor:
        return 0.5 * (
            self.field_spatial / controls.field_spatial
            + self.field_temporal / controls.field_temporal
        )

    def normalized_transport(self, controls: VariogramControlMagnitudes) -> Tensor:
        values = []
        for index in range(4):
            values.extend(
                (
                    self.transport_spatial[index] / controls.transport_spatial[index],
                    self.transport_temporal[index] / controls.transport_temporal[index],
                )
            )
        return torch.stack(values).sum() / 8.0


def arm_objective(
    arm: str,
    terms: VariogramTerms,
    controls: VariogramControlMagnitudes,
    *,
    auxiliary_lambda: float = PGL_VARIOGRAM_LAMBDA,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return objective and normalized field/transport diagnostics."""

    if arm not in PGL_VARIOGRAM_ARMS or float(auxiliary_lambda) != 0.10:
        raise ValueError("variogram arm or lambda differs")
    field = terms.normalized_field(controls)
    transport = terms.normalized_transport(controls)
    if arm == "A":
        objective = terms.edm
    elif arm == "B":
        objective = terms.edm + auxiliary_lambda * controls.edm * field
    elif arm == "C":
        objective = terms.edm + auxiliary_lambda * controls.edm * transport
    else:
        objective = terms.edm + 0.5 * auxiliary_lambda * controls.edm * (
            field + transport
        )
    return objective, field, transport


def keyed_sampler_initial_noise(
    *,
    model: PersistentGlobalLocalEDM,
    reference: Tensor,
    epoch_zero_based: int,
    current_frame: int,
    members: int = PGL_VARIOGRAM_SAMPLER_MEMBERS,
) -> tuple[Tensor, tuple[int, ...]]:
    """Create four stateless structured-noise trajectories for one window."""

    if reference.ndim != 6 or reference.shape[0] != 1 or int(members) != 4:
        raise ValueError("variogram sampler reference/member contract differs")
    seeds: list[int] = []
    samples: list[Tensor] = []
    for member in range(members):
        payload = (
            f"{PGL_NOISE_BASE_SEED}:variogram_sampler:{epoch_zero_based}:"
            f"{current_frame}:{member}"
        ).encode("ascii")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        seeds.append(seed)
        samples.append(initial_noise_from_uint64(seed, reference=reference, model=model))
    return torch.stack(samples, dim=0).unsqueeze(0), tuple(seeds)


def fractional_periodic_roll(values: np.ndarray, shift_cells: float) -> np.ndarray:
    """Apply a positive, possibly fractional, NumPy-roll convention in z."""

    array = np.asarray(values)
    if array.ndim < 1 or not np.all(np.isfinite(array)) or not math.isfinite(shift_cells):
        raise ValueError("fractional roll input differs")
    n = array.shape[-1]
    coefficients = np.fft.rfft(array, axis=-1)
    modes = np.arange(coefficients.shape[-1], dtype=np.float64)
    phase = np.exp(-2j * np.pi * modes * float(shift_cells) / n)
    result = np.fft.irfft(coefficients * phase, n=n, axis=-1)
    return np.ascontiguousarray(result, dtype=array.dtype)


def training_transport_window(
    truth_by_frame: np.ndarray,
    *,
    current_frame: int,
    model_roll: int,
    model_z: int = 88,
) -> Tensor:
    """Return augmented native transport at current plus four future frames."""

    truth = np.asarray(truth_by_frame)
    if truth.shape != (432, 4, 16, 81):
        raise ValueError("training native transport truth shape differs")
    if not 0 <= int(current_frame) <= 427 or not 0 <= int(model_roll) < model_z:
        raise ValueError("training transport window identity differs")
    selected = truth[current_frame : current_frame + 5]
    physical_shift_native = float(model_roll) * selected.shape[-1] / float(model_z)
    shifted = fractional_periodic_roll(selected, physical_shift_native)
    return torch.from_numpy(np.ascontiguousarray(shifted, dtype=np.float32))


def _bank_payload(bank: IndexedPairBank) -> dict[str, np.ndarray]:
    return {
        "left": np.ascontiguousarray(bank.left, dtype=np.int64),
        "right": np.ascontiguousarray(bank.right, dtype=np.int64),
        "weight": np.ascontiguousarray(bank.weight, dtype=np.float64),
        "group": np.ascontiguousarray(bank.group, dtype=np.int64),
        "group_values": np.asarray(bank.group_values, dtype=np.float64),
        "group_name": np.asarray(bank.group_name),
        "metadata_json": np.asarray(
            json.dumps(dict(bank.metadata), sort_keys=True, allow_nan=False)
        ),
        "sha256": np.asarray(bank.sha256),
    }


def save_pair_banks(path: Path, banks: Mapping[str, IndexedPairBank]) -> dict[str, Any]:
    """Atomically save the four frozen pair banks without pickle objects."""

    target = Path(path)
    assert_development_path(target)
    if target.exists() or target.with_name(f".{target.name}.partial").exists():
        raise FileExistsError(target)
    if tuple(banks) != _PAIR_BANK_NAMES:
        raise ValueError("variogram pair-bank names or order differ")
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "development_run": np.asarray("85604"),
        "held_out_run_read": np.asarray(False),
        "held_out_85606_read": np.asarray(False),
    }
    for name, bank in banks.items():
        for field, values in _bank_payload(bank).items():
            arrays[f"{name}__{field}"] = values
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    with partial.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    partial.replace(target)
    return {
        "path": str(target),
        "sha256": sha256_path(target),
        "banks": {name: bank.to_record() for name, bank in banks.items()},
    }


def load_pair_banks(path: Path, *, expected_sha256: str) -> dict[str, IndexedPairBank]:
    target = Path(path)
    assert_development_path(target)
    if sha256_path(target) != str(expected_sha256):
        raise ValueError("variogram pair-bank artifact SHA-256 differs")
    result: dict[str, IndexedPairBank] = {}
    with np.load(target, allow_pickle=False) as data:
        if int(data["schema_version"]) != 1 or str(data["development_run"]) != "85604":
            raise ValueError("variogram pair-bank artifact identity differs")
        if bool(data["held_out_run_read"]) or bool(data["held_out_85606_read"]):
            raise ValueError("variogram pair-bank artifact reports held-out access")
        for name in _PAIR_BANK_NAMES:
            bank = IndexedPairBank(
                left=np.asarray(data[f"{name}__left"], dtype=np.int64),
                right=np.asarray(data[f"{name}__right"], dtype=np.int64),
                weight=np.asarray(data[f"{name}__weight"], dtype=np.float64),
                group=np.asarray(data[f"{name}__group"], dtype=np.int64),
                group_name=str(data[f"{name}__group_name"]),
                group_values=tuple(
                    float(value) for value in data[f"{name}__group_values"]
                ),
                metadata=json.loads(str(data[f"{name}__metadata_json"])),
            )
            if bank.sha256 != str(data[f"{name}__sha256"]):
                raise ValueError(f"variogram pair bank {name} hash differs")
            result[name] = bank
    return result


def save_training_transport_truth(path: Path, values: np.ndarray) -> dict[str, Any]:
    """Atomically store NumPy-authority local transport for frames ``[0,432)``."""

    target = Path(path)
    assert_development_path(target)
    truth = np.ascontiguousarray(values, dtype=np.float64)
    if truth.shape != (432, 4, 16, 81) or not np.all(np.isfinite(truth)):
        raise ValueError("native training transport truth shape or values differ")
    if target.exists() or target.with_name(f".{target.name}.partial").exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    with partial.open("xb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(1, dtype=np.int64),
            development_run=np.asarray("85604"),
            frame_index=np.arange(432, dtype=np.int64),
            quantities=np.asarray(PGL_TRANSPORT_QUANTITIES),
            local_weighted_wedge_contribution=truth,
            native_z=np.asarray(81, dtype=np.int64),
            zperiod=np.asarray(5, dtype=np.int64),
            held_out_run_read=np.asarray(False),
            held_out_85606_read=np.asarray(False),
            new_segment_read=np.asarray(False),
            new_nersc_data_read=np.asarray(False),
        )
        handle.flush()
    partial.replace(target)
    return {"path": str(target), "sha256": sha256_path(target), "shape": list(truth.shape)}


def load_training_transport_truth(path: Path, *, expected_sha256: str) -> np.ndarray:
    target = Path(path)
    assert_development_path(target)
    if sha256_path(target) != str(expected_sha256):
        raise ValueError("native training transport truth SHA-256 differs")
    with np.load(target, allow_pickle=False) as data:
        if (
            int(data["schema_version"]) != 1
            or str(data["development_run"]) != "85604"
            or not np.array_equal(data["frame_index"], np.arange(432))
            or tuple(str(value) for value in data["quantities"])
            != PGL_TRANSPORT_QUANTITIES
            or int(data["native_z"]) != 81
            or int(data["zperiod"]) != 5
            or bool(data["held_out_run_read"])
            or bool(data["held_out_85606_read"])
            or bool(data["new_segment_read"])
            or bool(data["new_nersc_data_read"])
        ):
            raise ValueError("native training transport truth identity differs")
        truth = np.asarray(data["local_weighted_wedge_contribution"], dtype=np.float64)
    if truth.shape != (432, 4, 16, 81) or not np.all(np.isfinite(truth)):
        raise ValueError("native training transport truth values differ")
    return np.ascontiguousarray(truth)


def score_variogram_terms(
    *,
    edm_loss: Tensor,
    members: Tensor,
    truth: Tensor,
    current: Tensor,
    transport: TorchSeparatrixTransport,
    transport_truth: Tensor,
    pair_banks: Mapping[str, IndexedPairBank],
    phi_mask: Tensor | None = None,
) -> VariogramTerms:
    """Compute fair field and member-wise authoritative transport terms."""

    if tuple(pair_banks) != _PAIR_BANK_NAMES:
        raise ValueError("variogram score pair-bank names differ")
    if members.ndim != 7 or truth.ndim != 6 or current.ndim != 5:
        raise ValueError("variogram score field shapes differ")
    if members.shape[1] != 4 or transport_truth.shape != (
        members.shape[0], 5, 4, 16, 81
    ):
        raise ValueError("variogram member or native truth contract differs")

    members_fixed = gauge_fix_phi(members, spatial_mask=phi_mask)
    truth_fixed = gauge_fix_phi(truth, spatial_mask=phi_mask)
    current_fixed = gauge_fix_phi(current[:, None], spatial_mask=phi_mask)[:, 0]
    field_spatial = fair_variogram_score(
        members_fixed, truth_fixed, pair_banks["field_spatial"]
    )
    temporal_members, temporal_truth = prepend_observed_current(
        members_fixed, truth_fixed, current_fixed
    )
    field_temporal = fair_variogram_score(
        temporal_members, temporal_truth, pair_banks["field_temporal"]
    )

    native_truth = transport_truth.to(device=members.device, dtype=torch.float32)
    local_members = transport(members)
    # The observed current transport is duplicated across members exactly.
    # It is conditioning information, not a generated sample, and introducing
    # a model88->native81 round-trip here would create an artificial t0 error.
    current_local = native_truth[:, None, :1].expand(
        native_truth.shape[0], members.shape[1], 1, *native_truth.shape[2:]
    )
    local_trajectory = torch.cat((current_local, local_members), dim=2)
    spatial_terms: list[Tensor] = []
    temporal_terms: list[Tensor] = []
    ordinary: dict[str, Tensor] = {
        "field_spatial": field_spatial.ordinary,
        "field_temporal": field_temporal.ordinary,
    }
    for quantity, name in enumerate(PGL_TRANSPORT_QUANTITIES):
        spatial_result = fair_variogram_score(
            local_members[:, :, :, quantity : quantity + 1],
            native_truth[:, 1:, quantity : quantity + 1],
            pair_banks["transport_spatial"],
        )
        temporal_result = fair_variogram_score(
            local_trajectory[:, :, :, quantity : quantity + 1],
            native_truth[:, :, quantity : quantity + 1],
            pair_banks["transport_temporal"],
        )
        spatial_terms.append(spatial_result.fair)
        temporal_terms.append(temporal_result.fair)
        ordinary[f"transport_spatial/{name}"] = spatial_result.ordinary
        ordinary[f"transport_temporal/{name}"] = temporal_result.ordinary
    return VariogramTerms(
        edm=edm_loss.float(),
        field_spatial=field_spatial.fair,
        field_temporal=field_temporal.fair,
        transport_spatial=tuple(spatial_terms),
        transport_temporal=tuple(temporal_terms),
        ordinary=ordinary,
    )


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise FloatingPointError("variogram training metric is empty or non-finite")
    return float(np.mean(values))


def train_variogram_arm(
    *,
    mean_model: nn.Module,
    edm: PersistentGlobalLocalEDM,
    transport: TorchSeparatrixTransport,
    training_dataset: _WindowDataset,
    transport_truth_by_frame: np.ndarray,
    pair_banks: Mapping[str, IndexedPairBank],
    controls: VariogramControlMagnitudes,
    output: Path,
    device: torch.device,
    paper0_commit: str,
    slurm_job_id: str,
    preflight: Mapping[str, Any],
    config: VariogramScreenConfig,
    on_update: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one matched warm-start arm and save only the fixed final EMA."""

    destination = Path(output)
    assert_development_path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("variogram warm-start training requires allocated CUDA")
    if (
        training_dataset.split != "train"
        or training_dataset.horizon != 4
        or len(training_dataset) != PGL_VARIOGRAM_EXPECTED_WINDOWS
        or np.asarray(transport_truth_by_frame).shape != (432, 4, 16, 81)
    ):
        raise ValueError("variogram training data contract differs")
    if preflight.get("status") != "passed" or preflight.get("development_run") != "85604":
        raise ValueError("variogram preflight did not pass")

    destination.mkdir(parents=True)
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)

    frozen_mean = mean_model.to(device, torch.float32).eval()
    frozen_mean.requires_grad_(False)
    raw_edm = edm.to(device, torch.float32).train()
    transport = transport.to(device, torch.float32).eval()
    transport.requires_grad_(False)
    initial_mean_hash = module_state_sha256(frozen_mean)
    initial_edm_hash = module_state_sha256(raw_edm)
    ema_edm = copy.deepcopy(raw_edm).to(device, torch.float32).eval()
    ema_edm.requires_grad_(False)
    optimizer = AdamW(
        raw_edm.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    optimizer_state_initially_empty = len(optimizer.state) == 0
    if not optimizer_state_initially_empty:
        raise AssertionError("warm-start optimizer unexpectedly restored state")

    run_config = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_pgl_variogram_warm_start",
        "mode": config.mode,
        "arm": config.arm,
        "development_run": "85604",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "training": config.to_record(),
        "controls": controls.to_record(),
        "pair_banks": {name: bank.to_record() for name, bank in pair_banks.items()},
        "parent": {
            "selected_seed": 1702,
            "selected_epoch": 20,
            "selected_checkpoint_sha256": (
                "4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb"
            ),
            "stored_optimizer_state_available": False,
            "interpretation": "matched_warm_start_with_fresh_optimizer",
        },
        "parameter_count": {
            "frozen_mean": parameter_count(frozen_mean),
            "trainable_stochastic": parameter_count(raw_edm),
        },
        "physics_derived_training_loss_used": config.physics_derived_training_loss_used,
        "future_truth_used_by_sampler": False,
        "held_out_85606_read": False,
        "held_out_run_read": False,
        "new_nersc_data_read": False,
        "new_segment_read": False,
    }
    write_strict_json_atomic(destination / "config.json", run_config)

    training_dataset.set_epoch(0)
    order = np.random.default_rng(
        np.random.SeedSequence([config.seed, 0, 0x50474C])
    ).permutation(len(training_dataset))
    selected_order = order[: config.training_windows]
    updates: list[dict[str, Any]] = []
    for group_start in range(0, len(selected_order), config.accumulation_windows):
        group = selected_order[group_start : group_start + config.accumulation_windows]
        if len(group) != config.accumulation_windows:
            raise RuntimeError("variogram accumulation group differs")
        optimizer.zero_grad(set_to_none=True)
        batch_logs: list[dict[str, float]] = []
        update_started = time.perf_counter()
        sampler_seeds: list[list[int]] = []
        group_frames: list[int] = []
        for dataset_index in group:
            item = training_dataset[int(dataset_index)]
            context, targets, current_frame = tensor_window(item, device)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                mean = mean_forecast_trajectory(frozen_mean, context).float()
            clean = raw_edm.normalize_residual(targets - mean)
            sigma, noise, _ = keyed_sigma_and_noise(
                base_seed=PGL_NOISE_BASE_SEED,
                epoch_zero_based=0,
                current_frame=current_frame,
                probe=0,
                reference=clean,
                noise_config=raw_edm.noise_config,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base = raw_edm.training_loss(
                    clean,
                    context[:, -1],
                    mean,
                    sigma=sigma,
                    noise=noise,
                ).loss
                initial_noise, seed_values = keyed_sampler_initial_noise(
                    model=raw_edm,
                    reference=mean,
                    epoch_zero_based=0,
                    current_frame=current_frame,
                )
                normalized = differentiable_sample_normalized(
                    raw_edm,
                    context[:, -1],
                    mean,
                    initial_noise,
                    steps=config.sampler_steps,
                    activation_checkpointing=True,
                )
                members = raw_edm.compose_fields(mean, normalized).float()
            sampler_seeds.append(list(seed_values))
            group_frames.append(current_frame)
            native_truth = training_transport_window(
                transport_truth_by_frame,
                current_frame=current_frame,
                model_roll=int(item["toroidal_roll"]),
            )[None].to(device=device, dtype=torch.float32)
            terms = score_variogram_terms(
                edm_loss=base,
                members=members,
                truth=targets,
                current=context[:, -1],
                transport=transport,
                transport_truth=native_truth,
                pair_banks=pair_banks,
            )
            objective, field_norm, transport_norm = arm_objective(
                config.arm, terms, controls
            )
            # Arm A deliberately traverses the same sampler graph with zero
            # weight.  This is a compute/memory control, not an auxiliary loss.
            if config.arm == "A":
                objective = objective + members.sum() * 0.0
            if not torch.isfinite(objective):
                raise FloatingPointError("variogram arm objective is non-finite")
            (objective / config.accumulation_windows).backward()
            log = {
                "objective": float(objective.detach().cpu()),
                "edm": float(terms.edm.detach().cpu()),
                "field_spatial_fair": float(terms.field_spatial.detach().cpu()),
                "field_temporal_fair": float(terms.field_temporal.detach().cpu()),
                "field_normalized": float(field_norm.detach().cpu()),
                "transport_normalized": float(transport_norm.detach().cpu()),
            }
            for index, name in enumerate(PGL_TRANSPORT_QUANTITIES):
                log[f"transport_spatial_fair/{name}"] = float(
                    terms.transport_spatial[index].detach().cpu()
                )
                log[f"transport_temporal_fair/{name}"] = float(
                    terms.transport_temporal[index].detach().cpu()
                )
            for name, value in terms.ordinary.items():
                log[f"ordinary/{name}"] = float(value.detach().cpu())
            batch_logs.append(log)

        norm = torch.nn.utils.clip_grad_norm_(raw_edm.parameters(), config.gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("variogram arm gradient norm is non-finite")
        optimizer.step()
        update_ema_model(ema_edm, raw_edm, decay=config.ema_decay)
        update_index = len(updates) + 1
        record: dict[str, Any] = {
            "optimizer_update": update_index,
            "current_frames": group_frames,
            "sampler_seed_rows": sampler_seeds,
            "preclip_gradient_norm": float(norm.detach().cpu()),
            "learning_rate": config.learning_rate,
            "wall_seconds": float(time.perf_counter() - update_started),
        }
        for name in batch_logs[0]:
            record[name] = _mean([value[name] for value in batch_logs])
        updates.append(record)
        if on_update is not None:
            on_update(record)

    if len(updates) != config.optimizer_updates:
        raise RuntimeError("variogram optimizer update count differs")
    final_mean_hash = module_state_sha256(frozen_mean)
    if final_mean_hash != initial_mean_hash:
        raise AssertionError("frozen selected mean changed during variogram training")
    selected_checkpoint = destination / "selected.pt"
    payload = {
        "schema_version": 1,
        "kind": "pgl_variogram_fixed_final_EMA_warm_start",
        "development_run": "85604",
        "arm": config.arm,
        "mode": config.mode,
        "seed": config.seed,
        "paper0_commit": str(paper0_commit),
        "completed_epoch": 1,
        "optimizer_updates": len(updates),
        "training": config.to_record(),
        "mean_model_state": {
            name: value.detach().cpu() for name, value in frozen_mean.state_dict().items()
        },
        "stochastic_model_state": {
            name: value.detach().cpu() for name, value in ema_edm.state_dict().items()
        },
        "parent_mean_state_sha256": initial_mean_hash,
        "parent_stochastic_state_sha256": initial_edm_hash,
        "checkpoint_selection_performed": False,
        "physics_derived_training_loss_used": config.physics_derived_training_loss_used,
        "held_out_run_read": False,
        "held_out_85606_read": False,
        "new_segment_read": False,
        "new_nersc_data_read": False,
    }
    save_torch_atomic(selected_checkpoint, payload)
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    result = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_pgl_variogram_warm_start",
        "status": "smoke_passed" if config.mode == "smoke" else "screen_training_completed",
        "mode": config.mode,
        "arm": config.arm,
        "development_run": "85604",
        "seed": config.seed,
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "completed_epochs": 1,
        "completed_optimizer_updates": len(updates),
        "training": config.to_record(),
        "history": updates,
        "initial_state_sha256": {
            "mean": initial_mean_hash,
            "stochastic": initial_edm_hash,
        },
        "final_state_sha256": {
            "mean": final_mean_hash,
            "stochastic_ema": module_state_sha256(ema_edm),
        },
        "selected_checkpoint": {
            "path": str(selected_checkpoint),
            "sha256": sha256_path(selected_checkpoint),
            "selection": "fixed_final_EMA_no_checkpoint_selection",
        },
        "fresh_optimizer": optimizer_state_initially_empty,
        "mean_frozen_bitwise": final_mean_hash == initial_mean_hash,
        "full_sampler_compute_control_executed": True,
        "sampler_steps": 18,
        "network_evaluations_per_member": 35,
        "sampler_members": 4,
        "peak_cuda_memory_bytes": peak_bytes,
        "peak_cuda_memory_GiB": float(peak_bytes / 2**30),
        "gpu": torch.cuda.get_device_name(device),
        "wall_seconds": float(time.perf_counter() - started),
        "physics_derived_training_loss_used": config.physics_derived_training_loss_used,
        "future_truth_used_by_sampler": False,
        "checkpoint_selection_performed": False,
        "held_out_run_read": False,
        "held_out_85606_read": False,
        "new_segment_read": False,
        "new_nersc_data_read": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    write_strict_json_atomic(destination / "result.json", result)
    return result
