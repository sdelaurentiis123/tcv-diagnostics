"""Matched end-to-end PGL training with hierarchical transport scores."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from .b5_residual_edm_full_training import update_ema_model
from .b5_residual_edm_training import module_state_sha256, parameter_count
from .codec_training import save_torch_atomic, sha256_path
from .model_data import assert_development_path, write_strict_json_atomic
from .models.persistent_global_local import PersistentGlobalLocalEDM
from .persistent_global_local_training import (
    PGL_HORIZON,
    PGL_NOISE_BASE_SEED,
    PGL_SEED,
    mean_forecast_trajectory,
    keyed_sigma_and_noise,
    tensor_window,
    weighted_mean_state_loss,
)
from .pgl_hierarchical_transport import (
    HierarchicalTransportScores,
    score_hierarchical_transport,
)
from .pgl_torch_transport import PGL_TRANSPORT_QUANTITIES, TorchSeparatrixTransport
from .pgl_variogram import IndexedPairBank, differentiable_sample_normalized
from .pgl_variogram_training import keyed_sampler_initial_noise, training_transport_window


PGL_HIERARCHICAL_ARMS = ("CONTROL", "TRANSPORT")
PGL_HIERARCHICAL_CHECKPOINT_UPDATES = (107, 214, 428)
PGL_HIERARCHICAL_EXPECTED_WINDOWS = 428
PGL_HIERARCHICAL_UPDATES_PER_EPOCH = 214
PGL_HIERARCHICAL_TOTAL_UPDATES = 428
PGL_HIERARCHICAL_STOCHASTIC_LR = 1.0e-6
PGL_HIERARCHICAL_MEAN_LR = 1.0e-7
PGL_HIERARCHICAL_TARGET_GRADIENT_RATIO = 0.25
PGL_HIERARCHICAL_GRADIENT_STARTS = (0, 142, 285, 427)
PGL_HIERARCHICAL_CONTROL_STARTS = tuple(
    int(value) for value in np.floor(np.linspace(0, 427, 32)).astype(np.int64)
)
PGL_HIERARCHICAL_CONTROL_SCHEMA = 1


class _WindowDataset(Protocol):
    split: str
    horizon: int

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...

    def set_epoch(self, epoch: int) -> None: ...


@dataclass(frozen=True)
class HierarchicalTrainingConfig:
    """Frozen smoke or two-epoch hierarchy experiment."""

    mode: str
    arm: str
    seed: int = PGL_SEED
    stochastic_learning_rate: float = PGL_HIERARCHICAL_STOCHASTIC_LR
    mean_learning_rate: float = PGL_HIERARCHICAL_MEAN_LR
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    accumulation_windows: int = 2
    sampler_members: int = 4
    sampler_steps: int = 18

    def __post_init__(self) -> None:
        if self.mode not in ("smoke", "screen"):
            raise ValueError("hierarchical mode must be smoke or screen")
        if self.arm not in PGL_HIERARCHICAL_ARMS:
            raise ValueError("hierarchical arm differs")
        frozen = {
            "seed": 1702,
            "stochastic_learning_rate": 1.0e-6,
            "mean_learning_rate": 1.0e-7,
            "betas": (0.9, 0.99),
            "weight_decay": 1.0e-4,
            "gradient_clip": 1.0,
            "ema_decay": 0.999,
            "accumulation_windows": 2,
            "sampler_members": 4,
            "sampler_steps": 18,
        }
        if any(getattr(self, key) != value for key, value in frozen.items()):
            raise ValueError("hierarchical optimization contract differs")

    @property
    def training_windows(self) -> int:
        return 2 if self.mode == "smoke" else 2 * PGL_HIERARCHICAL_EXPECTED_WINDOWS

    @property
    def optimizer_updates(self) -> int:
        return 1 if self.mode == "smoke" else PGL_HIERARCHICAL_TOTAL_UPDATES

    @property
    def checkpoints(self) -> tuple[int, ...]:
        return (1,) if self.mode == "smoke" else PGL_HIERARCHICAL_CHECKPOINT_UPDATES

    @property
    def physics_derived_training_loss_used(self) -> bool:
        return self.arm == "TRANSPORT"

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["betas"] = list(self.betas)
        record.update(
            {
                "epochs": 0.5 / 107 if self.mode == "smoke" else 2.0,
                "training_windows": self.training_windows,
                "optimizer_updates": self.optimizer_updates,
                "checkpoint_updates": list(self.checkpoints),
                "checkpoint_equivalent_epochs": (
                    [1.0 / PGL_HIERARCHICAL_UPDATES_PER_EPOCH]
                    if self.mode == "smoke"
                    else [0.5, 1.0, 2.0]
                ),
                "fresh_optimizer": True,
                "constant_learning_rates": True,
                "mean_to_stochastic_learning_rate_ratio": 0.1,
                "mean_feedback_gradient": "detached_between_steps",
                "edm_gradient_to_mean": False,
                "transport_gradient_to_mean": True,
                "fixed_duration_checkpoints_no_selection": True,
                "network_evaluations_per_member": 35,
                "future_truth_used_by_sampler": False,
                "physics_derived_training_loss_used": (
                    self.physics_derived_training_loss_used
                ),
                "held_out_85606_read": False,
                "new_nersc_data_read": False,
            }
        )
        return record


@dataclass(frozen=True)
class HierarchicalControlMagnitudes:
    """Frozen per-quantity initial magnitudes for every hierarchy component."""

    local_spatial: tuple[float, ...]
    local_temporal: tuple[float, ...]
    regional: tuple[float, ...]
    fourier_low: tuple[float, ...]
    fourier_transport_band: tuple[float, ...]
    global_crps: tuple[float, ...]

    def __post_init__(self) -> None:
        groups = (
            self.local_spatial,
            self.local_temporal,
            self.regional,
            self.fourier_low,
            self.fourier_transport_band,
            self.global_crps,
        )
        expected = len(PGL_TRANSPORT_QUANTITIES)
        if any(len(values) != expected for values in groups):
            raise ValueError("hierarchical controls require every quantity")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for values in groups
            for value in values
        ):
            raise ValueError("hierarchical controls must be positive and finite")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "HierarchicalControlMagnitudes":
        if (
            int(record.get("schema_version", -1)) != PGL_HIERARCHICAL_CONTROL_SCHEMA
            or record.get("development_run") != "85604"
            or record.get("current_frames") != list(PGL_HIERARCHICAL_CONTROL_STARTS)
            or record.get("sampler_members") != 4
            or record.get("sampler_steps") != 18
            or record.get("held_out_85606_read") is not False
            or record.get("new_nersc_data_read") is not False
        ):
            raise ValueError("hierarchical control artifact contract differs")

        def values(name: str) -> tuple[float, ...]:
            mapping = record.get(name, {})
            if tuple(mapping) != PGL_TRANSPORT_QUANTITIES:
                raise ValueError(f"hierarchical control quantities differ for {name}")
            return tuple(float(mapping[quantity]) for quantity in PGL_TRANSPORT_QUANTITIES)

        return cls(
            local_spatial=values("local_spatial"),
            local_temporal=values("local_temporal"),
            regional=values("regional"),
            fourier_low=values("fourier_low"),
            fourier_transport_band=values("fourier_transport_band"),
            global_crps=values("global_crps"),
        )

    def to_record(self) -> dict[str, Any]:
        def mapping(values: Sequence[float]) -> dict[str, float]:
            return {
                name: float(value)
                for name, value in zip(PGL_TRANSPORT_QUANTITIES, values)
            }

        return {
            "schema_version": PGL_HIERARCHICAL_CONTROL_SCHEMA,
            "scope": "old_85604_pgl_hierarchical_transport_initial_controls",
            "development_run": "85604",
            "current_frames": list(PGL_HIERARCHICAL_CONTROL_STARTS),
            "sampler_members": 4,
            "sampler_steps": 18,
            "local_spatial": mapping(self.local_spatial),
            "local_temporal": mapping(self.local_temporal),
            "regional": mapping(self.regional),
            "fourier_low": mapping(self.fourier_low),
            "fourier_transport_band": mapping(self.fourier_transport_band),
            "global_crps": mapping(self.global_crps),
            "normalization": "arithmetic_mean_on_32_fixed_training_starts",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
        }

    def normalize(
        self, scores: HierarchicalTransportScores
    ) -> tuple[Tensor, Tensor, Tensor]:
        local_values: list[Tensor] = []
        regional_values: list[Tensor] = []
        global_values: list[Tensor] = []
        for index in range(len(PGL_TRANSPORT_QUANTITIES)):
            local_values.append(
                0.5
                * (
                    scores.local_spatial[index] / self.local_spatial[index]
                    + scores.local_temporal[index] / self.local_temporal[index]
                )
            )
            regional_values.append(
                (
                    scores.regional[index] / self.regional[index]
                    + scores.fourier_low[index] / self.fourier_low[index]
                    + scores.fourier_transport_band[index]
                    / self.fourier_transport_band[index]
                )
                / 3.0
            )
            global_values.append(scores.global_crps[index] / self.global_crps[index])
        return (
            torch.stack(local_values).mean(),
            torch.stack(regional_values).mean(),
            torch.stack(global_values).mean(),
        )


@dataclass(frozen=True)
class HierarchicalTerms:
    mean: Tensor
    edm: Tensor
    scores: HierarchicalTransportScores

    @property
    def original(self) -> Tensor:
        return self.mean + self.edm


def score_hierarchical_terms(
    *,
    mean_loss: Tensor,
    edm_loss: Tensor,
    members: Tensor,
    transport: TorchSeparatrixTransport,
    transport_truth: Tensor,
    spatial_bank: IndexedPairBank,
    temporal_bank: IndexedPairBank,
) -> HierarchicalTerms:
    """Apply authoritative transport memberwise and compute the full hierarchy."""

    if members.ndim != 7 or members.shape[1:4] != (4, 4, 5):
        raise ValueError("hierarchical field members must be [B,4,4,5,x,y,z]")
    if transport_truth.shape != (members.shape[0], 5, 4, 16, 81):
        raise ValueError("hierarchical native transport truth shape differs")
    native_truth = transport_truth.to(device=members.device, dtype=torch.float32)
    local_members = transport(members)
    current_local = native_truth[:, None, :1].expand(
        native_truth.shape[0], members.shape[1], 1, *native_truth.shape[2:]
    )
    trajectory_members = torch.cat((current_local, local_members), dim=2)
    scores = score_hierarchical_transport(
        local_members=local_members,
        local_future_truth=native_truth[:, 1:],
        local_trajectory_members=trajectory_members,
        local_trajectory_truth=native_truth,
        spatial_bank=spatial_bank,
        temporal_bank=temporal_bank,
    )
    return HierarchicalTerms(mean=mean_loss.float(), edm=edm_loss.float(), scores=scores)


def hierarchical_objective(
    *,
    arm: str,
    terms: HierarchicalTerms,
    controls: HierarchicalControlMagnitudes,
    auxiliary_lambda: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if arm not in PGL_HIERARCHICAL_ARMS:
        raise ValueError("hierarchical arm differs")
    if not math.isfinite(float(auxiliary_lambda)) or float(auxiliary_lambda) <= 0.0:
        raise ValueError("hierarchical auxiliary multiplier must be finite and positive")
    local, regional, global_score = controls.normalize(terms.scores)
    auxiliary = local + regional + global_score
    if arm == "TRANSPORT":
        objective = terms.original + float(auxiliary_lambda) * auxiliary
    else:
        objective = terms.original + auxiliary * 0.0
    return objective, local, regional, global_score


def parameter_branches(
    mean_model: nn.Module, edm: PersistentGlobalLocalEDM
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Return a complete disjoint partition used for gradient accounting."""

    branches: dict[str, list[tuple[str, nn.Parameter]]] = {
        "mean": [(f"mean.{name}", value) for name, value in mean_model.named_parameters()],
        "stochastic_global": [],
        "stochastic_local_encoder": [],
        "stochastic_local_decoder": [],
    }
    decoder_markers = (
        ".merges.",
        ".decoders.",
        ".output_normalization.",
        ".output_projection.",
    )
    for name, value in edm.named_parameters():
        qualified = f"stochastic.{name}"
        if name.startswith("backbone.global_stream."):
            branches["stochastic_global"].append((qualified, value))
        elif name.startswith("backbone.local_stream.") and any(
            marker in name for marker in decoder_markers
        ):
            branches["stochastic_local_decoder"].append((qualified, value))
        elif name.startswith("backbone.local_stream."):
            branches["stochastic_local_encoder"].append((qualified, value))
        else:
            raise ValueError(f"unclassified stochastic parameter {name}")
    flattened = [id(value) for values in branches.values() for _, value in values]
    expected = [id(value) for value in list(mean_model.parameters()) + list(edm.parameters())]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(expected):
        raise AssertionError("parameter branches are not a disjoint complete partition")
    if any(not values for values in branches.values()):
        raise ValueError("a hierarchical parameter branch is empty")
    return branches


def loss_gradient_audit(
    losses: Mapping[str, Tensor],
    branches: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
    *,
    retain_graph: bool,
) -> dict[str, Any]:
    """Measure per-loss/per-branch gradient norms and mutual cosines."""

    ordered_parameters = [value for values in branches.values() for _, value in values]
    branch_lengths = [len(values) for values in branches.values()]
    vectors: dict[str, tuple[Tensor | None, ...]] = {}
    summaries: dict[str, Any] = {}
    loss_items = list(losses.items())
    for index, (loss_name, loss) in enumerate(loss_items):
        gradients = torch.autograd.grad(
            loss,
            ordered_parameters,
            retain_graph=(retain_graph or index < len(loss_items) - 1),
            allow_unused=True,
            create_graph=False,
        )
        vectors[loss_name] = gradients
        offset = 0
        branch_records: dict[str, Any] = {}
        total_squared = torch.zeros((), device=loss.device, dtype=torch.float64)
        for (branch_name, values), length in zip(branches.items(), branch_lengths):
            selected = gradients[offset : offset + length]
            offset += length
            squared = torch.zeros((), device=loss.device, dtype=torch.float64)
            tensors = 0
            parameters = 0
            for gradient, (_, parameter) in zip(selected, values):
                if gradient is not None:
                    squared = squared + gradient.detach().double().square().sum()
                    tensors += 1
                    parameters += int(parameter.numel())
            norm = torch.sqrt(squared)
            total_squared = total_squared + squared
            branch_records[branch_name] = {
                "gradient_norm": float(norm.cpu()),
                "nonzero_tensor_count": tensors,
                "reached_parameter_count": parameters,
            }
        total = torch.sqrt(total_squared)
        summaries[loss_name] = {
            "loss": float(loss.detach().cpu()),
            "total_gradient_norm": float(total.cpu()),
            "branches": branch_records,
        }

    cosines: dict[str, float] = {}
    names = list(vectors)
    for first_index, first in enumerate(names):
        first_norm = float(summaries[first]["total_gradient_norm"])
        for second in names[first_index + 1 :]:
            second_norm = float(summaries[second]["total_gradient_norm"])
            dot = 0.0
            for left, right in zip(vectors[first], vectors[second]):
                if left is not None and right is not None:
                    dot += float(torch.sum(left.detach().double() * right.detach().double()).cpu())
            denominator = first_norm * second_norm
            cosines[f"{first}__{second}"] = dot / denominator if denominator > 0.0 else 0.0
    return {"losses": summaries, "cosine_similarity": cosines}


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _mean(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise FloatingPointError("hierarchical training metric is empty or non-finite")
    return float(np.mean(values))


def train_hierarchical_arm(
    *,
    mean_model: nn.Module,
    edm: PersistentGlobalLocalEDM,
    transport: TorchSeparatrixTransport,
    training_dataset: _WindowDataset,
    derivative_rms: Tensor,
    transport_truth_by_frame: np.ndarray,
    spatial_bank: IndexedPairBank,
    temporal_bank: IndexedPairBank,
    controls: HierarchicalControlMagnitudes,
    auxiliary_lambda: float,
    gradient_calibration: Mapping[str, Any],
    output: Path,
    device: torch.device,
    paper0_commit: str,
    slurm_job_id: str,
    parent_checkpoint_sha256: str,
    config: HierarchicalTrainingConfig,
    on_update: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one matched hierarchy arm and save fixed-duration checkpoints."""

    destination = Path(output)
    assert_development_path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("hierarchical training requires allocated CUDA")
    if (
        training_dataset.split != "train"
        or training_dataset.horizon != PGL_HORIZON
        or len(training_dataset) != PGL_HIERARCHICAL_EXPECTED_WINDOWS
        or np.asarray(transport_truth_by_frame).shape != (432, 4, 16, 81)
        or derivative_rms.shape != (5,)
    ):
        raise ValueError("hierarchical training data contract differs")
    if (
        gradient_calibration.get("status") != "passed"
        or gradient_calibration.get("target_ratio") != 0.25
        or not math.isclose(
            float(gradient_calibration.get("auxiliary_lambda", -1.0)),
            float(auxiliary_lambda),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("hierarchical gradient calibration differs")

    destination.mkdir(parents=True)
    checkpoint_directory = destination / "checkpoints"
    checkpoint_directory.mkdir()
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)

    raw_mean = mean_model.to(device, torch.float32).train()
    raw_edm = edm.to(device, torch.float32).train()
    transport = transport.to(device, torch.float32).eval().requires_grad_(False)
    ema_mean = copy.deepcopy(raw_mean).to(device, torch.float32).eval().requires_grad_(False)
    ema_edm = copy.deepcopy(raw_edm).to(device, torch.float32).eval().requires_grad_(False)
    branches = parameter_branches(raw_mean, raw_edm)
    initial_hashes = {
        "mean": module_state_sha256(raw_mean),
        "stochastic": module_state_sha256(raw_edm),
    }
    optimizer = AdamW(
        [
            {
                "params": list(raw_edm.parameters()),
                "lr": config.stochastic_learning_rate,
                "name": "stochastic",
            },
            {
                "params": list(raw_mean.parameters()),
                "lr": config.mean_learning_rate,
                "name": "mean",
            },
        ],
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    if optimizer.state:
        raise AssertionError("hierarchical optimizer unexpectedly restored state")

    run_config = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_pgl_hierarchical_transport_training",
        "development_run": "85604",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "arm": config.arm,
        "training": config.to_record(),
        "controls": controls.to_record(),
        "auxiliary_lambda": float(auxiliary_lambda),
        "gradient_calibration": dict(gradient_calibration),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "parameter_count": {
            "mean": parameter_count(raw_mean),
            "stochastic": parameter_count(raw_edm),
            "branches": {
                name: sum(value.numel() for _, value in values)
                for name, values in branches.items()
            },
        },
        "physics_derived_training_loss_used": config.physics_derived_training_loss_used,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
    write_strict_json_atomic(destination / "config.json", run_config)

    histories: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    global_update = 0
    epochs = 1 if config.mode == "smoke" else 2
    for epoch_zero in range(epochs):
        training_dataset.set_epoch(epoch_zero)
        order = np.random.default_rng(
            np.random.SeedSequence([config.seed, epoch_zero, 0x48494552])
        ).permutation(len(training_dataset))
        if config.mode == "smoke":
            order = order[:2]
        for group_start in range(0, len(order), config.accumulation_windows):
            group = order[group_start : group_start + config.accumulation_windows]
            if len(group) != config.accumulation_windows:
                raise RuntimeError("hierarchical accumulation group differs")
            next_update = global_update + 1
            optimizer.zero_grad(set_to_none=True)
            update_started = time.perf_counter()
            batch_logs: list[dict[str, float]] = []
            gradient_audit: dict[str, Any] | None = None
            current_frames: list[int] = []
            for within_group, dataset_index in enumerate(group):
                item = training_dataset[int(dataset_index)]
                context, targets, current_frame = tensor_window(item, device)
                current_frames.append(current_frame)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mean = mean_forecast_trajectory(raw_mean, context).float()
                    mean_loss, _ = weighted_mean_state_loss(mean, targets, derivative_rms)
                    clean = raw_edm.normalize_residual(targets - mean.detach())
                    sigma, noise, _ = keyed_sigma_and_noise(
                        base_seed=PGL_NOISE_BASE_SEED,
                        epoch_zero_based=epoch_zero,
                        current_frame=current_frame,
                        probe=0,
                        reference=clean,
                        noise_config=raw_edm.noise_config,
                    )
                    edm_loss = raw_edm.training_loss(
                        clean,
                        context[:, -1],
                        mean.detach(),
                        sigma=sigma,
                        noise=noise,
                    ).loss
                    initial_noise, _ = keyed_sampler_initial_noise(
                        model=raw_edm,
                        reference=mean,
                        epoch_zero_based=epoch_zero,
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
                native_truth = training_transport_window(
                    transport_truth_by_frame,
                    current_frame=current_frame,
                    model_roll=int(item["toroidal_roll"]),
                )[None].to(device=device, dtype=torch.float32)
                terms = score_hierarchical_terms(
                    mean_loss=mean_loss,
                    edm_loss=edm_loss,
                    members=members,
                    transport=transport,
                    transport_truth=native_truth,
                    spatial_bank=spatial_bank,
                    temporal_bank=temporal_bank,
                )
                objective, local, regional, global_score = hierarchical_objective(
                    arm=config.arm,
                    terms=terms,
                    controls=controls,
                    auxiliary_lambda=auxiliary_lambda,
                )
                if within_group == 0 and next_update in config.checkpoints:
                    gradient_audit = loss_gradient_audit(
                        {
                            "original": terms.original,
                            "local": local,
                            "regional": regional,
                            "global": global_score,
                        },
                        branches,
                        retain_graph=True,
                    )
                if not torch.isfinite(objective):
                    raise FloatingPointError("hierarchical objective is non-finite")
                (objective / config.accumulation_windows).backward()
                log = {
                    "objective": float(objective.detach().cpu()),
                    "original": float(terms.original.detach().cpu()),
                    "mean": float(terms.mean.detach().cpu()),
                    "edm": float(terms.edm.detach().cpu()),
                    "local_normalized": float(local.detach().cpu()),
                    "regional_normalized": float(regional.detach().cpu()),
                    "global_normalized": float(global_score.detach().cpu()),
                }
                for index, name in enumerate(PGL_TRANSPORT_QUANTITIES):
                    for component in (
                        "local_spatial",
                        "local_temporal",
                        "regional",
                        "fourier_low",
                        "fourier_transport_band",
                        "global_crps",
                    ):
                        log[f"{component}/{name}"] = float(
                            getattr(terms.scores, component)[index].detach().cpu()
                        )
                batch_logs.append(log)

            preclip = torch.nn.utils.clip_grad_norm_(
                list(raw_mean.parameters()) + list(raw_edm.parameters()),
                config.gradient_clip,
            )
            if not torch.isfinite(preclip):
                raise FloatingPointError("hierarchical gradient norm is non-finite")
            optimizer.step()
            update_ema_model(ema_mean, raw_mean, decay=config.ema_decay)
            update_ema_model(ema_edm, raw_edm, decay=config.ema_decay)
            global_update = next_update
            record: dict[str, Any] = {
                "optimizer_update": global_update,
                "equivalent_epochs": global_update / PGL_HIERARCHICAL_UPDATES_PER_EPOCH,
                "current_frames": current_frames,
                "preclip_gradient_norm": float(preclip.detach().cpu()),
                "stochastic_learning_rate": config.stochastic_learning_rate,
                "mean_learning_rate": config.mean_learning_rate,
                "wall_seconds": float(time.perf_counter() - update_started),
                "gradient_audit": gradient_audit,
            }
            for name in batch_logs[0]:
                record[name] = _mean([row[name] for row in batch_logs])
            histories.append(record)
            if on_update is not None:
                on_update(record)

            if global_update in config.checkpoints:
                checkpoint = checkpoint_directory / f"fixed_update_{global_update:04d}.pt"
                payload = {
                    "schema_version": 1,
                    "kind": "pgl_hierarchical_transport_fixed_update",
                    "development_run": "85604",
                    "paper0_commit": str(paper0_commit),
                    "slurm_job_id": str(slurm_job_id),
                    "arm": config.arm,
                    "mode": config.mode,
                    "seed": config.seed,
                    "optimizer_update": global_update,
                    "equivalent_epochs": global_update
                    / PGL_HIERARCHICAL_UPDATES_PER_EPOCH,
                    "training": config.to_record(),
                    "auxiliary_lambda": float(auxiliary_lambda),
                    "mean_model_state": _cpu_tree(ema_mean.state_dict()),
                    "stochastic_model_state": _cpu_tree(ema_edm.state_dict()),
                    "raw_mean_model_state": _cpu_tree(raw_mean.state_dict()),
                    "raw_stochastic_model_state": _cpu_tree(raw_edm.state_dict()),
                    "optimizer_state": _cpu_tree(optimizer.state_dict()),
                    "state_sha256": {
                        "raw_mean": module_state_sha256(raw_mean),
                        "raw_stochastic": module_state_sha256(raw_edm),
                        "ema_mean": module_state_sha256(ema_mean),
                        "ema_stochastic": module_state_sha256(ema_edm),
                    },
                    "gradient_audit": gradient_audit,
                    "checkpoint_selection_performed": False,
                    "physics_derived_training_loss_used": (
                        config.physics_derived_training_loss_used
                    ),
                    "held_out_85606_read": False,
                    "new_nersc_data_read": False,
                }
                save_torch_atomic(checkpoint, payload)
                checkpoints.append(
                    {
                        "optimizer_update": global_update,
                        "equivalent_epochs": payload["equivalent_epochs"],
                        "path": str(checkpoint),
                        "sha256": sha256_path(checkpoint),
                        "selection": "fixed_duration_no_selection",
                        "state_sha256": payload["state_sha256"],
                    }
                )

    if global_update != config.optimizer_updates or len(checkpoints) != len(config.checkpoints):
        raise RuntimeError("hierarchical completed budget or checkpoint count differs")
    result = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_pgl_hierarchical_transport_training",
        "status": "smoke_passed" if config.mode == "smoke" else "screen_training_completed",
        "development_run": "85604",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "arm": config.arm,
        "mode": config.mode,
        "seed": config.seed,
        "training": config.to_record(),
        "completed_optimizer_updates": global_update,
        "initial_state_sha256": initial_hashes,
        "final_state_sha256": {
            "raw_mean": module_state_sha256(raw_mean),
            "raw_stochastic": module_state_sha256(raw_edm),
            "ema_mean": module_state_sha256(ema_mean),
            "ema_stochastic": module_state_sha256(ema_edm),
        },
        "checkpoints": checkpoints,
        "history": histories,
        "auxiliary_lambda": float(auxiliary_lambda),
        "gradient_calibration": dict(gradient_calibration),
        "fresh_optimizer": True,
        "checkpoint_selection_performed": False,
        "full_sampler_compute_control_executed": True,
        "peak_cuda_memory_GiB": float(torch.cuda.max_memory_allocated(device) / 2**30),
        "wall_seconds": float(time.perf_counter() - started),
        "gpu": torch.cuda.get_device_name(device),
        "physics_derived_training_loss_used": config.physics_derived_training_loss_used,
        "future_truth_used_by_sampler": False,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    write_strict_json_atomic(destination / "result.json", result)
    return result
