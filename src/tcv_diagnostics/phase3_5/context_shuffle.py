"""Fixed-seed B5 context-shuffle sensitivity for Phase 3.5.

This is frozen-model inference only.  The generated residual conditioned on a
mismatched chronological context is always added to the original H1 center,
so the comparison tests residual conditioning rather than deterministic-mean
movement.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from tcv_diagnostics.b5_residual_edm_forecast import (
    B5ForecastArtifact,
    B5ForecastSchema,
    initial_noise_from_uint64,
    load_scientific_sampler_seed_bank,
    load_selected_b5_model,
)

from .data import (
    B5_CHECKPOINT,
    B5_CHECKPOINT_SHA256,
    B5_FORECAST,
    B5_FORECAST_SHA256,
    B5_SEEDS,
    B5_SEEDS_SHA256,
)
from .diagnostics import band_power, cross_field_covariance, matrix_relative_distance


B5_TRAINING_COMMIT = "512c987d49a1a572430ed6f9fca18975798fc599"
SELECTED_TARGETS = tuple(
    block_start + offset
    for block_start in range(498, 624, 21)
    for offset in (5, 10, 15)
)


def mismatched_target(target: int) -> int:
    index = int(target) - 498
    if not 0 <= index < 126:
        raise ValueError("B5 shuffle target leaves validation")
    return 498 + (index + 63) % 126


def _noise_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _field_covariance_metrics(correct: np.ndarray, shuffled: np.ndarray) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for channel, field in enumerate(("Ne", "Pe", "Pi", "phi", "Vi")):
        first = float(np.mean(np.var(correct[:, channel], axis=0, ddof=1)))
        second = float(np.mean(np.var(shuffled[:, channel], axis=0, ddof=1)))
        rows.append(
            {
                "family": "field_variance",
                "quantity": field,
                "correct_context": first,
                "shuffled_context": second,
                "relative_change": (second - first) / first if first > 0.0 else float("nan"),
            }
        )
    first_covariance = cross_field_covariance(correct)
    second_covariance = cross_field_covariance(shuffled)
    rows.append(
        {
            "family": "cross_field_covariance",
            "quantity": "global_matrix",
            "correct_context": float(np.linalg.norm(first_covariance)),
            "shuffled_context": float(np.linalg.norm(second_covariance)),
            "relative_change": matrix_relative_distance(first_covariance, second_covariance),
        }
    )
    correct_power = band_power(correct)
    shuffled_power = band_power(shuffled)
    for band in correct_power:
        for channel, field in enumerate(("Ne", "Pe", "Pi", "phi", "Vi")):
            first = float(np.mean(correct_power[band][:, channel]))
            second = float(np.mean(shuffled_power[band][:, channel]))
            rows.append(
                {
                    "family": "spectral_band_covariance",
                    "quantity": f"{field}.{band}",
                    "correct_context": first,
                    "shuffled_context": second,
                    "relative_change": (second - first) / first if first > 0.0 else float("nan"),
                }
            )
    return rows


def run_b5_context_shuffle(
    *,
    validation_context_by_target: Mapping[int, np.ndarray],
    validation_h1_by_target: Mapping[int, np.ndarray],
    device: torch.device,
    member_metric_callback: Callable[[np.ndarray, np.ndarray, int], Sequence[Mapping[str, object]]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Generate the preregistered 18 eight-member mismatched ensembles."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B5 context-shuffle inference requires a Rusty H100")
    if tuple(sorted(validation_context_by_target)) != tuple(range(498, 624)):
        raise ValueError("B5 shuffle context map must cover exact validation targets")
    if tuple(sorted(validation_h1_by_target)) != tuple(range(498, 624)):
        raise ValueError("B5 shuffle H1 map must cover exact validation targets")
    seed_bank = load_scientific_sampler_seed_bank(B5_SEEDS, B5_SEEDS_SHA256)
    model = load_selected_b5_model(
        checkpoint=B5_CHECKPOINT,
        expected_checkpoint_sha256=B5_CHECKPOINT_SHA256,
        device=device,
        training_commit=B5_TRAINING_COMMIT,
    )
    model.eval()
    rows: list[dict[str, object]] = []
    noise_records: list[dict[str, object]] = []
    with B5ForecastArtifact(
        B5_FORECAST,
        expected_sha256=B5_FORECAST_SHA256,
        target_frames=tuple(range(498, 624)),
        seed_bank_path=B5_SEEDS,
        seed_bank_sha256=B5_SEEDS_SHA256,
        schema=B5ForecastSchema.frozen(),
    ) as artifact, torch.inference_mode():
        for target in SELECTED_TARGETS:
            mismatch = mismatched_target(target)
            target_index = target - 498
            correct = artifact.read(target_index, target_index + 1)[0, :8, 0]
            context = torch.from_numpy(
                np.asarray(validation_context_by_target[mismatch], dtype=np.float32)[None, None]
            ).to(device)
            mismatch_mean = torch.from_numpy(
                np.asarray(validation_h1_by_target[mismatch], dtype=np.float32)[None]
            ).to(device)
            original_mean = torch.from_numpy(
                np.asarray(validation_h1_by_target[target], dtype=np.float32)[None]
            ).to(device)
            condition = torch.cat((context[:, 0], mismatch_mean), dim=1)
            seeds = np.asarray(seed_bank[target_index, :8], dtype=np.uint64)
            noise = np.stack([initial_noise_from_uint64(seed) for seed in seeds])
            initial_noise = torch.from_numpy(noise[None]).to(device, torch.float32)
            normalized = model.sample_normalized(
                condition,
                initial_noise,
                steps=18,
                sigma_max=80.0,
                sigma_min=0.002,
                rho=7.0,
            )
            residual = model.denormalize_residual(normalized)
            shuffled = (original_mean[:, None] + residual)[0].detach().cpu().numpy()
            for record in _field_covariance_metrics(correct, shuffled):
                rows.append(
                    {
                        "target_frame": target,
                        "mismatched_context_target": mismatch,
                        "members": 8,
                        **record,
                    }
                )
            if member_metric_callback is not None:
                for record in member_metric_callback(correct, shuffled, target):
                    rows.append(
                        {
                            "target_frame": target,
                            "mismatched_context_target": mismatch,
                            "members": 8,
                            **dict(record),
                        }
                    )
            noise_records.append(
                {
                    "target_frame": target,
                    "seeds_uint64": [int(value) for value in seeds],
                    "noise_sha256": [_noise_digest(value) for value in noise],
                }
            )
    return rows, {
        "selected_targets": list(SELECTED_TARGETS),
        "mismatch_rule": "offset_63_modulo_126",
        "members_per_target": 8,
        "same_target_seed_prefix_used": True,
        "generated_residual_added_to_original_H1_center": True,
        "future_truth_used_for_generation": False,
        "noise_records": noise_records,
    }
