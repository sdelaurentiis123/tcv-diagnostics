"""Frozen-checkpoint toroidal equivariance audit (inference only)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from tcv_diagnostics.o2_forecast import load_selected_o2_model

from .data import H1_CHECKPOINT, H1_CHECKPOINT_SHA256, H1_CODEC, H1_CODEC_SHA256


FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
H1_TRAINING_COMMIT = "9035bc3ce9d2351cd17586f4429af8116d43a47e"


def load_frozen_h1(device: torch.device) -> torch.nn.Module:
    model = load_selected_o2_model(
        checkpoint=H1_CHECKPOINT,
        expected_checkpoint_sha256=H1_CHECKPOINT_SHA256,
        codec_checkpoint=H1_CODEC,
        expected_codec_sha256=H1_CODEC_SHA256,
        arm="C5P-H1",
        seed=1701,
        training_commit=H1_TRAINING_COMMIT,
        device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _relative_field_error(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, np.ndarray]:
    if reference.shape != candidate.shape or reference.ndim != 5:
        raise ValueError("equivariance comparison requires [batch,field,x,y,z]")
    numerator = torch.linalg.vector_norm((candidate - reference).flatten(2), dim=-1)
    denominator = torch.linalg.vector_norm(reference.flatten(2), dim=-1)
    values = (numerator / torch.clamp_min(denominator, torch.finfo(reference.dtype).tiny)).detach().cpu().numpy()
    return float(np.mean(values)), np.mean(values, axis=0)


def audit_frozen_h1_equivariance(
    contexts: np.ndarray,
    targets: np.ndarray,
    *,
    target_frames: Sequence[int],
    device: torch.device,
    shift_batch: int = 4,
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    """Audit every z shift while rolling every history frame together."""

    context_array = np.asarray(contexts, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    frames = tuple(int(value) for value in target_frames)
    if (
        context_array.ndim != 6
        or context_array.shape[1] != 1
        or target_array.shape != context_array[:, 0].shape
        or context_array.shape[0] != len(frames)
        or context_array.shape[-1] != 88
    ):
        raise ValueError("equivariance inputs differ from frozen H1 geometry")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 3.5 equivariance inference requires the Rusty H100")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_frozen_h1(device)
    rows: list[dict[str, float | int | str]] = []
    curve_records: dict[str, object] = {}
    with torch.inference_mode():
        for state_index, frame in enumerate(frames):
            context = torch.from_numpy(context_array[state_index : state_index + 1]).to(device)
            truth = torch.from_numpy(target_array[state_index : state_index + 1]).to(device)
            current = context[:, -1]
            base_reconstruction = model.codec.decode(model.codec.encode(current))
            base_forecast = model(context)
            per_scope_curves = {
                "codec": np.zeros(88, dtype=np.float64),
                "H1": np.zeros(88, dtype=np.float64),
            }
            for start in range(0, 88, int(shift_batch)):
                shifts = tuple(range(start, min(start + int(shift_batch), 88)))
                rolled_context = torch.cat([torch.roll(context, shift, dims=-1) for shift in shifts], dim=0)
                rolled_current = rolled_context[:, -1]
                rolled_truth = torch.cat([torch.roll(truth, shift, dims=-1) for shift in shifts], dim=0)
                codec_output = model.codec.decode(model.codec.encode(rolled_current))
                h1_output = model(rolled_context)
                for local, shift in enumerate(shifts):
                    codec_reference = torch.roll(base_reconstruction, shift, dims=-1)
                    h1_reference = torch.roll(base_forecast, shift, dims=-1)
                    for scope, reference, candidate in (
                        ("codec", codec_reference, codec_output[local : local + 1]),
                        ("H1", h1_reference, h1_output[local : local + 1]),
                    ):
                        aggregate, fields = _relative_field_error(reference, candidate)
                        per_scope_curves[scope][shift] = aggregate
                        forecast_aggregate, forecast_fields = _relative_field_error(
                            rolled_truth[local : local + 1], candidate
                        )
                        rows.append(
                            {
                                "target_frame": frame,
                                "scope": scope,
                                "field": "equal_field",
                                "shift_cells": shift,
                                "shift_modulo_4": shift % 4,
                                "equivariance_relative_error": aggregate,
                                "error_against_rolled_truth": forecast_aggregate,
                            }
                        )
                        for channel, field in enumerate(FIELDS):
                            rows.append(
                                {
                                    "target_frame": frame,
                                    "scope": scope,
                                    "field": field,
                                    "shift_cells": shift,
                                    "shift_modulo_4": shift % 4,
                                    "equivariance_relative_error": float(fields[channel]),
                                    "error_against_rolled_truth": float(forecast_fields[channel]),
                                }
                            )
            curve_records[str(frame)] = {}
            for scope, curve in per_scope_curves.items():
                amplitudes = np.abs(np.fft.rfft(curve - np.mean(curve), norm="ortho"))
                modulo_means = [float(np.mean(curve[np.arange(88) % 4 == value])) for value in range(4)]
                median = float(np.median(curve[1:]))
                curve_records[str(frame)][scope] = {
                    "curve": curve.tolist(),
                    "Fourier_amplitudes": amplitudes.tolist(),
                    "modulo_4_means": modulo_means,
                    "modulo_4_range_over_median_nonzero": (
                        (max(modulo_means) - min(modulo_means)) / median if median > 0.0 else None
                    ),
                }
    return rows, curve_records
