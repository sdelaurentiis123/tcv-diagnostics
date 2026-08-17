"""Checkpoint and state-view contracts for matched Paper 0 O1 evaluation.

The functions here do not run the BOUT++ elliptic solve or transport operator.
They make the GPU reconstruction stage auditable: selected checkpoints must
match the frozen run, physical values are obtained only through the frozen
training normalization, and E6B is mapped to the common view without clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .codec_training import CodecRunConfig, sha256_path
from .matched_codec_metrics import CodecViewSpec
from .model_training_data import FAMILY_FIELDS, ModelNormalization, VOLUME_SHAPE
from .models import build_codec
from .resampling import periodic_resample_float32


COMMON_FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
COMMON_CROSS_PAIRS = (("Ne", "phi"), ("Pe", "phi"), ("Pi", "phi"))
NATIVE81_FIELDS = {
    "c5p": ("Ne", "Pe", "Pi", "phi"),
    "e6b": ("Ne", "Pe", "Pi", "Vort"),
}

C5P_VIEW = CodecViewSpec(
    name="c5p_native_and_common",
    fields=COMMON_FIELDS,
    spectral_fields=COMMON_FIELDS,
    cross_pairs=COMMON_CROSS_PAIRS,
)
E6B_NATIVE_VIEW = CodecViewSpec(
    name="e6b_native",
    fields=FAMILY_FIELDS["e6b"],
    spectral_fields=FAMILY_FIELDS["e6b"],
    cross_pairs=(),
)
E6B_COMMON_VIEW = CodecViewSpec(
    name="e6b_derived_common",
    fields=COMMON_FIELDS,
    spectral_fields=COMMON_FIELDS,
    cross_pairs=COMMON_CROSS_PAIRS,
)


@dataclass(frozen=True)
class SelectedCodecIdentity:
    checkpoint: str
    checkpoint_sha256: str
    training_result: str
    training_result_sha256: str
    training_commit: str
    codec: str
    family: str
    seed: int
    selected_epoch: int
    selected_global_step: int
    selected_validation_equal_channel_mae: float

    def to_record(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_result": self.training_result,
            "training_result_sha256": self.training_result_sha256,
            "training_commit": self.training_commit,
            "codec": self.codec,
            "family": self.family,
            "seed": self.seed,
            "selected_epoch": self.selected_epoch,
            "selected_global_step": self.selected_global_step,
            "selected_validation_equal_channel_mae": (
                self.selected_validation_equal_channel_mae
            ),
        }


def native_view_spec(family: str) -> CodecViewSpec:
    if family == "c5p":
        return C5P_VIEW
    if family == "e6b":
        return E6B_NATIVE_VIEW
    raise ValueError(f"unsupported state family {family!r}")


def validate_selected_checkpoint(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    payload: Mapping[str, Any],
    training_result_path: Path,
    training_result_sha256: str,
    training_result: Mapping[str, Any],
    codec: str,
    family: str,
    seed: int,
) -> SelectedCodecIdentity:
    """Require a selected checkpoint to match its completed frozen run."""

    actual_checkpoint_hash = sha256_path(checkpoint_path)
    if actual_checkpoint_hash != checkpoint_sha256:
        raise ValueError("selected checkpoint SHA-256 differs from the lock")
    actual_result_hash = sha256_path(training_result_path)
    if actual_result_hash != training_result_sha256:
        raise ValueError("training result SHA-256 differs from the lock")
    if training_result.get("scope") != "O1_codec_full":
        raise ValueError("training result is not a completed full O1 run")
    if training_result.get("held_out_85606_read") is not False:
        raise ValueError("training result does not preserve the 85606 blind lock")
    if training_result.get("development_run") != "85604":
        raise ValueError("training result is not for development run 85604")
    if training_result.get("completed_epochs") != 200:
        raise ValueError("training result did not complete the frozen budget")
    if training_result.get("physics_derived_loss_used") is not False:
        raise ValueError("training result reports a forbidden physics loss")
    selected_record = training_result.get("selected_checkpoint", {})
    if selected_record.get("sha256") != checkpoint_sha256:
        raise ValueError("training result and checkpoint lock disagree")

    expected = CodecRunConfig.frozen(
        mode="full",
        codec=codec,
        family=family,
        seed=seed,
    )
    expected_config = expected.to_record()
    if training_result.get("config") != expected_config:
        raise ValueError("training result configuration differs from the frozen run")
    if payload.get("config") != expected_config:
        raise ValueError("checkpoint configuration differs from the frozen run")
    if payload.get("kind") != "selected_model":
        raise ValueError("checkpoint is not a selected-model artifact")
    if "optimizer_state" in payload:
        raise ValueError("selected checkpoint unexpectedly contains optimizer state")
    if "model_state" not in payload or "reload_probe" not in payload:
        raise ValueError("selected checkpoint is incomplete")
    training_commit = str(training_result.get("paper0_commit", ""))
    if not training_commit or payload.get("paper0_commit") != training_commit:
        raise ValueError("checkpoint and training-result commits differ")

    epoch = int(payload.get("epoch", -1))
    global_step = int(payload.get("global_step", -1))
    if not 0 <= epoch < expected.epochs:
        raise ValueError("selected epoch is outside the frozen training budget")
    expected_step = (epoch + 1) * expected.optimizer_steps_per_epoch
    if global_step != expected_step:
        raise ValueError("selected checkpoint has an inconsistent global step")
    if epoch != int(training_result.get("selected_epoch", -1)):
        raise ValueError("selected epoch differs between checkpoint and result")
    selected_loss = float(
        training_result.get("selected_validation_equal_channel_mae", math.nan)
    )
    if (
        not math.isfinite(selected_loss)
        or float(payload.get("validation_loss")) != selected_loss
    ):
        raise ValueError("selected validation loss differs or is non-finite")
    if training_result.get("checkpoint_reload_bitwise_exact") is not True:
        raise ValueError("training did not pass exact checkpoint reload")

    return SelectedCodecIdentity(
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=actual_checkpoint_hash,
        training_result=str(training_result_path),
        training_result_sha256=actual_result_hash,
        training_commit=training_commit,
        codec=codec,
        family=family,
        seed=int(seed),
        selected_epoch=epoch,
        selected_global_step=global_step,
        selected_validation_equal_channel_mae=selected_loss,
    )


def restore_selected_codec(
    *,
    payload: Mapping[str, Any],
    codec: str,
    family: str,
    device: torch.device,
) -> nn.Module:
    """Build a codec and load only the validated selected model state."""

    model = build_codec(codec, len(FAMILY_FIELDS[family]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device=device, dtype=torch.float32)
    model.requires_grad_(False)
    model.eval()
    return model


def decode_physical_batch(
    normalization: ModelNormalization,
    fields: Sequence[str],
    standardized: np.ndarray,
) -> np.ndarray:
    """Invert frozen scalar normalization for a ``[T,C,X,Y,Z]`` batch."""

    array = np.asarray(standardized)
    expected_tail = (len(fields), *VOLUME_SHAPE)
    if array.ndim != 5 or array.shape[1:] != expected_tail:
        raise ValueError(
            f"standardized batch has shape {array.shape}, expected [T,{expected_tail}]"
        )
    return np.stack(
        [normalization.decode_volume(fields, frame) for frame in array],
        axis=0,
    )


def encode_physical_batch(
    normalization: ModelNormalization,
    fields: Sequence[str],
    physical: np.ndarray,
) -> np.ndarray:
    """Apply frozen scalar normalization to a ``[T,C,X,Y,Z]`` batch."""

    array = np.asarray(physical)
    expected_tail = (len(fields), *VOLUME_SHAPE)
    if array.ndim != 5 or array.shape[1:] != expected_tail:
        raise ValueError(
            f"physical batch has shape {array.shape}, expected [T,{expected_tail}]"
        )
    return np.stack(
        [
            normalization.encode_volume(fields, list(frame))
            for frame in array
        ],
        axis=0,
    )


def derive_e6b_common_physical(
    e6b_physical: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Map E6B to ``[Ne,Pe,Pi,phi,Vi]`` without floors or clipping."""

    state = np.asarray(e6b_physical, dtype=np.float64)
    potential = np.asarray(phi, dtype=np.float64)
    expected = (state.shape[0], *VOLUME_SHAPE) if state.ndim == 5 else None
    if state.ndim != 5 or state.shape[1:] != (6, *VOLUME_SHAPE):
        raise ValueError("E6B state must have axes [T,6,64,32,88]")
    if potential.shape != expected:
        raise ValueError("derived potential must have axes [T,64,32,88]")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(potential)):
        raise ValueError("E6B common-view inputs contain non-finite values")
    density = state[:, 0]
    if np.any(density <= 0.0):
        raise ValueError("E6B common view refuses non-positive decoded density")
    ion_velocity = state[:, 4] / (2.0 * density)
    result = np.stack(
        [density, state[:, 1], state[:, 2], potential, ion_velocity],
        axis=1,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("E6B common-view derivation produced non-finite values")
    return result


def native81_candidate_fields(
    family: str,
    physical_reconstruction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Resample the downstream elliptic/transport inputs from 88 to 81 points."""

    fields = FAMILY_FIELDS.get(family)
    if fields is None:
        raise ValueError(f"unsupported state family {family!r}")
    state = np.asarray(physical_reconstruction)
    if state.ndim != 5 or state.shape[1:] != (len(fields), *VOLUME_SHAPE):
        raise ValueError("physical reconstruction has the wrong state shape")
    indices = {field: index for index, field in enumerate(fields)}
    return {
        field: periodic_resample_float32(
            state[:, indices[field]],
            81,
            axis=-1,
        )
        for field in NATIVE81_FIELDS[family]
    }
