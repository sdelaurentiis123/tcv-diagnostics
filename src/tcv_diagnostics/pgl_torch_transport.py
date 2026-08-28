"""Differentiable transcription of authoritative separatrix transport.

The implementation is intentionally narrow: it decodes C5P, applies the
frozen periodic model-88 to native-81 resampling, and returns member-wise local
contributions on the confined separatrix. The NumPy authority in
``transport.py`` and ``codec_transport.py`` remains the scientific reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .codec_transport import CodecTransportGeometry
from .model_training_data import ModelNormalization
from .resampling import periodic_resample_float32
from .transport import single_null_y_neighbors


PGL_TRANSPORT_QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)


@dataclass(frozen=True)
class TorchDecoderRecord:
    mean: float
    standard_deviation: float
    transform: str
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("Torch decoder mean must be finite")
        if not math.isfinite(self.standard_deviation) or self.standard_deviation <= 0:
            raise ValueError("Torch decoder scale must be positive")
        if self.transform not in ("identity", "log_offset"):
            raise ValueError("Torch decoder transform differs")
        if self.transform == "log_offset" and (
            not math.isfinite(self.offset) or self.offset <= 0.0
        ):
            raise ValueError("Torch log-offset decoder requires a positive offset")

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "TorchDecoderRecord":
        transform = dict(record["transform"])
        return cls(
            mean=float(record["mean"]),
            standard_deviation=float(record["population_standard_deviation"]),
            transform=str(transform["name"]),
            offset=float(transform.get("offset", 0.0)),
        )

    def decode(self, values: Tensor) -> Tensor:
        transformed = values * self.standard_deviation + self.mean
        if self.transform == "log_offset":
            return torch.exp(transformed) - self.offset
        return transformed


def decoder_records_from_normalization(
    normalization: ModelNormalization,
) -> tuple[TorchDecoderRecord, ...]:
    records: list[TorchDecoderRecord] = []
    for field in ("Ne", "Pe", "Pi", "phi", "Vi"):
        value = normalization.records[field]
        records.append(
            TorchDecoderRecord(
                mean=float(value.mean),
                standard_deviation=float(value.standard_deviation),
                transform=str(value.transform["name"]),
                offset=float(value.transform.get("offset", 0.0)),
            )
        )
    return tuple(records)


def periodic_resample_matrix(source: int = 88, target: int = 81) -> np.ndarray:
    """Return the frozen SciPy resampler as a differentiable linear matrix."""

    if source < 2 or target < 2:
        raise ValueError("periodic resampling sizes must be at least two")
    basis = np.eye(int(source), dtype=np.float64)
    matrix = periodic_resample_float32(basis, int(target), axis=1)
    if matrix.shape != (source, target) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("periodic resampling matrix differs")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def resample_matrix_sha256(matrix: np.ndarray) -> str:
    values = np.ascontiguousarray(matrix, dtype="<f4")
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("resampling matrix must be finite and two-dimensional")
    return hashlib.sha256(values.tobytes()).hexdigest()


def apply_periodic_resample(values: Tensor, matrix: Tensor) -> Tensor:
    """Apply a fixed last-axis periodic resampling matrix."""

    if values.ndim < 1 or matrix.ndim != 2 or values.shape[-1] != matrix.shape[0]:
        raise ValueError("periodic Torch resampling shapes differ")
    if not torch.isfinite(values).all() or not torch.isfinite(matrix).all():
        raise ValueError("periodic Torch resampling inputs must be finite")
    return torch.matmul(values, matrix.to(device=values.device, dtype=values.dtype))


def _spectral_shift_z(values: Tensor, shifts: Tensor, *, zperiod: int) -> Tensor:
    if values.ndim < 1 or values.shape[:-1] != shifts.shape:
        raise ValueError("Torch spectral shift shape differs")
    if not torch.isfinite(values).all() or not torch.isfinite(shifts).all():
        raise ValueError("Torch spectral shift inputs must be finite")
    original_dtype = values.dtype
    coefficients = torch.fft.rfft(values.float(), dim=-1)
    modes = torch.arange(
        coefficients.shape[-1], device=values.device, dtype=torch.float32
    )
    phase_angle = float(zperiod) * shifts.float()[..., None] * modes
    phase = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle))
    shifted = torch.fft.irfft(
        coefficients * phase, n=values.shape[-1], dim=-1
    )
    return shifted.to(original_dtype)


def _gather_y(values: Tensor, indices: Tensor) -> Tensor:
    if values.ndim < 3 or indices.shape != values.shape[-3:-1]:
        raise ValueError("Torch shifted-y gather shapes differ")
    view = (1,) * (values.ndim - 3) + (*indices.shape, 1)
    expanded = indices.reshape(view).expand(*values.shape[:-3], *indices.shape, values.shape[-1])
    return torch.gather(values, -2, expanded)


def _mc_slope(minus: Tensor, center: Tensor, plus: Tensor) -> Tensor:
    first = 2.0 * (plus - center)
    second = 0.5 * (plus - minus)
    third = 2.0 * (center - minus)
    same = (first * second > 0.0) & (first * third > 0.0)
    magnitude = torch.minimum(
        torch.abs(first), torch.minimum(torch.abs(second), torch.abs(third))
    )
    return torch.where(same, torch.sign(first) * magnitude, torch.zeros_like(first))


class TorchSeparatrixTransport(nn.Module):
    """Member-wise C5P-to-local-separatrix transport with no trainable state."""

    def __init__(
        self,
        geometry: CodecTransportGeometry,
        decoder_records: Sequence[TorchDecoderRecord],
        *,
        source_z: int = 88,
        zperiod: int = 5,
        resample_matrix: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        records = tuple(decoder_records)
        if len(records) != 5:
            raise ValueError("Torch transport requires five C5P decoder records")
        if geometry.jacobian.ndim != 2 or geometry.jacobian.shape[1] < 3:
            raise ValueError("Torch transport geometry differs")
        self.decoder_records = records
        self.zperiod = int(zperiod)
        if self.zperiod != 5:
            raise ValueError("Paper 0 Torch transport requires zperiod five")
        matrix = (
            periodic_resample_matrix(source_z, int(round(2.0 * math.pi / (geometry.dz * zperiod))))
            if resample_matrix is None
            else np.asarray(resample_matrix, dtype=np.float32)
        )
        if matrix.ndim != 2 or matrix.shape[0] != source_z:
            raise ValueError("Torch transport resampling matrix differs")
        self.source_z = int(source_z)
        self.native_z = int(matrix.shape[1])
        self.resample_sha256 = resample_matrix_sha256(matrix)
        self.register_buffer("resample", torch.from_numpy(np.ascontiguousarray(matrix)))
        for name in ("jacobian", "g11", "g23", "bxy", "z_shift", "dy"):
            values = np.asarray(getattr(geometry, name), dtype=np.float64)
            self.register_buffer(name, torch.from_numpy(np.ascontiguousarray(values)))
        shift = np.asarray(geometry.shift_angle, dtype=np.float64).copy()
        shift[geometry.topology.separatrix_x_index :] = 0.0
        if not np.all(np.isfinite(shift)):
            raise ValueError("used Torch transport shift angles are non-finite")
        self.register_buffer("shift_angle", torch.from_numpy(shift))
        minus, plus, valid = single_null_y_neighbors(
            geometry.jacobian.shape[0],
            geometry.jacobian.shape[1],
            geometry.topology,
        )
        self.register_buffer("minus_y", torch.from_numpy(minus.astype(np.int64)))
        self.register_buffer("plus_y", torch.from_numpy(plus.astype(np.int64)))
        self.register_buffer("valid_y", torch.from_numpy(valid))
        face_positions = np.flatnonzero(
            geometry.left_cell_indices
            == geometry.region_masks.separatrix_face_left_cell_index
        )
        if face_positions.size != 1:
            raise ValueError("Torch transport separatrix face is not unique")
        face_position = int(face_positions[0])
        selected_rows = np.flatnonzero(geometry.separatrix_face_mask[face_position])
        expected = np.arange(
            geometry.topology.core_lower_y,
            geometry.topology.core_upper_y + 1,
            dtype=np.int64,
        )
        if not np.array_equal(selected_rows, expected):
            raise ValueError("Torch transport separatrix rows differ")
        self.face_left = int(geometry.left_cell_indices[face_position])
        self.register_buffer("separatrix_rows", torch.from_numpy(selected_rows))
        self.dz = float(geometry.dz)
        self.topology = geometry.topology

    def _decode(self, standardized: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if standardized.ndim != 7 or standardized.shape[3] != 5:
            raise ValueError("Torch transport input must be [B,M,K,5,x,y,z]")
        if standardized.shape[-1] != self.source_z or not torch.isfinite(standardized).all():
            raise ValueError("Torch transport standardized trajectory differs")
        decoded = [
            record.decode(standardized[:, :, :, index].float())
            for index, record in enumerate(self.decoder_records)
        ]
        native = [
            apply_periodic_resample(value, self.resample.float())
            for value in decoded[:4]
        ]
        if not all(torch.isfinite(value).all() for value in native):
            raise FloatingPointError("Torch decoded/resampled fields are non-finite")
        return native[0], native[1], native[2], native[3]

    def _shifted_ddy(self, potential: Tensor) -> Tensor:
        n_x, n_y, _ = potential.shape[-3:]
        if (n_x, n_y) != tuple(self.jacobian.shape):
            raise ValueError("Torch potential and geometry differ")
        minus = _gather_y(potential, self.minus_y)
        plus = _gather_y(potential, self.plus_y)
        minus_shift = torch.gather(self.z_shift, 1, self.minus_y).clone()
        plus_shift = torch.gather(self.z_shift, 1, self.plus_y).clone()
        inner = torch.arange(
            self.topology.separatrix_x_index, device=potential.device
        )
        minus_shift[inner, self.topology.core_lower_y] -= self.shift_angle[inner]
        plus_shift[inner, self.topology.core_upper_y] += self.shift_angle[inner]
        leading = (1,) * (potential.ndim - 3)
        minus_shift_full = minus_shift.reshape(*leading, n_x, n_y).expand(
            *potential.shape[:-3], n_x, n_y
        )
        plus_shift_full = plus_shift.reshape(*leading, n_x, n_y).expand_as(
            minus_shift_full
        )
        aligned_minus = _spectral_shift_z(
            minus, minus_shift_full, zperiod=self.zperiod
        )
        aligned_plus = _spectral_shift_z(
            plus, plus_shift_full, zperiod=self.zperiod
        )
        aligned_derivative = 0.5 * (aligned_plus - aligned_minus)
        center_shift = self.z_shift.reshape(*leading, n_x, n_y).expand_as(
            minus_shift_full
        )
        standard = _spectral_shift_z(
            aligned_derivative, -center_shift, zperiod=self.zperiod
        )
        dy = self.dy.reshape(*leading, n_x, n_y, 1)
        return standard / dy

    def _local_face_flow(self, advected: Tensor, potential: Tensor) -> Tensor:
        if advected.shape != potential.shape:
            raise ValueError("Torch advected/potential shapes differ")
        left = self.face_left
        right = left + 1
        # Source-matched x-z component: MC states and periodic corner velocity.
        left_center = advected[..., left, :, :]
        left_state = left_center + 0.5 * _mc_slope(
            advected[..., left - 1, :, :],
            left_center,
            advected[..., left + 1, :, :],
        )
        right_center = advected[..., right, :, :]
        right_state = right_center - 0.5 * _mc_slope(
            advected[..., right - 1, :, :],
            right_center,
            advected[..., right + 1, :, :],
        )
        phi_left = potential[..., left, :, :]
        phi_right = potential[..., right, :, :]
        corner_plus = 0.25 * (
            phi_left
            + torch.roll(phi_left, -1, dims=-1)
            + phi_right
            + torch.roll(phi_right, -1, dims=-1)
        )
        corner_minus = 0.25 * (
            phi_left
            + torch.roll(phi_left, 1, dims=-1)
            + phi_right
            + torch.roll(phi_right, 1, dims=-1)
        )
        face_jacobian = 0.5 * (self.jacobian[left] + self.jacobian[right])
        face_shape = (1,) * (advected.ndim - 3) + (face_jacobian.numel(), 1)
        face_jacobian_view = face_jacobian.reshape(face_shape)
        xz_velocity = face_jacobian_view * (
            corner_plus - corner_minus
        ) / self.dz
        xz_state = torch.where(xz_velocity > 0.0, left_state, right_state)
        xz_flow = xz_velocity * xz_state

        # Source-matched shifted-x-y component: Fromm positive states.
        derivative = self._shifted_ddy(potential)
        metric = self.g11 * self.g23 / self.bxy.square()
        metric_shape = (1,) * (advected.ndim - 3) + (metric.shape[1], 1)
        xy_derivative = 0.5 * (
            metric[left].reshape(metric_shape) * derivative[..., left, :, :]
            + metric[right].reshape(metric_shape) * derivative[..., right, :, :]
        )
        xy_velocity = face_jacobian_view * xy_derivative
        fromm_left = torch.clamp_min(
            advected[..., left, :, :]
            + 0.25
            * (advected[..., left + 1, :, :] - advected[..., left - 1, :, :]),
            0.0,
        )
        fromm_right = torch.clamp_min(
            advected[..., right, :, :]
            - 0.25
            * (advected[..., right + 1, :, :] - advected[..., right - 1, :, :]),
            0.0,
        )
        xy_state = torch.where(xy_velocity > 0.0, fromm_left, fromm_right)
        return xz_flow + xy_velocity * xy_state

    def forward(self, standardized: Tensor) -> Tensor:
        """Return ``[B,M,K,quantity,16,native_z]`` weighted contributions."""

        ne, pe, pi, phi = self._decode(standardized)
        rows = self.separatrix_rows
        weight = (self.dy[self.face_left].index_select(0, rows) * self.dz).float()
        outputs: list[Tensor] = []
        for advected, factor in ((ne, 1.0), (pe, 1.5), (pi, 1.5)):
            flow = self._local_face_flow(advected, phi)
            selected = flow.index_select(-2, rows)
            outputs.append(float(factor) * selected * weight[None, :, None])
        outputs.append(outputs[1] + outputs[2])
        result = torch.stack(outputs, dim=3)
        expected = (*standardized.shape[:3], 4, rows.numel(), self.native_z)
        if result.shape != expected or not torch.isfinite(result).all():
            raise FloatingPointError("Torch separatrix transport output differs")
        return result

    def to_record(self) -> dict[str, Any]:
        return {
            "quantities": list(PGL_TRANSPORT_QUANTITIES),
            "source_z": self.source_z,
            "native_z": self.native_z,
            "zperiod": self.zperiod,
            "resample_matrix_sha256": self.resample_sha256,
            "face_left_model_x": self.face_left,
            "separatrix_rows": self.separatrix_rows.cpu().tolist(),
            "memberwise_before_reduction": True,
            "physical_SI_multiplier_applied": False,
            "output": "normalized_weighted_local_wedge_contribution",
        }
