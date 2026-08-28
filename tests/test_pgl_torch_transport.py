from __future__ import annotations

import math

import numpy as np
import torch

from tcv_diagnostics.codec_transport import build_codec_transport_geometry
from tcv_diagnostics.pgl_torch_transport import (
    TorchDecoderRecord,
    TorchSeparatrixTransport,
    apply_periodic_resample,
    periodic_resample_matrix,
    resample_matrix_sha256,
)
from tcv_diagnostics.resampling import periodic_resample_float32
from tcv_diagnostics.transport import (
    SingleNullTopology,
    radial_exb_face_flow_partial,
    toroidal_wedge_spacing,
)


def synthetic_geometry(n_x: int = 8, n_y: int = 8, n_z: int = 7):
    x = np.arange(n_x, dtype=np.float64)[:, None]
    y = np.arange(n_y, dtype=np.float64)[None, :]
    jacobian = 1.2 + 0.01 * x + 0.02 * y
    g11 = 0.9 + 0.005 * x + 0.002 * y
    g23 = 0.2 + 0.001 * x - 0.002 * y
    bxy = 1.1 + 0.002 * x + 0.003 * y
    z_shift = 0.01 * x + 0.003 * y
    dy = 0.04 + 0.001 * x + 0.0005 * y
    penalty = np.zeros((n_x, n_y), dtype=np.float64)
    radius = np.asarray([0.6, 0.7, 0.8, 0.95, 1.1, 0.9, 0.75, 0.65])
    topology = SingleNullTopology(
        separatrix_x_index=4,
        core_lower_y=2,
        core_upper_y=5,
        pfr_lower_y=1,
        pfr_upper_y=6,
    )
    shift_angle = np.full(n_x, np.nan)
    shift_angle[:4] = np.asarray([0.04, 0.05, 0.06, 0.07])
    return build_codec_transport_geometry(
        jacobian=jacobian,
        g11=g11,
        g23=g23,
        bxy=bxy,
        z_shift=z_shift,
        dy=dy,
        shift_angle=shift_angle,
        penalty_mask=penalty,
        separatrix_face_major_radius=radius,
        dz=toroidal_wedge_spacing(n_z, zperiod=5),
        topology=topology,
    )


def identity_decoders() -> tuple[TorchDecoderRecord, ...]:
    return tuple(
        TorchDecoderRecord(0.0, 1.0, "identity") for _ in range(5)
    )


def test_resample_matrix_matches_frozen_scipy_transform() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(5, 88)).astype(np.float32)
    matrix = periodic_resample_matrix(88, 81)
    observed = apply_periodic_resample(
        torch.from_numpy(values), torch.from_numpy(matrix)
    ).numpy()
    expected = periodic_resample_float32(values, 81, axis=-1)
    assert np.allclose(observed, expected, atol=2e-6, rtol=0.0)
    assert len(resample_matrix_sha256(matrix)) == 64


def test_log_offset_decoder_matches_definition_and_has_gradient() -> None:
    record = TorchDecoderRecord(0.4, 0.7, "log_offset", offset=0.2)
    values = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    observed = record.decode(values)
    expected = torch.exp(values * 0.7 + 0.4) - 0.2
    assert torch.equal(observed, expected)
    observed.sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def numpy_local_reference(values: np.ndarray, geometry) -> np.ndarray:
    native = periodic_resample_float32(values[..., :4, :, :, :], 7, axis=-1)
    state = {
        "Ne": native[..., 0, :, :, :],
        "Pe": native[..., 1, :, :, :],
        "Pi": native[..., 2, :, :, :],
        "phi": native[..., 3, :, :, :],
    }
    face_position = int(
        np.flatnonzero(
            geometry.left_cell_indices
            == geometry.region_masks.separatrix_face_left_cell_index
        )[0]
    )
    rows = np.flatnonzero(geometry.separatrix_face_mask[face_position])
    outputs = []
    for field, factor in (("Ne", 1.0), ("Pe", 1.5), ("Pi", 1.5)):
        result = radial_exb_face_flow_partial(
            state[field],
            state["phi"],
            geometry.jacobian,
            geometry.g11,
            geometry.g23,
            geometry.bxy,
            geometry.z_shift,
            geometry.dy,
            geometry.shift_angle,
            dz=geometry.dz,
            topology=geometry.topology,
            zperiod=5,
            positive=True,
        )
        weight = geometry.dy[result.left_cell_indices[face_position], rows] * geometry.dz
        outputs.append(
            factor
            * result.flow[..., face_position, rows, :]
            * weight.reshape((1,) * (result.flow.ndim - 3) + (rows.size, 1))
        )
    outputs.append(outputs[1] + outputs[2])
    return np.stack(outputs, axis=3)


def test_torch_transport_matches_numpy_authority_and_is_gauge_invariant() -> None:
    geometry = synthetic_geometry()
    operator = TorchSeparatrixTransport(
        geometry,
        identity_decoders(),
        source_z=8,
        resample_matrix=periodic_resample_matrix(8, 7),
    )
    rng = np.random.default_rng(8)
    values = rng.normal(scale=0.1, size=(1, 2, 4, 5, 8, 8, 8)).astype(np.float32)
    values[..., 0, :, :, :] += 1.2
    values[..., 1, :, :, :] += 1.0
    values[..., 2, :, :, :] += 0.9
    observed = operator(torch.from_numpy(values)).detach().numpy()
    expected = numpy_local_reference(values, geometry)
    relative = np.linalg.norm(observed - expected) / np.linalg.norm(expected)
    assert relative < 2e-5
    assert np.max(np.abs(observed - expected)) < 2e-5

    shifted = values.copy()
    shifted[..., 3, :, :, :] += 13.0
    shifted_output = operator(torch.from_numpy(shifted)).detach().numpy()
    assert np.allclose(shifted_output, observed, atol=2e-5, rtol=0.0)


def test_torch_transport_backpropagates_memberwise() -> None:
    geometry = synthetic_geometry()
    operator = TorchSeparatrixTransport(
        geometry,
        identity_decoders(),
        source_z=8,
        resample_matrix=periodic_resample_matrix(8, 7),
    )
    torch.manual_seed(10)
    values = (0.1 * torch.randn(1, 2, 4, 5, 8, 8, 8)).requires_grad_(True)
    with torch.no_grad():
        values[..., 0, :, :, :].add_(1.2)
        values[..., 1, :, :, :].add_(1.0)
        values[..., 2, :, :, :].add_(0.9)
    output = operator(values)
    assert output.shape == (1, 2, 4, 4, 4, 7)
    output.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad[..., (0, 1, 2, 3), :, :, :]) > 0
