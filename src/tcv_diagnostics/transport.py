"""Partial, source-matched transport primitives for the Paper 0 oracle.

Only the radial ``x-z`` face-flow component of the executed Hermes-3
``Div_n_bxGrad_f_B_XPPM`` operator is implemented here.  The full radial ExB
flow also contains a shifted-field-line ``x-y`` component because the run used
``poloidal_flows = true``.  Accordingly, every public result and function in
this module is explicitly named ``partial`` or ``xz_component`` and must not be
reported as total particle or energy transport.

The frozen definition and release blockers are in
``paper0/protocol/PHASE2_TRANSPORT_PROTOCOL.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PartialRadialFaceFlow:
    """Radial ``x-z`` face flow on the guard-independent interior faces.

    ``flow`` and ``velocity_factor`` have axes ``[..., face, y, z]``.
    ``left_cell_indices[f]`` identifies the radial cell immediately to the
    left of face ``f`` in the supplied field array.
    """

    flow: np.ndarray
    velocity_factor: np.ndarray
    left_cell_indices: np.ndarray
    component: str = "radial_exb_xz_component_partial"


@dataclass(frozen=True)
class PartialRadialDivergence:
    """Divergence implied by consecutive partial radial face flows."""

    divergence: np.ndarray
    cell_indices: np.ndarray
    component: str = "radial_exb_xz_component_partial"


def _real_finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def toroidal_wedge_spacing(n_z: int, *, zperiod: int = 5) -> float:
    """Return ``2*pi/(zperiod*n_z)`` for the periodic simulated wedge."""

    if not isinstance(n_z, (int, np.integer)) or n_z < 3:
        raise ValueError("n_z must be an integer of at least three")
    if not isinstance(zperiod, (int, np.integer)) or zperiod <= 0:
        raise ValueError("zperiod must be a positive integer")
    return 2.0 * math.pi / (int(zperiod) * int(n_z))


def monotonized_central_slope(
    minus: np.ndarray,
    center: np.ndarray,
    plus: np.ndarray,
) -> np.ndarray:
    """Return the executed Hermes Monotonized-Central reconstruction slope.

    This implements ``minmod(2*(plus-center), 0.5*(plus-minus),
    2*(center-minus))``.  A zero or a sign disagreement returns zero.
    """

    minus_array = _real_finite("MC minus", minus)
    center_array = _real_finite("MC center", center)
    plus_array = _real_finite("MC plus", plus)
    try:
        minus_array, center_array, plus_array = np.broadcast_arrays(
            minus_array, center_array, plus_array
        )
    except ValueError as error:
        raise ValueError("MC stencil arrays are not broadcast-compatible") from error

    first = 2.0 * (plus_array - center_array)
    second = 0.5 * (plus_array - minus_array)
    third = 2.0 * (center_array - minus_array)
    same_nonzero_sign = (first * second > 0.0) & (first * third > 0.0)
    magnitude = np.minimum(
        np.abs(first), np.minimum(np.abs(second), np.abs(third))
    )
    return np.where(same_nonzero_sign, np.sign(first) * magnitude, 0.0)


def mc_radial_face_states_partial(
    advected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source-matched left/right MC states on safe radial faces.

    The input axes are ``[..., x, y, z]``.  Because the model dataset omits
    radial guard cells, only faces whose two possible upwind stencils are
    entirely present are returned.  For ``X`` supplied cells these faces have
    left-cell indices ``1 .. X-3``.
    """

    q = _real_finite("advected field", advected)
    if q.ndim < 3:
        raise ValueError("advected field must have trailing axes [x, y, z]")
    n_x, n_y, n_z = q.shape[-3:]
    if n_x < 4:
        raise ValueError("at least four radial cells are required")
    if n_y < 1 or n_z < 3:
        raise ValueError("y must be nonempty and periodic z needs at least three cells")

    left_indices = np.arange(1, n_x - 2, dtype=np.int64)
    left_center = np.take(q, left_indices, axis=-3)
    left_slope = monotonized_central_slope(
        np.take(q, left_indices - 1, axis=-3),
        left_center,
        np.take(q, left_indices + 1, axis=-3),
    )
    state_from_left = left_center + 0.5 * left_slope

    right_indices = left_indices + 1
    right_center = np.take(q, right_indices, axis=-3)
    right_slope = monotonized_central_slope(
        np.take(q, right_indices - 1, axis=-3),
        right_center,
        np.take(q, right_indices + 1, axis=-3),
    )
    state_from_right = right_center - 0.5 * right_slope
    return state_from_left, state_from_right, left_indices


def _cell_geometry(
    name: str,
    values: np.ndarray | float,
    *,
    n_x: int,
    n_y: int,
    require_positive: bool,
) -> np.ndarray:
    array = _real_finite(name, values)
    if array.ndim == 0:
        array = np.full((n_x, n_y), float(array), dtype=np.float64)
    elif array.shape != (n_x, n_y):
        raise ValueError(
            f"{name} must be scalar or have shape {(n_x, n_y)}, got {array.shape}"
        )
    if require_positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def radial_exb_xz_face_flow_partial(
    advected: np.ndarray,
    potential: np.ndarray,
    jacobian: np.ndarray,
    *,
    dz: np.ndarray | float,
) -> PartialRadialFaceFlow:
    """Evaluate only the conservative radial ``x-z`` ExB face-flow term.

    Inputs have trailing axes ``[..., x, y, z]``.  ``jacobian`` must have
    shape ``[x, y]`` and ``dz`` may be scalar or ``[x, y]``.  The calculation
    uses float64, periodic ``z`` corners, the source-matched MC upwind state,
    and the exact Jacobian placement of the audited Hermes revision.

    This is deliberately not a total transport metric: the shifted-poloidal
    contribution is absent.
    """

    q = _real_finite("advected field", advected)
    phi = _real_finite("potential", potential)
    if q.shape != phi.shape:
        raise ValueError(
            f"advected field and potential shapes differ: {q.shape} versus {phi.shape}"
        )
    if q.ndim < 3:
        raise ValueError("fields must have trailing axes [x, y, z]")
    n_x, n_y, n_z = q.shape[-3:]
    if n_z < 3:
        raise ValueError("periodic z needs at least three cells")

    jacobian_array = _cell_geometry(
        "jacobian", jacobian, n_x=n_x, n_y=n_y, require_positive=False
    )
    dz_array = _cell_geometry(
        "dz", dz, n_x=n_x, n_y=n_y, require_positive=True
    )
    state_from_left, state_from_right, left_indices = (
        mc_radial_face_states_partial(q)
    )

    phi_left = np.take(phi, left_indices, axis=-3)
    phi_right = np.take(phi, left_indices + 1, axis=-3)
    corner_plus = 0.25 * (
        phi_left
        + np.roll(phi_left, -1, axis=-1)
        + phi_right
        + np.roll(phi_right, -1, axis=-1)
    )
    corner_minus = 0.25 * (
        phi_left
        + np.roll(phi_left, 1, axis=-1)
        + phi_right
        + np.roll(phi_right, 1, axis=-1)
    )

    face_jacobian = 0.5 * (
        jacobian_array[left_indices] + jacobian_array[left_indices + 1]
    )
    face_dz = dz_array[left_indices]
    leading_singletons = (1,) * (q.ndim - 3)
    face_geometry_shape = (*leading_singletons, left_indices.size, n_y, 1)
    velocity_factor = (
        face_jacobian.reshape(face_geometry_shape)
        * (corner_plus - corner_minus)
        / face_dz.reshape(face_geometry_shape)
    )
    upwind_state = np.where(
        velocity_factor > 0.0, state_from_left, state_from_right
    )
    return PartialRadialFaceFlow(
        flow=velocity_factor * upwind_state,
        velocity_factor=velocity_factor,
        left_cell_indices=left_indices,
    )


def divergence_from_xz_face_flow_partial(
    face_result: PartialRadialFaceFlow,
    jacobian: np.ndarray,
    *,
    dx: np.ndarray | float,
) -> PartialRadialDivergence:
    """Return the volume-normalized divergence of consecutive partial faces."""

    flow = _real_finite("partial radial face flow", face_result.flow)
    if flow.ndim < 3:
        raise ValueError("face flow must have trailing axes [face, y, z]")
    indices = np.asarray(face_result.left_cell_indices)
    if indices.ndim != 1 or indices.size != flow.shape[-3]:
        raise ValueError("left-cell indices do not match the face axis")
    if indices.size < 2 or not np.array_equal(
        np.diff(indices), np.ones(indices.size - 1, dtype=np.int64)
    ):
        raise ValueError("at least two consecutive radial faces are required")

    jacobian_array = _real_finite("jacobian", jacobian)
    if jacobian_array.ndim != 2 or jacobian_array.shape[1] != flow.shape[-2]:
        raise ValueError("jacobian must have shape [x, y] matching the face flow")
    n_x, n_y = jacobian_array.shape
    dx_array = _cell_geometry(
        "dx", dx, n_x=n_x, n_y=n_y, require_positive=True
    )
    cell_indices = indices[1:]
    if cell_indices[-1] >= n_x:
        raise ValueError("face indices exceed the supplied geometry")
    denominator = jacobian_array[cell_indices] * dx_array[cell_indices]
    if np.any(denominator == 0.0):
        raise ValueError("jacobian times dx must be nonzero")
    leading_singletons = (1,) * (flow.ndim - 3)
    denominator_shape = (*leading_singletons, cell_indices.size, n_y, 1)
    divergence = (flow[..., 1:, :, :] - flow[..., :-1, :, :]) / denominator.reshape(
        denominator_shape
    )
    return PartialRadialDivergence(
        divergence=divergence,
        cell_indices=cell_indices,
    )
