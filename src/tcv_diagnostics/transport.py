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


@dataclass(frozen=True)
class SingleNullTopology:
    """Logical connections needed by shifted ``DDY`` on a single-null mesh.

    ``separatrix_x_index`` is the first supplied radial cell outside the
    separatrix. The four ``y`` indices name the two nontrivial connections for
    cells inside the separatrix:

    - ``core_lower_y <-> core_upper_y`` (branch cut with ``ShiftAngle``);
    - ``pfr_lower_y <-> pfr_upper_y`` (private-flux connection, no twist).
    """

    separatrix_x_index: int
    core_lower_y: int
    core_upper_y: int
    pfr_lower_y: int
    pfr_upper_y: int


@dataclass(frozen=True)
class PartialShiftedYDerivative:
    """Shifted ``DDY`` values with physical-target cells explicitly masked."""

    values: np.ndarray
    valid_mask: np.ndarray
    minus_y_indices: np.ndarray
    plus_y_indices: np.ndarray
    component: str = "shifted_ddy_partial_physical_y_boundaries"


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


def spectral_shift_z(
    values: np.ndarray,
    shifts_radians: np.ndarray | float,
    *,
    zperiod: int = 5,
) -> np.ndarray:
    """Apply the audited BOUT++ Fourier shift along the last axis.

    For the stored wedge, a Fourier coefficient at stored index ``k`` is
    multiplied by ``exp(i * k * zperiod * shift)``. Positive ``shift`` thus
    returns the periodic field evaluated at ``z + shift``. ``shifts_radians``
    must broadcast over all axes except the final periodic axis.
    """

    array = _real_finite("spectral-shift input", values)
    if array.ndim < 1 or array.shape[-1] < 3:
        raise ValueError("spectral-shift input needs at least three z cells")
    if not isinstance(zperiod, (int, np.integer)) or zperiod <= 0:
        raise ValueError("zperiod must be a positive integer")
    shifts = _real_finite("spectral shifts", shifts_radians)
    try:
        shifts = np.broadcast_to(shifts, array.shape[:-1])
    except ValueError as error:
        raise ValueError(
            "spectral shifts must broadcast over every non-z input axis"
        ) from error

    coefficients = np.fft.rfft(array, axis=-1)
    stored_k = np.arange(coefficients.shape[-1], dtype=np.float64)
    phase = np.exp(
        1j * int(zperiod) * shifts[..., None] * stored_k.reshape(
            (1,) * shifts.ndim + (stored_k.size,)
        )
    )
    return np.fft.irfft(coefficients * phase, n=array.shape[-1], axis=-1)


def single_null_y_neighbors(
    n_x: int,
    n_y: int,
    topology: SingleNullTopology,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return logical lower/upper neighbors and the guard-independent mask."""

    if not isinstance(n_x, (int, np.integer)) or n_x < 1:
        raise ValueError("n_x must be a positive integer")
    if not isinstance(n_y, (int, np.integer)) or n_y < 3:
        raise ValueError("n_y must be an integer of at least three")
    if not 0 <= topology.separatrix_x_index <= n_x:
        raise ValueError("separatrix_x_index is outside the supplied radial grid")
    indices = (
        topology.pfr_lower_y,
        topology.core_lower_y,
        topology.core_upper_y,
        topology.pfr_upper_y,
    )
    if not (
        0 <= indices[0] < indices[1] <= indices[2] < indices[3] < n_y
    ):
        raise ValueError("single-null y connection indices are inconsistent")

    minus = np.broadcast_to(
        np.arange(n_y, dtype=np.int64)[None, :] - 1, (n_x, n_y)
    ).copy()
    plus = np.broadcast_to(
        np.arange(n_y, dtype=np.int64)[None, :] + 1, (n_x, n_y)
    ).copy()
    valid = np.ones((n_x, n_y), dtype=bool)
    valid[:, 0] = False
    valid[:, -1] = False
    minus[:, 0] = 0
    plus[:, -1] = n_y - 1

    inside = slice(0, topology.separatrix_x_index)
    plus[inside, topology.pfr_lower_y] = topology.pfr_upper_y
    minus[inside, topology.pfr_upper_y] = topology.pfr_lower_y
    minus[inside, topology.core_lower_y] = topology.core_upper_y
    plus[inside, topology.core_upper_y] = topology.core_lower_y
    return minus, plus, valid


def _gather_y(values: np.ndarray, y_indices: np.ndarray) -> np.ndarray:
    leading_singletons = (1,) * (values.ndim - 3)
    index_shape = (*leading_singletons, *y_indices.shape, 1)
    expanded_indices = np.broadcast_to(y_indices.reshape(index_shape), values.shape)
    return np.take_along_axis(values, expanded_indices, axis=-2)


def shifted_ddy_single_null_partial(
    potential: np.ndarray,
    z_shift: np.ndarray,
    dy: np.ndarray,
    shift_angle: np.ndarray,
    *,
    topology: SingleNullTopology,
    zperiod: int = 5,
) -> PartialShiftedYDerivative:
    """Evaluate BOUT++ shifted-metric ``DDY`` away from physical targets.

    The input has trailing axes ``[..., x, y, z]``. Geometry arrays ``z_shift``
    and ``dy`` have shape ``[x, y]`` and ``shift_angle`` has shape ``[x]``.
    The result is exact for logical neighbors supplied by ``topology`` but
    marks the two target-adjacent ``y`` cells invalid because the model dataset
    does not contain their physical-boundary guards.

    This source transcription remains a candidate until compared with the
    hash-locked BOUT++ executable oracle.
    """

    phi = _real_finite("potential", potential)
    if phi.ndim < 3:
        raise ValueError("potential must have trailing axes [x, y, z]")
    n_x, n_y, n_z = phi.shape[-3:]
    if n_z < 3:
        raise ValueError("periodic z needs at least three cells")
    z_shift_array = _cell_geometry(
        "z_shift", z_shift, n_x=n_x, n_y=n_y, require_positive=False
    )
    dy_array = _cell_geometry(
        "dy", dy, n_x=n_x, n_y=n_y, require_positive=True
    )
    shift_angle_array = _real_finite("shift_angle", shift_angle)
    if shift_angle_array.shape != (n_x,):
        raise ValueError(f"shift_angle must have shape {(n_x,)}")

    minus_y, plus_y, valid = single_null_y_neighbors(n_x, n_y, topology)
    minus_standard = _gather_y(phi, minus_y)
    plus_standard = _gather_y(phi, plus_y)

    x_indices = np.arange(n_x, dtype=np.int64)[:, None]
    minus_shift = z_shift_array[x_indices, minus_y]
    plus_shift = z_shift_array[x_indices, plus_y]
    if topology.separatrix_x_index > 0:
        inner_x = np.arange(topology.separatrix_x_index, dtype=np.int64)
        minus_shift[inner_x, topology.core_lower_y] -= shift_angle_array[inner_x]
        plus_shift[inner_x, topology.core_upper_y] += shift_angle_array[inner_x]

    minus_aligned = spectral_shift_z(
        minus_standard, minus_shift, zperiod=zperiod
    )
    plus_aligned = spectral_shift_z(plus_standard, plus_shift, zperiod=zperiod)
    derivative_aligned = 0.5 * (plus_aligned - minus_aligned)
    derivative_standard = spectral_shift_z(
        derivative_aligned, -z_shift_array, zperiod=zperiod
    )

    leading_singletons = (1,) * (phi.ndim - 3)
    geometry_shape = (*leading_singletons, n_x, n_y, 1)
    values = derivative_standard / dy_array.reshape(geometry_shape)
    mask_shape = (*leading_singletons, n_x, n_y, 1)
    values = np.where(valid.reshape(mask_shape), values, np.nan)
    return PartialShiftedYDerivative(
        values=values,
        valid_mask=valid,
        minus_y_indices=minus_y,
        plus_y_indices=plus_y,
    )


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
