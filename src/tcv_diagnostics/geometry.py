"""Prospectively frozen single-null geometry masks for Paper 0.

The definitions are fixed in
``paper0/protocol/PHASE2_GEOMETRY_UNITS_PROTOCOL.md``.  They operate on the
64-by-32 model crop, never infer plasma regions from image coordinates, and
do not represent experimental diagnostic response functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transport import SingleNullTopology


@dataclass(frozen=True)
class SingleNullRegionMasks:
    """Cell masks and frozen landmark indices on a cropped single-null grid."""

    strict_wall_interior: np.ndarray
    wall_crossing: np.ndarray
    wall_exterior: np.ndarray
    operator_interior: np.ndarray
    confined_edge: np.ndarray
    private_flux: np.ndarray
    scrape_off_layer: np.ndarray
    separatrix_cell_band: np.ndarray
    outboard_midplane: np.ndarray
    x_point_topology_stencil: np.ndarray
    inner_divertor_leg: np.ndarray
    outer_divertor_leg: np.ndarray
    separatrix_x_index: int
    separatrix_face_left_cell_index: int
    core_lower_y: int
    core_upper_y: int
    outboard_midplane_y: int

    def primary_partition(self) -> np.ndarray:
        """Return the integer multiplicity of the three disjoint regions."""

        return (
            self.confined_edge.astype(np.int8)
            + self.private_flux.astype(np.int8)
            + self.scrape_off_layer.astype(np.int8)
        )


def _numeric_penalty_mask(penalty_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(penalty_mask)
    if np.iscomplexobj(array):
        raise ValueError("penalty_mask must be real-valued")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("penalty_mask must have a numeric dtype")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("penalty_mask must have axes [x, y]")
    if not np.all(np.isfinite(array)):
        raise ValueError("penalty_mask contains non-finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("penalty_mask must lie in the closed interval [0, 1]")
    return array


def build_single_null_region_masks(
    penalty_mask: np.ndarray,
    separatrix_face_major_radius: np.ndarray,
    *,
    topology: SingleNullTopology,
) -> SingleNullRegionMasks:
    """Build the frozen Paper 0 masks from topology and wall metadata.

    ``penalty_mask`` is the Hypnotoad mask on the model crop. Fractional values
    describe a wall-crossing poloidal edge and are excluded rather than used as
    cell-area weights. ``separatrix_face_major_radius`` is ``Rxy_xlow`` on the
    exact separatrix face and is used only to select the discrete outboard
    midplane row within the closed-field segment.
    """

    penalty = _numeric_penalty_mask(penalty_mask)
    n_x, n_y = penalty.shape
    if n_x < 2 or n_y < 3:
        raise ValueError("geometry needs at least two x and three y cells")
    if not 1 <= topology.separatrix_x_index < n_x:
        raise ValueError("separatrix must lie between supplied radial cells")
    if not (
        0 < topology.pfr_lower_y < topology.core_lower_y
        <= topology.core_upper_y < topology.pfr_upper_y < n_y - 1
    ):
        raise ValueError("single-null topology indices are inconsistent")

    face_radius = np.asarray(separatrix_face_major_radius)
    if np.iscomplexobj(face_radius):
        raise ValueError("separatrix face major radius must be real-valued")
    if not np.issubdtype(face_radius.dtype, np.number):
        raise TypeError("separatrix face major radius must be numeric")
    face_radius = np.asarray(face_radius, dtype=np.float64)
    if face_radius.shape != (n_y,):
        raise ValueError(
            f"separatrix face major radius must have shape {(n_y,)}"
        )
    if not np.all(np.isfinite(face_radius)):
        raise ValueError("separatrix face major radius contains non-finite values")

    strict = penalty == 0.0
    crossing = (penalty > 0.0) & (penalty < 1.0)
    exterior = penalty == 1.0
    if not np.all(
        strict.astype(np.int8)
        + crossing.astype(np.int8)
        + exterior.astype(np.int8)
        == 1
    ):
        raise ValueError("wall categories must be mutually exclusive and exhaustive")

    operator = np.zeros((n_x, n_y), dtype=bool)
    operator[:, 1:-1] = True
    eligible = strict & operator
    sep = topology.separatrix_x_index

    confined = np.zeros_like(eligible)
    confined[:sep, topology.core_lower_y : topology.core_upper_y + 1] = True
    confined &= eligible

    private_flux = np.zeros_like(eligible)
    private_flux[:sep, 1 : topology.pfr_lower_y + 1] = True
    private_flux[:sep, topology.pfr_upper_y : n_y - 1] = True
    private_flux &= eligible

    sol = np.zeros_like(eligible)
    sol[sep:, 1:-1] = True
    sol &= eligible

    partition = (
        confined.astype(np.int8)
        + private_flux.astype(np.int8)
        + sol.astype(np.int8)
    )
    if np.any(partition > 1) or not np.array_equal(partition == 1, eligible):
        raise ValueError("primary topology masks do not partition the eligible cells")

    separatrix_band = np.zeros_like(eligible)
    separatrix_band[sep - 1 : sep + 1, 1:-1] = True
    separatrix_band &= eligible

    core_rows = np.arange(
        topology.core_lower_y, topology.core_upper_y + 1, dtype=np.int64
    )
    core_radius = face_radius[core_rows]
    maximizers = np.flatnonzero(core_radius == np.max(core_radius))
    if maximizers.size != 1:
        raise ValueError("outboard-midplane row is not uniquely defined")
    outboard_y = int(core_rows[maximizers[0]])
    outboard = np.zeros_like(eligible)
    outboard[:, outboard_y] = True
    outboard &= eligible

    xpoint = np.zeros_like(eligible)
    xpoint_rows = np.asarray(
        [
            topology.pfr_lower_y,
            topology.core_lower_y,
            topology.core_upper_y,
            topology.pfr_upper_y,
        ],
        dtype=np.int64,
    )
    xpoint[np.ix_(np.asarray([sep - 1, sep]), xpoint_rows)] = True
    xpoint &= eligible

    inner_leg = np.zeros_like(eligible)
    inner_leg[:, 1 : topology.pfr_lower_y + 1] = True
    inner_leg &= eligible
    outer_leg = np.zeros_like(eligible)
    outer_leg[:, topology.pfr_upper_y : n_y - 1] = True
    outer_leg &= eligible

    return SingleNullRegionMasks(
        strict_wall_interior=strict,
        wall_crossing=crossing,
        wall_exterior=exterior,
        operator_interior=operator,
        confined_edge=confined,
        private_flux=private_flux,
        scrape_off_layer=sol,
        separatrix_cell_band=separatrix_band,
        outboard_midplane=outboard,
        x_point_topology_stencil=xpoint,
        inner_divertor_leg=inner_leg,
        outer_divertor_leg=outer_leg,
        separatrix_x_index=sep,
        separatrix_face_left_cell_index=sep - 1,
        core_lower_y=topology.core_lower_y,
        core_upper_y=topology.core_upper_y,
        outboard_midplane_y=outboard_y,
    )


def confined_separatrix_surface_mask(
    masks: SingleNullRegionMasks,
    left_cell_indices: np.ndarray,
    *,
    operator_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``[face,y]`` mask for the closed-field separatrix surface."""

    indices = np.asarray(left_cell_indices)
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("left_cell_indices must contain integers")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("left_cell_indices must be a nonempty one-dimensional array")
    if np.unique(indices).size != indices.size:
        raise ValueError("left_cell_indices must be unique")
    n_x, n_y = masks.strict_wall_interior.shape
    if np.any(indices < 0) or np.any(indices + 1 >= n_x):
        raise ValueError("face indices exceed the supplied cell masks")

    face_positions = np.flatnonzero(
        indices == masks.separatrix_face_left_cell_index
    )
    if face_positions.size != 1:
        raise ValueError("exactly one confined-separatrix face must be available")
    face_position = int(face_positions[0])

    surface = np.zeros((indices.size, n_y), dtype=bool)
    core_rows = np.zeros(n_y, dtype=bool)
    core_rows[masks.core_lower_y : masks.core_upper_y + 1] = True
    adjacent_strict = (
        masks.strict_wall_interior[masks.separatrix_face_left_cell_index]
        & masks.strict_wall_interior[masks.separatrix_x_index]
    )
    surface[face_position] = core_rows & adjacent_strict

    if operator_valid_mask is not None:
        valid = np.asarray(operator_valid_mask, dtype=bool)
        if valid.shape != surface.shape:
            raise ValueError("operator_valid_mask must have shape [face, y]")
        surface &= valid

    expected = masks.core_upper_y - masks.core_lower_y + 1
    if int(np.sum(surface)) != expected:
        raise ValueError(
            "confined-separatrix surface lost a core row to wall or operator masking"
        )
    return surface
