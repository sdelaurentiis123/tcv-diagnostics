#!/usr/bin/env python3
"""Validate the retained potential solve, then compare the paired boundary arms."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = Path(__file__).resolve().parent
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_hermes_radial_flow_oracle import (  # noqa: E402
    assemble_xy_partitions,
    assembled_xyz,
    canonical_xy,
)
from compare_shifted_ddy_oracle import (  # noqa: E402
    MODEL_X_SLICE,
    TOPOLOGY,
    scalar_integer,
    sha256_file,
)
from tcv_diagnostics.codec_transport import (  # noqa: E402
    build_codec_transport_geometry,
)
from tcv_diagnostics.transport import (  # noqa: E402
    PartialCombinedRadialFaceFlow,
    hermes_transport_scales,
    integrate_radial_surface_flow,
    normalized_particle_flow_to_si,
    normalized_pressure_flow_to_si,
    radial_exb_face_flow_partial,
    toroidal_wedge_spacing,
)


FRAME_INDICES = (0, 156, 312, 467, 623)
INPUT_FIELDS = ("Ne", "Pe", "Pi", "Vort", "phi")
BOUNDARY_SIDES = ("inner", "outer")
DEFAULT_ATOL = 5.0e-10
DEFAULT_RTOL = 5.0e-10
PHI_TO_VOLTS = 50.0
PRESSURE_DENOMINATOR = 3672.0


def frame_label(frame: int) -> str:
    return f"f{frame:03d}"


def _selected_values(
    values: np.ndarray, cell_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(cell_mask)
    if mask.dtype != np.bool_:
        raise TypeError("comparison mask must be boolean")
    if array.ndim < 3 or mask.shape != array.shape[-3:-1]:
        raise ValueError("mask must match the trailing non-z spatial axes")
    shaped = mask.reshape((1,) * (array.ndim - 3) + mask.shape + (1,))
    expanded = np.broadcast_to(shaped, array.shape)
    return array[expanded], expanded


def continuous_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    cell_mask: np.ndarray,
    *,
    location_axes: tuple[str, ...],
    include_sign_error: bool = False,
) -> dict[str, Any]:
    """Return finite continuous metrics without assigning a materiality label."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape:
        raise ValueError("candidate and reference shapes differ")
    if len(location_axes) != candidate_array.ndim:
        raise ValueError("location axis labels do not match the array")
    candidate_values, expanded = _selected_values(candidate_array, cell_mask)
    reference_values, _ = _selected_values(reference_array, cell_mask)
    finite = np.isfinite(candidate_values) & np.isfinite(reference_values)
    nonfinite_count = int(candidate_values.size - np.count_nonzero(finite))

    if not np.any(finite):
        return {
            "point_count": int(candidate_values.size),
            "nonfinite_count": nonfinite_count,
            "relative_l2": None,
            "rmse": None,
            "bias": None,
            "correlation": None,
            "max_abs_reference": None,
            "maximum_absolute_difference": None,
            "maximum_location": None,
            **(
                {"sign_error_count": None, "sign_error_fraction": None}
                if include_sign_error
                else {}
            ),
        }

    candidate_finite = candidate_values[finite]
    reference_finite = reference_values[finite]
    difference = candidate_finite - reference_finite
    numerator = float(np.sum(difference * difference, dtype=np.float64))
    denominator = float(
        np.sum(reference_finite * reference_finite, dtype=np.float64)
    )
    relative_l2 = math.sqrt(numerator / denominator) if denominator > 0.0 else None
    rmse = math.sqrt(numerator / difference.size)
    bias = float(np.mean(difference, dtype=np.float64))
    centered_candidate = candidate_finite - np.mean(
        candidate_finite, dtype=np.float64
    )
    centered_reference = reference_finite - np.mean(
        reference_finite, dtype=np.float64
    )
    correlation_denominator = math.sqrt(
        float(np.sum(centered_candidate * centered_candidate, dtype=np.float64))
        * float(
            np.sum(centered_reference * centered_reference, dtype=np.float64)
        )
    )
    correlation = (
        float(
            np.sum(
                centered_candidate * centered_reference, dtype=np.float64
            )
            / correlation_denominator
        )
        if correlation_denominator > 0.0
        else None
    )

    full_finite = (
        expanded
        & np.isfinite(candidate_array)
        & np.isfinite(reference_array)
    )
    full_difference = np.where(
        full_finite, np.abs(candidate_array - reference_array), -np.inf
    )
    flat = int(np.argmax(full_difference))
    location = np.unravel_index(flat, candidate_array.shape)
    result: dict[str, Any] = {
        "point_count": int(candidate_values.size),
        "nonfinite_count": nonfinite_count,
        "relative_l2": relative_l2,
        "rmse": rmse,
        "bias": bias,
        "correlation": correlation,
        "max_abs_reference": float(np.max(np.abs(reference_finite))),
        "maximum_absolute_difference": float(full_difference.flat[flat]),
        "maximum_location": {
            axis: int(index) for axis, index in zip(location_axes, location)
        },
    }
    if include_sign_error:
        sign_error = np.sign(candidate_finite) != np.sign(reference_finite)
        result["sign_error_count"] = int(np.count_nonzero(sign_error))
        result["sign_error_fraction"] = float(np.mean(sign_error))
    return result


def gate_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    cell_mask: np.ndarray,
    *,
    location_axes: tuple[str, ...],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    result = continuous_metrics(
        candidate,
        reference,
        cell_mask,
        location_axes=location_axes,
    )
    reference_scale = result["max_abs_reference"]
    tolerance = (
        None
        if reference_scale is None
        else float(atol + rtol * reference_scale)
    )
    result["acceptance_tolerance"] = tolerance
    result["passed"] = bool(
        result["nonfinite_count"] == 0
        and tolerance is not None
        and result["maximum_absolute_difference"] is not None
        and result["maximum_absolute_difference"] <= tolerance
    )
    return result


def bitwise_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate_array = np.ascontiguousarray(candidate, dtype="<f8")
    reference_array = np.ascontiguousarray(reference, dtype="<f8")
    if candidate_array.shape != reference_array.shape:
        raise ValueError("bitwise comparison shapes differ")
    candidate_bits = candidate_array.view("<u8")
    reference_bits = reference_array.view("<u8")
    mismatch = candidate_bits != reference_bits
    return {
        "shape": list(candidate_array.shape),
        "point_count": int(candidate_array.size),
        "nonfinite_candidate_count": int(
            np.count_nonzero(~np.isfinite(candidate_array))
        ),
        "nonfinite_reference_count": int(
            np.count_nonzero(~np.isfinite(reference_array))
        ),
        "bitwise_mismatch_count": int(np.count_nonzero(mismatch)),
        "passed": bool(not np.any(mismatch)),
    }


def constant_shift_aligned_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    cell_mask: np.ndarray,
    *,
    location_axes: tuple[str, ...],
) -> dict[str, Any]:
    candidate_values, _ = _selected_values(candidate, cell_mask)
    reference_values, _ = _selected_values(reference, cell_mask)
    finite = np.isfinite(candidate_values) & np.isfinite(reference_values)
    shift = (
        float(np.mean(candidate_values[finite] - reference_values[finite]))
        if np.any(finite)
        else 0.0
    )
    result = continuous_metrics(
        np.asarray(candidate, dtype=np.float64) - shift,
        reference,
        cell_mask,
        location_axes=location_axes,
    )
    result["removed_candidate_minus_reference_constant"] = shift
    result["diagnostic_only"] = True
    return result


def scalar_float(dataset: Any, name: str) -> float:
    values = np.asarray(dataset.variables[name][:], dtype=np.float64)
    if values.size != 1:
        raise ValueError(f"{name} must be scalar in every rank output")
    return float(values.reshape(-1)[0])


def potential_region_masks(regions: Any) -> dict[str, np.ndarray]:
    return {
        "confined_edge": regions.confined_edge,
        "private_flux": regions.private_flux,
        "scrape_off_layer": regions.scrape_off_layer,
        "separatrix_cell_band": regions.separatrix_cell_band,
        "outboard_midplane": regions.outboard_midplane,
        "x_point_topology_stencil": regions.x_point_topology_stencil,
        "inner_divertor_leg": regions.inner_divertor_leg,
        "outer_divertor_leg": regions.outer_divertor_leg,
    }


def face_region_masks(geometry: Any) -> dict[str, np.ndarray]:
    """Map cell regions to faces fully contained in each region."""

    left = geometry.left_cell_indices
    masks = {}
    for name, cell_mask in potential_region_masks(
        geometry.region_masks
    ).items():
        face_mask = (
            geometry.strict_face_mask
            & cell_mask[left]
            & cell_mask[left + 1]
        )
        if np.any(face_mask):
            masks[name] = face_mask
    return masks


def build_geometry(grid: Any) -> Any:
    arrays = {
        name: np.asarray(grid.variables[source][:], dtype=np.float64)[
            MODEL_X_SLICE
        ]
        for name, source in {
            "jacobian": "J",
            "g11": "g11",
            "g23": "g23",
            "bxy": "Bxy",
            "z_shift": "zShift",
            "dy": "dy",
            "penalty_mask": "penalty_mask",
        }.items()
    }
    shift_angle = np.asarray(
        grid.variables["ShiftAngle"][:], dtype=np.float64
    )[MODEL_X_SLICE]
    separatrix_radius = np.asarray(
        grid.variables["Rxy_xlow"][18], dtype=np.float64
    )
    return build_codec_transport_geometry(
        **arrays,
        shift_angle=shift_angle,
        separatrix_face_major_radius=separatrix_radius,
        dz=toroidal_wedge_spacing(81, zperiod=5),
        topology=TOPOLOGY,
    )


def calculate_transport(
    state: dict[str, np.ndarray], geometry: Any
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
]:
    flows: dict[str, np.ndarray] = {}
    surfaces: dict[str, np.ndarray] = {}
    template: PartialCombinedRadialFaceFlow | None = None
    for quantity, field, factor in (
        ("particle", "Ne", 1.0),
        ("electron_internal_energy", "Pe", 1.5),
        ("ion_internal_energy", "Pi", 1.5),
    ):
        faces = radial_exb_face_flow_partial(
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
        if not np.array_equal(
            faces.left_cell_indices, geometry.left_cell_indices
        ):
            raise ValueError("transport face indices differ from frozen geometry")
        if not np.array_equal(faces.valid_mask, geometry.operator_valid_mask):
            raise ValueError("transport valid mask differs from frozen geometry")
        flows[quantity] = float(factor) * faces.flow
        surfaces[quantity] = float(factor) * integrate_radial_surface_flow(
            faces,
            geometry.dy,
            dz=geometry.dz,
            face_mask=geometry.separatrix_face_mask,
        )
        if template is None:
            template = faces
    assert template is not None
    flows["total_internal_energy"] = (
        flows["electron_internal_energy"] + flows["ion_internal_energy"]
    )
    surfaces["total_internal_energy"] = (
        surfaces["electron_internal_energy"] + surfaces["ion_internal_energy"]
    )
    return flows, surfaces, template.left_cell_indices


def potential_effect_summary(
    instantaneous: np.ndarray,
    retained: np.ndarray,
    geometry: Any,
) -> dict[str, Any]:
    all_cells = np.ones((64, 32), dtype=bool)
    regions = potential_region_masks(geometry.region_masks)
    per_frame = {
        frame_label(frame): continuous_metrics(
            instantaneous[position],
            retained[position],
            all_cells,
            location_axes=("x", "y", "z"),
        )
        for position, frame in enumerate(FRAME_INDICES)
    }
    pooled = continuous_metrics(
        instantaneous,
        retained,
        all_cells,
        location_axes=("selected_frame_position", "x", "y", "z"),
    )
    regional = {
        name: continuous_metrics(
            instantaneous,
            retained,
            mask,
            location_axes=("selected_frame_position", "x", "y", "z"),
        )
        for name, mask in regions.items()
    }
    difference = instantaneous - retained
    radial_rms = np.sqrt(
        np.mean(difference * difference, axis=(0, 2, 3), dtype=np.float64)
    )
    return {
        "reference_arm": "retained_saved_midpoint",
        "candidate_arm": "instantaneous_stored_adjacent_interior_target",
        "normalized": {
            "per_frame": per_frame,
            "pooled": pooled,
            "by_geometry_region_pooled": regional,
            "radial_rms_profile": radial_rms.tolist(),
        },
        "volts": {
            "per_frame": {
                frame_label(frame): continuous_metrics(
                    PHI_TO_VOLTS * instantaneous[position],
                    PHI_TO_VOLTS * retained[position],
                    all_cells,
                    location_axes=("x", "y", "z"),
                )
                for position, frame in enumerate(FRAME_INDICES)
            },
            "pooled": continuous_metrics(
                PHI_TO_VOLTS * instantaneous,
                PHI_TO_VOLTS * retained,
                all_cells,
                location_axes=("selected_frame_position", "x", "y", "z"),
            ),
            "radial_rms_profile": (PHI_TO_VOLTS * radial_rms).tolist(),
        },
    }


def transport_effect_summary(
    retained_state: dict[str, np.ndarray],
    instantaneous_state: dict[str, np.ndarray],
    geometry: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    retained_flows, retained_surfaces, left_indices = calculate_transport(
        retained_state, geometry
    )
    instantaneous_flows, instantaneous_surfaces, candidate_left = (
        calculate_transport(instantaneous_state, geometry)
    )
    if not np.array_equal(left_indices, candidate_left):
        raise ValueError("paired transport arms returned different faces")

    region_masks = face_region_masks(geometry)
    scales = hermes_transport_scales(temperature_norm_ev=50.0)
    quantities: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for quantity in retained_flows:
        retained = retained_flows[quantity]
        instantaneous = instantaneous_flows[quantity]
        per_frame = {
            frame_label(frame): continuous_metrics(
                instantaneous[position],
                retained[position],
                geometry.strict_face_mask,
                location_axes=("safe_face_position", "y", "z"),
                include_sign_error=True,
            )
            for position, frame in enumerate(FRAME_INDICES)
        }
        pooled = continuous_metrics(
            instantaneous,
            retained,
            geometry.strict_face_mask,
            location_axes=(
                "selected_frame_position",
                "safe_face_position",
                "y",
                "z",
            ),
            include_sign_error=True,
        )
        regional = {
            name: continuous_metrics(
                instantaneous,
                retained,
                mask,
                location_axes=(
                    "selected_frame_position",
                    "safe_face_position",
                    "y",
                    "z",
                ),
                include_sign_error=True,
            )
            for name, mask in region_masks.items()
        }
        retained_surface = np.asarray(
            retained_surfaces[quantity], dtype=np.float64
        )
        instantaneous_surface = np.asarray(
            instantaneous_surfaces[quantity], dtype=np.float64
        )
        difference_surface = instantaneous_surface - retained_surface
        if quantity == "particle":
            retained_si = normalized_particle_flow_to_si(
                retained_surface, scales
            )
            instantaneous_si = normalized_particle_flow_to_si(
                instantaneous_surface, scales
            )
            si_units = "particles_per_second"
        else:
            retained_si = normalized_pressure_flow_to_si(
                retained_surface, scales
            )
            instantaneous_si = normalized_pressure_flow_to_si(
                instantaneous_surface, scales
            )
            si_units = "watts_internal_energy_advection"
        quantities[quantity] = {
            "strict_local_faces": {
                "per_frame": per_frame,
                "pooled": pooled,
            },
            "by_geometry_region_pooled": regional,
            "confined_separatrix_wedge": {
                "frame_indices": list(FRAME_INDICES),
                "retained_normalized": retained_surface.tolist(),
                "instantaneous_normalized": instantaneous_surface.tolist(),
                "instantaneous_minus_retained_normalized": (
                    difference_surface.tolist()
                ),
                "retained_si": retained_si.tolist(),
                "instantaneous_si": instantaneous_si.tolist(),
                "instantaneous_minus_retained_si": (
                    instantaneous_si - retained_si
                ).tolist(),
                "si_units": si_units,
            },
        }
        arrays[f"retained_surface_{quantity}"] = retained_surface
        arrays[f"instantaneous_surface_{quantity}"] = instantaneous_surface
    return {
        "reference_arm": "retained_saved_midpoint",
        "candidate_arm": "instantaneous_stored_adjacent_interior_target",
        "transport_definition": {
            "particle": "radial_exb_face_flow(Ne, phi)",
            "electron_internal_energy": (
                "1.5 * radial_exb_face_flow(Pe, phi)"
            ),
            "ion_internal_energy": "1.5 * radial_exb_face_flow(Pi, phi)",
            "total_internal_energy": "electron plus ion",
            "called_total_heat_flux": False,
            "nonlinear_operator_applied_separately_to_each_arm": True,
        },
        "strict_face_mask_count": int(np.sum(geometry.strict_face_mask)),
        "confined_separatrix_mask_count": int(
            np.sum(geometry.separatrix_face_mask)
        ),
        "regional_face_rule": (
            "strict operator-valid face with both adjacent cells in the "
            "named authoritative geometry region"
        ),
        "quantities": quantities,
    }, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--extraction-record", type=Path, required=True)
    parser.add_argument("--bout-output", type=Path, nargs="+", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.arrays.exists():
        raise FileExistsError("refusing to overwrite a comparison artifact")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("reconstruction tolerances must be nonnegative")
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for the elliptic oracle") from error

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        manifest["development_run"] != "85604"
        or manifest["held_out_85606_access_allowed"]
        or manifest["training_allowed"]
    ):
        raise ValueError("manifest violates the frozen development-only scope")
    gate_lock = manifest["source_reconstruction_gate"]
    if (
        args.atol != float(gate_lock["continuous_atol"])
        or args.rtol != float(gate_lock["continuous_rtol"])
    ):
        raise ValueError("command tolerances differ from the frozen gate")

    extraction = json.loads(
        args.extraction_record.read_text(encoding="utf-8")
    )
    if (
        extraction["development_run"] != "85604"
        or extraction["held_out_run_read"]
        or extraction["training_performed"]
    ):
        raise ValueError("extraction record violates the frozen scope")
    if extraction["frame_indices"] != list(FRAME_INDICES):
        raise ValueError("extraction record frame set differs from protocol")
    if extraction["canonical_file_sha256"] != sha256_file(args.canonical):
        raise ValueError("canonical input hash differs from extraction record")
    if extraction["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("manifest hash differs from extraction record")

    with netCDF4.Dataset(args.canonical, "r") as canonical_file:
        dimensions = {
            name: int(len(dimension))
            for name, dimension in canonical_file.dimensions.items()
        }
        expected_dimensions = {
            "selected_frame": 5,
            "x": 64,
            "y": 32,
            "z": 81,
            "side": 2,
        }
        if dimensions != expected_dimensions:
            raise ValueError(f"unexpected canonical dimensions: {dimensions}")
        frames = np.asarray(
            canonical_file.variables["frame_index"][:], dtype=np.int64
        )
        if not np.array_equal(frames, FRAME_INDICES):
            raise ValueError("canonical frame indices differ from protocol")
        canonical = {
            name: np.asarray(
                canonical_file.variables[name][:], dtype=np.float64
            )
            for name in INPUT_FIELDS
        }
        boundary = {
            name: np.asarray(
                canonical_file.variables[name][:], dtype=np.float64
            )
            for name in (
                "saved_midpoint",
                "instantaneous_target",
                "midpoint_departure",
            )
        }
    if not np.array_equal(
        boundary["midpoint_departure"],
        boundary["saved_midpoint"] - boundary["instantaneous_target"],
    ):
        raise ValueError("canonical boundary departure is inconsistent")

    with netCDF4.Dataset(args.grid, "r") as grid_file:
        geometry = build_geometry(grid_file)
    if geometry.jacobian.shape != (64, 32):
        raise ValueError("frozen geometry crop has unexpected shape")

    all_cells = np.ones((64, 32), dtype=bool)
    transport_interior = np.zeros((64, 32), dtype=bool)
    transport_interior[:, 1:31] = True
    arrays: dict[str, np.ndarray] = {
        "canonical_phi": canonical["phi"],
        "saved_midpoint": boundary["saved_midpoint"],
        "instantaneous_target": boundary["instantaneous_target"],
        "midpoint_departure": boundary["midpoint_departure"],
    }
    input_echoes: dict[str, Any] = {}
    boundary_echoes: dict[str, Any] = {}
    reconstruction: dict[str, Any] = {}
    retained_all = np.full_like(canonical["phi"], np.nan)
    instantaneous_all = np.full_like(canonical["phi"], np.nan)

    with ExitStack() as stack:
        outputs = [
            stack.enter_context(netCDF4.Dataset(path, "r"))
            for path in args.bout_output
        ]
        decomposition = []
        for path, output in zip(args.bout_output, outputs):
            metadata = {
                "path": str(path),
                "NXPE": scalar_integer(output, "NXPE"),
                "NYPE": scalar_integer(output, "NYPE"),
                "MYSUB": scalar_integer(output, "MYSUB"),
                "PE_XIND": scalar_integer(output, "PE_XIND"),
                "PE_YIND": scalar_integer(output, "PE_YIND"),
            }
            if (
                metadata["NXPE"] != 1
                or metadata["NYPE"] != 4
                or metadata["MYSUB"] != 8
                or metadata["PE_XIND"] != 0
            ):
                raise ValueError(f"unexpected rank decomposition: {metadata}")
            if not math.isclose(
                scalar_float(output, "paper0_rho_s0_meters"),
                float(manifest["normalization"]["rho_s0_meters"]),
                rel_tol=1.0e-14,
                abs_tol=0.0,
            ):
                raise ValueError("compiled rho_s0 differs from manifest")
            if scalar_integer(output, "paper0_zperiod") != 5:
                raise ValueError("compiled zperiod differs from manifest")
            decomposition.append(metadata)
        if sorted(item["PE_YIND"] for item in decomposition) != [0, 1, 2, 3]:
            raise ValueError("rank outputs do not cover four unique y partitions")

        for position, frame in enumerate(FRAME_INDICES):
            label = frame_label(frame)
            for output in outputs:
                if (
                    scalar_integer(
                        output, f"canonical_frame_index_{label}"
                    )
                    != frame
                ):
                    raise ValueError(f"compiled frame marker differs for {label}")

            frame_echoes: dict[str, Any] = {}
            for field in INPUT_FIELDS:
                compiled = assembled_xyz(
                    outputs, decomposition, f"input_{field}_{label}"
                )[MODEL_X_SLICE]
                frame_echoes[field] = bitwise_metrics(
                    compiled, canonical[field][position]
                )
            input_echoes[label] = frame_echoes

            frame_boundary_echoes: dict[str, Any] = {}
            for source_name in ("saved_midpoint", "instantaneous_target"):
                for side_index, side in enumerate(BOUNDARY_SIDES):
                    compiled = assemble_xy_partitions(
                        [
                            (
                                metadata["PE_YIND"],
                                canonical_xy(
                                    output.variables[
                                        f"{source_name}_{side}_{label}"
                                    ]
                                ),
                            )
                            for metadata, output in zip(
                                decomposition, outputs
                            )
                        ]
                    )[MODEL_X_SLICE]
                    expected = np.broadcast_to(
                        boundary[source_name][position, side_index][None, :],
                        (64, 32),
                    )
                    frame_boundary_echoes[f"{source_name}_{side}"] = (
                        bitwise_metrics(compiled, expected)
                    )
            boundary_echoes[label] = frame_boundary_echoes

            retained = assembled_xyz(
                outputs, decomposition, f"retained_phi_{label}"
            )[MODEL_X_SLICE]
            instantaneous = assembled_xyz(
                outputs, decomposition, f"instantaneous_phi_{label}"
            )[MODEL_X_SLICE]
            retained_all[position] = retained
            instantaneous_all[position] = instantaneous
            raw_full = gate_metrics(
                retained,
                canonical["phi"][position],
                all_cells,
                location_axes=("x", "y", "z"),
                atol=args.atol,
                rtol=args.rtol,
            )
            raw_interior = gate_metrics(
                retained,
                canonical["phi"][position],
                transport_interior,
                location_axes=("x", "y", "z"),
                atol=args.atol,
                rtol=args.rtol,
            )
            reconstruction[label] = {
                "raw_full_physical_domain": raw_full,
                "raw_guard_independent_transport_interior": raw_interior,
                "constant_shift_aligned_full_diagnostic": (
                    constant_shift_aligned_metrics(
                        retained,
                        canonical["phi"][position],
                        all_cells,
                        location_axes=("x", "y", "z"),
                    )
                ),
                "passed": bool(raw_full["passed"] and raw_interior["passed"]),
            }

    arrays["retained_phi"] = retained_all
    arrays["instantaneous_phi"] = instantaneous_all
    volume_echoes_passed = all(
        metrics["passed"]
        for frame_metrics in input_echoes.values()
        for metrics in frame_metrics.values()
    )
    boundary_echoes_passed = all(
        metrics["passed"]
        for frame_metrics in boundary_echoes.values()
        for metrics in frame_metrics.values()
    )
    reconstruction_passed = all(
        metrics["passed"] for metrics in reconstruction.values()
    )
    reconstruction_gate_passed = bool(
        volume_echoes_passed
        and boundary_echoes_passed
        and reconstruction_passed
    )

    paired_effect: dict[str, Any]
    if reconstruction_gate_passed:
        potential = potential_effect_summary(
            instantaneous_all, retained_all, geometry
        )
        retained_state = {
            "Ne": canonical["Ne"],
            "Pe": canonical["Pe"],
            "Pi": canonical["Pi"],
            "phi": retained_all,
        }
        instantaneous_state = {
            "Ne": canonical["Ne"],
            "Pe": canonical["Pe"],
            "Pi": canonical["Pi"],
            "phi": instantaneous_all,
        }
        transport, transport_arrays = transport_effect_summary(
            retained_state, instantaneous_state, geometry
        )
        arrays.update(transport_arrays)
        arrays["radial_phi_rms_normalized"] = np.asarray(
            potential["normalized"]["radial_rms_profile"], dtype=np.float64
        )
        paired_effect = {
            "status": "computed_after_reconstruction_gate_passed",
            "materiality_label_assigned": False,
            "potential": potential,
            "transport": transport,
        }
    else:
        paired_effect = {
            "status": "blocked_by_source_reconstruction_gate",
            "materiality_label_assigned": False,
            "potential": None,
            "transport": None,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    result = {
        "schema_version": 1,
        "phase": "phase2_potential_elliptic_85604_paired_oracle",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "frame_indices": list(FRAME_INDICES),
        "physical_independence_claimed_for_frames": False,
        "native_z_samples": 81,
        "zperiod": 5,
        "equation": {
            "pressure_correction": "Pi_hat = Pi - Pe / 3672",
            "coefficient_C": "2 / Bxy^2",
            "right_hand_side": "Vort * Bxy^2 / 2",
            "solver_type": "cyclic",
            "split_n0": False,
            "radial_boundary_flags": "INVERT_SET",
        },
        "source_reconstruction_gate": {
            "continuous_atol": args.atol,
            "continuous_rtol": args.rtol,
            "volume_input_echoes": input_echoes,
            "boundary_input_echoes": boundary_echoes,
            "per_frame_reconstruction": reconstruction,
            "volume_input_echoes_passed": volume_echoes_passed,
            "boundary_input_echoes_passed": boundary_echoes_passed,
            "all_frame_reconstructions_passed": reconstruction_passed,
            "passed": reconstruction_gate_passed,
            "constant_shift_alignment_can_change_gate": False,
        },
        "paired_boundary_effect": paired_effect,
        "decision": {
            "paired_effect_interpretable": reconstruction_gate_passed,
            "selected_frames_establish_all_frame_stability": False,
            "automatic_state_change_authorized": False,
            "automatic_training_authorized": False,
        },
        "artifacts": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "canonical": str(args.canonical),
            "canonical_sha256": sha256_file(args.canonical),
            "extraction_record": str(args.extraction_record),
            "extraction_record_sha256": sha256_file(args.extraction_record),
            "bout_outputs": [str(path) for path in args.bout_output],
            "bout_output_sha256": {
                str(path): sha256_file(path) for path in args.bout_output
            },
            "geometry": str(args.grid),
            "geometry_sha256": sha256_file(args.grid),
            "comparison_arrays": str(args.arrays),
            "comparison_arrays_sha256": sha256_file(args.arrays),
        },
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "source_reconstruction_gate_passed": (
                    reconstruction_gate_passed
                ),
                "paired_boundary_effect_status": paired_effect["status"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if reconstruction_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
