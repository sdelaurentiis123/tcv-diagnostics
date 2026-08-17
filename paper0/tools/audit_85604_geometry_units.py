#!/usr/bin/env python3
"""Execute the frozen 85604 geometry, unit, and ensemble-transport gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import netCDF4
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tcv_diagnostics.geometry import (  # noqa: E402
    build_single_null_region_masks,
    confined_separatrix_surface_mask,
)
from tcv_diagnostics.metrics import apply_memberwise  # noqa: E402
from tcv_diagnostics.transport import (  # noqa: E402
    PartialCombinedRadialFaceFlow,
    SingleNullTopology,
    hermes_transport_scales,
    integrate_radial_surface_flow,
    normalized_particle_flow_to_si,
    normalized_pressure_flow_to_si,
    radial_exb_face_flow_partial,
    toroidal_wedge_spacing,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(dataset: netCDF4.Dataset, name: str) -> float:
    values = np.asarray(dataset.variables[name][...])
    if values.size != 1:
        raise ValueError(f"{name} is not scalar")
    return float(values.reshape(-1)[0])


def integer(dataset: netCDF4.Dataset, name: str) -> int:
    value = scalar(dataset, name)
    if not float(value).is_integer():
        raise ValueError(f"{name} is not integer-valued")
    return int(value)


def read_array(dataset: netCDF4.Dataset, name: str) -> np.ndarray:
    values = np.asarray(dataset.variables[name][...], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def relative_match(actual: float, expected: float, tolerance: float) -> bool:
    return bool(abs(actual - expected) <= tolerance * abs(expected))


def parse_bout_constant(source: str, name: str) -> float:
    match = re.search(
        rf"constexpr\s+BoutReal\s+{re.escape(name)}\s*=\s*([^;]+);", source
    )
    if match is None:
        raise ValueError(f"could not parse BOUT++ constant {name}")
    return float(match.group(1).strip())


def synthetic_surface_gate() -> dict[str, Any]:
    flow = np.empty((2, 2, 4, 5), dtype=np.float64)
    flow[0].fill(2.0)
    flow[1].fill(-3.0)
    result = PartialCombinedRadialFaceFlow(
        flow=flow,
        xz_flow=flow.copy(),
        xy_flow=np.zeros_like(flow),
        valid_mask=np.ones((2, 4), dtype=bool),
        left_cell_indices=np.asarray([1, 2], dtype=np.int64),
    )
    dy = np.broadcast_to(
        np.asarray([1.0, 2.0, 3.0, 4.0])[None, :], (5, 4)
    ).copy()
    selected = np.zeros((2, 4), dtype=bool)
    selected[0, 1:3] = True
    wedge = integrate_radial_surface_flow(
        result, dy, dz=0.25, face_mask=selected
    )
    full = integrate_radial_surface_flow(
        result,
        dy,
        dz=0.25,
        face_mask=selected,
        toroidal_replication=5,
    )
    expected = np.asarray([12.5, -18.75])
    return {
        "wedge": wedge.tolist(),
        "hand_expected": expected.tolist(),
        "full_torus_equivalent": full.tolist(),
        "hand_sum_exact": bool(np.array_equal(wedge, expected)),
        "replication_factor_exact": bool(np.array_equal(full, 5.0 * wedge)),
        "sign_preserved": bool(wedge[0] > 0.0 and wedge[1] < 0.0),
    }


def synthetic_memberwise_gate() -> dict[str, Any]:
    n_batch, n_member, n_x, n_y, n_z = 1, 2, 5, 32, 9
    angle = 2.0 * np.pi * np.arange(n_z) / n_z
    cosine = np.cos(angle)
    sine = np.sin(angle)
    q = np.empty((n_batch, n_member, n_x, n_y, n_z))
    phi = np.empty_like(q)
    q[:, 0] = 2.0 + 0.5 * cosine
    q[:, 1] = 2.0 + 0.25 * cosine
    phi[:, 0] = sine
    phi[:, 1] = -sine
    geometry = np.ones((n_x, n_y), dtype=np.float64)
    zeros = np.zeros((n_x, n_y), dtype=np.float64)
    dz = toroidal_wedge_spacing(n_z, zperiod=5)
    selected = np.zeros((2, n_y), dtype=bool)
    selected[0, 1:-1] = True
    topology = SingleNullTopology(
        separatrix_x_index=0,
        core_lower_y=8,
        core_upper_y=23,
        pfr_lower_y=7,
        pfr_upper_y=24,
    )

    def diagnostic(q_member: np.ndarray, phi_member: np.ndarray) -> np.ndarray:
        faces = radial_exb_face_flow_partial(
            q_member,
            phi_member,
            geometry,
            geometry,
            zeros,
            geometry,
            zeros,
            geometry,
            np.zeros(n_x),
            dz=dz,
            topology=topology,
            zperiod=5,
            positive=True,
        )
        return integrate_radial_surface_flow(
            faces, geometry, dz=dz, face_mask=selected
        )

    memberwise = apply_memberwise(diagnostic, q, phi, member_axis=1)
    central0 = 0.5 * (np.roll(sine, -1) - np.roll(sine, 1))
    expected = np.asarray(
        [
            30.0 * np.sum((2.0 + 0.5 * cosine) * central0),
            30.0 * np.sum((2.0 + 0.25 * cosine) * -central0),
        ]
    )[None, :]
    mean_field = diagnostic(np.mean(q, axis=1), np.mean(phi, axis=1))
    member_mean = np.mean(memberwise, axis=1)
    return {
        "memberwise": memberwise.tolist(),
        "hand_expected": expected.tolist(),
        "mean_of_memberwise_transport": member_mean.tolist(),
        "transport_of_ensemble_mean_fields": mean_field.tolist(),
        "memberwise_matches_hand": bool(
            np.allclose(memberwise, expected, rtol=1e-14, atol=1e-13)
        ),
        "noncommutation_demonstrated": bool(
            np.max(np.abs(member_mean - mean_field)) > 1.0
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["development_run"] != "85604":
        raise ValueError("geometry audit is locked to development run 85604")
    if manifest["held_out_85606_access_allowed"]:
        raise ValueError("manifest unexpectedly permits held-out access")

    expected_sources = manifest["sources"]
    source_hashes = {
        "geometry": sha256(args.geometry),
        "hypnotoad_mesh": sha256(args.hypnotoad_mesh_source),
        "bout_constants": sha256(args.bout_constants),
        "hermes_div_ops": sha256(args.hermes_source_root / "src/div_ops.cxx"),
        "hermes_evolve_density": sha256(
            args.hermes_source_root / "src/evolve_density.cxx"
        ),
    }
    source_hash_match = {
        "geometry": source_hashes["geometry"]
        == expected_sources["geometry"]["sha256"],
        "hypnotoad_mesh": source_hashes["hypnotoad_mesh"]
        == expected_sources["hypnotoad"]["critical_file_sha256"],
        "bout_constants": source_hashes["bout_constants"]
        == expected_sources["boutpp"]["critical_file_sha256"],
        "hermes_div_ops": source_hashes["hermes_div_ops"]
        == "458eeecbd6da1afb882d0de2b652271fc2c2ca142c39a636a52f3adc5c16ef3f",
        "hermes_evolve_density": source_hashes["hermes_evolve_density"]
        == "3c766083078ec17d737a7ac595868adf1706e0596a9e614bb3ac73f071c1834d",
    }

    hypnotoad_source = args.hypnotoad_mesh_source.read_text(encoding="utf-8")
    bout_source = args.bout_constants.read_text(encoding="utf-8")
    div_source = (args.hermes_source_root / "src/div_ops.cxx").read_text(
        encoding="utf-8"
    )
    density_source = (
        args.hermes_source_root / "src/evolve_density.cxx"
    ).read_text(encoding="utf-8")
    source_semantics = {
        "hypnotoad_x_increases_radially": (
            "x always increases radially across grid" in hypnotoad_source
        ),
        "hypnotoad_penalty_is_edge_fraction": all(
            token in hypnotoad_source
            for token in (
                "def calcPenaltyMask",
                "Both ends of the cell are outside the wall",
                "Cell crosses the wall",
                "/ calc_distance(p1, p2)",
            )
        ),
        "hermes_positive_face_flux_moves_i_to_i_plus_one": all(
            token in div_source
            for token in (
                "result(i, j, k) += flux",
                "result(i + 1, j, k) -= flux",
            )
        ),
        "density_uses_negative_divergence": (
            "ddt(N) = -Div_n_bxGrad_f_B_XPPM" in density_source
        ),
    }

    qe = parse_bout_constant(bout_source, "qe")
    proton_mass = parse_bout_constant(bout_source, "Mp")
    constants_match = bool(
        qe == expected_sources["boutpp"]["constants"]["electron_charge_coulomb"]
        and proton_mass
        == expected_sources["boutpp"]["constants"]["proton_mass_kg"]
    )

    grid_config = manifest["grid"]
    expected_topology = grid_config["topology"]
    with netCDF4.Dataset(args.geometry, "r") as dataset:
        full_penalty = read_array(dataset, "penalty_mask")
        if list(full_penalty.shape) != grid_config["full_shape"]:
            raise ValueError("geometry shape differs from frozen manifest")
        offset = int(grid_config["model_x_to_grid_x_offset"])
        model_nx, model_ny = map(int, grid_config["model_shape"])
        model_slice = slice(offset, offset + model_nx)
        penalty = full_penalty[model_slice]
        psixy = read_array(dataset, "psixy")[model_slice]
        psixy_xlow = read_array(dataset, "psixy_xlow")
        r_xlow = read_array(dataset, "Rxy_xlow")
        z_xlow = read_array(dataset, "Zxy_xlow")
        r_corners = read_array(dataset, "Rxy_corners")
        z_corners = read_array(dataset, "Zxy_corners")
        dy = read_array(dataset, "dy")[model_slice]
        psi_axis = scalar(dataset, "psi_axis")
        psi_bdry = scalar(dataset, "psi_bdry")
        observed_topology = {
            name: integer(dataset, name)
            for name in (
                "ixseps1",
                "ixseps2",
                "jyseps1_1",
                "jyseps1_2",
                "jyseps2_1",
                "jyseps2_2",
            )
        }
        hypnotoad_revision = str(dataset.getncattr("hypnotoad_git_hash"))
        hypnotoad_diff = str(dataset.getncattr("hypnotoad_git_diff"))

    topology = SingleNullTopology(
        separatrix_x_index=int(expected_topology["model_first_sol_x"]),
        core_lower_y=int(expected_topology["core_y_inclusive"][0]),
        core_upper_y=int(expected_topology["core_y_inclusive"][1]),
        pfr_lower_y=int(expected_topology["inner_leg_y_inclusive"][1]),
        pfr_upper_y=int(expected_topology["outer_leg_y_inclusive"][0]),
    )
    global_sep_face = int(expected_topology["ixseps1"])
    masks = build_single_null_region_masks(
        penalty,
        r_xlow[global_sep_face],
        topology=topology,
    )
    safe_faces = np.arange(1, model_nx - 2, dtype=np.int64)
    operator_valid = np.ones((safe_faces.size, model_ny), dtype=bool)
    operator_valid[:, (0, model_ny - 1)] = False
    surface = confined_separatrix_surface_mask(
        masks, safe_faces, operator_valid_mask=operator_valid
    )

    strict_operator = masks.strict_wall_interior & masks.operator_interior
    penalty_counts = {
        "strict": int(np.sum(masks.strict_wall_interior & masks.operator_interior)),
        "fractional": int(np.sum(masks.wall_crossing & masks.operator_interior)),
        "exterior": int(np.sum(masks.wall_exterior & masks.operator_interior)),
    }
    primary_counts = {
        "confined_edge": int(np.sum(masks.confined_edge)),
        "private_flux": int(np.sum(masks.private_flux)),
        "scrape_off_layer": int(np.sum(masks.scrape_off_layer)),
    }
    partition = masks.primary_partition()
    partition_gate = bool(
        np.array_equal(partition, strict_operator.astype(np.int8))
        and np.max(partition) == 1
    )

    radial_differences = np.diff(psixy, axis=0)
    sep_error = float(np.max(np.abs(psixy_xlow[global_sep_face] - psi_bdry)))
    psi_norm = (psixy - psi_axis) / (psi_bdry - psi_axis)
    xpoint_a = np.asarray(
        [
            r_corners[global_sep_face, topology.core_lower_y],
            z_corners[global_sep_face, topology.core_lower_y],
        ]
    )
    xpoint_b = np.asarray(
        [
            r_corners[global_sep_face, topology.pfr_upper_y],
            z_corners[global_sep_face, topology.pfr_upper_y],
        ]
    )
    xpoint_mismatch = float(np.linalg.norm(xpoint_a - xpoint_b))
    omp_y = masks.outboard_midplane_y
    omp_rz = [float(r_xlow[global_sep_face, omp_y]), float(z_xlow[global_sep_face, omp_y])]
    inner_target_rz = [
        float(r_xlow[global_sep_face, 1]),
        float(z_xlow[global_sep_face, 1]),
    ]
    outer_target_rz = [
        float(r_xlow[global_sep_face, model_ny - 2]),
        float(z_xlow[global_sep_face, model_ny - 2]),
    ]

    normalization = manifest["normalization"]
    scales = hermes_transport_scales(
        density_norm_m_minus_3=float(normalization["Nnorm_m_minus_3"]),
        temperature_norm_ev=float(normalization["Tnorm_eV"]),
        magnetic_field_norm_t=float(normalization["Bnorm_T"]),
        electron_charge_c=qe,
        proton_mass_kg=proton_mass,
    )
    scale_values = {
        "Cs0_m_per_s": scales.sound_speed_m_per_s,
        "Omega_ci_per_s": scales.ion_cyclotron_frequency_per_s,
        "rho_s0_m": scales.sound_gyroradius_m,
        "particle_rate_scale_per_s": scales.particle_rate_scale_per_s,
        "pressure_flow_scale_W": scales.pressure_flow_scale_w,
    }
    scale_expected_keys = {
        "Cs0_m_per_s": "Cs0_m_per_s",
        "Omega_ci_per_s": "Omega_ci_per_s",
        "rho_s0_m": "rho_s0_m",
        "particle_rate_scale_per_s": "particle_rate_scale_per_s",
        "pressure_flow_scale_W": "pressure_flow_scale_W",
    }
    scale_tolerance = float(
        manifest["acceptance_gates"]["normalization_relative_tolerance"]
    )
    scale_matches = {
        result_key: relative_match(
            value, float(normalization[manifest_key]), scale_tolerance
        )
        for result_key, value in scale_values.items()
        for manifest_key in [scale_expected_keys[result_key]]
    }
    conversion_probe = np.asarray([-2.0, 0.0, 3.0])
    particle_si = normalized_particle_flow_to_si(conversion_probe, scales)
    pressure_si = normalized_pressure_flow_to_si(conversion_probe, scales)
    internal_si = normalized_pressure_flow_to_si(1.5 * conversion_probe, scales)
    unit_linearity = bool(
        np.array_equal(
            particle_si, conversion_probe * scales.particle_rate_scale_per_s
        )
        and np.array_equal(
            pressure_si, conversion_probe * scales.pressure_flow_scale_w
        )
        and np.array_equal(internal_si, 1.5 * pressure_si)
    )

    surface_synthetic = synthetic_surface_gate()
    memberwise_synthetic = synthetic_memberwise_gate()
    expected_design = manifest["protocol_design_observations"]
    topology_match = bool(
        all(observed_topology[name] == int(expected_topology[name]) for name in observed_topology)
        and observed_topology["ixseps1"] - offset
        == int(expected_topology["model_first_sol_x"])
    )
    mask_counts_match = bool(
        penalty_counts == expected_design["penalty_counts_operator_interior"]
        and primary_counts == expected_design["strict_primary_mask_counts"]
    )
    source_revision_gate = bool(
        hypnotoad_revision == expected_sources["hypnotoad"]["revision"]
        and hypnotoad_diff == ""
    )

    gates = {
        "source_hashes": bool(all(source_hash_match.values())),
        "source_revisions": source_revision_gate,
        "source_semantics": bool(all(source_semantics.values())),
        "bout_constants": constants_match,
        "topology": topology_match,
        "penalty_categories": bool(
            np.array_equal(
                (
                    masks.strict_wall_interior.astype(np.int8)
                    + masks.wall_crossing.astype(np.int8)
                    + masks.wall_exterior.astype(np.int8)
                ),
                np.ones_like(penalty, dtype=np.int8),
            )
        ),
        "mask_partition": partition_gate,
        "mask_counts": mask_counts_match,
        "separatrix_psi": sep_error
        <= float(manifest["acceptance_gates"]["separatrix_psi_absolute_tolerance"]),
        "radial_psi_orientation": bool(np.all(radial_differences > 0.0)),
        "xpoint": xpoint_mismatch
        <= float(
            manifest["acceptance_gates"]["xpoint_duplicate_distance_tolerance_m"]
        ),
        "outboard_midplane": omp_y
        == int(manifest["acceptance_gates"]["unique_outboard_midplane_y"]),
        "inner_outer_target_order": bool(inner_target_rz[0] < outer_target_rz[0]),
        "confined_separatrix_surface": int(np.sum(surface))
        == int(
            manifest["acceptance_gates"]["strict_confined_separatrix_y_count"]
        ),
        "normalization": bool(all(scale_matches.values())),
        "unit_linearity": unit_linearity,
        "surface_known_answer": bool(
            surface_synthetic["hand_sum_exact"]
            and surface_synthetic["replication_factor_exact"]
            and surface_synthetic["sign_preserved"]
        ),
        "memberwise_nonlinear_transport": bool(
            memberwise_synthetic["memberwise_matches_hand"]
            and memberwise_synthetic["noncommutation_demonstrated"]
        ),
        "held_out_85606_untouched": True,
    }

    return {
        "schema_version": 1,
        "phase": manifest["phase"],
        "status": "passed" if all(gates.values()) else "failed",
        "development_run": "85604",
        "held_out_85606_accessed": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "sources": {
            "hashes": source_hashes,
            "hash_matches": source_hash_match,
            "revision_matches": source_revision_gate,
            "semantics": source_semantics,
            "parsed_constants": {
                "electron_charge_c": qe,
                "proton_mass_kg": proton_mass,
            },
        },
        "geometry": {
            "full_shape": list(full_penalty.shape),
            "model_shape": list(penalty.shape),
            "topology": observed_topology,
            "penalty_counts_operator_interior": penalty_counts,
            "primary_mask_counts": primary_counts,
            "strict_operator_cell_count": int(np.sum(strict_operator)),
            "separatrix_face_left_model_x": masks.separatrix_face_left_cell_index,
            "confined_separatrix_y_count": int(np.sum(surface)),
            "positive_radial_psi_differences": int(np.sum(radial_differences > 0.0)),
            "total_radial_psi_differences": int(radial_differences.size),
            "minimum_radial_psi_difference": float(np.min(radial_differences)),
            "separatrix_face_max_abs_psi_error": sep_error,
            "psi_normalized_range": [float(np.min(psi_norm)), float(np.max(psi_norm))],
            "xpoint_RZ_m": xpoint_a.tolist(),
            "xpoint_duplicate_RZ_m": xpoint_b.tolist(),
            "xpoint_mismatch_m": xpoint_mismatch,
            "outboard_midplane_y": omp_y,
            "outboard_midplane_separatrix_RZ_m": omp_rz,
            "inner_target_separatrix_RZ_m": inner_target_rz,
            "outer_target_separatrix_RZ_m": outer_target_rz,
            "positive_direction_on_confined_separatrix": "+x_outward",
        },
        "normalization": {
            "values": scale_values,
            "relative_tolerance": scale_tolerance,
            "matches": scale_matches,
            "conversion_probe": {
                "normalized": conversion_probe.tolist(),
                "particle_rate_per_s": particle_si.tolist(),
                "pressure_flow_W": pressure_si.tolist(),
                "internal_energy_flow_W_after_explicit_1p5": internal_si.tolist(),
            },
        },
        "synthetic_surface": surface_synthetic,
        "synthetic_memberwise_transport": memberwise_synthetic,
        "gates": gates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--hypnotoad-mesh-source", type=Path, required=True)
    parser.add_argument("--bout-constants", type=Path, required=True)
    parser.add_argument("--hermes-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    try:
        result = audit(args)
    except Exception as error:  # Preserve a compact failure artifact.
        result = {
            "schema_version": 1,
            "phase": "phase2_85604_geometry_units_ensemble_transport",
            "status": "error",
            "development_run": "85604",
            "held_out_85606_accessed": False,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
