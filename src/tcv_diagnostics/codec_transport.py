"""Frozen O1 codec-transport comparison primitives.

This module contains no file I/O, checkpoint loading, or shot-specific paths.
It implements the four state paths and acceptance logic frozen in
``paper0/protocol/PHASE2_O1_TRANSPORT_PROTOCOL.md``.  Transport remains an
evaluation diagnostic and is never used as a training loss here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .data_protocol import C5_FIELDS
from .geometry import (
    SingleNullRegionMasks,
    build_single_null_region_masks,
    confined_separatrix_surface_mask,
)
from .resampling import (
    finalize_paired_statistics,
    linear_quantile,
    merge_paired_sufficient_statistics,
    paired_sufficient_statistics,
)
from .transport import (
    SingleNullTopology,
    integrate_radial_surface_flow,
    radial_exb_face_flow_partial,
)


TRANSPORT_QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)

STATE_PATHS = ("P0", "P1", "P2", "R")

O1_COMPARISONS = {
    "P0_vs_P1_state_gap": ("P0", "P1"),
    "P1_vs_P2_input_roundtrip": ("P1", "P2"),
    "P2_vs_R_codec_only": ("P2", "R"),
    "P0_vs_R_authoritative": ("P0", "R"),
}

MATCHED_O1_COMPARISON = {
    "truth_vs_reconstruction": ("truth", "reconstruction"),
}

MATCHED_O1_TRANSPORT_THRESHOLDS = {
    "strict_faces": {
        "relative_l2_max": 0.25,
        "rms_ratio": (0.75, 1.25),
        "pearson_correlation_min": 0.85,
        "weighted_sign_disagreement_max": 0.15,
    },
    "separatrix": {
        "relative_l2_max": 0.20,
        "absolute_normalized_bias_max": 0.10,
        "rms_ratio": (0.80, 1.20),
        "pearson_correlation_min": 0.90,
        "weighted_sign_disagreement_max": 0.10,
    },
    "separatrix_block": {
        "relative_l2_max": 0.30,
        "absolute_normalized_bias_max": 0.15,
        "pearson_correlation_min": 0.80,
        "weighted_sign_disagreement_max": 0.15,
        "required_passing_blocks": 7,
    },
}


def _finite_real(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class CodecTransportGeometry:
    """Geometry and masks used by every O1 state path."""

    jacobian: np.ndarray
    g11: np.ndarray
    g23: np.ndarray
    bxy: np.ndarray
    z_shift: np.ndarray
    dy: np.ndarray
    shift_angle: np.ndarray
    dz: float
    topology: SingleNullTopology
    region_masks: SingleNullRegionMasks
    left_cell_indices: np.ndarray
    operator_valid_mask: np.ndarray
    strict_face_mask: np.ndarray
    separatrix_face_mask: np.ndarray


def build_codec_transport_geometry(
    *,
    jacobian: np.ndarray,
    g11: np.ndarray,
    g23: np.ndarray,
    bxy: np.ndarray,
    z_shift: np.ndarray,
    dy: np.ndarray,
    shift_angle: np.ndarray,
    penalty_mask: np.ndarray,
    separatrix_face_major_radius: np.ndarray,
    dz: float,
    topology: SingleNullTopology,
) -> CodecTransportGeometry:
    """Validate native-grid geometry and construct both frozen face masks."""

    arrays = {
        "jacobian": _finite_real("jacobian", jacobian),
        "g11": _finite_real("g11", g11),
        "g23": _finite_real("g23", g23),
        "bxy": _finite_real("bxy", bxy),
        "z_shift": _finite_real("z_shift", z_shift),
        "dy": _finite_real("dy", dy),
    }
    shape = arrays["jacobian"].shape
    if len(shape) != 2 or shape[0] < 4 or shape[1] < 3:
        raise ValueError("cell geometry must have shape [x,y] with x>=4 and y>=3")
    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"{name} shape {array.shape} differs from {shape}")
    if np.any(arrays["bxy"] == 0.0):
        raise ValueError("bxy must be nonzero")
    if np.any(arrays["dy"] <= 0.0):
        raise ValueError("dy must be strictly positive")
    if not math.isfinite(float(dz)) or float(dz) <= 0.0:
        raise ValueError("dz must be finite and strictly positive")

    shift = np.asarray(shift_angle, dtype=np.float64)
    if shift.shape != (shape[0],):
        raise ValueError(f"shift_angle must have shape {(shape[0],)}")
    if not np.all(np.isfinite(shift[: topology.separatrix_x_index])):
        raise ValueError("used inner-branch shift angles must be finite")

    regions = build_single_null_region_masks(
        penalty_mask,
        separatrix_face_major_radius,
        topology=topology,
    )
    left_indices = np.arange(1, shape[0] - 2, dtype=np.int64)
    operator_valid = np.ones((left_indices.size, shape[1]), dtype=bool)
    operator_valid[:, (0, shape[1] - 1)] = False
    strict_faces = (
        operator_valid
        & regions.strict_wall_interior[left_indices]
        & regions.strict_wall_interior[left_indices + 1]
    )
    separatrix = confined_separatrix_surface_mask(
        regions,
        left_indices,
        operator_valid_mask=operator_valid,
    )
    if not np.any(strict_faces):
        raise ValueError("strict face mask is empty")

    return CodecTransportGeometry(
        **arrays,
        shift_angle=shift,
        dz=float(dz),
        topology=topology,
        region_masks=regions,
        left_cell_indices=left_indices,
        operator_valid_mask=operator_valid,
        strict_face_mask=strict_faces,
        separatrix_face_mask=separatrix,
    )


def c5t_transport_state(
    ne: np.ndarray,
    te: np.ndarray,
    ti: np.ndarray,
    phi: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct a transport state from C5T coordinates in float64."""

    ne_array = _finite_real("Ne", ne)
    te_array = _finite_real("Te", te)
    ti_array = _finite_real("Ti", ti)
    phi_array = _finite_real("phi", phi)
    if not (
        ne_array.shape == te_array.shape == ti_array.shape == phi_array.shape
    ):
        raise ValueError("C5T transport fields must have identical shapes")
    return {
        "Ne": ne_array,
        "Pe": ne_array * te_array,
        "Pi": ne_array * ti_array,
        "phi": phi_array,
    }


def direct_pressure_transport_state(
    ne: np.ndarray,
    pe: np.ndarray,
    pi: np.ndarray,
    phi: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the authoritative direct-pressure P0 state without clipping."""

    state = {
        "Ne": _finite_real("Ne", ne),
        "Pe": _finite_real("Pe", pe),
        "Pi": _finite_real("Pi", pi),
        "phi": _finite_real("phi", phi),
    }
    shapes = {array.shape for array in state.values()}
    if len(shapes) != 1:
        raise ValueError("direct-pressure transport fields must have identical shapes")
    return state


def _scaled_face_contributions(
    flow: np.ndarray,
    geometry: CodecTransportGeometry,
    factor: float,
) -> np.ndarray:
    face_weights = geometry.dy[geometry.left_cell_indices] * geometry.dz
    weighted = np.asarray(flow, dtype=np.float64) * face_weights[None, ..., None]
    selected = weighted[:, geometry.strict_face_mask, :]
    if not np.all(np.isfinite(selected)):
        raise ValueError("strict-wall face contributions contain non-finite values")
    return float(factor) * selected


def evaluate_transport_state(
    state: Mapping[str, np.ndarray],
    geometry: CodecTransportGeometry,
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate particle and internal-energy transport for one state path.

    State arrays must have axes ``[time,x,y,z]``.  The nonlinear operator is
    applied independently to each state path before any comparison or
    ensemble reduction.
    """

    if set(state) != {"Ne", "Pe", "Pi", "phi"}:
        raise ValueError("transport state must contain exactly Ne, Pe, Pi, and phi")
    arrays = {name: _finite_real(name, values) for name, values in state.items()}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("transport state arrays must have identical shapes")
    shape = next(iter(shapes))
    if len(shape) != 4:
        raise ValueError("transport state arrays must have axes [time,x,y,z]")
    if shape[1:3] != geometry.jacobian.shape:
        raise ValueError("transport state x-y shape differs from geometry")

    outputs: dict[str, dict[str, np.ndarray]] = {}
    for quantity, field, factor in (
        ("particle", "Ne", 1.0),
        ("electron_internal_energy", "Pe", 1.5),
        ("ion_internal_energy", "Pi", 1.5),
    ):
        faces = radial_exb_face_flow_partial(
            arrays[field],
            arrays["phi"],
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
        if not np.array_equal(faces.left_cell_indices, geometry.left_cell_indices):
            raise ValueError("transport operator returned unexpected radial faces")
        if not np.array_equal(faces.valid_mask, geometry.operator_valid_mask):
            raise ValueError("transport operator returned an unexpected valid mask")
        surface = float(factor) * integrate_radial_surface_flow(
            faces,
            geometry.dy,
            dz=geometry.dz,
            face_mask=geometry.separatrix_face_mask,
        )
        outputs[quantity] = {
            "strict_face_contributions": _scaled_face_contributions(
                faces.flow, geometry, factor
            ),
            "separatrix_wedge": np.asarray(surface, dtype=np.float64),
        }

    outputs["total_internal_energy"] = {
        reduction: (
            outputs["electron_internal_energy"][reduction]
            + outputs["ion_internal_energy"][reduction]
        )
        for reduction in ("strict_face_contributions", "separatrix_wedge")
    }
    return outputs


def per_frame_relative_l2(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    """Return a relative L2 value for every leading time frame."""

    reference_array = _finite_real("frame reference", reference)
    candidate_array = _finite_real("frame candidate", candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("frame reference and candidate shapes differ")
    if reference_array.ndim < 2:
        raise ValueError("per-frame comparison needs time plus sample axes")
    axes = tuple(range(1, reference_array.ndim))
    denominator = np.sum(reference_array * reference_array, axis=axes)
    if np.any(denominator <= 0.0):
        raise ValueError("a reference frame has zero L2 norm")
    numerator = np.sum(
        np.square(candidate_array - reference_array), axis=axes
    )
    return np.sqrt(numerator / denominator)


class TransportComparisonAccumulator:
    """Stream paired face and separatrix statistics for fixed state paths."""

    def __init__(
        self,
        comparisons: Mapping[str, tuple[str, str]] = O1_COMPARISONS,
    ) -> None:
        self.comparisons = dict(comparisons)
        if not self.comparisons:
            raise ValueError("at least one comparison is required")
        self.frames = 0
        self._face_stats: dict[tuple[str, str], list[dict[str, int | float]]] = {
            (comparison, quantity): []
            for comparison in self.comparisons
            for quantity in TRANSPORT_QUANTITIES
        }
        self._surface_stats: dict[
            tuple[str, str], list[dict[str, int | float]]
        ] = {
            (comparison, quantity): []
            for comparison in self.comparisons
            for quantity in TRANSPORT_QUANTITIES
        }
        paths = sorted({path for pair in self.comparisons.values() for path in pair})
        self._surface_values: dict[tuple[str, str], list[np.ndarray]] = {
            (path, quantity): []
            for path in paths
            for quantity in TRANSPORT_QUANTITIES
        }

    def update(
        self,
        path_outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    ) -> None:
        required_paths = {path for pair in self.comparisons.values() for path in pair}
        if set(path_outputs) != required_paths:
            raise ValueError(
                f"expected path outputs {sorted(required_paths)}, got "
                f"{sorted(path_outputs)}"
            )
        frame_counts: set[int] = set()
        for path in required_paths:
            if set(path_outputs[path]) != set(TRANSPORT_QUANTITIES):
                raise ValueError(f"{path} transport quantities are incomplete")
            for quantity in TRANSPORT_QUANTITIES:
                surface = _finite_real(
                    f"{path}.{quantity}.separatrix_wedge",
                    path_outputs[path][quantity]["separatrix_wedge"],
                )
                if surface.ndim != 1:
                    raise ValueError("separatrix values must have one time axis")
                frame_counts.add(surface.size)
                self._surface_values[(path, quantity)].append(surface.copy())
        if len(frame_counts) != 1:
            raise ValueError("state paths disagree on chunk frame count")
        self.frames += next(iter(frame_counts))

        for comparison, (reference_path, candidate_path) in self.comparisons.items():
            for quantity in TRANSPORT_QUANTITIES:
                reference = path_outputs[reference_path][quantity]
                candidate = path_outputs[candidate_path][quantity]
                self._face_stats[(comparison, quantity)].append(
                    paired_sufficient_statistics(
                        reference["strict_face_contributions"],
                        candidate["strict_face_contributions"],
                    )
                )
                self._surface_stats[(comparison, quantity)].append(
                    paired_sufficient_statistics(
                        reference["separatrix_wedge"],
                        candidate["separatrix_wedge"],
                    )
                )

    def finalize(self) -> dict[str, Any]:
        if self.frames <= 0:
            raise ValueError("cannot finalize an empty transport comparison")
        surface_values = {
            path: {
                quantity: np.concatenate(self._surface_values[(path, quantity)])
                for quantity in TRANSPORT_QUANTITIES
            }
            for path in sorted({key[0] for key in self._surface_values})
        }
        results: dict[str, Any] = {}
        for comparison, (reference_path, candidate_path) in self.comparisons.items():
            quantities: dict[str, Any] = {}
            for quantity in TRANSPORT_QUANTITIES:
                face_sufficient = merge_paired_sufficient_statistics(
                    self._face_stats[(comparison, quantity)]
                )
                surface_sufficient = merge_paired_sufficient_statistics(
                    self._surface_stats[(comparison, quantity)]
                )
                reference = surface_values[reference_path][quantity]
                candidate = surface_values[candidate_path][quantity]
                tails: dict[str, float] = {}
                tail_values: dict[str, dict[str, float]] = {}
                for probability in (0.95, 0.99):
                    label = f"p{int(probability * 100)}"
                    reference_quantile = linear_quantile(
                        np.abs(reference), probability
                    )
                    candidate_quantile = linear_quantile(
                        np.abs(candidate), probability
                    )
                    if reference_quantile <= 0.0:
                        raise ValueError(
                            f"{comparison} {quantity} reference {label} is zero"
                        )
                    tails[f"absolute_value_{label}_ratio"] = (
                        candidate_quantile / reference_quantile
                    )
                    tail_values[label] = {
                        "reference": reference_quantile,
                        "candidate": candidate_quantile,
                    }
                quantities[quantity] = {
                    "strict_faces": {
                        "sufficient_statistics": face_sufficient,
                        "metrics": finalize_paired_statistics(face_sufficient),
                    },
                    "separatrix": {
                        "sufficient_statistics": surface_sufficient,
                        "metrics": finalize_paired_statistics(surface_sufficient),
                        "tail_quantiles": tail_values,
                        **tails,
                    },
                }
            results[comparison] = {
                "reference_path": reference_path,
                "candidate_path": candidate_path,
                "quantities": quantities,
            }
        return {
            "frames": self.frames,
            "comparisons": results,
            "surface_series_normalized": surface_values,
        }


def _metrics(
    summary: Mapping[str, Any],
    comparison: str,
    quantity: str,
    reduction: str,
) -> Mapping[str, Any]:
    return summary["comparisons"][comparison]["quantities"][quantity][reduction][
        "metrics"
    ]


def _codec_reduction_passes(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    require_rms_ratio: bool,
) -> tuple[bool, dict[str, bool]]:
    correlation = metrics["pearson_correlation"]
    sign_disagreement = metrics["weighted_sign_disagreement"]
    criteria = {
        "relative_l2": metrics["relative_l2"] <= thresholds["relative_l2_max"],
        "absolute_normalized_bias": abs(metrics["normalized_bias"])
        <= thresholds.get("absolute_normalized_bias_max", math.inf),
        "pearson_correlation": correlation is not None
        and correlation >= thresholds["pearson_correlation_min"],
        "weighted_sign_disagreement": sign_disagreement is not None
        and sign_disagreement <= thresholds["weighted_sign_disagreement_max"],
    }
    if require_rms_ratio:
        lower, upper = thresholds["rms_ratio"]
        criteria["rms_ratio"] = lower <= metrics["rms_ratio"] <= upper
    return all(criteria.values()), criteria


def evaluate_codec_comparison_gate(
    overall: Mapping[str, Any],
    temporal_blocks: list[Mapping[str, Any]],
    *,
    comparison: str,
    face_thresholds: Mapping[str, Any],
    surface_thresholds: Mapping[str, Any],
    block_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen face, surface, and 7-of-8 block gate."""

    if len(temporal_blocks) != 8:
        raise ValueError("the frozen O1 gate requires exactly eight temporal blocks")
    quantity_results: dict[str, Any] = {}
    for quantity in TRANSPORT_QUANTITIES:
        face_metrics = _metrics(overall, comparison, quantity, "strict_faces")
        face_pass, face_criteria = _codec_reduction_passes(
            face_metrics, face_thresholds, require_rms_ratio=True
        )
        surface_metrics = _metrics(overall, comparison, quantity, "separatrix")
        surface_pass, surface_criteria = _codec_reduction_passes(
            surface_metrics, surface_thresholds, require_rms_ratio=True
        )
        block_records = []
        for index, block in enumerate(temporal_blocks):
            block_metrics = _metrics(block, comparison, quantity, "separatrix")
            block_pass, block_criteria = _codec_reduction_passes(
                block_metrics, block_thresholds, require_rms_ratio=False
            )
            block_records.append(
                {
                    "block_index": index,
                    "passes": block_pass,
                    "criteria": block_criteria,
                }
            )
        passing_blocks = sum(record["passes"] for record in block_records)
        blocks_pass = passing_blocks >= int(
            block_thresholds["required_passing_blocks"]
        )
        quantity_results[quantity] = {
            "strict_faces": {"passes": face_pass, "criteria": face_criteria},
            "separatrix": {
                "passes": surface_pass,
                "criteria": surface_criteria,
            },
            "temporal_blocks": {
                "passes": blocks_pass,
                "passing_blocks": passing_blocks,
                "required_passing_blocks": int(
                    block_thresholds["required_passing_blocks"]
                ),
                "blocks": block_records,
            },
            "passes": face_pass and surface_pass and blocks_pass,
        }
    passes = all(record["passes"] for record in quantity_results.values())
    return {
        "comparison": comparison,
        "quantities": quantity_results,
        "passes": passes,
        "status": "pass" if passes else "fail",
    }


def build_matched_o1_transport_gate(
    *,
    overall: Mapping[str, Any],
    temporal_blocks: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply only the transport thresholds in the matched O1/O2 protocol."""

    gate = evaluate_codec_comparison_gate(
        overall,
        temporal_blocks,
        comparison="truth_vs_reconstruction",
        face_thresholds=MATCHED_O1_TRANSPORT_THRESHOLDS["strict_faces"],
        surface_thresholds=MATCHED_O1_TRANSPORT_THRESHOLDS["separatrix"],
        block_thresholds=MATCHED_O1_TRANSPORT_THRESHOLDS[
            "separatrix_block"
        ],
    )
    return {
        "thresholds": {
            name: {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in thresholds.items()
            }
            for name, thresholds in MATCHED_O1_TRANSPORT_THRESHOLDS.items()
        },
        **gate,
    }


def build_o1_transport_gate(
    *,
    overall: Mapping[str, Any],
    temporal_blocks: list[Mapping[str, Any]],
    input_field_max_relative_l2: Mapping[str, float],
    preliminary_status: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Build all frozen O1 stop/go decisions without changing thresholds."""

    if preliminary_status not in {"pass", "fail"}:
        raise ValueError("preliminary_status must be pass or fail")
    if set(input_field_max_relative_l2) != set(C5_FIELDS):
        raise ValueError("input alignment must contain exactly the five C5T fields")
    input_limit = float(thresholds["input_field_max_per_frame_relative_l2"])
    input_fields = {
        field: {
            "maximum_per_frame_relative_l2": float(value),
            "threshold": input_limit,
            "passes": float(value) <= input_limit,
        }
        for field, value in input_field_max_relative_l2.items()
    }
    input_fields_pass = bool(input_fields) and all(
        record["passes"] for record in input_fields.values()
    )

    input_roundtrip_quantities = {}
    for quantity in TRANSPORT_QUANTITIES:
        face_value = float(
            _metrics(
                overall,
                "P1_vs_P2_input_roundtrip",
                quantity,
                "strict_faces",
            )["relative_l2"]
        )
        surface_value = float(
            _metrics(
                overall,
                "P1_vs_P2_input_roundtrip",
                quantity,
                "separatrix",
            )["relative_l2"]
        )
        face_pass = face_value <= float(
            thresholds["input_roundtrip_face_relative_l2"]
        )
        surface_pass = surface_value <= float(
            thresholds["input_roundtrip_surface_relative_l2"]
        )
        input_roundtrip_quantities[quantity] = {
            "strict_face_relative_l2": face_value,
            "separatrix_relative_l2": surface_value,
            "passes": face_pass and surface_pass,
        }
    input_roundtrip_pass = all(
        record["passes"] for record in input_roundtrip_quantities.values()
    )

    state_quantities = {}
    for quantity in TRANSPORT_QUANTITIES:
        limit = float(
            thresholds[
                "state_gap_particle_electron_relative_l2"
                if quantity in {"particle", "electron_internal_energy"}
                else "state_gap_ion_total_relative_l2"
            ]
        )
        face_value = float(
            _metrics(overall, "P0_vs_P1_state_gap", quantity, "strict_faces")
            ["relative_l2"]
        )
        surface_value = float(
            _metrics(overall, "P0_vs_P1_state_gap", quantity, "separatrix")
            ["relative_l2"]
        )
        state_quantities[quantity] = {
            "threshold": limit,
            "strict_face_relative_l2": face_value,
            "separatrix_relative_l2": surface_value,
            "passes": face_value <= limit and surface_value <= limit,
        }
    state_pass = all(record["passes"] for record in state_quantities.values())

    codec_only = evaluate_codec_comparison_gate(
        overall,
        temporal_blocks,
        comparison="P2_vs_R_codec_only",
        face_thresholds=thresholds["codec_face"],
        surface_thresholds=thresholds["codec_surface"],
        block_thresholds=thresholds["codec_surface_block"],
    )
    authoritative = evaluate_codec_comparison_gate(
        overall,
        temporal_blocks,
        comparison="P0_vs_R_authoritative",
        face_thresholds=thresholds["codec_face"],
        surface_thresholds=thresholds["codec_surface"],
        block_thresholds=thresholds["codec_surface_block"],
    )
    full_pass = (
        preliminary_status == "pass"
        and state_pass
        and authoritative["passes"]
        and input_fields_pass
        and input_roundtrip_pass
    )
    return {
        "input_alignment": {
            "fields": input_fields,
            "passes": input_fields_pass,
            "status": "pass" if input_fields_pass else "fail",
        },
        "input_roundtrip": {
            "quantities": input_roundtrip_quantities,
            "passes": input_roundtrip_pass,
            "status": "pass" if input_roundtrip_pass else "fail",
        },
        "c5t_state_adequacy": {
            "quantities": state_quantities,
            "passes": state_pass,
            "status": "pass" if state_pass else "fail",
        },
        "codec_only_transport": codec_only,
        "authoritative_transport": authoritative,
        "prior_preliminary": {
            "status": preliminary_status,
            "passes": preliminary_status == "pass",
        },
        "full_codec_acceptance": {
            "passes": full_pass,
            "status": "pass" if full_pass else "fail",
            "conditions": {
                "input_alignment": input_fields_pass,
                "input_roundtrip": input_roundtrip_pass,
                "prior_preliminary": preliminary_status == "pass",
                "c5t_state_adequacy": state_pass,
                "authoritative_transport": authoritative["passes"],
            },
        },
    }
