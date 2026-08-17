"""End-to-end common-view and transport evaluation for matched O1 codecs."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .codec_transport import (
    MATCHED_O1_COMPARISON,
    CodecTransportGeometry,
    TransportComparisonAccumulator,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from .matched_codec_evaluation import (
    COMMON_FIELDS,
    E6B_COMMON_VIEW,
    derive_e6b_common_components,
    encode_physical_batch,
)
from .matched_codec_metrics import MatchedCodecAccumulator
from .matched_o1_transport import (
    MatchedCandidateArtifact,
    MatchedPhiArtifact,
    NativeTruthCatalog,
)
from .model_training_data import CodecFrameDataset, ModelDatasetCatalog
from .resampling import periodic_resample_float32


VALIDATION_BLOCK_FRAMES = 16


def _chunk_stop(
    cursor: int,
    stop: int,
    *,
    chunk_frames: int,
    block_origin: int | None,
) -> int:
    candidate = min(cursor + chunk_frames, stop)
    if block_origin is not None:
        block_stop = (
            block_origin
            + ((cursor - block_origin) // VALIDATION_BLOCK_FRAMES + 1)
            * VALIDATION_BLOCK_FRAMES
        )
        candidate = min(candidate, block_stop)
    return candidate


def evaluate_e6b_common_interval(
    *,
    catalog: ModelDatasetCatalog,
    candidate: MatchedCandidateArtifact,
    phi: MatchedPhiArtifact,
    split: str,
    frames: Sequence[int],
    seed: int,
    chunk_frames: int,
) -> dict[str, Any]:
    """Score E6B in the shared C5P view and verify exact boundary bypass."""

    ordered = tuple(int(frame) for frame in frames)
    if candidate.family != "e6b" or candidate.frames != ordered or phi.frames != ordered:
        raise ValueError("E6B common-view artifacts do not match the requested frames")
    if not 1 <= int(chunk_frames) <= VALIDATION_BLOCK_FRAMES:
        raise ValueError("common-view chunk size must lie in 1..16")
    block_origin = ordered[0] if split == "validation" else None
    if split == "validation" and len(ordered) != 8 * VALIDATION_BLOCK_FRAMES:
        raise ValueError("E6B validation common view must contain eight blocks")

    truth = CodecFrameDataset(
        catalog,
        family="c5p",
        split=split,
        frames=ordered,
        augment=False,
        seed=seed,
        return_physical=True,
    )
    boundary_truth = CodecFrameDataset(
        catalog,
        family="e6b",
        split=split,
        frames=ordered,
        augment=False,
        seed=seed,
        return_physical=True,
    )
    overall = MatchedCodecAccumulator(spec=E6B_COMMON_VIEW)
    blocks = (
        [MatchedCodecAccumulator(spec=E6B_COMMON_VIEW) for _ in range(8)]
        if split == "validation"
        else []
    )
    boundary_count = 0
    boundary_maximum_difference = 0.0
    boundary_bitwise_exact = True
    try:
        cursor = ordered[0]
        while cursor < ordered[-1] + 1:
            stop = _chunk_stop(
                cursor,
                ordered[-1] + 1,
                chunk_frames=chunk_frames,
                block_origin=block_origin,
            )
            positions = range(cursor - ordered[0], stop - ordered[0])
            truth_items = [truth[position] for position in positions]
            boundary_items = [boundary_truth[position] for position in positions]
            truth_frames = np.asarray(
                [item["frame_index"] for item in truth_items], dtype=np.int64
            )
            if not np.array_equal(truth_frames, np.arange(cursor, stop)) or not np.array_equal(
                truth_frames,
                np.asarray(
                    [item["frame_index"] for item in boundary_items],
                    dtype=np.int64,
                ),
            ):
                raise ValueError("common-view truth datasets are not aligned")

            standardized_truth = np.stack(
                [item["volume"] for item in truth_items], axis=0
            )
            physical_truth = np.stack(
                [item["physical_volume"] for item in truth_items], axis=0
            )
            components = candidate.read_model88(cursor, stop)
            phi_model = periodic_resample_float32(
                phi.read(cursor, stop),
                88,
                axis=-1,
            )
            physical_reconstruction = derive_e6b_common_components(
                ne=components["Ne"],
                pe=components["Pe"],
                pi=components["Pi"],
                nvi=components["NVi"],
                phi=phi_model,
            )
            standardized_reconstruction = encode_physical_batch(
                catalog.normalization,
                COMMON_FIELDS,
                physical_reconstruction,
            )
            overall.update(
                standardized_truth,
                standardized_reconstruction,
                physical_truth,
                physical_reconstruction,
            )
            if split == "validation":
                block = (cursor - ordered[0]) // VALIDATION_BLOCK_FRAMES
                blocks[block].update(
                    standardized_truth,
                    standardized_reconstruction,
                    physical_truth,
                    physical_reconstruction,
                )

            expected_boundary = np.stack(
                [item["physical_boundary"] for item in boundary_items], axis=0
            )
            actual_boundary = candidate.read_boundary(cursor, stop)
            difference = np.abs(
                actual_boundary.astype(np.float64)
                - expected_boundary.astype(np.float64)
            )
            boundary_count += int(difference.size)
            boundary_maximum_difference = max(
                boundary_maximum_difference,
                float(np.max(difference)),
            )
            boundary_bitwise_exact = boundary_bitwise_exact and np.array_equal(
                actual_boundary,
                expected_boundary,
            )
            cursor = stop
    finally:
        truth.close()
        boundary_truth.close()

    return {
        "frames": [ordered[0], ordered[-1] + 1],
        "frame_count": len(ordered),
        "overall": overall.finalize(),
        "blocks": [block.finalize() for block in blocks],
        "boundary_bypass": {
            "element_count": boundary_count,
            "maximum_absolute_difference": boundary_maximum_difference,
            "bitwise_exact": bool(boundary_bitwise_exact),
            "passes": bool(boundary_bitwise_exact),
        },
        "native_phi_to_model_grid": {
            "source_samples": 81,
            "target_samples": 88,
            "method": "frozen_unwindowed_periodic_scipy_resample_float32",
            "metric_band_max_k": 7,
            "zperiod": 5,
        },
    }


def evaluate_matched_transport_interval(
    *,
    truth: NativeTruthCatalog,
    candidate: MatchedCandidateArtifact,
    phi: MatchedPhiArtifact | None,
    geometry: CodecTransportGeometry,
    split: str,
    frames: Sequence[int],
    chunk_frames: int,
) -> dict[str, Any]:
    """Apply the authoritative native-81 transport operator path by path."""

    ordered = tuple(int(frame) for frame in frames)
    if candidate.frames != ordered:
        raise ValueError("transport candidate does not match requested frames")
    if candidate.family == "e6b" and (phi is None or phi.frames != ordered):
        raise ValueError("E6B transport requires matching exact potential")
    if candidate.family == "c5p" and phi is not None:
        raise ValueError("C5P transport must use its own reconstructed potential")
    if not 1 <= int(chunk_frames) <= VALIDATION_BLOCK_FRAMES:
        raise ValueError("transport chunk size must lie in 1..16")
    block_origin = ordered[0] if split == "validation" else None
    if split == "validation" and len(ordered) != 8 * VALIDATION_BLOCK_FRAMES:
        raise ValueError("transport validation must contain eight blocks")

    overall = TransportComparisonAccumulator(MATCHED_O1_COMPARISON)
    blocks = (
        [TransportComparisonAccumulator(MATCHED_O1_COMPARISON) for _ in range(8)]
        if split == "validation"
        else []
    )
    cursor = ordered[0]
    while cursor < ordered[-1] + 1:
        stop = _chunk_stop(
            cursor,
            ordered[-1] + 1,
            chunk_frames=chunk_frames,
            block_origin=block_origin,
        )
        truth_fields = truth.read(
            cursor,
            stop,
            fields=("Ne", "Pe", "Pi", "phi"),
        )
        candidate_fields = candidate.read_native(cursor, stop)
        candidate_phi = (
            candidate_fields["phi"]
            if candidate.family == "c5p"
            else phi.read(cursor, stop)
        )
        outputs = {
            "truth": evaluate_transport_state(
                direct_pressure_transport_state(
                    truth_fields["Ne"],
                    truth_fields["Pe"],
                    truth_fields["Pi"],
                    truth_fields["phi"],
                ),
                geometry,
            ),
            "reconstruction": evaluate_transport_state(
                direct_pressure_transport_state(
                    candidate_fields["Ne"],
                    candidate_fields["Pe"],
                    candidate_fields["Pi"],
                    candidate_phi,
                ),
                geometry,
            ),
        }
        overall.update(outputs)
        if split == "validation":
            block = (cursor - ordered[0]) // VALIDATION_BLOCK_FRAMES
            blocks[block].update(outputs)
        cursor = stop

    return {
        "frames": [ordered[0], ordered[-1] + 1],
        "frame_count": len(ordered),
        "overall": overall.finalize(),
        "blocks": [block.finalize() for block in blocks],
        "operator": {
            "grid": "authoritative_native_64x32x81",
            "zperiod": 5,
            "state": ["Ne", "Pe", "Pi", "phi"],
            "nonlinear_quantities_computed_separately_by_path": True,
            "clipping_or_model_output_repair": False,
        },
    }
