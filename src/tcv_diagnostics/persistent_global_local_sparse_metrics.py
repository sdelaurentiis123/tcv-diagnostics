"""Sparse selected-start adapters around immutable, hash-locked B2 metrics.

The historical B2 metric modules are scientific artifacts whose byte hashes are
referenced by earlier protocols.  These adapters preserve their numerical
implementations and change only target-coordinate bookkeeping for the
prospectively selected persistent-pilot starts.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .b2_field_scoring import B2FieldScoreAccumulator
from .b2_spectral_metrics import B2SpectralAccumulator
from .b2_transport_metrics import B2TransportAccumulator


def _sparse_targets(values: Sequence[int]) -> tuple[int, ...]:
    targets = tuple(int(value) for value in values)
    if not targets or targets != tuple(sorted(set(targets))):
        raise ValueError("persistent sparse targets must be strictly increasing")
    if targets == tuple(range(targets[0], targets[-1] + 1)):
        raise ValueError("persistent sparse adapter requires genuinely sparse targets")
    return targets


def _pseudo_targets(count: int) -> tuple[int, ...]:
    return tuple(range(1000, 1000 + int(count)))


class PersistentSparseFieldAccumulator(B2FieldScoreAccumulator):
    """Exact B2 field calculations with explicit selected-frame coordinates."""

    def __init__(
        self,
        *,
        model_seed: int,
        target_frames: Sequence[int],
        region_masks: Mapping[str, np.ndarray],
        validation_blocks: Sequence[Sequence[int]],
        volume_shape: tuple[int, int, int] = (64, 32, 88),
    ) -> None:
        targets = _sparse_targets(target_frames)
        blocks = tuple(tuple(int(value) for value in block) for block in validation_blocks)
        if tuple(value for block in blocks for value in block) != targets:
            raise ValueError("persistent field blocks do not partition sparse targets")
        pseudo = _pseudo_targets(len(targets))
        pseudo_blocks = []
        cursor = 0
        for block in blocks:
            pseudo_blocks.append(pseudo[cursor : cursor + len(block)])
            cursor += len(block)
        super().__init__(
            model_seed=model_seed,
            target_frames=pseudo,
            region_masks=region_masks,
            volume_shape=volume_shape,
            validation_blocks=tuple(pseudo_blocks),
        )
        self.target_frames = targets
        self.blocks = blocks
        self.block_index = {
            target: index for index, block in enumerate(blocks) for target in block
        }

    def finalize(self) -> dict[str, Any]:
        record = super().finalize()
        record["target_frames"] = list(self.target_frames)
        record["target_frames_are_explicit_indices"] = True
        for block_record, block in zip(
            record["chronological_blocks_eligible_union"], self.blocks
        ):
            block_record["target_frames"] = list(block)
            block_record["target_frames_are_explicit_indices"] = True
        return record


class PersistentSparseSpectralAccumulator(B2SpectralAccumulator):
    """Exact B2 spectral calculations with explicit selected-frame coordinates."""

    def __init__(
        self,
        *,
        model_seed: int,
        target_frames: Sequence[int],
        eligible_xy_mask: np.ndarray,
        volume_shape: tuple[int, int, int] = (64, 32, 88),
        zperiod: int = 5,
    ) -> None:
        targets = _sparse_targets(target_frames)
        super().__init__(
            model_seed=model_seed,
            target_frames=_pseudo_targets(len(targets)),
            eligible_xy_mask=eligible_xy_mask,
            volume_shape=volume_shape,
            zperiod=zperiod,
        )
        self.target_frames = targets

    def finalize(self) -> dict[str, Any]:
        record = super().finalize()
        record["target_frames"] = list(self.target_frames)
        record["target_frames_are_explicit_indices"] = True
        return record


class PersistentSparseTransportAccumulator(B2TransportAccumulator):
    """Exact B2 transport calculations with explicit selected-frame coordinates."""

    def __init__(
        self,
        *,
        model_seed: int,
        target_frames: Sequence[int],
        event_thresholds: Mapping[str, float],
        detailed: bool,
    ) -> None:
        targets = _sparse_targets(target_frames)
        super().__init__(
            model_seed=model_seed,
            target_frames=_pseudo_targets(len(targets)),
            event_thresholds=event_thresholds,
            detailed=detailed,
        )
        self.target_frames = targets

    def finalize(self) -> dict[str, Any]:
        record = super().finalize()
        record["target_frames"] = list(self.target_frames)
        record["target_frames_are_explicit_indices"] = True
        return record
