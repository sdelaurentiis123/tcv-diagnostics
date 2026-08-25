from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tcv_diagnostics.b2_field_metrics import B2_ALL_REGIONS
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES
from tcv_diagnostics.persistent_global_local_sparse_metrics import (
    PersistentSparseFieldAccumulator,
    PersistentSparseSpectralAccumulator,
    PersistentSparseTransportAccumulator,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASHES = {
    "src/tcv_diagnostics/b2_field_scoring.py": "9bc110067a7124de015202e7eac40030fccda655322f8386dc1d1adca20b9112",
    "src/tcv_diagnostics/b2_spectral_metrics.py": "382fc683519d01185d0e5314196cd0c62f5e39e60f5e1aa06478e74acda8761e",
    "src/tcv_diagnostics/b2_transport_metrics.py": "b78ea33f641fe6409ca5a55503f3729013f2da3cc78f93671f63c6fadafcb02e",
    "src/tcv_diagnostics/o2_training_data.py": "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a",
}


def _regions() -> dict[str, np.ndarray]:
    union = np.ones(12, dtype=bool)
    records = {name: union.copy() for name in B2_ALL_REGIONS}
    records["confined_edge"] = np.arange(12) < 4
    records["private_flux"] = (np.arange(12) >= 4) & (np.arange(12) < 8)
    records["scrape_off_layer"] = np.arange(12) >= 8
    return records


def _transport_case(seed: int):
    generator = np.random.default_rng(seed)
    forecast = {}
    truth = {}
    for index, quantity in enumerate(TRANSPORT_QUANTITIES):
        observed = np.arange(5, dtype=np.float64) + index
        forecast[quantity] = {
            "strict_face_contributions": observed[None] + generator.normal(size=(32, 5)),
            "separatrix_wedge": np.asarray(index + 1.0) + generator.normal(size=32),
        }
        truth[quantity] = {
            "strict_face_contributions": observed,
            "separatrix_wedge": np.asarray([index + 1.0]),
        }
    return forecast, truth


def test_historical_metric_sources_remain_byte_identical():
    for relative, expected in FROZEN_HASHES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_sparse_field_adapter_preserves_explicit_target_and_block_coordinates():
    targets = (498, 502)
    scorer = PersistentSparseFieldAccumulator(
        model_seed=1702,
        target_frames=targets,
        region_masks=_regions(),
        validation_blocks=((498,), (502,)),
        volume_shape=(2, 2, 3),
    )
    generator = np.random.default_rng(4)
    for target in targets:
        truth = generator.normal(size=(5, 2, 2, 3))
        forecast = truth[None, None] + generator.normal(
            scale=0.2, size=(32, 1, 5, 2, 2, 3)
        )
        scorer.update(
            target_frame=target,
            standardized_forecast=forecast,
            standardized_truth=truth,
        )
    result = scorer.finalize()
    assert result["target_frames"] == [498, 502]
    assert result["target_frames_are_explicit_indices"] is True
    assert [item["target_frames"] for item in result["chronological_blocks_eligible_union"]] == [
        [498],
        [502],
    ]


def test_sparse_spectral_and_transport_adapters_keep_real_target_order():
    targets = (498, 502)
    spectral = PersistentSparseSpectralAccumulator(
        model_seed=1702,
        target_frames=targets,
        eligible_xy_mask=np.ones((4, 4), dtype=bool),
        volume_shape=(4, 4, 16),
    )
    generator = np.random.default_rng(8)
    for target in targets:
        truth = generator.normal(size=(5, 4, 4, 16))
        forecast = truth[None] + generator.normal(
            scale=0.1, size=(32, 5, 4, 4, 16)
        )
        spectral.update(
            target_frame=target,
            physical_forecast=forecast,
            physical_truth=truth,
        )
    assert spectral.finalize()["target_frames"] == [498, 502]

    transport = PersistentSparseTransportAccumulator(
        model_seed=1702,
        target_frames=targets,
        event_thresholds={quantity: 0.0 for quantity in TRANSPORT_QUANTITIES},
        detailed=True,
    )
    for index, target in enumerate(targets):
        forecast, truth = _transport_case(index)
        transport.update(
            target_frame=target,
            forecast_outputs=forecast,
            truth_outputs=truth,
        )
    record = transport.finalize()
    assert record["target_frames"] == [498, 502]
    assert record["quantities"]["particle"]["separatrix_time_series"][
        "target_frame"
    ] == [498, 502]
