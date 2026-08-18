from __future__ import annotations

from pathlib import Path

import numpy as np

from tcv_diagnostics.codec_transport import build_codec_transport_geometry
from tcv_diagnostics.o2_scoring import score_o2_forecast
from tcv_diagnostics.resampling import periodic_resample_float32
from tcv_diagnostics.transport import SingleNullTopology, toroidal_wedge_spacing


class _IdentityNormalization:
    def decode_volume(self, fields, values):
        assert tuple(fields) == ("Ne", "Pe", "Pi", "phi", "Vi")
        return np.asarray(values, dtype=np.float64)


class _Catalog:
    normalization = _IdentityNormalization()


class _TruthDataset:
    def __init__(self, catalog, **kwargs):
        del catalog
        self.target_frames = tuple(kwargs["target_frames"])
        self.context_frames = int(kwargs["context_frames"])
        self.physical = _model_field()

    def __getitem__(self, index):
        target = self.target_frames[index]
        return {
            "target_frame_index": np.int64(target),
            "target": self.physical.astype(np.float32),
            "physical_target": self.physical.astype(np.float32),
            "physical_context": self.physical.astype(np.float32)[None],
        }

    def close(self):
        return None


class _Forecast:
    def __init__(self, path: Path):
        self.path = path
        self.sha256 = "a" * 64
        self.target_frames = (498,)
        self.metadata = {"kind": "known_answer", "target_truth_read": False}
        self.values = _model_field().astype(np.float32)

    def read(self, start, stop):
        assert (start, stop) == (0, 1)
        return self.values[None]

    def timing_record(self):
        return {"target_count": 1, "total_seconds": 0.0}


class _NativeTruth:
    def __init__(self):
        values = periodic_resample_float32(_model_field()[None], 81, axis=-1)
        self.fields = {
            "Ne": values[:, 0].astype(np.float64),
            "Pe": values[:, 1].astype(np.float64),
            "Pi": values[:, 2].astype(np.float64),
            "phi": values[:, 3].astype(np.float64),
        }

    def read(self, start, stop, *, fields):
        assert (start, stop) == (498, 499)
        return {field: self.fields[field] for field in fields}


def _model_field() -> np.ndarray:
    z = 2.0 * np.pi * np.arange(88, dtype=np.float64) / 88.0
    x = np.arange(64, dtype=np.float64)[:, None, None]
    y = np.arange(32, dtype=np.float64)[None, :, None]
    wave = np.sin(z)[None, None, :]
    ne = 2.0 + 0.01 * x + 0.1 * np.cos(z)[None, None, :]
    pe = 3.0 + 0.02 * y + 0.15 * np.cos(z + 0.2)[None, None, :]
    pi = 4.0 + 0.01 * x + 0.01 * y + 0.12 * np.cos(z - 0.1)[None, None, :]
    phi = (1.0 + 0.01 * y) * wave
    vi = np.broadcast_to(0.2 * wave, (64, 32, 88))
    return np.stack(
        [
            np.broadcast_to(ne, (64, 32, 88)),
            np.broadcast_to(pe, (64, 32, 88)),
            np.broadcast_to(pi, (64, 32, 88)),
            np.broadcast_to(phi, (64, 32, 88)),
            vi,
        ],
        axis=0,
    )


def _geometry():
    shape = (64, 32)
    ones = np.ones(shape, dtype=np.float64)
    zeros = np.zeros(shape, dtype=np.float64)
    topology = SingleNullTopology(
        separatrix_x_index=32,
        core_lower_y=8,
        core_upper_y=23,
        pfr_lower_y=7,
        pfr_upper_y=24,
    )
    radius = 1.0 + 0.2 * np.cos(2.0 * np.pi * np.arange(32) / 32.0)
    return build_codec_transport_geometry(
        jacobian=ones,
        g11=ones,
        g23=zeros,
        bxy=ones,
        z_shift=zeros,
        dy=ones,
        shift_angle=np.zeros(64),
        penalty_mask=zeros,
        separatrix_face_major_radius=radius,
        dz=toroidal_wedge_spacing(81, zperiod=5),
        topology=topology,
    )


def test_truth_is_loaded_only_by_separate_scorer_and_exact_forecast_scores_zero(
    monkeypatch,
    tmp_path: Path,
):
    forecast_path = tmp_path / "forecast.h5"
    forecast_path.write_bytes(b"known-answer-placeholder")
    monkeypatch.setattr(
        "tcv_diagnostics.o2_scoring.OneStepWindowDataset",
        _TruthDataset,
    )
    result = score_o2_forecast(
        catalog=_Catalog(),
        forecast_artifact=_Forecast(forecast_path),
        native_truth=_NativeTruth(),
        geometry=_geometry(),
        target_frames=(498,),
        scientific_authority=False,
    )
    assert result["target_truth_used_during_forecast_generation"] is False
    assert result["scientific_authority"] is False
    assert result["field_spectral_cross"]["overall"][
        "aggregate_equal_channel_rmse_standardized"
    ] == 0.0
    comparison = result["transport"]["overall"]["comparisons"][
        "truth_vs_forecast"
    ]
    for quantity in comparison["quantities"].values():
        assert quantity["strict_faces"]["metrics"]["relative_l2"] < 1.0e-6
        assert quantity["separatrix"]["metrics"]["relative_l2"] < 1.0e-6
