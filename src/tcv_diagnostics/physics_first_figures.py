"""Physics-first, matched-realization diagnostics for old 85604 forecasts.

This module contains only array reductions and plotting helpers.  It never
selects a model, trains a network, or opens a data source by itself.  The
entrypoint under ``paper0/tools`` owns the strict artifact and split checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from .b5_covariance_localization import exact_separatrix_local_contributions
from .codec_transport import (
    CodecTransportGeometry,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from .model_training_data import FAMILY_FIELDS, ModelDatasetCatalog
from .resampling import periodic_resample_float32


PHYSICS_FIGURE_TARGET = 509
PHYSICS_FIGURE_Z_PLANE = 44
PHYSICS_FIGURE_MODEL_SEED = 1702
PHYSICS_FIGURE_MEMBERS = 32
PHYSICS_FIGURE_DISPLAY_MEMBERS = 16
PHYSICS_FIGURE_STARTS = (
    497,
    501,
    504,
    508,
    511,
    515,
    518,
    522,
    525,
    529,
    532,
    536,
    537,
    541,
    544,
    548,
    552,
    555,
    559,
    562,
    566,
    570,
    573,
    577,
    578,
    582,
    585,
    589,
    593,
    597,
    600,
    604,
    608,
    612,
    615,
    619,
)
PHYSICS_FIGURE_H1_TARGETS = tuple(value + 1 for value in PHYSICS_FIGURE_STARTS)
TRANSPORT_VARIOGRAM_LAGS = (1, 2, 4, 8, 16, 32, 40)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_LENGTH = 3
BOOTSTRAP_SEED = 856_040_828

MODEL_LABELS = {
    "deterministic": "codec-free multi-lead operator",
    "b5_context": "conditioned field-residual diffusion",
    "ecrd": "equivariant conditioned diffusion",
    "persistent": "persistent global-local generator",
}
MODEL_COLORS = {
    "deterministic": "#007C91",
    "b5_context": "#E69F00",
    "ecrd": "#7A5195",
    "persistent": "#2A9D8F",
}
TRUTH_COLOR = "#111111"


def _finite(name: str, values: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(array, dtype=dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def lower_sample_median_target(
    per_target_transport: Sequence[Mapping[str, Any]],
    *,
    allowed_targets: Sequence[int] = PHYSICS_FIGURE_H1_TARGETS,
) -> tuple[int, list[dict[str, float | int]]]:
    """Reconstruct and select the lower sample median absolute wedge error.

    The frozen score stores the relative L2 error and truth RMS of a scalar
    wedge value.  Their product is exactly the absolute error.  The 36-value
    population has two central order statistics, so the mechanically chosen
    representative is the lower sample median (rank 18 of 36).
    """

    allowed = tuple(int(value) for value in allowed_targets)
    if len(allowed) != len(set(allowed)) or not allowed:
        raise ValueError("allowed target population must be nonempty and unique")
    records: list[dict[str, float | int]] = []
    for item in per_target_transport:
        target = int(item["target_frame"])
        if target not in allowed:
            continue
        scalar = item["quantities"]["particle"]["separatrix_wedge"]
        error = float(scalar["ensemble_mean_relative_l2"]) * float(
            scalar["truth_rms"]
        )
        if not math.isfinite(error) or error < 0.0:
            raise ValueError("representative transport error is invalid")
        records.append({"target_frame": target, "absolute_error": error})
    if {int(item["target_frame"]) for item in records} != set(allowed):
        raise ValueError("score does not cover the frozen shared target population")
    ordered = sorted(
        records,
        key=lambda item: (float(item["absolute_error"]), int(item["target_frame"])),
    )
    selected = int(ordered[(len(ordered) - 1) // 2]["target_frame"])
    return selected, ordered


def standardized_toroidal_fluctuations(
    values: np.ndarray,
    *,
    eligible_xy: np.ndarray,
) -> np.ndarray:
    """Remove each x-y cell's toroidal mean and fix the potential gauge.

    The final four axes are ``[field,x,y,z]``; arbitrary leading sample axes
    are retained.  Gauge fixing is performed independently for every leading
    sample on the strict eligible volume.
    """

    array = _finite("standardized fields", values, dtype=np.float32)
    if array.ndim < 4 or array.shape[-4:] != (5, 64, 32, 88):
        raise ValueError("field tensor must end in [5,64,32,88]")
    mask = np.asarray(eligible_xy, dtype=bool)
    if mask.shape != (64, 32) or not np.any(mask):
        raise ValueError("eligible mask must be nonempty [64,32]")
    result = np.asarray(
        array - np.mean(array, axis=-1, keepdims=True, dtype=np.float64),
        dtype=np.float32,
    )
    phi = result[..., 3, :, :, :]
    selected = phi[..., mask, :]
    gauge = np.mean(selected, axis=(-2, -1), keepdims=True, dtype=np.float64)
    phi -= np.asarray(gauge, dtype=np.float32)[..., None]
    return np.ascontiguousarray(result)


class NePhiCrossSpectrumAccumulator:
    """Pool memberwise Ne-phi cross spectra over targets and eligible cells."""

    def __init__(self, eligible_xy: np.ndarray, *, zperiod: int = 5) -> None:
        mask = np.asarray(eligible_xy, dtype=bool)
        if mask.shape != (64, 32) or not np.any(mask):
            raise ValueError("cross-spectrum mask must be nonempty [64,32]")
        if int(zperiod) != 5:
            raise ValueError("old-85604 physical mode mapping requires zperiod=5")
        self.mask = mask
        self.zperiod = int(zperiod)
        modes = 88 // 2 + 1
        self.cross_sum = np.zeros(modes, dtype=np.complex128)
        self.ne_power_sum = np.zeros(modes, dtype=np.float64)
        self.phi_power_sum = np.zeros(modes, dtype=np.float64)
        self.sample_count = 0

    def update(self, values: np.ndarray) -> None:
        array = _finite("cross-spectrum fields", values, dtype=np.float32)
        if array.ndim != 5 or array.shape[1:] != (5, 64, 32, 88):
            raise ValueError("cross-spectrum values must be [sample,5,64,32,88]")
        ne = np.asarray(array[:, 0, self.mask, :], dtype=np.float64)
        phi = np.asarray(array[:, 3, self.mask, :], dtype=np.float64)
        ne -= np.mean(ne, axis=-1, keepdims=True)
        phi -= np.mean(phi, axis=-1, keepdims=True)
        ne_hat = np.fft.rfft(ne, axis=-1) / ne.shape[-1]
        phi_hat = np.fft.rfft(phi, axis=-1) / phi.shape[-1]
        axes = (0, 1)
        self.cross_sum += np.sum(ne_hat * np.conjugate(phi_hat), axis=axes)
        self.ne_power_sum += np.sum(np.abs(ne_hat) ** 2, axis=axes)
        self.phi_power_sum += np.sum(np.abs(phi_hat) ** 2, axis=axes)
        self.sample_count += int(ne.shape[0] * ne.shape[1])

    def finalize(self) -> dict[str, np.ndarray | int]:
        if self.sample_count <= 0:
            raise RuntimeError("cross-spectrum accumulator is empty")
        cross = self.cross_sum / self.sample_count
        ne_power = self.ne_power_sum / self.sample_count
        phi_power = self.phi_power_sum / self.sample_count
        denominator = ne_power * phi_power
        coherence = np.divide(
            np.abs(cross) ** 2,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
        coherence = np.clip(coherence, 0.0, 1.0)
        k = np.arange(cross.size, dtype=np.int64)
        return {
            "stored_k": k,
            "physical_n": self.zperiod * k,
            "cross_spectrum": cross,
            "cross_magnitude": np.abs(cross),
            "phase_degrees": np.rad2deg(np.angle(cross)),
            "coherence": coherence,
            "ne_power": ne_power,
            "phi_power": phi_power,
            "sample_count": self.sample_count,
        }


def first_order_toroidal_variogram(
    local_contributions: np.ndarray,
    *,
    lags: Sequence[int] = TRANSPORT_VARIOGRAM_LAGS,
) -> np.ndarray:
    """Return the first-order periodic structure function per leading sample.

    Inputs end in ``[separatrix_row,native_z]``.  Every lag receives equal
    average weight across rows and toroidal positions.
    """

    values = _finite("local transport", local_contributions)
    if values.ndim < 2 or values.shape[-2:] != (16, 81):
        raise ValueError("local transport must end in [16,81]")
    lag_values = tuple(int(value) for value in lags)
    if not lag_values or any(value <= 0 or value > 40 for value in lag_values):
        raise ValueError("variogram lags must lie in native periodic half-domain")
    result = [
        np.mean(
            np.abs(values - np.roll(values, -lag, axis=-1)),
            axis=(-2, -1),
            dtype=np.float64,
        )
        for lag in lag_values
    ]
    return np.stack(result, axis=-1)


def moving_block_bootstrap_indices(
    sample_count: int,
    *,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Noncircular chronological moving-block bootstrap index bank."""

    n = int(sample_count)
    length = int(block_length)
    count = int(replicates)
    if n < length or length <= 0 or count <= 0:
        raise ValueError("moving-block bootstrap dimensions are invalid")
    candidates = n - length + 1
    blocks_needed = math.ceil(n / length)
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    starts = generator.integers(0, candidates, size=(count, blocks_needed))
    offsets = np.arange(length, dtype=np.int64)
    indices = (starts[..., None] + offsets).reshape(count, -1)[:, :n]
    return np.asarray(indices, dtype=np.int64)


def bootstrap_curve_mean(
    per_target_curves: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mean curve and pointwise 95% interval under a fixed bootstrap bank."""

    values = _finite("per-target curves", per_target_curves)
    bank = np.asarray(indices, dtype=np.int64)
    if values.ndim != 2 or bank.ndim != 2 or bank.shape[1] != values.shape[0]:
        raise ValueError("bootstrap curve or index shape differs")
    if np.any(bank < 0) or np.any(bank >= values.shape[0]):
        raise ValueError("bootstrap indices leave the curve population")
    replicates = np.mean(values[bank], axis=1, dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0, dtype=np.float64),
        "lower_2p5": np.quantile(replicates, 0.025, axis=0, method="linear"),
        "upper_97p5": np.quantile(replicates, 0.975, axis=0, method="linear"),
    }


def decode_c5p_members(
    catalog: ModelDatasetCatalog,
    standardized_members: np.ndarray,
) -> np.ndarray:
    """Decode any member count of C5P fields without member reduction."""

    values = _finite("standardized C5P members", standardized_members, dtype=np.float32)
    if values.ndim != 5 or values.shape[1:] != (5, 64, 32, 88):
        raise ValueError("C5P members must be [member,5,64,32,88]")
    fields = FAMILY_FIELDS["c5p"]
    decoded = np.stack(
        [
            catalog.normalization.records[field].decode(values[:, channel])
            for channel, field in enumerate(fields)
        ],
        axis=1,
    )
    if decoded.shape != values.shape or not np.all(np.isfinite(decoded)):
        raise RuntimeError("decoded C5P member tensor differs")
    return np.asarray(decoded, dtype=np.float64)


def authoritative_local_transport(
    *,
    physical_model88: np.ndarray,
    native_truth: Mapping[str, np.ndarray],
    geometry: CodecTransportGeometry,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Evaluate every member jointly, then select exact separatrix cells.

    This is a vectorized use of the same frozen nonlinear operator as the B2,
    ECRD, and persistent-model scorers.  It supports a deterministic one-member
    forecast without pretending that forecast is an ensemble.
    """

    forecast = _finite("physical model88 forecast", physical_model88)
    if forecast.ndim != 5 or forecast.shape[1:] != (5, 64, 32, 88):
        raise ValueError("physical model88 forecast must be [member,5,64,32,88]")
    if set(native_truth) != {"Ne", "Pe", "Pi", "phi"}:
        raise ValueError("native truth fields differ")
    truth = {
        name: _finite(f"native truth {name}", values)
        for name, values in native_truth.items()
    }
    if any(values.shape != (1, 64, 32, 81) for values in truth.values()):
        raise ValueError("native truth must have [1,64,32,81]")
    native_forecast = periodic_resample_float32(
        np.asarray(forecast[:, :4], dtype=np.float32), 81, axis=-1
    ).astype(np.float64)
    forecast_state = direct_pressure_transport_state(
        native_forecast[:, 0],
        native_forecast[:, 1],
        native_forecast[:, 2],
        native_forecast[:, 3],
    )
    truth_state = direct_pressure_transport_state(
        truth["Ne"], truth["Pe"], truth["Pi"], truth["phi"]
    )
    forecast_outputs = evaluate_transport_state(forecast_state, geometry)
    truth_outputs = evaluate_transport_state(truth_state, geometry)
    forecast_local, forecast_closure = exact_separatrix_local_contributions(
        forecast_outputs,
        strict_face_mask=geometry.strict_face_mask,
        separatrix_face_mask=geometry.separatrix_face_mask,
    )
    truth_local, truth_closure = exact_separatrix_local_contributions(
        truth_outputs,
        strict_face_mask=geometry.strict_face_mask,
        separatrix_face_mask=geometry.separatrix_face_mask,
    )
    return forecast_local, truth_local, max(forecast_closure, truth_closure)


@dataclass(frozen=True)
class PlotGeometry:
    major_radius: np.ndarray
    vertical_position: np.ndarray
    separatrix_major_radius: np.ndarray
    separatrix_vertical_position: np.ndarray
    separatrix_arc_length: np.ndarray
    toroidal_separations: np.ndarray


def build_plot_geometry(
    *,
    major_radius: np.ndarray,
    vertical_position: np.ndarray,
    separatrix_major_radius: np.ndarray,
    separatrix_vertical_position: np.ndarray,
    native_dz: float,
    lags: Sequence[int] = TRANSPORT_VARIOGRAM_LAGS,
) -> PlotGeometry:
    """Validate physical R-Z coordinates and derive separatrix distances."""

    r = _finite("Rxy", major_radius)
    z = _finite("Zxy", vertical_position)
    face_r = _finite("separatrix R", separatrix_major_radius)
    face_z = _finite("separatrix Z", separatrix_vertical_position)
    if r.shape != (64, 32) or z.shape != r.shape:
        raise ValueError("plot geometry must be [64,32]")
    if face_r.shape != (16,) or face_z.shape != (16,):
        raise ValueError("separatrix plot geometry must contain 16 rows")
    if not math.isfinite(float(native_dz)) or float(native_dz) <= 0.0:
        raise ValueError("native toroidal spacing is invalid")
    increments = np.sqrt(np.diff(face_r) ** 2 + np.diff(face_z) ** 2)
    arc = np.concatenate(([0.0], np.cumsum(increments)))
    separations = np.asarray(
        [float(lag) * float(native_dz) * float(np.mean(face_r)) for lag in lags],
        dtype=np.float64,
    )
    return PlotGeometry(
        major_radius=r,
        vertical_position=z,
        separatrix_major_radius=face_r,
        separatrix_vertical_position=face_z,
        separatrix_arc_length=arc,
        toroidal_separations=separations,
    )


class OneStepForecastReader:
    """Strict reader for one frozen ECRD-format M32 forecast artifact."""

    def __init__(self, path: Path, *, arm: str, model_seed: int = 1702) -> None:
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        attrs = self.handle.attrs
        required = {
            "development_run": "85604",
            "arm": arm,
            "model_seed": int(model_seed),
            "ensemble_size": 32,
            "horizon_frames": 1,
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "target_truth_used_as_model_input": False,
            "zperiod": 5,
        }
        for name, expected in required.items():
            if attrs.get(name) != expected:
                raise ValueError(f"one-step forecast attribute {name} differs")
        self.targets = tuple(
            int(value) for value in self.handle["target_frame_index"][:]
        )
        if self.targets != tuple(range(498, 624)):
            raise ValueError("one-step forecast target interval differs")
        dataset = self.handle["standardized_forecast"]
        if dataset.shape != (126, 32, 1, 5, 64, 32, 88):
            raise ValueError("one-step forecast tensor schema differs")
        self.dataset = dataset
        self.index = {target: index for index, target in enumerate(self.targets)}

    def read(self, target: int) -> np.ndarray:
        if int(target) not in self.index:
            raise ValueError("requested target leaves one-step forecast")
        values = np.asarray(
            self.dataset[self.index[int(target)], :, 0], dtype=np.float32
        )
        if values.shape != (32, 5, 64, 32, 88) or not np.all(np.isfinite(values)):
            raise ValueError("one-step forecast read differs")
        return values

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "OneStepForecastReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class PersistentForecastReader:
    """Strict reader for the frozen seed-1702 four-frame PGL artifact."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        attrs = self.handle.attrs
        required = {
            "development_run": "85604",
            "ensemble_size": 32,
            "future_frames": 4,
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "new_nersc_data_read": False,
            "target_truth_read_during_generation": False,
            "zperiod": 5,
        }
        for name, expected in required.items():
            if attrs.get(name) != expected:
                raise ValueError(f"persistent forecast attribute {name} differs")
        starts = tuple(int(value) for value in self.handle["current_frame"][:])
        if starts != PHYSICS_FIGURE_STARTS:
            raise ValueError("persistent forecast start population differs")
        target = np.asarray(self.handle["target_frame"][:], dtype=np.int64)
        expected_targets = np.asarray(
            [[start + step for step in range(1, 5)] for start in starts],
            dtype=np.int64,
        )
        if not np.array_equal(target, expected_targets):
            raise ValueError("persistent target coordinate differs")
        dataset = self.handle["standardized_forecast"]
        if dataset.shape != (36, 32, 4, 5, 64, 32, 88):
            raise ValueError("persistent forecast tensor schema differs")
        self.dataset = dataset
        self.index = {start: index for index, start in enumerate(starts)}

    def read(self, *, start: int, horizon: int) -> np.ndarray:
        if int(start) not in self.index or int(horizon) not in (1, 2, 3, 4):
            raise ValueError("persistent forecast coordinate differs")
        values = np.asarray(
            self.dataset[self.index[int(start)], :, int(horizon) - 1],
            dtype=np.float32,
        )
        if values.shape != (32, 5, 64, 32, 88) or not np.all(np.isfinite(values)):
            raise ValueError("persistent forecast read differs")
        return values

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "PersistentForecastReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _apply_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _overlay_geometry(ax: Any, plot: PlotGeometry, masks: Any) -> None:
    r, z = plot.major_radius, plot.vertical_position
    ax.contour(
        r,
        z,
        np.asarray(masks.strict_wall_interior, dtype=float),
        levels=[0.5],
        colors=["#4B5563"],
        linewidths=0.55,
    )
    ax.plot(
        plot.separatrix_major_radius,
        plot.separatrix_vertical_position,
        color="#111111",
        linewidth=0.9,
    )
    xp = np.asarray(masks.x_point_topology_stencil, dtype=bool)
    ax.scatter(r[xp], z[xp], s=7, marker="x", color="#B2182B", linewidths=0.6)
    y = int(masks.outboard_midplane_y)
    x = int(masks.separatrix_face_left_cell_index)
    ax.scatter([r[x, y]], [z[x, y]], s=12, marker="o", facecolors="none", edgecolors="#2166AC", linewidths=0.8)
    ax.set_aspect("equal", adjustable="box")


def save_field_comparison_figure(
    path: Path,
    *,
    truth: np.ndarray,
    forecast_means: Mapping[str, np.ndarray],
    eligible_xy: np.ndarray,
    plot_geometry: PlotGeometry,
    region_masks: Any,
    target_frame: int = PHYSICS_FIGURE_TARGET,
    z_plane: int = PHYSICS_FIGURE_Z_PLANE,
) -> None:
    """Plot matched R-Z truth, forecast mean, and error for Ne and phi."""

    import matplotlib.pyplot as plt

    _apply_plot_style()
    observed = standardized_toroidal_fluctuations(
        np.asarray(truth)[None], eligible_xy=eligible_xy
    )[0]
    ordered = tuple(MODEL_LABELS)
    if tuple(forecast_means) != ordered:
        raise ValueError("field-comparison model order differs")
    prepared = {
        key: standardized_toroidal_fluctuations(
            np.asarray(value)[None], eligible_xy=eligible_xy
        )[0]
        for key, value in forecast_means.items()
    }
    mask = ~np.asarray(eligible_xy, dtype=bool)
    limits = {
        channel: float(np.max(np.abs(observed[channel, eligible_xy, :])))
        for channel in (0, 3)
    }
    fig, axes = plt.subplots(len(ordered), 6, figsize=(18.0, 10.8), constrained_layout=True)
    images: dict[int, Any] = {}
    titles = (
        "Truth $\\delta N_e$",
        "Forecast $\\delta N_e$",
        "Error $\\delta N_e$",
        "Truth $\\delta \\phi$",
        "Forecast $\\delta \\phi$",
        "Error $\\delta \\phi$",
    )
    for row, key in enumerate(ordered):
        candidate = prepared[key]
        arrays = (
            observed[0, :, :, z_plane],
            candidate[0, :, :, z_plane],
            candidate[0, :, :, z_plane] - observed[0, :, :, z_plane],
            observed[3, :, :, z_plane],
            candidate[3, :, :, z_plane],
            candidate[3, :, :, z_plane] - observed[3, :, :, z_plane],
        )
        for column, values in enumerate(arrays):
            ax = axes[row, column]
            field_channel = 0 if column < 3 else 3
            image = ax.pcolormesh(
                plot_geometry.major_radius,
                plot_geometry.vertical_position,
                np.ma.array(values, mask=mask),
                shading="auto",
                cmap="RdBu_r",
                vmin=-limits[field_channel],
                vmax=limits[field_channel],
                rasterized=True,
            )
            images[field_channel] = image
            _overlay_geometry(ax, plot_geometry, region_masks)
            if row == 0:
                ax.set_title(titles[column])
            if column == 0:
                ax.set_ylabel(f"{MODEL_LABELS[key]}\n$Z$ (m)")
            else:
                ax.set_yticklabels([])
            if row == len(ordered) - 1:
                ax.set_xlabel("$R$ (m)")
            else:
                ax.set_xticklabels([])
    fig.colorbar(
        images[0],
        ax=axes[:, :3],
        location="bottom",
        shrink=0.72,
        pad=0.025,
        label="$N_e$ fluctuation or error (training standard deviations)",
    )
    fig.colorbar(
        images[3],
        ax=axes[:, 3:],
        location="bottom",
        shrink=0.72,
        pad=0.025,
        label="$\\phi$ fluctuation or error (training standard deviations)",
    )
    fig.suptitle(
        f"Matched one-step forecast at frame {target_frame} · {3.1319:.4f} µs · stored toroidal plane z={z_plane}",
        fontsize=12,
        fontweight="normal",
    )
    path = Path(path)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def save_transport_profile_figure(
    path: Path,
    *,
    truth_local: np.ndarray,
    forecast_local: Mapping[str, np.ndarray],
    poloidal_distance: np.ndarray,
    target_frame: int = PHYSICS_FIGURE_TARGET,
) -> None:
    """Plot local and cumulative memberwise particle transport at one target."""

    import matplotlib.pyplot as plt

    _apply_plot_style()
    observed = _finite("truth particle transport", truth_local)
    if observed.shape != (16, 81):
        raise ValueError("truth local transport must be [16,81]")
    distance = _finite("poloidal distance", poloidal_distance)
    if distance.shape != (16,):
        raise ValueError("poloidal distance must contain 16 rows")
    if tuple(forecast_local) != tuple(MODEL_LABELS):
        raise ValueError("transport-profile model order differs")
    truth_curve = np.sum(observed, axis=-1, dtype=np.float64)
    truth_cumulative = np.cumsum(truth_curve)
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 6.0), sharex="col", constrained_layout=True)
    for column, key in enumerate(MODEL_LABELS):
        members = _finite(f"{key} local transport", forecast_local[key])
        expected_members = 1 if key == "deterministic" else PHYSICS_FIGURE_MEMBERS
        if members.shape != (expected_members, 16, 81):
            raise ValueError(f"{key} local transport member shape differs")
        curves = np.sum(members, axis=-1, dtype=np.float64)
        cumulative = np.cumsum(curves, axis=-1)
        color = MODEL_COLORS[key]
        display_count = min(PHYSICS_FIGURE_DISPLAY_MEMBERS, curves.shape[0])
        for values in curves[:display_count]:
            axes[0, column].plot(distance, values, color=color, alpha=0.17, linewidth=0.65)
        for values in cumulative[:display_count]:
            axes[1, column].plot(distance, values, color=color, alpha=0.17, linewidth=0.65)
        if curves.shape[0] > 1:
            lower, upper = np.quantile(curves, (0.05, 0.95), axis=0)
            axes[0, column].fill_between(distance, lower, upper, color=color, alpha=0.13, linewidth=0)
            lower, upper = np.quantile(cumulative, (0.05, 0.95), axis=0)
            axes[1, column].fill_between(distance, lower, upper, color=color, alpha=0.13, linewidth=0)
        axes[0, column].plot(distance, np.mean(curves, axis=0), color=color, linewidth=1.6, label="ensemble mean" if curves.shape[0] > 1 else "forecast")
        axes[1, column].plot(distance, np.mean(cumulative, axis=0), color=color, linewidth=1.6)
        axes[0, column].plot(distance, truth_curve, color=TRUTH_COLOR, linewidth=2.0, label="truth")
        axes[1, column].plot(distance, truth_cumulative, color=TRUTH_COLOR, linewidth=2.0)
        axes[0, column].axhline(0.0, color="#9CA3AF", linewidth=0.5)
        axes[1, column].axhline(0.0, color="#9CA3AF", linewidth=0.5)
        axes[0, column].set_title(MODEL_LABELS[key])
        axes[1, column].set_xlabel("distance along confined separatrix, $s$ (m)")
    axes[0, 0].set_ylabel("weighted particle contribution\n(Hermes transport units per cell)")
    axes[1, 0].set_ylabel("cumulative one-fifth-wedge\nparticle transport (Hermes units)")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle(
        f"Memberwise particle transport at frame {target_frame} · nonlinear diagnostic evaluated before ensemble reduction",
        fontsize=12,
        fontweight="normal",
    )
    path = Path(path)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def save_transport_variogram_figure(
    path: Path,
    *,
    separation_m: np.ndarray,
    h1: Mapping[str, Mapping[str, np.ndarray]],
    pgl_h4: Mapping[str, Mapping[str, np.ndarray]] | None,
) -> None:
    """Plot bootstrapped truth and ensemble-expected transport variograms."""

    import matplotlib.pyplot as plt

    _apply_plot_style()
    separation = _finite("toroidal separation", separation_m)
    if separation.shape != (len(TRANSPORT_VARIOGRAM_LAGS),):
        raise ValueError("transport-variogram separation shape differs")
    expected_h1 = ("truth", *tuple(MODEL_LABELS))
    if tuple(h1) != expected_h1:
        raise ValueError("h1 variogram model order differs")
    panels = 2 if pgl_h4 is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(12.2 if panels == 2 else 6.2, 4.2), sharey=True, constrained_layout=True)
    if panels == 1:
        axes = np.asarray([axes])

    def draw(ax: Any, records: Mapping[str, Mapping[str, np.ndarray]], title: str) -> None:
        for key, record in records.items():
            color = TRUTH_COLOR if key == "truth" else MODEL_COLORS[key]
            label = "truth" if key == "truth" else MODEL_LABELS[key]
            mean = _finite(f"{key} variogram mean", record["mean"])
            lower = _finite(f"{key} variogram lower", record["lower_2p5"])
            upper = _finite(f"{key} variogram upper", record["upper_97p5"])
            ax.fill_between(separation, lower, upper, color=color, alpha=0.10, linewidth=0)
            ax.plot(separation, mean, color=color, linewidth=2.0 if key == "truth" else 1.5, marker="o", markersize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("mean toroidal arc separation at separatrix (m)")
        ax.set_yscale("log")
        ax.grid(True, which="both", color="#E5E7EB", linewidth=0.55)

    draw(axes[0], h1, "one frame · 3.1319 µs")
    axes[0].legend(frameon=False, loc="best", fontsize=7.2)
    if pgl_h4 is not None:
        if tuple(pgl_h4) != ("truth", "persistent"):
            raise ValueError("h4 variogram model order differs")
        draw(axes[1], pgl_h4, "four frames · 12.5276 µs")
        axes[1].legend(frameon=False, loc="best", fontsize=8.0)
    axes[0].set_ylabel("first-order $|\\Delta\\Gamma_{cell}|$ (Hermes transport units)")
    fig.suptitle("Spatial scale of confined-separatrix particle-transport coherence", fontsize=12, fontweight="normal")
    path = Path(path)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def save_ne_phi_coupling_figure(
    path: Path,
    *,
    records: Mapping[str, Mapping[str, np.ndarray | int]],
) -> None:
    """Plot Ne-phi cross magnitude, phase, and coherence by physical n=5k."""

    import matplotlib.pyplot as plt

    _apply_plot_style()
    expected = ("truth", *tuple(MODEL_LABELS))
    if tuple(records) != expected:
        raise ValueError("cross-field model order differs")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), constrained_layout=True)
    for key, record in records.items():
        n = np.asarray(record["physical_n"], dtype=np.int64)[1:]
        color = TRUTH_COLOR if key == "truth" else MODEL_COLORS[key]
        label = "truth" if key == "truth" else MODEL_LABELS[key]
        linewidth = 2.1 if key == "truth" else 1.4
        axes[0].plot(n, np.maximum(np.asarray(record["cross_magnitude"])[1:], 1.0e-18), color=color, linewidth=linewidth, label=label)
        axes[1].plot(n, np.asarray(record["phase_degrees"])[1:], color=color, linewidth=linewidth)
        axes[2].plot(n, np.asarray(record["coherence"])[1:], color=color, linewidth=linewidth)
    for ax in axes:
        ax.axvspan(20, 35, color="#DCEAF2", alpha=0.75, linewidth=0)
        ax.set_xlabel("physical toroidal mode $n=5k$")
        ax.grid(True, color="#E5E7EB", linewidth=0.55)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("$|S_{N_e,\\phi}|$ (training-standardized units²)")
    axes[0].set_title("cross-spectrum magnitude")
    axes[1].set_ylabel("phase $\\arg S_{N_e,\\phi}$ (degrees)")
    axes[1].set_ylim(-180, 180)
    axes[1].set_title("cross-field phase")
    axes[2].set_ylabel("magnitude-squared coherence")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("cross-field coherence")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Memberwise density-potential coupling at one-frame horizon", fontsize=12, fontweight="normal")
    path = Path(path)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
