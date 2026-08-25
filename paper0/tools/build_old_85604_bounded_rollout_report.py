#!/usr/bin/env python3
"""Build the figure-first old-85604 bounded-rollout evidence report.

This tool consumes only already-scored development-run artifacts.  It does not
load a simulation, checkpoint, or held-out run, and it never evaluates a
physics quantity inside a training loop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
FIELD_LABELS = {
    "Ne": r"$N_e$",
    "Pe": r"$P_e$",
    "Pi": r"$P_i$",
    "phi": r"$\phi$",
    "Vi": r"$V_i$",
}
BANDS = ("k1_3", "k4_5", "k6_7")
BAND_LABELS = {
    "k1_3": r"$n=5$–$15$",
    "k4_5": r"$n=20$–$25$",
    "k6_7": r"$n=30$–$35$",
}
CROSS_PAIRS = ("Ne-phi", "Pe-phi", "Pi-phi")
PAIR_LABELS = {
    "Ne-phi": r"$N_e$–$\phi$",
    "Pe-phi": r"$P_e$–$\phi$",
    "Pi-phi": r"$P_i$–$\phi$",
}
QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)
QUANTITY_LABELS = {
    "particle": "particle",
    "electron_internal_energy": "electron energy",
    "ion_internal_energy": "ion energy",
    "total_internal_energy": "total energy",
}
SEEDS = ("1701", "1702", "1703")
HORIZONS = (4, 8)
CADENCE_US = 3.131905426352636
DECORRELATION_FRAMES = 2.244

METHOD_LABELS = {
    "persistence": "Persistence",
    "autoregressive_lead1": "Repeated 1-frame step",
    "autoregressive_lead2": "Repeated 2-frame step",
    "autoregressive_lead4": "Repeated 4-frame step",
    "direct": "Direct terminal prediction",
}
METHOD_SHORT = {
    "persistence": "persistence",
    "autoregressive_lead1": "1-frame steps",
    "autoregressive_lead2": "2-frame steps",
    "autoregressive_lead4": "4-frame steps",
    "direct": "direct terminal",
}
METHOD_COLORS = {
    "persistence": "#6f7882",
    "autoregressive_lead1": "#0072b2",
    "autoregressive_lead2": "#009e73",
    "autoregressive_lead4": "#cc79a7",
    "direct": "#d55e00",
}
METHOD_MARKERS = {
    "persistence": "x",
    "autoregressive_lead1": "o",
    "autoregressive_lead2": "s",
    "autoregressive_lead4": "D",
    "direct": "^",
}

STATE_SHA256 = "ddca83ca524c412d14a9db96bfdd2f09085b92f9998f865c331ba0932f4b3fe3"
PER_TARGET_SHA256 = "1d9d1d466e29c6e0d94236112dbdbeca887a3e97187f9e02ec1976243752bc67"
PHYSICS_SHA256 = "0215347da9b75ac8b8c2538844a6b8e4e78bffd1f6c6fe35937f7146129d40fc"
EXAMPLE_SHA256 = "5b6beba2c326373be3c313ccbaf942764eb08eec870a1deca2d08ddaa937297d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state-metrics", type=Path, required=True)
    parser.add_argument("--per-target", type=Path, required=True)
    parser.add_argument("--physics-metrics", type=Path, required=True)
    parser.add_argument("--example-fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def validate_inputs(
    state: Mapping[str, Any],
    physics: Mapping[str, Any],
    *,
    state_path: Path,
    per_target_path: Path,
    physics_path: Path,
    example_path: Path,
) -> None:
    expected_hashes = {
        state_path: STATE_SHA256,
        per_target_path: PER_TARGET_SHA256,
        physics_path: PHYSICS_SHA256,
        example_path: EXAMPLE_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{path} SHA-256 differs: {actual}")
    if (
        state.get("scope") != "post_ecrd_old_85604_bounded_rollout"
        or state.get("development_run") != "85604"
        or state.get("held_out_85606_read") is not False
        or state.get("new_nersc_data_read") is not False
        or state.get("guard_frames_read") is not False
        or state.get("physics_derived_metric") is not False
        or tuple(state.get("fields", ())) != FIELDS
        or tuple(sorted(int(item) for item in state.get("by_seed", {}))) != (1701, 1702, 1703)
    ):
        raise ValueError("bounded state metric contract differs")
    if (
        physics.get("scope") != "post_ecrd_old_85604_bounded_rollout_physics"
        or physics.get("development_run") != "85604"
        or physics.get("held_out_85606_read") is not False
        or physics.get("new_nersc_data_read") is not False
        or physics.get("guard_frames_read") is not False
        or physics.get("training_performed") is not False
        or physics.get("physics_derived_loss_used") is not False
        or physics.get("zperiod") != 5
        or physics.get("mode_mapping") != "n=5k"
        or tuple(physics.get("fields", ())) != FIELDS
        or tuple(sorted(int(item) for item in physics.get("by_seed", {}))) != (1701, 1702, 1703)
    ):
        raise ValueError("bounded physics metric contract differs")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "svg.hashsalt": "paper0-old-85604-bounded-rollout-20260825",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, description: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": stem.replace("-", " ").title(),
        "Description": description,
        "Date": "2026-08-25",
        "Source": "TCV/Hermes run 85604 development evidence only",
    }
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    fig.savefig(svg, metadata=metadata)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(png, dpi=220, metadata=metadata)
    plt.close(fig)


def method_order(horizon: int, *, include_persistence: bool = True) -> list[str]:
    methods = ["autoregressive_lead1", "autoregressive_lead2"]
    if horizon == 8:
        methods.append("autoregressive_lead4")
    methods.append("direct")
    return (["persistence"] if include_persistence else []) + methods


def method_legend_handles(methods: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.7,
            markersize=5,
            label=METHOD_SHORT[method],
        )
        for method in methods
    ]


def seed_interval(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("expected three finite seed values")
    return float(np.min(array)), float(np.median(array)), float(np.max(array))


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=9)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#4d5964", linewidth=1.1))


def plot_protocol(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.1), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("A. What the operator learns", loc="left", fontweight="bold")
    draw_box(ax, (0.04, 0.61), 0.25, 0.20, "Current five-field state\n$N_e, P_e, P_i, \\phi, V_i$", "#0072b2")
    draw_box(ax, (0.37, 0.61), 0.25, 0.20, "Requested lead\n1, 2, 4, 8, or 16 frames", "#009e73")
    draw_box(ax, (0.70, 0.61), 0.25, 0.20, "Joint state increment\non the full 64×32×88 grid", "#d55e00")
    draw_arrow(ax, (0.29, 0.71), (0.37, 0.71))
    draw_arrow(ax, (0.62, 0.71), (0.70, 0.71))
    ax.text(0.04, 0.43, "Codec-free", color="#0072b2", fontweight="bold")
    ax.text(0.04, 0.34, "Circular toroidal padding; no toroidal downsampling", color="#26313b")
    ax.text(0.04, 0.25, "2,174,021 trainable parameters; all five fields predicted jointly", color="#26313b")
    ax.text(0.04, 0.16, "Field-only derivative loss; no flux, spectrum, phase, or PDE term", color="#26313b")

    ax = axes[1]
    ax.set_title("B. What the bounded rollout compares", loc="left", fontweight="bold")
    draw_box(ax, (0.04, 0.70), 0.20, 0.15, "True start\nstate", "#6f7882")
    paths = [
        (0.34, 0.76, "one terminal jump", "#d55e00"),
        (0.34, 0.53, "repeated 4-frame steps", "#cc79a7"),
        (0.34, 0.30, "repeated 2-frame steps", "#009e73"),
        (0.34, 0.07, "repeated 1-frame steps", "#0072b2"),
    ]
    for x, y, label, color in paths:
        draw_box(ax, (x, y), 0.28, 0.13, label, color)
        draw_arrow(ax, (0.24, 0.77), (x, y + 0.065))
        draw_box(ax, (0.72, y), 0.23, 0.13, "same terminal\ntruth", color)
        draw_arrow(ax, (x + 0.28, y + 0.065), (0.72, y + 0.065))
    ax.text(0.04, 0.01, "Every intermediate autoregressive input is a complete predicted state; no future truth is fed back.", fontsize=8.2)
    fig.suptitle("Current experiment: finite-time learning separated from feedback error", fontsize=14, fontweight="bold")
    save_figure(fig, output_dir, "bounded-rollout-protocol", "Codec-free multilead training and matched bounded-rollout comparison.")


def plot_historical_ladder(repo: Path, output_dir: Path) -> None:
    source = strict_json(repo / "paper0/results/phase3_b5_residual_edm_one_seed_localization_6901661.json")
    models = source["model_comparison"]["models"]
    keys = (
        "H1 deterministic",
        "B3 functional noise",
        "B4 PDE-Refiner",
        "B5 joint residual EDM",
    )
    labels = (
        "Deterministic\nlatent",
        "Global-noise\nconditional",
        "Iterative\nrefiner",
        "Joint-residual\ndiffusion",
    )
    colors = ("#6f7882", "#0072b2", "#cc79a7", "#009e73")
    x = np.arange(4)
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.5), constrained_layout=True)

    values = [models[key]["rmse_relative_to_H1"] for key in keys]
    bars = axes[0].bar(x, values, color=colors)
    axes[0].axhline(1, color="#26313b", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.85, 1.04)
    axes[0].set_title("A. Mean-field RMSE", loc="left", fontweight="bold")
    axes[0].set_ylabel("relative to deterministic baseline")

    values = [models[key]["fair_crps_relative_to_H1_MAE"] for key in keys]
    axes[1].bar(x, values, color=colors)
    axes[1].axhline(1, color="#26313b", linestyle="--", linewidth=1)
    axes[1].set_ylim(0.58, 1.04)
    axes[1].set_title("B. Marginal fair CRPS", loc="left", fontweight="bold")
    axes[1].set_ylabel("relative to deterministic MAE")

    pkeys = keys[1:]
    px = np.arange(3)
    values = [models[key]["spread_skill"] for key in pkeys]
    axes[2].bar(px, values, color=colors[1:])
    axes[2].axhspan(0.8, 1.25, color="#009e73", alpha=0.12)
    axes[2].axhline(1, color="#26313b", linewidth=1)
    axes[2].set_ylim(0, 1.3)
    axes[2].set_title("C. Pixel spread / error", loc="left", fontweight="bold")
    axes[2].set_ylabel("1 is calibrated in aggregate")
    axes[2].set_xticks(px, labels[1:])

    width = 0.35
    power = [models[key]["power_checks_passing"] for key in pkeys]
    coherence = [models[key]["realization_coherence_checks_passing"] for key in pkeys]
    axes[3].bar(px - width / 2, power, width, color="#009e73", label="band-power checks")
    axes[3].bar(px + width / 2, coherence, width, color="#d55e00", label="realization checks")
    axes[3].set_ylim(0, 15.7)
    axes[3].set_title("D. Spectral checks passed", loc="left", fontweight="bold")
    axes[3].set_ylabel("of 15 field × mode-band checks")
    axes[3].set_xticks(px, labels[1:])
    axes[3].legend(loc="upper left", fontsize=8)

    for ax in axes[:2]:
        ax.set_xticks(x, labels)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.8)
        for patch in ax.patches:
            height = float(patch.get_height())
            ax.text(patch.get_x() + patch.get_width() / 2, height + 0.012, f"{height:.2f}" if height <= 2 else f"{height:.0f}", ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("Earlier one-step models improved marginals without recovering the realized joint field", fontsize=14, fontweight="bold")
    save_figure(fig, output_dir, "historical-model-ladder", "Descriptive comparison of the earlier one-step deterministic and probabilistic model families.")


def training_results(repo: Path) -> list[dict[str, Any]]:
    names = (
        "post_ecrd_old_85604_stage2_multilead_6936393.json",
        "post_ecrd_old_85604_stage2_multilead_seed1702_6936642.json",
        "post_ecrd_old_85604_stage2_multilead_seed1703_6936641.json",
    )
    results = [strict_json(repo / "paper0/results" / name) for name in names]
    if tuple(item["seed"] for item in results) != (1701, 1702, 1703):
        raise ValueError("multilead training seed order differs")
    return results


def plot_training(repo: Path, output_dir: Path) -> None:
    results = training_results(repo)
    epochs = np.arange(1, 5)
    train = np.asarray([[row["training_persistence_normalized_loss_mean"] for row in result["history"]] for result in results])
    valid = np.asarray([[row["validation"]["mean_shared_persistence_normalized_mse_ratio"] for row in result["history"]] for result in results])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)
    for ax, values, ylabel, title in (
        (axes[0], train, "training loss / persistence", "A. Field-only training objective"),
        (axes[1], valid, "validation error / persistence", "B. Five-lead chronological validation"),
    ):
        low, med, high = np.min(values, axis=0), np.median(values, axis=0), np.max(values, axis=0)
        ax.fill_between(epochs, low, high, color="#0072b2", alpha=0.16, label="three-seed range")
        ax.plot(epochs, med, color="#0072b2", marker="o", linewidth=2.2, label="seed median")
        for seed_values in values:
            ax.plot(epochs, seed_values, color="#0072b2", linewidth=0.8, alpha=0.35)
        ax.axhline(1, color="#6f7882", linestyle="--", linewidth=1, label="persistence")
        ax.set_xticks(epochs)
        ax.set_xlabel("completed epoch (533 optimizer updates each)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(fontsize=8)
    axes[0].set_ylim(0.20, 1.05)
    axes[1].set_ylim(0.43, 1.03)
    fig.suptitle("Explicit supervision at 1, 2, 4, 8, and 16 saved frames converges consistently", fontsize=14, fontweight="bold")
    save_figure(fig, output_dir, "multilead-training-curves", "Four-epoch three-seed training and chronological validation curves for the codec-free multilead operator.")


def state_terminal_record(state: Mapping[str, Any], seed: str, horizon: int, method: str) -> Mapping[str, Any]:
    return state["by_seed"][seed][str(horizon)]["terminal"][method]


def plot_state_skill(state: Mapping[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.1), constrained_layout=True)
    for ax, horizon in zip(axes, HORIZONS):
        methods = method_order(horizon, include_persistence=False)
        y = np.arange(len(methods))
        for index, method in enumerate(methods):
            low, med, high = seed_interval(
                state_terminal_record(state, seed, horizon, method)["mean_field_persistence_relative_skill"]
                for seed in SEEDS
            )
            ax.errorbar(
                med,
                index,
                xerr=np.asarray([[med - low], [high - med]]),
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=7,
                capsize=4,
                linewidth=2,
            )
            ax.text(high + 0.018, index, f"{med:.3f}", va="center", fontsize=8)
        ax.axvline(0, color="#6f7882", linestyle="--", linewidth=1)
        ax.set_yticks(y, [METHOD_LABELS[item] for item in methods])
        ax.invert_yaxis()
        ax.set_xlim(-0.08, 0.62)
        ax.set_xlabel(r"state skill $1-\mathrm{MSE}_{model}/\mathrm{MSE}_{persistence}$")
        ax.set_title(
            f"{chr(65 + HORIZONS.index(horizon))}. {horizon} frames = {horizon * CADENCE_US:.3f} µs = {horizon / DECORRELATION_FRAMES:.2f} decorrelation times",
            loc="left",
            fontweight="bold",
        )
    fig.suptitle("Every path beats persistence in state space, but the preferred step size changes with horizon", fontsize=14, fontweight="bold")
    save_figure(fig, output_dir, "bounded-rollout-state-skill", "Terminal equal-channel standardized state skill; points are seed medians and whiskers are seed ranges.")


def plot_error_growth(state: Mapping[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3), constrained_layout=True)
    horizon = 8
    composition = state["by_seed"][SEEDS[0]][str(horizon)]["composition_depth"]
    for method in composition:
        elapsed = sorted(int(key) for key in composition[method])
        x = [composition[method][str(key)]["elapsed_frames"] for key in elapsed]
        values = np.asarray(
            [
                [
                    state["by_seed"][seed][str(horizon)]["composition_depth"][method][str(key)]["mean_field_rmse"]
                    for key in elapsed
                ]
                for seed in SEEDS
            ]
        )
        axes[0].fill_between(x, np.min(values, axis=0), np.max(values, axis=0), color=METHOD_COLORS[method], alpha=0.12)
        axes[0].plot(x, np.median(values, axis=0), color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], linewidth=2, label=METHOD_LABELS[method])
    persistence = state["persistence"][str(horizon)]["mean_field_rmse"]
    axes[0].scatter([horizon], [persistence], marker="x", s=70, color=METHOD_COLORS["persistence"], label="Persistence at frame 8", zorder=5)
    axes[0].set_xlabel("elapsed saved frames")
    axes[0].set_ylabel("equal-channel standardized RMSE")
    axes[0].set_xticks([1, 2, 3, 4, 6, 8], [f"{x}\n{x * CADENCE_US:.1f} µs" for x in [1, 2, 3, 4, 6, 8]])
    axes[0].set_title("A. Error accumulated inside each autonomous composition", loc="left", fontweight="bold")
    axes[0].legend(fontsize=8)

    methods = method_order(8, include_persistence=False)
    matrix = np.asarray(
        [
            [
                np.median(
                    [state_terminal_record(state, seed, 8, method)["per_field"][field]["persistence_relative_skill"] for seed in SEEDS]
                )
                for field in FIELDS
            ]
            for method in methods
        ]
    )
    image = axes[1].imshow(matrix, cmap="RdYlBu", vmin=0, vmax=0.62, aspect="auto")
    axes[1].set_xticks(np.arange(len(FIELDS)), [FIELD_LABELS[item] for item in FIELDS])
    axes[1].set_yticks(np.arange(len(methods)), [METHOD_LABELS[item] for item in methods])
    axes[1].set_title("B. Eight-frame state skill by field", loc="left", fontweight="bold")
    axes[1].grid(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1].text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    colorbar.set_label("seed-median skill; positive beats persistence")
    fig.suptitle("Frequent small updates retain more structure but accumulate pressure error", fontsize=14, fontweight="bold")
    save_figure(fig, output_dir, "bounded-rollout-error-growth", "Error growth through eight-frame autonomous compositions and terminal field skill.")


def physics_record(physics: Mapping[str, Any], seed: str, horizon: int, method: str) -> Mapping[str, Any]:
    if method == "persistence":
        return physics["common_persistence"][str(horizon)]["field_spectral_cross"]
    return physics["by_seed"][seed][str(horizon)]["field_spectral_cross"][method]


def physics_values(
    physics: Mapping[str, Any],
    horizon: int,
    method: str,
    getter: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "persistence":
        value = np.asarray(getter(physics_record(physics, SEEDS[0], horizon, method)), dtype=float)
        return value, value, value
    values = np.asarray([getter(physics_record(physics, seed, horizon, method)) for seed in SEEDS], dtype=float)
    return np.min(values, axis=0), np.median(values, axis=0), np.max(values, axis=0)


def field_band_vector(record: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray([record["field_band_summaries"][field][band][key] for band in BANDS for field in FIELDS], dtype=float)


def plot_spectra(physics: Mapping[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 9.1), constrained_layout=True)
    x = np.arange(len(FIELDS) * len(BANDS))
    for row, horizon in enumerate(HORIZONS):
        for column, (key, ylabel, ideal) in enumerate(
            (
                ("power_ratio", "predicted / true band power", 1.0),
                ("truth_power_weighted_transfer_coherence", "prediction–truth transfer coherence", 1.0),
            )
        ):
            ax = axes[row, column]
            for method in method_order(horizon):
                low, med, high = physics_values(physics, horizon, method, lambda record, key=key: field_band_vector(record, key))
                ax.plot(x, med, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], markersize=3.8, linewidth=1.6, label=METHOD_SHORT[method])
                if method != "persistence":
                    ax.fill_between(x, low, high, color=METHOD_COLORS[method], alpha=0.09)
            ax.axhline(ideal, color="#26313b", linestyle="--", linewidth=1)
            for boundary in (4.5, 9.5):
                ax.axvline(boundary, color="#aeb5bb", linewidth=0.8)
            ax.set_xticks(x, [FIELD_LABELS[field] for _band in BANDS for field in FIELDS])
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{chr(65 + row * 2 + column)}. {horizon}-frame terminal · {'band amplitude' if column == 0 else 'same-realization placement'}",
                loc="left",
                fontweight="bold",
            )
            if column == 0:
                ax.set_yscale("log")
                ax.set_ylim(0.05, 4.2)
                ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4], ["0.1", "0.25", "0.5", "1", "2", "4"])
                ax.set_ylabel("predicted / true band power (log scale)")
            else:
                ax.set_ylim(0, 1.02)
            if row == 0 and column == 0:
                ax.legend(
                    handles=method_legend_handles(method_order(8)),
                    ncol=3,
                    fontsize=8,
                    loc="upper center",
                )
            top = ax.secondary_xaxis("top")
            top.set_xticks([2, 7, 12], [BAND_LABELS[band] for band in BANDS])
            top.tick_params(length=0, pad=4)
    fig.suptitle("Small-step composition preserves statistical power; all paths lose high-mode realization coherence", fontsize=14, fontweight="bold")
    save_figure(fig, output_dir, "bounded-rollout-spectral-fidelity", "Physical toroidal band-power ratios and prediction-to-truth transfer coherence; n=5k.")


def cross_vector(record: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray([record["cross_field_band_summaries"][pair][band][key] for band in BANDS for pair in CROSS_PAIRS], dtype=float)


def plot_cross_field(physics: Mapping[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.8), constrained_layout=True)
    x = np.arange(len(CROSS_PAIRS) * len(BANDS))
    metrics = (
        ("truth_cross_amplitude_weighted_absolute_phase_error_degrees", "absolute cross-phase error (degrees)", 20.0),
        ("truth_cross_amplitude_weighted_absolute_coherence_change", "absolute cross-coherence change", 0.15),
    )
    for row, horizon in enumerate(HORIZONS):
        for column, (key, ylabel, guide) in enumerate(metrics):
            ax = axes[row, column]
            for method in method_order(horizon):
                low, med, high = physics_values(physics, horizon, method, lambda record, key=key: cross_vector(record, key))
                ax.plot(x, med, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], markersize=4, linewidth=1.7, label=METHOD_SHORT[method])
                if method != "persistence":
                    ax.fill_between(x, low, high, color=METHOD_COLORS[method], alpha=0.09)
            ax.axhline(guide, color="#d55e00", linestyle="--", linewidth=1, label="historical one-step guide" if row == 0 and column == 0 else None)
            for boundary in (2.5, 5.5):
                ax.axvline(boundary, color="#aeb5bb", linewidth=0.8)
            ax.set_xticks(x, [PAIR_LABELS[pair] for _band in BANDS for pair in CROSS_PAIRS])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{chr(65 + row * 2 + column)}. {horizon}-frame terminal", loc="left", fontweight="bold")
            top = ax.secondary_xaxis("top")
            top.set_xticks([1, 4, 7], [BAND_LABELS[band] for band in BANDS])
            top.tick_params(length=0, pad=4)
            if row == 0 and column == 0:
                guide_handle = Line2D(
                    [0],
                    [0],
                    color="#d55e00",
                    linestyle="--",
                    linewidth=1,
                    label="historical one-step guide",
                )
                ax.legend(
                    handles=[*method_legend_handles(method_order(8)), guide_handle],
                    ncol=3,
                    fontsize=8,
                    loc="upper center",
                )
    fig.suptitle("Mean cross-phase can remain plausible while cross-field coherence and realization placement fail", fontsize=14, fontweight="bold")
    save_figure(fig, output_dir, "bounded-rollout-cross-field", "Density/pressure–potential cross-phase error and coherence change by physical toroidal band.")


def transport_record(physics: Mapping[str, Any], seed: str, horizon: int, method: str, quantity: str) -> Mapping[str, Any]:
    if method == "persistence":
        root = physics["common_persistence"][str(horizon)]["transport"]
    else:
        root = physics["by_seed"][seed][str(horizon)]["transport"]
    return root["comparisons"][f"truth_vs_{method}"]["quantities"][quantity]


def transport_values(
    physics: Mapping[str, Any], horizon: int, method: str, reduction: str, key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "persistence":
        value = np.asarray([transport_record(physics, SEEDS[0], horizon, method, quantity)[reduction]["metrics"][key] for quantity in QUANTITIES], dtype=float)
        return value, value, value
    values = np.asarray(
        [
            [transport_record(physics, seed, horizon, method, quantity)[reduction]["metrics"][key] for quantity in QUANTITIES]
            for seed in SEEDS
        ],
        dtype=float,
    )
    return np.min(values, axis=0), np.median(values, axis=0), np.max(values, axis=0)


def plot_transport(physics: Mapping[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.2), constrained_layout=True)
    x = np.arange(len(QUANTITIES))
    metrics = (
        ("strict_faces", "relative_l2", "strict local-face relative L2", (0, 1.55)),
        ("separatrix", "relative_l2", "integrated separatrix relative L2", (0, 1.05)),
        ("separatrix", "pearson_correlation", "integrated separatrix correlation", (-0.45, 1.0)),
    )
    for row, horizon in enumerate(HORIZONS):
        for column, (reduction, key, ylabel, ylim) in enumerate(metrics):
            ax = axes[row, column]
            offset = np.linspace(-0.24, 0.24, len(method_order(horizon)))
            for delta, method in zip(offset, method_order(horizon)):
                low, med, high = transport_values(physics, horizon, method, reduction, key)
                ax.errorbar(
                    x + delta,
                    med,
                    yerr=np.vstack([med - low, high - med]),
                    color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method],
                    markersize=5,
                    linewidth=1.5,
                    capsize=2.5,
                    label=METHOD_SHORT[method],
                )
            ax.axhline(0 if key == "relative_l2" else 1, color="#26313b", linestyle="--", linewidth=1)
            ax.set_xticks(x, [QUANTITY_LABELS[item].replace(" ", "\n") for item in QUANTITIES])
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{chr(65 + row * 3 + column)}. {horizon}-frame terminal", loc="left", fontweight="bold")
            if row == 0 and column == 0:
                ax.legend(
                    handles=method_legend_handles(method_order(8)),
                    ncol=2,
                    fontsize=8,
                    loc="lower left",
                )
    fig.suptitle("Repeated one-frame forecasts best preserve integrated transport, but local transport remains poor", fontsize=14, fontweight="bold")
    save_figure(fig, output_dir, "bounded-rollout-transport", "Geometry-aware local-face and integrated confined-separatrix transport metrics.")


def surface_series(physics: Mapping[str, Any], seed: str, horizon: int, method: str, quantity: str) -> np.ndarray:
    if method == "persistence":
        root = physics["common_persistence"][str(horizon)]["transport"]
    else:
        root = physics["by_seed"][seed][str(horizon)]["transport"]
    return np.asarray(root["surface_series_normalized"][method][quantity], dtype=float)


def truth_series(physics: Mapping[str, Any], horizon: int, quantity: str) -> np.ndarray:
    return np.asarray(physics["common_persistence"][str(horizon)]["transport"]["surface_series_normalized"]["truth"][quantity], dtype=float)


def plot_transport_traces(physics: Mapping[str, Any], output_dir: Path) -> None:
    horizon = 8
    selected_quantities = ("particle", "total_internal_energy")
    selected_methods = ("persistence", "autoregressive_lead1", "autoregressive_lead4", "direct")
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.8), sharex=True, constrained_layout=True)
    for ax, quantity, letter in zip(axes, selected_quantities, ("A", "B")):
        truth = truth_series(physics, horizon, quantity)
        scale = float(np.sqrt(np.mean(np.square(truth))))
        target_frames = np.arange(496 + horizon, 624)
        elapsed_us = (target_frames - target_frames[0]) * CADENCE_US
        ax.plot(elapsed_us, truth / scale, color="#111820", linewidth=2.2, label="truth")
        for method in selected_methods:
            if method == "persistence":
                med = surface_series(physics, SEEDS[0], horizon, method, quantity) / scale
                low = high = med
            else:
                values = np.asarray([surface_series(physics, seed, horizon, method, quantity) for seed in SEEDS]) / scale
                low, med, high = np.min(values, axis=0), np.median(values, axis=0), np.max(values, axis=0)
            ax.plot(elapsed_us, med, color=METHOD_COLORS[method], linewidth=1.35, label=METHOD_SHORT[method])
            if method != "persistence":
                ax.fill_between(elapsed_us, low, high, color=METHOD_COLORS[method], alpha=0.10)
        ax.axhline(0, color="#6f7882", linewidth=0.8)
        ax.set_ylabel("wedge transport / truth RMS")
        ax.set_title(f"{letter}. {QUANTITY_LABELS[quantity].capitalize()} transport", loc="left", fontweight="bold")
    axes[0].legend(ncol=5, fontsize=8, loc="upper center")
    axes[-1].set_xlabel("chronological target time within rolling-origin validation (µs)")
    fig.suptitle("Eight-frame rolling-origin forecasts: small steps retain transport timing; direct jumps collapse toward a smooth trace", fontsize=14, fontweight="bold")
    fig.text(0.5, -0.01, "Each point is the terminal value of a separate 25.055-µs forecast; this is not one uninterrupted free trajectory. Bands show the three-seed range.", ha="center", fontsize=8.5)
    save_figure(fig, output_dir, "bounded-rollout-transport-traces", "Rolling-origin terminal particle and total-energy separatrix transport traces at horizon eight.")


def fluctuation_plane(values: np.ndarray, field_index: int, poloidal_index: int = 16) -> np.ndarray:
    plane = np.asarray(values[field_index, :, poloidal_index, :], dtype=float)
    return plane - np.mean(plane, axis=-1, keepdims=True)


def plot_example_fields(example_path: Path, output_dir: Path) -> None:
    with np.load(example_path) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    methods = ("truth", "autoregressive_lead1", "autoregressive_lead4", "direct")
    titles = ("Truth at frame 568", "Eight 1-frame steps", "Two 4-frame steps", "Direct 8-frame prediction")
    fields = (("Pe", 1), ("phi", 3))
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.1), constrained_layout=True)
    for row, (field, field_index) in enumerate(fields):
        target = fluctuation_plane(arrays["h8_truth"], field_index)
        candidates: list[np.ndarray] = []
        for method in methods:
            if method == "truth":
                candidates.append(target)
            else:
                stack = np.stack([fluctuation_plane(arrays[f"h8_seed{seed}_{method}"], field_index) for seed in SEEDS])
                candidates.append(np.median(stack, axis=0))
        limit = float(np.quantile(np.abs(target), 0.995))
        for column, (candidate, title) in enumerate(zip(candidates, titles)):
            image = axes[row, column].imshow(candidate, origin="lower", aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit))
            axes[row, column].set_title(title if row == 0 else "", fontsize=10)
            axes[row, column].set_xlabel("toroidal cell (88; periodic)")
            if column == 0:
                axes[row, column].set_ylabel(f"{FIELD_LABELS[field]} fluctuation\nradial cell")
            else:
                axes[row, column].set_yticklabels([])
            axes[row, column].grid(False)
        colorbar = fig.colorbar(image, ax=axes[row, :], fraction=0.012, pad=0.008)
        colorbar.set_label("simulation-normalized fluctuation")
    fig.suptitle("One 25.055-µs example: lower pixel error can coincide with visibly depleted toroidal structure", fontsize=14, fontweight="bold")
    fig.text(0.5, -0.01, "Radial–toroidal plane at fixed poloidal index y=16. Predictions are pointwise medians over the three independently trained seeds; nonlinear metrics were computed seed by seed.", ha="center", fontsize=8.5)
    save_figure(fig, output_dir, "bounded-rollout-example-fields", "Representative radial–toroidal pressure and potential fluctuation planes at frame 568.")


def write_state_csv(state: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for method in method_order(horizon, include_persistence=False):
            for seed in SEEDS:
                record = state_terminal_record(state, seed, horizon, method)
                rows.append(
                    {
                        "seed": seed,
                        "horizon_frames": horizon,
                        "horizon_microseconds": horizon * CADENCE_US,
                        "horizon_decorrelation_times": horizon / DECORRELATION_FRAMES,
                        "method": method,
                        "mean_field_rmse": record["mean_field_rmse"],
                        "mean_field_persistence_relative_skill": record["mean_field_persistence_relative_skill"],
                        **{f"{field}_skill": record["per_field"][field]["persistence_relative_skill"] for field in FIELDS},
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_spectral_csv(physics: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for method in method_order(horizon):
            seeds: Sequence[str | None] = (None,) if method == "persistence" else SEEDS
            for seed in seeds:
                record = physics_record(physics, seed or SEEDS[0], horizon, method)
                for band in BANDS:
                    for field in FIELDS:
                        values = record["field_band_summaries"][field][band]
                        rows.append(
                            {
                                "seed": "baseline" if seed is None else seed,
                                "horizon_frames": horizon,
                                "method": method,
                                "field": field,
                                "band": band,
                                "n_low": values["n_low"],
                                "n_high": values["n_high"],
                                "power_ratio": values["power_ratio"],
                                "truth_power_weighted_transfer_coherence": values["truth_power_weighted_transfer_coherence"],
                            }
                        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_cross_csv(physics: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for method in method_order(horizon):
            seeds: Sequence[str | None] = (None,) if method == "persistence" else SEEDS
            for seed in seeds:
                record = physics_record(physics, seed or SEEDS[0], horizon, method)
                for band in BANDS:
                    for pair in CROSS_PAIRS:
                        values = record["cross_field_band_summaries"][pair][band]
                        rows.append(
                            {
                                "seed": "baseline" if seed is None else seed,
                                "horizon_frames": horizon,
                                "method": method,
                                "pair": pair,
                                "band": band,
                                "n_low": values["n_low"],
                                "n_high": values["n_high"],
                                "phase_error_degrees": values["truth_cross_amplitude_weighted_absolute_phase_error_degrees"],
                                "absolute_coherence_change": values["truth_cross_amplitude_weighted_absolute_coherence_change"],
                            }
                        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_transport_csv(physics: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for method in method_order(horizon):
            seeds: Sequence[str | None] = (None,) if method == "persistence" else SEEDS
            for seed in seeds:
                for quantity in QUANTITIES:
                    record = transport_record(physics, seed or SEEDS[0], horizon, method, quantity)
                    for reduction in ("strict_faces", "separatrix"):
                        values = record[reduction]["metrics"]
                        rows.append(
                            {
                                "seed": "baseline" if seed is None else seed,
                                "horizon_frames": horizon,
                                "method": method,
                                "quantity": quantity,
                                "reduction": reduction,
                                "relative_l2": values["relative_l2"],
                                "normalized_bias": values["normalized_bias"],
                                "rms_ratio": values["rms_ratio"],
                                "pearson_correlation": values["pearson_correlation"],
                                "weighted_sign_disagreement": values["weighted_sign_disagreement"],
                            }
                        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def median_transport_summary(physics: Mapping[str, Any], horizon: int, method: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for quantity in QUANTITIES:
        records = [transport_record(physics, seed, horizon, method, quantity) for seed in SEEDS]
        result[quantity] = {
            "strict_face_relative_l2_median": float(np.median([record["strict_faces"]["metrics"]["relative_l2"] for record in records])),
            "separatrix_relative_l2_median": float(np.median([record["separatrix"]["metrics"]["relative_l2"] for record in records])),
            "separatrix_correlation_median": float(np.median([record["separatrix"]["metrics"]["pearson_correlation"] for record in records])),
        }
    return result


def write_summary_json(state: Mapping[str, Any], physics: Mapping[str, Any], path: Path) -> None:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_bounded_rollout_evidence_summary",
        "development_run": "85604",
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "guard_frames_read": False,
        "physics_derived_loss_used": False,
        "source_sha256": {
            "state_metrics": STATE_SHA256,
            "per_target_state_rmse": PER_TARGET_SHA256,
            "physics_metrics": PHYSICS_SHA256,
            "example_fields": EXAMPLE_SHA256,
        },
        "cadence_microseconds": CADENCE_US,
        "representative_decorrelation_frames": DECORRELATION_FRAMES,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "seed_interpretation": "three_independent_training_initializations_not_a_probabilistic_forecast_ensemble",
        "horizons": {},
        "scientific_conclusion": (
            "Coarse/direct steps minimize terminal field error but strongly deplete toroidal power and integrated transport. "
            "Repeated one-frame steps preserve the evaluated statistics and separatrix transport best, yet still lose high-mode realization coherence and local transport fidelity."
        ),
        "next_experiment": "field_only_four_step_autoregressive_finetune_of_the_codec_free_one_frame_operator",
    }
    for horizon in HORIZONS:
        horizon_record: dict[str, Any] = {}
        for method in method_order(horizon, include_persistence=False):
            horizon_record[method] = {
                "state_skill_seed_median": float(np.median([state_terminal_record(state, seed, horizon, method)["mean_field_persistence_relative_skill"] for seed in SEEDS])),
                "state_skill_seed_range": [
                    float(np.min([state_terminal_record(state, seed, horizon, method)["mean_field_persistence_relative_skill"] for seed in SEEDS])),
                    float(np.max([state_terminal_record(state, seed, horizon, method)["mean_field_persistence_relative_skill"] for seed in SEEDS])),
                ],
                "transport": median_transport_summary(physics, horizon, method),
            }
        summary["horizons"][str(horizon)] = horizon_record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def report_html() -> str:
    figure_root = "../figures/post_ecrd_old_85604_bounded_rollout"
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Figure-first evidence report for transport-faithful emulation on TCV/Hermes development run 85604.">
<title>Paper 0 — bounded rollout evidence on TCV/Hermes 85604</title>
<style>
:root{{--paper:#fff;--ground:#f2f4f5;--ink:#17212b;--muted:#5d6974;--rule:#dfe4e8;--teal:#087f8c;--teal-soft:#e6f4f4;--orange:#a94f16;--orange-soft:#fff0e6;--serif:"Iowan Old Style","Charter",Georgia,serif;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;--mono:"SFMono-Regular",Consolas,monospace}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--ground);color:var(--ink);font-family:var(--serif);font-size:18px;line-height:1.58}}.page{{max-width:76rem;margin:auto;background:var(--paper);min-height:100vh;padding:0 3rem 5rem;box-shadow:0 0 45px rgba(20,30,40,.07)}}header{{padding:4rem 0 2.5rem;border-bottom:1px solid var(--rule)}}.eyebrow,.section-label{{font-family:var(--sans);text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;color:var(--teal);font-weight:700}}h1{{font-size:3.2rem;line-height:1.04;letter-spacing:-.03em;margin:.6rem 0 1rem;max-width:66rem}}h2{{font-size:1.8rem;line-height:1.18;margin:.2rem 0 .7rem}}h3{{font-size:1.12rem;margin:0 0 .35rem}}.dek,.lede{{max-width:58rem;color:var(--muted)}}.dek{{font-size:1.18rem;line-height:1.45}}.meta{{display:flex;gap:.6rem 1.3rem;flex-wrap:wrap;font:0.78rem var(--sans);color:var(--muted);margin-top:1.2rem}}nav{{border-bottom:1px solid var(--rule);overflow-x:auto}}nav div{{display:flex;gap:1.2rem;padding:.9rem 0;white-space:nowrap}}nav a{{font:0.78rem var(--sans);color:var(--muted);text-decoration:none}}section{{padding:3.1rem 0;border-bottom:1px solid var(--rule);scroll-margin-top:1rem}}.tldr{{border-left:4px solid var(--teal);background:var(--teal-soft);padding:1.1rem 1.4rem;max-width:62rem}}.tldr li{{margin:.42rem 0}}.next{{border-left-color:var(--orange);background:var(--orange-soft)}}.definitions{{display:grid;grid-template-columns:repeat(3,1fr);gap:1.2rem;margin:1.5rem 0}}.definitions>div{{border-top:3px solid var(--rule);padding-top:.75rem}}.definitions p{{font:0.86rem/1.5 var(--sans);color:var(--muted);margin:.2rem 0}}figure{{margin:2rem 0 2.4rem}}figure img{{display:block;width:100%;height:auto;border:1px solid var(--rule);background:#fff}}figcaption{{font:0.83rem/1.48 var(--sans);color:var(--muted);padding:.65rem .15rem 0}}.fignum{{color:var(--teal);font-weight:700}}.source{{display:block;font:0.72rem/1.45 var(--mono);margin-top:.25rem;color:#7a858f}}.claim-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.4rem}}.claim-grid>div{{border-top:3px solid var(--rule);padding-top:.8rem}}code{{font-family:var(--mono);font-size:.82em}}footer{{padding:2rem 0;color:var(--muted);font:0.78rem/1.5 var(--sans)}}@media(max-width:760px){{body{{font-size:16px}}.page{{padding:0 1rem 3rem;box-shadow:none}}h1{{font-size:2.25rem}}.definitions,.claim-grid{{grid-template-columns:1fr}}}}@media print{{body{{background:#fff;font-size:10.5pt}}.page{{max-width:none;padding:0;box-shadow:none}}nav{{display:none}}figure{{break-inside:avoid}}section{{padding:1.5rem 0}}h1{{font-size:28pt}}}}
</style>
</head>
<body><div class="page">
<header><p class="eyebrow">Paper 0 · current state · 25 August 2026</p><h1>What the codec-free emulator preserves—and what it still destroys</h1><p class="dek">A figure-first account of the complete old-85604 model ladder, the new three-seed multilead operator, autonomous feedback at four and eight saved frames, and the transport reversal hidden by ordinary field error.</p><div class="meta"><span>development simulation 85604 only</span><span>85606 remains sealed</span><span>new NERSC data not read</span><span>physics used only for evaluation</span><span>W&B run p0oldboundphys-j6937203</span></div></header>
<nav aria-label="Contents"><div><a href="#answer">Answer</a><a href="#definitions">Definitions</a><a href="#arc">Research arc</a><a href="#training">Current model</a><a href="#state">State rollout</a><a href="#spectra">Spectra</a><a href="#transport">Transport</a><a href="#fields">Fields</a><a href="#decision">Decision</a></div></nav>
<main>
<section id="answer"><p class="section-label">01 · One-minute answer</p><h2>The model learned a useful transition, but not yet a transport-faithful emulator</h2><div class="tldr"><ul><li><strong>Finite-time state prediction works in aggregate.</strong> Every tested learned path has positive mean five-field skill at four and eight frames for all three training seeds. The one exception at field level is seed 1701’s eight-frame repeated one-frame path, whose Pe and Pi skills are slightly negative.</li><li><strong>Pixel accuracy and transport fidelity rank the paths differently.</strong> At eight frames, two four-frame steps have the best state skill, while eight one-frame steps preserve toroidal power and separatrix transport best.</li><li><strong>Direct long jumps behave like conditional-mean smoothers.</strong> Their average evaluated band power falls to roughly 11–17% of truth and their integrated transport error is about twice persistence.</li><li><strong>Small steps are the positive result, not the final answer.</strong> They improve eight-frame separatrix transport over persistence for every quantity and every seed, but local-face error remains above one and high-mode realization coherence is essentially lost.</li><li><strong>No assimilation claim is open.</strong> This is a deterministic three-seed study, not a calibrated forecast ensemble.</li></ul></div></section>

<section id="definitions"><p class="section-label">02 · Definitions and scope</p><h2>Exactly what was forecast</h2><div class="definitions"><div><h3>Fields</h3><p><strong>Ne</strong> electron density; <strong>Pe</strong> electron pressure; <strong>Pi</strong> ion pressure; <strong>φ</strong> electrostatic potential; <strong>Vi</strong> parallel ion velocity. They are predicted jointly on a 64×32×88 grid.</p></div><div><h3>Time</h3><p>One saved frame is 3.131905 µs. Four and eight frames are 12.528 and 25.055 µs, or 1.78 and 3.56 representative decorrelation times.</p></div><div><h3>Toroidal modes</h3><p>The saved wedge is one fifth of the torus. Stored Fourier index k therefore maps to physical mode n=5k. The reported bands are n=5–15, 20–25, and 30–35.</p></div><div><h3>State skill</h3><p>Skill is 1 − model MSE / persistence MSE in training-normalized field space. Positive values beat a frozen current state; one would be exact.</p></div><div><h3>Uncertainty shown</h3><p>Points are medians across seeds 1701, 1702, and 1703; bands or whiskers are their minimum–maximum range. These are training replicates, not stochastic ensemble members.</p></div><div><h3>Transport</h3><p>The validated geometry-aware operator computes local radial face contributions and their integral over the confined separatrix in the simulated one-fifth wedge.</p></div></div><figure><img src="{figure_root}/bounded-rollout-protocol.svg" alt="Protocol diagram for multilead training and bounded autonomous rollout comparison"><figcaption><span class="fignum">Figure 1.</span> The operator predicts the complete five-field increment and never compresses through a learned codec. The bounded comparison holds start frames and terminal truth fixed while varying only how the requested interval is traversed.<span class="source">Forecast job 6937051 · scoring job 6937203 · no future truth enters an autoregressive context</span></figcaption></figure></section>

<section id="arc"><p class="section-label">03 · Research arc</p><h2>Why the project moved beyond marginal calibration</h2><p class="lede">Earlier one-step models explored deterministic prediction, global functional noise, iterative refinement, and joint field-residual diffusion. The most probabilistically useful model improved fair CRPS and aggregate pixel spread, but every stochastic family placed only four of fifteen material spectral bands in the correct next-frame realization.</p><figure><img src="{figure_root}/historical-model-ladder.svg" alt="Earlier deterministic and probabilistic one-step model comparison"><figcaption><span class="fignum">Figure 2.</span> Lower mean error and fair CRPS did not raise realization-level spectral success. The labels describe mechanisms rather than internal experiment codes; the comparison is descriptive, not a single-factor ablation.<span class="source">Tracked compact result phase3_b5_residual_edm_one_seed_localization_6901661.json · 85604 validation</span></figcaption></figure></section>

<section id="training"><p class="section-label">04 · Current model</p><h2>A lead-time-conditioned, codec-free, toroidally equivariant transition operator</h2><p class="lede">The present model predicts normalized state derivatives from one complete C5P state. It downsamples only the two nonperiodic axes, uses circular toroidal padding, and is supervised directly at leads 1, 2, 4, 8, and 16. No spectrum, phase, flux, transport, PDE, or conservation quantity appears in its loss.</p><figure><img src="{figure_root}/multilead-training-curves.svg" alt="Three-seed training and chronological validation curves"><figcaption><span class="fignum">Figure 3.</span> All three seeds converge to nearly the same four-epoch solution after 2,132 optimizer updates. The checkpoint metric is the unweighted mean persistence-normalized Ne/Pe/Pi derivative error over all five trained leads.<span class="source">Training jobs 6936393, 6936642, and 6936641 · selected epoch 4 for every seed</span></figcaption></figure></section>

<section id="state"><p class="section-label">05 · Autonomous state rollout</p><h2>Feedback is not generically broken; composition depth matters</h2><p class="lede">At four frames, repeated one- and two-frame updates outperform a direct four-frame jump. At eight frames, two four-frame updates have the lowest ordinary state error. Eight one-frame updates accumulate the largest pressure error, even though they later prove best on spectra and transport.</p><figure><img src="{figure_root}/bounded-rollout-state-skill.svg" alt="State skill by terminal horizon and rollout path"><figcaption><span class="fignum">Figure 4.</span> Every seed has positive mean five-field skill for every learned path. At field level, seed 1701’s eight-frame repeated one-frame Pe and Pi skills are −0.021 and −0.025; all other field/seed/path entries are positive. Error bars are seed ranges, not sampling confidence intervals.<span class="source">state_metrics.json SHA-256 {STATE_SHA256}</span></figcaption></figure><figure><img src="{figure_root}/bounded-rollout-error-growth.svg" alt="Autonomous error growth and field-specific eight-frame skill"><figcaption><span class="fignum">Figure 5.</span> The one-frame map accumulates error smoothly; the eight-frame pressure fields are its weakest outputs. The lower pixel error of the coarser maps is consistent with conditional-mean smoothing over the requested interval.<span class="source">41,240 matched per-target field records · per_target_state_rmse.csv SHA-256 {PER_TARGET_SHA256}</span></figcaption></figure></section>

<section id="spectra"><p class="section-label">06 · Spectral and cross-field physics</p><h2>The field-error winner is the spectral loser</h2><p class="lede">Persistence nearly preserves aggregate power and mean cross-phase simply because those statistics change slowly; it does not place the future realization correctly. Repeated one-frame forecasts add genuine short-horizon placement skill and preserve much more power. Direct and four-frame paths lose most nonaxisymmetric power by the eight-frame terminal.</p><figure><img src="{figure_root}/bounded-rollout-spectral-fidelity.svg" alt="Toroidal band power ratios and realization transfer coherence"><figcaption><span class="fignum">Figure 6.</span> The repeated one-frame path is closest to the true power distribution, particularly in n=20–25. At four frames it retains meaningful mid-band transfer coherence; by eight frames every method is near zero in high-mode realization coherence. Matching power after decorrelation is not the same as predicting the exact eddies.<span class="source">member-free deterministic metrics evaluated seed by seed · zperiod=5 · n=5k</span></figcaption></figure><figure><img src="{figure_root}/bounded-rollout-cross-field.svg" alt="Density and pressure versus potential phase and coherence errors"><figcaption><span class="fignum">Figure 7.</span> Mean density/pressure–potential phase stays comparatively stable for small steps, but coherence changes grow. The direct eight-frame path substantially degrades both. Persistence’s tiny phase error reflects stationary aggregate cross-statistics, not forecast skill.<span class="source">cross-spectra computed in physical decoded field space · no cross-field term used in training</span></figcaption></figure></section>

<section id="transport"><p class="section-label">07 · Transport</p><h2>Small steps recover useful integrated transport—but not its local spatial organization</h2><p class="lede">At four frames, repeated one-frame forecasts reduce separatrix relative L2 error to 0.28–0.34 across quantities, compared with 0.41–0.42 for persistence. At eight frames, they remain better than persistence for every quantity and seed. Their strict local-face error is still about 1.38 at eight frames, so the spatial transport map is not faithful.</p><figure><img src="{figure_root}/bounded-rollout-transport.svg" alt="Local face and integrated separatrix transport metrics"><figcaption><span class="fignum">Figure 8.</span> State-optimal coarse steps are transport-poor: direct and two-four-frame forecasts have integrated errors near 0.9 at eight frames, roughly twice persistence. Repeated one-frame forecasts retain the best integrated magnitude and timing, but they do not pass a strong local-transport criterion.<span class="source">authoritative native-81 geometry and BOUT++-matched radial face operator · nonlinear transport computed for each seed prediction before aggregation</span></figcaption></figure><figure><img src="{figure_root}/bounded-rollout-transport-traces.svg" alt="Rolling-origin particle and total energy separatrix transport traces"><figcaption><span class="fignum">Figure 9.</span> Each plotted point is the end of a separate eight-frame autonomous forecast. Small steps track a substantial fraction of the chronological transport variation; coarse and direct paths collapse toward smooth, poorly correlated traces.<span class="source">targets 504–623 · fixed forecast interval 25.055 µs · traces normalized by truth RMS for each quantity</span></figcaption></figure></section>

<section id="fields"><p class="section-label">08 · What the terminal fields look like</p><h2>The smoothing is visible before transport is calculated</h2><p class="lede">The example below removes the toroidal mean at each radial cell to expose nonaxisymmetric fluctuation structure. It is a visualization of one fixed rolling-origin start, not a cherry-picked performance statistic.</p><figure><img src="{figure_root}/bounded-rollout-example-fields.svg" alt="Terminal pressure and potential fluctuation planes"><figcaption><span class="fignum">Figure 10.</span> Direct and repeated four-frame forecasts are smoother than truth. Repeated one-frame forecasts retain more toroidal texture, but that texture is not placed with high realization coherence after 3.56 decorrelation times.<span class="source">start frame 560 → target frame 568 · fixed poloidal index y=16 · pointwise seed median shown only for visualization</span></figcaption></figure></section>

<section id="decision"><p class="section-label">09 · Decision</p><h2>One next experiment: train the small-step map on its own four-step feedback</h2><div class="tldr next"><p><strong>Hypothesis:</strong> the one-frame operator already preserves the transport-bearing statistics best, but it was selected only by teacher-forced direct errors. A four-step free-running, field-only fine-tune should reduce its accumulated state drift without teaching it to collapse toward a smooth long-lead conditional mean.</p><ul><li>Initialize from each frozen three-seed multilead checkpoint.</li><li>Roll the one-frame map autonomously for four steps during training; never replace a predicted intermediate state with truth.</li><li>Supervise all four true future fields only through the existing channel-normalized state loss.</li><li>Retain a one-step term so short-horizon accuracy cannot be traded away silently.</li><li>Select checkpoints with state-only chronological validation, then evaluate the same frozen spectral, cross-field, and transport suite.</li></ul><p>If this does not improve four/eight-frame state stability while retaining the one-frame path’s physics, stop tuning this local deterministic operator and move to the planned nonlocal/state-complete or persistent stochastic architecture.</p></div><div class="claim-grid"><div><h3>Supported now</h3><ul><li>The codec-free operator learns finite-time 85604 transitions reproducibly.</li><li>Autonomous feedback can beat direct prediction; the effect depends on learned step size.</li><li>Field RMSE is not a valid model-selection proxy for transport.</li><li>Small-step forecasts contain useful integrated transport information beyond persistence.</li></ul></div><div><h3>Still unsupported</h3><ul><li>No path is transport-faithful locally.</li><li>No deterministic seed set is a calibrated ensemble.</li><li>No conclusion exists for 85606 or the new NERSC data.</li><li>Assimilation, diagnostic ranking, and steering remain closed.</li></ul></div></div></section>
</main><footer>Authoritative raw forecasts remain on shared Rusty storage and are identified by SHA-256. The report is regenerated from compact state metrics, physics metrics, and a fixed example-field artifact; it does not load 85606, newer NERSC data, a checkpoint, or a simulation.</footer>
</div></body></html>'''


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve(strict=True)
    state_path = args.state_metrics.resolve(strict=True)
    per_target_path = args.per_target.resolve(strict=True)
    physics_path = args.physics_metrics.resolve(strict=True)
    example_path = args.example_fields.resolve(strict=True)
    state = strict_json(state_path)
    physics = strict_json(physics_path)
    validate_inputs(
        state,
        physics,
        state_path=state_path,
        per_target_path=per_target_path,
        physics_path=physics_path,
        example_path=example_path,
    )
    configure_style()
    figure_dir = repo / "paper0/figures/post_ecrd_old_85604_bounded_rollout"
    plot_protocol(figure_dir)
    plot_historical_ladder(repo, figure_dir)
    plot_training(repo, figure_dir)
    plot_state_skill(state, figure_dir)
    plot_error_growth(state, figure_dir)
    plot_spectra(physics, figure_dir)
    plot_cross_field(physics, figure_dir)
    plot_transport(physics, figure_dir)
    plot_transport_traces(physics, figure_dir)
    plot_example_fields(example_path, figure_dir)

    result_dir = repo / "paper0/results"
    write_state_csv(state, result_dir / "post_ecrd_old_85604_bounded_rollout_state_6937051.csv")
    write_spectral_csv(physics, result_dir / "post_ecrd_old_85604_bounded_rollout_spectra_6937203.csv")
    write_cross_csv(physics, result_dir / "post_ecrd_old_85604_bounded_rollout_cross_field_6937203.csv")
    write_transport_csv(physics, result_dir / "post_ecrd_old_85604_bounded_rollout_transport_6937203.csv")
    write_summary_json(state, physics, result_dir / "post_ecrd_old_85604_bounded_rollout_summary_6937203.json")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_html(), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
