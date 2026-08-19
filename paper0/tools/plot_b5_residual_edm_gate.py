#!/usr/bin/env python3
"""Render fully labeled B5 one-seed localization figures from compact metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
BANDS = ("k1_3", "k4_5", "k6_7")
CROSS_PAIRS = ("Ne-phi", "Pe-phi", "Pi-phi")
QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)
MODEL_ORDER = (
    "H1 deterministic",
    "B3 functional noise",
    "B4 PDE-Refiner",
    "B5 joint residual EDM",
)
PROBABILISTIC_MODELS = MODEL_ORDER[1:]
COLORS = {
    "H1 deterministic": "#6C757D",
    "B3 functional noise": "#0072B2",
    "B4 PDE-Refiner": "#D55E00",
    "B5 joint residual EDM": "#009E73",
    "field": "#0072B2",
    "spectral": "#CC79A7",
    "transport": "#D55E00",
}
FIELD_LABELS = {
    "Ne": r"$N_e$",
    "Pe": r"$P_e$",
    "Pi": r"$P_i$",
    "phi": r"$\phi$",
    "Vi": r"$V_i$",
}
BAND_LABELS = {
    "k1_3": "$k=1$–$3$\n($n=5$–$15$)",
    "k4_5": "$k=4$–$5$\n($n=20$–$25$)",
    "k6_7": "$k=6$–$7$\n($n=30$–$35$)",
}
PAIR_LABELS = {
    "Ne-phi": r"$N_e$–$\phi$",
    "Pe-phi": r"$P_e$–$\phi$",
    "Pi-phi": r"$P_i$–$\phi$",
}
QUANTITY_LABELS = {
    "particle": "particle",
    "electron_internal_energy": "electron energy",
    "ion_internal_energy": "ion energy",
    "total_internal_energy": "total energy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("scope") != "phase3_B5_joint_residual_EDM_one_seed_localization_85604":
        raise ValueError("input is not the compact B5 localization result")
    if result.get("held_out_85606_read") is not False:
        raise ValueError("result does not certify exclusion of held-out 85606")
    if result.get("thresholds_changed_by_this_tool") is not False:
        raise ValueError("result reports a threshold change")
    if result["task"].get("zperiod") != 5 or result["task"].get(
        "mode_mapping"
    ) != "n=5k":
        raise ValueError("result does not use the frozen toroidal-mode mapping")
    return result


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
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "svg.hashsalt": "tcv-diagnostics-paper0-b5",
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str, result_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": stem.replace("-", " ").title(),
        "Source": str(result_path),
        "Description": (
            "B5 one-step failure localization on development run 85604 only; "
            "held-out 85606 was not accessed."
        ),
        "Date": "2026-08-19",
    }
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(svg_path, metadata=metadata)
    # Matplotlib writes insignificant spaces after SVG path commands.  Strip
    # them so git's whitespace check is quiet and regenerated figures remain
    # byte-for-byte reproducible.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(output_dir / f"{stem}.png", dpi=220, metadata=metadata)
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars: Sequence[Any], fmt: str = ".3f") -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            format(height, fmt),
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=0,
        )


def plot_training(result: Mapping[str, Any], output_dir: Path, path: Path) -> None:
    records = result["training"]["epochs"]
    epochs = np.asarray([item["completed_epoch"] for item in records], dtype=float)
    train_edm = np.asarray([item["train_EDM_loss"] for item in records], dtype=float)
    train_mse = np.asarray(
        [item["train_unweighted_MSE"] for item in records], dtype=float
    )
    candidates = [item for item in records if item["validation_EDM_loss"] is not None]
    candidate_epochs = np.asarray(
        [item["completed_epoch"] for item in candidates], dtype=float
    )
    validation_edm = np.asarray(
        [item["validation_EDM_loss"] for item in candidates], dtype=float
    )
    validation_mse = np.asarray(
        [item["validation_unweighted_MSE"] for item in candidates], dtype=float
    )
    selected_epoch = result["training"]["selected_completed_epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3), constrained_layout=True)
    for ax, train, validation, ylabel, title in (
        (
            axes[0],
            train_edm,
            validation_edm,
            "EDM denoising loss (equal element/channel)",
            "Frozen selection score",
        ),
        (
            axes[1],
            train_mse,
            validation_mse,
            "Unweighted residual-space MSE",
            "Companion optimization diagnostic",
        ),
    ):
        ax.plot(
            epochs,
            train,
            color="#6C757D",
            linewidth=1.6,
            alpha=0.8,
            label="training epoch mean",
        )
        ax.plot(
            candidate_epochs,
            validation,
            color=COLORS["B5 joint residual EDM"],
            marker="o",
            markersize=4,
            linewidth=2.0,
            label="fixed-seed validation (every 5 epochs)",
        )
        ax.axvline(
            selected_epoch,
            color="#000000",
            linestyle="--",
            linewidth=1.0,
            label="selected epoch 100",
        )
        ax.set_xlabel("Completed epoch (430 training targets per epoch)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right")
    axes[0].annotate(
        f"selected validation EDM loss\n{validation_edm[-1]:.6f}",
        (candidate_epochs[-1], validation_edm[-1]),
        xytext=(-112, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=8.5,
    )
    fig.suptitle(
        "B5 joint field-space residual EDM training — seed 1701, run 85604 only",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.012,
        "Checkpoint rule: earliest lowest fixed-seed validation EDM loss after the complete frozen 100-epoch budget. "
        "No field, spectral, transport, or 85606 metric entered selection.",
        ha="center",
        fontsize=8.5,
    )
    save(fig, output_dir, "b5-training-curves", path)


def plot_model_comparison(
    result: Mapping[str, Any], output_dir: Path, path: Path
) -> None:
    models = result["model_comparison"]["models"]
    x = np.arange(len(MODEL_ORDER), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), constrained_layout=True)

    width = 0.36
    mae = np.asarray([models[name]["mae_relative_to_H1"] for name in MODEL_ORDER])
    rmse = np.asarray([models[name]["rmse_relative_to_H1"] for name in MODEL_ORDER])
    bars_mae = axes[0, 0].bar(x - width / 2, mae, width, color="#56B4E9", label="MAE / H1 MAE")
    bars_rmse = axes[0, 0].bar(x + width / 2, rmse, width, color="#E69F00", label="RMSE / H1 RMSE")
    axes[0, 0].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0, 0].set_ylim(0.84, 1.04)
    axes[0, 0].set_ylabel("Error relative to deterministic H1")
    axes[0, 0].set_title("A. One-step ensemble-mean field error (lower is better)")
    axes[0, 0].set_xticks(x, [name.replace(" ", "\n", 1) for name in MODEL_ORDER])
    axes[0, 0].legend()
    annotate_bars(axes[0, 0], bars_mae)
    annotate_bars(axes[0, 0], bars_rmse)

    crps = np.asarray(
        [models[name]["fair_crps_relative_to_H1_MAE"] for name in MODEL_ORDER]
    )
    bars = axes[0, 1].bar(
        x, crps, color=[COLORS[name] for name in MODEL_ORDER], width=0.64
    )
    axes[0, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0, 1].set_ylim(0.55, 1.06)
    axes[0, 1].set_ylabel("Fair CRPS / deterministic H1 MAE")
    axes[0, 1].set_title("B. Marginal proper score (lower is better)")
    axes[0, 1].set_xticks(x, [name.replace(" ", "\n", 1) for name in MODEL_ORDER])
    annotate_bars(axes[0, 1], bars)

    px = np.arange(len(PROBABILISTIC_MODELS), dtype=float)
    spread = np.asarray([models[name]["spread_skill"] for name in PROBABILISTIC_MODELS])
    bars = axes[1, 0].bar(
        px,
        spread,
        color=[COLORS[name] for name in PROBABILISTIC_MODELS],
        width=0.62,
    )
    axes[1, 0].axhspan(0.80, 1.25, color="#009E73", alpha=0.12, label="frozen primary target 0.80–1.25")
    axes[1, 0].axhline(1.0, color="#333333", linewidth=1.0)
    axes[1, 0].set_ylim(0, 1.32)
    axes[1, 0].set_ylabel("Corrected RMS spread / ensemble-mean RMSE")
    axes[1, 0].set_title("C. Aggregate pixel spread–skill")
    axes[1, 0].set_xticks(px, [name.replace(" ", "\n", 1) for name in PROBABILISTIC_MODELS])
    axes[1, 0].legend(loc="upper left")
    annotate_bars(axes[1, 0], bars)

    power = np.asarray(
        [models[name]["power_checks_passing"] for name in PROBABILISTIC_MODELS]
    )
    coherence = np.asarray(
        [
            models[name]["realization_coherence_checks_passing"]
            for name in PROBABILISTIC_MODELS
        ]
    )
    bars_power = axes[1, 1].bar(
        px - width / 2,
        power,
        width,
        color="#009E73",
        label="expected-member power",
    )
    bars_coherence = axes[1, 1].bar(
        px + width / 2,
        coherence,
        width,
        color="#CC79A7",
        label="ensemble-mean realization coherence",
    )
    axes[1, 1].axhline(15, color="#333333", linestyle="--", linewidth=1.0)
    axes[1, 1].set_ylim(0, 16.5)
    axes[1, 1].set_ylabel("Passing material field-band checks (of 15)")
    axes[1, 1].set_title("D. Spectral amplitude is not realization fidelity")
    axes[1, 1].set_xticks(px, [name.replace(" ", "\n", 1) for name in PROBABILISTIC_MODELS])
    axes[1, 1].legend(loc="lower left")
    annotate_bars(axes[1, 1], bars_power, ".0f")
    annotate_bars(axes[1, 1], bars_coherence, ".0f")

    fig.suptitle(
        "B3–B5 one-step comparison: B5 improves mean, marginal score, spread, and power—but not realization fidelity",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.012,
        "All values use the same 126 chronological 85604 validation targets and M32 for probabilistic models. "
        "The architectures and losses differ, so this is a descriptive comparison, not a single-factor ablation.",
        ha="center",
        fontsize=8.5,
    )
    save(fig, output_dir, "b5-model-comparison", path)


def annotated_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    passes: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Any = None,
    decimals: int = 2,
) -> None:
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, aspect="auto")
    ax.grid(False)
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = f"{value:.{decimals}f}\n{'PASS' if passes[row, column] else 'FAIL'}"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "#009E73" if passes[row, column] else "#D55E00",
                    "alpha": 0.82,
                    "linewidth": 1.0,
                },
            )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label(colorbar_label)


def plot_field_spectra(
    result: Mapping[str, Any], output_dir: Path, path: Path
) -> None:
    source = result["spectral_and_cross_field"]["field_bands"]

    def matrix(key: str) -> np.ndarray:
        return np.asarray(
            [[source[field][band][key] for band in BANDS] for field in FIELDS],
            dtype=float,
        )

    power = matrix("member_expected_power_ratio")
    power_pass = matrix("power_ratio_passes").astype(bool)
    coherence = matrix("ensemble_mean_realization_coherence")
    coherence_pass = matrix("realization_coherence_passes").astype(bool)
    spread = matrix("mode_power_spread_skill")
    spread_pass = matrix("mode_power_spread_skill_passes").astype(bool)
    coverage = matrix("mode_power_I31_coverage")
    coverage_pass = matrix("mode_power_I31_coverage_passes").astype(bool)

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.8), constrained_layout=True)
    rows = [FIELD_LABELS[field] for field in FIELDS]
    columns = [BAND_LABELS[band] for band in BANDS]
    annotated_heatmap(
        axes[0, 0],
        power,
        power_pass,
        rows,
        columns,
        title="A. Expected member power ratio (gate 0.75–1.30)",
        colorbar_label="member-expected power / truth power",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.05),
    )
    annotated_heatmap(
        axes[0, 1],
        coherence,
        coherence_pass,
        rows,
        columns,
        title="B. Correct next-frame realization (gate ≥ 0.80)",
        colorbar_label="ensemble-mean field coherence with truth",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    annotated_heatmap(
        axes[1, 0],
        spread,
        spread_pass,
        rows,
        columns,
        title="C. Mode-power spread–skill (gate 0.67–1.50)",
        colorbar_label="ensemble mode-power spread / mode-power error",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    annotated_heatmap(
        axes[1, 1],
        coverage,
        coverage_pass,
        rows,
        columns,
        title="D. M32 widest interval coverage (gate 0.75–0.995)",
        colorbar_label="empirical coverage of I31 order-statistic interval",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    fig.suptitle(
        "B5 toroidal field bands: restored amplitude does not imply correct or calibrated structure",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.012,
        "Stored wedge index k maps to full-torus mode number n=5k because zperiod=5. "
        "Every cell is an independently frozen gate check; PASS/FAIL is not inferred from color.",
        ha="center",
        fontsize=8.5,
    )
    save(fig, output_dir, "b5-field-spectral-localization", path)


def plot_cross_field(
    result: Mapping[str, Any], output_dir: Path, path: Path
) -> None:
    source = result["spectral_and_cross_field"]["cross_field_bands"]

    def matrix(key: str) -> np.ndarray:
        return np.asarray(
            [[source[pair][band][key] for band in BANDS] for pair in CROSS_PAIRS],
            dtype=float,
        )

    phase = matrix("absolute_cross_phase_error_degrees")
    phase_pass = matrix("cross_phase_passes").astype(bool)
    coherence = matrix("absolute_cross_coherence_change")
    coherence_pass = matrix("cross_coherence_change_passes").astype(bool)
    projection_spread = np.asarray(
        [
            [
                np.mean(
                    [
                        source[pair][band]["projections"][projection]["spread_skill"]
                        for projection in ("real", "imaginary")
                    ]
                )
                for band in BANDS
            ]
            for pair in CROSS_PAIRS
        ],
        dtype=float,
    )
    projection_spread_pass = np.asarray(
        [
            [
                all(
                    source[pair][band]["projections"][projection][
                        "spread_skill_passes"
                    ]
                    for projection in ("real", "imaginary")
                )
                for band in BANDS
            ]
            for pair in CROSS_PAIRS
        ],
        dtype=bool,
    )
    projection_coverage = np.asarray(
        [
            [
                np.mean(
                    [
                        source[pair][band]["projections"][projection]["I31_coverage"]
                        for projection in ("real", "imaginary")
                    ]
                )
                for band in BANDS
            ]
            for pair in CROSS_PAIRS
        ],
        dtype=float,
    )
    projection_coverage_pass = np.asarray(
        [
            [
                all(
                    source[pair][band]["projections"][projection][
                        "I31_coverage_passes"
                    ]
                    for projection in ("real", "imaginary")
                )
                for band in BANDS
            ]
            for pair in CROSS_PAIRS
        ],
        dtype=bool,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.4), constrained_layout=True)
    rows = [PAIR_LABELS[pair] for pair in CROSS_PAIRS]
    columns = [BAND_LABELS[band] for band in BANDS]
    annotated_heatmap(
        axes[0, 0],
        phase,
        phase_pass,
        rows,
        columns,
        title="A. Member-expected cross-phase error (gate ≤ 20°)",
        colorbar_label="absolute phase error (degrees)",
        cmap="magma_r",
        vmin=0,
        vmax=20,
    )
    annotated_heatmap(
        axes[0, 1],
        coherence,
        coherence_pass,
        rows,
        columns,
        title="B. Member-expected cross-coherence change (gate ≤ 0.15)",
        colorbar_label="absolute coherence change",
        cmap="magma_r",
        vmin=0,
        vmax=0.30,
    )
    annotated_heatmap(
        axes[1, 0],
        projection_spread,
        projection_spread_pass,
        rows,
        columns,
        title="C. Cross-spectrum projection spread–skill (gate 0.67–1.50)",
        colorbar_label="mean of real/imaginary spread–skill ratios",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    annotated_heatmap(
        axes[1, 1],
        projection_coverage,
        projection_coverage_pass,
        rows,
        columns,
        title="D. Cross-spectrum projection I31 coverage (gate 0.75–0.995)",
        colorbar_label="mean real/imaginary empirical coverage",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    fig.suptitle(
        "B5 cross-field structure: mean phase survives, but high-band coherence and joint uncertainty do not",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.012,
        "Panels C–D display the mean of the real and imaginary cross-spectrum projections for readability; "
        "PASS requires both separately frozen projection checks to pass. All 18 projection checks fail.",
        ha="center",
        fontsize=8.5,
    )
    save(fig, output_dir, "b5-cross-field-localization", path)


def _grouped_transport_panel(
    ax: plt.Axes,
    values: Mapping[str, Mapping[str, float | None]],
    key: str,
    models: Sequence[str],
    *,
    title: str,
    ylabel: str,
    gate_line: float | None,
    gate_label: str | None,
    ylim: tuple[float, float],
) -> None:
    x = np.arange(len(QUANTITIES), dtype=float)
    width = 0.18 if len(models) == 4 else 0.23
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * width
    for offset, model in zip(offsets, models):
        y = np.asarray([values[q][model][key] for q in QUANTITIES], dtype=float)
        bars = ax.bar(
            x + offset,
            y,
            width,
            color=COLORS[model],
            label=model,
            alpha=0.9,
        )
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6.7,
                rotation=90,
            )
    if gate_line is not None:
        ax.axhline(
            gate_line,
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            label=gate_label,
        )
    ax.set_xticks(x, [QUANTITY_LABELS[q] for q in QUANTITIES], rotation=12)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=7.2, ncol=2)


def plot_transport(result: Mapping[str, Any], output_dir: Path, path: Path) -> None:
    values = result["model_comparison"]["transport_by_quantity"]
    fig, axes = plt.subplots(2, 2, figsize=(16.2, 10.0), constrained_layout=True)
    _grouped_transport_panel(
        axes[0, 0],
        values,
        "strict_relative_l2",
        MODEL_ORDER,
        title="A. Local geometry-aware face transport",
        ylabel="Strict facewise relative L2 error",
        gate_line=0.40,
        gate_label="frozen maximum 0.40",
        ylim=(0, 0.82),
    )
    _grouped_transport_panel(
        axes[0, 1],
        values,
        "separatrix_relative_l2",
        MODEL_ORDER,
        title="B. Integrated separatrix wedge mean",
        ylabel="Separatrix relative L2 error",
        gate_line=0.30,
        gate_label="frozen maximum 0.30",
        ylim=(0, 0.36),
    )
    _grouped_transport_panel(
        axes[1, 0],
        values,
        "separatrix_fair_crps_relative_to_H1_error",
        MODEL_ORDER,
        title="C. Separatrix marginal proper score",
        ylabel="Fair CRPS / deterministic H1 absolute error",
        gate_line=1.0,
        gate_label="better than H1 below 1",
        ylim=(0, 1.20),
    )
    _grouped_transport_panel(
        axes[1, 1],
        values,
        "separatrix_spread_skill",
        PROBABILISTIC_MODELS,
        title="D. Separatrix ensemble calibration",
        ylabel="Separatrix spread / ensemble-mean error",
        gate_line=0.67,
        gate_label="frozen lower bound 0.67 (upper 1.50)",
        ylim=(0, 0.75),
    )
    fig.suptitle(
        "B5 transport: useful integrated means and proper scores coexist with failed local structure and calibration",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.014,
        "Particle and heat-flux operators are applied independently to every ensemble member before reduction. "
        "Transport of ensemble-mean fields is never used as the probabilistic diagnostic.",
        ha="center",
        fontsize=8.5,
    )
    save(fig, output_dir, "b5-transport-localization", path)


def plot_chronology(result: Mapping[str, Any], output_dir: Path, path: Path) -> None:
    blocks = result["chronology"]["blocks"]
    x = np.arange(len(blocks), dtype=float)
    labels = [f"{b['target_frames'][0]}–{b['target_frames'][1] - 1}" for b in blocks]
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.6), constrained_layout=True)

    for key, label, color, marker in (
        ("H1_rmse", "H1 deterministic", COLORS["H1 deterministic"], "o"),
        ("B4_rmse", "B4 PDE-Refiner", COLORS["B4 PDE-Refiner"], "s"),
        ("B5_rmse", "B5 joint residual EDM", COLORS["B5 joint residual EDM"], "^"),
    ):
        axes[0, 0].plot(
            x,
            [block[key] for block in blocks],
            color=color,
            marker=marker,
            linewidth=2.0,
            label=label,
        )
    axes[0, 0].set_title("A. One-step field RMSE by chronological block")
    axes[0, 0].set_ylabel("Equal-channel standardized RMSE")
    axes[0, 0].legend()

    for key, label, color, marker in (
        ("B4_fair_crps", "B4 PDE-Refiner", COLORS["B4 PDE-Refiner"], "s"),
        ("B5_fair_crps", "B5 joint residual EDM", COLORS["B5 joint residual EDM"], "^"),
    ):
        ratios = [block[key] / block["H1_mae"] for block in blocks]
        axes[0, 1].plot(
            x,
            ratios,
            color=color,
            marker=marker,
            linewidth=2.0,
            label=label,
        )
    axes[0, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("B. Marginal score relative to H1 in each block")
    axes[0, 1].set_ylabel("Fair CRPS / block-matched H1 MAE")
    axes[0, 1].legend()

    spread = [block["B5_spread_skill"] for block in blocks]
    axes[1, 0].plot(
        x,
        spread,
        color=COLORS["B5 joint residual EDM"],
        marker="o",
        linewidth=2.2,
    )
    axes[1, 0].axhspan(0.80, 1.25, color="#009E73", alpha=0.12)
    axes[1, 0].axhline(1.0, color="#333333", linewidth=1.0)
    for index, value in enumerate(spread):
        axes[1, 0].annotate(f"{value:.2f}", (index, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    axes[1, 0].set_ylim(0.65, 1.28)
    axes[1, 0].set_title("C. B5 aggregate pixel spread–skill by block")
    axes[1, 0].set_ylabel("Corrected spread / ensemble-mean RMSE")

    width = 0.24
    for offset, family in zip((-width, 0.0, width), ("field", "spectral", "transport")):
        fractions = [
            100.0 * block["failed_check_counts"][family]["fraction_failed"]
            for block in blocks
        ]
        bars = axes[1, 1].bar(
            x + offset,
            fractions,
            width,
            color=COLORS[family],
            label=family,
        )
        for bar, block in zip(bars, blocks):
            failed = block["failed_check_counts"][family]["failed"]
            total = block["failed_check_counts"][family]["total"]
            axes[1, 1].annotate(
                f"{failed}/{total}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
                rotation=90,
            )
    axes[1, 1].set_title("D. Frozen checks failed in every chronological block")
    axes[1, 1].set_ylabel("Failed checks within family (%)")
    axes[1, 1].set_ylim(0, 80)
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=20)
        ax.set_xlabel("85604 validation target-frame block (inclusive labels)")
    fig.suptitle(
        "B5 chronology: mean and marginal gains are stable, but no block passes the complete physics gate",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.012,
        "Blocks are contiguous and ordered; they are not treated as independent physical shots. "
        "The frozen rule required at least five of six passing blocks, while zero pass all families.",
        ha="center",
        fontsize=8.5,
    )
    save(fig, output_dir, "b5-chronological-localization", path)


def main() -> None:
    args = parse_args()
    result = load_result(args.result)
    configure_style()
    plot_training(result, args.output_dir, args.result)
    plot_model_comparison(result, args.output_dir, args.result)
    plot_field_spectra(result, args.output_dir, args.result)
    plot_cross_field(result, args.output_dir, args.result)
    plot_transport(result, args.output_dir, args.result)
    plot_chronology(result, args.output_dir, args.result)
    generated = sorted(str(path) for path in args.output_dir.glob("b5-*.*"))
    print(
        json.dumps(
            {
                "generated": generated,
                "count": len(generated),
                "held_out_85606_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
