#!/usr/bin/env python3
"""Render the frozen Phase 2 O1 codec-oracle figures from stored metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FIELDS = ("Ne", "Te", "Ti", "phi", "Vi")
SPECTRAL_FIELDS = ("Ne", "Te", "Ti", "phi")
CODECS = ("f8", "z44")
COLORS = {"f8": "#0072B2", "z44": "#D55E00"}
MARKERS = {"f8": "o", "z44": "s"}
DISPLAY = {"Ne": r"$N_e$", "Te": r"$T_e$", "Ti": r"$T_i$", "phi": r"$\phi$", "Vi": r"$V_i$"}
BANDS = ("low_nonaxisymmetric", "coherent_study", "upper_study")
BAND_LABELS = {
    "low_nonaxisymmetric": "$n=5$–$15$",
    "coherent_study": "$n=20$–$25$",
    "upper_study": "$n=30$–$35$",
}
PAIR_LABELS = {
    "Ne-phi": r"$N_e$–$\phi$",
    "Te-phi": r"$T_e$–$\phi$",
    "Ti-phi": r"$T_i$–$\phi$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("result_type") != "phase2_o1_codec_reconstruction_oracle_compact":
        raise ValueError("input is not the compact Phase 2 O1 codec result")
    if result["scope"].get("shot_85606_accessed") is not False:
        raise ValueError("result does not certify exclusion of 85606")
    if result["scope"].get("zperiod") != 5:
        raise ValueError("result does not use frozen zperiod=5")
    return result


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
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
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str, result_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": stem.replace("-", " ").title(),
        "Source": str(result_path),
        "Description": "Phase 2 O1; run 85604 only; 85606 not accessed.",
    }
    fig.savefig(output_dir / f"{stem}.svg", metadata=metadata)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, metadata=metadata)
    plt.close(fig)


def plot_field_reconstruction(
    result: dict[str, Any], output_dir: Path, result_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), constrained_layout=True)
    x = np.arange(len(FIELDS), dtype=float)
    width = 0.36
    gate = 0.10
    for offset, codec in zip((-width / 2, width / 2), CODECS):
        values = np.asarray(
            [
                result["codec_results"][codec]["field_metrics_legacy_standardized"][field][
                    "rmse"
                ]
                for field in FIELDS
            ]
        )
        bars = axes[0].bar(
            x + offset,
            values / gate,
            width,
            label=codec,
            color=COLORS[codec],
            alpha=0.88,
        )
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1.2)
    axes[0].text(4.52, 1.0, " frozen gate", va="center", fontsize=8)
    axes[0].set_xticks(x, [DISPLAY[field] for field in FIELDS])
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Standardized RMSE / 0.10 gate")
    axes[0].set_title("Every field passes the pixel-error gate")
    axes[0].legend(title="Codec")

    for codec in CODECS:
        blocks = result["codec_results"][codec]["temporal_blocks"]
        midpoints = np.asarray(
            [(block["start_inclusive"] + block["stop_exclusive"] - 1) / 2 for block in blocks]
        )
        values = np.asarray(
            [block["aggregate_five_field_rmse_legacy_standardized"] for block in blocks]
        )
        axes[1].plot(
            midpoints,
            values,
            color=COLORS[codec],
            marker=MARKERS[codec],
            linewidth=2.0,
            markersize=5,
            label=codec,
        )
        axes[1].annotate(
            f"{codec} {values[-1]:.3f}",
            (midpoints[-1], values[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    axes[1].axvline(500, color="#555555", linestyle="--", linewidth=1.1)
    axes[1].text(
        494,
        axes[1].get_ylim()[1] * 0.98,
        "legacy storage boundary\nframe 500",
        ha="right",
        va="top",
        fontsize=8,
    )
    axes[1].set_xlabel("85604 frame (block midpoint; 78 frames per block)")
    axes[1].set_ylabel("Five-field standardized RMSE")
    axes[1].set_title("Error increases in the late trajectory blocks")
    axes[1].legend(title="Codec")

    fig.suptitle(
        "O1 isolates representation error: full 85604 codec reconstruction",
        fontsize=14,
        fontweight="normal",
    )
    save(fig, output_dir, "codec-field-reconstruction", result_path)


def plot_spectral_transfer(
    result: dict[str, Any], output_dir: Path, result_path: Path
) -> None:
    fig, axes = plt.subplots(
        len(SPECTRAL_FIELDS),
        3,
        figsize=(16, 12.5),
        sharex=True,
        constrained_layout=True,
    )
    n = np.asarray(
        result["codec_results"]["f8"]["mode_curves_k0_to_k16"]["full_torus_n"],
        dtype=float,
    )[1:]
    for row, field in enumerate(SPECTRAL_FIELDS):
        truth = np.asarray(
            result["codec_results"]["f8"]["mode_curves_k0_to_k16"]["fields"][field][
                "truth_power_fraction_of_nonaxisymmetric"
            ],
            dtype=float,
        )[1:]
        axes[row, 0].plot(n, 100 * truth, color="#4D4D4D", marker=".", linewidth=1.7)
        axes[row, 0].fill_between(n, 100 * truth, 1e-4, color="#4D4D4D", alpha=0.10)
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_ylim(1e-3, max(80.0, float(100 * truth.max() * 1.5)))
        axes[row, 0].set_ylabel(f"{DISPLAY[field]}\ntruth power (%)")

        axes[row, 1].axhspan(0.8, 1.25, color="#777777", alpha=0.08)
        axes[row, 1].axhline(0.8, color="#555555", linestyle="--", linewidth=0.9)
        axes[row, 2].axhspan(0.9, 1.0, color="#777777", alpha=0.08)
        axes[row, 2].axhline(0.9, color="#555555", linestyle="--", linewidth=0.9)

        for codec in CODECS:
            curves = result["codec_results"][codec]["mode_curves_k0_to_k16"]["fields"][field]
            ratio = np.asarray(curves["reconstruction_power_ratio"], dtype=float)[1:]
            coherence = np.asarray(
                curves["truth_to_reconstruction_coherence"], dtype=float
            )[1:]
            axes[row, 1].plot(
                n,
                ratio,
                color=COLORS[codec],
                marker=MARKERS[codec],
                markersize=3.5,
                linewidth=1.7,
                label=codec,
            )
            axes[row, 2].plot(
                n,
                coherence,
                color=COLORS[codec],
                marker=MARKERS[codec],
                markersize=3.5,
                linewidth=1.7,
                label=codec,
            )

        axes[row, 1].set_ylim(0.25, 1.12)
        axes[row, 2].set_ylim(0.35, 1.015)
        axes[row, 1].set_ylabel("Reconstructed / truth power")
        axes[row, 2].set_ylabel("Truth–reconstruction coherence")
        for column in range(3):
            axes[row, column].set_xticks([5, 15, 25, 35, 50, 65, 80])
            axes[row, column].axvline(17.5, color="#999999", linewidth=0.6)
            axes[row, column].axvline(27.5, color="#999999", linewidth=0.6)
            axes[row, column].axvline(37.5, color="#999999", linewidth=0.6)

    axes[0, 0].set_title("Truth mode power")
    axes[0, 1].set_title("Amplitude transfer (gate: 0.80–1.25)")
    axes[0, 2].set_title("Realization transfer (gate: ≥ 0.90)")
    for column in range(3):
        axes[-1, column].set_xlabel("Full-torus toroidal mode $n=5k$")
    axes[0, 2].legend(title="Codec", loc="lower left")
    axes[0, 0].text(10, 65, "$n=5$–$15$", ha="center", fontsize=8)
    axes[0, 0].text(22.5, 65, "$20$–$25$", ha="center", fontsize=8)
    axes[0, 0].text(32.5, 65, "$30$–$35$", ha="center", fontsize=8)
    axes[0, 0].text(58.5, 65, "$40$–$80$", ha="center", fontsize=8)
    axes[1, 2].annotate(
        r"$T_e$, $n=30$–$35$ is the narrow f8 gate failure",
        xy=(35, 0.872),
        xytext=(43, 0.67),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=8,
    )
    fig.suptitle(
        "O1 toroidal spectral transfer (stored wedge index k maps to full-torus n=5k)",
        fontsize=14,
        fontweight="normal",
    )
    save(fig, output_dir, "codec-spectral-transfer", result_path)


def plot_gate_robustness(
    result: dict[str, Any], output_dir: Path, result_path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    blocks = np.arange(8)
    for panel, field in enumerate(("Te", "Ti")):
        for codec in CODECS:
            values = [
                block["field_band_summaries"][field]["upper_study"][
                    "truth_power_weighted_transfer_coherence"
                ]
                for block in result["codec_results"][codec]["temporal_blocks"]
            ]
            axes[panel].plot(
                blocks,
                values,
                color=COLORS[codec],
                marker=MARKERS[codec],
                linewidth=2,
                label=codec,
            )
        axes[panel].axhline(0.9, color="#555555", linestyle="--", linewidth=1.1)
        axes[panel].text(7.1, 0.9, "gate", va="center", fontsize=8)
        axes[panel].set_ylim(0.78, 0.94)
        axes[panel].set_title(f"{DISPLAY[field]} transfer coherence, $n=30$–$35$")
        axes[panel].set_xlabel("Fixed 78-frame block")
        axes[panel].set_ylabel("Weighted coherence")
        axes[panel].set_xticks(blocks)

    for codec in CODECS:
        values = [
            block["cross_field_band_summaries"]["Ne-phi"]["upper_study"][
                "truth_cross_amplitude_weighted_absolute_coherence_change"
            ]
            for block in result["codec_results"][codec]["temporal_blocks"]
        ]
        axes[2].plot(
            blocks,
            values,
            color=COLORS[codec],
            marker=MARKERS[codec],
            linewidth=2,
            label=codec,
        )
    axes[2].axhline(0.1, color="#555555", linestyle="--", linewidth=1.1)
    axes[2].text(7.1, 0.1, "gate", va="center", fontsize=8)
    axes[2].set_ylim(0.04, 0.12)
    axes[2].set_title(r"$N_e$–$\phi$ coherence change, $n=30$–$35$")
    axes[2].set_xlabel("Fixed 78-frame block")
    axes[2].set_ylabel("Absolute coherence change")
    axes[2].set_xticks(blocks)
    axes[0].legend(title="Codec")
    fig.suptitle(
        "Frozen O1 gates require at least 7 of 8 temporal blocks",
        fontsize=14,
        fontweight="normal",
    )
    save(fig, output_dir, "codec-gate-robustness", result_path)


def plot_cross_field(
    result: dict[str, Any], output_dir: Path, result_path: Path
) -> None:
    pairs = ("Ne-phi", "Te-phi", "Ti-phi")
    labels = [f"{PAIR_LABELS[pair]}\n{BAND_LABELS[band]}" for pair in pairs for band in BANDS]
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.7), constrained_layout=True)
    for offset, codec in zip((-width / 2, width / 2), CODECS):
        phase = []
        change = []
        for pair in pairs:
            for band in BANDS:
                metrics = result["codec_results"][codec]["cross_field_band_summaries"][pair][band]
                phase.append(
                    metrics[
                        "truth_cross_amplitude_weighted_absolute_phase_error_degrees"
                    ]
                )
                change.append(
                    metrics[
                        "truth_cross_amplitude_weighted_absolute_coherence_change"
                    ]
                )
        axes[0].bar(
            x + offset,
            np.asarray(phase) / 15.0,
            width,
            color=COLORS[codec],
            alpha=0.88,
            label=codec,
        )
        axes[1].bar(
            x + offset,
            np.asarray(change) / 0.10,
            width,
            color=COLORS[codec],
            alpha=0.88,
            label=codec,
        )
    for axis in axes:
        axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.1)
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.set_ylim(0, 1.12)
        axis.set_ylabel("Metric / frozen limit")
    axes[0].set_title("Cross-phase error / 15° limit")
    axes[1].set_title("Absolute coherence change / 0.10 limit")
    axes[0].legend(title="Codec")
    fig.suptitle(
        "Cross-field structure across every primary pair and material gate band",
        fontsize=14,
        fontweight="normal",
    )
    save(fig, output_dir, "codec-cross-field", result_path)


def main() -> None:
    args = parse_args()
    result_path = Path(args.result).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    result = load_result(result_path)
    configure_style()
    plot_field_reconstruction(result, output_dir, result_path)
    plot_spectral_transfer(result, output_dir, result_path)
    plot_gate_robustness(result, output_dir, result_path)
    plot_cross_field(result, output_dir, result_path)
    print(f"wrote four O1 figure pairs to {output_dir}")


if __name__ == "__main__":
    main()
